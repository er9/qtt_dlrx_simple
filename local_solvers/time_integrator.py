"""Time-integrator layer of the local-solver stack.

Defines the :class:`TimeIntegrator` base class and DMRG-based integrators
(:class:`TDDMRG`, :class:`TDVP_DMRG`, :class:`TDVP_Krylov_DMRG`, and
variants) along with the :class:`TimeIntegMethod` enum (Euler, RK2, RK4,
...). These integrators extend the local Evaluators to advance a
matrix-product state in time by repeatedly forming and solving the local
projected problem for the right-hand side F(ket) and source terms.
"""
import pdb
from abc import ABC

import helper_quimb
from setup_.defaults import *
import local_solvers.helper_tn as helper_tn

from local_solvers.defaults import *
from local_solvers.mps_classes import MPS
from local_solvers.local_evaluator import LocalEvaluator
from local_solvers.local_dmrg_eval import DMRGEvaluator
from local_solvers.local_cross_eval import CrossEvaluator
from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross
import local_solvers.tensor_callables as tc
import helper_TE
import local_solvers.helper_dmrg_loc as helper_dmrg

if TYPE_CHECKING:
    from grid import Grid
    from setup_.configs import DerivativeConfiguration


class TimeIntegMethod(IntEnum):
    Euler = 1
    RK2 = 22
    SSPRK3 = 3
    RK4 = 4
    LW = 2
    SL = 5
    CN = 223
    CN6 = 226
    BE = 221
    EXACT = 0



class TimeIntegrator(LocalEvaluator, ABC):

    def __init__(self,
                 ket_state: Union['MPS', 'qtn.MatrixProductState'],
                 linear_operators: Sequence['qtn.MatrixProductOperator'],
                 sources: Sequence[Union['MPS', 'qtn.MatrixProductState']] = None,
                 nonlinear_terms: Sequence[Union['Term_DMRG', 'Term_Cross']] = None,
                 # constraints: Sequence[Sequence[Union['Term_DMRG','Term_Cross']]] = None,
                 constraints: Sequence[Sequence[Union[qtn.MatrixProductState, qtn.MatrixProductOperator]]] = None,
                 constraint_vals: Sequence[Union[float, qtn.MatrixProductState, Term]] = None,
                 constraint_funcs: Sequence[Callable] = None,  ## expressions that should yield 0 / be minimized (via scipy)
                 direction: SweepDirection = SweepDirection.RIGHT,
                 max_bond: int = None, cutoff: float = None,
                 conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
                 max_tot_iter: int = DEFAULT_MAX_TOT_ITER, max_wrong_iter: int = DEFAULT_MAX_WRONG_ITER,
                 copy_obj: 'TimeIntegrator' = None,
                 # combine_terms_func: 'Callable' = None,
                 dt = 0.1, time = None,
                 te_order_target = 4, te_order_final = 4,
                 grid: 'Grid' = None, ax_deriv_configs: dict['Axis','DerivativeConfiguration'] = None,
                 time_mpo_list: dict[Any, 'MPOType']=None, upwind_mpo_list: dict[Any, 'MPOType']=None,
                 verbose=0, verbose_plot=False, local_euler_func: Callable = None,
                 ):

        # print('upwind mpo list', upwind_mpo_list)
        # exit()

        self.num_evals = 0

        if copy_obj is not None:
            super().__init__(ket_state, None, copy_obj=copy_obj)

            self.init_ket = self.terms[0].ket
            self.linear_operators = copy_obj.linear_operators
            self.sources = copy_obj.sources
            # self.nonlinear_terms = copy_obj.nonlinear_terms
            self.constraints = [c.copy() for c in copy_obj.constraints]
            self.constraint_vals = copy_obj.constraint_vals

            self.self_term = self.terms[0]
            num_lin = len(copy_obj.linear_terms)
            num_src = len(copy_obj.source_terms)
            num_nlin = len(copy_obj.nonlinear_terms)
            num_cons = len(self.constraints)
            self.linear_terms = self.terms[1: num_lin + 1]
            self.source_terms = self.terms[num_lin + 1: num_lin + 1 + num_src]
            self.nonlinear_terms = self.terms[num_lin + num_src + 1: num_lin + num_src + num_lin + 1]
            self.constraint_terms = self.terms[num_lin + num_src + num_nlin + 1:
                                               num_lin + num_src + num_nlin + num_cons + 1]
                                    # copy_obj.constraint_terms
            self.extra_terms_dict = copy_obj.extra_terms_dict

            cons_val_terms = self.terms[num_lin + num_src + num_lin + num_cons + 1:]
            iter_val_terms = iter(cons_val_terms)
            self.constraint_vals = [next(iter_val_terms) if isinstance(c,Term) else c for c in copy_obj.constraint_vals]

            self.dt = copy_obj.dt
            self.time = copy_obj.time
            self.te_order_target = copy_obj.te_order_target
            if self.te_order_target == 0:
                self.te_order_target = 223
            self.te_order_final = copy_obj.te_order_final
            self.local_euler_func = copy_obj.local_euler_func

            self.time_mpo_list = copy_obj.time_mpo_list
            self.upwind_mpo_list = copy_obj.upwind_mpo_list
            self.verbose_plot = copy_obj.verbose_plot
            self.verbose = copy_obj.verbose

            self.upwind_func = copy_obj.upwind_func
            self.upwind_deriv_func = copy_obj.upwind_deriv_func

        else:
            self.verbose = verbose

            self.init_ket = ket_state.copy()
            self.te_order_target = te_order_target if te_order_target != 0 else 223
            self.te_order_final = te_order_final
            self.local_euler_func = local_euler_func

            self.linear_operators = linear_operators
            self.sources = sources if sources is not None else []
            self.nonlinear_terms = nonlinear_terms if nonlinear_terms is not None else []
            self.constraints = constraints if constraints is not None else []
            self.constraint_vals = [*constraint_vals] if constraint_vals is not None else []
            self.time_mpo_list = time_mpo_list
            self.upwind_mpo_list = upwind_mpo_list

            self.self_term: Term = None
            self.linear_terms: list[Term] = []
            self.source_terms: list[Term] = []
            self.constraint_terms: list[Term] = []
            self.constraint_val_terms = []
            self.extra_terms_dict: dict[Any, list[Term]] = {}

            self._direction = direction

            # terms = self.initialize_terms(self.init_ket)   ## now moved into super().__init__
            ## otherwise TDVP only updates once (dt/2 instead of dt) when doing back and forth sweeping
            ## because terms are not updated with self.out

            super().__init__(ket_state, None, direction, max_bond=max_bond, cutoff=cutoff,
                             conv_tol=conv_tol, max_iter=max_iter, max_tot_iter=max_tot_iter,
                             max_wrong_iter=max_wrong_iter, copy_obj=copy_obj,
                             grid=grid, ax_deriv_configs=ax_deriv_configs
                             )  # , combine_terms_func=combine_terms_func)

            # if nonlinear_terms is not None:
            #     for t in nonlinear_terms:
            #         if t.ket is not self.init_ket:
            #             print('init2 nonlinear ket is not self.ket')
            #             exit()

            self.dt = dt
            self.time = time
            self.verbose_plot = verbose_plot
            self.upwind_func = None
            self.upwind_deriv_func = None


    def initialize_terms(self, ket_state: Union['MPS', 'qtn.MatrixProductState'], cur_orthog=None,
                         **kwargs
                         # linear_operators: Sequence[qtn.MatrixProductOperator],
                         # sources: Optional[Sequence[Union['qtn.MatrixProductState','MPS']]],
                         # nonlinear_terms: Optional[Sequence[Union['Term_DMRG','Term_Cross']]]
                         ):

        LocalTerm = self.term_class()
        compress_opts = {'max_bond': self.max_bond, 'cutoff': self.cutoff}

        term_self = LocalTerm(ket_state)
        self.self_term = term_self

        if len(self.linear_operators) > 0:
            # term_linear = [LocalTerm(ket_state, operators=[mpo for mpo in self.linear_operators],
            #                          cur_orthog=cur_orthog)]
            term_linear = [LocalTerm(ket_state, operators=[mpo], cur_orthog=cur_orthog, **compress_opts)
                           for mpo in self.linear_operators]
        else:
            term_linear = []
        self.linear_terms = term_linear

        term_sources = [LocalTerm(source, bra=ket_state, cur_orthog=cur_orthog, **compress_opts) for source in self.sources]
        # term_sources = [LocalTerm(source, bra=self.ket) for source in self.sources]
        self.source_terms = term_sources
        if self.verbose > 4:
            print('self.source terms', self.source_terms)

        nl_terms = []
        for nl in self.nonlinear_terms:
            nl_term = nl.copy(ket_state, ket_state)
            nl_term.cutoff = self.cutoff
            nl_term.max_bond = self.max_bond
            nl_terms += [nl_term]

        self.nonlinear_terms = nl_terms
        nonlinear_terms = self.nonlinear_terms

        # nonlinear_terms = self.nonlinear_terms
        # for n in self.nonlinear_terms:
        #     n.ket = ket_state
        #     n.bra = ket_state

        term_constraints = []
        for constraint in self.constraints:
            if isinstance(constraint, qtn.MatrixProductState):
                term_constraints += [LocalTerm(constraint, bra=ket_state, cur_orthog=cur_orthog, **compress_opts)]
                ## conjugate of what it should be
            else:
                tmp = [constraint] if not isinstance(constraint, (list,tuple)) else constraint
                term_constraints += [LocalTerm(ket_state, operators=tmp, cur_orthog=cur_orthog, **compress_opts)]
        self.constraint_terms = term_constraints

        constraint_val_all = []
        constraint_val_terms = []
        for cv in self.constraint_vals:
            if isinstance(cv, qtn.MatrixProductState):
                constraint_val_terms += [LocalTerm(cv, bra=ket_state, cur_orthog=cur_orthog, **compress_opts)]
                ## the conjugate of what it should be
                constraint_val_all += [constraint_val_terms[-1]]
            else:
                constraint_val_all += [cv]  # float or Term
        self.constraint_val_terms = constraint_val_all

        ###### extra terms ######
        ### time mpo list ###
        extra_terms_list = []
        if self.time_mpo_list is not None:
            for key, ops_list in self.time_mpo_list.items():
                self.extra_terms_dict[key] = [LocalTerm(ket_state, operators=[mpo for mpo in ops_list], **compress_opts)]
                extra_terms_list += self.extra_terms_dict[key]

        if self.upwind_mpo_list is not None:
            for key, ops_list in self.upwind_mpo_list.items():
                if isinstance(ops_list[0], qtn.MatrixProductOperator):
                    # self.extra_terms_dict[key] = [LocalTerm(ket_state, operators=[mpo for mpo in ops_list])]
                    self.extra_terms_dict[key] = [LocalTerm(ket_state, operators=[mpo], **compress_opts) # cur_orthog=cur_orthog)
                                                  for mpo in ops_list]
                elif isinstance(ops_list[0], qtn.MatrixProductState):
                    self.extra_terms_dict[key] = [LocalTerm(mps, bra=ket_state, **compress_opts) for mps in ops_list]
                else:
                    if self.verbose > 2:
                        print('type', type(ops_list[0]))
                    raise TypeError
                extra_terms_list += self.extra_terms_dict[key]

        return [term_self, *term_linear, *term_sources, *nonlinear_terms, *term_constraints,
                *constraint_val_terms, *extra_terms_list]  # , *extra_terms_list]


    def _set_local_solve_func(self, te_order: int):
        # te_order = 1
        # te_order = 223
        # te_order = 0
        # print('te order', te_order)
        if te_order == TimeIntegMethod.RK4:
            func = self.local_rk4
        elif te_order == TimeIntegMethod.RK2:
            func = self.local_rk2
        elif te_order == TimeIntegMethod.SSPRK3:
            func = self.local_ssprk3
        elif te_order == TimeIntegMethod.EXACT:
            func = self.local_exact_solve
        elif te_order == TimeIntegMethod.Euler:
            func = self.local_euler
        elif te_order == TimeIntegMethod.LW:
            func = self.local_lax_wendroff
        elif te_order == TimeIntegMethod.CN:
            func = self.local_implicit_solve
            # def func(left_site_pos: int, nsites: int, return_intermediates=False, **kwargs):
            #     self.local_implicit_solve(left_site_pos, nsites, return_intermediates=return_intermediates,
            #                               weight=0.6, **kwargs)
        elif te_order == TimeIntegMethod.CN6:
            def func(left_site_pos: int, nsites: int, return_intermediates=False, **kwargs):
                return self.local_implicit_solve(left_site_pos, nsites, return_intermediates=return_intermediates,
                                          weight=0.6, **kwargs)
        elif te_order == TimeIntegMethod.BE:
            def func(left_site_pos: int, nsites: int, return_intermediates=False, **kwargs):
                return self.local_implicit_solve(left_site_pos, nsites, return_intermediates=return_intermediates,
                                          weight=1.0, **kwargs)
        elif te_order == TimeIntegMethod.EXACT:
            func = self.local_exact_solve
        else:
            raise ValueError(f'invalid te_order ({te_order}) provided')

        self._local_solve_func = func

    # @classmethod
    # def euler_solver(cls, dt: Numeric,
    #                  init_state: Union[qtn.MatrixProductState, 'MPS'],
    #                  linear_operators: Sequence['qtn.MatrixProductOperator'],
    #                  sources: Optional[Sequence[Union['Term', 'qtn.MatrixProductState', MPS]]] = None,
    #                  nonlinear_terms: Optional[Sequence['Term']] = None,
    #                  ):
    #     """ df/dt = Af + nonlinear terms + sources
    #     """
    #     TermClass = cls.term_class
    #
    #     lin_term = TermClass(init_state.copy(), operators=linear_operators)
    #
    #     src_terms = []
    #     for src in sources:   ## include capability to deal with time-dependence
    #         if isinstance(src, qtn.MatrixProductState):
    #             src = TermClass(src.copy())
    #         elif isinstance(src, Term):
    #             pass
    #         else:
    #             raise TypeError('invalid type for source terms')
    #         src_terms += [src]
    #
    #     nonlin_terms = [] if nonlinear_terms is None else nonlinear_terms
    #
    #     out = cls(init_state, [lin_term, src_terms, nonlin_terms], dt=dt, time=0., )


    def _site_solve(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor' = None, return_intermediates=True,
                    ) -> tuple[qtn.Tensor, Numeric]:

        # print('new TimeIntegrator site solve', left_site_pos)

        plot_partitions = False
        if plot_partitions:
            copy = self.init_ket.copy()
            l_partition = copy[:left_site_pos]
            r_partition = copy[left_site_pos + nsites:]

            if l_partition.num_tensors > 0:
                l_inds = [copy.site_ind_id.format(i) for i in range(left_site_pos)]
                xl_bond = copy.bond(left_site_pos, left_site_pos - 1)
                l_size = copy.bond_size(left_site_pos, left_site_pos - 1)

                l_tens = l_partition.contract()
                l_tens.transpose(xl_bond, *l_inds, inplace=True)
                plt.figure()
                plt.plot(l_tens.data.reshape(l_size, -1).T)
                plt.title(f'left bases, {left_site_pos}')

            if r_partition.num_tensors > 0:
                r_inds = [copy.site_ind_id.format(i) for i in range(left_site_pos + nsites, self.init_ket.L)]
                xr_bond = copy.bond(left_site_pos + nsites - 1, left_site_pos + nsites)
                r_size = copy.bond_size(left_site_pos + nsites - 1, left_site_pos + nsites)

                r_tens = r_partition.contract()
                r_tens.transpose(xr_bond, *r_inds, inplace=True)
                plt.figure()
                plt.plot(r_tens.data.reshape(r_size, -1).T)
                plt.title(f'right bases, {left_site_pos}')

            plt.show()

        at_end = (left_site_pos == self.L - nsites) if self.direction == SweepDirection.RIGHT else (left_site_pos == 0)
        return_intermediates = return_intermediates if not at_end else False
        # print('at end', at_end, 'inter', return_intermediates)
        # if at_end:
        #     pdb.set_trace()

        # print('TI return intermediates', return_intermediates)
        if at_end:
            # self._set_local_solve_func(self.te_order_target)
            self._set_local_solve_func(self.te_order_final)
        else:
            self._set_local_solve_func(self.te_order_target)

        out = super()._site_solve(left_site_pos, nsites, site_tens=site_tens,
                                  return_intermediates=return_intermediates)   ## target_rdm_list, error

        # ### plot out ###
        # import local_solvers.helper_cross as helper_cross
        # coords = helper_cross.get_selectors(self.out, left_site_pos, nsites)
        # selectors = []
        # for c in coords:
        #     selectors += [int("".join(str(x) for x in c), 2)]
        #
        # inds = []
        # i = left_site_pos
        # if i > 0:
        #     inds += [self.out.bond(i, i - 1)]
        # inds += [self.out.site_ind(i + ix) for ix in range(nsites)]
        # if i + nsites < self.out.L:
        #     inds += [self.out.bond(i + nsites - 1, i + nsites)]
        #
        # tens = out[0][0]
        # print('out', out)
        # print('tens', tens, len(out))
        # tens.transpose(*inds, inplace=True)
        # plt.plot(selectors, tens.data.reshape(-1), 'o')
        # plt.title('solved site')
        # plt.show()
        #
        # #####

        ## targeting "intermediate ket" takes care of nonlinear terms
        # if not at_end and self.solver_type in [LocalSolverType.DMRG, LocalSolverType.TDDMRG, LocalSolverType.TDVP]:
        #     out_targets = out[0]
        #     nonlin_targets = []
        #     if self.verbose > 2:
        #         print('self.nonlinear terms', self.nonlinear_terms)
        #
        #     for nl_term in self.nonlinear_terms:
        #         if self.verbose > 2:
        #             print('nl term', nl_term._proj_vec_targets)
        #         nonlin_targets += [nl * out_targets[0].norm()/nl.norm() for nl in nl_term._proj_vec_targets]
        #         if self.verbose > 2:
        #             print('added nonlinear target terms', len(nonlin_targets), [tg.norm() for tg in nonlin_targets])
        #         # if len(nonlin_targets) == 0:
        #         #     raise RuntimeError
        #
        #     out = (*out[0], *nonlin_targets), out[1]

        # print('TI site solve out', out)

        return out


    def deriv_func(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor' = None, time: Numeric = None
                   ) -> 'qtn.Tensor':
        """ compute df/dt = Af + sources + nonlinear terms O[f]
        """
        if self.verbose > 3:
            print('deriv time', time, self.time)

        # print('site tens', site_tens.norm() if site_tens is not None else None)

        # if self.upwind_deriv_func is not None:
        #     return self.local_deriv_upwind(left_site_pos, nsites, time=time, site_tens=site_tens,
        #                                    return_intermediates=False, )

        # pdb.set_trace()

        site_tens_list = []
        for term in self.linear_terms:
            site_tens_list += [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)]

            # from local_solvers import helper_cross
            # print('site tens', site_tens, len(term.operators))
            # print('term.intermediates', term._intermediate_kets, term.num_tiers)
            # helper_cross.plot_submat(term.bra,  # term.get_intermediate_ket(0),
            #                          left_site_pos, nsites, site_tens_list[-1],
            #                          ref_kets=[helper_quimb.apply(term.operators[0], term.ket)],
            #                          plt_title='lin term submat')

            # dt = self.dt
            # if dt < 0:
            #     print('backwards TE * -1')
            #     site_tens_list[-1].modify(apply=lambda x: x * -1)

        if self.verbose > 4:
            print('evaluate nonlinear terms and sources')
        for term in self.nonlinear_terms + self.sources:
            site_tens_list += [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens,
                                                       verbose_plot=self.verbose_plot)]

            if self.verbose > 4:
                print('term intermediate', term._intermediate_kets)
                print('term blocks', term.blocks)
            # from local_solvers import helper_cross
            # helper_cross.plot_submat(term.ket, left_site_pos, 1, site_tens_list[-1],
            #                          select_inds=self.self_term.bra.select_inds,
            #                          ref_kets=[term.ket, term._intermediate_kets[0]],
            #                          plt_title='eval site deriv nl')
            # print('term intermediate sites', term._intermediate_sites)
            # for ix in range(term.num_tiers - 1):
            #     inter_tens = term._intermediate_sites[ix]
            #     inter_ket = term._intermediate_kets[ix]
            #     print('inte tens', inter_tens)
            #     for inter_t in inter_tens:
            #         helper_cross.plot_submat(inter_ket, left_site_pos, 1, inter_t,
            #                                  # select_inds=self.self_term.bra.select_inds,
            #                                  ref_kets=[term.ket, inter_ket],
            #                                  plt_title=f'eval site deriv nl inter {ix}')

            # if left_site_pos == self.ket.L - 2:
            #     print(term._intermediate_kets.keys(), term.num_tiers)
            #     plt.figure()
            #     plt.plot(helper_quimb.to_dense(term.bra, [term.bra.site_ind_id.format(i) for i in range(self.L)]).data.reshape(-1),
            #              label='bra')
            #     plt.plot(helper_quimb.to_dense(term._intermediate_kets[0],
            #                                    [term.bra.site_ind_id.format(i) for i in range(self.L)]).data.reshape(-1),
            #              label='ket0')
            #     plt.plot(helper_quimb.to_dense(term.ket, [term.ket.site_ind_id.format(i) for i in range(self.L)]).data.reshape(-1), '--',
            #              label='ket')
            #     plt.title(f'term {left_site_pos}')
            #     plt.legend()
            #     plt.show()

            ## doesn't perform well if incorporate site_tens


        # ### plot (cross)  ###
        # import local_solvers.helper_cross as helper_cross
        # coords = helper_cross.get_selectors(self.out, left_site_pos, nsites)
        # selectors = []
        # for c in coords:
        #     selectors += [int("".join(str(x) for x in c), 2)]
        #
        # inds = []
        # i = left_site_pos
        # if i > 0:
        #     inds += [self.out.bond(i, i - 1)]
        # inds += [self.out.site_ind(i + ix) for ix in range(nsites)]
        # if i + nsites < self.out.L:
        #     inds += [self.out.bond(i + nsites - 1, i + nsites)]
        #
        # for tens in site_tens_list:
        #     tens.transpose(*inds, inplace=True)
        #     plt.plot(selectors, tens.data.reshape(-1), 'x')
        # plt.title('time integ deriv func eval')
        # plt.show()

        # print('deriv func', site_tens_list)
        # print('DERIV FUNC OUT', [t.norm() for t in site_tens_list])
        # print('site tens list', site_tens_list)
        tot_site = helper_tn.sum_tens(site_tens_list)
        # print('tot ssite', tot_site.norm())
        return tot_site


    def local_lax_wendroff(self, left_site_pos: int, nsites: int, return_intermediates=False,
                                 dt: Numeric = None, time: Numeric = None, site_tens: 'qtn.Tensor'=None
                                 ) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
        """ lax-wendroff
        """
        dt = self.dt if dt is None else dt      ## already included in terms
        ref_ket = self.out # self.init_ket if self.solver_type == LocalSolverType.Cross else self.out
        ## changed this for time_integrator_cross_2

        if site_tens is None:
            ket_x = self.self_term.get_evaluated_site(left_site_pos, nsites).copy()
            # ket_x = self.terms[0].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
            # ket_x.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
        else:
            # ket_x = self.self_term.get_evaluated_site(left_site_pos, nsites).copy()
            # print('ket x - site_tens', helper_quimb.add_tensors(ket_x * -1, site_tens).norm())
            # pdb.set_trace()
            ket_x = site_tens

        ket_x = ket_x.copy()
        ket_x.modify(apply=lambda x: x * 10 ** self.out.exponent)    ## incorporate exponent

        if nsites == 0:
            tmp = self.out.bond(left_site_pos, left_site_pos + 1)
            inds = [tmp + '_L', tmp + '_R']
        else:
            inds = []
            if left_site_pos > 0:
                inds += [self.out.bond(left_site_pos, left_site_pos - 1)]
            inds += [self.out.site_ind(left_site_pos + i) for i in range(nsites)]
            if left_site_pos + nsites < self.out.L:
                inds += [self.out.bond(left_site_pos + nsites - 1, left_site_pos + nsites)]

        ket_x = ket_x.transpose(*inds, inplace=True)
        ket_x_orig = ket_x.copy()

        intermediate_tens = [ket_x_orig]

        if self.verbose > 2:
            print('LW', left_site_pos, nsites)

        for it in [0, 1]:  # , op_terms in self.extra_terms_dict.items():
            op_terms = self.extra_terms_dict[it]

            dt_ = dt/2 if it == 0 else dt

            ## averaged tens
            if it == 0:
                avg_term = self.extra_terms_dict['avg'][0]
                site_tens = (ket_x if avg_term.ket is ref_ket else None)
                avg_x = avg_term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)
                tens_list = [avg_x]
                # print('avg x', avg_x.norm(), ket_x_orig.norm())
                # pdb.set_trace()
                # tens_list = [ket_x_orig]

                # ket = self.out.copy()
                # site_tens.transpose_like(ket[left_site_pos], inplace=True)
                # ket[left_site_pos].modify(data=site_tens.data)
                # avg_data = self.grid.map_mps_to_state(self.grid.make_gridTN(ket))
                # orig_data = self.grid.map_mps_to_state(self.grid.make_gridTN(self.out))

                # plt.figure()
                # plt.imshow(avg_data)
                # plt.colorbar()
                # plt.title('avg dist local')
                #
                # plt.figure()
                # plt.imshow(orig_data - avg_data)
                # plt.colorbar()
                # plt.title('orig - avg dist local')
                # plt.show()
            else:
                tens_list = [ket_x_orig]

            ## keys should be 0 and 1 for the different stages
            for op_term in op_terms:
                site_tens = (ket_x.copy() if op_term.ket is ref_ket else None)
                op_tens = op_term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)
                # op_tens = op_tens.copy()
                # print('op term exp', op_term.ket.exponent, op_term.bra.exponent, [op.exponent for op in op_term.operators])
                # print('op term exp', op_term.vec_block.exponent, [op.exponent for op in op_term.op_blocks[0]])
                # print('op term', op_term.op_blocks)
                # print('op tens norm', op_tens.norm())
                op_tens.modify(apply=lambda x: x * dt_)
                ## some terms are not MPO blocks; rather just MPS that need to be evaluated at certain grid points

                tens_list += [op_tens]

            # deriv_sum = helper_tn.sum_tens(tens_list[1:], transpose_bonds=ket_x.inds).copy()
            # print('deriv norm', deriv_sum.norm())

            out_tens = helper_tn.sum_tens(tens_list, transpose_bonds=ket_x.inds).copy()
            # out_tens.transpose_like(ket_x, inplace=True)
            out_tens.modify(apply=lambda x: x * 10 ** self.out.exponent)
            ## 1: output of first stage, 2: derivative to be aded to ket_orig

            if it == 0:
                ket_x = out_tens.copy()     ## 1: output of first stage, 2: derivative to be aded to ket_orig
            intermediate_tens += [out_tens]


        out_tens.transpose_like(ket_x, inplace=True)
        out_tens.modify(apply=lambda x: x * 10 ** (-self.out.exponent))
        for x_tens in intermediate_tens:
            x_tens.modify(apply=lambda x: x * 10 ** (-self.out.exponent))

        if return_intermediates:  ## new sites, intermediates
            return out_tens, intermediate_tens

        return out_tens


    def local_lax_wendroff_so(self, left_site_pos: int, nsites: int, return_intermediates=False,
                                 dt: Numeric = None, time: Numeric = None, site_tens: 'qtn.Tensor'=None
                                 ) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
        """ lax-wendroff with split-operator scheme (1D advection), maccormack scheme
        """
        ref_ket = self.out  # self.init_ket if self.solver_type == LocalSolverType.Cross else self.out

        dt = self.dt if dt is None else dt      ## already included in terms
        dt_ = dt / 2
        if self.verbose > 2:
            print('LWSO', left_site_pos, nsites)

        if site_tens is None:
            ket_x = self.self_term.get_evaluated_site(left_site_pos, nsites).copy()
            # ket_x = self.terms[0].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
            # ket_x.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
        else:
            # ket_x = self.self_term.get_evaluated_site(left_site_pos, nsites).copy()
            # print('ket x - site_tens', helper_quimb.add_tensors(ket_x * -1, site_tens).norm())
            # pdb.set_trace()
            ket_x = site_tens

        ket_x = ket_x.copy()
        ket_x.modify(apply=lambda x: x * 10 ** self.out.exponent)    ## incorporate exponent

        if nsites == 0:
            tmp = self.out.bond(left_site_pos, left_site_pos + 1)
            inds = [tmp + '_L', tmp + '_R']
        else:
            inds = []
            if left_site_pos > 0:
                inds += [self.out.bond(left_site_pos, left_site_pos - 1)]
            inds += [self.out.site_ind(left_site_pos + i) for i in range(nsites)]
            if left_site_pos + nsites < self.out.L:
                inds += [self.out.bond(left_site_pos + nsites - 1, left_site_pos + nsites)]

        ket_x = ket_x.transpose(*inds, inplace=True)
        ket_x_orig = ket_x.copy()

        intermediate_tens = [ket_x_orig]

        keys = self.extra_terms_dict.keys()

        for it in [0, 1]:
            keys_ = keys if it==0 else [*keys][::-1]

            for key in keys_:     ## Axis
                op_terms = self.extra_terms_dict[key]
                fd_term = op_terms[0]
                bd_term = op_terms[1]


                site_tens = (ket_x if fd_term.ket is ref_ket else None)
                fd_deriv = fd_term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)
                fd_deriv.modify(apply=lambda x: x * dt_)
                site_tens_prime = helper_tn.sum_tens([site_tens, fd_deriv])

                site_tens_prime_ = (site_tens_prime if fd_term.ket is ref_ket else None)
                bd_deriv = bd_term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens_prime_)
                bd_deriv.modify(apply=lambda x: x * dt_)

                new_site_tens = helper_tn.sum_tens([site_tens, site_tens_prime, bd_deriv])
                new_site_tens.modify(apply=lambda x: x / 2)

                ket_x = new_site_tens

                intermediate_tens += [ket_x.copy()]

        out_tens = ket_x
        out_tens.modify(apply=lambda x: x * 10 ** (-self.out.exponent))
        for x_tens in intermediate_tens:
            x_tens.modify(apply=lambda x: x * 10 ** (-self.out.exponent))

        if return_intermediates:  ## new sites, intermediates
            return out_tens, intermediate_tens

        return out_tens


    def euler_func(self, left_site_pos: int, nsites: int, dt: Numeric = None, time: Numeric = None,
                   deriv: 'qtn.Tensor' = None, site_tens: 'qtn.Tensor' = None):
        """ compute df/dt = Af + sources + nonlinear terms
                assumes the first term is the linear term, followed by sources and nonlinear terms
                (though those are effectively the same)
            for the given effective operators
        """
        # site_inds = list(range(left_site_pos, left_site_pos + nsites))

        # direction = self.direction
        # at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

        # current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])

        # ket_x = self.terms[0].proj_vec if site_tens is None else site_tens
        if site_tens is None:
            # ket_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
            ket_x = self.terms[0].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
            ket_x.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
        else:
            ket_x = site_tens

        # print('site tens', site_tens.inds, site_tens.data)
        self.num_evals += ket_x.size
        # print('self.num evals', self.num_evals, ket_x.size)

        if deriv is None:
            deriv = self.deriv_func(left_site_pos, nsites, time=time, site_tens=site_tens)

        dt = self.dt if dt is None else dt
        # print('EULER DT', dt)
        deriv = deriv.copy()    ## already excludes self.ket.exponent
        deriv.modify(apply=lambda x: x * dt)
        # print('EULER deriv', deriv.norm())
        # print('EULER ket x', ket_x.norm())
        ket_x = ket_x.copy()
        ket_x.modify(apply=lambda x: x * 10**(-self.init_ket.exponent))  ## remove exponent
        # print('exponent', self.init_ket.exponent)
        # print('EULER ket x', ket_x.norm())
        site_tens_list = [ket_x, deriv]

        # print(self.terms[0].bra is self.ket)
        # print(self.terms[0].vec_block.bra is self.ket)
        # print('ket_x', ket_x)
        # print('deriv', deriv)

        tot_site = helper_tn.sum_tens(site_tens_list)
        # print('euler tot site', tot_site.norm())

        tot_site.modify(apply=lambda x: x * 10 ** self.init_ket.exponent)  ## include exponent for input into bra
        return tot_site


    def exponential_func(self, left_site_pos: int, nsites: int, dt: Numeric = None, time: Numeric = None,
                         # deriv: 'qtn.Tensor' = None,
                         site_tens: 'qtn.Tensor' = None, return_intermediates=False):
        """ compute df/dt = Af + sources + nonlinear terms
                assumes the first term is the linear term, followed by sources and nonlinear terms
                (though those are effectively the same)
            ## here we focus on Euler time step (EXP1, ETD1)
            ## u_{n+1} = exp(dt J_n) u_n + dt phi_1(dt J_n) N_n  where J: effective operator, N: effective source terms
        """
        dt = self.dt if dt is None else dt
        # site_inds = list(range(left_site_pos, left_site_pos + nsites))

        # direction = self.direction
        # at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

        # current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])

        # ket_x = self.terms[0].proj_vec if site_tens is None else site_tens
        if site_tens is None:
            # # ket_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
            # ket_x = self.terms[0].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
            # ket_x.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
            ket_x = self.self_term.vec_block.get_projected(left_site_pos, nsites, return_combined=True)
            ket_x.reindex(self.self_term.vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
        else:
            ket_x = site_tens

        ket_x = ket_x.copy()
        ket_x.modify(apply=lambda x: x * 10 ** (-self.init_ket.exponent))  ## remove exponent
        sq_shape = ket_x.size

        ## nonlinear operators
        sources = []
        for term in self.nonlinear_terms + self.source_terms:
            # data = term.ket.to_dense().reshape(-1)
            # plt.figure()
            # plt.plot(data * 10**(term.ket.exponent))
            # plt.title(f'source term, {term.ket.exponent}')
            # plt.show()
            # print('DERIV FUNC site tens', site_tens.norm() if site_tens is not None else None)
            sources += [term.get_evaluated_site(left_site_pos, nsites)]  # , site_tens=site_tens)]

        if len(sources) > 0:
            tot_sources = helper_tn.sum_eff_TNs(sources)
            tot_sources.transpose_like(ket_x, inplace=True)
            tot_sources.modify(apply=lambda x: x * 10 ** (-self.init_ket.exponent))  ## remove exponent
        else:
            tot_sources = None


        ## effective operators
        eff_ops = []
        for term in self.linear_terms:
            eff_ops += [*term.get_eff_operator(left_site_pos, nsites)[0]]
            ## dict for each "level": list of effective operators

        if len(eff_ops) > 0:
            term = next(iter(self.linear_terms))
            b2k_inds = term.projected_bra_to_ket(left_site_pos, nsites)
            k2b_inds = {k: b for b, k in b2k_inds.items()}

            ket_inds = ket_x.inds
            bra_inds = [k2b_inds[k] for k in ket_inds]
            tot_eff_op = helper_tn.sum_eff_TNs(eff_ops)
            tot_eff_op.transpose(*bra_inds, *ket_inds, inplace=True)

            op_data = tot_eff_op.data.reshape(sq_shape,sq_shape)

            eff_evals, eff_evecs = np.linalg.eig(op_data)
            inv_eff_evecs = np.linalg.inv(eff_evecs)
            exp_eff_op = (eff_evecs * np.exp(dt * eff_evals)) @ inv_eff_evecs

            phi1 = None
            if tot_sources is not None:
                if self.verbose > 1:
                    print('tot sources is not None')

                ## phi_1(J) = J^{-1} (exp(J) - I)
                ## phi_2(J) = J^{-2} (exp(J) - I - J)
                ## phi_3(J) = J^{-3} (exp(J) - I - J - J^2/2) ...
                ## phi_k(J) = J^{-1} (phi_{k-1}(J) - phi_{k-1}(J))

                ## here we focus on Euler time step (EXP1, ETD1)
                ## u_{n+1} = exp(dt J_n) u_n + dt pi_1(dt J_n) N_n
                ## where J: effective operator, N: effective source / nonlinear terms

                tmp = (np.exp(eff_evals * dt) - 1) / eff_evals / dt
                phi1_D = np.where(np.abs(eff_evals) > 10e-15 / dt, tmp, 1.)
                phi1 = (eff_evecs * phi1_D) @ inv_eff_evecs

        else:
            exp_eff_op = None
            phi1 = None         ## = 1

        ## here we focus on Euler time step (EXP1, ETD1)
        ## u_{n+1} = exp(dt J_n) u_n + dt phi_1(dt J_n) N_n  where J: effective operator, N: effective source terms

        ## linear term
        lin_term = exp_eff_op @ ket_x.data.reshape(-1) if exp_eff_op is not None else ket_x.data.reshape(-1)
        ## nonlinear/source term
        nl_term = phi1 @ tot_sources.data.reshape(-1) if phi1 is not None else \
                        (tot_sources.data.reshape(-1) if tot_sources is not None else 0.)

        out = lin_term + nl_term * dt
        tot_site = qtn.Tensor(out.reshape(ket_x.shape), inds = ket_x.inds)
        # tot_site.modify(apply=lambda x: x * 10 ** self.ket.exponent)  ## include exponent for input into bra

        # print('tot site', tot_site.norm(), ket_x.norm())

        if return_intermediates:
            return tot_site, [tot_site, ket_x]

        return tot_site

    def local_euler(self, left_site_pos: int, nsites: int, return_intermediates=False,
                  dt: Numeric=None, time: Numeric=None, site_tens: 'qtn.Tensor'=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        dt = self.dt if dt is None else dt
        if self.verbose:
            print('EULER DT', dt, left_site_pos)

        if self.local_euler_func is not None:
            self.combine_terms_func = self.local_euler_func
            return super()._site_solve(left_site_pos, nsites, site_tens=site_tens)

        # print('local Euler', left_site_pos, nsites, site_tens)
        if site_tens is None:
            self.terms[0].get_evaluated_site(left_site_pos, nsites)
            state0 = self.terms[0].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
        else:
            state0 = site_tens
        state0.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)

        def deriv_func(site_tens: 'qtn.Tensor', time=None , **kwargs):
            return self.deriv_func(left_site_pos, nsites, site_tens=site_tens, time=time)

        def euler_func(site_tens: 'qtn.Tensor', dt, deriv0: qtn.Tensor=None, time=None, **kwargs):
            return self.euler_func(left_site_pos, nsites, deriv=deriv0, dt=dt, site_tens=site_tens, time=time)

        def scale_func(site_tens: 'qtn.Tensor', coeff: Numeric, inplace: bool = False):
            site_tens = site_tens if inplace else site_tens.copy()
            site_tens.modify(apply=lambda x: x * coeff)
            return site_tens

        def add_func(tens1: 'qtn.Tensor', tens2: 'qtn.Tensor', inplace=False, **kwargs):
            return helper_quimb.add_tensors(tens1, tens2, inplace=inplace)

        deriv0 = deriv_func(state0, time=self.time)
        state1 = euler_func(state0, dt, deriv0)
        # print('return intermediates', return_intermediates)
        # out = state0

        ## remove exponent from result
        if return_intermediates:  ## new sites, intermediates
            out = (state1, [state0])
            out[0].modify(apply=lambda x: x * 10**(-self.init_ket.exponent))
            out[1][0].modify(apply=lambda x: x * 10**(-self.init_ket.exponent))

            ## note: derivs (out[1][1:4]) already don't contain exponents. (it was added in euler_func step)

            # nonlin_targets = []
            # for nl_term in self.nonlinear_terms:
            #     nonlin_targets += [nl * state0.norm()/nl.norm() for nl in nl_term._proj_vec_targets]
            #     print('added nonlinear target terms', len(nonlin_targets), [tg.norm() for tg in nonlin_targets])
            #
            # intermediates = (*out[1], *nonlin_targets)
            # return out[0], intermediates

        else:
            out = state1
            out.modify(apply=lambda x: x * 10 ** (-self.init_ket.exponent))

        return out


    def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False,
                  dt: Numeric=None, time: Numeric=None, site_tens: 'qtn.Tensor'=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        dt = self.dt if dt is None else dt
        # print('RK4 DT', dt, left_site_pos, nsites, time)

        # print('local RK4', left_site_pos, nsites, site_tens)
        if site_tens is None:
            state0 = self.self_term.get_evaluated_site(left_site_pos, nsites)
            # state0 = self.self_term.vec_block.get_projected(left_site_pos, nsites, return_combined=True)
        else:
            state0 = site_tens
        state0.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)

        def deriv_func(site_tens: 'qtn.Tensor', time=None , **kwargs):
            return self.deriv_func(left_site_pos, nsites, site_tens=site_tens, time=time)

        def euler_func(site_tens: 'qtn.Tensor', dt, deriv0: qtn.Tensor=None, time=None, **kwargs):
            return self.euler_func(left_site_pos, nsites, deriv=deriv0, dt=dt, site_tens=site_tens, time=time)

        def scale_func(site_tens: 'qtn.Tensor', coeff: Numeric, inplace: bool = False):
            site_tens = site_tens if inplace else site_tens.copy()
            site_tens.modify(apply=lambda x: x * coeff)
            return site_tens

        def add_func(tens1: 'qtn.Tensor', tens2: 'qtn.Tensor', inplace=False, **kwargs):
            return helper_quimb.add_tensors(tens1, tens2, inplace=inplace)

        out = helper_TE.rk4(state0, dt, euler_func, deriv_func, add_func, scale_func, time=self.time,
                            return_intermediates=return_intermediates)

        ## remove exponent from result
        if return_intermediates:  ## new sites, intermediates
            out[0].modify(apply=lambda x: x * 10**(-self.init_ket.exponent))
            out[1][0].modify(apply=lambda x: x * 10**(-self.init_ket.exponent))

            ## note: derivs (out[1][1:4]) already don't contain exponents. (it was added in euler_func step)

            # nonlin_targets = []
            # for nl_term in self.nonlinear_terms:
            #     nonlin_targets += [nl * state0.norm()/nl.norm() for nl in nl_term._proj_vec_targets]
            #     print('added nonlinear target terms', len(nonlin_targets), [tg.norm() for tg in nonlin_targets])
            #
            # intermediates = (*out[1], *nonlin_targets)
            # return out[0], intermediates

        else:
            out.modify(apply=lambda x: x * 10 ** (-self.init_ket.exponent))

        return out

    def local_rk(self, te_order: int, left_site_pos: int, nsites: int, return_intermediates=False,
                  dt: Numeric=None, time: Numeric=None, site_tens: 'qtn.Tensor'=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        dt = self.dt if dt is None else dt
        time = self.time if time is None else time
        if self.verbose:
            print(f'RK{te_order} DT', dt, time, left_site_pos, nsites)

        if te_order == 1:
            rk_func = helper_TE.euler
        elif te_order == 2:
            rk_func = helper_TE.rk2
        elif te_order == 3:
            rk_func = helper_TE.ssprk4   # ssprk3 or ssprk4
        elif te_order == 4:
            rk_func = helper_TE.rk4

        # print('local RK', left_site_pos, nsites, site_tens)
        if site_tens is None:
            state0 = self.self_term.get_evaluated_site(left_site_pos, nsites)
            # state0 = self.self_term.vec_block.get_projected(left_site_pos, nsites, return_combined=True)
        else:
            state0 = site_tens
            # state0 = self.self_term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)
        state0.reindex(self.self_term.vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
        # state0 = None

        def deriv_func(site_tens: 'qtn.Tensor', time=None , **kwargs):
            if te_order == 1:
                site_tens = None
            return self.deriv_func(left_site_pos, nsites, site_tens=site_tens, time=time)

        def euler_func(site_tens: 'qtn.Tensor', dt, deriv0: qtn.Tensor=None, time=None, **kwargs):
            return self.euler_func(left_site_pos, nsites, deriv=deriv0, dt=dt, site_tens=site_tens, time=time)

        def scale_func(site_tens: 'qtn.Tensor', coeff: Numeric, inplace: bool = False):
            site_tens = site_tens if inplace else site_tens.copy()
            site_tens.modify(apply=lambda x: x * coeff)
            return site_tens

        def add_func(tens1: 'qtn.Tensor', tens2: 'qtn.Tensor', inplace=False, **kwargs):
            return helper_quimb.add_tensors(tens1, tens2, inplace=inplace)

        # deriv0 = self.deriv_func(left_site_pos, nsites, site_tens=site_tens, time=time)
        # tmp = self.euler_func(left_site_pos, nsites, deriv=deriv0, dt=dt, site_tens=site_tens, time=time)
        # deriv0 = deriv_func(state0, time=time)
        # tmp = euler_func(state0, dt, deriv0=deriv0, time=time)
        # out = tmp
        out = rk_func(state0, dt, euler_func, deriv_func, add_func, scale_func, time=time,
                            return_intermediates=return_intermediates)

        # print('self.init ket', self.init_ket.exponent)
        # print('self.out', self.out.exponent)
        # print('out', out)
        # print('tmp', tmp)
        # print(helper_quimb.add_tensors(out, tmp * -1).norm())
        # print(helper_quimb.add_tensors(out, state0 * -1).norm())
        # pdb.set_trace()

        return out

        # ## remove exponent from result
        # if return_intermediates:  ## new sites, intermediates
        #     out[0].modify(apply=lambda x: x * 10**(-self.init_ket.exponent))
        #     out[1][0].modify(apply=lambda x: x * 10**(-self.init_ket.exponent))
        #
        #     ## note: derivs (out[1][1:4]) already don't contain exponents. (it was added in euler_func step)
        #
        #     # nonlin_targets = []
        #     # for nl_term in self.nonlinear_terms:
        #     #     nonlin_targets += [nl * state0.norm()/nl.norm() for nl in nl_term._proj_vec_targets]
        #     #     print('added nonlinear target terms', len(nonlin_targets), [tg.norm() for tg in nonlin_targets])
        #     #
        #     # intermediates = (*out[1], *nonlin_targets)
        #     # return out[0], intermediates
        #
        # else:
        #     out.modify(apply=lambda x: x * 10 ** (-self.init_ket.exponent))
        #
        # return out



    def local_rk2(self, left_site_pos: int, nsites: int, return_intermediates=False,
                  dt: Numeric=None, time: Numeric=None, site_tens: 'qtn.Tensor'=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        return self.local_rk(2, left_site_pos, nsites, return_intermediates=return_intermediates,
                             dt=dt, time=time, site_tens=site_tens)

    def local_ssprk3(self, left_site_pos: int, nsites: int, return_intermediates=False,
                  dt: Numeric = None, time: Numeric = None, site_tens: 'qtn.Tensor' = None) -> Union[
        qtn.Tensor, Sequence['qtn.Tensor']]:

        return self.local_rk(3, left_site_pos, nsites, return_intermediates=return_intermediates,
                             dt=dt, time=time, site_tens=site_tens)



    def setup_crank_nicolson(self, left_site_pos: int, nsites: int, dt: Numeric = None, time: Numeric = None,
                             site_tens: qtn.Tensor = None, weight: Numeric = 0.5):
        """ implicit time differentiation:  dy/dt = F(y) + b + nonlinear(y)
            y_(n+1) = y_(n) + h/2 ( F(y_(n+1)) + F(y_(n)) )
            y_(n+1) = y_(n) + h * weight * F(y_(n+1)) + h * (1-weight) * F(y_(n))

            dy/dt = Ay + b, assuming b is constant in time
            y_(n+1) = y_(n) + h ( weight * (A(y_(n+1)) + b) + (1-weight) * (A(y_(n)) + b) )
            (I - A * h * weight) y_[n+1] = (I + (1 - weight) * h * A) y_[n] + h * b
        """
        dt = self.dt if dt is None else dt

        lin_ops = []
        for lin_term in self.linear_terms:
            op = lin_term.get_eff_operator(left_site_pos, nsites)[0]
            lin_ops += [op1.copy() for op1 in op]

        # lin_ops = self.linear_terms[0].get_eff_operator(left_site_pos, nsites)[0]
        # lin_ops = [op.copy() for op in lin_ops]
        # print('lin ops', lin_ops)

        for op in lin_ops:
            helper_quimb.scalar_multiply(op, - dt * weight, inplace=True)
            # op.exponent += [dt * weight]

        ## identity
        bra_to_ket_inds = self.terms[0].projected_bra_to_ket(left_site_pos, nsites)
        bra_inds = [b for b in bra_to_ket_inds.keys() if b is not None]
        ket_inds = [bra_to_ket_inds[b] for b in bra_inds]
        # sq_shape = [self.terms[0].ket[left_site_pos].ind_size(k) for k in ket_inds]
        # sq_size = int(np.prod(sq_shape))
        # iden = np.eye(sq_size).reshape(*sq_shape,*sq_shape)
        # iden_tens = qtn.Tensor(iden, inds=[*bra_inds, *ket_inds])
        # lhs = lin_ops + [iden_tens]

        ## RHS
        ## the (1 - weight) * dt * F(f) term
        ### assumes site_tens is not scaled by ket (term bra) exponent (exponent is incorporated)
        # rhs = [self.self_term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens).copy()]  ## linop

        # ## old RHS
        # # rhs = [self.linear_terms[0].get_evaluated_site(left_site_pos, nsites, site_tens=site_tens).copy()]  ## linop
        # rhs = []
        # for lin_term in self.linear_terms:
        #     op = lin_term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens).copy()  ## linop
        #     rhs += [op.copy()]
        # print('len rhs', len(rhs))
        #
        # rhs += [term.get_evaluated_site(left_site_pos, nsites).copy()
        #         for term in self.source_terms + self.nonlinear_terms]    ## sources, nonlin
        # for rhs_term in rhs:
        #     rhs_term.modify(apply=lambda x: x * (1 - weight) * dt)
        # print('len rhs', len(rhs))
        #
        # ## the I * y[n] term --> moved to start
        # if site_tens is None:
        #     # rhs += [self.self_term.vec_block.projected_site.copy()]
        #     proj_vec = self.self_term.get_evaluated_site(left_site_pos, nsites)
        #     proj_vec.reindex(bra_to_ket_inds, inplace=True)
        #     rhs += [proj_vec]
        #
        #     sq_shape = [proj_vec.ind_size(k) for k in ket_inds]
        # else:
        #     site_tens = site_tens.copy()
        #     site_tens.modify(apply=lambda x: x * 10 ** -self.init_ket.exponent)
        #     rhs += [site_tens]
        #
        #     sq_shape = [site_tens.ind_size(k) for k in ket_inds]
        # # print('site tens norm', rhs[-1].norm())

        ## NEW RHS
        ## linear operator term
        rhs_op = [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens).copy() for term in
                  self.linear_terms]
        for rhs_tens in rhs_op:
            rhs_tens.modify(apply=lambda x: x * (1 - weight) * dt)

        ## source terms + nonlinear terms
        rhs_src = [term.get_evaluated_site(left_site_pos, nsites).copy() for term in
                   self.source_terms + self.nonlinear_terms]
        for rhs_tens in rhs_src:
            rhs_tens.modify(apply=lambda x: x * dt)

        ## the I * y[n] term
        if site_tens is None:
            # rhs = [self.self_term.vec_block.projected_site.copy()]
            proj_vec = self.self_term.get_evaluated_site(left_site_pos, nsites)
            proj_vec.reindex(bra_to_ket_inds, inplace=True)
            rhs = [proj_vec]
            sq_shape = [proj_vec.ind_size(k) for k in ket_inds]
        else:
            site_tens = site_tens.copy()
            site_tens.modify(apply=lambda x: x * 10 ** -self.out.exponent)
            rhs = [site_tens]
            sq_shape = [site_tens.ind_size(k) for k in ket_inds]

        rhs += rhs_op
        rhs += rhs_src

        sq_size = int(np.prod(sq_shape))
        iden = np.eye(sq_size).reshape(*sq_shape, *sq_shape)
        iden_tens = qtn.Tensor(iden, inds=[*bra_inds, *ket_inds])
        lhs = lin_ops + [iden_tens]

        # print('rhs', rhs, [t.norm() for t in rhs])

        return lhs, rhs


    def local_implicit_solve(self, left_site_pos: int, nsites: int, return_intermediates=False,
                             dt: Numeric = None, time: Numeric = None, site_tens: 'qtn.Tensor' = None,
                             **solver_kwargs) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        if self.verbose:
            print('implicit solver', solver_kwargs)

        if site_tens is None:
            state0 = self.self_term.get_evaluated_site(left_site_pos, nsites)
            # state0 = self.self_term.vec_block.get_projected(left_site_pos, nsites, return_combined=True)
        else:
            state0 = site_tens
        state0.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
        ## using init_guess for targeting didn't work, for some reason?

        lhs, rhs = self.setup_crank_nicolson(left_site_pos, nsites, dt=dt, time=time, site_tens=site_tens,
                                             **solver_kwargs)
        # init_guess = rhs[-1].copy()  ## old version
        init_guess = rhs[0].copy()  # self.self_term.get_evaluated_site(left_site_pos, nsites)
        output_to_input_inds = self.terms[0].projected_bra_to_ket(left_site_pos, nsites)

        rhs = helper_tn.sum_tens(rhs)
        solve_cgd = tc.get_cgd_func(has_constraints=(len(self.constraint_terms) > 0))

        constraint_vals = []
        for cv in self.constraint_vals:
            if isinstance(cv, Term):
                constraint_vals += [cv.get_evaluated_site(left_site_pos, nsites)]   ## <x|c>
                ## do not include site_tens because ket corresponds to reference vec
            else:
                constraint_vals += [qtn.Tensor(cv)]

        constraint_ops = []
        for ct in self.constraint_terms:
            if len(ct.operators) > 0:
                constraint_ops += [ct.get_eff_operator(left_site_pos, nsites)[0]]      ## <x|A|x>
            else:
                constraint_ops += [ct.get_evaluated_site(left_site_pos, nsites)[0]]    ## <x|c>


        out, err = solve_cgd(rhs, lhs, output_to_input_inds, init_guess=init_guess,
                             constraint_tns=constraint_ops, constraint_vals=constraint_vals)

        if return_intermediates:
            return out, [state0, rhs, out]
        else:
            return out


    def local_exact_solve(self, left_site_pos: int, nsites: int, return_intermediates=False,
                          dt: Numeric = None, time: Numeric = None, site_tens: qtn.Tensor = None,
                          **solver_kwargs) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        out = self.exponential_func(left_site_pos, nsites, dt=dt, time=time, site_tens=site_tens,
                                    return_intermediates=return_intermediates)
        return out



class TDDMRG(TimeIntegrator, DMRGEvaluator):
    """ RK4 follows Feiguin and White
        CN obtains three intermediate states by simply taking three time steps
    """

    def local_euler(self, left_site_pos: int, nsites: int, return_intermediates=False, dt: Numeric = None,
                  time: Numeric = None, site_tens=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        if self.verbose > 1:
            print('TDDMRG local Euler', self.time)
        dt = self.dt if dt is None else dt

        if return_intermediates:
            out, rk_states = super().local_euler(left_site_pos, nsites, return_intermediates=True, dt=dt, time=time,
                                                 site_tens=site_tens)
            s0, = rk_states      ## initial x, stage 1, stage 2, stage 3

            if self.verbose > 1:
                print('targeting 0, dt state')
            return out, (s0, out)
        else:
            # exit()

            return super().local_euler(left_site_pos, nsites, return_intermediates=False, dt=dt, time=time)


    def local_rk3(self, left_site_pos: int, nsites: int, return_intermediates=False, dt: Numeric = None,
                  time: Numeric = None, site_tens=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:


        if self.verbose > 1:
            print('TDDMRG local RK3', self.time)
        dt = self.dt if dt is None else dt

        if return_intermediates:
            out, rk_states = super().local_rk(3, left_site_pos, nsites, return_intermediates=True, dt=dt, time=time,
                                               site_tens=site_tens)
            s0, k1, k2 = rk_states      ## initial x, stage 1, stage 2, stage 3

            k1.transpose_like(s0, inplace=True)
            k2.transpose_like(s0, inplace=True)

            if self.verbose > 1:
                print('targeting 0, dt state')
            return out, (s0, out)

        else:
            return super().local_rk(3, left_site_pos, nsites, return_intermediates=False, dt=dt, time=time)


    def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False, dt: Numeric = None,
                  time: Numeric = None, site_tens=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        # s0 = self.terms[0].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
        # s0.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)

        if self.verbose > 1:
            print('TDDMRG local RK4', self.time)
        # exit()

        dt = self.dt if dt is None else dt

        # if left_site_pos == 2:
        #     exit()

        if return_intermediates:
            out, rk_states = super().local_rk4(left_site_pos, nsites, return_intermediates=True, dt=dt, time=time,
                                               site_tens=site_tens)

            # k0, = rk_states

            s0, k1, k2, k3, k4 = rk_states      ## initial x, stage 1, stage 2, stage 3
            # print('s0', s0.norm())
            # print('derive norms', k1.norm() * dt, k2.norm() * dt, k3.norm() * dt, k4.norm() * dt)

            k1.transpose_like(s0, inplace=True)
            k2.transpose_like(s0, inplace=True)
            k3.transpose_like(s0, inplace=True)
            k4.transpose_like(s0, inplace=True)

            ## tot_denmat:  1/3 * rho(psi03) + 1/6 * rho(psi13) + 1/6 * rho(psi23) + 1/3 * rho(psi33)
            ## include weights here
            psi03 = s0.copy()
            psi03.modify(apply=lambda x: x * np.sqrt(1. / 3))
            psi13 = s0.copy()
            psi13.modify(apply=lambda x: x + dt * (1. / 162 * (31 * k1.data + 14 * k2.data + 14 * k3.data - 5 * k4.data)))
            psi13.modify(apply=lambda x: x * np.sqrt(1. / 6))
            psi23 = s0.copy()
            psi23.modify(apply=lambda x: x + dt * (1. / 81 * (16 * k1.data + 20 * k2.data + 20 * k3.data - 2 * k4.data)))
            psi23.modify(apply=lambda x: x * np.sqrt(1. / 6))
            psi33 = s0.copy()
            psi33.modify(apply=lambda x: x + dt * (1. / 6 * (k1.data + 2 * k2.data + 2 * k3.data + k4.data)))
            psi33.modify(apply=lambda x: x * np.sqrt(1. / 3))

            ## out and psi33 are the same thing, but psi33 is scaled

            if self.verbose:
                print('classic TD-DMRG')
            return out, (psi03, psi13, psi23, psi33)
            # print('only 0, dt target')
            # return out, (psi03, psi33)

        else:
            # exit()

            return super().local_rk4(left_site_pos, nsites, return_intermediates=False, dt=dt, time=time)


    def local_implicit_solve(self, left_site_pos: int, nsites: int, return_intermediates=False,
                             dt: Numeric = None, time: Numeric = None, site_tens: 'qtn.Tensor' = None,
                             **solver_kwargs) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        # print('local implicit')
        # exit()

        dt = self.dt if dt is None else dt
        # out = super().local_implicit_solve(left_site_pos, nsites, dt=dt, time=time, **solver_kwargs)
        out1 = super().local_implicit_solve(left_site_pos, nsites, dt=dt / 3, time=time, site_tens=site_tens, **solver_kwargs)
        out1_ = out1.copy()
        out1_.modify(apply=lambda x: x * 10 ** self.init_ket.exponent)
        out2 = super().local_implicit_solve(left_site_pos, nsites, dt=dt / 3, time=time, site_tens=out1_, **solver_kwargs)
        out2_ = out2.copy()
        out2_.modify(apply=lambda x: x * 10 ** self.init_ket.exponent)
        out3 = super().local_implicit_solve(left_site_pos, nsites, dt=dt / 3, time=time, site_tens=out2_, **solver_kwargs)
        out = out3.copy()
        #
        out0 = self.self_term.vec_block.projected_site.copy()
        out0.modify(apply=lambda x: x * 10 ** self.init_ket.exponent)
        out0.modify(apply=lambda x: x * np.sqrt(1. / 3))
        out1.modify(apply=lambda x: x * np.sqrt(1. / 6))
        out2.modify(apply=lambda x: x * np.sqrt(1. / 6))
        out3.modify(apply=lambda x: x * np.sqrt(1. / 3))

        if return_intermediates:
            # return out, (out)  # (out0, out1, out2)
            # return out, (out0, out1, out2, out3)
            if self.verbose > 1:
                print('only 0, dt target')
            return out, (out0, out3)
        else:
            return out




class TDDMRG_mod1(TimeIntegrator, DMRGEvaluator):
    """ 1. use three first-order time steps for targeting to enable taking larger time steps
            very unstable bc Euler is amplifying, i suppose
        2. only compute target up ot dt/2 (ie. target 0, dt/6, 2 dt/6, dt/2)
            not sure why but this also failed spectacularly.
        3. target [0, 2*dt/3]
            stable but incorrect
        final step: use RK4, in some number of time steps.
    """
    _rdm_points = [0., 1./3, 2./3, 1.]
    _rdm_weights = [1./3, 1./6, 1./6, 1./3]

    def set_rdm_locs(self, new_points: Sequence[Numeric], new_weights: Sequence[Numeric]):
        assert(np.sum(new_weights) == 1), f'new weights must sum to 1.0, not {np.sum(new_weights)}'
        assert(np.all([0 <= pt <= 1 for pt in new_points])), 'new points must be between 0 and 1'
        assert(len(new_points) == len(new_weights)), f'length of provided lists must be the same'

        self._rdm_points = new_points
        self._rdm_weights = new_weights

    def get_rdm_locs(self):
        return self._rdm_points, self._rdm_weights

    # def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False,
    #               dt: Numeric = None, time: Numeric = None, site0 = None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
    #     """ use three first-order time steps for targeting to enable taking larger time steps
    #         final step: use RK4, in some number of time steps.
    #         (note: RK4 requires 4 evaluations per time step)
    #     """
    #     dt = self.dt if dt is None else dt
    #
    #     if return_intermediates:   ## is not at end
    #
    #         # self.terms[0].get_evaluated_site(left_site_pos, nsites)
    #         # k0 = self.terms[0].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
    #         # k0.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
    #
    #         k1 = self.euler_func(left_site_pos, nsites, dt=dt/3)
    #         k2 = self.euler_func(left_site_pos, nsites, dt=dt/3, site_tens=k1)
    #         out = self.euler_func(left_site_pos, nsites, dt=dt/3, site_tens=k2)
    #
    #         k0 = self.self_term.vec_block.projected_site.copy()
    #         k0.modify(apply=lambda x: x * 10 ** self.ket.exponent)
    #
    #         print('k0', k0.norm(), k1.norm(), k2.norm(), out.norm(), self.ket.exponent)
    #         # exit()
    #
    #         k1.transpose_like(k0, inplace=True)
    #         k2.transpose_like(k0, inplace=True)
    #         out.transpose_like(k0, inplace=True)
    #
    #         ## tot_denmat:  1/3 * rho(psi03) + 1/6 * rho(psi13) + 1/6 * rho(psi23) + 1/3 * rho(psi33)
    #         ## include weights here
    #         psi03 = k0.copy()
    #         psi03.modify(apply=lambda x: x * np.sqrt(1. / 3))
    #         psi13 = k1.copy()
    #         psi13.modify(apply=lambda x: x * np.sqrt(1. / 6))
    #         psi23 = k2.copy()
    #         psi23.modify(apply=lambda x: x * np.sqrt(1. / 6))
    #         psi33 = out.copy()
    #         psi33.modify(apply=lambda x: x * np.sqrt(1. / 3))
    #
    #         ## out and psi33 are the same thing
    #
    #         return out, (psi03, psi13, psi23, psi33)
    #
    #     else:
    #         out = super().local_rk4(left_site_pos, nsites, return_intermediates=False, dt=dt / 2, time=time)
    #         # out = super().local_rk4(left_site_pos, nsites, return_intermediates=False, dt=dt/2, time=time)
    #         # out = super().local_rk4(left_site_pos, nsites, return_intermediates=False, dt=dt/2, time=time, state0=out)
    #         return out

    def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False, dt: Numeric = None,
                  time: Numeric = None, site_tens=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        # s0 = self.terms[0].vec_block.get_projected(left_site_pos, nsites, return_combined=True)
        # s0.reindex(self.terms[0].vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)

        if self.verbose > 1:
            print('new local RK4')
        # exit()

        dt = self.dt if dt is None else dt

        # if left_site_pos == 2:
        #     exit()

        s1 = self.euler_func(left_site_pos, nsites, dt=dt / 2, site_tens=site_tens)

        s0 = self.self_term.vec_block.projected_site.copy()
        s0.modify(apply=lambda x: x * 10 ** self.init_ket.exponent)

        s2 = self.euler_func(left_site_pos, nsites, dt=dt / 2, site_tens=s1)
        s3 = self.euler_func(left_site_pos, nsites, dt=dt / 2, site_tens=s2)
        s4 = s0 * 2 / 3 + s3 * 1 / 3
        out = self.euler_func(left_site_pos, nsites, dt=dt / 2, site_tens=s4)

        if return_intermediates:

            # s1 = self.euler_func(left_site_pos, nsites, dt=dt / 2)
            #
            # s0 = self.self_term.vec_block.projected_site.copy()
            # s0.modify(apply=lambda x: x * 10 ** self.ket.exponent)
            #
            # s2 = self.euler_func(left_site_pos, nsites, dt=dt/2, site_tens=s1)
            # s3 = self.euler_func(left_site_pos, nsites, dt=dt/2, site_tens=s2)
            # s4 = s0 * 2/3 + s3 * 1/3
            # out = self.euler_func(left_site_pos, nsites, dt=dt/2, site_tens=s4)


            s1.transpose_like(s0, inplace=True)
            s2.transpose_like(s0, inplace=True)
            s3.transpose_like(s0, inplace=True)
            s4.transpose_like(s0, inplace=True)
            out.transpose_like(s0, inplace=True)

            ## tot_denmat:  1/3 * rho(psi03) + 1/6 * rho(psi13) + 1/6 * rho(psi23) + 1/3 * rho(psi33)
            ## include weights here
            psi03 = s0.copy()
            psi03.modify(apply=lambda x: x * np.sqrt(1. / 3))
            psi13 = s2.copy()
            psi13.modify(apply=lambda x: x * np.sqrt(1. / 6))
            psi23 = s3.copy()
            psi23.modify(apply=lambda x: x * np.sqrt(1. / 6))
            psi33 = out.copy()
            psi33.modify(apply=lambda x: x * np.sqrt(1. / 3))

            ## out and psi33 are the same thing, but psi33 is scaled
            # print('norms', psi03.norm(), psi13.norm(), psi23.norm(), psi33.norm())

            # print('classic TD-DMRG')
            # return out, (psi03, psi13, psi23, psi33)
            if self.verbose > 1:
                print('only 0, dt target')
            return out, (psi03, psi33)
            # print('only 0, 2 dt/3 target')
            # return out, (psi03, psi23)

        else:
            # exit()
            # return super().local_rk4(left_site_pos, nsites, return_intermediates=False, dt=dt, time=time)
            return out
            # out = super().local_rk4(left_site_pos, nsites, return_intermediates=False, dt=dt / 2, time=time)
            # print(out.norm(), self.ket.exponent)
            # out = super().local_rk4(left_site_pos, nsites, return_intermediates=False, dt=dt / 2, time=time, state0=out)
            # print(out.norm(), self.ket.exponent)
            # return out


class TDVP_DMRG(TimeIntegrator, DMRGEvaluator):
    """ performs TDVP
    """
    def _set_local_solve_func(self, te_order: int):
        if te_order == TimeIntegMethod.LW:
            func = self.local_lax_wendroff_so
            self._local_solve_func = func
        else:
            super()._set_local_solve_func(te_order)

    def _site_time_evolution(self, site_tens: 'qtn.Tensor', left_site_pos: int, dt: Numeric):
        orig_dt = self.dt
        self.dt = dt
        # print('site tens SHAPE', site_tens.shape if site_tens is not None else None)
        # print('ket tens SHAPE', self.out[left_site_pos].shape)
        # print('site time evol', site_tens)
        out, err = self._site_solve(left_site_pos, 1, site_tens=site_tens)
        self.dt = orig_dt
        return out[0], err

    def _bond_time_evolution(self, bond_tens: 'qtn.Tensor', left_site_pos: int, dt: Numeric):
        orig_dt = self.dt
        self.dt = dt
        # print('bond tens SHAPE', bond_tens.shape if bond_tens is not None else None)
        out, err = self._bond_solve(left_site_pos, site_tens=bond_tens)
        self.dt = orig_dt
        return out[0], err

    # def _site_solve(self, left_site_pos: int, nsites: int, site_tens: qtn.Tensor = None
    #                 ) -> tuple[Sequence[qtn.Tensor], Numeric]:
    #     """
    #     sites: int or slice(start, stop, step)
    #     """
    #     site_inds = list(range(left_site_pos, left_site_pos + nsites))
    #
    #     direction = self.direction
    #     at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)
    #     print('TDVP SITE SOLVE', left_site_pos, at_end)
    #
    #     if at_end:
    #         self._set_local_solve_func(self.te_order_final)
    #     else:
    #         self._set_local_solve_func(self.te_order_target)
    #
    #     if nsites == 1:
    #         ix = left_site_pos
    #         ix2 = ix + 1 if direction == SweepDirection.RIGHT else ix - 1
    #     elif nsites == 2:
    #         if direction == SweepDirection.RIGHT:
    #             ix, ix2 = left_site_pos, left_site_pos + 1
    #         else:
    #             ix2, ix = left_site_pos, left_site_pos + 1
    #     else:
    #         raise ValueError
    #
    #     if site_tens is None:
    #         current_x = qtn.tensor_contract(*[self.out[i] for i in site_inds])
    #     else:
    #         current_x = site_tens.copy()
    #
    #     # tot_rdm = None
    #     # tot_site = None
    #     eval_tens_list = []
    #     target_rdm_list = []
    #
    #     func = self._local_solve_func
    #     print('func', func)
    #     print('site tens', site_tens)
    #     if func is not None:
    #         # if not at_end:
    #         #     tot_site, target_rdm_list = func(left_site_pos, nsites, return_intermediates=False)
    #         # else:
    #         #     tot_site = func(left_site_pos, nsites, return_intermediates=False)
    #         tot_site = func(left_site_pos, nsites, return_intermediates=False, site_tens=site_tens)
    #         eval_tens_list = [tot_site]
    #     else:
    #         raise NotImplementedError
    #         # for term in self.terms:
    #         #     site_tens = term.get_evaluated_site(left_site_pos, nsites)
    #         #     site_tens.modify(apply=lambda x: x * 10 ** term.bra.exponent)  ## include term.bra exponent
    #         #     eval_tens_list += [site_tens]
    #         #
    #         #     target_list = [t.copy() for t in term.proj_vec_targets]
    #         #     target_rdm_list += target_list
    #         #     if self.ket is term.bra:
    #         #         ## to normalize Ax, x so that they have similar weights, but not normalize across different terms
    #         #         site_norm = site_tens.norm()
    #         #         for t in target_list:
    #         #             t.modify(apply=lambda x: x * site_norm / t.norm())
    #         #         target_rdm_list += [site_tens]
    #         #         ## so self.ket/term.bra captures both evaluated site and targets for term.ket
    #         #
    #         # # tot_site = helper_tn.sum_tens(site_tens_list)
    #         # tot_site = self.combine_terms_func(eval_tens_list)
    #         # tot_site.modify(apply=lambda x: x * 10 ** (-self.ket.exponent))  ## exclude ket exponent
    #
    #     current_x.transpose_like(tot_site, inplace=True)
    #     try:
    #         site_err = np.linalg.norm(tot_site.data - current_x.data) / np.linalg.norm(current_x.data)
    #     except ValueError:  ## shape mismatch
    #         site_err = np.nan
    #         # print('current x', current_x)
    #         # print('tot site', tot_site)
    #         # print('self.ket', self.ket)
    #         # print('self.', self.terms[0].bra is self.ket)
    #         # print('self.bra', self.terms[0].bra)
    #
    #     self._new_ket_site = (left_site_pos, nsites, direction, eval_tens_list)
    #
    #     return tot_site, site_err


    def _site_solve(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor' = None, return_intermediates=False
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:

        if self.verbose > 2:
            print("TDVP SITE SOLVE", left_site_pos, nsites, self.te_order_target)

        self._set_local_solve_func(self.te_order_target)
        # print('self.dt', self.dt)
        # # pdb.set_trace()
        # if self.dt > 0:
        #     self._set_local_solve_func(self.te_order_target)
        # else:
        #     print('self.dt', self.dt)
        #     self._set_local_solve_func(self.te_order_target)  # (TimeIntegMethod.RK4)
        #     print('local solve', self._local_solve_func)
        #     # pdb.set_trace()

        # self._set_local_solve_func(TimeIntegMethod.RK4)
        # if self.te_order_target in [223, 226, 0]:
        #     self._set_local_solve_func(self.te_order_target)
        # else:
        #     ## force RK4 (pass if want to use self.te_order_target or self.te_order_final
        #     self._set_local_solve_func(TimeIntegMethod.RK4)
        #     # print('self.te_order target', self.te_order_target)
        #     # self._set_local_solve_func(self.te_order_target)
        out, err = super()._site_solve(left_site_pos, nsites, site_tens=site_tens, return_intermediates=False)
        return out, err


    def _bond_solve(self, left_site_pos: int, site_tens: 'qtn.Tensor' = None, return_intermediates=False
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:

        if self.verbose > 2:
            print("TDVP BOND SOLVE", left_site_pos, self.te_order_target)

        self._set_local_solve_func(self.te_order_target)
        # if self.dt > 0:
        #     self._set_local_solve_func(self.te_order_target)
        # else:
        #     print('bond self.dt', self.dt)
        #     self._set_local_solve_func(self.te_order_target)  # (TimeIntegMethod.RK4)
        #     print('local solve', self._local_solve_func)
        #     # pdb.set_trace()

        # self._set_local_solve_func(self.te_order_target)
        # self._set_local_solve_func(TimeIntegMethod.RK4)
        # if self.te_order_target in [223, 226, 0]:
        #     self._set_local_solve_func(self.te_order_target)
        # else:
        #     self._set_local_solve_func(TimeIntegMethod.RK4)
        #     # self._set_local_solve_func(self.te_order_target)
        out, err = super()._bond_solve(left_site_pos, site_tens=site_tens,
                                       return_intermediates=False)
        return out, err

    def _update_1site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection', filter_bases=False,
                      grid=None, ax_deriv_configs=None):
        """ update ket, bra with new_site
            i: int of mps site
            canonicalize and then back-propagate "bond" (if not at end)
        """
        if self.verbose > 2:
            print('new TDVP update 1 site', i, direction)

        at_end = (i == 0 if direction == SweepDirection.LEFT else i == self.L - 1)
        if at_end:
            super()._update_1site(i, site_i, direction)
            return

        if isinstance(site_i, (tuple, list)):
            site_i = helper_tn.sum_tens(site_i)     ## all other sites are the same (and in canonical form)

        x_ind = self.out.bond(i, i + direction)
        left_inds = [ind for ind in self.out[i].inds if ind != x_ind]
        new_Q, new_R = qtn.tensor_split(site_i, left_inds, absorb='right',
                                        # max_bond=self.max_bond, cutoff=CUTOFF,
                                        bond_ind=x_ind+'_tmp', method='qr')

        if direction == SweepDirection.LEFT:
            bond_reindex_dict = {x_ind: x_ind + '_L', x_ind + '_tmp': x_ind + '_R'}
        else:  # direction is to the right
            bond_reindex_dict = {x_ind: x_ind + '_R', x_ind + '_tmp': x_ind + '_L'}
        new_R.reindex(bond_reindex_dict, inplace=True)

        # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
        new_Q.transpose_like(self.out[i], inplace=True)
        self.out[i].modify(data=new_Q.data)

        for term in self.terms:
            if term is not None:
                term.update_intermediate_kets(i, 1, direction)


        if not at_end:
            self.update_blocks(i, direction=direction)

            ## back-propagation of R ##
            # print('back propagation of R', i, direction, new_R)
            # print('self.cur_orthog', self.cur_orthog)
            self.time = self.time + self.dt
            back_i = i if direction == SweepDirection.RIGHT else i - 1
            new_R, err = self._bond_time_evolution(new_R, back_i, -self.dt)     ## bond between bond j, j + 1
            new_R.reindex({v: k for k,v in bond_reindex_dict.items()},inplace=True)
            next_site = qtn.tensor_contract(new_R, self.out[i + direction])
            next_site.transpose_like(self.out[i + direction], inplace=True)
            self.out[i + direction].modify(data=next_site.data)
            self.out._cur_orthog = i + direction
            self.time = self.time - self.dt

            # self.out[i].modify(data=self.init_ket[i].data, inds=self.init_ket[i].inds)
            # self.out[i + direction].modify(data=self.init_ket[i + direction].data,
            #                                inds=self.init_ket[i + direction].inds)
            # self.out._cur_orthog = i + direction

        return


    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        if self.verbose > 2:
            print('new TDVP update 2 site', i, direction)

        ## canonicalize and then back-propagate "site" (if not at end)
        at_end = (i == 1 if direction == SweepDirection.LEFT else i == self.L - 2)

        if isinstance(site_i, (tuple, list)):
            site_i = helper_tn.sum_tens(site_i)

        x_ind = self.out.bond(i, i + direction)
        left_inds = [ind for ind in self.out[i].inds if ind != x_ind]
        # q, r = qtn.tensor_split(site_i, left_inds, absorb='right', max_bond=self.max_bond,
        #                         cutoff=(CUTOFF if self.cutoff is None else self.cutoff),
        #                         bond_ind=x_ind)
        q, r = helper_dmrg.tensor_svd(site_i, left_inds, absorb='right', max_bond=self.max_bond,
                                      cutoff=(CUTOFF if self.cutoff is None else self.cutoff),
                                      bond_ind=x_ind)

        # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
        q.transpose_like(self.out[i], inplace=True)
        self.out[i].modify(data=q.data)

        ## extend environments to include newly canonical site i
        # for term in self.terms:
        #     term.update_vecblock_bra(i, 1, direction)

        for term in self.terms:
            if term is not None:
                term.update_intermediate_kets(i, 2, direction)

        if not at_end:
            self.update_blocks(i, direction=direction)

        if not at_end:
            ## back-propagation of R
            # r.transpose_like(self.ket[i + direction], inplace=True)
            # self.ket[i + direction].modify(data=r.data)
            self.time = self.time + self.dt
            next_site, err = self._site_time_evolution(r, i + direction, -self.dt)
            self.time = self.time - self.dt
            # next_site = qtn.tensor_contract(new_q, self.ket[i + direction])
        else:
            next_site = r
        next_site.transpose_like(self.out[i + direction], inplace=True)
        self.out[i + direction].modify(data=next_site.data)
        self.out._cur_orthog = i + direction

        # plt.figure()
        # out_gtn = self.grid.make_gridTN(self.out)
        # # plt.plot(out_gtn.get_data(), label='Q')
        # plt.plot(out_gtn.get_data(), label='R')
        # plt.legend()
        # plt.show()

        # self.out[i].modify(data=self.init_ket[i].data, inds=self.init_ket[i].inds)
        # self.out[i + direction].modify(data=self.init_ket[i + direction].data,
        #                                inds=self.init_ket[i + direction].inds)
        # self.out._cur_orthog = i + direction

        return


class TDVP_Krylov_DMRG(TDVP_DMRG, DMRGEvaluator):
    """ performs TDVP but with projection based back-propagation
    """
    _ovlp_projector = None

    # def _site_time_evolution(self, site_tens: 'qtn.Tensor', left_site_pos: int, direction: SweepDirection, dt: Numeric):
    #     self.dt = -self.dt
    #     out = super()._site_solve(left_site_pos, 1, site_tens=site_tens)
    #     self.dt = -self.dt
    #     return out
    #
    # def _bond_time_evolution(self, bond_tens: 'qtn.Tensor', left_site_pos: int, direction: SweepDirection, dt: Numeric):
    #     self.dt = -self.dt
    #     out = super()._bond_solve(left_site_pos, site_tens=bond_tens)
    #     self.dt = - self.dt
    #     return out
    #
    # def _bond_solve(self, left_site_pos: int, site_tens: 'qtn.Tensor' = None
    #                 ) -> tuple[qtn.Tensor, Numeric]:
    #
    #     self._set_local_solve_func(TimeIntegMethod.RK4)
    #     out = super()._bond_solve(left_site_pos, site_tens=site_tens)
    #     return out

    def _update_1site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection', filter_bases=False,
                      grid=None, ax_deriv_configs=None):
        """ update ket, bra with new_site
            i: int of mps site
            canonicalize and then back-propagate "bond" (if not at end)
        """
        if self.verbose > 1:
            print('Krylov TDDMRG UPDATE 1 site', i, self.L, direction)

        at_end = (i == 0 if direction == SweepDirection.LEFT else i == self.L - 1)
        if at_end:
            super()._update_1site(i, site_i, direction)
            return

        if isinstance(site_i, (tuple, list)):
            # assert(len(site_i) == 1)
            site_i = helper_tn.sum_tens(site_i)

        x_ind = self.init_ket.bond(i, i + direction)
        left_inds = [ind for ind in self.init_ket[i].inds if ind != x_ind]
        b2k = self.terms[0].projected_bra_to_ket(i,1)
        # k2b = {v: k for k,v in b2k.items()}

        tens_i = self.init_ket[i].copy()
        # old_Q, old_R = qtn.tensor_split(self.ket[i], left_inds, absorb='right',
        #                                 bond_ind=f'p_{x_ind}', cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)

        new_Q, _ = qtn.tensor_split(site_i, left_inds, absorb='right', max_bond=self.max_bond, cutoff=self.cutoff,
                                    bond_ind=f'p_{x_ind}')

        # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
        new_Q.transpose_like(self.init_ket[i], inplace=True)
        self.init_ket[i].modify(data=new_Q.data)

        # new_Q_conj = new_Q.reindex({f'p_{x_ind}': f'p_{x_ind}_'}).conj()
        # ovlp_projector = qtn.tensor_contract(new_Q_conj, old_Q)
        # new_R = qtn.tensor_contract(ovlp_projector, old_R)

        ## old site projected onto new Q
        new_Q_conj = new_Q.conj()
        new_R = qtn.tensor_contract( new_Q_conj, tens_i )

        new_M = qtn.tensor_contract(new_R, self.init_ket[i + direction])
        current_M = self.init_ket[i + direction]
        new_M.transpose_like(current_M, inplace=True)
        current_M.modify(data=new_M.data)
        self.init_ket.cur_orthog = i + direction

        self.out[i].modify(data=self.init_ket[i].data, inds=self.init_ket[i].inds)
        self.out[i + direction].modify(data=self.init_ket[i + direction].data,
                                       inds=self.init_ket[i + direction].inds)
        self.out.cur_orthog = i + direction

        if not at_end:
            self.update_blocks(i, direction=direction)

        return


    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        if self.verbose > 1:
            print('Kyrlov TDDMRG UPDATE 2 site', i, self.L, direction)

        ## canonicalize and then back-propagate "site" (if not at end)
        at_end = (i == 1 if direction == SweepDirection.LEFT else i == self.L - 2)
        # if at_end:        ## causes issues with update
        #     super()._update_2site(i, site_i, direction)
        #     return

        if isinstance(site_i, (tuple, list)):
            site_i = helper_tn.sum_tens(site_i)

        x_ind = self.init_ket.bond(i, i + direction)
        left_inds = [ind for ind in self.init_ket[i].inds if ind != x_ind]
        # b2k = self.linear_terms[0].projected_bra_to_ket(i, 1)
        # k2b = {v: k for k, v in b2k.items()}
        # bra_left_inds = [k2b[k_ind] for k_ind in left_inds]

        # old_Q, old_R = qtn.tensor_split(self.ket[i], left_inds, absorb='right',
        #                                 bond_ind=f'p_{x_ind}', cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
        # old_M = qtn.tensor_contract(old_R, self.ket[i + direction])
        old_M0 = self.init_ket[i].copy()
        old_M1 = self.init_ket[i + direction].copy()

        new_Q, new_M = qtn.tensor_split(site_i, left_inds, absorb='right', max_bond=self.max_bond, cutoff=self.cutoff,
                                        bond_ind=f'p_{x_ind}')

        # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
        new_Q.transpose_like(self.init_ket[i], inplace=True)
        self.init_ket[i].modify(data=new_Q.data)

        if not at_end:
            ## old site projected onto new Q

            # new_Q_conj = new_Q.reindex({f'p_{x_ind}': f'p_{x_ind}_'}).conj()
            # ovlp_projector = qtn.tensor_contract(old_M0, new_Q_conj)
            # new_M1 = qtn.tensor_contract(ovlp_projector, old_M1)

            new_Q_conj = new_Q.conj()
            new_M1 = qtn.tensor_contract(new_Q_conj, old_M0, old_M1)
        else:
            new_M1 = new_M

        new_M1.transpose_like(old_M1, inplace=True)
        self.init_ket[i + direction].modify(data=new_M1.data)
        self.init_ket.cur_orthog = i + direction

        ## i think this should just be the same as computing
        ## new_M1 = qtn.tensor_contract(self.ket[i], self.ket[i+1], new_Q_conj)

        ## extend environments to include newly canonical site i
        # for term in self.terms:
        #     term.update_vecblock_bra(i, 1, direction)

        self.out[i].modify(data=self.init_ket[i].data, inds=self.init_ket[i].inds)
        self.out[i + direction].modify(data=self.init_ket[i + direction].data,
                                       inds=self.init_ket[i + direction].inds)
        self.out.cur_orthog = i + direction

        if not at_end:
            self.update_blocks(i, direction=direction)

        return


def _mpo_firstderivative_center(L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                bc_offset=0, bc_offset_r=None, compress_opts=None):
    """ S+|x> = |x+1> , S-|x> = |x-1>
        df/dx = \sum_i (S- - S+)|x_i>a
        is for a 1D system so dim does not need to be specified
        scale by 1/dt later
        boundary_condition:  boundary condition to use when taking derivative

        FD coeffs found using python package FinDiff
    """

    if q != 2:
        raise NotImplementedError('check q=2 implementation for binary mapping, esp if not pbc')

    if order == 1:
        c1, c2, c3, c4 = (1. / 2, 0, 0, 0)
    elif order == 2:
        c1, c2, c3, c4 = (2. / 3, -1. / 12, 0, 0)
    elif order == 3:
        c1, c2, c3, c4 = (3. / 4, -3. / 20, 1. / 60, 0)
    elif order == 4:
        c1, c2, c3, c4 = (4. / 5, -1. / 5, 4. / 105, -1. / 280)
    else:
        raise NotImplementedError

    coeffs = np.array([0, c1, c2, c3, c4])

    #### build n-ary +/- operator ####
    ## define operator such that operating on the desired spatial dimension
    sp = np.diag([1, ] * (q - 1), k=-1)
    sm = np.diag([1, ] * (q - 1), k=1)

    ## build MPO
    iden = np.eye(q)
    zero = np.zeros((q, q))

    # print('center', left_bc, right_bc)

    if left_bc == BCType.PERIODIC:
        mpo_tens = [np.array([iden, sm + sp, sp + sm])]  ## ignore dummy size 1 bond
        for i in range(1, L - 2):
            mpo_tens += [np.array([[iden, sm, sp],
                                   [zero, sp, zero],
                                   [zero, zero, sm]])]
        mpo_tens += [np.array([[iden, sm, sp, zero, zero],
                               [zero, sp, zero, iden, zero],
                               [zero, zero, sm, zero, iden]])]
        mpo_tens += [np.array([c1 * (sm - sp), (c1 * sp + c2 * iden + c3 * sm), -(c1 * sm + c2 * iden + c3 * sp),
                               (c4 * iden + c3 * sp), -(c4 * iden + c3 * sm)])]
        ## ignore dummy size 1 bond
        mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                        upper_ind_id='o({})', lower_ind_id='i({})')

    elif left_bc == BCType.ANTIPERIODIC:
        mpo_tens = [np.array([iden, sm - sp, sp - sm])]  ## ignore dummy size 1 bond
        for i in range(1, L - 2):
            mpo_tens += [np.array([[iden, sm, sp],
                                   [zero, sp, zero],
                                   [zero, zero, sm]])]
        mpo_tens += [np.array([[iden, sm, sp, zero, zero],
                               [zero, sp, zero, iden, zero],
                               [zero, zero, sm, zero, iden]])]
        mpo_tens += [np.array([c1 * (sm - sp), (c1 * sp + c2 * iden + c3 * sm), -(c1 * sm + c2 * iden + c3 * sp),
                               (c4 * iden + c3 * sp), -(c4 * iden + c3 * sm)])]
        ## ignore dummy size 1 bond
        mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                        upper_ind_id='o({})', lower_ind_id='i({})')

    else:
        m0 = np.diag([1.] + [0.] * (q - 1))
        m1 = np.diag([0.] * (q - 1) + [1.])

        nb = 4  # int(np.ceil(np.log2(order*2+1)))   ## number of bits to encode boundary condition
        bc0 = np.zeros((q ** nb, q ** nb))
        bc1 = np.zeros((q ** nb, q ** nb))

        import findiff

        def get_bulk_boundary_stencil():
            """ function that returns bulk coefficientss at the boundaries
            """
            ## edge boundary conditions: use the same number of points in stencil
            if order == 1:  # 3-pt stencil
                fd_mat_0 = np.array([[0, c1, c2]])
                # fd_mat_L = fd_mat_0[:, ::-1] * -1

            elif order == 2:  # 5-pt stencil
                ## 000.. points
                v0 = np.array([0., c1, c2, c3, c4])
                v1 = np.array([-c1, 0., c1, c2, c3])
                fd_mat_0 = np.array([v0, v1])

                ## 111.. points
                # fd_mat_L = np.array([v1[::-1], v0[::-1]]) * -1

            elif order == 3:  # 7-pt stencil
                v0 = np.array([0., c1, c2, c3, c4, 0., 0.])
                v1 = np.array([-c1, 0., c1, c2, c3, c4, 0.])
                v2 = np.array([-c2, -c1, 0., c1, c2, c3, c4])
                fd_mat_0 = np.array([v0, v1, v2])
                # fd_mat_L = np.array([v2[::-1], v1[::-1], v0[::-1]]) * -1

            elif order == 4:  # order == 4
                v0 = np.array([0., c1, c2, c3, c4, 0., 0., 0., 0.])
                v1 = np.array([-c1, 0., c1, c2, c3, c4, 0., 0., 0.])
                v2 = np.array([-c2, -c1, 0., c1, c2, c3, c4, 0., 0.])
                v3 = np.array([-c3, -c2, -c1, 0., c1, c2, c3, c4, 0.])
                fd_mat_0 = np.array([v0, v1, v2, v3])
                # fd_mat_L = np.array([v3[::-1], v2[::-1], v1[::-1], v0[::-1]]) * -1

            else:
                raise ValueError(f'order must be 1-4, not {order}')

            fd_mat_L = fd_mat_0[::-1, ::-1] * -1

            return fd_mat_0, fd_mat_L

        def get_gen_boundary_stencil(num_stencil=2 * order + 1, offset=0):
            """ function that returns open boundary condition stencils in matrix form
            """
            coeffs = []
            for x0 in range(offset, offset + order):
                stencil = findiff.coefficients(1, offsets=list(range(-x0, num_stencil - x0)))
                coeffs += [stencil['coefficients']]
                # print('stencil', stencil)

            end_bc_mat_0 = np.array(coeffs)
            end_bc_mat_L = end_bc_mat_0[::-1, ::-1] * -1
            return end_bc_mat_0, end_bc_mat_L

        left_bulk, right_bulk = get_bulk_boundary_stencil()
        # print('left bulk', left_bulk)
        # print('right bulk', right_bulk)

        ## left-hand side boundaries
        if left_bc == BCType.SYMMETRIC or left_bc == BCType.ANTISYMMETRIC \
                or left_bc == BCType.ZEROGRADIENT or left_bc == BCType.ZEROVALUE \
                or left_bc == BCType.ABSORBING or left_bc == BCType.REFLECTING:
            ## add np.array([[c1, c2, c3, c4], [c2, c3, c4, 0.], [c3, c4, 0., 0.], [c4, 0., 0., 0.]])
            ## to bulk matrix

            ### bc_offset == 0 if zero is included at the boundary
            ### bc_offset > 0 means y=0 not included (max = 1; half-step offset)
            ### bc_offset < 0 means y=0 is included.

            num_boundary = order + max(0, bc_offset)
            end_bc_mat_0 = np.zeros((num_boundary, 2 * order + max(0, bc_offset)))

            for i in range(num_boundary):
                nc = order - i
                if bc_offset == 1:
                    # reflected from symmetry axis
                    end_bc_mat_0[i, bc_offset:bc_offset + nc + 1] -= coeffs[i:order + 1] * np.sign(left_bc.value)
                    # remove stuff past symmetry axis
                    num_bound = min(bc_offset, i)
                    end_bc_mat_0[i, :num_bound] += coeffs[num_bound + i - 1:i - 1:-1]
                elif bc_offset == 0:
                    # reflected from symmetry axis
                    end_bc_mat_0[i, 1:nc + 1] -= coeffs[1 + i:order + 1] * np.sign(left_bc.value)
                elif bc_offset == -1:
                    # reflected from symmetry axis
                    end_bc_mat_0[i, 0:nc] -= coeffs[i - bc_offset:order + 1] * np.sign(left_bc.value)
                else:
                    raise NotImplementedError

        elif left_bc == BCType.OPEN:
            end_bc_mat_0 = -left_bulk
            end_bc_mat_0 += get_gen_boundary_stencil(offset=0)[0]

        else:
            raise ValueError(f'{left_bc} not a valid BCType')

        ## right-hand side boundaries
        if right_bc == BCType.SYMMETRIC or right_bc == BCType.ANTISYMMETRIC \
                or right_bc == BCType.ZEROGRADIENT or right_bc == BCType.ZEROVALUE \
                or right_bc == BCType.REFLECTING or right_bc == BCType.ABSORBING:
            ## build matrix out of coefficients, which will be added to bulk matrix

            ### bc_offset == 0 if zero is included at the boundary
            ### bc_offset < 0 means y=0 not included (min = -1; half-step offset)
            ### bc_offset > 0 means y=0 is included.

            bc_offset_r = bc_offset if bc_offset_r is None else bc_offset_r
            offset = max(0, -bc_offset_r)
            num_boundary = order + offset

            end_bc_mat_1 = np.zeros((num_boundary, 2 * order + offset))
            # bc_ind = bc_offset + 1

            for i in range(num_boundary):
                nc = order - i

                if bc_offset_r == -1:
                    # reflected from symmetry axis
                    end_bc_mat_1[-1 - i, -nc - 1 + bc_offset_r:bc_offset_r] += \
                        coeffs[i:order + 1][::-1] * np.sign(right_bc.value)

                    # remove stuff past symmetry axis
                    num_bound = min(-bc_offset_r, i)
                    if num_bound > 0:
                        end_bc_mat_1[-i - 1, -num_bound:] -= coeffs[i:num_bound + i]

                elif bc_offset_r == 0:
                    # reflected from symmetry axis
                    end_bc_mat_1[-1 - i, 2 * order - nc - 1:2 * order - 1] += \
                        coeffs[i + bc_offset_r + 1:order + bc_offset_r + 1][::-1] * np.sign(right_bc.value)

                elif bc_offset_r == 1:
                    # reflected from symmetry axis
                    end_bc_mat_1[-1 - i, -nc:] += \
                        coeffs[i + bc_offset_r:order + bc_offset_r][::-1] * np.sign(right_bc.value)
                else:
                    raise NotImplementedError

        elif right_bc == BCType.OPEN:
            end_bc_mat_1 = -right_bulk
            end_bc_mat_1 += get_gen_boundary_stencil(offset=0)[1]

        else:
            raise ValueError(f'{right_bc} not a valid BCType')

        nx, ny = end_bc_mat_0.shape
        bc0[:nx, :ny] = end_bc_mat_0
        nx, ny = end_bc_mat_1.shape
        bc1[-nx:, -ny:] = end_bc_mat_1

        ## 0 side
        bc0 = np.reshape(bc0, (q,) * nb * 2)
        bc0_tens = qtn.Tensor(bc0, inds=tuple([f'o({i})' for i in range(nb)] + \
                                              [f'i({i})' for i in range(nb)]))
        bc0_mpo = helper_quimb.mpx_from_dense(bc0_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')

        ## L-1 side
        bc1 = np.reshape(bc1, (q,) * nb * 2)
        bc1_tens = qtn.Tensor(bc1, inds=tuple([f'o({i})' for i in range(nb)] + \
                                              [f'i({i})' for i in range(nb)]))
        bc1_mpo = helper_quimb.mpx_from_dense(bc1_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')

        ## leading 00... or 1... strings
        if L - nb >= 2:
            mpo_0 = qtn.MatrixProductOperator(
                [np.array([m0])] + [np.array([[m0]])] * (L - nb - 2) + [np.array([m0])],
                shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
            mpo_1 = qtn.MatrixProductOperator(
                [np.array([m1])] + [np.array([[m1]])] * (L - nb - 2) + [np.array([m1])],
                shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
        elif L - nb == 1:
            mpo_0 = qtn.TensorNetwork([qtn.Tensor(m0, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
            mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                          site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
            mpo_1 = qtn.TensorNetwork([qtn.Tensor(m1, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
            mpo_1.view_as(qtn.MatrixProductOperator, like=mpo_0, inplace=True)
        elif L - nb == 0:
            mpo_0 = qtn.TensorNetwork([])
            mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                          site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
            mpo_1 = qtn.TensorNetwork([])
            mpo_1.view_as(qtn.MatrixProductOperator, like=mpo_0, inplace=True)
        else:
            raise ValueError('FD order too large for current grid size')

        helper_quimb.append_mpx(mpo_0, bc0_mpo, inplace=True)
        helper_quimb.append_mpx(mpo_1, bc1_mpo, inplace=True)

        ### the bulk mpo
        mpo_tens = [np.array([iden, sm, sp])]  ## ignore dummy size 1 bond
        for i in range(1, L - 2):
            mpo_tens += [np.array([[iden, sm, sp],
                                   [zero, sp, zero],
                                   [zero, zero, sm]])]
        mpo_tens += [np.array([[iden, sm, sp, zero, zero],
                               [zero, sp, zero, iden, zero],
                               [zero, zero, sm, zero, iden]])]
        mpo_tens += [np.array([c1 * (sm - sp), (c1 * sp + c2 * iden + c3 * sm), -(c1 * sm + c2 * iden + c3 * sp),
                               (c4 * iden + c3 * sp), -(c4 * iden + c3 * sm)])]  ## ignore size 1 bond

        mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                        upper_ind_id='o({})', lower_ind_id='i({})')
        helper_quimb.add_MPO(mpo, mpo_0, inplace=True)
        helper_quimb.add_MPO(mpo, mpo_1, inplace=True)

        helper_quimb.compress(mpo, compress_opts=compress_opts)

    # if bc_offset != 0:
    #     print('center diff')
    #     op_tens = mpo.contract(all) * 10 ** mpo.exponent
    #     op_tens.transpose(*[mpo.upper_ind_id.format(i) for i in range(mpo.L)],
    #                       *[mpo.lower_ind_id.format(i) for i in range(mpo.L)],
    #                       inplace=True)
    #     op_mat = op_tens.data.reshape(2 ** mpo.L, 2 ** mpo.L)
    #     print('op mat', left_bc, right_bc, bc_offset, bc_offset_r)
    #     print('L', op_mat[:5, :5])
    #     print('R', op_mat[-5:, -5:])

    return mpo


def build_firstderivative_mpo_k(L, **kwargs):
    vmax, vmin = 12, -12
    dve = (vmax - vmin) / (2 ** L)
    ks = np.linspace(-np.pi/dve, np.pi/dve, 2**L, endpoint=False)
    ks = qtn.Tensor(ks.reshape((2,)*L), inds=tuple([f'i{i}' for i in range(L)]))
    mps_ks = helper_quimb.mpx_from_dense(ks, L, ['i{}'])
    helper_quimb.scalar_multiply(mps_ks,-1.j,inplace=True)
    mpo = helper_quimb.mps_to_diag_mpo(mps_ks)
    return mpo


# def get_select_elem_mpo(L: int, q: int, sel_ind: int, upper_ind_id: str = 'o({})',
#                         lower_ind_id: str = 'i({})',
#                         site_tag_id: str = 'X({})') -> MPOType:
#     """ build MPO to select certain elements specified by inds
#     """
#
#     def get_position_inds(idx):
#         """ returns physical bond indices (0,1) of MPS that corresponds to vector position idx
#         """
#         if L == 1:
#             return [idx]
#
#         assert (q==2), 'atm method only implemented for q=2'
#
#         if idx < 0:
#             idx = q**L + idx
#
#         if q==2:   str_b = bin(idx)[2:]
#         else:
#             if idx == 0:  str_b = [0]
#             else:
#                 str_b = []
#                 while idx:
#                     str_b.append(int(idx % q))
#                     idx //= q
#                 str_b = str_b[::-1]
#
#         if len(str_b) > L:
#             print('ind not in binaryTN of len L',L)
#             raise ValueError
#         elif len(str_b) < L:
#             str_b = '0'*(L-len(str_b)) + str_b
#
#         ind_list = [int(s) for s in str_b]
#         return ind_list
#
#     inds_list = get_position_inds(sel_ind)
#
#
#     mdict = {}
#     for b in range(q):
#         mvec = np.zeros((q,))
#         mvec[b] = 1.
#         mdict[b] = np.diag(mvec)
#
#     assert (len(inds_list) == L), 'inds_list needs to be desired index in q-nary'
#
#     mpo_0 = None
#     if L >= 2:
#         ms = [mdict[b] for b in inds_list]
#         mpo_0 = qtn.MatrixProductOperator(
#             [np.array([ms[0]])] + [np.array([[m]]) for m in ms[1:-1]] + [np.array([ms[-1]])],
#             shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
#     elif L == 1:
#         m = mdict[inds_list[0]]
#         mpo_0 = qtn.TensorNetwork([qtn.Tensor(m, inds=(upper_ind_id.format(0), lower_ind_id.format(0)),
#                                               tags=(site_tag_id.format(0),))])
#         mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
#                       site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
#     elif L == 0:
#         mpo_0 = qtn.TensorNetwork([])
#         mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
#                       site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
#     return mpo_0
