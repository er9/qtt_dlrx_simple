"""Mixed-projection time integrator for the local-solver stack.

Defines :class:`TDMixed`, which combines :class:`TimeIntegrator` with
:class:`MixedEvaluator` to advance a matrix-product state in time using the
mixed (blended Galerkin/cross) local solve. As with the cross integrator the
working ket and output share the same state, updated only at the final stage
of each time step.
"""
import pdb

import helper_quimb
# import sample_helper_quimb
from setup_.defaults import *
import local_solvers.helper_tn as helper_tn

from local_solvers.defaults import *
import local_solvers.helper_cross_2 as helper_cross
from local_solvers.mps_classes import MPS
from local_solvers.terms_3 import Term, Term_Cross
import helper_TE
from local_solvers.local_mixed_eval import local_mixed_evaluator, MixedEvaluator, Term_Mixed

from local_solvers.time_integrator import TimeIntegrator, TDVP_DMRG, TimeIntegMethod
import local_solvers.helper_mixed as helper_mixed

class TDMixed(TimeIntegrator, MixedEvaluator):
    """ differs in that self.out and self.ket are the same; only updated at last time step
        target both update and original ket; but self.ket remains
    """
    def __init__(self,
                 ket_state: Union['MPS', 'qtn.MatrixProductState'],
                 linear_operators: Sequence['qtn.MatrixProductOperator'],
                 sources: Sequence[Union['MPS', 'qtn.MatrixProductState']] = None,
                 nonlinear_terms: Sequence[Union['Term_Cross']] = None,
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

        self.version = flags.get('version', 'X')  # 'G' for Galerkin projector; override via flags['version']
        self.num_evals = 0

        if copy_obj is not None:
            super().__init__(ket_state, None, copy_obj=copy_obj)

        else:

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

            # terms = self.initialize_terms(self.init_ket)

            super(MixedEvaluator, self).__init__(ket_state, None, direction, max_bond=max_bond, cutoff=cutoff,
                                                 conv_tol=conv_tol, max_iter=max_iter,
                                                 max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter,
                                                 copy_obj=copy_obj, grid=grid, ax_deriv_configs=ax_deriv_configs
                                                 )  # , combine_terms_func=combine_terms_func)

            # if nonlinear_terms is not None:
            #     for t in nonlinear_terms:
            #         if t.ket is not self.init_ket:
            #             print('init2 nonlinear ket is not self.ket')
            #             exit()

            self.dt = dt
            self.time = time
            self.verbose = verbose
            self.verbose_plot = verbose_plot
            # self.upwind_func = None
            # self.upwind_deriv_func = None

        # self.soln = None
        # deriv_mps = None
        # for m in self.linear_operators:
        #     tmp = helper_quimb.apply(m, self.init_ket)
        #     if deriv_mps is None:
        #         deriv_mps = tmp
        #     else:
        #         deriv_mps = helper_quimb.add_MPS(tmp, deriv_mps)
        #
        # helper_quimb.scalar_multiply(deriv_mps, dt, inplace=True)
        #
        # self.soln = helper_quimb.add_MPS(self.init_ket, deriv_mps)

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.out = self.ket

    # def _set_local_solve_func(self, te_order: int):
    #     if te_order == TimeIntegMethod.LW:
    #         ## assumes correct upwind_deriv_func is provided
    #         func = self.local_lax_wendroff_so
    #         self._local_solve_func = func
    #     elif te_order == TimeIntegMethod.SL:
    #         self._local_solve_func = self.local_sl
    #     else:
    #         super()._set_local_solve_func(te_order)

    def initialize_terms(self, ket_state: Union['MPS', 'qtn.MatrixProductState'], cur_orthog=None, **kwargs
                         # euler_state: Union['MPS', 'qtn.MatrixProductState'] = None,
                         # linear_operators: Sequence[qtn.MatrixProductOperator],
                         # sources: Sequence[Union['qtn.MatrixProductState', 'MPS']],
                         # nonlinear_terms: Sequence[Union['Term_DMRG', 'Term_Cross']]
                         ):

        num_tiers = 1   # if self.te_order_target in [4, 223, 226] else None
        compress_opts = {'max_bond': self.max_bond, 'cutoff': self.cutoff}

        LocalTerm = self.term_class()
        term_self = LocalTerm(ket_state, **compress_opts) # if euler_state is None else LocalTerm(euler_state)
        self.self_term = term_self

        ## df/dt terms: linear terms, sources, nonlinear terms
        # terms = super()._initialize_terms(ket_state, linear_operators, sources, nonlinear_terms)
        term_linear = []
        if len(self.linear_operators) > 0:
            # term_linear = [LocalTerm(ket_state, operators=[mpo for mpo in self.linear_operators],
            #                          num_tiers=num_tiers)]
            # for mpo in self.linear_operators:
            #     mpo.distribute_exponent()
            term_linear += [LocalTerm(ket_state, operators=[mpo], num_tiers=num_tiers, **compress_opts)
                            for mpo in self.linear_operators]
        self.linear_terms = term_linear

        term_sources = [LocalTerm(source, bra=ket_state, cur_orthog=cur_orthog, **compress_opts) for source in self.sources]
        self.source_terms = term_sources

        # nonlinear_terms = self.nonlinear_terms
        nl_terms = []
        for nl in self.nonlinear_terms:
            nl_term = nl.copy(ket_copy=ket_state, bra_copy=ket_state)
            nl_term.max_bond = self.max_bond
            nl_term.cutoff = self.cutoff
            # ket_mpo = helper_quimb.mps_to_diag_mpo(ket_state.copy())
            # ref2 = helper_quimb.apply_zipup(ket_mpo, ket_state.copy(), compress=True)
            # nl_term.init_intermediate_ket = ref2
            nl_terms += [nl_term]
        self.nonlinear_terms = nl_terms
        nonlinear_terms = self.nonlinear_terms

        ###### extra terms ######
        ### time mpo list ###
        extra_terms_list = []
        if self.time_mpo_list is not None:
            for key, ops_list in self.time_mpo_list.items():
                # self.extra_terms_dict[key] = [LocalTerm(ket_state, operators=[mpo for mpo in ops_list], num_tiers=num_tiers)]
                # for mpo in ops_list:
                #     mpo.distribute_exponent()
                self.extra_terms_dict[key] = [LocalTerm(ket_state, operators=[mpo], num_tiers=num_tiers, **compress_opts)
                                              for mpo in ops_list]
                extra_terms_list += self.extra_terms_dict[key]

        if self.upwind_mpo_list is not None:
            for key, ops_list in self.upwind_mpo_list.items():
                if isinstance(ops_list[0], qtn.MatrixProductOperator):
                    # self.extra_terms_dict[key] = [LocalTerm(ket_state, operators=[mpo for mpo in ops_list], num_tiers=num_tiers)]
                    self.extra_terms_dict[key] = [LocalTerm(ket_state, operators=[mpo], num_tiers=num_tiers, **compress_opts)
                                                  for mpo in ops_list]
                elif isinstance(ops_list[0], qtn.MatrixProductState):
                    self.extra_terms_dict[key] = [LocalTerm(mps, bra=ket_state, **compress_opts) for mps in ops_list]
                else:
                    print('type', type(ops_list[0]))
                    raise TypeError
                extra_terms_list += self.extra_terms_dict[key]

        return [term_self, *term_linear, *term_sources, *nonlinear_terms, *extra_terms_list]


    def deriv_func(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor' = None, time: Numeric = None
                   ) -> 'qtn.Tensor':
        """ compute df/dt = Af + sources + nonlinear terms O[f]
        """
        if self.upwind_deriv_func is not None:
            return self.local_deriv_upwind(left_site_pos, nsites, time=time, site_tens=site_tens,
                                           return_intermediates=False,)
        else:
            return super().deriv_func(left_site_pos, nsites, time=time, site_tens=site_tens)

        # site_tens_list = []
        # ## do something with time with nonlinear term
        # for term in self.terms[1:]:
        #     # print('DERIV FUNC site tens', site_tens.norm() if site_tens is not None else None)
        #     site_tens_list += [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)]
        # # print('deriv func', site_tens_list)
        # # print('DERIV FUNC OUT', [t.norm() for t in site_tens_list])
        # tot_site = helper_tn.sum_tens(site_tens_list)
        # return tot_site


    def euler_func(self, left_site_pos: int, nsites: int, dt: Numeric = None, time: Numeric = None,
                   deriv: 'qtn.Tensor' = None, site_tens: 'qtn.Tensor' = None):
        """ compute df/dt = Af + sources + nonlinear terms
                assumes the first term is the linear term, followed by sources and nonlinear terms
                (though those are effectively the same)
            for the given effective operators
            term0:  Identity * f
            term1:  Af
            to be implemented:
                term2:  nonlinear terms
                term3:  source terms
        """
        if self.verbose > 2:
            print('MIXED EULER FUNC')

        if deriv is None:
            deriv = self.deriv_func(left_site_pos, nsites, time=time, site_tens=site_tens)

        dt = self.dt if dt is None else dt
        # print('EULER DT', dt)
        deriv = deriv.copy()    ## already excludes self.ket.exponent
        deriv.modify(apply=lambda x: x * dt)
        # print('EULER deriv', deriv.norm())
        # print('EULER ket x', ket_x.norm())

        ## original state * I (with the same output indices as self.terms[1].vec_block)
        if site_tens is None:
            ket_x0 = self.self_term.get_evaluated_site(left_site_pos, nsites).copy()
        else:
            ket_x0 = site_tens
        # exit()

        if self.verbose > 1:
            print('euler left site pos add num evals', left_site_pos)
        self.num_evals += ket_x0.size

        ## ket exponent already removed
        # print('EULER ket x', ket_x.norm())
        site_tens_list = [ket_x0, deriv]

        # print(self.terms[0].bra is self.ket)
        # print(self.terms[0].vec_block.bra is self.ket)
        # print('ket_x', ket_x)
        # print('deriv', deriv)

        tot_site = helper_tn.sum_tens(site_tens_list)

        # helper_cross.plot_submat(self.self_term.ket, left_site_pos, 1, tot_site,
        #                          select_inds=self.self_term.bra.select_inds)
        #
        # # deriv_mps = self.init_ket
        # deriv_mps = None
        # for m in self.linear_operators:
        #     tmp = helper_quimb.apply(m, self.init_ket)
        #     if deriv_mps is None:
        #         deriv_mps = tmp
        #     else:
        #         deriv_mps = helper_quimb.add_MPS(tmp, deriv_mps)
        #
        # helper_quimb.scalar_multiply(deriv_mps, dt, inplace=True)
        #
        # helper_cross.plot_submat(self.self_term.ket, left_site_pos, 1, deriv,
        #                          select_inds=self.self_term.bra.select_inds,
        #                          ref_kets=[deriv_mps])

        # print('euler tot site', tot_site.norm())
        tot_site.modify(apply=lambda x: x * 10 ** self.out.exponent)  ## include exponent for input into bra
        return tot_site


    def local_euler(self, left_site_pos: int, nsites: int, return_intermediates=False,
                    dt: Numeric = None, time: Numeric = None, site_tens: 'qtn.Tensor'=None
                    ) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
        if self.verbose > 2:
            print('local EULER', self.time)
        return self.local_rk(1, left_site_pos, nsites, return_intermediates=return_intermediates,
                             dt=dt, time=time, site_tens=site_tens)


    def local_euler_upwind(self, left_site_pos: int, nsites: int, return_intermediates=False,
                    dt: Numeric = None, time: Numeric = None, site_tens: 'qtn.Tensor'=None
                    ) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
        """ euler step but with element-wise operations
        """
        dt = self.dt if dt is None else dt
        # if self.upwind_func is None:
        #     return self.local_euler(left_site_pos, nsites, return_intermediates=return_intermediates,
        #                             dt=dt, time=time ,site_tens=site_tens)

        ### get center of orthogonality and the actual sites
        coords = helper_cross.get_selectors(self.out, left_site_pos, nsites)

        selectors = []
        for c in coords:
            selectors += [int("".join(str(x) for x in c), 2)]
        # selectors = [self.grid.get_axis_positions(c) for c in coords]

        if site_tens is None:
            # ket_x = self.self_term.vec_block.get_projected(left_site_pos, nsites, return_combined=True)
            # ket_x.reindex(self.self_term.vec_block.projected_bra_to_ket(left_site_pos, nsites), inplace=True)
            ket_x = self.self_term.get_evaluated_site(left_site_pos, nsites).copy()
        else:
            ket_x = site_tens

        ket_x = ket_x.copy()
        ket_x.modify(apply=lambda x: x * 10 ** self.out.exponent)    ## incorporate exponent

        inds = []
        if left_site_pos > 0:
            inds += [self.out.bond(left_site_pos, left_site_pos - 1)]
        inds += [self.out.site_ind(left_site_pos + i) for i in range(nsites)]
        if left_site_pos + nsites < self.out.L:
            inds += [self.out.bond(left_site_pos + nsites - 1, left_site_pos + nsites)]

        ket_x = ket_x.transpose(*inds, inplace=True)

        ####
        # deriv = self.deriv_func(left_site_pos, nsites, time=time, site_tens=site_tens)
        # deriv.transpose_like(ket_x, inplace=True)
        # deriv.modify(apply=lambda x: x * 10 ** self.out.exponent)
        ####

        #### get linear operator terms and source terms
        ### allows for decimation of these intermediate kets too
        ### the idea is to incorporate these terms in the targeting
        ### but i guess the initial targeting is what matters the most
        # site_tens_list = []
        # for term in self.linear_terms:
        #     site_tens_list += [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)]
        #
        # for term in self.sources:
        #     site_tens_list += [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens,
        #                                                verbose_plot=self.verbose_plot)]
        ####

        upwind_submats = {}
        for key, op_terms in self.extra_terms_dict.items():
            tens_list = []
            for op_term in op_terms:
                # print('op term exp', [op.exponent for op in op_term.operators], op_term.ket.exponent, op_term.bra.exponent)
                op_tens = op_term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)
                # print('op bra is bra?', op_term.bra is self.out)

                # print(op_term.bra is self.out)
                # helper_cross.plot_submat(self.out, left_site_pos, nsites, op_tens,
                #                          ref_kets=[helper_quimb.apply(op_term.operators[0], op_term.ket)],
                #                          plt_title='op term submat')

                tens_list += [op_tens]
            up_tens = helper_tn.sum_tens(tens_list, transpose_bonds=ket_x.inds)
            up_tens.transpose_like(ket_x, inplace=True)
            up_tens.modify(apply=lambda x: x * 10 ** self.out.exponent)
            upwind_submats[key] = up_tens  # deriv  # up_tens

        if self.verbose > 2:
            print('call upwind func')
        out_x = self.upwind_func(dt, self.init_ket, ket_x, selectors, upwind_submats,
                                 left_site_pos=left_site_pos, nsites=nsites, select_inds=self.out.select_inds)
        out_x.modify(apply=lambda x: x * 10 ** (-self.out.exponent))
        return out_x


    def local_deriv_upwind(self, left_site_pos: int, nsites: int, return_intermediates=False,
                           dt: Numeric = None, time: Numeric = None, site_tens: 'qtn.Tensor'=None
                           ) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
        """ euler step but with element-wise operations
        """
        dt = self.dt if dt is None else dt
        # print('local deriv upwind', time)

        ### get center of orthogonality and the actual sites
        coords = helper_cross.get_selectors(self.out, left_site_pos, nsites)

        # selectors = [self.grid.get_axis_positions(c) for c in coords]
        selectors = []
        for c in coords:
            q = self.out.phys_dim(0)    ## assumes all the same
            selectors += [int("".join(str(x) for x in c), q)]

        # site_tens = None  ## was uncommented... seems ok with burgers tho?
        if site_tens is None:
            if self.verbose > 2:
                print('self', self.self_term)
            ket_x = self.self_term.get_evaluated_site(left_site_pos, nsites).copy()
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
                if self.version == 'G':
                    inds += [self.out.bond(left_site_pos, left_site_pos - 1)]
                elif self.version == 'X':
                    inds += [self.out.bond(left_site_pos, left_site_pos - 1) + '_x']

            inds += [self.out.site_ind(left_site_pos + i) for i in range(nsites)]

            if left_site_pos + nsites < self.out.L:
                if self.version == 'G':
                    inds += [self.out.bond(left_site_pos + nsites - 1, left_site_pos + nsites)]
                elif self.version == 'X':
                    inds += [self.out.bond(left_site_pos + nsites - 1, left_site_pos + nsites) + '_x']

        ket_x = ket_x.transpose(*inds, inplace=True)

        upwind_submats = {}
        for key, op_terms in self.extra_terms_dict.items():
            if len(key) == 3 and time is not None:
                key_time = key[-1]
                if key_time != np.round(time, 10):
                    continue

            tens_list = []
            for op_term in op_terms:
                op_tens = op_term.get_evaluated_site(left_site_pos, nsites,
                                                     site_tens=(ket_x if op_term.ket is self.out else None))
                ## some terms are not MPO blocks; rather just MPS that need to be evaluated at certain grid points

                # print('op bra is bra?', op_term.bra is self.out)

                # print(op_term.bra is self.out)
                # helper_cross.plot_submat(self.out, left_site_pos, nsites, op_tens,
                #                          ref_kets=[helper_quimb.apply(op_term.operators[0], op_term.ket)],
                #                          plt_title='op term submat')

                tens_list += [op_tens]
            up_tens = helper_tn.sum_tens(tens_list, transpose_bonds=ket_x.inds)
            up_tens.transpose_like(ket_x, inplace=True)
            up_tens.modify(apply=lambda x: x * 10 ** self.out.exponent)
            upwind_submats[key] = up_tens  # deriv  # up_tens

        if self.verbose > 2:
            print('call deriv upwind func')

        out_x = self.upwind_deriv_func(dt, self.init_ket, ket_x, selectors, upwind_submats,
                                       left_site_pos=left_site_pos, nsites=nsites, select_inds=self.out.select_inds,
                                       time=time)
        out_x.modify(apply=lambda x: x * 10 ** (-self.out.exponent))
        return out_x

    # def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False, dt: Numeric = None,
    #               time: Numeric = None, site_tens=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
    #
    #     dt = self.dt if dt is None else dt
    #     return super().local_rk4(left_site_pos, nsites, return_intermediates=False, dt=dt, time=time,
    #                              site_tens=site_tens)
    #     # return self.local_euler(left_site_pos, nsites, dt=dt, time=time)


    def local_rk4_upwind(self, left_site_pos: int, nsites: int, return_intermediates=False, dt: Numeric = None,
                  time: Numeric = None, site_tens=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
        raise NotImplementedError


    def local_rk4(self, left_site_pos: int, nsites: int, return_intermediates=False, dt: Numeric = None,
                  time: Numeric = None, site_tens=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        # print('local RK4', self.time)
        dt = self.dt if dt is None else dt


        ## out and psi33 are the same thing, but psi33 is scaled
        if return_intermediates:
            out, rk_states = self.local_rk(4, left_site_pos, nsites, return_intermediates=return_intermediates,
                                           dt=dt, time=time, site_tens=site_tens)

            # print('target all states')   ## assume returning intermediate stages
            # return out, (*rk_states, out)

            s0, k1, k2, k3, k4, out = rk_states      ## initial x, stage 1, stage 2, stage 3

            ## tot_denmat:  1/3 * rho(psi03) + 1/6 * rho(psi13) + 1/6 * rho(psi23) + 1/3 * rho(psi33)
            ## include weights here
            psi03 = s0.copy()
            # psi03.modify(apply=lambda x: x * np.sqrt(1. / 3))
            psi13 = s0.copy()
            psi13.modify(apply=lambda x: x + dt * (1. / 162 * (31 * k1.data + 14 * k2.data + 14 * k3.data - 5 * k4.data)))
            # psi13.modify(apply=lambda x: x * np.sqrt(1. / 6))
            psi23 = s0.copy()
            psi23.modify(apply=lambda x: x + dt * (1. / 81 * (16 * k1.data + 20 * k2.data + 20 * k3.data - 2 * k4.data)))
            # psi23.modify(apply=lambda x: x * np.sqrt(1. / 6))
            psi33 = s0.copy()
            psi33.modify(apply=lambda x: x + dt * (1. / 6 * (k1.data + 2 * k2.data + 2 * k3.data + k4.data)))
            # psi33.modify(apply=lambda x: x * np.sqrt(1. / 3))

            ########## return output and intermediate stages #########
            ## different from DMRG, final output is the first target
            ## bc we're usually working on self.out; init ket is kept untouched.

            if self.verbose:
                print('(X) classic TD-DMRG')
            return out, (psi03, psi13, psi23, psi33)

            # print('only 0, dt target')
            # return out, (out, psi03)
        else:
            out = super().local_rk(4, left_site_pos, nsites, return_intermediates=False, dt=dt, time=time,
                                   site_tens=site_tens)
            return out



    def local_rk(self, te_order: int, left_site_pos: int, nsites: int, return_intermediates=False, dt: Numeric = None,
                  time: Numeric = None, site_tens=None) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        # print('local RK()', te_order, self.time)
        dt = self.dt if dt is None else dt
        if return_intermediates:
            out, rk_states = super().local_rk(te_order, left_site_pos, nsites, return_intermediates=True, dt=dt, time=time,
                                               site_tens=site_tens)

            ########## return output and intermediate stages #########
            ## different from DMRG, final output is the first target
            ## bc we're usually working on self.out; init ket is kept untouched.
            if self.verbose > 1:
                print('target all states')   ## assume returning intermediate stages
            return out, (*rk_states, out)

            # print('only 0, dt target')
            # return out, (out, psi03)
        else:
            if self.verbose > 2:
                print('x2 site tens', site_tens)
            out = super().local_rk(te_order, left_site_pos, nsites, return_intermediates=False, dt=dt, time=time,
                                   site_tens=site_tens)
            return out



    def local_implicit_solve(self, left_site_pos: int, nsites: int, return_intermediates=False,
                             dt: Numeric = None, time: Numeric = None,
                             **solver_kwargs) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:

        dt = self.dt if dt is None else dt
        return super().local_implicit_solve(left_site_pos, nsites, dt=dt, time=time,
                                            return_intermediates=return_intermediates, **solver_kwargs)



    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection', filter_bases=False,
                      grid=None, ax_deriv_configs: dict['Axis','DerivativeConfiguration']=None):
        """ update ket, bra with new_site
            i: int of mps site
        """
        if self.verbose > 1:
            print('TE mixed update 1')
        # exit()
        #
        # helper_cross.plot_submat(self.out, i, 1, self.out[i],
        #                          select_inds=self.out.select_inds,
        #                          ref_kets=[self.init_ket], plt_title='b4 update1')
        #


        at_end = (i == 0) if direction == SweepDirection.LEFT else (i == self.L - 1)

        # ### new version 03/01: target intermediate kets separately
        # print('self.out', self.out)
        # print('update site i', i, at_end, site_i)   ## first site is original (un-time evolved) site
        # print('site i', len(site_i), site_i[0].shape)
        helper_mixed.update_ket(self.out, site_i, i, 1, direction=direction, max_bond=self.max_bond, cutoff=self.cutoff,
                                version=self.version)
        # print('out[i]', i, self.out[i])

        for term in self.terms:
            if term is not None:
                term.update_intermediate_kets(i, 1, direction)

        # # #######################
        # distance = helper_quimb.distance(self.init_ket, self.out)
        # print('distance', helper_quimb.distance(self.init_ket, self.out))
        # if distance > 1.0e-10 and not at_end:
        #     plt.figure()
        #     plt.plot(np.real(helper_quimb.to_dense(self.init_ket).reshape(-1)))
        #     plt.plot(np.real(helper_quimb.to_dense(self.out).reshape(-1)),'--')
        #
        #     plt.figure()
        #     plt.plot(np.real(helper_quimb.to_dense(self.init_ket).reshape(-1)))
        #     plt.plot(np.real(helper_quimb.to_dense(self.out).reshape(-1)),'--')
        #     plt.show()




        if not at_end:

            # for term in self.terms:
            #     term.canonize_ket_tens(i, 1, direction, max_bond=None)

            self.update_blocks(i, direction)

            if self.verbose_plot:
                tmp_gtn = self.grid.make_gridTN(self.out)
                lefts, rights = tmp_gtn.get_bases(i + direction)
                for l in [*lefts[:5], *rights[:5]]:
                    ldata = l.get_data()
                    if ldata.ndim == 1:
                        plt.figure()
                        plt.plot(np.real(ldata))
                        plt.plot(np.imag(ldata))
                    elif ldata.ndim == 2:
                        plt.figure()
                        plt.imshow(np.real(ldata))
                        plt.colorbar()
                        plt.figure()
                        plt.imshow(np.imag(ldata))
                        plt.colorbar()
                    else:
                        raise NotImplementedError
                    plt.title(f'basis fct {i + direction}')
                    plt.show()


        # print('updated ket', i, self.ket.cur_orthog, self.out.cur_orthog )
        # helper_cross.check_orthog(self.ket)
        # helper_cross.check_orthog(self.out)

        ## CHECK ORTHOG
        ind1 = i if at_end else i + direction
        if self.verbose > 1:
            print('site i check orthog', ind1)

        # tmp1, tmp2 = helper_mixed.check_orthog(self.out)
        # if tmp1 != tmp2:
        #     helper_mixed.check_orthog(self.out, verbose=True)
        #     raise ValueError

        #
        # ## also a way to check orthog
        # # print('at end?', at_end)
        # # print('self.term', self.self_term.cur_orthog, ind1)
        # # chk_site = self.self_term.get_evaluated_site(ind1, 1)
        # # orig_site = self.out[ind1]
        # # print('diff proj site', (chk_site + orig_site * -1).norm())


        return

    def _update_2site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        if self.verbose > 1:
            print('TE mixed update 2', i)

        # print('self.out select inds', self.out.select_inds)

        # coords = helper_cross.get_selectors(self.out, left_site_pos, 2)
        # selectors = []
        # for c in coords:
        #     selectors += [int("".join(str(x) for x in c), 2)]
        #
        # inds = []
        # if left_site_pos > 0:
        #     inds += [self.out.bond(left_site_pos, left_site_pos - 1)]
        # inds += [self.out.site_ind(left_site_pos), self.out.site_ind(left_site_pos + 1)]
        # if left_site_pos + 2 < self.out.L:
        #     inds += [self.out.bond(left_site_pos + 1, left_site_pos + 2)]
        #
        # plt.figure()
        # site_i[0].transpose(*inds, inplace=True)
        # print('site i', inds, site_i)
        # plt.plot(selectors, site_i[0].data.reshape(-1), 'x')

        # ### out ket ###
        # coords = helper_cross.get_selectors(self.out, left_site_pos, 2)
        # selectors = []
        # for c in coords:
        #     selectors += [int("".join(str(x) for x in c), 2)]
        #
        # tens = term.evaluated_site
        # print('tens', tens)
        # tens.transpose(*inds, inplace=True)
        # plt.plot(selectors, tens.data.reshape(-1), 'o')
        #
        # ### intermediate ket
        # coords = helper_cross.get_selectors(out_ket, left_site_pos, 2)
        # selectors = []
        # for c in coords:
        #     selectors += [int("".join(str(x) for x in c), 2)]
        #
        # tens = term.intermediate_sites[0]
        # print('tens', tens)
        # tens[0].transpose(*inds, inplace=True)
        # plt.plot(selectors, tens[0].data.reshape(-1), 'x')
        #
        # plt.title('nonlinear term')
        # plt.show()

        # ### new version 03/13: target intermediate kets separately
        helper_mixed.update_ket(self.out, site_i, i, 2, direction=direction, max_bond=self.max_bond, cutoff=self.cutoff,
                                version=self.version)

        for term in self.terms:
            if term is not None:
                term.update_intermediate_kets(i, 2, direction)
            # if term.num_tiers > 1:
            #     for si in range(term.num_tiers - 1):
            #         ket = term.get_intermediate_ket(si)
            #         inter_site = term.intermediate_sites[si]
            #         # sel_inds = self.out.select_inds[i] if direction > 0 else self.out.select_inds[i + 1]
            #         helper_cross.update_1site(ket, i, inter_site, direction, decimate_only=True,
            #                                   select_inds=self.out.select_inds[i],)
            #         ## since it's update with self.out.select_inds, ket should have the same rank as self.out
            #     # kets = [term.get_intermediate_ket(i) for i in range(term.num_tiers - 1)]
            #     # tensors = [term.intermediate_sites[i] for i in range(term.num_tiers - 1)]
            #     # helper_cross.update_kets(kets, tensors, i, 1, direction=direction)

        # #######
        # # print(self.out.exponent, self.init_ket.exponent, [[t_op.exponent for t_op in t.operators] for t in self.terms])
        # kets, tensors = [self.out], [site_i]
        # for term in self.terms:
        #     if term.num_tiers > 1:
        #         kets += [term.get_intermediate_ket(i) for i in range(term.num_tiers - 1)]
        #         tensors += [term.intermediate_sites[i] for i in range(term.num_tiers - 1)]
        #
        # helper_cross.update_kets(kets, tensors, i, 2, direction=direction, max_bond=self.max_bond, cutoff=self.cutoff)

        #### original version
        # inds_r, inds_c = helper_cross.update_2site(self.out, left_site_pos, site_i, direction, max_bond=self.max_bond,)
        #
        #
        # # print('select inds', i, select_inds)
        # # helper_cross.decimate_2site(self.ket, left_site_pos, inds_r, inds_c, direction)
        # for term in self.terms:
        #     print('ket is bra', self.ket is term.bra)
        #     print('out is bra', self.out is term.bra)
        #     # helper_cross.update_2site(term.bra, left_site_pos, [term.evaluated_site], direction,
        #     #                           inds_r=inds_r, inds_c=inds_c)
        #     # helper_cross.decimate_2site(term.ket, left_site_pos, inds_r, inds_c, direction)
        #     term.update_intermediate_kets(i, 2, direction) #, select_inds=(inds_r, inds_c))
        #     print('updated terms')
        #
        # print('check ket')
        # chk1 = helper_cross.check_orthog(self.ket)
        # print('chk1', left_site_pos, chk1)
        # if chk1[0] != chk1[1]:
        #     exit()
        #
        # chk1 = helper_cross.check_orthog(self.out)
        # print('check out', left_site_pos, chk1)
        # if chk1[0] != chk1[1]:
        #     exit()
        #
        # for term in self.terms:
        #     chk1 = helper_cross.check_orthog(term.ket)
        #     print('term check ket', left_site_pos, chk1)
        #     if chk1[0] != chk1[1]:
        #         exit()
        #
        #     if True:  # left_site_pos == self.ket.L - 2:
        #         print(term._intermediate_kets.keys(), term.num_tiers)
        #         plt.figure()
        #         plt.plot(helper_quimb.to_dense(term.bra, [term.bra.site_ind_id.format(i) for i in range(self.L)]).data.reshape(-1),
        #                  label='bra')
        #         if term.num_tiers > 1:
        #             plt.plot(helper_quimb.to_dense(term._intermediate_kets[0],
        #                                            [term.bra.site_ind_id.format(i) for i in range(self.L)]).data.reshape(-1),
        #                  label='ket0')
        #         plt.plot(helper_quimb.to_dense(term.ket, [term.ket.site_ind_id.format(i) for i in range(self.L)]).data.reshape(-1), '--',
        #                  label='ket')
        #         plt.title(f'updated term {left_site_pos}')
        #         plt.legend()
        #         plt.show()
        #
        # # for term in self.terms:
        # #     term.update_intermediate_kets(i, 2, direction)
        #
        # # for term in self.terms:
        # #     term.canonize_ket_tens(i, 1, direction, max_bond=None)
        #
        #     # helper_cross.check_orthog(term.vec_block.bra, term.vec_block.bra.select_inds,
        #     #                           [term.vec_block.bra.site_ind_id])
        #

        self.update_blocks(i, direction)

        # plt.figure()
        # term = self.terms[-1]
        # ket_data = term.ket.to_dense()
        # int_data = term.get_intermediate_ket(0).to_dense()
        # bra_data = term.bra.to_dense()
        #
        # plt.plot(ket_data, label='ket')
        # plt.plot(int_data, label='int', ls='--')
        # plt.plot(bra_data, label='bra', ls=':')
        # plt.legend()
        # plt.title(f'2-site updated term final {i}')
        # plt.show()

        # for chk in kets:
        #     print('chk', chk)
        # exit()

        return


class TDVPMixed(TDMixed, TDVP_DMRG):

    def _set_local_solve_func(self, te_order: int):
        if te_order == TimeIntegMethod.LW:
            func = self.local_lax_wendroff_so
            self._local_solve_func = func
        else:
            super()._set_local_solve_func(te_order)

    # def _site_solve(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor' = None, return_intermediates=False
    #                 ) -> tuple[Sequence[qtn.Tensor], Numeric]:
    #
    #     print("TDVP SITE SOLVE", left_site_pos, nsites, self.te_order_target)
    #     self._set_local_solve_func(self.te_order_target)
    #     # self._set_local_solve_func(TimeIntegMethod.RK4)
    #     # if self.te_order_target in [223, 226, 0]:
    #     #     self._set_local_solve_func(self.te_order_target)
    #     # else:
    #     #     ## force RK4 (pass if want to use self.te_order_target or self.te_order_final
    #     #     self._set_local_solve_func(TimeIntegMethod.RK4)
    #     #     # print('self.te_order target', self.te_order_target)
    #     #     # self._set_local_solve_func(self.te_order_target)
    #     out, err = super()._site_solve(left_site_pos, nsites, site_tens=site_tens, return_intermediates=False)
    #     return out, err
    #
    #
    def _bond_solve(self, left_site_pos: int, site_tens: 'qtn.Tensor' = None, return_intermediates=False
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:

        if self.verbose > 1:
            print("TDVP MIXED BOND SOLVE", left_site_pos, self.te_order_target)
        # self._set_local_solve_func(self.te_order_target)
        # self._set_local_solve_func(TimeIntegMethod.RK4)
        # if self.te_order_target in [223, 226, 0]:
        #     self._set_local_solve_func(self.te_order_target)
        # else:
        #     self._set_local_solve_func(TimeIntegMethod.RK4)
        #     # self._set_local_solve_func(self.te_order_target)
        out, err = super(TDMixed, self)._bond_solve(left_site_pos, site_tens=site_tens,
                                                    return_intermediates=False)
        return out, err

    def _update_1site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection', filter_bases=False,
                      grid=None, ax_deriv_configs=None):
        """ update ket, bra with new_site
            i: int of mps site
            canonicalize and then back-propagate "bond" (if not at end)
        """
        if self.verbose > 2:
            print('new TDVP mixed update 1 site', i, direction)

        at_end = (i == 0 if direction == SweepDirection.LEFT else i == self.L - 1)
        if at_end:
            super()._update_1site(i, site_i, direction)
            return

        if isinstance(site_i, (tuple, list)):
            site_i = helper_tn.sum_tens(site_i)     ## all other sites are the same (and in canonical form)

        x_ind = self.out.bond(i, i + direction)
        left_inds = [ind for ind in self.out[i].inds if ind != x_ind]
        phys_inds = [self.out.site_ind(i)]
        right_inds = [x_ind]

        # tmp = site_i.reindex({ind: ind[:-2] for ind in site_i.inds if ind[-1] == 'x'})
        # helper_cross.plot_submat(self.out, i, 1, tmp, ref_kets=[self.out])

        if self.version != 'G':
            site_i = helper_mixed.convert_elementwise_to_basis(self.out, site_i, i, 1)

        new_Q, new_R = qtn.tensor_split(site_i, left_inds, absorb='right',
                                        # max_bond=self.max_bond, cutoff=CUTOFF,
                                        bond_ind=x_ind + '_tmp', method='qr')

        # print('new Q', new_Q, 'new R', new_R, 'direction', direction)

        if direction == SweepDirection.LEFT:
            bond_reindex_dict = {x_ind: x_ind + '_L', x_ind + '_tmp': x_ind + '_R'}
        else:  # direction is to the right
            bond_reindex_dict = {x_ind: x_ind + '_R', x_ind + '_tmp': x_ind + '_L'}
        new_R.reindex(bond_reindex_dict, inplace=True)

        new_Q.transpose_like(self.out[i], inplace=True)
        self.out[i].modify(data=new_Q.data)

        ## update select inds
        ind1 = i
        left_inds, phys_inds, right_inds = helper_mixed.get_inds(self.out, ind1)
        if direction > 0:
            prev_tens = self.out.select_tens.get(ind1 - direction)
            tmp_left_ind = left_inds[0] + '_x' if i > 0 else None
            tens = self.out[i].copy() if i == 0 else qtn.tensor_contract(prev_tens, self.out[i])
            sel_inds, TC, TR_inv, TR = helper_mixed.tensor_get_submat(tens, tmp_left_ind, right_inds[0], phys_inds[0],)
            # T1, T2 = tensor_xr(tens, [phys_ind], right_inds, sel_inds)

        else:
            prev_tens = self.out.select_tens.get(ind1 - direction)
            tmp_right_ind = right_inds[0] + '_x' if ind1 < self.out.L - 1 else None
            tens = self.out[ind1].copy() if ind1 == self.out.L - 1 else qtn.tensor_contract(prev_tens, self.out[ind1])
            sel_inds, TC, TR_inv, TR = helper_mixed.tensor_get_submat(tens, tmp_right_ind, left_inds[0], phys_inds[0])
            # T1, T2 = tensor_xr(tens, [phys_ind], left_inds, sel_inds)

        self.out.select_inds[ind1] = sel_inds
        self.out.select_tens[ind1] = TR
        self.out.select_tens_inv[ind1] = TR_inv

        ####

        tens2 = self.out[i + direction]
        self.next_Q = tens2.reindex(bond_reindex_dict, inplace=False)

        ## canonicalize intermediate kets (to specific inds)
        for term in self.terms:
            term.update_intermediate_kets(i, 1, direction)

        self.update_blocks(i, direction=direction)

        ## back-propagation of R ##
        # print('back propagation of R', i, direction) #, new_R)
        back_i = i if direction == SweepDirection.RIGHT else i - 1

        ## need to convert new_R from basis to elements
        if self.version != 'G':
            new_R = helper_mixed.convert_basis_to_elementwise(self.out, new_R, back_i, 0)

        new_R, err = self._bond_time_evolution(new_R, back_i, -self.dt)     ## bond between bond j, j + 1

        ## convert new_R from elements to basis
        if self.version != 'G':
            new_R = helper_mixed.convert_elementwise_to_basis(self.out, new_R, back_i, 0)

        next_site = qtn.tensor_contract(new_R, self.next_Q.copy())
        next_site.transpose_like(self.out[i + direction], inplace=True)
        self.out[i + direction].modify(data=next_site.data)


        ## CHECK ORTHOG
        if self.verbose > 1:
            print("tdvp 1 site")
        tmp1, tmp2 = helper_mixed.check_orthog(self.out)
        if tmp1 != tmp2:
            raise ValueError

        self.out._cur_orthog = i + direction
        return


    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        if self.verbose > 2:
            print('new TDVP mixed update 2 site', i, direction)

        ## canonicalize and then back-propagate "site" (if not at end)
        at_end = (i == 1 if direction == SweepDirection.LEFT else i == self.L - 2)

        if isinstance(site_i, (tuple, list)):
            site_i = helper_tn.sum_tens(site_i)

        ## this only does decimation (projection onto the eigenvalues)
        left_site_pos = i if direction > 0 else i - 1
        helper_mixed.update_and_replace_2site(self.out, left_site_pos, site_i, direction=direction,
                                              max_bond=self.max_bond, cutoff=self.cutoff,
                                              version=self.version)

        ## CHECK ORTHOG
        if self.verbose > 1:
            print("tdvp 2 site (1)")
        tmp1, tmp2 = helper_mixed.check_orthog(self.out)
        if tmp1 != tmp2:
            raise ValueError

        new_R = self.out[i + direction]

        # phys_inds = [self.out.site_ind(i)]
        # x_ind = self.out.bond(i, i + direction)
        # right_inds = [self.out.site_ind(i + direction)]
        # if not at_end:
        #     right_inds += [self.out.bond(i + direction, i + direction * 2)]
        #
        # new_Q, new_R, inds_r, inds_c = helper_cross.tensor_compress(site_i, phys_inds, right_inds,
        #                                                             max_bond=self.max_bond, cutoff=self.cutoff,
        #                                                             bond_ind=x_ind + '_tmp', return_inds=True)
        #
        #
        # # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
        # new_Q.transpose_like(self.out[i], inplace=True)
        # self.out[i].modify(data=new_Q.data)
        # self.out.select_inds[i] = inds_r
        # new_R.transpose_like(self.out[i + direction], inplace=True)
        # new_R.modify(inds=self.out[i + direction].inds)
        # self.out[i + direction].modify(data=new_R.data)

        ## canonicalize intermediate kets (do each individually)
        for term in self.terms:
            # left_site_pos = i if direction > 0 else i - 1
            term.update_intermediate_kets(i, 2, direction=direction)

        ## extend environments to include newly canonical site i
        if not at_end:
            self.update_blocks(i, direction=direction)
        #
        # for term in self.terms:
        #     print('term', term.ket is self.out, term.bra is self.out)
        #     print('block', self.out is term.vec_block.ket, self.out is term.vec_block.bra)
        #     if len(term.op_blocks) > 0:
        #         for block in term.op_blocks[0]:
        #             print('block', self.out is block.ket, self.out is block.bra)
        #

        if not at_end:
            ## back-propagation of R
            # print('back propagation of R', i, direction) #, new_R)
            # r.transpose_like(self.ket[i + direction], inplace=True)
            # self.ket[i + direction].modify(data=r.data)

            if self.version != 'G':
                new_R = helper_mixed.convert_basis_to_elementwise(self.out, new_R, i + direction, 1)
            next_site, err = self._site_time_evolution(new_R, i + direction, -self.dt)
            if self.version != 'G':
                next_site = helper_mixed.convert_elementwise_to_basis(self.out, next_site, i + direction, 1)
            next_site.transpose_like(self.out[i + direction], inplace=True)
            self.out[i + direction].modify(data=next_site.data)
        else:
            pass

        ## CHECK ORTHOG
        if self.verbose > 1:
            print("tdvp 2 site (2)", i)
        tmp1, tmp2 = helper_mixed.check_orthog(self.out)
        if tmp1 != tmp2:
            raise ValueError

        self.out._cur_orthog = i + direction
        # print('updated out', self.out)

        return

    # def _update_1site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection', filter_bases=False,
    #                   grid=None, ax_deriv_configs=None):
    #     """ update ket, bra with new_site
    #         i: int of mps site
    #         canonicalize and then back-propagate "bond" (if not at end)
    #     """
    #     if self.verbose:
    #         print('new TDVP Cross update 1 site', i, direction)
    #
    #     at_end = (i == 0 if direction == SweepDirection.LEFT else i == self.L - 1)
    #     if at_end:
    #         super()._update_1site(i, site_i, direction)
    #         return
    #
    #     if isinstance(site_i, (tuple, list)):
    #         site_i = helper_tn.sum_tens(site_i)     ## all other sites are the same (and in canonical form)
    #
    #     x_ind = self.out.bond(i, i + direction)
    #     left_inds = [ind for ind in self.out[i].inds if ind != x_ind]
    #     phys_inds = [self.out.site_ind(i)]
    #     right_inds = [x_ind]
    #
    #     new_Q, new_R, inds_r, inds_c = helper_cross.tensor_compress(site_i, phys_inds, right_inds,
    #                                                                 # max_bond=self.max_bond, cutoff=self.cutoff,
    #                                                                 bond_ind=x_ind+'_tmp', return_inds=True)
    #
    #     if direction == SweepDirection.LEFT:
    #         bond_reindex_dict = {x_ind: x_ind + '_L', x_ind + '_tmp': x_ind + '_R'}
    #     else:  # direction is to the right
    #         bond_reindex_dict = {x_ind: x_ind + '_R', x_ind + '_tmp': x_ind + '_L'}
    #     new_R.reindex(bond_reindex_dict, inplace=True)
    #
    #     new_Q.transpose_like(self.out[i], inplace=True)
    #     self.out[i].modify(data=new_Q.data)
    #     self.out.select_inds[i] = inds_r
    #
    #     tens2 = self.out[i + direction]
    #     self.next_Q = tens2.reindex(bond_reindex_dict, inplace=False)
    #     new_tens2 = qtn.tensor_contract(new_R, self.next_Q)
    #     new_tens2 = new_tens2.transpose_like(tens2, inplace=True)
    #     # tens2.modify(data=new_tens2.data)
    #
    #     ## canonicalize intermediate kets (to specific inds)
    #     for term in self.terms:
    #         if term.num_tiers > 1:      ## there should be no intermediate kets
    #             print('term inter', term)
    #             for si in range(term.num_tiers - 1):
    #                 ket = term.get_intermediate_ket(si)
    #                 inter_site = term.intermediate_sites[si]
    #                 # sel_inds = self.out.select_inds[i] if direction > 0 else self.out.select_inds[i + 1]
    #                 helper_cross.update_1site(ket, i, inter_site, direction,
    #                                           select_inds=self.out.select_inds[i], decimate_only=True)
    #
    #     # for term in self.terms:
    #     #     # print('term.out', term.bra is self.out)    ## True
    #     #     # pdb.set_trace()
    #     #     if term.num_tiers > 1:
    #     #         kets = [term.get_intermediate_ket(i) for i in range(term.num_tiers - 1)]
    #     #         tensors = [term.intermediate_sites[i] for i in range(term.num_tiers - 1)]
    #     #         helper_cross.update_kets(kets, tensors, i, 1, direction=direction)
    #
    #     self.update_blocks(i, direction=direction)
    #
    #     ## back-propagation of R ##
    #     # print('back propagation of R', i, direction) #, new_R)
    #     back_i = i if direction == SweepDirection.RIGHT else i - 1
    #     new_R, err = self._bond_time_evolution(new_R, back_i, -self.dt)     ## bond between bond j, j + 1
    #     next_site = qtn.tensor_contract(new_R, self.next_Q.copy())
    #     next_site.transpose_like(self.out[i + direction], inplace=True)
    #     self.out[i + direction].modify(data=next_site.data)
    #     self.out._cur_orthog = i + direction
    #     return


    # def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
    #     """ update ket, bra with new_site
    #         i: mps_site
    #     """
    #     if self.verbose:
    #         print('new TDVP Cross update 2 site', i, direction)
    #
    #     ## canonicalize and then back-propagate "site" (if not at end)
    #     at_end = (i == 1 if direction == SweepDirection.LEFT else i == self.L - 2)
    #
    #     if isinstance(site_i, (tuple, list)):
    #         site_i = helper_tn.sum_tens(site_i)
    #
    #     phys_inds = [self.out.site_ind(i)]
    #     x_ind = self.out.bond(i, i + direction)
    #     right_inds = [self.out.site_ind(i + direction)]
    #     if not at_end:
    #         right_inds += [self.out.bond(i + direction, i + direction * 2)]
    #
    #     new_Q, new_R, inds_r, inds_c = helper_cross.tensor_compress(site_i, phys_inds, right_inds,
    #                                                                 max_bond=self.max_bond, cutoff=self.cutoff,
    #                                                                 bond_ind=x_ind + '_tmp', return_inds=True)
    #
    #
    #     # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
    #     new_Q.transpose_like(self.out[i], inplace=True)
    #     self.out[i].modify(data=new_Q.data)
    #     self.out.select_inds[i] = inds_r
    #     new_R.transpose_like(self.out[i + direction], inplace=True)
    #     new_R.modify(inds=self.out[i + direction].inds)
    #     self.out[i + direction].modify(data=new_R.data)
    #
    #     ## canonicalize intermediate kets (do each individually)
    #     for term in self.terms:
    #         if term.num_tiers > 1:
    #             for si in range(term.num_tiers - 1):
    #                 ket = term.get_intermediate_ket(si)
    #                 inter_site = term.intermediate_sites[si]
    #                 # sel_inds = self.out.select_inds[i] if direction > 0 else self.out.select_inds[i + 1]
    #                 helper_cross.update_1site(ket, i, inter_site, direction,
    #                                           select_inds=self.out.select_inds[i], decimate_only=True)
    #
    #     ## extend environments to include newly canonical site i
    #     if not at_end:
    #         self.update_blocks(i, direction=direction)
    #     #
    #     # for term in self.terms:
    #     #     print('term', term.ket is self.out, term.bra is self.out)
    #     #     print('block', self.out is term.vec_block.ket, self.out is term.vec_block.bra)
    #     #     if len(term.op_blocks) > 0:
    #     #         for block in term.op_blocks[0]:
    #     #             print('block', self.out is block.ket, self.out is block.bra)
    #     #
    #
    #     if not at_end:
    #         ## back-propagation of R
    #         # print('back propagation of R', i, direction) #, new_R)
    #         # r.transpose_like(self.ket[i + direction], inplace=True)
    #         # self.ket[i + direction].modify(data=r.data)
    #         next_site, err = self._site_time_evolution(new_R, i + direction, -self.dt)
    #         # next_site = qtn.tensor_contract(new_q, self.ket[i + direction])
    #     else:
    #         next_site = new_R
    #     next_site.transpose_like(self.out[i + direction], inplace=True)
    #     # print("next site", next_site)
    #     self.out[i + direction].modify(data=next_site.data)
    #     self.out._cur_orthog = i + direction
    #     # print('updated out', self.out)
    #
    #     # self.out[i].modify(data=self.init_ket[i].data, inds=self.init_ket[i].inds)
    #     # self.out[i + direction].modify(data=self.init_ket[i + direction].data,
    #     #                                inds=self.init_ket[i + direction].inds)
    #     # self.out._cur_orthog = i + direction
    #
    #     return


    # def local_lax_wendroff_so(self, left_site_pos: int, nsites: int, return_intermediates=False,
    #                        dt: Numeric = None, time: Numeric = None, site_tens: 'qtn.Tensor' = None
    #                        ) -> Union[qtn.Tensor, Sequence['qtn.Tensor']]:
    #
    #     return super(TDVPCross, self).local_lax_wendroff_so(left_site_pos, nsites,
    #                                                       return_intermediates=return_intermediates,
    #                                                       dt = dt, time=time, site_tens=site_tens,)



def global_rk_cross(dt, te_order, ket_state, deriv_func, nsites: int = 1, max_bond=None, cutoff=None,
                    time:Numeric=None, verbose: int = 0):
    """
    state1 = state0 + deriv0(state0) * 0.5 * dt  --> sel inds x; sel inds 0 -> x
    state2 = state0 + deriv1(state1) * 0.5 * dt  --> sel inds x; sel inds x -> x
    state3 = state0 + deriv2(state2) * dt        --> sel inds x; sel inds x -> x
    out = state0 + (deriv0 + deriv1 * 2 + deriv2 * 2 + deriv3(state3))/6  --> sel inds x
    """

    state0 = ket_state.copy()

    if verbose:
        print(f'global cross RK{te_order} DT', dt, 'nsites', nsites)

    if te_order == 1:
        rk_func = helper_TE.euler
    elif te_order == 2:
        rk_func = helper_TE.rk2
    elif te_order == 3:
        rk_func = helper_TE.ssprk4  # ssprk3 or ssprk4
    elif te_order == 4:
        rk_func = helper_TE.rk4
    else:
        raise ValueError

    # print('nsites', nsites, 'deriv func', deriv_func)

    # def deriv_func(state, time=None, **kwargs):
    #     linop_terms = [Term_Cross(state0.copy(), operators=linear_operators, max_bond=max_bond)]
    #     nonlin_terms = [nl_term.create_like(ket=state) for nl_term in nonlinear_terms]
    #     deriv0 = local_cross_evaluator(linop_terms + source_terms + nonlin_terms, max_bond=max_bond, nsites=nsites)
    #     return deriv0

    if state0.L == 1:
        def add_func(mps1, mps2, inplace=False, **kwargs):
            out = mps1 if inplace else mps1.copy()
            out = helper_quimb.add_MPS(out, mps2, inplace=True)
            return out

        def scale_func(mps1, val, inplace=True, **kwargs):
            return helper_quimb.scalar_multiply(mps1, val, inplace=inplace)

        def euler_func(mps1, dt, deriv0=None, time=None, **kwargs):
            if deriv0 is None:
                deriv0 = deriv_func(mps1, time=time)
            if deriv0 is None:
                return mps1.copy()
            state1 = add_func(mps1, scale_func(deriv0, dt, inplace=False), inplace=False)
            return state1
    else:
        def add_func(mps1, mps2, inplace=False, **kwargs):
            init_guess = mps1 if inplace else mps1.copy()
            out = local_mixed_evaluator([Term_Mixed(mps1.copy()), Term_Cross(mps2.copy())], init_guess=init_guess)
            return out

        def scale_func(mps1, val, inplace=True, **kwargs):
            return helper_quimb.scalar_multiply(mps1, val, inplace=inplace)

        def euler_func(mps1, dt, deriv0=None, **kwargs):
            if deriv0 is None:
                deriv0 = deriv_func(mps1)
            deriv0 = scale_func(deriv0, dt, inplace=False)
            if deriv0 is None:
                return mps1.copy()
            state1 = local_mixed_evaluator([Term_Mixed(mps1.copy()), Term_Mixed(deriv0)], max_bond=max_bond, nsites=nsites)
            return state1

    out = rk_func(state0, dt, euler_func, deriv_func, add_func, scale_func, time=time, return_intermediates=False)
    return out


def global_rk4_cross(dt, ket_state, linear_operators, sources=None, nonlinear_terms=None,
                     nsites: int = 1, max_bond=None):
    """
    state1 = state0 + deriv0(state0) * 0.5 * dt  --> sel inds x; sel inds 0 -> x
    state2 = state0 + deriv1(state1) * 0.5 * dt  --> sel inds x; sel inds x -> x
    state3 = state0 + deriv2(state2) * dt        --> sel inds x; sel inds x -> x
    out = state0 + (deriv0 + deriv1 * 2 + deriv2 * 2 + deriv3(state3))/6  --> sel inds x
    """

    nsites = 1
    state0 = ket_state.copy()

    source_terms = [Term_Cross(source, max_bond=max_bond) for source in sources] if sources is not None else []
    nonlinear_terms = [] if nonlinear_terms is None else nonlinear_terms

    def deriv_func(state, time=None, **kwargs):
        linop_terms = [Term_Cross(state0.copy(), operators=linear_operators, max_bond=max_bond)]
        nonlin_terms = [nl_term.create_like(ket=state) for nl_term in nonlinear_terms]
        deriv0 = local_mixed_evaluator(linop_terms + source_terms + nonlin_terms, max_bond=max_bond, nsites=nsites)
        return deriv0

    def add_func(obj1, obj2, inplace=False, **kwargs):
        init_guess = obj1 if inplace else obj1.copy()
        out = local_mixed_evaluator([Term_Cross(obj1.copy()), Term_Cross(obj2.copy())], init_guess=init_guess)
        return out

    def scale_func(obj1, val, inplace=True, **kwargs):
        return helper_quimb.scalar_multiply(obj1, val, inplace=inplace)

    def euler_func(state, dt, deriv0=None, **kwargs):
        if deriv0 is None:
            deriv0 = deriv_func(state)
        deriv0 = scale_func(deriv0, dt, inplace=False)
        state1 = local_mixed_evaluator([Term_Cross(state.copy()), Term_Cross(deriv0)], max_bond=max_bond, nsites=nsites)
        return state1

    out = helper_TE.rk4(state0, dt, euler_func, deriv_func, add_func, scale_func)
    return out

