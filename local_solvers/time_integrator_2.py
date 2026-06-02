"""Global-projection time integrators for the local-solver stack.

Extends :mod:`local_solvers.time_integrator` with
:class:`TimeIntegratorGlobal` and related variants that include the full
ket, its right-hand side F(ket), and source terms together in the projector
(rather than projecting them separately), giving an alternative
time-stepping scheme on top of the same Evaluator machinery.
"""
# from abc import ABC
#
# import helper_quimb
# from setup_.defaults import *
# import local_solvers.helper_tn as helper_tn

# from local_solvers.defaults import *
# from local_solvers.mps_classes import MPS
# from local_solvers.local_evaluator import LocalEvaluator
# from local_solvers.local_dmrg_eval import DMRGEvaluator
# from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross
# import local_solvers.tensor_callables as tc
# import helper_TE
import helper_quimb
from local_solvers.time_integrator import *

if TYPE_CHECKING:
    from grid import Grid
    from setup_.configs import DerivativeConfiguration


class TimeIntegratorGlobal(TimeIntegrator, ABC):
    """ fully include ket, F(ket), sources in projector
    """
    #
    # def __init__(self,
    #              ket_state: Union['MPS', 'qtn.MatrixProductState'],
    #              linear_operators: Sequence['qtn.MatrixProductOperator'],
    #              sources: Sequence[Union['MPS', 'qtn.MatrixProductState']] = None,
    #              nonlinear_terms: Sequence[Union['Term_DMRG', 'Term_Cross']] = None,
    #              constraints: Sequence[Sequence[Union[qtn.MatrixProductState, qtn.MatrixProductOperator]]] = None,
    #              constraint_vals: Sequence[Union[float, qtn.MatrixProductState, Term]] = None,
    #              constraint_funcs: Sequence[Callable] = None,   ## expressions that should yield 0 / be minimized (via scipy)
    #              direction: SweepDirection = SweepDirection.RIGHT,
    #              max_bond: int = None,
    #              conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
    #              max_tot_iter: int = DEFAULT_MAX_TOT_ITER, max_wrong_iter: int = DEFAULT_MAX_WRONG_ITER,
    #              copy_obj: 'TimeIntegrator' = None,
    #              # combine_terms_func: 'Callable' = None,
    #              dt = 0.1, time = None,
    #              te_order_target = 4, te_order_final = 4,
    #              grid: 'Grid' = None, ax_deriv_configs: dict['Axis','DerivativeConfiguration'] = None,
    #              ):
    #
    #     if copy_obj is not None:
    #         super().__init__(ket_state, None, copy_obj=copy_obj)    ## ket should already be expanded
    #
    #     else:
    #
    #         self.ket = ket_state
    #         self.linear_operators = linear_operators
    #         self.sources = sources if sources is not None else []
    #         self.nonlinear_terms = nonlinear_terms if nonlinear_terms is not None else []
    #
    #         self.add_global_projector_to_ket()
    #
    #         super().__init__(ket_state, linear_operators, sources, nonlinear_terms, constraints=constraints,
    #                          constraint_vals=constraint_vals, constraint_funcs=constraint_funcs,
    #                          direction=direction, max_bond=max_bond, conv_tol=conv_tol, max_iter=max_iter,
    #                          max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter, copy_obj=copy_obj,
    #                          grid=grid, ax_deriv_configs=ax_deriv_configs,
    #                          dt=dt, time=time, te_order_final=te_order_final, te_order_target=te_order_target,)



    # @classmethod
    # def add_global_projector_to_ket(cls, ket_state: 'MPSType', linear_operators: Sequence['MPOType'],
    #                                 sources: Sequence['MPSType']=None, direction=-1):
    #
    #     canon_form = 'right' if direction > 0 else 'left'
    #
    #     terms = [helper_quimb.canonize(ket_state, form=canon_form, scale=False)]
    #     ## things might not work if exponent is not zero
    #
    #     for lin_op in linear_operators:
    #         term_ = helper_quimb.apply_zipup(lin_op, ket_state, compress=False, compress_opts={'form':canon_form})
    #         terms += [term_]
    #
    #     if sources is not None:
    #         for term in sources:
    #             term_ = helper_quimb.canonize(term, form=canon_form)
    #             terms += [term_]
    #
    #     out = helper_quimb.add_MPS_list(terms, direction=direction, do_canonize=False, do_final_update=False, inplace=True)
    #     ## update self.ket
    #
    #     return out


    def add_global_projector_to_ket(self, ket_state=None, direction=-1):

        canon_form = 'right' if direction > 0 else 'left'

        ket_state = self.init_ket if ket_state is None else ket_state
        terms = [helper_quimb.canonize(ket_state, form=canon_form, scale=False)]
        ## things might not work if exponent is not zero

        for lin_op in self.linear_operators:
            term_ = helper_quimb.apply_zipup(lin_op, ket_state, compress=False, compress_opts={'form':canon_form})
            term_ = helper_quimb.scalar_multiply(term_, 0.00625, inplace=True)
            terms += [term_]

        if self.sources is not None:
            for term in self.sources:
                term_ = helper_quimb.canonize(term, form=canon_form)
                terms += [term_]

        out = helper_quimb.add_MPS_list(terms, direction=direction, do_canonize=False, do_final_update=False, inplace=True)
        ## update self.ket

        return out


    def initialize_terms(self, ket_state: Union['MPS', 'qtn.MatrixProductState'], cur_orthog=None,
                         **kwargs
                         # linear_operators: Sequence[qtn.MatrixProductOperator],
                         # sources: Optional[Sequence[Union['qtn.MatrixProductState','MPS']]],
                         # nonlinear_terms: Optional[Sequence[Union['Term_DMRG','Term_Cross']]]
                         ):

        # ket_state = self.__class__.add_global_projector_to_ket(ket_state, self.linear_operators, self.sources,
        #                                                        direction=self.direction * -1)
        print('init ket rank', ket_state.max_bond())
        ket_state = self.add_global_projector_to_ket(ket_state=ket_state, direction=self.direction * -1)
        print('expanded ket rank', ket_state.max_bond())

        ## 0 if self.direction * -1 = LEFT, else self.ket.L - 1
        canon_i = ket_state.L - 1 if self.direction == SweepDirection.LEFT else 0

        return super().initialize_terms(ket_state, cur_orthog=canon_i)




class TDDMRG_Global(TimeIntegratorGlobal, TDDMRG):
    """ RK4 follows Feiguin and White
        but ket is expanded by source, lin_ops * ket (or, more generally, F(ket), where d/dt ket + F(ket) = Q)
    """

    # def __init__(self,
    #              ket_state: Union['MPS', 'qtn.MatrixProductState'],
    #              linear_operators: Sequence['qtn.MatrixProductOperator'],
    #              sources: Sequence[Union['MPS', 'qtn.MatrixProductState']] = None,
    #              nonlinear_terms: Sequence[Union['Term_DMRG', 'Term_Cross']] = None,
    #              # constraints: Sequence[Sequence[Union['Term_DMRG','Term_Cross']]] = None,
    #              constraints: Sequence[Sequence[Union[qtn.MatrixProductState, qtn.MatrixProductOperator]]] = None,
    #              constraint_vals: Sequence[Union[float, qtn.MatrixProductState, Term]] = None,
    #              constraint_funcs: Sequence[Callable] = None,
    #              ## expressions that should yield 0 / be minimized (via scipy)
    #              direction: SweepDirection = SweepDirection.RIGHT,
    #              max_bond: int = None,
    #              conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
    #              max_tot_iter: int = DEFAULT_MAX_TOT_ITER, max_wrong_iter: int = DEFAULT_MAX_WRONG_ITER,
    #              copy_obj: 'TimeIntegrator' = None,
    #              # combine_terms_func: 'Callable' = None,
    #              dt=0.1, time=None,
    #              te_order_target=4, te_order_final=4,
    #              grid: 'Grid' = None, ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None,
    #              ):
    #
    #     super().__init__(ket_state, linear_operators, sources=sources, nonlinear_terms=nonlinear_terms,
    #                      constraints=constraints, constraint_vals=constraint_vals, constraint_funcs=constraint_funcs,
    #                      direction=direction, max_bond=max_bond, conv_tol=conv_tol, max_iter=max_iter,
    #                      max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter, copy_obj=copy_obj,
    #                      grid=grid, ax_deriv_configs=ax_deriv_configs,
    #                      dt=dt, time=time, te_order_final=te_order_final, te_order_target=te_order_target,
    #                      )
    #     print('self.direction', self.direction)

class TDVP_Global(TimeIntegratorGlobal, TDVP_DMRG):
    """ RK4 follows Feiguin and White
        but ket is expanded by source, lin_ops * ket (or, more generally, F(ket), where d/dt ket + F(ket) = Q)
    """

class TDLocal_Global(TimeIntegratorGlobal, DMRGEvaluator):
    """
    local TE but using basis defined from expanded basis, without any sweeping.
    """
    def solve_l2r(self, nsites: int, canonize=False, verbose=False, filter_bases=False, **kwargs):
        """ direction l2r means ket currently in right canonical form, cur_orthog = 0
        """

        print('solve l2r', self.max_bond, self.dt)

        L = self.L

        if canonize:
            canon_site = 0
            self.canonize(canon_site)
            self.direction = SweepDirection.RIGHT

        ## right sweep
        assert(0 <= self.cur_orthog <= nsites-1), f'cur_orthog should be 0 not {self.cur_orthog}'
        assert(self.direction == SweepDirection.RIGHT), f'should be SweepDirection.RIGHT, not {self.direction}'
        direction = self.direction


        # if self.cur_orthog == 0:
        #     print('check orthog', self.check_right_orthog()) #self.ket, self.ket_select_inds, [self.ket.site_ind_id]))
        # else:
        #     print('check orthog', self.check_left_orthog()) #self.ket, self.ket_select_inds, [self.ket.site_ind_id]))
        # print('check orthog ket', helper_quimb.check_orthog(self.ket))
        # print('check orthog out', helper_quimb.check_orthog(self.out))

        # print('x norms', [self.ket[i].norm() for i in range(self.ket.L)])

        self.direction = SweepDirection.LEFT
        print('self.direction', self.direction)

        ## one right to left sweep
        i = 0
        site_i, site_err = self._site_solve(i, nsites, return_intermediates=False)
        if site_err > 10 ** 5:
            print('l2r site error too large', site_err)
            exit()
        tot_err = site_err

        ## update ket
        ## use opposite direction to signify that sweep is at the end
        if nsites == 1:
            # site_i = self.ket[i]
            self._update_1site(i, site_i, self.direction, filter_bases=filter_bases,
                               grid=self.grid, ax_deriv_configs=self.ax_deriv_configs)
        elif nsites == 2:
            self._update_2site(i, site_i, self.direction)
        else:
            raise NotImplementedError

        return self.init_ket, tot_err


    def solve_r2l(self, nsites: int, canonize=False, verbose=False, filter_bases=False, **kwargs):

        print('solve r2l', self.max_bond, self.dt)

        L = self.L

        if canonize:
            canon_site = L-1
            self.canonize(canon_site)
            self.direction = SweepDirection.LEFT

        ## right sweep
        # assert (self.cur_orthog == L-1), f'cur_orthog should be {L-1} not {self.cur_orthog}'
        assert (L-nsites <= self.cur_orthog <= L - 1), f'cur_orthog should be {L - 1} not {self.cur_orthog}'
        assert (self.direction == SweepDirection.LEFT), f'direction should be SweepDirection.LEFT, not {self.direction}'
        direction = self.direction

        # print('check orthog ket', helper_quimb.check_orthog(self.ket))
        # print('check orthog out', helper_quimb.check_orthog(self.out))

        self.direction = SweepDirection.RIGHT

        i = L - 1
        left_site_pos = i - nsites + 1
        site_i, site_err = self._site_solve(left_site_pos, nsites)
        if site_err > 10 ** 5:
            print('r2l site err too large', site_err)
            exit()
        tot_err = site_err

        ## update ket, bra
        if nsites == 1:
            self._update_1site(i, site_i, self.direction, filter_bases=filter_bases,
                               grid=self.grid, ax_deriv_configs=self.ax_deriv_configs)
        elif nsites == 2:
            self._update_2site(i, site_i, self.direction)
        else:
            raise NotImplementedError

        return self.init_ket, tot_err
