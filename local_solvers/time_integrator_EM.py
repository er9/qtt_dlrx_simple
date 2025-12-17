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

if TYPE_CHECKING:
    from grid import Grid
    from gridTN import GridTN
    from field import Field
    from coord.coord_sys import CoordinateSystem
    from setup_.configs import DerivativeConfiguration


class TimeIntegMethod(IntEnum):
    Euler = 1
    RK4 = 4
    CN = 223
    CN6 = 226
    EXACT = 0

MPS_type = Union['MPS', 'qtn.MatrixProductState']
MPO_type = 'qtn.MatrixProductOperator'

class TimeIntegratorBlock(ABC):

    def __init__(self,
                 ket_states: dict[Any, MPS_type],
                 linear_operators: dict[Any, Sequence['MPOType']],
                 sources: dict[Any, Sequence[MPS_type]] = None,
                 nonlinear_terms: dict[Any, Callable] = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 max_bond: int = None,
                 conv_tol: float = DEFAULT_CONV_TOL, max_iter: int = DEFAULT_MAX_ITER,
                 max_tot_iter: int = DEFAULT_MAX_TOT_ITER, max_wrong_iter: int = DEFAULT_MAX_WRONG_ITER,
                 copy_obj: 'TimeIntegratorBlock' = None,
                 # combine_terms_func: 'Callable' = None,
                 dt = 0.1, time = None,
                 te_order_target = 4, te_order_final = 4,
                 grid: 'Grid' = None, ax_deriv_configs: dict['Axis','DerivativeConfiguration'] = None,
                 time_mpo_list=None, verbose_plot=False, local_te_func = None,
                 ):

        if copy_obj is not None:
            # super().__init__(ket_states, None, copy_obj=copy_obj)

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
            self.local_te_func = copy_obj.local_te_func

            # self.time_mpo_list = copy_obj.time_mpo_list
            self.verbose_plot = copy_obj.verbose_plot

        else:

            self.init_ket = {k: v.copy() for k, v in ket_states.items()}
            self.keys = ket_states.keys()

            self.te_order_target = te_order_target if te_order_target != 0 else 223
            self.te_order_final = te_order_final
            self.local_te_func = local_te_func

            self.linear_operators = linear_operators
            self.sources = sources if sources is not None else {}
            self.nonlinear_terms = nonlinear_terms if nonlinear_terms is not None else {}
            # self.time_mpo_list = time_mpo_list

            # self.self_term: Term = None
            # self.linear_terms: list[Term] = []
            # self.source_terms: list[Term] = []
            # self.extra_terms_dict: dict[Any, list[Term]] = {}
            self.self_term_dict: dict[Any, Term] = {}
            self.linear_terms_dict: dict[Any, list[Term]] = {}
            self.source_terms_dict: dict[Any, list[Term]] = {}
            self.nonlinear_terms_dict: dict[Any, list[Term]] = {}

            self._direction = direction

            self.initialize_terms(self.init_ket)

            self.dt = dt
            self.time = time
            self.verbose_plot = verbose_plot

    @classmethod
    def term_class(cls):
        raise NotImplementedError

    def initialize_terms(self, ket_states: dict[Any, 'MPS_type'], cur_orthog=None,):

        LocalTerm = self.term_class()

        term_self_dict = {}
        term_linear_dict = {}
        term_nonlin_dict = {}
        term_source_dict = {}

        for k, ket_state in ket_states.items():

            sources = self.sources.get(k, [])

            ## if DMRG; project sources and ket onto the same basis
            if len(sources) > 0:
                if isinstance(LocalTerm, Term_DMRG):
                    helper_quimb.add_MPS_list([ket_state, *sources], inplace=True, do_final_update=False,)
                term_source_dict[k] = [LocalTerm(source, cur_orthog=cur_orthog) for source in sources]

            ## self term
            term_self = LocalTerm(ket_state)
            term_self_dict[k] = term_self

            ## operator terms
            lin_ops = self.linear_operators.get(k, [])
            if len(lin_ops) > 0:
                term_linear_dict[k] = [LocalTerm(ket_state, operators=[mpo for mpo in self.linear_operators],
                                                 cur_orthog=cur_orthog)]

            ## nonlinear terms
            nlin_fcts = self.nonlinear_terms.get(k, [])
            if len(nlin_fcts) > 0:
                nonlinear_terms = []
                for nlin_fct in nlin_fcts:
                    if isinstance(nlin_fct, dict):
                        ## kwargs of LocalTerm: e.g. operator: list MPOs; mps_coeffs; mps_power
                        nlin_term = LocalTerm(ket_state, **nlin_fct)
                    else:
                        nlin_term = LocalTerm(ket_state, mps_func=nlin_fct)
                    nonlinear_terms += [nlin_term]
                term_nonlin_dict[k] = nonlinear_terms

        self.self_term_dict = term_self_dict
        self.linear_terms_dict = term_linear_dict
        self.source_terms_dict = term_source_dict
        self.nonlinear_terms_dict = term_nonlin_dict
        return


    # def _set_local_solve_func(self, te_order: int):
    #     # te_order = 1
    #     # te_order = 223
    #     # te_order = 0
    #     print('te order', te_order)
    #     if te_order == TimeIntegMethod.RK4:
    #         func = self.local_rk4
    #     elif te_order == TimeIntegMethod.EXACT:
    #         func = self.local_exact_solve
    #     elif te_order == TimeIntegMethod.Euler:
    #         func = self.local_euler
    #     elif te_order == TimeIntegMethod.CN:
    #         func = self.local_implicit_solve
    #     elif te_order == TimeIntegMethod.CN6:
    #         def func(left_site_pos: int, nsites: int, return_intermediates=False):
    #             self.local_implicit_solve(left_site_pos, nsites, return_intermediates=return_intermediates,
    #                                       weight=0.6)
    #     elif te_order == TimeIntegMethod.EXACT:
    #         func = self.local_exact_solve
    #     else:
    #         raise ValueError(f'invalid te_order ({te_order}) provided')
    #
    #     self._local_solve_func = func

    def euler_func(self, dt: Numeric, sites_dict: dict[Any, Sequence[qtn.Tensor]], time: Numeric=None):
        raise NotImplementedError

    def _site_solve(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor' = None, return_intermediates=True,
                    ) -> tuple[qtn.Tensor, Numeric]:

        raise NotImplementedError



def evolve_E_part(part: 'str', dt, E_comp: 'GridTN', B: 'Field', coords_x: 'CoordinateSystem',
                  matl_params: 'UnitsConfiguration',
                  deriv_pml_correction_gtn: Union['GridTN', 'Callable']=None,
                  E_pml_correction_gtn: Union['GridTN', 'Callable']=None,
                  current_density: Optional['Field'] = None,
                  do_upwind=False, permittivity=1.0, permeability=1.0,
                  max_bond: int=None, cutoff: Numeric=CUTOFF, inplace=True):
    """ dE/dt = c * curl B - eps0 J
        part: 'zx' or 'zy'
    """
    grid_X = E_comp.grid
    X, Y, Z = coords_x.coords
    LocalTerm = Term_Cross
    ref_mps = None

    for compID, comp in B.components.items():
        new_ax_deriv_configs = comp.ax_deriv_configs
        for ax in grid_X.axes:
            if comp is not None:
                deriv_config = comp.ax_deriv_configs[ax].copy()
                deriv_config.update(order=0, fd_type=FDType.BACKWARD)
                comp.ax_deriv_configs[ax] = deriv_config

    if E_comp.data is not None:
        ref_mps = E_comp.data
    self_term = LocalTerm(E_comp.data) if E_comp.data is not None else None

    derivB = None
    if part == 'zx':
        Bx = B.components.get(X, None)
        if Bx is not None and Bx.data is not None:
            ax_y = coords_x.get_axis(Y)
            derivB = Bx.take_firstderivative(ax=ax_y)
            derivB.scalar_multiply(-1, inplace=True)
    elif part == 'zy':
        By = B.components.get(Y, None)
        derivB = None
        if By is not None and By.data is not None:
            ax_x = coords_x.get_axis(X)
            derivB = By.take_firstderivative(ax=ax_x)
    else:
        raise NotImplementedError

    if (derivB is not None and derivB.data is not None):
        derivB = derivB.scalar_multiply(dt)
        if ref_mps is not None:
            helper_quimb.match_inner_inds(derivB.data, ref_mps)
        else:
            ref_mps = derivB.data
        derivB_term = LocalTerm(derivB.data)
    else:
        derivB_term = None

    ## pml correction term
    if deriv_pml_correction_gtn is not None and deriv_pml_correction_gtn.data is not None:
        if ref_mps is not None:
            helper_quimb.match_inner_inds(deriv_pml_correction_gtn.data, ref_mps)
        else:
            ref_mps = deriv_pml_correction_gtn.data
        deriv_pml_term = LocalTerm(deriv_pml_correction_gtn.data)
    else:
        deriv_pml_term = None

    ## E field
    if E_pml_correction_gtn is not None and E_pml_correction_gtn.data is not None:
        if ref_mps is not None:
            helper_quimb.match_inner_inds(E_pml_correction_gtn.data, ref_mps)
        else:
            ref_mps = E_pml_correction_gtn.data
        E_pml_term = LocalTerm(E_pml_correction_gtn.data)
    else:
        E_pml_term = None


    ## current
    j: Optional[Field] = current_density
    jc = None
    j_term = None
    if j is not None:
        if part[0] == 'z':
            jc = current_density.components.get(Z, None)
        elif part[0] == 'y':
            jc = current_density.components.get(Y, None)
        elif part[0] == 'x':
            jc = current_density.components.get(X, None)
        else:
            raise ValueError

        if jc is not None:
            if matl_params.is_cgs:
                jc = jc.scalar_multiply(-4 * np.pi, inplace=False)
            else:
                jc = jc.scalar_multiply(-1 / matl_params.eps0, inplace=False)

            if ref_mps is not None:
                helper_quimb.match_inner_inds(jc.data, ref_mps)

            jc = jc.scalar_multiply(dt)
            j_term = LocalTerm(jc.data)

    terms = [self_term, derivB_term, deriv_pml_term, E_pml_term, j_term]

    deriv2E_x_term, deriv2E_y_term = None, None
    if do_upwind:
        if E_comp is not None and E_comp.data is not None:
            Z = np.sqrt(permeability / permittivity) / permeability
            ax_x, ax_y = grid_X.axes
            deriv2E_x = E_comp.take_secondderivative(ax_x, None)
            deriv2E_y = E_comp.take_secondderivative(ax_y, None)
            deriv2E_x.scalar_multiply(-1 / Z * ax_x.dx / 2 * dt, inplace=True)
            deriv2E_y.scalar_mulitply(-1 / Z * ax_y.dx / 2 * dt, inplace=True)

            deriv2E_x_term = LocalTerm(deriv2E_x.data)
            deriv2E_y_term = LocalTerm(deriv2E_y.data)

    terms += [deriv2E_x_term, deriv2E_y_term]
    print('terms', terms)

    def eval_func(tens_list: Sequence[Optional[qtn.Tensor]]):
        """ perform Euler time step of E component
        """
        E_tens, derivB_tens, deriv_pml_tens, E_pml_tens, j_tens, deriv2E_x_tens, deriv2E_y_tens = tens_list
        ref_tens = None

        print('in eval func')
        E_data = 0.
        if E_tens is not None:
            if ref_tens is None:
                ref_tens = E_tens
            E_data = E_tens.data

        derivB_data = 0.
        if derivB_tens is not None:
            if ref_tens is None:
                ref_tens = derivB_tens
            else:
                derivB_tens.transpose_like(ref_tens, inplace=True)
            derivB_data = derivB_tens.data

        deriv_pml_data = 1.
        if deriv_pml_tens is not None:
            if ref_tens is None:
                ref_tens = deriv_pml_tens
            else:
                deriv_pml_tens.transpose_like(ref_tens, inplace=True)
            deriv_pml_data = deriv_pml_tens.data

        E_pml_data = 1.
        if E_pml_tens is not None:
            if ref_tens is None:
                ref_tens = E_pml_tens
            else:
                E_pml_tens.transpose_like(ref_tens, inplace=True)
            E_pml_data = E_pml_tens.data

        j_data = 0.
        if j_tens is not None:
            if ref_tens is not None:
                j_tens.transpose_like(ref_tens, inplace=True)
            else:
                ref_tens = j_tens
            j_data = j_tens.data

        deriv2E_x_data = 0.
        if deriv2E_x_tens is not None:
            deriv2E_x_tens.transpose_like(ref_tens, inplace=True)
            deriv2E_x_data = deriv2E_x_tens.data

        deriv2E_y_data = 0.
        if deriv2E_y_tens is not None:
            deriv2E_y_tens.transpose_like(ref_tens, inplace=True)
            deriv2E_y_data = deriv2E_y_tens.data

        out = E_pml_data * E_data + (derivB_data * deriv_pml_data + j_data + deriv2E_x_data + deriv2E_y_data) # * dt
        out_tens = qtn.Tensor(out, ref_tens.inds)
        return out_tens


    from local_solvers.local_cross_eval import CrossEvaluator

    # if E_comp is not None and E_comp.data is not None:
    #     init_guess = E_comp.data
    # elif derivB is not None and derivB.data is not None:
    #     init_guess = derivB.data
    # elif jc is not None and jc.data is not None:
    #     init_guess = jc.data
    # else:
    #     return None

    all_mps = []
    if E_comp is not None and E_comp.data is not None:
        all_mps += [ E_comp.data ]
    if derivB is not None and derivB.data is not None:
        all_mps += [ derivB.data ]
    if jc is not None and jc.data is not None:
        all_mps += [jc.data]

    # print('all mps', all_mps)
    # print('all mps', [m.max_bond() if m is not None else None for m in all_mps])
    init_guess = helper_quimb.add_MPS_list(all_mps, compress_opts={'max_bond': max_bond})

    print('init guess', init_guess.max_bond())
    helper_quimb.match_inner_inds(init_guess, ref_mps, inplace=True)

    nsites = 2
    solver = CrossEvaluator(init_guess, terms, combine_terms_func=eval_func,
                            max_bond=max_bond, cutoff=cutoff)
    solver.solve(nsites)

    if inplace:
        E_comp.data = solver.solution
    else:
        E_comp = E_comp.create_like(solver.solution)

    return E_comp


def evolve_E_part_dmrg(part: 'str', dt, E_comp: 'GridTN', B: 'Field', coords_x: 'CoordinateSystem',
                       matl_params: 'UnitsConfiguration',
                       deriv_pml_correction_gtn: Union['GridTN', 'Callable']=None,
                       E_pml_correction_gtn: Union['GridTN', 'Callable']=None,
                       current_density: Optional['Field'] = None,
                       do_upwind=False, permittivity=1.0, permeability=1.0,
                       max_bond: int=None, cutoff: Numeric=CUTOFF, inplace=True):
    """ dE/dt = c * curl B - eps0 J
        part: 'zx' or 'zy'
    """
    grid_X = E_comp.grid
    X, Y, Z = coords_x.coords
    LocalTerm = Term_DMRG
    ref_mps = None

    ## pml correction term
    if deriv_pml_correction_gtn is not None and deriv_pml_correction_gtn.data is not None:
        if ref_mps is not None:
            helper_quimb.match_inner_inds(deriv_pml_correction_gtn.data, ref_mps)
        else:
            ref_mps = deriv_pml_correction_gtn.data
        deriv_pml_data = deriv_pml_correction_gtn.data
    else:
        deriv_pml_data = None

    ## E field pml
    if E_pml_correction_gtn is not None and E_pml_correction_gtn.data is not None:
        if ref_mps is not None:
            helper_quimb.match_inner_inds(E_pml_correction_gtn.data, ref_mps)
        else:
            ref_mps = E_pml_correction_gtn.data
        E_pml_data = E_pml_correction_gtn.data
    else:
        E_pml_data = None

    for compID, comp in B.components.items():
        new_ax_deriv_configs = comp.ax_deriv_configs
        for ax in grid_X.axes:
            if comp is not None:
                deriv_config = comp.ax_deriv_configs[ax].copy()
                deriv_config.update(order=0, fd_type=FDType.BACKWARD)
                comp.ax_deriv_configs[ax] = deriv_config

    self_term = None
    if E_comp.data is not None:
        # E_comp.data.distribute_exponent()
        ref_mps = E_comp.data
        if E_pml_data is None:
            self_term = LocalTerm(E_comp.data) if E_comp.data is not None else None
        else:
            # E_comp_data = helper_quimb.scalar_multiply(E_comp.data, E_pml_correction_gtn.get_data()[0,0])
            # self_term = LocalTerm(E_comp_data)  # , operators=[E_pml_data.copy()]) if E_comp.data is not None else None
            self_term = LocalTerm(E_comp.data, operators=[E_pml_data.copy()]) if E_comp.data is not None else None


    derivB = None
    if part == 'zx':
        Bx = B.components.get(X, None)
        if Bx is not None and Bx.data is not None:
            ax_y = coords_x.get_axis(Y)
            derivB = Bx.take_firstderivative(ax=ax_y)
            derivB.scalar_multiply(-1, inplace=True)
    elif part == 'zy':
        By = B.components.get(Y, None)
        derivB = None
        if By is not None and By.data is not None:
            ax_x = coords_x.get_axis(X)
            derivB = By.take_firstderivative(ax=ax_x)
    else:
        raise NotImplementedError

    if (derivB is not None and derivB.data is not None):
        derivB.scalar_multiply(dt, inplace=True)
        # derivB.data.distribute_exponent()
        if ref_mps is not None:
            helper_quimb.match_inner_inds(derivB.data, ref_mps)
        else:
            ref_mps = derivB.data
        if deriv_pml_data is None:
            derivB_term = LocalTerm(derivB.data)
        else:
            # print('deriv pml', deriv_pml_correction_gtn.get_data()[0, 0])
            # exit()
            # derivB_data = helper_quimb.scalar_multiply(derivB.data, deriv_pml_correction_gtn.get_data()[0, 0])
            # derivB_term = LocalTerm(derivB_data)  # , operators=[deriv_pml_data.copy()])
            derivB_term = LocalTerm(derivB.data, operators=[deriv_pml_data.copy()])
    else:
        derivB_term = None


    ## current
    j: Optional[Field] = current_density
    jc = None
    j_term = None
    if j is not None:
        if part[0] == 'z':
            jc = current_density.components.get(Z, None)
        elif part[0] == 'y':
            jc = current_density.components.get(Y, None)
        elif part[0] == 'x':
            jc = current_density.components.get(X, None)
        else:
            raise ValueError

        if jc is not None:
            if matl_params.is_cgs:
                jc = jc.scalar_multiply(-4 * np.pi, inplace=False)
            else:
                jc = jc.scalar_multiply(-1 / matl_params.eps0, inplace=False)
            jc = jc.scalar_multiply(dt, inplace=True)

            if ref_mps is not None:
                helper_quimb.match_inner_inds(jc.data, ref_mps)

            # jc.data.distribute_exponent()
            j_term = LocalTerm(jc.data)
            # if deriv_pml_data is None:
            #     j_term = LocalTerm(jc.data)
            # else:
            #     j_term = LocalTerm(jc.data, operators=[deriv_pml_data.copy()])

    terms = [self_term, derivB_term, j_term]

    deriv2E_x_term, deriv2E_y_term = None, None
    if do_upwind:
        if E_comp is not None and E_comp.data is not None:
            Z = np.sqrt(permeability / permittivity) / permeability
            ax_x, ax_y = grid_X.axes
            deriv2E_x = E_comp.take_secondderivative(ax_x, None)
            deriv2E_y = E_comp.take_secondderivative(ax_y, None)
            deriv2E_x.scalar_multiply(-1 / Z * ax_x.dx / 2 * dt, inplace=True)
            deriv2E_y.scalar_mulitply(-1 / Z * ax_y.dx / 2 * dt, inplace=True)

            deriv2E_x_term = LocalTerm(deriv2E_x.data)
            deriv2E_y_term = LocalTerm(deriv2E_y.data)

        terms += [deriv2E_x_term, deriv2E_y_term]

    print('terms', terms)

    def eval_func(tens_list: Sequence[Optional[qtn.Tensor]]):
        """ perform Euler time step of E component
        """
        E_tens, derivB_tens, j_tens, deriv2E_x_tens, deriv2E_y_tens = tens_list
        ref_tens = None

        print('in eval func')
        E_data = 0.
        if E_tens is not None:
            if ref_tens is None:
                ref_tens = E_tens
            E_data = E_tens.data

        derivB_data = 0.
        if derivB_tens is not None:
            if ref_tens is None:
                ref_tens = derivB_tens
            else:
                derivB_tens.transpose_like(ref_tens, inplace=True)
            derivB_data = derivB_tens.data

        j_data = 0.
        if j_tens is not None:
            if ref_tens is not None:
                j_tens.transpose_like(ref_tens, inplace=True)
            else:
                ref_tens = j_tens
            j_data = j_tens.data

        deriv2E_x_data = 0.
        if deriv2E_x_tens is not None:
            deriv2E_x_tens.transpose_like(ref_tens, inplace=True)
            deriv2E_x_data = deriv2E_x_tens.data

        deriv2E_y_data = 0.
        if deriv2E_y_tens is not None:
            deriv2E_y_tens.transpose_like(ref_tens, inplace=True)
            deriv2E_y_data = deriv2E_y_tens.data


        out = E_data + (derivB_data + j_data + deriv2E_x_data + deriv2E_y_data)  # * dt
        out_tens = qtn.Tensor(out, ref_tens.inds)

        print('in dmrg eval func')
        # exit()

        return out_tens


    from local_solvers.local_dmrg_eval import DMRGEvaluator

    # if E_comp is not None and E_comp.data is not None:
    #     init_guess = E_comp.data
    # elif derivB is not None and derivB.data is not None:
    #     init_guess = derivB.data
    # elif jc is not None and jc.data is not None:
    #     init_guess = jc.data
    # else:
    #     return None

    all_mps = []
    if E_comp is not None and E_comp.data is not None:
        all_mps += [ E_comp.data ]
    if derivB is not None and derivB.data is not None:
        all_mps += [ derivB.data ]
    if jc is not None and jc.data is not None:
        all_mps += [jc.data]

    # print('all mps', all_mps)
    # print('all mps', [m.max_bond() if m is not None else None for m in all_mps])
    init_guess = helper_quimb.add_MPS_list(all_mps, compress_opts={'max_bond': max_bond})

    print('init guess', init_guess.max_bond())

    helper_quimb.match_inner_inds(init_guess, ref_mps, inplace=True)
    init_guess.distribute_exponent()

    nsites = 1
    solver = DMRGEvaluator(init_guess, terms, combine_terms_func=eval_func,
                            max_bond=max_bond, cutoff=cutoff)
    solver.solve(nsites)

    if inplace:
        E_comp.data = solver.solution
    else:
        E_comp = E_comp.create_like(solver.solution)

    # E_comp.compress(compress_opts={'max_bond': max_bond, 'cutoff': cutoff})

    return E_comp
