from abc import ABC

import helper_quimb
from setup_.defaults import *
from setup_.configs import *
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

""" FDTD with local updates; altered to fit other upwinding methods
"""

def evolve_E_part(part: 'str', dt, E_comp: 'GridTN', B: 'Field', coords_x: 'CoordinateSystem',
                  matl_params: 'UnitsConfiguration',
                  deriv_pml_correction_gtn: Union['GridTN', 'Callable']=None,
                  E_pml_correction_gtn: Union['GridTN', 'Callable']=None,
                  do_upwind=False, permittivity=1.0, permeability=1.0,
                  current_density: Optional['Field'] = None, max_bond: int=None, cutoff: Numeric=CUTOFF,
                  inplace=True, verbose_plot=False):
    """ dE/dt = c * curl B - eps0 J
        part: 'zx' or 'zy'
    """
    grid_X = E_comp.grid
    X, Y, Z = coords_x.coords
    # LocalTerm = Term_Cross
    ref_mps = E_comp.data

    compress_config = CompressionConfiguration()
    compress_config.set_compress_opts(1, max_bond=max_bond, cutoff=cutoff)


    for compID, comp in B.components.items():
        new_ax_deriv_configs = comp.ax_deriv_configs
        for ax in grid_X.axes:
            if comp is not None:
                deriv_config = comp.ax_deriv_configs[ax].copy()
                deriv_config.update(order=0, fd_type=FDType.BACKWARD)
                comp.ax_deriv_configs[ax] = deriv_config


    ## upwind term:  c/2 ( dx * d^2 /dx^2 + dy * d^2/dy^2) Ez
    ## note: this upwinding assumes uniform dielectric constant
    if do_upwind:
        deriv2E_x, deriv2E_y = None, None
        if E_comp is not None and E_comp.data is not None:
            Z = np.sqrt(permeability / permittivity)
            ax_x, ax_y = grid_X.axes
            deriv2E_x = E_comp.take_secondderivative(ax_x, None)
            deriv2E_y = E_comp.take_secondderivative(ax_y, None)
            deriv2E_x.scalar_multiply(-1 / Z * ax_x.dx / 2, inplace=True )
            deriv2E_y.scalar_mulitply(-1 / Z * ax_y.dx / 2, inplace=True )

    derivB = None
    if part == 'zx':
        Bx = B.components.get(X, None)
        if Bx is not None and Bx.data is not None:
            ax_y = coords_x.get_axis(Y)
            derivB = Bx.take_firstderivative(ax=ax_y)
            derivB.scalar_multiply(-1, inplace=True)
            # derivB.data.distribute_exponent()
    elif part == 'zy':
        By = B.components.get(Y, None)
        derivB = None
        if By is not None and By.data is not None:
            ax_x = coords_x.get_axis(X)
            derivB = By.take_firstderivative(ax=ax_x)
            # derivB.data.distribute_exponent()
    else:
        raise NotImplementedError

    # if derivB is not None:
    #     derivB.scalar_multiply(dt)

    if ref_mps is None and derivB is not None:
        ref_mps = derivB.data

    # if (derivB is not None and derivB.data is not None):
    #     derivB = derivB.scalar_multiply(dt)
    #     if ref_mps is not None:
    #         helper_quimb.match_inner_inds(derivB.data, ref_mps)
    #     else:
    #         ref_mps = derivB.data

    ## current
    j: Optional[Field] = current_density
    jc = None
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

            # jc.data.distribute_exponent()
            # jc.scalar_multiply(dt, inplace=True)

            # if ref_mps is not None:
            #     helper_quimb.match_inner_inds(jc.data, ref_mps)

        if ref_mps is None:
            ref_mps = jc.data

    # ## pml correction term
    # if deriv_pml_correction_gtn is not None and deriv_pml_correction_gtn.data is not None:
    #     if ref_mps is not None:
    #         helper_quimb.match_inner_inds(deriv_pml_correction_gtn.data, ref_mps)
    #     else:
    #         ref_mps = deriv_pml_correction_gtn.data
    #
    # ## E field pml correction term
    # if E_pml_correction_gtn is not None and E_pml_correction_gtn.data is not None:
    #     if ref_mps is not None:
    #         helper_quimb.match_inner_inds(E_pml_correction_gtn.data, ref_mps)
    #     else:
    #         ref_mps = E_pml_correction_gtn.data
    #
    #############################

    source_dict = {}
    if derivB is not None and derivB.data is not None:
        source_dict['deriv_B'] = [derivB.data]
    if jc is not None and jc.data is not None:
        source_dict['j'] = [jc.data]

    pml_dict = {}
    if deriv_pml_correction_gtn is not None and deriv_pml_correction_gtn.data is not None:
        pml_dict['deriv_pml'] = [deriv_pml_correction_gtn.data]
    if E_pml_correction_gtn is not None and E_pml_correction_gtn.data is not None:
        pml_dict['E_pml'] = [E_pml_correction_gtn.data]

    upwind_dict = {}
    if do_upwind:
        upwind_dict['deriv2E_x'] = [deriv2E_x.data]
        upwind_dict['deriv2E_y'] = [deriv2E_y.data]

    def upwind_func(dt: Numeric, ket: 'MPSType', submat: 'qtn.Tensor', selectors: Sequence[int],
                    upwind_terms_dict: dict[Any, 'qtn.Tensor'], ref_deriv:'qtn.Tensor'=None,
                    left_site_pos=None, nsites:int = None, select_inds = None):

        E_tens = submat
        deriv_pml_tens = upwind_terms_dict.get('deriv_pml', None)
        E_pml_tens = upwind_terms_dict.get('E_pml', None)
        derivB_tens = upwind_terms_dict.get('deriv_B', None)
        j_tens = upwind_terms_dict.get('j', None)

        deriv2E_x_tens = upwind_terms_dict.get('deriv2E_x', None)
        deriv2E_y_tens = upwind_terms_dict.get('deriv2E_y', None)

        ref_tens = None

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

        # print('E pml data', E_pml_data)
        # print('E data', E_data)
        # print('derivB_data', derivB_data)
        # print('deriv pml', deriv_pml_data)
        # print('j data', j_data)

        out = E_pml_data * E_data + (derivB_data * deriv_pml_data + j_data + deriv2E_x_data + deriv2E_y_data) * dt
        out_tens = qtn.Tensor(out, ref_tens.inds)

        # from local_solvers import helper_cross
        # ref_kets = [E_comp.data]
        # if derivB is not None:
        #     ref_kets += [derivB.data]  # scalar_multiply(dt).data]
        # # if jc is not None:
        # #     ref_kets += [jc.data]
        # if derivB_tens is not None:
        #     print('select inds', select_inds)
        #     helper_cross.plot_submat(ket, left_site_pos, nsites, out_tens, select_inds=select_inds,
        #                              ref_kets=ref_kets)


        # print('upwind func diff', out - E_data)
        return out_tens

    ### build init guess
    all_mps = []
    if E_comp is not None and E_comp.data is not None:
        all_mps += [ E_comp.data ]
    if derivB is not None and derivB.data is not None:
        all_mps += [ derivB.data ]
    if jc is not None and jc.data is not None:
        all_mps += [jc.data]

    if E_comp is not None and E_comp.data is not None:
        print('not zero E comp')
        if max_bond is None or E_comp.max_bond() < max_bond:
            init_guess = helper_quimb.add_MPS_list(all_mps, compress_opts={'max_bond': max_bond}, do_final_update=False)
        else:
            init_guess = E_comp.data
    else:
        print('zero E comp')
        init_guess = helper_quimb.add_MPS_list(all_mps, compress_opts={'max_bond': max_bond})
        co = helper_quimb.check_orthog(init_guess)[0]
        init_guess[co].modify(apply=lambda x: x * 0.0)

    # init_guess = helper_quimb.add_MPS_list(all_mps, compress_opts={'max_bond': max_bond})

    print('init guess', init_guess.max_bond())
    helper_quimb.match_inner_inds(init_guess, ref_mps, inplace=True)
    E_comp.data = init_guess
    print('E comp max bond', E_comp.max_bond())
    # E_comp_copy = E_comp.copy()
    # exit()

    # nsites = 2
    # solver = CrossEvaluator(init_guess, terms, combine_terms_func=eval_func,
    #                         max_bond=max_bond, cutoff=cutoff)
    # solver.solve(nsites)

    out_E_comp = E_comp.evolve_tdmrg_new(dt, [], te_order=1, inplace=inplace,
                                         compress_config=compress_config, solver_type=LocalSolverType.TDCross,
                                         upwind_mpo_list={**pml_dict, **source_dict},
                                         upwind_func=upwind_func,
                                         verbose_plot=verbose_plot,)

    # print('diff', out_E_comp.distance(E_comp_copy))
    # exit()

    return out_E_comp


