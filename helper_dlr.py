"""Standard (orthogonal-projector) dynamical low-rank time integration.

Provides routines that evolve a low-rank tensor-network state in time by
projecting the time derivative onto the tangent space using the standard
orthogonal projector formulation of dynamical low-rank approximation (DLRA).
This module is outdated and has been superseded by the interpolative-DLR
routines.
"""
from setup_.configs import *
import scipy.linalg
import helper_quimb as helper
import helper_TE
import gridTN as GTN

if TYPE_CHECKING:
    from axis import Axis
    from gridTN import GridTN


def dlr_compute_time_derivative_subspace(dist_gtn: 'GridTN', deriv_mpos: Sequence['GridTN'], subspace: Sequence['Axis'],
                                         canonize=True, compress_level: int = 1, compress_opts_dict=None
                                         ) -> Optional['GridTN']:
    """ compute time derivative for subspace of dist_gtn
    """

    if canonize:
        dist_gtn = dist_gtn.canonize_axes(subspace, inplace=True)

    new_grid = dist_gtn.grid.get_subgrid(subspace)
    compress_opts_dict = {} if compress_opts_dict is None else compress_opts_dict

    ## advection terms
    proj_total: 'Optional[GridTN]' = None
    for mpo in deriv_mpos:
        proj_x = dist_gtn.project(mpo, proj_axes=subspace, new_grid=new_grid, canonize=False, compress=True)
        proj_total = proj_x.add_subgtn(proj_total, open_bc=False)
        # print('proj_x', helper.norm(proj_x.data))
        # print('proj_x', proj_x.data)

    if compress_level:
        if proj_total is not None:
            compress_opts = compress_opts_dict.get(compress_level, None)
            proj_total.compress(compress_opts=compress_opts, inplace=True)

    # print('deriv total GRID', helper.norm(proj_total).data, subspace)

    return proj_total


def dlr_compute_time_derivative_site(dist_gtn: 'GridTN', deriv_mpos: Sequence['GridTN'], site_ind: int, nsites=1,
                                     canonize=True, **kwargs) -> 'qtn.Tensor':
    """ compute projection of MPS onto site(s) [specified by nsites] or bond (right of site_ind) [nsites=0]
    """
    if canonize:
        if nsites == 0:
            dist_gtn.convert_to_USVT(canon_site=site_ind)
        else:
            dist_gtn.canonize(i=site_ind, inplace=True)

    proj_total: 'Optional[qtn.Tensor]' = None
    for mpo in deriv_mpos:
        if nsites == 0:
            proj_x = dist_gtn.project_bond(mpo, bond_ind=site_ind)
        else:
            proj_x = dist_gtn.project_site(mpo, site_ind, nsites=nsites)
        if proj_total is None:
            proj_total = proj_x
        else:
            proj_x.transpose_like_(proj_total)
            proj_total.modify(apply=lambda x: x + proj_x.data)

    return proj_total


def dlr_exact_site(dist_gtn: 'GridTN', dt, deriv_mpos: Sequence['GridTN'], site_ind: int, nsites=1,
                   inplace=False, canonize=True, compress_direction=1, compress_level=1, compress_opts_dict=None
                   ) -> Optional['GridTN']:
    """ compute projection of MPS onto site(s) [specified by nsites] or bond (right of site_ind) [nsites=0]
    """
    dist_gtn = dist_gtn if inplace else dist_gtn.copy()
    compress_opts_dict = {} if compress_opts_dict is None else compress_opts_dict
    compress_opts = compress_opts_dict.get(compress_level, None)

    if dist_gtn.data is None:
        return dist_gtn

    if canonize:
        if nsites == 0:
            dist_gtn.convert_to_USVT(canon_site=site_ind)
        else:
            dist_gtn.canonize(i=site_ind, inplace=True)

    proj_total: 'Optional[qtn.Tensor]' = None
    bonds_i, bonds_o = [], []
    for mpo in deriv_mpos:
        if nsites == 0:
            # print('dist gtn usvt', dist_gtn.data)
            proj_x, bonds_i, bonds_o = dist_gtn.project_op_bond(mpo, bond_ind=site_ind)
            print('proj x 0', proj_x, bonds_i, bonds_o)
        else:
            proj_x, bonds_i, bonds_o = dist_gtn.project_op_site(mpo, site_ind, nsites=nsites)
            print('roj_x', proj_x.norm())
        if proj_total is None:
            proj_total = proj_x
        else:
            proj_x.transpose_like_(proj_total)
            proj_total.modify(apply=lambda x: x + proj_x.data)

    proj_total.transpose(*bonds_i, *bonds_o, inplace=True)
    fuse_map = {'in': tuple(bonds_i), 'out': tuple(bonds_o)}
    proj_op = proj_total.fuse(fuse_map)
    # plt.figure()
    # plt.imshow(proj_op.data)
    # plt.colorbar()
    # plt.show()
    # print('proj op', proj_total.shape)
    proj_op.modify(apply=lambda x: scipy.linalg.expm(x * -dt))
    shape_i = [proj_total.ind_size(bi) for bi in bonds_i]
    shape_o = [proj_total.ind_size(bo) for bo in bonds_o]
    exp_total = proj_op.unfuse(fuse_map, {'in': shape_i, 'out': shape_o})

    reindex_map = {bo: bi for bo, bi in zip(bonds_o, bonds_i)}
    if nsites > 0:
        active_tens: 'qtn.Tensor' = qtn.tensor_contract(*dist_gtn.data[site_ind: site_ind + nsites], exp_total)
    else:
        active_tens: 'qtn.Tensor' = qtn.tensor_contract(dist_gtn.get_S_tensor(canon_site=site_ind), exp_total)
    active_tens.reindex(reindex_map, inplace=True)


    T1 = active_tens
    s_ind_range = range(site_ind, site_ind + nsites - 1) if compress_direction > 0 else \
        range(site_ind + nsites - 1, site_ind, -1)
    for s_ind in s_ind_range:
        rix, lix = dist_gtn.data[s_ind].filter_bonds(dist_gtn.data[s_ind + 1 * compress_direction])
        T2, T1 = T1.split(lix, get='tensors', absorb='right', **compress_opts)

        dist_gtn_tens = dist_gtn.data[s_ind]
        T2.transpose_like(dist_gtn_tens, inplace=True)
        dist_gtn_tens.modify(data=T2.data)

    if nsites > 0:
        s_ind = site_ind + nsites - 1 if compress_direction > 0 else site_ind
        dist_gtn_tens = dist_gtn.data[s_ind]
    else:
        dist_gtn_tens = dist_gtn.get_S_tensor(canon_site=site_ind)
    T1.transpose_like(dist_gtn_tens, inplace=True)
    dist_gtn_tens.modify(data=T1.data)

    # print('dist_gtn norm', dist_gtn.norm(is_sqrt=True))

    return dist_gtn


def dlr_euler_subspace(dist_gtn: 'GridTN', dt, deriv: 'GridTN', projected_axes: Sequence['Axis'],
                       inplace=False, adapt=False, canonize=True, compress_level=1, compress_opts_dict=None,
                       ) -> Optional['GridTN']:
    """ df/dt = ...
        projected onto qtn manifold (by x, v)
        projected axes:  axes that remain after projection.
    """
    dist_gtn = dist_gtn if inplace else dist_gtn.copy()
    compress_opts_dict = {} if compress_opts_dict is None else compress_opts_dict

    if dist_gtn.data is None:
        return dist_gtn

    if canonize:
        dist_gtn = dist_gtn.canonize_axes(projected_axes, inplace=True, scale=False)

    ## update data (forward)
    deriv = deriv.scalar_multiply(-dt, inplace=False)  # df/dt + mpos[f] = 0
    dist_gtn = dist_gtn.add_subgtn(deriv, open_bc=adapt, inplace=True, compress=compress_level,
                                   compress_opts=compress_opts_dict.get(compress_level, None))
    ## just compress active area, return to valid canonical form (canonical within subspace)

    return dist_gtn


def dlr_euler_bond(dist_gtn: 'GridTN', dt, deriv: 'qtn.Tensor', bond_ind: int, inplace=False, canonize=True, **kwargs
                   ) -> Optional['GridTN']:
    """ df/dt = ...
        projected onto qtn manifold defined by X, V in canonical form (TE of PX * PV (df/dt))
        (actual bond is specified by bond_ind)
        for backwards TE, set -dt
    """
    dist_gtn = dist_gtn if inplace else dist_gtn.copy()

    if dist_gtn.data is None:
        return dist_gtn

    if canonize:
        dist_gtn.convert_to_USVT(canon_site=bond_ind)

    tens_S = dist_gtn.get_S_tensor()
    deriv.transpose_like(tens_S, inplace=True)
    tens_S.modify(apply=lambda data: data - dt * deriv.data * 10 ** (-dist_gtn.exponent))

    dist_gtn.convert_from_USVT()
    return dist_gtn


def dlr_euler_site(dist_gtn: 'GridTN', dt, deriv: 'qtn.Tensor', site_ind: int, nsites=1, inplace=False, canonize=True,
                   compress_direction=1, compress_level=1, compress_opts_dict=None
                   ) -> Optional['GridTN']:
    """ df/dt = ...
        projected onto qtn manifold defined remaining sites
        for backwards TE, set -dt
        direction: canonicalization direction of output (+1: left canon, -1: right canon)
    """
    dist_gtn = dist_gtn if inplace else dist_gtn.copy()
    compress_opts_dict = {} if compress_opts_dict is None else compress_opts_dict
    compress_opts = compress_opts_dict.get(compress_level, None)

    if dist_gtn.data is None:
        return dist_gtn

    if canonize:
        dist_gtn.canonize(i=site_ind, inplace=True, scale=False)

    active_tens: 'qtn.Tensor' = dist_gtn.data[site_ind: site_ind + nsites].contract_tags(all)
    deriv.transpose_like(active_tens, inplace=True)
    # deriv.modify(apply=lambda data: data * 10**(-dist_gtn.exponent))
    active_tens.modify(apply=lambda data: data - dt * deriv.data * 10**(-dist_gtn.exponent))

    T1 = active_tens
    s_ind_range = range(site_ind, site_ind + nsites - 1) if compress_direction > 0 else \
        range(site_ind + nsites - 1, site_ind, -1)
    # print('s ind range', s_ind_range, compress_direction)
    for s_ind in s_ind_range:
        # print('s_ind', s_ind)
        rix, lix = dist_gtn.data[s_ind].filter_bonds(dist_gtn.data[s_ind + 1 * compress_direction])
        T2, T1 = T1.split(lix, get='tensors', absorb='right', **compress_opts)

        dist_gtn_tens = dist_gtn.data[s_ind]
        T2.transpose_like(dist_gtn_tens, inplace=True)
        dist_gtn_tens.modify(data=T2.data)

    s_ind = site_ind + nsites - 1 if compress_direction > 0 else site_ind
    # print('s_ind', s_ind)
    dist_gtn_tens = dist_gtn.data[s_ind]
    T1.transpose_like(dist_gtn_tens, inplace=True)
    dist_gtn_tens.modify(data=T1.data)

    # print('euler site check orthog', site_ind)
    # print(helper.check_orthog(dist_gtn.data))

    return dist_gtn


def dlr_subspace_time_evolution(dist_gtn: 'GridTN', dt, deriv_mpos, projected_axes: Sequence['Axis'], te_order = 4,
                                deriv0: Optional['GridTN'] = None, inplace=False, canonize=True, adapt=False,
                                compress_level = 1, compress_level_2 = 4, compress_opts_dict = None) -> 'GridTN':
    """ time integration of projected subspace, as dictated by te_order. limited to RK TE methods
    """
    dist_gtn = dist_gtn if inplace else dist_gtn.copy()
    compress_opts_dict = {} if compress_opts_dict is None else compress_opts_dict

    if canonize:
        dist_gtn = dist_gtn.canonize_axes(projected_axes, inplace=True)

    def euler_func(dist_gtn_, dt_, deriv0=None, inplace=False, compress_level=1, adapt=adapt, **kwargs):
        return dlr_euler_subspace(dist_gtn_, dt_, deriv0, projected_axes, canonize=False, inplace=inplace,
                                  adapt=adapt, compress_level=compress_level, compress_opts_dict=compress_opts_dict)

    def deriv_func(dist_gtn_, compress_level=1, **kwargs):
        return dlr_compute_time_derivative_subspace(dist_gtn_, deriv_mpos, projected_axes, canonize=False,
                                                 compress_level=compress_level, compress_opts_dict=compress_opts_dict )

    def add_func(deriv1: 'GridTN', deriv2: 'GridTN', compress_level=1, inplace=False, **kwargs):
        ## function for adding derivatives together
        return deriv1.add_subgtn(deriv2, open_bc=False, inplace=inplace, compress=compress_level,
                                 compress_opts=compress_opts_dict.get(compress_level))

    if deriv0 is None:
        deriv0 = deriv_func(dist_gtn, canonize=canonize, compress_level=compress_level_2,
                            compress_opts_dict=compress_opts_dict)

    out = helper_TE.time_integration(dist_gtn, dt, euler_func, deriv_func, add_func, GTN.scalar_multiply,
                                      te_order=te_order, deriv0=deriv0, compress_level=compress_level,
                                      compress_opts_dict=compress_opts_dict)
    dist_gtn.data = out.data
    return dist_gtn


def dlr_site_time_evolution(dist_gtn: 'GridTN', dt, deriv_mpos, site_ind, nsites=1, te_order = 4,
                            deriv0: Optional['GridTN'] = None, inplace=False, canonize=True, compress_direction = 1,
                            compress_level = 1, compress_level_2 = 4, compress_opts_dict = None) -> 'GridTN':
    """ time integration of specific site, as dictated by te_order. limited to RK TE methods
    """
    dist_gtn = dist_gtn if inplace else dist_gtn.copy()

    if dist_gtn.data is None:
        return dist_gtn

    if canonize:
        dist_gtn.canonize(i=site_ind, inplace=True, scale=False)

    if te_order == 0:
        return dlr_exact_site(dist_gtn, dt, deriv_mpos, site_ind, nsites=nsites, inplace=True, canonize=False,
                              compress_direction=compress_direction, compress_level=compress_level,
                              compress_opts_dict=compress_opts_dict)

    def euler_func(dist_gtn_, dt_, deriv0=None, compress_level=1, inplace=False, **kwargs):
        return dlr_euler_site(dist_gtn_, dt_, deriv0, site_ind, nsites = nsites, canonize=False, inplace=inplace,
                              compress_direction=compress_direction, compress_level=compress_level,
                              compress_opts_dict=compress_opts_dict)

    def deriv_func(dist_gtn_, **kwargs) -> 'qtn.Tensor':
        return dlr_compute_time_derivative_site(dist_gtn_, deriv_mpos, site_ind, nsites=nsites, canonize=False)


    def add_func(t1, t2, inplace=False, **kwargs):
        return helper.add_tensors(t1, t2, inplace=inplace)

    if deriv0 is None:
        deriv0 = deriv_func(dist_gtn, canonize=canonize, compress_level=compress_level_2,
                            compress_opts_dict=compress_opts_dict)

    out = helper_TE.time_integration(dist_gtn, dt, euler_func, deriv_func, add_func, helper.scale_tensors,
                                      te_order=te_order, deriv0=deriv0, compress_level=compress_level,
                                      compress_opts_dict=compress_opts_dict)
    dist_gtn.data = out.data
    return dist_gtn


def dlr_bond_time_evolution(dist_gtn: 'GridTN', dt, deriv_mpos, bond_ind, te_order = 4,
                            deriv0: Optional['GridTN'] = None, inplace=False, canonize=True, compress_direction = 1,
                            compress_level = 1, compress_level_2 = 4, compress_opts_dict = None) -> 'GridTN':
    """ time integration of specific site, as dictated by te_order. limited to RK TE methods
    """
    dist_gtn = dist_gtn if inplace else dist_gtn.copy()

    if dist_gtn.data is None:
        return dist_gtn

    if canonize:
        dist_gtn.convert_to_USVT(canon_site=bond_ind, inplace=True)

    if te_order == 0:
        return dlr_exact_site(dist_gtn, dt, deriv_mpos, bond_ind, nsites=0, inplace=True, canonize=False,
                              compress_direction=compress_direction, compress_level=compress_level,
                              compress_opts_dict=compress_opts_dict)


    def euler_func(dist_gtn_, dt_, deriv0=None, inplace=False, **kwargs):
        return dlr_euler_bond(dist_gtn_, dt_, deriv0, bond_ind, canonize=False, inplace=inplace,)

    def deriv_func(dist_gtn_, **kwargs) -> 'qtn.Tensor':
        return dlr_compute_time_derivative_site(dist_gtn_, deriv_mpos, bond_ind, nsites=0, canonize=False)

    def add_func(t1, t2, inplace=inplace, **kwargs):
        return helper.add_tensors(t1, t2, inplace=inplace)

    if deriv0 is None:
        deriv0 = deriv_func(dist_gtn, canonize=canonize, compress_level=compress_level_2,
                            compress_opts_dict=compress_opts_dict)

    out = helper_TE.time_integration(dist_gtn, dt, euler_func, deriv_func, add_func, helper.scale_tensors,
                                      te_order=te_order, deriv0=deriv0, compress_level=compress_level,
                                      compress_opts_dict=compress_opts_dict)
    dist_gtn.data = out.data
    # dist_gtn.convert_from_USVT(inplace=True)
    return dist_gtn
