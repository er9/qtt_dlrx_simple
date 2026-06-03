""" v2 works.
    this is a cleaner version + elemental multiplication of f(x) * g(y)
    where f, g are element-wise operations
"""
import pdb

import helper_quimb
from setup_.configs import *
import time
# from scipy import linalg
import scipy.linalg as linalg
import quimb.tensor as qtn
# import helper_quimb as helper
from local_solvers.defaults import *

MPO_type = Union['qtn.MatrixProductOperator']
MPS_type = Union['qtn.MatrixProductState']

from local_solvers.mps_classes import MPS
from local_solvers.helper_cross_2 import get_inds, iso_left_inds, iso_right_inds, plot_submat


###########################
###    MPS methods      ###
###########################

class CrossSolver(Enum):
    MAXVOL = 'maxvol'
    DEIM = 'deim'

DEFAULT_SOLVER = CrossSolver.DEIM
# DEFAULT_SOLVER = CrossSolver.MAXVOL


def cur_split(tens: qtn.Tensor, inds_r: Sequence[int], cu_bond=None, ur_bond=None):
    """ 2D data
    """
    ind1, ind2 = tens.inds

    if cu_bond is None:
        cu_bond = ind1 + '_'
    if ur_bond is None:
        ur_bond = ind2 + '_'

    data = tens.data
    q, r = np.linalg.qr(data)
    submat = q[inds_r, :] @ r
    u = np.linalg.pinv(submat)

    c = qtn.Tensor(data=q, inds=[ind1, cu_bond],)
    u = qtn.Tensor(data=u, inds=[cu_bond, ur_bond], )
    r = qtn.Tensor(data=submat, inds=[ur_bond, ind2],)
    return c, u, r

# def tensor_xr(tens: qtn.Tensor, phys_inds: list[str], right_inds: list[str],
#               inds_r: Sequence[int], bond_ind='xx',):
#     """ 3D data --> 2D data
#     """




def check_left_select_inds(mpx: 'qtn.MatrixProductState', select_inds: dict[int, Sequence[int]],
                           select_tens: dict[int, qtn.Tensor], select_tens_inv: dict[int, qtn.Tensor]) -> int:
    """ returns index of first tensor that is not left canonical
    """
    mpx = mpx.copy()
    L = len(mpx) if isinstance(mpx, (list, tuple)) else mpx.L

    for i in range(L):
        prev_tens = select_tens.get(i - 1, None)
        if prev_tens is None:
            tens = mpx[i].copy()
        else:
            tens = qtn.tensor_contract(mpx[i], prev_tens)
            tens.transpose_like(mpx[i], inplace=True)
            tens.modify(inds=mpx[i].inds)

        bonds_l, bonds_r = iso_left_inds(mpx, i, [mpx.site_ind_id])

        tens = tens.fuse({'eff': bonds_l}) # bond_p + bond_l})
        tens.transpose('eff', *bonds_r, inplace=True)

        try:
            if len(select_inds[i]) == 0:
                raise KeyError
            selected_data = tens.data[select_inds[i], :]
            sel_tens = select_tens[i]
            sel_tens.transpose(bonds_r[0] + '_x', bonds_r[0], inplace=True)
            err = np.linalg.norm(selected_data - sel_tens.data)

            sel_tens_inv = select_tens_inv[i]
            sel_tens_inv.transpose(bonds_r[0], bonds_r[0] + '_x', inplace=True)
            r = sel_tens.shape[0]
            err1 = np.linalg.norm(sel_tens_inv.data @ sel_tens.data - np.eye(r))
            # print('inverse err1', err1)
            # err2 = np.linalg.norm(sel_tens.data @ sel_tens_inv.data - np.eye(r))
            # print('inverse err2', err2)
        except (KeyError, ValueError, IndexError):
            print(f'left: select ind not defined or incompatible for {i}')
            return i

        if err > 1.0e-8 or err1 > 1.0e-08:
            print('check sel inds not left orthog', i, err) #, env)
            print('inverse issues', i, err1)
            if err > 1.0e-5 or err1 > 1.0e-05:
                break

    return i


def check_right_select_inds(mpx: 'qtn.MatrixProductState', select_inds: dict[int, Sequence[int]],
                            select_tens: dict[int, qtn.Tensor], select_tens_inv: dict[int, qtn.Tensor]) -> int:
    """ returns index of first tensor that is not left canonical
    """
    mpx = mpx.copy()
    L = len(mpx) if isinstance(mpx, (list, tuple)) else mpx.L
    for i in range(L-1, -1,-1):
        prev_tens = select_tens.get(i+1, None)
        if prev_tens is None:
            tens = mpx[i].copy()
        else:
            tens = qtn.tensor_contract(mpx[i], prev_tens)
            tens.transpose_like(mpx[i], inplace=True)
            tens.modify(inds=mpx[i].inds)

        bonds_r, bonds_l = iso_right_inds(mpx, i, [mpx.site_ind_id])

        tens = tens.fuse({'eff': bonds_l})  # bond_p + bond_l})
        tens.transpose('eff', *bonds_r, inplace=True)

        try:
            if len(select_inds[i]) == 0:
                raise KeyError
            selected_data = tens.data[select_inds[i], :]
            sel_tens = select_tens[i]
            sel_tens.transpose(bonds_r[0] + '_x', bonds_r[0], inplace=True)
            err = np.linalg.norm(selected_data - sel_tens.data)

            sel_tens_inv = select_tens_inv[i]
            sel_tens_inv.transpose(bonds_r[0], bonds_r[0] + '_x', inplace=True)
            r = sel_tens.shape[0]
            err1 = np.linalg.norm(sel_tens_inv.data @ sel_tens.data - np.eye(r))
            # print('inverse err1', err1)
            # err2 = np.linalg.norm(sel_tens.data @ sel_tens_inv.data - np.eye(r))
            # print('inverse err2', err2)
        except (KeyError, ValueError, IndexError):
            print(f'right: select ind not defined or incompatible for {i}')
            return i

        if err > 1.0e-8 or err1 > 1.0e-08:
            print('check sel inds not right orthog', i, err)  # , env)
            print('inverse issues', i, err1)
            if err > 1.0e-5 or err1 > 1.0e-5:
                break

    return i


# def get_inds_from_tens_list(tens_list, site_ind_ids: Sequence[str], site_inds: Sequence[int],
#                             solver_type=DEFAULT_SOLVER, prev_tens=None):
#
#     ## length of list must be greater than two
#     prev_tens = prev_tens.copy() if prev_tens is not None else None
#     select_inds = []
#     for i in range(len(tens_list)):
#         tens_i = tens_list[i]
#         tens = tens_i.copy() if prev_tens is None else qtn.tensor_contract(prev_tens, tens_i)
#
#         phys_inds = [site_ind_id.format(site_inds[i]) for site_ind_id in site_ind_ids]
#         if i < len(tens_list) - 1:
#             right_inds, bonds_L = tens.filter_bonds(tens_list[i + 1])
#             left_inds = [b for b in bonds_L if b not in phys_inds]
#         else:
#             left_inds, bonds_R = tens.filter_bonds(T1)
#             right_inds = [b for b in bonds_R if b not in phys_inds]
#
#         inds_r = select_rows(tens_list[i], right_inds[0], left_inds[0], phys_inds[0])
#         T1, T2 = tensor_xr(tens, phys_inds, right_inds, inds_r)
#         select_inds += inds_r
#         prev_tens = T2
#
#     return select_inds, T2
#
#
# def canonize_tens_list(tens_list: list['qtn.Tensor'], site_ind_ids: Sequence[str], site_inds: Sequence[int],
#                        inplace=True, solver_type=DEFAULT_SOLVER):
#
#     tens_list = helper_quimb.canonize_tens_list(tens_list, inplace=inplace)
#     sel_inds, sel_tens = get_inds_from_tens_list(tens_list, site_ind_ids, site_inds, solver_type=solver_type)
#     return tens_list, sel_inds, sel_tens



def update_1site(mps: MPS, left_site_pos: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection',
                 max_bond:int = None, cutoff:float=CUTOFF, solver_type=DEFAULT_SOLVER, version=None,
                 verbose_plot=False):
    """ update ket, bra with new_site; list of sites --> target these separately.
        i: int of mps site
        get row/column selected inds
        inplace operation
    """
    version = flags.get('version', 'X') if version is None else version
    # mps_copy = mps.copy()
    print('mixed update 1 site', 'max_bond', max_bond, 'cutoff', cutoff)

    if not isinstance(site_i, (list, tuple)):
        site_i = [site_i]

    ### project site_i onto orthogonal basis (currently it's in element-wise basis
    if version != 'G':      ## if 'G', already in basis representation

        # if left_site_pos == mps.L-2:
        #     for ix, tmp in enumerate(site_i):
        #         tmp = tmp.reindex({ind: ind[:-2] for ind in tmp.inds if ind[-1] =='x'})
        #         plot_submat(mps, left_site_pos, 1, tmp, plt_title=f'update 1 site {ix}')

        proj_sites = []
        for tens in site_i:
            out = convert_elementwise_to_basis(mps, tens, left_site_pos, 1)
            proj_sites += [out]
        site_i = proj_sites

        if verbose_plot:
            tmp1 = mps.copy()
            ref_data = helper_quimb.to_dense(mps)
            for tens_g in proj_sites:
                tmp1[left_site_pos].transpose_like(tens_g, inplace=True)
                tmp1[left_site_pos].modify(data=tens_g.data)
                tmp1_data = helper_quimb.to_dense(tmp1)
                print('diff', np.linalg.norm(tmp1_data - ref_data))
                plt.figure()
                plt.plot((tmp1_data - ref_data).reshape(-1))
                plt.show()

        # if left_site_pos == mps.L // 2:
        #     tmp = mps.copy()
        #     for t in site_i:
        #         t.transpose_like(tmp[left_site_pos], inplace=True)
        #         tmp[left_site_pos].modify(data=t.data)
        #
        #         plt.figure()
        #         plt.plot(np.real(helper_quimb.to_dense(tmp).reshape(-1)))
        #         plt.plot(np.real(helper_quimb.to_dense(mps).reshape(-1)), '--')
        #
        #         plt.figure()
        #         plt.plot(np.imag(helper_quimb.to_dense(tmp).reshape(-1)))
        #         plt.plot(np.imag(helper_quimb.to_dense(mps).reshape(-1)), '--')
        #         plt.show()


    from local_solvers.helper_dmrg_loc import update_1site as update_1site_dmrg
    print('cutoff', cutoff)
    print('left site pos', left_site_pos)
    print('check orthog', check_orthog(mps))
    update_1site_dmrg(mps, left_site_pos, site_i, direction=direction, max_bond=max_bond, cutoff=cutoff)
    out = mps

    # print('update 1 site distance', helper_quimb.distance(mps, mps_copy ))

    ind1 = left_site_pos
    ind2 = left_site_pos + direction
    if ind2 < 0 or ind2 > out.L - 1:
        return out  ## no decimation is performed

    ## update select inds
    left_inds, phys_inds, right_inds = get_inds(out, ind1)
    if direction > 0:
        prev_tens = out.select_tens.get(ind1 - direction)
        tmp_left_ind = left_inds[0] + '_x' if ind1 > 0 else None
        tens = out[ind1].copy() if ind1 == 0 else qtn.tensor_contract(prev_tens, out[ind1])
        sel_inds, TC, TR_inv, TR = tensor_get_submat(tens, tmp_left_ind, right_inds[0], phys_inds[0],
                                                     solver_type=solver_type, oversample=False)
        # T1, T2 = tensor_xr(tens, [phys_ind], right_inds, sel_inds)

    else:
        prev_tens = out.select_tens.get(ind1 - direction)
        tmp_right_ind = right_inds[0] + '_x' if ind1 < out.L - 1 else None
        tens = out[ind1].copy() if ind1 == out.L - 1 else qtn.tensor_contract(prev_tens, out[ind1])
        sel_inds, TC, TR_inv, TR = tensor_get_submat(tens, tmp_right_ind, left_inds[0], phys_inds[0],
                                                     solver_type=solver_type, oversample=False)
        # T1, T2 = tensor_xr(tens, [phys_ind], left_inds, sel_inds)
    out.select_inds[ind1] = sel_inds
    out.select_inds.pop(ind2)
    out.select_tens[ind1] = TR
    out.select_tens.pop(ind2)
    out.select_tens_inv[ind1] = TR_inv
    out.select_tens_inv.pop(ind2)

    out._cur_orthog = left_site_pos + direction

    return out


def update_2site(mps: MPS, left_site_pos: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection',
                 max_bond: int = None, cutoff: float = CUTOFF, solver_type=DEFAULT_SOLVER, version=None):
    """ update ket, bra with new_site; list of sites --> target these separately.
        i: int of mps site
        inplace operation
    """
    version = flags.get('version', 'X') if version is None else version
    if not isinstance(site_i, (list, tuple)):
        site_i = [site_i]

    if version != 'G':

        # if left_site_pos == mps.L-2:
        #     for ix, tmp in enumerate(site_i):
        #         tmp = tmp.reindex({ind: ind[:-2] for ind in tmp.inds if ind[-1] =='x'})
        #         plot_submat(mps, left_site_pos, 2, tmp, plt_title=f'update 2 site {ix}')

        proj_sites = []
        for tens in site_i:
            out = convert_elementwise_to_basis(mps, tens, left_site_pos, 2)
            proj_sites += [out]
        site_i = proj_sites

    print('version', version)
    print('update 2 site', site_i)

    copy_mps = mps.copy()

    from local_solvers.helper_dmrg_loc import update_2site as update_2site_dmrg
    print('dmrg update 2-site update', max_bond, cutoff)
    update_2site_dmrg(mps, left_site_pos, site_i, direction=direction, max_bond=max_bond, cutoff=cutoff)
    out = mps

    print('difference', helper_quimb.distance(mps, copy_mps))

    # plt.figure()
    # plt.plot(np.real(helper_quimb.to_dense(copy_mps).reshape(-1)))
    # plt.plot(np.real(helper_quimb.to_dense(mps).reshape(-1)), '--')
    # plt.title('update 2 site real')
    #
    # plt.figure()
    # plt.plot(np.imag(helper_quimb.to_dense(copy_mps).reshape(-1)))
    # plt.plot(np.imag(helper_quimb.to_dense(mps).reshape(-1)), '--')
    # plt.title('update 2 site imag')
    # plt.show()

    if direction > 0:
        ind1, ind2 = left_site_pos, left_site_pos + 1
    else:
        ind1, ind2 = left_site_pos + 1, left_site_pos

    if ind2 < 0 or ind2 > out.L - 1:
        return out  ## no decimation is perfromed

    ## update select inds ##
    left_inds, phys_inds, right_inds = get_inds(out, ind1)
    if direction > 0:
        prev_tens = out.select_tens[ind1 - 1] if ind1 > 0 else None
        tens = out[ind1].copy() if prev_tens is None else qtn.tensor_contract(prev_tens, out[ind1])
        tmp_left_ind = left_inds[0] + '_x' if len(left_inds) == 1 else None
        sel_inds, TC, TR_inv, TR = tensor_get_submat(tens, tmp_left_ind, right_inds[0], phys_inds[0], solver_type=solver_type)
        # T1, T2 = tensor_xr(tens, [phys_ind], right_inds, sel_inds)
    else:
        prev_tens = out.select_tens[ind1 + 1] if ind1 < out.L - 1 else None
        tens = out[ind1].copy() if prev_tens is None else qtn.tensor_contract(prev_tens, out[ind1])
        tmp_right_ind = right_inds[0] + '_x' if len(right_inds) == 1 else None
        sel_inds, TC, TR_inv, TR = tensor_get_submat(tens, tmp_right_ind, left_inds[0], phys_inds[0], solver_type=solver_type)
        # T1, T2 = tensor_xr(tens, [phys_ind], left_inds, sel_inds)
    out.select_inds[ind1] = sel_inds
    out.select_inds.pop(ind2)
    out.select_tens[ind1] = TR
    out.select_tens.pop(ind2)
    out.select_tens_inv[ind1] = TR_inv
    out.select_tens_inv.pop(ind2)
    return out


def update_ket(mps: 'MPS', tensors: Union[qtn.Tensor, Sequence[qtn.Tensor]], i: int, nsites: int,
               direction: SweepDirection, max_bond: int = None, cutoff: float = None, version=None,
               verbose_plot=False):
    """ update ket with corresponding (targeting) tensors
        update mps_list[i] with tensors[i][0] (ideally the original tensor if just doing decimation)
        target all tensors in "tensors" list; shared across all "mps"
    """
    version = flags.get('version', 'X') if version is None else version
    if nsites == 1:
        return update_1site(mps, i, tensors, direction, max_bond=max_bond, cutoff=cutoff, version=version,
                            verbose_plot=verbose_plot)
    elif nsites == 2:
        left_site_pos = i - 1 if direction < 0 else i
        return update_2site(mps, left_site_pos, tensors, direction, max_bond=max_bond, cutoff=cutoff, version=version)
    else:
        raise ValueError


def update_and_replace_2site(mps: MPS, left_site_pos: int, site_tens: qtn.Tensor, direction: 'SweepDirection',
                             max_bond: int = None, cutoff: float = CUTOFF, solver_type=DEFAULT_SOLVER, version=None):
    """ update ket, bra with new_site; list of sites --> target these separately.
        i: int of mps site
        inplace operation
    """
    version = flags.get('version', 'X') if version is None else version
    from local_solvers.helper_dmrg_loc import tensor_svd

    if version != 'G':

        # if left_site_pos == mps.L-2:
        #     tmp = site_tens.reindex({ind: ind[:-2] for ind in site_tens.inds if ind[-1] =='x'})
        #     plot_submat(mps, left_site_pos, 2, tmp, plt_title=f'update+replace 2 site')

        site_tens = convert_elementwise_to_basis(mps, site_tens, left_site_pos, 2)


    copy_mps = mps.copy()

    print('version', version)
    # print('update + replace 2 site', site_tens)

    # i = left_site_pos if direction > 0 else left_site_pos + 1
    if direction > 0:
        ind1, ind2 = left_site_pos, left_site_pos + 1
    else:
        ind1, ind2 = left_site_pos + 1, left_site_pos

    x_ind = mps.bond(ind1, ind2)
    left_inds = [ind for ind in mps[ind1].inds if ind != x_ind]
    q, r = tensor_svd(site_tens, left_inds, absorb='right', max_bond=max_bond,
                      cutoff=(CUTOFF if cutoff is None else cutoff),
                      bond_ind=x_ind)

    # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
    q.transpose_like(mps[ind1], inplace=True)
    mps[ind1].modify(data=q.data)

    r.transpose_like(mps[ind2], inplace=True)
    mps[ind2].modify(data=r.data)

    print('difference', helper_quimb.distance(mps, copy_mps))

    # plt.figure()
    # plt.plot(np.real(helper_quimb.to_dense(mps).reshape(-1)), '--')
    # plt.plot(np.real(helper_quimb.to_dense(copy_mps).reshape(-1)))
    # plt.title('update 2 site real')
    #
    # plt.figure()
    # plt.plot(np.imag(helper_quimb.to_dense(mps).reshape(-1)), '--')
    # plt.plot(np.imag(helper_quimb.to_dense(copy_mps).reshape(-1)))
    # plt.title('update 2 site imag')
    # plt.show()

    out = mps
    # if ind2 < 0 or ind2 > out.L - 1:
    #     return out  ## no decimation is perfromed

    ## update select inds ##
    left_inds, phys_inds, right_inds = get_inds(out, ind1)
    if direction > 0:
        prev_tens = out.select_tens[ind1 - 1] if ind1 > 0 else None
        tens = out[ind1].copy() if prev_tens is None else qtn.tensor_contract(prev_tens, out[ind1])
        tmp_left_ind = left_inds[0] + '_x' if len(left_inds) == 1 else None
        sel_inds, TC, TR_inv, TR = tensor_get_submat(tens, tmp_left_ind, right_inds[0], phys_inds[0], solver_type=solver_type)
        # T1, T2 = tensor_xr(tens, [phys_ind], right_inds, sel_inds)
    else:
        prev_tens = out.select_tens[ind1 + 1] if ind1 < out.L - 1 else None
        tens = out[ind1].copy() if prev_tens is None else qtn.tensor_contract(prev_tens, out[ind1])
        tmp_right_ind = right_inds[0] + '_x' if len(right_inds) == 1 else None
        sel_inds, TC, TR_inv, TR = tensor_get_submat(tens, tmp_right_ind, left_inds[0], phys_inds[0], solver_type=solver_type)
        # T1, T2 = tensor_xr(tens, [phys_ind], left_inds, sel_inds)
    out.select_inds[ind1] = sel_inds
    out.select_inds.pop(ind2)
    out.select_tens[ind1] = TR
    out.select_tens.pop(ind2)
    out.select_tens_inv[ind1] = TR_inv
    out.select_tens_inv.pop(ind2)
    return out



def convert_elementwise_to_basis(proj_mps, site_tens, left_site_pos: int, nsites: int):
    """ project element-wise tensors onto bra basis
    """
    if nsites == 0:
        return convert_elementwise_to_basis_bond(proj_mps, site_tens, left_site_pos)

    b_left_inv = proj_mps.select_tens_inv.get(left_site_pos - 1, None)
    b_right_inv = proj_mps.select_tens_inv.get(left_site_pos + nsites, None)

    l_bond = proj_mps.bond(left_site_pos, left_site_pos - 1) if left_site_pos > 0 else None
    r_bond = proj_mps.bond(left_site_pos + nsites, left_site_pos + nsites - 1) \
                    if left_site_pos + nsites < proj_mps.L else None

    # if b_left_inv is not None:
    #     b_left_inv = b_left_inv.transpose(l_bond + '_', l_bond + '_x', inplace=False)
    #     b_left_inv.modify(inds=(l_bond + '_', l_bond))
    #
    # if b_right_inv is not None:
    #     b_right_inv = b_right_inv.transpose(r_bond + '_', r_bond + '_x', inplace=False)
    #     b_right_inv.modify(inds=(r_bond + '_', r_bond))

    tmp = [site_tens]
    if b_left_inv is not None:  tmp += [b_left_inv]
    if b_right_inv is not None:  tmp += [b_right_inv]
    # print('site tens', site_tens)
    # print('b left', b_left_inv)
    # print('b right', b_right_inv)
    # pdb.set_trace()
    out = qtn.tensor_contract(*tmp)

    # if l_bond is not None:
    #     out.reindex({l_bond + '_': l_bond}, inplace=True)
    # if r_bond is not None:
    #     out.reindex({r_bond + '_': r_bond}, inplace=True)

    return out


def convert_basis_to_elementwise(proj_mps, site_tens, left_site_pos: int, nsites: int):
    """ project basis tensors onto element-wise representation
    """
    if nsites == 0:
        return convert_basis_to_elementwise_bond(proj_mps, site_tens, left_site_pos)

    b_left = proj_mps.select_tens.get(left_site_pos - 1, None)
    b_right = proj_mps.select_tens.get(left_site_pos + nsites, None)

    # print('basis to selement')
    # print('site tens', site_tens)
    # print('b left', b_left)
    # print('b right', b_right)

    l_bond = proj_mps.bond(left_site_pos, left_site_pos - 1) if left_site_pos > 0 else None
    r_bond = proj_mps.bond(left_site_pos + nsites, left_site_pos + nsites - 1) \
                    if left_site_pos + nsites < proj_mps.L else None

    # print('l bond', l_bond, 'r_bond', r_bond)

    # if b_left is not None:
    #     b_left = b_left.transpose(l_bond + '_', l_bond + '_tmp', inplace=False)
    #     b_left.modify(inds=(l_bond + '_', l_bond))
    #
    # if b_right is not None:
    #     b_right = b_right.transpose(r_bond + '_', r_bond + '_tmp', inplace=False)
    #     b_right.modify(inds=(r_bond + '_', r_bond))

    tmp = [site_tens]
    if b_left is not None:  tmp += [b_left]
    if b_right is not None:  tmp += [b_right]
    # print('site tens', site_tens)
    # print('b left', b_left_inv)
    # print('b right', b_right_inv)
    # pdb.set_trace()
    out = qtn.tensor_contract(*tmp)
    # if l_bond is not None:
    #     out.reindex({l_bond + '_x': l_bond}, inplace=True)
    # if r_bond is not None:
    #     out.reindex({r_bond + '_x': r_bond}, inplace=True)

    return out


def convert_elementwise_to_basis_bond(proj_mps, site_tens, left_site_pos: int):
    """ project element-wise tensors onto bra basis, nsites = 0
    """
    b_left_inv = proj_mps.select_tens_inv.get(left_site_pos, None)
    b_right_inv = proj_mps.select_tens_inv.get(left_site_pos + 1, None)

    x_bond = proj_mps.bond(left_site_pos, left_site_pos + 1)
    b_left_inv = b_left_inv.reindex({x_bond + '_x' : x_bond + '_x_L', x_bond: x_bond + '_L'})
    b_right_inv = b_right_inv.reindex({x_bond + '_x': x_bond + '_x_R', x_bond: x_bond + '_R'})

    tmp = [site_tens]
    if b_left_inv is not None:  tmp += [b_left_inv]
    if b_right_inv is not None:  tmp += [b_right_inv]
    out = qtn.tensor_contract(*tmp)

    return out


def convert_basis_to_elementwise_bond(proj_mps, site_tens, left_site_pos: int):
    """ project basis tensors onto element-wise representation
    """

    b_left = proj_mps.select_tens.get(left_site_pos, None)
    b_right = proj_mps.select_tens.get(left_site_pos + 1, None)

    x_bond = proj_mps.bond(left_site_pos, left_site_pos + 1)
    b_left = b_left.reindex({x_bond: x_bond + '_L', x_bond + '_x': x_bond + '_x_L'})
    b_right = b_right.reindex({x_bond: x_bond + '_R', x_bond + '_x': x_bond + '_x_R'})

    tmp = [site_tens]
    if b_left is not None:  tmp += [b_left]
    if b_right is not None:  tmp += [b_right]
    out = qtn.tensor_contract(*tmp)

    return out


#####################################
### cross interpolation (max vol) ###
#####################################

def maxvol_inds(A: 'np.ndarray', max_iters=DEFAULT_MAX_TOT_ITER, conv_tol=DEFAULT_CONV_TOL, do_qr=True,
                inds_r_guess: Sequence[int] = None):
    """
    modified so that even if n < r, environment will be identity
    :param A:  n x r matrix. want to find r x r submatrix of A
    :param max_iters:
    :param conv_tol:
    :param do_qr:
    :return: inds to obtain submatrix of A
    """
    n, r = A.shape  ## A is 2D matrix: alpha_{i-1} * d_{i} x alpha_{i}
    # print('A shape', A.shape)
    if n <= r:
        r = n

    do_qr = True
    if do_qr:
        Q, R_dmp = np.linalg.qr(A)
    else:
        Q = A.copy()

    if inds_r_guess is None:
        p, l_dmp, u_dmp = linalg.lu(Q, p_indices=True)  # A = P L U (permutation, lower tri with unit diagonal, upper tri)
        p = np.argsort(p)       # so that P A = L U, as in matlab
        inds = p[:r]  ## p is a list of indices (instead of full permutation matrix)
    else:
        inds = inds_r_guess

    # ## L typically has 1's along the diagonal.  (though it is an m x n matrix)
    # ## sometimes, other rows in L could easily be chosen instead (e.g. last element in row = 1.0)
    # valid_rows = np.nonzero(np.abs(l_dmp[:,-1] - 1.0) < 1.0e-12)[0]
    # # valid_rows = np.append(np.arange(r-1), valid_rows)
    # print('valid rows', valid_rows)
    # # r_inds = np.random.permutation(len(valid_rows))
    # # inds = p[r_inds[:r]]
    # if len(valid_rows) > 1:
    #     new_row = np.random.randint(0,len(valid_rows))
    #     print('new row', new_row)
    #     inds[-1] = p[valid_rows[new_row]]
    # print('inds', inds, p)
    # print('L', l_dmp)
    # print('U', u_dmp)

    if n <= r:
        # return slice(0,n)
        return inds

    submat = Q[inds, :]  # permute rows of A
    B = Q @ np.linalg.pinv(submat)       # Q Q^-1 = B = [I, Z]
    # print('B', B)
    ## note: np.linalg.inv(submat) gives bad results if n >= r

    ## start iterations
    it, err = 0, 1.0
    while it < max_iters:  # and err < conv_tol:
        # print('B', B)
        max_ind = np.argmax(np.abs(B))
        i0, j0 = max_ind // (B.shape[1]), max_ind % (B.shape[1])

        max_val = B[i0, j0]
        if np.abs(max_val) <= 1 + 1.0e-8:
            break

        ## rank 1 update of B from switching rows i0 and inds[j0]
        old_row = inds[j0]
        B = B + np.outer(B[:, j0], (B[old_row, :] - B[i0, :])) / B[i0, j0]
        inds[j0] = i0  ## update old row with new row; other row not included in inds (r largest components)

        it += 1

    # print('MAXVOL')
    # print('final B', B[p,:], np.max(B))
    # print('final B', B[inds, :], np.max(B))
    # submat = Q[inds, :]
    # print('final Q Q^-1', (Q @ np.linalg.inv(submat))[inds,:])

    return np.sort(inds)



########################
###   deim methods   ###
########################


def deim_inds(W: 'np.ndarray', max_r: int = None):
    """
    deim instead of maxvol
    :param W:  n x r matrix whose columns are singular vectors. want to find r x r submatrix of A
    :param max_iters:
    :param conv_tol:
    :param do_qr:
    :return: inds to obtain submatrix of A

    """
    wi = W[:, 0]
    inds = [np.argmax(np.abs(wi))]

    max_r = min(max_r, W.shape[1]) if max_r is not None else W.shape[1]
    ## oversampling

    # diff_r = min(W.shape[0] - max_r, 5)
    # print('over sampling?', W.shape, max_r, diff_r)
    # max_r = max_r + diff_r

    for i in range(1, max_r):
        wi = W[:, i]

        ## schur complement to update the identity?
        W_ = W[:, :i]
        V = W_[inds, :]
        c = np.linalg.solve(V, wi[inds])
        r = wi - W[:, :i] @ c

        p_ind = np.argmax(np.abs(r))
        inds += [p_ind]

    return inds


# def select_rows(tens: qtn.Tensor, lbond: str, rbond: str, phys_bond: str, solver_type=DEFAULT_SOLVER):
#     """ matricize tensor (phys_bond, lbond) x r bond
#         obtain selection indices via deim
#     """
#     fuse_inds = [phys_bond, lbond] if lbond is not None else [phys_bond]
#     tens = tens.transpose(*fuse_inds, rbond)
#     tens_ = tens.fuse({f'xx': fuse_inds, f'oo': [rbond]})
#     tens_.transpose('xx', 'oo', inplace=True)
#     if solver_type == CrossSolver.DEIM:
#         inds_r = deim_inds(tens_.data)
#     elif solver_type == CrossSolver.MAXVOL:
#         inds_r = maxvol_inds(tens_.data)
#     else:
#         raise ValueError
#     return inds_r

def tensor_get_submat(tens: qtn.Tensor, lbond: str, rbond: str, phys_bond: str, solver_type=DEFAULT_SOLVER,
                      oversample=False):
    """ matricize tensor (phys_bond, lbond) x r bond
        obtain selection indices via deim
    """
    fuse_inds = [phys_bond, lbond] if lbond is not None else [phys_bond]
    tens = tens.transpose(*fuse_inds, rbond)
    tens_ = tens.fuse({f'xx': fuse_inds, f'oo': [rbond]})
    tens_.transpose('xx', 'oo', inplace=True)

    ## oversampling
    tens_copy = tens_.copy()
    if oversample:
        diff_r = min(tens_copy.shape[0] - tens_copy.shape[1], 1)
        print('over sampling?', tens_copy.shape, diff_r)
        if diff_r > 0:
            add_rand = np.random.random((tens_copy.shape[0], diff_r)) * 1.0e-08
            tens_copy.modify(data=np.hstack([tens_copy.data, add_rand]))

    if solver_type == CrossSolver.DEIM:
        inds_r = deim_inds(tens_copy.data)
    elif solver_type == CrossSolver.MAXVOL:
        inds_r = maxvol_inds(tens_.data)
    else:
        raise ValueError

    tens_.reindex({'xx': rbond + '_', 'oo': rbond}, inplace=True)
    TC, TU, TR = cur_split(tens_, inds_r, cu_bond=rbond, ur_bond=rbond + '_x')
    ## TC: rbond + '_'  //  rbond  (throw away)
    ## TU: rbond  //  rbond + '_x'
    ## TR: rbond + 'x' // rbond

    return inds_r, TC, TU, TR


def check_orthog(mps: 'MPS'):
    import local_solvers.helper_cross_2 as helper_cross
    print('check X orthog', mps.cur_orthog)
    # if mps.cur_orthog is None:
    #     indL1, indR1 = mps.check_select_inds()
    # else:
    try:
        ind1 = mps.cur_orthog
        ref_tens = convert_basis_to_elementwise(mps, mps[ind1], ind1, 1)
        ref_tens.reindex( {ind: ind[:-2] for ind in ref_tens.inds if ind[-1] == 'x'} , inplace=True)
        is_canon = helper_cross.check_center_orthog(mps, mps.cur_orthog, ref_tens=ref_tens)
        indL1, indR1 = (ind1, ind1) if is_canon else (-1, mps.L)
    except KeyError:
        indL1, indR1 = mps.check_select_inds()
    print('check G orthog')
    indL2, indR2 = helper_quimb.check_orthog(mps)
    indL = min(indL1, indL2)
    indR = max(indR1, indR2)
    return indL, indR


# def deim_split(A: np.ndarray, return_inds=False):
#     """
#     decompose matrix using DEIM; obtain low-rank approximation
#     A is 2D matrix: (alpha_{i-1} * d_{i}) x alpha_{i}
#     """
#
#     u, s, vt = np.linalg.svd(A, full_matrices=False)
#     Q = u
#     R_dmp = (vt.T * s).T
#
#     inds = deim_inds(u)
#     submat = Q[inds, :]  # permute rows of A
#
#     B = Q @ np.linalg.pinv(submat)       # Q Q^-1 = B = [I, Z]
#     T1 = B
#     T2 = (submat @ R_dmp)
#
#     if return_inds:
#         return T1, T2, inds  # np.sort(inds)
#     else:
#         return T1, T2
#
#
# def deim_compress(A: np.ndarray, max_bond, cutoff=None, return_inds=False):
#     """
#     decompose matrix using DEIM; obtain low-rank approximation
#     truncate via singular values, and then perform DEIM?
#     or, just perform DEIM up to desired rank?
#     i don't think it really matters
#     """
#
#     # print('deim compression', A.shape, max_bond, cutoff)
#
#     n, r = A.shape  ## A is 2D matrix:  e.g 2 site tensor, (alpha_{i-1} * d_{i}) x alpha_{i+1} * d{i+1}
#
#     if (max_bond is None or n < max_bond or r < max_bond) and cutoff is None:
#         T1, T2, inds_r = deim_split(A, return_inds=True)
#         if return_inds:
#             inds_c = list(range(r))
#             return T1, T2, inds_r, inds_c
#         else:
#             return T1, T2
#
#     u, s, vt = np.linalg.svd(A, full_matrices=False)
#     u0, s0, vt0 = u.copy(), s.copy(), vt.copy()
#     if max_bond is not None:
#         u = u[:, :max_bond]
#         s = s[:max_bond]
#         vt = vt[:max_bond, :]
#         print('max bond', max_bond, 'err', np.linalg.norm(s[max_bond:])/np.linalg.norm(s))
#
#     if cutoff is not None:
#         # cum_sum = np.cumsum((s[::-1])**2/s[0]**2)   # ordered smallest to largest
#         cum_sum = np.cumsum((s[::-1]) ** 2 / np.linalg.norm(s) ** 2)  # ordered smallest to largest
#         # print('cum sum', cum_sum, (cum_sum < cutoff)[:10])
#         cut_ind = np.argmin(cum_sum < cutoff)
#         if cut_ind != 0:
#             print('cutoff', cutoff, 'cut ind', -cut_ind, 'err', np.linalg.norm(s[-cut_ind:]) / np.linalg.norm(s))
#             u = u[:, :-cut_ind]
#             s = s[:-cut_ind]
#             vt = vt[:-cut_ind, :]
#
#
#     # pdb.set_trace()
#     ## old version
#     inds_r = deim_inds(u,) # max_bond)
#     inds_c = deim_inds(vt.T.conj(),) # max_bond)
#     # ## new version
#     # inds_r = deim_inds(u0, len(s))
#     # inds_c = deim_inds(vt0.T.conj(), len(s))
#
#     # _, _, ind = deim_split(A, return_inds=True)
#
#     # print('inds r', inds_r)
#     # print('inds c', inds_c)
#     # print('split ind', ind)
#
#     ## old version version: construct T1, T2 from low-rank tensor
#     submat_u = u[inds_r, :]
#     T1 = u @ np.linalg.pinv(submat_u)
#
#     R_dmp = (vt.T * s).T
#     T2 = submat_u @ R_dmp
#
#     ## new version: construct T1, T2 from original tensor
#     # print('new deim implementation')
#     # submat_u0 = u0[inds_r, :]
#     # T1 = u0 @ np.linalg.pinv(submat_u0)
#     #
#     # R0_dmp = (vt0.T * s0).T
#     # T2 = submat_u0 @ R0_dmp
#
#     # print('T1 diff', np.linalg.norm(T1-T10), 'T2 diff', np.linalg.norm(T2-T20))
#     #
#     # plt.figure()
#     # plt.plot( (T1 @ T2).reshape(-1) )
#     # plt.plot((T10 @ T20).reshape(-1))
#     # plt.show()
#
#     if return_inds:
#         return T1, T2, inds_r, inds_c  # np.sort(inds)
#     else:
#         return T1, T2
