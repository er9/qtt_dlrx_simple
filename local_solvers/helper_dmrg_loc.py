import pdb

import numpy as np

import helper_quimb
from local_solvers.helper_cross_2 import approx_svd
from setup_.configs import *
import time
import scipy.linalg as linalg
import quimb.tensor as qtn
import helper_quimb as helper
from local_solvers.defaults import *
import local_solvers.helper_tn as helper_tn
from local_solvers.mps_classes import MPS

if TYPE_CHECKING:
    from grid import Grid

MPO_type = Union['qtn.MatrixProductOperator']
MPS_type = Union['qtn.MatrixProductState']

def tensor_svd(tens: 'qtn.Tensor', left_inds: Sequence[str], absorb: Literal['left','right'],
                 max_bond=MAXBOND, min_bond=MINBOND, cutoff=CUTOFF, bond_ind: str=None):

    return helper.tensor_svd(tens, left_inds, absorb=absorb, max_bond=max_bond, min_bond=min_bond, cutoff=cutoff,
                             bond_ind=bond_ind)


def get_cut_ind(svals, cutoff=CUTOFF, max_bond=MAXBOND, min_bond=MINBOND, is_squared=False):
    """ for eigenvalues, is_squared=True
    """
    return helper_quimb.get_cut_ind(svals, cutoff, max_bond=max_bond, min_bond=min_bond, is_squared=is_squared)

def get_tens_rdm(tens: 'qtn.Tensor', ket_iso: Sequence[str], bra_iso: Optional[Sequence[str]]=None,
                 transpose_inds: Optional[Sequence[str]]=None):
    """ tens.conj() * tens, contracted over inds not specified in bra to ket inds
    """
    if bra_iso is None:
        bra_iso = [ind + '_' for ind in ket_iso]

    if transpose_inds is None:
        transpose_inds = [*ket_iso, *bra_iso]

    tens_conj = tens.conj()
    tens_conj.reindex({k: b for k, b in zip(ket_iso, bra_iso)}, inplace=True)

    rdm = qtn.tensor_contract(tens_conj, tens, output_inds=transpose_inds)

    return rdm


def get_expanded_qr(tens_list, left_inds, max_bond=MAXBOND, min_bond=MINBOND, cutoff=CUTOFF, bond_ind:str=None):
    """ tens_list[0] is the main one to keep; rest are set to zero
    """
    print('get expanded qr')
    T1 = tens_list[0].copy()
    r_inds = [ind for ind in T1.inds if ind not in left_inds]
    r_size = [T1.ind_size(ind) for ind in r_inds]
    for T2 in tens_list[1:]:
        T1 = qtn.tensor_direct_product(T1, T2.copy(), sum_inds=left_inds, inplace=True)

    new_q, new_r = tensor_svd(T1, left_inds, absorb='right', max_bond=max_bond, min_bond=min_bond, cutoff=cutoff,
                              bond_ind=bond_ind)
    r_data = new_r.data
    for ind, size in zip(r_inds, r_size):
        idx = new_r.inds.index(ind)
        r_data = np.moveaxis(r_data, idx, 0)
        r_data = r_data[:size]
        r_data = np.moveaxis(r_data, 0, idx)
        new_r.modify(data = r_data)
    return new_q, new_r



def get_tot_rdm(tensors: Sequence['qtn.Tensor'], ket_iso: Sequence[str], bra_iso: Optional[Sequence[str]]=None,
                transpose_inds: Optional[Sequence[str]]=None):

    rdms = [get_tens_rdm(tens, ket_iso, bra_iso, transpose_inds=transpose_inds) for tens in tensors]
    # print('rdms', [t.norm() for t in rdms])
    # for t in rdms:
    #     t.modify(apply=lambda x: x / t.norm())
    tot_rdm = helper_tn.sum_tens(rdms)
    return tot_rdm


def expand_mats(tensors: Sequence['qtn.Tensor'], fuse_inds: Sequence[str],
                transpose_inds: Optional[Sequence[str]]=None):

    out = tensors[0]
    for tens in tensors[1:]:
        out = qtn.tensor_direct_product(out, tens, sum_inds=fuse_inds)

    return out


def get_rdm_eig(tot_rdm: 'qtn.Tensor', ket_iso: Sequence[str], bra_iso: Optional[Sequence[str]]=None,
                new_ind: str=None, max_bond=None, cutoff=None, return_weights=False
                ) -> Union[tuple[qtn.Tensor,np.ndarray], qtn.Tensor]:
    """ A = Q Lambda Q.T  --> return Q
    """

    if bra_iso is None:
        bra_iso = [ind + '_' for ind in ket_iso]

    # if transpose_inds is None:
    #     transpose_inds = [*ket_iso, *bra_iso]

    tot_rdm.transpose(*ket_iso, *bra_iso, inplace=True)
    nb = len(ket_iso)

    denmat_shape = tot_rdm.shape
    tens_shape = denmat_shape[:nb]
    sq_shape = np.prod(denmat_shape[:nb])

    eigval, eigvec = np.linalg.eigh(tot_rdm.data.reshape(sq_shape, sq_shape))

    sort_inds = np.argsort(np.abs(eigval))[::-1]

    # ## pre 12/30/24 version
    # cutoff = CUTOFF if cutoff is None else cutoff
    # ## apply cut-off
    # ev_max = eigval[sort_inds[0]]
    # new_sort_inds = []
    # for si in sort_inds:
    #     if np.abs(eigval[si]) > cutoff * np.abs(ev_max):
    #         new_sort_inds += [si]
    #     else:
    #         break
    # sort_inds = new_sort_inds
    # # print('len cutoff', CUTOFF, len(sort_inds))
    #
    # if max_bond is not None:
    #     sort_inds = np.argsort(np.abs(eigval))[::-1]
    #     eigvec = eigvec[:, sort_inds[:max_bond]]
    #     # print('eig trunc', max_bond, np.linalg.norm(eigval[sort_inds[max_bond:]]))
    #
    #     if return_weights:
    #         eigval = eigval[sort_inds[:max_bond]]
    # else:
    #     eigval = eigval[sort_inds]
    #     eigvec = eigvec[:, sort_inds]

    ## newer version
    eigval = eigval[sort_inds]
    eigvec = eigvec[:, sort_inds]

    # if max_bond is not None:
    #     eigvec = eigvec[:, :max_bond]
    #     eigval = eigval[:max_bond]
    #
    # ## apply cut-off
    # cutoff = CUTOFF if cutoff is None else cutoff
    # if cutoff is not None:
    #     ev_max = np.max(np.abs(eigval))  # eigval[0]
    #
    #     # new_sort_inds = []
    #     # for si in range(len(eigval)):
    #     #     print('eval', eigval[si], cutoff * np.abs(ev_max))
    #     #     if np.abs(eigval[si]) > cutoff * np.abs(ev_max):
    #     #         new_sort_inds += [si]
    #     #     else:
    #     #         break
    #     # print('len new sort inds', new_sort_inds, len(eigval))
    #
    #     # new_sort_inds = []
    #     cum_sum = np.cumsum(np.abs(eigval[::-1]))   # --> smallest (sum from end) to largest (including largest eigval)
    #     cut_ind = np.argmin(cum_sum / ev_max < cutoff)
    #     if MINBOND is not None:     ## ensure a minimum bond dimension
    #         min_ind = max(len(eigval) - MINBOND, 0)
    #         cut_ind = min(cut_ind, min_ind)     ## cut fewer elements from the end
    #     if cut_ind != 0:
    #         print('rdm eig cutoff', cutoff, 'cut ind', -cut_ind, 'err', np.linalg.norm(eigval[-cut_ind:]) / ev_max)
    #         eigval = eigval[:-cut_ind]
    #         eigvec = eigvec[:, :-cut_ind]

    cut_ind = get_cut_ind(eigval, cutoff=cutoff, max_bond=max_bond, is_squared=True)
    eigval = eigval[:cut_ind]
    eigvec = eigvec[:, :cut_ind]

    eigvec = eigvec.reshape(*tens_shape, -1)
    eigvec = qtn.Tensor(data=eigvec, inds=tuple(ket_iso) + (new_ind,))

    if return_weights:
        return eigvec, eigval

    return eigvec


def get_rdm_eig_filtered(tot_rdm: 'qtn.Tensor',
                         partition: 'qtn.MatrixProductState', site_inds: Sequence[int],
                         ket_iso: Sequence[str], bra_iso: Optional[Sequence[str]]=None,
                         filter_range: int = 1,
                         new_ind: str=None, max_bond=None, return_weights=False
                         ) -> Union[tuple[qtn.Tensor,np.ndarray], qtn.Tensor]:
    """ A = Q Lambda Q.T  --> return Q
        filter eigvecs also based on frequency of basis fct on partition
        only works for 1D right now
    """

    if bra_iso is None:
        bra_iso = [ind + '_' for ind in ket_iso]

    # if transpose_inds is None:
    #     transpose_inds = [*ket_iso, *bra_iso]

    tot_rdm.transpose(*ket_iso, *bra_iso, inplace=True)
    nb = len(ket_iso)

    denmat_shape = tot_rdm.shape
    tens_shape = denmat_shape[:nb]
    sq_shape = np.prod(denmat_shape[:nb])

    eigval, eigvec = np.linalg.eigh(tot_rdm.data.reshape(sq_shape, sq_shape))

    eigvec_tens = eigvec.reshape(*tens_shape, -1)
    eigvec_tens = qtn.Tensor(data=eigvec_tens, inds=tuple(ket_iso) + (new_ind,))

    ## high-pass frequencies
    Lp = partition.num_tensors + 1
    xp = np.linspace(0, 1, 2 ** Lp, endpoint=False)
    nx = np.arange(-filter_range, filter_range + 1) + ((2 ** Lp) // 2)
    # nx = np.arange(-2**Lp // 2, 2**Lp //2)
    # print('nx', nx)
    # print('Lp', Lp, 2**Lp)
    # print('new ind', new_ind, ket_iso, bra_iso)

    xp_, nx_ = np.meshgrid(xp, nx, indexing='ij')
    bases = np.exp(1.j * 2 * np.pi * xp_ * nx_) / np.sqrt(2 ** Lp)

    # plt.figure()
    # plt.plot(np.real(bases))
    # plt.title('real bases')
    # plt.figure()
    # plt.plot(np.imag(bases))
    # plt.title('imag bases')
    # plt.show()

    # ket_ind = partition.site_ind_id
    # partition_tens = qtn.tensor_contract(*partition.tensors, eigvec_tens)
    # partition_tens.transpose(*[ket_ind.format(i) for i in site_inds], new_ind)
    # plt.figure()
    # plt.plot(np.real(partition_tens.data.reshape(2**Lp,-1) * eigval))
    # plt.show()

    ket_ind = partition.site_ind_id
    tmp_bond = 'bases'
    bases_ket = qtn.Tensor(bases.reshape(*(2,) * Lp, -1),
                           inds=[ket_ind.format(i) for i in site_inds] + [tmp_bond])
    ovlp = qtn.TensorNetwork([bases_ket, partition, eigvec_tens])
    ovlp = ovlp.contract()

    ## weights[a] = sum_b |ovlp[a,b]|^2
    ind = ovlp.inds.index(tmp_bond)
    ovlp.modify(apply=lambda data: np.abs(data)**2)

    # plt.figure()
    # plt.imshow(np.real(ovlp.data))
    # plt.colorbar()
    # plt.show()

    ovlp.modify(data=np.sum(ovlp.data, axis=ind), inds=(new_ind,))
    print('ovlp', ovlp.data)
    ## penalize large overlaps; if no overlap, scale by 1
    ovlp.modify(apply=lambda data: np.exp(-data))
    print('exo ovlp', ovlp.data)
    ovlp_weights = ovlp.data  # np.where(eigval / np.max(np.abs(eigval)) > 1.0e-3, 1, ovlp.data)  # ovlp.data
    print('exo ovlp', ovlp_weights)

    # plt.figure()
    # plt.plot(np.abs(eigval)/np.max(np.abs(eigval)), label='eigval')
    # plt.plot(ovlp_weights, label='weights')
    # # plt.plot(ovlp_.data, label='ovlp')
    # # plt.plot(ovlp.data, label='weights')
    # plt.legend()
    # plt.show()

    ## reweight eigenvalues
    eigval = eigval * ovlp_weights

    if max_bond is not None:
        sort_inds = np.argsort(np.abs(eigval))[::-1]
        eigvec = eigvec[:, sort_inds[:max_bond]]
        print('eig trunc', np.sum(np.abs(eigval[sort_inds[max_bond:]])))

        if True: # return_weights:
            eigval = eigval[sort_inds[:max_bond]]
    # else:
    #     eigvec = eigvec[:, sort_inds]

    eigvec = eigvec.reshape(*tens_shape, -1)
    eigvec = qtn.Tensor(data=eigvec, inds=tuple(ket_iso) + (new_ind,))

    # partition_tens = qtn.tensor_contract(*partition.tensors, eigvec)
    # partition_tens.transpose(*[ket_ind.format(i) for i in site_inds], new_ind)
    # plt.figure()
    # plt.plot(np.real(partition_tens.data.reshape(2 ** (Lp), -1) * eigval))
    # plt.show()

    if return_weights:
        return eigvec, eigval

    return eigvec


def update_1site(mps: 'MPS', left_site_pos: int, site_i: Union['qtn.Tensor', Sequence['qtn.Tensor']],
                 direction: 'SweepDirection', max_bond: int=None, cutoff: float=None, filter_bases=False,
                 grid:'Grid'=None, ax_deriv_configs=None):
    """ update ket, bra with new_site
        i: int of mps site
    """
    if isinstance(site_i, qtn.Tensor):
        site_i = [site_i]

    # print('check orthog', left_site_pos)
    # helper_quimb.check_orthog(mps)
    # print('update 1 site', [t.norm() for t in site_i])


    # print('update 1 site', left_site_pos, [t.norm() for t in site_i], direction)
    if direction == SweepDirection.RIGHT:
        ind1 = left_site_pos
        ind2 = left_site_pos + 1 if left_site_pos < mps.L - 1 else None
        ## canonicalize ket to next site
        if ind1 < mps.L - 1:
            mps._cur_orthog = ind1 + direction
            # self.cur_orthog = ind1 + direction

    else:
        ind1 = left_site_pos
        ind2 = (left_site_pos - 1) if left_site_pos > 0 else None
        ## canonicalize ket to next site
        if ind1 > 0:
            # self.cur_orthog = ind1 + direction
            mps._cur_orthog = ind1 + direction

    if ind2 is None:  ## is at_end

        ## update ket
        site_i = helper_tn.sum_tens(site_i)
        site_i.transpose_like(mps[ind1], inplace=True)
        mps[ind1].modify(data=site_i.data)
    else:

        x_bond = mps.bond(left_site_pos, left_site_pos + direction)    ## bond to not contract over
        ket_iso = [ind for ind in site_i[0].inds if ind != x_bond]
        bra_iso = [ind + '_' for ind in ket_iso]

        # # [new_ket_site, self.ket[i]]
        expanded_tens = expand_mats(site_i, ket_iso)
        eigvec, _ = tensor_svd(expanded_tens, ket_iso, absorb='right', max_bond=max_bond, cutoff=cutoff,
                               bond_ind=x_bond)
        # print('eigvec shape', eigvec.shape)

        # # [new_ket_site, self.ket[i]]
        # tot_rdm = get_tot_rdm(site_i, ket_iso, bra_iso, )
        # print('tot rdm size', tot_rdm.shape)
        # eigvec = get_rdm_eig(tot_rdm, ket_iso, bra_iso, new_ind=x_bond + '_',
        #                      max_bond=max_bond, cutoff=cutoff)

        decimate(mps, left_site_pos, eigvec, direction)
        mps._cur_orthog = ind2

    return

def decimate(mps: 'MPS', i: int, canon_site_i: 'qtn.Tensor', direction: 'SweepDirection',):
    """ move canonical center using new canon_site_i
        assumes mps orthogonality center currently is at i --> moves to i+1, i-1
    """
    # print('decimate orthog')
    # print(helper_quimb.check_orthog(mps))
    canon_site_i = canon_site_i.copy()

    if direction == SweepDirection.RIGHT:
        ind1 = i
        ind2 = i + 1 if i < mps.L - 1 else None
        ## canonicalize ket to next site
        if i < mps.L - 1:
            mps._cur_orthog = ind1 + direction

    else:
        ind1 = i
        ind2 = (i - 1) if i > 0 else None
        ## canonicalize ket to next site
        if i > 0:
            # self.cur_orthog = ind1 + direction
            mps._cur_orthog = ind1 + direction

    # print('canon site i', canon_site_i)

    x_bond = mps.bond(i, i + direction)  ## bond to not contract over
    # x_bond = self.ket_horizontal_bond(i, direction)  ## bond to not contract over
    # ket_iso = [ind for ind in canon_site_i.inds if (ind != x_bond or ind != x_bond + '_')]
    # bra_iso = [ind + '_' for ind in ket_iso]
    if x_bond in canon_site_i.inds:
        canon_site_i = canon_site_i.reindex({x_bond: x_bond + '_'}, inplace=False)

    # mps[ind1].transpose('i(0)', x_bond, inplace=True)
    # mps[ind2].transpose('i(1)', x_bond, mps.bond(1, 2), inplace=True)
    # print('mps[ind1]', ind1, mps[ind1].inds, mps[ind1].norm())
    # print(mps[ind1].data)
    # print('mps[ind2]', ind2, mps[ind2].inds, mps[ind2].norm())
    # print(mps[ind2].data)

    # print('mps[ind1]', mps[ind1])
    # print('canon site i', canon_site_i)
    next_site = qtn.tensor_contract(canon_site_i.conj(), mps[ind1], mps[ind2])
    canon_site_i.transpose_like(mps[ind1], inplace=True)
    next_site.transpose_like(mps[ind2], inplace=True)
    mps[ind1].modify(data=canon_site_i.data)
    mps[ind2].modify(data=next_site.data)
    # mps[ind1].transpose('i(0)', x_bond, inplace=True)
    # mps[ind2].transpose('i(1)', x_bond, mps.bond(1, 2), inplace=True)
    # print('mps[ind1]', ind1, mps[ind1].inds, mps[ind1].norm())
    # print(mps[ind1].data)
    # print('mps[ind2]', ind2, mps[ind2].inds, mps[ind2].norm())
    # print(mps[ind2].data)
    return


def update_2site(mps: 'MPS', left_site_pos: int, site_i: Union['qtn.Tensor', Sequence['qtn.Tensor']],
                 direction: 'SweepDirection', max_bond: int=None, cutoff: float=None, decimate_only=False):
    """ update ket, bra with new_site
        i: int of mps site
    """
    if isinstance(site_i, qtn.Tensor):
        site_i = [site_i]

    # print('check orthog', left_site_pos)
    # helper_quimb.check_orthog(mps)
    # print('update 1 site', [t.norm() for t in site_i])


    # print('update 2 site', left_site_pos, mps.L, direction)
    if direction == SweepDirection.RIGHT:
        ind1 = left_site_pos
        ind2 = left_site_pos + 1 if left_site_pos < mps.L - 1 else None
        at_end = ind2 == mps.L - 1
        ## canonicalize ket to next site
        if ind1 < mps.L - 1:
            mps._cur_orthog = ind1 + direction
            # self.cur_orthog = ind1 + direction

    else:
        ind2 = left_site_pos
        ind1 = left_site_pos + 1
        at_end = ind2 == 0
        ## canonicalize ket to next site
        if ind1 > 0:
            # self.cur_orthog = ind1 + direction
            mps._cur_orthog = ind1 + direction


    if at_end and not decimate_only:
        # raise NotImplementedError
        site_i = helper_tn.sum_tens(site_i)

        x_ind = mps.bond(ind1, ind1 + direction)
        left_inds = [ind for ind in mps[ind1].inds if ind != x_ind]
        # q, r = qtn.tensor_split(site_i, left_inds, absorb='right', max_bond=max_bond, cutoff=cutoff, bond_ind=x_ind)
        q, r = tensor_svd(site_i, left_inds, absorb='right', max_bond=max_bond, cutoff=cutoff, bond_ind=x_ind)

        q.transpose_like(mps[ind1], inplace=True)
        mps[ind1].modify(data=q.data)

        r.transpose_like(mps[ind1 + direction], inplace=True)
        mps[ind1 + direction].modify(data=r.data)
        mps._cur_orthog = ind1 + direction

    else:

        x_bonds = [mps.bond(ind2, ind2 + direction), mps.site_ind(ind2)]    ## bonds to not contract over
        ket_iso = [ind for ind in site_i[0].inds if ind not in x_bonds]
        bra_iso = [ind + '_' for ind in ket_iso]

        # [new_ket_site, self.ket[i]]
        x_bond = mps.bond(ind1, ind2)
        tot_rdm = get_tot_rdm(site_i, ket_iso, bra_iso, )
        eigvec  = get_rdm_eig(tot_rdm, ket_iso, bra_iso, new_ind=x_bond + '_', max_bond=max_bond, cutoff=cutoff)
        # print('eigvec', eigvec.norm())
        # print(eigvec.data)

        decimate(mps, ind1, eigvec, direction)
        mps._cur_orthog = ind2

    # print('cur orthog', mps.cur_orthog)

    return


def filter_bases_1site(mps: 'MPS', left_site_pos: int, site_i: Union['qtn.Tensor', Sequence['qtn.Tensor']],
                       direction: 'SweepDirection', max_bond: int=None, filter_bases=False,
                       grid:'Grid'=None, ax_deriv_configs=None):

    avg_gtn = grid.get_averaging_mpo(ax_deriv_configs=ax_deriv_configs, spread=3)
    avg_mpo = avg_gtn.data
    if avg_mpo.upper_ind_id == mps.site_ind_id:
        avg_mpo.upper_ind_id = avg_mpo.upper_ind_id + '_'
    avg_mpo.lower_ind_id = mps.site_ind_id

    l_ket = mps[:left_site_pos]
    r_ket = mps[left_site_pos + 1:]

    l_bond = mps.bond(left_site_pos, left_site_pos - 1) if left_site_pos > 0 else None
    r_bond = mps.bond(left_site_pos, left_site_pos + 1) if left_site_pos < mps.L - 1 else None

    l_bra = l_ket.conj(inplace=False)
    if l_bond is not None:
        l_bra[left_site_pos - 1].reindex({l_bond: l_bond + '_'}, inplace=True)
        l_bra.site_ind_id = avg_mpo.upper_ind_id

    r_bra = r_ket.conj(inplace=False)
    if r_bond is not None:
        r_bra[left_site_pos + 1].reindex({r_bond: r_bond + '_'}, inplace=True)
        r_bra.site_ind_id = avg_mpo.upper_ind_id

    new_site_i = []
    for si in site_i:
        proj_avg_mps = qtn.TensorNetwork([avg_gtn.data, l_ket, r_ket, l_bra, r_bra, si])

        # if l_ket.num_tensors > 0:
        #     print('l ket', l_ket)
        #     print('l bra', l_bra)
        # if r_ket.num_tensors > 0:
        #     print('r ket', r_ket)
        #     print('r bra', r_bra)
        # print('site i', site_i)
        #
        # print('proj avg mps', proj_avg_mps)

        new_si = proj_avg_mps.contract() * (10 ** proj_avg_mps.exponent)
        reindex_inds = {avg_mpo.upper_ind_id.format(left_site_pos): avg_mpo.lower_ind_id.format(left_site_pos)}
        if l_bond is not None:
            reindex_inds[l_bond + '_'] = l_bond
        if r_bond is not None:
            reindex_inds[r_bond + '_'] = r_bond
        new_si.reindex(reindex_inds, inplace=True)
        new_site_i += [new_si]

    return new_site_i
