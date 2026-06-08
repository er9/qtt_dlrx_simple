""" v2 works.
    this is a cleaner version + elemental multiplication of f(x) * g(y)
    where f, g are element-wise operations
"""
import helper_quimb
from setup_.configs import *
import time
# from scipy import linalg
import scipy.linalg as linalg
import quimb.tensor as qtn
# import helper_quimb as helper
from local_solvers.defaults import *
import local_solvers.helper_tn as helper_tn
from local_solvers.mps_classes import MPS

MPO_type = Union['qtn.MatrixProductOperator']
MPS_type = Union['qtn.MatrixProductState']


def get_selectors(mps, left_pos_site, nsites, select_inds=None):
    """ figure out which grid points are selected, knowing that indices are selected from hyperindex (physical, virtual)
        this might not work if different select_inds are provided, since it's dependent on the
        ranks of the specific MPS
    """
    sel_inds_l, sel_inds_r = [[]], [[]]
    select_inds = mps.select_inds if select_inds is None else select_inds

    if nsites == 0:
        left_pos_site = left_pos_site + 1

    if left_pos_site > 0:
        prev_virt = [[c] for c in  select_inds[0]]
        for j in range(1, left_pos_site):
            bond_size = len(prev_virt)  # mps.bond_size(j, j - 1)
            sel_inds = []
            for ind in select_inds[j]:
                phys, virt = ind // bond_size, ind % bond_size
                # print('ind', ind, phys, virt)
                sel_inds += [prev_virt[virt] + [phys]]
            prev_virt = sel_inds
        sel_inds_l = prev_virt

    if left_pos_site + nsites < mps.L:
        prev_virt = [[c] for c in  select_inds[mps.L - 1]]
        for j in range(mps.L - 2, left_pos_site + nsites - 1, -1):
            bond_size = len(prev_virt)  # mps.bond_size(j, j + 1)
            sel_inds = []
            for ind in select_inds[j]:
                phys, virt = ind // bond_size, ind % bond_size
                # print('phys virt', ind, phys, virt)
                sel_inds += [[phys] + prev_virt[virt]]
            prev_virt = sel_inds
        sel_inds_r = prev_virt

    # print('l,r', sel_inds_l, sel_inds_r)

    tot_sel_inds = []
    for l in sel_inds_l:
        for i in np.ndindex((2,)*nsites):
            for r in sel_inds_r:
                tot_sel_inds += [tuple(l) + i + tuple(r)]

    # print('tot sel inds', tot_sel_inds)
    # str_sel_inds = [[str(x) for x in inds] for inds in tot_sel_inds]
    # print('tot sel inds', [int("".join(x),2) for x in str_sel_inds])
    return tot_sel_inds


def select_tensor(sel_inds, lbond:str, rbond:str, pbond:str, lsize:int, rsize:int, phys_dim:int, tag:str='T'):
    sel_array = np.zeros((lsize * phys_dim, rsize))
    # if side == 'left':
    #     ## fuse_inds = phys_inds + left_inds
    #     sel_array = np.zeros((lsize * phys_dim, rsize))
    # elif side == 'right':
    #     ## fuse_inds = phys_inds + right_inds
    #     sel_array = np.zeros((rsize * phys_dim, lsize))
    # else:
    #     raise ValueError


    for ix, ind in enumerate(sel_inds):
        sel_array[ind, ix] = 1.0

    if lbond is not None:
        sel_array = sel_array.reshape(phys_dim, lsize, rsize)
        sel_tens = qtn.Tensor(data=sel_array, inds=(pbond, lbond, rbond), tags=(tag,))
    else:
        sel_tens = qtn.Tensor(data=sel_array, inds=(pbond, rbond), tags=(tag,))

    return sel_tens

def get_projector(mps, left_pos_site, nsites, get_right=True, get_left=True):
    """ build tensor that selects specified inds
    """
    select_inds = mps.select_inds

    left_tensors = []
    if get_left:
        for i in range(left_pos_site):
            lbond = (mps.bond(i, i - 1)) if i > 0 else None
            rbond = (mps.bond(i, i + 1)) if i < mps.L else None
            pbond = mps.site_ind_id.format(i)

            l_size = 1 if i == 0 else mps.bond_size(i, i - 1)
            r_size = len(select_inds[i])
            phys_dim = mps.phys_dim(i)

            ## fuse_inds = phys_inds + left_inds
            sel_tens = select_tensor(select_inds[i], lbond, rbond, pbond, l_size, r_size, phys_dim, tag=f'P{i}')

            # sel_array = np.zeros((l_size * phys_dim, r_size))
            # for ix, ind in enumerate(select_inds[i]):
            #     sel_array[ind, ix] = 1.0
            # if i > 0:
            #     sel_array = sel_array.reshape(phys_dim, l_size, r_size)
            #     sel_tens = qtn.Tensor(data=sel_array, inds=(pbond, lbond, rbond), tags=(f'P{i}',))
            # else:
            #     sel_tens = qtn.Tensor(data=sel_array, inds=(pbond, rbond), tags=(f'P{i}',))

            left_tensors += [sel_tens]

    right_tensors = []
    if get_right:
        for i in range(left_pos_site + nsites, mps.L):
            lbond = (mps.bond(i, i - 1)) if i > 0 else None
            rbond = (mps.bond(i, i + 1)) if i < mps.L - 1 else None
            pbond = mps.site_ind_id.format(i)

            r_size = 1 if i == mps.L-1 else mps.bond_size(i, i + 1)
            l_size = len(select_inds[i])
            phys_dim = mps.phys_dim(i)

            ## fuse_inds = phys_inds + virtual_inds
            sel_tens = select_tensor(select_inds[i], rbond, lbond, pbond, r_size, l_size, phys_dim, tag=f'P{i}')

            # sel_array = np.zeros((r_size * phys_dim, l_size))
            # for ix, ind in enumerate(select_inds[i]):
            #     sel_array[ind, ix] = 1.0
            # if i < mps.L - 1:
            #     sel_array = sel_array.reshape(phys_dim, r_size, l_size)
            #     sel_tens = qtn.Tensor(data=sel_array, inds=(pbond, rbond, lbond), tags=(f'P{i}',))
            # else:
            #     sel_tens = qtn.Tensor(data=sel_array, inds=(pbond, lbond), tags=(f'P{i}',))
            right_tensors += [sel_tens]

    return left_tensors, right_tensors


def get_points(mps, left_pos_site, nsites, select_inds=None):
    """ get selection indices for dense representation """
    coords = get_selectors(mps, left_pos_site, nsites, select_inds=select_inds)

    selectors = []
    for c in coords:
        q = mps.phys_dim(0)
        selectors += [int("".join(str(x) for x in c), q)]

    return selectors


def get_inds(mps: Union['MPS', Sequence['qtn.Tensor']], ind: int, site_ind_ids: Optional[Sequence[str]]=None):
    L = mps.L if isinstance(mps, qtn.MatrixProductState) else len(mps)
    if site_ind_ids:
        phys_inds = [s.format(ind) for s in site_ind_ids]
    else:
        phys_inds = [mps.site_ind_id.format(ind)]

    if ind == 0:
        right_inds = mps[ind].bonds(mps[ind + 1])
        left_inds = [ix for ix in mps[ind].inds if ix not in [*right_inds, *phys_inds]]
    elif ind == L - 1:
        left_inds = mps[ind].bonds(mps[ind - 1])
        right_inds = [ix for ix in mps[ind].inds if ix not in [*left_inds, *phys_inds]]
    else:
        left_inds = mps[ind].bonds(mps[ind - 1])
        right_inds = mps[ind].bonds(mps[ind + 1])

    return [*left_inds], phys_inds, [*right_inds]

def iso_left_inds(mps: Union['MPS', Sequence['qtn.Tensor']], ind: int, site_ind_ids: Optional[Sequence[str]]=None):
    left_inds, phys_inds, right_inds = get_inds(mps, ind, site_ind_ids)
    return phys_inds + left_inds, right_inds
    # return left_inds + phys_inds, right_inds

def iso_right_inds(mps: Union['MPS', Sequence['qtn.Tensor']], ind: int, site_ind_ids: Optional[Sequence[str]]=None):
    left_inds, phys_inds, right_inds = get_inds(mps, ind, site_ind_ids)
    return left_inds, phys_inds + right_inds


def plot_submat(mps: 'MPS', left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor', select_inds=None, ref_kets=None,
                plt_title='', verbose=False):

    ref_kets = [mps] if ref_kets is None else ref_kets

    plt.figure()
    plt.title(plt_title)
    for it, ref_ket in enumerate(ref_kets):
        if isinstance(ref_ket, qtn.MatrixProductState):
            plt.plot(np.real(ref_ket.to_dense()) * 10 ** ref_ket.exponent, label=f'init {it}')
            plt.plot(np.imag(ref_ket.to_dense()) * 10 ** ref_ket.exponent, '--', label=f'_init {it}')
        else:
            plt.plot(np.real(ref_ket), label=f'init {it}')
            plt.plot(np.imag(ref_ket), '--', label=f'_init {it}')

    coords = get_selectors(mps, left_site_pos, nsites, select_inds=select_inds)

    selectors = []
    for c in coords:
        q = mps.phys_dim(0)
        selectors += [int("".join(str(x) for x in c), q)]

    # print('selectors', selectors)
    # print('nsites', nsites)
    if nsites == 0:
        x_ind = mps.bond(left_site_pos, left_site_pos + 1)
        inds = [x_ind + '_L', x_ind + '_R']
    else:
        inds = []
        if left_site_pos > 0:
            inds += [mps.bond(left_site_pos, left_site_pos - 1)]
        inds += [mps.site_ind(left_site_pos + i) for i in range(nsites)]
        if left_site_pos + nsites < mps.L:
            inds += [mps.bond(left_site_pos + nsites - 1, left_site_pos + nsites)]
        if verbose:
            print('inds', inds)

    site_tens = site_tens.transpose(*inds, inplace=False)
    # print('site tens', site_tens.data.reshape(-1))
    # print('plot submat', site_tens)
    # print('selectors', selectors)
    # print('select inds', mps.select_inds)
    plt.plot(selectors, np.real(site_tens.data.reshape(-1)), 'o', label='site tens')
    plt.plot(selectors, np.imag(site_tens.data.reshape(-1)), 'x', label='_site tens')
    plt.legend()
    plt.show()


def check_left_orthog(mpx: Union['qtn.MatrixProductState', Sequence[qtn.Tensor]],
                      select_inds: dict[int, Sequence[int]], site_ind_ids: Sequence[str], verbose=False) -> int:
    """ returns index of first tensor that is not left canonical
    """
    mpx = mpx.copy()
    L = len(mpx) if isinstance(mpx, (list, tuple)) else mpx.L
    for i in range(L):
        tens = mpx[i].copy()

        # bond_l = [next(iter(mpx[i].bonds(mpx[i - 1])))] if i > 0 else []
        # bond_r = [next(iter(mpx[i].bonds(mpx[i + 1])))] if i < L-1 else []
        # bond_p = [s.format(i) for s in site_ind_ids]
        bonds_l, bonds_r = iso_left_inds(mpx, i, site_ind_ids)
        # bonds_l = bond_p + bond_l

        tens = tens.fuse({'eff': bonds_l}) # bond_p + bond_l})
        tens.transpose('eff', *bonds_r, inplace=True)
        try:
            if len(select_inds[i]) == 0:
                raise KeyError
            env = tens.data[select_inds[i], :]
            err = np.linalg.norm(env - np.eye(env.shape[0]))
        except (KeyError, ValueError, IndexError):
            if verbose:
                print(f'left: select ind not defined or incompatible for {i}')
            return i

        if err > 1.0e-8:
            if verbose:
                print('not left orthog', i, err) #, env)
            if err > 1.0e-5:
                break

    return i


def check_right_orthog(mpx: Union['qtn.MatrixProductState', Sequence['qtn.Tensor']],
                       select_inds: dict[int, Sequence[int]], site_ind_ids: Sequence[str], verbose=False) -> int:
    """ returns index of first tensor that is not left canonical
    """
    mpx = mpx.copy()
    L = len(mpx) if isinstance(mpx, (list, tuple)) else mpx.L
    for i in range(L - 1, -1, -1):
        tens = mpx[i].copy()

        # bond_l = [next(iter(mpx[i].bonds(mpx[i - 1])))] if i > 0 else []
        # bond_r = [next(iter(mpx[i].bonds(mpx[i + 1])))] if i < L - 1 else []
        # bond_p = [s.format(i) for s in site_ind_ids]  # [mpx.site_ind_id.format(i)]
        bonds_l, bonds_r = iso_right_inds(mpx, i, site_ind_ids)
        # bonds_r = bond_p + bond_r

        tens = tens.fuse({'eff': bonds_r})
        tens.transpose('eff', *bonds_l, inplace=True)
        try:
            if len(select_inds[i]) == 0:
                raise KeyError
            env = tens.data[select_inds[i], :]
            err = np.linalg.norm(env - np.eye(env.shape[0]))
        except (KeyError, ValueError, IndexError):
            if verbose:
                print(f'right: select ind not defined or incompatible for {i} ')
            return i

        if err > 1.0e-8:
            if verbose:
                print('not right orthog', i, err) #, env)
            if err > 1.0e-5:
                break

    return i

def check_center_orthog(mpx: Union['qtn.MatrixProductState', 'MPS'], cur_orthog: int, ref_tens=None, verbose=False) -> bool:

    mpx = mpx.copy()
    if verbose:
        print('cross check center orthog')

    select_inds = mpx.select_inds
    site_ind_ids = [mpx.site_ind_id]

    tens_left = None
    for i in range(cur_orthog):
        if verbose:
            print('check left i', i)
        tens = mpx[i].copy()
        if tens_left is not None:
            tens_ = qtn.tensor_contract(tens, tens_left)
            tens_.transpose_like(tens, inplace=True)
            tens.modify(data=tens_.data)

        bonds_l, bonds_r = iso_left_inds(mpx, i, site_ind_ids)
        # bonds_l = bond_p + bond_l

        tens = tens.fuse({'eff': bonds_l}) # bond_p + bond_l})
        tens.transpose('eff', *bonds_r, inplace=True)
        try:
            if len(select_inds[i]) == 0:
                raise KeyError

            sel_data = tens.data[select_inds[i]]
            tens_left = qtn.Tensor(data=sel_data, inds=tens.inds)

        except (KeyError, ValueError, IndexError):
            if verbose:
                print(f'left: select ind not defined or incompatible for {i}')
            return False

    tens_right = None
    for i in range(mpx.L-1, cur_orthog, -1):
        if verbose:
            print('check right i', i)

        tens = mpx[i].copy()
        if tens_right is not None:
            tens_ = qtn.tensor_contract(tens, tens_right)
            tens_.transpose_like(tens, inplace=True)
            tens.modify(data=tens_.data)

        bonds_l, bonds_r = iso_right_inds(mpx, i, site_ind_ids)

        tens = tens.fuse({'eff': bonds_r})
        tens.transpose('eff', *bonds_l, inplace=True)
        try:
            if len(select_inds[i]) == 0:
                raise KeyError
            sel_data = tens.data[select_inds[i]]
            tens_right = qtn.Tensor(data=sel_data, inds=tens.inds)

        except (KeyError, ValueError, IndexError):
            if verbose:
                print(f'right: select ind not defined or incompatible for {i} ')
            return False

    tens = mpx[cur_orthog]

    proj_tens = tens.copy()
    if tens_left is not None:
        proj_tens = qtn.tensor_contract(proj_tens, tens_left)
        proj_tens.transpose_like(tens, inplace=True)
        proj_tens.modify(inds=tens.inds)
    if tens_right is not None:
        proj_tens = qtn.tensor_contract(proj_tens, tens_right)
        proj_tens.transpose_like(tens, inplace=True)
        proj_tens.modify(inds=tens.inds)

    if ref_tens is None:
        ref_tens = tens

    ## verify that projection yields the same tensor
    proj_tens.transpose_like(ref_tens, inplace=True)
    diff_tens = helper_quimb.add_tensors(proj_tens * -1, ref_tens)
    if verbose:
        print('ref tens', ref_tens.data, ref_tens.norm())
        print('difference norm', diff_tens.norm() / ref_tens.norm() )

    return diff_tens.norm() / ref_tens.norm() < 1.0e-13


def check_orthog(mpx: Union['qtn.MatrixProductState', 'MPS'], select_inds: dict[int, Sequence[int]] = None,
                 site_ind_ids: Sequence[str] = None, verbose=False) -> tuple[int, int]:

    mpx = mpx.copy()

    if verbose:
        print('check orthog')
    if select_inds is None:
        select_inds = mpx.select_inds
    if site_ind_ids is None:
        site_ind_ids = [mpx.site_ind_id]
    left = check_left_orthog(mpx, select_inds, site_ind_ids)
    right = check_right_orthog(mpx, select_inds, site_ind_ids)
    return left, right


def combine_inds(*inds: Sequence[int]):
    tot_inds = set(inds[0])
    for ind_list in inds:
        tot_inds = tot_inds.union(set(ind_list))
    return list(tot_inds)



###########################
###    MPS methods      ###
###########################

class CrossSolver(Enum):
    MAXVOL = 'maxvol'
    DEIM = 'deim'

DEFAULT_SOLVER = CrossSolver.DEIM
# DEFAULT_SOLVER = CrossSolver.MAXVOL

def tensor_compress(tens: qtn.Tensor, phys_inds: list[str], right_inds: list[str], max_bond: int = None,
                    cutoff: float = None, return_inds=False, solver_type=DEFAULT_SOLVER, bond_ind='xx',
                    include_inds_r=None, include_inds_c=None, verbose=False, expand_u=False):
    """ compress_split but for a single tensor in the MPS
    """
    left_inds = [ind for ind in tens.inds if (ind not in phys_inds + right_inds)]
    assert (len(left_inds) <= 1), 'not specific enough if have multiple virtual "left inds"'

    ## must follow specific order for fusing unshared bonds
    fuse_inds = phys_inds + left_inds
    tens = tens.transpose(*fuse_inds, *right_inds, inplace=True)
    shape_left = tens.shape[:len(fuse_inds)]
    shape_right = tens.shape[len(fuse_inds):]
    tens_ = tens.fuse({f'xx': fuse_inds, f'oo': right_inds})
    # nr, nc = q_.shape[0], q_.shape[-1]
    if solver_type == CrossSolver.DEIM:
        q, r, inds_r, inds_c = deim_compress(tens_.data, max_bond=max_bond, cutoff=cutoff, return_inds=True,
                                             include_inds_r=include_inds_r, expand_u=expand_u)
    else:
        if include_inds_r is not None:
            raise NotImplementedError
        if expand_u is True:
            raise NotImplementedError
        q, r, inds_r, inds_c = cross_compress(tens_.data, do_qr=True, max_bond=max_bond, cutoff=cutoff, return_inds=True,
                                              include_inds_r=include_inds_r)

    if False: # len(inds_r) > tens_.shape[-1]:
        if verbose:
            print("TENSOR SPLIT")
            print('tens', tens.shape, fuse_inds, right_inds)
            print('select inds', inds_r)
            print('check canon', tens_.shape, len(inds_r), [q[inds_r[i], i] for i in range(len(inds_r))])
            # verbose =True
            # pdb.set_trace()


    q = q.reshape(*shape_left, -1)
    r = r.reshape(-1, *shape_right)

    tens1 = qtn.Tensor(q, inds=(*fuse_inds, bond_ind))
    tens2 = qtn.Tensor(r, inds=(bond_ind, *right_inds))

    if verbose:
        chk = tens1 @ tens2
        chk.transpose_like(tens, inplace=True)
        print('diff', np.linalg.norm(chk.data - tens.data) )

    if return_inds:
        return tens1, tens2, inds_r, inds_c
    else:
        return tens1, tens2


def tensor_select_rows(tens: qtn.Tensor, lbond: str, rbond: str, phys_bond: str, solver_type=DEFAULT_SOLVER,
                       oversample=False, verbose=False):
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
        diff_r = min(tens_copy.shape[0] - tens_copy.shape[1], 5)
        if verbose:
            print('over sampling?', tens_copy.shape, diff_r)
        if diff_r > 0:
            add_rand = np.random.random((tens_copy.shape[0], diff_r)) * 1.0e-08
            tens_copy.modify(data=np.hstack([tens_copy.data, add_rand]))

    if solver_type == CrossSolver.DEIM:
        inds_r = deim_inds(tens_copy.data)
    elif solver_type == CrossSolver.MAXVOL:
        inds_r = maxvol_inds(tens_copy.data)
    else:
        raise ValueError
    return inds_r


def tensor_xr(tens: qtn.Tensor, phys_inds: list[str], right_inds: list[str],
              inds_r: Sequence[int], bond_ind='xx',):
    """ compress_split but for a single tensor in the MPS
    """
    left_inds = [ind for ind in tens.inds if (ind not in phys_inds + right_inds)]
    assert (len(left_inds) <= 1), 'not specific enough if have multiple virtual "left inds"'

    ## must follow specific order for fusing unshared bonds
    fuse_inds = phys_inds + left_inds
    tens = tens.transpose(*fuse_inds, *right_inds, inplace=True)
    shape_left = tens.shape[:len(fuse_inds)]
    shape_right = tens.shape[len(fuse_inds):]
    tens_ = tens.fuse({f'xx': fuse_inds, f'oo': right_inds})
    # nr, nc = q_.shape[0], q_.shape[-1]

    q, r = np.linalg.qr(tens_.data)
    submat = q[inds_r, :]
    x = q @ np.linalg.pinv(submat)
    r = tens_.data[inds_r, :]

    x = x.reshape(*shape_left, -1)
    r = r.reshape(-1, *shape_right)

    tens1 = qtn.Tensor(x, inds=(*fuse_inds, bond_ind))
    tens2 = qtn.Tensor(r, inds=(bond_ind, *right_inds))

    return tens1, tens2


def tensor_canonize_with_inds_1site(tens1: qtn.Tensor, tens2: qtn.Tensor, select_inds: Sequence[int],
                                    phys_inds: list[str], right_inds: list[str], inplace=False, do_qr=True,
                                    solver_type=DEFAULT_SOLVER):
    """ get suboptimal projection with specified inds
    """

    # raise NotImplementedError

    tens1 = tens1 if inplace else tens1.copy()
    tens2 = tens2 if inplace else tens2.copy()
    # print('tens1', tens1.shape, tens2.shape, do_qr)

    left_inds = [ind for ind in tens1.inds if (ind not in phys_inds + right_inds)]
    assert (len(left_inds) <= 1), 'not specific enough if have multiple virtual "left inds"'
    tens2_remaining_inds = [ind for ind in tens2.inds if ind not in right_inds]

    ## must follow specific order for fusing unshared bonds
    fuse_inds = phys_inds + left_inds
    tens1 = tens1.transpose(*fuse_inds, *right_inds, inplace=True)
    tens2 = tens2.transpose(*right_inds, *tens2_remaining_inds, inplace=True)
    # print('tens1 shape', tens1.shape, tens2.shape)

    # print('do qr', do_qr)
    if do_qr:
        q, r = qtn.tensor_split(tens1, fuse_inds, absorb='right', method='qr',
                                # method='svd', cutoff=0.0,
                                bond_ind=f'__tmp__')

        q.transpose_like(tens1, inplace=True)
        tens1.modify(data=q.data)
        rtens2 = qtn.tensor_contract(r, tens2)
        rtens2.transpose_like(tens2, inplace=True)
        tens2.modify(data=rtens2.data)

    # print('tens1 shape out', tens1.shape, tens2.shape)

    tens_ = tens1.fuse({f'xx': fuse_inds, f'oo': right_inds})

    # print('select inds tens', select_inds)
    tens_.modify(apply=lambda x: x[select_inds, :])  ## submat
    tens1.modify(apply=lambda x: np.tensordot(x, np.linalg.pinv(tens_.data), axes=(-1, 0)))
    tens2.modify(apply=lambda x: np.tensordot(tens_.data, x, axes=(-1, 0)))

    return tens1, tens2


def tensor_canonize_with_inds_2site(tens1: qtn.Tensor, tens2: qtn.Tensor, phys_inds: list[str], right_inds: list[str],
                                    inplace=False, do_qr=False, solver_type = DEFAULT_SOLVER, verbose=False):
    """ combine tens1 * tens2 before doing decomposition, for when len(select_inds) > size of remaining rank
    """

    if verbose:
        print('in tensor canonize with inds 2site')
    raise RuntimeError

    tens1 = tens1 if inplace else tens1.copy()
    tens2 = tens2 if inplace else tens2.copy()
    if verbose:
        print('tens1', tens1.shape, tens2.shape, do_qr)

    left_inds = [ind for ind in tens1.inds if (ind not in phys_inds + right_inds)]
    assert (len(left_inds) <= 1), 'not specific enough if have multiple virtual "left inds"'
    tens2_remaining_inds = [ind for ind in tens2.inds if ind not in right_inds]

    ## must follow specific order for fusing unshared bonds
    fuse_inds = phys_inds + left_inds
    tens1 = tens1.transpose(*fuse_inds, *right_inds, inplace=True)
    tens2 = tens2.transpose(*right_inds, *tens2_remaining_inds, inplace=True)

    tens = qtn.tensor_contract(tens1, tens2)

    ## v3
    shape1 = tens1.shape[:len(fuse_inds)]
    shape2 = tens2.shape[-len(tens2_remaining_inds):]
    tens_data = tens.data.reshape(np.prod(shape1), np.prod(shape2))
    # o1, o2 = cross_select(tens_data, select_inds, do_qr=True)
    if solver_type is CrossSolver.DEIM:
        o1, o2, inds_r = deim_split(tens_data, return_inds=True)
    else:
        o1, o2, inds_r = cross_split(tens_data, do_qr=True, return_inds=True)
    # o1, o2, inds_r, inds_c = cross_compress(tens_data, 4, do_qr=True, return_inds=True)
    # print('optimal inds_r', inds_r, inds_c)

    o1 = o1.reshape(*shape1, -1)
    tens1.modify(data=o1)
    o2 = o2.reshape(-1, *shape2)
    tens2.modify(data=o2)

    if verbose:
        print("CANONIZE SELECT")
        print('tens1', tens1.fuse({f'xx': fuse_inds, f'oo': right_inds}).data)

    # print('final Q Q^-1', (Q @ np.linalg.inv(submat))[inds, :])

    return tens1, tens2


def canonize_tens_list(tens_list: list['qtn.Tensor'], site_ind_ids: Sequence[str], site_inds: Sequence[int],
                       inplace=True, solver_type=DEFAULT_SOLVER):

    L = len(tens_list)
    tens_list = tens_list if inplace else [t.copy() for t in tens_list]
    select_inds_list = []

    for i in range(0, L-1):
        right_inds, unshared_inds = tens_list[i].filter_bonds(tens_list[i+1])
        ## shared and unshared bonds
        phys_inds = [s.format(site_inds[i]) for s in site_ind_ids]
        left_inds = [ind for ind in unshared_inds if ind not in phys_inds]
        assert(len(left_inds)<=1), 'not specific enough if have multiple virtual "left inds"'

        ## must follow specific order for fusing unshared bonds
        fuse_inds = phys_inds + left_inds

        ## A = A[:,cols] (A_)^-1 A[rows,:] = C (A_)^-1 R
        ## canon form: want C (A_)^-1
        ## C is essentially the tensor core
        ## Let C = QT -->
        ## --> C A_^-1 = Q T T^-1 Q_^-1 = Q Q_^-1

        tens = tens_list[i].transpose(*fuse_inds, *right_inds, inplace=True)
        shape_left = tens.shape[:len(fuse_inds)]
        shape_right = tens.shape[len(fuse_inds):]

        tens = tens.fuse({f'xx{i}': fuse_inds, f'oo{i}': right_inds})
        if solver_type is CrossSolver.DEIM:
            q, r, inds_r = deim_split(tens.data, return_inds=True)
        else:
            q, r, inds_r = cross_split(tens.data, do_qr=True, return_inds=True)
        q = q.reshape(*shape_left, -1)
        r = r.reshape(-1, *shape_right)

        tens_list[i].modify(data=q)

        r_tens = qtn.Tensor(r, inds=(f'tmp{i}', *right_inds))
        next_site = qtn.tensor_contract(r_tens, tens_list[i + 1])
        next_site.transpose_like(tens_list[i + 1], inplace=True)
        tens_list[i + 1].modify(data=next_site.data)

        select_inds_list += [inds_r]

    return tens_list, select_inds_list


def compress_tens_list(tens_list: list['qtn.Tensor'], site_ind_ids: Sequence[str], site_inds: Sequence[int],
                       inplace=True, max_bond=None, cutoff=CUTOFF, solver_type=DEFAULT_SOLVER):

    L = len(tens_list)
    tens_list = tens_list if inplace else [t.copy() for t in tens_list]
    select_inds_list = []

    for i in range(0, L-1):
        right_inds, unshared_inds = tens_list[i].filter_bonds(tens_list[i+1])
        ## shared and unshared bonds
        phys_inds = [s.format(site_inds[i]) for s in site_ind_ids]
        left_inds = [ind for ind in unshared_inds if ind not in phys_inds]
        assert(len(left_inds)<=1), 'not specific enough if have multiple virtual "left inds"'

        ## must follow specific order for fusing unshared bonds
        fuse_inds = phys_inds + left_inds

        ## A = A[:,cols] (A_)^-1 A[rows,:] = C (A_)^-1 R
        ## canon form: want C (A_)^-1
        ## C is essentially the tensor core
        ## Let C = QT -->
        ## --> C A_^-1 = Q T T^-1 Q_^-1 = Q Q_^-1

        tens = tens_list[i].transpose(*fuse_inds, *right_inds, inplace=True)
        shape_left = tens.shape[:len(fuse_inds)]
        shape_right = tens.shape[len(fuse_inds):]
        tens = tens.fuse({f'xx{i}': fuse_inds, f'oo{i}': right_inds})
        if solver_type is CrossSolver.DEIM:
            q, r, inds_r, inds_c = deim_compress(tens.data, max_bond=max_bond, cutoff=cutoff, return_inds=True)
        else:
            q, r, inds_r, inds_c = cross_compress(tens.data, do_qr=True, max_bond=max_bond, cutoff=cutoff, return_inds=True)
        q = q.reshape(*shape_left, -1)
        r = r.reshape(-1, *shape_right)

        tens_list[i].modify(data=q)

        r_tens = qtn.Tensor(r, inds=(f'tmp{i}', *right_inds))
        next_site = qtn.tensor_contract(r_tens, tens_list[i + 1])
        next_site.transpose_like(tens_list[i + 1], inplace=True)
        tens_list[i + 1].modify(data=next_site.data)

        select_inds_list += [inds_r]

    return tens_list, select_inds_list


def canonize(mps: 'qtn.MatrixProductState', i:int, cur_orthog:int=None, select_inds: dict[int,Sequence[int]]=None,
             solver_type=DEFAULT_SOLVER):
    """ inplace canonicalization via cross
    """
    # print('helper cross canonize select inds', select_inds, cur_orthog, i)

    tens_left, tens_right = [], []
    max_ind = mps.L - 1
    # i = mps.get_mps_ind(i)
    mps_inds = list(range(mps.L))
    updated_inds_l, updated_inds_r = [], []
    for lx in mps_inds:
        if (0 if cur_orthog is None else cur_orthog) <= lx <= i:
            updated_inds_l += [lx]
            tens_left += [mps.select_tensors((mps.site_tag_id.format(lx),))[0]]
        if (max_ind if cur_orthog is None else cur_orthog) >= lx >= i:
            if lx not in updated_inds_l:
                updated_inds_r += [lx]
            tens_right += [mps.select_tensors((mps.site_tag_id.format(lx),))[0]]

    _, sel_inds_list_l = canonize_tens_list(tens_left, [mps.site_ind_id], updated_inds_l, inplace=True,
                                            solver_type=solver_type)
    _, sel_inds_list_r = canonize_tens_list(tens_right[::-1], [mps.site_ind_id], updated_inds_r[::-1], inplace=True,
                                            solver_type=solver_type)

    ###
    select_inds = {} if select_inds is None else select_inds
    for i in range(len(sel_inds_list_l)):
        select_inds[updated_inds_l[i]] = sel_inds_list_l[i]
    for i in range(len(sel_inds_list_r)):
        select_inds[updated_inds_r[-1 - i]] = sel_inds_list_r[i]

    if mps.select_inds is None:
        mps.select_inds = select_inds
    else:
        mps.select_inds.update(select_inds)

    ## updated select inds; no update for site i
    # for i in range(len(sel_inds_list_l)):
    #     mps.ket_select_inds_l[updated_inds_l[i]] = sel_inds_list_l[i]
    # for i in range(len(sel_inds_list_r)):
    #     mps.ket_select_inds_r[updated_inds_r[-1 - i]] = sel_inds_list_r[i]

    # print('updated inds', updated_inds)
    # if mps.in_ind == mps.out_ind:
    #     mps.set_bra_from_ket(sites=updated_inds_l + updated_inds_r)
        ## bra select inds should have already been updated--same object as self.ket_select_inds_x

    # print('helper cross canonize new sel inds', select_inds)

    return mps, select_inds


def compress(mps: 'qtn.MatrixProductState', form='left', do_canonize=True,
             max_bond: int = None, cutoff: float = None, solver_type=DEFAULT_SOLVER):
    """ inplace canonicalization via cross
    """
    if form == 'left':
        if canonize:
            mps, select_inds = canonize(mps, 0)
        else:
            select_inds = mps.select_inds
        cur_orthog = 0
        i = mps.L - 1
    else:
        if canonize:
           mps, select_inds = canonize(mps, mps.L-1)
        else:
            select_inds = mps.select_inds
        cur_orthog = mps.L - 1
        i = 0

    tens_left, tens_right = [], []
    max_ind = mps.L - 1
    # i = mps.get_mps_ind(i)
    mps_inds = list(range(mps.L))
    updated_inds_l, updated_inds_r = [], []
    for lx in mps_inds:
        if (0 if cur_orthog is None else cur_orthog) <= lx <= i:
            updated_inds_l += [lx]
            tens_left += [mps.select_tensors((mps.site_tag_id.format(lx),))[0]]
        if (max_ind if cur_orthog is None else cur_orthog) >= lx >= i:
            if lx not in updated_inds_l:
                updated_inds_r += [lx]
            tens_right += [mps.select_tensors((mps.site_tag_id.format(lx),))[0]]

    _, sel_inds_list_l = compress_tens_list(tens_left, [mps.site_ind_id], updated_inds_l, inplace=True,
                                            max_bond=max_bond, cutoff=cutoff,
                                            solver_type=solver_type)
    _, sel_inds_list_r = compress_tens_list(tens_right[::-1], [mps.site_ind_id], updated_inds_r[::-1], inplace=True,
                                            max_bond=max_bond, cutoff=cutoff,
                                            solver_type=solver_type)

    ###
    select_inds = {} if select_inds is None else select_inds
    for i in range(len(sel_inds_list_l)):
        select_inds[updated_inds_l[i]] = sel_inds_list_l[i]
    for i in range(len(sel_inds_list_r)):
        select_inds[updated_inds_r[-1 - i]] = sel_inds_list_r[i]

    if mps.select_inds is None:
        mps.select_inds = select_inds
    else:
        mps.select_inds.update(select_inds)

    ## updated select inds; no update for site i
    # for i in range(len(sel_inds_list_l)):
    #     mps.ket_select_inds_l[updated_inds_l[i]] = sel_inds_list_l[i]
    # for i in range(len(sel_inds_list_r)):
    #     mps.ket_select_inds_r[updated_inds_r[-1 - i]] = sel_inds_list_r[i]

    # print('updated inds', updated_inds)
    # if mps.in_ind == mps.out_ind:
    #     mps.set_bra_from_ket(sites=updated_inds_l + updated_inds_r)
        ## bra select inds should have already been updated--same object as self.ket_select_inds_x

    # print('helper cross canonize new sel inds', select_inds)

    return mps, select_inds



def target_MPS(*mps_list: MPS, inplace=False, direction = SweepDirection.RIGHT,
               max_bond: int = None, do_canonize=True, solver_type=DEFAULT_SOLVER):
    """ project mps onto space shared by all mps
        i.e., select specific inds and obtain resulting approximation
        but this may not be a good/accurate approximation.
    """
    ref_mps = next(iter(mps_list))
    L = ref_mps.L
    site_ind_id = ref_mps.site_ind_id

    if not inplace:
        mps_list = [mps.copy() for mps in mps_list]

    if do_canonize:
        for mps in mps_list:
            if direction > 0:
                canonize(mps, 0)
            else:
                canonize(mps, L - 1)

    for mps in mps_list[1:]:
        helper_quimb.match_inner_inds(mps, ref_mps, inplace=True)

    if direction == SweepDirection.RIGHT:
        for i in range(L - 1):
            ind1, ind2 = i, i + 1

            right_inds = [ref_mps.bond(ind2, ind2 + 1)] if i < L - 1 else []
            right_inds = [site_ind_id.format(ind2)] + right_inds

            iso_tensors = []
            for mps in mps_list:
                tens1, tens2 = mps[i], mps[i + 1]
                iso_tensors += [qtn.tensor_contract(tens1, tens2)]

            inds_r, inds_c, new_tensors = tensor_compress_multiple(*iso_tensors,
                                                                   phys_inds=[site_ind_id.format(i)],
                                                                   right_inds=right_inds,
                                                                   max_bond=max_bond, return_inds=True,
                                                                   solver_type=solver_type)
            for mps, (t1, t2) in zip(mps_list, new_tensors):
                tens1, tens2 = mps[i], mps[i + 1]
                t1.transpose_like(tens1, inplace=True)
                tens1.modify(data = t1.data)
                t2.transpose_like(tens2, inplace=True)
                tens2.modify(data=t2.data)
                mps.select_inds[i] = inds_r

    else:
        for i in range(L - 1, 0, -1):
            ind1, ind2 = i, i - 1
            iso_tensors = []
            for mps in mps_list:
                tens1, tens2 = mps[i], mps[i - 1]
                iso_tensors += [qtn.tensor_contract(tens1, tens2)]

            right_inds = [ref_mps.bond(ind2, ind2 - 1)] if i > 0 else []
            right_inds = [site_ind_id.format(ind2)] + right_inds

            inds_r, inds_c, new_tensors = tensor_compress_multiple(*iso_tensors,
                                                                   phys_inds=[site_ind_id.format(i)],
                                                                   right_inds=right_inds,
                                                                   max_bond=max_bond, return_inds=True,
                                                                   solver_type=solver_type)
            for mps, (t1, t2) in zip(mps_list, new_tensors):
                tens1, tens2 = mps[i], mps[i - 1]
                t1.transpose_like(tens1, inplace=True)
                tens1.modify(data=t1.data)
                t2.transpose_like(tens2, inplace=True)
                tens2.modify(data=t2.data)
                mps.select_inds[i] = inds_r

    return mps_list



def add_MPS_list(mps_list: Sequence[MPS], inplace=False, direction=1, do_final_update=True, do_canonize=True,
                 compress_opts: dict=None):
    """ if do_final_update: use local scheme.
            would it e more efficient to just sample these data points from the TT?
            if one uses a bad submatrix at one step, then can one even obtain an accurate submatrix later on?
            i think the idea is that if [I B]
        else:
            do basis expansion.
    """
    raise NotImplementedError


def update_1site(mps: MPS, left_site_pos: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection',
                 select_inds: Sequence[int] = None, max_bond:int = None, decimate_only=False,
                 solver_type=DEFAULT_SOLVER, plot_verbosity:int = 0, verbose=False):
    """ update ket, bra with new_site; list of sites --> target these separately.
        i: int of mps site
        don't actually do the update--just select indices
        would this actually work?
        select_inds acts as an initial guess
    """
    if isinstance(site_i, (tuple, list)):
        site_i = helper_tn.sum_tens(site_i)
    # if isinstance(site_i, qtn.Tensor):
    #     site_i = [site_i]

    if mps.select_inds is None:
        mps.select_inds = {}


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

    if ind2 is None:   ## is at end
        ## update mps
        if not decimate_only:
            # print('at end not decimate only')
            site_i.transpose_like(mps[ind1], inplace=True)
            mps[ind1].modify(data=site_i.data)
        else:
            pass
            # print('did not update mps')

    else:

        # x_bond = mps.bond(left_site_pos, left_site_pos + direction)  ## bond to not contract over
        # fuse_inds = [mps.phys_dim(ind1)]
        # if 0 <= (left_site_pos - direction) <= mps.L - 1:
        #     fuse_inds += [mps.bond(left_site_pos, left_site_pos - direction)]
        # ket_iso = [ind for ind in site_i[0].inds if ind != x_bond]   ## bonds to fuse
        phys_inds = [mps.site_ind(ind1)]
        right_inds = [mps.bond(left_site_pos, left_site_pos + direction)]   ## bond to not contract over

        if decimate_only:
            if select_inds is None:
                q_dmp, r_dmp, sel_idxs, ind_c = tensor_compress(site_i, phys_inds, right_inds, max_bond=max_bond,
                                                                return_inds=True, solver_type=solver_type)
                # print('dec only select inds', sel_idxs)
            else:
                sel_idxs = select_inds

            tensor_canonize_with_inds_1site(mps[ind1], mps[ind2], sel_idxs, phys_inds, right_inds, inplace=True,)
            mps.select_inds[ind1] = sel_idxs

        else:

            if plot_verbosity:
                print('ind 1', ind1)
                check_orthog(mps)

                plt.figure()
                plt.plot(mps.to_dense(), label='init')

                nsites = 1
                coords = get_selectors(mps, left_site_pos, nsites)

                selectors = []
                for c in coords:
                    selectors += [int("".join(str(x) for x in c), 2)]

                inds = []
                if left_site_pos > 0:
                    inds += [mps.bond(left_site_pos, left_site_pos - 1)]
                inds += [mps.site_ind(left_site_pos + i) for i in range(nsites)]
                if left_site_pos + nsites < mps.L:
                    inds += [mps.bond(left_site_pos + nsites - 1, left_site_pos + nsites)]

                site_tens = mps[ind1].transpose(*inds, inplace=False)
                plt.plot(selectors, site_tens.data.reshape(-1), 'x', label='init')

            site_i.transpose_like(mps[ind1], inplace=True)
            mps[ind1].modify(data=site_i.data)

            if plot_verbosity:
                plt.plot(mps.to_dense(), label='sub')
                site_tens = mps[ind1].transpose(*inds, inplace=False)
                plt.plot(selectors, site_tens.data.reshape(-1), 'x', label='sub')

            if select_inds is None:
                # _, select_inds = compress_tens_list([mps[ind1], mps[ind2]],
                #                                     site_ind_ids=[mps.site_ind_id], site_inds=[ind1, ind2],
                #                                     inplace=True, max_bond=max_bond)
                _, select_inds = canonize_tens_list([mps[ind1], mps[ind2]],
                                                    site_ind_ids=[mps.site_ind_id], site_inds=[ind1, ind2],
                                                    inplace=True, solver_type=solver_type)
                # print('select inds', select_inds)
                mps.select_inds[ind1] = select_inds[0]
                sel_idxs = select_inds[0]

            else:
                sel_idxs = select_inds
                tensor_canonize_with_inds_1site(mps[ind1], mps[ind2], sel_idxs, phys_inds, right_inds, inplace=True,
                                                solver_type=solver_type)
                mps.select_inds[ind1] = sel_idxs

            if plot_verbosity:
                plt.plot(mps.to_dense(), ':', label='canon')
                plt.legend()
                plt.show()

        mps._cur_orthog = ind2

        return sel_idxs


def update_2site(mps: 'MPS', left_site_pos: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection',
                 inds_r: Sequence[int] = None, inds_c: Sequence[int] = None,
                 max_bond: int=None, plot_verbosity: int = 0):
    """ update ket, bra with new_site
        i: mps_site
        only decimate / only keep select inds --- can't guarantee select_inds are consistent?
    """

    if inds_c is not None and inds_r is not None:
        return decimate_2site(mps, left_site_pos, inds_r, inds_c, direction)


    if isinstance(site_i, (tuple, list)):
        site_i = helper_tn.sum_tens(site_i)

    # print('update 2 site', site_i.norm(), direction)
    if direction == SweepDirection.RIGHT:
        ind1, ind2 = left_site_pos, left_site_pos + 1   # self.mps_inds[i], self.mps_inds[i + 1]
        split_inds, unshared_inds = mps[ind1].filter_bonds(mps[ind2])  # shared, not shared
        right_inds = [ik for ik in mps[ind2].inds if ik not in split_inds]

        phys_inds = [mps.site_ind_id.format(ind1)]
        left_inds = [ind for ind in unshared_inds if ind not in phys_inds]
        left_inds = phys_inds + left_inds

        site_i.transpose(*left_inds, *right_inds, inplace=True)
        shape_l = site_i.shape[:len(left_inds)]
        shape_r = site_i.shape[-len(right_inds):]
        site_i_mat = site_i.data.reshape(np.prod(shape_l), np.prod(shape_r))

        mat1, mat2, inds_r, inds_c = cross_compress(site_i_mat, max_bond, do_qr=True, return_inds=True,
                                                    inds_r_guess=inds_r, inds_c_guess=inds_c)

        mat1 = mat1.reshape(shape_l + (-1,))
        mat2 = mat2.reshape((-1,) + shape_r)

        ## update ket
        mps.select_inds[ind1] = inds_r
        mps.select_inds[ind2] = inds_c
        mps[ind1].modify(data=mat1, inds=tuple(left_inds) + tuple(split_inds))
        mps[ind2].modify(data=mat2, inds=tuple(split_inds) + tuple(right_inds))

    else:
        ind1, ind2 = left_site_pos + 1, left_site_pos
        split_inds, unshared_inds = mps[ind1].filter_bonds(mps[ind2])  # shared, not shared
        right_inds = [ik for ik in mps[ind2].inds if ik not in split_inds]

        phys_inds = [mps.site_ind_id.format(ind1)]
        left_inds = [ind for ind in unshared_inds if ind not in phys_inds]
        left_inds = phys_inds + left_inds

        site_i.transpose(*left_inds, *right_inds, inplace=True)
        shape_l = site_i.shape[:len(left_inds)]
        shape_r = site_i.shape[-len(right_inds):]
        site_i_mat = site_i.data.reshape(np.prod(shape_l), np.prod(shape_r))
        # mat1, mat2, select_inds = cross_split(site_i_mat, do_qr=True, return_inds=True)

        mat1, mat2, inds_r, inds_c = cross_compress(site_i_mat, max_bond, do_qr=True, return_inds=True,
                                                    inds_r_guess=inds_r, inds_c_guess=inds_c
                                                    )

        mat1 = mat1.reshape(shape_l + (-1,))
        mat2 = mat2.reshape((-1,) + shape_r)

        ## update ket
        mps[ind1].modify(data=mat1, inds=tuple(left_inds) + tuple(split_inds))
        mps[ind2].modify(data=mat2, inds=tuple(split_inds) + tuple(right_inds))
        mps.select_inds[ind1] = inds_r
        mps.select_inds[ind2] = inds_c

    mps._cur_orthog = ind2

    return inds_r, inds_c


def _update_2site_svd(mps: 'MPS', left_site_pos: int, site_i: 'qtn.Tensor', direction: 'SweepDirection',
                      max_bond: int = None):
    """ update ket, bra with new_site
        i: mps_site
    """
    if isinstance(site_i, (tuple, list)):
        site_i = helper_tn.sum_tens(site_i)

    if direction == SweepDirection.RIGHT:
        ind1, ind2 = left_site_pos, left_site_pos + 1  # self.mps_inds[i], self.mps_inds[i + 1]
        split_inds, left_inds = mps[ind1].filter_bonds(mps[ind2])  # shared, not shared
        site1, site2 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=max_bond)

        ## update ket
        mps[ind1].modify(data=site1.data, inds=site1.inds)
        mps[ind2].modify(data=site2.data, inds=site2.inds)

        _, select_inds = canonize_tens_list([mps[ind1], mps[ind2]],
                                             site_ind_ids=[mps.site_ind_id],
                                             site_inds=[ind1, ind2], inplace=True)
        mps.select_inds[ind1] = select_inds[0]

    else:
        ind1, ind2 = left_site_pos + 1, left_site_pos
        split_inds, left_inds = mps[ind1].filter_bonds(mps[ind2])  # inds only on ket[i]
        site2, site1 = site_i.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=max_bond)

        ## update ket
        mps[ind1].modify(data=site2.data, inds=site2.inds)
        mps[ind2].modify(data=site1.data, inds=site1.inds)

        _, select_inds = canonize_tens_list([mps[ind1], mps[ind2]],
                                            site_ind_ids=[mps.site_ind_id],
                                            site_inds=[ind1, ind2], inplace=True)
        mps.select_inds[ind1] = select_inds[0]

    mps._cur_orthog = ind2

    return


def decimate_1site(mps: 'MPS', left_site_pos: int, select_inds: Sequence[int], direction: 'SweepDirection',):
    """ update ket, bra with new_site
        i: mps_site
        only decimate / only keep select inds --- can't guarantee select_inds are consistent?
    """
    # raise NotImplementedError
    # if isinstance(site_i, (tuple, list)):
    #     site_i = helper_tn.sum_tens(site_i)

    # print('update 2 site', site_i.norm(), direction)
    if direction == SweepDirection.RIGHT:
        ind1, ind2 = left_site_pos, left_site_pos + 1   # self.mps_inds[i], self.mps_inds[i + 1]

        left_inds, phys_inds, right_inds = get_inds(mps, ind1)
        tensor_canonize_with_inds_1site(mps[ind1], mps[ind2], select_inds, phys_inds, right_inds,
                                  inplace=True)

    else:
        ind1, ind2 = left_site_pos, left_site_pos - 1

        left_inds, phys_inds, right_inds = get_inds(mps, ind1)
        tensor_canonize_with_inds_1site(mps[ind1], mps[ind2], select_inds, phys_inds, left_inds,
                                  inplace=True)

    mps.cur_orthog = ind2
    mps.select_inds[ind1] = select_inds

    return


def decimate_2site(mps: 'MPS', left_site_pos: int, inds_r: Sequence[int], inds_c: Sequence[int],
                   direction: 'SweepDirection', verbose=False):
    """ update ket, bra with new_site
        i: mps_site
        only decimate / only keep select inds --- can't guarantee select_inds are consistent?
    """
    check_orthog(mps)

    # print('update 2 site', site_i.norm(), direction)
    if direction == SweepDirection.RIGHT:
        ind1, ind2 = left_site_pos, left_site_pos + 1   # self.mps_inds[i], self.mps_inds[i + 1]
        left_inds, phys_inds, right_inds = get_inds(mps, ind1)
        tensor_canonize_with_inds_2site(mps[ind1], mps[ind2], inds_r, phys_inds, right_inds,
                                  inplace=True)

    else:
        ind1, ind2 = left_site_pos + 1, left_site_pos
        left_inds, phys_inds, right_inds = get_inds(mps, ind1)
        tensor_canonize_with_inds_2site(mps[ind1], mps[ind2], inds_c, phys_inds, left_inds,
                                  inplace=True)

    mps.cur_orthog = ind2
    mps.select_inds[ind1] = inds_r
    mps.select_inds[ind2] = inds_c

    if verbose:
        print('ind1', ind1, ind2, direction)
    check_orthog(mps)

    return

def update_ket(mps: 'MPS', tensors: Union[qtn.Tensor, Sequence[qtn.Tensor]], i: int, nsites: int,
               direction: SweepDirection, max_bond: int = None, cutoff: float = None, decimate_only=False):
    """ update ket with corresponding (targeting) tensors
        update mps_list[i] with tensors[i][0] (ideally the original tensor if just doing decimation)
        target all tensors in "tensors" list; shared across all "mps"
    """
    if isinstance(tensors, qtn.Tensor):
        tensors = [tensors]

    ref_mps = mps

    L = ref_mps.L
    site_ind_id = ref_mps.site_ind_id
    left_site_pos = i if (nsites == 1 or direction == SweepDirection.RIGHT) else i - 1
    at_end = (left_site_pos == L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

    ### targets:  [out] or [original tensor, *targets]
    if direction == SweepDirection.RIGHT:
        if nsites == 1:
            ind1, ind2 = i, i + 1
            phys_inds = [ref_mps.site_ind(ind1)]
            right_inds = [ref_mps.bond(ind1, ind2)] if not at_end else []
        elif nsites == 2:
            ind1, ind2 = i, i + 1
            phys_inds = [ref_mps.site_ind(ind1)]
            right_inds = [ref_mps.site_ind(ind2)]
            if not at_end:
                right_inds += [ref_mps.bond(ind2, ind2 + 1)]
            ## right_inds:  phys + virtual
        else:
            raise NotImplementedError
    else:
        if nsites == 1:
            ind1, ind2 = i, i - 1
            phys_inds = [ref_mps.site_ind(ind1)]
            right_inds = [ref_mps.bond(ind1, ind2)] if not at_end else []
        elif nsites == 2:
            ind2, ind1 = i - 1, i
            phys_inds = [ref_mps.site_ind(ind1)]
            right_inds = [ref_mps.site_ind(ind2)]
            if not at_end:
                right_inds += [ref_mps.bond(ind2, ind2 - 1)]
            ## right_inds:  phys + virtual
        else:
            raise NotImplementedError

    if at_end and decimate_only:
        return

    if not at_end or nsites == 2:
        T1, T2, inds_r, inds_c = tensor_compress_expand(tensors, phys_inds, right_inds,
                                                        max_bond=max_bond, cutoff=cutoff,)
        # inds_r, inds_c, new_cur_tens = tensor_compress_multiple(tensors, phys_inds, right_inds,
        #                                                          max_bond=max_bond, cutoff=cutoff)
        # T1, T2 = new_cur_tens

        it = 0

        if mps.select_inds is None:
            mps.select_inds = {}

        # print('update kets check orthog', ind1)
        # chk = check_orthog(mps)
        # if chk[0] != chk[1]:
        #     print('mps', it)
        #     exit()

        it += 1

        mps.select_inds[ind1] = inds_r
        mps.select_inds[ind2] = inds_c

        site1, site2 = mps[ind1], mps[ind2]

        T1.transpose_like(site1, inplace=True)
        site1.modify(data=T1.data)
        if nsites == 1:
            ind_r = right_inds[-1]   # bond between site1, site2
            size_r1 = site2.ind_size(ind_r)
            size_r2 = T2.ind_size(ind_r)
            if size_r1 != size_r2:
                ### need to expand site2 to size of T2 by padding with zeros (decimate only)
                site2.transpose(ind_r, *[ix for ix in site2.inds if ix != ind_r], inplace=True)
                site2_x = np.zeros((size_r2, *site2.shape[1:]), dtype=site2.dtype)
                site2_x[:size_r1] = site2.data
                site2.modify(data=site2_x)
            T2 = qtn.tensor_contract(T2, site2)
            T2.transpose_like(site2, inplace=True)
            site2.modify(data=T2.data)
            mps._cur_orthog = ind2
        elif nsites == 2:
            # print('T1', T1)
            # print('T2', T2)

            if not at_end:
                for r_ind in right_inds:
                    r_size = tensors[0].ind_size(r_ind)
                    T2_data = T2.data
                    r_idx = T2.inds.index(r_ind)
                    T2_data = np.moveaxis(T2_data, r_idx, 0)
                    T2_data = T2_data[:r_size]
                    T2_data = np.moveaxis(T2_data, 0, r_idx)
                    T2.modify(data=T2_data)
                    # T2.isel({r_ind: slice(r_size)}, inplace=True)

            # print('T2', T2)
            # pdb.set_trace()

            ## need to select only the original state (equivalent to padding next tensor)
            T2.transpose_like(site2, inplace=True)
            site2.modify(data=T2.data)

            # print(' new update kets check orthog', ind1)
            mps._cur_orthog = ind2

            # plt.figure()
            # plt.plot(mps_init, label='pre update')
            # plt.plot(helper_quimb.to_dense(mps), '--', label='after update')
            # plt.title(f'mps {it}')
            # plt.legend()
            # plt.show()

    else:
        ## update without canonicalization (only if nsites == 1)
        # print('at end', tensors)
        # pdb.set_trace()
        tens = tensors[0]
        site1 = mps[ind1]
        T1 = tens.transpose_like(site1, inplace=False)
        site1.modify(data=T1.data)

        mps._cur_orthog = ind1

    return


def update_kets(mps_list: Sequence['MPS'], tensors: Sequence[Sequence[qtn.Tensor]], i: int, nsites: int,
                direction: SweepDirection, max_bond: int = None, cutoff: float = None):
    """ update all kets in mps_list with corresponding tensors
        update mps_list[i] with tensors[i][0] (ideally the original tensor if just doing decimation)
        target all tensors in "tensors" list; shared across all "mps"
    """

    ref_mps = next(iter(mps_list))

    L = ref_mps.L
    site_ind_id = ref_mps.site_ind_id
    left_site_pos = i if (nsites == 1 or direction == SweepDirection.RIGHT) else i - 1
    at_end = (left_site_pos == L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

    ### targets:  out, intermediate kets, ...

    if direction == SweepDirection.RIGHT:
        if nsites == 1:
            ind1, ind2 = i, i + 1
            phys_inds = [ref_mps.site_ind(ind1)]
            right_inds = [ref_mps.bond(ind1, ind2)] if not at_end else []
        elif nsites == 2:
            ind1, ind2 = i, i + 1
            phys_inds = [ref_mps.site_ind(ind1)]
            right_inds = [ref_mps.site_ind(ind2)]
            if not at_end:
                right_inds += [ref_mps.bond(ind2, ind2 + 1)]
            ## right_inds:  phys + virtual
        else:
            raise NotImplementedError
    else:
        if nsites == 1:
            ind1, ind2 = i, i - 1
            phys_inds = [ref_mps.site_ind(ind1)]
            right_inds = [ref_mps.bond(ind1, ind2)] if not at_end else []
        elif nsites == 2:
            ind2, ind1 = i - 1, i
            phys_inds = [ref_mps.site_ind(ind1)]
            right_inds = [ref_mps.site_ind(ind2)]
            if not at_end:
                right_inds += [ref_mps.bond(ind2, ind2 - 1)]
            ## right_inds:  phys + virtual
        else:
            raise NotImplementedError


    if not at_end or nsites == 2:
        inds_r, inds_c, new_cur_tens = tensor_compress_multiple(tensors, phys_inds, right_inds,
                                                                 max_bond=max_bond, cutoff=cutoff)

        it = 0
        for mps, (T1, T2) in zip(mps_list, new_cur_tens):

            # mps_init = mps.to_dense()

            if mps.select_inds is None:
                mps.select_inds = {}

            # print('update kets check orthog', ind1)
            # chk = check_orthog(mps)
            # if chk[0] != chk[1]:
            #     print('mps', it)
            #     exit()

            it += 1

            mps.select_inds[ind1] = inds_r
            mps.select_inds[ind2] = inds_c

            site1, site2 = mps[ind1], mps[ind2]

            # print('T1', T1)
            # print('T2', T2)
            # print('site1', site1)
            # print('site2', site2)

            T1.transpose_like(site1, inplace=True)
            site1.modify(data=T1.data)
            if nsites == 1:
                T2 = qtn.tensor_contract(T2, site2)
                T2.transpose_like(site2, inplace=True)
                site2.modify(data=T2.data)
            elif nsites == 2:
                T2.transpose_like(site2, inplace=True)
                site2.modify(data=T2.data)

                # if at_end:
                #     T2.transpose_like(site2, inplace=True)
                #     site2.modify(data=T2.data)
                # else:
                #     T2 = site1 @ site2 @ T1.conj()
                #     T2.transpose_like(site2, inplace=True)
                #     site2.modify(data=T2.data)

            # print(' new update kets check orthog', ind1)
            mps._cur_orthog = ind2

            # plt.figure()
            # plt.plot(mps_init, label='pre update')
            # plt.plot(helper_quimb.to_dense(mps), '--', label='after update')
            # plt.title(f'mps {it}')
            # plt.legend()
            # plt.show()

    else:
        ## update without canonicalization (only if nsites == 1)
        for mps, targets in zip(mps_list, tensors):
            # print('update kets check orthog', ind1)
            # check_orthog(mps)
            tens = targets[0]

            site1 = mps[ind1]
            T1 = tens.transpose_like(site1, inplace=False)
            site1.modify(data=T1.data)

            mps._cur_orthog = ind1

    return


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


def cross_split(A: np.ndarray, do_qr=False, cutoff: float=None, max_iters=DEFAULT_MAX_TOT_ITER, return_inds=False,
                sel_inds_guess:Sequence[int]=None, verbose=False):
    ### decompose matrix using cross interpolation; obtain low-rank approximation

    # raise RuntimeError

    n, r = A.shape  ## A is 2D matrix: (alpha_{i-1} * d_{i}) x alpha_{i}

    do_qr = True

    if do_qr or n < r:
        # Q, R_dmp = np.linalg.qr(A) #, mode='complete')

        u, s, vt = np.linalg.svd(A, full_matrices=False)
        if cutoff is not None:
            # cum_sum = np.cumsum((s[::-1]) ** 2 / s[0] ** 2)  # ordered smallest to largest
            # cut_ind = np.argmax(cum_sum < cutoff)
            cum_sum = np.cumsum((s[::-1]) ** 2 / np.linalg.norm(s) ** 2)  # ordered smallest to largest
            cut_ind = np.argmin(cum_sum < cutoff)
            if cut_ind != 0:
                if verbose:
                    print('cross split cutoff', cutoff, 'cut ind', -cut_ind,
                          'err', np.linalg.norm(s[-cut_ind:]) / np.linalg.norm(s))
                u = u[:, :-cut_ind]
                s = s[:-cut_ind]
                vt = vt[:-cut_ind, :]


        Q = u
        R_dmp = (vt.T * s).T

        do_qr = True
    else:
        Q = A.copy()

    n, r = Q.shape      # n x n (if n < r) or n x r

    if sel_inds_guess is None:
        p, l_dmp, u_dmp = linalg.lu(Q, p_indices=True)  # A = P L U (permutation, lower tri with unit diagonal, upper tri)
        p = np.argsort(p)       # so that P A = L U, as in matlab
        inds = p[:r]  ## p is a list of indices (instead of full permutation matrix)
    else:
        inds = sel_inds_guess

    submat = Q[inds, :]  # permute rows of A

    B = regularized_solve(Q, submat, use_new_method=False)
    # B = Q @ np.linalg.pinv(submat)       # Q Q^-1 = B = [I, Z]
    ## note: np.linalg.inv(submat) gives bad results if n > r

    ## start iterations
    it, err = 0, 1.0
    while it < max_iters:  # and err < conv_tol:
        # print('B', B)
        max_ind = np.argmax(np.abs(B))
        i0, j0 = max_ind // (B.shape[1]), max_ind % (B.shape[1])
        max_val = B[i0, j0]
        # print('i0,j0', i0, j0, max_val,'\n')
        if np.abs(max_val) <= 1 + 1.0e-8:
            break

        ## rank 1 update of B from switching rows i0 and inds[j0]
        old_row = inds[j0]
        B = B + np.outer(B[:, j0], (B[old_row, :] - B[i0, :])) / B[i0, j0]
        inds[j0] = i0  ## update old row with new row; other row not included in inds (r largest components)

        it += 1

    ## A = A[:,cols] (A_)^-1 A[rows,:] = C (A_)^-1 R
    ## canon form: want C (A_)^-1
    ## C is essentially the tensor core
    ## Let C = QT -->
    ## --> C A_^-1 = Q T T^-1 Q_^-1 = Q Q_^-1

    T1 = B
    submat = Q[inds,:]
    T2 = (submat @ R_dmp) if do_qr else (submat)

    # print('final B', B[p,:], np.max(B))
    # print('final B', B[inds, :], np.max(B))
    # print('final B', B[np.sort(inds), :], np.max(B))
    # submat = Q[inds, :]
    # print('final Q Q^-1', (Q @ np.linalg.inv(submat))[inds,:])

    # print('err B', np.linalg.norm(B - (Q @ np.linalg.inv(submat))))
    # print('err A', np.linalg.norm(A[:, :] - T1 @ T2))

    if return_inds:
        return T1, T2, inds  # np.sort(inds)
    else:
        return T1, T2


def cross_compress(A: np.ndarray, max_bond, do_qr=True, max_iters=DEFAULT_MAX_TOT_ITER, return_inds=False,
                   cutoff: float = None, conv_tol=DEFAULT_CONV_TOL, include_inds_r=None,
                   inds_r_guess: Sequence[int] = None, inds_c_guess: Sequence[int] = None, verbose=False):
    ### decompose matrix using cross interpolation; obtain low-rank approximation

    # raise RuntimeError
    do_qr = True

    n, r = A.shape  ## A is 2D matrix:  e.g 2 site tensor, (alpha_{i-1} * d_{i}) x alpha_{i+1} * d{i+1}

    if max_bond is None or n < max_bond or r < max_bond:
        T1, T2, inds_r =  cross_split(A, do_qr=do_qr, cutoff=cutoff, max_iters=max_iters, return_inds=return_inds,
                                      sel_inds_guess=inds_r_guess)
        if return_inds:
            inds_c = list(range(r))
            return T1, T2, inds_r, inds_c
        else:
            return T1, T2

    r = int( min(r,max_bond) )

    ## randomly select r elements in A; make A n x r
    # inds_r = slice(0, r, 1) if inds_r_guess is None else inds_r_guess
    inds_r = list(range(r)) if inds_r_guess is None else inds_r_guess
    inds_c = inds_c_guess

    prev_A = np.zeros(A.shape, dtype=A.dtype)
    it = 0

    while it < max_iters:

        RA = A[inds_r, :].T
        inds_c = maxvol_inds(RA, do_qr=do_qr, inds_r_guess=inds_c)

        # print('compress split', inds_r, inds_c)

        CA = A[:, inds_c]
        if cutoff is not None:
            if verbose:
                print('Warning: finite cutoff being used in cross split', cutoff)
        T1, submat, inds_r = cross_split(CA, do_qr=do_qr, cutoff=cutoff, max_iters=max_iters, return_inds=True,
                                         sel_inds_guess=inds_r)

        it_A = T1 @ A[inds_r, :]        ## A = A[:, inds_c] @ submat^-1 @ A[inds_r, :]
        err = np.linalg.norm(it_A - prev_A) / np.linalg.norm(it_A)
        # print('err', err)
        prev_A = it_A
        it += 1

        if it == max_iters:
            T2 = A[inds_r, :]
            if verbose:
                print('compress err', err)

        if err < conv_tol:
            T2 = A[inds_r, :]
            break

    if return_inds:
        return T1, T2, inds_r, inds_c  # np.sort(inds)
    else:
        return T1, T2


# def cross_compress_multiple(*As: np.ndarray, max_bond:int = None, do_qr=True, max_iters=DEFAULT_MAX_TOT_ITER,
#                             conv_tol=DEFAULT_CONV_TOL):
#     """ decompose multiple matrices using cross interpolation, such that they all share the same row/column selectors
#         assumes that all are of the same shape
#     """
#
#     n, r = next(iter(As)).shape  ## A is 2D matrix:  e.g 2 site tensor, (alpha_{i-1} * d_{i}) x alpha_{i+1} * d{i+1}
#
#     print('cross compress multiple', max_bond)
#     if max_bond is None or n < max_bond or r < max_bond:
#         T1, T2, inds_r = cross_split(As[0], do_qr=do_qr, max_iters=max_iters, return_inds=True)
#         inds_c = []
#         cur_tens = [(T1, T2)]
#
#         # inds_r = list(range(min(n, r)))
#         # inds_c = inds_r
#         #
#         # cur_tens = []
#         for A in As[1:]:
#             # print('A', A.shape)
#             # print('inds_c', inds_c)
#             T1, T2 = cross_select(A, inds_r, do_qr=do_qr)
#             cur_tens += [(T1, T2)]
#
#         return inds_r, inds_c, cur_tens
#
#     ### o.w., actually need to do compression
#
#     print('START')
#
#     # get initial inds_r, inds_c for first A
#     tot_inds_r, tot_inds_c = set(), set()
#     for A in As:
#         ref_inds_r = list(tot_inds_r) if len(tot_inds_r) > 0 else None
#         ref_inds_c = list(tot_inds_c) if len(tot_inds_c) > 0 else None
#         r = max(r, len(ref_inds_r))
#
#         T1, T2, inds_r, inds_c = cross_compress(A, r, do_qr=True, return_inds=True,
#                                                 inds_r_guess=ref_inds_r, inds_c_guess=ref_inds_c)
#
#         # print('indss', inds_r, inds_c)
#         if len(inds_r) == 0:
#             print('len inds r is 0')
#             exit()
#         tot_inds_r = tot_inds_r.union(set(inds_r))
#         tot_inds_c = tot_inds_c.union(set(inds_c))
#         # print('tot inds r', tot_inds_r)
#
#     ## todo: is there a way to sort inds_r, inds_c by importance?  maybe LU decomposition does it?
#     ## todo: adaptive Cross decomposition?
#
#     # print('FINAL tot inds r', tot_inds_r)
#     inds_r = list(tot_inds_r)
#     inds_c = list(tot_inds_c)
#
#     cur_tens = []
#     for A in As:
#         CA = A[:, inds_c]
#         T1, submat = cross_select(CA, inds_r, do_qr=True)  # T1:  T1[inds_r, :] = eye()
#         T2 = A[inds_r, :]
#
#         if T1.shape[1] == 0:
#             print('inds_r', inds_r)
#             print('T1 shape is 0', T1, T2)
#             exit()
#         cur_tens += [(T1, T2)]
#
#     return inds_r, inds_c, cur_tens

########################
###   deim methods   ###
########################


def deim_inds_old(W: 'np.ndarray', max_r: int = None, include_inds=None):
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


def deim_inds_1(W: 'np.ndarray', max_r: int = None, include_inds=None):
    """
    note that we're doing XR instead of CX decomposition
    deim instead of maxvol
    :param W:  n x r matrix whose columns are singular vectors. want to find r x r submatrix of A
    :param max_iters:
    :param conv_tol:
    :param do_qr:
    :return: inds to obtain submatrix of A

    """
    max_r = min(max_r, W.shape[1]) if max_r is not None else W.shape[1]
    indices = np.zeros(max_r, dtype=int)

    # First index: maximum magnitude
    indices[0] = np.argmax(np.abs(W[:, 0]))

    # Subsequent indices
    for j in range(1, max_r):
        W_selected = W[indices[:j], :j]

        # Solve with conjugate transpose
        c = np.linalg.solve(W_selected, W[indices[:j], j])

        # Residual
        r = W[:, j] - W[:, :j] @ c

        # Maximum magnitude of residual
        indices[j] = np.argmax(np.abs(r))

    if include_inds is not None:
        indices = list(set(*indices, *include_inds))

    return indices

def deim_inds(W: 'np.ndarray', max_r: int = None, include_inds=None, verbose=False):
    """Q-DEIM: QR-based DEIM for better numerical stability."""
    max_r = min(max_r, W.shape[1]) if max_r is not None else W.shape[1]

    add_rand_noise = False
    if add_rand_noise:
        if verbose:
            print('adding random noise')
        noise = np.random.random(W.shape) * 1.0e-09
        W = W + noise

    # QR decomposition of U^T
    Q, R, P = linalg.qr(W.T, pivoting=True)

    # First k pivots are the selected indices
    indices = P[:max_r]
    if include_inds is not None:
        indices = list(set(*indices, *include_inds))

    return indices


def regularized_solve(A: np.ndarray, R: np.ndarray, use_new_method=False, regularization=0.0):
    """ A = XR decomposition
    """

    lambda_I = regularization * np.eye(R.shape[0], dtype=R.dtype)
    if use_new_method:
        # Compute regularized inverse
        R_RH = R @ R.conj().T
        inv2 = np.linalg.inv(R_RH + lambda_I)
        X = A @ R.conj().T @ inv2

    else:
        X = A @ np.linalg.pinv(R)

    return X

def deim_split(A: np.ndarray, return_inds=False, max_bond=None, expand_u=False):
    """
    decompose matrix using DEIM; obtain low-rank approximation
    A is 2D matrix: (alpha_{i-1} * d_{i}) x alpha_{i}
    """

    size_l, size_r = A.shape
    max_bond = size_l if max_bond is None else min(size_l, max_bond)

    u, s, vt = np.linalg.svd(A, full_matrices=(A.shape[0] > A.shape[1]) and expand_u)  # full_matrices=False)
    # print('svd', u.shape, len(s), vt.shape)
    Q = u
    R_dmp = (vt.T * s).T

    # print('shapes', A.shape, u.shape, R_dmp.shape, max_bond)

    # ## add noise to expand rank
    if expand_u:

        if max_bond > size_r:
            add_size = max(min(2, max_bond - size_r), 0)

            u = u[:, :len(s) + add_size]
            R_dmp = np.vstack([R_dmp, np.zeros((add_size, R_dmp.shape[-1]))])
            Q = u

    inds = deim_inds(u)
    submat = u[inds, :]  # permute rows of A

    use_new_method = False
    if use_new_method:

        B = regularized_solve(Q, submat, use_new_method=True, regularization=0.1)
        T1 = B
        T2 = submat @ R_dmp

        # R = submat @ R_dmp
        # X = regularized_solve(A, R, use_new_method=True)
        # T1, T2 = X, R

    else:
        B = regularized_solve(Q, submat, use_new_method=False)
        T1 = B
        T2 = submat @ R_dmp

        # B = Q @ np.linalg.pinv(submat)       # Q Q^-1 = B = [I, Z]
        # # print('B', B)
        # ## note: np.linalg.inv(submat) gives bad results if n > r
        #
        # ## A = A[:,cols] (A_)^-1 A[rows,:] = C (A_)^-1 R
        # ## left canon form: want T1 = C (A_)^-1
        # ## C is essentially the tensor core
        # ## Let C = QT -->
        # ## --> C A_^-1 = Q T T^-1 Q_^-1 = Q Q_^-1
        #
        # # print('deim split', A.shape)
        # # print('B', B.shape, 'submat', submat.shape)
        # # print('R_dmp', R_dmp.shape)
        #
        # T1 = B
        # # submat = Q[inds,:]
        # # print('submat', submat)
        # T2 = (submat @ R_dmp)
        # # print('T2 norm', np.linalg.norm(T2))
        #
        # # for i in range(T1.shape[0]):
        # #     # print(np.abs(T1[i,:] - 1) < 1.0e-10)
        # #     j = np.argmax(np.abs(T1[i,:] - 1) < 1.0e-10)
        # #     print('i,j = 1', i, j, T1[i,j] - 1 )
        #
        # # print('final B', B[p,:], np.max(B))
        # # print('final B', B[inds, :], np.max(B))
        # # print('final B', B[np.sort(inds), :], np.max(B))
        # # submat = Q[inds, :]
        # # print('final Q Q^-1', (Q @ np.linalg.inv(submat))[inds,:])
        #
        # # print('err B', np.linalg.norm(B - (Q @ np.linalg.inv(submat))))
        # # print('err A', np.linalg.norm(A[:, :] - T1 @ T2))

    if return_inds:
        return T1, T2, inds  # np.sort(inds)
    else:
        return T1, T2


def approx_svd(A: np.ndarray, max_bond=None, cutoff=CUTOFF, full_matrices=False, verbose=False):
    ### full_matrices --> really only want full u
    u, s, vt = np.linalg.svd(A, full_matrices=full_matrices and A.shape[1] <= A.shape[0])
    # u0, s0, vt0 = u.copy(), s.copy(), vt.copy()
    if max_bond is not None:
        u = u[:, :max_bond]
        s = s[:max_bond]
        vt = vt[:max_bond, :]
        # print('deim max bond', max_bond, 'err', np.linalg.norm(s0[max_bond:])/np.linalg.norm(s0))

    # print('mpx approx svd', len(s), max_bond, cutoff)

    if cutoff is not None and not full_matrices:
        # cum_sum = np.cumsum((s[::-1])**2/s[0]**2)   # ordered smallest to largest
        cum_sum = np.cumsum((s[::-1]) ** 2)  # ordered smallest to largest
        cum_sum = cum_sum / cum_sum[-1]
        # print('cum sum', cum_sum, (cum_sum < cutoff)[:10])
        cut_ind = np.argmin(cum_sum < cutoff)
        if MINBOND is not None:     ## ensure a minimum bond dimension
            min_ind = max(len(s) - MINBOND, 0)
            # print('cut_ind/min_ind', cut_ind, min_ind)
            cut_ind = min(cut_ind, min_ind)     ## cut fewer elements from the end
        # print('cut ind', cut_ind)
        if cut_ind != 0:
            if verbose:
                print('deim cutoff', cutoff, 'cut ind', -cut_ind, 'err', np.linalg.norm(s[-cut_ind:]) / np.linalg.norm(s))
            u = u[:, :-cut_ind]
            s = s[:-cut_ind]
            vt = vt[:-cut_ind, :]
    return u, s, vt

def deim_compress(A: np.ndarray, max_bond, cutoff=CUTOFF, return_inds=False,
                  include_inds_r: Sequence[int]=None, expand_u=False, verbose=False):
    """
    decompose matrix using DEIM; obtain low-rank approximation
    truncate via singular values, and then perform DEIM?
    or, just perform DEIM up to desired rank?
    i don't think it really matters
    """

    # print('deim compression', A.shape, max_bond, cutoff)

    n, r = A.shape  ## A is 2D matrix:  e.g 2 site tensor, (alpha_{i-1} * d_{i}) x alpha_{i+1} * d{i+1}

    if (max_bond is None or n < max_bond or r < max_bond) and cutoff is None:
        T1, T2, inds_r = deim_split(A, return_inds=True, max_bond=max_bond, expand_u=expand_u)
        if return_inds:
            inds_c = list(range(r))
            return T1, T2, inds_r, inds_c
        else:
            return T1, T2

    u, s, vt = approx_svd(A, max_bond, cutoff, full_matrices=expand_u)
    inds_r = deim_inds(u, max_bond)
    inds_c = []

    if include_inds_r is not None:
        inds_r = list(set(inds_r).union(set(include_inds_r)))

    submat_u = u[inds_r, :]
    R_dmp = (vt.T * s).T
    if expand_u:
        add_size = submat_u.shape[1] - R_dmp.shape[0]
        R_dmp = np.vstack([R_dmp, np.zeros((add_size, R_dmp.shape[1]))])
        if verbose:
            print('add size', add_size)

    use_new_method = False
    if use_new_method:
        B = regularized_solve(u, submat_u, use_new_method=True, regularization=0.1)
        T1 = B
        T2 = u[inds_r, :] @ R_dmp

        # R = submat @ R_dmp
        # X = regularized_solve(A, R, use_new_method=True)
        # T1, T2 = X, R

    else:
        B = regularized_solve(u, submat_u, use_new_method=False)
        T1 = B
        T2 = submat_u @ R_dmp

        # ## old version version: construct T1, T2 from low-rank tensor
        # submat_u = u[inds_r, :]
        # T1 = u @ np.linalg.pinv(submat_u)
        #
        # R_dmp = (vt.T * s).T
        # T2 = submat_u @ R_dmp

        if False:  # len(inds_r) > r:
            if verbose:
                print('len inds', len(inds_r), r)
                print('T1', [(submat_u @ np.linalg.pinv(submat_u))[i, i] for i in range(len(inds_r))], np.linalg.cond(submat_u))
                # print('T1', submat_u @ np.linalg.pinv(submat_u))
                # print('canon error', np.linalg.norm(T1[inds_r, :] @ T2 - submat_u @ R_dmp))
                print('canon error', np.linalg.norm(A[inds_r, :] - submat_u @ R_dmp))

            R0_dmp = R_dmp.copy()
            R0_dmp[:, A.shape[1]//2:] *= 0.5
            T2_0 = submat_u @ R0_dmp
            A_0 = A.copy()
            A_0[:, A.shape[1]//2:] *= 0.5
            if verbose:
                print('canon error', np.linalg.norm(A_0[inds_r, :] - T2_0))

            # pdb.set_trace()

    if return_inds:
        return T1, T2, inds_r, inds_c  # np.sort(inds)
    else:
        return T1, T2




def tensor_compress_multiple_deim(tensors: Sequence[Sequence[qtn.Tensor]], # tens1: qtn.Tensor, tens2: qtn.Tensor,
                                 # select_inds: Sequence[int],  # inds_r: Sequence[int], inds_c: Sequence[int],
                                 phys_inds: list[str], right_inds: list[str], max_bond: int = None,
                                 do_qr=True):
    """ find inds_r, inds_c for CUR decomposition that works for all tensors in the list
    """
    old_right_inds = [x for x in right_inds]

    combined_tensors = []
    target_shape = []
    for targets in tensors:
        right_inds = old_right_inds
        if len(right_inds) == 1:
            target_shape += [targets[0].ind_size(right_inds[0])]
        else:
            target_shape += [[targets[0].ind_size(ri) for ri in right_inds]]
            for t in targets:
                t.fuse({'ir': right_inds}, inplace=True)
            right_inds = ['ir']

        tens = combine_targets(targets, phys_inds, right_inds)
        combined_tensors += [tens]

        ## target_shape is later used to select the portion of the combined tensor corresponding to each target.
        ## if targets contains more than one tensor, at the end, only the first target is retained
        ## we want to change this by instead padding the next tensor with 0s (so all interpolating fcts are retained)?
        ## but maybe it's lost when building the next effective site anyways...
        ## i believe in AMEn it's removed

    tens = next(iter(combined_tensors))
    left_inds = [ind for ind in tens.inds if (ind not in phys_inds + right_inds)]
    assert (len(left_inds) <= 1), 'not specific enough if have multiple virtual "left inds"'


    # print('TENSOR COMPRESS EXPAND')
    T1, T2, inds_r, inds_c = tensor_compress_expand(combined_tensors, phys_inds, right_inds, max_bond=max_bond)
    # print('inds r', inds_r, len(inds_r), np.max(inds_r), )
    # print('T1', T1.data)
    ## T1: [I Z]
    ## T2: "R"

    #### select parts of T2 to match with original tensors (which were originally appended together)
    # subT2 = T2.copy()
    right_ind = right_inds[0]
    r_pos = T2.inds.index(right_ind)

    cur_tensors = []
    ind1 = 0
    for tens, r_size in zip(combined_tensors, target_shape):

        subT2: 'qtn.Tensor' = T2.copy()
        ind2 = ind1 + np.prod([tens.ind_size(ri) for ri in right_inds])
        ## assumes right_inds is of length 1
        ## separate different contributions from combined_tensors
        ## only keep columns from main_tens of each list of targets
        subT2.modify(apply = lambda x: np.take(x, list(range(ind1, ind1 + np.prod(r_size))), axis=r_pos))
        ind1 = ind2

        if len(old_right_inds) > 1:
            subT2 = subT2.unfuse({right_ind: old_right_inds},{right_ind: r_size})

        cur_tensors += [(T1.copy(), subT2)]

    # print('inds', inds_r, inds_c)
    # print('right inds', right_inds)
    # print('cur tensors', cur_tensors)

    return inds_r, inds_c, cur_tensors



########################

def tensor_compress_expand(targets: Sequence[qtn.Tensor], phys_inds: list[str], right_inds: list[str],
                           max_bond: int = None, cutoff: float = None, solver_type=DEFAULT_SOLVER,
                           only_keep_first=True):
    """ expand basis using all targets
        if len(targets) == 1, equivalent to tensor_compress
    """
    main_tens = targets[0]
    left_inds = [ind for ind in main_tens.inds if (ind not in phys_inds + right_inds)]
    assert (len(left_inds) <= 1), 'not specific enough if have multiple virtual "left inds"'

    ## must follow specific order for fusing unshared bonds
    fuse_inds = phys_inds + left_inds
    r_size = [main_tens.ind_size(ri) for ri in right_inds]

    version = 'oversample'

    ## new method  v2
    if version == 'oversample':
        ## first tens
        inds_r = None
        if len(targets) > 1:
            inds_r_set = set()
            for tens in targets:
                T1, T2, inds_r, inds_c = tensor_compress(tens, phys_inds, right_inds, max_bond=max_bond, cutoff=cutoff,
                                                         return_inds=True, solver_type=solver_type)
                inds_r_set.update(set(inds_r))

            inds_r = list(inds_r_set)

        ## expand tens
        out = main_tens
        for tens in targets[1:]:
            out = qtn.tensor_direct_product(out, tens, sum_inds=fuse_inds)

        T1, T2, inds_r1, inds_c = tensor_compress(out, phys_inds, right_inds, max_bond=max_bond, cutoff=cutoff,
                                                  return_inds=True, solver_type=solver_type,
                                                  include_inds_r=inds_r, expand_u=False)  # expand_u=(inds_r is not None))
        inds_r = inds_r1



        # #### new method  ## leads to oversampling but it helps
        # ## first tens
        # T1, T2, inds_r, inds_c = tensor_compress(main_tens, phys_inds, right_inds, max_bond=max_bond, cutoff=cutoff,
        #                                          return_inds=True, solver_type=solver_type, verbose=True)
        #
        # ## expand tens
        # if len(targets) > 1:
        #     out = main_tens
        #     for tens in targets[1:]:
        #         out = qtn.tensor_direct_product(out, tens, sum_inds=fuse_inds)
        #
        #     T1, T2, inds_r1, inds_c = tensor_compress(out, phys_inds, right_inds, max_bond=max_bond, cutoff=cutoff,
        #                                               return_inds=True, solver_type=solver_type,
        #                                               include_inds_r=inds_r)
        #     # # pdb.set_trace()
        #     # # print('len inds', len(inds_r), len(inds_r1))
        #     # # print('len(inds)', set(inds_r1).difference(set(inds_r)))
        #     # T1, T2 = tensor_xr(main_tens, phys_inds, right_inds, inds_r1 )
        #     inds_r = inds_r1

    elif version == 'random':
        #### old method
        ## expand tens
        out = main_tens
        for tens in targets[1:]:
            out = qtn.tensor_direct_product(out, tens, sum_inds=fuse_inds)

        left_inds = [ind for ind in out.inds if ind not in phys_inds + right_inds]
        l_ind = left_inds[0] if len(left_inds) > 0 else None
        r_ind = right_inds[0] if len(right_inds) > 0 else None


        inds_r = tensor_select_rows(out, l_ind, r_ind, phys_inds[0], oversample=True)
        T1, T2 = tensor_xr(out, phys_inds, right_inds, inds_r)

        # T1, T2, inds_r, inds_c = tensor_compress(out, phys_inds, right_inds, max_bond=max_bond, cutoff=cutoff,
        #                                          return_inds=True, solver_type=solver_type)

        # from local_solvers.helper_mixed import tensor_get_submat
        # left_inds = [ind for ind in out.inds if ind not in phys_inds + right_inds]
        # l_ind = left_inds[0] if len(left_inds) > 0 else None
        # r_ind = right_inds[0] if len(right_inds) > 0 else None
        # inds_r, TC, TU, TR = tensor_get_submat(out, l_ind, r_ind, phys_inds[0], oversample=True)
        # T1 = TC @ TU
        # T2 = TR
        inds_c = []

    else:

        #### old method
        ## expand tens
        out = main_tens
        for tens in targets[1:]:
            out = qtn.tensor_direct_product(out, tens, sum_inds=fuse_inds)

        T1, T2, inds_r, inds_c = tensor_compress(out, phys_inds, right_inds, max_bond=max_bond, cutoff=cutoff,
                                                 return_inds=True, solver_type=solver_type)


    ## only keep columns from main_tens (assumes right_inds is of length 1)
    if only_keep_first:
        for ix, ir in enumerate(right_inds):
            ind_pos = T2.inds.index(ir)
            # print('T2', T2, right_inds, ind_pos, r_size[ix])
            T2.modify(apply=lambda x: np.take(x, list(range(r_size[ix])), axis=ind_pos))
        # print('new T2', T2)

    return T1, T2, inds_r, inds_c


def combine_targets(targets: Sequence[qtn.Tensor], phys_inds: list[str], right_inds: list[str],):

    main_tens = targets[0]
    left_inds = [ind for ind in main_tens.inds if (ind not in phys_inds + right_inds)]
    assert (len(left_inds) <= 1), 'not specific enough if have multiple virtual "left inds"'
    fuse_inds = phys_inds + left_inds

    ## expand tens
    out = main_tens
    for tens in targets[1:]:
        out = qtn.tensor_direct_product(out, tens, sum_inds=fuse_inds)

    # print('combined target out', out.shape)

    return out


def tensor_compress_multiple(tensors: Sequence[Sequence[qtn.Tensor]], # tens1: qtn.Tensor, tens2: qtn.Tensor,
                             # select_inds: Sequence[int],  # inds_r: Sequence[int], inds_c: Sequence[int],
                             phys_inds: list[str], right_inds: list[str], max_bond: int = None, cutoff: float = None,
                             solver_type = DEFAULT_SOLVER):
    """ find inds_r, inds_c for CUR decomposition that works for all tensors in the list
        do basis expansion (without compression of tensors[0])
    """
    # tens_list = []
    # for tens in tensors:
    #     tens = tens.transpose(*fuse_inds, *right_inds, inplace=True)
    #     tens_ = tens.fuse({f'xx': fuse_inds, f'oo': right_inds})
    #     tens_list += [tens_.data]

    old_right_inds = [x for x in right_inds]

    combined_tensors = []
    target_shape = []
    for targets in tensors:
        right_inds = old_right_inds
        if len(right_inds) == 1:
            target_shape += [targets[0].ind_size(right_inds[0])]
        else:
            target_shape += [[targets[0].ind_size(ri) for ri in right_inds]]
            for t in targets:
                t.fuse({'ir': right_inds}, inplace=True)
            right_inds = ['ir']

        tens = combine_targets(targets, phys_inds, right_inds)
        combined_tensors += [tens]

        ## target_shape is later used to select the portion of the combined tensor corresponding to each target.
        ## if targets contains more than one tensor, at the end, only the first target is retained
        ## we want to change this by instead padding the next tensor with 0s (so all interpolating fcts are retained)?
        ## but maybe it's lost when building the next effective site anyways...
        ## i believe in AMEn it's removed

    tens = next(iter(combined_tensors))
    left_inds = [ind for ind in tens.inds if (ind not in phys_inds + right_inds)]
    assert (len(left_inds) <= 1), 'not specific enough if have multiple virtual "left inds"'

    # print('TENSOR COMPRESS EXPAND')
    T1, T2, inds_r, inds_c = tensor_compress_expand(combined_tensors, phys_inds, right_inds,
                                                    max_bond=max_bond, cutoff=cutoff,
                                                    solver_type=solver_type)
    ## T1: [I Z]
    ## T2: "R"

    #### select parts of T2 to match with original tensors (which were originally appended together)
    # subT2 = T2.copy()
    right_ind = right_inds[0]
    r_pos = T2.inds.index(right_ind)

    cur_tensors = []
    ind1 = 0
    for tens, r_size in zip(combined_tensors, target_shape):

        subT2: 'qtn.Tensor' = T2.copy()
        ind2 = ind1 + np.prod([tens.ind_size(ri) for ri in right_inds])
        ## assumes right_inds is of length 1
        ## separate different contributions from combined_tensors
        ## only keep columns from main_tens of each list of targets
        subT2.modify(apply = lambda x: np.take(x, list(range(ind1, ind1 + np.prod(r_size))), axis=r_pos))
        ind1 = ind2

        if len(old_right_inds) > 1:
            subT2 = subT2.unfuse({right_ind: old_right_inds},{right_ind: r_size})

        cur_tensors += [(T1.copy(), subT2)]

    # print('inds', inds_r, inds_c)
    # print('right inds', right_inds)
    # print('cur tensors', cur_tensors)

    return inds_r, inds_c, cur_tensors


def tensor_compress_multiple_seqs(tensors: Sequence[Sequence[qtn.Tensor]], # tens1: qtn.Tensor, tens2: qtn.Tensor,
                                  # select_inds: Sequence[int],  # inds_r: Sequence[int], inds_c: Sequence[int],
                                  phys_inds: list[str], right_inds: list[str], max_bond: int = None, cutoff: float = None,
                                  solver_type = DEFAULT_SOLVER):
    """ find inds_r, inds_c for CUR decomposition that works for all tensors in the list
        do basis expansion (without compression of tensors[0])
    """
    # tens_list = []
    # for tens in tensors:
    #     tens = tens.transpose(*fuse_inds, *right_inds, inplace=True)
    #     tens_ = tens.fuse({f'xx': fuse_inds, f'oo': right_inds})
    #     tens_list += [tens_.data]

    old_right_inds = [x for x in right_inds]

    combined_tensors = []
    target_shape = []
    for targets in tensors:
        right_inds = old_right_inds
        if len(right_inds) == 1:
            target_shape += [targets[0].ind_size(right_inds[0])]
        else:
            target_shape += [[targets[0].ind_size(ri) for ri in right_inds]]
            for t in targets:
                t.fuse({'ir': right_inds}, inplace=True)
            right_inds = ['ir']

        tens = combine_targets(targets, phys_inds, right_inds)
        combined_tensors += [tens]

        ## target_shape is later used to select the portion of the combined tensor corresponding to each target.
        ## if targets contains more than one tensor, at the end, only the first target is retained
        ## we want to change this by instead padding the next tensor with 0s (so all interpolating fcts are retained)?
        ## but maybe it's lost when building the next effective site anyways...
        ## i believe in AMEn it's removed

    tens = next(iter(combined_tensors))
    left_inds = [ind for ind in tens.inds if (ind not in phys_inds + right_inds)]
    assert (len(left_inds) <= 1), 'not specific enough if have multiple virtual "left inds"'

    # print('TENSOR COMPRESS EXPAND')
    T1, T2, inds_r, inds_c = tensor_compress_expand(combined_tensors, phys_inds, right_inds,
                                                    max_bond=max_bond, cutoff=cutoff,
                                                    solver_type=solver_type)
    ## T1: [I Z]
    ## T2: "R"

    #### select parts of T2 to match with original tensors (which were originally appended together)
    # subT2 = T2.copy()
    right_ind = right_inds[0]
    r_pos = T2.inds.index(right_ind)

    cur_tensors = []
    ind1 = 0
    for tens, r_size in zip(combined_tensors, target_shape):

        subT2: 'qtn.Tensor' = T2.copy()
        ind2 = ind1 + np.prod([tens.ind_size(ri) for ri in right_inds])
        ## assumes right_inds is of length 1
        ## separate different contributions from combined_tensors
        ## only keep columns from main_tens of each list of targets
        subT2.modify(apply = lambda x: np.take(x, list(range(ind1, ind1 + np.prod(r_size))), axis=r_pos))
        ind1 = ind2

        if len(old_right_inds) > 1:
            subT2 = subT2.unfuse({right_ind: old_right_inds},{right_ind: r_size})

        cur_tensors += [(T1.copy(), subT2)]

    # print('inds', inds_r, inds_c)
    # print('right inds', right_inds)
    # print('cur tensors', cur_tensors)

    return inds_r, inds_c, cur_tensors

