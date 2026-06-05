"""Helper functions for quimb ``Tensor`` and tensor-network objects.

The utility backbone of the codebase: low-level tensor arithmetic (add / scale /
elementwise multiply), orthogonality checks and canonicalization, singular values and
entanglement entropy, and other operations on quimb MPS / MPO / tensor-network objects
that the grid, field, and solver modules build on.
"""

# from setup_.configs import *
# import numpy as np
import scipy.sparse
import quimb

from setup_.defaults import *
import time
import quimb.tensor as qtn
import local_solvers.helper_tn as helper_tn

""" helper functions for working with quimb Tensor or Tensor Network objects
"""




def get_cut_ind(svals, cutoff=CUTOFF, max_bond=MAXBOND, min_bond=MINBOND, is_squared=False):
    """Determine the number of singular values to keep under a truncation policy.

    Parameters
    ----------
    svals : numpy.ndarray
        Singular values (or eigenvalues if ``is_squared`` is True), in
        descending order.
    cutoff : float
        SVD truncation threshold; the smallest values whose cumulative squared
        weight (relative to the total) falls below this are discarded
        (default module const CUTOFF).
    max_bond : int or None
        Maximum bond dimension; the kept count is capped at this value. A value
        ``<= 0`` is treated as None (no cap). Default MAXBOND.
    min_bond : int or None
        Minimum bond dimension to retain regardless of ``cutoff``. Default MINBOND.
    is_squared : bool
        If True, ``svals`` are already eigenvalues (squared singular values);
        otherwise they are squared internally.

    Returns
    -------
    int
        The number of leading singular values to keep.
    """
    if max_bond is not None and max_bond <= 0:
        max_bond = None

    tot_num = len(svals)
    cut_ind = tot_num

    eigval = svals ** 2 if not is_squared else svals

    if cutoff is not None:
        ev_max = np.max(np.abs(eigval))
        cum_sum = np.cumsum(np.abs(eigval[::-1]))  # --> smallest (sum from end) to largest (including largest eigval)
        cut_ind = np.argmin(cum_sum / cum_sum[-1] < cutoff)

        if min_bond is not None:  ## ensure a minimum bond dimension
            min_ind = max(len(eigval) - min_bond, 0)
            cut_ind = min(cut_ind, min_ind)  ## cut fewer elements from the end

        cut_ind = tot_num - cut_ind

    if max_bond is not None:
        cut_ind = min(cut_ind, max_bond)

    return cut_ind

def tensor_svd(tens: 'qtn.Tensor', left_inds: Sequence[str], absorb: Literal['left','right'],
                 max_bond=MAXBOND, min_bond=MINBOND, cutoff=CUTOFF, bond_ind: str=None):
    """Split a tensor by SVD and truncate, absorbing singular values to one side.

    Parameters
    ----------
    tens : qtn.Tensor
        The tensor to split.
    left_inds : Sequence[str]
        Index labels forming the left (U) factor; the remaining indices form
        the right (V) factor.
    absorb : {'left', 'right'}
        Which SVD factor receives the singular values.
    max_bond : int or None
        Maximum bond dimension to keep. Default MAXBOND.
    min_bond : int or None
        Minimum bond dimension to keep. Default MINBOND.
    cutoff : float
        SVD truncation threshold. Default module const CUTOFF.
    bond_ind : str, optional
        Label for the new bond index between the factors; a temporary name is
        used if None.

    Returns
    -------
    tuple of qtn.Tensor
        The truncated ``(u, vt)`` factors.
    """
    if bond_ind is None:
        bond_ind = '__tmp__'

    u, s, vt = qtn.tensor_split(tens, left_inds, bond_ind=bond_ind, absorb=None, get='tensors',
                                cutoff=1.0e-40, cutoff_mode=CUTOFF_MODE)

    svals = s.data
    cut_ind = get_cut_ind(svals, cutoff=cutoff, max_bond=max_bond, min_bond=min_bond)
    bond_idx = u.inds.index(bond_ind)
    u_data = np.moveaxis(u.data, bond_idx, 0)
    u_data = u_data[:cut_ind]
    u_data = np.moveaxis(u_data, 0, bond_idx)
    u.modify(data=u_data)

    bond_idx = vt.inds.index(bond_ind)
    vt_data = np.moveaxis(vt.data, bond_idx, 0)
    vt_data = vt_data[:cut_ind]
    vt_data = np.moveaxis(vt_data, 0, bond_idx)
    vt.modify(data=vt_data)

    if absorb == 'left':
        ix = u.inds.index(bond_ind)
        u_data = np.moveaxis(u.data, ix, -1)
        u_data = u_data * svals[:cut_ind]
        u_data = np.moveaxis(u_data, -1, ix)
        u.modify(data=u_data)

    elif absorb == 'right':
        ix = vt.inds.index(bond_ind)
        vt_data = np.moveaxis(vt.data, ix, -1)
        vt_data = vt_data * svals[:cut_ind]
        vt_data = np.moveaxis(vt_data, -1, ix)
        vt.modify(data=vt_data)
    else:
        raise ValueError

    return u, vt


def add_tensors(tens1: 'qtn.Tensor', tens2: 'qtn.Tensor', inplace=False):
    """Add two tensors elementwise after aligning their index order.

    Parameters
    ----------
    tens1 : qtn.Tensor
        First tensor; its index order defines the output layout.
    tens2 : qtn.Tensor
        Second tensor; transposed to match ``tens1`` before adding.
    inplace : bool
        Modify ``tens1`` in place if True, else operate on a copy.

    Returns
    -------
    qtn.Tensor
        The elementwise sum ``tens1 + tens2``.
    """
    new_tens = tens1 if inplace else tens1.copy()
    tens2 = tens2.transpose_like(tens1)
    new_tens.modify(apply=lambda data: tens2.data + data)
    return new_tens


def scale_tensors(tens, val, inplace=False):
    """Multiply a tensor's data by a scalar.

    Parameters
    ----------
    tens : qtn.Tensor
        The tensor to scale.
    val : Numeric
        Scalar multiplier applied to the tensor data.
    inplace : bool
        Modify ``tens`` in place if True, else operate on a copy.

    Returns
    -------
    qtn.Tensor
        The scaled tensor.
    """
    new_tens = tens if inplace else tens.copy()
    new_tens.modify(apply=lambda data: data * val)
    return new_tens


def elem_mult_tensors(tens1: 'qtn.Tensor', tens2: 'qtn.Tensor', inplace=False):
    """Multiply two tensors elementwise after aligning their index order.

    Parameters
    ----------
    tens1 : qtn.Tensor
        First tensor; its index order defines the output layout.
    tens2 : qtn.Tensor
        Second tensor; transposed to match ``tens1`` before multiplying.
    inplace : bool
        Modify ``tens1`` in place if True, else operate on a copy.

    Returns
    -------
    qtn.Tensor
        The elementwise (Hadamard) product ``tens1 * tens2``.
    """
    new_tens = tens1 if inplace else tens1.copy()
    tens2 = tens2.transpose_like(tens1)
    new_tens.modify(apply=lambda data: tens2.data * data)
    return new_tens


def diag_mult(tens: 'qtn.Tensor', diag_vec: 'qtn.Tensor', inplace=False):
    """Multiply a tensor along an index by a diagonal given as a vector tensor.

    The indices of ``diag_vec`` are fused (when multi-dimensional) and matched
    against the corresponding fused indices of ``tens``, applying the diagonal
    along that index; the fused indices are then unfused again.

    Parameters
    ----------
    tens : qtn.Tensor
        The tensor to scale along the shared index/indices.
    diag_vec : qtn.Tensor
        Vector tensor whose entries form the diagonal; if it has more than one
        index, its indices are fused to a single index.
    inplace : bool
        Modify ``tens`` in place if True, else operate on a copy.
    verbose : bool
        Print ``tens`` and ``diag_vec`` for diagnostics if True.

    Returns
    -------
    qtn.Tensor
        The tensor with the diagonal applied.
    """
    new_tens = tens if inplace else tens.copy()
    num_diag = diag_vec.ndim
    if num_diag > 1:
        inds_dict = {'x_ind': diag_vec.inds}
        inds_shape = {'x_ind': diag_vec.shape}
        new_tens.fuse(inds_dict, inplace=True)
        diag_vec = diag_vec.fuse(inds_dict)

    diag_ind = diag_vec.inds[0]
    new_tens.multiply_index_diagonal(diag_ind, diag_vec.data, inplace=True)

    # iso_inds = [ind for ind in tens.inds if ind != diag_ind] + [diag_ind]
    # print('iso inds', iso_inds)
    # print('new tens', new_tens)
    # new_tens.transpose(*iso_inds, inplace=True)
    # new_tens.modify(data=diag_vec.data * new_tens.data)

    if num_diag > 1:
        new_tens.unfuse(inds_dict, inds_shape, inplace=True)
    return new_tens


def tensor_transpose_inds(tens: 'qtn.Tensor', swap_inds: dict[str, str], inplace=False):
    """Swap pairs of index labels on a tensor.

    Parameters
    ----------
    tens : qtn.Tensor
        The tensor whose index labels are swapped.
    swap_inds : dict[str, str]
        Mapping of index labels to swap; each ``key -> value`` pair causes the
        two labels to be exchanged.
    inplace : bool
        Modify ``tens`` in place if True, else operate on a copy.

    Returns
    -------
    qtn.Tensor
        The tensor with the index labels swapped.
    """
    tens = tens if inplace else tens.copy()
    tens.reindex({**{sw: st for sw, st in swap_inds.items()},
                  **{st: sw for sw, st in swap_inds.items()}}, inplace=True)

    return tens


def tensor_direct_product(tens1: qtn.Tensor, tens2: qtn.Tensor, sum_inds=None, inplace=False,
                          force_match=False, auto_transpose=False):
    """Form the direct (block-diagonal) product of two tensors.

    Parameters
    ----------
    tens1 : qtn.Tensor
        First tensor; the result is written into it.
    tens2 : qtn.Tensor
        Second tensor.
    sum_inds : sequence of str, optional
        Indices that are summed/stacked rather than block-concatenated.
    inplace : bool
        Modify ``tens1`` in place if True, else operate on a copy (only
        relevant when ``auto_transpose`` is True).
    force_match : bool
        If True, zero-pad each tensor along mismatched index sizes so the two
        tensors share the same shape before taking the direct product.
    auto_transpose : bool
        If True, transpose ``tens2`` to match ``tens1``'s index order first.

    Returns
    -------
    qtn.Tensor
        The direct product, stored in ``tens1``.
    """
    if auto_transpose:
        tens1 = tens1 if inplace else tens1.copy()
        tens2 = tens2.transpose_like(tens1, inplace=False)

    if force_match:
        ind_sizes_1 = tens1.shape
        ind_sizes_2 = tens2.shape
        expand_1 = [(0, max(0, s2 - s1)) for (s1, s2) in zip(ind_sizes_1, ind_sizes_2)]
        expand_2 = [(0, max(0, s1 - s2)) for (s1, s2) in zip(ind_sizes_1, ind_sizes_2)]
        tens1.modify(apply=lambda x: np.pad(x, expand_1, constant_values=(0, 0)))
        tens2.modify(apply=lambda x: np.pad(x, expand_2, constant_values=(0, 0)))

    tens1.direct_product(tens2, sum_inds=sum_inds, inplace=True)
    return tens1


def tensor_adjust_shape(tens: qtn.Tensor, ind_sizes: dict):
    """Pad or trim a tensor along given indices to the requested sizes (in place).

    Parameters
    ----------
    tens : qtn.Tensor
        The tensor to reshape; modified in place.
    ind_sizes : dict
        Mapping ``ind -> desired size``; each index is zero-padded if larger
        than the current size or trimmed if smaller.

    Returns
    -------
    qtn.Tensor
        The resized tensor (same object as ``tens``).
    """
    tens_data = tens.data.copy()
    for ind, size in ind_sizes.items():
        ind_pos = tens.inds.index(ind)
        tens_data = np.moveaxis(tens_data, ind_pos, 0)
        if size > tens_data.shape[0]:  ## pad
            pad_size = size - tens_data.shape[0]
            tens_data = np.pad(tens_data, ((0, pad_size),) + ((0, 0),) * (tens_data.ndim - 1), constant_values=(0, 0))
        elif size < tens_data.shape[0]:  ## trim
            tens_data = tens_data[:size, ...]
        tens_data = np.moveaxis(tens_data, 0, ind_pos)
    tens.modify(data=tens_data)
    return tens


def zero_mps(L, max_bond=2, orthog=0):
    """Construct an all-zero MPS of length L.

    A random MPS is canonicalized around site ``orthog`` and that site is then
    zeroed, yielding an MPS that evaluates to zero while keeping a valid bond
    structure.

    Parameters
    ----------
    L : int
        Number of sites.
    max_bond : int
        Bond dimension of the underlying random state. Default 2.
    orthog : int
        Site about which to canonize and which is zeroed. Default 0.

    Returns
    -------
    MPS
        The zero MPS with ``exponent`` set to 0.0.
    """
    out = qtn.MPS_rand_state(L=L, bond_dim=max_bond)
    canonize(out, i=orthog)
    out[orthog].modify(apply=lambda x: x * 0)
    out.exponent = 0.0
    return out


def pad_mpx_virtuals(mpx: Union['MPSType, MPOType'], max_bond: int):
    """Zero-pad the virtual (bond) indices of an mpx up to a target bond size.

    For each interior bond smaller than ``max_bond``, both adjacent tensors are
    direct-product padded with zeros so the shared bond reaches ``max_bond``.
    Modifies ``mpx`` in place.

    Parameters
    ----------
    mpx : MPSType or MPOType
        The matrix product state/operator whose bonds are padded.
    max_bond : int
        Target bond dimension for every interior bond.

    Returns
    -------
    MPSType or MPOType
        The padded network (same object as ``mpx``).
    """
    for i in range(mpx.L - 1):
        bond_size = mpx.bond_size(i, i + 1)
        if bond_size < max_bond:
            shared_bond = mpx.bond(i, i + 1)

            # mpx[i].moveindex(shared_bond, -1, inplace=True)
            inds = [ix for ix in mpx[i].inds if ix != shared_bond] + [shared_bond]
            mpx[i].transpose(*inds, inplace=True)

            pad_shape = mpx[i].shape[:-1] + (max_bond - bond_size,)
            pad_zero = qtn.Tensor(np.zeros(pad_shape), inds=mpx[i].inds)
            qtn.tensor_direct_product(mpx[i], pad_zero, sum_inds=mpx[i].inds[:-1], inplace=True)

            # mpx[i + 1].moveindex(mpx.bond(i, i + 1), 0, inplace=True)
            inds = [shared_bond] + [ix for ix in mpx[i + 1].inds if ix != shared_bond]
            mpx[i + 1].transpose(*inds, inplace=True)
            pad_shape = (max_bond - bond_size,) + mpx[i + 1].shape[1:]
            pad_zero = qtn.Tensor(np.zeros(pad_shape), inds=mpx[i + 1].inds)
            qtn.tensor_direct_product(mpx[i + 1], pad_zero, sum_inds=mpx[i + 1].inds[1:], inplace=True)

    return mpx


def mpx_add_label(mpx, label, check_exist=True) -> qtn.TensorNetwork1D:
    """Append a label suffix to the site tag and index id formats of an mpx (in place).

    Parameters
    ----------
    mpx : qtn.TensorNetwork1D
        The MPS or MPO whose tag/index id format strings are extended.
    label : str
        Suffix appended to ``site_tag_id`` and to the physical index ids
        (``site_ind_id`` for an MPS; ``upper_ind_id``/``lower_ind_id`` for an MPO).
    check_exist : bool
        Guard controlling whether the suffix is applied based on whether the
        label is already present in the relevant id format.

    Returns
    -------
    qtn.TensorNetwork1D
        The relabeled network (same object as ``mpx``).
    """
    if not (check_exist and label in mpx.site_tag_id):
        mpx.site_tag_id = mpx.site_tag_id + label

    if isinstance(mpx, qtn.MatrixProductState):
        if not (check_exist and label not in mpx.site_ind_id):
            mpx.site_ind_id = mpx.site_ind_id + label
    elif isinstance(mpx, qtn.MatrixProductOperator):
        if not (check_exist and label not in mpx.upper_ind_id):
            mpx.upper_ind_id = mpx.upper_ind_id + label
            mpx.lower_ind_id = mpx.lower_ind_id + label

    return mpx


def collect_exponent(mpx: qtn.TensorNetwork, inplace=False, verbose=False):
    """Factor out the network norm into the ``exponent`` attribute.

    Divides the first tensor by the network norm and adds ``log10(norm)`` to
    ``mpx.exponent``, leaving a normalized network whose magnitude is carried as
    a base-10 exponent.

    Parameters
    ----------
    mpx : qtn.TensorNetwork
        The network whose norm is absorbed into the exponent.
    inplace : bool
        Modify ``mpx`` in place if True, else operate on a copy.
    verbose : bool
        Print norm diagnostics before and after if True.

    Returns
    -------
    qtn.TensorNetwork
        The normalized network with updated ``exponent``.
    """
    mpx = mpx if inplace else mpx.copy()
    mpx_norm = mpx.copy().norm()
    if verbose:
        print('collect exp mpx norm', mpx_norm, norm(mpx))
    mpx[0].modify(apply=lambda x: x * 1. / mpx_norm)
    # mpx.distribute_exponent()
    # scalar_multiply(mpx, 1./mpx_norm, inplace=True)
    mpx.exponent += np.log10(mpx_norm)
    if verbose:
        print('mpx normalized?', mpx.copy().norm(), norm(mpx))
    return mpx


def norm(mpx: qtn.TensorNetwork, verbose=False):
    """Return the Frobenius norm of an MPS/MPO including its exponent factor.

    Computed as ``sqrt(<x|x>)`` via :func:`ovlp` (which incorporates the
    base-10 ``exponent``). A length-1 network returns the single tensor's norm.
    Slightly negative overlaps within tolerance trigger a debugger trace.

    Parameters
    ----------
    mpx : qtn.TensorNetwork
        The MPS/MPO to measure; ``None`` returns ``0.0``.
    verbose : bool
        Print several norm computations for diagnostics if True.

    Returns
    -------
    float
        The norm of ``mpx``.
    """
    # return qtn.expec_TN_1D(mpx.H, mpx) * 10**(mpx.exponent)
    if mpx is None:
        return 0.0

    if mpx.L == 1:
        return mpx[0].norm()

    if verbose:
        # mpx.distribute_exponent()
        print('np norm', np.linalg.norm(to_dense(mpx)))
        print('mpx norm', mpx.norm() * 10 ** mpx.exponent)
        print('mpx norm', mpx.copy().norm() * 10 ** mpx.exponent)
        # mpx.distribute_exponent()
        # print('mpx norm', mpx.norm())
        # print('mpx norm', mpx.copy().norm())
        print('?', np.abs(mpx.norm(tags=all)) * 10 ** mpx.exponent, mpx.exponent)
        print('?1', np.abs(mpx.copy().norm()) * 10 ** mpx.exponent, mpx.exponent)
        print('?2', np.linalg.norm(to_dense(mpx)))
        print('ovlp?', ovlp(mpx, mpx.conj()))

    out = ovlp(mpx, mpx.conj())
    if out < -1 * np.sqrt(CUTOFF):
        print('<x|x> yields negative value', out)
        pdb.set_trace()
        # raise ValueError('<x|x> yields negative value', out)
    out = np.sqrt(out)

    # out = mpx.copy().norm()
    # # print('out', out, mpx.exponent)
    # if not np.isnan(out):
    #     out *= 10 ** mpx.exponent
    # else:
    #     out = ovlp(mpx, mpx.conj())
    #     if out < -1 * np.sqrt(CUTOFF):
    #         raise ValueError('<x|x> yields negative value')
    #     out = np.sqrt(out)

    # print('norm out', out)

    return out


def ovlp(mps1: Union['MPSType', 'MPOType'], mps2: Union['MPSType', 'MPOType']):
    """Contract two MPS/MPO networks to compute their overlap (inner product).

    Both inputs are copied and canonized about site 0, ``mps2`` is reindexed to
    share ``mps1``'s site/upper/lower index ids, and the combined network is
    contracted. The result is scaled by ``10 ** (mps1.exponent + mps2.exponent)``.
    A length-1 network special-cases a direct matrix/vector product.

    Parameters
    ----------
    mps1 : MPSType or MPOType
        First network (ket side).
    mps2 : MPSType or MPOType
        Second network (bra side); should already be conjugated by the caller
        when a true inner product is wanted.

    Returns
    -------
    Numeric
        The overlap value scaled by the combined exponent.
    """
    if mps1.L == 1:
        # if True: # mps1 == mps2 or mps1[0] == mps2[0]:
        #     print('is same')
        #     if scipy.sparse.issparse(mps1[0].data):
        #         # out = 0.0
        #         sp_data = mps1[0].data.tocoo()
        #         out = np.linalg.norm(sp_data.data) ** 2
        #         # print(sp_data.data)
        #         # for zz in mps1[0].data.data:
        #         #     print('??', zz)
        #         #     raise RuntimeError
        #         #     # out += np.abs(val)**2
        #         # out = np.sqrt(out)
        #     else:
        #         out = mps1[0].norm()
        #     return out * 10 ** (mps1.exponent + mps2.exponent)

        out = mps1[0].data @ mps2[0].data
        if isinstance(out, (float, complex)):
            return out * 10 ** (mps1.exponent + mps2.exponent)
        return np.sum(out.diagonal()) * 10 ** (mps1.exponent + mps2.exponent)

    mps1 = mps1.copy()
    mps2 = mps2.copy()
    canonize(mps1, i=0)
    canonize(mps2, i=0)

    tn = mps1.view_as(qtn.TensorNetwork)
    if mps2 is None:
        return 0.0
    # mps2.mangle_inner_()
    try:
        mps2 = mps2.reindex_sites(mps1.site_ind_id)
    except AttributeError:
        mps2 = mps2.reindex_upper_sites(mps1.upper_ind_id)
        mps2 = mps2.reindex_lower_sites(mps1.lower_ind_id)
    tn.add(mps2, check_collisions=True)

    # tn_copy = tn.copy()
    # print('ovlp', np.vdot(to_dense(mps1), to_dense(mps2)))
    # print('ovlp slwo', ovlp_slow(mps1, mps2))
    return tn.contract() * 10 ** (mps1.exponent + mps2.exponent)


def ovlp_slow(mps1: Union['MPSType', 'MPOType'], mps2: Union['MPSType', 'MPOType']):
    """Compute the overlap of two networks by an explicit site-by-site sweep.

    A reference/debug variant of :func:`ovlp`: mangles ``mps2``'s inner bonds,
    canonizes both about site 0, then contracts the environment one site at a
    time. The result is scaled by ``10 ** (mps1.exponent + mps2.exponent)``.

    Parameters
    ----------
    mps1 : MPSType or MPOType
        First network.
    mps2 : MPSType or MPOType
        Second network.

    Returns
    -------
    Numeric
        The overlap value scaled by the combined exponent.
    """
    mps2 = mps2.copy()
    mps2.mangle_inner_()
    canonize(mps1, i=0)
    canonize(mps2, i=0)

    env = mps1[0] @ mps2[0]
    for i in range(1, mps1.L):
        env = env @ mps1[i] @ mps2[i]

    env = env * 10 ** (mps1.exponent + mps2.exponent)

    return env


def distance(tn1, tn2, method='auto'):
    """Compute the Frobenius norm of the difference ``tn1 - tn2``.

    Both networks are copied and their exponents distributed across tensors
    before calling :func:`quimb.tensor.tensor_network_distance`.

    Parameters
    ----------
    tn1 : qtn.TensorNetwork
        First network.
    tn2 : qtn.TensorNetwork
        Second network.
    method : str
        Distance method passed to quimb (e.g. ``'auto'``, ``'dense'``,
        ``'overlap'``). Default ``'auto'``.

    Returns
    -------
    float
        The Frobenius distance between the two networks.
    """
    tn1 = tn1.copy()
    tn2 = tn2.copy()
    tn1.distribute_exponent()
    tn2.distribute_exponent()
    # print('distance tn norms', tn1.norm(), tn2.norm())
    return qtn.tensor_network_distance(tn1, tn2, method=method)


def max_inner_bond(mpx: 'qtn.TensorNetwork1D'):
    """Return the largest interior bond dimension of a 1D tensor network.

    Parameters
    ----------
    mpx : qtn.TensorNetwork1D
        The MPS/MPO whose bonds are inspected.

    Returns
    -------
    int
        The maximum size over all interior bonds.
    """
    if mpx is None:
        return None
    bond_sizes = [mpx.bond_size(i, i + 1) for i in range(mpx.L - 1)]
    return np.max(bond_sizes)


def inner_bond_sizes(mpx: 'qtn.TensorNetwork1D'):
    """Return the list of interior bond dimensions of a 1D tensor network.

    Parameters
    ----------
    mpx : qtn.TensorNetwork1D
        The MPS/MPO whose bonds are inspected.

    Returns
    -------
    list of int
        Sizes of each interior bond, from left to right.
    """
    if mpx is None:
        return None
    return [mpx.bond_size(i, i + 1) for i in range(mpx.L - 1)]


def add_rand_noise(mpx: 'qtn.TensorNetwork', strength=0.001, inplace=False):
    """Add Gaussian random noise to every tensor in a network.

    Parameters
    ----------
    mpx : qtn.TensorNetwork
        The network to perturb.
    strength : float
        Standard deviation of the normally distributed noise added to each
        tensor's data. Default 0.001.
    inplace : bool
        Modify ``mpx`` in place if True, else operate on a copy.

    Returns
    -------
    qtn.TensorNetwork
        The perturbed network.
    """
    mpx = mpx if inplace else mpx.copy()
    for t in mpx.tensors:
        rand_t = quimb.randn(t.shape, scale=strength)
        ## normally distributed data with stdev given by strength
        t.modify(apply=lambda x: x + rand_t.data)
    return mpx


def expectation_value(mpx: 'MPSType', obs_mpo: 'MPOType', bra: 'MPSType' = None):
    """Compute the expectation value ``<bra| obs_mpo |mpx>``.

    Aligns the bra, observable MPO, and ket index ids, then contracts the
    three-network sandwich site by site. The result is scaled by
    ``10 ** (mpx.exponent + obs_mpo.exponent + bra.exponent)``. A length-1
    observable with sparse data is handled by a direct matrix product.

    Parameters
    ----------
    mpx : MPSType
        The ket state.
    obs_mpo : MPOType
        The observable operator, applied between bra and ket.
    bra : MPSType, optional
        The bra state; defaults to the conjugate of ``mpx`` (with mangled inner
        bonds) when None.

    Returns
    -------
    Numeric
        The expectation value scaled by the combined exponent.
    """
    # print('MEAS EXPEC HELPER QUIMB')
    if bra is None:  bra = mpx.conj(mangle_inner=True)

    try:
        obs_mpo.lower_ind_id = mpx.site_ind_id
    except ValueError:  ## upper == lower
        obs_mpo.upper_ind_id = obs_mpo.upper_ind_id + '_'
        obs_mpo.lower_ind_id = mpx.site_ind_id
    bra.site_ind_id = obs_mpo.upper_ind_id
    # print('expec ket', mpx.site_ind_id, mpx)
    # print('expec bra', bra.site_ind_id, bra)
    # print('expec obs_mpo', obs_mpo.upper_ind_id, obs_mpo.lower_ind_id, obs_mpo)

    site_tags = set([mpx.site_tag_id, obs_mpo.site_tag_id, bra.site_tag_id])

    if obs_mpo.L == 1 and scipy.sparse.issparse(obs_mpo[0].data):
        if obs_mpo[0].inds[0] == bra.site_ind_id.format(0):
            tn_ = bra[0].data @ obs_mpo[0].data @ mpx[0].data
        else:
            tn_ = mpx[0].data @ obs_mpo[0].data @ bra.data[0].data
    else:
        tn_ = qtn.TensorNetwork([bra, obs_mpo, mpx], virtual=False)
        for i in range(mpx.L - 1, 0, -1):
            x_tags = [st.format(i) for st in site_tags] + [st.format(i - 1) for st in site_tags]
            tn_ = tn_.contract_tags(x_tags)

    tn_ = tn_ * 10 ** (mpx.exponent + obs_mpo.exponent + bra.exponent)

    return tn_

    # print('tn outer', tn_.outer_inds())
    # print('bra', bra)
    # print('obs mpo',obs_mpo)
    # print('mpx', mpx)a

    # return tn_.contract() * 10 ** (mpx.exponent + obs_mpo.exponent + bra.exponent)
    # return qtn.expec_TN_1D(bra, obs_mpo, mpx) * 10 ** mpx.exponent
    # return mpx.comp_norms() * 10 ** (mpx.exponent)


def singular_values(mpx: Union['MPSType', 'MPOType'], i: int, cur_orthog=None):
    """Return the singular values at bond ``i`` (between sites i-1 and i).

    Parameters
    ----------
    mpx : MPSType or MPOType
        The network to inspect.
    i : int
        Bond index at which the Schmidt/singular values are computed.
    cur_orthog : int or tuple, optional
        Current orthogonality center, passed to quimb to avoid recanonizing.

    Returns
    -------
    numpy.ndarray
        The singular values at bond ``i``.
    """
    sing_vals = mpx.singular_values(i, cur_orthog=cur_orthog)
    return sing_vals


def singular_values_all(mpx: Union['MPSType', 'MPOType']) -> list:
    """Return the singular values at every interior bond.

    Right-canonizes ``mpx`` first, then collects singular values bond by bond.

    Parameters
    ----------
    mpx : MPSType or MPOType
        The network to inspect (right-canonized in place).

    Returns
    -------
    list of numpy.ndarray
        Singular values for each interior bond, from left to right.
    """
    sing_vals = []
    mpx.right_canonize()
    for i in range(1, mpx.L):
        sing_vals += [singular_values(mpx, i, cur_orthog=i - 1)]
    return sing_vals


def entanglement_entropy(mpx: Union['MPSType', 'MPOType'], i: int, cur_orthog=None):
    """Compute the von Neumann entanglement entropy at bond ``i``.

    Normalizes the singular values at bond ``i`` and returns
    ``S = -sum_i p_i log2(p_i)`` with ``p_i = sigma_i^2`` (base-2 entropy).

    Parameters
    ----------
    mpx : MPSType or MPOType
        The network to inspect.
    i : int
        Bond index (between sites i-1 and i) at which entropy is measured.
    cur_orthog : int or tuple, optional
        Current orthogonality center, passed to quimb to avoid recanonizing.

    Returns
    -------
    float
        The entanglement entropy at bond ``i``.
    """
    sing_vals = mpx.singular_values(i, cur_orthog=cur_orthog)
    sing_vals *= 1. / np.linalg.norm(sing_vals)
    EE = -1 * np.sum(sing_vals ** 2 * np.log2(sing_vals ** 2))
    # print('mpx EE', mpx.calc_current_orthog_center(), sing_vals, EE)
    return EE


def entanglement_entropy_all(mpx: Union['MPSType', 'MPOType']) -> list:
    """Compute the von Neumann entanglement entropy at every interior bond.

    Right-canonizes ``mpx`` first, then evaluates the entropy bond by bond.

    Parameters
    ----------
    mpx : MPSType or MPOType
        The network to inspect (right-canonized in place).

    Returns
    -------
    list of float
        Entanglement entropy for each interior bond, from left to right.
    """
    EEs = []
    mpx.right_canonize()
    for i in range(1, mpx.L):
        EEs += [entanglement_entropy(mpx, i)]  # ,cur_orthog=i-1)]
    return EEs


def check_left_orthog(mpx: Union[Sequence, 'TN1Type'], right_ancillas: tuple[str] = None, verbose=False) -> int:
    """Find the index of the first tensor that is not left-canonical.

    Sweeps left to right contracting each tensor with its conjugate over the
    shared right bond and checks the result is the identity (within 1e-12).

    Parameters
    ----------
    mpx : Sequence or TN1Type
        The MPS/MPO (or list of tensors) to check.
    right_ancillas : tuple[str], optional
        Extra index labels on the last tensor to include when checking
        right-canonicality of the final site.
    verbose : bool
        Print the offending site and error magnitude when a violation is found.

    Returns
    -------
    int
        Index of the first non-left-canonical tensor (or the last index if all
        pass).
    """
    L = len(mpx) if isinstance(mpx, (list, tuple)) else mpx.L
    for i in range(L):
        tens = mpx[i]
        tens_conj = tens.conj()
        out_inds = None

        if i < L - 1:
            shared, not_shared = tens.filter_bonds(mpx[i + 1])
            tens_conj.reindex({ind: ind + '_tmp' for ind in shared}, inplace=True)
            try:
                ind_size = tens.ind_size(shared[0])
            except IndexError:  ## eg inserted matrices between tensors in mps
                continue

        else:
            if right_ancillas is None:
                ind_size = 0
            else:
                tens_conj.reindex({ind: ind + '_tmp' for ind in right_ancillas}, inplace=True)
                ind_size = np.prod([tens.ind_size(ind) for ind in right_ancillas])
                out_inds = [ind + '_tmp' for ind in right_ancillas] + [ind for ind in right_ancillas]

        check_tens = tens.contract(tens_conj)
        if out_inds is not None:
            check_tens.transpose(*out_inds, inplace=True)

        if ind_size == 0:
            if np.linalg.norm(check_tens - 1) > 1.0e-12:
                if verbose:
                    print('left orthog error', i, np.linalg.norm(check_tens - 1))
                break
        else:
            if np.linalg.norm(check_tens.data.reshape(ind_size, ind_size) - np.eye(ind_size)) / ind_size > 1.0e-12:
                if verbose:
                    print('left orthog error', i, np.linalg.norm(check_tens.data.reshape(ind_size, -1) - np.eye(ind_size)))
                break

    return i


def check_right_orthog(mpx: Union[Sequence, 'TN1Type'], left_ancillas: tuple[str] = None, verbose=False) -> int:
    """Find the index of the first tensor that is not right-canonical.

    Sweeps right to left contracting each tensor with its conjugate over the
    shared left bond and checks the result is the identity (within 1e-12).

    Parameters
    ----------
    mpx : Sequence or TN1Type
        The MPS/MPO (or list of tensors) to check.
    left_ancillas : tuple[str], optional
        Extra index labels on the first tensor to include when checking
        left-canonicality of the initial site.
    verbose : bool
        Print the offending site and error magnitude when a violation is found.

    Returns
    -------
    int
        Index of the first non-right-canonical tensor (or the first index if
        all pass).
    """
    L = len(mpx) if isinstance(mpx, (list, tuple)) else mpx.L
    for i in range(L - 1, -1, -1):
        tens = mpx[i]
        tens_conj = tens.conj()
        out_inds = None
        if i > 0:
            shared, not_shared = tens.filter_bonds(mpx[i - 1])
            tens_conj.reindex({ind: ind + '_tmp' for ind in shared}, inplace=True)
            try:
                ind_size = tens.ind_size(shared[0])
            except IndexError:  ## eg inserted matrices between tensors in mps
                continue
        else:
            if left_ancillas is None:
                ind_size = 0
            else:
                tens_conj.reindex({ind: ind + '_tmp' for ind in left_ancillas}, inplace=True)
                ind_size = np.prod([tens.ind_size(ind) for ind in left_ancillas])
                out_inds = [ind + '_tmp' for ind in left_ancillas] + [ind for ind in left_ancillas]

        check_tens = tens.contract(tens_conj)
        if out_inds is not None:
            check_tens.transpose(*out_inds, inplace=True)

        if ind_size == 0:
            if np.linalg.norm(check_tens - 1) > 1.0e-12:
                if verbose:
                    print('right orthog error', i, np.linalg.norm(check_tens - 1))
                break
        else:
            if np.linalg.norm(check_tens.data.reshape(ind_size, ind_size) - np.eye(ind_size)) / ind_size > 1.0e-12:
                if verbose:
                    print('right orthog error', i, np.linalg.norm(check_tens.data.reshape(ind_size, -1) - np.eye(ind_size)))
                # print('check_tens', check_tens.data)
                break
    return i


def check_orthog(mpx: Union[Sequence, 'TN1Type'], left_ancillas=None, right_ancillas=None, verbose=False) -> tuple[int, int]:
    """Return the orthogonality center bounds of an mpx as a (left, right) pair.

    Combines :func:`check_left_orthog` and :func:`check_right_orthog` to report
    how far left- and right-canonical forms extend.

    Parameters
    ----------
    mpx : Sequence or TN1Type
        The MPS/MPO (or list of tensors) to check.
    left_ancillas : tuple[str], optional
        Extra index labels on the first tensor for the right-orthogonality
        check.
    right_ancillas : tuple[str], optional
        Extra index labels on the last tensor for the left-orthogonality check.
    verbose : bool
        Print a diagnostic header if True.

    Returns
    -------
    tuple[int, int]
        ``(left, right)`` indices delimiting the canonical region.
    """
    if verbose:
        print('DMRG check orthog')
    left = check_left_orthog(mpx, right_ancillas=right_ancillas)
    right = check_right_orthog(mpx, left_ancillas=left_ancillas)
    return left, right


def pad_mps(mps: 'MPSType', mps_inds: Sequence[int], mps_L: int, inplace=False, pad_ind_size=None):
    """Pad an MPS up to length ``mps_L`` by inserting trivial sites.

    The original sites are placed at positions ``mps_inds`` (which must be in
    increasing order) and gaps are filled with padding tensors of ones
    (selected to a single component when ``pad_ind_size`` is None, else of size
    ``pad_ind_size``). Not an inplace operation unless ``inplace`` is set.

    Parameters
    ----------
    mps : MPSType
        The MPS to pad.
    mps_inds : Sequence[int]
        Target positions (increasing) of the original sites within the padded
        chain. If None or already ``range(mps_L)``, ``mps`` is returned as is.
    mps_L : int
        Length of the resulting padded MPS.
    inplace : bool
        Pass through to the underlying renumbering as in place if True.
    pad_ind_size : int, optional
        Physical dimension of padding sites; if None, padding sites carry a
        trivial (selected) physical index.

    Returns
    -------
    MPSType
        The padded MPS of length ``mps_L``.
    """
    if mps_inds is None or list(mps_inds) == list(range(mps_L)):
        return mps

    if len(mps_inds) > 1:
        assert (np.all(np.diff(mps_inds) > 0)), 'mps_inds must be in increasing order'

    site_tag_id = mps.site_tag_id
    site_ind_id = mps.site_ind_id

    padded_mps = renumber_mps(mps, list(range(mps.L)), mps_inds, site_tag_id=site_tag_id, site_ind_id=site_ind_id,
                              inplace=inplace)

    # print('helper renumbered mps', padded_mps)

    # if pad_ind_size is None:
    #     pad_ind_size = mps.ind_size(0)

    ind1 = 0
    for ind2 in mps_inds:
        if ind2 == ind1:  # no padded tensors on left
            ind1 += 1
            continue

        ## else pad from ind1 to ind2
        if ind1 == 0:
            if pad_ind_size is None:
                ones = ones_mps(ind2, 1, site_ind_id=site_ind_id, site_tag_id=site_tag_id)
                for idx in range(ind2):
                    ones[idx].isel({site_ind_id.format(idx): 0}, inplace=True)
            else:
                ones = ones_mps(ind2, pad_ind_size, site_ind_id=site_ind_id, site_tag_id=site_tag_id)
            padded_mps.add(ones)
            padded_mps.new_bond(site_tag_id.format(ind2 - 1), site_tag_id.format(ind2))
            ind1 = ind2 + 1
        else:
            for i in range(ind1, ind2):
                tag1, tag2 = site_tag_id.format(i - 1), site_tag_id.format(ind2)
                tens1, tens2 = padded_mps.select_tensors(tag1)[0], padded_mps.select_tensors(tag2)[0]
                bond_dim = tens1.shared_bond_size(tens2)
                padded_mps.insert_operator(np.eye(bond_dim), site_tag_id.format(i - 1), site_tag_id.format(ind2),
                                           tags=(site_tag_id.format(i),), inplace=True)
                if pad_ind_size:
                    tens = padded_mps.select_tensors((site_tag_id.format(i),))[0]
                    tens.new_ind(site_ind_id.format(i + 1), pad_ind_size)
            ind1 = ind2 + 1

    if ind2 < mps_L - 1:  ## need to pad tensors on right
        if pad_ind_size is None:
            ones = ones_mps(mps_L - ind2 - 1, 1, site_ind_id=site_ind_id, site_tag_id=site_tag_id)
            for idx in range(mps_L - ind2):
                ones[idx].isel({site_ind_id.format(idx): 0}, inplace=True)
            renumber_mps(ones, list(range(mps_L - ind2 - 1)), list(range(ind2 + 1, mps_L)), site_ind_id=site_ind_id,
                         site_tag_id=site_tag_id, inplace=True)
        else:
            ones = ones_mps(mps_L - ind2 - 1, pad_ind_size, site_ind_id=site_ind_id, site_tag_id=site_tag_id)
            renumber_mps(ones, list(range(mps_L - ind2 - 1)), list(range(ind2 + 1, mps_L)), site_ind_id=site_ind_id,
                         site_tag_id=site_tag_id, inplace=True)
        # print('padded_mps', padded_mps, ones)
        padded_mps.add(ones)
        padded_mps.new_bond(site_tag_id.format(ind2), site_tag_id.format(ind2 + 1))

    padded_mps._L = padded_mps.num_tensors

    # print('helper padded mps', padded_mps)

    return padded_mps


def mps_outerproduct(mps1: 'MPSType', mps2: 'MPSType', site_ind_id=None, site_tag_id=None, return_mps=True):
    """Form the site-wise outer product of two MPS.

    At each site the two physical legs are either fused into one new physical
    leg (``return_mps=True``) or kept as separate upper/lower legs of an MPO
    (``return_mps=False``). Inner bonds are fused and exponents are summed.

    Parameters
    ----------
    mps1 : MPSType
        First MPS (must equal ``mps2`` in length and differ in ``site_ind_id``).
    mps2 : MPSType
        Second MPS.
    site_ind_id : str, optional
        Site index id of the result; defaults to ``mps1.site_ind_id``.
    site_tag_id : str, optional
        Site tag id of the result; defaults to ``mps1.site_tag_id``.
    return_mps : bool
        If True, fuse the two physical legs into one and return an MPS; if
        False, return an MPO with separate upper/lower legs.

    Returns
    -------
    MPSType or MPOType
        The outer-product network.
    """
    assert (mps1.L == mps2.L), 'mps1 and mps2 need to have the same length'

    if site_ind_id is None:  site_ind_id = mps1.site_ind_id
    if site_tag_id is None:  site_tag_id = mps1.site_tag_id

    if mps1.site_ind_id == mps2.site_ind_id:
        raise RuntimeError('mps1, mps2 must have different site_ind_id')

    mps2 = mps2.copy()
    mps2.mangle_inner_()

    new_mps = qtn.TensorNetwork([])
    for x in range(mps1.L):
        tens = mps1[x].contract(mps2[x])
        if return_mps:
            ind1, ind2 = mps1.site_ind(x), mps2.site_ind(x)
            if ind2 in tens.inds:
                if ind1 in tens.inds:
                    tens.fuse({site_ind_id.format(x): (ind1, ind2)}, inplace=True)
                else:
                    tens.reindex({ind2: site_ind_id.format(x)}, inplace=True)
        tens.drop_tags()
        tens.add_tag(site_tag_id.format(x))
        new_mps.add(tens)

    if return_mps:
        new_mps.view_like(mps1, site_ind_id=site_ind_id, site_tag_id=site_tag_id, inplace=True)
    else:
        new_mps.view_as(qtn.MatrixProductOperator, upper_ind_id=mps1.site_ind_id, lower_ind_id=mps2.site_ind_id,
                        site_tag_id=site_tag_id, inplace=True, cyclic=False, L=new_mps.num_tensors)

    new_mps.fuse_multibonds(inplace=True)
    new_mps.exponent = mps1.exponent + mps2.exponent

    return new_mps


def mpo_transpose(mpo_, mangle_inner=True, inplace=False):
    """Swap an MPO's upper and lower index id labels (relabel only).

    Exchanges ``upper_ind_id`` and ``lower_ind_id`` via a temporary name; this
    relabels the legs without permuting tensor data, so it does not actually
    transpose the underlying matrices.

    Parameters
    ----------
    mpo_ : MPOType
        The MPO to relabel.
    mangle_inner : bool
        Mangle inner bond labels to keep them unique if True.
    inplace : bool
        Modify ``mpo_`` in place if True, else operate on a copy.

    Returns
    -------
    MPOType
        The relabeled MPO.
    """
    mpo_ = mpo_ if inplace else mpo_.copy()
    uid = mpo_.upper_ind_id
    lid = mpo_.lower_ind_id
    mpo_.lower_ind_id = '_tmp{}_'
    mpo_.upper_ind_id = lid
    mpo_.lower_ind_id = uid
    if mangle_inner:  mpo_.mangle_inner_()
    return mpo_


def mpo_flip_upper_lower(mpo_, mangle_inner=True, inplace=False):
    """Swap an MPO's upper and lower index ids without reindexing tensors.

    Sets the private ``_upper_ind_id``/``_lower_ind_id`` attributes directly so
    the existing leg labels are reinterpreted, effecting an actual transpose of
    the operator (the data stays put but the roles of the legs flip).

    Parameters
    ----------
    mpo_ : MPOType
        The MPO to transpose.
    mangle_inner : bool
        Mangle inner bond labels to keep them unique if True.
    inplace : bool
        Modify ``mpo_`` in place if True, else operate on a copy.

    Returns
    -------
    MPOType
        The transposed MPO.
    """
    mpo_ = mpo_ if inplace else mpo_.copy()
    uid = mpo_.upper_ind_id
    lid = mpo_.lower_ind_id
    mpo_._lower_ind_id = '_tmp{}_'
    mpo_._upper_ind_id = lid
    mpo_._lower_ind_id = uid
    if mangle_inner:  mpo_.mangle_inner_()
    return mpo_


def mpo_conj_transpose(mpo_, mangle_inner=True, inplace=False):
    """Return the conjugate transpose (adjoint) of an MPO.

    Flips upper/lower legs via :func:`mpo_flip_upper_lower` and conjugates the
    tensor data.

    Parameters
    ----------
    mpo_ : MPOType
        The MPO to take the adjoint of.
    mangle_inner : bool
        Mangle inner bond labels to keep them unique if True.
    inplace : bool
        Modify ``mpo_`` in place if True, else operate on a copy.

    Returns
    -------
    MPOType
        The conjugate-transposed MPO.
    """
    mpo_ = mpo_flip_upper_lower(mpo_, mangle_inner=mangle_inner, inplace=inplace)
    mpo_.conj(inplace=True)
    return mpo_


def mps_flip_lr(mps: 'qtn.MatrixProductState', inplace=True):
    """Reverse the site ordering of an MPS (left-right flip).

    Remaps every site tag and physical index id ``i -> L-1-i``.

    Parameters
    ----------
    mps : qtn.MatrixProductState
        The MPS to reverse; ``None`` returns ``None``.
    inplace : bool
        Modify ``mps`` in place if True (default), else operate on a copy.

    Returns
    -------
    qtn.MatrixProductState
        The site-reversed MPS.
    """
    if mps is None:
        return
    mps = mps if inplace else mps.copy()
    L = mps.L
    ind_map = {mps.site_ind_id.format(i): mps.site_ind_id.format(L - 1 - i) for i in range(L)}
    tag_map = {mps.site_tag_id.format(i): mps.site_tag_id.format(L - 1 - i) for i in range(L)}
    mps.retag(tag_map, inplace=True)
    mps.reindex(ind_map, inplace=True)
    return mps


def mpo_flip_lr(mpo: 'qtn.MatrixProductOperator', upper_L=None, lower_L=None, inplace=True):
    """Reverse the site ordering of an MPO (left-right flip).

    Remaps site tags ``i -> L-1-i`` and independently remaps the upper and
    lower physical index ids, allowing for MPOs where the upper and lower legs
    span different numbers of sites.

    Parameters
    ----------
    mpo : qtn.MatrixProductOperator
        The MPO to reverse.
    upper_L : int, optional
        Number of upper-leg sites to remap; defaults to ``mpo.L``.
    lower_L : int, optional
        Number of lower-leg sites to remap; defaults to ``mpo.L``.
    inplace : bool
        Modify ``mpo`` in place if True (default), else operate on a copy.

    Returns
    -------
    qtn.MatrixProductOperator
        The site-reversed MPO.
    """
    mpo = mpo if inplace else mpo.copy()
    ## in case not all mpo's have upper AND lower inds (but inds are indexed from 0, ..., upper/lower_L)
    upper_L = mpo.L if upper_L is None else upper_L
    lower_L = mpo.L if lower_L is None else lower_L
    upper_map = {mpo.upper_ind_id.format(i): mpo.upper_ind_id.format(upper_L - 1 - i) for i in range(upper_L)}
    lower_map = {mpo.lower_ind_id.format(i): mpo.lower_ind_id.format(lower_L - 1 - i) for i in range(lower_L)}
    tag_map = {mpo.site_tag_id.format(i): mpo.site_tag_id.format(mpo.L - 1 - i) for i in range(mpo.L)}
    mpo.retag(tag_map, inplace=True)
    mpo.reindex(upper_map, inplace=True)
    mpo.reindex(lower_map, inplace=True)
    return mpo


def renumber_mps(tn: 'MPSType', old_inds, new_inds, site_tag_id=None, site_ind_id=None, inplace=False) -> 'MPSType':
    """Renumber selected MPS sites from old indices to new indices.

    Useful, e.g., when some sites are removed/integrated out and the remaining
    tensors must be renumbered. Not designed to change the ``site_tag_id`` /
    ``site_ind_id`` format strings themselves.

    Parameters
    ----------
    tn : MPSType
        The MPS whose sites are renumbered.
    old_inds : Sequence[int]
        Current site numbers to remap.
    new_inds : Sequence[int]
        New site numbers (parallel to ``old_inds``).
    site_tag_id : str, optional
        Site tag id format; defaults to ``tn.site_tag_id``.
    site_ind_id : str, optional
        Site index id format; defaults to ``tn.site_ind_id``.
    inplace : bool
        Modify ``tn`` in place if True, else operate on a copy.

    Returns
    -------
    MPSType
        The renumbered MPS.
    """
    tn = tn if inplace else tn.copy()
    site_tag_id = tn.site_tag_id if site_tag_id is None else site_tag_id
    site_ind_id = tn.site_ind_id if site_ind_id is None else site_ind_id

    old_tens_list = [tn.select_tensors(site_tag_id.format(x))[0] for x in old_inds]
    for x in range(len(old_inds)):
        tens = old_tens_list[x]
        tens.reindex({site_ind_id.format(old_inds[x]): site_ind_id.format(new_inds[x])}, inplace=True)
        tens.retag({site_tag_id.format(old_inds[x]): site_tag_id.format(new_inds[x])}, inplace=True)
    return tn


def renumber_mpo(tn: 'MPOType', old_inds, new_inds, site_tag_id=None, upper_ind_id=None, lower_ind_id=None,
                 inplace=False) -> 'MPOType':
    """Renumber selected MPO sites from old indices to new indices.

    The MPO analogue of :func:`renumber_mps`; remaps both upper and lower legs
    plus the site tag. Not designed to change the ``site_tag_id`` /
    ``upper_ind_id`` / ``lower_ind_id`` format strings themselves.

    Parameters
    ----------
    tn : MPOType
        The MPO whose sites are renumbered.
    old_inds : Sequence[int]
        Current site numbers to remap.
    new_inds : Sequence[int]
        New site numbers (parallel to ``old_inds``).
    site_tag_id : str, optional
        Site tag id format; defaults to ``tn.site_tag_id``.
    upper_ind_id : str, optional
        Upper index id format; defaults to ``tn.upper_ind_id``.
    lower_ind_id : str, optional
        Lower index id format; defaults to ``tn.lower_ind_id``.
    inplace : bool
        Modify ``tn`` in place if True, else operate on a copy.

    Returns
    -------
    MPOType
        The renumbered MPO.
    """
    tn = tn if inplace else tn.copy()
    site_tag_id = tn.site_tag_id if site_tag_id is None else site_tag_id
    upper_ind_id = tn.upper_ind_id if upper_ind_id is None else upper_ind_id
    lower_ind_id = tn.lower_ind_id if lower_ind_id is None else lower_ind_id

    old_tens_list = [tn.select_tensors(site_tag_id.format(x))[0] for x in old_inds]
    for x in range(len(old_inds)):
        tens = old_tens_list[x]
        tens.reindex({upper_ind_id.format(old_inds[x]): upper_ind_id.format(new_inds[x]),
                      lower_ind_id.format(old_inds[x]): lower_ind_id.format(new_inds[x]), }, inplace=True)
        tens.retag({site_tag_id.format(old_inds[x]): site_tag_id.format(new_inds[x])}, inplace=True)
    return tn


def match_inner_inds(mps, ref_mps, inplace=True, append=''):
    """Relabel an MPS's inner bond indices to match a reference MPS.

    For each interior bond, reindexes the bond of ``mps`` to the corresponding
    bond label of ``ref_mps`` (optionally with ``append`` appended).

    Parameters
    ----------
    mps : MPSType
        The MPS whose bond labels are changed.
    ref_mps : MPSType
        Reference MPS supplying the target bond labels.
    inplace : bool
        Modify ``mps`` in place if True (default), else operate on a copy.
    append : str
        Suffix appended to each reference bond label before assigning.

    Returns
    -------
    MPSType
        The MPS with matched inner bond labels.
    """
    mps = mps if inplace else mps.copy()
    for i in range(mps.L - 1):
        indR1 = mps.bond(i, i + 1)
        indR2 = ref_mps.bond(i, i + 1)
        mps[i].reindex({indR1: indR2 + append}, inplace=True)
        mps[i + 1].reindex({indR1: indR2 + append}, inplace=True)
    return mps


def replace_mps(mps: qtn.MatrixProductState, mps_new: qtn.MatrixProductState) -> 'MPSType':
    """Copy the tensor data of ``mps_new`` into ``mps`` (in place).

    For each site, aligns index ordering (bonds and physical leg) between the
    two MPS and overwrites ``mps``'s data with that of ``mps_new``; the
    exponent is copied too.

    Parameters
    ----------
    mps : qtn.MatrixProductState
        The MPS whose data is overwritten.
    mps_new : qtn.MatrixProductState
        The MPS supplying the new data and exponent.

    Returns
    -------
    MPSType
        ``mps`` with replaced data.
    """
    for i in range(mps.L):
        new = mps_new[i]
        old = mps[i]
        inds_old, inds_new = [], []
        if i > 0:
            inds_old += [mps.bond(i - 1, i)]
            inds_new += [mps_new.bond(i - 1, i)]
        if i < mps.L - 1:
            inds_old += [mps.bond(i, i + 1)]
            inds_new += [mps_new.bond(i, i + 1)]
        inds_old += [mps.site_ind_id.format(i)]
        inds_new += [mps_new.site_ind_id.format(i)]

        new.transpose(*inds_new, inplace=True)
        old.transpose(*inds_old, inplace=True)
        old.modify(data=new.data)
    mps.exponent = mps_new.exponent
    return mps


def mps_to_diag_mpo(mps: 'MPSType', upper_ind_id=None, lower_ind_id=None, sparse=False) -> 'MPOType':
    """Convert an MPS into a diagonal MPO carrying the MPS entries on its diagonal.

    Each physical leg is split into upper/lower legs through COPY (delta)
    tensors so the resulting operator is diagonal in the physical basis. A
    length-1 MPS can be built directly as a sparse CSR diagonal when
    ``sparse=True``.

    Parameters
    ----------
    mps : MPSType
        The MPS to embed on the diagonal.
    upper_ind_id : str, optional
        Upper index id of the output MPO; left at the derived default if None.
    lower_ind_id : str, optional
        Lower index id of the output MPO; left at the derived default if None.
    sparse : bool
        For a length-1 MPS, build the diagonal as a scipy.sparse CSR array.

    Returns
    -------
    MPOType
        The diagonal MPO.
    """
    bond_name = mps.site_ind_id
    tens_name = mps.site_tag_id

    qs = mps.shape  # physical bond dimensions; physical bonds are all external dims

    out = mps.view_as(qtn.TensorNetwork, inplace=False)

    bond_name_1 = bond_name + '[1]'
    bond_name_2 = bond_name + '[2]'

    if sparse and mps.L == 1:

        ## assumes data is represented as a vector
        tens = mps[0]
        q = tens.size
        rows = cols = list(range(q))
        diag_csr_data = scipy.sparse.csr_array((tens.data, (rows, cols)), shape=(q, q))

        # from quimb import qu
        # tmp = qu(diag_coo_data, qtype='dop', sparse=True)

        diag_tens = qtn.Tensor(data=diag_csr_data, inds=(bond_name_1.format(0), bond_name_2.format(0)),
                               tags=(mps.site_tag_id.format(0),))
        out = qtn.TensorNetwork([])
        out.add_tensor(diag_tens)

        out.exponent = mps.exponent


    else:
        for i in range(mps.num_tensors):

            d_ijk = qtn.tensor_core.COPY_tensor(qs[i],
                                                (bond_name.format(i), bond_name_1.format(i), bond_name_2.format(i)),
                                                tags=(f'd_ijk({i})',))
            out.add_tensor(d_ijk)
            if mps.num_tensors == 1:
                out_tens = out.contract(tags=(tens_name.format(i), f'd_ijk({i})'), inplace=True)
                out = qtn.TensorNetwork([out_tens])
                out.exponent = mps.exponent
            else:
                out.contract(tags=(tens_name.format(i), f'd_ijk({i})'), inplace=True)

    out.view_as(qtn.MatrixProductOperator, inplace=True, like=mps,
                upper_ind_id=bond_name_1, lower_ind_id=bond_name_2)
    out.drop_tags([f'd_ijk({i})' for i in range(mps.L)])

    if upper_ind_id is not None:   out.upper_ind_id = upper_ind_id
    if lower_ind_id is not None:   out.lower_ind_id = lower_ind_id

    return out


def zipup_fuse(mpx, inplace=False, direction=1):
    """Fuse/recompress multi-bonds between neighbouring tensors via a zip-up sweep.

    Sweeps along the chain splitting each tensor (SVD, absorbing to the right,
    using module ``CUTOFF``/``CUTOFF_MODE``) and contracting the remainder into
    the neighbour, falling back to the eig method on linear-algebra failure.

    Parameters
    ----------
    mpx : MPSType or MPOType
        The network to fuse.
    inplace : bool
        Modify ``mpx`` in place if True, else operate on a copy.
    direction : int
        Sweep direction: ``> 0`` left-to-right, ``<= 0`` right-to-left.

    Returns
    -------
    MPSType or MPOType
        The fused network.
    """
    mpx = mpx if inplace else mpx.copy()

    if direction > 0:
        for i in range(mpx.L - 1):
            T1, T2 = mpx[i], mpx[i + 1]
            rix, lix = T1.filter_bonds(T2)
            try:
                TL, TR = T1.split(lix, get='tensors', absorb='right', cutoff_mode=CUTOFF_MODE, right_inds=rix,
                                  cutoff=CUTOFF)
            except np.linalg.LinAlgError:
                TL, TR = T1.split(lix, get='tensors', absorb='right', cutoff_mode=CUTOFF_MODE, right_inds=rix,
                                  cutoff=CUTOFF, method='eig')

            T1.modify(data=TL.data, inds=TL.inds)
            T2_ = T2.contract(TR)
            T2.modify(data=T2_.data, inds=T2_.inds)

    else:
        for i in range(mpx.L - 1, 0, -1):
            T1, T2 = mpx[i], mpx[i - 1]
            rix, lix = T1.filter_bonds(T2)
            try:
                TL, TR = T1.split(lix, get='tensors', absorb='right', cutoff_mode=CUTOFF_MODE, right_inds=rix,
                                  cutoff=CUTOFF)
            except np.linalg.LinAlgError:
                TL, TR = T1.split(lix, get='tensors', absorb='right', cutoff_mode=CUTOFF_MODE, right_inds=rix,
                                  cutoff=CUTOFF, method='eig')

            T1.modify(data=TL.data, inds=TL.inds)
            T2_ = T2.contract(TR)
            T2.modify(data=T2_.data, inds=T2_.inds)

    # print('zipup fuse out', mpx)
    return mpx


# @profile
def apply(mpo1, mpx2, compress=False, verbose=False, compress_opts: dict = None) -> Union['MPSType', 'MPOType']:
    """Apply an MPO to an MPS/MPO by exact contraction, optionally compressing.

    Aligns the operator's lower legs with ``mpx2``'s physical legs and contracts
    site by site (handling sparse length-1 operators and an IndexError fallback
    in quimb). Exponents of the two networks are summed. ``mpo1.site_tag_id`` is
    set to match ``mpx2``.

    Parameters
    ----------
    mpo1 : MPOType
        The operator to apply; may have a different ``site_tag_id`` or be
        length 1.
    mpx2 : MPSType or MPOType
        The state/operator the MPO acts on.
    compress : bool
        Compress the result (via :func:`compress`) if True (only for L > 1).
    verbose : bool
        Print diagnostics during compression if True.
    compress_opts : dict, optional
        Options forwarded to :func:`compress` (reads keys such as
        ``'max_bond'``, ``'cutoff'``, ``'cutoff_mode'``, ``'form'``).

    Returns
    -------
    MPSType or MPOType
        The result of the application.
    """
    mpo1.site_tag_id = mpx2.site_tag_id

    if mpo1.L == 1:
        A, x = mpo1.copy(), mpx2.copy()

        # align the indices
        A.lower_ind_id = "__tmp{}__"
        if isinstance(mpx2, qtn.MatrixProductState):
            A.upper_ind_id = x.site_ind_id
            x.reindex_sites_("__tmp{}__")
        elif isinstance(mpx2, qtn.MatrixProductOperator):
            A.upper_ind_id = x.upper_ind_id
            x.reindex_upper_sites_("__tmp{}__")

        ## to deal with potentially sparse mats
        A_tens = A[0]
        A_mat = A_tens.data
        x_vec = x[0].data

        new_mpx = x.copy()
        if isinstance(new_mpx, qtn.MatrixProductState):
            if A_tens.inds[0] == A.upper_ind_id.format(0):
                out = A_mat @ x_vec
            else:
                out = A_mat.T @ x_vec
            new_mpx[0].modify(data=out, inds=(x.site_ind_id.format(0),), tags=(x.site_tag_id.format(0),))

        elif isinstance(new_mpx, qtn.MatrixProductOperator):
            if A_tens.inds[0] == A.upper_ind_id.format(0):
                if x[0].inds[0] == "__tmp0__":
                    out = A_mat @ x_vec
                else:
                    out = A_mat @ x_vec.T
            else:
                if x[0].inds[0] == "__tmp0__":
                    out = A_mat.T @ x_vec
                else:
                    out = A_mat.T @ x_vec.T
            new_mpx[0].modify(data=out, inds=(x.upper_ind_id.format(0), x.lower_ind_id.format(0)),
                              tags=(x.site_tag_id.format(0),))

        new_mpx.exponent = x.exponent + A.exponent

        # # form total network and contract each site
        # x.add(A)  # adding tensors to contract with each other A*x
        # x.exponent = mpo1.exponent + mpx2.exponent
        # x.contract_ind("__tmp0__")
        # new_mpx = x

    else:
        A, x = mpo1.copy(), mpx2.copy()

        try:
            # align the indices
            tmp_str = "__" + qtn.rand_uuid() + "{}__"
            A.lower_ind_id = tmp_str
            new_mpx = A.apply(x, inplace=False)  ## takes exponents into account

        except IndexError:  # in quimb

            new_mpx = x.copy()

            # align the indices
            tmp_str = "__" + qtn.rand_uuid() + "{}__"
            A.lower_ind_id = tmp_str

            if isinstance(mpx2, qtn.MatrixProductState):
                A.upper_ind_id = x.site_ind_id
                x.reindex_sites_(tmp_str)

            elif isinstance(mpx2, qtn.MatrixProductOperator):
                A.upper_ind_id = x.upper_ind_id
                x.reindex_upper_sites_(tmp_str)

            for i in range(x.L):
                if scipy.sparse.issparse(A[i].data):
                    A[i].modify(data=A[i].data.todense())
                new_tens = qtn.tensor_contract(A[i], x[i])
                new_tens.transpose_like(new_mpx[i], inplace=True)
                new_mpx[i].modify(data=new_tens.data)

        # print('apply', x.exponent, A.exponent, new_mpx.exponent)
        if compress:
            new_mpx = compress_func(new_mpx, scale=True, verbose=verbose, compress_opts=compress_opts)

    return new_mpx


# @profile
def apply_zipup(mpo1: 'MPOType', mpx2: Union['MPSType', 'MPOType'],
                compress=False, verbose=False, compress_opts: dict = None) -> Union['MPSType', 'MPOType']:
    """Apply an MPO to an MPS/MPO using the approximate zip-up contraction.

    Sweeps along the chain contracting operator and state tensors, splitting
    (SVD, absorbing right) on the fly to bound the bond dimension, normalizing
    each new tensor and accumulating ``log10`` norms into the exponent, with an
    eig-based retry when nans appear. The starting/final canonical ``form`` is
    chosen so the optional final compression yields the requested form.

    Parameters
    ----------
    mpo1 : MPOType
        The operator to apply.
    mpx2 : MPSType or MPOType
        The state/operator the MPO acts on.
    compress : bool
        Perform a final :func:`compress` pass if True.
    verbose : bool
        Print diagnostics (e.g. nan retries) if True.
    compress_opts : dict, optional
        Options forwarded to the splits and final compression; ``'cutoff'`` and
        ``'cutoff_mode'`` default to the module constants and ``'form'`` (default
        ``'right'``) controls sweep direction.

    Returns
    -------
    MPSType or MPOType or None
        The contracted result, or ``None`` if a zero norm is encountered.
    """
    mpo1.site_tag_id = mpx2.site_tag_id

    if compress_opts is None:
        compress_opts = {}
    else:
        compress_opts = compress_opts.copy()
    compress_opts.setdefault('cutoff', CUTOFF)
    compress_opts.setdefault('cutoff_mode', CUTOFF_MODE)

    form = compress_opts.get('form', 'right')
    if not compress:
        form = 'right' if form == 'left' else 'left'
        # we don't perform the last compression to put it in the desired form

    if mpo1.L == 1:
        return apply(mpo1, mpx2)
        # A, x = mpo1.copy(), mpx2.copy()
        #
        # # align the indices
        # A.lower_ind_id = "__tmp{}__"
        # if isinstance(mpx2, qtn.MatrixProductState):
        #     A.upper_ind_id = x.site_ind_id
        #     x.reindex_sites_("__tmp{}__")
        # elif isinstance(mpx2, qtn.MatrixProductOperator):
        #     A.upper_ind_id = x.upper_ind_id
        #     x.reindex_upper_sites_("__tmp{}__")
        #
        # ## to deal with potentially sparse mats
        # A_tens = A[0]
        # A_mat = A_tens.data
        # x_vec = x[0].data
        #
        # if A_tens.inds[0] == A.upper_ind_id.format(0):
        #     out = A_mat @ x_vec
        # else:
        #     out = A_mat.T @ x_vec
        # new_mpx = x.copy()
        # new_mpx[0].modify(data=out, inds=(x.site_ind_id.format(0),))
        #
        # # # form total network and contract each site
        # # x.add(A)  # adding tensors to contract with each other A*x
        # # x.exponent = mpo1.exponent + mpx2.exponent
        # # x.contract_ind("__tmp0__")
        # # new_mpx = x

    else:
        A, x = mpo1.copy(), mpx2.copy().mangle_inner_()
        ## perhaps optional? bc i take norm out separately?
        # canonize(A, i=0)
        # canonize(x, i=0)

        # align the indices
        A.lower_ind_id = "__tmp{}__"
        if isinstance(mpx2, qtn.MatrixProductState):
            A.upper_ind_id = x.site_ind_id
            x.reindex_sites_("__tmp{}__")
        elif isinstance(mpx2, qtn.MatrixProductOperator):
            A.upper_ind_id = x.upper_ind_id
            x.reindex_upper_sites_("__tmp{}__")

        new_mpx = qtn.TensorNetwork([])

        if form == 'right':  # first put in left canonical form
            new_tens = qtn.tensor_contract(A[0], x[0])
            new_tens.drop_tags()

            t_norm = new_tens.norm()
            if t_norm == 0.0:
                return None
            new_tens.modify(apply=lambda data: data / t_norm)
            new_mpx.exponent += np.log10(t_norm)

            A_idx, open_idx = new_tens.filter_bonds(A[1])
            x_idx, open_idx = new_tens.filter_bonds(x[1])
            lix = [idx for idx in open_idx if idx not in A_idx + x_idx]
            tensL, tensR = new_tens.split(lix, absorb='right', cutoff_mode=CUTOFF_MODE, right_inds=A_idx + x_idx,
                                          cutoff=compress_opts.get('cutoff', CUTOFF), ltags=(x.site_tag(0)))

            it = 0
            while np.any(np.isnan(tensL.data)) or np.any(np.isnan(tensR.data)):
                tensL, tensR = new_tens.split(lix, method='eig', absorb='right', cutoff_mode=CUTOFF_MODE,
                                              right_inds=A_idx + x_idx,
                                              cutoff=compress_opts.get('cutoff', CUTOFF), ltags=(x.site_tag(0)))
                print('apply zipup split yielded nans', 0, new_tens.norm())
                it += 1
                if it > 10:
                    raise ValueError('apply zipup split yielding nans')

            new_mpx.add(tensL)

            for i in range(1, x.L - 1):
                # print('contract', tensR, A[i], x[i])
                # new_tens = qtn.tensor_contract(tensR,x[i])
                # new_tens = qtn.tensor_contract(new_tens,A[i])
                new_tens = qtn.tensor_contract(tensR, x[i], A[i])
                new_tens.drop_tags()

                t_norm = new_tens.norm()
                if t_norm == 0.0:
                    return None
                new_tens.modify(apply=lambda data: data / t_norm)
                new_mpx.exponent += np.log10(t_norm)

                A_idx, open_idx = new_tens.filter_bonds(A[i + 1])
                x_idx, open_idx = new_tens.filter_bonds(x[i + 1])
                lix = [idx for idx in open_idx if idx not in A_idx + x_idx]
                # print('new_tens', new_tens.norm())
                tensL, tensR = new_tens.split(lix, absorb='right', cutoff_mode=CUTOFF_MODE, right_inds=A_idx + x_idx,
                                              cutoff=compress_opts.get('cutoff', CUTOFF), ltags=(x.site_tag(i)))

                it = 0
                while np.any(np.isnan(tensL.data)) or np.any(np.isnan(tensR.data)):
                    tensL, tensR = new_tens.split(lix, method='eig', absorb='right', cutoff_mode=CUTOFF_MODE,
                                                  right_inds=A_idx + x_idx,
                                                  cutoff=compress_opts.get('cutoff', CUTOFF), ltags=(x.site_tag(i)))
                    print('apply zipup split yielded nans', i, new_tens.norm())
                    it += 1
                    if it > 10:
                        raise ValueError('apply zipup split yielding nans')

                new_mpx.add(tensL)

            new_tens = qtn.tensor_contract(tensR, A[-1], x[-1])
            new_mpx.add(new_tens)
            i = x.L - 1

        else:
            new_tens = qtn.tensor_contract(A[x.L - 1], x[x.L - 1])
            new_tens.drop_tags()

            t_norm = new_tens.norm()
            if t_norm == 0.0:
                return None
            new_tens.modify(apply=lambda data: data / t_norm)
            new_mpx.exponent += np.log10(t_norm)

            A_idx, open_idx = new_tens.filter_bonds(A[x.L - 2])
            x_idx, open_idx = new_tens.filter_bonds(x[x.L - 2])
            lix = [idx for idx in open_idx if idx not in A_idx + x_idx]
            tensL, tensR = new_tens.split(lix, absorb='right', cutoff_mode=CUTOFF_MODE, right_inds=A_idx + x_idx,
                                          cutoff=compress_opts.get('cutoff', CUTOFF), ltags=(x.site_tag(x.L - 1)))

            it = 0
            while np.any(np.isnan(tensL.data)) or np.any(np.isnan(tensR.data)):
                tensL, tensR = new_tens.split(lix, method='eig', absorb='right', cutoff_mode=CUTOFF_MODE,
                                              right_inds=A_idx + x_idx,
                                              cutoff=compress_opts.get('cutoff', CUTOFF), ltags=(x.site_tag(x.L - 1)))
                print('apply zipup split yielded nans: site', x.L - 1, new_tens.norm())
                it += 1
                if it > 10:
                    raise ValueError('apply zipup split yielding nans')

            new_mpx.add(tensL)

            for i in range(x.L - 2, 0, -1):
                # print('contract', tensR, A[i], x[i])
                # new_tens = qtn.tensor_contract(tensR, x[i])
                # new_tens = qtn.tensor_contract(new_tens, A[i])
                new_tens = qtn.tensor_contract(tensR, x[i], A[i])
                new_tens.drop_tags()

                t_norm = new_tens.norm()
                if t_norm == 0.0:
                    return None
                new_tens.modify(apply=lambda data: data / t_norm)
                new_mpx.exponent += np.log10(t_norm)

                A_idx, open_idx = new_tens.filter_bonds(A[i - 1])
                x_idx, open_idx = new_tens.filter_bonds(x[i - 1])
                lix = [idx for idx in open_idx if idx not in A_idx + x_idx]
                # print('new_tens', new_tens.norm())
                tensL, tensR = new_tens.split(lix, absorb='right', cutoff_mode=CUTOFF_MODE, right_inds=A_idx + x_idx,
                                              cutoff=compress_opts.get('cutoff', CUTOFF), ltags=(x.site_tag(i)))

                it = 0
                while np.any(np.isnan(tensL.data)) or np.any(np.isnan(tensR.data)):
                    tensL, tensR = new_tens.split(lix, method='eig', absorb='right', cutoff_mode=CUTOFF_MODE,
                                                  right_inds=A_idx + x_idx,
                                                  cutoff=compress_opts.get('cutoff', CUTOFF), ltags=(x.site_tag(i)))
                    print('apply zipup split yielded nans: site', i, new_tens.norm())
                    it += 1
                    if it > 10:
                        raise ValueError('apply zipup split yielding nans')

                new_mpx.add(tensL)
                # print('tensL', tensL.norm())

            new_tens = qtn.tensor_contract(tensR, A[0], x[0])
            # print('new_tens', tensR.norm(), A[0].norm(), x[0].norm(), new_tens.norm())
            new_mpx.add(new_tens)
            i = 0

        new_mpx = new_mpx.view_like(x, inplace=True)
        new_mpx.exponent += A.exponent + x.exponent
        new_mpx.strip_exponent(new_mpx[i])

        # print('zip up before', new_mpx.max_bond())
        if compress:
            compress_opts['form'] = form
            new_mpx = compress_func(new_mpx, scale=True, verbose=verbose, canonize=False, compress_opts=compress_opts)
            # print('zip up after', new_mpx.max_bond())

    return new_mpx


# @profile
def apply_rdm(mpo1: 'MPOType', mps2: 'MPSType', bra_mpo1: 'MPOType' = None, bra_mps2: 'MPSType' = None,
              direction=1, left_env=None, right_env=None, open_end=False, compress_opts: dict = None,
              compress=True, verbose=False) -> Union['MPSType', tuple['MPSType', 'qtn.Tensor']]:
    """Apply an MPO to an MPS using the reduced-density-matrix (RDM) scheme.

    Builds environment tensors from one end, forms a reduced density matrix at
    each site, diagonalizes it to obtain truncated isometries (eigenvectors of
    the largest eigenvalues, capped by ``max_bond`` / ``cutoff``), and strings
    them into the result MPS. When ``compress`` is True the sweep direction is
    reversed and a final compression puts the state in the matching canonical
    form. A length-1 MPO falls back to :func:`apply`.

    Parameters
    ----------
    mpo1 : MPOType
        The operator to apply.
    mps2 : MPSType
        The state acted on.
    bra_mpo1 : MPOType, optional
        Bra-side operator; internally regenerated as the conjugate of ``mpo1``.
    bra_mps2 : MPSType, optional
        Bra-side state; internally regenerated as the conjugate of ``mps2``.
    direction : int
        ``< 0``: left-to-right, result left-canonical; ``> 0``: right-to-left,
        result right-canonical. Negated internally when ``compress`` is True.
    left_env : qtn.Tensor, optional
        Precomputed left environment (used when ``direction < 0``).
    right_env : qtn.Tensor, optional
        Precomputed right environment (used when ``direction > 0``).
    open_end : bool
        If True, leave the terminal site open and also return the carry tensor
        ``C_tens``.
    compress_opts : dict, optional
        Compression options; reads ``'cutoff'`` (default CUTOFF) and
        ``'max_bond'`` (default None) and sets ``'form'`` for the final pass.
    compress : bool
        Perform the reverse-sweep compression if True.
    verbose : bool
        Print environment/RDM diagnostics if True.

    Returns
    -------
    MPSType or tuple[MPSType, qtn.Tensor]
        The new MPS, plus the carry tensor when ``open_end`` is True.
    """
    if mpo1.L == 1:
        return apply(mpo1, mps2)

    mpo1 = mpo1.copy()
    mpo1 = mpo1.mangle_inner_()
    mps2 = mps2.mangle_inner_()
    L = mpo1.L
    cutoff = compress_opts.get('cutoff', CUTOFF)
    max_bond = compress_opts.get('max_bond', None)

    if verbose:
        print('mpo1', mpo1.max_bond(), 'mps2', mps2.max_bond())
        print('apply rdm mpo1', mpo1)
        print('apply rdm mps2', mps2)
        print('apply rdm bra mpo1', bra_mpo1)
        print('apply rdm bra mps2', bra_mps2)

    if compress:
        direction *= -1

    if True:  # bra_mps2 is None:
        bra_mps2 = mps2.conj(mangle_inner=False)
        for tens in bra_mps2:
            tens.reindex({ind: ind + '_' for ind in tens.inds}, inplace=True)
        bra_mps2._site_ind_id = bra_mps2.site_ind_id + '_'

    if True:  # bra_mpo1 is None:
        bra_mpo1 = mpo1.conj(mangle_inner=False)
        bra_mpo1 = mpo_flip_upper_lower(bra_mpo1, mangle_inner=False, inplace=True)
        for tens in bra_mpo1:
            tens.reindex({ind: ind + '_' for ind in tens.inds}, inplace=True)
        bra_mpo1._lower_ind_id = bra_mpo1.lower_ind_id + '_'
        bra_mpo1._upper_ind_id = bra_mpo1.upper_ind_id + '_'

    # qtn.tensor_network_align(mps2, mpo1, bra_mpo1, bra_mps2, inplace=True)
    mpo1.upper_ind_id = mpo1.upper_ind_id + '_x_'
    bra_mpo1.lower_ind_id = mpo1.upper_ind_id
    x_ind_id = mpo1.upper_ind_id

    idx0 = 0 if direction > 0 else L - 1
    idx1 = 0 if direction < 0 else L - 1

    ## ancillas at "far" end (away from end at which we start building envs)
    mps_anc = [ind for ind in mps2[idx0].inds if ind not in
               [mps2.site_ind(idx0), mps2.bond(idx0, idx0 + 1 * np.sign(direction))]]
    mpo_anc = [ind for ind in mpo1[idx0].inds if ind not in
               [mpo1.upper_ind(idx0), mpo1.lower_ind(idx0), mpo1.bond(idx0, idx0 + 1 * np.sign(direction))]]

    #### build envs
    env = right_env if direction > 0 else left_env
    envs = {idx1: env}

    ## dir = -1:  (L) 0 to L-1, dir = 1:   (R) L-1 to 0
    for i in range(idx1, idx0, -1 * np.sign(direction)):
        next_i = i - np.sign(direction)
        contract_tens_list = [mps2[i], mpo1[i], bra_mpo1[i], bra_mps2[i]]
        if env is not None:
            contract_tens_list += [env]

        env = qtn.TensorNetwork(contract_tens_list)
        env = env.contract()
        envs[next_i] = env
        if verbose:
            print('contract tens list', contract_tens_list)
            print('env', i, env.shape)

    ## get unitaries
    Us = {}
    C_tens = None

    ## dir = 1:   (L) 0 to L-1  --> left canon
    ## dir = -1:  (R) L-1 to 0  --> right canon
    for i in range(idx0, idx1 + np.sign(direction), np.sign(direction)):
        next_i = i + np.sign(direction)
        prev_i = i - np.sign(direction)

        ### last site.
        ### if not True, tens1 is rdm of site[idx1] + appropriate env (hack for comb geometry)
        if i == idx1 and not open_end:
            contract_tens_list = [mps2[idx1], mpo1[idx1]]
            if C_tens is not None:
                contract_tens_list += [C_tens]
            tens1 = qtn.tensor_contract(*contract_tens_list)
            tens1.drop_tags()
            tens1.add_tag(mps2.site_tag(idx1))
            Us[idx1] = tens1
            break

        bra_mpo_tens = bra_mpo1[i].reindex({x_ind_id.format(i): x_ind_id.format(i) + '_o_'})
        contract_tens_list = [mps2[i], mpo1[i], bra_mpo_tens, bra_mps2[i]]
        env = envs.get(i, None)
        if env is not None:
            contract_tens_list += [env]
        if C_tens is not None:
            C_tens_conj = C_tens.conj(inplace=False)
            C_tens_conj.reindex({ind: ind + '_' for ind in C_tens.inds}, inplace=True)
            contract_tens_list += [C_tens, C_tens_conj]

        # rdm = qtn.tensor_contract( *contract_tens_list )
        rdm = qtn.TensorNetwork([contract_tens_list])
        rdm = rdm.contract()

        if verbose:  print('rdm', i, rdm.shape, rdm.inds)

        ket_inds = [x_ind_id.format(i)]
        if i == idx0:
            ket_inds += mps_anc + mpo_anc
        if C_tens is not None:
            ket_inds += [Us[prev_i].inds[-1]]  ## ind from evecs_tens

        bra_inds = [x_ind_id.format(i) + '_o_'] + [ind + '_' for ind in ket_inds[1:]]
        rdm.transpose(*ket_inds, *bra_inds, inplace=True)
        sq_shape = rdm.shape[:len(ket_inds)]

        ### quimb version
        rdm_mat = rdm.data.reshape(np.prod(sq_shape), -1)
        if max_bond is None:
            evals, evecs = quimb.eigh(rdm_mat, k=-1, sort=True, return_vecs=True)
            if evals[0] / evals[-1] < cutoff:
                cut_ind_1 = np.argmax(evals / evals[-1] < cutoff)
                cut_ind_2 = np.argmax(evals < CUTOFF)
                cut_ind = max(cut_ind_1, cut_ind_2)
                evals = evals[cut_ind:]
                evecs = evecs[:, cut_ind:]  ## input inds, sort_inds (horizontal bond)
        else:
            evals, evecs = quimb.eigh(rdm_mat,
                                      k=(max_bond if max_bond is not None else -1), tol=cutoff,
                                      which='LM', sort=True, return_vecs=True, )
        ## evals assorted in ascending order; targetting largest magnitude eigvals
        evecs = evecs[:, ::-1].conj()  ## qarray object

        ### numpy version
        # rdm_mat = rdm.data.reshape(np.prod(sq_shape), -1)
        # # if verbose:   print('rdm mat is H', np.linalg.norm(rdm_mat - rdm_mat.T.conj()))
        # evals, evecs = np.linalg.eigh(rdm_mat)
        #
        # if verbose:  print('cutoff', cutoff, 'max_bond', max_bond)
        # sort_inds = np.argsort(evals)[::-1]
        # if max_bond is not None:
        #     sort_inds = sort_inds[:max_bond]
        #
        # evals = evals[sort_inds]
        # if evals[-1]/evals[0] < cutoff:
        #     cut_ind = np.argmax(evals/evals[0] < cutoff)
        #     sort_inds = sort_inds[:cut_ind]
        #
        # evecs = evecs[:,sort_inds].conj()     ## input inds, sort_inds (horizontal bond)
        #####

        h_ind = f'h_{i}'  # if i != idx1 else last_ind
        evecs_tens = qtn.Tensor(data=evecs.reshape(*sq_shape, -1), inds=ket_inds + [h_ind],
                                tags=mps2.site_tag(i))
        Us[i] = evecs_tens.conj(inplace=False)

        contract_tens_list = [mps2[i], mpo1[i], evecs_tens]
        if C_tens is not None:
            contract_tens_list += [C_tens]
        C_tens = qtn.tensor_contract(*contract_tens_list)

    ## string Us together to make new MPS
    new_mpx = qtn.TensorNetwork([Us[i] for i in range(L)])
    new_mpx.view_like(mps2, inplace=True, site_ind_id=x_ind_id)
    new_mpx.site_ind_id = mps2.site_ind_id
    new_mpx.exponent = mps2.exponent + mpo1.exponent

    if compress:
        compress_opts['form'] = 'left' if idx1 < idx0 else 'right'
        # print('check orthog', check_orthog(new_mpx), new_mpx.exponent, compress_opts['form'])
        ## recall that we switched direction at beginnong of function
        new_mpx = compress_func(new_mpx, scale=True, canonize=False, compress_opts=compress_opts)

    if open_end:
        return new_mpx, C_tens
    else:
        return new_mpx


def sum_list(*mpxes: Union[qtn.MatrixProductState, qtn.MatrixProductOperator], zipup=False, inplace=False,
             compress=False, compress_opts: dict = None):
    """Sum an arbitrary number of MPS/MPO operands, skipping ``None`` terms.

    Accumulates the terms with :func:`add_MPO` or :func:`add_MPS` depending on
    type, optionally compressing the running total after each addition.

    Parameters
    ----------
    *mpxes : qtn.MatrixProductState or qtn.MatrixProductOperator
        The operands to sum; ``None`` entries are ignored.
    zipup : bool
        Accepted for signature compatibility (passed through is not used here).
    inplace : bool
        Accepted for signature compatibility.
    compress : bool
        Compress the running total after each addition if True.
    compress_opts : dict, optional
        Options forwarded to :func:`compress` (e.g. ``'max_bond'``,
        ``'cutoff'``).

    Returns
    -------
    qtn.MatrixProductState or qtn.MatrixProductOperator or None
        The summed network, or ``None`` if all operands were ``None``.
    """
    tot_mpx = None
    for mpx in mpxes:
        if mpx is None:  continue

        if tot_mpx is None:
            tot_mpx = mpx.copy()
            continue

        if isinstance(mpx, qtn.MatrixProductOperator):
            add_MPO(tot_mpx, mpx, inplace=True, compress=False)
        else:
            add_MPS(tot_mpx, mpx, inplace=True, compress=False)

        if compress:
            tot_mpx = compress_func(tot_mpx, compress_opts=compress_opts)

    return tot_mpx


def add_MPS_target(mps1: qtn.MatrixProductState, mps2: qtn.MatrixProductState, inplace=False,
                   direction=1, do_final_update=True,
                   compress_opts: dict = None, verbose=False, ):
    """Add two MPS via a DMRG-style sweep that targets ``mps1``'s basis.

    Accounts for the norms stored in each ``mps.exponent``, canonizes both
    operands, then sweeps updating one site at a time (``loc.update_1site``) and
    projecting ``mps2`` onto the growing ``mps1`` basis (``loc.decimate``). With
    ``do_final_update=False`` the terminal site is not updated, so ``mps1``'s
    basis is merely expanded to include ``mps2`` rather than the sum formed.

    Parameters
    ----------
    mps1 : qtn.MatrixProductState
        First MPS; ``None``/``-inf`` exponent returns a copy of ``mps2``.
    mps2 : qtn.MatrixProductState
        Second MPS; ``None``/``-inf`` exponent returns ``mps1``.
    inplace : bool
        Modify ``mps1`` in place if True, else operate on a copy.
    direction : int
        ``>= 0`` sweeps left-to-right (end left-canonical); ``< 0`` sweeps
        right-to-left (end right-canonical).
    do_final_update : bool
        Whether to perform the final-site update (form the actual sum).
    compress_opts : dict, optional
        Reads ``'cutoff'`` (default CUTOFF) and ``'max_bond'`` (default MAXBOND).
    verbose : bool
        Print exponent/orthogonality diagnostics if True.

    Returns
    -------
    qtn.MatrixProductState
        The summed (or basis-expanded) MPS.
    """
    cutoff = compress_opts.get('cutoff', CUTOFF) if compress_opts is not None else CUTOFF
    max_bond = compress_opts.get('max_bond', MAXBOND) if compress_opts is not None else MAXBOND

    if mps1 is None or np.isneginf(mps1.exponent):   return mps2.copy()
    if mps2 is None or np.isneginf(mps2.exponent):
        if inplace:
            return mps1
        else:
            return mps1.copy()

    out = mps1 if inplace else mps1.copy()
    tmp = mps2.reindex_sites(mps1.site_ind_id)
    tmp = match_inner_inds(tmp, mps1, inplace=True)

    import local_solvers.helper_dmrg_loc as loc

    # out.distribute_exponent()
    tmp.exponent -= out.exponent
    tmp.distribute_exponent()
    tmp.exponent = out.exponent

    if direction >= 0:
        out = canonize(out, i=0, scale=False)
        tmp = canonize(tmp, i=0, scale=False)

        if out is None:
            return None

        # print('out', check_orthog(out))
        # print('tmp', check_orthog(tmp))

        for i in range(out.L - 1):
            loc.update_1site(out, i, [out[i], tmp[i]], direction=1, max_bond=max_bond)
            loc.decimate(tmp, i, out[i], direction=1)

            # print('out check orthog', i, check_orthog(out))
            # print('tmp check orthog', i, check_orthog(tmp))

        if do_final_update:
            i = out.L - 1
            loc.update_1site(out, i, [out[i], tmp[i]], direction=1, max_bond=max_bond)

    else:
        out = canonize(out, i=out.L - 1, scale=False)
        tmp = canonize(tmp, i=out.L - 1, scale=False)

        if out is None:
            return None

        for i in range(out.L - 1, 0, -1):
            loc.update_1site(out, i, [out[i], tmp[i]], direction=-1, max_bond=max_bond)
            loc.decimate(tmp, i, out[i], direction=-1)

        if do_final_update:
            i = 0
            loc.update_1site(out, i, [out[i], tmp[i]], direction=-1, max_bond=max_bond)

    return out


def target_MPS(mps: qtn.MatrixProductState, target_mps: qtn.MatrixProductState, inplace=False,
               site: int = 0):
    """Project an MPS onto the subspace spanned by ``target_mps``.

    Computes the left and right overlap environments between ``mps`` and the
    (conjugated, canonized) ``target_mps`` about ``site``, contracts them with
    ``mps[site]`` to form the projected central tensor, and rebuilds the MPS
    using ``target_mps``'s tensors away from ``site``.

    Parameters
    ----------
    mps : qtn.MatrixProductState
        The MPS to project.
    target_mps : qtn.MatrixProductState
        The MPS defining the target subspace.
    inplace : bool
        Modify ``mps`` in place if True, else operate on a copy.
    site : int
        The central site about which the projection is performed.

    Returns
    -------
    qtn.MatrixProductState
        The projected MPS.
    """

    target_mps = target_mps.conj(inplace=False)
    target_mps.mangle_inner_()
    target_mps.site_ind_id = mps.site_ind_id
    canonize(target_mps, i=site)

    l_ovlp = qtn.TensorNetwork([mps[:site], target_mps[:site]])
    if l_ovlp.num_tensors > 0:
        l_ovlp = l_ovlp.contract()

    r_ovlp = qtn.TensorNetwork([mps[site + 1:], target_mps[site + 1:]])
    if r_ovlp.num_tensors > 0:
        r_ovlp = r_ovlp.contract()

    new_site = qtn.TensorNetwork([l_ovlp, r_ovlp, mps[site]])
    new_site = new_site.contract()
    reindex_inds = {}
    if site > 0:
        reindex_inds[target_mps.bond(site, site - 1)] = mps.bond(site, site - 1)
    if site < mps.L - 1:
        reindex_inds[target_mps.bond(site, site + 1)] = mps.bond(site, site + 1)
    new_site = new_site.reindex(reindex_inds, inplace=False)  # back to mps inds

    mps_copy = mps.copy()
    new_mps = mps if inplace else mps.copy()
    match_inner_inds(target_mps, new_mps, inplace=True)
    for i in range(mps.L):
        site_i = new_mps[i]
        if i == site:
            # print('site i', site_i, site_i.data.shape)
            # print('new site', new_site, new_site.data.shape)
            # new_site.transpose(*site_i.inds, inplace=True)
            # # new_site.transpose_like(site_i, inplace=True)
            # site_i.modify(data = new_site.data)
            assert (set(new_site.inds) == (set(site_i.inds)))
            site_i.modify(data=new_site.data, inds=new_site.inds)
        else:
            assert (set(target_mps[i].inds) == set(site_i.inds))
            # target_mps[i].transpose_like(site_i, inplace=True)
            # site_i.modify(data = target_mps.data)
            site_i.modify(data=target_mps[i].data, inds=target_mps[i].inds)

    # print('target mps check orthog', site)
    # check_orthog(new_mps)
    # print(distance(new_mps, mps_copy))
    # print(distance(new_mps[site+1:], target_mps[site+1:]))

    return new_mps


def add_MPS_list(mps_list: Sequence[qtn.MatrixProductState], inplace=False,
                 direction=1, do_final_update=True, do_canonize=True,
                 compress_opts: dict = None, norm_cutoff=None, update_with_zero=False):
    """Add a list of MPS together via a single DMRG-style sweep.

    Generalizes :func:`add_MPS_target` to many operands: all terms are reindexed
    and matched to the first non-trivial term ``out``, optionally canonized, then
    a single sweep updates each site of ``out`` from all operands
    (``loc.update_1site``) while projecting the rest onto ``out``'s basis
    (``loc.decimate``); a final compression sets the canonical form. With
    ``do_final_update=False`` only the basis is expanded.

    Parameters
    ----------
    mps_list : Sequence[qtn.MatrixProductState]
        The MPS operands to sum; leading ``None``/``-inf`` terms are skipped.
    inplace : bool
        Modify the first operand in place if True.
    direction : int
        ``< 0``: final form right-canonical; otherwise left-canonical.
    do_final_update : bool
        Whether to perform the final-site update and compression.
    do_canonize : bool
        Whether to canonize operands before sweeping.
    compress_opts : dict, optional
        Reads ``'cutoff'`` (default CUTOFF) and ``'max_bond'`` (default MAXBOND).
    norm_cutoff : float
        set mps to zero if its norm below this cutoff value
    update_with_zero: bool
        replace the final tensor with a zero tensor

    Returns
    -------
    qtn.MatrixProductState
        The summed (or basis-expanded) MPS.
    """
    cutoff = compress_opts.get('cutoff', CUTOFF) if compress_opts is not None else CUTOFF
    max_bond = compress_opts.get('max_bond', MAXBOND) if compress_opts is not None else MAXBOND

    if not do_final_update:
        direction = direction * -1

    iter_mps = iter(mps_list)
    out = None
    it = 0
    while out is None or np.isneginf(out.exponent):
        out = next(iter_mps)
        if out is not None:
            out = out if inplace else out.copy()
        it += 1

    new_list = []
    for mps2 in mps_list[it:]:
        if mps2 is None:
            continue
        tmp = mps2.reindex_sites(out.site_ind_id)
        tmp = match_inner_inds(tmp, out, inplace=True)

        if do_canonize and do_final_update:
            tmp.exponent -= out.exponent
            tmp.distribute_exponent()
            tmp.exponent = out.exponent

        new_list += [tmp]

    import local_solvers.helper_dmrg_loc as loc

    if direction < 0:  ## end (after compression) is in right canoncial form
        if do_canonize:
            out = canonize(out, i=0, scale=False)
            for tmp in new_list:
                tmp = canonize(tmp, i=0, scale=False)

        # print('out', check_orthog(out))
        # print('tmp', check_orthog(tmp))

        for i in range(out.L - 1):
            ### update site i in "out"
            loc.update_1site(out, i, [out[i], *[tmp[i] for tmp in new_list]],
                             direction=1, cutoff=cutoff/10)

            ### project other terms onto "out" basis
            for tmp in new_list:
                loc.decimate(tmp, i, out[i], direction=1)
                # print('tmp check orthog', i + 1, check_orthog(tmp))

        if do_final_update or update_with_zero:
            i = out.L - 1
            loc.update_1site(out, i, [out[i], *[tmp[i] for tmp in new_list]],
                             direction=1, cutoff=cutoff/10) #, max_bond=max_bond)

            compress_opts = {'max_bond': max_bond, 'cutoff': cutoff, 'form': 'right'}
            out = compress_func(out, compress_opts=compress_opts, scale=False, norm_cutoff=norm_cutoff)

        if update_with_zero and out is not None:
            out[0].modify(apply=lambda x: x * 0)

    else:  ## end (after compression) is in left canonical form
        if do_canonize:
            out = canonize(out, i=out.L - 1, scale=False)
            for tmp in new_list:
                tmp = canonize(tmp, i=out.L - 1, scale=False)

        # print('out', check_orthog(out))

        for i in range(out.L - 1, 0, -1):
            loc.update_1site(out, i, [out[i], *[tmp[i] for tmp in new_list]],
                             direction=-1, cutoff=cutoff/10)
            # print('out check orthog', i - 1, check_orthog(out))

            for tmp in new_list:
                loc.decimate(tmp, i, out[i], direction=-1)
                # print('tmp check orthog', i - 1, check_orthog(tmp))

        if do_final_update or update_with_zero:
            i = 0
            tens = helper_tn.sum_tens([out[i], *[tmp[i] for tmp in new_list]] )
            loc.update_1site(out, i, tens, direction=-1, cutoff=cutoff/10) #, max_bond=max_bond)

            compress_opts = {'max_bond': max_bond, 'cutoff': cutoff, 'form': 'left'}
            out = compress_func(out, compress_opts=compress_opts, scale=False, norm_cutoff=norm_cutoff)

        if update_with_zero and out is not None:
            out[out.L-1].modify(apply=lambda x: x * 0)
            out.exponent = 0.0

    return out


def add_MPS(mps1: qtn.MatrixProductState, mps2: qtn.MatrixProductState, zipup=False, inplace=False,
            compress=False, compress_opts: dict = None, verbose=False):
    """Add two MPS directly (or via zip-up), accounting for their exponents.

    Canonizes both operands, rescales ``mps2``'s exponent relative to ``mps1``,
    then either uses :func:`add_zipup` or quimb's ``add_MPS`` followed by an
    optional :func:`compress`.

    Parameters
    ----------
    mps1 : qtn.MatrixProductState
        First MPS; ``None``/``-inf`` exponent returns a copy of ``mps2``.
    mps2 : qtn.MatrixProductState
        Second MPS; ``None``/``-inf`` exponent returns ``mps1``.
    zipup : bool
        Use the zip-up addition (:func:`add_zipup`) instead of direct addition.
    inplace : bool
        Modify ``mps1`` in place if True, else operate on a copy.
    compress : bool
        Compress the result if True.
    compress_opts : dict, optional
        Options forwarded to compression (e.g. ``'max_bond'``, ``'cutoff'``,
        ``'form'``).
    verbose : bool
        Print diagnostics during compression if True.

    Returns
    -------
    qtn.MatrixProductState
        The summed MPS.
    """
    if mps1 is None or np.isneginf(mps1.exponent):   return mps2.copy()
    if mps2 is None or np.isneginf(mps2.exponent):
        if inplace:
            return mps1
        else:
            return mps1.copy()

    out = mps1 if inplace else mps1.copy()
    tmp = mps2.reindex_sites(mps1.site_ind_id)

    canonize(out, scale=True)
    canonize(tmp, scale=True)

    # out.distribute_exponent()
    tmp.exponent -= out.exponent
    tmp.distribute_exponent()
    tmp.exponent = out.exponent
    # print('normal add', distance(tmp, mps2))

    # print('out', out.exponent, 'tmp', tmp.exponent)
    # print('sum', np.linalg.norm(to_dense(out) + to_dense(tmp)))

    if zipup:
        out = add_zipup(out, tmp, inplace=True, compress=compress, compress_opts=compress_opts)
    else:
        out.add_MPS(tmp, inplace=True)
        if compress:  out = compress_func(out, compress_opts=compress_opts, scale=False,
                                          verbose=verbose)

    # out.equalize_norms()

    return out


def add_MPO(mpo1: qtn.MatrixProductOperator, mpo2: qtn.MatrixProductOperator,
            zipup=False, inplace=False, compress=False, compress_opts: dict = None):
    """Add two MPOs, aligning their leg ids and accounting for their exponents.

    Reconciles ``mpo2``'s upper/lower index ids with ``mpo1`` (transposing if
    they are swapped), rescales its exponent, then adds via :func:`add_zipup`,
    quimb's ``add_MPO``, or a direct sparse sum for length-1 sparse operators.

    Parameters
    ----------
    mpo1 : qtn.MatrixProductOperator
        First MPO; ``None``/``-inf`` exponent returns a copy of ``mpo2``.
    mpo2 : qtn.MatrixProductOperator
        Second MPO; ``None``/``-inf`` exponent returns ``mpo1``.
    zipup : bool
        Use the zip-up addition (:func:`add_zipup`) instead of direct addition.
    inplace : bool
        Modify ``mpo1`` in place if True, else operate on a copy.
    compress : bool
        Compress the result if True.
    compress_opts : dict, optional
        Options forwarded to compression (e.g. ``'max_bond'``, ``'cutoff'``).

    Returns
    -------
    qtn.MatrixProductOperator
        The summed MPO.
    """
    if mpo1 is None or np.isneginf(mpo1.exponent):   return mpo2.copy()
    if mpo2 is None or np.isneginf(mpo2.exponent):
        if inplace:
            return mpo1
        else:
            return mpo1.copy()

    out = mpo1 if inplace else mpo1.copy()

    tmp = mpo2.copy()
    if tmp.upper_ind_id != mpo1.upper_ind_id:
        if mpo1.upper_ind_id == tmp.lower_ind_id:
            tmp.lower_ind_id = '_tmp{}_'
            tmp.upper_ind_id = mpo1.upper_ind_id
            tmp.lower_ind_id = mpo1.lower_ind_id
        else:
            tmp.upper_ind_id = mpo1.upper_ind_id

    if tmp.lower_ind_id != mpo1.lower_ind_id:
        tmp.lower_ind_id = mpo1.lower_ind_id

    # out.distribute_exponent()
    tmp.exponent -= out.exponent
    tmp.distribute_exponent()

    if zipup:
        add_zipup(out, tmp, inplace=True, compress=compress, compress_opts=compress_opts)
    else:
        if out.L == 1 and scipy.sparse.issparse(out[0].data):
            if out[0].inds == tmp[0].inds:
                out[0].modify(data=out[0].data + tmp[0].data)
            else:
                out[0].modify(data=out[0].data + tmp[0].data.T)
        else:
            out.add_MPO(tmp, inplace=True)
            if compress:  out = compress_func(out, compress_opts=compress_opts)

    return out


def add_submpx(mpx1: qtn.TensorNetwork1D,
               mpx2: qtn.TensorNetwork1D,
               mpx2_ind_range: tuple[int, int], open_bc=False,
               inplace=False, compress=False, compress_opts: dict = None):
    """Add a shorter mpx2 into a contiguous sub-range of a longer mpx1.

    ``mpx2`` acts on the continuous site range ``mpx2_ind_range`` of ``mpx1``
    (same site ordering). Tensors are combined with direct products over the
    physical (and boundary) indices; ``mpx1`` should be in canonical form
    around that chunk. With ``open_bc`` the boundary ancilla bonds are handled
    by direct-producting the neighbouring tensors and extending the active range.

    Parameters
    ----------
    mpx1 : qtn.TensorNetwork1D
        The longer host network.
    mpx2 : qtn.TensorNetwork1D
        The shorter network added into ``mpx1``'s sub-range.
    mpx2_ind_range : tuple[int, int]
        ``(min_ind, max_ind)`` site range of ``mpx1`` that ``mpx2`` covers.
    open_bc : bool
        If True, treat the chunk boundaries as open rather than summing over the
        boundary ancilla bonds.
    inplace : bool
        Modify ``mpx1`` in place if True, else operate on a copy.
    compress : bool
        Canonize and compress the active range after adding if True.
    compress_opts : dict, optional
        Options forwarded to :func:`compress_tens_list`.

    Returns
    -------
    qtn.TensorNetwork1D
        The combined network.
    """
    new_mpx1 = mpx1 if inplace else mpx1.copy()
    mpx2 = mpx2.copy()
    mpx2.exponent -= mpx1.exponent
    mpx2.distribute_exponent()

    # print('mpx1', new_mpx1)
    # print('mpx2', mpx2)

    min_ind, max_ind = mpx2_ind_range
    sub_L = max_ind - min_ind
    active_range = [min_ind, max_ind]

    # print('add submpx active range', active_range)

    def get_ancilla_ind(tens, shared_inds):
        """Return the single index of ``tens`` not in ``shared_inds`` (an ancilla).

        Parameters
        ----------
        tens : qtn.Tensor
            The tensor to inspect.
        shared_inds : sequence of str
            Index labels considered shared (physical or bond legs).

        Returns
        -------
        str or None
            The leftover ancilla index label, or None if none remain.
        """
        anc_ind = None
        for ind in tens.inds:
            if not ind in shared_inds:
                anc_ind = ind
                continue
            else:
                pass
        return anc_ind

    def get_phys_inds(mpx, site_ind):
        """Return the physical index labels of ``mpx`` at a given site.

        Parameters
        ----------
        mpx : qtn.TensorNetwork1D
            The MPS or MPO.
        site_ind : int
            Site number whose physical leg(s) are requested.

        Returns
        -------
        list of str
            ``[upper, lower]`` for an MPO, ``[site]`` for an MPS.
        """
        if isinstance(mpx, qtn.MatrixProductOperator):
            return [mpx.upper_ind_id.format(site_ind), mpx.lower_ind_id.format(site_ind)]
        elif isinstance(mpx, qtn.MatrixProductState):
            return [mpx.site_ind_id.format(site_ind)]
        else:
            raise TypeError

    # check if sub_gtn[0] has a left bond that needs to be summed over
    mpx_left = get_ancilla_ind(mpx1[min_ind], get_phys_inds(mpx1, min_ind) + list(mpx1[min_ind + 1].inds))
    # check if sub_gtn[L-1] has a right bond that needs to be summed over
    mpx_right = get_ancilla_ind(mpx1[max_ind - 1],
                                get_phys_inds(mpx1, max_ind - 1) + list(mpx1[max_ind - 2].inds))

    ## add tensors
    for i in range(sub_L):
        i_ = min_ind + i
        iL_, iR_ = i_ - 1, i_ + 1
        t1, t2 = mpx1[i_], mpx2[i]

        sum_inds = get_phys_inds(mpx1, i_)

        ## match inds if necessary
        if set(t1.inds) != set(t2.inds):
            reindex_map = {}

            phys_ind_1 = get_phys_inds(mpx1, i_)
            phys_ind_2 = get_phys_inds(mpx2, i)

            for ix1, ix2 in zip(phys_ind_1, phys_ind_2):
                if ix1 != ix2:
                    reindex_map[ix2] = ix1

            if i_ == 0:
                left_ind_1 = get_ancilla_ind(t1, phys_ind_1 + list(mpx1[iR_].inds))
            else:
                left_ind_1, not_shared = t1.filter_bonds(mpx1[iL_])
                left_ind_1 = next(iter(left_ind_1))

            if i == 0:
                left_ind_2 = get_ancilla_ind(t2, phys_ind_2 + list(mpx2[i + 1].inds))
            else:
                left_ind_2, not_shared = t2.filter_bonds(mpx2[i - 1])
                left_ind_2 = next(iter(left_ind_2))

            if left_ind_1 != left_ind_2:
                reindex_map[left_ind_2] = left_ind_1

            if i_ == mpx1.L - 1:
                right_ind_1 = get_ancilla_ind(t1, phys_ind_1 + list(mpx1[iL_].inds))
            else:
                right_ind_1, not_shared = t1.filter_bonds(mpx1[iR_])
                right_ind_1 = next(iter(right_ind_1))

            if i == sub_L - 1:
                right_ind_2 = get_ancilla_ind(t2, phys_ind_2 + list(mpx2[i - 1].inds))
            else:
                right_ind_2, not_shared = t2.filter_bonds(mpx2[i + 1])
                right_ind_2 = next(iter(right_ind_2))

            if right_ind_1 != right_ind_2:
                reindex_map[right_ind_2] = right_ind_1

            t2 = t2.reindex(reindex_map)

        ## add with open_bc or not
        if i == 0 and mpx_left is not None:
            if not open_bc:
                sum_inds += [mpx_left]
            else:
                if i_ > 0:
                    tL = new_mpx1[iL_]
                    sum_inds_L = [ind for ind in tL.inds if ind != mpx_left]
                    tL.direct_product_(tL, sum_inds=sum_inds_L)
                    active_range[0] -= 1

        elif i == sub_L - 1 and mpx_right is not None:
            # print('mpx right', mpx_right, i, sub_L - 1)
            if not open_bc:
                sum_inds += [mpx_right]
            else:
                if i_ < new_mpx1.L - 1:
                    tR = new_mpx1[iR_]
                    sum_inds_R = [ind for ind in tR.inds if ind != mpx_right]
                    tR.direct_product_(tR, sum_inds=sum_inds_R)
                    active_range[1] += 1

        # print('t1', t1)
        # print('t2', t2)
        # print('sum inds', sum_inds)
        t1.direct_product_(t2, sum_inds=sum_inds)

    if compress:
        # print('new mpx1 add', new_mpx1.max_bond())
        tens_list = new_mpx1[active_range[0]:active_range[1]]
        # print('tens list', tens_list)
        reverse_tens_list = [t for t in tens_list][::-1]
        canonize_tens_list(*reverse_tens_list, inplace=True)
        compress_tens_list(*[t for t in tens_list], compress_opts=compress_opts, inplace=True)
        # print('new mpx1 add', new_mpx1.max_bond())

    return new_mpx1


# @profile
def add_zipup(mpx1: Union[qtn.MatrixProductState, qtn.MatrixProductOperator],
              mpx2: Union[qtn.MatrixProductState, qtn.MatrixProductOperator],
              inplace=False, compress=False, compress_opts: dict = None):
    """Add two MPS/MPO of equal length via a zip-up (direct-product + compress) sweep.

    Sweeps along the chain forming the direct product of corresponding tensors
    over the physical legs and compressing each adjacent pair on the fly
    (:func:`compress_tens_list`), accumulating norms into the exponent. The
    pre-compression ``form`` is flipped when ``compress`` is True so the final
    pass yields the requested form.

    Parameters
    ----------
    mpx1 : qtn.MatrixProductState or qtn.MatrixProductOperator
        First operand; the result is built from a copy (or in place).
    mpx2 : qtn.MatrixProductState or qtn.MatrixProductOperator
        Second operand (must match ``mpx1`` in length).
    inplace : bool
        Modify ``mpx1`` in place if True, else operate on a copy.
    compress : bool
        Perform a final :func:`compress` pass if True.
    compress_opts : dict, optional
        Reads ``'form'`` (default ``'right'``) and is forwarded to compression.

    Returns
    -------
    qtn.MatrixProductState or qtn.MatrixProductOperator or None
        The summed network, or ``None`` if a tensor collapses to empty.
    """
    if mpx1.L != mpx2.L:
        raise ValueError("Can't add MPS with another of different length.")

    new_mpx = mpx1 if inplace else mpx1.copy()

    form = compress_opts.get('form', 'right') if compress_opts is not None else 'right'
    if compress:
        form = 'left' if form == 'right' else 'right'
        ## form for add+canon, so that final compression yields desired form

    prev_tens = None
    site_inds = range(new_mpx.L - 1, -1, -1) if form == 'right' else range(new_mpx.L)

    for i in site_inds:
        t1, t2 = new_mpx[i], mpx2[i]

        if set(t1.inds) != set(t2.inds):
            # Need to use bonds to match indices
            reindex_map = {}

            if i > 0 or mpx1.cyclic:
                pair = ((i - 1) % mpx1.L, i)
                reindex_map[mpx2.bond(*pair)] = new_mpx.bond(*pair)

            if i < new_mpx.L - 1 or mpx1.cyclic:
                pair = (i, (i + 1) % mpx1.L)
                reindex_map[mpx2.bond(*pair)] = new_mpx.bond(*pair)

            t2 = t2.reindex(reindex_map)

        if isinstance(new_mpx, qtn.MatrixProductState):
            sum_inds = new_mpx.site_ind(i)
        else:
            sum_inds = (new_mpx.upper_ind(i), new_mpx.lower_ind(i))
        t1.direct_product_(t2, sum_inds=sum_inds)

        if prev_tens is not None:
            compress_tens_list(prev_tens, t1, inplace=True)
            if prev_tens.data is None or t1.data is None:
                return None

        new_mpx.strip_exponent(t1)
        prev_tens = t1

    if compress:
        new_mpx = compress_func(new_mpx, compress_opts=compress_opts)

    # print('check orthog')
    # check_orthog(new_mpx)

    return new_mpx


def regularize(mpx: Union['MPSType', 'MPOType'], inplace=True):
    """Reset an mpx's exponent to its actual norm and spread it over the tensors.

    Sets ``mpx.exponent = log10(norm(mpx))`` and then distributes that exponent
    evenly across the constituent tensors.

    Parameters
    ----------
    mpx : MPSType or MPOType
        The network to regularize.
    inplace : bool
        Modify ``mpx`` in place if True (default), else operate on a copy.

    Returns
    -------
    MPSType or MPOType
        The regularized network.
    """
    mpx = mpx if inplace else mpx.copy()
    mpx_norm = norm(mpx)
    mpx._exponent = np.log10(mpx_norm)
    mpx.distribute_exponent()
    return mpx


def canonize(mps: Union['MPSType', 'MPOType'], scale=True, form='right', i=None, cur_orthog=None, bra=None):
    """Canonicalize an MPS/MPO around a site, optionally absorbing the norm (in place).

    Left- and right-compresses sites to move the orthogonality center to ``i``
    (using :func:`left_compress_site`/:func:`right_compress_site`, which catch
    linear-algebra errors). When ``scale`` is True the center site is normalized
    and its norm absorbed into ``mps.exponent`` (via the nested
    ``site_norm_to_exponent``). Returns ``None`` when the state is numerically
    zero or its exponent drops below ``-30``. A length-1 network is returned
    unchanged.

    Parameters
    ----------
    mps : MPSType or MPOType
        The network to canonize.
    scale : bool
        Absorb the center-site norm into the exponent if True.
    form : {'right', 'left'}
        Default canonical form used to pick the center when ``i`` is None
        (``'right'`` -> i=0, ``'left'`` -> i=L-1).
    i : int, optional
        Target orthogonality center site.
    cur_orthog : int, tuple, or 'calc', optional
        Current orthogonality center to start from; ``'calc'`` recomputes it via
        :func:`check_orthog`, an int is treated as ``(i, i)``, None as the full
        range.
    bra : MPSType, optional
        If given, its tensors are overwritten with the conjugate of the
        canonized ``mps`` and its exponent synced.

    Returns
    -------
    MPSType or MPOType or None
        The canonized network, or ``None`` if it is effectively zero.
    """
    if mps.L == 1:
        return mps

    if mps is None or np.isneginf(mps.exponent):   return None

    # if not scale:
    #     mps_copy = mps.copy()
    # else:
    #     mps_copy = mps
    mps_copy = mps

    if i is None:
        if form == 'right':
            i = 0
        elif form == 'left':
            i = mps.L - 1

    if cur_orthog is None:
        cur_orthog = (0, mps.L - 1)
    elif cur_orthog == 'calc':
        # cur_orthog = mps.calc_current_orthog_center()
        cur_orthog = check_orthog(mps)
    elif isinstance(cur_orthog, int):
        cur_orthog = (cur_orthog, cur_orthog)

    # if not isinstance(cur_orthog, tuple):
    #     if cur_orthog < 0:
    #         cur_orthog = (0, mps.L-1)
    #     else:
    #         cur_orthog = (cur_orthog, cur_orthog)

    try:
        if False:  # not scale:
            return mps.canonize_func(i, )
        else:

            ## helper function to normalize site i, shift weight into mps.exponent
            def site_norm_to_exponent(ix):
                """Normalize site ``ix`` of ``mps`` and add its norm to the exponent.

                Parameters
                ----------
                ix : int
                    Site index to normalize; raises ``LinAlgError`` (and sets the
                    exponent to ``-inf``) if the tensor norm is zero.

                Returns
                -------
                None
                """
                # print('canon remove site norm', ix)
                tag = mps.site_tag(ix)
                tens = mps[tag]
                if tens.norm() == 0.0:
                    mps.exponent = -np.inf
                    raise np.linalg.LinAlgError
                # tens = mps.select_tensors(tag)
                # print('site norm to exponent', tens.norm())
                mps.strip_exponent(tens)  # normalizes tensor, adds norm to exponent

            ## canonicalize
            for x in range(cur_orthog[0], i):
                # mps.left_canonize_site(x)
                left_compress_site(mps, x, max_bond=None, cutoff=CUTOFF,
                                   cutoff_mode=CUTOFF_MODE)  ## catches LinAlg error
                # try:
                #     mps.left_compress_site(x, max_bond=None, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                # except (np.linalg.LinAlgError, ValueError):
                #     mps.left_compress_site(x, method='eig', max_bond=None, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                if False:  # scale:
                    site_norm_to_exponent(x + 1)

            for x in range(cur_orthog[1], i, -1):
                # mps.right_canonize_site(x)
                right_compress_site(mps, x, max_bond=None, cutoff=CUTOFF,
                                    cutoff_mode=CUTOFF_MODE)  ## catches LinAlg error
                # try:
                #     mps.right_compress_site(x, max_bond=None, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                # except (np.linalg.LinAlgError, ValueError):
                #     mps.right_compress_site(x, method='eig', max_bond=None, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                if False:  # scale:
                    site_norm_to_exponent(x - 1)

            if scale:
                site_norm_to_exponent(i)

    except (np.linalg.LinAlgError, ValueError):
        if np.isinf(mps.exponent):
            mps = None
        elif norm(mps_copy) < np.sqrt(CUTOFF):
            mps = None
        else:
            raise np.linalg.LinAlgError

    if bra is not None:
        if mps is None:
            bra.exponent = -np.inf
        else:
            old_bra = bra.copy()
            bra.site_ind_id = mps.site_ind_id
            for i in range(mps.L):
                bra[i].modify(data=mps[i].data.conj(), inds=mps[i].inds)
            bra.exponent = mps.exponent
            match_inner_inds(bra, old_bra, inplace=True)
            bra.site_ind_id = old_bra.site_ind_id

    if mps is not None and mps.exponent < -30:
        mps = None
        bra = None

    return mps


def compress(mps: Union[MPSType, MPOType], scale=True, verbose=False, canonize=True, compress_opts: dict = None,
             bra: Optional[Union[MPSType, MPOType]] = None, norm_cutoff=None, ref_norm=1.0, renorm: bool = None):
    """Canonicalize and compress an MPS/MPO, absorbing the norm into the exponent.

    Canonizes (unless ``canonize=False``) and then compresses site by site in
    the requested ``form``, normalizing the boundary site and accumulating its
    norm into ``mps.exponent`` (via the nested ``site_norm_to_exponent``). An
    inplace operation. If ``do_midpt`` is set in ``compress_opts`` the work is
    delegated to :func:`compress_midpt`. May return ``None`` when the norm falls
    below ``norm_cutoff * ref_norm`` or the state is numerically zero. A
    length-1 network is returned unchanged.

    Parameters
    ----------
    mps : MPSType or MPOType
        The network to compress.
    scale : bool
        Absorb the boundary-site norm into the exponent if True.
    verbose : bool
        Print compression diagnostics (singular values, errors) if True.
    canonize : bool
        Canonicalize before compressing if True.
    compress_opts : dict, optional
        Compression options (copied internally). Reads/sets ``'cutoff'``
        (default CUTOFF), ``'cutoff_mode'`` (default CUTOFF_MODE), ``'form'``
        (default ``'right'``), ``'renorm'``, ``'max_bond'``, ``'adapt'``, and
        the popped keys ``'norm_cutoff'``, ``'ref_norm'``, ``'do_midpt'`` and
        ``'midpt'``.
    bra : MPSType or MPOType, optional
        If given, its tensors are overwritten with the conjugate of the
        compressed ``mps`` and its exponent synced.
    norm_cutoff : float, optional
        Relative norm threshold below which the result is discarded (returns
        None); enables the norm check when not None.
    ref_norm : float
        Reference norm against which ``norm_cutoff`` is measured. Default 1.0.
    renorm : bool, optional
        Renormalization flag passed through to the underlying compression.

    Returns
    -------
    MPSType or MPOType or None
        The compressed network, or ``None`` if it falls below tolerance.
    """
    if mps.L == 1:
        return mps

    if verbose:
        mps_copy = mps.copy()

    check_norm = (norm_cutoff is not None)

    # norm_cutoff = 0.0
    # if norm_cutoff is None:
    #     norm_cutoff = np.sqrt(CUTOFF)  # np.sqrt(CUTOFF) if CUTOFF_MODE == 'rsum2' else CUTOFF

    if compress_opts is None:
        compress_opts = {}
    else:
        compress_opts = compress_opts.copy()
    compress_opts.setdefault('cutoff', CUTOFF)
    compress_opts.setdefault('cutoff_mode', CUTOFF_MODE)
    compress_opts.setdefault('form', 'right')
    compress_opts.setdefault('renorm', renorm)
    if norm_cutoff is None:
        norm_cutoff = compress_opts.pop('norm_cutoff', np.sqrt(CUTOFF))
    else:
        compress_opts.pop('norm_cutoff', np.sqrt(CUTOFF))
    ref_norm = compress_opts.pop('ref_norm', ref_norm)
    # print('cutoff', compress_opts, 'norm cutoff', norm_cutoff)

    # if compress_opts.get('max_bond',None) is None:
    #     canonize = False

    # if mps is None or np.isneginf(mps.exponent):   return None
    if mps is None or (check_norm and mps.exponent < np.log10(norm_cutoff * ref_norm)):
        # pdb.set_trace()
        return None

    # print('helper compress norm', mps.exponent, norm(mps), ref_norm, norm(mps)/ref_norm)
    # if np.isnan(norm(mps)): #  or norm(mps) / ref_norm < norm_cutoff:
    #     return None

    if verbose:   print('compress', compress_opts)

    do_midpt = compress_opts.pop('do_midpt', False)
    if do_midpt:
        midpt = compress_opts.pop('midpt', mps.L // 2)
        compress_midpt(mps, midpt, scale=scale, verbose=verbose, compress_opts=compress_opts)
        return mps

    try:
        if False:  # not scale:
            if canonize:
                orthog_int = mps.L - 1 if compress_opts['form'] == 'right' else 0
                mps.canonize_func(orthog_int, )
            mps.compress(**compress_opts)
            return mps
        else:
            form = compress_opts.pop('form', 'right')

            ## helper function to normalize site i, shift weight into mps.exponent
            def site_norm_to_exponent(ix):
                """Normalize site ``ix`` of ``mps`` and add its norm to the exponent.

                Parameters
                ----------
                ix : int
                    Site index to normalize; raises ``LinAlgError`` (and sets the
                    exponent to ``-inf``) if the tensor norm is zero.

                Returns
                -------
                None
                """
                tag = mps.site_tag_id.format(ix)
                tens = mps[tag]
                if tens.norm() == 0.0:
                    mps.exponent = -np.inf
                    raise np.linalg.LinAlgError
                mps.strip_exponent(tens)  # normalizes tensor, adds norm to exponent

            if form == 'left':
                ## canonicalize
                if canonize:
                    for i in range(mps.L - 1, 0, -1):
                        mps.right_canonize_site(i)
                        # right_compress_site(mps, i, cutoff=0.0)
                        # site_norm_to_exponent(i - 1)
                    if scale:  site_norm_to_exponent(0)

                # print('right canonized? check orthog')
                # check_right_orthog(mps)

                ## compress
                for i in range(mps.L - 1):
                    # mps.left_compress_site(i, **compress_opts)
                    left_compress_site(mps, i, **compress_opts)

                    # ## check for nans
                    # it = 0
                    # while np.any(np.isnan(mps[i].data)):
                    #     print('right compress yielded nan')
                    #     mps.left_compress_site(i, method='eig', **compress_opts)
                    #     it += 1
                    #     if it > 10:
                    #         raise ValueError('left compress yielded nan')

                    # site_norm_to_exponent(i+1)
                if scale:   site_norm_to_exponent(mps.L - 1)

            elif form == 'right' or form is None:
                ## canonicalize
                if canonize:
                    for i in range(mps.L - 1):
                        mps.left_canonize_site(i)
                        # left_compress_site(mps, i, cutoff=0.0)    ## this specifically caused slightly larger errors
                        # site_norm_to_exponent(i+1)
                    if scale:  site_norm_to_exponent(mps.L - 1)

                # print('left canonized?', check_left_orthog(mps))

                ## compress
                for i in range(mps.L - 1, 0, -1):
                    if verbose:
                        print('before orthog i', i, mps.calc_current_orthog_center())
                        print('before', i, mps.singular_values(i, cur_orthog=i))

                    # mps.right_compress_site(i, **compress_opts)
                    right_compress_site(mps, i, **compress_opts)

                    # ## check for nans
                    # it = 0
                    # while np.any( np.isnan(mps[i].data) ):
                    #     print('right compress yielded nan', mps.exponent)
                    #     mps.right_compress_site(i, method='eig', **compress_opts)
                    #     it += 1
                    #     if it > 10:
                    #         raise ValueError('right compress yielded nan')

                    if verbose:
                        svals = mps.singular_values(i, cur_orthog=i - 1)
                        print('after', i, svals, np.linalg.norm(svals))
                        mps.right_canonize_site(i)
                        print('after orthog i', i, mps.calc_current_orthog_center())

                    # site_norm_to_exponent(i-1)
                if scale:  site_norm_to_exponent(0)

    except np.linalg.LinAlgError or ZeroDivisionError:
        print('compress error', i, mps.exponent, mps.norm())
        if np.isinf(mps.exponent):
            mps = None
        elif mps.singular_values(i) == [0]:
            mps = None
        else:
            raise np.linalg.LinAlgError

    if verbose:
        if mps is not None:
            print('mps exp', mps.exponent)
        else:
            print('mps None')

    if mps is None or (check_norm and mps.exponent < np.log10(norm_cutoff * ref_norm)):
        # pdb.set_trace()
        mps = None

    if check_norm and abs(norm(mps)) / ref_norm < norm_cutoff:
        # pdb.set_trace()
        mps = None

    if bra is not None:
        if mps is None:
            bra.exponent = -np.inf
        else:
            for i in range(mps.L):
                bra[i].modify(data=mps[i].data.conj())
            bra.exponent = mps.exponent

    if verbose:  # verbose:
        if mps is not None:
            print('compression error', distance(mps_copy, mps) / norm(mps_copy))
        else:
            print('mps norm is 0?', mps_copy.norm())

    return mps


def conservative_compress(mps: MPSType, bases: Sequence['qtn.MatrixProductState'] = None,
                          proj_vals: Optional[list[Numeric]] = None, compress_opts=None, canonize=True, verbose=False):
    """Compress an MPS while preserving its projections onto given basis MPS.

    Projects out the ``bases`` components from ``mps`` (measuring or using
    supplied projection values), compresses the orthogonal remainder with a
    reduced ``max_bond`` (max_bond - 2 to leave room), then adds the basis
    contributions back so the chosen projections are retained.

    Parameters
    ----------
    mps : MPSType
        The MPS to compress.
    bases : Sequence[qtn.MatrixProductState], optional
        Basis MPS whose overlaps with ``mps`` are preserved.
    proj_vals : list[Numeric], optional
        Projection coefficients to impose; if None, they are measured from
        ``mps`` via overlaps.
    compress_opts : dict, optional
        Compression options; ``'max_bond'`` is reduced by 2 for the remainder.
    canonize : bool
        Accepted for signature compatibility (not used directly here).
    verbose : bool
        Print orthogonality/error diagnostics if True.

    Returns
    -------
    MPSType
        The compressed MPS with preserved projections.
    """
    # L = mps.L
    # iden = qtn.MPO_identity(mps.L)

    # normed_bases = []
    # for ix, b in enumerate(bases):
    #     bnorm = norm(b)
    #     if proj_vals is not None:
    #         proj_vals[ix] = proj_vals[ix] / bnorm
    #     b = scalar_multiply(b, 1./bnorm)
    #     normed_bases += [b]
    #
    # ## get projectors of normalized and orthogonalized bases
    # sum_bases = add_MPS_list(bases, inplace=False, do_final_update=False)

    if verbose:
        print('compress opts', compress_opts)
    compress_opts_mod = {k: v for k, v in compress_opts.items()}
    max_bond = compress_opts_mod.get('max_bond', None)
    if max_bond is not None:
        compress_opts_mod['max_bond'] = compress_opts_mod['max_bond'] - 2

    meas_proj_vals = []
    remainder_mps = mps
    for b in bases:
        if verbose:
            print('b norm', ovlp(b, b))
        meas_proj_val = ovlp(remainder_mps, b)
        meas_proj_vals += [meas_proj_val]
        remainder_mps = add_MPS(remainder_mps, scalar_multiply(b, -meas_proj_val))
        if verbose:
            print('orthogonal?', ovlp(b, remainder_mps))
    compress(remainder_mps, compress_opts=compress_opts_mod)
    if verbose:
        print('remainder max bond', remainder_mps.max_bond())

    ## check
    for b in bases:
        if verbose:
            print('b max bond', b.max_bond())
        if verbose:
            print('still orthogonal?', ovlp(b, remainder_mps))

    mps_list = [remainder_mps]
    for ix, b in enumerate(bases):
        coeff = meas_proj_vals[ix] if proj_vals is None else proj_vals[ix]
        mps_list += [scalar_multiply(b, coeff)]

    # out = add_MPS_list(mps_list, inplace=True, compress_opts=compress_opts)
    out = add_MPS(mps_list[0], mps_list[1])

    ## check
    for ix, b in enumerate(bases):
        coeff = meas_proj_vals[ix] if proj_vals is None else proj_vals[ix]
        if verbose:
            print('proj val diff', ovlp(b, out), ovlp(b, out) - coeff)

    if verbose:
        print('compress err', distance(mps, out))
        print('total max bond', out.max_bond())

    return out


def compress_rdm(mps: MPSType, scale=True, verbose=False, direction=1, compress_opts: dict = None,
                 bra: Optional[Union[MPSType, MPOType]] = None,
                 open_end=False, back_compress=True, right_env=None, left_env=None):
    """Compress an MPS using the reduced-density-matrix (RDM) scheme (not in place).

    Builds environments from one end, diagonalizes the per-site reduced density
    matrix to obtain truncated isometries (largest-eigenvalue eigenvectors,
    bounded by ``max_bond`` / ``cutoff``), and strings them into a new MPS. With
    ``back_compress`` the direction is reversed and a final :func:`compress`
    pass sets the canonical form. MPO inputs are delegated to :func:`compress`.
    The norm is absorbed into ``mps.exponent`` when ``scale`` is True.

    Parameters
    ----------
    mps : MPSType
        The MPS (or MPO) to compress.
    scale : bool
        Absorb the norm into the exponent if True.
    verbose : bool
        Print environment/RDM diagnostics if True.
    direction : int
        ``> 0``: build right-to-left; ``< 0``: left-to-right. Negated internally
        when ``back_compress`` is True.
    compress_opts : dict, optional
        Reads ``'cutoff'`` (default CUTOFF), ``'cutoff_mode'`` (default
        CUTOFF_MODE), ``'max_bond'``; ``'form'`` is set from ``direction``.
    bra : MPSType or MPOType, optional
        Bra-side network; internally regenerated as the conjugate of ``mps``.
    open_end : bool
        If True, leave the terminal site open and also return the carry tensor.
    back_compress : bool
        Reverse the sweep and perform a final compression to the target form.
    right_env : qtn.Tensor, optional
        Precomputed right environment (used when ``direction > 0``).
    left_env : qtn.Tensor, optional
        Precomputed left environment (used when ``direction < 0``).

    Returns
    -------
    MPSType or tuple[MPSType, qtn.Tensor]
        The compressed MPS, plus the carry tensor when ``open_end`` is True.
    """
    if compress_opts is None:
        compress_opts = {}
    else:
        compress_opts = compress_opts.copy()
    compress_opts.setdefault('cutoff', CUTOFF)
    compress_opts.setdefault('cutoff_mode', CUTOFF_MODE)
    compress_opts['form'] = 'left' if direction > 0 else 'right'  ## final canonical form (after back compress

    if isinstance(mps, qtn.MatrixProductOperator):
        return compress_func(mps, scale=scale, verbose=verbose, canonize=True,
                             compress_opts=compress_opts, bra=bra)

    mps = mps.mangle_inner_()
    L = mps.L
    cutoff = compress_opts.get('cutoff', CUTOFF)
    max_bond = compress_opts.get('max_bond', None)

    if verbose:
        print('mps', mps.max_bond())
        print('apply rdm bra mps2', bra)

    if back_compress:
        direction *= -1

    if True:  # bra is None:
        bra = mps.conj(mangle_inner=False)
        for tens in bra:
            tens.reindex({ind: ind + '_' for ind in tens.inds}, inplace=True)
        bra._site_ind_id = bra.site_ind_id + '_'
        bra.site_ind_id = mps.site_ind_id

    # qtn.tensor_network_align(mps2, mpo1, bra_mpo1, bra_mps2, inplace=True)
    # mpo1.upper_ind_id = mpo1.upper_ind_id + '_x_'
    # bra_mpo1.lower_ind_id = mpo1.upper_ind_id
    x_ind_id = mps.site_ind_id

    idx0 = 0 if direction > 0 else L - 1
    idx1 = 0 if direction < 0 else L - 1

    ## ancillas at "far" end (away from end at which we start building envs)
    mps_anc = [ind for ind in mps[idx0].inds if ind not in
               [mps.site_ind(idx0), mps.bond(idx0, idx0 + 1 * np.sign(direction))]]

    #### build envs
    env = right_env if direction > 0 else left_env
    envs = {idx1: env}

    ## dir = -1:  (L) 0 to L-1, dir = 1:   (R) L-1 to 0
    for i in range(idx1, idx0, -1 * np.sign(direction)):
        next_i = i - np.sign(direction)
        contract_tens_list = [mps[i], bra[i]]
        if env is not None:
            contract_tens_list += [env]

        env = qtn.TensorNetwork(contract_tens_list)
        env = env.contract()
        envs[next_i] = env
        if verbose:
            print('contract tens list', contract_tens_list)
            print('env', i, env.shape)

    ## get unitaries
    Us = {}
    C_tens = None

    ## dir = 1:   (L) 0 to L-1  --> left canon
    ## dir = -1:  (R) L-1 to 0  --> right canon
    for i in range(idx0, idx1 + np.sign(direction), np.sign(direction)):
        next_i = i + np.sign(direction)
        prev_i = i - np.sign(direction)

        ### last site.
        ### if not True, tens1 is rdm of site[idx1] + appropriate env (hack for comb geometry)
        if i == idx1 and not open_end:
            contract_tens_list = [mps[idx1]]
            if C_tens is not None:
                contract_tens_list += [C_tens]
            tens1 = qtn.tensor_contract(*contract_tens_list)
            tens1.drop_tags()
            tens1.add_tag(mps.site_tag(idx1))
            Us[idx1] = tens1
            break

        bra_tens = bra[i].reindex({x_ind_id.format(i): x_ind_id.format(i) + '_o_'})
        contract_tens_list = [mps[i], bra_tens]
        env = envs.get(i, None)
        if env is not None:
            contract_tens_list += [env]
        if C_tens is not None:
            C_tens_conj = C_tens.conj(inplace=False)
            C_tens_conj.reindex({ind: ind + '_' for ind in C_tens.inds}, inplace=True)
            contract_tens_list += [C_tens, C_tens_conj]

        # rdm = qtn.tensor_contract( *contract_tens_list )
        rdm = qtn.TensorNetwork([contract_tens_list])
        rdm = rdm.contract()

        if verbose:  print('rdm', i, rdm.shape, rdm.inds)

        ket_inds = [x_ind_id.format(i)]
        if i == idx0:
            ket_inds += mps_anc
        if C_tens is not None:
            ket_inds += [Us[prev_i].inds[-1]]  ## ind from evecs_tens

        bra_inds = [x_ind_id.format(i) + '_o_'] + [ind + '_' for ind in ket_inds[1:]]
        rdm.transpose(*ket_inds, *bra_inds, inplace=True)
        sq_shape = rdm.shape[:len(ket_inds)]

        ### quimb version
        rdm_mat = rdm.data.reshape(np.prod(sq_shape), -1)
        if max_bond is None:
            evals, evecs = quimb.eigh(rdm_mat, k=-1,
                                      sort=True, return_vecs=True, )
        else:
            evals, evecs = quimb.eigh(rdm_mat,
                                      k=(max_bond if max_bond is not None else -1), tol=cutoff, which='LM',
                                      sort=True, return_vecs=True, )
        ## evals assorted in ascending order; targetting largest magnitude eigvals
        evecs = evecs[:, ::-1].conj()  ## qarray object

        ### numpy version
        # rdm_mat = rdm.data.reshape(np.prod(sq_shape), -1)
        # # if verbose:   print('rdm mat is H', np.linalg.norm(rdm_mat - rdm_mat.T.conj()))
        # evals, evecs = np.linalg.eigh(rdm_mat)
        #
        # if verbose:  print('cutoff', cutoff, 'max_bond', max_bond)
        # sort_inds = np.argsort(evals)[::-1]
        # if max_bond is not None:
        #     sort_inds = sort_inds[:max_bond]
        #
        # evals = evals[sort_inds]
        # if evals[-1]/evals[0] < cutoff:
        #     cut_ind = np.argmax(evals/evals[0] < cutoff)
        #     sort_inds = sort_inds[:cut_ind]
        #
        # evecs = evecs[:,sort_inds].conj()     ## input inds, sort_inds (horizontal bond)
        #####

        h_ind = f'h_{i}'  # if i != idx1 else last_ind
        evecs_tens = qtn.Tensor(data=evecs.reshape(*sq_shape, -1), inds=ket_inds + [h_ind],
                                tags=mps.site_tag(i))
        Us[i] = evecs_tens.conj(inplace=False)

        contract_tens_list = [mps[i], evecs_tens]
        if C_tens is not None:
            contract_tens_list += [C_tens]
        C_tens = qtn.tensor_contract(*contract_tens_list)

    ## string Us together to make new MPS
    new_mpx = qtn.TensorNetwork([Us[i] for i in range(L)])
    new_mpx.view_like(mps, inplace=True, site_ind_id=x_ind_id)
    new_mpx.site_ind_id = mps.site_ind_id
    new_mpx.exponent = mps.exponent

    if back_compress:
        compress_opts['form'] = 'left' if idx1 < idx0 else 'right'
        if verbose:
            print('back compress')
            print('check orthog', check_orthog(new_mpx), new_mpx.exponent, compress_opts['form'])
        ## recall that we switched direction at beginnong of function
        new_mpx = compress_func(new_mpx, scale=True, canonize=False, compress_opts=compress_opts)

    if open_end:
        return new_mpx, C_tens
    else:
        return new_mpx


compress_func = compress


def compress_midpt(mps, mid_pt, scale=True, verbose=False, timeit=False, mid_compress_opts=None, compress_opts=None):
    """Compress an MPS by canonizing toward a midpoint and sweeping outward (in place).

    Left-canonizes the left half and right-canonizes the right half toward
    ``mid_pt``, compresses the central bond extracting singular values, inserts
    ``diag(svals)``/``diag(1/svals)`` operators, then compresses each half
    outward and re-contracts the singular-value operators. Norms are absorbed
    into ``mps.exponent`` when ``scale`` is True. Returns ``None`` when the
    exponent drops below ``-30`` or the state is numerically zero.

    Parameters
    ----------
    mps : MPSType
        The MPS to compress.
    mid_pt : int
        The midpoint site about which the two-sided sweep is organized.
    scale : bool
        Absorb site norms into the exponent if True.
    verbose : bool
        Print shape/orthogonality diagnostics if True.
    timeit : bool
        If True, also collect and return per-step canonize and compress timings.
    mid_compress_opts : dict, optional
        Compression options for the central-bond split; defaults to
        ``compress_opts``.
    compress_opts : dict, optional
        Compression options for the outward sweeps (the ``'form'`` key is
        popped and ignored).

    Returns
    -------
    MPSType or tuple or None
        The compressed MPS, or ``(mps, canon_times, comp_times)`` when
        ``timeit`` is True, or ``None`` if it falls below tolerance.
    """
    # print('compress midpt')
    if mps is None or np.isneginf(mps.exponent):   return None

    if compress_opts is None:
        compress_opts = {}
    else:
        compress_opts = compress_opts.copy()

    if mid_compress_opts is None:
        mid_compress_opts = compress_opts
    else:
        mid_compress_opts = mid_compress_opts.copy()

    if mps.exponent < -30:   return None

    if verbose:   print('compress', compress_opts, mid_compress_opts)
    if timeit:    canon_times, comp_times = [], []

    try:

        form = compress_opts.pop('form', 'right')  ## toss this

        ## helper function to normalize site i, shift weight into mps.exponent
        def site_norm_to_exponent(i):
            """Normalize site ``i`` of ``mps`` and add its norm to the exponent.

            Parameters
            ----------
            i : int
                Site index to normalize.

            Returns
            -------
            None
            """
            tag = mps.site_tag(i)
            tens = mps[tag]
            mps.strip_exponent(tens)  # normalizes tensor, adds norm to exponent

        ## canonicalize
        ## canonicalize LHS to left canonical
        for i in range(mid_pt):
            time1 = time.time()
            mps.left_canonize_site(i)
            if timeit:   canon_times += [time.time() - time1]
            if scale:    site_norm_to_exponent(i + 1)

        ## canonicalize RHS to right canonical
        for i in range(mps.L - 1, mid_pt, -1):
            time1 = time.time()
            mps.right_canonize_site(i)
            if timeit:   canon_times += [time.time() - time1]
            if scale:    site_norm_to_exponent(i - 1)

        # print('check orthog i',mid_pt,mps.calc_current_orthog_center())

        ## compress mid point --> M = midpt -> M = midpt - 1
        svals = right_compress_site(mps, mid_pt, return_svals=True, **mid_compress_opts)
        # print('svals',svals, mid_pt, np.linalg.norm(svals))
        # print(mps.exponent)
        # print([mps[i].norm() for i in range(mps.L)])
        if verbose:
            print('mid compress shape', mid_pt, mps[mid_pt].shape, mid_compress_opts)

        ## compression scheme 1
        mps.insert_operator(np.diag(1. / svals), mps.site_tag(mid_pt - 1), mps.site_tag(mid_pt),
                            tags='svals_inv', inplace=True)
        mps.insert_operator(np.diag(svals), 'svals_inv', mps.site_tag(mid_pt),
                            tags='svals', inplace=True)

        ## compress left hand side -> right canonical

        for i in range(mid_pt - 1, 0, -1):
            if verbose:
                print('after', i, mps[i].shape)
                print('before orthog i', i, check_orthog(mps))
                # print('before', i, mps.singular_values(i, cur_orthog=i))

            time1 = time.time()
            right_compress_site(mps, i, **compress_opts)
            if timeit:   comp_times += [time.time() - time1]

            if verbose:
                print('after', i, mps[i].shape)
                print('after', i, mps.singular_values(i, cur_orthog=i - 1))
                mps.right_canonize_site(i)
                print('after orthog i', i, check_orthog(mps))

        ## compress right hand side -> to the left canonical
        mps.contract_between('svals', mps.site_tag(mid_pt))

        for i in range(mid_pt, mps.L - 1):
            # mps.left_compress_site(i, **compress_opts)
            time1 = time.time()
            left_compress_site(mps, i, **compress_opts)
            if timeit:   comp_times += [time.time() - time1]

        mps.contract_between('svals_inv', mps.site_tag(mid_pt))
        mps[mid_pt].drop_tags(('svals', 'svals_inv'))


    except(np.linalg.LinAlgError, ValueError):
        print('error', i, mps.singular_values(i), mps.exponent)
        if np.isinf(mps.exponent):
            mps = None
        elif mps.singular_values(i)[0] <= 1e-12:
            mps = None
        else:
            raise np.linalg.LinAlgError

    # print('mps exp',mps.exponent)

    if timeit:
        return mps, canon_times, comp_times

    if mps.exponent < -30:   return None
    return mps


def left_compress_site(mps, i, return_svals=False, verbose=False, **compress_opts):
    """In-place left compression of site ``i`` into site ``i+1`` of an MPS.

    Splits site ``i`` (SVD) and absorbs the right factor into site ``i+1``. With
    ``adapt`` (or ``return_svals``) the bond is trimmed adaptively relative to
    the largest singular value; otherwise quimb's ``left_compress_site`` is used
    with eig and random-restart fallbacks on linear-algebra / zero-division
    errors and nan results.

    Parameters
    ----------
    mps : MPSType
        The MPS to modify.
    i : int
        Site index to compress (paired with ``i+1``).
    return_svals : bool
        If True, force adaptive splitting and return the singular values.
    verbose : bool
        Print diagnostics on zero-division fallback if True.
    **compress_opts
        Compression options. Reads ``'max_bond'`` (default -1), ``'cutoff'``
        (default CUTOFF), and the popped ``'adapt'`` / ``'adapt_cutoff'``
        (fraction below the largest singular value, default 0.1).

    Returns
    -------
    numpy.ndarray or None
        The singular values if ``return_svals`` is True, else None.
    """
    max_bond = compress_opts.get('max_bond', -1)
    cutoff = compress_opts.get('cutoff', CUTOFF)
    # renorm = compress_opts.get('renorm', 1)
    do_adapt = compress_opts.pop('adapt', False)
    adapt_cutoff = compress_opts.pop('adapt_cutoff', 0.1)  # cutoff fraction below DMAX singular val

    if return_svals and not do_adapt:
        do_adapt = True
        adapt_cutoff = 0.0

    if do_adapt:  # and max_bond is not None:
        if max_bond is None:
            max_bond = -1

        T1, T2 = mps[i], mps[i + 1]
        rix, lix = T1.filter_bonds(T2)
        U, svals, VT = T1.split(lix, get='arrays', absorb=None, cutoff_mode=CUTOFF_MODE,
                                cutoff=compress_opts.get('cutoff', CUTOFF))

        try:
            # U = U[...,:max_bond]
            # svals = svals[:max_bond]
            # VT = VT[:max_bond,...]

            if max_bond > len(svals):
                svals_ref = svals[-1] / svals[0]
            else:
                svals_ref = svals[max_bond] / svals[0]
            U, svals, VT = qtn.decomp._trim_and_renorm_SVD(U, svals, VT,
                                                           (svals_ref * adapt_cutoff) ** 2, 4,  # cutoff, 'rsum2'
                                                           max_bond, None, 1)  # max_bond, absorb, renormalize
            # # keep_ind = np.argmin(svals >= svals_ref*adapt_cutoff)
            # # if keep_ind == 0:   raise(IndexError)  # keep all singular values
            # # U = np.take(U, range(keep_ind), axis=-1)
            # # svals = svals[:keep_ind]
            # # VT = VT[:keep_ind]
        except IndexError:
            pass

        VT = (svals * (VT.T)).T
        TL = qtn.Tensor(data=U, inds=(*lix, 'tmp'))
        TR = qtn.Tensor(data=VT, inds=('tmp', *rix))
        TR = TR.contract(T2)

        TL.transpose_like(T1, inplace=True)
        TR.transpose_like(T2, inplace=True)
        T1.modify(data=TL.data)
        T2.modify(data=TR.data)

        if return_svals:
            return svals

    else:
        site_i1: qtn.Tensor = mps[i].copy()
        site_i2: qtn.Tensor = mps[i + 1].copy()
        try:
            if MINBOND is None:
                mps.left_compress_site(i, **compress_opts)
                ## method='eig' seemed to yield larger error? though perhaps wrt a different method
            else:
                _, left_inds = mps[i].filter_bonds(mps[i+1])
                q, r = tensor_svd(mps[i].copy(), left_inds, absorb='right', max_bond=max_bond, cutoff=cutoff)
                q.transpose_like(mps[i], inplace=True)
                mps[i].modify(data=q.data)
                r = qtn.tensor_contract(r, mps[i+1])
                r.transpose_like(mps[i+1], inplace=True)
                mps[i+1].modify(data=r.data)
                # print('left compress minbond', MINBOND)

                # bond = next(iter(q.bonds(r)))
                # bond_size = q.ind_size(bond)
                # if bond_size < MINBOND:
                #     print('(L) bond size', bond_size, bond)
                #     # pdb.set_trace()

            if np.any(np.isnan(mps[i].data)):
                raise np.linalg.LinAlgError(f'left compress site {i} yielded nan')
        except (ValueError, np.linalg.LinAlgError):
            # site_i.transpose_like(mps[i], inplace=True)
            mps[i].modify(data=site_i1.data, inds=site_i1.inds)
            mps[i + 1].modify(data=site_i2.data, inds=site_i2.inds)
            mps.left_compress_site(i, method='eig', **compress_opts)
            if np.any(np.isnan(mps[i].data)):
                raise np.linalg.LinAlgError(f'left compress (eig) site {i} yielded nan')
        except ZeroDivisionError:
            print('left canon zero division error')
            # tens = site_i1.copy()
            # tens.modify(data=np.random.random(tens.shape))
            mps[i].modify(data=np.random.random(site_i1.shape))
            mps.left_compress_site(i, **compress_opts)
            mps[i + 1].modify(apply=lambda x: x * 0)
            # return None


def right_compress_site(mps, i, return_svals=False, verbose=False, **compress_opts):
    """In-place right compression of site ``i`` into site ``i-1`` of an MPS.

    Splits site ``i`` (SVD) and absorbs the left factor into site ``i-1``. With
    ``adapt`` (or ``return_svals``) the bond is trimmed adaptively relative to
    the largest singular value; otherwise quimb's ``right_compress_site`` is
    used with eig and random-restart fallbacks on linear-algebra /
    zero-division errors and nan results.

    Parameters
    ----------
    mps : MPSType
        The MPS to modify.
    i : int
        Site index to compress (paired with ``i-1``).
    return_svals : bool
        If True, force adaptive splitting and return the singular values.
    verbose : bool
        Print diagnostics on zero-division fallback if True.
    **compress_opts
        Compression options. Reads ``'max_bond'`` (default -1), ``'cutoff'``
        (default CUTOFF), and the popped ``'adapt'`` / ``'adapt_cutoff'``
        (fraction below the largest singular value, default 0.1).

    Returns
    -------
    numpy.ndarray or None
        The singular values if ``return_svals`` is True, else None.
    """
    max_bond = compress_opts.get('max_bond', -1)
    cutoff = compress_opts.get('cutoff', CUTOFF)
    # renorm = int(compress_opts.get('renorm', 1))
    do_adapt = compress_opts.pop('adapt', False)
    adapt_cutoff = compress_opts.pop('adapt_cutoff', 0.1)  # cutoff fraction below DMAX singular val

    if return_svals and not do_adapt:
        do_adapt = True
        adapt_cutoff = 0.0

    if do_adapt:  # and max_bond is not None:
        if max_bond is None:
            max_bond = -1

        T1, T2 = mps[i - 1], mps[i]
        # print('T1',T1, 'T2',T2)
        lix, rix = T2.filter_bonds(T1)
        U, svals, VT = T2.split(lix, get='arrays', absorb=None, cutoff_mode=CUTOFF_MODE, right_inds=rix,
                                cutoff=compress_opts.get('cutoff', CUTOFF))

        try:
            # U = U[...,:max_bond]
            # svals = svals[:max_bond]
            # VT = VT[:max_bond,...]

            if max_bond > len(svals):
                svals_ref = svals[-1] / svals[0]
            else:
                svals_ref = svals[max_bond] / svals[0]
            U, svals, VT = qtn.decomp._trim_and_renorm_SVD(U, svals, VT,
                                                           (svals_ref * adapt_cutoff) ** 2, 4,  # cutoff, 'rsum2'
                                                           max_bond, None, 1)  # max_bond, absorb, renormalize
            # print(svals_ref,len(svals))
            # # keep_ind = np.argmin(svals >= svals_ref*adapt_cutoff)
            # # # print('r keep ind',max_bond,len(svals),keep_ind, svals_ref, np.min(svals))
            # # if keep_ind == 0:   raise(IndexError)  # keep all singular values
            # # U = np.take(U, range(keep_ind), axis=-1)
            # # svals = svals[:keep_ind]
            # # VT = VT[:keep_ind]
        except IndexError:  ## max_bond > len(svals)
            pass

        U = U * svals
        TL = qtn.Tensor(data=U, inds=(*lix, 'tmp'))
        TR = qtn.Tensor(data=VT, inds=('tmp', *rix))
        TL = TL.contract(T1)

        TL.transpose_like(T1, inplace=True)
        TR.transpose_like(T2, inplace=True)
        T1.modify(data=TL.data)
        T2.modify(data=TR.data)

        if return_svals:
            return svals

    else:
        site_i1: qtn.Tensor = mps[i].copy()
        site_i2: qtn.Tensor = mps[i - 1].copy()
        try:
            if MINBOND is None:
                mps.right_compress_site(i, **compress_opts)
            else:
                _, right_inds = mps[i].filter_bonds(mps[i - 1])
                q, r = tensor_svd(mps[i].copy(), right_inds, absorb='right', max_bond=max_bond, cutoff=cutoff)

                # bond = next(iter(q.bonds(r)))
                # bond_size = q.ind_size(bond)
                # if bond_size < MINBOND:
                #     print('bond size', bond_size, bond)
                #     # pdb.set_trace()

                q.transpose_like(mps[i], inplace=True)
                mps[i].modify(data=q.data)
                r = qtn.tensor_contract(r, mps[i-1])
                r.transpose_like(mps[i-1], inplace=True)
                mps[i-1].modify(data=r.data)

            # print('bond size', i, i-1, mps.bond_size(i,i-1), 'cutoff', cutoff)

            if np.any(np.isnan(mps[i].data)):
                raise np.linalg.LinAlgError(f'right compress site {i} yielded nan')
        except (ValueError, np.linalg.LinAlgError):
            # site_i.transpose_like(mps[i], inplace=True)
            mps[i].modify(data=site_i1.data, inds=site_i1.inds)
            mps[i - 1].modify(data=site_i2.data, inds=site_i2.inds)
            mps.right_compress_site(i, method='eig', **compress_opts)
            if np.any(np.isnan(mps[i].data)):
                raise np.linalg.LinAlgError(f'right compress (eig) site {i} yielded nan')
        except ZeroDivisionError:
            print('right canon zero division error')
            # tens = site_i1.copy()
            # tens.modify(data=np.random.random(tens.shape))
            mps[i].modify(data=np.random.random(site_i1.shape))
            mps.right_compress_site(i, **compress_opts)
            mps[i - 1].modify(apply=lambda x: x * 0)
            # return None


def canonize_tens_list(*tens, inplace=False, full_matrices=False):
    """Left-canonicalize a free list of tensors from left to right.

    With ``full_matrices`` a complete QR is performed at each step (preserving
    the full column space); otherwise the work is delegated to
    :func:`compress_tens_list` with an SVD at module ``CUTOFF``/``CUTOFF_MODE``.

    Parameters
    ----------
    *tens : qtn.Tensor
        The ordered tensors to canonize (chained by shared bonds).
    inplace : bool
        Modify the given tensors in place if True, else operate on copies.
    full_matrices : bool
        Use a full (complete) QR rather than an SVD-based compression.

    Returns
    -------
    tuple or list of qtn.Tensor
        The canonized tensors.
    """
    if full_matrices:
        tens = tens if inplace else [t.copy() for t in tens]
        for i in range(len(tens) - 1):
            shared, left = tens[i].filter_bonds(tens[i + 1])
            remaining = [i for i in tens[i + 1].inds if i not in shared]
            tens[i].transpose(*left, *shared, inplace=True)
            tens[i + 1].transpose(*shared, *remaining, inplace=True)
            shape_l, shape_r = tens[i].shape[:-1], tens[i].shape[-1]
            q, r = np.linalg.qr(tens[i].data.reshape(-1, shape_r), mode='complete')
            tens[i].modify(data=q.reshape(*shape_l, -1))
            tens[i + 1].modify(apply=lambda data: np.tensordot(r, data, axes=[-1, 0]))
        return tens
    else:
        # return compress_tens_list(*tens, inplace=inplace) #, compress_opts={'method':'qr'})
        return compress_tens_list(*tens, inplace=inplace, compress_opts={'method': 'svd', 'cutoff': CUTOFF,
                                                                         'cutoff_mode': CUTOFF_MODE})


def compress_tens_list(*tens, inplace=False, compress_opts=None, two_site=False, verbose=False):
    """Compress a free list of tensors from left to right (assumed canonical).

    At each adjacent pair, splits the left tensor (absorbing singular values to
    the right) and contracts the right factor into the next tensor; with
    ``two_site`` the pair is contracted before splitting. Falls back to the eig
    method on failure or nan results.

    Parameters
    ----------
    *tens : qtn.Tensor
        The ordered tensors to compress (assumed already canonicalized).
    inplace : bool
        Modify the given tensors in place if True, else operate on copies.
    compress_opts : dict, optional
        Split options (copied internally); ``'absorb'`` is forced to ``'right'``
        and ``'cutoff'``/``'cutoff_mode'`` default to the module constants.
    two_site : bool
        Contract each adjacent pair into one tensor before splitting if True.
    verbose : bool
        Print diagnostics on eig fallback / nan results if True.

    Returns
    -------
    tuple of qtn.Tensor
        The compressed tensors.
    """
    if compress_opts is None:
        compress_opts = {}
    else:
        compress_opts = compress_opts.copy()
    compress_opts['absorb'] = 'right'
    compress_opts.setdefault('cutoff', CUTOFF)
    compress_opts.setdefault('cutoff_mode', CUTOFF_MODE)
    norm_cutoff = compress_opts.pop('norm_cutoff', None)

    if not inplace:
        tens = tuple([t.copy() for t in tens])

    for i in range(len(tens) - 1):

        # if np.abs(tens[i].norm()) < 1.0e-15:
        #     tens[i].modify(data=None, inds=())

        if two_site:  ## contracting tensors together before splitting
            _, bonds_L = tens[i].filter_bonds(tens[i + 1])
            # _, bonds_R = tens[i+1].filter_bonds(tens[i])
            active_tens = tens[i].contract(tens[i + 1])
            TL, TR = active_tens.split(bonds_L, bond_ind=next(iter(_)), **compress_opts)
        else:
            bonds_R, bonds_L = tens[i].filter_bonds(tens[i + 1])

            # t_norm = tens[i].norm()
            # # print('t before', tens[i].norm(), t_norm)
            # tens[i].modify(apply=lambda data: data / t_norm)
            # print('t after', tens[i].norm())

            # DMAX = compress_opts.get('max_bond',None)
            # if DMAX is not None:
            #     u,s,vt = tens[i].split(bonds_L, get='arrays', absorb=None, max_bond=DMAX)
            #     print('s vals', s/np.max(s), np.max(s), np.any(np.isnan(u)), np.any(np.isnan(vt)))
            # print('tens[i]', np.any(np.isnan(tens[i].data)))

            ## rename tens[i] shared bond (bondsR), name split bond bondR
            # bR = next(iter(bonds_R))
            # tens[i].reindex({bR: bR + 'tmp'}, inplace=True)
            # tens[i+1].reindex({bR: bR + 'tmp'}, inplace=True)
            try:
                TL, TR = tens[i].split(bonds_L, **compress_opts)
            except (ZeroDivisionError):
                return tens     # just stop canonicalizing
            except ValueError:
                # TL, TR = tens[i].split(bonds_L, method='eig', **compress_opts)
                print('eig split; tens i norm', tens[i].norm())
                compress_opts.pop('method', None)
                TL, TR = qtn.tensor_split(tens[i], bonds_L, method='eig', **compress_opts)

            it, max_it = 0, 10
            while np.any(np.isnan(TL.data)) and it < max_it:
                print('TL is nan')
                compress_opts.pop('method', None)
                TL, TR = tens[i].split(bonds_L, method='eig', **compress_opts)
                it += 1

            # TR.modify(apply=lambda data: data * t_norm)
            TR = TR.contract(tens[i + 1])

        if TL.ndim == tens[i].ndim:
            TL.transpose_like(tens[i], inplace=True)
            TR.transpose_like(tens[i + 1], inplace=True)
            tens[i].modify(data=TL.data)
            tens[i + 1].modify(data=TR.data)
        else:
            tens[i].modify(data=TL.data, inds=TL.inds)
            tens[i + 1].modify(data=TR.data, inds=TR.inds)

    return tens


def scalar_multiply(mps: 'qtn.TensorNetwork', scalar_const: Numeric, inplace=False):
    """Multiply a network by a scalar, storing magnitude in the exponent.

    The magnitude ``log10(|scalar_const|)`` is added to ``mps.exponent`` while
    the sign (or complex phase) is spread over a single tensor. Plain
    arrays/floats are multiplied directly.

    Parameters
    ----------
    mps : qtn.TensorNetwork
        The network (or ndarray/float) to scale; ``None`` returns ``None``.
    scalar_const : Numeric
        The scalar multiplier (real or complex).
    inplace : bool
        Modify ``mps`` in place if True, else operate on a copy.

    Returns
    -------
    qtn.TensorNetwork or ndarray or float or None
        The scaled network/value.
    """
    if mps is None:
        return None

    val = np.abs(scalar_const)

    if isinstance(mps, (np.ndarray, float)):
        mps = mps * scalar_const
    else:
        mps = mps if inplace else mps.copy()

        if np.isreal(scalar_const):
            if np.real(scalar_const) < 0:
                sign = -1
            else:
                sign = 1
        else:
            # sign = np.exp(-1.j * np.angle(scalar_const))
            sign = np.exp(1.j * np.angle(scalar_const))
            # print('WARNING: CHANGED SIGN OF COMPLEX TERM IN SCALAR MULTIPLY')

        if val != 0:
            mps.exponent += np.log10(val)  # change total norm of MPS
        else:
            mps.exponent += -np.inf
        mps.multiply(sign, inplace=True, spread_over=1)

    return mps


def scalar_add(mps, scalar_val, inplace=False, compress=False, compress_opts: dict = None):
    """Add a constant scalar to every element represented by an MPS.

    Builds a rank-1 "all ones" network scaled by ``scalar_val`` (via
    :func:`sum_tensornetwork`) and adds it to ``mps`` with :func:`add_MPS`.

    Parameters
    ----------
    mps : MPSType
        The MPS to which the scalar is added.
    scalar_val : Numeric
        The constant to add to every element.
    inplace : bool
        Modify ``mps`` in place if True, else operate on a copy.
    compress : bool
        Compress the result of the addition if True.
    compress_opts : dict, optional
        Options forwarded to compression.

    Returns
    -------
    MPSType
        The MPS with the scalar added.
    """
    mps = mps if inplace else mps.copy()

    scalar_val_mps = sum_tensornetwork(like_tn=mps, scale=scalar_val)
    scalar_val_mps = scalar_val_mps.view_like(mps, inplace=True)  # view as MPS

    add_MPS(mps, scalar_val_mps, inplace=True, compress=compress, compress_opts=compress_opts)
    return mps


# def elemental_multiply(mps1,mps2,inplace=False,compress=False,compress_opts=None):
#     """ h(x) = f(x) * g(x)
#         elemental multiplication of the two MPS
#         for actual elemental_multiply, cannot do inplace. (inplace is for scalar mult)
#     """
#     if np.isscalar(mps1):
#         return scalar_multiply(mps2,mps1,inplace=inplace)
#     if np.isscalar(mps2):
#         return scalar_multiply(mps1,mps2,inplace=inplace)
#
#     if (mps1 is None or np.isneginf(mps1.exponent)) or (mps2 is None or np.isneginf(mps2.exponent)):
#         return None
#
#     # mps1.distribute_exponent()
#     # mps2.distribute_exponent()
#
#     bond_name = mps1.site_ind_id
#     tens_name = mps1.site_tag_id
#
#     if mps1.L == 1:
#         out = mps1.copy()
#         t1 = out.select_tensors(mps1.site_tag(0))[0]
#         t2 = mps2.select_tensors(mps2.site_tag(0))[0]
#         new_data = t1.data * t2.data * 10 ** mps2.exponent
#         t1.modify(data=new_data)
#         return out
#
#
#     # physical bond dimensions; physical bonds are all external dims
#     qs = mps1.shape
#
#     out = mps1.view_as(qtn.TensorNetwork,inplace=False)
#
#     bond_name_1 = bond_name+'[1]'
#     bond_name_2 = bond_name+'[2]'
#     for i in range(mps1.num_tensors):
#
#         d_ijk = qtn.tensor_core.COPY_tensor(qs[i],
#                                 (bond_name.format(i), bond_name_1.format(i), bond_name_2.format(i)),
#                                 tags=(f'd_ijk({i})',))
#         out.add_tensor( d_ijk )
#         out.contract( tags=(tens_name.format(i),f'd_ijk({i})'), inplace=True )
#
#     out.view_as(qtn.MatrixProductOperator, inplace=True, like=mps1,
#                                            upper_ind_id=bond_name_1, lower_ind_id=bond_name_2)
#     out.drop_tags([f'd_ijk({i})' for i in range(mps1.L)])
#
#     out = apply(out,mps2)
#     if compress:   compress_func(out, compress_opts=compress_opts)
#
#     return out


# def get_convolve_op(L,q=2,out_id=None,in1_id=None,in2_id=None,site_tag_id=None):
#     """ get operator that convolves two input vectors -> output
#         f(i) = sum_k g(k) h(i-k)
#         output is of the same length as the input
#         a 1D TN with 3 i/o legs
#     """
#     if q != 2:  raise NotImplementedError('convolution for q!=2 not implemented')
#     conv_op_tn = qtn.TensorNetwork([])
#
#     out_id = 'ind1,{}' if out_id is None else out_id
#     in1_id = 'ind1,{}' if in1_id is None else in1_id
#     in2_id = 'ind1,{}' if in2_id is None else in2_id
#     site_tag_id = 'site{}' if site_tag_id is None else site_tag_id
#
#     T0 = qtn.Tensor( np.array([[[[0.,1.],[1.,0.]],[[1.,0.],[0.,0.]]],
#                                [[[0.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]]),
#                      inds=(out_id.format(0),'b0',in1_id.format(0),in2_id.format(0)),
#                      tags=(site_tag_id.format(0),) )
#     conv_op_tn.add(T0)
#
#     for x in range(1,L-1):
#         T1 = qtn.Tensor( np.array([[[[[1.,0.],[0.,0.]],[[0.,0.],[0.,0.]]],
#                                     [[[0.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]],
#                                    [[[[0.,1.],[1.,0.]],[[1.,0.],[0.,0.]]],
#                                     [[[0.,0.],[0.,0.]],[[0.,0.],[0.,1.]]]]]),
#                          inds=(out_id.format(x),f'b{x-1}',f'b{x}',in1_id.format(x),in2_id.format(x)),
#                          tags=(site_tag_id.format(x),) )
#         conv_op_tn.add(T1)
#
#     TL = qtn.Tensor( np.array([[[[1.,0.],[0.,0.]],[[0.,1.],[1.,0.]]],
#                                [[[0.,0.],[0.,1.]],[[0.,0.],[0.,0.]]]]),
#                      inds=(f'b{L-2}', out_id.format(L-1),in1_id.format(L-1),in2_id.format(L-1)),
#                      tags=(site_tag_id.format(L-1),) )
#     conv_op_tn.add(TL)
#     return conv_op_tn


# def convolve(mps1,mps2,compress=False,site_ind_id=None,site_tag_id=None,**compress_opts):
#     """ h(x) = convolve( f(x), g(x) )
#         convolution of the two MPS
#     """
#     ### This with probably assume binary ordering
#
#     if site_ind_id is None:  site_ind_id = mps1.site_ind_id
#     if site_tag_id is None:  site_tag_id = mps1.site_tag_id
#
#     convolve_op = get_convolve_op(mps1.L,site_tag_id=site_tag_id,out_id='out{}',
#                                          in1_id=mps1.site_ind_id,in2_id=mps2.site_ind_id)
#     convolve_op.add(mps1,mps2)
#     new_mps = convolve_op.contract()
#     new_mps.view_like(mps1, site_ind_id='out{}', site_tag_id=site_tag_id, inplace=True)
#     new_mps.site_ind_id = site_ind_id
#     return new_mps


def tn1D_from_dense(tensor: qtn.Tensor, ns: int, site_nlegs: Sequence[int], site_tag_id='I{}',
                    direction=0, split_opts=None) -> qtn.TensorNetwork:
    """Split a dense tensor into a 1D tensor network of ``ns`` sites.

    Index labels are taken from ``tensor`` (in order); ``site_nlegs`` dictates
    how many legs each site receives, hence where the SVD splits occur. The
    tensor norm is factored into ``tn.exponent``; a near-zero norm yields an
    all-zero chain.

    Parameters
    ----------
    tensor : qtn.Tensor
        The dense tensor to decompose (indices already ordered by site).
    ns : int
        Number of sites in the resulting network.
    site_nlegs : Sequence[int]
        Number of physical legs assigned to each site.
    site_tag_id : str
        Format string for site tags. Default ``'I{}'``.
    direction : int
        ``0`` splits left-to-right; nonzero splits right-to-left.
    split_opts : dict, optional
        Split options (copied); ``'absorb'`` forced to ``'right'`` and
        ``'cutoff'``/``'cutoff_mode'`` default to the module constants.

    Returns
    -------
    qtn.TensorNetwork
        The resulting 1D tensor network with norm carried in ``exponent``.
    """
    tensor_list = []

    if split_opts is None:
        split_opts = {}
    else:
        split_opts = split_opts.copy()
    split_opts['absorb'] = 'right'
    split_opts.setdefault('cutoff', CUTOFF)
    split_opts.setdefault('cutoff_mode', CUTOFF_MODE)

    inds_list = tensor.inds
    ind0 = 0

    norm = tensor.norm()
    if norm ** 2 < CUTOFF:
        for i in range(ns):
            ind_names = tuple(inds_list[ind0:ind0 + site_nlegs[i]])
            ind_sizes = tuple([tensor.ind_size(ind_name) for ind_name in ind_names])
            ind0 += site_nlegs[i]

            if i == 0:
                ztens = np.zeros(ind_sizes + (1,))
                tmp_inds = (f'_tmp_{i}',)
            elif i == ns - 1:
                ztens = np.zeros(ind_sizes + (1,))
                tmp_inds = (f'_tmp_{i - 1}',)
            else:
                ztens = np.zeros(ind_sizes + (1, 1))
                tmp_inds = (f'_tmp_{i - 1}', f'_tmp_{i}')

            tensor_list += [qtn.Tensor(ztens, inds=ind_names + tmp_inds, tags=(site_tag_id.format(i)))]
        norm = 1.

    else:
        tensor.modify(data=tensor.data * (1. / norm))  # (1./norm) * tensor

        if direction == 0:
            sites_n = range(ns - 1)
            indL = ns - 1
            ind0 = 0
        else:
            sites_n = range(ns - 1, 0, -1)
            indL = 0
            ind0 = len(inds_list)

        TM = tensor
        iso_inds = ()
        for i in sites_n:
            if direction == 0:
                iso_inds += inds_list[ind0:ind0 + site_nlegs[i]]
                ind0 += site_nlegs[i]
            else:
                iso_inds += inds_list[ind0 - site_nlegs[i]:ind0]
                ind0 -= site_nlegs[i]
            # print('TM', TM, i)
            # TL, TM = TM.split(left_inds=iso_inds,bond_ind=f'vb{i}',get='tensors',**compress_opts)
            TL, TM = TM.split(left_inds=iso_inds, get='tensors', **split_opts)
            TL.add_tag(site_tag_id.format(i))
            iso_inds = tuple(TL.bonds(TM))  # (f'vb{i}',)
            tensor_list += [TL]
        # print('TM', TM, indL)
        TM.add_tag(site_tag_id.format(indL))
        tensor_list += [TM]

    tn = qtn.TensorNetwork(tensor_list)
    tn.exponent += np.log10(norm)
    return tn


def mpx_from_dense(tensor: qtn.Tensor, ns: int, site_inds_list: Sequence[str], site_tag_id='I{}',
                   direction=0, return_mpx=True, split_opts=None, left_anc: list[str] = None, left_ind=0):
    """Decompose a dense tensor (legs ordered by site) into an MPS/MPO.

    Successively SVD-splits the tensor site by site using ``site_inds_list`` to
    name the physical legs; the norm is carried in ``tn.exponent``. Returns
    ``None`` if the norm is near zero. The result is viewed as an MPS (one
    physical leg) or MPO (two legs) when ``return_mpx`` is True.

    Parameters
    ----------
    tensor : qtn.Tensor
        The dense tensor to decompose.
    ns : int
        Number of sites.
    site_inds_list : Sequence[str]
        Physical index id format strings (1 for MPS, 2 for MPO).
    site_tag_id : str
        Format string for site tags. Default ``'I{}'``.
    direction : int
        ``>= 0`` splits left-to-right; ``< 0`` right-to-left.
    return_mpx : bool
        Return a viewed MPS/MPO if True; else return the raw tensor list scaled
        by ``norm**(1/ns)``.
    split_opts : dict, optional
        Split options (copied); ``'absorb'`` forced to ``'right'`` and
        ``'cutoff'``/``'cutoff_mode'`` default to the module constants.
    left_anc : list[str], optional
        Ancilla index labels carried on the leftmost split.
    left_ind : int
        Starting site number for tagging. Default 0.

    Returns
    -------
    qtn.TensorNetwork or list[qtn.Tensor] or None
        The MPS/MPO (or raw tensor list), or ``None`` for a near-zero norm.
    """
    tensor_list = []
    errs = []

    if split_opts is None:
        split_opts = {}
    else:
        split_opts = split_opts.copy()
    split_opts['absorb'] = 'right'
    split_opts.setdefault('cutoff', CUTOFF)
    split_opts.setdefault('cutoff_mode', CUTOFF_MODE)
    norm_cutoff = split_opts.pop('norm_cutoff', np.sqrt(split_opts['cutoff']))

    # if direction == 0:
    #     site_n = range(ns - 1)
    #     indL = ns - 1
    # else:  # direction == 1:
    #     site_n = range(ns - 1, 0, -1)
    #     indL = 0

    if direction >= 0:
        site_n = range(left_ind, left_ind + ns - 1)
        indL = left_ind + ns - 1
    else:  # direction == 1:
        site_n = range(left_ind + ns - 1, left_ind, -1)
        indL = left_ind

    norm = tensor.norm()
    if norm ** 2 < CUTOFF:
        return None
        # for i in range(ns):
        #     ind_names = tuple([ind_name.format(i) for ind_name in site_inds_list])
        #     ind_sizes = tuple([tensor.ind_size(ind_name) for ind_name in ind_names])
        #
        #     if i == 0:
        #         if ns > 1:
        #             ztens = np.zeros(ind_sizes + (1,))
        #             tmp_inds = (f'_tmp_{i}',)
        #         else:
        #             ztens = np.zeros(ind_sizes)
        #             tmp_inds = ()
        #     elif i == ns - 1:
        #         ztens = np.zeros(ind_sizes + (1,))
        #         tmp_inds = (f'_tmp_{i - 1}',)
        #     else:
        #         ztens = np.zeros(ind_sizes + (1, 1))
        #         tmp_inds = (f'_tmp_{i - 1}', f'_tmp_{i}')
        #
        #     tensor_list += [qtn.Tensor(ztens, inds=ind_names + tmp_inds, tags=(site_tag_id.format(i)))]
        #
        # norm = 0.
    else:
        tensor = 1. / norm * tensor

        TM = tensor
        left_inds = [] if left_anc is None else left_anc
        for i in site_n:
            left_inds += [ind_name.format(i) for ind_name in site_inds_list]
            TL, TM = TM.split(left_inds=left_inds, bond_ind=f'vb{i}', get='tensors', **split_opts)
            TL.add_tag(site_tag_id.format(i))
            left_inds = [f'vb{i}']
            tensor_list += [TL]
        TM.add_tag(site_tag_id.format(indL))
        tensor_list += [TM]

    tn = qtn.TensorNetwork(tensor_list)
    tn.exponent += np.log10(norm)
    if return_mpx:
        if len(site_inds_list) == 1:
            qtn.MatrixProductState.from_TN(tn, inplace=True, cyclic=False, L=ns,
                                           site_tag_id=site_tag_id,
                                           site_ind_id=site_inds_list[0])
        elif len(site_inds_list) == 2:
            qtn.MatrixProductOperator.from_TN(tn, inplace=True, cyclic=False, L=ns,
                                              site_tag_id=site_tag_id,
                                              upper_ind_id=site_inds_list[0],
                                              lower_ind_id=site_inds_list[1])
        return tn
    else:
        scale = norm ** (1. / ns)
        for i in range(ns):     tensor_list[i] *= scale
        return tensor_list


def mpx_from_dense_new(tensor: 'qtn.Tensor', ns: int, site_inds_list: list['str'], site_tag_id='I{}', left_ind=0,
                       left_anc=None, right_anc=None, direction=0, split_opts=None
                       ) -> Union['qtn.MatrixProductState', 'qtn.MatrixProductOperator', 'qtn.TensorNetwork']:
    """Decompose a dense tensor into an MPS/MPO (variant with left/right ancillas).

    Like :func:`mpx_from_dense` but supports both left and right ancilla legs and
    folds the norm back into the terminal tensor rather than the exponent.
    Returns ``None`` for a near-zero norm. The result is always viewed as an MPS
    (one physical leg) or MPO (two legs).

    Parameters
    ----------
    tensor : qtn.Tensor
        The dense tensor to decompose.
    ns : int
        Number of sites.
    site_inds_list : list[str]
        Physical index id format strings (1 for MPS, 2 for MPO).
    site_tag_id : str
        Format string for site tags. Default ``'I{}'``.
    left_ind : int
        Starting site number for tagging. Default 0.
    left_anc : str, optional
        Ancilla index carried at the left end (used when ``direction >= 0``).
    right_anc : str, optional
        Ancilla index carried at the right end (used when ``direction < 0``).
    direction : int
        ``>= 0`` splits left-to-right; ``< 0`` right-to-left.
    split_opts : dict, optional
        Split options (copied); ``'absorb'`` forced to ``'right'`` and
        ``'cutoff'``/``'cutoff_mode'`` default to the module constants.

    Returns
    -------
    qtn.MatrixProductState or qtn.MatrixProductOperator or qtn.TensorNetwork or None
        The decomposed network, or ``None`` for a near-zero norm.
    """
    tensor_list = []
    errs = []

    if split_opts is None:
        split_opts = {}
    else:
        split_opts = split_opts.copy()
    split_opts['absorb'] = 'right'
    split_opts.setdefault('cutoff', CUTOFF)
    split_opts.setdefault('cutoff_mode', CUTOFF_MODE)

    if direction >= 0:
        site_n = range(left_ind, left_ind + ns - 1)
        indL = left_ind + ns - 1
    else:  # direction == 1:
        site_n = range(left_ind + ns - 1, left_ind, -1)
        indL = left_ind

    norm = tensor.norm()
    if norm ** 2 < CUTOFF:
        return None

    else:
        tensor = tensor * 1. / norm

        TM = tensor
        if direction >= 0:
            left_inds = [] if left_anc is None else [left_anc]
        else:
            left_inds = [] if right_anc is None else [right_anc]

        for i in site_n:
            left_inds += [ind_name.format(i) for ind_name in site_inds_list]
            TL, TM = TM.split(left_inds=left_inds, get='tensors', **split_opts)
            TL.add_tag(site_tag_id.format(i))
            left_inds, _ = TL.filter_bonds(TM)  # [f'vb{i}']  ## shared, not shared
            tensor_list += [TL]
        TM.add_tag(site_tag_id.format(indL))
        tensor_list += [TM * norm]

    tn = qtn.TensorNetwork(tensor_list)

    if len(site_inds_list) == 1:
        qtn.MatrixProductState.from_TN(tn, inplace=True, cyclic=False, L=ns,
                                       site_tag_id=site_tag_id,
                                       site_ind_id=site_inds_list[0])
    elif len(site_inds_list) == 2:
        qtn.MatrixProductOperator.from_TN(tn, inplace=True, cyclic=False, L=ns,
                                          site_tag_id=site_tag_id,
                                          upper_ind_id=site_inds_list[0],
                                          lower_ind_id=site_inds_list[1])
    return tn


def mps_to_gamma_lambda(mps: 'qtn.MatrixProductState', cur_orthog=None, cutoff=CUTOFF, verbose=False,
                        ) -> tuple[list[qtn.Tensor], list[qtn.Tensor], Numeric]:
    """Convert an MPS to the Vidal Gamma-Lambda (canonical) form.

    Sweeps left to right performing SVDs, accumulating the (normalized)
    singular-value tensors as ``lambdas`` and the gauge-transformed site tensors
    as ``gammas`` (multiplying in the inverse of the previous lambda), with the
    total ``log10`` norm carried separately. A round-trip check via
    :func:`gamma_lambda_to_mps` is run for diagnostics.

    Parameters
    ----------
    mps : qtn.MatrixProductState
        The MPS to convert.
    cur_orthog : int or tuple, optional
        Current orthogonality center passed to the initial canonization.
    cutoff : float
        SVD truncation threshold. Default module const CUTOFF.
    verbose : bool
        Print norm/round-trip-error diagnostics if True.

    Returns
    -------
    tuple[list[qtn.Tensor], list[qtn.Tensor], Numeric]
        ``(gammas, lambdas, tot_norm)`` where ``tot_norm`` is the base-10 log
        norm.
    """
    mps = mps.canonize(where=0, cur_orthog=cur_orthog)
    # s_tag = '_S_{}_'

    mps_copy = mps.copy()
    mps = mps.copy()

    tens_i = mps[0]
    gammas, lambdas = [], []
    inv_lam = None
    tot_norm = 0.0
    for i in range(mps.L - 1):

        right_bonds, left_bonds = tens_i.filter_bonds(mps[i + 1])
        bond_r = mps.bond(i, i + 1)
        tens_i.reindex({bond_r: bond_r + '_'}, inplace=True)
        mps[i + 1].reindex({bond_r: bond_r + '_'}, inplace=True)
        # print('left bonds', tens_i, left_bonds, i)
        tens_L, tens_S, tens_R = tens_i.split(left_bonds, method='svd', absorb=None, bond_ind=bond_r, cutoff=cutoff,
                                              cutoff_mode=CUTOFF_MODE)
        norm_S = tens_S.norm()
        # print('tens S', tens_S.data, norm_S, tens_L.norm())
        # print('svals', singular_values(mps_copy, i+1))
        tens_S.modify(apply=lambda x: x / norm_S)
        tot_norm += np.log10(norm_S)
        # print('tens_i', tens_i)
        # print('tens_L', tens_L)
        # print('tens_R', tens_R)
        tens_S.drop_tags()
        tens_R.drop_tags()
        new_R = tens_R.contract(tens_S, output_inds=tens_R.inds)
        # print('S data', tens_S.data)
        # print('svals', ref_mps.copy().singular_values(i + 1))
        #
        # ref_mps.left_compress_site(i, cutoff=1.0e-20)
        # # ref_mps._left_decomp_site(i,method='svd',cutoff=1.0e-10)
        # print('check orthog', i+1)
        # check_orthog(ref_mps)
        # mps_tens = ref_mps[i]
        # if inv_lam is not None:
        #     mps_tens = mps_tens.reindex({ref_mps.bond(i,i-1): inv_lam.inds[0]})
        # mps_tens.transpose_like(tens_L, inplace=True)
        # print('diff', np.linalg.norm(mps_tens.data - tens_L.data))
        # print('mps tens', mps_tens.data)
        # print('tens L', tens_L.data)

        tens_i = qtn.tensor_contract(new_R, mps[i + 1])
        # tens_S.add_tag(s_tag.format(i))

        if inv_lam is not None:
            # print('apply inv_lam', inv_lam)
            tens_L_invS = qtn.tensor_contract(tens_L, inv_lam, output_inds=tens_L.inds)
            tens_L = tens_L_invS
            # print('tensL norm', tens_L.norm())
            # tens_L_invS.transpose_like(tens_L, inplace=True)
            # tens_L.modify(data=tens_L_invS.data)

        inv_lam = tens_S.copy()
        inv_lam.modify(apply=lambda x: 1. / x)
        # iden = inv_lam.contract(tens_S, output_inds=tens_S.inds)
        # print(iden.data)

        gammas += [tens_L.copy()]
        tens_lam = tens_S.copy()
        # tens_lam.modify(apply = lambda x: x / norm_S)
        # print('tens_lam', tens_lam.data)
        lambdas += [tens_lam]

    if inv_lam is not None:
        tens_i_invS = qtn.tensor_contract(tens_i, inv_lam, output_inds=tens_i.inds)
        tens_i = tens_i_invS
        # tens_i_invS.transpose_like(tens_i, inplace=True)
        # tens_i.modify(data=tens_i_invS.data)

    gammas += [tens_i.copy()]
    if verbose:
        print('gl tot norm', tot_norm)

    check = gamma_lambda_to_mps(gammas, lambdas, view_like=mps_copy)
    mps_copy.exponent = tot_norm
    if verbose:
        print('GL ERR', distance(check, mps_copy), check.norm(), mps_copy.norm())

    return gammas, lambdas, tot_norm


def gamma_lambda_to_mps(gammas, lambdas, canon_site=0, view_like=None) -> 'qtn.TensorNetwork1D':
    """Reconstruct an MPS from Vidal Gamma-Lambda tensors, canonical at a site.

    Builds left-canonical tensors to the left of ``canon_site`` and
    right-canonical tensors to its right by contracting each gamma with the
    appropriate neighbouring lambda, then assembles the central site.

    Parameters
    ----------
    gammas : list[qtn.Tensor]
        The Gamma (site) tensors.
    lambdas : list[qtn.Tensor]
        The Lambda (singular-value) tensors between sites.
    canon_site : int
        Site about which the reconstructed MPS is canonical. Default 0.
    view_like : qtn.TensorNetwork1D, optional
        Template network to copy index/tag ids from; if None the result is
        viewed as a generic MPS.

    Returns
    -------
    qtn.TensorNetwork1D
        The reconstructed MPS.
    """
    L = len(gammas)
    new_mpx = qtn.TensorNetwork([])
    for i in range(canon_site):
        if i == 0:
            tens_i = gammas[i]
        else:
            lam_tens = lambdas[i - 1]
            lam_tens.drop_tags()
            tens_i = qtn.tensor_contract(gammas[i], lam_tens, output_inds=gammas[i].inds)

        # tens_i_cc = tens_i.conj()
        # bR = lambdas[i].inds[0]
        # tens_i_cc.reindex({bR: bR + '_'}, inplace=True)
        # out = tens_i_cc.contract(tens_i)
        # print('left canon?', i, np.linalg.norm(out.data - np.eye(out.shape[0])))

        new_mpx.add(tens_i)

    for i in range(L - 1, canon_site, -1):
        if i == L - 1:
            tens_i = gammas[i]
        else:
            lam_tens = lambdas[i]
            lam_tens.drop_tags()
            tens_i = qtn.tensor_contract(gammas[i], lam_tens, output_inds=gammas[i].inds)

        # tens_i_cc = tens_i.conj()
        # bL = lambdas[i - 1].inds[0]
        # tens_i_cc.reindex({bL: bL + '_'}, inplace=True)
        # out = tens_i_cc.contract(tens_i)
        # print('right canon?', i, np.linalg.norm(out.data - np.eye(out.shape[0])))

        new_mpx.add(tens_i)

    contract_tens_list = [gammas[canon_site]]
    if canon_site > 0:
        contract_tens_list += [lambdas[canon_site - 1]]
    if canon_site < L - 1:
        contract_tens_list += [lambdas[canon_site]]

    tens_i = qtn.tensor_contract(*contract_tens_list, output_inds=gammas[canon_site].inds)
    new_mpx.add(tens_i)

    if view_like is None:
        new_mpx.view_as(qtn.MatrixProductState, L=L, cyclic=False, inplace=True)
    else:
        new_mpx.view_like(view_like, inplace=True)

    return new_mpx


def check_gamma_lambda_to_mps(gammas, lambdas, mps, verbose=False):
    """Diagnostic: verify Gamma-Lambda tensors reproduce a reference MPS.

    For every choice of canonical center, reconstructs the left-, right-, and
    central-canonical tensors from ``gammas``/``lambdas`` and compares them
    (canonicality and per-tensor distance) against the canonized reference
    ``mps``, printing the results when ``verbose``.

    Parameters
    ----------
    gammas : list[qtn.Tensor]
        The Gamma (site) tensors.
    lambdas : list[qtn.Tensor]
        The Lambda (singular-value) tensors.
    mps : qtn.MatrixProductState
        Reference MPS to compare against (compressed/canonized internally).
    verbose : bool
        Print per-site canonicality and distance diagnostics if True.

    Returns
    -------
    None
    """
    L = len(gammas)
    mps = compress_func(mps, scale=False)

    for canon_site in range(L):
        mps = canonize(mps, i=canon_site, scale=False)

        ## left canonical tensors
        for i in range(canon_site):
            if i == 0:
                tens_i = gammas[i]
            else:
                lam_tens = lambdas[i - 1]
                lam_tens.drop_tags()
                tens_i = qtn.tensor_contract(gammas[i], lam_tens, output_inds=gammas[i].inds)

            inds = []
            if i > 0:
                bL = next(iter(qtn.bonds(gammas[i], gammas[i - 1])))
                inds += [bL]
            if i < L - 1:
                bR = next(iter(qtn.bonds(gammas[i], gammas[i + 1])))
                inds += [bR]
            bi = mps.site_ind(i)
            inds += [bi]
            tens_i = tens_i.transpose(*inds)

            if bR is not None:
                tens_cc = tens_i.conj()
                tens_cc.reindex({bR: bR + '_'}, inplace=True)
                out = tens_i.contract(tens_cc)
                if verbose:
                    print('tens i canon L?', i, np.linalg.norm(out.data - np.eye(out.shape[0])))

            mps_tens = mps[i]
            inds = []
            if i > 0:
                bLm = mps.bond(i, i - 1)
                inds += [bLm]
            if i < L - 1:
                bRm = mps.bond(i, i + 1)
                inds += [bRm]
            bi = mps.site_ind(i)
            inds += [bi]
            mps_tens = mps_tens.transpose(*inds)

            if verbose:
                print('mps tens', mps_tens)
                print('tens i', tens_i)

            if i > 0:
                if verbose:
                    print('bL', bL, bLm)
                # tens_cc = mps_tens.conj()
                # tens_cc.reindex({bR: bR + '_'}, inplace=True)
                tens_cc.reindex({bL: bLm}, inplace=True)
                out = mps_tens.contract(tens_cc)
                if verbose:
                    print('out', out.data)
                    print('mps tens canon L?', i, np.linalg.norm(out.data - np.eye(out.shape[0])))

            if verbose:
                print('mps dist L', i, np.linalg.norm(tens_i.data - mps_tens.data))

        ## right canonical tensors
        for i in range(L - 1, canon_site, -1):
            if i == L - 1:
                tens_i = gammas[i]
            else:
                lam_tens = lambdas[i]
                lam_tens.drop_tags()
                tens_i = qtn.tensor_contract(gammas[i], lam_tens, output_inds=gammas[i].inds)

            inds = []
            if i > 0:
                bL = next(iter(qtn.bonds(gammas[i], gammas[i - 1])))
                inds += [bL]
            if i < L - 1:
                bR = next(iter(qtn.bonds(gammas[i], gammas[i + 1])))
                inds += [bR]
            bi = mps.site_ind(i)
            inds += [bi]
            tens_i = tens_i.transpose(*inds)

            if bL is not None:
                tens_cc = tens_i.conj()
                tens_cc.reindex({bL: bL + '_'}, inplace=True)
                out = tens_i.contract(tens_cc)
                if verbose:
                    print('out', out)
                    print('tens i canon R?', i, np.linalg.norm(out.data - np.eye(out.shape[0])))

            mps_tens = mps[i]
            inds = []
            if i > 0:
                bL = mps.bond(i, i - 1)
                inds += [bL]
            if i < L - 1:
                bR = mps.bond(i, i + 1)
                inds += [bR]
            bi = mps.site_ind(i)
            inds += [bi]
            mps_tens = mps_tens.transpose(*inds)

            if bL is not None:
                tens_cc = mps_tens.conj()
                tens_cc.reindex({bL: bL + '_'}, inplace=True)
                out = mps_tens.contract(tens_cc)
                if verbose:
                    print('mps tens canon R?', i, np.linalg.norm(out.data - np.eye(out.shape[0])))

            if verbose:
                print('mps dist R', i, np.linalg.norm(tens_i.data - mps_tens.data))

        i = canon_site
        contract_tens_list = [gammas[canon_site]]
        if canon_site > 0:
            contract_tens_list += [lambdas[canon_site - 1]]
        if canon_site < L - 1:
            contract_tens_list += [lambdas[canon_site]]

        tens_i = qtn.tensor_contract(*contract_tens_list, output_inds=gammas[canon_site].inds)

        inds = []
        if i > 0:
            bL = next(iter(qtn.bonds(gammas[i], gammas[i - 1])))
            inds += [bL]
        if i < L - 1:
            bR = next(iter(qtn.bonds(gammas[i], gammas[i + 1])))
            inds += [bR]
        bi = mps.site_ind(i)
        inds += [bi]
        tens_i = tens_i.transpose(*inds)

        mps_tens = mps[i]
        inds = []
        if i > 0:
            bL = mps.bond(i, i - 1)
            inds += [bL]
        if i < L - 1:
            bR = mps.bond(i, i + 1)
            inds += [bR]
        bi = mps.site_ind(i)
        inds += [bi]
        mps_tens = mps_tens.transpose(*inds)

        if verbose:
            print('mps dist M', i, np.linalg.norm(tens_i.data - mps_tens.data))

    return


def get_submpx(mpx: Union[qtn.MatrixProductState, qtn.MatrixProductOperator], ind1: int, ind2: int, reindex=True):
    """Extract a contiguous sub-network of sites ``[ind1, ind2)`` from an mpx.

    Copies the selected tensors and, when ``reindex`` is True, renumbers their
    site tags and physical legs to start from 0; the exponent is preserved.

    Parameters
    ----------
    mpx : qtn.MatrixProductState or qtn.MatrixProductOperator
        The source network.
    ind1 : int
        First site (inclusive) of the sub-range.
    ind2 : int
        End site (exclusive); negative values count from the end.
    reindex : bool
        Renumber the extracted sites/legs to start at 0 if True.

    Returns
    -------
    qtn.TensorNetwork
        The extracted sub-network of length ``ind2 - ind1``.
    """
    ind2 = mpx.L + ind2 + 1 if ind2 < 0 else ind2

    new_tens_list = []
    for ix, i in enumerate(range(ind1, ind2)):
        new_tens = mpx[i].copy()
        if reindex:
            new_tens.retag({mpx.site_tag_id.format(i): mpx.site_tag_id.format(ix)}, inplace=True)
            if isinstance(mpx, qtn.MatrixProductState):
                new_tens.reindex({mpx.site_ind_id.format(i): mpx.site_ind_id.format(ix)}, inplace=True)
            elif isinstance(mpx, qtn.MatrixProductOperator):
                new_tens.reindex({mpx.upper_ind_id.format(i): mpx.upper_ind_id.format(ix),
                                  mpx.lower_ind_id.format(i): mpx.lower_ind_id.format(ix)}, inplace=True)

        new_tens_list += [new_tens]

    out_mpx = qtn.TensorNetwork(new_tens_list)
    out_mpx.exponent = mpx.exponent
    out_mpx.view_like(mpx, L=(ind2 - ind1), inplace=True)
    return out_mpx


def append_mpx(mpx1, mpx2, inplace=False, mps_use_lower=True):
    """Append ``mpx2`` to the end of ``mpx1``, joining them into one mpx.

    Retags ``mpx2``'s sites to continue after ``mpx1``, reconciles the physical
    index ids (resolving collisions), creates a bond-dimension-1 link between
    the junction tensors when needed, and views the result as an MPO if either
    operand is an MPO. Both operands must be the same kind, except that a mixed
    MPS/MPO case is supported with the MPS leg aligned to the MPO's lower (or
    upper) leg.

    Parameters
    ----------
    mpx1 : qtn.TensorNetwork1D
        The leading network.
    mpx2 : qtn.TensorNetwork1D
        The network appended after ``mpx1`` (returns ``mpx1`` if empty).
    inplace : bool
        Modify ``mpx1`` in place if True, else operate on a copy.
    mps_use_lower : bool
        For a mixed MPS/MPO append, align the MPS ``site_ind_id`` with the MPO's
        ``lower_ind_id`` (True) or ``upper_ind_id`` (False); e.g. for partial
        integration.

    Returns
    -------
    qtn.TensorNetwork1D
        The concatenated network of length ``L1 + L2``.
    """
    L1, L2 = mpx1.num_tensors, mpx2.num_tensors

    if L2 == 0:  return mpx1

    # mpx2 = mpx2 if inplace else mpx2.copy()
    # mpx2.retag_sites(mpx1.site_tag_id)

    mpx2 = mpx2.retag({**{mpx2.site_tag_id.format(i): mpx1.site_tag_id.format(i + L1) \
                          for i in range(L2)}}, inplace=False)

    if isinstance(mpx2, qtn.MatrixProductState):
        if isinstance(mpx1, qtn.MatrixProductState):
            site_ind_id = mpx1.site_ind_id
        else:
            site_ind_id = mpx1.lower_ind_id if mps_use_lower else mpx1.upper_ind_id
            upper_ind_id = mpx1.upper_ind_id
            lower_ind_id = mpx1.lower_ind_id
        mpx2 = mpx2.reindex({**{mpx2.site_ind_id.format(i): site_ind_id.format(i + L1) \
                                for i in range(L2)}}, inplace=True)
    elif isinstance(mpx2, qtn.MatrixProductOperator):
        if isinstance(mpx1, qtn.MatrixProductOperator):
            if mpx2.upper_ind_id == mpx1.lower_ind_id:
                mpx2.upper_ind_id = mpx1.lower_ind_id + qtn.rand_uuid()[-2:]
            if mpx2.lower_ind_id == mpx1.upper_ind_id:
                mpx2.lower_ind_id = mpx1.upper_ind_id + qtn.rand_uuid()[-2:]
            upper_ind_id = mpx1.upper_ind_id
            lower_ind_id = mpx1.lower_ind_id
        else:
            mpx1.site_ind_id = mpx2.lower_ind_id if mps_use_lower else mpx2.upper_ind_id
            upper_ind_id = mpx2.upper_ind_id
            lower_ind_id = mpx2.lower_ind_id

        mpx2 = mpx2.reindex({**{mpx2.upper_ind_id.format(i): upper_ind_id.format(i + L1) \
                                for i in range(L2)}}, inplace=True)
        mpx2 = mpx2.reindex({**{mpx2.lower_ind_id.format(i): lower_ind_id.format(i + L1) \
                                for i in range(L2)}}, inplace=True)

    new = mpx1 if inplace else mpx1.copy()
    new.add_tensor_network(mpx2)  # checks for collisions
    new._L = L1 + L2

    if L1 != 0 and len(new[L1 - 1].bonds(new[L1])) == 0:
        new.new_bond(new.site_tag(L1 - 1), new.site_tag(L1))

    if isinstance(mpx2, qtn.MatrixProductOperator) or isinstance(mpx1, qtn.MatrixProductOperator):
        new.view_as(qtn.MatrixProductOperator, L=L1 + L2, cyclic=False, inplace=True,
                    upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=mpx1.site_tag_id, )

    # print('exponents?', mpx1.exponent, mpx2.exponent, new.exponent, mpx1.exponent + mpx2.exponent)
    if not inplace:
        new.exponent = mpx1.exponent + mpx2.exponent  ## should already be taken care of

    return new


def split_mpx(mpx, split_ind) -> tuple[Optional['qtn.TensorNetwork1D'], Optional['qtn.TensorNetwork1D']]:
    """Split an mpx into two sub-networks at ``split_ind``.

    Returns the left part (sites ``[0, split_ind)``) and right part (sites
    ``[split_ind, L)``); the right part's sites are renumbered to start at 0 and
    the exponent is split evenly (halved) between the two parts.

    Parameters
    ----------
    mpx : qtn.TensorNetwork1D
        The network to split.
    split_ind : int
        The site index at which to split.

    Returns
    -------
    tuple[Optional[qtn.TensorNetwork1D], Optional[qtn.TensorNetwork1D]]
        ``(left, right)``; one element is ``None`` for the degenerate splits.
    """
    L1, L2 = split_ind, mpx.L - split_ind

    if L2 == 0:  return mpx, None
    if L2 == mpx.L - 1:   return None, mpx

    mpx1 = mpx[:split_ind]
    mpx2 = mpx[split_ind:]

    # mpx2 = mpx2.retag({**{mpx2.site_tag_id.format(i + L1): mpx1.site_tag_id.format(i) \
    #                    for i in range(L2)}}, inplace=False)

    if isinstance(mpx2, qtn.MatrixProductState):
        mpx2 = renumber_mps(mpx2, list(range(L1, mpx.L)), list(range(L2)), inplace=True)
    elif isinstance(mpx2, qtn.MatrixProductOperator):
        mpx2 = renumber_mpo(mpx2, list(range(L1, mpx.L)), list(range(L2)), inplace=True)

    mpx1.view_like(mpx, L=L1, inplace=True)
    mpx2.view_like(mpx, L=L2, inplace=True)
    mpx1.exponent = mpx.exponent / 2  # np.sqrt(mpx.exponent)
    mpx2.exponent = mpx.exponent / 2  # np.sqrt(mpx.exponent)
    return mpx1, mpx2


def ones_mps(L, q, site_ind_id='i{}', site_tag_id='T{}'):
    """Construct an MPS whose dense vector is all ones.

    Parameters
    ----------
    L : int
        Number of sites.
    q : int or tuple or list
        Physical dimension per site (a scalar is broadcast to all sites).
    site_ind_id : str
        Format string for physical index ids. Default ``'i{}'``.
    site_tag_id : str
        Format string for site tags. Default ``'T{}'``.

    Returns
    -------
    qtn.MatrixProductState
        The all-ones MPS.
    """
    if not (isinstance(q, tuple) or isinstance(q, list)):  q = (q,) * L
    ones = sum_tensornetwork(L=L, qs=q, inds=site_ind_id, tags=site_tag_id)
    ones.view_as(qtn.MatrixProductState, inplace=True, cyclic=False, L=L,
                 site_ind_id=site_ind_id, site_tag_id=site_tag_id)
    return ones


# def sum(tn,sum_tn=None):   ### previously L1norm
#     tn = tn.view_as(qtn.TensorNetwork)
#     if sum_tn is None:   sum_tn = sum_tensornetwork(like_tn=tn, L=tn.num_tensors)
#     tn.add(sum_tn)
#     return tn.contract() * 10**tn.exponent


def sum_tensornetwork(like_tn=None, scale=1., L=1, qs=None, inds=None, tags=None):
    """Build a rank-1 "summing" tensor network of all-ones vectors.

    Produces a chain of all-ones rank-1 tensors (joined by trivial bonds) whose
    contraction with another network sums its elements. When ``like_tn`` is
    given the outer indices/tags are taken from it; otherwise they are derived
    from ``L``, ``qs``, ``inds`` and ``tags``. The whole network is scaled by
    ``scale`` (spread as ``scale**(1/L)`` per tensor via :func:`scalar_multiply`).

    Parameters
    ----------
    like_tn : qtn.TensorNetwork, optional
        Template whose outer dims/inds/tags define the result's legs.
    scale : float
        Overall scalar applied to the network. Default 1.0.
    L : int
        Number of sites when no template is given. Default 1.
    qs : int or sequence of int, optional
        Physical dimension(s) per site.
    inds : str or list of str, optional
        Physical index id format or explicit labels.
    tags : str or list of str, optional
        Site tag format or explicit tags.

    Returns
    -------
    qtn.TensorNetwork
        The summing tensor network.
    """
    if like_tn is not None:
        inds_dims_list = like_tn.outer_dims_inds()
        tensor_list = []
        i = 0
        for q, ind_name in inds_dims_list:
            tensor_list += [qtn.Tensor(np.ones(q), inds=(ind_name,),
                                       tags=(like_tn.site_tag_id.format(i),))]
            i += 1
        sum_tn = qtn.TensorNetwork(tensor_list)
        for i in range(len(tensor_list) - 1):
            tensL = sum_tn.select_tensors(like_tn.site_tag_id.format(i))[0]
            tensR = sum_tn.select_tensors(like_tn.site_tag_id.format(i + 1))[0]
            tensL.new_bond(tensR)
    else:

        if isinstance(qs, int):    qs = [qs] * L

        if isinstance(tags, list):
            tags_list = tags
        elif isinstance(tags, str):
            tags_list = [tags.format(i) for i in range(L)]
        else:
            tags_list = [f'L1({i})' for i in range(L)]

        if like_tn is not None:
            inds_list = like_tn.outer_inds()
        elif isinstance(inds, list):
            inds_list = inds
        elif isinstance(inds, str):
            inds_list = [inds.format(i) for i in range(L)]
        else:
            inds_list = [f'tmp{(i)}' for i in range(L)]

        sum_tn = qtn.TensorNetwork([qtn.Tensor(np.ones(qs[i]), inds=(inds_list[i],),
                                               tags=(tags_list[i],)) for i in range(L)])
        for i in range(len(tags_list) - 1):
            tensL = sum_tn.select_tensors(tags_list[i])[0]
            tensR = sum_tn.select_tensors(tags_list[i + 1])[0]
            tensL.new_bond(tensR)

    if scale != 1.:   scalar_multiply(sum_tn, scale, inplace=True)
    return sum_tn


def apply_gate(mpx: qtn.TensorNetwork1D, sites: Sequence[int], gate: np.ndarray,
               compress_opts=None, inplace=False, invert_gate=False, direction=1):
    """Apply a dense gate to one site of an MPS/MPO.

    Canonizes around the target site if needed, contracts the gate tensor onto
    the site, redecomposes it via :func:`mpx_from_dense_new`, and writes the new
    tensors back. Only single-site gates are currently supported (multi-site
    raises NotImplementedError).

    Parameters
    ----------
    mpx : qtn.TensorNetwork1D
        The MPS/MPO to act on.
    sites : Sequence[int]
        Site index(es) the gate acts on (length 1 only).
    gate : np.ndarray
        The dense gate array.
    compress_opts : dict, optional
        Split options forwarded to :func:`mpx_from_dense_new`.
    inplace : bool
        Modify ``mpx`` in place if True, else operate on a copy.
    invert_gate : bool
        Apply the conjugate (inverse) gate with swapped in/out legs if True.
    direction : int
        Decomposition sweep direction passed to :func:`mpx_from_dense_new`.

    Returns
    -------
    qtn.TensorNetwork1D
        The network with the gate applied.
    """
    mpo = mpx if inplace else mpx.copy()
    cur_orthog: Optional[tuple[int, int]] = None  # mpx._cur_orthog

    if len(sites) == 1:
        q_min, q_max = sites[0], sites[0]
    else:
        q_min, q_max = min(*sites), max(*sites)
        raise NotImplementedError

    if cur_orthog is None or q_min < cur_orthog[0] or q_max > cur_orthog[1]:
        target_orthog = q_min if cur_orthog is None else (q_min if q_min < cur_orthog[0] else q_max)
        mpo.canonize(where=(target_orthog,), cur_orthog=cur_orthog)

    if isinstance(mpo, qtn.MatrixProductOperator):
        site_inds = [mpo.upper_ind_id, mpo.lower_ind_id]
        istr = mpo.upper_ind_id
    else:
        site_inds = [mpo.site_ind_id]
        istr = mpo.site_ind_ind_id

    if invert_gate:
        gate_tens = qtn.Tensor(np.conj(gate), inds=tuple([istr.format(i) for i in sites] +
                                                         [istr.format(i) + '_' for i in sites]))
    else:
        gate_tens = qtn.Tensor(gate, inds=tuple([istr.format(i) + '_' for i in sites] +
                                                [istr.format(i) for i in sites]))
    tens = qtn.tensor_contract(*mpo[q_min:q_max + 1], gate_tens)
    tens.drop_tags()
    tens.reindex({istr.format(i) + '_': istr.format(i) for i in sites}, inplace=True)

    bond_l = mpo.bond(q_min, q_min - 1) if q_min > 0 else None
    bond_r = mpo.bond(q_max, q_max + 1) if q_max < mpo.L - 1 else None

    tens_mpo = mpx_from_dense_new(tens, (q_max - q_min + 1), site_inds,
                                  left_ind=q_min, left_anc=bond_l, right_anc=bond_r,
                                  direction=direction, split_opts=compress_opts)

    # mpx._cur_orthog = (q_max, q_max) if direction >= 0 else (q_min, q_min)

    ## update self
    for i in range(q_min, q_max + 1):
        new_tens = next(iter(tens_mpo.select(tags=tens_mpo.site_tag_id.format(i))))
        mpo[i].modify(data=new_tens.data, inds=new_tens.inds)

    return mpo


# def to_dense(mpx: qtn.TensorNetwork, inds_seq: Sequence[str]) -> qtn.Tensor:
#
#     out_shape = tuple([mpx.ind_size(ind) for ind in inds_seq])
#     out_data = np.empty(out_shape, dtype=mpx.tensors[0].data.dtype)
#
#     for elems in np.ndindex(out_shape):
#         out = mpx.isel({ind: elem for ind, elem in zip(inds_seq, elems)})
#         out_data[elems] = out.contract()
#
#     out_data = out_data * 10**mpx.exponent
#
#     out_tens = qtn.Tensor(out_data, inds_seq)
#     return out_tens

def to_dense(mpx: qtn.TensorNetwork, inds_seq: Sequence[str] = None) -> np.ndarray:
    """Contract a network to a dense array, applying its exponent factor.

    Parameters
    ----------
    mpx : qtn.TensorNetwork
        The network to densify.
    inds_seq : Sequence[str], optional
        Desired output index ordering; defaults to quimb's natural ordering.

    Returns
    -------
    numpy.ndarray
        The dense array scaled by ``10 ** mpx.exponent``.
    """
    inds_seq = [] if inds_seq is None else inds_seq
    out_data = mpx.to_dense(*inds_seq)
    out_data = out_data * 10 ** mpx.exponent
    # out_tens = qtn.Tensor(out_data, inds_seq)
    return out_data


def partition_1D_mps(mps: 'qtn.MatrixProductState', i: int, do_canonize=True):
    """Split an MPS at bond ``i`` and plot the left/right Schmidt bases.

    Optionally canonizes about site ``i``, splits site ``i`` (absorbing singular
    values to the left), contracts each side into a single dense tensor, and
    plots the resulting left and right basis vectors with matplotlib.

    Parameters
    ----------
    mps : qtn.MatrixProductState
        The MPS to partition (operates on a copy).
    i : int
        Bond/site index to split at; must satisfy ``0 <= i < mps.L - 1``.
    do_canonize : bool
        Canonize about site ``i`` before splitting if True.

    Returns
    -------
    tuple[qtn.Tensor, qtn.Tensor]
        The contracted left and right partition tensors.
    """
    assert (0 <= i < mps.L - 1), f'i should be within mps of length {mps.L}, not {i}'

    mps = mps.copy()

    if do_canonize:
        mps = canonize(mps, i=i, scale=False)

    l_partition = mps[:i]
    r_partition = mps[i + 1:]

    ## site i
    left_inds = [mps.site_ind_id.format(i)]
    if i > 0:
        left_inds += [mps.bond(i, i - 1)]
    # r_ind = mps.bond(i, i + 1)
    bond_ind = 'tmp'

    mps_i = mps[i].copy()
    mps_i.drop_tags()
    l_tens, r_tens = qtn.tensor_split(mps_i, left_inds=left_inds, absorb='left',
                                      ltags=mps.site_tag_id.format(i), rtags=mps.site_tag(i + 1), stags='S',
                                      bond_ind=bond_ind)

    l_partition.add([l_tens])
    r_partition.add([r_tens])

    if l_partition.num_tensors > 0:
        l_inds = [mps.site_ind_id.format(i) for i in range(i + 1)]
        xl_bond = bond_ind
        l_size = l_tens.ind_size(bond_ind)
        # xl_bond = mps.bond(i,i-1)
        # l_size = mps.bond_size(i,i-1)

        l_tens = l_partition.contract_tags(tags=all)
        l_tens.transpose(xl_bond, *l_inds, inplace=True)
        plt.figure()
        plt.plot(l_tens.data.reshape(l_size, -1).T)
        plt.title(f'left bases, {i}')

    if r_partition.num_tensors > 0:
        r_inds = [mps.site_ind_id.format(i) for i in range(i + 1, mps.L)]
        xr_bond = bond_ind
        r_size = r_tens.ind_size(bond_ind)
        # xr_bond = mps.bond(i, i + 1)
        # r_size = mps.bond_size(i, i + 1)

        r_tens = r_partition.contract_tags(tags=all)
        r_tens.transpose(xr_bond, *r_inds, inplace=True)
        plt.figure()
        plt.plot(r_tens.data.reshape(r_size, -1).T)
        plt.title(f'right bases, {i}')

    # full_tens = qtn.tensor_contract(l_tens, r_tens, preserve_tensor=True)
    # full_tens.transpose(*[mps.site_ind_id.format(i) for i in range(mps.L)], inplace=True)

    # plt.figure()
    # plt.plot(full_tens.data.reshape(-1))

    plt.show()

    return l_tens, r_tens

