"""Numpy-based serialization for tensor-train (quimb 1-D TN) data.

Replaces the previous pickle-based persistence of :class:`~gridTN.GridTN` /
:class:`~gridTN_1D.GridTN1D` data with self-contained ``.npz`` archives. Per-site
tensor arrays are stored as individual array entries (so ``np.load`` runs with
``allow_pickle=False``); all reconstruction metadata (index labels, tags, id
patterns, the carried ``exponent`` magnitude, the concrete class, and the
:class:`~setup_.enums.DataType`) is stored as a single JSON string entry.

The serializer round-trips plain quimb ``MatrixProductState`` / ``MatrixProductOperator``
/ generic ``TensorNetwork1D`` objects, the custom ``MatrixProductTensor`` (extra physical
legs) and ``MatrixProductStateUSVT`` (canonical-S form), and bare numeric scalars
(``DataType.Num``).
"""

import json

import numpy as np
import quimb.tensor as qtn

from setup_.quimb_TN1D import MatrixProductTensor, MatrixProductStateUSVT

FORMAT_VERSION = 1


def _tn_meta_and_arrays(data):
    """Capture a quimb 1-D TN as a JSON-able metadata dict plus a list of site arrays.

    Parameters
    ----------
    data : qtn.TensorNetwork1D
        The MPS / MPO / MPX / TN3 / USVT object to describe.

    Returns
    -------
    tuple of (dict, list of np.ndarray)
        The metadata block and the per-site data arrays (in tensor order).
    """
    tensors = list(data.tensors)
    arrays = [np.asarray(t.data) for t in tensors]
    sites = [{'inds': list(t.inds), 'tags': list(t.tags)} for t in tensors]

    cls = type(data)
    meta = {
        'format_version': FORMAT_VERSION,
        'class_module': cls.__module__,
        'class_name': cls.__name__,
        'L': int(data.L),
        'cyclic': bool(getattr(data, 'cyclic', False)),
        'exponent': float(getattr(data, 'exponent', 0.0)),
        'site_tag_id': getattr(data, 'site_tag_id', 'I{}'),
        'sites': sites,
    }

    # id patterns differ between MPS and MPO-like networks
    if isinstance(data, qtn.MatrixProductOperator):
        meta['upper_ind_id'] = data.upper_ind_id
        meta['lower_ind_id'] = data.lower_ind_id
    else:
        meta['site_ind_id'] = getattr(data, 'site_ind_id', 'k{}')

    # custom-subclass extras
    if isinstance(data, MatrixProductTensor):
        meta['extra_ind_ids'] = list(data.extra_ind_ids)
    if isinstance(data, MatrixProductStateUSVT):
        meta['canon_site'] = int(data.canon_site)

    return meta, arrays


def _rebuild_tn(meta, arrays):
    """Rebuild a quimb 1-D TN from a metadata block and its site arrays.

    Parameters
    ----------
    meta : dict
        Metadata produced by :func:`_tn_meta_and_arrays`.
    arrays : list of np.ndarray
        Per-site data arrays (in tensor order).

    Returns
    -------
    qtn.TensorNetwork1D
        The reconstructed network, with its ``exponent`` restored.
    """
    tensors = [qtn.Tensor(arr, inds=tuple(s['inds']), tags=set(s['tags']))
               for arr, s in zip(arrays, meta['sites'])]
    tn = qtn.TensorNetwork(tensors)

    name = meta['class_name']
    L = meta['L']
    cyclic = meta['cyclic']
    site_tag_id = meta['site_tag_id']

    if name == 'MatrixProductTensor':
        out = MatrixProductTensor(tn, site_tag_id=site_tag_id,
                                  upper_ind_id=meta['upper_ind_id'],
                                  lower_ind_id=meta['lower_ind_id'],
                                  extra_ind_ids=tuple(meta.get('extra_ind_ids', ())))
    elif name == 'MatrixProductStateUSVT':
        out = MatrixProductStateUSVT(tn, canon_site=meta['canon_site'],
                                     site_tag_id=site_tag_id,
                                     site_ind_id=meta['site_ind_id'])
    elif 'upper_ind_id' in meta:
        # MPO / MPO-like (incl. unknown MPO subclasses -> nearest base)
        out = tn.view_as(qtn.MatrixProductOperator, inplace=True, L=L, cyclic=cyclic,
                         site_tag_id=site_tag_id, upper_ind_id=meta['upper_ind_id'],
                         lower_ind_id=meta['lower_ind_id'])
    else:
        # MPS / MPX / unknown MPS subclass -> nearest base
        out = tn.view_as(qtn.MatrixProductState, inplace=True, L=L, cyclic=cyclic,
                         site_tag_id=site_tag_id, site_ind_id=meta['site_ind_id'])

    out.exponent = meta['exponent']
    return out


def tn1d_to_npz(data, fstr):
    """Save a GridTN1D ``data`` object to ``<fstr>.npz`` without pickle.

    Parameters
    ----------
    data : qtn.TensorNetwork1D or float or complex
        The tensor-train (MPS / MPO / MPX / TN3 / USVT) or a numeric scalar
        (``DataType.Num``) to persist.
    fstr : str
        Path prefix; the ``.npz`` extension is appended.

    Returns
    -------
    str
        The full path written (``<fstr>.npz``).
    """
    path = fstr + '.npz'
    if isinstance(data, (float, complex, np.floating, np.complexfloating)):
        meta = {'format_version': FORMAT_VERSION, 'data_type': 'Num',
                'value_real': float(np.real(data)), 'value_imag': float(np.imag(data))}
        np.savez_compressed(path, meta=np.array(json.dumps(meta)))
        return path

    meta, arrays = _tn_meta_and_arrays(data)
    meta['data_type'] = 'TN'
    array_kwargs = {f'arr_{i}': a for i, a in enumerate(arrays)}
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **array_kwargs)
    return path


def tn1d_from_npz(fstr):
    """Load a GridTN1D ``data`` object from ``<fstr>.npz`` (pickle-free).

    Parameters
    ----------
    fstr : str
        Path prefix; the ``.npz`` extension is appended.

    Returns
    -------
    qtn.TensorNetwork1D or complex or float
        The reconstructed tensor-train, or the stored scalar for ``Num`` data.

    Raises
    ------
    IOError
        If ``<fstr>.npz`` does not exist.
    """
    path = fstr + '.npz'
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['meta']))
        if meta.get('data_type') == 'Num':
            val = complex(meta['value_real'], meta['value_imag'])
            return val if meta['value_imag'] != 0.0 else val.real
        arrays = [z[f'arr_{i}'] for i in range(meta['L'])]
        return _rebuild_tn(meta, arrays)


def comb_to_npz(exponent, sign, spine, branch_data, fstr):
    """Save a comb-layout tuple ``(exponent, sign, spine, branch_data)`` to ``<fstr>.npz``.

    Each branch (and the optional spine) is itself a GridTN1D ``data`` object and is
    serialized into the same archive under prefixed keys, so the whole comb state lives
    in one pickle-free file.

    Parameters
    ----------
    exponent : float
        The comb's carried base-10 log magnitude.
    sign : Numeric
        The comb's carried sign / phase.
    spine : qtn.TensorNetwork1D or None
        The shared spine tensor-train, or None.
    branch_data : list of (qtn.TensorNetwork1D or None)
        Per-branch tensor-trains; entries may be None for inactive branches.
    fstr : str
        Path prefix; the ``.npz`` extension is appended.

    Returns
    -------
    str
        The full path written (``<fstr>.npz``).
    """
    path = fstr + '.npz'
    arrays = {}
    branch_metas = []
    for i, bd in enumerate(branch_data):
        if bd is None:
            branch_metas.append(None)
            continue
        bmeta, barrays = _tn_meta_and_arrays(bd)
        for k, a in enumerate(barrays):
            arrays[f'branch{i}__arr_{k}'] = a
        branch_metas.append(bmeta)

    has_spine = spine is not None
    spine_meta = None
    if has_spine:
        spine_meta, spine_arrays = _tn_meta_and_arrays(spine)
        for k, a in enumerate(spine_arrays):
            arrays[f'spine__arr_{k}'] = a

    meta = {
        'format_version': FORMAT_VERSION,
        'data_type': 'Comb',
        'exponent': float(exponent),
        'sign': [float(np.real(sign)), float(np.imag(sign))],
        'n_branches': len(branch_data),
        'has_spine': has_spine,
        'branches': branch_metas,
        'spine': spine_meta,
    }
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def comb_from_npz(fstr):
    """Load a comb-layout tuple from ``<fstr>.npz`` (pickle-free).

    Parameters
    ----------
    fstr : str
        Path prefix; the ``.npz`` extension is appended.

    Returns
    -------
    tuple
        ``(exponent, sign, spine, branch_data)`` where ``spine`` is a reconstructed
        tensor-train or None and ``branch_data`` is a list of reconstructed
        tensor-trains (None for inactive branches).

    Raises
    ------
    IOError
        If ``<fstr>.npz`` does not exist.
    """
    path = fstr + '.npz'
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['meta']))
        sign_r, sign_i = meta['sign']
        sign = complex(sign_r, sign_i)
        if sign_i == 0.0:
            sign = sign.real
        exponent = meta['exponent']

        branch_data = []
        for i, bmeta in enumerate(meta['branches']):
            if bmeta is None:
                branch_data.append(None)
                continue
            barrays = [z[f'branch{i}__arr_{k}'] for k in range(bmeta['L'])]
            branch_data.append(_rebuild_tn(bmeta, barrays))

        spine = None
        if meta['has_spine']:
            smeta = meta['spine']
            sarrays = [z[f'spine__arr_{k}'] for k in range(smeta['L'])]
            spine = _rebuild_tn(smeta, sarrays)

    return exponent, sign, spine, branch_data
