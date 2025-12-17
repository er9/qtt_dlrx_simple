from enum import Enum, IntEnum
from typing import Union
from setup_.defaults import *
import numpy as np
import scipy.linalg

import quimb.tensor as qtn
from quimb.tensor import MatrixProductState
import helper_quimb


class PauliType(IntEnum):
    I = 0
    X = 1
    Y = 2
    Z = 3

I = np.eye(2)
X = np.array([[0.,1.],[1.,0.]])
Y = np.array([[0,1.j],[-1.j,0.]])
Z = np.array([[1.,0.],[0.,-1.]])

II = np.einsum('ij,kl->ikjl', I, I)
IX = np.einsum('ij,kl->ikjl', I, X)
IY = np.einsum('ij,kl->ikjl', I, Y)
IZ = np.einsum('ij,kl->ikjl', I, Z)
#
XI = np.einsum('ij,kl->ikjl', X, I)
XX = np.einsum('ij,kl->ikjl', X, X)
XY = np.einsum('ij,kl->ikjl', X, Y)
XZ = np.einsum('ij,kl->ikjl', X, Z)
#
YI = np.einsum('ij,kl->ikjl', Y, I)
YX = np.einsum('ij,kl->ikjl', Y, X)
YY = np.real(np.einsum('ij,kl->ikjl', Y, Y))
YZ = np.einsum('ij,kl->ikjl', Y, Z)
#
ZI = np.einsum('ij,kl->ikjl', Z, I)
ZX = np.einsum('ij,kl->ikjl', Z, X)
ZY = np.einsum('ij,kl->ikjl', Z, Y)
ZZ = np.einsum('ij,kl->ikjl', Z, Z)

pauli_dicts = {'II': II, 'IX': IX, 'IY': IY, 'IZ': IZ,
               'XI': XI, 'XX': XX, 'XY': XY, 'XZ': XZ,
               'YI': YI, 'YX': YX, 'YY': YY, 'YZ': YZ,
               'ZI': ZI, 'ZX': ZX, 'ZY': ZY, 'ZZ': ZZ}

class Gate:
    """ Quantum circuit gate
    """

    def __init__(self, qubits: tuple[int,...], operator):

        self.qubits = qubits
        self.num_qubits = len(qubits)
        self.data = operator
        assert(operator.shape == (2,) * 2 * self.num_qubits), \
            f"shape {operator.shape} not compatible with {self.num_qubits} qubit operator"
        assert (self.check_unitary()), 'provided gate is not unitary'

    def __repr__(self):
        return f'Gate({self.qubits})'

    @property
    def npts(self):
        return 2**self.num_qubits

    def copy(self, deep=False):
        return self.__class__(self.qubits, self.data.copy() if deep else self.data)

    def modify(self, new_data):
        assert (new_data.shape == (2,) * 2 * self.num_qubits), \
            f"shape {new_data.shape} not compatible with {self.num_qubits} qubit operator"
        self.data = new_data
        assert(self.check_unitary()), 'modified data is not unitary'

    def get_canon_direction(self, next_gate: 'Gate'):
        """ get direction in which MPS should be canonicalized after applying gate, depending on
            what the next gate is
        """
        direction = -1 if np.min(next_gate.qubits) < np.min(self.qubits) else 1
        return direction

    def check_unitary(self):
        npts = self.npts
        check = self.data.reshape(npts, npts)
        u1 = np.dot(check.T.conj(), check)
        u2 = np.dot(check, check.T.conj())
        err1 = np.linalg.norm(u1 - np.eye(npts))
        err2 = np.linalg.norm(u2 - np.eye(npts))
        if err1 > 1.0e-8 or err2 > 1.0e-8:
            print('gate is not unitary', err1, err2)
            return False
        else:
            return True

    @classmethod
    def build_unitary_from_iso(cls, qubits: tuple[int,...], iso_mat: 'np.ndarray', take_transpose=False):

        L = len(qubits)
        npts = 2**L
        check = iso_mat.reshape(npts, npts)
        # print('tens data', check, np.dot(check.T, check), np.dot(check, check.T))

        if not take_transpose:
            iso_id = np.dot(check.T, check)
            iso_cols = np.where(np.abs(np.diag(iso_id)) < 0.1)[0]   # cols to fill; should be 1 or 0
            # print('iso cols', iso_cols)

            proj_id = np.eye(npts) - np.dot(check, check.T)
            q, r = np.linalg.qr(proj_id)
            nbs = np.argsort(np.abs(np.diag(r)))[-1:-1-len(iso_cols):-1]
            # nbs = np.where(np.abs(np.diag(r)) > 1.0e-12)[0]
            # print('nbs', nbs)
            # print('r', r)

            check[:, iso_cols] = q[:, nbs]
        else:
            iso_id = np.dot(check, check.T)
            iso_rows = np.where(np.abs(np.diag(iso_id)) < 0.1)[0]   # rows to fill; should be 1 or 0

            proj_id = np.eye(npts) - np.dot(check.T, check)
            q, r = np.linalg.qr(proj_id.T.conj())
            nbs = np.argsort(np.abs(np.diag(r)))[-1:-1 - len(iso_rows):-1]
            # nbs = np.where(np.abs(np.diag(r)) > 1.0e-12)[0]

            check[iso_rows, :] = q[:, nbs].conj()

        # print('new data', check, np.dot(check.T, check), np.dot(check, check.T))

        return Gate(qubits, check.reshape((2,)*(2*L)),)


class PGate:
    """ Parameterized gate
    """

    def __init__(self, qubits: tuple[int,...], params: tuple[str,...], func):

        self.qubits = qubits
        self.num_qubits = len(qubits)
        self.func = func    ## returns matrix of appropriate size
        self.params = params

    def evaluate(self, **param_vals):
        return Gate(self.qubits, self.func(**param_vals))

    def copy(self):
        return self.__class__(self.qubits, self.params, self.func)



class Unitary2C(PGate):
    """ Unitary gate
        H = sum of paulis must be Hermitian -- 10 real + 6 complex terms (real along the diagonal)
        s.t.  1j * H is antiHermitian
    """

    def __init__(self, qubits: tuple[int,...]):
        params = ('IX','IY','IZ', 'XI','XX','XY','XZ', 'YI','YX','YY','YZ', 'ZI','ZX','ZY','ZZ')
        super().__init__(qubits, params, self.func)

    def func(self, **param_vals) -> 'np.ndarray':
        mat = np.zeros((4,4))
        for p in self.params:
            if p in param_vals:
                mat += param_vals[p] * pauli_dicts[p]

        out = scipy.linalg.expm(-1.j * mat)
        return out

    def jac(self, **param_vals):
        """
        :return: dU/d_param for each param
        """
        derivs = []
        for p in self.params:
            derivs += [ -1.j * np.dot( pauli_dicts[p], self.func(**param_vals) ) ]


class Unitary2R(PGate):
    """ Unitary gate
        H = sum of paulis must be Hermitian -- 10 real + 6 complex terms (real along the diagonal)
        s.t.  1j * H is antiHermitian
    """

    def __init__(self, qubits: tuple[int,...]):
        params = ('IX','IZ', 'XI','XX','XZ','YY','ZI','ZX','ZZ')
        super().__init__(qubits, params, self.func)

    def func(self, **param_vals) -> 'np.ndarray':
        mat = np.zeros((4,4))
        for p in self.params:
            if p in param_vals:
                mat += param_vals[p] * pauli_dicts[p]

        out = scipy.linalg.expm(-1.j * mat)
        return out


#### mps functions
def apply_gate(self: 'MatrixProductState', gate_i: 'Gate', compress_opts=None, inplace=False, invert_gate=False, direction=1):

    mps = self if inplace else self.copy()
    try:
        cur_orthog = self._cur_orthog
    except AttributeError:
        cur_orthog = (0,0)
        self._cur_orthog = cur_orthog

    # q_min, q_max = min(*gate_i.qubits), max(*gate_i.qubits)
    if len(gate_i.qubits) == 1:
        q_min, q_max = gate_i.qubits[0], gate_i.qubits[0]
    else:
        q_min, q_max = min(*gate_i.qubits), max(*gate_i.qubits)

    if cur_orthog is None or q_min < cur_orthog[0] or q_max > cur_orthog[1]:
        target_orthog = q_min if cur_orthog is None else (q_min if q_min < cur_orthog[0] else q_max)
        mps.canonize(where=target_orthog, cur_orthog=cur_orthog)

    istr = mps.site_ind_id if isinstance(mps, MatrixProductState) else mps.upper_ind_id
    if invert_gate:
        gate_tens = qtn.Tensor(np.conj(gate_i.data), inds=tuple([istr.format(i) for i in gate_i.qubits] +
                                                                [istr.format(i) + '_' for i in gate_i.qubits]))
    else:
        gate_tens = qtn.Tensor(gate_i.data, inds=tuple([istr.format(i) + '_' for i in gate_i.qubits] +
                                                       [istr.format(i) for i in gate_i.qubits]))
    tens = qtn.tensor_contract(*mps[q_min:q_max + 1], gate_tens)
    tens.drop_tags()
    tens.reindex({istr.format(i) + '_': istr.format(i) for i in gate_i.qubits}, inplace=True)

    bond_l = mps.bond(q_min, q_min - 1) if q_min > 0 else None
    bond_r = mps.bond(q_max, q_max + 1) if q_max < mps.L - 1 else None

    tn_inds = [mps.site_ind_id] if isinstance(mps, MatrixProductState) else [mps.upper_ind_id, mps.lower_ind_id]
    tens_mps = helper_quimb.mpx_from_dense_new(tens, (q_max - q_min + 1), tn_inds, direction=direction,
                                               left_anc=bond_l, right_anc=bond_r, left_ind=q_min,
                                               split_opts=compress_opts)

    self._cur_orthog = (q_max, q_max) if direction >= 0 else (q_min, q_min)

    ## update self
    for i in range(q_min, q_max + 1):
        new_tens = next(iter(tens_mps.select(tags=tens_mps.site_tag_id.format(i))))
        mps[i].modify(data=new_tens.data, inds=new_tens.inds)

    return mps


def apply_gates(self: 'MatrixProductState', gates_list: list['Gate',...], compress_opts=None, inplace=False, invert_gates=False,
                return_intermediates=False) -> Union['MatrixProductState', dict[int, 'MatrixProductState']]:

    mps = self if inplace else self.copy()

    intermediates = {}
    gate_inds = range(len(gates_list)-1,-1,-1) if invert_gates else range(len(gates_list))

    # cur_orthog = mps._cur_orthog
    for ig in gate_inds:
        gate_i = gates_list[ig]
        q_min, q_max = min(*gate_i.qubits), max(*gate_i.qubits)

        try:
            next_gate = gates_list[ig-1] if invert_gates else gates_list[ig+1]
            direction = gate_i.get_canon_direction(next_gate)
        except IndexError:
            direction = -1 if q_min < mps.L / 2 else 1

        apply_gate(mps, gate_i, compress_opts=compress_opts, inplace=True, invert_gate=invert_gates,
                       direction=direction)

        if return_intermediates:
            intermediates[ig] = mps.copy()

    if return_intermediates:
        return intermediates

    return mps
