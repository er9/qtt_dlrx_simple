from operator import ifloordiv
from typing import Union, Optional, Sequence
from enums import IntEnum
import defaults
import numpy as np
import quimb.tensor as qtn
from quimb.tensor.tensor_core import TensorNetwork
from quimb.tensor.tensor_1d import TensorNetwork1D, TensorNetwork1DFlat, MatrixProductOperator, MatrixProductState

import helper_quimb
from axis import Axis
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from setup_.quimb_TN1D import MatrixProductTensor, MatrixProductGroup

""" construct an MPO with sparsity / blocks
    do not contract until applying to an MPS
    (though sometimes the bulk matrix is compressible, e.g., FFT)
"""

class BlockedTN1D:
    def __init__(self,
                 grid: 'Grid1D',
                 tn1d_dict: dict[tuple['Axis',...], qtn.TensorNetwork1D],
                 ):

        self.grid = grid
        self.comps = tn1d_dict
        self._exponent = 0.0
        self._sign = 1.0
        self._cur_orthog = None

    @property
    def L(self):
        return self.grid.L

    @property
    def parts(self):
        return list(self.comps.keys())

    @property
    def site_tag_id(self):
        ref_mpo = self.comps[next(iter(self.comps))]
        return ref_mpo.site_tag_id

    @site_tag_id.setter
    def site_tag_id(self, site_tag_id: str):
        for k, v in self.comps.items():
            v.site_tag_id = site_tag_id

    def select(self, index: tuple['Axis',...]):
        for k in self.comps.keys():
            if index in k:
                return self.comps[k]
            if isinstance(index, int) and self.grid.axes[index] in k:
                return self.comps[k]

    def update(self, index: tuple['Axis',...], new_mps: 'qtn.MatrixProductState'):
        for k in self.comps.keys():
            if index in k:
                self.comps[k] = new_mps
            if isinstance(index, int) and self.grid.axes[index] in k:
                self.comps[k] = new_mps

    def scalar_multiply(self, scalar_const):
        """ an inplace operation """

        val = np.abs(scalar_const)
        if np.isreal(scalar_const):
            scalar_const = np.real(scalar_const)
            if scalar_const < 0.0:
                sign = -1
            else:
                sign = 1
        else:
            sign = np.exp(1.j * np.angle(scalar_const))
        self._sign = sign

        if val != 0:
            self._exponent += np.log10(val)  # change total norm of MPS
        else:
            self._exponent += -np.inf


class BlockedState1D(BlockedTN1D):
    def __init__(self,
                 grid: 'Grid1D',
                 mps_dict: dict[tuple['Axis',...], qtn.MatrixProductState],
                 ):

        self.comps: dict[tuple['Axis',...], 'qtn.MatrixProductState'] = {}
        super().__init__(grid, mps_dict)

    @property
    def site_ind_id(self):
        ref_mps = self.comps[next(iter(self.comps))]
        return ref_mps.site_ind_id

    @site_ind_id.setter
    def site_ind_id(self, site_ind_id: str):
        for k, v in self.comps.items():
            v.lower_ind_id = site_ind_id

    def diagonalize(self, upper_ind_id=None, lower_ind_id=None, sparse=False) -> 'BlockedOperator1D':
        mpo_dict = {}
        for k, mps in self.comps:
            mpo_dict[k] = helper_quimb.mps_to_diag_mpo(mps, upper_ind_id, lower_ind_id, sparse)
        return BlockedOperator1D(self.grid, mpo_dict)



class BlockedOperator1D(BlockedTN1D):

    def __init__(self,
                 grid: 'Grid1D',
                 mpo_dict: dict[tuple['Axis',...], qtn.MatrixProductOperator],
                 ):

        self.comps: dict[tuple['Axis',...], 'qtn.MatrixProductOperator'] = {}
        super().__init__(grid, mpo_dict)

    @property
    def upper_ind_id(self):
        ref_mpo = self.comps[next(iter(self.comps))]
        return ref_mpo.upper_ind_id

    @upper_ind_id.setter
    def upper_ind_id(self, upper_ind_id: str):
        for k, v in self.comps.items():
            v.upper_ind_id = upper_ind_id

    @property
    def lower_ind_id(self):
        ref_mpo = self.comps[next(iter(self.comps))]
        return ref_mpo.lower_ind_id

    @lower_ind_id.setter
    def lower_ind_id(self, lower_ind_id: str):
        for k, v in self.comps.items():
            v.lower_ind_id = lower_ind_id

    @property
    def exponent(self):
        return np.sum([op.exponent for op in self.comps.values()])

    def get_site(self, i: int, axes=None) -> Sequence['qtn.Tensor']:

        axes = self.grid.axes if axes is None else axes
        upper_ind = self.upper_ind_id.format(i)
        lower_ind = self.lower_ind_id.format(i)

        tens_list = []
        for key, op in self.comps.items():
            if set(key).issubset(set(axes)):
                op_tens = op[i]
                op_tens = op_tens.unfuse({upper_ind: [upper_ind + f'{ax}' for ax in key],
                                          lower_ind: [lower_ind + f'{ax}' for ax in key]},
                                         {upper_ind: [ax.q for ax in key],
                                          lower_ind: [ax.q for ax in key]}
                                         )
                tens_list += [op_tens]
            elif len(set(key).intersection(set(axes))) == 0:    # no intersection
                continue
            else:
                raise ValueError(f'operator not separable into desired axes: {axes}')

        return tens_list

    def apply_to_dense(self, mps_gtn: 'GridTN1D', compress_opts=None):
        """ apply (essentially with zipup)
            mps_gtn can be MPO... but mps_gtn is dense
        """
        mps = mps_gtn.data
        raise NotImplementedError

    def apply_to_blocked(self, mps_gtn: 'BlockedState1D', compress_opts=None):
        raise NotImplementedError




from local_solvers.blocks import BlockDMRG, BlockCross, SweepDirection, EnvironmentType


class BlockedOpBlock_DMRG(BlockDMRG):

    def __init__(self, grid: 'Grid1D', ket: 'qtn.MatrixProductState', operator: BlockedOperator1D,
                 bra: BlockedState1D=None, cur_orthog: int = None,):
        """ block with blocked operator, ket and bra are dense
        """
        self.grid = grid
        self.operator: 'BlockedOperator1D' = operator
        super().__init__(ket, bra, operator, cur_orthog=cur_orthog)

    def extend_env(self, i: int, direction: SweepDirection):

        ket_ind = self.ket_site_ind.format(i)
        bra_ind = self.bra_site_ind.format(i)

        env = self.envs[i - direction]
        tens_list = [env] if env is not None else []

        ket_tens = self.ket[i].copy()
        bra_tens = self.bra[i].copy()
        tens_list += [ket_tens, bra_tens]

        if self.operator is not None:
            ket_tens.unfuse({ket_ind: ket_ind + f'{ax}' for ax in self.grid.axes},
                            {ket_ind: [ax.q for ax in self.grid.axes]}, inplace=True)
            bra_tens.unfuse({bra_ind: bra_ind + f'{ax}' for ax in self.grid.axes},
                            {bra_ind: [ax.q for ax in self.grid.axes]}, inplace=True)

            op_tens_list = self.operator.get_site(i, axes=self.grid.axes)
            tens_list += op_tens_list

        out = qtn.tensor_contract(*tens_list)
        self.envs[i] = out
        self.cur_orthog = i + direction

        return


    def get_projected_site(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
                            return_intermediates=False, site_tens: 'qtn.Tensor' = None):

        if site_tens is not None:
            raise NotImplementedError

        if nsites == 0:
            return self.get_projected_bond(left_site_pos, return_combined=return_combined,
                                           transpose_bonds=transpose_bonds)

        # site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        assert (left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'

        A_left = self.envs[left_site_pos - 1]
        A_right = self.envs[left_site_pos + nsites]

        # A_eff = None
        A_eff = qtn.TensorNetwork([])
        for si in site_inds:
            op_tens_list = self.operator.get_site(si, axes=self.grid.axes)
            A_eff.add(op_tens_list)

        if A_left is not None:
            A_eff.add(A_left.copy())
        if A_right is not None:
            A_eff.add(A_right.copy())

        A_eff.exponent += self.operator.exponent  ## causes issues?
        # A_eff.exponent = self.operator.exponent

        if return_combined:
            A_eff_tens = qtn.tensor_contract(*A_eff.tensors, preserve_tensor=True)
            A_eff_tens.modify(apply=lambda data: data * 10 ** self.operator.exponent)
            if transpose_bonds is not None:
                A_eff_tens.transpose(*transpose_bonds, inplace=True)
            self._projected_site = A_eff_tens.copy()
            return A_eff_tens

        if return_intermediates:
            return A_eff, []

        return A_eff


class TNBlock:

    def __init__(self, kets: list['qtn.MatrixProductState'],
                 bra: 'qtn.MatrixProductState',
                 operator: 'MatrixProductGroup',
                 cur_orthog: int = None,):
        """ kets, bras need to already have the correct labels/connectivity
        """
        L = kets[0].L

        self.kets = kets
        self.operator = operator
        self.bra = bra
        operator.upper_ind_id = self.bra_ind_id
        # self.op_ids = [op.site_tag_id for op in operators]

        ## left and right envs at site i
        self.envs: dict[int, Optional['qtn.Tensor']] = {i: None for i in range(-1, L + 1)}
        self.cur_orthog = cur_orthog

        self._exponent = 0.0
        self._sign = 1.0

    @property
    def L(self):
        return self.kets[0].L

    @property
    def env_type(self):
        raise NotImplementedError

    @property
    def bra_ind_id(self):
        return self.bra.site_ind_id + '_' if (self.operator is not None) else self.bra.site_ind_id

    @property
    def exponent(self):
        return np.sum(self.operator.exponent) + self._exponent

    @property
    def sign(self):
        return self._sign

    def initialize_envs(self, cur_orthog: int):
        if cur_orthog is not None:
            for i in range(cur_orthog):
                self.extend_env(i, direction=SweepDirection.RIGHT)

            for i in range(self.L - 1, cur_orthog, -1):
                self.extend_env(i, direction=SweepDirection.LEFT)
        self.cur_orthog = cur_orthog

    def reinitialize(self):
        self.cur_orthog = None
        for k in self.envs:
            self.envs[k] = None

    def get_tensors(self, i: int, exclude_ket=False, exclude_bra=False):
        """ get tensors corresponding to site "i"
            but exclude tensors listed in "exclude"
        """
        tens_list = [ket[i] for ket in self.kets if ket.site_tag_id not in exclude]

        ## bra
        if not exclude_bra:
            bra_tens = self.bra[i].conj()
            bra_tens = bra_tens.reindex({ind: ind + '_' for ind in bra_tens.inds})
            tens_list += [bra_tens]

        ## operator
        if isinstance(self.operator, MatrixProductGroup):
            tens_list += self.operator.get_tensors(i)
        elif isinstance(self.operator, qtn.MatrixProductOperator):
            tens_list += self.operator[i]
        return tens_list

    # def get_op_tensors(self, i: int):
    #     return [op[i] for op in self.operators]


    def projected_bra_to_ket(self, left_site_pos:int, nsites, ket: 'qtn.MatrixProductState') -> dict[str, str]:
        b2k = {}
        if left_site_pos > 0:
            bl = self.bra.bond(left_site_pos, left_site_pos - 1) + '_'
            kl = ket.bond(left_site_pos, left_site_pos - 1)
            b2k[bl] = kl
        if left_site_pos + nsites < self.bra.L - 1:
            br = self.bra.bond(left_site_pos + nsites - 1, left_site_pos + nsites) + '_'
            kr = ket.bond(left_site_pos + nsites - 1, left_site_pos + nsites)
            b2k[br] = kr
        for i in range(left_site_pos, left_site_pos + nsites):
            b2k[self.bra_ind_id.format(i)] = ket.site_ind_id.format(i)
        return b2k


    # def get_projected(self, left_site_pos: int, nsites: int):
    #     if nsites == 0:
    #         return self.get_projected_bond(left_site_pos)
    #     else:
    #         return self.get_projected_site(left_site_pos, nsites)


    def get_projected(self, left_site_pos: int, nsites: int) -> 'qtn.TensorNetwork':
        """ always return uncombined
            all sites except bra. Also exclude any tensors listed in "exclude"
            (e.g. getting the effective operator wrt the excluded)
        """
        if nsites == 0:
            return self.get_projected_bond(left_site_pos)

        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        assert (left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'

        A_left = self.envs[left_site_pos - 1]
        A_right = self.envs[left_site_pos + nsites]


        A_eff = qtn.TensorNetwork([])
        for si in site_inds:
            A_eff.add(self.get_tensors(si, exclude_ket=True, exclude_bra=True))

        if A_left is not None:
            A_eff.add(A_left.copy())
        if A_right is not None:
            A_eff.add(A_right.copy())

        A_eff.exponent += self.exponent
        return A_eff

    def get_projected_bond(self, left_site_pos: int, exclude: list[str]=None):
        A_left = self.envs[left_site_pos].copy()
        A_right = self.envs[left_site_pos + 1].copy()

        exclude = [] if exclude is None else exclude

        for ket in self.kets:
            if ket.site_tag_id in exclude:      ## not included
                xb_ind = ket.bond(left_site_pos, left_site_pos + 1)
                A_left.reindex({xb_ind: xb_ind + '_L', }, inplace=True)
                A_right.reindex({xb_ind: xb_ind + '_R', }, inplace=True)

        xb_ind = self.bra.bond(left_site_pos, left_site_pos + 1)
        A_left.reindex({xb_ind: xb_ind + '_L_', }, inplace=True)
        A_right.reindex({xb_ind: xb_ind + '_R_', }, inplace=True)

        A_eff = qtn.TensorNetwork([])
        if A_left is not None:
            A_eff.add(A_left)
        if A_right is not None:
            A_eff.add(A_right)

        A_eff.exponent += self.exponent
        return A_eff

    def extend_env(self, i, direction: SweepDirection):
        raise NotImplementedError


class TNBlock_DMRG(TNBlock):

    @property
    def env_type(self):
        return EnvironmentType.DMRG

    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        """ build left environment or right environment to include tensors at site i
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        i_tens_list = self.get_tensors(i)

        env_L, env_R = self.envs[i - 1], self.envs[i + 1]
        env = env_L if direction == SweepDirection.RIGHT else env_R
        envs = [env] if env is not None else []

        new_env = qtn.tensor_contract(*i_tens_list, *envs)

        self.envs[i] = new_env
        self.cur_orthog = i + direction
        return new_env



class TNBlock_Cross(TNBlock):

    @property
    def env_type(self):
        return EnvironmentType.CROSS

    def extend_env(self, i, direction: SweepDirection):
        raise NotImplementedError
