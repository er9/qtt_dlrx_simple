"""Block layer of the local-solver stack.

Defines the :class:`Block` hierarchy that caches the partial (effective)
environments contracted from the left and right of the current site during a
sweep. Provides DMRG-style blocks (full Galerkin contractions) and cross-style
blocks (index-selected / interpolatory contractions), together with the
:func:`init_block` factory that builds the appropriate block for a given
:class:`EnvironmentType`. These blocks are the lowest-level building units
consumed by the Term and Evaluator layers.
"""
import numpy as np
from abc import ABC

import helper_quimb
from setup_.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
# from setup_.quimb_TN1D import IndexedMPS, IndexedMPO
import helper_quimb as helper
import local_solvers.helper_tn as helper_tn
import local_solvers.helper_dmrg_loc as helper_dmrg
from local_solvers.mps_classes import MPS, MPO
# from local_solvers.local_evaluator import SweepDirection
import local_solvers.helper_cross as helper_cross

ProjType = Union[MPS, qtn.MatrixProductState, dict[int, tuple[int,...]]]

class SweepDirection(IntEnum):
    ## int denotes where the MPS needs to be canonicalized to
    LEFT = -1
    RIGHT = 1

class EnvironmentType(IntEnum):
    DMRG = 0
    CROSS = 1

def init_block(env_type: EnvironmentType, ket, bra, operator=None,
               anc_env_left=None, anc_env_right=None,
               select_inds_ket=None, select_inds_bra=None,
               cur_orthog=False) -> 'Block':

    if env_type == EnvironmentType.DMRG:
        if operator is None:
            return BlockVector_DMRG(ket.L, ket, bra, anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                                    cur_orthog=cur_orthog)
        else:
            if operator in ['identity', 'I', 'iden', 'Iden', 'Identity']:
                return BlockOperator_DMRG(ket.L, ket, bra, anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                                          cur_orthog=cur_orthog)
            else:
                return BlockOperator_DMRG(ket.L, ket, bra, operator=operator,
                                          anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                                          cur_orthog=cur_orthog)
    elif env_type == EnvironmentType.CROSS:
        if operator is None:
            return BlockVector_Cross(ket.L, ket, bra, anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                                     select_inds_bra=select_inds_bra, cur_orthog=cur_orthog)
        else:
            if operator in ['identity', 'I', 'iden', 'Iden', 'Identity']:
                return BlockOperator_Cross(ket.L, ket, bra, anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                                           select_inds_ket=select_inds_ket, select_inds_bra=select_inds_bra,
                                           cur_orthog=cur_orthog)
            else:
                return BlockOperator_Cross(ket.L, ket, bra, operator=operator,
                                           anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                                           select_inds_bra=select_inds_bra, select_inds_ket=select_inds_ket,
                                           cur_orthog=cur_orthog)



class Block:

    def __init__(self, # env_type: EnvironmentType,
                 ket: Union['MPS', 'qtn.MatrixProductState'],
                 bra: Optional[Union['MPS', 'qtn.MatrixProductState']],
                 operator: 'qtn.MatrixProductOperator' = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 # mps_inds: Sequence[int] = None,
                 cur_orthog = None,
                 ):

        # self.L = L
        L = ket.L

        self._ket: 'MPS' = None
        self._bra: Optional['MPS'] = None
        self._operator: Optional['MPO'] = None

        ## left and right envs at site i
        self.envs: dict[int, Optional['qtn.Tensor']] = {i: None for i in range(-1,L + 1)}
        if anc_env_left is not None:
            self.envs[-1] = anc_env_left.copy()
        if anc_env_right is not None:
            self.envs[L] = anc_env_right.copy()

        self.ket = ket
        self.bra = bra
        self.operator = operator  # .copy() if operator is not None else operator  ## why does this cause issues?

        # self.env_type = env_type
        self.cur_orthog = cur_orthog
        self.init_envs(cur_orthog)
        self._projected_site = None


    @property
    def L(self):
        return self.ket.L

    @property
    def env_type(self):
        raise NotImplementedError

    @property
    def ket(self):
        return self._ket

    @ket.setter
    def ket(self, mps: qtn.MatrixProductState):
        mps.view_as(MPS, inplace=True)
        self._ket = mps

    @property
    def bra(self):
        return self._bra

    @bra.setter
    def bra(self, mps: Optional[qtn.MatrixProductState]):
        if mps is not None:
            mps.view_as(MPS, inplace=True)
            self._bra = mps
            # if mps.site_ind_id != self.ket.site_ind_id:
            mps.site_ind_id = self.ket.site_ind_id
            # self.update_horizontal_bonds()

    @property
    def bra_site_ind(self):
        if self.operator is not None:
            return self.operator.upper_ind_id
        else:
            return self.ket.site_ind_id

    @property
    def ket_site_ind(self):
        return self.ket_site_ind

    @property
    def operator(self):
        return self._operator

    @operator.setter
    def operator(self, mpo: Optional[qtn.MatrixProductOperator]):
        if mpo is not None:
            self._operator = mpo
            # if self.bra is not None:
            #     self.bra.site_ind_id = self.bra.site_ind_id + '_'
            mpo.upper_ind_id = self.ket.site_ind_id + '_' if self.bra is None else self.bra.site_ind_id + '_'
            mpo.lower_ind_id = self.ket.site_ind_id
            if mpo.bond(0, 1) == self.ket.bond(0, 1) or mpo.bond(0,1) == self.bra.bond(0,1):
                mpo.mangle_inner_()
        return

    def init_envs(self, cur_orthog: int):
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

    def reinitialize_i(self, i: int):
        self.envs[i] = None

    def bra_site(self, i: int):
        # i = self.mps_inds[i]
        if self.bra is None:
            tens = self.ket.get_bra_tens(i, reindex_phys=False)
            return tens
        else:
            reindex_phys = self.bra_site_ind != self.ket.site_ind_id
            return self.bra.get_bra_tens(i, reindex_phys=reindex_phys)

    def bra_horizontal_bond(self, i: int, step: Union[int, SweepDirection]):
        if self.bra is None:
            ind = self.ket_horizontal_bond(i, step)
            if ind is not None:
                ind = ind + '_'
            return ind

        ind = self.bra.bond(i, i + step)
        if ind is not None:
            ind += '_'
        return ind

    def ket_horizontal_bond(self, i: int, step: int):
        return self.ket.bond(i, i + step)

    def _projected_bra_to_ket_bond(self, left_site_pos: int):
        """ projected bra to ket when extracting a "bond"
        """
        bx_bond = self.bra_horizontal_bond(left_site_pos, 1)
        kx_bond = self.ket_horizontal_bond(left_site_pos, 1)

        inds_dict = {bx_bond + 'L_': kx_bond + '_L', bx_bond + 'R_': kx_bond + '_R'}
        return inds_dict

    def projected_bra_to_ket(self, left_site_pos: int, nsites: int):
        if nsites == 0:
            return self._projected_bra_to_ket_bond(left_site_pos)

        inds_dict = {self.bra_horizontal_bond(left_site_pos, -1):
                         self.ket_horizontal_bond(left_site_pos, -1),
                     self.bra_horizontal_bond(left_site_pos + nsites - 1, 1):
                         self.ket_horizontal_bond(left_site_pos + nsites - 1, 1)}
        for ix in range(left_site_pos, left_site_pos + nsites):
            if self.operator is not None:
                inds_dict[self.operator.upper_ind(ix)] = self.ket.site_ind(ix)
            else:
                if self.bra is not None:
                    inds_dict[self.bra.site_ind(ix) + '_'] = self.ket.site_ind(ix)
                # if self.bra is None:
                #     inds_dict[self.bra.site_ind(ix) + '_'] = self.ket.site_ind(ix)
                # else:
                #     inds_dict[self.bra.site_ind(ix) + '_'] = self.ket.site_ind(ix)
        return inds_dict

    @property
    def projected_site(self):
        return self._projected_site

    def copy(self, ket_copy=None, bra_copy=None):

        if ket_copy is None:
            ket_copy = self.ket.copy()

        if bra_copy is None:
            bra_copy = self.bra.copy() if self.bra is not None else None

        out = self.__class__(ket_copy, bra_copy,)
        out.operator = self.operator.copy() if self.operator is not None else None

        out.envs = {k: (env.copy() if env is not None else None) for k, env in self.envs.items()}
        return out

    @property
    def exponent(self):
        raise NotImplementedError
        # if self.operator is not None:
        #     return self.operator.exponent  ## <x|A|x>
        # else:
        #     return self.ket.exponent    ## <x|b>

    @property
    def anc_env_L(self):
        return self.envs[-1]

    @property
    def anc_env_R(self):
        return self.envs[self.L]

    def clear_envs(self):
        for k in range(self.L):
            self.envs[k] = None

    def get_projected(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
                      return_intermediates=False, site_tens: 'qtn.Tensor'=None):
        raise NotImplementedError

    def get_projected_bond(self, left_site_pos:int, return_combined=False, transpose_bonds=None):
        """ get environment for bond between left_site_pos, left_site_pos + 1
            assumes ket, bra at left_site_pos are canonical and left_site_pos+1 is not updated
            or vice versa (left_site_pos + 1 is now canonical, and left_site_pos is not updated)
        """
        raise NotImplementedError

    def shift_left(self):
        self.extend_env(self.cur_orthog, direction=SweepDirection.LEFT)
        # self.cur_orthog += 1

    def shift_right(self):
        self.extend_env(self.cur_orthog, direction=SweepDirection.RIGHT)
        # self.cur_orthog -= 1

    def extend_env(self, i: int, direction: SweepDirection):
        raise NotImplementedError

    def norm(self):
        """ returns <bra|operator|ket>, without
        """
        if self.cur_orthog is None:
            raise NotImplementedError

        ix = self.cur_orthog
        env_L, env_R = self.envs[ix - 1], self.envs[ix + 1]
        tens_list = [self.ket[ix], self.bra[ix]]
        tens_list += [env_L] if env_L is not None else []
        tens_list += [env_R] if env_R is not None else []
        tens_list += [self.operator[ix]] if self.operator is not None else []

        ovlp = qtn.tensor_contract(*tens_list)

        tot_exponent = self.ket.exponent + self.bra.exponent
        if self.operator is not None:
            tot_exponent += self.operator.exponent

        return ovlp * 10 ** tot_exponent


class BlockDMRG(Block, ABC):

    @property
    def env_type(self):
        return EnvironmentType.DMRG

    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        """ build left environment or right environment to include tensors at site i
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        # bra_tens = [self.bra_site(i)]
        # ket_tens = [self.ket[self.mps_inds[i]]]
        bra_tens = [self.bra_site(i)]
        ket_tens = [self.ket.get_tens(i)]

        if isinstance(self.operator, MatrixProductOperatorTN):
            op_tens = self.operator[i]  # [self.mps_inds[i]]
        elif self.operator is None:
            op_tens = []
        else:
            op_tens = [self.operator[i]]  # [self.mps_inds[i]]]
            # op_tens = [mpo[pos] for mpo in self.operators]

        new_env = qtn.TensorNetwork(bra_tens + ket_tens + op_tens)
        # print('bra tens', bra_tens[0])
        # print('ket tens', ket_tens[0])
        # print('op tens', op_tens[0])

        env_L, env_R = self.envs[i - 1], self.envs[i + 1]
        env = env_L if direction == SweepDirection.RIGHT else env_R
        if env is not None:
            new_env.add(env)


        # new_env_exp = new_env.exponent
        new_env = new_env.contract_tags(all)

        if self.operator is None and direction < 0 and i > 0:
            bra = self.bra.conj()
            bra.mangle_inner_(append='_')
            bra_tensors = [bra[ix] for ix in range(i, bra.L)]
            ket_tensors = [self.ket[ix] for ix in range(i, self.ket.L)]
            tmp_env = qtn.tensor_contract(*bra_tensors, *ket_tensors)
            # print('self.ket', self.ket.exponent, helper_quimb.check_orthog(self.ket))
            tmp_env.transpose_like(new_env, inplace=True)
            # print('extend env diff', np.linalg.norm(tmp_env.data - new_env.data))
            # if i == 1:
            #     print('tmp env', tmp_env.data, tmp_env.inds)


        self.envs[i] = new_env
        self.cur_orthog = i + direction

        return new_env



class BlockCross(Block, ABC):

    @property
    def env_type(self):
        return EnvironmentType.CROSS

####

class BlockVector(Block, ABC):

    @property
    def exponent(self):
        return self.ket.exponent

    def get_projected_bond(self, left_site_pos:int, return_combined=False, transpose_bonds=None,
                           site_tens:'qtn.Tensor'=None, return_intermediates=False):
        """ get environment for bond between left_site_pos, left_site_pos + 1
            assumes ket, bra at left_site_pos are canonical and left_site_pos+1 is not updated
            or vice versa (left_site_pos + 1 is now canonical, and left_site_pos is not updated)
        """
        if site_tens is not None:
            if return_intermediates:
                return site_tens, []
            return site_tens

        # print('GET PROJECTED BOND')
        # print('left site pos', left_site_pos, self.cur_orthog)

        # if direction == SweepDirection.RIGHT:
        #     assert (self.cur_orthog - 1 == left_site_pos), 'orthogonality center not within unprojected sites'
        # else:
        #     assert (self.cur_orthog == left_site_pos), 'orthogonality center not within unprojected sites'

        b_left = self.envs[left_site_pos].copy()
        b_right = self.envs[left_site_pos + 1].copy()

        x_ind = self.bra_horizontal_bond(left_site_pos, 1)
        b_left.reindex({x_ind: x_ind + 'L_'}, inplace=True)
        b_right.reindex({x_ind: x_ind + 'R_'}, inplace=True)

        # print('b left', b_left)
        # print('b right', b_right)

        b_eff = qtn.TensorNetwork([])
        # if isinstance(site_inds, (list, tuple)):
        #     for si in site_inds:
        #         b_eff.add(self.ket[si])
        # else:
        #     b_eff.add(self.ket[site_inds])
        # print('self.ket', [self.ket[si] for si in site_inds])

        if b_left is not None:
            b_eff.add(b_left)
        if b_right is not None:
            b_eff.add(b_right)

        b_eff.exponent += self.ket.exponent

        if return_combined:
            b_eff_tens = qtn.tensor_contract(*b_eff.tensors, preserve_tensor=True)
            b_eff_tens.modify(apply=lambda data: data * 10 ** self.ket.exponent)
            if transpose_bonds is not None:
                b_eff_tens.transpose(*transpose_bonds, inplace=True)
            self._projected_site = b_eff_tens
            b_eff = b_eff_tens

        self._projected_site = b_eff

        if return_intermediates:
            return b_eff, []

        return b_eff

    def get_projected(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
                      return_intermediates=False, site_tens:'qtn.Tensor'=None):
        """ incorporate ket exponent, remove bra exponent
        """
        # print('blocks: get projected vec', left_site_pos, nsites)

        # print('self.ket', self.ket)
        # print('self.bra', self.bra)

        if nsites == 0:
            return self.get_projected_bond(left_site_pos, return_combined=return_combined, site_tens=site_tens,
                                           transpose_bonds=transpose_bonds, return_intermediates=return_intermediates)

        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        # print('left site pos', left_site_pos, 'cur orthog', self.cur_orthog)
        assert (left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'

        b_left = self.envs[left_site_pos - 1]
        # b_right = self.envs[left_site_pos + nsites]
        b_right = self.envs.get(left_site_pos + nsites, None)
        # print('b left', b_left)
        # print('b right', b_right)

        ## check envs from block envs
        # bra = self.bra.conj().copy()
        # bra.mangle_inner_()
        # tmp_right = qtn.tensor_contract(*bra.tensors[1:], *self.ket.tensors[1:])
        # tmp_right.transpose_like(b_right, inplace=True)
        # print('tmp right', tmp_right, b_right)
        # print('env r diff', tmp_right.data - b_right.data)
        # print('env r diff', tmp_right.data.T - b_right.data)
        # print('b right', b_right.data, b_right.inds)
        # print('tmp right', tmp_right.data, tmp_right.inds)

        b_eff = qtn.TensorNetwork([])

        # print('site tens', site_tens)

        if site_tens is None:
            if isinstance(site_inds, (list, tuple)):
                for si in site_inds:
                    b_eff.add(self.ket[si])
            else:
                b_eff.add(self.ket[site_inds])
            # print('self.ket', [self.ket[si] for si in site_inds])

            b_eff.exponent += self.ket.exponent

        else:
            if isinstance(site_tens,(tuple,list)):
                b_eff.add(*site_tens)
            else:
                b_eff.add(site_tens)

        if b_left is not None:
            b_eff.add(b_left)
        if b_right is not None:
            b_eff.add(b_right)

        # ## remove bra exponent
        # b_eff.exponent -= self.bra.exponent

        if return_combined:
            b_eff_tens = qtn.tensor_contract(*b_eff.tensors, preserve_tensor=True)
            b_eff_tens.modify(apply=lambda data: data * 10 ** b_eff.exponent)  # self.ket.exponent)
            if transpose_bonds is not None:
                b_eff_tens.transpose(*transpose_bonds, inplace=True)
            self._projected_site = b_eff_tens
            b_eff = b_eff_tens

        self._projected_site = b_eff

        # print('remove bra exponent')
        # b_eff.modify(apply=lambda x: x * 10 ** (-self.bra.exponent))

        if return_intermediates:
            b2k_dict = self.projected_bra_to_ket(left_site_pos, nsites)
            return b_eff, [b_eff.copy()]  # [b_eff.reindex(b2k_dict, inplace=False)]

        return b_eff

    def decimate(self, pos: int, direction: SweepDirection, new_ket_site: 'qtn.Tensor' = None, max_bond: int = None):
        raise NotImplementedError

    def update_bra(self, left_site_pos: int, nsites: int, direction: SweepDirection, new_site: 'qtn.Tensor' = None,
                   max_bond: int = None):
        # raise NotImplementedError
        return


class BlockOperator(Block, ABC):

    @property
    def exponent(self):
        return self.operator.exponent if self.operator is not None else 0.0

    def get_projected_bond(self, left_site_pos:int, return_combined=False, transpose_bonds=None):
        """ get environment for bond between left_site_pos, left_site_pos + 1
            assumes ket, bra at left_site_pos are canonical and left_site_pos+1 is not updated
            or vice versa (left_site_pos + 1 is now canonical, and left_site_pos is not updated)
        """
        # if direction == SweepDirection.RIGHT:
        #     assert (self.cur_orthog - 1 == left_site_pos), 'orthogonality center not within unprojected sites'
        # else:
        #     assert (self.cur_orthog == left_site_pos), 'orthogonality center not within unprojected sites'

        A_left = self.envs[left_site_pos].copy()
        A_right = self.envs[left_site_pos + 1].copy()

        xb_ind = self.bra_horizontal_bond(left_site_pos, 1)
        xk_ind = self.ket_horizontal_bond(left_site_pos, 1)
        A_left.reindex({xb_ind: xb_ind + 'L_', xk_ind: xk_ind + '_L'}, inplace=True)
        A_right.reindex({xb_ind: xb_ind + 'R_', xk_ind: xk_ind + '_R'}, inplace=True)

        A_eff = qtn.TensorNetwork([])
        # if isinstance(site_inds, (list, tuple)):
        #     for si in site_inds:
        #         A_eff.add(self.operator[si])
        # else:
        #     A_eff.add(self.operator[site_inds])

        if A_left is not None:
            A_eff.add(A_left)
        if A_right is not None:
            A_eff.add(A_right)

        A_eff.exponent += self.operator.exponent

        if return_combined:
            A_eff_tens = qtn.tensor_contract(*A_eff.tensors, preserve_tensor=True)
            A_eff_tens.modify(apply=lambda data: data * 10 ** self.operator.exponent)
            if transpose_bonds is not None:
                A_eff_tens.transpose(*transpose_bonds, inplace=True)
            self._projected_site = A_eff_tens
            return A_eff_tens

        self._projected_site = A_eff
        return A_eff

    def get_projected(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
                      return_intermediates=False, site_tens: 'qtn.Tensor'=None):

        # raise RuntimeError

        if site_tens is not None:
            raise NotImplementedError

        if nsites == 0:
            return self.get_projected_bond(left_site_pos, return_combined=return_combined,
                                           transpose_bonds=transpose_bonds)

        # site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        site_inds = list(range(left_site_pos, left_site_pos + nsites))

        # helper_quimb.check_orthog(self.ket)
        # helper_quimb.check_orthog(self.bra)
        # print(self.envs.keys())

        assert(left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'

        A_left = self.envs[left_site_pos - 1]
        A_right = self.envs[left_site_pos + nsites]

        # A_eff = None
        A_eff = qtn.TensorNetwork([])
        if isinstance(site_inds, (list, tuple)):
            for si in site_inds:
                A_eff.add(self.operator[si].copy())
        else:
            A_eff.add(self.operator[site_inds].copy())

        if A_left is not None:
            A_eff.add(A_left.copy())
        if A_right is not None:
            A_eff.add(A_right.copy())

        A_eff.exponent += self.operator.exponent      ## causes issues?
        # A_eff.exponent = self.operator.exponent

        if return_combined:
            A_eff_tens = qtn.tensor_contract(*A_eff.tensors, preserve_tensor=True)
            A_eff_tens.modify(apply=lambda data: data * 10 ** self.operator.exponent)
            if transpose_bonds is not None:
                A_eff_tens.transpose(*transpose_bonds, inplace=True)
            self._projected_site = A_eff_tens.copy()
            return A_eff_tens

        self._projected_site = A_eff.copy()

        if return_intermediates:
            return A_eff, []

        return A_eff

#####

class BlockVector_DMRG(BlockDMRG, BlockVector):

    def __init__(self, # L: int,
                 ket: 'MPS',
                 bra: 'MPS',
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 # mps_inds: Sequence[int] = None,
                 cur_orthog: int = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        super().__init__(ket, bra, anc_env_left=anc_env_left, anc_env_right=anc_env_right, # mps_inds=mps_inds,
                         cur_orthog=cur_orthog)

    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        """ build left environment or right environment to include tensors at site i
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        # self.decimate(i, direction)
        new_env = super().extend_env(i, direction)
        return new_env


    def get_projected_ket(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor'=None):
        """ returns P X
            includes ket exponent
        """
        # x_eff = self.get_projected(left_site_pos, nsites)
        ## but need to do re-indexing

        k_left = self.envs[left_site_pos - 1]
        k_right = self.envs[left_site_pos + nsites]

        xtens_list = [self.ket[si] for si in range(left_site_pos, left_site_pos + nsites)]
        # if site_tens is None:
        #     xtens_list = [self.ket[si] for si in range(left_site_pos, left_site_pos + nsites)]
        # else:
        #     if isinstance(site_tens, (tuple, list)):
        #         xtens_list = list(*site_tens)
        #     else:
        #         xtens_list = [site_tens]

        if k_left is not None:
            ket_bond_l = next(iter(xtens_list[0].bonds(k_left)))
            bra_bond_l = [ind for ind in k_left.inds if ind != ket_bond_l][0]
        else:
            ket_bond_l, bra_bond_l = None, None

        if k_right is not None:
            ket_bond_r = next(iter(xtens_list[-1].bonds(k_right)))
            bra_bond_r = [ind for ind in k_right.inds if ind != ket_bond_r][0]
        else:
            ket_bond_r, bra_bond_r = None, None

        if k_left is not None:
            xtens_list += [k_left]
        if k_right is not None:
            xtens_list += [k_right]

        x_eff = qtn.tensor_contract(*xtens_list)
        x_eff.modify(apply=lambda data: data * 10 ** self.ket.exponent)
        x_eff.reindex({bra_bond_l: ket_bond_l, bra_bond_r: ket_bond_r}, inplace=True)
        return x_eff


    def decimate(self, pos: int, direction: SweepDirection, new_ket_site: Sequence['qtn.Tensor'] = None,
                 max_bond: int = None):
        ### put into extend env, so that envs are always identity?
        ### canonicalization / decimation probably more important when bra is None

        ind1, ind2 = pos, pos + direction
        at_end = (ind2 == self.L or ind2 == -1)

        if self.bra is None:
            if new_ket_site is None:
                helper.canonize_tens_list(self.ket[ind1], self.ket[ind2], inplace=True,
                                          full_matrices=False)
            else:
                helper_dmrg.update_1site(self.ket, pos, new_ket_site, direction, max_bond=max_bond)
        else:
            ## assumes bra is in the desired canonical form
            pass
            # if new_ket_site is None:
            #     new_ket_site = self.bra[pos].conj()
            #     new_ket_site.reindex(self.projected_bra_to_ket(pos, 1), inplace=True)
            #
            # helper_dmrg.decimate(self.ket, pos, new_ket_site, direction)

        self.cur_orthog = ind2

        return

class BlockVector_Cross(BlockCross, BlockVector):

    def __init__(self, # L: int,
                 ket: 'MPS',
                 bra: 'MPS' = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 cur_orthog: int = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        super().__init__(ket, bra,
                         anc_env_left=anc_env_left, anc_env_right=anc_env_right, # mps_inds=mps_inds,
                         cur_orthog=cur_orthog)

    @property
    def select_inds_bra(self):
        if self.bra is not None:
            return self.bra.select_inds
        else:
            return self.ket.select_inds


    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        """ build left environment or right environment
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        ## new env is ket_and_env[select_inds]
        ket_tens = self.ket.get_tens(i)   # self.ket[self.mps_inds[i]]
        site_ind = self.ket.site_ind(i)   # self.ket.site_ind_id.format(self.mps_inds[i])

        bond_old = self.bra_horizontal_bond(i - direction, direction)
        bond_new = self.bra_horizontal_bond(i, direction)
        bond_ket = self.ket_horizontal_bond(i, direction)

        env_L, env_R = self.envs[i - 1], self.envs[i + 1]
        if direction == SweepDirection.RIGHT:
            if env_L is not None:
                ket_and_env = qtn.tensor_contract(ket_tens, env_L)
            else:
                ket_and_env = ket_tens.copy()
        else:
            if env_R is not None:
                ket_and_env = qtn.tensor_contract(ket_tens, env_R)
            else:
                ket_and_env = ket_tens.copy()

        # print('extend env', i, direction)
        # helper_cross.plot_submat(self.bra, i, 1, ket_tens, select_inds=self.select_inds_bra, ref_ket=self.ket)

        ## env should be the selected columns?
        ## left_env @ ket_tens @ right_env should be the submatrix
        ## select_inds are determined wrt to bra;
        ## which should be fine bc we're selecting from env @ ket_tens;
        ## bra ind should be size of bra so select_inds correctly selects things.
        if bond_old is not None:
            new_env = ket_and_env.fuse({bond_new: [site_ind, bond_old]})
            new_env.transpose(bond_new, bond_ket, inplace=True)
        else:
            new_env = ket_and_env.reindex({site_ind: bond_new})
            new_env.transpose(bond_new, bond_ket, inplace=True)

        select_inds = self.select_inds_bra[i]
        # print('build env select inds', i, select_inds)
        if select_inds is None:
            raise ValueError('need to compute/update select_inds to extend env')

        new_env.modify(apply=lambda x: x[select_inds, :])
        # print('vec new env', new_env.data)
        # print('ket cur orthog', self.ket.cur_orthog, direction)
        # print('check orthog', helper_cross.check_orthog(self.ket, self.ket.select_inds, [self.ket.site_ind_id]))

        self.envs[i] = new_env
        self.cur_orthog = i + direction
        return new_env


    def update_bra(self, left_site_pos: int, nsites: int, direction: SweepDirection, new_site: 'qtn.Tensor' = None,
                   max_bond: int = None):

        if self.bra is None or new_site is None:
            return

        if nsites == 1:
            helper_cross.update_1site(self.bra, left_site_pos, new_site, direction, max_bond=max_bond)
        else:
            helper_cross.update_2site(self.bra, left_site_pos, new_site, direction, max_bond=max_bond)



    def decimate(self, pos: int, direction: SweepDirection, new_ket_site: 'qtn.Tensor' = None, max_bond=None):

        ind1, ind2 = pos, pos + direction
        at_end = (ind2 == self.L or ind2 == -1)

        if at_end:
            return

        if self.bra is None:
            if new_ket_site is None:
                _, sel_inds = helper_cross.canonize_tens_list([self.ket[ind1], self.ket[ind2]],
                                                              [self.ket.site_ind_id], [ind1, ind2],
                                                              inplace=True,
                                                              )
                self.ket.select_inds[ind1] = sel_inds[0]

            else:
                helper_cross.update_1site(self.ket, pos, new_ket_site, direction, max_bond=max_bond,
                                          decimate_only=True)

        else:
            # self.update_bra(pos, 1, direction, new_site=new_ket_site, max_bond=max_bond)
            pass

        self.cur_orthog = ind2

        return



class BlockPowerKet_DMRG(BlockVector_DMRG):

    def __init__(self, # L: int,
                 ket: 'qtn.MatrixProductState',
                 bra: 'qtn.MatrixProductState' = None,
                 # operator: 'qtn.MatrixProductState' = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 # mps_inds: Sequence[int] = None,
                 cur_orthog: int = None,
                 power: int = 2,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        self._ket_envs = {i: None for i in range(-1, ket.L + 1)}
        # if anc_env_left is not None:
        #     self.envs[-1] = anc_env_left.copy()
        # if anc_env_right is not None:
        #     self.envs[L] = anc_env_right.copy()

        self.power = power
        self._pows_list = None

        mpo = helper.mps_to_diag_mpo(ket, upper_ind_id=ket.site_ind_id + '_', lower_ind_id=ket.site_ind_id)
        super(BlockVector_DMRG, self).__init__(ket, bra, operator=mpo, anc_env_left=anc_env_left,
                         anc_env_right=anc_env_right, # mps_inds=mps_inds,
                         cur_orthog=cur_orthog)

        # assert(bra is not None), 'bra needs to be defined for storing output'
        ### this was changed
        # if bra is not None:
        #     self.bra.site_ind_id = self.bra_site_ind()


    @property
    def bra_site_ind(self):
        return self.ket.site_ind_id + '_'


    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        """ env is for P Diag(X) P.T
        """
        assert(self.bra is not None), 'self.bra should not be none'

        ### compute env for original ket <bra | ket>
        bra_tens = self.bra.get_bra_tens(i, reindex_phys=False)
        ket_tens = self.ket[i]

        new_env = qtn.TensorNetwork([bra_tens, ket_tens])
        env_L, env_R = self._ket_envs[i - 1], self._ket_envs[i + 1]
        env = env_L if direction == SweepDirection.RIGHT else env_R
        if env is not None:
            new_env.add(env)
        new_env = new_env.contract_tags(all)
        self._ket_envs[i] = new_env


        ### compute env for the diagonal operator <bra | diag(ket) | bra>
        ### bra is already in canonical form; provides info for decimation
        bra_tens = self.bra.get_bra_tens(i, reindex_phys=True)
        ket_tens = self.bra[i].copy()     ## decimated site defining the projector

        op_tens = self.ket[i].reindex({ind: ind + '_o_' for ind in self.ket[i].inds})
        # op_tens = self.bra[i].reindex({ind: ind + '_o_' for ind in self.bra[i].inds})
        # op_tens.modify(apply=lambda x: x * 10**self.ket.exponent)
        site_ind = self.ket.site_ind_id.format(i)  # self.mps_inds[i])
        d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(i),
                                            (site_ind, site_ind + '_o_', site_ind + '_'),
                                            tags=(f'd_ijk({i})',))

        new_env = qtn.TensorNetwork([bra_tens, ket_tens, op_tens, d_ijk])

        env_L, env_R = self.envs[i - 1], self.envs[i + 1]
        env = env_L if direction == SweepDirection.RIGHT else env_R
        if env is not None:
            new_env.add(env)

        new_env = new_env.contract_tags(all)

        self.envs[i] = new_env
        self.cur_orthog = i + direction
        return new_env


    def get_pows(self, left_site_pos: int, nsites: int, return_combined=True) -> Sequence['qtn.Tensor']:
        """ returns (P diag(X) P.T)**power P X
            includes ket exponent
        """
        return self._pows_list
        # site_inds = list(range(left_site_pos, left_site_pos + nsites))
        # assert (left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
        #     'orthogonality center not within unprojected sites'
        #
        # A_left = self.envs[left_site_pos - 1]
        # A_right = self.envs[left_site_pos + nsites]
        #
        # A_eff = qtn.TensorNetwork([])
        # if isinstance(site_inds, (list, tuple)):
        #     for si in site_inds:
        #         phys_ind = self.ket.site_ind(si)
        #         d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(si),
        #                                             (phys_ind, phys_ind + '_o_', phys_ind + '_'),
        #                                             )
        #         op_tens = self.ket[si].copy()
        #         op_tens.reindex({ind: ind + '_o_' for ind in op_tens.inds}, inplace=True)
        #         A_eff.add([op_tens, d_ijk])
        # else:
        #     phys_ind = self.ket.site_ind(site_inds)
        #     d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(site_inds),
        #                                         (phys_ind, phys_ind + '_o_', phys_ind + '_'),
        #                                         )
        #     op_tens = self.ket[site_inds].copy()
        #     A_eff.add([op_tens, d_ijk])
        #
        # if A_left is not None:
        #     A_eff.add(A_left)
        # if A_right is not None:
        #     A_eff.add(A_right)
        #
        # A_eff.exponent += self.ket.exponent
        #
        # pows_list = []
        #
        # Ax_eff = qtn.tensor_contract(*[self.ket[si] for si in site_inds])
        # Ax_eff.modify(apply=lambda data: data * 10 ** self.ket.exponent)
        #
        # pows_list += [Ax_eff.copy()]
        #
        # it = 1
        # while it < self.power:
        #     Ax_eff = qtn.tensor_contract(*A_eff.tensors, Ax_eff, preserve_tensor=True)
        #     Ax_eff.modify(apply=lambda data: data * 10 ** self.ket.exponent)
        #     Ax_eff.modify(inds=[ind[:-1] for ind in Ax_eff.inds])  ## remove the '_' to go from bra inds -> ket inds
        #     it += 1
        #
        #     pows_list += [Ax_eff.copy()]
        #
        # # if return_combined:
        # #     A_eff_tens = qtn.tensor_contract(*A_eff.tensors, preserve_tensor=True)
        # #     A_eff_tens.modify(apply=lambda data: data * 10 ** self.ket.exponent)
        # #     if transpose_bonds is not None:
        # #         A_eff_tens.transpose(*transpose_bonds, inplace=True)
        # #     return A_eff_tens
        #
        # self._pows_list = pows_list
        #
        # return pows_list


    def get_projected_ket(self, left_site_pos: int, nsites: int, site_tens:'qtn.Tensor'=None):
        """ returns P X
            includes ket exponent
        """
        k_left = self._ket_envs[left_site_pos - 1]
        k_right = self._ket_envs[left_site_pos + nsites]

        # xtens_list = [self.ket[si] for si in range(left_site_pos, left_site_pos + nsites)]
        if site_tens is None:
            xtens_list = [self.ket[si] for si in range(left_site_pos, left_site_pos + nsites)]
            # site_tens = qtn.tensor_contract(*[self.ket[si] for si in range(left_site_pos, left_site_pos + nsites)])
        else:
            if isinstance(site_tens, (tuple, list)):
                xtens_list = list(*site_tens)
            else:
                xtens_list = [site_tens]

        if k_left is not None:
            ket_bond_l = next(iter(xtens_list[0].bonds(k_left)))
            bra_bond_l = [ind for ind in k_left.inds if ind != ket_bond_l][0]
        else:
            ket_bond_l, bra_bond_l = None, None

        if k_right is not None:
            ket_bond_r = next(iter(xtens_list[-1].bonds(k_right)))
            bra_bond_r = [ind for ind in k_right.inds if ind != ket_bond_r][0]
        else:
            ket_bond_r, bra_bond_r = None, None

        if k_left is not None:
            xtens_list += [k_left]
        if k_right is not None:
            xtens_list += [k_right]

        x_eff = qtn.tensor_contract(*xtens_list)
        x_eff.modify(apply=lambda data: data * 10 ** self.ket.exponent)
        x_eff.reindex({bra_bond_l: ket_bond_l, bra_bond_r: ket_bond_r}, inplace=True)

        # proj_ket = self.bra.copy()
        # x_eff.transpose_like(proj_ket[left_site_pos], inplace=True)
        # proj_ket[left_site_pos].modify(data=x_eff.data)

        # plt.figure()
        # plt.plot(helper.to_dense(proj_ket), label='proj ket')
        # plt.plot(helper.to_dense(self.ket), label='ket')
        # plt.legend()
        # plt.title('get proj ket')
        # plt.show()


        return x_eff


    def get_projected(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
                      return_intermediates=False, site_tens: qtn.Tensor = None):
        """ returns (P diag(X) P.T)**power P X
            includes ket exponent
            removes bra exponent
        """
        if getattr(self, 'verbose', 0):
            print('PowerKet get projected')
        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        assert (left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'

        if site_tens is None:
            xeff = self.get_projected_ket(left_site_pos, nsites)  ## includes ket exponent
        else:
            # xeff = site_tens
            xeff = self.get_projected_ket(left_site_pos, nsites, site_tens=site_tens)  ## includes ket exponent

        self._projected_site = xeff

        A_left = self.envs[left_site_pos - 1]
        A_right = self.envs[left_site_pos + nsites]

        A_eff = qtn.TensorNetwork([])
        if isinstance(site_inds, (list, tuple)):

            # op_tens = xeff.copy()  ## includes ket exponent
            op_tens = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
            op_tens.modify(apply=lambda x: x * 10**self.ket.exponent)
            op_tens.reindex({ind: ind + '_o_' for ind in op_tens.inds}, inplace=True)
            tens_list = [op_tens]

            for si in site_inds:

                phys_ind = self.ket.site_ind(si)
                d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(si),
                                                    (phys_ind, phys_ind + '_o_', phys_ind + '_'),
                                                    )
                tens_list += [d_ijk]

            A_eff.add(tens_list)
        else:
            phys_ind = self.ket.site_ind(site_inds)
            d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(site_inds),
                                                (phys_ind, phys_ind + '_o_', phys_ind + '_'),
                                                )
            op_tens = xeff.copy()  # self.ket[site_inds].copy()
            A_eff.add([op_tens, d_ijk])

        if A_left is not None:
            A_eff.add(A_left)
        if A_right is not None:
            A_eff.add(A_right)

        A_eff.exponent += self.ket.exponent

        # Ax_eff = qtn.tensor_contract(*[self.ket[si] for si in site_inds])
        # Ax_eff.modify(apply=lambda data: data * 10 ** self.ket.exponent)
        Ax_eff = xeff.copy()
        pows_list = [Ax_eff.copy()]

        it = 1
        while it < self.power:
            Ax_eff = qtn.tensor_contract(*A_eff.tensors, Ax_eff, preserve_tensor=True)
            # Ax_eff.modify(apply=lambda data: data * 10 ** self.ket.exponent)
            Ax_eff.modify(inds=[ind[:-1] for ind in Ax_eff.inds])   ## remove the '_' to go from bra inds -> ket inds
            it += 1

            pows_list += [Ax_eff.copy()]

        self._pows_list = pows_list

        # if return_combined:
        #     A_eff_tens = qtn.tensor_contract(*A_eff.tensors, preserve_tensor=True)
        #     A_eff_tens.modify(apply=lambda data: data * 10 ** self.ket.exponent)
        #     if transpose_bonds is not None:
        #         A_eff_tens.transpose(*transpose_bonds, inplace=True)
        #     return A_eff_tens

        # self._projected_site = Ax_eff

        # ## remove bra exponent
        # Ax_eff.modify(apply=lambda x: x * 10 ** (-self.bra.exponent))
        # for t in pows_list:
        #     t.modify(apply=lambda x: x * 10 ** (-self.bra.exponent))

        if return_intermediates:
            mod_pows_list = [t.copy() for t in pows_list]
            for t in mod_pows_list:
                t.modify(apply=lambda x: x / t.norm())
            return Ax_eff, mod_pows_list       ## no intermediates

        return Ax_eff


    def update_bra(self, left_site_pos: int, nsites: int, direction: SweepDirection,
                   new_site: Sequence['qtn.Tensor'] = None, max_bond: int = None):
        ### recall that we haven't taken complex conj of self.bra -- that is done as needed

        if direction == SweepDirection.RIGHT:
            ind1, ind2 = left_site_pos, left_site_pos + 1
        else:
            if nsites == 1:
                ind1, ind2 = left_site_pos, left_site_pos - 1
            elif nsites == 2:
                ind1, ind2 = left_site_pos, left_site_pos + 1
            else:
                raise NotImplementedError

        at_end = (ind2 == self.L or ind2 == -1)

        # if new_site is None:
        #     print('get proj pos', left_site_pos, self.cur_orthog)
        #     new_site = self.get_projected(left_site_pos, 1, return_combined=True)

        if self.bra is None:
            return

        if new_site is None:
            if nsites == 1:
                ind1, ind2 = left_site_pos, left_site_pos + direction
            elif nsites == 2:
                if direction == SweepDirection.RIGHT:
                    ind1, ind2 = left_site_pos, left_site_pos + 1
                else:
                    ind1, ind2 = left_site_pos + 1, left_site_pos
            else:
                raise NotImplementedError
            helper.canonize_tens_list(self.bra[ind1], self.bra[ind2])

        else:
            if nsites == 1:
                helper_dmrg.update_1site(self.bra, left_site_pos, new_site, direction, max_bond=max_bond)
            else:
                raise NotImplementedError


    def decimate(self, pos: int, direction: SweepDirection, new_ket_site: Sequence['qtn.Tensor'] = None, max_bond=None):

        ind1, ind2 = pos, pos + direction
        at_end = (ind2 == self.L or ind2 == -1)

        if at_end:
            return

        if self.bra is None:
            if new_ket_site is not None:
                helper_dmrg.update_1site(self.ket, pos, new_ket_site, direction=direction, max_bond=max_bond)
            else:
                super().decimate(pos, direction, new_ket_site=None, max_bond=max_bond)
        else:
            ## assumes bra is already updated
            super().decimate(pos, direction, new_ket_site=new_ket_site, max_bond=max_bond)


class BlockPowerOpKet_DMRG(BlockPowerKet_DMRG):

    def __init__(self, # L: int,
                 ket: 'qtn.MatrixProductState',
                 bra: 'qtn.MatrixProductState' = None,
                 operator: 'qtn.MatrixProductOperator' = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 # mps_inds: Sequence[int] = None,
                 cur_orthog: int = None,
                 power: int = 2,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        if operator is not None:
            operator.upper_ind_id = ket.site_ind_id + '_'
            operator.lower_ind_id = ket.site_ind_id
        self.operator_k = operator

        super().__init__(ket, bra, anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                         cur_orthog=cur_orthog, power=power)


    def exponent(self):
        return self.ket.exponent + self.operator_k.exponent

    def copy(self, ket_copy=None, bra_copy=None):
        new_obj = super().copy(ket_copy=ket_copy, bra_copy=bra_copy)
        new_obj.operator_k = self.operator_k
        return new_obj

    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        """ env is for P Diag(AX) P.T
        """
        assert(self.bra is not None), 'self.bra should not be none'

        ### compute env for original ket
        bra_tens = self.bra.get_bra_tens(i, reindex_phys=True)
        ket_tens = self.ket[i]
        opk_tens = self.operator_k[i]

        new_env = qtn.TensorNetwork([bra_tens, opk_tens, ket_tens])
        env_L, env_R = self._ket_envs[i - 1], self._ket_envs[i + 1]
        env = env_L if direction == SweepDirection.RIGHT else env_R
        if env is not None:
            new_env.add(env)
        new_env = new_env.contract_tags(all)
        self._ket_envs[i] = new_env


        ### compute env for the diagonal operator
        ### bra is already in canonical form; provides info for decimation
        bra_tens = self.bra.get_bra_tens(i, reindex_phys=True)
        ket_tens = self.bra[i]     ## decimated site defining the projector, already captures A*ket

        site_ind = self.ket.site_ind_id.format(i)
        op_tens = ket_tens.reindex({ind: ind + '_o_' for ind in ket_tens.inds})     ## A * ket
        # opk_tens = opk_tens.reindex({ind: ind + '_ok_' for ind in opk_tens.inds})
        # opk_tens.reindex({site_ind + '_ok_': site_ind + '_o_'}, inplace=True)

        d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(i),
                                            # (site_ind, site_ind + '_' + '_ok_', site_ind + '_'),
                                            (site_ind, site_ind + '_o_', site_ind + '_'),
                                            tags=(f'd_ijk({i})',))

        # print('diag op env')
        # print('bra', bra_tens)
        # print('ket', ket_tens)
        # print('op tens', op_tens)
        # print('opk_tens', opk_tens)
        # print('dijk', d_ijk)

        # new_env = qtn.TensorNetwork([bra_tens, ket_tens, op_tens, opk_tens, d_ijk])
        new_env = qtn.TensorNetwork([bra_tens, ket_tens, op_tens, d_ijk])

        env_L, env_R = self.envs[i - 1], self.envs[i + 1]
        env = env_L if direction == SweepDirection.RIGHT else env_R
        if env is not None:
            new_env.add(env)

        new_env = new_env.contract_tags(all)

        # print('new env', new_env)
        # exit()

        self.envs[i] = new_env
        self.cur_orthog = i + direction
        return new_env


    def get_pows(self, left_site_pos: int, nsites: int, return_combined=True) -> Sequence['qtn.Tensor']:
        """ returns (P diag(X) P.T)**power P X
            includes ket exponent
        """
        return self._pows_list


    def get_projected_ket(self, left_site_pos: int, nsites: int, site_tens:'qtn.Tensor'=None):
        """ returns P A*X
            includes ket exponent
        """
        k_left = self._ket_envs[left_site_pos - 1]
        k_right = self._ket_envs[left_site_pos + nsites]

        if site_tens is None:
            xtens_list = [self.ket[si] for si in range(left_site_pos, left_site_pos + nsites)]
        else:
            site_tens = site_tens.copy()
            site_tens.modify(apply=lambda x: x * 10**-self.ket.exponent)
            xtens_list = [site_tens]

        # if site_tens is None:
        #     xtens_list = [self.ket[si] for si in range(left_site_pos, left_site_pos + nsites)]
        # else:
        #     if isinstance(site_tens, (tuple, list)):
        #         xtens_list = list(*site_tens)
        #     else:
        #         xtens_list = [site_tens]

        if k_left is not None:
            ket_bond_l = next(iter(xtens_list[0].bonds(k_left)))
            bra_bond_l = [ind for ind in k_left.inds if ind != ket_bond_l][0]
        else:
            ket_bond_l, bra_bond_l = None, None

        if k_right is not None:
            ket_bond_r = next(iter(xtens_list[-1].bonds(k_right)))
            bra_bond_r = [ind for ind in k_right.inds if ind != ket_bond_r][0]
        else:
            ket_bond_r, bra_bond_r = None, None

        # print('k left', k_left)
        # print('k right', k_right)

        if k_left is not None:
            xtens_list += [k_left]
        if k_right is not None:
            xtens_list += [k_right]

        ## self.operator_k tensors
        xtens_list += [self.operator_k[si] for si in range(left_site_pos, left_site_pos + nsites)]

        # print('xtens list', xtens_list)
        # exit()

        x_eff = qtn.tensor_contract(*xtens_list)
        x_eff.modify(apply=lambda data: data * 10 ** (self.ket.exponent + self.operator_k.exponent))

        u_ind, l_ind = self.bra_site_ind, self.ket.site_ind_id
        x_eff.reindex({**{bra_bond_l: ket_bond_l, bra_bond_r: ket_bond_r},
                       **{u_ind.format(si): l_ind.format(si) for si in range(left_site_pos, left_site_pos + nsites)}},
                      inplace=True)
        return x_eff


    def get_projected(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
                      return_intermediates=False, site_tens: qtn.Tensor = None):
        """ returns (P diag(A*X) P.T)**power P A*X
            includes ket exponent
            removes bra exponent
        """
        if getattr(self, 'verbose', 0):
            print('PowerOpKet get projected')
            print(self.ket.exponent, self.operator_k.exponent)

        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        assert (left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'

        if site_tens is None:
            xeff = self.get_projected_ket(left_site_pos, nsites)  ## includes ket + op_k exponent
        else:
            # xeff = site_tens # .copy()
            xeff = self.get_projected_ket(left_site_pos, nsites, site_tens=site_tens)  ## includes ket + op_k exp

        A_left = self.envs[left_site_pos - 1]
        A_right = self.envs[left_site_pos + nsites]

        A_eff = qtn.TensorNetwork([])
        if isinstance(site_inds, (list, tuple)):

            op_tens = xeff.copy()  ## includes ket + op_k exponent
            op_tens.reindex({ind: ind + '_o_' for ind in op_tens.inds}, inplace=True)
            tens_list = [op_tens]

            for si in site_inds:
                phys_ind = self.ket.site_ind(si)

                # ## self.operator_k tensors    ## already included in xeff/op_tens
                # opk_tens = self.operator_k[si].copy()
                # opk_tens.reindex({ind: ind + '_ok_' for ind in opk_tens.inds}, inplace=True)
                # opk_tens.reindex({phys_ind + '_ok_': phys_ind + '_o_'}, inplace=True)

                ## copy tensor
                d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(si),
                                                    # (phys_ind, phys_ind + '__ok_', phys_ind + '_'),
                                                    (phys_ind, phys_ind + '_o_', phys_ind + '_'),
                                                    )
                tens_list += [d_ijk] # , opk_tens]

            A_eff.add(tens_list)
        else:
            phys_ind = self.ket.site_ind(site_inds)

            # ## self.operator_k tensors    ## already included in xeff/op_tens
            # opk_tens = self.operator_k[site_inds].copy()
            # opk_tens.reindex({ind: ind + '_ok_' for ind in opk_tens.inds}, inplace=True)
            # opk_tens.reindex({phys_ind + '_ok_': phys_ind + '_o_'}, inplace=True)

            ## copy tensor
            d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(site_inds),
                                                (phys_ind, phys_ind + '_o_', phys_ind + '_'),
                                                )
            op_tens = xeff.copy()  # self.ket[site_inds].copy()
            A_eff.add([op_tens, d_ijk]) #, opk_tens])

        if A_left is not None:
            A_eff.add(A_left)
        if A_right is not None:
            A_eff.add(A_right)

        A_eff.exponent += self.ket.exponent + self.operator_k.exponent

        # Ax_eff = qtn.tensor_contract(*[self.ket[si] for si in site_inds])
        # Ax_eff.modify(apply=lambda data: data * 10 ** self.ket.exponent)
        Ax_eff = xeff.copy()
        pows_list = [Ax_eff.copy()]

        it = 1
        while it < self.power:
            Ax_eff = qtn.tensor_contract(*A_eff.tensors, Ax_eff, preserve_tensor=True)
            # Ax_eff.modify(apply=lambda data: data * 10 ** self.ket.exponent)
            Ax_eff.modify(inds=[ind[:-1] for ind in Ax_eff.inds])   ## remove the '_' to go from bra inds -> ket inds
            it += 1

            pows_list += [Ax_eff.copy()]

        self._pows_list = pows_list


        if return_intermediates:
            mod_pows_list = [t.copy() for t in pows_list]
            for t in mod_pows_list:
                t.modify(apply=lambda x: x / t.norm())
            return Ax_eff, mod_pows_list       ## no intermediates

        return Ax_eff



class BlockDiagKet_DMRG(BlockDMRG, BlockOperator):

    def __init__(self, # L: int,
                 ket: 'qtn.MatrixProductState',
                 bra: 'qtn.MatrixProductState' = None,
                 # operator: 'qtn.MatrixProductState',
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 # mps_inds: Sequence[int] = None,
                 cur_orthog: int = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        mpo = helper.mps_to_diag_mpo(ket, upper_ind_id=ket.site_ind_id + '_', lower_ind_id=ket.site_ind_id)
        # bra = ket.conj()
        # for tens in bra.tensors:
        #     tens.reindex({ind: ind + '_' for ind in tens.inds}, inplace=True)
        super().__init__(ket, bra, operator=mpo, anc_env_left=anc_env_left,
                         anc_env_right=anc_env_right, # mps_inds=mps_inds,
                         cur_orthog=cur_orthog)

        if bra is not None:
            self.bra.site_ind_id = self.bra_site_ind()

    @property
    def bra_site_ind(self):
        return self.ket.site_ind_id + '_'

    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        ### assuming ket is already in canonical form

        ket_tens = self.ket[i]  # [self.mps_inds[i]]
        # bra_tens = self.bra_site(i)
        if self.bra is None:
            # bra_tens = ket_tens.conj()
            # bra_tens.modify(inds = [ind + '_' for ind in bra_tens.inds])
            bra_tens = self.ket.get_bra_tens(i, reindex_phys=True)
        else:
            bra_tens = self.bra.get_bra_tens(i, reindex_phys=True)  # [self.mps_inds[i]]
        op_tens = ket_tens.reindex({ind: ind + '_o_' for ind in ket_tens.inds})
        site_ind = self.ket.site_ind_id.format(i)  # self.mps_inds[i])
        d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(i),
                                            (site_ind, site_ind + '_o_', site_ind + '_'),
                                            tags=(f'd_ijk({i})',))

        new_env = qtn.TensorNetwork([bra_tens, ket_tens, op_tens, d_ijk])

        env_L, env_R = self.envs[i - 1], self.envs[i + 1]
        env = env_L if direction == SweepDirection.RIGHT else env_R
        if env is not None:
            new_env.add(env)

        new_env = new_env.contract_tags(all)

        self.envs[i] = new_env
        self.cur_orthog = i + direction
        return new_env

    def get_projected(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
                      return_intermediates=False, site_tens: 'qtn.Tensor'=None):
        # site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        assert (left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'
        # print('get projected', left_site_pos, nsites)
        # helper.check_orthog(self.ket)
        # print('diag block ket', self.ket)

        A_left = self.envs[left_site_pos - 1]
        A_right = self.envs[left_site_pos + nsites]

        if not isinstance(site_inds, (list, tuple)):
            site_inds = [site_inds]
            site_tens = [site_tens] if site_tens is not None else [self.ket[si] for si in site_inds]
        else:
            if site_tens is None:
                site_tens = [self.ket[si] for si in site_inds]

        A_eff = qtn.TensorNetwork([])

        for si, stens in zip(site_inds, site_tens):
            phys_ind = self.ket.site_ind(si)
            d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(si),
                                                (phys_ind, phys_ind + '_o_', phys_ind + '_'),
                                                )
            op_tens = stens.copy()
            op_tens.reindex({ind: ind + '_o_' for ind in op_tens.inds}, inplace=True)
            A_eff.add([op_tens, d_ijk])

        # if isinstance(site_inds, (list, tuple)):
        #     for si in site_inds:
        #
        #         phys_ind = self.ket.site_ind(si)
        #         d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(si),
        #                                             (phys_ind, phys_ind + '_o_', phys_ind + '_'),
        #                                             )
        #         op_tens = self.ket[si].copy()
        #         op_tens.reindex({ind: ind + '_o_' for ind in op_tens.inds}, inplace=True)
        #         A_eff.add([op_tens, d_ijk])
        # else:
        #     phys_ind = self.ket.site_ind(site_inds)
        #     d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(site_inds),
        #                                         (phys_ind, phys_ind + '_o_', phys_ind + '_'),
        #                                         )
        #     op_tens = self.ket[site_inds].copy()
        #     A_eff.add([op_tens, d_ijk])

        if A_left is not None:
            A_eff.add(A_left)
        if A_right is not None:
            A_eff.add(A_right)

        A_eff.exponent += self.ket.exponent

        if return_combined:
            A_eff_tens = qtn.tensor_contract(*A_eff.tensors, preserve_tensor=True)
            A_eff_tens.modify(apply=lambda data: data * 10 ** self.ket.exponent)
            if transpose_bonds is not None:
                A_eff_tens.transpose(*transpose_bonds, inplace=True)
            A_eff = A_eff_tens

        if return_intermediates:
            return A_eff, []

        return A_eff

    def projected_bra_to_ket(self, left_site_pos: int, nsites: int):
        inds_dict = {self.bra_horizontal_bond(left_site_pos, -1):
                         self.ket_horizontal_bond(left_site_pos, -1),
                     self.bra_horizontal_bond(left_site_pos + nsites - 1, 1):
                         self.ket_horizontal_bond(left_site_pos + nsites - 1, 1)}
        for ix in range(left_site_pos, left_site_pos + nsites):
            inds_dict[self.ket.site_ind(ix) + '_'] = self.ket.site_ind(ix)
        return inds_dict


class BlockOperator_DMRG(BlockDMRG, BlockOperator):

    def __init__(self, # L: int,
                 ket: 'qtn.MatrixProductState',
                 bra: 'qtn.MatrixProductState',
                 operator: 'qtn.MatrixProductOperator' = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 # mps_inds: Sequence[int] = None,
                 cur_orthog: int = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        super().__init__(ket, bra=bra, operator=operator, anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                         # mps_inds=mps_inds,
                         cur_orthog=cur_orthog)


class BlockOperator_Cross(BlockCross, BlockOperator):

    def __init__(self, # L: int,
                 ket: 'MPS',
                 bra: 'MPS',
                 operator: 'qtn.MatrixProductOperator' = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 # mps_inds: Sequence[int] = None,
                 # select_inds_bra: dict[int, Optional[Sequence[int]]] = None,
                 # select_inds_ket: dict[int, Optional[Sequence[int]]] = None,
                 cur_orthog: int = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
            select_inds_bra/ket: inds to select when moving canonical site from i to i +/- 1 (decimation)
                assumes select_inds are correctly set regardless of decimation direction
        """
        super().__init__(ket, bra, operator=operator,
                         anc_env_left=anc_env_left, anc_env_right=anc_env_right, # mps_inds=mps_inds,
                         # select_inds_bra=select_inds_bra, select_inds_ket=select_inds_ket,
                         cur_orthog=cur_orthog)

    # def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:  #, select_inds=None) -> qtn.Tensor:
    #     """ build left environment or right environment
    #         assumes bra, ket are properly canonicalized, bra already cc'ed if specified
    #         pos: position of env to extend
    #     """
    #     ## new env is ket_and_env[select_inds]
    #     op_tens = self.operator[self.mps_inds[i]]
    #     upper_ind = self.operator.upper_ind_id.format(self.mps_inds[i])
    #     lower_ind = self.operator.lower_ind_id.format(self.mps_inds[i])
    #
    #     bra_old = self.bra_horizontal_bond(i - direction, direction)
    #     ket_old = self.ket_horizontal_bond(i - direction, direction)
    #     op_old = self.operator.bond(i - direction, i) if bra_old is not None else None
    #     bra_new = self.bra_horizontal_bond(i, direction)
    #     ket_new = self.ket_horizontal_bond(i, direction)
    #     op_new = self.operator.bond(i, i + direction)
    #
    #     envL, envR = self.envs[i - 1], self.envs[i + 1]
    #     env_tens = envL if direction == SweepDirection.RIGHT else envR
    #     if env_tens is not None:
    #         ket_and_env = qtn.tensor_contract(op_tens, env_tens)
    #     else:
    #         ket_and_env = op_tens.copy()
    #
    #     if bra_old is not None:
    #         new_env = ket_and_env.fuse({bra_new: [upper_ind, bra_old],
    #                                     ket_new: [lower_ind, ket_old]})
    #         new_env.transpose(bra_new, ket_new, op_new, inplace=True)
    #     else:
    #         new_env = ket_and_env.reindex({upper_ind: bra_new,
    #                                        lower_ind: ket_new})
    #         new_env.transpose(bra_new, ket_new, op_new, inplace=True)
    #
    #     bra_select_inds = self.select_inds_bra[i]
    #     # bra_select_inds = self.select_inds_ket[i]
    #     if bra_select_inds is None:
    #         raise ValueError('need to compute/update bra (select_inds) to extend env')
    #
    #     ket_select_inds = self.select_inds_ket[i]
    #     if ket_select_inds is None:
    #         raise ValueError('need to compute/update ket (select_inds) to extend env')
    #
    #     # op_ind = next(iter(new_env.bonds(op_tens)))
    #     new_env.transpose(bra_new, ket_new, op_new, inplace=True)
    #     new_env.modify(apply=lambda x: (x[bra_select_inds, :, :])[:, ket_select_inds, :])
    #     # print('data', new_env.data.ndim)
    #     # new_data = new_env.data[bra_select_inds, :, :]
    #     # new_data = new_data[:, ket_select_inds, :]
    #     # new_env.modify(data=new_data)
    #
    #     self.envs[i] = new_env
    #     self.cur_orthog = i + direction
    #     return new_env


    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:  #, select_inds=None) -> qtn.Tensor:
        """ build left environment or right environment
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        # print('op block extend', self.ket is self.bra)
        # print('check orthog', helper_cross.check_orthog(self.ket, self.ket.select_inds, [self.ket.site_ind_id]))

        ## new env is ket_and_env[select_inds]
        op_tens = self.operator[i]  # [self.mps_inds[i]]
        upper_ind = self.operator.upper_ind_id.format(i)  # (self.mps_inds[i])
        lower_ind = self.operator.lower_ind_id.format(i)  # (self.mps_inds[i])

        bra_old = self.bra_horizontal_bond(i - direction, direction)
        ket_old = self.ket_horizontal_bond(i - direction, direction)
        op_old = self.operator.bond(i - direction, i) if bra_old is not None else None
        bra_new = self.bra_horizontal_bond(i, direction)
        ket_new = self.ket_horizontal_bond(i, direction)
        op_new = self.operator.bond(i, i + direction)

        envL, envR = self.envs[i - 1], self.envs[i + 1]
        env_tens = envL if direction == SweepDirection.RIGHT else envR
        ket_tens = self.ket[i]   ## tensor with selected "columns" / "rows"
        if env_tens is not None:
            ket_and_env = qtn.tensor_contract(op_tens, env_tens, ket_tens)
        else:
            ket_and_env = qtn.tensor_contract(op_tens, ket_tens)

        if bra_old is not None:
            new_env = ket_and_env.fuse({bra_new: [upper_ind, bra_old]})
            new_env.transpose(bra_new, ket_new, op_new, inplace=True)
        else:
            new_env = ket_and_env.reindex({upper_ind: bra_new,
                                           })
            new_env.transpose(bra_new, ket_new, op_new, inplace=True)

        # bra_select_inds = self.select_inds_bra[i]
        bra_select_inds = self.bra.select_inds[i]
        if bra_select_inds is None:
            raise ValueError('need to compute/update bra (select_inds) to extend env')

        # ket_select_inds = self.select_inds_ket[i]
        # if ket_select_inds is None:
        #     raise ValueError('need to compute/update ket (select_inds) to extend env')

        # op_ind = next(iter(new_env.bonds(op_tens)))
        new_env.transpose(bra_new, ket_new, op_new, inplace=True)
        new_env.modify(apply=lambda x: x[bra_select_inds, :, :])
        # print('data', new_env.data.ndim)
        # new_data = new_env.data[bra_select_inds, :, :]
        # new_data = new_data[:, ket_select_inds, :]
        # new_env.modify(data=new_data)

        self.envs[i] = new_env
        self.cur_orthog = i + direction
        return new_env


class BlockDiagOperator(BlockOperator, ABC):

    _mps_to_mpo_operator = None

    # def __init__(self, L: int,
    #              ket: 'qtn.MatrixProductState',
    #              operator: 'qtn.MatrixProductState',
    #              bra: 'qtn.MatrixProductState' = None,
    #              anc_env_left: 'qtn.Tensor' = None,
    #              anc_env_right: 'qtn.Tensor' = None,
    #              mps_inds: Sequence[int] = None,
    #              cur_orthog: int = None,
    #              mps_to_mpo_operator: 'MatrixProductTensor' = None,
    #              ):
    #     """ initialized assuming that all the indices are properly aligned
    #         side: which side of canonical site the environment corresponds to
    #         ket: MPS for ket test wavefunction
    #         bra: MPS for bra test wavefunction
    #     """
    #     # mpo = helper.mps_to_diag_mpo(ket, upper_ind_id=ket.site_ind_id + '_', lower_ind_id=ket.site_ind_id)
    #     ## mostly for correct initialization
    #     super().__init__(L, ket, bra, operator=None, anc_env_left=anc_env_left,
    #                      anc_env_right=anc_env_right, mps_inds=mps_inds, cur_orthog=cur_orthog)
    #     self.operator = operator
    #     if bra is not None:
    #         self.bra.site_ind_id = self.bra_site_ind()
    #     self.mps_to_mpo_operator = mps_to_mpo_operator

    @property
    def bra_site_ind(self):
        return self.ket.site_ind_id + '_'

    @property
    def operator(self) -> 'qtn.MatrixProductState':
        return self._operator

    @operator.setter
    def operator(self, mps: Optional[qtn.MatrixProductState]):
        if mps is self.ket:
            mps = None

        if mps is not None:
            self._operator = mps
            helper_quimb.match_inner_inds(mps, self.ket, append='_o_', inplace=True)
            # mpo.upper_ind_id = self.ket.site_ind_id + '_' if self.bra is None else self.bra.site_ind_id
            # mpo.lower_ind_id = self.ket.site_ind_id
        return

    @property
    def mps_to_mpo_operator(self) -> 'MatrixProductTensor':
        return self._mps_to_mpo_operator

    @mps_to_mpo_operator.setter
    def mps_to_mpo_operator(self, mptn: 'MatrixProductTensor'):
        if mptn is not None:
            mptn = mptn.copy()
            extra_inds = mptn.extra_ind_ids
            assert len(extra_inds) == 1, 'need an MPT with 3 physical inds per tensor'
            mptn.upper_ind_id = self.bra_site_ind()
            mptn.lower_ind_id = self.ket.site_ind_id()
            mptn.reindex_extra_inds(extra_inds[0], self.ket.site_ind_id + '_o_')
            self._mps_to_mpo_operator = mptn
        else:
            self._mps_to_mpo_operator = None


    def get_dijk(self, i: int):
        if self.mps_to_mpo_operator is None:
            site_ind = self.ket.site_ind_id.format(i)  # self.mps_inds[i])
            d_ijk = qtn.tensor_core.COPY_tensor(self.ket.phys_dim(i),
                                                (site_ind, site_ind + '_o_', site_ind + '_'),
                                                tags=(f'd_ijk({i})',))
            return d_ijk
        else:
            return self.mps_to_mpo_operator[i]

    def get_op_tens(self, i: int):
        if self.operator is None:
            op_tens = self.ket[i].copy()
            op_tens.reindex({ind: ind + '_o_' for ind in op_tens.inds}, inplace=True)
        else:
            op_tens = self.operator[i].copy()
            op_tens.reindex({self.operator.site_ind(i): self.ket.site_ind(i) + '_o_'}, inplace=True)
        return op_tens

    def get_projected(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
                      return_intermediates=False, site_tens: 'qtn.Tensor'= None):

        if site_tens is not None:
            raise NotImplementedError

        # site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        assert (left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'
        # helper.check_orthog(self.ket)
        # print('diag block ket', self.ket)

        A_left = self.envs[left_site_pos - 1]
        A_right = self.envs[left_site_pos + nsites]

        A_eff = qtn.TensorNetwork([])
        for si in site_inds:

            d_ijk = self.get_dijk(si)
            op_tens = self.get_op_tens(si)
            A_eff.add([op_tens, d_ijk])

        if A_left is not None:
            A_eff.add(A_left)
        if A_right is not None:
            A_eff.add(A_right)

        A_eff.exponent += (self.operator.exponent if self.operator is not None else self.ket.exponent)

        if return_combined:
            A_eff_tens = qtn.tensor_contract(*A_eff.tensors, preserve_tensor=True)
            A_eff_tens.modify(apply=lambda data: data * 10 ** A_eff.exponent)  # self.ket.exponent)
            if transpose_bonds is not None:
                A_eff_tens.transpose(*transpose_bonds, inplace=True)
            A_eff = A_eff_tens

        if return_intermediates:
            return A_eff, []

        return A_eff

    def projected_bra_to_ket(self, left_site_pos: int, nsites: int):
        inds_dict = {self.bra_horizontal_bond(left_site_pos, -1):
                         self.ket_horizontal_bond(left_site_pos, -1),
                     self.bra_horizontal_bond(left_site_pos + nsites - 1, 1):
                         self.ket_horizontal_bond(left_site_pos + nsites - 1, 1)}
        for ix in range(left_site_pos, left_site_pos + nsites):
            inds_dict[self.ket.site_ind(ix) + '_'] = self.ket.site_ind(ix)
        return inds_dict



class BlockDiagOperator_DMRG(BlockDMRG, BlockDiagOperator):

    def __init__(self, # L: int,
                 ket: 'qtn.MatrixProductState',
                 bra: Optional['qtn.MatrixProductState'],
                 operator: 'qtn.MatrixProductState' = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 # mps_inds: Sequence[int] = None,
                 cur_orthog: int = None,
                 mps_to_mpo_operator: Optional['MatrixProductTensor'] = None,
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
        """
        super().__init__(ket, bra=bra, operator=operator, anc_env_left=anc_env_left, anc_env_right=anc_env_right,
                         # mps_inds=mps_inds,
                         cur_orthog=cur_orthog)

        self.operator: 'qtn.MatrixProductState' = operator
        self.mps_to_mpo_operator = mps_to_mpo_operator


    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        ### assuming ket is already in canonical form

        ket_tens = self.ket.get_tens(i)  # [self.mps_inds[i]]
        if self.bra is None:
            # bra_tens = ket_tens.conj()
            # bra_tens.modify(inds=[ind + '_' for ind in bra_tens.inds])
            bra_tens = self.ket.get_bra_tens(i, reindex_phys=True)
        else:
            bra_tens = self.bra.get_bra_tens(i, reindex_phys=True) # [self.mps_inds[i]]

        op_tens = self.get_op_tens(i)
        d_ijk = self.get_dijk(i)
        new_env = qtn.TensorNetwork([bra_tens, ket_tens, op_tens, d_ijk])

        env_L, env_R = self.envs[i - 1], self.envs[i + 1]
        env = env_L if direction == SweepDirection.RIGHT else env_R
        if env is not None:
            new_env.add(env)

        new_env = new_env.contract_tags(all)

        self.envs[i] = new_env
        self.cur_orthog = i + direction
        return new_env


class BlockDiagOperator_Cross(BlockCross, BlockDiagOperator):

    def __init__(self, # L: int,
                 ket: Union['MPS','qtn.MatrixProductState'],
                 bra: Optional[Union['MPS','qtn.MatrixProductState']],
                 operator: Union['MPS','qtn.MatrixProductState'] = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 # mps_inds: Sequence[int] = None,
                 # select_inds_bra: dict[int, Optional[Sequence[int]]] = None,
                 # select_inds_ket: dict[int, Optional[Sequence[int]]] = None,
                 cur_orthog: int = None,
                 mps_to_mpo_operator: Optional['MatrixProductTensor'] = None
                 ):
        """ initialized assuming that all the indices are properly aligned
            side: which side of canonical site the environment corresponds to
            ket: MPS for ket test wavefunction
            bra: MPS for bra test wavefunction
            select_inds_bra/ket: inds to select when moving canonical site from i to i +/- 1 (decimation)
                assumes select_inds are correctly set regardless of decimation direction
        """
        super().__init__(ket, bra, operator=operator,
                         anc_env_left=anc_env_left, anc_env_right=anc_env_right, # mps_inds=mps_inds,
                         # select_inds_bra=select_inds_bra, select_inds_ket=select_inds_ket,
                         cur_orthog=cur_orthog)

        self.operator: 'MPS' = operator
        if bra is not None:
            self.bra.site_ind_id = self.bra_site_ind()
        self.mps_to_mpo_operator = mps_to_mpo_operator


    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:  #, select_inds=None) -> qtn.Tensor:
        """ build left environment or right environment
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        ## new env is ket_and_env[select_inds]
        op_tens = self.get_op_tens(i)
        d_ijk = self.get_dijk(i)
        # op_tens = self.operator[self.mps_inds[i]]
        # op_tens = op_tens.reindex({ind: ind + '_o_' for ind in ket_tens.inds})
        # op_tens = ket_tens.reindex({ind: ind + '_o_' for ind in ket_tens.inds})
        upper_ind = self.bra_site_ind().format(i)  # self.mps_inds[i])
        lower_ind = self.ket.site_ind_id.format(i)  # self.mps_inds[i])

        bra_old = self.bra_horizontal_bond(i - direction, direction)
        # ket_old = self.ket_horizontal_bond(i - direction, direction)
        # op_old = self.operator.bond(i - direction, i) if bra_old is not None else None
        bra_new = self.bra_horizontal_bond(i, direction)
        ket_new = self.ket_horizontal_bond(i, direction)
        op_new = self.operator.bond(i, i + direction)

        envL, envR = self.envs[i - 1], self.envs[i + 1]
        env_tens = envL if direction == SweepDirection.RIGHT else envR
        ket_tens = self.ket[i]   ## tensor with selected "columns" / "rows"
        if env_tens is not None:
            ket_and_env = qtn.tensor_contract(op_tens, d_ijk, env_tens, ket_tens)
        else:
            ket_and_env = qtn.tensor_contract(op_tens, d_ijk, ket_tens)

        if bra_old is not None:
            new_env = ket_and_env.fuse({bra_new: [upper_ind, bra_old],})
            new_env.transpose(bra_new, ket_new, op_new, inplace=True)
        else:
            new_env = ket_and_env.reindex({upper_ind: bra_new,})
            new_env.transpose(bra_new, ket_new, op_new, inplace=True)

        # bra_select_inds = self.select_inds_bra[i]
        bra_select_inds = self.bra.select_inds[i]
        if bra_select_inds is None:
            raise ValueError('need to compute/update bra (select_inds) to extend env')

        new_env.transpose(bra_new, ket_new, op_new, inplace=True)
        new_env.modify(apply=lambda x: x[bra_select_inds, :, :])

        self.envs[i] = new_env
        self.cur_orthog = i + direction
        return new_env


