"""Mixed-projection blocks for the local-solver stack.

Extends the DMRG block hierarchy from :mod:`local_solvers.blocks` with the
:class:`BlockMixed` family used by the mixed (combined Galerkin/cross)
projection scheme. These blocks track both full-contraction and
index-selected environments at each site so a single sweep can combine
DMRG-style and cross-style projections, supporting the dynamical low-rank
"X"/"G" variants used by the mixed Term and Evaluator layers.
"""
import pdb

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
from local_solvers.blocks import Block,BlockDMRG, BlockOperator_DMRG, BlockVector_DMRG, BlockDiagOperator_DMRG, EnvironmentType
import local_solvers.helper_mixed as helper_mixed

ProjType = Union[MPS, qtn.MatrixProductState, dict[int, tuple[int,...]]]

class SweepDirection(IntEnum):
    ## int denotes where the MPS needs to be canonicalized to
    LEFT = -1
    RIGHT = 1

version = 'X'   ## 'G' for Galerkin projection

class BlockMixed(BlockDMRG, ABC):

    def __init__(self, # env_type: EnvironmentType,
                 ket: Union['MPS', 'qtn.MatrixProductState'],
                 bra: Optional[Union['MPS', 'qtn.MatrixProductState']],
                 operator: 'qtn.MatrixProductOperator' = None,
                 anc_env_left: 'qtn.Tensor' = None,
                 anc_env_right: 'qtn.Tensor' = None,
                 # mps_inds: Sequence[int] = None,
                 cur_orthog = None,
                 ):

        self.version = flags.get('version', version)  ## 'G' for Galerkin method; override via flags['version']

        self.select_envs = {-1: None, ket.L: None}
        super(BlockDMRG, self).__init__(ket, bra, operator, cur_orthog=cur_orthog,
                                        anc_env_left=anc_env_left, anc_env_right=anc_env_right)


    @property
    def env_type(self):
        return EnvironmentType.DMRG

    def copy(self, ket_copy=None, bra_copy=None):

        out = super().copy(ket_copy=ket_copy, bra_copy=bra_copy)
        out.select_envs = {k: (env.copy() if env is not None else None) for k, env in self.select_envs.items()}
        return out

    # def projected_bra_to_ket_x(self, left_site_pos: int, nsites: int):
    #     b2k_dict = super().projected_bra_to_ket(left_site_pos, nsites)
    #     b2k_dict_x = {k + '_x': v + '_x' for k,v in b2k_dict.items() if k is not None}
    #     return b2k_dict_x

    def _projected_bra_to_ket_bond_x(self, left_site_pos: int):
        """ projected bra to ket when extracting a "bond"
        """
        bx_bond = self.bra_horizontal_bond(left_site_pos, 1) + '_x'
        kx_bond = self.ket_horizontal_bond(left_site_pos, 1) + '_x'

        inds_dict = {bx_bond + '_L': kx_bond + '_L', bx_bond + '_R': kx_bond + '_R'}
        return inds_dict

    def projected_bra_to_ket_x(self, left_site_pos: int, nsites: int):
        if nsites == 0:
            return self._projected_bra_to_ket_bond_x(left_site_pos)

        inds_dict = {self.bra_horizontal_bond(left_site_pos, -1):
                         self.ket_horizontal_bond(left_site_pos, -1),
                     self.bra_horizontal_bond(left_site_pos + nsites - 1, 1):
                         self.ket_horizontal_bond(left_site_pos + nsites - 1, 1)}

        inds_dict_x = {k + '_x': v + '_x' for k, v in inds_dict.items() if k is not None}
        inds_dict = inds_dict_x

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

    # def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
    #     """ build left environment or right environment to include tensors at site i
    #         assumes bra, ket are properly canonicalized, bra already cc'ed if specified
    #         pos: position of env to extend
    #     """
    #     # bra_tens = [self.bra_site(i)]
    #     # ket_tens = [self.ket[self.mps_inds[i]]]
    #     bra_tens = [self.bra_site(i)]
    #     ket_tens = [self.ket.get_tens(i)]
    #
    #     if isinstance(self.operator, MatrixProductOperatorTN):
    #         op_tens = self.operator[i]  # [self.mps_inds[i]]
    #     elif self.operator is None:
    #         op_tens = []
    #     else:
    #         op_tens = [self.operator[i]]  # [self.mps_inds[i]]]
    #         # op_tens = [mpo[pos] for mpo in self.operators]
    #
    #     new_env = qtn.TensorNetwork(bra_tens + ket_tens + op_tens)
    #     # print('bra tens', bra_tens[0])
    #     # print('ket tens', ket_tens[0])
    #     # print('op tens', op_tens[0])
    #
    #     env_L, env_R = self.envs[i - 1], self.envs[i + 1]
    #     env = env_L if direction == SweepDirection.RIGHT else env_R
    #     if env is not None:
    #         new_env.add(env)
    #
    #
    #     # new_env_exp = new_env.exponent
    #     new_env = new_env.contract_tags(all)
    #
    #     if self.operator is None and direction < 0 and i > 0:
    #         bra = self.bra.conj()
    #         bra.mangle_inner_(append='_')
    #         bra_tensors = [bra[ix] for ix in range(i, bra.L)]
    #         ket_tensors = [self.ket[ix] for ix in range(i, self.ket.L)]
    #         tmp_env = qtn.tensor_contract(*bra_tensors, *ket_tensors)
    #         # print('self.ket', self.ket.exponent, helper_quimb.check_orthog(self.ket))
    #         tmp_env.transpose_like(new_env, inplace=True)
    #         # print('extend env diff', np.linalg.norm(tmp_env.data - new_env.data))
    #         # if i == 1:
    #         #     print('tmp env', tmp_env.data, tmp_env.inds)
    #
    #
    #     self.envs[i] = new_env
    #     self.cur_orthog = i + direction
    #
    #     return new_env



#####

class BlockVector_Mixed(BlockMixed, BlockVector_DMRG):

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
        self.version = flags.get('version', version)  ## 'G' for Galerkin method; override via flags['version']
        self.select_envs = {-1: None, ket.L: None}
        super(BlockMixed, self).__init__(ket, bra, anc_env_left=anc_env_left, anc_env_right=anc_env_right, # mps_inds=mps_inds,
                                         cur_orthog=cur_orthog)
        self._projected_site_x = None

    @property
    def projected_site_x(self):
        return self._projected_site_x

    @property
    def select_inds_bra(self):
        if self.bra is not None:
            return self.bra.select_inds
        else:
            return self.ket.select_inds

    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        """ build left environment or right environment to include tensors at site i
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        # self.decimate(i, direction)
        new_env_orthog = super().extend_env(i, direction)

        x_version = flags.get('x_version', 'select')  #  'select' or 'proj'; override via flags['x_version']
        print('x version', x_version)

        if x_version == 'proj':
            g2x = self.bra.select_tens.get(i, None)
            ## bra (g) --> bra(x)
            b2k = {**{ind: ind[:-2] + '__x' for ind in g2x.inds if ind[-1] == 'x'},   ## bra x
                   **{ind: ind + '_' for ind in g2x.inds if ind[-1] != 'x'}}
            g2x = g2x.reindex(b2k)
            sel_env = new_env_orthog @ g2x
            self.select_envs[i] = sel_env

        elif x_version == 'select':
            #### update select_envs ####
            ## new env is ket_and_env[select_inds],  bra_ind + '_x' // ket_ind
            ket_tens = self.ket.get_tens(i)  # self.ket[self.mps_inds[i]]
            site_ind = self.ket.site_ind(i)  # self.ket.site_ind_id.format(self.mps_inds[i])

            bond_old = self.bra_horizontal_bond(i - direction, direction)
            if bond_old is not None:     bond_old = bond_old + '_x'
            bond_new = self.bra_horizontal_bond(i, direction) + '_x'
            bond_ket = self.ket_horizontal_bond(i, direction)

            # env_L, env_R = self.select_envs[i - 1], self.select_envs[i + 1]
            if direction == SweepDirection.RIGHT:
                env_L = self.select_envs[i - 1]
                if env_L is not None:
                    ket_and_env = qtn.tensor_contract(ket_tens, env_L)
                else:
                    ket_and_env = ket_tens.copy()
            else:
                env_R = self.select_envs[i + 1]
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

            self.select_envs[i] = new_env

        return new_env_orthog


    def get_projected(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
                      return_intermediates=False, site_tens: 'qtn.Tensor'=None):
        """ assumes that the provided site_tens is element-wise
        """
        if self.version != 'G':
            if site_tens is not None:
                site_tens = helper_mixed.convert_elementwise_to_basis(self.ket, site_tens, left_site_pos, nsites)
                # basis_sites = self.elementwise_to_basis([site_tens], left_site_pos, nsites)
                # site_tens = basis_sites[0]
        return super().get_projected(left_site_pos, nsites, return_combined=return_combined,
                                     transpose_bonds=transpose_bonds, return_intermediates=return_intermediates,
                                     site_tens=site_tens)


    def get_projected_bond_X(self, left_site_pos:int, return_combined=False, transpose_bonds=None,
                                    site_tens:'qtn.Tensor'=None, return_intermediates=False, version_=None):
        """ get environment for bond between left_site_pos, left_site_pos + 1
            assumes ket, bra at left_site_pos are canonical and left_site_pos+1 is not updated
            or vice versa (left_site_pos + 1 is now canonical, and left_site_pos is not updated)
            output: lbond + '_x', rbond + '_x'
        """
        if site_tens is None:
            raise NotImplementedError

        if version_ is None:
            version_ = self.version

        if version_ != 'G':
            site_tens = helper_mixed.convert_elementwise_to_basis(self.bra, site_tens, left_site_pos, 0)

        b_left = self.select_envs[left_site_pos].copy()             ## ket_ind + '_x', ket_ind
        b_right = self.select_envs[left_site_pos + 1].copy()        ## ket_ind + '_x', ket_ind

        if version_ == 'G':
            x_ind = self.bra_horizontal_bond(left_site_pos, 1)
            b_left.reindex({x_ind: x_ind + '_L'}, inplace=True)
            b_right.reindex({x_ind: x_ind + '_R'}, inplace=True)
        else:
            x_ind = self.bra.bond(left_site_pos, left_site_pos + 1)  ## X out / G in?
            b_left.reindex({x_ind: x_ind + '_L', x_ind + '__x': x_ind + '__x_L'}, inplace=True)
            b_right.reindex({x_ind: x_ind + '_R', x_ind + '__x': x_ind + '__x_R'}, inplace=True)

        b_eff = qtn.TensorNetwork([site_tens])
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
            self._projected_site_x = b_eff_tens
            b_eff = b_eff_tens

        self._projected_site_x = b_eff

        if return_intermediates:
            return b_eff, []

        return b_eff


    def get_projected_X(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
                               return_intermediates=False, site_tens:'qtn.Tensor'=None, version_=None):
        """ element-wise selection
            incorporate ket exponent, remove bra exponent
            output: lbond + '_x', rbond + '_x'
        """
        if nsites == 0:
            return self.get_projected_bond_X(left_site_pos, return_combined=return_combined, site_tens=site_tens,
                                                 transpose_bonds=transpose_bonds,
                                                 return_intermediates=return_intermediates, version_=version_)

        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        # print('left site pos', left_site_pos, 'cur orthog', self.cur_orthog)
        assert (left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'

        version_ = self.version if version_ is None else version_

        # b_left = self.envs[left_site_pos - 1]
        # b_right = self.envs.get(left_site_pos + nsites, None)
        b_left = self.select_envs.get(left_site_pos - 1, None)          ## ket_ind + '_x', ket_ind
        b_right = self.select_envs.get(left_site_pos + nsites, None)    ## ket_ind + '_x', ket_ind

        b_eff = qtn.TensorNetwork([])
        if site_tens is None:
            if isinstance(site_inds, (list, tuple)):
                for si in site_inds:
                    b_eff.add(self.ket[si])
            else:
                b_eff.add(self.ket[site_inds])

            if b_left is not None:
                b_eff.add(b_left)
            if b_right is not None:
                b_eff.add(b_right)

            b_eff.exponent += self.ket.exponent

        else:

            # basis_sites = self.elementwise_to_basis([site_tens], left_site_pos, nsites)
            # site_tens = basis_sites[0]

            # if isinstance(site_tens,(tuple,list)):
            #     b_eff.add(*site_tens)
            # else:
            #     b_eff.add(site_tens)

            if version_ != 'G':
                site_tens = helper_mixed.convert_elementwise_to_basis(self.bra, site_tens, left_site_pos, nsites)

            b_eff.add(site_tens)

            if b_left is not None:
                b_eff.add(b_left)
            if b_right is not None:
                b_eff.add(b_right)

            b_eff.exponent += self.ket.exponent

        # ## remove bra exponent
        # b_eff.exponent -= self.bra.exponent

        if return_combined:
            b_eff_tens = qtn.tensor_contract(*b_eff.tensors, preserve_tensor=True)
            b_eff_tens.modify(apply=lambda data: data * 10 ** b_eff.exponent)  # self.ket.exponent)
            if transpose_bonds is not None:
                b_eff_tens.transpose(*transpose_bonds, inplace=True)
            self._projected_site_x = b_eff_tens
            b_eff = b_eff_tens

        self._projected_site_x = b_eff

        # print('remove bra exponent')
        # b_eff.modify(apply=lambda x: x * 10 ** (-self.bra.exponent))

        if return_intermediates:
            b2k_dict = self.projected_bra_to_ket(left_site_pos, nsites)
            return b_eff, [b_eff.copy()]  # [b_eff.reindex(b2k_dict, inplace=False)]

        return b_eff

    def get_projected_X_new(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
                            return_intermediates=False, site_tens: 'qtn.Tensor' = None):
        """ element-wise selection
            incorporate ket exponent, remove bra exponent
            output: lbond + '_x', rbond + '_x'
        """
        # if nsites == 0:
        #     return self.get_projected_bond_X(left_site_pos, return_combined=return_combined, site_tens=site_tens,
        #                                      transpose_bonds=transpose_bonds,
        #                                      return_intermediates=return_intermediates)
        #

        out = super().get_projected(left_site_pos, nsites, return_combined=True, site_tens=site_tens,
                                  transpose_bonds=transpose_bonds,
                                  return_intermediates=return_intermediates)
        if return_intermediates:
            proj_site, targets = out
        else:
            proj_site = out
            targets = []

        b2k_dict = self.projected_bra_to_ket(left_site_pos, nsites)

        proj_site = proj_site.reindex(b2k_dict)
        proj_site_x = helper_mixed.convert_basis_to_elementwise(self.bra, proj_site, left_site_pos, nsites)
        targets_x = []
        for t in targets:
            t = t.reindex(b2k_dict)
            tx = helper_mixed.convert_basis_to_elementwise(self.bra, t, left_site_pos, nsites)
            targets_x += [tx]

        self._projected_site_x = proj_site_x

        if return_intermediates:
            return proj_site_x, targets_x
        else:
            return proj_site_x


    def elementwise_to_basis(self, site_tens_list, left_site_pos: int, nsites: int):
        """ project element-wise tensors onto bra basis
        """
        raise NotImplementedError
        b_left_inv = self.bra.select_tens_inv.get(left_site_pos - 1, None)
        b_right_inv = self.bra.select_tens_inv.get(left_site_pos + nsites, None)

        l_bond = self.bra.bond(left_site_pos, left_site_pos - 1) if left_site_pos > 0 else None
        r_bond = self.bra.bond(left_site_pos + nsites, left_site_pos + nsites - 1) \
                        if left_site_pos + nsites < self.L else None

        if b_left_inv is not None:
            b_left_inv = b_left_inv.transpose(l_bond + '_', l_bond + '_tmp', inplace=False)
            b_left_inv.modify(inds=(l_bond + '_', l_bond))

        if b_right_inv is not None:
            b_right_inv = b_right_inv.transpose(r_bond + '_', r_bond + '_tmp', inplace=False)
            b_right_inv.modify(inds=(r_bond + '_', r_bond))

        proj_sites = []
        for site_tens in site_tens_list:
            tmp = [site_tens]
            if b_left_inv is not None:  tmp += [b_left_inv]
            if b_right_inv is not None:  tmp += [b_right_inv]
            out = qtn.tensor_contract(b_left_inv, b_right_inv, site_tens)
            if l_bond is not None:
                out.reindex({l_bond + '_': l_bond}, inplace=True)
            if r_bond is not None:
                out.reindex({r_bond + '_': r_bond}, inplace=True)
            proj_sites += [out]

        return proj_sites


    # def decimate(self, pos: int, direction: SweepDirection, new_ket_site: Sequence['qtn.Tensor'] = None,
    #              max_bond: int = None):
    #     ### put into extend env, so that envs are always identity?
    #     ### canonicalization / decimation probably more important when bra is None
    #
    #     ind1, ind2 = pos, pos + direction
    #     at_end = (ind2 == self.L or ind2 == -1)
    #
    #     if self.bra is None:
    #         if new_ket_site is None:
    #             helper.canonize_tens_list(self.ket[ind1], self.ket[ind2], inplace=True,
    #                                       full_matrices=False)
    #         else:
    #             helper_dmrg.update_1site(self.ket, pos, new_ket_site, direction, max_bond=max_bond)
    #     else:
    #         ## assumes bra is in the desired canonical form
    #         pass
    #         # if new_ket_site is None:
    #         #     new_ket_site = self.bra[pos].conj()
    #         #     new_ket_site.reindex(self.projected_bra_to_ket(pos, 1), inplace=True)
    #         #
    #         # helper_dmrg.decimate(self.ket, pos, new_ket_site, direction)
    #
    #     self.cur_orthog = ind2
    #
    #     return


class BlockOperator_Mixed(BlockMixed, BlockOperator_DMRG):

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
        self.version = flags.get('version', version)  ## 'G' for Galerkin method; override via flags['version']
        self.select_envs = {-1: None, ket.L: None}
        super(BlockMixed, self).__init__(ket, bra, operator=operator,
                                         anc_env_left=anc_env_left, anc_env_right=anc_env_right, # mps_inds=mps_inds,
                                         # select_inds_bra=select_inds_bra, select_inds_ket=select_inds_ket,
                                         cur_orthog=cur_orthog)

    def extend_env(self, i: int, direction: SweepDirection) -> qtn.Tensor:
        """ build left environment or right environment to include tensors at site i
            assumes bra, ket are properly canonicalized, bra already cc'ed if specified
            pos: position of env to extend
        """
        #### orthogonal envs ####
        new_env_orthog = super().extend_env(i, direction)

        x_version = flags.get('x_version', 'select')  #  'select' or 'proj'; override via flags['x_version']

        if x_version == 'proj':
            g2x = self.bra.select_tens.get(i, None)
            ## bra (g) --> bra(x)
            b2k = {**{ind: ind[:-2] + '__x' for ind in g2x.inds if ind[-2:] == '_x'},  ## bra x
                   **{ind: ind + '_' for ind in g2x.inds if ind[-2:] != '_x'}}
            g2x = g2x.reindex(b2k)
            sel_env = new_env_orthog @ g2x
            self.select_envs[i] = sel_env

        elif x_version == 'select':

            # print('orig SAE extend env', 'op is None', self.operator is None)
            #### new env is ket_and_env[select_inds] (bra_ind + '_x', ket_ind)  ####
            op_tens = self.operator[i]
            upper_ind = self.operator.upper_ind_id.format(i)  # (self.mps_inds[i])
            # lower_ind = self.operator.lower_ind_id.format(i)  # (self.mps_inds[i])

            bra_old = self.bra_horizontal_bond(i - direction, direction)
            if bra_old is not None:     bra_old = bra_old + '_x'
            # ket_old = self.ket_horizontal_bond(i - direction, direction)
            # op_old = self.operator.bond(i - direction, i) if bra_old is not None else None
            bra_new = self.bra_horizontal_bond(i, direction) + '_x'
            ket_new = self.ket_horizontal_bond(i, direction)
            op_new = self.operator.bond(i, i + direction)

            # envL, envR = self.select_envs[i - 1], self.select_envs[i + 1]
            # env_tens = envL if direction == SweepDirection.RIGHT else envR

            if direction == SweepDirection.RIGHT:
                env_tens = self.select_envs[i - 1]
            else:
                env_tens = self.select_envs[i + 1]

            ket_tens = self.ket[i]  ## tensor with selected "columns" / "rows"
            if env_tens is not None:
                ket_and_env = qtn.tensor_contract(op_tens, env_tens, ket_tens)
            else:
                ket_and_env = qtn.tensor_contract(op_tens, ket_tens)

            if bra_old is not None:     ## not at end
                new_env = ket_and_env.fuse({bra_new: [upper_ind, bra_old]})
                new_env.transpose(bra_new, ket_new, op_new, inplace=True)
            else:   ## at end
                new_env = ket_and_env.reindex({upper_ind: bra_new,})
                new_env.transpose(bra_new, ket_new, op_new, inplace=True)

            bra_select_inds = self.bra.select_inds[i]
            if bra_select_inds is None:
                raise ValueError('need to compute/update bra (select_inds) to extend env')

            new_env.transpose(bra_new, ket_new, op_new, inplace=True)
            new_env.modify(apply=lambda x: x[bra_select_inds, :, :])

            self.select_envs[i] = new_env

        return new_env_orthog

    def get_projected_bond_XG(self, left_site_pos:int, return_combined=False, transpose_bonds=None):
        """ get environment for bond between left_site_pos, left_site_pos + 1
            assumes ket, bra at left_site_pos are canonical and left_site_pos+1 is not updated
            or vice versa (left_site_pos + 1 is now canonical, and left_site_pos is not updated)
            G (orthog) input, X (elementwise) output
        """
        A_left = self.select_envs[left_site_pos].copy()
        A_right = self.select_envs[left_site_pos + 1].copy()

        xb_ind = self.bra_horizontal_bond(left_site_pos, 1)
        xk_ind = self.ket_horizontal_bond(left_site_pos, 1)
        A_left.reindex({xb_ind + "_x": xb_ind + '_x_L', xk_ind: xk_ind + '_L'}, inplace=True)
        A_right.reindex({xb_ind + "_x": xb_ind + '_x_R', xk_ind: xk_ind + '_R'}, inplace=True)

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


    def get_projected_XG(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
                         return_intermediates=False, site_tens: 'qtn.Tensor'=None):

        # raise RuntimeError

        if site_tens is not None:
            raise NotImplementedError

        if nsites == 0:
            return self.get_projected_bond_XG(left_site_pos, return_combined=return_combined,
                                                    transpose_bonds=transpose_bonds)

        # site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        assert(left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'

        A_left = self.select_envs[left_site_pos - 1]
        A_right = self.select_envs[left_site_pos + nsites]

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

    def get_projected_bond_XX(self, left_site_pos:int, return_combined=False, transpose_bonds=None):
        """ get environment for bond between left_site_pos, left_site_pos + 1
            assumes ket, bra at left_site_pos are canonical and left_site_pos+1 is not updated
            or vice versa (left_site_pos + 1 is now canonical, and left_site_pos is not updated)
            X (elementwise) input, X (elementwise) output
        """
        A_left = self.select_envs[left_site_pos].copy()         ## bra ind + "_x", ket ind
        A_right = self.select_envs[left_site_pos + 1].copy()    ## bra ind + "_x". ket ind

        ## convert elm input to galerkin basis
        ul_inv = self.ket.select_tens_inv.get(left_site_pos, None)          ## ket_ind, ket_ind + "_x"
        ur_inv = self.ket.select_tens_inv.get(left_site_pos + 1, None)      ## ket_ind, ket_ind + "_x"

        A_left = qtn.tensor_contract(A_left, ul_inv)    ## bra_ind + '_x' // ket_ind
        A_right = qtn.tensor_contract(A_right, ur_inv)  ## bra_ind + '_x' // ket_ind

        xb_ind = self.bra_horizontal_bond(left_site_pos, 1)
        xk_ind = self.ket_horizontal_bond(left_site_pos, 1)
        A_left.reindex({xb_ind + "_x": xb_ind + '_x_L', xk_ind + '_x': xk_ind + '_x_L'}, inplace=True)
        A_right.reindex({xb_ind + "_x": xb_ind + '_x_R', xk_ind + '_x': xk_ind + '_x_R'}, inplace=True)

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


    def get_projected_XX(self, left_site_pos: int, nsites: int, return_combined=False, transpose_bonds=None,
                         return_intermediates=False, site_tens: 'qtn.Tensor'=None):

        # raise RuntimeError

        if site_tens is not None:
            raise NotImplementedError

        if nsites == 0:
            return self.get_projected_bond_XX(left_site_pos, return_combined=return_combined,
                                                    transpose_bonds=transpose_bonds)

        # site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        assert(left_site_pos <= self.cur_orthog < left_site_pos + nsites), \
            'orthogonality center not within unprojected sites'

        A_left = self.select_envs[left_site_pos - 1]            ## bra ind + "_x", ket ind
        A_right = self.select_envs[left_site_pos + nsites]      ## bra ind + "_x", ket ind

        # print('get op XX', A_left)
        # print('get op XX', A_right)
        # print('A right G', self.envs[left_site_pos + nsites])
        # print('sel tens', self.bra.select_tens.get(left_site_pos + nsites, None))
        #
        # orthog_env_r = self.envs[left_site_pos + nsites]
        # sel_mat = self.bra.select_tens.get(left_site_pos + nsites, None)
        # k2b = {**{ind: ind[:-2] + '__x' for ind in sel_mat.inds if ind[-1] == 'x'},
        #        **{ind: ind + '_' for ind in sel_mat.inds if ind[-1] != 'x'}}
        # sel_mat = sel_mat.reindex(k2b)
        # sel_orthog_env_r = qtn.tensor_contract(sel_mat, orthog_env_r)
        # print('env diff R', (sel_orthog_env_r - A_right).norm())
        #
        # inv_sel_mat = self.bra.select_tens_inv.get(left_site_pos + nsites, None)
        # print('inv sel mat', inv_sel_mat)
        # k2b = {ind: ind + '_' for ind in inv_sel_mat.inds if ind[-1] != 'x'}
        # inv_sel_mat = inv_sel_mat.reindex(k2b)
        # print('inv sel mat', inv_sel_mat)
        # print('is inv?', np.linalg.norm( (inv_sel_mat @ sel_mat).data - np.eye(sel_mat.shape[0])) )

        ## convert elm input to galerkin basis
        ul_inv = self.ket.select_tens_inv.get(left_site_pos - 1, None)      ## ket_ind, ket_ind + "_x"
        ur_inv = self.ket.select_tens_inv.get(left_site_pos + nsites, None)  ## ket_ind, ket_ind + "_x"

        if ul_inv is not None:
            A_left = qtn.tensor_contract(A_left, ul_inv)    ## bra_ind + '_x' // ket_ind
        if ur_inv is not None:
            A_right = qtn.tensor_contract(A_right, ur_inv)  ## bra_ind + '_x' // ket_ind

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

        # A_eff.exponent += self.operator.exponent      ## causes issues?
        A_eff.exponent = self.operator.exponent

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

