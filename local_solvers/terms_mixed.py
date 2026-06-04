"""Mixed Term layer of the local-solver stack.

Defines :class:`Term_Mixed`, a :class:`Term` subclass that combines DMRG-style
(Galerkin) and cross-style (interpolatory) projections within a single term,
using the mixed blocks from :mod:`local_solvers.blocks_mixed`. It supports the
dynamical low-rank "X"/"G" variants and feeds the mixed Evaluator and time
integrator.
"""
import numpy as np
from abc import ABC

import helper_quimb
from setup_.defaults import *
from local_solvers.blocks import BlockVector_DMRG, BlockOperator_DMRG
from local_solvers.blocks_mixed import *
import local_solvers.helper_cross_2 as helper_cross
from local_solvers.mps_classes import MPS, MPO
import local_solvers.tensor_callables as tc
from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross, BlockType
import local_solvers.helper_mixed as helper_mixed

MPType = Union['qtn.MatrixProductState', 'qtn.MatrixProductOperator']
SelectIndsType = dict[int, Sequence[int]]



class Term_Mixed(Term):

    def __init__(self,
                 ket: 'MPS',
                 bra: Optional['MPS'] = None,
                 operators: Optional[Sequence[qtn.MatrixProductOperator]] = None,
                 operator_k: Optional[qtn.MatrixProductOperator] = None,
                 # eval_func: Optional[Callable] = None,
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 # ket_select_inds: dict[int, Sequence[int]] = None,
                 # bra_select_inds: dict[int, Sequence[int]] = None,
                 mpo_poly: Union[Sequence[int], int] = 1,
                 mps_power: int = 1, mps_coeff: Numeric = 1.0,
                 mps_func: Callable = None,
                 max_bond: int = None, cutoff=CUTOFF,
                 num_tiers: int = 1,
                 ):

        self.version = flags.get('version', 'X')  #  XG, G, or X  (XG only compatible with X in other files); override via flags['version']

        if num_tiers is None:
            max_mpo_poly = len(mpo_poly) - 1 if isinstance(mpo_poly, (list, tuple)) else abs(mpo_poly)
            num_tiers = max_mpo_poly + 1

        super().__init__(ket, bra, operators,
                         operator_k=operator_k,
                         cur_orthog=cur_orthog, direction=direction,
                         mpo_poly=mpo_poly, mps_power=mps_power, mps_coeff=mps_coeff, mps_func=mps_func,
                         max_bond=max_bond, cutoff=cutoff, num_tiers=num_tiers)

        self._proj_vec_targets_x = None
        self._intermediate_sites_x = {}

    @property
    def type(self) -> LocalSolverType:
        return LocalSolverType.MIXED

    # @classmethod
    # def canonize_func(cls, mps, orthog):
    #     helper_quimb.canonize(mps, i=orthog, scale=False, bra=None)
    #     mps.cur_orthog = orthog
    #     return mps

    @classmethod
    def canonize_func(cls, mps: 'MPS', orthog, cur_orthog:int =None, **kwargs):
        # helper_quimb.canonize(mps, scale=False, i=orthog, cur_orthog=cur_orthog)
        if mps is None:
            return None

        mps.canonize(orthog, cur_orthog=cur_orthog)
        if cur_orthog is None:
            mps.get_select_inds(0, orthog)
            mps.get_select_inds(mps.L - 1, orthog)
        else:
            mps.get_select_inds(cur_orthog, orthog)
        return mps

    @classmethod
    def check_func(cls, mps: 'MPS'):
        indL1, indR1 = mps.check_select_inds()
        indL2, indR2 = helper_quimb.check_orthog(mps)
        indL = min(indL1, indL2)
        indR = max(indR1, indR2)
        return indL, indR

    @classmethod
    def canonize_tens_list(cls, tens_list, inplace=True, full_matrices=False):
        raise NotImplementedError
        tens_list = helper_quimb.canonize_tens_list(*tens_list, inplace=inplace, full_matrices=full_matrices)
        return tens_list

    @classmethod
    def compress_tens_list(cls, tens_list, inplace=True, max_bond=None, **kwargs):
        raise NotImplementedError
        tens_list = helper_quimb.compress_tens_list(*tens_list, inplace=inplace, compress_opts={'max_bond': max_bond})
        return tens_list


    def projected_bra_to_ket(self, left_site_pos: int, nsites: int, version_=None):
        # raise NotImplementedError
        version_ = self.version if version_ is None else version_
        if len(self.op_blocks) == 0 or len(self.op_blocks[0]) == 0:
            if version_ == 'G':
                b_to_k = self.vec_block.projected_bra_to_ket(left_site_pos, nsites)
            else:
                b_to_k = self.vec_block.projected_bra_to_ket_x(left_site_pos, nsites)
        else:
            if version_ == 'G':
                b_to_k = self.op_blocks[0][0].projected_bra_to_ket(left_site_pos, nsites)
            else:
                b_to_k = self.op_blocks[0][0].projected_bra_to_ket_x(left_site_pos, nsites)
        return b_to_k

    def initialize_blocks(self):

        if self.init_intermediate_ket is None:
            intermediate_kets = {i: self.bra.copy() for i in range(self.num_tiers - 1)}
        else:
            self.canonize_func(self.init_intermediate_ket, 0 if self.direction > 0 else self.ket.L - 1)
            intermediate_kets = {i: self.init_intermediate_ket.copy() for i in range(self.num_tiers - 1)}

        num_ops = 0 if self.operators is None else len(self.operators)

        ## projection of self.ket
        bra = intermediate_kets.get(0, self.bra if num_ops == 0 else self.ket)
        vec_block = BlockVector_Mixed(self.ket, bra, cur_orthog=self.cur_orthog)
        # if self.mps_power == 1:
        #         vec_block = BlockVector_DMRG(self.ket, bra, cur_orthog=self.cur_orthog)
        #     else:
        #         vec_block = BlockPowerOpKet_DMRG(self.ket, bra, operator=self.operator_k,
        #                                          cur_orthog=self.cur_orthog, power=self.mps_power)
        #         # vec_block = BlockOperator_DMRG(self.ket, bra, operator=self.operator_k,
        #         #                                cur_orthog=self.cur_orthog)
        # else:
        #     if self.operator_k is None:
        #         vec_block = BlockPowerKet_DMRG(self.ket, bra, cur_orthog=self.cur_orthog, power=self.mps_power)
        #     else:
        #         vec_block = BlockPowerOpKet_DMRG(self.ket, bra, operator=self.operator_k,
        #                                          cur_orthog=self.cur_orthog, power=self.mps_power)

        ## projection of operators
        op_blocks_all = {}
        for it in range(self.num_tiers):
            op_blocks = []
            ket = intermediate_kets.get(it, self.bra if it > 0 else self.ket)
            bra = intermediate_kets.get(it + 1, self.bra)

            for op in self.operators:
                if isinstance(op, qtn.MatrixProductOperator):
                    op_blocks += [BlockOperator_Mixed(ket, bra, operator=op.copy(),
                                                     anc_env_left=None, anc_env_right=None,
                                                     cur_orthog=self.cur_orthog)]
                elif isinstance(op, qtn.MatrixProductState):
                    op_blocks += [BlockOperator_Mixed(ket, bra, operator=helper_quimb.mps_to_diag_mpo(op),
                                                         anc_env_left=None, anc_env_right=None,
                                                         cur_orthog=self.cur_orthog)]
                else:
                    raise TypeError('not a valid type for operator')

            op_blocks_all[it] = op_blocks

        self._intermediate_kets = intermediate_kets
        self.op_blocks: Sequence[BlockOperator_Mixed] = op_blocks_all
        self.vec_block: BlockVector_Mixed = vec_block


    def get_evaluated_site(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor'=None,
                           verbose_plot=False, version_=None):
        ### when site_tens is provided, it is usually in the bra coordinates
        ### which is different from the intermediate ket coordinates
        ### so, we need to have bra select inds and intermediate ket select inds be the same
        ### site_tens, if provided is elementwise

        if verbose_plot:
            plt.figure()

            plt.plot(self.ket.to_dense(), label='ket')
            plt.plot(self.bra.to_dense(), label='bra/out')
            if self.num_tiers > 1:
                if getattr(self, 'verbose', 0):
                    print(len(self._intermediate_kets))
                plt.plot(self.get_intermediate_ket(0).to_dense(), label='inter')

            plt.title('init get evaluated site')
            plt.legend()
            plt.show()

        # print('TERM CHECK ORTHOG', left_site_pos, nsites)
        # helper = helper_quimb if self.type == LocalSolverType.DMRG else helper_cross
        # print('check self.ket')
        # helper.check_orthog(self.ket)
        # for it, op_blocks in self.op_blocks.items():
        #     print('it', it)
        #     for op in op_blocks:
        #         print('check op ket')
        #         helper.check_orthog(op.ket)
        #         print('check op bra')
        #         helper.check_orthog(op.bra)
        #         print('op bra', op.bra is self.bra, op.ket is self.ket)
        #     print('vec block', (self.vec_block.bra is self.bra), (self.vec_block.ket is self.ket))
        #     print('check vec bra')
        #     helper.check_orthog(self.vec_block.bra)
        #     print('check op ket')
        #     helper.check_orthog(self.vec_block.ket)

        if not self.initialized:
            raise RuntimeError('term has not yet been initialized')

        # print('self.vec block', self.vec_block)
        # print('site tens', site_tens)
        # exit()

        version_ = self.version if version_ is None else version_

        if version_ == 'G':
            b2k_dict_x = self.vec_block.projected_bra_to_ket(left_site_pos, nsites)
        else:
            b2k_dict_x = self.vec_block.projected_bra_to_ket_x(left_site_pos, nsites)

        ### get vec block
        ## elementwise: bra_ind + '_x' reindexed to ket_ind + '_x'
        if version_ == 'G':
            site_tens, vec_targets = self.vec_block.get_projected(left_site_pos, nsites, return_combined=True,
                                                                    return_intermediates=True,
                                                                    site_tens=site_tens)
        elif version_ == 'XG' and len(self.operators) > 0:
            site_tens, vec_targets = self.vec_block.get_projected(left_site_pos, nsites, return_combined=True,
                                                                  return_intermediates=True,
                                                                  site_tens=site_tens)
            site_tens = site_tens.reindex({ind: ind[:-1] for ind in site_tens.inds if ind[-1] == '_'})
            ## bra to ket reindex, since this isn't covered by b2kdict
        else:
            # print('get vec block X', self)
            # print('site tens', site_tens)
            # print('self.num tiers', self.num_tiers)
            # print('out and ket', self.bra is self.vec_block.ket)
            # if site_tens is not None:
            #     pdb.set_trace()
            site_tens, vec_targets = self.vec_block.get_projected_X(left_site_pos, nsites, return_combined=True,
                                                                    return_intermediates=True,
                                                                    site_tens=site_tens if self.bra is self.vec_block.ket else None,
                                                                    version_=version_)
            if getattr(self, 'verbose', 0):
                print('vec block get proj X')

        # if len(self.operators) > 0:
        #     print('got basis proj')     ## ket_ind
        #     print('self.vec block', self.vec_block.bra is self.vec_block.ket)
        #     print(self.vec_block.bra is self.bra)
        #     site_tens, vec_targets = self.vec_block.get_projected(left_site_pos, nsites,
        #                                                           return_combined=True,
        #                                                           return_intermediates=True,
        #                                                           site_tens=site_tens)
        #     print('vec targets', vec_targets)
        # else:
        #     print('got elementwise proj')  ## ket_ind + '_x'
        #     site_tens, vec_targets = self.vec_block.get_projected_X(left_site_pos, nsites, return_combined=True,
        #                                                             return_intermediates=True,
        #                                                             site_tens=site_tens)

        site_tens.reindex(b2k_dict_x, inplace=True)
        for t in vec_targets:
            t.reindex(b2k_dict_x, inplace=True)

        for t in vec_targets:
            t.modify(apply=lambda x: x * 10 ** (-self.bra.exponent))
        if version_ == 'G':
            self._proj_vec_targets = vec_targets
        else:
            self._proj_vec_targets_x = vec_targets

        if version_ == 'G':
            if 0 in self._intermediate_kets:
                self._intermediate_sites[0] = vec_targets
        else:
            if 0 in self._intermediate_kets:
                self._intermediate_sites_x[0] = vec_targets


        # # ########### plot ####
        # if verbose_plot and self.__class__ is Term_Cross:
        #     coords = helper_cross.get_selectors(self.bra, left_site_pos, nsites)
        #     selectors = []
        #     for c in coords:
        #         selectors += [int("".join(str(x) for x in c), 2)]
        #     # print('coords', coords)
        #     # print('selectors', selectors)
        #
        #     inds = []
        #     if left_site_pos > 0:
        #         inds += [self.ket.bond(left_site_pos, left_site_pos - 1)]
        #     inds += [self.ket.site_ind(left_site_pos + i) for i in range(nsites)]
        #     if left_site_pos + nsites < self.ket.L:
        #         inds += [self.ket.bond(left_site_pos + nsites - 1, left_site_pos + nsites)]
        #
        #     plt.figure()
        #     site_tens.transpose(*inds, inplace=True)
        #     plt.plot(selectors, site_tens.data.reshape(-1), 'x', label='site tens')
        #
        #     plt.plot(self.ket.to_dense(), label='ket')
        #     plt.plot(self.bra.to_dense(), label='bra/out')
        #
        #     plt.title('get evaluated site')
        #     plt.legend()
        #     plt.show()
        #     #########


        bonds_o = [k for k in b2k_dict_x.keys() if k is not None]
        bonds_i = [b2k_dict_x[bo] for bo in bonds_o]

        eff_ops = {}
        max_it = self.num_tiers
        for it, blocks in self.op_blocks.items():
            eff_ops_list = []
            for env in blocks:
                # print('env ket is vec block bra', env.ket is self.vec_block.bra)
                # print(env.ket.select_tens_inv.items())
                if version_ == 'G':
                    eff_op = env.get_projected(left_site_pos, nsites, return_combined=False)
                elif version_ == 'XG':
                    eff_op = env.get_projected_XG(left_site_pos, nsites, return_combined=False)
                else:
                    eff_op = env.get_projected_XX(left_site_pos, nsites, return_combined=False)
                ## bra_ind + '_x' // ket_ind + '_x'

                # if it == max_it - 1:  ## is the last tier; get element-wise info
                #     print('op got XG')
                #     eff_op = env.get_projected_XG(left_site_pos, nsites, return_combined=False)
                # else:
                #     print('op got projected')
                #     eff_op = env.get_projected(left_site_pos, nsites, return_combined=False)

                eff_ops_list += [eff_op]
                # b2k_dict = env.projected_bra_to_ket(left_site_pos, nsites)
            eff_ops[it] = eff_ops_list
            # print('eff ops', [t.norm() for t in eff_ops_list])

        # if self.num_tiers > 1:
        #     helper_cross.check_orthog(self.get_intermediate_ket(0))

        # print('self.eval func', self.eval_func)
        eff_ops = None if len(eff_ops) == 0 else eff_ops
        out_site, intermediates = self.eval_func(site_tens, eff_ops, b2k_dict_x, return_intermediates=True)
        # out_site, intermediates = self.eval_func(site_tens_g, eff_ops, b2k_dict_x, return_intermediates=True)

        # self._intermediate_sites_x = intermediates

        for k, vals in intermediates.items():
            if version_ == 'G':
                if k in self._intermediate_sites:
                    self._intermediate_sites[k] += vals
                else:
                    self._intermediate_sites[k] = vals
            else:
                if k in self._intermediate_sites_x:
                    self._intermediate_sites_x[k] += vals
                else:
                    self._intermediate_sites_x[k] = vals

        # self._intermediate_sites = intermediates   ## dict[int, Seq[Tensor]]
        self._out_site = out_site
        for it, inter in intermediates.items():
            # inter.modify(apply=lambda x: x * 10 ** (-self.bra.exponent))
            for t in inter:
                t.modify(apply = lambda x: x * 10 ** (-self.bra.exponent))

        ## need to reindex to ket inds
        out_site.reindex(b2k_dict_x, inplace=True)

        # print('remove bra exponent')
        site_tens.modify(apply = lambda x: x * 10 ** (-self.bra.exponent))
        out_site.modify(apply = lambda x: x * 10 ** (-self.bra.exponent))

        return out_site


    def get_eff_operator(self, left_site_pos: int, nsites: int, return_combined = False, transpose_bonds = None,
                         ) -> Union[dict[int,list], 'qtn.Tensor']:
        ### maybe this should return eff op for all levels combined?
        # raise NotImplementedError

        if self.version == 'X':
            eff_ops = {it: [op.get_projected_XX(left_site_pos, nsites, return_combined=False) for op in ops]
                       for it, ops in self.op_blocks.items()}
        elif self.version == 'XG':
            eff_ops = {it: [op.get_projected_XG(left_site_pos, nsites, return_combined=False) for op in ops]
                       for it, ops in self.op_blocks.items()}
        elif self.version == 'G':
            eff_ops = {it: [op.get_projected(left_site_pos, nsites, return_combined=False) for op in ops]
                       for it, ops in self.op_blocks.items()}
        else:
            raise ValueError

        if return_combined:
            for k, ops in eff_ops.items():
                eff_op = helper_tn.sum_eff_TNs(ops, transpose_bonds=transpose_bonds)
                eff_ops[k] = eff_op

        return eff_ops

    def update_intermediate_kets(self, i: int, nsites: int, direction: SweepDirection,):

        if getattr(self, 'verbose', 0):
            print('update intermediate kets')

        if nsites == 1 or direction == SweepDirection.RIGHT:
            left_site_pos = i
        else:
            left_site_pos = i - 1

        # print('update intermediate kets?', self.num_tiers)
        # print('intermediate sites', self.intermediate_sites)
        # exit()

        at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

        for it in range(self.num_tiers - 1):

            # print('tier', it)

            # if isinstance(self, Term_DMRG) and isinstance(self.vec_block, BlockPowerKet_DMRG):
            #     new_ket_site = [self.proj_vec_eval] + [*self.proj_vec[it]]
            # if isinstance(self, Term_DMRG):
            #     # new_ket_site = [*self.proj_vec_eval, self.proj_vec[it]]
            #     new_ket_site = self.proj_vec[it]
            # else:   ## is Term_Cross
            #     new_ket_site = self.proj_vec_eval

            new_ket_site = self._intermediate_sites_x[it]    ## list of tensors

            # print('new ket site', new_ket_site, len(new_ket_site))
            # print('self.proj vec targets', self.proj_vec_targets, len(self.proj_vec_targets))
            # pdb.set_trace()

            ### altered PowerKet to not need to do this. but doesn't really change anything
            # if it == 0 and not at_end and isinstance(self, Term_DMRG):  ## works for Cross but not necessary?
            #     new_ket_site = [*new_ket_site, *self.proj_vec_targets]    # self.proj_vec,
            #     ## i think proj_vec is included in proj_vec_targets

            if nsites == 1:

                # for tens in new_ket_site:
                #
                #     tmp = self._intermediate_kets[it].copy()
                #
                #
                #     ket = self.vec_block.ket
                #     vbra = self.vec_block.bra
                #     ket2 = helper_quimb.to_dense(ket) ** 2
                #
                #     tens_ = tens.reindex({ind: ind[:-2] for ind in tens.inds if ind[-1] == 'x'})
                #     helper_cross.plot_submat(vbra, left_site_pos, 1, tens_, ref_kets=[ket, ket2])
                #
                #     tens_g = helper_mixed.convert_elementwise_to_basis(tmp, tens, left_site_pos, 1)
                #     print('tens', tens_g)
                #     tmp[left_site_pos].transpose_like(tens_g, inplace=True)
                #     tmp[left_site_pos].modify(data = tens_g.data)
                #     plt.figure()
                #     plt.plot(helper_quimb.to_dense(tmp))
                #     plt.title(f'intermediate ket {it}')
                #     plt.show()

                helper_mixed.update_1site(self._intermediate_kets[it], left_site_pos, new_ket_site, direction,
                                         max_bond=self.max_bond, cutoff=self.cutoff)

                # tmp1, tmp2 = helper_mixed.check_orthog(self._intermediate_kets[it])
                # print('intermediate orthog', tmp1, tmp2)
                # if tmp1 != tmp2:
                #     raise RuntimeError

                # tmp = self._intermediate_kets[it].copy()
                # ind2 = left_site_pos + direction
                # new_site_tens = tmp[ind2]
                # new_site_x = helper_mixed.convert_basis_to_elementwise(tmp, new_site_tens, ind2, 1)
                # new_site_x = new_site_x.reindex({ind: ind[:-2] for ind in new_site_x.inds if ind[-1] == 'x'})
                # helper_cross.plot_submat(tmp, ind2, 1, new_site_x)

                # plt.figure()
                # tmp = self._intermediate_kets[it].copy()
                # plt.plot(helper_quimb.to_dense(tmp))
                # plt.title(f'updated intermediate ket {it}')
                # plt.show()
            else:
                helper_mixed.update_2site(self._intermediate_kets[it], left_site_pos, new_ket_site, direction,
                                         max_bond=self.max_bond, cutoff=self.cutoff)

        return



