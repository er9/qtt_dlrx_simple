"""Term layer of the local-solver stack.

Defines the :class:`Term` abstract base class and its concrete subclasses
:class:`Term_DMRG` and :class:`Term_Cross`. A Term packages a ket, optional
bra and operators (with polynomial powers and coefficients) into a single
additive contribution to the local problem, owning the underlying Blocks and
producing the effective per-site tensors that the Evaluator layer combines and
solves.
"""
import pdb

import numpy as np
from abc import ABC

import helper_quimb
from setup_.defaults import *
from local_solvers.blocks import *
# from local_solvers_old.helper_tn import get_mps_matching_inds, sum_tens
# from helper_cross_v3 import canonize, canonize_tens_list, compress_tens_list
import local_solvers.helper_cross_2 as helper_cross
from local_solvers.mps_classes import MPS, MPO
import local_solvers.tensor_callables as tc
import local_solvers.helper_dmrg_loc as helper_dmrg
from local_solvers.blocks import BlockPowerKet_DMRG
import local_solvers.helper_mixed as helper_mixed

MPType = Union['qtn.MatrixProductState', 'qtn.MatrixProductOperator']
SelectIndsType = dict[int, Sequence[int]]

class BlockType(Enum):
    DMRG = 'dmrg'
    CROSS = 'cross'
    MIXED = 'mixed'

## todo: write wrapper to check if self.initialized

class Term(ABC):

    def __init__(self,
                 ket: Union['MPS',qtn.MatrixProductState],
                 bra: Optional[Union['MPS',qtn.MatrixProductState]],
                 operators: Optional[Sequence[qtn.MatrixProductOperator]],
                 operator_k: Optional[qtn.MatrixProductOperator] = None,
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 mpo_poly: Union[Sequence[int], int] = 1,  ## int can also be -1, powers ordered from 0 to highest
                 mps_power: int = 1, mps_coeff: Numeric = 1.0,
                 mps_func: Callable = None,
                 max_bond: int = None, cutoff: float=CUTOFF,
                 num_tiers: int = 1,  ## number of intermediate kets in between operators = num_tiers - 1
                 # **canon_kwargs
                 ):

        self.initialized = False

        if operators is not None and len(operators) > 0:
            max_mpo_poly = len(mpo_poly) - 1 if isinstance(mpo_poly, (list, tuple)) else abs(mpo_poly)
        else:
            max_mpo_poly = 0

        self.num_tiers = min(num_tiers, max_mpo_poly + 1)

        self.op_blocks: dict[int, Sequence[BlockOperator]] = {0: []}
        self.vec_block: BlockVector = None

        self._ket = None
        self._bra = None
        self._operators = []

        self.ket = ket
        self.bra = bra
        self.operators = operators
        self.operator_k = operator_k  ## applied to ket before any power/coefficient
        self.cur_orthog = cur_orthog
        self._direction = direction

        if mps_func is not None:
            self.mps_power = None
            self.mps_coeff = None
            self.mps_func = mps_func
        else:
            self.mps_power = mps_power
            self.mps_coeff = mps_coeff
            if mps_func is None:
                mps_func = tc.get_element_wise_func(lambda x: mps_coeff * x ** mps_power)
            self.mps_func = mps_func
        self.eval_func = self.get_callable(mpo_poly, mps_func)
        self.mpo_poly = mpo_poly if operators is not None and len(operators) > 0 else None

        self.max_bond = max_bond
        self.cutoff = cutoff
        # self._proj_vec: qtn.Tensor = None        ## evaluated vecblock site
        self._proj_vec_targets: Sequence[qtn.Tensor] = []  ## extra targets for vecblock bra (mostly for PowerKet)
        self._out_site: Optional['qtn.Tensor'] = None      ## new tensor core following function evaluation of active site
        self._intermediate_sites: dict[int, Sequence[qtn.Tensor]] = {}    ## evaluated site for intermediate kets
        self._intermediate_kets: dict[int, MPS] = {}
        self._init_intermediate_ket = None
        ## intermediate projectors. these don't actually store the final solution, just the projectors for it

        # self.canon_kwargs = canon_kwargs

    @property
    def type(self) -> LocalSolverType:
        raise NotImplementedError

    @property
    def init_intermediate_ket(self):
        return self._init_intermediate_ket

    @init_intermediate_ket.setter
    def init_intermediate_ket(self, mps):
        mps = mps.copy()
        mps.distribute_exponent()
        self.canonize_func(mps, 0 if self._direction > 0 else -1)
        helper_quimb.match_inner_inds(mps, self.ket, inplace=True)
        self._init_intermediate_ket = mps

    # @property
    # def cur_orthog(self) -> int:
    #
    #     obls, vbl = self.blocks
    #
    #     co = None
    #     if vbl is not None:
    #         co = vbl.cur_orthog
    #         for obl in obls:
    #             if co != obl.cur_orthog:
    #                 raise ValueError('term 3 blocks do not have orthog center')
    #
    #     return co

    @classmethod
    def get_callable(cls, mpo_poly=1, mps_func=None):

        def eval_func(site_tens: 'qtn.Tensor', eff_ops: dict[int, Sequence['qtn.TensorNetwork']],
                      output_to_input_inds: dict[str, str], return_intermediates: bool = False
                      ) -> Union['qtn.Tensor', tuple['qtn.Tensor', dict[int, 'qtn.Tensor']]]:
            """ intermediate values only needed for intermediate kets
                1: vec_block value
                2: vec_block * op_eff value
                3: vec_block * op_eff[1] * op_eff[2] value
            """
            # print('IN EVAL FUNC')
            num_tiers = len(eff_ops) if eff_ops is not None else 0

            site_tens = site_tens.copy()
            if mps_func is not None:
                try:
                    site_tens = mps_func(site_tens)
                except TypeError:
                    site_tens.modify(apply=lambda x: mps_func(x))
                # print('mps func', mps_func(qtn.Tensor(np.array([1.0]), inds=('a',))).data)

            intermediates = {0: [site_tens.copy()]} # if num_tiers > 1 else {}      ## mps_func(proj_vec)
            if eff_ops is not None and len(eff_ops[0]) > 0:
                if mpo_poly == -1:
                    cgd_func = tc.get_cgd_func()
                    out_tens, err = cgd_func(site_tens, eff_ops[0], output_to_input_inds)
                    print('cgd err', err)
                elif mpo_poly == 0:
                    out_tens = site_tens    ## technically this should return 1??
                    raise NotImplementedError
                elif isinstance(mpo_poly, int):
                    power_func = tc.get_power_func(mpo_poly)
                    out_tens, intermediates = power_func(site_tens, eff_ops, output_to_input_inds,
                                                         return_intermediates=True)
                    # print('intermediates', intermediates)
                    # intermediates[0] = [site_tens.copy()]
                    # print('!!!! power func intermediates', intermediates)
                    # print('eval eff op', out_tens.norm())
                elif isinstance(mpo_poly, (list, tuple)):
                    apply_func = tc.apply_effective_op

                    out_tens_list = []
                    for it, coeff in enumerate(mpo_poly):
                        if coeff != 0:
                            out_tens = site_tens.copy()
                            out_tens.modify(apply=lambda x: x * coeff)
                            out_tens_list += [out_tens]
                        if it < len(mpo_poly) - 1:
                            it_ = min(it, num_tiers - 1)
                            site_tens = apply_func(site_tens, eff_ops[it_], output_to_input_inds)
                            # if it_ + 1 < num_tiers:
                            #     print('set intermediate', it_)
                            #     intermediates[it_] = [site_tens.copy()]
                    out_tens = helper_tn.sum_tens(out_tens_list)
                else:
                    raise NotImplementedError
            else:
                out_tens = site_tens

            if return_intermediates:
                # print('eval func intermediates !!', intermediates)
                return out_tens, intermediates

            return out_tens

        return eval_func

    # @property
    # def proj_vec(self) -> 'qtn.Tensor':
    #     return self._proj_vec

    @property
    def proj_vec_targets(self) -> Sequence['qtn.Tensor']:
        return self._proj_vec_targets

    @property
    def intermediate_sites(self) -> dict[int, Sequence['qtn.Tensor']]:
        return self._intermediate_sites

    @property
    def evaluated_site(self) -> 'qtn.Tensor':
        return self._out_site

    @classmethod
    def canonize_func(cls, mps, orthog, cur_orthog: int = None, **kwargs):
        raise NotImplementedError

    @classmethod
    def check_func(cls, mps):
        raise NotImplementedError

    @classmethod
    def canonize_tens_list(cls, tens_list, inplace=True, **kwargs):
        raise NotImplementedError

    @classmethod
    def compress_tens_list(cls, tens_list, inplace=True, max_bond=None, **kwargs):
        raise NotImplementedError

    @property
    def L(self) -> int:
        return self.ket.L

    @property
    def ket(self) -> 'MPS':
        return self._ket

    @ket.setter
    def ket(self, ket: 'MPS'):
        if not isinstance(ket, MPS):
            ket.view_as(MPS, inplace=True)
        self._ket = ket

    def reinitialize(self):
        self.vec_block.reinitialize()
        for it, blocks in self.op_blocks.items():
            for block in blocks:
                block.reinitialize()

    def canonize(self, orthog, cur_orthog: int = None):
        # print('term sel inds', self.ket.select_inds, self.bra.select_inds)
        # print('self. num tiers', self.num_tiers)
        # print('self. intermediate kets', self._intermediate_kets)
        # exit()
        ### assume self.ket, self.bra orthogonality doesn't matter,
        ### or is already of the appropriate canonicalization
        # self.canonize_func(self.ket, orthog, cur_orthog=cur_orthog)
        # self.canonize_func(self.bra, orthog, cur_orthog=cur_orthog)

        # print('self.initialized', self.initialized)
        # if self.initialized:
        #     print('self.ket', self.vec_block.ket.select_inds)
        #     print('self.bra', self.vec_block.bra.select_inds)
        for it, inter_ket in self._intermediate_kets.items():
            # print("???")
            # print('inter ket sel inds', inter_ket.select_inds)
            self.canonize_func(inter_ket, orthog, cur_orthog=cur_orthog)
            # print('inter ket sel inds', inter_ket.select_inds)
            # exit()

        # print('term sel inds', self.ket.select_inds, self.bra.select_inds)
        return self.ket

    def check_orthog(self):
        ### ket does not need to be in canonical form
        # o1, o2 = self.check_func(self.ket)
        # if self.operators is not None and len(self.operators) > 0:
        #     assert(o1 == o2 and o1 == self.cur_orthog), f'term ket center {(o1,o2)}; term center {self.cur_orthog}'

        # o1, o2 = self.check_func(self.ket)
        # # assert (o1 == o2 and o1 == self.cur_orthog), f'term bra center {(o1, o2)}; term center {self.cur_orthog}'
        # assert (o1 == o2), f'term ket center {(o1, o2)}; term center {self.cur_orthog}'

        o1, o2 = self.check_func(self.bra)
        # assert (o1 == o2 and o1 == self.cur_orthog), f'term bra center {(o1, o2)}; term center {self.cur_orthog}'
        # assert (o1 == o2), f'term bra center {(o1, o2)}; term center {self.cur_orthog}'

        for it, inter_ket in self._intermediate_kets.items():
            o1, o2 = self.check_func(inter_ket)
            # assert (o1 == o2 and o1 == self.cur_orthog), f'term intermediate ket {it} center {(o1, o2)};' \
            #                                              f' term center {self.cur_orthog}'
            assert (o1 == o2), f'term intermediate ket {it} center {(o1, o2)};' \
                                                         f' term center {self.cur_orthog}'
        return


    @property
    def ket_ind_id(self):
        return self.ket.site_ind_id

    # @property
    # def bra(self) -> Optional['MPS']:
    #     return self._intermediate_kets[self.num_tiers-1]

    @property
    def bra(self) -> Optional['MPS']:
        return self._bra if self._bra is not None else self.ket

    @bra.setter
    def bra(self, bra: 'MPS'):
        if bra is not None:
            if not isinstance(bra, MPS):
                bra.view_as(MPS, inplace=True)
            self._bra = bra

            # op_blocks = self.op_blocks.get(self.num_tiers - 1, [])
            # if len(op_blocks) > 0:
            #     for op_block in op_blocks:
            #         op_block.bra = self._bra
            # else:
            #     if self.vec_block is not None:
            #         self.vec_block.bra = self._bra



    def canonize_bra(self, cur_orthog):
        if self.bra is not None:
            self.canonize_func(self.bra, cur_orthog)
        return self.bra

    @property
    def bra_ind_id(self):
        return self.ket.site_ind_id + '_'

    @property
    def operators(self) -> Sequence['qtn.MatrixProductOperator']:
        return self._operators

    @operators.setter
    def operators(self, operators: Optional[Sequence[Union[Sequence[MPType], MPType]]]):
        if operators is None:
            operators = []

        operators = [(op.copy() if isinstance(op, qtn.MatrixProductOperator) else op) for op in operators]

        for op in operators:
            op.upper_ind_id = self.bra_ind_id
            op.lower_ind_id = self.ket_ind_id

        self._operators = operators

    @property
    def direction(self) -> SweepDirection:
        return self._direction

    @direction.setter
    def direction(self, new_direction: SweepDirection):
        self._direction = new_direction
        if self.initialized:
            for it, blocks in self.op_blocks.items():
                for block in blocks:
                    block.direction = self._direction
            self.vec_block.direction = self._direction


    def get_intermediate_ket(self, tier: int):
        # it_ = min(tier, self.num_tiers - 1)
        if tier >= self.num_tiers - 1:
            return self.bra
        return self._intermediate_kets[tier]


    def copy_new(self, ket_copy=None, bra_copy=None):

        if ket_copy is None:
            ket_copy = self.ket.copy()

        if bra_copy is None:
            bra_copy = self.bra.copy() if self._bra is not None else None

        # out = self.__class__(ket, bra, )
        out = self.__class__(ket_copy, bra_copy,
                             [op.copy() for op in self.operators],
                             operator_k=self.operator_k,
                             cur_orthog=self.cur_orthog,
                             direction=self.direction,
                             num_tiers=self.num_tiers,
                             max_bond=self.max_bond, cutoff=self.cutoff)
        out.mps_func = self.mps_func
        out.mpo_poly = self.mpo_poly
        out.eval_func = self.eval_func

        out._intermediate_kets = {k: v.copy() for k, v in self._intermediate_kets.items()}
        out._out_site = self._out_site.copy() if self._out_site is not None else None
        out._init_intermediate_ket = self.init_intermediate_ket

        return out

    def copy(self, ket_copy=None, bra_copy=None):

        out = self.copy_new(ket_copy, bra_copy)

        ket_copy = out.ket

        # if ket_copy is None:
        #     ket_copy = self.ket.copy()
        #
        # if bra_copy is None:
        #     bra_copy = self.bra.copy() if self.bra is not None else None
        #
        # # out = self.__class__(ket, bra, )
        # out = self.__class__(ket_copy, bra_copy, [op.copy() for op in self.operators],
        #                      cur_orthog=self.cur_orthog,
        #                      direction=self.direction,
        #                      max_bond=self.max_bond)
        # out.eval_func = self.eval_func

        if self.initialized:
            vec_block = self.vec_block.copy(ket_copy=ket_copy, bra_copy=out.get_intermediate_ket(0))
            # if len(self.op_blocks) == 0:
            #     vec_block = self.vec_block.copy(ket_copy=ket_copy, bra_copy=inter_kets[0])
            # else:
            #     vec_block = self.vec_block.copy(ket_copy=ket_copy)

            op_blocks_dict = {}
            for i in range(self.num_tiers - 1):
                op_blocks = []
                for op_block in self.op_blocks[i]:
                    op_blocks += [op_block.copy(ket_copy=out.get_intermediate_ket(i),
                                                bra_copy=out.get_intermediate_ket(i+1))]
                op_blocks_dict[i] = op_blocks

            out.op_blocks = op_blocks_dict
            out.vec_block = vec_block

        out.initialized = self.initialized
        out._init_intermediate_ket = self.init_intermediate_ket
        return out


    def match_inds(self, other: 'Term'):
        self.ket.site_ind_id = other.ket.site_ind_id

        helper_quimb.match_inner_inds(self.ket, other.ket)
        helper_quimb.match_inner_inds(self.ket, other.ket)

        # if self.bra is not None:
        #     self.bra.site_ind_id = other.bra_ind_id
        #     helper_quimb.match_inner_inds(self.bra, other.bra)
        #     helper_quimb.match_inner_inds(self.bra, other.bra)
        for tier, inter_ket in self._intermediate_kets.items():
            inter_ket.site_ind_id = self.ket.site_ind_id
            helper_quimb.match_inner_inds(inter_ket, self.ket)

        try:
            ref_op = next(iter(other.operators))
        except StopIteration:
            ref_op = None

        for op in self.operators:
            op.upper_ind_id = self.bra_ind_id
            op.lower_ind_id = self.ket_ind_id
            if ref_op is not None:
                helper_quimb.match_inner_inds(op, ref_op)


    def initialize(self):
        # if self.initialized:
        #     return

        # ## officially setting what was stored, does index matching
        # self.ket = self.ket
        # self.bra = self.bra
        # self.operators = self.operators

        # print('cur orthog', self.cur_orthog)

        cur_orthog = self.cur_orthog
        if cur_orthog is None:
            cur_orthog = 0 if self.direction == SweepDirection.RIGHT else self.L - 1
            # print('term init', cur_orthog)
        self.canonize(cur_orthog)
        self.cur_orthog = cur_orthog

        print('term init?')
        try:
            check_func = self.check_func
            check_func(self.bra)
        except:
            self.canonize_func(self.bra, cur_orthog)
        print('term init bra checked')

        self.initialize_blocks()

        self.initialized = True

    def initialize_blocks(self):
        raise NotImplementedError

    @property
    def blocks(self):
        return self.op_blocks, self.vec_block

    def projected_bra_to_ket(self, left_site_pos: int, nsites: int):
        if len(self.op_blocks) == 0 or len(self.op_blocks[0]) == 0:
            b_to_k = self.vec_block.projected_bra_to_ket(left_site_pos, nsites)
        else:
            b_to_k = self.op_blocks[0][0].projected_bra_to_ket(left_site_pos, nsites)
        return b_to_k


    # def get_projected_intermediate_kets(self, left_site_pos: int, nsites: int) -> dict[int, 'qtn.Tensor']:
    #
    #     b2k_dict = self.vec_block.projected_bra_to_ket(left_site_pos, nsites)
    #     site_tens = self.vec_block.get_projected(left_site_pos, nsites, return_combined=True)   ## includes mps func
    #     site_tens.reindex(b2k_dict, inplace=True)
    #
    #     # if site_tens is None:
    #     #     site_tens = self.vec_block.get_projected(left_site_pos, nsites, return_combined=True)
    #     #     site_tens.reindex(b2k_dict, inplace=True)
    #     # else:
    #     #     site_tens = site_tens.copy()
    #
    #     # print('b2k dict', b2k_dict)
    #
    #     eff_ops = {}
    #     for it, blocks in self.op_blocks.items():
    #         eff_ops_list = []
    #         for env in blocks:
    #             eff_op = env.get_projected(left_site_pos, nsites, return_combined=False)
    #             eff_ops_list += [eff_op]
    #             b2k_dict = env.projected_bra_to_ket(left_site_pos, nsites)
    #         eff_ops[it] = eff_ops_list
    #
    #     output_to_input_inds = b2k_dict
    #     out_site = self.eval_func(site_tens, eff_ops, output_to_input_inds)
    #     ## somehow extra intermediate terms?
    #
    #     ## need to reindex to ket inds
    #     out_site.reindex(b2k_dict, inplace=True)
    #
    #     vec_eval = self.mps_func(site_tens) if self.mps_func is not None else site_tens
    #         ## only different for DMRG Power Ket
    #     vec_eval.modify(apply=lambda x: x * 10 ** (-self.bra.exponent))
    #
    #     site_tens.modify(apply=lambda x: x * 10 ** (-self.bra.exponent))
    #     out_site.modify(apply=lambda x: x * 10 ** (-self.bra.exponent))
    #
    #     if isinstance(self.vec_block, BlockPowerKet_DMRG):
    #         targets = [t.copy() for t in self.vec_block.get_pows(left_site_pos, nsites)]
    #         for t in targets:
    #             # t.modify(apply=lambda x: x * 10 ** (-self.bra.exponent))
    #             t.modify(apply=lambda x: x / t.norm())
    #         self._extra_targets = targets
    #         self._proj_vec = self.get_projected_intermediate_kets(left_site_pos, nsites, )
    #         self._proj_vec_eval = vec_eval
    #     else:
    #         # self._proj_vec[0] = site_tens.copy()
    #         self._proj_vec = self.get_projected_intermediate_kets(left_site_pos, nsites, )
    #         self._proj_vec_eval = vec_eval
    #
    #     return out_site


    def update_intermediate_kets(self, i: int, nsites: int, direction: SweepDirection,):

        if nsites == 1 or direction == SweepDirection.RIGHT:
            left_site_pos = i
        else:
            left_site_pos = i - 1

        # print('update intermediate kets?', self.num_tiers)
        # print('intermediate sites', self.intermediate_sites)
        # exit()

        at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

        for it in range(self.num_tiers - 1):

            # if isinstance(self, Term_DMRG) and isinstance(self.vec_block, BlockPowerKet_DMRG):
            #     new_ket_site = [self.proj_vec_eval] + [*self.proj_vec[it]]
            # if isinstance(self, Term_DMRG):
            #     # new_ket_site = [*self.proj_vec_eval, self.proj_vec[it]]
            #     new_ket_site = self.proj_vec[it]
            # else:   ## is Term_Cross
            #     new_ket_site = self.proj_vec_eval

            new_ket_site = self.intermediate_sites[it]    ## list of tensors

            # print('new ket site', new_ket_site, len(new_ket_site))
            # print('self.proj vec targets', self.proj_vec_targets, len(self.proj_vec_targets))
            # pdb.set_trace()

            ### altered PowerKet to not need to do this. but doesn't really change anything
            # if it == 0 and not at_end and isinstance(self, Term_DMRG):  ## works for Cross but not necessary?
            #     new_ket_site = [*new_ket_site, *self.proj_vec_targets]    # self.proj_vec,
            #     ## i think proj_vec is included in proj_vec_targets

            if isinstance(self, Term_DMRG):
                if nsites == 1:
                    helper_dmrg.update_1site(self._intermediate_kets[it], left_site_pos, new_ket_site, direction,
                                             max_bond=self.max_bond, cutoff=self.cutoff)

                else:
                    helper_dmrg.update_2site(self._intermediate_kets[it], left_site_pos, new_ket_site, direction,
                                             max_bond=self.max_bond, cutoff=self.cutoff)
            elif isinstance(self, Term_Cross):

                ### select inds is bra inds

                if nsites == 1:
                    # helper_cross.update_1site(self._intermediate_kets[it], left_site_pos, new_ket_site, direction,
                    #                           select_inds=self.bra.select_inds[left_site_pos],
                    #                           max_bond=self.max_bond)
                    helper_cross.update_ket(self._intermediate_kets[it], new_ket_site, i, 1, direction=direction,
                                            max_bond=self.max_bond, cutoff=self.cutoff,
                                            decimate_only=(not at_end))

                elif nsites == 2:
                    ind1, ind2 = (left_site_pos, left_site_pos + 1) if direction > 0 else (left_site_pos + 1, left_site_pos)
                    # helper_cross.update_2site(self._intermediate_kets[it], left_site_pos, new_ket_site, direction,
                    #                           # select_inds=self.bra.select_inds[left_site_pos],
                    #                           inds_r = self.bra.select_inds[ind1],
                    #                           inds_c = self.bra.select_inds[ind2],
                    #                           max_bond=self.max_bond)
                    helper_cross.update_ket(self._intermediate_kets[it], new_ket_site, i, 2, direction=direction,
                                            max_bond=self.max_bond, cutoff=self.cutoff,
                                            decimate_only=(not at_end))
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            # self.vec_block.update_bra(left_site_pos, nsites, direction, new_ket_site, max_bond=self.max_bond)

        return


    def extend_env(self, i, direction): #, new_ket_site=None):

        # self.vec_block.decimate(i, direction=direction, max_bond=self.max_bond) #, new_ket_site=new_ket_site)
        # self.update_intermediate_kets(i, 1, direction)
        self.vec_block.extend_env(i, direction=direction)
        for it, blocks in self.op_blocks.items():
            for block in blocks:
                block.extend_env(i, direction=direction)

        self.cur_orthog = i + direction
        self._intermediate_sites = {}
        self._proj_vec_targets = []


    def get_evaluated_site(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor'=None,
                           verbose_plot=False):
        #, direction: SweepDirection, site_tens=None,):

        # print('get_evaluated_site site tens:', site_tens)

        ### when site_tens is provided, it is usually in the bra coordinates
        ### which is different from the intermediate ket coordinates
        ### so, we need to have bra select inds and intermediate ket select inds be the same

        if verbose_plot:
            plt.figure()

            plt.plot(self.ket.to_dense(), label='ket')
            plt.plot(self.bra.to_dense(), label='bra/out')
            if self.num_tiers > 1:
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

        b2k_dict = self.vec_block.projected_bra_to_ket(left_site_pos, nsites)
        if site_tens is None:
            site_tens, vec_targets = self.vec_block.get_projected(left_site_pos, nsites, return_combined=True,
                                                                  return_intermediates=True)
            site_tens.reindex(b2k_dict, inplace=True)
            for t in vec_targets:
                t.reindex(b2k_dict, inplace=True)

            # helper_cross.plot_submat(self.vec_block.bra,
            #                          left_site_pos, nsites, site_tens,
            #                          ref_kets=[self.ket, self.get_intermediate_ket(0)], plt_title='term vec block')

            # print('site tens', site_tens, site_tens.norm())
            # print('vec targets', vec_targets)

            ## keep?
            # for t in vec_targets:
            #     t.modify(apply=lambda x: x / t.norm())

            vec_targets = [t.copy() for t in vec_targets]
            for t in vec_targets:
                t.modify(apply=lambda x: x * 10 ** (-self.bra.exponent))

            # self._proj_vec_targets = vec_targets
            self._proj_vec_targets += vec_targets
        else:

            # helper_cross.plot_submat(self.bra, left_site_pos, 1, site_tens,
            #                          ref_kets=[self.bra, self.ket], plt_title='site_tens')

            site_tens, vec_targets = self.vec_block.get_projected(left_site_pos, nsites, return_combined=True,
                                                                  return_intermediates=True,
                                                                  site_tens=site_tens)
            # site_tens, vec_targets = site_tens.copy(), [site_tens.copy()]

            site_tens.reindex(b2k_dict, inplace=True)
            for t in vec_targets:
                t.reindex(b2k_dict, inplace=True)

            # site_tens = site_tens.copy()
            # vec_targets = [site_tens.copy()]
            for t in vec_targets:
                t.modify(apply=lambda x: x * 10 ** (-self.bra.exponent))
            # self._proj_vec_targets = vec_targets
            self._proj_vec_targets += vec_targets

            # if self.num_tiers > 1:
            #     print('self exponents', self.ket.exponent, self.bra.exponent, self._intermediate_kets[0].exponent)
            #     helper_cross.plot_submat(self.bra, left_site_pos, nsites, site_tens,
            #                              ref_kets=[self.bra, self.ket], plt_title='site_tens 2')
            #     helper_cross.plot_submat(self.vec_block.bra, left_site_pos, nsites, tmp,
            #                              ref_kets=[self.vec_block.ket, self.vec_block.bra], plt_title='site_tens 2')
            #
            #     # for vt in vec_targets:
            #     #     helper_cross.plot_submat(self.bra, left_site_pos, 2, vt,
            #     #                              ref_kets=[self.bra, self.ket], plt_title='site_tens vec targets')

        if 0 in self._intermediate_kets:
            self._intermediate_sites[0] = vec_targets

        # ########### plot ####
        if verbose_plot and self.__class__ is Term_Cross:
            coords = helper_cross.get_selectors(self.bra, left_site_pos, nsites)
            # print('left site pos', left_site_pos)
            # print('ket', self.ket)
            # print('bra', self.bra)
            # print('tens', site_tens)

            selectors = []
            for c in coords:
                selectors += [int("".join(str(x) for x in c), 2)]
            # print('coords', coords)
            # print('selectors', selectors)

            inds = []
            if left_site_pos > 0:
                inds += [self.ket.bond(left_site_pos, left_site_pos - 1)]
            inds += [self.ket.site_ind(left_site_pos + i) for i in range(nsites)]
            if left_site_pos + nsites < self.ket.L:
                inds += [self.ket.bond(left_site_pos + nsites - 1, left_site_pos + nsites)]

            plt.figure()
            site_tens.transpose(*inds, inplace=True)
            plt.plot(selectors, site_tens.data.reshape(-1), 'x', label='site tens')

            plt.plot(self.ket.to_dense(), label='ket')
            plt.plot(self.bra.to_dense(), label='bra/out')

            plt.title('get evaluated site')
            plt.legend()
            plt.show()
            #########

        # self._proj_vec = site_tens
        # print('site tens', site_tens.inds, site_tens.norm()) #, site_tens.data)

        # ## incorporate vec block bra exponent into site_tens
        # site_tens.modify(apply=lambda x: x * 10 ** self.vec_block.bra.exponent)

        # print('b2k dict', b2k_dict)
        bonds_o = [k for k in b2k_dict.keys() if k is not None]
        bonds_i = [b2k_dict[bo] for bo in bonds_o]

        eff_ops = {}
        for it, blocks in self.op_blocks.items():
            eff_ops_list = []
            for env in blocks:
                eff_op = env.get_projected(left_site_pos, nsites, return_combined=False)

                # op = qtn.tensor_contract(*eff_op.tensors)
                # op.transpose(*bonds_o, *bonds_i, inplace=True)
                # # print('op', env.bra is env.ket)
                # # sq_shape = int(np.round(np.sqrt(op.size)))
                # # print(op.inds, op.data.reshape(sq_shape,sq_shape))

                eff_ops_list += [eff_op]
                # b2k_dict = env.projected_bra_to_ket(left_site_pos, nsites)
            eff_ops[it] = eff_ops_list
            # print('eff ops', [t.norm() for t in eff_ops_list])

        # if self.num_tiers > 1:
        #     helper_cross.check_orthog(self.get_intermediate_ket(0))

        # print('self.eval func', self.eval_func)
        eff_ops = None if len(eff_ops) == 0 else eff_ops
        out_site, intermediates = self.eval_func(site_tens, eff_ops, b2k_dict, return_intermediates=True)
        # print('proj intermediates', intermediates)

        # print('left site pos', left_site_pos)
        # if self.num_tiers > 1:
        #     tmp = intermediates[0].copy()
        #     ket_copy = self._intermediate_kets[0].copy()
        #
        #     if nsites == 1:
        #         verbosity = 0 # 1 if (left_site_pos in [2,5,6]) else 0
        #         # print('bra', self.bra.select_inds)
        #         # print('ket copy', ket_copy.select_inds)
        #         # print('term direction', self.direction)
        #         helper_cross.update_1site(ket_copy, left_site_pos, tmp, direction=self.direction,
        #                                   plot_verbosity=verbosity)
        #     else:
        #         verbosity = 0 #1 if left_site_pos == 4 else 0
        #         helper_cross.update_2site(ket_copy, left_site_pos, tmp, direction=self.direction,
        #                                   plot_verbosity=verbosity)

        # print('intermediates', intermediates)
        for k, vals in intermediates.items():
            if k in self._intermediate_sites:
                self._intermediate_sites[k] += vals
            else:
                self._intermediate_sites[k] = vals
        # self._intermediate_sites = intermediates

        # print('updateed', self._intermediate_sites)
        # pdb.set_trace()
        self._out_site = out_site
        for it, inter in intermediates.items():
            # inter.modify(apply=lambda x: x * 10 ** (-self.bra.exponent))
            for t in inter:
                t.modify(apply = lambda x: x * 10 ** (-self.bra.exponent))

        ## need to reindex to ket inds
        out_site.reindex(b2k_dict, inplace=True)

        # print('remove bra exponent')
        site_tens.modify(apply = lambda x: x * 10 ** (-self.bra.exponent))
        out_site.modify(apply = lambda x: x * 10 ** (-self.bra.exponent))

        # print('len ops', len(self.op_blocks[0]))
        # op_ket_1 = helper_quimb.apply(self.op_blocks[0][0].operator, self.get_intermediate_ket(0))
        # op_ket_2 = helper_quimb.apply(self.op_blocks[0][0].operator, self.ket)
        # helper_cross.plot_submat(self.bra, left_site_pos, nsites, site_tens,
        #                          ref_kets=[op_ket_1, op_ket_2], plt_title='term op block')


        # print('term (self)', self, self.blocks)
        # if self.num_tiers > 1:
        #     helper_cross.plot_submat(self.bra, left_site_pos, 1, out_site,
        #                              ref_kets=[self.bra, self.ket], plt_title='term bra')
        #     for ix in range(self.num_tiers - 1):
        #         inter_tens = intermediates[ix]
        #         ket = self._intermediate_kets[ix]
        #         for inter in inter_tens:
        #             helper_cross.plot_submat(ket, left_site_pos, 1, inter,
        #                                      ref_kets=[self.bra, self.ket], plt_title=f'term inter {ix}')

        return out_site


    def get_eff_operator(self, left_site_pos: int, nsites: int, return_combined = False, transpose_bonds = None,
                         ) -> Union[dict[int,list], 'qtn.Tensor']:
        ### maybe this should return eff op for all levels combined?

        eff_ops = {it: [op.get_projected(left_site_pos, nsites, return_combined=False) for op in ops]
                   for it, ops in self.op_blocks.items()}

        if return_combined:
            for k, ops in eff_ops.items():
                eff_op = helper_tn.sum_eff_TNs(ops, transpose_bonds=transpose_bonds)
                eff_ops[k] = eff_op

        return eff_ops



class Term_DMRG(Term):

    def __init__(self,
                 ket: qtn.MatrixProductState,
                 bra: Optional[qtn.MatrixProductState] = None,
                 operators: Optional[Sequence[qtn.MatrixProductOperator]] = None,
                 operator_k: Optional[qtn.MatrixProductOperator] = None,
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 mpo_poly: Union[Sequence[int], int] = 1,
                 mps_power: int = 1, mps_coeff: Numeric = 1,
                 max_bond: int = None, cutoff: float=CUTOFF,
                 num_tiers: int = 1
                 ):

        mps_func = None
        if mps_coeff != 1:
            mps_func = tc.get_element_wise_func(lambda x: x * mps_coeff)
        super().__init__(ket, bra, operators,
                         operator_k=operator_k, cur_orthog=cur_orthog, direction=direction,
                         mpo_poly=mpo_poly, mps_func=mps_func, max_bond=max_bond, cutoff=cutoff, num_tiers=num_tiers)
        self.mps_power = mps_power
        # if mps_coeff != 1:
        #     self.mps_func = tc.get_element_wise_func(lambda x: x * mps_coeff)
        # else:
        #     self.mps_func = None

    @property
    def type(self) -> LocalSolverType:
        return LocalSolverType.DMRG

    # @classmethod
    # def canonize_func(cls, mps, orthog):
    #     helper_quimb.canonize(mps, i=orthog, scale=False, bra=None)
    #     mps.cur_orthog = orthog
    #     return mps

    @classmethod
    def canonize_func(cls, mps, orthog, cur_orthog:int =None, **kwargs):

        i = orthog
        # print('CANONIZE CUR ORTHOG', mps.cur_orthog)
        # print('check', helper_quimb.check_orthog(mps))
        cur_orthog = None  # mps.cur_orthog
        tens_left, tens_right = [], []
        max_ind = mps.L - 1
        updated_inds = []
        for lx in range(mps.L):
            if (0 if cur_orthog is None else cur_orthog) <= lx <= i:
                updated_inds += [lx]
                # tens_left += [mps.select_tensors((mps.site_tag_id.format(lx),))[0]]
                tens_left += [mps[lx]]
            if (max_ind if cur_orthog is None else cur_orthog) >= lx >= i:
                if lx not in updated_inds:
                    updated_inds += [lx]
                # tens_right += [mps.select_tensors((mps.site_tag_id.format(lx),))[0]]
                tens_right += [mps[lx]]
        # tens_left  = [self.ket[lx] for lx in sorted(self.mps_inds) if lx <= i]
        # tens_right = [self.ket[rx] for rx in sorted(self.mps_inds) if rx >= i]

        # print([t.copy() for t in tens_left])
        # print([t.copy() for t in tens_right])
        # print([tens_left])
        # print([tens_right])

        helper.canonize_tens_list(*tens_left, inplace=True)
        helper.canonize_tens_list(*(tens_right[::-1]), inplace=True)

        mps._cur_orthog = orthog
        return mps

    @classmethod
    def check_func(cls, mps):
        return helper_quimb.check_orthog(mps)
        # raise NotImplementedError

    @classmethod
    def canonize_tens_list(cls, tens_list, inplace=True, full_matrices=False):
        tens_list = helper_quimb.canonize_tens_list(*tens_list, inplace=inplace, full_matrices=full_matrices)
        return tens_list

    @classmethod
    def compress_tens_list(cls, tens_list, inplace=True, max_bond=None, **kwargs):
        tens_list = helper_quimb.compress_tens_list(*tens_list, inplace=inplace, compress_opts={'max_bond': max_bond})
        return tens_list

    # def canonize(self, cur_orthog):
    #     self.canonize_func(self.ket, cur_orthog)
    #     for it, inter_ket in self._intermediate_kets.items():
    #         self.canonize_func(inter_ket, cur_orthog)

    # def canonize_bra(self, cur_orthog):
    #     if self.bra is not None:
    #         self.canonize_func(self.bra, cur_orthog)

    def copy_new(self, ket_copy=None, bra_copy=None):
        out = super().copy_new(ket_copy, bra_copy)
        out.mps_power = self.mps_power
        return out

    def copy(self, ket_copy=None, bra_copy=None):
        out = super().copy(ket_copy, bra_copy)
        out.mps_power = self.mps_power
        return out

    def initialize_blocks(self):

        if self.init_intermediate_ket is None:
            intermediate_kets = {i: self.bra.copy() for i in range(self.num_tiers - 1)}
        else:
            self.canonize_func(self.init_intermediate_ket, 0 if self.direction > 0 else self.ket.L - 1)
            intermediate_kets = {i: self.init_intermediate_ket.copy() for i in range(self.num_tiers - 1)}

        # print('init blocks cur orthog', self.check_orthog)
        num_ops = 0 if self.operators is None else len(self.operators)

        ## projection of self.ket
        bra = intermediate_kets.get(0, self.bra if num_ops == 0 else self.ket)
        if self.mps_power == 1:
            if self.operator_k is None:
                vec_block = BlockVector_DMRG(self.ket, bra, cur_orthog=self.cur_orthog)
            else:
                vec_block = BlockPowerOpKet_DMRG(self.ket, bra, operator=self.operator_k,
                                                 cur_orthog=self.cur_orthog, power=self.mps_power)
                # vec_block = BlockOperator_DMRG(self.ket, bra, operator=self.operator_k,
                #                                cur_orthog=self.cur_orthog)
        else:
            if self.operator_k is None:
                vec_block = BlockPowerKet_DMRG(self.ket, bra, cur_orthog=self.cur_orthog, power=self.mps_power)
            else:
                vec_block = BlockPowerOpKet_DMRG(self.ket, bra, operator=self.operator_k,
                                                 cur_orthog=self.cur_orthog, power=self.mps_power)

        ## projection of operators
        op_blocks_all = {}
        for it in range(self.num_tiers):
            op_blocks = []
            ket = intermediate_kets.get(it, self.bra if it > 0 else self.ket)
            bra = intermediate_kets.get(it + 1, self.bra)

            for op in self.operators:
                if isinstance(op, qtn.MatrixProductOperator):
                    op_blocks += [BlockOperator_DMRG(ket, bra, operator=op.copy(),
                                                     anc_env_left=None, anc_env_right=None,
                                                     cur_orthog=self.cur_orthog)]
                elif isinstance(op, qtn.MatrixProductState):
                    op_blocks += [BlockDiagOperator_DMRG(ket, bra, operator=op.copy(),
                                                         anc_env_left=None, anc_env_right=None,
                                                         cur_orthog=self.cur_orthog)]
                else:
                    raise TypeError('not a valid type for operator')

            op_blocks_all[it] = op_blocks

        self._intermediate_kets = intermediate_kets
        self.op_blocks = op_blocks_all
        self.vec_block = vec_block
        # if vec_block.bra is not None and vec_block.bra is not self.bra:
        #     print('check orthog', helper.check_orthog(vec_block.bra))
        #     print('self.cur orthog', self.cur_orthog)
        #     exit()


    # def get_evaluated_rdm(self, left_site_pos: int, nsites: int, direction: SweepDirection,
    #                       transpose_inds: Sequence[str] = None, site_tens: 'qtn.Tensor' = None) -> 'qtn.Tensor':
    #
    #     if site_tens is None:
    #         site_tens = self.get_evaluated_site(left_site_pos, nsites)  ## has ket inds
    #
    #     at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)
    #     assert (not at_end), 'should not need to evaluate rdm for targeting if at end'
    #
    #     ## horizontal bond to contract over (right ind if sweep l2r, left ind if sweep r2l)
    #     if direction == SweepDirection.RIGHT:
    #         x_bond = self.ket.bond(left_site_pos + nsites, left_site_pos + nsites - 1)
    #     else:
    #         x_bond = self.ket.bond(left_site_pos, left_site_pos - 1)
    #
    #     ## bonds to not contract over
    #     b_to_k = self.projected_bra_to_ket(left_site_pos, nsites)
    #     k_to_b = {v: k for k, v in b_to_k.items()}
    #     ket_iso = [ind for ind in site_tens.inds if ind != x_bond]
    #     new_bra_iso = [k_to_b[k] for k in ket_iso]
    #
    #     if transpose_inds is None:
    #         transpose_inds = [*ket_iso, *new_bra_iso]
    #
    #     ## evaluated rdm
    #     rdm = helper_dmrg.get_tens_rdm(site_tens, ket_iso, new_bra_iso, transpose_inds=transpose_inds)
    #     # site_conj = site_tens.conj()
    #     # site_conj.reindex({k: b for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)
    #     # rdm = qtn.tensor_contract(site_conj, site_tens, output_inds=transpose_inds)
    #     # rdm.modify(apply = lambda x: x / rdm.norm())
    #     print('rdm 1 norm', rdm.norm())
    #
    #     # ## self vec rdm
    #     # if isinstance(self.vec_block, BlockPowerKet_DMRG):
    #     #     vec_tens = qtn.tensor_contract(*[self.vec_block.ket[si] for si in range(left_site_pos, left_site_pos + nsites)])
    #     #     vec_tens.modify(apply = lambda x: x * 10**self.vec_block.ket.exponent)
    #     #     # vec_tens = self.vec_block.get_projected(left_site_pos, nsites, return_combined=True)
    #     #     # vec_tens.reindex({b: k for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)
    #     #
    #     #     vec_rdm = helper_dmrg.get_tens_rdm(vec_tens, ket_iso, new_bra_iso, transpose_inds=transpose_inds)
    #     #     # vec_conj = vec_tens.conj()
    #     #     # vec_conj.reindex({k: b for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)
    #     #     # vec_rdm = qtn.tensor_contract(vec_conj, vec_tens, output_inds=transpose_inds)
    #     #
    #     #     # vec_rdm.modify(apply = lambda x: x / vec_rdm.norm())
    #     #     print('rdm 2 norm', vec_rdm.norm())
    #     #
    #     #     rdm = helper.add_tensors(rdm, vec_rdm)
    #
    #     return rdm


    # def extend_env(self, i, direction, new_ket_site=None):
    #     self.vec_block.decimate(i, direction=direction, new_ket_site=new_ket_site)
    #     self.vec_block.extend_env(i, direction=direction)
    #     for block in self.op_blocks:
    #         block.extend_env(i, direction=direction)



class Term_Cross(Term):

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

        if num_tiers is None:
            max_mpo_poly = len(mpo_poly) - 1 if isinstance(mpo_poly, (list, tuple)) else abs(mpo_poly)
            num_tiers = max_mpo_poly + 1

        super().__init__(ket, bra, operators,
                         operator_k=operator_k,
                         cur_orthog=cur_orthog, direction=direction,
                         mpo_poly=mpo_poly, mps_power=mps_power, mps_coeff=mps_coeff, mps_func=mps_func,
                         max_bond=max_bond, cutoff=cutoff, num_tiers=num_tiers)

    @property
    def type(self) -> LocalSolverType:
        return LocalSolverType.Cross

    @classmethod
    def canonize_func(cls, mps: MPS, orthog: int, cur_orthog: int=None, select_inds=None):
        print('X canonize mps cur orthog', mps._cur_orthog, cur_orthog)
        select_inds = mps.select_inds if select_inds is None else select_inds
        if select_inds is None:
            select_inds = {}
        helper_cross.canonize(mps, orthog, cur_orthog=cur_orthog, select_inds=select_inds)
        mps._cur_orthog = orthog
        return mps

    @classmethod
    def check_func(cls, mps):
        return helper_cross.check_orthog(mps)

    @classmethod
    def canonize_tens_list(cls, tens_list: list['qtn.Tensor'], inplace=True, site_ind_ids: Sequence[str] = None,
                           site_inds: Sequence[int] = None):
        """ site_ind_ids, site_inds not actually optional
        """
        tens_list, sel_inds =  helper_cross.canonize_tens_list(tens_list, site_ind_ids, site_inds,
                                                               inplace=inplace)
        return tens_list, sel_inds

    @classmethod
    def compress_tens_list(cls, tens_list: list['qtn.Tensor'], inplace=True, max_bond=None,
                           site_ind_ids: Sequence[str] = None, site_inds: Sequence[int] = None):
        """ site_ind_ids, site_inds not actually optional
        """
        tens_list, sel_inds = helper_cross.compress_tens_list(tens_list, site_ind_ids, site_inds,
                                                              inplace=inplace, max_bond=max_bond)
        return tens_list, sel_inds

    # def canonize(self, cur_orthog):
    #     self.canonize_func(self.ket, cur_orthog)
    #     for it, inter_ket in self._intermediate_kets.items():
    #         self.canonize_func(inter_ket, cur_orthog)

    # def canonize_bra(self, cur_orthog):
    #     if self.bra is not None:
    #         self.canonize(self.bra, cur_orthog)

    def copy_new(self, ket_copy=None, bra_copy=None):

        if ket_copy is None:
            ket_copy = self.ket.copy()

        if bra_copy is None:
            bra_copy = self.bra.copy() if self._bra is not None else None

        # out = self.__class__(ket, bra, )
        out = self.__class__(ket_copy, bra_copy,
                             [op.copy() for op in self.operators],
                             cur_orthog=self.cur_orthog,
                             direction=self.direction,
                             num_tiers=self.num_tiers,
                             max_bond=self.max_bond, cutoff=self.cutoff,
                             mps_func=self.mps_func)
        out.mps_func = self.mps_func
        out.mpo_poly = self.mpo_poly
        out.eval_func = self.eval_func

        out._intermediate_kets = {k: v.copy() for k, v in self._intermediate_kets.items()}

        return out


    def create_like(self, new_ket):

        out = self.__class__(new_ket, None,
                             [op.copy() for op in self.operators],
                             cur_orthog=self.cur_orthog,
                             direction=self.direction,
                             num_tiers=self.num_tiers,
                             max_bond=self.max_bond, cutoff=self.cutoff,
                             mps_func=self.mps_func)
        out.mps_func = self.mps_func
        out.mpo_poly = self.mpo_poly
        out.eval_func = self.eval_func

        return out


    # def canonize(self, orthog, cur_orthog: int = None):
    #     ### assume self.ket, self.bra orthogonality doesn't matter,
    #     ### or is already of the appropriate canonicalization
    #     # self.canonize_func(self.ket, orthog, cur_orthog=cur_orthog)
    #     # self.canonize_func(self.bra, orthog, cur_orthog=cur_orthog)
    #
    #     # print('self.initialized', self.initialized)
    #     # if self.initialized:
    #     #     print('self.ket', self.vec_block.ket.select_inds)
    #     #     print('self.bra', self.vec_block.bra.select_inds)
    #
    #
    #     sel_inds = self.bra.select_inds
    #     # print('canonize ket', orthog, cur_orthog)
    #     # print('sel inds', sel_inds)
    #     self.ket = helper_cross.target_select_inds(self.ket, sel_inds, inplace=True, direction=SweepDirection.LEFT)
    #
    #     print('check orthog bra 2')
    #     self.check_func(self.bra)
    #     self.check_func(self.ket)
    #     print('canonize term 2')
    #
    #     for it, inter_ket in self._intermediate_kets.items():
    #         self.canonize_func(inter_ket, orthog, cur_orthog=cur_orthog)
    #
    #     # self.check_orthog()
    #     # print('done canonize term')
    #
    #     return self.ket



    def canonize_ket_tens(self, left_site_pos: int, nsites: int, direction: SweepDirection, max_bond:int = None):

        if direction == SweepDirection.RIGHT:
            assert(left_site_pos < self.L - 1), 'cannot left canonicalize last site'
            site_inds = range(left_site_pos, left_site_pos + nsites + 1)
        else:
            assert(left_site_pos > 0), 'cannot right canonicalize 0th site'
            site_inds = range(left_site_pos + nsites - 1, left_site_pos - 2, -1)

        tens_list = [self.ket[ix] for ix in site_inds]
        _, select_inds = self.compress_tens_list(tens_list, site_ind_ids=[self.ket.site_ind_id],
                                                 site_inds=list(site_inds), inplace=True,
                                                 max_bond=max_bond)
        # print('select inds', select_inds)
        for ix, sel_inds in zip(site_inds, select_inds):
            # self.ket_select_inds[ix] = sel_inds
            self.ket.select_inds[ix] = sel_inds

        # print('self.ket select inds', self.ket.select_inds)
        self.ket._cur_orthog = site_inds[-1]
        self.cur_orthog = site_inds[-1]

        # print('canonnize ket tens check orthog')
        # print(helper_cross.check_orthog(self.ket, self.ket.select_inds, [self.ket.site_ind_id]))


    def initialize_blocks(self):

        # intermediate_kets = {i: self.bra.copy() for i in range(self.num_tiers - 1)}
        if self.init_intermediate_ket is None:
            intermediate_kets = {i: self.bra.copy() for i in range(self.num_tiers - 1)}
        else:
            intermediate_kets = {i: self.init_intermediate_ket.copy() for i in range(self.num_tiers - 1)}

        # print('init blocks cur orthog', self.check_orthog)
        num_ops = 0 if self.operators is None else len(self.operators)

        ## projection of self.ket  (have tier for self.ket, unless no operator is applied so tier is self.bra)
        bra = intermediate_kets.get(0, self.bra if num_ops == 0 else self.ket)
        vec_block = BlockVector_Cross(self.ket, bra, cur_orthog=self.cur_orthog)

        # ## projection of operators
        # op_blocks_all = {}
        # for it in range(self.num_tiers - 1):
        #     op_blocks = []
        #     ket = intermediate_kets.get(it, self.bra)
        #     bra = intermediate_kets.get(it + 1, self.bra)
        #
        #     for op in self.operators:
        #         if isinstance(op, qtn.MatrixProductOperator):
        #             op_blocks += [BlockOperator_Cross(ket, bra, operator=op,
        #                                              anc_env_left=None, anc_env_right=None,
        #                                              cur_orthog=self.cur_orthog)]
        #         elif isinstance(op, qtn.MatrixProductState):
        #             op_blocks += [BlockDiagOperator_Cross(ket, bra, operator=op,
        #                                                  anc_env_left=None, anc_env_right=None,
        #                                                  cur_orthog=self.cur_orthog)]
        #         else:
        #             raise TypeError('not a valid type for operator')
        #
        #     op_blocks_all[it] = op_blocks

        ## projection of operators
        op_blocks_all = {}
        for it, op in enumerate(self.operators):
            op_blocks = []
            ket = intermediate_kets.get(it, self.bra if it > 0 else self.ket) # if it > 0 else intermediate_kets.get(it, self.ket)
            bra = intermediate_kets.get(it + 1, self.bra)

            if isinstance(op, qtn.MatrixProductOperator):
                op_blocks += [BlockOperator_Cross(ket, bra, operator=op,
                                                  anc_env_left=None, anc_env_right=None,
                                                  cur_orthog=self.cur_orthog)]
            elif isinstance(op, qtn.MatrixProductState):
                op_blocks += [BlockDiagOperator_Cross(ket, bra, operator=op,
                                                      anc_env_left=None, anc_env_right=None,
                                                      cur_orthog=self.cur_orthog)]
            else:
                raise TypeError('not a valid type for operator')

            op_blocks_all[it] = op_blocks

        self._intermediate_kets = intermediate_kets
        self.op_blocks = op_blocks_all
        self.vec_block = vec_block

from local_solvers.terms_mixed import Term_Mixed

