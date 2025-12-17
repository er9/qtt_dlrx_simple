import numpy as np
from abc import ABC

from setup_.defaults import *
from local_solvers.blocks import *
# from local_solvers_old.helper_tn import get_mps_matching_inds, sum_tens
# from helper_cross_v3 import canonize, canonize_tens_list, compress_tens_list
import local_solvers.helper_cross as helper_cross
from local_solvers.mps_classes import MPS, MPO

MPType = Union['qtn.MatrixProductState', 'qtn.MatrixProductOperator']
SelectIndsType = dict[int, Sequence[int]]

class BlockType(Enum):
    DMRG = 'dmrg'
    CROSS = 'cross'

## todo: write wrapper to check if self.initialized

class Term(ABC):

    def __init__(self,
                 ket: Union['MPS',qtn.MatrixProductState],
                 bra: Optional[Union['MPS',qtn.MatrixProductState]],
                 operators: Optional[Sequence[qtn.MatrixProductOperator]],
                 eval_func: Callable,
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 # **canon_kwargs
                 ):

        self._ket = None
        self._bra = None
        self._operators = []

        self.ket = ket
        self.bra = bra
        self.operators = operators
        self.cur_orthog = cur_orthog
        self.direction = direction

        self.initialized = False
        if eval_func is None:
            def eval_func(site_tens: 'qtn.Tensor', *args, **kwargs):
                return site_tens.copy()
        self.eval_func = eval_func

        self.op_blocks: Sequence[BlockOperator] = []
        self.vec_block: BlockVector = None

        # self.canon_kwargs = canon_kwargs


    @classmethod
    def canonize(cls, mps, orthog):
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

    def canonize_ket(self, cur_orthog):
        self.canonize(self.ket, cur_orthog)
        return self.ket

    @property
    def ket_ind_id(self):
        return self.ket.site_ind_id

    @property
    def bra(self) -> Optional['MPS']:
        return self._bra

    @bra.setter
    def bra(self, bra: 'MPS'):
        if bra is not None:
            if not isinstance(bra, MPS):
                bra.view_as(MPS, inplace=True)
            self._bra = bra

    def canonize_bra(self, cur_orthog):
        if self.bra is not None:
            self.canonize(self.bra, cur_orthog)
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

    def copy(self, ket_copy=None, bra_copy=None):

        if ket_copy is None:
            ket_copy = self.ket.copy()

        if bra_copy is None:
            bra_copy = self.bra.copy() if self.bra is not None else None

        if len(self.op_blocks) == 0:
            vec_block = self.vec_block.copy(ket_copy=ket_copy, bra_copy=bra_copy)
        else:
            vec_block = self.vec_block.copy(ket_copy=ket_copy)

        op_blocks = []
        for op_block in self.op_blocks:
            op_blocks += [op_block.copy(ket_copy=ket_copy, bra_copy=bra_copy)]

        # out = self.__class__(ket, bra, )
        out = self.__class__(ket_copy, bra_copy, [op.copy() for op in self.operators],
                             self.eval_func,
                             cur_orthog=self.cur_orthog,
                             direction=self.direction)

        out.op_blocks = op_blocks
        out.vec_block = vec_block
        out.initialized = self.initialized

        # print('vec block bra is ket', vec_block.ket is out.ket)
        # print('out.ket is ket_copy', out.ket is ket_copy)
        # print('vec_block bra is bra_copy', vec_block.bra is out.bra)

        # print('COPIED TERM')
        # exit()

        return out

    def match_inds(self, other: 'Term'):
        self.ket.site_ind_id = other.ket.site_ind_id

        helper_quimb.match_inner_inds(self.ket, other.ket)
        helper_quimb.match_inner_inds(self.ket, other.ket)

        if self.bra is not None:
            self.bra.site_ind_id = other.bra_ind_id
            helper_quimb.match_inner_inds(self.bra, other.bra)
            helper_quimb.match_inner_inds(self.bra, other.bra)

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
        if self.initialized:
            return

        # ## officially setting what was stored, does index matching
        # self.ket = self.ket
        # self.bra = self.bra
        # self.operators = self.operators

        if self.cur_orthog is None:
            cur_orthog = 0 if self.direction == SweepDirection.RIGHT else self.L - 1
            self.canonize_ket(cur_orthog)
            self.canonize_bra(cur_orthog)
            self.cur_orthog = cur_orthog

        self.initialize_blocks()

        self.initialized = True

    def initialize_blocks(self):
        raise NotImplementedError

    @property
    def blocks(self):
        return self.op_blocks, self.vec_block

    def projected_bra_to_ket(self, left_site_pos: int, nsites: int):
        if len(self.op_blocks) == 0:
            b_to_k = self.vec_block.projected_bra_to_ket(left_site_pos, nsites)
        else:
            b_to_k = self.op_blocks[0].projected_bra_to_ket(left_site_pos, nsites)
        return b_to_k

    def extend_env(self, i, direction, new_ket_site=None):
        self.vec_block.decimate(i, direction=direction, new_ket_site=new_ket_site)
        self.vec_block.extend_env(i, direction=direction)
        for block in self.op_blocks:
            block.extend_env(i, direction=direction)

        self.cur_orthog = i + direction

    # def extend_env(self, i, direction):
    #     for block in self.blocks:
    #         block.extend_env(i, direction=direction)
        # for op_blocks, target_block in self.term_blocks:
        #     for op_block in op_blocks:
        #         if isinstance(op_block, BlockOperator):
        #             op_block.extend_env(i, direction=direction)
        #     target_block.extend_env(i, direction=direction)

    def get_evaluated_site(self, left_site_pos: int, nsites: int, site_tens=None):

        if not self.initialized:
            raise RuntimeError('term has not yet been initialized')

        # site_inds = list(range(left_site_pos, left_site_pos + nsites))
        # site_tens = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
        # site_tens.modify(apply=lambda data: data * 10 ** self.ket.exponent)
        if site_tens is None:
            site_tens = self.vec_block.get_projected(left_site_pos, nsites, return_combined=True)
        b2k_dict = self.vec_block.projected_bra_to_ket(left_site_pos, nsites)
        site_tens.reindex(b2k_dict, inplace=True)
        # print('b2k dict', b2k_dict)

        eff_ops = []
        for env in self.op_blocks:
            eff_op = env.get_projected(left_site_pos, nsites, return_combined=False)
            eff_ops += [eff_op]
            b2k_dict = env.projected_bra_to_ket(left_site_pos, nsites)

        output_to_input_inds = b2k_dict
        out_site = self.eval_func(site_tens, eff_ops, output_to_input_inds)

        ## need to reindex to ket inds
        # print('site tens', out_site)
        # print('b2k dict', b2k_dict)
        out_site.reindex(b2k_dict, inplace=True)
        # print('site tens', out_site)
        # print('DONE')

        return out_site



class Term_DMRG(Term):

    def __init__(self,
                 ket: qtn.MatrixProductState,
                 bra: Optional[qtn.MatrixProductState] = None,
                 operators: Optional[Sequence[qtn.MatrixProductOperator]] = None,
                 eval_func: Optional[Callable] = None,
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 ):
        super().__init__(ket, bra, operators, eval_func, cur_orthog=cur_orthog, direction=direction)


    @classmethod
    def canonize(cls, mps, orthog):
        helper_quimb.canonize(mps, i=orthog, scale=False, bra=None)
        mps.cur_orthog = orthog
        return mps

    @classmethod
    def canonize_tens_list(cls, tens_list, inplace=True, full_matrices=False):
        tens_list = helper_quimb.canonize_tens_list(*tens_list, inplace=inplace, full_matrices=full_matrices)
        return tens_list

    @classmethod
    def compress_tens_list(cls, tens_list, inplace=True, max_bond=None, **kwargs):
        tens_list = helper_quimb.compress_tens_list(*tens_list, inplace=inplace, compress_opts={'max_bond': max_bond})
        return tens_list

    def canonize_ket(self, cur_orthog):
        self.canonize(self.ket, cur_orthog)

    # def canonize_bra(self, cur_orthog):
    #     if self.bra is not None:
    #         self.canonize(self.bra, cur_orthog)

    def initialize_blocks(self):
        op_blocks = []
        target_use_bra = True

        for op in self.operators:
            if isinstance(op, qtn.MatrixProductOperator):
                op_blocks += [BlockOperator_DMRG(self.ket, self.bra, operator=op,
                                                 anc_env_left=None, anc_env_right=None,
                                                 cur_orthog=self.cur_orthog)]
                target_use_bra = False

            elif op == self.ket or op == 'ket' or isinstance(op, qtn.MatrixProductState):
                if op == 'ket':  op = None
                op_blocks += [BlockDiagOperator_DMRG(self.ket, self.bra, operator=op,
                                                     anc_env_left=None, anc_env_right=None,
                                                     cur_orthog=self.cur_orthog)]

            else:
                raise TypeError('not a valid type for operator')

        vec_conj = self.bra if target_use_bra else None
        vec_block = BlockVector_DMRG(self.ket, vec_conj, cur_orthog=self.cur_orthog)

        self.op_blocks = op_blocks
        self.vec_block = vec_block


    def get_evaluated_rdm(self, left_site_pos: int, nsites: int, direction: SweepDirection,
                          transpose_inds: Sequence[str] = None, site_tens: 'qtn.Tensor' = None) -> 'qtn.Tensor':

        print('here')

        if site_tens is None:
            print('site tens is None')
            site_tens = self.get_evaluated_site(left_site_pos, nsites)  ## has ket inds

        print('site tens norm', site_tens.norm())

        at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)
        assert (not at_end), 'should not need to evaluate rdm for targeting if at end'

        ## horizontal bond to contract over (right ind if sweep l2r, left ind if sweep r2l)
        if direction == SweepDirection.RIGHT:
            x_bond = self.ket.bond(left_site_pos + nsites, left_site_pos + nsites - 1)
        else:
            x_bond = self.ket.bond(left_site_pos, left_site_pos - 1)

        ## bonds to not contract over
        b_to_k = self.projected_bra_to_ket(left_site_pos, nsites)
        # if len(self.op_blocks) == 0:
        #     b_to_k = self.vec_block.projected_bra_to_ket(left_site_pos, nsites)
        # else:
        #     b_to_k = self.op_blocks[0].projected_bra_to_ket(left_site_pos, nsites)
        k_to_b = {v: k for k, v in b_to_k.items()}
        ket_iso = [ind for ind in site_tens.inds if ind != x_bond]
        new_bra_iso = [k_to_b[k] for k in ket_iso]

        if transpose_inds is None:
            transpose_inds = [*ket_iso, *new_bra_iso]

        ## evaluated rdm
        site_conj = site_tens.conj()
        site_conj.reindex({k: b for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)

        print('site', site_tens)
        print('site conj', site_conj)

        rdm = qtn.tensor_contract(site_conj, site_tens, output_inds=transpose_inds)

        print('rdm 1 norm', rdm.norm())

        ## self vec rdm
        if self.operators is None or len(self.operators) == 0 or isinstance(self.op_blocks[0], BlockDiagOperator_DMRG):
            vec_tens = self.vec_block.get_projected(left_site_pos, nsites, return_combined=True)
            vec_tens.reindex({b: k for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)
            vec_conj = vec_tens.conj()
            vec_conj.reindex({k: b for k, b in zip(ket_iso, new_bra_iso)}, inplace=True)
            vec_rdm = qtn.tensor_contract(vec_conj, vec_tens, output_inds=transpose_inds)

            print('rdm 2 norm', vec_rdm.norm())

            rdm = helper.add_tensors(rdm, vec_rdm)
            print('tot rdm norm', rdm.norm())

        return rdm


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
                 eval_func: Optional[Callable] = None,
                 cur_orthog: int = None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 # ket_select_inds: dict[int, Sequence[int]] = None,
                 # bra_select_inds: dict[int, Sequence[int]] = None,
                 ):

        # self.ket_select_inds = ket_select_inds
        # self.bra_select_inds = bra_select_inds
        # self.op_blocks: Sequence[Union[BlockOperator_Cross, BlockDiagOperator_Cross]] = []

        # if not isinstance(ket, MPS):
        #     ket.view_as(MPS, inplace=True)
        #     ket.select_inds = {} if ket_select_inds is None else ket_select_inds
        #
        # if bra is not None and not isinstance(bra, MPS):
        #     bra.view_as(MPS, inplace=True)
        #     bra.select_inds = {} if bra_select_inds is None else bra_select_inds

        super().__init__(ket, bra, operators, eval_func, cur_orthog=cur_orthog, direction=direction)


    @classmethod
    def canonize(cls, mps: MPS, orthog: int):
        if mps.select_inds is None:
            mps.select_inds = {}
        helper_cross.canonize(mps, orthog, select_inds=mps.select_inds)
        mps.cur_orthog = orthog
        return mps

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

    # def canonize_ket(self, cur_orthog):
    #     self.canonize(self.ket, cur_orthog)
    #
    # def canonize_bra(self, cur_orthog):
    #     if self.bra is not None:
    #         self.canonize(self.bra, cur_orthog)

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
        self.ket.cur_orthog = site_inds[-1]

        # print('canonnize ket tens check orthog')
        # print(helper_cross.check_orthog(self.ket, self.ket.select_inds, [self.ket.site_ind_id]))


    def initialize_blocks(self):
        op_blocks = []
        target_use_bra = True

        for op in self.operators:
            if isinstance(op, qtn.MatrixProductOperator):
                op_blocks += [BlockOperator_Cross(self.ket, self.bra, operator=op,
                                                  anc_env_left=None, anc_env_right=None,
                                                  # select_inds_bra=self.bra_select_inds,
                                                  # select_inds_ket=self.ket_select_inds,
                                                  cur_orthog=self.cur_orthog)]
                target_use_bra = False

            elif isinstance(op, qtn.MatrixProductState):
                op_blocks += [BlockDiagOperator_Cross(self.ket, self.bra, operator=op,
                                                      anc_env_left=None, anc_env_right=None,
                                                      cur_orthog=self.cur_orthog,
                                                      # select_inds_bra=self.bra_select_inds,
                                                      # select_inds_ket=self.select_inds_ket,
                                                      ## don't need select_inds_ket i think
                                                      )]

            else:
                raise TypeError('not a valid type for operator')

        vec_conj = self.bra if target_use_bra else None
        # sel_inds = self.bra_select_inds if target_use_bra else self.ket_select_inds
        vec_block = BlockVector_Cross(self.ket, vec_conj, cur_orthog=self.cur_orthog,
                                      )  #select_inds_bra=sel_inds)

        self.op_blocks = op_blocks
        self.vec_block = vec_block



