"""Base evaluator layer of the local-solver stack.

Defines :class:`LocalEvaluator`, the common base class for the DMRG, cross,
and mixed evaluators. It orchestrates the site-by-site sweep over an MPS --
canonicalization, collecting per-site Term contributions, combining them, and
updating the working state -- and provides the shared machinery
(:class:`SweepDirection`, :class:`SolveMethod`, index-matching helpers) that
the specialised evaluators and time integrators build upon.
"""
import numpy as np
from abc import ABC

import helper_quimb
import local_solvers.helper_cross_2 as helper_cross
from setup_.defaults import *
from local_solvers.defaults import *
import time
import scipy.sparse.linalg
from scipy import linalg
import quimb.tensor as qtn
# from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
# from local_solvers_old.environments_v2 import Environment, EnvironmentSide
# from local_solvers_old.environments_v2 import TargetEnv_DMRG, TargetEnv_Cross
# from local_solvers_old.environments_v2 import OperatorEnv_DMRG, OperatorEnv_Cross
# from local_solvers_old.local_solver import SweepDirection
# from local_solvers_old.blocks import EnvironmentType, init_block, Block, BlockOperator, BlockVector, BlockVector_Cross, BlockOperator_Cross

from gridTN_1D import GridTN1D
import local_solvers.helper_tn as helper_tn

# from local_solvers_old.local_evaluator import get_mps_matching_inds
# from local_solvers_old.helper_cross_v3 import check_left_orthog, check_right_orthog, check_orthog
# from local_solvers_old.helper_cross_v2 import maxvol_inds, canonize, cross_compress

# from local_solvers_old.local_solver_v2 import *  # LocalSolver
# from local_solvers_old.helper_cross_v3 import CrossSolver, canonize, canonize_tens_list, cross_compress
# from local_solvers_old.helper_cross_v3 import compress_tens_list
from local_solvers.mps_classes import MPS
from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross
from local_solvers.terms_mixed import Term_Mixed
import local_solvers.helper_mixed as helper_mixed

if TYPE_CHECKING:
    from grid import Grid
    from setup_.configs import DerivativeConfiguration


class SweepDirection(IntEnum):
    ## int denotes where the MPS needs to be canonicalized to
    LEFT = -1
    RIGHT = 1

class SolveMethod(Enum):
    CGD = 'CGD'  # local CGD using quimb tensors
    CGDx = 'CGDx'  # local CGD using numpy
    LSQ = 'LSQ'  # least squares regression


# DEFAULT_MAX_ITER = 50 # 20  # 10
# DEFAULT_MAX_TOT_ITER = 50
# DEFAULT_CONV_TOL = 1.0e-6
# DEFAULT_SOLVE = SolveMethod.CGD
# DEFAULT_MAX_WRONG_ITER = 50  # 10



# def local_cross_evaluator(*terms: tuple[ Sequence[Union[GridTN1D, Callable[[np.ndarray], np.ndarray]]],
#                                          GridTN1D ],
#                           nsites=1, direction: SweepDirection.LEFT, max_bond=None,
#                           combine_terms_func:Callable=None):
#     """ terms are sequences of ((A, B, ...), x)
#         where A, B are operators or functions that are applied to x in
#         the order A(B(x))
#
#         algorithm: compute rdm for each term
#         sum together terms by summing rdms together (state-averaging method)
#         projecting next site onto new rdm
#     """
#     opx, ref_x = terms[0]
#     L = ref_x.L
#
#     mps_terms = []
#     for ops, x in terms:
#         new_ops = [op.data if isinstance(op, GridTN1D) else op for op in ops]
#         mps_terms += [(new_ops, x.data.copy())]
#
#     solver = CrossEvaluator(mps_terms, combine_terms_func=combine_terms_func,
#                             init_direction=direction, max_bond=max_bond)
#     solver.solve(nsites)
#
#     out_gtn = GridTN1D(ref_x.grid, solver.ket)
#     return out_gtn


class LocalEvaluator:

    def __init__(self,
                 init_guess: Union['MPS', 'qtn.MatrixProductState'],
                 # bra_state: 'qtn.MatrixProductState',
                 terms: Sequence[Union['Term_DMRG','Term_Cross']],
                 # cur_orthog: int=None,
                 direction: SweepDirection = SweepDirection.RIGHT,
                 max_bond: int =None, cutoff: Numeric=CUTOFF,
                 conv_tol: float=DEFAULT_CONV_TOL, max_iter: int=DEFAULT_MAX_ITER,
                 max_tot_iter: int=DEFAULT_MAX_TOT_ITER, max_wrong_iter: int=DEFAULT_MAX_WRONG_ITER,
                 copy_obj: 'LocalEvaluator' = None,
                 combine_terms_func: 'Callable' = None,
                 grid: 'Grid' = None, ax_deriv_configs: dict['Axis','DerivativeConfiguration']=None,
                 ):

        self._direction = None

        if copy_obj:
            self.verbose = getattr(copy_obj, 'verbose', 0)
            self.L = copy_obj.L
            self.out = copy_obj.out.copy()
            self.out.view_as(MPS, inplace=True)
            # self.bra = copy_obj.bra.copy()
            new_terms = []
            for term in copy_obj.terms:
                if term is not None:
                    new_term = term.copy()
                    if term.bra is copy_obj.out:
                        new_term.bra = self.out
                    new_term.initialize()
                    new_terms += [new_term]
                else:
                    new_terms += [None]
            self._terms = new_terms

            self.direction = copy_obj.direction
            canon_site = 0 if self.direction == SweepDirection.RIGHT else self.out.L - 1
            if not np.isnan(self.cur_orthog):   # is nan means left_cur_orthog > right_cur_orthog
                assert(self.cur_orthog == canon_site), \
                    f'solver cur_orthog {self.cur_orthog} not consistent with direction {self.direction}'

            self.conv_tol = copy_obj.conv_tol
            self.max_tot_iter = copy_obj.max_tot_iter
            self.max_iter = copy_obj.max_iter
            self.max_wrong_iter = copy_obj.max_wrong_iter
            self.max_bond = copy_obj.max_bond
            self.cutoff = copy_obj.cutoff

            self.combine_terms_func = copy_obj.combine_terms_func
            self._local_solve_func = copy_obj._local_solve_func

            ## solution
            self._new_ket_site = copy_obj._new_ket_site
            # self._current_ket_sites = {k: v.copy() for k, v in copy_obj._current_ket_sites.items()}
            # self.out = copy_obj.out.copy()

            ## results
            self.err = copy_obj.err
            self.is_conv = copy_obj.is_conv
            # self.cur_orthog = copy_obj.cur_orthog

            self.grid = copy_obj.grid
            self.ax_deriv_configs = copy_obj.ax_deriv_configs

            # print('HERE')
            # self.check_orthog()
            # print('done init check')
        else:
            # subclasses (e.g. TimeIntegrator) may set self.verbose before super().__init__;
            # only default it here if not already set
            if not hasattr(self, 'verbose'):
                self.verbose = 0
            self.L : int = init_guess.L
            # self.cur_orthog = cur_orthog
            self.max_bond = max_bond
            self.cutoff = cutoff
            self._direction = direction

            if not isinstance(init_guess, MPS):
                init_guess.view_as(MPS, inplace=True)

            self.out = init_guess.copy()
            canon_site = 0 if self.direction == SweepDirection.RIGHT else self.L - 1
            canon_func = self.term_class().canonize_func
            canon_func(self.out, canon_site)

            # self.out = self.out.copy(deep=True)
            # self.bra = ket_state.conj(mangle_inner=True)
            if terms is None:
                terms = self.initialize_terms(self.out)
            self._terms = terms
            self.terms = terms      ## updates bra in the terms? and initialize

            ## solution
            self._new_ket_site: tuple[int, int, SweepDirection, Sequence[qtn.Tensor]] = tuple()

            ## convergence params
            self.conv_tol = conv_tol
            self.max_iter = max_iter
            self.max_tot_iter = max_tot_iter
            self.max_wrong_iter = max_wrong_iter

            ## callables for combining operators, targets. defaults to summation
            if combine_terms_func is None:
                combine_terms_func = helper_tn.sum_tens
            self.combine_terms_func = combine_terms_func
            self._local_solve_func = None

            ## results
            self.err = np.inf  # self.check_err()
            self.is_conv = False  # np.abs(self.err) < conv_tol

            self.grid = grid
            self.ax_deriv_configs = ax_deriv_configs

        self.out_norm = helper_quimb.norm(self.out)


    def initialize_terms(self, ket_state: Union['MPS', 'qtn.MatrixProductState'], cur_orthog=None,
                         **kwargs) -> Sequence[Term]:
        raise NotImplementedError

    @property
    def solver_type(self) -> LocalSolverType:
        raise NotImplementedError

    @property
    def solution(self):
        raise NotImplementedError

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction: SweepDirection):
        self._direction = direction
        for term in self.terms:
            if term is not None:
                term.direction = direction

    @property
    def conv_kwargs(self):
        out = {'conv_tol': self.conv_tol, 'max_iter': self.max_iter, 'max_tot_iter': self.max_tot_iter,
               'max_wrong_iter': self.max_wrong_iter,}
        return out

    @classmethod
    def term_class(cls) -> Union[Type[Term_DMRG], Type[Term_Cross]]:
        raise NotImplementedError

    @property
    def terms(self) -> Sequence[Union['Term_DMRG', 'Term_Cross']]:
        return self._terms
        ## terms can be created, and then initialized separately

    @terms.setter
    def terms(self, terms_list: list[Term]):
        canon_site = 0 if self.direction == SweepDirection.RIGHT else self.L - 1

        terms = []
        ref_term = None
        for term_ in terms_list:
            # if not isinstance(term_, Term):
            #     args, kwargs = term_
            #     term_ = self.term_class(*args, **kwargs)
            #     term_.bra = self.bra

            if term_ is None:
                terms += [term_]
                continue

            if ref_term is None:
                ref_term = term_
                if term_._bra is not None:
                    if self.verbose:
                        print('term bra is not None')
                    self.out = term_.bra
                    self.term_class().canonize_func(self.out, canon_site)
            else:
                term_.match_inds(ref_term)

            term_.bra = self.out
            self.term_class().canonize_func(term_.ket, canon_site)
            # if term_.bra is None:
            #     term_.bra = self.ket

            term_._direction = self.direction
            term_.max_bond = self.max_bond
            term_.initialize()
            # print('term_ setter cur orthog', self.direction, term_.cur_orthog)
            terms += [term_]

        self._terms = terms

    def copy(self):
        new_solver = self.__class__(self.out, self.terms, copy_obj=self)
        return new_solver

    # def get_ket_to_bra_inds(self, i) -> dict[str, str]:
    #     """ inds mapping inds on self.bra to corresponding inds on self.ket
    #     """
    #     return helper_tn.get_mps_matching_inds(self.ket, self.bra, i)
    #
    # def get_bra_to_ket_inds(self, i) -> dict[str, str]:
    #     ket_to_bra = self.get_ket_to_bra_inds(i)
    #     bra_to_ket = {item: k for k, item in ket_to_bra.items()}
    #     return bra_to_ket

    # def update_bra_from_ket(self, sites=None):
    #
    #     if self.bra is None:
    #         pass
    #     else:  ## inplace update
    #         sites = range(self.L) if sites is None else sites
    #         for ix in sites:
    #             ket_to_bra_inds = self.get_ket_to_bra_inds(ix)
    #             ket_tens = self.ket[ix].conj()
    #             ket_tens = ket_tens.reindex(ket_to_bra_inds)
    #             self.bra[ix].modify(data=ket_tens.data, inds=ket_tens.inds)
    #     return

    @property
    def cur_orthog(self) -> int:

        if self.term_class() is Term_DMRG:
            ket_orthog_l, ket_orthog_r = helper_quimb.check_orthog(self.out)
        elif self.term_class() is Term_Cross:
            ket_orthog_l, ket_orthog_r = helper_cross.check_orthog(self.out)
        elif self.term_class() is Term_Mixed:
            ket_orthog_l, ket_orthog_r = helper_mixed.check_orthog(self.out)
        else:
            raise TypeError

        if self.verbose:
            print('ket orthog l', ket_orthog_l, 'ket orthog r', ket_orthog_r)
        assert(ket_orthog_l>=ket_orthog_r), 'ket is not in orthogonal form'

        cur_orthog = np.nan
        if ket_orthog_l == ket_orthog_r:
            cur_orthog = ket_orthog_l
            for term in self.terms:
                if term is None:  continue

                # ## for nsites = 1
                # lb, ub = term.cur_orthog, term.cur_orthog

                ## for nsites = 2, at the ends
                if self.direction < 0:
                    lb, ub = term.cur_orthog, term.cur_orthog + 1
                else:
                    lb, ub = term.cur_orthog - 1, term.cur_orthog

                if not (lb <= cur_orthog <= ub):
                    if self.verbose:
                        print('cur orthog', cur_orthog, term.cur_orthog)
                    raise ValueError('orthogonalities of terms not consistent')

        return cur_orthog


    def check_orthog(self):
        raise NotImplementedError


    def canonize(self, target_orthog: int, cur_orthog: int = None):

        self.term_class().canonize_func(self.out, target_orthog, cur_orthog=cur_orthog)
        if self.out is not self.out:
            # self.out = self.ket.copy()
            self.term_class().canonize_func(self.out, target_orthog, cur_orthog=cur_orthog)

        # mps = self.ket
        # ind1, ind2 = 0, 1
        # mps[ind1].transpose('i(0)', mps.bond(0, 1), inplace=True)
        # mps[ind2].transpose('i(1)', mps.bond(0, 1), mps.bond(1, 2), inplace=True)
        # print('mps[ind1]', ind1, mps[ind1].inds, mps[ind1].norm())
        # print(mps[ind1].data)
        # print('mps[ind2]', ind2, mps[ind2].inds, mps[ind2].norm())
        # print(mps[ind2].data)

        # print('CANONIZE CHECK 1')
        # i = 0
        # print('ket vs out', helper_quimb.add_tensors(self.out[i], self.ket[i] * -1).norm())
        # print('ket vs out', helper_quimb.add_tensors(self.out[i + 1], self.ket[i + 1] * -1).norm())

        # self.update_bra_from_ket()

        # print('local eval canonize target orthog', target_orthog)
        if self.verbose:
            print('term canonize')
        for term in self.terms:
            if term is None:  continue

            term.canonize(target_orthog) # , cur_orthog=self.cur_orthog)

            if term.initialized:
                for i in range(term.L - 1, target_orthog, -1):
                    term.extend_env(i, direction=SweepDirection.LEFT)
                for i in range(0, target_orthog):
                    term.extend_env(i, direction=SweepDirection.RIGHT)
                # if target_orthog < self.cur_orthog:
                #     for i in range(self.cur_orthog, target_orthog, -1):
                #         term.extend_env(i, direction=SweepDirection.LEFT)
                # else:
                #     for i in range(self.cur_orthog, target_orthog):
                #         term.extend_env(i, direction=SweepDirection.RIGHT)

        if self.verbose:
            print('eval.cur_orthog', self.cur_orthog)


    def update_ket_from_out(self):
        raise NotImplementedError
        i = self.out.cur_orthog
        assert(self.out.cur_orthog == i), f'orthogonality centers do not match {i}, {self.out.cur_orthog}'

        # print('self.ket', self.ket)
        # print('self.out', self.out)

        # for x in range(self.L):
        #     print(x, helper_quimb.add_tensors(self.ket[x], self.out[x] * -1).norm())

        self.out[i].modify(data=self.out[i].data, inds=self.out[i].inds)

        # print('updated ket from out')
        # print('self.ket is vecblock ket', self.ket is self.terms[0].vec_block.ket)
        # print('self.ket is vecblock bra', self.ket is self.terms[0].vec_block.bra)
        # raise RuntimeError
        # exit()
        return


    def _site_solve(self, left_site_pos: int, nsites: int, site_tens: 'qtn.Tensor' = None,
                    return_intermediates=True) -> tuple[qtn.Tensor, Numeric]:
        """
        sites: int or slice(start, stop, step)
        """
        raise NotImplementedError

    def solve_l2r_adapt(self, canonize=True, verbose=False, filter_bases=False, **kwargs):

        # print('solve l2r adapt', self.max_bond)

        L = self.L

        if canonize:
            canon_site = 0
            self.canonize(canon_site)
            self.direction = SweepDirection.RIGHT

        ## right sweep
        if not np.isnan(self.cur_orthog):  # is nan means left_cur_orthog > right_cur_orthog
            assert(0 <= self.cur_orthog <= 0), f'cur_orthog should be 0 not {self.cur_orthog}'
        assert(self.direction == SweepDirection.RIGHT), f'should be SweepDirection.RIGHT, not {self.direction}'
        direction = self.direction

        phys_dims = [self.out.phys_dim(i) for i in range(self.out.L)]

        ## one right to left sweep
        tot_err = 0.0
        i = 0
        while i < L:

            if i == L-1:
                nsites = 1
            else:
                if self.max_bond is None:
                    max_rank = min(np.prod(phys_dims[:i + 1]).item(), np.prod(phys_dims[i + 1:]).item())
                else:
                    max_rank = min(self.max_bond, np.prod(phys_dims[:i + 1]).item(), np.prod(phys_dims[i + 1:]).item())
                current_rank = self.out.bond_size(i, i + 1)
                nsites = 2 if current_rank < max_rank else 1

                # print('l2r begin adapt, site solve', i, nsites, max_rank)
                # pdb.set_trace()

            site_i, site_err = self._site_solve(i, nsites)
            # print('site err', site_err)
            if site_err/self.out_norm > 10 ** 5:
                if self.verbose:
                    print('l2r site error too large', site_err)
                raise RuntimeError
            tot_err += site_err

            ## update ket (+ canonicalization)
            if nsites == 1:
                self._update_1site(i, site_i, direction, filter_bases=filter_bases,
                                   grid=self.grid, ax_deriv_configs=self.ax_deriv_configs)
                # self._update_1site_with_Z(i, site_i, direction)
            elif nsites == 2:
                self._update_2site(i, site_i, direction)
                # self._update_2site_with_Z(i, site_i, direction)
            else:
                raise NotImplementedError

            i += 1

            ## update A, b blocks (left)
            # if i < L - 1:
            #     self.update_blocks(i, direction)

        return self.out, tot_err


    def solve_r2l_adapt(self, canonize=True, verbose=False, filter_bases=False, **kwargs):

        # print('solve r2l adapt', self.max_bond)

        L = self.L

        if canonize:
            canon_site = L-1
            self.canonize(canon_site)
            self.direction = SweepDirection.LEFT

        ## right sweep
        # assert (self.cur_orthog == L-1), f'cur_orthog should be {L-1} not {self.cur_orthog}'
        if not np.isnan(self.cur_orthog):  # is nan means left_cur_orthog > right_cur_orthog
            assert (L - 1 <= self.cur_orthog <= L - 1), f'cur_orthog should be {L - 1} not {self.cur_orthog}'
        assert (self.direction == SweepDirection.LEFT), f'direction should be SweepDirection.LEFT, not {self.direction}'
        direction = self.direction

        phys_dims = [self.out.phys_dim(i) for i in range(self.out.L)]

        tot_err = 0.0
        i = L - 1
        while i >= 0:

            if i == 0:
                nsites = 1
            else:
                if self.max_bond is None:
                    max_rank = min(np.prod(phys_dims[:i]).item(), np.prod(phys_dims[i:]).item())
                else:
                    max_rank = min(self.max_bond, np.prod(phys_dims[:i]).item(), np.prod(phys_dims[i:]).item())
                current_rank = self.out.bond_size(i, i - 1)
                nsites = 2 if current_rank < max_rank else 1

                # print('r2l begin adapt, site solve', i, nsites, max_rank)
                # pdb.set_trace()

            left_site_pos = i - nsites + 1
            site_i, site_err = self._site_solve(left_site_pos, nsites)
            # print('site err', site_err)
            if site_err/self.out_norm > 10 ** 5:
                if self.verbose:
                    print('r2l site err too large', site_err)
                raise RuntimeError
            tot_err += site_err

            ## update ket, bra
            if nsites == 1:
                self._update_1site(i, site_i, direction, filter_bases=filter_bases,
                                   grid=self.grid, ax_deriv_configs=self.ax_deriv_configs)
                # self._update_1site_with_Z(i, nsites, direction)
            elif nsites == 2:
                self._update_2site(i, site_i, direction)
                # self._update_2site_with_Z(i, nsites, direction)
            else:
                raise NotImplementedError

            i -= 1

            ## update A, b envs (right)
            # if i > 0:
            #     self.update_blocks(i, direction)

        return self.out, tot_err


    def solve_l2r(self, nsites: int, canonize=True, verbose=False, filter_bases=False, **kwargs):

        if self.verbose:
            print("SOLVE L2R")

        if nsites > 2:
            return self.solve_l2r_adapt(canonize=canonize, verbose=verbose, filter_bases=filter_bases, **kwargs)

        # print('solve l2r', self.max_bond, 'nsites', nsites)


        L = self.L

        if canonize:
            canon_site = 0
            self.canonize(canon_site)
            self.direction = SweepDirection.RIGHT

        ## right sweep
        if not np.isnan(self.cur_orthog):  # is nan means left_cur_orthog > right_cur_orthog
            assert(0 <= self.cur_orthog <= nsites-1), f'cur_orthog should be 0 not {self.cur_orthog}'
        assert(self.direction == SweepDirection.RIGHT), f'should be SweepDirection.RIGHT, not {self.direction}'
        direction = self.direction

        # print('check orthog', self.check_orthog())
        # if self.cur_orthog == 0:
        #     print('check orthog', self.check_right_orthog()) #self.ket, self.ket_select_inds, [self.ket.site_ind_id]))
        # else:
        #     print('check orthog', self.check_left_orthog()) #self.ket, self.ket_select_inds, [self.ket.site_ind_id]))
        # print('check orthog ket', helper_quimb.check_orthog(self.ket))
        # print('check orthog out', helper_quimb.check_orthog(self.out))

        # print('x norms', [self.ket[i].norm() for i in range(self.ket.L)])

        ## one right to left sweep
        tot_err = 0.0
        for i in range(L - nsites + 1):
            # print('begin site solve', i)
            site_i, site_err = self._site_solve(i, nsites)
            # print('site err', site_err)
            # if site_err/self.out_norm > 10 ** 5:
            #     print('l2r site error too large', site_err)
            #     raise RuntimeError
            tot_err += site_err

            ## update ket (+ canonicalization)
            if nsites == 1:
                self._update_1site(i, site_i, direction, filter_bases=filter_bases,
                                   grid=self.grid, ax_deriv_configs=self.ax_deriv_configs)
                # self._update_1site_with_Z(i, site_i, direction)
            elif nsites == 2:
                self._update_2site(i, site_i, direction)
                # self._update_2site_with_Z(i, site_i, direction)
            else:
                raise NotImplementedError

            ## update A, b blocks (left)
            # if i < L - 1:
            #     self.update_blocks(i, direction)

        return self.out, tot_err


    def solve_r2l(self, nsites: int, canonize=True, verbose=False, filter_bases=False, **kwargs):

        if self.verbose:
            print('SOLVE R2L')

        if nsites > 2:
            return self.solve_r2l_adapt(canonize=canonize, verbose=verbose, filter_bases=filter_bases, **kwargs)

        # print('solve r2l', self.max_bond)

        L = self.L

        if canonize:
            canon_site = L-1
            self.canonize(canon_site)
            self.direction = SweepDirection.LEFT

        ## right sweep
        # assert (self.cur_orthog == L-1), f'cur_orthog should be {L-1} not {self.cur_orthog}'
        if not np.isnan(self.cur_orthog):  # is nan means left_cur_orthog > right_cur_orthog
            assert (L-nsites <= self.cur_orthog <= L - 1), f'cur_orthog should be {L - 1} not {self.cur_orthog}'
        assert (self.direction == SweepDirection.LEFT), f'direction should be SweepDirection.LEFT, not {self.direction}'
        direction = self.direction

        # print('check orthog ket', helper_quimb.check_orthog(self.ket))
        # print('check orthog out', helper_quimb.check_orthog(self.out))

        tot_err = 0.0
        for i in range(L - 1, nsites - 2, -1):
            left_site_pos = i - nsites + 1
            site_i, site_err = self._site_solve(left_site_pos, nsites)
            # print('site err', site_err)
            if site_err/self.out_norm > 10 ** 5:
                if self.verbose:
                    print('r2l site err too large', site_err)
                raise RuntimeError
            tot_err += site_err

            ## update ket, bra
            if nsites == 1:
                self._update_1site(i, site_i, direction, filter_bases=filter_bases,
                                   grid=self.grid, ax_deriv_configs=self.ax_deriv_configs)
                # self._update_1site_with_Z(i, nsites, direction)
            elif nsites == 2:
                self._update_2site(i, site_i, direction)
                # self._update_2site_with_Z(i, nsites, direction)
            else:
                raise NotImplementedError

            ## update A, b envs (right)
            # if i > 0:
            #     self.update_blocks(i, direction)

        return self.out, tot_err


    def solve(self, nsites: int, conv_tol=0, verbose=False, **kwargs):
        """ iterative solver for entire MPS
            minimize || Ax-b ||_2 = <Ax|Ax> + <b|b> - <Ax|b> - <b|Ax>
            sweep through sites i:
                d/dT*[i] () = d/dT*[i] <Ax|Ax> - <Ax|b> = 0
                    A_eff = d/dT*[i] d/dT[i] <Ax|Ax>
                    b_eff = d/dT*[i] <Ax|b>
                --> A_eff T[i] = b_eff

            A can be the identity (A is None) --> optimizing || x - b ||_2
            with proper canonicalization of x, A_left, A_right should be identity

            A: self.operators:
                can be composed of a series of MPOs (A1, ..., Am) order from ket to bra
            b: self.targets:
                can be written as a sum of MPSs (scaling included in the MPS; not kept track of separately)
            x: self.ket (updated in place)
        """
        verbose = True
        conv_tol = self.conv_tol if conv_tol == 0 else conv_tol

        L = self.out.L
        # canon_site = 0 if self.direction == SweepDirection.RIGHT else L - 1
        # helper_quimb.canonize(self.out, i=0)
        # self.canonize(canon_site)
        if self.verbose:
            print('solve', self.terms)
        for term in self.terms:
            if self.verbose:
                print('local_evaluator solve: term check orthog')
            if term is not None:
                term.check_orthog()
            # if self.term_class() is Term_DMRG:
            #     bra_canon = helper_quimb.check_orthog(term.vec_block.bra)
            #     ket_canon = helper_quimb.check_orthog(term.vec_block.ket)
            # elif self.term_class() is Term_Cross:
            #     import local_solvers.helper_cross as helper_cross
            #     bra_canon = helper_cross.check_orthog(term.vec_block.bra) #, term.vec_block.bra.select_inds, [term.vec_block.bra.site_ind_id])
            #     ket_canon = helper_cross.check_orthog(term.vec_block.ket) #, term.vec_block.ket.select_inds, [term.vec_block.ket.site_ind_id])
            # else:
            #     raise NotImplementedError
            # print('canon site', canon_site)
            # print('bra canon', bra_canon)
            # print('ket canon', ket_canon)

            # if not (bra_canon[0] >= canon_site and bra_canon[1] <= canon_site):
            #     # term.bra.canonize(canon_site)
            #     raise RuntimeError
            # if not (ket_canon[0] >= canon_site and ket_canon[1] <= canon_site):
            #     # term.ket.canonize(canon_site)
            #     raise RuntimeError

        # if self.cur_orthog == 0:
        #     print('check orthog', self.check_right_orthog()) #self.ket, self.ket_select_inds, [self.ket.site_ind_id]))
        # else:
        #     print('check orthog', self.check_left_orthog()) #self.ket, self.ket_select_inds, [self.ket.site_ind_id]))

        # print('x norms', [self.ket[i].norm() for i in range(self.ket.L)])
        # print('canonized')

        ## iterative solver
        it = 0
        err = self.err

        conv_it, prev_err = 0, err
        num_wrong_it = 0

        # self.max_tot_iter = 1  # 10

        # min_ket, min_err = self.ket, err
        if self.verbose:
            print('self copy before', self.terms)
        min_solver, min_err = self.copy(), err
        # direction = self.direction
        if self.verbose:
            print('conv tol', conv_tol, 'max iter', self.max_iter, self.max_wrong_iter)
        while it < 1 or (err > conv_tol and conv_it < self.max_iter
                and it < self.max_tot_iter and num_wrong_it < self.max_wrong_iter):

            it += 1
            # tot_err = 0.0

            ## right sweep
            if self.direction == SweepDirection.RIGHT:
                _, err = self.solve_l2r(nsites, canonize=False, verbose=verbose)

                # for i in range(L - nsites + 1):
                #     site_i, site_err = self._site_solve(i, nsites)
                #     if site_err > 10**5:
                #         print('site error too large')
                #         exit()
                #     tot_err += site_err
                #
                #     ## update ket (+ canonicalization)
                #     if nsites == 1:
                #         self._update_1site(i, site_i, direction)
                #         # self._update_1site_with_Z(i, site_i, direction)
                #     elif nsites == 2:
                #         self._update_2site(i, site_i, direction)
                #         # self._update_2site_with_Z(i, site_i, direction)
                #     else:
                #         raise NotImplementedError
                #
                #     ## update A, b blocks (left)
                #     if i < L - 1:
                #         self.update_blocks(i, direction)
                #
                # err = tot_err  # self.check_err(dense=True)

            else:

                ## left sweep
                _, err = self.solve_r2l(nsites, canonize=False, verbose=verbose)

                # ## left sweep
                # for i in range(L - 1, nsites - 2, -1):
                #     left_site_pos = i - nsites + 1
                #     site_i, site_err = self._site_solve(left_site_pos, nsites)
                #     if site_err > 10**5:
                #         print('site err too large')
                #         exit()
                #     tot_err += site_err
                #
                #     ## update ket, bra
                #     if nsites == 1:
                #         self._update_1site(i, site_i, direction)
                #         # self._update_1site_with_Z(i, nsites, direction)
                #     elif nsites == 2:
                #         self._update_2site(i, site_i, direction)
                #         # self._update_2site_with_Z(i, nsites, direction)
                #     else:
                #         raise NotImplementedError
                #
                #     ## update A, b envs (right)
                #     if i > 0:
                #         self.update_blocks(i, direction)
                #
                # err = tot_err  # self.check_err(dense=True)

            self.err = err
            self.is_conv = err < self.conv_tol

            if verbose:
                print('err', it, self.max_iter, err)
                # print('err np', self.check_err_np())

            # direction *= -1  ## swaps sweep direction
            self.direction = self.direction * -1

            if np.abs((prev_err - err) / err) < 1.0e-4:
                conv_it += 1

            prev_err = err

            if err < min_err or np.isnan(err):
                min_solver = self.copy()
                # min_ket = self.ket.copy()
                min_err = err if not np.isnan(err) else min_err
                num_wrong_it = 0
            else:
                if self.verbose:
                    print('Warning: solve error went up', err, min_err)
                num_wrong_it += 1
                if self.verbose:
                    print('num wrong', num_wrong_it, self.max_wrong_iter)

        # conv = min_err < conv_tol

        ## revert to optimal results
        if err > min_err:
            # pass
            self.out = min_solver.out
            self._new_ket_site = min_solver._new_ket_site
            # self.bra = min_solver.bra
            self._terms = min_solver.terms
            self.err = min_solver.err
            self.is_conv = min_solver.is_conv
            # self.cur_orthog = min_solver.cur_orthog

        if self.verbose:
            print('solver num iter', it)
        # exit()

        return self.out, self.err, self.is_conv

    def solve_1site(self, **solve_kwargs):
        return self.solve(1, **solve_kwargs)

    def solve_2site(self, **solve_kwargs):
        return self.solve(2, **solve_kwargs)

    def update_blocks(self, i: int, direction: SweepDirection): #, new_ket_site=None):
        for term in self.terms:
            if term is None:  continue
            # print('term extend env', term)
            term.extend_env(i, direction=direction) #, new_ket_site=new_ket_site)

    def _update_1site(self, i: int, site_i: Union['qtn.Tensor', Sequence['qtn.Tensor']],
                      direction: 'SweepDirection', filter_bases=False,
                      grid: 'Grid'=None, ax_deriv_configs: dict['Axis','DerivativeConfiguration']=None) -> None:
        raise NotImplementedError

    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection') -> None:
        raise NotImplementedError
