"""DMRG local evaluator for the local-solver stack.

Implements :class:`DMRGEvaluator` and the :func:`local_dmrg_evaluator` entry
point, which sweep over an MPS and update each site by summing the DMRG-style
(full Galerkin) Term contributions via state-averaged reduced density
matrices. This is the Evaluator-layer driver for variational/Galerkin local
solves.
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
# from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
# from local_solvers_old.environments_v2 import Environment, EnvironmentSide
# from local_solvers_old.environments_v2 import TargetEnv_DMRG, TargetEnv_Cross
# from local_solvers_old.environments_v2 import OperatorEnv_DMRG, OperatorEnv_Cross
# from local_solvers_old.local_solver import SweepDirection
# from local_solvers_old.blocks import EnvironmentType, init_block, Block, BlockOperator, BlockVector, BlockVector_Cross, BlockOperator_Cross

# from gridTN_1D import GridTN1D
# import local_solvers.helper_tn as helper_tn

# from local_solvers_old.local_evaluator import get_mps_matching_inds
# from local_solvers_old.helper_cross_v3 import check_left_orthog, check_right_orthog, check_orthog
# from local_solvers_old.helper_cross_v2 import maxvol_inds, canonize, cross_compress

# from local_solvers_old.local_solver_v2 import *  # LocalSolver
# from local_solvers_old.helper_cross_v3 import CrossSolver, canonize, canonize_tens_list, cross_compress
# from local_solvers_old.helper_cross_v3 import compress_tens_list
import local_solvers.helper_dmrg_loc as helper_dmrg
from local_solvers.blocks import BlockPowerKet_DMRG
# from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross
from local_solvers.local_evaluator import *



def local_dmrg_evaluator(terms: Sequence['Term_DMRG'],
                          init_guess: Optional['qtn.MatrixProductState'] = None,
                          max_bond=None, cutoff=CUTOFF, combine_terms_func:Callable=None,
                          nsites=1, direction: SweepDirection = SweepDirection.RIGHT,
                          ):
    """ terms are sequences of ((A, B, ...), x)
        where A, B are operators or functions that are applied to x in
        the order A(B(x))

        algorithm: compute rdm for each term
        sum together terms by summing rdms together (state-averaging method)
        projecting next site onto new rdm
    """
    init_guess = terms[0].ket.copy() if init_guess is None else init_guess

    solver = DMRGEvaluator(init_guess, terms, combine_terms_func=combine_terms_func,
                            direction=direction, max_bond=max_bond, cutoff=cutoff)
    solver.solve(nsites)

    # plt.figure()
    # plt.plot(helper_quimb.to_dense(solver.terms[0].ket).reshape(-1),
    #          label='ket')
    # plt.plot(helper_quimb.to_dense(solver.out).reshape(-1),
    #          label='out')
    # plt.show()

    # out_gtn = GridTN1D(ref_x.grid, solver.ket)
    # return out_gtn
    return solver.out


class DMRGEvaluator(LocalEvaluator):

    @classmethod
    def term_class(cls) -> Type['Term_DMRG']:
        return Term_DMRG

    @property
    def solver_type(self) -> LocalSolverType:
        return LocalSolverType.DMRG

    @property
    def solution(self) -> 'MPS':
        # out = self.ket.copy()
        # left_site, nsites, direction, site_i = self._new_ket_site
        #
        # at_end = left_site + nsites == self.L if direction == SweepDirection.RIGHT else (left_site == 0)
        # assert at_end, 'solution should be obtained at the end of the sweep'
        # print('at end', at_end, left_site, nsites, direction)
        # print('self.cur_orthog', self.cur_orthog, self.direction)
        #
        # if nsites == 1:
        #     helper_dmrg.update_1site(out, left_site, site_i, direction, max_bond=self.max_bond)
        # # elif nsites == 2:
        # #     helper_dmrg.update_2site(self.ket, left_site, site_i, direction, max_bond=self.max_bond)
        # else:
        #     raise NotImplementedError
        #
        # return out

        return self.out

    def check_orthog(self):
        for term in self.terms:
            if term is not None:
                term.check_orthog()
            # bra_canon = helper_quimb.check_orthog(term.vec_block.bra)
            # ket_canon = helper_quimb.check_orthog(term.vec_block.ket)
        return

    # def _site_solve(self, left_site_pos: int, nsites: int) -> tuple[Sequence[qtn.Tensor], Numeric]:
    #     """
    #     sites: int or slice(start, stop, step)
    #     """
    #     site_inds = list(range(left_site_pos, left_site_pos + nsites))
    #
    #     direction = self.direction
    #     at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)
    #
    #     # ref_x_data = self.vecs[0]
    #
    #     if nsites == 1:
    #         ix = left_site_pos
    #         ix2 = ix + 1 if direction == SweepDirection.RIGHT else ix - 1
    #     elif nsites == 2:
    #         if direction == SweepDirection.RIGHT:
    #             ix, ix2 = left_site_pos, left_site_pos + 1
    #         else:
    #             ix2, ix = left_site_pos, left_site_pos + 1
    #     else:
    #         raise ValueError
    #
    #     current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
    #     # current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)
    #
    #     # tot_rdm = None
    #     # tot_site = None
    #     site_tens_list = []
    #     site_rdm_list = []
    #     for term in self.terms:
    #
    #         site_tens = term.get_evaluated_site(left_site_pos, nsites)
    #
    #         # if isinstance(site_tens, list):
    #         #     site_tens = [t.modify(apply=lambda x: x * 10 ** -self.ket.exponent) for t in site_tens]
    #         #     site_tens_list += [site_tens]
    #         # else:
    #         #     site_tens.modify(apply=lambda x: x * 10 ** -self.ket.exponent)
    #         #     site_tens_list += [site_tens]
    #
    #         # site_tens.modify(apply=lambda x: x * 10 ** -self.ket.exponent)
    #         ### moved to get_evaluated_site
    #         site_tens_list += [site_tens]
    #
    #         vec_block = term.vec_block
    #         if not at_end and isinstance(vec_block, BlockPowerKet_DMRG) and self.ket is vec_block.bra:
    #             # vec_tens = term.proj_vec
    #             # site_rdm_list += [vec_tens]
    #             # pow_list = term.proj_vec
    #             pow_list = [t.copy() for t in term.proj_vec] #vec_block.get_pows(left_site_pos, nsites)]
    #             # for t in pow_list:
    #             #     t.modify(apply=lambda x: x / t.norm())
    #             site_rdm_list += pow_list
    #
    #     # tot_site = helper_tn.sum_tens(site_tens_list)
    #     tot_site = self.combine_terms_func(site_tens_list)
    #
    #     current_x.transpose_like(tot_site, inplace=True)
    #     try:
    #         # print('tot site', tot_site)
    #         # print('current x', current_x)
    #         # print('self,max bond', self.max_bond)
    #         site_err = np.linalg.norm(tot_site.data - current_x.data) / np.linalg.norm(current_x.data)
    #     except ValueError:  ## shape mismatch
    #         site_err = np.nan
    #
    #     if at_end:
    #         return [tot_site], site_err
    #
    #     else:
    #         return site_tens_list + site_rdm_list, site_err


    def _bond_solve(self, left_site_pos: int, site_tens: qtn.Tensor=None, return_intermediates=True,
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        update bond between left_site_pos, left_site_pos + 1
        """
        # print('BOND SOLVE', left_site_pos)
        site_inds = list(range(left_site_pos, left_site_pos + 1))

        direction = self.direction
        at_end = (left_site_pos == self.L - 1) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

        ix = left_site_pos
        ix2 = ix + 1  # if direction == SweepDirection.RIGHT else ix - 1

        if site_tens is None:
            raise NotImplementedError
            # current_x = qtn.tensor_contract(*[self.ket[i] for i in site_inds])
        else:
            current_x = site_tens.copy()
        # current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)

        # tot_rdm = None
        # tot_site = None
        site_tens_list = []
        site_rdm_list = []

        func = self._local_solve_func
        if func is not None:
            if not at_end and return_intermediates:
                tot_site, site_rdm_list = func(left_site_pos, 0, return_intermediates=True, site_tens=site_tens)
            else:
                tot_site = func(left_site_pos, 0, return_intermediates=False, site_tens=site_tens)

        else:
            for term in self.terms:
                site_tens = term.get_evaluated_site(left_site_pos, 0)
                site_tens_list += [site_tens]

                vec_block = term.vec_block
                if not at_end and isinstance(vec_block, BlockPowerKet_DMRG) and self.out is vec_block.bra:
                    pow_list = [t.copy() for t in term.proj_vec_targets] #vec_block.get_pows(left_site_pos, nsites)]
                    site_rdm_list += pow_list

            # tot_site = helper_tn.sum_tens(site_tens_list)
            tot_site = self.combine_terms_func(site_tens_list)

        current_x.transpose_like(tot_site, inplace=True)
        try:
            site_err = np.linalg.norm(tot_site.data - current_x.data) / np.linalg.norm(current_x.data)
        except ValueError:  ## shape mismatch
            site_err = np.nan

        # if at_end:
        #     return [tot_site], site_err
        # else:
        #     return site_tens_list + [*site_rdm_list], site_err

        if return_intermediates and not at_end:
            return site_tens_list + [*site_rdm_list], site_err
        else:
            return [tot_site], site_err


    def _site_solve(self, left_site_pos: int, nsites: int, site_tens: qtn.Tensor=None, return_intermediates=True,
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        """
        site_inds = list(range(left_site_pos, left_site_pos + nsites))

        # print('SITE SOLVE CHECK 1')
        # i = left_site_pos
        # print('ket vs out', helper_quimb.add_tensors(self.out[i], self.ket[i] * -1).norm())
        # print('ket vs out', helper_quimb.add_tensors(self.out[i + 1], self.ket[i + 1] * -1).norm())

        direction = self.direction
        at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)
        # print('DMRG SITE SOLVE', left_site_pos, at_end)

        if nsites == 1:
            ix = left_site_pos
            ix2 = ix + 1 if direction == SweepDirection.RIGHT else ix - 1
        elif nsites == 2:
            if direction == SweepDirection.RIGHT:
                ix, ix2 = left_site_pos, left_site_pos + 1
            else:
                ix2, ix = left_site_pos, left_site_pos + 1
        else:
            raise ValueError

        current_x = qtn.tensor_contract(*[self.out[i] for i in site_inds])
        # if site_tens is None:
        #     current_x = qtn.tensor_contract(*[self.out[i] for i in site_inds])
        # else:
        #     current_x = site_tens.copy()

        # current_x.modify(apply=lambda x: x * 10 ** self.ket.exponent)
        ### don't include ket exponent bc ket_exponent is removed from site_tens later
        ### current_x is used for error comparison

        # tot_rdm = None
        # tot_site = None
        eval_tens_list = []
        target_rdm_list = []

        func = self._local_solve_func
        if func is not None:
            if not at_end and return_intermediates:
                tot_site, target_rdm_list = func(left_site_pos, nsites, return_intermediates=True,
                                                 site_tens=site_tens)
                # print('here?')
            else:
                tot_site = func(left_site_pos, nsites, return_intermediates=False, site_tens=site_tens)
                # print('here at end')
            eval_tens_list = [tot_site]

        else:
            target_rdm_list = []
            for term in self.terms:  # [1:]:
                num_ops = len(term.operators)
                if term is not None:

                    eval_site = term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)  ## element-wise

                    eval_site.modify(apply=lambda x: x * 10 ** term.bra.exponent)  ## include term.bra exponent
                    ## not sure i fully understand why for index selection... maybe with how envs are defined.

                    if term.vec_block.bra is self.out:
                        target_list = [t.copy() for t in term.proj_vec_targets]
                        ## to normalize Ax, x so that they have similar weights, but not normalize across different terms
                        for t in target_list:
                            t.modify(apply=lambda x: x * 1.0 / t.norm())
                        # target_rdm_list += [site_tens]
                        target_rdm_list += target_list

                    eval_tens_list += [eval_site]
                else:
                    eval_tens_list += [None]

            # tot_site = helper_tn.sum_tens(site_tens_list)
            tot_site = self.combine_terms_func(eval_tens_list)
            tot_site.modify(apply=lambda x: x * 10 ** (-self.out.exponent))  ## exclude ket exponent
            target_rdm_list = [tot_site.copy() / tot_site.norm(), *target_rdm_list]

        current_x.transpose_like(tot_site, inplace=True)
        try:
            site_err = np.linalg.norm(tot_site.data - current_x.data) / np.linalg.norm(current_x.data)
        except ValueError:  ## shape mismatch
            site_err = np.nan
            # print('current x', current_x)
            # print('tot site', tot_site)
            # print('self.ket', self.ket)
            # print('self.', self.terms[0].bra is self.ket)
            # print('self.bra', self.terms[0].bra)

        self._new_ket_site = (left_site_pos, nsites, direction, eval_tens_list)
        # self._current_ket_sites[left_site_pos] = tot_site

        # print('site err', site_err)
        #
        # print('SITE SOLVE CHECKE 2')
        # i = left_site_pos
        # print('ket vs out', helper_quimb.add_tensors(self.out[i], self.ket[i] * -1).norm())
        # print('ket vs out', helper_quimb.add_tensors(self.out[i + 1], self.ket[i + 1] * -1).norm())

        # if at_end:
        #     return [tot_site], site_err
        #
        # else:
        #     return target_rdm_list, site_err
        #     # return eval_tens_list + target_rdm_list, site_err

        if return_intermediates and not at_end:
            return target_rdm_list, site_err
        else:
            return [tot_site], site_err


    def _update_1site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection', filter_bases=False,
                      grid: 'Grid'=None, ax_deriv_configs: dict['Axis','DerivativeConfiguration']=None):
        """ update ket, bra with new_site
            i: int of mps site
        """
        # print('ket vs out', helper_quimb.add_tensors(self.out[i], self.ket[i] * -1).norm())
        # print('ket vs out', helper_quimb.add_tensors(self.out[i + 1], self.ket[i + 1] * -1).norm())

        at_end = (i == 0 if direction == SweepDirection.LEFT else i == self.L - 1)
        # print('at end?', at_end)

        # print('UPDATE OUT 1 site', i)
        # print('site_i', [s.norm() for s in site_i])
        # print('site i', site_i)
        # print(self.out)
        helper_dmrg.update_1site(self.out, i, site_i, direction, max_bond=self.max_bond, cutoff=self.cutoff,
                                 filter_bases=filter_bases, grid=grid, ax_deriv_configs=ax_deriv_configs)

        # plt.figure()
        # plt.plot(helper_quimb.to_dense(self.terms[0].ket).reshape(-1),
        #          label='ket')
        # plt.plot(helper_quimb.to_dense(self.out).reshape(-1),
        #          label='out')
        # plt.title(f'i {i}/{self.out.L}')
        # plt.legend()
        # plt.show()

        # print('update vecblock')
        for term in self.terms:
            if term is not None:
                term.update_intermediate_kets(i, 1, direction)

        #     tmp = term.bra
        #     plt.plot(helper_quimb.to_dense(tmp, inds).data.reshape(-1), label='tmp')
        #
        # plt.legend()
        # plt.title(f'update 1 {i}')
        # plt.show()

        # if at_end:
        #     exit()

        # print('update blocks')
        if not at_end:
            self.update_blocks(i, direction=direction)

        return


    def _update_2site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        at_end = (i == 1 if direction == SweepDirection.LEFT else i == self.L - 2)
        left_site_ind = i if direction == SweepDirection.RIGHT else i - 1

        if not at_end:
            # raise NotImplementedError
            helper_dmrg.update_2site(self.out, left_site_ind, site_i, direction,
                                     max_bond=self.max_bond, cutoff=self.cutoff)

            # print('update vecblock')
            for term in self.terms:
                if term is not None:
                    term.update_intermediate_kets(i, 2, direction)

            # print('update blocks')
            self.update_blocks(i, direction=direction)


        else:
            helper_dmrg.update_2site(self.out, left_site_ind, site_i, direction,
                                     max_bond=self.max_bond, cutoff=self.cutoff)

            # x_ind = self.ket.bond(i, i + direction)
            # left_inds = [ind for ind in self.ket[i].inds if ind != x_ind]
            # q, r = qtn.tensor_split(site_i, left_inds, absorb='right', max_bond=self.max_bond, bond_ind=x_ind)
            #
            # q.transpose_like(self.out[i], inplace=True)
            # self.out[i].modify(data=q.data)
            #
            # r.transpose_like(self.out[i + direction], inplace=True)
            # self.out[i + direction].modify(data=r.data)

            # print('update vecblock')
            for term in self.terms:
                if term is not None:
                    term.update_intermediate_kets(i, 2, direction)

        return

