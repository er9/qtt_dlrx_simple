"""Mixed-projection local evaluator for the local-solver stack.

Implements :class:`MixedEvaluator` (a subclass of :class:`CrossEvaluator`)
and the :func:`local_mixed_evaluator` entry point, which sweep over an MPS and
update each site by combining mixed Term contributions that blend DMRG-style
(Galerkin) and cross-style (interpolatory) projections. Supports the
dynamical low-rank "X"/"G" variants used for the mixed solve.
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

from gridTN_1D import GridTN1D
import local_solvers.helper_tn as helper_tn

from local_solvers.local_evaluator import *
from local_solvers.local_cross_eval import CrossEvaluator
import local_solvers.helper_dmrg_loc as helper_dmrg
import local_solvers.helper_mixed as helper_mixed
from local_solvers.terms_mixed import Term_Mixed





def local_mixed_evaluator(terms: Sequence['Term_Mixed'],
                          init_guess: Optional['qtn.MatrixProductState'] = None,
                          max_bond=None, cutoff=None, combine_terms_func:Callable=None,
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

    print('nsites', nsites)
    solver = MixedEvaluator(init_guess, terms, combine_terms_func=combine_terms_func,
                            direction=direction, max_bond=max_bond, cutoff=cutoff)
    solver.solve(nsites)

    # out_gtn = GridTN1D(ref_x.grid, solver.ket)
    # return out_gtn
    return solver.solution


class MixedEvaluator(CrossEvaluator):

    version = flags.get('version', 'X')
    # 'version' ('X' default / 'G' Galerkin) is read at use sites via flags.get('version', 'X')

    @classmethod
    def term_class(cls) -> Type['Term_Mixed']:
        return Term_Mixed

    @property
    def terms(self) -> Sequence[Union['Term_DMRG', 'Term_Cross', 'Term_Mixed']]:
        return self._terms

    @terms.setter
    def terms(self, terms_list: list[Term]):
        canon_site = 0 if self.direction == SweepDirection.RIGHT else self.L - 1

        terms = []
        ref_term = None
        for term_ in terms_list:

            if term_ is not None:

                if ref_term is None:
                    ref_term = term_
                    if term_._bra is not None:
                        self.out = term_._bra  # .copy()
                        self.term_class().canonize_func(self.out, canon_site)
                else:
                    term_.match_inds(ref_term)

                # term_.bra = self.ket
                if term_._bra is None:
                    term_.bra = self.out  # self.ket.copy()
                self.term_class().canonize_func(term_.ket, canon_site)

                # term_.bra = self.out  # self.ket.copy()

                term_._direction = self.direction
                term_.max_bond = self.max_bond
                term_.initialize()
                # print('term_ setter cur orthog', self.direction, term_.cur_orthog)

            terms += [term_]        ## include placeholder Nones

        self._terms = terms

    @property
    def solver_type(self) -> LocalSolverType:
        return LocalSolverType.MIXED

    @property
    def solution(self) -> 'MPS':
        return self.out

    def _site_solve(self, left_site_pos: int, nsites: int, site_tens: qtn.Tensor=None, return_intermediates=True,
                    ) -> tuple[Sequence[qtn.Tensor], Numeric]:
        """
        sites: int or slice(start, stop, step)
        """
        print('in MIXED site solve')

        direction = self.direction
        at_end = (left_site_pos == self.L - nsites) if direction == SweepDirection.RIGHT else (left_site_pos == 0)

        site_inds = list(range(left_site_pos, left_site_pos + nsites))
        site_l = left_site_pos
        site_r = left_site_pos + nsites - 1
        inds_l = [self.out.bond(site_l, site_l - 1)] if site_l > 0 else []
        inds_r = [self.out.bond(site_r, site_r + 1)] if site_r < self.L - 1 else []

        current_x = qtn.tensor_contract(*[self.out[i] for i in site_inds]).copy()   ## basis

        ### BL * B * BR = d/dT*[i] <x|b>
        solve_func = self._local_solve_func
        if solve_func is None:
            term_site_tens = []
            target_rdm_list = []
            for term in self.terms:  # [1:]:
                num_ops = len(term.operators)
                if term is not None:

                    eval_site = term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)  ## element-wise

                    eval_site.modify(apply=lambda x: x * 10 ** term.bra.exponent)  ## include term.bra exponent
                    ## not sure i fully understand why for index selection... maybe with how envs are defined.

                    # #### check ####
                    # # tmp_ket = helper_quimb.apply(term.operators[0], term.ket)
                    # tmp_tmp = eval_site.reindex({ind: ind[:-2] for ind in eval_site.inds if ind[-1] == 'x'})
                    # tmp_ket = term.ket
                    # tmp_ket_data = helper_quimb.to_dense(tmp_ket)
                    # helper_cross.plot_submat(self.out, left_site_pos, nsites, tmp_tmp, select_inds=self.out.select_inds,
                    #                          ref_kets=[tmp_ket_data**3, self.out], plt_title='Ax')
                    #
                    # vec_tmp = term.vec_block.projected_site_x.copy()
                    # vec_tmp.modify(apply=lambda x: x * 10 ** term.bra.exponent)
                    # vec_proj = helper_mixed.convert_elementwise_to_basis(term.bra, vec_tmp, left_site_pos, nsites)
                    # vec_tmp = vec_tmp.reindex({ind: ind[:-2] for ind in vec_tmp.inds if ind[-1] == 'x'})
                    # ket_proj = term.bra.copy()
                    #
                    # # vec_proj = term.proj_vec_targets[0]
                    # # vec_proj = term.vec_block.projected_site.copy()
                    # vec_proj.modify(apply=lambda x: x * 10 ** -term.bra.exponent)
                    # print('vec proj', vec_proj)
                    # print('ket proj', ket_proj[left_site_pos])
                    # vec_proj.reindex({ind: ind[:-1] for ind in  vec_proj.inds if ind[-1] == '_'}, inplace=True)
                    # vec_proj.transpose_like(ket_proj[left_site_pos], inplace=True)
                    # ket_proj[left_site_pos].modify(data=vec_proj.data)
                    # helper_cross.plot_submat(self.out, left_site_pos, nsites, vec_tmp, select_inds=self.out.select_inds,
                    #                          ref_kets=[term.ket, ket_proj], plt_title='x')


                    # print('here?', term.vec_block.bra is self.out, num_ops)
                    # pdb.set_trace()

                    if term.vec_block.bra is self.out and num_ops > 0:
                        ## proj_vec_targets are elementwise measurements
                        # pdb.set_trace()
                        tmp = term.vec_block.projected_site_x.copy()
                        # tmp.modify(apply=lambda x: x * 10 ** term.bra.exponent)
                        target_rdm_list += [tmp]  # [t.copy() for t in term.proj_vec_targets_x]

                        # target_rdm_list += [helper_mixed.convert_basis_to_elementwise(term.vec_block.bra, t,
                        #                                                               left_site_pos, nsites)
                        #                     for t in term.proj_vec_targets]

                        # chk1 = term.vec_block.projected_site_x.copy()
                        # chk2 = helper_mixed.convert_basis_to_elementwise(term.vec_block.bra, term.proj_vec_targets[0],
                        #                                                  left_site_pos, nsites)
                        # print('proj site x')
                        # print(term.vec_block.projected_site_x)
                        # print('proj site g')
                        # print(term.vec_block.projected_site)
                        # print('vec targets ')
                        # print(term.proj_vec_targets[0])
                        # tmp3 = term.vec_block.projected_site.copy()
                        # tmp3 = tmp3.reindex({ind: ind[:-1] for ind in tmp3.inds if ind[-1] == '_'})
                        # chk3 = helper_mixed.convert_basis_to_elementwise(term.vec_block.bra, tmp3,
                        #                                                  left_site_pos, nsites)
                        # print('diff', (chk2 - chk1).norm())
                        # print('diff', (chk3 - chk1).norm())
                        # print(len(term.proj_vec_targets))
                        # pdb.set_trace()
                        # # print('target rdm list', target_rdm_list[-1])
                        #
                        # chk1g = chk1.reindex({ind: ind[:-2] for ind in chk1.inds if ind[-1] == 'x'})
                        # helper_cross.plot_submat(self.out, left_site_pos, nsites, chk1g,
                        #                          select_inds=self.out.select_inds,
                        #                          ref_kets=[term.ket], plt_title='x')
                        # ## an exact match of elements on term.ket
                        # ## here, just measuring the values of ket; ket is not projected onto bra.
                        # ## (that's select envs)
                        #
                        # chk3g = chk3.reindex({ind: ind[:-2] for ind in chk3.inds if ind[-1] == 'x'})
                        # helper_cross.plot_submat(self.out, left_site_pos, nsites, chk3g,
                        #                          select_inds=self.out.select_inds,
                        #                          ref_kets=[term.ket], plt_title='x')
                        # ## not an exact match.
                        # ## ket is projected onto bra (that's what projected site is) and then sampled


                        # if num_ops == 0:    ## proj_vec_targets are elementwise measurements
                        #     target_rdm_list += [t.copy() for t in term.proj_vec_targets]
                        # else:
                        #     ## tensors are in orthogonal basis. need to convert to element-wise rep
                        #     ## needed to target bra so that it fully captures term ket w/o modification
                        #     target_rdm_list += [helper_mixed.convert_basis_to_elementwise(term.vec_block.bra, t,
                        #                                                               left_site_pos, nsites)
                        #                         for t in term.proj_vec_targets]

                    term_site_tens += [eval_site]
                else:
                    term_site_tens += [None]

            site_tens = self.combine_terms_func(term_site_tens)
            # target_rdm_list = [site_tens]
            # helper_cross.plot_submat(self.out, left_site_pos, nsites, site_tens,
            #                          select_inds=self.out.select_inds, plt_title='out')

            site_tens.modify(apply=lambda x: x * 10 ** (-self.out.exponent))  ## remove self.ket exponent
            out_tensors = [site_tens]
            target_rdm_list = out_tensors + target_rdm_list
            print('target rdm list', target_rdm_list)


        else:
            ### CHANGED HERE
            if return_intermediates:
                out_site, target_rdm_list = solve_func(left_site_pos, nsites, return_intermediates=return_intermediates, site_tens=site_tens)
            else:
                out_site = solve_func(left_site_pos, nsites, return_intermediates=return_intermediates,site_tens=site_tens)
            out_tensors = [out_site]
            ## first tensor is tensor of interest. other tensors are for targeting

            # term_site_tens = [current_x]
            # print('getting evaluated site of terms (2)')
            # for term in self.terms[1:]:
            #     term_site_tens += [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens)]
            #     term_site_tens[-1].modify(apply=lambda x: x * 10 ** term.bra.exponent)  ## include term.bra exponent


        ## solve for T[i]
        x_eff = out_tensors[0].copy()

        # ## (env_self[pos])^-1 x_eff (env_self[pos+1]); because not canonical
        # ## but it should be canonical?
        # tensors = [x_eff]
        # # if site_l > 0:
        # #     inv_env_l = self.ket_block.envs[site_l-1]
        # #     if inv_env_l is not None:
        # #         ## indeed, these terms aren't needed (except for ket to bra reindex)
        # #         print('skipping inv env L')
        # #         # inv_env_l = inv_env_l.copy()
        # #         # inv_env_l.modify(apply=lambda x: np.linalg.inv(x).T)
        # #         tensors += [inv_env_l]
        # # if site_r < self.L-1:
        # #     inv_env_r = self.ket_block.envs[site_r+1]
        # #     # print('inv_env_r', inv_env_r)
        # #     if inv_env_r is not None:
        # #         ## indeed, these lines aren't needed (except for ket to bra reindex)
        # #         print('skipping inv env R')
        # #         # inv_env_r = inv_env_r.copy()
        # #         # inv_env_r.modify(apply=lambda x: np.linalg.inv(x).T)
        # #         tensors += [inv_env_r]
        #
        # x_eff = qtn.tensor_contract(*tensors)  ## changes bra to ket inds
        if self.version == 'G':
            x_eff.transpose(*(inds_l + [self.out.site_ind_id.format(i) for i in site_inds] + inds_r), inplace=True)
        else:
            xinds_l, xinds_r = [il + '_x' for il in inds_l], [ir + '_x' for ir in inds_r]
            x_eff.transpose(*(xinds_l + [self.out.site_ind_id.format(i) for i in site_inds] + xinds_r), inplace=True)

        self._new_ket_site = (left_site_pos, nsites, self.direction, [x_eff])
        # self._current_ket_sites[left_site_pos] = x_eff

        ## comparing element-wise
        if self.version == 'G':
            current_x_x = current_x
        else:
            current_x_x = helper_mixed.convert_basis_to_elementwise(self.out, current_x, left_site_pos, nsites)
            current_x_x.transpose_like(x_eff, inplace=True)
        try:
            site_err = np.linalg.norm(x_eff.data - current_x_x.data) / np.linalg.norm(current_x_x.data)
        except ValueError:  ## shape mismatch
            site_err = np.nan
        print('site err', site_err)

        # ### plot out
        # coords = helper_cross.get_selectors(self.out, left_site_pos, nsites)
        # selectors = []
        # for c in coords:
        #     selectors += [int("".join(str(x) for x in c), 2)]
        #
        # inds = []
        # i = left_site_pos
        # if i > 0:
        #     inds += [self.out.bond(i, i - 1)]
        # inds += [self.out.site_ind(i + ix) for ix in range(nsites)]
        # if i + nsites < self.out.L:
        #     inds += [self.out.bond(i + nsites - 1, i + nsites)]
        #
        # for tens in term_site_tens + [x_eff]:
        #     tens.transpose(*inds, inplace=True)
        #     plt.plot(selectors, tens.data.reshape(-1), 'x')
        # plt.title('cross eval')
        # plt.show()

        if return_intermediates and not at_end:
            return target_rdm_list, site_err
        else:
            return out_tensors, site_err


    def _update_1site(self, i: int, site_i: 'qtn.Tensor', direction: 'SweepDirection', filter_bases=False,
                      grid=None, ax_deriv_configs: dict['Axis','DerivativeConfiguration']=None):
        """ update ket, bra with new_site
            i: int of mps site
        """

        ## need to project site_i tensors onto the orthogonal manifold

        at_end = (i == 0) if direction == SweepDirection.LEFT else (i == self.L - 1)


        ket, tensors = self.out, [site_i] if not isinstance(site_i, (list, tuple)) else site_i

        for term in self.terms:
            if term is not None:
                term.update_intermediate_kets(i, 1, direction)

        # # kets, tensors = [self.ket, self.out], [self.terms[0].proj_vec_targets, site_i]
        # for ix, term in enumerate(self.terms):
        #     if term is None:  continue
        #
        #     # if ix > 3:
        #     #     tensors += [term._proj_vec_targets]
        #
        #     if term.num_tiers > 1:
        #         kets += [term.get_intermediate_ket(i) for i in range(term.num_tiers - 1)]
        #         tensors += [term.intermediate_sites[i] for i in range(term.num_tiers - 1)]

        helper_mixed.update_ket(ket, tensors, i, 1, direction=direction, max_bond=self.max_bond, cutoff=self.cutoff)


        if not at_end:
            self.update_blocks(i, direction)

        return


    def _update_2site(self, i: int, site_i: Sequence['qtn.Tensor'], direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
        """
        print('UPDATE2', i)

        ket, tensors = self.out, [site_i] if not isinstance(site_i, (list, tuple)) else site_i
        # for term in self.terms:
        #     if term is None:  continue
        #     tensors += [term._proj_vec_targets]

        # print('tensors', tensors)
        helper_mixed.update_ket(ket, tensors, i, 2, direction=direction, max_bond=self.max_bond, cutoff=self.cutoff)

        for term in self.terms:
            if term is not None:
                term.update_intermediate_kets(i, 2, direction)

        self.update_blocks(i, direction)

        return

