import pdb

import numpy as np
import itertools
import quimb
import scipy.linalg

import helper_quimb
# from helper_block_tddmrg_3 import BlockTDDMRGSolver
from setup_.defaults import *
from local_solvers.defaults import *

import quimb.tensor as qtn
# from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
# import helper_quimb as helper
# import helper_dmrg
# import helper_dmrg_2
# from local_solvers.mps_classes import MPS
import local_solvers.helper_dmrg_loc as helper_dmrg

# import helper_TE
import local_solvers.helper_tn as helper_tn
# from helper_dmrg import *
# from helper_tdvp_v2 import *
# from local_solvers.time_integrator import TDVP_DMRG, TDDMRG, TimeIntegrator
# from helper_block_dmrg import combine_vec_tens, combine_mat_tens, extract_vec_tens

import local_solvers.helper_cross_2 as helper_cross
# from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross
# from helper_dmrg import (qtn_conjugate_gradient_squared_1site,
#                          qtn_conjugate_gradient_descent_1site)
from helper_block_tddmrg_3 import BlockTimeIntegrator, BlockTDDMRGSolver, BlockTDDMRGXSolver

""" perform block time evolution 
    intended application is for Vlasov-Maxwell
    so we assume that the grid is ordered by scale (not necessarily all of the same length)
    in the future could factorize, but unclear if there is any advantage there.
    need to deal with updating the current as well
    
    I guess it doesn't have to be ordered by grid scale 
    If of uneven lengths then hold the remainder fixed while updating the others.
    (E.g., updating one dimension while holding the others fixed)
    Need to keep track of 
    
    Should define a sweep ordering/schedule (for hierarchical time evolution)
    with a corresponding time step size
"""

def block_tdvp(dt: float, ncomps: int, A_mats: dict[tuple[int, int], Sequence['MPO_type']],
               x_vecs: dict[int, 'qtn.MatrixProductState'],
               b_vecs: dict[int, Sequence['MPS_type']] = None,
               masks: dict[int, Sequence['MPO_type']] = None,
               verbose_output=False,
               init_direction=SweepDirection.RIGHT,
               constraints: dict[int, 'qtn.MatrixProductOperator'] = None,
               constraint_vals: dict[int, 'qtn.MatrixProductState'] = None,
               shared_projs: list[set] = None,
               backward_weights: dict[tuple[int, int], float]=None,
               do_adapt=True,
               te_order=4, grid=None,  **solver_kwargs):
    print('BLOCK TDVP 3')

    # x_vecs = {k: x for k, x in x_vecs.items() if helper_quimb.norm(x) > np.sqrt(CUTOFF)}
    x_vecs_ = {}
    x_ref = x_vecs[next(iter(x_vecs))]
    for k, v in x_vecs.items():
        helper_quimb.match_inner_inds(v, x_ref, inplace=True)
        if v.exponent < -10:
            if helper_quimb.norm(v) < np.sqrt(CUTOFF):
                continue
        x_vecs_[k] = v
    x_vecs = x_vecs_

    for k, x in x_vecs.items():
        x.distribute_exponent()
    ## taking out exponent requires some care? modifying Ax * 10**-exponent didn't work?
    ## causes issues if exponent is not distributed

    # for k, As in A_mats.items():
    #     for A in As:
    #         A.distribute_exponent()


    ## targeting so sources, masks all exist on the same manifold
    if b_vecs is not None:

        ## remove any inds that were none
        rm_inds = []
        for k, xs in b_vecs.items():
            if all([x is None for x in xs]):
                rm_inds += [k]

        for k in rm_inds:
            b_vecs.pop(k)

        ## add kets if sources are not None
        for k, xs in b_vecs.items():

            if len(xs) > 0:
                for x in xs:  ## match inner inds
                    if x is not None:
                        helper_quimb.match_inner_inds(x, x_ref, inplace=True)

                if not k in x_vecs:
                    print('adding to x vec from b', k)
                    new_xk, it = None, 0
                    while new_xk is None and it < len(xs):
                        new_xk = xs[it]
                        it += 1
                    if new_xk is not None:
                        new_xk = xs[0].copy()
                        for tens in new_xk.tensors:
                            tens.modify(apply=lambda x: x * 0)
                        x_vecs[k] = new_xk

                ## basis expansion is done in the TDDMRG init

    if masks is not None:
        for k, xs in masks.items():
            if k in x_vecs and len(xs) > 0:
                for x in xs:    ## match inner inds
                    helper_quimb.match_inner_inds(x, x_ref, inplace=True)
                # print('x vec', helper_quimb.max_inner_bond(x_vecs[k]))
                # helper_quimb.add_MPS_list([x_vecs[k], *xs], do_final_update=False, inplace=True,
                #                           compress_opts={'max_bond': solver_kwargs.get('max_bond', None),
                #                                          'cutoff': solver_kwargs.get('cutoff', CUTOFF)}, )
                # print('expanded x vec', helper_quimb.max_inner_bond(x_vecs[k]))


    solver = BlockTDVPSolver(ncomps, x_vecs, sources=b_vecs, operators=A_mats, masks=masks,
                               te_order=te_order, direction=init_direction,
                               constraints=constraints, constraint_vals=constraint_vals,
                               grid=grid, backward_weights=backward_weights,
                               **solver_kwargs)

    solver.take_time_step(dt, init_direction=init_direction, do_adapt=do_adapt)

    if verbose_output:
        return solver.kets, solver.err, solver.is_conv
    else:
        return solver.kets


def block_tdvpx(dt: float, ncomps: int, A_mats: dict[tuple[int, int], Sequence['MPO_type']],
                x_vecs: dict[int, 'qtn.MatrixProductState'],
                b_vecs: dict[int, Sequence['MPS_type']] = None,
                masks: dict[int, Sequence['MPO_type']] = None,
                verbose_output=False,
                init_direction=SweepDirection.RIGHT,
                constraints: dict[int, 'qtn.MatrixProductOperator'] = None,
                constraint_vals: dict[int, 'qtn.MatrixProductState'] = None,
                shared_projs: list[set] = None,
                backward_weights: dict[tuple[int, int], float]=None,
                do_adapt=True,
                te_order=4, grid=None, **solver_kwargs):
    print('BLOCK TDVP-X 3')

    # x_vecs_ = {}
    # for k, v in x_vecs.items():
    #     if v.exponent < -10:
    #         if helper_quimb.norm(v) < np.sqrt(CUTOFF):
    #             continue
    #     x_vecs_[k] = v
    # x_vecs = x_vecs_

    for k, x in x_vecs.items():
        x.distribute_exponent()

    for k, xs in A_mats.items():
        for x in xs:
            x.distribute_exponent()

    if b_vecs is not None:
        for k, xs in b_vecs.items():
            for x in xs:
                if k in x_vecs:
                    helper_quimb.match_inner_inds(x, x_vecs[k], inplace=True)

    if masks is not None:
        for k, xs in masks.items():
            for x in xs:
                if k in x_vecs:
                    helper_quimb.match_inner_inds(x, x_vecs[k], inplace=True)

    solver = BlockTDVPXSolver(ncomps, x_vecs, sources=b_vecs, operators=A_mats, masks=masks,
                                te_order=te_order, direction=init_direction,
                                constraints=constraints, constraint_vals=constraint_vals,
                                grid=grid, backward_weights=backward_weights,
                                **solver_kwargs)

    solver.take_time_step(dt, init_direction=init_direction, do_adapt=do_adapt)

    if verbose_output:
        return solver.kets, solver.err, solver.is_conv
    else:
        return solver.kets



class BlockTDVPIntegrator(BlockTimeIntegrator):
    def ket_inds(self, left_site_pos, nsites, comp=None):
        if nsites > 0:
            return super().ket_inds(left_site_pos, nsites, comp=comp)

        ref_ket = self.ket if comp is None else self.kets[comp]
        bond_ind = ref_ket.bond(left_site_pos, left_site_pos + 1)
        return [bond_ind + '_L', bond_ind + '_R']

    def _bond_time_evolution(self, dt, left_site_pos, bond_ind: str, # sweep_direction=SweepDirection.RIGHT,
                             site_tens_dict: dict[int, 'qtn.Tensor']=None):
        """
        sites: between left_site_pos, left_site_pos + 1
        solve local A' x = b',  (1/2?) <x|Ax> - <x|b> = 0
            where A' = d/dT[i] d/dT[i]^* <x|A|x> = d/dT[i] d/dT[i]^* \sum_ij <x_j|A_ji|x_i>
            where b' = d/dT[i]^* <x|b> = d/dT[i] \sum_ij <x_j|b_i>
        """
        te_order_target = 22 if (self.te_order == 0) else self.te_order  # 4  # 22  # 22
        te_order_final = self.te_order  # 4  # 22  # 22

        nsites = 0
        # if sweep_direction == SweepDirection.RIGHT:
        #     ind1 = left_site_pos
        # else:
        #     ind1 = left_site_pos + 1

        bonds_i = [bond_ind + '_L', bond_ind + '_R']
        bonds_o = [bond_ind + '_L_', bond_ind + '_R_']

        new_xeff_dict = self.get_time_evolved_site(dt, left_site_pos, nsites,
                                                   bonds_i, bonds_o,
                                                   te_order=te_order_target,
                                                   site_tens_dict=site_tens_dict,
                                                   return_intermediates=False,
                                                   )
        return new_xeff_dict

    def _site_time_evolution(self, dt, left_site_pos, nsites=1,
                             sweep_direction=SweepDirection.RIGHT, site_tens_dict: dict[int, 'qtn.Tensor'] = None,
                             return_intermediates=False):
        # print('TDVP SITE EVOLUTION', left_site_pos, nsites)

        # return_intermediates = True if left_site_pos < 3 else False
        out = super()._site_time_evolution(dt, left_site_pos, nsites=nsites, sweep_direction=sweep_direction,
                                            site_tens_dict=site_tens_dict, return_intermediates=return_intermediates)

        expand_ket = False # True
        if expand_ket:
            print('Warning: not sure if this is implemented correctly')
            raise NotImplementedError
            eff_Ax_dict = self._get_Ax_eff_dict_(left_site_pos, nsites, site_tens_dict=out,
                                                 negative_dt=(dt < 0))
            return out, eff_Ax_dict
        else:
            return out


    def _update_0site(self, i: int, sweep_direction: SweepDirection,
                      site_i_dict: dict[int, Union[Sequence['qtn.Tensor'],'qtn.Tensor']],):
        """ update site i by contracting tensor i with "bond" tensors
        """
        if self.verbose:
            print('update 0 site', i, sweep_direction)

        for ii, ket in self.states_dict.items():
            try:
                next_site = site_i_dict[ii]
                if isinstance(next_site, (tuple, list)):
                    next_site = helper_tn.sum_tens(next_site)

                bond_ind = next_site.inds[0][:-2]
                if sweep_direction > 0:
                    next_site.reindex({bond_ind + '_R': bond_ind}, inplace=True)
                else:
                    next_site.reindex({bond_ind + '_L': bond_ind}, inplace=True)

                next_site = qtn.tensor_contract(next_site, ket[i])
                next_site.transpose_like(ket[i], inplace=True)
                ket[i].modify(data=next_site.data)
                ket._cur_orthog = i

            except KeyError:
                pass


    def _replace_1site(self, i: int, site_i_dict: dict[int, Union[Sequence['qtn.Tensor'],'qtn.Tensor']],):
        """ replace site i by contracting tensor i with new tensors
        """
        for ii, ket in self.states_dict.items():
            try:
                next_site = site_i_dict[ii]
                if isinstance(next_site, (tuple, list)):
                    next_site = helper_tn.sum_tens(next_site)

                next_site.transpose_like(ket[i], inplace=True)
                ket[i].modify(data=next_site.data)
                ket._cur_orthog = i
            except KeyError:
                pass

    def take_time_step(self, dt, grid=None, do_adapt=False, init_direction=SweepDirection.RIGHT, **kwargs):
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
        # verbose = True
        self.dt = dt
        L = self.ket.L
        # print('L', self.L, self.ket.L)
        if init_direction != self.direction:
            canon_site = 0 if init_direction == SweepDirection.RIGHT else L - 1
            self.direction = init_direction
            self.canonize(canon_site)

        ## ovlp <Ax|b>, <Ax|x> init envs
        if init_direction == SweepDirection.RIGHT:
            # self._build_all_envs_right(1, canonize=False)

            ### left to right sweep
            print('do adapt', do_adapt)
            self.take_time_step_l2r(dt, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)
            ### right to left sweep
            # self.take_time_step_r2l(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)

        else:
            # self._build_all_envs_left(1, canonize=False)

            ### right to left sweep
            self.take_time_step_r2l(dt, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)
            # ### left to right sweep
            # self.take_time_step_l2r(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)

        print('done tdvp sweep')
        for k, v in self.states_dict.items():
            print('k', k, helper_quimb.inner_bond_sizes(v))

        return self.kets


class BlockTDVPSolver(BlockTDVPIntegrator, BlockTDDMRGSolver):
    """ TD-DMRG solver with multiple components at once
        assume that the grid is ordered by scale
    """

    def _update_1site(self, i: int, site_i_dict: dict[int, 'qtn.Tensor'], direction: 'SweepDirection'):

        if self.verbose:
            print('new TDVP update 1 site', i, direction)

        extra_dict = None
        if isinstance(site_i_dict, tuple):
            site_i_dict, extra_dict = site_i_dict

        at_end = (i == self.L - 1) if direction > 0 else (i == 0)

        if at_end:
            super()._update_1site(i, site_i_dict, direction)
            return

        ## update self.current_state
        new_Rs = {}
        # old_sites = {}
        x_ind = None
        for ii, ket in self.states_dict.items():
            try:
                # old_sites[ii] = ket[i].copy()

                new_ket_tens = site_i_dict[ii]  # .copy()
                if isinstance(new_ket_tens, (tuple, list)):
                    new_ket_tens = helper_tn.sum_tens(new_ket_tens)  ## all other sites are the same (and in canonical form)

                # self.check_projected_source(ii, i, 1)
                # self.check_new_tens(ii, new_ket_tens, i, 1)

                x_ind = ket.bond(i, i + direction)
                left_inds = [ind for ind in ket[i].inds if ind != x_ind]

                if extra_dict is None:
                    ## standard QR
                    new_Q, new_R = qtn.tensor_split(new_ket_tens, left_inds, absorb='right',
                                                    # max_bond=self.max_bond, cutoff=CUTOFF,
                                                    bond_ind=x_ind + '_tmp', method='qr')
                else:
                    ## expand q before doing backwards TE
                    ## extra_dict[ii]:  A[oo,ii] * x[ii] for each ii
                    new_Q, new_R = helper_dmrg.get_expanded_qr([new_ket_tens, *extra_dict[ii]], left_inds,
                                                               # max_bond=self.max_bond, cutoff=CUTOFF,
                                                               bond_ind=x_ind + '_tmp')

                if direction == SweepDirection.LEFT:
                    bond_reindex_dict = {x_ind: x_ind + '_L', x_ind + '_tmp': x_ind + '_R'}
                else:  # direction is to the right
                    bond_reindex_dict = {x_ind: x_ind + '_R', x_ind + '_tmp': x_ind + '_L'}
                new_R.reindex(bond_reindex_dict, inplace=True)

                new_Q.transpose_like(ket[i], inplace=True)
                ket[i].modify(data=new_Q.data)
                new_Rs[ii] = new_R.copy()

            except KeyError:
                pass

        # self.check_projected_state(2, i + direction, 1)

        # print('check self terms')
        # for ii, term in self.self_terms.items():
        #     bra = self.kets[ii]
        #     ket = self.kets[ii]
        #     print('term ket', term.ket is ket)
        #     print('term bra', term.bra is bra)
        #     print('term vec block', term.vec_block.bra is ket, term.vec_block.ket is ket)
        #
        # print('check op terms')
        # for (oo, ii), terms in self.op_terms.items():
        #     bra = self.kets[oo]
        #     ket = self.kets[ii]
        #     for term in terms:
        #         print('term ket', term.ket is ket)
        #         print('term bra', term.bra is bra)
        #         print('term vec block', term.vec_block.bra is ket, term.vec_block.ket is ket)
        #         for op_block in term.op_blocks[0]:
        #             print('term op block', op_block.bra is bra, op_block.ket is ket)
        #
        # pdb.set_trace()

        # self.update_blocks(i, direction=direction)
        for term in self.terms:
            term.extend_env(i, direction=direction)

        # ### Backwards time evolution at bond between i, i + direction
        left_site_pos = i if direction > 0 else i - 1
        new_R_dict = self._bond_time_evolution(-self.dt, left_site_pos, x_ind, site_tens_dict=new_Rs)

        # # td-dmrg analog
        # if left_site_pos < 3:
        #     new_R_dict = {}
        #     for ii, new_R in new_Rs.items():
        #         new_Q = self.states_dict[ii][i].copy()
        #         new_Q.reindex({x_ind: x_ind + '_'}, inplace=True)
        #         new_Q = new_Q.conj()
        #         tens = qtn.tensor_contract(old_sites[ii], new_Q)
        #         new_R_dict[ii] = tens

        # print('new R dict', new_R_dict)
        self._update_0site(i + direction, direction, new_R_dict)
        self.time = self.time - self.dt

        # for ii, ket in self.kets.items():
        #     helper_quimb.check_orthog(ket)
        #
        #     gtn = self.grid.make_gridTN(ket)
        #     plt.figure()
        #     plt.imshow(gtn.get_data())
        #     plt.colorbar()
        #     plt.show()

        return


    def _update_2site(self, left_site_pos: int, site_i_dict: dict[int, 'qtn.Tensor'], direction: 'SweepDirection'):

        if self.verbose:
            print('new TDVP update 2 site', left_site_pos, direction)

        extra_dict = None
        if isinstance(site_i_dict, tuple):
            site_i_dict, extra_dict = site_i_dict

        at_end = (left_site_pos == self.L - 2) if direction > 0 else (left_site_pos == 0)

        if at_end:
            super()._update_2site(left_site_pos, site_i_dict, direction)
            return

        i1 = left_site_pos if direction > 0 else left_site_pos + 1
        i2 = left_site_pos + 1 if direction > 0 else left_site_pos

        ## update self.current_state
        new_Rs = {}
        old_sites = {}
        for ii, ket in self.states_dict.items():
            try:
                ## canonicalize and then back-propagate "site" (if not at end)
                new_ket_tens = site_i_dict[ii]  # .copy()
                old_sites[ii] = ket[i1].copy()
                old_site = qtn.tensor_contract(ket[i1], ket[i2])

                if isinstance(new_ket_tens, (tuple, list)):
                    new_ket_tens = helper_tn.sum_tens(new_ket_tens)  ## all other sites are the same (and in canonical form)

                # self.check_projected_source(ii, left_site_pos, 2)
                # self.check_new_tens(ii, new_ket_tens, left_site_pos, 2)

                x_ind = ket.bond(i1, i2)
                left_inds = [ind for ind in ket[i1].inds if ind != x_ind]
                if extra_dict is None:
                    ## standard SVD
                    q, r = helper_dmrg.tensor_svd(new_ket_tens, left_inds, absorb='right', max_bond=self.max_bond,
                                                  cutoff=self.cutoff, bond_ind=x_ind)
                else:
                    ## expand q before doing backwards TE
                    ## extra_dict[ii]:  A[oo,ii] * x[ii] for each ii
                    print('expanded', ii, len(extra_dict[ii]))
                    q, r = helper_dmrg.get_expanded_qr([new_ket_tens, *extra_dict[ii]], left_inds,
                                                       max_bond=self.max_bond, cutoff=CUTOFF, bond_ind=x_ind)


                # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
                q.transpose_like(ket[i1], inplace=True)
                ket[i1].modify(data=q.data)
                r.transpose_like(ket[i2], inplace=True)
                ket[i2].modify(data=r.data)
                new_Rs[ii] = r

            except KeyError:
                pass

        # print('check projected state i2', i2)
        # self.check_projected_state(2, i2, 1)
        # self.check_projected_source(2, i2, 1)

        # self.update_blocks(i, direction=direction)
        for term in self.terms:
            term.extend_env(i1, direction=direction)

        ### Backwards time evolution
        new_Rs_dict_tdvp = self._site_time_evolution(-self.dt, i2, 1, sweep_direction=direction,
                                                     site_tens_dict=new_Rs)
        if isinstance(new_Rs_dict_tdvp, tuple):
            new_Rs_dict_tdvp, _ = new_Rs_dict_tdvp

        # # # td-dmrg equivalent
        # new_Rs_dict = {}
        # for ii, new_R in new_Rs.items():
        #     new_Q = self.states_dict[ii][i1].copy()
        #     new_Q.reindex({x_ind: x_ind + '_'}, inplace=True)
        #     new_Q = new_Q.conj()
        #
        #     old_R = self.states_dict[ii][i2].copy()
        #
        #     tens = qtn.tensor_contract(old_sites[ii], new_Q, old_R)
        #     new_Rs_dict[ii] = tens
        # self._replace_1site(i2, new_Rs_dict)
        # tdmrg_kets = {k: v.copy() for k, v in self.states_dict.items()}

        self._replace_1site(i2, new_Rs_dict_tdvp)
        # tdvp_kets = {k: v.copy() for k, v in self.states_dict.items()}

        # for ii in tdvp_kets.keys():
        #     ket_tdvp = tdvp_kets[ii]
        #     ket_tdmrg = tdmrg_kets[ii]
        #     gtn_tdvp = self.grid.get_ones_mps()
        #     gtn_tdmrg = self.grid.get_ones_mps()
        #     self.grid.dmrg_to_gtn_format(gtn_tdvp, ket_tdvp)
        #     self.grid.dmrg_to_gtn_format(gtn_tdmrg, ket_tdmrg)
        #
        #     tdvp_data = gtn_tdvp.get_data()
        #     tdmrg_data = gtn_tdmrg.get_data()
        #     err_data = tdvp_data - tdmrg_data
        #
        #     ax_x, ax_y = self.grid.axes[:2]
        #     tdvp_data = ax_x.basis.get_realspace_1D(tdvp_data, 0)
        #     tdvp_data = ax_y.basis.get_realspace_1D(tdvp_data, 1)
        #     tdmrg_data = ax_x.basis.get_realspace_1D(tdmrg_data, 0)
        #     tdmrg_data = ax_y.basis.get_realspace_1D(tdmrg_data, 1)
        #     err_data = ax_x.basis.get_realspace_1D(err_data, 0)
        #     err_data = ax_y.basis.get_realspace_1D(err_data, 1)
        #
        #     plt.figure()
        #     plt.imshow(np.real(tdvp_data))
        #     plt.colorbar()
        #     plt.figure()
        #     plt.imshow(np.real(tdmrg_data))
        #     plt.colorbar()
        #     plt.figure()
        #     plt.imshow(np.real(err_data))
        #     plt.title(f'diff {i2}, comp{ii}')
        #     plt.colorbar()
        #     plt.show()


        return


class BlockTDVPXSolver(BlockTDVPIntegrator, BlockTDDMRGXSolver):
    """ TD-DMRG solver with multiple components at once
        assume that the grid is ordered by scale
    """

    def _update_1site(self, i: int, site_i_dict: dict[int, 'qtn.Tensor'], direction: 'SweepDirection'):

        if self.verbose:
            print('new TDVP update 1 site', i, direction)

        at_end = (i == self.L - 1) if direction > 0 else (i == 0)

        if at_end:
            super()._update_1site(i, site_i_dict, direction)
            return

        ## update self.current_state
        new_Rs = {}
        old_sites = {}
        x_ind = None
        for ii, ket in self.states_dict.items():
            try:
                old_sites[ii] = ket[i].copy()
                new_ket_tens = site_i_dict[ii]  # .copy()

                if isinstance(new_ket_tens, (tuple, list)):
                    new_ket_tens = helper_tn.sum_tens(new_ket_tens)  ## all other sites are the same (and in canonical form)

                x_ind = ket.bond(i, i + direction)
                left_inds = [ind for ind in ket[i].inds if ind != x_ind]

                phys_inds = [self.ket.site_ind(i)]
                right_inds = [x_ind]
                new_Q, new_R, inds_r, inds_c = helper_cross.tensor_compress(new_ket_tens, phys_inds, right_inds,
                                                                            return_inds=True, bond_ind=x_ind + '_tmp')
                ket.select_inds[i] = inds_r
                # ket.select_inds.pop(i + direction)

                # new_Q, new_R = qtn.tensor_split(new_ket_tens, left_inds, absorb='right',
                #                                 # max_bond=self.max_bond, cutoff=CUTOFF,
                #                                 bond_ind=x_ind + '_tmp', method='qr')

                if direction == SweepDirection.LEFT:
                    bond_reindex_dict = {x_ind: x_ind + '_L', x_ind + '_tmp': x_ind + '_R'}
                else:  # direction is to the right
                    bond_reindex_dict = {x_ind: x_ind + '_R', x_ind + '_tmp': x_ind + '_L'}
                new_R.reindex(bond_reindex_dict, inplace=True)

                new_Q.transpose_like(ket[i], inplace=True)
                ket[i].modify(data=new_Q.data)
                new_Rs[ii] = new_R.copy()

            except KeyError:
                pass

        # print('check self terms')
        # for ii, term in self.self_terms.items():
        #     bra = self.kets[ii]
        #     ket = self.kets[ii]
        #     print('term ket', term.ket is ket)
        #     print('term bra', term.bra is bra)
        #     print('term vec block', term.vec_block.bra is ket, term.vec_block.ket is ket)
        #
        # print('check op terms')
        # for (oo, ii), terms in self.op_terms.items():
        #     bra = self.kets[oo]
        #     ket = self.kets[ii]
        #     for term in terms:
        #         print('term ket', term.ket is ket)
        #         print('term bra', term.bra is bra)
        #         print('term vec block', term.vec_block.bra is ket, term.vec_block.ket is ket)
        #         for op_block in term.op_blocks[0]:
        #             print('term op block', op_block.bra is bra, op_block.ket is ket)
        #
        # pdb.set_trace()

        # self.update_blocks(i, direction=direction)
        for term in self.terms:
            term.extend_env(i, direction=direction)

        ### Backwards time evolution at bond between i, i + direction
        left_site_pos = i if direction > 0 else i - 1
        new_R_dict = self._bond_time_evolution(-self.dt, left_site_pos, x_ind, site_tens_dict=new_Rs)

        # # td-dmrg analog
        # new_R_dict = {}
        # for ii, new_R in new_Rs.items():
        #     new_Q = self.states_dict[ii][i].copy()
        #     new_Q.reindex({x_ind: x_ind + '_'}, inplace=True)
        #     new_Q = new_Q.conj()
        #     tens = qtn.tensor_contract(old_sites[ii], new_Q)
        #     new_R_dict[ii] = tens

        # print('new R dict', new_R_dict)
        self._update_0site(i + direction, direction, new_R_dict)
        self.time = self.time - self.dt

        # for ii, ket in self.kets.items():
        #     helper_quimb.check_orthog(ket)
        #
        #     gtn = self.grid.make_gridTN(ket)
        #     plt.figure()
        #     plt.imshow(gtn.get_data())
        #     plt.colorbar()
        #     plt.show()

        return


    def _update_2site(self, left_site_pos: int, site_i_dict: dict[int, 'qtn.Tensor'], direction: 'SweepDirection'):

        if self.verbose:
            print('new TDVP-X update 2 site', left_site_pos, direction)

        at_end = (left_site_pos == self.L - 2) if direction > 0 else (left_site_pos == 0)

        if at_end:
            super(BlockTDDMRGXSolver, self)._update_2site(left_site_pos, site_i_dict, direction)
            return

        i1 = left_site_pos if direction > 0 else left_site_pos + 1
        i2 = left_site_pos + 1 if direction > 0 else left_site_pos

        ## update self.current_state
        new_Rs = {}
        old_sites = {}
        for ii, ket in self.states_dict.items():
            try:
                ## canonicalize and then back-propagate "site" (if not at end)
                new_ket_tens = site_i_dict[ii]  # .copy()
                old_sites[ii] = ket[i1].copy()

                if isinstance(new_ket_tens, (tuple, list)):
                    new_ket_tens = helper_tn.sum_tens(new_ket_tens)  ## all other sites are the same (and in canonical form)

                x_ind = ket.bond(i1, i2)

                phys_inds = [self.ket.site_ind(i1)]
                right_inds = [ind for ind in self.ket[i2].inds if ind != x_ind]
                q, r, inds_r, inds_c = helper_cross.tensor_compress(new_ket_tens, phys_inds, right_inds,
                                                                    return_inds=True, bond_ind=x_ind + '_tmp')
                ket.select_inds[i1] = inds_r
                ket.select_inds.pop(i2)

                # left_inds = [ind for ind in ket[i1].inds if ind != x_ind]
                # q, r = qtn.tensor_split(new_ket_tens, left_inds, absorb='right', max_bond=self.max_bond,
                #                         cutoff=(CUTOFF if self.cutoff is None else self.cutoff),
                #                         bond_ind=x_ind)

                # helper_dmrg.update_1site(self.ket, i, site_i, direction, max_bond=self.max_bond)
                q.transpose_like(ket[i1], inplace=True)
                ket[i1].modify(data=q.data)
                r.transpose_like(ket[i2], inplace=True)
                r.modify(inds=ket[i2].inds)
                new_Rs[ii] = r

            except KeyError:
                pass

        # self.update_blocks(i, direction=direction)
        for term in self.terms:
            term.extend_env(i1, direction=direction)

        # ### Backwards time evolution
        new_Rs_dict_tdvp = self._site_time_evolution(-self.dt, i2, 1, sweep_direction=direction, site_tens_dict=new_Rs)

        # # # td-dmrg equivalent
        # new_Rs_dict = {}
        # for ii, new_R in new_Rs.items():
        #     new_Q = self.states_dict[ii][i1].copy()
        #     new_Q.reindex({x_ind: x_ind + '_'}, inplace=True)
        #     new_Q = new_Q.conj()
        #
        #     old_R = self.states_dict[ii][i2].copy()
        #
        #     tens = qtn.tensor_contract(old_sites[ii], new_Q, old_R)
        #     new_Rs_dict[ii] = tens

        # self._replace_1site(i2, new_Rs_dict)
        # tdmrg_kets = {k: v.copy() for k, v in self.states_dict.items()}

        self._replace_1site(i2, new_Rs_dict_tdvp)
        # tdvp_kets = {k: v.copy() for k, v in self.states_dict.items()}

        # for ii in tdvp_kets.keys():
        #     ket_tdvp = tdvp_kets[ii]
        #     ket_tdmrg = tdmrg_kets[ii]
        #     gtn_tdvp = self.grid.make_gridTN(ket_tdvp)
        #     gtn_tdmrg = self.grid.make_gridTN(ket_tdmrg)
        #
        #     tdvp_data = gtn_tdvp.get_data()
        #     tdmrg_data = gtn_tdmrg.get_data()
        #
        #     # plt.figure()
        #     # plt.imshow(tdvp_data)
        #     # plt.colorbar()
        #     # plt.figure()
        #     # plt.imshow(tdmrg_data)
        #     # plt.colorbar()
        #     plt.figure()
        #     plt.imshow(tdvp_data - tdmrg_data)
        #     plt.title(f'{i2}, comp{ii}')
        #     plt.colorbar()
        #     plt.show()


        return
