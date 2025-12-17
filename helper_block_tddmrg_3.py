import pdb

import numpy as np
import itertools
import quimb
import scipy.linalg

import helper_quimb
from setup_.defaults import *
from local_solvers.defaults import *

import quimb.tensor as qtn
from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
import helper_quimb as helper
# import helper_dmrg
# import helper_dmrg_2
from local_solvers.mps_classes import MPS
import local_solvers.helper_dmrg_loc as helper_dmrg

import helper_TE
import local_solvers.helper_tn as helper_tn
# from helper_dmrg import *
# from helper_tdvp_v2 import *
from local_solvers.time_integrator import TDVP_DMRG, TDDMRG, TimeIntegrator
from helper_block_dmrg import combine_vec_tens, combine_mat_tens, extract_vec_tens

import local_solvers.helper_cross_2 as helper_cross
from local_solvers.terms_3 import Term, Term_DMRG, Term_Cross
import local_solvers.helper_mixed as helper_mixed
from local_solvers.terms_mixed import Term_Mixed
from helper_dmrg import (qtn_conjugate_gradient_squared_1site,
                         qtn_conjugate_gradient_descent_1site)

from grid1D import Grid1D

# class BaseSolver(Enum):
#     LOCAL = LocalSolver
#     DMRG = DMRGSolver
#     # TDDMRG = TDDMRGSolver_v4
#     # TD_DMRG = TDDMRG
#     # TDVP = TDVP_DMRG
#     TDDMRG = TimeIntegrator
#     LINEAR = LinearSolver
#     # LINEAR2 = helper_dmrg_2.LinearSolver

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

def block_tddmrg(dt: float, ncomps: int, A_mats: dict[tuple[int, int], Sequence['MPO_type']],
                 x_vecs: dict[int, 'qtn.MatrixProductState'],
                 b_vecs: dict[int, Sequence['MPS_type']] = None,
                 masks: dict[int, Sequence['MPO_type']] = None,
                 verbose_output=False,
                 init_direction=SweepDirection.RIGHT,
                 constraints: dict[int, 'qtn.MatrixProductOperator'] = None,
                 constraint_vals: dict[int, 'qtn.MatrixProductState'] = None,
                 shared_projs: list[set] = None,
                 backward_weights: dict[tuple[int, int], float]=None,
                 te_order=4, grid=None, return_info=False, **solver_kwargs):
    print('BLOCK TD-DMRG 3')


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


    solver = BlockTDDMRGSolver(ncomps, x_vecs, sources=b_vecs, operators=A_mats, masks=masks,
                               te_order=te_order, direction=init_direction,
                               constraints=constraints, constraint_vals=constraint_vals,
                               grid=grid,
                               **solver_kwargs)

    solver.take_time_step(dt, init_direction=init_direction)

    if verbose_output:
        return solver.kets, solver.err, solver.is_conv
    elif return_info:
        return solver.kets, {'num_evals': solver.num_evals}
    else:
        return solver.kets


def block_tddmrgx(dt: float, ncomps: int, A_mats: dict[tuple[int, int], Sequence['MPO_type']],
                  x_vecs: dict[int, 'qtn.MatrixProductState'],
                  b_vecs: dict[int, Sequence['MPS_type']] = None,
                  masks: dict[int, Sequence['MPO_type']] = None,
                  verbose_output=False,
                  init_direction=SweepDirection.RIGHT,
                  constraints: dict[int, 'qtn.MatrixProductOperator'] = None,
                  constraint_vals: dict[int, 'qtn.MatrixProductState'] = None,
                  shared_projs: list[set] = None,
                  backward_weights: dict[tuple[int, int], float]=None,
                  verbose_plot=False,
                  te_order=4, grid=None, return_info=False, **solver_kwargs):
    print('BLOCK TD-DMRG-X 3')

    # x_vecs_ = {}
    # for k, v in x_vecs.items():
    #     if v.exponent < -10:
    #         if helper_quimb.norm(v) < np.sqrt(CUTOFF):
    #             continue
    #     x_vecs_[k] = v
    # x_vecs = x_vecs_

    ref_x = x_vecs[next(iter(x_vecs))]

    for k, x in x_vecs.items():
        x.distribute_exponent()
        if x != ref_x:
            helper_quimb.match_inner_inds(x, ref_x, inplace=True)

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

    solver = BlockTDDMRGXSolver(ncomps, x_vecs, sources=b_vecs, operators=A_mats, masks=masks,
                                te_order=te_order, direction=init_direction,
                                constraints=constraints, constraint_vals=constraint_vals,
                                grid=grid,
                                **solver_kwargs)
    solver.verbose_plot = verbose_plot

    solver.take_time_step(dt, init_direction=init_direction)

    if verbose_output:
        return solver.kets, solver.err, solver.is_conv
    elif return_info:
        return solver.kets, {'num_evals': solver.num_evals}
    else:
        return solver.kets


def block_tddmrgm(dt: float, ncomps: int, A_mats: dict[tuple[int, int], Sequence['MPO_type']],
                  x_vecs: dict[int, 'qtn.MatrixProductState'],
                  b_vecs: dict[int, Sequence['MPS_type']] = None,
                  masks: dict[int, Sequence['MPO_type']] = None,
                  verbose_output=False,
                  init_direction=SweepDirection.RIGHT,
                  constraints: dict[int, 'qtn.MatrixProductOperator'] = None,
                  constraint_vals: dict[int, 'qtn.MatrixProductState'] = None,
                  shared_projs: list[set] = None,
                  backward_weights: dict[tuple[int, int], float]=None,
                  te_order=4, grid=None, return_info=False, verbose_plot=False, **solver_kwargs):
    print('BLOCK TD-DMRG-MIXED 3')

    # x_vecs_ = {}
    # for k, v in x_vecs.items():
    #     if v.exponent < -10:
    #         if helper_quimb.norm(v) < np.sqrt(CUTOFF):
    #             continue
    #     x_vecs_[k] = v
    # x_vecs = x_vecs_

    ref_x = x_vecs[next(iter(x_vecs))]

    for k, x in x_vecs.items():
        x.distribute_exponent()
        if x != ref_x:
            helper_quimb.match_inner_inds(x, ref_x, inplace=True)

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
                    # print('x vec', helper_quimb.max_inner_bond(x_vecs[k]))
                    helper_quimb.add_MPS_list([x_vecs[k], *xs], do_final_update=False, inplace=True,
                                              direction=init_direction * -1,
                                              compress_opts={'max_bond': solver_kwargs.get('max_bond', None),
                                                             'cutoff': solver_kwargs.get('cutoff', CUTOFF)}, )
                    print('expanded x vec', helper_quimb.max_inner_bond(x_vecs[k]))


    solver = BlockTDDMRGMSolver(ncomps, x_vecs, sources=b_vecs, operators=A_mats, masks=masks,
                                te_order=te_order, direction=init_direction,
                                constraints=constraints, constraint_vals=constraint_vals,
                                grid=grid,
                                **solver_kwargs)
    solver.verbose_plot = verbose_plot

    solver.take_time_step(dt, init_direction=init_direction)

    if verbose_output:
        return solver.kets, solver.err, solver.is_conv
    elif return_info:
        return solver.kets, {'num_evals': solver.num_evals}
    else:
        return solver.kets


class BlockTimeIntegrator:
    """ TD-DMRG solver with multiple components at once
        assume that the grid is ordered by scale
    """

    def __init__(self, ncomps, x_state: dict[int, 'qtn.MatrixProductState'],
                 sources: Optional[dict[int, Sequence['MPS_type']]] = None,
                 operators: Optional[dict[tuple[int, int], Sequence['MPO_type']]] = None,
                 masks: Optional[dict[int, Sequence['MPS_type']]] = None,
                 # operators_H: Optional[dict[tuple[int, int], Sequence['MPO_type']]] = None,
                 # constraints: Optional[Sequence[dict[int, Sequence['MPO_type']]]] = None,
                 # constraint_vals: Optional[Sequence[dict[int, 'MPS_type']]] = None,
                 # base_solver=BaseSolver.TDDMRG,
                 mps_inds: Optional[Sequence[int]] = None,
                 # left_envs: Optional[dict[int,Sequence['Environment']]] = None,
                 # right_envs: Optional[dict[int,Sequence['Environment']]] = None,
                 # virtual: bool = False,
                 conv_tol=DEFAULT_CONV_TOL,
                 max_iter=DEFAULT_MAX_ITER, max_tot_iter=DEFAULT_MAX_TOT_ITER,
                 max_wrong_iter=DEFAULT_MAX_WRONG_ITER,
                 direction=SweepDirection.RIGHT,
                 te_order=4, max_bond=None, cutoff=None,
                 grid: 'Grid1D'=None, verbose=False,
                 solver=None,
                 backward_weights: dict[tuple[int, int], Sequence[float]] = None,
                 expand_kets: bool=False,
                 **env_kwargs):
        """ each solver indexed by output ind, input ind (j, i)
                computes b_eff:  eg. d/dT*[s] <x_j | b_i>
                computes A_eff:  eg. d/dT*[s] d/dT[s] <x_j | O_ji | x_i>

                e.g., vlasov-Maxwell
                fe, fi, Ex, Ey, Ez, Bx, By, Bz
                A_ee, A_ii = v d/dx + F(E, B) * d/dv
                A_EB = curl(B)
                A_BE = curl(E)
                b_E = current(fe, fi)
        """
        self.num_evals = {k: 0 for k in x_state.keys()}

        self.num_comps = ncomps

        for k, v in x_state.items():
            if not isinstance(v, MPS):
                v.view_as(MPS, inplace=True)

        self.states_dict: dict[int, MPS] = x_state
        self.init_ket = {k: v.copy() for k, v in x_state.items()}
        self.operators = operators
        self.sources = sources
        self.masks = masks
        self.direction = direction

        self.te_order = te_order

        self.err = np.inf
        self.is_conv = False

        self.conv_tol = conv_tol if conv_tol is not None else DEFAULT_CONV_TOL
        self.max_iter = max_iter
        self.max_tot_iter = max_tot_iter
        self.max_wrong_iter = max_wrong_iter
        self.max_bond = max_bond
        self.cutoff = cutoff

        self.grid = grid
        self.dt = None
        self.time = 0.0
        self.verbose = verbose
        self.backward_weights = backward_weights

        if solver is not None:  ## for copying
            pass

        else:

            if x_state is None:     ## shouldn't be none
                x_state = {}

            i = 0 if self.direction == SweepDirection.RIGHT else self.L - 1
            print('target canon', i)

            ## add bras where expected
            zeros = {}
            for (oo, ii) in self.operators.keys():
                if oo not in x_state:
                    zero_oo = zeros.get(oo, [])
                    if ii in x_state:
                        x = x_state[ii]
                        ops = self.operators[(oo,ii)]
                        zero_oo += [helper.apply_zipup(op, x) for op in ops]
                        # zero_oo += [x_state[ii]]
                        print('adding bra where expected', oo, ii)
                    zeros[oo] = zero_oo
                    # if len(zero_oo) > 0:
                    #     new_oo = helper_quimb.add_MPS_list(zero_oo, update_with_zero=True)
                    #     x_state[oo] = new_oo

            ref_mps = x_state[next(iter(x_state))]
            for oo, zero_oo in zeros.items():
                print('oo', len(zero_oo))
                if len(zero_oo) > 0:
                    new_oo = helper_quimb.add_MPS_list(zero_oo, update_with_zero=True, direction=self.direction * -1,
                                                       norm_cutoff=np.sqrt(cutoff))  ## should ideally divide by dt
                    if new_oo is not None:
                        helper_quimb.match_inner_inds(new_oo, ref_mps, inplace=True)
                        x_state[oo] = new_oo
                        helper_quimb.check_orthog(new_oo)

            ### expand basis of initial state using sources
            if sources is not None and (self.term_class is Term_DMRG):
                for k, src_kets in sources.items():     ## a list of MPS
                    x_ket = x_state.get(k, None)
                    x_ket = helper.add_MPS_list([x_ket, *src_kets], do_final_update=False, inplace=False,
                                                direction=self.direction * -1)
                    x_state[k] = x_ket

            ### expand basis of initial state using masks
            if masks is not None and (self.term_class is Term_DMRG):
                print('expand initial state with masks')
                for k, msk_kets in masks.items():     ## a list of MPS
                    x_ket = x_state.get(k, None)
                    if x_ket is not None:
                        x_ket = helper.add_MPS_list([x_ket, *msk_kets], do_final_update=False, inplace=False)
                        x_state[k] = x_ket

            # ref_init_x = x_state[next(iter(x_state))]

            if not expand_kets:
                print('canonize (no expand)')
                self.canonize(i)
            else:
                print('expand ket')
                x_state = self.expand_kets(i, x_state, operators, sources=sources, max_bond=max_bond, cutoff=cutoff)
                # self.verbose = True
                # try:
                #     x_state = self.expand_kets(i, x_state, operators, sources=sources, max_bond=max_bond, cutoff=cutoff)
                #     self.verbose = True
                # except np.linalg.LinAlgError:
                #     print('expand ket failed')
                #     self.canonize(i)
                #     self.verbose = False

            ## distribute exponent
            for oo, ket in x_state.items():
                ket[i].modify(apply=lambda x: x * 10**ket.exponent)
                ket.exponent = 0.0

            # for k, v in x_state.items():
            #     print('init ket post canonize', k, v)

            self.states_dict: dict[int, MPS] = x_state
            self.self_terms: dict[int, Term] = {}
            self.op_terms: dict[tuple[int, int], Sequence[Term]] = {}
            self.source_terms: dict[int, Sequence[Term]] = {}
            self.mask_terms: dict[int, Sequence[Term]] = {}
            self.initialize_terms()

    combined_ind_o = 'out_ind'
    combined_ind_i = 'in_ind'

    @property
    def L(self):
        k = next(iter(self.states_dict))
        return self.states_dict[k].L

    @property
    def term_class(self):
        raise NotImplementedError

    @property
    def solver_type(self):
        raise NotImplementedError

    @classmethod
    def expand_kets(cls, target_orthog: int, kets: dict[int, qtn.MatrixProductState],
                    operators: dict[tuple[int, int], Sequence[qtn.MatrixProductOperator]],
                    sources: dict[int, Sequence[qtn.MatrixProductState]] = None,
                    max_bond: int = None, cutoff: float=None,):

        L = kets[next(iter(kets))].L
        if target_orthog != 0 and target_orthog != L - 1:
            raise NotImplementedError

        compress_opts = {'max_bond': max_bond, 'cutoff': cutoff}

        if sources is None:
            sources = {}

        targets = {}
        for (oo, ii), ops in operators.items():
            if oo not in targets:
                targets[oo] = []
            targets[oo] += [helper_quimb.apply_zipup(op, kets[ii], compress=True, compress_opts=compress_opts)
                            for op in ops]

        for oo, sources in sources.items():
            if oo not in targets:
                targets[oo] = []
            targets[oo] += sources

        direction = 1 if target_orthog == L - 1 else -1

        for oo, mps_list in targets.items():
            new_oo = helper_quimb.add_MPS_list([kets[oo], *mps_list], direction=direction,
                                               compress_opts=compress_opts, do_final_update=False,
                                               verbose=(oo==5))
            kets[oo] = new_oo

        return kets


    def _get_linop_terms(self, init_x, init_bra, operators):
        raise NotImplementedError

    def _get_mask_terms(self, init_bra, masks):
        raise NotImplementedError

    def _init_all_linear_terms(self):
        op_terms = {}
        ref_op = self.operators[next(iter(self.operators))][0]
        for (oo, ii), As in self.operators.items():
            ## As: a list of operators which may be updated during the sweep
            ## need to keep track of the MPSs used to build As and how they're built

            init_x = self.states_dict.get(ii, None)
            init_bra = self.states_dict.get(oo, None)

            if init_x is None or init_bra is None:
                continue

            if ref_op is None:
                try:
                    ref_op = next(op for op in As if op is not None)
                except StopIteration:
                    raise StopIteration
                # ref_op = As[0]
            else:
                for i in range(len(As)):
                    if As[i] is None:  continue
                    As[i].upper_ind_id = ref_op.upper_ind_id + '_tmp'
                    As[i].lower_ind_id = ref_op.lower_ind_id
                    As[i].upper_ind_id = ref_op.upper_ind_id
                    helper.match_inner_inds(As[i], ref_mps=ref_op, inplace=True)

            op_terms[(oo, ii)] = self._get_linop_terms(init_x, init_bra, As)

        self.op_terms = op_terms
        return

    def _init_all_source_terms(self):
        source_terms = {}
        if self.sources is not None:
            for oo, bs in self.sources.items():
                init_bra = self.states_dict.get(oo, None)
                source_terms[oo] = [self.term_class(b, init_bra, num_tiers=1) for b in bs]
        self.source_terms = source_terms
        return

    def _init_all_mask_terms(self):
        mask_terms = {}
        if self.masks is not None:
            for oo, bs in self.masks.items():
                init_bra = self.states_dict.get(oo, None)
                if init_bra is not None:
                    mask_terms[oo] = self._get_mask_terms(init_bra, bs)
        self.mask_terms = mask_terms
        return

    def initialize_terms(self):

        self_terms = {}
        for oo, ket in self.states_dict.items():
            self_terms[oo] = self.term_class(ket)
        self.self_terms = self_terms

        self._init_all_linear_terms()
        self._init_all_source_terms()
        self._init_all_mask_terms()

        for term_ in self.terms:
            term_._direction = self.direction
            term_.max_bond = self.max_bond
            term_.initialize()

            # if term_.operators is not None and len(term_.operators) > 0:
            #     ket2 = term_.ket.copy()
            #     bra2 = term_.bra.copy()
            #     bra2.mangle_inner_(append='_')
            #     bra2.site_ind_id = ket2.site_ind_id + '_'

        return

    def add_new_state(self, comp: int, cur_orthog: int):

        pdb.set_trace()
        ref_bras = []
        if comp in self.states_dict:
            raise RuntimeError(f'comp {comp} already in states_dict')

        for (oo, ii), terms in self.op_terms.items():
            if oo != comp:
                continue
            ref_bras += [terms[0].bra.copy()]

        direction = (-1 if cur_orthog < ref_bras[0].L // 2 else 1)
        ref_bra = helper_quimb.add_MPS_list(ref_bras, direction=direction,
                                            compress_opts={'cutoff':self.cutoff, 'max_bond':self.max_bond},)

        init_cur_orthog = (0 if direction < 0 else ref_bra.L -1)
        canonize_func = self.term_class.canonize_func
        ref_bra = canonize_func(ref_bra, cur_orthog, cur_orthog=init_cur_orthog)
        helper_quimb.match_inner_inds(ref_bra, self.ket, inplace=True)
        ref_bra[cur_orthog].modify(apply=lambda x: x * 0)
        self.states_dict[comp] = ref_bra
        self_term = self.term_class(ref_bra)
        self_term.cur_orthog = cur_orthog
        self_term.initialize()
        self.self_terms[comp] = self_term

        ## initialize envs
        for (oo, ii), terms in self.op_terms.items():
            if oo != comp:
                continue

            for term in terms:
                term.bra = ref_bra
                term.reinitialize()
                term.cur_orthog = cur_orthog
                term.initialize()



    # @property
    # def left_envs(self):
    #     all_left_envs = {}
    #     for k, term in self.op_terms.items():
    #         all_left_envs[k] = term.left_envs
    #
    #     for k, b_envs in self.b_envsLs.items():
    #         if k not in all_left_envs:
    #             all_left_envs[k] = b_envs
    #         else:
    #             all_left_envs[k] = all_left_envs + b_envs
    #
    #     return all_left_envs
    #
    # @property
    # def right_envs(self):
    #     all_right_envs = {}
    #     for k, solver in self.solvers.items():
    #         all_right_envs[k] = solver.right_envs
    #
    #     for k, b_envs in self.b_envsRs.items():
    #         if k not in all_right_envs:
    #             all_right_envs[k] = b_envs
    #         else:
    #             all_right_envs[k] = all_right_envs + b_envs
    #
    #     return all_right_envs

    @property
    def kets(self):
        return self.states_dict

    @property
    def ket(self):
        tmp = iter(self.states_dict)
        while True:
            ket = self.states_dict[next(tmp)]
            if ket is not None:
                break
        # solver = self.solvers[next(iter(self.solvers))]
        return ket

    @property
    def terms(self) -> Sequence['Term']:

        terms = []
        for k, v in self.self_terms.items():
            terms += [v]
        for k, v in self.op_terms.items():
            terms += [*v]
        for k, v in self.source_terms.items():
            terms += [*v]
        for k, v in self.mask_terms.items():
            terms += [*v]

        return terms

    # @property
    # def bras(self):
    #     all_bras = {}
    #     for (oo, ii), solver in self.solvers.items():
    #         all_bras[oo] = solver.bra
    #     return all_bras
    #
    # @property
    # def bra(self):
    #     ### do it tihs way for the correct index labels
    #     tmp = iter(self.solvers)
    #     while True:
    #         solver = self.solvers[next(tmp)]
    #         if solver.bra is not None:
    #             break
    #     # solver = self.solvers[next(iter(self.solvers))]
    #     return solver.bra

    @property
    def component_ind(self):
        return self.ket.site_ind_id

    def create_like(self, **kwargs):

        states_dict = kwargs.get('states_dict', self.states_dict)
        operators = kwargs.get('operators', self.operators)
        sources = kwargs.get('sources', self.sources)

        new_solver = self.__class__(self.num_comps, states_dict, operators=operators, sources=sources,
                                    conv_tol = self.conv_tol, max_iter = self.max_iter,
                                    max_tot_iter = self.max_tot_iter, max_wrong_iter = self.max_wrong_iter,
                                    max_bond = self.max_bond, cutoff = self.cutoff, te_order=self.te_order,
                                    # solver=self,
                                    )
        return new_solver

    def copy(self):
        states_dict = {k: ket.copy() for k, ket in self.states_dict.items()}

        new_solver = self.__class__(self.num_comps, states_dict, operators=self.operators, sources=self.sources,
                                    conv_tol=self.conv_tol, max_iter=self.max_iter,
                                    max_tot_iter=self.max_tot_iter, max_wrong_iter=self.max_wrong_iter,
                                    direction=self.direction,
                                    max_bond=self.max_bond, cutoff=self.cutoff, te_order=self.te_order,
                                    solver=self,
                                    )

        new_op_terms = {}
        for (oo, ii), terms in self.op_terms.items():
            new_op_terms[(oo,ii)] = [term.copy(ket_copy=new_solver.kets[ii], bra_copy=new_solver.kets[oo])
                                     for term in terms]

        new_source_terms = {}
        for oo, terms in self.source_terms.items():
            new_source_terms[oo] = [term.copy(bra_copy=new_solver.kets[oo]) for term in terms]

        new_solver.op_terms = new_op_terms
        new_solver.source_terms = new_source_terms
        new_solver.time = self.time
        new_solver.dt = self.dt

        return new_solver


    def get_ket_to_bra_inds(self, left_site_pos, nsites=1, comp=None) -> dict[str, str]:
        """ inds mapping inds on self.bra to corresponding inds on self.ket
        """
        dict = {ind: ind + '_' for ind in self.ket_inds(left_site_pos, nsites, comp=comp)}
        dict[self.component_ind] = self.component_ind + '_'
        return dict

    def get_bra_to_ket_inds(self, left_site_pos, nsites=1, comp=None) -> dict[str, str]:
        ket_to_bra = self.get_ket_to_bra_inds(left_site_pos, nsites, comp=comp)
        bra_to_ket = {item: k for k, item in ket_to_bra.items()}
        return bra_to_ket

    def ket_inds(self, left_site_pos, nsites, comp=None):
        ref_ket = self.ket if comp is None else self.kets.get(comp, self.ket)
        iL, iR = left_site_pos, nsites + left_site_pos - 1
        left_ind = [ref_ket.bond(iL, iL - 1)] if iL > 0 else []
        right_ind = [ref_ket.bond(iR, iR + 1)] if iR < ref_ket.L - 1 else []
        site_ind = [ref_ket.site_ind(i) for i in range(iL, iR + 1)]
        return [*left_ind, *right_ind, *site_ind]

    def bra_inds(self, left_site_pos, nsites, ket_inds=None):
        ket_inds = self.ket_inds(left_site_pos, nsites) if ket_inds is None else ket_inds
        ket_to_bra_inds = self.get_ket_to_bra_inds(left_site_pos, nsites)
        bra_inds = [ket_to_bra_inds[k] for k in ket_inds]
        return bra_inds


    def canonize(self, i, cur_orthog=None):

        canonize_func = self.term_class.canonize_func

        for k, ket in self.states_dict.items():
            # helper_quimb.canonize(ket, scale=False, i=i, cur_orthog=cur_orthog)
            canonize_func(ket, i, cur_orthog=cur_orthog)

        try:
            for term in self.terms:
                if term.initialized:
                    term.reinitialize()
                    term.cur_orthog = i
                    term.direction = self.direction
                    term.initialize()
        except AttributeError:     ## terms not yet defined
            pass

    def _build_all_envs_right(self, nsites: int, end: int = None, canonize=True):
        """ compute all right envs for <Ax|b>
        """
        L = self.L
        end = 0 if end is None else end

        for term in self.terms:
            for i in range(L-1, end -1, -1):
                term.extend_env(i, direction=-1)

        return

    def _build_all_envs_left(self, nsites: int, end: int = None, canonize=True):
        """ compute all left envs for <Ax|b>
        """
        end = self.L if end is None else end

        for term in self.terms:
            for i in range(end):
                term.extend_env(i, direction=1)
        return

    def _update_envs_left(self, pos: int, canonize=False):

        if canonize:
            self.canonize(pos + 1)

        for term in self.terms:
            term.extend_env(pos, direction=1)

        return

    def _update_envs_right(self, pos: int, canonize=False):
        if canonize:
            self.canonize(pos - 1)

        for term in self.terms:
            term.extend_env(pos, direction=-1)

        return


    def check_new_tens(self, out_comp: int, new_tens: 'qtn.Tensor', left_site_pos: int, nsites: int, plot=True):
        copy_oo = self.kets[out_comp]
        oo = out_comp
        tens_ = new_tens

        bra = copy_oo.copy()
        if nsites == 1:
            m = bra[left_site_pos]
            m.modify(data=tens_.data, inds=tens_.inds)
        else:
            m1 = bra[left_site_pos]
            m2 = bra[left_site_pos + 1]
            left_inds = tens_.bonds(m1)
            T1, T2 = qtn.tensor_split(tens_, left_inds, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
            T1.transpose_like(m1, inplace=True)
            T2.transpose_like(m2, inplace=True)
            m1.modify(data=T1.data)
            m2.modify(data=T2.data)

        gtn = self.grid.make_gridTN(bra)
        data = gtn.get_realspace_data()

        if plot:
            plt.figure()
            plt.imshow(np.real(data))
            plt.title(f'updated state {oo}, site({left_site_pos}, {nsites})')
            plt.colorbar()
            plt.show()

        return data

    def check_projected_state(self, out_comp, left_site_pos: int, nsites: int):
        """ check the projection of Ax onto the output basis.
        """
        ideal_data = []
        proj_data = []
        copy_oo = self.kets[out_comp]
        for (oo, ii), ops in self.operators.items():
            print('ii', ii)
            if oo != out_comp:
                continue

            copy_ii = self.init_ket[ii]
            for op in ops:
                Aket = helper_quimb.apply(op.copy(), copy_ii.copy())
                plt.figure()
                gtn = self.grid.make_gridTN(Aket)
                print('A ket', Aket)
                plt.imshow(np.real(gtn.get_realspace_data()))
                plt.title(f'Ax ideal({oo},{ii})')
                plt.colorbar()

                ideal_data += [gtn.get_realspace_data()]

                bra = copy_oo.copy()
                # bra = bra.conj()
                bra.mangle_inner_(append='_')
                # bra.site_ind_id = bra.site_ind_id + '_'
                bra_tensors = ([bra[i].conj() for i in range(left_site_pos)] +
                               [bra[i].conj() for i in range(left_site_pos + nsites, bra.L)])
                tens_ = qtn.tensor_contract(*bra_tensors, *Aket.tensors)
                tens_.modify(apply=lambda x: x * 10**op.exponent)
                if nsites == 1:
                    m = bra[left_site_pos]
                    m.modify(data=tens_.data, inds=tens_.inds)
                else:
                    m1 = bra[left_site_pos]
                    m2 = bra[left_site_pos + 1]
                    left_inds = tens_.bonds(m1)
                    T1, T2 = qtn.tensor_split(tens_, left_inds, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                    T1.transpose_like(m1, inplace=True)
                    T2.transpose_like(m2, inplace=True)
                    m1.modify(data=T1.data)
                    m2.modify(data=T2.data)

                plt.figure()
                gtn = self.grid.make_gridTN(bra)
                plt.imshow(np.real(gtn.get_realspace_data()))
                plt.title(f'Ideal Proj Ax({oo},{ii}), site({left_site_pos}, {nsites})')
                plt.colorbar()
                plt.show()

                proj_data += [gtn.get_realspace_data()]

        sources = self.sources.get(out_comp, None)
        source_data = np.zeros_like(proj_data[0])
        for source in sources:
            gtn = self.grid.make_gridTN(source)
            source_data += gtn.get_realspace_data()

        if len(proj_data) >= 2:
            plt.figure()
            plt.imshow(np.real(proj_data[0] + proj_data[1] + source_data))
            plt.title('real proj diff')
            plt.colorbar()

            plt.figure()
            plt.imshow(np.imag(proj_data[0] + proj_data[1] + source_data))
            plt.title('imag proj diff')
            plt.colorbar()

            plt.figure()
            plt.imshow(np.real(ideal_data[0] + ideal_data[1] + source_data))
            plt.title('real ideal diff')
            plt.colorbar()

            plt.figure()
            plt.imshow(np.imag(ideal_data[0] + ideal_data[1] + source_data))
            plt.title('imag ideal diff')
            plt.colorbar()

            plt.show()
        return

    def check_projected_source(self, out_comp, left_site_pos: int, nsites: int):

        sources = self.sources.get(out_comp, None)
        if sources is not None:
            for source in sources:
                bra = self.kets[out_comp].copy()
                # bra = bra.conj()
                bra.mangle_inner_(append='_')
                # bra.site_ind_id = bra.site_ind_id + '_'
                bra_tensors = ([bra[i].conj() for i in range(left_site_pos)] +
                               [bra[i].conj() for i in range(left_site_pos + nsites, bra.L)])
                tens_ = qtn.tensor_contract(*bra_tensors, *source.tensors)
                tens_.modify(apply=lambda x: x * 10**source.exponent)
                print('source', source.exponent)

                if nsites == 1:
                    m = bra[left_site_pos]
                    m.modify(data=tens_.data, inds=tens_.inds)
                else:
                    m1 = bra[left_site_pos]
                    m2 = bra[left_site_pos + 1]
                    left_inds = tens_.bonds(m1)
                    T1, T2 = qtn.tensor_split(tens_, left_inds, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                    T1.transpose_like(m1, inplace=True)
                    T2.transpose_like(m2, inplace=True)
                    m1.modify(data=T1.data)
                    m2.modify(data=T2.data)

                plt.figure()
                gtn = self.grid.make_gridTN(bra)
                plt.imshow(np.real(gtn.get_realspace_data()))
                plt.title(f'source b({out_comp}), site({left_site_pos}, {nsites})')
                plt.colorbar()

                bra = source
                plt.figure()
                gtn = self.grid.make_gridTN(bra)
                plt.imshow(np.real(gtn.get_realspace_data()))
                plt.title(f'ideal b({out_comp})')
                plt.colorbar()

            plt.show()

        return

    ######

    # def canonize(self, i, cur_orthog=None):
    #     ### relies on fact that canonicalization is done in the td-DMRG site solve ??? does it?
    #
    #     ## update current states and bras
    #     for ik in self.states_dict:
    #         print('ik', ik, i, self.states_dict[ik] is None)
    #         out_mps = helper.canonize(self.states_dict[ik], i=i, cur_orthog=cur_orthog, scale=False,
    #                                   bra=self.current_bras[ik])
    #
    #         if out_mps is None:     ## o.w. is inplace
    #             self.states_dict[ik] = None
    #             self.current_bras[ik] = None
    #
    #     ## update state and bra in each solver
    #     for k, solver in self.solvers.items():
    #         oo, ii = k
    #         bra_o = self.current_bras[oo]
    #         ket_i = self.states_dict[ii]
    #
    #         if bra_o is None:
    #             solver.bra = None
    #         else:
    #             for s in range(solver.L):
    #                 solver.bra[s].transpose_like(bra_o[s], inplace=True)
    #                 solver.bra[s].modify(data=bra_o[s].data)
    #
    #         if ket_i is None:
    #             solver.ket = None
    #         else:
    #             for s in range(solver.L):
    #                 solver.ket[s].transpose_like(ket_i[s], inplace=True)
    #                 solver.ket[s].modify(data=ket_i[s].data)
    #
    #     #     if solver.ket is not None:
    #     #         helper.check_orthog(solver.ket)
    #     #     if solver.bra is not None:
    #     #         helper.check_orthog(solver.bra)
    #     #
    #     # exit()
    #
    #     return

    def _update_1site(self, i: int, site_i_dict: dict[int, Union[Sequence['qtn.Tensor'], 'qtn.Tensor']],
                      direction: 'SweepDirection'):

        at_end = (i == self.L - 1) if direction > 0 else (i == 0)

        ## update self.current_state
        for ii, ket in self.states_dict.items():
            try:
                new_ket_tens = site_i_dict[ii] # .copy()
                helper_dmrg.update_1site(ket, i, new_ket_tens, direction, max_bond=self.max_bond, cutoff=self.cutoff)

            except KeyError:
                pass

        if not at_end:
            for term in self.terms:
                term.extend_env(i, direction=direction)

        return

    def _update_2site(self, left_site_pos: int, site_i_dict: dict[int, Union[Sequence['qtn.Tensor'], 'qtn.Tensor']],
                      direction: 'SweepDirection'):

        # print('update 2 site', left_site_pos)
        at_end = (left_site_pos == self.L - 2) if direction > 0 else (left_site_pos == 0)
        print('at end TD-DMRG', left_site_pos, direction, at_end)


        ## update self.current_state
        for ii, ket in self.states_dict.items():
            try:
                new_ket_tens = site_i_dict[ii]  # .copy()
                # print('tens norm', i , ii, [ten.norm() for ten in new_ket_tens])
                helper_dmrg.update_2site(ket, left_site_pos, new_ket_tens, direction, max_bond=self.max_bond,
                                         cutoff=self.cutoff, decimate_only=(not at_end))

            except KeyError:
                pass

        if not at_end:

            for term in self.terms:

                # for ii, ket in self.states_dict.items():
                #     print('term', ii, term.bra is ket, term.ket is ket)
                #     print('vec block', ii, term.vec_block.bra is ket, term.vec_block.ket is ket)
                #     if len(term.op_blocks) > 0:
                #         for ops in term.op_blocks[0]:
                #             print('op block', ii, ops.bra is ket, ops.ket is ket)
                #     pdb.set_trace()

                i = left_site_pos if direction > 0 else left_site_pos + 1
                term.extend_env(i, direction=direction)

        return

    ####

    @classmethod
    def compute_Ax(cls, A_effs_dict_: dict[tuple[int, int], Sequence['qtn.TensorNetwork']], x_vec_tens_: 'qtn.Tensor',
                   ref_x_dict: dict[int, 'qtn.Tensor'] = None, ref_b_eff_dict: dict[int, 'qtn.Tensor'] = None,
                   out_keys: Sequence = (), bra_inds: Sequence = (), out_shape: Sequence = (),
                   in_keys: Sequence = (), ket_inds: Sequence = (), in_shape: Sequence = (),
                   out_inds=('out_ind',), **kwargs) -> 'qtn.Tensor':

        Ax_dict = {}
        input_is_ket = out_inds == (cls.combined_ind_o,)
        # print('input is ket', input_is_ket)
        if input_is_ket:
            out_keys_ = out_keys  # b key ordering
            bra_inds_ = bra_inds  # out = output inds
            x_dict_ = extract_vec_tens(x_vec_tens_.data, in_keys, in_shape, ref_x_dict)
            ## tens have ket_inds

            # print('ref b eff', ref_b_eff_dict)
            for ix_, ref_tens in ref_b_eff_dict.items():
                Ax_dict[ix_] = qtn.Tensor(np.zeros(ref_tens.shape), inds=ref_tens.inds)

        else:
            out_keys_ = in_keys  #  out_keys  # x key ordering # out_keys  # b key ordering
            bra_inds_ = ket_inds  # out = input inds
            x_dict_ = extract_vec_tens(x_vec_tens_.data, out_keys, out_shape, ref_b_eff_dict)
            ## tens have bra_inds

            # print('ref x eff', ref_b_eff_dict)
            for ix_, ref_tens in ref_x_dict.items():
                Ax_dict[ix_] = qtn.Tensor(np.zeros(ref_tens.shape), inds=ref_tens.inds)

        for (ooo, iii), A_eff_tns in A_effs_dict_.items():
            tot_A_eff = None
            ix_, ix_c = (ooo, iii) if input_is_ket else (iii, ooo)  ## out_index, contracted index
            # print('k', ix_, ix_c)
            ## sum over A_eff|x> for A_eff in list of A_eff_tns
            for A_eff_tn in A_eff_tns:
                if ix_c not in x_dict_:
                    continue
                Ax_tens = qtn.tensor_contract(*A_eff_tn.tensors, x_dict_[ix_c])
                Ax_tens.modify(apply=lambda x: x * 10 ** A_eff_tn.exponent)
                if tot_A_eff is None:
                    tot_A_eff = Ax_tens.transpose(*bra_inds_)
                else:
                    Ax_tens.transpose_like(tot_A_eff, inplace=True)
                    tot_A_eff.modify(apply=lambda x: x + Ax_tens.data)

            ### sum over ix_c
            if tot_A_eff is not None:
                if ix_ not in Ax_dict:
                    Ax_dict[ix_] = tot_A_eff
                else:
                    tot_A_eff.transpose_like(Ax_dict[ix_], inplace=True)
                    Ax_dict[ix_].modify(apply=lambda x: x + tot_A_eff.data)

        Ax_vec = combine_vec_tens(Ax_dict, keys=out_keys_)
        if Ax_vec is None:  ## for compatibility with constraints
            Ax_vec = combine_vec_tens(Ax_dict) # , keys=(ix_,))
        Ax_vec_tens = qtn.Tensor(data=Ax_vec, inds=out_inds)
        return Ax_vec_tens

    @classmethod
    def compute_xAx(cls, A_effs_dict_: dict[tuple[int, int], Sequence['qtn.TensorNetwork']],
                    x_vec_tens_i: 'qtn.Tensor', x_vec_tens_o: 'qtn.Tensor',
                    ref_x_dict: dict[int, 'qtn.Tensor'] = None, ref_b_eff_dict: dict[int, 'qtn.Tensor'] = None,
                    out_keys: Sequence = (), in_keys: Sequence = (), in_shape: Sequence = (),
                    precomp_Ax=None, **kwargs) -> Numeric:

        x_dict_i = extract_vec_tens(x_vec_tens_i.data, in_keys, in_shape, ref_x_dict)  # in inds
        x_dict_o = extract_vec_tens(x_vec_tens_o.data, in_keys, in_shape, ref_b_eff_dict)  # out inds
        tot_xAx = 0.0
        if precomp_Ax is None:
            for (ooo, iii), A_eff_tns in A_effs_dict_.items():
                for A_eff_tn in A_eff_tns:
                    out = qtn.tensor_contract(*A_eff_tn.tensors, x_dict_i[iii], x_dict_o[ooo])
                    out *= 10 ** A_eff_tn.exponent
                    tot_xAx += out
        else:
            precomp_Ax_dict = extract_vec_tens(precomp_Ax.data, in_keys, in_shape, ref_b_eff_dict)  # out inds
            for ooo in out_keys:
                out = qtn.tensor_contract(precomp_Ax_dict[ooo], x_dict_o[ooo])
                tot_xAx += out

        return tot_xAx


    def _get_x_eff_dict_(self, left_site_pos: int, nsites: int, site_tens_dict=None):
        """
        BL * B * BR = d/dT*[s] <x_j|b_i> d_ij
        dict indexes xj
        """
        # ket_site_inds = [self.ket.site_ind_id.format(si) for si in site_inds]
        # bra_site_inds = [self.bra.site_ind_id.format(si) for si in site_inds]

        x_eff_dict = {}
        site_tens_dict = site_tens_dict if site_tens_dict is not None else {}

        ### BL * B * BR = d/dT*[i] <x|b>
        for oo, term in self.self_terms.items():

            x_effs = [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens_dict.get(oo, None))]

            if len(x_effs) == 0:  continue

            x_eff = helper_tn.sum_tens(x_effs)
            x_eff_dict[oo] = x_eff

        return x_eff_dict


    def _get_b_eff_dict_(self, left_site_pos: int, nsites: int): #, site_tens_dict=None):
        """
        BL * B * BR = d/dT*[s] <x_j|b_i> d_ij
        dict indexes xj
        """
        # ket_site_inds = [self.ket.site_ind_id.format(si) for si in site_inds]
        # bra_site_inds = [self.bra.site_ind_id.format(si) for si in site_inds]

        b_eff_dict = {}
        # site_tens_dict = site_tens_dict if site_tens_dict is not None else {}

        ### BL * B * BR = d/dT*[i] <x|b>
        for oo, terms in self.source_terms.items():

            b_effs = [term.get_evaluated_site(left_site_pos, nsites) #, site_tens=site_tens_dict.get(oo, None))
                      for term in terms]

            if len(b_effs) == 0:  continue

            b_eff = helper_tn.sum_tens(b_effs)
            b_eff_dict[oo] = b_eff

        return b_eff_dict


    def _get_Ax_eff_dict_(self, left_site_pos: int, nsites: int, site_tens_dict=None, negative_dt=False,
                          sum_terms=True, plot_verbose=False):
        """
        AL * A @ x * AR = d/dT*[s] <x_j|A|x_i>
        dict indexes xj
        """

        Ax_eff_dict = {}
        site_tens_dict = site_tens_dict if site_tens_dict is not None else {}

        ### BL * B * BR = d/dT*[i] <x|b>
        for (oo, ii), terms in self.op_terms.items():

            # ket_site_inds = [self.ket.site_ind_id.format(si) for si in range(left_site_pos, left_site_pos + nsites)]
            # bra_site_inds = [self.bra.site_ind_id.format(si) for si in range(left_site_pos, left_site_pos + nsites)]
            b2k_dict = self.get_bra_to_ket_inds(left_site_pos, nsites, comp=oo)

            # print('site tens dict', site_tens_dict)
            # for term in terms:
            #     print('Ax check orthog!')
            #     term.check_orthog()
            #     # helper_quimb.check_orthog(term.ket)
            #     # helper_quimb.check_orthog(term.bra)
            #     ## not the same bc env might not be normalized to 1? why? maybe bc of scalar multiply?

            # for op in self.operators[(oo,ii)]:
            #     print('op exponent', op.exponent)
            #
            # print('state exponent', self.states_dict[oo].exponent)
            Ax_effs = [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens_dict.get(ii, None))
                       for term in terms]
            ## for backward time integration
            if negative_dt and self.backward_weights is not None:
                weights = self.backward_weights.get((oo, ii), [])
                print('negative dt weights', ii, oo, weights)
                for coeff, Ax_eff in zip(weights, Ax_effs):
                    Ax_eff.modify(apply=lambda x: x * coeff)

            ## apply mask d/dt x = c * A * x
            c_terms = self.mask_terms.get(oo, None)
            if c_terms is not None:
                ## c * Ax_effs
                cAx_effs = []
                for c_term in c_terms:
                    for tens_ in Ax_effs:
                        cAx_effs = [c_term.get_evaluated_site(left_site_pos, nsites, tens_)]
                Ax_effs = cAx_effs

            for Ax_eff in Ax_effs:
                Ax_eff.reindex(b2k_dict, inplace=True)

            # plot_verbose = self.verbose
            if oo == 5 and plot_verbose: # and left_site_pos < 3: # and site_tens_dict is None:
                copy_oo = self.states_dict[oo].copy()
                copy_ii = self.states_dict[ii].copy()
                print('Axeff left site pos', left_site_pos, 'nsites', nsites, 'num terms', len(Ax_effs))
                for ix, new_site in enumerate(Ax_effs):
                    npts_z = self.grid.axes[2].npts
                    term = terms[ix]
                    exact_op = helper_quimb.apply(term.operators[0], copy_ii)
                    # print('exact op', exact_op)
                    plt.figure()
                    gtn = self.grid.get_ones_mps()
                    self.grid.dmrg_to_gtn_format(gtn, exact_op)
                    plt.imshow(np.real(gtn.get_realspace_data()[:,:,npts_z//2]))
                    plt.title(f'global oo,ii {oo},{ii}; {left_site_pos},{nsites}')
                    plt.colorbar()

                    if nsites == 2:
                        if left_site_pos > 0:
                            left_inds = [copy_oo.bond(left_site_pos, left_site_pos - 1), copy_oo.site_ind(left_site_pos)]
                        else:
                            left_inds = [copy_oo.site_ind(left_site_pos)]
                        print('new site', new_site.norm())
                        if new_site.norm() > 0:
                            T1, T2 = qtn.tensor_split(new_site, left_inds = left_inds, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                        else:
                            print('copy oo tensor', left_site_pos, copy_oo[left_site_pos].norm())
                            # T1, _ = qtn.tensor_split(copy_oo[left_site_pos], left_inds=left_inds, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                            T1 = copy_oo[left_site_pos].copy()
                            T2 = copy_oo[left_site_pos + 1].copy()
                            T2.modify(apply=lambda x: x * 0)
                        T1.transpose_like(copy_oo[left_site_pos], inplace=True)
                        T2.transpose_like(copy_oo[left_site_pos + 1], inplace=True)
                        copy_oo[left_site_pos].modify(data=T1.data)
                        copy_oo[left_site_pos + 1].modify(data=T2.data)
                    elif nsites == 1:
                        new_site.transpose_like(copy_oo[left_site_pos], inplace=True)
                        copy_oo[left_site_pos].modify(data=new_site.data)

                    plt.figure()
                    gtn = self.grid.get_ones_mps()
                    self.grid.dmrg_to_gtn_format(gtn, copy_oo)
                    plt.imshow(np.real(gtn.get_realspace_data())[:,:,npts_z//2])
                    plt.title(f'oo,ii {oo},{ii}; {left_site_pos},{nsites}')
                    plt.colorbar()
                    # plt.show()

                for it, eff_Ax in enumerate(Ax_effs):
                    # ket = self.states_dict[ii].copy()
                    ket = terms[it].ket.copy()
                    print('term bra', terms[it].bra is self.states_dict[oo])
                    print('term ket', terms[it].ket is self.states_dict[ii])


                    plt.figure()
                    gtn = self.grid.make_gridTN(ket.copy())
                    plt.imshow(np.real(gtn.get_realspace_data())[:,:,npts_z//2])
                    plt.title(f'x {ii}')
                    plt.colorbar()

                    Aket = helper_quimb.apply(terms[it].operators[0].copy(), ket.copy())
                    plt.figure()
                    gtn = self.grid.make_gridTN(Aket)
                    plt.imshow(np.real(gtn.get_realspace_data())[:,:,npts_z//2])
                    plt.title(f' Ax ideal')
                    plt.colorbar()

                    ## project ideal output
                    # bra = self.states_dict[oo].copy()
                    bra = terms[it].bra.copy()
                    # bra = bra.conj()
                    bra.mangle_inner_(append='_')
                    # bra.site_ind_id = bra.site_ind_id + '_'
                    bra_tensors = ([bra[i].conj() for i in range(left_site_pos)] +
                                   [bra[i].conj() for i in range(left_site_pos + nsites, bra.L)])
                    tens_ = qtn.tensor_contract(*bra_tensors, *Aket.tensors)
                    if nsites == 1:
                        m = bra[left_site_pos]
                        m.modify(data=tens_.data, inds=tens_.inds)
                    else:
                        m1 = bra[left_site_pos]
                        m2 = bra[left_site_pos + 1]
                        left_inds = tens_.bonds(m1)
                        if tens_.norm() > 0:
                            T1, T2 = qtn.tensor_split(tens_, left_inds, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                        else:
                            T1 = m1.copy()
                            T2 = m2.copy()
                            T2.modify(apply=lambda x: x * 0)
                        T1.transpose_like(m1, inplace=True)
                        T2.transpose_like(m2, inplace=True)
                        m1.modify(data=T1.data)
                        m2.modify(data=T2.data)

                    plt.figure()
                    gtn = self.grid.make_gridTN(bra)
                    plt.imshow(np.real(gtn.get_realspace_data())[:,:,npts_z//2])
                    plt.title(f'Ideal Proj Ax({oo},{ii})')
                    plt.colorbar()

                    bra2 = bra.copy()
                    bra2.site_ind_id = bra.site_ind_id + '_'
                    bra_tensors = [bra2[i] for i in range(left_site_pos)] + [bra2[i] for i in range(left_site_pos + nsites, bra.L)]
                    ket_tensors = [ket[i] for i in range(left_site_pos)] + [ket[i] for i in range(left_site_pos + nsites, ket.L)]
                    eff_op = qtn.TensorNetwork([*bra_tensors, *ket_tensors, terms[it].operators[0].tensors])
                    eff_op_tens = eff_op.contract()
                    ket_inds = self.ket_inds(left_site_pos, nsites)
                    bra_inds = self.bra_inds(left_site_pos, nsites)
                    eff_op_tens = eff_op_tens.transpose(*bra_inds, *ket_inds)
                    eff_op_tens.modify(apply=lambda x: x * 10**(terms[it].operators[0].exponent))
                    # print('eff op tens', eff_op_tens.data, eff_op_tens.inds)

                    # for ix in range(9, self.L-1):
                    #     print('i', ix)
                    #     bra_tensors = [bra2[i] for i in range(ix + 1, bra.L)]
                    #     ket_tensors = [ket[i] for i in range(ix + 1, ket.L)]
                    #     op_tensors = [terms[it].operators[0][i] for i in range(ix + 1, ket.L)]
                    #     env_r = qtn.TensorNetwork([*bra_tensors, *ket_tensors, *op_tensors])
                    #     env_r_tens = env_r.contract()
                    #     print('evn r (2)', env_r_tens.data, env_r_tens.inds, env_r.shape)

                    ket = self.states_dict[oo].copy()
                    eff_Ax = eff_Ax.reindex({bi: ki for bi, ki in zip(bra_inds, ket_inds)}, inplace=False)
                    if nsites == 2:
                        left_inds = eff_Ax.bonds(ket[left_site_pos])
                        T1, T2 = qtn.tensor_split(eff_Ax, left_inds,)
                        m1 = ket[left_site_pos]
                        m2 = ket[left_site_pos + 1]
                        T1.transpose_like(m1, inplace=True)
                        m1.modify(data=T1.data)
                        T2.transpose_like(m2, inplace=True)
                        m2.modify(data=T2.data)
                    else:
                        m = ket[left_site_pos]
                        eff_Ax.transpose_like(m, inplace=True)
                        m.modify(data=eff_Ax.data) #, inds=eff_Ax.inds)

                    print('left site pos', left_site_pos)
                    print('state', oo, ii, self.states_dict[oo][left_site_pos])
                    if nsites == 2:
                        print('state', oo, ii, self.states_dict[oo][left_site_pos + 1])

                    plt.figure()
                    gtn = self.grid.make_gridTN(ket)
                    plt.imshow(np.real(gtn.get_realspace_data())[:,:,npts_z//2])
                    plt.title(f'Proj Ax({oo},{ii})')
                    plt.colorbar()
                    plt.show()

                    print('end verbose plot')

            if oo not in Ax_eff_dict:
                Ax_eff_dict[oo] = Ax_effs
            else:
                Ax_eff_dict[oo] += Ax_effs

        if sum_terms:
            for k in Ax_eff_dict.keys():
                Ax_eff_dict[k] = [helper_tn.sum_tens(Ax_eff_dict[k])]

        return Ax_eff_dict

    def _get_A_eff_dict_(self, left_site_pos: int, nsites: int, negative_dt=False):
        """
        AL * A @ x * AR = d/dT*[s] d/dT[s] <x_j|A|x_i>
        dict indexes xj
        """
        A_eff_dict = {}
        # site_tens_dict = site_tens_dict if site_tens_dict is not None else {}

        ### BL * B * BR = d/dT*[i] <x|b>
        for (oo, ii), terms in self.op_terms.items():

            A_eff_dict[(oo, ii)] = []
            for term in terms:
                eff_ops = term.get_eff_operator(left_site_pos, nsites)[0]

                if negative_dt and self.backward_weights is not None:
                    weights = self.backward_weights.get((oo, ii), [])
                    for coeff, eff_op in zip(weights, eff_ops):
                        eff_op.tensors[0].modify(apply=lambda x: x * coeff)

                A_eff_dict[(oo, ii)] += eff_ops
                ## assume only 1 tier

        return A_eff_dict


    def cgd(self, left_site_pos: int, nsites: int, x_dict: dict[int, qtn.Tensor],
            A_effs_dict: dict[tuple[int,int], Sequence[qtn.TensorNetwork]],
            b_eff_dict: dict[int, qtn.Tensor], is_H=False,
            constraint_mats_dict=None, constraint_vals_dict=None, **kwargs):

        ## combine As, bs
        start = left_site_pos
        end = left_site_pos + nsites
        site_inds = range(start, end)

        ket_inds = self.ket_inds(left_site_pos, nsites)
        bra_inds = self.bra_inds(left_site_pos, nsites, ket_inds=ket_inds)
        ket_to_bra_inds = {self.component_ind: self.component_ind + '_'}
        ket_to_bra_inds.update(self.get_ket_to_bra_inds(left_site_pos, nsites))
        bra_to_ket_inds = {v: k for k, v in ket_to_bra_inds.items()}
        bra_inds = [ket_to_bra_inds[k] for k in ket_inds]

        #################################################

        if len(A_effs_dict) == 0:
            new_x_dict = {}
            for ik in b_eff_dict.keys():
                x_eff = b_eff_dict[ik].reindex(bra_to_ket_inds)
                # x_eff.modify(apply=lambda x: x * 10 ** (-self.kets[ik].exponent))
                new_x_dict[ik] = x_eff
        else:
            in_keys = list(x_dict.keys())
            out_keys = list(b_eff_dict.keys())
            in_keys.sort(), out_keys.sort()
            in_shape = [x_dict[k].size for k in in_keys]
            out_shape = [b_eff_dict[k].size for k in out_keys]

            for k, tens in x_dict.items():
                tens.transpose(*ket_inds, inplace=True)

            beff_dict_scaled = {}
            for k, tens in b_eff_dict.items():
                tens_ = tens.copy()
                if k in self.states_dict:
                    tens_.modify(apply=lambda x: x * 10 ** self.states_dict[k].exponent)
                tens_.transpose(*ket_inds, inplace=True)
                tens_.reindex({ki: bi for ki, bi in zip(ket_inds, bra_inds)}, inplace=True)
                beff_dict_scaled[k] = tens_
            b_eff_dict = beff_dict_scaled

            # print('in keys', in_keys, out_keys)
            x_vec = combine_vec_tens(x_dict, keys=in_keys)
            b_vec = combine_vec_tens(b_eff_dict, keys=out_keys)

            combined_ind_i, combined_ind_o = self.__class__.combined_ind_i, self.__class__.combined_ind_o
            x_vec_tens = qtn.Tensor(data=x_vec, inds=(combined_ind_i,))
            b_vec_tens = qtn.Tensor(data=b_vec, inds=(combined_ind_o,))

            def local_compute_Ax(A_effs_dict_: dict[tuple[int, int], Sequence['qtn.TensorNetwork']],
                                 x_vec_tens_: 'qtn.Tensor', out_inds,
                                 out_keys_=None, bra_inds_=None, out_shape_=None,
                                 in_keys_=None, ket_inds_=None, in_shape_=None,
                                 ref_x_dict=None, ref_b_eff_dict=None,
                                 **kwargs) -> 'qtn.Tensor':

                if ref_x_dict is None:  ref_x_dict = x_dict
                if ref_b_eff_dict is None:  ref_b_eff_dict = b_eff_dict
                if out_keys_ is None:   out_keys_ = out_keys
                if bra_inds_ is None:   bra_inds_ = bra_inds
                if out_shape_ is None:  out_shape_ = out_shape
                if in_keys_ is None:    in_keys_ = in_keys
                if ket_inds_ is None:   ket_inds_ = ket_inds
                if in_shape_ is None:   in_shape_ = in_shape

                return self.__class__.compute_Ax(A_effs_dict_, x_vec_tens_,
                                                 ref_x_dict=ref_x_dict, ref_b_eff_dict=ref_b_eff_dict,
                                                 out_keys=out_keys_, bra_inds=bra_inds_, out_shape=out_shape_,
                                                 in_keys=in_keys_, ket_inds=ket_inds_, in_shape=in_shape_,
                                                 out_inds=out_inds, **kwargs)

            def local_compute_xAx(A_effs_dict_: dict[tuple[int, int], Sequence['qtn.TensorNetwork']],
                                  x_vec_tens_i: 'qtn.Tensor', x_vec_tens_o: 'qtn.Tensor', out_inds,
                                  precomp_Ax=None, **kwargs) -> 'Numeric':

                return self.__class__.compute_xAx(A_effs_dict_, x_vec_tens_i, x_vec_tens_o,
                                                  ref_x_dict=x_dict, ref_b_eff_dict=b_eff_dict,
                                                  out_keys=out_keys, bra_inds=bra_inds, out_shape=out_shape,
                                                  in_keys=in_keys, ket_inds=ket_inds, in_shape=in_shape,
                                                  out_inds=out_inds, precomp_Ax=precomp_Ax, **kwargs)

            bra_to_ket_inds_tmp = {**bra_to_ket_inds, combined_ind_o: combined_ind_i}

            if is_H:
                x_eff, error = qtn_conjugate_gradient_descent_1site(A_effs_dict, b_vec_tens, x_vec_tens,
                                                                    bra_to_ket_inds_tmp,
                                                                    conv_tol=np.sqrt(self.cutoff) / 100,
                                                                    contract_Ax=local_compute_Ax,
                                                                    contract_xAx=local_compute_xAx)
            else:
                x_eff, error = qtn_conjugate_gradient_squared_1site(A_effs_dict, b_vec_tens, x_vec_tens,
                                                                    bra_to_ket_inds_tmp,
                                                                    conv_tol=np.sqrt(self.cutoff) / 100,
                                                                    contract_Ax=local_compute_Ax,
                                                                    contract_xAx=local_compute_xAx)
            new_x_dict = extract_vec_tens(x_eff.data, in_keys, in_shape, x_dict)
            print('cgd error', error)

            out_dict_unscaled = {}
            for k, tens in new_x_dict.items():
                tens_ = tens.copy()
                if k in self.states_dict:
                    tens_.modify(apply=lambda x: x * 10 ** (-self.states_dict[k].exponent))
                out_dict_unscaled[k] = tens_
            new_x_dict = out_dict_unscaled

        return new_x_dict


    def lsq(self, left_site_pos: int, nsites: int, x_dict: dict[int,'qtn.Tensor'],
            A_effs_dict: dict[tuple[int,int],'qtn.Tensor'],
            b_eff_dict: dict[int,'qtn.Tensor'],
            constraint_mats_dict=None, constraint_vals_dict=None,
            **kwargs):
        """ solve A x = b
        """
        ## combine As, bs
        start = left_site_pos
        end = left_site_pos + nsites
        site_pos = range(start, end)

        ket_inds = self.ket_inds(left_site_pos, nsites)
        ket_to_bra_inds = {self.component_ind: self.component_ind + '_'}
        ket_to_bra_inds.update(self.get_ket_to_bra_inds(left_site_pos, nsites))
        bra_to_ket_inds = {v: k for k, v in ket_to_bra_inds}
        bra_inds = [ket_to_bra_inds[k] for k in ket_inds]

        ########################################

        for k, A_eff in A_effs_dict.items():
            A_eff.transpose(*bra_inds, *ket_inds, inplace=True)

        for k, b_eff in b_eff_dict.items():
            b_eff.transpose(*ket_inds, inplace=True)
            # b_eff.reindex({k: b for k, b in zip(ket_inds, bra_inds)}, inplace=True)

        for k, x_eff in x_dict.items():
            x_eff.transpose(*ket_inds, inplace=True)

        ## solve for T[i]
        new_x_dict = {}
        if len(A_effs_dict) == 0:
            for ik in b_eff_dict.keys():
                x_eff = b_eff_dict[ik].reindex(bra_to_ket_inds)
                ## todo: need to check this
                k_exp = -self.states_dict[ik].exponent
                x_eff.modify(apply=lambda x: x * 10 ** (-k_exp))
                new_x_dict[ik] = x_eff
        else:

            in_keys = list(x_dict.keys())
            out_keys = list(b_eff_dict.keys())
            in_keys.sort(), out_keys.sort()
            in_shape = [x_dict[k].size for k in in_keys]
            out_shape = [b_eff_dict[k].size for k in out_keys]

            # inds_i_list = x_dict[in_keys[0]].inds
            # ket_to_bra = {}
            # for i in sites:
            #     ket_to_bra.update(self.get_ket_to_bra_inds(i))
            # inds_o_list = [ket_to_bra[k] for k in inds_i_list]

            beff_dict_scaled = {}
            for k, tens in b_eff_dict.items():
                tens_ = tens.copy()
                if k in self.states_dict:
                    tens_.modify(apply=lambda x: x * 10 ** self.states_dict[k].exponent)
                tens_.transpose(*ket_inds, inplace=True)
                beff_dict_scaled[k] = tens_

            # x_vec = combine_vec_tens(xeff_dict_scaled, keys=in_keys)
            b_vec = combine_vec_tens(beff_dict_scaled, keys=out_keys)

            # print('in keys', in_keys, 'out keys', out_keys)
            # print('in shape', in_shape, 'out shape', out_shape)
            A_mat = combine_mat_tens(A_effs_dict, out_keys, in_keys, out_shape, in_shape)
            # tmp = A_mat - np.eye(len(A_mat))
            # print('A - I is AH?', np.linalg.norm(tmp + tmp.conj().T))

            x_eff_data = np.linalg.solve(A_mat, b_vec)
            new_x_dict = extract_vec_tens(x_eff_data, in_keys, in_shape, beff_dict_scaled)

            ############################

            out_dict_unscaled = {}
            for k, tens in new_x_dict.items():
                tens_ = tens.copy()
                tens_.modify(apply=lambda x: x * 10 ** (-self.states_dict[k].exponent))
                out_dict_unscaled[k] = tens_
            new_x_dict = out_dict_unscaled

        return new_x_dict

    def local_exact(self, dt: Numeric, left_site_pos: int, nsites: int,
                    bonds_i: Sequence[str], bonds_o: Sequence[str],
                    site_tens_dict=None,
                    te_order=4,
                    return_intermediates=False) -> dict[int, Sequence['qtn.Tensor']]:

        beff_dict = self._get_b_eff_dict_(left_site_pos, nsites, site_tens_dict=site_tens_dict)
        Aeff_dict = self._get_A_eff_dict_(left_site_pos, nsites, negative_dt=(dt < 0))
        xeff_dict = self._get_x_eff_dict_(left_site_pos, nsites, site_tens_dict=site_tens_dict)

        if dt == 0:
            return xeff_dict

        bra_to_ket_inds = self.get_bra_to_ket_inds(left_site_pos, nsites)
        ket_to_bra_inds = {v: k for k, v in bra_to_ket_inds.items()}

        if self.masks is not None:
            raise NotImplementedError

        def exact_func(xeff_dict_: dict[int, 'qtn.Tensor'], dt, **kwargs) -> dict[int, 'qtn.Tensor']:

            # assert (beff_dict is None or len(beff_dict)==0), 'exact TE for df/dt = Af + b is not defined'

            keys_list = list(xeff_dict_.keys())
            npts_list = [xeff_dict_[k].size for k in keys_list]

            inds_i_list = xeff_dict_[keys_list[0]].inds
            inds_o_list = [ket_to_bra_inds[k] for k in inds_i_list]

            beff_dict_unscaled = beff_dict.copy()
            for k, tens in beff_dict_unscaled.items():
                tens.transpose(*inds_o_list, inplace=True)
                tens_ = tens.copy()
                tens_.modify(apply=lambda x: x * 10 ** self.states_dict[k].exponent)
                beff_dict_unscaled[k] = tens_

            xeff_dict_unscaled = {}
            for k, tens in xeff_dict_.items():
                tens_ = tens.copy()
                tens_.modify(apply=lambda x: x * 10 ** self.states_dict[k].exponent)
                tens_.transpose(*inds_i_list, inplace=True)
                xeff_dict_unscaled[k] = tens_

            if len(beff_dict_unscaled) > 0:
                for k, tens_ in xeff_dict_.items():
                    if k not in beff_dict_unscaled:
                        beff_dict_unscaled[k] = qtn.Tensor(np.zeros(tens_.shape), inds=tens_.inds)

            x_vec = combine_vec_tens(xeff_dict_unscaled, keys=keys_list)
            b_vec = combine_vec_tens(beff_dict_unscaled, keys=keys_list)  # scaled with x exponent

            # deriv_dict = deriv_func(xeff_dict_)
            Aeff_dict_tn = {k: helper_tn.sum_eff_TNs(Aeff, transpose_bonds=[*inds_o_list, *inds_i_list])
                            for k, Aeff in Aeff_dict.items()}
            mat = combine_mat_tens(Aeff_dict_tn, keys_o=keys_list, keys_i=keys_list,
                                   shapes_o=npts_list, shapes_i=npts_list)

            if b_vec is None:
                # print('HERE!!!!! EXACT WITH NO SOURCE')
                expmat = scipy.linalg.expm(mat * dt)
                out = np.dot(expmat, x_vec)
            else:
                x_size, = x_vec.shape
                evals, evecs = np.linalg.eig(mat)
                inv_evecs = np.linalg.inv(evecs)
                exp_At = (evecs * np.exp(dt * evals)) @ inv_evecs
                # print('evals', evals)
                # exp_At = scipy.linalg.expm(mat)

                # print('inv?', np.linalg.norm(evecs @ inv_evecs - np.eye(x_size)))
                # print('inv?', np.linalg.norm(inv_evecs @ evecs - np.eye(x_size)))

                ## phi_1(A) = (exp(A) - 1)/A = V (exp(D) - 1)/D V^-1
                ## where (exp(D)-1)/D = 1 where D = 0
                # tmp = (np.exp(evals * dt) - 1) /evals /dt
                phi1_D = np.where(np.abs(evals) > 10e-15 / dt, (np.exp(evals * dt) - 1) / evals / dt, 1.)
                phi1 = (evecs * phi1_D) @ inv_evecs

                ## compute phi @ b_vec
                nl_term = phi1 @ b_vec

                out = np.dot(exp_At, x_vec) + nl_term * dt  # np.dot(phi1, b_vec)

            # out_tens = qtn.Tensor(out, inds=vec.inds)
            out_dict = extract_vec_tens(out, keys_list, npts_list, ref_dict=xeff_dict_)  # _scaled)
            out_dict_scaled = {}
            for k, tens in out_dict.items():
                tens_ = tens.copy()
                tens_.modify(apply=lambda x: x * 10 ** (-self.states_dict[k].exponent))
                out_dict_scaled[k] = tens_

            return out_dict_scaled

        ### perform actual TE
        print('exact te', left_site_pos, nsites)
        out = helper_TE.exact(xeff_dict, dt, exact_func, return_intermediates=return_intermediates)

        return out


    def get_time_evolved_site(self, dt: Numeric, left_site_pos: int, nsites: int,
                              bonds_i: Sequence[str], bonds_o: Sequence[str],
                              site_tens_dict=None,
                              te_order=4,
                              return_intermediates=False) -> dict[int, Sequence['qtn.Tensor']]:
        """ dx/dt = Ax + b
        """

        if te_order == 0:
            return self.local_exact(dt, left_site_pos, nsites, bonds_i, bonds_o, site_tens_dict=site_tens_dict,
                                    te_order=te_order, return_intermediates=return_intermediates)


        beff_dict = self._get_b_eff_dict_(left_site_pos, nsites) ## kets are independent of changeing ket
        Aeff_dict = self._get_A_eff_dict_(left_site_pos, nsites, negative_dt=(dt < 0))
        xeff_dict = self._get_x_eff_dict_(left_site_pos, nsites, site_tens_dict=site_tens_dict)

        # if True: #  nsites == 0:
        #     print('get evolved site', left_site_pos, nsites)
        #     if site_tens_dict is not None:
        #         print('site tens dict', site_tens_dict.keys())
        #     print('beff', beff_dict.keys())
        #     print('Aeff', Aeff_dict.keys())
        #     print('xeff', xeff_dict.keys())

        if dt == 0:
            return xeff_dict

        bra_to_ket_inds = self.get_bra_to_ket_inds(left_site_pos, nsites)
        ket_to_bra_inds = {v: k for k, v in bra_to_ket_inds.items()}

        # ## scale beffs to match exponents of x
        # beff_dict_scaled = {k: beff.copy() for k, beff in beff_dict.items()}     # dictionary of tensors
        # for k, beff in beff_dict_scaled.items():
        #     beff.modify(apply=lambda x: x * 10**(-self.states_dict[k].exponent))


        def deriv_func(site_tens_dict_, **kwargs) -> dict[int,'qtn.Tensor']:

            for k, v in site_tens_dict_.items():
                self.num_evals[k] += v.size

            eff_Ax_dict = self._get_Ax_eff_dict_(left_site_pos, nsites, site_tens_dict=site_tens_dict_,
                                                 negative_dt = (dt < 0))

            for oo, beff in beff_dict.items():
                # beff = beff_dict_scaled.get(oo, None)
                if beff is not None:
                    ## beff = sum(b_eff tensor networks); should already include exponent from bra*source
                    ## need it to match exponent of output y = Ax + b
                    # beff_tn = qtn.TensorNetwork([beff], virtual=False)
                    if oo in eff_Ax_dict:
                        eff_Ax_dict[oo] += [beff]
                    else:
                        eff_Ax_dict[oo] = [beff]

            # for oo, eff_Axs in eff_Ax_dict.items():
            #     b2k_dict = self.get_ket_to_bra_inds(left_site_pos, nsites, oo)
            #     print('b2k', b2k_dict)
            #     for tmp in eff_Axs:
            #         tmp.reindex(b2k_dict, inplace=True)

            # if nsites > 0:
            #     out_comp = 2
            #     tmp_data = 0
            #     for it , tens in enumerate(eff_Ax_dict[out_comp]):
            #         data = self.check_new_tens(out_comp, tens, left_site_pos, nsites, plot=False)
            #         plt.figure()
            #         plt.imshow(np.real(data))
            #         plt.colorbar()
            #         plt.title(f'Ez part {it}, ({left_site_pos}, {nsites}), dt {np.sign(dt)}')
            #
            #         tmp_data = tmp_data + data
            #
            #     plt.figure()
            #     plt.imshow(np.real(tmp_data))
            #     plt.colorbar()
            #     plt.title(f'Ez deriv ({left_site_pos}, {nsites}), dt {np.sign(dt)}')
            #     plt.show()


            out_dict = {}
            tot_norm = 0.
            for oo, eff_Axs in eff_Ax_dict.items():
                out = helper_tn.sum_tens(eff_Axs, transpose_bonds=bonds_i)
                if out is not None:
                    out_dict[oo] = out
                    tot_norm += out.norm() ** 2

            ## old version
            # out_dict = {}
            # tot_norm = 0.
            # for oo, eff_Axs in eff_Ax_dict.items():
            #     # print('eff Axs', [(eff.exponent if eff is not None else None) for eff in eff_Axs])
            #     out = helper_tn.sum_tens(eff_Axs, transpose_bonds=bonds_o)
            #     # out = eff_Axs.copy()   ## already summed over in _get_Ax_eff_dict
            #     if out is not None:
            #         # out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
            #         out.reindex({bo: bi for bo, bi in zip(bonds_o, bonds_i)}, inplace=True)
            #         # out.modify(apply=lambda x: x * dt)
            #         out_dict[oo] = out
            #         # print('deriv', oo, out.norm())
            #         tot_norm += out.norm()**2
            # # print('tot deriv', np.sqrt(tot_norm))

            return out_dict

        def add_func(tens1_dict, tens2_dict, inplace=False, **kwargs) -> dict[int,'qtn.Tensor']:
            out_dict = tens1_dict if inplace else {k: tens.copy() for k, tens in tens1_dict.items()}
            tot_keys = set(tens1_dict.keys()).union(tens2_dict.keys())
            for k in tot_keys:
                try:
                    tens2 = tens2_dict[k]
                    try:
                        tens1 = out_dict[k]
                        out_dict[k] = helper.add_tensors(tens1, tens2, inplace=False)
                    except KeyError:
                        out_dict[k] = tens2.copy()
                except KeyError:
                    pass
            return out_dict

        def scale_func(tens_dict, scale_val, inplace=False, **kwargs) -> dict[int,'qtn.Tensor']:
            out_dict = tens_dict if inplace else {k: tens.copy() for k, tens in tens_dict.items()}
            for k, tens in out_dict.items():
                tens.modify(apply=lambda x: x * scale_val)
            return out_dict

        def euler_func(state, dt, deriv0, **kwargs) -> dict[int,'qtn.Tensor']:

            # for k, v in deriv0.items():
            #     ket = self.kets[k].copy()
            #     plt.figure()
            #     gtn = self.grid.make_gridTN(ket)
            #     plt.imshow(gtn.get_data())
            #     plt.colorbar()
            #     plt.title('ket')
            #
            #     m = ket[left_site_pos]
            #     v.transpose_like(m, inplace=True)
            #     print('v', v.inds, m.inds)
            #     ket[left_site_pos].modify(data=v.data, inds=v.inds)
            #     print('deriv', k)
            #     plt.figure()
            #     gtn = self.grid.make_gridTN(ket)
            #     plt.imshow(gtn.get_data())
            #     plt.title('deriv')
            #     plt.colorbar()
            #     plt.show()

            return add_func(state, scale_func(deriv0, dt))

        ### implicit funcs
        def add_tn_func(tens1_dict, tens2_dict, inplace=False, is_list=True,
                        **kwargs) -> dict[int,Sequence['qtn.TensorNetwork']]:
            """ elements in tens1_dict, tens2_dict are sequences of TNs
            """
            if is_list:
                out_dict = tens1_dict if inplace else {k: [tn.copy() for tn in tns] for k, tns in tens1_dict.items()}
            else:
                out_dict = tens1_dict if inplace else {k: tns.copy() for k, tns in tens1_dict.items()}
            tot_keys = set(tens1_dict.keys()).union(tens2_dict.keys())
            for k in tot_keys:
                try:
                    tens2s = tens2_dict[k]
                    try:
                        tens1s = out_dict[k]
                        # out_dict[k] = tens1s + [tn.copy() for tn in tens2s]
                        if is_list:
                            out_dict[k] = tens1s + ([tn.copy() for tn in tens2s])
                        else:
                            ## jank fix for Crank-Nicolson TE with source
                            try:
                                out_dict[k] = tens1s + tens2s.copy()
                            except ValueError:
                                out_dict[k] = tens1s + tens2s.reindex(bra_to_ket_inds)
                    except KeyError:
                        # out_dict[k] = [tn.copy() for tn in tens2s]
                        out_dict[k] = [tn.copy() for tn in tens2s] if is_list else tens2s.copy()
                except KeyError:
                    pass

            return out_dict


        def scale_tn_func(tens_dict: dict[Any,Sequence['qtn.TensorNetwork']], scale_val: float,
                          inplace=False, is_list=True, **kwargs
                          ) -> dict[Union[int,tuple[int,int]],Sequence['qtn.TensorNetwork']]:
            if is_list:
                new_dict = tens_dict if inplace else {k: [tn.copy() for tn in tns] for k, tns in tens_dict.items()}
                for k, tns in new_dict.items():
                    for tn in tns:
                        helper.scalar_multiply(tn, scale_val, inplace=True)
            else:       # is not list
                new_dict = tens_dict if inplace else {k: tn.copy() for k, tn in tens_dict.items()}
                for k, tn in new_dict.items():
                    if isinstance(tn,qtn.Tensor):
                        tn.modify(apply=lambda x: x * scale_val)
                    else:
                        helper.scalar_multiply(tn, scale_val, inplace=True)
            return new_dict


        def solve_func_cgd(deriv_ops, explicit_contribution, init_guess=None, max_iter=None, conv_tol=None,
                           constraint_mats=None, constraint_vals=None,
                           **kwargs):
            conv_kwargs = {}
            conv_kwargs['max_iter'] = 500 if max_iter is None else max_iter
            conv_kwargs['conv_tol'] = 1.0e-6 if conv_tol is None else conv_tol

            init_guess = {k: tens.copy() for k, tens in xeff_dict.items()} if init_guess is None else init_guess
            # exp_keys = list(explicit_contribution.keys())
            # print('exp keys', exp_keys, init_guess.keys())
            # for k in exp_keys:
            #     if k not in init_guess:
            #         tens = explicit_contribution[k]
            #         if False:  # tens.norm() < np.sqrt(self.cutoff):
            #             print('dropping explicit contribution', k)
            #             explicit_contribution.pop(k)
            #         else:
            #             print('adding x[k]', k)
            #             zero_tens = tens.copy()
            #             zero_tens.modify(apply=lambda x: x * 0)
            #             xeff_dict[k] = zero_tens.copy()
            #             init_guess[k] = zero_tens.copy()
            #             self.add_new_state(k, cur_orthog=left_site_pos)
            #             print('updated Aeff_dict')
            #             pdb.set_trace()

            out = self.cgd(left_site_pos, nsites, init_guess, deriv_ops, explicit_contribution,
                           constraint_mats_dict=constraint_mats, constraint_vals_dict=constraint_vals,
                           max_iter=max_iter, conv_tol=conv_tol, **kwargs)
            # print('solve err', err)
            return out


        def solve_func_lsq(deriv_ops: dict[int,Sequence['qtn.TensorNetwork']],
                           explicit_contribution: dict[int,'qtn.Tensor'],
                           init_guess=None, max_iter=None, conv_tol=None,
                           constraint_mats=None, constraint_vals=None, **kwargs):

            # shape_ = explicit_contribution.shape
            # size_ = int(np.prod(shape_))

            conv_kwargs = {}
            conv_kwargs['max_iter'] = 500 if max_iter is None else max_iter
            conv_kwargs['conv_tol'] = 1.0e-6 if conv_tol is None else conv_tol

            deriv_ops_ = {k: helper_tn.sum_eff_TNs(d_ops, [*bonds_o, *bonds_i]) for k, d_ops in deriv_ops.items()}
            # explicit_contribution = explicit_contribution.transpose(*bonds_i)
            # print('A eff tens', A_eff_tens, explicit_contribution)

            init_guess = {k: tens.copy() for k, tens in xeff_dict.items()}

            out = self.lsq(left_site_pos, nsites, init_guess, deriv_ops_, explicit_contribution,
                           max_iter=max_iter, conv_tol=conv_tol,
                           constraint_mats_dict=constraint_mats, constraint_vals_dict=constraint_vals,
                           **kwargs)
            return out


        def identity_func(xeff_dict: dict[int,Union['qtn.Tensor', 'qtn.TensorNetwork']]):
            iden_dict = {}
            for i, state in xeff_dict.items():
                shape_ = state.transpose(*bonds_i).shape
                size_ = int(np.prod(shape_))
                # iden = qtn.Tensor( np.eye(size_).reshape(*shape_, *shape_),
                #                    inds=(*bonds_o,*bonds_i) )
                # iden_dict[(i,i)] = [ qtn.TensorNetwork([iden]) ]

                iden_mats = [qtn.Tensor(np.eye(sh_), inds=(bo, bi)) for sh_, bo, bi in zip(shape_, bonds_o, bonds_i)]
                iden_dict[(i,i)] = [qtn.TensorNetwork(iden_mats)]

            return iden_dict

        ######################
        ### perform actual TE


        if te_order == 1:
            out = helper_TE.euler(xeff_dict, dt, euler_func, deriv_func, add_func, scale_func,
                                   return_intermediates=return_intermediates)

            # if return_intermediates:
            #     ## combine targets in each intermediate state (dict[int, Tensor])
            #     targets = {}
            #     for k in out[0].keys():
            #         targets[k] = [target[k] for target in out[1]]
            #     out = targets
            #     # pdb.set_trace()

        elif te_order == 4:
            out = helper_TE.rk4(xeff_dict, dt, euler_func, deriv_func, add_func, scale_func,
                                return_intermediates=return_intermediates)

            # if return_intermediates:
            #     ## combine targets in each intermediate state (dict[int, Tensor])
            #
            #     targets = {}
            #     for k in out[0].keys():
            #         targets[k] = [target[k] for target in out[1]]
            #
            #         # s0, k1, k2, k3, k4 = targets[k]  ## initial x, stage 1, stage 2, stage 3
            #         #
            #         # k1.transpose_like(s0, inplace=True)
            #         # k2.transpose_like(s0, inplace=True)
            #         # k3.transpose_like(s0, inplace=True)
            #         # k4.transpose_like(s0, inplace=True)
            #         #
            #         # ## tot_denmat:  1/3 * rho(psi03) + 1/6 * rho(psi13) + 1/6 * rho(psi23) + 1/3 * rho(psi33)
            #         # psi03 = s0.copy()
            #         # psi03.modify(apply=lambda x: x * np.sqrt(1. / 3))
            #         # psi13 = s0.copy()
            #         # psi13.modify(apply=lambda x: x + dt * (
            #         #             1. / 162 * (31 * k1.data + 14 * k2.data + 14 * k3.data - 5 * k4.data)))
            #         # psi13.modify(apply=lambda x: x * np.sqrt(1. / 6))
            #         # psi23 = s0.copy()
            #         # psi23.modify(
            #         #     apply=lambda x: x + dt * (1. / 81 * (16 * k1.data + 20 * k2.data + 20 * k3.data - 2 * k4.data)))
            #         # psi23.modify(apply=lambda x: x * np.sqrt(1. / 6))
            #         # psi33 = s0.copy()
            #         # psi33.modify(apply=lambda x: x + dt * (1. / 6 * (k1.data + 2 * k2.data + 2 * k3.data + k4.data)))
            #         # psi33.modify(apply=lambda x: x * np.sqrt(1. / 3))
            #         #
            #         # targets[k] = [psi03, psi13, psi23, psi33]
            #
            #     out = targets
            #     # pdb.set_trace()

        elif te_order == 22:
            # print('cn te', left_site_pos, nsites)

            if self.masks is not None:

                ## should somehow reuse "deriv_func" / _get_eff_Ax_dict for both "explicit" contribution
                # and "implicit" contribution; e.g. both should call _get_eff_Ax_dict

                raise NotImplementedError

            ## beff_dict:  (implied) exponent = x.exponent
            ## Aeff_dict:  exponent = A.exponent
            solve_func =  solve_func_cgd  # solve_func_lsq
            out = helper_TE.solve_crank_nicolson(xeff_dict, euler_func, deriv_func, add_tn_func, scale_tn_func,
                                                 Aeff_dict, beff_dict, identity_func,
                                                 # solve_func_cgd, dt,        ## does not work well with constraints
                                                 solve_func, dt,
                                                 # constraint_mats=constrained_mat_dict,
                                                 # constraint_vals=constrained_val_dict,
                                                 return_intermediates=return_intermediates,
                                                 )
                                                 # Aeff_dict, identity_func, solve_func_lsq, dt)
        elif (str(self.te_order)[:2] == '22'):
            wk = str(self.te_order)[2:]
            weight = np.round(int(wk) * 10 ** (-len(wk)), len(wk))
            # print(f'cn te {weight}', left_site_pos)
            out = helper_TE.solve_crank_nicolson(xeff_dict, euler_func, deriv_func, add_tn_func, scale_tn_func,
                                                 Aeff_dict, beff_dict, identity_func, solve_func_cgd, dt,
                                                 # constraint_mats=constrained_mat_dict,
                                                 # constraint_vals=constrained_val_dict,
                                                 return_intermediates=return_intermediates,
                                                 weight=weight)
        else:
            raise NotImplementedError

        if return_intermediates:
            ## out = final state, (state0, targets, out)
            ## combine targets in each intermediate state (dict[int, Tensor])
            targets = {}
            for k in out[0].keys():
                k_targets = []
                for target in out[1]:  ## list of intermediates
                    tens_k = target.get(k, None)
                    # if tens_k is None:
                    #     tens_k = out[0][k].copy()
                    #     tens_k.modify(apply=lambda x: x * 0)
                    k_targets += [tens_k]
                targets[k] = k_targets
            out = targets
            # pdb.set_trace()

        return out


    # def get_site_total_rdm(self, dt, left_site_pos, x_bond, nsites=1, te_order=4, **kwargs) -> dict[int,'qtn.Tensor']:
    #     """ sum together RDMs to combine multieple tensors...
    #     """
    #     site_inds = self.get_mps_ind(slice(left_site_pos,left_site_pos + nsites,1))
    #     dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
    #
    #     ## change indices from bra (missing T*[i]) to ket
    #     ket_to_bra_inds = {}
    #     for site_p in site_inds:
    #         ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))
    #
    #     bonds_i = dist_submpx.outer_inds()
    #     bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]
    #
    #     A_effs_dict, b_eff_dict = {}, {}  ## includes exponents (bra + ket + operator)
    #     for key, solver in self.solvers.items():
    #         ### AL * A * AR = d/dT*[i] d/dT[i] <x_j|A_ji|x_i>
    #         if solver.ket is not None and solver.bra is not None:
    #             A_effs_dict[key] = solver._get_A_effs(left_site_pos, nsites)
    #             # print('A effs', key, A_effs_dict[key], [eff.exponent for eff in A_effs_dict[key]])
    #             # print('solver ket', key[1], solver.ket.exponent)
    #             # print('solver bra', key[0], solver.bra.exponent)
    #             # print('ops', [op.exponent for op in solver.operators])
    #
    #     b_eff_dict = self._get_b_eff_dict_(site_inds, bonds_o)
    #
    #     x_dict = {}
    #     for k, ket in self.kets.items():
    #         if ket is not None:
    #             if isinstance(site_inds, (list, tuple)):
    #                 ket_tensors = [ket[si] for si in site_inds]
    #             else:
    #                 ket_tensors = ket[site_inds]
    #             x_tens = qtn.tensor_contract(*ket_tensors)
    #             x_tens.transpose(*bonds_i, inplace=True)
    #             x_dict[k] = x_tens
    #
    #     ## pad x and b so that they contain the same components
    #     for bk in b_eff_dict.keys():
    #         if bk not in x_dict:
    #             zero = b_eff_dict[bk].copy()
    #             zero.modify(apply=lambda x: x * 0)
    #             zero.reindex({b:k for k, b in ket_to_bra_inds.items()}, inplace=True)
    #             x_dict[bk] = zero
    #
    #     ## bond to contract over
    #     def compute_denmat(ket_dict: dict[int,'qtn.Tensor'], is_ket_inds=True):
    #         denmat_dict = {}
    #         for k_, ket_ in ket_dict.items():
    #             if not is_ket_inds:
    #                 ket_ = ket_.reindex({b: kk for kk, b in ket_to_bra_inds.items()})
    #             bra_ = ket_.conj()
    #             bra_.reindex({kk: b for kk, b in ket_to_bra_inds.items() if kk != x_bond}, inplace=True)
    #             denmat_dict[k_] = qtn.tensor_contract(bra_, ket_)
    #         return denmat_dict
    #
    #     rho_bonds_i = [bi for bi in bonds_i if bi != x_bond]
    #     rho_bonds_o = [ket_to_bra_inds[ind] for ind in rho_bonds_i]
    #
    #     # target_times = [0, dt/3, 2*dt/3, dt]
    #     # weights = [1./3, 1./6, 1./6, 1./3]
    #     # target_times = [0, dt/3, 2*dt/3, 4*dt/3, dt]
    #     # weights = [1./3, 1./9, 1./9., 1./9, 1./3]
    #     # target_times = [0, (1 - np.sqrt(3)/2) * dt , dt, (1 + np.sqrt(3)/2) * dt ]
    #     # weights = [1./4, 1./4, 1./4, 1./4]
    #     # target_times = [dt]
    #     # weights = [1.]
    #     target_times = [0, dt]
    #     weights = [0.5, 0.5]
    #
    #     ## for accumulated time steps
    #     sort_inds = np.argsort(target_times)
    #     target_times = [target_times[i] for i in sort_inds]
    #     target_times = [0] + [target_times[i + 1] - target_times[i] for i in range(len(sort_inds)-1)]
    #     weights = [weights[i] for i in sort_inds]
    #
    #     ## constraints
    #     eff_constraint_dict, eff_cvals_dict = self.get_site_constraints(left_site_pos, nsites=nsites,)
    #
    #     tot_denmat = {}
    #     for w, tt in zip(weights, target_times):
    #         new_xeff_dict = self.get_time_evolved_site(tt, site_inds, x_dict, A_effs_dict, b_eff_dict,
    #                                                    bonds_i, bonds_o, te_order=te_order,
    #                                                    constrained_mat_dict=eff_constraint_dict,
    #                                                    constrained_val_dict=eff_cvals_dict)
    #         x_dict = new_xeff_dict       ## for accumulated time steps
    #
    #
    #         ## clean divergences
    #
    #         # tot_norm2 = 0
    #         # for k, val in new_xeff_dict.items():
    #         #     print('evolved x eff', k, val.norm())
    #         #     tot_norm2 += val.norm()**2 * 10**(2*self.current_state[k].exponent)
    #         #     # print(self.current_state[k])
    #         # print('tot norm', tot_norm2)
    #
    #         rho0_dict = compute_denmat(new_xeff_dict)
    #         for k, rho0 in rho0_dict.items():
    #             rho0.transpose(*rho_bonds_i, *rho_bonds_o, inplace=True)  # |ket><bra|
    #             rho0.modify(apply=lambda x: x * w)
    #
    #         for k in rho0_dict.keys():
    #             if k not in tot_denmat:
    #                 tot_denmat[k] = rho0_dict[k].copy()
    #             else:
    #                 tot_denmat[k].modify(apply=lambda x: x + rho0_dict[k].data)
    #
    #     ### add source term to tot_denmat?
    #     ### should already be accounted for after doing time evolution?
    #     source_denmat_dict = compute_denmat(b_eff_dict, is_ket_inds=False)
    #     for k in source_denmat_dict.keys():
    #         # print('srouce denmat', source_denmat_dict[k])
    #         if k not in tot_denmat:
    #             tot_denmat[k] = source_denmat_dict[k].copy()
    #         else:
    #             tens = source_denmat_dict[k]
    #             tens.transpose_like(tot_denmat[k], inplace=True)
    #             tens.modify(apply=lambda x: x / tens.norm() * tot_denmat[k].norm() * 0.25)
    #             tot_denmat[k].modify(apply=lambda x: x + tens.data)
    #
    #     return tot_denmat


    # @profile
    def _site_time_evolution(self, dt, left_site_pos, nsites=1,
                             sweep_direction=SweepDirection.RIGHT, site_tens_dict: dict[int, 'qtn.Tensor']=None,
                             return_intermediates=True):
        """
        sites: int or slice(start, stop, step)
        solve local A' x = b',  (1/2?) <x|Ax> - <x|b> = 0
            where A' = d/dT[i] d/dT[i]^* <x|A|x> = d/dT[i] d/dT[i]^* \sum_ij <x_j|A_ji|x_i>
            where b' = d/dT[i]^* <x|b> = d/dT[i] \sum_ij <x_j|b_i>
        """
        te_order_target = 22 if (self.te_order == 0) else self.te_order  # 4  # 22  # 22
        te_order_final = self.te_order  # 4  # 22  # 22


        if sweep_direction == SweepDirection.RIGHT:
            at_end = (left_site_pos == self.L - nsites)
        else:
            at_end = (left_site_pos == 0)

        ket_inds = self.ket_inds(left_site_pos, nsites)
        bra_inds = self.bra_inds(left_site_pos, nsites, ket_inds)
        bonds_o = bra_inds
        bonds_i = ket_inds

        new_xeff_dict = self.get_time_evolved_site(dt, left_site_pos, nsites,
                                                   # x_dict, A_effs_dict, b_eff_dict,
                                                   bonds_i, bonds_o,
                                                   te_order=te_order_final if at_end else te_order_target,
                                                   return_intermediates=(return_intermediates and not at_end),
                                                   site_tens_dict=site_tens_dict
                                                   )

        return new_xeff_dict


    # @profile
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

        # print('take time step max bond')
        # for k, v in self.kets.items():
        #     print('k', k, v.max_bond())
        # pdb.set_trace()

        ## ovlp <Ax|b>, <Ax|x> init envs
        if init_direction == SweepDirection.RIGHT:
            # self._build_all_envs_right(1, canonize=False)

            ### left to right sweep
            self.take_time_step_l2r(dt, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)
            ### right to left sweep
            # self.take_time_step_r2l(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)

        else:
            # self._build_all_envs_left(1, canonize=False)

            ### right to left sweep
            self.take_time_step_r2l(dt, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)
            # ### left to right sweep
            # self.take_time_step_l2r(dt / 2, grid=grid, do_adapt=do_adapt, canonize=False, build_envs=False, **kwargs)

        print('done sweep')
        for k, v in self.states_dict.items():
            print('k', k, helper_quimb.inner_bond_sizes(v))

        return self.kets

    def take_time_step_l2r(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, verbose=False,
                           **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.L
        max_bond = self.max_bond
        # do_adapt = False
        print('l2r block tddmrg', dt, 'do adapt', do_adapt)
        # print('ket L', L)
        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if canonize:
            self.direction = SweepDirection.RIGHT
            self.canonize(0)   ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_right(1, canonize=False)

        ## right sweep
        nsites = 0
        direction = SweepDirection.RIGHT
        site_ind = 0
        cur_orthog = 0
        while site_ind < L:

            if site_ind == L - 1 and nsites == 2:
                ## previously updated L-2, L-1
                if verbose:  print('continue', site_ind)
                site_ind += 1
                continue

            ## num sites for update
            if site_ind == L - 1:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                # max_bond_ = max_bond  # min(max_bond, 2**(cur_orthog+1), 2**(L-cur_orthog-1))
                max_bond_ = min(max_bond, 2**(site_ind+1), 2**(L-site_ind-1))
                # print('r2l', self.ket.bond_size(site_ind, site_ind + 1), max_bond_)
                adapt = do_adapt and self.ket.bond_size(site_ind, site_ind + 1) < max_bond_

            # adapt=True
            # adapt=False
            nsites = 2 if adapt else 1

            ## forward propagation
            if verbose:  print('FORWARD prop sites (LR)', list(range(site_ind,site_ind+nsites)) )
            new_xeff_dict = self._site_time_evolution(dt, left_site_pos=site_ind, nsites=nsites, sweep_direction=direction)

            ## update kets
            if nsites == 1:
                self._update_1site(site_ind, new_xeff_dict, direction)
            elif nsites == 2:
                self._update_2site(site_ind, new_xeff_dict, direction)
            ## updates environment within function to site_ind + nsites - 1

            cur_orthog = site_ind + nsites - 1
            site_ind = site_ind + nsites - 1 if nsites > 1 else site_ind + nsites

        print('done left to right sweep')
        return self.ket


    def take_time_step_r2l(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, verbose=False,
                           **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.L
        # do_adapt = False
        max_bond = self.max_bond
        print('r2l', dt, 'do adapt', do_adapt)
        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if canonize:
            self.direction = SweepDirection.LEFT
            self.canonize(L-1)  ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_left(1, canonize=False)

        ### right to left sweep
        nsites = 0
        direction = SweepDirection.LEFT
        site_ind = L - 1
        cur_orthog = L - 1
        while site_ind >= 0:

            if site_ind == 0 and nsites == 2:
                ## previously updated 0, 1
                if verbose:   print('continue', site_ind)
                site_ind -= 1
                continue

            if site_ind == 0:
                adapt = False
            elif max_bond is None:
                adapt = do_adapt
            else:
                max_bond_ = min(max_bond, 2 ** site_ind, 2 ** (L - site_ind))  # if site_ind != L-1 else 2
                # print('r2l', self.ket.bond_size(site_ind, site_ind - 1), max_bond_)
                adapt = do_adapt and self.ket.bond_size(site_ind, site_ind - 1) < max_bond_

            # adapt = True
            # adapt = False
            nsites = 2 if adapt else 1
            cur_orthog = site_ind - nsites + 1    ## get left site ind

            if verbose:  print('FORWARD prop sites (RL)', list(range(cur_orthog,cur_orthog+nsites)) )
            new_xeff_dict = self._site_time_evolution(dt, cur_orthog, nsites=nsites, sweep_direction=direction)

            ## update kets
            if nsites == 1:
                self._update_1site(cur_orthog, new_xeff_dict, direction)
            elif nsites == 2:
                self._update_2site(cur_orthog, new_xeff_dict, direction)

            site_ind = site_ind - nsites + 1 if nsites > 1 else site_ind - nsites

        print('done right to left sweep')
        return self.ket


class BlockTDDMRGSolver(BlockTimeIntegrator):
    """ TD-DMRG solver with multiple components at once
        assume that the grid is ordered by scale
    """

    @property
    def term_class(self):
        return Term_DMRG
        # raise NotImplementedError

    @property
    def solver_type(self):
        return LocalSolverType.DMRG

    def _get_linop_terms(self, init_x, init_bra, As):
        # return [self.term_class(init_x, init_bra, operators=As)]
        return [self.term_class(init_x, init_bra, operators=[A]) for A in As]

    def _get_mask_terms(self, init_bra, bs):
        return [self.term_class(init_bra, operators=bs)]



class BlockTDDMRGXSolver(BlockTimeIntegrator):
    """ TD-DMRG solver with multiple components at once
        assume that the grid is ordered by scale
    """

    @property
    def term_class(self):
        return Term_Cross

    @property
    def solver_type(self):
        return LocalSolverType.TDCross

    def _get_linop_terms(self, init_x, init_bra, As):
        return [self.term_class(init_x, init_bra, operators=[A], num_tiers=1) for A in As]

    def _get_mask_terms(self, init_bra, bs):
        print('interpolative mask')
        return [self.term_class(b, init_bra, num_tiers=1) for b in bs]

    def _update_1site(self, i: int, site_i_dict: dict[int, 'qtn.Tensor'], direction: 'SweepDirection'):

        at_end = (i == self.L - 1) if direction > 0 else (i == 0)

        if set(self.states_dict.keys()) != set(site_i_dict.keys()):
            raise ValueError

        ## update self.current_state
        for ii, ket in self.states_dict.items():
            try:
                new_ket_tens = site_i_dict[ii] # .copy()

                if self.verbose_plot and ii == 2:
                    tens_list = [new_ket_tens] if not isinstance(new_ket_tens, (list, tuple)) else new_ket_tens
                    for _, tens in enumerate(tens_list):

                        print('tens', _, tens)
                        tmp = ket.copy()

                        tens_ = tens.reindex({ind: ind[:-2] for ind in tens.inds if ind[-1] == 'x'})
                        helper_cross.plot_submat(tmp, i, 1, tens_)

                        tmp[i].transpose_like(tens_, inplace=True)
                        tmp[i].modify(data = tens_.data)
                        plt.figure()
                        plt.imshow(helper_quimb.to_dense(tmp).reshape(2**(ket.L //2), -1))
                        plt.title(f'ket {ii} replace tens {_}')
                        plt.show()


                helper_cross.update_ket(ket, new_ket_tens, i, 1, direction=direction,
                                        max_bond=self.max_bond, cutoff=self.cutoff,
                                        # decimate_only=(not at_end)
                                        )

                # print('UPDATE 1SITE', helper_quimb.distance(ket, self.init_ket[ii]))
                # pdb.set_trace()

            except KeyError:
                helper_cross.canonize(ket, i + direction.value, cur_orthog=i)

        if not at_end:
            for term in self.terms:

                # for ii, ket in self.states_dict.items():
                #     print('term', ii, term.bra is ket, term.ket is ket)
                #     print('vec block', ii, term.vec_block.bra is ket, term.vec_block.ket is ket)
                #     if len(term.op_blocks) > 0:
                #         for ops in term.op_blocks[0]:
                #             print('op block', ii, ops.bra is ket, ops.ket is ket)
                #
                # print('update term', term)
                # print('term bra', term.bra is term.vec_block.bra, None if term.operators is None else len(term.operators))
                # print('term ket', term.ket is term.vec_block.bra)
                # for ii, ket in self.states_dict.items():
                #     print('term bra', ii, term.bra is ket)
                #     print('term bra', term.vec_block.bra.select_inds.keys())
                #
                # pdb.set_trace()

                term.extend_env(i, direction=direction)

        return

    def _update_2site(self, left_site_pos: int, site_i_dict: dict[int, 'qtn.Tensor'], direction: 'SweepDirection'):

        at_end = (left_site_pos == self.L - 2) if direction > 0 else (left_site_pos == 0)
        i = left_site_pos if direction > 0 else left_site_pos + 1

        ## update self.current_state
        for ii, ket in self.states_dict.items():
            try:
                new_ket_tens = site_i_dict[ii].copy()
                helper_cross.update_ket(ket, new_ket_tens, i, 2, direction=direction,
                                        max_bond=self.max_bond, cutoff=self.cutoff)

            except KeyError:
                pass

        if not at_end:
            for term in self.terms:
                term.extend_env(i, direction=direction)

        return

    def _get_Ax_eff_dict_(self, left_site_pos: int, nsites: int, site_tens_dict=None, negative_dt=False):
        """
        AL * A @ x * AR = d/dT*[s] <x_j|A|x_i>
        dict indexes xj
        """
        # ket_site_inds = [self.ket.site_ind_id.format(si) for si in site_inds]
        # bra_site_inds = [self.bra.site_ind_id.format(si) for si in site_inds]

        Ax_eff_dict = {}
        site_tens_dict = site_tens_dict if site_tens_dict is not None else {}

        ### BL * B * BR = d/dT*[i] <x|b>
        for (oo, ii), terms in self.op_terms.items():

            # print('site tens dict', site_tens_dict)
            # for term in terms:
            #     print('Ax check orthog!')
            #     term.check_orthog()
            #     helper_cross.check_orthog(term.ket)
            #     helper_cross.check_orthog(term.bra)
            #     ## not the same bc env might not be normalized to 1? why? maybe bc of scalar multiply?

            Ax_effs = [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens_dict.get(ii, None))
                       for term in terms]

            ## for backward time integration
            if negative_dt and self.backward_weights is not None:
                weights = self.backward_weights.get((oo, ii), [])
                print('negative dt weights', ii, oo, weights)
                for coeff, Ax_eff in zip(weights, Ax_effs):
                    Ax_eff.modify(apply=lambda x: x * coeff)

            ## apply mask d/dt x = c * A * x
            print('interpolative mask application', oo)
            c_terms = self.mask_terms.get(oo, None)
            if c_terms is not None:
                ## c * Ax_effs
                for c_term in c_terms:
                    c_tens = c_term.get_evaluated_site(left_site_pos, nsites)

                    # for t in terms:
                    #     print('c tens bra is term bra', c_term.bra is t.bra)
                    # pdb.set_trace()

                    if self.term_class is Term_Mixed and self.verbose_plot:
                        c_tens_ = c_tens.reindex({ind: ind[:-2] for ind in c_tens.inds if ind[-1] == 'x'})
                        proj_c = c_term.bra.copy()
                        # print('c tens_', c_tens)
                        c_tens_g = helper_mixed.convert_elementwise_to_basis(proj_c, c_tens, left_site_pos, nsites)
                        if nsites == 1:
                            # print('c tens g', c_tens_g)
                            # print('proj c', proj_c[left_site_pos])
                            c_tens_g.transpose_like(proj_c[left_site_pos], inplace=True)
                            proj_c[left_site_pos].modify(data=c_tens_g.data)
                        helper_cross.plot_submat(c_term.bra, left_site_pos, nsites, c_tens_,
                                                 ref_kets=[proj_c]
                                                 # ref_kets=[c_term.ket]
                                                 )

                    for tens_ in Ax_effs:
                        c_tens.transpose_like(tens_, inplace=True)
                        tens_.modify(apply = lambda x: x * c_tens.data)

            b2k_dict = self.get_bra_to_ket_inds(left_site_pos, nsites, comp=oo)
            for Ax_eff in Ax_effs:
                Ax_eff.reindex(b2k_dict, inplace=True)

            # ## check with cross TT
            # for it, eff_Ax in enumerate(Ax_effs):
            #     ket = self.states_dict[ii]
            #     ket_init = self.init_ket[ii]
            #     print('ket is', ket is term.ket, ket is term.vec_block.ket, ket is term.vec_block.bra)
            #     if len(term.op_blocks) > 0:
            #         print('ket is op', term.vec_block.bra is ket, term.op_blocks[0][0].ket is ket)
            #     print('check init', ket is ket_init, helper_quimb.distance(ket, ket_init))
            #
            #     ket = ket.copy()
            #     ket_init = ket_init.copy()
            #
            #     plt.figure()
            #     gtn = self.grid.make_gridTN(ket)
            #     x_data = gtn.get_data()
            #     plt.imshow(gtn.get_data())
            #     plt.title(f'x {ii}')
            #     plt.colorbar()
            #
            #     plt.figure()
            #     gtn = self.grid.make_gridTN(ket_init)
            #     plt.imshow(gtn.get_data() - x_data)
            #     plt.title(f'chk x {ii}')
            #     plt.colorbar()
            #
            #     Aket = helper_quimb.apply(terms[it].operators[0].copy(), ket.copy())
            #     plt.figure()
            #     gtn = self.grid.make_gridTN(Aket)
            #     plt.imshow(gtn.get_data())
            #     plt.title(f' Ax ideal')
            #     plt.colorbar()
            #
            #     ## project ideal output
            #     # bra = self.states_dict[oo].copy()
            #     bra = terms[it].bra.copy()
            #     bra = bra.conj()
            #     bra.mangle_inner_(append='_')
            #     # bra.site_ind_id = bra.site_ind_id + '_'
            #
            #     left_bra, right_bra = helper_cross.get_projector(bra, left_site_pos, 1)
            #     # bra_tensors = [bra[i] for i in range(left_site_pos)] + [bra[i] for i in range(left_site_pos + 1, bra.L)]
            #     tens_ = qtn.tensor_contract(*left_bra, *right_bra, *Aket.tensors)
            #     print('tens_', tens_)
            #     m = bra[left_site_pos]
            #     print('mbra m', m)
            #     m.modify(data=tens_.data, inds=tens_.inds)
            #     plt.figure()
            #     gtn = self.grid.make_gridTN(bra)
            #     plt.imshow(gtn.get_data())
            #     plt.title(f'X Ideal Proj Ax({oo},{ii})')
            #     plt.colorbar()
            #
            #     bra2 = bra.copy()
            #     bra2.site_ind_id = bra.site_ind_id + '_'
            #     left_bra, right_bra = helper_cross.get_projector(bra2, left_site_pos, 1)
            #     ket_tensors = [ket[i] for i in range(left_site_pos)] + [ket[i] for i in range(left_site_pos + 1, ket.L)]
            #     eff_op = qtn.TensorNetwork([*left_bra, *right_bra, *ket_tensors, terms[it].operators[0].tensors])
            #     eff_op_tens = eff_op.contract()
            #     ket_inds = self.ket_inds(left_site_pos, nsites)
            #     bra_inds = self.bra_inds(left_site_pos, nsites)
            #     eff_op_tens = eff_op_tens.transpose(*bra_inds, *ket_inds)
            #     # print('eff x op tens', eff_op_tens.data, eff_op_tens.inds)
            #
            #     # for ix in range(9, self.L-1):
            #     #     print('i', ix)
            #     #     _, right_bra = helper_cross.get_projector(bra2, ix, 1, get_left=False)
            #     #     ket_tensors = [ket[i] for i in range(ix + 1, ket.L)]
            #     #     op_tensors = [terms[it].operators[0][i] for i in range(ix + 1, ket.L)]
            #     #     env_r = qtn.TensorNetwork([*right_bra, *ket_tensors, *op_tensors])
            #     #     env_r_tens = env_r.contract()
            #     #     print('evn r (2)', env_r_tens.data, env_r_tens.inds, env_r.shape)
            #
            #     ket = self.states_dict[oo].copy()
            #     m = ket[left_site_pos]
            #     m.modify(data=eff_Ax.data, inds=eff_Ax.inds)
            #
            #     plt.figure()
            #     gtn = self.grid.make_gridTN(ket)
            #     plt.imshow(gtn.get_data())
            #     plt.title(f'X Proj Ax({oo},{ii})')
            #     plt.colorbar()
            #     plt.show()

            if oo not in Ax_eff_dict:
                Ax_eff_dict[oo] = Ax_effs
            else:
                Ax_eff_dict[oo] += Ax_effs

        for k in Ax_eff_dict.keys():
            Ax_eff_dict[k] = [helper_tn.sum_tens(Ax_eff_dict[k])]

        return Ax_eff_dict


class BlockTDDMRGMSolver(BlockTDDMRGXSolver):
    """ TD-DMRG solver with multiple components at once
        assume that the grid is ordered by scale
    """
    def ket_inds(self, left_site_pos, nsites, comp=None):
        ref_ket = self.ket if comp is None else self.kets.get(comp, self.ket)
        iL, iR = left_site_pos, nsites + left_site_pos - 1
        left_ind = [ref_ket.bond(iL, iL - 1) + '_x'] if iL > 0 else []
        right_ind = [ref_ket.bond(iR, iR + 1) + '_x'] if iR < ref_ket.L - 1 else []
        site_ind = [ref_ket.site_ind(i) for i in range(iL, iR + 1)]
        return [*left_ind, *right_ind, *site_ind]

    def get_ket_to_bra_inds(self, left_site_pos, nsites=1, comp=None) -> dict[str, str]:
        """ inds mapping inds on self.bra to corresponding inds on self.ket
        """
        dict = {ind: ind[:-2] + '__x' for ind in self.ket_inds(left_site_pos, nsites, comp=comp)}
        dict[self.component_ind] = self.component_ind + '_'
        return dict

    # def get_bra_to_ket_inds(self, left_site_pos, nsites=1, comp=None) -> dict[str, str]:
    #     ket_to_bra = self.get_ket_to_bra_inds(left_site_pos, nsites, comp=comp)
    #     bra_to_ket = {item: k for k, item in ket_to_bra.items()}
    #     return bra_to_ket

    @property
    def term_class(self):
        return Term_Mixed

    @property
    def solver_type(self):
        return LocalSolverType.MIXED

    # def _get_linop_terms(self, init_x, init_bra, As):
    #     return [self.term_class(init_x, init_bra, operators=[A], num_tiers=1) for A in As]
    #
    # def _get_mask_terms(self, init_bra, bs):
    #     print('mixed mask')
    #     return [self.term_class(b, init_bra, num_tiers=1) for b in bs]


    # def _get_mask_terms(self, init_bra, bs):
    #     return [self.term_class(init_bra, operators=bs)]
    #
    #
    # def _get_Ax_eff_dict_(self, left_site_pos: int, nsites: int, site_tens_dict=None, negative_dt=False,
    #                       sum_terms=True, plot_verbose=False):
    #     """
    #     AL * A @ x * AR = d/dT*[s] <x_j|A|x_i>
    #     dict indexes xj
    #     """
    #
    #     Ax_eff_dict = {}
    #     site_tens_dict = site_tens_dict if site_tens_dict is not None else {}
    #
    #     ### BL * B * BR = d/dT*[i] <x|b>
    #     for (oo, ii), terms in self.op_terms.items():
    #
    #         # ket_site_inds = [self.ket.site_ind_id.format(si) for si in range(left_site_pos, left_site_pos + nsites)]
    #         # bra_site_inds = [self.bra.site_ind_id.format(si) for si in range(left_site_pos, left_site_pos + nsites)]
    #         b2k_dict = self.get_bra_to_ket_inds(left_site_pos, nsites, comp=oo)
    #
    #         # print('site tens dict', site_tens_dict)
    #         # for term in terms:
    #         #     print('Ax check orthog!')
    #         #     term.check_orthog()
    #         #     # helper_quimb.check_orthog(term.ket)
    #         #     # helper_quimb.check_orthog(term.bra)
    #         #     ## not the same bc env might not be normalized to 1? why? maybe bc of scalar multiply?
    #
    #         # for op in self.operators[(oo,ii)]:
    #         #     print('op exponent', op.exponent)
    #         #
    #         # print('state exponent', self.states_dict[oo].exponent)
    #         Ax_effs = [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens_dict.get(ii, None))
    #                    for term in terms]
    #         ## for backward time integration
    #         if negative_dt and self.backward_weights is not None:
    #             weights = self.backward_weights.get((oo, ii), [])
    #             print('negative dt weights', ii, oo, weights)
    #             for coeff, Ax_eff in zip(weights, Ax_effs):
    #                 Ax_eff.modify(apply=lambda x: x * coeff)
    #
    #         ## apply mask d/dt x = c * A * x
    #         c_terms = self.mask_terms.get(oo, None)
    #         if c_terms is not None:
    #             ## c * Ax_effs
    #             cAx_effs = []
    #             for c_term in c_terms:
    #                 for tens_ in Ax_effs:
    #                     cAx_effs = [c_term.get_evaluated_site(left_site_pos, nsites, tens_)]
    #             Ax_effs = cAx_effs
    #
    #         for Ax_eff in Ax_effs:
    #             Ax_eff.reindex(b2k_dict, inplace=True)
    #
    #         # plot_verbose = self.verbose
    #         if oo == 5 and plot_verbose: # and left_site_pos < 3: # and site_tens_dict is None:
    #             copy_oo = self.states_dict[oo].copy()
    #             copy_ii = self.states_dict[ii].copy()
    #             print('Axeff left site pos', left_site_pos, 'nsites', nsites, 'num terms', len(Ax_effs))
    #             for ix, new_site in enumerate(Ax_effs):
    #                 npts_z = self.grid.axes[2].npts
    #                 term = terms[ix]
    #                 exact_op = helper_quimb.apply(term.operators[0], copy_ii)
    #                 # print('exact op', exact_op)
    #                 plt.figure()
    #                 gtn = self.grid.get_ones_mps()
    #                 self.grid.dmrg_to_gtn_format(gtn, exact_op)
    #                 plt.imshow(np.real(gtn.get_realspace_data()[:,:,npts_z//2]))
    #                 plt.title(f'global oo,ii {oo},{ii}; {left_site_pos},{nsites}')
    #                 plt.colorbar()
    #
    #                 if nsites == 2:
    #                     if left_site_pos > 0:
    #                         left_inds = [copy_oo.bond(left_site_pos, left_site_pos - 1), copy_oo.site_ind(left_site_pos)]
    #                     else:
    #                         left_inds = [copy_oo.site_ind(left_site_pos)]
    #                     print('new site', new_site.norm())
    #                     if new_site.norm() > 0:
    #                         T1, T2 = qtn.tensor_split(new_site, left_inds = left_inds, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
    #                     else:
    #                         print('copy oo tensor', left_site_pos, copy_oo[left_site_pos].norm())
    #                         # T1, _ = qtn.tensor_split(copy_oo[left_site_pos], left_inds=left_inds, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
    #                         T1 = copy_oo[left_site_pos].copy()
    #                         T2 = copy_oo[left_site_pos + 1].copy()
    #                         T2.modify(apply=lambda x: x * 0)
    #                     T1.transpose_like(copy_oo[left_site_pos], inplace=True)
    #                     T2.transpose_like(copy_oo[left_site_pos + 1], inplace=True)
    #                     copy_oo[left_site_pos].modify(data=T1.data)
    #                     copy_oo[left_site_pos + 1].modify(data=T2.data)
    #                 elif nsites == 1:
    #                     new_site.transpose_like(copy_oo[left_site_pos], inplace=True)
    #                     copy_oo[left_site_pos].modify(data=new_site.data)
    #
    #                 plt.figure()
    #                 gtn = self.grid.get_ones_mps()
    #                 self.grid.dmrg_to_gtn_format(gtn, copy_oo)
    #                 plt.imshow(np.real(gtn.get_realspace_data())[:,:,npts_z//2])
    #                 plt.title(f'oo,ii {oo},{ii}; {left_site_pos},{nsites}')
    #                 plt.colorbar()
    #                 # plt.show()
    #
    #             for it, eff_Ax in enumerate(Ax_effs):
    #                 # ket = self.states_dict[ii].copy()
    #                 ket = terms[it].ket.copy()
    #                 print('term bra', terms[it].bra is self.states_dict[oo])
    #                 print('term ket', terms[it].ket is self.states_dict[ii])
    #
    #
    #                 plt.figure()
    #                 gtn = self.grid.make_gridTN(ket.copy())
    #                 plt.imshow(np.real(gtn.get_realspace_data())[:,:,npts_z//2])
    #                 plt.title(f'x {ii}')
    #                 plt.colorbar()
    #
    #                 Aket = helper_quimb.apply(terms[it].operators[0].copy(), ket.copy())
    #                 plt.figure()
    #                 gtn = self.grid.make_gridTN(Aket)
    #                 plt.imshow(np.real(gtn.get_realspace_data())[:,:,npts_z//2])
    #                 plt.title(f' Ax ideal')
    #                 plt.colorbar()
    #
    #                 ## project ideal output
    #                 # bra = self.states_dict[oo].copy()
    #                 bra = terms[it].bra.copy()
    #                 # bra = bra.conj()
    #                 bra.mangle_inner_(append='_')
    #                 # bra.site_ind_id = bra.site_ind_id + '_'
    #                 bra_tensors = ([bra[i].conj() for i in range(left_site_pos)] +
    #                                [bra[i].conj() for i in range(left_site_pos + nsites, bra.L)])
    #                 tens_ = qtn.tensor_contract(*bra_tensors, *Aket.tensors)
    #                 if nsites == 1:
    #                     m = bra[left_site_pos]
    #                     m.modify(data=tens_.data, inds=tens_.inds)
    #                 else:
    #                     m1 = bra[left_site_pos]
    #                     m2 = bra[left_site_pos + 1]
    #                     left_inds = tens_.bonds(m1)
    #                     if tens_.norm() > 0:
    #                         T1, T2 = qtn.tensor_split(tens_, left_inds, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
    #                     else:
    #                         T1 = m1.copy()
    #                         T2 = m2.copy()
    #                         T2.modify(apply=lambda x: x * 0)
    #                     T1.transpose_like(m1, inplace=True)
    #                     T2.transpose_like(m2, inplace=True)
    #                     m1.modify(data=T1.data)
    #                     m2.modify(data=T2.data)
    #
    #                 plt.figure()
    #                 gtn = self.grid.make_gridTN(bra)
    #                 plt.imshow(np.real(gtn.get_realspace_data())[:,:,npts_z//2])
    #                 plt.title(f'Ideal Proj Ax({oo},{ii})')
    #                 plt.colorbar()
    #
    #                 bra2 = bra.copy()
    #                 bra2.site_ind_id = bra.site_ind_id + '_'
    #                 bra_tensors = [bra2[i] for i in range(left_site_pos)] + [bra2[i] for i in range(left_site_pos + nsites, bra.L)]
    #                 ket_tensors = [ket[i] for i in range(left_site_pos)] + [ket[i] for i in range(left_site_pos + nsites, ket.L)]
    #                 eff_op = qtn.TensorNetwork([*bra_tensors, *ket_tensors, terms[it].operators[0].tensors])
    #                 eff_op_tens = eff_op.contract()
    #                 ket_inds = self.ket_inds(left_site_pos, nsites)
    #                 bra_inds = self.bra_inds(left_site_pos, nsites)
    #                 eff_op_tens = eff_op_tens.transpose(*bra_inds, *ket_inds)
    #                 eff_op_tens.modify(apply=lambda x: x * 10**(terms[it].operators[0].exponent))
    #                 # print('eff op tens', eff_op_tens.data, eff_op_tens.inds)
    #
    #                 # for ix in range(9, self.L-1):
    #                 #     print('i', ix)
    #                 #     bra_tensors = [bra2[i] for i in range(ix + 1, bra.L)]
    #                 #     ket_tensors = [ket[i] for i in range(ix + 1, ket.L)]
    #                 #     op_tensors = [terms[it].operators[0][i] for i in range(ix + 1, ket.L)]
    #                 #     env_r = qtn.TensorNetwork([*bra_tensors, *ket_tensors, *op_tensors])
    #                 #     env_r_tens = env_r.contract()
    #                 #     print('evn r (2)', env_r_tens.data, env_r_tens.inds, env_r.shape)
    #
    #                 ket = self.states_dict[oo].copy()
    #                 eff_Ax = eff_Ax.reindex({bi: ki for bi, ki in zip(bra_inds, ket_inds)}, inplace=False)
    #                 if nsites == 2:
    #                     left_inds = eff_Ax.bonds(ket[left_site_pos])
    #                     T1, T2 = qtn.tensor_split(eff_Ax, left_inds,)
    #                     m1 = ket[left_site_pos]
    #                     m2 = ket[left_site_pos + 1]
    #                     T1.transpose_like(m1, inplace=True)
    #                     m1.modify(data=T1.data)
    #                     T2.transpose_like(m2, inplace=True)
    #                     m2.modify(data=T2.data)
    #                 else:
    #                     m = ket[left_site_pos]
    #                     eff_Ax.transpose_like(m, inplace=True)
    #                     m.modify(data=eff_Ax.data) #, inds=eff_Ax.inds)
    #
    #                 print('left site pos', left_site_pos)
    #                 print('state', oo, ii, self.states_dict[oo][left_site_pos])
    #                 if nsites == 2:
    #                     print('state', oo, ii, self.states_dict[oo][left_site_pos + 1])
    #
    #                 plt.figure()
    #                 gtn = self.grid.make_gridTN(ket)
    #                 plt.imshow(np.real(gtn.get_realspace_data())[:,:,npts_z//2])
    #                 plt.title(f'Proj Ax({oo},{ii})')
    #                 plt.colorbar()
    #                 plt.show()
    #
    #                 print('end verbose plot')
    #
    #         if oo not in Ax_eff_dict:
    #             Ax_eff_dict[oo] = Ax_effs
    #         else:
    #             Ax_eff_dict[oo] += Ax_effs
    #
    #     if sum_terms:
    #         for k in Ax_eff_dict.keys():
    #             Ax_eff_dict[k] = [helper_tn.sum_tens(Ax_eff_dict[k])]
    #
    #     return Ax_eff_dict


    def _update_1site(self, i: int, site_i_dict: dict[int, 'qtn.Tensor'], direction: 'SweepDirection'):

        at_end = (i == self.L - 1) if direction > 0 else (i == 0)

        if set(self.states_dict.keys()) != set(site_i_dict.keys()):
            raise ValueError

        ## update self.current_state
        for ii, ket in self.states_dict.items():
            print('UPDATE 1SITE', ii)
            try:
                new_ket_tens = site_i_dict[ii] # .copy()

                if self.verbose_plot and ii == 2:
                    tens_list = [new_ket_tens] if not isinstance(new_ket_tens, (list, tuple)) else new_ket_tens
                    for _, tens in enumerate(tens_list):
                        tens = tens.copy()
                        print('tens', _, tens)
                        tmp = ket.copy()

                        tens_ = tens.reindex({ind: ind[:-2] for ind in tens.inds if ind[-1] == 'x'})
                        helper_cross.plot_submat(tmp, i, 1, tens_, plt_title=f'tens {_}')

                        tens_g = helper_mixed.convert_elementwise_to_basis(tmp, tens, i, 1)
                        print('tens', tens_g)
                        tmp[i].transpose_like(tens_g, inplace=True)
                        tmp[i].modify(data = tens_g.data)
                        plt.figure()
                        plt.imshow(helper_quimb.to_dense(tmp).reshape(2**(ket.L //2), -1))
                        plt.title(f'ket {ii} replace tens {_}')
                        plt.show()

                # if not at_end:
                #     mask_tens = []
                #     mask_terms_ii = self.mask_terms.get(ii, [])
                #     for mt in mask_terms_ii:
                #         mask_tens += [mt.get_evaluated_site(i,1)]
                #     new_ket_tens = new_ket_tens + mask_tens


                print('update ket', self.max_bond, self.cutoff)
                helper_mixed.update_ket(ket, new_ket_tens, i, 1, direction=direction,
                                        max_bond=self.max_bond, cutoff=self.cutoff,
                                        # decimate_only=(not at_end)
                                        verbose_plot=self.verbose_plot,
                                        )

                # print('UPDATE 1SITE', helper_quimb.distance(ket, self.init_ket[ii]))
                # pdb.set_trace()

            except KeyError:
                helper_mixed.canonize(ket, i + direction.value, cur_orthog=i)

        if not at_end:
            for term in self.terms:

                # for ii, ket in self.states_dict.items():
                #     print('term', ii, term.bra is ket, term.ket is ket)
                #     print('vec block', ii, term.vec_block.bra is ket, term.vec_block.ket is ket)
                #     if len(term.op_blocks) > 0:
                #         for ops in term.op_blocks[0]:
                #             print('op block', ii, ops.bra is ket, ops.ket is ket)
                #
                # print('update term', term)
                # print('term bra', term.bra is term.vec_block.bra, None if term.operators is None else len(term.operators))
                # print('term ket', term.ket is term.vec_block.bra)
                # for ii, ket in self.states_dict.items():
                #     print('term bra', ii, term.bra is ket)
                #     print('term bra', term.vec_block.bra.select_inds.keys())
                #
                # pdb.set_trace()

                term.extend_env(i, direction=direction)

        return

    def _update_2site(self, left_site_pos: int, site_i_dict: dict[int, 'qtn.Tensor'], direction: 'SweepDirection'):

        at_end = (left_site_pos == self.L - 2) if direction > 0 else (left_site_pos == 0)
        i = left_site_pos if direction > 0 else left_site_pos + 1

        ## update self.current_state
        for ii, ket in self.states_dict.items():
            try:
                new_ket_tens = site_i_dict[ii].copy()
                helper_mixed.update_ket(ket, new_ket_tens, i, 2, direction=direction,
                                        max_bond=self.max_bond, cutoff=self.cutoff)

            except KeyError:
                pass

        if not at_end:
            for term in self.terms:
                term.extend_env(i, direction=direction)

        return


    # def _get_A_eff_dict_(self, left_site_pos: int, nsites: int, negative_dt=False):
    #     """
    #     AL * A @ x * AR = d/dT*[s] d/dT[s] <x_j|A|x_i>
    #     dict indexes xj
    #     """
    #     A_eff_dict = {}
    #     # site_tens_dict = site_tens_dict if site_tens_dict is not None else {}
    #
    #     ### BL * B * BR = d/dT*[i] <x|b>
    #     for (oo, ii), terms in self.op_terms.items():
    #
    #         A_eff_dict[(oo, ii)] = []
    #         for term in terms:
    #             eff_ops = term.get_eff_operator(left_site_pos, nsites)[0]
    #
    #             if negative_dt and self.backward_weights is not None:
    #                 weights = self.backward_weights.get((oo, ii), [])
    #                 for coeff, eff_op in zip(weights, eff_ops):
    #                     eff_op.tensors[0].modify(apply=lambda x: x * coeff)
    #
    #             A_eff_dict[(oo, ii)] += eff_ops
    #             ## assume only 1 tier
    #
    #     return A_eff_dict

    # def _get_Ax_eff_dict_(self, left_site_pos: int, nsites: int, site_tens_dict=None, negative_dt=False):
    #     """
    #     AL * A @ x * AR = d/dT*[s] <x_j|A|x_i>
    #     dict indexes xj
    #     """
    #     # ket_site_inds = [self.ket.site_ind_id.format(si) for si in site_inds]
    #     # bra_site_inds = [self.bra.site_ind_id.format(si) for si in site_inds]
    #
    #     Ax_eff_dict = {}
    #     site_tens_dict = site_tens_dict if site_tens_dict is not None else {}
    #
    #     ### BL * B * BR = d/dT*[i] <x|b>
    #     for (oo, ii), terms in self.op_terms.items():
    #
    #         # print('site tens dict', site_tens_dict)
    #         # for term in terms:
    #         #     print('Ax check orthog!')
    #         #     term.check_orthog()
    #         #     helper_cross.check_orthog(term.ket)
    #         #     helper_cross.check_orthog(term.bra)
    #         #     ## not the same bc env might not be normalized to 1? why? maybe bc of scalar multiply?
    #
    #         Ax_effs = [term.get_evaluated_site(left_site_pos, nsites, site_tens=site_tens_dict.get(ii, None))
    #                    for term in terms]
    #
    #         ## for backward time integration
    #         if negative_dt and self.backward_weights is not None:
    #             weights = self.backward_weights.get((oo, ii), [])
    #             print('negative dt weights', ii, oo, weights)
    #             for coeff, Ax_eff in zip(weights, Ax_effs):
    #                 Ax_eff.modify(apply=lambda x: x * coeff)
    #
    #         ## apply mask d/dt x = c * A * x
    #         c_terms = self.mask_terms.get(oo, None)
    #         if c_terms is not None:
    #             ## c * Ax_effs
    #             for c_term in c_terms:
    #                 c_tens = c_term.get_evaluated_site(left_site_pos, nsites)
    #                 for tens_ in Ax_effs:
    #                     c_tens.transpose_like(tens_, inplace=True)
    #                     tens_.modify(apply = lambda x: x * c_tens.data)
    #
    #         b2k_dict = self.get_bra_to_ket_inds(left_site_pos, nsites, comp=oo)
    #         for Ax_eff in Ax_effs:
    #             Ax_eff.reindex(b2k_dict, inplace=True)
    #
    #         # ## check with cross TT
    #         # for it, eff_Ax in enumerate(Ax_effs):
    #         #     ket = self.states_dict[ii]
    #         #     ket_init = self.init_ket[ii]
    #         #     print('ket is', ket is term.ket, ket is term.vec_block.ket, ket is term.vec_block.bra)
    #         #     if len(term.op_blocks) > 0:
    #         #         print('ket is op', term.vec_block.bra is ket, term.op_blocks[0][0].ket is ket)
    #         #     print('check init', ket is ket_init, helper_quimb.distance(ket, ket_init))
    #         #
    #         #     ket = ket.copy()
    #         #     ket_init = ket_init.copy()
    #         #
    #         #     plt.figure()
    #         #     gtn = self.grid.make_gridTN(ket)
    #         #     x_data = gtn.get_data()
    #         #     plt.imshow(gtn.get_data())
    #         #     plt.title(f'x {ii}')
    #         #     plt.colorbar()
    #         #
    #         #     plt.figure()
    #         #     gtn = self.grid.make_gridTN(ket_init)
    #         #     plt.imshow(gtn.get_data() - x_data)
    #         #     plt.title(f'chk x {ii}')
    #         #     plt.colorbar()
    #         #
    #         #     Aket = helper_quimb.apply(terms[it].operators[0].copy(), ket.copy())
    #         #     plt.figure()
    #         #     gtn = self.grid.make_gridTN(Aket)
    #         #     plt.imshow(gtn.get_data())
    #         #     plt.title(f' Ax ideal')
    #         #     plt.colorbar()
    #         #
    #         #     ## project ideal output
    #         #     # bra = self.states_dict[oo].copy()
    #         #     bra = terms[it].bra.copy()
    #         #     bra = bra.conj()
    #         #     bra.mangle_inner_(append='_')
    #         #     # bra.site_ind_id = bra.site_ind_id + '_'
    #         #
    #         #     left_bra, right_bra = helper_cross.get_projector(bra, left_site_pos, 1)
    #         #     # bra_tensors = [bra[i] for i in range(left_site_pos)] + [bra[i] for i in range(left_site_pos + 1, bra.L)]
    #         #     tens_ = qtn.tensor_contract(*left_bra, *right_bra, *Aket.tensors)
    #         #     print('tens_', tens_)
    #         #     m = bra[left_site_pos]
    #         #     print('mbra m', m)
    #         #     m.modify(data=tens_.data, inds=tens_.inds)
    #         #     plt.figure()
    #         #     gtn = self.grid.make_gridTN(bra)
    #         #     plt.imshow(gtn.get_data())
    #         #     plt.title(f'X Ideal Proj Ax({oo},{ii})')
    #         #     plt.colorbar()
    #         #
    #         #     bra2 = bra.copy()
    #         #     bra2.site_ind_id = bra.site_ind_id + '_'
    #         #     left_bra, right_bra = helper_cross.get_projector(bra2, left_site_pos, 1)
    #         #     ket_tensors = [ket[i] for i in range(left_site_pos)] + [ket[i] for i in range(left_site_pos + 1, ket.L)]
    #         #     eff_op = qtn.TensorNetwork([*left_bra, *right_bra, *ket_tensors, terms[it].operators[0].tensors])
    #         #     eff_op_tens = eff_op.contract()
    #         #     ket_inds = self.ket_inds(left_site_pos, nsites)
    #         #     bra_inds = self.bra_inds(left_site_pos, nsites)
    #         #     eff_op_tens = eff_op_tens.transpose(*bra_inds, *ket_inds)
    #         #     # print('eff x op tens', eff_op_tens.data, eff_op_tens.inds)
    #         #
    #         #     # for ix in range(9, self.L-1):
    #         #     #     print('i', ix)
    #         #     #     _, right_bra = helper_cross.get_projector(bra2, ix, 1, get_left=False)
    #         #     #     ket_tensors = [ket[i] for i in range(ix + 1, ket.L)]
    #         #     #     op_tensors = [terms[it].operators[0][i] for i in range(ix + 1, ket.L)]
    #         #     #     env_r = qtn.TensorNetwork([*right_bra, *ket_tensors, *op_tensors])
    #         #     #     env_r_tens = env_r.contract()
    #         #     #     print('evn r (2)', env_r_tens.data, env_r_tens.inds, env_r.shape)
    #         #
    #         #     ket = self.states_dict[oo].copy()
    #         #     m = ket[left_site_pos]
    #         #     m.modify(data=eff_Ax.data, inds=eff_Ax.inds)
    #         #
    #         #     plt.figure()
    #         #     gtn = self.grid.make_gridTN(ket)
    #         #     plt.imshow(gtn.get_data())
    #         #     plt.title(f'X Proj Ax({oo},{ii})')
    #         #     plt.colorbar()
    #         #     plt.show()
    #
    #         if oo not in Ax_eff_dict:
    #             Ax_eff_dict[oo] = Ax_effs
    #         else:
    #             Ax_eff_dict[oo] += Ax_effs
    #
    #     for k in Ax_eff_dict.keys():
    #         Ax_eff_dict[k] = [helper_tn.sum_tens(Ax_eff_dict[k])]
    #
    #     return Ax_eff_dict
