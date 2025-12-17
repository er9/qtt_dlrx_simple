# import numpy as np
# import quimb
#
# from setup_.defaults import *
# import quimb.tensor as qtn
# from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
# import helper_quimb as helper
# import helper_dmrg
import helper_dmrg_2
from helper_tdvp_v2 import TDDMRGSolver_v2

from helper_dmrg import *

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gridTN import GridTN

""" implicit solver for Ax = b, where x, b are composed of multiple components
"""


class BaseSolver(Enum):
    LOCAL = LocalSolver
    DMRG = DMRGSolver
    TDDMRG = TDDMRGSolver_v2
    LINEAR = LinearSolver
    LINEAR2 = helper_dmrg_2.LinearSolver


def implicit_solver(ncomps: int, A_mats: dict[tuple[int, int], Sequence['MPO_type']],
                    b_vecs: dict[int, Sequence['MPS_type']],
                    init_guess: dict[int, 'qtn.MatrixProductState'] = None, nsites=2, verbose_output=False,
                    init_direction=SweepDirection.RIGHT,
                    **solver_kwargs):
    print('BLOCK DMRG 1')
    solver = BlockedLinearSolver(ncomps, init_guess, b_vecs, A_mats, **solver_kwargs)
    solver.solve(nsites, init_direction=init_direction)

    if verbose_output:
        return solver.kets, solver.err, solver.is_conv
    else:
        return solver.kets


########## helper functions ###########
# @profile
def combine_mat_tens(mat_dict: dict[(int, int), qtn.Tensor], keys_o: Sequence, keys_i: Sequence,
                     shapes_o: Sequence[int], shapes_i: Sequence[int]):
    """ all tensors are transposed to a certain order
    """
    tot_npts_o = np.sum(shapes_o)
    tot_npts_i = np.sum(shapes_i)

    tot_mat = np.zeros((tot_npts_o, tot_npts_i), dtype=complex)

    for k, mat in mat_dict.items():
        # print('combine mat', k, mat.inds, mat.norm())
        o_idx, i_idx = keys_o.index(k[0]), keys_i.index(k[1])
        start_ind_o, end_ind_o = int(np.sum(shapes_o[:o_idx])), int(np.sum(shapes_o[:o_idx + 1]))
        start_ind_i, end_ind_i = int(np.sum(shapes_i[:i_idx])), int(np.sum(shapes_i[:i_idx + 1]))
        # print('o idx', o_idx, 'i_idx', i_idx)
        # print('shapes o', shapes_o, 'shapes i', shapes_i)
        # print('inds o', start_ind_o, end_ind_o)
        # print('inds i', start_ind_i, end_ind_i)

        mat_data = mat.data.reshape(shapes_o[o_idx], shapes_i[i_idx])  # <x|A|x>
        # tot_mat[start_ind_o: end_ind_o, start_ind_i: end_ind_i] = \
        #     tot_mat[start_ind_o: end_ind_o, start_ind_i: end_ind_i] + mat_data
        tot_mat[start_ind_o: end_ind_o, start_ind_i: end_ind_i] = mat_data
        ### APPARENTLY IS SLOW STEP
        # plt.figure()
        # plt.imshow(mat_data)
        # plt.colorbar()
        # plt.show()

    return tot_mat


def extract_vec_tens(vec, keys, npts, ref_dict: dict[int, qtn.Tensor]):
    """ vector to dict of tensors
    # """
    # print('extract vec tens')
    # print('vec', vec.shape, npts)
    # print('keys', keys)
    vec_dict = {}
    for i in range(len(keys)):
        start_ind, end_ind = int(np.sum(npts[:i])), int(np.sum(npts[:i + 1]))
        new_x = ref_dict[keys[i]].copy()
        try:
            new_x_data = vec[start_ind:end_ind].reshape(new_x.shape)
            new_x.modify(data=new_x_data)
            vec_dict[keys[i]] = new_x
        except ValueError:
            pass    ## could pad with zeros

    return vec_dict


def combine_vec_tens(vec_dict: dict[int, qtn.Tensor], keys=None, tot_shape=None):
    """ all tensors are transposed to a certain order
    """
    if keys is None:
        keys = sorted(vec_dict.keys())

    # print('combine vec', keys)

    tot_vec = None
    for k in keys:
        if k not in vec_dict:  continue

        if tot_vec is None:
            tmp = vec_dict[k].data
            tot_vec = np.reshape(tmp, -1)
        else:
            tot_vec = np.append(tot_vec, np.reshape(vec_dict[k].data, -1))

    return tot_vec


##############################


class BlockedLocalSolver:
    """ DMRG solver with multiple components at once
    """

    def __init__(self, ncomps, trial_state: dict[int, 'qtn.MatrixProductState'],
                 targets: Optional[dict[int, Sequence['MPS_type']]] = None,
                 operators: Optional[dict[tuple[int, int], Sequence['MPO_type']]] = None,
                 operators_H: Optional[dict[tuple[int, int], Sequence['MPO_type']]] = None,
                 base_solver=BaseSolver.LOCAL,
                 mps_inds: Optional[Sequence[int]] = None,
                 # left_envs: Optional[dict[int,Sequence['Environment']]] = None,
                 # right_envs: Optional[dict[int,Sequence['Environment']]] = None,
                 virtual: bool = False, conv_tol=DEFAULT_CONV_TOL,
                 max_iter=DEFAULT_MAX_ITER, max_tot_iter=DEFAULT_MAX_TOT_ITER,
                 max_wrong_iter=DEFAULT_MAX_WRONG_ITER, max_bond=None, solvers=None,
                 **env_kwargs):
        """ each solver indexed by output ind, input ind (j, i)
                computes b_eff:  eg. d/dT*[s] <x_j | b_i>
                computes A_eff:  eg. d/dT*[s] d/dT[s] <x_j | O_ji | x_i>
                where x_j, operators_H don't have to be conj of x_i, operators
        """

        self.num_comps = ncomps
        if solvers is not None:  ## for copying
            pass

        else:
            solvers: dict[tuple[int, int], LocalSolver] = {}
            if trial_state is None:
                trial_state = {}
                # if targets is not None:
                #     trial_state = {k: bs[0].copy() for k, bs in targets.items()}
                # else:
                #     raise NotImplementedError

            for k, mpo in operators.items():
                print(k, mpo)

            if operators_H is None:
                operators_H = {(k[1], k[0]): [helper.mpo_conj_transpose(mpo) for mpo in mpos]
                               for k, mpos in operators.items()}
            ## bra is k[1], target it k[0]; operators_H are transpose of operators[k[0], k[1]]

            ref_init_x = targets[next(iter(targets))][0]
                # if trial_state is None else trial_state[next(iter(trial_state))]

            ## jank fix for MPS obtained from comb
            check_phys_inds = False
            if len(ref_init_x.shape) != ref_init_x.L:
                check_phys_inds = True

            # print('ref init x', ref_init_x)
            # print('site ind id', ref_init_x.site_ind_id)
            ref_op = operators[next(iter(operators))][0]
            for (oo, ii), As in operators.items():

                bs = targets.get(ii, None)

                # if bs is None and ii not in trial_state:
                #     continue

                init_x = trial_state.get(ii, None)
                if init_x is None:
                    # btmp = targets.get(oo, None)
                    # init_x = btmp[0].copy() if btmp is not None else \
                    #     qtn.MPS_rand_state(ref_init_x.L, ref_init_x.max_bond(), ref_init_x.phys_dim(1))
                    init_x = qtn.MPS_rand_state(ref_init_x.L, ref_init_x.max_bond(), ref_init_x.phys_dim(1))
                    helper.scalar_multiply(init_x, 0.01, inplace=True)

                    if check_phys_inds:
                        for mi in range(init_x.L):
                            if ref_init_x.site_ind(mi) not in ref_init_x[mi].inds:
                                init_x[mi].isel({init_x.site_ind(mi): 0}, inplace=True)
                        # print(init_x)
                    trial_state[ii] = init_x

                init_x2 = trial_state.get(oo, None)
                if init_x2 is None:
                    # btmp = targets.get(oo, None)
                    # init_x2 = btmp[0].copy() if btmp is not None else \
                    #     qtn.MPS_rand_state(ref_init_x.L, ref_init_x.max_bond(), ref_init_x.phys_dim(1))
                    init_x2 = qtn.MPS_rand_state(ref_init_x.L, ref_init_x.max_bond(), ref_init_x.phys_dim(1))
                    helper.scalar_multiply(init_x2, 0.01, inplace=True)
                    if check_phys_inds:
                        for mi in range(init_x2.L):
                            if ref_init_x.site_ind(mi) not in ref_init_x[mi].inds:
                                init_x2[mi].isel({init_x2.site_ind(mi): 0}, inplace=True)
                    trial_state[oo] = init_x2

                # init_x.expand_bond_dimension(max_bond_x, inplace=True)
                # print('max bond x', max_bond_x)
                # print('init_x', init_x.max_bond(), init_x)

                #### !!! match inner inds, bond dimensions of init_x !!! ####
                if ref_init_x is None:
                    ref_init_x = init_x
                else:
                    helper.match_inner_inds(init_x, ref_init_x, inplace=True)
                    init_x.site_ind_id = ref_init_x.site_ind_id

                    helper.match_inner_inds(init_x2, ref_init_x, inplace=True)
                    init_x2.site_ind_id = ref_init_x.site_ind_id

                # init_x.distribute_exponent()
                # for A in As:
                #     if A is not None:
                #         A.distribute_exponent()
                # if bs is not None:
                #     for b in bs:
                #         if b is not None:
                #             b.distribute_exponent()

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

                select_envs = {kw: (env.get(oo, None) if env is not None else None) for kw, env in env_kwargs.items()}
                ## input (ket) is the same if ii is the same, so envs only depend on oo

                ## set up solver
                init_bra = init_x2.conj()
                As_H = None if oo == ii else operators_H[(oo, ii)]
                # As_H = operators_H.get((oo, ii), None)
                # print('o,i', oo, ii, [A is not None for A in As])
                # if As_H is not None:  print([A is not None for A in As_H])
                solver = base_solver.value(init_x, bs, As, operators_H=As_H, bra_state=init_bra,  # virtual=virtual,
                                           in_ind=ii, out_ind=oo,
                                           mps_inds=mps_inds, **select_envs,
                                           conv_tol=conv_tol, max_iter=max_iter,
                                           max_tot_iter=max_tot_iter, max_bond=max_bond)

                solvers[(oo, ii)] = solver

        self.solvers = solvers
        self.err = np.inf
        self.is_conv = False

        self.conv_tol = conv_tol if conv_tol is not None else DEFAULT_CONV_TOL
        self.max_iter = max_iter
        self.max_tot_iter = max_tot_iter
        self.max_wrong_iter = max_wrong_iter
        self.max_bond = max_bond

    @property
    def L(self):
        k = next(iter(self.solvers))
        return len(self.solvers[k].mps_inds)  # self.mps_inds[1] - self.mps_inds[0]

    # @property
    # def max_bond(self):
    #     return np.max([solver.max_bond for k, solver in self.solvers.items()])

    @property
    def num_targets(self):
        k = next(iter(self.solvers))
        return len(self.solvers[k]._targets)

    @property
    def num_operators(self):
        k = next(iter(self.solvers))
        return len(self.solvers[k]._operators)

    @property
    def left_envs(self):
        all_left_envs = {}
        for k, solver in self.solvers.items():
            all_left_envs[k] = solver.left_envs
        return all_left_envs

    @property
    def right_envs(self):
        all_right_envs = {}
        for k, solver in self.solvers.items():
            all_right_envs[k] = solver.right_envs
        return all_right_envs

    @property
    def kets(self):
        all_kets = {}
        for (oo, ii), solver in self.solvers.items():
            all_kets[ii] = solver.ket
        return all_kets

    @property
    def ket(self):
        solver = self.solvers[next(iter(self.solvers))]
        return solver.ket

    @property
    def bra(self):
        solver = self.solvers[next(iter(self.solvers))]
        return solver.bra

    @property
    def component_ind(self):
        return '_comp_ind_'

    def create_like(self, copy=True, **kwargs):
        new_solvers = {}
        for k, solver in self.solvers.items():
            new_solvers[k] = solver.create_like(copy=copy, **kwargs)
        return self.__class__(self.num_comps, None, solvers=new_solvers)

    def copy(self, deep=True):
        new_solvers = {}
        for k, solver in self.solvers.items():
            new_solvers[k] = solver.copy(deep=deep)
        new_block_solver = self.__class__(self.num_comps, None, solvers=new_solvers)
        new_block_solver.err = self.err
        new_block_solver.is_conv = self.is_conv
        return new_block_solver

    def reinitialize_envs(self):
        """ if boundary envs are None, uses old values
        """
        for k, solver in self.solvers.items():
            solver.reinitialize_envs()

        self.err = np.inf
        self.is_conv = False

    def reinitialize_envs_i(self, i):
        """ if boundary envs are None, uses old values
        """
        for k, solver in self.solvers.items():
            solver.reinitialize_envs_i(i)

    def get_ket_to_bra_inds(self, i) -> dict[str, str]:
        """ inds mapping inds on self.bra to corresponding inds on self.ket
        """
        solver = self.solvers[next(iter(self.solvers))]
        return solver.get_ket_to_bra_inds(i)

    def get_bra_to_ket_inds(self, i) -> dict[str, str]:
        ket_to_bra = self.get_ket_to_bra_inds(i)
        bra_to_ket = {item: k for k, item in ket_to_bra.items()}
        return bra_to_ket

    def _modify_solver_ket_and_bra(self, ix, func, mod_sites: Union[Sequence[int], int], *args, keyed_kwargs=None,
                                   **kwargs):
        ## diag solvers
        for k, solver in self.solvers.items():
            if k[0] == k[1]:
                keyed_kwarg = {kw: keyed_dict[k] for kw, keyed_dict in keyed_kwargs.items()
                               } if keyed_kwargs is not None else {}
                func(solver, ix, *args, **keyed_kwarg, **kwargs)

        for k, solver in self.solvers.items():
            out_ind, in_ind = k
            if out_ind == in_ind:
                continue

            ref_k_solver = self.solvers[in_ind, in_ind]
            ref_b_solver = self.solvers[out_ind, out_ind]
            # print('ref k exp', ref_k_solver.ket.exponent, ref_k_solver.bra.exponent)
            # print('ref b exp', ref_b_solver.ket.exponent, ref_b_solver.bra.exponent)
            if ix is all:
                solver._ket = ref_k_solver.ket
                solver._bra = ref_b_solver.bra
                solver.reinitialize_envs()
            else:
                inds = self.get_mps_ind(mod_sites)
                if isinstance(inds, int):  inds = [inds]
                solver.update_bra_tens([ref_b_solver.bra[ind] for ind in inds], sites=mod_sites,
                                       new_exponent=ref_b_solver.bra.exponent)
                solver.update_ket_tens([ref_k_solver.ket[ind] for ind in inds], sites=mod_sites,
                                       new_exponent=ref_k_solver.ket.exponent)

                if isinstance(mod_sites, int):
                    solver.reinitialize_envs_i(mod_sites)
                else:
                    for ix in mod_sites:
                        solver.reinitialize_envs_i(ix)

                # print('sovler exp', solver.ket.exponent, solver.bra.exponent)
                # print([env.ket is solver.ket for env in solver.left_envs])
                # print([env.ket is solver.ket for env in solver.right_envs])
                # print([env.bra is solver.bra for env in solver.left_envs])
                # print([env.bra is solver.bra for env in solver.right_envs])
        return

    def canonize(self, i, cur_orthog=None):

        if cur_orthog is None:
            mod_sites = list(range(0, self.L))
        else:
            mod_sites = list(range(i, cur_orthog)) if i < cur_orthog else list(range(cur_orthog, i))

        self._modify_solver_ket_and_bra(i, func_canonize, mod_sites, cur_orthog=cur_orthog)
        # for k, solver in self.solvers.items():
        #     solver.canonize(i, cur_orthog=cur_orthog)
        return

    def compress(self, i, cur_orthog=None, compress_opts=None):

        if cur_orthog is None:
            mod_sites = list(range(0, self.L))
        else:
            mod_sites = list(range(i, cur_orthog)) if i < cur_orthog else list(range(cur_orthog, i))

        self._modify_solver_ket_and_bra(i, func_compress, mod_sites, cur_orthog=cur_orthog, compress_opts=compress_opts)
        # for k, solver in self.solvers.items():
        #     solver.compress(i, cur_orthog=cur_orthog, compress_opts=compress_opts)
        return

    def left_canonize_site(self, i):
        mod_sites = [i, i + 2]
        self._modify_solver_ket_and_bra(i, func_left_canonize_site, mod_sites)
        # for k, solver in self.solvers.items():
        #     solver.left_canonize_site(i)
        return

    def right_canonize_site(self, i):
        mod_sites = [i - 1, i + 1]
        self._modify_solver_ket_and_bra(i, func_right_canonize_site, mod_sites)
        # for k, solver in self.solvers.items():
        #     solver.right_canonize_site(i)
        return

    def add_rand(self, canon_direction, strength=0.01):
        """ canon direction:  sweep direction left/right -> right/left canon
        """
        mod_sites = list(range(self.L))
        self._modify_solver_ket_and_bra(all, func_add_rand, mod_sites, canon_direction, strength=strength)

    def get_mps_ind(self, i: Union[int, Sequence[int], slice]):
        solver = self.solvers[next(iter(self.solvers))]
        if isinstance(i, (int, slice)):
            return solver.mps_inds[i]
        else:
            return [solver.mps_inds[ii] for ii in i]

    #########################

    def _update_1site(self, i: int, site_i: dict[int, 'qtn.Tensor'], direction: 'SweepDirection'):
        """ update ket
            i: int of mps site
            site_i:  has an extra ind indexing the component
        """

        if direction == SweepDirection.RIGHT.value:
            mod_sites = [i, i + 1] if i < self.L else [i]
        else:
            mod_sites = [i - 1, i] if i > 0 else [i]

        site_i = {(i, i): tens for i, tens in site_i.items()}  ## to be compatbile with func below
        self._modify_solver_ket_and_bra(i, func_update_1site, mod_sites, direction=direction,
                                        keyed_kwargs={'site_i', site_i})

        # ## first update x_i / x_i solvers
        # for key, solver in self.solvers.items():
        #     out_ind, in_ind = key
        #     if out_ind == in_ind:
        #         site_i_k = site_i[in_ind]
        #         solver._update_1site(i, site_i_k, direction=direction)
        #
        # ## update cross solvers
        # for key, solver in self.solvers.items():
        #     out_ind, in_ind = key
        #     if out_ind == in_ind:
        #         continue
        #     ref_k_solver = self.solvers[in_ind, in_ind]
        #     ref_b_solver = self.solvers[out_ind, out_ind]
        #     if direction == SweepDirection.RIGHT.value:
        #         if i < self.L:
        #             solver.update_bra_tens(ref_b_solver.bra[i: i + 2], sites=[i, i + 1],
        #                                    new_exponent=ref_b_solver.bra.exponent)
        #             solver.update_ket_tens(ref_k_solver.ket[i: i + 2], sites=[i, i + 1],
        #                                    new_exponent=ref_k_solver.ket.exponent)
        #         else:
        #             solver.update_bra_tens(ref_b_solver.bra[i: i + 1], sites=[i],
        #                                    new_exponent=ref_b_solver.bra.exponent)
        #             solver.update_ket_tens(ref_k_solver.ket[i: i + 1], sites=[i],
        #                                    new_exponent=ref_k_solver.ket.exponent)
        #     else:
        #         if i > 0:
        #             solver.update_bra_tens(ref_b_solver.bra[i - 1: i + 1], sites=[i - 1, i],
        #                                    new_exponent=ref_b_solver.bra.exponent)
        #             solver.update_ket_tens(ref_k_solver.ket[i - 1: i + 1], sites=[i - 1, i],
        #                                    new_exponent=ref_k_solver.ket.exponent)
        #         else:
        #             solver.update_bra_tens(ref_b_solver.bra[i: i + 1], sites=[i],
        #                                    new_exponent=ref_b_solver.bra.exponent)
        #             solver.update_ket_tens(ref_k_solver.ket[i: i + 1], sites=[i],
        #                                    new_exponent=ref_k_solver.ket.exponent)

        return

    def _update_2site(self, i: int, site_i: dict[int, 'qtn.Tensor'], direction: 'SweepDirection'):
        """ update ket, bra with new_site
            i: mps_site
            site_i:  has an extra ind indexing the component
        """
        if direction == SweepDirection.RIGHT.value:
            mod_sites = [i, i + 1] if i < self.L else [i]
        else:
            mod_sites = [i - 1, i] if i > 0 else [i]

        site_i = {(i, i): tens for i, tens in site_i.items()}  ## to be compatbile with func below
        self._modify_solver_ket_and_bra(i, func_update_2site, mod_sites, direction=direction,
                                        keyed_kwargs={'site_i': site_i})

        # ## first update x_i / x_i solvers
        # for key, solver in self.solvers.items():
        #     out_ind, in_ind = key
        #     if out_ind == in_ind:
        #         site_i_k = site_i[in_ind]
        #         print('solver max bond', solver.max_bond)
        #         solver._update_2site(i, site_i_k, direction=direction)
        #
        # ## update cross solver with new kets/bras
        # for key, solver in self.solvers.items():
        #
        #     out_ind, in_ind = key
        #     if out_ind == in_ind:
        #         continue
        #     print('in_ind', in_ind, out_ind)
        #     ref_k_solver = self.solvers[in_ind, in_ind]
        #     ref_b_solver = self.solvers[out_ind, out_ind]
        #
        #     if key == (3, 2):
        #         print('init!!')
        #         print('ref k solver', in_ind, ref_k_solver.ket)
        #         print('ref b solver', out_ind, ref_b_solver.bra)
        #         print('updated ket', solver.ket)
        #         print('updated bra', solver.bra)
        #
        #     if direction == SweepDirection.RIGHT.value:
        #         print('update sites', i, i + 1, 'direction', direction)
        #         solver.update_bra_tens(ref_b_solver.bra[i: i + 2], sites=[i, i + 1],
        #                                new_exponent=ref_b_solver.bra.exponent)
        #         solver.update_ket_tens(ref_k_solver.ket[i: i + 2], sites=[i, i + 1],
        #                                new_exponent=ref_k_solver.ket.exponent)
        #         for ix in [i, i + 1]:
        #             solver.reinitialize_envs_i(ix)
        #     else:
        #         print('update sites', i - 1, i, 'direction', direction)
        #         solver.update_bra_tens(ref_b_solver.bra[i - 1: i + 1], sites=[i - 1, i],
        #                                new_exponent=ref_b_solver.bra.exponent)
        #         solver.update_ket_tens(ref_k_solver.ket[i - 1: i + 1], sites=[i - 1, i],
        #                                new_exponent=ref_k_solver.ket.exponent)
        #
        #         for ix in [i, i - 1]:
        #             solver.reinitialize_envs_i(ix)
        #
        #     if key == (3, 2):
        #         print('ref k solver', ref_k_solver.ket)
        #         print('ref b solver', ref_b_solver.bra)
        #         print('updated ket', solver.ket)
        #         print('updated bra', solver.bra)

        return

    def _get_env_func(self, env_type):
        raise NotImplementedError

    def _build_all_envs_right(self, nsites: int, end: int = None, canonize=True):
        """ compute all right envs for <Ax|b>
        """
        for key, solver in self.solvers.items():
            solver._build_all_envs_right(nsites, end=end, canonize=canonize)
        return

    def _build_all_envs_left(self, nsites: int, end: int = None, canonize=True):
        """ compute all left envs for <Ax|b>
        """
        for key, solver in self.solvers.items():
            solver._build_all_envs_left(nsites, end=end, canonize=canonize)
        return

    def _update_envs_left(self, pos: int, canonize=False):
        for key, solver in self.solvers.items():
            solver._update_envs_left(pos, canonize=canonize)
        return

    def _update_envs_right(self, pos: int, canonize=False):
        for key, solver in self.solvers.items():
            solver._update_envs_right(pos, canonize=canonize)
        return

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    def check_err(self, *args, **kwargs) -> 'float':
        """ also updates self.err, self.is_conv
        """
        raise NotImplementedError


class BlockedLinearSolver(BlockedLocalSolver):
    def __init__(self, ncomps, trial_state: dict[int, 'qtn.MatrixProductState'],
                 targets: Optional[dict[int, Sequence['MPS_type']]] = None,
                 operators: Optional[dict[tuple[int, int], Sequence['MPO_type']]] = None,
                 is_H=True,
                 mps_inds: Sequence[int] = None,
                 norm_env0_Ls: dict[int, Sequence['qtn.Tensor']] = None,
                 norm_env0_Rs: dict[int, Sequence['qtn.Tensor']] = None,
                 ovlp_env0_Ls: dict[tuple[int, int], Sequence['qtn.Tensor']] = None,
                 ovlp_env0_Rs: dict[tuple[int, int], Sequence['qtn.Tensor']] = None,
                 err_env0_Ls: dict[int, Sequence['qtn.Tensor']] = None,
                 err_env0_Rs: dict[int, Sequence['qtn.Tensor']] = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER,
                 max_tot_iter=DEFAULT_MAX_TOT_ITER, max_wrong_iter=DEFAULT_MAX_WRONG_ITER,
                 max_bond=None, solve_type: SolveMethod = DEFAULT_SOLVE,
                 solvers=None,
                 ):

        # operators = [] if operators is None else list(operators)
        self.A_envsLs, self.A_envsRs = [], []
        self.b_envsLs, self.b_envsRs = [], []

        init_state = trial_state
        super().__init__(ncomps, init_state, targets=targets, operators=operators, mps_inds=mps_inds, conv_tol=conv_tol,
                         max_iter=max_iter, max_bond=max_bond, max_tot_iter=max_tot_iter, max_wrong_iter=max_wrong_iter,
                         base_solver=BaseSolver.LINEAR,
                         norm_env0_Rs=norm_env0_Rs, norm_env0_Ls=norm_env0_Ls,
                         ovlp_env0_Rs=ovlp_env0_Rs, ovlp_env0_Ls=ovlp_env0_Ls,
                         err_env0_Rs=err_env0_Rs, err_env0_Ls=err_env0_Ls,
                         solvers=solvers)

        for k, solver in self.solvers.items():
            solver.is_H = False  # is_H
            solver.solve_type = solve_type

        self.solve_type = solve_type
        self.is_H = False  # is_H

    combined_ind_o = 'out_ind'
    combined_ind_i = 'in_ind'

    @classmethod
    # @profile
    def compute_Ax(cls, A_effs_dict_: dict[tuple[int, int], Sequence['qtn.TensorNetwork']], x_vec_tens_: 'qtn.Tensor',
                   ref_x_dict: dict[int, 'qtn.Tensor'] = None, ref_b_eff_dict: dict[int, 'qtn.Tensor'] = None,
                   out_keys: Sequence = (), bra_inds: Sequence = (), out_shape: Sequence = (),
                   in_keys: Sequence = (), ket_inds: Sequence = (), in_shape: Sequence = (),
                   out_inds=('out_ind',), **kwargs) -> 'qtn.Tensor':

        Ax_dict = {}
        input_is_ket = out_inds == (cls.combined_ind_o,)
        if input_is_ket:
            out_keys_ = out_keys  # b key ordering
            bra_inds_ = bra_inds  # out = output inds
            x_dict_ = extract_vec_tens(x_vec_tens_.data, in_keys, in_shape, ref_x_dict)
            ## tens have ket_inds
        else:
            out_keys_ = out_keys  # b key ordering
            bra_inds_ = ket_inds  # out = input inds
            x_dict_ = extract_vec_tens(x_vec_tens_.data, in_keys, in_shape, ref_b_eff_dict)
            ## tens have bra_inds

        for (ooo, iii), A_eff_tns in A_effs_dict_.items():
            tot_A_eff = None
            ix_, ix_c = (ooo, iii) if input_is_ket else (iii, ooo)  ## out_index, contracted index

            ## sum over A_eff|x> for A_eff in list of A_eff_tns
            for A_eff_tn in A_eff_tns:
                Ax_tens = qtn.tensor_contract(*A_eff_tn.tensors, x_dict_[ix_c])
                Ax_tens.modify(apply=lambda x: x * 10 ** A_eff_tn.exponent)
                if tot_A_eff is None:
                    tot_A_eff = Ax_tens.transpose(*bra_inds_)
                else:
                    Ax_tens.transpose_like(tot_A_eff, inplace=True)
                    tot_A_eff.modify(apply=lambda x: x + Ax_tens.data)

            ### sum over ix_c
            if ix_ not in Ax_dict:
                Ax_dict[ix_] = tot_A_eff
            else:
                Ax_dict[ix_].modify(apply=lambda x: x + tot_A_eff.data)

        Ax_vec = combine_vec_tens(Ax_dict, keys=out_keys_)
        Ax_vec_tens = qtn.Tensor(data=Ax_vec, inds=out_inds)
        return Ax_vec_tens

    @classmethod
    def compute_xAx(cls, A_effs_dict_: dict[tuple[int, int], Sequence['qtn.TensorNetwork']],
                    x_vec_tens_i: 'qtn.Tensor', x_vec_tens_o: 'qtn.Tensor',
                    ref_x_dict: dict[int, 'qtn.Tensor'] = None, ref_b_eff_dict: dict[int, 'qtn.Tensor'] = None,
                    out_keys: Sequence = (), bra_inds: Sequence = (), out_shape: Sequence = (),
                    in_keys: Sequence = (), ket_inds: Sequence = (), in_shape: Sequence = (),
                    precomp_Ax=None, **kwargs) -> Numeric:

        x_dict_i = extract_vec_tens(x_vec_tens_i.data, in_keys, in_shape, ref_x_dict)  # in inds
        x_dict_o = extract_vec_tens(x_vec_tens_o.data, in_keys, in_shape, ref_b_eff_dict)  # out inds
        tot_xAx = 0.0
        if precomp_Ax is None:
            for (ooo, iii), A_eff_tns in A_effs_dict_.items():
                for A_eff_tn in A_eff_tns:
                    # print('x', x_dict_i[iii], x_dict_o[ooo])
                    out = qtn.tensor_contract(*A_eff_tn.tensors, x_dict_i[iii], x_dict_o[ooo])
                    out *= 10 ** A_eff_tn.exponent
                    tot_xAx += out
        else:
            precomp_Ax_dict = extract_vec_tens(precomp_Ax.data, in_keys, in_shape, ref_b_eff_dict)  # out inds
            for ooo in out_keys:
                out = qtn.tensor_contract(precomp_Ax_dict[ooo], x_dict_o[ooo])
                tot_xAx += out

        return tot_xAx

    def _get_b_eff_dict_(self, sites, bra_inds):
        """
        BL * B * BR = d/dT*[s] <x_j|b_i> d_ij
        dict indexes xj
        """
        b_eff_dict = {}
        ### d/dT*[s] <x_j| A_ji.T | b_i>
        for key, solver in self.solvers.items():
            if key[0] == key[1]:
                out = solver._get_b_eff(sites)
                if out is not None:
                    out.transpose(*bra_inds, inplace=True)
                    b_eff_dict[key[0]] = out
                else:
                    shape = [solver.bra.ind_size(b_ind) for b_ind in bra_inds]
                    # print('shape?', shape)
                    # exit()
                    b_eff_dict[key[0]] = qtn.Tensor(np.zeros(tuple(shape)), inds=bra_inds)
                # print('b eff size', key[0], b_eff_dict[key[0]].shape)
        return b_eff_dict

    # @profile
    def _site_solve(self, sites: Union[int, slice], ) -> dict[int, qtn.Tensor]:
        """
        sites: int or slice(start, stop, step)
        solve local A' x = b',  (1/2?) <x|Ax> - <x|b> = 0
            where A' = d/dT[i] d/dT[i]^* <x|A|x> = d/dT[i] d/dT[i]^* \sum_ij <x_j|A_ji|x_i>
            where b' = d/dT[i]^* <x|b> = d/dT[i] \sum_ij <x_j|b_i>
        """
        # print("SITE SOLVE", self.solve_type, sites)
        site_pos = list(range(self.L))[sites]
        site_inds = self.get_mps_ind(sites)

        ## combine As, bs
        start = np.min(site_pos) if isinstance(site_pos, list) else site_pos
        end = np.max(site_pos) if isinstance(site_pos, list) else site_pos

        bra_inds = []  # [self.component_ind + '_']
        if start > 0:
            bra_inds += [self.bra.bond(start - 1, start)]
        if end < self.L - 1:
            bra_inds += [self.bra.bond(end, end + 1)]

        ## JANK fix for comb branch
        bra_inds += [self.bra.site_ind_id.format(i) for i in site_inds
                     if self.bra.site_ind_id.format(i) in self.bra[i].inds]

        bra_to_ket_inds = {self.component_ind + '_': self.component_ind}
        for site_p in site_pos:
            bra_to_ket_inds.update(self.get_bra_to_ket_inds(site_p))
        ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]

        old_x_shape = {}
        for ki, ket in self.kets.items():
            ind_size_dict = {}
            if start > 0:
                ind_size_dict[ket.bond(start - 1, start)] = ket.bond_size(start - 1, start)
            if end < self.L - 1:
                ind_size_dict[ket.bond(end, end + 1)] = ket.bond_size(end, end + 1)
            old_x_shape[ki] = ind_size_dict

        A_effs_dict, b_eff_dict = {}, {}  ## includes exponents (bra + ket + operator)
        for key, solver in self.solvers.items():
            ### AL * A * AR = d/dT*[i] d/dT[i] <x_j|A_ji|x_i>
            A_effs_dict[key] = solver._get_A_effs(sites, return_sum=(not self.solve_type is SolveMethod.CGD))
            if self.solve_type is not SolveMethod.CGD:
                A_effs_dict[key].transpose(*bra_inds, *ket_inds, inplace=True)

            # ### BL * B * BR = d/dT*[s] <x_j|b_i> d_ij
            # ### d/dT*[s] <x_j| A_ji.T | b_i>
            # if key[0] == key[1]:
            #     print('get b eff', key)
            #     b_eff_dict[key[0]] = solver._get_b_eff(sites)
            #     b_eff_dict[key[0]].transpose(*bra_inds, inplace=True)
            #     # print('b eff size', key[0], b_eff_dict[key[0]].shape)
        b_eff_dict = self._get_b_eff_dict_(sites, bra_inds)

        x_dict = {}
        for k, ket in self.kets.items():
            if isinstance(site_inds, (list, tuple)):
                ket_tensors = [ket[si] for si in site_inds]
            else:
                ket_tensors = ket[site_inds]
            x_tens = qtn.tensor_contract(*ket_tensors)
            x_tens.transpose(*ket_inds, inplace=True)
            x_dict[k] = x_tens

        ## solve for T[i]
        new_x_dict = {}
        if len(A_effs_dict) == 0:
            for ik in b_eff_dict.keys():
                x_eff = b_eff_dict[ik].reindex(bra_to_ket_inds)
                x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent - self.bra.exponent))
                new_x_dict[ik] = x_eff
            # x_eff = combined_b_eff.reindex(bra_to_ket_inds)
            # x_eff.modify(apply=lambda x: x * 10 ** (-self.ket.exponent - self.bra.exponent))
            ## b_eff_exponent already in b_eff tensor
        else:
            if self.solve_type is SolveMethod.CGD:

                in_keys = list(x_dict.keys())
                out_keys = list(b_eff_dict.keys())
                in_keys.sort(), out_keys.sort()
                # in_keys = out_keys = list(range(6))
                in_shape = [x_dict[k].size for k in in_keys]
                out_shape = [b_eff_dict[k].size for k in out_keys]

                x_vec = combine_vec_tens(x_dict, keys=in_keys)
                b_vec = combine_vec_tens(b_eff_dict, keys=out_keys)

                combined_ind_i, combined_ind_o = self.__class__.combined_ind_i, self.__class__.combined_ind_o
                x_vec_tens = qtn.Tensor(data=x_vec, inds=(combined_ind_i,))
                b_vec_tens = qtn.Tensor(data=b_vec, inds=(combined_ind_o,))

                def local_compute_Ax(A_effs_dict_: dict[tuple[int, int], Sequence['qtn.TensorNetwork']],
                                     x_vec_tens_: 'qtn.Tensor', out_inds, **kwargs) -> 'qtn.Tensor':

                    return self.__class__.compute_Ax(A_effs_dict_, x_vec_tens_,
                                                     ref_x_dict=x_dict, ref_b_eff_dict=b_eff_dict,
                                                     out_keys=out_keys, bra_inds=bra_inds, out_shape=out_shape,
                                                     in_keys=in_keys, ket_inds=ket_inds, in_shape=in_shape,
                                                     out_inds=out_inds, **kwargs)

                def local_compute_xAx(A_effs_dict_: dict[tuple[int, int], Sequence['qtn.TensorNetwork']],
                                      x_vec_tens_i: 'qtn.Tensor', x_vec_tens_o: 'qtn.Tensor', out_inds,
                                      precomp_Ax=None, **kwargs) -> 'qtn.Tensor':

                    return self.__class__.compute_xAx(A_effs_dict_, x_vec_tens_i, x_vec_tens_o,
                                                      ref_x_dict=x_dict, ref_b_eff_dict=b_eff_dict,
                                                      out_keys=out_keys, bra_inds=bra_inds, out_shape=out_shape,
                                                      in_keys=in_keys, ket_inds=ket_inds, in_shape=in_shape,
                                                      out_inds=out_inds, precomp_Ax=precomp_Ax, **kwargs)

                bra_to_ket_inds_tmp = {**bra_to_ket_inds, combined_ind_o: combined_ind_i}
                if self.is_H:
                    x_eff, error = qtn_conjugate_gradient_descent_1site(A_effs_dict, b_vec_tens, x_vec_tens,
                                                                        bra_to_ket_inds_tmp,
                                                                        contract_Ax=local_compute_Ax,
                                                                        contract_xAx=local_compute_xAx)
                else:
                    x_eff, error = qtn_conjugate_gradient_squared_1site(A_effs_dict, b_vec_tens, x_vec_tens,
                                                                        bra_to_ket_inds_tmp,
                                                                        contract_Ax=local_compute_Ax,
                                                                        contract_xAx=local_compute_xAx)
                # x_eff, error = qtn_conjugate_gradient_descent_1site(A_eff, b_eff, None, bra_to_ket_inds)
                # x_eff.modify(apply=lambda x: x * 10 ** (-A_eff_exponent))
                new_x_dict = extract_vec_tens(x_eff.data, in_keys, in_shape, x_dict)

            else:  # self.solve_type is SolveMethod.CGDx:

                in_keys = list(x_dict.keys())
                out_keys = list(b_eff_dict.keys())
                in_keys.sort(), out_keys.sort()
                in_shape = [x_dict[k].size for k in in_keys]
                out_shape = [b_eff_dict[k].size for k in out_keys]

                x_vec = combine_vec_tens(x_dict, keys=in_keys)
                b_vec = combine_vec_tens(b_eff_dict, keys=out_keys)

                # print('in keys', in_keys, 'out keys', out_keys)
                A_mat = combine_mat_tens(A_effs_dict, out_keys, in_keys, out_shape, in_shape)
                # print('A is H?', np.linalg.norm(A_mat - A_mat.conj().T))

                if self.solve_type is SolveMethod.CGDx:
                    if self.is_H:
                        x_eff_data, error = conjugate_gradient_descent(A_mat, b_vec, x=x_vec)
                    else:
                        x_eff_data, error = conjugate_gradient_squared(A_mat, b_vec, x=x_vec)

                    new_x_dict = extract_vec_tens(x_eff_data, in_keys, in_shape, x_dict)
                    # exit()
                else:  ## ALS optimzation
                    x_eff_data = np.linalg.solve(A_mat, b_vec)
                    new_x_dict = extract_vec_tens(x_eff_data, in_keys, in_shape, x_dict)

        return new_x_dict

    # @profile
    def solve(self, nsites: int, init_direction=SweepDirection.RIGHT, conv_tol=0,
              skip_inds: set = None, opt_inds: set = None, verbose=False, **kwargs):
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
        conv_tol = self.conv_tol if conv_tol == 0 else conv_tol
        L = self.ket.L
        # print('L', self.L, self.ket.L)
        canon_site = 0 if init_direction == SweepDirection.RIGHT else L - 1
        self.canonize(canon_site)

        ## ovlp <Ax|b>, <Ax|x> init envs
        if init_direction == SweepDirection.RIGHT:
            self._build_all_envs_right(nsites, canonize=False)
        else:
            self._build_all_envs_left(nsites, canonize=False)

        ## iterative solver
        it = 0
        print('dmrg self.max_bond', self.max_bond, self.ket.max_bond())
        if self.max_bond is None or self.ket.max_bond() < self.max_bond:
            print('getting error')
            err = self.check_err(dense=True)  ## true is actually faster for large MPO bond dim?
            print('init error dmrg', err, self.err)
        else:
            err = self.err

        conv_it, prev_err = 0, err
        num_wrong_it = 0

        # min_ket, min_err = self.ket, err
        min_solver, min_err = self.copy(), err
        direction = init_direction
        while err > conv_tol and conv_it < self.max_iter \
                and it < self.max_tot_iter and num_wrong_it < self.max_wrong_iter:

            it += 1

            ## right sweep
            if direction == SweepDirection.RIGHT:

                for i in range(L - nsites + 1):
                    ix = self.get_mps_ind(i)
                    inds = slice(i, i + nsites)
                    # print('update', i, inds)

                    if skip_inds is not None:
                        if skip_inds.issubset({ix + iix for iix in range(nsites)}):
                            self.left_canonize_site(i)  # now centered at i+1
                            self._update_envs_left(i, canonize=False)
                            continue

                    if opt_inds is not None:
                        # print('opt inds', i, opt_inds, self.L)
                        # print(opt_inds.intersection({ix + iix for iix in range(nsites)}))
                        if not opt_inds.intersection({self.get_mps_ind(i + iix) for iix in range(nsites)}):
                            # print('opt', i, ix, {self.get_mps_ind(i + iix) for iix in range(nsites)}, opt_inds)
                            self.left_canonize_site(i)  # now centered at i+1
                            self._update_envs_left(i, canonize=False)
                            continue

                    # print('update L', inds)
                    site_i = self._site_solve(inds)

                    ## update ket (+ canonicalization)
                    if nsites == 1:
                        # print('LR one site', site_i)
                        self._update_1site(i, site_i, direction)
                    elif nsites == 2:
                        # print('LR two site', site_i)
                        self._update_2site(i, site_i, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (left)
                    self._update_envs_left(i, canonize=False)
                    # print('err', i, self.check_err())

                self.canon_direction = EnvironmentSide.LEFT
                err = self.check_err(dense=True)

            else:
                ## left sweep
                for i in range(L - 1, nsites - 2, -1):
                    ix = self.get_mps_ind(i)
                    inds = slice(i - nsites + 1, i + 1)
                    # print('update', i, inds)

                    if skip_inds is not None:
                        if skip_inds.issubset({self.get_mps_ind(i - iix) for iix in range(nsites)}):
                            self.right_canonize_site(i)  # now centered at i-1
                            self._update_envs_right(i, canonize=False)
                            continue

                    if opt_inds is not None:
                        # print('opt inds', i, opt_inds)
                        # print(opt_inds.intersection({ix - iix for iix in range(nsites)}))
                        if not opt_inds.intersection({ix - iix for iix in range(nsites)}):
                            self.right_canonize_site(i)  # now centered at i-1
                            self._update_envs_right(i, canonize=False)
                            continue

                    # print('update R', inds)
                    site_i = self._site_solve(inds)

                    ## update ket, bra
                    if nsites == 1:
                        # print('RL one site', site_i)
                        self._update_1site(i, site_i, direction)
                    elif nsites == 2:
                        # print('RL two site', site_i)
                        self._update_2site(i, site_i, direction)
                    else:
                        raise NotImplementedError

                    ## update A, b envs (right)
                    self._update_envs_right(i, canonize=False)
                    # print('err', i, self.check_err())

                self.canon_direction = EnvironmentSide.RIGHT
                err = self.check_err(dense=True)

            if verbose:
                print('err', it, err)
                # print('err np', self.check_err_np())

            if np.abs((prev_err - err) / err) < 1.0e-4:
                conv_it += 1

            # if err > prev_err:
            #     ## add noise to ket
            #     print('adding noise LR?')
            #     self.add_rand(direction, strength=0.2)
            #     if direction == SweepDirection.RIGHT:  # was sweeping L to R
            #         # self.canonize(self.L - 1)
            #         self._build_all_envs_left(nsites, canonize=False)
            #     else:
            #         # self.canonize(0)
            #         self._build_all_envs_right(nsites, canonize=False)
            #     err = self.check_err()
            #     print('new err', err)

            prev_err = err

            if err < min_err:
                min_solver = self.copy(deep=True)
                ### COPY IS KIND OF EXPENSIVE
                # min_ket = self.ket.copy()
                min_err = err
                num_wrong_it = 0
            else:
                print('Warning: solve error went up', err, min_err)
                num_wrong_it += 1

            direction *= -1  ## swaps sweep direction

        # conv = min_err < conv_tol

        ## revert to optimal results
        if err > min_err:
            self.solvers = {k: solver.copy() for k, solver in min_solver.solvers.items()}
            self.err = min_solver.err
            self.is_conv = min_solver.is_conv
            # self._ket = min_solver.ket
            # self._bra = min_solver.bra
            # self.A_envsRs = min_solver.A_envsRs
            # self.A_envsLs = min_solver.A_envsLs
            # self.b_envsLs = min_solver.b_envsLs
            # self.b_envsRs = min_solver.b_envsRs
            # self.err = min_solver.err
            # self.is_conv = min_solver.is_conv
            # self.canon_direction = min_solver.canon_direction

        # print(self.kets)
        return self.kets, self.err, self.is_conv

    def solve_1site(self, **solve_kwargs):
        return self.solve(1, **solve_kwargs)

    def solve_2site(self, **solve_kwargs):
        return self.solve(2, **solve_kwargs)

    def check_err_np(self):
        """ check err using numpy """
        As, bs, xs = {}, {}, {}
        iis, oos = [], []
        for (oo, ii), solver in self.solvers.items():
            if self.num_operators > 0:
                # A = np.zeros((2 ** self.L, 2 ** self.L))
                if solver.num_operators == 1:
                    op = solver.operators[0]
                else:
                    op = solver.operators[oo]  # solver.in_ind

                op_tens = op.contract(all) * 10 ** op.exponent
                op_tens.transpose(*[op.upper_ind_id.format(i) for i in solver.mps_inds],
                                  *[op.lower_ind_id.format(i) for i in solver.mps_inds],
                                  inplace=True)
                op_mat = op_tens.data.reshape(2 ** self.L, 2 ** self.L)
                # A = np.dot(op_mat, A)
                A = op_mat
            else:
                A = np.eye(2 ** self.L)
            As[oo, ii] = A

            if oo == ii:
                b = np.zeros(2 ** self.L)
                for target in solver.vecs:
                    target_tens = target.contract(all) * 10 ** target.exponent
                    # print('target tens?', target_tens)
                    target_tens.transpose(*[target.site_ind_id.format(i) for i in solver.mps_inds], inplace=True)
                    target_vec = target_tens.data.reshape(-1)
                    b = b + target_vec

                bs[ii] = b
                oos += [ii]

                x_tens = solver.bra.contract(all) * 10 ** solver.bra.exponent
                x_tens.transpose(*[solver.bra.site_ind_id.format(i) for i in solver.mps_inds], inplace=True)
                x_vec = x_tens.data.reshape(-1)
                xs[ii] = x_vec
                iis += [ii]

        # xAAx_ii = {}
        # Ax_s = {ii: {oo: np.dot(As.get((oo, ii), 0), xs[ii]) for oo in oos} for ii in iis}
        # for ii, iii in np.ndindex(6, 6):
        #     xAAx_ii[ii, iii] = np.sum([np.dot(Ax_s[iii][oo], Ax_s[ii][oo]) for oo in oos])
        # print('xAAx ii')
        # for k, val in xAAx_ii.items():
        #     print(k, val)

        errors2 = {}
        for oo in oos:
            # Ax_i = {ii: np.dot(As.get((oo, ii), 0), xs[ii]) for ii in iis}
            # print('b', oo, 'Ax', iis, [np.dot(Ax_i[ii].conj(), bs[oo]) for ii in iis])
            # print('xAAx', oo, iis, [np.linalg.norm(np.dot(As.get((oo, ii), 0), xs[ii])) ** 2 for ii in iis])
            # xAAx_ii = {(iii, ii): np.dot(Ax_i[iii], Ax_i[ii]) for ii, iii in np.ndindex(6, 6)}
            # print('xAAx_ii')
            # for k, val in xAAx_ii.items():
            #     print(k, val)
            Ax = np.sum([np.dot(As.get((oo, ii), 0), xs[ii]) for ii in iis], axis=0)
            errors2[oo] = np.linalg.norm(bs[oo] - Ax) ** 2
        # print('err2s', errors2)

        b2norms = {oo: np.linalg.norm(bs[oo]) ** 2 for oo in oos}
        err2 = np.sum([errors2[oo] for oo in oos]) / np.sum([b2norms[oo] for oo in oos])
        err = np.sqrt(err2)
        return err

    # @profile
    def check_err(self, dense=False) -> float:
        """ calculates sqrt(<b-Ax|b-Ax>) = sqrt(<Ax|Ax> + <b|b> - 2Re[<Ax|b>])
            for each output component:
                0 = \sum_i A_oi |x_i> - |b_o>
                \sum_i <A_oi x_i | A_oi x_i> - 2Re[<b_o|A_oi|x_i>] + <b_o|b_o>
        """
        # print('np err', self.check_err_np())
        dense = False
        if dense:
            raise NotImplementedError

        else:
            ## || |bo> - sum_i Aoi |xi> ||^2 =
            ## <bo|bo> - sum_i <bo | Aoi | xi> * 2 + sum_i,i' <xi'|Aoi'.T Aoi|xi>
            normAAs = {}  # key o: list [<A_oi x_i | A_oi x_i>]
            ovlps = {}  # key o: list [<b_o|A_oi|x_i>]
            b2_norms = {}  # key o

            if self.num_operators > 1:
                raise NotImplementedError

            def compute_cross_Ax(solver1: LocalSolver, solver2: LocalSolver):
                ### <solver 2 | solver 1>
                A1, x1 = solver1.operators[0].copy(), solver1.ket.copy()
                # A2, x2 = solver2.operators_H[0].copy(), solver2.ket.conj()
                A2, x2 = helper.mpo_flip_upper_lower(solver2.operators[0].conj()), solver2.ket.conj()
                A2.upper_ind_id = solver2.operators_H[0].upper_ind_id
                A2.lower_ind_id = solver2.operators_H[0].lower_ind_id
                x2.site_ind_id = A2.upper_ind_id
                ovlp = qtn.TensorNetwork([A1, x1, A2, x2])
                exponent = ovlp.exponent
                ovlp = ovlp.contract_tags(all)
                ovlp = ovlp * 10 ** exponent
                return ovlp

            for (oo, ii), solver in self.solvers.items():

                # norm_Ax = solver._get_xAAx_()  ## includes exponent

                if oo not in normAAs:
                    normAAs[oo] = {}

                ### \sum_j <A_j x_j | A_i x_i>
                norm_Ax = 0.0
                for (ooo, iii), solver_x in self.solvers.items():
                    if oo == ooo:  # and ii != iii:  # and oo != iii:
                        # print('iii:', oo, iii)
                        ## <solver_x | solver>
                        cross_AA = compute_cross_Ax(solver, solver_x)  ## includes exponent
                        norm_Ax += cross_AA
                normAAs[oo][ii] = norm_Ax  # norm Ax summed over j (iii); ii is i'
                # print('saved oo', oo, 'ii', ii, normAAs[oo])

                ### \sum_o <AH_o x_o | b_i>
                ovlp_b_oi = solver._get_xAb_()  ## includes exponent
                # print('ovlp b oi', oo, ii, ovlp_b_oi)
                if ii not in ovlps.keys():
                    ovlps[ii] = 2 * np.real(ovlp_b_oi)
                else:
                    ovlps[ii] += 2 * np.real(ovlp_b_oi)

                if ii not in b2_norms:
                    b_norm = solver.targets_norm  ## includes exponent
                    b2_norms[ii] = b_norm ** 2

            # for ii, normAAs_i in normAAs.items():
            #     print('norm AA ', ii, normAAs[ii].keys(), [norm_AA_o for oo, norm_AA_o in normAAs_i.items()])
            # print('ovlps', ovlps)

            normAAs = {oo: np.sum([norm_AA_i for ii, norm_AA_i in normAAs_o.items()]) for oo, normAAs_o in
                       normAAs.items()}
            # print('norm AAs', normAAs)
            # print('residual^2', (norm_Ax + self.targets_norm ** 2 - 2 * np.real(ovlp_b_tot)) / self.targets_norm**2)
            err2_o = {ii: (normAAs[ii] + b2_norms[ii] - ovlps[ii]) for ii in b2_norms.keys()}
            b2_norm = np.sum([np.abs(b2norm) for oo, b2norm in b2_norms.items()])

            err2 = np.sum([np.abs(val) for ii, val in err2_o.items()]) / b2_norm
            err = np.sqrt(np.abs(err2))

        # print('err2s', err2_o)
        print('err', err, self.conv_tol)
        self.err = err
        self.is_conv = err < self.conv_tol
        return err
