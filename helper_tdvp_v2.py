"""Updated TDVP / projector-splitting DLR time integration (version 2).

Defines ``TDVPSolver_v2``, an updated version of the :mod:`helper_tdvp` solver
that adds corrections to the equation of motion (following Einkemmer's approach)
on top of the time-dependent variational principle / projector-splitting
integrator for dynamical low-rank approximation. Part of the older monolithic
(legacy) solver stack.
"""
from helper_tdvp import _sum_eff_TNs
from setup_.configs import *
import scipy.linalg
import scipy.sparse.linalg
import helper_quimb as helper
from helper_dmrg import *
from helper_tdvp import *
import local_solvers.helper_tn as helper_tn
import helper_TE
from helper_dmrg import qtn_conjugate_gradient_descent_1site
from setup_.quimb_TN1D import MatrixProductStateUSVT as MPS_USVT
import gridTN as GTN


""" TDVP but with corrections to equation of motion
    corrections use (canonical) basis from previous iteration; vary weights along bond
    follow Einkemmer's method of modifying equation of motion
"""

class TDVPSolver_v2(TDVPSolver):

    # def __init__(self, trial_state: 'qtn.MatrixProductState',
    #              operators: Optional[Sequence['MPO_type']] = None,
    #              targets: Optional[Sequence['MPS_type']] = None,
    #              mps_inds: Sequence[int] = None,
    #              norm_env0_Ls: Sequence['qtn.Tensor'] = None,
    #              norm_env0_Rs: Sequence['qtn.Tensor'] = None,
    #              ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
    #              ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
    #              te_order=0, compress_config: CompressionConfiguration=None,
    #              backprop_edge=False):
    #
    #     super().__init__(trial_state, targets=targets, operators=operators,
    #                      mps_inds=mps_inds, norm_env0_Ls=norm_env0_Ls,
    #                      norm_env0_Rs=norm_env0_Rs, ovlp_env0_Ls=ovlp_env0_Ls,
    #                      ovlp_env0_Rs=ovlp_env0_Rs, te_order=te_order, compress_config=compress_config,
    #                      backprop_edge=backprop_edge)


    def _site_time_evolution_v1(self, dt, left_site_pos, nsites=1,
                                compress_direction=CompressDirection.RIGHT,
                                ) -> Optional[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        with original trotterized back prop
        """
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)

        ## A_eff * site projector (the back TE for repeated bond/site)
        if compress_direction == CompressDirection.RIGHT:
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]
        # print('ket to bra inds', ket_to_bra_inds)
        # print('bonds o', bonds_o)
        # print('bonds i', bonds_i)
        # deriv_total = sum_Aeffs()
        # deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bonds_o + list(bonds_i))
        # print('deriv total', deriv_total)

        site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                            inplace=True, te_order=self.te_order,
                            compress_direction=compress_direction,
                            compress_level=1, compress_opts_dict=self.compress_config)

        ## i think this is already done in canonize
        self.set_bra_from_ket(sites = list(range(left_site_pos,left_site_pos + nsites)))

        ## back prop
        if not at_end:
            # print('project self.ket', ind1)
            right_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
            bra_left_inds = [ket_to_bra_inds[k_ind] for k_ind in left_inds]

            Q, R = qtn.tensor_split(self.ket[ind1], left_inds=left_inds, method='qr', absorb='right')
            # print('Q', Q.inds)
            # print('self.ket[ind1]', self.ket[ind1].inds)
            # Q.transpose_like(self.ket[ind1], inplace=True)
            # Q.modify(data = self.ket[ind1].data)
            # Q = Q.reindex(ket_to_bra_inds)
            QT = Q.conj()
            QT.reindex(ket_to_bra_inds, inplace=True)
            bond_m = next(iter(Q.bonds(R)))
            QT.reindex({bond_m: bond_m + '_'}, inplace=True)
            # Q.reindex({ind_l: ind_l + 'o' for ind_l in bra_left_inds}, inplace=True)
            # Q.add_tag('Q')

            ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
            A_effs = self._get_A_effs(left_site_pos, nsites)

            A_effs_bp = []
            for A_eff in A_effs:
                A_eff_bp = qtn.TensorNetwork([A_eff, QT, Q], virtual=False)
                A_eff_bp.multiply(-1, inplace=True)
                A_effs_bp += [A_eff_bp]

            bonds_i = R.inds
            bonds_o = [(ket_to_bra_inds[ind] if ind != bond_m else bond_m + '_') for ind in bonds_i]

            site_time_evolution(qtn.TensorNetwork([R], virtual=True),
                                dt, A_effs_bp, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                                inplace=True, te_order=self.te_order,
                                compress_direction=compress_direction,
                                compress_level=1, compress_opts_dict=self.compress_config)

            new_QR = qtn.tensor_contract(Q, R)
            new_QR.transpose_like(self.ket[ind1], inplace=True)
            self.ket[ind1].modify(data=new_QR.data)

            ## i think this is already done in canonize
            self.set_bra_from_ket(sites=list(range(left_site_pos, left_site_pos + nsites)))

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            # print('site inds', site_inds[-1], at_end, self.L)
            if at_end:
                # print('canonize', site_inds[0], site_inds[-1])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)     ## also sets bra from ket
            for i in range(ind1, ind2):
                print('(site) update envs left', i, 'orthog at', site_inds[-1])
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_left(i, canonize=False)
        else:
            if at_end:
                # print('canonize', site_inds[-1], site_inds[0])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket
            # for i in range(left_site_pos + nsites, left_site_pos, -1):
            for i in range(ind1, ind2, -1):
                print('(site) update envs right', i, 'orthog at', site_inds[0] - 1)
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_right(i, canonize=False)

        return self.ket


    def _site_time_evolution(self, dt, left_site_pos, nsites=1,
                                compress_direction=CompressDirection.RIGHT,
                                ) -> Optional[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        with original trotterized back prop
        """
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)

        ## A_eff * site projector (the back TE for repeated bond/site)
        if compress_direction == CompressDirection.RIGHT:
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]
        # print('ket to bra inds', ket_to_bra_inds)
        # print('bonds o', bonds_o)
        # print('bonds i', bonds_i)
        # deriv_total = sum_Aeffs()
        # deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bonds_o + list(bonds_i))
        # print('deriv total', deriv_total)

        site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                            inplace=True, te_order=self.te_order,
                            compress_direction=compress_direction,
                            compress_level=1, compress_opts_dict=self.compress_config)

        ## i think this is already done in canonize
        self.set_bra_from_ket(sites = list(range(left_site_pos,left_site_pos + nsites)))

        ## back prop
        if not at_end:
            # print('project self.ket', ind1)
            right_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
            bra_left_inds = [ket_to_bra_inds[k_ind] for k_ind in left_inds]

            Q, R = qtn.tensor_split(self.ket[ind1], left_inds=left_inds, method='qr', absorb='right')
            QTi = Q.conj()
            QTi.reindex({ind_l: ind_l + 'i' for ind_l in left_inds}, inplace=True)
            QTi.add_tag('QTi')
            proj_i = qtn.TensorNetwork([Q, QTi])
            # print('proj_i', proj_i)

            Q = Q.reindex(ket_to_bra_inds)
            QT = Q.conj()
            Q.reindex({ind_l: ind_l + 'o' for ind_l in bra_left_inds}, inplace=True)
            Q.add_tag('Qo')
            proj_o = qtn.TensorNetwork([Q, QT])
            # print('proj_o', proj_o)


            ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
            A_effs = self._get_A_effs(left_site_pos, nsites)

            A_effs_bp = []
            for A_eff in A_effs:
                # A_eff_bp = qtn.TensorNetwork([A_eff, proj_i, proj_o], virtual=False)
                A_eff_bp = qtn.TensorNetwork([A_eff, proj_o], virtual=False)
                # print('A eff bp', A_eff_bp)
                A_eff_bp.multiply(-1, inplace=True)
                A_eff_bp.mangle_inner_()
                Q_tens = A_eff_bp.select_tensors('Qo')[0]
                Q_tens.reindex({ind_l + 'o': ind_l for ind_l in bra_left_inds}, inplace=True)
                # Q_tens = A_eff_bp.select_tensors('QTi')[0]
                # Q_tens.reindex({ind_l + 'i': ind_l for ind_l in left_inds}, inplace=True)
                # print('A eff bp', A_eff_bp)
                # print('A eff bp', A_eff_bp.outer_inds(), A_eff.outer_inds())
                A_effs_bp += [A_eff_bp]

            A_effs = A_effs_bp

            site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                                inplace=True, te_order=self.te_order,
                                compress_direction=compress_direction,
                                compress_level=1, compress_opts_dict=self.compress_config)

            ## i think this is already done in canonize
            self.set_bra_from_ket(sites=list(range(left_site_pos, left_site_pos + nsites)))

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            # print('site inds', site_inds[-1], at_end, self.L)
            if at_end:
                # print('canonize', site_inds[0], site_inds[-1])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)     ## also sets bra from ket
            for i in range(ind1, ind2):
                print('(site) update envs left', i, 'orthog at', site_inds[-1])
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_left(i, canonize=False)
        else:
            if at_end:
                # print('canonize', site_inds[-1], site_inds[0])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket
            # for i in range(left_site_pos + nsites, left_site_pos, -1):
            for i in range(ind1, ind2, -1):
                print('(site) update envs right', i, 'orthog at', site_inds[0] - 1)
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_right(i, canonize=False)

        return self.ket

    def _site_time_evolution_v3(self, dt, left_site_pos, nsites=1,
                                compress_direction=CompressDirection.RIGHT,
                                ) -> Optional[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        not trottered TE
        """
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)
        # print('A eff', left_site_pos, nsites)
        # print('A eff', A_effs[-1].outer_inds())

        ## A_eff * site projector (the back TE for repeated bond/site)
        if compress_direction == CompressDirection.RIGHT:
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

        if True:  # not at_end:
            # print('project self.ket', ind1)
            right_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
            bra_left_inds = [ket_to_bra_inds[k_ind] for k_ind in left_inds]

            Q, R = qtn.tensor_split(self.ket[ind1], left_inds=left_inds, method='qr', absorb='right')

            # Q = Q.reindex(ket_to_bra_inds)
            # QT = Q.conj()
            # Q.reindex({ind_l: ind_l + 'o' for ind_l in bra_left_inds}, inplace=True)
            # Q.add_tag('Q')

            QTi = Q.conj()
            QTi.reindex({ind_l: ind_l + 'i' for ind_l in left_inds}, inplace=True)
            QTi.add_tag('QTi')
            proj_i = qtn.TensorNetwork([Q, QTi])
            # print('proj_i', proj_i)

            Q = Q.reindex(ket_to_bra_inds)
            QT = Q.conj()
            Q.reindex({ind_l: ind_l + 'o' for ind_l in bra_left_inds}, inplace=True)
            Q.add_tag('Qo')
            proj_o = qtn.TensorNetwork([Q, QT])
            # print('proj_o', proj_o)

            A_effs_bp = []
            for A_eff in A_effs:
                # A_eff_bp = qtn.TensorNetwork([A_eff, Q, QT], virtual=False)
                A_eff_bp = qtn.TensorNetwork([A_eff, proj_i, proj_o], virtual=False)
                # print('A eff bp', A_eff_bp)
                A_eff_bp.multiply(-1, inplace=True)
                A_eff_bp.mangle_inner_()
                # Q_tens = A_eff_bp.select_tensors('Q')[0]
                # Q_tens.reindex({ind_l + 'o': ind_l for ind_l in bra_left_inds}, inplace=True)
                Q_tens = A_eff_bp.select_tensors('Qo')[0]
                Q_tens.reindex({ind_l + 'o': ind_l for ind_l in bra_left_inds}, inplace=True)
                Q_tens = A_eff_bp.select_tensors('QTi')[0]
                Q_tens.reindex({ind_l + 'i': ind_l for ind_l in left_inds}, inplace=True)
                # print('A eff bp', A_eff_bp)
                # print('A eff bp', A_eff_bp.outer_inds(), A_eff.outer_inds())
                A_effs_bp += [A_eff_bp]

            A_effs = A_effs + A_effs_bp

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]
        # print('ket to bra inds', ket_to_bra_inds)
        # print('bonds o', bonds_o)
        # print('bonds i', bonds_i)
        # deriv_total = sum_Aeffs()
        # deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bonds_o + list(bonds_i))
        # print('deriv total', deriv_total)

        site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                            inplace=True, te_order=self.te_order,
                            compress_direction=compress_direction,
                            compress_level=1, compress_opts_dict=self.compress_config)

        ## i think this is already done in canonize
        self.set_bra_from_ket(sites = list(range(left_site_pos,left_site_pos + nsites)))

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            # print('site inds', site_inds[-1], at_end, self.L)
            if at_end:
                # print('canonize', site_inds[0], site_inds[-1])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                # print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)     ## also sets bra from ket
            for i in range(ind1, ind2):
                # print('(site) update envs left', i, 'orthog at', site_inds[-1])
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_left(i, canonize=False)
        else:
            if at_end:
                # print('canonize', site_inds[-1], site_inds[0])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                # print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket
            # for i in range(left_site_pos + nsites, left_site_pos, -1):
            for i in range(ind1, ind2, -1):
                # print('(site) update envs right', i, 'orthog at', site_inds[0] - 1)
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_right(i, canonize=False)

        return self.ket


    def _site_time_evolution_v4(self, dt, left_site_pos, nsites=1,
                                compress_direction=CompressDirection.RIGHT,
                                ) -> Optional[qtn.MatrixProductState]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time
        trotter TE with |new><old| H |old><new|
        """
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)

        ## A_eff * site projector (the back TE for repeated bond/site)
        if compress_direction == CompressDirection.RIGHT:
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]
        # print('ket to bra inds', ket_to_bra_inds)
        # print('bonds o', bonds_o)
        # print('bonds i', bonds_i)
        # deriv_total = sum_Aeffs()
        # deriv_total = self._sum_eff_TNs(A_effs, transpose_bonds=bonds_o + list(bonds_i))
        # print('deriv total', deriv_total)

        right_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
        Q_old, R_old = qtn.tensor_split(self.ket[ind1], left_inds=left_inds, method='qr', absorb='right')
        bond_m = next(iter(Q_old.bonds(R_old)))

        site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                            inplace=True, te_order=self.te_order,
                            compress_direction=compress_direction,
                            compress_level=1, compress_opts_dict=self.compress_config)

        ## i think this is already done in canonize
        self.set_bra_from_ket(sites=list(range(left_site_pos, left_site_pos + nsites)))

        ## back prop
        if not at_end:
            # print('project self.ket', ind1)
            right_inds, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
            bra_left_inds = [ket_to_bra_inds[k_ind] for k_ind in left_inds]

            Q, R = qtn.tensor_split(self.ket[ind1], left_inds=left_inds, method='qr', absorb='right',
                                    bond_ind=bond_m)
            # QTi = Q.conj()
            QTi = Q_old.conj()
            QTi.reindex({ind_l: ind_l + 'i' for ind_l in left_inds}, inplace=True)
            QTi.add_tag('QTi')
            proj_i = qtn.TensorNetwork([Q_old, QTi])
            # print('proj_i', proj_i)

            QT_old = Q_old.conj().reindex(ket_to_bra_inds)
            # Q = Q.reindex(ket_to_bra_inds)
            Q = Q_old.reindex(ket_to_bra_inds)
            Q.reindex({ind_l: ind_l + 'o' for ind_l in bra_left_inds}, inplace=True)
            Q.add_tag('Qo')
            proj_o = qtn.TensorNetwork([Q, QT_old])
            # print('proj_o', proj_o)

            ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
            A_effs = self._get_A_effs(left_site_pos, nsites)

            A_effs_bp = []
            for A_eff in A_effs:
                # A_eff_bp = qtn.TensorNetwork([A_eff, Q, QT], virtual=False)
                A_eff_bp = qtn.TensorNetwork([A_eff, proj_i, proj_o], virtual=False)
                # print('A eff bp', A_eff_bp)
                A_eff_bp.multiply(-1, inplace=True)
                A_eff_bp.mangle_inner_()
                Q_tens = A_eff_bp.select_tensors('Qo')[0]
                Q_tens.reindex({ind_l + 'o': ind_l for ind_l in bra_left_inds}, inplace=True)
                Q_tens = A_eff_bp.select_tensors('QTi')[0]
                Q_tens.reindex({ind_l + 'i': ind_l for ind_l in left_inds}, inplace=True)
                # print('A eff bp', A_eff_bp)
                # print('A eff bp', A_eff_bp.outer_inds(), A_eff.outer_inds())
                A_effs_bp += [A_eff_bp]

            A_effs = A_effs_bp

            site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                                inplace=True, te_order=self.te_order,
                                compress_direction=compress_direction,
                                compress_level=1, compress_opts_dict=self.compress_config)

            ## i think this is already done in canonize
            self.set_bra_from_ket(sites=list(range(left_site_pos, left_site_pos + nsites)))

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            if at_end:
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket
            for i in range(ind1, ind2):
                self._update_envs_left(i, canonize=False)
        else:
            if at_end:
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket
            for i in range(ind1, ind2, -1):
                self._update_envs_right(i, canonize=False)

        return self.ket



    def take_time_step_l2r(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, verbose=False,
                           **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        max_bond = self.max_bond
        # do_adapt = False
        print('l2r tdvp V2', dt, 'do adapt', do_adapt)
        # print('ket L', L)
        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if canonize:
            self.canonize(0)   ## does not scale canonical tensors

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_right(1, canonize=False)

        ## right sweep
        nsites = 0
        direction = CompressDirection.RIGHT
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
                # print('mod max bond:', 'site', site_ind, 'cur', cur_orthog, L, max_bond_)
                # print('max bond', max_bond_, cur_orthog)
                # print('bond size', self.ket.bond_size(cur_orthog, cur_orthog + 1))
                # adapt = do_adapt and self.ket.bond_size(cur_orthog, cur_orthog + 1) < max_bond_
                adapt = do_adapt and self.ket.bond_size(site_ind, site_ind + 1) < max_bond_
                # print('bond size', self.ket.bond_size(site_ind, site_ind + 1))
                # print('do_adapt', adapt)

            # if not adapt:  # one site update of site_ind
            #     if nsites == 2:  ## previously updated this site
            #         nsites = 1
            #         site_ind += 1
            #         continue

            ## backward propagation (skipped for first iteration)
            # print('self.ket norm', helper.norm(self.ket))
            # if 0 < site_ind < L:
            #     if nsites == 1:
            #         # bond_ind = site_ind - 1
            #         if verbose:  print('BACK prop (LR) bond ind', cur_orthog, direction)
            #         # print('check orthog', helper.check_orthog(self.ket))
            #         self._bond_time_evolution(-dt, cur_orthog, compress_direction=direction )
            #
            #         ## move canonicalization to next site (site_ind)
            #         self.canonize(site_ind, cur_orthog=cur_orthog)      ## updates bra internally
            #         # print('(bond) update left env', cur_orthog, 'canon at', site_ind)
            #         # print('check orthog', helper.check_orthog(self.ket))
            #         self._update_envs_left(cur_orthog, canonize=False)
            #
            #     else:
            #         if verbose:  print('BACK prop (LR) site ind', site_ind, direction)
            #         # print('check orthog', helper.check_orthog(self.ket))
            #         self._site_time_evolution(-dt, site_ind, nsites = 1, compress_direction=direction)


            nsites = 2 # if adapt else 1
            print('nsites', nsites, adapt)

            ## forward propagation
            # print('before cur orthog', cur_orthog)
            # print('self.ket norm', helper.norm(self.ket))
            # print('check orthog', helper.check_orthog(self.ket))
            if verbose:  print('FORWARD prop sites (LR)', list(range(site_ind,site_ind+nsites)) )
            self._site_time_evolution(dt, left_site_pos=site_ind, nsites=nsites, compress_direction=direction)
                ## updates environment within function to site_ind + nsites - 1
            cur_orthog = site_ind + nsites - 1
            site_ind = site_ind + nsites - 1 if nsites > 1 else site_ind + nsites
            # print('check orthog', helper.check_orthog(self.ket))
            # print('next cur orthog', cur_orthog, 'next site ind', site_ind)

        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if self.backprop_edge:
            raise NotImplementedError('backprop of last site in TDVP not implemented')

        print('done left to right sweep')
        return self.ket


    def take_time_step_r2l(self, dt, grid=None, do_adapt=False, canonize=False, build_envs=False, verbose=False,
                           **kwargs):
        """ A: self.operators:  dictates time evolution
            x: self.ket (updated in place)
        """
        L = self.ket.L
        # do_adapt = False
        max_bond = self.max_bond
        print('r2l tdvp V2', dt, 'do adapt', do_adapt)
        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if canonize:
            self.canonize(L-1)  ## does not scale canonical tensors
            # print('ket canon')
            # helper.check_orthog(self.ket)
            # print('bra canon')
            # helper.check_orthog(self.bra)

        ## ovlp <Ax|b>, <Ax|x> init envs
        if build_envs:
            self._build_all_envs_left(1, canonize=False)

        ### right to left sweep
        nsites = 0
        direction = CompressDirection.LEFT
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
                # max_bond_ = min(max_bond, 2 ** site_ind, 2 ** (L - site_ind))
                # print('max bond', max_bond_, site_ind)
                # print('bond size', site_ind, site_ind-1, self.ket.bond_size(site_ind, site_ind - 1))
                # adapt = do_adapt and self.ket.bond_size(site_ind, site_ind - 1) < max_bond_

                # max_bond_ = max_bond  # min(max_bond, 2 ** cur_orthog, 2 ** (L - cur_orthog))
                max_bond_ = min(max_bond, 2 ** site_ind, 2 ** (L - site_ind))  # if site_ind != L-1 else 2
                # print('mod max bond: site', site_ind, 'cur', cur_orthog, L, max_bond_,)
                # print('max bond', max_bond_, cur_orthog)
                # print('bond size', cur_orthog, cur_orthog - 1, self.ket.bond_size(cur_orthog, cur_orthog - 1))
                # adapt = do_adapt and self.ket.bond_size(cur_orthog, cur_orthog - 1) < max_bond_
                adapt = do_adapt and self.ket.bond_size(site_ind, site_ind - 1) < max_bond_
                # print('bond size', self.ket.bond_size(site_ind, site_ind - 1))
                # print('do_adapt', adapt)

            # if not adapt:  # one site update of site_ind
            #     if nsites == 2:  ## previously updated this site
            #         nsites = 1
            #         site_ind -= 1
            #         continue

            # if 0 <= site_ind < L - 1:    ## skip initial back propagation
            #     if nsites == 1:
            #         if verbose:  print('BACK prop (RL) bond ind', cur_orthog, direction )
            #         # print('check orthog', helper.check_orthog(self.ket))
            #         self._bond_time_evolution(-dt, cur_orthog, compress_direction=direction)
            #
            #         ## move canonicalization to next site (site_ind)
            #         self.canonize(site_ind, cur_orthog=cur_orthog)  ## updates bra internally
            #         # print('(bond) update right env', cur_orthog, 'canon at', site_ind)
            #         # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
            #         self._update_envs_right(cur_orthog, canonize=False)
            #         # print('new site ind', site_ind)
            #     else:
            #         if verbose:  print('BACK prop (RL) site ind', site_ind, direction)
            #         self._site_time_evolution(-dt, site_ind, nsites=1, compress_direction=direction)

                # print('back prop (-1) check orthog', site_ind, cur_orthog)
                # helper.check_orthog(self.ket)
                # helper.check_orthog(self.bra)
                # if grid is not None:
                #     plt.figure()
                #     gtn = grid.make_gridTN(data=self.ket)
                #     plt.imshow(np.abs(gtn.get_data()) ** 2)
                #     plt.colorbar()
                #     plt.show()

            nsites = 2  # if adapt else 1
            print('nsites', nsites, adapt)
            cur_orthog = site_ind - nsites + 1    ## get left site ind

            # print('self.ket norm', helper.norm(self.ket))
            # print('check orthog', helper.check_orthog(self.ket))
            if verbose:  print('FORWARD prop sites (RL)', list(range(cur_orthog,cur_orthog+nsites)) )
            self._site_time_evolution(dt, cur_orthog, nsites=nsites, compress_direction=direction)
            site_ind = site_ind - nsites + 1 if nsites > 1 else site_ind - nsites
            # print('check orthog', helper.check_orthog(self.ket))
            # print( 'next cur orthog', cur_orthog, 'next site ind', site_ind,)

            # print('forward prop (-1) check orthog', cur_orthog)
            # helper.check_orthog(self.ket)
            # helper.check_orthog(self.bra)
            # if grid is not None:
            #     plt.figure()
            #     gtn = grid.make_gridTN(data=self.ket)
            #     plt.imshow(np.abs(gtn.get_data()) ** 2)
            #     plt.colorbar()
            #     plt.show()

        # print('self.ket norm', helper.norm(self.ket), self.ket.exponent)

        if self.backprop_edge:
            raise NotImplementedError('backprop of last site in TDVP not implemented')

        print('done right to left sweep')
        return self.ket


class TDDMRGSolver_v2(TDMRGSolver):

    ### interesting; this does |k><k| = |k1><k1| + |k2><k2| + ...
    ### as opposed to |k> = |k1> + |k2> + ...
    ### TD-DMRG by Fenguin and White

    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 targets: Optional[Sequence['MPS_type']] = None,
                 operators: Optional[Sequence['MPO_type']] = None,
                 operators_H: Optional[Sequence['MPO_type']] = None,  ## not used
                 bra_state: Optional['qtn.MatrixProductState'] = None,
                 in_ind: int = 0, out_ind: int = 0,
                 mps_inds: Sequence[int] = None,
                 # norm_env0_Ls: Sequence['qtn.Tensor'] = None,  # for A envs
                 # norm_env0_Rs: Sequence['qtn.Tensor'] = None,  # for A envs
                 # ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 # ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 ket_env0_L: Optional['qtn.Tensor'] = None,
                 ket_env0_R: Optional['qtn.Tensor'] = None,
                 backprop_edge=False,
                 te_order=0, compress_config: CompressionConfiguration = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_tot_iter=DEFAULT_MAX_TOT_ITER,
                 max_wrong_iter=DEFAULT_MAX_WRONG_ITER, max_bond=None,
                 **env_kwargs,
                 ):

        # super().__init__(trial_state, targets=targets, operators=operators, mps_inds=mps_inds)
        super().__init__(trial_state, targets=targets, operators=operators, bra_state=bra_state,
                         mps_inds=mps_inds, in_ind=in_ind, out_ind=out_ind,
                         **env_kwargs,
                         # norm_env0_Ls=norm_env0_Ls, norm_env0_Rs=norm_env0_Rs,
                         # ovlp_env0_Ls=ovlp_env0_Ls, ovlp_env0_Rs=ovlp_env0_Rs,
                         te_order=te_order, compress_config=compress_config)

        self.ket_env0_L = ket_env0_L
        self.ket_env0_R = ket_env0_R
        self.ovlp_proj = None
        self.backprop_edge = backprop_edge  ## do back prop at end of chain if False


    def create_like(self, copy=True, new_ket=None, **kwargs):
        norm_env0_Ls = [A_envL[0] for A_envL in self.A_envsLs]
        norm_env0_Rs = [A_envR[self.L - 1] for A_envR in self.A_envsRs]

        new_solver = TDVPSolver((self.ket.copy() if copy else self.ket) if new_ket is None else new_ket,
                                operators=kwargs.get('operators', [o.copy() if copy else o for o in self.operators]),
                                mps_inds=kwargs.get('mps_ind_range', self.mps_inds),
                                norm_env0_Ls=kwargs.get('norm_env0_Ls', norm_env0_Ls),
                                norm_env0_Rs=kwargs.get('norm_env0_Rs', norm_env0_Rs),
                                te_order=kwargs.get('te_order', self.te_order),
                                compress_config=kwargs.get('compress_config', self.compress_config),
                                backprop_edge=kwargs.get('backprop_edge', self.backprop_edge),
                                # solve_type=kwargs.get('solve_type', self.solve_type)
                                )
        return new_solver

    def get_rk4_eff(self, dt, left_site_pos, nsites=1) -> Sequence['qtn.Tensor']:
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)
        b_effs = self._get_b_effs(left_site_pos, nsites)    ## only has source exponent

        # tmp_A_eff = self._sum_eff_TNs(A_effs, transpose_bonds=[*bonds_o,*bonds_i])
        # tmp_A_eff_data = tmp_A_eff.data.reshape(np.prod(dist_submpx.shape),-1)
        # # evals, evecs = np.linalg.eigh(1.j*tmp_A_eff_data)
        # evals, evecs = scipy.sparse.linalg.eigsh(1.j*tmp_A_eff_data, k=1, which='LM')
        # sort_inds = np.argsort(np.abs(evals))[::-1]
        # print('self.ket', self.ket.max_bond())
        # print('is A', left_site_pos, tmp_A_eff.shape, np.linalg.norm(tmp_A_eff_data + tmp_A_eff_data.T.conj()))
        # print('max evals', left_site_pos, evals[sort_inds[:10]])

        def deriv_func(dist_submpx_, dt=1) -> 'qtn.Tensor':
            eff_Ax = []
            for deriv_tn in A_effs:
                # print('deriv_tn', helper.norm(deriv_tn))
                op = qtn.tensor_contract(*deriv_tn.tensors)
                op.transpose(*bonds_o, *bonds_i, inplace=True)
                # sq_shape = int(np.round(np.sqrt(op.size)))
                # print(op.inds, op.data.reshape(sq_shape,sq_shape))
                # print('dist submpx', dist_submpx_)
                eff_Ax += [qtn.TensorNetwork([deriv_tn, dist_submpx_])]
                # print('eff_ax', eff_Ax[-1].exponent, deriv_tn.exponent)
                # print('dist_submpx exponent', dist_submpx_.exponent)
                ## dist_submpx exponent should equal 0

            # print('beffs', b_effs)
            for source_tens in b_effs:
                # print(source_tens.norm(), helper.norm(source_tens), source_tens.exponent)
                scaled_source = qtn.TensorNetwork([source_tens])
                scaled_source.exponent += -self.ket.exponent
                eff_Ax += [scaled_source]

            # print('eff Ax', [helper.norm(op) for op in eff_Ax], [op.exponent for op in eff_Ax])

            out = _sum_eff_TNs(eff_Ax, transpose_bonds=bonds_o)
            # out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
            out.reindex({bo: bi for bo, bi in zip(bonds_o, bonds_i)}, inplace=True)
            out.modify(apply=lambda x: x * dt)
            return out

        # print('deriv dist submpx', dist_submpx.norm())
        # print(dist_submpx.tensors[0].inds, dist_submpx.tensors[0].data)

        k1 = deriv_func(dist_submpx, dt=dt)

        T1 = dist_submpx.contract()
        k1.transpose_like(T1, inplace=True)
        s1 = k1.copy()
        s1.modify(apply=lambda x: x * 0.5 + T1.data)
        k2 = deriv_func(s1, dt=dt)  ## assumes constant A; should be A(t+dt/2)

        k2.transpose_like(T1, inplace=True)
        s2 = k2.copy()
        s2.modify(apply=lambda x: x * 0.5 + T1.data)
        k3 = deriv_func(s2, dt=dt)  ## assumes constant A; should be A(t+dt/2)

        k3.transpose_like(T1, inplace=True)
        s3 = k3.copy()
        s3.modify(apply=lambda x: x + T1.data)
        k4 = deriv_func(s3, dt=dt)

        # print('deriv norms', k1.norm(), k2.norm(), k3.norm(), k4.norm())

        return k1, k2, k3, k4

    def _site_time_evolution_dmrg(self, dt, left_site_pos, proj=None, nsites=1,
                                  compress_direction=CompressDirection.RIGHT,
                                  ) -> tuple[Optional[qtn.MatrixProductState], Optional[qtn.Tensor]]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time via RK4 according to Fenguin + White
        proj is placeholder for compatibility with parent class. not actually used here
        """
        # print('site TE', left_site_pos)

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        # A_effs = self._get_A_effs(left_site_pos, nsites)

        # ## A_eff * site projector (the back TE for repeated bond/site)
        if compress_direction == CompressDirection.RIGHT:
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        ### build rk4 vectors
        T1 = dist_submpx.contract()
        k1, k2, k3, k4 = self.get_rk4_eff(dt, left_site_pos, nsites=nsites)
        ## if at end, can solve TE using other means

        ## targeted time steps
        if not at_end:
            T1.transpose_like(k1, inplace=True)
            psi13 = T1.copy()
            psi13.modify(apply=lambda x: x + 1. / 162 * (31 * k1.data + 14 * k2.data + 14 * k3.data - 5 * k4.data))
            psi23 = T1.copy()
            psi23.modify(apply=lambda x: x + 1. / 81 * (16 * k1.data + 20 * k2.data + 20 * k3.data - 2 * k4.data))
            psi33 = T1.copy()
            psi33.modify(apply=lambda x: x + 1. / 6 * (k1.data + 2 * k2.data + 2 * k3.data + k4.data))

            # print('T1', T1.norm())
            # print('psi', psi13.norm(), psi23.norm(), psi33.norm())

            ## bond to contract over
            if compress_direction == CompressDirection.RIGHT:
                x_bond = self.ket.bond(ind1 + nsites - 1, ind1 + nsites)
            else:
                x_bond = self.ket.bond(ind1 - nsites + 1, ind1 - nsites)

            def compute_denmat(ket: 'qtn.Tensor'):
                bra = ket.conj()
                bra.reindex({k: b for k, b in ket_to_bra_inds.items() if k != x_bond}, inplace=True)
                return qtn.tensor_contract(bra, ket)

            rho_bonds_i = [bi for bi in bonds_i if bi != x_bond]
            rho_bonds_o = [ket_to_bra_inds[ind] for ind in rho_bonds_i]
            nb = len(rho_bonds_o)

            rho0 = compute_denmat(T1)
            rho0.transpose(*rho_bonds_i, *rho_bonds_o, inplace=True)  # |ket><bra|
            rho1 = compute_denmat(psi13)
            rho1.transpose_like(rho0, inplace=True)
            rho2 = compute_denmat(psi23)
            rho2.transpose_like(rho0, inplace=True)
            rho3 = compute_denmat(psi33)
            rho3.transpose_like(rho0, inplace=True)
            tot_denmat = rho0.copy()

            # print('rhos', rho0.norm() * 1./3, rho1.norm() * 1./6, rho2.norm() * 1./6, rho3.norm() * 1./3)

            tot_denmat.modify(apply=lambda x: x * 1. / 3 + rho1.data * 1. / 6 +
                                              rho2.data * 1. / 6 + rho3.data * 1. / 3)
            # print('rho norms', rho0.norm(), rho1.norm(), rho2.norm(), rho3.norm(), tot_denmat.norm())
            denmat_shape = tot_denmat.shape
            tens_shape = denmat_shape[:nb]
            sq_shape = np.prod(denmat_shape[:nb])
            # print('tot denmat', tot_denmat.inds)
            # print('tot denmat', tot_denmat.data.reshape(sq_shape, sq_shape))

            # test = tot_denmat.data.reshape(sq_shape, sq_shape)
            # print('rho is H', np.linalg.norm(test - test.T.conj()))
            eigval, eigvec = np.linalg.eigh(tot_denmat.data.reshape(sq_shape, sq_shape))

            sort_inds = np.argsort(np.abs(eigval))[::-1]
            ev_max = eigval[sort_inds[0]]
            # eigval_inds = np.where(np.abs(eigval) > 1.0e-12 * ev_max)
            new_sort_inds = []
            for si in sort_inds:
                if np.abs(eigval[si]) > CUTOFF * np.abs(ev_max):
                    new_sort_inds += [si]
                else:
                    break
            sort_inds = new_sort_inds

            # sort_inds = [si for si in sort_inds if eigval[si] > 1.0e-12 * ev_max]
            # eigval = eigval[eigval_inds[0]]
            # eigvec = eigvec[:, eigval_inds[0]]
            # sort_inds = np.argsort(np.abs(eigval))[::-1]
            if self.max_bond is not None:
                # print('eigvals', eigval[sort_inds])
                # eigval = eigval[sort_inds[:self.max_bond]]
                # eigvec = eigvec[:, sort_inds[:self.max_bond]]
                if self.max_bond // nsites > 0:
                    eigvec = eigvec[:, sort_inds[:self.max_bond // nsites]]
                # ## don't think this is the proper way to do 2-site, but DMRG doesn't seem to need 2-site
                # ## to reach full bond dimension. if don't divide by nsites, compressing eigvecs later causes
                # ## bad compression errors
                # print('eigvec teyp', eigvec.dtype)

            eigvec = eigvec.reshape(*tens_shape, -1)
            eigvec = qtn.Tensor(data=eigvec, inds=tuple(rho_bonds_i) + (x_bond + '_',))

            # print('eigvec norm', eigvec.norm())
            # print('eigvec', eigvec.data)

            if nsites == 1:
                # print(self.ket[ind1].transpose('i(0)', self.ket.bond(0, 1)).data)
                # print(self.ket[ind2].transpose('i(1)', self.ket.bond(0, 1), self.ket.bond(1, 2)).data)

                eigvec.transpose_like(self.ket[ind1], inplace=True)
                next_site = qtn.tensor_contract(eigvec.conj(), self.ket[ind1], self.ket[ind2])
                next_site.transpose_like(self.ket[ind2], inplace=True)
                self.ket[ind1].modify(data=eigvec.data)
                self.ket[ind2].modify(data=next_site.data)
                # print('self.ket[ind1]', ind1, self.ket[ind1].norm(), self.ket.exponent)
                # print('self.ket[ind2]', ind2, self.ket[ind2].norm(), self.ket.exponent)

                # print(self.ket[ind1].transpose('i(0)', self.ket.bond(0, 1)).data)
                # print(self.ket[ind2].transpose('i(1)', self.ket.bond(0, 1), self.ket.bond(1, 2)).data)

                # print('ind1', ind1, ind2)
                # helper.check_orthog(self.ket)
            elif nsites == 2:
                ## decompose eigvec into two sites, tensor contract ind1 site
                # bond_m = self.ket.bond(ind1, ind2)
                bond_ms, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
                bond_m = next(iter(bond_ms))
                # print('eigvec', eigvec)
                # print('left_inds', left_inds)
                ##
                ev1, ev2 = qtn.tensor_split(eigvec, left_inds, absorb='right', bond_ind=bond_m + '_',
                                            max_bond=self.max_bond)
                next_site = qtn.tensor_contract(ev1.conj(), self.ket[ind1], self.ket[ind2])
                ev1.transpose_like(self.ket[ind1], inplace=True)
                next_site.transpose_like(self.ket[ind2], inplace=True)
                self.ket[ind1].modify(data=ev1.data)
                self.ket[ind2].modify(data=next_site.data)
                # print('ind1', ind1, ind2)
                # helper.check_orthog(self.ket)
            else:
                raise NotImplementedError

        else:

            ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
            A_effs = self._get_A_effs(left_site_pos, nsites)

            ### sources
            b_effs = self._get_b_effs(left_site_pos, nsites)
            b_effs_scaled = [helper.scalar_multiply(b_eff, 10**(-self.ket.exponent)) for b_eff in b_effs]


            ## change indices from bra (missing T*[i]) to ket
            site_ind = self.mps_inds[left_site_pos]
            ket_to_bra_inds = self.get_ket_to_bra_inds(site_ind)

            bonds_i = dist_submpx.outer_inds()
            bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

            site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                                sources=b_effs_scaled,
                                inplace=True, te_order=self.te_order,
                                compress_direction=compress_direction,
                                compress_level=1, compress_opts_dict=self.compress_config)

            # new_ket = T1.copy()
            # new_ket.modify(apply=lambda x: x + (k1.data + 2*k2.data + 2*k3.data + k4.data)/6 )
            # if nsites == 1:
            #     new_ket.transpose_like(self.ket[ind1], inplace=True)
            #     self.ket[ind1].modify(data=new_ket.data)
            # elif nsites == 2:
            #     left_inds, bond_ms = self.ket[ind1].filter_bonds(self.ket[ind2])
            #     bond_m = next(iter(bond_ms))
            #     ev1, ev2 = qtn.tensor_split(new_ket, left_inds, absorb='right', bond_ind=bond_m + '_')
            #     ev1.transpose_like(self.ket[ind1], inplace=True)
            #     ev2.transpose_like(self.ket[ind2], inplace=True)
            #     self.ket[ind1].modify(data=ev1.data)
            #     self.ket[ind2].modify(data=ev2.data)
            # else:
            #     raise NotImplementedError

        ## i think this is already done in canonize
        self.set_bra_from_ket(sites=list(range(left_site_pos, left_site_pos + nsites)))

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            # print('site inds', site_inds[-1], at_end, self.L)
            ## canonicalization is not needed here, but is used by subclass TDDMRGSolver_v3 for ket0, bra0
            if at_end:
                # print('canonize', site_inds[0], site_inds[-1])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                # print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket

            for i in range(ind1, ind2):
                # print('(site) update envs left', i, 'orthog at', site_inds[-1])
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_left(i, canonize=False)
        else:
            if at_end:
                # print('canonize', site_inds[-1], site_inds[0])
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                # print('canonize', ind1, ind2)
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket

            for i in range(ind1, ind2, -1):
                # print('(site) update envs right', i, 'orthog at', site_inds[0] - 1)
                # print('check orthog', helper.check_orthog(self.ket), helper.check_orthog(self.bra))
                self._update_envs_right(i, canonize=False)

        return self.ket, None



class TDDMRGSolver_v3(TDDMRGSolver_v2):
    """
    TD-DMRG++ by Ronca and Chan
    """

    def __init__(self, trial_state: 'qtn.MatrixProductState',
                 operators: Optional[Sequence['MPO_type']] = None,
                 targets: Optional[Sequence['MPS_type']] = None,
                 bra_state: Optional['qtn.MatrixProductState'] = None,
                 in_ind: int = 0, out_ind: int = 0,
                 mps_inds: Sequence[int] = None,
                 norm_env0_Ls: Sequence['qtn.Tensor'] = None,  # for A envs
                 norm_env0_Rs: Sequence['qtn.Tensor'] = None,  # for A envs
                 # ovlp_env0_Ls: Sequence['qtn.Tensor'] = None,
                 # ovlp_env0_Rs: Sequence['qtn.Tensor'] = None,
                 ket_env0_L: Optional['qtn.Tensor'] = None,
                 ket_env0_R: Optional['qtn.Tensor'] = None,
                 backprop_edge=False,
                 te_order=0, compress_config: CompressionConfiguration = None,
                 conv_tol=DEFAULT_CONV_TOL, max_iter=DEFAULT_MAX_ITER, max_tot_iter=DEFAULT_MAX_TOT_ITER,
                 max_wrong_iter=DEFAULT_MAX_WRONG_ITER, max_bond=None,
                 **env_kwargs):

        super().__init__(trial_state, targets=targets, operators=operators, bra_state=bra_state,
                         mps_inds=mps_inds,
                         in_ind=in_ind, out_ind=out_ind,
                         **env_kwargs,
                         # norm_env0_Ls=norm_env0_Ls, norm_env0_Rs=norm_env0_Rs,
                         # ovlp_env0_Ls=ovlp_env0_Ls, ovlp_env0_Rs=ovlp_env0_Rs,
                         te_order=te_order, compress_config=compress_config)

        self.ket_env0_L = ket_env0_L
        self.ket_env0_R = ket_env0_R
        self.ovlp_proj = None
        self.backprop_edge = backprop_edge  ## do back prop at end of chain if False

        self.ket0 = self.ket.copy()
        self.bra0 = self.ket0.conj()
        self.bra0.mangle_inner_(append='_')
        self.bra0.site_ind_id = self.ket.site_ind_id
        norm_env0_Ls = [None] * self.num_operators if norm_env0_Ls is None else norm_env0_Ls
        norm_env0_Rs = [None] * self.num_operators if norm_env0_Rs is None else norm_env0_Rs
        self.A0_envsLs = [Environment(self.L, EnvironmentSide.LEFT, self.ket0, self.bra, self.operators[i],
                                      init_env=norm_env0_Ls[i], mps_inds=self.mps_inds)
                          for i in range(self.num_operators)]
        self.A0_envsRs = [Environment(self.L, EnvironmentSide.RIGHT, self.ket0, self.bra, self.operators[i],
                                      init_env=norm_env0_Rs[i], mps_inds=self.mps_inds)
                          for i in range(self.num_operators)]
        self.S0_envsL = Environment(self.L, EnvironmentSide.LEFT,  self.ket, self.bra0, mps_inds=self.mps_inds)
        self.S0_envsR = Environment(self.L, EnvironmentSide.RIGHT, self.ket, self.bra0, mps_inds=self.mps_inds)
        # exit()


    @property
    def left_envs(self):
        return  self.A_envsLs + self.b_envsLs + self.A0_envsLs + [self.S0_envsL]

    @property
    def right_envs(self):
        return self.A_envsRs + self.b_envsRs + self.A0_envsRs + [self.S0_envsR]


    def create_like(self, copy=True, new_ket=None, **kwargs):
        norm_env0_Ls = [A_envL[0] for A_envL in self.A_envsLs]
        norm_env0_Rs = [A_envR[self.L - 1] for A_envR in self.A_envsRs]

        new_solver = TDVPSolver((self.ket.copy() if copy else self.ket) if new_ket is None else new_ket,
                                operators=kwargs.get('operators', [o.copy() if copy else o for o in self.operators]),
                                mps_inds=kwargs.get('mps_ind_range', self.mps_inds),
                                norm_env0_Ls=kwargs.get('norm_env0_Ls', norm_env0_Ls),
                                norm_env0_Rs=kwargs.get('norm_env0_Rs', norm_env0_Rs),
                                te_order=kwargs.get('te_order', self.te_order),
                                compress_config=kwargs.get('compress_config', self.compress_config),
                                backprop_edge=kwargs.get('backprop_edge', self.backprop_edge),
                                # solve_type=kwargs.get('solve_type', self.solve_type)
                                )
        return new_solver


    def copy(self, deep=True):
        new_solver = self.create_like(copy=deep)

        new_solver.A_envsLs = [A_envsL.copy() for A_envsL in self.A_envsLs]
        new_solver.A_envsRs = [A_envsR.copy() for A_envsR in self.A_envsRs]
        new_solver.A0_envsLs = [A_envsL.copy() for A_envsL in self.A0_envsLs]
        new_solver.A0_envsRs = [A_envsR.copy() for A_envsR in self.A0_envsRs]
        new_solver.S0_envsL = self.S0_envsL.copy()
        new_solver.S0_envsR = self.S0_envsR.copy()

        # new_solver.left_envs = new_solver.A_envsLs + new_solver.b_envsLs
        # new_solver.right_envs = new_solver.A_envsRs + new_solver.b_envsRs

        # new_solver.solve_type = self.solve_type
        # new_solver.err = self.err
        # new_solver.is_conv = self.is_conv
        return new_solver


    def canonize(self, i, cur_orthog=None):

        ## canonize ket, bra
        super().canonize(i, cur_orthog=cur_orthog)

        ## also canonize ket0
        tens_left, tens_right = [], []
        max_ind = np.max(self.mps_inds)
        i = self.get_mps_ind(i)
        updated_inds = []
        for lx in sorted(self.mps_inds):
            if (0 if cur_orthog is None else cur_orthog) <= lx <= i:
                updated_inds += [lx]
                tens_left += [self.ket0.select_tensors((self.ket0.site_tag_id.format(lx),))[0]]
            if (max_ind if cur_orthog is None else cur_orthog) >= lx >= i:
                if lx not in updated_inds:
                    updated_inds += [lx]
                tens_right += [self.ket0.select_tensors((self.ket0.site_tag_id.format(lx),))[0]]
        # tens_left  = [self.ket[lx] for lx in sorted(self.mps_inds) if lx <= i]
        # tens_right = [self.ket[rx] for rx in sorted(self.mps_inds) if rx >= i]
        helper.canonize_tens_list(*tens_left, inplace=True)
        helper.canonize_tens_list(*(tens_right[::-1]), inplace=True)

        # print('updated inds', updated_inds)
        # print('updated inds', len(tens_left + tens_right[1:]) )

        ### update bra0
        ### there's some bug/misalignement of ix and tens
        for ix in updated_inds:
            i = ix  # self.mps_inds[ix]
            ket_to_bra_inds = {k: b for k, b in self.get_ket_to_bra_inds(ix).items() if k != self.ket.site_ind(ix)}
            ket_tens = self.ket0[i]
            bra_tens = ket_tens.reindex(ket_to_bra_inds).conj()
            self.bra0[i].modify(data=bra_tens.data, inds=tuple(bra_tens.inds))

            # bra_to_ket_inds = {b: k for k, b in ket_to_bra_inds.items()}
            # test = self.bra0[i].reindex(bra_to_ket_inds)
            # test = test.transpose_like(self.ket0[i])
            # print('bra err', i, np.linalg.norm(test.data - self.ket0[i].data))

            ## envs should already be reinitialized
            # self.reinitialize_envs_i(ix)


    def _get_A0_effs(self, left_site_pos, nsites):

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]

        A_effs = []
        for i in range(self.num_operators):
            A_left  = self.A0_envsLs[i][left_site_pos]
            A_right = self.A0_envsRs[i][left_site_pos + nsites - 1]

            A_eff = qtn.TensorNetwork([])
            A = self.operators[i]
            if isinstance(site_inds, (list,tuple)):
                for si in site_inds:
                    A_eff.add(A[si])
            else:
                A_eff.add(A[site_inds])

            if A_left is not None:
                A_eff.add(A_left)
            if A_right is not None:
                A_eff.add(A_right)

            #### !!! CHANGED HERE
            A_eff.exponent += A.exponent
            A_effs += [A_eff]

        return A_effs



    def get_rk4_eff(self, dt, left_site_pos, nsites=1) -> Sequence['qtn.Tensor']:
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        ## this is just ket0 projected into the new basis.

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        def deriv_func(dist_submpx_, dt=1) -> 'qtn.Tensor':
            eff_Ax = []
            for deriv_tn in A_effs:
                # print('deriv_tn', deriv_tn)
                # print('dist submpx', dist_submpx_)
                eff_Ax += [qtn.TensorNetwork([deriv_tn, dist_submpx_])]
                # print('eff_ax', eff_Ax[-1].exponent, deriv_tn.exponent)
                # print('dist_submpx exponent', dist_submpx_.exponent)
                ## dist_submpx exponent should equal 0
            out = _sum_eff_TNs(eff_Ax, transpose_bonds=bonds_o)
            # out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
            out.reindex({bo: bi for bo, bi in zip(bonds_o, bonds_i)}, inplace=True)
            out.modify(apply=lambda x: x * dt)
            return out

        k1 = self.get_rk4_k1(dt, left_site_pos, nsites)

        T1 = dist_submpx.contract()
        k1.transpose_like(T1, inplace=True)
        s1 = k1.copy()
        s1.modify(apply=lambda x: x * 0.5 + T1.data)
        k2 = deriv_func(s1, dt=dt)  ## assumes constant A; should be A(t+dt/2)

        k2.transpose_like(T1, inplace=True)
        s2 = k2.copy()
        s2.modify(apply=lambda x: x * 0.5 + T1.data)
        k3 = deriv_func(s2, dt=dt)  ## assumes constant A; should be A(t+dt/2)

        k3.transpose_like(T1, inplace=True)
        s3 = k3.copy()
        s3.modify(apply=lambda x: x + T1.data)
        k4 = deriv_func(s3, dt=dt)

        return k1, k2, k3, k4


    def get_rk4_k1(self, dt, left_site_pos, nsites=1) -> 'qtn.Tensor':
        """ obtain first Runge-Kutta vector from ket0
        """
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        # dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        dist_submpx = qtn.TensorNetwork([self.ket0[ix] for ix in site_inds], virtual=True)

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A0_effs(left_site_pos, nsites)
        # print('ket0 orthog', left_site_pos)
        # helper.check_orthog(self.ket0)
        # print('bra0 orthog', left_site_pos)
        # helper.check_orthog(self.bra0)

        # ### project current ket onto old ket (ket0)
        # ovlp_L = self.S0_envsL[left_site_pos]
        # ovlp_R = self.S0_envsR[left_site_pos + nsites - 1]
        # if ovlp_L is not None:
        #     # ovlp_L = ovlp_L.conj()
        #     ## bra to ket, ket to temp
        #     l_ind = self.ket0.bond(left_site_pos, left_site_pos - 1)
        #     ovlp_L = ovlp_L.reindex({l_ind: l_ind + 'kk'})
        #     ovlp_L = ovlp_L.reindex({l_ind + '_': l_ind}, inplace=True)
        #     dist_submpx = dist_submpx.reindex({l_ind: l_ind + 'kk'})
        #     dist_submpx.add([ovlp_L])
        # if ovlp_R is not None:
        #     # ovlp_R = ovlp_R.conj()
        #     r_ind = self.ket0.bond(left_site_pos + nsites - 1, left_site_pos + nsites)
        #     ovlp_R = ovlp_R.reindex({r_ind: r_ind + 'kk'})
        #     ovlp_R = ovlp_R.reindex({r_ind + '_': r_ind}, inplace=True)
        #     dist_submpx = dist_submpx.reindex({r_ind: r_ind + 'kk'})
        #     dist_submpx.add([ovlp_R])
        #
        # ######

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        def deriv_func(dist_submpx_, dt=1) -> 'qtn.Tensor':
            eff_Ax = []
            for deriv_tn in A_effs:
                # print('deriv_tn', deriv_tn)
                # print('dist submpx', dist_submpx_)
                eff_Ax += [qtn.TensorNetwork([deriv_tn, dist_submpx_])]
                # print('eff_ax', eff_Ax[-1].exponent, deriv_tn.exponent)
                # print('dist_submpx exponent', dist_submpx_.exponent)
                ## dist_submpx exponent should equal 0
            out = _sum_eff_TNs(eff_Ax, transpose_bonds=bonds_o)
            # out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
            out.reindex({bo: bi for bo, bi in zip(bonds_o, bonds_i)}, inplace=True)
            out.modify(apply=lambda x: x * dt)
            return out

        k1 = deriv_func(dist_submpx, dt=dt)
        return k1


class TDDMRGSolver_v4(TDDMRGSolver_v2):
    """
    TD-DMRG by Feiguin and White, but with different targeted states (SSPRK)
    """

    def get_time_evolved_site(self, dt, left_site_pos, nsites=1, T1=None) -> 'qtn.Tensor':

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))
        bra_to_ket_inds = {b: k for k, b in ket_to_bra_inds.items()}

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)    # A exponent only
        b_effs = self._get_b_effs(left_site_pos, nsites)    # source exponent only
        b_effs_scaled = [helper.scalar_multiply(b_eff, 10 ** (-self.ket.exponent)) for b_eff in b_effs]

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        def deriv_func(dist_submpx_, dt=1, **kwargs) -> 'qtn.Tensor':
            eff_Ax = []
            for deriv_tn in A_effs:
                eff_Ax += [qtn.TensorNetwork([deriv_tn, dist_submpx_])]
                ## dist_submpx exponent should equal 0

            for source_tens in b_effs_scaled:
                eff_Ax += [qtn.TensorNetwork([source_tens])]

            out = _sum_eff_TNs(eff_Ax, transpose_bonds=bonds_o)
            # out = qtn.tensor_contract(deriv_tens, *dist_submpx_)
            out.reindex({bo: bi for bo, bi in zip(bonds_o, bonds_i)}, inplace=True)
            out.modify(apply=lambda x: x * dt)
            # print('v4 deriv', out.norm())
            return out

        def add_func(tens1, tens2, inplace=False, is_list=True, **kwargs) -> 'qtn.Tensor':
            if is_list:
                sum = [t.copy() for t in tens1] if inplace else tens1
            else:
                sum = tens1 if inplace else tens1.copy()

            if is_list:
                sum += tens2
                return sum
            else:
                if not isinstance(tens2, list):
                    tens2 = [tens2]
                for t2 in tens2:
                    if isinstance(t2, qtn.TensorNetwork):
                        t2 = t2.contract()
                        t2.reindex(bra_to_ket_inds, inplace=True)
                    sum = helper.add_tensors(sum, t2)
                return sum

        def scale_func(tens, scale_val, inplace=False, **kwargs) -> 'qtn.Tensor':
            out = tens if inplace else tens.copy()
            out.modify(apply=lambda x: x * scale_val)
            return out

        def euler_func(state, dt, deriv0, **kwargs) -> 'qtn.Tensor':
            return add_func(state, scale_func(deriv0, dt), is_list=False)


        T1 = dist_submpx.contract() if T1 is None else T1
        if dt == 0:
            return T1

        # print('td-dmrg targeting: ssprk4')
        # out = helper_TE.ssprk4(T1, dt, euler_func, deriv_func, add_func, scale_func)
        if self.te_order == 4: # or self.te_order == 0:
            print('td-dmrg targeting: rk4', self.te_order)
            out = helper_TE.rk4(T1, dt, euler_func, deriv_func, add_func, scale_func)

        # ###
        # def exact_func(state, dt, inplace=False, **kwargs):
        #
        #     state = state if inplace else state.copy()
        #
        #     deriv_tens = helper_tn.sum_eff_TNs(A_effs)
        #     deriv_tens.transpose(*bonds_o, *bonds_i, inplace=True)
        #
        #     if len(bonds_o) > 0:
        #         fuse_map = {'out': tuple(bonds_o), 'in': tuple(bonds_i)}
        #         proj_op = deriv_tens.fuse(fuse_map, inplace=False)
        #         proj_op.transpose('out', 'in', inplace=True)
        #     else:
        #         proj_op = deriv_tens
        #
        #     # print('proj op', proj_total.shape)
        #     if len(proj_op.inds) > 0:
        #         sp_expm = scipy.linalg.expm(proj_op.data * dt)
        #         ### line 331 in scipy.linalg._matfuncs.py: change empty to zeros
        #
        #         # print('is H', np.linalg.norm(proj_op.data - proj_op.data.conj().T),
        #         #       'is AH', np.linalg.norm(proj_op.data + proj_op.data.conj().T))
        #
        #         proj_op.modify(data=sp_expm)
        #
        #     else:
        #         proj_op.modify(apply=lambda x: np.exp(x * dt))
        #
        #     shape_i = [deriv_tens.ind_size(bi) for bi in bonds_i]
        #     shape_o = [deriv_tens.ind_size(bo) for bo in bonds_o]
        #     exp_total = proj_op.unfuse(fuse_map, {'out': shape_o, 'in': shape_i})
        #
        #     reindex_map = {bo: bi for bo, bi in zip(bonds_o, bonds_i)}
        #     active_tens: 'qtn.Tensor' = qtn.tensor_contract(state, exp_total)
        #     active_tens.reindex(reindex_map, inplace=True)
        #
        #     active_tens.transpose_like(state, inplace=True)
        #     state.modify(data=active_tens.data)
        #     return state
        #
        # if self.te_order == 0:
        #     print('td-dmrg targeting: exact')
        #     out = helper_TE.exact(T1, dt, exact_func)

        ####
        def scale_tn_func(tens, scale_val, inplace=False, **kwargs) -> 'qtn.TensorNetwork':
            # out = tens if inplace else tens.copy()
            if isinstance(tens, list):
                tens = [helper.scalar_multiply(t, scale_val, inplace=inplace) for t in tens]
            else:
                tens = helper.scalar_multiply(tens, scale_val, inplace=inplace)
            # out.modify(apply=lambda x: x * scale_val)
            return tens

        output_to_input_inds = {bo: bi for bo, bi in zip(bonds_o, bonds_i)}
        # print('Aeff', A_effs[0].outer_inds())
        # print('distmpx', dist_submpx.outer_inds())
        # print('output to input inds', output_to_input_inds)

        ## implicit solver
        def solve_func_cgd(deriv_ops, explicit_contribution, init_guess=None,
                           max_iter=None, conv_tol=None, **kwargs):

            conv_kwargs = {}
            conv_kwargs['max_iter'] = 500 if max_iter is None else max_iter
            conv_kwargs['conv_tol'] = 1.0e-6 if conv_tol is None else conv_tol

            init_guess = T1.copy()
            # print('T1 shape', T1.shape, 32 * 6)
            # init_guess.modify(apply=lambda x: x + np.random.random(x.shape) * 0.01)
            out, err = qtn_conjugate_gradient_squared_1site(deriv_ops, explicit_contribution, init_guess,
                                                            output_to_input_inds, **conv_kwargs)
            print('solve err', err)
            return out


        def solve_func_lsq(deriv_ops, explicit_contribution, init_guess=None,
                           max_iter=None, conv_tol=None, **kwargs):

            shape_ = explicit_contribution.shape
            size_ = int(np.prod(shape_))

            conv_kwargs = {}
            conv_kwargs['max_iter'] = 500 if max_iter is None else max_iter
            conv_kwargs['conv_tol'] = 1.0e-6 if conv_tol is None else conv_tol

            A_eff_tens = _sum_eff_TNs(deriv_ops, [*bonds_o, *bonds_i])
            explicit_contribution = explicit_contribution.transpose(*bonds_i)
            # print('A eff tens', A_eff_tens, explicit_contribution)

            Amat = A_eff_tens.data.reshape(size_,size_)
            bvec = explicit_contribution.data.reshape(size_)

            out_data = np.linalg.solve(Amat, bvec)

            out = qtn.Tensor(out_data.reshape(shape_), inds=bonds_i)

            # print('err', np.linalg.norm( Amat @ out_data - bvec ) )

            # init_guess = T1.copy()
            # init_guess.modify(apply=lambda x: x + np.random.random(x.shape) * 0.01)
            # out, err = qtn_conjugate_gradient_descent_1site(deriv_ops, explicit_contribution, init_guess,
            #                                                 output_to_input_inds, **conv_kwargs)
            # print('solve err', err)
            return out

        def identity_func(state: Union['qtn.Tensor', 'qtn.TensorNetwork']):
            shape_ = state.shape
            size_ = int(np.prod(shape_))
            iden = qtn.Tensor( np.eye(size_).reshape(*shape_, *shape_),
                               inds=(*bonds_o,*bonds_i) )
            return qtn.TensorNetwork([iden])

        if self.te_order == 22 or self.te_order ==0:
            print('td-dmrg targeting: crank-nicolson')
            out = helper_TE.solve_crank_nicolson(T1, euler_func, deriv_func, add_func, scale_tn_func,
                                                 A_effs, b_effs_scaled, identity_func, solve_func_cgd, dt)

        return out


    def get_site_total_rdm(self, dt, left_site_pos, x_bond, nsites=1, **kwargs) -> 'qtn.Tensor':
        """
        """
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        A_effs = self._get_A_effs(left_site_pos, nsites)

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        ## bond to contract over
        def compute_denmat(ket: 'qtn.Tensor'):
            bra = ket.conj()
            bra.reindex({k: b for k, b in ket_to_bra_inds.items() if k != x_bond}, inplace=True)
            return qtn.tensor_contract(bra, ket)

        rho_bonds_i = [bi for bi in bonds_i if bi != x_bond]
        rho_bonds_o = [ket_to_bra_inds[ind] for ind in rho_bonds_i]

        target_times = [0, dt/3, 2*dt/3, dt]
        weights = [1./3, 1./6, 1./6, 1./3]
        # target_times = [0, dt/3, 2*dt/3, 4*dt/3, dt]
        # weights = [1./3, 1./9, 1./9., 1./9, 1./3]
        # target_times = [0, (1 - np.sqrt(3)/2) * dt , dt, (1 + np.sqrt(3)/2) * dt ]
        # weights = [1./4, 1./4, 1./4, 1./4]
        # target_times = [dt]
        # weights = [1.]
        # target_times = [0, dt]
        # weights = [0.5, 0.5]

        ## for accumulated time steps
        sort_inds = np.argsort(target_times)
        target_times = [0] + [target_times[i + 1] - target_times[i] for i in sort_inds[:-1]]
        weights = [weights[i] for i in sort_inds]

        T1 = None
        tot_denmat = None
        for w, tt in zip(weights, target_times):
            # print('tt', tt)
            new_T1 = self.get_time_evolved_site(tt, left_site_pos, nsites=nsites, T1=T1)
            T1 = new_T1     ## for accumulated time steps

            rho0 = compute_denmat(new_T1)
            rho0.transpose(*rho_bonds_i, *rho_bonds_o, inplace=True)  # |ket><bra|

            if tot_denmat is None:
                tot_denmat = rho0.copy()
                tot_denmat.modify(apply=lambda x: x * w)
            else:
                rho0.transpose_like(tot_denmat, inplace=True)
                tot_denmat.modify(apply=lambda x: x + rho0.data * w)

        return tot_denmat


    def get_site_total_rdm_orig(self, dt, left_site_pos, x_bond, nsites=1, compress_direction=CompressDirection.RIGHT
                                ) -> 'qtn.Tensor':
        """equiv to original?"""

        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        # A_effs = self._get_A_effs(left_site_pos, nsites)

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        # ## A_eff * site projector (the back TE for repeated bond/site)
        if compress_direction == CompressDirection.RIGHT:
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

        ### build rk4 vectors
        T1 = dist_submpx.contract()
        k1, k2, k3, k4 = self.get_rk4_eff(dt, left_site_pos, nsites=nsites)
        ## if at end, can solve TE using other means

        ## targeted time steps

        T1.transpose_like(k1, inplace=True)
        psi13 = T1.copy()
        psi13.modify(apply=lambda x: x + 1. / 162 * (31 * k1.data + 14 * k2.data + 14 * k3.data - 5 * k4.data))
        psi23 = T1.copy()
        psi23.modify(apply=lambda x: x + 1. / 81 * (16 * k1.data + 20 * k2.data + 20 * k3.data - 2 * k4.data))
        psi33 = T1.copy()
        psi33.modify(apply=lambda x: x + 1. / 6 * (k1.data + 2 * k2.data + 2 * k3.data + k4.data))

        ## bond to contract over
        if compress_direction == CompressDirection.RIGHT:
            x_bond = self.ket.bond(ind1 + nsites - 1, ind1 + nsites)
        else:
            x_bond = self.ket.bond(ind1 - nsites + 1, ind1 - nsites)

        def compute_denmat(ket: 'qtn.Tensor'):
            bra = ket.conj()
            bra.reindex({k: b for k, b in ket_to_bra_inds.items() if k != x_bond}, inplace=True)
            return qtn.tensor_contract(bra, ket)

        rho_bonds_i = [bi for bi in bonds_i if bi != x_bond]
        rho_bonds_o = [ket_to_bra_inds[ind] for ind in rho_bonds_i]
        nb = len(rho_bonds_o)

        rho0 = compute_denmat(T1)
        rho0.transpose(*rho_bonds_i, *rho_bonds_o, inplace=True)  # |ket><bra|
        rho1 = compute_denmat(psi13)
        rho1.transpose_like(rho0, inplace=True)
        rho2 = compute_denmat(psi23)
        rho2.transpose_like(rho0, inplace=True)
        rho3 = compute_denmat(psi33)
        rho3.transpose_like(rho0, inplace=True)
        tot_denmat = rho0.copy()
        tot_denmat.modify(apply=lambda x: x * 1. / 3 + rho1.data * 1. / 6 +
                                          rho2.data * 1. / 6 + rho3.data * 1. / 3)

        return tot_denmat


    def _site_time_evolution_dmrg(self, dt, left_site_pos, proj=None, nsites=1,
                                  compress_direction=CompressDirection.RIGHT,
                                  ) -> tuple[Optional[qtn.MatrixProductState], Optional[qtn.Tensor]]:
        """
        sites: int or slice(start, stop, step)
        propagate sites forward in time via RK4 according to Fenguin + White
        proj is placeholder for compatibility with parent class. not actually used here
        """
        site_inds = self.mps_inds[left_site_pos:left_site_pos + nsites]
        dist_submpx = qtn.TensorNetwork([self.ket[ix] for ix in site_inds], virtual=True)
        submpx_tags = [self.ket.site_tag_id.format(ix) for ix in site_inds]

        ## change indices from bra (missing T*[i]) to ket
        ket_to_bra_inds = {}
        for site_p in site_inds:
            ket_to_bra_inds.update(self.get_ket_to_bra_inds(site_p))

        ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
        # A_effs = self._get_A_effs(left_site_pos, nsites)

        # ## A_eff * site projector (the back TE for repeated bond/site)
        if compress_direction == CompressDirection.RIGHT:
            ind1 = site_inds[0]
            ind2 = ind1 + 1
            at_end = ind1 == self.L - 1
            # at_end = (ind1 + nsites - 1 == self.L - 1)
        else:
            ind1 = site_inds[-1]
            ind2 = ind1 - 1
            at_end = ind1 == 0
            # at_end = (ind1 - nsites + 1 == 0)

        bonds_i = dist_submpx.outer_inds()
        bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

        ## targeted time steps
        if not at_end:

            ## bond to contract over
            if compress_direction == CompressDirection.RIGHT:
                x_bond = self.ket.bond(ind1 + nsites - 1, ind1 + nsites)
            else:
                x_bond = self.ket.bond(ind1 - nsites + 1, ind1 - nsites)

            ## target single state (at dt)
            # new_ket = self.get_time_evolved_site(dt, left_site_pos, nsites=nsites)
            # max_bond = self.max_bond // nsites if self.max_bond is not None else None
            # left_inds = [bi for bi in bonds_i if bi != x_bond]
            # eigvec, _ = qtn.tensor_split(new_ket, left_inds, absorb='right', bond_ind=x_bond + '_',
            #                              max_bond=max_bond)

            ## full density matrix method
            tot_denmat = self.get_site_total_rdm(dt, left_site_pos, x_bond, nsites,
                                                 compress_direction=compress_direction)

            rho_bonds_i = [bi for bi in bonds_i if bi != x_bond]
            rho_bonds_o = [ket_to_bra_inds[ind] for ind in rho_bonds_i]
            nb = len(rho_bonds_o)

            tot_denmat.transpose(*rho_bonds_i, *rho_bonds_o, inplace=True)

            denmat_shape = tot_denmat.shape
            tens_shape = denmat_shape[:nb]
            sq_shape = np.prod(denmat_shape[:nb])

            # test = tot_denmat.data.reshape(sq_shape, sq_shape)
            # print('rho is H', np.linalg.norm(test - test.T.conj()))
            eigval, eigvec = np.linalg.eigh(tot_denmat.data.reshape(sq_shape, sq_shape))

            sort_inds = np.argsort(np.abs(eigval))[::-1]
            ev_max = eigval[sort_inds[0]]
            # eigval_inds = np.where(np.abs(eigval) > 1.0e-12 * ev_max)
            new_sort_inds = []
            for si in sort_inds:
                if np.abs(eigval[si]) > 10e-20 * np.abs(ev_max):
                    new_sort_inds += [si]
                else:
                    break
            sort_inds = new_sort_inds

            if self.max_bond is not None:
                if self.max_bond // nsites > 0:
                    eigvec = eigvec[:, sort_inds[:self.max_bond // nsites]]
                # ## don't think this is the proper way to do 2-site, but DMRG doesn't seem to need 2-site
                # ## to reach full bond dimension. if don't divide by nsites, compressing eigvecs later causes
                # ## bad compression errors
                # print('eigvec teyp', eigvec.dtype)

            eigvec = eigvec.reshape(*tens_shape, -1)
            eigvec = qtn.Tensor(data=eigvec, inds=tuple(rho_bonds_i) + (x_bond + '_',))

            if nsites == 1:
                eigvec.transpose_like(self.ket[ind1], inplace=True)
                next_site = qtn.tensor_contract(eigvec.conj(), self.ket[ind1], self.ket[ind2])
                next_site.transpose_like(self.ket[ind2], inplace=True)
                self.ket[ind1].modify(data=eigvec.data)
                self.ket[ind2].modify(data=next_site.data)
                # print('ind1', ind1, ind2)
                # helper.check_orthog(self.ket)
            elif nsites == 2:
                ## decompose eigvec into two sites, tensor contract ind1 site
                # bond_m = self.ket.bond(ind1, ind2)
                bond_ms, left_inds = self.ket[ind1].filter_bonds(self.ket[ind2])
                bond_m = next(iter(bond_ms))
                # print('eigvec', eigvec)
                # print('left_inds', left_inds)
                ##
                ev1, ev2 = qtn.tensor_split(eigvec, left_inds, absorb='right', bond_ind=bond_m + '_',
                                            max_bond=self.max_bond)
                next_site = qtn.tensor_contract(ev1.conj(), self.ket[ind1], self.ket[ind2])
                ev1.transpose_like(self.ket[ind1], inplace=True)
                next_site.transpose_like(self.ket[ind2], inplace=True)
                self.ket[ind1].modify(data=ev1.data)
                self.ket[ind2].modify(data=next_site.data)
                # print('ind1', ind1, ind2)
                # helper.check_orthog(self.ket)
            else:
                raise NotImplementedError

        else:

            ### AL * A * AR = d/dT*[i] d/dT[i] <x|Ax>
            A_effs = self._get_A_effs(left_site_pos, nsites)

            b_effs = self._get_b_effs(left_site_pos, nsites)
            b_effs_scaled = [helper.scalar_multiply(b_eff, 10 ** (-self.ket.exponent)) for b_eff in b_effs]

            ## change indices from bra (missing T*[i]) to ket
            site_ind = self.mps_inds[left_site_pos]
            ket_to_bra_inds = self.get_ket_to_bra_inds(site_ind)

            bonds_i = dist_submpx.outer_inds()
            bonds_o = [ket_to_bra_inds[ind] for ind in bonds_i]

            site_time_evolution(dist_submpx, dt, A_effs, submpx_tags, bonds_i=bonds_i, bonds_o=bonds_o,
                                inplace=True, te_order=self.te_order,
                                sources=b_effs_scaled,
                                compress_direction=compress_direction,
                                compress_level=1, compress_opts_dict=self.compress_config)


        ## i think this is already done in canonize
        self.set_bra_from_ket(sites=list(range(left_site_pos, left_site_pos + nsites)))

        ## inplace update of ket, bra
        if compress_direction == CompressDirection.RIGHT:
            ## canonicalization is not needed here, but is used by subclass TDDMRGSolver_v3 for ket0, bra0
            if at_end:
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket

            for i in range(ind1, ind2):
                self._update_envs_left(i, canonize=False)
        else:
            if at_end:
                self.canonize(ind1, cur_orthog=ind1)  ## also sets bra from ket
            else:
                self.canonize(ind2, cur_orthog=ind1)  ## also sets bra from ket

            for i in range(ind1, ind2, -1):
                self._update_envs_right(i, canonize=False)

        return self.ket, None

