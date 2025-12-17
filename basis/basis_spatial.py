import scipy.sparse

import helper_quimb
from setup_.configs import *
import helper_quimb as helper
from basis.basis import Basis
import basis.findiff_coeffs as fd_coeff

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axis import Axis


class SpatialBasis(Basis):

    def __init__(self):
        self.type = BasisType.SPATIAL

    def get_realspace_1D(self, data, ax_ind, **kwargs):
        """ ie. coeffs of delta fcts """
        return data

    def get_realspace_nD(self, data: 'np.ndarray', ax_inds: list[int], x0: Numeric = 0.0, xL: Numeric = 1.0,
                         npts: Numeric = 128):
        """ ie. coeffs of delta fcts """
        return data

    # @classmethod
    # def elemental_multiply(cls, gtn1, gtn2, inplace=False, **compress_opts):
    #     gtn1 = gtn1 if inplace else gtn1.copy()
    #     gtn1.elemental_multiply(gtn2, **compress_opts)
    #     return gtn1

    def get_ones_mps(self, ax: 'Axis', site_ind_id='i({})', site_tag_id='X({})', anc_dim=None, anc_name_l=None,
                     anc_name_r=None) -> 'MPSType':
        q, L = ax.q, ax.L
        ones = np.ones((q,))
        if anc_dim is None:
            mps_0 = None
            if L >= 2:
                mps_0 = qtn.MatrixProductState(
                    [np.array([ones])] + [np.array([[ones]])] * (L - 2) + [np.array([ones])],
                    shape='lrp', site_tag_id=site_tag_id, site_ind_id=site_ind_id)
            elif L == 1:
                mps_0 = qtn.TensorNetwork(
                    [qtn.Tensor(ones, inds=(site_ind_id.format(0),), tags=(site_tag_id.format(0),))])
                mps_0.view_as(qtn.MatrixProductState, inplace=True, L=1, cyclic=False,
                              site_tag_id=site_tag_id, site_ind_id=site_ind_id)
            elif L == 0:
                mps_0 = qtn.TensorNetwork([])
                mps_0.view_as(qtn.MatrixProductState, inplace=True, L=0, cyclic=False,
                              site_tag_id=site_tag_id, site_ind_id=site_ind_id)
        else:
            ones = np.tensordot(np.eye(anc_dim), ones, axes=([], []))
            # ones_tensors = [qtn.Tensor(ones, inds=(blah, blah, site_ind_id.format(i),), tags=(site_tag_id.format(i),)) for i in range(L)]
            mps_0 = qtn.MatrixProductState([ones] * L,
                                           shape='lrp', site_tag_id=site_tag_id, site_ind_id=site_ind_id)
            mps_0[0].reindex({mps_0[0].inds[0]: anc_name_l}, inplace=True)
            mps_0[-1].reindex({mps_0[-1].inds[1]: anc_name_r}, inplace=True)

        return mps_0

    def build_elemental_multiply_tn(self, ax, in1_ind_id, in2_ind_id, out_ind_id, site_tag_id, cutoff=CUTOFF):
        """ d_ijk TN
        """
        L, q = ax.L, ax.q
        out = qtn.TensorNetwork([])
        for i in range(L):
            d_ijk = qtn.tensor_core.COPY_tensor(q, (in1_ind_id.format(i), in2_ind_id.format(i), out_ind_id.format(i)),
                                                tags=(site_tag_id.format(i),))
            out.add_tensor(d_ijk)
        # out.view_as(qtn.TensorNetwork1D, L=L, site_tag_id=site_tag_id, inplace=True)
        out.view_as(MatrixProductTensor, inplace=True, L=L, cyclic=False, site_tag_id=site_tag_id,
                    upper_ind_id=out_ind_id, lower_ind_id=in1_ind_id, extra_ind_ids=(in2_ind_id,))
        return out

    def build_xmultiply_mps(self, ax: 'Axis', x_power=1, offset=0.0, scale=1.0, split_opts: dict = None):
        """ (a(x-b))**c * f(x)
            ax: Axis object
        """
        mps = ax.map_state_to_mps((ax.xpts * scale + offset) ** x_power, split_opts=split_opts)
        return mps

    def build_xmultiply_mpo(self, ax: 'Axis', x_power=1, offset=0.0, scale=1.0, split_opts: dict = None):
        """ (a(x-b))**c * f(x)
            ax: Axis object
        """
        mps = ax.map_state_to_mps((ax.xpts * scale + offset) ** x_power, split_opts=split_opts)
        mpo = helper.mps_to_diag_mpo(mps, sparse=(ax.L==1))
        return mpo

    def build_firstderivative_mpo(self, ax, deriv_opts=None, compress_opts=None):
        """ get first derivative along specified axis of tn_grid (MPO along 1D)
        """
        q, L = ax.q, ax.L

        deriv_opts = {} if deriv_opts is None else deriv_opts
        left_bc = deriv_opts.get('left_bc', DEFAULT_BC)
        right_bc = deriv_opts.get('right_bc', DEFAULT_BC)
        order = deriv_opts.get('order', DEFAULT_ORDER)
        fd_type = deriv_opts.get('fd_type', DEFAULT_FDTYPE)
        offset = deriv_opts.get('offset', 0)
        offset_r = deriv_opts.get('offset_r', None)
        bc_value_l = deriv_opts.get('bc_value_l', 0.0)
        bc_value_r = deriv_opts.get('bc_value_r', 0.0)

        # print('build deriv mpo', order, left_bc, right_bc, fd_type)
        # print('deriv mpo', L)

        if L == 1:
            if fd_type == FDType.FORWARD:
                mpo = self._matrix_firstderivative_forward(L, q, left_bc, right_bc, order=order, bc_offset=offset,
                                                           bc_offset_r=offset_r,
                                                           bc_value_l=bc_value_l, bc_value_r=bc_value_r,
                                                           compress_opts=compress_opts)
            elif fd_type == FDType.BACKWARD:
                mpo = self._matrix_firstderivative_backward(L, q, left_bc, right_bc, order=order, bc_offset=offset,
                                                            bc_offset_r=offset_r,
                                                            bc_value_l=bc_value_l, bc_value_r=bc_value_r,
                                                            compress_opts=compress_opts)
            elif fd_type == FDType.CENTER:
                mpo = self._matrix_firstderivative_center(L, q, left_bc, right_bc, order=order, bc_offset=offset,
                                                          bc_offset_r=offset_r,
                                                          bc_value_l=bc_value_l, bc_value_r=bc_value_r,
                                                          compress_opts=compress_opts)
            else:
                raise NotImplementedError
        else:
            if fd_type == FDType.FORWARD:
                mpo = self._mpo_firstderivative_forward(L, q, left_bc, right_bc, order=order, bc_offset=offset,
                                                        bc_offset_r=offset_r,
                                                        bc_value_l=bc_value_l, bc_value_r=bc_value_r,
                                                        compress_opts=compress_opts)

                # ref_mpo = self._matrix_firstderivative_forward(1, q**L, left_bc, right_bc, order=order, bc_offset=offset,
                #                                            compress_opts=compress_opts)
                #
                # mpo_data = ax.map_mpo_to_operator(mpo)
                # ref_mpo_data = ref_mpo.to_dense()
                # print('mpo', np.round(mpo_data[-10:, -10:],4))
                # print('ref', np.round(ref_mpo_data[-10:, -10:],4))
                # exit()

            elif fd_type == FDType.BACKWARD:

                mpo = self._mpo_firstderivative_backward(L, q, left_bc, right_bc, order=order, bc_offset=offset,
                                                         bc_offset_r=offset_r,
                                                         bc_value_l=bc_value_l, bc_value_r=bc_value_r,
                                                         compress_opts=compress_opts)
                # ref_mpo = self._matrix_firstderivative_backward(1, q**L, left_bc, right_bc, order=order, bc_offset=offset,
                #                                                compress_opts=compress_opts)
                #
                # mpo_data = ax.map_mpo_to_operator(mpo)
                # ref_mpo_data = ref_mpo.to_dense()
                # print('mpo', np.round(mpo_data[:10, :10],4))
                # print('ref', np.round(ref_mpo_data[:10, :10],4))
                # exit()

            elif fd_type == FDType.CENTER:
                mpo = self._mpo_firstderivative_center(L, q, left_bc, right_bc, order=order, bc_offset=offset,
                                                       bc_offset_r=offset_r,
                                                       bc_value_l=bc_value_l, bc_value_r=bc_value_r,
                                                       compress_opts=compress_opts)

                # ref_mpo = self._matrix_firstderivative_center(1, q**L, left_bc, right_bc, order=order, bc_offset=offset,
                #                                                compress_opts=compress_opts)
                #
                # mpo_data = ax.map_mpo_to_operator(mpo)
                # ref_mpo_data = ref_mpo.to_dense()
                # print('mpo', np.round(mpo_data[:10, :10],4))
                # print('ref', np.round(ref_mpo_data[:10, :10],4))
                # print('mpo', np.round(mpo_data[-10:, -10:], 4))
                # print('ref', np.round(ref_mpo_data[-10:, -10:], 4))
                # exit()

            else:
                raise NotImplementedError

        # if mpo.L == 1:
        #     mpo_data = mpo[0].data
        #     mpo_data = mpo_data.todense()
        # else:
        #     mpo_data = ax.map_mpo_to_operator(mpo)
        # print('fd type', fd_type)
        # print('mpo', np.round(mpo_data[:10, :10],4))
        # print('mpo', np.round(mpo_data[-10:, -10:], 4))
        # exit()

        helper.scalar_multiply(mpo, 1. / ax.dx, inplace=True)
        # if ax.is_flipped:
        #     helper.mpo_flip_lr(mpo, inplace=True)
        mpo = ax.map.transform_mpo(mpo)
        return mpo


    def build_firstderivative_mpo_inverse(self, ax, deriv_opts=None, compress_opts=None):
        """ get first derivative along specified axis of tn_grid (MPO along 1D)
        """
        q, L = ax.q, ax.L

        deriv_opts = {} if deriv_opts is None else deriv_opts
        left_bc = deriv_opts.get('left_bc', DEFAULT_BC)
        right_bc = deriv_opts.get('right_bc', DEFAULT_BC)
        order = deriv_opts.get('order', DEFAULT_ORDER)
        fd_type = deriv_opts.get('fd_type', DEFAULT_FDTYPE)
        offset = deriv_opts.get('offset', 0)
        offset_r = deriv_opts.get('offset_r', None)
        bc_value_l = deriv_opts.get('bc_value_l', 0.0)
        bc_value_r = deriv_opts.get('bc_value_r', 0.0)

        # print('build deriv mpo', order, left_bc, right_bc, fd_type)
        # print('deriv mpo', L)

        if fd_type == FDType.FORWARD:
            ddx = self._matrix_firstderivative_forward(1, q**L, left_bc, right_bc, order=order, bc_offset=offset,
                                                       bc_offset_r=offset_r,
                                                       bc_value_l=bc_value_l, bc_value_r=bc_value_r,
                                                       compress_opts=compress_opts)
        elif fd_type == FDType.BACKWARD:
            ddx = self._matrix_firstderivative_backward(1, q**L, left_bc, right_bc, order=order, bc_offset=offset,
                                                        bc_offset_r=offset_r,
                                                        bc_value_l=bc_value_l, bc_value_r=bc_value_r,
                                                        compress_opts=compress_opts)
        elif fd_type == FDType.CENTER:
            ddx = self._matrix_firstderivative_center(1, q**L, left_bc, right_bc, order=order, bc_offset=offset,
                                                      bc_offset_r=offset_r,
                                                      bc_value_l=bc_value_l, bc_value_r=bc_value_r,
                                                      compress_opts=compress_opts)
        else:
            raise NotImplementedError

        ddx_mat = ddx[0].data.toarray()
        # if fd_type == FDType.BACKWARD:
        #     ddx_mat[0,:] = np.ones(q**L)   # average = 0
        # else:
        #     ddx_mat[-1, :] = np.ones(q ** L)  # average = 0
        ### the problem with doing the above is that it doesn't yield the identity when actually acting on d/dx

        # print('ddx mat', ddx_mat[:8, :8])
        # ddx_inv = np.linalg.pinv(ddx_mat )
        # print('ddx inv', ddx_inv[:8, :8])
        # u, s, vt = np.linalg.svd(ddx_mat)
        # print('s', s, len(s), q**L)

        print('ddx mat', ddx_mat)
        ddx_inv = np.tril(np.ones((q**L,q**L)))  ## first order integration
        # ddx_inv[:,0] = 0.5
        # ddx_inv[0,0] = 0
        print('ddx inv', ddx_inv)
        print('inv', (ddx_mat @ ddx_inv)[:8,:8])
        print('inv', (ddx_inv @ ddx_mat)[:8,:8])
        print('inv', np.linalg.norm(ddx_mat @ ddx_inv - np.eye(q ** L)))
        print('inv', np.linalg.norm(ddx_inv @ ddx_mat - np.eye(q ** L)))
        mpo_inv = ax.map_operator_to_mpo(ddx_inv, split_opts={'cutoff': 1.0e-28, 'cutoff_mode': CUTOFF_MODE})
        helper_quimb.scalar_multiply(mpo_inv, ax.dx, inplace=True)
        return mpo_inv


    def build_secondderivative_mpo(self, ax, deriv_opts=None, compress_opts=None, eeo_grid=False):
        """ get first derivative along specified axis
        """
        q, L = ax.q, ax.L
        fd_type = FDType.CENTER

        deriv_opts = {} if deriv_opts is None else deriv_opts
        left_bc = deriv_opts.get('left_bc', DEFAULT_BC)
        right_bc = deriv_opts.get('right_bc', DEFAULT_BC)
        order = deriv_opts.get('order', DEFAULT_ORDER)
        fd_type = deriv_opts.get('fd_type', DEFAULT_FDTYPE)
        offset = deriv_opts.get('offset', 0)
        offset_r = deriv_opts.get('offset_r', None)
        bc_value_l = deriv_opts.get('bc_value_l', 0.0)
        bc_value_r = deriv_opts.get('bc_value_r', 0.0)

        fd_type, order = FDType.CENTER, 1
        print('Warning: forcing 2nd deriv to centered stencil, order 1')
        if fd_type == FDType.CENTER:
            if L == 1:
                mpo = self._matrix_secondderivative_center(L, q, left_bc, right_bc, order=order, bc_offset=offset,
                                                           bc_offset_r=offset_r, compress_opts=compress_opts,
                                                           bc_value_l=bc_value_l, bc_value_r=bc_value_r,
                                                           eeo_grid=eeo_grid)
            else:
                mpo = self._mpo_secondderivative_center(L, q, left_bc, right_bc, order=order, bc_offset=offset,
                                                        bc_offset_r=offset_r, compress_opts=compress_opts,
                                                        bc_value_l=bc_value_l, bc_value_r=bc_value_r,
                                                        eeo_grid=eeo_grid)
                # ref = self._matrix_secondderivative_center(1, q**L, left_bc, right_bc, order=order, bc_offset=offset,
                #                                            compress_opts=compress_opts)
                #
                # mpo_data = ax.map_mpo_to_operator(mpo)
                # ref_data = ref.to_dense()
                # print('mpo', np.round(mpo_data[:10, :10],4))
                # print('ref', np.round(ref_data[:10, :10],4))
                # print('mpo', np.round(mpo_data[-10:, -10:], 4))
                # print('ref', np.round(ref_data[-10:, -10:], 4))
                # exit()

        else:
            raise NotImplementedError

        helper.scalar_multiply(mpo, 1. / ax.dx ** 2, inplace=True)
        # if ax.is_flipped:
        #     helper.mpo_flip_lr(mpo, inplace=True)
        mpo = ax.map.transform_mpo(mpo)
        return mpo


    def build_mth_derivative_mpo(self, ax: 'Axis', deriv_order: int, deriv_opts: dict = None,
                                 compress_opts: dict = None, eeo_grid=False) -> MPOType:

        deriv_opts = {} if deriv_opts is None else deriv_opts
        left_bc = deriv_opts.get('left_bc', DEFAULT_BC)
        right_bc = deriv_opts.get('right_bc', DEFAULT_BC)
        order = deriv_opts.get('order', DEFAULT_ORDER)
        fd_type = deriv_opts.get('fd_type', DEFAULT_FDTYPE)
        offset = deriv_opts.get('offset', 0)
        offset_r = deriv_opts.get('offset_r', None)
        bc_value_l = deriv_opts.get('bc_value_l', 0.0)
        bc_value_r = deriv_opts.get('bc_value_r', 0.0)

        mpo = self._mpo_higher_order_derivative(ax.L, ax.q, deriv_order, left_bc=left_bc, right_bc=right_bc,
                                                 order=order, fd_type=fd_type)

        helper.scalar_multiply(mpo, 1. / ax.dx ** deriv_order, inplace=True)
        mpo = ax.map.transform_mpo(mpo)
        return mpo


    def get_integral_weight(self, ax):
        """ normalization for integral_mps
        """
        if ax.endpoint and ax.startpoint:  # contains both start and endpoint -- use trapezoidal rule
            return self.build_integral_mps(ax)

            # integ_mps = ax.get_iden_mps()
            # endpt_correction = ax.get_select_elems_mps([0, ax.npts-1])
            # helper.scalar_multiply(endpt_correction, -0.5, inplace=True)
            # integ_mps = helper.add_MPS(integ_mps, endpt_correction, inplace=True)
            #
            # # plt.figure()
            # # plt.plot(ax.map_mps_to_state(integ_mps))
            # # plt.title('ax integ weights mps')
            # # plt.show()
            #
            # helper.scalar_multiply(integ_mps, ax.dx, inplace=True)
            # ## integ_mps = ax.map.transform_mps(integ_mps)  ## already taken care of with select_elems
            # return integ_mps
        else:
            return ax.dx

    def build_integral_mps(self, ax, is_sqrt=False, site_ind_id='i({})', site_tag_id='T({})'):
        """ get MPS that integrates out axis
        """
        if ax.is_even:
            integ_mps = ax.get_iden_mps()
            if ax.startpoint and ax.endpoint:  # contains startpoint and endpoint -- use trapezoidal rule
                endpt_correction = ax.get_select_elems_mps([0, ax.npts - 1])
                endpt_correction = helper.scalar_multiply(endpt_correction, -0.5)
                integ_mps = helper.add_MPS(integ_mps, endpt_correction)

                # plt.figure()
                # plt.plot(ax.map_mps_to_state(integ_mps))
                # plt.title('ax integ mps')
                # plt.show()

            helper.scalar_multiply(integ_mps, ax.dx, inplace=True)

        else:
            raise NotImplementedError
            ## something like
            # integ_mps = ax.map_state_to_mps(ax.dx)
        return integ_mps


    def build_coarse_grain_mpx(self, ax: 'Axis', depth:int, is_sqrt=False, upper_ind_id='o({})', lower_ind_id='i({})',
                               site_tag_id='T({})') -> MPOType:
        """ get MPX that coarse grains a function on Axis 'ax'
            performs partial integration (trapezoidal)
            returns MPO with some tensor cores missing upper index
        """
        Axis = ax.__class__
        fine_ax = Axis(depth, q=ax.q, dx=ax.dx, endpoint=False)
        fine_integral = fine_ax.get_integral_mps(lower_ind_id, site_tag_id)     ## uniform ones * dx
        helper.scalar_multiply(fine_integral, 1./ax.dx * ax.q**(-depth), inplace=True)
        ## why? averaging over num grid points bc dx with be scaled by q**depth
        ## also remove * dx multiplication

        coarse_ax = Axis(ax.L - depth, q=ax.q, dx=ax.dx*(ax.q**depth))
        coarse_iden = coarse_ax.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id)

        out_mpx = helper_quimb.append_mpx(coarse_iden, fine_integral)
        out_mpx = ax.map.transform_mpo(out_mpx)
        return out_mpx


    def build_coarse_select_mpx(self, ax: 'Axis', depth: int, is_sqrt=False, upper_ind_id='o({})', lower_ind_id='i({})',
                               site_tag_id='T({})') -> MPOType:
        """ get MPX that coarse grains a function on Axis 'ax'
        """
        Axis = ax.__class__
        fine_ax = Axis(depth, q=ax.q, dx=ax.dx, endpoint=False)
        fine_integral = fine_ax.get_select_elems_mps([0], site_ind_id=lower_ind_id, site_tag_id=site_tag_id)  ## uniform ones * dx
        # helper.scalar_multiply(fine_integral, 1. / ax.dx * ax.q ** (-depth), inplace=True)
        ## why? averaging over num grid points bc dx with be scaled by q**depth
        ## also remove * dx multiplication

        coarse_ax = Axis(ax.L - depth, q=ax.q, dx=ax.dx * (ax.q ** depth))
        coarse_iden = coarse_ax.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id)

        out_mpx = helper_quimb.append_mpx(coarse_iden, fine_integral)
        out_mpx = ax.map.transform_mpo(out_mpx)
        return out_mpx



    def build_indefinite_integral_mps(self, ax, order=1):
        """ get MPS that takes indefinite integral along axis
        """
        raise NotImplementedError

    # def get_correction_term(self, L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order: int = DEFAULT_ORDER,
    #                         bc_value_l: Numeric = 0.0, bc_value_r: Numeric = 0.0):
    #
    #     correction_L = np.zeros((order,))
    #     correction_R = np.zeros((order,))
    #
    #     ## left bc
    #     if left_bc > 0:     ## Neumann / symmetric /
    #         correction_R =
    #     else:

    def _mpo_firstderivative_center(self, L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                    bc_offset=0, bc_offset_r=None, bc_value_l=0.0, bc_value_r=0.0,
                                    compress_opts=None):
        """ S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>a
            is for a 1D system so dim does not need to be specified
            scale by 1/dt later
            boundary_condition:  boundary condition to use when taking derivative

            FD coeffs found using python package FinDiff
        """
        verbose = False

        if q != 2:
            raise NotImplementedError('check q=2 implementation for binary mapping, esp if not pbc')

        if order == 1:
            c1, c2, c3, c4 = (1. / 2, 0, 0, 0)
        elif order == 2:
            c1, c2, c3, c4 = (2. / 3, -1. / 12, 0, 0)
        elif order == 3:
            c1, c2, c3, c4 = (3. / 4, -3. / 20, 1. / 60, 0)
        elif order == 4:
            c1, c2, c3, c4 = (4. / 5, -1. / 5, 4. / 105, -1. / 280)
        else:
            return self._mpo_higher_order_derivative(L, q, 1, left_bc, right_bc, order=order,
                                                     compress_opts=compress_opts)

        coeffs = np.array([0, c1, c2, c3, c4])

        #### build n-ary +/- operator ####
        ## define operator such that operating on the desired spatial dimension
        sp = np.diag([1, ] * (q - 1), k=-1)
        sm = np.diag([1, ] * (q - 1), k=1)

        ## build MPO
        iden = np.eye(q)
        zero = np.zeros((q, q))

        # bc_offset, bc_offset_r = 2, -2
        # left_bc, right_bc = BCType.ZEROGRADIENT, BCType.ZEROGRADIENT

        if verbose:
            print('center diff', left_bc, right_bc, order, bc_offset, bc_offset_r)
            print('coeffs', coeffs)

        if left_bc == BCType.PERIODIC:
            mpo_tens = [np.array([iden, sm + sp, sp + sm])]  ## ignore dummy size 1 bond
            for i in range(1, L - 2):
                mpo_tens += [np.array([[iden, sm, sp],
                                       [zero, sp, zero],
                                       [zero, zero, sm]])]
            mpo_tens += [np.array([[iden, sm, sp, zero, zero],
                                   [zero, sp, zero, iden, zero],
                                   [zero, zero, sm, zero, iden]])]
            mpo_tens += [np.array([c1 * (sm - sp), (c1 * sp + c2 * iden + c3 * sm), -(c1 * sm + c2 * iden + c3 * sp),
                                   (c4 * iden + c3 * sp), -(c4 * iden + c3 * sm)])]
            ## ignore dummy size 1 bond
            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')

        elif left_bc == BCType.ANTIPERIODIC:
            mpo_tens = [np.array([iden, sm - sp, sp - sm])]  ## ignore dummy size 1 bond
            for i in range(1, L - 2):
                mpo_tens += [np.array([[iden, sm, sp],
                                       [zero, sp, zero],
                                       [zero, zero, sm]])]
            mpo_tens += [np.array([[iden, sm, sp, zero, zero],
                                   [zero, sp, zero, iden, zero],
                                   [zero, zero, sm, zero, iden]])]
            mpo_tens += [np.array([c1 * (sm - sp), (c1 * sp + c2 * iden + c3 * sm), -(c1 * sm + c2 * iden + c3 * sp),
                                   (c4 * iden + c3 * sp), -(c4 * iden + c3 * sm)])]
            ## ignore dummy size 1 bond
            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')

        else:
            m0 = np.diag([1.] + [0.] * (q - 1))
            m1 = np.diag([0.] * (q - 1) + [1.])

            nb = 4  # int(np.ceil(np.log2(order*2+1)))   ## number of bits to encode boundary condition
            bc0 = np.zeros((q ** nb, q ** nb))
            bc1 = np.zeros((q ** nb, q ** nb))

            def get_bulk_boundary_stencil():
                """ function that returns bulk coefficientss at the boundaries
                """
                ## edge boundary conditions: use the same number of points in stencil
                if order == 1:  # 3-pt stencil
                    fd_mat_0 = np.array([[0, c1, c2]])
                    # fd_mat_L = fd_mat_0[:, ::-1] * -1

                elif order == 2:  # 5-pt stencil
                    ## 000.. points
                    v0 = np.array([0., c1, c2, c3, c4])
                    v1 = np.array([-c1, 0., c1, c2, c3])
                    fd_mat_0 = np.array([v0, v1])

                    ## 111.. points
                    # fd_mat_L = np.array([v1[::-1], v0[::-1]]) * -1

                elif order == 3:  # 7-pt stencil
                    v0 = np.array([0., c1, c2, c3, c4, 0., 0.])
                    v1 = np.array([-c1, 0., c1, c2, c3, c4, 0.])
                    v2 = np.array([-c2, -c1, 0., c1, c2, c3, c4])
                    fd_mat_0 = np.array([v0, v1, v2])
                    # fd_mat_L = np.array([v2[::-1], v1[::-1], v0[::-1]]) * -1

                elif order == 4:  # order == 4
                    v0 = np.array([0., c1, c2, c3, c4, 0., 0., 0., 0.])
                    v1 = np.array([-c1, 0., c1, c2, c3, c4, 0., 0., 0.])
                    v2 = np.array([-c2, -c1, 0., c1, c2, c3, c4, 0., 0.])
                    v3 = np.array([-c3, -c2, -c1, 0., c1, c2, c3, c4, 0.])
                    fd_mat_0 = np.array([v0, v1, v2, v3])
                    # fd_mat_L = np.array([v3[::-1], v2[::-1], v1[::-1], v0[::-1]]) * -1

                else:
                    raise ValueError(f'order must be 1-4, not {order}')

                fd_mat_L = fd_mat_0[::-1, ::-1] * -1

                return fd_mat_0, fd_mat_L

            def get_gen_boundary_stencil(num_stencil=2 * order + 1, offset=0):
                """ function that returns open boundary condition stencils in matrix form
                """
                import findiff

                coeffs = []
                for x0 in range(offset, offset + order):
                    stencil = findiff.coefficients(1, offsets=list(range(-x0, num_stencil - x0)))
                    coeffs += [stencil['coefficients']]
                    # print('stencil', stencil)

                end_bc_mat_0 = np.array(coeffs)
                end_bc_mat_L = end_bc_mat_0[::-1, ::-1] * -1
                return end_bc_mat_0, end_bc_mat_L

            left_bulk, right_bulk = get_bulk_boundary_stencil()

            ## left-hand side boundaries
            if left_bc == BCType.SYMMETRIC or left_bc == BCType.ANTISYMMETRIC \
                    or left_bc == BCType.ZEROGRADIENT or left_bc == BCType.ZEROVALUE \
                    or left_bc == BCType.NEUMANN or left_bc == BCType.DIRICHLET \
                    or left_bc == BCType.ABSORBING or left_bc == BCType.REFLECTING:
                ## for bc_offset == 0
                ## add np.array([[c1, c2, c3, c4], [c2, c3, c4, 0.], [c3, c4, 0., 0.], [c4, 0., 0., 0.]])
                ## to bulk matrix

                ### bc_offset == 0 if zero is included at the boundary
                ### bc_offset > 0 means left boundary y=0 not included // (max = 2; full-step offset)
                ### bc_offset < 0 means left boundary y=0 is included.

                num_boundary = order - min(0, bc_offset)
                end_bc_mat_0 = np.zeros((num_boundary, 2 * order + max(0, bc_offset)))
                correction_0 = np.zeros((num_boundary,))

                for i in range(num_boundary):
                    nc = order - i

                    if nc < bc_offset - 1:
                        continue

                    if bc_offset == 1:      ## x_(1/2), x_(3/2), ...
                        # # reflected from symmetry axis
                        end_bc_mat_0[i, :nc] -= coeffs[i + 1:order + 1] * np.sign(left_bc.value)

                        ## corrections:
                        # print('np.sum', coeffs[i + 1:order + 1])
                        if left_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## if Dirichlet with non-zero value (B), need to add correction term of 2B
                            correction_0[i] = -2 * bc_value_l * np.sum(coeffs[i + 1:order + 1])
                            if bc_value_l != 0.:
                                raise NotImplementedError
                        elif left_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
                            pass
                            ## we don't actually need to know B, just x_(1/2) = x_(-1/2), etc.
                        else:
                            raise NotImplementedError

                    elif bc_offset == 2:    ## x_1, x_2, ...
                        # reflected from symmetry axis
                        end_bc_mat_0[i, :nc - 1] -= coeffs[i + 2:order + 1] * np.sign(left_bc.value)

                        ## corrections
                        # print('np.sum', coeffs[i + 1:order + 1])
                        if left_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## if Dirichlet with non-zero value (B),
                            ## need to add correction term of B for endpoint and 2B for remaining points
                            weights = coeffs[i + 1] + 2 * np.sum(coeffs[i + 2:order + 1])
                            correction_0[i] = -1 * bc_value_l * weights
                            if bc_value_l != 0.:
                                raise NotImplementedError
                        elif left_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
                            ## assuming derivative = 0 at x_(0)
                            ## order 1: B = (4 x_(1), -x_(2) ) / 3
                            ## order 2: B = (15 x_(1), -6 x_(2), x_(3) ) / 10
                            ## order 3: B = (56 x_(1), -28 x_(2), 8 x_(3), -x_(4) ) / 35
                            ## order 4: B = (210 x_(1), -120 x_(2), 45 x_(3), -10 x_(4), x_(5) ) / 126
                            weight = coeffs[i + 1]
                            if order == 1:
                                end_bc_mat_0[i, 0] -= 4 / 3 * weight
                                end_bc_mat_0[i, 1] -= -1 / 3 * weight
                            elif order == 2:
                                end_bc_mat_0[i, 0] -= 15 / 10 * weight
                                end_bc_mat_0[i, 1] -= -6 / 10 * weight
                                end_bc_mat_0[i, 2] -= 1 / 10 * weight
                            elif order == 3:
                                end_bc_mat_0[i, 0] -= 56 / 35 * weight
                                end_bc_mat_0[i, 1] -= -28 / 35 * weight
                                end_bc_mat_0[i, 2] -= 8 / 35 * weight
                                end_bc_mat_0[i, 3] -= -1 / 35 * weight
                            elif order == 4:
                                end_bc_mat_0[i, 0] -= 210 / 126 * weight
                                end_bc_mat_0[i, 1] -= -120 / 126 * weight
                                end_bc_mat_0[i, 2] -= 45 / 126 * weight
                                end_bc_mat_0[i, 3] -= -10 / 126 * weight
                                end_bc_mat_0[i, 4] -= 1 / 126 * weight
                            else:
                                raise NotImplementedError

                    elif bc_offset == 0:    ## x_0, x_1, x_2, ...
                        # reflected from symmetry axis
                        end_bc_mat_0[i, 1:nc + 1] -= coeffs[1 + i:order + 1] * np.sign(left_bc.value)

                        ## corrections
                        # print('np.sum', coeffs[i + 1:order + 1])
                        if left_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## DIRCHLET BC: needs correction term from boundary value (which is included)
                            # correction_0[i] = -2 * bc_value_l * np.sum(coeffs[i + 1:order + 1])
                            end_bc_mat_0[i, 0] -= 2 * np.sum(coeffs[i + 1:order + 1])

                    elif bc_offset == -1:   ## x_(-1/2), x_(1/2), x_(3/2), ...
                        # reflected from symmetry axis
                        end_bc_mat_0[i, 2: 2 + nc] -= coeffs[i + 1:order + 1] * np.sign(left_bc.value)

                        ## DIRCHLET BC: needs correction term from boundary value
                        # print('np.sum', coeffs[i + 1:order + 1])
                        if left_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            # for all orders: B = (x_(-1/2) + x_(1/2)) / 2
                            weight = np.sum(coeffs[i + 1: order + 1])  # * 2
                            end_bc_mat_0[i, 0] -= weight
                            end_bc_mat_0[i, 1] -= weight
                        else:
                            ## without enforced symmetry of x_(-1/2), x_(1/2), difficult to determine
                            ## x_(-3/2), x_(-5/2), ... even though we can compute the expected value at the boundary
                            raise ValueError(f'boundary condition {left_bc} with offset {bc_offset} cannot be '
                                             f'easily determined')
                    else:
                        raise NotImplementedError

            elif left_bc == BCType.OPEN:
                end_bc_mat_0 = -left_bulk
                end_bc_mat_0 += get_gen_boundary_stencil(offset=0)[0]

            else:
                raise ValueError(f'{left_bc} not a valid BCType')

            ## right-hand side boundaries
            if right_bc == BCType.SYMMETRIC or right_bc == BCType.ANTISYMMETRIC \
                    or right_bc == BCType.ZEROGRADIENT or right_bc == BCType.ZEROVALUE \
                    or right_bc == BCType.NEUMANN or right_bc == BCType.DIRICHLET \
                    or right_bc == BCType.REFLECTING or right_bc == BCType.ABSORBING:
                ## build matrix out of coefficients, which will be added to bulk matrix

                ### bc_offset == 0 if zero is included at the boundary
                ### bc_offset < 0 means right boundary y=L not included (min = -2; full-step offset)
                ### bc_offset > 0 means right boundary y=L is included.

                bc_offset_r = bc_offset if bc_offset_r is None else bc_offset_r
                offset = max(0, bc_offset_r)
                num_boundary = order + offset

                end_bc_mat_1 = np.zeros((num_boundary, 2 * order + offset))
                correction_1 = np.zeros((num_boundary,))
                # bc_ind = bc_offset + 1

                for i in range(num_boundary):
                    nc = order - i
                    if nc < abs(bc_offset_r) - 1:
                        continue

                    if bc_offset_r == 1:       # ..., x_(L-3/2), x_(L-1/2), x_(L+1/2)
                        # reflected from symmetry axis
                        ## end_bc_mat_1[-1 - i, -nc:] += coeffs[i + 1:order + 1][::-1] * np.sign(right_bc.value)
                        ## end_bc_mat_0[i, 2: 2 + nc] -= coeffs[i + 1:order + 1] * np.sign(left_bc.value)
                        end_bc_mat_1[-1 - i, -2 - nc:-2] += coeffs[i + 1:order + 1][::-1] * np.sign(right_bc.value)
                        ## corrections:
                        # print('np.sum', coeffs[i + 1:order + 1])
                        if right_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## DIRCHLET BC: needs correction term from boundary value
                            # for all orders: B = (x_(L-1/2) + x_(L+1/2)) / 2
                            weight = np.sum(coeffs[i + 1: order + 1])  # * 2
                            end_bc_mat_1[-1 - i, -1] += weight
                            end_bc_mat_1[-1 - i, -2] += weight
                        else:
                            ## without enforced symmetry of x_(L-1/2), x_(L+1/2), difficult to determine
                            ## x_(L+3/2), x_(L+5/2), ... even though we can compute the expected value at the boundary
                            raise ValueError(f'boundary condition {right_bc} with offset {bc_offset_r} cannot be '
                                             f'easily determined')

                    elif bc_offset_r == 0:      # ..., x_(L-1), x_L
                        # reflected from symmetry axis
                        end_bc_mat_1[-1 - i, 2 * order - nc - 1:2 * order - 1] += \
                            coeffs[i + 1:order + 1][::-1] * np.sign(right_bc.value)

                        ## corrections
                        # print('np.sum', coeffs[i + 1:order + 1])
                        if right_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## DIRCHLET BC: needs correction term from boundary value (which is included)
                            # correction_0[i] = -2 * bc_value_l * np.sum(coeffs[i + 1:order + 1])
                            end_bc_mat_1[-1 - i, -1] += 2 * np.sum(coeffs[i + 1:order + 1])

                    elif bc_offset_r == -1:      # ..., x_(L-3/2), x_(L-1/2)
                        # reflected from symmetry axis
                        ## end_bc_mat_1[-1 - i, -2 - nc:-2] += coeffs[i + 1:order +1][::-1] * np.sign(right_bc.value)
                        end_bc_mat_1[-1 - i, -nc:] += coeffs[i + 1:order + 1][::-1] * np.sign(right_bc.value)

                        ## corrections:
                        # print('np.sum', coeffs[i + 1:order + 1])
                        if right_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## if Dirichlet with non-zero value (B), need to add correction term of 2B
                            correction_1[-1 - i] = 2 * bc_value_r * np.sum(coeffs[i + 1:order + 1])
                            if bc_value_r != 0.:
                                raise NotImplementedError
                        elif right_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
                            pass
                            ## we don't actually need to know B, just x_(1/2) = x_(-1/2), etc.

                    elif bc_offset_r == -2:     # ..., x_(L-2), x_(L-1)
                        if nc > 1:
                            # print('coeffs', coeffs[i+ 2: order + 1])
                            end_bc_mat_1[-1 - i, -nc + 1:] += coeffs[i + 2:order + 1][::-1] * np.sign(right_bc.value)

                        ## corrections
                        if right_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## if Dirichlet with non-zero value (B),
                            ## need to add correction term of B for endpoint and 2B for remaining points
                            weights = coeffs[i + 1] + 2 * np.sum(coeffs[i + 2:order + 1])
                            correction_1[i] = -1 * bc_value_r * weights
                            if bc_value_r != 0.:
                                raise NotImplementedError
                        elif right_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
                            ## assuming derivative = 0 at x_(0)
                            ## order 1: B = (4 x_(1), -x_(2) ) / 3
                            ## order 2: B = (15 x_(1), -6 x_(2), x_(3) ) / 10
                            ## order 3: B = (56 x_(1), -28 x_(2), 8 x_(3), -x_(4) ) / 35
                            ## order 4: B = (210 x_(1), -120 x_(2), 45 x_(3), -10 x_(4), x_(5) ) / 126
                            weight = coeffs[i + 1]
                            # print('right bc  -2 SYMMETRIC', weight, order)
                            if order == 1:
                                end_bc_mat_1[-1 - i, -1] += 4 / 3 * weight
                                end_bc_mat_1[-1 - i, -2] += -1 / 3 * weight
                            elif order == 2:
                                end_bc_mat_1[-1 - i, -1] += 15 / 10 * weight
                                end_bc_mat_1[-1 - i, -2] += -6 / 10 * weight
                                end_bc_mat_1[-1 - i, -3] += 1 / 10 * weight
                            elif order == 3:
                                end_bc_mat_1[-1 - i, -1] += 56 / 35 * weight
                                end_bc_mat_1[-1 - i, -2] += -28 / 35 * weight
                                end_bc_mat_1[-1 - i, -3] += 8 / 35 * weight
                                end_bc_mat_1[-1 - i, -4] += -1 / 35 * weight
                            elif order == 4:
                                end_bc_mat_1[-1 - i, -1] += 210 / 126 * weight
                                end_bc_mat_1[-1 - i, -2] += -120 / 126 * weight
                                end_bc_mat_1[-1 - i, -3] += 45 / 126 * weight
                                end_bc_mat_1[-1 - i, -4] += -10 / 126 * weight
                                end_bc_mat_1[-1 - i, -5] += 1 / 126 * weight
                            else:
                                raise NotImplementedError

                    else:
                        raise NotImplementedError

            elif right_bc == BCType.OPEN:
                end_bc_mat_1 = -right_bulk
                end_bc_mat_1 += get_gen_boundary_stencil(offset=0)[1]

            else:
                raise ValueError(f'{right_bc} not a valid BCType')

            nx, ny = end_bc_mat_0.shape
            bc0[:nx, :ny] = end_bc_mat_0
            nx, ny = end_bc_mat_1.shape
            bc1[-nx:, -ny:] = end_bc_mat_1

            if verbose:
                # np.set_printoptions(precision=4)
                print('end bc mat 0')
                print(end_bc_mat_0)
                print(correction_0)

                print('end bc mat 1')
                print(end_bc_mat_1)
                print(correction_1)

            ## 0 side
            bc0 = np.reshape(bc0, (q,) * nb * 2)
            bc0_tens = qtn.Tensor(bc0, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                  [f'i({i})' for i in range(nb)]))
            bc0_mpo = helper.mpx_from_dense(bc0_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')

            ## L-1 side
            bc1 = np.reshape(bc1, (q,) * nb * 2)
            bc1_tens = qtn.Tensor(bc1, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                  [f'i({i})' for i in range(nb)]))
            bc1_mpo = helper.mpx_from_dense(bc1_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')

            ## leading 00... or 1... strings
            if L - nb >= 2:
                mpo_0 = qtn.MatrixProductOperator(
                    [np.array([m0])] + [np.array([[m0]])] * (L - nb - 2) + [np.array([m0])],
                    shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                mpo_1 = qtn.MatrixProductOperator(
                    [np.array([m1])] + [np.array([[m1]])] * (L - nb - 2) + [np.array([m1])],
                    shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
            elif L - nb == 1:
                mpo_0 = qtn.TensorNetwork([qtn.Tensor(m0, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
                mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                              site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                mpo_1 = qtn.TensorNetwork([qtn.Tensor(m1, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
                mpo_1.view_as(qtn.MatrixProductOperator, like=mpo_0, inplace=True)
            elif L - nb == 0:
                mpo_0 = qtn.TensorNetwork([])
                mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                              site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                mpo_1 = qtn.TensorNetwork([])
                mpo_1.view_as(qtn.MatrixProductOperator, like=mpo_0, inplace=True)
            else:
                raise ValueError('FD order too large for current grid size')

            if bc0_mpo is not None:
                helper.append_mpx(mpo_0, bc0_mpo, inplace=True)
            else:
                if not (bc_offset == 2 and order==1 and left_bc.value < 0):
                    raise ValueError('end bc correction is zero when it should not be?')
                mpo_0 = None

            if bc1_mpo is not None:
                helper.append_mpx(mpo_1, bc1_mpo, inplace=True)
            else:
                if not (bc_offset_r == -2 and order==1 and right_bc.value < 0):
                    raise ValueError('end bc correction is zero when it should not be?')
                mpo_1 = None

            ### the bulk mpo
            mpo_tens = [np.array([iden, sm, sp])]  ## ignore dummy size 1 bond
            for i in range(1, L - 2):
                mpo_tens += [np.array([[iden, sm, sp],
                                       [zero, sp, zero],
                                       [zero, zero, sm]])]
            mpo_tens += [np.array([[iden, sm, sp, zero, zero],
                                   [zero, sp, zero, iden, zero],
                                   [zero, zero, sm, zero, iden]])]
            mpo_tens += [np.array([c1 * (sm - sp), (c1 * sp + c2 * iden + c3 * sm), -(c1 * sm + c2 * iden + c3 * sp),
                                   (c4 * iden + c3 * sp), -(c4 * iden + c3 * sm)])]  ## ignore size 1 bond

            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')
            helper.add_MPO(mpo, mpo_0, inplace=True)
            helper.add_MPO(mpo, mpo_1, inplace=True)

            helper.compress(mpo, compress_opts=compress_opts)

        # if True:  # bc_offset != 0:
        #     print('center diff')
        #     print('left bc', left_bc, 'right bc', right_bc)
        #     op_tens = mpo.contract(all) * 10 ** mpo.exponent
        #     op_tens.transpose(*[mpo.upper_ind_id.format(i) for i in range(mpo.L)],
        #                       *[mpo.lower_ind_id.format(i) for i in range(mpo.L)],
        #                       inplace=True)
        #     op_mat = op_tens.data.reshape(2 ** mpo.L, 2 ** mpo.L)
        #     print('op mat', left_bc, right_bc, bc_offset, bc_offset_r)
        #     print('L', op_mat[:5, :5])
        #     print('R', op_mat[-5:, -5:])
        #
        # exit()

        return mpo

    @classmethod
    def _mpo_firstderivative_forward(self, L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                     bc_offset=0, bc_offset_r=None, bc_value_l=0.0, bc_value_r=0.0, compress_opts=None):
        """ S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>
            is for a 1D system so dim does not need to be specified
            scale by 1/dt later
            forward diff or backward diff
            boundary_condition:  boundary condition to use when taking derivative

            FD coeffs found using python package FinDiff
        """
        print('forward FD mpo')
        # print('config', left_bc, right_bc, order)

        if q != 2:
            raise NotImplementedError('check q=2 implementation for binary mapping, esp if not pbc')

        bc_offset_r = bc_offset - 2 if bc_offset_r is None else bc_offset_r
        # right_bc = BCType.ANTISYMMETRIC
        # order = 2
        # bc_offset_r = -1
        print('bc', right_bc, 'order', order, 'bc offset_r', bc_offset_r)

        acc = 2 * order if order > 0 else 1
        coeffs_fd = fd_coeff.coefficients(deriv=1, acc=acc)
        c = coeffs_fd['forward']['coefficients']
        coeffs = c
        # print('FD coeffs', c)

        nc = len(c)
        nb = int(np.floor(np.log(nc + max(0, bc_offset_r)) / np.log(q)) + 1)  ## number of sites at the end
        c = np.append(c, np.zeros(q ** nb - nc))
        # c *= -1

        #### build n-ary +/- operator ####
        ## define operator such that operating on the desired spatial dimension
        sm = np.diag([1, ] * (q - 1), k=1)
        sp = np.diag([1, ] * (q - 1), k=-1)

        ## build MPO
        iden = np.eye(q)
        zero = np.zeros((q, q))

        sign = right_bc.value  # -1, 1 if AP or P; else some other int
        if abs(sign) != 1:
            sign = 0

        #### build finite difference MPO for PBC/APBC ####
        if nb == 2:
            if abs(sign) == 1:
                mpo_tens = [np.array([iden, sm, sign * sp])]  ## ignore dummy size 1 bond
                for i in range(1, L - nb):
                    mpo_tens += [np.array([[iden, sm, zero],
                                           [zero, sp, zero],
                                           [zero, zero, sp]])]
                mpo_tens += [np.array([[iden, sm, zero],
                                       [zero, sp, zero],
                                       [zero, sp, iden]])]
                mpo_tens += [np.array([c[0] * iden + c[1] * sm,
                                       c[1] * sp + c[2] * iden + c[3] * sm,
                                       c[3] * sp])]
            else:

                mpo_tens = [np.array([iden, sm])]  ## ignore dummy size 1 bond
                for i in range(1, L - nb):
                    mpo_tens += [np.array([[iden, sm],
                                           [zero, sp]])]
                mpo_tens += [np.array([[iden, sm],
                                       [zero, sp]])]
                mpo_tens += [np.array([c[0] * iden + c[1] * sm,
                                       c[1] * sp + c[2] * iden + c[3] * sm])]

            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')
        else:
            npts = q ** L
            op = np.zeros((npts,) * 2)
            for i in range(npts):
                for j in range(nc):
                    if i + j < npts:
                        op[i, i + j] = c[j]
                    elif sign:
                        op[i, (i + j) % npts] = c[j] * sign

            op_tensor = np.reshape(op, (q,) * (2 * L))
            op_tensor = qtn.Tensor(op_tensor, inds=[f'o({i})' for i in range(L)] + \
                                                   [f'i({i})' for i in range(L)])
            mpo = helper.mpx_from_dense(op_tensor, L, ('o({})', 'i({})'), site_tag_id='B{}')

        #### get boundary conditions (subtract existing coeffs used above) ####
        #### bc forward diff, only right bc matters ####

        if abs(sign) != 1:  # not periodic or antiperiodic, apply bc's

            num_stencil = acc + 1
            right_bulk = np.zeros((num_stencil - 1, num_stencil))
            for i in range(num_stencil - 1):
                for j in range(nc):
                    try:
                        right_bulk[i, i + j + 1] = c[j]
                    except IndexError:
                        continue

            def get_gen_boundary_stencil(num_stencil=num_stencil, offset=0):
                """ function that returns open boundary condition stencils in matrix form
                    offset must be less than num_stencil
                """
                coeffs = np.zeros((num_stencil - 1, num_stencil + offset))
                for x0 in range(0, offset - 1):
                    stencil = fd_coeff.coefficients(deriv=1, offsets=list(range(num_stencil)))
                    coeffs[x0, x0 + 1:x0 + 1 + num_stencil] = stencil['coefficients']
                    # print('stencil', x0, stencil)
                for x0 in range(offset - 1, num_stencil - 1):
                    i0 = (offset - 1) - x0
                    stencil = fd_coeff.coefficients(deriv=1, offsets=list(range(i0, i0 + num_stencil)))
                    # print('stencil', x0, stencil)
                    coeffs[x0, -num_stencil:] = stencil['coefficients']

                end_bc_mat_L = np.array(coeffs)
                # print('gen bc', end_bc_mat_L)
                return end_bc_mat_L

            # print('c', c)
            # print('coeffs', coeffs)

            if right_bc == BCType.SYMMETRIC or right_bc == BCType.ANTISYMMETRIC \
                    or right_bc == BCType.ZEROGRADIENT or right_bc == BCType.ZEROVALUE \
                    or right_bc == BCType.REFLECTING or right_bc == BCType.ABSORBING:
                ## build matrix out of coefficients, which will be added to bulk matrix

                ### bc_offset == 0 if zero is included at the boundary
                ### bc_offset < 0 means y=0 not included (min = -1; half-step offset)
                ### bc_offset > 0 means y=0 is included.

                # bc_offset_r = bc_offset if bc_offset_r is None else bc_offset_r
                offset = max(0, bc_offset_r)
                num_boundary = acc + offset + 1

                end_bc_mat_1 = np.zeros((num_boundary, acc + offset + 1))
                correction_1 = np.zeros((num_boundary,))
                # bc_ind = bc_offset + 1

                for i in range(num_boundary):
                    nc = acc - i
                    if nc < abs(bc_offset_r) - 1:
                        continue

                    # print('i', i, 'nc', nc)

                    if bc_offset_r == 1:  # ..., x_(L-3/2), x_(L-1/2), x_(L+1/2)
                        # reflected from symmetry axis
                        end_bc_mat_1[-1 - i, -2 - nc:-2] += coeffs[i + 1:acc + 1][::-1] * np.sign(right_bc.value)
                        ## corrections:
                        # print('np.sum', coeffs[i + 1:acc + 1])
                        if right_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## DIRCHLET BC: needs correction term from boundary value
                            # for all orders: B = (x_(L-1/2) + x_(L+1/2)) / 2
                            weight = np.sum(coeffs[i + 1: acc + 1])
                            end_bc_mat_1[-1 - i, -1] += weight
                            end_bc_mat_1[-1 - i, -2] += weight
                        else:
                            ## without enforced symmetry of x_(L-1/2), x_(L+1/2), difficult to determine
                            ## x_(L+3/2), x_(L+5/2), ... even though we can compute the expected value at the boundary
                            raise ValueError(f'boundary condition {right_bc} with offset {bc_offset_r} cannot be '
                                             f'easily determined')

                    elif bc_offset_r == 0:  # ..., x_(L-1), x_L
                        # reflected from symmetry axis
                        end_bc_mat_1[-1 - i, - nc - 1:-1] += \
                            coeffs[i + 1:acc + 1][::-1] * np.sign(right_bc.value)

                        ## corrections
                        # print('np.sum', coeffs[i + 1:acc + 1])
                        if right_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## DIRCHLET BC: needs correction term from boundary value (which is included)
                            end_bc_mat_1[-1 - i, -1] += 2 * np.sum(coeffs[i + 1:acc + 1])

                        elif right_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
                            ## derivative is zero, so cancel everything out
                            if i == 0:
                                end_bc_mat_1[-1 - i, :] *= 0
                                end_bc_mat_1[-1 - i, -1] -= np.sum(coeffs[0])

                        # print('end bc mat 1', end_bc_mat_1)


                    elif bc_offset_r == -1:  # ..., x_(L-3/2), x_(L-1/2)
                        # reflected from symmetry axis
                        if nc > 0:
                            end_bc_mat_1[-1 - i, -nc:] += coeffs[i + 1:acc + 1][::-1] * np.sign(right_bc.value)

                        ## corrections:
                        # print('np.sum', coeffs[i + 1:acc + 1])
                        if right_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## if Dirichlet with non-zero value (B), need to add correction term of 2B
                            correction_1[-1 - i] = 2 * bc_value_r * np.sum(coeffs[i + 1:acc + 1])
                            if bc_value_r != 0.:
                                raise NotImplementedError
                        elif right_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
                            pass    ## we don't actually need to know B, just x_(1/2) = x_(-1/2), etc.

                    elif bc_offset_r == -2:  # ..., x_(L-2), x_(L-1)
                        print('i', i, 'nc', nc, acc)
                        if nc > 1:
                            end_bc_mat_1[-1 - i, -nc + 1:] += coeffs[i + 2:acc + 1][::-1] * np.sign(right_bc.value)

                        ## corrections
                        # print('coeffs piece', coeffs[i + 1: acc + 1])
                        if right_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## if Dirichlet with non-zero value (B),
                            ## need to add correction term of B for endpoint and 2B for remaining points
                            weights = coeffs[i + 1] + 2 * np.sum(coeffs[i + 2:acc + 1])
                            correction_1[i] = bc_value_r * weights
                            if bc_value_r != 0.:
                                raise NotImplementedError

                        elif right_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
                            ## assuming derivative = 0 at x_(L), value at boundary B = x_L
                            ## order 0: B = (-x_(1) )
                            ## order 1: B = (4 x_(1), -x_(2) ) / 3
                            ## order 2: B = (15 x_(1), -6 x_(2), x_(3) ) / 10
                            ## order 3: B = (56 x_(1), -28 x_(2), 8 x_(3), -x_(4) ) / 35
                            ## order 4: B = (210 x_(1), -120 x_(2), 45 x_(3), -10 x_(4), x_(5) ) / 126
                            weight = coeffs[i + 1]
                            # print('right bc  -2 SYMMETRIC', weight, order)
                            if order == 0:
                                end_bc_mat_1[-1 - i, -1] += -1 * weight
                            elif order == 1:
                                end_bc_mat_1[-1 - i, -1] += 4 / 3 * weight
                                end_bc_mat_1[-1 - i, -2] += -1 / 3 * weight
                            elif order == 2:
                                end_bc_mat_1[-1 - i, -1] += 15 / 10 * weight
                                end_bc_mat_1[-1 - i, -2] += -6 / 10 * weight
                                end_bc_mat_1[-1 - i, -3] += 1 / 10 * weight
                            elif order == 3:
                                end_bc_mat_1[-1 - i, -1] += 56 / 35 * weight
                                end_bc_mat_1[-1 - i, -2] += -28 / 35 * weight
                                end_bc_mat_1[-1 - i, -3] += 8 / 35 * weight
                                end_bc_mat_1[-1 - i, -4] += -1 / 35 * weight
                            elif order == 4:
                                end_bc_mat_1[-1 - i, -1] += 210 / 126 * weight
                                end_bc_mat_1[-1 - i, -2] += -120 / 126 * weight
                                end_bc_mat_1[-1 - i, -3] += 45 / 126 * weight
                                end_bc_mat_1[-1 - i, -4] += -10 / 126 * weight
                                end_bc_mat_1[-1 - i, -5] += 1 / 126 * weight
                            else:
                                raise NotImplementedError

                    else:
                        raise NotImplementedError

            elif right_bc == BCType.OPEN:

                end_bc_mat_1 = -right_bulk
                end_bc_mat_1 += get_gen_boundary_stencil(offset=0)

            else:
                raise ValueError(f'{right_bc} not a valid BCType')

            # print('end bc mat 1')
            # print(end_bc_mat_1)

            if np.linalg.norm(end_bc_mat_1) > 1.0e-12:
                bc1 = np.zeros((q ** nb, q ** nb))
                nx, ny = end_bc_mat_1.shape
                bc1[-nx:, -ny:] = end_bc_mat_1

                m0 = np.diag([1.] + [0.] * (q - 1))
                m1 = np.diag([0.] * (q - 1) + [1.])

                ## L-1 side
                bc1 = np.reshape(bc1, (q,) * nb * 2)
                bc1_tens = qtn.Tensor(bc1, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                      [f'i({i})' for i in range(nb)]))
                bc1_mpo = helper.mpx_from_dense(bc1_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')

                ## leading 00... or 1... strings
                if L - nb >= 2:
                    mpo_1 = qtn.MatrixProductOperator(
                        [np.array([m1])] + [np.array([[m1]])] * (L - nb - 2) + [np.array([m1])],
                        shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                elif L - nb == 1:
                    mpo_1 = qtn.TensorNetwork([qtn.Tensor(m1, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
                    mpo_1.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                                  site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                elif L - nb == 0:
                    mpo_1 = qtn.TensorNetwork([])
                    mpo_1.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                                  site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                else:
                    raise ValueError('FD order too large for current grid size')

                helper.append_mpx(mpo_1, bc1_mpo, inplace=True)
                helper.add_MPO(mpo, mpo_1, inplace=True)

            helper.compress(mpo, compress_opts=compress_opts)

        # if True:  # bc_offset != 0:
        #     print('forward diff')
        #     op_tens = mpo.contract(all) * 10 ** mpo.exponent
        #     op_tens.transpose(*[mpo.upper_ind_id.format(i) for i in range(mpo.L)],
        #                       *[mpo.lower_ind_id.format(i) for i in range(mpo.L)],
        #                       inplace=True)
        #     op_mat = op_tens.data.reshape(2 ** mpo.L, 2 ** mpo.L)
        #     print('op mat', left_bc, right_bc, bc_offset, bc_offset_r)
        #     print('L', op_mat[:5, :5])
        #     print('R', op_mat[-5:, -5:])

        return mpo

    def _mpo_firstderivative_backward(self, L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                      bc_offset=0, bc_offset_r=None, bc_value_l=0.0, bc_value_r=0.0,
                                      compress_opts=None):
        """ S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>
            is for a 1D system so dim does not need to be specified
            scale by 1/dt later
            forward diff or backward diff
            boundary_condition:  boundary condition to use when taking derivative

            FD coeffs found using python package FinDiff
        """
        if q != 2:
            raise NotImplementedError('check q=2 implementation for binary mapping, esp if not pbc')

        print('backward FD mpo')

        # left_bc = BCType.ANTISYMMETRIC
        # order = 2
        # bc_offset = 2
        print('bc', left_bc, 'order', order, 'bc offset', bc_offset)

        acc = 2 * order if order > 0 else 1
        coeffs_fd = fd_coeff.coefficients(deriv=1, acc=acc)
        # coeffs_fd = fd_coeff.coefficients(deriv=1, acc=2 * order if order>0 else 1)
        c = coeffs_fd['backward']['coefficients'][::-1]

        nc = len(c)
        nb = int(np.floor(np.log(nc + max(0, -bc_offset)) / np.log(q)) + 1)  ## number of sites at the end
        c = np.append(c, np.zeros(q ** nb - nc))

        #### build n-ary +/- operator ####
        ## define operator such that operating on the desired spatial dimension
        sm = np.diag([1, ] * (q - 1), k=1)
        sp = np.diag([1, ] * (q - 1), k=-1)

        ## build MPO
        iden = np.eye(q)
        zero = np.zeros((q, q))

        sign = left_bc.value  # 1, -1 if P, AP; else some other number
        if abs(sign) != 1:
            sign = 0

        #### build finite difference MPO for PBC/APBC ####
        if nb == 2:
            if abs(sign) == 1:
                mpo_tens = [np.array([iden, sp, sign * sm])]  ## ignore dummy size 1 bond
                for i in range(1, L - nb):
                    mpo_tens += [np.array([[iden, sp, zero],
                                           [zero, sm, zero],
                                           [zero, zero, sm]])]

                mpo_tens += [np.array([[iden, sp, zero],
                                       [zero, sm, zero],
                                       [zero, sm, iden]])]
                mpo_tens += [np.array([c[0] * iden + c[1] * sp,
                                       c[1] * sm + c[2] * iden + c[3] * sp,
                                       c[3] * sm])]

            else:
                mpo_tens = [np.array([iden, sp])]  ## ignore dummy size 1 bond
                for i in range(1, L - nb):
                    mpo_tens += [np.array([[iden, sp],
                                           [zero, sm]])]

                mpo_tens += [np.array([[iden, sp],
                                       [zero, sm]])]
                mpo_tens += [np.array([c[0] * iden + c[1] * sp,
                                       c[1] * sm + c[2] * iden + c[3] * sp])]

            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')
        else:
            npts = q ** L
            op = np.zeros((npts,) * 2)
            for i in range(npts):
                for j in range(nc):
                    if i - j >= 0:
                        op[i, i - j] = c[j]
                    elif sign:
                        op[i, i - j] = c[j] * sign

            op_tensor = np.reshape(op, (q,) * (2 * L))
            op_tensor = qtn.Tensor(op_tensor, inds=[f'o({i})' for i in range(L)] + \
                                                   [f'i({i})' for i in range(L)])
            mpo = helper.mpx_from_dense(op_tensor, L, ('o({})', 'i({})'), site_tag_id='B{}')

        #### get boundary conditions (subtract existing coeffs used above) ####
        #### bc forward diff, only right bc matters ####

        if abs(sign) != 1:  # not periodic or antiperiodic, apply bc's

            num_stencil = acc + 1
            left_bulk = np.zeros((num_stencil - 1, num_stencil))
            for i in range(num_stencil - 1):
                for j in range(nc):
                    if i - j < 0:
                        continue
                    else:
                        left_bulk[i, i - j] = c[j]

            # coeffs_mat = np.zeros((q ** nb,) * 2)
            # for i in range(q ** nb):
            #     for j in range(nc):
            #         if i - j < 0:
            #             continue
            #         else:
            #             coeffs_mat[i, i - j] = c[j]

            # print('left bulk', left_bulk)

            def get_gen_boundary_stencil(num_stencil=num_stencil, offset=0):
                """ function that returns open boundary condition stencils in matrix form
                    offset must be less than num_stencil
                """
                coeffs = np.zeros((num_stencil - 1, num_stencil + offset))
                for x0 in range(num_stencil - offset - 1):
                    i0 = -x0 - offset
                    stencil = fd_coeff.coefficients(deriv=1, offsets=list(range(i0, i0 + num_stencil)))
                    coeffs[x0, :num_stencil] = stencil['coefficients']

                for x0 in range(num_stencil - offset - 1, num_stencil - 1):
                    stencil = fd_coeff.coefficients(deriv=1, offsets=list(range(-num_stencil + 1, 1)))
                    coeffs[x0, x0 - (num_stencil - offset) + 1:x0 + offset + 1] = stencil['coefficients']

                end_bc_mat_0 = coeffs
                # print('gen bc', end_bc_mat_0)

                return end_bc_mat_0

            if left_bc == BCType.SYMMETRIC or left_bc == BCType.ANTISYMMETRIC \
                    or left_bc == BCType.ZEROGRADIENT or left_bc == BCType.ZEROVALUE \
                    or left_bc == BCType.ABSORBING or left_bc == BCType.REFLECTING:
                ## add np.array([[c1, c2, c3, c4], [c2, c3, c4, 0.], [c3, c4, 0., 0.], [c4, 0., 0., 0.]])
                ## to bulk matrix

                ### bc_offset == 0 if zero is included at the boundary
                ### bc_offset > 0 means y=0 not included (max = 1; half-step offset)
                ### bc_offset < 0 means y=0 is included.

                num_boundary = acc + max(0, bc_offset)
                end_bc_mat_0 = np.zeros((num_boundary, acc + 1 + max(0, bc_offset)))

                # print('coeffs', coeffs)
                # print('c', c)

                num_boundary = acc - min(0, bc_offset) + 1
                end_bc_mat_0 = np.zeros((num_boundary, acc - min(0, bc_offset) + 1))
                correction_0 = np.zeros((num_boundary,))

                for i in range(num_boundary):
                    nc = acc - i

                    if nc < bc_offset:
                        continue

                    if bc_offset == 1:  ## x_(1/2), x_(3/2), ...
                        ## desired stencil
                        end_bc_mat_0[i, :nc] += c[i + 1:acc + 1] * np.sign(left_bc.value)
                        ## c is already negative of the forward difference stencil

                        ## corrections:
                        # print('np.sum', c[i + 1:acc + 1])
                        if left_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## if Dirichlet with non-zero value (B), need to add correction term of 2B
                            correction_0[i] = 2 * bc_value_l * np.sum(c[i + 1:acc + 1])
                            if bc_value_l != 0.:
                                raise NotImplementedError
                        elif left_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
                            pass    ## we don't actually need to know B, just x_(1/2) = x_(-1/2), etc.
                        else:
                            raise NotImplementedError

                    elif bc_offset == 2:  ## x_1, x_2, ...
                        # reflected from symmetry axis
                        end_bc_mat_0[i, :nc - 1] += c[i + 2:acc + 1] * np.sign(left_bc.value)

                        ## corrections
                        # print('np.sum', c[i + 1:acc + 1])
                        if left_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## if Dirichlet with non-zero value (B),
                            ## need to add correction term of B for endpoint and 2B for remaining points
                            weights = c[i + 1] + 2 * np.sum(c[i + 2:acc + 1])
                            correction_0[i] = bc_value_l * weights
                            if bc_value_l != 0.:
                                raise NotImplementedError
                        elif left_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
                            ## assuming derivative = 0 at x_(0)
                            ## order 1: B = (4 x_(1), -x_(2) ) / 3
                            ## order 2: B = (15 x_(1), -6 x_(2), x_(3) ) / 10
                            ## order 3: B = (56 x_(1), -28 x_(2), 8 x_(3), -x_(4) ) / 35
                            ## order 4: B = (210 x_(1), -120 x_(2), 45 x_(3), -10 x_(4), x_(5) ) / 126
                            weight = c[i + 1]
                            if order <= 1:
                                end_bc_mat_0[i, 0] += 4 / 3 * weight
                                end_bc_mat_0[i, 1] += -1 / 3 * weight
                            elif order == 2:
                                end_bc_mat_0[i, 0] += 15 / 10 * weight
                                end_bc_mat_0[i, 1] += -6 / 10 * weight
                                end_bc_mat_0[i, 2] += 1 / 10 * weight
                            elif order == 3:
                                end_bc_mat_0[i, 0] += 56 / 35 * weight
                                end_bc_mat_0[i, 1] += -28 / 35 * weight
                                end_bc_mat_0[i, 2] += 8 / 35 * weight
                                end_bc_mat_0[i, 3] += -1 / 35 * weight
                            elif order == 4:
                                end_bc_mat_0[i, 0] += 210 / 126 * weight
                                end_bc_mat_0[i, 1] += -120 / 126 * weight
                                end_bc_mat_0[i, 2] += 45 / 126 * weight
                                end_bc_mat_0[i, 3] += -10 / 126 * weight
                                end_bc_mat_0[i, 4] += 1 / 126 * weight
                            else:
                                raise NotImplementedError

                    elif bc_offset == 0:  ## x_0, x_1, x_2, ...
                        # reflected from symmetry axis
                        # print('i', i, nc, i + 1, order - 1)
                        end_bc_mat_0[i, 1:nc + 1] += c[i + 1:acc + 1] * np.sign(left_bc.value)

                        ## corrections
                        # print('np.sum', c[i + 1:acc + 1])
                        if left_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## DIRCHLET BC: needs correction term from boundary value (which is included)
                            # correction_0[i] = -2 * bc_value_l * np.sum(coeffs[i + 1:order + 1])
                            end_bc_mat_0[i, 0] += 2 * np.sum(c[i + 1:acc + 1])
                        elif left_bc in [BCType.SYMMETRIC, BCType.NEUMANN, BCType.ZEROGRADIENT]:
                            ## cancel out all the terms so that derivative is 0
                            if i == 0:
                                end_bc_mat_0[i, :] *= 0
                                end_bc_mat_0[i, 0] -= c[0]

                    elif bc_offset == -1:  ## x_(-1/2), x_(1/2), x_(3/2), ...
                        # reflected from symmetry axis
                        end_bc_mat_0[i, 2: 2 + nc] += c[i + 1:acc + 1] * np.sign(left_bc.value)

                        # ## DIRCHLET BC: needs correction term from boundary value
                        # print('np.sum', c[i + 1:acc + 1])
                        if left_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            # for all orders: B = (x_(-1/2) + x_(1/2)) / 2
                            weight = np.sum(c[i + 1: order + 1])
                            end_bc_mat_0[i, 0] += weight
                            end_bc_mat_0[i, 1] += weight
                        else:
                            ## without enforced symmetry of x_(-1/2), x_(1/2), difficult to determine
                            ## x_(-3/2), x_(-5/2), ... even though we can compute the expected value at the boundary
                            raise ValueError(f'boundary condition {left_bc} with offset {bc_offset} cannot be '
                                             f'easily determined')
                    else:
                        raise NotImplementedError

                # print('end bc mat 0')
                # print(end_bc_mat_0)

            elif left_bc == BCType.OPEN:

                end_bc_mat_0 = -left_bulk
                end_bc_mat_0 += get_gen_boundary_stencil(offset=0)

            else:
                raise ValueError(f'{left_bc} not a valid BCType')

            if np.linalg.norm(end_bc_mat_0) > 1.0e-12:
                bc0 = np.zeros((q ** nb, q ** nb))
                nx, ny = end_bc_mat_0.shape
                bc0[:nx, :ny] = end_bc_mat_0

                m0 = np.diag([1.] + [0.] * (q - 1))
                m1 = np.diag([0.] * (q - 1) + [1.])

                ## 0 side
                bc0 = np.reshape(bc0, (q,) * nb * 2)
                bc0_tens = qtn.Tensor(bc0, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                      [f'i({i})' for i in range(nb)]))
                # print('bc0', bc0)
                # print('bc0', bc0_tens)
                bc0_mpo = helper.mpx_from_dense(bc0_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')
                # print('bc0 mpo', bc0_mpo)

                ## leading 00... or 1... strings
                if L - nb >= 2:
                    mpo_0 = qtn.MatrixProductOperator(
                        [np.array([m0])] + [np.array([[m0]])] * (L - nb - 2) + [np.array([m0])],
                        shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                elif L - nb == 1:
                    mpo_0 = qtn.TensorNetwork([qtn.Tensor(m0, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
                    mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                                  site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                elif L - nb == 0:
                    mpo_0 = qtn.TensorNetwork([])
                    mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                                  site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                else:
                    raise ValueError('FD order too large for current grid size')

                # print('mpo_0', mpo_0)
                # print('bc0', bc0_mpo)
                helper.append_mpx(mpo_0, bc0_mpo, inplace=True)
                helper.add_MPO(mpo, mpo_0, inplace=True)

            helper.compress(mpo, compress_opts=compress_opts)

        # if True:  # bc_offset != 0:
        #     print('backward diff')
        #     op_tens = mpo.contract(all) * 10 ** mpo.exponent
        #     op_tens.transpose(*[mpo.upper_ind_id.format(i) for i in range(mpo.L)],
        #                       *[mpo.lower_ind_id.format(i) for i in range(mpo.L)],
        #                       inplace=True)
        #     op_mat = op_tens.data.reshape(2 ** mpo.L, 2 ** mpo.L)
        #     print('op mat', left_bc, right_bc, bc_offset, bc_offset_r)
        #     print('L', op_mat[:5, :5])
        #     print('R', op_mat[-5:, -5:])

        return mpo

    def _mpo_secondderivative_center(self, L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                     bc_offset=0, bc_offset_r=None, bc_value_l=0.0, bc_value_r=0.0,
                                     compress_opts=None, eeo_grid=False):
        """ for dim1=dim2=dim=0, since this is 1D
            S+|x> = |x+1> , S-|x> = |x-1>
            d2f/dx2 = \sum_i 1/2*(S- + S+ - 2I)|x_i>
            dim specifies which dimension in which shift is taken, indexed starting at 0
            scale by 1/dt later
            FD coeffs found using python package FinDiff
        """
        if q != 2:
            raise NotImplementedError('check q=2 implementation for binary mapping, esp if not pbc')

        # left_bc = BCType.ANTISYMMETRIC
        # right_bc = BCType.ANTISYMMETRIC
        # order = 3
        # bc_offset = 2
        # bc_offset_r = -2
        print('second derivative mpo', left_bc, order, bc_offset, bc_offset_r)

        if order == 1 or order == 0:
            c0, c1, c2, c3, c4 = (-2., 1., 0., 0., 0.)
            if eeo_grid:
                c0, c1, c2, c3, c4 = (-0.5, 0., 0.25, 0., 0.)
                order = 2   # for building mps later on in the code
        elif order == 2:
            c0, c1, c2, c3, c4 = (-5. / 2, 4. / 3, -1. / 12, 0, 0)
        elif order == 3:
            c0, c1, c2, c3, c4 = (-49. / 18, 3. / 2, -3 / 20, 1 / 90, 0)
        elif order == 4:
            c0, c1, c2, c3, c4 = (-205. / 72, 8. / 5, -1 / 5, 8 / 315, -1 / 560)
        else:
            return self._mpo_higher_order_derivative(L, q, 2, left_bc, right_bc, order=order,
                                                     compress_opts=compress_opts)

        # print('coeffs', c0, c1, c2, c3, c4)
        coeffs = np.array([c0, c1, c2, c3, c4])

        #### build n-ary +/- operator ####
        ## define operator such that operating on the desired spatial dimension
        sm = np.diag([1, ] * (q - 1), k=1)
        sp = np.diag([1, ] * (q - 1), k=-1)

        ## build MPO
        iden = np.eye(q)
        zero = np.zeros((q, q))

        if left_bc == BCType.PERIODIC:  # 0000 <--> 1111
            mpo_tens = [np.array([iden, sp + sm, sm + sp])]  ## ignore dummy size 1 bond
            for i in range(1, L - 2):
                mpo_tens += [np.array([[iden, sp, sm],
                                       [zero, sm, zero],
                                       [zero, zero, sp]])]
            mpo_tens += [np.array([[iden, sp, sm, zero, zero],
                                   [zero, sm, zero, iden, zero],
                                   [zero, zero, sp, zero, iden]])]
            mpo_tens += [
                np.array([c1 * (sm + sp) + c0 * iden, (c1 * sm + c2 * iden + c3 * sp), (c1 * sp + c2 * iden + c3 * sm),
                          (c4 * iden + c3 * sm), (c4 * iden + c3 * sp)])]
            ## ignore dummy size 1 bond
            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')

        elif left_bc == BCType.ANTIPERIODIC:
            mpo_tens = [np.array([iden, sp - sm, sm - sp])]  ## ignore dummy size 1 bond
            for i in range(1, L - 2):
                mpo_tens += [np.array([[iden, sp, sm],
                                       [zero, sm, zero],
                                       [zero, zero, sp]])]
            mpo_tens += [np.array([[iden, sp, sm, zero, zero],
                                   [zero, sm, zero, iden, zero],
                                   [zero, zero, sp, zero, iden]])]
            mpo_tens += [
                np.array([c1 * (sm + sp) + c0 * iden, (c1 * sm + c2 * iden + c3 * sp), (c1 * sp + c2 * iden + c3 * sm),
                          (c4 * iden + c3 * sm), (c4 * iden + c3 * sp)])]
            ## ignore dummy size 1 bond
            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')

        else:  # forward/backward diff, not center diff for 000, 111
            m0 = np.diag([1.] + [0.] * (q - 1))
            m1 = np.diag([0.] * (q - 1) + [1.])

            nb = 4  # int(np.ceil(np.log2(order*2+1)))   ## number of bits to encode boundary condition
            bc0 = np.zeros((q ** nb, q ** nb))
            bc1 = np.zeros((q ** nb, q ** nb))

            num_stencil = 2 * order + 2

            def get_bulk_boundary_stencil():
                """ function that returns bulk coefficients at the boundaries
                """
                ## edge boundary conditions: use the same number of points in stencil
                if order == 1:  # 3-pt stencil
                    fd_mat_0 = np.array([[c0, c1, c2, c3]])

                elif order == 2:  # 5-pt stencil
                    ## 000.. points
                    v0 = np.array([c0, c1, c2, c3, c4, 0.])
                    v1 = np.array([c1, c0, c1, c2, c3, c4])
                    fd_mat_0 = np.array([v0, v1])

                elif order == 3:  # 7-pt stencil
                    v0 = np.array([c0, c1, c2, c3, c4, 0., 0., 0.])
                    v1 = np.array([c1, c0, c1, c2, c3, c4, 0., 0.])
                    v2 = np.array([c2, c1, c0, c1, c2, c3, c4, 0.])
                    fd_mat_0 = np.array([v0, v1, v2])

                elif order == 4:  # order == 4
                    v0 = np.array([c0, c1, c2, c3, c4, 0., 0., 0., 0., 0.])
                    v1 = np.array([c1, c0, c1, c2, c3, c4, 0., 0., 0., 0.])
                    v2 = np.array([c2, c1, c0, c1, c2, c3, c4, 0., 0., 0.])
                    v3 = np.array([c3, c2, c1, c0, c1, c2, c3, c4, 0., 0.])
                    fd_mat_0 = np.array([v0, v1, v2, v3])

                else:
                    raise ValueError(f'order must be 1-4, not {order}')

                fd_mat_L = fd_mat_0[::-1, ::-1]

                return fd_mat_0, fd_mat_L

            def get_gen_boundary_stencil(num_stencil=num_stencil, offset=0):
                """ function that returns open boundary condition stencils in matrix form
                """
                coeffs = []
                for x0 in range(offset, offset + order):
                    stencil = fd_coeff.coefficients(2, offsets=list(range(-x0, num_stencil - x0)))
                    # print('stencil', stencil)
                    coeffs += [stencil['coefficients']]

                end_bc_mat_0 = np.array(coeffs)
                end_bc_mat_L = end_bc_mat_0[::-1, ::-1]
                # print('gen bc', end_bc_mat_0)
                return end_bc_mat_0, end_bc_mat_L

            left_bulk, right_bulk = get_bulk_boundary_stencil()
            # print('left bulk', left_bulk)
            # print('right bulk', right_bulk)

            ## left-hand side boundaries
            if left_bc == BCType.SYMMETRIC or left_bc == BCType.ANTISYMMETRIC \
                    or left_bc == BCType.ZEROGRADIENT or left_bc == BCType.ZEROVALUE \
                    or left_bc == BCType.DIRICHLET or left_bc == BCType.NEUMANN:
                ## add np.array([[c1, c2, c3, c4], [c2, c3, c4, 0.], [c3, c4, 0., 0.], [c4, 0., 0., 0.]])
                ## to bulk matrix

                ### bc_offset == 0 if zero is included at the boundary
                ### bc_offset > 0 means y=0 not included (max = 1; half-step offset)
                ### bc_offset < 0 means y=0 is included.

                # acc = order * 2
                num_boundary = order - min(0, bc_offset) + 1
                end_bc_mat_0 = np.zeros((num_boundary, order - min(0, bc_offset) + 1))
                correction_0 = np.zeros((num_boundary,))

                for i in range(num_boundary):
                    nc = order - i

                    # if nc < 1:
                    #     continue

                    if bc_offset == -1:
                        # reflected from symmetry axis
                        end_bc_mat_0[i, 2:nc + 2] += coeffs[i + 1:order + 1] * np.sign(left_bc.value)
                    elif bc_offset == 0:
                        # reflected from symmetry axis
                        end_bc_mat_0[i, 1:nc + 1] += coeffs[1 + i:order + 1] * np.sign(left_bc.value)
                    elif bc_offset == 1:
                        # reflected from symmetry axis
                        end_bc_mat_0[i, 0:nc] += coeffs[i + 1:order + 1] * np.sign(left_bc.value)
                    elif bc_offset == 2:
                        end_bc_mat_0[i, 0:nc] += coeffs[i + 2:order + 2] * np.sign(left_bc.value)

                        ## corrections
                        if left_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## if Dirichlet with non-zero value (B),
                            ## need to add correction term of B for endpoint and 2B for remaining points
                            if bc_value_l != 0.:
                                weights = coeffs[i + 1] + 2 * np.sum(coeffs[i + 2:order + 1])
                                correction_0[i] = bc_value_l * weights
                                raise NotImplementedError

                        elif left_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
                            ## assuming derivative = 0 at x_(0)
                            ## order 1: B = (4 x_(1), -x_(2) ) / 3
                            ## order 2: B = (15 x_(1), -6 x_(2), x_(3) ) / 10
                            ## order 3: B = (56 x_(1), -28 x_(2), 8 x_(3), -x_(4) ) / 35
                            ## order 4: B = (210 x_(1), -120 x_(2), 45 x_(3), -10 x_(4), x_(5) ) / 126
                            weight = coeffs[i + 1]
                            if order <= 1:
                                end_bc_mat_0[i, 0] += 4 / 3 * weight
                                end_bc_mat_0[i, 1] += -1 / 3 * weight
                            elif order == 2:
                                end_bc_mat_0[i, 0] += 15 / 10 * weight
                                end_bc_mat_0[i, 1] += -6 / 10 * weight
                                end_bc_mat_0[i, 2] += 1 / 10 * weight
                            elif order == 3:
                                end_bc_mat_0[i, 0] += 56 / 35 * weight
                                end_bc_mat_0[i, 1] += -28 / 35 * weight
                                end_bc_mat_0[i, 2] += 8 / 35 * weight
                                end_bc_mat_0[i, 3] += -1 / 35 * weight
                            elif order == 4:
                                end_bc_mat_0[i, 0] += 210 / 126 * weight
                                end_bc_mat_0[i, 1] += -120 / 126 * weight
                                end_bc_mat_0[i, 2] += 45 / 126 * weight
                                end_bc_mat_0[i, 3] += -10 / 126 * weight
                                end_bc_mat_0[i, 4] += 1 / 126 * weight
                            else:
                                raise NotImplementedError
                    else:
                        raise NotImplementedError

            elif left_bc == BCType.OPEN:
                end_bc_mat_0 = -left_bulk
                end_bc_mat_0 += get_gen_boundary_stencil(offset=0)[0]

            else:
                raise ValueError(f'{left_bc} not a valid BCType')

            ## right-hand side boundaries
            if right_bc == BCType.SYMMETRIC or right_bc == BCType.ANTISYMMETRIC \
                    or right_bc == BCType.ZEROGRADIENT or right_bc == BCType.ZEROVALUE \
                    or right_bc == BCType.DIRICHLET or right_bc == BCType.NEUMANN:
                ## build matrix out of coefficients, which will be added to bulk matrix

                ### bc_offset == 0 if zero is included at the boundary
                ### bc_offset < 0 means y=0 not included (min = -1; half-step offset)
                ### bc_offset > 0 means y=0 is included.

                offset = max(0, bc_offset_r)
                num_boundary = order + offset + 1

                end_bc_mat_1 = np.zeros((num_boundary, order + offset + 1))
                correction_1 = np.zeros((num_boundary,))
                # bc_ind = bc_offset + 1

                for i in range(num_boundary):
                    nc = order - i

                    # if nc < 1:  # abs(bc_offset_r) - 1:
                    #     continue

                    if bc_offset_r == -1:
                        # reflected from symmetry axis
                        if nc > 0:
                            end_bc_mat_1[-1 - i, -nc:] += coeffs[i + 1:order + 1][::-1] * np.sign(right_bc.value)

                    elif bc_offset_r == 0:
                        # reflected from symmetry axis
                        end_bc_mat_1[-1 - i, - nc - 1:- 1] += \
                            coeffs[i + 1:order + 1][::-1] * np.sign(right_bc.value)

                    elif bc_offset_r == 1:
                        # reflected from symmetry axis
                        end_bc_mat_1[-1 - i, -nc - 2: -2] += \
                            coeffs[i + 1:order + 1][::-1] * np.sign(right_bc.value)

                    elif bc_offset_r == -2:
                        ## antisymmetric: 0 at the excluded endpoint, so it can be ignored.
                        if nc > 1:
                            end_bc_mat_1[-1 - i, -nc + 1:] += coeffs[i + 2:order + 1][::-1] * np.sign(right_bc.value)

                        ## corrections
                        if right_bc in [BCType.DIRICHLET, BCType.ANTISYMMETRIC, BCType.ZEROVALUE]:
                            ## if Dirichlet with non-zero value (B),
                            ## need to add correction term of B for endpoint and 2B for remaining points
                            if bc_value_r != 0.:
                                weights = coeffs[i + 1] + 2 * np.sum(coeffs[i + 2:order + 1])
                                correction_1[i] = -1 * bc_value_r * weights
                                raise NotImplementedError
                        elif right_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
                            ## assuming derivative = 0 at x_(0)
                            ## order 1: B = (4 x_(1), -x_(2) ) / 3
                            ## order 2: B = (15 x_(1), -6 x_(2), x_(3) ) / 10
                            ## order 3: B = (56 x_(1), -28 x_(2), 8 x_(3), -x_(4) ) / 35
                            ## order 4: B = (210 x_(1), -120 x_(2), 45 x_(3), -10 x_(4), x_(5) ) / 126
                            weight = coeffs[i + 1]
                            # print('right bc  -2 SYMMETRIC', weight, order)
                            if order == 1:
                                end_bc_mat_1[-1 - i, -1] += 4 / 3 * weight
                                end_bc_mat_1[-1 - i, -2] += -1 / 3 * weight
                            elif order == 2:
                                end_bc_mat_1[-1 - i, -1] += 15 / 10 * weight
                                end_bc_mat_1[-1 - i, -2] += -6 / 10 * weight
                                end_bc_mat_1[-1 - i, -3] += 1 / 10 * weight
                            elif order == 3:
                                end_bc_mat_1[-1 - i, -1] += 56 / 35 * weight
                                end_bc_mat_1[-1 - i, -2] += -28 / 35 * weight
                                end_bc_mat_1[-1 - i, -3] += 8 / 35 * weight
                                end_bc_mat_1[-1 - i, -4] += -1 / 35 * weight
                            elif order == 4:
                                end_bc_mat_1[-1 - i, -1] += 210 / 126 * weight
                                end_bc_mat_1[-1 - i, -2] += -120 / 126 * weight
                                end_bc_mat_1[-1 - i, -3] += 45 / 126 * weight
                                end_bc_mat_1[-1 - i, -4] += -10 / 126 * weight
                                end_bc_mat_1[-1 - i, -5] += 1 / 126 * weight
                            else:
                                raise NotImplementedError

                    else:
                        raise NotImplementedError

            elif right_bc == BCType.OPEN:
                end_bc_mat_1 = -right_bulk
                end_bc_mat_1 += get_gen_boundary_stencil(offset=0)[1]

            else:
                raise ValueError(f'{right_bc} not a valid BCType')

            # print('end bc mat 0')
            # print(end_bc_mat_0)
            # print('end bc mat 1')
            # print(end_bc_mat_1)
            # exit()

            nx, ny = end_bc_mat_0.shape
            bc0[:nx, :ny] = end_bc_mat_0
            nx, ny = end_bc_mat_1.shape
            bc1[-nx:, -ny:] = end_bc_mat_1

            ## 0 side
            bc0 = np.reshape(bc0, (q,) * nb * 2)
            bc0_tens = qtn.Tensor(bc0, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                  [f'i({i})' for i in range(nb)]))
            bc0_mpo = helper.mpx_from_dense(bc0_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')

            ## L-1 side
            bc1 = np.reshape(bc1, (q,) * nb * 2)
            bc1_tens = qtn.Tensor(bc1, inds=tuple([f'o({i})' for i in range(nb)] + \
                                                  [f'i({i})' for i in range(nb)]))
            bc1_mpo = helper.mpx_from_dense(bc1_tens, nb, ('o({})', 'i({})'), site_tag_id='X({})')
            if bc1_mpo is None:
                bc1_mpo = qtn.MatrixProductOperator([np.zeros((1,q,q))] + [np.zeros((1,1,q,q))] * (nb - 2)
                                                    + [np.zeros((1,q,q))], shape='lrud',
                                                    site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')

            ## leading 00... or 1... strings
            if L - nb >= 2:
                mpo_0 = qtn.MatrixProductOperator(
                    [np.array([m0])] + [np.array([[m0]])] * (L - nb - 2) + [np.array([m0])],
                    shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                mpo_1 = qtn.MatrixProductOperator(
                    [np.array([m1])] + [np.array([[m1]])] * (L - nb - 2) + [np.array([m1])],
                    shape='lrud', site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
            elif L - nb == 1:
                mpo_0 = qtn.TensorNetwork([qtn.Tensor(m0, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
                mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                              site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                mpo_1 = qtn.TensorNetwork([qtn.Tensor(m1, inds=('o(0)', 'i(0)'), tags=('X(0)',))])
                mpo_1.view_as(qtn.MatrixProductOperator, like=mpo_0, inplace=True)
            elif L - nb == 0:
                mpo_0 = qtn.TensorNetwork([])
                mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                              site_tag_id='X({})', upper_ind_id='o({})', lower_ind_id='i({})')
                mpo_1 = qtn.TensorNetwork([])
                mpo_1.view_as(qtn.MatrixProductOperator, like=mpo_0, inplace=True)
            else:
                raise ValueError('FD order too large for current grid size')

            helper.append_mpx(mpo_0, bc0_mpo, inplace=True)
            helper.append_mpx(mpo_1, bc1_mpo, inplace=True)

            ### the bulk mpo
            mpo_tens = [np.array([iden, sp, sm])]  ## ignore dummy size 1 bond
            for i in range(1, L - 2):
                mpo_tens += [np.array([[iden, sp, sm],
                                       [zero, sm, zero],
                                       [zero, zero, sp]])]
            mpo_tens += [np.array([[iden, sp, sm, zero, zero],
                                   [zero, sm, zero, iden, zero],
                                   [zero, zero, sp, zero, iden]])]
            mpo_tens += [
                np.array([c1 * (sm + sp) + c0 * iden, (c1 * sm + c2 * iden + c3 * sp), (c1 * sp + c2 * iden + c3 * sm),
                          (c4 * iden + c3 * sm), (c4 * iden + c3 * sp)])]  ## ignore size 1 bond

            mpo = qtn.MatrixProductOperator(mpo_tens, shape='lrud', site_tag_id='B({})',
                                            upper_ind_id='o({})', lower_ind_id='i({})')
            helper.add_MPO(mpo, mpo_0, inplace=True)
            helper.add_MPO(mpo, mpo_1, inplace=True)

            helper.compress(mpo, compress_opts=compress_opts)

            # data = mpo.to_dense() * 10**mpo.exponent
            # print(data[:5,:5])
            # print(data[-5:, -5:])
            # exit()
        return mpo

    def _mpo_higher_order_derivative(self, L, q, deriv_order, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC,
                                     order=DEFAULT_ORDER, fd_type=FDType.CENTER, compress_opts=None):
        """ derivs:  tuples of the form (axis, step size, derivative degree)
            open boundary conditions
            used for central difference with order > 4
        """
        print('warning: boundary conditions not applied', left_bc)

        import findiff

        if fd_type != FDType.CENTER:
            print(f'WARNING: centered stencil is used, not {fd_type}')

        if order != 1:
            raise NotImplementedError(f'only FD order 1 is implemented, not {order}')

        npts = q ** L
        num_stencil = 2 * (deriv_order // 2) + (2 * order - 1)    # for centered fdiff
        num_side = num_stencil // 2
        coeffs = findiff.coefficients(deriv_order, 2*order)['center']['coefficients']
        print('coeffs', coeffs, num_stencil)

        inds = []
        for ix in range(num_side, npts - num_side):    ## for each row where the entire stencil fits
            inds += [(ix, ix + jx - num_side, coeffs[jx]) for jx in range(num_stencil)]

        if left_bc == BCType.PERIODIC:
            for ix in range(0, num_side):
                inds += [(ix, (ix + jx - num_side) % npts, coeffs[jx]) for jx in range(num_stencil)]
            for ix in range(npts - num_side, npts):
                inds += [(ix, (ix + jx - num_side) % npts, coeffs[jx]) for jx in range(num_stencil)]

        elif left_bc == BCType.ANTIPERIODIC:
            for ix in range(0, num_side):
                inds += [(ix, (ix + jx - num_side) % npts, coeffs[jx] * (-1)**((ix + jx)//npts))
                         for jx in range(num_stencil)]
            for ix in range(npts - num_side, npts):
                inds += [(ix, (ix + jx - num_side) % npts, coeffs[jx] * (-1)**((ix + jx)//npts))
                         for jx in range(num_stencil)]
        else:
            for x0 in range(num_side):
                offsets = list(range(-x0, num_stencil - x0 + 1))
                stencil = findiff.coefficients(deriv_order, offsets=offsets)
                # print('x0', x0)
                # print('offsets', offsets)
                # print('stencil', stencil['coefficients'], stencil['accuracy'])
                inds += [(x0, jx, stencil['coefficients'][jx]) for jx in range(len(offsets))]
            for x0 in range(npts - num_side, npts):
                offsets = list(range(npts - x0 - 1 - num_stencil, npts - x0))
                stencil = findiff.coefficients(deriv_order, offsets=offsets)
                # print('x0', x0)
                # print('offsets', offsets)
                # print('stencil', stencil['coefficients'], stencil['accuracy'])
                inds += [(x0, x0 + offsets[jx], stencil['coefficients'][jx]) for jx in range(len(offsets))]

        coo_data = np.array(inds)
        idxs = (coo_data[:, 0], coo_data[:, 1])
        dats = (coo_data[:, 2])
        derivative = scipy.sparse.coo_matrix((dats, idxs), shape=(npts,npts))
        op = derivative.toarray()

        # plt.figure()
        # plt.imshow(op)
        # plt.title('deriv op')
        # plt.colorbar()
        # plt.show()

        # print('deriv order', deriv_order, order)
        # print('derivative', derivative.toarray())
        # print('derivative', derivative.toarray()[:10,:10])
        # pdb.set_trace()

        op = op.reshape((q,)*(2*L))
        ## guess:  initial shape is (o10, ..., oL0) (o11,..., oL1) x (i10, ...,iL0) (i11, ..., iL1)
        ##         where i(a,b) corresponds to ath input leg along bth dimension
        op_tensor = qtn.Tensor(op, inds=[f'o({i})' for i in range(L)] + \
                                        [f'i({i})' for i in range(L)])

        mpo = helper.mpx_from_dense(op_tensor, L, ('o({})', 'i({})'), site_tag_id='B{}',
                                    split_opts=compress_opts)

        return mpo

    def _matrix_firstderivative_center(self, L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                       bc_offset=0, bc_offset_r=None, bc_value_l=0.0, bc_value_r=0.0,
                                       compress_opts=None):
        """ S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>a
            is for a 1D system so dim does not need to be specified
            scale by 1/dt later
            boundary_condition:  boundary condition to use when taking derivative

            FD coeffs found using python package FinDiff
            default: bc_offset_r = bc_offset if bc_offset_r is None
        """
        if bc_value_l != 0.0 or bc_value_r != 0:
            raise NotImplementedError

        if L != 1:
            raise NotImplementedError('matrix form of first deriv requires L=1')

        print('center FD matrix')

        if order == 1:
            c1, c2, c3, c4 = (1. / 2, 0, 0, 0)
            coeffs = [0, c1]
        elif order == 2:
            c1, c2, c3, c4 = (2. / 3, -1. / 12, 0, 0)
            coeffs = [0, c1, c2]
        elif order == 3:
            c1, c2, c3, c4 = (3. / 4, -3. / 20, 1. / 60, 0)
            coeffs = [0, c1, c2, c3]
        elif order == 4:
            c1, c2, c3, c4 = (4. / 5, -1. / 5, 4. / 105, -1. / 280)
            coeffs = [0, c1, c2, c3, c4]
        else:
            return self._mpo_higher_order_derivative(L, q, 1, left_bc, right_bc, order=order,
                                                     compress_opts=compress_opts)
        nc = order + 1

        data = []
        rows = []
        cols = []

        bc_offset_r = bc_offset if bc_offset_r is None else bc_offset_r
        bc_offset_l = 0 if np.abs(right_bc.value) == 1 else bc_offset  # 0 if PBC/APBC
        bc_offset_r = 0 if np.abs(right_bc.value) == 1 else bc_offset_r  # 0 if PBC/APBC
        bc_offset_l = -bc_offset_l
        bc_offset_r = -bc_offset_r
        left_num_bc_rows = nc - 1 + max(0, bc_offset_l)  # rows that need special treatment for the boundaries
        right_num_bc_rows = nc - 1 - min(0, bc_offset_r)  # rows that need special treatment for the boundaries

        for i in range(left_num_bc_rows, q - right_num_bc_rows):
            data += coeffs[1:] + [c * -1 for c in coeffs[1:]]
            cols += [(i + m) for m in range(1, nc)] + [(i - m) for m in range(1, nc)]
            rows += [i] * (nc - 1) * 2
            # print('data', coeffs[1:] + [c*-1 for c in coeffs[1:]])
            # print('cols', i, [(i + m) for m in range(1, nc)] + [(i - m) for m in range(1, nc)])
            # print('rows', [i] * (nc-1)*2)

        ## right BCs
        for i in range(right_num_bc_rows):  # rows that need special treatment

            if right_bc == BCType.PERIODIC or right_bc == BCType.ANTIPERIODIC:
                rows += [q - i - 1] * (nc - 1) * 2

                ## coeffs < i (in bulk)
                cols += [(q - i - m - 1) % q for m in range(1, nc)]
                data += [c * -1 for c in coeffs[1:]]

                ## coeffs > i (hits boundary)
                cols += [(q - i + m - 1) % q for m in range(1, nc)]  # wrap to left
                if left_bc.value < 0:
                    data += [coeffs[m] * np.sign(i - m + 0.1) for m in range(1, nc)]
                else:
                    data += coeffs[1:]

            elif right_bc == BCType.OPEN:
                rows += [q - i - 1] * (nc * 2 - 1)

                open_coeffs = fd_coeff.coefficients(deriv=1, offsets=list(range(i - (2 * nc - 1) + 1, i + 1)))
                cols += [m for m in range(q - (2 * nc - 1), q)]
                data += list(open_coeffs['coefficients'])

            elif right_bc == BCType.SYMMETRIC or right_bc == BCType.ANTISYMMETRIC \
                    or right_bc == BCType.REFLECTING or right_bc == BCType.ZEROGRADIENT \
                    or right_bc == BCType.ABSORBING or right_bc == BCType.ZEROVALUE:
                ### weights to be added to bulk stencil
                ### bc_offset = 0 means y=0 is included, at the boundary
                ### bc_offset > 0 means y=0 is not included (max = 1)
                ### bc_offset < 0 means y=0 is included, + bc_offset extra data points
                ###     if bc_offset == 1 --> c[last+1] = c[last]
                ###     if bc_offset == 0 --> c[last+1] = c[last-1]
                ###     if bc_offset ==-1 --> c[last+1] = c[last-2]

                rows += [q - i - 1] * (nc - 1) * 2

                ## coeffs < i (in bulk)
                cols += [(q - i - m - 1) % q for m in range(1, nc)]
                data += [c * -1 for c in coeffs[1:]]

                ## coeffs > i (at boundary)
                data_i, cols_i = [], []
                for j in range(1, nc):
                    if - i + j > min(0, bc_offset_r):  # out of bounds
                        cols_i += [q - j + i - 1 + bc_offset_r]
                        data_i += [coeffs[j] * np.sign(right_bc.value)]
                    else:
                        cols_i += [q + j - i - 1]
                        data_i += [coeffs[j]]

                data += data_i
                cols += cols_i

        ## left BCs
        for i in range(left_num_bc_rows):  # rows that need special treatment

            if left_bc == BCType.PERIODIC or left_bc == BCType.ANTIPERIODIC:
                rows += [i] * (nc - 1) * 2

                ## coeffs > i (in bulk)
                cols += [(i + m) % q for m in range(1, nc)]
                data += coeffs[1:]

                ## coeffs < i (at boundary)
                cols += [(i - m) % q for m in range(1, nc)]
                if left_bc.value < 0:
                    data += [coeffs[m] * -1 * np.sign(i - m + 0.1) for m in range(1, nc)]
                else:
                    data += [c * -1 for c in coeffs[1:]]

            elif left_bc == BCType.OPEN:
                rows += [i] * (2 * nc - 1)

                open_coeffs = fd_coeff.coefficients(deriv=1, offsets=list(range(-i, -i + (2 * nc - 1))))
                cols += [m for m in range(2 * nc - 1)]
                data += list(open_coeffs['coefficients'])

            elif left_bc == BCType.SYMMETRIC or left_bc == BCType.ANTISYMMETRIC \
                    or left_bc == BCType.REFLECTING or left_bc == BCType.ZEROGRADIENT \
                    or left_bc == BCType.ABSORBING or left_bc == BCType.ZEROVALUE:
                ### weights to be added to bulk stencil
                ### bc_offset = 0 means y=0 is included, at the boundary
                ### bc_offset > 0 means y=0 is not included (max = 1)
                ### bc_offset < 0 means y=0 is included, + bc_offset extra data points
                ###     if bc_offset == 1 --> c[first-1] = c[first+2]
                ###     if bc_offset == 0 --> c[first-1] = c[first+1]
                ###     if bc_offset ==-1 --> c[first-1] = c[first]

                rows += [i] * (nc - 1) * 2

                ## coeffs > i (in bulk)
                cols += [(i + m) % q for m in range(1, nc)]
                data += coeffs[1:]

                cols_i, data_i = [], []
                for j in range(1, nc):
                    if i - j < max(0, bc_offset_l):  # out of bounds
                        cols_i += [j - i + bc_offset_l]
                        data_i += [coeffs[j] * -1 * np.sign(left_bc.value)]
                    else:
                        cols_i += [i - j]
                        data_i += [coeffs[j] * -1]

                cols += cols_i
                data += data_i

        deriv_mat = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(q, q))
        # deriv_mat = deriv_mat.tocsr()
        # deriv_mat = deriv_mat.toarray()
        # print(deriv_mat.toarray())
        deriv_mat_tens = qtn.Tensor(deriv_mat, inds=('o(0)', 'i(0)'), tags=('B(0)',))
        deriv_mat_mpo = qtn.TensorNetwork([deriv_mat_tens])
        deriv_mat_mpo = deriv_mat_mpo.view_as(qtn.MatrixProductOperator, L=1, cyclic=False, inplace=True,
                                              upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='B({})')

        return deriv_mat_mpo

    @classmethod
    def _matrix_firstderivative_forward(self, L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                        bc_offset=0, bc_offset_r=None, bc_value_l=0.0, bc_value_r=0.0,
                                        compress_opts=None):
        """ S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>
            is for a 1D system so dim does not need to be specified
            scale by 1/dt later
            forward diff or backward diff
            boundary_condition:  boundary condition to use when taking derivative

            FD coeffs found using python package FinDiff
        """
        if bc_value_l != 0.0 or bc_value_r != 0:
            raise NotImplementedError

        if L != 1:
            raise NotImplementedError('matrix form of first deriv requires L=1')

        print('forward FD mpo')

        acc = 2 * order if order > 0 else 1
        coeffs_fd = fd_coeff.coefficients(deriv=1, acc=acc)
        coeffs = list(coeffs_fd['forward']['coefficients'])
        nc = len(coeffs)

        data = []
        rows = []
        cols = []

        bc_offset_r = bc_offset if bc_offset_r is None else bc_offset_r
        bc_offset_r = 0 if np.abs(right_bc.value) == 1 else bc_offset_r  # 0 if PBC/APBC
        bc_offset_r = -bc_offset_r

        right_num_bc_rows = nc - 1 - min(0, bc_offset_r)  # rows that need special treatment for the boundaries

        for i in range(q - right_num_bc_rows):
            data += coeffs
            cols += [(i + m) for m in range(nc)]
            rows += [i] * nc
            # print('data', coeffs[1:] + [c*-1 for c in coeffs[1:]])
            # print('cols', i, [(i + m) for m in range(1, nc)] + [(i - m) for m in range(1, nc)])
            # print('rows', [i] * (nc-1)*2)

        ## right BCs
        for i in range(right_num_bc_rows):  # rows that need special treatment

            if right_bc == BCType.PERIODIC or right_bc == BCType.ANTIPERIODIC:
                rows += [q - i - 1] * nc

                ## coeffs > i (hits boundary)
                cols += [(q - i + m - 1) % q for m in range(nc)]  # wrap to left
                if left_bc.value < 0:
                    data += [coeffs[m] * np.sign(i - m + 0.1) for m in range(nc)]
                else:
                    data += coeffs

            elif right_bc == BCType.OPEN:
                rows += [q - i - 1] * nc

                open_coeffs = fd_coeff.coefficients(deriv=1, offsets=list(range(i - nc + 1, i + 1)))
                cols += [m for m in range(q - nc, q)]
                data += list(open_coeffs['coefficients'])

            elif right_bc == BCType.SYMMETRIC or right_bc == BCType.ANTISYMMETRIC \
                    or right_bc == BCType.REFLECTING or right_bc == BCType.ZEROGRADIENT \
                    or right_bc == BCType.ABSORBING or right_bc == BCType.ZEROVALUE:
                ### weights to be added to bulk stencil
                ### bc_offset = 0 means y=0 is included, at the boundary
                ### bc_offset > 0 means y=0 is included (max = 1)
                ### bc_offset < 0 means y=0 is not included, + bc_offset extra data points
                ###     if bc_offset == 1 --> c[last+1] = c[last]
                ###     if bc_offset == 0 --> c[last+1] = c[last-2]
                ###     if bc_offset ==-1 --> c[last+1] = c[last-1]
                ###     if bc_offset ==-2 --> c[last+1] = boundary value

                rows += [q - i - 1] * nc

                ### coeffs before the boundary
                data_i, cols_i = [], []
                for j in range(min(i + 1, nc)):
                    # print('j', j, coeffs)
                    cols_i += [q - i - 1 + j]
                    data_i += [coeffs[j]]

                for j in range(i + 1, nc):
                    if bc_offset_r == -2 and i + 1 == 1:
                        rows.pop(-1)
                        continue
                    cols_i += [q - j - 1 + bc_offset_r]
                    data_i += [coeffs[j] * np.sign(right_bc.value)]

                data += data_i
                cols += cols_i

        deriv_mat = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(q, q))
        # deriv_mat = deriv_mat.tocsr()
        # deriv_mat = deriv_mat.toarray()
        # print(deriv_mat.toarray()[-10:,-10:])
        deriv_mat_tens = qtn.Tensor(deriv_mat, inds=('o(0)', 'i(0)'), tags=('B(0)',))
        deriv_mat_mpo = qtn.TensorNetwork([deriv_mat_tens])
        deriv_mat_mpo = deriv_mat_mpo.view_as(qtn.MatrixProductOperator, L=1, cyclic=False, inplace=True,
                                              upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='B({})')

        return deriv_mat_mpo

    def _matrix_firstderivative_backward(self, L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                         bc_offset=0, bc_offset_r=None, bc_value_l=0.0, bc_value_r=0.0, compress_opts=None):
        """ S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>
            is for a 1D system so dim does not need to be specified
            scale by 1/dt later
            forward diff or backward diff
            boundary_condition:  boundary condition to use when taking derivative

            FD coeffs found using python package FinDiff
        """
        if bc_value_l != 0.0 or bc_value_r != 0:
            raise NotImplementedError

        if L != 1:
            raise NotImplementedError('matrix form of first deriv requires L=1')

        print('backward FD mpo')

        acc = 2 * order if order > 0 else 1
        coeffs_fd = fd_coeff.coefficients(deriv=1, acc=acc)
        coeffs = list(coeffs_fd['backward']['coefficients'][::-1])
        nc = len(coeffs)

        data = []
        rows = []
        cols = []

        bc_offset_l = 0 if np.abs(left_bc.value) == 1 else bc_offset  # 0 if PBC/APBC
        bc_offset_l = -bc_offset_l
        left_num_bc_rows = nc - 1 + max(0, bc_offset_l)  # rows that need special treatment for the boundaries

        for i in range(left_num_bc_rows, q):
            data += coeffs
            cols += [(i - m) for m in range(nc)]
            rows += [i] * nc
            # print('data', coeffs[1:] + [c*-1 for c in coeffs[1:]])
            # print('cols', i, [(i + m) for m in range(1, nc)] + [(i - m) for m in range(1, nc)])
            # print('rows', [i] * (nc-1)*2)

        ## left BCs
        for i in range(left_num_bc_rows):  # rows that need special treatment

            if left_bc == BCType.PERIODIC or left_bc == BCType.ANTIPERIODIC:
                rows += [i] * nc

                ## coeffs < i (at boundary)
                cols += [(i - m) % q for m in range(nc)]
                if left_bc.value < 0:
                    data += [coeffs[m] * np.sign(i - m + 0.1) for m in range(nc)]
                else:
                    data += [c for c in coeffs]

            elif left_bc == BCType.OPEN:
                rows += [i] * nc

                open_coeffs = fd_coeff.coefficients(deriv=1, offsets=list(range(-i, -i + nc)))
                cols += [m for m in range(nc)]
                data += list(open_coeffs['coefficients'])

            elif left_bc == BCType.SYMMETRIC or left_bc == BCType.ANTISYMMETRIC \
                    or left_bc == BCType.REFLECTING or left_bc == BCType.ZEROGRADIENT \
                    or left_bc == BCType.ABSORBING or left_bc == BCType.ZEROVALUE:
                ### weights to be added to bulk stencil
                ### bc_offset = 0 means y=0 is included, at the boundary
                ### bc_offset > 0 means y=0 is not included (max = 1)
                ### bc_offset < 0 means y=0 is included, + bc_offset extra data points
                ###     if bc_offset == 1 --> c[first-1] = c[first+2]
                ###     if bc_offset == 0 --> c[first-1] = c[first+1]
                ###     if bc_offset ==-1 --> c[first-1] = c[first]

                rows += [i] * nc

                cols_i, data_i = [], []
                for j in range(nc):
                    if i - j < max(0, bc_offset_l):  # out of bounds
                        cols_i += [j - i + bc_offset_l]
                        data_i += [coeffs[j] * np.sign(left_bc.value)]
                    else:
                        cols_i += [i - j]
                        data_i += [coeffs[j]]

                cols += cols_i
                data += data_i

        deriv_mat = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(q, q))
        # deriv_mat = deriv_mat.tocsr()
        # deriv_mat = deriv_mat.toarray()
        # print(deriv_mat.toarray()[:10,:10])
        deriv_mat_tens = qtn.Tensor(deriv_mat, inds=('o(0)', 'i(0)'), tags=('B(0)',))
        deriv_mat_mpo = qtn.TensorNetwork([deriv_mat_tens])
        deriv_mat_mpo = deriv_mat_mpo.view_as(qtn.MatrixProductOperator, L=1, cyclic=False, inplace=True,
                                              upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='B({})')

        return deriv_mat_mpo

    def _matrix_secondderivative_center(self, L, q, left_bc=DEFAULT_BC, right_bc=DEFAULT_BC, order=DEFAULT_ORDER,
                                        bc_offset=0, bc_offset_r=None, bc_value_l=0.0, bc_value_r=0.0,
                                        compress_opts=None, eeo_grid=False):
        """ for dim1=dim2=dim=0, since this is 1D
            S+|x> = |x+1> , S-|x> = |x-1>
            d2f/dx2 = \sum_i 1/2*(S- + S+ - 2I)|x_i>
            dim specifies which dimension in which shift is taken, indexed starting at 0
            scale by 1/dt later
            FD coeffs found using python package FinDiff
        """
        if eeo_grid:
            raise NotImplementedError

        if bc_value_l != 0.0 or bc_value_r != 0:
            raise NotImplementedError

        if L != 1:
            raise NotImplementedError('matrix form of second deriv requires L=1')

        if order == 1:
            c0, c1, c2, c3, c4 = [-2., 1., 0., 0., 0.]
            coeffs = [c0, c1]
        elif order == 2:
            c0, c1, c2, c3, c4 = [-5. / 2, 4. / 3, -1. / 12, 0, 0]
            coeffs = [c0, c1, c2]
        elif order == 3:
            c0, c1, c2, c3, c4 = [-49. / 18, 3. / 2, -3 / 20, 1 / 90, 0]
            coeffs = [c0, c1, c2, c3]
        elif order == 4:
            c0, c1, c2, c3, c4 = [-205. / 72, 8. / 5, -1 / 5, 8 / 315, -1 / 560]
            coeffs = [c0, c1, c2, c3, c4]
        else:
            return self._matrix_higher_order_derivative(L, q, 2, left_bc, right_bc, order=order,
                                                        compress_opts=compress_opts)
        # print('coeffs', c0, c1, c2, c3, c4)

        nc = order + 1

        data = []
        rows = []
        cols = []

        bc_offset_r = bc_offset if bc_offset_r is None else bc_offset_r
        bc_offset_l = 0 if np.abs(right_bc.value) == 1 else bc_offset  # 0 if PBC/APBC
        bc_offset_r = 0 if np.abs(right_bc.value) == 1 else bc_offset_r  # 0 if PBC/APBC
        left_num_bc_rows = nc - 1 + max(0, bc_offset_l)  # rows that need special treatment for the boundaries
        right_num_bc_rows = nc - 1 - min(0, bc_offset_r)  # rows that need special treatment for the boundaries

        for i in range(left_num_bc_rows, q - right_num_bc_rows):
            data += coeffs + [c for c in coeffs[1:]]
            cols += [(i + m) for m in range(nc)] + [(i - m) for m in range(1, nc)]
            rows += [i] * (nc * 2 - 1)
            # print('data', coeffs[1:] + [c*-1 for c in coeffs[1:]])
            # print('cols', i, [(i + m) for m in range(1, nc)] + [(i - m) for m in range(1, nc)])
            # print('rows', [i] * (nc-1)*2)

        ## right BCs
        for i in range(right_num_bc_rows):  # rows that need special treatment

            if right_bc == BCType.PERIODIC or right_bc == BCType.ANTIPERIODIC:
                rows += [q - i - 1] * (nc * 2 - 1)

                ## coeffs < i (in bulk)
                cols += [(q - i - m - 1) % q for m in range(nc)]
                data += [c for c in coeffs]

                ## coeffs > i (hits boundary)
                cols += [(q - i + m - 1) % q for m in range(1, nc)]  # wrap to left
                if left_bc.value < 0:
                    data += [coeffs[m] * np.sign(i - m + 0.1) for m in range(1, nc)]
                else:
                    data += coeffs[1:]

            elif right_bc == BCType.OPEN:
                rows += [q - i - 1] * (nc * 2 - 1)

                open_coeffs = fd_coeff.coefficients(deriv=2, offsets=list(range(i - (2 * nc - 1) + 1, i + 1)))
                cols += [m for m in range(q - (2 * nc - 1), q)]
                data += list(open_coeffs['coefficients'])

            elif right_bc == BCType.SYMMETRIC or right_bc == BCType.ANTISYMMETRIC \
                    or right_bc == BCType.REFLECTING or right_bc == BCType.ZEROGRADIENT \
                    or right_bc == BCType.ABSORBING or right_bc == BCType.ZEROVALUE:
                ### weights to be added to bulk stencil
                ### bc_offset = 0 means y=0 is included, at the boundary
                ### bc_offset > 0 means y=0 is not included (max = 1)
                ### bc_offset < 0 means y=0 is included, + bc_offset extra data points
                ###     if bc_offset == 1 --> c[last+1] = c[last]
                ###     if bc_offset == 0 --> c[last+1] = c[last-1]
                ###     if bc_offset ==-1 --> c[last+1] = c[last-2]

                rows += [q - i - 1] * (nc * 2 - 1)

                ## coeffs < i (in bulk)
                cols += [(q - i - m - 1) % q for m in range(nc)]
                data += [c for c in coeffs]

                # print('right cols', q - i - 1, [(q - i - m - 1) % q for m in range(nc)])

                ## coeffs > i (at boundary)
                data_i, cols_i = [], []
                for j in range(1, nc):
                    if - i + j > min(0, bc_offset_r):  # out of bounds
                        cols_i += [q - j + i - 1 + bc_offset_r]
                        data_i += [coeffs[j] * np.sign(right_bc.value)]
                    else:
                        cols_i += [q + j - i - 1]
                        data_i += [coeffs[j]]

                # print('right cols', q - i - 1, cols_i)
                # print('right data', q - i - 1, data_i)

                data += data_i
                cols += cols_i

        ## left BCs
        for i in range(left_num_bc_rows):  # rows that need special treatment

            if left_bc == BCType.PERIODIC or left_bc == BCType.ANTIPERIODIC:
                rows += [i] * (nc * 2 - 1)

                ## coeffs > i (in bulk)
                cols += [(i + m) % q for m in range(nc)]
                data += coeffs

                ## coeffs < i (at boundary)
                cols += [(i - m) % q for m in range(1, nc)]
                if left_bc.value < 0:
                    data += [coeffs[m] * np.sign(i - m + 0.1) for m in range(1, nc)]
                else:
                    data += [c for c in coeffs[1:]]

            elif left_bc == BCType.OPEN:
                rows += [i] * (2 * nc - 1)

                open_coeffs = fd_coeff.coefficients(deriv=2, offsets=list(range(-i, -i + (2 * nc - 1))))
                cols += [m for m in range(2 * nc - 1)]
                data += list(open_coeffs['coefficients'])

            elif left_bc == BCType.SYMMETRIC or left_bc == BCType.ANTISYMMETRIC \
                    or left_bc == BCType.REFLECTING or left_bc == BCType.ZEROGRADIENT \
                    or left_bc == BCType.ABSORBING or left_bc == BCType.ZEROVALUE:
                ### weights to be added to bulk stencil
                ### bc_offset = 0 means y=0 is included, at the boundary
                ### bc_offset > 0 means y=0 is not included (max = 1)
                ### bc_offset < 0 means y=0 is included, + bc_offset extra data points
                ###     if bc_offset == 1 --> c[first-1] = c[first+2]
                ###     if bc_offset == 0 --> c[first-1] = c[first+1]
                ###     if bc_offset ==-1 --> c[first-1] = c[first]

                rows += [i] * (nc * 2 - 1)

                ## coeffs > i (in bulk)
                cols += [(i + m) % q for m in range(nc)]
                data += coeffs
                # print('left cols', [(i + m) % q for m in range(nc)])

                cols_i, data_i = [], []
                for j in range(1, nc):
                    if i - j < max(0, bc_offset_l):  # out of bounds
                        cols_i += [j - i + bc_offset_l]
                        data_i += [coeffs[j] * np.sign(left_bc.value)]
                    else:
                        cols_i += [i - j]
                        data_i += [coeffs[j]]

                # print('left cols', i, cols_i)
                # print('left data', i, data_i)

                cols += cols_i
                data += data_i

        deriv_mat = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(q, q))
        # print('second deriv centered', deriv_mat.toarray()[:10, :10])
        # print('second deriv centered', deriv_mat.toarray()[-10:, -10:])
        # deriv_mat = deriv_mat.tocsr()
        deriv_mat = deriv_mat.toarray()
        deriv_mat_tens = qtn.Tensor(deriv_mat, inds=('o(0)', 'i(0)'), tags=('B(0)',))
        deriv_mat_mpo = qtn.TensorNetwork([deriv_mat_tens])
        deriv_mat_mpo = deriv_mat_mpo.view_as(qtn.MatrixProductOperator, L=1, cyclic=False, inplace=True,
                                              upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='B({})')

        return deriv_mat_mpo
