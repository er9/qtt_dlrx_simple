"""Fourier (k-space) basis: a :class:`Basis` representing fields by their Fourier
modes, where derivatives become diagonal multipliers and products become
convolutions."""
from setup_.configs import *
import numpy as np
import quimb.tensor as qtn
import helper_quimb as helper
# from axis import Axis
# import axis_map
from basis.basis import Basis

if TYPE_CHECKING:
    from axis import Axis
    from layout.layout import Layout


class FourierBasis(Basis):

    def __init__(self):
        self.type = BasisType.FOURIER

    def from_realspace_1D(self, func, num_modes: int):
        raise NotImplementedError

    def get_realspace_1D(self, data, ax_ind, **kwargs):
        """ coeffs:  coefficients of Hermite polynomials (-K, -K+1, ..., 0, ..., K-1)
            x_window:  x0,xL specifying domain of interest
            x_offset:  offset of center of Hermite polynomials
            x_scale:   scaling of x in Hermite polynomials     (x <- x_scale*x + x_offset)
            note: numpy fct is physicists Hermite polynomials (no normalization)
        """
        coeffs_ = np.fft.ifftshift(data, axes=ax_ind)  # moves 0 frequency term (at npts//2) to 0
        out_data = np.fft.ifft(coeffs_, axis=ax_ind) * data.shape[ax_ind]
        # out_data = np.fft.ifftshift(out_data, axes=ax_ind)

        # data = np.moveaxis(data, ax_ind, -1)
        # coeffs_  = np.fft.fftshift(data, axes=-1)   # moves 0 frequency term (at npts//2) to 0
        # out_data = np.fft.ifftn(coeffs_, axes=[-1]) * data.shape[-1]
        # out_data = np.fft.ifftshift(out_data, axes=-1)
        # out_data = np.moveaxis(out_data, -1, ax_ind)
        return out_data

    def get_realspace_nD(self, data: 'np.ndarray', ax_inds: list[int], **kwargs):
        """ ie. coeffs of delta fcts """
        data_size = np.prod([data.shape[x] for x in ax_inds])
        coeffs_ = np.fft.fftshift(data, axes=ax_inds)  # moves 0 frequency term (at npts//2) to 0
        out_data = np.fft.ifftn(coeffs_, axes=ax_inds) * data_size   ## norm='forward'?
        # out_data = np.fft.ifftshift(out_data, axes=ax_inds)
        return out_data

    # @classmethod
    # def elemental_multiply(cls,tn1,tn2,ax1=None,ax2=None,inplace=False,**compress_opts):
    #     tn1 = tn1 if inplace else tn1.copy()
    #     tn1.convolve(tn2,ax1=ax1,ax2=ax2,**compress_opts)
    #     return tn1


    def get_ones_mps(self, ax: 'Axis', site_ind_id='i({})', site_tag_id='X({})', anc_dim=None, anc_name_l=None,
                     anc_name_r=None) -> 'MPSType':
        # if anc_dim is not None and anc_dim != 1:
        #     raise NotImplementedError

        mps_0 = ax.get_select_elems_mps([ax.zero_ind], site_ind_id=site_ind_id, site_tag_id=site_tag_id)
        if anc_dim is None or anc_dim == 0:
            pass
        elif anc_dim == 1:
            if anc_name_l is not None:
                # mps_0[0].reindex({mps_0[0].inds[0]: anc_name_l}, inplace=True)
                mps_0[0].new_ind(anc_name_l, size=anc_dim)
            if anc_name_r is not None:
                # mps_0[-1].reindex({mps_0[-1].inds[1]: anc_name_r}, inplace=True)
                mps_0[-1].new_ind(anc_name_r, size=anc_dim)
        else:
            for i in range(mps_0.L):
                tens = mps_0[i]
                out_inds = [site_ind_id.format(i)]
                if i > 0:
                    out_inds += [mps_0.bond(i, i - 1)]
                if i < mps_0.L - 1:
                    out_inds += [mps_0.bond(i, i + 1)]
                tens.transpose(*out_inds, inplace=True)
                if i == 0 or i == mps_0.L - 1:

                    if mps_0.L == 1:
                        new_inds = tens.inds + (anc_name_l, anc_name_r)
                    else:
                        new_inds = tens.inds + (anc_name_l,) if i == 0 else tens.inds + (anc_name_r,)

                    if tens.ndim == 2:
                        tens.modify(data = np.tensordot(tens.data[:,0], np.eye(anc_dim), axes=0), inds=new_inds)
                    elif tens.ndim == 1:
                        tens.modify(data=np.tensordot(tens.data, np.eye(anc_dim), axes=0), inds=new_inds)
                        # tens.modify(data=tens.data.reshape(-1,1), inds=new_inds)
                    else:
                        raise ValueError

                else:
                    tens.modify(data = np.tensordot(tens.data[:,0,0], np.eye(anc_dim), axes=0))


        return mps_0


    def build_elemental_multiply_tn_old(self, ax: 'Axis', in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                                    site_tag_id='d_ijk({})'):
        L, q = ax.L, ax.q
        if q != 2:  raise NotImplementedError('convolution for q!=2 not implemented')

        conv_op_tn = qtn.TensorNetwork([])
        inds_list = list(range(L))

        i0 = inds_list[0]
        T0 = qtn.Tensor(np.array([[[[0.,1.],[1.,0.]],[[1.,0.],[0.,0.]]],
                                  [[[0.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]]),
                        # inds=(in1_ind_id.format(i0), f'b{i0}', out_ind_id.format(i0), in2_ind_id.format(i0)),
                        inds=(out_ind_id.format(i0), f'b{i0}', in1_ind_id.format(i0), in2_ind_id.format(i0)),
                        tags=(site_tag_id.format(i0),))
        conv_op_tn.add(T0)

        for x in range(1, L - 1):
            ix_, ix = inds_list[x - 1], inds_list[x]
            T1 = qtn.Tensor(np.array([[[[[1.,0.],[0.,0.]],[[0.,0.],[0.,0.]]],
                                       [[[0.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]],
                                      [[[[0.,1.],[1.,0.]],[[1.,0.],[0.,0.]]],
                                       [[[0.,0.],[0.,0.]],[[0.,0.],[0.,1.]]]]]),
                            # inds=(in1_ind_id.format(ix), f'b{ix_}', f'b{ix}',
                            #       out_ind_id.format(ix), in2_ind_id.format(ix)),
                            inds=(out_ind_id.format(ix), f'b{ix_}', f'b{ix}',
                                  in1_ind_id.format(ix), in2_ind_id.format(ix)),
                            tags=(site_tag_id.format(ix),))
            conv_op_tn.add(T1)

        ix_, iL = inds_list[-2], inds_list[-1]
        TL = qtn.Tensor(np.array([[[[1.,0.],[0.,0.]],[[0.,1.],[1.,0.]]],
                                  [[[0.,0.],[0.,1.]],[[0.,0.],[0.,0.]]]]),
                        # inds=(f'b{ix_}', in1_ind_id.format(iL), out_ind_id.format(iL), in2_ind_id.format(iL)),
                        inds=(f'b{ix_}', out_ind_id.format(iL), in1_ind_id.format(iL), in2_ind_id.format(iL)),
                        tags=(site_tag_id.format(iL),))
        conv_op_tn.add(TL)

        conv_op_tn = conv_op_tn.view_as(MatrixProductTensor, inplace=True, L=L, cyclic=False, site_tag_id=site_tag_id,
                                        upper_ind_id=out_ind_id, lower_ind_id=in1_ind_id, extra_ind_ids=(in2_ind_id,))

        # conv_op_tn = ax.map.transform_tn1d(conv_op_tn, out_ind_id, in1_ind_id, in2_ind_id, inplace=True)
        conv_op_tn = ax.map.transform_tn1d(conv_op_tn)
        return conv_op_tn


    def build_elemental_multiply_tn(self, ax: 'Axis', in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                                    site_tag_id='d_ijk({})', contract=True, cutoff=CUTOFF):
        L, q = ax.L, ax.q
        if q != 2 and L != 1:
            raise NotImplementedError('convolution for q!=2 and L!=1 not implemented')

        do_reshape_to_one = False
        npts = q ** L
        if L == 1:
            from axis import Axis
            q = 2
            L = int(round(np.log2(npts)))
            do_reshape_to_one = True
            old_ax = ax
            ax = Axis(L, q, x0=old_ax.x0, dx=old_ax.dx, endpoint=old_ax.endpoint)

        # mps_x = ax.map_state_to_mps(vec_data)
        # mps_x = helper.mps_flip_lr(mps_x)
        # mpo_x = helper.mps_to_diag_mpo(mps_x)

        ## QFT flips k-space MPS
        # qft_mpo = helper.mpo_flip_lr(ax.get_qft_mpo_v2())
        # qft_inv = helper.mpo_flip_lr(ax.get_qft_mpo_v2(inverse=True))
        # qft_inv_2 = helper.mpo_flip_lr(ax.get_qft_mpo_v2(inverse=True))
        qft_mpo = ax.get_qft_mpo_v2(flip_lr=True)
        qft_inv = ax.get_qft_mpo_v2(flip_lr=True, inverse=True)
        qft_inv_2 = ax.get_qft_mpo_v2(flip_lr=True, inverse=True)
        ## want forward orthogonality; current is balanced orthog
        helper.scalar_multiply(qft_inv_2, np.sqrt(ax.npts), inplace=True)

        qft_mpo.site_tag_id = site_tag_id
        qft_mpo.lower_ind_id = 'xtmp1{}'
        qft_mpo.upper_ind_id = out_ind_id
        qft_mpo.mangle_inner_()

        qft_inv.site_tag_id = site_tag_id
        qft_inv.upper_ind_id = 'xtmp2{}'
        qft_inv.lower_ind_id = in1_ind_id
        qft_inv.mangle_inner_()

        qft_inv_2.site_tag_id = site_tag_id
        qft_inv_2.upper_ind_id = 'xtmp3{}'
        qft_inv_2.lower_ind_id = in2_ind_id

        # fx = np.linspace(-np.pi / ax.dx, np.pi / ax.dx, ax.npts, endpoint=False)
        # fx_k = self.build_xmultiply_mps(ax)
        # fx_check = helper.apply(qft_inv_2, fx_k)
        # fx_check = helper.mps_flip_lr(fx_check)
        # plt.figure()
        # plt.plot(fx)
        # plt.plot(ax.map_mps_to_state(fx_check))
        # plt.show()
        # exit()

        # # original
        # d_ijk = np.zeros((2,2,2))
        # d_ijk[0,0,0] = 1.0
        # d_ijk[1,1,1] = 1.0
        # d_ijks = [qtn.Tensor(d_ijk, inds=(f'xtmp1{i}',f'xtmp2{i}', f'xtmp3{i}')) for i in range(L)]
        #
        # tn = [qtn.tensor_contract(qft_mpo[i], qft_inv[i], qft_inv_2[i], d_ijks[i]) for i in range(L)]
        #
        # conv_op_tn = qtn.TensorNetwork(tn)
        # conv_op_tn.exponent = qft_mpo.exponent + qft_inv.exponent + qft_inv_2.exponent
        # # inds_list = list(range(L))
        #
        # conv_op_tn = conv_op_tn.view_as(MatrixProductTensor, inplace=True, L=L, cyclic=False, site_tag_id=site_tag_id,
        #                                 upper_ind_id=out_ind_id, lower_ind_id=in1_ind_id, extra_ind_ids=(in2_ind_id,))
        # # print('conv op tn', conv_op_tn)
        # helper.zipup_fuse(conv_op_tn, inplace=True, direction=0)
        # helper.compress(conv_op_tn, compress_opts={'form':'left'})
        # print('conv op tn', conv_op_tn)
        # print('conv op tn', conv_op_tn.max_bond())
        # exit()

        #####################
        ## new
        if contract:
            prev_tens = []
            left_inds = []
            tn = []
            for i in range(L):
                d_ijk = np.zeros((2,2,2))
                d_ijk[0,0,0] = 1.0
                d_ijk[1,1,1] = 1.0
                d_ijk = qtn.Tensor(d_ijk, inds=(f'xtmp1{i}',f'xtmp2{i}', f'xtmp3{i}'))

                tens = qtn.tensor_contract(qft_mpo[i], qft_inv[i], qft_inv_2[i], d_ijk, *prev_tens)
                tens.drop_tags()

                if i < L-1:
                    left_inds += [in1_ind_id.format(i), in2_ind_id.format(i), out_ind_id.format(i)]
                    t1, t2 = qtn.tensor_split(tens, left_inds, cutoff=cutoff, cutoff_mode=CUTOFF_MODE, absorb='right',
                                              ltags=site_tag_id.format(i))

                    left_inds = [next(iter(t1.bonds(t2)))]
                    prev_tens = [t2]
                    tn += [t1]
                else:
                    tens.add_tag(site_tag_id.format(i))
                    tn += [tens]

            conv_op_tn = qtn.TensorNetwork(tn)
            conv_op_tn.exponent = qft_mpo.exponent + qft_inv.exponent + qft_inv_2.exponent
            # inds_list = list(range(L))

            conv_op_tn = conv_op_tn.view_as(MatrixProductTensor, inplace=True, L=L, cyclic=False, site_tag_id=site_tag_id,
                                            upper_ind_id=out_ind_id, lower_ind_id=in1_ind_id, extra_ind_ids=(in2_ind_id,))
            # print('conv op tn', conv_op_tn)
            # helper.zipup_fuse(conv_op_tn, inplace=True, direction=0)
            # helper.compress(conv_op_tn, compress_opts={'form':'left'})
            print('compress', cutoff)
            helper.compress(conv_op_tn, compress_opts={'form': 'right', 'cutoff': cutoff}, canonize=False)
            # print('conv op tn', conv_op_tn)
            # print('conv op tn', conv_op_tn.max_bond())
            # exit()

        else:
            prev_tens = []
            left_inds = []
            tn = []
            for i in range(L):
                d_ijk = np.zeros((2, 2, 2))
                d_ijk[0, 0, 0] = 1.0
                d_ijk[1, 1, 1] = 1.0
                d_ijk = qtn.Tensor(d_ijk, inds=(f'xtmp1{i}', f'xtmp2{i}', f'xtmp3{i}'), tags=site_tag_id.format(i))

                tn += [qft_mpo[i], qft_inv[i], qft_inv_2[i], d_ijk]

            conv_op_tn = qtn.TensorNetwork(tn)
            conv_op_tn.exponent = qft_mpo.exponent + qft_inv.exponent + qft_inv_2.exponent

        # conv_op_tn = ax.map.transform_tn1d(conv_op_tn)

        ###########3

        if do_reshape_to_one:
            # raise RuntimeError
            out = conv_op_tn.contract()
            out.transpose(*[out_ind_id.format(i) for i in range(L)],
                          *[in1_ind_id.format(i) for i in range(L)],
                          *[in2_ind_id.format(i) for i in range(L)], inplace=True)
            out_data = out.data.reshape(npts,npts,npts)

            out_tn = qtn.TensorNetwork([qtn.Tensor(out_data, inds=(out_ind_id.format(0), in1_ind_id.format(0),
                                                                   in2_ind_id.format(0)), tags=site_tag_id.format(0))])
            out_tn.view_as(MatrixProductTensor, inplace=True, L=1, cyclic=False, site_tag_id=site_tag_id,
                           upper_ind_id=out_ind_id, lower_ind_id=in1_ind_id, extra_ind_ids=(in2_ind_id,))
            out_tn.exponent = conv_op_tn.exponent
            conv_op_tn = out_tn

        return conv_op_tn


    def build_xmultiply_mps(self, ax, x_power=1, offset=0.0, scale=1.0, split_opts=None):
        """ x * exp(ikx) for all k...
        """
        # print('build x multiply mps k')
        fx = np.linspace(-np.pi / ax.dx, np.pi / ax.dx, ax.npts, endpoint=False)
        fx = (scale * fx + offset) ** x_power
        fx_ks = np.fft.fftshift( np.fft.fft(fx, norm='forward') )
        mps_x = ax.map_state_to_mps(fx_ks)
        # plt.figure()
        # plt.plot(np.real(fx_ks))
        # plt.plot(np.imag(fx_ks))
        # plt.plot(ax.map_mps_to_state(mps_x))
        # plt.show()
        return mps_x

    def build_xmultiply_mpo_v2(self, ax, x_power=1, offset=0.0, scale=1.0, split_opts=None):
        """ x * exp(ikx) for all k...
        """
        # print('build x multiply mpo k')
        mps_x = self.build_xmultiply_mps(ax, x_power=x_power, offset=offset, scale=scale, split_opts=split_opts)
        elem_mult = self.build_elemental_multiply_tn(ax, in2_ind_id=mps_x.site_ind_id)
        mpo_x = qtn.TensorNetwork([qtn.tensor_contract(mps_x[i], elem_mult[i]) for i in range(ax.L)])
        mpo_x.fuse_multibonds(inplace=True,)
        mpo_x.exponent = mps_x.exponent + elem_mult.exponent
        mpo_x.view_as(qtn.MatrixProductOperator, inplace=True, L=ax.L, cyclic=False,
                      upper_ind_id=elem_mult.upper_ind_id, lower_ind_id=elem_mult.lower_ind_id,
                      site_tag_id=mps_x.site_tag_id)

        # comp = ax.map_mps_to_state(mps_x)
        # mat = ax.map_mpo_to_operator(mpo_x)
        # vec = np.zeros(len(mat))
        # vec[len(vec)//2] = 1.0
        #
        # plt.figure()
        # plt.imshow(np.real(mat))
        # plt.colorbar()
        #
        # plt.figure()
        # plt.imshow(np.imag(mat))
        # plt.colorbar()
        #
        # plt.figure()
        # plt.plot(mat @ vec)
        # # print('comp', comp.shape, vec.shape)
        # plt.plot(np.convolve(comp, vec, mode='same'))
        # plt.plot(comp,'k--')
        # plt.show()
        #
        # exit()

        return mpo_x

    # def build_xmultiply_mpo(self, ax, x_power=1, offset=0.0, scale=1.0, split_opts=None):
    #     """ x * exp(ikx) for all k...
    #     """
    #     fx = np.linspace(-np.pi / ax.dx, np.pi / ax.dx, ax.npts, endpoint=False)
    #     fx = (scale * fx + offset) ** x_power
    #     fx_ks = np.fft.fftshift(np.fft.fft(fx, norm='forward'))
    #
    #     return self.build_fmultiply_mpo_ks(ax, fx_ks)

    def build_xmultiply_mpo(self, ax, x_power=1, offset=0.0, scale=1.0, split_opts=None):
        """ x * exp(ikx) for all k...
        """
        dk = ax.dx
        real_x = np.linspace(-np.pi / dk, np.pi / dk, ax.npts, endpoint=False)
        from basis.basis_spatial import SpatialBasis
        from axis import Axis
        ax_spatial = Axis(ax.L, xpts=real_x, ax_map=ax.map)
        fx_mpo = SpatialBasis().build_xmultiply_mpo(ax_spatial, x_power=x_power, offset=offset, scale=scale)

        qft_mpo = ax_spatial.get_qft_mpo_v2()
        qft_mpo_inv = ax_spatial.get_qft_mpo_v2(inverse=True)

        fk_mpo = helper.apply_zipup(fx_mpo, qft_mpo_inv, compress=False)
        fk_mpo = helper.apply_zipup(qft_mpo, fk_mpo, compress=True, compress_opts=split_opts)
        helper.mpo_flip_lr(fk_mpo, inplace=True)
        return fk_mpo

        # fx = np.linspace(-np.pi / ax.dx, np.pi / ax.dx, ax.npts, endpoint=False)
        # fx = (scale * fx + offset) ** x_power
        # fx_ks = np.fft.fftshift(np.fft.fft(fx, norm='forward'))
        #
        # return self.build_fmultiply_mpo_ks(ax, fx_ks)

    def build_fmultiply_mpo_ks(self, ax: 'Axis', vec_data: np.ndarray):
        """ x * exp(ikx) for all k...
        """
        # print('build x multiply mpo k')
        mps_x = ax.map_state_to_mps(vec_data, site_ind_id='tmp{}')

        out_ind_id = 'o{}'
        in1_ind_id = 'i{}'
        site_tag_id = mps_x.site_tag_id

        elem_mult = self.build_elemental_multiply_tn(ax, in1_ind_id=in1_ind_id, in2_ind_id=mps_x.site_ind_id,
                                                     out_ind_id=out_ind_id, site_tag_id=site_tag_id,
                                                     contract=False, )

        tensors = []
        prev_tens = []
        left_inds = []
        for i in range(ax.L):
            tn_tens = elem_mult.select_tensors(site_tag_id.format(i))
            tens = qtn.tensor_contract(*tn_tens, mps_x[i], *prev_tens)

            if i < ax.L - 1:
                left_inds += [in1_ind_id.format(i), out_ind_id.format(i)]
                t1, t2 = qtn.tensor_split(tens, left_inds, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE, absorb='right',
                                          ltags=site_tag_id.format(i))

                t2.drop_tags()
                left_inds = [next(iter(t1.bonds(t2)))]
                prev_tens = [t2]
                tensors += [t1]
            else:
                tens.add_tag(site_tag_id.format(i))
                tensors += [tens]

        # mpo_x = qtn.TensorNetwork([qtn.tensor_contract(mps_x[i], elem_mult[i]) for i in range(ax.L)])
        mpo_x = qtn.TensorNetwork(tensors)
        mpo_x.fuse_multibonds(inplace=True,)
        mpo_x.exponent = mps_x.exponent + elem_mult.exponent
        mpo_x.view_as(qtn.MatrixProductOperator, inplace=True, L=ax.L, cyclic=False,
                      upper_ind_id=out_ind_id, lower_ind_id=in1_ind_id,
                      site_tag_id=site_tag_id)
        mpo_x = helper.compress(mpo_x, compress_opts={'form':'right'}, canonize=False)
        print('mpo x max rank', mpo_x.max_bond())

        return mpo_x


    def build_fmultiply_mpo(self, ax: 'Axis', vec_data: np.ndarray):
        """ x * exp(ikx) for all k...
            vec data is in real space
        """
        raise NotImplementedError
        do_reshape_to_one = False
        if ax.L == 1:
            q, L = 2, int(np.round(np.log2(ax.q)))

            from axis import Axis
            ax = Axis(L, q)
            do_reshape_to_one = True

        mps_x = ax.map_state_to_mps(vec_data)
        mps_x = helper.mps_flip_lr(mps_x)
        mpo_x = helper.mps_to_diag_mpo(mps_x)

        # plt.figure()
        # plt.imshow(np.real(ax.map_mpo_to_operator(mpo_x)))
        # plt.title('diag(real vec data) fliplr')
        # plt.show()

        ## QFT flips k-space MPS
        qft_mpo = helper.mpo_flip_lr(ax.get_qft_mpo_v2())
        qft_inv = helper.mpo_flip_lr(ax.get_qft_mpo_v2(inverse=True))
        # qft_mpo = ax.get_qft_mpo_v2(flip_lr=True)
        # qft_inv = ax.get_qft_mpo_v2(flip_lr=True, inverse=True)

        # test_vec = vec_data
        # # test_vec = np.sin(ax.xpts / (ax.xpts[-1] - ax.xpts[0] + ax.dx) * 2 * np.pi)
        # # test_vec = np.ones(ax.q**ax.L)
        # test_mps = ax.map_state_to_mps(test_vec)
        # test_mps = helper.mps_flip_lr(test_mps)
        # check = helper.apply(qft_mpo, test_mps)
        # check2 = helper.apply(qft_inv, check)
        # check2 = helper.mps_flip_lr(check2)
        # plt.figure()
        # plt.plot(np.real(ax.map_mps_to_state(check)))
        # plt.plot(np.real(ax.map_mps_to_state(check2)))
        # plt.plot(test_vec, 'k--')
        # plt.title('re ifft, fft (test)')
        #
        # plt.figure()
        # plt.plot(np.imag(ax.map_mps_to_state(check)))
        # plt.plot(np.imag(ax.map_mps_to_state(check2)))
        # plt.title('im ifft, fft (test)')
        # plt.show()

        out = helper.apply(mpo_x, qft_inv)
        out = helper.apply_zipup(qft_mpo, out, compress=True)

        if do_reshape_to_one:
            mpo_data = helper.to_dense(out)
            mpo_1 = qtn.TensorNetwork(
                [qtn.Tensor(mpo_data, inds=(out.upper_ind_id.format(0), out.lower_ind_id.format(0)),
                            tags=(out.site_tag_id.format(0)))])
            mpo_1.view_as(qtn.MatrixProductOperator, cyclic=False, inplace=True, L=1,
                          upper_ind_id=out.upper_ind_id, lower_ind_id=out.lower_ind_id, site_tag_id=out.site_tag_id)
            out = mpo_1

        # plt.figure()
        # plt.imshow(np.real(ax.map_mpo_to_operator(out)))
        # plt.colorbar()
        # plt.title('fmult out re')
        # plt.figure()
        # plt.imshow(np.imag(ax.map_mpo_to_operator(out)))
        # plt.colorbar()
        # plt.title('fmult out im')
        # plt.show()
        # exit()
        # print('fmultiply (k) max bond', out.max_bond())
        return out

    def build_firstderivative_mpo(self, ax, **kwargs):
        ks = ax.xpts
        mps_ks = ax.map_state_to_mps(ks)
        helper.scalar_multiply(mps_ks, 1.j, inplace=True)
        mpo = helper.mps_to_diag_mpo(mps_ks)
        return mpo

    def build_firstderivative_mpo_inverse(self, ax, **kwargs):
        ks = ax.xpts
        inv_ks = 1./ks
        inv_ks[ks==0] = 0
        mps_ks = ax.map_state_to_mps(inv_ks)
        ## ideally would provide constraint for k=0
        helper.scalar_multiply(mps_ks, -1.j, inplace=True)
        mpo = helper.mps_to_diag_mpo(mps_ks)
        return mpo


    def build_secondderivative_mpo(self, ax, **kwargs):
        ks = ax.xpts**2
        mps_k2s = ax.map_state_to_mps(ks)
        helper.scalar_multiply(mps_k2s, -1., inplace=True)
        mpo = helper.mps_to_diag_mpo(mps_k2s)
        return mpo

    def build_mth_derivative_mpo(self, ax: 'Axis', deriv_order:int, **kwargs):
        ks = ax.xpts**deriv_order
        mps_k2s = ax.map_state_to_mps(ks)
        helper.scalar_multiply(mps_k2s, 1.j**deriv_order, inplace=True)
        mpo = helper.mps_to_diag_mpo(mps_k2s)
        return mpo


    def get_integral_weight(self, ax: 'Axis'):
        """ normalization for integral_mps
        """
        return 2 * np.pi / ax.dx


    def build_integral_mps(self, ax, is_sqrt=False, site_ind_id='i({})', site_tag_id='T({})'):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        # domain is from -np.pi/dk to np.pi/dk
        if is_sqrt:
            raise NotImplementedError
            ## this is wrong
            # integ_mps = ax.get_ones_mps(site_ind_id, site_tag_id) * 2 * np.pi / ax.dx
        else:
            print('integ ax zero ind', ax.zero_ind)
            integ_mps = ax.get_select_elems_mps([ax.zero_ind]) * 2 * np.pi / ax.dx

        return integ_mps


    def build_indefinite_integral_mps(self, ax, order=1):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        raise NotImplementedError




class RealFourierBasis(FourierBasis):
    """ assumes data is real and that a_k = a_{-k}^* for some wavevector k
        (and a_0 is real)
    """

    def get_realspace_1D(self, data, ax_ind, **kwargs):
        """ coeffs:  coefficients of wavevector (0, k, 2k, ...)
            x_window:  x0,xL specifying domain of interest
            x_offset:  offset of center of Hermite polynomials
            x_scale:   scaling of x in Hermite polynomials     (x <- x_scale*x + x_offset)
            note: numpy fct is physicists Hermite polynomials (no normalization)
        """
        # data = np.moveaxis(data, ax_ind, -1)
        # npts = data.shape[-1]
        # out_data = np.fft.irfftn(data, axes=[-1], s=[2*npts]) # 0 frequency term at 0
        # out_data = np.fft.ifftshift(out_data, axes=-1) * data.shape[-1] * 2
        # out_data = np.moveaxis(out_data, -1, ax_ind)

        print('real real space', ax_ind)
        npts = data.shape[ax_ind]
        out_data = np.fft.irfft(data, axis=ax_ind, n=2 * npts)  # 0 frequency term at 0
        out_data = np.fft.ifftshift(out_data, axes=ax_ind) * npts * 2
        return out_data


    def get_realspace_nD(self, data: 'np.ndarray', ax_inds: list[int], **kwargs):
        """ ie. coeffs of delta fcts """
        data_size = np.prod([data.shape[x] for x in ax_inds])
        out_data = np.fft.irfftn(data, axes=ax_inds) * data_size * (2**len(ax_inds))
        return out_data


    def build_elemental_multiply_tn(self, ax, in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                                    site_tag_id='d_ijk({})'):
        """ requires taking complex conjugates of MPSs via MPOs, which i don't think is possible
            no reduction in number of operations in real representation
            just memory and enforcing of real-ness
        """
        L, q = ax.L, ax.q

        full_ax = ax.__class__(L+1, q=ax.q, ax_map=axis_map.MirrorMap())
        conv_op_tn = super().build_elemental_multiply_tn(full_ax, in1_ind_id, in2_ind_id, out_ind_id, site_tag_id)
        ## Mirror mapping --> # (-kmax,-kmax+1,...,-1) (kmax-1, kmax-2, ..., 1, 0)

        from axis import get_reverse_diag_mpo

        # apply SX's on all qubits --> (0, 1, ... kmax-1) (-1, -2, ... -kmax)
        flip_ind_mpo = get_reverse_diag_mpo(L+1, q, site_tag_id=site_tag_id, upper_ind_id=out_ind_id,
                                            lower_ind_id=out_ind_id+'_tmp_')         ## MPO of SX's
        ### conv_op_tn = conv_op_tn.apply(flip_ind_mpo)     # applies SX's on output leg (upper_ind_id)
            ### TODO: above should work but doesn't. not sure why yet (below still works)
        conv_op_tn.upper_ind_id = out_ind_id + '_tmp_'
        conv_op_tn.add(flip_ind_mpo)
        for i in range(L+1):
            conv_op_tn.contract(site_tag_id.format(i), inplace=True)
        conv_op_tn.fuse_multibonds(inplace=True)
        conv_op_tn.view_like(conv_op_tn, upper_ind_id=out_ind_id, extra_ind_ids=conv_op_tn.extra_ind_ids, inplace=True)
            ### TODO: fix above so that not need to specify extra_ind_ids
        # print('conv op tn', conv_op_tn)
        # print('conv_op tn', conv_op_tn.upper_ind_id, conv_op_tn.lower_ind_id, conv_op_tn.extra_ind_ids)


        ## after apply SX's on input legs (will do later)
        ## shift to avoid repeating the k=0 element for anc=1 (when applying to two (0, 1, ..., kmax-1) MPS)
        ## apply SX's on all qubits --> (-2, -3, ... -kmax-1)  (but input is (0, 1, ..., kmax-1)
        bin_ax = ax.__class__(L, q=ax.q, ax_map=axis_map.BinaryMap())
        shift_mpo = bin_ax.get_shift_mpo(1, upper_ind_id=in1_ind_id, lower_ind_id=in1_ind_id + '_tmp_',
                                         site_tag_id=site_tag_id)
        helper.renumber_mpo(shift_mpo, list(range(L)), list(range(1, L + 1)), inplace=True)
        tens_sel_1 = qtn.Tensor(np.array([[0.,0.],[0.,1.]]),
                              inds=(shift_mpo.upper_ind_id.format(0), shift_mpo.lower_ind_id.format(0)),
                              tags=(shift_mpo.site_tag_id.format(0),))
        shift_mpo.add(tens_sel_1)
        shift_mpo.view_like(shift_mpo, L=L+1, inplace=True)
        shift_mpo[0].new_bond(shift_mpo[1])

        ## inputs for anc=0 remain unchanged
        iden = bin_ax.get_iden_mpo(upper_ind_id=in1_ind_id, lower_ind_id=in1_ind_id+'_tmp_', site_tag_id=site_tag_id)
        helper.renumber_mpo(iden, list(range(L)), list(range(1, L + 1)), inplace=True)
        tens_sel_0 = qtn.Tensor(np.array([[1., 0.], [0., 0.]]),
                                inds=(shift_mpo.upper_ind_id.format(0), shift_mpo.lower_ind_id.format(0)),
                                tags=(shift_mpo.site_tag_id.format(0),))
        iden.add(tens_sel_0)
        iden.view_like(iden, L=L+1, inplace=True)
        iden[0].new_bond(iden[1])

        helper.add_MPO(shift_mpo, iden, inplace=True)
        # print('shift mpo', shift_mpo)

        ## apply to TN
        adj_inds_mpo = helper.apply(flip_ind_mpo, shift_mpo)
        # adj_inds_mpo = shift_mpo
        # print('adj inds mpo', adj_inds_mpo)


        adj_inds_mpo2 = adj_inds_mpo.copy(virtual=True)
        adj_inds_mpo2.upper_ind_id = in2_ind_id
        adj_inds_mpo2.lower_ind_id = in2_ind_id + '_tmp_'

        conv_op_tn.add([adj_inds_mpo, adj_inds_mpo2], virtual=True)   # still a MatrixProductTensor
        for i in range(L+1):
            conv_op_tn.contract((site_tag_id.format(i)), inplace=True)
            tens = conv_op_tn.select_tensors(site_tag_id.format(i))[0]
            tens.reindex({in1_ind_id.format(i)+'_tmp_': in1_ind_id.format(i),
                          in2_ind_id.format(i)+'_tmp_': in2_ind_id.format(i)}, inplace=True)

        conv_op_tn.fuse_multibonds(inplace=True)
        conv_op_tn.exponent += adj_inds_mpo.exponent + adj_inds_mpo2.exponent


        ## take first tensor and treat input legs as ancilla
        conv_op_tn_rest, conv_op_tn_0 = conv_op_tn.partition(site_tag_id.format(0))

        conv_op_tn_0 = conv_op_tn_0.tensors[0]
        conv_op_tn_0.isel({out_ind_id.format(0):0}, inplace=True)   # keep only positive ks
        conv_op_tn_0.reindex({in1_ind_id.format(0):f'anc_elem_mult_1_{ax}',
                              in2_ind_id.format(0):f'anc_elem_mult_2_{ax}'}, inplace=True)

        for i in range(1,L+1):
            tens = conv_op_tn_rest.select_tensors(site_tag_id.format(i))[0]
            tens.reindex({in1_ind_id.format(i): in1_ind_id.format(i-1),
                          in2_ind_id.format(i): in2_ind_id.format(i-1),
                          out_ind_id.format(i): out_ind_id.format(i-1)}, inplace=True)
            tens.retag({site_tag_id.format(i): site_tag_id.format(i-1)}, inplace=True)

            if i == 1:
                tens_new = tens.contract(conv_op_tn_0)
                tens.modify(data=tens_new.data, inds=tens_new.inds)

        # print('conv op tn rest exp', conv_op_tn_rest.exponent)
        conv_op_tn_rest.exponent = conv_op_tn.exponent
        conv_op_tn_rest.view_like(conv_op_tn, L=L, inplace=True)
        ## should be in binary ordering from k = (0, 1, ..., kmax)
        conv_op_tn = ax.map.transform_tn1d(conv_op_tn_rest)
        helper.compress(conv_op_tn)

        # print('tot conv op', conv_op_tn)
        # print(type(conv_op_tn))

        # ## check
        # # check = ax.get_ones_mps(site_ind_id=in2_ind_id, site_tag_id=site_tag_id)
        # check = ax.get_select_elems_mps([1], site_ind_id=in2_ind_id, site_tag_id=site_tag_id)
        # check[0].new_ind(f'anc_elem_mult_2_{ax}', size=1)
        # check = helper.add_MPS(check, check)
        # conv_op_tn_check = conv_op_tn.copy()
        # conv_op_tn_check.add(check)
        # for i in range(L):
        #     conv_op_tn_check.contract(site_tag_id.format(i), inplace=True)
        # conv_op_tn_check = conv_op_tn_check.view_as(qtn.MatrixProductOperator, upper_ind_id=out_ind_id,
        #                                             lower_ind_id=in1_ind_id)
        # conv_op_tn_check.isel({f'anc_elem_mult_1_{ax}': 0}, inplace=True)
        # conv_op_data = ax.map_mpo_to_operator(conv_op_tn_check)
        # print(type(conv_op_tn))
        #
        # plt.figure()
        # plt.imshow(conv_op_data.T)
        # plt.colorbar()
        # plt.show()
        # exit()

        return conv_op_tn


    @classmethod
    def convert_mps_to_full(cls, grid, mps_data, anc_ind=1, inplace=False):
        """ mps + mps.conj()
        """
        mps_data = mps_data if inplace else mps_data.copy()

        for ax in grid.axes:
            if not ax.is_real_k():  continue

            ax_inds = grid.get_inds_in_axis(ax)
            if ax.map.is_flipped():
                mps_data[ax_inds[-1]].new_ind(f'anc_elem_mult_{anc_ind}_{ax}')
            else:
                mps_data[ax_inds[0]].new_ind(f'anc_elem_mult_{anc_ind}_{ax}')

        conj_data = mps_data.conj()  ## should be ok to take complex conj. of entire state
        mps_data = helper.add_MPS(mps_data, conj_data, inplace=True, compress=False)
        return mps_data


    def build_xmultiply_mpo(self, ax, x_power=1, offset=0.0, scale=1.0, split_opts=None):
        """ x * exp(ikx) for all k...
        """
        raise NotImplementedError

    # def build_firstderivative_mpo(self, ax, **kwargs):
    #     return super().build_firstderivative_mpo(ax, **kwargs)
    #
    # def build_secondderivative_mpo(self, ax, **kwargs):
    #     return super().build_secondderivative_mpo(ax, **kwargs)

    def get_integral_weight(self, ax: 'Axis'):
        """ normalization for integral_mps
            integ F[k] F*[k'] dk dk' = sum k,k' a[k] a*[k'] delta(k,k') * 2*np.pi/dk for k = (-kmax,kmax)
            -->  a[0] + 2 * ( a[1] + a[2] + ... ) * 2*np.pi/dk
        """
        # integ_mps = ax.get_ones_mps()
        # helper.scalar_multiply(integ_mps, 2., inplace=True)
        # select_zero_mps = ax.get_select_elems_mps([0])
        # helper.scalar_multiply(select_zero_mps, -1., inplace=True)
        # integ_mps = helper.add_MPS( integ_mps, select_zero_mps, inplace=True)

        integ_mps = ax.map_state_to_mps(np.array([1.] + [2.]*(ax.npts-1)))
        helper.scalar_multiply(integ_mps, super().get_integral_weight(ax), inplace=True)
        return integ_mps
        # return 2 * super().get_integral_weight(ax)

    # def build_integral_mps(self, ax, is_sqrt=False, site_ind_id='i({})', site_tag_id='T({})'):
    #     """ get MPS that integrates along Axis ax. multiply by dx here
    #     """
    #     integ_mps = super().build_integral_mps(ax, is_sqrt=is_sqrt, site_ind_id=site_ind_id, site_tag_id=site_tag_id)
    #
    #     # # domain is from -np.pi/dk to np.pi/dk
    #     # if is_sqrt:
    #     #     raise NotImplementedError
    #     #     ## this is wrong
    #     #     # integ_mps = ax.get_ones_mps(site_ind_id, site_tag_id) * 2 * np.pi / ax.dx
    #     # else:
    #     #     # ax.zero_ind should be 0
    #     #     integ_mps = ax.get_select_elems_mps([ax.zero_ind]) * 2 * np.pi / ax.dx
    #
    #     return integ_mps
    #
    def build_indefinite_integral_mps(self, ax, order=1):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        raise NotImplementedError
