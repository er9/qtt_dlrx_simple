import re
import functools
from setup_.defaults import *
import helper_quimb as helper
from layout.layout import Layout

""" LayoutParallellF:   parallel: n-dimensions are ordered in parallel (x0,y0,z0),(x1,y1,z1),...
                                  ie. interwoven
                        F:  tensors corresponding to different dimensions remain factorized
"""

class LayoutParallelF(Layout):

    @classmethod
    def L(cls, axes: Sequence['Axis']) -> int:
        mpx_L = sum([ax.L for ax in axes])
        return mpx_L

    @classmethod
    @functools.lru_cache()
    def get_inds_in_axis(cls, axes: tuple['Axis'], ax: 'Axis', ax_ind=None) -> list:
        """ get indices corresponding to tensors along a given axis (for factorized TNs)
            assumes axes are all aligned at index 0
            probably should cache this
        """
        if ax_ind is None:
            # axIDs = [ax.axID for ax in axes]
            ax_ind = axes.index(ax)

        if all([ax_.L == ax.L for ax_ in axes]):     # all have the same length
            ndim = len(axes)
            L0 = axes[0].L
            return [ndim * i + ax_ind for i in range(L0)]
        else:
            Ls = [ax_.L for ax_ in axes]
            n_greater = 0

            inds = []
            for i in range(ax.L):
                n_i = sum([L > i for L in Ls[:ax_ind]])
                # number of axes that precede ax of length >= i
                inds += [n_greater + n_i]
                n_greater += n_i + sum([L > i for L in Ls[ax_ind:]])
                # add current tens, axes that follow ax of len >= i

            return inds


    @classmethod
    @functools.lru_cache()
    def shape(cls, axes: tuple['Axis']) -> tuple:
        """ get shape
            assumes axes are all aligned at index 0
        """
        if np.all([ax_.L == axes[0].L for ax_ in axes]):  # all have the same length
            qs = [ax.q for ax in axes]
            L0 = axes[0].L
            return tuple(qs)*L0
        else:
            Lmax = max([ax_.L for ax_ in axes])
            qs = []
            for i in range(Lmax):
                for ax_ in axes:
                    if ax_.L > i:
                        qs += [ax_.q]
            return tuple(qs)


    #########################
    ## convert array to TN ##
    #########################

    @classmethod
    # @profile
    def map_state_to_mps(cls, axes, state, site_ind_id='i({})', site_tag_id='T({})', direction=0, split_opts=None,
                         ancilla_right=(), ancilla_right_inds=(), ancilla_left=(), ancilla_left_inds=()):
        """ state is initially written as an ndim-dimensional tensor of with each leg of size q**ax.L
            dim 0 x dim 1 x dim 2 ....
        """
        Ls = tuple([ax_.L for ax_ in axes])
        L = np.sum(Ls)

        ## fold each axis into n-ary form, each dimension considered serially
        ## eg. (q x q x q ...) ** self.dim
        shape_seq = super().shape(axes)
        num_pts = np.prod(shape_seq)
        shape_seq = ancilla_left + shape_seq + ancilla_right

        assert (state.size % num_pts == 0), \
            f'state size should be multiple of {num_pts} not {state.shape}'

        assert (state.ndim - len(ancilla_left) - len(ancilla_right) == len(axes)), \
            'state should be K-dimensional form'
        for i in range(len(axes)):
            ax = axes[i]
            state = ax.map.transform_vector(state, axis=len(ancilla_left) + i)

        ###
        new_tens = state.reshape(shape_seq)

        ## group dims for same TN position together
        # axT = super()._transpose_sequential_to_parallel_inds(Ls)
        ax_inds = super()._transpose_sequential_to_parallel_inds(axes)
        tens_inds = super().get_tensor_inds(axes)
        axT = [tens_inds[x] for x in ax_inds]

        if ancilla_left or ancilla_right:

            num_anc_l, num_anc_r = len(ancilla_left), len(ancilla_right)

            axT += list(range(len(axT), len(axT) + num_anc_r))
            axT = list(range(num_anc_l)) + [ax_ind + num_anc_l for ax_ind in axT]
            tens_inds = ancilla_left_inds \
                        + tuple([site_ind_id.format(ix) for ix in range(L)]) \
                        + ancilla_right_inds

            new_tens = new_tens.transpose(axT)
            new_tens = qtn.Tensor(new_tens, inds=tens_inds)

            if len(ancilla_left) == 0:           # only ancilla on right
                site_nlegs = (1,) * (L-1) + (num_anc_r + 1,)
            elif len(ancilla_right) == 0:        # only ancilla on left
                site_nlegs = (num_anc_l + 1,) + (1,) * (L-1)
            else:                           # both exist
                site_nlegs = (num_anc_l + 1,) + (1,) * (L-2) + (num_anc_r + 1,)

            new_mps = helper.tn1D_from_dense(new_tens, L, site_nlegs, direction=direction, split_opts=split_opts,
                                             site_tag_id=site_tag_id)
            new_mps =  qtn.MatrixProductState.from_TN(new_mps, inplace=True, cyclic=False, L=L,
                                                      site_tag_id=site_tag_id, site_ind_id=site_ind_id)


        else:
            new_tens = state.reshape(shape_seq)
            new_tens = new_tens.transpose(axT)

            ## decompose into an MPS using quimb
            new_tens = qtn.Tensor(new_tens, inds=[site_ind_id.format(i) for i in range(sum(Ls))])
            new_mps = helper.mpx_from_dense(new_tens, sum(Ls), [site_ind_id], site_tag_id=site_tag_id, return_mpx=True,
                                            split_opts=split_opts)

        return new_mps


    # @classmethod
    # def map_state_to_mps(cls, axes, state, site_ind_id='i({})', site_tag_id='T({})', direction=0, split_opts=None,
    #                      ancilla_right=(), ancilla_right_inds=(), ancilla_left=(), ancilla_left_inds=()):
    #     """ state is initially written as an ndim-dimensional tensor of with each leg of size q**ax.L
    #         dim 0 x dim 1 x dim 2 ....
    #     """
    #     Ls = tuple([ax_.L for ax_ in axes])
    #
    #     ## fold each axis into n-ary form, each dimension considered serially
    #     ## eg. (q x q x q ...) ** self.dim
    #     shape_seq = super().shape(axes)
    #     num_pts = np.prod(shape_seq)
    #
    #     assert (state.size % num_pts == 0), \
    #         f'state size should be multiple of {num_pts} not {state.shape}'
    #
    #     assert (state.ndim - len(ancilla_left) - len(ancilla_right) == len(axes)), \
    #         'state should be K-dimensional form'
    #     for i in range(len(axes)):
    #         ax = axes[i]
    #         state = ax.map.transform_vector(state, axis=len(ancilla_left)+i)
    #
    #     if ancilla_left or ancilla_right:
    #         raise NotImplementedError
    #
    #     else:
    #         new_tens = state.reshape(shape_seq)
    #
    #         ## group dims for same TN position together
    #         # axT = super()._transpose_sequential_to_parallel_inds(Ls)
    #         ax_inds = super()._transpose_sequential_to_parallel_inds(axes)
    #         tens_inds = super().get_tensor_inds(axes)
    #         axT = [tens_inds[x] for x in ax_inds]
    #         # print('axis_map state to mps', cls, axT)
    #         # axT = []
    #         # for x in range(self.L0):
    #         #     axT += [x + i * self.L0 for i in range(ndim)]
    #         new_tens = new_tens.transpose(axT)
    #
    #         ## decompose into an MPS using quimb
    #         # if split_opts is None:   split_opts = {}
    #         # else:                    split_opts = split_opts.copy()
    #         # split_opts['absorb'] = 'right'
    #         new_tens = qtn.Tensor(new_tens, inds=[site_ind_id.format(i) for i in range(sum(Ls))])
    #         new_mps = helper.mpx_from_dense(new_tens, sum(Ls), [site_ind_id], site_tag_id=site_tag_id, return_mpx=True,
    #                                         split_opts=split_opts)
    #
    #     return new_mps


    @classmethod
    def map_operator_to_mpo(cls, axes, operator, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='T({})',
                            direction=0, split_opts=None, ancilla_right=(), ancilla_right_inds=(), ancilla_left=(),
                            ancilla_left_inds=()):
        """ state is initially written as an ndim-dimensional tensor of length q**L
            n-dim operator is (i0 x i1 x i2 ...) x (o0 x o1 x o2 ...)
        """
        Ls = [ax_.L for ax_ in axes]

        ## fold each axis into n-ary form, each dimension considered serially
        ## eg. (q x q x q ...) ** self.dim for out x in
        shape_seq = super().shape(axes)   ## shape with sequential ordering of axes
        num_pts = np.prod(shape_seq)

        assert (operator.size == num_pts**2), \
            ('state size should be ' + num_pts**2 + 'not ' + str(operator.shape))

        if ancilla_left or ancilla_right:
            raise NotImplementedError

        assert (operator.ndim - len(ancilla_left) - len(ancilla_right) == 2 * len(axes)), \
            'state should be K-dimensional form'
        for i in range(len(axes)):
            ax = axes[i]
            operator = ax.map.transform_operator(operator, axis1=len(ancilla_left) + i,
                                                 axis2=len(ancilla_left) + len(axes) + i)

        new_tens = operator.reshape(shape_seq*2)

        ## group dims for same spatial position together
        ax_inds = super()._transpose_sequential_to_parallel_inds(axes)      # sequential to parallel axT
        tens_inds = super().get_tensor_inds(axes)   # how indices of data tensor are ordered (sequential ordering)
        axTo = [tens_inds[x] for x in ax_inds]
        axTi = [sum(Ls) + axTo[x] for x in range(sum(Ls))]
        # print('axis_map operator to mpo', cls, axTo, axTi)
        new_tens = new_tens.transpose(np.append(axTo,axTi))

        ## decompose into an MPS using quimb
        # if split_opts is None:   split_opts = {}
        # else:                    split_opts = split_opts.copy()
        # split_opts['absorb'] = 'right'
        new_tens = qtn.Tensor(new_tens, inds=[upper_ind_id.format(i) for i in range(sum(Ls))] + \
                                             [lower_ind_id.format(i) for i in range(sum(Ls))])
        new_mpo = helper.mpx_from_dense(new_tens, sum(Ls), [upper_ind_id, lower_ind_id], site_tag_id=site_tag_id,
                                        return_mpx=True, split_opts=split_opts)
        return new_mpo


    @classmethod
    def map_mps_to_state(cls, axes, mps, ax_select=None, ancilla_right=(), ancilla_right_inds=(), ancilla_left=(),
                         ancilla_left_inds=()):
        """ convert MPS into 1-D np.ndarray
        """
        ## contract and reorder to sequential ordering
        # ax_inds = super()._transpose_parallel_to_sequential_inds(axes)      # parallel to sequential axT
        # tens_inds = super().get_tensor_inds(axes)   # how indices of data tensor are ordered (sequential ordering)
        out_axes = axes

        if ax_select is not None and len(ax_select) > 0:
            mps = mps.copy()
            for ax, select_ind in ax_select.items():
                if ax in axes:
                    select_mps = ax.get_select_elems_mps([select_ind], site_tag_id=mps.site_tag_id,
                                                         site_ind_id=mps.site_ind_id)
                    mps_inds = cls.get_inds_in_axis(axes, ax)
                    helper.renumber_mps(select_mps, list(range(select_mps.L)), mps_inds, inplace=True)
                    mps.add(select_mps)

            out_axes = [ax for ax in axes if ax not in ax_select]

        ax_inds = []
        for ax in out_axes:
            ax_inds += cls.get_inds_in_axis(axes, ax)
        tens_inds = super().get_tensor_inds(out_axes)

        out_idxs = [ax_inds[x] for x in tens_inds]
        out_inds = [mps.site_ind_id.format(i) for i in out_idxs]
        # print('axis_map mps to state', cls, axT)

        tensor: qtn.Tensor = mps.contract()  # contract all tensors
        if isinstance(tensor, qtn.Tensor):
            tensor.transpose(*ancilla_left_inds, *out_inds, *ancilla_right_inds, inplace=True)

            ## need to reshape to ndim-dimensional tensor
            out_data = tensor.data.reshape(ancilla_left+tuple([ax.npts for ax in out_axes])+ancilla_right)

            for i in range(len(out_axes)):
                ax = out_axes[i]
                out_data = ax.map.inverse_transform_vector(out_data, axis=len(ancilla_left) + i)
        elif isinstance(tensor, (float, complex)):
            out_data = tensor

        # if len(ancilla_left) + len(ancilla_right) > 0:
        #     ax_inds = tuple([f'i({ax})' for ax in axes])
        #     out_tens = qtn.Tensor(data=out_data * (10 ** mps.exponent),
        #                           inds=ancilla_left_inds + ax_inds + ancilla_right_inds)
        #     return out_tens
        # else:
        #     return out_data * (10 ** mps.exponent)
        return out_data * (10 ** mps.exponent)


    @classmethod
    def map_mpo_to_operator(cls, axes, mpo, ax_select=None, ancilla_right=(), ancilla_right_inds=(), ancilla_left=(),
                            ancilla_left_inds=()):
        """ convert MPO into 2*K-D np.ndarray (o0 x o1 ...) x (i0 x i1 ...)
        """
        ## contract and reorder to sequential ordering
        # ax_inds = super()._transpose_parallel_to_sequential_inds(axes)
        # tens_inds = super().get_tensor_inds(axes)  # seqeuntial
        # axT = [ax_inds[x] for x in tens_inds]
        # print('axis_map mpo to operator', cls, axT)

        out_axes = axes

        if ax_select is not None and len(ax_select) > 0:
            mpo = mpo.copy()
            for ax, select_ind in ax_select.items():
                if ax in axes:
                    if isinstance(select_ind, tuple):
                        select_u, select_d = select_ind
                    else:
                        select_u = select_d = select_ind
                    select_mps_u = ax.get_select_elems_mps([select_u], site_tag_id=mpo.site_tag_id,
                                                           site_ind_id=mpo.upper_ind_id)
                    select_mps_d = ax.get_select_elems_mps([select_d], site_tag_id=mpo.site_tag_id,
                                                           site_ind_id=mpo.lower_ind_id)
                    mps_inds = cls.get_inds_in_axis(axes, ax)
                    helper.renumber_mps(select_mps_u, list(range(select_mps_u.L)), mps_inds, inplace=True)
                    helper.renumber_mps(select_mps_d, list(range(select_mps_d.L)), mps_inds, inplace=True)
                    mpo.add(select_mps_u)
                    mpo.add(select_mps_d)

            out_axes = [ax for ax in axes if ax not in ax_select]

        ax_inds = []
        for ax in out_axes:
            ax_inds += cls.get_inds_in_axis(axes, ax)
        tens_inds = super().get_tensor_inds(out_axes)
        out_idxs = [ax_inds[x] for x in tens_inds]

        tensor: qtn.Tensor = mpo.contract()  # contract all tensors
        out_inds = [mpo.upper_ind_id.format(i) for i in out_idxs] + \
                   [mpo.lower_ind_id.format(i) for i in out_idxs]
        tensor.transpose(*ancilla_left_inds, *out_inds, *ancilla_right_inds, inplace=True)

        ## need to reshape to ndim-dimensional tensor
        out_data = tensor.data.reshape(ancilla_left + tuple([ax.npts for ax in out_axes]) * 2 + ancilla_right)

        for i in range(len(out_axes)):
            ax = out_axes[i]
            out_data = ax.map.inverse_transform_operator(out_data, axis1=len(ancilla_left) + i,
                                                         axis2=len(ancilla_left) + len(out_axes) + i)

        # if len(ancilla_left) + len(ancilla_right) > 0:
        #     ax_inds = tuple([f'o({ax})' for ax in axes] + [f'i({ax})' for ax in axes])
        #     out_tens = qtn.Tensor(data=out_data * (10 ** mpo.exponent),
        #                           inds=ancilla_left_inds + ax_inds + ancilla_right_inds)
        #     return out_tens
        # else:
        #     return out_data * (10 ** mpo.exponent)
        return out_data * (10 ** mpo.exponent)


    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    @classmethod
    def make_mps_ndim(cls, axes, mps_1d_dict):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            if MPS not defined along that dimension, use a ones vector (constant along that dimension)
        """
        Ls = tuple([ax_.L for ax_ in axes])
        ndim = len(axes)
        Lmax = max(Ls)

        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]

        if ndim == 1:
            return ref_mps

        new_mps = super().make_mps_ndim(axes,mps_1d_dict)  ### technically a TN object
        new_mps.drop_tags([f'dim_{ax}' for ax in axes])
        new_exponent = new_mps.exponent

        site_tag_id = ref_mps.site_tag_id
        site_ind_id = ref_mps.site_ind_id

        ## contract dims together
        if Lmax == 1:
            new_tens = new_mps.contract(ref_mps.site_tag_id.format(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_mps = qtn.TensorNetwork([new_tens])
        else:
            for i in range(Lmax):
                new_mps.contract(ref_mps.site_tag_id.format(i), inplace=True)
            new_mps.fuse_multibonds(inplace=True)

        ## split tensors
        new_ind = 0
        tens_list = []
        for i in range(Lmax):
            tens = new_mps.select_tensors(site_tag_id.format(i))[0].copy()
            active_axes = [ax_ for ax_ in axes if ax_.L > i]
            for ax in active_axes[:-1]:
                left_bond = () if len(tens_list) == 0 else tuple(tens.bonds(tens_list[-1]))
                tensL, tens = tens.split(left_inds=left_bond + (f'i({i}),{ax}',),
                                         absorb='right', ltags=f'dim_{ax}',
                                         cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)

                tensL.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
                tensL.reindex({f'i({i}),{ax}': site_ind_id.format(new_ind)}, inplace=True)
                tens_list += [tensL.copy()]
                new_ind += 1

            tens.add_tag(f'dim_{active_axes[-1]}')
            tens.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
            tens.reindex({f'i({i}),{active_axes[-1]}': site_ind_id.format(new_ind)}, inplace=True)
            tens_list += [tens.copy()]
            new_ind += 1


        new_mps = qtn.TensorNetwork(tens_list)
        new_mps.exponent = new_exponent
        new_mps.view_like(ref_mps, L=sum(Ls), inplace=True)
        return new_mps


    @classmethod
    def make_mpx_ndim(cls, axes, mps_1d_dict):
        """ combine 1-D MPSs into K-dimensional MPO
            if MPS not defined along that dimension, pad with the identity MPO
        """
        Ls = tuple([ax_.L for ax_ in axes])
        ndim = len(axes)
        Lmax = max(Ls)

        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]
        # out_axIDs = [axID for axID in axIDs if axID not in mps_1d_dict]
        out_axes = [ax for ax in axes if ax not in mps_1d_dict]

        if len(out_axes) == 0:
            return cls.make_mps_ndim(axes, mps_1d_dict)

        if ndim == 1:
            return ref_mps

        site_tag_id = ref_mps.site_tag_id
        lower_ind_id = ref_mps.site_ind_id
        upper_ind_id = 'o({})'

        new_mpx = super().make_mpx_ndim(axes,mps_1d_dict)  ### technically a TN object
        new_mpx.drop_tags([f'dim_{ax}' for ax in axes])
        new_exponent = new_mpx.exponent

        ## contract dims together
        if Lmax == 1:
            new_tens = new_mpx.contract(ref_mps.site_tag_id.format(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_mpx = qtn.TensorNetwork([new_tens])
        else:
            for i in range(Lmax):
                new_mpx.contract(ref_mps.site_tag_id.format(i), inplace=True)
            new_mpx.fuse_multibonds(inplace=True)

        ## split tensors
        tens_list = []
        new_ind = 0
        for i in range(Lmax):
            tens = new_mpx.select_tensors(site_tag_id.format(i))[0].copy()
            active_axes = [ax_ for ax_ in axes if ax_.L > i]
            for ax in active_axes:
                if ax not in out_axes:  ## no output leg, make a dummy one of size 1
                    tens.unfuse({f'i({i}),{ax}': (f'i({i}),{ax}', f'o({i}),{ax}')},
                                {f'i({i}),{ax}': (tens.ind_size(f'i({i}),{ax}'), 1)}, inplace=True)

            for ax in active_axes[:-1]:
                if len(tens_list) == 0:
                    right_bonds, left_bonds = tens.filter_bonds(new_mpx.select_tensors(site_tag_id.format(1))[0])
                    phys_inds = [f'i({i}),{ax}' for ax in active_axes] + [f'o({i}),{ax}' for ax in active_axes]
                    left_bond = tuple([ind for ind in left_bonds if ind not in phys_inds])
                else:
                    left_bond = tuple(tens.bonds(tens_list[-1]))
                tensL, tens = tens.split(left_inds=left_bond + (f'i({i}),{ax}', f'o({i}),{ax}'),
                                         absorb='right', ltags=f'dim_{ax}',
                                         cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)

                tensL.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
                tensL.reindex({f'i({i}),{ax}': lower_ind_id.format(new_ind),
                               f'o({i}),{ax}': upper_ind_id.format(new_ind)}, inplace=True)
                tens_list += [tensL.copy()]
                new_ind += 1

            tens.add_tag(f'dim_{active_axes[-1]}')
            tens.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
            tens.reindex({f'i({i}),{active_axes[-1]}': lower_ind_id.format(new_ind),
                          f'o({i}),{active_axes[-1]}': upper_ind_id.format(new_ind)}, inplace=True)
            tens_list += [tens.copy()]
            new_ind += 1


        new_mpx = qtn.TensorNetwork(tens_list)
        new_mpx.exponent = new_exponent

        new_mpx.view_as(qtn.MatrixProductOperator, inplace=True, site_tag_id=ref_mps.site_tag_id,
                        lower_ind_id=ref_mps.site_ind_id, upper_ind_id='o({})',
                        L=sum(Ls), cyclic=ref_mps.cyclic)
        return new_mpx


    @classmethod
    def make_mpo_ndim(cls, axes, mpo_1d_dict):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        Ls = tuple([ax_.L for ax_ in axes])
        ndim = len(axes)
        Lmax = max(Ls)

        ref_mpo = mpo_1d_dict[next(iter(mpo_1d_dict))]

        if ndim == 1:
            return ref_mpo

        new_mpo = super().make_mpo_ndim(axes, mpo_1d_dict)
        new_mpo.drop_tags([f'dim_{ax}' for ax in axes])
        new_exponent = new_mpo.exponent

        site_tag_id = ref_mpo.site_tag_id
        upper_ind_id = ref_mpo.upper_ind_id
        lower_ind_id = ref_mpo.lower_ind_id

        ## contract dims together
        if Lmax == 1:
            new_tens = new_mpo.contract(ref_mpo.site_tag_id.format(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_mpo = qtn.TensorNetwork([new_tens])
        else:
            for i in range(Lmax):
                new_mpo.contract(ref_mpo.site_tag_id.format(i), inplace=True)
            new_mpo.fuse_multibonds(inplace=True)

        ## split tensors
        new_ind = 0
        tens_list = []
        for i in range(Lmax):
            tens = new_mpo.select_tensors(site_tag_id.format(i))[0].copy()
            active_axes = [ax_ for ax_ in axes if ax_.L > i]
            for ax in active_axes[:-1]:
                # left_bond = () if len(tens_list) == 0 else tuple(tens.bonds(tens_list[-1]))
                if len(tens_list) == 0:
                    right_bonds, left_bonds = tens.filter_bonds(new_mpo.select_tensors(site_tag_id.format(1))[0])
                    phys_inds = [f'i({i}),{ax}' for ax in active_axes] + [f'o({i}),{ax}' for ax in active_axes]
                    left_bond = tuple([ind for ind in left_bonds if ind not in phys_inds])
                else:
                    left_bond = tuple(tens.bonds(tens_list[-1]))

                tensL, tens = tens.split(left_inds=left_bond + (f'i({i}),{ax}', f'o({i}),{ax}'),
                                         absorb='right', ltags=f'dim_{ax}',
                                         cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                tensL.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
                tensL.reindex({f'i({i}),{ax}': lower_ind_id.format(new_ind),
                               f'o({i}),{ax}': upper_ind_id.format(new_ind)}, inplace=True)
                tens_list += [tensL.copy()]
                new_ind += 1

            tens.add_tag(f'dim_{active_axes[-1]}')
            tens.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
            tens.reindex({f'i({i}),{active_axes[-1]}': lower_ind_id.format(new_ind),
                          f'o({i}),{active_axes[-1]}': upper_ind_id.format(new_ind)}, inplace=True)
            # if i == active_axes[-1].L:

            tens_list += [tens.copy()]
            new_ind += 1

        new_mpo = qtn.TensorNetwork(tens_list)
        new_mpo.exponent = new_exponent
        new_mpo.view_like(ref_mpo, L=sum(Ls), inplace=True)

        return new_mpo


    @classmethod
    def make_tn1D_ndim_old(cls, axes, tn1D_1d_dict, upper_ind_id='i({})', lower_ind_id='o({})', extra_ind_ids=(),
                           site_tag_id='T({})'):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        Ls = tuple([ax_.L for ax_ in axes])
        ndim = len(axes)
        Lmax = max(Ls)
        ind_ids = (upper_ind_id, lower_ind_id) + extra_ind_ids

        if ndim == 1:
            return tn1D_1d_dict[next(iter(tn1D_1d_dict))]

        new_tn = super().make_tn1D_ndim(axes, tn1D_1d_dict, upper_ind_id, lower_ind_id, extra_ind_ids,
                                        site_tag_id)
        # new_tn.drop_tags([f'dim_{ax}' for ax in axes])
        new_exponent = new_tn.exponent

        pos = 0
        for ax in axes:
            for ix in range(ax.L):
                tn_tens = new_tn.select_tensors([f'dim_{ax}', site_tag_id.format(ix)], which='all')[0]
                tn_tens.reindex({**{upper_ind_id.format(ix) + f',{ax}': upper_ind_id.format(pos),
                                        lower_ind_id.format(ix) + f',{ax}': lower_ind_id.format(pos),},
                                     **{extra_id.format(ix) + f',{ax}': extra_id.format(pos) for extra_id in extra_ind_ids}
                                    }, inplace=True)
                tn_tens.drop_tags()
                tn_tens.add_tag(site_tag_id.format(pos))

                pos += 1

        ## contract dims together
        if Lmax == 1:
            new_tens = new_tn.contract(site_tag_id.format(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_tn = qtn.TensorNetwork([new_tens])
            new_tn.exponent = new_exponent
        else:

            final_inds = {ax: cls.get_inds_in_axis(axes, ax, ax_ind=ix) for (ix, ax) in enumerate(axes)}

            from gate import Gate, apply_gates

            tot_L = np.sum([ax.L for ax in axes])
            iden_tn = qtn.MPO_identity(tot_L)

            swap_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).reshape((2,2,2,2))

            def apply_swap(tn_, start_ind, dest_ind):

                print('start ind', start_ind, 'dest ind', dest_ind)

                if start_ind == dest_ind:
                    return tn_

                assert(start_ind > dest_ind), f'start ind should be less than dest ind {start_ind}, {dest_ind}'

                swaps = []
                for ix in range(start_ind, dest_ind + 1, -1):
                    swaps += [Gate((ix, ix - 1), swap_mat)]

                helper.canonize(tn_, i = start_ind)
                tn_ = apply_gates(tn_, swaps, )
                return tn_

            tmp_start = 0
            for i in range(Lmax):
                ax_lens = [ax.L for ax in axes]
                ax_count = 0
                for ix, ax in enumerate(axes):
                    try:
                        ind2 = final_inds[ax][i]
                        ind1 = tmp_start + np.sum([max(0, xl - i) for xl in ax_lens[:ix]], dtype=int)

                        iden_tn = apply_swap(iden_tn, ind1, ind2)
                        ax_count += 1
                    except IndexError:
                        pass

                tmp_start += ax_count

            ## apply swap gates to each leg
            helper.compress(iden_tn)
            print('iden tn', iden_tn.max_bond())
            iden_tn.distribute_exponent()
            id_upper = iden_tn.upper_ind_id
            id_lower = iden_tn.lower_ind_id
            rand_upper = 'U{}' # qtn.rand_uuid() + '{}'
            rand_lower = 'L{}' # qtn.rand_uuid() + '{}'
            # rand_extra = [qtn.rand_uuid() + '{}' for _ in range(len(extra_ind_ids))]

            for leg in range(len(extra_ind_ids) + 2):

                iden1 = iden_tn.copy()
                iden1.mangle_inner_()

                # print('new tn', new_tn)

                prev_tens = []
                tens_list = []
                left_inds = []
                for ix in range(tot_L):
                    # print('new tn', new_tn.select_tensors([site_tag_id.format(ix)]))
                    # tens_tn1D = new_tn[ix]
                    tens_tn1D = new_tn.select_tensors([site_tag_id.format(ix)])[0]
                    tens_u = iden1[ix]

                    # tens_tn1D.reindex({**{upper_ind_id.format(ix): rand_upper.format(ix),
                    #                       lower_ind_id.format(ix): rand_lower.format(ix),},
                    #                    **{extra_id.format(ix): rand_ex.format(ix)
                    #                       for extra_id, rand_ex in zip(extra_ind_ids, rand_extra)}},
                    #                    inplace=True)

                    if leg == 0:
                        tens_tn1D.reindex({upper_ind_id.format(ix): rand_upper.format(ix)}, inplace=True)
                        tens_u.reindex({id_upper.format(ix): upper_ind_id.format(ix),
                                        id_lower.format(ix): rand_upper.format(ix),},
                                       inplace=True)
                    elif leg == 1:
                        tens_tn1D.reindex({lower_ind_id.format(ix): rand_lower.format(ix)}, inplace=True)
                        tens_u.reindex({id_upper.format(ix): lower_ind_id.format(ix),
                                        id_lower.format(ix): rand_lower.format(ix),},
                                       inplace=True)
                    elif leg > 2:
                        extra_ind_id = extra_ind_ids[leg-2]
                        rand_extra = 'X{}'
                        tens_tn1D.reindex({extra_ind_id.format(ix): rand_extra.format(ix)}, inplace=True)
                        tens_u.reindex({id_upper.format(ix): extra_ind_id.format(ix),
                                        id_lower.format(ix): rand_extra.format(ix)}, inplace=True)

                    new_tens = qtn.tensor_contract(tens_tn1D, tens_u, *prev_tens)

                    left_inds += [upper_ind_id.format(ix), lower_ind_id.format(ix),
                                  *[ex_id.format(ix) for ex_id in extra_ind_ids]]
                    TL, TR = qtn.tensor_split(new_tens, left_inds=left_inds, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                    TL.drop_tags()
                    TR.drop_tags()
                    TL.add_tag(site_tag_id.format(ix))

                    print('ix', ix, new_tens.shape, TR.shape)
                    tens_list += [TL]
                    prev_tens = [TR]
                    left_inds = [next(iter(qtn.bonds(TL, TR)))]

                new_tn = qtn.TensorNetwork(tens_list)

            new_tn.fuse_multibonds(inplace=True)
            new_tn.exponent = new_exponent

            # ### after contraction, do QR and then contract R with the next set of tensors
            # ### kind of like in zipup algorithm
            # left_inds = []
            # for i in range(Lmax):
            #     new_tn.contract(site_tag_id.format(i), inplace=True)
            #
            #     if i < Lmax - 1:
            #         tens = new_tn.select_tensors(site_tag_id.format(i))[0]
            #         for ax in axes:
            #             ax_str = ',' + str(ax)
            #             left_inds += [upper_ind_id.format(i) + ax_str, lower_ind_id.format(i) + ax_str] \
            #                          + [ind.format(i) + ax_str for ind in extra_ind_ids]
            #         bond_ind = qtn.rand_uuid()
            #         tensL, tensR = qtn.tensor_split(tens, left_inds, absorb='right', bond_ind=bond_ind,
            #                                         cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
            #         tens.modify(data=tensL.data, inds=tensL.inds, tags=(site_tag_id.format(i),))
            #         tensR.modify(tags=(site_tag_id.format(i + 1)))
            #         new_tn.add(tensR)
            #         left_inds = [bond_ind]
            #
            # new_tn.fuse_multibonds(inplace=True)

        # ## split tensors
        # new_ind = 0
        # tens_list = []
        # for i in range(Lmax):
        #     tens = new_tn.select_tensors(site_tag_id.format(i))[0].copy()
        #     active_axes = [ax_ for ax_ in axes if ax_.L > i]
        #
        #     for ax in active_axes[:-1]:
        #
        #         inds = ind_ids
        #         # if ax in tn1D_1d_dict:    # is general tensor
        #         #     inds = ind_ids
        #         # else:                       # is identity MPO
        #         #     inds = (upper_ind_id, lower_ind_id)
        #
        #         left_bond = () if len(tens_list) == 0 else tuple(tens.bonds(tens_list[-1]))
        #         tensL, tens = tens.split(left_inds=left_bond + tuple([iid.format(i) + f',{ax}' for iid in inds]),
        #                                  absorb='right', ltags=f'dim_{ax}',
        #                                  cutoff=1.0E-30, cutoff_mode='rsum2')
        #
        #         if ax in tn1D_1d_dict:  # is general tensor
        #             inds = ind_ids
        #         else:  # is identity MPO
        #             inds = (upper_ind_id, lower_ind_id)
        #         tensL.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
        #         tensL.reindex({iid.format(i) + f',{ax}': iid.format(new_ind) for iid in inds}, inplace=True)
        #         tens_list += [tensL.copy()]
        #         new_ind += 1
        #
        #     tens.add_tag(f'dim_{active_axes[-1]}')
        #     if active_axes[-1] in tn1D_1d_dict:  # is general tensor
        #         inds = ind_ids
        #     else:  # is identity MPO
        #         inds = (upper_ind_id, lower_ind_id)
        #     tens.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
        #     tens.reindex({iid.format(i) + f',{active_axes[-1]}': iid.format(new_ind) for iid in inds}, inplace=True)
        #     tens_list += [tens.copy()]
        #     new_ind += 1

        # new_tn = qtn.TensorNetwork(tens_list)
        # new_tn.exponent = new_exponent
        # new_tn.view_as(qtn.TensorNetwork1D, inplace=True, L=new_tn.num_tensors, site_tag_id=site_tag_id)
        new_tn.view_as(MatrixProductTensor, inplace=True, L=new_tn.num_tensors, cyclic=False,
                       site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                       extra_ind_ids=extra_ind_ids)
        return new_tn

    @classmethod
    def make_tn1D_ndim(cls, axes, tn1D_1d_dict, upper_ind_id='i({})', lower_ind_id='o({})', extra_ind_ids=(),
                       site_tag_id='T({})'):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        Ls = tuple([ax_.L for ax_ in axes])
        ndim = len(axes)
        Lmax = max(Ls)
        ind_ids = (upper_ind_id, lower_ind_id) + extra_ind_ids

        if ndim == 1:
            return tn1D_1d_dict[next(iter(tn1D_1d_dict))]

        new_tn = super().make_tn1D_ndim(axes, tn1D_1d_dict, upper_ind_id, lower_ind_id, extra_ind_ids,
                                        site_tag_id)
        # new_tn.drop_tags([f'dim_{ax}' for ax in axes])
        new_exponent = new_tn.exponent

        ## contract dims together
        if Lmax == 1:
            new_tens = new_tn.contract(site_tag_id.format(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_tn = qtn.TensorNetwork([new_tens])
            new_tn.exponent = new_exponent
        else:

            # ## contract like sites (grouped)
            # for i in range(Lmax):
            #     new_tn.contract(site_tag_id.format(i), inplace=True)
            # new_tn.fuse_multibonds(inplace=True)

            tn = []
            prev_tens = []
            left_inds = []
            for i in range(Lmax):  ## assumes original op probably in right canonical form? (elemental_multiply_tn)
                active_axes = [ax for ax in axes if ax.L > i]
                select_tens = new_tn.select_tensors(site_tag_id.format(i))
                tens = qtn.tensor_contract(*select_tens, *prev_tens)
                tens.drop_tags()
                if i < Lmax - 1:
                    left_inds += [upper_ind_id.format(i) + f',{ax}' for ax in active_axes]
                    left_inds += [lower_ind_id.format(i) + f',{ax}' for ax in active_axes]
                    for xtra_id in extra_ind_ids:
                        left_inds += [xtra_id.format(i) + f',{ax}' for ax in active_axes]
                    t1, t2 = qtn.tensor_split(tens, left_inds, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE, absorb='right',
                                              ltags=site_tag_id.format(i))

                    left_inds = [next(iter(t1.bonds(t2)))]
                    prev_tens = [t2]
                    tn += [t1]
                else:
                    tens.add_tag(site_tag_id.format(i))
                    tn += [tens]

            exponent = new_tn.exponent
            new_tn = qtn.TensorNetwork(tn)
            new_tn.exponent = exponent

            new_tn.view_as(MatrixProductTensor, inplace=True, L=new_tn.num_tensors, cyclic=False,
                           site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                           extra_ind_ids=extra_ind_ids)

            helper.compress(new_tn, compress_opts={'form': 'right'}, canonize=False)

            ## split tensors
            new_ind = 0
            tens_list = []
            for i in range(Lmax):
                tens = new_tn.select_tensors(site_tag_id.format(i))[0].copy()
                active_axes = [ax_ for ax_ in axes if ax_.L > i]

                for ax in active_axes[:-1]:

                    inds = ind_ids
                    # if ax in tn1D_1d_dict:    # is general tensor
                    #     inds = ind_ids
                    # else:                       # is identity MPO
                    #     inds = (upper_ind_id, lower_ind_id)

                    left_bond = () if len(tens_list) == 0 else tuple(tens.bonds(tens_list[-1]))
                    tensL, tens = tens.split(left_inds=left_bond + tuple([iid.format(i) + f',{ax}' for iid in inds]),
                                             absorb='right', ltags=f'dim_{ax}',
                                             cutoff=1.0E-30, cutoff_mode='rsum2')

                    if ax in tn1D_1d_dict:  # is general tensor
                        inds = ind_ids
                    else:  # is identity MPO
                        inds = (upper_ind_id, lower_ind_id)
                    tensL.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
                    tensL.reindex({iid.format(i) + f',{ax}': iid.format(new_ind) for iid in inds}, inplace=True)
                    tens_list += [tensL.copy()]
                    new_ind += 1

                tens.add_tag(f'dim_{active_axes[-1]}')
                if active_axes[-1] in tn1D_1d_dict:  # is general tensor
                    inds = ind_ids
                else:  # is identity MPO
                    inds = (upper_ind_id, lower_ind_id)
                tens.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
                tens.reindex({iid.format(i) + f',{active_axes[-1]}': iid.format(new_ind) for iid in inds}, inplace=True)
                tens_list += [tens.copy()]
                new_ind += 1

            new_tn = qtn.TensorNetwork(tens_list)
            new_tn.exponent = new_exponent

        # new_tn.view_as(qtn.TensorNetwork1D, inplace=True, L=new_tn.num_tensors, site_tag_id=site_tag_id)
        new_tn.view_as(MatrixProductTensor, inplace=True, L=new_tn.num_tensors, cyclic=False,
                       site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                       extra_ind_ids=extra_ind_ids)
        return new_tn


    @classmethod
    # @profile
    def pad_mps_to_grid(cls, axes, scalar_field_mps, mps_axes):
        """ pad scalar_field_mps, which exists on axes labeled by axIDs
            in such that they exist in (target_ndim)-dimensional space
            fields are constant along all dimensions not specified by self.axes
            need either new_grid or target_ndim
        """
        pad_axes = [ax for ax in axes if ax not in mps_axes]
        tot_L = np.sum([ax.L for ax in axes])
        max_L = np.max([ax.L for ax in axes])

        site_tag_id = scalar_field_mps.site_tag_id
        site_ind_id = scalar_field_mps.site_ind_id

        old_ax_inds = {ax: cls.get_inds_in_axis(tuple(mps_axes), ax) for ax in mps_axes}
        new_ax_inds = {ax: cls.get_inds_in_axis(tuple(axes), ax) for ax in axes}

        iden_mps_dict = {ax: ax.get_iden_mps() for ax in pad_axes}

        new_mps_tens = []
        for i in range(max_L):

            # left_inds, right_inds = [], []
            possible_lefts, possible_rights, new_is = [], [], []
            tens_list = []
            for ax in axes:

                if i >= ax.L:
                    continue

                if ax in mps_axes:
                    old_ix, new_ix = old_ax_inds[ax][i], new_ax_inds[ax][i]
                    old_tag_id = site_tag_id.format(old_ix)
                    new_tag_id = site_tag_id.format(new_ix)
                    old_ind_id = site_ind_id.format(old_ix)
                    new_ind_id = site_ind_id.format(new_ix)
                    tens = scalar_field_mps.select_tensors(old_tag_id)[0].copy()
                    tens.reindex({old_ind_id: new_ind_id}, inplace=True)
                    # tens.drop_tags(tags=[old_tag_id])
                    # tens.add_tag(new_tag_id)
                    tens_list += [tens]

                    new_is += [new_ix]
                    if i < ax.L:
                        right, left_and_phys = scalar_field_mps[old_ix].filter_bonds(scalar_field_mps[old_ix + 1])
                        left = [x for x in left_and_phys if x != old_ind_id]
                    else:
                        left, right_and_phys = scalar_field_mps[old_ix].filter_bonds(scalar_field_mps[old_ix - 1])
                        right = [x for x in right_and_phys if x != old_ind_id]
                    possible_lefts += [*left]
                    possible_rights += [*right]
                else:
                    ix = new_ax_inds[ax][i]
                    ones = iden_mps_dict[ax][i].squeeze()
                    tens = qtn.Tensor(data = ones.data,
                                      inds = (site_ind_id.format(ix),),
                                      # tags = (site_tag_id.format(ix),)
                                      )
                    tens_list += [tens]
                    new_is += [ix]

            tens_group = qtn.TensorNetwork(tens_list)
            new_ind_ids = [site_ind_id.format(ix) for ix in new_is]
            lr_inds = [ind_ for ind_ in tens_group.outer_inds() if ind_ not in new_ind_ids]
            left_inds = list( set(lr_inds).intersection(set(possible_lefts)) )
            right_inds = list( set(lr_inds).intersection(set(possible_rights)) )

            new_tens_list = []
            new_tens = qtn.tensor_contract(*tens_list)
            new_tens.drop_tags()
            new_is = np.sort(new_is)   # increasing i
            for ix in new_is[:-1]:
                left = (*left_inds, site_ind_id.format(ix))
                t1, new_tens = qtn.tensor_split(new_tens, left_inds=left, absorb='right')
                t1.add_tag(site_tag_id.format(ix))
                left_inds = t1.bonds(new_tens)
                new_tens_list += [t1]
            new_tens.add_tag(site_tag_id.format(new_is[-1]))
            new_tens_list += [new_tens]

            new_mps_tens += new_tens_list

        new_mps = qtn.TensorNetwork(new_mps_tens)
        new_mps.view_as(qtn.MatrixProductState, inplace=True, cyclic=False, L=tot_L,
                        site_ind_id=site_ind_id, site_tag_id=site_tag_id)
        new_mps.exponent = scalar_field_mps.exponent

        return new_mps


    # @classmethod
    # def pad_mps_to_grid(cls, axes, scalar_field_mps, mps_axes):
    #     """ pad scalar_field_mps, which exists on axes labeled by axIDs
    #         in such that they exist in (target_ndim)-dimensional space
    #         fields are constant along all dimensions not specified by self.axes
    #         need either new_grid or target_ndim
    #     """
    #     pad_axes = [ax for ax in axes if ax not in mps_axes]
    #     tot_L = np.sum([ax.L for ax in axes])
    #     # ones = np.ones([ax.q for ax in pad_axes])
    #
    #     site_tag_id = scalar_field_mps.site_tag_id
    #     site_ind_id = scalar_field_mps.site_ind_id
    #
    #     new_mps = scalar_field_mps.copy()
    #     old_ax_inds, new_ax_inds = [], []
    #     for dim in range(len(mps_axes)):
    #         ax = mps_axes[dim]
    #         old_ax_inds += cls.get_inds_in_axis(tuple(mps_axes), ax, ax_ind=dim)
    #         new_ax_inds += cls.get_inds_in_axis(tuple(axes), ax)
    #
    #     new_mps.retag({site_tag_id.format(old_i): site_tag_id.format(new_i)
    #                    for old_i, new_i in zip(old_ax_inds,new_ax_inds)}, inplace=True)
    #     new_mps.reindex({site_ind_id.format(old_i): site_ind_id.format(new_i)
    #                      for old_i, new_i in zip(old_ax_inds,new_ax_inds)}, inplace=True)
    #
    #     ## tag for finding matching site_tag_id for arbitrary integer label
    #     search_tag = site_tag_id
    #     search_tag = re.sub('\)', '\)', search_tag)
    #     search_tag = re.sub('\(', '\(', search_tag)
    #     search_tag = search_tag.format('[\d]+')
    #
    #     new_inds = []
    #     new_qs = {}
    #     for ax in pad_axes:   # will be in increasing dim order
    #         new_inds += cls.get_inds_in_axis(tuple(axes), ax)
    #         new_qs.update({idx: ax for idx in new_inds})
    #
    #     new_inds = np.sort(new_inds)
    #     for i in new_inds:      # need to add this missing tensor into mps, sorted
    #         ax = new_qs[i]
    #         try:
    #             if i == 0:  raise KeyError
    #             tensL = new_mps.select_tensors(site_tag_id.format(i - 1))[0]
    #         except KeyError:
    #             tensL = None
    #         try:
    #             if i == tot_L - 1:  raise KeyError
    #             tensR = new_mps.select_tensors(site_tag_id.format(i + 1))[0]
    #         except KeyError:
    #             tensR = None
    #
    #         if tensL is None and tensR is None:  ## must be i=0
    #             new_mps.add(qtn.Tensor(data=np.ones((ax.q,)),
    #                                    inds=(site_ind_id.format(i),),
    #                                    tags=(site_tag_id.format(i),)))
    #         elif tensR is None:  ## no immediate existing neighbor to the right
    #             ## find bond that needs to be inserted into
    #             tagL = site_tag_id.format(i - 1)
    #             neighbors = new_mps.select_neighbors(tagL)
    #             tagR, indR = None, None
    #             for t in neighbors:
    #                 for t_tag in t.tags:
    #                     match = re.search(search_tag, t_tag)    # f
    #                     try:
    #                         tagR = match[0]
    #                         indR = int(re.search('[\d]+', tagR)[0])
    #                         if indR < i:  raise IndexError
    #                         break
    #                     except(IndexError, TypeError):
    #                         indR = None
    #                         tagR = None     # was to the right of tens
    #
    #                 if tagR is not None:
    #                     break
    #
    #             if tagR is not None:
    #                 tensR = new_mps.select_tensors(tagR)[0]
    #                 shared_ind = tuple(tensL.bonds(tensR))[0]  # should only be one bond
    #                 ind_size = tensL.ind_size(shared_ind)
    #
    #                 ## change virtual bond names
    #                 tensL.reindex({shared_ind: f'_h{i - 1}_'}, inplace=True)
    #                 tensR.reindex({shared_ind: f'_h{i}_'}, inplace=True)
    #                 # tensR.reindex({shared_ind: f'_h{indR - 1}_'}, inplace=True)
    #
    #                 one_id = np.einsum('ij,k->ijk', np.eye(ind_size), np.ones((ax.q,)))
    #                 new_mps.add(qtn.Tensor(data=one_id,
    #                                        inds=(f'_h{i - 1}_', f'_h{i}_', site_ind_id.format(i)),
    #                                        tags=(site_tag_id.format(i),)))
    #             else:  ## add ind to tagL (was at the end of the mps)
    #                 if f'_h{i - 1}_' not in tensL.inds:
    #                     tensL.new_ind(f'_h{i - 1}_', axis=1)
    #
    #                 new_mps.add(qtn.Tensor(data=np.ones((ax.q,)).reshape(1, ax.q),
    #                                        inds=(f'_h{i - 1}_', site_ind_id.format(i)),
    #                                        tags=(site_tag_id.format(i),)))
    #
    #         elif tensL is None:     ## no immediate neighbor to the left
    #             ## find bond that needs to be inserted into
    #             tagR = site_tag_id.format(i + 1)
    #             neighbors = new_mps.select_neighbors(tagR)
    #             tagL, indL = None, None
    #             for t in neighbors:
    #                 for t_tag in t.tags:
    #                     match = re.search(search_tag, t_tag)  # f
    #                     try:
    #                         tagL = match[0]
    #                         indL = int(re.search('[\d]+', tagL)[0])
    #                         if indL > i:  raise IndexError
    #                         break
    #                     except(IndexError, TypeError):
    #                         indL = None
    #                         tagL = None  # was to the right of tens
    #
    #                 if tagL is not None:
    #                     break
    #
    #             if tagL is not None:    ## tensors exist to the left
    #                 tensL = new_mps.select_tensors(tagL)[0]
    #                 shared_ind = tuple(tensR.bonds(tensL))[0]  # should only be one bond
    #                 ind_size = tensR.ind_size(shared_ind)
    #
    #                 ## change virtual bond names
    #                 tensL.reindex({shared_ind: f'_h{i - 1}_'}, inplace=True)
    #                 tensR.reindex({shared_ind: f'_h{i}_'}, inplace=True)
    #
    #                 one_id = np.einsum('ij,k->ijk', np.eye(ind_size), np.ones((ax.q,)))
    #                 new_mps.add(qtn.Tensor(data=one_id,
    #                                        inds=(f'_h{i - 1}_', f'_h{i}_', site_ind_id.format(i)),
    #                                        tags=(site_tag_id.format(i),)))
    #
    #             else:  ## add ind to tagL (was at the end of the mps)
    #                 tensR.new_ind(f'_h{i}_', axis=1)
    #                 new_mps.add(qtn.Tensor(data=np.ones((ax.q,)).reshape(1, ax.q),
    #                                        inds=(f'_h{i}_', site_ind_id.format(i)),
    #                                        tags=(site_tag_id.format(i),)))
    #
    #
    #         else:  ## right neighbors exist
    #             try:  ## need to insert new tens along bond
    #                 shared_ind = tuple(tensL.bonds(tensR))[0]
    #                 ind_size = tensL.ind_size(shared_ind)
    #                 tensL.reindex({shared_ind: f'_h{i - 1}_'}, inplace=True)
    #                 tensR.reindex({shared_ind: f'_h{i}_'}, inplace=True)
    #             except IndexError:  ## virtual bonds should already be indexed by _h{i}_
    #                 try:
    #                     ind_size = tensL.ind_size(f'_h{i - 1}_')
    #                 except ValueError:   ## right end of mps, need to insert ind of size 1
    #                     ind_size = 1
    #                     tensL.new_ind(f'_h{i - 1}_', ind_size)
    #                 tensR.new_ind(f'_h{i}_', ind_size)
    #
    #             one_id = np.einsum('ij,k->ijk', np.eye(ind_size), np.ones((ax.q,)))
    #             new_mps.add(qtn.Tensor(data=one_id,
    #                                    inds=(f'_h{i - 1}_', f'_h{i}_', site_ind_id.format(i)),
    #                                    tags=(site_tag_id.format(i),)))
    #
    #     new_mps = new_mps.view_like(scalar_field_mps, L=new_mps.num_tensors)
    #     # new_mps.exponent = scalar_field_mps.exponent
    #     return new_mps

    @classmethod
    # @profile
    def pad_mpo_to_grid(cls, axes, scalar_field_mpo, mpo_axes):
        """ pad scalar_field_mps, which exists on axes labeled by axIDs
            in such that they exist in (target_ndim)-dimensional space
            fields are constant along all dimensions not specified by self.axes
            need either new_grid or target_ndim
        """
        pad_axes = [ax for ax in axes if ax not in mpo_axes]
        tot_L = np.sum([ax.L for ax in axes])
        max_L = np.max([ax.L for ax in axes])

        site_tag_id = scalar_field_mpo.site_tag_id
        # site_ind_id = scalar_field_mps.site_ind_id
        upper_ind_id = scalar_field_mpo.upper_ind_id
        lower_ind_id = scalar_field_mpo.lower_ind_id

        old_ax_inds = {ax: cls.get_inds_in_axis(tuple(mpo_axes), ax) for ax in mpo_axes}
        new_ax_inds = {ax: cls.get_inds_in_axis(tuple(axes), ax) for ax in axes}

        new_mps_tens = []
        for i in range(max_L):

            # left_inds, right_inds = [], []
            possible_lefts, possible_rights, new_is = [], [], []
            tens_list = []
            for ax in axes:

                if i >= ax.L:
                    continue

                if ax in mpo_axes:
                    old_ix, new_ix = old_ax_inds[ax][i], new_ax_inds[ax][i]
                    old_tag_id = site_tag_id.format(old_ix)
                    new_tag_id = site_tag_id.format(new_ix)
                    old_upper_id = upper_ind_id.format(old_ix)
                    new_upper_id = upper_ind_id.format(new_ix)
                    old_lower_id = lower_ind_id.format(old_ix)
                    new_lower_id = lower_ind_id.format(new_ix)
                    tens = scalar_field_mpo.select_tensors(old_tag_id)[0].copy()
                    tens.reindex({old_upper_id: new_upper_id, old_lower_id: new_lower_id}, inplace=True)
                    # tens.drop_tags(tags=[old_tag_id])
                    # tens.add_tag(new_tag_id)
                    tens_list += [tens]

                    new_is += [new_ix]
                    if i < ax.L:
                        right, left_and_phys = scalar_field_mpo[old_ix].filter_bonds(scalar_field_mpo[old_ix + 1])
                        left = [x for x in left_and_phys if x not in [old_upper_id, old_lower_id]]
                    else:
                        left, right_and_phys = scalar_field_mpo[old_ix].filter_bonds(scalar_field_mpo[old_ix - 1])
                        right = [x for x in right_and_phys if x not in [old_upper_id, old_lower_id]]
                    possible_lefts += [*left]
                    possible_rights += [*right]
                else:
                    ix = new_ax_inds[ax][i]
                    tens = qtn.Tensor(data=np.eye(ax.q),
                                      inds=(upper_ind_id.format(ix), lower_ind_id.format(ix)),
                                      # tags = (site_tag_id.format(ix),)
                                      )
                    tens_list += [tens]
                    new_is += [ix]

            tens_group = qtn.TensorNetwork(tens_list)
            new_ind_ids = [upper_ind_id.format(ix) for ix in new_is] + [lower_ind_id.format(ix) for ix in new_is]
            lr_inds = [ind_ for ind_ in tens_group.outer_inds() if ind_ not in new_ind_ids]
            left_inds = list(set(lr_inds).intersection(set(possible_lefts)))
            right_inds = list(set(lr_inds).intersection(set(possible_rights)))

            new_tens_list = []
            new_tens = qtn.tensor_contract(*tens_list)
            new_tens.drop_tags()
            new_is = np.sort(new_is)  # increasing i
            for ix in new_is[:-1]:
                left = (*left_inds, upper_ind_id.format(ix), lower_ind_id.format(ix))
                t1, new_tens = qtn.tensor_split(new_tens, left_inds=left, absorb='right')
                t1.add_tag(site_tag_id.format(ix))
                left_inds = t1.bonds(new_tens)
                new_tens_list += [t1]
            new_tens.add_tag(site_tag_id.format(new_is[-1]))
            new_tens_list += [new_tens]

            new_mps_tens += new_tens_list

        new_mps = qtn.TensorNetwork(new_mps_tens)
        new_mps.view_as(qtn.MatrixProductOperator, inplace=True, cyclic=False, L=tot_L,
                        upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)
        new_mps.exponent = scalar_field_mpo.exponent

        return new_mps


    # @classmethod
    # def pad_mpo_to_grid(cls, axes, scalar_field_mpo, mpo_axes):
    #     """ pad scalar_field_mpo, which exists on axes labeled by axIDs
    #         in such that they exist in (target_ndim)-dimensional space
    #         fields are constant along all dimensions not specified by self.axes
    #         need either new_grid or target_ndim
    #     """
    #     pad_axes = [ax for ax in axes if ax not in mpo_axes]
    #     tot_L = np.sum([ax.L for ax in axes])
    #     # print('pad axes', pad_axes)
    #     # ones = np.ones([ax.q for ax in pad_axes])
    #
    #     site_tag_id = scalar_field_mpo.site_tag_id
    #     upper_ind_id = scalar_field_mpo.upper_ind_id
    #     lower_ind_id = scalar_field_mpo.lower_ind_id
    #
    #     new_mpo = scalar_field_mpo.copy()
    #     old_ax_inds, new_ax_inds = [], []
    #     for dim in range(len(mpo_axes)):
    #         ax = mpo_axes[dim]
    #         old_ax_inds += cls.get_inds_in_axis(tuple(mpo_axes), ax, ax_ind=dim)
    #         new_ax_inds += cls.get_inds_in_axis(tuple(axes), ax)
    #
    #     new_mpo.retag({site_tag_id.format(old_i): site_tag_id.format(new_i)
    #                    for old_i, new_i in zip(old_ax_inds, new_ax_inds)}, inplace=True)
    #     new_mpo.reindex({**{upper_ind_id.format(old_i): upper_ind_id.format(new_i)
    #                         for old_i, new_i in zip(old_ax_inds, new_ax_inds)},
    #                      **{lower_ind_id.format(old_i): lower_ind_id.format(new_i)
    #                         for old_i, new_i in zip(old_ax_inds, new_ax_inds)},}, inplace=True)
    #
    #     ## tag for finding matching site_tag_id for arbitrary integer label
    #     search_tag = site_tag_id
    #     search_tag = re.sub('\)', '\)', search_tag)
    #     search_tag = re.sub('\(', '\(', search_tag)
    #     search_tag = search_tag.format('[\d]+')
    #
    #     new_inds = []
    #     new_qs = {}
    #     for ax in pad_axes:  # will be in increasing dim order
    #         new_inds += cls.get_inds_in_axis(tuple(axes), ax)
    #         new_qs.update({idx: ax for idx in new_inds})
    #         # print('pad ax new inds',new_inds)
    #
    #     new_inds = np.sort(new_inds)
    #     for i in new_inds:  # need to add this missing tensor into mps, sorted
    #         ax = new_qs[i]
    #         try:
    #             if i == 0:  raise KeyError
    #             tensL = new_mpo.select_tensors(site_tag_id.format(i - 1))[0]
    #         except KeyError:
    #             tensL = None
    #         try:
    #             if i == tot_L - 1:  raise KeyError
    #             tensR = new_mpo.select_tensors(site_tag_id.format(i + 1))[0]
    #         except KeyError:
    #             tensR = None
    #
    #         if tensL is None and tensR is None:  ## must be i=0
    #             new_mpo.add(qtn.Tensor(data=np.eye(ax.q),
    #                                    inds=(upper_ind_id.format(i), lower_ind_id.format(i)),
    #                                    tags=(site_tag_id.format(i),)))
    #         elif tensR is None:  ## no immediate existing neighbor to the right
    #             ## find bond that needs to be inserted into
    #             tagL = site_tag_id.format(i - 1)
    #             neighbors = new_mpo.select_neighbors(tagL)
    #             tagR, indR = None, None
    #             for t in neighbors:
    #                 for t_tag in t.tags:
    #                     match = re.search(search_tag, t_tag)  # f
    #                     try:
    #                         tagR = match[0]
    #                         indR = int(re.search('[\d]+', tagR)[0])
    #                         if indR < i:  raise IndexError
    #                         break
    #                     except(IndexError, TypeError):
    #                         indR = None
    #                         tagR = None  # was to the right of tens
    #
    #                 if tagR is not None:
    #                     break
    #
    #             if tagR is not None:
    #                 tensR = new_mpo.select_tensors(tagR)[0]
    #                 shared_ind = tuple(tensL.bonds(tensR))[0]  # should only be one bond
    #                 ind_size = tensL.ind_size(shared_ind)
    #
    #                 ## change virtual bond names
    #                 tensL.reindex({shared_ind: f'_h{i - 1}_'}, inplace=True)
    #                 tensR.reindex({shared_ind: f'_h{i}_'}, inplace=True)
    #
    #                 iden_id = np.einsum('ij,kl->ijkl', np.eye(ind_size), np.eye(ax.q))
    #                 new_mpo.add(qtn.Tensor(data=iden_id,
    #                                        inds=(f'_h{i - 1}_', f'_h{i}_',
    #                                              upper_ind_id.format(i), lower_ind_id.format(i)),
    #                                        tags=(site_tag_id.format(i),)))
    #             else:  ## add ind to tagL (was at the end of the mps)
    #                 if f'_h{i - 1}_' not in tensL.inds:
    #                     tensL.new_ind(f'_h{i - 1}_', axis=1)
    #
    #                 new_mpo.add(qtn.Tensor(data=np.eye(ax.q).reshape(1, ax.q, ax.q),
    #                                        inds=(f'_h{i - 1}_', upper_ind_id.format(i), lower_ind_id.format(i)),
    #                                        tags=(site_tag_id.format(i),)))
    #
    #         elif tensL is None:  ## no immediate neighbor to the left
    #             ## find bond that needs to be inserted into
    #             tagR = site_tag_id.format(i + 1)
    #             neighbors = new_mpo.select_neighbors(tagR)
    #             tagL, indL = None, None
    #             for t in neighbors:
    #                 for t_tag in t.tags:
    #                     match = re.search(search_tag, t_tag)  # f
    #                     try:
    #                         tagL = match[0]
    #                         indL = int(re.search('[\d]+', tagL)[0])
    #                         if indL > i:  raise IndexError
    #                         break
    #                     except(IndexError, TypeError):
    #                         indL = None
    #                         tagL = None  # was to the right of tens
    #
    #                 if tagL is not None:
    #                     break
    #
    #             if tagL is not None:  ## tensors exist to the left
    #                 tensL = new_mpo.select_tensors(tagL)[0]
    #                 shared_ind = tuple(tensR.bonds(tensL))[0]  # should only be one bond
    #                 ind_size = tensR.ind_size(shared_ind)
    #
    #                 ## change virtual bond names
    #                 tensL.reindex({shared_ind: f'_h{i - 1}_'}, inplace=True)
    #                 tensR.reindex({shared_ind: f'_h{i}_'}, inplace=True)
    #
    #                 iden_id = np.einsum('ij,kl->ijkl', np.eye(ind_size), np.eye(ax.q))
    #                 new_mpo.add(qtn.Tensor(data=iden_id,
    #                                        inds=(f'_h{i - 1}_', f'_h{i}_',
    #                                              upper_ind_id.format(i), lower_ind_id.format(i)),
    #                                        tags=(site_tag_id.format(i),)))
    #
    #             else:  ## add ind to tagL (was at the end of the mps)
    #                 tensR.new_ind(f'_h{i}_', axis=1)
    #                 new_mpo.add(qtn.Tensor(data=np.eye(ax.q).reshape(1, ax.q, ax.q),
    #                                        inds=(f'_h{i}_', upper_ind_id.format(i), lower_ind_id.format(i)),
    #                                        tags=(site_tag_id.format(i),)))
    #
    #
    #         else:  ## left and right neighbors exist
    #             try:  ## need to insert new tens along bond
    #                 shared_ind = tuple(tensL.bonds(tensR))[0]
    #                 ind_size = tensL.ind_size(shared_ind)
    #                 tensL.reindex({shared_ind: f'_h{i - 1}_'}, inplace=True)
    #                 tensR.reindex({shared_ind: f'_h{i}_'}, inplace=True)
    #             except IndexError:  ## virtual bonds should already be indexed by _h{i}_
    #                 try:
    #                     ind_size = tensL.ind_size(f'_h{i - 1}_')
    #                 except ValueError:  ## right end of mps, need to insert ind of size 1
    #                     ind_size = 1
    #                     tensL.new_ind(f'_h{i - 1}_', ind_size)
    #                 tensR.new_ind(f'_h{i}_', ind_size)
    #
    #             iden_id = np.einsum('ij,kl->ijkl', np.eye(ind_size), np.eye(ax.q))
    #             new_mpo.add(qtn.Tensor(data=iden_id,
    #                                    inds=(f'_h{i - 1}_', f'_h{i}_',
    #                                          upper_ind_id.format(i), lower_ind_id.format(i)),
    #                                    tags=(site_tag_id.format(i),)))
    #
    #     new_mpo = new_mpo.view_like(scalar_field_mpo, L=new_mpo.num_tensors)
    #     # new_mps.exponent = scalar_field_mps.exponent
    #     return new_mpo



class LayoutParallelF_ConstantL(LayoutParallelF):

    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    @classmethod
    def make_mps_ndim(cls, axes, mps_1d_dict):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            if MPS not defined along that dimension, use a ones vector (constant along that dimension)
        """
        ndim = len(axes)
        L0 = axes[0].L
        # axIDs = [ax.axID for ax in axes]

        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]

        if ndim == 1:
            return ref_mps

        new_mps = super().make_mps_ndim(axes,mps_1d_dict)  ### technically a TN object
        new_mps.drop_tags([f'dim_{ax}' for ax in axes])
        new_exponent = new_mps.exponent

        site_tag_id = ref_mps.site_tag_id
        site_ind_id = ref_mps.site_ind_id

        ## contract dims together
        if L0 == 1:
            new_tens = new_mps.contract(ref_mps.site_tag(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_mps = qtn.TensorNetwork([new_tens])
        else:
            for i in range(L0):
                new_mps.contract(ref_mps.site_tag(i), inplace=True)
            new_mps.fuse_multibonds(inplace=True)

        ## split tensors
        tens_list = []
        for i in range(L0):
            tens = new_mps.select_tensors(site_tag_id.format(i))[0].copy()
            for ax in axes[:-1]:
                left_bond = () if len(tens_list) == 0 else tuple(tens.bonds(tens_list[-1]))
                tensL, tens = tens.split(left_inds=left_bond + (f'i({i}),{ax}',),
                                         absorb='right', ltags=f'dim_{ax}',
                                         cutoff=1.0E-30, cutoff_mode='rsum2')
                tens_list += [tensL.copy()]
            tens.add_tag(f'dim_{axes[-1]}')
            tens_list += [tens.copy()]

        ## reindex and retag
        new_ind = 0
        for tens_d in tens_list:
            dim = new_ind % ndim
            i = new_ind // ndim
            tens_d.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
            tens_d.reindex({f'i({i}),{axes[dim]}': site_ind_id.format(new_ind)}, inplace=True)
            new_ind += 1

        new_mps = qtn.TensorNetwork(tens_list)
        new_mps.exponent = new_exponent
        new_mps.view_like(ref_mps, L=L0 * ndim, inplace=True)
        return new_mps


    @classmethod
    def make_mpx_ndim(cls, axes, mps_1d_dict):
        """ combine 1-D MPSs into K-dimensional MPO
            if MPS not defined along that dimension, pad with the identity MPO
        """
        ndim = len(axes)
        L0 = axes[0].L
        # axIDs = [ax.axID for ax in axes]

        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]
        # out_axIDs = [axID for axID in axIDs if axID not in mps_1d_dict]
        out_axes = [ax for ax in axes if ax not in mps_1d_dict]

        if len(out_axes) == 0:
            return cls.make_mps_ndim(axes, mps_1d_dict)

        if ndim == 1:
            return ref_mps

        site_tag_id = ref_mps.site_tag_id
        lower_ind_id = ref_mps.site_ind_id
        upper_ind_id = 'o({})'

        new_mpx = super().make_mpx_ndim(axes,mps_1d_dict)  ### technically a TN object
        new_mpx.drop_tags([f'dim_{ax}' for ax in axes])
        new_exponent = new_mpx.exponent

        ## contract dims together
        if ref_mps.L == 1:
            new_tens = new_mpx.contract(ref_mps.site_tag(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_mpx = qtn.TensorNetwork([new_tens])
        else:
            for i in range(L0):
                new_mpx.contract(ref_mps.site_tag(i), inplace=True)
            new_mpx.fuse_multibonds(inplace=True)

        ## split tensors
        tens_list = []
        for i in range(L0):
            tens = new_mpx.select_tensors(site_tag_id.format(i))[0].copy()
            for ax in axes:
                if ax not in out_axes:  ## no output leg, make a dummy one of size 1
                    tens.unfuse({f'i({i}),{ax}': (f'i({i}),{ax}', f'o({i}),{ax}')},
                                {f'i({i}),{ax}': (tens.ind_size(f'i({i}),{ax}'), 1)}, inplace=True)
            for ax in axes[:-1]:
                left_bond = () if len(tens_list) == 0 else tuple(tens.bonds(tens_list[-1]))
                tensL, tens = tens.split(left_inds=left_bond + (f'i({i}),{ax}', f'o({i}),{ax}'),
                                         absorb='right', ltags=f'dim_{ax}',
                                         cutoff=1.0E-30, cutoff_mode='rsum2')
                tens_list += [tensL.copy()]
            tens.add_tag(f'dim_{axes[-1]}')
            tens_list += [tens.copy()]

        ## reindex and retag
        new_ind = 0
        for tens_d in tens_list:
            dim = new_ind % ndim
            i = new_ind // ndim
            tens_d.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
            tens_d.reindex({f'i({i}),{axes[dim]}': lower_ind_id.format(new_ind),
                            f'o({i}),{axes[dim]}': upper_ind_id.format(new_ind)}, inplace=True)
            new_ind += 1

        new_mpx = qtn.TensorNetwork(tens_list)
        new_mpx.exponent = new_exponent

        new_mpx.view_as(qtn.MatrixProductOperator, inplace=True, site_tag_id=ref_mps.site_tag_id,
                        lower_ind_id=ref_mps.site_ind_id, upper_ind_id='o({})',
                        L=L0*ndim, cyclic=ref_mps.cyclic)
        return new_mpx


    @classmethod
    def make_mpo_ndim(cls, axes, mpo_1d_dict):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        ndim = len(axes)
        L0 = axes[0].L

        ref_mpo = mpo_1d_dict[next(iter(mpo_1d_dict))]

        if ndim == 1:
            return ref_mpo

        new_mpo = super().make_mpo_ndim(axes, mpo_1d_dict)
        new_mpo.drop_tags([f'dim_{ax}' for ax in axes])
        new_exponent = new_mpo.exponent

        site_tag_id = ref_mpo.site_tag_id
        upper_ind_id = ref_mpo.upper_ind_id
        lower_ind_id = ref_mpo.lower_ind_id

        ## contract dims together
        if ref_mpo.L == 1:
            new_tens = new_mpo.contract(ref_mpo.site_tag(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_mpo = qtn.TensorNetwork([new_tens])
        else:
            for i in range(L0):
                new_mpo.contract(ref_mpo.site_tag(i), inplace=True)
            new_mpo.fuse_multibonds(inplace=True)

        ## split tensors
        tens_list = []
        for i in range(L0):
            tens = new_mpo.select_tensors(site_tag_id.format(i))[0].copy()
            for ax in axes[:-1]:
                left_bond = () if len(tens_list) == 0 else tuple(tens.bonds(tens_list[-1]))
                tensL, tens = tens.split(left_inds=left_bond + (f'i({i}),{ax}', f'o({i}),{ax}'),
                                         absorb='right', ltags=f'dim_{ax}',
                                         cutoff=1.0E-30, cutoff_mode='rsum2')
                tens_list += [tensL.copy()]
            tens.add_tag(f'dim_{axes[-1]}')
            tens_list += [tens.copy()]

        ## reindex and retag
        new_ind = 0
        for tens_d in tens_list:
            dim = new_ind % ndim
            i = new_ind // ndim
            tens_d.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
            tens_d.reindex({f'i({i}),{axes[dim]}': lower_ind_id.format(new_ind),
                            f'o({i}),{axes[dim]}': upper_ind_id.format(new_ind)}, inplace=True)
            new_ind += 1

        new_mpo = qtn.TensorNetwork(tens_list)
        new_mpo.exponent = new_exponent
        new_mpo.view_like(ref_mpo, L=L0 * ndim, inplace=True)

        return new_mpo


    @classmethod
    def make_tn1D_ndim(cls, axes, tn1D_1d_dict, upper_ind_id='i({})', lower_ind_id='o({})', extra_ind_ids=(),
                       site_tag_id='T({})'):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        ndim = len(axes)
        L0 = axes[0].L
        # axIDs = [ax.axID for ax in axes]
        ind_ids = (upper_ind_id, lower_ind_id) + extra_ind_ids

        if ndim == 1:
            return tn1D_1d_dict[next(iter(tn1D_1d_dict))]

        new_tn = super().make_tn1D_ndim(axes, tn1D_1d_dict, upper_ind_id, lower_ind_id, extra_ind_ids,
                                        site_tag_id)
        new_tn.drop_tags([f'dim_{ax}' for ax in axes])
        new_exponent = new_tn.exponent

        ## contract dims together
        if L0 == 1:
            new_tens = new_tn.contract(site_tag_id.format(0), inplace=True)
            ## in this case, new_tens return is a Tensor bc full TN is contracted
            new_tn = qtn.TensorNetwork([new_tens])
        else:
            for i in range(L0):
                new_tn.contract(site_tag_id.format(i), inplace=True)
            new_tn.fuse_multibonds(inplace=True)

        ## split tensors
        tens_list = []
        for i in range(L0):
            tens = new_tn.select_tensors(site_tag_id.format(i))[0].copy()
            for ax in axes[:-1]:

                inds = ind_ids
                # if ax in tn1D_1d_dict:    # is general tensor
                #     inds = ind_ids
                # else:                       # is identity MPO
                #     inds = (upper_ind_id, lower_ind_id)

                left_bond = () if len(tens_list) == 0 else tuple(tens.bonds(tens_list[-1]))
                tensL, tens = tens.split(left_inds=left_bond + tuple([iid.format(i) + f',{ax}' for iid in inds]),
                                         absorb='right', ltags=f'dim_{ax}',
                                         cutoff=1.0E-30, cutoff_mode='rsum2')
                tens_list += [tensL.copy()]
            tens.add_tag(f'dim_{axes[-1]}')
            tens_list += [tens.copy()]

        ## reindex and retag
        new_ind = 0
        for tens_d in tens_list:
            dim = new_ind % ndim
            i = new_ind // ndim

            if axes[dim] in tn1D_1d_dict:  # is general tensor
                inds = ind_ids
            else:  # is identity MPO
                inds = (upper_ind_id, lower_ind_id)

            tens_d.retag({site_tag_id.format(i): site_tag_id.format(new_ind)}, inplace=True)
            tens_d.reindex({iid.format(i) + f',{axes[dim]}': iid.format(new_ind) for iid in inds}, inplace=True)
            new_ind += 1

        new_tn = qtn.TensorNetwork(tens_list)
        new_tn.exponent = new_exponent
        # new_tn.view_as(qtn.TensorNetwork1D, inplace=True, L=new_tn.num_tensors, site_tag_id=site_tag_id)
        new_tn.view_as(MatrixProductTensor, inplace=True, L=new_tn.num_tensors, cyclic=False,
                       site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                       extra_ind_ids=extra_ind_ids)
        return new_tn


    @classmethod
    def pad_mps_to_grid(cls, axes, scalar_field_mps, mps_axes):
        """ pad scalar_field_mps, which exists on axes labeled by axIDs
            in such that they exist in (target_ndim)-dimensional space
            fields are constant along all dimensions not specified by self.axes
            need either new_grid or target_ndim
        """
        L0 = axes[0].L
        ndim = len(axes)

        pad_axes = tuple([ax for ax in axes if ax not in mps_axes])
        # ones = np.ones([ax.q for ax in pad_axes])

        site_tag_id = scalar_field_mps.site_tag_id
        site_ind_id = scalar_field_mps.site_ind_id

        new_mps = scalar_field_mps.copy()
        for dim in range(len(mps_axes)):
            ax = mps_axes[dim]
            old_ax_inds = cls.get_inds_in_axis(mps_axes, ax, ax_ind=dim)
            new_ax_inds = cls.get_inds_in_axis(axes, ax)
            new_mps.retag({site_tag_id.format(old_i): site_tag_id.format(new_i)
                           for old_i, new_i in zip(old_ax_inds,new_ax_inds)}, inplace=True)
            new_mps.reindex({site_ind_id.format(old_i): site_ind_id.format(new_i)
                             for old_i, new_i in zip(old_ax_inds,new_ax_inds)}, inplace=True)

        ## tag for finding matching site_tag_id for arbitrary integer label
        search_tag = site_tag_id
        search_tag = re.sub('\)', '\)', search_tag)
        search_tag = re.sub('\(', '\(', search_tag)
        search_tag = search_tag.format('[\d]+')

        for ax in pad_axes:   # will be in increasing dim order
            new_inds = cls.get_inds_in_axis(axes, ax)

            for i in new_inds:      # need to add this missing tensor into mps, sorted
                try:
                    tensL = new_mps.select_tensors(site_tag_id.format(i - 1))[0]
                except KeyError:
                    tensL = None
                try:
                    tensR = new_mps.select_tensors(site_tag_id.format(i + 1))[0]
                except KeyError:
                    tensR = None

                if tensL is None and tensR is None:  ## must be i=0
                    new_mps.add(qtn.Tensor(data=np.ones((ax.q,)).reshape(1, ax.q),
                                           inds=(f'_h{i}_', site_ind_id.format(i)),
                                           tags=(site_tag_id.format(i),)))
                elif tensR is None:  ## no immediate existing neighbor to the right
                    ## find bond that needs to be inserted into
                    tagL = site_tag_id.format(i - 1)
                    neighbors = new_mps.select_neighbors(tagL)
                    tagR, indR = None, None
                    for t in neighbors:
                        for t_tag in t.tags:
                            match = re.search(search_tag, t_tag)    # f
                            try:
                                tagR = match[0]
                                indR = int(re.search('[\d]+', tagR)[0])
                                if indR < i:  raise IndexError
                                break
                            except(IndexError, TypeError):
                                indR = None
                                tagR = None     # was to the right of tens

                        if tagR is not None:
                            break

                    if tagR is not None:
                        tensR = new_mps.select_tensors(tagR)[0]
                        shared_ind = tuple(tensL.bonds(tensR))[0]  # should only be one bond
                        ind_size = tensL.ind_size(shared_ind)

                        ## change virtual bond names
                        tensL.reindex({shared_ind: f'_h{i - 1}_'}, inplace=True)
                        tensR.reindex({shared_ind: f'_h{indR - 1}_'}, inplace=True)

                        one_id = np.einsum('ij,k->ijk', np.eye(ind_size), np.ones((ax.q,)))
                        new_mps.add(qtn.Tensor(data=one_id,
                                               inds=(f'_h{i - 1}_', f'_h{i}_', site_ind_id.format(i)),
                                               tags=(site_tag_id.format(i),)))
                    else:  ## add ind to tagL (was at the end of the mps)
                        tensL.new_ind(f'_h{i - 1}_', axis=1)
                        new_mps.add(qtn.Tensor(data=np.ones((ax.q,)).reshape(1, ax.q),
                                               inds=(f'_h{i - 1}_', site_ind_id.format(i)),
                                               tags=(site_tag_id.format(i),)))

                else:  ## left and right neighbors exist
                    try:  ## need to insert new tens along bond
                        shared_ind = tuple(tensL.bonds(tensR))[0]
                        ind_size = tensL.ind_size(shared_ind)
                        tensL.reindex({shared_ind: f'_h{i - 1}_'}, inplace=True)
                        tensR.reindex({shared_ind: f'_h{i}_'}, inplace=True)
                    except IndexError:  ## virtual bonds should already be indexed by _h{i}_
                        ind_size = tensL.ind_size(f'_h{i - 1}_')

                    one_id = np.einsum('ij,k->ijk', np.eye(ind_size), np.ones((ax.q,)))
                    new_mps.add(qtn.Tensor(data=one_id,
                                           inds=(f'_h{i - 1}_', f'_h{i}_', site_ind_id.format(i)),
                                           tags=(site_tag_id.format(i),)))

        new_mps = new_mps.view_like(scalar_field_mps, L=L0*ndim)
        # new_mps.exponent = scalar_field_mps.exponent
        return new_mps


    @classmethod
    def pad_mpo_to_grid(cls, axes, scalar_field_mpo, mpo_axes):
        """ pad scalar_field_mpo, which exists on axes labeled by axIDs
            in such that they exist in (target_ndim)-dimensional space
            fields are constant along all dimensions not specified by self.axes
            need either new_grid or target_ndim
        """
        L0 = axes[0].L
        ndim = len(axes)

        pad_axes = [ax for ax in axes if ax not in mpo_axes]
        # ones = np.ones([ax.q for ax in pad_axes])

        site_tag_id = scalar_field_mpo.site_tag_id
        upper_ind_id = scalar_field_mpo.upper_ind_id
        lower_ind_id = scalar_field_mpo.lower_ind_id

        new_mpo = scalar_field_mpo.copy()
        for dim in range(len(mpo_axes)):
            ax = mpo_axes[dim]
            old_ax_inds = cls.get_inds_in_axis(mpo_axes, ax, ax_ind=dim)
            new_ax_inds = cls.get_inds_in_axis(axes, ax)
            new_mpo.retag({site_tag_id.format(old_i): site_tag_id.format(new_i)
                           for old_i, new_i in zip(old_ax_inds, new_ax_inds)}, inplace=True)
            new_mpo.reindex({**{upper_ind_id.format(old_i): upper_ind_id.format(new_i)
                                for old_i, new_i in zip(old_ax_inds, new_ax_inds)},
                             **{lower_ind_id.format(old_i): lower_ind_id.format(new_i)
                                for old_i, new_i in zip(old_ax_inds, new_ax_inds)},}, inplace=True)

        ## tag for finding matching site_tag_id for arbitrary integer label
        search_tag = site_tag_id
        search_tag = re.sub('\)', '\)', search_tag)
        search_tag = re.sub('\(', '\(', search_tag)
        search_tag = search_tag.format('[\d]+')

        for ax in pad_axes:  # will be in increasing dim order
            new_inds = cls.get_inds_in_axis(axes, ax)

            for i in new_inds:  # need to add this missing tensor into mps, sorted
                try:
                    tensL = new_mpo.select_tensors(site_tag_id.format(i - 1))[0]
                except KeyError:
                    tensL = None
                try:
                    tensR = new_mpo.select_tensors(site_tag_id.format(i + 1))[0]
                except KeyError:
                    tensR = None

                if tensL is None and tensR is None:  ## must be i=0
                    new_mpo.add(qtn.Tensor(data=np.eye(ax.q).reshape(1, ax.q, ax.q),
                                           inds=(f'_h{i}_', upper_ind_id.format(i), lower_ind_id.format(i)),
                                           tags=(site_tag_id.format(i),)))
                elif tensR is None:  ## no immediate existing neighbor to the right
                    ## find bond that needs to be inserted into
                    tagL = site_tag_id.format(i - 1)
                    neighbors = new_mpo.select_neighbors(tagL)
                    tagR, indR = None, None
                    for t in neighbors:
                        for t_tag in t.tags:
                            match = re.search(search_tag, t_tag)  # f
                            try:
                                tagR = match[0]
                                indR = int(re.search('[\d]+', tagR)[0])
                                if indR < i:  raise IndexError
                                break
                            except(IndexError, TypeError):
                                indR = None
                                tagR = None  # was to the right of tens

                        if tagR is not None:
                            break

                    if tagR is not None:
                        tensR = new_mpo.select_tensors(tagR)[0]
                        shared_ind = tuple(tensL.bonds(tensR))[0]  # should only be one bond
                        ind_size = tensL.ind_size(shared_ind)

                        ## change virtual bond names
                        tensL.reindex({shared_ind: f'_h{i - 1}_'}, inplace=True)
                        tensR.reindex({shared_ind: f'_h{indR - 1}_'}, inplace=True)

                        iden_id = np.einsum('ij,kl->ijkl', np.eye(ind_size), np.eye(ax.q))
                        new_mpo.add(qtn.Tensor(data=iden_id,
                                               inds=(f'_h{i - 1}_', f'_h{i}_',
                                                     upper_ind_id.format(i), lower_ind_id.format(i)),
                                               tags=(site_tag_id.format(i),)))
                    else:  ## add ind to tagL (was at the end of the mps)
                        tensL.new_ind(f'_h{i - 1}_', axis=1)
                        new_mpo.add(qtn.Tensor(data=np.eye(ax.q).reshape(1, ax.q, ax.q),
                                               inds=(f'_h{i - 1}_', upper_ind_id.format(i), lower_ind_id.format(i)),
                                               tags=(site_tag_id.format(i),)))

                else:  ## left and right neighbors exist
                    try:  ## need to insert new tens along bond
                        shared_ind = tuple(tensL.bonds(tensR))[0]
                        ind_size = tensL.ind_size(shared_ind)
                        tensL.reindex({shared_ind: f'_h{i - 1}_'}, inplace=True)
                        tensR.reindex({shared_ind: f'_h{i}_'}, inplace=True)
                    except IndexError:  ## virtual bonds should already be indexed by _h{i}_
                        ind_size = tensL.ind_size(f'_h{i - 1}_')

                    iden_id = np.einsum('ij,kl->ijkl', np.eye(ind_size), np.eye(ax.q))
                    new_mpo.add(qtn.Tensor(data=iden_id,
                                           inds=(f'_h{i - 1}_', f'_h{i}_',
                                                 upper_ind_id.format(i), lower_ind_id.format(i)),
                                           tags=(site_tag_id.format(i),)))

        new_mpo = new_mpo.view_like(scalar_field_mpo, L=L0 * ndim)
        # new_mps.exponent = scalar_field_mps.exponent
        return new_mpo

