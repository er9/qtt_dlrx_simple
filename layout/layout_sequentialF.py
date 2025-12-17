import helper_quimb
from setup_.defaults import *
from quimb.tensor import MatrixProductState as MPS
from quimb.tensor import MatrixProductOperator as MPO

import helper_quimb as helper
from layout.layout import Layout

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

""" Layout_sequentialF:  sequential: n-dimensions are ordered sequentailly (x0,x1,...), (y0,y1...), ...,
                         F:  tensors corresponding to different dimensions remain factorized
"""

class LayoutSequentialF(Layout):

    @classmethod
    def L(cls, axes: Sequence['Axis']) -> int:
        return super().L(axes)
        # mpx_L = sum([ax.L for ax in axes])
        # return mpx_L

    @classmethod
    def get_inds_in_axis(cls, axes: tuple['Axis'], ax: 'Axis', ax_ind=None) -> list:
        """ get indices corresponding to tensors along a given axis (for factorized TNs)
            currently is NOT ordered.
        """
        if ax_ind is None:
            # axIDs = [ax.axID for ax in axes]
            ax_ind = axes.index(ax)
        Ls = [ax.L for ax in axes]
        return [sum(Ls[:ax_ind]) + i for i in range(axes[ax_ind].L)]

    @classmethod
    def shape(cls, axes: tuple['Axis']) -> tuple:
        """ get shape, axes are sequentially ordered
        """
        return super().shape(axes)


    #########################
    ## convert array to TN ##
    #########################

    @classmethod
    def map_state_to_mps(cls, axes, state, site_ind_id='i({})', site_tag_id='T({})', direction=0, split_opts=None,
                         ancilla_right=(), ancilla_right_inds=(), ancilla_left=(), ancilla_left_inds=()) -> MPSType:
        """ state is initially written as an ndim-dimensional tensor of length q**L
            ancilla_right, and ancilla_left: add ancilla legs (ints = size of leg)
                to the right and left sides of MPS. only one of them can be -1
        """
        tens_shape = cls.shape(axes)
        num_pts = np.prod(tens_shape)
        L = len(tens_shape)
        inds_list = cls.get_tensor_inds(axes)   # takes care of left/right flipping

        assert (state.size % num_pts == 0), \
            (f'state size should be a multiple of {num_pts} not ' + str(tens_shape))

        assert (state.ndim - len(ancilla_left) - len(ancilla_right) == len(axes)), \
            'state should be K-dimensional form'

        for i in range(len(axes)):
            ax = axes[i]
            state = ax.map.transform_vector(state, axis=len(ancilla_left)+i)

        if len(ancilla_right) > 0 or len(ancilla_left) > 0:   # add ancillas

            new_tens = state.reshape(ancilla_left+tens_shape+ancilla_right)
            inds = ancilla_left_inds + tuple([site_ind_id.format(i) for i in inds_list]) + ancilla_right_inds

            if len(ancilla_left) == 0:           # only ancilla on right
                site_nlegs = (1,) * (L-1) + (len(ancilla_right)+1,)
            elif len(ancilla_right) == 0:        # only ancilla on left
                site_nlegs = (len(ancilla_left) + 1,) + (1,) * (L-1)
            else:                           # both exist
                site_nlegs = (len(ancilla_left)+1,) + (1,) * (L-2) + (len(ancilla_right)+1,)

            ## decompose into an MPS using quimb
            new_tens = qtn.Tensor(new_tens, inds=inds)

            ## desired ind ordering
            new_inds = ancilla_left_inds + tuple([site_ind_id.format(i) for i in range(L)]) + ancilla_right_inds
            new_tens.transpose(*new_inds, inplace=True)
            new_mps = helper.tn1D_from_dense(new_tens, L, site_nlegs, site_tag_id=site_tag_id, direction=direction,
                                             split_opts=split_opts)
            new_mps =  MPS.from_TN(new_mps, inplace=True, cyclic=False, L=L,
                                   site_tag_id=site_tag_id, site_ind_id=site_ind_id)
        else:
            ## fold each axis into n-ary form, each dimension considered serially
            ## eg. (q x q x q ...) ** self.dim
            new_tens = state.reshape(tens_shape)
            ## decompose into an MPS using quimb
            new_tens = qtn.Tensor(new_tens, inds=[site_ind_id.format(i) for i in inds_list])
            new_mps = helper.mpx_from_dense(new_tens, L, [site_ind_id], site_tag_id=site_tag_id,
                                            return_mpx=True, split_opts=split_opts)

        # print('layout sequential map state to mps new mps', [ax.map for ax in axes])
        # print('new mps', new_mps)
        return new_mps


    @classmethod
    def map_operator_to_mpo(cls, axes, operator, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='T({})',
                            direction=0, split_opts=None, ancilla_right=(), ancilla_right_inds=(), ancilla_left=(),
                            ancilla_left_inds=()) -> MPOType:
        """ state is initially written as an ndim-dimensional tensor of length q**L
            n-dim operator is (i0 x i1 x i2 ...) x (o0 x o1 x o2 ...)
        """
        tens_shape = cls.shape(axes)
        num_pts = np.prod(tens_shape)
        L = len(tens_shape)
        inds_list = cls.get_tensor_inds(axes)   # takes care of l/r flipping if needed

        assert (operator.size % num_pts**2 == 0), \
            (f'state size should be multiple of {num_pts**2} not ' + str(operator.shape))

        assert (operator.ndim - len(ancilla_left) - len(ancilla_right) == 2*len(axes)), \
            'state should be K-dimensional form'
        for i in range(len(axes)):
            ax: 'Axis' = axes[i]
            operator = ax.map.transform_operator(operator, axis1=len(ancilla_left)+i,
                                                 axis2=len(ancilla_left)+len(axes)+i)

        ## inds of current tensor
        inds = tuple([upper_ind_id.format(i) for i in inds_list]) + \
               tuple([lower_ind_id.format(i) for i in inds_list])


        if len(ancilla_right) > 0 or len(ancilla_left) > 0:   # add ancillas

            new_tens = operator.reshape(ancilla_left+tens_shape*2+ancilla_right)

            if len(ancilla_left) == 0:           # only ancilla on right
                site_nlegs = (2,) * (L-1) + (2+len(ancilla_right),)
            elif len(ancilla_right) == 0:        # only ancilla on left
                site_nlegs = (len(ancilla_left)+2,) + (2,) * (L-1)
            else:                           # both exist
                site_nlegs = (len(ancilla_left)+2,) + (2,) * (L-2) + (len(ancilla_right)+2,)

            ## reordered tensor inds
            new_inds = []
            for i in range(L):
                new_inds += [upper_ind_id.format(i), lower_ind_id.format(i)]
            new_inds = tuple(new_inds)

            inds = ancilla_left_inds + inds + ancilla_right_inds
            new_inds = ancilla_left_inds + new_inds + ancilla_right_inds

            ## decompose into an MPS using quimb
            new_tens = qtn.Tensor(new_tens, inds=inds)
            new_tens.transpose(*new_inds, inplace=True)
            new_mpo = helper.tn1D_from_dense(new_tens, L, site_nlegs, site_tag_id=site_tag_id, direction=direction,
                                             split_opts=split_opts)
            new_mpo =  MPO.from_TN(new_mpo, inplace=True, cyclic=False, L=L,
                                    site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)

        else:
            ## fold each axis into n-ary form, each dimension considered serially
            ## eg. (q x q x q ...) ** self.dim for out x in
            # new_tens = self.axis_map.array_reshape(operator, tens_shape*2)
            new_tens = operator.reshape(tens_shape*2)

            ## decompose into an MPS using quimb
            new_tens = qtn.Tensor(new_tens, inds=inds)
            new_mpo = helper.mpx_from_dense(new_tens, L, [upper_ind_id, lower_ind_id], site_tag_id=site_tag_id,
                                            return_mpx=True, split_opts=split_opts)


        return new_mpo


    @classmethod
    def map_mps_to_state(cls, axes, mps, ax_select=None, ancilla_right=(), ancilla_right_inds=(),
                         ancilla_left=(), ancilla_left_inds=()) \
            -> Union[np.ndarray, qtn.Tensor]:
        """ convert MPS into 1-D np.ndarray
            TO DO: include ancilla?
        """
        L = sum([ax.L for ax in axes])
        out_axes = axes
        out_shape = tuple([ax.npts for ax in axes])
        out_idxs = cls.get_tensor_inds(axes)   # takes care of l/r flipping if needed

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
            out_shape = tuple([ax.npts for ax in out_axes])

            ax_inds = []
            for ax in out_axes:
                ax_inds += cls.get_inds_in_axis(axes, ax)

            tens_inds = cls.get_tensor_inds(out_axes)  ## takes ax.map into account
            out_idxs = [ax_inds[x] for x in tens_inds]

        ## old contract
        tensor = mps.contract()  # contract all tensors
        if isinstance(tensor, (float, np.ndarray, complex)):
            out_data = tensor
        else:
            out_inds = [mps.site_ind_id.format(i) for i in out_idxs]
            tensor.transpose(*ancilla_left_inds, *out_inds, *ancilla_right_inds, inplace=True)

            ## need to reshape to ndim-dimensional tensor
            out_data = tensor.data.reshape(ancilla_left + out_shape + ancilla_right)

            # ## new contract
            # out_inds = [mps.site_ind_id.format(i) for i in out_idxs]
            # inds_seq = [*ancilla_left_inds, *out_inds, *ancilla_right_inds]
            # tensor = helper_quimb.to_dense(mps, inds_seq)
            # # tensor = mps.to_dense(inds_seq)
            #
            # ## need to reshape to ndim-dimensional tensor
            # out_data = tensor.data.reshape(ancilla_left + out_shape + ancilla_right)

            for i in range(len(out_axes)):
                ax = out_axes[i]
                # print('ax', ax, ax.map, len(ancilla_left), out_data.shape, len(out_axes))
                out_data = ax.map.inverse_transform_vector(out_data, axis=len(ancilla_left)+i)

        # if len(ancilla_left) + len(ancilla_right) > 0:
        #     ax_inds = tuple([f'i({ax})' for ax in axes])
        #     out_tens = qtn.Tensor(data=out_data * (10 ** mps.exponent),
        #                           inds=ancilla_left_inds + ax_inds + ancilla_right_inds)
        #     return out_tens
        # else:
        #     return out_data * (10 ** mps.exponent)
        return out_data * (10 ** mps.exponent)


    @classmethod
    def map_mpo_to_operator(cls, axes, mpo: 'qtn.MatrixProductOperator', ax_select=None, ancilla_right=(),
                            ancilla_right_inds=(), ancilla_left=(), ancilla_left_inds=()) -> Union[np.ndarray, qtn.Tensor]:
        """ convert MPO into 2*K-D np.ndarray (o0 x o1 ...) x (i0 x i1 ...)
            TO DO: include ancilla?
        """
        L = sum([ax.L for ax in axes])
        out_axes = axes
        out_shape = tuple([ax.npts for ax in axes])
        out_idxs = cls.get_tensor_inds(axes)  # takes care of l/r flipping if needed

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
            out_shape = tuple([ax.npts for ax in out_axes])

            ax_inds = []
            for ax in out_axes:
                ax_inds += cls.get_inds_in_axis(axes, ax)

            tens_inds = cls.get_tensor_inds(out_axes)  ## takes ax.map into account
            out_idxs = [ax_inds[x] for x in tens_inds]

        tensor: qtn.Tensor = mpo.contract()  # contract all tensors
        out_inds = [mpo.upper_ind_id.format(i) for i in out_idxs] + \
                   [mpo.lower_ind_id.format(i) for i in out_idxs]
        tensor.transpose(*ancilla_left_inds, *out_inds, *ancilla_right_inds, inplace=True)

        ## need to reshape to ndim-dimensional tensor
        out_data = tensor.data.reshape(ancilla_left + out_shape * 2 + ancilla_right)

        for i in range(len(out_axes)):
            ax = out_axes[i]
            out_data = ax.map.inverse_transform_operator(out_data, axis1=len(ancilla_left)+i, axis2=len(out_axes)+i)

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
    def make_mps_ndim(cls, axes, mps_1d_dict) -> MPSType:
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            if MPS not defined along that dimension, use a ones vector (constant along that dimension)
        """
        ndim = len(axes)
        L = cls.L(axes)

        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]

        if ndim == 1:       return ref_mps

        new_mps = super().make_mps_ndim(axes,mps_1d_dict)  ### technically a TN object, all tags same as ref

        site_ind_id = ref_mps.site_ind_id
        site_tag_id = ref_mps.site_tag_id

        ## add bond between last tens of dim and first tens of dim+1
        for dim in range(ndim - 1):
            axL = axes[dim]
            tL = new_mps.select_tensors((site_tag_id.format(axL.L - 1), f'dim_{axL}'))[0]
            axR = axes[dim + 1]
            tR = new_mps.select_tensors((site_tag_id.format(0), f'dim_{axR}'))[0]
            tL.new_bond(tR)  # size of 1

        ## reindex physical bonds
        for x in range(ndim):
            ax = axes[x]
            ax_inds = cls.get_inds_in_axis(axes, ax, ax_ind=x)
            for i in range(ax.L):
                new_ind = ax_inds[i]
                tens = new_mps.select_tensors((site_tag_id.format(i), f'dim_{ax}'))[0]
                tens.reindex({f'i({i}),{ax}': site_ind_id.format(new_ind)}, inplace=True)
                tens.drop_tags()
                tens.add_tag(site_tag_id.format(new_ind))

        # new_mps already contains new exponent
        new_mps.view_like(ref_mps, L=L, inplace=True)
        return new_mps


    @classmethod
    def make_mpx_ndim(cls, axes, mps_1d_dict) -> MPOType:
        """ combine 1-D MPSs into K-dimensional MPO
            if MPS not defined along that dimension, pad with the identity MPO
        """
        ndim = len(axes)
        L = cls.L(axes)

        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]
        out_axes = [ax for ax in axes if ax not in mps_1d_dict]

        if ndim == 1:   return ref_mps
        if len(out_axes) == 0:   return cls.make_mps_ndim(axes, mps_1d_dict)

        new_mpx = super().make_mpx_ndim(axes, mps_1d_dict)  ### technically a TN object

        site_tag_id = ref_mps.site_tag_id
        lower_ind_id = ref_mps.site_ind_id
        upper_ind_id = 'o({})'

        ## add bond between last tens of dim and first tens of dim+1
        for dim in range(ndim - 1):
            axL = axes[dim]
            tL = new_mpx.select_tensors((site_tag_id.format(axL.L - 1), f'dim_{axL}'))[0]
            axR = axes[dim+1]
            tR = new_mpx.select_tensors((site_tag_id.format(0), f'dim_{axR}'))[0]

            if scipy.sparse.issparse(tL.data):
                tL.modify(data=np.array(tL.data.todense()))
            if scipy.sparse.issparse(tR.data):
                tR.modify(data=np.array(tR.data.todense()))

            tL.new_bond(tR)  # size of 1

        ## reindex physical bonds
        for x in range(ndim):
            ax = axes[x]
            ax_inds = cls.get_inds_in_axis(axes, ax, ax_ind=x)
            for i in range(ax.L):
                new_ind = ax_inds[i]
                tens = new_mpx.select_tensors((site_tag_id.format(i), f'dim_{ax}'))[0]

                if ax not in out_axes:  ## no output leg, make a dummy one of size 1
                    tens.reindex({f'i({i}),{ax}': lower_ind_id.format(new_ind)}, inplace=True)
                    tens.new_ind(upper_ind_id.format(new_ind))
                else:
                    tens.reindex({f'i({i}),{ax}': lower_ind_id.format(new_ind),
                                  f'o({i}),{ax}': upper_ind_id.format(new_ind)}, inplace=True)
                tens.drop_tags()
                tens.add_tag(site_tag_id.format(new_ind))

        # new_mpx already contains new exponent
        new_mpx.view_as(MPO, inplace=True, site_tag_id=ref_mps.site_tag_id,
                        lower_ind_id=lower_ind_id, upper_ind_id=upper_ind_id,
                        L=L, cyclic=ref_mps.cyclic)
        return new_mpx


    @classmethod
    def make_mpo_ndim(cls, axes, mpo_1d_dict) -> MPOType:
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        ndim = len(axes)
        L = cls.L(axes)

        # if len(mpo_1d_dict) == 0:
        #     ax = axes[0]
        #     mpo_1d_dict = {ax: ax.get_iden_mpo()}

        ref_mpo = mpo_1d_dict[next(iter(mpo_1d_dict))]

        if ndim == 1:
            return ref_mpo

        new_mpo = super().make_mpo_ndim(axes,mpo_1d_dict)

        site_tag_id = ref_mpo.site_tag_id
        lower_ind_id = ref_mpo.lower_ind_id
        upper_ind_id = ref_mpo.upper_ind_id

        ## add bond between last tens of dim and first tens of dim+1
        for dim in range(ndim - 1):
            axL = axes[dim]
            tL = new_mpo.select_tensors((site_tag_id.format(axL.L - 1), f'dim_{axL}'))[0]
            axR = axes[dim + 1]
            tR = new_mpo.select_tensors((site_tag_id.format(0), f'dim_{axR}'))[0]

            if scipy.sparse.issparse(tL.data):
                tL.modify(data=np.array(tL.data.todense()))
            if scipy.sparse.issparse(tR.data):
                tR.modify(data=np.array(tR.data.todense()))

            tL.new_bond(tR)  # size of 1

        ## reindex physical bonds
        for x in range(ndim):
            ax = axes[x]
            ax_inds = cls.get_inds_in_axis(axes, ax, ax_ind=x)
            for i in range(ax.L):
                new_ind = ax_inds[i]
                tens = new_mpo.select_tensors((site_tag_id.format(i), f'dim_{ax}'))[0]
                tens.reindex({f'i({i}),{ax}': lower_ind_id.format(new_ind),
                              f'o({i}),{ax}': upper_ind_id.format(new_ind)}, inplace=True)
                tens.drop_tags()
                tens.add_tag(site_tag_id.format(new_ind))

        # new_mpo already contains new exponent
        new_mpo.view_like(ref_mpo, L=L, inplace=True)
        return new_mpo


    @classmethod
    def make_tn1D_ndim(cls, axes, tn1D_1d_dict, upper_ind_id='i({})', lower_ind_id='o({})', extra_ind_ids=(),
                       site_tag_id='T({})') -> TNType:
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        ndim = len(axes)

        if ndim == 1:
            return tn1D_1d_dict[next(iter(tn1D_1d_dict))]

        new_tn = super().make_tn1D_ndim(axes, tn1D_1d_dict, upper_ind_id, lower_ind_id, extra_ind_ids, site_tag_id)
        ind_ids = (upper_ind_id, lower_ind_id) + extra_ind_ids

        # ## add bond between last tens of dim and first tens of dim+1
        # for dim in range(ndim - 1):
        #     axL = axes[dim]
        #     tL = new_tn.select_tensors((site_tag_id.format(axL.L - 1), f'dim_{axL.axID}'))[0]
        #     axR = axes[dim + 1]
        #     tR = new_tn.select_tensors((site_tag_id.format(0), f'dim_{axR.axID}'))[0]
        #     tL.new_bond(tR)  # size of 1

        ## reindex physical bonds
        for x in range(ndim):
            ax = axes[x]
            ax_inds = cls.get_inds_in_axis(axes, ax, ax_ind=x)

            inds = ind_ids
            # if ax in tn1D_1d_dict:  # is general tensor
            #     inds = ind_ids
            # else:  # is identity MPO
            #     inds = (upper_ind_id, lower_ind_id)

            for i in range(ax.L):
                new_ind = ax_inds[i]
                tens = new_tn.select_tensors((site_tag_id.format(i), f'dim_{ax}'))[0]
                tens.reindex({ind_.format(i)+f',{ax}': ind_.format(new_ind) for ind_ in inds}, inplace=True)
                tens.drop_tags()
                tens.add_tag(site_tag_id.format(new_ind))

        new_tn.view_as(MatrixProductTensor, inplace=True,  L=new_tn.num_tensors, cyclic=False,
                       site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                       extra_ind_ids=extra_ind_ids)

        # new_tn already contains new exponent
        return new_tn


    # @classmethod
    # def pad_mps_to_grid(cls, axes: tuple['Axis'], scalar_field_mps: 'MPSType', mps_axes: tuple['Axis']) -> MPSType:
    #     """ pad scalar_field_mps, which exists on axes labeled by axIDs
    #         in such that they exist in (target_ndim)-dimensional space
    #         fields are constant along all dimensions not specified by self.axes
    #         need either new_grid or target_ndim
    #     """
    #     L = cls.L(axes)
    #     pad_axes = [ax for ax in axes if ax not in mps_axes]
    #
    #     site_ind_id = scalar_field_mps.site_ind_id
    #     site_tag_id = scalar_field_mps.site_tag_id
    #
    #     new_mps = scalar_field_mps.view_as(qtn.TensorNetwork)
    #     for x in range(len(mps_axes)):
    #         ax = mps_axes[x]
    #         old_ax_inds = cls.get_inds_in_axis(mps_axes, ax, ax_ind=x)
    #         new_ax_inds = cls.get_inds_in_axis(axes, ax)
    #         for i in range(len(old_ax_inds)):
    #             old_ind = old_ax_inds[i]
    #             new_ind = new_ax_inds[i]
    #             new_mps.reindex({site_ind_id.format(old_ind): f'_tmp{new_ind}_'}, inplace=True)
    #             new_mps.retag({site_tag_id.format(old_ind): f'_TMP{new_ind}_'}, inplace=True)
    #
    #     for ax in pad_axes:
    #         one_mps = ax.get_iden_mps(site_ind_id=site_ind_id, site_tag_id=site_tag_id)
    #         new_ax_inds = cls.get_inds_in_axis(axes, ax)
    #         for i in range(ax.L):
    #             new_ind = new_ax_inds[i]
    #             one_mps.reindex({site_ind_id.format(i): f'_tmp{new_ind}_'}, inplace=True)
    #             one_mps.retag({site_tag_id.format(i): f'_TMP{new_ind}_'}, inplace=True)
    #         new_mps.add(one_mps)
    #
    #     # print('new mps', new_mps)
    #
    #     L0 = 0
    #     for subL in [ax.L for ax in axes[:-1]]:
    #         L0 = L0 + subL
    #         tens1 = new_mps.select_tensors(f'_TMP{L0 - 1}_')[0]
    #         tens2 = new_mps.select_tensors(f'_TMP{L0}_')[0]
    #         qtn.new_bond(tens1, tens2)
    #
    #     # print('new mps 3', new_mps)
    #
    #     new_mps.fuse_multibonds(inplace=True)
    #     new_mps.view_like(scalar_field_mps, inplace=True, L=L,
    #                       site_tag_id='_TMP{}_', site_ind_id='_tmp{}_')
    #     new_mps.site_ind_id = site_ind_id
    #     new_mps.site_tag_id = site_tag_id
    #     # new_mps.exponent = scalar_field_mps.exponent  ## not needed
    #     return new_mps


    @classmethod
    def pad_mps_to_grid(cls, axes: tuple['Axis'], scalar_field_mps: 'MPSType', mps_axes: tuple['Axis']) -> MPSType:
        """ pad scalar_field_mps, which exists on axes labeled by axIDs
            in such that they exist in (target_ndim)-dimensional space
            fields are constant along all dimensions not specified by self.axes
            need either new_grid or target_ndim
        """
        L = cls.L(axes)
        old_L = scalar_field_mps.L
        pad_axes = [ax for ax in axes if ax not in mps_axes]
        # print('PAD MPS')
        # print('pad axes', pad_axes)
        # print('mps axes', mps_axes)

        site_ind_id = scalar_field_mps.site_ind_id
        site_tag_id = scalar_field_mps.site_tag_id

        new_mps = scalar_field_mps.view_as(qtn.TensorNetwork)

        old_new_r = 0
        new_ind_dims_l = {}
        new_ind_dims_r = {}
        ## adjust horizontal bond names
        for x in range(len(mps_axes)):
            ax = mps_axes[x]
            old_ax_inds = cls.get_inds_in_axis(mps_axes, ax, ax_ind=x)
            new_ax_inds = cls.get_inds_in_axis(axes, ax)
            # print('ax', ax, old_ax_inds, new_ax_inds)

            ind_l = old_ax_inds[0]
            ind_r = old_ax_inds[-1]
            new_l = new_ax_inds[0]
            new_r = new_ax_inds[-1]

            if x < len(mps_axes) - 1:
                next_new_ax_inds = cls.get_inds_in_axis(axes, mps_axes[x + 1])
                next_new_l = next_new_ax_inds[0]
            else:
                next_new_l = None

            ## left (if x == 0)
            TL0 = new_mps.select_tensors(site_tag_id.format(ind_l))[0]
            if ind_l == 0:
                if new_l != 0:
                    size_l, name_l = 1, f'h{ind_l}' + str(ax)
                    TL0.new_ind(name_l + '_L')
                    new_ind_dims_r[new_l] = size_l, name_l + '_L'
            else:  ## from previous iteration
                if new_l != old_new_r + 1:
                    new_ind_dims_r[new_l] = size_r, name_r + '_L'

            ## right
            TR0 = new_mps.select_tensors(site_tag_id.format(ind_r))[0]
            if ind_r == old_L - 1:
                if new_ax_inds[-1] != L - 1:
                    size_r, name_r = 1, f'h{ind_r}' + str(ax)
                    TR0.new_ind(name_r + '_R')
                    new_ind_dims_l[new_r] = size_r, name_r + '_R'
            else:
                if next_new_l != new_r + 1:
                    TR1 = new_mps.select_tensors(site_tag_id.format(ind_r + 1))[0]
                    name_r = next(iter(qtn.bonds(TR0, TR1)))
                    size_r = qtn.bonds_size(TR0, TR1)
                    ## reindex virtual bond
                    TR1.reindex({name_r: name_r + '_L'}, inplace=True)
                    TR0.reindex({name_r: name_r + '_R'}, inplace=True)

                    new_ind_dims_l[new_r] = size_r, name_r + '_R'
            old_new_r = new_r

        ## renumber mps
        for x in range(len(mps_axes)):
            ax = mps_axes[x]
            old_ax_inds = cls.get_inds_in_axis(mps_axes, ax, ax_ind=x)
            new_ax_inds = cls.get_inds_in_axis(axes, ax)

            for i in range(len(old_ax_inds)):
                old_ind = old_ax_inds[i]
                new_ind = new_ax_inds[i]

                new_mps.reindex({site_ind_id.format(old_ind): f'_tmp{new_ind}_'}, inplace=True)
                new_mps.retag({site_tag_id.format(old_ind): f'_TMP{new_ind}_'}, inplace=True)

        # print('new mps', new_mps)
        # print('pad axes', pad_axes)

        for ax in pad_axes:
            new_ax_inds = cls.get_inds_in_axis(axes, ax)
            # print('new ax inds', new_ax_inds)
            new_l = new_ax_inds[0]
            new_r = new_ax_inds[-1]

            anc_dim_l, anc_name_l = new_ind_dims_l.get(new_l - 1, (None, None))
            anc_dim_r, anc_name_r = new_ind_dims_r.get(new_r + 1, (None, None))
            if anc_dim_l is not None and anc_dim_r is not None:
                assert (anc_dim_l == anc_dim_r), f'ancilla dims must be the same, currently {anc_dim_l}, {anc_dim_r}'
            anc_dim = anc_dim_l if anc_dim_r is None else anc_dim_r

            if anc_name_l is None:  # and new_l != 0:
                anc_name_l = f'h{new_l}' + str(ax) + '_L'
                new_ind_dims_r[new_ax_inds[0]] = anc_dim, anc_name_l
            if anc_name_r is None:  # and new_r != L-1:
                anc_name_r = f'h{new_r}' + str(ax) + '_R'
                new_ind_dims_l[new_ax_inds[-1]] = anc_dim, anc_name_r

            anc_dim = 1 if anc_dim is None else anc_dim
            iden_mps = ax.get_iden_mps(site_ind_id=site_ind_id, site_tag_id=site_tag_id,
                                       anc_dim=anc_dim, anc_name_l=anc_name_l, anc_name_r=anc_name_r)
            if new_l == 0 and anc_dim is not None:
                iden_mps[0].isel({anc_name_l: 0}, inplace=True)
            if new_r == L - 1 and anc_dim is not None:
                iden_mps[-1].isel({anc_name_r: 0}, inplace=True)

            for i in range(ax.L):
                new_ind = new_ax_inds[i]
                iden_mps.reindex({site_ind_id.format(i): f'_tmp{new_ind}_'}, inplace=True)
                iden_mps.retag({site_tag_id.format(i): f'_TMP{new_ind}_'}, inplace=True)
            new_mps.add(iden_mps)

        # print('new mps 2', new_mps)

        # L0 = 0
        # for subL in [ax.L for ax in axes[:-1]]:
        #     L0 = L0 + subL
        #     tens1 = new_mps.select_tensors(f'_TMP{L0 - 1}_')[0]
        #     tens2 = new_mps.select_tensors(f'_TMP{L0}_')[0]
        #     qtn.new_bond(tens1, tens2)
        #
        # print('new mps 3', new_mps)

        new_mps.fuse_multibonds(inplace=True)
        new_mps.view_like(scalar_field_mps, inplace=True, L=L,
                          site_tag_id='_TMP{}_', site_ind_id='_tmp{}_')
        new_mps.site_ind_id = site_ind_id
        new_mps.site_tag_id = site_tag_id
        # new_mps.exponent = scalar_field_mps.exponent  ## not needed
        return new_mps


    @classmethod
    def pad_mpo_to_grid(cls, axes, scalar_field_mpo, mpo_axes) -> MPOType:
        """ pad scalar_field_mpo, which exists on axes labeled by axIDs
                    in such that they exist in (target_ndim)-dimensional space
                    fields are constant along all dimensions not specified by self.axes
                    need either new_grid or target_ndim
        """
        L = cls.L(axes)
        old_L = scalar_field_mpo.L
        pad_axes = [ax for ax in axes if ax not in mpo_axes]
        dummy_ones = False if axes[0].q > 2 else True       ## if not a QUANTIZED TT

        upper_ind_id = scalar_field_mpo.upper_ind_id
        lower_ind_id = scalar_field_mpo.lower_ind_id
        site_tag_id = scalar_field_mpo.site_tag_id

        new_mpo = scalar_field_mpo.view_as(qtn.TensorNetwork)

        old_new_r = 0
        new_ind_dims = {}
        ## adjust horizontal bond names
        for x in range(len(mpo_axes)):
            ax = mpo_axes[x]
            old_ax_inds = cls.get_inds_in_axis(mpo_axes, ax, ax_ind=x)
            new_ax_inds = cls.get_inds_in_axis(axes, ax)

            ind_l = old_ax_inds[0]
            ind_r = old_ax_inds[-1]
            new_l = new_ax_inds[0]
            new_r = new_ax_inds[-1]
            # print('inds', ind_l, ind_r, new_l, new_r)

            if x < len(mpo_axes) - 1:
                next_new_ax_inds = cls.get_inds_in_axis(axes, mpo_axes[x+1])
                next_new_l = next_new_ax_inds[0]
            else:
                next_new_l = None

            ## left (if x == 0)
            TL0 = new_mpo.select_tensors(site_tag_id.format(ind_l))[0]
            if dummy_ones:   ## don't add bonds of size 1 of in (sparse) TT format
                if ind_l == 0:
                    if new_ax_inds[0] != 0:
                        size_l, name_l = 1, f'h{ind_l}' + str(ax)
                        TL0.new_ind(name_l + '_L')
                        new_ind_dims[new_l] = size_l, name_l + '_L'
                else:   ## from previous iteration
                    if new_l != old_new_r + 1:
                        new_ind_dims[new_l] = size_r, name_r + '_L'

            ## right
            TR0 = new_mpo.select_tensors(site_tag_id.format(ind_r))[0]
            if dummy_ones:   ## don't add bonds of size 1 of in (sparse) TT format
                if ind_r == old_L - 1:
                    if new_ax_inds[-1] != L - 1:
                        size_r, name_r = 1, f'h{ind_r}' + str(ax)
                        TR0.new_ind(name_r + '_R')
                        new_ind_dims[new_r] = size_r, name_r + '_R'
                else:
                    if next_new_l != new_r + 1:
                        TR1 = new_mpo.select_tensors(site_tag_id.format(ind_r + 1))[0]
                        name_r = next(iter(qtn.bonds(TR0, TR1)))
                        size_r = qtn.bonds_size(TR0, TR1)
                        ## reindex virtual bond
                        TR1.reindex({name_r: name_r + '_L'}, inplace=True)
                        TR0.reindex({name_r: name_r + '_R'}, inplace=True)

                        new_ind_dims[new_r] = size_r, name_r + '_R'
                old_new_r = new_r

        ## renumber mps
        for x in range(len(mpo_axes)):
            ax = mpo_axes[x]
            old_ax_inds = cls.get_inds_in_axis(mpo_axes, ax, ax_ind=x)
            new_ax_inds = cls.get_inds_in_axis(axes, ax)

            for i in range(len(old_ax_inds)):
                old_ind = old_ax_inds[i]
                new_ind = new_ax_inds[i]

                new_mpo.reindex({upper_ind_id.format(old_ind): f'_tmpU{new_ind}_',
                                 lower_ind_id.format(old_ind): f'_tmpL{new_ind}_',}, inplace=True)
                new_mpo.retag({site_tag_id.format(old_ind): f'_TMP{new_ind}_'}, inplace=True)

        # print('new mpo', new_mpo)
        # print('pad axes', pad_axes)

        for ax in pad_axes:
            # print('pad ax', ax)
            new_ax_inds = cls.get_inds_in_axis(axes, ax)
            new_l = new_ax_inds[0]
            new_r = new_ax_inds[-1]
            # print('new ax inds', new_ax_inds)

            anc_dim_l, anc_name_l = new_ind_dims.get(new_l-1, (None,None))
            anc_dim_r, anc_name_r = new_ind_dims.get(new_r+1, (None,None))
            if anc_dim_l is not None and anc_dim_r is not None:
                assert(anc_dim_l == anc_dim_r),f'ancilla dims must be the same, currently {anc_dim_l}, {anc_dim_r}'
            anc_dim = anc_dim_l if anc_dim_r is None else anc_dim_r
            # print('anc l', anc_dim_l, anc_name_l)
            # print('anc r', anc_dim_r, anc_name_r)

            if dummy_ones:
                if anc_name_l is None: #  and new_l != 0:
                    anc_name_l = f'h{new_l}'+str(ax) + '_L'
                    new_ind_dims[new_ax_inds[0]] = anc_dim, anc_name_l
                if anc_name_r is None: # and new_r != L-1:
                    anc_name_r = f'h{new_r}' + str(ax) + '_R'
                    new_ind_dims[new_ax_inds[-1]] = anc_dim, anc_name_r

                anc_dim = 1 if anc_dim is None else anc_dim
                iden_mpo = ax.get_iden_mpo(upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id,
                                           anc_dim=anc_dim, anc_name_l=anc_name_l, anc_name_r=anc_name_r)
                if new_l == 0 and anc_dim is not None:
                    iden_mpo[0].isel({anc_name_l:0}, inplace=True)
                if new_r == L-1 and anc_dim is not None:
                    iden_mpo[-1].isel({anc_name_r:0}, inplace=True)
            else:
                iden_mpo = ax.get_iden_mpo(upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                           site_tag_id=site_tag_id,)
            for i in range(ax.L):
                new_ind = new_ax_inds[i]
                iden_mpo.reindex({upper_ind_id.format(i): f'_tmpU{new_ind}_',
                                  lower_ind_id.format(i): f'_tmpL{new_ind}_',}, inplace=True)
                iden_mpo.retag({site_tag_id.format(i): f'_TMP{new_ind}_'}, inplace=True)
            new_mpo.add(iden_mpo)

        # print('new mpo 2', new_mpo)

        # L0 = 0
        # for subL in [ax.L for ax in axes[:-1]]:
        #     L0 = L0 + subL
        #     tens1 = new_mpo.select_tensors(f'_TMP{L0 - 1}_')[0]
        #     tens2 = new_mpo.select_tensors(f'_TMP{L0}_')[0]
        #     qtn.new_bond(tens1, tens2)

        # print('new mpo 3', new_mpo)

        new_mpo.fuse_multibonds(inplace=True)
        new_mpo.view_like(scalar_field_mpo, inplace=True, L=L,
                          site_tag_id='_TMP{}_', upper_ind_id='_tmpU{}_', lower_ind_id='_tmpL{}_')
        new_mpo.upper_ind_id = upper_ind_id
        new_mpo.lower_ind_id = lower_ind_id
        new_mpo.site_tag_id = site_tag_id
        # new_mps.exponent = scalar_field_mps.exponent  ## not needed
        return new_mpo


    @classmethod
    def apply_partial_mps(cls, mps, mpx, axes):
        raise NotImplementedError
