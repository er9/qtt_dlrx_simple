"""Base :class:`Layout` class for multidimensional QTT layouts, providing the
index transposition logic used to interleave or sequence per-axis cores within a
single 1-D tensor network. Concrete layouts (sequential, parallel-factorized,
parallel-grouped) subclass this."""
import helper_quimb
from setup_.defaults import *
from basis.basis_spatial import SpatialBasis
import time
from functools import lru_cache
import numpy as np
import quimb.tensor as qtn
from axis import Axis


class Layout:

    @staticmethod
    def _transpose_sequential_to_parallel_inds_constantL(L, ndim) -> list:
        axT = []
        for x in range(L):
            axT += [x + i * L for i in range(ndim)]
        return axT

    @staticmethod
    def _transpose_parallel_to_sequential_inds_constantL(L, ndim) -> list:
        axT = []
        for x in range(ndim):
            axT += [x + i * ndim for i in range(L)]
        return axT

    @staticmethod
    @lru_cache
    def _transpose_sequential_to_parallel_inds(axes: tuple['Axis']) -> Sequence:
        """ do some sort of caching for this too
            assumes Ls are aligned at index 0
        """
        Ls = [ax.L for ax in axes]
        if all([L==Ls[0] for L in Ls]):
            return Layout._transpose_sequential_to_parallel_inds_constantL(Ls[0],len(Ls))

        axes_inds = [list(range(Ls[0]))]
        for L in Ls[1:]:
            ind_pos = axes_inds[-1][-1] + 1
            axes_inds += [list(range(ind_pos,ind_pos+L))]

        axT = []
        Ls_left = list(Ls)
        while any([L>0 for L in Ls_left]):
            for ax_active in range(len(Ls)):
                try:
                    axT += [axes_inds[ax_active].pop(0)]
                    Ls_left[ax_active] -= 1
                except IndexError:
                    pass

        # ## assumes Ls are aligned at index =-1
        # Ls_left = list(Ls)
        # for i in range(np.sum(Ls)):
        #     ax_active = np.argmax(Ls_left)
        #     axT += [axes_inds[ax_active].pop(0)]
        #     Ls_left[ax_active] -= 1
        return axT

    @staticmethod
    @lru_cache
    def _transpose_parallel_to_sequential_inds(axes: tuple['Axis']) -> Sequence:
        Ls = [ax.L for ax in axes]
        if all([L == Ls[0] for L in Ls]):
            return Layout._transpose_parallel_to_sequential_inds_constantL(Ls[0], len(Ls))

        axes_inds = [[] for _ in Ls]

        i = 0
        while i < sum(Ls):
            for ax_active in range(len(Ls)):
                if len(axes_inds[ax_active]) < Ls[ax_active]:
                    axes_inds[ax_active] += [i]
                    i += 1

        # ## commented out: assumes Ls are aligned at index =-1
        # Ls_left = list(Ls)
        # for i in range(np.sum(Ls)):
        #     ax_active = np.argmax(Ls_left)
        #     axes_inds[ax_active] += [i]
        #     Ls_left[ax_active] -= 1

        axT = [i for ax_inds in axes_inds for i in ax_inds]
        return axT

    @classmethod
    def L(cls, axes: tuple['Axis']) -> int:
        mpx_L = sum([ax.L for ax in axes])
        return mpx_L

    @classmethod
    def get_inds_in_axis(cls, axes: tuple['Axis'], ax:'Axis', ax_ind=None) -> list:
        raise NotImplementedError

    @classmethod
    def shape(cls, axes: tuple['Axis']) -> tuple:
        """ get shape assuming sequential ordering
        """
        out_shape = ()
        for ax in axes:
            out_shape += ax.shape()
        return out_shape

    @classmethod
    def get_tensor_inds(cls, axes: Sequence['Axis']) -> list[int]:
        """ get indices to reorder data tensor
            accounting for if ax.is_flipped = True
        """
        L0 = 0
        inds_list = []
        for ax in axes:
            ax_tens_inds = ax.map.get_tens_order(ax.L)
            inds_list += [L0 + i for i in ax_tens_inds]
            # if ax.is_flipped:
            #     inds_list += [L0 + i for i in range(ax.L - 1, -1, -1)]
            # else:
            #     inds_list += [L0 + i for i in range(ax.L)]
            L0 += ax.L
        return inds_list

    # @classmethod
    # def get_active_inds(cls, full_axes: Sequence['Axis'],
    #                     active_axes: Sequence['Axis'] = None, inactive_axes: Sequence['Axis'] = None) -> list[int]:
    #     """ get indices to reorder data tensor
    #         accounting for if ax.is_flipped = True
    #     """
    #     L0 = 0
    #     inds_list = []
    #     for ax in full_axes:
    #         if (active_axes is None or ax in active_axes) and (inactive_axes is None or ax not in inactive_axes):
    #             ax_tens_inds = ax.map.get_tens_order(ax.L)
    #             inds_list += [L0 + i for i in ax_tens_inds]
    #         L0 += ax.L
    #     return inds_list


    #########################################
    ## convert between np.ndarray and MPX ##
    #########################################

    @classmethod
    def map_state_to_mps(cls, axes: tuple[Axis], state: np.ndarray,
                         site_ind_id: str='i({})', site_tag_id: str='T({})', direction:int =0, split_opts=None,
                         ancilla_right:tuple[int] = (), ancilla_right_inds:tuple[str]=(),
                         ancilla_left:tuple[int] = (), ancilla_left_inds:tuple[str]=()) \
            -> 'MPSType':
        """ convert state represented as vector converted to mps state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        raise NotImplementedError

    @classmethod
    def map_mps_to_state(cls, axes: tuple['Axis'], mps: 'MPSType', ax_select: dict['Axis',int] = None,
                         ancilla_right:tuple[int] = (), ancilla_right_inds:tuple[str] = (),
                         ancilla_left:tuple[int] = (), ancilla_left_inds:tuple[str] = ()) -> Union[np.ndarray,qtn.Tensor]:
        """ convert MPS into 1-D np.ndarray
        """
        raise NotImplementedError

    @classmethod
    def map_operator_to_mpo(cls, axes: tuple['Axis'], operator: np.ndarray, upper_ind_id='o({})', lower_ind_id='i({})',
                            site_tag_id='T({})', direction:int=0, split_opts=None,
                            ancilla_right:tuple[int] = (), ancilla_right_inds:tuple[str]=(),
                            ancilla_left:tuple[int] = (), ancilla_left_inds:tuple[str]=()) -> 'MPOType':
        """ convert operator represented as high-dimensional matrix to mpo state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        raise NotImplementedError

    @classmethod
    def map_mpo_to_operator(cls, axes: tuple['Axis'], mpo: 'MPOType',
                            ax_select: dict['Axis',Union[int,tuple[int]]] = None,
                            ancilla_right:tuple[int]=(), ancilla_right_inds:tuple[str]=(),
                            ancilla_left:tuple[int]=(), ancilla_left_inds:tuple[str]=()) \
            -> Union[np.ndarray,qtn.Tensor]:
        """ convert MPO into 1-D np.ndarray
        """
        raise NotImplementedError


    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    @classmethod
    def make_mps_ndim(cls, axes: tuple['Axis'], mps_1d_dict: dict['Axis','MPSType']):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            if MPS not defined along that dimension, use a ones vector (constant along that dimension)
        """
        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]
        ref_tag = ref_mps.site_tag_id
        # if isinstance(ref_mps, qtn.TensorNetwork1D):
        #     ref_tag = ref_mps.site_tag_id
        # else:
        #     ref_tag = 'T({})'

        mps_list = []
        new_exponent = 0.0
        for ax in axes:
            site_ind_id = 'i({})' + f',{ax}'
            if ax not in mps_1d_dict:
                mps_ = ax.get_iden_mps(site_ind_id=site_ind_id, site_tag_id=ref_tag)
            else:
                data = mps_1d_dict[ax]
                if isinstance(data, qtn.MatrixProductState):
                    mps_ = data.copy()
                    mps_.reindex_sites(site_ind_id, inplace=True)
                    mps_.retag_sites(ref_tag, inplace=True)
                # elif isinstance(data, np.ndarray):
                #     mps_ = ax.map_state_to_mps(data, site_ind_id=site_ind_id, site_tag_id=ref_tag)
                else:
                    raise NotImplementedError
                new_exponent += mps_.exponent

            mps_.add_tag(f'dim_{ax}')
            mps_list.append(mps_)

        new_mps = qtn.TensorNetwork(mps_list)
        new_mps.exponent = new_exponent
        new_mps.view_as(qtn.TensorNetwork1D, inplace=True, L=new_mps.num_tensors, site_tag_id=ref_tag)
        return new_mps


    @classmethod
    def make_mpx_ndim(cls, axes: tuple['Axis'], mps_1d_dict: dict['Axis':'MPSType']):
        """ combine 1-D MPSs into K-dimensional MPO
            if MPS not defined along that dimension, pad with the identity MPO
        """
        ref_mps = mps_1d_dict[next(iter(mps_1d_dict))]
        ref_tag = ref_mps.site_tag_id
        # if isinstance(ref_mps, qtn.TensorNetwork1D):
        #     ref_tag = ref_mps.site_tag_id
        # else:
        #     ref_tag = 'T({})'

        mpx_list = []
        out_dims = []
        new_exponent = 0.0
        for ax in axes:
            site_ind_id = 'i({})' + f',{ax}'    # have choice to make 'o({})'+f',{ax}'?
            if ax not in mps_1d_dict:
                out_dims += [ax]
                mpx_ = ax.get_iden_mpo(upper_ind_id='o({})' + f',{ax}',
                                       lower_ind_id='i({})' + f',{ax}',
                                       site_tag_id=ref_mps.site_tag_id)
            else:
                data = mps_1d_dict[ax]
                if isinstance(data, qtn.MatrixProductState):
                    mpx_ = data.copy()
                    mpx_.reindex_sites(site_ind_id, inplace=True)
                    mpx_.retag_sites(ref_tag, inplace=True)
                # elif isinstance(data, np.ndarray):
                #     mpx_ = ax.map_state_to_mps(data, site_ind_id=site_ind_id, site_tag_id=ref_tag)
                else:
                    raise NotImplementedError
                new_exponent += mpx_.exponent

            mpx_.add_tag(f'dim_{ax}')
            mpx_list.append(mpx_)

        new_mpx = qtn.TensorNetwork(mpx_list)
        new_mpx.exponent = new_exponent
        new_mpx.view_as(qtn.TensorNetwork1D, inplace=True, L=new_mpx.num_tensors, site_tag_id=ref_tag)
        return new_mpx


    @classmethod
    def make_mpo_ndim(cls, axes: tuple['Axis'], mpo_1d_dict: dict['Axis',MPOType]) -> TNType:
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        ref_mpo = mpo_1d_dict[next(iter(mpo_1d_dict))]
        ref_tag = ref_mpo.site_tag_id
        # if isinstance(ref_mpo, qtn.TensorNetwork1D):
        #     ref_tag = ref_mpo.site_tag_id
        # else:
        #     ref_tag = 'T({})'

        mpo_list = []
        new_exponent = 0.0
        for ax in axes:
            upper_ind_id = 'o({})' + f',{ax}'
            lower_ind_id = 'i({})' + f',{ax}'
            if ax not in mpo_1d_dict:
                mpo_ = ax.get_iden_mpo(upper_ind_id=upper_ind_id,
                                       lower_ind_id=lower_ind_id,
                                       site_tag_id=ref_tag)
            else:
                data = mpo_1d_dict[ax]
                if isinstance(data, qtn.MatrixProductOperator):
                    mpo_ = data.copy()
                    mpo_.reindex_upper_sites(upper_ind_id, inplace=True)
                    mpo_.reindex_lower_sites(lower_ind_id, inplace=True)
                    mpo_.retag_sites(ref_tag, inplace=True)
                # elif isinstance(data, np.ndarray):
                #     mpo_ = ax.map_operator_to_mpo(data, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                #                                   site_tag_id=ref_tag)
                else:
                    raise TypeError('make mpo ndim needs dict of MPO objects')
                new_exponent += mpo_.exponent

            # helper_quimb.compress(mpo_)
            # print('make mpo ndim mpo exponent', mpo_.exponent)
            #
            # new_exponent += mpo_.exponent

            mpo_.add_tag(f'dim_{ax}')
            mpo_list.append(mpo_)

        new_mpo = qtn.TensorNetwork(mpo_list)
        new_mpo.exponent = new_exponent
        new_mpo.view_as(qtn.TensorNetwork1D, inplace=True, L=new_mpo.num_tensors, site_tag_id=ref_tag)
        return new_mpo


    @classmethod
    def make_tn1D_ndim(cls, axes: tuple['Axis'], tn1D_1d_dict: dict['Axis','TN1Type'],
                       upper_ind_id='i({})', lower_ind_id='o({})', extra_ind_ids=(),
                       site_tag_id='T({})'):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.

        """
        ind_ids = (upper_ind_id, lower_ind_id) + extra_ind_ids

        new_tn = qtn.TensorNetwork([])
        new_exponent = 0.0
        for ax in axes:
            if ax not in tn1D_1d_dict:
                tn_ = ax.get_diagonalize_mps_tn(in1_ind_id=lower_ind_id + f',{ax}',
                                                in2_ind_id=extra_ind_ids[0] + f',{ax}',
                                                out_ind_id=upper_ind_id + f',{ax}',
                                                site_tag_id=site_tag_id)
                tn_.add_tag(f'dim_{ax}')
                # mpo_ = ax.get_iden_mpo(upper_ind_id=upper_ind_id + f',{ax}',
                #                        lower_ind_id=lower_ind_id + f',{ax}',
                #                        site_tag_id=site_tag_id)
                # new_tn.add(mpo_)
            else:
                tn_ = tn1D_1d_dict[ax].copy()
                for i in range(tn_.num_tensors):
                    tens_ = tn_.select_tensors(site_tag_id.format(i))[0]
                    tens_.reindex({ind_.format(i): ind_.format(i)+f',{ax}' for ind_ in ind_ids},
                                  inplace=True)
                    tens_.add_tag(site_tag_id.format(i))
                    tens_.add_tag(f'dim_{ax}')

            new_exponent += tn_.exponent
            new_tn.add(tn_)

            new_tn.exponent = new_exponent
            new_tn.view_as(qtn.TensorNetwork1D, inplace=True, L=new_tn.num_tensors, site_tag_id=site_tag_id)
        return new_tn

    @classmethod
    def pad_mps_to_grid(cls, axes: tuple['Axis'], scalar_field_mps: 'MPSType', mps_axes: tuple['Axis']):
        raise NotImplementedError

    @classmethod
    def pad_mpo_to_grid(cls, axes: tuple['Axis'], scalar_field_mpo: 'MPOType', mpo_axes: tuple['Axis']):
        raise NotImplementedError

