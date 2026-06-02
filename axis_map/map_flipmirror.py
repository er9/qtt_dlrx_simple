"""Flipped mirror axis map: the mirror quantization of :mod:`axis_map.map_mirror`
with an additional left/right flip of the core ordering."""
from functools import lru_cache
import helper_quimb as helper
from setup_.configs import *
from axis_map.map import AxisMap
from axis_map.map_mirror import MirrorMap

""" binary mapping + two's complement + flip left/right
    equivalently, fold data in half   <---|---> and then do original transformation + flip l/r
    NOT DEBUGGED
"""

## flipped binary TN class
class FlipMirrorMap(MirrorMap):

    @classmethod
    def get_position_inds(cls, L, q, idx):
        """ returns physical bond indices (0,1) of MPS that corresponds to vector position idx
        """
        ind_list = super().get_position_inds(L, q, idx)
        ind_list = ind_list[::-1]   # flip lr
        return ind_list

    @classmethod
    def get_inds_position(cls, q, inds_list):
        """ returns index in vector corresponding to physical bond indices specified in inds_list
        """
        inds_list = inds_list[::-1]     # flip lr
        idx = super().get_inds_position(q, inds_list)
        return idx

    @classmethod
    def get_tens_order(cls, L):
        return range(L - 1, -1, -1)

    @classmethod
    def is_flipped(cls):
        return True

    # @classmethod
    # def get_map_mpo(cls, L, q=2, site_tag_id='T({})', upper_ind_id='o({})', lower_ind_id='i({})'):
    #     assert(q==2), f'q must be 2, not {q}'
    #     mirror_mpo = super()._twos_complement_mpo(L, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
    #                                               lower_ind_id=lower_ind_id)
    #     flip_mpo = super()._flip_mpo(L, q, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
    #                          lower_ind_id=lower_ind_id)
    #     return helper.apply(flip_mpo, mirror_mpo)
    #
    # @classmethod
    # def get_inverse_map_mpo(cls, L, q=2, site_tag_id='T({})', upper_ind_id='o({})', lower_ind_id='i({})'):
    #     mirror_mpo = super()._twos_complement_mpo(L, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
    #                                         lower_ind_id=lower_ind_id)
    #     flip_mpo = super()._flip_mpo(L, q, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
    #                          lower_ind_id=lower_ind_id)
    #     return helper.apply(mirror_mpo, flip_mpo)

    # @classmethod
    # def _flip_mpo(cls, L, q=2, site_tag_id='T({})', upper_ind_id='o({})', lower_ind_id='i({})'):
    #     flip_tens = np.zeros((q, q))
    #     for i in range(q):
    #         flip_tens[i, -1 - i] = 1.0
    #
    #     mpo_0 = qtn.MatrixProductOperator(
    #         [np.array([flip_tens])] + [np.array([[flip_tens]])] * (L - 2) + [np.array([flip_tens])],
    #         shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    #     return mpo_0

    ### transform data and 1D TNs ###

    # @classmethod
    # def transform_vector(cls, ndarray, axis=0):
    #     """ [0, 1, ..., L/2, ... L-1] --> [L/2-1, ..., 0, L/2, ... L-1]
    #         assumes ndarray.ndim = dimensionality of system (K)
    #         flip_lr not performed here
    #     """
    #     data = np.moveaxis(ndarray, axis, 0)
    #     data_shape = data.shape
    #     L = data_shape[0]
    #     new_inds = tuple(range(L // 2 - 1, -1, -1))
    #     data = data.reshape((2, L // 2) + data_shape[1:])
    #     data0 = data[0]
    #     data1 = data[1, new_inds]
    #     data = np.array([data0, data1]).reshape(data_shape)
    #     data = np.moveaxis(data, 0, axis)
    #     return data
    #
    # @classmethod
    # def inverse_transform_vector(cls, ndarray, axis=0):
    #     """ [L/2-1, ..., 0, L/2, ... L-1]  --> [0, 1, ..., L/2, ... L-1]
    #         assumes ndarray.ndim = dimensionality of system (K)
    #     """
    #     return cls.transform_vector(ndarray, axis=axis)
    #
    # @classmethod
    # def transform_operator(cls, ndarray, axis1=0, axis2=1):
    #     """ [0, 1, ..., L/2, ... L-1] --> [L/2-1, ..., 0, L/2, ... L-1]
    #         for input and output legs
    #         assumes ndarray.ndim = dimensionality of system (K)
    #     """
    #     data = ndarray
    #     for axis in [axis1, axis2]:
    #         data = np.moveaxis(data, axis, 0)
    #         data_shape = data.shape
    #         L = data_shape[0]
    #         new_inds = tuple(range(L // 2 - 1, -1, -1))
    #         data = data.reshape((2, L // 2) + data_shape[1:])
    #         data0 = data[0]
    #         data1 = data[1, new_inds]
    #         data = np.array([data0, data1]).reshape(data_shape)
    #         data = np.moveaxis(data, 0, axis)
    #     return data
    #
    # @classmethod
    # def inverse_transform_operator(cls, ndarray, axis1=0, axis2=1):
    #     """ [L/2-1, ..., 0, L/2, ... L-1]  --> [0, 1, ..., L/2, ... L-1]
    #         for input and output legs
    #         assumes ndarray.ndim = dimensionality of system (K)
    #     """
    #     return cls.transform_operator(ndarray, axis1, axis2)

    @classmethod
    def transform_mps(cls, mps: 'MPSType'):
        new_mps = super().transform_mps(mps)    # should be a different object
        # mirror_mpo = super()._twos_complement_mpo(mps.L, site_tag_id=mps.site_tag_id)
        # new_mps = helper.apply(mirror_mpo, mps, compress=True)
        helper.mps_flip_lr(new_mps, inplace=True)
        return new_mps

    def inverse_transform_mps(cls, mps: 'MPSType'):
        new_mps = helper.mps_flip_lr(mps, inplace=False)
        new_mps = super().transform_mps(new_mps)
        return new_mps

    @classmethod
    def transform_mpo(cls, mpo: 'MPOType', upper_L=None, lower_L=None):
        new_mpo = super().transform_mpo(mpo)  # should be a different object
        # mirror_mpo = cls.get_map_mpo(mpo.L, mpo.phys_dim(0), site_tag_id=mpo.site_tag_id,
        #                           upper_ind_id=mpo.upper_ind_id, lower_ind_id=mpo.lower_ind_id)
        # new_mpo = helper.apply(mirror_mpo, mpo, compress=False)
        # new_mpo = helper.apply(new_mpo, mirror_mpo, compress=True)
        new_mpo = helper.mpo_flip_lr(new_mpo, inplace=True)
        return new_mpo

    def inverse_transform_mpo(cls, mpo: 'MPSType'):
        new_mpo = helper.mps_flip_lr(mpo, inplace=False)
        new_mpo = super().transform_mps(new_mpo)
        return new_mpo

    @classmethod
    def transform_tn1d(cls, tn1d: 'MPTType'):
        new_tn1d = super().transform_tn1d(tn1d) # should be a different object

        ## flip l/r inds of 1D TN
        L = new_tn1d.L
        site_ind_ids = (new_tn1d.upper_ind_id, new_tn1d.lower_ind_id) + new_tn1d.extra_ind_ids
        for site_ind_id in site_ind_ids:
            ind_map = {site_ind_id.format(i): site_ind_id.format(L - 1 - i) for i in range(L)}
            new_tn1d.reindex(ind_map, inplace=True)
        tag_map = {new_tn1d.site_tag_id.format(i): new_tn1d.site_tag_id.format(L - 1 - i) for i in range(L)}
        new_tn1d.retag(tag_map, inplace=True)

        return new_tn1d
