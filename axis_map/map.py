"""Base class for axis quantization maps that translate between a physical grid
index along one dimension and the per-core physical indices of a QTT/MPS, and
that build the corresponding mapping MPOs. Concrete orderings (binary, mirror,
and their flipped variants) subclass :class:`AxisMap`."""
import numpy as np
import quimb.tensor as qtn

from setup_.configs import *
import helper_quimb as helper

""" mapping of grid points along 1 dimension Parent class
"""

class AxisMap:

    @classmethod
    def array_reshape(cls,ndarray,*shape):
        return ndarray.reshape(*shape)

    @classmethod
    def get_position_inds(cls,L,q,idx):
        """ returns physical bond indices (0,1) of MPS that corresponds to vector position idx
        """
        raise NotImplementedError

    @classmethod
    def get_inds_position(cls,q,inds_list):
        """ returns index in vector coresponding to physical bond indices specified in inds_list
        """
        raise NotImplementedError

    @classmethod
    def get_tens_order(cls, L):
        # return range(L)
        if cls.is_flipped():
            return range(L-1, -1, -1)
        else:
            return range(L)

    @classmethod
    def is_flipped(cls):
        return False

    @classmethod
    def get_map_mpo(cls, L, q=2, site_tag_id='T({})', upper_ind_id='o({})', lower_ind_id='i({})'):
        return None

    @classmethod
    def get_inverse_map_mpo(cls, L, q=2, site_tag_id='T({})', upper_ind_id='o({})', lower_ind_id='i({})'):
        return None

    #############

    @classmethod
    def _flip_mpo(cls, L, q=2, site_tag_id='T({})', upper_ind_id='o({})', lower_ind_id='i({})'):
        raise NotImplementedError
        # flip_tens = np.zeros((q, q))
        # for i in range(q):
        #     flip_tens[i, -1 - i] = 1.0
        #
        # mpo_0 = qtn.MatrixProductOperator(
        #     [np.array([flip_tens])] + [np.array([[flip_tens]])] * (L - 2) + [np.array([flip_tens])],
        #     shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
        # return mpo_0

    # @lru_cache(maxsize=128)
    @classmethod
    def _twos_complement_mpo(cls, L, site_tag_id='T({})', upper_ind_id='o({})', lower_ind_id='i({})'):
        select_0 = np.array([[1., 0.], [0., 0.]])
        select_1 = np.array([[0., 0.], [0., 1.]])
        iden = np.eye(2)
        flip = np.array([[0., 1.], [1., 0.]])
        zero = np.zeros((2, 2))

        mpo_0 = qtn.MatrixProductOperator(
            [np.array([select_0, select_1])] +
            [np.array([[iden, zero], [zero, flip]])] * (L - 2) +
            [np.array([iden, flip])],
            shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
        return mpo_0


    ### transform data and 1D TNs ###

    @classmethod
    def transform_vector(cls, ndarray, axis=0):
        return ndarray

    @classmethod
    def inverse_transform_vector(cls, ndarray, axis=0):
        return ndarray

    @classmethod
    def transform_operator(cls, ndarray, axis1=0, axis2=1):
        return ndarray

    @classmethod
    def inverse_transform_operator(cls, ndarray, axis1=0, axis2=1):
        return ndarray

    @classmethod
    def transform_mps(cls, mps: 'MPSType'):
        return mps

    @classmethod
    def transform_mpo(cls, mpo: 'MPOType', upper_L=None, lower_L=None):
        return mpo

    @classmethod
    def transform_tn1d(cls, tn1d: 'MPTType'):
        return tn1d

    @classmethod
    def inverse_transform_mps(cls, mps: 'MPSType'):
        return cls.transform_mps(mps)

    @classmethod
    def inverse_transform_mpo(cls, mpo: 'MPOType'):
        return cls.transform_mpo(mpo)

    @classmethod
    def inverse_transform_tn1d(cls, tn1d: 'TN1Type', *site_ind_ids):
        return cls.transform_tn1d(tn1d)







