"""Mirror axis map: binary quantization combined with a two's-complement fold
that mirrors the coarsest cell, following the QTT construction of Ripoll's QTT
paper."""
from functools import lru_cache
from setup_.configs import *
import helper_quimb as helper
from axis_map.map import AxisMap
from axis_map.map_binary import BinaryMap

""" binary mapping + two's complement
    equivalently, fold data in half   <---|---> and then do original transformation
"""

## binary TN class
# class MirrorMap(BinaryMap):
class MirrorMap(AxisMap):

    @classmethod
    def get_position_inds(cls, L, q, idx):
        """ returns physical bond indices (0,1) of MPS that corresponds to vector position idx
        """
        if L == 1:
            if idx > q//2:
                return [L-idx]
            else:
                return [idx]

        assert (q == 2), 'atm method only implemented for q=2'

        if idx < 0:
            idx = q**L + idx

        if q == 2:
            str_b = bin(idx)[2:]
        else:
            if idx == 0:
                str_b = [0]
            else:
                str_b = []
                while idx:
                    str_b.append(int(idx % q))
                    idx //= q
                str_b = str_b[::-1]

        if len(str_b) > L:
            print('ind not in binaryTN of len L', L)
            raise ValueError
        elif len(str_b) < L:
            str_b = '0' * (L - len(str_b)) + str_b

        ind_list = [int(s) for s in str_b]
        if ind_list[0] == 1:
            for i in range(1,L):
                ind_list[i] = (ind_list[i] + 1)%2

        return ind_list


    @classmethod
    def get_inds_position(cls, q, inds_list):
        """ returns index in vector corresponding to physical bond indices specified in inds_list
        """
        inds_list_new = inds_list[:1]
        if inds_list[0] == 1:
            for i in range(1, len(inds_list)):
                inds_list_new += [(inds_list[i] + 1) % 2]

        bstring = ''
        for b in inds_list_new:   bstring += str(b)
        i = bstring
        idx = int(i, q)
        return idx

    @classmethod
    def get_map_mpo(cls, L, q=2, site_tag_id='T({})', upper_ind_id='o({})', lower_ind_id='i({})'):
        return super()._twos_complement_mpo(L, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                                            lower_ind_id=lower_ind_id)

    @classmethod
    def get_inverse_map_mpo(cls, L, q=2, site_tag_id='T({})', upper_ind_id='o({})', lower_ind_id='i({})'):
        return super()._twos_complement_mpo(L, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                                            lower_ind_id=lower_ind_id)

    # # @lru_cache(maxsize=128)
    # @classmethod
    # def _twos_complement_mpo(cls, L, site_tag_id='T({})', upper_ind_id='o({})', lower_ind_id='i({})'):
    #     select_0 = np.array([[1., 0.], [0., 0.]])
    #     select_1 = np.array([[0., 0.], [0., 1.]])
    #     iden = np.eye(2)
    #     flip = np.array([[0., 1.], [1., 0.]])
    #     zero = np.zeros((2, 2))
    #
    #     mpo_0 = qtn.MatrixProductOperator(
    #         [np.array([select_0, select_1])] +
    #         [np.array([[iden, zero], [zero, flip]])] * (L - 2) +
    #         [np.array([iden, flip])],
    #         shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    #     return mpo_0

    ### transform data and 1D TNs ###

    @classmethod
    def transform_vector(cls, ndarray, axis=0):
        """ [0, 1, ..., L/2, ... L-1] --> [L/2-1, ..., 0, L/2, ... L-1]
            assumes ndarray.ndim = dimensionality of system (K)
        """
        data = np.moveaxis(ndarray, axis, 0)
        data_shape = data.shape
        L = data_shape[0]
        new_inds = tuple(range(L//2-1,-1,-1))
        data = data.reshape((2,L//2) + data_shape[1:])
        data0 = data[0]
        data1 = data[1,new_inds]
        data = np.array([data0,data1]).reshape(data_shape)
        data = np.moveaxis(data, 0, axis)
        return data

    @classmethod
    def inverse_transform_vector(cls, ndarray, axis=0):
        """ [L/2-1, ..., 0, L/2, ... L-1]  --> [0, 1, ..., L/2, ... L-1]
            assumes ndarray.ndim = dimensionality of system (K)
        """
        return cls.transform_vector(ndarray, axis=axis)

    @classmethod
    def transform_operator(cls, ndarray, axis1=0, axis2=1):
        """ [0, 1, ..., L/2, ... L-1] --> [L/2-1, ..., 0, L/2, ... L-1]
            for input and output legs
            assumes ndarray.ndim = dimensionality of system (K)
        """
        data = ndarray
        for axis in [axis1, axis2]:
            data = np.moveaxis(data, axis, 0)
            data_shape = data.shape
            L = data_shape[0]
            new_inds = tuple(range(L // 2 - 1, -1, -1))
            data = data.reshape((2, L // 2) + data_shape[1:])
            data0 = data[0]
            data1 = data[1, new_inds]
            data = np.array([data0, data1]).reshape(data_shape)
            data = np.moveaxis(data, 0, axis)
        return data

    @classmethod
    def inverse_transform_operator(cls, ndarray, axis1=0, axis2=1):
        """ [L/2-1, ..., 0, L/2, ... L-1]  --> [0, 1, ..., L/2, ... L-1]
            for input and output legs
            assumes ndarray.ndim = dimensionality of system (K)
        """
        return cls.transform_operator(ndarray, axis1, axis2)


    @classmethod
    def transform_mps(cls, mps: 'MPSType'):
        twos_complement = cls._twos_complement_mpo(mps.L)
        new_mps = helper.apply(twos_complement, mps, compress=True)
        return new_mps

    @classmethod
    def transform_mpo(cls, mpo: 'MPOType', upper_L=None, lower_L=None):
        lower_L = mpo.L if lower_L is None else lower_L
        upper_L = mpo.L if upper_L is None else upper_L

        twos_complement = cls._twos_complement_mpo(upper_L, mpo.site_tag_id, mpo.upper_ind_id, mpo.lower_ind_id)
        new_mpo = helper.apply(twos_complement, mpo, compress=False)
        twos_complement = cls._twos_complement_mpo(lower_L, mpo.site_tag_id, mpo.upper_ind_id, mpo.lower_ind_id)
        new_mpo = helper.apply(new_mpo,twos_complement, compress=True)
        return new_mpo

    @classmethod
    def transform_tn1d(cls, tn1d: 'MPTType'):
        tn1d = tn1d.copy()  # if inplace else tn1d.copy()
        site_ind_ids = (tn1d.upper_ind_id, tn1d.lower_ind_id) + tn1d.extra_ind_ids

        for n in range(len(site_ind_ids)):
            site_ind_id = site_ind_ids[n]
            twos_complement = cls._twos_complement_mpo(tn1d.L, tn1d.site_tag_id, f'_TMP{n}'+'{}_', site_ind_id)
            tn1d.add(twos_complement)

        for i in range(tn1d.L):
            tn1d.contract([tn1d.site_tag_id.format(i)], inplace=True)
        tn1d.fuse_multibonds(inplace=True)

        for n in range(len(site_ind_ids)):
            site_ind_id = site_ind_ids[n]
            tn1d.reindex( {f'_TMP{n}{i}_': site_ind_id.format(i) for i in range(tn1d.L)}, inplace=True )

        helper.compress(tn1d)
        # helper.compress_tens_list([tn1d[i] for i in range(tn1d.L)], inplace=True)

        return tn1d


