# from setup_.configs import *
# import numpy as np
import quimb

from setup_.defaults import *
import time
import quimb.tensor as qtn
import helper_quimb


def sum_tens(tens_list: Sequence['qtn.Tensor'], transpose_bonds=None):

    tens = tens_list[0]
    it = 0
    while tens is None:
        it += 1
        tens = tens_list[it]

    # tens = tens_list[0].copy()
    tens = tens.copy()
    for t1 in tens_list[it + 1:]:
        helper_quimb.add_tensors(tens, t1, inplace=True)

    if transpose_bonds is not None:
        tens.transpose(*transpose_bonds, inplace=True)

    return tens


def prod_tens(tens_list: Sequence['qtn.Tensor'], transpose_bonds=None):
    tens = tens_list[0].copy()
    for t1 in tens_list[1:]:
        helper_quimb.elem_mult_tensors(tens, t1, inplace=True)

    if transpose_bonds is not None:
        tens.transpose(*transpose_bonds, inplace=True)

    return tens


def sum_eff_TNs(eff_tns: Sequence[Union['qtn.TensorNetwork','qtn.Tensor']], transpose_bonds=None):

    if transpose_bonds is None:
        eff_tn = next(iter(eff_tns))
        transpose_bonds = eff_tn.outer_inds() if isinstance(eff_tn,qtn.TensorNetwork) else eff_tn.inds

    A_eff_ = None
    for A_eff_tn in eff_tns:
        if isinstance(A_eff_tn, qtn.Tensor):
            A_eff_tens = A_eff_tn
        else:
            A_eff_tens = qtn.tensor_contract(*A_eff_tn.tensors, preserve_tensor=True)
            A_eff_tens.modify(apply=lambda data: data * 10 ** A_eff_tn.exponent)

        if len(transpose_bonds) > 0:
            A_eff_tens.transpose(*transpose_bonds, inplace=True)

        if A_eff_ is None:
            A_eff_ = A_eff_tens.copy()
        else:
            A_eff_.modify(apply=lambda data: data + A_eff_tens.data)
    return A_eff_


def prod_eff_TNs(eff_tns: Sequence['qtn.TensorNetwork'], transpose_bonds=None):

    transpose_bonds = eff_tns[0].outer_inds() if transpose_bonds is None else transpose_bonds

    A_eff_ = None
    for A_eff_tn in eff_tns:
        A_eff_tens = qtn.tensor_contract(*A_eff_tn.tensors, preserve_tensor=True)
        A_eff_tens.modify(apply=lambda data: data * 10 ** A_eff_tn.exponent)
        if len(transpose_bonds) > 0:
            A_eff_tens.transpose(*transpose_bonds, inplace=True)

        if A_eff_ is None:
            A_eff_ = A_eff_tens.copy()
        else:
            A_eff_.modify(apply=lambda data: data * A_eff_tens.data)
    return A_eff_


def get_mps_matching_inds(ket: qtn.MatrixProductState, bra: qtn.MatrixProductState, i:int) -> dict[str, str]:
    """ inds mapping inds on self.bra to corresponding inds on self.ket
    """
    inds_dict = {ket.site_ind(i): bra.site_ind(i)}

    # print('self.mps inds', self.L, self.mps_inds, i)
    # b1 = bra.select_tensors(bra.site_tag_id.format(i))[0]
    # k1 = ket.select_tensors(ket.site_tag_id.format(i))[0]
    b1 = bra[i]
    k1 = ket[i]

    ## right bond
    if i < ket.L - 1:
        # inds_dict[self.ket.bond(i,i+1)] = self.bra.bond(i,i+1)
        b2 = bra[i + 1]
        k2 = ket[i + 1]

        bbond = next(iter(b1.bonds(b2)))
        kbond = next(iter(k1.bonds(k2)))
        inds_dict[kbond] = bbond

    ## left bond
    if i > 0:
        # inds_dict[self.ket.bond(i, i - 1)] = self.bra.bond(i, i - 1)
        # ind2 = self.mps_inds[i - 1]
        # b2 = self.bra.select_tensors(self.bra.site_tag_id.format(ind2))[0]
        # k2 = self.ket.select_tensors(self.ket.site_tag_id.format(ind2))[0]
        b2 = bra[i - 1]
        k2 = ket[i - 1]

        bbond = next(iter(b1.bonds(b2)))
        kbond = next(iter(k1.bonds(k2)))
        inds_dict[kbond] = bbond

    ## ancilla bonds (L-1)
    if i == ket.L - 1 or i == 0:
        items = [item for k, item in inds_dict.items()]
        anc_b = [ind for ind in bra[i].inds if ind not in items]
        anc_k = [ind for ind in ket[i].inds if ind not in inds_dict.keys()]
        for (ab, ak) in zip(anc_b, anc_k):
            inds_dict[ak] = ab

    # print('inds dict', i, inds_dict, ind1)

    return inds_dict
