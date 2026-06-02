"""Generalization of quimb's 1-D tensor networks: the :class:`MatrixProductTensor`
class extends MPS/MPO to support more than two physical indices per core, as
needed for the QTT layouts used here."""
from typing import Union, Optional, Sequence
import numpy as np
import quimb.tensor as qtn
from quimb.tensor.tensor_core import TensorNetwork
from quimb.tensor.tensor_1d import TensorNetwork1D, TensorNetwork1DFlat, MatrixProductOperator, MatrixProductState


class MatrixProductTensor(MatrixProductOperator, TensorNetwork1DFlat, TensorNetwork1D, TensorNetwork):
    """ generalization of MPS, MPO to higher number of legs at each tensor
    """

    _EXTRA_PROPS = (
        '_L',
        'cyclic',
        '_site_tag_id',
        '_upper_ind_id',
        '_lower_ind_id',
        '_extra_ind_ids',  ## list of extra inds on each tensor
    )

    def __init__(self, tensors: Union[Sequence['qtn.Tensor'], TensorNetwork], site_tag_id='T({})',
                 upper_ind_id='o({})', lower_ind_id='i({})', extra_ind_ids=(), **tn_opts):
        if not isinstance(tensors, TensorNetwork):
            tensors = TensorNetwork(tensors, **tn_opts)

        tensors = tensors.view_as(qtn.MatrixProductOperator, inplace=True, L=tensors.num_tensors, cyclic=False,
                                  site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
        ## if False, it may try to call the copy() function, which causes recursion issues if tensors in a MPTensor
        super().__init__(tensors)

        self._extra_ind_ids = extra_ind_ids

    def copy(self, virtual=False, deep=False):
        """ TODO: copying returns an MPO object??
        """
        extra_ind_ids = self.extra_ind_ids
        self.view_as(qtn.MatrixProductOperator, inplace=True)
        out = self.copy(virtual=virtual, deep=deep)
        out.view_as(MatrixProductTensor, inplace=True, extra_ind_ids=extra_ind_ids)
        self.view_as(MatrixProductTensor, inplace=True, extra_ind_ids=extra_ind_ids)
        # print('out', type(out), 'self', type(self))
        return out

    # def view_like(self, like, inplace=False, **kwargs):
    #     raise NotImplementedError

    def reindex_extra_inds(self, extra_idx, new_ind):
        old_ind = self._extra_ind_ids[extra_idx]
        self.reindex({old_ind.format(i): new_ind.format(i) for i in range(self.L)}, inplace=True)
        self._extra_ind_ids = self._extra_ind_ids[:extra_idx] + (new_ind,) + self.extra_ind_ids[extra_idx + 1:]

    @property
    def extra_ind_ids(self):
        return self._extra_ind_ids



# class MatrixProductGroup(TensorNetwork):
class MatrixProductGroup(TensorNetwork1D):
    """ generalization of MPS, MPO to higher number of legs at each tensor
    """

    _EXTRA_PROPS = (
        '_L',
        'cyclic',
        '_site_tag_id',
        '_upper_ind_id',
        '_lower_ind_id',
        '_extra_ind_ids',  ## list of extra inds on each tensor
        '_internal_tags',  ## list of extra site tags (tagging the MPS/MPO in this group)
    )

    def __init__(self, L: int, inputs: Sequence[TensorNetwork1D], fixed_mpxs: Sequence[TensorNetwork1D],
                 upper_ind_id='o({})', lower_ind_id='i({})', extra_ind_ids=(),
                 virtual=True, **tn_opts):

        ## if False, it may try to call the copy() function, which causes recursion issues if tensors in a MPTensor
        self._mpxs: Sequence[TensorNetwork1D] = [*inputs, *fixed_mpxs]
        super().__init__(self._mpxs, virtual=virtual)

        self._L = L
        self._upper_ind_id = upper_ind_id
        self._lower_ind_id = lower_ind_id
        self._extra_ind_ids = extra_ind_ids
        self._inputs = inputs
        self._fixed_mpxs = fixed_mpxs
        self._internal_tags = [mpx.site_tag_id for mpx in self._mpxs]


    def copy(self, virtual=False, deep=False):
        """ TODO: copying returns an MPO object??
        """
        fixed_mpxs = [mpx.copy(virtual=virtual, deep=deep) for mpx in self._fixed_mpxs]
        inputs = [mpx.copy(virtual=virtual, deep=deep) for mpx in self._inputs]
        out = self.__class__(self._L, fixed_mpxs, inputs, self._upper_ind_id, self._lower_ind_id,
                             self._extra_ind_ids, virtual=True)
        return out

    # def view_like(self, like, inplace=False, **kwargs):
    #     raise NotImplementedError

    @property
    def upper_ind_id(self):
        return self._upper_ind_id

    @upper_ind_id.setter
    def upper_ind_id(self, value):
        ## find corresponding MPX
        ref_mpx = None
        i = 0
        while True:
            mpx = self._mpxs[i]
            if isinstance(mpx, qtn.MatrixProductOperator):
                if self._upper_ind_id == mpx.upper_ind_id:
                    mpx.upper_ind_id = value
                    break   ## there should ony be one instance
                elif self._upper_ind_id == mpx.lower_ind_id:
                    mpx.lower_ind_id = value
                    break  ## there should ony be one instance
            elif isinstance(mpx, qtn.MatrixProductState):   # this would probably never happen
                if self._upper_ind_id == mpx.site_ind_id:
                    mpx.site_ind_id = value
                    break
            if isinstance(mpx, MatrixProductTensor):
                if self._upper_ind_id in mpx.extra_ind_ids:
                    mpx.reindex_extra_inds(self._upper_ind_id, value)
                    break
            i += 1
        self._upper_ind_id = value

    @property
    def lower_ind_id(self):
        return self._lower_ind_id

    @lower_ind_id.setter
    def lower_ind_id(self, value):
        ## find corresponding MPX and change appropriate index label
        i = 0
        while True:
            mpx = self._mpxs[i]
            if isinstance(mpx, qtn.MatrixProductOperator):
                if self._lower_ind_id == mpx.lower_ind_id:
                    mpx.upper_ind_id = value
                    break  ## there should ony be one instance
                elif self._lower_ind_id == mpx.upper_ind_id:
                    mpx.upper_ind_id = value
                    break  ## there should ony be one instance
            elif isinstance(mpx, qtn.MatrixProductState):   # this would probably never happen
                if self._lower_ind_id == mpx.site_ind_id:
                    mpx.site_ind_id = value
                    break
            if isinstance(mpx, MatrixProductTensor):
                if self._lower_ind_id in mpx.extra_ind_ids:
                    idx = mpx.extra_ind_ids.index(self._lower_ind_id)
                    mpx.reindex_extra_inds(idx, value)
                    break
            i += 1
        self._lower_ind_id = value

    def reindex_extra_inds(self, extra_ind, new_ind):

        i = 0
        while True:
            mpx = self._mpxs[i]

            if isinstance(mpx, MatrixProductTensor):
                if self._lower_ind_id in mpx.extra_ind_ids:
                    idx = mpx.extra_ind_ids.index(self._lower_ind_id)
                    mpx.reindex_extra_inds(idx, new_ind)
                    break
            i += 1

        self._extra_ind_ids = self._extra_ind_ids[:idx] + (new_ind,) + self.extra_ind_ids[idx + 1:]

    @property
    def extra_ind_ids(self):
        return self._extra_ind_ids

    @property
    def internal_tags(self):
        return self._internal_tags

    @property
    def inputs(self):
        return self._inputs

    @property
    def mpxs(self):
        return self._mpxs

    def get_tensors(self, i, exclude: Sequence[str] = None):
        if exclude is None:
            exclude = []

        tensors = []
        for mpx in self._mpxs:
            if mpx.site_tag_id not in exclude:
                tensors += [*mpx.select_tensors(mpx.site_tag_id.format(i))]
        return tensors

    @property
    def exponents(self):
        return np.sum([mpx.exponent for mpx in self._mpxs])

class MatrixProductStateUSVT(MatrixProductState):
    """ MPS in L-L...-L-S(ij)-R-...-R form
        canon site i means S(ij) between sites i, i+1
    """
    _EXTRA_PROPS = (
        '_L',
        'cyclic',
        '_site_tag_id',
        '_site_ind_id',
        '_canon_site',  ## list of extra inds on each tensor
    )

    def __init__(self, tensors: Union[Sequence['qtn.Tensor'], TensorNetwork], canon_site=0, site_tag_id='T({})',
                 site_ind_id='o({})', **tn_opts):

        if not isinstance(tensors, TensorNetwork):
            tensors = TensorNetwork(tensors, **tn_opts)

        tensors = tensors.view_as(qtn.MatrixProductState, inplace=False, L=tensors.num_tensors - 1, cyclic=False,
                                  site_tag_id=site_tag_id, site_ind_id=site_ind_id)
        super().__init__(tensors)

        assert (canon_site < tensors.num_tensors - 1), 'canon_site must be an int between 0, mps.L-2'
        self._canon_site = canon_site

    @property
    def s_tag(self):
        return self.__class__._s_tag()

    @classmethod
    def _s_tag(cls):
        return '_S_'

    @classmethod
    def from_MPS(cls, mps: 'qtn.MatrixProductState', canon_site=0, cur_orthog=None, direction=1):
        """ direction = 1: do QR decomposition
            direction = -1: do LQ decomposition
        """
        mps = mps.copy()
        out = cls._split_i(mps, canon_site, cur_orthog=cur_orthog, direction=direction)
        return out

    def copy(self, virtual=False, deep=False):
        """ TODO: check class??
        """
        tensors = [t.copy() for t in self.tensors] if deep else self.tensors
        out = self.__class__(tensors, canon_site=self._canon_site, site_ind_id=self.site_ind_id,
                             site_tag_id=self.site_tag_id)
        out.exponent = self.exponent
        return out

    @property
    def canon_site(self):
        return self._canon_site

    @canon_site.setter
    def canon_site(self, i):
        self.canonize(i)

    def canonize(self, where, cur_orthog='calc', bra=None) -> 'MatrixProductStateUSVT':
        """ where cannot be a tuple
        """
        cur_orthog = self.canon_site
        if where == cur_orthog:
            return self

        self._contract_i()
        self.__class__._split_i(self, where, cur_orthog=cur_orthog, inplace=True)
        return self

    def _contract_i(self):
        cur_orthog = self._canon_site
        # print('contract i cur orthog', cur_orthog)
        self.contract_tags((self.site_tag_id.format(cur_orthog), self.s_tag), inplace=True, which='any')
        self[cur_orthog].drop_tags((self.s_tag,))
        return self

    def to_MPS(self, inplace=False) -> 'qtn.MatrixProductState':
        out = self if inplace else self.copy()
        out._contract_i()
        out.view_as(qtn.MatrixProductState, inplace=True)
        return out

    @classmethod
    def _split_i(cls, mps: 'qtn.MatrixProductState', where, cur_orthog=None, inplace=False, direction=1,
                 ) -> 'MatrixProductStateUSVT':
        mps = mps if inplace else mps.copy()
        if direction > 0:
            where = mps.L - 2 if where == mps.L - 1 else int(where)
        else:
            where = 1 if where == 0 else int(where)
        if cur_orthog is not None:
            cur_orthog = int(cur_orthog)

        mps = mps.canonize(where=where, cur_orthog=cur_orthog)  ## inplace?
        tens_i = mps[where]

        if direction > 0:
            right_bonds, left_bonds = tens_i.filter_bonds(mps[where + 1])
            tens_L, tens_S = tens_i.split(left_bonds, method='qr', absorb='right', bond_ind=right_bonds[0] + '_i')
        else:
            left_bonds, right_bonds = tens_i.filter_bonds(mps[where - 1])
            tens_L, tens_S = tens_i.split(right_bonds, method='qr', absorb='right', bond_ind=left_bonds[0] + '_i')
        tens_S.drop_tags()
        tens_S.add_tag(cls._s_tag())

        if inplace:
            mps.view_as(cls)
            tens_i.modify(data=tens_L.data, inds=tens_L.inds)
            mps.add(tens_S)
            return mps
        else:
            tensors = [t for t in mps if t is not tens_i] + [tens_L, tens_S]
            out = cls(tensors, where, site_ind_id=mps.site_ind_id, site_tag_id=mps.site_tag_id)
            out.exponent = mps.exponent
            return out

    def get_S_tensor(self) -> 'qtn.Tensor':
        return self.select_tensors((self.s_tag,))[0]


###########################################

def view_as_indexed(mpx, site_positions: Sequence[int] = None, inplace=False
                    ) -> Union['IndexedTN1D', 'IndexedMPS', 'IndexedMPO', 'MatrixProductTN']:
    mpx = mpx if inplace else mpx.copy()

    if isinstance(mpx, IndexedTN1D):
        if site_positions is not None and site_positions != mpx.site_positions:
            mpx.set_site_positions(site_positions)
        return mpx

    site_positions = list(range(mpx.L)) if site_positions is None else None

    if isinstance(mpx, MatrixProductState):
        mpx.__class__ = IndexedMPS
        mpx._site_positions = site_positions
    elif isinstance(mpx, MatrixProductOperator):
        mpx.__class__ = IndexedMPO
        mpx._site_positions = site_positions
    if isinstance(mpx, MatrixProductTN):
        for mp in mpx.mpxs:
            view_as_indexed(mp, site_positions, inplace=True)

    return mpx


class IndexedTN1D(TensorNetwork1D):

    def __init__(self, *args, site_pos: Sequence[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._site_positions = list(range(self.L)) if site_pos is None else site_pos

    @property
    def site_positions(self):
        return self._site_positions

    @site_positions.setter
    def site_positions(self, site_positions):
        self.set_site_positions(site_positions)

    def __getitem__(self, pos):  ## need to modify so that we don't take mod of pos
        """ same as super; pos is the site position already identified with that site
            ie. pos is in self.site_positions
        """
        which = 'any'
        # print('pos', pos)
        if isinstance(pos, slice):
            # ind = self.site_positions[pos]
            tags = [self.site_tag_id.format(i) for i in range(pos.start, pos.stop)]
        elif isinstance(pos, (tuple, list)):
            # ind = self.site_positions[pos]
            tags = [self.site_tag_id.format(i) for i in pos]
        elif isinstance(pos, int):
            # ind = self.site_positions[pos]
            tags = [self.site_tag_id.format(pos)]
        elif isinstance(pos, qtn.Tensor):
            tags = pos.tags
            which = 'all'
        else:
            tags = pos

        tensors = self.select_tensors(tags, which=which)

        if len(tensors) == 1:
            return tensors[0]

        return tensors

    def get_ith_site(self, pos):
        """ get's ith element in MPX (uses site positions)
        """
        which = 'any'
        if isinstance(pos, slice):
            ind = self.site_positions[pos]
            tags = [self.site_tag_id.format(i) for i in ind]
        elif isinstance(pos, int):
            ind = self.site_positions[pos]
            tags = [self.site_tag_id.format(ind)]
        elif isinstance(pos, qtn.Tensor):
            tags = pos.tags
            which = 'all'
        else:
            tags = pos

        tensors = self.select_tensors(tags, which=which)

        if len(tensors) == 1:
            return tensors[0]

        return tensors

    def maybe_convert_coo(self, x):
        if isinstance(x, int):
            return super().maybe_convert_coo(self.site_positions[x])
        elif isinstance(x, slice):
            return [super().maybe_convert_coo(ind) for ind in self.site_positions[x]]
        else:
            return x

    def gen_site_coos(self):
        return tuple(self.site_positions)

    def set_site_positions(self, site_positions=None):
        raise NotImplementedError

    def copy(self, virtual=False, deep=False):
        out = super().copy(virtual=virtual, deep=deep)
        out.__class__ = self.__class__
        out._site_positions = self._site_positions
        return out
        # raise NotImplementedError

    def site_tag(self, i):
        """ note:  defining this like so (particular site_ind(self,i) in MPS class
            causes some issues with specific functions in quimb, eg. taking the norm /
            the cantract fucntion -- must specify tags=all
        """
        # return super().site_tag(self.site_positions[i])
        # return self.site_tag_id.format(self.site_positions[i])
        return self.site_tag_id.format(i)

    def retag_sites(self, new_id, where=None, inplace=False):
        if where is None:
            where = self.site_positions
        elif isinstance(where, slice):
            where = self.slice2sites(where)
        else:
            where = [self.site_positions[where]]

        # return super().reindex_sites(new_id, where, inplace=inplace)
        return self.retag({self.site_tag_id.format(i): new_id.format(i) for i in where},
                          inplace=inplace)


class IndexedMPS(IndexedTN1D, MatrixProductState):

    def set_site_positions(self, site_positions=None):
        import helper_quimb

        if site_positions is None:
            site_positions = list(range(self.L))

        if self.site_positions != site_positions:
            # print('site positions', site_positions)
            helper_quimb.renumber_mps(self, self.site_positions, site_positions, inplace=True)
            # print('renumbered self?', self)
            self._site_positions = site_positions

    def site_ind(self, i):
        # print('site ind', self.site_positions, i)
        # return super().site_ind(self.site_positions[i])   # takes mod
        # return self.site_ind_id.format(self.site_positions[i])
        return self.site_ind_id.format(i)

    # def slice2sites(self, tag_slice):
    #     return self.site_positions[tag_slice]

    def reindex_sites(self, new_id, where=None, inplace=False):
        if where is None:
            where = self.site_positions
        elif isinstance(where, slice):
            where = self.slice2sites(where)
        else:
            where = [self.site_positions[where]]

        # return super().reindex_sites(new_id, where, inplace=inplace)
        return self.reindex({self.site_ind_id.format(i): new_id.format(i) for i in where},
                            inplace=inplace)

    def add_MPS(self, other, inplace=False, compress=False, **compress_opts):

        if self.L != other.L:
            raise ValueError("Can't add MPS with another of different length.")

        new_mps = self if inplace else self.copy()

        for ix in range(new_mps.L):
            i = new_mps._site_positions[ix]
            t1, t2 = new_mps[i], other[i]

            if set(t1.inds) != set(t2.inds):
                # Need to use bonds to match indices
                reindex_map = {}

                if ix > 0 or self.cyclic:
                    j = new_mps._site_positions[ix - 1]
                    pair = (j, i)
                    reindex_map[other.bond(*pair)] = new_mps.bond(*pair)

                if ix < new_mps.L - 1 or self.cyclic:
                    j = new_mps._site_positions[ix + 1]
                    pair = (i, j)
                    reindex_map[other.bond(*pair)] = new_mps.bond(*pair)

                t2 = t2.reindex(reindex_map)

            t1.direct_product_(t2, sum_inds=new_mps.site_ind(i))

        if compress:
            new_mps.compress(**compress_opts)

        return new_mps

    # def copy(self):
    #     out = super().copy()
    #     # out = MatrixProductState.from_TN(self, inplace=False, site_tag_id=self.site_tag_id,
    #     #                                  site_ind_id=self.site_ind_id)
    #     out.__class__ = IndexedMPS
    #     out._site_positions = self._site_positions
    #     return out


class IndexedMPO(IndexedTN1D, MatrixProductOperator):

    def set_site_positions(self, site_positions=None):
        import helper_quimb

        if site_positions is None:
            site_positions = list(range(self.L))

        if self.site_positions != site_positions:
            helper_quimb.renumber_mpo(self, self.site_positions, site_positions, inplace=True)
            self._site_positions = site_positions

    def lower_ind(self, i):
        # return super().lower_ind(self.site_positions[i])
        # return self.lower_ind_id.format(self.site_positions[i])
        return self.lower_ind_id.format(i)

    def upper_ind(self, i):
        # return super().upper_ind(self.site_positions[i])
        # return self.upper_ind_id.format(self.site_positions[i])
        return self.upper_ind_id.format(i)

    def reindex_lower_sites(self, new_id, where=None, inplace=False):
        if where is None:
            inds = self.site_positions
        else:
            start = 0 if where.start is None else where.start
            stop = self.L if where.stop is ... else where.stop
            inds = self.site_positions[start:stop]

        return self.reindex({self.lower_ind_id.format(i): new_id.format(i)
                             for i in inds}, inplace=inplace)

    def reindex_upper_sites(self, new_id, where=None, inplace=False):
        if where is None:
            inds = self.site_positions
        else:
            start = 0 if where.start is None else where.start
            stop = self.L if where.stop is ... else where.stop
            inds = self.site_positions[start:stop]

        return self.reindex({self.upper_ind_id.format(i): new_id.format(i)
                             for i in inds}, inplace=inplace)

    # def copy(self):
    #     out = MatrixProductOperator.from_TN(self, inplace=False, site_tag_id=self.site_tag_id,
    #                                         upper_ind_id=self.upper_ind_id, lower_ind_id=self.lower_ind_id)
    #     out.__class__ = IndexedMPS
    #     out._site_positions = self._site_positions
    #     return out


###########################################


class MatrixProductTN(TensorNetwork):
    """ TN of MPS * MPO
    """

    def __init__(self, mpxs: Sequence[Union['qtn.MatrixProductState', 'qtn.MatrixProductOperator',
                                            'IndexedMPS', 'IndexedMPO']] = None,
                 data: Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']] = None,
                 virtual=False, ):

        mpxs = mpxs if virtual else [mpx.copy() for mpx in mpxs]

        if data is None:
            super().__init__(mpxs, virtual=True)
        else:
            data = data if virtual else [t.copy() for t in data]
            super().__init__(data, virtual=True)
        # self.data = self
        self.mpxs = mpxs
        # self.L = mpxs[0].L
        # self.tensors = self.data.tensors

    @property
    def data(self):
        return self

    @property
    def L(self):
        return self._L

    @property
    def _L(self):
        return self.mpxs[0].L

    @_L.setter
    def _L(self, new_L):
        self.mpxs[0]._L = new_L

    @property
    def inds(self):
        return self.outer_inds()

    def copy(self, virtual=False, deep=False):
        out = self.__class__(mpxs=self.mpxs, virtual=virtual, data=self.tensors)
        out.exponent = self.exponent
        return out

    # def conj(self, mangle_inner=False, inplace=False):
    #     out = self if inplace else self.copy()
    #     out.data.conj(mangle_inner=mangle_inner, inplace=True)
    #     return out

    def view_as(self, cls_name, inplace=False, **kwargs):
        out = self if inplace else self.copy()
        if cls_name is qtn.MatrixProductState or cls_name is qtn.MatrixProductOperator:
            for i in range(self.L):
                out = out.contract(out.site_tags(i), inplace=True)
            out.view_as(cls_name, inplace=True)
            return out
        elif cls_name is qtn.TensorNetwork:
            return out.data
        else:
            raise TypeError

    def get(self, i):
        """ get tensors associated with position i
        """
        if isinstance(i, int):
            return list(self.select_tensors(tags=self.site_tag(i), which='any'))
        elif isinstance(i, (tuple, list)):
            site_tags = set()
            for ind in i:
                site_tags.update(self.site_tag(ind))
            return list(self.select_tensors(tags=site_tags, which='any'))
        else:
            site_tags = set()
            for ind in range(i.start, i.stop):
                site_tags.update(self.site_tag(ind))
            return list(self.select_tensors(tags=site_tags, which='any'))
        # return [mpx[i] for mpx in self.mpxs]

    def __getitem__(self, pos):
        return self.get(pos)

    @property
    def site_tag_id(self):
        return self.mpxs[0].site_tag_id

    def site_tag(self, i):
        tags = set()
        for mpx in self.mpxs:
            tags.add(mpx.site_tag(i))
        return tags

    def bond(self, i, j):
        """ get associated bonds
        """
        tens1 = self.select(self.site_tag(i), which='any')  # can be multiple tensors
        tens2 = self.select(self.site_tag(j), which='any')
        # inds1 = tens1.outer_inds()
        # inds2 = tens2.outer_inds()
        left, shared, right = qtn.group_inds(tens1, tens2)
        return shared
        # return [mpx.bond(i, j) for mpx in self.mpxs]

    def bond_size(self, i, j):
        """ get associated bonds
        """
        bonds = self.bond(i, j)
        bond_sizes = [self.ind_size(b) for b in iter(bonds)]
        return np.prod(bond_sizes)
        # return [mpx.bond_size(i, j) for mpx in self.mpxs]

    def max_bond(self):
        bonds = [np.prod([mpx.bond_size(i, i + 1) for mpx in self.mpxs])
                 for i in range(self.L - 1)]
        return np.max(bonds)

    def apply(self, other, check_collisions=True):
        raise NotImplementedError
        # # out = self.copy()
        # other, out = qtn.tensor_network_align(other, self, inplace=False)
        # out.add(other, virtual=True, check_collisions=check_collisions)
        # out.upper_ind_id = other.upper_ind_id
        # return out

    # def add(self, other: Union['qtn.TensorNetwork','MatrixProductTN'], virtual=False, check_collisions=True):
    #     out = self.data.copy()
    #     other_tens = other.tensors if virtual else [t.copy() for t in other.tensors]
    #     out.add(other_tens, check_collisions=check_collisions)
    #     return out


class MatrixProductStateTN(MatrixProductTN):
    """ MPS * MPO that acts like an MPS
        mpos[0] is applied first, e.g. An An-1 ... A1 A0 (times x)
    """

    def __init__(self, mps: 'qtn.MatrixProductState', mpos: list['qtn.MatrixProductOperator'] = None,
                 data: Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']] = None,
                 virtual=False,
                 site_ind_id=None):

        mpos = [] if mpos is None else mpos
        super().__init__([mps] + mpos, virtual=virtual, data=data)
        # self._operators = self.mpxs[1:]
        # self._mps = self.mpxs[0]

        if len(self._operators) > 0:
            if site_ind_id is None:
                site_ind_id = self._mps.site_ind_id

            if len(self._operators) > 0:
                self._mps.site_ind_id = site_ind_id + '_tmp_'
                self._operators[0].lower_ind_id = self._mps.site_ind_id
                qtn.tensor_network_align(*self._operators[::-1], self._mps, inplace=True)
                self._operators[-1].upper_ind_id = site_ind_id

        # print('self.data', self.data)
        # self.data = qtn.TensorNetwork([self._mps] + self._operators, virtual=True)
        self._site_ind_id = site_ind_id

    @property
    def _mps(self):
        return self.mpxs[0]

    @property
    def _operators(self):
        return self.mpxs[1:]

    @property
    def site_ind_id(self):
        return self._site_ind_id
        # return self._mps.site_ind_id      ## doesn't work

    @site_ind_id.setter
    def site_ind_id(self, new_id):
        if len(self._operators) > 0:
            last_op = self._operators[-1]
            old_id = last_op.upper_ind_id
            last_op.upper_ind_id = new_id
            site_pos = last_op.site_positions if isinstance(last_op, IndexedTN1D) else range(self.L)
            if site_pos is None:  site_pos = range(self.L)
            try:
                self.reindex({old_id.format(i): new_id.format(i) for i in site_pos}, inplace=True)
            except ValueError:
                pass
            # print('updated?', self._operators[-1].upper_ind_id)

        else:
            old_id = self._mps.upper_ind_id
            self._mps.site_ind_id = new_id
            site_pos = self._mps.site_positions if isinstance(self._mps, IndexedTN1D) else range(self.L)
            if site_pos is None:  site_pos = range(self.L)
            try:
                self.reindex({old_id.format(i): new_id.format(i) for i in site_pos}, inplace=True)
            except ValueError:
                pass

        self._site_ind_id = new_id

    def site_ind(self, i):
        return self._site_ind_id.format(i)

    def copy(self, virtual=False, deep=False) -> 'MatrixProductStateTN':
        out = self.__class__(self._mps, self._operators, virtual=virtual, site_ind_id=self.site_ind_id,
                             data=self.tensors)
        out.exponent = self.exponent
        return out

    def reindex_sites(self, new_site_ind_id, inplace=False):
        out = self if inplace else self.copy()
        if len(out._operators) > 0:
            out._operators[-1].upper_ind_id = new_site_ind_id
        else:
            out._mps.site_ind_id = new_site_ind_id
        return out

    def apply(self, other, check_collisions=True):
        out = self.copy()
        other = other.copy()
        if out.site_ind_id == other.upper_ind_id:
            other.lower_ind_id += '_tmp'
        other, out = qtn.tensor_network_align(other, self, inplace=True)
        out.add(other, virtual=True, check_collisions=check_collisions)
        out.upper_ind_id = other.upper_ind_id
        return out


class MatrixProductOperatorTN(MatrixProductTN):
    """ stacked MPOs: mpos[0] is applied first, e.g. An An-1 ... A1 A0 (times x)
    """

    def __init__(self, mpos: Sequence['qtn.MatrixProductOperator'],
                 data: Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']] = None,
                 virtual=False,
                 lower_ind_id=None, upper_ind_id=None):

        super().__init__(mpos, virtual=virtual, data=data)
        # self._operators = self.mpxs

        if len(self._operators) > 0:
            qtn.tensor_network_align(*self._operators[::-1], inplace=True)
            if lower_ind_id is not None:
                self._operators[0].lower_ind_id = lower_ind_id
            if upper_ind_id is not None:
                self._operators[-1].upper_ind_id = upper_ind_id

        # self.data = qtn.TensorNetwork(self._operators, virtual=True)
        self._upper_ind_id = self._operators[-1].upper_ind_id
        self._lower_ind_id = self._operators[0].lower_ind_id

    @property
    def _operators(self):
        return self.mpxs

    @property
    def upper_ind_id(self):
        return self._upper_ind_id

    @upper_ind_id.setter
    def upper_ind_id(self, new_id):
        self._operators[-1].upper_ind_id = new_id
        self._upper_ind_id = new_id

    @property
    def lower_ind_id(self):
        return self._lower_ind_id

    @lower_ind_id.setter
    def lower_ind_id(self, new_id):
        self._operators[0].lower_ind_id = new_id
        self._lower_ind_id = new_id

    def copy(self, virtual=False, deep=False):
        out = self.__class__(self._operators, virtual=virtual, upper_ind_id=self.upper_ind_id,
                             lower_ind_id=self.lower_ind_id, data=self.tensors)
        out.exponent = self.exponent
        return out

    def apply(self, other, check_collisions=True):
        out = self.copy()
        other = other.copy()
        if out.upper_ind_id == other.upper_ind_id:
            other.lower_ind_id += '_tmp'
        other, out = qtn.tensor_network_align(other, self, inplace=True)
        out.add(other, virtual=True, check_collisions=check_collisions)
        out.upper_ind_id = other.upper_ind_id
        return out
