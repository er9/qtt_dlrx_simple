"""GridTN1DComb: QTT tensor network in the comb (tree-like) layout.

Concrete :class:`~gridTN_composite.GridTN_Composite` over a :class:`~grid_comb.GridsComb`,
representing a multidimensional QTT as a comb of 1-D MPS/MPO "teeth". Provides the
comb-specific compression and the DMRG / TDVP / interpolative-DLR solver entry points for
this layout.
"""

from setup_.configs import *
from setup_.quimb_TN1D import MatrixProductStateUSVT as MPS_USVT
from setup_.quimb_TN1D import MatrixProductStateTN, MatrixProductOperatorTN
import pickle
import helper_quimb as helper
import helper_dmrg
import helper_dmrg_2
import helper_tdvp
from helper_dmrg import LinearSolver, DMRGSolver, SweepDirection
from helper_dmrg import Environment, EnvironmentSide
from helper_tdvp import TDVPSolver, TDMRGSolver
from grid import Grid
from grid1D import Grid1D
from gridTN import GridTN
from gridTN_composite import GridTN_Composite
from gridTN_1D import GridTN1D

from multiprocessing import Process, Queue

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axis import Axis
    from grid import Grid
    from grid1D import Grid1D
    from grid_comb import GridsComb


""" class to work with TN's that look like multiple MPS connected at one end by another MPS
    ie. o---o---o---o---o
        |   |   |   |   |
        x-  x-  x-  x-  x-
        |   |   |   |   |
        x-  x-  x-  x-  x-
        |   |   |   |   |
        x-  x-  x-  x-  x-
"""

UnitGridType = Union['Grid', 'Axis']
CombDataType = tuple[dict['Grid', 'GridTN1D'], MPSType] or dict['Grid', 'GridTN1D']

NUM_TYPES = (float, complex, int, np.int32, np.int64, np.complex64, np.complex128, np.float64)  # , np.float128)


class GridTN1DComb(GridTN_Composite):

    def __init__(self, grid: 'GridsComb', branches: dict['Grid', 'GridTN'] = None, spine: 'MPSType' = None,
                 exponent=0.0, sign=1.0, ax_deriv_configs: DerivativeConfiguration = None, virtual=False):
        """ branches: List of nbranch GridTN1Ds (each of some length) that are the vertical branches
                can have Nones as placeholders, in which that dim is not represented by an MPS
                each GridTN1D has an associated GridLayout object
            spine: MPS of length nbranch that connects the branches (the 'o's)
                Note: not associated with a grid
                if is None, assumes outer product between the branches
                otherwise, assumes all tensors are in correct final form
                          (ie. branches have bonds with spine)
        """
        super().__init__(grid, ax_deriv_configs=ax_deriv_configs)
        # self.grid = grid
        ## Attributes: grid, _data, # ngrids, axes, axIDs, ax_map, ndim

        ## self._data replacement

        self._branches: dict['Grid', 'GridTN'] = {}
        self._spine: 'MPSType' = None
        self._data_type: Optional['DataType'] = None
        if branches is not None:
            self._active_grids = [gk for gk, b in branches.items() if b is not None and b.data is not None]
        else:
            self._active_grids = []

        if not virtual:
            branches = {gk: b.copy(deep=True) for gk, b in branches.items()} if branches is not None else None
            spine = spine.copy() if spine is not None else None

        self.spine = spine
        self.branches = branches
        # self.data = (self.branches, self.spine)
        ## defines self.spine, self.branches

        self._exponent = exponent
        self._sign = sign

    def __repr__(self):
        return '\nbranches: ' + str(self.branches) + '\nspine: ' + str(self.spine)

    def __getitem__(self, gr: Union['Grid', int]) -> 'GridTN':
        """ return tn branch corresponding to axis i
        """
        return self.get_branch(gr)
        # return self.branches[gr]   ## allow KeyError to be raised

    def __setitem__(self, gr: Union['Grid', int], new_branch: 'GridTN'):
        """ set ith component of the field. for convenience
        """
        self.set_branch(gr, new_branch)
        # if isinstance(gr,int):
        #     gr = self.grid.grids[gr]
        # self._branches[gr] = new_branch
        # self._update_data_type([new_branch.data_type])

    @property
    def exponent(self):
        return self._exponent

    @property
    def sign(self):
        return self._sign

    def _get_data(self):
        if len(self.branches) == 0 and self.spine is None:
            # self.spine = None
            return None
        return self.branches, self.spine

    def _set_data(self, data: CombDataType):
        if data is None:
            self._branches = {}
            self._spine = None
            self._active_grids = []
            self._exponent = 0.
            self._sign = 1.
        elif isinstance(data, tuple):
            if len(data) > 0:
                self._spine = data[1]
            else:
                self.spine = None
            self._branches: dict['Grid', GridTN1D] = data[0]
            self._update_active_grids()  ## included in set_branches
        elif isinstance(data, dict):  # Dictionary of {gridID: GridTN1Ds}
            self.branches: dict['Grid', GridTN1D] = data
            # self._update_active_grids()
            self.spine: MPSType = None
        # elif isinstance(data, qtn.TensorNetwork1D):
        #     gr_1D = Grid1D(self.grid.gridID, self.grid.axes)
        #     self.branches: dict['Grid',GridTN1D] = {gr_1D: gr_1D.make_gridTN(data) }
        #     self.spine: MPSType = None
        elif isinstance(data, GridTN1DComb):
            if data.data is not None:
                if isinstance(data.spine, (float, complex)):
                    self._spine = data.spine
                else:
                    self._spine = data.spine.copy()
                    self._branches = {k: b.copy() for k, b in data.branches.items()}
                    self._update_active_grids()
                    self._exponent = data.exponent
                    self._sign = data.sign
                    self._set_data_type([self._branches[nb].data_type for nb in self._branches.keys()])
            else:
                self.data = None
        elif isinstance(data, (float,complex)):
            # print('self.grid', self.grid)
            # print('self.grid', self.grid.grids)
            self._spine = data
            # self._update_active_grids()
            # self._set_data_type([self._branches[nb].data_type for nb in self._branches.keys()])
        else:
            raise NotImplementedError(f'comb data type? {type(data)}')

    data = property(fget=_get_data, fset=_set_data)

    ## define self.spine
    def _set_spine(self, spine: 'MPSType'):
        if spine is None:
            if self.num_active_branches > 0:
                self._init_none_spine()
                # spine = helper.ones_mps(self.grid.ngrids, 1, site_ind_id='tt({})', site_tag_id='TT({})')
                # for gr in self.grid.grids:
                #     if gr not in self._active_grids:
                #         idx = self.grid.get_grid_ind(gr)
                #         spine[idx].isel({spine.site_ind_id.format(idx):0}, inplace=True)
        else:
            assert (spine.L == self.grid.ngrids), 'spine length and num branches need to be equal'
        self._spine = spine

    def _init_none_spine(self):
        # active_grids = self._active_grids if active_grids is None else active_grids
        spine = helper.ones_mps(self.grid.ngrids, 1, site_ind_id='tt({})', site_tag_id='TT({})')
        for gr in self.grid.grids:
            if gr not in self._active_grids:
                idx = self.grid.get_grid_ind(gr)
                spine[idx].isel({spine.site_ind_id.format(idx): 0}, inplace=True)
        self._spine = spine
        return spine

    def _get_spine(self):
        if self._spine is None:
            return self._spine
        # elif len(self._active_grids) == 0 and self._spine.exponent == 0. and self._sign == 1.:
        #     return None
        else:
            return self._spine

    spine = property(fget=_get_spine, fset=_set_spine)

    ## define self.branches
    def _get_branches(self):
        return self._branches

    def _set_branches(self, branches: dict['Grid', 'GridTN']):
        ## branches = None equivalent to empty dictionary
        if branches is not None and len(branches) > 0:
            if self.spine is None:
                self._init_none_spine()
            for gk, branch in branches.items():
                self.set_branch(gk, branch)
        self._update_active_grids()

    branches = property(fget=_get_branches, fset=_set_branches)

    ## update spine site_ind_ids
    def _get_spine_ind_id(self):
        return self.spine.site_ind_id if self.spine is not None else None

    def _update_spine_ind_id(self, new_spine_ind):
        """ reindex spine inds and update branch inds. an inplace operation
        """
        old_ind = self.spine.site_ind_id
        self.spine.site_ind_id = new_spine_ind
        for br in self._get_active_branches():
            tens0 = br.get_anchor_tens()
            idx = self.grid.get_grid_ind(br.grid)
            tens0.reindex({old_ind.format(idx): new_spine_ind.format(idx)}, inplace=True)

    spine_ind_id = property(fget=_get_spine_ind_id, fset=_update_spine_ind_id)

    def _update_active_grids(self):
        self._active_grids = [gk for gk, b in self._branches.items() if b is not None and b.data is not None]

    def _get_active_branches(self) -> list['GridTN']:
        return [self.branches[gr] for gr in self._active_grids]

    def _get_active_grids(self) -> list['Grid']:
        return self._active_grids

    def _get_nbranches(self):
        return len(self._active_grids)

    num_active_branches = property(fget=_get_nbranches)
    active_branches = property(fget=_get_active_branches)
    active_grids = property(fget=_get_active_grids)

    def _get_data_type(self):
        return self._data_type

    def _set_data_type(self, data_types):
        is_mps = [dc == DataType.MPS for dc in data_types]
        is_mpo = [dc == DataType.MPO for dc in data_types]
        is_tn3 = [dc == DataType.TN3 for dc in data_types]

        if all(is_mps):
            self._data_type = DataType.MPS
        elif all(is_mpo):
            self._data_type = DataType.MPO
        elif np.all(np.array(is_mpo) + np.array(is_mps)):
            self._data_type = DataType.MPX
        elif all(is_tn3):
            self._data_type = DataType.TN3

    def _update_data_type(self, data_types):
        orig_data_type = self._data_type

        self._set_data_type(data_types)
        new_data_type = self._data_type

        if orig_data_type is None or orig_data_type == new_data_type:
            pass
        elif (orig_data_type == DataType.MPS and new_data_type == DataType.MPO) or \
                (orig_data_type == DataType.MPO and new_data_type == DataType.MPS):
            self._data_type = DataType.MPX
        else:
            raise TypeError

    data_type = property(fget=_get_data_type)

    def get_branch(self, nb: Union[int, 'Grid']) -> Optional['GridTN']:
        if isinstance(nb, (int, np.int32, np.int64)):
            nb = self.grid.grids[nb]

        return self._branches.get(nb, None)
        # # print(self._branches[nb])
        # try:
        #     return self._branches[nb]
        # except KeyError:
        #     return None

    # def get_branch_num(self, gr):
    #     return self.grid.grids.index(gr)
    ### get grid ind

    def set_branch(self, nb: Union[int, 'Grid'], new_branch: 'GridTN'):
        if new_branch is None or new_branch.data is None:
            return

        if self.spine is None:
            self._init_none_spine()

        if isinstance(nb, (int, np.int32, np.int64)):
            idx = nb
            gr = self.grid.grids[idx]
        else:
            gr = nb
            idx = self.grid.get_grid_ind(gr)

        site_ind = self.spine.site_ind(idx)
        tens0 = new_branch.get_anchor_tens()
        # print('set branch', self.grid.axes, nb, idx)
        # print('comb TN', self)

        ## adding new branch
        if gr not in self._active_grids:
            spine_tens = self.spine[idx]
            tens0 = new_branch.get_anchor_tens()
            spine_tens.new_bond(tens0, name=site_ind)
            self._active_grids += [gr]
        else:
            if site_ind not in tens0.inds:  ## ie. adding new branch
                tens0.new_ind(site_ind, size=self.spine.phys_dim(idx))
            else:  ## replacing existing branch
                assert (tens0.ind_size(site_ind)), 'spine and branch need same ind size'

        self._branches[gr] = new_branch
        self._update_data_type([new_branch.data_type])

    def create_like(self, new_data=None, exponent=0.0, sign=1.0) -> 'GridTN1DComb':
        """ create a copy but without any data
        """
        if isinstance(new_data, tuple):
            new_branches, new_spine = new_data
        else:
            new_branches = new_data
            new_spine = None

        new_tnc = self.__class__(self.grid, branches=new_branches, spine=new_spine, exponent=exponent, sign=sign,
                                 ax_deriv_configs=self.ax_deriv_configs)
        return new_tnc

    def copy(self, deep=True) -> 'GridTN1DComb':
        """ return a copy (copy MPO data)
            if not deep, don't copy the data. default is deep=True
        """
        if self.grid.ngrids == 0:
            new_tnc = self.grid.make_gridTN(self.spine)     ## spine is a float value
        else:
            branches = {gk: b.copy(deep=deep) for gk, b in self.branches.items()}
            spine = self.spine.copy() if (deep and self.spine is not None) else self.spine
            new_tnc = self.create_like(new_data=(branches, spine), exponent=self.exponent, sign=self.sign)
            # new_tnc = self.__class__(self.grid, branches=branches, spine=spine, exponent=self.exponent, sign=self.sign,
            #                          ax_deriv_configs=self.ax_deriv_configs)
            new_tnc.is_constant = self.is_constant
            new_tnc.constant_axes = self.constant_axes
            new_tnc.canon_site = self.canon_site
        return new_tnc

    def conj(self, inplace=False, mangle_inner=False) -> 'GridTN1DComb':
        new_gtn = self if inplace else self.copy()

        if new_gtn.spine is not None:
            new_gtn.spine.conj(mangle_inner=mangle_inner, inplace=True)
        for gk, b in new_gtn.branches.items():
            b.conj(mangle_inner=mangle_inner, inplace=True)

        new_gtn._sign = np.conj(new_gtn._sign)
        return new_gtn

    def save_data(self, fstr):
        branch_data = []
        for gk in self.grid.grids:
            branch = self.get_branch(gk)
            if branch is not None:
                branch_data += [self.get_branch(gk).data]
            else:
                branch_data += [None]

        data = (self._exponent, self._sign, self.spine, branch_data)
        pickle.dump(data, open(fstr + '.pkl', 'wb'))
        print('saved data', fstr + '.pkl')

    def max_bond(self) -> int:
        max_bonds = [b.max_bond() for b in self.active_branches]
        if self.spine is not None:
            max_bonds += [self.spine.max_bond()]
        if len(max_bonds) > 0:
            return max(max_bonds)
        else:
            return np.nan

    def all_virtual_sizes(self) -> int:
        if self.data is not None:
            all_bonds = [b.all_virtual_sizes() for b in self.active_branches]
            if self.spine is not None:
                all_bonds += [[self.spine.bond_size(i,i+1) for i in range(self.spine.L - 1)]]
            return all_bonds
        else:
            return []


    def all_smallest_singular_values(self, max_bond=None) -> list[Numeric]:
        min_svals = []
        self.canonize(inplace=True, i=0)
        cur_orthog = 0

        svals_all = helper.singular_values_all(self.spine.copy())
        for svals in svals_all:
            if max_bond is not None and len(svals) < max_bond:
                min_svals += [0] 
            else:
                min_svals += [svals[-1]] 
        for b in self.active_branches:
            gk_idx = self.grid.get_grid_ind(b.grid)
            helper.canonize(self.spine, i=gk_idx, cur_orthog=cur_orthog)
            cur_orthog = gk_idx
            _, left_inds = self.spine[gk_idx].filter_bonds(b.get_anchor_tens())

            spine_singvals = self.spine[gk_idx].singular_values(left_inds=left_inds)
            if max_bond is not None and len(spine_singvals) < max_bond:
                min_svals += [0] 
            else:
                min_svals += [spine_singvals[-1]] 

            # print('spine cur orthog', gk_idx, self.spine.calc_current_orthog_center())
            helper.canonize_tens_list(self.spine[gk_idx], b.get_anchor_tens(), inplace=True)
            # min_svals += helper.singular_values_all(b.data.copy())[1:]
            min_svals += b.all_smallest_singular_values(max_bond=max_bond)
            helper.canonize_tens_list(b.get_anchor_tens(), self.spine[gk_idx], inplace=True)
            # print('branch cur orthog', b.data.calc_current_orthog_center())
        return min_svals


    def entanglement_entropy_all(self) -> list[Numeric]:
        EEs = []
        self.canonize(inplace=True, i=0)
        cur_orthog = 0
        if self.data is None:
            return []

        EEs += helper.entanglement_entropy_all(self.spine.copy())
        for b in self.active_branches:
            gk_idx = self.grid.get_grid_ind(b.grid)
            helper.canonize(self.spine, i=gk_idx, cur_orthog=cur_orthog)
            cur_orthog = gk_idx
            _, left_inds = self.spine[gk_idx].filter_bonds(b.get_anchor_tens())

            spine_singvals = self.spine[gk_idx].singular_values(left_inds=left_inds)
            spine_singvals *= 1. / np.linalg.norm(spine_singvals)
            EE = -1 * np.sum(spine_singvals ** 2 * np.log(spine_singvals ** 2))
            EEs += [EE]

            # print('spine cur orthog', gk_idx, self.spine.calc_current_orthog_center())
            helper.canonize_tens_list(self.spine[gk_idx], b.get_anchor_tens(), inplace=True)
            EEs += helper.entanglement_entropy_all(b.data.copy()) # [1:]
            helper.canonize_tens_list(b.get_anchor_tens(), self.spine[gk_idx], inplace=True)
            # print('branch cur orthog', b.data.calc_current_orthog_center())
        return EEs

    def check_orthog(self):
        for gk, branch in self.branches.items():
            print('branch orthog', gk)
            gk_idx = self.grid.get_grid_ind(gk)
            branch_ext = self._attach_spine_tens_to_branch(gk_idx, is_mps=(self.data_type == DataType.MPS))
            helper.check_orthog(branch_ext)  # , left_ancillas=(self.spine_ind_id.format(gk_idx),))
        print('spine orthog')
        helper.check_orthog(self.spine)
        return

    def mangle_inner(self, inplace=True, append=None):
        out = self if inplace else self.copy()
        for k, b in out.branches.items():
            b.data.mangle_inner_(append=append)
        out.spine.mangle_inner_(append=append)
        return out

    def collect_exponents(self):
        """ in place collection of all exponents
        """
        if self.grid.ngrids == 0:
            return 0.0

        for gk, branch in self.branches.items():
            if branch is not None and branch.data is not None:
                self._exponent += branch.data.exponent
                branch.data.exponent = 0.0
        if self.spine is not None:
            self._exponent += self.spine.exponent
            self.spine.exponent = 0.0
        return self._exponent

    def distribute_exponents(self, val=None, across_all=False):
        """ absorb exponent into self
        """
        if self.grid.ngrids == 0:
            return self

        self.collect_exponents()
        if val is None:
            val = self._exponent

        if self.canon_site is None or across_all:
            active = self.active_branches
            if len(active) > 0:
                b_val = val * (1. / len(active))
                for branch in active:
                    branch.data.exponent += b_val
                    branch.data.distribute_exponent()
                self._exponent -= val

            else:       ## distribute across spine
                self.spine.exponent = val
                self.spine.distribute_exponent()
                self._exponent -= val

        else:
            tens_active = self.spine[self.canon_site]
            tens_active.modify(apply=lambda x: x * 10 ** val)
            self._exponent -= val

        return

    def distribute_sign(self):
        """ put sign into spine
        """
        # helper.scalar_multiply(self.spine, self.sign, inplace=True)
        self.spine[0].modify(apply=lambda x: x * self.sign)
        self._sign = 1.0


    def transpose(self, inplace=True, mangle_inner=False):
        """ take transpose of MPO
        """
        gtn = self if inplace else self.copy()
        for gr, branch in self.branches.items():
            helper.mpo_flip_upper_lower(branch.data, inplace=True, mangle_inner=mangle_inner)
        return gtn

    def get_like_iden(self):
        branches = {gr: branch.get_like_iden() for gr, branch in self.branches.items()}
        return self.create_like(branches)

    ############################
    # fcts to generate tncombs #
    ############################

    @classmethod
    def get_ones_mps(cls, cgrid: 'GridsComb', site_ind_id='i({})', site_tag_id='X({})') -> 'GridTN1DComb':
        """ build ones vector mps on specified grid
        """
        branches = {}
        for gl in cgrid.grids:
            branch = gl.get_ones_mps(site_ind_id=site_ind_id, site_tag_id=site_tag_id)
            branches[gl] = branch
        return cls(cgrid, branches=branches)

    @classmethod
    def get_iden_mpo(cls, cgrid: 'GridsComb', upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') \
            -> 'GridTN1DComb':
        """ build identity mpo on specified grid
        """
        branches = {}
        for gl in cgrid.grids:
            branch = gl.get_iden_mpo(upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                     site_tag_id=site_tag_id)
            branches[gl] = branch
        return cls(cgrid, branches=branches)

    @classmethod
    def get_select_elems_mps(cls, cgrid: 'GridsComb', inds: Sequence[int], site_ind_id='i({})', site_tag_id='X({})') \
            -> 'GridTN1DComb':
        """ build MPO to select certain elements specified by inds
            grids:  list of GridLayout objects to build MPS from
            inds:   list of ints specifying position for each GridLayout obj. Position for multi-dim grids should
                    be tuples or lists of the inds along each Axis. Indices for 1-dimensional grids can be ints
        """
        branches = {}
        assert (cgrid.ndim == len(inds)), 'number of inds needs to match total number of Axis objects in grids'

        ind0 = 0
        for gr in cgrid.grids:
            active_inds = inds[ind0:ind0 + gr.ndim]
            ind0 += gr.ndim
            branch = gr.get_select_elems_mps(active_inds, site_ind_id=site_ind_id, site_tag_id=site_tag_id)
            branches[gr] = branch

        return cls(cgrid, branches=branches)

    @classmethod
    def get_select_elem_mpo(cls, cgrid: 'GridsComb', inds: list[int], upper_ind_id='i({})', lower_ind_id='o({})',
                            site_tag_id='X({})') -> 'GridTN1DComb':
        """ build MPO to select certain elements specified by inds
            grids:  list of GridLayout objects to build MPS from
            inds:   list of ints specifying position for each GridLayout obj. Position for multi-dim grids should
                    be tuples or lists of the inds along each Axis. Indices for 1-dimensional grids can be ints
        """
        branches = {}
        assert (cgrid.ndim == len(inds)), 'number of inds needs to match total number of axis in grids'

        ind0 = 0
        for gr in cgrid.grids:
            active_inds = inds[ind0:ind0 + gr.ndim]
            ind0 += gr.ndim
            branch = gr.get_select_elems_mpo(active_inds, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                             site_tag_id=site_tag_id)
            branches[gr] = branch

        return cls(cgrid, branches=branches)

    @classmethod
    def from_dense_state(cls, data, grid: 'GridsComb', site_ind_id='i({})', site_tag_id='T({})',
                         tt_tag_id='TT({})', tt_ind_id='tt({})', split_opts=None,
                         ax_deriv_configs=None, axes=None) -> 'GridTN1DComb':
        """ decompose dense nb-dimensional array along each dimension (Tensor Train decomposition)
            data: nbranches-dimensional data
            grids: list of GridLayout objects corresponding to each dimension
            site_ind_id: root site_ind_id for the branches
            site_tag_id: root tag id for the TNComb
            **tt_split_opts: parameters for tensor_split
            axes: specifies which axis each dimension of data corresponds to. None defaults to grid.
        """
        L = grid.ngrids
        if split_opts is None:   split_opts = {}
        tt_split_opts = split_opts.pop('tt_compress_opts', split_opts)
        branch_split_opts = split_opts
        # print('tt_split opts', tt_split_opts)
        # print('branch split opts', branch_split_opts)

        ## sort axes by grid
        if axes is not None:
            grid_axes = grid.axes
            ax_inds = [grid_axes.index(ax) for ax in axes]
            sort_inds = np.argsort(ax_inds)
            data = data.transpose(sort_inds)
            axes = [axes[s] for s in sort_inds]
        else:
            axes = grid.axes
            ax_inds = list(range(len(axes)))

        spine_tensors, branches = [], {}
        ix = 0
        left_anc = []
        exponent = 0.0
        for gr in grid.grids:
            subgr_axes = [ax for ax in axes[:gr.ndim] if ax in gr.axes]
            num_active = len(subgr_axes)

            if num_active == 0:
                ## just add spine tensor that is the identity? with correct indices. change left anc
                raise NotImplementedError
                continue

            ## move active dims so that all ancilla are at the left
            num_anc_l = len(left_anc)
            current = list(range(num_anc_l, num_anc_l + num_active))
            target = list(range(-num_active,0))
            data = np.moveaxis(data, current, target)

            # remaining_inds = range(len(subgr_axes), len(axes))
            ax_inds = ax_inds[num_active:]
            axes = axes[num_active:]

            left_ancilla_inds = tuple(left_anc + [f'tmp_dim({i})' for i in ax_inds])
            left_ancilla_shape = data.shape[:-num_active]
            out = gr.build_mps_from_subgtns([(subgr_axes, data)], site_ind_id=site_ind_id + f',g({gr.gridID})',
                                            site_tag_id=site_tag_id + f',g({gr.gridID})',
                                            ancilla_left=left_ancilla_shape, ancilla_left_inds=left_ancilla_inds,
                                            compress_opts=branch_split_opts)

            if out is None or out.data is None:
                continue

            ## ancilla info appears in the first tensor
            tens0 = out.data[0]
            left_inds = [out.data.bond(0, 1), out.data.site_ind_id.format(0)]
            tens0, rest = qtn.tensor_split(tens0, left_inds, absorb='right', bond_ind=tt_ind_id.format(ix),
                                           **tt_split_opts)
            out.data[0].modify(data=tens0.data, inds=tens0.inds)
            branches[gr] = out  # gr.pad_gtn_to_grid(out)

            if len(axes) > 0:
                spine_tens, rest = qtn.tensor_split(rest, left_inds=([tt_ind_id.format(ix)] + left_anc),
                                                    absorb='right', **tt_split_opts)
                left_anc, _ = rest.filter_bonds(spine_tens)
                rest.transpose(*left_anc, *[f'tmp_dim({i})' for i in ax_inds], inplace=True)
            else:
                norm = rest.norm()
                exponent += np.log10(norm)
                spine_tens = rest
                spine_tens.modify(apply=lambda x: x / norm)
            spine_tens.drop_tags()
            spine_tens.add_tag(tt_tag_id.format(ix))
            spine_tensors += [spine_tens]

            ix += 1
            data = rest.data

        ## decompose into an MPS using quimb
        if len(spine_tensors) > 0:
            spine = qtn.TensorNetwork(spine_tensors)
            spine.view_as(qtn.MatrixProductState, cyclic=False, L=len(spine_tensors),
                          site_tag_id=tt_tag_id, site_ind_id=tt_ind_id, inplace=True)
        else:
            spine = None

        # spine = qtn.TensorNetwork([])
        # branches = {}
        # for nb in range(L):
        #
        #     gr: 'Grid' = grid.grids[nb]
        #
        #     if gr is None:  ## no grid corresponding to this dim
        #         ## should probably be some placeholder subclass of Grid
        #         if nb < L - 1:
        #             helper.canonize_tens_list(tt_format[nb], tt_format[nb + 1], inplace=True)
        #         spine.add([tt_format[nb]])
        #         continue
        #
        #     if nb == 0 and L == 1:
        #         nlegs = 0  ## no legs along spine
        #         inds = (tt_format.site_ind_id.format(nb),)
        #     elif nb == 0:
        #         nlegs = 1  ## left-most TT tensor has only 1 leg along spine
        #         inds = (tt_format.bond(nb, nb + 1), tt_format.site_ind_id.format(nb))
        #     elif nb == L - 1:
        #         nlegs = 1  ## right-most TT tensor has only 1 leg along spine
        #         inds = (tt_format.bond(nb - 1, nb), tt_format.site_ind_id.format(nb))
        #     else:
        #         nlegs = 2
        #         inds = (tt_format.bond(nb - 1, nb), tt_format.bond(nb, nb + 1),
        #                 tt_format.site_ind_id.format(nb))
        #
        #     branch_tens = tt_format[nb].transpose(*inds, inplace=False)
        #     branch_data = branch_tens.data
        #     shape_lr = branch_data.shape[:nlegs]  # l/r legs in spine mps
        #     # anc_ind_ids = tuple([f'anc{i}' for i in range(nlegs)])
        #     anc_ind_ids = inds[:nlegs]
        #     # print('anc ind ids', anc_ind_ids, shape_lr)
        #
        #     ## fold each axis into d-ary form eg. (q x q x q ...) ** self.dim
        #     # branch_data = branch_data.reshape(shape_lr+gr.shape)
        #
        #     # # new_inds = [site_ind_id.format(x) + f',g({gr.gridID})' for x in range(gr.L)]
        #     # # new_tens = qtn.Tensor( data=branch_data, inds=shape_lr+new_inds )
        #
        #     shape_branch = tuple([ax.npts for ax in gr.axes])
        #     branch_data = branch_data.reshape(shape_lr + shape_branch)
        #     gtn_mps = gr.map_state_to_mps(branch_data, site_ind_id=site_ind_id + f',g({gr.gridID})',
        #                                   site_tag_id=site_tag_id + f',g({gr.gridID})',
        #                                   direction=1, split_opts=branch_split_opts,
        #                                   ancilla_left=shape_lr, ancilla_left_inds=anc_ind_ids)
        #     if len(anc_ind_ids) > 0:
        #         tens0 = gtn_mps.data[0]
        #         TL, TR = tens0.split(anc_ind_ids, get='tensors', bond_ind=tt_ind_id.format(nb),
        #                              absorb='left', **branch_split_opts)
        #         tens0.modify(data=TR.data, inds=TR.inds)
        #         TL.drop_tags()
        #         TL.add_tag(tt_tag_id.format(nb))
        #
        #         if nb < L - 1:  # push singular values into next tensor in spine
        #             helper.compress_tens_list(TL, tt_format[nb + 1], inplace=True, compress_opts=tt_split_opts)
        #
        #         spine.add([TL])  # the remaining tensor containing L,R
        #
        #     branches[gr] = gtn_mps
        #
        # if spine.num_tensors > 0:
        #     spine = spine.view_like(tt_format, inplace=True)
        # else:
        #     spine = None

        # print('spine', spine)
        # for k, b in branches.items():
        #     print('branch', k, b)

        state_tn = cls(grid, branches=branches, spine=spine, exponent=exponent,
                       ax_deriv_configs=ax_deriv_configs)
        return state_tn


    # @classmethod
    # def from_dense_state(cls, data, grid: 'GridsComb', site_ind_id='i({})', site_tag_id='T({})',
    #                      tt_tag_id='TT({})', tt_ind_id='tt({})', split_opts=None,
    #                      ax_deriv_configs=None) -> 'GridTN1DComb':
    #     """ decompose dense nb-dimensional array along each dimension (Tensor Train decomposition)
    #         data: nbranches-dimensional data
    #         grids: list of GridLayout objects corresponding to each dimension
    #         site_ind_id: root site_ind_id for the branches
    #         site_tag_id: root tag id for the TNComb
    #         **tt_split_opts: parameters for tensor_split
    #     """
    #     L = grid.ngrids
    #     if split_opts is None:   split_opts = {}
    #     tt_split_opts = split_opts.pop('tt_compress_opts', split_opts)
    #     branch_split_opts = split_opts
    #     # print('tt_split opts', tt_split_opts)
    #     # print('branch split opts', branch_split_opts)
    #
    #     ## decompose into an MPS using quimb
    #     dims = [gr.npts for gr in grid]
    #     data = data.reshape(dims)
    #     tt_format = cls.construct_spine(L, data, tt_tag_id=tt_tag_id, tt_ind_id=tt_ind_id, split_opts=tt_split_opts)
    #     if tt_format is None:
    #         return cls(grid)
    #
    #     spine = qtn.TensorNetwork([])
    #     branches = {}
    #     for nb in range(L):
    #
    #         gr: 'Grid' = grid.grids[nb]
    #
    #         if gr is None:  ## no grid corresponding to this dim
    #             ## should probably be some placeholder subclass of Grid
    #             if nb < L - 1:
    #                 helper.canonize_tens_list(tt_format[nb], tt_format[nb + 1], inplace=True)
    #             spine.add([tt_format[nb]])
    #             continue
    #
    #         if nb == 0 and L == 1:
    #             nlegs = 0  ## no legs along spine
    #             inds = (tt_format.site_ind_id.format(nb),)
    #         elif nb == 0:
    #             nlegs = 1  ## left-most TT tensor has only 1 leg along spine
    #             inds = (tt_format.bond(nb, nb + 1), tt_format.site_ind_id.format(nb))
    #         elif nb == L - 1:
    #             nlegs = 1  ## right-most TT tensor has only 1 leg along spine
    #             inds = (tt_format.bond(nb - 1, nb), tt_format.site_ind_id.format(nb))
    #         else:
    #             nlegs = 2
    #             inds = (tt_format.bond(nb - 1, nb), tt_format.bond(nb, nb + 1),
    #                     tt_format.site_ind_id.format(nb))
    #
    #         branch_tens = tt_format[nb].transpose(*inds, inplace=False)
    #         branch_data = branch_tens.data
    #         shape_lr = branch_data.shape[:nlegs]  # l/r legs in spine mps
    #         # anc_ind_ids = tuple([f'anc{i}' for i in range(nlegs)])
    #         anc_ind_ids = inds[:nlegs]
    #         # print('anc ind ids', anc_ind_ids, shape_lr)
    #
    #         ## fold each axis into d-ary form eg. (q x q x q ...) ** self.dim
    #         # branch_data = branch_data.reshape(shape_lr+gr.shape)
    #
    #         # # new_inds = [site_ind_id.format(x) + f',g({gr.gridID})' for x in range(gr.L)]
    #         # # new_tens = qtn.Tensor( data=branch_data, inds=shape_lr+new_inds )
    #
    #         shape_branch = tuple([ax.npts for ax in gr.axes])
    #         branch_data = branch_data.reshape(shape_lr + shape_branch)
    #         gtn_mps = gr.map_state_to_mps(branch_data, site_ind_id=site_ind_id + f',g({gr.gridID})',
    #                                       site_tag_id=site_tag_id + f',g({gr.gridID})',
    #                                       direction=1, split_opts=branch_split_opts,
    #                                       ancilla_left=shape_lr, ancilla_left_inds=anc_ind_ids)
    #         if len(anc_ind_ids) > 0:
    #             tens0 = gtn_mps.data[0]
    #             TL, TR = tens0.split(anc_ind_ids, get='tensors', bond_ind=tt_ind_id.format(nb),
    #                                  absorb='left', **branch_split_opts)
    #             tens0.modify(data=TR.data, inds=TR.inds)
    #             TL.drop_tags()
    #             TL.add_tag(tt_tag_id.format(nb))
    #
    #             if nb < L - 1:  # push singular values into next tensor in spine
    #                 helper.compress_tens_list(TL, tt_format[nb + 1], inplace=True, compress_opts=tt_split_opts)
    #
    #             spine.add([TL])  # the remaining tensor containing L,R
    #
    #         branches[gr] = gtn_mps
    #
    #     if spine.num_tensors > 0:
    #         spine = spine.view_like(tt_format, inplace=True)
    #     else:
    #         spine = None
    #
    #     state_tn = cls(grid, branches=branches, spine=spine, exponent=tt_format.exponent,
    #                    ax_deriv_configs=ax_deriv_configs)
    #     return state_tn

    @classmethod
    def from_dense_operator(cls, data, grid: 'GridsComb', upper_ind_id='o({})', lower_ind_id='i({})',
                            site_tag_id='T({})', split_opts=None, tt_tag_id='TT({})', tt_ind_id='tt({})',
                            **kwargs) -> 'GridTN1DComb':
        """ decompose dense nbranch-dimensional array along each dimension (Tensor Train decomposition)
                operator: o0 x o1 x ... i0 x i1 x ... where 0,1,.. refers to dimension/branch
                each branch is then decomposed into MPO (o00 x o01 x... i00 x i01 x...) for 0
                mpo shape of o (or i) given by dims_list[dim]
            data: nb*2-dimensional data
            grids: list of GridLayout objects corresponding to each dimension
            upper_ind_id:  root upper_ind_id for branches
            lower_ind_id:  root lower_ind_id for branches
            site_tag_id: root tag id for the TNComb
            **split_opts: parameters for tensor_split
                also 'tt_compress_opts' is key for tt_compress_opts (eg. {'max_bond': D,'cutoff': 1.0e-10})
                compression parameters used for initial splitting into tensor train / spine
        """
        K = grid.ngrids
        if split_opts is None:   split_opts = {}
        tt_split_opts = split_opts.pop('tt_compress_opts', split_opts)
        branch_split_opts = split_opts

        ## decompose operator data into an MPS using quimb
        # axT = [2*x for x in range(K)] + [2*x+1 for x in range(K)]
        axT = [2 * x for x in range(K)] + [2 * x + 1 for x in range(K)]
        axT = np.argsort(axT)
        # print('axT', axT)
        data = data.transpose(axT)
        dims = [gl.npts ** 2 for gl in grid]
        data = data.reshape(dims)
        tt_format = cls.construct_spine(K, data, tt_tag_id=tt_tag_id, tt_ind_id=tt_ind_id, split_opts=tt_split_opts)

        spine = qtn.TensorNetwork([])
        branches = {}
        for nb in range(K):

            gr: 'Grid' = grid.grids[nb]

            if gr is None:  ## no branch corresponding to this dim
                if nb < K - 1:
                    helper.canonize_tens_list(tt_format[nb], tt_format[nb + 1], inplace=True)
                spine.add([tt_format[nb]])
                continue

            if nb == 0 and K == 1:
                nlegs = 0  ## no legs along spine
                inds = (tt_format.site_ind_id.format(nb),)
            elif nb == 0:
                nlegs = 1
                inds = (tt_format.bond(nb, nb + 1), tt_format.site_ind_id.format(nb))
            elif nb == K - 1:
                nlegs = 1
                inds = (tt_format.bond(nb - 1, nb), tt_format.site_ind_id.format(nb))
            else:
                nlegs = 2
                inds = (tt_format.bond(nb - 1, nb), tt_format.bond(nb, nb + 1),
                        tt_format.site_ind_id.format(nb))

            # print('inds', inds)
            branch_tens = tt_format[nb].transpose(*inds, inplace=False)
            shape_lr = branch_tens.shape[:nlegs]  # l/r legs in spine mps
            branch_data = branch_tens.data.reshape(shape_lr + (gr.npts, gr.npts))
            # anc_ind_ids = tuple([tt_format.site_ind_id.format(i) for i in range(nlegs)])
            anc_ind_ids = inds[:nlegs]

            ## fold each axis into d-ary form eg. (q x q x q ...) * 2
            # branch_data = branch_data.reshape(shape_lr+gr.shape()*2)
            # # new_inds_1 = [upper_ind_id.format(x) + f',g({gr.gridID})' for x in range(gr.L)]
            # # new_inds_2 = [lower_ind_id.format(x) + f',g({gr.gridID})' for x in range(gr.L)]
            # # new_tens = qtn.Tensor( data=branch_data, inds=shape_lr+new_inds_1+new_inds_2 )

            gtn_mpo = gr.map_operator_to_mpo(branch_data, upper_ind_id=upper_ind_id + f',g({gr.gridID})',
                                             lower_ind_id=lower_ind_id + f',g({gr.gridID})',
                                             site_tag_id=site_tag_id + f',g({gr.gridID})',
                                             direction=1, split_opts=branch_split_opts,
                                             ancilla_left=shape_lr, ancilla_left_inds=anc_ind_ids)
            if len(anc_ind_ids) > 0:
                tens0 = gtn_mpo.data[0]
                TL, TR = tens0.split(anc_ind_ids, get='tensors', bond_ind=tt_ind_id.format(nb),
                                     absorb='left', **branch_split_opts)
                tens0.modify(data=TR.data, inds=TR.inds)
                TL.drop_tags()
                TL.add_tag(tt_tag_id.format(nb))

                if nb < K - 1:  # push singular values into next tensor in spine
                    helper.compress_tens_list(TL, tt_format[nb + 1], inplace=True, compress_opts=tt_split_opts)

                spine.add([TL])  # the remaining tensor containing L,R

            branches[gr] = gtn_mpo

            # mpo = qtn.TensorNetwork([])
            # for x in range(gl.L-1,-1,-1):
            #     new_tens, TR = new_tens.split(shape_lr + new_inds_1[:x] + new_inds_2[:x],
            #                                   get='tensors', rtags=site_tag_id+f',g({gl.gridID})',
            #                                   absorb='left', **branch_split_opts)
            #     mpo.add(TR)
            # mpo.view_as(qtn.MatrixProductOperator, L=gl.L, cyclic=False, inplace=True,
            #                                        site_tag_id=site_tag_id+f',g({gl.gridID})',
            #                                        upper_ind_id=upper_ind_id+f',g({gl.gridID})',
            #                                        lower_ind_id=lower_ind_id+f',g({gl.gridID})')
            #
            # ## add mpo to branches list
            # grid_mpo = GridTN1D(gl, data=mpo)
            # branches += [grid_mpo]
            #
            # ## update and canonicalize spine
            # new_tens.drop_tags()
            # new_tens.add_tag(tt_tag_id.format(nb))
            # bond_tt, = new_tens.bonds(mpo[0])
            # new_tens.reindex({bond_tt: tt_ind_id.format(nb)}, inplace=True)
            #
            # if nb < L-1:
            #     helper.compress_tens_list(new_tens, tt_format[nb+1], inplace=True, **tt_split_opts)
            #
            # spine.add([new_tens])  # the remaining tensor containing L,R

        if spine.num_tensors > 0:
            spine = spine.view_like(tt_format, inplace=True)
        else:
            spine = None

        state_tn = cls(grid, branches=branches, spine=spine, exponent=tt_format.exponent)
        return state_tn

    # @classmethod
    # def pad_mps_to_grid(cls, mps_ax_dict, cgrid: GridsComb) -> 'GridTN1DComb':
    #     """ takes dictionary with axIDs as keys to create GridTN1DComb object
    #         pad empty grids with ones
    #     """
    #     axIDs = mps_ax_dict.keys()
    #     new_branches = {}
    #     active_grids = cgrid.get_active_grids(axIDs)
    #     for grid in cgrid.grids:
    #         if grid in active_grids:
    #             active_axIDs = [axID for axID in grid.axIDs if axID in mps_ax_dict]
    #             grid_mps_dict = {axID: mps_ax_dict[axID] for axID in active_axIDs}
    #             grid_mps = grid.make_mps_ndim(grid_mps_dict)
    #             new_branches[grid.gridID] = GridTN1D(grid, grid_mps)
    #         else:
    #             new_branches[grid.gridID] = grid.get_ones_mps()
    #
    #     mps_comb = cls(cgrid, branches = new_branches)
    #     return mps_comb
    #
    #
    # @classmethod
    # def pad_mpx_to_grid(cls, mps_ax_dict, cgrid: GridsComb) -> 'GridTN1DComb':
    #     """ takes dictionary with axIDs as keys to create GridTN1DComb object
    #         does not pad remaining axes
    #     """
    #     axIDs = mps_ax_dict.keys()
    #     new_branches = {}
    #     for grid in cgrid.get_active_grids(axIDs):
    #         active_axIDs = [axID for axID in grid.axIDs if axID in mps_ax_dict]
    #         grid_mps_dict = {axID: mps_ax_dict[axID] for axID in active_axIDs}
    #         grid_mpx = grid.make_mpx_ndim(grid_mps_dict)
    #         new_branches[grid.gridID] += [GridTN1D(grid, grid_mpx)]
    #
    #     mpo_comb = cls(cgrid, branches = new_branches)
    #     return mpo_comb
    #
    #
    # @classmethod
    # def pad_mpo_to_grid(cls, mpo_ax_dict, cgrid: GridsComb) -> 'GridTN1DComb':
    #     axIDs = mpo_ax_dict.keys()
    #     new_branches = {}
    #     for grid in cgrid.get_active_grids(axIDs):
    #         active_axIDs = [axID for axID in grid.axIDs if axID in mpo_ax_dict]
    #         grid_mpo_dict = {axID: mpo_ax_dict[axID] for axID in active_axIDs}
    #         grid_mpo = grid.make_mpo_ndim(grid_mpo_dict)
    #         new_branches[grid.gridID] += [GridTN1D(grid, grid_mpo)]
    #
    #     mpo_comb = cls(cgrid, branches=new_branches)
    #     return mpo_comb

    @classmethod
    def construct_spine(cls, L: int, data: np.ndarray, tt_ind_id='tt({})', tt_tag_id='TT({})', split_opts=None) \
            -> qtn.MatrixProductState:
        """ get mps connecting the dimensions together
            data is n-dimensional in desired order; no reshaping done
                 if data is None, defaults to data = 1.
            if provided, makes a TT decomposition (decompose by axis)
            Ootput MPS is in RIGHT CANONICAL form
        """
        new_tens = qtn.Tensor(data, inds=[tt_ind_id.format(i) for i in range(L)])
        spine_mps = helper.mpx_from_dense(new_tens, L, [tt_ind_id], site_tag_id=tt_tag_id, direction=1, return_mpx=True,
                                          split_opts=split_opts)  ## right canonical
        return spine_mps

    def get_data(self, ax_order=None, ax_select=None, pad_data=False) -> np.ndarray:
        """ convert to data
            axes ordered like output (0,1,2..) x input (0,1,2...)
        """
        # print('get data self', self)
        out = self.contract(ax_select=ax_select)
        # print('contract', out)
        if out is None:
            return None

        if ax_select is None:
            ax_select = {}

        inds_list_o = []
        inds_list_i = []
        for subgr in self.grid.grids:
            branch = self.get_branch(subgr)
            if branch is None or branch.data is None:
                if pad_data:
                    if self.data_type == DataType.MPO or self.data_type == DataType.MPX:
                        o_inds = [f'o({ax})' for ax in subgr.axes]
                        i_inds = [f'i({ax})' for ax in subgr.axes]
                        pad_iden = np.eye(subgr.npts).reshape([ax.npts for ax in subgr.axes] * 2)
                        pad_tens = qtn.Tensor(data=pad_iden, inds=tuple(o_inds + i_inds))
                        inds_list_o += o_inds
                        inds_list_i += i_inds
                    elif self.data_type == DataType.MPS:
                        out_axes = [ax for ax in subgr.axes if ax not in ax_select]
                        o_inds = [f'i({ax})' for ax in out_axes]
                        pad_ones = np.ones([ax.npts for ax in out_axes])
                        pad_tens = qtn.Tensor(data=pad_ones, inds=tuple(o_inds))
                        inds_list_o += o_inds
                    else:
                        raise TypeError('branch data needs to by MPS or MPO')
                    out = out.contract(pad_tens)
            elif isinstance(branch.data, qtn.MatrixProductOperator):
                inds_list_o += [f'o({ax})' for ax in subgr.axes if ax not in ax_select]
                inds_list_i += [f'i({ax})' for ax in subgr.axes if ax not in ax_select]
            elif isinstance(branch.data, qtn.MatrixProductState):
                inds_list_o += [f'i({ax})' for ax in subgr.axes if ax not in ax_select]

        if ax_order is not None:
            new_inds_o, new_inds_i = [], []
            for ax in ax_order:
                if f'o({ax})' in inds_list_o:
                    new_inds_o += [f'o({ax})']
                elif f'i({ax})' in inds_list_o:
                    new_inds_o += [f'i({ax})']
                if f'i({ax})' in inds_list_i:
                    new_inds_i += [f'i({ax})']
            inds_list_o = new_inds_o
            inds_list_i = new_inds_i

        out.transpose(*(inds_list_o + inds_list_i), inplace=True)
        return out.data

    #############################
    ## operations with TN comb ##
    #############################
    # @profile
    def apply(self, other: 'GridTN1DComb', inplace=False, zipup=True, use_mg=False, compress=False, compress_opts=None,
              add_cc=False, sub_compress_opts=None, **kwargs) -> 'GridTN1DComb':
        """ mpo: MPO TNcomb; mpo.grid.grids can be a subset of self.grid.grids
            new_grids:  final grid that the TN should occupy. Note that the grid should not
                        require reordering of the original grids
                        self is padded with ones if other components are MPS
        """
        # print('apply self', self)
        # print('apply other', other)
        # print('add cc', add_cc)

        # zipup = False
        # zipup = True
        # print('zipup?', zipup)

        if self.data is None:
            return self.create_like(new_data=None)

        if zipup:
            return self.apply_zipup(other, inplace=inplace, add_cc=add_cc,
                                    compress=compress, compress_opts=compress_opts,
                                    sub_compress_opts=sub_compress_opts)

        gtnc = self if inplace else self.copy(deep=False)  # original data will be replaced
        other = other.copy()

        new_branches = {}

        if not gtnc.grid == other.grid:

            ## prioritize matching of the grids with missing axes
            cgrid = gtnc.grid if gtnc.grid.ngrids > other.grid.ngrids else other.grid
            other = cgrid.pad_gtn_to_grid(other)

            branch_inds1 = None
            branch_inds2 = None

            # # branch ordering is determined by self and then other if not included already
            # pad_grids = [gl for gl in other.grid.grids if gl not in gtnc.grid.grids]
            # if len(pad_grids) > 0:
            #     grids_list = gtnc.grid.grids + tuple(pad_grids)
            #     print('grids list', grids_list)
            #     cgrid = gtnc.grid.__class__(gtnc.grid.gridID + other.grid.gridID, grids_list)
            #
            #     branch_inds1 = [grids_list.index(gl) for gl in gtnc.grid.grids]
            #     # assert (np.diff(branch_inds1) > 0), 'new grid must not reorder self.grids'
            # else:
            #     cgrid = gtnc.grid
            #     branch_inds1 = None  # range(len(cgrid.cgrids))
            #
            # grids_list = cgrid.grids
            # branch_inds2 = [grids_list.index(gl) for gl in other.grid.grids]
            # # assert (np.diff(branch_inds2) > 0), 'new grid must not reorder other.grids'
        else:
            cgrid = gtnc.grid
            branch_inds1 = branch_inds2 = None  # range(len(cgrid.grids))

        # print('branch inds', branch_inds1, branch_inds2)

        nbranches = len(cgrid.grids)
        # if branch_inds1:
        #     gtnc = gtnc.pad_self_to_new_grid(cgrid, inplace=True, branch_inds=branch_inds1)
        # if branch_inds2:
        #     other = other.pad_self_to_new_grid(cgrid, inplace=True, branch_inds=branch_inds2)

        ## combine the spines
        spine1 = gtnc.spine
        spine2 = other.spine

        if add_cc:  # gtnc -> gtnc + complex conj
            # spine + spine_CC, but don't sum over physical inds (bonds to the branches)
            for i in range(spine1.L):
                tens = spine1[i]
                tens.direct_product(tens.conj(), inplace=True)

        spine_ind_id = None
        if spine1 is not None:
            spine_ind_id = spine1.site_ind_id
            spine1 = helper.pad_mps(spine1, branch_inds1, nbranches)
            if branch_inds1 is not None:
                branch_old_to_new1 = {spine_ind_id.format(i): spine_ind_id.format(branch_inds1[i])
                                      for i in range(gtnc.grid.ngrids)}
            else:
                branch_old_to_new1 = {}

        if spine2 is not None:
            if spine_ind_id == other.spine_ind_id:
                other.spine_ind_id = other.spine_ind_id + '_'
            spine2 = helper.pad_mps(spine2, branch_inds2, nbranches)
            if branch_inds2 is not None:
                branch_old_to_new2 = {other.spine_ind_id.format(i):
                                          other.spine_ind_id.format(branch_inds2[i])
                                      for i in range(other.grid.ngrids)}
            else:
                branch_old_to_new2 = {}

        if spine1 is not None and spine2 is not None:
            new_spine = helper.mps_outerproduct(spine1, spine2, site_ind_id=spine_ind_id)
            new_spine.fuse_multibonds(inplace=True)
        elif spine2 is None:
            new_spine = spine1
        elif spine1 is None:
            new_spine = spine2
            spine_ind_id = spine2.site_ind_id
        else:
            new_spine = None
            spine_ind_id = None

        # if gtnc.spine is not None:
        #     spine_ind_id = spine1.site_ind_id
        #     spine1 = helper.pad_mps(spine1, branch_inds1, nbranches)

        #     if spine2 is not None:
        #         # site_ind_id2 = spine2.site_ind_id
        #         if spine_ind_id == other.spine_ind_id:
        #             other.spine_ind_id = other.spine_ind_id + '_'
        #         # print('change spine2', spine2)
        #         # if site_ind_id2 == spine1.site_ind_id:
        #         #     spine2.site_ind_id = site_ind_id2 + '_'
        #         spine2 = helper.pad_mps(spine2, branch_inds2, nbranches)
        #         new_spine = helper.mps_outerproduct(spine1, spine2, site_ind_id=spine_ind_id)
        #         new_spine.fuse_multibonds(inplace=True)
        #         # print('new spine', new_spine)
        #     else:
        #         new_spine = spine1
        # else:
        #     if spine2 is not None:
        #         spine_ind_id = spine2.site_ind_id
        #         new_spine = helper.pad_mps(spine2, branch_inds2, nbranches)
        #     else:
        #         spine_ind_id = None
        #         new_spine = None

        ## apply branches to each other
        for nb in range(nbranches):

            subgrid = cgrid.grids[nb]
            # print('subgrid', nb, subgrid)

            try:
                b_other = other.branches[subgrid]
            except KeyError:
                b_other = None

            try:
                b_self = gtnc.branches[subgrid]
            except KeyError:

                if b_other is not None:
                    b_other = b_other.copy()
                    old_nb = nb if branch_inds2 is None else branch_inds2.index(nb)
                    tt_inds_old = spine2.site_ind_id.format(old_nb)
                    tt_inds_new = new_spine.site_ind_id.format(nb)
                    b_other.get_anchor_tens().reindex({tt_inds_old: tt_inds_new}, inplace=True)

                if gtnc.data_type == DataType.MPS:
                    b_self = subgrid.get_iden_mps()
                    if b_other is None:
                        qtn.new_bond(b_self.get_anchor_tens(), new_spine[nb], name=spine_ind_id.format(nb))
                else:
                    b_self = None

            if b_self is None and b_other is None:
                new_branch = None  # subgrid.make_empty_gridTN()
                # print('None branch', nb, new_spine)
            elif b_self is None and b_other is not None:
                new_branch = b_other.copy() if gtnc.data_type is not None else None

                old_nb = nb if branch_inds2 is None else branch_inds2.index(nb)
                tt_inds_old = spine2.site_ind_id.format(old_nb)
                tt_inds_new = new_spine.site_ind_id.format(nb)
                new_branch.get_anchor_tens().reindex({tt_inds_old: tt_inds_new}, inplace=True)

            elif b_self is not None and b_other is None:
                new_branch = b_self.copy()

                ## previously commented out?
                if branch_inds1 is not None:
                    old_nb = branch_inds1.index(nb)
                    tt_inds_old = new_spine.site_ind_id.format(old_nb)
                    tt_inds_new = new_spine.site_ind_id.format(nb)
                    new_branch.get_anchor_tens().reindex({tt_inds_old: tt_inds_new}, inplace=True)

                if add_cc:  # gtnc -> gtnc + complex conj
                    new_branch.add(new_branch.conj(), inplace=True)
            else:

                b_self.get_anchor_tens().reindex(branch_old_to_new1, inplace=True)
                b_other.get_anchor_tens().reindex(branch_old_to_new2, inplace=True)

                tt_inds_self = new_spine[nb].bonds(b_self.get_anchor_tens())
                try:
                    tt_ind_self = next(iter(tt_inds_self))
                except StopIteration:
                    tt_ind_self = None

                # if spine1 is not None:
                #     tt_inds_self = spine1[nb].bonds(b_self[0])
                #     tt_ind_self = next(iter(tt_inds_self))
                # else:
                #     tt_ind_self = None

                try:
                    tt_inds_other = spine2[nb].bonds(b_other.get_anchor_tens())
                    tt_ind_other = next(iter(tt_inds_other))
                except StopIteration:
                    try:  ## b_self is padded identity mps
                        tt_inds_other = new_spine[nb].bonds(b_other.get_anchor_tens())
                        tt_ind_other = next(iter(tt_inds_other))
                    except StopIteration:
                        tt_ind_other = None

                # if tt_ind_other == tt_ind_self and tt_ind_self is not None:
                #     b_other[0].reindex({tt_ind_other:tt_ind_other+'_'},inplace=True)
                #     spine2[nb].reindex({tt_ind_other:tt_ind_other+'_'},inplace=True)
                #     tt_ind_other += '_'

                # print(b_self.max_bond(), b_other.max_bond())
                new_branch = b_self.apply(b_other, compress=False, add_cc=add_cc)

                if tt_ind_self and tt_ind_other:  # not both None
                    new_branch.data[0].fuse({spine_ind_id.format(nb): (tt_ind_self, tt_ind_other)}, inplace=True)
                    # ordering consistent w/ helper.mps_outerproduct

            # new_grid_mpx = GridTN1D(subgrid, data=new_branch)
            if new_branch is not None:
                new_branches[subgrid] = new_branch  # new_grid_mpx

        gtnc._spine = new_spine
        gtnc._branches = new_branches
        gtnc._update_active_grids()
        gtnc._set_data_type([new_branches[nb].data_type for nb in new_branches.keys()])

        gtnc.canon_site = None
        if compress:
            gtnc.compress(compress_opts=compress_opts, sub_compress_opts=sub_compress_opts)

        gtnc._exponent += other._exponent
        gtnc._sign *= other._sign
        gtnc.collect_exponents()

        gtnc.is_constant *= other.is_constant
        gtnc.constant_axes = [ax for ax in gtnc.constant_axes if ax in other.constant_axes]

        return gtnc

    # @profile
    def apply_zipup(self, other: 'GridTN1DComb', inplace=False, add_cc=False, compress=False, compress_opts=None,
                    sub_compress_opts=None) \
            -> 'GridTN1DComb':
        """ mpo: MPO TNcomb; mpo.grid.grids can be a subset of self.grid.grids
            new_grids:  final grid that the TN should occupy. Note that the grid should not
                        require reordering of the original grids
                        self is padded with ones if other components are MPS
        """
        gtnc = self if inplace else self.copy(deep=False)  # original data will be replaced
        if compress_opts is None:
            compress_opts = {}

        new_branches = {}

        if not gtnc.grid == other.grid:
            # branch ordering is determined by self and then other if not included already
            pad_grids = [gl for gl in other.grid.grids if gl not in gtnc.grid.grids]
            if len(pad_grids) > 0:
                grids_list = gtnc.grid.grids + tuple(pad_grids)
                cgrid = GridsComb(gtnc.grid.gridID + other.grid.gridID, grids_list)

                new_branch_inds1 = [grids_list.index(gl) for gl in gtnc.grid.grids]
            else:
                cgrid = gtnc.grid
                new_branch_inds1 = None  # range(len(cgrid.cgrids))

            grids_list = cgrid.grids
            new_branch_inds2 = [grids_list.index(gl) for gl in other.grid.grids]
        else:
            cgrid = gtnc.grid
            new_branch_inds1 = new_branch_inds2 = None  # range(len(cgrid.grids))

        # print('branch inds', branch_inds1, branch_inds2)

        nbranches = len(cgrid.grids)

        ## combine the spines
        spine1 = gtnc.spine
        spine2 = other.spine

        spine_ind_id = spine1.site_ind_id
        spine1 = helper.pad_mps(spine1, new_branch_inds1, nbranches)
        if new_branch_inds1 is not None:
            branch_old_to_new1 = {spine_ind_id.format(i): spine_ind_id.format(new_branch_inds1[i])
                                  for i in range(gtnc.grid.ngrids)}
        else:
            branch_old_to_new1 = {}

        if spine_ind_id == other.spine_ind_id:
            other.spine_ind_id = other.spine_ind_id + '_o_'
        spine2 = helper.pad_mps(spine2, new_branch_inds2, nbranches)
        if new_branch_inds2 is not None:
            branch_old_to_new2 = {other.spine_ind_id.format(i): other.spine_ind_id.format(new_branch_inds2[i])
                                  for i in range(other.grid.ngrids)}
        else:
            branch_old_to_new2 = {}

        # add_cc = False
        # print('apply spine', spine1, check_elem_mult_anc)
        if add_cc:  # gtnc -> gtnc + complex conj
            # print('here?')
            # spine + spine_CC, but don't sum over physical inds (bonds to the branches)
            for i in range(spine1.L):
                tens = spine1[i]
                tens.direct_product(tens.conj(), inplace=True)
        # print('apply spine', spine1)

        # new_spine = helper.mps_outerproduct(spine1, spine2, site_ind_id=spine_ind_id)
        # new_spine.fuse_multibonds(inplace=True)
        new_spine = qtn.TensorNetwork([])
        new_spine.exponent = spine1.exponent + spine2.exponent

        # print('spine 1', spine1, spine2)
        # print(gtnc)
        # print(other)

        ## apply branches to each other
        prev_spine_tens = None
        canon_spine_tens = None
        # prev_spine_mat = None
        for nb in range(nbranches):
            subgrid = cgrid.grids[nb]

            b_self = gtnc.get_branch(subgrid)
            b_other = other.get_branch(subgrid)
            if b_self is None:
                if gtnc.data_type is None:
                    continue
                elif gtnc.data_type == DataType.MPS:
                    b_self = subgrid.get_iden_mps()
                    qtn.new_bond(b_self.get_anchor_tens(), spine1[nb], name=spine1.site_ind_id.format(nb))
                else:
                    b_self = None

            if b_self is None and b_other is None:
                new_branch = None
                contract_spine_list = [spine1[nb], spine2[nb]]
                if prev_spine_tens is not None:
                    contract_spine_list += [prev_spine_tens]

            elif b_self is None and b_other is not None:
                new_branch = b_other.canonize(i=b_other.get_anchor_ind()) if gtnc.data_type is not None else None

                tens0 = new_branch.get_anchor_tens()
                old_nb = nb if new_branch_inds2 is None else new_branch_inds2.index(nb)
                tt_inds_old = spine2.site_ind_id.format(old_nb)
                tt_inds_new = spine1.site_ind_id.format(nb)
                tens0.reindex({tt_inds_old: f'tmp_spine{nb}'}, inplace=True)
                spine2[nb].reindex({tt_inds_old: f'tmp_spine{nb}'}, inplace=True)

                # # new_branch = b_other.copy() if gtnc.data_type==DataType.MPO else None   ## change 9/15/22
                # spine_tens = qtn.tensor_contract(spine1[nb], spine2[nb])
                # spine_tens.reindex({spine2.site_ind_id.format(nb): tt_inds_new}, inplace=True)

                shared_inds, left_inds = tens0.filter_bonds(spine2[nb])
                tensL, tensR = qtn.tensor_split(tens0, left_inds=left_inds, absorb='right', bond_ind=tt_inds_new)

                tens0.modify(data=tensL.data, inds=tensL.inds)
                contract_spine_list = [spine1[nb], spine2[nb], tensR]

                # spine_tens = qtn.tensor_contract(tensR, spine1[nb], spine2[nb])
                # spine_tens.reindex({'tmp': tt_inds_new}, inplace=True)

            elif b_self is not None and b_other is None:
                # new_branch = b_self.copy()
                new_branch = b_self.canonize(i=b_self.get_anchor_ind(), inplace=False)

                old_nb = nb if new_branch_inds1 is None else new_branch_inds1.index(nb)
                tt_inds_old = spine1.site_ind_id.format(old_nb)
                tt_inds_new = spine1.site_ind_id.format(nb)

                tens0 = new_branch.get_anchor_tens()
                tens0.reindex({tt_inds_old: f'tmp_spine{nb}'}, inplace=True)
                spine1[nb].reindex({tt_inds_old: f'tmp_spine{nb}'}, inplace=True)

                # if new_branch_inds1 is not None:
                #     old_nb = new_branch_inds1.index(nb)
                #     tt_inds_old = spine1.site_ind_id.format(old_nb)
                #     new_branch.get_anchor_tens().reindex({tt_inds_old: tt_inds_new}, inplace=True)

                if add_cc:  # gtnc -> gtnc + complex conj
                    # print('z', new_branch.max_bond(), new_branch[0].shape)
                    new_branch.add(new_branch.conj(), inplace=True)

                # spine_tens = qtn.tensor_contract(spine1[nb], spine2[nb])

                # tens0 = new_branch.get_anchor_tens()
                shared_inds, left_inds = tens0.filter_bonds(spine1[nb])
                tensL, tensR = qtn.tensor_split(tens0, left_inds=left_inds, absorb='right', bond_ind=tt_inds_new)

                tens0.modify(data=tensL.data, inds=tensL.inds)
                contract_spine_list = [tensR, spine1[nb], spine2[nb]]
                # spine_tens = qtn.tensor_contract(tensR, spine1[nb], spine2[nb])

            else:
                # print('b self is not None, b other is not None')
                b_other = b_other.copy()
                b_self.get_anchor_tens().reindex(branch_old_to_new1, inplace=True)
                b_other.get_anchor_tens().reindex(branch_old_to_new2, inplace=True)

                tt_inds_self = spine1[nb].bonds(b_self.get_anchor_tens())
                tt_ind_self = next(iter(tt_inds_self))

                tt_inds_other = spine2[nb].bonds(b_other.get_anchor_tens())
                tt_ind_other = next(iter(tt_inds_other))

                ## apply zip-up with -> canon_site = 0
                if b_self.get_anchor_ind() == 0:
                    canon_form = 'right'
                else:
                    raise NotImplementedError
                branch_compress_opts = {'cutoff': CUTOFF, 'cutoff_mode': CUTOFF_MODE, 'form': canon_form}
                new_branch = b_self.apply(b_other, zipup=True, compress=False, compress_opts=branch_compress_opts,
                                          add_cc=add_cc)

                b0 = new_branch.get_anchor_tens()
                b0.reindex({tt_ind_self: f'tmp_spine{nb}_self', tt_ind_other: f'tmp_spine{nb}_other'},
                           inplace=True)
                spine1_tens = spine1[nb].reindex({tt_ind_self: f'tmp_spine{nb}_self'})
                spine2_tens = spine2[nb].reindex({tt_ind_other: f'tmp_spine{nb}_other'})

                new_s0, new_b0 = b0.split((f'tmp_spine{nb}_self', f'tmp_spine{nb}_other'),
                                          cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE,
                                          absorb='left', ltags=(), bond_ind=spine_ind_id.format(nb))
                ### check for nans
                it = 0
                while np.any(np.isnan(new_s0.data)) or np.any(np.isnan(new_b0.data)):
                    new_s0, new_b0 = b0.split((f'tmp_spine{nb}_self', f'tmp_spine{nb}_other'), method='eig',
                                              cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE,
                                              absorb='left', ltags=(), bond_ind=spine_ind_id.format(nb))
                    print('branch to spine split yielded nans')
                    it += 1
                    if it > 10:
                        raise ValueError('split yielding nans')

                b0.modify(data=new_b0.data, inds=new_b0.inds)

                contract_spine_list = [spine1_tens, spine2_tens, new_s0]
                # spine_tens = qtn.tensor_contract(spine1[nb], spine2[nb], new_s0)

            ### calculate new spine tens
            if canon_spine_tens is not None:
                contract_spine_list += [canon_spine_tens]

            # print('contract spine list', contract_spine_list)
            spine_tens = qtn.tensor_contract(*contract_spine_list)
            spine_tens.modify(tags=spine1[nb].tags)

            if nb < nbranches - 1:
                if prev_spine_tens is not None:
                    left_inds, _ = spine_tens.filter_bonds(prev_spine_tens)
                else:
                    left_inds = []
                left_inds = left_inds + [spine_ind_id.format(nb)]

                spine_tens, canon_spine_tens = spine_tens.split(left_inds, cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE,
                                                                absorb='right', rtags=())

            if new_branch is not None:
                new_branches[subgrid] = new_branch  # new_grid_mpx
                # print('set new branch', new_branch)

            new_spine.add(spine_tens, virtual=True)
            # new_spine.strip_exponent(spine_tens)  # normalizes tensor, adds norm to exponent
            prev_spine_tens = spine_tens

        new_spine.fuse_multibonds(inplace=True)

        gtnc._spine = new_spine.view_like(spine1, inplace=True)
        gtnc._branches = new_branches
        gtnc._set_data_type([new_branches[nb].data_type for nb in new_branches.keys()])
        gtnc._update_active_grids()

        gtnc.canon_site = nbranches - 1

        # for gr, b in gtnc.branches.items():
        #     print('check b orthog', helper.check_orthog(b.data))
        # print('check spine orthog', gtnc.canon_site, helper.check_orthog(gtnc.spine))

        max_bond = compress_opts.get('max_bond', None)
        cutoff = compress_opts.get('cutoff', CUTOFF)
        if compress:  # and max_bond is not None and cutoff > CUTOFF:
            gtnc.compress(inplace=True, compress_opts=compress_opts, sub_compress_opts=sub_compress_opts,
                          branch_keys=range(nbranches - 1, -1, -1), canonize=False)

        gtnc._exponent += other.exponent
        gtnc._sign *= other.sign
        gtnc.collect_exponents()

        gtnc.is_constant *= other.is_constant
        gtnc.constant_axes = [ax for ax in gtnc.constant_axes if ax in other.constant_axes]

        return gtnc

    # @profile
    def apply_rdm(self, other, bra_self=None, bra_other=None, left_env=None, right_env=None,
                  direction=1, inplace=False, compress_opts=None, verbose=False, **kwargs) -> 'GridTN1DComb':
        """ Apply grid_mpo to self, assuming they exist on the same grid
            contract (and compress) using rdm method
        """

        from mp_apply_rdm_comb import apply_rdm as apply_rdm_mp

        return apply_rdm_mp(self, other, direction=direction, inplace=inplace,
                            compress_opts=compress_opts, verbose=verbose)


    # @profile
    def add(self, other: 'GridTN1DComb', zipup=False, inplace=False, compress_type=CompressType.SVD, compress=False,
            compress_opts=None, sub_compress_opts=None) -> 'GridTN1DComb':
        """ add two TNcombs. can be two separate grids. will pad with ones
            if MPS or identity if MPO
        """
        # zipup = compress  # False
        gtnc = self if inplace else self.copy()

        # print('gtnc', gtnc.data, gtnc.num_active_branches, gtnc.exponent, gtnc.sign)
        # print('other', other.data, other.num_active_branches, other.exponent, other.sign)

        if other is None or other.data is None:
            return gtnc

        if gtnc.data is None:
            # print('self.grid', self.grid.axes)
            # print('other.grid', other.grid.axes)
            if gtnc.grid != other.grid:
                other = other.pad_self_to_new_grid(gtnc.grid)
            gtnc.data = other  ## copies the data
            gtnc.is_constant = other.is_constant
            gtnc.constant_axes = other.constant_axes
            return gtnc


        if gtnc.grid.ngrids == 0 and other.grid.ngrids == 0:
            gtnc._spine = gtnc.spine + other.spine
            return gtnc
        elif other.grid.ngrids == 0:
            other = other.pad_self_to_new_grid(gtnc.grid, inplace=True)
        elif gtnc.grid.ngrids == 0:
            gtnc.pad_self_to_new_grid(other.grid, inplace=True)


        if zipup:
            return gtnc.add_zipup(other, inplace=True, compress=compress, compress_opts=compress_opts,
                                  sub_compress_opts=sub_compress_opts)

        # if other.grid != self.grid:
        #     if set(other.grid.axes).issubset(self.grid.axes):
        #         other = self.grid.pad_gtn_to_grid(other)
        #     else:
        #         raise ValueError(f'gtn_mpx2 on grid {other.grid} not compatible with self on grid {self.grid}')
        # else:
        #     other = other.copy()
        other = other.copy()

        gtnc.collect_exponents()
        other.collect_exponents()

        other.distribute_exponents(val=other._exponent - gtnc._exponent, across_all=True)

        if gtnc.grid != other.grid:
            # branch ordering is determined by self and then other if not included already
            pad_grids = [gl for gl in other.grid.grids if gl.gridID not in gtnc.grid.gridIDs]
            if len(pad_grids) > 0:
                grids_list = gtnc.grid.grids + tuple(pad_grids)
                cgrid = gtnc.grid.__class__(gtnc.grid.gridID + other.grid.gridID, grids_list)
                branch_inds1 = [grids_list.index(gl) for gl in gtnc.grid.grids]
            else:
                cgrid = gtnc.grid
                branch_inds1 = None

            grids_list = cgrid.grids
            branch_inds2 = [grids_list.index(gl) for gl in other.grid.grids]
        else:
            cgrid = gtnc.grid
            branch_inds1 = branch_inds2 = None

        nbranches = len(cgrid)
        if branch_inds1:
            gtnc = gtnc.pad_self_to_new_grid(cgrid, inplace=True, branch_inds=branch_inds1)
        if branch_inds2:
            other = other.pad_self_to_new_grid(cgrid, inplace=True, branch_inds=branch_inds2)

        spine1 = gtnc.spine
        spine2 = other.spine

        ## make spine2 match spine inds and interior bonds
        for nb in range(nbranches):
            branch = other.get_branch(cgrid.grids[nb])
            old_ind_id = spine2.site_ind(nb)
            new_ind_id = spine1.site_ind(nb)
            if branch is not None:
                branch.get_anchor_tens().reindex({old_ind_id: new_ind_id}, inplace=True)

        spine2.site_ind_id = spine1.site_ind_id
        for nb in range(nbranches - 1):
            old_bond_ij = spine2.bond(nb, nb + 1)
            new_bond_ij = spine1.bond(nb, nb + 1)
            tens_i = spine2[nb]
            tens_j = spine2[nb + 1]
            tens_i.reindex({old_bond_ij: new_bond_ij}, inplace=True)
            tens_j.reindex({old_bond_ij: new_bond_ij}, inplace=True)

        ## combine the spines
        new_spine = spine1
        # print('spine1', new_spine, spine2)
        for nb in range(nbranches):

            ## add branches together
            branch1 = gtnc.get_branch(cgrid.grids[nb])
            branch2 = other.get_branch(cgrid.grids[nb])

            if branch1 is None and branch2 is None:
                new_spine[nb].direct_product(spine2[nb], inplace=True)
                continue

            elif branch1 is None:  ## branch 2 is not None
                if isinstance(branch2.data, qtn.MatrixProductState):
                    branch1 = branch2.grid.get_ones_mps(site_ind_id=branch2.data.site_ind_id,
                                                        site_tag_id=branch2.data.site_tag_id)
                elif isinstance(branch2.data, qtn.MatrixProductOperator):
                    branch1 = branch2.grid.get_iden_mpo(upper_ind_id=branch2.data.upper_ind_id,
                                                        lower_ind_id=branch2.data.lower_ind_id,
                                                        site_tag_id=branch2.data.site_tag_id)
                # tens0 = branch1.get_anchor_tens()
                # tens0.new_ind(new_spine.site_ind(nb))
                # gtnc._active_grids += [cgrid.grids[nb]]
                gtnc.set_branch(nb, branch1)
            elif branch2 is None:
                if isinstance(branch1.data, qtn.MatrixProductState):
                    branch2 = branch1.grid.get_ones_mps(site_ind_id=branch1.data.site_ind_id,
                                                        site_tag_id=branch1.data.site_tag_id)
                elif isinstance(branch1.data, qtn.MatrixProductOperator):
                    branch2 = branch1.grid.get_iden_mpo(upper_ind_id=branch1.data.upper_ind_id,
                                                        lower_ind_id=branch1.data.lower_ind_id,
                                                        site_tag_id=branch1.data.site_tag_id)
                # tens0 = branch2.get_anchor_tens()
                # tens0.new_ind(new_spine.site_ind(nb))
                other.set_branch(nb, branch2)

            # print('self', self)
            # print('other', other)
            #
            # print('branch 1', branch1)
            # print('branch 2', branch2)

            branch1.add(branch2, inplace=True)
            if nb == 0:
                spine2_tens = spine2[nb].copy() * (other._sign / self._sign)
            else:
                spine2_tens = spine2[nb]

            new_spine[nb].direct_product(spine2_tens, inplace=True)

        gtnc.canon_site = None
        if compress:
            gtnc.compress(compress_opts=compress_opts, sub_compress_opts=sub_compress_opts)

        gtnc.is_constant *= other.is_constant
        gtnc.constant_axes = [ax for ax in gtnc.constant_axes if ax in other.constant_axes]
        return gtnc

    def add_dmrg(self, *other_gtns: 'GridTN1DComb', inplace=False, compress_opts=None, **dmrg_opts) -> 'GridTN1DComb':
        """ add multiple gtns together using DMRG solver
        """
        if self.grid.ngrids > 2:
            raise NotImplementedError

        return super().add_dmrg(*other_gtns, inplace=inplace, compress_opts=compress_opts, **dmrg_opts)

        # from local_solvers.local_dmrg_eval import local_dmrg_evaluator, Term_DMRG
        #
        # if self.grid.ngrids <= 2:
        #     raise NotImplementedError
        #
        # DMAX = compress_opts.get('max_bond', None) if compress_opts is not None else None
        #
        # gtn = self if inplace else self.copy()
        # is_mps = self.data_type == DataType.MPS
        #
        # self_tt = self.grid.gtn_to_dmrg_format(gtn, is_mps=is_mps)
        # terms = [ Term_DMRG(self_tt.copy()) ]
        # terms += [ Term_DMRG(self.grid.gtn_to_dmrg_format(ogtn, is_mps=is_mps).copy()) for ogtn in other_gtns ]
        #
        # func_mps = local_dmrg_evaluator(terms, max_bond=DMAX, **dmrg_opts)
        #
        # gtn = self.grid.dmrg_to_gtn_format(gtn, func_mps, is_mps=is_mps)
        # return gtn

    # @profile
    def add_zipup(self, other: 'GridTN1DComb', inplace=False, compress=False, compress_opts=None,
                  sub_compress_opts=None) -> 'GridTN1DComb':
        """ add two TNcombs. can be two separate grids. will pad with ones
            if MPS or identity if MPO
        """
        gtnc = self if inplace else self.copy()

        # print('gtnc', gtnc.data, gtnc.num_active_branches, gtnc.exponent, gtnc.sign)
        # print('other', other.data, other.num_active_branches, other.exponent, other.sign)

        if other.data is None:
            return gtnc

        if gtnc.data is None:
            # print('add returned other')
            gtnc.data = other  ## copies the data
            return gtnc

        other = other.copy()

        # if other.grid != gtnc.grid:
        #     if set(other.grid.axes).issubset(gtnc.grid.axes):
        #         print('padded other')
        #         other = gtnc.grid.pad_gtn_to_grid(other)
        #     else:
        #         raise ValueError(f'gtn_mpx2 ong grid {other.grid} not compatible with self on grid {self.grid}')
        # else:
        #     other = other.copy()

        if gtnc.canon_site != other.canon_site:
            gtnc.canonicalize_around_i(0)
            other.canonicalize_around_i(0)

        gtnc.collect_exponents()
        other.collect_exponents()

        other.distribute_exponents(val=other._exponent - gtnc._exponent)

        if gtnc.grid != other.grid:
            # branch ordering is determined by self and then other if not included already
            pad_grids = [gl for gl in other.grid.grids if gl.gridID not in gtnc.grid.gridIDs]
            if len(pad_grids) > 0:
                grids_list = gtnc.grid.grids + tuple(pad_grids)
                cgrid = gtnc.grid.__class__(gtnc.grid.gridID + other.grid.gridID, grids_list)
                branch_inds1 = [grids_list.index(gl) for gl in gtnc.grid.grids]
            else:
                cgrid = gtnc.grid
                branch_inds1 = None

            grids_list = cgrid.grids
            branch_inds2 = [grids_list.index(gl) for gl in other.grid.grids]
        else:
            cgrid = gtnc.grid
            branch_inds1 = branch_inds2 = None

        # print('gtnc', gtnc)
        # print('other', other)

        nbranches = len(cgrid)
        if branch_inds1:
            gtnc = gtnc.pad_self_to_new_grid(cgrid, inplace=True, branch_inds=branch_inds1)
        if branch_inds2:
            other = other.pad_self_to_new_grid(cgrid, inplace=True, branch_inds=branch_inds2)

        spine1 = gtnc.spine
        spine2 = other.spine

        ## make spine2 match spine inds and interior bonds
        for nb in range(nbranches):
            branch = other.get_branch(cgrid.grids[nb])
            old_ind_id = spine2.site_ind(nb)
            new_ind_id = spine1.site_ind(nb)
            if branch is not None:
                branch.get_anchor_tens().reindex({old_ind_id: new_ind_id}, inplace=True)

        spine2.site_ind_id = spine1.site_ind_id
        for nb in range(nbranches - 1):
            old_bond_ij = spine2.bond(nb, nb + 1)
            new_bond_ij = spine1.bond(nb, nb + 1)
            tens_i = spine2[nb]
            tens_j = spine2[nb + 1]
            tens_i.reindex({old_bond_ij: new_bond_ij}, inplace=True)
            tens_j.reindex({old_bond_ij: new_bond_ij}, inplace=True)

        ## combine the spines
        new_spine = spine1
        for nb in range(nbranches):

            ## add branches together
            branch1 = gtnc.get_branch(cgrid.grids[nb])
            branch2 = other.get_branch(cgrid.grids[nb])

            if branch1 is None and branch2 is None:
                new_spine[nb].direct_product(spine2[nb], inplace=True)
                continue

            elif branch1 is None:  ## branch 2 is not None
                if isinstance(branch2.data, qtn.MatrixProductState):
                    branch1 = branch2.grid.get_ones_mps(site_ind_id=branch2.data.site_ind_id,
                                                        site_tag_id=branch2.data.site_tag_id)
                elif isinstance(branch2.data, qtn.MatrixProductOperator):
                    branch1 = branch2.grid.get_iden_mpo(upper_ind_id=branch2.data.upper_ind_id,
                                                        lower_ind_id=branch2.data.lower_ind_id,
                                                        site_tag_id=branch2.data.site_tag_id)
                gtnc.set_branch(nb, branch1)

            elif branch2 is None:
                if isinstance(branch1.data, qtn.MatrixProductState):
                    branch2 = branch1.grid.get_ones_mps(site_ind_id=branch1.data.site_ind_id,
                                                        site_tag_id=branch1.data.site_tag_id)
                elif isinstance(branch1.data, qtn.MatrixProductOperator):
                    branch2 = branch1.grid.get_iden_mpo(upper_ind_id=branch1.data.upper_ind_id,
                                                        lower_ind_id=branch1.data.lower_ind_id,
                                                        site_tag_id=branch1.data.site_tag_id)
                other.set_branch(nb, branch2)

            zipup_compress_opts = {'form': 'right', 'max_bond': None}
            # branch1.add(branch2, inplace=True, compress=True, compress_opts=zipup_compress_opts)
            branch1 = branch1.add(branch2, inplace=True, zipup=True, compress=False, compress_opts=zipup_compress_opts)

            if branch1 is None:
                print('CHECK: RETURNING NONE')
                return None

            if nb == 0:
                spine2_tens = spine2[nb].copy() * (other._sign / self._sign)
            else:
                spine2_tens = spine2[nb]

            new_spine[nb].direct_product(spine2_tens, inplace=True)

            ## compress branch into spine
            b0 = branch1.get_anchor_tens()
            helper.compress_tens_list(b0, new_spine[nb], inplace=True)
            if b0.data or new_spine[nb].data is None:
                print('CHECK: RETURNING NONE')
                return None

            ## compress spine
            if nb > 0:
                helper.compress_tens_list(new_spine[nb - 1], new_spine[nb], inplace=True)

            new_spine.strip_exponent(new_spine[nb])  # normalizes tensor, adds norm to exponent
            # tens_norm = new_spine[nb].norm()
            # new_spine[nb].modify(apply=lambda data: data/tens_norm)
            # norm_val *= tens_norm
            # # new_spine.exponent += np.log10(tens_norm)
            #
            # # tag = new_spine.site_tag(nb)
            # # tens = new_spine[tag]
            # # print('tens', tens)
            # # new_spine.strip_exponent(new_spine[nb])  # normalizes tensor, adds norm to exponent

        # gtnc._exponent += np.log10(norm_val)
        # new_spine.exponent += np.log10(np.abs(norm_val))

        gtnc.canon_site = nbranches - 1
        # if compress and compress_opts is not None and \
        #         compress_opts.get('max_bond',None) is not None and compress_opts.get('cutoff',CUTOFF) > CUTOFF:
        if compress:
            gtnc.compress(compress_opts=compress_opts, sub_compress_opts=sub_compress_opts)
            # print('exponents 2', gtnc._exponent, gtnc.spine.exponent, [b.exponent for gk, b in gtnc.branches.items()])

        gtnc.is_constant *= other.is_constant
        gtnc.constant_axes = [ax for ax in gtnc.constant_axes if ax in other.constant_axes]

        return gtnc

    def scalar_multiply(self, scalar_val: Numeric, inplace=False):
        tnc = self if inplace else self.copy()

        abs_val = np.abs(scalar_val)
        if np.isreal(scalar_val):
            if scalar_val < 0:
                sign = -1
            else:
                sign = 1
        else:
            sign = np.exp(-1.j * np.angle(scalar_val))

        tnc._exponent += np.log10(abs_val)
        tnc._sign *= sign

        # helper.scalar_multiply(tnc.spine, sign, inplace=True)
        # branch = tnc.active_branches[0]
        # helper.scalar_multiply(branch.data, sign, inplace=True)

        return tnc

    def evaluate_func(self, func: Callable, max_bond=None, inplace=False) -> 'GridTN1DComb':

        if self.num_active_branches > 2:
            raise NotImplementedError

        # from local_solvers.local_cross_eval import local_cross_evaluator, Term_Cross
        from local_solvers.local_cross_eval_old import local_cross_evaluator, Term_Cross

        gtn = self if inplace else self.copy()
        tt_rep = self.grid.gtn_to_dmrg_format(gtn, is_mps=(self.data_type==DataType.MPS))

        term1c = Term_Cross(tt_rep.copy(), mps_func=func)
        nsites = 2  # 1 if (max_bond is None or max_bond <= self.max_bond()) else 2
        func_mps = local_cross_evaluator([term1c], nsites=nsites, max_bond=max_bond)
        # func_gtn = tt_rep.create_like(func_mps)

        out = self.grid.dmrg_to_gtn_format(gtn, func_mps, is_mps=(self.data_type==DataType.MPS))
        return out



    def canonize(self, inplace=True, scale=True, form='right', i: Union[int, 'Grid'] = None, cur_orthog=None):
        """ put in canonical form: all branches right canonical, spine mixed canon form centered at i
        """
        if i is None:
            i = 0
        elif isinstance(i, Grid):
            i = self.grid.get_grid_ind(i)

        gtnc = self if inplace else self.copy()

        gtnc.canonicalize_around_i(i, redo_canon=True)
        # branch = gtnc.get_branch(i)
        # branch.canonize(inplace=True, i=branch.get_anchor_ind())
        # helper.canonize_tens_list(branch.get_anchor_tens(), gtnc.spine[i], inplace=True)
        # gtnc.collect_exponents()
        # gtnc.canon_site = i
        return gtnc

    # @profile
    def canonicalize_around_i(self, i: Union[int, 'Grid'], scale=True, redo_canon=True):
        """ canonicalizes the comb such that spine is center canonical at branch/site i
        """
        ### TODO: take into account comb canon site

        if isinstance(i, Grid):
            i = self.grid.get_grid_ind(i)

        ## canonicalize branches
        for gr in self.active_grids:
            nb = self.grid.get_grid_ind(gr)
            # if nb == i:  continue    ### don't canonicalize active branch

            branch = self.get_branch(gr)
            anchor_idx = branch.get_anchor_ind()
            # print('branch.canon_site', branch.canon_site, redo_canon)
            if redo_canon or branch.canon_site is None or branch.canon_site < 0 or branch.canon_site != anchor_idx:
                branch.canonize(inplace=True, i=anchor_idx, scale=scale)
            self.collect_exponents()

            ## canonicalize into spine
            helper.canonize_tens_list(branch.data[anchor_idx], self.spine[nb], inplace=True)
            # print('branch to spine', nb, branch.data[anchor_idx].norm(), self.spine[nb].norm())
            # print('canonize around', np.any(np.isnan(self.spine[nb].data)), \
            #                         [np.any(np.isnan(branch.data[n].data)) for n in range(branch.data.L)])

            # orthog_check = helper.check_right_orthog([self.spine[nb]] + [branch.data[i] for i in range(branch.data.L)])
            # print('check branch right orthog C', nb, gr, orthog_check)
            # if orthog_check != anchor_idx:
            #     exit()

        ## canonicalize spine
        out = helper.canonize(self.spine, i=i, cur_orthog=None, scale=scale)
        # print('check spine orthog', i, helper.check_orthog(self.spine))
        if out is None:
            self.data = None
        else:
            self.collect_exponents()
            self.canon_site = i
        return self

    def compress(self, inplace=True, verbose=False, compress_type=CompressType.SVD, compress_opts=None,
                 sub_compress_opts=None, branch_keys=None, direction=0, norm_cutoff=None, conservative=False,
                 **kwargs):

        if isinstance(self.data, (int, float, complex, np.number)):
            return self if inplace else self.create_like()

        if self.data_type is DataType.MPO:
            return self.compress_v1(inplace=inplace, verbose=verbose, compress_type=compress_type,
                                    compress_opts=compress_opts, sub_compress_opts=sub_compress_opts,
                                    branch_keys=branch_keys, direction=direction, norm_cutoff=None, **kwargs)
        else:
            if self.grid.ngrids <= 2:
                gtn = self if inplace else self.copy()
                if gtn is None or gtn.data is None:
                    return gtn
                tt_rep = self.grid.gtn_to_dmrg_format(gtn, is_mps=(self.data_type is DataType.MPS))
                helper.compress(tt_rep, compress_opts=compress_opts)
                new_gtn = self.grid.dmrg_to_gtn_format(gtn, tt_rep, is_mps=(self.data_type is DataType.MPS))
                return new_gtn

            return self.compress_v1(inplace=inplace, verbose=verbose, compress_type=compress_type,
                                    compress_opts=compress_opts, sub_compress_opts=sub_compress_opts,
                                    branch_keys=branch_keys, direction=direction, norm_cutoff=None, **kwargs)

    # @profile
    def compress_v1(self, inplace=True, verbose=False, compress_type=CompressType.SVD, compress_opts=None,
                    sub_compress_opts=None, branch_keys=None, direction=0, norm_cutoff=None, **kwargs):
        """ compress the comb. inplace operation
            compress_level: sets level of compression determined by grid on each branch,
                and in get_tt_compress_opts for the spine
            compress_opts: overwrite default. 'tt_compress_opts' can be used to key a
                dict that contains compress_opts for the tensor-train/spine part.

            TODO: change to compress into spine immediately bc branch is already canonicalized
        """
        version = 1
        gtnc = self if inplace else self.copy()
        if gtnc.data is None:
            return gtnc

        if branch_keys is None:
            if direction % 2 == 0:
                branch_inds = range(gtnc.grid.ngrids)
            else:
                branch_inds = range(gtnc.grid.ngrids - 1, -1, -1)
        else:
            if isinstance(branch_keys[0], (int, np.int32, np.int64)):
                branch_inds = branch_keys
            else:
                branch_inds = [gtnc.grid.get_grid_ind(subgr) for subgr in branch_keys]

            if direction % 2 == 0:
                branch_inds = np.sort(list(branch_inds))
            else:
                branch_inds = np.sort(list(branch_inds))[::-1]

        canon_site = gtnc.canon_site  # canonicalized around this site
        if verbose:
            print('GTNC CANON before canon', canon_site)
            print('compress branch inds', branch_inds, branch_inds[::-1], branch_inds[-1])
            print('comb check orthog')
            gtnc.check_orthog()

        if canon_site != branch_inds[-1]:
            gtnc.canonicalize_around_i(branch_inds[-1])
            canon_site = gtnc.canon_site  # branch_inds[-1]  # canonicalized around this site
            ## branch is also canonicalized

        if gtnc.data is None:  ## can happen after canonicalization
            return gtnc

        gtnc.collect_exponents()

        if verbose:
            print('GTNC CANON post canon', canon_site)
            print('comb check orthog')
            gtnc.check_orthog()

        if compress_opts is None:
            compress_opts = {}

        if sub_compress_opts is None:
            tt_compress_opts = compress_opts
        else:
            tt_compress_opts = sub_compress_opts.get(SubCompressConfigType.TT, compress_opts)

        # print('BRANCH INDS', branch_inds[::-1], canon_site)
        for bi in branch_inds[::-1]:

            ## move canonicalization + compress on spine to new site if necessary
            if canon_site != bi:

                comp_opts = {} if version == 2 else tt_compress_opts
                if version == 2:
                    if bi > canon_site:
                        helper.canonize_tens_list(*gtnc.spine[canon_site:bi + 1], inplace=True)
                    elif bi < canon_site:
                        helper.canonize_tens_list(*gtnc.spine[canon_site:bi - 1:-1], inplace=True)
                else:
                    if bi > canon_site:
                        if verbose:   print('compress spine', canon_site, bi + 1)
                        helper.compress_tens_list(*gtnc.spine[canon_site:bi + 1],
                                                  inplace=True, compress_opts=comp_opts)
                    elif bi < canon_site:
                        if verbose:   print('compress spine', canon_site, bi - 1)
                        helper.compress_tens_list(*gtnc.spine[canon_site:bi - 1:-1],
                                                  inplace=True, compress_opts=comp_opts)

                    if verbose and gtnc.data_type is DataType.MPS:
                        print('spine orthog', helper.check_orthog(gtnc.spine))

            branch = gtnc.get_branch(bi)
            if branch is None:
                continue

            if verbose:
                print('compress spine into branch', bi)

            ## compress spine into branch
            spine_tens = gtnc.spine[bi]
            helper.compress_tens_list(spine_tens, branch.get_anchor_tens(), inplace=True,
                                      compress_opts=tt_compress_opts)

            # print('spine tens orthog', helper.check_left_orthog([spine_tens], right_ancillas=(self.spine_ind_id.format(bi),)))
            # print('bi orthog', bi, helper.check_right_orthog(branch.data, left_ancillas=(self.spine_ind_id.format(bi),)))

            ## compress then canonicalize branch
            mod_compress_opts = compress_opts.copy()
            mod_compress_opts['form'] = 'left'

            if verbose and gtnc.data_type is DataType.MPS:
                branch_ext = gtnc._attach_spine_tens_to_branch(bi, is_mps=(gtnc.data_type == DataType.MPS))
                print('check orthog 1', helper.check_orthog(branch_ext))

            branch.compress(inplace=True, compress_opts=mod_compress_opts, canonize=False, norm_cutoff=norm_cutoff)
            branch.canonize(inplace=True, i=0)

            if branch.data is None:
                gtnc.data = None
                gtnc._exponent = 0
                gtnc._sign = 1
                break

            ## move canon from branch into spine
            helper.canonize_tens_list(branch.get_anchor_tens(), spine_tens, inplace=True)
            canon_site = bi

            if verbose and gtnc.data_type is DataType.MPS:
                branch_ext = gtnc._attach_spine_tens_to_branch(bi, is_mps=(gtnc.data_type == DataType.MPS))
                print('check orthog 2', helper.check_orthog(branch_ext), branch_ext.L)

        gtnc.canon_site = canon_site

        if version == 2:
            mod_tt_opts = tt_compress_opts.copy()
            if canon_site == 0:
                mod_tt_opts['form'] = 'left'
                # print('check spine right', helper.check_right_orthog(gtnc.spine))
                helper.compress(gtnc.spine, compress_opts=mod_tt_opts, canonize=False)
                gtnc.canon_site = gtnc.spine.L - 1 if gtnc.spine is not None else None
                # print('check spine left', helper.check_left_orthog(gtnc.spine))
            else:
                mod_tt_opts['form'] = 'right'
                helper.compress(gtnc.spine, compress_opts=mod_tt_opts,
                                canonize=(canon_site != gtnc.spine.L - 1))
                gtnc.canon_site = 0 if gtnc.spine is not None else None

        gtnc.collect_exponents()
        return gtnc

    def compress_v2(self, inplace=True, verbose=False, compress_type=CompressType.SVD, compress_opts=None,
                    sub_compress_opts=None, branch_keys=None, direction=0, norm_cutoff=None, **kwargs):
        """ compress the comb. inplace operation
            compress_level: sets level of compression determined by grid on each branch,
                and in get_tt_compress_opts for the spine
            compress_opts: overwrite default. 'tt_compress_opts' can be used to key a
                dict that contains compress_opts for the tensor-train/spine part.

            TODO: change to compress into spine immediately bc branch is already canonicalized
        """

        if self.data_type is DataType.MPO:
            return self.compress_v1(inplace=inplace, verbose=verbose, compress_type=compress_type,
                                    compress_opts=compress_opts, sub_compress_opts=sub_compress_opts,
                                    branch_keys=branch_keys, direction=direction, **kwargs)

        gtnc = self if inplace else self.copy()
        if gtnc.data is None:
            return gtnc

        # verbose = True

        if branch_keys is None:
            if direction % 2 == 0:
                branch_inds = range(gtnc.grid.ngrids)
            else:
                branch_inds = range(gtnc.grid.ngrids - 1, -1, -1)
        else:
            if isinstance(branch_keys[0], (int, np.int32, np.int64)):
                branch_inds = branch_keys
            else:
                branch_inds = [gtnc.grid.get_grid_ind(subgr) for subgr in branch_keys]

            if direction % 2 == 0:
                branch_inds = np.sort(list(branch_inds))
            else:
                branch_inds = np.sort(list(branch_inds))[::-1]

        canon_site = gtnc.canon_site  # canonicalized around this site
        if canon_site != branch_inds[-1]:
            gtnc.canonicalize_around_i(branch_inds[-1])
            canon_site = gtnc.canon_site  # branch_inds[-1]  # canonicalized around this site
            ## branch is also canonicalized

        # gtnc.collect_exponents()
        gtnc.distribute_exponents()

        if compress_opts is None:
            compress_opts = {}

        if sub_compress_opts is None:
            tt_compress_opts = compress_opts
        else:
            tt_compress_opts = sub_compress_opts.get(SubCompressConfigType.TT, compress_opts)

        if verbose:   print('BRANCH INDS', branch_inds[::-1], canon_site)
        for bi in branch_inds[::-1]:

            ## move canonicalization + compress on spine to new site if necessary
            ## should skip for bi == branch_inds[-1]
            if canon_site != bi:
                comp_opts = tt_compress_opts
                if bi > canon_site:
                    if verbose:   print('compress spine', canon_site, bi + 1)
                    helper.compress_tens_list(*gtnc.spine[canon_site:bi + 1],
                                              inplace=True, compress_opts=comp_opts)
                elif bi < canon_site:
                    if verbose:   print('compress spine', canon_site, bi - 1)
                    helper.compress_tens_list(*gtnc.spine[canon_site:bi - 1:-1],
                                              inplace=True, compress_opts=comp_opts)

            if verbose:
                print('spine orthog', helper.check_orthog(gtnc.spine))

            ## canonize to left canon, compress to right (spine + branch)
            if verbose:    print('canonize branch', bi)
            branch = gtnc.get_branch(bi)
            if branch is None:
                print('skipping compress branch because None', bi)
                continue

            branch_extended = gtnc._attach_spine_tens_to_branch(bi, is_mps=(gtnc._data_type is DataType.MPS))
            branch_extended = helper.compress(branch_extended, scale=True, canonize=True, norm_cutoff=norm_cutoff)
            branch_extended.distribute_exponent()

            if branch_extended is None:
                gtnc.data = None
                gtnc._exponent = 0
                gtnc._sign = 1
                break

            ## need below? yes...
            gtnc._update_spine_tens_and_branch(bi, branch_extended)

            canon_site = bi
            if verbose and gtnc.data_type is DataType.MPS:
                # print('spine orthog', helper.check_orthog(gtnc.spine))
                print('check gtnc orthog')
                gtnc.check_orthog()

        gtnc.canon_site = canon_site
        gtnc.collect_exponents()
        return gtnc

    def compress_v3(self, inplace=True, verbose=False, compress_type=CompressType.SVD, compress_opts=None,
                    sub_compress_opts=None, branch_keys=None, direction=0, norm_cutoff=None, **kwargs):
        """ compress the comb. inplace operation
            compress_level: sets level of compression determined by grid on each branch,
                and in get_tt_compress_opts for the spine
            compress_opts: overwrite default. 'tt_compress_opts' can be used to key a
                dict that contains compress_opts for the tensor-train/spine part.

            TODO: change to compress into spine immediately bc branch is already canonicalized
        """

        gtnc = self if inplace else self.copy()
        if gtnc.data is None:
            return gtnc

        if branch_keys is None:
            if direction % 2 == 0:
                branch_inds = range(gtnc.grid.ngrids)
            else:
                branch_inds = range(gtnc.grid.ngrids - 1, -1, -1)
        else:
            if isinstance(branch_keys[0], (int, np.int32, np.int64)):
                branch_inds = branch_keys
            else:
                branch_inds = [gtnc.grid.get_grid_ind(subgr) for subgr in branch_keys]

            if direction % 2 == 0:
                branch_inds = np.sort(list(branch_inds))
            else:
                branch_inds = np.sort(list(branch_inds))[::-1]

        canon_site = gtnc.canon_site  # canonicalized around this site
        if canon_site != branch_inds[-1]:
            gtnc.canonicalize_around_i(branch_inds[-1])
            canon_site = gtnc.canon_site  # branch_inds[-1]  # canonicalized around this site
            ## branch is also canonicalized

        gtnc.collect_exponents()
        # gtnc.distribute_exponents()

        if compress_opts is None:
            compress_opts = {}

        if sub_compress_opts is None:
            tt_compress_opts = compress_opts
        else:
            tt_compress_opts = sub_compress_opts.get(SubCompressConfigType.TT, compress_opts)

        if verbose:    print('BRANCH INDS', branch_inds[::-1], canon_site)
        for bi in branch_inds[::-1]:

            ## move canonicalization + compress on spine to new site if necessary
            ## should skip for bi == branch_inds[-1]
            if canon_site != bi:
                comp_opts = tt_compress_opts
                if bi > canon_site:
                    if verbose:    print('compress spine', canon_site, bi + 1)
                    helper.compress_tens_list(*gtnc.spine[canon_site:bi + 1],
                                              inplace=True, compress_opts=comp_opts)
                elif bi < canon_site:
                    if verbose:    print('compress spine', canon_site, bi - 1)
                    helper.compress_tens_list(*gtnc.spine[canon_site:bi - 1:-1],
                                              inplace=True, compress_opts=comp_opts)

            if verbose:
                print('spine orthog', helper.check_orthog(gtnc.spine))

            if verbose:
                print('canonize branch', bi)
            branch = gtnc.get_branch(bi)

            if branch is None:
                print('skipping compress branch because None', bi)
                continue

            ## canonize spine into branch
            spine_tens = gtnc.spine[bi]
            helper.compress_tens_list(spine_tens, branch.get_anchor_tens(), inplace=True,
                                      compress_opts={})

            ## canonize to left canon, compress to right (spine + branch)
            branch = branch.compress(inplace=True, scale=False, canonize=True, compress_opts=compress_opts,
                                     norm_cutoff=norm_cutoff)

            if branch is None:
                gtnc.data = None
                gtnc._exponent = 0
                gtnc._sign = 1
                break

            ## compress branch into spine
            helper.compress_tens_list(branch.get_anchor_tens(), spine_tens, inplace=True,
                                      compress_opts=tt_compress_opts)

            canon_site = bi
            if verbose and gtnc.data_type is DataType.MPS:
                # print('spine orthog', helper.check_orthog(gtnc.spine))
                print('check gtnc orthog')
                gtnc.check_orthog()

        gtnc.canon_site = canon_site

        spine_tens = gtnc.spine[branch_inds[0]]
        tens0_norm = spine_tens.norm()
        spine_tens.modify(apply=lambda x: x / tens0_norm)
        gtnc._exponent += np.log10(tens0_norm)

        gtnc.collect_exponents()
        return gtnc

    def compress_rdm(self, inplace=True, verbose=False, direction=1, compress_opts=None, sub_compress_opts=None,
                     open_end=False, left_env=None, right_env=None, back_compress=True,
                     **kwargs) -> 'GridTN1DComb':

        from mp_compress_rdm_comb import compress_rdm as compress_rdm_mp

        return compress_rdm_mp(self, direction=direction, inplace=inplace, compress_opts=compress_opts,
                               verbose=verbose, back_compress=back_compress)

    def compress_branch(self, branch_ind, inplace=True, cur_orthog=None, compress_opts=None):
        """ compress the branch positioned at branch_num
            a little outdated
        """
        gtnc = self if inplace else self.copy()

        branch = gtnc.branches[branch_ind]

        if compress_opts is None:
            compress_opts = {}
        else:
            compress_opts = compress_opts.copy()
        tt_compress_opts = compress_opts.pop('tt_compress_opts', {})

        if gtnc.spine is not None:

            if cur_orthog is None:
                gtnc.canonicalize_around_i(branch_ind)
            elif cur_orthog != branch_ind:
                branch.canonize(inplace=True, i=0)
                if branch_ind < cur_orthog:
                    helper.canonize_tens_list(*gtnc.spine[cur_orthog:branch_ind - 1:-1], inplace=True)
                else:
                    helper.canonize_tens_list(*gtnc.spine[cur_orthog:branch_ind + 1], inplace=True)
            else:
                pass

            ## canonicalize spine into branch
            spine_tens = gtnc.spine[branch_ind]
            helper.canonize_tens_list(spine_tens, branch[0], inplace=True)

        mod_compress_opts = compress_opts.copy()
        mod_compress_opts['form'] = 'right'
        branch.compress(compress_opts=compress_opts)

        if gtnc.spine is not None:
            helper.compress_tens_list(branch[0], gtnc.spine[branch_ind], inplace=True, compress_opts=tt_compress_opts)

        return gtnc

    def contract(self, ax_select: dict['Axis', int] = None) -> Optional['qtn.Tensor']:
        """ contract comb TN
            check_collisions: check collisions of ind labels between branches
            fuse: whether or not to fuse inds of a single branch
            out_ind_id:  ind_id to use when fusing indices of branches
        """
        tens = None
        self.collect_exponents()  # not necessary?

        if ax_select is None:
            ax_select = {}

        if self.data is None:
            return None

        # for branch in self.active_branches:
        #     gk = self.grid.get_grid_ind(branch.grid)

        for gk in range(self.grid.ngrids)[::-1]:

            branch = self.get_branch(gk)
            # gk = self.grid.get_grid_ind(branch.grid)

            if branch is None:
                tens_k = self.spine[gk]

            else:
                tens_k = branch.get_data(ax_select=ax_select)
                # exponent += branch.data.exponent

                left_anc = (self.spine.site_ind_id.format(gk),)
                if branch.data_type == DataType.MPS:
                    ## actually split by axes
                    out_axes = [ax for ax in branch.grid.axes if ax not in ax_select]
                    tens_k = qtn.Tensor(data=tens_k, inds=left_anc + tuple([f'i({ax})' for ax in out_axes]))
                elif branch.data_type == DataType.MPO:
                    out_axes = [ax for ax in branch.grid.axes if ax not in ax_select]
                    tens_k = qtn.Tensor(data=tens_k, inds=left_anc + tuple([f'o({ax})' for ax in out_axes] +
                                                                           [f'i({ax})' for ax in out_axes]))
                # if fuse:
                # try:
                #     tens_k.fuse({f'i({branch.grid.gridID})':
                #              tuple([branch.data.site_ind_id.format(i) for i in range(branch.data.L)])},
                #                 inplace=True)
                # except AttributeError:   # branch was an MPO
                #     tens_k.fuse(
                #         {f'o({branch.grid.gridID})':
                #              tuple([branch.data.upper_ind_id.format(i) for i in range(branch.data.L)]),
                #          f'i({branch.grid.gridID})':
                #              tuple([branch.data.lower_ind_id.format(i) for i in range(branch.data.L)])},
                #         inplace=True)

                # if self.spine is not None:
                # if tens_k is not None:
                tens_k = tens_k.contract(self.spine[gk])
                # else:                    tens_k = self.spine[gk]

            if tens is None:
                tens = tens_k.copy()
            elif np.isscalar(tens):
                if tens_k is not None:
                    tens_k.modify(apply=lambda x: x * tens)
                    tens = tens_k
            else:
                if tens_k is not None:
                    tens = tens.contract(tens_k)

        return tens * (10 ** self._exponent) * self._sign

    def mps_to_diag_mpo(self, lower_ind_id='i({})', upper_ind_id='o({})', inplace=False) -> 'GridTN':

        assert (self.data_type == DataType.MPS), f'gtn needs to MPS type not {self.data_type}'

        gtnc = self if inplace else self.copy(deep=False)

        # if not inplace:
        #     gtnc.spine = self.spine.copy()
        #     gtnc.exponent = self.exponent
        #     gtnc.sign = self.sign

        for gk, branch in gtnc.branches.items():
            if branch is not None:
                branch.mps_to_diag_mpo(lower_ind_id=lower_ind_id, upper_ind_id=upper_ind_id, inplace=True)
        gtnc._data_type = DataType.MPO

        return gtnc

    #######################
    ## change dimensions ##
    #######################

    def pad_self_to_new_grid(self, new_grid: 'GridsComb', branch_inds: Sequence[int] = None, inplace=False,
                             explicit_pad_mps=False, explicit_pad_mpo=False):
        """ new_grids: list of grids to expand TN to
            explicit_pad:  whether to explicitly add ones mps/iden mpos
        """
        # print('PAD SELF TO GRID')

        tnc = self if inplace else self.copy()

        # print('tnc', tnc)
        # print('grid', new_grid)

        if branch_inds is None:
            # branch_inds = [new_grid.get_grid_ind(gl) for gl in tnc.grid.grids]
            branch_grids = [new_grid.get_grid_with_all_axes(gl.axes) for gl in tnc.grid.grids]
            branch_inds = [new_grid.get_grid_ind(gr) for gr in branch_grids]
        # print('branch inds', branch_inds)

        if tnc.spine is not None:
            # assert (np.diff(branch_inds) > 0), 'new grid must not reorder self.grids'
            if self.grid.ngrids == 0:   ## spine is scalar number
                new_spine = helper.ones_mps(new_grid.ngrids, 1, site_ind_id='tt({})', site_tag_id='TT({})')
                helper.scalar_multiply(new_spine, self.spine, inplace=True)
            else:
                new_spine = helper.pad_mps(tnc.spine, branch_inds, len(new_grid))
        else:
            new_spine = None

        # print('new spine', tnc.spine, new_spine)

        new_branches = {}
        for gl in new_grid.grids:

            tnc_gr = tnc.grid.get_grid_with_any_axes(gl.axes)
            if tnc_gr is None:
                tnc_branch = None
            else:
                tnc_branch = tnc.get_branch(tnc_gr)

            if tnc_branch is not None:
                tnc_branch = gl.pad_gtn_to_grid(tnc_branch)

                old_idx = self.grid.get_grid_ind(tnc_gr)
                new_idx = new_grid.get_grid_ind(gl)
                tnc_branch.data[0].reindex({tnc.spine.site_ind_id.format(old_idx):
                                                tnc.spine.site_ind_id.format(new_idx)}, inplace=True)
                new_branches[gl] = tnc_branch

            else:
                if explicit_pad_mps:
                    new_mps = gl.get_ones_mps()
                    new_branches[gl] = new_mps
                elif explicit_pad_mpo:
                    new_mpo = gl.get_iden_mpo()
                    new_branches[gl] = new_mpo
                else:
                    if new_spine is not None:
                        idx = new_grid.get_grid_ind(gl)
                        new_spine[idx].isel({new_spine.site_ind_id.format(idx): 0}, inplace=True)
                    # new_branches[gl] = None

        tnc.constant_axes += [ax for ax in new_grid.axes if ax not in tnc.grid.axes]

        tnc.grid = new_grid
        tnc._branches = new_branches
        tnc._set_data_type([new_branches[nb].data_type for nb in new_branches.keys()])
        tnc._spine = new_spine
        tnc.canon_site = None
        tnc._update_active_grids()
        return tnc

    def pad_branch_to_axes(self, new_axes, inplace=False):
        """ Map data on a branch to a new set of axes
        """
        raise NotImplementedError

    def take_outerproduct(self, other_gtn, virtual=False):
        """ take outerproduct with another gridTN
        """
        if isinstance(other_gtn, GridTN):
            self.set_branch(other_gtn.grid, other_gtn if virtual else other_gtn.copy())
        elif isinstance(other_gtn, GridTN1DComb):
            if [gr in self.grid.grids for gr in other_gtn.grid.grids]:
                assert ([gr not in self.active_branches for gr in other_gtn.active_branches]), \
                    'cannot take outerproduct with overlapping grids'
                raise NotImplementedError
                # pad other to self, take outerproduct of the spines, combine branches
            elif [gr in other_gtn.grid.grids for gr in self.grid.grids]:
                assert ([gr not in self.active_branches for gr in other_gtn.active_branches]), \
                    'cannot take outerproduct with overlapping grids'
                raise NotImplementedError
                # pad self to other, take outerproduct of the spines, combine branches
            else:
                raise NotImplementedError

    #####################
    ## data processing ##
    #####################

    # def interpolate_data(self, nsites, inplace=False) -> 'GridTN1DComb':
    #     """ extend grid s.t. data is interpolated between existing grid points
    #     """
    #     raise NotImplementedError
    #
    # def gaussian_smooth_data(self, inplace=False, sigma=0.3, proc_axes: list[Axis] = None, mode='wrap', compress=True,
    #                          **compress_opts) -> 'GridTN1DComb':
    #     """ process the data in some way
    #     """
    #     gtn = self if inplace else self.copy(deep=False)
    #     blur_mpo = self.grid.gaussian_smooth_mpo(sigma=sigma, proc_axes=proc_axes, mode=mode)
    #     if blur_mpo is not None:
    #         gtn.apply(blur_mpo, inplace=True, compress=True, **compress_opts)
    #     return gtn
    #
    #
    # def apply_absorbing_bc(self, x_ax, v_ax, left_bc, right_bc, inplace=False, compress=True, **compress_opts):
    #     """ apply absorbing bc to MPS state, assumes TN dimensions are x1,x2,x3,v1,v2,v3
    #         (prev: apply absorbing bc to derivative MPO, assumes TN dimensions are x1,x2,x3,v1,v2,v3)
    #     """
    #     gtn = self if inplace else self.copy(deep=False)
    #     absorb_mpo = self.grid.absorbing_bc_mpo(x_ax, v_ax, left_bc, right_bc, compress=False, **compress_opts)
    #     if absorb_mpo is not None:
    #         gtn.apply(absorb_mpo, inplace=True, compress=True, **compress_opts)
    #     return gtn
    #
    #
    # def apply_reflecting_v_bc(self, x_ax, v_ax, left_bc, right_bc, inplace=False, compress=True, **compress_opts):
    #     """ apply reflecting_v bc to state.
    #         apply to derivative MPO?
    #     """
    #     raise NotImplementedError

    ###############################################
    ### build derivative and integral operators ###
    ###############################################

    # @profile
    def apply_elemental_multiply_op(self, lower_ind_id='i({})', upper_ind_id='o({})', axes=None, add_cc=False,
                                    take_mps_cc=False, compress=False, compress_opts=None):
        """ apply elemental multiply operator to self (e.g. convolution or d_ijk fct)
        """
        self.collect_exponents()
        add_cc = self.grid.has_basis_k_real()

        new_branches = {}
        for gk, b in self.branches.items():
            b_axes = None if axes is None else [ax for ax in gk.axes if ax in axes]
            new_b = b.apply_elemental_multiply_op(lower_ind_id=lower_ind_id, upper_ind_id=upper_ind_id, axes=b_axes,
                                                  add_cc=add_cc, take_mps_cc=take_mps_cc, compress=compress,
                                                  compress_opts=compress_opts)
            new_branches[gk] = new_b

        spine = self.spine.copy() if self.spine is not None else None
        if take_mps_cc:
            spine.conj(inplace=True)
        spine.mangle_inner_()

        if add_cc:  # gtnc -> gtnc + complex conj
            # spine + spine_CC, but don't sum over physical inds (bonds to the branches)
            for i in range(spine.L):
                tens = spine[i]
                tens.direct_product(tens.conj(), inplace=True)

        gtn_mpo = self.__class__(self.grid, branches=new_branches, spine=spine, exponent=self._exponent,
                                 sign=self._sign)
        gtn_mpo.is_constant = self.is_constant
        gtn_mpo.constant_axes = self.constant_axes
        if compress:
            gtn_mpo.compress(inplace=True, compress_opts=compress_opts)

        return gtn_mpo

    # def apply_xmultiply_mpo(self, x_axes=None, inplace=False, compress=True, **compress_opts):
    #     """ perform x*f(x) operation, where x_dims specifies which axis/axes to multiply
    #         eg. 1D:  x*f(x,y,...)
    #             2D:  x*y*f(x,y,...)
    #     """
    #     raise NotImplementedError
    #
    # def take_firstderivative(self, ax, bc, deriv_params=None, recalc=False, compress=True, **compress_opts):
    #     """ self.order: finite difference expansion order """
    #     raise NotImplementedError
    #
    # def take_secondderivative(self,ax1,ax2,bc1,bc2,recalc=False,compress=True,**compress_opts):
    #     """ self.order: finite difference expansion order """
    #     raise NotImplementedError

    # # @profile
    # def integrate(self, integ_axes=None, is_sqrt=False,
    #                new_grid=None, new_ax_deriv_configs=None,
    #                compress=False, compress_opts=None, **kwargs):
    #     """ integrate over the specified axes
    #         ancilla_reindex:  if is_sqrt, reindex ancilla inds such that they are not contracted together
    #             dict[old str: new str]
    #     """
    #     if new_ax_deriv_configs is None:
    #         new_ax_deriv_configs = self.ax_deriv_configs
    #
    #     if new_grid is None:
    #         new_grid = self.grid
    #         # if integ_axes is None or len(integ_axes) == self.grid.ndim:     # integrate over all axes
    #         #     new_grid = self.grid
    #         # else:
    #         #     new_grid = self.grid.get_subgrid([ax for ax in self.grid.axes if ax not in integ_axes])
    #
    #
    #     if self.data is None:
    #         return new_grid.make_empty_gridTN(ax_deriv_configs=new_ax_deriv_configs)
    #
    #
    #     new_branches = {}
    #     new_spine = qtn.TensorNetwork([])
    #
    #     self.collect_exponents()
    #     exponent = self._exponent if not is_sqrt else self._exponent * 2
    #     sign = self._sign if not is_sqrt else 1.
    #
    #     spine = self.spine
    #
    #     for gk, b in self.branches.items():
    #
    #         gk_idx = self.grid.get_grid_ind(gk)
    #
    #         if integ_axes is not None:
    #             active_axes = [ax for ax in integ_axes if ax in b.grid.axes]
    #         else:
    #             active_axes = b.grid.axes
    #
    #         if len(active_axes) == 0:   # no part of branch is integrated over
    #             new_ind = new_grid.get_grid_ind(gk)
    #             old_tt_ind = spine.site_ind_id.format(gk_idx)
    #
    #             if is_sqrt:
    #                 # raise NotImplementedError
    #                 ## need to square spine_tens, square the branch (like elemental multiply?)
    #
    #
    #                 ## spine tens * cc
    #                 spine_tens: qtn.Tensor = spine[gk_idx].copy()
    #                 orig_inds = spine_tens.inds
    #                 spine_tens_cc = spine_tens.conj()
    #                 spine_tens_cc.reindex({ind: ind+'_' for ind in orig_inds}, inplace=True)
    #                 spine_tens = spine_tens.contract(spine_tens_cc)
    #                 # spine_tens.fuse({ind: (ind,ind+'_') for ind in orig_inds}, inplace=True)
    #                 spine_tens.fuse({old_tt_ind: (old_tt_ind, old_tt_ind+'_')}, inplace=True)
    #                 spine_tens.drop_tags()
    #                 spine_tens.add_tag(spine.site_tag_id.format(new_ind))
    #                 spine_tens.reindex({old_tt_ind: spine.site_ind_id.format(new_ind)}, inplace=True)
    #
    #                 ## branch * cc
    #                 branch_copy: 'GridTN' = b.copy()
    #                 branch_cc = branch_copy.conj(mangle_inner=True)
    #                 branch_cc.data[0].reindex({old_tt_ind: old_tt_ind+'_'}, inplace=True)
    #                 branch_copy.elemental_multiply(branch_cc, inplace=True, zipup=True, compress_opts={'form':'right'})
    #                 branch_copy.data[0].fuse({old_tt_ind: (old_tt_ind, old_tt_ind+'_')}, inplace=True)
    #                 branch_copy.data[0].reindex({old_tt_ind: spine.site_ind_id.format(new_ind)}, inplace=True)
    #
    #                 # print('branch_copy orthog', helper.check_right_orthog(branch_copy))
    #                 helper.compress_tens_list( branch_copy.data[0], spine_tens, inplace=True )
    #
    #                 new_spine.add(spine_tens)
    #                 new_branches[gk] = branch_copy
    #
    #                 # print('compressed', gk, new_spine, spine_tens, new_branches[gk])
    #
    #             else:
    #                 new_ind = new_grid.get_grid_ind(gk)
    #                 spine_tens = spine[gk_idx].copy()
    #                 spine_tens.drop_tags()
    #                 spine_tens.add_tag(spine.site_tag_id.format(new_ind))
    #                 spine_tens.reindex({spine.site_ind_id.format(gk_idx): spine.site_ind_id.format(new_ind)},
    #                                    inplace=True)
    #                 new_spine.add(spine_tens)
    #                 branch_copy = b.copy()
    #                 branch_copy.data[0].reindex({spine.site_ind_id.format(gk_idx): spine.site_ind_id.format(new_ind)},
    #                                             inplace=True)
    #                 new_branches[gk] = branch_copy
    #             continue
    #
    #         ## will be (at least partially) integrated
    #         if is_sqrt:
    #             tt_ind = spine.site_ind_id.format(gk_idx)
    #             ancilla_reindex = {tt_ind: tt_ind+'_'}
    #         else:
    #             ancilla_reindex = None
    #
    #         new_b = b.integrate(integ_axes=active_axes, is_sqrt=is_sqrt, ancilla_reindex=ancilla_reindex,
    #                             compress_opts=compress_opts)
    #
    #         if isinstance(new_b, NUM_TYPES):
    #             exponent += np.log10(np.abs(new_b))
    #             sign *= np.exp(-1.j * np.angle(new_b))
    #         elif isinstance(new_b, qtn.Tensor):
    #
    #             if is_sqrt:
    #                 orig_inds = spine[gk_idx].inds
    #                 sqrt_tens = spine[gk_idx].reindex({t_ind: t_ind+'_' for t_ind in orig_inds})
    #                 sqrt_tens.conj(inplace=True)
    #                 spine_tens = qtn.tensor_contract(spine[gk_idx], sqrt_tens, new_b)
    #             else:
    #                 spine_tens = spine[gk_idx].contract(new_b)
    #
    #             if isinstance(spine_tens, NUM_TYPES):
    #                 exponent += np.log10(np.abs(spine_tens))
    #                 sign *= np.exp(-1.j * np.angle(spine_tens))
    #                 continue
    #
    #             spine_tens.drop_tags()
    #             # print('spine tens integ', integ_axes, spine_tens, gk, gk in new_grid.grids)
    #             if gk in new_grid.grids:
    #                 new_ind = new_grid.get_grid_ind(gk)
    #                 if integ_axes is not None and not all([ax in integ_axes for ax in gk.axes]):
    #                     spine_tens.new_ind(spine.site_ind_id.format(new_ind), size=1, axis=-1)
    #                     # print('add ind?', spine_tens)
    #                 spine_tens.add_tag(spine.site_tag_id.format(new_ind))
    #                 new_spine.add(spine_tens)
    #             else:   ## combine spine tens with other spine tensors
    #                 # gr_ind = self.grid.get_grid_ind(gk)
    #                 # print('select grid ind', gk, gr_ind, spine.site_ind_id.format(gr_ind))
    #                 # assert(spine_tens.ind_size(spine.site_ind_id.format(gr_ind)) == 1), \
    #                 #     'size of integrated branch should be one or included in new grid'
    #                 # spine_tens.isel({spine.site_ind_id.format(gr_ind): 0}, inplace=True)
    #
    #                 try:
    #                     # tens_str = spine.site_tag_id.format(new_ind)
    #                     # spine_tens.add_tag(tens_str)
    #                     # new_spine.add(spine_tens)
    #                     # print('new spine pre', new_spine, tens_str)
    #                     # print(new_spine.select_tensors(tens_str))
    #                     # out = new_spine.contract_tags([tens_str], inplace=True)
    #                     # print('new spine post', new_spine, out)
    #                     ### new_ind is from previous iteration in new grid
    #                     tmp = new_spine.select_tensors(spine.site_tag_id.format(new_ind))[0]
    #                     # print('integrate', branch_copy.max_bond(), tmp.shape, spine_tens.shape)
    #                     out = tmp.contract(spine_tens)
    #                     tmp.modify(data=out.data, inds=out.inds)
    #                 except NameError:
    #                     # print('contract here', spine[gk_idx + 1], spine_tens)
    #                     spine[gk_idx + 1] = spine[gk_idx + 1].contract(spine_tens)
    #                     # print('spine?', spine[gk_idx + 1])
    #                 # print('integ spine', spine)
    #
    #         elif isinstance(new_b, GridTN1D):
    #             new_branches[gk] = new_b
    #             spine_tens = spine[gk_idx].copy()
    #             if is_sqrt:
    #                 sqrt_tens = spine_tens.reindex({t_ind: t_ind + '_' for t_ind in spine_tens.inds})
    #                 sqrt_tens.conj(inplace=True)
    #                 spine_tens = spine_tens.contract(sqrt_tens)
    #                 spine_tens.fuse({tt_ind: (tt_ind, tt_ind+'_')}, inplace=True)
    #             new_spine.add(spine_tens)
    #         else:
    #             raise NotImplementedError
    #
    #
    #     if len(new_branches) > 0:
    #         if new_spine.num_tensors > 0:
    #             new_spine = new_spine.view_like(spine, inplace=True, L=new_grid.ngrids)
    #             new_spine.fuse_multibonds(inplace=True)
    #         else:
    #             new_spine = None
    #
    #         # print('new grid', new_grid, new_spine)
    #         gtn_mpo = self.__class__(new_grid, branches=new_branches, spine=new_spine,
    #                                  exponent = exponent, sign = sign,
    #                                  ax_deriv_configs=new_ax_deriv_configs)
    #         # gtn_mpo.exponent = exponent
    #         # gtn_mpo.sign = sign
    #         return gtn_mpo
    #     else:       ## is number
    #         val = 10**exponent * sign
    #         if new_spine.num_tensors > 0:
    #             # val *= new_spine.contract().data.item()
    #             val *= new_spine.contract()
    #         return val

    def integrate(self, integ_axes=None, is_sqrt=False, new_grid=None, new_ax_deriv_configs=None,
                  exclude_weights=False, compress=False, compress_opts=None,
                  ancilla_reindex: dict[str, str] = None, **kwargs) \
            -> Union[Numeric, qtn.Tensor]:
        """ integrate dx f(x) O(x) dx
            integrate dx g*(x) O(x) g(x) dx
        """
        if self.data is None:
            return np.nan

        return self.meas_expec(None, is_sqrt=is_sqrt, integ_axes=integ_axes, new_grid=new_grid,
                               exclude_weights=exclude_weights, new_ax_deriv_configs=new_ax_deriv_configs,
                               ancilla_reindex=ancilla_reindex, compress=compress, compress_opts=compress_opts)

    # @profile
    def meas_expec(self, obs_gtn: Optional[Union[dict['Grid', 'GridTN'], 'GridTN1DComb']], integ_axes=None,
                   is_sqrt=False, new_ax_deriv_configs=None, new_grid=None, exclude_axes=None, exclude_weights=False,
                   compress=False, compress_opts=None, **kwargs):
        """ integrate dx f(x) O(x) dx
            integrate dx g*(x) O(x) g(x) dx
        """
        if self.data is None:
            return 0.0 if (integ_axes is None or integ_axes == self.grid.axes) else None

        if isinstance(self.spine, (float, complex)):
            if obs_gtn is None:
                return np.abs(self.spine)**2 if is_sqrt else self.spine
            else:
                raise NotImplementedError

        if exclude_axes is not None and set(exclude_axes) == set(self.grid.axes):
            return None

        if integ_axes is None:
            integ_axes = self.grid.axes

        self.collect_exponents()

        new_branches = {}
        new_spine = qtn.TensorNetwork([])
        exponent = self._exponent if not is_sqrt else self._exponent * 2
        sign = self._sign if not is_sqrt else 1.0
        output_is_gtn = False

        # print('meas expec')
        # print('gtn', self.max_bond())
        # print('obs', obs_gtn.max_bond() if obs_gtn is not None else None)
        # print('gtn', self)
        # print('obs', obs_gtn)

        obs_gtn = {} if obs_gtn is None else obs_gtn

        if isinstance(obs_gtn, GridTN1DComb):
            obs_gtn.collect_exponents()
            obs_gtn.pad_self_to_new_grid(self.grid)
            obs_gtn_spine = obs_gtn.spine
            if obs_gtn.spine_ind_id == self.spine_ind_id:
                obs_gtn.spine_ind_id = obs_gtn_spine.site_ind_id + '_op_'
            exponent += obs_gtn._exponent
        else:
            obs_gtn_spine = None

        spine = self.spine.copy()
        # if obs_gtn_spine is not None:
        #     obs_gtn.spine_ind_id = obs_gtn.spine_ind_id + '_o_'

        for gk in self.grid.grids:
            gk_idx = self.grid.get_grid_ind(gk)
            tt_ind = self.spine_ind_id.format(gk_idx)

            gk_integ_axes = [ax for ax in integ_axes if ax in gk.axes]
            exclude_gk_axes = None if exclude_axes is None else [ax for ax in exclude_axes if ax in gk.axes]
            exclude_gk_branch = exclude_gk_axes == list(gk.axes)

            if exclude_gk_branch:
                # ## add spine tens
                # contract_spine_list = [spine[gk_idx]]
                # if obs_gtn_spine is not None:
                #     contract_spine_list += [obs_gtn_spine[gk_idx]]
                # if is_sqrt:
                #     spine_tens_cc = spine[gk_idx].conj()
                #     spine_tens_cc.reindex({t_ind: t_ind + '_' for t_ind in spine_tens_cc.inds}, inplace=True)
                #     contract_spine_list += [spine_tens_cc]
                #
                # spine_tens = qtn.tensor_contract(*contract_spine_list)
                # new_spine.add(spine_tens)
                continue

            b = self.get_branch(gk)

            try:
                obs_gtn_gk = obs_gtn[gk]
                if b is None:
                    b = gk.get_ones_mps()

                # print('GTN1D COMB OBS GTN', gk, obs_gtn[gk])
                # print('b', b)

                zipup_direction = -1 if b.get_anchor_ind() == 0 else 1
                expec = b.meas_expec(obs_gtn_gk, integ_axes=gk_integ_axes, is_sqrt=is_sqrt,
                                     ancilla_reindex={tt_ind: tt_ind + '_'}, exclude_weights=exclude_weights,
                                     zipup=True, zipup_direction=zipup_direction)

                if isinstance(expec, NUM_TYPES):
                    exponent += np.log10(np.abs(expec))
                    sign *= np.exp(-1.j * np.angle(expec))

                elif isinstance(expec, qtn.Tensor):
                    contract_spine_list = [expec, spine[gk_idx]]
                    if obs_gtn_spine is not None:
                        contract_spine_list += [obs_gtn_spine[gk_idx]]
                    if is_sqrt:
                        spine_tens_cc = spine[gk_idx].conj()
                        spine_tens_cc.reindex({t_ind: t_ind + '_' for t_ind in spine_tens_cc.inds}, inplace=True)
                        contract_spine_list += [spine_tens_cc]

                    # spine_tens = qtn.tensor_contract(*contract_spine_list)
                    # new_spine.add(spine_tens)
                    for tens in contract_spine_list:
                        tens.add_tag(spine.site_tag_id.format(gk_idx))
                        new_spine.add(tens)

                elif isinstance(expec, GridTN1D):
                    output_is_gtn = True

                    spine_tens = spine[gk_idx].copy()
                    contract_spine_list = [spine_tens]
                    # expec_anchor = expec[b.get_anchor_ind()]

                    if obs_gtn_spine is not None:
                        contract_spine_list += [obs_gtn_spine[gk_idx]]

                    if is_sqrt:
                        spine_tens_cc = spine[gk_idx].conj()
                        spine_tens_cc.reindex({t_ind: t_ind + '_' for t_ind in spine_tens_cc.inds}, inplace=True)
                        contract_spine_list += [spine_tens_cc]

                    tens_b = expec[b.get_anchor_ind()]
                    b_inds_L = [ind for ind in tens_b.inds if ind not in [tt_ind, tt_ind + '_']]
                    tens_b1, tens_b2 = tens_b.split(b_inds_L, absorb='right', bond_ind='tmp',
                                                    cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                    # print('expec to spine tens b2', tens_b2)
                    contract_spine_list += [tens_b2]
                    expec_anchor = expec[b.get_anchor_ind()]
                    expec_anchor.modify(data=tens_b1.data, inds=tens_b1.inds)

                    # spine_tens = qtn.tensor_contract(*contract_spine_list)
                    # spine_tens.reindex({'tmp': tt_ind}, inplace=True)
                    # expec_anchor.reindex({'tmp': tt_ind}, inplace=True)
                    # new_spine.add(spine_tens)

                    spine_tens.reindex({tt_ind: f'tmp_ancb{gk_idx}'}, inplace=True)
                    tens_b2.reindex({tt_ind: f'tmp_ancb{gk_idx}'}, inplace=True)
                    tens_b2.reindex({'tmp': tt_ind}, inplace=True)
                    expec_anchor.reindex({'tmp': tt_ind}, inplace=True)
                    for tens in contract_spine_list:
                        tens.add_tag(spine.site_tag_id.format(gk_idx))
                        new_spine.add(tens)

                    new_branches[gk] = expec

                elif expec is None:
                    return 0.

                else:
                    raise NotImplementedError

            except KeyError:  ## no branch in obs_gtn
                if len(gk_integ_axes) > 0:  ## partial integration in branch

                    if b is None:
                        expec = 1.0
                    else:
                        expec = b.integrate(integ_axes=gk_integ_axes, is_sqrt=is_sqrt,
                                            ancilla_reindex={tt_ind: tt_ind + '_'})

                    if isinstance(expec, NUM_TYPES):
                        exponent += np.log10(np.abs(expec))
                        sign *= np.exp(-1.j * np.angle(expec))
                    elif isinstance(expec, qtn.Tensor):
                        # ind = self.grid.grid_inds[gk]

                        expec.drop_tags()
                        contract_spine_list = [expec, spine[gk_idx]]
                        if obs_gtn_spine is not None:
                            contract_spine_list += [obs_gtn_spine[gk_idx]]
                        if is_sqrt:
                            spine_tens_cc = spine[gk_idx].conj()
                            spine_tens_cc.reindex({t_ind: t_ind + '_' for t_ind in spine_tens_cc.inds}, inplace=True)
                            contract_spine_list += [spine_tens_cc]

                        # spine_tens = qtn.tensor_contract(*contract_spine_list, preserve_tensor=True)
                        # new_spine.add(spine_tens)

                        for tens in contract_spine_list:
                            tens.add_tag(spine.site_tag_id.format(gk_idx))
                            new_spine.add(tens)

                    elif isinstance(expec, qtn.TensorNetwork):
                        output_is_gtn = True

                        spine_tens = spine[gk_idx].copy()
                        contract_spine_list = [spine_tens]
                        if obs_gtn_spine is not None:
                            contract_spine_list += [obs_gtn_spine[gk_idx]]
                        if is_sqrt:
                            spine_tens_cc = spine[gk_idx].conj()
                            spine_tens_cc.reindex({t_ind: t_ind + '_' for t_ind in spine_tens_cc.inds}, inplace=True)
                            contract_spine_list += [spine_tens_cc]

                        ## should do SVD/QR decomp of anchor tensr to avoid doing fuse_multibonds()
                        raise NotImplementedError('the following has not been checked')
                        # spine_tens = qtn.tensor_contract(*contract_spine_list)
                        # temp = qtn.TensorNetwork([spine_tens, expec[ b.get_anchor_ind()] ], virtual=True)
                        # temp.fuse_multibonds(inplace=True)
                        # new_spine.add(spine_tens)

                        tens_b = expec[b.get_anchor_ind()]
                        b_inds_L = [ind for ind in tens_b.inds if ind not in [tt_ind, tt_ind + '_']]
                        tens_b1, tens_b2 = tens_b.split(b_inds_L, absorb='right', bond_ind='tmp',
                                                        cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                        # print('expec to spine tens b2', tens_b2)
                        contract_spine_list += [tens_b2]
                        expec_anchor = expec[b.get_anchor_ind()]
                        expec_anchor.modify(data=tens_b1.data, inds=tens_b1.inds)

                        spine_tens.reindex({anc_b: f'tmp_ancb{gk_idx}'}, inplace=True)
                        tens_b2.reindex({anc_b: f'tmp_ancb{gk_idx}'}, inplace=True)
                        tens_b2.reindex({'tmp': tt_ind}, inplace=True)
                        expec_anchor.reindex({'tmp': tt_ind}, inplace=True)
                        for tens in contract_spine_list:
                            tens.add_tag(spine.site_tag_id.format(gk_idx))
                            new_spine.add(tens)

                        new_branches[gk] = expec
                else:
                    output_is_gtn = True
                    if b is not None:
                        spine_tens = spine[gk_idx].copy()
                        if is_sqrt:
                            anc_b = next(iter(spine_tens.bonds(b.get_anchor_tens())))
                            # b_cc = b.conj(mangle_inner=True)
                            b_cc = b.copy()
                            b_cc.data.mangle_inner_()
                            b_cc.get_anchor_tens().reindex({anc_b: anc_b + '_'}, inplace=True)
                            b_cc = b_cc.apply_elemental_multiply_op(take_mps_cc=True)
                            # print('here?')
                            canon_opts = {'form': 'right'} if b.get_anchor_ind() == 0 else {'form': 'left'}
                            b2 = b.apply(b_cc, zipup=True, compress_opts=canon_opts)

                            btens = b2.get_anchor_tens()
                            # print('b2 anchor tens', btens)
                            lix = (anc_b, anc_b + '_')
                            btensL, btensR = qtn.tensor_split(btens, lix, absorb='left', bond_ind='tmp',
                                                              cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                            btens.modify(data=btensR.data, inds=btensR.inds)
                            # b2.get_anchor_tens().fuse({anc_b: (anc_b, anc_b + '_')}, inplace=True)
                            # new_branches[gk] = b2

                            spine_tens_cc = spine[gk_idx].conj()
                            spine_tens_cc.reindex({t_ind: t_ind + '_' for t_ind in spine_tens_cc.inds}, inplace=True)

                            # new_spine_tens = qtn.tensor_contract(spine_tens, spine_tens_cc, btensL)
                            # new_spine_tens.reindex({'tmp': anc_b}, inplace=True)
                            # btens.reindex({'tmp': anc_b}, inplace=True)
                            # new_spine.add( new_spine_tens )

                            contract_spine_list = [spine_tens, spine_tens_cc, btensL]
                            spine_tens.reindex({anc_b: f'tmp_ancb{gk_idx}'}, inplace=True)
                            btensL.reindex({anc_b: f'tmp_ancb{gk_idx}'}, inplace=True)
                            btensL.reindex({'tmp': tt_ind}, inplace=True)
                            btens.reindex({'tmp': tt_ind}, inplace=True)
                            for tens in contract_spine_list:
                                tens.add_tag(spine.site_tag_id.format(gk_idx))
                                new_spine.add(tens)

                            new_branches[gk] = b2
                        else:
                            new_branches[gk] = b.copy()
                            new_spine.add(spine_tens)

        # print('new branches', new_branches)
        # print('new spine', new_spine)

        # if len(new_branches) > 0:
        # print('integ axes', integ_axes, self.grid.axes)
        # if len(integ_axes) < self.grid.ndim:

        # if new_spine.num_tensors > 0:
        #     new_spine = new_spine.view_like(spine, inplace=True, L=new_spine.num_tensors)
        #     if new_spine.num_tensors == len(integ_axes):
        #         helper.zipup_fuse(new_spine, inplace=True)
        # else:
        #     new_spine = None

        if new_grid is None:
            new_grid = self.grid

        if output_is_gtn:  # len(integ_axes) < self.grid.ngrids:
            ## fixing spine to be compatible with new grid
            ## assumes that order of the grids in new_grid are the same as in self.grid
            block_contract_idxs = [[0]]
            for gk_idx in range(1, len(self.grid.grids)):
                gr = self.grid.grids[gk_idx]
                if gr in new_grid.grids:
                    assert (block_contract_idxs[-1][-1] + 1 == new_grid.get_grid_ind(gr)), \
                        'new grid not compatible with existing grid'
                    block_contract_idxs += [[gk_idx]]
                else:
                    block_contract_idxs[-1] += [gk_idx]

            print('block contract idx', block_contract_idxs)
            new_spine_exponent = new_spine.exponent

            # print('new spine', new_spine)

            for block_idx in range(len(block_contract_idxs)):
                contract_idxs = block_contract_idxs[block_idx]
                ## contract tensors of these indices together
                new_spine = new_spine.contract_tags(
                    [spine.site_tag_id.format(idx) for idx in contract_idxs],
                    which='any', inplace=True)

                if isinstance(new_spine, qtn.Tensor):
                    new_spine = qtn.TensorNetwork([new_spine])
                    new_spine.exponent = new_spine_exponent

                old_idx = contract_idxs[0]
                tens = new_spine.select_tensors(spine.site_tag_id.format(contract_idxs[-1]))[0]
                tens.drop_tags()
                tens.add_tag(spine.site_tag_id.format(block_idx))

                gr = self.grid.grids[old_idx]
                if gr in new_branches:
                    if old_idx != block_idx:
                        reindex_dict = {spine.site_ind_id.format(old_idx):
                                            spine.site_ind_id.format(block_idx)}
                        tens.reindex(reindex_dict, inplace=True)
                        new_branches[gr][0].reindex(reindex_dict, inplace=True)
                ## o.w. no branch exists so no reindexing necessary

            new_spine = new_spine.view_like(spine, inplace=True, L=len(block_contract_idxs))
            helper.zipup_fuse(new_spine, inplace=True)

            # print('new grid', new_grid)
            # print('new grid', self.grid.grids, new_grid.grids)
            # print('new spine', new_spine)
            # print('new branches', new_branches.items())

            gtn_mpo = self.__class__(new_grid, branches=new_branches, spine=new_spine,
                                     exponent=exponent, sign=sign,
                                     ax_deriv_configs=self.ax_deriv_configs)
            # gtn_mpo.exponent = exponent
            # gtn_mpo.sign = sign
            return gtn_mpo
        else:
            val = 10 ** exponent * sign
            if new_spine.num_tensors > 0:
                val *= new_spine.contract()  # .data.item()
            return val

    def ovlp(self, other: 'GridTN1DComb') -> Numeric:
        """ TODO: need to check this is correct more carefully
        """
        self.collect_exponents()
        other.collect_exponents()

        if self.data is not None and other.data is not None:
            out = None

            exponent = self.exponent + other.exponent
            sign = self.sign * other.sign
            scale = sign * 10 ** exponent

            rename = False
            if other.spine_ind_id == self.spine_ind_id:
                self.spine_ind_id = self.spine_ind_id + '_tmp'
                rename = True

            self.spine.mangle_inner_()

            for nb in range(self.grid.ngrids):

                branch1 = self.get_branch(nb)
                branch2 = other.get_branch(nb)

                if branch1 is None:
                    branch1 = self.grid.grids[nb].get_ones_mps()

                if branch2 is None:
                    branch2 = self.grid.grids[nb].get_ones_mps()

                # ovlp_b = self.get_branch(nb).ovlp(other.get_branch(nb))
                ovlp_b = branch1.ovlp(branch2)

                if isinstance(ovlp_b, NUM_TYPES):
                    scale *= ovlp_b
                else:
                    if out is None:
                        # print(self.spine[nb], other.spine[nb], ovlp_b, out)
                        out = qtn.tensor_contract(self.spine[nb], other.spine[nb], ovlp_b)
                    elif isinstance(out, qtn.Tensor):
                        # print(self.spine[nb], other.spine[nb], ovlp_b, out)
                        # print(type(out), type(self.spine[nb]), type(other.spine[nb]), type(ovlp_b))
                        out = qtn.tensor_contract(out, self.spine[nb], other.spine[nb], ovlp_b)
                    else:
                        raise TypeError

            out *= scale

            if rename:
                self.spine_ind_id = self.spine_ind_id[:-4]

            return out
        else:
            return 0.0

    def canonize_axes(self, axes: Sequence['Axis'], inplace=True, scale=True):
        canon_i = self.get_canon_site_from_axes(axes)
        return self.canonize(inplace=inplace, scale=scale, i=canon_i)

    def get_canon_site_from_axes(self, axes: Sequence['Axis']):
        ax_inds = [self.grid.axes.index(ax) for ax in axes]
        if ax_inds == np.range(0, len(axes)):
            canon_i = len(axes)
        elif ax_inds == np.range(self.grid.ndim - len(axes), self.grid.ndim):
            canon_i = self.grid.ndim - len(axes)
        else:
            raise ValueError('axes must either all be on left or right of MPS')
        return canon_i

    #######################
    ###  MPS_USVT fcts  ###
    #######################

    # def convert_to_USVT(self, canon_site, inplace=False, cur_orthog=None):
    #     return self.canonize(i=canon_site, inplace=inplace)
    #
    # def convert_from_USVT(self, inplace=False, canon_site=None):
    #     return self if inplace else self.copy()
    #
    # def get_S_tensor(self, canon_site=0):
    #     return self.spine[canon_site]

    def project_bond(self, obs_gtn: 'GridTN1DComb', bond_ind):

        is_sqrt = True

        new_spine = qtn.TensorNetwork([])
        exponent = self._exponent if not is_sqrt else self._exponent * 2
        sign = self._sign if not is_sqrt else 1.0

        if isinstance(obs_gtn, GridTN1DComb):
            obs_gtn.collect_exponents()
            obs_gtn.pad_self_to_new_grid(self.grid)
            obs_gtn_spine = obs_gtn.spine
            exponent += obs_gtn._exponent
        else:
            obs_gtn_spine = None

        spine = self.spine.copy()
        if obs_gtn_spine is not None:
            obs_gtn.spine_ind_id = obs_gtn.spine_ind_id + '_o_'

        for gk in self.grid.grids:

            gk_idx = self.grid.get_grid_ind(gk)
            tt_ind = self.spine_ind_id.format(gk_idx)
            b = self.get_branch(gk)

            try:
                obs_gtn_gk = obs_gtn[gk]
                if b is None:
                    b = gk.get_ones_mps()

                expec = b.meas_expec(obs_gtn_gk, is_sqrt=True,
                                     ancilla_reindex={tt_ind: tt_ind + '_'})

                if isinstance(expec, NUM_TYPES):
                    exponent += np.log10(np.abs(expec))
                    sign *= np.exp(-1.j * np.angle(expec))

                elif isinstance(expec, qtn.Tensor):
                    contract_spine_list = [expec, spine[gk_idx]]
                    if obs_gtn_spine is not None:
                        contract_spine_list += [obs_gtn_spine[gk_idx]]

                    if gk_idx != bond_ind:
                        spine_tens_cc = spine[gk_idx].conj()
                        spine_tens_cc.reindex({t_ind: t_ind + '_' for t_ind in spine_tens_cc.inds}, inplace=True)
                        contract_spine_list += [spine_tens_cc]

                    spine_tens = qtn.tensor_contract(*contract_spine_list)
                    # spine_tens.new_ind(spine.site_ind_id.format(new_spine.num_tensors+1), size=1, axis=-1)
                    new_spine.add(spine_tens)

                elif expec is None:
                    return 0.

                else:
                    raise NotImplementedError

            except KeyError:  ## no branch in obs_gtn

                if b is None:
                    expec = 1.0
                else:
                    expec = b.integrate(is_sqrt=is_sqrt, ancilla_reindex={tt_ind: tt_ind + '_'})

                if isinstance(expec, NUM_TYPES):
                    exponent += np.log10(np.abs(expec))
                    sign *= np.exp(-1.j * np.angle(expec))

                elif isinstance(expec, qtn.Tensor):
                    contract_spine_list = [expec, spine[gk_idx]]
                    if obs_gtn_spine is not None:
                        contract_spine_list += [obs_gtn_spine[gk_idx]]
                    if is_sqrt and gk_idx != bond_ind:
                        spine_tens_cc = spine[gk_idx].conj()
                        spine_tens_cc.reindex({t_ind: t_ind + '_' for t_ind in spine_tens_cc.inds}, inplace=True)
                        contract_spine_list += [spine_tens_cc]

                    spine_tens = qtn.tensor_contract(*contract_spine_list)
                    # spine_tens.new_ind(spine.site_ind_id.format(new_spine.num_tensors + 1), size=1, axis=-1)
                    new_spine.add(spine_tens)

                elif expec is None:
                    return 0.

                else:
                    raise NotImplementedError

        ### contract new spine missing cc at site specified by bond_ind
        if new_spine.num_tensors > 0:
            out: 'qtn.Tensor' = new_spine.contract_tags(all)
            out.modify(apply=lambda x: x * sign * 10 ** exponent)
            reindex_dict = {i + '_': i for i in spine[bond_ind].inds}
            out.reindex(reindex_dict, inplace=True)
        else:
            out: 'Numeric' = sign * 10 ** exponent

        return out

    ####################

    def meas_ovlp_branch(self, gk_idx: int, target_gtn: 'GridTN1DComb') -> Optional['qtn.Tensor']:
        """ measure <b|O|k> along one branch
        """
        if self.data is None:
            return None

        gk = self.grid.grids[gk_idx]
        tt_ind = self.spine_ind_id.format(gk_idx)

        b = self[gk]
        if b is None:
            b = gk.get_ones_mps()

        try:
            obs_gtn_spine = target_gtn.spine
        except AttributeError:
            obs_gtn_spine = None
        if obs_gtn_spine is not None and obs_gtn_spine.site_ind_id == self.spine.site_ind_id:
            target_gtn.spine_ind_id = obs_gtn_spine.site_ind_id + '_b_'

        try:
            target_gtn_gk = target_gtn[gk] if target_gtn is not None else None
        except KeyError:
            target_gtn_gk = gk.get_ones_mps()

        # if bra is None and is_sqrt:
        #     cc_branch = b.conj()
        #     for t in cc_branch.data.tensors:
        #         t.reindex({ind: ind + '_' for ind in t.inds}, inplace=True)
        #     cc_branch.data._site_ind_id = b.data.site_ind_id + '_'

        # print('MEAS EXPEC BRANCH', b.exponent, obs_gtn_gk.exponent)
        # print('branch norm', b.data.norm(), obs_gtn_gk.data.norm())
        # print('check orthog', helper.check_right_orthog(b.data, left_ancillas=(tt_ind,)))
        return b.ovlp(target_gtn_gk)

    def meas_expec_branch(self, gk_idx: int, obs_gtn, is_sqrt=False, bra=None, exclude_weights=False) \
            -> Optional['qtn.Tensor']:
        """ measure <b|O|k> along one branch
        """
        if self.data is None:
            return None

        gk = self.grid.grids[gk_idx]
        tt_ind = self.spine_ind_id.format(gk_idx)

        b = self[gk]
        if b is None:
            b = gk.get_ones_mps()

        try:
            obs_gtn_spine = obs_gtn.spine
        except AttributeError:
            obs_gtn_spine = None
        if obs_gtn_spine is not None and obs_gtn_spine.site_ind_id == self.spine.site_ind_id:
            obs_gtn.spine_ind_id = obs_gtn_spine.site_ind_id + '_op_'

        try:
            obs_gtn_gk = obs_gtn[gk] if obs_gtn is not None else None
        except KeyError:
            obs_gtn_gk = None

        # if bra is None and is_sqrt:
        #     cc_branch = b.conj()
        #     for t in cc_branch.data.tensors:
        #         t.reindex({ind: ind + '_' for ind in t.inds}, inplace=True)
        #     cc_branch.data._site_ind_id = b.data.site_ind_id + '_'

        # print('MEAS EXPEC BRANCH', b.exponent, obs_gtn_gk.exponent)
        # print('branch norm', b.data.norm(), obs_gtn_gk.data.norm())
        # print('check orthog', helper.check_right_orthog(b.data, left_ancillas=(tt_ind,)))
        expec = b.meas_expec(obs_gtn_gk, is_sqrt=is_sqrt, ancilla_reindex={tt_ind: tt_ind + '_'},
                             exclude_weights=exclude_weights)
        # print('expec norm', expec.norm(), expec)

        # bra = b.data.conj()
        # bra.reindex({tt_ind: tt_ind + '_'}, inplace=True)
        # tmp = helper.expectation_value(b.data, obs_gtn_gk.data, bra=bra)
        # print('tmp', tmp, tmp.norm())
        # if np.abs(tmp.norm() - expec.norm()) > 1.0e-12:
        #     print('diff tmp')
        #     exit()

        return expec

    def _contract_branch_expec_spine(self, grid_idx: int, obs_gtn,
                                     branch_expec: Union[Numeric, 'qtn.Tensor', 'qtn.TensorNetwork'],
                                     spine_expec: Optional[Union[Numeric, 'qtn.Tensor', 'qtn.TensorNetwork']],
                                     is_sqrt=False, bra_gtn=None) -> Optional['qtn.Tensor']:
        """ measure <b|O|k> along one branch
        """
        # print(grid_idx)
        # print('branch expec', branch_expec)
        # print('spine expec', spine_expec)

        spine = self.spine
        # print('check spine orthog', helper.check_orthog(spine))
        gk_idx = grid_idx

        if isinstance(obs_gtn, GridTN1DComb):
            # obs_gtn.collect_exponents()
            obs_gtn.pad_self_to_new_grid(self.grid)
            obs_gtn_spine = obs_gtn.spine
        else:
            obs_gtn_spine = None

        expec = branch_expec

        def _merge_spine_expec(expec_piece):

            if expec is None:  ## = 0
                return None

            elif isinstance(expec, NUM_TYPES):  # spine is None
                if isinstance(expec_piece, NUM_TYPES):
                    return expec * expec_piece
                elif isinstance(expec_piece, qtn.Tensor):
                    return expec_piece.modify(apply=lambda x: x * expec)
                elif isinstance(expec_piece, qtn.TensorNetwork):
                    return expec_piece.multiply(expec)
                elif expec_piece is None:
                    return expec
                else:
                    raise NotImplementedError

            elif isinstance(expec, (qtn.Tensor, qtn.TensorNetwork)):
                if isinstance(expec_piece, NUM_TYPES):
                    return expec.multiply(expec_piece)
                elif isinstance(expec_piece, (qtn.TensorNetwork, qtn.Tensor)):
                    return qtn.TensorNetwork([expec_piece, expec])
                elif expec_piece is None:
                    return expec
                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

        # print('spine expec', spine_expec)
        expec = _merge_spine_expec(spine_expec)
        # print('expec', expec)
        # print('merged expec', expec.norm())

        if expec is None:
            return None

        elif isinstance(expec, NUM_TYPES):
            contract_spine_list = qtn.TensorNetwork([spine[gk_idx]])
            if obs_gtn_spine is not None:
                contract_spine_list.add(obs_gtn_spine[gk_idx])
            if bra_gtn is not None:
                contract_spine_list.add(bra_gtn.spine[gk_idx])
            else:
                if is_sqrt:
                    spine_tens_cc = spine[gk_idx].conj()
                    spine_tens_cc.reindex({t_ind: t_ind + '_' for t_ind in spine_tens_cc.inds}, inplace=True)
                    contract_spine_list.add(spine_tens_cc)
            # print('contract spine list', contract_spine_list)
            expec_tot = contract_spine_list.contract()
            expec_tot.modify(apply=lambda x: expec * x)
            return expec_tot

        elif isinstance(expec, (qtn.Tensor, qtn.TensorNetwork)):
            # print('spine tens norm', spine[gk_idx].norm())
            contract_spine_list = qtn.TensorNetwork([expec, spine[gk_idx]])
            if obs_gtn_spine is not None:
                contract_spine_list.add(obs_gtn_spine[gk_idx])
            if bra_gtn is not None:
                contract_spine_list.add(bra_gtn.spine[gk_idx])
            else:
                if is_sqrt:
                    spine_tens_cc = spine[gk_idx].conj()
                    spine_tens_cc.reindex({t_ind: t_ind + '_' for t_ind in spine_tens_cc.inds}, inplace=True)
                    contract_spine_list.add(spine_tens_cc)
            # print('contract spine list', contract_spine_list)
            # print(contract_spine_list.norm())
            return contract_spine_list.contract()

        else:
            raise NotImplementedError

    def project_op_site(self, obs_gtn, site_ind, nsites=1, left_env=None, right_env=None, branch_env=None) \
            -> tuple['qtn.Tensor', Sequence[str], Sequence[str]]:
        """ assumes site is along the spine
            assumes appropriate canonical form (orthog center on spine at site_ind) without scaling
                actually scaling should be fine as long as it's put into the exponent?
            left_env:  env from spine + other branches
            right_env: env from branch at site_ind
        """
        if isinstance(obs_gtn, GridTN1DComb):
            # obs_gtn.collect_exponents()
            obs_gtn.pad_self_to_new_grid(self.grid)
            obs_gtn_spine = obs_gtn.spine
            exponent = obs_gtn.exponent
            # print('project op site', exponent)
        else:
            obs_gtn_spine = None
            exponent = 0.0

        expec_tn = qtn.TensorNetwork([])

        if left_env is None and right_env is None:
            # print('computing spine env')
            exclude_axes = []
            for ind in range(site_ind, site_ind + nsites):
                exclude_axes += list(self.grid.grids[ind].axes)
            spine_env = self.meas_expec(obs_gtn, exclude_axes=exclude_axes, is_sqrt=True)
            if spine_env is not None:
                spine_env.modify(apply=lambda x: x * 10 ** (-(self.exponent * 2 + obs_gtn.exponent)))
            # print('calc spine env', spine_env)
            if spine_env is not None:
                expec_tn.add(spine_env)
        else:
            if left_env is not None:
                expec_tn.add(left_env)
            if right_env is not None:
                expec_tn.add(right_env)

        ## branch envs
        if branch_env is None:
            # print('computing branch env')
            for ind in range(site_ind, site_ind + nsites):
                # print('self.exponent', self[ind].exponent, self.exponent, obs_gtn.exponent)
                branch_env = self.meas_expec_branch(ind, obs_gtn, is_sqrt=True)
                ### this does not include global any exponents.
                # branch_env.modify(apply = lambda x: x * 10**(-(self.exponent * 2 + obs_gtn.exponent)))
                # gk = self.grid.grids[ind]
                # branch = self[gk]
                # branch_cc = branch.conj()
                # branch_cc.data.mangle_inner_(append='_')
                # btens0 = branch_cc.get_anchor_tens()
                # if btens0 == branch_cc.data[1].ndim:   ## ancilla bond
                #     for ind in btens0.inds:
                #         if ind != branch_cc.data.site_ind(0) and ind not in branch_cc.data[1]:
                #             b_anc = ind
                #             break
                #     btens0.reindex({b_anc: b_anc + '_'}, inplace=True)
                #
                # branch_env = qtn.TensorNetwork([branch.data, branch_cc.data, obs_gtn[gk].data])
                # branch_env.exponent = 0.0
                expec_tn.add(branch_env)
        else:
            expec_tn.add(branch_env)

        if obs_gtn_spine is not None:
            for ind in range(site_ind, site_ind + nsites):
                expec_tn.add(obs_gtn_spine[ind])

        # print('PROJECT SITE', expec_tn)

        # print('expec tn', expec_tn.shape)
        # out: 'qtn.Tensor' = expec_tn.contract_tags(all)
        # # print('proj op out', out.norm(), exponent)
        # out.modify(apply=lambda x: x * 10 ** exponent)
        # # print('proj op out', out.norm(), exponent)

        out = expec_tn
        ### !!! CHANGED THIS:  = -> +=
        out.exponent += exponent
        ### TO CHECK:  perhaps ok because exponent = 0?  i think it's ok now?

        ### projected operator indices
        self_ind_L, conj_ind_L = None, None
        if site_ind > 0:
            self_ind_L = self.spine.bond(site_ind, site_ind - 1)
            conj_ind_L = self_ind_L + '_'
            # reindex_dict[conj_ind_L] = self_ind_L
        self_ind_R, conj_ind_R = None, None
        if site_ind < self.spine.L - nsites:
            self_ind_R = self.spine.bond(site_ind + nsites - 1, site_ind + nsites)
            conj_ind_R = self_ind_R + '_'
            # reindex_dict[conj_ind_R] = self_ind_R

        bonds_i = [self.spine_ind_id.format(site_ind + i) for i in range(nsites)]
        bonds_o = [self.spine_ind_id.format(site_ind + i) + '_' for i in range(nsites)]

        if self_ind_L is not None:
            bonds_i += [self_ind_L]
        if self_ind_R is not None:
            bonds_i += [self_ind_R]

        if conj_ind_L is not None:
            bonds_o += [conj_ind_L]
        if conj_ind_R is not None:
            bonds_o += [conj_ind_R]

        return out, bonds_i, bonds_o

    def project_op_bond(self, obs_gtn, bond_ind, left_env=None, right_env=None, branch_env=None, direction=1,
                        **kwargs) -> tuple['qtn.Tensor', Sequence[str], Sequence[str]]:
        """ project obs_gtn (mpo); spine is of the form LLL-(S)-RRR
        """
        # print('project op bond')
        # s_tens = self.spine.select_tensors(self.spine.s_tag)[0]

        if isinstance(obs_gtn, GridTN1DComb):
            # obs_gtn.collect_exponents()
            obs_gtn.pad_self_to_new_grid(self.grid)
            obs_gtn_spine = obs_gtn.spine
            exponent = obs_gtn._exponent
        else:
            obs_gtn_spine = None
            exponent = 0.0

        spine_tens = self.spine[bond_ind]
        spine_tens_cc = spine_tens.conj()
        spine_tens_cc.reindex({ind: ind + '_' for ind in spine_tens_cc.inds}, inplace=True)

        expec_tn = qtn.TensorNetwork([spine_tens, spine_tens_cc])

        if left_env is None and right_env is None:
            # print('computing spine env')
            exclude_axes = list(self.grid.grids[bond_ind].axes)
            spine_env = self.meas_expec(obs_gtn, exclude_axes=exclude_axes, is_sqrt=True, exclude_weights=True)
            if spine_env is not None:
                spine_env.modify(apply=lambda x: x * 10 ** (-(self.exponent * 2 + obs_gtn.exponent)))
            # print('calc spine env', spine_env)
            if spine_env is not None:
                expec_tn.add(spine_env)
        else:
            if left_env is not None:
                expec_tn.add(left_env)
            if right_env is not None:
                expec_tn.add(right_env)

        ## branch envs
        if branch_env is None:
            # print('computing branch env')
            branch_env = self.meas_expec_branch(bond_ind, obs_gtn, is_sqrt=True, exclude_weights=True)

        expec_tn.add(branch_env)

        if obs_gtn_spine is not None:
            expec_tn.add(obs_gtn_spine[bond_ind])

        # # print('expec tn', expec_tn)
        # out: 'qtn.Tensor' = expec_tn.contract_tags(all)
        # # print('proj op site out', out.norm(), exponent)
        # out.modify(apply=lambda x: x * 10 ** exponent)
        # # print('proj op site out', out.norm(), exponent)

        out = expec_tn
        ### !!! CHANGED = -> +=
        out.exponent += exponent

        ### projected operator indices
        s_tens = self.spine.select_tensors(self.spine.s_tag)[0]
        if direction > 0:
            self_ind_L = next(iter(qtn.bonds(self.spine[bond_ind], s_tens)))
            self_ind_R = next(iter(qtn.bonds(self.spine[bond_ind + 1], s_tens)))
        else:
            self_ind_L = next(iter(qtn.bonds(self.spine[bond_ind - 1], s_tens)))
            self_ind_R = next(iter(qtn.bonds(self.spine[bond_ind], s_tens)))

        bonds_i, bonds_o = [], []
        if self_ind_L is not None:
            bonds_i += [self_ind_L]
            bonds_o += [self_ind_L + '_']
        if self_ind_R is not None:
            bonds_i += [self_ind_R]
            bonds_o += [self_ind_R + '_']

        return out, bonds_i, bonds_o

    def project_vec_site(self, target_gtn, site_ind, nsites=1, left_env=None, right_env=None, branch_env=None) \
            -> tuple['qtn.Tensor', Sequence[str]]:
        """ assumes site is along the spine
            assumes appropriate canonical form (orthog center on spine at site_ind) without scaling
                actually scaling should be fine as long as it's put into the exponent?
            left_env:  env from spine + other branches
            right_env: env from branch at site_ind
        """
        if isinstance(target_gtn, GridTN1DComb):
            target_gtn.pad_self_to_new_grid(self.grid)
            target_gtn_spine = target_gtn.spine
            exponent = target_gtn.exponent
            # print('project op site', exponent)
        else:
            target_gtn_spine = None
            exponent = 0.0

        expec_tn = qtn.TensorNetwork([])

        if left_env is None and right_env is None:
            # print('computing spine env')
            exclude_axes = []
            for ind in range(site_ind, site_ind + nsites):
                exclude_axes += list(self.grid.grids[ind].axes)
            spine_env = self.ovlp(target_gtn, exclude_axes=exclude_axes)
            if spine_env is not None:
                spine_env.modify(apply=lambda x: x * 10 ** (-(self.exponent * 2 + target_gtn.exponent)))
            # print('calc spine env', spine_env)
            if spine_env is not None:
                expec_tn.add(spine_env)
        else:
            if left_env is not None:
                expec_tn.add(left_env)
            if right_env is not None:
                expec_tn.add(right_env)

        ## branch envs
        if branch_env is None:
            # print('computing branch env')
            for ind in range(site_ind, site_ind + nsites):
                # print('self.exponent', self[ind].exponent, self.exponent, target_gtn.exponent)
                branch_env = self.meas_ovlp_branch(ind, target_gtn)
                expec_tn.add(branch_env)
        else:
            expec_tn.add(branch_env)

        if target_gtn_spine is not None:
            for ind in range(site_ind, site_ind + nsites):
                expec_tn.add(target_gtn_spine[ind])

        # print('PROJECT SITE', expec_tn)

        # out: 'qtn.Tensor' = expec_tn.contract_tags(all)
        # print('proj op out', out.norm(), exponent)
        # out.modify(apply=lambda x: x * 10 ** exponent)
        # print('proj op out', out.norm(), exponent)

        out = expec_tn
        ### !!! CHANGED = -> +=
        out.exponent += exponent

        ### projected operator indices
        self_ind_L, conj_ind_L = None, None
        if site_ind > 0:
            self_ind_L = self.spine.bond(site_ind, site_ind - 1)
            conj_ind_L = self_ind_L + '_'
            # reindex_dict[conj_ind_L] = self_ind_L
        self_ind_R, conj_ind_R = None, None
        if site_ind < self.spine.L - nsites:
            self_ind_R = self.spine.bond(site_ind + nsites - 1, site_ind + nsites)
            conj_ind_R = self_ind_R + '_'
            # reindex_dict[conj_ind_R] = self_ind_R

        # bonds_i = [self.spine_ind_id.format(site_ind + i) for i in range(nsites)]
        bonds_o = [self.spine_ind_id.format(site_ind + i) + '_' for i in range(nsites)]

        # if self_ind_L is not None:
        #     bonds_i += [self_ind_L]
        # if self_ind_R is not None:
        #     bonds_i += [self_ind_R]

        if conj_ind_L is not None:
            bonds_o += [conj_ind_L]
        if conj_ind_R is not None:
            bonds_o += [conj_ind_R]

        return out, bonds_o

    def project_vec_bond(self, target_gtn, bond_ind, left_env=None, right_env=None, branch_env=None,
                         **kwargs) -> tuple['qtn.Tensor', Sequence[str]]:
        """ project obs_gtn (mpo); spine is of the form LLL-(S)-RRR
        """
        print('project vec bond')
        # s_tens = self.spine.select_tensors(self.spine.s_tag)[0]

        if isinstance(target_gtn, GridTN1DComb):
            # obs_gtn.collect_exponents()
            target_gtn.pad_self_to_new_grid(self.grid)
            obs_gtn_spine = target_gtn.spine
            exponent = target_gtn._exponent
        else:
            obs_gtn_spine = None
            exponent = 0.0

        spine_tens = self.spine[bond_ind]
        spine_tens_cc = spine_tens.conj()
        spine_tens_cc.reindex({ind: ind + '_' for ind in spine_tens_cc.inds}, inplace=True)

        expec_tn = qtn.TensorNetwork([spine_tens, spine_tens_cc])

        if left_env is None and right_env is None:
            print('computing spine env')
            exclude_axes = list(self.grid.grids[bond_ind].axes)
            spine_env = self.ovlp(target_gtn, exclude_axes=exclude_axes)
            if spine_env is not None:
                spine_env.modify(apply=lambda x: x * 10 ** (-(self.exponent * 2 + target_gtn.exponent)))
            # print('calc spine env', spine_env)
            if spine_env is not None:
                expec_tn.add(spine_env)
        else:
            if left_env is not None:
                expec_tn.add(left_env)
            if right_env is not None:
                expec_tn.add(right_env)

        ## branch envs
        if branch_env is None:
            branch_env = self.meas_ovlp_branch(bond_ind, target_gtn)

        expec_tn.add(branch_env)

        if obs_gtn_spine is not None:
            expec_tn.add(obs_gtn_spine[bond_ind])

        # print('expec tn', expec_tn)
        # out: 'qtn.Tensor' = expec_tn.contract_tags(all)
        # # print('proj op site out', out.norm(), exponent)
        # out.modify(apply=lambda x: x * 10 ** exponent)
        # # print('proj op site out', out.norm(), exponent)

        out = expec_tn
        ### !!! CHANGED = -> +=
        out.exponent += exponent

        ### projected operator indices
        s_tens = self.spine.select_tensors(self.spine.s_tag)[0]
        self_ind_L = next(iter(qtn.bonds(self.spine[bond_ind], s_tens)))
        self_ind_R = next(iter(qtn.bonds(self.spine[bond_ind + 1], s_tens)))

        bonds_o = []
        if self_ind_L is not None:
            bonds_o += [self_ind_L + '_']
        if self_ind_R is not None:
            bonds_o += [self_ind_R + '_']

        return out, bonds_o

    def _attach_spine_tens_to_branch(self, gk_idx, is_mps=True) -> 'qtn.TensorNetwork1D':
        """ addiiton of spine tensor to branch
            used with TDVP/TDDMRG
        """
        branch_ = self.get_branch(gk_idx)
        spine_ = self.spine

        if self.spine.exponent != 0.0 or (branch_ is not None and branch_.exponent != 0.0):
            raise RuntimeError('attach spine tens to branch, exponents are not collected')
        # exit()

        if branch_ is None:
            branch_ = self.grid.grids[gk_idx].get_ones_mps() if is_mps else self.grid.grids[gk_idx].get_iden_mpo()
            self[gk_idx] = branch_
            # b0 = branch_.get_anchor_tens()
            # b0.new_bond(spine_[gk_idx], name=self.spine_ind_id.format(gk_idx))
            branch_ = self[gk_idx].data.copy()
        else:
            branch_ = branch_.data.copy()

        # print('attach spine to branch exp', branch_.exponent, spine_.exponent)
        if is_mps:
            helper.renumber_mps(branch_, list(range(branch_.L)), list(range(1, branch_.L + 1)), inplace=True)
        else:
            helper.renumber_mpo(branch_, list(range(branch_.L)), list(range(1, branch_.L + 1)), inplace=True)
        spine_tens = spine_[gk_idx].copy()
        spine_tens.add_tag(branch_.site_tag_id.format(0))

        branch_.add(spine_tens)
        branch_._L = branch_.L + 1
        return branch_

    def _update_spine_tens_and_branch(self, gk_idx, new_mpx, is_canon=False):
        """ inplace update
            used with TDVP/TDDMRG
        """
        branch = self[gk_idx]

        if self.spine.exponent != 0.0 or (branch is not None and branch.exponent != 0.0):
            raise RuntimeError('attach spine tens to branch, exponents are not collected')

        if branch is None:
            branch = new_mpx[1:]
            if self.data_type is DataType.MPS:
                helper.renumber_mps(branch, list(range(1, new_mpx.L)), list(range(new_mpx.L - 1)), inplace=True)
            else:
                helper.renumber_mpo(branch, list(range(1, new_mpx.L)), list(range(new_mpx.L - 1)), inplace=True)
        else:
            for i in range(1, new_mpx.L):
                # print('branch data exponents', branch.data.exponent, dist_mpx.exponent)
                b_tens = branch.data[i - 1]
                new_b_tens = new_mpx[i]
                try:
                    if self.data_type is DataType.MPO:
                        new_b_tens.reindex({branch.data.upper_ind_id.format(i): branch.data.upper_ind_id.format(i - 1)},
                                           inplace=True)
                    new_b_tens.transpose_like(b_tens, inplace=True)
                except ValueError:
                    # print('i', i, new_b_tens.ndim, branch.L, new_mpx.L)
                    # print('new mpx', new_mpx[i - 1:i + 1])
                    # print('new mpx', new_mpx[i:i + 2])
                    indL, indR = new_mpx.bond(i, i - 1), new_mpx.bond(i, i + 1)
                    # print('branch', branch.data[i-2], branch.data[i-1], branch.data[i])
                    if i == 1:
                        indL_, = branch.data[i - 1].bonds(self.spine[gk_idx])
                    else:
                        indL_ = branch.data.bond(i - 2, i - 1)
                    indR_ = branch.data.bond(i, i - 1) if indR is not None else None
                    new_b_tens.reindex({indL: indL_, indR: indR_}, inplace=True)
                    # print('b_tens', b_tens, new_b_tens, indL, indL_, indR, indR_)
                    if self.data_type is DataType.MPO:
                        new_b_tens.reindex({branch.data.upper_ind_id.format(i): branch.data.upper_ind_id.format(i - 1)},
                                           inplace=True)
                        # print('new b tens', new_b_tens, b_tens)
                    new_b_tens.transpose_like(b_tens, inplace=True)
                b_tens.modify(data=new_b_tens.data)

            if (np.abs(branch.exponent) > 1.0e-12) or (np.abs(self.spine.exponent)) > 1.0e-12 or \
                    (np.abs(new_mpx.exponent) > 1.0e-12):
                print('exponents branch + spine', new_mpx.exponent, branch.exponent, self.spine.exponent)
                print('some part of gtnc has a local exponent')
                if np.abs(new_mpx.exponent - self.exponent) > 1.0e-12:
                    print('exponents', new_mpx.exponent, self.exponent)
                    raise ValueError('updating spine/branch must be done with mpx with same exponent as before')
            if np.abs(branch.exponent) > 1.0e-12:
                raise ValueError('branch to be updated should have no exponent')
            if np.abs(self.spine.exponent) > 1.0e-12:
                raise ValueError('spine to be updated should have no exponent')

                # branch.data.exponent = new_mpx.exponent

        if not is_canon:
            branch.canon_site = -1
        else:
            branch.canon_site = branch.get_anchor_ind()

        new_spine_tens = new_mpx[0]
        new_spine_tens.drop_tags((new_mpx.site_tag_id.format(0),))
        new_spine_tens.transpose_like_(self.spine[gk_idx])
        self.spine[gk_idx].modify(data=new_spine_tens.data)
        return self

    def _attach_spine_tens_to_branch_v2(self, gk_idx, is_mps=True) -> 'qtn.TensorNetwork1D':
        """ addiiton of spine tensor to branch
            used with DMRG
        """
        # self.distribute_sign()
        branch_ = self.get_branch(gk_idx)
        anchor_idx = 0 if branch_ is None else branch_.get_anchor_ind()

        if self.spine.exponent != 0.0 or (branch_ is not None and branch_.exponent != 0.0):
            raise RuntimeError('attach spine tens to branch, exponents are not collected')

        if branch_ is None:
            # print('branch is None')
            branch_ = self.grid.grids[gk_idx].get_ones_mps() if is_mps else self.grid.grids[gk_idx].get_iden_mpo()
            self[gk_idx] = branch_
            # b0 = branch_.get_anchor_tens()
            # b0.new_bond(spine_[gk_idx], name=self.spine_ind_id.format(gk_idx))
            branch_ = self[gk_idx].data.copy()
        else:
            branch_ = branch_.data.copy()

        new_b0 = branch_[anchor_idx].contract(self.spine[gk_idx])
        branch_[anchor_idx].modify(data=new_b0.data, inds=new_b0.inds)

        return branch_

    def _update_spine_tens_and_branch_v2(self, gk_idx, new_mpx, is_canon=False):
        """ inplace update
            used with DMRG
            if is_canon left, towards spine:  pushes canon site into spine
            if is_canon right, towards end: spliting of first tensor shouldn't really change canonicalization??
        """
        # is_canon = False
        if is_canon:
            raise Warning('check canonicalization of state after splitting tensor')

        branch = self[gk_idx]
        anchor_idx = 0 if branch is None else branch.get_anchor_ind()  # prob should fix at some point
        new_mpx = new_mpx.copy()

        if self.spine.exponent != 0.0 or (branch is not None and branch.exponent != 0.0):
            raise RuntimeError('attach spine tens to branch, exponents are not collected')

        if isinstance(new_mpx, qtn.MatrixProductState):
            left_inds = (new_mpx.site_ind(0), new_mpx.bond(0, 1))
        elif isinstance(new_mpx, qtn.MatrixProductOperator):
            left_inds = (new_mpx.upper_ind(0), new_mpx.lower_ind(0), new_mpx.bond(0, 1))
        else:
            raise TypeError
        new_branch_tens, new_spine_tens = qtn.tensor_split(new_mpx[0], left_inds,
                                                           bond_ind=self.spine.site_ind_id.format(gk_idx),
                                                           absorb='right', cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)

        if branch is None:
            branch = new_mpx
            self.spine[gk_idx].modify(data=new_spine_tens.data, inds=new_spine_tens.inds)
        else:
            new_spine_tens.transpose_like(self.spine[gk_idx], inplace=True)
            self.spine[gk_idx].modify(data=new_spine_tens.data)

            for i in range(new_mpx.L):
                # print('branch data exponents', branch.data.exponent, dist_mpx.exponent)
                b_tens = branch.data[i]
                new_b_tens = new_branch_tens if i == anchor_idx else new_mpx[i].copy()
                try:
                    new_b_tens.transpose_like(b_tens, inplace=True)
                except ValueError:
                    indL, indR = new_mpx.bond(i, i - 1), new_mpx.bond(i, i + 1)
                    # print('branch', branch.data[i-2], branch.data[i-1], branch.data[i])
                    if i == 0:
                        indL_, = branch.data[i].bonds(self.spine[gk_idx])
                    else:
                        indL_ = branch.data.bond(i - 1, i)
                    indR_ = branch.data.bond(i, i + 1) if indR is not None else None
                    new_b_tens.reindex({indL: indL_, indR: indR_}, inplace=True)
                    # print('b_tens', b_tens, new_b_tens, indL, indL_, indR, indR_)
                    new_b_tens.transpose_like(b_tens, inplace=True)
                b_tens.modify(data=new_b_tens.data)

            if (np.abs(branch.exponent) > 1.0e-12) or (np.abs(self.spine.exponent)) > 1.0e-12 or \
                    (np.abs(new_mpx.exponent) > 1.0e-12):
                print('exponents branch + spine', new_mpx.exponent, branch.exponent, self.spine.exponent)
                print('some part of gtnc has a local exponent')
                if np.abs(new_mpx.exponent - self.exponent) > 1.0e-12:
                    print('exponents', new_mpx.exponent, self.exponent)
                    raise ValueError('updating spine/branch must be done with mpx with same exponent as before')
            if np.abs(branch.exponent) > 1.0e-12:
                raise ValueError('branch to be updated should have no exponent')
            if np.abs(self.spine.exponent) > 1.0e-12:
                raise ValueError('spine to be updated should have no exponent')

                # branch.data.exponent = new_mpx.exponent

        if not is_canon:
            branch.canon_site = -1
        else:
            branch.canon_site = branch.get_anchor_ind()

        return self

    def evolve_tdvp(self, dt, mpo_list: Sequence['GridTN1DComb'], te_order=0, do_adapt=False, inplace=False,
                    compress_config: 'CompressionConfiguration' = None, expand_basis=None):
        """ this ordering gets good results for Buneman? JK it doesn't actually?
        """
        verbose=False

        gtn = self if inplace else self.copy()
        # gtn.collect_exponents()
        # print('gtn exponent', gtn.exponent)
        # gtn.distribute_exponents()
        # gtn_copy = gtn.copy()

        mpo_list = [mpo.copy() for mpo in mpo_list]
        for mpo in mpo_list:
            # mpo_copy = mpo.copy()
            mpo.collect_exponents()
            # mpo.distribute_exponents()
            mpo.distribute_sign()
            # print('mpo sign', mpo.sign)
            # print('mpo diff', mpo_copy.exponent, mpo.exponent, mpo_copy.sign, mpo.sign, mpo_copy.distance(mpo))
        # exit()
        # num_mpos = len(mpo_list)
        if len(mpo_list) == 0:
            return gtn

        gtn.canonicalize_around_i(0, scale=True, redo_canon=True)
        gtn.collect_exponents()  # need if scale is True
        gtn.distribute_sign()

        ## build branch envs
        mpo_envs = []
        spine_adapt = False
        for mpo in mpo_list:

            if mpo.spine.site_ind_id == gtn.spine.site_ind_id:
                mpo.spine_ind_id = mpo.spine_ind_id + '_op_'

            left_envs, right_envs, branch_envs = {}, {}, {}
            for gk_idx in range(gtn.grid.ngrids - 1, 0, -1):
                # print('branch exponents', gtn[gk_idx].exponent, mpo[gk_idx].exponent)
                ### hacky solution to <x|O|x>
                expec = gtn.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=True)
                ### includes branch, mpo branch exponents. ideally, shouldn't
                # expec.modify(apply = lambda x: x * 10**(-(mpo[gk_idx].exponent + 2*gtn[gk_idx].exponent)))
                expec.drop_tags()
                branch_envs[gk_idx] = expec
                right_envs[gk_idx - 1] = gtn._contract_branch_expec_spine(gk_idx, mpo, expec,
                                                                          right_envs.get(gk_idx, None),
                                                                          is_sqrt=True)

            mpo_envs += [(left_envs, right_envs, branch_envs)]

        ## left to right sweep
        for gk_idx in range(gtn.grid.ngrids):

            # print('CHECK ORTHOG AROUND', gk_idx)
            # print('spine', helper.check_orthog(gtn.spine))
            # for i in range(gtn.grid.ngrids):
            #     print('branch', i, helper.check_right_orthog(gtn[i].data,
            #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))
            # gtn.check_orthog()

            branch_dir = 1 if gk_idx != gtn.grid.ngrids - 1 else -1  # 1: r2l (edge to anchor), -1: l2r
            gtn._branch_local_TE(dt / 2, gk_idx, mpo_list, mpo_envs, sweep_direction=branch_dir, te_order=te_order,
                                 do_adapt=do_adapt, inplace=True, compress_config=compress_config,
                                 solver_type=LocalSolverType.TDVP)

            #### update spine
            if gk_idx < gtn.grid.ngrids - 1:
                ## if already updated
                # print('gk_idx', gk_idx, gtn.grid.ngrids - 1, spine_adapt)
                # if gk_idx == gtn.grid.ngrids - 1:  # and spine_adapt:
                #     continue

                # print('UPDATE SPINE', gk_idx)
                spine_bond_size = gtn.spine.bond_size(gk_idx, gk_idx + 1)
                max_bond = np.inf if compress_config is None else compress_config.max_bonds[1]
                # if max_bond is not None:
                #     max_bond = min(max_bond, 2 ** (gk_idx + 1), 2 ** (self.grid.ndims - gk_idx - 1))
                if verbose:  print('spine bond size', spine_bond_size, gk_idx, gk_idx + 1)
                spine_adapt = do_adapt and spine_bond_size < max_bond
                print('spine adapt', spine_adapt)

                nsites = 2 if spine_adapt else 1
                if verbose:  print('spine tdvp (LR)', gk_idx, nsites)
                gtn._spine_tdvp(dt / 2, gk_idx, nsites, mpo_list, mpo_envs, sweep_direction=1,
                                compress_config=compress_config)
                ## now canonizcalied to gk_idx + 1

        ## right to left sweep
        for gk_idx in range(gtn.grid.ngrids - 1, -1, -1):

            # print('CHECK ORTHOG AROUND', gk_idx)
            # print('spine', helper.check_orthog(gtn.spine))
            # for i in range(gtn.grid.ngrids):
            #     print('branch', i, helper.check_right_orthog(gtn[i].data,
            #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))

            ## build left to right
            branch_dir = -1 if gk_idx != gtn.grid.ngrids - 1 else 1  ## 1: r2l, -1: l2r
            gtn._branch_local_TE(dt / 2, gk_idx, mpo_list, mpo_envs, sweep_direction=branch_dir, te_order=te_order,
                                 do_adapt=do_adapt, inplace=True, compress_config=compress_config,
                                 solver_type=LocalSolverType.TDVP)

            #### update spine
            if gk_idx > 0:

                ## if already updated
                # print('gk_idx', gk_idx, gtn.grid.ngrids - 1, spine_adapt)
                if gk_idx == 0:  # and spine_adapt:
                    continue

                if verbose:  print('UPDATE SPINE (RL)', gk_idx)
                spine_bond_size = gtn.spine.bond_size(gk_idx, gk_idx - 1)
                if verbose:  print('spine bond size', spine_bond_size, gk_idx, gk_idx - 1)
                max_bond = np.inf if compress_config is None else compress_config.max_bonds[1]
                spine_adapt = do_adapt and spine_bond_size < max_bond
                print('spine adapt', spine_adapt)

                nsites = 2 if spine_adapt else 1
                left_gk_idx = gk_idx - nsites + 1
                gtn._spine_tdvp(dt / 2, left_gk_idx, nsites, mpo_list, mpo_envs, sweep_direction=-1,
                                compress_config=compress_config)

        return gtn

    def evolve_tdvp_v2(self, dt, mpo_list: Sequence['GridTN1DComb'], te_order=0, do_adapt=True, inplace=False,
                       compress_config: 'CompressionConfiguration' = None, expand_basis=None, sweep_direction=1):
        """ this ordering gets "stuck" results for Buneman
        """
        gtn = self if inplace else self.copy()
        gtn.collect_exponents()
        # gtn_copy = gtn.copy()

        mpo_list = [mpo.copy() for mpo in mpo_list]
        for mpo in mpo_list:
            mpo.collect_exponents()
            mpo.distribute_sign()
        # num_mpos = len(mpo_list)

        # sweep_direction = np.random.randint(0, 2)
        # sweep_direction = -1 if sweep_direction == 0 else 1
        # # sweep_direction = -1
        # print('evolve tdvp sweep direction', sweep_direction)

        if sweep_direction > 0:
            gtn.canonicalize_around_i(0, scale=False, redo_canon=True)
        else:
            gtn.canonicalize_around_i(gtn.grid.ngrids - 1, scale=False, redo_canon=True)

        ## build branch envs
        mpo_envs = []
        for mpo in mpo_list:

            if mpo.spine.site_ind_id == gtn.spine.site_ind_id:
                mpo.spine_ind_id = mpo.spine_ind_id + '_op_'

            left_envs, right_envs, branch_envs = {}, {}, {}
            if sweep_direction > 0:
                for gk_idx in range(gtn.grid.ngrids - 1, 0, -1):
                    # print('branch exponents', gtn[gk_idx].exponent, mpo[gk_idx].exponent)
                    ### hacky solution to <x|O|x>
                    expec = gtn.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=True)
                    ### includes branch, mpo branch exponents. ideally, shouldn't
                    # expec.modify(apply = lambda x: x * 10**(-(mpo[gk_idx].exponent + 2*gtn[gk_idx].exponent)))
                    expec.drop_tags()
                    branch_envs[gk_idx] = expec
                    right_envs[gk_idx - 1] = gtn._contract_branch_expec_spine(gk_idx, mpo, expec,
                                                                              right_envs.get(gk_idx, None),
                                                                              is_sqrt=True)
            else:
                for gk_idx in range(0, gtn.grid.ngrids - 1):
                    # print('branch exponents', gtn[gk_idx].exponent, mpo[gk_idx].exponent)
                    ### hacky solution to <x|O|x>
                    expec = gtn.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=True)
                    ### includes branch, mpo branch exponents. ideally, shouldn't
                    # expec.modify(apply = lambda x: x * 10**(-(mpo[gk_idx].exponent + 2*gtn[gk_idx].exponent)))
                    expec.drop_tags()
                    branch_envs[gk_idx] = expec
                    left_envs[gk_idx + 1] = gtn._contract_branch_expec_spine(gk_idx, mpo, expec,
                                                                             left_envs.get(gk_idx, None),
                                                                             is_sqrt=True)

            mpo_envs += [(left_envs, right_envs, branch_envs)]

        # dt = dt / 2
        for sweep_direction in [1]:  # , -1]:
            print('comb sweep', sweep_direction)

            ## left to right sweep or right to left sweep
            gk_inds = range(gtn.grid.ngrids) if sweep_direction > 0 else range(gtn.grid.ngrids - 1, -1, -1)

            for gk_idx in gk_inds:

                not_at_edge = gk_idx < gtn.grid.ngrids - 1 if sweep_direction > 0 else gk_idx > 0

                ## L2R + R2L
                out = gtn._branch_local_TE(dt, gk_idx, mpo_list, mpo_envs, sweep_direction=0, te_order=te_order,
                                           do_adapt=do_adapt, inplace=True, compress_config=compress_config,
                                           solver_type=LocalSolverType.TDVP)

                #### update spine
                if not_at_edge:
                    print('UPDATE SPINE', gk_idx)
                    spine_bond_size = gtn.spine.bond_size(gk_idx, gk_idx + 1 * sweep_direction)
                    max_bond = np.inf if compress_config is None else compress_config.max_bonds[1]
                    spine_adapt = False  # do_adapt and spine_bond_size < max_bond

                    nsites = 2 if spine_adapt else 1
                    # print('spine bond', spine_bond_size, nsites)
                    ## TDVP update
                    gtn._spine_tdvp(dt, gk_idx, nsites, mpo_list, mpo_envs, sweep_direction=sweep_direction,
                                    compress_config=compress_config, local_solver=LocalSolverType.TDVP,
                                    back_prop_dt=dt)
                    ## now canonizcalied to gk_idx +/- 1

        return gtn

    def evolve_tdvp_v3(self, dt, mpo_list: Sequence['GridTN1DComb'], te_order=0, do_adapt=True, inplace=False,
                       compress_config: 'CompressionConfiguration' = None, expand_basis=None):
        """ this ordering gets good results for Buneman? JK it doesn't actually?
            ---- ----      --------------
            ^  | ^  |      |            ^
            |  v |  v ...  v   .    .   |

            not efficient implementation (extra canonicalizations where not needed)
        """

        gtn = self if inplace else self.copy()
        gtn.collect_exponents()

        mpo_list = [mpo.copy() for mpo in mpo_list]
        for mpo in mpo_list:
            # mpo_copy = mpo.copy()
            mpo.collect_exponents()
            # mpo.distribute_exponents()
            mpo.distribute_sign()
            # print('mpo sign', mpo.sign)
            # print('mpo diff', mpo_copy.exponent, mpo.exponent, mpo_copy.sign, mpo.sign, mpo_copy.distance(mpo))
        # exit()
        # num_mpos = len(mpo_list)

        gtn.canonicalize_around_i(0, scale=False, redo_canon=True)
        # gtn.collect_exponents()     # need if scale is True
        canon_branch = None
        first_idx = None

        ## build branch envs
        mpo_envs = []
        for mpo in mpo_list:

            if mpo.spine.site_ind_id == gtn.spine.site_ind_id:
                mpo.spine_ind_id = mpo.spine_ind_id + '_op_'

            left_envs, right_envs, branch_envs = {}, {}, {}
            for gk_idx in range(gtn.grid.ngrids - 1, 0, -1):
                # print('branch exponents', gtn[gk_idx].exponent, mpo[gk_idx].exponent)
                ### hacky solution to <x|O|x>
                expec = gtn.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=True)
                ### includes branch, mpo branch exponents. ideally, shouldn't
                # expec.modify(apply = lambda x: x * 10**(-(mpo[gk_idx].exponent + 2*gtn[gk_idx].exponent)))
                expec.drop_tags()
                branch_envs[gk_idx] = expec
                right_envs[gk_idx - 1] = gtn._contract_branch_expec_spine(gk_idx, mpo, expec,
                                                                          right_envs.get(gk_idx, None),
                                                                          is_sqrt=True)

            mpo_envs += [(left_envs, right_envs, branch_envs)]

        ## left to right sweep
        for idx in range(gtn.num_active_branches - 1):

            # print('CHECK ORTHOG AROUND', gk_idx)
            # print('spine', helper.check_orthog(gtn.spine))
            # for i in range(gtn.grid.ngrids):
            #     print('branch', i, helper.check_right_orthog(gtn[i].data,
            #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))

            gk_idx = gtn.grid.get_grid_ind(gtn.active_grids[idx])
            next_gk_idx = gtn.grid.get_grid_ind(gtn.active_grids[idx + 1])

            ## branch r2l (end to anchor)
            gtn._branch_local_TE(dt / 2, gk_idx, mpo_list, mpo_envs, sweep_direction=1, te_order=te_order,
                                 do_adapt=do_adapt, inplace=True, compress_config=compress_config,
                                 solver_type=LocalSolverType.TDVP)
            first_idx = gk_idx if first_idx is None else first_idx

            #### update spine
            for spine_idx in range(gk_idx, next_gk_idx):
                ## if already updated
                # print('gk_idx', gk_idx, gtn.grid.ngrids - 1, spine_adapt)
                # if gk_idx == gtn.grid.ngrids - 1:  # and spine_adapt:
                #     continue

                # print('UPDATE SPINE', gk_idx)
                spine_bond_size = gtn.spine.bond_size(spine_idx, spine_idx + 1)
                max_bond = np.inf if compress_config is None else compress_config.max_bonds[1]
                spine_adapt = False  # do_adapt and spine_bond_size < max_bond

                nsites = 2 if spine_adapt else 1
                print('spine tdvp left site', spine_idx, nsites)
                gtn._spine_tdvp(dt / 2, spine_idx, nsites, mpo_list, mpo_envs, sweep_direction=1,
                                forward_prop=(spine_idx != next_gk_idx - 1), compress_config=compress_config)
                ## now canonicalized to next_gk_idx

            #### update branch at next_gk_idx, l2r
            gtn._branch_local_TE(dt / 2, next_gk_idx, mpo_list, mpo_envs, sweep_direction=-1, te_order=te_order,
                                 do_adapt=do_adapt, inplace=True, compress_config=compress_config,
                                 solver_type=LocalSolverType.TDVP)
            last_idx = next_gk_idx

        #### rightmost to leftmost branch sweep
        ## rightmost branch r2l (end to anchor)
        print('right to left', last_idx, first_idx)
        gtn._branch_local_TE(dt / 2, last_idx, mpo_list, mpo_envs, sweep_direction=1, te_order=te_order,
                             do_adapt=do_adapt, inplace=True, compress_config=compress_config,
                             solver_type=LocalSolverType.TDVP)

        ## update spine (left to right), then canonize spine
        # spine_bond_size = gtn.spine.bond_size(last_idx, last_idx - 1)
        # max_bond = np.inf if compress_config is None else compress_config.max_bonds[1]
        spine_adapt = False  # do_adapt and spine_bond_size < max_bond

        nsites = 2 if spine_adapt else 1
        left_gk_idx = last_idx - nsites + 1
        # print('UPDATE SPINE (RL)', last_idx, left_gk_idx)
        gtn._spine_tdvp(dt / 2, left_gk_idx, nsites, mpo_list, mpo_envs, sweep_direction=-1,
                        compress_config=compress_config)
        ## updates spine envs at last_idx

        ## canonize and update spine envs
        for spine_idx in range(last_idx - 1, first_idx, -1):
            print('CANONIZE SPINE', spine_idx)
            helper.canonize_tens_list(gtn.spine[spine_idx], gtn.spine[spine_idx - 1], inplace=True)
            gtn._update_spine_envs(spine_idx, mpo_list, mpo_envs, -1)

        ## leftmost branch r2l (anchor to end)
        gtn._branch_local_TE(dt / 2, first_idx, mpo_list, mpo_envs, sweep_direction=-1, te_order=te_order,
                             do_adapt=do_adapt, inplace=True, compress_config=compress_config,
                             solver_type=LocalSolverType.TDVP)

        return gtn

    def evolve_tdmrg_v2(self, dt, mpo_list: Sequence['GridTN1DComb'], te_order=0, do_adapt=True, inplace=False,
                        compress_config: 'CompressionConfiguration' = None, expand_basis=None, sweep_direction=1):
        """ this ordering gets normal results for Buneman if spine is evolved using TDVP
            if we want to evolve spine with tdDMRG, we need to be more careful with amount each
            site/bond is evolved by. probably need single site/bond evolution.
        """
        gtn = self if inplace else self.copy()
        gtn.collect_exponents()
        # gtn_copy = gtn.copy()

        spine_use_tdmrg = False

        mpo_list = [mpo.copy() for mpo in mpo_list]
        for mpo in mpo_list:
            mpo.collect_exponents()
            mpo.distribute_sign()
        # num_mpos = len(mpo_list)

        if sweep_direction > 0:
            gtn.canonicalize_around_i(0, scale=False, redo_canon=True)
        else:
            gtn.canonicalize_around_i(gtn.grid.ngrids - 1, scale=False, redo_canon=True)

        ## build branch envs
        mpo_envs = []
        spine_adapt = False
        for mpo in mpo_list:

            if mpo.spine.site_ind_id == gtn.spine.site_ind_id:
                mpo.spine_ind_id = mpo.spine_ind_id + '_op_'

            left_envs, right_envs, branch_envs = {}, {}, {}
            for gk_idx in range(gtn.grid.ngrids - 1, 0, -1):
                # print('branch exponents', gtn[gk_idx].exponent, mpo[gk_idx].exponent)
                ### hacky solution to <x|O|x>
                expec = gtn.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=True)
                ### includes branch, mpo branch exponents. ideally, shouldn't
                # expec.modify(apply = lambda x: x * 10**(-(mpo[gk_idx].exponent + 2*gtn[gk_idx].exponent)))
                expec.drop_tags()
                branch_envs[gk_idx] = expec
                right_envs[gk_idx - 1] = gtn._contract_branch_expec_spine(gk_idx, mpo, expec,
                                                                          right_envs.get(gk_idx, None),
                                                                          is_sqrt=True)

            mpo_envs += [(left_envs, right_envs, branch_envs)]

        ## left to right sweep or right to left sweep
        gk_inds = range(gtn.grid.ngrids) if sweep_direction > 0 else range(gtn.grid.ngrids - 1, -1, -1)

        ovlp_projs = {}
        spine_ovlp_proj = None
        for gk_idx in gk_inds:

            # print('CHECK ORTHOG AROUND', gk_idx)
            # print('spine', helper.check_orthog(gtn.spine))
            # for i in range(gtn.grid.ngrids):
            #     print('branch', i, helper.check_right_orthog(gtn[i].data,
            #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))

            not_at_edge = gk_idx < gtn.grid.ngrids - 1 if sweep_direction > 0 else gk_idx > 0

            backprop_edge = not_at_edge if spine_use_tdmrg else False
            print('branch backprop last site', gk_idx, backprop_edge)

            ## this actually isn't important to the alg; only important for tensors ancilla inds
            # if spine_ovlp_proj is None:
            spine_ovlp_proj = None
            if gk_idx < gtn.grid.ngrids - 1:
                tt_ind = gtn.spine.bond(gk_idx, gk_idx + 1)
                tt_ind_size = gtn.spine[gk_idx].ind_size(tt_ind)
                spine_ovlp_proj = qtn.Tensor(np.eye(tt_ind_size), inds=(tt_ind, tt_ind + '_'))

            if gk_idx > 0:
                tt_ind = gtn.spine.bond(gk_idx, gk_idx - 1)
                tt_ind_size = gtn.spine[gk_idx].ind_size(tt_ind)
                spine_ovlp_proj_2 = qtn.Tensor(np.eye(tt_ind_size), inds=(tt_ind, tt_ind + '_'))
                if spine_ovlp_proj is None:
                    spine_ovlp_proj = spine_ovlp_proj_2
                else:
                    # spine_ovlp_proj = qtn.tensor_contract(spine_ovlp_proj, spine_ovlp_proj_2)
                    spine_ovlp_proj = qtn.TensorNetwork([spine_ovlp_proj, spine_ovlp_proj_2])

            # print('spine ovlp', spine_ovlp_proj)

            ## L2R + R2L
            out = gtn._branch_local_TE(dt, gk_idx, mpo_list, mpo_envs, sweep_direction=0, te_order=te_order,
                                       do_adapt=do_adapt, inplace=True, compress_config=compress_config,
                                       solver_type=LocalSolverType.TDDMRG,
                                       backprop_edge=backprop_edge, ket_env0_L=spine_ovlp_proj)

            ## is spine is evolved with TDDMRG, need to keep track of overlap projectors of branch
            if spine_use_tdmrg:
                ovlp_projs[gk_idx] = out.ovlp_proj

            #### update spine
            if not_at_edge:
                ## if already updated
                # print('gk_idx', gk_idx, gtn.grid.ngrids - 1, spine_adapt)
                # if gk_idx == gtn.grid.ngrids - 1:  # and spine_adapt:
                #     continue

                # print('UPDATE SPINE', gk_idx)
                spine_bond_size = gtn.spine.bond_size(gk_idx, gk_idx + 1 * sweep_direction)
                max_bond = np.inf if compress_config is None else compress_config.max_bonds[1]
                spine_adapt = False  # do_adapt and spine_bond_size < max_bond

                nsites = 2 if spine_adapt else 1
                print('spine bond', spine_bond_size, nsites)
                if spine_use_tdmrg:
                    #### Warning: does not evolve in time correctly
                    ovlp_proj = ovlp_projs[gk_idx]
                    nsites = 2  # if spine_adapt else 1
                    out = gtn._spine_tdmrg(dt / 2, gk_idx, nsites, mpo_list, mpo_envs, ovlp_proj,
                                           sweep_direction=sweep_direction, compress_config=compress_config, )
                    spine_ovlp_proj = out[-1]
                    # print('spine ovlp proj', spine_ovlp_proj.norm())
                else:  ## TDVP update
                    gtn._spine_tdvp(dt, gk_idx, nsites, mpo_list, mpo_envs, sweep_direction=sweep_direction,
                                    compress_config=compress_config, local_solver=LocalSolverType.TDVP,
                                    back_prop_dt=dt)
                ## now canonizcalied to gk_idx +/- 1

        return gtn


    def _evolve_tdmrg_new_2D(self, dt, linear_mpo_list: Sequence['GridTN1DComb'], te_order=0, do_adapt=True, inplace=False,
                         compress_config: 'CompressionConfiguration' = None, nonlinear_terms=None, sources=None,
                         solver_type=LocalSolverType.TDDMRG):

        max_bond = compress_config.get_compress_opts(1)['max_bond'] if compress_config is not None else None

        if self.grid.ngrids > 2:
            raise NotImplementedError

        # from local_solvers.local_cross_eval import local_cross_evaluator, Term_Cross
        from local_solvers.time_integrator import TDDMRG as TimeInteg

        gtn = self if inplace else self.copy()
        dist_mpx = self.grid.gtn_to_dmrg_format(gtn, is_mps=True)  # (self.data_type is DataType.MPS))

        linear_mpo_list = [self.grid.gtn_to_dmrg_format(mpo.copy(), is_mps=False) for mpo in linear_mpo_list]
        # for mpo in linear_mpo_list:
        #     mpo.collect_exponents()
        #     mpo.distribute_sign()

        solver = TimeInteg(dist_mpx, [mpo for mpo in linear_mpo_list],
                           sources=sources, nonlinear_terms=nonlinear_terms,
                           te_order_target=te_order, te_order_final=te_order,
                           max_bond=max_bond, dt=dt / 2)

        solver.solve_l2r(1, canonize=True)
        solver.direction = solver.direction * -1
        solver.update_ket_from_out()
        solver.solve_r2l(1, canonize=False)
        # gtn.data = solver.out

        out = self.grid.dmrg_to_gtn_format(gtn, solver.out, is_mps=True)
        return out

    def evolve_tdmrg_new(self, dt, linear_mpo_list: Sequence['GridTN1DComb'], te_order=0, do_adapt=True, inplace=False,
                         compress_config: 'CompressionConfiguration' = None, nonlinear_terms=None, sources=None,
                         solver_type=LocalSolverType.TDDMRG, filter_bases=False, time=None):

        if self.grid.ngrids <= 2:
            return self._evolve_tdmrg_new_2D(dt, linear_mpo_list, te_order=te_order, do_adapt=do_adapt,
                                             inplace=inplace, compress_config = compress_config,
                                             nonlinear_terms=nonlinear_terms, sources=sources,
                                             solver_type=solver_type)
        else:
            raise NotImplementedError


    def evolve_tdmrg(self, dt, linear_mpo_list: Sequence['GridTN1DComb'], te_order=0, do_adapt=True, inplace=False,
                     compress_config: 'CompressionConfiguration' = None):
        """ this ordering gets stuck and yields bad results for Buneman
        """

        gtn = self if inplace else self.copy()
        gtn.collect_exponents()
        # gtn_copy = gtn.copy()

        spine_use_tdmrg = False  # True

        linear_mpo_list = [mpo.copy() for mpo in linear_mpo_list]
        for mpo in linear_mpo_list:
            mpo.collect_exponents()
            mpo.distribute_sign()
        # num_mpos = len(mpo_list)

        gtn.canonicalize_around_i(0, scale=False, redo_canon=True)

        ## build branch envs
        mpo_envs = []
        spine_adapt = False
        for mpo in linear_mpo_list:

            if mpo.spine.site_ind_id == gtn.spine.site_ind_id:
                mpo.spine_ind_id = mpo.spine_ind_id + '_op_'

            left_envs, right_envs, branch_envs = {}, {}, {}
            for gk_idx in range(gtn.grid.ngrids - 1, 0, -1):
                # print('branch exponents', gtn[gk_idx].exponent, mpo[gk_idx].exponent)
                ### hacky solution to <x|O|x>
                expec = gtn.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=True)
                ### includes branch, mpo branch exponents. ideally, shouldn't
                # expec.modify(apply = lambda x: x * 10**(-(mpo[gk_idx].exponent + 2*gtn[gk_idx].exponent)))
                expec.drop_tags()
                branch_envs[gk_idx] = expec
                right_envs[gk_idx - 1] = gtn._contract_branch_expec_spine(gk_idx, mpo, expec,
                                                                          right_envs.get(gk_idx, None),
                                                                          is_sqrt=True)

            mpo_envs += [(left_envs, right_envs, branch_envs)]

        ## left to right sweep
        ovlp_projs = {}
        spine_ovlp_proj = None
        for gk_idx in range(gtn.grid.ngrids):

            # print('CHECK ORTHOG AROUND', gk_idx)
            # print('spine', helper.check_orthog(gtn.spine))
            # for i in range(gtn.grid.ngrids):
            #     print('branch', i, helper.check_right_orthog(gtn[i].data,
            #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))

            backprop_edge = gk_idx < gtn.grid.ngrids - 1 if spine_use_tdmrg else False
            print('branch backprop last site', gk_idx, backprop_edge)

            branch_dir = 1 if gk_idx != gtn.grid.ngrids - 1 else -1  # 1: r2l (edge to anchor), -1: l2r
            spine_ovlp_proj = None
            if branch_dir < 0:
                if gk_idx < gtn.grid.ngrids - 1:
                    tt_ind = gtn.spine.bond(gk_idx, gk_idx + 1)
                    tt_ind_size = gtn.spine[gk_idx].ind_size(tt_ind)
                    spine_ovlp_proj = qtn.Tensor(np.eye(tt_ind_size), inds=(tt_ind, tt_ind + '_'))

                if gk_idx > 0:
                    tt_ind = gtn.spine.bond(gk_idx, gk_idx - 1)
                    tt_ind_size = gtn.spine[gk_idx].ind_size(tt_ind)
                    spine_ovlp_proj_2 = qtn.Tensor(np.eye(tt_ind_size), inds=(tt_ind, tt_ind + '_'))
                    if spine_ovlp_proj is None:
                        spine_ovlp_proj = spine_ovlp_proj_2
                    else:
                        # spine_ovlp_proj = qtn.tensor_contract(spine_ovlp_proj, spine_ovlp_proj_2)
                        spine_ovlp_proj = qtn.TensorNetwork([spine_ovlp_proj, spine_ovlp_proj_2])

            ## ovlp proj doesn't matter for R2L direction
            out = gtn._branch_local_TE(dt / 2, gk_idx, linear_mpo_list, mpo_envs, sweep_direction=branch_dir,
                                       te_order=te_order,
                                       do_adapt=do_adapt, inplace=True, compress_config=compress_config,
                                       solver_type=LocalSolverType.TDDMRG,
                                       ket_env0_L=spine_ovlp_proj,
                                       backprop_edge=backprop_edge)

            ## is spine is evolved with TDDMRG, need to keep track of overlap projectors of branch
            if spine_use_tdmrg:
                ovlp_projs[gk_idx] = out.ovlp_proj

            #### update spine
            if gk_idx < gtn.grid.ngrids - 1:
                ## if already updated
                # print('gk_idx', gk_idx, gtn.grid.ngrids - 1, spine_adapt)
                # if gk_idx == gtn.grid.ngrids - 1:  # and spine_adapt:
                #     continue

                spine_bond_size = gtn.spine.bond_size(gk_idx, gk_idx + 1)
                max_bond = np.inf if compress_config is None else compress_config.max_bonds[1]
                spine_adapt = do_adapt and spine_bond_size < max_bond

                nsites = 2 if spine_adapt else 1
                if spine_use_tdmrg:
                    if spine_ovlp_proj is not None:
                        ovlp_proj = qtn.TensorNetwork([ovlp_projs[gk_idx], spine_ovlp_proj])
                    else:
                        ovlp_proj = ovlp_projs[gk_idx]
                    nsites = 2  # if spine_adapt else 1
                    out = gtn._spine_tdmrg(dt / 2, gk_idx, nsites, linear_mpo_list, mpo_envs, ovlp_proj,
                                           sweep_direction=1, compress_config=compress_config, )
                    spine_ovlp_proj = out[-1]
                    print('spine ovlp proj', spine_ovlp_proj.norm())
                else:  ## TDVP update
                    gtn._spine_tdvp(dt / 2, gk_idx, nsites, linear_mpo_list, mpo_envs, sweep_direction=1,
                                    compress_config=compress_config, local_solver=LocalSolverType.TDVP)
                ## now canonizcalied to gk_idx + 1

        ## right to left sweep
        ovlp_projs = {}
        for gk_idx in range(gtn.grid.ngrids - 1, -1, -1):

            # print('CHECK ORTHOG AROUND', gk_idx)
            # print('spine', helper.check_orthog(gtn.spine))
            # for i in range(gtn.grid.ngrids):
            #     print('branch', i, helper.check_right_orthog(gtn[i].data,
            #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))

            backprop_edge = gk_idx > 0 if spine_use_tdmrg else False
            print('branch backprop last site', gk_idx, backprop_edge)

            spine_ovlp_proj = None
            branch_dir = -1 if gk_idx != gtn.grid.ngrids - 1 else 1  # 1: r2l (edge to anchor), -1: l2r
            if branch_dir < 0:
                if gk_idx < gtn.grid.ngrids - 1:
                    tt_ind = gtn.spine.bond(gk_idx, gk_idx + 1)
                    tt_ind_size = gtn.spine[gk_idx].ind_size(tt_ind)
                    spine_ovlp_proj = qtn.Tensor(np.eye(tt_ind_size), inds=(tt_ind, tt_ind + '_'))

                if gk_idx > 0:
                    tt_ind = gtn.spine.bond(gk_idx, gk_idx - 1)
                    tt_ind_size = gtn.spine[gk_idx].ind_size(tt_ind)
                    spine_ovlp_proj_2 = qtn.Tensor(np.eye(tt_ind_size), inds=(tt_ind, tt_ind + '_'))
                    if spine_ovlp_proj is None:
                        spine_ovlp_proj = spine_ovlp_proj_2
                    else:
                        # spine_ovlp_proj = qtn.tensor_contract(spine_ovlp_proj, spine_ovlp_proj_2)
                        spine_ovlp_proj = qtn.TensorNetwork([spine_ovlp_proj, spine_ovlp_proj_2])

            ## build right to left (end to anchor) (same as in previous sweep)
            # ## build l2r (anchor to end) (opposite previous sweep)
            # print('l2r dmrg')
            out = gtn._branch_local_TE(dt / 2, gk_idx, linear_mpo_list, mpo_envs, sweep_direction=branch_dir,
                                       te_order=te_order,
                                       do_adapt=do_adapt, inplace=True, compress_config=compress_config,
                                       solver_type=LocalSolverType.TDDMRG,
                                       backprop_edge=backprop_edge,
                                       ket_env0_L=spine_ovlp_proj
                                       )

            if spine_use_tdmrg:
                ovlp_projs[gk_idx] = out.ovlp_proj

            #### update spine
            if gk_idx > 0:

                # ## if already updated
                # # print('gk_idx', gk_idx, gtn.grid.ngrids - 1, spine_adapt)
                # if gk_idx == 0:  # and spine_adapt:
                #     continue

                print('UPDATE SPINE (RL)', gk_idx)
                spine_bond_size = gtn.spine.bond_size(gk_idx, gk_idx - 1)
                max_bond = np.inf if compress_config is None else compress_config.max_bonds[1]
                spine_adapt = do_adapt and spine_bond_size < max_bond

                nsites = 2 if spine_adapt else 1
                left_gk_idx = gk_idx - nsites + 1
                if spine_use_tdmrg:
                    if spine_ovlp_proj is not None:
                        ovlp_proj = qtn.TensorNetwork([ovlp_projs[gk_idx], spine_ovlp_proj])
                    else:
                        ovlp_proj = ovlp_projs[gk_idx]

                    nsites = 2  # if spine_adapt else 1
                    left_gk_idx = gk_idx - nsites + 1
                    print('left gk idx?', left_gk_idx)

                    out = gtn._spine_tdmrg(dt / 2, left_gk_idx, nsites, linear_mpo_list, mpo_envs, ovlp_proj,
                                           sweep_direction=-1, compress_config=compress_config, )
                    spine_ovlp_proj = out[-1]
                    print('spine ovlp proj', spine_ovlp_proj.norm())
                else:
                    gtn._spine_tdvp(dt / 2, left_gk_idx, nsites, linear_mpo_list, mpo_envs, sweep_direction=-1,
                                    compress_config=compress_config)

        return gtn

    def solve(self, operator: 'GridTN1DComb', compress_type: 'CompressType', inplace=False, use_A2=True,
              compress_opts=None, is_H=False, init_guess: 'GridTN1DComb' = None, verbose_output=False, **kwargs
              ) -> Union[tuple['GridTN1DComb', float, bool], 'GridTN1DComb']:
        """ solves Ax=b
        """
        max_bond = None if compress_opts is None else compress_opts.get('max_bond', None)

        ## if 1D
        if self.grid.ngrids == 1:
            gtn = self if inplace else self.copy()
            gtn.collect_exponents()
            # gtn.distribute_exponents()
            gtn.distribute_sign()

            operator.collect_exponents()
            operator.distribute_sign()

            branch = gtn._attach_spine_tens_to_branch_v2(0, is_mps=True)
            branch.exponent = gtn.exponent
            branch_op = operator._attach_spine_tens_to_branch_v2(0, is_mps=False)
            branch_op.exponent = operator.exponent

            if init_guess is None:
                init_guess_branch = branch.copy()
            else:
                init_guess = init_guess.copy()
                init_guess.spine_ind_id = gtn.spine_ind_id
                helper.match_inner_inds(init_guess.spine, gtn.spine, inplace=True)
                init_guess.collect_exponents()
                init_guess.distribute_sign()
                # init_guess.distribute_exponents()
                init_guess_branch = init_guess._attach_spine_tens_to_branch(0, is_mps=True)
                init_guess_branch.site_ind_id = branch.site_ind_id
                init_guess_branch.exponent = init_guess.exponent

            if use_A2:
                soln, err, is_conv = helper_dmrg_2.dmrg_solve_2(branch, branch_op, init_guess=init_guess_branch,
                                                                is_H=is_H, max_bond=max_bond, **kwargs)
            else:
                soln, err, is_conv = helper_dmrg.dmrg_solve(branch, branch_op, init_guess=init_guess_branch, is_H=is_H,
                                                            max_bond=max_bond, **kwargs)
            helper.match_inner_inds(soln, branch, inplace=True)
            # branch.exponent = 0.0
            gtn._exponent = soln.exponent  # - gtn[0].exponent
            soln.exponent = 0.0
            gtn._update_spine_tens_and_branch_v2(0, soln)
            gtn.canon_site = None

            if verbose_output:
                return gtn, err, is_conv
            return gtn

        ## if 2D
        elif self.grid.ngrids == 2:
            gtn = self if inplace else self.copy()
            gtn.collect_exponents()
            # gtn.distribute_exponents()
            gtn.distribute_sign()

            operator.collect_exponents()
            operator.distribute_sign()

            ## maybe could get slightly better results if contract spine tens into branch
            branch0 = gtn._attach_spine_tens_to_branch_v2(0, is_mps=True)
            branch0 = helper.mps_flip_lr(branch0, inplace=False)
            branch1 = gtn._attach_spine_tens_to_branch_v2(1, is_mps=True)
            branch = helper.append_mpx(branch0, branch1, inplace=False)
            branch.exponent = gtn.exponent

            branch0_op = operator._attach_spine_tens_to_branch_v2(0, is_mps=False)
            branch0_op = helper.mpo_flip_lr(branch0_op, inplace=False)
            branch1_op = operator._attach_spine_tens_to_branch_v2(1, is_mps=False)
            branch_op = helper.append_mpx(branch0_op, branch1_op, inplace=False)
            branch_op.exponent = operator.exponent

            if init_guess is None:
                init_guess_branch = None  # branch.copy()
            else:
                init_guess = init_guess.copy()
                init_guess.spine_ind_id = gtn.spine_ind_id
                helper.match_inner_inds(init_guess.spine, gtn.spine, inplace=True)
                init_guess.collect_exponents()
                init_guess.distribute_sign()
                # init_guess.distribute_exponents()
                init_guess_branch0 = init_guess._attach_spine_tens_to_branch_v2(0, is_mps=True)
                init_guess_branch0 = helper.mps_flip_lr(init_guess_branch0, inplace=False)
                init_guess_branch1 = init_guess._attach_spine_tens_to_branch_v2(1, is_mps=True)
                init_guess_branch = helper.append_mpx(init_guess_branch0, init_guess_branch1, inplace=False)
                init_guess_branch.site_ind_id = branch.site_ind_id
                init_guess_branch.exponent = init_guess.exponent

            # print('gtn', gtn)
            # print('branch', branch)
            # print('branch op', branch_op)
            if use_A2:
                soln, err, is_conv = helper_dmrg_2.dmrg_solve_2(branch, branch_op, init_guess=init_guess_branch,
                                                                is_H=is_H, max_bond=max_bond, **kwargs)
            else:
                soln, err, is_conv = helper_dmrg.dmrg_solve(branch, branch_op, init_guess=init_guess_branch, is_H=is_H,
                                                            max_bond=max_bond, **kwargs)
            # print('soln', soln)
            soln0, soln1 = helper.split_mpx(soln, gtn[0].data.L)  ## in case gtn[0].data.L != grid[0].L
            # soln0.distribute_exponent()
            # soln1.distribute_exponent()

            helper.match_inner_inds(soln0, branch0, inplace=True)
            helper.mps_flip_lr(soln0, inplace=True)

            soln1.site_ind_id = branch1.site_ind_id
            soln1.site_tag_id = branch1.site_tag_id
            helper.match_inner_inds(soln1, branch1, inplace=True)

            gtn._exponent = soln.exponent
            soln0.exponent = 0.0  # soln.exponent   ## already set global exponent
            soln1.exponent = 0.0  # soln.exponent
            gtn._sign = 1.0
            gtn._update_spine_tens_and_branch_v2(0, soln0)  ## doesn't update exponent
            gtn._update_spine_tens_and_branch_v2(1, soln1)  ## doesn't update exponent
            gtn.canon_site = None

            if verbose_output:
                return gtn, err, is_conv
            return gtn

        ## else
        conv_tol = kwargs.get('conv_tol', helper_dmrg.DEFAULT_CONV_TOL)
        max_iter = kwargs.get('max_iter', helper_dmrg.DEFAULT_MAX_ITER)
        do_adapt = True
        compress_config = CompressionConfiguration()
        compress_config.set_compress_opts(1, compress_opts)

        operator = operator.copy()
        operator.collect_exponents()
        operator.distribute_sign()
        mpo_list = [operator]

        target = self.copy()
        target.collect_exponents()
        target.distribute_sign()
        target_list = [target]

        init_guess = self.copy()
        init_guess.collect_exponents()

        init_guess.canonicalize_around_i(0, scale=False, redo_canon=True)

        ## build branch envs
        mpo_envs = []
        for mpo in mpo_list:

            if mpo.spine.site_ind_id == init_guess.spine.site_ind_id:
                mpo.spine_ind_id = mpo.spine_ind_id + '_op_'

            left_envs, right_envs, branch_envs = {}, {}, {}
            for gk_idx in range(init_guess.grid.ngrids - 1, 0, -1):
                # print('branch exponents', gtn[gk_idx].exponent, mpo[gk_idx].exponent)
                ### hacky solution to <x|O|x>
                expec = init_guess.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=True)
                ### includes branch, mpo branch exponents. ideally, shouldn't
                # expec.modify(apply = lambda x: x * 10**(-(mpo[gk_idx].exponent + 2*gtn[gk_idx].exponent)))
                expec.drop_tags()
                branch_envs[gk_idx] = expec
                right_envs[gk_idx - 1] = init_guess._contract_branch_expec_spine(gk_idx, mpo, expec,
                                                                                 right_envs.get(gk_idx, None),
                                                                                 is_sqrt=True)

            mpo_envs += [(left_envs, right_envs, branch_envs)]

        # print('mpo list', mpo_list)
        # print('mpo envs', mpo_envs)

        ## build branch target ovlp envs
        gtn_conj = init_guess.conj()
        gtn_conj.spine.mangle_inner_(append='_')
        gtn_conj.spine_ind_id = gtn_conj.spine_ind_id + '_'

        target_envs = []
        target_list = [target.copy() for target in target_list]
        for target in target_list:
            target.collect_exponents()

            if target.spine.site_ind_id == gtn_conj.spine.site_ind_id:
                target.spine_ind_id = target.spine_ind_id + '_b_'

            left_envs, right_envs, branch_envs = {}, {}, {}
            for gk_idx in range(gtn_conj.grid.ngrids - 1, 0, -1):
                # print('branch exponents', gtn[gk_idx].exponent, mpo[gk_idx].exponent)
                ### hacky solution to <x|O|x>
                expec = gtn_conj.meas_ovlp_branch(gk_idx, target)
                ### includes branch, mpo branch exponents. ideally, shouldn't
                # expec.modify(apply = lambda x: x * 10**(-(mpo[gk_idx].exponent + 2*gtn[gk_idx].exponent)))
                expec.drop_tags()
                branch_envs[gk_idx] = expec
                right_envs[gk_idx - 1] = gtn_conj._contract_branch_expec_spine(gk_idx, None, expec,
                                                                               right_envs.get(gk_idx, None),
                                                                               bra_gtn=target, )

            target_envs += [(left_envs, right_envs, branch_envs)]

        # print('mpo list', target_list)
        # print('mpo envs', target_envs)

        ## iterative solver
        err = target.distance(init_guess.apply(operator))
        it, conv_it, prev_err, min_err = 0, 0, err, err
        direction = 1
        soln, min_soln = init_guess, init_guess

        while err > conv_tol and conv_it < max_iter and it < helper_dmrg.DEFAULT_MAX_TOT_ITER:

            it += 1

            soln, err = soln.sweep_dmrg(target_list, mpo_list, target_envs, mpo_envs, do_adapt=do_adapt, inplace=True,
                                        compress_config=compress_config, sweep_direction=direction)
            soln, err = soln.sweep_dmrg(target_list, mpo_list, target_envs, mpo_envs, do_adapt=do_adapt, inplace=True,
                                        compress_config=compress_config, sweep_direction=direction * -1)

            if np.abs((prev_err - err) / err) < 1.0e-4:
                conv_it += 1

            prev_err = err

            if err < min_err:
                min_soln = soln.copy(deep=True)
                # min_ket = self.ket.copy()
                min_err = err
            else:
                print('Warning: solve error went up', err, min_err)

        ## revert to optimal results
        if inplace:
            self.data = min_soln
            return self
        else:
            return min_soln

    def sweep_dmrg(self, target_list: Sequence['GridTN1DComb'], mpo_list: Sequence['GridTN1DComb'],
                   target_envs: Sequence[Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']]],
                   mpo_envs: Sequence[Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']]],
                   do_adapt=True, inplace=True, compress_config: 'CompressionConfiguration' = None,
                   sweep_direction=1):
        """ this ordering gets normal results for Buneman if spine is evolved using TDVP
            if we want to evolve spine with tdDMRG, we need to be more careful with amount each
            site/bond is evolved by. probably need single site/bond evolution.
        """
        gtn = self if inplace else self.copy()

        ## left to right sweep or right to left sweep
        gk_inds = range(gtn.grid.ngrids) if sweep_direction > 0 else range(gtn.grid.ngrids - 1, -1, -1)

        for gk_idx in gk_inds:

            # print('CHECK ORTHOG AROUND', gk_idx)
            # print('spine', helper.check_orthog(gtn.spine))
            # for i in range(gtn.grid.ngrids):
            #     print('branch', i, helper.check_right_orthog(gtn[i].data,
            #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))

            not_at_edge = gk_idx < gtn.grid.ngrids - 1 if sweep_direction > 0 else gk_idx > 0

            # print('target envs', target_envs)
            # print('mpo envs', mpo_envs)

            ## L2R + R2L branch
            out = gtn._branch_local_solver(gk_idx, target_list, mpo_list, target_envs, mpo_envs,
                                           sweep_direction=0, do_adapt=do_adapt, inplace=True,
                                           compress_config=compress_config, solver_type=LocalSolverType.DMRG,
                                           )
            solver, err = out

            #### update spine
            if not_at_edge:
                ## if already updated
                # print('gk_idx', gk_idx, gtn.grid.ngrids - 1, spine_adapt)
                # if gk_idx == gtn.grid.ngrids - 1:  # and spine_adapt:
                #     continue

                # print('UPDATE SPINE', gk_idx)
                spine_bond_size = gtn.spine.bond_size(gk_idx, gk_idx + 1 * sweep_direction)
                max_bond = np.inf if compress_config is None else compress_config.max_bonds[1]
                spine_adapt = do_adapt and spine_bond_size < max_bond

                nsites = 2 if spine_adapt else 1
                gtn_spine, err = gtn._spine_solve(gk_idx, nsites, target_list, mpo_list, target_envs, mpo_envs,
                                                  sweep_direction=sweep_direction, compress_config=compress_config, )
                ## now canonizcalied to gk_idx +/- 1

        return gtn, err

    # def evolve_local(self, dt, mpo_list: Sequence['GridTN1DComb'], te_order=0, do_adapt=True, inplace=False,
    #                  solver_type=LocalSolverType.TDDMRG, spine_solver_type=LocalSolverType.TDVP,
    #                  compress_config: 'CompressionConfiguration'=None, expand_basis=None):
    #
    #     gtn = self if inplace else self.copy()
    #     gtn.collect_exponents()
    #     # print('gtn exponent', gtn.exponent)
    #     # gtn.distribute_exponents()
    #     # gtn_copy = gtn.copy()
    #
    #     ovlp_proj = None    ## only used for TDDMRG
    #     if solver_type is LocalSolverType.TDDMRG:
    #         gtn_copy = gtn.copy()
    #         BranchLocalSolver = TDMRGSolver
    #         is_TE = True
    #     elif solver_type is LocalSolverType.TDVP:
    #         BranchLocalSolver = TDVPSolver
    #         is_TE = True
    #     elif solver_type is LocalSolverType.DMRG:
    #         BranchLocalSolver = DMRGSolver
    #         is_TE = False
    #     elif solver_type is LocalSolverType.LINSOLVE:
    #         BranchLocalSolver = LinearSolver
    #         is_TE = False
    #     else:
    #         raise NotImplementedError
    #
    #
    #     mpo_list = [mpo.copy() for mpo in mpo_list]
    #     for mpo in mpo_list:
    #         # mpo_copy = mpo.copy()
    #         mpo.collect_exponents()
    #         # mpo.distribute_exponents()
    #         mpo.distribute_sign()
    #         # print('mpo sign', mpo.sign)
    #         # print('mpo diff', mpo_copy.exponent, mpo.exponent, mpo_copy.sign, mpo.sign, mpo_copy.distance(mpo))
    #     # exit()
    #     num_mpos = len(mpo_list)
    #
    #
    #     gtn.canonicalize_around_i(0, scale=False, redo_canon=True)
    #     # gtn.collect_exponents()     # need if scale is True
    #
    #     ## build branch envs
    #     mpo_envs = []
    #     spine_adapt = False
    #     for mpo in mpo_list:
    #
    #         if mpo.spine.site_ind_id == gtn.spine.site_ind_id:
    #             mpo.spine_ind_id = mpo.spine_ind_id + '_op_'
    #
    #         left_envs, right_envs, branch_envs = {}, {}, {}
    #         for gk_idx in range(gtn.grid.ngrids-1, 0, -1):
    #             # print('branch exponents', gtn[gk_idx].exponent, mpo[gk_idx].exponent)
    #             ### hacky solution to <x|O|x>
    #             expec = gtn.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=True)
    #             ### includes branch, mpo branch exponents. ideally, shouldn't
    #             # expec.modify(apply = lambda x: x * 10**(-(mpo[gk_idx].exponent + 2*gtn[gk_idx].exponent)))
    #             expec.drop_tags()
    #             branch_envs[gk_idx] = expec
    #             right_envs[gk_idx-1] = gtn._contract_branch_expec_spine(gk_idx, mpo, expec,
    #                                                                     right_envs.get(gk_idx, None),
    #                                                                     is_sqrt=True)
    #
    #         mpo_envs += [(left_envs, right_envs, branch_envs)]
    #
    #     def get_site_mpo_envs(gk_idx_):
    #         site_mpo_envs = []
    #         for mpo_ind in range(len(mpo_list)):
    #             # print(mpo_envs[mpo_ind])
    #             left_envs, right_envs, branch_envs = mpo_envs[mpo_ind]
    #             left_env = left_envs.get(gk_idx_, None)
    #             right_env = right_envs.get(gk_idx_, None)
    #             # print('left env', left_env, left_env.norm() if left_env is not None else None)
    #             # print('right env', right_env, right_env.norm() if right_env is not None else None)
    #             if left_env is not None and right_env is not None:
    #                 site_mpo_envs += [qtn.tensor_contract(left_env, right_env)]
    #             elif left_env is not None:
    #                 site_mpo_envs += [left_env.copy()]
    #             elif right_env is not None:
    #                 site_mpo_envs += [right_env.copy()]
    #             # print('site mpo envs', site_mpo_envs[-1].norm())
    #             # print('modify with exponent', mpo_list[mpo_ind].exponent, gtn.exponent)
    #             site_mpo_envs[-1].modify(apply=lambda x: x * 10**mpo_list[mpo_ind].exponent)
    #             # print('site mpo envs', site_mpo_envs[-1].norm(), mpo_list[mpo_ind].exponent)
    #         return site_mpo_envs
    #
    #     ## left to right sweep
    #     for gk_idx in range(gtn.grid.ngrids):
    #
    #         # print('CHECK ORTHOG AROUND', gk_idx)
    #         # print('spine', helper.check_orthog(gtn.spine))
    #         # for i in range(gtn.grid.ngrids):
    #         #     print('branch', i, helper.check_right_orthog(gtn[i].data,
    #         #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))
    #
    #         ## ovlp proj doesn't matter for this direction unless we do spine TDDMRG
    #         # if solver_type is LocalSolverType.TDDMRG:
    #         #     b = gtn[gk_idx]
    #         #     b.canonize(inplace=True, scale=False, i=0)
    #         #     spine_tens_copy = gtn.spine[gk_idx].copy()
    #
    #         out = gtn._branch_local_TE(dt / 2, gk_idx, mpo_list, mpo_envs, sweep_direction=1, te_order=te_order,
    #                                    do_adapt=do_adapt, inplace=True, compress_config=compress_config,
    #                                    solver_type=LocalSolverType.TDDMRG, ovlp_proj=ovlp_proj)
    #
    #
    #         # ## new mpx object
    #         # dist_mpx = gtn._attach_spine_tens_to_branch(gk_idx, is_mps=True)
    #         #
    #         # site_mpo_envs = get_site_mpo_envs(gk_idx)
    #         # ## mpo exponent included in envs
    #         #
    #         # ## tdvp of branch + spine. TDVP does not change exponent of the ket.
    #         # b_mpos = []
    #         # for mpo in mpo_list:
    #         #     mpo_branch = mpo._attach_spine_tens_to_branch(gk_idx, is_mps=False)
    #         #     # print('mpo branch exponent', gk_idx, mpo_branch.exponent, gtn[gk_idx].exponent)
    #         #     b_mpos += [mpo_branch]
    #         #
    #         # if solver_type is LocalSolverType.TDVP:
    #         #     tdvp_solver = TDVPSolver(dist_mpx, operators=b_mpos, te_order=te_order,
    #         #                              norm_env0_Ls=site_mpo_envs, compress_config=compress_config)
    #         #     tdvp_solver.take_time_step_r2l(dt / 2, do_adapt=do_adapt, canonize=True, build_envs=True)
    #         # elif solver_type is LocalSolverType.TDDMRG:
    #         #     ind = gtn.spine.bond(gk_idx, gk_idx + 1)
    #         #     ind_size = gtn.spine[gk_idx].ind_size(ind)
    #         #     left_env = qtn.Tensor(np.eye(ind_size), inds=(ind, ind+'_'))
    #         #     tdvp_solver = TDMRGSolver(dist_mpx, operators=b_mpos, te_order=te_order, ket_env0_L=left_env,
    #         #                               norm_env0_Ls=site_mpo_envs, compress_config=compress_config)
    #         #     # tdvp_solver.take_time_step(dt, do_adapt=do_adapt)
    #         #     tdvp_solver.take_time_step_r2l(dt / 2, do_adapt=do_adapt, canonize=True, build_envs=True)
    #         # else:
    #         #     raise NotImplementedError
    #         # ## now canon at site 0 (spine tens)
    #         #
    #         #
    #         # #### update comb tensors / spine
    #         # gtn._update_spine_tens_and_branch(gk_idx, tdvp_solver.ket)
    #         #
    #         # # print('comb orthog? should be', gk_idx)
    #         # # print('spine', helper.check_orthog(gtn.spine))
    #         # # for i in range(gtn.grid.ngrids):
    #         # #     print('branch', i, helper.check_right_orthog(gtn[i].data,
    #         # #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))
    #         #
    #         # #### update envs with new branch
    #         # for mpo_ind in range(num_mpos):
    #         #     mpo = mpo_list[mpo_ind]
    #         #     left_envs, right_envs, branch_envs = mpo_envs[mpo_ind]
    #         #     # print('update expec', gk_idx, mpo[gk_idx].exponent, gtn[gk_idx].exponent)
    #         #     expec = gtn.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=True)
    #         #     ### does not include global exponents
    #         #     # expec.modify(apply=lambda x: x * 10**(-(gtn.exponent * 2 + mpo.exponent)))
    #         #     expec.drop_tags()
    #         #     branch_envs[gk_idx] = expec
    #
    #
    #         #### update spine
    #         if gk_idx < gtn.grid.ngrids - 1:
    #
    #             ## if already updated
    #             # print('gk_idx', gk_idx, gtn.grid.ngrids - 1, spine_adapt)
    #             # if gk_idx == gtn.grid.ngrids - 1:  # and spine_adapt:
    #             #     continue
    #
    #             # print('UPDATE SPINE', gk_idx)
    #             spine_bond_size = gtn.spine.bond_size(gk_idx, gk_idx + 1)
    #             max_bond = np.inf if compress_config is None else compress_config.max_bonds[1]
    #             # if max_bond is not None:
    #             #     max_bond = min(max_bond, 2 ** (gk_idx + 1), 2 ** (self.grid.ndims - gk_idx - 1))
    #             spine_adapt = do_adapt and spine_bond_size < max_bond
    #
    #             nsites = 2 if spine_adapt else 1
    #             # print('spine tdvp left site', gk_idx, nsites)
    #             if is_TE:
    #                 gtn._spine_tdvp(dt / 2, gk_idx, nsites, mpo_list, mpo_envs, sweep_direction=1,
    #                                 compress_config=compress_config, local_solver=spine_solver_type)
    #                 ## now canonizcalied to gk_idx + 1
    #
    #                 # if solver_type is LocalSolverType.TDDMRG:
    #                 #     ovlp_proj_branch = out[1]  # doesn't include last optimized site (spine tens)
    #                 #     ovlp_proj = qtn.tensor_contract(ovlp_proj_branch, gtn.spine[gk_idx], spine_tens_copy)
    #             else:
    #                 raise NotImplementedError
    #
    #     ## right to left sweep
    #     for gk_idx in range(gtn.grid.ngrids-1, -1, -1):
    #
    #         print('CHECK ORTHOG AROUND', gk_idx)
    #         print('spine', helper.check_orthog(gtn.spine))
    #         for i in range(gtn.grid.ngrids):
    #             print('branch', i, helper.check_right_orthog(gtn[i].data,
    #                                                          left_ancillas=(gtn.spine_ind_id.format(i),)))
    #
    #         if solver_type is LocalSolverType.TDDMRG:
    #             b = gtn[gk_idx]
    #             b.canonize(inplace=True, scale=False, i=0)
    #             spine_tens_copy = gtn.spine[gk_idx].copy()
    #
    #         # ## build left to right
    #         # gtn._branch_local_TE(dt / 2, gk_idx, mpo_list, mpo_envs, sweep_direction=-1, te_order=te_order,
    #         #                      do_adapt=do_adapt, inplace=True, compress_config=compress_config,
    #         #                      solver_type=solver_type)
    #
    #         ## new mpx object
    #         dist_mpx = gtn._attach_spine_tens_to_branch(gk_idx, is_mps=True)
    #         # print('dist mpx norm', helper.norm(dist_mpx))
    #         # print('dist mpx check orthog', helper.check_orthog(dist_mpx))
    #
    #         site_mpo_envs = get_site_mpo_envs(gk_idx)
    #
    #         ## tdvp of branch + spine. TDVP does not change exponent of the ket.
    #         b_mpos = []
    #         for mpo in mpo_list:
    #             mpo_branch = mpo._attach_spine_tens_to_branch(gk_idx, is_mps=False)
    #             # print('mpo branch exponent', gk_idx, mpo_branch.exponent, gtn[gk_idx].exponent)
    #             b_mpos += [mpo_branch]
    #
    #         if solver_type is LocalSolverType.TDVP:
    #             tdvp_solver = TDVPSolver(dist_mpx, operators=b_mpos, te_order=te_order,
    #                                      norm_env0_Ls=site_mpo_envs, compress_config=compress_config)
    #             print('l2r')
    #             tdvp_solver.take_time_step_l2r(dt / 2, do_adapt=do_adapt, canonize=True, build_envs=True)
    #             ## ^ i think canonize can be false
    #             ## now canon at site L-1
    #             tdvp_solver.canonize(0, cur_orthog=dist_mpx.L)
    #         elif solver_type is LocalSolverType.TDDMRG:
    #             ind = gtn.spine.bond(gk_idx, gk_idx + 1)
    #             ind_size = gtn.spine[gk_idx].ind_size(ind)
    #             left_env = qtn.Tensor(np.eye(ind_size), inds=(ind, ind + '_'))
    #             tdvp_solver = TDMRGSolver(dist_mpx, operators=b_mpos, te_order=te_order, ket_env0_L=left_env,
    #                                       norm_env0_Ls=site_mpo_envs, compress_config=compress_config)
    #             tdvp_solver.take_time_step_l2r(dt / 2, do_adapt=do_adapt, canonize=True, build_envs=True)
    #             tdvp_solver.canonize(0, cur_orthog=dist_mpx.L)
    #         else:
    #             raise NotImplementedError
    #         ## now canon at site 0
    #
    #         #### update comb tensors / spine
    #         gtn._update_spine_tens_and_branch(gk_idx, tdvp_solver.ket)
    #
    #         # print('comb orthog? should be', gk_idx)
    #         # print('spine', helper.check_orthog(gtn.spine))
    #         # for i in range(gtn.grid.ngrids):
    #         #     print('branch', i, helper.check_right_orthog(gtn[i].data,
    #         #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))
    #
    #         #### update envs with new branch
    #         for mpo_ind in range(num_mpos):
    #             mpo = mpo_list[mpo_ind]
    #             left_envs, right_envs, branch_envs = mpo_envs[mpo_ind]
    #             # print('update expec', gk_idx, mpo[gk_idx].exponent, gtn[gk_idx].exponent)
    #             expec = gtn.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=True)
    #             ### does not include global exponents
    #             # expec.modify(apply=lambda x: x * 10**(-(gtn.exponent * 2 + mpo.exponent)))
    #             expec.drop_tags()
    #             branch_envs[gk_idx] = expec
    #
    #
    #         #### update spine
    #         if gk_idx > 0:
    #
    #             ## if already updated
    #             # print('gk_idx', gk_idx, gtn.grid.ngrids - 1, spine_adapt)
    #             if gk_idx == 0:  # and spine_adapt:
    #                 continue
    #
    #             print('UPDATE SPINE (RL)', gk_idx)
    #             spine_bond_size = gtn.spine.bond_size(gk_idx, gk_idx + 1)
    #             max_bond = np.inf if compress_config is None else compress_config.max_bonds[1]
    #             spine_adapt = do_adapt and spine_bond_size < max_bond
    #
    #             nsites = 2 if spine_adapt else 1
    #             left_gk_idx = gk_idx - nsites + 1
    #             if is_TE:
    #                 gtn._spine_tdvp(dt / 2, left_gk_idx, nsites, mpo_list, mpo_envs, sweep_direction=-1,
    #                                 compress_config=compress_config, local_solver=spine_solver_type)
    #             else:
    #                 raise NotImplementedError
    #
    #     return gtn

    def _branch_local_TE(self, dt, gk_idx, mpo_list: Sequence['GridTN'],
                         mpo_envs: Sequence[Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']]],
                         sweep_direction=1, te_order=4, do_adapt=False, inplace=True,
                         compress_config: CompressionConfiguration = None,
                         solver_type=LocalSolverType.TDDMRG, **solver_kwargs
                         ):

        gtn = self if inplace else self.copy()
        gtn.collect_exponents()
        # gtn.distribute_exponents()

        # print('branch local TE', gk_idx)
        # self.check_orthog()
        # print('done check', self.get_branch(gk_idx).canon_site)

        ## new mpx object
        dist_mpx = gtn._attach_spine_tens_to_branch(gk_idx, is_mps=True)
        # print('dist mpx norm', helper.norm(dist_mpx))
        # print('dist mpx check orthog', helper.check_orthog(dist_mpx))
        num_mpos = len(mpo_list)

        def get_site_mpo_envs(gk_idx_):
            # print('get site mpo envs', gk_idx)
            site_mpo_envs = []
            for mpo_ind in range(len(mpo_list)):
                # print(mpo_envs[mpo_ind])
                left_envs, right_envs, branch_envs = mpo_envs[mpo_ind]
                left_env = left_envs.get(gk_idx_, None)
                right_env = right_envs.get(gk_idx_, None)
                # print('left env', left_env, left_env.norm() if left_env is not None else None)
                # print('right env', right_env, right_env.norm() if right_env is not None else None)
                if left_env is not None and right_env is not None:
                    # site_mpo_envs += [qtn.tensor_contract(left_env, right_env)]
                    site_mpo_envs += [qtn.TensorNetwork([left_env, right_env])]
                elif left_env is not None:
                    # site_mpo_envs += [left_env.copy()]
                    site_mpo_envs += [qtn.TensorNetwork([left_env])]
                elif right_env is not None:
                    # site_mpo_envs += [right_env.copy()]
                    site_mpo_envs += [qtn.TensorNetwork([right_env])]
                # print('site mpo envs', site_mpo_envs[-1].norm())
                # print('modify with exponent', mpo_list[mpo_ind].exponent, gtn.exponent)
                # site_mpo_envs[-1].modify(apply=lambda x: x * 10**mpo_list[mpo_ind].exponent)
                site_mpo_envs[-1].exponent = mpo_list[mpo_ind].exponent

            return site_mpo_envs

        site_mpo_envs = get_site_mpo_envs(gk_idx)

        ## tdvp of branch + spine. TDVP does not change exponent of the ket.
        b_mpos = []
        for mpo in mpo_list:
            mpo_branch = mpo._attach_spine_tens_to_branch(gk_idx, is_mps=False)
            # print('mpo branch exponent', gk_idx, mpo_branch.exponent, gtn[gk_idx].exponent)
            b_mpos += [mpo_branch]

        if solver_type is LocalSolverType.TDVP:
            tdvp_solver = TDVPSolver(dist_mpx, operators=b_mpos, te_order=te_order,
                                     norm_env0_Ls=site_mpo_envs, compress_config=compress_config,
                                     **solver_kwargs)
            # tdvp_solver.take_time_step_l2r(dt / 2, do_adapt=do_adapt, canonize=True, build_envs=True)
            # ## ^ i think canonize can be false
            # ## now canon at site L-1
            # tdvp_solver.canonize(0, cur_orthog=dist_mpx.L)
        elif solver_type is LocalSolverType.TDDMRG:
            ## relevant solver kwargs:  ket_env0_L, time_evolve_edge
            # print('TDDMRG site mpo envs', site_mpo_envs)
            tdvp_solver = TDMRGSolver(dist_mpx, operators=b_mpos, te_order=te_order,
                                      norm_env0_Ls=site_mpo_envs, compress_config=compress_config,
                                      **solver_kwargs)
        else:
            raise NotImplementedError

        if sweep_direction > 0:
            # print('sweep r2l')
            tdvp_solver.take_time_step_r2l(dt, do_adapt=do_adapt, canonize=True, build_envs=True)
        elif sweep_direction < 0:
            # print('sweep l2r')
            tdvp_solver.take_time_step_l2r(dt, do_adapt=do_adapt, canonize=True, build_envs=True)
            tdvp_solver.canonize(0, cur_orthog=dist_mpx.L - 1)
            tdvp_solver._build_all_envs_right(1, canonize=False)  ## these aren't used but should be
        else:
            if isinstance(tdvp_solver, TDMRGSolver):
                tdvp_solver.backprop_edge = False
            tdvp_solver.take_time_step_l2r(dt / 2, do_adapt=do_adapt, canonize=True, build_envs=True)
            if isinstance(tdvp_solver, TDMRGSolver):
                tdvp_solver.backprop_edge = solver_kwargs.get('backprop_edge', False)
            tdvp_solver.take_time_step_r2l(dt / 2, do_adapt=do_adapt, canonize=False, build_envs=False)
        ## now canon at site 0

        #### update comb tensors / spine
        gtn._update_spine_tens_and_branch(gk_idx, tdvp_solver.ket)
        # tens0 = tdvp_solver.ket.select(gtn.spine.site_tag_id.format(gk_idx))
        # tens0.add_tag(tdvp_solver.ket.site_tag_id.format(0))
        # print('check TE update ket orthog')
        # print(tdvp_solver.ket)
        # helper.check_orthog(tdvp_solver.ket)

        # print('comb orthog? should be', gk_idx)
        # print('spine', helper.check_orthog(gtn.spine))
        # for i in range(gtn.grid.ngrids):
        #     print('branch', i, helper.check_right_orthog(gtn[i].data,
        #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))

        #### update envs with new branch
        for mpo_ind in range(num_mpos):
            mpo = mpo_list[mpo_ind]
            left_envs, right_envs, branch_envs = mpo_envs[mpo_ind]
            expec = gtn.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=True)
            # expec = tdvp_solver.right_envs[mpo_ind][0]
            ### does not include global exponents
            expec.drop_tags()
            branch_envs[gk_idx] = expec

        return tdvp_solver

    def _spine_get_proj_op(gtn, mpo, mpo_env, gk_idx_, nsites_, direction=1):
        """ define projected operator for spine
            direction only relevant for nsites_ = 0
        """
        # mpo = mpo_list[mpo_ind_]
        left_envs_, right_envs_, branch_envs_ = mpo_env  # mpo_envs[mpo_ind_]

        if nsites_ > 0:
            left_env = left_envs_.get(gk_idx_, None)
            right_env = right_envs_.get(gk_idx_ + nsites_ - 1, None)
            # print('branch evns', gk_idx_, nsites_, [branch_envs_.get(gk_idx_ + ns) for ns in range(nsites_)])
            branch_env = qtn.TensorNetwork([branch_envs_.get(gk_idx_ + ns) for ns in range(nsites_)])
            ## envs should not have exponent contributions (mpo exponent included in method)
            proj_op, bonds_i, bonds_o = gtn.project_op_site(mpo, gk_idx_, nsites=nsites_, left_env=left_env,
                                                            right_env=right_env, branch_env=branch_env)

        else:
            left_env = left_envs_.get(gk_idx_, None)
            right_env = right_envs_.get(gk_idx_, None)
            branch_env = branch_envs_.get(gk_idx_)
            ## envs should not have exponent contributions (mpo exponent included in method)

            ## assume spine already in USVT form
            proj_op, bonds_i, bonds_o = gtn.project_op_bond(mpo, gk_idx_, left_env=left_env,
                                                            right_env=right_env, branch_env=branch_env,
                                                            direction=direction)

        return proj_op, bonds_i, bonds_o

    def _spine_get_ovlp_vec(gtn, target, target_env, gk_idx_, nsites_):
        """ define projected operator for spine
        """
        # mpo = mpo_list[mpo_ind_]
        left_envs_, right_envs_, branch_envs_ = target_env  # mpo_envs[mpo_ind_]

        if nsites_ > 0:
            left_env = left_envs_.get(gk_idx_, None)
            right_env = right_envs_.get(gk_idx_ + nsites_ - 1, None)
            # print('branch evns', gk_idx_, nsites_, [branch_envs_.get(gk_idx_ + ns) for ns in range(nsites_)])
            branch_env = qtn.TensorNetwork([branch_envs_.get(gk_idx_ + ns) for ns in range(nsites_)])
            ## envs should not have exponent contributions (mpo exponent included in method)
            proj_vec, bonds_i = gtn.project_vec_site(target, gk_idx_, nsites=nsites_, left_env=left_env,
                                                     right_env=right_env, branch_env=branch_env)

        else:
            left_env = left_envs_.get(gk_idx_, None)
            right_env = right_envs_.get(gk_idx_, None)
            branch_env = branch_envs_.get(gk_idx_)
            ## envs should not have exponent contributions (mpo exponent included in method)

            ## assume spine already in USVT form
            proj_vec, bonds_i = gtn.project_vec_bond(target, gk_idx_, left_env=left_env,
                                                     right_env=right_env, branch_env=branch_env)

        return proj_vec, bonds_i

    def _update_spine_envs(gtn, gk_idx_, mpo_list, mpo_envs, sweep_direction):
        # print('UPDATE SPINE ENVS')
        num_mpos = len(mpo_list)
        for mpo_ind in range(num_mpos):

            left_envs, right_envs, branch_envs = mpo_envs[mpo_ind]

            if sweep_direction > 0:  ## update left env at gk_idx
                # print('update left env', gk_idx_ + 1)
                left_envs[gk_idx_ + 1] = gtn._contract_branch_expec_spine(gk_idx_, mpo_list[mpo_ind],
                                                                          branch_envs[gk_idx_],
                                                                          left_envs.get(gk_idx_, None),
                                                                          is_sqrt=True)

                # print('remove old right env', gk_idx_ - 1)
                if gk_idx_ > 0:
                    right_envs.pop(gk_idx_)

            else:  ## update right env at gk_idx
                # print('update right env', gk_idx_ - 1)
                right_envs[gk_idx_ - 1] = gtn._contract_branch_expec_spine(gk_idx_, mpo_list[mpo_ind],
                                                                           branch_envs[gk_idx_],
                                                                           right_envs.get(gk_idx_, None),
                                                                           is_sqrt=True)

                # print('remove old left env', gk_idx_ + 1)
                if gk_idx_ > 0:
                    left_envs.pop(gk_idx_)

        return

    def _update_spine_ovlp_envs(gtn, gk_idx_, mps_list, mps_envs, sweep_direction):
        # print('UPDATE SPINE OVLP ENVS')
        num_mps = len(mps_list)
        for ind in range(num_mps):

            left_envs, right_envs, branch_envs = mps_envs[ind]

            if sweep_direction > 0:  ## update left env at gk_idx
                # print('update left env', gk_idx_ + 1)
                left_envs[gk_idx_ + 1] = gtn._contract_branch_expec_spine(gk_idx_, None,
                                                                          branch_envs[gk_idx_],
                                                                          left_envs.get(gk_idx_, None),
                                                                          bra_gtn=mps_list[ind])

                # print('remove old right env', gk_idx_ - 1)
                if gk_idx_ > 0:
                    right_envs.pop(gk_idx_)

            else:  ## update right env at gk_idx
                # print('update right env', gk_idx_ - 1)
                right_envs[gk_idx_ - 1] = gtn._contract_branch_expec_spine(gk_idx_, None,
                                                                           branch_envs[gk_idx_],
                                                                           right_envs.get(gk_idx_, None),
                                                                           bra_gtn=mps_list[ind])

                # print('remove old left env', gk_idx_ + 1)
                if gk_idx_ > 0:
                    left_envs.pop(gk_idx_)

        return

    def _spine_tdvp(self, dt, left_gk_idx, nsites, mpo_list: Sequence['GridTN'],
                    mpo_envs: Sequence[Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']]],
                    sweep_direction: int = 1, te_order=4, inplace=True,
                    forward_prop=False,
                    compress_config: CompressionConfiguration = None,
                    local_solver=LocalSolverType.TDVP, backprop_dt=None, **solver_kwargs):
        """ local time evolution for spine
            solver kwargs:  for DMRG, relevant kwargs are "ket_env0_L(R)" and "backprop_edge"
        """
        verbose = False
        gtn = self if inplace else self.copy()
        num_mpos = len(mpo_list)
        # print('spine tdvp', left_gk_idx, te_order)

        #### update of spine:
        ## if 1 site:  back prop of S in spine
        ## if 2 site:  back prop (0) + forward prop (0, 1) + back prop (1)
        if nsites == 2:
            if verbose:  print('spine update 2 site')

            if sweep_direction > 0:
                gk_idx, next_gk_idx = left_gk_idx, left_gk_idx + 1
            else:
                gk_idx, next_gk_idx = left_gk_idx + 1, left_gk_idx

            if verbose:  print('gk_idx', gk_idx, next_gk_idx)

            ## back 1
            op_list = []
            for mpo_ind in range(num_mpos):
                # proj_op, bonds_i, bonds_o = _get_proj_op(mpo_ind, gk_idx, 1)
                proj_op, bonds_i, bonds_o = gtn._spine_get_proj_op(mpo_list[mpo_ind], mpo_envs[mpo_ind], gk_idx, 1)
                op_list += [proj_op]

            dt_ = dt if backprop_dt is None else backprop_dt
            # print('tdvp back prop', dt_)
            helper_tdvp.site_time_evolution(gtn.spine[gk_idx:gk_idx + 1], -dt_, op_list,
                                            [gtn.spine.site_tag(gk_idx)], bonds_i, bonds_o,
                                            te_order=te_order, inplace=True, compress_opts_dict=compress_config
                                            )
            ## forward 2
            op_list = []
            for mpo_ind in range(num_mpos):
                # proj_op, bonds_i, bonds_o = _get_proj_op(mpo_ind, left_gk_idx, 2)
                proj_op, bonds_i, bonds_o = gtn._spine_get_proj_op(mpo_list[mpo_ind], mpo_envs[mpo_ind], left_gk_idx, 2)
                op_list += [proj_op]

            helper_tdvp.site_time_evolution(gtn.spine[left_gk_idx:left_gk_idx + 2], dt, op_list,
                                            [gtn.spine.site_tag(left_gk_idx + i) for i in range(2)],
                                            bonds_i, bonds_o, inplace=True, te_order=te_order,
                                            compress_direction=sweep_direction, compress_opts_dict=compress_config)

            ## update left env at gk_idx
            # _update_spine_envs(gk_idx)
            gtn._update_spine_envs(gk_idx, mpo_list, mpo_envs, sweep_direction)

            ## back 1
            op_list = []
            for mpo_ind in range(num_mpos):
                # proj_op, bonds_i, bonds_o = _get_proj_op(mpo_ind, next_gk_idx, 1)
                proj_op, bonds_i, bonds_o = gtn._spine_get_proj_op(mpo_list[mpo_ind], mpo_envs[mpo_ind], next_gk_idx, 1)
                op_list += [proj_op]

            helper_tdvp.site_time_evolution(gtn.spine[next_gk_idx:next_gk_idx + 1], -dt, op_list,
                                            [gtn.spine.site_tag(next_gk_idx)], bonds_i, bonds_o,
                                            te_order=te_order, inplace=True, compress_opts_dict=compress_config
                                            )

        else:
            if verbose: print('update spine bond (1 site)', left_gk_idx)
            gk_idx = left_gk_idx
            next_gk_idx = gk_idx + 1 * sweep_direction
            if verbose:  print('gk idx', gk_idx, 'next', next_gk_idx)

            ## back 1
            # print('check spine orthog', gk_idx)
            # print(helper.check_orthog(gtn.spine))

            spine_usvt = MPS_USVT.from_MPS(gtn.spine, canon_site=gk_idx, cur_orthog=gk_idx, direction=sweep_direction)
            gtn._spine = spine_usvt
            # print('spine usvt', spine_usvt)

            op_list = []
            for mpo_ind in range(num_mpos):
                # proj_op, bonds_i, bonds_o = _get_proj_op(mpo_ind, gk_idx, 0)
                proj_op, bonds_i, bonds_o = gtn._spine_get_proj_op(mpo_list[mpo_ind], mpo_envs[mpo_ind], gk_idx, 0,
                                                                   direction=sweep_direction)
                op_list += [proj_op]

            S_tens = spine_usvt.get_S_tensor()
            if local_solver is LocalSolverType.TDVP:
                helper_tdvp.site_time_evolution(qtn.TensorNetwork([S_tens], virtual=True), -dt, op_list,
                                                [next(iter(S_tens.tags))], bonds_i, bonds_o, inplace=True,
                                                compress_direction=sweep_direction,
                                                compress_opts_dict=compress_config)
            elif local_solver is LocalSolverType.TDDMRG:
                raise NotImplementedError
            else:
                raise ValueError

            new_spine = spine_usvt.to_MPS(inplace=True)

            # print('gtn spine', gtn.spine)
            # print('gtn branches', gtn.branches)
            helper.canonize(gtn.spine, cur_orthog=gk_idx, i=gk_idx + 1 * sweep_direction, scale=False)

            # _update_spine_envs(left_gk_idx)
            gtn._update_spine_envs(gk_idx, mpo_list, mpo_envs, sweep_direction)

        ## forward prop of next site if forward_prop is True
        if verbose:  print('next spine forward prop?', forward_prop)
        if forward_prop:
            if verbose:  print('forward prop of site', next_gk_idx)
            op_list = []
            for mpo_ind in range(num_mpos):
                # proj_op, bonds_i, bonds_o = _get_proj_op(mpo_ind, next_gk_idx, 1)
                proj_op, bonds_i, bonds_o = gtn._spine_get_proj_op(mpo_list[mpo_ind], mpo_envs[mpo_ind], next_gk_idx, 1)
                op_list += [proj_op]

            helper_tdvp.site_time_evolution(gtn.spine[next_gk_idx:next_gk_idx + 1], dt, op_list,
                                            [gtn.spine.site_tag(next_gk_idx)], bonds_i, bonds_o,
                                            te_order=te_order, inplace=True, compress_opts_dict=compress_config
                                            )

        return gtn.spine, mpo_envs

    def _spine_tdmrg(self, dt, left_gk_idx, nsites, mpo_list: Sequence['GridTN'],
                     mpo_envs: Sequence[Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']]],
                     ovlp_projector: qtn.Tensor,
                     sweep_direction: int = 1, te_order=4, inplace=True,
                     compress_config: CompressionConfiguration = None):
        """ local time evolution for spine
            solver kwargs:  for DMRG, relevant kwargs are "ket_env0_L(R)" and "backprop_edge"
        """
        verbose = False
        gtn = self if inplace else self.copy()
        num_mpos = len(mpo_list)
        # print('spine tdvp', left_gk_idx)

        #### update of spine:
        ## if 1 site:  back prop of S in spine
        ## if 2 site:  back prop (0) + forward prop (0, 1) + back prop (1)
        if nsites == 2:
            if verbose:  print('spine update 2 site')

            if sweep_direction > 0:
                gk_idx, next_gk_idx = left_gk_idx, left_gk_idx + 1
            else:
                gk_idx, next_gk_idx = left_gk_idx + 1, left_gk_idx

            if verbose:  print('gk_idx', gk_idx, next_gk_idx)

            ## assumes gk_idx site was already backpropagated in time

            # print('spine', helper.check_orthog(gtn.spine))
            # for i in range(gtn.grid.ngrids):
            #     print('branch', i, helper.check_right_orthog(gtn[i].data,
            #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))

            helper.canonize(gtn.spine, cur_orthog=gk_idx, i=next_gk_idx, scale=False)
            # print('CHECK ORTHOG AROUND', next_gk_idx)
            # print('spine', helper.check_orthog(gtn.spine))
            # for i in range(gtn.grid.ngrids):
            #     print('branch', i, helper.check_right_orthog(gtn[i].data,
            #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))

            ## forward 2
            op_list = []
            for mpo_ind in range(num_mpos):
                proj_op, bonds_i, bonds_o = gtn._spine_get_proj_op(mpo_list[mpo_ind], mpo_envs[mpo_ind], left_gk_idx, 2)
                op_list += [proj_op]

            if ovlp_projector is None:  # no env (end of chain without ancilla)
                ovlp_projector_site = gtn.spine[gk_idx].copy()
            else:
                ovlp_projector_site = gtn.spine[gk_idx].copy()
                # print('ovlp projector', ovlp_projector.inds)
                # print('bonds_o', bonds_o)
                # print('bonds_i', bonds_i)
                if isinstance(ovlp_projector, qtn.Tensor):
                    ind_bra = next(iter(set(ovlp_projector.inds).intersection(set(bonds_o))))
                    ind_ket = next(iter(set(ovlp_projector.inds).intersection(set(bonds_i))))
                    reindex_dict = {ind_ket: ind_bra}
                elif isinstance(ovlp_projector, qtn.TensorNetwork):
                    reindex_dict = {}
                    for tens in ovlp_projector:
                        ind_bra = next(iter(set(tens.inds).intersection(set(bonds_o))))
                        ind_ket = next(iter(set(tens.inds).intersection(set(bonds_i))))
                        reindex_dict[ind_ket] = ind_bra
                ovlp_projector_site.reindex(reindex_dict, inplace=True)

            old_M = gtn.spine[next_gk_idx].copy()  ## this is orthogonality center
            helper_tdvp.site_time_evolution(gtn.spine[left_gk_idx:left_gk_idx + 2], dt, op_list,
                                            [gtn.spine.site_tag(left_gk_idx + i) for i in range(2)],
                                            bonds_i, bonds_o, inplace=True, te_order=te_order,
                                            compress_direction=sweep_direction, compress_opts_dict=compress_config)
            # print('done site time evolution', sweep_direction)

            # print('CHECK ORTHOG AROUND', next_gk_idx)
            # print('spine', helper.check_orthog(gtn.spine))
            # for i in range(gtn.grid.ngrids):
            #     print('branch', i, helper.check_right_orthog(gtn[i].data,
            #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))

            ## backprop next_gk_idx (use old site)
            ### update old M onto new t+dt basis
            ## bra, ket have different site_ind_ids
            # ovlp_projector_site.reindex({self.ket.site_ind(site0): self.bra.site_ind(site0)}, inplace=True)
            # print('ovlp projector', ovlp_projector)
            # print('ovlp projector site', ovlp_projector_site)
            # print('self.bra', site0, self.bra[site0])
            # exit()
            bra_tens0 = gtn.spine[gk_idx].conj()
            ind_open = gtn.spine.bond(left_gk_idx, left_gk_idx + 1)
            bra_tens0.reindex({**{bi: bo for bi, bo in zip(bonds_i, bonds_o)},
                               **{ind_open: ind_open + '_'}}, inplace=True)
            # print('ovlp proj site', ovlp_projector_site)
            # print('bra tens0', bra_tens0)
            ovlp_projector = qtn.tensor_contract(ovlp_projector_site, bra_tens0)  ## TE'd bra
            new_M = qtn.tensor_contract(ovlp_projector, old_M)
            # print('old M', old_M)
            # print('new M', new_M)

            current_M = gtn.spine[next_gk_idx]
            # print('current M', current_M)
            new_M.transpose_like(current_M, inplace=True)
            current_M.modify(data=new_M.data)

            gtn._update_spine_envs(gk_idx, mpo_list, mpo_envs, sweep_direction)

        else:
            raise NotImplementedError

        return gtn.spine, mpo_envs, ovlp_projector

    def _spine_solve(self, left_gk_idx, nsites, target_list: Sequence['GridTN'], mpo_list: Sequence['GridTN'],
                     target_envs: Sequence[Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']]],
                     mpo_envs: Sequence[Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']]],
                     sweep_direction: int = 1, inplace=True, compress_config: CompressionConfiguration = None,
                     **solver_kwargs):
        """ local time evolution for spine
            solver kwargs:  for DMRG, relevant kwargs are "ket_env0_L(R)" and "backprop_edge"
        """
        gtn = self if inplace else self.copy()
        num_mpos = len(mpo_list)
        num_targets = len(target_list)
        max_bond = compress_config.max_bonds[1]
        # print('spine tdvp', left_gk_idx)

        #### update of spine:
        ## if 1 site:  back prop of S in spine
        ## if 2 site:  back prop (0) + forward prop (0, 1) + back prop (1)
        if nsites == 2:
            print('spine update 2 site')

        A_effs = []
        for mpo_ind in range(num_mpos):
            # proj_op, bonds_i, bonds_o = _get_proj_op(mpo_ind, gk_idx, 1)
            proj_op, bonds_i, bonds_o = gtn._spine_get_proj_op(mpo_list[mpo_ind], mpo_envs[mpo_ind], left_gk_idx,
                                                               nsites)
            A_effs += [proj_op]
            # proj_op_tens = proj_op.contract_tags(all)
            # proj_op_tens.modify(apply=lambda data: data * 10**proj_op.exponent)
            # A_eff = proj_op_tens if A_eff is None else helper.add_tensors(A_eff, proj_op_tens)

        b_eff: Optional['qtn.Tensor'] = None
        for ind in range(num_targets):
            proj_vec, bonds_o = gtn._spine_get_ovlp_target(target_list[ind], target_envs[ind], left_gk_idx, nsites)
            proj_vec_tens = proj_vec.contract_tags(all)
            proj_vec_tens.modify(apply=lambda data: data * 10 ** proj_vec.exponent)
            b_eff = proj_vec_tens if b_eff is None else helper.add_tensors(b_eff, proj_vec_tens)

        bra_inds = b_eff.inds if b_eff is not None else []
        ket_inds = [ind + '_' for ind in bra_inds]
        bra_to_ket_inds = {b_ind: k_ind for b_ind, k_ind in zip(bra_inds, ket_inds)}
        # ket_inds = [bra_to_ket_inds[ind] for ind in bra_inds]

        ### SolveMethod.CGD
        ket_tensors = gtn.spine[left_gk_idx:left_gk_idx + 1]

        x_tens = qtn.tensor_contract(*ket_tensors)
        x_tens.transpose(*ket_inds, inplace=True)
        # x_tens.modify(apply=lambda x: x * 0.)

        is_H = False
        if is_H:
            x_eff, error = helper_dmrg.qtn_conjugate_gradient_descent_1site(A_effs, b_eff, x_tens, bra_to_ket_inds)
        else:
            x_eff, error = helper_dmrg.qtn_conjugate_gradient_squared_1site(A_effs, b_eff, x_tens, bra_to_ket_inds)
        # x_eff, error = qtn_conjugate_gradient_descent_1site(A_eff, b_eff, None, bra_to_ket_inds)
        # x_eff.modify(apply=lambda x: x * 10 ** (-A_eff_exponent))

        ## update spine with x_eff
        if nsites == 2:
            gk_idx_0 = left_gk_idx if sweep_direction > 0 else left_gk_idx + 1
            gk_idx_1 = left_gk_idx + 1 if sweep_direction > 0 else left_gk_idx

            split_inds, left_inds = gtn.spine[gk_idx_0].filter_bonds(gtn.spine[gk_idx_1])  # shared, not shared
            site1, site2 = x_eff.split(left_inds, absorb='right', bond_ind=split_inds[0], max_bond=max_bond,
                                       cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)

            ## update ket
            gtn.spine[gk_idx_0].modify(data=site1.data, inds=site1.inds)
            gtn.spine[gk_idx_1].modify(data=site2.data, inds=site2.inds)

        elif nsites == 1:
            gk_idx_0 = left_gk_idx
            x_eff.transpose_like(gtn.spine[left_gk_idx], inplace=True)
            gtn.spine[left_gk_idx].modify(data=x_eff.data)
            helper.canonize(gtn.spine, cur_orthog=left_gk_idx, i=left_gk_idx + 1 * sweep_direction, scale=False)
        else:
            raise NotImplementedError

        # _update_spine_envs(left_gk_idx)
        gtn._update_spine_envs(gk_idx_0, mpo_list, mpo_envs, sweep_direction)
        gtn._update_spine_ovlp_envs(gk_idx_0, target_list, target_envs, sweep_direction)

        return gtn.spine, error

    # def evolve_local_v2(self, dt, mpo_list: Sequence['GridTN1DComb'], te_order=0,
    #                     do_adapt=True, inplace=False, sweep_direction=1,
    #                     solver_type=LocalSolverType.TDDMRG, spine_solver_type=LocalSolverType.TDVP,
    #                     compress_config: 'CompressionConfiguration'=None, expand_basis=None):
    #
    #     gtn = self if inplace else self.copy()
    #     gtn.collect_exponents()
    #     gtn_copy = gtn.copy()
    #     num_gr = gtn.grid.ngrids
    #
    #     mpo_list = [mpo.copy() for mpo in mpo_list]
    #     for mpo in mpo_list:
    #         mpo.collect_exponents()
    #         mpo.distribute_sign()
    #     num_mpos = len(mpo_list)
    #
    #     if solver_type is LocalSolverType.TDDMRG:
    #         BranchLocalSolver = TDMRGSolver
    #         is_TE = True
    #     elif solver_type is LocalSolverType.TDVP:
    #         BranchLocalSolver = TDVPSolver
    #         is_TE = True
    #     elif solver_type is LocalSolverType.DMRG:
    #         BranchLocalSolver = DMRGSolver
    #         is_TE = False
    #     elif solver_type is LocalSolverType.LINSOLVE:
    #         BranchLocalSolver = LinearSolver
    #         is_TE = False
    #     else:
    #         raise NotImplementedError
    #
    #     # tot_expec = 0
    #     # for mpo in mpo_list:
    #     #     tot_expec += gtn.meas_expec(mpo, is_sqrt=True)
    #     # print('INIT MEAS EXPEC TOT', tot_expec)
    #
    #     if sweep_direction > 0:
    #         gtn.canonicalize_around_i(0, scale=False)
    #         b = gtn.get_branch(0)
    #         if b is not None:
    #             b.canonize(i=0, inplace=True, scale=False)
    #             helper.canonize_tens_list(b.get_anchor_tens(), gtn.spine[0], inplace=True)
    #     else:
    #         gtn.canonicalize_around_i(num_gr-1, scale=False)
    #         b = gtn.get_branch(num_gr-1)
    #         if b is not None:
    #             b.canonize(i=0, inplace=True, scale=False)
    #             helper.canonize_tens_list(b.get_anchor_tens(), gtn.spine[num_gr-1], inplace=True)
    #
    #     ## build branch envs
    #     mpo_envs = []
    #     spine_adapt = False
    #     for mpo in mpo_list:
    #
    #         if mpo.spine.site_ind_id == gtn.spine.site_ind_id:
    #             mpo.spine_ind_id = mpo.spine_ind_id + '_op_'
    #
    #         left_envs, right_envs, branch_envs = {}, {}, {}
    #         for gk_idx in range(gtn.grid.ndim-1, 0, -1):
    #             # print('branch exponents', gtn[gk_idx].exponent, mpo[gk_idx].exponent)
    #             expec = gtn.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=False)
    #             ### does not include any global exponents
    #             expec.drop_tags()
    #             # print('branch expec norm', gk_idx, expec.norm())
    #             branch_envs[gk_idx] = expec
    #             right_envs[gk_idx-1] = gtn._contract_branch_expec_spine(gk_idx, mpo, expec,
    #                                                                     right_envs.get(gk_idx, None),
    #                                                                     is_sqrt=True)
    #             # print('right env norm', gk_idx, right_envs[gk_idx-1].norm())
    #
    #         mpo_envs += [(left_envs, right_envs, branch_envs)]
    #
    #     gk_inds = range(gtn.grid.ngrids) if sweep_direction > 0 else range(gtn.grid.ngrids-1,-1,-1)
    #     for gk_idx in gk_inds:
    #
    #         # print('CHECK ORTHOG AROUND', gk_idx)
    #         # print('spine', helper.check_orthog(gtn.spine))
    #         # for i in range(gtn.grid.ngrids):
    #         #     print('branch', i, helper.check_right_orthog(gtn[i].data,
    #         #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))
    #
    #         # tt_ind = gtn.spine_ind_id.format(gk_idx)
    #         branch = gtn.get_branch(gk_idx)
    #
    #         ## new mpx object
    #         dist_mpx = gtn._attach_spine_tens_to_branch(gk_idx, is_mps=True)
    #         # print('dist mpx norm', helper.norm(dist_mpx))
    #         # print('dist mpx check orthog', helper.check_orthog(dist_mpx))
    #         # dist_mpx_copy = dist_mpx.copy()
    #
    #         site_mpo_envs = []
    #         for mpo_ind in range(len(mpo_list)):
    #             # print(mpo_envs[mpo_ind])
    #             left_envs, right_envs, branch_envs = mpo_envs[mpo_ind]
    #             left_env = left_envs.get(gk_idx, None)
    #             right_env = right_envs.get(gk_idx, None)
    #             if left_env is not None and right_env is not None:
    #                 site_mpo_envs += [qtn.tensor_contract(left_env, right_env)]
    #             elif left_env is not None:
    #                 site_mpo_envs += [left_env.copy()]
    #             elif right_env is not None:
    #                 site_mpo_envs += [right_env.copy()]
    #             # print('site mpo envs', site_mpo_envs[-1].norm())
    #             site_mpo_envs[-1].modify(apply=lambda x: x * 10**(mpo_list[mpo_ind].exponent))
    #             # print('site mpo envs', site_mpo_envs[-1].norm(), mpo_list[mpo_ind].exponent)
    #
    #         ## tdvp of branch + spine. TDVP does not change exponent of the ket.
    #         b_mpos = []
    #         for mpo in mpo_list:
    #             mpo_branch = mpo._attach_spine_tens_to_branch(gk_idx, is_mps=False)
    #             # print('mpo branch exponent', gk_idx, mpo_branch.exponent, gtn[gk_idx].exponent)
    #             b_mpos += [mpo_branch]
    #
    #         if is_TE:
    #             # tdvp_solver = TDVPSolver(dist_mpx, operators=b_mpos, te_order=te_order,
    #             #                          norm_env0_Ls=site_mpo_envs, compress_config=compress_config)
    #             tdvp_solver = BranchLocalSolver(dist_mpx, operators=b_mpos, te_order=te_order,
    #                                             norm_env0_Ls=site_mpo_envs, compress_config=compress_config)
    #             tdvp_solver.take_time_step(dt, do_adapt=do_adapt)  ##
    #         else:
    #             raise NotImplementedError
    #
    #         #### update comb tensors / spine
    #         gtn._update_spine_tens_and_branch(gk_idx, tdvp_solver.ket)
    #         # for i in range(1, dist_mpx.L):
    #         #     # print('branch data exponents', branch.data.exponent, dist_mpx.exponent)
    #         #     b_tens = branch.data[i-1]
    #         #     new_b_tens = tdvp_solver.ket[i]
    #         #     new_b_tens.transpose_like(b_tens, inplace=True)
    #         #     b_tens.modify(data=new_b_tens.data)
    #         #
    #         # new_spine_tens = tdvp_solver.ket[0]
    #         # new_spine_tens.drop_tags((dist_mpx.site_tag_id.format(0),))
    #         # new_spine_tens.transpose_like_(gtn.spine[gk_idx])
    #         # gtn.spine[gk_idx].modify(data = new_spine_tens.data)
    #
    #         # print('dist gtn', gk_idx, gtn.distance(gtn_copy))
    #
    #         # print('comb orthog? should be', gk_idx)
    #         # print('spine', helper.check_orthog(gtn.spine))
    #         # for i in range(gtn.grid.ngrids):
    #         #     print('branch', i, helper.check_right_orthog(gtn[i].data,
    #         #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))
    #
    #         #### update envs with new branch
    #         for mpo_ind in range(num_mpos):
    #             mpo = mpo_list[mpo_ind]
    #             left_envs, right_envs, branch_envs = mpo_envs[mpo_ind]
    #             expec = gtn.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=True)
    #             ### does not include global exponents
    #             # expec.modify(apply=lambda x: x * 10**(-(gtn.exponent * 2 + mpo.exponent)))
    #             expec.drop_tags()
    #             branch_envs[gk_idx] = expec
    #
    #
    #         #### update of spine:
    #         is_not_end = gk_idx < gtn.grid.ngrids - 1 if sweep_direction > 0 else gk_idx > 0
    #         if is_not_end:
    #
    #             ## if already updated
    #             # print('gk_idx', gk_idx, gtn.grid.ngrids - 1, spine_adapt)
    #             # if gk_idx == gtn.grid.ngrids - 1:  # and spine_adapt:
    #             #     continue
    #
    #             # print('UPDATE SPINE', gk_idx)
    #             spine_bond_size = gtn.spine.bond_size(gk_idx, gk_idx + 1)
    #             max_bond = np.inf if compress_config is None else compress_config.max_bonds[1]
    #             # if max_bond is not None:
    #             #     max_bond = min(max_bond, 2 ** (gk_idx + 1), 2 ** (self.grid.ndims - gk_idx - 1))
    #             spine_adapt = do_adapt and spine_bond_size < max_bond
    #
    #             nsites = 2 if spine_adapt else 1
    #             # print('spine tdvp left site', gk_idx, nsites)
    #             if is_TE:
    #                 gtn._spine_tdvp(dt, gk_idx, nsites, mpo_list, mpo_envs, sweep_direction=sweep_direction,
    #                                 compress_config=compress_config, local_solver=spine_solver_type)
    #             else:
    #                 raise NotImplementedError
    #
    #
    #     return gtn

    def _branch_local_solver(self, gk_idx, target_list: Sequence['GridTN'], mpo_list: Sequence['GridTN'],
                             target_envs: Sequence[Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']]],
                             mpo_envs: Sequence[Sequence[Union['qtn.Tensor', 'qtn.TensorNetwork']]],
                             sweep_direction=1, do_adapt=False, inplace=True,
                             compress_config: CompressionConfiguration = None,
                             solver_type=LocalSolverType.DMRG, **solver_kwargs
                             ):

        gtn = self if inplace else self.copy()
        max_bond = compress_config.max_bonds[1]
        cutoff = compress_config.cutoffs[1]

        ## new mpx object
        dist_mpx = gtn._attach_spine_tens_to_branch(gk_idx, is_mps=True)
        # print('dist mpx norm', helper.norm(dist_mpx))
        # print('dist mpx check orthog', helper.check_orthog(dist_mpx))
        num_mpos = len(mpo_list)
        num_targets = len(target_list)

        def get_gk_envs(gk_idx_, obj_list, env_list):
            site_envs = []
            for obj_ind in range(len(obj_list)):
                # print(mpo_envs[mpo_ind])
                left_envs, right_envs, branch_envs = env_list[obj_ind]
                left_env = left_envs.get(gk_idx_, None)
                right_env = right_envs.get(gk_idx_, None)
                # print('left env', left_env, left_env.norm() if left_env is not None else None)
                # print('right env', right_env, right_env.norm() if right_env is not None else None)
                if left_env is not None and right_env is not None:
                    site_envs += [qtn.tensor_contract(left_env, right_env)]
                elif left_env is not None:
                    site_envs += [left_env.copy()]
                elif right_env is not None:
                    site_envs += [right_env.copy()]
                # print('site mpo envs', site_mpo_envs[-1].norm())
                # print('modify with exponent', mpo_list[mpo_ind].exponent, gtn.exponent)
                site_envs[-1].modify(apply=lambda x: x * 10 ** obj_list[obj_ind].exponent)
                # print('site mpo envs', site_mpo_envs[-1].norm(), mpo_list[mpo_ind].exponent)

            return site_envs

        ## <x|Ax>, <x|b> envs
        site_mpo_envs = get_gk_envs(gk_idx, mpo_list, mpo_envs)
        site_target_envs = get_gk_envs(gk_idx, target_list, target_envs)

        ## also get <Ax|Ax>, <b|Ax> envs
        raise NotImplementedError

        ## update envs with global exponents
        for ind in range(num_mpos):
            tot_exponent = 2 * gtn.exponent + mpo_list[ind].exponent
            site_mpo_envs[ind].modify(apply=lambda x: x * 10 ** tot_exponent)

        for ind in range(num_targets):
            tot_exponent = gtn.exponent + target_list[ind].exponent
            site_target_envs[ind].modify(apply=lambda x: x * 10 ** tot_exponent)

        ## dmrg of branch + spine. DMRG does not change exponent of the ket.
        b_mpos = []
        for mpo in mpo_list:
            mpo_branch = mpo._attach_spine_tens_to_branch(gk_idx, is_mps=False)
            # print('mpo branch exponent', gk_idx, mpo_branch.exponent, gtn[gk_idx].exponent)
            b_mpos += [mpo_branch]

        t_mpss = []
        for target in target_list:
            target_branch = target._attach_spine_tens_to_branch(gk_idx, is_mps=True)
            t_mpss += [target_branch]

        if solver_type is LocalSolverType.DMRG:
            dmrg_solver = LinearSolver(dist_mpx, targets=t_mpss, operators=b_mpos, is_H=False,
                                       norm_env0_Ls=site_mpo_envs,
                                       ovlp_env0_Ls=site_target_envs,
                                       max_bond=max_bond, conv_tol=np.sqrt(cutoff), **solver_kwargs)
        else:
            raise NotImplementedError

        adapt = do_adapt and (max_bond is None or max_bond > dist_mpx.max_bond())
        nsites = 2 if adapt else 1
        if sweep_direction > 0:
            err = dmrg_solver.dmrg_sweep(nsites, direction=SweepDirection.RIGHT, canonize=True, build_envs=True)
        elif sweep_direction < 0:
            err = dmrg_solver.dmrg_sweep(nsites, direction=SweepDirection.LEFT, canonize=True, build_envs=True)
            dmrg_solver.canonize(0, cur_orthog=dist_mpx.L)
            dmrg_solver._build_all_envs_right(1, canonize=False)
        else:
            err = dmrg_solver.dmrg_sweep(nsites, direction=SweepDirection.RIGHT, canonize=True, build_envs=True)
            err = dmrg_solver.dmrg_sweep(nsites, direction=SweepDirection.LEFT, canonize=False, build_envs=False)

        ## now canon at site 0

        #### update comb tensors / spine
        gtn._update_spine_tens_and_branch(gk_idx, dmrg_solver.ket)

        # print('comb orthog? should be', gk_idx)
        # print('spine', helper.check_orthog(gtn.spine))
        # for i in range(gtn.grid.ngrids):
        #     print('branch', i, helper.check_right_orthog(gtn[i].data,
        #                                                  left_ancillas=(gtn.spine_ind_id.format(i),)))

        #### update envs with new branch
        for mpo_ind in range(num_mpos):
            # mpo = mpo_list[mpo_ind]
            left_envs, right_envs, branch_envs = mpo_envs[mpo_ind]
            # print('update expec', gk_idx, mpo[gk_idx].exponent, gtn[gk_idx].exponent)
            # expec = gtn.meas_expec_branch(gk_idx, mpo, is_sqrt=True, exclude_weights=True)
            expec = dmrg_solver.A_envsRs[mpo_ind][0]
            ### remove global exponents
            expec.modify(apply=lambda x: x * 10 ** (-(gtn.exponent * 2 + mpo_list[mpo_ind].exponent)))
            expec.drop_tags()
            branch_envs[gk_idx] = expec

        for ind in range(len(target_list)):
            # target = target_list[ind]
            left_envs, right_envs, branch_envs = target_envs[ind]
            expec = dmrg_solver.b_envsRs[ind][0]
            ### remove global exponents
            expec.modify(apply=lambda x: x * 10 ** (-(gtn.exponent + target_list[ind].exponent)))
            expec.drop_tags()
            branch_envs[gk_idx] = expec

        return dmrg_solver, err
