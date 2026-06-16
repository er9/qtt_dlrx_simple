"""GridsComb: composite grid for the comb (tree-like) QTT layout.

Concrete :class:`~grids_composite.CompositeGrid` that arranges its sub-grids as a comb,
the geometry consumed by :class:`~gridTN_1Dcomb.GridTN1DComb`. Maps states to and from
the comb tensor network and exposes the multidimensional operators on this layout.
"""

from setup_.configs import *

import tt_io
import helper_quimb as helper
from axis import Axis
from grid import Grid
from grids_composite import CompositeGrid
from gridTN_1D import GridTN1D
from gridTN_1Dcomb import GridTN1DComb

""" Group of Grid objects (which are groups of Axis objects) or other Grids objects
"""


class GridsComb(CompositeGrid):

    def __init__(self, gridID, grids: Iter[Grid]):
        """
        Inputs:
            axesID:  ID to use for this GridLayout object
            axes: list of Axis objects. CAN REPEAT THE SAME OBJECT?

        Attributes:
            grids:  list of Grid or Grids objects
            axes:   list of Axis objects associated with Grids
            ngrids: number of grids
            ndim:   number of Axis objects
        """
        self.layout_type = LayoutType.COMB
        super().__init__(gridID, grids)

    ################
    ## common TNs ##
    ################

    def make_empty_gridTN(self, ax_deriv_configs=None) -> 'GridTN1DComb':
        return GridTN1DComb(self, ax_deriv_configs=ax_deriv_configs)

    def make_zero_gridTN(self, max_bond: int = 2, orthog: int = 0,
                         ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None) -> 'GridTN':
        if self.ndim > 2:
            raise NotImplementedError

        raise NotImplementedError


    def make_gridTN(self, data=None, ax_deriv_configs=None) -> 'GridTN1DComb':
        # gtnc = GridTN1DComb(self, ax_deriv_configs=ax_deriv_configs)
        # gtnc.data = data
        gtnc = self.get_ones_mps(site_ind_id=data.site_ind_id)
        gtnc = self.dmrg_to_gtn_format(gtnc, data.copy())
        return gtnc

    def load_gtn_data(self, fstr, gtn=None, ax_deriv_configs=None, safe_pass=False) -> 'GridTN1DComb':
        try:
            print('comb loading data', fstr + '.npz')
            exponent, sign, spine, branch_data = tt_io.comb_from_npz(fstr)
            print('comb loaded data', fstr + '.npz')
        except IOError:
            print('data not found', fstr + '.npz')
            if not safe_pass:
                raise IOError
            return None

        if gtn is None:
            gtn = self.make_empty_gridTN(ax_deriv_configs=ax_deriv_configs)

        gtn._exponent = exponent
        gtn._sign = sign
        branches = {}
        assert (len(branch_data) == self.ngrids), 'num grids from loaded data incorrect'
        for i in range(self.ngrids):
            gr = self.grids[i]
            if branch_data[i] is not None:
                # print('\nbranch data', branch_data[i])
                branches[gr] = gr.make_gridTN(branch_data[i], ax_deriv_configs=gtn.ax_deriv_configs)
                # print('branch data gr', branches[gr])
        # gtn.spine = spine
        # gtn.branches = branches
        gtn.data = (branches, spine)
        print('loaded data', fstr + '.npz')

        # print('loaded branches', branches)
        # print('loaded spine', spine)
        # print('gtn', gtn)

        return gtn

    def get_ones_mps(self, site_ind_id='i({})', site_tag_id='X({})') -> GridTN1DComb:
        """ gtn_cls: GridTN class type
            build ones vector mps
        """
        mps_dict = {gr: gr.get_ones_mps(site_ind_id, site_tag_id) for gr in self.grids}
        return GridTN1DComb(self, mps_dict)

    def get_iden_mpo(self, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> GridTN1DComb:
        """ build identity mpo
        """
        mpo_dict = {gr: gr.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id) for gr in self.grids}
        return GridTN1DComb(self, mpo_dict)

    # def get_select_elems_mps(self, inds: Iter[int], site_ind_id='i({})', site_tag_id='X({})') -> GridTN1DComb:
    #     """ build MPO to select certain elements specified by inds
    #     """
    #     assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
    #     mps_dict = {gr.gridID: gr.get_select_elems_mps([ix], site_ind_id, site_tag_id)
    #                 for gr, ix in zip(self.grids, inds)}
    #     return GridTN1DComb(self, mps_dict)
    #
    # def get_select_elems_mpo(self, inds: Iter[int], upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') \
    #         -> GridTN1DComb:
    #     """ build MPO to select certain elements specified by inds
    #     """
    #     assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
    #     mpo_dict = {gr.gridID: gr.get_select_elems_mpo([ix], upper_ind_id, lower_ind_id, site_tag_id)
    #                 for gr, ix in zip(self.grids, inds)}
    #     return GridTN1DComb(self, mpo_dict)

    #########################################
    ## convert between np.ndarray and MPX ##
    #########################################

    def map_state_to_mps(self, state: np.ndarray, site_ind_id='i({})', site_tag_id='T({})', direction=0,
                         split_opts=None, ax_deriv_configs=None, ancilla_right=(), ancilla_right_inds=(),
                         ancilla_left=(), ancilla_left_inds=(), axes=None) -> 'GridTN1DComb':
        """ convert state represented as vector converted to mps state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        return GridTN1DComb.from_dense_state(state, self, site_ind_id, site_tag_id, split_opts=split_opts,
                                             ax_deriv_configs=ax_deriv_configs, axes=axes)

    def map_mps_to_state(self, gtn_mps: GridTN1DComb, ax_select=None, ancilla_right=(), ancilla_right_inds=(),
                         ancilla_left=(), ancilla_left_inds=()) -> np.ndarray:
        """ convert MPS into ndim-dim np.ndarray
        """
        return gtn_mps.get_data(ax_select=ax_select)

    def map_operator_to_mpo(self, operator, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='B({})',
                            direction=0, split_opts=None, ancilla_right=(), ancilla_right_inds=(), ancilla_left=(),
                            ancilla_left_inds=()) -> GridTN1DComb:
        """ convert operator represented as high-dimensional matrix to mpo state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        return GridTN1DComb.from_dense_operator(operator, self, upper_ind_id, lower_ind_id, site_tag_id,
                                                split_opts=split_opts)

    def map_mpo_to_operator(self, gtn_mpo, ax_select=None, ancilla_right=(), ancilla_right_inds=(),
                            ancilla_left=(), ancilla_left_inds=()) -> np.ndarray:
        """ convert MPO into ndim*2-dim np.ndarray (o0 o1 ... x i0 i1 ...)
        """
        return gtn_mpo.get_data(ax_select=ax_select)

    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    # def make_ndim(self, mpx_ax_dict):
    #     """ mpx dict is defined with axIDs as keys
    #     """
    #     new_branches = {}
    #     for gr in self.grids:
    #         active_axIDs = [axID for axID in gr.axIDs if axID in mpx_ax_dict]
    #         active_dict = {axID: mpx_ax_dict[axID] for axID in active_axIDs}
    #         if isinstance(active_dict[gr.axIDs[0]], qtn.MatrixProductState):
    #             out_gr = gr.make_mps_ndim(active_dict)
    #         elif isinstance(active_dict[gr.axIDs[0]], qtn.MatrixProductOperator):
    #             out_gr = gr.make_mpo_ndim(active_dict)
    #         else:
    #             # out_gr = gr.make_tn_ndim(active_dict)
    #             raise TypeError('legs of combs should be 1D TNs')
    #         new_branches[gr.gridID] = out_gr
    #
    #     mps_comb = GridTN1DComb(self, branches=new_branches)
    #     return mps_comb

    def make_mps_ndim(self, mps_ax_dict) -> 'GridTN1DComb':
        """ takes dictionary with axIDs as keys to create GridTN1DComb object
            pad empty grids with ones
        """
        new_branches = {}
        active_grids = self.get_active_grids(mps_ax_dict.keys())
        for grid in self.grids:
            if grid in active_grids:
                active_axes = [ax for ax in grid.axes if ax in mps_ax_dict]
                grid_mps_dict = {ax: mps_ax_dict[ax] for ax in active_axes}
                grid_mps = grid.make_mps_ndim(grid_mps_dict)
                new_branches[grid] = grid_mps
            else:
                new_branches[grid] = grid.get_ones_mps()

        mps_comb = GridTN1DComb(self, branches=new_branches)
        return mps_comb

    def make_mpx_ndim(self, mps_ax_dict) -> 'GridTN1DComb':
        """ takes dictionary with axIDs as keys to create GridTN1DComb object
            does not pad remaining axes
        """
        new_branches = {}
        for grid in self.get_active_grids(mps_ax_dict.keys()):
            active_axes = [ax for ax in grid.axes if ax in mps_ax_dict]
            grid_mps_dict = {ax: mps_ax_dict[ax] for ax in active_axes}
            grid_mpx = grid.make_mpx_ndim(grid_mps_dict)
            new_branches[grid] = grid_mpx

        mpo_comb = GridTN1DComb(self, branches=new_branches)
        return mpo_comb

    def make_mpo_ndim(self, mpo_ax_dict) -> 'GridTN1DComb':
        new_branches = {}
        for grid in self.get_active_grids(mpo_ax_dict.keys()):
            active_axes = [ax for ax in grid.axes if ax in mpo_ax_dict]
            grid_mpo_dict = {ax: mpo_ax_dict[ax] for ax in active_axes}
            grid_mpo = grid.make_mpo_ndim(grid_mpo_dict)
            new_branches[grid] = grid_mpo

        mpo_comb = GridTN1DComb(self, branches=new_branches)
        return mpo_comb

    def make_tn1d_ndim(self, tn_1d_dict, in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                       site_tag_id='D({})') -> 'GridTN1DComb':
        new_branches = {}
        for grid in self.get_active_grids(tn_1d_dict.keys()):
            active_axes = [ax for ax in grid.axes if ax in tn_1d_dict]
            grid_tn1_dict = {ax: tn_1d_dict[ax] for ax in active_axes}
            grid_mpo = grid.make_tn1d_ndim(grid_tn1_dict, in1_ind_id=in1_ind_id, in2_ind_id=in2_ind_id,
                                           out_ind_id=out_ind_id, recalc=True, store=False)
            new_branches[grid] = grid_mpo

        mpo_comb = GridTN1DComb(self, branches=new_branches)
        return mpo_comb

    def pad_gtn_to_grid(self, gtn, target_data_type=None):
        if gtn.grid == self:
            return gtn
        elif isinstance(gtn, GridTN1D):
            if gtn.grid in self.grids:
                return GridTN1DComb(self, branches={gtn.grid: gtn})
            else:
                raise NotImplementedError('GridTN1D.grid must be subgrid of GridComb')
        elif isinstance(gtn, GridTN1DComb):
            # assert all([subgr in self.grids for subgr in gtn.grid.grids])
            return gtn.pad_self_to_new_grid(self)
            # if gtn.spine is not None:
            #     mps_inds = [self.get_grid_ind(subgr) for subgr in gtn.grid.grids]
            #     spine = helper.pad_mps(gtn.spine, mps_inds, self.ngrids, pad_ind_size=1)
            # else:
            #     spine = None
            # return GridTN1DComb(self, branches=gtn.branches, spine=spine, exponent=gtn.exponent)
        else:
            raise NotImplementedError

    # def pad_mps_to_grid(self, tn_data, tn_grid) -> 'GridTN1DComb':
    #     """ pad data on some grid onto self
    #         cgrid:  Composite grid on which tn_data exists
    #     """
    #     if tn_grid == self:
    #         return GridTN1DComb(self, *tn_data)     # branches and maybe spine
    #     if tn_grid in self.grids:
    #         return GridTN1DComb(self, branches={tn_grid: tn_data})
    #     if tn_grid in self.axes:
    #         return self.make_mps_ndim({tn_grid: tn_data})
    #     raise TypeError

    # def pad_mpo_to_grid(self, tn_data, tn_grid) -> GridTN1DComb:
    #     """ pad MPO on mpo_grid onto self.grid
    #     """
    #     if tn_grid == self:
    #         return GridTN1DComb(self, *tn_data)  # branches and maybe spine
    #     if tn_grid in self.grids:
    #         return GridTN1DComb(self, branches={tn_grid: tn_data})
    #     if tn_grid in self.axes:
    #         return self.make_mpo_ndim({tn_grid: tn_data})
    #     print(tn_grid, self, self.grids, self.axes)
    #     raise TypeError

    def pad_mps_to_grid(self, scalar_field_mps, mps_axes) -> 'GridTN1DComb':
        """ pad data on some grid onto self
            cgrid:  Composite grid on which tn_data exists
        """
        subgr = self.get_active_grids(mps_axes)

        if len(subgr) > 1:
            raise NotImplementedError
        else:
            branch = subgr[0].pad_mps_to_grid(scalar_field_mps, mps_axes)
            return GridTN1DComb(self, branches={subgr[0]: branch})

    def pad_mpo_to_grid(self, scalar_field_mpo, mpo_axes) -> GridTN1DComb:
        """ pad MPO on mpo_grid onto self.grid
        """
        subgr = self.get_active_grids(mpo_axes)

        if len(subgr) > 1:
            raise NotImplementedError
        else:
            branch = subgr[0].pad_mps_to_grid(scalar_field_mpo, mpo_axes)
            return GridTN1DComb(self, branches={subgr[0]: branch})

    def take_outerproduct(self, *gtns):
        """ take outer product of gtns provided
            assert that the active grids of the gtns are not overlapping
        """
        raise NotImplementedError

    def build_indexed_gtn(self, dict_gtns: dict[any, 'GridTN1DComb'], index_order=None):
        """ builds a gtn from provided gtns in dict_gtns
            adds a tensor that indexes each gtn as specified by the key
                if an MPO, any will be a tuple (out, in)

            note:  add indexing tensor to the first active branch
        """
        if index_order is None:
            index_order = dict_gtns.keys()

        num_inds = len(index_order)
        num_items = len(dict_gtns.keys())
        # print('num inds', num_inds, num_items)
        ind_tens = np.zeros((num_inds, num_inds))

        b_ind = 0
        is_mpo = True
        tot_gtn: 'GridTN1DComb' = None
        it = 0
        for ind, gtn in dict_gtns.items():

            if gtn is None or gtn.data is None:
                num_items -= 1
                continue

            ind_position = list(index_order).index(ind)
            ind_tens[ind_position, it] = 1.0
            is_mpo = is_mpo and gtn.data_type is DataType.MPO

            gtn = gtn.copy()
            if b_ind is None:
                b_ind = gtn.active_grids[0]
            branch = gtn[b_ind]
            if branch is None:
                gr: 'Grid' = b_ind if isinstance(b_ind, Grid) else self.grids[b_ind]
                gtn[b_ind] = gr.get_iden_mpo() if is_mpo else gr.get_ones_mps()
                branch = gtn[b_ind]
                # print('padded gtn gr', gr)
            gtn_tens0 = branch.data[branch.L - 1]
            gtn_tens0.new_ind('ind_anc', 1, axis=0)

            if tot_gtn is None:
                tot_gtn = gtn
            else:
                tot_gtn = tot_gtn.add(gtn, compress=True)
                # print('tot gtn', tot_gtn.exponent, tot_gtn.canon_site)

            it += 1

        ind_tens = ind_tens[:, :num_items]

        ### add index tensor
        if tot_gtn is not None:
            tot_gtn_data = tot_gtn[b_ind].data
            index_tensor_ind = tot_gtn_data.L
            if is_mpo:
                # if index_tensor_ind == 0:
                #     tot_gtn_data = helper.renumber_mpo(tot_gtn.data, list(range(tot_gtn.L)), list(range(1,tot_gtn.L+1)))
                sq_num_inds = int(np.round(np.sqrt(num_inds)))
                # print('tot gtn data u/l inds', tot_gtn_data.upper_ind_id, tot_gtn_data.lower_ind_id)
                ind_tensor = qtn.Tensor(ind_tens.reshape(sq_num_inds, sq_num_inds, -1),
                                        inds=(tot_gtn_data.upper_ind_id.format(index_tensor_ind),
                                              tot_gtn_data.lower_ind_id.format(index_tensor_ind), 'ind_anc'),
                                        tags=(tot_gtn_data.site_tag_id.format(index_tensor_ind),))
            else:
                # if index_tensor_ind == 0:
                #     tot_gtn_data = helper.renumber_mps(tot_gtn.data, list(range(tot_gtn.L)), list(range(1, tot_gtn.L + 1)))
                # print('tot gtn data ind', tot_gtn_data.site_ind_id)
                ind_tensor = qtn.Tensor(ind_tens,
                                        inds=(tot_gtn_data.site_ind_id.format(index_tensor_ind), 'ind_anc'),
                                        tags=(tot_gtn_data.site_tag_id.format(index_tensor_ind),))
                # print('ind tensor', ind_tensor)

            tot_gtn_data.add(ind_tensor)
            tot_gtn_data._L = tot_gtn_data.num_tensors
            # print('tot gtn exponent', tot_gtn.exponent)
            # print('final tot gtn data', tot_gtn_data)

            tot_gtn[b_ind].data = tot_gtn_data
            # print('expanded tot gtn', tot_gtn)

        return tot_gtn

    def select_indexed_gtn(self, gtn: 'GridTN1DComb', select_ind: Union[int, tuple[int]]) -> 'GridTN':
        """ selects a subgtn from gtn
        """
        b_ind = 0  # gtn.active_grids[0]
        L = gtn[b_ind].data.L
        gtn = gtn.copy()
        gtn_b = gtn[b_ind]

        index_ind = L - 1
        index_tensor = gtn_b.data[index_ind]

        if gtn_b.data_type is DataType.MPS:
            index_tensor.isel({gtn_b.data.site_ind_id.format(index_ind): select_ind}, inplace=True)
        elif gtn_b.data_type is DataType.MPO:
            index_tensor.isel({gtn_b.data.upper_ind_id.format(index_ind): select_ind[0]}, inplace=True)
            index_tensor.isel({gtn_b.data.lower_ind_id.format(index_ind): select_ind[1]}, inplace=True)
        # print('index tensor isel', index_tensor.data)

        next_ind = index_ind + 1 if index_ind == 0 else index_ind - 1
        sub_gtn_data = gtn_b.data.contract((gtn_b.data.site_tag_id.format(index_ind),
                                            gtn_b.data.site_tag_id.format(next_ind)), inplace=False)
        sub_gtn_data[next_ind].drop_tags(gtn_b.data.site_tag_id.format(index_ind))
        # print('subgtn data', sub_gtn_data)

        if index_ind == 0:
            if gtn_b.data_type is DataType.MPO:
                helper.renumber_mpo(sub_gtn_data, list(range(1, L)), list(range(L - 1)), inplace=True)
            elif gtn_b.data_type is DataType.MPS:
                helper.renumber_mps(sub_gtn_data, list(range(1, L)), list(range(L - 1)), inplace=True)

        sub_gtn_data._L = sub_gtn_data.num_tensors
        sub_gtn_data.exponent = gtn_b.data.exponent

        if helper.norm(sub_gtn_data) < np.sqrt(CUTOFF):
            sub_gtn_data = None

        if sub_gtn_data is not None:
            sub_gtn = gtn_b.create_like(sub_gtn_data)
            gtn[b_ind] = sub_gtn
        else:
            gtn.data = None
        # print('gtn', b_ind, gtn)
        return gtn

    #####################
    ## post-processing ##
    #####################

    # def interpolate_data(self):
    #     """ process the data in some way
    #     """
    #     pass
    #
    # def gaussian_smooth_mpo(self, sigma=0.3, proc_axes=None, mode='wrap', compress=True, split_opts=None) \
    #         -> GridTN1DComb:
    #     """ apply gaussian smoothing to data along proc_axes (axes to process)
    #         mode = 'wrap' or 'mirror' or whatever other options
    #     """
    #     gtn_mpo1d = super().gaussian_smooth_mpo(sigma, proc_axes, mode, compress=compress, split_opts=split_opts)
    #
    #     out_mpo = GridTN1DComb(self)
    #     for mpo_ax_dict in gtn_mpo1d.data:
    #         tn_comb = self.make_mpo_ndim(mpo_ax_dict)
    #         out_mpo.add(tn_comb, inplace=True)
    #     out_mpo.compress(compress_opts=split_opts)
    #     return out_mpo
    #
    #
    # def absorbing_bc_mpo(self, x_ax: Axis, v_ax: Axis, left_bc: BCType, right_bc: BCType, compress=True,
    #                      compress_opts=None) -> GridTN1DComb:
    #     """ apply absorbing bc to MPS state, assumes TN dimensions are x1,x2,x3,v1,v2,v3
    #         (prev: apply absorbing bc to derivative MPO, assumes TN dimensions are x1,x2,x3,v1,v2,v3)
    #         returns list of MPO dicts to add together for final MPO
    #     """
    #     gtn_mpo1d = super().absorbing_bc_mpo(x_ax, v_ax, left_bc, right_bc)
    #
    #     out_mpo = GridTN1DComb(self)
    #     for mpo_ax_dict in gtn_mpo1d.data:
    #         tn_comb = self.make_mpo_ndim(mpo_ax_dict)
    #         out_mpo.add(tn_comb, inplace=True)
    #     if compress:
    #         out_mpo.compress(compress_opts=compress_opts)
    #     return out_mpo
    #
    #
    # def reflecting_v_bc_mpo(self, x_ax: Axis, v_ax: Axis, left_bc: BCType, right_bc: BCType, compress=True,
    #                         compress_opts=None) -> GridTN1DComb:
    #     """ reflecting_v bc MPO that takes v -> -v
    #     """
    #     gtn_mpo1d = super().reflecting_v_bc_mpo(x_ax, v_ax, left_bc, right_bc)
    #
    #     out_mpo = GridTN1DComb(self)
    #     for mpo_ax_dict in gtn_mpo1d.data:
    #         tn_comb = self.make_mpo_ndim(mpo_ax_dict)
    #         out_mpo.add(tn_comb, inplace=True)
    #     if compress:
    #         out_mpo.compress(compress_opts=compress_opts)
    #     return out_mpo

    ####################
    ### dmrg methods ###
    ####################

    def gtn_to_dmrg_format(self, gtn: 'GridTN1DComb', is_mps=True) -> 'GridTN1D':
        gtn.collect_exponents()
        if self.ngrids == 1:
            # plt.figure()
            # plt.plot(gtn.get_data())
            #
            # gtn_ = gtn.scalar_multiply(-0.1, inplace=False)
            # gtn_ = gtn_.compress()
            #
            # plt.plot(gtn_.get_data())
            #
            # test = self._1D_comb_to_mps(gtn, is_mps=is_mps)
            # test_out = self._mps_to_1D_comb(gtn_, test, is_mps=is_mps)
            #
            # plt.plot(test_out.get_data(), '--')
            # plt.show()
            return self._1D_comb_to_mps(gtn, is_mps=is_mps)
        elif self.ngrids == 2:
            return self._2D_comb_to_mps(gtn, is_mps=is_mps)

    def dmrg_to_gtn_format(self, gtn, soln, is_mps=True):
        gtn.collect_exponents()
        if self.ngrids == 1:
            return self._mps_to_1D_comb(gtn, soln, is_mps=is_mps)
        elif self.ngrids == 2:
            return self._mps_to_2D_comb(gtn, soln, is_mps=is_mps)

    def _1D_comb_to_mps(self, gtn, is_mps=True) -> 'GridTN1D':
        assert self.ngrids == 1, 'must be a 1D comb'
        if gtn.data is None:
            return None
        # is_mps = gtn.data_type is DataType.MPS
        branch = gtn._attach_spine_tens_to_branch_v2(0, is_mps=is_mps)
        branch.exponent = gtn.exponent
        helper.scalar_multiply(branch, gtn.sign, inplace=True)
        return branch

    def _2D_comb_to_mps(self, gtn, is_mps=True) -> 'GridTN1D':
        assert self.ngrids == 2, 'must be a 2D comb'
        if gtn.data is None:
            return None
        # is_mps = gtn.data_type is DataType.MPS
        branch0 = gtn._attach_spine_tens_to_branch_v2(0, is_mps=is_mps)
        branch0 = helper.mps_flip_lr(branch0, inplace=False) if is_mps else helper.mpo_flip_lr(branch0, inplace=False)
        branch1 = gtn._attach_spine_tens_to_branch_v2(1, is_mps=is_mps)
        branch = helper.append_mpx(branch0, branch1, inplace=False)
        branch.exponent = gtn.exponent
        helper.scalar_multiply(branch, gtn.sign, inplace=True)
        return branch

    def _mps_to_1D_comb(self, gtn, soln: Union[qtn.MatrixProductOperator, qtn.MatrixProductState], is_mps=True):
        assert self.ngrids == 1, 'must be a 1D comb'
        # is_mps = gtn.data_type is DataType.MPS
        # branch = gtn._attach_spine_tens_to_branch(0, is_mps=is_mps)
        branch = gtn[0].data

        helper.match_inner_inds(soln, branch, inplace=True)
        # branch.exponent = 0.0
        gtn._exponent = soln.exponent  # - gtn[0].exponent
        gtn._sign = 1.0
        soln.exponent = 0.0
        # gtn._update_spine_tens_and_branch(0, soln)
        gtn._update_spine_tens_and_branch_v2(0, soln)
        gtn.canon_site = None
        return gtn

    def _mps_to_2D_comb(self, gtn, soln: Union[qtn.MatrixProductOperator, qtn.MatrixProductState], is_mps=True):
        assert self.ngrids == 2, 'must be a 2D comb'
        # is_mps = gtn.data_type is DataType.MPS
        branch0 = gtn._attach_spine_tens_to_branch_v2(0, is_mps=is_mps)
        branch0 = helper.mps_flip_lr(branch0, inplace=False) if is_mps else helper.mpo_flip_lr(branch0, inplace=False)
        branch1 = gtn._attach_spine_tens_to_branch_v2(1, is_mps=is_mps)

        # print('soln', soln)
        if soln is None:
            gtn.data = None
            return gtn

        soln0, soln1 = helper.split_mpx(soln, gtn[0].data.L)
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
        return gtn

    def _setup_dmrg_solver(self, gtn: 'GridTN1DComb', operator: 'GridTN1DComb', compress_type: 'CompressType',
                           inplace=False, compress_opts=None, is_H=False, init_guess: 'GridTN1DComb' = None,
                           verbose_output=False, **kwargs):
        """ set up solver for DMRG LinearSolve
            return solver and branches (for reference)
        """
        max_bond = None if compress_opts is None else compress_opts.get('max_bond', None)

        ## if 1D
        if self.ngrids == 1:
            gtn.collect_exponents()
            gtn.distribute_sign()

            operator.collect_exponents()
            operator.distribute_sign()

            # branch = gtn._attach_spine_tens_to_branch(0, is_mps=True)
            # branch.exponent = gtn.exponent
            # branch_op = operator._attach_spine_tens_to_branch(0, is_mps=False)
            # branch_op.exponent = operator.exponent
            branch = self._1D_comb_to_mps(gtn, is_mps=True)
            branch_op = self._1D_comb_to_mps(operator, is_mps=False)

            if init_guess is None:
                init_guess_branch = branch.copy()
            else:
                init_guess = init_guess.copy()
                init_guess.spine_ind_id = gtn.spine_ind_id
                helper.match_inner_inds(init_guess.spine, gtn.spine, inplace=True)
                init_guess.collect_exponents()
                init_guess.distribute_sign()
                # init_guess.distribute_exponents()
                init_guess_branch = init_guess._attach_spine_tens_to_branch_v2(0, is_mps=True)
                init_guess_branch.site_ind_id = branch.site_ind_id
                init_guess_branch.exponent = init_guess.exponent

            soln_mps = branch
            operator = branch_op
            init_guess = init_guess_branch

        ## if 2D
        elif self.ngrids == 2:
            gtn.collect_exponents()
            gtn.distribute_sign()

            operator.collect_exponents()
            operator.distribute_sign()

            ## maybe could get slightly better results if contract spine tens into branch
            # branch0 = gtn._attach_spine_tens_to_branch(0, is_mps=True)
            # branch0 = helper.mps_flip_lr(branch0, inplace=False)
            # branch1 = gtn._attach_spine_tens_to_branch(1, is_mps=True)
            # branch = helper.append_mpx(branch0, branch1, inplace=False)
            # branch.exponent = gtn.exponent

            # branch0_op = operator._attach_spine_tens_to_branch(0, is_mps=False)
            # branch0_op = helper.mpo_flip_lr(branch0_op, inplace=False)
            # branch1_op = operator._attach_spine_tens_to_branch(1, is_mps=False)
            # branch_op = helper.append_mpx(branch0_op, branch1_op, inplace=False)
            # branch_op.exponent = operator.exponent
            branch = self._2D_comb_to_mps(gtn, is_mps=True)
            branch_op = self._2D_comb_to_mps(operator, is_mps=False)

            if init_guess is None:
                init_guess_branch = branch.copy()
            else:
                init_guess = init_guess.copy()
                init_guess.spine_ind_id = gtn.spine_ind_id
                helper.match_inner_inds(init_guess.spine, gtn.spine, inplace=True)
                init_guess.collect_exponents()
                init_guess.distribute_sign()
                # init_guess.distribute_exponents()
                init_guess_branch = self._2D_comb_to_mps(init_guess)
                # init_guess_branch0 = init_guess._attach_spine_tens_to_branch(0, is_mps=True)
                # init_guess_branch0 = helper.mps_flip_lr(init_guess_branch0, inplace=False)
                # init_guess_branch1 = init_guess._attach_spine_tens_to_branch(1, is_mps=True)
                # init_guess_branch = helper.append_mpx(init_guess_branch0, init_guess_branch1, inplace=False)
                # init_guess_branch.site_ind_id = branch.site_ind_id
                # init_guess_branch.exponent = init_guess.exponent

            # soln, err, is_conv = helper_dmrg.dmrg_solve(branch, branch_op, init_guess=init_guess_branch, is_H=is_H,
            #                                             max_bond=max_bond, **kwargs)

            soln_mps = branch
            operator = branch_op
            init_guess = init_guess_branch
            # ref_branches = [branch0, branch1]

        else:
            raise NotImplementedError

        from helper_dmrg import LinearSolver
        init_guess = soln_mps if init_guess is None else init_guess  ## should be zero, probably?
        solver = LinearSolver(init_guess, targets=[soln_mps.copy()], operators=[operator], is_H=is_H,
                              max_bond=max_bond, **kwargs)

        return solver

    def _extract_dmrg_soln(self, gtn: 'GridTN1DComb', soln, inplace=False):
        gtn = gtn if inplace else gtn.copy()

        ## if 1D
        if self.ngrids == 1:
            # branch, = branches

            # helper.match_inner_inds(soln, branch, inplace=True)
            # # branch.exponent = 0.0
            # gtn._exponent = soln.exponent  # - gtn[0].exponent
            # soln.exponent = 0.0
            # gtn._update_spine_tens_and_branch(0, soln)
            # gtn.canon_site = None
            self._mps_to_1D_comb(gtn, soln, is_mps=True)

        ## if 2D
        elif self.ngrids == 2:
            # branch0, branch1 = branches
            self._mps_to_2D_comb(gtn, soln, is_mps=True)

        else:
            raise NotImplementedError

        return gtn
