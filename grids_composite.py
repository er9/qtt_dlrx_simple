from setup_.configs import *

from grid import Grid
from gridTN import GridTN
from grid1D import Grid1D
from gridTN_composite import GridTN_Composite as compGTN

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis
    from gridTN import GridTN
    from gridTN_composite import GridTN_Composite

""" Group of Grid objects (which are groups of Axis objects) or other Grids objects
"""

class CompositeGrid(Grid):

    def __init__(self, gridID, grids: Iter[Grid or 'CompositeGrid']):
        """
        Inputs:
            axesID:  ID to use for this GridLayout object
            axes: list of Axis objects. CAN REPEAT THE SAME OBJECT?

        Attributes:
            grids:  list of Grid or Grids objects
            axes:   list of Axis objects associated with Grids
            ngrids: number of grids
        """

        self.gridID = gridID
        self.grids: tuple['Grid'] = tuple(grids)
        self.gridIDs = tuple([gr.gridID for gr in self.grids])
        self.ngrids = len(grids)

        axes = []
        self.ax_to_grid_map = {}
        for x in range(self.ngrids):
            axes += [ax for ax in grids[x].axes]
            self.ax_to_grid_map.update({ax: grids[x] for ax in grids[x].axes})

        self.grid_inds = {self.grids[x]: x for x in range(self.ngrids)}
        # self.axis_inds = {axes[x]: x for x in range(len(axes))}

        super().__init__(gridID, tuple(axes))

        # defines self.axes, self.ndim, self.npts, self.axIDs.
        # and private storage of MPOs
        # will overwrite ax_coordsys_map

        # self.ax_coordsys_map = {}
        # for gr in self.grids:
        #     self.ax_coordsys_map.update(gr.ax_coordsys_map)


    def __getitem__(self, item: int):
        return self.grids[item]

    def __len__(self):
        return self.ngrids


    def get_active_grids(self, axes: Sequence['Axis']) -> list[Grid]:
        return list(set([self.ax_to_grid_map[ax] for ax in axes]))


    def get_axis_grid(self, ax:'Axis') -> 'Grid':
        return self.ax_to_grid_map[ax]


    def get_grid_ind(self, gr:'Grid') -> int:
        return self.grid_inds[gr]

    def get_grid_with_all_axes(self, axes: Sequence['Axis']) -> 'Grid':
        grids = [self.get_axis_grid(ax) for ax in axes]
        if len(grids) > 1:
            assert(np.all([gr is grids[0] for gr in grids[1:]]))
        return grids[0]

    def get_grid_with_any_axes(self, axes: Sequence['Axis']) -> 'Grid':
        """ returns first hit
        """
        for ax in axes:
            try:
                return self.get_axis_grid(ax)
            except KeyError:
                pass
        return None

    #################

    def create_like(self, new_grids=None, new_gridID=None) -> 'CompositeGrid':
        """ create a new grid object with potentially new Axis objects and new gridID
        """
        if new_grids is None:    new_grids = self.grids
        if new_gridID is None:   new_gridID = self.gridID

        # print('new grids', new_grids, self.grids)
        new_grid = self.__class__(new_gridID, new_grids)
        return new_grid


    def create_like_from_axes(self, new_axes: Iter['Axis']=None, new_gridID=None) -> 'Grid':
        if new_axes is None:
            new_grids = self.grids
        else:
            new_grids = [gr for gr in self.grids if len(set(gr.axes).intersection(set(new_axes))) > 0]

        return self.create_like(new_grids=new_grids, new_gridID=new_gridID)


    def copy(self, deep: bool = False, new_gridID=None) -> 'CompositeGrid':
        """ make a copy of these Axes. can redefine gridID while not making
            deep copies of the saved MPOs
        """
        if new_gridID is None:   new_gridID = self.gridID
        new_grid = self.create_like(self.grids, new_gridID)

        if deep:
            new_grid._mpo_xmultiply = {k: mpo.copy() for k, mpo in self._mpo_xmultiply.items()}
            new_grid._mpo_firstderivatives = {k: mpo.copy() for k, mpo in self._mpo_firstderivatives}
            new_grid._mpo_secondderivatives = {k: mpo.copy() for k, mpo in self._mpo_secondderivatives}
            new_grid._mpx_integrals = {k: mpo.copy() for k, mpo in self._mpx_integrals}
        else:
            new_grid._mpo_xmultiply = self._mpo_xmultiply
            new_grid._mpo_firstderivatives = self._mpo_firstderivatives
            new_grid._mpo_secondderivatives = self._mpo_secondderivatives
            new_grid._mpx_integrals = self._mpx_integrals

        return new_grid


    # def get_subgrid(self, select_axes, new_gridID=None):
    #     ordered_axes = [ax for ax in self.axes if ax in select_axes]
    #     ordered_grids = []
    #     for ax in ordered_axes:
    #         ax_gr = self.get_axis_grid(ax)
    #         if ax_gr not in ordered_grids:
    #             ordered_grids += [ax_gr]
    #     return self.create_like(ordered_grids, new_gridID=new_gridID)

    def get_subgrid(self, select_axes, new_gridID=None):
        ordered_grids = []
        for gr in self.grids:
            ordered_axes = [ax for ax in gr.axes if ax in select_axes]
            if len(ordered_axes) > 0:
                ordered_grids += [gr.get_subgrid(ordered_axes)]

        return self.create_like(ordered_grids, new_gridID=new_gridID)

    ################
    ## common TNs ##
    ################

    def make_empty_gridTN(self, ax_deriv_configs=None) -> 'compGTN':
        # mps_dict = {}
        # for grid in self.grids:
        #     mps_dict[grid.gridID] = grid.make_empty_gridTN()
        return compGTN(self, data=None, ax_deriv_configs=ax_deriv_configs)

    def make_gridTN(self, data=None, ax_deriv_configs: dict['Axis','DerivativeConfiguration']=None) -> 'GridTN':
        """ make GTN given appropriate data: list of dicts[Grid, GTN]
        """
        return compGTN(self, data, ax_deriv_configs)

    def get_ones_mps(self, site_ind_id='i({})', site_tag_id='X({})') -> 'compGTN':
        """ gtn_cls: GridTN class type
            build ones vector mps
        """
        mps_dict = {gr: gr.get_ones_mps(site_ind_id, site_tag_id) for gr in self.grids}
        return compGTN(self, data=[mps_dict])

    def get_iden_mpo(self, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> 'compGTN':
        """ build identity mpo
        """
        mpo_dict = {gr: gr.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id) for gr in self.grids}
        return compGTN(self, data=[mpo_dict])

    # def get_select_elems_mps(self, inds, site_ind_id='i({})', site_tag_id='X({})') -> 'compGTN':
    #     """ build MPO to select certain elements specified by inds
    #     """
    #     assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
    #     mps_dict = {ax: ax.get_select_elems_mps(ix, site_ind_id, site_tag_id)
    #                 for ax, ix in zip(self.axes, inds)}
    #     return self.make_gtn_from_dicts(mps_dict, data_type=DataType.MPS)
    #
    # def get_select_elems_mpo(self, inds, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> 'compGTN':
    #     """ build MPO to select certain elements specified by inds
    #     """
    #     assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
    #     mpo_dict = {ax.coordinate: ax.get_select_elems_mpo(ix, upper_ind_id, lower_ind_id, site_tag_id)
    #                 for ax, ix in zip(self.axes, inds)}
    #     return self.make_gtn_from_dicts(mpo_dict, data_type=DataType.MPO)

    #########################################
    ## convert between np.ndarray and MPX ##
    #########################################

    def map_state_to_mps(self, state: np.ndarray, site_ind_id='i({})', site_tag_id='T({})', direction=0,
                         split_opts=None, ax_deriv_configs=None, ancilla_right=(), ancilla_right_inds=(),
                         ancilla_left=(), ancilla_left_inds=(), axes=None) -> MPSType:
        """ convert state represented as vector converted to mps state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        raise NotImplementedError

    def map_mps_to_state(self, gtn_mps: MPSType, ax_select=None, ancilla_right=(), ancilla_right_inds=(),
                         ancilla_left=(), ancilla_left_inds=()) -> np.ndarray:
        """ convert MPS into ndim-dim np.ndarray
        """
        raise NotImplementedError

    def map_operator_to_mpo(self, operator, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='B({})',
                            direction=0, split_opts=None, ancilla_right=(), ancilla_right_inds=(), ancilla_left=(),
                            ancilla_left_inds=()) -> MPOType:
        """ convert operator represented as high-dimensional matrix to mpo state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        raise NotImplementedError

    def map_mpo_to_operator(self, gtn_mpo, ax_select=None, ancilla_right=(), ancilla_right_inds=(), ancilla_left=(),
                            ancilla_left_inds=()) -> np.ndarray:
        """ convert MPO into ndim*2-dim np.ndarray (o0 o1 ... x i0 i1 ...)
        """
        raise NotImplementedError

    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    def make_mps_ndim(self, mps_1d_dict):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            if MPS not defined along that dimension, use a ones vector (constant along that dimension)
        """
        out = {}
        for gr in self.grids:
            active_axes = [ax for ax in gr.axes if ax in mps_1d_dict.keys()]
            active_dict = {ax: mps_1d_dict[ax] for ax in active_axes}
            out[gr] = gr.make_mps_ndim(active_dict)
        return compGTN(self, data=[out])

    def make_mpx_ndim(self, mps_1d_dict):
        """ combine 1-D MPSs into K-dimensional MPO
            if MPS not defined along that dimension, pad with the identity MPO
        """
        raise NotImplementedError

    def make_mpo_ndim(self, mpo_1d_dict):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        out = {}
        for gr in self.grids:
            active_axes = [ax for ax in gr.axes if ax in mpo_1d_dict.keys()]
            active_dict = {ax: mpo_1d_dict[ax] for ax in active_axes}
            out[gr.gridID] = gr.make_mpo_ndim(active_dict)
        return compGTN(self, data=[out])

    def make_tn1d_ndim(self, tn_1d_dict, in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                       site_tag_id='D({})'):
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        out = {}
        for gr in self.grids:
            active_axes = [ax for ax in gr.axes if ax in tn_1d_dict.keys()]
            active_dict = {ax: tn_1d_dict[ax] for ax in active_axes}
            out[gr.gridID] = gr.make_tn1d_ndim(active_dict, in1_ind_id=in1_ind_id, in2_ind_id=in2_ind_id,
                                               out_ind_id=out_ind_id, site_tagd_id=site_tag_id)
        return compGTN(self, data=[out])

    def pad_mps_to_grid(self, scalar_field_mps, mps_axes):
        """ pad MPS on mps_grid onto self.grid
            return TN/MPS associated with a particular grid level via padding
        """
        active_grids = self.get_active_grids(mps_axes)
        raise NotImplementedError

    def pad_mpo_to_grid(self, scalar_field_mpo, mpo_axes):
        """ pad MPO on mpo_grid onto self.grid
        """
        raise NotImplementedError


    def make_gtn_from_dicts(self, data:Iter[dict['Axis',Optional['TNType']]], data_type=DataType.MPO, compress=False,
                            compress_opts=None):
        """ return GridTN object
        """
        # out = self.make_empty_gridTN()
        out = None
        out_compress = False

        for data_dict in data:

            if data_type is DataType.MPS:
                new_out = self.make_mps_ndim(data_dict)
            elif data_type is DataType.MPO:
                new_out = self.make_mpo_ndim(data_dict)
            elif data_type is DataType.MPX:
                new_out = self.make_mpx_ndim(data_dict)
            elif data_type is DataType.TN3:
                new_out = self.make_tn1d_ndim(data_dict)
            else:
                raise TypeError(f'data type is currently {data_type}')

            if out is None:
                out = new_out         # don't compress bc input would have already been compressed
            else:
                out.add(new_out, inplace=True)
                out_compress = compress

            if out_compress:
                out.compress(inplace=True, compress_opts=compress_opts)

        return out


    # @profile
    def build_mps_from_subgtns(self, data_list, compress_opts=None, **kwargs):
        """ take outerproduct from gtns of subgrids
            data_list:  sequence of tuple(ax inds or axes, np.ndarray or MPS)
                                    or gridTN
        """
        def get_subgr_data(gtn_data):
            if isinstance(gtn_data, tuple):
                axes, gtn_data = gtn_data
                axes = [self.axes[i] if isinstance(i,int) else i for i in axes]
                sub_gr = self.get_subgrid(axes)
                if isinstance(gtn_data, np.ndarray):
                    gtn_mps = sub_gr.map_state_to_mps(gtn_data, axes=axes, **kwargs)
                elif isinstance(gtn_data, qtn.MatrixProductState):
                    assert (len(axes) == 1), 'putting MPO spanning multiple branches into comb not implemented'
                    gtn_mps = sub_gr.make_mps_ndim({axes[0]: gtn_data})
                else:
                    raise NotImplementedError
                out_gtn_mps = self.pad_gtn_to_grid(gtn_mps)
            elif isinstance(gtn_data, GridTN):
                out_gtn_mps = self.pad_gtn_to_grid(gtn_data)
            else:
                raise TypeError('not valid data in data_list')
            return out_gtn_mps

        out = get_subgr_data(data_list[0])
        for data in data_list[1:]:
            out2: 'GridTN' = get_subgr_data(data)
            out2 = out2.apply_elemental_multiply_op()
            # out2 = out2.mps_to_diag_mpo()
            # print('out', out)
            out = out.apply(out2, inplace=True, compress_opts=compress_opts)

        # print('out', out)
        out = out.compress(compress_opts=compress_opts)

        return out


    def build_mpo_from_subgtns(self, data_list):
        """ take outerproduct from gtns of subgrids
            data_list:  sequence of tuple(ax inds or axes, np.ndarray or MPS)
                                    or gridTN
        """
        def get_subgr_data(gtn_data):
            if isinstance(gtn_data, tuple):
                axes, gtn_data = gtn_data
                axes = [self.axes[i] if isinstance(i,int) else i for i in axes]
                sub_gr = self.get_subgrid(axes)
                if isinstance(gtn_data, np.ndarray):
                    gtn_mpo = sub_gr.map_operator_to_mpo(gtn_data)
                elif isinstance(gtn_data, qtn.MatrixProductOperator):
                    assert(len(axes)==1),'putting MPO spanning multiple branches into comb not implemented'
                    gtn_mpo = sub_gr.make_mpo_ndim({axes[0]: gtn_data})
                else:
                    raise NotImplementedError
                out_gtn_mpo = self.pad_gtn_to_grid(gtn_mpo)
            elif isinstance(gtn_data, GridTN):
                out_gtn_mpo = self.pad_gtn_to_grid(gtn_data)
            else:
                raise TypeError('not valid data in data_list')
            return out_gtn_mpo

        out = get_subgr_data(data_list[0])
        for data in data_list[1:]:
            out2: 'GridTN' = get_subgr_data(data)
            # out2 = out2.apply_elemental_multiply_op()
            # print('out', out)
            # print('out2', out2)
            out = out.apply(out2, inplace=False)

        # print('out', out)
        out = out.compress()
        return out



    def gtn_outerproduct(self, *gtns: Union['GridTN','compGTN']):
        """ take outer product of gtns provided
            assert that the active grids of the gtns are not overlapping
        """
        raise NotImplementedError

