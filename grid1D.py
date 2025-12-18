from scipy import linalg as spla

import axis
from setup_.configs import *
from setup_.quimb_TN1D import MatrixProductStateTN

import helper_quimb as helper
import helper_dmrg

from basis.basis_k import RealFourierBasis
# from layout.layout import LayoutType
from grid import Grid
from gridTN_1D import GridTN1D

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axis import Axis
    from layout.layout import Layout
    from gridTN import GridTN

""" Defines the multi-dimensional grid, built from Axis objects
    layout is the geometry used to represent high-dimensional TNs
    contains information like those involving differential operators (laplacian, cross-product)
    and common mathematical operations (like elemental multiply and xmultiply
"""


def get_layout(layout_type) -> 'Layout':
    """ return basis class given BasisType(Enum)
    """
    if layout_type == LayoutType.SEQUENTIAL:
        from layout.layout_sequentialF import LayoutSequentialF
        return LayoutSequentialF
    if layout_type == LayoutType.PARALLEL:
        from layout.layout_parallelF import LayoutParallelF
        return LayoutParallelF
    if layout_type == LayoutType.PARALLEL_GROUP:
        from layout.layout_parallelG import LayoutParallelG
        return LayoutParallelG


class Grid1D(Grid):

    def __init__(self, gridID, axes: Iter['Axis'], layout_type=LayoutType.SEQUENTIAL):
        """
        Inputs:
            axesID:  ID to use for this GridLayout object
            axes: list of Axis objects. CAN REPEAT THE SAME OBJECT?
            layout_type: LayoutType that specifies Grid class defining geometry of TN of multi-dimensional system.
                Defines mathematical operations applied to TN given said geometry
            order:  2*order+1 - point stencil for finite difference methods
            compress_opts: see quimb. eg. max_bond: max virtual bond dimension of TNs,
                                          cutoff: cutoff tolerance for singular values

        Attributes:
            gridID: ID for this GridLayout object
            axes:   list of Axis objects
            ndim:   number of Axis objects
            layout: Layout of the ndim-object mapped to a 1D grid
            order:  sets finite difference accuracy
        """
        self.layout_type = layout_type
        self.layout = get_layout(layout_type)
        self.L = self.layout.L(axes)

        super().__init__(gridID, axes)

    @property
    def num_tensors(self) -> int:
        if self.layout_type == LayoutType.PARALLEL_GROUP:
            return int(np.max([ax.L for ax in self.axes]))
        return int(np.sum([ax.L for ax in self.axes]))

    def get_inds_in_axis(self, ax, ax_ind=None) -> list:
        """ get inds in 1D TN corresponding to axis axID
        """
        return self.layout.get_inds_in_axis(self.axes, ax, ax_ind)

    def shape(self) -> tuple:
        """ get grid shape
        """
        return self.layout.shape(self.axes)

    def create_like(self, new_axes=None, new_gridID=None):
        """ create a new grid object with potentially new Axis objects and new gridID
        """
        if new_axes is None:     new_axes = self.axes
        if new_gridID is None:   new_gridID = self.gridID

        # print('new axes', new_axes, self.axes)
        new_grid = self.__class__(new_gridID, new_axes, self.layout_type)
        return new_grid

    def get_coarsen_inds(self, coarsen_level):
        coarsen_inds = []
        for ax in self.axes:
            c_ind = ax.get_coarseness_ind(coarsen_level)
            if c_ind is not None:
                axis_inds = self.get_inds_in_axis(ax)
                coarsen_inds += [axis_inds[c_ind]]
        return coarsen_inds

    # ######################
    # ## axis information ##
    # ######################
    #
    # def get_inds_in_axes(self,ax):
    #     """ get indices corresponding to tensors along a given axis (for factorized TNs)
    #     """
    #     return None
    #
    #
    # def get_ax_vals(self,axes=None):
    #     """ get axis values for axes specified by dims
    #         eg. get x, v data on the full grid
    #     """
    #     return_single = False
    #     if isinstance(axes,int):
    #         axes = [axes]
    #         return_single = True
    #
    #     if axes is None:   axes = range(self.ndim)
    #
    #     npts = self.q**self.L
    #
    #     out = {}
    #     for ax in axes:
    #         out[ax] = self.zero[ax] + np.arange(npts)*self.dx[ax]
    #
    #     if return_single:   return out[ax]
    #     else:               return out
    #
    #
    # def get_val_ind(self,ax,val):
    #     """ find index (int) of grid poin corresponding to desired value val along axis ax
    #     """
    #     grid_pts = self.get_ax_vals(ax)
    #     ind = np.argwhere(np.abs(grid_pts-val)<1.0-10)
    #     if len(ind) == 0:
    #         raise IndexError('get_val_ind: grid does not contain desired value',val)
    #     return ind[0]

    #
    # def get_ax_mesh(self,axes=None):
    #     """ get self.ndim -dimensional grid with axis values for axes specified by dims
    #         eg. get x, v data on the full grid
    #     """
    #     return_single = False
    #     if isinstance(axes,int):
    #         axes = [axes]
    #         return_single = True
    #
    #     if axes is None:
    #         axes = range(self.ndim)
    #
    #     ax_vals = self.get_ax_vals(axes)
    #
    #     out = {}
    #     for ax in axes:
    #         val_mps = self.map_state_to_mps(ax_vals[ax],ndim=1)
    #         out[ax] = self.make_mps_ndim({ax:val_mps})
    #
    #     if return_single:   return out[ax]
    #     else:               return out

    ################
    ## common TNs ##
    ################

    def make_empty_gridTN(self, ax_deriv_configs=None) -> 'GridTN1D':
        return GridTN1D(self, ax_deriv_configs=ax_deriv_configs)

    def make_zero_gridTN(self, max_bond: int = 2, orthog: int = 0,
                         ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None) -> 'GridTN':
        out = self.get_rand_state(max_bond=max_bond)
        out.canonize(inplace=True, i=orthog)
        out.data[orthog].modify(apply = lambda x: x * 0)
        out.data.exponent = 0.0
        return out

    def make_gridTN(self, data: TNType = None, ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None) \
            -> 'GridTN1D':
        return GridTN1D(self, data, ax_deriv_configs)

    def get_ones_mps(self, site_ind_id='i({})', site_tag_id='X({})') -> GridTN1D:
        """ gtn_cls: GridTN class type
            build ones vector mps
        """
        mps_dict = {ax: ax.get_iden_mps(site_ind_id, site_tag_id) for ax in self.axes}
        return self.make_mps_ndim(mps_dict)
        # return gtn_cls(self, data=new_mps)

    def get_iden_mpo(self, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> GridTN1D:
        """ build identity mpo
        """
        mpo_dict = {ax: ax.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id) for ax in self.axes}
        return self.make_mpo_ndim(mpo_dict)
        # return gtn_cls(self, data=self.make_mps_ndim(mpo_dict))

    def get_select_elem_mps_bin(self, inds: Iter[int], site_ind_id='i({})', site_tag_id='X({})') -> 'GridTN':
        """ build MPS to select certain elements specified by inds
        """
        from axis import get_select_elem_mps
        assert (self.L == len(inds)), 'number of inds needs to match number of tensor cores'
        q = self.axes[0].q
        sel_mps = get_select_elem_mps(self.L, q, inds, site_ind_id=site_ind_id, site_tag_id=site_tag_id)
        return self.make_gridTN(sel_mps)


    # def get_select_elems_mps(self, inds, site_ind_id='i({})', site_tag_id='X({})') -> GridTN1D:
    #     """ build MPO to select certain elements specified by inds
    #     """
    #     assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
    #     mps_dict = {ax: ax.get_select_elems_mps(ix, site_ind_id, site_tag_id)
    #                 for ax, ix in zip(self.axes,inds)}
    #     # return gtn_cls(self, data=self.make_mps_ndim(mps_dict))
    #     return self.make_mps_ndim(mps_dict)

    # def get_select_elems_mpo(self, inds, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> GridTN1D:
    #     """ build MPO to select certain elements specified by inds
    #     """
    #     assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
    #     mps_dict = {ax: ax.get_select_elems_mpo(ix, upper_ind_id, lower_ind_id, site_tag_id)
    #                 for ax, ix in zip(self.axes,inds)}
    #     return self.make_mps_ndim(mps_dict)
    #     # return gtn_cls(self, data=self.make_mps_ndim(mps_dict))

    # def get_shift_mpo(self, shifts: dict['Axis',int], ax_boundary_conditions: dict['Axis', 'DerivativeConfiguration'] = None,
    #                   upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> 'GridTN1D':
    #     """ build MPO to shift elements along each axes by specified amount
    #     """
    #     mpo_dict = {ax: ax.get_shift_mpo(shifts[ax], boundary_conditions=ax_boundary_conditions[ax],
    #                                      upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)
    #                 for ax in shifts.keys()}
    #     return self.make_mpo_ndim(mpo_dict)

    def get_rand_state(self, max_bond):
        rand_mps = qtn.MPS_rand_state(self.L, max_bond)
        return self.make_gridTN(rand_mps)

    #########################################
    ## convert between np.ndarray and MPX ##
    #########################################

    def map_state_to_mps(self, state: np.ndarray, site_ind_id='i({})', site_tag_id='T({})', direction=0,
                         split_opts=None, ax_deriv_configs=None, ancilla_right=(), ancilla_right_inds=(),
                         ancilla_left=(), ancilla_left_inds=(), axes=None) -> GridTN1D:
        """ convert state represented as vector converted to mps state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        if np.linalg.norm(state) < np.sqrt(CUTOFF):
            mps = None
        else:
            if axes is not None and axes != self.axes:
                mps = self.layout.map_state_to_mps(self.axes, state, site_ind_id=site_ind_id, site_tag_id=site_tag_id,
                                                   direction=direction, split_opts=split_opts,
                                                   ancilla_right=ancilla_right,
                                                   ancilla_right_inds=ancilla_right_inds,
                                                   ancilla_left=ancilla_left, ancilla_left_inds=ancilla_left_inds)
                mps = self.pad_mps_to_grid(mps, tuple(axes))
            else:
                mps = self.layout.map_state_to_mps(self.axes, state, site_ind_id=site_ind_id, site_tag_id=site_tag_id,
                                                   direction=direction, split_opts=split_opts, ancilla_right=ancilla_right,
                                                   ancilla_right_inds=ancilla_right_inds,
                                                   ancilla_left=ancilla_left, ancilla_left_inds=ancilla_left_inds)
        return GridTN1D(self, data=mps, ax_deriv_configs=ax_deriv_configs)

    def map_mps_to_state(self, gtn_mps: 'GridTN1D', ax_select=None, ancilla_right=(),
                         ancilla_right_inds=(), ancilla_left=(), ancilla_left_inds=()) \
            -> Union[np.ndarray, qtn.Tensor]:
        """ convert MPS into ndim-dim np.ndarray
        """
        return self.layout.map_mps_to_state(self.axes, gtn_mps.data, ax_select=ax_select,
                                            ancilla_right=ancilla_right, ancilla_right_inds=ancilla_right_inds,
                                            ancilla_left=ancilla_left, ancilla_left_inds=ancilla_left_inds)

    def map_operator_to_mpo(self, operator, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='B({})',
                            direction=0, split_opts=None, ancilla_right=(), ancilla_right_inds=(), ancilla_left=(),
                            ancilla_left_inds=()) -> GridTN1D:
        """ convert operator represented as high-dimensional matrix to mpo state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        mpo = self.layout.map_operator_to_mpo(self.axes, operator,
                                              upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                              site_tag_id=site_tag_id, direction=direction, split_opts=split_opts,
                                              ancilla_right=ancilla_right, ancilla_right_inds=ancilla_right_inds,
                                              ancilla_left=ancilla_left, ancilla_left_inds=ancilla_left_inds)
        return GridTN1D(self, data=mpo)

    def map_mpo_to_operator(self, gtn_mpo: 'GridTN1D', ax_select=None, ancilla_right=(), ancilla_right_inds=(),
                            ancilla_left=(),
                            ancilla_left_inds=()) -> np.ndarray:
        """ convert MPO into ndim*2-dim np.ndarray (o0 o1 ... x i0 i1 ...)
        """
        return self.layout.map_mpo_to_operator(self.axes, gtn_mpo.data, ax_select=ax_select,
                                               ancilla_right=ancilla_right, ancilla_right_inds=ancilla_right_inds,
                                               ancilla_left=ancilla_left, ancilla_left_inds=ancilla_left_inds)

    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    def make_mps_ndim(self, mps_1d_dict) -> Union[GridTN1D, Numeric]:
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            if MPS not defined along that dimension, use a ones vector (constant along that dimension)
        """
        mult_constant = 1.0
        keys = list(mps_1d_dict.keys())
        for ax in keys:
            assert (ax in self.axes), f'{ax} is not in Grid'
            state = mps_1d_dict[ax]
            if np.isscalar(state):
                mult_constant *= state
                mps_1d_dict.pop(ax)
            elif isinstance(state, np.ndarray):
                mps_state = ax.map_state_to_mps(state)
                mps_1d_dict[ax] = mps_state

        if len(mps_1d_dict) == 0:
            return mult_constant

        mps = self.layout.make_mps_ndim(self.axes, mps_1d_dict)
        helper.scalar_multiply(mps, mult_constant, inplace=True)
        out_gtn = GridTN1D(self, mps)

        if self.layout_type is LayoutType.PARALLEL:
            out_gtn.compress(inplace=True)

        return out_gtn

    def make_mpx_ndim(self, mps_1d_dict) -> GridTN1D:
        """ combine 1-D MPSs into K-dimensional MPO
            if MPS not defined along that dimension, pad with the identity MPO
        """
        for ax, state in mps_1d_dict.items():
            assert (ax in self.axes), f'{ax} is not in Grid'
            if isinstance(state, np.ndarray):
                mps_state = ax.map_state_to_mps(state)
                mps_1d_dict[ax] = mps_state

        mpx = self.layout.make_mpx_ndim(self.axes, mps_1d_dict)
        out_gtn = GridTN1D(self, mpx)

        if self.layout_type is LayoutType.PARALLEL:
            out_gtn = out_gtn.compress(inplace=True)

        return out_gtn

    def make_mpo_ndim(self, mpo_1d_dict) -> GridTN1D:
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        if len(mpo_1d_dict) == 0:
            return None

        for ax, operator in mpo_1d_dict.items():
            assert(ax in self.axes), f'{ax} is not in Grid'
            # if isinstance(operator, np.ndarray):
            if not isinstance(operator, qtn.TensorNetwork):
                mpo_state = ax.map_operator_to_mpo(operator)
                mpo_1d_dict[ax] = mpo_state

        mpo = self.layout.make_mpo_ndim(self.axes, mpo_1d_dict)
        out_gtn = GridTN1D(self, mpo)

        if self.layout_type is LayoutType.PARALLEL:
            out_gtn.compress(inplace=True)

        return out_gtn

    def make_tn1d_ndim(self, tn_1d_dict, in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                       site_tag_id='D({})') -> GridTN1D:
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        tn = self.layout.make_tn1D_ndim(self.axes, tn_1d_dict, upper_ind_id=out_ind_id, lower_ind_id=in1_ind_id,
                                        extra_ind_ids=(in2_ind_id,), site_tag_id=site_tag_id)
        return GridTN1D(self, tn)

    def pad_gtn_to_grid(self, gtn, target_data_type=None):
        if gtn.grid == self:
            return gtn.copy()
        if gtn.data_type == DataType.MPS:
            out = self.pad_mps_to_grid(gtn.data, gtn.grid.axes)
            out.is_constant = gtn.is_constant
            out.constant_axes = gtn.constant_axes + [ax for ax in self.axes if ax not in gtn.grid.axes]
            return out
        elif gtn.data_type == DataType.MPO:
            out = self.pad_mpo_to_grid(gtn.data, gtn.grid.axes)
            out.is_constant = gtn.is_constant
            out.constant_axes = gtn.constant_axes + [ax for ax in self.axes if ax not in gtn.grid.axes]
            return out
        elif gtn.data_type is None:
            return None
        elif gtn.data_type is DataType.Num:
            if target_data_type == DataType.MPS:
                out = self.get_ones_mps().scalar_multiply(gtn.data)
            elif target_data_type == DataType.MPO:
                out = self.get_iden_mpo().scalar_multiply(gtn.data)
            else:
                raise TypeError
            out.ax_deriv_configs = gtn.ax_deriv_configs
            return out
        else:
            raise NotImplementedError

    def pad_mps_to_grid(self, scalar_field_mps: 'MPSType', mps_axes: tuple['Axis']) -> GridTN1D:
        """ pad MPS on mps_grid onto self
        """
        mps = self.layout.pad_mps_to_grid(self.axes, scalar_field_mps, mps_axes)
        out_gtn = GridTN1D(self, mps)

        if self.layout_type is LayoutType.PARALLEL:
            out_gtn.compress(inplace=True)
        return out_gtn

    def pad_mpo_to_grid(self, scalar_field_mpo: 'MPOType', mpo_axes: tuple['Axis']) -> GridTN1D:
        """ pad MPO on mpo_grid onto self.grid
        """
        mpo = self.layout.pad_mpo_to_grid(self.axes, scalar_field_mpo, mpo_axes)
        out_gtn = GridTN1D(self, mpo)

        if self.layout_type is LayoutType.PARALLEL:
            out_gtn.compress(inplace=True)
        return out_gtn

    def apply_elemental_multiply_op(self, mps_gtn: 'GridTN1D', axes=None, upper_ind_id: str = 'o({})',
                                    lower_ind_id: str = 'i({})', site_tag_id: str = 'T({})',
                                    compress=False, compress_opts=None) -> 'GridTN':
        """
        """
        mps = mps_gtn.data

        if mps.site_ind_id == upper_ind_id:
            mps = mps.copy()
            mps.site_ind_id = mps.site_ind_id + '_tmp'

        if mps.site_ind_id == lower_ind_id:
            mps.copy()
            mps.site_ind_id = mps.site_ind_id + '_tmp'

        elem_mult_tn = self.get_elemental_multiply_tn(in1_ind_id=lower_ind_id, in2_ind_id=mps.site_ind_id,
                                                      out_ind_id=upper_ind_id, site_tag_id=site_tag_id,
                                                      x_axes=axes)

        new_mpo = qtn.TensorNetwork([])
        for x in range(self.L):
            tens_x1 = mps[x]
            tens_x2 = elem_mult_tn[x]
            new_tens = tens_x1.contract(tens_x2)  ## includes ancilla inds
            if tens_x1.tags != tens_x2.tags:
                new_tens.drop_tags(tags=tens_x1.tags)
            new_mpo.add(new_tens)

        new_mpo.fuse_multibonds(inplace=True)
        new_mpo = new_mpo.view_as(qtn.MatrixProductOperator, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                  site_tag_id=site_tag_id, inplace=True, cyclic=False, L=self.L)
        new_mpo.exponent = mps.exponent + elem_mult_tn.exponent

        if compress:
            helper.compress(new_mpo, compress_opts=compress_opts)

        return new_mpo

    def add_gtns(self, *mps_objs: Union['GridTN1D', Sequence['GridTN1D']], weights=None, zipup=True,
                 init_guess: 'qtn.MatrixProductState' = None, compress=True, sub_compress=True,
                 compress_type=CompressType.SVD, compress_opts=None, sub_compress_opts=None):
        """ add series of x, A*x using specified compress type
        """
        compress_opts = {} if compress_opts is None else compress_opts
        sub_compress_opts = {} if sub_compress_opts is None else sub_compress_opts
        weights = [1.0] * len(mps_objs) if weights is None else weights
        assert (len(weights) == len(mps_objs)), f'weight coefficients ({len(weights)})' + \
                                                f' needs to match mps objects ({len(mps_objs)})'

        if compress_type is CompressType.SVD:
            return super().add_gtns(*mps_objs, weights=weights, compress=compress, sub_compress=sub_compress,
                                    compress_type=compress_type, compress_opts=compress_opts,
                                    sub_compress_opts=sub_compress_opts)

        else:  ## DMRG or MG
            targets = []
            ref_gtn = None
            # print('weights', weights)
            for weight, item in zip(weights, mps_objs):
                if isinstance(item, GridTN1D):
                    assert (item.data_type is DataType.MPS), f'add gtn data type needs to MPS not {item.data_type}'
                    if item.data is not None:
                        targets += [helper.scalar_multiply(item.data, weight, inplace=False)]
                    init_guess = item.data if init_guess is None else init_guess
                    ref_gtn = item.copy() if ref_gtn is None else ref_gtn
                elif isinstance(item, qtn.MatrixProductState):
                    targets += [helper.scalar_multiply(item, weight, inplace=False)]
                    init_guess = item if init_guess is None else init_guess
                elif isinstance(item, (list, tuple)):
                    ## ordered how it's written:  [A, B, C, x]
                    ket = item[-1]
                    if isinstance(ket, GridTN1D):
                        assert (ket.data_type is DataType.MPS), \
                            f'add_gtn: last element in list needs to be MPS, not {ket.data_type}'
                        ref_gtn = ket.copy() if ref_gtn is None else ref_gtn
                        ket = ket.data
                        if ket is not None:
                            ket = helper.scalar_multiply(ket, weight, inplace=False)
                            init_guess = ket if init_guess is None else init_guess
                    elif isinstance(ket, qtn.MatrixProductState):
                        ket = helper.scalar_multiply(ket, weight, inplace=False)
                        init_guess = ket if init_guess is None else init_guess
                    else:
                        raise TypeError('incorrect ket data type', type(ket))

                    ops = []
                    for op in item[-2::-1]:
                        if isinstance(op, GridTN1D):
                            assert (op.data_type is DataType.MPO), \
                                f'add_gtn: last element in list needs to be MPS, not {ket.data_type}'
                            ops += [op.data]
                        elif isinstance(op, qtn.MatrixProductOperator):
                            ops += [op]
                        else:
                            raise TypeError('incorrect ket data type', type(op))

                    if ket is not None:
                        targets += [MatrixProductStateTN(mps=ket, mpos=ops)]

            # print('targets', targets)
            # print('target norms', [helper.norm(t) for t in targets])
            # print('init guess', helper.norm(init_guess), init_guess.max_bond())

            if compress_type is CompressType.DMRG:
                solver = helper_dmrg.LinearSolver(init_guess, targets=targets,
                                                  max_bond=compress_opts.get('max_bond', None))
            else:
                raise TypeError('not valid compress type', compress_type)

            solver.solve(2)
            # print('solver', solver.err, solver.is_conv)
            out_gtn = ref_gtn.create_like(new_data=solver.ket.view_as(qtn.MatrixProductState))
            return out_gtn

    def build_indexed_gtn(self, dict_gtns: dict[any, 'GridTN'], index_order=None) -> 'GridTN':
        """ builds a gtn from provided gtns in dict_gtns
            adds tensor that indexes each gtn as specified by the key
        """
        if index_order is None:
            index_order = dict_gtns.keys()

        num_inds = len(index_order)
        num_items = len(dict_gtns.keys())
        # print('num inds', num_inds, num_items)
        ind_tens = np.zeros((num_inds, num_inds))

        is_mpo = True
        tot_gtn: 'GridTN1D' = None
        it = 0
        for ind, gtn in dict_gtns.items():

            if gtn is None or gtn.data is None:
                num_items -= 1
                continue

            ind_position = list(index_order).index(ind)
            ind_tens[ind_position, it] = 1.0
            is_mpo = is_mpo and gtn.data_type is DataType.MPO

            gtn = gtn.copy()
            gtn_tens0 = gtn.get_anchor_tens()
            gtn_tens0.new_ind('ind_anc', 1, axis=0)

            if tot_gtn is None:
                tot_gtn = gtn
            else:
                tot_gtn = tot_gtn.add(gtn, compress=True)

            it += 1

        ind_tens = ind_tens[:, :num_items]

        if tot_gtn is None:
            return None

        ### add first tensor
        index_tensor_ind = 0 if tot_gtn.get_anchor_ind() == 0 else tot_gtn.L + 1
        if is_mpo:
            if index_tensor_ind == 0:
                tot_gtn_data = helper.renumber_mpo(tot_gtn.data, list(range(tot_gtn.L)), list(range(1, tot_gtn.L + 1)))
            else:
                tot_gtn_data = tot_gtn.data.copy()
            sq_num_inds = int(np.round(np.sqrt(num_inds)))
            ind_tensor = qtn.Tensor(ind_tens.reshape(sq_num_inds, sq_num_inds, -1),
                                    inds=(tot_gtn_data.upper_ind_id.format(index_tensor_ind),
                                          tot_gtn_data.lower_ind_id.format(index_tensor_ind), 'ind_anc'),
                                    tags=(tot_gtn_data.site_tag_id.format(index_tensor_ind),))
        else:
            if index_tensor_ind == 0:
                tot_gtn_data = helper.renumber_mps(tot_gtn.data, list(range(tot_gtn.L)), list(range(1, tot_gtn.L + 1)))
            else:
                tot_gtn_data = tot_gtn.data.copy()
            ind_tensor = qtn.Tensor(ind_tens,
                                    inds=(tot_gtn_data.site_ind_id.format(index_tensor_ind), 'ind_anc'),
                                    tags=(tot_gtn_data.site_tag_id.format(index_tensor_ind),))

        tot_gtn_data.add(ind_tensor)
        tot_gtn_data._L = tot_gtn_data.num_tensors
        # print('tot gtn exponent', tot_gtn.exponent)
        # print('final tot gtn data', tot_gtn_data)

        tot_gtn.data = tot_gtn_data
        return tot_gtn

    def select_indexed_gtn(self, gtn: 'GridTN', select_ind: Union[int, tuple[int]]) -> 'GridTN':
        """ selects a subgtn from gtn
        """
        L = gtn.data.L
        gtn = gtn.copy()
        index_tensor = gtn.get_anchor_tens()
        index_ind = gtn.get_anchor_ind()
        if gtn.data_type is DataType.MPS:
            index_tensor.isel({gtn.data.site_ind_id.format(index_ind): select_ind}, inplace=True)
        elif gtn.data_type is DataType.MPO:
            index_tensor.isel({gtn.data.upper_ind_id.format(index_ind): select_ind[0]}, inplace=True)
            index_tensor.isel({gtn.data.lower_ind_id.format(index_ind): select_ind[1]}, inplace=True)
        # print('index tensor isel', index_tensor.data)

        next_ind = index_ind + 1 if index_ind == 0 else index_ind - 1
        sub_gtn_data = gtn.data.contract((gtn.data.site_tag_id.format(index_ind),
                                          gtn.data.site_tag_id.format(next_ind)), inplace=False)

        # print('subgtn data', sub_gtn_data)

        if index_ind == 0:
            if gtn.data_type is DataType.MPO:
                helper.renumber_mpo(sub_gtn_data, list(range(1, L)), list(range(L - 1)), inplace=True)
            elif gtn.data_type is DataType.MPS:
                helper.renumber_mps(sub_gtn_data, list(range(1, L)), list(range(L - 1)), inplace=True)
        else:
            sub_gtn_data[next_ind].drop_tags(gtn.data.site_tag_id.format(index_ind))

        sub_gtn_data._L = sub_gtn_data.num_tensors
        sub_gtn_data.exponent = gtn.data.exponent

        if helper.norm(sub_gtn_data) < np.sqrt(CUTOFF):
            sub_gtn_data = None

        sub_gtn = gtn.create_like(sub_gtn_data)
        return sub_gtn

    ####################
    ### dmrg methods ###
    ####################

    def gtn_to_dmrg_format(self, gtn, is_mps=True):
        return gtn.data

    def dmrg_to_gtn_format(self, gtn, soln, is_mps=True, inplace=True):
        gtn = gtn if inplace else gtn.copy()
        gtn.data = soln
        return gtn

    def _setup_dmrg_solver(self, gtn: 'GridTN1D', operator: 'GridTN1D', compress_type: 'CompressType', inplace=False,
                           compress_opts=None, is_H=False, init_guess: 'GridTN' = None, verbose_output=False, **kwargs
                           ):
        """ set up solver for DMRG LinearSolve
        """
        from helper_dmrg import LinearSolver

        max_bond = compress_opts.get('max_bond', None) if compress_opts is not None else None
        init_guess_data = init_guess.data if init_guess is not None else None
        grid_mpx1 = self if inplace else self.copy()

        solver = LinearSolver(init_guess_data, targets=[grid_mpx1.data.copy()], operators=[operator.data], is_H=is_H,
                              max_bond=max_bond, **kwargs)

        return solver

    def _extract_dmrg_soln(self, gtn: 'GridTN', soln, inplace=False):
        gtn = gtn if inplace else gtn.copy()
        gtn.data = soln
        return gtn
