"""Grid: an n-dimensional grid assembled from :class:`~axis.Axis` objects.

A :class:`Grid` couples an ordered tuple of axes with a layout (the geometry used to
represent the high-dimensional tensor network) and exposes the multidimensional
differential operators (gradient, Laplacian, cross product) and common elementwise /
coordinate ("x") multiplications. Concrete layouts live in :mod:`grid1D` (1-D QTT) and
:mod:`grid_comb` (comb / tree-like layout).
"""

import matplotlib.pyplot as plt
from scipy import linalg as spla
from dataclasses import dataclass

from setup_.configs import *

import pickle
import helper_quimb as helper
# import helper_inverse as helper_inv
from axis import Axis
from gridTN import GridTN

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional
    from coord.coord_sys import Coordinate
    from gridTN_1D import GridTN1D

    GTN_TYPE = 'GridTN' or GridTN1D
    from field import Field, ScalarField
    # from coord.coord_sys import CoordinateSystem

""" Defines the multi-dimensional grid, built from Axis objects
    layout is the geometry used to represent high-dimensional TNs
    contains information like those involving differential operators (laplacian, cross-product)
    and common mathematical operations (like elemental multiply and xmultiply
"""


@dataclass(unsafe_hash=True)
class Grid:
    _frozen = False

    def __init__(self, gridID, axes: Iter['Axis']):  # , ax_deriv_configs=None):
        # axis_coordinate_systems: dict[Any, 'CoordinateSystem'] = None):
        """
        Inputs:
            axesID:  ID to use for this GridLayout object
            axes: list of Axis objects. CAN REPEAT THE SAME OBJECT?
            axis_coordinate_systems:  dict in which axIDs key a CoordinateSystem
                in which the corresponding Axis belongs

        Attributes:
            gridID: ID for this GridLayout object
            axes:   list of Axis objects
            ndim:   number of Axis objects
        """

        self.gridID = gridID
        self.axes: tuple['Axis'] = tuple(axes)
        self.ndim = len(axes)
        self.npts = np.prod([ax.npts for ax in axes])
        self.axIDs = tuple([ax.coordinate for ax in self.axes])

        ## to calculate
        self._mpo_xmultiply = {}
        self._tn_elem_multiply = {}
        self._mpo_firstderivatives = {}
        self._mpo_secondderivatives = {}
        self._mpo_mth_derivatives = {}
        self._mpx_integrals = {}
        self._mpo_integrals_indef = {}
        self._mpo_inverses = {}
        self._mpo_laplacian = {}

        self._frozen = True

    def __str__(self):
        return 'Grid(' + self.gridID + ',' + str(self.ndim) + ')'

    def __repr__(self):
        out_str = 'Grid('
        for ax in self.axes:
            out_str += f'{ax}'
        out_str += ')'
        return out_str

    def __delattr__(self, item):
        if self._frozen:
            raise AttributeError('not allowed to alter Grid attributes')
        object.__delattr__(self, item)

    def __setattr__(self, key, value):
        if self._frozen:
            raise AttributeError('not allowed to alter Grid attributes')
        object.__setattr__(self, key, value)

    def __eq__(self, other: 'Grid'):
        if DEEP_GRID_CHECK:
            return self is other
        else:
            return str(self) == str(other)
        # print('eq', (self.gridID, self.ndim, self.npts, self.axIDs))
        # print('eq', (other.gridID, other.ndim, other.npts, other.axIDs))
        # return (self.gridID, self.ndim, self.npts, self.axIDs) == \
        #        (other.gridID, other.ndim, other.npts, other.axIDs)

    @property
    def num_tensors(self) -> int:
        return int(np.sum([ax.L for ax in self.axes]))

    def get_inds_in_axis(self, ax, ax_ind=None) -> list:
        raise NotImplementedError

    def get_axis_ind(self, ax):
        return self.axes.index(ax)

    def shape(self) -> tuple:
        """ get grid shape
        """
        return tuple(ax.npts for ax in self.axes)

    def volume(self):
        """ get volume of space defined by these axes
        """
        return np.prod([ax.length() for ax in self.axes])

    def create_like(self, new_axes: Iter['Axis'] = None, new_gridID=None) -> 'Grid':
        """ create a new grid object with potentially new Axis objects and new gridID
        """
        if new_axes is None:
            new_axes = self.axes

        if new_gridID is None:
            new_gridID = self.gridID

        print('new axes', [ax.L for ax in new_axes])
        new_grid = self.__class__(new_gridID, new_axes)  # , self.ax_deriv_configs)
        return new_grid

    def create_like_from_axes(self, new_axes: Iter['Axis'] = None, new_gridID=None) -> 'Grid':
        return self.create_like(new_axes=new_axes, new_gridID=new_gridID)

    def copy(self, deep: bool = False, new_gridID=None) -> 'Grid':
        """ make a copy of these Axes. can redefine gridID while not making
            deep copies of the saved MPOs
        """
        new_grid = self.create_like(self.axes, new_gridID)

        if deep:
            new_grid._mpo_xmultiply = {k: mpo.copy() for k, mpo in self._mpo_xmultiply.items()}
            new_grid._mpo_firstderivatives = {k: mpo.copy() for k, mpo in self._mpo_firstderivatives}
            new_grid._mpo_secondderivatives = {k: mpo.copy() for k, mpo in self._mpo_secondderivatives}
            new_grid._mpo_mth_derivatives = {k: mpo.copy() for k, mpo in self._mpo_mth_derivatives}
            new_grid._mpx_integrals = {k: mpo.copy() for k, mpo in self._mpx_integrals}
        else:
            new_grid._mpo_xmultiply = self._mpo_xmultiply
            new_grid._mpo_firstderivatives = self._mpo_firstderivatives
            new_grid._mpo_secondderivatives = self._mpo_secondderivatives
            new_grid._mpo_mth_derivatives = self._mpo_mth_derivatives
            new_grid._mpx_integrals = self._mpx_integrals

        return new_grid

    def get_subgrid(self, select_axes, new_gridID=None):
        ordered_axes = [ax for ax in self.axes if ax in select_axes]
        return self.create_like(new_axes=ordered_axes, new_gridID=new_gridID)

    def has_basis_k_real(self):
        is_k_real = [ax.is_real_k() for ax in self.axes]
        return any(is_k_real)

    def is_k(self):
        """ is grid in Fourier space?
        """
        return all([ax.is_k() for ax in self.axes])

    def get_coarsen_inds(self, coarsen_level):
        """ get grid positions of tensors corresponding to desired coarsen_level
            smaller coarsen_level = finer grid
        """
        raise NotImplementedError

    def get_coarseness_levels(self):
        max_coarseness = np.max([ax.max_coarseness() for ax in self.axes])
        return {i: set(self.get_coarsen_inds(i)) for i in range(max_coarseness)}

    def get_axis_positions(self, sel_inds: Sequence[int], axes=None) -> dict['Axis', int]:
        """ get grid index along each axis from binary ind
        """
        out_dict = {}
        axes = self.axes if axes is None else axes
        for ax in axes:
            ax_inds = self.get_inds_in_axis(ax)
            # ax_bstr = '0b' + ''.join([str(sel_inds[ix]) for ix in ax_inds])
            # out_dict[ax] = int(ax_bstr, 2)
            out_dict[ax] = ax.get_inds_position([sel_inds[ix] for ix in ax_inds])
        return out_dict

    def get_indices_from_positions(self, ax_inds: dict['Axis', int]) -> list[int]:
        """ get binary ind corresponding to grid index along each axis (e.g., to select that index)
        """
        all_pos = []
        all_inds = []
        for ax in self.axes:
            all_pos += self.get_inds_in_axis(ax)
            all_inds += ax.get_position_inds(ax_inds[ax])

        sort_idxs = np.argsort(all_pos)
        inds = [all_inds[ix] for ix in sort_idxs]
        return inds


    def get_shifted_index_and_sign(self, sel_inds: Sequence[int], shifts: dict['Axis', 'int'],
                                   ax_deriv_configs: dict['Axis', 'DerivativeConfiguration']
                                   ) -> tuple[Sequence[int], int]:

        new_sign = 1

        ax_inds = self.get_axis_positions(sel_inds)
        # print('sel inds', sel_inds)
        # print('ax inds', ax_inds)
        new_ax_inds = {k: v for k, v in ax_inds.items()}

        for shift_ax, shift_amt in shifts.items():
            deriv_configs = ax_deriv_configs[shift_ax]
            # print('shift ax', shift_ax, deriv_configs)

            new_idx = ax_inds[shift_ax] + shift_amt

            left_bc, right_bc = deriv_configs.bc
            left_offset, right_offset = deriv_configs.offset, deriv_configs.offset_r
            npts = shift_ax.npts

            if new_idx < 0:
                if left_bc in [BCType.PERIODIC, BCType.ANTIPERIODIC]:
                    new_idx = npts + new_idx
                    new_sign *= np.sign(left_bc.value)
                elif left_bc in [BCType.ZEROGRADIENT, BCType.SYMMETRIC, BCType.NEUMANN]:
                    new_idx = -new_idx
                    new_sign *= 1
                elif left_bc in [BCType.ZEROVALUE, BCType.ANTISYMMETRIC, BCType.DIRICHLET]:
                    new_idx = -new_idx
                    new_sign *= 0 if left_offset == 0 else -1

            if new_idx >= npts:
                if right_bc in [BCType.PERIODIC, BCType.ANTIPERIODIC]:
                    new_idx = new_idx - npts
                    new_sign *= np.sign(right_bc.value)
                elif right_bc in [BCType.ZEROGRADIENT, BCType.SYMMETRIC, BCType.NEUMANN]:
                    new_idx = 2 * npts - new_idx
                    new_sign *= 1
                elif right_bc in [BCType.ZEROVALUE, BCType.ANTISYMMETRIC, BCType.DIRICHLET]:
                    new_idx = 2 * npts - new_idx
                    new_sign *= 0 if right_offset == 0 else -1

            new_ax_inds[shift_ax] = new_idx

        new_sel_inds = self.get_indices_from_positions(new_ax_inds)

        return new_sel_inds, new_sign

    ###########################
    ## extract info from MPS ##
    ###########################

    def meas_elem(self, ket: 'qtn.MatrixProductState', ind: int):
        """ measure the element in ket as indexed by ind
        """
        select_mps = self.get_select_elems_mps([ind])
        return helper.ovlp(ket, select_mps.data)


    def meas_elem_bin(self, ket: 'qtn.MatrixProductState', sel_inds: Sequence[int]):
        """ measure the element in ket as indexed by ind
        """
        select_mps = self.get_select_elem_mps_bin(sel_inds)
        return helper.ovlp(ket, select_mps.data)


    ################
    ## common TNs ##
    ################

    def make_empty_gridTN(self, ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None) -> 'GridTN':
        return GridTN(self)

    def make_zero_gridTN(self, ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None) -> 'GridTN':
        raise NotImplementedError

    def make_gridTN(self, data=None, ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None) -> 'GridTN':
        """ make GTN given appropriate data
        """
        return GridTN(self, data, ax_deriv_configs)

    def load_gtn_data(self, fstr, gtn=None, ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None,
                      safe_pass=False):
        """ load data from fstr, put into gtn (or empty GridTN)
        """
        # print('loading data', fstr + '.pkl')
        # gtn_data = pickle.load(open(fstr + '.pkl', 'rb'))
        # print('found data')
        try:
            print('loading data', fstr + '.pkl')
            gtn_data = pickle.load(open(fstr + '.pkl', 'rb'))
            print('found data')
        except IOError:
            print('data not found', fstr + '.pkl')
            if not safe_pass:
                raise IOError
            return None

        if gtn is None:
            gtn = self.make_empty_gridTN(ax_deriv_configs=ax_deriv_configs)
        gtn.data = gtn_data
        print('loaded data', fstr + '.pkl')
        return gtn

    def load_field_data(self, fstr, field_obj: 'Field', compIDs: Optional[list['Coordinate']] = None,
                        comp_ax_deriv_configs: dict['Coordinate', dict['Axis', 'DerivativeConfiguration']] = None,
                        safe_pass=True):
        """ load data from fstr, put into field (Field object)
        """
        compIDs = field_obj.componentIDs if compIDs is None else compIDs

        for compID in compIDs:
            comp_gtn = field_obj[compID]
            ax_deriv_configs = comp_ax_deriv_configs[compID] if comp_ax_deriv_configs is not None else None
            comp_gtn = self.load_gtn_data(fstr + str(compID), gtn=comp_gtn,
                                          ax_deriv_configs=ax_deriv_configs, safe_pass=safe_pass)
            if comp_gtn is not None:
                field_obj[compID] = comp_gtn
        # print('loaded data', fstr)
        return field_obj

    def load_scalar_field_data(self, fstr, scalar_field_obj: 'ScalarField',
                               ax_deriv_configs: dict['DerivativeConfiguration'] = None):
        """ load data from fstr, put into gtn (or empty GridTN)
        """
        comp_gtn = scalar_field_obj.component
        comp_gtn = self.load_gtn_data(fstr, gtn=comp_gtn, ax_deriv_configs=ax_deriv_configs)
        scalar_field_obj.component = comp_gtn
        # print('loaded data', fstr)
        return scalar_field_obj

    ############################

    def get_ones_mps(self, site_ind_id='i({})', site_tag_id='X({})') -> 'GridTN':
        """ gtn_cls: GridTN class type
            build ones vector mps
        """
        mps_dict = {ax: ax.get_iden_mps(site_ind_id, site_tag_id) for ax in self.axes}
        return self.make_mps_ndim(mps_dict)
        # return GridTN(self, data=[mps_dict])

    def get_iden_mps(self, site_ind_id='i({})', site_tag_id='X({})') -> 'GridTN':
        """ gtn_cls: GridTN class type
            build ones vector mps
        """
        return self.get_ones_mps(site_ind_id, site_tag_id)

    def get_iden_mpo(self, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> 'GridTN':
        """ build identity mpo
        """
        mpo_dict = {ax: ax.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id) for ax in self.axes}
        return self.make_mpo_ndim(mpo_dict)
        # return GridTN(self, data=[mpo_dict])

    def get_select_elems_mps(self, inds: Iter[int], site_ind_id='i({})', site_tag_id='X({})',
                             ax_deriv_config: dict['Axis', 'DerivativeConfiguration']=None) -> 'GridTN':
        """ build MPO to select certain elements specified by inds
        """
        ax_deriv_config = {} if ax_deriv_config is None else ax_deriv_config
        assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
        mps_dict = {ax: ax.get_select_elems_mps([ix], site_ind_id, site_tag_id,
                                                deriv_config=ax_deriv_config.get(ax,None))
                    for ax, ix in zip(self.axes, inds)}
        return self.make_mps_ndim(mps_dict)
        # return GridTN(self, data=[mps_dict])

    def get_select_elem_mps_bin(self, inds: Iter[int], site_ind_id='i({})', site_tag_id='X({})') -> 'GridTN':
        """ build MPS to select certain elements specified by inds
        """
        raise NotImplementedError

    def get_select_elems_mpo(self, inds: Iter[int], upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') \
            -> 'GridTN':
        """ build MPO to select certain elements specified by inds
        """
        assert (self.ndim == len(inds)), 'number of inds needs to match number of dims'
        mpo_dict = {ax: ax.get_select_elems_mpo([ix], upper_ind_id, lower_ind_id, site_tag_id)
                    for ax, ix in zip(self.axes, inds)}
        return self.make_mpo_ndim(mpo_dict)
        # return GridTN(self, data=[mpo_dict])
        # return gtn_cls(self, data=self.make_mps_ndim(mps_dict))

    def get_shift_mpo(self, shifts: dict['Axis', int],
                      ax_boundary_conditions: dict['Axis', 'DerivativeConfiguration'] = None,
                      upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})') -> 'GridTN':
        """ build MPO to shift elements along each axes by specified amount
        """
        mpo_dict = {ax: ax.get_shift_mpo(shifts[ax], boundary_conditions=ax_boundary_conditions[ax],
                                         upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)
                    for ax in shifts.keys()}
        return self.make_mpo_ndim(mpo_dict)
        # return GridTN(self, data=[mpo_dict])

    def get_rand_state(self, max_bond):
        raise NotImplementedError

    #########################################
    ## convert between np.ndarray and MPX ##
    #########################################

    def map_state_to_mps(self, state: np.ndarray, site_ind_id='i({})', site_tag_id='T({})',
                         direction=0, split_opts=None,
                         ax_deriv_configs: Optional[dict['Axis', 'DerivativeConfiguration']] = None,
                         ancilla_right: tuple[int] = (), ancilla_right_inds: tuple[str] = (),
                         ancilla_left: tuple[int] = (), ancilla_left_inds: tuple[str] = (),
                         axes: Sequence['Axis'] = None
                         ) -> 'GridTN':
        """ convert state represented as vector converted to mps state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        raise NotImplementedError

    def map_mps_to_state(self, gtn_mps: 'GridTN', ax_select: dict['Axis', int] = None,
                         ancilla_right: tuple[int] = (), ancilla_right_inds: tuple[str] = (),
                         ancilla_left: tuple[int] = (), ancilla_left_inds: tuple[str] = ()
                         ) -> Union[np.ndarray, qtn.Tensor]:
        """ convert MPS into ndim-dim np.ndarray
        """
        raise NotImplementedError

    def map_operator_to_mpo(self, operator, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='B({})',
                            direction=0, split_opts=None,
                            ancilla_right: tuple[int] = (), ancilla_right_inds: tuple[str] = (),
                            ancilla_left: tuple[int] = (), ancilla_left_inds: tuple[str] = ()
                            ) -> 'GridTN':
        """ convert operator represented as high-dimensional matrix to mpo state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        raise NotImplementedError

    def map_mpo_to_operator(self, gtn_mpo: 'GridTN',
                            ax_select: dict['Axis', Union[int, tuple[int]]] = None,
                            ancilla_right: tuple[int] = (), ancilla_right_inds: tuple[str] = (),
                            ancilla_left: tuple[int] = (), ancilla_left_inds: tuple[str] = ()
                            ) -> Union[np.ndarray, qtn.Tensor]:
        """ convert MPO into ndim*2-dim np.ndarray (o0 o1 ... x i0 i1 ...)
        """
        raise NotImplementedError

    ###########################################
    ## convert low-dim MPS to full grid size ##
    ###########################################

    def make_mps_ndim(self, mps_1d_dict: dict['Axis', Union['np.ndarray', 'MPSType']]) -> 'GTN_TYPE':
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            if MPS not defined along that dimension, use a ones vector (constant along that dimension)
        """
        return GridTN(self, data=[mps_1d_dict])

    def make_mpx_ndim(self, mps_1d_dict: dict['Axis', Union['np.ndarray', 'MPSType']]) -> 'GTN_TYPE':
        """ combine 1-D MPSs into K-dimensional MPO
            if MPS not defined along that dimension, pad with the identity MPO
        """
        raise NotImplementedError

    def make_mpo_ndim(self, mpo_1d_dict: dict['Axis', Union['np.ndarray', 'MPOType']]) -> 'GTN_TYPE':
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        return GridTN(self, data=[mpo_1d_dict])

    def make_tn1d_ndim(self, tn_1d_dict: dict['Axis', Union['np.ndarray', 'TN1Type']], **kwargs) -> 'GTN_TYPE':
        """ combine 1-D MPSs or MPOs into a K-dimensional MPS
            for MPS, MPS needs to be defined for all dimensions.
        """
        return GridTN(self, data=[tn_1d_dict])

    def make_gtn_from_dicts(self, data: Iter[dict['Axis', Optional['TNType']]], data_type=DataType.MPO, compress=False,
                            compress_opts=None):
        """ return GridTN object
        """
        # out = self.make_empty_gridTN()
        out = None
        out_compress = compress  # False

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
                # raise TypeError(f'data is currently type {type(data_dict)}')

            if out is None:
                out = new_out  # don't compress bc input would have already been compressed
            else:
                out.add(new_out, inplace=True)
                if compress:
                    out = out.compress(inplace=True, compress_opts=compress_opts)
                out_compress = False  # compress

            if out_compress:
                out = out.compress(inplace=True, compress_opts=compress_opts)

        return out

    def pad_gtn_to_grid(self, gtn: 'GridTN', target_data_type=None):
        """ pad GridTN onto self
        """
        raise NotImplementedError

    def pad_mps_to_grid(self, scalar_field_mps: 'MPSType', mps_axes: tuple['Axis']):
        """ pad MPS on mps_grid onto self
        """
        raise NotImplementedError

    def pad_mpo_to_grid(self, scalar_field_mpo: 'MPOType', mpo_axes: tuple['Axis']):
        """ pad MPO on mpo_grid onto self
        """
        raise NotImplementedError

    def apply_elemental_multiply_op(self, mps_gtn: 'MPSType', axes=None, upper_ind_id: str = 'o({})',
                                    lower_ind_id: str = 'i({})', site_tag_id: str = 'T({})',
                                    compress=False, compress_opts=None) -> 'GridTN':
        """
        """
        raise NotImplementedError

    # @profile
    def build_mps_from_subgtns(self, data_list, compress_opts=None, **kwargs) -> 'GridTN':
        """ take outerproduct from gtns of subgrids
            data_list:  sequence of tuple(ax inds or axes, np.ndarray or MPS)
                                    or gridTN
        """

        def get_subgr_data(gtn_data):
            if isinstance(gtn_data, tuple):
                axes, gtn_data = gtn_data
                axes = [self.axes[i] if isinstance(i, int) else i for i in axes]
                sub_gr = self.get_subgrid(axes)
                if isinstance(gtn_data, np.ndarray):
                    gtn_mps = sub_gr.map_state_to_mps(gtn_data, axes=axes, split_opts=compress_opts, **kwargs)
                else:
                    gtn_mps = sub_gr.make_gridTN(gtn_data)
                out_gtn_mps = self.pad_gtn_to_grid(gtn_mps)
            elif isinstance(gtn_data, GridTN):
                out_gtn_mps = self.pad_gtn_to_grid(gtn_data)
            else:
                raise TypeError('not valid data in data_list')
            return out_gtn_mps

        out = get_subgr_data(data_list[0])
        if out is None:
            return out

        for data in data_list[1:]:
            out2: 'GridTN' = get_subgr_data(data)
            out2 = out2.apply_elemental_multiply_op()

            # ax_x, ax_y, ax_vx, ax_vy, ax_vz = out2.grid.axes
            # tmp_data = out2.get_data(ax_select={ax_x: ax_x.npts//2, ax_vx: ax_vx.npts//2, ax_vy: ax_vy.npts//2, ax_vz: ax_vz.npts//2})
            # plt.figure()
            # plt.imshow(tmp_data)
            # plt.show()

            # out2 = out2.mps_to_diag_mpo()     ## wrong for spectral bases
            # print('out2', out2.data)
            out = out.apply(out2, inplace=True, zipup=True, compress=True, compress_opts=compress_opts)

        out = out.compress(compress_opts=compress_opts)
        return out

    def build_mpo_from_subgtns(self, data_list, compress=False, zipup=False):
        """ take outerproduct from gtns of subgrids
            data_list:  sequence of tuple(ax inds or axes, np.ndarray or MPS) or gridTN
        """

        def get_subgr_data(gtn_data):
            if isinstance(gtn_data, tuple):
                axes, gtn_data = gtn_data
                axes = [self.axes[i] if isinstance(i, int) else i for i in axes]
                sub_gr = self.get_subgrid(axes)
                if isinstance(gtn_data, np.ndarray):
                    gtn_mpo = sub_gr.map_operator_to_mpo(gtn_data)
                else:
                    gtn_mpo = sub_gr.make_gridTN(gtn_data)
                out_gtn_mpo = self.pad_gtn_to_grid(gtn_mpo)
            elif isinstance(gtn_data, GridTN):
                out_gtn_mpo = self.pad_gtn_to_grid(gtn_data)
            else:
                raise TypeError('not valid data in data_list')
            return out_gtn_mpo

        out = get_subgr_data(data_list[0])
        if out is None:
            return out

        for data in data_list[1:]:
            out2: 'GridTN' = get_subgr_data(data)
            # out2 = out2.apply_elemental_multiply_op()
            # print('out', out)
            # print('out2', out2)
            out = out.apply(out2, inplace=False, zipup=zipup)

        # print('out', out)
        if compress:
            out = out.compress()
        # print('build from subgtns no compress')
        return out

    #######################################################
    ## taking inverse. can probably be moved elsewhere? ##
    #######################################################

    def get_mpo_inverse(self, mpo: 'GridTN', boundary_conditions=None, use_dmrg=False, compress_opts=None) -> 'GridTN':
        """ mpo:  operator to take the inverse of
            boundary condition:  at the moment, an m x q**K**L array such that
                                 for each m, vector.T * data = mth boundary condition
            expensive method right now -- take inverse directly
            optimization method:
        """
        npts = self.npts
        if compress_opts is None:   compress_opts = {}

        if use_dmrg:
            mpo = mpo.copy()

            print('get inverse dmrg')
            if boundary_conditions is not None:
                if isinstance(boundary_conditions, np.ndarray):
                    num_bc = len(boundary_conditions)
                    tens = np.zeros((npts,) * self.ndim * 2)
                    tens[:num_bc, :] += boundary_conditions
                    bc = self.map_operator_to_mpo(tens)
                    mpo = helper.add_MPO(mpo, bc, inplace=True, compress=True, **compress_opts)
                elif isinstance(boundary_conditions, list) or isinstance(boundary_conditions, tuple):
                    for bc in boundary_conditions:
                        if not isinstance(bc, qtn.MatrixProductOperator):
                            bc = self.map_operator_to_mpo(bc)
                        mpo = helper.add_MPO(mpo, bc, inplace=True, compress=True, **compress_opts)

            inverse_mpo = helper_inv.get_mpo_inverse_dmrg(mpo, verbose=1, cutoff=1.0e-15, max_bond=48)
            # **self.get_compress_opts(-1))

        else:
            if True:  # self.npts < 2 ** 12:  # catch for some matrix size
                tens = self.map_mpo_to_operator(mpo)
                tens_shape = tens.shape

                if boundary_conditions is not None:
                    if isinstance(boundary_conditions, np.ndarray):
                        tens = tens.reshape(npts, npts)
                        num_bc = len(boundary_conditions)
                        tens[:num_bc, :] = tens[:num_bc, :] + boundary_conditions
                    elif isinstance(boundary_conditions, (tuple, list)):
                        for bc in boundary_conditions:
                            if isinstance(bc, qtn.MatrixProductOperator):
                                bc = self.map_mpo_to_operator(bc)
                            tens += bc
                    elif isinstance(boundary_conditions,
                                    mpo.__class__) and boundary_conditions.data_type is DataType.MPO:
                        mpo = mpo.add(boundary_conditions, compress=True)
                        tens = self.map_mpo_to_operator(mpo)

                tens = tens.reshape(npts, npts)

                inverse = spla.pinv(tens)
                # ans = np.dot(inverse,tens)
                print('inverse', np.linalg.norm(np.dot(inverse, tens) - np.eye(len(tens))),
                      np.linalg.norm(np.dot(tens, inverse) - np.eye(len(tens))))
                # print('bc', boundary_conditions)

                inverse = inverse.reshape(tens_shape)
                print('inverse', inverse.shape)
                inverse_mpo = self.map_operator_to_mpo(inverse)  # inverse.reshape(self.shape() * 2))

                # check = inverse_mpo.apply(mpo)
                # check_data = self.map_mpo_to_operator(check).reshape(npts,npts)
                # print('inverse error',np.linalg.norm(np.eye(npts) - check_data))

                # check_eye = self.get_iden_mpo()
                # check_eye = check_eye.add( check.scalar_multiply(-1) )
                # check_eye_tens = check_eye.contract()
                # print('inverse error gtn', check_eye_tens.norm())

                print('inverse max bond', inverse_mpo.max_bond())
                # exit()
            else:
                raise NotImplementedError
                inverse_mpo = helper_inv.do_newton_schultz_inv(mpo, conv_tol=5.0e-7, num_it=500,
                                                               compress=2, max_bond=150, cutoff=1.0e-30)
        return inverse_mpo

    def get_mps_inverse(self, mps: 'GridTN', use_dmrg=False, compress_opts=None) -> 'GridTN':
        """ mps:  diagonal of mpo to take the inverse of
            boundary condition:  at the moment, an m x q**K**L array such that
                                 for each m, vector.T * data = mth boundary condition
            expensive method right now -- take inverse directly
            optimization method:
        """
        if compress_opts is None:   compress_opts = {}

        if use_dmrg:
            mps = mps.copy()
            print('get inverse dmrg')
            raise NotImplementedError
        else:
            tens = self.map_mps_to_state(mps)
            ## catch k=0 case; choose s.t k=0 value doesn't change
            tens_iszero = np.nonzero(np.abs(tens) / np.max(np.abs(tens)) < 1.0e-12)
            # tens[tens_iszero] = 1.0

            inverse = 1. / tens  # spla.pinv(tens)
            inverse[tens_iszero] = 0.0  ## sets dc component for fourier basis
            inverse_mps = self.map_state_to_mps(inverse, split_opts=compress_opts)

        return inverse_mps

    def get_diag_mpo_inverse(self, mps: 'GridTN', use_dmrg=False, compress_opts=None) -> 'GridTN':
        """ mps:  diagonal of mpo to take the inverse of
            boundary condition:  at the moment, an m x q**K**L array such that
                                 for each m, vector.T * data = mth boundary condition
            expensive method right now -- take inverse directly
            optimization method:
        """
        if compress_opts is None:   compress_opts = {}

        # if use_dmrg:
        #     mps = mps.copy()
        #
        #     print('get inverse dmrg')
        #
        #     raise NotImplementedError
        #     inverse_mpo = helper_inv.get_mps_inverse_dmrg(mps, verbose=1, cutoff=1.0e-15, max_bond=48)
        #     # **self.get_compress_opts(-1))
        #
        # else:
        #     # tens = self.map_mps_to_state(mps)
        #     # ## catch k=0 case; choose s.t k=0 value doesn't change
        #     # tens_iszero = np.nonzero(np.abs(tens) < 1.0e-12)
        #     # tens[tens_iszero] = 1.0
        #     #
        #     # inverse = 1./tens   # spla.pinv(tens)
        #     # inverse_mps = self.map_state_to_mps(inverse, split_opts=compress_opts)

        inverse_mps = self.get_mps_inverse(mps, use_dmrg=use_dmrg, compress_opts=compress_opts)
        inverse_mpo = inverse_mps.mps_to_diag_mpo()

        return inverse_mpo

    #####################
    ## post-processing ##
    #####################

    def interpolate_data(self):
        """ process the data in some way
        """
        pass

    def gaussian_smooth_mpo(self, sigma=0.3, proc_axes: Optional[Sequence['Axis']] = None, mode='wrap',
                            split_opts: dict = None) -> 'GTN_TYPE':
        """ apply gaussian smoothing to data along proc_axes (axes to process)
            mode = 'wrap' or 'mirror' or whatever other options
        """
        ## current method uses wrap bcs
        if proc_axes is None:             proc_axes = self.axes
        if isinstance(proc_axes, Axis):   proc_axes = [proc_axes]

        mpo1d_dict = {}

        for ax in proc_axes:

            ## smoothing MPO
            nx = max(1, int(np.round(4 * sigma)))
            nb = int(np.ceil(np.log2(nx * 2 + 1)))  ## number of bits req to encode gaussian smoothing
            xpts = nx * 2 + 2
            # print('processing',nb,mode)

            xs = np.arange(xpts) - xpts // 2
            blur = np.exp(-xs ** 2 / 2 / sigma ** 2)
            blur = blur / np.sum(blur)

            blur_mat = np.zeros((ax.npts, ax.npts))
            if mode == 'wrap':
                for x0 in range(xpts // 2):
                    blur_mat[x0, :xpts // 2 + x0] = blur[xpts // 2 - x0:]
                    blur_mat[x0, -xpts // 2 + x0:] = blur[:xpts // 2 - x0]
            elif mode == 'mirror':
                for x0 in range(xpts // 2):
                    blur_mat[x0, :xpts // 2 + x0] = blur[xpts // 2 - x0:]
                    blur_mat[x0, 2 * x0 + 1:xpts // 2 + x0 + 1] += blur[:xpts // 2 - x0][::-1]

            for x0 in range(ax.npts - xpts + 1):
                blur_mat[xpts // 2 + x0, x0:x0 + xpts] = blur

            if mode == 'wrap':
                # print('blurring mode = wrap?')
                for x0 in range(xpts // 2 - 1):
                    blur_mat[-x0 - 1, -xpts // 2 - x0 - 1:] = blur[:-xpts // 2 + x0 + 1]
                    blur_mat[-xpts // 2 + x0 + 1, :x0 + 1] = blur[-x0 - 1:]
            elif mode == 'mirror':
                for x0 in range(xpts // 2 - 1):
                    blur_mat[-x0 - 1, -xpts // 2 - x0 - 1:] = blur[:-xpts // 2 + x0 + 1]
                    blur_mat[-x0 - 1, -xpts // 2 - x0 - 1:-2 * x0 - 1] += blur[:-xpts // 2 - x0]

            blur_mpo = ax.map_operator_to_mpo(blur_mat, split_opts=split_opts)
            mpo1d_dict[ax] = blur_mpo

        gtn_mpo = self.make_gtn_from_dicts([mpo1d_dict], data_type=DataType.MPO)
        return gtn_mpo

    def absorbing_bc_mpo(self, x_ax: 'Axis', v_ax: 'Axis', left_bc: BCType, right_bc: BCType, compress=True,
                         compress_opts: dict = None) -> 'GTN_TYPE':
        """ apply absorbing bc to MPS state, assumes TN dimensions are x1,x2,x3,v1,v2,v3
            (prev: apply absorbing bc to derivative MPO, assumes TN dimensions are x1,x2,x3,v1,v2,v3)
            returns list of MPO dicts to add together for final MPO
        """
        mpo_list = []

        v_vals = v_ax.xpts
        zero_ind = np.argmax(v_vals >= 0)
        if zero_ind == 0:
            if np.all(v_vals < 0):
                zero_ind = v_ax.npts
            elif np.all(v_vals > 0):
                zero_ind = -1

        if left_bc == BCType.ABSORBING:

            if zero_ind < 0:  ## all v positive (away from left boundary), nothing absorbed
                absorb_bc_L = None
            elif zero_ind == v_ax.npts:  ## all v negative (towards boundary), all absorbed
                mpo_L = x_ax.get_select_elems_mpo([0])
                helper.scalar_multiply(mpo_L, -1, inplace=True)
                absorb_bc_L = {x_ax: mpo_L}
            else:
                v_neg = v_ax.get_heaviside_mpo(zero_ind, get='left')  ## inds <= zero_ind
                mpo_L = x_ax.get_select_elems_mpo([0])
                helper.scalar_multiply(mpo_L, -1, inplace=True)
                absorb_bc_L = {x_ax: mpo_L, v_ax: v_neg}

            mpo_list += [absorb_bc_L]

        if right_bc == BCType.ABSORBING:

            if zero_ind == v_ax.npts:  ## all v negative (away from right boundary), nothing absorbed
                absorb_bc_R = None
            elif zero_ind < 0:  ## all v positive (towards boundary), all absorbed
                mpo_R = x_ax.get_select_elems_mpo([x_ax.npts - 1])
                helper.scalar_multiply(mpo_R, -1, inplace=True)
                absorb_bc_R = {x_ax: mpo_R}
            else:
                v_pos = v_ax.get_heaviside_mpo(zero_ind, get='right')  ## inds >= zero_ind
                mpo_R = x_ax.get_select_elems_mpo([x_ax.npts - 1])
                helper.scalar_multiply(mpo_R, -1, inplace=True)
                absorb_bc_R = {x_ax: mpo_R, v_ax: v_pos}

            mpo_list += [absorb_bc_R]

        if len(mpo_list) > 0:
            iden_mpo = {x_ax: x_ax.get_iden_mpo(), v_ax: v_ax.get_iden_mpo()}
            mpo_list += [iden_mpo]
            gtn_mpo = self.make_gtn_from_dicts(mpo_list, data_type=DataType.MPO)
        else:
            gtn_mpo = None

        return gtn_mpo

    def reflecting_v_bc_mpo(self, x_ax: 'Axis', v_ax: 'Axis', left_bc: BCType, right_bc: BCType, compress=True,
                            compress_opts: dict = None) -> 'GTN_TYPE':
        """ reflecting_v bc MPO that takes v -> -v
        """
        mpo_list = []
        except_inds = []  # 0, npts-1 for left, right boundary

        v_vals = v_ax.xpts  # corresponding velocity axis
        vpos_0 = np.argmin(v_vals >= 0)  # location of the zero in v_ax
        vneg_0 = np.nonzero(np.abs(v_vals - (-1 * v_vals[vpos_0])) < 1.0e-12)[0]

        if left_bc == BCType.REFLECTING:

            v_neg_to_pos = np.zeros((len(v_vals),) * 2)
            for i in range(len(v_vals) - vpos_0 + 1):
                try:
                    v_neg_to_pos[vpos_0 + i, vneg_0 - i] = 1.0
                except IndexError:
                    break

            if not np.all(v_neg_to_pos == 0):
                v_mpo = v_ax.map_operator_to_mpo(v_neg_to_pos, split_opts=compress_opts)
                mpo_0 = x_ax.get_select_elems_mpo([0])
                reflect_bc_L = {x_ax: mpo_0, v_ax: v_mpo}
                except_inds += [0]
                mpo_list += [reflect_bc_L]

        if right_bc == BCType.REFLECTING:

            v_pos_to_neg = np.zeros((len(v_vals),) * 2)
            for i in range(len(v_vals) - vpos_0 + 1):
                try:
                    v_pos_to_neg[vneg_0 - i, vpos_0 + i] = 1.0
                except IndexError:
                    break

            if not np.all(v_pos_to_neg == 0):
                v_mpo = v_ax.map_operator_to_mpo(v_pos_to_neg, split_opts=compress_opts)
                mpo_R = x_ax.get_select_elems_mpo([x_ax.npts - 1])
                reflect_bc_R = {x_ax: mpo_R, v_ax: v_mpo}
                except_inds += [x_ax.npts - 1]
                mpo_list += [reflect_bc_R]

        if len(except_inds) > 0:
            except_iden = {x_ax: x_ax.get_all_except_elem_mpo(except_inds),
                           v_ax: v_ax.get_iden_mpo()}
            mpo_list += [except_iden]

        gtn_mpo = self.make_gtn_from_dicts(mpo_list, data_type=DataType.MPO)
        return gtn_mpo

    ###############################################
    ### build derivative and integral operators ###
    ###############################################

    def get_xmultiply_mpo(self, x_axes: Sequence['Axis'] or None = None, x_power=1,
                          offsets: dict['Axis', Numeric] = None, scales: dict['Axis', Numeric] = None,
                          recalc=False, store=True, compress_opts: dict = None) -> 'GTN_TYPE':
        """ multiply mps by axis value  (x * f(x))
            x_axes: axes over which operation is applied
        """
        if x_axes is None:   x_axes = self.axes
        # axIDs = tuple([ax.axID for ax in x_axes])

        if isinstance(offsets, int) or isinstance(offsets, float):
            offsets_dict = {ax: offsets for ax in x_axes}
            offsets_ = offsets
        elif isinstance(offsets, dict):
            offsets_dict = {}
            offsets_ = []
            for ax in x_axes:
                try:
                    offsets_dict[ax] = offsets[ax]
                except KeyError:
                    offsets_dict[ax] = 0.0
                offsets_ += [offsets_dict[ax]]
            offsets_ = tuple(offsets)
        else:  # if offsets is None:
            offsets_dict = {ax: 0.0 for ax in x_axes}
            offsets_ = offsets

        if isinstance(scales, int) or isinstance(scales, float):
            scales_dict = {ax: scales for ax in x_axes}
            scales_ = scales
        elif isinstance(scales, dict):
            scales_dict = {}
            scales_ = []
            for ax in x_axes:
                try:
                    scales_dict[ax] = scales[ax]
                except KeyError:
                    scales_dict[ax] = 1.0
                scales += [scales_dict[ax]]
            scales_ = tuple(scales_)
        else:  # scales is None:
            scales_dict = {ax: 1.0 for ax in x_axes}
            scales_ = scales

        key = (tuple(x_axes), x_power, offsets_, scales_)

        try:
            if recalc:   raise KeyError
            return self._mpo_xmultiply[key].copy()
        except KeyError:
            mpo = {ax: ax.get_xmultiply_mpo(x_power=x_power, offset=offsets_dict[ax], scale=scales_dict[ax],
                                            split_opts=compress_opts)
                   for ax in x_axes}
            gtn_mpo = self.make_gtn_from_dicts([mpo], data_type=DataType.MPO)
            if store:
                self._mpo_xmultiply[key] = gtn_mpo
            print('xmultiply bond dimension', gtn_mpo.max_bond())
            # print('xmultiply singular values', helper.singular_values_all(gtn_mpo.data))
            return gtn_mpo.copy()

    def get_elemental_multiply_tn(self, x_axes: Sequence['Axis'] = None, recalc=False, store=True,
                                  in1_ind_id='i({})[1]', in2_ind_id='i({})[2]', out_ind_id='o({})',
                                  site_tag_id='d_ijk({})') -> 'GridTN':
        """ multiply mps by axis value  (x * f(x))
            x_axes: axes over which operation is applied
        """
        if x_axes is None:   x_axes = self.axes
        # axIDs = tuple([ax.axID for ax in x_axes])

        try:
            if recalc:   raise KeyError
            gtn_tn3 = self._tn_elem_multiply[frozenset(x_axes)].copy()
            tn3_ind_ids = set((gtn_tn3.data.upper_ind_id, gtn_tn3.data.lower_ind_id,) + gtn_tn3.data.extra_ind_ids)
            new_ind_ids = {in1_ind_id, in2_ind_id, out_ind_id}

            tmp = new_ind_ids.intersection(tn3_ind_ids)
            if len(tmp) > 0:
                gtn_tn3.data.upper_ind_id = gtn_tn3.data.upper_ind_id + 'TMP'
                gtn_tn3.data.lower_ind_id = gtn_tn3.data.lower_ind_id + 'TMP'
                gtn_tn3.data.reindex_extra_inds(0, gtn_tn3.data.extra_ind_ids[0] + 'TMP')

            if out_ind_id != gtn_tn3.data.upper_ind_id:
                store = True
                gtn_tn3.data.upper_ind_id = out_ind_id
            if in1_ind_id != gtn_tn3.data.lower_ind_id:
                store = True
                gtn_tn3.data.lower_ind_id = in1_ind_id
            if in2_ind_id != gtn_tn3.data.extra_ind_ids[0]:
                store = True
                gtn_tn3.data.reindex_extra_inds(0, in2_ind_id)

            if store:
                self._tn_elem_multiply[frozenset(x_axes)] = gtn_tn3

            return gtn_tn3

        except KeyError:
            tns = {ax: ax.get_elemental_multiply_tn(in1_ind_id, in2_ind_id, out_ind_id, site_tag_id)
                   for ax in x_axes}
            gtn_obj = self.make_tn1d_ndim(tns, in1_ind_id=in1_ind_id, in2_ind_id=in2_ind_id, out_ind_id=out_ind_id,
                                          site_tag_id=site_tag_id)
            if store:  self._tn_elem_multiply[frozenset(x_axes)] = gtn_obj
            print('elem multiply tn bond dimension', gtn_obj.max_bond())
            return gtn_obj.copy()

    def get_firstderivative_mpo(self, ax: 'Axis', deriv_config: 'DerivativeConfiguration' = None,
                                upwind_ax: Optional['Axis'] = None,
                                recalc=False, store=True) -> 'GTN_TYPE':
        #  bc: tuple[BCType] or BCType = DEFAULT_BC, order=DEFAULT_ORDER, fd_type=DEFAULT_FDTYPE,
        #  upwind=False, v_ax:Axis=None
        """ obtain MPO taking first derivative
            ax:  Axis objects to take derivative. if ax2 is None, take second deriv along ax1
            bc:  BCType or Tuples of BCTypes (left, right boundary conditions)
            order: finite difference expansion order
            extra_params:  v_ax for upwind
        """
        if deriv_config is None:
            deriv_config = DerivativeConfiguration()

        # fd_type = deriv_config.fd_type
        key = (ax,) + deriv_config.get_deriv_key() + (upwind_ax,)
        ## key = (ax.axID, bc, order, fd_type, [v_ax.axID])

        # print('grid 1st deriv', self, key)

        try:
            if recalc:   raise KeyError
            return self._mpo_firstderivatives[key].copy()
        except KeyError:
            if isinstance(upwind_ax, Axis):  # fd_type == FDType.UPWIND:
                print('upwind first deriv', upwind_ax)
                gtn_mpo = self._build_mpo_firstderivative_upwind(ax, upwind_ax, deriv_config)
                print('upwind gtn_mpo', gtn_mpo.max_bond())
            else:
                print('normal first deriv')
                coordsys = ax.coord_sys
                mpo_dict = coordsys.build_firstderivative_mpo(ax, deriv_config)
                helper.compress(mpo_dict[ax], scale=True)
                # gtn_mpo = self.make_gtn_from_dicts([mpo_dict], data_type=DataType.MPO)
                gtn_mpo = self.make_mpo_ndim(mpo_dict)
                # gtn_mpo.compress(inplace=True)
                print('first deriv gtn_mpo', gtn_mpo.max_bond())

            if store:   self._mpo_firstderivatives[key] = gtn_mpo

            # print('ax', ax, deriv_config)
            # if ax == self.axes[0]:
            #     print(np.round(gtn_mpo.get_data()[:10, 0, :10, 0],4))
            #     print(np.round(gtn_mpo.get_data()[-10:, 0, -10:, 0],4))
            # elif ax == self.axes[1]:
            #     print(np.round(gtn_mpo.get_data()[0, :10, 0, :10],4))
            #     print(np.round(gtn_mpo.get_data()[0, -10:, 0, -10:,],4))

            return gtn_mpo.copy()

    def get_secondderivative_mpo(self, ax1: 'Axis', ax2: 'Axis' or None,
                                 deriv_config1: 'DerivativeConfiguration' = None,
                                 deriv_config2: 'DerivativeConfiguration' = None,
                                 recalc=False, store=True) -> 'GTN_TYPE':
        """ obtain MPO taking second derivative
            ax1, ax2:  Axis objects to take derivative. if ax2 is None, take second deriv along ax1
            bc1, bc2:  BCType or Tuples of BCTypes (left, right boundary conditions)
            order: finite difference expansion order
        """
        if deriv_config1 is None:   deriv_config1 = DerivativeConfiguration()
        if deriv_config2 is None:   deriv_config2 = deriv_config1
        if ax2 is None:             ax2 = ax1

        key1 = (ax1,) + deriv_config1.get_deriv_key()
        key2 = (ax2,) + deriv_config2.get_deriv_key()
        # (ax1.axID, bc1, order1, fd_type1)
        # (ax2.axID, bc2, order2, fd_type2)
        try:
            if recalc:   raise KeyError
            return self._mpo_secondderivatives[(key1, key2)].copy()
        except KeyError:
            coordsys1 = ax1.coord_sys
            coordsys2 = ax2.coord_sys

            if coordsys1 is coordsys2 or coordsys2 is None:
                mpo_dict = coordsys1.build_secondderivative_mpo(ax1, None, deriv_config1, None)
            else:
                mpo_dict1 = coordsys1.build_firstderivative_mpo(ax1, deriv_config1)
                mpo_dict2 = coordsys2.build_firstderivative_mpo(ax2, deriv_config2)
                mpo_dict = {**mpo_dict1, **mpo_dict2}
                # mpo_dict = {ax1.axID: mpo1[ax1.axID], ax2.axID: mpo2[ax2.axID]}
                # mpo = self.make_mpo_ndim({ax1.axID: mpo1, ax2.axID: mpo2})

            gtn_mpo = self.make_gtn_from_dicts([mpo_dict], data_type=DataType.MPO)
            if store:   self._mpo_secondderivatives[(key1, key2)] = gtn_mpo

            # print('second deriv')
            # plt.figure()
            # ax_y = self.axes[1]
            # plt.imshow(gtn_mpo.get_data(ax_select={ax_y: (ax_y.npts//2, ax_y.npts//2)}))
            # plt.colorbar()
            # plt.title(f'second deriv {ax1}')
            # plt.show()

            return gtn_mpo.copy()


    def get_mth_derivative_mpo(self, ax: 'Axis', deriv_order: int, deriv_config: 'DerivativeConfiguration' = None,
                               recalc=False, store=True) -> 'GTN_TYPE':
        """ obtain MPO taking second derivative
            ax1, ax2:  Axis objects to take derivative. if ax2 is None, take second deriv along ax1
            bc1, bc2:  BCType or Tuples of BCTypes (left, right boundary conditions)
            order: finite difference expansion order
        """
        if deriv_config is None:   deriv_config = DerivativeConfiguration()

        key1 = (ax, deriv_order,) + deriv_config.get_deriv_key()
        try:
            if recalc:   raise KeyError
            return self._mpo_mth_derivatives[(key1,)].copy()
        except KeyError:
            # coordsys1 = ax.coord_sys

            mpo_dict = {ax: ax.build_mth_derivative_mpo(deriv_order, deriv_config=deriv_config)}
            gtn_mpo = self.make_gtn_from_dicts([mpo_dict], data_type=DataType.MPO)
            if store:   self._mpo_mth_derivatives[(key1,)] = gtn_mpo

            return gtn_mpo.copy()


    def get_dissipation_mpo(self, ax, strength: Numeric, deriv_order: int=2,
                            deriv_config: 'DerivativeConfiguration'=None):
        """
        Euler time step of dissipation:  f = f + eta d^m/dx^m f = (1 + eta d^m/dx^m) f
        for stability, assumes that the strength is small: eta << dt/dx^2
        strength: strength of dissipation (eta)
        deriv_order: order of derviative. e.g. order=1 --> m=2
            order = 1 (m=2): f_j + eta (f_{j+1} - 2 f_{j} + 1 f_{j-1})
            order = 2 (m=4): f_j + eta (-f_{j+2} + 4 f_{j+1} - 6f_j + 4f_{j-1} - f_{j-2})
            order = 3 (m=6): f_j + eta (f_{j+3} - 6 f_{j+2} + 15 f_{j+1} - 20 f_{j} + 15 f_{j-1} - 6 f_{j-2} + f_{j+3}
            i.e. mth derivative with 2nd order finite difference stencil
        """
        # assert(strength <= 1 / deriv_order ** 2), f'smoothing strength {strength} is too large'  # see p.114 in Durran
        assert(deriv_order % 2 == 0), f'deriv_order must be even, not {deriv_order}'

        deriv_config = deriv_config.copy()
        deriv_config.update(order=1, fd_type=FDType.CENTER)

        dissip_mpo = self.get_mth_derivative_mpo(ax, deriv_order, deriv_config=deriv_config)

        if ax.basis.type == BasisType.FOURIER:
            dx2_coeff = (1. / np.abs(ax.xpts[0])) ** deriv_order
        elif ax.basis.type == BasisType.SPATIAL:
            dx2_coeff = ax.dx ** deriv_order
        else:
            raise NotImplementedError

        if deriv_order % 4 == 0:
            dx2_coeff *= -1  ## to subtract contributions instead of add

        dissip_mpo = dissip_mpo.scalar_multiply(strength * dx2_coeff, inplace=False)

        return dissip_mpo


    def get_tridiag_mpo(self, ax_tridiag_vals: dict['Axis', tuple[Numeric,Numeric,Numeric]],
                        ax_deriv_configs: dict['Axis','DerivativeConfiguration']=None,):

        tridiag_ops = {ax: ax.get_tridiag_mpo(*vals, boundary_conditions=ax_deriv_configs[ax])
                       for ax, vals in ax_tridiag_vals.items()}
        return self.make_mpo_ndim(tridiag_ops)

    def get_averaging_mpo(self, axes: Sequence['Axis']=None, spread:int = 1,
                          ax_deriv_configs: dict['Axis','DerivativeConfiguration']=None,
                          stencil_type: FDType = FDType.CENTER) -> 'GridTN':
        """ fine-scale averaging
        """
        axes = self.axes if axes is None else axes
        if ax_deriv_configs is None:
            ax_deriv_configs = {}

        avg_ops = {
            ax: ax.get_averaging_mpo(spread=spread, stencil_type=stencil_type,
                                     boundary_conditions=ax_deriv_configs.get(ax, DerivativeConfiguration()))
            for ax in axes}
        avg_mpo = self.make_mpo_ndim(avg_ops)
        return avg_mpo

    def get_leapfrog_mpo(self, axes: Sequence['Axis']=None,
                          ax_deriv_configs: dict['Axis','DerivativeConfiguration']=None,
                          ) -> 'GridTN':
        """ fine-scale averaging
        """
        axes = self.axes if axes is None else axes
        if ax_deriv_configs is None:
            ax_deriv_configs = {}

        avg_ops = {
            ax: ax.get_leapfrog_mpo(boundary_conditions=ax_deriv_configs.get(ax, DerivativeConfiguration()))
            for ax in axes}
        avg_mpo = self.make_mpo_ndim(avg_ops)
        return avg_mpo

    def _build_mpo_firstderivative_upwind(self, x_ax: 'Axis', v_ax: 'Axis', deriv_params: 'DerivativeConfiguration') \
            -> 'GTN_TYPE':
        """ S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>
            ax:  Axis object along which to take derivative
            v_ax:  Axis object whose xpts (monotonically increasing) determine when
                   forward or backward finite differences is se
            scale by 1/dt later
            boundary_condition:  boundary condition to use when taking derivative
                BCType or Tuple of BCTypes (left, right boundary)
            order:  order of finite difference method
        """
        mpo_list = []
        print('upwind AX PARAMS', deriv_params)

        # v_ax = deriv_params.v_ax
        order = deriv_params.order
        offset = deriv_params.offset
        left_bc, right_bc = deriv_params.bc

        coordsys = x_ax.coord_sys
        deriv_config_F = DerivativeConfiguration(left_bc, right_bc, order, offset=offset, fd_type=FDType.FORWARD)
        deriv_config_B = DerivativeConfiguration(left_bc, right_bc, order, offset=offset, fd_type=FDType.BACKWARD)
        print('upwind F', deriv_config_F)
        print('upwind B', deriv_config_B)
        mpo_dict_f = coordsys.build_firstderivative_mpo(x_ax, deriv_config_F)
        mpo_dict_b = coordsys.build_firstderivative_mpo(x_ax, deriv_config_B)

        vels = v_ax.xpts
        zero_ind = np.argmax(vels >= 0)
        if vels[-1] < 0:
            zero_ind = v_ax.npts
        elif vels[0] > 0:
            zero_ind = -1

        # print('upwind',v_ax.axID,vels,zero_ind)

        if zero_ind < v_ax.npts:
            v_pos = v_ax.get_heaviside_mpo(zero_ind, get='right')  ## inds >= zero_ind
            mpo1 = {**mpo_dict_b, v_ax: v_pos}
            mpo_list += [mpo1]  ## this one is ok

        if zero_ind > 0:
            v_neg = v_ax.get_heaviside_mpo(zero_ind, get='xleft')  ## exclusive left
            mpo2 = {**mpo_dict_f, v_ax: v_neg}
            mpo_list += [mpo2]  ## this one is problematic

        print('mpo f', mpo_dict_f[x_ax].max_bond())
        print('mpo b', mpo_dict_b[x_ax].max_bond())
        gtn_mpo = self.make_gtn_from_dicts(mpo_list, data_type=DataType.MPO, compress=True)

        verbose_plot = False
        if verbose_plot:

            axes = gtn_mpo.grid.axes
            x1_ax, x2_ax, v1_ax, v2_ax = axes

            plt.figure()
            if v_ax == v1_ax:
                gtn_data = gtn_mpo.get_data(ax_select={x1_ax: 0, x2_ax: 0, v2_ax: 0})
            elif v_ax == v2_ax:
                gtn_data = gtn_mpo.get_data(ax_select={x1_ax: 0, x2_ax: 0, v1_ax: 0})
            plt.imshow(gtn_data)
            plt.title('x=0')
            plt.colorbar()

            plt.figure()

            if x_ax == x1_ax:
                gtn_data = gtn_mpo.get_data(ax_select={x2_ax: 0, v1_ax: -1, v2_ax: -1})
            elif x_ax == x2_ax:
                gtn_data = gtn_mpo.get_data(ax_select={x1_ax: 0, v1_ax: -1, v2_ax: -1})
            plt.imshow(gtn_data)
            plt.title('v=vs[-1]')
            plt.colorbar()

            mpo_data = x_ax.map_mpo_to_operator(mpo_dict_f[x_ax])
            plt.figure()
            plt.imshow(mpo_data)
            plt.title('forward')
            plt.colorbar()

            mpo_data = x_ax.map_mpo_to_operator(mpo_dict_b[x_ax])
            plt.figure()
            plt.imshow(mpo_data)
            plt.title('back')
            plt.colorbar()

            plt.show()

        return gtn_mpo

    def get_mpo_advection_SL(self, x_ax: 'Axis', v_ax: 'Axis', dt: float, deriv_params: 'DerivativeConfiguration',
                             advec_coeffs: np.array = None, method='') -> 'GTN_TYPE':
        ### TODO: store MPO
        if method == 'SL':
            print('SL advection')
            return self._get_mpo_advection_SL(x_ax, v_ax, dt, deriv_params, advec_coeffs)
        elif method == 'SL-pfc':
            print('SL-pfc advection')
            return self._get_mpo_advection_SL_pfc(x_ax, v_ax, dt, deriv_params, advec_coeffs)
        else:
            raise NotImplementedError

    # def _get_mpo_advection_SL(self, x_ax: 'Axis', v_ax: 'Axis', dt: float, deriv_params: 'DerivativeConfiguration',
    #                           advec_coeffs: np.ndarray = None,
    #                           pos_coeff: 'MPSType' = None, neg_coeff: 'MPSType' = None,) -> 'GTN_TYPE':
    #     """ derivative in x for SemiLagrangian t
    #         S+|x> = |x+1> , S-|x> = |x-1>
    #         df/dx = \sum_i (S- - S+)|x_i>
    #         ax:  Axis object along which to take derivative
    #         v_ax:  Axis object whose xpts (monotonically increasing) determine when
    #                forward or backward finite differences is se
    #         scale by 1/dt later
    #         boundary_condition:  boundary condition to use when taking derivative
    #             BCType or Tuple of BCTypes (left, right boundary)
    #         order:  order of finite difference method
    #
    #         see Kormann 2011 equation after 4.2 for f_(n+1)
    #         note: returns f_(n+1), NOT f_(n+1) - f_(n)
    #     """
    #     order = deriv_params.order
    #     left_bc, right_bc = deriv_params.bc
    #
    #     if order != 1:
    #         raise NotImplementedError
    #
    #     shift_0 = x_ax.get_iden_mpo()
    #     shift_p = x_ax.get_shift_mpo( 1, boundary_conditions=deriv_params)
    #     # print('shift p', x_ax.map_mpo_to_operator(shift_p))
    #     shift_m = x_ax.get_shift_mpo(-1, boundary_conditions=deriv_params)
    #
    #     # if right_bc == BCType.SYMMETRIC or right_bc == BCType.ANTISYMMETRIC:
    #     #     shift_p1_mat = np.zeros((x_ax.npts, x_ax.npts))
    #     #     shift_p1_mat[-1,-2] = 1. * np.sign(right_bc.value)
    #     #     shift_p1 = x_ax.map_operator_to_mpo(shift_p1_mat)
    #     #     helper.add_MPO(shift_p, shift_p1, inplace=True)
    #     #
    #     # if left_bc == BCType.SYMMETRIC or left_bc == BCType.ANTISYMMETRIC:
    #     #     shift_m1_mat = np.zeros((x_ax.npts, x_ax.npts))
    #     #     shift_m1_mat[0,1] = 1. * np.sign(left_bc.value)
    #     #     shift_m1 = x_ax.map_operator_to_mpo(shift_m1_mat)
    #     #     helper.add_MPO(shift_m, shift_m1, inplace=True)
    #
    #     # print(x_ax.map_mpo_to_operator(shift_p))
    #     # print(x_ax.map_mpo_to_operator(shift_m))
    #     # exit()
    #
    #     if pos_coeff is None or neg_coeff is None:
    #
    #         if advec_coeffs is None:        # use velocities
    #             advec_coeffs = v_ax.map_state_to_mps(v_ax.xpts)
    #             v_pos_mpo = v_ax.get_heaviside_mpo(v_ax.zero_ind, get='xright')
    #             v_neg_mpo = v_ax.get_heaviside_mpo(v_ax.zero_ind, get='xleft')
    #             pos_coeff = helper.apply(advec_coeffs, v_pos_mpo)
    #             neg_coeff = helper.apply(advec_coeffs, v_neg_mpo)
    #             helper.scalar_multiply(pos_coeff, dt / x_ax.dx, inplace=True)
    #             helper.scalar_multiply(neg_coeff, -1 * dt / x_ax.dx, inplace=True)
    #         else:
    #             dx = x_ax.dx
    #             v_pos = np.where(advec_coeffs <= 0, 0., advec_coeffs) * dt / dx
    #             v_neg = np.where(advec_coeffs >= 0, 0., advec_coeffs) * dt / dx * -1
    #             # v_abs = 1 - np.abs(advec_coeffs) * dt / dx
    #
    #             pos_coeff = v_ax.map_state_to_mps(v_pos)
    #             neg_coeff = v_ax.map_state_to_mps(v_neg)
    #             # v_abs_mps = v_ax.map_state_to_mps(v_abs)
    #
    #     abs_coeff = helper.add_MPS(pos_coeff, neg_coeff, inplace=False)
    #     helper.scalar_multiply(abs_coeff, -1, inplace=True)
    #     helper.add_MPS( abs_coeff, v_ax.get_iden_mps(), inplace=True )
    #
    #     v_pos_mpo = v_ax.apply_elemental_multiply_op(pos_coeff)
    #     v_neg_mpo = v_ax.apply_elemental_multiply_op(neg_coeff)
    #     v_abs_mpo = v_ax.apply_elemental_multiply_op(abs_coeff)
    #
    #     ## forward mpo
    #     mpo_list = [{x_ax: shift_0, v_ax: v_abs_mpo},
    #                 {x_ax: shift_p, v_ax: v_neg_mpo},
    #                 {x_ax: shift_m, v_ax: v_pos_mpo}]
    #
    #     ### for EM split step:  provide list of dicts for shift 0, p, m
    #
    #     gtn_mpo = self.make_gtn_from_dicts(mpo_list, data_type=DataType.MPO, compress=True)
    #
    #     return gtn_mpo

    def _get_mpo_advection_SL_pfc(self, x_ax: 'Axis', v_ax: 'Axis', dt: float,
                                  deriv_params: 'DerivativeConfiguration', advec_coeffs: np.ndarray = None,
                                  pos_coeff: 'MPSType' = None, neg_coeff: 'MPSType' = None, ) -> 'GTN_TYPE':
        """ S+|x> = |x+1> , S-|x> = |x-1>
            df/dx = \sum_i (S- - S+)|x_i>
            ax:  Axis object along which to take derivative
            v_ax:  Axis object whose xpts (monotonically increasing) determine when
                   forward or backward finite differences is se
            scale by 1/dt later
            boundary_condition:  boundary condition to use when taking derivative
                BCType or Tuple of BCTypes (left, right boundary)
            order:  order of finite difference method

            see Kormann 2022 section 3.4
            Note: returns f_(n+1) NOT f_(n+1) - f_(n)
        """
        order = deriv_params.order
        left_bc, right_bc = deriv_params.bc

        if order != 1:
            raise NotImplementedError

        if advec_coeffs is not None:
            raise NotImplementedError

        # if left_bc != BCType.PERIODIC and right_bc != BCType.PERIODIC:
        #     raise NotImplementedError

        shift_00 = x_ax.get_iden_mpo()
        shift_p1 = x_ax.get_shift_mpo(+1, boundary_conditions=deriv_params)
        shift_p2 = x_ax.get_shift_mpo(+2, boundary_conditions=deriv_params)
        shift_m1 = x_ax.get_shift_mpo(-1, boundary_conditions=deriv_params)
        shift_m2 = x_ax.get_shift_mpo(-2, boundary_conditions=deriv_params)

        # if right_bc == BCType.SYMMETRIC or right_bc == BCType.ANTISYMMETRIC:
        #     shift_p1b_mat = np.zeros((x_ax.npts, x_ax.npts))
        #     shift_p1b_mat[-1, -2] = 1. * np.sign(right_bc.value)
        #     shift_p1b = x_ax.map_operator_to_mpo(shift_p1b_mat)
        #     helper.add_MPO(shift_p1, shift_p1b, inplace=True)
        #
        #     shift_p2b_mat = np.zeros((x_ax.npts, x_ax.npts))
        #     shift_p2b_mat[-2, -2] = 1. * np.sign(right_bc.value)
        #     shift_p2b_mat[-1, -3] = 1. * np.sign(right_bc.value)
        #     shift_p2b = x_ax.map_operator_to_mpo(shift_p2b_mat)
        #     helper.add_MPO(shift_p2, shift_p2b, inplace=True)
        #
        # if left_bc == BCType.SYMMETRIC or left_bc == BCType.ANTISYMMETRIC:
        #     shift_m1b_mat = np.zeros((x_ax.npts, x_ax.npts))
        #     shift_m1b_mat[0, 1] = 1. * np.sign(left_bc.value)
        #     shift_m1b = x_ax.map_operator_to_mpo(shift_m1b_mat)
        #     helper.add_MPO(shift_m1, shift_m1b, inplace=True)
        #
        #     shift_m2_mat = np.zeros((x_ax.npts, x_ax.npts))
        #     shift_m2_mat[0, 2] = 1. * np.sign(left_bc.value)
        #     shift_m2_mat[1, 1] = 1. * np.sign(left_bc.value)
        #     shift_m2b = x_ax.map_operator_to_mpo(shift_m2_mat)
        #     helper.add_MPO(shift_m2, shift_m2b, inplace=True)

        # print('p1', x_ax.map_mpo_to_operator(shift_p1))
        # print('m1', x_ax.map_mpo_to_operator(shift_m1))
        # print('m2', x_ax.map_mpo_to_operator(shift_m2))
        # exit()

        dx = x_ax.dx
        pos_adv_vals = np.where(v_ax.xpts >= 0, 0., v_ax.xpts) * dt / dx * -1  # advection values for velocities > 0
        neg_adv_vals = np.where(v_ax.xpts <= 0, 0., v_ax.xpts) * dt / dx * -1  # advection values for velocities < 0

        # neg_adv_vals = v_pos * dt / dx * -1
        neg_weights_m1 = -neg_adv_vals * (1 - neg_adv_vals) * (2 - neg_adv_vals) / 6
        neg_weights_00 = (1 - neg_adv_vals) * (1 - neg_adv_vals * (2 - neg_adv_vals) / 6
                                               + neg_adv_vals * (1 + neg_adv_vals) / 6)
        neg_weights_p1 = neg_adv_vals * (1 - neg_adv_vals) * (1 + neg_adv_vals) / 6

        # pos_adv_vals = v_neg * dt / dx * -1
        pos_weights_p1 = -pos_adv_vals * (1 - pos_adv_vals) * (2 - pos_adv_vals) / 6
        pos_weights_00 = pos_adv_vals * (1 - (1 - pos_adv_vals) * (2 - pos_adv_vals) / 6
                                         + (1 - pos_adv_vals) * (1 + pos_adv_vals) / 6)
        pos_weights_m1 = pos_adv_vals * (1 - pos_adv_vals) * (1 + pos_adv_vals) / 6

        ## flux from left + f0
        weights_m2 = pos_weights_m1
        weights_m1 = pos_weights_00 + neg_weights_m1
        weights_00 = pos_weights_p1 + neg_weights_00 + 1
        weights_p1 = neg_weights_p1
        weights_p2 = np.zeros_like(pos_weights_p1)

        ## - flux from right
        weights_m1 -= pos_weights_m1
        weights_00 -= pos_weights_00 + neg_weights_m1
        weights_p1 -= pos_weights_p1 + neg_weights_00
        weights_p2 -= neg_weights_p1

        # ## flux from left - flux from right
        # weights_m2 = pos_weights_m1
        # weights_m1 = pos_weights_00 + neg_weights_m1 - pos_weights_m1
        # weights_00 = pos_weights_p1 + neg_weights_00 - pos_weights_00 - neg_weights_m1
        # weights_p1 = neg_weights_p1 - pos_weights_p1 - neg_weights_00
        # weights_p2 = neg_weights_p1

        weights_p2_mps = v_ax.map_state_to_mps(weights_p2)
        weights_p1_mps = v_ax.map_state_to_mps(weights_p1)
        weights_00_mps = v_ax.map_state_to_mps(weights_00)
        weights_m1_mps = v_ax.map_state_to_mps(weights_m1)
        weights_m2_mps = v_ax.map_state_to_mps(weights_m2)

        weights_p2_mpo = v_ax.apply_elemental_multiply_op(weights_p2_mps)
        weights_p1_mpo = v_ax.apply_elemental_multiply_op(weights_p1_mps)
        weights_00_mpo = v_ax.apply_elemental_multiply_op(weights_00_mps)
        weights_m1_mpo = v_ax.apply_elemental_multiply_op(weights_m1_mps)
        weights_m2_mpo = v_ax.apply_elemental_multiply_op(weights_m2_mps)

        ## forward mpo
        mpo_list = [{x_ax: shift_00, v_ax: weights_00_mpo},
                    {x_ax: shift_p1, v_ax: weights_p1_mpo},
                    {x_ax: shift_p2, v_ax: weights_p2_mpo},
                    {x_ax: shift_m1, v_ax: weights_m1_mpo},
                    {x_ax: shift_m2, v_ax: weights_m2_mpo}, ]

        gtn_mpo = self.make_gtn_from_dicts(mpo_list, data_type=DataType.MPO, compress=True)

        return gtn_mpo

    def laplacian_mpo(self, axes: Sequence['Axis'] = None,
                      ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None,
                      compID=None, recalc=False, store=True, eeo_grid=False) -> 'GTN_TYPE':
        """ L(f) = d^2/dx^2 f + d^2/dy^2 f + d^2/dz^2 z
            mps_scalar_field:  mps representing the scalar field
        """
        if axes is None:
            axes = self.axes

        if ax_deriv_configs is None:
            ax_deriv_configs = {ax: DerivativeConfiguration() for ax in axes}

        # axIDs = [ax.axID for ax in axes]
        try:
            if recalc:  raise KeyError
            return self._mpo_laplacian[frozenset(axes)].copy()

        except KeyError:
            mpo_list = []
            for ax in axes:
                coord_sys = ax.coord_sys  ## axes can be in different coord_sys
                # deriv_E_config = self.ax_deriv_configs[ax.axID]
                if compID is None:
                    # print('type coord', coord_sys)
                    mpo_list += coord_sys.build_laplacian_mpo([ax], ax_deriv_configs, eeo_grid=eeo_grid)
                else:
                    mpo_list += coord_sys.build_vector_laplacian_mpo(compID, [ax], ax_deriv_configs)
                # print('mpo list', ax, mpo_list[-1][ax])

            gtn_mpo = self.make_gtn_from_dicts(mpo_list, data_type=DataType.MPO)

            if compID is None and store:   self._mpo_laplacian[frozenset(axes)] = gtn_mpo
            # print('laplacian mpo', gtn_mpo)
            return gtn_mpo.copy()

    def inverse_laplacian_mpo(self, axes: Sequence['Axis'] = None, boundary_conditions=None,
                              ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None,
                              compress_opts: dict = None, eeo_grid=False) -> 'GTN_TYPE':
        """ L(f) = d^2/dx^2 f + d^2/dy^2 f + d^2/dz^2 z
            mps_scalar_field:  mps representing the scalar field
        """
        if axes is None:
            axes = self.axes

        config_list = []
        for ax in axes:
            dcfg = ax_deriv_configs.get(ax, None)
            config_list += [(ax, dcfg.bc if dcfg is not None else None)]
        config_list = tuple(config_list)
        # config_list = tuple([ax_deriv_configs.get(ax, None).left_bc for ax in axes])
        print('saved inverse', self._mpo_inverses.keys())

        if (axes, config_list) in self._mpo_inverses:
            print('loading stored inv lapl mpo')
            return self._mpo_inverses[(axes, config_list)].copy()

        print('inverse lapl', [(ax, ax_deriv_configs[ax].left_bc) for ax in axes])

        print('is k?', [ax.is_k() for ax in axes])
        if np.all([ax.is_k() for ax in axes]):
            print('obtaining inverse laplacian using 1/k**2')
            k2s_list = []
            for ax in axes:
                k2s = ax.xpts ** 2
                mps_k2s = ax.map_state_to_mps(k2s)
                helper.scalar_multiply(mps_k2s, -1., inplace=True)
                mps_k2s_dict = {ax_other: ax_other.get_all_ones_mps() for ax_other in axes if ax_other != ax}
                mps_k2s_dict[ax] = mps_k2s
                k2s_list += [mps_k2s_dict]

            gtn_mps = self.make_gtn_from_dicts(k2s_list, data_type=DataType.MPS)
            inv_laplacian_mpo = self.get_diag_mpo_inverse(gtn_mps, compress_opts=compress_opts)

        elif len(self.axes) >= 1 and np.all(
                [ax_deriv_configs[ax].left_bc == BCType.PERIODIC or ax.is_k() for ax in axes]):
            # elif np.all( [ax_deriv_configs[ax].left_bc==BCType.PERIODIC or ax.is_k() for ax in axes] ):
            print('obtaining inverse using pseudospectral methods')

            qft_mpos = {}
            inverse_qft_mpos = {}
            k2s_list = []
            k_vals_ax = {}
            for ax in axes:
                if ax.is_k():
                    k2s = ax.xpts ** 2
                else:
                    qft_mpo = ax.get_qft_mpo()
                    k_vals = ax.get_qft_ks()  ## np.ndarray vector
                    k2s = k_vals ** 2
                    # k_vals = ax.get_qft_freqs()  ## MPS
                    # k2s = ax.map_mps_to_state(k_vals) ** 2
                    k_vals_ax[ax] = k_vals
                    qft_mpos[ax] = qft_mpo
                    # inverse_qft_mpos[ax] = ax.get_qft_mpo(inverse=True)
                    inverse_qft_mpos[ax] = ax.get_inverse_qft_mpo()

                mps_k2s = ax.map_state_to_mps(k2s)
                # helper.scalar_multiply(mps_k2s, -1., inplace=True)
                k2s_list += [{ax: mps_k2s}]

            qft_gtn = self.make_mpo_ndim(qft_mpos)
            k2s_gtn = self.make_gtn_from_dicts(k2s_list, data_type=DataType.MPS)
            inv_k2s_gtn = self.get_diag_mpo_inverse(k2s_gtn)
            inv_k2s_gtn.scalar_multiply(-1, inplace=True)
            inverse_qft_gtn = self.make_mpo_ndim(inverse_qft_mpos)
            # print('qft gtn', qft_gtn)
            # print('inverse k2s', inv_k2s_gtn)
            # print('inverse qft gtn', inverse_qft_gtn)

            inv_laplacian_mpo = qft_gtn.apply(inv_k2s_gtn, compress=True, zipup=True,
                                              compress_opts={'cutoff': 1.0e-20, 'cutoff_mode': 'rsum2'})
            # k = 2 * np.pi / (self.axes[0].npts * self.axes[0].dx)
            # inv_laplacian_mpo = qft_gtn.scalar_multiply(-1./(k**2))   ### only valid for cos(kx), sin(kx)
            inv_laplacian_mpo = inv_laplacian_mpo.apply(inverse_qft_gtn, inplace=True, compress=True, zipup=True,
                                                        compress_opts={'cutoff': 1.0e-20, 'cutoff_mode': 'rsum2'})
            print('inv laplacian mpo', inv_laplacian_mpo)

            # kx = 2 * np.pi / (self.axes[0].dx * self.axes[0].npts)
            # ky = 2 * np.pi / (self.axes[1].dx * self.axes[1].npts)
            # kz = 2 * np.pi / (self.axes[2].dx * self.axes[2].npts)
            # k = np.sqrt(kx**2 + ky**2 + kz**2)
            # print('k', np.sqrt(kx**2 + ky**2 + kz**2))

            # ax_x, ax_y, ax_z = self.axes[0:3]
            # x_vals, y_vals, z_vals = ax_x.xpts, ax_y.xpts, ax_z.xpts
            # kx_vals, ky_vals, kz_vals = k_vals_ax[ax_x], k_vals_ax[ax_y], k_vals_ax[ax_z]
            # init_V_comps = [{ax_x: np.exp(-1.j * kx * x_vals), ax_y: np.exp(-1.j * ky * y_vals),
            #                  ax_z: np.exp(-1.j * kz * z_vals)},
            #                 {ax_x: np.exp(+1.j * kx * x_vals), ax_y: np.exp(+1.j * ky * y_vals),
            #                  ax_z: np.exp(+1.j * kz * z_vals)}]
            # init_V_gtn = self.make_gtn_from_dicts(init_V_comps, data_type=DataType.MPS)
            # qft_init = init_V_gtn.apply(qft_gtn)
            # qft_init_data = qft_init.get_data()
            # plt.figure()
            # plt.imshow(np.real(init_V_gtn.get_data()[:,:,0]))
            # plt.show()
            # plt.figure()
            # kx_idx, ky_idx, kz_idx = np.unravel_index(np.argmax(np.abs(qft_init_data)), (ax_x.npts, ax_y.npts, ax_z.npts))
            # plt.plot(kx_vals, np.real(qft_init_data)[:, ky_idx, kz_idx], 'x')
            # plt.plot(kx_vals, np.imag(qft_init_data)[:, ky_idx, kz_idx], '+')
            # plt.figure()
            # plt.plot(kx_vals, np.real(qft_init_data)[kx_idx, :, kz_idx], 'x')
            # plt.plot(kx_vals, np.imag(qft_init_data)[kx_idx, :, kz_idx], '+')
            # plt.figure()
            # plt.plot(kx_vals, np.real(qft_init_data)[kx_idx, ky_idx, :], 'x')
            # plt.plot(kx_vals, np.imag(qft_init_data)[kx_idx, ky_idx, :], '+')
            # plt.show()

            # laplacian_mpo = self.laplacian_mpo(store=False, ax_deriv_configs=ax_deriv_configs)
            # check = inv_laplacian_mpo.apply(laplacian_mpo)
            # check_data = self.map_mpo_to_operator(check).reshape(self.npts, self.npts)
            # print('inverse error', np.linalg.norm(np.eye(self.npts) - check_data))

        else:
            print('obtaining naive inverse')
            if ax_deriv_configs is None:
                ax_deriv_configs = {ax: DerivativeConfiguration() for ax in axes}

            laplacian_mpo = self.laplacian_mpo(ax_deriv_configs=ax_deriv_configs, store=False, eeo_grid=eeo_grid)
            inv_laplacian_mpo = self.get_mpo_inverse(laplacian_mpo, boundary_conditions=boundary_conditions)
            inv_laplacian_mpo.compress(inplace=True, compress_opts=compress_opts)
            print('inv laplacian mpo', inv_laplacian_mpo)

        # laplacian_mpo = self.laplacian_mpo(store=False, ax_deriv_configs=ax_deriv_configs)
        # check = inv_laplacian_mpo.apply(laplacian_mpo)
        # check_data = self.map_mpo_to_operator(check).reshape(self.npts, self.npts)
        # print(check_data)
        # print('inv',inv_laplacian_mpo.get_data().reshape(self.npts, self.npts))
        # print('lapl',laplacian_mpo.get_data().reshape(self.npts, self.npts))
        # print('inverse error', np.linalg.norm(np.eye(self.npts) - check_data))

        self._mpo_inverses[(axes, config_list)] = inv_laplacian_mpo

        return inv_laplacian_mpo.copy()

    def get_qft_mpo(self, ft_axes=None):
        qft_mpos = {}
        if ft_axes is None:
            ft_axes = self.axes
        for ax in ft_axes:
            if ax.basis.type == BasisType.SPATIAL:
                qft_mpo = ax.get_qft_mpo()
                qft_mpos[ax] = qft_mpo
        return self.make_mpo_ndim(qft_mpos)

    def get_inverse_qft_mpo(self, ft_axes=None):
        inverse_qft_mpos = {}
        if ft_axes is None:
            ft_axes = self.axes
        for ax in ft_axes:
            if ax.basis.type == BasisType.SPATIAL:
                inverse_qft_mpos[ax] = ax.get_inverse_qft_mpo()
        return self.make_mpo_ndim(inverse_qft_mpos)

    def get_qft_ks(self, power=1):
        ks_list = []
        for ax in self.axes:
            if ax.basis.type == BasisType.SPATIAL:
                k_vals = ax.get_qft_freqs()
                ks = k_vals ** power
                mps_ks = ax.map_state_to_mps(ks)
                ks_list += [{ax: mps_ks}]
            elif ax.basis.type == BasisType.FOURIER:
                ks_list += [{ax: ax.xpts}]
        return self.make_gtn_from_dicts(ks_list, data_type=DataType.MPS)

    #######################
    ## partial integrals ##
    #######################

    def get_integrals_mpx(self, axes: list['Axis'] or None = None, order=1, is_sqrt=False, recalc=False, store=True) \
            -> 'GTN_TYPE':
        """ get MPS taking integral along axis ax
            order*2: degree of polynomial expansion in Newton-Cotes
            for definite integrals over the entire space
        """
        if axes is None:   axes = self.axes
        # axIDs = [ax.axID for ax in axes]
        try:
            if recalc:  raise KeyError
            return self._mpx_integrals[(frozenset(axes), is_sqrt)].copy()

        except KeyError:
            out_mps_dict = {}
            for ax in axes:
                coordsys = ax.coord_sys
                integ = coordsys.build_integral_mps(ax, is_sqrt=is_sqrt)
                # integ = ax.basis.build_integral_mps()
                ## if sqrt:  integ = helper.mps_to_diag_mpo(self[i].conj())
                out_mps_dict[ax] = integ

            tot_dx = []
            for ax in self.axes:
                tot_dx += [2 * np.pi / ax.dx if ax.is_k() else ax.dx]
            tot_dx = np.prod(tot_dx)

            gtn_mps = self.make_gtn_from_dicts([out_mps_dict], data_type=DataType.MPX, compress=True,
                                               compress_opts={'ref_norm': tot_dx})
            # print('integ mps rank', gtn_mps.max_bond())
            if store:   self._mpx_integrals[(frozenset(axes), is_sqrt)] = gtn_mps
            return gtn_mps.copy()

    def get_coarse_grain_mpx(self, depth: int, axes: list['Axis'] or None = None, is_sqrt=False,
                             ) -> 'GTN_TYPE':
        """ get MPS taking integral along axis ax
            order*2: degree of polynomial expansion in Newton-Cotes
            for definite integrals over the entire space
        """
        if axes is None:   axes = self.axes
        out_mps_dict = {}
        for ax in axes:
            coordsys = ax.coord_sys
            integ = coordsys.build_coarse_grain_mpx(ax, depth, is_sqrt=is_sqrt)
            out_mps_dict[ax] = integ
        gtn_mps = self.make_gtn_from_dicts([out_mps_dict], data_type=DataType.MPO, compress=True)
        return gtn_mps

    def get_coarse_select_mpx(self, depth: int, axes: list['Axis'] or None = None, is_sqrt=False,
                             ) -> 'GTN_TYPE':
        """ get MPS taking integral along axis ax
            order*2: degree of polynomial expansion in Newton-Cotes
            for definite integrals over the entire space
        """
        if axes is None:   axes = self.axes
        out_mps_dict = {}
        for ax in axes:
            integ = ax.basis.build_coarse_select_mpx(ax, depth) #, is_sqrt=is_sqrt)
            out_mps_dict[ax] = integ
        gtn_mps = self.make_gtn_from_dicts([out_mps_dict], data_type=DataType.MPO, compress=True)
        return gtn_mps

    def add_gtns(self, *mps_objs: Union['GridTN', Sequence['GridTN']], weights=None, zipup=True,
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
            out = None
            for weight, item in zip(weights, mps_objs):
                if isinstance(item, GridTN):
                    out_comp = item.copy()
                elif isinstance(item, (list, tuple)):
                    ## ordered how it's written:  [A, B, C, x]
                    out_comp = item[-1].copy()
                    assert (isinstance(out_comp, GridTN)), f'out comp needs to GTN object not {type(out_comp)}'

                    for op in item[-2::-1]:
                        assert (isinstance(out_comp, GridTN)), f'out comp needs to GTN object not {type(op)}'
                        out_comp.apply(op, inplace=True, zipup=zipup, compress=sub_compress,
                                       compress_opts=sub_compress_opts)
                        out_comp.scalar_multiply(weight, inplace=True)
                else:
                    raise TypeError('not correct type', type(item))

                if out is None:
                    out = out_comp
                else:
                    out = out.add(out_comp, inplace=True, compress=False)

            if out is not None and compress:
                out.compress(compress_opts=compress_opts)

            return out

        else:
            raise NotImplementedError

    def build_indexed_gtn(self, dict_gtns: dict[any, 'GridTN'], index_order=None) -> 'GridTN':
        """ builds a gtn from provided gtns in dict_gtns
            adds a tensor that indexes each gtn as specified by the key
                if an MPO, any will be a tuple (out, in)
        """
        raise NotImplementedError

    def select_indexed_gtn(self, gtn: 'GridTN', select_ind: Union[int, tuple[int]]) -> 'GridTN':
        """ selects a subgtn from gtn
        """
        raise NotImplementedError

    ####################
    ### dmrg methods ###
    ####################

    def gtn_to_dmrg_format(self, gtn: 'GridTN', is_mps=True):
        raise NotImplementedError

    def dmrg_to_gtn_format(self, gtn, soln, is_mps=True, inplace=True):
        raise NotImplementedError

    def _setup_dmrg_solver(self, gtn: 'GridTN', operator: 'GridTN', compress_type: 'CompressType', inplace=False,
                           compress_opts=None, is_H=False, init_guess: 'GridTN' = None, verbose_output=False, **kwargs
                           ):
        """ set up solver for DMRG LinearSolve
        """
        raise NotImplementedError

    def _extract_dmrg_soln(self, gtn: 'GridTN', soln, inplace=False):
        raise NotImplementedError

    def axpby_dmrg(self, gtns: Sequence['GridTN'], ops: Sequence['GridTN'], compress_opts=None, **dmrg_opts):

        from local_solvers.local_dmrg_eval import local_dmrg_evaluator, Term_DMRG

        DMAX = compress_opts.get('max_bond', None) if compress_opts is not None else None

        gtn = gtns[0].copy()
        is_mps = gtn.data_type == DataType.MPS

        terms = []
        for i in range(len(gtns)):
            vec = gtns[i]
            op = ops[i] if i < len(ops) else None

            vec_mps = self.gtn_to_dmrg_format(vec, is_mps=True)
            op_mps = self.gtn_to_dmrg_format(op, is_mps=False)

            term = Term_DMRG(vec_mps.copy(), operators=[op_mps]) if op_mps is not None else Term_DRMG(vec_mps.copy())
            terms += [term]

        print('axpby func', DMAX)
        print('term bonds')
        for term in terms:
            print('term.ket', term.out)

        func_mps = local_dmrg_evaluator(terms, max_bond=DMAX, **dmrg_opts)

        gtn = self.dmrg_to_gtn_format(gtn, func_mps, is_mps=is_mps)
        return gtn

