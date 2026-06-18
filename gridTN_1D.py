"""GridTN1D: one-dimensional QTT tensor-network state (MPS / MPO).

Concrete :class:`~gridTN.GridTN` holding a quantized matrix product state or operator on
a :class:`~grid1D.Grid1D`. Implements the core QTT operations used by the PDE solvers:
construction from data, compression / truncation, application of differential and
multiplication operators, and the DMRG / TDVP / interpolative-DLR update routines (via
:mod:`helper_dmrg`, :mod:`helper_tdvp`, and :mod:`local_solvers`).
"""

import pdb

import numpy as np
import scipy.sparse

import helper_quimb
from setup_.configs import *
from setup_.quimb_TN1D import MatrixProductStateUSVT as MPS_USVT
from setup_.quimb_TN1D import MatrixProductStateTN
from helper_tdvp import TDVPSolver, TDVPSolver0, TDVPSolver2, TDMRGSolver

import helper_quimb as helper
import helper_dmrg
import helper_dmrg_2
import grid
from gridTN import GridTN
from basis.basis_k import FourierBasis
from basis.basis_k import RealFourierBasis
from basis.basis_spatial import SpatialBasis
from local_solvers.terms_3 import Term
from axis import get_select_elem_mps

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axis import Axis
    from grid1D import Grid1D

""" object with MPS/MPO as data, plus the GridLayout object (the grid) it's associated with
"""

DEFAULT_SOLVE_TYPE = helper_dmrg.DEFAULT_SOLVE
DEFAULT_OPT_NSITES = 2


########################################
## static methods to create GridTN1Ds ##
########################################

################
## common TNs ##
################

####################
## GridTN1D class ##
####################

class GridTN1D(GridTN):

    def __init__(self, grid: 'Grid1D', data: Optional[qtn.TensorNetwork1D] = None, ax_deriv_configs=None):
        """Construct a 1-D quantized tensor-network state on a grid.

        Parameters
        ----------
        grid : Grid1D
            Grid layout the tensor network lives on.
        data : qtn.TensorNetwork1D, optional
            MPS or MPO data consistent with ``grid``; left unset if None.
        ax_deriv_configs : dict, optional
            Per-Axis finite-difference configurations forwarded to the base class.
        """
        super().__init__(grid, ax_deriv_configs=ax_deriv_configs)
        # self.grid = grid
        ## Attributes: grids, ngrids, _data, axes, axIDs, ax_map, ndim

        if data is not None:
            self.data = data
            # self._set_data(data)

    def _get_data(self) -> TN1Type:
        """Return the underlying MPS/MPO/scalar data object."""
        return self._data

    def _set_data(self, data):
        """Set the underlying data, coercing several input forms to a 1-D TN.

        Parameters
        ----------
        data : ndarray, dict, qtn.TensorNetwork1D, GridTN1D, float, complex, or None
            Source data. Dense arrays and dicts are mapped to an MPS, falling
            back to an MPO on assertion failure; a GridTN1D contributes its
            ``.data``; scalars are stored as-is. None clears the data.
        """
        if data is None:
            self._data = data
        else:
            if isinstance(data, np.ndarray):
                try:
                    gtn = self.grid.map_state_to_mps(data)
                    new_data = gtn.data
                except AssertionError:
                    gtn = self.grid.map_operator_to_mpo(data)
                    new_data = gtn.data
            elif isinstance(data, dict):
                try:
                    gtn = self.grid.make_mps_ndim(data)
                    new_data = gtn.data
                except AssertionError:
                    gtn = self.grid.make_mpo_ndim(data)
                    new_data = gtn.data
            elif isinstance(data, qtn.TensorNetwork1D):
                new_data = data
            elif isinstance(data, GridTN1D):
                new_data = data.data
            elif isinstance(data, (float, complex)):
                new_data = data
            else:
                raise TypeError(type(data))

            self._data = new_data
            self.data_type = type(new_data)
            # self.is_MPS = isinstance(new_data,qtn.MatrixProductState)

    data = property(fget=_get_data, fset=_set_data)

    def __getitem__(self, ind):
        """Index the underlying MPS/MPO object.

        Parameters
        ----------
        ind : int or slice
            Site index (or slice) passed through to the data object.

        Returns
        -------
        qtn.Tensor or qtn.TensorNetwork1D
            The selected tensor(s).
        """
        return self._data[ind]

    def get_inds_in_axis(self, ax: 'Axis', ax_ind=None) -> list:
        """Get the 1-D TN site indices belonging to a given axis.

        Parameters
        ----------
        ax : Axis
            Axis whose site indices are requested.
        ax_ind : int, optional
            Select a single index within the axis instead of the full list.

        Returns
        -------
        list
            Site indices corresponding to ``ax``.
        """
        return self.grid.get_inds_in_axis(ax, ax_ind)

    def shape(self) -> tuple:
        """Return the dense shape of the underlying grid."""
        return self.grid.shape()

    @property
    def L(self):
        """Number of sites (length) of the underlying 1-D tensor network."""
        return self.data.L

    def mangle_inner(self, inplace=True, append=None):
        """Rename inner bond indices to avoid name collisions.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        append : str, optional
            Suffix appended to inner index names; forwarded to the data object.

        Returns
        -------
        GridTN1D
            The state with mangled inner indices.
        """
        out = self if inplace else self.copy()
        out.data.mangle_inner_(append=append)
        return out

    def create_like(self, new_data=None) -> 'GridTN1D':
        """Build a new GridTN1D sharing this grid and deriv configs.

        Parameters
        ----------
        new_data : qtn.TensorNetwork1D or compatible, optional
            Data for the new object; None creates an empty one.

        Returns
        -------
        GridTN1D
            A fresh instance on the same grid.
        """
        new_tn1D = self.__class__(self.grid, data=new_data, ax_deriv_configs=self.ax_deriv_configs)
        return new_tn1D

    def copy(self, deep=True) -> 'GridTN1D':
        """Copy this state, preserving constancy and canonical-form metadata.

        Parameters
        ----------
        deep : bool
            Deep-copy the underlying data if True, else share the data object.

        Returns
        -------
        GridTN1D
            The copied state.
        """
        new_tn1D = self.create_like()
        if self.data is not None:
            new_tn1D.data = self.data.copy() if (deep and self.data is not None) else self.data
        new_tn1D.is_constant = self.is_constant
        new_tn1D.constant_axes = self.constant_axes
        new_tn1D.canon_site = self.canon_site
        return new_tn1D

    def get_TN(self):
        """Return the underlying quimb tensor-network data object."""
        return self.data

    def conj(self, inplace=False, mangle_inner=False) -> 'GridTN1D':
        """Take the complex conjugate of the tensor network.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else return a copy.
        mangle_inner : bool
            Rename inner bond indices while conjugating to avoid collisions.

        Returns
        -------
        GridTN1D
            The conjugated state.
        """
        new_tn1D = self if inplace else self.copy()
        new_tn1D.data.conj(mangle_inner=mangle_inner, inplace=True)
        return new_tn1D

    # def norm(self, is_sqrt=False) -> Numeric:
    #     if self.data is not None:
    #         out = self.integrate(is_sqrt=is_sqrt)
    #         if np.isreal(out) or np.abs(np.imag(out)) < 1.0e-13:
    #             out = np.real(out)
    #         return out

    def ovlp(self, other) -> Numeric:
        """Compute the overlap <self|other> between two states.

        Parameters
        ----------
        other : GridTN1D
            The other state; returns 0.0 if either state has no data.

        Returns
        -------
        Numeric
            The overlap value.
        """
        if self.data is not None and other.data is not None:
            out = helper.ovlp(self.data, other.data)
            return out
        else:
            return 0.0

    # def distance(self, other) -> Numeric:
    #     if self.data is not None and other.data is not None:
    #         out = helper.distance(self.data, other.data)
    #         return out
    #     elif self.data is None: # other data is not None
    #         return np.inf
    #     else:
    #         return np.nan

    def max_bond(self) -> int:
        """Return the largest bond dimension of the tensor network.

        Returns
        -------
        int
            Maximum bond dimension; 1 for a scalar/constant component, NaN if
            there is no data.
        """
        if self.data is not None:
            if hasattr(self.data, 'max_bond'):
                return self.data.max_bond()
            else:
                return 1  # scalar/constant field component has trivial bond dimension
        else:
            return np.nan

    def num_elem(self) -> Numeric:
        """Return the total number of stored tensor entries.

        Returns
        -------
        Numeric
            Sum of ``tensor.size`` over all tensors, or 0 if there is no data.
        """
        if self.data is not None:
            # print('num elem', [t.shape for t in self.data.tensors])
            # print('num elem', np.sum([t.size for t in self.data.tensors]))
            return np.sum([t.size for t in self.data.tensors])
        else:
            return 0


    def mem_size(self) -> Numeric:
        """Return the in-memory size of the tensor data in kilobytes.

        Returns
        -------
        Numeric
            Total bytes (itemsize times size summed over tensors) divided by
            1000, or 0 if there is no data.
        """
        if self.data is not None:
            return np.sum([t.data.itemsize * t.data.size for t in self.data.tensors]) / 1000
        else:
            return 0

    def all_virtual_sizes(self) -> list[int]:
        """Return the list of inner (virtual) bond dimensions.

        Returns
        -------
        list[int]
            Bond size of each neighboring-site bond, or an empty list if there
            is no data.
        """
        if self.data is not None:
            return [self.data.bond_size(i,i+1) for i in range(self.data.L - 1)]
        else:
            return []

    def all_smallest_singular_values(self, max_bond=None) -> list[Numeric]:
        """Return the smallest singular value at each bond.

        Parameters
        ----------
        max_bond : int, optional
            If given, bonds with fewer than ``max_bond`` singular values report
            0 instead of their tail value.

        Returns
        -------
        list[Numeric]
            Smallest singular value per bond, or an empty list if no data.
        """
        if self.data is not None:
            svals_all = helper.singular_values_all(self.data.copy())
            smallest_vals = []
            for svals in svals_all:
                if max_bond is not None and len(svals) < max_bond:
                    smallest_vals += [0]
                else:
                    smallest_vals += [svals[-1]]
            return smallest_vals
        else:
            return []


    def entanglement_entropy_all(self) -> list[Numeric]:
        """Return the bipartite entanglement entropy across every bond."""
        return helper.entanglement_entropy_all(self.data)

    def check_orthog(self):
        """Diagnostic check of the canonical/orthogonality structure of the TN."""
        helper.check_orthog(self.data)

    def get_anchor_tens(self) -> qtn.Tensor:
        """Return the anchor tensor (the site holding any ancilla indices).

        Returns
        -------
        qtn.Tensor
            Tensor at the anchor index, or None if there is no data.
        """
        if self.data is not None:
            return self.data[self.get_anchor_ind()]

    def get_anchor_ind(self) -> int:
        """Return the anchor site index (always 0)."""
        return 0

    def transpose(self, inplace=True, mangle_inner=False):
        """Transpose an MPO by swapping its upper and lower physical indices.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        mangle_inner : bool
            Rename inner bond indices during the flip to avoid collisions.

        Returns
        -------
        GridTN1D
            The transposed operator.
        """
        gtn = self if inplace else self.copy()
        helper.mpo_flip_upper_lower(gtn.data, inplace=True, mangle_inner=mangle_inner)
        return gtn

    def get_like_iden(self):
        """Build an identity MPO matching this state's physical dimensions.

        Physical dimensions are read from the MPO upper indices, MPS site
        indices, or the grid shape depending on the data type. The result may
        have a different length L than the grid.

        Returns
        -------
        GridTN1D
            Identity MPO wrapped as a GridTN1D on this grid.
        """
        L = self.data.L

        if self.data_type is DataType.MPO:
            phys_dims = [self.data[i].ind_size(self.data.upper_ind_id.format(i)) for i in range(L)]
        elif self.data_type is DataType.MPS:
            phys_dims = [self.data[i].ind_size(self.data.site_ind_id.format(i)) for i in range(L)]
        else:
            phys_dims = self.grid.shape()

        if L >= 2:
            iden_list = [np.array([np.eye(phys_dims[0])])]
            for i in range(1, L - 1):
                iden_list += [np.array([[np.eye(phys_dims[i])]])]
            iden_list += [np.array([np.eye(phys_dims[L - 1])])]
        else:
            raise NotImplementedError
        mpo = qtn.MatrixProductOperator(iden_list, shape='lrud', site_tag_id=self.data.site_tag_id)

        return self.create_like(mpo)

    #######################################
    ## class methods to build common TNs ##
    #######################################

    @classmethod
    def get_ones_mps(cls, grid, site_ind_id='i({})', site_tag_id='X({})'):
        """Build the all-ones vector as an MPS on the given grid.

        Parameters
        ----------
        grid : Grid1D
            Grid to build the MPS on.
        site_ind_id : str
            Format string for site (physical) index names.
        site_tag_id : str
            Format string for site tags.

        Returns
        -------
        GridTN1D
            The all-ones MPS.
        """
        # return cls(grid, grid.get_ones_mps(site_ind_id,site_tag_id))
        return grid.get_ones_mps(site_ind_id, site_tag_id)
        # mps_dict = {ax.axID: ax.get_ones_mps(site_ind_id, site_tag_id) for ax in grid.axes}
        # # return cls(grid, data=grid.layout.make_mps_ndim(mps_dict))
        # new_mps = grid.make_mps_ndim(mps_dict)
        # return cls(grid, data=new_mps)

    @classmethod
    def get_iden_mpo(cls, grid, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})'):
        """Build the identity MPO on the given grid.

        Parameters
        ----------
        grid : Grid1D
            Grid to build the MPO on.
        upper_ind_id : str
            Format string for upper (output) physical index names.
        lower_ind_id : str
            Format string for lower (input) physical index names.
        site_tag_id : str
            Format string for site tags.

        Returns
        -------
        GridTN1D
            The identity MPO.
        """
        # return cls(grid, grid.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id))
        return grid.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id)
        # mpo_dict = {ax.axID: ax.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id)
        #             for ax in grid.axes}
        # # return cls(grid, data=grid.layout.make_mpo_ndim(mpo_dict))
        # new_mpo = grid.make_mpo_ndim(mpo_dict)
        # return cls(grid, new_mpo)

    @classmethod
    def get_select_elem_mps(cls, grid, inds, site_ind_id='i({})', site_tag_id='X({})'):
        """Build an MPS that selects the grid elements specified by ``inds``.

        Parameters
        ----------
        grid : Grid1D
            Grid to build the MPS on.
        inds : Sequence
            Per-axis indices identifying the elements to select.
        site_ind_id : str
            Format string for site (physical) index names.
        site_tag_id : str
            Format string for site tags.

        Returns
        -------
        GridTN1D
            The selector MPS.
        """
        # return cls(grid, grid.get_select_elems_mps(inds, site_ind_id, site_tag_id))
        return grid.get_select_elems_mps(inds, site_ind_id, site_tag_id)
        # axes = grid.axes
        # assert (len(axes) == len(inds)), 'number of inds needs to match number of dims'
        # mps_dict = {axes[x].axID: axes[x].get_select_elem_mps(inds[x], site_ind_id, site_tag_id)
        #             for x in range(len(axes))}
        # return cls(grid, data=grid.layout.make_mps_ndim(mps_dict))

    @classmethod
    def get_select_elem_mpo(cls, grid, inds, upper_ind_id='i({})', lower_ind_id='o({})', site_tag_id='X({})'):
        """Build an MPO that selects the grid elements specified by ``inds``.

        Parameters
        ----------
        grid : Grid1D
            Grid to build the MPO on.
        inds : Sequence
            Per-axis indices identifying the elements to select.
        upper_ind_id : str
            Format string for upper (output) physical index names.
        lower_ind_id : str
            Format string for lower (input) physical index names.
        site_tag_id : str
            Format string for site tags.

        Returns
        -------
        GridTN1D
            The selector MPO.
        """
        # return cls(grid, grid.get_select_elems_mpo(inds, upper_ind_id, lower_ind_id, site_tag_id))
        return grid.get_select_elems_mpo(inds, upper_ind_id, lower_ind_id, site_tag_id)
        # axes = grid.axes
        # assert (len(axes) == len(inds)), 'number of inds needs to match number of dims'
        # mpo_dict = {axes[x].axID: axes[x].get_select_elem_mpo(inds[x], upper_ind_id, lower_ind_id, site_tag_id)
        #             for x in range(len(axes))}
        # # return cls(grid, data=grid.layout.make_mpo_ndim(mpo_dict))
        # new_mpo = grid.make_mpo_ndim(mpo_dict)
        # return cls(grid, new_mpo)

    @classmethod
    def from_dense_state(cls, data, grid, site_ind_id='i({})', site_tag_id='T({})', split_opts=None, axes=None):
        """Construct a GridTN1D MPS from a dense state array.

        Parameters
        ----------
        data : ndarray
            Dense state to quantize into an MPS.
        grid : Grid1D
            Grid defining the quantization layout.
        site_ind_id : str
            Format string for site (physical) index names.
        site_tag_id : str
            Format string for site tags.
        split_opts : dict, optional
            Options forwarded to the SVD splitting during MPS construction.
        axes : optional
            Unused placeholder for axis ordering.

        Returns
        -------
        GridTN1D
            The resulting MPS state.
        """
        mps = grid.map_state_to_mps(data, site_ind_id, site_tag_id, split_opts=split_opts)
        return cls(grid, data=mps)

    @classmethod
    def from_dense_operator(cls, data, grid, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='T({})',
                            split_opts=None, **kwargs):
        """Construct a GridTN1D MPO from a dense operator array.

        Parameters
        ----------
        data : ndarray
            Dense operator to quantize into an MPO.
        grid : Grid1D
            Grid defining the quantization layout.
        upper_ind_id : str
            Format string for upper (output) physical index names.
        lower_ind_id : str
            Format string for lower (input) physical index names.
        site_tag_id : str
            Format string for site tags.
        split_opts : dict, optional
            Options forwarded to the SVD splitting during MPO construction.
        **kwargs
            Additional keyword arguments (currently unused).

        Returns
        -------
        GridTN1D
            The resulting MPO operator.
        """
        mpo = grid.map_operator_to_mpo(data, upper_ind_id, lower_ind_id, site_tag_id, split_opts=split_opts)
        return cls(grid, data=mpo)

    def get_data(self, ax_order=None, ax_select: Optional[dict[int]] = None, pad_data=False):
        """Contract the tensor network back to a dense state/operator array.

        Handles MPS and MPO data, tracking ancilla bonds on the boundary
        tensors and optionally transposing the dense result to a requested axis
        order.

        Parameters
        ----------
        ax_order : Sequence[Axis], optional
            Desired ordering of axes in the dense output.
        ax_select : dict[int], optional
            Per-axis index to slice out before contracting.
        pad_data : bool
            Unused placeholder flag.

        Returns
        -------
        ndarray or None
            Dense array, or None if there is no data.
        """
        if self.data is None:
            return None

        L = self.data.L
        if isinstance(self.data, qtn.MatrixProductState):
            shared, not_shared = self.data[0].filter_bonds(self.data[1])
            ancilla_left_inds = tuple([ix for ix in not_shared if ix != self.data.site_ind(0)])
            ancilla_left = tuple([self.data[0].ind_size(ix) for ix in ancilla_left_inds])

            shared, not_shared = self.data[-1].filter_bonds(self.data[-2])
            ancilla_right_inds = tuple([ix for ix in not_shared if ix != self.data.site_ind(L - 1)])
            ancilla_right = tuple([self.data[-1].ind_size(ix) for ix in ancilla_right_inds])

            out = self.grid.map_mps_to_state(self, ax_select=ax_select,
                                             ancilla_right=ancilla_right, ancilla_right_inds=ancilla_right_inds,
                                             ancilla_left=ancilla_left, ancilla_left_inds=ancilla_left_inds)

            if ax_order is not None:
                axT = [np.nonzero(np.array(self.grid.axes, dtype=object) == ax)[0].item() for ax in ax_order]
                out.transpose(axT)

            return out

        elif isinstance(self.data, qtn.MatrixProductOperator):
            shared, not_shared = self.data[0].filter_bonds(self.data[1])
            ancilla_left_inds = tuple([ix for ix in not_shared if
                                       not (ix == self.data.upper_ind(0) or ix == self.data.lower_ind(0))])
            ancilla_left = tuple([self.data[0].ind_size(ix) for ix in ancilla_left_inds])

            shared, not_shared = self.data[-1].filter_bonds(self.data[-2])
            ancilla_right_inds = tuple([ix for ix in not_shared if
                                        not (ix == self.data.upper_ind(L - 1) or ix == self.data.lower_ind(L - 1))])
            ancilla_right = tuple([self.data[-1].ind_size(ix) for ix in ancilla_right_inds])

            out = self.grid.map_mpo_to_operator(self, ax_select=ax_select,
                                                ancilla_right=ancilla_right, ancilla_right_inds=ancilla_right_inds,
                                                ancilla_left=ancilla_left, ancilla_left_inds=ancilla_left_inds)

            if ax_order is not None:
                axTo = [np.nonzero(self.grid.axes == ax) for ax in ax_order]
                axTi = [len(axTo) + x for x in axTo]
                out.transpose(axTo + axTi)

            return out

        elif self.data is None:
            return None

    #######################
    ## MPS/MPX functions ##
    #######################

    def _parse_input_TN1D(self, other):
        """Validate and unwrap another operand to a bare 1-D tensor network.

        Parameters
        ----------
        other : GridTN1D or qtn.TensorNetwork1D
            Operand to validate; a GridTN1D must share this grid, a raw MPX must
            match the grid shape.

        Returns
        -------
        qtn.TensorNetwork1D
            The unwrapped data of ``other``.
        """
        if isinstance(other, type(self)):
            assert (other.grid == self.grid), 'self and other gridTN need to live on the same grids'
            other = other.data
        else:
            mpx_shape = tuple([other.phys_dim(i) for i in range(other.L)])
            assert (mpx_shape == self.grid.shape), 'self and mpo need to have the same shape'
        return other

    def _match_grids(self, other, target_data_type=None):
        """Pad self and/or other so both live on a common grid.

        Whichever operand has the smaller axis set is padded up to the other's
        grid via ``pad_gtn_to_grid``; incompatible grids raise ValueError.

        Parameters
        ----------
        other : GridTN1D
            The other operand to reconcile grids with.
        target_data_type : DataType, optional
            Data type (MPS/MPO) the padded operand should be coerced to.

        Returns
        -------
        tuple[GridTN1D, GridTN1D]
            The (possibly padded) self and other on a shared grid.
        """
        grid_mpx1 = self
        if grid_mpx1.grid != other.grid:
            if set(other.grid.axes).issubset(set(grid_mpx1.grid.axes)):
                # print('mps1 > mps2')
                other = grid_mpx1.grid.pad_gtn_to_grid(other, target_data_type=target_data_type)
            elif set(grid_mpx1.grid.axes).issubset(set(other.grid.axes)):
                # print('mps1 < mps2')
                grid_mpx1 = other.grid.pad_gtn_to_grid(grid_mpx1, target_data_type=target_data_type)
                print('warning: elemental multiply not an inplace operation')
            else:
                # print('grid 1 axes', grid_mpx1.grid.axes, other.grid.axes)
                raise ValueError(f'grid compatibility issue with {grid_mpx1.grid}, {other.grid}')
        return grid_mpx1, other

    # @profile
    def apply(self, other, inplace=False, zipup=True, compress_type=CompressType.SVD, compress=False,
              compress_opts=None, add_cc=False, **kwargs) -> 'GridTN1D':
        """Apply operator ``other`` to this state, with optional compression.

        Grids are padded to match, then the product is formed either by a
        zip-up SVD apply or a plain apply followed by SVD/MG/DMRG compression
        depending on ``compress_type``.

        Parameters
        ----------
        other : GridTN1D
            Operator (MPO) or scalar to apply; None returns None.
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        zipup : bool
            Use the zip-up SVD apply algorithm when ``compress_type`` is SVD.
        compress_type : CompressType
            Compression backend (SVD, MG, or DMRG).
        compress : bool or int
            Whether to compress the product (compression level when truthy).
        compress_opts : dict, optional
            Overrides for compression parameters (e.g. ``max_bond``, ``form``).
        add_cc : bool
            Convert the state to its full real-Fourier representation first.
        **kwargs
            Extra options (e.g. ``init_guess``) forwarded to the MG/DMRG paths.

        Returns
        -------
        GridTN1D
            The resulting state after application.

        """
        # zipup = True
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        if other is None:
            return None

        grid_mpx1 = self if inplace else self.copy()
        grid_mpx1.is_constant *= other.is_constant
        grid_mpx1.constant_axes = [ax for ax in grid_mpx1.constant_axes if ax in other.constant_axes]

        ## pad grid if necessary
        grid_mpx1, other = grid_mpx1._match_grids(other, target_data_type=DataType.MPO)
        # print('padded other', other)

        if other.data_type == DataType.Num:
            grid_mpx1.scalar_multiply(other, inplace=True)
            return grid_mpx1

        # other = self._parse_input_TN1D(other)
        other = other.data
        assert (isinstance(other, qtn.MatrixProductOperator)), 'other needs to be an MPO type'

        grid_data = grid_mpx1.data
        if add_cc:
            # if not(len(grid_data.shape) == len(other.shape) or len(grid_data.shape)*2 == len(other.shape)):
            grid_data = RealFourierBasis.convert_mps_to_full(grid_mpx1.grid, grid_data)

        # print('apply 1D self', grid_data)
        # print('apply 1D other', other)

        # print('1D apply, zipup?', zipup, compress)
        if zipup and compress_type is CompressType.SVD:  # or compress:  ### and compress?
            form = compress_opts.get('form', 'right') if compress_opts is not None else 'right'

            # DMAX = compress_opts.get('max_bond', None) if compress_opts is not None else None
            # compress_back = compress  # (DMAX is not None)
            # print('zipup', compress_back)

            # new_mpx = helper.apply_zipup(other, grid_mpx1.data, compress=compress, compress_opts=compress_opts)
            new_mpx = helper.apply_zipup(other, grid_data, compress=compress, compress_opts=compress_opts)
            # print('zipup check orthog new mpx')
            # print(helper.check_right_orthog(new_mpx))
            grid_mpx1.canon_site = 0 if form == 'right' else grid_mpx1.L - 1
            # print(new_mpx.calc_current_orthog_center(), grid_mpx1.canon_site)

            # if compress_opts['max_bond'] is None:
            #     ## canonicalize but don't compress
            #     new_mpx = helper.apply_zipup(mpo, grid_mpx1.data, compress=False, compress_opts=compress_opts)
            # else:
            #     ## canonicalize and compress
            #     new_mpx = helper.apply_zipup(mpo, grid_mpx1.data, compress=True, compress_opts=compress_opts)
        else:
            ## no compression -- simple apply method
            # new_mpx = helper.apply(other, grid_mpx1.data)
            if compress:
                # print('APPLY', compress_type)
                if compress_type is CompressType.SVD or compress_opts.get('max_bond', None) is None:
                    new_mpx = helper.apply(other, grid_data)
                    helper.compress(new_mpx, compress_opts=compress_opts)
                elif compress_type is CompressType.MG:
                    mps_coarseness_inds = grid_mpx1.grid.get_coarseness_levels()

                    # new_mpx, err, is_conv = helper_mg.mg_apply(grid_data, other,
                    #                                            init_guess=kwargs.get('init_guess', None),
                    #                                            opt_nsites=DEFAULT_OPT_NSITES,
                    #                                            max_bond=compress_opts.get('max_bond', None)
                    #                                                     if compress_opts is not None else None,
                    #                                            solve_type=DEFAULT_SOLVE_TYPE,
                    #                                            coarseness_mps_inds=mps_coarseness_inds)

                    new_mpx = helper.apply(other, grid_data)
                    # print('mg compress', 'target norm', helper.norm(new_mpx))
                    if helper.norm(new_mpx) < np.sqrt(CUTOFF):
                        new_mpx = None
                    else:
                        new_mpx, err, is_conv = helper_mg.mg_compress(new_mpx,
                                                                      init_guess=kwargs.get('init_guess', None),
                                                                      opt_nsites=DEFAULT_OPT_NSITES,
                                                                      max_bond=compress_opts.get('max_bond', None)
                                                                      if compress_opts is not None else None,
                                                                      coarseness_mps_inds=mps_coarseness_inds)
                        new_mpx.view_as(qtn.MatrixProductState, inplace=True)
                elif compress_type is CompressType.DMRG:
                    # new_mpx, err, is_conv = helper_dmrg.dmrg_apply(grid_data, other,
                    #                                                init_guess=kwargs.get('init_guess', None),
                    #                                                opt_nsites=DEFAULT_OPT_NSITES,
                    #                                                max_bond=compress_opts.get('max_bond', None)
                    #                                                         if compress_opts is not None else None,
                    #                                                solve_type=DEFAULT_SOLVE_TYPE)

                    new_mpx = helper.apply(other, grid_data)
                    if helper.norm(new_mpx) < np.sqrt(CUTOFF):
                        new_mpx = None
                    else:
                        new_mpx, err, is_conv = helper_mg.dmrg_compress(new_mpx,
                                                                        init_guess=kwargs.get('init_guess', None),
                                                                        opt_nsites=DEFAULT_OPT_NSITES,
                                                                        max_bond=compress_opts.get('max_bond', None)
                                                                        if compress_opts is not None else None)

                        new_mpx.view_as(qtn.MatrixProductState, inplace=True)
                else:
                    raise ValueError('compress type not recognized', compress_type)
            else:
                new_mpx = helper.apply(other, grid_data)

        # new_mpx = helper.apply(mpo, grid_mpx1.data) # , compress=compress, compress_opts=compress_opts)
        # if compress:
        #     helper.compress(new_mpx, compress_opts=compress_opts)

        ## not inplace update of qtn.TensorNetwork obj.
        grid_mpx1.data = new_mpx
        return grid_mpx1

    def apply_rdm(self, other, bra_self=None, bra_other=None, left_env=None, right_env=None,
                  direction=1, open_end=False, inplace=False, compress=True, compress_opts=None, verbose=False,
                  ) -> Union['GridTN1D', tuple['GridTN1D', 'qtn.Tensor']]:
        """Apply an MPO to this state using a reduced-density-matrix sweep.

        Grids are matched, then ``helper.apply_rdm`` performs a directional
        sweep with optional left/right environments and compression. With
        ``open_end`` the central tensor C is returned alongside the new state.

        Parameters
        ----------
        other : GridTN1D
            Operator (MPO) to apply; None returns None.
        bra_self : GridTN1D, optional
            Bra state for self (currently unused inside the sweep).
        bra_other : GridTN1D, optional
            Bra operator (currently unused inside the sweep).
        left_env : qtn.Tensor, optional
            Precomputed left environment tensor.
        right_env : qtn.Tensor, optional
            Precomputed right environment tensor.
        direction : int
            Sweep direction; sets the resulting canonical site.
        open_end : bool
            If True, leave the boundary open and also return the C tensor.
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        compress : bool
            Compress during the sweep.
        compress_opts : dict, optional
            Overrides for compression parameters.
        verbose : bool
            Print diagnostic information.

        Returns
        -------
        GridTN1D or tuple[GridTN1D, qtn.Tensor]
            The resulting state, plus the C tensor when ``open_end`` is True.
        """

        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        if other is None:
            return None

        grid_mpx1 = self if inplace else self.copy()
        grid_mpx1.is_constant *= other.is_constant
        grid_mpx1.constant_axes = [ax for ax in grid_mpx1.constant_axes if ax in other.constant_axes]

        ## pad grid if necessary
        grid_mpx1, other = grid_mpx1._match_grids(other)
        # print('padded other', other)

        # other = self._parse_input_TN1D(other)
        other = other.data
        assert (isinstance(other, qtn.MatrixProductOperator)), 'other needs to be an MPO type'

        grid_data = grid_mpx1.data
        grid_data.site_ind_id = other.lower_ind_id
        # bra_other = bra_other.data if bra_other is not None else None
        # bra_self = bra_self.data if bra_self is not None else None
        out = helper.apply_rdm(other, grid_data,  # bra_mpo1=bra_other, bra_mps2=bra_self,
                               direction=direction, left_env=left_env, right_env=right_env,
                               compress=compress, compress_opts=compress_opts, open_end=open_end, verbose=verbose)

        ## not inplace update of qtn.TensorNetwork obj.
        grid_mpx1.canon_site = 0 if direction < 0 else grid_mpx1.L - 1
        # print('apply rdm 1 set canon site', grid_mpx1.canon_site)
        if open_end:
            new_mpx, C_tens = out
            grid_mpx1.data = new_mpx
            return grid_mpx1, C_tens
        else:
            new_mpx = out
            grid_mpx1.data = new_mpx
            return grid_mpx1

    def solve(self, operator: 'GridTN', compress_type: CompressType, inplace=False, use_A2=False, compress_opts=None,
              is_H=False, init_guess: Optional['GridTN'] = None, verbose_output=False, **kwargs
              ) -> Union[tuple['GridTN1D', float, bool], 'GridTN1D']:
        """Solve the linear system A x = b for x via local optimization.

        Here self is the right-hand side b and ``operator`` is A; the DMRG
        solver (optionally the A^2 normal-equation variant) returns x.

        Parameters
        ----------
        operator : GridTN
            The system matrix A (MPO); None returns self unchanged.
        compress_type : CompressType
            Solver backend; DMRG is supported, MG raises NotImplementedError.
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        use_A2 : bool
            Solve the squared/normal-equation system via ``dmrg_solve_2``.
        compress_opts : dict, optional
            Compression options; ``max_bond`` caps the solution bond dimension.
        is_H : bool
            Treat A as Hermitian, forwarded to the DMRG solver.
        init_guess : GridTN, optional
            Initial guess for x.
        verbose_output : bool
            If True, also return the residual error and convergence flag.
        **kwargs
            Additional options forwarded to the DMRG solver.

        Returns
        -------
        GridTN1D or tuple[GridTN1D, float, bool]
            The solution x, plus (error, is_converged) if ``verbose_output``.
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        if operator is None:
            return self if inplace else self.copy()

        grid_mpx1 = self if inplace else self.copy()
        grid_mpx1.is_constant *= operator.is_constant
        grid_mpx1.constant_axes = [ax for ax in grid_mpx1.constant_axes if ax in operator.constant_axes]

        ## pad grid if necessary
        grid_mpx1, operator = grid_mpx1._match_grids(operator)
        # print('padded other', other)

        if compress_type is CompressType.DMRG:
            # print('solve compress opts', compress_opts)
            init_guess_data = init_guess.data if init_guess is not None else None
            max_bond = compress_opts.get('max_bond', None) if compress_opts is not None else None
            if use_A2:
                out_data, err, is_conv = \
                    helper_dmrg_2.dmrg_solve_2(grid_mpx1.data, operator.data, init_guess=init_guess_data,
                                               opt_nsites=DEFAULT_OPT_NSITES, is_H=is_H,
                                               max_bond=max_bond, solve_type=DEFAULT_SOLVE_TYPE, **kwargs,
                                               )
            else:
                out_data, err, is_conv = \
                    helper_dmrg.dmrg_solve(grid_mpx1.data, operator.data, init_guess=init_guess_data,
                                           opt_nsites=DEFAULT_OPT_NSITES, is_H=is_H,
                                           max_bond=max_bond, solve_type=DEFAULT_SOLVE_TYPE, **kwargs,
                                           )

        elif compress_type is CompressType.MG:
            raise NotImplementedError('MG solver not yet implemented')
        else:
            raise ValueError('solver not compatible with', compress_type)

        grid_mpx1.data = out_data
        if verbose_output:
            return grid_mpx1, err, is_conv
        return grid_mpx1

    def add(self, gtn_mpx2, zipup=False, inplace=False, compress_type=CompressType.SVD,
            compress=False, compress_opts=None, **kwargs) -> 'GridTN1D':
        """Add another tensor network of matching type to this one.

        Grids are padded to match; MPS/MPO addition is dispatched by
        ``compress_type`` (SVD, MG, or DMRG) with optional zip-up and
        compression, updating the canonical-site bookkeeping.

        Parameters
        ----------
        gtn_mpx2 : GridTN1D
            Operand to add (same MPS/MPO type); None returns self unchanged.
        zipup : bool
            Use a zip-up SVD addition where supported.
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        compress_type : CompressType
            Addition/compression backend (SVD, MG, or DMRG).
        compress : bool or int
            Whether to compress the sum (compression level when truthy).
        compress_opts : dict, optional
            Overrides for compression parameters.
        **kwargs
            Additional options forwarded to the backend.

        Returns
        -------
        GridTN1D
            The summed state.
        """
        grid_mpx1 = self if inplace else self.copy(deep=True)

        if gtn_mpx2 is None:
            return grid_mpx1

        elif isinstance(gtn_mpx2, GridTN1D):
            if gtn_mpx2.data is None:
                return grid_mpx1

            ## pad grids as necessary
            grid_mpx1, gtn_mpx2 = grid_mpx1._match_grids(gtn_mpx2, target_data_type=grid_mpx1.data_type)
            # if gtn_mpx2.grid != self.grid:
            #     if set(gtn_mpx2.grid.axes).issubset(self.grid.axes):
            #         gtn_mpx2 = self.grid.pad_gtn_to_grid(gtn_mpx2)
            #     else:
            #         raise ValueError(f'gtn_mpx2 on grid {gtn_mpx2.grid} not compatible with self on grid {self.grid}')

            if grid_mpx1.data is None:
                try:
                    grid_mpx1.data = gtn_mpx2.data.copy()
                except AttributeError:
                    grid_mpx1.data = gtn_mpx2.data
                grid_mpx1.is_constant = gtn_mpx2.is_constant
                grid_mpx1.constant_axes = gtn_mpx2.constant_axes
                return grid_mpx1

        # print('gtn_mpx2', gtn_mpx2)
        mpx2 = self._parse_input_TN1D(gtn_mpx2)

        if compress_opts is None:
            compress_opts = {}

        if mpx2 is None:
            return grid_mpx1

        if isinstance(grid_mpx1.data, qtn.MatrixProductOperator):
            assert (isinstance(mpx2, qtn.MatrixProductOperator)), 'other needs to be an MPO type'
            grid_mpx1.data = helper.add_MPO(grid_mpx1.data, mpx2, inplace=True, zipup=zipup,
                           compress=compress, compress_opts=compress_opts)
        elif isinstance(grid_mpx1.data, qtn.MatrixProductState):
            assert (isinstance(mpx2, qtn.MatrixProductState)), 'other needs to be an MPS type'
            # print('compress type', compress_type)
            if compress_type is CompressType.SVD:
                grid_mpx1.data = helper.add_MPS(grid_mpx1.data, mpx2, inplace=True, zipup=zipup,
                               compress=compress, compress_opts=compress_opts)
            elif compress_type is CompressType.MG:
                mps_coarseness_inds = grid_mpx1.grid.get_coarseness_levels()
                helper_mg.mg_add_mps([grid_mpx1.data, mpx2], opt_nsites=DEFAULT_OPT_NSITES, **compress_opts,
                                     solve_type=DEFAULT_SOLVE_TYPE,
                                     coarseness_mps_inds=mps_coarseness_inds)
            elif compress_type is CompressType.DMRG:
                opt_nsites = 2      # DEFAULT_OPT_NSITES
                # out, err, is_conv = helper_dmrg.dmrg_add_mps([grid_mpx1.data, mpx2], opt_nsites=opt_nsites, **compress_opts,
                #                                              solve_type=DEFAULT_SOLVE_TYPE)
                out = helper.add_MPS(grid_mpx1.data, mpx2, inplace=True, compress=False)
                out, err, is_conv = helper_dmrg.dmrg_compress(out, opt_nsites=opt_nsites, **compress_opts)
                print('dmrg add err', err, 'is conv', is_conv)
                grid_mpx1.data = out

        elif grid_mpx1.data is None:
            grid_mpx1.data = mpx2.copy() if mpx2 is not None else None
        else:
            raise TypeError(f'self needs to be an MPO, MPS, or NoneType, not {type(self.data)}')

        if compress:
            if compress_opts.get('form', 'right') == 'right':
                self.canon_site = 0
            else:
                self.canon_site = self.L - 1
        else:
            self.canon_site = -1

        grid_mpx1.is_constant *= gtn_mpx2.is_constant
        grid_mpx1.constant_axes = [ax for ax in grid_mpx1.constant_axes if ax in gtn_mpx2.constant_axes]

        return grid_mpx1

    # def add_dmrg(self, *other_gtns: 'GridTN1D', inplace=False, compress_opts=None, **dmrg_opts) -> 'GridTN1D':
    #     """ add multiple gtns together using DMRG solver
    #     """
    #     from local_solvers.local_dmrg_eval import local_dmrg_evaluator, Term_DMRG
    #
    #     DMAX = compress_opts.get('max_bond', None) if compress_opts is not None else None
    #
    #     gtn = self if inplace else self.copy()
    #     is_mps = self.data_type == DataType.MPS
    #
    #     self_tt = self.grid.gtn_to_dmrg_format(gtn, is_mps=is_mps)
    #     terms = [Term_DMRG(self_tt.copy())]
    #     terms += [Term_DMRG(self.grid.gtn_to_dmrg_format(ogtn, is_mps=is_mps).copy()) for ogtn in other_gtns]
    #
    #     func_mps = local_dmrg_evaluator(terms, max_bond=DMAX, **dmrg_opts)
    #
    #     gtn = self.grid.dmrg_to_gtn_format(gtn, func_mps, is_mps=is_mps)
    #     return gtn


    def add_subgtn(self, sub_gtn_mpx2: 'GridTN', open_bc=False, inplace=False, zipup=False, compress=False,
                   compress_opts=None, **kwargs) -> 'GridTN':
        """Add a sub-grid tensor network into a contiguous index window of self.

        The sub-network's axes are mapped to a contiguous ``[min, max]`` range
        of self's site indices and added there via ``helper.add_submpx``.

        Parameters
        ----------
        sub_gtn_mpx2 : GridTN
            Sub-grid operand to embed and add; None returns self unchanged.
        open_bc : bool
            Use open boundary conditions at the insertion window.
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        zipup : bool
            Use a zip-up addition where supported.
        compress : bool
            Compress after insertion.
        compress_opts : dict, optional
            Overrides for compression parameters.
        **kwargs
            Additional options forwarded to the helper.

        Returns
        -------
        GridTN
            The state with the sub-network added.
        """
        gtn = self if inplace else self.copy()
        if sub_gtn_mpx2 is None:
            return gtn

        gtn_mpx = gtn.data
        sub_mpx = sub_gtn_mpx2.data
        if sub_mpx is None:
            return gtn

        sub_inds = []
        for ax in sub_gtn_mpx2.grid.axes:
            sub_inds += self.grid.get_inds_in_axis(ax)

        min_ind = np.min(sub_inds)
        max_ind = np.max(sub_inds)

        gtn.data = helper.add_submpx(gtn_mpx, sub_mpx, (min_ind, max_ind + 1), open_bc=open_bc, inplace=True,
                                     compress=compress,
                                     compress_opts=compress_opts)
        return gtn

    def scalar_multiply(self, scalar_const, inplace=False) -> 'GridTN1D':
        """Multiply the tensor network by a scalar constant.

        Parameters
        ----------
        scalar_const : Numeric
            Scalar factor to multiply into the data.
        inplace : bool
            Mutate self in place if True, else operate on a copy.

        Returns
        -------
        GridTN1D
            The scaled state.
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        grid_mpx1 = self if inplace else self.copy(deep=True)
        helper.scalar_multiply(grid_mpx1.data, scalar_const, inplace=True)
        return grid_mpx1


    def evaluate_func(self, func: Callable, max_bond=None, inplace=False):
        """Evaluate an element-wise function of the state via TT-cross.

        Builds a cross-interpolation term from ``func`` applied to this MPS and
        approximates the result as an MPS with a 2-site local cross evaluator.

        Parameters
        ----------
        func : Callable
            Element-wise function applied to the MPS values.
        max_bond : int, optional
            Bond-dimension cap for the cross-interpolated result.
        inplace : bool
            Mutate self in place if True, else return a copy.

        Returns
        -------
        GridTN1D
            The state holding ``func`` applied element-wise.
        """
        # from local_solvers.local_cross_eval_old import local_cross_evaluator, Term_Cross
        from local_solvers.local_cross_eval import local_cross_evaluator, Term_Cross

        term1c = Term_Cross(self.data.copy(), mps_func=func)
        nsites = 2  # if (max_bond is None or max_bond <= self.max_bond()) else 2
        func_mps = local_cross_evaluator([term1c], nsites=nsites, max_bond=max_bond)

        if inplace:
            self.data = func_mps
            return self
        else:
            return self.create_like(new_data=func_mps)


    def canonize(self, inplace=True, scale=True, form='right', i=None, cur_orthog=None) -> 'GridTN1D':
        """Put the MPS/MPO into mixed canonical form about a site.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        scale : bool
            Absorb the norm into the orthogonality center if True.
        form : str
            Default target form ('right' -> site 0, 'left' -> site L-1) used
            when ``i`` is not given.
        i : int, optional
            Explicit orthogonality-center site.
        cur_orthog : int, optional
            Current orthogonality center, used to shorten the canonization.

        Returns
        -------
        GridTN1D
            The canonized state with ``canon_site`` set to ``i``.
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        if i is None:
            if form == 'right':
                i = 0
            elif form == 'left':
                i = self.L - 1

        grid_mpx1 = self if inplace else self.copy(deep=True)
        helper.canonize(grid_mpx1.data, scale=scale, i=i, cur_orthog=cur_orthog)
        grid_mpx1.canon_site = i
        return grid_mpx1

    def canonize_axes(self, axes: Sequence['Axis'], inplace=True, scale=True):
        """Canonize so the orthogonality center lies within the given axes.

        Parameters
        ----------
        axes : Sequence[Axis]
            Axes whose index range should contain the orthogonality center.
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        scale : bool
            Absorb the norm into the orthogonality center if True.

        Returns
        -------
        GridTN1D
            The canonized state.
        """
        canon_i = self.get_canon_site_from_axes(axes)
        return self.canonize(inplace=inplace, scale=scale, i=canon_i)

    def get_canon_site_from_axes(self, axes: Sequence['Axis']):
        """Compute the canonical-center site contained within the given axes.

        Parameters
        ----------
        axes : Sequence[Axis]
            Axes that must lie contiguously on one side of the MPS.

        Returns
        -------
        int
            The site index to use as orthogonality center.
        """
        ax_inds = np.sort([self.grid.axes.index(ax) for ax in axes])
        if len(ax_inds) == 1 or all(np.diff(ax_inds) == 1):
            if ax_inds[-1] != self.grid.ndim - 1:
                canon_i = sum([ax.L for ax in self.grid.axes[:1 + ax_inds[-1]]])
                canon_i -= 1
            else:  ## last (right-most) axis in grid
                canon_i = self.L - sum([ax.L for ax in axes])  # - 1
            print('canon i', axes, canon_i)
        else:
            raise ValueError('axes must either all be on left or right of MPS')
        return canon_i

    def compress(self, inplace=True, verbose=False, canonize=True, compress_type=CompressType.SVD, compress_opts=None,
                 sub_compress_opts=None, norm_cutoff=None,
                 conservative=False,
                 **kwargs) -> 'GridTN1D':
        """Truncate the bond dimensions of the tensor network.

        Dispatches by ``compress_type``: SVD truncation (optionally a
        mass-conserving orthogonal compression against the ones vector), or
        local DMRG/MG compression (MPS only). Updates ``canon_site`` from the
        resulting form.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        verbose : bool
            Print diagnostic information.
        canonize : bool
            Canonize before truncating in the SVD path.
        compress_type : CompressType
            Compression backend (SVD, DMRG, or MG).
        compress_opts : dict, optional
            Overrides for compression parameters (e.g. ``max_bond``, ``form``,
            ``do_midpt``).
        sub_compress_opts : dict, optional
            Secondary compression options (unused in this path).
        norm_cutoff : float, optional
            Norm threshold below which the SVD compression drops the state.
        conservative : bool
            Use the conservative (mass-preserving) orthogonal compression.
        **kwargs
            Additional options forwarded to the SVD compressor.

        Returns
        -------
        GridTN1D
            The compressed state.
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        if isinstance(self.data, (int, float, complex, np.number)):
            return self if inplace else self.create_like()

        grid_mpx1 = self if inplace else self.copy(deep=True)

        if compress_type is CompressType.SVD:
            if conservative:
                print('conservative compress')
                ## mass conservation
                basis = self.grid.get_ones_mps()

                # plt.figure()
                # plt.imshow(basis.get_data())
                # plt.title('conservative basis')
                # plt.show()

                norm = basis.frobenius_norm()
                basis.scalar_multiply(1./norm, inplace=True)
                # proj_val = grid_mpx1.ovlp(basis)

                from local_solvers.local_dmrg_eval_orthog import local_orthogonal_compress_2

                out = local_orthogonal_compress_2(grid_mpx1.data, [basis.data],
                                                  max_bond=compress_opts['max_bond'])
                grid_mpx1.data = out

                # print('grid mpx1', grid_mpx1.ovlp(basis) - proj_val, np.abs( grid_mpx1.ovlp(basis) - proj_val))

                # grid_mpx1.data = helper.conservative_compress(grid_mpx1.data, bases=[basis.data],
                #                                               canonize=canonize, compress_opts=compress_opts)
            else:
                # print('compress compress opts', compress_opts)
                grid_mpx1.data = helper.compress(grid_mpx1.data, canonize=canonize, compress_opts=compress_opts,
                                                 norm_cutoff=norm_cutoff, verbose=verbose, **kwargs)

            if grid_mpx1.data is not None:
                if compress_opts is None:
                    grid_mpx1.canon_site = grid_mpx1.L - 1
                else:
                    if compress_opts.get('do_midpt', False):
                        grid_mpx1.canon_site = -1
                    else:
                        if compress_opts.get('form', 'right') == 'right':
                            grid_mpx1.canon_site = 0
                        else:
                            grid_mpx1.canon_site = grid_mpx1.L - 1

        elif compress_type is CompressType.DMRG:
            if grid_mpx1.data_type is DataType.MPO:
                raise NotImplementedError('local compress not implemented for MPOs')
            # print('compress opts', compress_opts)
            new_data, err, is_conv = helper_dmrg.dmrg_compress(grid_mpx1.data, **compress_opts,
                                                               solve_type=DEFAULT_SOLVE_TYPE)
            grid_mpx1.data = new_data.view_as(qtn.MatrixProductState, inplace=True)

        elif compress_type is CompressType.MG:
            if grid_mpx1.data_type is DataType.MPO:
                raise NotImplementedError('local compress not implemented for MPOs')
            mps_coarseness_inds = grid_mpx1.grid.get_coarseness_levels()
            new_data, err, is_conv = helper_mg.mg_compress(grid_mpx1.data, **compress_opts,
                                                           solve_type=DEFAULT_SOLVE_TYPE,
                                                           coarseness_mps_inds=mps_coarseness_inds)
            grid_mpx1.data = new_data.view_as(qtn.MatrixProductState, inplace=True)

        return grid_mpx1

    def compress_rdm(self, inplace=True, verbose=False, compress_opts=None, sub_compress_opts=None,
                     direction=1, open_end=False, left_env=None, right_env=None, back_compress=True,
                     **kwargs) -> Union['GridTN1D', tuple['GridTN1D', 'qtn.Tensor']]:
        """Compress the MPS using a reduced-density-matrix directional sweep.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        verbose : bool
            Print diagnostic information.
        compress_opts : dict, optional
            Overrides for compression parameters (e.g. ``max_bond``, ``form``,
            ``do_midpt``).
        sub_compress_opts : dict, optional
            Secondary compression options (unused in this path).
        direction : int
            Sweep direction; determines the resulting canonical site.
        open_end : bool
            If True, leave the boundary open and also return the C tensor.
        left_env : qtn.Tensor, optional
            Precomputed left environment tensor.
        right_env : qtn.Tensor, optional
            Precomputed right environment tensor.
        back_compress : bool
            Perform the back-sweep compression step.
        **kwargs
            Additional options forwarded to ``helper.compress_rdm``.

        Returns
        -------
        GridTN1D or tuple[GridTN1D, qtn.Tensor]
            The compressed state, plus the C tensor when ``open_end`` is True.
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        grid_mpx1 = self if inplace else self.copy(deep=True)

        out = helper.compress_rdm(grid_mpx1.data, compress_opts=compress_opts, verbose=verbose, direction=direction,
                                  open_end=open_end, left_env=left_env, right_env=right_env,
                                  back_compress=back_compress, **kwargs)

        grid_mpx1.data = out[0] if open_end else out

        if grid_mpx1.data is not None:
            if compress_opts is None:
                grid_mpx1.canon_site = grid_mpx1.L - 1
            else:
                if compress_opts.get('do_midpt', False):
                    grid_mpx1.canon_site = -1
                else:
                    if compress_opts.get('form', 'right') == 'right':
                        grid_mpx1.canon_site = 0
                    else:
                        grid_mpx1.canon_site = grid_mpx1.L - 1

        if open_end:
            C_tens = out[1]
            return grid_mpx1, C_tens
        else:
            return grid_mpx1


    def get_bases(self, i: int):
        """Extract the left and right basis functions about an orthogonality site.

        Assuming the orthogonality center is at site ``i``, splits the MPS into
        the left and right blocks and enumerates each ancilla branch as a
        separate sub-grid GridTN.

        Parameters
        ----------
        i : int
            Site holding the orthogonality center.

        Returns
        -------
        tuple[list, list]
            Lists of left-block and right-block GridTN basis states.
        """
        data = self.data.copy()

        left_axes = {}
        right_axes = {}

        for ax in self.grid.axes:
            inds = self.grid.get_inds_in_axis(ax)
            left_inds = [ix for ix in inds if ix < i]
            right_inds = [ix for ix in inds if ix > i]
            if len(left_inds) > 0:
                left_axes[ax] = left_inds
            if len(right_inds) > 0:
                right_axes[ax] = right_inds


        gtn_lefts = []
        if i > 0:
            new_left_axes = [ax.create_new(len(vals)) for ax, vals in left_axes.items()]
            grid_left = self.grid.create_like(new_axes=new_left_axes)
            anc_i = data.bond(i, i - 1)
            anc_i_size = data.bond_size(i, i - 1)
            left_mps = data[:i].copy()
            left_mps._L = i
            anc_tens = left_mps[i-1].copy()
            for ix in range(anc_i_size):
                new_anc_tens = anc_tens.isel({anc_i: ix}, inplace=False)
                left_mps[i-1].modify(data=new_anc_tens.data, inds=new_anc_tens.inds)
                gtn_lefts += [grid_left.make_gridTN(left_mps.copy())]

        gtn_rights = []
        if i < self.L - 1:
            new_right_axes = [ax.create_new(len(vals)) for ax, vals in right_axes.items()]
            grid_right = self.grid.create_like(new_axes=new_right_axes)
            right_mps = data[i+1:].copy()
            right_mps._L = len(right_mps.tensors)
            helper_quimb.renumber_mps(right_mps, list(range(i+1, self.L)), list(range(0, self.L-i-1)), inplace=True)

            anc_i = data.bond(i, i + 1)
            anc_i_size = data.bond_size(i, i + 1)
            anc_tens = right_mps[0].copy()
            for ix in range(anc_i_size):
                new_anc_tens = anc_tens.isel({anc_i: ix}, inplace=False)
                right_mps[0].modify(data=new_anc_tens.data, inds=new_anc_tens.inds)
                gtn_rights += [grid_right.make_gridTN(right_mps.copy())]

        return gtn_lefts, gtn_rights



    def expand_subspace(self, subspace_vecs: Sequence['GridTN'], orthog_direction=-1, compress_opts=None,
                        inplace=False):
        """Enlarge the MPS bond space by adding zeroed subspace vectors.

        Implements the basis-expansion of http://arxiv.org/abs/2005.06104 by
        canonizing both self and each subspace vector, zeroing the vector's
        center tensor, and adding it without compression.

        Parameters
        ----------
        subspace_vecs : Sequence[GridTN]
            States whose subspaces are folded into self.
        orthog_direction : int
            Sweep direction: +1 starts at site 0 (final center L-1); -1 starts
            at site L-1 (final center 0).
        compress_opts : dict, optional
            Provides ``max_bond`` and ``cutoff`` for the expansion.
        inplace : bool
            Mutate self in place if True, else operate on a copy.

        Returns
        -------
        GridTN1D
            The state with expanded subspace.
        """
        print('EXPANDING SUBSPACE')
        gtn = self if inplace else self.copy()
        max_bond = None if compress_opts is None else compress_opts.get('max_bond', None)
        cutoff = CUTOFF if compress_opts is None else compress_opts.get('cutoff', None)

        canon_site = 0 if orthog_direction > 0 else gtn.L - 1
        gtn.data.distribute_exponent()
        gtn.canonize(inplace=True, scale=False, i=canon_site)

        for vec in subspace_vecs:
            vec = vec.copy()
            vec.data.distribute_exponent()
            vec.canonize(inplace=True, scale=False, i=canon_site)
            vec.data[canon_site].modify(apply=lambda x: 0 * x)

            gtn.add(vec, inplace=True, compress=False)
        print('expanded gtn', gtn.max_bond())
        return gtn

    def expand_subspace_v2(self, subspace_vecs: Sequence['GridTN'], orthog_direction=-1, compress_opts=None,
                           inplace=False):
        """Enlarge the MPS bond space via per-site projected density matrices.

        Refined variant of the basis-expansion of
        http://arxiv.org/abs/2005.06104: sweeps site by site, projecting each
        subspace vector's density matrix orthogonal to self and keeping the
        leading eigenvectors (bounded by cutoff and max_bond) as new basis
        directions, synchronously updating self and the vectors.

        Parameters
        ----------
        subspace_vecs : Sequence[GridTN]
            States whose subspaces are folded into self.
        orthog_direction : int
            Sweep direction: +1 starts at site 0 (final center L-1); -1 starts
            at site L-1 (final center 0).
        compress_opts : dict, optional
            Provides ``max_bond`` and ``cutoff`` for the expansion.
        inplace : bool
            Mutate self in place if True, else operate on a copy.

        Returns
        -------
        GridTN1D
            The state with expanded subspace.
        """
        print('EXPANDING SUBSPACE')
        gtn = self if inplace else self.copy()
        max_bond = None if compress_opts is None else compress_opts.get('max_bond', None)
        cutoff = CUTOFF if compress_opts is None else compress_opts.get('cutoff', None)

        canon_site = 0 if orthog_direction > 0 else gtn.L - 1
        gtn.data.distribute_exponent()
        gtn.canonize(inplace=True, scale=False, i=canon_site)

        for vec in subspace_vecs:
            vec.data.distribute_exponent()
            vec.canonize(inplace=True, scale=False, i=canon_site)

        def _get_site_rho(tens, open_inds):
            """Form the single-site reduced density matrix sum_ij A_ijk A*_ijk'.

            Parameters
            ----------
            tens : qtn.Tensor
                Site tensor whose density matrix is computed.
            open_inds : Sequence[str]
                Indices kept open (not contracted) on both bra and ket.

            Returns
            -------
            qtn.Tensor
                The reduced density matrix tensor over the open indices.
            """
            tens_c = tens.conj()
            tens_c.reindex({ind: ind + '_' for ind in open_inds}, inplace=True)
            out = qtn.tensor_contract(tens, tens_c)

            out_inds = [ind + '_' for ind in open_inds]
            out.transpose(*out_inds, *open_inds, inplace=True)
            return out

        if orthog_direction > 0:

            while canon_site < gtn.L - 1:

                gtn.canonize(inplace=True, i=canon_site + 1, cur_orthog=canon_site, scale=False)
                gtn_tensL = gtn.data[canon_site]

                indL = gtn.data.bond(canon_site - 1, canon_site) if canon_site > 0 else None
                indR = gtn.data.bond(canon_site, canon_site + 1)
                indP = gtn.data.site_ind_id.format(canon_site)
                indL_size = gtn_tensL.ind_size(indL) if indL is not None else None
                indP_size = gtn_tensL.ind_size(indP)
                open_inds = (indP, indL) if indL is not None else (indP,)
                open_sizes = (indP_size, indL_size) if indL is not None else (indP_size,)
                tot_size = int(np.prod(open_sizes))
                projector = _get_site_rho(gtn.data[canon_site], open_inds)
                iden = np.eye(tot_size).reshape(*open_sizes, *open_sizes)
                projector.modify(apply=lambda x: iden - x)

                ## subspace components
                rho = None
                for vec in subspace_vecs:
                    indL = vec.data.bond(canon_site, canon_site - 1) if canon_site > 0 else None
                    indP = vec.data.site_ind_id, format(canon_site)
                    open_inds = (indP, indL) if indL is not None else (indP,)
                    rho_ = _get_site_rho(vec.data[canon_site], open_inds)
                    if rho is None:
                        rho = rho_
                    else:
                        rho.modify(apply=lambda x: x + rho_.data)

                projector_c = projector.conj()
                rho_data = rho.data
                num_axes = 2 if indL is not None else 1
                rho_data = np.tensordot(projector_c.data, rho_data, axes=num_axes)
                rho_data = np.tensordot(rho_data, projector.data, axes=num_axes)

                evals, evecs = np.linalg.eigh(rho_data.reshape(tot_size, -1))
                sort_inds = np.argsort(evals)[::-1]
                evals = evals[sort_inds]
                evecs = evecs[:, sort_inds]

                evals_sum = np.cumsum(evals / np.sum(evals))
                num_keep = np.argmax((1.0 - evals_sum) <= cutoff) + 1
                if max_bond is not None:
                    num_keep = min(num_keep, max_bond)

                if indL is not None:
                    A_ = evecs[:, :num_keep].reshape(indP_size, indL_size, -1)
                    A_tens_ = qtn.Tensor(A_, inds=(indP, indL, indR))
                else:
                    A_ = evecs[:, :num_keep]
                    A_tens_ = qtn.Tensor(A_, inds=(indP, indR))

                ## expand basis
                qtn.tensor_direct_product(gtn_tensL, A_tens_, sum_inds=(indP,), inplace=True)

                ## update canon site
                reindex_dict = {indR: indR + '_'}
                A_tens_vec = gtn_tensL.reindex(reindex_dict, inplace=False)

                new_vec_tens = qtn.tensor_contract(gtn.data[canon_site], gtn.data[canon_site + 1], A_tens_vec)
                new_vec_tens = new_vec_tens.reindex({indR + '_': indR}, inplace=True)
                new_vec_tens.transpose_like(gtn.data[canon_site + 1], inplace=True)
                gtn.data[canon_site + 1].modify(data=new_vec_tens.data)

                ## update vec canon site
                for vec in subspace_vecs:
                    v_indP = vec.data.site_ind_id.format(canon_site)
                    v_indR = vec.data.bond(canon_site + 1, canon_site)
                    v_indL = vec.data.bond(canon_site - 1, canon_site) if canon_site > 0 else None
                    reindex_dict = {indR: v_indR + '_', indP: v_indP}
                    if indL is not None:
                        reindex_dict[indL] = v_indL
                    A_tens_vec = gtn_tensL.reindex(reindex_dict, inplace=False)

                    new_vec_tens = qtn.tensor_contract(vec[canon_site], vec[canon_site + 1], A_tens_vec)
                    new_vec_tens = new_vec_tens.reindex({v_indR + '_': v_indR}, inplace=True)
                    new_vec_tens.transpose_like(vec[canon_site + 1], inplace=True)
                    vec[canon_site + 1].modify(data=new_vec_tens.data)

                canon_site += 1

            ## pad last site with zeros
            gtn_tensL = gtn.data[canon_site]
            expanded_size = gtn.data[canon_site - 1].ind_size(indR)
            gtn_tensL.expand_ind(indR, expanded_size)

        else:

            while canon_site > 0:

                gtn.canonize(inplace=True, i=canon_site - 1, cur_orthog=canon_site, scale=False)
                gtn_tensL = gtn.data[canon_site]
                old_gtn_tensL = gtn_tensL.copy()

                indR = gtn.data.bond(canon_site + 1, canon_site) if canon_site < gtn.L - 1 else None
                indL = gtn.data.bond(canon_site, canon_site - 1)
                indP = gtn.data.site_ind_id.format(canon_site)
                open_inds = (indP, indR) if indR is not None else (indP,)
                indP_size = gtn_tensL.ind_size(indP)
                indR_size = gtn_tensL.ind_size(indR) if indR is not None else None
                open_sizes = (indP_size, indR_size) if indR is not None else (indP_size,)
                tot_size = int(np.prod(open_sizes))
                projector = _get_site_rho(gtn.data[canon_site], open_inds)
                iden = np.eye(tot_size).reshape(*open_sizes, *open_sizes)
                projector.modify(apply=lambda x: iden - x)

                ## subspace components
                rho = None
                for vec in subspace_vecs:
                    # indL = vec.data.bond(canon_site, canon_site - 1)
                    v_indP = vec.data.site_ind_id.format(canon_site)
                    v_indR = vec.data.bond(canon_site + 1, canon_site) if canon_site < gtn.L - 1 else None
                    open_inds = (v_indP, v_indR) if v_indR is not None else (v_indP,)
                    rho_ = _get_site_rho(vec.data[canon_site], open_inds)
                    if rho is None:
                        rho = rho_
                    else:
                        rho.modify(apply=lambda x: x + rho_.data)

                projector_c = projector.conj()
                rho_data = rho.data
                # print('projector', projector.data.shape)
                # print('rho', rho.data.shape)
                num_axes = 2 if indR is not None else 1
                rho_data = np.tensordot(projector_c.data, rho_data, axes=num_axes)
                rho_data = np.tensordot(rho_data, projector.data, axes=num_axes)

                evals, evecs = np.linalg.eigh(rho_data.reshape(tot_size, -1))
                sort_inds = np.argsort(np.abs(evals))[::-1]
                evals = evals[sort_inds]
                evecs = evecs[:, sort_inds]

                # print('evals', evals, np.linalg.norm(rho_data))
                evals_sum = np.cumsum(evals / np.sum(evals))
                num_keep = np.argmax((1.0 - evals_sum) <= cutoff) + 1

                if max_bond is not None:
                    num_keep = min(num_keep, max_bond)

                if indR is not None:
                    A_ = evecs[:, :num_keep].reshape(indP_size, indR_size, -1)
                    A_tens_ = qtn.Tensor(A_, inds=(indP, indR, indL))
                else:
                    A_ = evecs[:, :num_keep]
                    A_tens_ = qtn.Tensor(A_[:, :], inds=(indP, indL))

                ## expand basis
                # print('gtn tensL old', gtn_tensL)
                qtn.tensor_direct_product(gtn_tensL, A_tens_, sum_inds=(indP,), inplace=True)
                # print('gtn tensL old', gtn_tensL)

                ## update canon site
                reindex_dict = {indL: indL + '_'}
                A_tens_vec = gtn_tensL.reindex(reindex_dict, inplace=False)

                new_vec_tens = qtn.tensor_contract(old_gtn_tensL, gtn.data[canon_site - 1], A_tens_vec)
                new_vec_tens.reindex({indL + '_': indL}, inplace=True)
                new_vec_tens.transpose_like(gtn.data[canon_site - 1], inplace=True)
                gtn.data[canon_site - 1].modify(data=new_vec_tens.data)

                for vec in subspace_vecs:
                    v_indP = vec.data.site_ind_id.format(canon_site)
                    v_indR = vec.data.bond(canon_site + 1, canon_site) if canon_site < gtn.L - 1 else None
                    v_indL = vec.data.bond(canon_site - 1, canon_site)
                    reindex_dict = {indL: v_indL + '_', indP: v_indP}
                    if indR is not None:
                        reindex_dict[indR] = v_indR
                    A_tens_vec = gtn_tensL.reindex(reindex_dict, inplace=False)

                    new_vec_tens = qtn.tensor_contract(vec[canon_site], vec[canon_site - 1], A_tens_vec)
                    new_vec_tens.reindex({indL + '_': indL}, inplace=True)
                    new_vec_tens.transpose_like(vec[canon_site - 1], inplace=True)
                    vec[canon_site - 1].modify(data=new_vec_tens.data)

                canon_site -= 1

            ## pad last site with zeros
            gtn_tensL = gtn.data[canon_site]
            expanded_size = gtn.data[canon_site - 1].ind_size(indL)
            gtn_tensL.expand_ind(indL, expanded_size)

        return gtn

    # #####################
    # ## data processing ##
    # #####################
    #
    # def get_mpo_inverse(self,mpo,boundary_conditions=None,use_dmrg=False):
    #     """ mpo:  operator to take the inverse of
    #         boundary condition:  at the moment, an m x q**K**L array such that
    #                              for each m, vector.T * data = mth boundary condition
    #         expensive method right now -- take inverse directly
    #         optimization method:
    #     """
    #
    #     return self.grid.get_mpo_inverse(mpo, boundary_conditions, use_dmrg)

    # ################
    # ## processing ##
    # ################

    def interpolate_data(self, depth: int, interp_dict: dict['Axis': 'qtn.MatrixProductOperator'],
                         new_grid: 'Grid1D' = None):
        """Refine the grid by interpolating the data onto a finer grid.

        Applies per-axis interpolation MPOs, decomposes the resulting fine
        tensors, and inserts them to build a state on a grid with ``depth``
        extra points per axis. Only outer-product interpolation schemes are
        supported.

        Parameters
        ----------
        depth : int
            Number of fine points introduced per coarse interval.
        interp_dict : dict[Axis, qtn.MatrixProductOperator]
            Per-axis interpolation operator; its tensors carry all coarse-grid
            indices with fine indices tagged "fine" (numbering may be off).
        new_grid : Grid1D, optional
            Target fine grid; constructed automatically if None.

        Returns
        -------
        GridTN1D
            The interpolated state on the fine grid.
        """
        grid = self.grid
        mps = self.data.copy()
        if mps is None:
            return self.copy()

        site_ind_id = mps.site_ind_id
        site_tag_id = mps.site_tag_id
        tmp_ind_id = tmp_tag_id = 'TMP{}'

        if new_grid is None:
            new_axes = [Axis(ax.L + depth, dx=ax.dx / depth, x0=ax.xpts[0] - depth * ax.dx / 2,
                             ax_map=ax.map, coordinate=ax.coordinate, is_flipped=ax.is_flipped, basis=ax.basis)
                        for ax in grid.axes]
            new_grid = grid.create_like(new_axes)
        else:
            new_axes = new_grid.axes


        ## apply operator, decompose and insert fine tensors
        for k, interp_mpo in interp_dict.items():
            interp_mpo.upper_ind_id = site_ind_id
            interp_mpo.lower_ind_id = tmp_ind_id

            mps.exponent += interp_mpo.exponent

        idx_axs = {}
        coarse_mps_idxs = []
        ## renumber all of the coarse tensors
        for i in range(grid.ndim):
            c_ax = grid.axes[i]
            f_ax = new_axes[i]

            coarse_inds = grid.get_inds_in_axis(c_ax)
            fine_inds = new_grid.get_inds_in_axis(f_ax)
            fine_inds_ = fine_inds[-len(coarse_inds):] if c_ax.map.is_flipped() \
                            else fine_inds[:len(coarse_inds)]

            idx_axs[tuple(fine_inds_)] = (c_ax, f_ax)
            coarse_mps_idxs += fine_inds_

            for old_idx, new_idx in zip(coarse_inds, fine_inds_):
                tens = mps[old_idx]
                tens.reindex({site_ind_id.format(old_idx): tmp_ind_id.format(new_idx)}, inplace=True)
                tens.retag({site_tag_id.format(old_idx): tmp_tag_id.format(new_idx)}, inplace=True)

            interp_mpo = interp_dict[c_ax]
            interp_mpo = c_ax.map.transform_mpo(interp_mpo) ## not an inplace operation
            helper.renumber_mpo(interp_mpo, list(range(interp_mpo.L)), fine_inds_, inplace=True)
            interp_mpo.mangle_inner_()
            interp_dict[c_ax] = interp_mpo

        coarse_mps_idxs = np.sort(coarse_mps_idxs)
        mps._L = new_grid.L
        mps._site_tag_id = tmp_tag_id
        mps._site_ind_id = tmp_ind_id

        for ix, i in enumerate(coarse_mps_idxs):
            c_ax, f_ax, fine_idxs_ = None, None, None
            for idxs in idx_axs:
                if i in idxs:
                    c_ax, f_ax = idx_axs[idxs]
                    fine_idxs_ = idxs
                    break

            interp_mpo: qtn.MatrixProductOperator = interp_dict[c_ax]

            fine_idxs = new_grid.get_inds_in_axis(f_ax)
            new_idxs = np.sort( list(set(fine_idxs).difference(set(fine_idxs_))) )

            idx1 = i
            idx2 = i + 1 if i < new_grid.L - 1 else None

            interp_mpo_tens = interp_mpo.select_tensors(interp_mpo.site_tag_id.format(i))[0]
            new_tens = qtn.tensor_contract(mps[idx1], interp_mpo_tens)

            prev_tens = None
            i_ = i
            while prev_tens is None and i_ > 0:
                i_ -= 1
                try:
                    prev_tens = mps[i_]
                except KeyError:
                    pass

            l_bonds = [*new_tens.bonds(prev_tens)] if prev_tens is not None else []

            if 'fine' in new_tens.inds:
                if c_ax.map.is_flipped():
                    new_tens.reindex({site_ind_id.format(idx1): site_ind_id.format(depth)}, inplace=True)
                    new_tens.unfuse({'fine': [site_ind_id.format(i) for i in range(depth - 1, -1, -1)]},
                                    {'fine': (2,) * depth}, inplace=True)
                    new_tens.drop_tags()
                else:
                    new_tens.reindex({site_ind_id.format(idx1): site_ind_id.format(0)}, inplace=True)
                    new_tens.unfuse({'fine': [site_ind_id.format(i) for i in range(1, depth + 1)]},
                                    {'fine': (2,) * depth}, inplace=True)
                    new_tens.drop_tags()

                fine_mps = helper.mpx_from_dense(new_tens, depth + 1, [site_ind_id], left_anc=l_bonds,
                                                 site_tag_id=tmp_tag_id)
                helper.renumber_mps(fine_mps, list(range(depth + 1)), np.sort([idx1, *new_idxs]), inplace=True)
                fine_mps.mangle_inner_()

                ## QR decomposition to push MPO bond to next tensor if dangling MPO bond
                fine_mps_tens = fine_mps.select_tensors(tmp_tag_id.format(idx1))[0]
                if fine_mps_tens.ndim > 3:
                    prev_fine_tens = fine_mps.select_tensors(tmp_tag_id.format(idx1 + depth - 2))[0]
                    lix = [*fine_mps_tens.bonds(prev_fine_tens)] if prev_fine_tens is not None else []
                    lix = lix + [site_ind_id.format(idx1)]
                    tensL, tensR = fine_mps_tens.split(lix, absorb='right')
                    fine_mps_tens.modify(data=tensL.data, inds=tensL.inds)

                    next_tens = tensR @ mps[idx2]
                    mps[idx2].modify(data=next_tens.data, inds=next_tens.inds)

                mps.delete([tmp_tag_id.format(idx1)], which='all')
                mps.add_tensor_network(fine_mps)  ## check exponent?

            else:
                ## like the zip-up algorithm (but not making guarantees on canonicalization
                lix = l_bonds + [site_ind_id.format(idx1)]
                if i + 1 < new_grid.L:
                    tensL, tensR = new_tens.split(lix, absorb='right')
                    # tensL.transpose_like(mps[idx1], inplace=True)
                    mps[idx1].modify(data=tensL.data, inds=tensL.inds)

                    tensR = qtn.tensor_contract(tensR, mps[idx2])
                    mps[idx2].modify(data=tensR.data, inds=tensR.inds)
                else:
                    mps[idx1].modify(data=new_tens.data, inds=new_tens.inds)

        mps._L = mps.num_tensors  ## should be equal to new_grid.L
        for i in range(new_grid.L):
            mps[i].reindex({tmp_ind_id.format(i): site_ind_id.format(i)}, inplace=True)
            mps[i].retag({tmp_ind_id.format(i): site_tag_id.format(i)}, inplace=True)

        mps._site_tag_id = site_tag_id
        mps._site_ind_id = site_ind_id

        new_gtn = GridTN1D(new_grid, mps)
        return new_gtn

    # def gaussian_smooth_data(self, inplace=False, sigma=0.3, proc_axes=None, mode='wrap', **compress_opts):
    #     """ process the data in some way
    #     """
    #     gtn = self if inplace else self.copy(deep=False)
    #     blur_mpo = self.grid.gaussian_smooth_mpo(sigma, proc_axes, mode)
    #
    #     if blur_mpo is not None:
    #         gtn.apply(blur_mpo, compress=False)  ## not a deep inplace operation
    #         gtn.compress(inplace=True, **compress_opts)
    #         # scalar_mps_field = helper.apply(blur_mpo, self.data, compress=compress,
    #         #                                 **self.grid.get_compress_opts(compress))
    #     # else:
    #         # scalar_mps_field = self.data if inplace else self.data.copy()
    #
    #     return gtn
    #
    #
    # def apply_absorbing_bc(self, x_ax, v_ax, left_bc, right_bc, inplace=False, compress=True, **compress_opts):
    #     """ apply absorbing bc to MPS state, assumes TN dimensions are x1,x2,x3,v1,v2,v3
    #         (prev: apply absorbing bc to derivative MPO, assumes TN dimensions are x1,x2,x3,v1,v2,v3)
    #     """
    #     gtn = self if inplace else self.copy(deep=False)
    #     absorb_bc_mpo = self.grid.absorbing_bc_mpo(x_ax, v_ax, left_bc, right_bc)
    #
    #     if absorb_bc_mpo is not None:
    #         gtn.apply(absorb_bc_mpo, inplace=True, compress=compress, **compress_opts)
    #     return gtn
    #
    #
    # def apply_reflecting_v_bc(self, x_ax, v_ax, left_bc, right_bc, inplace=False, compress=True, **compress_opts):
    #     """ apply reflecting_v bc to state.
    #         apply to derivative MPO?
    #     """
    #     gtn = self if inplace else self.copy()
    #     reflect_bc_mpo = self.grid.reflecting_v_bc_mpo(x_ax, v_ax, left_bc, right_bc)
    #
    #     if reflect_bc_mpo is not None:
    #         gtn.apply(reflect_bc_mpo, inplace=True, compress=compress, **compress_opts)
    #     return gtn

    ###############################
    ## application of operations ##
    ###############################

    def mps_to_diag_mpo(self, lower_ind_id='i({})', upper_ind_id='o({})', inplace=False) -> 'GridTN':
        """Convert an MPS into a diagonal MPO using COPY tensors.

        Each site tensor is contracted with a 3-leg COPY tensor so the MPS
        values appear on the diagonal of the resulting MPO.

        Parameters
        ----------
        lower_ind_id : str
            Format string for the MPO lower (input) physical indices.
        upper_ind_id : str
            Format string for the MPO upper (output) physical indices.
        inplace : bool
            Mutate self in place if True, else operate on a copy.

        Returns
        -------
        GridTN
            The diagonal MPO.
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        assert (self.data_type == DataType.MPS), f'gtn needs to MPS type not {self.data_type}'

        gtn = self if inplace else self.copy(deep=False)

        mps: qtn.MatrixProductState = self.data if inplace else self.data.copy()
        if lower_ind_id == mps.site_ind_id or upper_ind_id == mps.site_ind_id:
            mps.site_ind_id = lower_ind_id + '_tmp_'
        site_ind_id = mps.site_ind_id
        site_tag_id = mps.site_tag_id

        # # L, q = self.L
        # out = qtn.TensorNetwork([])
        # for ax in self.grid.axes:
        #     idxs = self.grid.get_inds_in_axis(ax)
        #     q = ax.q
        #     for i in idxs:
        #         tens = mps[i]
        #         d_ijk = qtn.tensor_core.COPY_tensor(q, (site_ind_id.format(i), lower_ind_id.format(i),
        #                                                 upper_ind_id.format(i)),
        #                                             tags=(site_tag_id.format(i),))
        #         out.add_tensor( tens.contract(d_ijk) )
        # # out.view_as(qtn.TensorNetwork1D, L=L, site_tag_id=site_tag_id, inplace=True)
        # out = out.view_as(qtn.MatrixProductOperator, inplace=True, L=self.L, cyclic=False, site_tag_id=site_tag_id,
        #                                              upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)

        out = qtn.TensorNetwork([])
        for i in range(self.grid.L):
            q = self.grid.shape()[i]
            tens = mps[i]
            d_ijk = qtn.tensor_core.COPY_tensor(q, (site_ind_id.format(i), lower_ind_id.format(i),
                                                    upper_ind_id.format(i)),
                                                tags=(site_tag_id.format(i),))
            out.add_tensor(tens.contract(d_ijk))

        # out.view_as(qtn.TensorNetwork1D, L=L, site_tag_id=site_tag_id, inplace=True)
        out = out.view_as(qtn.MatrixProductOperator, inplace=True, L=self.L, cyclic=False, site_tag_id=site_tag_id,
                          upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
        out.exponent = self.data.exponent

        gtn.data = out
        return gtn

    def apply_elemental_multiply_op(self, lower_ind_id='i({})', upper_ind_id='o({})', axes=None, add_cc=False,
                                    take_mps_cc=False, compress=False, compress_opts=None):
        """Turn this MPS into an MPO implementing element-wise multiplication.

        Contracts self with the grid's 3-leg elemental-multiply tensor network
        so the resulting MPO multiplies any other state element-wise by self.
        Single-tensor spatial data are handled as a sparse diagonal; real
        Fourier bases trigger the conjugate-augmented (add_cc) path. Assumes
        self's site indices run 0..grid.L-1 with no gaps.

        Parameters
        ----------
        lower_ind_id : str
            Format string for the output MPO lower (input) indices.
        upper_ind_id : str
            Format string for the output MPO upper (output) indices.
        axes : Sequence[Axis], optional
            Restrict the multiply to these axes.
        add_cc : bool
            Add the complex-conjugate branch (forced on for real-k bases).
        take_mps_cc : bool
            Return the operator transposed and conjugated.
        compress : bool
            Compress the resulting MPO.
        compress_opts : dict, optional
            Overrides for compression parameters.

        Returns
        -------
        GridTN1D
            The element-wise multiplication MPO.
        """
        if self.data is None:  # zero
            return self.create_like(new_data=None)

        if self.data_type == DataType.Num:
            return self.create_like(new_data=self.data)

        assert self.data_type == DataType.MPS, f'needs to be MPS to make MPO, not {self.data_type}'
        new_mpo = qtn.TensorNetwork([])
        contract_ind = self.data.site_ind_id

        if self.data.L == 1 and \
            all([isinstance(ax.basis, SpatialBasis) for ax in self.grid.axes]):

            ## data represented as a single tensor
            tens = self.data[0]
            q = self.grid.npts
            rows = cols = list(range(q))
            diag_csr_data = scipy.sparse.csr_array((tens.data, (rows,cols)), shape=(q,q))

            # from quimb import qu
            # tmp = qu(diag_coo_data, qtype='dop', sparse=True)

            diag_tens = qtn.Tensor(data=diag_csr_data, inds=(upper_ind_id.format(0), lower_ind_id.format(0)),
                                   tags=(self.data.site_tag_id.format(0),))
            new_mpo.add_tensor(diag_tens)

            new_mpo.exponent = self.data.exponent

        else:

            add_cc = self.grid.has_basis_k_real() or add_cc

            fix_tag = None
            if contract_ind == upper_ind_id:
                upper_ind_id += '_tmp'
                fix_tag = 'U'
            elif contract_ind == lower_ind_id:
                lower_ind_id += '_tmp'
                fix_tag = 'L'

            elem_mult_tn = self.grid.get_elemental_multiply_tn(in1_ind_id=lower_ind_id, in2_ind_id=contract_ind,
                                                               out_ind_id=upper_ind_id, site_tag_id='_tmp{}',
                                                               x_axes=axes)
            self_data = self.data
            elem_mult_tn_data = elem_mult_tn.data

            if add_cc:
                # conj_gtn = self.conj(inplace=False)
                # shift_inds = [1 if ax.is_real_k() else 0 for ax in self.grid.axes]
                # shift_mpo_gtn = self.grid.get_shift_mpo(shift_inds)
                # conj_gtn = conj_gtn.apply(shift_mpo_gtn)

                self_data = RealFourierBasis.convert_mps_to_full(self.grid, self_data, anc_ind=2)
                ### TODO:  need to deal with ancillas from comb branches (summing branches but not spine) move to Comb?
                # self_data = self_data.copy()
                #
                # for ax in self.grid.axes:
                #     if not ax.is_real_k():  continue
                #
                #     if ax.map.is_flipped():
                #         self_data[self.L - 1].new_ind(f'anc_elem_mult_1_{ax}')
                #     else:
                #         self_data[0].new_ind(f'anc_elem_mult_1_{ax}')
                #
                # conj_data = self_data.conj()  ## should be ok to take complex conj. of entire state
                # self_data = helper.add_MPS(self_data, conj_data, inplace=True, compress=False)

            for x in range(self.grid.L):
                tens_x1 = self_data[x]
                tens_x2 = elem_mult_tn_data[x]
                new_tens = tens_x1.contract(tens_x2)  ## includes ancilla inds
                new_tens.drop_tags(tags=tens_x2.tags)
                if fix_tag == 'L':
                    new_tens.reindex({lower_ind_id.format(x): lower_ind_id[:-4].format(x)}, inplace=True)
                elif fix_tag == 'U':
                    new_tens.reindex({upper_ind_id.format(x): upper_ind_id[:-4].format(x)}, inplace=True)

                new_mpo.add(new_tens)

            new_mpo.fuse_multibonds(inplace=True)

            if fix_tag == 'U':
                upper_ind_id = upper_ind_id[:-4]
            elif fix_tag == 'L':
                lower_ind_id = lower_ind_id[:-4]

            new_mpo.exponent = self.data.exponent + elem_mult_tn.data.exponent


        new_mpo = new_mpo.view_as(qtn.MatrixProductOperator, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                  site_tag_id=self.data.site_tag_id, inplace=True, cyclic=False, L=self.data.L)

        gmpo = self.create_like(new_data=new_mpo)
        gmpo.is_constant = self.is_constant
        gmpo.constant_axes = self.constant_axes
        # gmpo.data = new_mpo

        ## if wanted into mps to be it's complex conjugate
        if take_mps_cc:
            gmpo = gmpo.transpose(inplace=True)  ## force is real so it's even
            gmpo.conj(inplace=True)

        if compress:
            gmpo.compress(compress_opts=compress_opts)

        return gmpo

    # def apply_xmultiply_mpo(self, x_axes=None, inplace=False, compress=True, **compress_opts):
    #     """ perform x*f(x) operation, where x_dims specifies which axis/axes to multiply
    #         eg. 1D:  x*f(x,y,...)
    #             2D:  x*y*f(x,y,...) or x*z*f(x,y,z...)
    #     """
    #     gtn = self if inplace else self.copy()
    #     x_mpo = self.grid.get_xmultiply_mpo(x_axes)
    #     gtn.apply(x_mpo, compress=compress, **compress_opts)
    #     return gtn
    #
    #
    # def take_firstderivative(self, ax, bc, deriv_params=None, recalc=False, inplace=False, compress=True,
    #                          **compress_opts):
    #     """ df/dx_i
    #         bc = boundary condition
    #     """
    #     gtn = self if inplace else self.copy()
    #     if deriv_params is None:     deriv_params = {}
    #     ddx_mpo = self.grid.get_firstderivative_mpo(ax, bc, **deriv_params)
    #     gtn.apply(ddx_mpo, inplace=True, compres=compress, **compress_opts)
    #     return gtn
    #
    #
    # def take_secondderivative(self, ax1, ax2, bc1, bc2, inplace=False, recalc=False, compress=True, **compress_opts):
    #     """ d^2f/dx^2 {x} + d^2f/dy^2 {y} + d^2f/dz^2 {z}
    #     """
    #     gtn = self if inplace else self.copy()
    #     d2_mpo = self.grid.get_secondderivative_mpo(ax1, ax2, bc1, bc2)
    #     gtn.apply(d2_mpo, inplace=True, compress=compress, **compress_opts)
    #     return gtn

    ###############
    ## integration/partial integration ##
    ################

    def apply_mps(self, mps, inplace=False, compress=True, compress_opts=None):
        """Apply an MPS to self: MPO times MPS gives an MPS, MPS times MPS gives a scalar.

        If self is an MPO, it is transposed and applied to ``mps`` to yield a
        new MPS; if self is an MPS, the overlap with ``mps`` is returned.

        Parameters
        ----------
        mps : GridTN1D or qtn.MatrixProductState
            The MPS operand.
        inplace : bool
            Mutate self in place if True (MPO case only).
        compress : bool
            Compress the resulting product (MPO case).
        compress_opts : dict, optional
            Overrides for compression parameters.

        Returns
        -------
        GridTN1D or Numeric
            The resulting MPS (MPO case) or the scalar overlap (MPS case).
        """
        if isinstance(mps, GridTN1D):
            mps = mps.data

        assert isinstance(mps, qtn.MatrixProductState), 'supplied TN must be MPS'

        if isinstance(self.data, qtn.MatrixProductOperator):
            gtn = self if inplace else self.copy()
            helper.mpo_transpose(gtn.data, inplace=True)
            out = helper.apply(gtn.data, mps, compress=compress, compress_opts=compress_opts)
            gtn.data = out
            return gtn
        elif isinstance(self.data, qtn.MatrixProductState):
            out = helper.ovlp(self.data, mps)
            return out
        else:
            raise NotImplementedError

    # @profile
    def meas_expec(self, obs_gtn: 'GridTN1D', integ_axes=None, is_sqrt=False, ancilla_reindex=None,
                   new_grid: 'Grid1D' = None, exclude_axes=None, exclude_weights=False,
                   new_ax_deriv_configs=None, compress=False, compress_opts=None,
                   zipup=False, zipup_direction=-1, **kwargs):
        """Measure an observable, optionally integrating over a set of axes.

        Computes integral dx f O f (or f O f when ``is_sqrt``) over
        ``integ_axes``. Integrating all axes returns a scalar (or Tensor with
        ancilla bonds); integrating none applies the observable; a partial set
        returns a reduced-dimensionality GridTN1D on the remaining axes.

        Parameters
        ----------
        obs_gtn : GridTN1D
            Observable; an MPS observable is first turned into an element-wise
            multiply MPO. None measures the bare (square) norm.
        integ_axes : Sequence[Axis], optional
            Axes to integrate over; defaults to all axes.
        is_sqrt : bool
            Measure the quadratic form f O f rather than the linear f O.
        ancilla_reindex : dict, optional
            Mapping old_anc -> new_anc for the anchor tensor's ancilla when
            ``is_sqrt``; old and new ancilla are fused under the old name.
        new_grid : Grid1D, optional
            Output grid for the reduced result.
        exclude_axes : Sequence[Axis], optional
            Axes to exclude; not supported (raises if non-empty).
        exclude_weights : bool
            Use unit integration weights instead of axis quadrature weights.
        new_ax_deriv_configs : dict, optional
            Derivative configs for the output grid.
        compress : bool
            Compress the reduced result.
        compress_opts : dict, optional
            Overrides for compression parameters.
        zipup : bool
            Use zip-up application in the no-integration path.
        zipup_direction : int
            Direction (and resulting form) for the zip-up application.
        **kwargs
            Additional options (ignored).

        Returns
        -------
        Numeric, qtn.Tensor, or GridTN1D
            A scalar/Tensor when integrating all axes, else a reduced GridTN1D.
        """
        if exclude_axes is not None and len(exclude_axes) != 0:
            raise NotImplementedError('meas expec with exclude axes not implemented for GridTN1D')

        if integ_axes is None:
            integ_axes = self.grid.axes
            out_axes = []
        else:
            out_axes = [ax for ax in self.grid.axes if ax not in integ_axes]

        if new_grid is None and len(out_axes) > 0:
            new_grid = self.grid.create_like(new_axes=out_axes)

        if new_ax_deriv_configs is None and new_grid is not None:
            new_ax_deriv_configs = {ax: self.ax_deriv_configs[ax] for ax in new_grid.axes}

        if self.data is None:
            if len(out_axes) > 0:
                return new_grid.make_empty_gridTN(ax_deriv_configs=new_ax_deriv_configs)
            else:
                return np.nan

        if obs_gtn is not None and obs_gtn.data_type == DataType.MPS:
            obs_gtn = obs_gtn.apply_elemental_multiply_op()
            # obs_gtn = obs_gtn.mps_to_diag_mpo()

        assert self.data_type == DataType.MPS, f'integrate only defined for MPS GridTNs, not {self.data_type}'

        ## no integration
        if len(integ_axes) == 0:
            # print('1D meas expec no integration')

            if zipup:
                # print('compress', compress, compress_opts)
                form = 'right' if zipup_direction < 0 else 'left'
                if compress_opts is None:
                    compress_opts = {'form': form}
                else:
                    compress_opts = compress_opts.copy()
                    compress_opts['form'] = form

            if is_sqrt:
                gtn_cc = self.copy()  # conj()
                tens0 = gtn_cc.get_anchor_tens()  ## anchor tensor that will have ancilla
                if ancilla_reindex is not None:
                    tens0.reindex({anc: anc_new for anc, anc_new in ancilla_reindex.items()}, inplace=True)
                tmp = gtn_cc.apply_elemental_multiply_op(take_mps_cc=True)

                if obs_gtn is not None:
                    out = self.apply(obs_gtn, compress=False)
                else:
                    out = self

                out = out.apply(tmp, zipup=zipup, compress=compress, compress_opts=compress_opts)
                # print('zipup?', zipup, helper.check_right_orthog(out))
                return out
            else:
                if obs_gtn is not None:
                    return self.apply(obs_gtn, zipup=zipup, compress=compress, compress_opts=compress_opts)
                else:
                    return self.copy()

        ## integrate all
        elif len(out_axes) == 0:

            if is_sqrt:
                # ## actually correct (don't need convolution bc of cc for k-space)
                # self_conj = self.conj(mangle_inner=True)
                # # dx = np.asscalar( np.prod([ax.get_integral_weight() for ax in integ_axes]) )
                # # helper.scalar_multiply(data_conj, dx, inplace=True)
                # integ_weights = {ax: ax.get_integral_weight() for ax in integ_axes}
                # integ_weights = self.grid.make_mps_ndim(integ_weights)
                # if np.isscalar(integ_weights):
                #     self_conj.scalar_multiply(integ_weights, inplace=True)
                # else:
                #     integ_weights = integ_weights.mps_to_diag_mpo()
                #     self_conj.apply(integ_weights, inplace=True)
                #
                # data_conj = self_conj.data
                # if ancilla_reindex:  # for compatibility with comb
                #     data_conj.reindex(ancilla_reindex, inplace=True)
                # # print('self sqrt integ exp', self.data.exponent, data_conj.exponent)
                # result = helper.ovlp(self.data, data_conj)

                gtn_cc = self.conj()
                # print('gtn cc', gtn_cc.max_bond())
                tens0 = gtn_cc.get_anchor_tens()  ## anchor tensor that will have ancilla
                if ancilla_reindex is not None:
                    tens0.reindex({anc: anc_new for anc, anc_new in ancilla_reindex.items()}, inplace=True)

                if exclude_weights:
                    integ_weights = self.grid.get_ones_mps()
                else:
                    integ_weights = {ax: ax.get_integral_weight() for ax in integ_axes}
                    integ_weights = self.grid.make_mps_ndim(integ_weights)
                    # print('integ weights', type(integ_weights), integ_axes, integ_weights)
                    if isinstance(integ_weights, GridTN1D):
                        integ_weights.compress()
                        # print('integ weights', integ_axes, integ_weights)
                #
                # if len(gtn_cc.grid.axes) == 1:
                #     plt.figure()
                #     plt.plot(np.real(gtn_cc.get_data()))
                #     plt.plot(np.imag(gtn_cc.get_data()))
                #     plt.plot(np.real(self.get_data()), '--')
                #     plt.plot(np.imag(self.get_data()), ':')
                #     plt.title('gtn cc integ')
                #     plt.show()

                if np.isscalar(integ_weights):
                    gtn_cc.scalar_multiply(integ_weights, inplace=True)
                else:
                    # integ_weights = integ_weights.mps_to_diag_mpo()
                    # ## doesn't work for k-space bc get_mps_ndim uses [0,0,...,1,0,0,0] instead of 1 MPS
                    integ_weights = integ_weights.apply_elemental_multiply_op()
                    # print('integ weights', integ_weights)
                    gtn_cc.apply(integ_weights, inplace=True, zipup=True, compress=True,
                                 compress_opts={'max_bond': gtn_cc.max_bond()}
                                 )
                                 # compress_opts={'max_bond':64}) # gtn_cc.max_bond()})
                    # print('gtn cc', gtn_cc)
                    # print('self.data', self.data)

                if obs_gtn is not None:
                    result = helper.expectation_value(self.data, obs_gtn.data, bra=gtn_cc.data)
                else:
                    result = helper.ovlp(self.data, gtn_cc.data)
            else:

                integ = self.grid.get_integrals_mpx(axes=None, is_sqrt=False)

                if obs_gtn is not None:
                    result = helper.expectation_value(self.data, obs_gtn.data, bra=integ.data)
                else:
                    result = helper.ovlp(self.data, integ.data)

            return result  ## a scalar or Tensor (if there are ancilliary bonds)

        else:
            if exclude_weights:
                integ_dict = {iax: iax.get_ones_mps() for iax in integ_axes}
                integ_gtn = self.grid.make_gtn_from_dicts([integ_dict], data_type=DataType.MPX)
                # integ_gtn = self.grid.get_integrals_mpx(integ_axes, is_sqrt=False)  # .data
            else:
                integ_gtn = self.grid.get_integrals_mpx(integ_axes, is_sqrt=False)  # .data
            remaining_axes = [ax for ax in self.grid.axes if ax not in integ_axes]
            remaining_inds = []
            for ax in remaining_axes:
                remaining_inds += self.grid.get_inds_in_axis(ax)

            if new_grid is None:
                new_grid = self.grid.create_like(remaining_axes, self.grid.gridID + '_integ')

            if is_sqrt:
                # # tmp = helper.mps_to_diag_mpo(self.data.conj())
                # self_conj = self.copy()
                # self_conj.data.conj(inplace=True, mangle_inner=True)
                # if ancilla_reindex:
                #     self_conj.get_anchor_tens().reindex(ancilla_reindex, inplace=True)
                # tmp = self_conj.apply_elemental_multiply_op()
                # integ_mps = helper.apply(integ_mps, tmp.data)
                # if ancilla_reindex:
                #     fuse_anc = {anc_old: (anc_new, anc_old) for anc_new, anc_old in ancilla_reindex.items()}
                #     integ_mps[self.get_anchor_ind()].fuse(fuse_anc, inplace=True)

                gtn_cc = self.copy()  # conj()
                tens0 = gtn_cc.get_anchor_tens()  ## anchor tensor that will have ancilla
                if ancilla_reindex is not None:
                    tens0.reindex({anc: anc_new for anc, anc_new in ancilla_reindex.items()}, inplace=True)
                tmp = gtn_cc.apply_elemental_multiply_op(take_mps_cc=True)
                integ_gtn = tmp.apply(integ_gtn, inplace=False)

                # if ancilla_reindex:
                #     fuse_anc = {anc_old: (anc_new, anc_old) for anc_new, anc_old in ancilla_reindex.items()}
                #     integ_mps[self.get_anchor_ind()].fuse(fuse_anc, inplace=True)

                # v_ax = self.grid.axes[1]
                # plt.figure()
                # plt.imshow(np.abs(tmp.get_data(ax_select={v_ax: v_ax.npts // 2})))
                # plt.colorbar()
                # plt.title('cc')

            integ_mps = integ_gtn.data  # .copy()

            if obs_gtn is None:
                integ_mps.lower_ind_id = '_tmp{}_'
                integ_mps.upper_ind_id = self.data.site_ind_id
                comp_mps = self.data.reindex_sites('_tmp{}_')  ## not inplace

                tags_list = [integ_mps.site_tag_id, self.data.site_tag_id]
                if scipy.sparse.issparse(integ_mps[0].data):
                    if integ_mps[0].inds[-1] == integ_mps.lower_ind_id.format(0):
                        result = integ_mps[0].data @ comp_mps[0].data
                    else:
                        result = comp_mps[0].data @ integ_mps[0].data
                    result = qtn.Tensor(data=result, inds=(integ_mps.upper_ind_id.format(0),),
                                        tags=self.data.site_tag_id.format(0))
                    result = qtn.TensorNetwork([result])
                else:
                    result = qtn.TensorNetwork((integ_mps, comp_mps),
                                               virtual=True, check_collisions=False)
                result.exponent = integ_mps.exponent + comp_mps.exponent
            else:
                integ_mps.lower_ind_id = '_tmpU{}_'
                integ_mps.upper_ind_id = self.data.site_ind_id
                # integ_mps.site_tag_id = integ_mps.site_tag_id + 'I'
                obs_mpo = obs_gtn.data.copy()
                obs_mpo.mangle_inner_()
                obs_mpo.lower_ind_id = '_tmp{}_'
                obs_mpo.upper_ind_id = '_tmpU{}_'
                # obs_mpo.site_tag_id = obs_mpo.site_tag_id + 'O'
                comp_mps = self.data.reindex_sites('_tmp{}_')  ## not inplace

                tags_list = [integ_mps.site_tag_id, self.data.site_tag_id, obs_mpo.site_tag_id]

                if obs_mpo.L == 1 and (scipy.sparse.issparse(obs_mpo[0].data)
                                       or scipy.sparse.issparse(integ_mps[0].data)):

                    if obs_mpo[0].inds[-1] == comp_mps[0].inds[0]:
                        vec_data = obs_mpo[0].data @ comp_mps[0].data
                    else:
                        vec_data = comp_mps[0].data @ obs_mpo[0].data

                    if integ_mps[0].inds[-1] == integ_mps.lower_ind_id.format(0):
                        result = integ_mps[0].data @ vec_data
                    else:
                        result = vec_data @ integ_mps[0].data
                    result = result * 10 ** (integ_mps.exponent + comp_mps.exponent + obs_mpo.exponent)

                    result = qtn.Tensor(data=result, inds=(integ_mps.upper_ind_id.format(0),),
                                        tags=self.data.site_tag_id.format(0))
                    result = qtn.TensorNetwork([result])
                else:
                    result = qtn.TensorNetwork((integ_mps, obs_mpo, comp_mps),
                                               virtual=True, check_collisions=False)
                    result.exponent = (integ_mps.exponent + comp_mps.exponent + obs_mpo.exponent)
                # print('result norm', result.tensors[0].norm())
                # print('result exponent', integ_mps.exponent, comp_mps.exponent, obs_mpo.exponent)

            ## set contraction direction to minimize costs
            if self.grid.axes[-1] not in remaining_axes:
                site_range = range(integ_mps.L - 1, -1, -1)
                from_right = True
            else:
                site_range = range(integ_mps.L)
                from_right = False

            new_exponent = result.exponent
            for x in site_range:
                # tens_tags = [integ_mps.site_tag_id.format(x), self.data.site_tag_id.format(x)]
                tens_tags = [tag.format(x) for tag in tags_list]

                if from_right:
                    neighbor_tags = [tag.format(x - 1) for tag in tags_list]
                else:
                    neighbor_tags = [tag.format(x + 1) for tag in tags_list]

                try:
                    if integ_mps.phys_dim(x, 'upper') == 1:
                        ## if and MPO with 1 upper physical bonds, get rid of the inds of size 1
                        # print('select', result.select_tensors(tens_tags + neighbor_tags, which='any'))
                        result = result.contract(tens_tags + neighbor_tags, inplace=True, which='any')
                        if isinstance(result, qtn.Tensor):
                            tens = result
                        else:
                            tens = result.select_tensors(tens_tags, which='any')[0]
                        tens.isel({integ_mps.upper_ind(x): 0}, inplace=True)
                        tens.drop_tags(tens_tags)
                        # print('tens', tens, x)
                    else:
                        raise TypeError

                except TypeError:  ## don't integrate over bonds; end of chain
                    if isinstance(result, qtn.TensorNetwork) and result.num_tensors > 1:
                        result = result.contract(tens_tags, inplace=True, which='any')

                        ## QR compression of contraction
                        if isinstance(result, qtn.TensorNetwork):
                            tens_x = result.select_tensors(tens_tags)[0]
                            try:
                                tens_neighbor = result.select(neighbor_tags, which='any')
                                iso_inds = set(tens_neighbor.outer_inds()).intersection(tens_x.inds)
                                # iso_size = np.prod([tens_x.ind_size(ind) for ind in list(iso_inds)])
                                # out_size = np.prod([tens_x.ind_size(ind) for ind in tens_x.inds if ind not in iso_inds])
                                if True:  # iso_size > out_size:
                                    tens_x1, tens_x2 = qtn.tensor_split(tens_x, list(iso_inds), method='qr', absorb='left',
                                                                        cutoff=CUTOFF, cutoff_mode=CUTOFF_MODE)
                                    tens_x1.drop_tags()
                                    tens_x1.add_tag(neighbor_tags[0])
                                    result.delete(tens_x.tags, which='all')
                                    result.add([tens_x1, tens_x2])
                            except KeyError:
                                pass  ## no neighbors / end of the chain
                            # print('results', x, result)

            if isinstance(result, qtn.Tensor):
                ## already caught integ_axes = all case, so will always need to be in TN form
                result = qtn.TensorNetwork([result])
                result.exponent = new_exponent

            result.fuse_multibonds(inplace=True)

            if self.data.site_tag_id != integ_mps.site_tag_id:
                result.drop_tags([integ_mps.site_tag_id.format(x) for x in site_range])

            ## renumber tensors
            if not self.grid.layout_type == LayoutType.PARALLEL_GROUP:
                remaining_inds = np.sort(remaining_inds)
                helper.renumber_mps(result, remaining_inds, range(new_grid.L), self.data.site_tag_id,
                                    self.data.site_ind_id, inplace=True)

            ## convert to MPS
            result.view_like(self.data, L=new_grid.L, inplace=True)
            # result.exponent = integ_mps.exponent + comp_mps.exponent

            ## convert to GTN object
            gtn_result = self.__class__(new_grid, data=result, ax_deriv_configs=new_ax_deriv_configs)

            if compress:
                gtn_result.compress(compress_opts=compress_opts)

            return gtn_result

    def project(self, obs_gtn, proj_axes, new_grid=None, canonize=True, compress=False, compress_opts=None,
                ancilla_reindex=None, new_ax_deriv_configs=None, **kwargs) -> Optional['GridTN1D']:
        """Project an operator onto the tangent manifold of self: PROJ O|state>.

        Canonizes self about the projection axes, contracts the integrated axes
        against the conjugated state and (optionally) the observable, and
        returns the reduced result as a GridTN1D on the projection axes.

        Parameters
        ----------
        obs_gtn : GridTN1D
            Observable (MPO) to project; None projects the bare state.
        proj_axes : Sequence[Axis]
            Axes retained in the projected output.
        new_grid : Grid1D, optional
            If given (truthy), a projection subgrid is built from ``proj_axes``.
        canonize : bool
            Canonize about the projection axes first.
        compress : bool
            Compress the projected result.
        compress_opts : dict, optional
            Overrides for compression parameters.
        ancilla_reindex : dict, optional
            Mapping old_anc -> new_anc applied to the conjugate anchor tensor.
        new_ax_deriv_configs : dict, optional
            Derivative configs for the output grid.
        **kwargs
            Additional options (ignored).

        Returns
        -------
        GridTN1D or None
            The projected state, or None if self has no data.
        """
        if self.data is None:
            return None

        gtn = self.copy()
        if canonize:
            gtn = gtn.canonize_axes(proj_axes, inplace=True, scale=False)
            ## need projection tensors to actually be in canonical form (no coeffs)

        if new_grid is not None:
            # select_axes = [ax for ax in gtn.grid.axes if ax not in proj_axes]
            # print('grid select_axes', select_axes)
            new_grid = gtn.grid.get_subgrid(proj_axes, new_gridID=self.grid.gridID + '_proj')

        if new_ax_deriv_configs is None and new_grid is not None:
            new_ax_deriv_configs = {ax: self.ax_deriv_configs[ax] for ax in new_grid.axes}

        # print('project check orthog')
        # helper.check_orthog(gtn.data)
        # print('gtn data', helper.norm(gtn.data))

        integ_axes = [ax for ax in self.grid.axes if ax not in proj_axes]
        integ_inds = []
        for ax in integ_axes:
            integ_inds += self.grid.get_inds_in_axis(ax)
        # print('integ inds', integ_inds)

        remaining_axes = proj_axes
        remaining_inds = []
        for ax in remaining_axes:
            remaining_inds += self.grid.get_inds_in_axis(ax)

        gtn_cc = gtn.conj(mangle_inner=False, inplace=False)
        tens0 = gtn_cc.get_anchor_tens()  ## anchor tensor that will have ancilla
        if ancilla_reindex is not None:
            tens0.reindex({anc: anc_new for anc, anc_new in ancilla_reindex.items()}, inplace=True)

        gtn.data.mangle_inner_(append='_')
        proj_mps = gtn_cc.data

        if obs_gtn is None:
            proj_mps.site_ind_id = gtn.data.site_ind_id
            comp_mps = gtn.data.reindex_sites('_tmp{}_')  ## not inplace
            proj_tens = [proj_mps[ind] for ind in integ_inds]

            tags_list = [proj_mps.site_tag_id, gtn.data.site_tag_id]
            result = qtn.TensorNetwork((proj_tens, comp_mps),
                                       virtual=True, check_collisions=False)
            # result.exponent = comp_mps.exponent
            new_exponent = comp_mps.exponent
        else:
            proj_mps.site_tag_id = gtn.data.site_tag_id
            obs_mpo = obs_gtn.data.copy()
            assert isinstance(obs_mpo, qtn.MatrixProductOperator), 'obs_mpo must be MPO'
            obs_mpo.mangle_inner_()
            obs_mpo.lower_ind_id = '_tmp{}_'
            obs_mpo.upper_ind_id = gtn.data.site_ind_id
            # obs_mpo.site_tag_id = obs_mpo.site_tag_id + 'O'
            comp_mps = gtn.data.reindex_sites('_tmp{}_')  ## not inplace
            proj_tens = [proj_mps[ind] for ind in integ_inds]

            tags_list = [proj_mps.site_tag_id, gtn.data.site_tag_id, obs_mpo.site_tag_id]
            result = qtn.TensorNetwork((proj_tens, obs_mpo, comp_mps),
                                       virtual=True, check_collisions=False)
            # result.exponent = comp_mps.exponent + obs_mpo.exponent
            new_exponent = comp_mps.exponent + obs_mpo.exponent

        ## set contraction direction to minimize costs
        if self.grid.axes[-1] not in remaining_axes:
            site_range = range(proj_mps.L - 1, -1, -1)
            from_right = True
        else:
            site_range = range(proj_mps.L)
            from_right = False

        for x in site_range:
            tens_tags = [tag.format(x) for tag in tags_list]

            if x in integ_inds:
                ## if and MPO with 1 upper physical bonds, get rid of the inds of size 1
                if from_right:
                    neighbor_tags = [tag.format(x - 1) for tag in tags_list] if x > 0 \
                        else [tag.format(np.min(remaining_inds)) for tag in tags_list]
                else:
                    neighbor_tags = [tag.format(x + 1) for tag in tags_list] if x < self.L \
                        else [tag.format(np.max(remaining_inds)) for tag in tags_list]

                # print('contract', x, tens_tags, neighbor_tags)
                result = result.contract(tens_tags + neighbor_tags, inplace=True, which='any')
                if isinstance(result, qtn.Tensor):
                    tens = result
                else:
                    tens = result.select_tensors(tens_tags, which='any')[0]
                tens.drop_tags(tens_tags)
                # print('tens', tens, x)
            else:
                result = result.contract(tens_tags, inplace=True, which='any')

            # print('result', result)

        if isinstance(result, qtn.Tensor):
            ## already caught integ_axes = all case, so will always need to be in TN form
            result = qtn.TensorNetwork([result])
            result.exponent = new_exponent
        else:
            result.fuse_multibonds(inplace=True)
            result.exponent = new_exponent

        # print('result', helper.norm(result), result.exponent, new_exponent)

        if gtn.data.site_tag_id != proj_mps.site_tag_id:
            result.drop_tags([proj_mps.site_tag_id.format(x) for x in site_range])

        ## renumber tensors
        if not gtn.grid.layout_type == LayoutType.PARALLEL_GROUP:
            remaining_inds = np.sort(remaining_inds)
            helper.renumber_mps(result, remaining_inds, range(new_grid.L), self.data.site_tag_id,
                                gtn.data.site_ind_id, inplace=True)

        ## convert to MPS
        result = result.view_like(self.data, L=new_grid.L, inplace=True)
        result.exponent = new_exponent
        # result.exponent = integ_mps.exponent + comp_mps.exponent

        ## convert to GTN object
        gtn_result = self.__class__(new_grid, data=result, ax_deriv_configs=new_ax_deriv_configs)

        # plt.figure()
        # plt.imshow(obs_gtn.get_data(ax_select = {obs_gtn.grid.axes[1]: (1,2)}))
        # plt.title('projected mpo (x)')
        # plt.colorbar()
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(obs_gtn.get_data(ax_select={obs_gtn.grid.axes[0]: 32}))
        # plt.title('projected mpo (v)')
        # plt.colorbar()
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(gtn_result.get_data(), aspect='auto')
        # plt.title('projected result')
        # plt.colorbar()
        # plt.show()
        #
        # from grid1D import Grid1D
        #
        # x_L = obs_gtn.grid.axes[0].L
        # grid_x = Grid1D('x', [obs_gtn.grid.axes[0]])
        # obs_x = obs_gtn.data[:x_L].copy()
        # obs_x._L = x_L
        # obs_x[-1].squeeze(inplace=True)
        # obs_x_gtn = grid_x.make_gridTN(obs_x)
        # psi_x = gtn.data[:x_L].copy()
        # psi_x._L = x_L
        # print('obs_x', obs_x)
        # print('psi_x', psi_x)
        # psi_x[-1].squeeze(inplace=True)
        # psi_x_gtn = grid_x.make_gridTN(psi_x)
        # Op_x = psi_x_gtn.apply(obs_x_gtn)
        # plt.figure()
        # plt.plot(psi_x_gtn.get_data(), label='f')
        # plt.plot(np.diag(obs_x_gtn.get_data()), label='diag op')
        # plt.plot(Op_x.get_data(), label='op*f')
        # print('expec x', helper.expectation_value(psi_x, obs_x))
        # plt.show()

        if compress:
            gtn_result.compress(compress_opts=compress_opts, inplace=True)

        return gtn_result

    def integrate(self, integ_axes=None, is_sqrt=False, new_grid=None, new_ax_deriv_configs=None, exclude_weights=False,
                  compress=False, compress_opts=None, ancilla_reindex: dict[str, str] = None, **kwargs) \
            -> Union[Numeric, qtn.Tensor]:
        """Integrate the state (optionally as a quadratic form) over axes.

        Thin wrapper over :meth:`meas_expec` with no observable: computes
        integral dx f O (linear) or integral dx g* O g (when ``is_sqrt``).

        Parameters
        ----------
        integ_axes : Sequence[Axis], optional
            Axes to integrate over; defaults to all axes.
        is_sqrt : bool
            Integrate the quadratic form g* g rather than the linear f.
        new_grid : Grid1D, optional
            Output grid for a partial integration.
        new_ax_deriv_configs : dict, optional
            Derivative configs for the output grid.
        exclude_weights : bool
            Use unit weights instead of axis quadrature weights.
        compress : bool
            Compress a reduced result.
        compress_opts : dict, optional
            Overrides for compression parameters.
        ancilla_reindex : dict[str, str], optional
            Ancilla reindexing forwarded to :meth:`meas_expec`.
        **kwargs
            Additional options (ignored).

        Returns
        -------
        Numeric or qtn.Tensor
            The integral value, NaN if there is no data.
        """
        if self.data is None:
            return np.nan

        return self.meas_expec(None, is_sqrt=is_sqrt, integ_axes=integ_axes, new_grid=new_grid,
                               exclude_weights=exclude_weights, new_ax_deriv_configs=new_ax_deriv_configs,
                               ancilla_reindex=ancilla_reindex, compress=compress, compress_opts=compress_opts)

    def meas_elem(self, sel_inds: Sequence[int], site_ind_id='i({})', site_tag_id='X({})',):
        """Read out a single tensor-network element by its per-core indices.

        Parameters
        ----------
        sel_inds : Sequence[int]
            Index along each tensor core (in core order) of the element.
        site_ind_id : str
            Format string for site (physical) index names of the selector MPS.
        site_tag_id : str
            Format string for site tags of the selector MPS.

        Returns
        -------
        Numeric
            The selected element value (overlap with the selector MPS).
        """
        q = self.grid.axes[0].q
        sel_mps = get_select_elem_mps(self.grid.L, q, sel_inds, site_ind_id, site_tag_id, )
        sel_gtn = self.create_like(new_data=sel_mps)
        return self.ovlp(sel_gtn)


    def meas_shifted_elem(self, sel_inds: Sequence[int], shifts: dict['Axis', int],
                          ax_deriv_configs: ['Axis', 'DerivativeConfiguration'],):
        """Read out an element after applying per-axis index shifts.

        Computes the shifted index and sign (from finite-difference stencils),
        selects the corresponding entry, and scales by the stored exponent and
        sign.

        Parameters
        ----------
        sel_inds : Sequence[int]
            Base per-core indices of the element to read.
        shifts : dict[Axis, int]
            Integer index shift to apply along each axis.
        ax_deriv_configs : dict[Axis, DerivativeConfiguration]
            Per-Axis finite-difference configs determining shift wrap and sign.

        Returns
        -------
        qtn.Tensor
            The shifted element value (scaled by exponent and sign).
        """
        new_sel_inds, new_sign = self.grid.get_shifted_index_and_sign(sel_inds, shifts, ax_deriv_configs)
        site_ind_id = self.data.site_ind_id
        tensors = []
        for it, c in enumerate(new_sel_inds):
            tensors += [self.data[it].isel({site_ind_id.format(it): c})]
        out = qtn.tensor_contract(*tensors)
        out = out * 10**self.exponent * new_sign
        return out

    #######################
    ###  MPS_USVT fcts  ###
    #######################

    def convert_to_USVT(self, canon_site, inplace=False, cur_orthog=None):
        """Convert the MPS data to USVT (explicit singular-value) form.

        Parameters
        ----------
        canon_site : int
            Bond/site at which the singular-value (S) tensor is placed.
        inplace : bool
            Mutate self in place if True, else return a copy.
        cur_orthog : int, optional
            Current orthogonality center to speed up the conversion.

        Returns
        -------
        GridTN1D
            The state with USVT-form data.
        """
        out = MPS_USVT.from_MPS(self.data, canon_site=canon_site, cur_orthog=cur_orthog)
        if inplace:
            self.data = out
            return self
        else:
            return self.create_like(new_data=out)

    def convert_from_USVT(self, inplace=False, canon_site=None):
        """Convert USVT-form data back to a plain MPS.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        canon_site : int, optional
            If given, re-canonize about this site after conversion.

        Returns
        -------
        GridTN1D
            The state with plain MPS data.
        """
        new_gtn = self if inplace else self.copy()
        cur_orthog = self.canon_site
        assert (isinstance(new_gtn.data, MPS_USVT)), 'new_gtn data needs to be in MPS_USVT form'
        new_gtn.data.to_MPS(inplace=True)
        if canon_site is not None:
            new_gtn.canonize(inplace=True, scale=False, i=canon_site, cur_orthog=cur_orthog)
        return new_gtn

    def get_S_tensor(self, **kwargs):
        """Return the singular-value (S) tensor of USVT-form data.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.

        Returns
        -------
        qtn.Tensor
            The S tensor.
        """
        assert (isinstance(self.data, MPS_USVT)), 'new_gtn data needs to be in MPS_USVT form'
        return self.data.get_S_tensor()

    def project_bond(self, obs_gtn, bond_ind):
        """Project an MPO observable onto the S bond of USVT-form data.

        Contracts the ket, conjugated bra (with mangled inner bonds), and the
        observable while leaving the bond around the S tensor open.

        Parameters
        ----------
        obs_gtn : GridTN1D
            Observable MPO supplying the upper/lower index ids.
        bond_ind : int
            Site whose bond to the S tensor is projected open.

        Returns
        -------
        qtn.Tensor
            The projected (bond-open) observable tensor.
        """
        self_data = self.data.copy()
        conj_data = self.data.conj(inplace=False, mangle_inner=True)
        s_tens = self_data.select_tensors(self_data.s_tag)[0]
        s_conj = conj_data._pop_tensor(next(iter(conj_data._get_tids_from_tags(s_tens.tags))))
        self_ind_L = next(iter(qtn.bonds(self_data[bond_ind], s_tens)))
        conj_ind_L = next(iter(qtn.bonds(conj_data[bond_ind], s_conj)))
        self_ind_R = next(iter(qtn.bonds(self_data[bond_ind + 1], s_tens)))
        conj_ind_R = next(iter(qtn.bonds(conj_data[bond_ind + 1], s_conj)))
        reindex_dict = {conj_ind_L: self_ind_L, conj_ind_R: self_ind_R}

        self_data.site_ind_id = obs_gtn.data.lower_ind_id
        conj_data.site_ind_id = obs_gtn.data.upper_ind_id

        expec_tn = qtn.TensorNetwork((self_data, conj_data, obs_gtn.data))
        exponent = obs_gtn.exponent + self_data.exponent
        out: 'qtn.Tensor' = expec_tn.contract_tags(all)
        out.modify(apply=lambda x: x * 10 ** exponent)
        out.reindex(reindex_dict, inplace=True)
        return out

    def project_site(self, obs_gtn, site_ind, nsites=1) -> 'qtn.Tensor':
        """Project an MPO observable onto the open legs of a local site block.

        Contracts the ket, conjugated bra, and observable, leaving the physical
        legs of the ``nsites`` block (and any boundary bonds) open.

        Parameters
        ----------
        obs_gtn : GridTN1D
            Observable MPO supplying the upper/lower index ids.
        site_ind : int
            Starting site of the local block.
        nsites : int
            Number of sites in the local block (1- or 2-site update).

        Returns
        -------
        qtn.Tensor
            The projected local tensor with reindexed bonds.
        """
        self_data = self.data.copy()
        conj_data = self.data.conj(inplace=False, mangle_inner=True)
        # print('project site', site_ind, 'nsites', nsites)
        # helper.check_orthog(self_data)
        self_data.site_ind_id = obs_gtn.data.lower_ind_id
        conj_data.site_ind_id = obs_gtn.data.upper_ind_id
        exponent = self_data.exponent + obs_gtn.data.exponent

        reindex_dict = {}  # obs_gtn.data.upper_ind_id: obs_gtn.data.lower_ind_id}
        for ix in range(site_ind, site_ind + nsites):
            reindex_dict[obs_gtn.data.upper_ind_id.format(ix)] = obs_gtn.data.lower_ind_id.format(ix)
        if site_ind > 0:
            self_ind_L = self_data.bond(site_ind, site_ind - 1)
            conj_ind_L = conj_data.bond(site_ind, site_ind - 1)
            reindex_dict[conj_ind_L] = self_ind_L
        if site_ind < self.data.L - 1 - nsites:
            self_ind_R = self_data.bond(site_ind + nsites - 1, site_ind + nsites)
            conj_ind_R = conj_data.bond(site_ind + nsites - 1, site_ind + nsites)
            reindex_dict[conj_ind_R] = self_ind_R

        for s_ind in range(site_ind, site_ind + nsites):
            s_tens = self_data[s_ind]
            s_conj = conj_data._pop_tensor(next(iter(conj_data._get_tids_from_tags(s_tens.tags))))  ## tossed

        expec_tn = qtn.TensorNetwork([self_data, conj_data, obs_gtn.data])
        out: 'qtn.Tensor' = expec_tn.contract_tags(all)
        out.modify(apply=lambda x: x * 10 ** exponent)
        # print('proj site norm', out.norm(), exponent, self_data.exponent, obs_gtn.data.exponent)
        out.reindex(reindex_dict, inplace=True)
        return out

    def project_op_site(self, obs_gtn, site_ind, nsites=1, left_env=None, right_env=None) \
            -> tuple['qtn.Tensor', Sequence[str], Sequence[str]]:
        """Build the local effective operator for an MPO over a site block.

        Contracts ket, conjugated bra, and observable while popping the block's
        own site tensors, yielding an operator tensor acting on the block legs
        plus boundary bonds (returned as separate input/output index lists).

        Parameters
        ----------
        obs_gtn : GridTN1D
            Observable MPO supplying the upper/lower index ids.
        site_ind : int
            Starting site of the local block.
        nsites : int
            Number of sites in the local block (1- or 2-site update).
        left_env : qtn.Tensor, optional
            Precomputed left environment (currently unused).
        right_env : qtn.Tensor, optional
            Precomputed right environment (currently unused).

        Returns
        -------
        tuple[qtn.Tensor, Sequence[str], Sequence[str]]
            The local operator tensor and its input/output index name lists.
        """
        self_data = self.data.copy()
        conj_data = self.data.conj(inplace=False, mangle_inner=False)
        conj_data.mangle_inner_(append='_')
        obs_gtn_data = obs_gtn.data.mangle_inner_()
        # print('project site', site_ind, 'nsites', nsites)
        # helper.check_orthog(self_data)
        self_data.site_ind_id = obs_gtn.data.lower_ind_id
        conj_data.site_ind_id = obs_gtn.data.upper_ind_id
        exponent = obs_gtn.data.exponent

        # reindex_dict = {}  # obs_gtn.data.upper_ind_id: obs_gtn.data.lower_ind_id}
        # for ix in range(site_ind, site_ind + nsites):
        #     reindex_dict[obs_gtn.data.upper_ind_id.format(ix)] = obs_gtn.data.lower_ind_id.format(ix)
        self_ind_L, conj_ind_L = None, None
        if site_ind > 0:
            self_ind_L = self_data.bond(site_ind, site_ind - 1)
            conj_ind_L = conj_data.bond(site_ind, site_ind - 1)
            # reindex_dict[conj_ind_L] = self_ind_L
        self_ind_R, conj_ind_R = None, None
        if site_ind < self.data.L - nsites:
            self_ind_R = self_data.bond(site_ind + nsites - 1, site_ind + nsites)
            conj_ind_R = conj_data.bond(site_ind + nsites - 1, site_ind + nsites)
            # reindex_dict[conj_ind_R] = self_ind_R

        for s_ind in range(site_ind, site_ind + nsites):
            s_tens = self_data[s_ind]
            s_tens = self_data._pop_tensor(next(iter(self_data._get_tids_from_tags(s_tens.tags))))  ## tossed
            s_conj = conj_data._pop_tensor(next(iter(conj_data._get_tids_from_tags(s_tens.tags))))  ## tossed

        expec_tn = qtn.TensorNetwork([self_data, conj_data, obs_gtn_data])
        out: 'qtn.Tensor' = expec_tn.contract_tags(all)
        out.modify(apply=lambda x: x * 10 ** exponent)
        bonds_i = [obs_gtn.data.lower_ind_id.format(ix) for ix in range(site_ind, site_ind + nsites)]
        if self_ind_L is not None:
            bonds_i += [self_ind_L]
        if self_ind_R is not None:
            bonds_i += [self_ind_R]

        bonds_o = [obs_gtn.data.upper_ind_id.format(ix) for ix in range(site_ind, site_ind + nsites)]
        if conj_ind_L is not None:
            bonds_o += [conj_ind_L]
        if conj_ind_R is not None:
            bonds_o += [conj_ind_R]
        # print('proj site norm', out.norm(), exponent, self_data.exponent, obs_gtn.data.exponent)
        # out.reindex(reindex_dict, inplace=True)
        return out, bonds_i, bonds_o

    def project_op_bond(self, obs_gtn, bond_ind, left_env=None, right_env=None, **kwargs) \
            -> tuple['qtn.Tensor', Sequence[str], Sequence[str]]:
        """Build the local effective operator for an MPO across the S bond.

        Projects ``obs_gtn`` onto the LLL-(S)-RRR structure of USVT-form data,
        popping the S tensor and returning the operator on the surrounding
        bonds (as separate input/output index lists).

        Parameters
        ----------
        obs_gtn : GridTN1D
            Observable MPO supplying the upper/lower index ids.
        bond_ind : int
            Site whose bond to the S tensor is projected.
        left_env : qtn.Tensor, optional
            Precomputed left environment (currently unused).
        right_env : qtn.Tensor, optional
            Precomputed right environment (currently unused).
        **kwargs
            Additional options (ignored).

        Returns
        -------
        tuple[qtn.Tensor, Sequence[str], Sequence[str]]
            The local operator tensor and its input/output index name lists.
        """
        self_data = self.data.copy()
        conj_data = self.data.conj(inplace=False, mangle_inner=False)
        conj_data.mangle_inner_(append='_')
        exponent = obs_gtn.data.exponent

        s_tens = self_data.select_tensors(self_data.s_tag)[0]
        s_tens = self_data._pop_tensor(next(iter(self_data._get_tids_from_tags(s_tens.tags))))
        s_conj = conj_data._pop_tensor(next(iter(conj_data._get_tids_from_tags(s_tens.tags))))

        # print('s conj', s_tens, self_data[bond_ind], self_data[bond_ind + 1])
        # print('s conj', s_conj, conj_data[bond_ind], conj_data[bond_ind + 1])

        self_ind_L = next(iter(qtn.bonds(self_data[bond_ind], s_tens)))
        conj_ind_L = next(iter(qtn.bonds(conj_data[bond_ind], s_conj)))
        self_ind_R = next(iter(qtn.bonds(self_data[bond_ind + 1], s_tens)))
        conj_ind_R = next(iter(qtn.bonds(conj_data[bond_ind + 1], s_conj)))
        # reindex_dict = {conj_ind_L: self_ind_L, conj_ind_R: self_ind_R}

        self_data.site_ind_id = obs_gtn.data.lower_ind_id
        conj_data.site_ind_id = obs_gtn.data.upper_ind_id

        expec_tn = qtn.TensorNetwork((self_data, conj_data, obs_gtn.data))
        out: 'qtn.Tensor' = expec_tn.contract_tags(all)
        out.modify(apply=lambda x: x * 10 ** exponent)
        # out.reindex(reindex_dict, inplace=True)

        bonds_i = []
        if self_ind_L is not None:
            bonds_i += [self_ind_L]
        if self_ind_R is not None:
            bonds_i += [self_ind_R]

        bonds_o = []
        if conj_ind_L is not None:
            bonds_o += [conj_ind_L]
        if conj_ind_R is not None:
            bonds_o += [conj_ind_R]
        # print('proj site norm', out.norm(), exponent, self_data.exponent, obs_gtn.data.exponent)
        # out.reindex(reindex_dict, inplace=True)
        return out, bonds_i, bonds_o

    def evolve_tdvp(self, dt, mpo_list, te_order=0, do_adapt=False, inplace=False,
                    compress_config: CompressionConfiguration = None, expand_basis=None):
        """Advance the state one TDVP time step via the helper_tdvp_v2 solver.

        Distributes operator exponents, optionally expands the basis, then runs
        ``TDVPSolver_v2.take_time_step`` and (if a basis was expanded)
        recompresses the result.

        Parameters
        ----------
        dt : float
            Time step.
        mpo_list : Sequence[GridTN1D]
            Hamiltonian MPO terms; an empty list returns the state unchanged.
        te_order : int
            Time-integration order code selecting the local integrator.
        do_adapt : bool
            Rank-adaptation flag (forced to False inside this method).
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        compress_config : CompressionConfiguration, optional
            Source of compression options (level 1) for the solver/recompress.
        expand_basis : Sequence[GridTN], optional
            Subspace vectors to expand the basis with before stepping.

        Returns
        -------
        GridTN1D
            The evolved state.
        """
        print('tdvp 1D', te_order)
        gtn = self if inplace else self.copy()

        if len(mpo_list) == 0:
            return gtn

        # print('gtn exponent', gtn.exponent)
        # gtn.data.distribute_exponent()

        # plt.figure()
        # plt.imshow(gtn.get_data())
        # plt.colorbar()
        # plt.title('seq')
        # plt.show()

        # for mpo in mpo_list:
        #     ax_x, ax_v = gtn.grid.axes
        #     mpo_data = mpo.get_data(ax_select={ax_v: (1, 2)})
        #     plt.figure()
        #     plt.imshow(mpo_data)
        #     plt.colorbar()
        #
        #     mpo_data = mpo.get_data(ax_select={ax_x: (0, 0)})
        #     plt.figure()
        #     plt.imshow(mpo_data)
        #     plt.colorbar()
        #     plt.show()
        #     # print('mpo exp', mpo.data.exponent)
        #     # mpo.data.distribute_exponent()

        for mpo in mpo_list:
            mpo.data.distribute_exponent()

        ## expand basis
        if expand_basis is not None:
            gtn.expand_subspace(expand_basis, orthog_direction=-1, inplace=True,
                                compress_opts=compress_config.get_compress_opts(1))

        # from helper_tdvp_v2 import TDVPSolver as tdsolver
        from helper_tdvp_v2 import TDVPSolver_v2 as tdsolver
        # from helper_tddmrg import TDDMRG_Solver as tdsolver
        # from helper_tdvp_v2 import TDDMRGSolver_v4 as tdsolver

        dist_mpx = gtn.data

        # print('dist mpx', dist_mpx.max_bond())
        # deriv_mpx = [self.apply(mpo_).scalar_multiply(dt) for mpo_ in mpo_list]
        # dist_mpx = helper.add_MPS_list([dist_mpx, *[mpx.data for mpx in deriv_mpx]], do_final_update=False)
        # print('dist mpx', dist_mpx.max_bond())
        # pdb.set_trace()

        # dist_mpx.distribute_exponent()
        tdvp_solver = tdsolver(dist_mpx, operators=[mpo.data for mpo in mpo_list], te_order=te_order,
                                      compress_config=compress_config)
        do_adapt=False  # True
        tdvp_solver.take_time_step(dt, do_adapt=do_adapt)
        gtn.data = tdvp_solver.ket  ## not actually an inplace operation...
        if expand_basis is not None:
            gtn.compress(compress_opts=compress_config.get_compress_opts(1), inplace=True)
            print('compressed tdvp gtn', gtn.max_bond())

        return gtn

    def evolve_tdmrg(self, dt, mpo_list, te_order=0, do_adapt=True, inplace=False, compress_config=None):
        """Advance the state one TD-DMRG time step via the helper_tdvp_v2 solver.

        Wraps ``TDMRGSolver.take_time_step`` over the given operator MPOs.

        Parameters
        ----------
        dt : float
            Time step.
        mpo_list : Sequence[GridTN1D]
            Hamiltonian MPO terms.
        te_order : int
            Time-integration order code selecting the local integrator.
        do_adapt : bool
            Rank-adaptation flag forwarded to the solver.
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        compress_config : CompressionConfiguration, optional
            Compression configuration forwarded to the solver.

        Returns
        -------
        GridTN1D
            The evolved state.
        """
        print('tdmrg 1D', te_order)
        gtn = self if inplace else self.copy()
        dist_mpx = gtn.data

        from helper_tdvp_v2 import TDMRGSolver as tdsolver
        # from helper_tdvp_v2 import TDDMRGSolver_v2 as tdsolver
        # from helper_tdvp_v2 import TDDMRGSolver_v4 as tdsolver

        # dist_mpx.distribute_exponent()
        # for mpo in mpo_list:
        #     mpo.data.distribute_exponent()

        tdvp_solver = tdsolver(dist_mpx, operators=[mpo.data for mpo in mpo_list], te_order=te_order,
                                  compress_config=compress_config)
        tdvp_solver.take_time_step(dt, do_adapt=do_adapt)
        gtn.data = tdvp_solver.ket  ## not actually an inplace operation...
        return gtn

    def evolve_tdvp_new(self, dt, linear_mpo_list, te_order=0, do_adapt=True, inplace=False,
                        compress_config: CompressionConfiguration = None, nonlinear_terms=None, sources=None,
                        solver_type=LocalSolverType.TDDMRG, filter_bases=False, time=None,
                        upwind_func=None, upwind_deriv_func=None, direction=1, verbose=False, **kwargs):
        """Advance the state one step with the two-sweep local time integrator.

        Performs a half-step (dt/2) left-to-right canonizing sweep followed by a
        half-step right-to-left sweep. The integrator is selected by
        ``solver_type``/``te_order``: TDDMRG/DMRG use the DMRG-style TDVP_DMRG;
        otherwise the cross (TDVPCross) or mixed (TDVPMixed) integrators are
        used. Records ``gtn.info['num_evals']`` and ``gtn.info['internal_rank']``,
        then compresses using max_bond from compress_config level 1 and cutoff
        from level 2.

        Parameters
        ----------
        dt : float
            Full time step (each sweep advances dt/2).
        linear_mpo_list : Sequence[GridTN1D]
            Linear operator MPO terms.
        te_order : int
            Time-integration order code selecting the local integrator.
        do_adapt : bool
            Rank adaptation; widens the local update (nsites) when set.
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        compress_config : CompressionConfiguration, optional
            Source of max_bond (level 1) and cutoff (level 2) options.
        nonlinear_terms : optional
            Nonlinear term objects forwarded to the integrator.
        sources : Sequence[GridTN1D], optional
            Source terms; their data is forwarded to the integrator.
        solver_type : LocalSolverType
            Selects the DMRG-style vs cross/mixed local integrator.
        filter_bases : bool
            Filter local bases during the sweeps.
        time : float, optional
            Current simulation time (advanced by dt/2 between sweeps).
        upwind_func : callable, optional
            Upwind flux function passed to the integrator.
        upwind_deriv_func : callable, optional
            Upwind derivative function passed to the integrator.
        direction : int
            Nominal sweep direction (the method itself does l2r then r2l).
        verbose : bool
            Print diagnostic information.
        **kwargs
            Additional options forwarded to the time integrator.

        Returns
        -------
        GridTN1D
            The evolved state.
        """
        max_bond = compress_config.get_compress_opts(2)['max_bond'] if compress_config is not None else None
        cutoff = compress_config.get_compress_opts(2).get('cutoff', None) if compress_config is not None else None
        if verbose:
            print('tdvp max bond, cutoff', max_bond, cutoff)
        compress_opts = compress_config.get_compress_opts(1)

        if verbose:
            print('tdvp 1D new', 'direction', direction, 'te order', te_order)
        gtn = self if inplace else self.copy()
        dist_mpx = gtn.data.copy()

        dist_mpx.distribute_exponent()      ## bug somewhere -- doesn't work if exponent not distributed
        for mpo in linear_mpo_list:
            mpo.data.distribute_exponent()

        helper.canonize(dist_mpx, i=0, scale=False)

        sources = [s.data for s in sources] if sources is not None else None

        if solver_type in [LocalSolverType.TDDMRG, LocalSolverType.DMRG]:
            # from local_solvers.time_integrator import TDVP_Krylov_DMRG as TimeInteg
            from local_solvers.time_integrator import TDVP_DMRG as TimeInteg

            # max_bond, max_bond_2 = None, None
            # cutoff = cutoff if cutoff is not None else CUTOFF

            solver = TimeInteg(dist_mpx, [mpo.data for mpo in linear_mpo_list],
                               sources=sources, nonlinear_terms=nonlinear_terms,
                               te_order_target=te_order, te_order_final=te_order,
                               max_bond=max_bond, cutoff=cutoff, dt=dt/2, time=time,
                               # max_bond=max_bond, dt=dt, time=time,
                               # max_tot_iter=2,
                               grid=self.grid, ax_deriv_configs=self.ax_deriv_configs, **kwargs)

            solver.upwind_func = upwind_func
            solver.upwind_deriv_func = upwind_deriv_func
            solver.dt = dt / 2
            solver.time = time

            nsites = 2 if max_bond is None else (3 if do_adapt else 1)
            if verbose:
                print('local tdvp new', solver.dt, 'per sweep', 'nsites', nsites)

            if False: # max_bond_2 is not None and max_bond_2 != max_bond:
                print('w/ post compress')
                #### two sweeps ####
                # print('ket', solver.ket.cur_orthog, helper.check_orthog(solver.ket))
                # print('out', solver.out.cur_orthog, helper.check_orthog(solver.out))
                solver.solve_l2r(nsites, canonize=True, filter_bases=filter_bases)
                # solver.solve(1)
                solver.direction = solver.direction * -1
                solver.solve_r2l(nsites, canonize=False, filter_bases=filter_bases)
                gtn.data = solver.out
                # gtn.compress(inplace=True, compress_opts=compress_opts)
                # print('gtn.data', gtn.data.cur_orthog)

                # ## one sweep
                # if direction > 0:
                #     solver.solve_l2r(nsites, canonize=True, filter_bases=filter_bases)
                # else:
                #     solver.solve_r2l(nsites, canonize=True, filter_bases=filter_bases)
                # gtn.data = solver.out

            else:
                #### two sweeps ####
                solver.solve_l2r(nsites, canonize=True, filter_bases=filter_bases)
                solver.direction = solver.direction * -1
                # solver.update_ket_from_out()
                solver.time = time + dt/2
                solver.solve_r2l(nsites, canonize=False, filter_bases=filter_bases)
                gtn.data = solver.out

                # ## one sweep
                # if direction > 0:
                #     solver.solve_l2r(nsites, canonize=True, filter_bases=filter_bases)
                # else:
                #     solver.solve_r2l(nsites, canonize=True, filter_bases=filter_bases)
                # gtn.data = solver.out

                # ### fourth order trotter
                # p = 2
                # coeff = 1 / (2*p - (2*p)**(1/3))
                # dt1 = coeff * dt
                # dt2 = (1 - coeff * p * 2) * dt
                # print('coeff', coeff, 1 - coeff * p * 2)
                # print('dt1', dt1, 'dt2', dt2)
                # # raise RuntimeError
                #
                # for it in range(2 * p + 1):
                #     solver.dt = dt1
                #     solver.solve_l2r(nsites, canonize=True, filter_bases=filter_bases)
                #     solver.direction = solver.direction * -1
                #     solver.time = time + dt1
                #     solver.solve_r2l(nsites, canonize=False, filter_bases=filter_bases)
                #     solver.direction = solver.direction * -1
                #
                # gtn.data = solver.out

            # solver.solve(1, canonize=True, filter_bases=filter_bases)
            # # solver.update_ket_from_out()
            # # solver.solve(1, canonize=True, filter_bases=filter_bases)
            # gtn.data = solver.out
            if verbose:
                print('TDVP pre compress bonds', gtn.data.max_bond(), helper.inner_bond_sizes(solver.out))
            gtn.info['num_evals'] = solver.num_evals
            gtn.info['internal_rank'] = solver.out.max_bond()

            gtn.compress(inplace=True, compress_opts={**compress_config.get_compress_opts(1), 'form': 'left'},
                         canonize=False)    ## bec cur orthog is at i=0 after r2l sweep

            if verbose:
                print('compress opts', compress_opts)
                print('TDVP post compress bonds', helper.inner_bond_sizes(solver.out))

        else:
            # if te_order == 3:   ## 68
            #     solver_type = LocalSolverType.MIXED
            #     # from local_solvers.time_integrator_mixed import TDVPMixed as TimeInteg
            #     te_order = 4
            # elif te_order == 2:   ## 67
            #     solver_type = LocalSolverType.MIXED
            #     # from local_solvers.time_integrator_mixed import TDVPMixed as TimeInteg
            #     te_order = 223
            # else:   ## 65, 69
            #     from local_solvers.time_integrator_cross_2 import TDVPCross as TimeInteg

            if solver_type is LocalSolverType.MIXED:
                from local_solvers.time_integrator_mixed import TDMixed as TimeInteg
            else:
                from local_solvers.time_integrator_cross_2 import TDCross as TimeInteg


            # max_bond, max_bond_2 = None, None
            # cutoff = CUTOFF   # cutoff * 1.0e-2 if cutoff is not None else CUTOFF
            # cutoff = cutoff if cutoff is not None else CUTOFF
            if verbose:
                print('tdvp-x max bond, cutoff', max_bond, cutoff)

            solver = TimeInteg(dist_mpx, [mpo.data for mpo in linear_mpo_list],
                               sources=sources, nonlinear_terms=nonlinear_terms,
                               te_order_target=te_order, te_order_final=te_order,
                               max_bond=max_bond, cutoff=cutoff, dt=dt, time=time,
                               grid=self.grid, ax_deriv_configs=self.ax_deriv_configs,
                               **kwargs)
            solver.upwind_func = upwind_func
            solver.upwind_deriv_func = upwind_deriv_func
            solver.dt = dt / 2
            solver.time = time

            # print('local tdvp new', solver.dt, 'per sweep')
            nsites = 2 if max_bond is None or dist_mpx.max_bond() < max_bond else 1 # (3 if do_adapt else 1)
            # nsites = 2 if max_bond is None else (3 if do_adapt else 1)

            if False: # max_bond_2 is not None and max_bond_2 != max_bond:
                # print('w/ post compress')
                # #### two sweeps ####
                # print('ket', solver.ket.cur_orthog, helper.check_orthog(solver.ket))
                # print('out', solver.out.cur_orthog, helper.check_orthog(solver.out))
                solver.solve_l2r(nsites, canonize=True, filter_bases=filter_bases)
                # solver.solve(1)
                solver.direction = solver.direction * -1
                solver.solve_r2l(nsites, canonize=False, filter_bases=filter_bases)
                gtn.data = solver.out
                gtn.compress(inplace=True, compress_opts=compress_opts)
                # print('gtn.data', gtn.data.cur_orthog)

                # #### one sweep ####
                # if direction > 0:
                #     solver.solve_l2r(nsites, canonize=True, filter_bases=filter_bases)
                # else:
                #     solver.solve_r2l(nsites, canonize=True, filter_bases=filter_bases)
                # gtn.data = solver.out


            else:
                #### two sweeps ####
                solver = TimeInteg(dist_mpx, [mpo.data for mpo in linear_mpo_list],
                                   sources=sources, nonlinear_terms=nonlinear_terms,
                                   te_order_target=te_order, te_order_final=te_order,
                                   max_bond=max_bond,
                                   cutoff=cutoff,
                                   dt=dt / 2, time=time,
                                   grid=self.grid, ax_deriv_configs=self.ax_deriv_configs,
                                   **kwargs)
                solver.upwind_func = upwind_func
                solver.upwind_deriv_func = upwind_deriv_func

                solver.solve_l2r(nsites, canonize=True, filter_bases=filter_bases)

                # # solver.update_ket_from_out()
                # solver = TimeInteg(solver.out, [mpo.
                #                    data for mpo in linear_mpo_list],
                #                    sources=sources, nonlinear_terms=nonlinear_terms,
                #                    te_order_target=te_order, te_order_final=te_order,
                #                    max_bond=max_bond, cutoff=cutoff,
                #                    dt=dt / 2, time=time,
                #                    grid=self.grid, ax_deriv_configs=self.ax_deriv_configs,
                #                    direction=solver.direction * -1,
                #                    **kwargs)
                # solver.upwind_func = upwind_func
                # solver.upwind_deriv_func = upwind_deriv_func
                solver.time = time + dt / 2 if time is not None else None
                solver.direction = solver.direction * -1
                solver.solve_r2l(nsites, canonize=False, filter_bases=filter_bases)
                gtn.data = solver.out

                # ##### one sweep ####
                # solver = TimeInteg(dist_mpx, [mpo.data for mpo in linear_mpo_list],
                #                    sources=sources, nonlinear_terms=nonlinear_terms,
                #                    te_order_target=te_order, te_order_final=te_order,
                #                    max_bond=max_bond, cutoff=cutoff, dt=dt, time=time,
                #                    grid=self.grid, ax_deriv_configs=self.ax_deriv_configs,
                #                    direction=direction,
                #                    **kwargs)
                # solver.upwind_func = upwind_func
                # solver.upwind_deriv_func = upwind_deriv_func
                #
                # if direction > 0:
                #     solver.solve_l2r(nsites, canonize=True, filter_bases=filter_bases)
                # else:
                #     solver.solve_r2l(nsites, canonize=True, filter_bases=filter_bases)
                # gtn.data = solver.out

                # ### fourth order trotter
                # p = 2
                # coeff = 1 / (2*p - (2*p)**(1/3))
                # dt1 = coeff * dt
                # dt2 = (1 - coeff * p * 2) * dt
                # print('coeff', coeff, 1 - coeff * p * 2)
                # print('dt1', dt1, 'dt2', dt2)
                # # raise RuntimeError
                #
                # for it in range(2 * p + 1):
                #     solver.dt = dt1
                #     solver.solve_l2r(nsites, canonize=True, filter_bases=filter_bases)
                #     solver.direction = solver.direction * -1
                #     solver.time = time + dt1
                #     solver.solve_r2l(nsites, canonize=False, filter_bases=filter_bases)
                #     solver.direction = solver.direction * -1
                #
                # gtn.data = solver.out

            if verbose:
                print('TDVP-X pre compress bonds', gtn.data.max_bond(), helper.inner_bond_sizes(solver.out))

            gtn.info['num_evals'] = solver.num_evals
            gtn.info['internal_rank'] = solver.out.max_bond()

            # print('do x compress')
            # from local_solvers.helper_cross_2 import compress as compress_x
            # out, sel_inds = compress_x(solver.out, form='left', max_bond=max_bond,
            #                            cutoff=cutoff, do_canonize=False)
            # gtn.data = out

            # solver.solve(1, canonize=True, filter_bases=filter_bases)
            # # solver.update_ket_from_out()
            # # solver.solve(1, canonize=True, filter_bases=filter_bases)
            # gtn.data = solver.out

            gtn.compress(inplace=True, compress_opts=compress_opts, canonize=True)
            if verbose:
                print('compress opts', compress_opts)
                print('TDVP-X post compress', helper.inner_bond_sizes(solver.out))


        # print('dist mpx max bond', gtn.data.max_bond())
        # pdb.set_trace()

        return gtn


    def evolve_tdmrg_new(self, dt, linear_mpo_list, te_order=0, do_adapt=True, inplace=False,
                         compress_config: CompressionConfiguration = None, nonlinear_terms=None, sources=None,
                         solver_type=LocalSolverType.TDDMRG, conservative=False,
                         filter_bases=False, verbose_plot=False, verbose=False, time=None, upwind_func=None, upwind_deriv_func=None,
                         direction=1,
                         **kwargs):
        """Advance the state one step with a single directional TD-DMRG sweep.

        Canonizes to site 0, runs one direction-aware (l2r/r2l) sweep with an
        expanded working max_bond (scaled by te_order), records
        ``gtn.info['num_evals']`` and ``gtn.info['internal_rank']``, then
        applies an optional conservative compression back to the target bond.
        The cross/mixed integrators are used for non-DMRG ``solver_type``.

        Parameters
        ----------
        dt : float
            Time step.
        linear_mpo_list : Sequence[GridTN1D]
            Linear operator MPO terms.
        te_order : int
            Time-integration order code; also scales the working max_bond.
        do_adapt : bool
            Rank-adaptation flag forwarded to the integrator.
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        compress_config : CompressionConfiguration, optional
            Source of max_bond (level 1) and cutoff (level 2) options.
        nonlinear_terms : optional
            Nonlinear term objects forwarded to the integrator.
        sources : Sequence[GridTN1D], optional
            Source terms; their data is forwarded to the integrator.
        solver_type : LocalSolverType
            Selects the DMRG-style vs cross/mixed local integrator.
        conservative : bool
            Use mass-conserving compression for the post-sweep truncation.
        filter_bases : bool
            Filter local bases during the sweep.
        verbose_plot : bool
            Emit per-step plots from the integrator.
        verbose : bool
            Print diagnostic information.
        time : float, optional
            Current simulation time.
        upwind_func : callable, optional
            Upwind flux function passed to the integrator.
        upwind_deriv_func : callable, optional
            Upwind derivative function passed to the integrator.
        direction : int
            Sweep direction (l2r if > 0, else r2l).
        **kwargs
            Additional options forwarded to the time integrator.

        Returns
        -------
        GridTN1D
            The evolved state.
        """
        if verbose:
            print('evolve tdmrg new', 'direction', direction, 'te order', te_order)

        max_bond = compress_config.get_compress_opts(1)['max_bond'] if compress_config is not None else None
        if max_bond is not None:
            if te_order in [223, 226]:
                max_bond = max_bond * 3
            else:
                max_bond = (max_bond * max(2,te_order + 1)) if te_order != 0 else None

        if verbose:
            print('td-dmrg expanded max bond', max_bond, compress_config.get_compress_opts(1)['max_bond'])

        compress_opts = compress_config.get_compress_opts(1)
        max_bond_2 = compress_config.get_compress_opts(1)['max_bond'] if compress_config is not None else None
        cutoff = compress_config.get_compress_opts(2).get('cutoff', None) if compress_config is not None else None

        gtn = self if inplace else self.copy()
        dist_mpx = gtn.data  # .copy()

        dist_mpx.distribute_exponent()
        for mpo in linear_mpo_list:
            mpo.data.distribute_exponent()

        helper.canonize(dist_mpx, i=0, scale=False)
        # print('dist mpx')

        sources = [s.data for s in sources] if sources is not None else None
        # for s in sources:
        #     s.distribute_exponent()
        # print('mps srouces', [helper.norm(s) for s in sources])

        # for t in nonlinear_terms:
        #     print('nonlin ket is tddmrg_gtn1d', t.ket is dist_mpx)

        if solver_type in [LocalSolverType.TDDMRG, LocalSolverType.DMRG]:
            from local_solvers.time_integrator import TDDMRG as TimeInteg

            # max_bond, max_bond_2 = None if max_bond is None else max_bond * 2, None
            max_bond, max_bond_2 = None, None
            # cutoff = cutoff if cutoff is not None else CUTOFF
            if verbose:
                print('tddmrg max bond, cutoff', max_bond, cutoff)

            solver = TimeInteg(dist_mpx, [mpo.data for mpo in linear_mpo_list],
                               sources=sources, nonlinear_terms=nonlinear_terms,
                               te_order_target=te_order, te_order_final=te_order,
                               # max_bond=max_bond, dt=dt/2,
                               max_bond=max_bond, cutoff=cutoff, dt=dt, time=time,
                               max_tot_iter=2,
                               grid=self.grid, ax_deriv_configs=self.ax_deriv_configs,
                               verbose_plot=verbose_plot, direction=direction,
                               **kwargs
                               )

            if max_bond_2 is not None and max_bond_2 != max_bond:
                # print('w/ post compress, max_bond', max_bond_2)
                if direction > 0:
                    solver.solve_l2r(1, canonize=True, filter_bases=filter_bases)
                else:
                    # solver.direction = solver.direction * -1
                    solver.solve_r2l(1, canonize=True, filter_bases=filter_bases)
                gtn.data = solver.out
                if verbose:
                    print('tddmrg pre compress bond', solver.out.max_bond())
                gtn.compress(inplace=True, compress_opts=compress_opts, conservative=conservative)
                # print('gtn.data', gtn.data.cur_orthog)

            else:
                if direction > 0:
                    solver.solve_l2r(1, canonize=True, filter_bases=filter_bases)
                else:
                    solver.solve_r2l(1, canonize=True, filter_bases=filter_bases)
                gtn.data = solver.out

                if verbose:
                    print('(Td-dmrg) pre compress ranks', helper.inner_bond_sizes(solver.out))
                gtn.info['num_evals'] = solver.num_evals
                gtn.info['internal_rank'] = solver.out.max_bond()

                gtn.compress(inplace=True, compress_opts=compress_opts, conservative=conservative)

            # solver.solve(1, canonize=True, filter_bases=filter_bases)
            # # solver.update_ket_from_out()
            # # solver.solve(1, canonize=True, filter_bases=filter_bases)
            # gtn.data = solver.out
        else:

            # ## for advection problem?
            # if te_order == 3:   ## 88
            #     solver_type = LocalSolverType.MIXED
            #     te_order = 4
            # elif te_order == 2:   ## 87
            #     solver_type = LocalSolverType.MIXED
            #     te_order = 0
            # else:   ## 85, 89
            #     from local_solvers.time_integrator_cross_2 import TDCross as TimeInteg

            if solver_type is LocalSolverType.MIXED:
                from local_solvers.time_integrator_mixed import TDMixed as TimeInteg
            else:
                from local_solvers.time_integrator_cross_2 import TDCross as TimeInteg

            # max_bond, max_bond_2 = None, None
            # cutoff = cutoff if cutoff is not None else CUTOFF
            if verbose:
                print('tddmrg-x modified max bond, cutoff', max_bond, cutoff)

            nsites = 1

            # print('HERE', upwind_func)
            # exit()

            # max_bond = compress_config.get_compress_opts(1)['max_bond'] if compress_config is not None else None

            ## to do second order time step, would need to reinitialize...
            if verbose:
                print('time integ', TimeInteg)

            solver = TimeInteg(dist_mpx.copy(), [mpo.data for mpo in linear_mpo_list],
                               direction=direction,
                               sources=sources, nonlinear_terms=nonlinear_terms,
                               te_order_target=te_order, te_order_final=te_order,
                               max_bond=max_bond, cutoff=cutoff, dt=dt, time=time,
                               grid=self.grid, verbose_plot=verbose_plot, **kwargs)
            solver.upwind_func = upwind_func
            solver.upwind_deriv_func = upwind_deriv_func

            if direction > 0:
                solver.solve_l2r(nsites, canonize=True)
            else:
                solver.solve_r2l(nsites, canonize=True)
            # solver.solve(nsites)
            # out = solver.solution

            # nonlinear_terms_2 = []
            # for t in nonlinear_terms:
            #     new_term = t.create_like(out.copy())
            #     new_term.direction = -1
            #     nonlinear_terms_2 += [new_term]
            #
            # solver = TimeInteg(out.copy(), [mpo.data for mpo in linear_mpo_list],
            #                    direction=-1,
            #                    sources=sources, nonlinear_terms=nonlinear_terms_2,
            #                    max_bond=max_bond, dt=dt/2)
            # solver.solve_r2l(nsites, canonize=True)
            # # solver.solve(nsites)
            # out = solver.solution

            gtn.data = solver.out  # solution

            if verbose:
                print('tddmrg-x pre compress bond', solver.out.max_bond(), helper.inner_bond_sizes(solver.out))
            gtn.info['num_evals'] = solver.num_evals
            gtn.info['internal_rank'] = solver.out.max_bond()

            if verbose:
                print('compress opts', compress_opts)
            gtn.compress(inplace=True, compress_opts=compress_opts, conservative=conservative)



            # print('do x compress')
            # from local_solvers.helper_cross_2 import compress as compress_x
            # out, sel_inds = compress_x(solver.out, form=('right' if direction > 0 else 'left'), max_bond=max_bond,
            #                            cutoff=cutoff, do_canonize=True)
            # gtn.data = out

            # gtn.compress(inplace=True, canonize=True, compress_opts=compress_opts, conservative=conservative)

        return gtn

    def evolve_time_local_global(self, dt, linear_mpo_list, te_order=0, do_adapt=True, inplace=False,
                                 compress_config: CompressionConfiguration = None, nonlinear_terms=None, sources=None,
                                 solver_type=LocalSolverType.TDDMRG, filter_bases=False, time=None):
        """Advance the state one step with a combined local/global integrator.

        Canonizes to site 0, runs a single left-to-right sweep of the chosen
        TDLocal_Global / TDDMRG_Global / TDVP_Global integrator, then applies a
        direction-aware conservative compression. Only TDDMRG/DMRG solver types
        are supported; others raise NotImplementedError.

        Parameters
        ----------
        dt : float
            Time step.
        linear_mpo_list : Sequence[GridTN1D]
            Linear operator MPO terms.
        te_order : int
            Time-integration order code selecting the local integrator.
        do_adapt : bool
            Rank-adaptation flag (unused in this path).
        inplace : bool
            Mutate self in place if True, else operate on a copy.
        compress_config : CompressionConfiguration, optional
            Source of max_bond and compression options (level 1).
        nonlinear_terms : optional
            Nonlinear term objects forwarded to the integrator.
        sources : Sequence[GridTN1D], optional
            Source terms; their data is forwarded to the integrator.
        solver_type : LocalSolverType
            Must be TDDMRG or DMRG; other values raise NotImplementedError.
        filter_bases : bool
            Filter local bases during the sweep.
        time : float, optional
            Current simulation time.

        Returns
        -------
        GridTN1D
            The evolved state.
        """
        max_bond = compress_config.get_compress_opts(1)['max_bond'] if compress_config is not None else None
        compress_opts = {k: v for k,v in compress_config.get_compress_opts(1).items()}

        print('local/global TE 1D', te_order)
        gtn = self if inplace else self.copy()
        dist_mpx = gtn.data.copy()

        dist_mpx.distribute_exponent()
        for mpo in linear_mpo_list:
            mpo.data.distribute_exponent()

        helper.canonize(dist_mpx, i=0, scale=False)
        # print('dist mpx')

        sources = [s.data for s in sources] if sources is not None else []
        for s in sources:
            s.distribute_exponent()
        # print('mps sources', [helper.norm(s) for s in sources])

        # solver_type = LocalSolverType.TDCross

        if solver_type in [LocalSolverType.TDDMRG, LocalSolverType.DMRG]:

            solver_alg = '1step'

            if solver_alg == 'tddmrg':
                from local_solvers.time_integrator_2 import TDDMRG_Global as TimeInteg
                dt_ = dt
                if max_bond is not None:
                    max_bond = max_bond * 4
            elif solver_alg == 'tdvp':
                from local_solvers.time_integrator_2 import TDVP_Global as TimeInteg
                dt_ = dt
            elif solver_alg == '1step':
                from local_solvers.time_integrator_2 import TDLocal_Global as TimeInteg
                dt_ = dt
            else:
                raise NotImplementedError

            solver = TimeInteg(dist_mpx, [mpo.data for mpo in linear_mpo_list],
                               sources=sources, nonlinear_terms=nonlinear_terms,
                               te_order_target=te_order, te_order_final=te_order,
                               max_bond=max_bond, dt=dt_, time=time,
                               max_tot_iter=2,
                               # direction = 1,  # -1,
                               grid=self.grid, ax_deriv_configs=self.ax_deriv_configs)

            print('local tddmrg', solver.dt)

            if solver_alg == 'tddmrg':
                print('global tddmrg w/ post compress')
                solver.solve_l2r(1, canonize=True, filter_bases=filter_bases)
                gtn.data = solver.out
                ## in left canonical form if l2r
                target_canon_dir = 'right' if solver.direction > 0 else 'left'

            elif solver_alg == 'tdvp':
                print('global tdvp w/ post compress')
                solver.solve_l2r(1, canonize=True, filter_bases=filter_bases)

                # ## compress and reinitialized --> might as well do that separately
                # solver.direction = solver.direction * -1
                # solver.update_ket_from_out()
                # solver.add_global_projector_to_ket(direction=solver.direction )
                # solver.initialize_terms(solver.ket)
                # solver.solve_r2l(1, canonize=False, filter_bases=filter_bases)

                gtn.data = solver.out

                ## in left canonical form if l2r
                target_canon_dir = 'right' if solver.direction > 0 else 'left'

            elif solver_alg == '1step':
                print('global 1step w/ post compress')
                solver.solve_l2r(1, canonize=False, filter_bases=filter_bases)
                # solver.solve_r2l(1, canonize=False, filter_bases=filter_bases)
                gtn.data = solver.out
                # print('gtn.data', gtn.data.cur_orthog)

                ## in right canonical if "l2r"
                target_canon_dir = 'left' if solver.direction > 0 else 'right'

            else:
                raise NotImplementedError

            compress_opts['form'] = target_canon_dir
            gtn.compress(inplace=True, canonize=False, compress_opts=compress_opts, conservative=True)

            # solver.solve(1, canonize=True, filter_bases=filter_bases)
            # # solver.update_ket_from_out()
            # # solver.solve(1, canonize=True, filter_bases=filter_bases)
            # gtn.data = solver.out
        else:
            raise NotImplementedError
        return gtn



    # def evolve_tdmrg_mdpt(self, dt, linear_mpo_list, te_order=0, do_adapt=True, inplace=False,
    #                       compress_config: CompressionConfiguration = None, nonlinear_terms=None, sources=None,
    #                       solver_type=LocalSolverType.TDDMRG, filter_bases=False):
    #
    #     max_bond = None
    #
    #     print('tdmrg 1D new', te_order)
    #     gtn = self if inplace else self.copy()
    #     dist_mpx = gtn.data.copy()
    #
    #     dist_mpx.distribute_exponent()
    #     for mpo in linear_mpo_list:
    #         mpo.data.distribute_exponent()
    #
    #     sources = [s.data for s in sources] if sources is not None else None
    #     # for s in sources:
    #     #     s.distribute_exponent()
    #     # print('mps srouces', [helper.norm(s) for s in sources])
    #
    #     # solver_type = LocalSolverType.TDCross
    #
    #     if solver_type in [LocalSolverType.TDDMRG, LocalSolverType.DMRG]:
    #         from local_solvers.time_integrator import TDDMRG as TimeInteg
    #
    #         solver = TimeInteg(dist_mpx, [mpo.data for mpo in linear_mpo_list],
    #                            sources=sources, nonlinear_terms=nonlinear_terms,
    #                            te_order_target=te_order, te_order_final=te_order,
    #                            max_bond=max_bond, dt=dt/2,
    #                            # max_bond=max_bond, dt=dt,
    #                            grid=self.grid, ax_deriv_configs=self.ax_deriv_configs)
    #
    #         print('local tddmrg', solver.dt)
    #         solver.solve_l2r(1, canonize=True, filter_bases=filter_bases)
    #         solver.direction = solver.direction * -1
    #         solver.update_ket_from_out()
    #         solver.solve_r2l(1, canonize=False, filter_bases=filter_bases)
    #         gtn.data = solver.out
    #
    #         # solver.solve(1, canonize=True, filter_bases=filter_bases)
    #         # # solver.update_ket_from_out()
    #         # # solver.solve(1, canonize=True, filter_bases=filter_bases)
    #         # gtn.data = solver.out
    #     else:
    #
    #         # from local_solvers.time_integrator_cross import global_rk4_cross
    #         #
    #         # nsites = 1
    #         # out = global_rk4_cross(dt / 2, dist_mpx.copy(), [mpo.data for mpo in linear_mpo_list],
    #         #                        sources=sources, nonlinear_terms=nonlinear_terms,
    #         #                        nsites = nsites, max_bond=max_bond
    #         #                        )
    #         #
    #         # out = global_rk4_cross(dt / 2, out.copy(), [mpo.data for mpo in linear_mpo_list],
    #         #                        sources=sources, nonlinear_terms=nonlinear_terms,
    #         #                        nsites=nsites, max_bond=max_bond
    #         #                        )
    #         # gtn.data = out
    #
    #         from local_solvers.time_integrator_cross import TDCross as TimeInteg
    #
    #         nsites = 1  # 1  # 2 if dist_mpx.max_bond() < max_bond else 1
    #
    #         ## to do second order time step, would need to reinitialize...
    #         solver = TimeInteg(dist_mpx.copy(), [mpo.data for mpo in linear_mpo_list],
    #                            sources=sources, nonlinear_terms=nonlinear_terms,
    #                            max_bond=max_bond, dt=dt / 2)
    #         solver.solve_l2r(nsites, canonize=True)
    #         out = solver.solution
    #         # print('solver terms', solver.terms[0].ket is solver.terms[1].ket)
    #
    #         # gtn.data = solver.terms[1].op_blocks[0][0].ket
    #         # plt.figure()
    #         # plt.imshow(gtn.get_data())
    #         # plt.title('ket')
    #         #
    #         # gtn.data = solver.solution
    #         # plt.figure()
    #         # plt.imshow(gtn.get_data())
    #         # plt.title('soln')
    #         # plt.show()
    #
    #         solver = TimeInteg(out.copy(), [mpo.data for mpo in linear_mpo_list],
    #                            sources=sources, nonlinear_terms=solver.nonlinear_terms,
    #                            max_bond=max_bond, dt=dt / 2, direction=-1)
    #         solver.solve_r2l(nsites, canonize=False)
    #
    #         # plt.figure()
    #         # plt.imshow(gtn.get_data())
    #         # plt.title('ket')
    #         #
    #         # gtn.data = solver.solution
    #         # plt.figure()
    #         # plt.imshow(gtn.get_data())
    #         # plt.title('soln')
    #         # plt.show()
    #
    #         gtn.data = solver.solution
    #
    #     return gtn
    #
    # def evolve_tdvp0(self, dt, dfdt_list, te_order=0, do_adapt=True, inplace=False,
    #                  compress_config=None):
    #     gtn = self if inplace else self.copy()
    #     dist_mpx = gtn.data
    #     tdvp_solver = TDVPSolver0(dist_mpx, targets=[mps.data for mps in dfdt_list], te_order=te_order,
    #                               compress_config=compress_config)
    #     tdvp_solver.take_time_step(dt, do_adapt=do_adapt)
    #     gtn.data = tdvp_solver.ket  ## not actually an inplace operation...
    #     return gtn
