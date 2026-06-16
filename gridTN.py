"""GridTN: base tensor-network state paired with a grid.

Parent class for tensor-network data (MPS / MPO) associated with a :class:`~grid.Grid`.
Concrete subclasses are :class:`~gridTN_1D.GridTN1D` (1-D QTT / MPS / MPO) and the
composite comb-layout variants in :mod:`gridTN_composite` / :mod:`gridTN_1Dcomb`.
Provides the static creation helpers and the common tensor-network interface used by
:mod:`field` and the PDE solvers.
"""

import tt_io

import helper_dmrg
from setup_.configs import *
import helper_quimb as helper
from helper_tdvp import TDVPSolver

from basis.basis_spatial import SpatialBasis
from basis.basis_k import FourierBasis

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import quimb.tensor as qtn
    from axis import Axis
    from coord.coord_sys import Coordinate
    from coord.coord_sys import CoordinateSystem
    from grid import Grid
    from grids_composite import CompositeGrid

    GridType = Union[Grid, CompositeGrid, 'Grid1D', 'GridsComb']  # tuple[Grid]

""" object with MPS/MPO as data, plus the GridLayout object (the grid) it's associated with
"""


########################################
## static methods to create GridTN1Ds ##
########################################


####################
## GridTN1D class ##
####################

class GridTN:

    def __init__(self, grid: 'GridType', data: list[dict['Axis', 'TNType']] = None,
                 ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None):
        """Initialize a GridTN by binding tensor-network data to a grid.

        Parameters
        ----------
        grid : GridType
            Grid (or composite grid) on which the tensor network lives.
        data : list of dict[Axis, TNType], optional
            Tensor-network payload in a format compatible with ``grid``; a list
            of dicts keyed by the grid's axis IDs. Defaults to None (empty).
        ax_deriv_configs : dict[Axis, DerivativeConfiguration], optional
            Per-Axis finite-difference configuration. If None, a default
            :class:`DerivativeConfiguration` is created for each grid axis.
        """

        self.grid = grid
        self._data_type = None
        self.data = data  # list of dicts with grid.axIDs as keys
        # self.is_sqrt = False
        self.info = {}  # solver instrumentation (e.g. 'internal_rank', 'num_evals')

        if ax_deriv_configs is None:
            self.ax_deriv_configs = {ax: DerivativeConfiguration() for ax in self.grid.axes}
        else:
            self.ax_deriv_configs = ax_deriv_configs

        self.canon_site = None  # None: not canonical; otherwise indexes site canonicalized around
        self.is_constant = False
        self.constant_axes = []

    def __repr__(self):
        """Return a string representation showing the class and stored data."""
        return str(self.__class__) + ' ' + str(self.data)

    def __getitem__(self, item):
        """Index into the underlying tensor network.

        Parameters
        ----------
        item : object
            Index or key selecting a tensor / sub-network.

        Returns
        -------
        object
            The selected tensor-network element.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    @property
    def exponent(self):
        """Base-10 log magnitude carried on the underlying tensor network."""
        return self.data.exponent

    def mangle_inner(self, inplace=True):
        """Rename inner (virtual) bond indices to avoid collisions.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else return a copy.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def _get_data(self) -> 'TNType':
        """Return the underlying tensor-network data object (``data`` getter)."""
        return self._data

    def _set_data(self, data):
        """Set the underlying tensor-network data (``data`` setter).

        Infers and stores ``data_type`` from the first entry and resets the
        constant-axes bookkeeping.

        Parameters
        ----------
        data : TNType or None
            New tensor-network payload, or None to clear it.
        """
        self._data = data
        if data is not None:
            data_ = data[0]
            self.data_type = type(data_[next(iter(data_))])
        self.constant_axes = []
        self.is_constant = False

    data = property(fget=_get_data, fset=_set_data)

    def _get_data_type(self):
        """Return the cached :class:`DataType` of the stored data (``data_type`` getter)."""
        return self._data_type

    def _set_data_type(self, data_class):
        """Map a data class to its :class:`DataType` and store it (``data_type`` setter).

        Parameters
        ----------
        data_class : type
            Class of the stored payload (e.g. ``qtn.MatrixProductState``,
            ``qtn.MatrixProductOperator``, ``MatrixProductTensor``, or a numeric
            type).

        Raises
        ------
        TypeError
            If ``data_class`` is not an MPS, MPO, TN1, MPT, or numeric type.
        """
        if data_class == qtn.MatrixProductState or issubclass(data_class, qtn.MatrixProductState):
            self._data_type = DataType.MPS
        elif data_class == qtn.MatrixProductOperator:
            self._data_type = DataType.MPO
        elif data_class == qtn.TensorNetwork1D:
            self._data_type = DataType.MPX
        elif data_class == MatrixProductTensor:
            self._data_type = DataType.TN3
        elif issubclass(data_class,(float,complex)):
            self._data_type = DataType.Num
        else:
            print(data_class)
            raise TypeError('GridTN data must be MPS, MPO, TN1, or MPT or Number')

    data_type = property(fget=_get_data_type, fset=_set_data_type)

    def update_deriv_params(self, ax, left_bc: Optional[BCType or str] = None,
                            right_bc: Optional[BCType or str] = None, order: Optional[int] = None,
                            fd_type: Optional[FDType or str] = None, offset: Optional[int] = None):
        """Update the finite-difference configuration for one axis in place.

        Parameters
        ----------
        ax : Axis
            Axis whose :class:`DerivativeConfiguration` is updated.
        left_bc : BCType or str, optional
            Boundary condition at the left/lower edge.
        right_bc : BCType or str, optional
            Boundary condition at the right/upper edge.
        order : int, optional
            Finite-difference expansion order.
        fd_type : FDType or str, optional
            Finite-difference stencil type (e.g. centered, upwind).
        offset : int, optional
            Grid offset (in grid-point increments) for the stencil.
        """
        dp = self.ax_deriv_configs[ax]
        dp.update(left_bc, right_bc, order, fd_type, offset)  # , v_ax)

    def create_like(self, new_data=None) -> 'GridTN':
        """Create a new GridTN sharing this object's grid and configuration.

        Parameters
        ----------
        new_data : TNType, optional
            Data payload for the new GridTN. Defaults to None (empty).

        Returns
        -------
        GridTN
            A new field on the same grid wrapping ``new_data``.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def copy(self, deep=True) -> 'GridTN':
        """Return a copy of this GridTN.

        Parameters
        ----------
        deep : bool
            Copy the underlying tensor data if True, else share references.

        Returns
        -------
        GridTN
            The copied field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def conj(self, inplace=False, mangle_inner=False) -> 'GridTN':
        """Return the complex conjugate of the field.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else return a copy.
        mangle_inner : bool
            Rename inner (virtual) bond indices to avoid collisions if True.

        Returns
        -------
        GridTN
            The conjugated field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def save_data(self, fstr):
        """Save the underlying data to ``<fstr>.npz`` as numpy arrays (no pickle).

        Parameters
        ----------
        fstr : str
            Path prefix; the ``.npz`` extension is appended.
        """
        tt_io.tn1d_to_npz(self.data, fstr)
        print('saved data', fstr + '.npz')

    def reload_data(self, fstr, ax_deriv_configs=None):
        """Reload saved data into this GridTN via the grid loader.

        Parameters
        ----------
        fstr : str
            Path prefix of the ``.npz`` file to load.
        ax_deriv_configs : dict[Axis, DerivativeConfiguration], optional
            Per-Axis finite-difference configs to attach to the loaded field.

        Returns
        -------
        GridTN
            This field with ``self.data`` set from the loaded payload.
        """
        grid = self.grid
        return grid.load_gtn_data(fstr, self, ax_deriv_configs=ax_deriv_configs)

    @classmethod
    def load_data(cls, grid: 'Grid', fstr, ax_deriv_configs=None):
        """Construct a new GridTN from saved data on a given grid.

        Parameters
        ----------
        grid : Grid
            Grid on which to build the loaded field.
        fstr : str
            Path prefix of the ``.npz`` file to load.
        ax_deriv_configs : dict[Axis, DerivativeConfiguration], optional
            Per-Axis finite-difference configs to attach to the new field.

        Returns
        -------
        GridTN
            A new field with data loaded from ``fstr``.
        """
        gtn = cls(grid, data=None, ax_deriv_configs=ax_deriv_configs)
        return grid.load_gtn_data(fstr, gtn)

    def get_TN(self):
        """Return the underlying quimb tensor network.

        Returns
        -------
        qtn.TensorNetwork
            The wrapped tensor network.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def norm(self, is_sqrt=False) -> Numeric:
        """Compute the integral norm of the field over the grid.

        Integrates the field (or its square) over the grid and discards a
        negligible imaginary part. With ``is_sqrt`` the square root of the
        integral is returned.

        Parameters
        ----------
        is_sqrt : bool
            Treat the field as a square-root representation; integrate its
            square and return the square root of the result.

        Returns
        -------
        Numeric
            The norm, or None if the field has no data.
        """
        if self.data is not None:
            if self.data_type is DataType.Num:
                return self.data
            out = self.integrate(is_sqrt=is_sqrt)
            if np.isreal(out) or np.abs(np.imag(out)) < 1.0e-13:
                out = np.real(out)
            if is_sqrt:
                out = np.sqrt(out)
            return out

    def frobenius_norm(self) -> Numeric:
        """Compute the Frobenius norm sqrt(<self|self>) of the field.

        Returns
        -------
        Numeric
            The Frobenius norm, or 0.0 if the field has no data.
        """
        if self.data is not None:
            # out = helper.ovlp(self.data, self.data.conj())
            self_conj = self.conj(mangle_inner=True, inplace=False)
            out = self.ovlp(self_conj)
            out = np.sqrt(np.abs(out))
            # print('fro norm', out) # , self.data.norm() * 10**self.exponent)
            return out
        else:
            return 0.0

    def normalize(self, target_value, is_sqrt=False, inplace=False) -> 'GridTN':
        """Rescale the field so its norm equals a target value.

        For a spatial basis this is a plain scalar multiplication; for a Fourier
        basis only the k=0 term would be rescaled (the spatial branch is taken
        unconditionally in the current code).

        Parameters
        ----------
        target_value : Numeric
            Desired norm after rescaling.
        is_sqrt : bool
            Treat the field as a square-root representation when computing the
            current norm.
        inplace : bool
            Mutate self in place if True, else return a copy.

        Returns
        -------
        GridTN
            The rescaled field.
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)  # return None?

        is_spatial = [ax.basis.type == BasisType.SPATIAL for ax in self.grid.axes]
        is_fourier = [ax.basis.type == BasisType.FOURIER for ax in self.grid.axes]

        self_norm = self.norm(is_sqrt=is_sqrt)
        scale_val = target_value / np.conj(self_norm)
        if np.isnan(scale_val) and np.abs(self_norm) < 1.0e-12:
            scale_val = 0.0

        if True:  # np.all(is_spatial):
            out = self.scalar_multiply(scale_val, inplace=inplace)
        elif np.any(is_fourier):
            # only adjust k=0 term which may be normalized to 0
            axes_k = [self.grid.axes[i] for i in range(len(is_fourier)) if is_fourier[i]]
            select_dict = {ax: ax.get_select_elems_mpo([ax.zero_ind]) for ax in axes_k}
            select_zero_gtn = self.grid.make_mpo_ndim(select_dict)
            select_zero_gtn.scalar_multiply(scale_val - 1, inplace=True)
            iden_gtn = self.grid.get_iden_mpo()
            scale_gtn = iden_gtn.add(select_zero_gtn)

            out = self.apply(scale_gtn, inplace=inplace, compress=True)

        else:
            raise NotImplementedError

        return out

    def ovlp(self, other) -> Numeric:
        """Compute the overlap (inner product) <self|other>.

        Parameters
        ----------
        other : GridTN
            Field to contract against self.

        Returns
        -------
        Numeric
            The overlap value.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def distance(self, other) -> Numeric:
        """Compute the Frobenius distance ||self - other||.

        Forms the difference (without compression) and returns its Frobenius
        norm.

        Parameters
        ----------
        other : GridTN
            Field to subtract from self.

        Returns
        -------
        Numeric
            The Frobenius norm of the difference.
        """

        # self_conj = self.conj(mangle_inner=True, inplace=False)
        # norm2a = self.ovlp(self_conj)
        # norm2b = other.ovlp(other.conj(mangle_inner=True, inplace=False))
        # # ovlp_ab = self.ovlp(other.conj(inplace=False))
        # ovlp_ab = other.ovlp(self_conj)
        # dist = norm2a + norm2b - 2 * np.real(ovlp_ab)
        #
        # if dist < 0:
        #     print('neg dist', dist)
        # # if dist < 0 and np.abs(dist) < 1.0e-13:
        # #     print('small neg dist', dist)
        # #     dist = np.abs(dist)
        # dist1 = np.sqrt(dist)

        other = other.scalar_multiply(-1, inplace=False)
        diff = self.add(other, inplace=False, compress=False)
        dist = diff.frobenius_norm()

        # print('distances', dist, dist1)
        return dist

    def max_bond(self) -> int:
        """Return the largest virtual bond dimension of the tensor network.

        Returns
        -------
        int
            The maximum bond dimension.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def mem_size(self) -> Numeric:
        """Return the memory footprint of the tensor network in kilobytes.

        Returns
        -------
        Numeric
            Approximate size in KB.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def num_elem(self) -> Numeric:
        """Return the raw count of stored tensor elements.

        Returns
        -------
        Numeric
            Number of elements

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``. T
        """
        raise NotImplementedError

    def all_virtual_sizes(self) -> int:
        """Return the sizes of all virtual (bond) indices.

        Returns
        -------
        int
            The collection of virtual bond dimensions.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def all_smallest_singular_values(self, max_bond=None) -> int:
        """Return the smallest singular value at each bond.

        Parameters
        ----------
        max_bond : int, optional
            Bond-dimension limit used when evaluating the singular values.

        Returns
        -------
        int
            The smallest singular value per bond.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def entanglement_entropy_all(self) -> list[Numeric]:
        """Return the entanglement entropy across every bond.

        Returns
        -------
        list of Numeric
            Entanglement entropy at each cut.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def check_orthog(self):
        """Check the canonical/orthogonality structure of the tensor network.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def get_anchor_ind(self) -> int:
        """Return the index of the "lead" tensor.

        The lead tensor is used for connecting this GridTN to other GridTNs in
        composite grid systems.

        Returns
        -------
        int
            Index of the anchor tensor.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def get_anchor_tens(self) -> qtn.Tensor:
        """Return the "lead" tensor used to connect GTNs in composite grids.

        Returns
        -------
        qtn.Tensor
            The anchor tensor.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def transpose(self, inplace=True, mangle_inner=False):
        """Transpose the MPO by swapping its upper and lower physical indices.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else return a copy.
        mangle_inner : bool
            Rename inner (virtual) bond indices to avoid collisions if True.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def get_like_iden(self):
        """Return an identity MPO matching this field's shape and grid.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    #######################################
    ## class methods to build common TNs ##
    #######################################

    #### move to grid_* classes
    @classmethod
    def get_ones_mps(cls, grid: 'GridType', site_ind_id='i({})', site_tag_id='X({})') -> 'GriDTN':
        """Build an all-ones vector MPS on the specified grid.

        Parameters
        ----------
        grid : GridType
            Grid on which to build the MPS.
        site_ind_id : str
            Format string for physical (site) index labels.
        site_tag_id : str
            Format string for per-site tensor tags.

        Returns
        -------
        GridTN
            The ones-vector field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    @classmethod
    def get_iden_mpo(cls, grid: 'GridType', upper_ind_id='i({})', lower_ind_id='o({})',
                     site_tag_id='X({})') -> 'GridTN':
        """Build an identity MPO on the specified grid.

        Parameters
        ----------
        grid : GridType
            Grid on which to build the MPO.
        upper_ind_id : str
            Format string for upper physical index labels.
        lower_ind_id : str
            Format string for lower physical index labels.
        site_tag_id : str
            Format string for per-site tensor tags.

        Returns
        -------
        GridTN
            The identity-operator field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    @classmethod
    def get_select_elem_mps(cls, grid: 'GridType', inds, site_ind_id='i({})', site_tag_id='X({})') -> 'GridTN':
        """Build an MPS that selects the grid elements given by ``inds``.

        Parameters
        ----------
        grid : GridType
            Grid on which to build the selector.
        inds : Sequence[int]
            Element indices to select.
        site_ind_id : str
            Format string for physical (site) index labels.
        site_tag_id : str
            Format string for per-site tensor tags.

        Returns
        -------
        GridTN
            The selector field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    @classmethod
    def get_select_elem_mpo(cls, grid: 'GridType', inds, upper_ind_id='i({})', lower_ind_id='o({})',
                            site_tag_id='X({})') -> 'GridTN':
        """Build an MPO that selects the grid elements given by ``inds``.

        Parameters
        ----------
        grid : GridType
            Grid on which to build the selector.
        inds : Sequence[int]
            Element indices to select.
        upper_ind_id : str
            Format string for upper physical index labels.
        lower_ind_id : str
            Format string for lower physical index labels.
        site_tag_id : str
            Format string for per-site tensor tags.

        Returns
        -------
        GridTN
            The selector-operator field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    @classmethod
    def from_dense_state(cls, data: np.ndarray, grid: 'GridType', site_ind_id='i({})',
                         site_tag_id='T({})', split_opts=None, axes=None) -> 'GridTN':
        """Build an MPS field by decomposing a dense state array.

        Parameters
        ----------
        data : np.ndarray
            Dense state values to factor into an MPS.
        grid : GridType
            Grid on which to build the field.
        site_ind_id : str
            Format string for physical (site) index labels.
        site_tag_id : str
            Format string for per-site tensor tags.
        split_opts : dict, optional
            Options forwarded to the tensor-splitting routine.
        axes : Sequence[Axis], optional
            Axis ordering / subset to decompose along.

        Returns
        -------
        GridTN
            The MPS field representing ``data``.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    @classmethod
    def from_dense_operator(cls, data: np.ndarray, grid: 'GridType', upper_ind_id='o({})',
                            lower_ind_id='i({})', site_tag_id='T({})', split_opts=None, **kwargs) -> 'GridTN':
        """Build an MPO field by decomposing a dense operator array.

        Parameters
        ----------
        data : np.ndarray
            Dense operator values to factor into an MPO.
        grid : GridType
            Grid on which to build the field.
        upper_ind_id : str
            Format string for upper physical index labels.
        lower_ind_id : str
            Format string for lower physical index labels.
        site_tag_id : str
            Format string for per-site tensor tags.
        split_opts : dict, optional
            Options forwarded to the tensor-splitting routine.
        **kwargs
            Additional options forwarded to the construction routine.

        Returns
        -------
        GridTN
            The MPO field representing ``data``.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    ####

    def get_data(self, ax_order: Sequence['Axis'] = None, ax_select: dict['Axis', int] = None,
                 pad_data=False) -> np.ndarray:
        """Contract the tensor network into a dense array.

        Parameters
        ----------
        ax_order : Sequence[Axis], optional
            Desired axis ordering of the output array.
        ax_select : dict[Axis, int], optional
            Per-axis index to fix, slicing the output along those axes.
        pad_data : bool
            Pad the dense output to the full grid extent if True.

        Returns
        -------
        np.ndarray
            The dense array of field values.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def get_squeezed_data(self) -> tuple[np.ndarray, 'Grid']:
        """Return dense data with constant axes collapsed, plus the reduced grid.

        Fixes each constant axis at its constant index and drops it from the
        grid.

        Returns
        -------
        tuple[np.ndarray, Grid]
            The squeezed dense array and the subgrid excluding constant axes.
        """
        data = self.get_data(ax_select={ax: ax.get_constant_ind() for ax in self.constant_axes})
        data_gr = self.grid.get_subgrid([ax for ax in self.grid.axes if ax not in self.constant_axes])
        return data, data_gr

    #######################
    ## MPS/MPX functions ##
    #######################

    def apply(self, other: 'GridTN', inplace=False, zipup=False, use_mg=False, compress=False, compress_opts=None,
              add_cc=False, **kwargs) -> 'GridTN':
        """Apply an MPO ``other`` to self (both on the same grid).

        Parameters
        ----------
        other : GridTN
            Operator (MPO) field to apply, living on the same grid as self.
        inplace : bool
            Mutate self in place if True, else return a copy.
        zipup : bool
            Use the zip-up MPO-MPS apply algorithm if True.
        use_mg : bool
            Use a multigrid contraction strategy if True.
        compress : int
            Compression-aggressiveness level; selects compression parameters
            from the compression level.
        compress_opts : dict, optional
            Overrides for the compression parameters chosen by ``compress``.
        add_cc : bool
            Add the complex conjugate of the result if True.
        **kwargs
            Additional options forwarded to the apply routine.

        Returns
        -------
        GridTN
            The resulting field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def apply_rdm(self, other, bra_self=None, bra_other=None, left_env=None, right_env=None,
                  direction=1, inplace=False, compress_opts=None, **kwargs) -> 'GridTN1D':
        """Apply an MPO to self and contract/compress via the RDM method.

        Parameters
        ----------
        other : GridTN
            Operator (MPO) field to apply, living on the same grid as self.
        bra_self : GridTN, optional
            Bra used to form the reduced density matrix for self.
        bra_other : GridTN, optional
            Bra used to form the reduced density matrix for ``other``.
        left_env : qtn.Tensor, optional
            Precomputed left environment tensor.
        right_env : qtn.Tensor, optional
            Precomputed right environment tensor.
        direction : int
            Sweep direction (1 left-to-right, -1 right-to-left).
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress_opts : dict, optional
            Compression parameters for the RDM-based truncation.
        **kwargs
            Additional options forwarded to the apply routine.

        Returns
        -------
        GridTN1D
            The resulting field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def solve(self, operator: 'GridTN', compress_type: 'CompressType', inplace=False, use_A2=True, compress_opts=None,
              is_H=False, init_guess: 'GridTN' = None, verbose_output=False, **kwargs
              ) -> Union[tuple['GridTN', float, bool], 'GridTN']:
        """Solve the linear system ``operator @ x = self`` for x.

        Here self plays the role of the right-hand side b and ``operator`` is A;
        the returned field is the solution x.

        Parameters
        ----------
        operator : GridTN
            The system matrix A (MPO field).
        compress_type : CompressType
            Compression scheme used during the local solve.
        inplace : bool
            Mutate self in place if True, else return a copy.
        use_A2 : bool
            Solve the normal equations using A^2 (A^dagger A) if True.
        compress_opts : dict, optional
            Compression parameters for the solve.
        is_H : bool
            Treat A as Hermitian if True.
        init_guess : GridTN, optional
            Initial guess for the solution.
        verbose_output : bool
            Also return the residual and a convergence flag if True.
        **kwargs
            Additional options forwarded to the local solver.

        Returns
        -------
        GridTN or tuple[GridTN, float, bool]
            The solution x, or (x, residual, converged) if ``verbose_output``.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def add(self, gtn_mpx2: 'GridTN', zipup=False, inplace=False, compress_type=CompressType.SVD,
            compress=False, compress_opts=None, **kwargs) -> 'GridTN':
        """Add another field of the same type and grid to self.

        Parameters
        ----------
        gtn_mpx2 : GridTN
            Field to add; must be the same data type and grid as self.
        zipup : bool
            Accepted for signature compatibility but ignored.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress_type : CompressType
            Accepted for signature compatibility but ignored.
        compress : int
            Accepted for signature compatibility but ignored.
        compress_opts : dict, optional
            Accepted for signature compatibility but ignored.
        **kwargs
            Accepted for signature compatibility but ignored.

        Returns
        -------
        GridTN
            The summed field.

        Notes
        -----
        The ``zipup``/``compress``/``compress_opts`` arguments are ignored: this
        implementation performs a plain ``data += data`` addition with no
        compression.
        """
        gtn = self if inplace else self.copy()
        if gtn.data is None:
            gtn.data = gtn_mpx2.data
        else:
            gtn.data += gtn_mpx2.data
        return gtn

    def add_list(self, *other_gtns: 'GridTN', compress_opts=None) -> 'GridTN':
        """Add several fields to self via the batched MPS-list summation helper.

        Parameters
        ----------
        *other_gtns : GridTN
            Fields to add to self.
        compress_opts : dict, optional
            Compression parameters; ``max_bond`` sets the bond-dimension limit.
            ``cutoff`` sets the cutoff threshold limit.

        Returns
        -------
        GridTN
            The summed and compressed field.
        """

        max_bond = compress_opts.get('max_bond', None) if compress_opts is not None else None
        cutoff = compress_opts.get('cutoff', CUTOFF) if compress_opts is not None else CUTOFF

        gtn = self.copy()
        is_mps = self.data_type == DataType.MPS

        ix = 0
        if gtn.data is None:
            gtn_iter = iter(other_gtns)
            while gtn.data is None:
                gtn = next(gtn_iter)
                ix += 1

        self_tt = self.grid.gtn_to_dmrg_format(gtn, is_mps=is_mps)
        other_gtns = other_gtns[ix:]
        other_tts = [self.grid.gtn_to_dmrg_format(ogtn) for ogtn in other_gtns]

        out = helper.add_MPS_list([self_tt, *other_tts], compress_opts={'max_bond': max_bond, 'cutoff': cutoff})
        gtn = self.grid.dmrg_to_gtn_format(gtn, out, is_mps=is_mps)
        return gtn


    def add_dmrg(self, *other_gtns: 'GridTN', inplace=False, compress_opts=None, **dmrg_opts) -> 'GridTN':
        """Add multiple fields together using the DMRG fitting solver.

        Each summand is converted into a DMRG term and fit to a single MPS by
        the local DMRG evaluator.

        Parameters
        ----------
        *other_gtns : GridTN
            Fields to add to self.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress_opts : dict, optional
            Compression parameters; ``max_bond`` sets the DMRG bond-dimension
            limit.
        **dmrg_opts
            Additional options forwarded to ``local_dmrg_evaluator``.

        Returns
        -------
        GridTN
            The summed field.
        """
        from local_solvers.local_dmrg_eval import local_dmrg_evaluator, Term_DMRG

        DMAX = compress_opts.get('max_bond', None) if compress_opts is not None else None
        print('add drmg compress_opts', compress_opts)
        # exit()

        gtn = self if inplace else self.copy()
        is_mps = self.data_type == DataType.MPS

        self_tt = self.grid.gtn_to_dmrg_format(gtn, is_mps=is_mps)
        if self_tt is None:
            print('gtn', gtn)
            gtn_iter = iter(other_gtns)
            while gtn.data is None:
                gtn = next(gtn_iter)

        terms = [Term_DMRG(self_tt.copy())] if self_tt is not None else []
        for ogtn in other_gtns:
            op_mps = self.grid.gtn_to_dmrg_format(ogtn, is_mps=is_mps)
            if op_mps is not None:
                terms += [Term_DMRG(op_mps.copy())]

        print('add func max bond?', DMAX)
        print('term bonds')
        for term in terms:
            print('term.ket', term.ket.max_bond())

        func_mps = local_dmrg_evaluator(terms, max_bond=DMAX, **dmrg_opts)

        gtn = self.grid.dmrg_to_gtn_format(gtn, func_mps, is_mps=is_mps)
        return gtn


    def add_subgtn(self, sub_gtn_mpx2: 'GridTN', open_bc=False, inplace=False, zipup=False, compress=False,
                   compress_opts=None, **kwargs) -> 'GridTN':
        """Add a field defined on a subgrid into self.

        Parameters
        ----------
        sub_gtn_mpx2 : GridTN
            Field on a subset of self's axes to be embedded and added.
        open_bc : bool
            Treat the embedding bonds as open (no wrap) if True.
        inplace : bool
            Mutate self in place if True, else return a copy.
        zipup : bool
            Use the zip-up MPO-MPS apply algorithm if True.
        compress : int
            Compression-aggressiveness level applied after the operation.
        compress_opts : dict, optional
            Overrides for the compression parameters.
        **kwargs
            Additional options forwarded to the addition routine.

        Returns
        -------
        GridTN
            The combined field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def sum_apply(self, mpos: Sequence['GridTN'], inplace=False, zipup=False, compress=False, compress_opts=None):
        """Apply each MPO to self and sum the results.

        Parameters
        ----------
        mpos : Sequence[GridTN]
            Operator (MPO) fields to apply to self.
        inplace : bool
            Mutate self in place if True, else return a copy.
        zipup : bool
            Use the zip-up MPO-MPS apply algorithm for each application if True.
        compress : int
            Compression-aggressiveness level applied to the final sum.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The summed field.
        """
        new_gtn = self if inplace else self.copy()

        out = None
        for mpo in mpos:
            comp = self.apply(mpo, inplace=False, zipup=zipup, compress=False)
            out = comp if out is None else out.add(comp, inplace=True, compress=False)

        if compress:
            out.compress(inplace=True, compress_opts=compress_opts)

        new_gtn.data = out.data
        return new_gtn


    def sum_apply_dmrg(self, mpos: Sequence['GridTN'], inplace=False, compress_opts=None, **dmrg_opts):
        """Apply a list of MPOs to self and sum the results via DMRG fitting.

        Builds a single DMRG term carrying self as the ket and ``mpos`` as
        operators, then fits the sum to an MPS with the local DMRG evaluator.

        Parameters
        ----------
        mpos : Sequence[GridTN]
            Operator (MPO) fields to apply to self.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress_opts : dict, optional
            Compression parameters; ``max_bond`` sets the DMRG bond-dimension
            limit.
        **dmrg_opts
            Additional options forwarded to ``local_dmrg_evaluator``.

        Returns
        -------
        GridTN
            The summed field.
        """
        # new_gtn = self if inplace else self.copy()

        # out = None
        # for mpo in mpos:
        #     comp = self.apply(mpo, inplace=False, zipup=zipup, compress=False)
        #     out = comp if out is None else out.add(comp, inplace=True, compress=False)
        #
        # if compress:
        #     out.compress(inplace=True, compress_opts=compress_opts)
        #
        # new_gtn.data = out.data
        #
        from local_solvers.local_dmrg_eval import local_dmrg_evaluator, Term_DMRG

        DMAX = compress_opts.get('max_bond', None) if compress_opts is not None else None
        print('sum apply drmg compress_opts', compress_opts)

        gtn = self if inplace else self.copy()
        is_mps = self.data_type == DataType.MPS

        self_tt = self.grid.gtn_to_dmrg_format(gtn, is_mps=is_mps)
        ops = [self.grid.gtn_to_dmrg_format(op, is_mps=False) for op in mpos]
        term = Term_DMRG(self_tt.copy(), operators=ops)

        print('sum apply func max bond?', DMAX)

        func_mps = local_dmrg_evaluator([term], max_bond=DMAX, **dmrg_opts)

        gtn = self.grid.dmrg_to_gtn_format(gtn, func_mps, is_mps=is_mps)
        return gtn


    def scalar_multiply(self, scalar_const: Numeric, inplace=False) -> 'GridTN':
        """Multiply the field by a scalar constant.

        Parameters
        ----------
        scalar_const : Numeric
            Scalar to multiply the field by.
        inplace : bool
            Mutate self in place if True, else return a copy.

        Returns
        -------
        GridTN
            The scaled field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    # def elemental_multiply(self, gtn_mps2: 'GridTN', inplace=False, compress=False, compress_opts=None) -> 'GridTN':
    #     raise NotImplementedError

    # @profile
    def elemental_multiply(self, gtn_mps2: 'GridTN', inplace=False, zipup=False, compress_type=CompressType.SVD,
                           compress=False, compress_opts=None, sub_compress_opts=None, add_cc=False) -> 'GridTN':
        """Multiply two fields element-wise (Hadamard product) on the grid.

        Converts one operand into a diagonal MPO and applies it to the other so
        the result is the pointwise product of the two fields. Grids are padded
        to a common set of axes when they differ.

        Parameters
        ----------
        gtn_mps2 : GridTN
            Field to multiply element-wise with self. May also be a raw quimb
            MPS/MPO of matching shape.
        inplace : bool
            Mutate self in place if True, else return a copy.
        zipup : bool
            Use the zip-up MPO-MPS apply algorithm if True.
        compress_type : CompressType
            Compression scheme used during the apply.
        compress : int
            Compression-aggressiveness level applied during the apply.
        compress_opts : dict, optional
            Overrides for the compression parameters.
        sub_compress_opts : dict, optional
            Compression parameters for sub-network (intermediate) truncation.
        add_cc : bool
            Add the complex conjugate of the result if True (default changed
            from True to False).

        Returns
        -------
        GridTN
            The element-wise product field.
        """
        ### changed add_cc default from True to False

        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        if gtn_mps2 is None or gtn_mps2.data is None:
            if inplace:
                self.data = None
                return self
            else:
                return self.create_like(new_data=None)

        # # print(self.data, gtn_mps2.data)
        # if self.data_type is DataType.Num or gtn_mps2.data_type is DataType.Num:
        #     if self.data_type is DataType.Num and gtn_mps2.data_type is DataType.Num:
        #         new_data = self.data * gtn_mps2.data
        #         ref = self
        #     elif self.data_type is DataType.Num:
        #         new_data = helper.scalar_multiply(gtn_mps2.data, self.data)
        #         ref = gtn_mps2.grid
        #         if inplace:
        #             self.grid = gtn_mps2.grid
        #     elif gtn_mps2.data_type is DataType.Num:
        #         new_data = helper.scalar_multiply(self.data, gtn_mps2.data)
        #         ref = self
        #     else:
        #         raise ValueError
        #
        #     if inplace:
        #         self.data = new_data
        #         return self
        #     else:
        #         out = ref.create_like(new_data=new_data)
        #         return out

        grid_mpx1 = self if inplace else self.copy()

        if isinstance(gtn_mps2, qtn.MatrixProductState):
            mps2_shape = tuple([gtn_mps2.phys_dim(i) for i in gtn_mps2.L])
            assert (self.grid.shape == mps2_shape), 'mps2 needs to be the same shape as self.grid'
            gtn_mps2 = self.create_like(new_data=gtn_mps2)
            if self.data_type == DataType.MPS:  ## convert gtn_mps2 to MPO
                grid_mpo2 = gtn_mps2.apply_elemental_multiply_op()
            else:  ## use fact that self is MPO. apply method should take care of it
                grid_mpo2 = grid_mpx1
                grid_mpx1 = gtn_mps2

        elif isinstance(gtn_mps2, qtn.MatrixProductOperator):
            mps2_shape = tuple([gtn_mps2.phys_dim(i) for i in gtn_mps2.L])
            assert (self.grid.shape == mps2_shape), 'mps2 needs to be the same shape as self.grid'
            grid_mpo2 = self.create_like()
            grid_mpo2.data = gtn_mps2

        elif isinstance(gtn_mps2, GridTN):

            # print('gtn_mps2 is GTN', grid_mpx1.grid.axes, gtn_mps2.grid.axes)
            if grid_mpx1.grid != gtn_mps2.grid:
                if set(grid_mpx1.grid.axes).issubset(set(gtn_mps2.grid.axes)):
                    # print('mps1 < mps2')
                    grid_mpx1 = gtn_mps2.grid.pad_gtn_to_grid(grid_mpx1)
                    print('warning: elemental multiply not an inplace operation')
                elif set(gtn_mps2.grid.axes).issubset(set(grid_mpx1.grid.axes)):
                    # print('mps1 > mps2')
                    gtn_mps2 = grid_mpx1.grid.pad_gtn_to_grid(gtn_mps2, target_data_type=DataType.MPO)
                else:
                    raise ValueError(f'grid compatibility issue with {grid_mpx1.grid}, {gtn_mps2.grid}')

            # print('gtn_mps2 data type', gtn_mps2.data_type)
            # print('gtn', gtn_mps2)
            # print(gtn_mps2.__class__)
            if gtn_mps2.data_type == DataType.MPS:
                # assert self.grid == gtn_mps2.grid, 'grid mps2 needs to be the same axes as self'

                # if self.data_type == DataType.MPO:
                #     print('WARNING: ELEMENTAL MULTIPLY CHANGED FOR MPO * MPS')
                #
                # grid_mpo2 = gtn_mps2.apply_elemental_multiply_op()

                gtn_mps2 = gtn_mps2
                if self.data_type == DataType.MPS:  ## convert gtn_mps2 to MPO
                    grid_mpo2: 'GridTN' = gtn_mps2.apply_elemental_multiply_op()
                else:  ## use fact that self is MPO. apply method should take care of it
                    ## this does not accurately capture elemental multiply for arbitrary grid_mpx1 MPOs
                    raise NotImplementedError('maybe elem mult MPO * MPS needs to be redefined')
                    grid_mpo2 = grid_mpx1
                    grid_mpx1 = gtn_mps2

            elif isinstance(gtn_mps2, GridTN) and gtn_mps2.data_type == DataType.MPO:
                # assert self.grid == gtn_mps2.grid, 'grid mps2 needs to be the same axes as self'
                grid_mpo2 = gtn_mps2

        else:
            raise TypeError(f'other must be GridTN with MPS or MPO data type, not {type(gtn_mps2)}')

        # print('elem mult', grid_mpx1.max_bond(), grid_mpo2.max_bond())
        grid_mpx1 = grid_mpx1.apply(grid_mpo2, inplace=True, zipup=zipup, compress_type=compress_type,
                                    compress=compress,
                                    compress_opts=compress_opts, add_cc=add_cc, sub_compress_opts=sub_compress_opts)
        # max_bond = compress_opts.get('max_bond', None)
        # grid_mpx1 = grid_mpx1.apply_rdm(grid_mpo2, inplace=True, compress_opts=compress_opts, compress=(max_bond is None))

        # add_cc used to be True; probably for real Fourier basis??
        return grid_mpx1

    def xmultiply(self, x_axes: Sequence['Axis'], x_power: int = 1, offsets: dict[Any, Numeric] = 0.0,
                  scales: dict[Any, Numeric] = 1.0, inplace=False, zipup=False, compress_type=CompressType.SVD,
                  compress=False, compress_opts=None) -> 'GridTN1D':
        """Multiply the field by powers of the coordinate(s) along given axes.

        Computes ``prod_i (scale_i * x_i + offset_i)**x_power * f`` by applying
        the corresponding coordinate-multiplication MPO.

        Parameters
        ----------
        x_axes : Sequence[Axis]
            Axes whose coordinates multiply the field.
        x_power : int
            Power applied to each coordinate factor.
        offsets : dict or Numeric
            Per-axis additive offset(s) applied to the coordinate.
        scales : dict or Numeric
            Per-axis multiplicative scale(s) applied to the coordinate.
        inplace : bool
            Mutate self in place if True, else return a copy.
        zipup : bool
            Use the zip-up MPO-MPS apply algorithm if True.
        compress_type : CompressType
            Compression scheme used during the apply.
        compress : int
            Compression-aggressiveness level applied during the apply.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN1D
            The coordinate-multiplied field.
        """

        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        grid_mpx1 = self if inplace else self.copy()
        xmult_mpo = grid_mpx1.grid.get_xmultiply_mpo(x_axes, x_power=x_power, offsets=offsets, scales=scales)

        # print('XMULT GTN', compress_type)
        grid_mpx1.apply(xmult_mpo, inplace=True, zipup=zipup, compress_type=compress_type,
                        compress=compress, compress_opts=compress_opts)
        # if compress:
        #     grid_mpx1 = grid_mpx1.compress(inplace=True, compress_opts=compress_opts)

        return grid_mpx1

    def take_qft(self, qft_axes=None, inverse=False, inplace=False, compress=True, compress_opts=None):
        """Apply the quantum Fourier transform along selected axes.

        Parameters
        ----------
        qft_axes : Sequence[Axis], optional
            Axes to transform. Defaults to all grid axes.
        inverse : bool
            Apply the inverse QFT instead of the forward transform if True.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the transform.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The transformed field.
        """
        qft_axes = self.grid.axes if qft_axes is None else qft_axes
        # qft_ops = {ax: ax.get_qft_mpo_v2(inverse=inverse) for ax in qft_axes}
        if inverse:
            qft_ops = {ax: ax.get_inverse_qft_mpo() for ax in qft_axes}
        else:
            qft_ops = {ax: ax.get_qft_mpo() for ax in qft_axes}
        qft_mpo = self.grid.make_mpo_ndim(qft_ops)
        out = self.apply(qft_mpo, inplace=inplace, zipup=True, compress=compress, compress_opts=compress_opts)
        return out

    # def average_fine_scale(self, avg_axes=None, spread:int = 1, inplace=False, compress=True, compress_opts=None, mu=0.5):
    #     """
    #     apply average mpo along each of the axes specified by avg_axes
    #     spread: number of neighbors to include
    #     mu: weight on central element
    #
    #     x_j <-- \sum_i=1^spread (1-mu)/2 spread (x_(j-i) + x_(j+i)) + mu x_(j)
    #     eg if spread = 1:
    #         x_j <-- (1-mu)/2 x_j-1 + mu x_j + (1-mu)/2 x_j+1
    #     """
    #     print('performing fine scale avg')
    #     # raise RuntimeError
    #     avg_axes = self.grid.axes if avg_axes is None else avg_axes
    #     avg_ops = {ax: ax.get_averaging_mpo(spread=spread, mu=mu,
    #                                         boundary_conditions=self.ax_deriv_configs.get(ax, DerivativeConfiguration()))
    #                for ax in avg_axes}
    #     avg_mpo = self.grid.make_mpo_ndim(avg_ops)
    #     out = self.apply(avg_mpo, inplace=inplace, zipup=True, compress=compress, compress_opts=compress_opts)
    #     return out


    def average_fine_scale(self, avg_axes=None, spread:int = 1, inplace=False, compress=True, compress_opts=None, mu=0.5):
        """Apply a fine-scale neighbor-averaging (smoothing) MPO along axes.

        Each point is replaced by a weighted average of itself and its
        neighbors::

            x_j <-- sum_{i=1}^{spread} (1-mu)/(2*spread) (x_{j-i} + x_{j+i}) + mu x_j

        e.g. for ``spread = 1``: ``x_j <-- (1-mu)/2 x_{j-1} + mu x_j + (1-mu)/2 x_{j+1}``.

        Parameters
        ----------
        avg_axes : Sequence[Axis], optional
            Axes to average over. Defaults to all grid axes.
        spread : int
            Number of neighbors on each side to include in the average.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the averaging.
        compress_opts : dict, optional
            Overrides for the compression parameters.
        mu : float
            Weight placed on the central element.

        Returns
        -------
        GridTN
            The smoothed field.
        """
        print('performing fine scale avg')
        # raise RuntimeError
        avg_axes = self.grid.axes if avg_axes is None else avg_axes
        avg_ops = {ax: ax.get_averaging_mpo(spread=spread, mu=mu,
                                            boundary_conditions=self.ax_deriv_configs.get(ax, DerivativeConfiguration()))
                   for ax in avg_axes}
        avg_mpo = self.grid.make_mpo_ndim(avg_ops)
        out = self.apply(avg_mpo, inplace=inplace, zipup=True, compress=compress, compress_opts=compress_opts)
        return out

    def add_dissipation(self, strength: Numeric, deriv_order: int=2, deriv_axes=None, inplace=False,
                        compress=True, compress_opts:dict = None, verbose_plot=False):
        """Apply one explicit Euler dissipation step to the field.

        Computes ``f <- f + eta d^m/dx^m f = (1 + eta d^m/dx^m) f`` by adding the
        dissipation term from :meth:`get_dissipation`. For stability the strength
        is assumed small (``eta << dt/dx^2``). Stencils (with ``eta`` the
        strength)::

            deriv_order = 1 (m=2): f_j + eta (f_{j+1} - 2 f_j + f_{j-1})
            deriv_order = 2 (m=4): f_j + eta (-f_{j+2} + 4 f_{j+1} - 6 f_j + 4 f_{j-1} - f_{j-2})
            deriv_order = 3 (m=6): f_j + eta (f_{j+3} - 6 f_{j+2} + 15 f_{j+1} - 20 f_j + 15 f_{j-1} - 6 f_{j-2} + f_{j-3})

        i.e. the m-th derivative with a 2nd-order finite-difference stencil.

        Parameters
        ----------
        strength : Numeric
            Dissipation strength eta.
        deriv_order : int
            Order code of the derivative (order=1 gives m=2, etc.).
        deriv_axes : Sequence[Axis], optional
            Axes along which to dissipate. Defaults to all grid axes.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the addition.
        compress_opts : dict, optional
            Overrides for the compression parameters.
        verbose_plot : bool
            Plot/print the dissipation difference for diagnostics if True.

        Returns
        -------
        GridTN
            The dissipated field.
        """
        if verbose_plot:
            old = self.copy()

        dissip_f = self.get_dissipation(strength, deriv_order, deriv_axes=deriv_axes, compress=False)
        out = self.add(dissip_f, inplace=inplace, compress=compress, compress_opts=compress_opts)

        if verbose_plot:
            if old.data is not None:
                print('dissip diff', old.distance(out) / old.frobenius_norm())

                diff_data = dissip_f.get_data()  # old.add(out.scalar_multiply(-1)).get_data()
                plt.figure()
                # plt.imshow(np.log10(np.abs(diff_data)))
                plt.imshow(diff_data)
                plt.colorbar()
                plt.title('dissipation diff')
                plt.show()

        return out

    def get_dissipation(self, strength: Numeric, deriv_order: int=2, deriv_axes=None,
                        compress=True, compress_opts:dict = None):
        """Build the dissipation term ``eta d^m/dx^m f`` of an Euler step.

        Returns the increment added by :meth:`add_dissipation`, i.e.
        ``eta d^m/dx^m f`` evaluated with a centered 2nd-order finite-difference
        stencil and the ``dx`` factors removed. For stability the strength is
        assumed small (``eta << dt/dx^2``).

        Parameters
        ----------
        strength : Numeric
            Dissipation strength eta.
        deriv_order : int
            Order code of the derivative; must be even.
        deriv_axes : Sequence[Axis], optional
            Axes along which to dissipate. Defaults to all grid axes.
        compress : int
            Compression-aggressiveness level applied to the result.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The dissipation increment field.

        Notes
        -----
        When ``deriv_order % 4 == 0`` the sign of ``strength`` is flipped before
        scaling, so that the dissipation is sign-correct for those orders.
        """
        # assert(strength <= 1 / deriv_order ** 2), f'smoothing strength {strength} is too large'  # see p.114 in Durran
        assert(deriv_order % 2 == 0), f'deriv_order must be even, not {deriv_order}'

        ax_deriv_configs = {k: v.copy() for k,v in self.ax_deriv_configs.items()}
        for k, v in ax_deriv_configs.items():
            v.update(order=1, fd_type=FDType.CENTER)

        if deriv_order % 4 == 0:
            strength = strength * -1

        dissip_f = self.take_mth_laplacian(deriv_order, deriv_axes=deriv_axes, ax_deriv_configs=ax_deriv_configs,
                                           remove_dx2=True, compress=compress, compress_opts=compress_opts)
        dissip_f = dissip_f.scalar_multiply(strength, inplace=True)

        return dissip_f


    def average_neighbor(self, avg_axes=None, spread:int = 1, inplace=False, compress=True, compress_opts=None, mu=0.5):
        """Apply a neighbor-averaging (smoothing) MPO along selected axes.

        Each point is replaced by a weighted average of itself and its
        neighbors::

            x_j <-- sum_{i=1}^{spread} (1-mu)/(2*spread) (x_{j-i} + x_{j+i}) + mu x_j

        e.g. for ``spread = 1``: ``x_j <-- (1-mu)/2 x_{j-1} + mu x_j + (1-mu)/2 x_{j+1}``.

        Parameters
        ----------
        avg_axes : Sequence[Axis], optional
            Axes to average over. Defaults to all grid axes.
        spread : int
            Number of neighbors on each side to include in the average.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the averaging.
        compress_opts : dict, optional
            Overrides for the compression parameters.
        mu : float
            Weight placed on the central element.

        Returns
        -------
        GridTN
            The smoothed field.
        """
        avg_axes = self.grid.axes if avg_axes is None else avg_axes
        avg_ops = {ax: ax.get_neighbor_avg_mpo(spread=spread, mu=mu,
                                               boundary_conditions=self.ax_deriv_configs.get(ax, DerivativeConfiguration()))
                   for ax in avg_axes}
        avg_mpo = self.grid.make_mpo_ndim(avg_ops)
        out = self.apply(avg_mpo, inplace=inplace, zipup=True, compress=compress, compress_opts=compress_opts)
        return out

    def average_leapfrog(self, avg_axes=None, inplace=False, compress=True, compress_opts=None):
        """Apply a leapfrog-averaging MPO along selected axes.

        Parameters
        ----------
        avg_axes : Sequence[Axis], optional
            Axes to average over. Defaults to all grid axes.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the averaging.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The averaged field.
        """
        avg_axes = self.grid.axes if avg_axes is None else avg_axes
        # avg_ops = {ax: ax.get_leapfrog_mpo(boundary_conditions=self.ax_deriv_configs.get(ax, DerivativeConfiguration()))
        #            for ax in avg_axes}
        # avg_mpo = self.grid.make_mpo_ndim(avg_ops)
        avg_mpo = self.grid.get_leapfrog_mpo(avg_axes, self.ax_deriv_configs)
        out = self.apply(avg_mpo, inplace=inplace, zipup=True, compress=compress, compress_opts=compress_opts)
        return out

    def shift_cell_to_midpoint(self, shift_axes=None, inplace=False, compress=True, compress_opts=None,
                               inverse=False):
        """Shift the field by half a cell to the cell midpoint.

        Computes ``1/2 (x_i + x_{i+1})`` along each shift axis, or
        ``1/2 (x_i + x_{i-1})`` when ``inverse`` is True, and updates the
        boundary conditions accordingly.

        Parameters
        ----------
        shift_axes : Sequence[Axis], optional
            Axes to shift. Defaults to all grid axes.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the shift.
        compress_opts : dict, optional
            Overrides for the compression parameters.
        inverse : bool
            Shift toward the lower neighbor (``x_{i-1}``) instead of the upper
            neighbor if True.

        Returns
        -------
        GridTN
            The shifted field.
        """
        shift_axes = self.grid.axes if shift_axes is None else shift_axes
        print('shift axes', shift_axes)

        shift = -1 if inverse else 1
        shift_op = {ax: ax.get_shift_mpo(shift,
                                         boundary_conditions=self.ax_deriv_configs.get(ax, DerivativeConfiguration()))
                    for ax in shift_axes}
        shift_mpo = self.grid.make_mpo_ndim(shift_op)
        avg_mpo = self.grid.get_iden_mpo().add(shift_mpo)
        avg_mpo.scalar_multiply(0.5, inplace=True)
        out = self.apply(avg_mpo, inplace=inplace, zipup=True, compress=compress, compress_opts=compress_opts)

        new_deriv_configs = {}
        for ax, dc in self.ax_deriv_configs.items():
            new_deriv_configs[ax] = dc.shifted_bc(inverse=inverse) if ax in shift_axes else dc.copy()
        out.ax_deriv_configs = new_deriv_configs

        return out

    def evaluate_func(self, func: Callable, max_bond=None, inplace=False):
        """Evaluate a scalar function on the field's values element-wise.

        Parameters
        ----------
        func : Callable
            Function applied to the field values.
        max_bond : int, optional
            Bond-dimension limit for the result.
        inplace : bool
            Mutate self in place if True, else return a copy.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def canonize(self, inplace=True, scale=True, form='right', i=None, cur_orthog=None) -> 'GridTN':
        """Put the tensor network into canonical (mixed/one-sided) form.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else return a copy.
        scale : bool
            Absorb the norm into the carried exponent / center tensor if True.
        form : str
            Canonical form to enforce (e.g. ``'left'``, ``'right'``).
        i : int, optional
            Site to canonicalize around (orthogonality center).
        cur_orthog : int or tuple, optional
            Current orthogonality center, used to minimize work.

        Returns
        -------
        GridTN
            The canonicalized field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def canonize_axes(self, axes: Sequence['Axis'], inplace=True, scale=True) -> 'GridTN':
        """Canonicalize the tensor network around the sites of given axes.

        Parameters
        ----------
        axes : Sequence[Axis]
            Axes defining the orthogonality center.
        inplace : bool
            Mutate self in place if True, else return a copy.
        scale : bool
            Absorb the norm into the carried exponent / center tensor if True.

        Returns
        -------
        GridTN
            The canonicalized field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def get_canon_site_from_axes(self, axes: Sequence['Axis']):
        """Return the site index to canonicalize around for the given axes.

        Parameters
        ----------
        axes : Sequence[Axis]
            Axes whose corresponding canonical site is requested.

        Returns
        -------
        int
            The site index for the orthogonality center.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def compress(self, inplace=True, verbose=False, canonize=True, compress_type=CompressType.SVD,
                 compress_opts: dict = None, sub_compress_opts: Optional[dict['SubCompressConfigType', dict]] = None,
                 norm_cutoff: float = None, conservative=False, **kwargs):
        """Compress (truncate) the tensor-network bonds.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else return a copy.
        verbose : bool
            Print compression diagnostics if True.
        canonize : bool
            Canonicalize before truncating if True.
        compress_type : CompressType
            Compression scheme (e.g. SVD).
        compress_opts : dict, optional
            Compression parameters (e.g. ``max_bond``, ``cutoff``).
        sub_compress_opts : dict[SubCompressConfigType, dict], optional
            Per-stage compression parameters for sub-network truncation.
        norm_cutoff : float, optional
            Norm-based truncation threshold.
        conservative : bool
            Use mass-conserving compression if True.
        **kwargs
            Additional options forwarded to the compression routine.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def compress_rdm(self, inplace=True, verbose=False, compress_opts=None, sub_compress_opts=None, direction=1,
                     open_end=False, left_env=None, right_env=None, back_compress=True,
                     **kwargs):
        """Compress the tensor network using the reduced-density-matrix method.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else return a copy.
        verbose : bool
            Print compression diagnostics if True.
        compress_opts : dict, optional
            Compression parameters (e.g. ``max_bond``, ``cutoff``).
        sub_compress_opts : dict, optional
            Per-stage compression parameters for sub-network truncation.
        direction : int
            Sweep direction (1 left-to-right, -1 right-to-left).
        open_end : bool
            Treat the end bond as open (no environment closure) if True.
        left_env : qtn.Tensor, optional
            Precomputed left environment tensor.
        right_env : qtn.Tensor, optional
            Precomputed right environment tensor.
        back_compress : bool
            Perform a back-sweep compression after the forward sweep if True.
        **kwargs
            Additional options forwarded to the compression routine.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def get_bases(self, i: int):
        """Return the MPS basis functions assuming the center is at site i.

        Parameters
        ----------
        i : int
            Site index of the orthogonality center.

        Returns
        -------
        object
            The basis functions at site ``i``.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def contract(self):
        """Contract the underlying tensor network to a scalar/dense result.

        Returns
        -------
        object
            The contracted value.
        """
        return self.data.contract()

    def scalar_add(self, scalar_val, inplace=False):
        """Add a constant scalar value to every grid point of the field.

        Parameters
        ----------
        scalar_val : Numeric
            Constant to add across the grid.
        inplace : bool
            Mutate self in place if True, else return a copy.

        Returns
        -------
        GridTN
            The shifted field.
        """
        gtn = self if inplace else self.copy()
        ones = self.grid.get_ones_mps()
        ones.scalar_multiply(scalar_val, inplace=True)
        gtn = gtn.add(ones, inplace=True)
        return gtn

    # def apply_partial_mpx(self, mpo: MPOType or MPSType or 'GridTN', reset_grid=True, compress=True,
    #           **compress_opts) -> 'GridTN':
    #     raise NotImplementedError

    def apply_qft(self, ft_axes=None, inplace=False, compress=False, compress_opts=None):
        """Apply the QFT to the field (similarity transform for an MPO).

        For an MPS the forward QFT MPO is applied; for an MPO the operator is
        conjugated by the QFT (``Q M Q^{-1}``).

        Parameters
        ----------
        ft_axes : Sequence[Axis], optional
            Axes to transform. Defaults to all grid axes.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the transform.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The transformed field.
        """
        gtn = self if inplace else self.copy()

        if gtn.data_type == DataType.MPS:
            qft_mpo = self.grid.get_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?
            gtn.apply(qft_mpo, inplace=True, zipup=True, compress=compress, compress_opts=compress_opts)
        elif gtn.data_type == DataType.MPO:
            qft_mpo = self.grid.get_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?
            iqft_mpo = self.grid.get_inverse_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?
            gtn_ = iqft_mpo.apply(gtn, zipup=True)
            gtn_.apply(qft_mpo, inplace=True, zipup=True, compress=compress, compress_opts=compress_opts)
            gtn.data = gtn_.data
        else:
            raise NotImplementedError

        return gtn

    def apply_inverse_qft(self, ft_axes=None, inplace=False, compress=False, compress_opts=None):
        """Apply the inverse QFT to the field (similarity transform for an MPO).

        For an MPS the inverse QFT MPO is applied; for an MPO the operator is
        conjugated by the inverse QFT.

        Parameters
        ----------
        ft_axes : Sequence[Axis], optional
            Axes to transform. Defaults to all grid axes.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the transform.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The transformed field.
        """
        gtn = self if inplace else self.copy()

        if gtn.data_type == DataType.MPS:
            qft_mpo = self.grid.get_inverse_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?
            gtn.apply(qft_mpo, inplace=True, zipup=True, compress=compress, compress_opts=compress_opts)
        elif gtn.data_type == DataType.MPO:
            print('inverse qft mpo')
            qft_gtn = self.grid.get_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?
            inv_qft_gtn = self.grid.get_inverse_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?

            gtn_1 = qft_gtn.apply(gtn, zipup=True)
            gtn_1.apply(inv_qft_gtn, inplace=True, zipup=True, compress=True, compress_opts=compress_opts)
            gtn.data = gtn_1.data
        else:
            raise NotImplementedError

        return gtn

    #######################
    ###  MPS_USVT fcts  ###
    #######################

    def convert_to_USVT(self, canon_site, inplace=False, cur_orthog=None):
        """Convert the MPS into a split U S V^T form around a canonical site.

        Parameters
        ----------
        canon_site : int
            Site index around which to split into U, S, V^T factors.
        inplace : bool
            Mutate self in place if True, else return a copy.
        cur_orthog : int or tuple, optional
            Current orthogonality center, used to minimize work.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def convert_from_USVT(self, inplace=False, canon_site=None):
        """Recombine a U S V^T form back into a standard MPS.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else return a copy.
        canon_site : int, optional
            Site index of the current split / orthogonality center.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def get_S_tensor(self, **kwargs):
        """Return the central singular-value (S) tensor of a U S V^T form.

        Parameters
        ----------
        **kwargs
            Options forwarded to the extraction routine.

        Returns
        -------
        qtn.Tensor
            The S tensor.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    ################
    ## processing ##
    ################

    def _coarsen_grid(self, depth: int, method: 'str', is_sqrt=False) -> 'GridTN':
        """Coarsen the grid by ``depth`` levels via a coarse-graining MPO.

        Builds a coarsened grid (fewer sites, larger ``dx``), applies the
        appropriate coarse-graining or selection operator, contracts away the
        removed indices, and returns the field on the new grid.

        Parameters
        ----------
        depth : int
            Number of levels to coarsen each axis by.
        method : str
            ``'select'`` to subsample points, otherwise integrate (average) over
            the removed fine indices.
        is_sqrt : bool
            Treat the field as a square-root representation when building the
            coarse-graining operator.

        Returns
        -------
        GridTN
            The field on the coarsened grid.
        """

        grid = self.grid
        Axis = grid.axes[0].__class__
        new_axes = [Axis(ax.L - depth, dx=depth * ax.dx, x0=ax.xpts[0] + depth * ax.dx / 2,
                         ax_map=ax.map, coordinate=ax.coordinate, is_flipped=ax.is_flipped, basis=ax.basis)
                    for ax in grid.axes]
        new_grid = grid.create_like(new_axes)

        if method=='select':
            coarse_grain_op = grid.get_coarse_select_mpx(depth, is_sqrt=False)
        else:
            coarse_grain_op = grid.get_coarse_grain_mpx(depth, is_sqrt=is_sqrt)

        # L_ = grid.L - depth * len(new_axes)
        gtn_coarse = self.apply(coarse_grain_op)
        coarse_mps = gtn_coarse.data
        site_ind_id = self.data.site_ind_id
        site_tag_id = self.data.site_tag_id

        ## contract coarse tensors
        # if no site_ind, merge with next tensor/tensor on the right
        mps_idxs = []
        for i in range(grid.L):
            tens = coarse_mps[i]
            if site_ind_id.format(i) not in tens.inds:
                next_tens = coarse_mps[mps_idxs[-1]] if i == grid.L - 1 else coarse_mps[i + 1]
                new_tens = qtn.tensor_contract(tens, next_tens)
                next_tens.modify(data=new_tens.data, inds=new_tens.inds)
                coarse_mps.delete(site_tag_id.format(i), which='all')
            else:
                mps_idxs += [i]

        coarse_mps._L = coarse_mps.num_tensors
        coarse_mps.exponent = self.data.exponent + coarse_grain_op.exponent
        helper.renumber_mps(coarse_mps, mps_idxs, list(range(coarse_mps._L)), inplace=True )

        coarse_gtn = self.__class__(new_grid, coarse_mps)
        return coarse_gtn

    def coarsen_grid_integrate(self, depth: int, is_sqrt=False) -> 'GridTN':
        """Coarsen the grid by integrating (averaging) over removed points.

        Parameters
        ----------
        depth : int
            Number of levels to coarsen each axis by.
        is_sqrt : bool
            Treat the field as a square-root representation when coarse-graining.

        Returns
        -------
        GridTN
            The field on the coarsened grid.
        """
        return self._coarsen_grid(depth, 'integrate', is_sqrt=is_sqrt)

    def coarsen_grid_select(self, depth: int) -> 'GridTN':
        """Coarsen the grid by subsampling (selecting) points.

        Parameters
        ----------
        depth : int
            Number of levels to coarsen each axis by.

        Returns
        -------
        GridTN
            The field on the coarsened grid.
        """
        return self._coarsen_grid(depth, 'select')


    def interpolate_data(self, depth: int, interp_dict: dict['Axis': 'qtn.MatrixProductOperator'],
                         new_grid: 'Grid'=None):
        """Refine the grid by interpolating data between existing grid points.

        Extends the grid by ``depth`` levels and fills the new fine points using
        the per-axis interpolation MPOs. Only outer-product interpolation
        schemes are supported.

        Parameters
        ----------
        depth : int
            Number of refinement levels to add.
        interp_dict : dict[Axis, qtn.MatrixProductOperator]
            Per-axis interpolation operators whose tensors carry all indices
            decomposed on the coarse grid (fine indices labeled ``"fine"``,
            though not necessarily numbered appropriately).
        new_grid : Grid, optional
            Target refined grid. If None, one is derived from ``depth``.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def gaussian_smooth_data(self, sigma=0.3, proc_axes: Optional[Sequence['Axis']] = None, mode='wrap',
                             inplace=False, compress=True, compress_opts=None) -> 'GridTN':
        """Smooth the field with a Gaussian-blur MPO along selected axes.

        Parameters
        ----------
        sigma : float
            Standard deviation (in grid units) of the Gaussian kernel.
        proc_axes : Sequence[Axis], optional
            Axes to smooth along. Defaults to all grid axes.
        mode : str
            Boundary-handling mode for the blur (e.g. ``'wrap'``).
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the smoothing.
        compress_opts : dict, optional
            Overrides for the compression parameters (also passed as the blur
            MPO's split options).

        Returns
        -------
        GridTN
            The smoothed field.
        """
        gtn = self if inplace else self.copy(deep=False)
        blur_mpo = self.grid.gaussian_smooth_mpo(sigma=sigma, proc_axes=proc_axes, mode=mode, split_opts=compress_opts)
        # leave out compress_opts for default (most accurate) compression if compress=True

        if blur_mpo is not None:
            gtn = gtn.apply(blur_mpo, inplace=True, compress=compress, compress_opts=compress_opts)  ## not deep inplace
        return gtn

    def apply_absorbing_bc(self, x_ax: 'Axis', v_ax: 'Axis', inplace=False, compress=True, compress_opts=None) \
            -> 'GridTN':
        """Apply absorbing boundary conditions along a position/velocity pair.

        Applies the absorbing-BC MPO to the field when the position axis has an
        absorbing boundary. Assumes TN dimensions are ordered
        ``x1, x2, x3, v1, v2, v3``.

        Parameters
        ----------
        x_ax : Axis
            Position axis carrying the boundary condition.
        v_ax : Axis
            Velocity axis paired with ``x_ax``.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the operation.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The field with absorbing BCs applied.
        """
        gtn = self if inplace else self.copy(deep=False)
        left_bc, right_bc = gtn.ax_deriv_configs[x_ax].bc

        if left_bc == BCType.ABSORBING or right_bc == BCType.ABSORBING:
            absorb_bc_mpo = self.grid.absorbing_bc_mpo(x_ax, v_ax, left_bc, right_bc, compress=False)
            # print(absorb_bc_mpo, absorb_bc_mpo.data)
            # leave out compress_opts for default (most accurate) compression if compress=True

            # if absorb_bc_mpo is not None: # and absorb_bc_mpo.data is not None:
            gtn = gtn.apply(absorb_bc_mpo, inplace=True, compress=compress, compress_opts=compress_opts)

        return gtn

    def apply_reflecting_v_bc(self, x_ax, v_ax, left_bc, right_bc, inplace=False, compress=True, compress_opts=None):
        """Apply velocity-reflecting boundary conditions to the field.

        Parameters
        ----------
        x_ax : Axis
            Position axis carrying the boundary.
        v_ax : Axis
            Velocity axis to reflect.
        left_bc : BCType
            Boundary condition at the left/lower edge.
        right_bc : BCType
            Boundary condition at the right/upper edge.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the operation.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The field with reflecting velocity BCs applied.
        """
        gtn = self if inplace else self.copy()
        reflect_bc_mpo = self.grid.reflecting_v_bc_mpo(x_ax, v_ax, left_bc, right_bc, compress=False)
        # leave out compress_opts for default (most accurate) compression if compress=True

        if reflect_bc_mpo is not None:
            if compress_opts is None:   compress_opts = {}
            gtn = gtn.apply(reflect_bc_mpo, inplace=True, compress=compress, compress_opts=compress_opts)
        return gtn

    ###############################################
    ### build derivative and integral operators ###
    ###############################################

    def mps_to_diag_mpo(self, lower_ind_id='i({})', upper_ind_id='o({})', inplace=False) -> 'GridTN':
        """Convert an MPS into a diagonal MPO.

        Parameters
        ----------
        lower_ind_id : str
            Format string for lower physical index labels of the MPO.
        upper_ind_id : str
            Format string for upper physical index labels of the MPO.
        inplace : bool
            Mutate self in place if True, else return a copy.

        Returns
        -------
        GridTN
            The diagonal-operator field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def apply_elemental_multiply_op(self, lower_ind_id='i({})', upper_ind_id='o({})', axes=None, add_cc=False,
                                    take_mps_cc=False, compress=False, compress_opts=None):
        """Build the elemental-multiply operator (diagonal MPO) for self.

        Produces the operator used for element-wise products (e.g. convolution or
        a d_ijk function) by turning the field into a diagonal MPO.

        Parameters
        ----------
        lower_ind_id : str
            Format string for lower physical index labels of the MPO.
        upper_ind_id : str
            Format string for upper physical index labels of the MPO.
        axes : Sequence[Axis], optional
            Axes over which to form the diagonal operator. Defaults to all.
        add_cc : bool
            Add the complex conjugate contribution if True.
        take_mps_cc : bool
            Conjugate the MPS before forming the operator if True.
        compress : int
            Compression-aggressiveness level applied to the operator.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The diagonal multiply-operator field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    # def apply_xmultiply_mpo(self, x_axes: list['Axis'] or None = None, offsets: dict['Axis', Numeric] = 0.0,
    #                         scales: dict['Axis', Numeric] = 1.0,inplace=False, compress=False, compress_opts=None):
    #     """ perform x*f(x) operation, where x_dims specifies which axis/axes to multiply
    #         eg. 1D:  x*f(x,y,...)
    #             2D:  x*y*f(x,y,...)
    #     """
    #     if self.data is None:
    #         return self
    #
    #     gtn = self if inplace else self.copy()
    #     x_mpo = self.grid.get_xmultiply_mpo(x_axes, offsets=offsets, scales=scales)
    #     if compress_opts is None:   compress_opts = {}
    #     gtn.apply(x_mpo, compress=compress, compress_opts=compress_opts)
    #     return gtn

    ## put bc into deriv_params to avoid carrying it around when calling curl, laplacian, etc.
    # @profile
    def take_firstderivative(self, ax: 'Axis', upwind_ax: Optional['Axis'] = None,
                             ax_deriv_config: dict['Axis', 'DerivativeConfiguration'] = None,
                             recalc=False, inplace=False, compress=False, compress_opts=None):
        """Take the first derivative ``df/dx_i`` along an axis.

        Parameters
        ----------
        ax : Axis
            Axis to differentiate along.
        upwind_ax : Axis, optional
            Axis whose sign selects the upwind direction for upwind stencils.
        ax_deriv_config : dict[Axis, DerivativeConfiguration], optional
            Per-Axis finite-difference configs; overrides ``self.ax_deriv_configs``
            for this call.
        recalc : bool
            Force recomputation of the derivative MPO instead of using a cache.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the operation.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The differentiated field.
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        gtn = self if inplace else self.copy()
        deriv_config = gtn.ax_deriv_configs[ax] if ax_deriv_config is None else ax_deriv_config[ax]
        ddx_mpo = self.grid.get_firstderivative_mpo(ax, deriv_config=deriv_config, upwind_ax=upwind_ax, recalc=recalc)
        gtn.apply(ddx_mpo, inplace=True)

        if compress:
            gtn = gtn.compress(inplace=True, compress_opts=compress_opts)
        return gtn

    def take_secondderivative(self, ax1: 'Axis', ax2: 'Axis' or None,
                              recalc=False, inplace=False, compress=False, compress_opts=None):
        """Take a second derivative (or mixed partial) along one or two axes.

        Computes ``d^2 f / dx1^2`` when ``ax2`` is None, otherwise the mixed
        partial ``d^2 f / (dx1 dx2)``.

        Parameters
        ----------
        ax1 : Axis
            First differentiation axis.
        ax2 : Axis or None
            Second differentiation axis for a mixed partial, or None for a pure
            second derivative along ``ax1``.
        recalc : bool
            Force recomputation of the derivative MPO instead of using a cache.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the operation.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The differentiated field.
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        gtn = self if inplace else self.copy()
        deriv_config1 = gtn.ax_deriv_configs[ax1]
        deriv_config2 = gtn.ax_deriv_configs[ax2] if ax2 is not None else None
        d2_mpo = self.grid.get_secondderivative_mpo(ax1, ax2, deriv_config1=deriv_config1, deriv_config2=deriv_config2,
                                                    recalc=recalc)
        gtn = gtn.apply(d2_mpo, inplace=True, compress=compress, compress_opts=compress_opts)
        return gtn

    def take_mth_derivative(self, deriv_order: int, ax: 'Axis', deriv_config: 'DerivativeConfiguration'=None,
                            recalc=False, inplace=False, compress=False, compress_opts=None):
        """Take the m-th derivative ``d^m/dx^m`` along a specified axis.

        Parameters
        ----------
        deriv_order : int
            Order m of the derivative.
        ax : Axis
            Axis to differentiate along.
        deriv_config : DerivativeConfiguration, optional
            Finite-difference config to use; overrides this axis's config.
        recalc : bool
            Force recomputation of the derivative MPO instead of using a cache.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the operation.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The differentiated field.
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        gtn = self if inplace else self.copy()
        deriv_config1 = gtn.ax_deriv_configs[ax] if deriv_config is None else deriv_config
        d2_mpo = self.grid.get_mth_derivative_mpo(ax, deriv_order, deriv_config=deriv_config1, recalc=recalc)
        gtn = gtn.apply(d2_mpo, inplace=True, compress=compress, compress_opts=compress_opts)
        return gtn

    def take_mth_laplacian(self, deriv_order: int, deriv_axes=None, coord_sys: 'CoordinateSystem'=None,
                           ax_deriv_configs: dict['Axis': 'DerivativeConfiguration'] = None, remove_dx2=False,
                           recalc=False, inplace=False, compress=False, compress_opts=None
                           ) -> 'GridTN':
        """Compute the m-th-order Laplacian (sum of ``d^m/dx^m`` over axes).

        Sums the m-th derivative of the field along each derivative axis; valid
        for even m. Optionally removes the ``dx^m`` scaling per axis.

        Parameters
        ----------
        deriv_order : int
            Order m of each derivative (even).
        deriv_axes : Sequence[Axis], optional
            Axes to include in the Laplacian. Defaults to all grid axes.
        coord_sys : CoordinateSystem, optional
            Coordinate system for a curvilinear Laplacian (currently raises
            ``NotImplementedError`` when provided).
        ax_deriv_configs : dict[Axis, DerivativeConfiguration], optional
            Per-Axis finite-difference configs; defaults to
            ``self.ax_deriv_configs``.
        remove_dx2 : bool
            Multiply each term by the appropriate ``dx^m`` (or Fourier analogue)
            factor to remove the grid-spacing scaling if True.
        recalc : bool
            Force recomputation of the derivative MPOs instead of using a cache.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied to the final sum.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The Laplacian field.
        """

        if coord_sys is not None:
            mpo_list = coord_sys.build_mth_laplacian_mpo(deriv_order, axes=deriv_axes,
                                                         deriv_configs=self.ax_deriv_configs,)
            raise NotImplementedError

        if ax_deriv_configs is None:
            ax_deriv_configs = self.ax_deriv_configs

        if deriv_axes is None:
            deriv_axes = self.grid.axes

        out_mps = None
        for ax in deriv_axes:
            out = self.take_mth_derivative(deriv_order, ax, deriv_config=ax_deriv_configs[ax])
            if remove_dx2:
                if ax.basis.type == BasisType.FOURIER:
                    dx2_coeff = (1. / np.abs(ax.xpts[0]))**deriv_order
                elif ax.basis.type == BasisType.SPATIAL:
                    dx2_coeff = ax.dx**deriv_order
                else:
                    raise NotImplementedError
                out.scalar_multiply(dx2_coeff, inplace=True)

            if out_mps is None:
                out_mps = out
            else:
                out_mps = out_mps.add(out, inplace=True, compress=False)

        if compress and out_mps is not None:
            out_mps = out_mps.compress(compress_opts=compress_opts, inplace=True)

        return out_mps


    # def take_mth_laplacian(self, deriv_order: 'int', axes: Sequence['Axis']=None,
    #                         recalc=False, inplace=False, compress=False, compress_opts=None):
    #     """
    #     """
    #     if self.data is None:
    #         return self if inplace else self.create_like(new_data=None)
    #
    #     tmp = None  # self if inplace else self.copy()
    #     deriv_config1 = gtn.ax_deriv_configs[ax]
    #     d2_mpo = self.grid.get_mth_derivative_mpo(ax, deriv_order, deriv_config=deriv_config1, recalc=recalc)
    #
    #     for ax in
    #     gtn = self.apply(d2_mpo, inplace=False, compress=compress, compress_opts=compress_opts)
    #
    #     return gtn


    def integrate(self, integ_axes: Sequence['Axis'] = None, is_sqrt=False, ancilla_reindex: dict[str, str] = None,
                  new_grid: 'Grid' = None, new_ax_deriv_configs: dict[Any, 'DerivativeConfiguration'] = None,
                  exclude_weights=False, compress=False, compress_opts=None):
        """Integrate the field (or its square) over selected axes.

        Parameters
        ----------
        integ_axes : Sequence[Axis], optional
            Axes to integrate over. Defaults to all grid axes.
        is_sqrt : bool
            Treat the field as a square-root representation and integrate its
            square if True.
        ancilla_reindex : dict[str, str], optional
            Index relabeling for ancilla legs left after partial integration.
        new_grid : Grid, optional
            Grid to associate with the partially integrated result.
        new_ax_deriv_configs : dict[Any, DerivativeConfiguration], optional
            Per-Axis finite-difference configs for the result grid.
        exclude_weights : bool
            Omit the quadrature weights from the integral if True.
        compress : int
            Compression-aggressiveness level applied to the result.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        Numeric or GridTN
            A scalar when all axes are integrated, otherwise the reduced field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def meas_elem(self, sel_inds: Sequence[int], site_ind_id='i({})', site_tag_id='X({})',):
        """Measure a single field element at the given multi-index.

        Parameters
        ----------
        sel_inds : Sequence[int]
            Per-axis indices selecting the element to read off.
        site_ind_id : str
            Format string for physical (site) index labels.
        site_tag_id : str
            Format string for per-site tensor tags.

        Returns
        -------
        Numeric
            The selected field value.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def meas_shifted_elem(self, sel_inds: Sequence[int], shifts: dict['Axis', int],
                          ax_deriv_configs: ['Axis', 'DerivativeConfiguration'],):
        """Measure a field element after shifting by per-axis offsets.

        Parameters
        ----------
        sel_inds : Sequence[int]
            Per-axis indices selecting the base element.
        shifts : dict[Axis, int]
            Per-axis shift (in grid points) applied before reading the element.
        ax_deriv_configs : dict[Axis, DerivativeConfiguration]
            Per-Axis finite-difference configs supplying the boundary handling
            used for the shift.

        Returns
        -------
        Numeric
            The shifted field value.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError


    def meas_expec(self, obs_gtn, integ_axes=None, is_sqrt=False, new_grid=None, exclude_axes=None,
                   exclude_weights=False, new_ax_deriv_configs: dict[Any, 'DerivativeConfiguration'] = None,
                   compress=False, compress_opts=None, **kwargs) -> Numeric:
        """Measure the expectation value of an observable against the field.

        Parameters
        ----------
        obs_gtn : GridTN
            Observable (operator or field) to measure.
        integ_axes : Sequence[Axis], optional
            Axes to integrate over. Defaults to all grid axes.
        is_sqrt : bool
            Treat the field as a square-root representation if True.
        new_grid : Grid, optional
            Grid to associate with a partially measured result.
        exclude_axes : Sequence[Axis], optional
            Axes to leave un-integrated.
        exclude_weights : bool
            Omit quadrature weights from the integral if True.
        new_ax_deriv_configs : dict[Any, DerivativeConfiguration], optional
            Per-Axis finite-difference configs for the result grid.
        compress : int
            Compression-aggressiveness level applied during the measurement.
        compress_opts : dict, optional
            Overrides for the compression parameters.
        **kwargs
            Additional options forwarded to the measurement routine.

        Returns
        -------
        Numeric
            The expectation value (or reduced field if axes are excluded).

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def project(self, obs_gtn, proj_axes, new_grid=None, canonize=True, compress=False, compress_opts=None,
                new_ax_deriv_configs=None, **kwargs) -> 'GridTN':
        """Project an observable field onto the tangent manifold of self.

        Parameters
        ----------
        obs_gtn : GridTN
            Field to project onto self's manifold.
        proj_axes : Sequence[Axis]
            Axes defining the projection / bases.
        new_grid : Grid, optional
            Grid to associate with the projected result.
        canonize : bool
            Canonicalize self before projecting if True.
        compress : int
            Compression-aggressiveness level applied to the result.
        compress_opts : dict, optional
            Overrides for the compression parameters.
        new_ax_deriv_configs : dict[Axis, DerivativeConfiguration], optional
            Per-Axis finite-difference configs for the result grid.
        **kwargs
            Additional options forwarded to the projection routine.

        Returns
        -------
        GridTN
            The projected field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def project_bond(self, obs_gtn, bond_ind) -> 'qtn.Tensor':
        """Project an observable onto the bond-tensor subspace of self.

        Parameters
        ----------
        obs_gtn : GridTN
            Field to project.
        bond_ind : str
            Bond (virtual) index defining the projection.

        Returns
        -------
        qtn.Tensor
            The projected bond tensor.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def project_site(self, obs_gtn, site_ind, nsites=1) -> 'qtn.Tensor':
        """Project an observable onto the site-tensor subspace of self.

        Parameters
        ----------
        obs_gtn : GridTN
            Field to project.
        site_ind : int
            Site index defining the projection.
        nsites : int
            Number of contiguous sites spanned by the projection block.

        Returns
        -------
        qtn.Tensor
            The projected site tensor.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def project_op_bond(self, obs_gtn, bond_ind, left_env: 'qtn.Tensor' = None, right_env: 'qtn.Tensor' = None,
                        **kwargs) -> tuple['qtn.Tensor', Sequence[str], Sequence[str]]:
        """Project an operator onto the bond-tensor subspace of self.

        Parameters
        ----------
        obs_gtn : GridTN
            Operator field to project.
        bond_ind : str
            Bond (virtual) index defining the projection.
        left_env : qtn.Tensor, optional
            Precomputed left environment tensor.
        right_env : qtn.Tensor, optional
            Precomputed right environment tensor.
        **kwargs
            Additional options forwarded to the projection routine.

        Returns
        -------
        tuple[qtn.Tensor, Sequence[str], Sequence[str]]
            The projected operator tensor and its associated index labels.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def project_op_site(self, obs_gtn, site_ind, nsites=1, left_env: 'qtn.Tensor' = None,
                        right_env: 'qtn.Tensor' = None) \
            -> tuple['qtn.Tensor', Sequence[str], Sequence[str]]:
        """Project an operator onto the site-tensor subspace of self.

        Parameters
        ----------
        obs_gtn : GridTN
            Operator field to project.
        site_ind : int
            Site index defining the projection.
        nsites : int
            Number of contiguous sites spanned by the projection block.
        left_env : qtn.Tensor, optional
            Precomputed left environment tensor.
        right_env : qtn.Tensor, optional
            Precomputed right environment tensor.

        Returns
        -------
        tuple[qtn.Tensor, Sequence[str], Sequence[str]]
            The projected operator tensor and its associated index labels.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def stagger_grid(self, stagger_vals: dict['Axis', int], inplace=False, compress=False, compress_opts=None):
        """Stagger (sub-grid shift) the field along each axis.

        Shifts the field by the requested amount in half-grid-point increments,
        using a shift MPO (and a half-step averaging MPO for odd half-steps), and
        updates the per-axis offsets/boundary handling accordingly.

        Parameters
        ----------
        stagger_vals : dict[Axis, int]
            Per-axis stagger amount in half-grid-point increments.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress : int
            Compression-aggressiveness level applied after the operation.
        compress_opts : dict, optional
            Overrides for the compression parameters.

        Returns
        -------
        GridTN
            The staggered field.
        """

        gtn = self if inplace else self.copy()

        stagger_mpos = {}
        for ax, val in stagger_vals.items():
            if val != 0:
                deriv_config = gtn.ax_deriv_configs[ax]
                shift = val // 2
                avg = val % 2

                stagger_mpo = ax.get_shift_mpo(shift, boundary_conditions=deriv_config)
                if avg:  ## 1/2 grid step discretization
                    stagger_mpo_2 = ax.get_shift_mpo(shift + 1, boundary_conditions=deriv_config)
                    helper.add_MPO(stagger_mpo, stagger_mpo_2, inplace=True, compress=False)
                    helper.scalar_multiply(stagger_mpo, 0.5, inplace=True)

                if not (deriv_config.left_bc == BCType.PERIODIC or deriv_config.left_bc == BCType.ANTIPERIODIC
                        or deriv_config.left_bc != BCType.OPEN):
                    deriv_config.offset = deriv_config.offset + val

                stagger_mpos[ax] = stagger_mpo

        # orig_data = gtn.get_data()

        stagger_mpo = gtn.grid.make_mpo_ndim(stagger_mpos)
        gtn = gtn.apply(stagger_mpo, inplace=True, compress=compress, compress_opts=compress_opts)

        # new_data = gtn.get_data()

        # plt.figure()
        # plt.plot(orig_data)
        # plt.plot(new_data)
        # plt.show()

        return gtn

    def evolve_tdvp(self, dt, mpo_list, te_order=0, do_adapt=False, inplace=False, compress_config=None,
                    expand_basis: Sequence['GridTN'] = None):
        """Evolve the field one time step using the TDVP integrator.

        Parameters
        ----------
        dt : Numeric
            Time-step size.
        mpo_list : Sequence[GridTN]
            Operator (MPO) terms defining the generator of the evolution.
        te_order : int
            Time-integration order code.
        do_adapt : bool
            Adapt the bond dimensions during evolution if True.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress_config : dict, optional
            Compression configuration applied during the sweep.
        expand_basis : Sequence[GridTN], optional
            Extra fields used to enrich/expand the variational basis.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def evolve_tdmrg(self, dt, mpo_list, te_order=0, do_adapt=True, inplace=False, compress_config=None):
        """Evolve the field one time step using the time-dependent DMRG sweep.

        Parameters
        ----------
        dt : Numeric
            Time-step size.
        mpo_list : Sequence[GridTN]
            Operator (MPO) terms defining the generator of the evolution.
        te_order : int
            Time-integration order code.
        do_adapt : bool
            Adapt the bond dimensions during evolution if True.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress_config : dict, optional
            Compression configuration applied during the sweep.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def evolve_tdvp_new(self, dt, linear_mpo_list, te_order=0, do_adapt=True, inplace=False, compress_config=None,
                        nonlinear_terms = None, sources=None, solver_type=LocalSolverType.TDDMRG,
                       filter_bases=False, time=None):
        """Evolve the field one time step using the revised TDVP integrator.

        Parameters
        ----------
        dt : Numeric
            Time-step size.
        linear_mpo_list : Sequence[GridTN]
            Linear operator (MPO) terms of the generator.
        te_order : int
            Time-integration order code.
        do_adapt : bool
            Adapt the bond dimensions during evolution if True.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress_config : dict, optional
            Compression configuration applied during the sweep.
        nonlinear_terms : optional
            Nonlinear terms of the evolution operator.
        sources : optional
            Source/forcing terms added at each step.
        solver_type : LocalSolverType
            Local solver used for the sweeps.
        filter_bases : bool
            Filter the variational bases during evolution if True.
        time : Numeric, optional
            Current simulation time (for time-dependent terms).

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def evolve_tdmrg_new(self, dt, linear_mpo_list, te_order=0, do_adapt=True, inplace=False, compress_config=None,
                         nonlinear_terms = None, sources=None, solver_type=LocalSolverType.TDDMRG,
                         filter_bases=False, verbose_plot=False, time=None, direction=1,
                         upwind_func=None, upwind_deriv_func=None,
                         **kwargs):
        """Evolve the field one time step using the revised time-dependent DMRG.

        Parameters
        ----------
        dt : Numeric
            Time-step size.
        linear_mpo_list : Sequence[GridTN]
            Linear operator (MPO) terms of the generator.
        te_order : int
            Time-integration order code.
        do_adapt : bool
            Adapt the bond dimensions during evolution if True.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress_config : dict, optional
            Compression configuration applied during the sweep.
        nonlinear_terms : optional
            Nonlinear terms of the evolution operator.
        sources : optional
            Source/forcing terms added at each step.
        solver_type : LocalSolverType
            Local solver used for the sweeps.
        filter_bases : bool
            Filter the variational bases during evolution if True.
        verbose_plot : bool
            Plot/print diagnostics during the step if True.
        time : Numeric, optional
            Current simulation time (for time-dependent terms).
        direction : int
            Sweep direction (1 left-to-right, -1 right-to-left).
        upwind_func : Callable, optional
            Function selecting the upwind direction for advection terms.
        upwind_deriv_func : Callable, optional
            Function providing the upwind derivative for advection terms.
        **kwargs
            Additional options forwarded to the integrator.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    def evolve_time_local_global(self, dt, linear_mpo_list, te_order=0, do_adapt=True, inplace=False, compress_config=None,
                         nonlinear_terms = None, sources=None, solver_type=LocalSolverType.TDDMRG,
                         filter_bases=False, time=None):
        """Evolve the field one step using a combined local/global time scheme.

        Parameters
        ----------
        dt : Numeric
            Time-step size.
        linear_mpo_list : Sequence[GridTN]
            Linear operator (MPO) terms of the generator.
        te_order : int
            Time-integration order code.
        do_adapt : bool
            Adapt the bond dimensions during evolution if True.
        inplace : bool
            Mutate self in place if True, else return a copy.
        compress_config : dict, optional
            Compression configuration applied during the step.
        nonlinear_terms : optional
            Nonlinear terms of the evolution operator.
        sources : optional
            Source/forcing terms added at each step.
        solver_type : LocalSolverType
            Local solver used for the local sweeps.
        filter_bases : bool
            Filter the variational bases during evolution if True.
        time : Numeric, optional
            Current simulation time (for time-dependent terms).

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        raise NotImplementedError

    # def evolve_tdvp0(self, dt, target_list, te_order=0, do_adapt=True, inplace=False, compress_config=None):
    #     raise NotImplementedError

    ##########################
    ## sqrt field functions ##
    ##########################

    def get_complex_conj(self, inplace=False) -> 'GridTN':
        """Return the element-wise complex conjugate of the field.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else return a copy.

        Returns
        -------
        GridTN
            The complex-conjugated field.

        Notes
        -----
        Overridden by subclasses; the base implementation raises
        ``NotImplementedError``.
        """
        ## needs to be overwritten within each class
        # out = self if inplace else self.copy()
        raise NotImplementedError

    def mult_cc(self, inplace=False):
        """Multiply the field element-wise by its own complex conjugate.

        Yields the pointwise squared modulus ``|f|^2`` of the field.

        Parameters
        ----------
        inplace : bool
            Mutate self in place if True, else return a copy.

        Returns
        -------
        GridTN
            The field of squared moduli.
        """
        cc = self.get_complex_conj()
        out = self.elemental_multiply(cc, inplace=inplace)
        return out


#########################
#### class functions ####
#########################

def add(gtn1: 'GridTN', gtn2: 'GridTN', inplace=False, **kwargs):
    """Add two fields by delegating to ``gtn1.add``.

    Parameters
    ----------
    gtn1 : GridTN
        Left operand (receiver of the addition).
    gtn2 : GridTN
        Right operand to add.
    inplace : bool
        Mutate ``gtn1`` in place if True, else return a copy.
    **kwargs
        Additional options forwarded to :meth:`GridTN.add`.

    Returns
    -------
    GridTN
        The summed field.
    """
    return gtn1.add(gtn2, inplace=inplace, **kwargs)


def add_dmrg(*gtns: 'GridTN', inplace=False, **kwargs):
    """Add several fields by delegating to ``gtns[0].add_dmrg``.

    Parameters
    ----------
    *gtns : GridTN
        Fields to add; the first is the receiver.
    inplace : bool
        Mutate the first field in place if True, else return a copy.
    **kwargs
        Additional options forwarded to :meth:`GridTN.add_dmrg`.

    Returns
    -------
    GridTN
        The summed field.

    Notes
    -----
    ``gtns[1:]`` is forwarded as a single positional tuple argument (it is not
    unpacked), so the receiver gets one tuple rather than separate fields.
    """
    return gtns[0].add_dmrg(gtns[1:], inplace=inplace, **kwargs)


def scalar_multiply(gtn1: 'GridTN', const: 'Numeric', inplace=False, **kwargs):
    """Multiply a field by a scalar by delegating to ``gtn1.scalar_multiply``.

    Parameters
    ----------
    gtn1 : GridTN
        Field to scale.
    const : Numeric
        Scalar multiplier.
    inplace : bool
        Mutate ``gtn1`` in place if True, else return a copy.
    **kwargs
        Additional options forwarded to :meth:`GridTN.scalar_multiply`.

    Returns
    -------
    GridTN
        The scaled field.
    """
    return gtn1.scalar_multiply(const, inplace=inplace, **kwargs)


def euler(gtn1: 'GridTN', dt, deriv0=None, inplace=False, **kwargs):
    """Take one explicit Euler step ``gtn1 + dt * deriv0``.

    Parameters
    ----------
    gtn1 : GridTN
        Current state.
    dt : Numeric
        Time-step size.
    deriv0 : GridTN, optional
        Time derivative of the state. If None, the state is returned unchanged.
    inplace : bool
        Mutate ``gtn1`` in place if True, else return a copy.
    **kwargs
        Additional options forwarded to :meth:`GridTN.add`.

    Returns
    -------
    GridTN
        The advanced state.
    """
    if deriv0 is None:
        return gtn1 if inplace else gtn1.copy()

    return gtn1.add(deriv0.scalar_multiply(dt), inplace=inplace, **kwargs)


def compute_derivative(gtn1: 'GridTN', derivative_mpos: Sequence['GridTN'], **kwargs):
    """Apply and sum a list of derivative MPOs to a field.

    Parameters
    ----------
    gtn1 : GridTN
        Field to differentiate.
    derivative_mpos : Sequence[GridTN]
        Derivative operator (MPO) fields to apply and sum.
    **kwargs
        Additional options forwarded to :meth:`GridTN.sum_apply`.

    Returns
    -------
    GridTN
        The summed derivative field.
    """
    return gtn1.sum_apply(derivative_mpos, **kwargs)
