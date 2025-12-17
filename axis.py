import pdb

import axis
import helper_quimb
from setup_.configs import *
from dataclasses import dataclass
import helper_quimb as helper
from coord.coord_sys import Coordinate, CoordinateSystem
from coord.cartesian import CartesianCoordinateSpace
from basis.basis_spatial import SpatialBasis
from basis.basis_k import FourierBasis, RealFourierBasis
from basis.basis_hermite import HermiteBasis
from axis_map.map_binary import BinaryMap
# from axis_map.map_flipbinary import FlipBinaryMap
# from axis_map.map_mirror import MirrorMap

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from basis.basis import Basis
    from axis_map.map import AxisMap

""" Class defining a dimension
    Specifies dimension ID, basis representation, discretization/grid points along axis
    Build operators to take differentials, multiply by x (defined by Basis)
"""


def parse_bc_type(bc_id) -> 'BCType':
    """ return BasisType(Enum) given accepted basis_id (strs)
    """
    if isinstance(bc_id, BCType):
        return bc_id

    if bc_id in ['periodic', 'p', 'P']:
        return BCType.PERIODIC
    elif bc_id in ['antiperiodic', 'ap', 'AP']:
        return BCType.ANTIPERIODIC
    elif bc_id in ['zero gradient', 'zg', 'ZG']:
        return BCType.ZEROGRADIENT
    elif bc_id in ['zero value', 'zv', 'ZV']:  # previosuly 'r','R','reflecting','reflect']
        return BCType.ZEROVALUE
    elif bc_id in ['reflecting', 'r', 'R']:
        return BCType.REFLECTING
    elif bc_id in ['absorbing', 'a', 'A']:
        return BCType.ABSORBING
    else:
        return BCType.OPEN


def parse_basis_type(basis_id) -> 'BasisType':
    """ return BasisType(Enum) given accepted basis_id (strs)
    """
    if isinstance(basis_id, BasisType):
        return basis_id

    # if basis_id in ['cart','cartesian','Cartesian']:
    #     return BasisType.CARTESIAN
    # elif basis_id in ['cyl','cylindrical','Cylindrical']:
    #     return BasisType.CYLINDRICAL
    # elif basis_id in ['sph','spherical','Spherical']:
    #     return BasisType.CYLINDRICAL

    if basis_id in ['spatial', 'Spatial']:
        return BasisType.SPATIAL
    elif basis_id in ['k', 'fourier', 'Fourier']:
        return BasisType.FOURIER
    elif basis_id in ['H', 'Hermite']:
        return BasisType.HERMITE


def get_basis(basis_type) -> 'Basis':
    """ return basis class given BasisType(Enum)
    """
    if not isinstance(basis_type, Enum):
        basis_type = parse_basis_type(basis_type)

    # if basis_type == BasisType.CARTESIAN:
    #     from basis.basis_cart import CartesianBasis
    #     return CartesianBasis
    # if basis_type == BasisType.CYLINDRICAL:
    #     from basis.basis_cyl import CylindricalBasis
    #     return CylindricalBasis
    # if basis_type == BasisType.SPHERICAL:
    #     import basis.basis_sph as SphericalBasis
    #     return SphericalBasis

    if basis_type == BasisType.SPATIAL:
        from basis.basis_spatial import SpatialBasis
        return SpatialBasis()
    elif basis_type == BasisType.FOURIER:
        from basis.basis_k import FourierBasis
        return FourierBasis()
    elif basis_type == BasisType.HERMITE:
        from basis.basis_hermite import HermiteBasis
        return HermiteBasis()


@dataclass(unsafe_hash=True)
class Axis:
    _frozen = False

    def __init__(self, L: int, q: int = 2, coordinate: 'Coordinate' = None,
                 x0: int = 0, dx: Numeric = 1, xpts: Optional[np.ndarray] = None,
                 endpoint=False, startpoint=True,
                 basis: 'Basis' = None, is_flipped=False, ax_map: 'AxisMap' = None,
                 base_coarseness=0):
        """ Inputs:
                axID:   Coordinate of Axis. Can be shared across different Axis objects
                q, L:   determines vector size (L-dimensional tensor with size of q)
                x0:     value of first point in discretization (ie. xpts[0]) (default = 0)
                dx:     spacing of grid points (default = 1)
                xpts:   an np.array specifying the actual points in the discretization
                        allows for non-even discretizations. must be of length q**L
                        (default = None)
                basis_type:  basis class identifier (BasisType or one of accepted strings)
                        (default = BasisType.CARTESIAN)

            Attributes:
                coordinate:    coordinate identifier
                q, L:    parameters of vector size, q**L
                basis:   basis class
                xpts:    x-points along this dimension. must be monotically increasing, but can be unevenly spaced
                dx:      spacing of xpts. either int (if is_even) or np.array (not is_even)
                is_even: if discretization is evenly spaced
                is_flipped:  if True, tensors are ordered fine to coarse (left to right)
                             else, tensors are ordered coarse to fine (right to left)

        """
        self.coordinate = coordinate
        self.q = q
        self.L = L
        self.npts = q ** L
        self._get_xpts(x0, dx, xpts)
        self.is_flipped = is_flipped
        self.endpoint = endpoint
        self.startpoint = startpoint
        self.base_coarseness = base_coarseness
        self._xmult_mpos = {}

        if ax_map is None:
            ax_map = BinaryMap()
        self.map = ax_map

        # self.basis_type = parse_basis_type(basis)
        if basis is None:
            self.basis = SpatialBasis()
        else:
            self.basis = basis

        ## assigned when included in a CoordinateSystem object
        self.coord_sys = CoordinateSystem('_DUMMY', self)

        zero_ind = np.argmax(self.xpts >= 0)  ## assumes xpts are monotonically increasing
        if self.xpts[zero_ind] != 0:
            if zero_ind > 0:  ## return crossover from neg to positive values
                zero_ind = (zero_ind - 1, zero_ind)
            else:  ## 0 doesn't exist
                zero_ind = None
        self.zero_ind = zero_ind

        self._frozen = True

    def __str__(self):
        # return '(' + self.coord_sys.coordID + ',' + str(self.coordinate) + ')'
        return f'Axis({self.L},{self.q},{self.coordinate})'

    def __repr__(self):
        out_str = f'Axis({self.L},{self.q},{self.coordinate})'
        return out_str

    def __delattr__(self, item):
        if self._frozen:
            raise AttributeError('not allowed to alter Axis attributes')
        object.__delattr__(self, item)

    def __setattr__(self, key, value):
        if self._frozen:
            if key == 'coord_sys' and self.coord_sys.coordID == '_DUMMY':
                object.__setattr__(self, key, value)
            else:
                raise AttributeError('not allowed to alter Axis attributes')
        object.__setattr__(self, key, value)

    def __eq__(self, other: 'Axis'):
        if DEEP_GRID_CHECK:
            return self is other
        else:
            return repr(self) == repr(other)

    ### use default
    # def __eq__(self,other:'Axis'):
    #     bool0 = (self.axID == other.axID)
    #     bool1 = (self.basis_type == other.basis_type)
    #     bool2 = (self.q == other.q)
    #     bool3 = (self.L == other.L)
    #     bool4 = (self.xpts[0] == other.xpts[0])
    #     bool5 = (self.dx == self.dx)
    #     bool6 = (self.coord_type == other.coord_type)
    #     return bool0 and bool1 and bool2 and bool3 # and bool4

    def shape(self) -> tuple[int,...]:
        return (self.q,) * self.L

    def _get_xpts(self, x0, dx, xpts) -> np.ndarray:
        if xpts is None:
            self.is_even = True
            self.dx = dx
            self.xpts = np.arange(self.q ** self.L) * dx + x0
            self.x0 = x0
        else:
            diff = np.diff(xpts)
            self.xpts = xpts
            assert (len(xpts) == self.q ** self.L), f'provided xpts must be of length {self.q ** self.L}'

            if np.all(np.abs(diff - diff[0]) < 1.0E-10 * diff):
                self.is_even = True
                self.dx = diff[0]
                self.x0 = xpts[0]
            else:
                self.is_even = False
                self.dx = diff
                self.x0 = xpts[0]
        return self.xpts

    def create_like(self, new_coord: Coordinate):  # , new_coord_sys=None, new_coord_type=None):
        """ create new Axis object, potentially with new coordinate
        """
        new_ax = self.__class__(self.L, self.q, coordinate=new_coord, basis=self.basis, xpts=self.xpts,
                                endpoint=self.endpoint, startpoint=self.startpoint, ax_map=self.map,
                                # is_flipped=self.is_flipped)
                                )
        # new_ax.xpts = self.xpts
        # new_ax.dx = self.dx
        # new_ax.is_even = self.is_even
        # new_ax.zero_ind = self.zero_ind
        return new_ax

    def create_new(self, L, q=None, xpts=None):  # , new_coord_sys=None, new_coord_type=None):
        """ create new Axis object, potentially with new coordinate
        """
        q = self.q if q is None else q
        L = self.L if L is None else L
        xpts = np.linspace(0, 1, q**L, endpoint=self.endpoint) if xpts is None else xpts

        new_ax = self.__class__(L, q, coordinate=self.coordinate, basis=self.basis, ax_map=self.map,
                                xpts=xpts, endpoint=self.endpoint, startpoint=self.startpoint,
                                )
        return new_ax

    def get_val_ind(self, val: Numeric) -> Optional[int]:
        """ find index (int) of grid poin corresponding to desired value val along axis ax
        """
        ind = np.argwhere(np.abs(self.xpts - val) < 1.0 - 10)
        try:
            return ind[0]
        except IndexError:
            print('value', val, 'not found in grid')
            return None

    def get_position_inds(self, idx: int) -> list[int]:
        """ returns physical bond indices (0,1) of MPS that corresponds to vector position idx
        """
        return self.map.get_position_inds(self.L, self.q, idx)

    def get_inds_position(self, inds_list: Iterable[int]):  # , ax_map: 'AxisMap' = None) -> int:
        """ returns index in vector corresponding to physical bond indices specified in inds_list
        """
        return self.map.get_inds_position(self.q, inds_list)

    def get_coarseness_ind(self, coarseness_level):
        """ for multigrid methods. finer grid --> smaller number (coarsened first)
            return # indexing tensor corresponding to desired coarseness level
        """
        c_level = coarseness_level - self.base_coarseness
        if c_level < 0 or c_level >= self.L:
            return None
        else:
            if self.map.is_flipped():
                return c_level
            else:
                return self.L - 1 - c_level

    def max_coarseness(self):
        return self.base_coarseness + self.L

    def is_k(self):
        if isinstance(self.basis, FourierBasis):
            return True
        return False

    def is_real_k(self):
        if isinstance(self.basis, RealFourierBasis):
            return True
        return False

    def length(self):
        if isinstance(self.basis, SpatialBasis):
            return self.xpts[-1] - self.xpts[0]
        elif isinstance(self.basis, FourierBasis):
            return 2 * np.pi / self.dx
        else:
            raise NotImplementedError

    def get_constant_ind(self):
        """ if is constant, select this element
        """
        if isinstance(self.basis, FourierBasis):
            return self.npts // 2
        elif isinstance(self.basis, RealFourierBasis):
            return 0
        elif isinstance(self.basis, SpatialBasis):
            return 0
        else:
            raise NotImplementedError

    ###########################
    ## extract info from MPS ##
    ###########################

    def meas_elem(self, ket: 'qtn.MatrixProductState', ind: int):
        """ measure the element in ket as indexed by ind
        """
        select_mps = self.get_select_elems_mps([ind])
        return helper.ovlp(ket, select_mps)

    ######################
    ## build common TNs ##
    ######################

    def get_all_ones_mps(self, site_ind_id='i({})', site_tag_id='X({})') -> MPSType:
        """ build ones vector mps in shape of self
            TODO: rename this and get_ones_mps??
        """
        return get_ones_mps(self.L, self.q, site_ind_id, site_tag_id)

    def get_iden_mps(self, site_ind_id='i({})', site_tag_id='X({})',
                     anc_dim=None, anc_name_l=None, anc_name_r=None) -> MPSType:
        """ build ones vector mps in shape of self
        """
        ## takes map into account
        return self.basis.get_ones_mps(self, site_ind_id, site_tag_id, anc_dim, anc_name_l, anc_name_r)
        # return get_ones_mps(self.L, self.q, site_ind_id, site_tag_id)

    def get_iden_mpo(self, upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='X({})',
                     anc_dim=None, anc_name_l=None, anc_name_r=None, L:int =None) -> MPOType:
        """ build identity mpo
        """
        L = self.L if L is None else L
        if anc_dim is None:
            return get_iden_mpo(L, self.q, upper_ind_id, lower_ind_id, site_tag_id)
        else:
            return get_iden_mpo_anc(L, self.q, upper_ind_id, lower_ind_id, site_tag_id, anc_dim,
                                    anc_name_l, anc_name_r)

    def get_select_elems_mps(self, inds: Iter[int], site_ind_id='i({})', site_tag_id='X({})',
                             compress=False, compress_opts: dict = None, deriv_config: 'DerivativeConfiguration'=None
                             ) -> MPSType:
        """ build MPS to select certain elements specified by inds
        """
        # ind_list = self.get_position_inds(inds[0])
        # print('select ind', inds[0], ind_list, self.map)
        mps = None   # get_select_elem_mps(self.L, self.q, ind_list, site_ind_id, site_tag_id)
        for ind in inds:

            sign = 1
            if deriv_config is not None:
                if ind < 0:
                    if deriv_config.left_bc in [BCType.ZEROGRADIENT, BCType.ZEROVALUE, BCType.ANTISYMMETRIC,
                                                BCType.SYMMETRIC, BCType.NEUMANN, BCType.DIRICHLET]:
                        ind = abs(ind)
                    elif deriv_config.left_bc in [BCType.PERIODIC, BCType.ANTIPERIODIC]:
                        ind = ind % self.npts
                    sign = np.sign(deriv_config.left_bc.value)
                elif ind >= self.npts:
                    if deriv_config.right_bc in [BCType.ZEROGRADIENT, BCType.ZEROVALUE, BCType.ANTISYMMETRIC,
                                                BCType.SYMMETRIC, BCType.NEUMANN, BCType.DIRICHLET]:
                        ind = abs(ind)
                    elif deriv_config.right_bc in [BCType.PERIODIC, BCType.ANTIPERIODIC]:
                        ind = ind % self.npts
                    sign = np.sign(deriv_config.right_bc.value)

            ind_list = self.get_position_inds(ind)
            mps_ = get_select_elem_mps(self.L, self.q, ind_list, site_ind_id, site_tag_id)
            helper.scalar_multiply(mps_, sign, inplace=True)
            if mps is not None:
                helper.add_MPS(mps, mps_, inplace=True, compress=False)
            else:
                mps = mps_

        if compress:
            if compress_opts is None:   compress_opts = {}
            helper.compress(mps, **compress_opts)

        return mps

    def get_select_elems_mpo(self, inds: Iter[int], upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='X({})',
                             compress=False, compress_opts: dict = None) -> MPOType:
        """ build MPO to select certain elements specified by inds
        """
        ind_list = self.get_position_inds(inds[0])
        mpo = get_select_elem_mpo(self.L, self.q, ind_list, upper_ind_id, lower_ind_id, site_tag_id, )
        for ind in inds[1:]:
            ind_list = self.get_position_inds(ind)
            mpo_ = get_select_elem_mpo(self.L, self.q, ind_list, upper_ind_id, lower_ind_id, site_tag_id, )
            helper.add_MPO(mpo, mpo_, inplace=True, compress=False)

        if compress:
            if compress_opts is None:   compress_opts = {}
            helper.compress(mpo, **compress_opts)

        return mpo

    def get_all_except_elem_mps(self, inds: Iter[int], site_ind_id='i({})', site_tag_id='X({})') -> MPSType:
        """ build MPO in which all elements are 1 except at element in inds which are 0
        """
        out = self.get_all_ones_mps(site_ind_id, site_tag_id)
        select_mps = self.get_select_elems_mps(inds, site_ind_id, site_tag_id)
        helper.scalar_multiply(select_mps, -1, inplace=True)
        helper.add_MPS(out, select_mps, inplace=True, compress=False)
        return out

    def get_all_except_elem_mpo(self, inds: Iter[int], upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='X({})') \
            -> MPOType:
        """ build MPS in which all elements along diagonal are 1 except at element in inds which are 0
        """
        out = self.get_iden_mpo(upper_ind_id, lower_ind_id, site_tag_id)
        select_mpo = self.get_select_elems_mpo(inds, upper_ind_id, lower_ind_id, site_tag_id)
        helper.scalar_multiply(select_mpo, -1, inplace=True)
        helper.add_MPO(out, select_mpo, inplace=True, compress=False)
        return out

    def get_heaviside_mpo(self, zero_ind: int, get='right') -> MPOType:
        """ get Heaviside step function where side denoted by 'get' is 1. x denotes that zero_ind
            is excluded
        """
        out = get_heaviside_mpo(self.L, self.q, zero_ind, get)
        out = self.map.transform_mpo(out)
        return out

    def get_reverse_diag_mpo(self, upper_ind_id: str = 'o{}', lower_ind_id: str = 'i{}', site_tag_id: str = 'T{}') \
            -> MPOType:
        """ build MPO with 1's along backward diagonal
        """
        out = get_reverse_diag_mpo(self.L, self.q, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                   site_tag_id=site_tag_id)
        out = self.map.transform_mpo(out)
        return out

    def get_shift_mpo(self, shift: int, boundary_conditions: 'DerivativeConfiguration' = None,
                      upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='X({})',
                      L: int=None) -> MPOType:
        """ build MPO with 1's along diagonal shifted by k
        """
        L = self.L if L is None else L
        # print('get shift mpo', self, boundary_conditions)
        out = get_shift_mpo(L, self.q, shift, boundary_conditions=boundary_conditions,
                            upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)
        out = self.map.transform_mpo(out)
        return out

    def get_tridiag_mpo(self, a, b, c, boundary_conditions: 'DerivativeConfiguration' = None,
                        upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='X({})',) -> MPOType:
        """ diag(a) + diag(b,1) + diag(c,-1)
        """
        m0 = self.get_shift_mpo( 0, boundary_conditions=boundary_conditions,
                                 upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)
        m1 = self.get_shift_mpo( 1, boundary_conditions=boundary_conditions,
                                 upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)
        m2 = self.get_shift_mpo(-1, boundary_conditions=boundary_conditions,
                                upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)

        helper_quimb.scalar_multiply(m0, a, inplace=True)
        helper_quimb.scalar_multiply(m1, b, inplace=True)
        helper_quimb.scalar_multiply(m2, c, inplace=True)

        out_mpo = m0.copy()
        out_mpo = helper_quimb.add_MPO(out_mpo, m1)
        out_mpo = helper_quimb.add_MPO(out_mpo, m2)
        out_mpo = helper.compress(out_mpo)

        # print('tridiag', a, b, c)
        # np.set_printoptions(2)
        # print(self.map_mpo_to_operator(out_mpo))

        # k_axis = self.create_like(self.coordinate)
        # k_axis.basis = FourierBasis()
        # # conv_op = k_axis.get_elemental_multiply_tn()
        # vec = np.zeros((self.npts,))
        # vec[self.npts // 2] = a
        # vec[self.npts // 2 + 1] = b
        # vec[self.npts // 2 - 1] = c
        # vec_mps = self.map_state_to_mps(vec)
        # out_mpo = k_axis.apply_elemental_multiply_op( vec_mps, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id )
        return out_mpo



    #########################################
    ## convert between np.ndarray and MPX ##
    #########################################

    def map_state_to_mps(self, state, site_ind_id: str = 'i({})', site_tag_id: str = 'T({})',
                         split_opts: dict = None) -> MPSType:
        """ convert state represented as vector converted to mps state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        ## i think transform mps does not require us to transform vector array
        # # state = self.axis_map.transform_array(state)   ## assume data only along axis dimension
        out = map_state_to_mps(self.L, self.q, state, site_ind_id, site_tag_id, split_opts=split_opts)
        out = self.map.transform_mps(out)
        return out

    def map_mps_to_state(self, mps) -> np.ndarray:
        """ convert MPS into ndim-dim np.ndarray
        """
        mps = self.map.inverse_transform_mps(mps)
        return map_mps_to_state(mps)

    def map_operator_to_mpo(self, operator, upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                            site_tag_id: str = 'T({})', split_opts: dict = None) -> MPOType:
        """ convert operator represented as high-dimensional matrix to mpo state
            here, returns original vec as a single tensor as quimb.TensorNetwork object
        """
        out = map_operator_to_mpo(self.L, self.q, operator, upper_ind_id, lower_ind_id, site_tag_id,
                                  split_opts=split_opts)
        out = self.map.transform_mpo(out)
        return out

    def map_mpo_to_operator(self, mpo) -> np.ndarray:
        """ convert MPO into ndim*2-dim np.ndarray (o0 o1 ... x i0 i1 ...)
        """
        mpo = self.map.inverse_transform_mpo(mpo)
        return map_mpo_to_operator(mpo, flip=self.is_flipped)

    ##################################
    ## build derivatives, xmultiply ##
    ##################################

    def build_firstderivative_mpo(self, deriv_config: 'DerivativeConfiguration') -> MPOType:
        """ build MPO that takes first derivative
            deriv_kwargs:
                for Spatial Basis:  bc (boundary condition), order, fd_type
            (mpo already flipped/transformed in basis method if need be)
        """
        if not self.is_even and isinstance(self.basis, SpatialBasis):
            raise NotImplementedError('first derivative for uneven dx for spatial basis not supported')

        if deriv_config.fd_type == FDType.FVM:
            ## derivative for finite volume
            # mpo = self.get_iden_mpo()
            mpo_m1 = helper.scalar_multiply(self.get_shift_mpo(-1, boundary_conditions=deriv_config), -1)
            # mpo_m2 = helper.scalar_multiply(self.get_shift_mpo(-2, boundary_conditions=deriv_config), -1)
            mpo_p1 = helper.scalar_multiply(self.get_shift_mpo( 1, boundary_conditions=deriv_config),  1)
            # mpo = helper.add_MPO(mpo, mpo_m1)
            # mpo = helper.add_MPO(mpo, mpo_m2)
            # mpo = helper.add_MPO(mpo, mpo_p1)
            mpo = helper.add_MPO(mpo_m1, mpo_p1)
            helper.compress(mpo)
            mpo = helper.scalar_multiply(mpo, 1./ 2 / self.dx, inplace=True)

            # data = self.map_mpo_to_operator(mpo)
            # plt.figure()
            # plt.imshow(data)
            # plt.show()

        else:
            mpo = self.basis.build_firstderivative_mpo(self, deriv_opts=deriv_config.deriv_params)
            # print('mpo', self, deriv_config.order, deriv_config.fd_type, mpo.max_bond())

        return mpo


    def build_firstderivative_mpo_inverse(self, deriv_config: 'DerivativeConfiguration') -> MPOType:
        """ build MPO that takes first derivative
            deriv_kwargs:
                for Spatial Basis:  bc (boundary condition), order, fd_type
            (mpo already flipped/transformed in basis method if need be)
        """
        # if not self.is_even and isinstance(self.basis, SpatialBasis):
        #     raise NotImplementedError('first derivative for uneven dx for spatial basis not supported')

        print('deriv config', deriv_config.deriv_params)
        mpo = self.basis.build_firstderivative_mpo_inverse(self, deriv_opts=deriv_config.deriv_params)
        # print('mpo', self, deriv_config.order, deriv_config.fd_type, mpo.max_bond())

        return mpo


    def build_secondderivative_mpo(self, deriv_config: 'DerivativeConfiguration'=None, eeo_grid=False) -> MPOType:
        """ deriv_kwargs:
                for Spatial Basis:  bc (boundary condition), order, fd_type
            (mpo already flipped/transformed in basis method if need be)
        """
        if not self.is_even and isinstance(self.basis, SpatialBasis):  ## real space method
            raise NotImplementedError('second derivative for uneven dx for spatial basis not supported')
        mpo = self.basis.build_secondderivative_mpo(self, deriv_config.deriv_params, eeo_grid=eeo_grid)
        return mpo

    def build_mth_derivative_mpo(self, deriv_order: int, deriv_config: 'DerivativeConfiguration'=None, eeo_grid=False) -> MPOType:
        """ deriv_kwargs:
                for Spatial Basis:  bc (boundary condition), order, fd_type
            (mpo already flipped/transformed in basis method if need be)
        """
        if not self.is_even and isinstance(self.basis, SpatialBasis):  ## real space method
            raise NotImplementedError('second derivative for uneven dx for spatial basis not supported')
        deriv_params = deriv_config.deriv_params if deriv_config is not None else None
        mpo = self.basis.build_mth_derivative_mpo(self, deriv_order, deriv_opts=deriv_params,
                                                  eeo_grid=eeo_grid)
        return mpo

    def get_xmultiply_mps(self, x_power: int = 1, offset: Numeric = 0.0, scale: Numeric = 1.0, split_opts: dict = None) \
            -> MPOType:
        """ x * f(x) MPO
            (mpo already flipped/transformed in basis method if need be)
        """
        mpo = self.basis.build_xmultiply_mps(self, x_power=x_power, offset=offset, scale=scale, split_opts=split_opts)
        return mpo

    def get_xmultiply_mpo(self, x_power: int = 1, offset: Numeric = 0.0, scale: Numeric = 1.0, split_opts: dict = None) \
            -> MPOType:
        """ x * f(x) MPO
            (mpo already flipped/transformed in basis method if need be)
        """
        try:
            mpo = self._xmult_mpos[(x_power, offset, scale)]
        except KeyError:
            mpo = self.basis.build_xmultiply_mpo(self, x_power=x_power, offset=offset, scale=scale, split_opts=split_opts)
            self._xmult_mpos[(x_power, offset, scale)] = mpo
        return mpo

    def get_fmultiply_mpo(self, vec_data: np.ndarray) \
            -> MPOType:
        """ x * f(x) MPO
            (mpo already flipped/transformed in basis method if need be)
        """
        mpo = self.basis.build_fmultiply_mpo(self, vec_data)
        return mpo

    def get_elemental_multiply_tn(self, in1_ind_id: str = 'i({})[1]', in2_ind_id: str = 'i({})[2]',
                                  out_ind_id: str = 'o({})', site_tag_id='d_ijk({})', cutoff=CUTOFF) -> MPTType:
        """ x * f(x) MPO
            (tn1d already flipped/transformed in basis method if need be)
        """
        tn1d = self.basis.build_elemental_multiply_tn(self, in1_ind_id, in2_ind_id, out_ind_id, site_tag_id,
                                                      cutoff=cutoff)
        return tn1d

    def get_diagonalize_mps_tn(self, in1_ind_id: str = 'i({})[1]', in2_ind_id: str = 'i({})[2]',
                               out_ind_id: str = 'o({})', site_tag_id='d_ijk({})') -> MPTType:
        """ convert MPS to MPO
            (tn1d already flipped/transformed in basis method if need be)
        """
        tn1d = SpatialBasis().build_elemental_multiply_tn(self, in1_ind_id, in2_ind_id, out_ind_id, site_tag_id)
        return tn1d

    def apply_elemental_multiply_op(self, mps, upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                                    site_tag_id: str = 'T({})', compress=False, compress_opts=None,
                                    elem_mult_tn = None) -> MPOType:
        if mps.site_ind_id == upper_ind_id:
            mps = mps.copy()
            mps.site_ind_id = mps.site_ind_id + '_tmp'

        if mps.site_ind_id == lower_ind_id:
            mps.copy()
            mps.site_ind_id = mps.site_ind_id + '_tmp'

        if elem_mult_tn is None:
            elem_mult_tn = self.get_elemental_multiply_tn(in1_ind_id=lower_ind_id, in2_ind_id=mps.site_ind_id,
                                                          out_ind_id=upper_ind_id, site_tag_id=site_tag_id)

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

    def get_integral_mps(self, site_ind_id: str = 'i({})', site_tag_id: str = 'T({})') -> MPSType:
        """ integral (reimann sum)
            (mps already flipped/transformed in basis method if need be)
        """
        if not self.is_even and isinstance(self.basis, SpatialBasis):
            raise NotImplementedError
        return self.basis.build_integral_mps(self, site_ind_id=site_ind_id, site_tag_id=site_tag_id)


    def get_coarse_grain_mpx(self, depth: int, upper_ind_id: str='i({})', lower_ind_id: str='o({})',
                             site_tag_id: str='T(}})') -> MPOType:
        """ partial integral
        """
        return self.basis.build_coarse_grain_mpx(self, depth, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                                 site_tag_id=site_tag_id)

    def get_coarse_select_mpx(self, depth: int, upper_ind_id: str = 'i({})', lower_ind_id: str = 'o({})',
                             site_tag_id: str = 'T(}})') -> MPOType:
        """ partial integral
        """
        return self.basis.build_coarse_select_mpx(self, depth, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id,
                                                 site_tag_id=site_tag_id)

    def get_integral_weight(self):
        return self.basis.get_integral_weight(self)


    def get_averaging_mpo(self, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                          boundary_conditions: 'DerivativeConfiguration' = None,
                          L:int = None, mu:float =0.5, spread=1) -> MPOType:
        """ if spread=1
                x_j <-- (1-mu)/2 x_(j-1) + mu x_(j) + (1-mu)/2 x_(j+1)
            else:
                x_j <-- \sum_i=1^spread (1-mu)/2 spread (x_(j-i) + x_(j+i)) + mu x_(j)
        """
        weight = mu

        k0 = self.get_iden_mpo(upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id, L=L)
        k0 = helper.scalar_multiply(k0, weight)
        out = k0

        for i in range(1, spread + 1):
            kp = self.get_shift_mpo(i, boundary_conditions=boundary_conditions, L=L)
            km = self.get_shift_mpo(-i, boundary_conditions=boundary_conditions, L=L)

            weight = (1 - mu) if spread == 1 else (1 - mu) / spread
            kp = helper.scalar_multiply(kp, weight / 2)
            km = helper.scalar_multiply(km, weight / 2)

            out = helper.add_MPO(out, kp)
            out = helper.add_MPO(out, km)

        out.compress()
        return out


    def get_neighbor_avg_mpo(self, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                             boundary_conditions: 'DerivativeConfiguration' = None,
                             L:int = None, mu:float =0.5, spread=1) -> MPOType:
        """ if spread=1
                x_j <-- (1-mu)/2 x_(j-1) + mu x_(j) + (1-mu)/2 x_(j+1)
            else:
                x_j <-- \sum_i=1^spread (1-mu)/2 spread (x_(j-i) + x_(j+i)) + mu x_(j)
        """
        weight = mu

        k0 = self.get_iden_mpo(upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id, L=L)
        k0 = helper.scalar_multiply(k0, weight)
        out = k0

        for i in range(1, spread + 1):
            # kp = self.get_shift_mpo(i, boundary_conditions=boundary_conditions, L=L)
            km = self.get_shift_mpo(-i, boundary_conditions=boundary_conditions, L=L)

            weight = (1 - mu) if spread == 1 else (1 - mu) / spread
            # kp = helper.scalar_multiply(kp, weight)
            km = helper.scalar_multiply(km, weight)

            # out = helper.add_MPO(out, kp)
            out = helper.add_MPO(out, km)

        out.compress()
        return out


    def get_leapfrog_mpo(self, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                          boundary_conditions: 'DerivativeConfiguration' = None,
                          L:int = None, spread=1) -> MPOType:
        """ x_j <-- 0.5 x_(j-1) + 0.5 x_(j+1)
        """
        kp = self.get_shift_mpo(1, boundary_conditions=boundary_conditions, L=L)
        km = self.get_shift_mpo(-1, boundary_conditions=boundary_conditions, L=L)

        out = helper.add_MPO(km, kp)
        helper.scalar_multiply(out, 0.5, inplace=True)
        out.compress()

        return out


    def get_qft_mpo_v2(self, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                       inverse=False, flip_lr=False, cutoff=CUTOFF):
        """ obtain MPO that takes discrete Fourier transform
            following Jielun Chen paper format
        """
        assert self.is_even, 'QFT not defined for uneven axis spacing'

        qft_mpo = get_qft_operator(self.L, self.q, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                                   lower_ind_id=lower_ind_id, inverse=inverse, cutoff=cutoff)

        if flip_lr:
            qft_mpo = helper.mpo_flip_lr(qft_mpo)

        qft_mpo = self.map.transform_mpo(qft_mpo)
        # helper.scalar_multiply(qft_mpo, 1./np.sqrt(self.npts), inplace=True)

        return qft_mpo

    def get_qft_freqs(self):
        k_bound = np.pi / self.dx
        k_vals = np.linspace(-k_bound, k_bound, self.npts)
        ks_mps = self.map_state_to_mps(k_vals)
        ks_mps = self.map.transform_mps(ks_mps)
        ks_mps = helper.mps_flip_lr(ks_mps)  ## qft flips tens order
        return ks_mps


    ## old QFT stuff
    def get_qft_mpo(self, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',):
        """ obtain MPO that takes discrete Fourier transform
        """
        assert self.is_even, 'QFT not defined for uneven axis spacing'

        qft_mpo = get_qft_mpo(self.L, self.q, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                                   lower_ind_id=lower_ind_id)
        qft_mpo = self.map.transform_mpo(qft_mpo)
        # helper.scalar_multiply(qft_mpo, 1./np.sqrt(self.npts), inplace=True)

        return qft_mpo

    # def get_qft_mpo_pad_k(self, num_pad=1, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})',
    #                       lower_ind_id: str = 'i({})'):
    #     """ obtain MPO that takes discrete Fourier transform
    #     """
    #     assert self.is_even, 'QFT not defined for uneven axis spacing'
    #     qft_mpo = get_qft_operator(self.L + num_pad, self.q, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
    #                                lower_ind_id=lower_ind_id)
    #
    #     ## removed interpolated finer grid in real space
    #     for x in range(num_pad):
    #         tens_x = qft_mpo[self.L + x]
    #         tens_x.isel({lower_ind_id.format(self.L + x): 0}, inplace=True)
    #
    #     qft_mpo = self.map.transform_mpo(qft_mpo, upper_L=self.L + num_pad, lower_L=self.L)
    #     # helper.scalar_multiply(qft_mpo, np.sqrt(self.npts), inplace=True)
    #
    #     return qft_mpo

    def get_qft_ks(self, num_pad=0):
        """ get k's associated with qft in vector form
        """
        assert num_pad >= 0, 'padding k must be >= 0'

        L = self.L + num_pad
        kvals = np.fft.fftfreq(self.q ** L) * 2 * np.pi / self.dx

        ## set padded k's to zero
        if num_pad > 0:
            kmax = (self.npts / 2) * 2 * np.pi / self.dx
            kvals = np.where(kvals >= kmax, np.zeros_like(kvals), kvals)
            kvals = np.where(kvals < -kmax, np.zeros_like(kvals), kvals)

        kvals = kvals.reshape((self.q,) * L).transpose().reshape(-1)  # for binary rep
        ## kvals = self.map.transform_vector(kvals)
        return kvals

    def get_inverse_qft_mpo(self, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})'):
        """ obtain MPO that takes discrete Fourier transform
        """
        assert self.is_even, 'QFT not defined for uneven axis spacing'

        qft_mpo = get_inverse_qft_mpo(self.L, self.q, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                                      lower_ind_id=lower_ind_id)
        qft_mpo = self.map.inverse_transform_mpo(qft_mpo)
        # helper.scalar_multiply(qft_mpo, 1./self.npts, inplace=True)

        return qft_mpo
    #
    # def get_inverse_qft_mpo_pad_k(self, num_pad=1, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})',
    #                               lower_ind_id: str = 'i({})'):
    #     """ obtain MPO that takes discrete Fourier transform
    #     """
    #     assert self.is_even, 'QFT not defined for uneven axis spacing'
    #     qft_mpo = get_inverse_qft_mpo(self.L + num_pad, self.q, site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
    #                                   lower_ind_id=lower_ind_id)
    #
    #     ## removed interpolated finer grid in real space
    #     for x in range(num_pad):
    #         tens_x = qft_mpo[self.L + x]
    #         tens_x.isel({lower_ind_id.format(self.L + x): 0}, inplace=True)
    #
    #     qft_mpo = self.map.transform_mpo(qft_mpo, lower_L=self.L + num_pad, upper_L=self.L)
    #     # helper.scalar_multiply(qft_mpo, np.sqrt(self.npts), inplace=True)
    #
    #     return qft_mpo


######################
## static functions ##
######################

def get_ones_mps(L: int, q: int, site_ind_id: str = 'i({})', site_tag_id: str = 'X({})') -> MPSType:
    """ build ones vector mps
    """
    ones = np.ones((q,))
    mps_0 = None
    if L >= 2:
        mps_0 = qtn.MatrixProductState(
            [np.array([ones])] + [np.array([[ones]])] * (L - 2) + [np.array([ones])],
            shape='lrp', site_tag_id=site_tag_id, site_ind_id=site_ind_id)
    elif L == 1:
        mps_0 = qtn.TensorNetwork([qtn.Tensor(ones, inds=('i(0)',), tags=('X(0)',))])
        mps_0.view_as(qtn.MatrixProductState, inplace=True, L=1, cyclic=False,
                      site_tag_id=site_tag_id, site_ind_id=site_ind_id)
    elif L == 0:
        mps_0 = qtn.TensorNetwork([])
        mps_0.view_as(qtn.MatrixProductState, inplace=True, L=0, cyclic=False,
                      site_tag_id=site_tag_id, site_ind_id=site_ind_id)
    return mps_0


def get_iden_mpo(L: int, q: int, upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                 site_tag_id: str = 'X({})') -> MPOType:
    """ build identity mpo
    """
    mpo_0 = None
    if L >= 2:
        iden = np.eye(q)
        mpo_0 = qtn.MatrixProductOperator(
            [np.array([iden])] + [np.array([[iden]])] * (L - 2) + [np.array([iden])],
            shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    elif L == 1:
        # iden = np.eye(q)
        iden = scipy.sparse.eye(q, dtype='d', format='coo')
        mpo_0 = qtn.TensorNetwork([qtn.Tensor(iden, inds=(upper_ind_id.format(0), lower_ind_id.format(0)),
                                              tags=(site_tag_id.format(0),))])
        mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                      site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    elif L == 0:
        mpo_0 = qtn.TensorNetwork([])
        mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                      site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    return mpo_0


def get_iden_mpo_anc(L: int, q: int, upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                     site_tag_id: str = 'X({})', anc_dim: int = 1, anc_name_l: str = None,
                     anc_name_r: str = None) -> MPOType:
    """ build identity mpo with ancilla at the two ends
    """
    iden = np.eye(q * anc_dim).reshape(anc_dim, q, anc_dim, q).transpose(0, 2, 1, 3)
    mpo_0 = None

    anc_name_l = 'anc({})_L' if anc_name_l is None else anc_name_l
    anc_name_r = 'anc({})_R' if anc_name_r is None else anc_name_r

    if L >= 2:
        T0 = qtn.Tensor(iden, inds=(anc_name_l, 'vb0', upper_ind_id.format(0), lower_ind_id.format(0)),
                        tags=(site_tag_id.format(0),))
        TL = qtn.Tensor(iden, inds=(f'vb{L - 2}', anc_name_r, upper_ind_id.format(L - 1), lower_ind_id.format(L - 1)),
                        tags=(site_tag_id.format(L - 1),))
        tensors = [T0] + [
            qtn.Tensor(iden, inds=(f'vb{i - 1}', f'vb{i}', upper_ind_id.format(i), lower_ind_id.format(i),),
                       tags=(site_tag_id.format(i))) for i in range(1, L - 1)] + [TL]
        mpo_0 = qtn.TensorNetwork(tensors)
        mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=L, cyclic=False,
                      site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
        mpo_0.mangle_inner_()
    elif L == 1:
        mpo_0 = qtn.TensorNetwork([qtn.Tensor(iden, inds=(anc_name_l, anc_name_r, upper_ind_id.format(0),
                                                          lower_ind_id.format(0),),
                                              tags=(site_tag_id.format(0),))])
        mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                      site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    elif L == 0:
        mpo_0 = qtn.TensorNetwork([])
        mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                      site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)

    return mpo_0


def get_select_elem_mps(L: int, q: int, inds_list: Sequence[int], site_ind_id: str = 'i({})',
                        site_tag_id: str = 'X({})') \
        -> MPSType:
    """ build MPS to select certain elements specified by inds
        inds_list:  binary strings
    """
    mdict = {}
    for b in range(q):
        mvec = np.zeros((q,))
        mvec[b] = 1.
        mdict[b] = mvec

    mps_0 = None
    if L >= 2:
        ms = [mdict[b] for b in inds_list]
        mps_0 = qtn.MatrixProductState(
            [np.array([ms[0]])] + [np.array([[m]]) for m in ms[1:-1]] + [np.array([ms[-1]])],
            shape='lrp', site_tag_id=site_tag_id, site_ind_id=site_ind_id)
    elif L == 1:
        m = mdict[inds_list[0]]
        mps_0 = qtn.TensorNetwork([qtn.Tensor(m, inds=(site_ind_id.format(0),), tags=(site_tag_id.format(0),))])
        mps_0.view_as(qtn.MatrixProductState, inplace=True, L=1, cyclic=False,
                      site_tag_id=site_tag_id, site_ind_id=site_ind_id)
    elif L == 0:
        mps_0 = qtn.TensorNetwork([])
        mps_0.view_as(qtn.MatrixProductState, inplace=True, L=0, cyclic=False,
                      site_tag_id=site_tag_id, site_ind_id=site_ind_id)
    return mps_0


def get_select_elem_mpo(L: int, q: int, inds_list: Sequence[int], upper_ind_id: str = 'o({})',
                        lower_ind_id: str = 'i({})',
                        site_tag_id: str = 'X({})') -> MPOType:
    """ build MPO to select certain elements specified by inds
    """
    mdict = {}
    for b in range(q):
        mvec = np.zeros((q,))
        mvec[b] = 1.
        mdict[b] = np.diag(mvec)

    assert (len(inds_list) == L), 'inds_list needs to be desired index in q-nary'

    mpo_0 = None
    if L >= 2:
        ms = [mdict[b] for b in inds_list]
        mpo_0 = qtn.MatrixProductOperator(
            [np.array([ms[0]])] + [np.array([[m]]) for m in ms[1:-1]] + [np.array([ms[-1]])],
            shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    elif L == 1:
        m = mdict[inds_list[0]]
        mpo_0 = qtn.TensorNetwork([qtn.Tensor(m, inds=(upper_ind_id.format(0), lower_ind_id.format(0)),
                                              tags=(site_tag_id.format(0),))])
        mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=1, cyclic=False,
                      site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    elif L == 0:
        mpo_0 = qtn.TensorNetwork([])
        mpo_0.view_as(qtn.MatrixProductOperator, inplace=True, L=0, cyclic=False,
                      site_tag_id=site_tag_id, upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
    return mpo_0


def map_state_to_mps(L: int, q: int, state: np.ndarray, site_ind_id: str = 'i({})', site_tag_id: str = 'T({})',
                     split_opts: dict = None) -> MPSType:
    """ convert state represented as vector converted to mps state
        here, returns original vec as a single tensor as quimb.TensorNetwork object
    """
    if split_opts is None:   split_opts = {}
    state = state.reshape((q,) * L)
    state_tensor = qtn.Tensor(data=state, inds=[site_ind_id.format(x) for x in range(L)])
    state_mps = helper.mpx_from_dense(state_tensor, L, (site_ind_id,), site_tag_id=site_tag_id, split_opts=split_opts)
    return state_mps


def map_operator_to_mpo(L: int, q: int, operator: np.ndarray, upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})',
                        site_tag_id: str = 'T({})', split_opts: dict = None) -> MPOType:
    """ convert operator represented as a matrix (output x input) as an MPO
            here, returns original vec as a single tensor as quimb.TensorNetwork object
    """
    if split_opts is None:   split_opts = {}
    operator = operator.reshape((q,) * L * 2)
    state_tensor = qtn.Tensor(data=operator, inds=[upper_ind_id.format(x) for x in range(L)] +
                                                  [lower_ind_id.format(x) for x in range(L)])
    state_mpo = helper.mpx_from_dense(state_tensor, L, (upper_ind_id, lower_ind_id), site_tag_id=site_tag_id,
                                      split_opts=split_opts)
    return state_mpo


def map_mps_to_state(mps: MPSType, flip=False) -> np.ndarray:
    """ convert MPS into 1-D np.ndarray
    """
    L = mps.L
    q = mps.phys_dim()

    tensor = mps.contract(tags=all)  # contract all tensors
    if flip:
        tensor.transpose(*[mps.site_ind_id.format(i) for i in range(L - 1, -1, -1)], inplace=True)
    else:
        tensor.transpose(*[mps.site_ind_id.format(i) for i in range(L)], inplace=True)

    ## need to reshape to ndim-dimensional tensor
    out_data = tensor.data  # self.axis_map.array_reshape(tensor.data, (self.q,)*self.L*ndim)
    out_data = out_data.reshape((q ** L,))

    return out_data * (10 ** mps.exponent)


def map_mpo_to_operator(mpo: MPOType, flip=False) -> np.ndarray:
    """ convert MPO into 2*K-D np.ndarray (o0 x o1 ...) x (i0 x i1 ...)
    """
    L = mpo.L
    q = mpo.phys_dim()

    tensor = mpo.contract()  # contract all tensors
    if flip:
        out_inds = [mpo.upper_ind_id.format(i) for i in range(L - 1, -1, -1)] + \
                   [mpo.lower_ind_id.format(i) for i in range(L - 1, -1, -1)]
    else:
        out_inds = [mpo.upper_ind_id.format(i) for i in range(L)] + \
                   [mpo.lower_ind_id.format(i) for i in range(L)]
    tensor.transpose(*out_inds, inplace=True)

    ## need to reshape to ndim-dimensional tensor
    out_data = tensor.data
    out_data = out_data.reshape((q ** L,) * 2)

    return out_data * (10 ** mpo.exponent)


def get_heaviside_mpo(L: int, q: int, zero_ind: int, get='right', upper_ind_id: str = 'o({})',
                      lower_ind_id: str = 'i({})', site_tag_id: str = 'T({})') -> MPOType:
    """ get mpo representing heaviside mpo function
        zero_ind specifies index at which the step occurs
        get: 'right','xright','left','xleft'
           right: inds >= zero_ind = 1 else 0
           left:  inds <= zero_ind = 1 else 0
           x:  excludes zero_ind
    """
    npts = q ** L

    if isinstance(zero_ind, tuple):  # zero grid pt not included
        zero_ind_L, zero_ind_R = zero_ind
        zero_ind_L1, zero_ind_R1 = zero_ind_L, zero_ind_R
    else:
        zero_ind_L = zero_ind_R = zero_ind
        zero_ind_L1, zero_ind_R1 = zero_ind_L - 1, zero_ind_R + 1

    if (zero_ind_R == -1 and get == 'right') or (zero_ind_L == npts and get == 'left'):
        return get_iden_mpo(L, q, upper_ind_id, lower_ind_id, site_tag_id)
    elif (zero_ind_L == -1 and get == 'left') or (zero_ind_R == npts and get == 'right'):
        return None
    else:
        pass

    if get == 'right':
        vec = np.append(np.zeros((zero_ind_R,)), np.ones((npts - zero_ind_R,)))
    elif get == 'xright':
        vec = np.append(np.zeros((zero_ind_R1,)), np.ones((npts - zero_ind_R1,)))
    elif get == 'xleft':
        vec = np.append(np.ones((zero_ind_L1 + 1,)), np.zeros((npts - zero_ind_L1 - 1,)))
    elif get == 'left':
        vec = np.append(np.ones((zero_ind_L + 1,)), np.zeros((npts - zero_ind_L - 1,)))
    else:
        raise ValueError('need valid input for get parameter')

    # vec = vec.reshape((q,)*L)
    # vec_tensor = qtn.Tensor(data=vec,inds=[lower_ind_id.format(x) for x in range(L)])
    # heaviside_mps = helper.mpx_from_dense(vec_tensor,L,(lower_ind_id,),site_tag_id=site_tag_id)
    heaviside_mps = map_state_to_mps(L, q, vec, lower_ind_id, site_tag_id)  # default compress opts
    heaviside_mpo = helper.mps_to_diag_mpo(heaviside_mps, upper_ind_id=upper_ind_id,
                                           lower_ind_id=lower_ind_id)
    return heaviside_mpo


def get_reverse_diag_mpo(L: int, q: int, upper_ind_id: str = 'o{}', lower_ind_id: str = 'i{}',
                         site_tag_id: str = 'T{}') -> 'MPOType':
    """ mpo that flip inds:  SX * SX * ...
    """
    sx = np.zeros((q, q))
    for i in range(q):
        sx[i, -1 - i] = 1.0

    tens_list = [np.array([sx])] + [np.array([[sx]])] * (L - 2) + [np.array([sx])]
    return qtn.MatrixProductOperator(tens_list, shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                                     lower_ind_id=lower_ind_id)


# def get_shift_mpo(L: int, q: int, shift: int, left_bc: 'BCType'=None, right_bc: 'BCType'=None,
#                   upper_ind_id: str = 'o{}', lower_ind_id: str = 'i{}', site_tag_id:str = 'T{}') -> 'MPOType':
#     """ make an MPO with 1's shifted from the diagonal
#     """
#     if shift == 0:
#         tens_list = [np.array([np.eye(q)])] + [np.array([[np.eye(q)]])] * (L - 2) + [np.array([np.eye(q)])]
#         return qtn.MatrixProductOperator(tens_list, shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
#                                          lower_ind_id=lower_ind_id)
#
#     if q != 2:
#         raise NotImplementedError
#
#     sp = np.array([[0., 0.], [1., 0.]])
#     sm = np.array([[0., 1.], [0., 0.]])
#     zero = np.zeros((q, q))
#
#     mat1 = sm if shift > 0 else sp
#     mat2 = sp if shift > 0 else sm
#
#     # nbits = np.round(np.log(shift) / np.log(q))  # number of bits to encode shift value
#     nbits = int(np.floor(np.log(np.abs(shift)) / np.log(q)) + 1)  # number of bits to encode shift value
#
#     if abs(shift) == 1:
#         last_tens = [np.array([mat1, mat2])]
#     elif abs(shift) == q:
#         last_tens = [np.array([[np.eye(q), mat1], [zero, mat2]]),
#                      np.array([zero, np.eye(q)])]
#     else:
#         raise NotImplementedError
#
#     if left_bc == BCType.PERIODIC:
#         tens_list = [np.array([np.eye(q), mat1 + mat2])] + \
#                     [np.array([[np.eye(q), mat1], [zero, mat2]])] * (L - nbits - 1)
#     elif left_bc == BCType.ANTIPERIODIC:
#         tens_list = [np.array([np.eye(q), mat1 - mat2])] + \
#                     [np.array([[np.eye(q), mat1], [zero, mat2]])] * (L - nbits - 1)
#     else:   # OPEN, no boundaries
#         tens_list = [np.array([np.eye(q), mat1])] + \
#                     [np.array([[np.eye(q), mat1], [zero, mat2]])] * (L - nbits - 1)
#
#     tens_list += last_tens
#
#     out = qtn.MatrixProductOperator(tens_list, shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
#                                      lower_ind_id=lower_ind_id)
#     return out


def get_shift_mpo(L: int, q: int, shift: int, boundary_conditions: 'DerivativeConfiguration' = None,
                  upper_ind_id: str = 'o{}', lower_ind_id: str = 'i{}', site_tag_id: str = 'T{}') -> 'MPOType':
    """ make an MPO with 1's shifted from the diagonal
    """
    if boundary_conditions is None:
        boundary_conditions = DerivativeConfiguration()

    left_bc, right_bc = boundary_conditions.bc
    bc_offset = boundary_conditions.offset
    bc_offset_r = boundary_conditions.offset_r

    if L == 1:
        npts = q ** L
        if left_bc in [BCType.PERIODIC, BCType.ANTIPERIODIC]:
            data = [left_bc.value if abs((i + shift) // npts) else 1 for i in range(npts)]
            rows = list(range(npts))
            cols = [(i + shift)%npts for i in range(npts)]
        else:
            data = [1] * (npts - abs(shift))
            if shift >= 0:
                rows = list(range(npts - shift))
                cols = list(range(shift, npts))
            else:
                rows = list(range(-shift, npts))
                cols = list(range(npts + shift))

            if left_bc in [BCType.SYMMETRIC, BCType.ANTISYMMETRIC, BCType.ZEROGRADIENT, BCType.ZEROVALUE,
                           BCType.DIRICHLET, BCType.NEUMANN]:

                if bc_offset == 0:

                    # if shift < 0:
                    #     if left_bc in [BCType.ANTISYMMETRIC, BCType.ZEROVALUE, BCType.DIRICHLET]:
                    #         rows.pop(0)
                    #         cols.pop(0)
                    #         data.pop(0)

                    for i in range(-shift):
                        rows += [i]
                        cols += [abs(shift) - i]
                        data += [1 * left_bc.value]

                elif bc_offset == 1:
                    for i in range(abs(shift)):
                        rows += [i]
                        cols += [abs(shift) - i - 1]
                        data += [1 * left_bc.value]

                elif bc_offset == 2:
                    if left_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
                        raise NotImplementedError

            else:
                raise NotImplementedError

            if right_bc in [BCType.SYMMETRIC, BCType.ANTISYMMETRIC, BCType.ZEROGRADIENT, BCType.ZEROVALUE,
                            BCType.DIRICHLET, BCType.NEUMANN]:

                if bc_offset_r == 0:

                    # if shift > 0:
                    #     if right_bc in [BCType.ANTISYMMETRIC, BCType.ZEROVALUE, BCType.DIRICHLET]:
                    #         rows.pop(-1)
                    #         cols.pop(-1)
                    #         data.pop(-1)

                    for i in range(shift):
                        rows += [npts - shift + i]
                        cols += [npts - i - 2]
                        data += [1 * left_bc.value]

                elif bc_offset_r == -1:
                    for i in range(shift):
                        rows += [npts - shift + i]
                        cols += [npts - i - 1]
                        data += [1 * left_bc.value]
                            
                elif bc_offset_r == -2:

                    if shift > 1 and right_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
                        raise NotImplementedError

                    for i in range(1, shift):
                        rows += [npts - shift + i]
                        cols += [npts - i]
                        data += [1 * left_bc.value]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError



        shift_mat = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(npts, npts))
        shift_tens = qtn.Tensor(shift_mat, inds=(upper_ind_id.format(0), lower_ind_id.format(0)),
                                tags=(site_tag_id.format(0),))
        out = qtn.TensorNetwork([shift_tens])
        out = out.view_as(qtn.MatrixProductOperator, L=1, cyclic=False, inplace=True,
                          upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)

        # plt.figure()
        # plt.imshow(shift_mat.todense())
        # plt.title(f'shift {shift},{left_bc} {bc_offset},{right_bc} {bc_offset_r}')
        # plt.show()
        # # exit()

        return out

    if shift == 0:
        tens_list = [np.array([np.eye(q)])] + [np.array([[np.eye(q)]])] * (L - 2) + [np.array([np.eye(q)])]
        return qtn.MatrixProductOperator(tens_list, shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                                         lower_ind_id=lower_ind_id)

    if q != 2:
        raise NotImplementedError

    sp = np.array([[0., 0.], [1., 0.]])
    sm = np.array([[0., 1.], [0., 0.]])
    zero = np.zeros((q, q))

    mat1 = sm if shift > 0 else sp
    mat2 = sp if shift > 0 else sm

    # nbits = int(np.floor(np.log(np.abs(shift)) / np.log(q)) + 1)  # number of bits to encode shift value
    # print('nbits', nbits)

    if left_bc == BCType.PERIODIC:
        shift = abs(shift) % q ** L * np.sign(shift)
        if abs(shift) > q ** L / 2:
            shift = (q ** L - np.abs(shift)) * -1 * np.sign(shift)

    def get_last_tens(bstr, is_end=False):
        ## actual last tens (column vector)
        if bstr[-1] == '0':
            tens_list_0 = get_last_tens(bstr[:-1], is_end=False) if len(bstr) > 1 else []
            tens_list_1 = [np.array([np.eye(q)])] if is_end else [np.array([[np.eye(q)]])]
            return tens_list_0 + tens_list_1
        elif bstr[-1] == '1':
            if len(bstr) == 1:
                return [np.array([mat1, mat2])] if is_end else [np.array([[mat1], [mat2]])]
            else:
                if bstr[0] == '0':
                    tens_list_0 = [np.array([[np.eye(q), mat1], [zero, mat2]])]
                    tens_list_1 = get_last_tens(bstr[1:], is_end=is_end)
                    return tens_list_0 + tens_list_1
                elif bstr[0] == '1':
                    tens_list_0 = [np.array([[mat1, zero], [mat2, np.eye(q)]])]
                    tens_list_1 = get_last_tens(bstr[1:], is_end=is_end)
                    return tens_list_0 + tens_list_1
            return np.array([mat1, mat2])

    # print('shift str', str(bin(abs(shift))))
    shift_str = str(bin(abs(shift)))[2:]
    nbits = len(shift_str)
    if nbits == L:
        raise ValueError('ax shifted by too much', shift)
    # print('shift str', nbits, shift_str, shift_str[-1])
    last_tens = get_last_tens(shift_str, is_end=True)

    if left_bc == BCType.PERIODIC:
        tens_list = [np.array([np.eye(q), mat1 + mat2])] + \
                    [np.array([[np.eye(q), mat1], [zero, mat2]])] * (L - nbits - 1)
    elif left_bc == BCType.ANTIPERIODIC:
        tens_list = [np.array([np.eye(q), mat1 - mat2])] + \
                    [np.array([[np.eye(q), mat1], [zero, mat2]])] * (L - nbits - 1)
    else:  # OPEN, no boundaries
        tens_list = [np.array([np.eye(q), mat1])] + \
                    [np.array([[np.eye(q), mat1], [zero, mat2]])] * (L - nbits - 1)

    tens_list += last_tens

    out = qtn.MatrixProductOperator(tens_list, shape='lrud', site_tag_id=site_tag_id, upper_ind_id=upper_ind_id,
                                    lower_ind_id=lower_ind_id)

    # if True:  # bc_offset != 0:
    #     print('center diff')
    #     op_tens = out.contract(all) * 10 ** out.exponent
    #     op_tens.transpose(*[out.upper_ind_id.format(i) for i in range(out.L)],
    #                       *[out.lower_ind_id.format(i) for i in range(out.L)],
    #                       inplace=True)
    #     op_mat = op_tens.data.reshape(2 ** out.L, 2 ** out.L)
    #     print('op mat', left_bc, right_bc, bc_offset, bc_offset_r)
    #     print('L', op_mat[:5, :5])
    #     print('R', op_mat[-5:, -5:])


    ## boundary condition corrections ##
    order = boundary_conditions.order
    if left_bc in [BCType.SYMMETRIC, BCType.ANTISYMMETRIC, BCType.ZEROGRADIENT, BCType.ZEROVALUE,
                   BCType.DIRICHLET, BCType.NEUMANN]:

        # if bc_offset > 0 and shift < 0:
        #     nbits_offset = nbits

        # print('shift MPO L BC', left_bc, bc_offset, shift)

        size = max(order + 1, np.abs(-shift + max(0, bc_offset)))
        nbits_offset = int(np.floor(np.log(size) / np.log(q)) + 1)
        end_mat_0 = np.zeros((q ** nbits_offset, q ** nbits_offset))
        # print('end mat 0 shape', end_mat_0.shape)

        if bc_offset == 2:     ## (no x_0) x_1, x_2, ... -> x_-1, x_0, x_1, ...
            for i in range(-shift):
                # print('i', i, -shift - 1)
                if i < (-shift - 1):    ## x_-2, x_-1, ....
                    ## DIRICHLET (0 value):   x_-1 = -x_1, x_-2 = - x_2
                    ## NEUMANN (0 gradient):  x_-1 = x_1, x_-2 = x_2
                    end_mat_0[i, -shift - i - 2] = 1.0 * np.sign(left_bc)

                else:
                    ## corrections (not actually implemented)
                    if left_bc in [BCType.ANTISYMMETRIC, BCType.DIRICHLET, BCType.ZEROVALUE]:
                        ## set site to boundary_conditions.bc_value_l
                        ## zeros if bc_value_l = 0, as is the default
                        if boundary_conditions.value_l != 0.:
                            raise NotImplementedError

                    elif left_bc in [BCType.SYMMETRIC, BCType.NEUMANN, BCType.ZEROGRADIENT]:
                        ## evaluate the x_0 term
                        order = boundary_conditions.order
                        ## order 1: B = (4 x_(1), -x_(2) ) / 3
                        ## order 2: B = (15 x_(1), -6 x_(2), x_(3) ) / 10
                        ## order 3: B = (56 x_(1), -28 x_(2), 8 x_(3), -x_(4) ) / 35
                        ## order 4: B = (210 x_(1), -120 x_(2), 45 x_(3), -10 x_(4), x_(5) ) / 126
                        if order <= 1:
                            end_mat_0[i, 0] += 4 / 3
                            end_mat_0[i, 1] += -1 / 3
                        elif order == 2:
                            end_mat_0[i, 0] += 15 / 10
                            end_mat_0[i, 1] += -6 / 10
                            end_mat_0[i, 2] += 1 / 10
                        elif order == 3:
                            end_mat_0[i, 0] += 56 / 35
                            end_mat_0[i, 1] += -28 / 35
                            end_mat_0[i, 2] += 8 / 35
                            end_mat_0[i, 3] += -1 / 35
                        elif order == 4:
                            end_mat_0[i, 0] += 210 / 126
                            end_mat_0[i, 1] += -120 / 126
                            end_mat_0[i, 2] += 45 / 126
                            end_mat_0[i, 3] += -10 / 126
                            end_mat_0[i, 4] += 1 / 126
                        else:
                            print('order', order)
                            raise NotImplementedError
        else:
            ## bc_offset =  1:  x_(-3/2), x_(-1/2), | x_(1/2)
            ## bc_offset =  0:  x_(-2), x_(-1), | x_(0)
            ## bc_offset = -1:  x_(-5/2), x_(-3/2), | x_(-1/2), x_(1/2)
            for i in range(-shift):
                # print('i', i, -shift - bc_offset - i)
                end_mat_0[i, -shift - bc_offset - i] = 1.0 * np.sign(left_bc.value)

        # print('end mat 0 (L)', end_mat_0)
        # exit()

        if not np.linalg.norm(end_mat_0) < 1.0e-16:
            bc0_tens = qtn.Tensor(end_mat_0.reshape((q,) * nbits_offset * 2),
                                  inds=tuple([f'o({i})' for i in range(nbits_offset)] + \
                                             [f'i({i})' for i in range(nbits_offset)]))
            bc0_mpo = helper.mpx_from_dense(bc0_tens, nbits_offset, ('o({})', 'i({})'), site_tag_id='X({})')

            mpo_0 = get_select_elem_mpo(L - nbits_offset, q, [0] * (L - nbits_offset))
            helper.append_mpx(mpo_0, bc0_mpo, inplace=True)
            helper.add_MPO(out, mpo_0, inplace=True)

    elif left_bc == BCType.PERIODIC or left_bc == BCType.ANTIPERIODIC or left_bc == BCType.OPEN:
        pass
    else:
        raise NotImplementedError(f'check left bc {left_bc}')

    if right_bc in [BCType.SYMMETRIC, BCType.ANTISYMMETRIC, BCType.ZEROGRADIENT, BCType.ZEROVALUE,
                   BCType.DIRICHLET, BCType.NEUMANN]:
        # try:
        #     # print(np.log(np.abs(shift - min(0, bc_offset_r))))
        #     nbits_offset = int(np.floor(np.log(np.abs(shift - min(0, bc_offset_r))) / np.log(q)) + 1)
        # except OverflowError:
        #     nbits_offset = 1

        # end_mat_0 = np.zeros((q ** nbits_offset, q ** nbits_offset))
        # for i in range(shift):
        #     try:
        #         end_mat_0[-i - 1, q ** nbits_offset - shift + bc_offset_r + i - 1] = 1.0 * np.sign(right_bc.value)
        #     except IndexError:
        #         pass

        # right_bc = BCType.ANTISYMMETRIC
        # bc_offset_r = 1
        # shift = 3
        # print('SHIFT MPO R BC', right_bc, bc_offset_r, shift)

        size = max(order + 1, np.abs(shift - min(0, bc_offset_r)))
        nbits_offset = int(np.floor(np.log(size) / np.log(q)) + 1)
        end_mat_0 = np.zeros((q ** nbits_offset, q ** nbits_offset))
        # print('end mat 0 shape', end_mat_0.shape)

        if bc_offset_r == -2:     ## ..., x_(L-2), x_(L-1) -> ..., x_(L-1), x_L, x_(L+1), ...
            for i in range(shift):
                # print('i', i, shift - 1)
                if i < (shift - 1):    ## x_-2, x_-1, ....
                    ## DIRICHLET (0 value):   x_-1 = -x_1, x_-2 = - x_2
                    ## NEUMANN (0 gradient):  x_-1 = x_1, x_-2 = x_2
                    end_mat_0[-1 - i, -1 - shift + i + 2] = 1.0 * np.sign(right_bc)

                else:
                    ## corrections (not actually implemented)
                    if right_bc in [BCType.ANTISYMMETRIC, BCType.DIRICHLET, BCType.ZEROVALUE]:
                        ## set site to boundary_conditions.bc_value_l
                        ## zeros if bc_value_l = 0, as is the default
                        if boundary_conditions.value_l != 0.:
                            raise NotImplementedError

                    elif right_bc in [BCType.SYMMETRIC, BCType.NEUMANN, BCType.ZEROGRADIENT]:
                        ## evaluate the x_0 term
                        ## order 1: B = (4 x_(1), -x_(2) ) / 3
                        ## order 2: B = (15 x_(1), -6 x_(2), x_(3) ) / 10
                        ## order 3: B = (56 x_(1), -28 x_(2), 8 x_(3), -x_(4) ) / 35
                        ## order 4: B = (210 x_(1), -120 x_(2), 45 x_(3), -10 x_(4), x_(5) ) / 126
                        if order <= 1:
                            ## a x^2 + b = y
                            ## but x is negative
                            ## a = (y2 - y1) / 3 dx**2
                            ## (y2 - y1) / 3 + b = y1
                            ## (y2-y1)/3 + b = y1 --> b = 4 y1 /3 - y2 /3
                            ## 4 (y2-y1)/3 + b = y2 --> b = -1/3 y2 + 4/3 y1
                            end_mat_0[-1 - i, -1] += 4 / 3
                            end_mat_0[-1 - i, -2] += -1 / 3
                        elif order == 2:
                            end_mat_0[-1 - i, -1] += 15 / 10
                            end_mat_0[-1 - i, -2] += -6 / 10
                            end_mat_0[-1 - i, -3] += 1 / 10
                        elif order == 3:
                            end_mat_0[-1 - i, -1] += 56 / 35
                            end_mat_0[-1 - i, -2] += -28 / 35
                            end_mat_0[-1 - i, -3] += 8 / 35
                            end_mat_0[-1 - i, -4] += -1 / 35
                        elif order == 4:
                            end_mat_0[-1 - i, -1] += 210 / 126
                            end_mat_0[-1 - i, -2] += -120 / 126
                            end_mat_0[-1 - i, -3] += 45 / 126
                            end_mat_0[-1 - i, -4] += -10 / 126
                            end_mat_0[-1 - i, -5] += 1 / 126
                        else:
                            raise NotImplementedError

            # print('end mat 0 (R)', end_mat_0)
            # exit()
        else:
            ## bc_offset =  1:  x_(-3/2), x_(-1/2), | x_(1/2)
            ## bc_offset =  0:  x_(-2), x_(-1), | x_(0)
            ## bc_offset = -1:  x_(-5/2), x_(-3/2), | x_(-1/2), x_(1/2)
            for i in range(shift):
                # print('i', i, -1 - shift - bc_offset_r + i)
                end_mat_0[-1 - i, -1 - shift - bc_offset_r + i] = 1.0 * np.sign(right_bc.value)
                # end_mat_0[-i - 1, q ** nbits_offset - shift + bc_offset_r + i - 1] = 1.0 * np.sign(right_bc.value)

        # print('end mat 0 (R)', end_mat_0)
        # exit()

        if not np.linalg.norm(end_mat_0) < 1.0e-16:
            bc0_tens = qtn.Tensor(end_mat_0.reshape((q,) * nbits_offset * 2),
                                  inds=tuple([f'o({i})' for i in range(nbits_offset)] + \
                                             [f'i({i})' for i in range(nbits_offset)]))
            bc0_mpo = helper.mpx_from_dense(bc0_tens, nbits_offset, ('o({})', 'i({})'), site_tag_id='X({})')

            mpo_0 = get_select_elem_mpo(L - nbits_offset, q, [1] * (L - nbits_offset))
            mpo_0 = helper.append_mpx(mpo_0, bc0_mpo, inplace=False)    ## not sure why inplace=True causes a bug

            # ax = Axis(L, q)
            # tmp_out = ax.map_mpo_to_operator(mpo_0)
            # print('bc0', tmp_out[-5:,-5:])
            #
            # out_dat = ax.map_mpo_to_operator(out)
            # print('out', out_dat[-5:,-5:])

            helper.add_MPO(out, mpo_0, inplace=True)


    elif right_bc == BCType.PERIODIC or right_bc == BCType.ANTIPERIODIC or right_bc == BCType.OPEN:
        pass
    else:
        raise NotImplementedError(f'check right bc {right_bc}')

    # if bc_offset_r == -2:
    #     out_data = map_mpo_to_operator(out)
    #     print('bcs', left_bc, right_bc)
    #     print('shift', shift, out_data[:10, :10])
    #     print('shift', shift, out_data[-10:, -10:])
    #     # exit()

    # print('out', out)
    # print('shift', shift)
    # print('out', out.to_dense().reshape(q**L, q**L))
    # plt.figure()
    # plt.imshow(out.to_dense().reshape(q**L, q**L))
    # plt.title(f'shift {shift}')
    # plt.show()

    return out

def get_qft_operator(L, q, inverse=False, reorder_coarse=True, cutoff=CUTOFF, **mpo_args):
    """
    follows the "QFT has low entanglement" paper, hadamards included in each qft layer
    :param L: number of qubits
    :param inverse: whether to do inverse QFT or normal QFT
    :param reorder_coarse: swap 0 and 1 on the coarse inds
    :param mpo_args: formatted strings for inds, tensor tags, etc..
    :return:
        note: applying qft operator results in flipped qubits
              (coarse -> fine becomes fine -> coarse)
        note: if reorder_coarse: after flipping, frequencies match those from np.fft.fft_freq
              else: after flipping, frequencies are ordered from -kmax to kmax
    """

    if q != 2 and L != 1:
        raise NotImplementedError

    reshape_to_one = False
    npts = q**L
    if L == 1:
        q = 2
        L = int(np.round(np.log2(npts)))
        reshape_to_one = True

    iden = np.eye(q)
    zero = np.zeros((q, q))
    C0_gate = np.array([[1., 0.], [0., 0.]])
    C1_gate = np.array([[0., 0.], [0., 1.]])
    H_gate = 1. / np.sqrt(2) * np.array([[1., 1.], [1., -1.]])
    HC0_gate = np.dot(C0_gate, H_gate)
    HC1_gate = np.dot(C1_gate, H_gate)
    P_gate = lambda th: np.array([[1., 0.], [0., np.exp(-1.j * th)]])

    def _build_qft_layer(num_layer):
        ### num_layer indexed starting at 0

        if num_layer >= L - 1:
            return []

        ## pad front with identities
        if num_layer > 0:
            tens_list = [np.array([iden])]
            tens_list += [np.array([[iden]])] * (num_layer - 1)
            tens_list += [np.array([[HC0_gate, HC1_gate]])]
        else:
            tens_list = [np.array([HC0_gate, HC1_gate])]

        for i in range(1, L - num_layer - 1):
            tens_list += [np.array([[iden, zero], [zero, P_gate(np.pi / 2 ** i)]])]

        if num_layer == L - 2:
            ## phase gate * H
            tens_list += [np.array([H_gate, np.dot(H_gate, P_gate(np.pi / 2 ** (L - num_layer - 1)))])]
        else:
            tens_list += [np.array([iden, P_gate(np.pi / 2 ** (L - num_layer - 1))])]

        return tens_list

    mpo = qtn.MatrixProductOperator(_build_qft_layer(0), 'lrud')
    for nl in range(1, L - 1):
        next_mpo = qtn.MatrixProductOperator(_build_qft_layer(nl), 'lrud')
        mpo = helper.apply_zipup(next_mpo, mpo, compress=True, compress_opts={'cutoff': cutoff})
        ## apply zipup from left to right, recompress/canonicalize from left to right.
        ## this is what is done in the paper.

    if reorder_coarse:
        ## coarse tens is now the last tens
        mpo = helper.apply_gate(mpo, (L - 1,), np.array([[0.,1.],[1.,0.]]), inplace=True)

    if inverse:
        mpo = helper.mpo_conj_transpose(mpo, inplace=True)

    if reshape_to_one:
        mpo_data = helper.to_dense(mpo)
        mpo_1 = qtn.TensorNetwork([qtn.Tensor(mpo_data, inds=(mpo.upper_ind_id.format(0), mpo.lower_ind_id.format(0)),
                                            tags=(mpo.site_tag_id.format(0)))])
        mpo_1.view_as(qtn.MatrixProductOperator, cyclic=False, inplace=True, L=1,
                      upper_ind_id=mpo.upper_ind_id, lower_ind_id=mpo.lower_ind_id, site_tag_id=mpo.site_tag_id)
        mpo = mpo_1

    return mpo


def get_qft_mpo(L, q, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})'):
    """ obtain MPO that takes discrete Fourier transform
    """
    if L == 1:
        mat = np.eye(q)
        qft_mat = np.fft.fft(mat, axis=0)
        qft_tens = qtn.Tensor(qft_mat, inds=(upper_ind_id.format(0), lower_ind_id.format(0)),
                              tags=(site_tag_id.format(0),))
        qft_mpo = qtn.TensorNetwork([qft_tens])
        qft_mpo = qft_mpo.view_as(qtn.MatrixProductOperator, L=1, cyclic=False, inplace=True,
                                  upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)
        return qft_mpo

    if q != 2:
        raise NotImplementedError

    zero = np.zeros((q, q), dtype=complex)
    delta = np.eye(q, dtype=complex)
    # hadamard = 1. / np.sqrt(2) * np.array([[1., 1.],[1., -1.]])
    hadamard_0 = 1. / np.sqrt(2) * np.array([[1., 1.], [0., 0.]])
    hadamard_1 = 1. / np.sqrt(2) * np.array([[0., 0.], [1., -1.]])

    qft_mpo = None
    for i in range(L):

        tens_list = []

        for j in range(i):
            if j == 0:
                tens_list += [np.array([delta])]
            else:
                tens_list += [np.array([[delta]])]

        if i == 0:
            tens_list += [np.array([hadamard_0, hadamard_1])]
        elif i == L - 1:
            tens_list += [np.array([hadamard_0 + hadamard_1])]
        else:
            tens_list += [np.array([[hadamard_0, hadamard_1]])]

        for j in range(i + 1, L):
            op00 = delta
            op11 = np.array([[1., 0.], [0., np.exp(-1.j * np.pi / (2 ** (j - i)))]])
            if j != L - 1:
                tens_list += [np.array([[op00, zero], [zero, op11]])]
            else:
                tens_list += [np.array([op00, op11])]

        mpo_i = qtn.MatrixProductOperator(tens_list, shape='lrud', site_tag_id=site_tag_id,
                                          upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
        if qft_mpo is None:
            qft_mpo = mpo_i
        else:
            qft_mpo = helper.apply(mpo_i, qft_mpo, compress=True,
                                   compress_opts={'cutoff': 1.0e-20, 'cutoff_mode': 'rsum2'})

    return qft_mpo


def get_inverse_qft_mpo(L, q, site_tag_id: str = 'T({})', upper_ind_id: str = 'o({})', lower_ind_id: str = 'i({})'):
    """ obtain MPO that takes discrete Fourier transform
    """
    if L == 1:
        mat = np.eye(q)
        qft_mat = np.fft.ifft(mat, axis=1)
        qft_tens = qtn.Tensor(qft_mat, inds=(upper_ind_id.format(0), lower_ind_id.format(0)),
                              tags=(site_tag_id.format(0),))
        qft_mpo = qtn.TensorNetwork([qft_tens])
        qft_mpo = qft_mpo.view_as(qtn.MatrixProductOperator, L=1, cyclic=False, inplace=True,
                                  upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id, site_tag_id=site_tag_id)
        return qft_mpo

    if q != 2:
        raise NotImplementedError

    zero = np.zeros((q, q), dtype=complex)
    delta = np.eye(q, dtype=complex)
    # hadamard = 1. / np.sqrt(2) * np.array([[1., 1.],[1., -1.]])
    hadamard_0 = 1. / np.sqrt(2) * np.array([[1., 1.], [0., 0.]]).T
    hadamard_1 = 1. / np.sqrt(2) * np.array([[0., 0.], [1., -1.]]).T

    qft_mpo = None
    for i in range(L):

        tens_list = []

        for j in range(i):
            if j == 0:
                tens_list += [np.array([delta])]
            else:
                tens_list += [np.array([[delta]])]

        if i == 0:
            tens_list += [np.array([hadamard_0, hadamard_1])]
        elif i == L - 1:
            tens_list += [np.array([hadamard_0 + hadamard_1])]
        else:
            tens_list += [np.array([[hadamard_0, hadamard_1]])]

        for j in range(i + 1, L):
            op00 = delta
            op11 = np.array([[1., 0.], [0., np.exp(1.j * np.pi / (2 ** (j - i)))]])
            if j != L - 1:
                tens_list += [np.array([[op00, zero], [zero, op11]])]
            else:
                tens_list += [np.array([op00, op11])]

        mpo_i = qtn.MatrixProductOperator(tens_list, shape='lrud', site_tag_id=site_tag_id,
                                          upper_ind_id=upper_ind_id, lower_ind_id=lower_ind_id)
        if qft_mpo is None:
            qft_mpo = mpo_i
        else:
            qft_mpo = helper.apply(qft_mpo, mpo_i, compress=True,
                                   compress_opts={'cutoff': 1.0e-20, 'cutoff_mode': 'rsum2'})
    return qft_mpo
