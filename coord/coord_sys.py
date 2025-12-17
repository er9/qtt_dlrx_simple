from setup_.configs import *

from dataclasses import dataclass
# import axis as axis_class

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis
    from grid import Grid
    from gridTN import GridTN
    from field import Field, ScalarField

def parse_coordsys_type(coordsys_id) -> CoordinateSystemType:
    """ return BasisType(Enum) given accepted basis_id (strs)
    """
    if isinstance(coordsys_id, CoordinateSystem):  # is already the class
        return coordsys_id

    if isinstance(coordsys_id, CoordinateSystemType):
        return coordsys_id

    if coordsys_id in ['cart', 'cartesian', 'Cartesian']:
        return CoordinateSystemType.CARTESIAN
    elif coordsys_id in ['cyl', 'cylindrical', 'Cylindrical']:
        return CoordinateSystemType.CYLINDRICAL
    elif coordsys_id in ['sph', 'spherical', 'Spherical']:
        return CoordinateSystemType.SPHERICAL

def get_coordinate_system(coordsys_type) -> 'CoordinateSystem':
    """ return basis class given BasisType(Enum)
    """
    if isinstance(coordsys_type, CoordinateSystem):  # is already the class
        return coordsys_type

    if not isinstance(coordsys_type,Enum):
        coordsys_type = parse_coordsys_type(coordsys_type)

    if coordsys_type == CoordinateSystemType.CARTESIAN:
        from coord.cartesian import CartesianCoordinateSpace
        return CartesianCoordinateSpace
    if coordsys_type == CoordinateSystemType.CYLINDRICAL:
        from coord.cylindrical import CylindricalCoordinateSpace
        return CylindricalCoordinateSpace
    if coordsys_type == CoordinateSystemType.SPHERICAL:
        import coord.spherical as SphericalCoordinateSpace
        return SphericalCoordinateSpace



@dataclass(frozen=True)
class Coordinate:
    name : str
    type : CoordinateType = CoordinateType.X

    def __str__(self):
        return f'{self.name}{self.type.value}'


class CoordinateSystem:

    def __init__(self, coordID: str, *axes: 'Axis', coords: Optional[list['Coordinate']]=None):

        self.coordID = coordID
        self._axes = axes

        for ax in axes:
            try:
                ax.coord_sys = self
            except AttributeError:
                raise AttributeError(f'{ax.coordinate} Axis already assigned to a CoordinateSystem {ax.coord_sys.coordID}')

        if coords is None:
            self.coords = [ax.coordinate for ax in axes]
            self.coord_axes = {ax.coordinate: ax for ax in axes}
            # self.coords = {ax.axID.type: ax.axID for ax in axes}
        else:
            self.coords = coords
            self.coord_axes = {}
            for c in coords:
                try:
                    self.coord_axes[c] = next(ax for ax in axes if ax.coordinate == c)
                except StopIteration:
                    self.coord_axes[c] = None

        self.type_coords = {}

    @property
    def axes(self):
        return self._axes

    def add_axes(self, new_axes):
        self._axes += new_axes
        for c in self.coords:
            try:
                self.coord_axes[c] = next(ax for ax in new_axes if ax.coordinate == c)
            except StopIteration:
                self.coord_axes[c] = None


    def get_coord(self, coord_type: Union['CoordinateType',int]):
        try:
            coord = next(iter(c for c in self.coords if c.type == coord_type))
        except StopIteration:
            coord = None
        return coord


    def get_axis(self, coord_type: 'CoordinateType'):
        if isinstance(coord_type, Coordinate):
            coord_type = coord_type.type
        coord = self.get_coord(coord_type)
        return self.coord_axes.get(coord,None)


    def build_firstderivative_mpo(self, ax:'Axis', deriv_config: 'DerivativeConfiguration') -> dict['Axis', MPOType]:
        """ obtain MPO taking second derivative
            ax:  Axis objects to take derivative. if ax2 is None, take second deriv along ax1
            coord: X, Y, Z ... doesnt matter for Cartesian coordinates
            deriv_params:
                bc:  BCType or Tuples of BCTypes (left, right boundary conditions)
                order: finite difference expansion order
        """
        mpo = ax.build_firstderivative_mpo(deriv_config)
        # print('coord sys first deriv', ax, deriv_config)
        return {ax: mpo}


    def build_secondderivative_mpo(self, ax1:'Axis', ax2:Optional['Axis'], deriv_config1: 'DerivativeConfiguration',
                                   deriv_config2:Optional['DerivativeConfiguration'] = None) -> dict['Axis', MPOType]:
        """ obtain MPO taking second derivative
            ax1, ax2:  Axis objects to take derivative. if ax2 is None, take second deriv along ax1
            coord1, coord2: X, Y, Z ... doesnt matter for Cartesian coordinates
            deriv_params:
                bc1, bc2:  BCType or Tuples of BCTypes (left, right boundary conditions)
                order: finite difference expansion order
                fd_type: 'center', 'forwards', 'backwards'
        """
        if deriv_config2 is None:   deriv_config2 = deriv_config1

        if ax2 is None or ax1 == ax2:
            mpo1 = ax1.build_secondderivative_mpo(deriv_config1)
            mpo_dict = {ax1: mpo1}
        else:
            mpo1 = ax1.build_firstderivative_mpo(deriv_config1)
            mpo2 = ax2.build_firstderivative_mpo(deriv_config2)
            mpo_dict = {ax1: mpo1, ax2: mpo2}
        return mpo_dict


    def build_laplacian_mpo(self, axes: Sequence['Axis'], deriv_configs: dict['Axis','DerivativeConfiguration'])\
            -> list[dict[Any, MPOType]]:
        """ obtain MPO taking Laplacain over specified axes
        """
        # raise NotImplementedError
        mpo_list = []
        for ax in axes:
            out = ax.build_secondderivative_mpo(deriv_configs[ax])
            mpo_list += [{ax: out}]
        return mpo_list


    def build_mth_laplacian_mpo(self, deriv_order: int, axes: Sequence['Axis'],
                                deriv_configs: dict['Axis', 'DerivativeConfiguration'],
                                eeo_grid: bool = False) -> list[dict['Axis', MPOType]]:
        mpo_list = []
        for ax in axes:
            out = ax.build_mth_derivative_mpo(deriv_order, deriv_configs[ax], eeo_grid=eeo_grid)
            mpo_list += [{ax: out}]
        return mpo_list


    def build_vector_laplacian_mpo(self, compID, axes: Sequence['Axis'],
                                   deriv_configs: dict['Axis','DerivativeConfiguration']
                                   ) -> list[dict['Axis', MPOType]]:
        """ obtain MPO taking Laplacian over specified axes
        """
        raise NotImplementedError


    def build_integral_mps(self, ax:'Axis', is_sqrt=False) -> MPSType:
        return ax.basis.build_integral_mps(ax, is_sqrt=is_sqrt)

    def build_coarse_grain_mpx(self, ax:'Axis', depth:int, is_sqrt=False) -> MPOType:
        return ax.basis.build_coarse_grain_mpx(ax, depth, is_sqrt=is_sqrt)



    # def take_firstderivative(self, gtn: 'GridTN', coord: int, deriv_params=None, inplace=False,
    #                          compress=True, **compress_opts):
    #     """ df/dx_i
    #         deriv_params:   bc = boundary condition
    #                         order = finite difference stencil order
    #     """
    #     raise NotImplementedError
    #
    # def take_secondderivative(self, gtn: 'GridTN', coord1: int, coord2: Optional[int]=None,
    #                           deriv_params=None, inplace=False, compress=True, **compress_opts):
    #     """ df/dx_i
    #         deriv_params:   bc = boundary condition
    #                         order = finite difference stencil order
    #     """
    #     raise NotImplementedError

    # def integrate_axis(self, gtn: 'GridTN', ax: Axis, inplace=False):
    #     raise NotImplementedError
    #
    # def integrate(self, gtn: 'GridTN', axes: list[Axis] = None, inplace=False):
    #     raise NotImplementedError


    def gradient(self, field: 'Field', deriv_axes: Iter['Axis'] = None, compress=False, compress_opts:dict=None) -> 'Field':
        """ df/dx {x} + df/dy {y} + df/dz {z}
            gtn:  GridTN (MPS) representing the scalar field
            coords:  coordinates to include in calculation of Laplacian
                (CoordinateType.X, .Y, or int)
        """
        assert(field.ncomp == 1),'gradient only defined only for scalar field'
        if deriv_axes is None:
            deriv_axes = [self.coord_axes.get(c,None) for c in self.coords]

        field_comp = field.components[field.componentIDs[0]]    # is a scalar field
        out_data = self._gtn_gradient(field_comp, deriv_axes, compress, compress_opts)
        return field.create_like(new_components=out_data)


    def vector_derivative(self, field: 'Field', deriv_ax: 'Axis', compIDs=None, compress=False,
                          compress_opts:dict=None) -> 'Field':
        """ d/dx (Ax {x}), d/dx (Ay {y}), d/dx (Az {z})
        """
        if compIDs is None:   compIDs = field.componentIDs

        comps = {compID: field[compID] for compID in compIDs}
        out_data = self._gtn_vector_derivative(comps, deriv_ax, compress, compress_opts)
        return field.create_like(new_components=out_data)


    def laplacian(self, field: 'ScalarField', deriv_axes: Iter['Axis'] = None, compress=False, compress_opts:dict=None,
                  inner_compress=False, inner_compress_opts=None) -> 'ScalarField':
        """ L(f) = d^2/dx^2 f + d^2/dy^2 f + d^2/dz^2 z
            gtn:  GridTN (MPS) representing the scalar field
            coords:  coordinates to include in calculation of Laplacian
                (CoordinateType.X, .Y, or int)
        """
        assert(field.ncomp == 1), 'laplacian only defined only for scalar field'
        if deriv_axes is None:
            deriv_axes = [self.coord_axes.get(c, None) for c in self.coords]

        # field_comp = field.components[field.componentIDs[0]]  # is a scalar field
        field_comp = field.component
        out_data = self._gtn_laplacian(field_comp, deriv_axes, compress, compress_opts,
                                       inner_compress=inner_compress, inner_compress_opts=inner_compress_opts)
        return field.create_like_scalar(new_component=out_data)


    def mth_laplacian(self, field: 'ScalarField', deriv_order:int, deriv_axes: Iter['Axis'] = None, compress=False,
                      compress_opts:dict=None, inner_compress=False, inner_compress_opts=None) -> 'ScalarField':
        """ L(f) = d^2/dx^2 f + d^2/dy^2 f + d^2/dz^2 z
            gtn:  GridTN (MPS) representing the scalar field
            coords:  coordinates to include in calculation of Laplacian
                (CoordinateType.X, .Y, or int)
        """
        assert(field.ncomp == 1), 'laplacian only defined only for scalar field'
        if deriv_axes is None:
            deriv_axes = [self.coord_axes.get(c, None) for c in self.coords]

        # field_comp = field.components[field.componentIDs[0]]  # is a scalar field
        field_comp = field.component
        out_data = self._gtn_mth_laplacian(field_comp, deriv_axes, compress, compress_opts,
                                           inner_compress=inner_compress, inner_compress_opts=inner_compress_opts)
        return field.create_like_scalar(new_component=out_data)


    def vector_laplacian(self, field: 'Field', deriv_axes: Iter['Axis'] = None, out_comps: Iter = None,
                         compress=False, compress_opts:dict=None,
                         inner_compress=False, inner_compress_opts:dict=None) -> 'Field':
        """ VL(A) = L(Ax) {x} + L(Ay) {y} + L(Az) {z}
            gtn:  GridTN (MPS) representing the scalar field
            coords:  coordinates to include in calculation of Laplacian
                (CoordinateType.X, Y, or int)
        """
        if out_comps is None:   out_comps = field.componentIDs
        if deriv_axes is None:
            deriv_axes = [self.coord_axes.get(c,None) for c in self.coords]

        field_comps = {compID: field[compID] for compID in out_comps}
        out_data = self._gtn_vector_laplacian(field_comps, deriv_axes, compress, compress_opts,
                                              inner_compress, inner_compress_opts)
        return field.create_like(new_components=out_data)


    def curl(self, field: 'Field', out_comps: Iter[int] = None, compress=False, compress_opts:dict=None,
             inner_compress=False, inner_compress_opts=None) -> 'Field':
        """ Curl(F) =  ( d/dy Az - d/dz Ay ) {x} +
                       ( d/dz Ax - d/dx Az ) {y} +
                       ( d/dx Ay - d/dy Ax ) {z}
            gtn_comps:  dictionary of gtn_comps, with CoordinateType/int as keys
            out_comps: tuple of which output components of curl is desired (list of component keys)
        """
        # if out_coords is None:      out_comps = self.axIDs[:3]
        # else:                       out_comps = [self.coords_ax[c].axID for c in out_coords]

        # print('curl out coords', out_coords)

        out_data = self._gtn_curl(field.components, out_comps, compress, compress_opts,
                                  inner_compress, inner_compress_opts)
        return field.create_like(new_components=out_data)


    def divergence(self, field: 'Field', deriv_axes: Iter['Axis']=None,
                   ax_deriv_configs: dict['Axis','DerivativeConfiguration'] = None,
                   compress=False, compress_opts:dict=None,
                   inner_compress=False, inner_compress_opts:dict=None) -> 'ScalarField':
        """ d/dx Ax + d/dy Ay + d/dz Az
            gtn_comps:  dictionary of gtn_comps, with CoordinateType/int as keys
            comps:  components of gtn_comps to include when computing divergence
        """
        if deriv_axes is None:
            deriv_axes = [self.coord_axes.get(c,None) for c in self.coords]

        out_data = self._gtn_divergence(field.components, deriv_axes, ax_deriv_configs, compress, compress_opts,
                                        inner_compress, inner_compress_opts)
        return field.create_like_scalar(new_component=out_data)  # scalar field


    def convective_operator(self, field1: 'Field', field2: 'Field' = None, out_compIDs: Iter=None,
                            # gtn_comps1: dict[Any, 'GridTN'], gtn_comps2: dict[Any, 'GridTN'] = None,
                            compress=False, compress_opts:dict=None,
                            inner_compress=False, inner_compress_opts:dict=None) -> 'Field':
        """ (A \cdot \nabla) B = A \cdot \grad B_x {x} + ... {y} + ....
            gtn_comps1:  A
            gtn_comps2:  B  (if None, A is used)
            out_comps:  field components to compute (default, determined by components of gtn_comps2)
        """
        if field2 is None:        field2 = field1
        if out_compIDs is None:   out_compIDs = field1.componentIDs

        A_comps = field1.componentIDs  # take derivative wrt ax corresponding to comps of A

        out = {}
        for j in A_comps:

            ax = self.coord_axes.get(j,None)
            # if ax is None:      continue ???

            ## d/dx B_x, d/dx B_y, d/dx B_z (which vector fields determined by out_compIDs)
            gradj_B = self._gtn_vector_derivative({oc: field2[oc] for oc in out_compIDs}, ax,
                                                  compress=inner_compress, compress_opts=inner_compress_opts)
            # part of dot product:  Aj * d/dx_j B_x, B_y, B_z
            for i in out_compIDs:
                dot_out = field1[j].elemental_multiply(gradj_B[i], compress=inner_compress,
                                                       compress_opts=inner_compress_opts)
                try:
                    comp_i = out[i]
                    if comp_i is None:
                        out[i] = dot_out
                    else:
                        comp_i.add(dot_out, inplace=True, compress=inner_compress, compress_opts=inner_compress_opts)
                except KeyError:
                    out[i] = dot_out

        if compress:
            for i, comp_i in out.items():
                comp_i.compress(compress_opts=compress_opts)

        return field1.create_like(new_components=out)



    ##########################################
    # methods actually implementing operations
    ##########################################

    # def _gtn_laplacian_mpo(self, grid: 'Grid', axes: Sequence['Axis'] = None,
    #                        ax_deriv_kwargs: Optional[dict[Axis, DerivativeConfiguration]] = None) -> 'GridTN':
    #     """ d^2/dx^2 + d^2/dy^2 + d^2/dz^2
    #         the same for all components
    #     """
    #     raise NotImplementedError
    #
    # def _gtn_vector_laplacian_mpo(self, compID, grid: 'Grid', axes: Sequence['Axis'] = None,
    #                               ax_deriv_kwargs: Optional[dict[Axis, DerivativeConfiguration]] = None) -> 'GridTN':
    #     """ d^2/dx^2 + d^2/dy^2 + d^2/dz^2
    #         the same for all components
    #     """
    #     raise NotImplementedError

    def _gtn_gradient(self, gtn: 'GridTN', axes: Iter['Axis'] = None, compress=False, compress_opts:dict=None):
        """ df/dx {x} + df/dy {y} + df/dz {z}
                    gtn:  'GridTN' (MPS) representing the scalar field
                    coords:  coordinates to include in calculation of Laplacian
                        (CoordinateType.X, .Y, or int)
                """
        raise NotImplementedError

    def _gtn_vector_derivative(self, gtn_comps: dict['Coordinate','GridTN'], ax: 'Axis', compress=False,
                               compress_opts:dict=None):
        """ d/dx (Ax {x}), d/dx (Ay {y}), d/dx (Az {z})
        """
        raise NotImplementedError

    def _gtn_laplacian(self, gtn: 'GridTN', axes: Iter['Axis'] = None, compress=False, compress_opts:dict=None,
                       inner_compress=False, inner_compress_opts:dict=None):
        """ L(f) = d^2/dx^2 f + d^2/dy^2 f + d^2/dz^2 z
            gtn:  GridTN (MPS) representing the scalar field
            coords:  coordinates to include in calculation of Laplacian
                (CoordinateType.X, .Y, or int)
        """
        raise NotImplementedError

    def _gtn_mth_laplacian(self, gtn: 'GridTN', deriv_order:int, axes: Iter['Axis'] = None, compress=False,
                           compress_opts:dict=None, inner_compress=False, inner_compress_opts:dict=None):
        """ L(f) = d^2/dx^2 f + d^2/dy^2 f + d^2/dz^2 z
            gtn:  GridTN (MPS) representing the scalar field
            coords:  coordinates to include in calculation of Laplacian
                (CoordinateType.X, .Y, or int)
        """
        raise NotImplementedError


    def _gtn_vector_laplacian(self, gtn_comps: dict['Coordinate', 'GridTN'], axes: Iter['Axis'] = None,
                              compress=False, compress_opts:dict=None,
                              inner_compress=False, inner_compress_opts:dict=None):
        """ VL(A) = L(Ax) {x} + L(Ay) {y} + L(Az) {z}
            gtn:  GridTN (MPS) representing the scalar field
            coords:  coordinates to include in calculation of Laplacian
                (CoordinateType.X, .Y, or int)
        """
        raise NotImplementedError

    def _gtn_curl(self, gtn_comps: dict['Coordinate','GridTN'], out_comps: Iter = None,
                  compress=False, compress_opts:dict=None,
                  inner_compress=False, inner_compress_opts:dict=None):
        """ Curl(F) =  ( d/dy Az - d/dz Ay ) {x} +
                       ( d/dz Ax - d/dx Az ) {y} +
                       ( d/dx Ay - d/dy Ax ) {z}
            gtn_comps:  dictionary of gtn_comps, with CoordinateType/int as keys
            out_comps: tuple of which output components of curl is desired (list of component keys)
        """
        raise NotImplementedError

    def _gtn_divergence(self, gtn_comps: dict['Coordinate','GridTN'], axes: Iter['Axis']= None,
                        ax_deriv_configs: dict['Axis','DerivativeConfiguration'] = None,
                        compress=False, compress_opts:dict=None,
                        inner_compress=False, inner_compress_opts:dict=None):
        """ d/dx Ax + d/dy Ay + d/dz Az
            gtn_comps:  dictionary of gtn_comps, with CoordinateType/int as keys
            comps:  components of gtn_comps to include when computing divergence
        """
        raise NotImplementedError

    # def _gtn_vector_derivatives(self, gtn_comps: dict[Any, 'GridTN'], ax: 'Axis', deriv_params:dict=None, compress=True,
    #                             compress_opts:dict=None):
    #     """ d/dx (Ax {x}), d/dx (Ay {y}), d/dx (Az {z})
    #         for spherical, cylindrical coordinates, need to account for d{i}/dx
    #     """
    #     raise NotImplementedError


    def _gtn_curl_mpos(self, gtn_comps: dict['Coordinate','GridTN'], out_comps: Iter = None,
                       compress=False, compress_opts:dict=None,
                       inner_compress=False, inner_compress_opts:dict=None):
        """ Curl(F) =  ( d/dy Az - d/dz Ay ) {x} +
                       ( d/dz Ax - d/dx Az ) {y} +
                       ( d/dx Ay - d/dy Ax ) {z}
            gtn_comps:  dictionary of gtn_comps, with CoordinateType/int as keys
            out_comps: tuple of which output components of curl is desired (list of component keys)
        """
        raise NotImplementedError

    def _gtn_divergence_mpos(self, gtn_comps: 'Field', axes: Iter['Axis']= None,
                             ax_deriv_configs: dict['Axis','DerivativeConfiguration'] = None,
                             compress=False, compress_opts:dict=None,
                             inner_compress=False, inner_compress_opts:dict=None):
        """ d/dx Ax + d/dy Ay + d/dz Az
            gtn_comps:  dictionary of gtn_comps, with CoordinateType/int as keys
            comps:  components of gtn_comps to include when computing divergence
        """
        raise NotImplementedError

    def _gtn_gradient_mpos(self, gtn_comps: dict['Coordinate','GridTN'], axes: Iter['Axis']= None,
                           ax_deriv_configs: dict['Axis','DerivativeConfiguration'] = None,
                           compress=False, compress_opts:dict=None,
                           inner_compress=False, inner_compress_opts:dict=None):
        """ d/dx Ax + d/dy Ay + d/dz Az
            gtn_comps:  dictionary of gtn_comps, with CoordinateType/int as keys
            comps:  components of gtn_comps to include when computing divergence
        """
        raise NotImplementedError