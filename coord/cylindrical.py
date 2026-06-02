"""Cylindrical coordinate system (R, THETA, Z): a partial :class:`CoordinateSystem`
implementation providing the radial/azimuthal/axial differential operators."""
from setup_.configs import *
import helper_quimb as helper
from setup_.enums import CoordinateType
from coord_sys import CoordinateSystem
from axis import Axis
from gridTN import GridTN


class CylindricalCoordinateSpace(CoordinateSystem):

    def get_R(self) -> Axis:
        return self.coords_ax[CoordinateType.R]

    def get_THETA(self) -> Axis:
        return self.coords_ax[CoordinateType.THETA]

    def get_Z(self) -> Axis:
        return self.coords_ax[CoordinateType.Z]

    def is_R(self,ax):
        return ax == self.coords_ax[CoordinateType.R]

    def is_THETA(self,ax):
        return ax == self.coords_ax[CoordinateType.THETA]

    def is_Z(self,ax):
        return ax == self.coords_ax[CoordinateType.Z]

    def _get_inv_R_vals_mpo(self):
        axR = self.get_R()
        rvals = axR.xpts
        rvals_inv = axR.map_state_to_mps(1. / rvals)
        rvals_inv = helper.mps_to_diag_mpo(rvals_inv)
        return rvals_inv


    def build_firstderivative_mpo(self, ax, deriv_config):
        coord = ax.coord_type
        if coord == CoordinateType.R:
            return self._build_firstderivative_mpo_R(**deriv_config)
        elif coord == CoordinateType.THETA:
            return self._build_firstderivative_mpo_THETA(**deriv_config)
        elif coord == CoordinateType.Z:
            return self._build_firstderivative_mpo_Z(**deriv_config)

    def build_secondderivative_mpo(self, ax1, ax2, deriv_config1, deriv_config2=None):
        raise NotImplementedError

    def build_integral_mps(self, ax, is_sqrt=False):
        raise NotImplementedError

    def build_coarse_grain_mpx(self, ax:'Axis', depth:int, is_sqrt=False) -> MPOType:
        raise NotImplementedError


    # def take_firstderivative(self, gtn: GridTN, coord, deriv_params=None, inplace=False, compress=True,
    #                          **compress_opts):
    #     """ df/dx_i
    #         deriv_params:   bc = boundary condition
    #                         order = finite difference stencil order,
    #                         fd_type:  'center', 'forwards', 'backwards'
    #     """
    #     gtn = gtn if inplace else gtn.copy(deep=False)  # data is overwritten in apply anyways (not modified)
    #     if deriv_params is None:  deriv_params = {}
    #
    #     mpo_ax_dict = self.build_firstderivative_mpo(coord, **deriv_params)
    #     ddx_mpo = gtn.grid.make_mpo_ndim(mpo_ax_dict)
    #     gtn.apply(ddx_mpo, inplace=True, compress=compress, **compress_opts)
    #     return gtn
    #
    # def take_secondderivative(self, gtn, coord1, coord2=None,
    #                           deriv_params=None, inplace=False, compress=True, **compress_opts):
    #     """ df/dx_i
    #         deriv_params:   bc = boundary condition
    #                         order = finite difference stencil order
    #     """
    #     ### include component number of gtn?
    #     ### make the definition for fields?
    #     raise NotImplementedError

    def build_laplacian_mpo(self, axes: Sequence['Axis'], deriv_configs: dict['Axis','DerivativeConfiguration'])\
            -> list[dict[Any, MPOType]]:
        """ obtain MPO taking Laplacain over specified axes
        """
        raise NotImplementedError


    def _gtn_gradient(self, gtn: GridTN, axes: list[int] = None, compress=False, compress_opts=None):
        """ df/dx {x} + df/dy {y} + df/dz {z}
                    gtn:  GridTN (MPS) representing the scalar field
                    coords:  coordinates to include in calculation of Laplacian
                        (CoordinateType.X, .Y, or int)
                """
        # can be implemented in coord_sys or in field?
        raise NotImplementedError

    def _gtn_laplacian(self, gtn: GridTN, axes: list[int] = None, compress=False, compress_opts=None,
                       inner_compress=False, inner_compress_opts=None):
        """ L(f) = d^2/dx^2 f + d^2/dy^2 f + d^2/dz^2 z
            gtn:  GridTN (MPS) representing the scalar field
            coords:  coordinates to include in calculation of Laplacian
                (CoordinateType.X, .Y, or int)
        """
        # can be implemented in coord_sys or in field?
        raise NotImplementedError

    def _gtn_vector_laplacian(self, gtn_comps: dict[int, GridTN], axes: list[int] = None, compress=False,
                              compress_opts=None):
        """ VL(A) = L(Ax) {x} + L(Ay) {y} + L(Az) {z}
            gtn:  GridTN (MPS) representing the scalar field
            coords:  coordinates to include in calculation of Laplacian
                (CoordinateType.X, .Y, or int)
        """
        raise NotImplementedError

    def _gtn_curl(self, gtn_comps: dict[int, GridTN], out_comps: list[int] = None, compress=False, compress_opts=None):
        """ Curl(F) =  ( d/dy Az - d/dz Ay ) {x} +
                       ( d/dz Ax - d/dx Az ) {y} +
                       ( d/dx Ay - d/dy Ax ) {z}
            gtn_comps:  dictionary of gtn_comps, with CoordinateType/int as keys
            out_comps: tuple of which output components of curl is desired (list of component keys)
        """
        raise NotImplementedError

    def _gtn_divergence(self, gtn_comps: dict[int, GridTN], axes: list[int] = None, compress=False, compress_opts=None,
                        inner_compress=False, inner_compress_opts=None):
        """ d/dx Ax + d/dy Ay + d/dz Az
            gtn_comps:  dictionary of gtn_comps, with CoordinateType/int as keys
            comps:  components of gtn_comps to include when computing divergence
        """
        raise NotImplementedError

    def _gtn_vector_derivative(self, gtn_comps: dict[int, GridTN], ax: int, compress=False, compress_opts=None):
        """ d/dx (Ax {x}), d/dx (Ay {y}), d/dx (Az {z})
            for spherical, cylindrical coordinates, need to account for d{i}/dx
        """
        raise NotImplementedError



    #### define scalar field derivative MPOs ####
    def _build_firstderivative_mpo_R(self, **deriv_params):
        ax = self.get_R()
        ddx_mpo = ax.build_firstderivative_mpo(**deriv_params)
        return {ax.coordinate: ddx_mpo}

    def _build_firstderivative_mpo_THETA(self, **deriv_params):
        ax = self.get_THETA()
        ddx_mpo = ax.build_firstderivative_mpo(**deriv_params)
        axR = self.get_R()
        rvals = axR.xpts
        rvals_inv = axR.map_state_to_mps(1. / rvals)
        rvals_inv = helper.mps_to_diag_mpo(rvals_inv)
        return {axR.coordinate: rvals_inv, ax.coordinate: ddx_mpo}

    def _build_firstderivative_mpo_Z(self, **deriv_params):
        ax = self.get_Z()
        ddx_mpo = ax.build_firstderivative_mpo(**deriv_params)
        return {ax.coordinate: ddx_mpo}


    def _take_firstderivative_THETA_R(self, gtn_comps: dict[int,GridTN], deriv_params=None, compress=True,
                                      **compress_opts):
        """ MPO to take derivative of R component along THETA axis
        """
        gtn_R = gtn_comps[CoordinateType.R]
        if deriv_params is None:  deriv_params = {}
        ## deriv of unit vector R-HAT
        rvals_inv = self._get_inv_R_vals_mpo()
        rvals_inv_gtn = gtn_R.grid.make_mpo_ndim({self.get_R().coordinate: rvals_inv})
        gtn_TH = gtn_R.apply(rvals_inv_gtn, inplace=False, compress=compress, **compress_opts)
        ## deriv of R field component
        gtn_R = self.take_firstderivative(gtn_R, CoordinateType.THETA, deriv_params=deriv_params,
                                        inplace=False, compress=compress, **compress_opts)
        return {CoordinateType.R: gtn_R, CoordinateType.THETA: gtn_TH}


    def _take_firstderivative_THETA_THETA(self, gtn_comps: dict[int,GridTN], deriv_params=None, compress=True,
                                          **compress_opts):
        """ MPO to take derivative of R component along THETA axis
        """
        gtn_TH = gtn_comps[CoordinateType.THETA]
        if deriv_params is None:  deriv_params = {}
        ## deriv of unit vector THETA-HAT
        rvals_inv = self._get_inv_R_vals_mpo()
        rvals_inv_gtn = gtn_TH.grid.make_mpo_ndim({self.get_R().coordinate: rvals_inv})
        gtn_R = gtn_TH.apply(rvals_inv_gtn, inplace=False, compress=compress, **compress_opts)
        gtn_R.scalar_multiply(-1, inplace=True)  # apply is not an MPS deep operation
        ## deriv of THETA field component
        gtn_TH = self.take_firstderivative(gtn_TH, CoordinateType.THETA, deriv_params=deriv_params,
                                        inplace=False, compress=compress, **compress_opts)
        return {CoordinateType.R: gtn_R, CoordinateType.THETA: gtn_TH}


    #### define vector field derivative MPOs ####
    def _build_firstderivative_mpo_R_R(self, **deriv_params):
        """ MPO to take derivative of R component along R axis
        """
        return {CoordinateType.R: self._build_firstderivative_mpo_R(**deriv_params)}

    def _build_firstderivative_mpo_R_THETA(self, **deriv_params):
        """ MPO to take derivative of THETA component along R axis
        """
        return {CoordinateType.THETA: self._build_firstderivative_mpo_R(**deriv_params)}

    def _build_firstderivative_mpo_R_Z(self, **deriv_params):
        """ MPO to take derivative of z component along R axis
        """
        return {CoordinateType.Z: self._build_firstderivative_mpo_R(**deriv_params)}



