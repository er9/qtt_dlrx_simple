import numpy as np
from setup_.configs import *
import helper_quimb as helper
from setup_.enums import CoordinateType
from coord_sys import CoordinateSystem
from axis import Axis

class SphericalCoordinateSpace(CoordinateSystem):

    def get_R(self) -> Axis:
        return self.coords_ax[CoordinateType.R]

    def get_THETA(self) -> Axis:
        return self.coords_ax[CoordinateType.THETA]

    def get_PHI(self) -> Axis:
        return self.coords_ax[CoordinateType.PHI]

    def is_R(self,ax):
        return ax == self.coords_ax[CoordinateType.R]

    def is_THETA(self,ax):
        return ax == self.coords_ax[CoordinateType.THETA]

    def is_PHI(self,ax):
        return ax == self.coords_ax[CoordinateType.PHI]


    def build_mpo_firstderivative(self, ax, coord_type, **deriv_params):
        if self.is_R(ax):
            return self._build_mpo_firstderivative_R(ax, **deriv_params)
        elif self.is_THETA(ax):
            return self._build_mpo_firstderivative_THETA(ax, **deriv_params)
        elif self.is_PHI(ax):
            return self._build_mpo_firstderivative_PHI(ax, **deriv_params)

    def build_mpo_secondderivative(self, ax, coord_type, **deriv_params):
        raise NotImplementedError


    def build_laplacian_mpo(self, axes: Sequence['Axis'], deriv_configs: dict['Axis','DerivativeConfiguration'])\
            -> list[dict[Any, MPOType]]:
        """ obtain MPO taking Laplacain over specified axes
        """
        raise NotImplementedError


    def _build_mpo_firstderivative_R(self, ax, **deriv_params):
        ddx_mpo = ax.build_firstderivative_mpo(DEFAULT_BC, **deriv_params)
        return {ax.coordinate: ddx_mpo}

    def _build_mpo_firstderivative_THETA(self, ax, **deriv_params):
        ddx_mpo = ax.build_firstderivative_mpo(DEFAULT_BC, **deriv_params)
        axR = self.get_R()
        rvals = axR.xpts
        rvals_inv = axR.map_state_to_mps(1. / rvals)
        rvals_inv = helper.mps_to_diag_mpo(rvals_inv)
        return {axR.coordinate: rvals_inv, ax.coordinate: ddx_mpo}

    def _build_mpo_firstderivative_PHI(self, ax, **deriv_params):
        ddx_mpo = ax.build_firstderivative_mpo(DEFAULT_BC, **deriv_params)
        axR = self.get_R()
        rvals = axR.xpts
        rvals_inv = axR.map_state_to_mps(1. / rvals)
        rvals_inv = helper.mps_to_diag_mpo(rvals_inv)
        axTH = self.get_THETA()
        thetas = axTH.xpts
        thetas_inv = axR.map_state_to_mps(1. / np.sin(thetas))
        thetas_inv = helper.mps_to_diag_mpo(thetas_inv)
        return {axR.coordinate: rvals_inv, axTH.coordinate: thetas_inv, ax.coordinate: ddx_mpo}

    def build_integral_mps(self, ax, is_sqrt=False):
        raise NotImplementedError

    def build_coarse_grain_mpx(self, ax: 'Axis', depth: int, is_sqrt=False) -> MPOType:
        raise NotImplementedError