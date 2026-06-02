"""Shared configuration helpers for the plasma/Vlasov test drivers.

Provides factory functions for plasma unit/species configurations, grid axes,
and Maxwellian distributions used to set up the Vlasov simulations. Not a
standalone test itself.
"""
import os, sys, pickle, time, glob

sys.path.append('../')

from setup_.configs import *
import setup_.helper as helper_test
import helper_quimb as helper

from axis import Axis
import axis_map
from basis.basis_spatial import SpatialBasis
from basis.basis_k import FourierBasis
from basis.basis_hermite import AWHermiteGaussianBasis
from basis.basis_hermite import HermiteGaussianBasis
from coord.coord_sys import Coordinate


def maxwellian(*vs, vth2=1.0, density=1.0, flow=0.0):
    """ vs: velocity grid for each dimension
            summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
        flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    return helper_test.maxwellian(*vs, vth2=vth2, density=density, flow=flow)



def config_params(units_config: UnitsConfiguration = None, mass_e=1.0, mass_ratio=1836, 
                  n0_e=1.0, n0_ratio=1.0, T_e=1.0, T_ratio = 1.0
                  ):

    plasma_config = UnitsConfiguration() if units_config is None else units_config
    
    mass_i = mass_ratio * mass_e  # mass of ion
    n0_i = n0_ratio * n0_e  # ion number density
    T_i = T_ratio * T_e  # ion temperature [eV]
    
    ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
    elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)

    return {'ion': ion_config, 'elc': elc_config}




def config_axes(Lxs=6, Lves=6, Lvis=6, K=1, q=2, k=0.10,
                x_lims = (-1,1), ve_lims = (-1, 1), vi_lims = (-1, 1),
                x_map_key='F', ve_map_key='F', vi_map_key=None,
                basis_xs=SpatialBasis(), basis_ves=SpatialBasis(), basis_vis=None,
                x_coords=None, ve_coords=None, vi_coords=None,
                ):

    outputs = {}

    if basis_vis is None:
        basis_vis = basis_ves

    if vi_map_key is None:
        vi_map_key = ve_map_key

    if x_coords is None:
        x_coords = (CoordinateType.X, CoordinateType.Y, CoordinateType.Z)

    if ve_coords is None:
        ve_coords = (CoordinateType.X, CoordinateType.Y, CoordinateType.Z)

    if vi_coords is None:
        vi_coords = ve_coords


    #############################
    ## start building PDE system
    #############################

    ## specify k values, hermite moments used
    x_axes, ve_axes, vi_axes = (), (), ()
    x_maps  = helper_test.get_maps(x_map_key)
    ve_maps = helper_test.get_maps(ve_map_key)
    vi_maps = helper_test.get_maps(vi_map_key)

    for dim in range(K):

        Lx = Lxs[dim] if isinstance(Lxs, tuple) else Lxs
        Lve = Lves[dim] if isinstance(Lves, tuple) else Lves
        Lvi = Lvis[dim] if isinstance(Lvis, tuple) else Lvis

        basis_x = basis_xs[dim] if isinstance(basis_xs, tuple) else basis_xs
        basis_ve = basis_ves[dim] if isinstance(basis_ves, tuple) else basis_ves
        basis_vi = basis_vis[dim] if isinstance(basis_vis, tuple) else basis_vis

        x_map = x_maps[dim] if len(x_maps) == K else x_maps[0]
        ve_map = ve_maps[dim] if len(ve_maps) == K else ve_maps[0]
        vi_map = vi_maps[dim] if len(vi_maps) == K else vi_maps[0]

        x_lim = x_lims[dim] if isinstance(x_lims[dim], tuple) else x_lims
        ve_lim = ve_lims[dim] if isinstance(ve_lims[dim], tuple) else ve_lims
        vi_lim = vi_lims[dim] if isinstance(vi_lims[dim], tuple) else vi_lims

        x_coord = x_coords[dim]
        ve_coord = ve_coords[dim]
        vi_coord = vi_coords[dim]


        # dk = 0.10 -- lowest frequency mode
        npts_x, npts_ve, npts_vi = q ** Lx, q ** Lve, q ** Lvi

        if isinstance(basis_x, FourierBasis):
            kmax = k * npts_x / 2
            x_vals = np.linspace(-kmax, kmax, npts_x, endpoint=False)
        elif isinstance(basis_x, SpatialBasis):
            x_vals = np.linspace(*x_lim, npts_x, endpoint=False)
        else:
            raise NotImplementedError

        if isinstance(basis_ve, AWHermiteGaussianBasis):
            ve_vals = np.arange(npts_ve)
        elif isinstance(basis_ve, SpatialBasis):
            ve_vals = np.linspace(*ve_lim,npts_ve,endpoint=False)
        else:
            raise NotImplementedError

        if isinstance(basis_vi, AWHermiteGaussianBasis):
            vi_vals = np.arange(npts_vi)
        elif isinstance(basis_vi, SpatialBasis):
            vi_vals = np.linspace(*vi_lim,npts_vi,endpoint=False)
        else:
            raise NotImplementedError

        ## define axis objects; position -> k-space,
        X = Coordinate(f'X{dim}', x_coord)
        VE = Coordinate(f'V{dim}', ve_coord)
        VI = Coordinate(f'V{dim}', vi_coord)

        ax_x  = Axis(Lx , q, coordinate=X , xpts=x_vals , basis=basis_x , ax_map=x_map)
        ax_ve = Axis(Lve, q, coordinate=VE, xpts=ve_vals, basis=basis_ve, ax_map=ve_map)
        ax_vi = Axis(Lvi, q, coordinate=VI, xpts=vi_vals, basis=basis_vi, ax_map=vi_map)

        x_axes = x_axes + (ax_x,)
        ve_axes = ve_axes + (ax_ve,)
        vi_axes = vi_axes + (ax_vi,)

    # coords_x = CartesianCoordinateSpace('X', *x_axes)
    # coords_vi = CartesianCoordinateSpace('Vi', *vi_axes)
    # coords_ve = CartesianCoordinateSpace('Ve', *ve_axes)

    outputs['x_axes'] = x_axes
    outputs['vi_axes'] = vi_axes
    outputs['ve_axes'] = ve_axes

    return outputs


