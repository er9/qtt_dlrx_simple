"""Example driver: QTT inverse-Laplacian operator on a 2-D radial-wave dataset.

Builds a 2-D quantized-tensor-train grid, constructs the inverse-Laplacian MPO,
and measures the QTT rank and entanglement entropy of the resulting data.
"""
import os, sys, pickle, time, glob
sys.path.append('../')

from setup_.configs import *
import setup_.helper as helper_test
import helper_quimb as helper

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D

"""
Measure rank and entanglement entropy of a 2-D data set
"""


q = 2
Lx, Ly = 6, 6
npts_x, npts_y = q**Lx, q**Ly

## grid paramters
x_vals = np.linspace(-10, 10, q**Lx, endpoint=True)
y_vals = np.linspace(-10, 10, q**Lx, endpoint=True)


## load data
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing='ij')
k = 3
data = np.real(np.exp(1.j * k * np.sqrt(x_mesh ** 2 + y_mesh ** 2)))
offset = 2

## QTT rank truncation / compression parameters
max_rank = None   # choose QTT rank adaptively. Can also set to an integer value.
tol = 1.0e-6     # accuracy roughly within tol

## define axis objects
X = Coordinate('X', CoordinateType.X)
Y = Coordinate('Y', CoordinateType.Y)

TN_layout = LayoutType.SEQUENTIAL       ## one of LayoutType.SEQUENTIAL, LayoutType.PARALLEL_GROUP
ax_map_key = 'FB'                       ## one of 'FF','BF','FB','BB'  ('FF' for PARALLEL_GROUP)

map_x, map_y = get_maps(ax_map_key)
print('maps', map_x, map_y)
ax_x = Axis(Lx, q, coordinate=X, xpts=x_vals, ax_map=map_x)
ax_y = Axis(Ly, q, coordinate=Y, xpts=y_vals, ax_map=map_y)


coords_x = CartesianCoordinateSpace('X', ax_x, ax_y)

## configure derivatives  (not relevant here but will be useful later)
deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=1, fd_type=FDType.CENTER)

## define grid
grid = Grid1D('XVi', (ax_x, ax_y), layout_type=TN_layout)

ax_deriv_configs = {ax_x: DerivativeConfiguration(left_bc=BCType.PERIODIC),
                    ax_y: DerivativeConfiguration(left_bc=BCType.PERIODIC),
                    }
grid.inverse_laplacian_mpo(ax_deriv_configs=ax_deriv_configs)
print('max rank', 2**Lx)
