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
x_vals = np.linspace(0, 100, q**Lx, endpoint=False)
y_vals = np.linspace(0, 100, q**Lx, endpoint=False)


## load data
x_mesh, y_mesh = np.meshgrid(x_vals, y_vals, indexing='ij')
data = np.cos(x_mesh)  * np.sin(y_mesh)

## QTT rank truncation / compression parameters
max_rank = None   # choose QTT rank adaptively. Can also set to an integer value.
tol = 1.0e-08     # accuracy roughly within tol

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

## map data to QTT
qtt = grid.map_state_to_mps(data, split_opts={'max_bond': max_rank, 'cutoff': tol**2})

print('max rank', qtt.max_bond())
print('internal bond sizes', helper.inner_bond_sizes(qtt.data))
print('entanglement entropy at all bonds', helper.entanglement_entropy_all(qtt.data))

low_rank_data = qtt.get_data()

print('error', np.linalg.norm(low_rank_data - data) / np.linalg.norm(data))
