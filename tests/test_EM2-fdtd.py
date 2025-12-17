import os, sys, time, glob
sys.path.append('../')

from setup_.configs import *

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
# from pde_vlasovEM import VlasovMaxwell
from pde_EM import Maxwell

save_figs = False
save_data = False
load_data = False
restart = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

"""
EM pulse with staggered Yee cell 
http://ammar-hakim.org/sj/je/je6/je6-maxwell-solvers.html
"""

if restart or save_data or save_figs:
    sdir = '/pool001/erikaye/tns_data/EM-fdtd-TM/'
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    fdir = 'data/EM-fdtd-TM/'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    print('fdir', fdir)


#######################
# problem parameters #
#######################

permittivity = 1.0 # 1/100  # 1.0    # permittivity of free space
permeability = 1.0    # permeability of free space
eV = 1.0              # elementary charge or Joule-eV conversion factor
matl_config = UnitsConfiguration(eps0=permittivity, mu0=permeability)

## initial conditions ##
# cavity size
Lx = 1.0
Ly = 1.0

E0 = 1.         # electric field magnitude
beta = 25       # width of the initial Gaussian pulse
c = 1/np.sqrt(permittivity*permeability)    # speed of light

#########################
# EM pulse in 2D cavity #
#########################

q = 2
L = 6
K = 2
npts = q**L
TN_layout = LayoutType.SEQUENTIAL

print('c', matl_config.c)
T = 3.0 * (Lx / matl_config.c)  # total simulation time
cfl = 0.75                      # fraction of maximum allowed time step (according to CFL)
order = 1                       # order of finite difference spatial derivative
te_order = 31                   # specify time evolution scheme:  31 = FDTD, 4 = RK4.

save_every_nt = 20      # save points (in units of time steps)

##### compression options ####
## even with modest CUTOFF = 1.0e-8, div(B) is not preserved
DMAX = None
compress_style = 1
compress_style_mod = 0

te_compress_E = CompressionConfiguration()
te_compress_phi = CompressionConfiguration()

for ci in range(1,compress_style+1):
    te_compress_E.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)
    te_compress_phi.set_compress_opts(ci, max_bond=None, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)

te_compress_psi = te_compress_phi
te_compress_B = te_compress_E

comp_levels = list(range(1,6))
if compress_style_mod == 1:   # don't compress inside the derivative
    comp_levels = [1,2,3,4,0]
elif compress_style_mod == 2:   # don't compress inside the derivative
    comp_levels = [1,2,3,0,0]

#############################
## start building PDE system
#############################

## define coordinates (will label axes and vector components)
X = Coordinate('X', CoordinateType.X)
Y = Coordinate('Y', CoordinateType.Y)
Z = Coordinate('Z', CoordinateType.Z)

## define axis objects
x_vals = np.linspace(-Lx,Lx,npts,endpoint=False)
x1_vals = x_vals + (x_vals[1]-x_vals[0])/2
y_vals = np.linspace(-Lx,Ly,npts,endpoint=False)
y1_vals = y_vals + (y_vals[1]-y_vals[0])/2

ax_x = Axis(L, q, coordinate=X, xpts=x_vals)
ax_y = Axis(L, q, coordinate=Y, xpts=y_vals)
## below: in theory one can build a not-quantized TT, though it's less developed
# ax_x = Axis(1, q**L, coordinate=X, xpts=x_vals)
# ax_y = Axis(1, q**L, coordinate=Y, xpts=y_vals)

## define a Cartesian coordinate system
coord_sys = CartesianCoordinateSpace('cart', ax_x, ax_y, coords=[X,Y,Z])

## define time step
dx, dy = ax_x.dx, ax_y.dx
cfl_params = ((dx, dy), (c, c))
dt = cfl * cfl_limit( cfl_params )
num_tsteps = int(T/dt)
print('dt', dt, cfl*1./(c/ax_x.dx + c/ax_y.dx), num_tsteps)
print('dx', dx, 'dy', dy)

## define grids
grid_xy = Grid1D('XY', (ax_x,ax_y), layout_type=TN_layout)

## define initial grid1DTNs
Ez = lambda x, y: E0*np.exp(-beta*(x**2 + y**2))
# ## B at t=1/2*dt
# By = lambda x, y: E0*(beta*2*x)*np.exp(-beta*(x**2 + y**2)) * dt/2
# Bx = lambda x, y: -E0*(beta*2*y)*np.exp(-beta*(x**2 + y**2)) * dt/2
# def Bx(x, y):
#     return - E0 * np.exp(-beta * x**2) * (np.exp(-beta*(y+dy/2)**2) - np.exp(-beta*(y-dy/2)**2)) / dy * dt/2
#
# def By(x, y):
#     return E0 * np.exp(-beta * y**2) * (np.exp(-beta*(x+dx/2)**2) - np.exp(-beta*(x-dx/2)**2)) / dx * dt/2


data_x, data_y = np.meshgrid(x_vals,y_vals,indexing='ij')
data_x1, data_y1 = np.meshgrid(x1_vals,y1_vals,indexing='ij')
init_Ez = Ez(data_x, data_y)   # i, j, k+1/2
# init_Bx = Bx(data_x, data_y)  # i, j+1/2, k+1/2  # init at time dt/2
# init_By = By(data_x, data_y)  # i+1/2, j, k+1/2  # init at time dt/2

plt.figure()
plt.imshow(init_Ez)
plt.colorbar()
plt.show()

## configure derivatives and set them
deriv_zg = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)
deriv_zv = DerivativeConfiguration(left_bc=BCType.ZEROVALUE, order=order, fd_type=FDType.CENTER)
deriv_zg1 = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER, offset=1)
deriv_zv1 = DerivativeConfiguration(left_bc=BCType.ZEROVALUE, order=order, fd_type=FDType.CENTER, offset=1)

init_Ex_gtn = grid_xy.make_empty_gridTN()
init_Ey_gtn = grid_xy.make_empty_gridTN()
init_Ez_gtn = grid_xy.map_state_to_mps(init_Ez, ax_deriv_configs={ax_x: deriv_zv, ax_y: deriv_zv})      # i, j, k+1/2
# init_Bx_gtn = grid_xy.map_state_to_mps(init_Bx, ax_deriv_configs={ax_x: deriv_zv, ax_y: deriv_zg1})     # i, j+1/2, k+1/2
# init_By_gtn = grid_xy.map_state_to_mps(init_By, ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zv})     # i+1/2, j, k+1/2
init_Bx_gtn = grid_xy.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zv, ax_y: deriv_zg1})     # i, j+1/2, k+1/2
init_By_gtn = grid_xy.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg1, ax_y: deriv_zv})     # i+1/2, j, k+1/2


init_E_field = Field('E', grid_xy, data={X: init_Ex_gtn, Y: init_Ey_gtn, Z: init_Ez_gtn},
                     compress_config=te_compress_E)
init_B_field = Field('B', grid_xy, data={X: init_Bx_gtn, Y:init_By_gtn}, compress_config=te_compress_B)


### BUILD PDE
em_sys = Maxwell(init_E_field, init_B_field, phi=None, psi=None,
                     coords_x=coord_sys, matl_params=matl_config, 
                     is_electrostatic=False, clean=False, is_yee=True, normalize=False,
                     te_order=te_order, compress_levels=comp_levels )

for compID, comp in em_sys.B.components.items():
    for ax in em_sys.grid_X.axes:
        deriv_config = comp.ax_deriv_configs[ax].copy()
        deriv_config.update(order=0, fd_type=FDType.FORWARD)
        comp.ax_deriv_configs[ax] = deriv_config

## div(B) at initialization
divB = em_sys.get_divB()  # B.divergence(coord_sys=coord_sys)
divB_data_0 = divB.get_comp_data()
if divB_data_0 is not None:
    plt.figure()
    plt.imshow(divB_data_0)
    plt.title('div B')
    plt.colorbar()
    plt.show()
    print('norm divB', np.linalg.norm(divB_data_0))
else:
    divB_data_0 = 0.


nt = 0
Ez_t  = [init_Ez]
max_bond_Es = [[0, 0, init_E_field[Z].max_bond()]]
max_bond_Bs = [[0,0,0]]
divBs = [0.]
ts = [0.]

while nt < num_tsteps:
    em_sys = em_sys.next_time_step(dt, compress_level=1, is_first_time_step=(nt==0))
    ## set is_first_time_step=False if already initialized B at time t=dt/2

    Ez_data = em_sys.E.get_comp_data(Z)
    Ez_t += [Ez_data]
    E_bonds = [em_sys.E.max_bond(C) for C in [X,Y,Z]]
    print('E bonds', E_bonds)
    max_bond_Es += [E_bonds]

    B_bonds = [em_sys.B[C].max_bond() for C in [X,Y,Z]]
    print('B bonds', B_bonds)
    max_bond_Bs += [B_bonds]

    divBs += [em_sys.check_divB().norm()]
    print('div B', divBs[-1])
    print('dt', dt)
    ts += [ts[-1] + dt]

    if np.abs(nt*dt-T/2) < dt/2 or np.abs(nt*dt-T) < dt/2:
        plt.figure()
        plt.imshow(Ez_data)
        plt.xlabel('y')
        plt.ylabel('x')
        plt.colorbar()
        plt.title(f'Ez(x, y, t={nt*dt:1.3f})')
        plt.show()

    if nt%save_every_nt==0:
        plt.figure()
        plt.plot(x_vals, Ez_data[:, npts // 2 - 1])
        plt.xlabel('x')
        plt.ylabel('Ez')
        plt.title(f'E(x, y={y_vals[npts//2]:1.2f}, t={nt*dt:1.3f})')

        plt.figure()
        plt.plot(ts, divBs)
        plt.xlabel('t')
        plt.ylabel('|div(B)|')

        plt.show()

    nt += 1
    # norm = vp_sys.fe.integrate('all')[0] - vp_sys.fi.integrate('all')[0]
    # norm = np.sum(fe_data)*dx*dv - np.sum(init_fi)*dx*dv
    print('result norm', nt, em_sys.E[Z].max_bond(), q ** L)


    divB = em_sys.get_divB()
    divB_data = divB.get_comp_data()
    # plt.figure()
    # plt.imshow(divB_data)
    # plt.title('div B')
    # plt.colorbar()
    # plt.show()
    print('norm d/dt divB', np.linalg.norm(divB_data-divB_data_0))

Ez_data = em_sys.E.get_comp_data(Z)

plt.figure()
plt.plot(x_vals, Ez_data[:, npts // 2 - 1])
plt.xlabel('x')
plt.ylabel('Ez')
plt.title(f'E(x, y={y_vals[npts//2]:1.2f}, t={nt*dt:1.3f})')

plt.figure()
plt.imshow(Ez_data)
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()
plt.title(f'Ez(x, y, t={nt*dt:1.3f})')

plt.figure()
plt.plot(ts, divBs)
plt.xlabel('t')
plt.ylabel('|div(B)|')

plt.show()


