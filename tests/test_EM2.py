import os, sys, time, glob

import numpy as np

sys.path.append('../')

from setup_.defaults import *
from setup_.configs import *

from axis import Axis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from grid_comb import GridsComb
from field import Field, ScalarField
# from pde_vlasovEM import VlasovMaxwell
from pde_EM import Maxwell


"""
EM pulse with staggered Yee cell 
http://ammar-hakim.org/sj/je/je6/je6-maxwell-solvers.html
"""


save_figs = False
save_data = True
load_data = False
restart = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)

use_mask = 'box'  # 'box'

if restart or save_data or save_figs:
    # sdir = '/pool001/erikaye/tns_data/VM-TM/'
    # if not os.path.exists(sdir):
    #     os.makedirs(sdir)

    # fdir = 'data_cross/EM2_251013/'
    fdir = 'data_cross/EM2_251104_xx1/'
    # fdir = 'data_cross/EM2_251020/'
    os.makedirs(fdir, exist_ok=True)

    print('fdir', fdir)

#######################
# standard parameters #
#######################

permittivity = 1.0  # permittivity of free space
permeability = 1.0  # permeability of free space
eV = 1.0  # elementary charge or Joule-eV conversion factor
matl_config = UnitsConfiguration(eps0=permittivity, mu0=permeability)

#########################
# EM pulse in 2D cavity #
#########################

q = 2
L = flags.get('Lx', 8)
K = 2
npts = q ** L
# TN_layout = LayoutType.COMB  # LayoutType.SEQUENTIAL  # PARALLEL_GROUP
# TN_layout = LayoutType.PARALLEL_GROUP  # LayoutType.SEQUENTIAL
TN_layout = LayoutType.SEQUENTIAL

T = 1.5
cfl = 0.75
order = 1
te_order = flags.get('te_order', 441)
te_order = flags.get('te_order_EM', te_order)
## 331:  orthogonal td-dmrg + Euler
## 771:  mixed td-dmrg + Euler
## 441:  cross td-dmrg + Euler


## initial conditions
Lx = 1.0
Ly = 1.0

E0 = 1.
beta = 25 if use_mask is None else 25 * 4
Ez = lambda x, y: E0 * np.exp(-beta * (x ** 2 + y ** 2))
c = 1 / np.sqrt(permittivity * permeability)

#############################
## start building PDE system
#############################

##### compression options ####
DMAX = flags.get('DMAX', None)
cutoff = flags.get('cutoff', 1.0e-14)  # flags.get('cutoff', CUTOFF)
### note: cutoff = \varepsilon**2
compress_style = 1
compress_style_mod = 0
upwind = True  # flags.get('do_upwind', False)

te_compress_E = CompressionConfiguration()
te_compress_phi = CompressionConfiguration()

for ci in range(1, compress_style + 1):
    te_compress_E.set_compress_opts(ci, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=cutoff)
    te_compress_phi.set_compress_opts(ci, max_bond=None, cutoff_mode=CUTOFF_MODE, cutoff=CUTOFF)

te_compress_psi = te_compress_phi
te_compress_B = te_compress_E

comp_levels = list(range(1, 6))
if compress_style_mod == 1:  # don't compress inside the derivative
    comp_levels = [1, 2, 3, 4, 0]
elif compress_style_mod == 2:  # don't compress inside the derivative
    comp_levels = [1, 2, 3, 0, 0]

## define coordinates (will label axes and vector components)
X = Coordinate('X', CoordinateType.X)
Y = Coordinate('Y', CoordinateType.Y)
Z = Coordinate('Z', CoordinateType.Z)

## define axis objects
x_vals = np.linspace(-Lx, Lx, npts)
y_vals = np.linspace(-Lx, Ly, npts)

ax_x = Axis(L, q, coordinate=X, xpts=x_vals)
ax_y = Axis(L, q, coordinate=Y, xpts=y_vals)

coord_sys = CartesianCoordinateSpace('cart', ax_x, ax_y, coords=[X, Y, Z])

## define time step
dx, dy = ax_x.dx, ax_y.dx
cfl_params = ((dx, dy), (c, c))
dt = cfl * cfl_limit(cfl_params)
num_tsteps = int(T / dt)
print('dt', dt, cfl * 1. / (c / ax_x.dx + c / ax_y.dx), num_tsteps)
print('dx', dx, 'dy', dy)

## define grids
if TN_layout is LayoutType.COMB:
    grid1 = Grid1D('GX', (ax_x,), layout_type=LayoutType.SEQUENTIAL)
    grid2 = Grid1D('GY', (ax_y,), layout_type=LayoutType.SEQUENTIAL)
    grid_xy = GridsComb('Gr', (grid1, grid2))
else:
    grid_xy = Grid1D('XY', (ax_x, ax_y), layout_type=TN_layout)

## define initial grid1DTNs
data_x, data_y = np.meshgrid(x_vals, y_vals, indexing='ij')
init_Ez = Ez(data_x, data_y)

n_refract = 100
if use_mask == 'box':
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
    # mask_array = (abs(x_mesh) < Lx / 2) * (abs(y_mesh) < Ly / 2)

    w = 20
    mask_x = 0.5 * (np.tanh((x_vals + 0.5) * w) - np.tanh((x_vals - 0.5) * w))
    mask_y = mask_x
    mask_array = np.outer(mask_x, mask_y)

    mask_array = (1 - mask_array) * 1. / n_refract ** 2 + mask_array

    # mask_array[:, 1:-1] = 0.5 * (mask_array[:, :-2] + mask_array[:, 2:])
    # mask_array[1:-1, :] = 0.5 * (mask_array[:-2, :] + mask_array[2:, :])

    eps_inv_mask = grid_xy.map_state_to_mps(mask_array)
elif use_mask == 'circle':
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
    mask_array = float(np.sqrt(x_mesh ** 2 + y_mesh ** 2) < Lx / 2)
    mask_array = (1 - mask_array) * 1. / n_refract ** 2 + mask_array
    eps_inv_mask = grid_xy.map_state_to_mps(mask_array)
else:
    eps_inv_mask = None

if eps_inv_mask is not None:
    plt.figure()
    plt.imshow(eps_inv_mask.get_data())
    plt.colorbar()
    plt.show()



plt.figure()
plt.imshow(init_Ez)
plt.colorbar()
plt.show()

## configure derivatives
if 551 <= te_order <= 559 or 661 <= te_order <= 669:
    deriv_zg = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
    deriv_zv = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
else:
    deriv_zg = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)
    deriv_zv = DerivativeConfiguration(left_bc=BCType.ZEROVALUE, order=order, fd_type=FDType.CENTER)


## E fields perpendicular to conductor can be discontinuous (surface charge) --> open
## B fields parallel to conductor must be continuous --> open
init_Ex_gtn = grid_xy.make_empty_gridTN()
init_Ey_gtn = grid_xy.make_empty_gridTN()
init_Ez_gtn = grid_xy.map_state_to_mps(init_Ez, ax_deriv_configs={ax_x: deriv_zv, ax_y: deriv_zv})
## B fields perpendicular to conductor must be continuous --> symmetric
## B fields parallel to conductor can be discontinuous (surface current)--> open
init_Bx_gtn = grid_xy.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zv, ax_y: deriv_zg})
init_By_gtn = grid_xy.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_zg, ax_y: deriv_zv})
init_phi_gtn = grid_xy.make_empty_gridTN()
init_psi_gtn = grid_xy.make_empty_gridTN()

init_E_field = Field('E', grid_xy, data={X: init_Ex_gtn, Y: init_Ey_gtn, Z: init_Ez_gtn},
                     compress_config=te_compress_E, is_sqrt=True)
init_B_field = Field('B', grid_xy, data={X: init_Bx_gtn, Y: init_By_gtn}, compress_config=te_compress_B,
                     is_sqrt=True)
init_phi_field = ScalarField('phi', grid_xy, data=init_phi_gtn, compress_config=te_compress_phi)
init_psi_field = ScalarField('psi', grid_xy, data=init_psi_gtn, compress_config=te_compress_psi)

### BUILD PDE
em_sys = Maxwell(init_E_field, init_B_field, phi=init_phi_field, psi=init_psi_field,
                 coords_x=coord_sys, matl_params=matl_config,
                 is_electrostatic=False, clean=False, normalize=False,
                 te_order=te_order, compress_levels=comp_levels, is_yee=(te_order==31),
                 upwind=upwind,
                 rel_eps_inv=eps_inv_mask)
em_sys.realspace_smoother(E_comps=[], B_comps=[], phi=True, psi=True)

# vm_sys = VlasovMaxwell(None, None, init_E_field, init_B_field, init_phi_field, init_psi_field,
#                        coords_x=coord_sys, coords_ve=None, coords_vi=None,
#                        matl_params=matl_config,
#                        te_order=te_order, compress_levels=comp_levels,
#                        )


nt = 0
Ez_t = [init_Ez]
max_bond_Es = [[0, 0, init_E_field[Z].max_bond()]]
max_bond_Bs = [[0, 0, 0]]
pre_bond_Es = [[0, 0, init_E_field[Z].max_bond()]]
pre_bond_Bs = [[0, 0, 0]]
num_E_evals = [[np.nan, np.nan, np.nan]]
num_B_evals = [[np.nan, np.nan, np.nan]]
# divEs = [em_sys.check_poisson().norm()]
divBs = [em_sys.check_divB().norm()]
ts = [0.]

while nt < num_tsteps:
    em_sys = em_sys.next_time_step(dt, compress_level=1, is_first_time_step=(nt == 0),
                                   apply_constraints=True, direction=(1 if nt % 2 == 0 else -1))

    # if nt > 95:
    #     em_sys.verbose_plot = True

    try:
        E_bonds = [em_sys.E[C].info['internal_rank'] for C in [X, Y, Z]]
        print('pre comp E bonds', E_bonds)
        pre_bond_Es += [E_bonds]
        if te_order not in [1, 4]:
            num_E_evals += [[em_sys.E[C].info['num_evals'] for C in [X, Y, Z]]]
            print('num E evals', num_E_evals[-1])

        B_bonds = [em_sys.B[C].info['internal_rank'] for C in [X, Y, Z]]
        print('pre comp B bonds', B_bonds)
        pre_bond_Bs += [B_bonds]
        if te_order not in [1, 4]:
            num_B_evals += [[em_sys.B[C].info['num_evals'] for C in [X, Y, Z]]]
            print('num B evals', num_B_evals[-1])
    except KeyError:
        if nt > 1:
            raise KeyError
        pre_bond_Es += [[np.nan, np.nan, np.nan]]
        pre_bond_Bs += [[np.nan, np.nan, np.nan]]
        num_E_evals += [[np.nan, np.nan, np.nan]]
        num_B_evals += [[np.nan, np.nan, np.nan]]

    em_sys.E.compress(inplace=True, compress_level=1, norm_cutoff=np.sqrt(cutoff))
    em_sys.B.compress(inplace=True, compress_level=1, norm_cutoff=np.sqrt(cutoff))

    ts += [ts[-1] + dt]

    Ez_data = em_sys.E.get_comp_data(Z)
    Ez_t += [Ez_data]
    E_bonds = [em_sys.E.max_bond(C) for C in [X, Y, Z]]
    print('E bonds', E_bonds)
    max_bond_Es += [E_bonds]

    B_bonds = [em_sys.B[C].max_bond() for C in [X, Y, Z]]
    print('B bonds', B_bonds)
    max_bond_Bs += [B_bonds]

    # print('em sys', em_sys.check_divB().norm())
    divBs += [em_sys.check_divB().component.frobenius_norm()]
    print('div B', divBs[-1])
    print('dt', dt)

    nt += 1
    # norm = vp_sys.fe.integrate('all')[0] - vp_sys.fi.integrate('all')[0]
    # norm = np.sum(fe_data)*dx*dv - np.sum(init_fi)*dx*dv
    print('result norm', nt, ts[-1], em_sys.E[Z].max_bond(), q ** L)

    # if np.abs(nt * dt - 1.0 /2 ) < dt / 2  or np.abs(nt * dt - 2.0 /2 ) < dt / 2 or np.abs(nt * dt - 3.0 / 2) < dt / 2:
    if np.abs(nt * dt - 0.90) < dt / 2 or np.abs(nt * dt - 1.17) < dt / 2 or np.abs(
                nt * dt - 1.5) < dt / 2:
        plt.figure()
        plt.imshow(np.real(Ez_data))
        plt.xlabel('y')
        plt.ylabel('x')
        plt.colorbar()
        plt.title(f'Ez(x, y, t={nt * dt:1.3f})')
        # plt.show()

        if save_data:
            time = nt * dt
            pickle.dump(Ez_data, open(fdir + f'/L{L}_c{cutoff}_te{te_order}_Ez{np.round(time,1)}.pkl','wb'))

    if nt % 100 == 0: # 25 == 0:
        plt.figure()
        plt.plot(x_vals, Ez_data[:, npts // 2 - 1])
        plt.xlabel('x')
        plt.ylabel('Ez')
        plt.title(f'E(x, y={y_vals[npts // 2]:1.2f}, t={nt * dt:1.3f})')
        # plt.show()

        plt.figure()
        plt.imshow(np.real(Ez_data))
        plt.xlabel('y')
        plt.ylabel('x')
        plt.colorbar()
        plt.title(f'Ez(x, y, t={nt * dt:1.3f})')
        plt.show()

        plt.figure()
        plt.imshow(np.real(em_sys.B.get_comp_data(X)))
        plt.xlabel('y')
        plt.ylabel('x')
        plt.colorbar()
        plt.title(f'Bx(x, y, t={nt * dt:1.3f})')
        plt.show()

        plt.figure()
        plt.imshow(np.real(em_sys.B.get_comp_data(Y)))
        plt.xlabel('y')
        plt.ylabel('x')
        plt.colorbar()
        plt.title(f'By(x, y, t={nt * dt:1.3f})')
        plt.show()

        plt.figure()
        plt.plot(ts,np.array(max_bond_Es)[:, 2], label='rank Ez')
        plt.plot(ts, np.array(max_bond_Bs)[:, 0], label='rank Bx')
        plt.plot(ts, np.array(max_bond_Bs)[:, 1], label='rank By')

        plt.plot(ts, np.array(max_bond_Es)[:, 2], ':', label='rank Ez')
        plt.plot(ts, np.array(max_bond_Bs)[:, 0], ':', label='rank Bx')
        plt.plot(ts, np.array(max_bond_Bs)[:, 1], ':', label='rank By')
        plt.title('ranks')
        plt.show()


        plt.figure()
        plt.plot(ts, divBs)
        plt.xlabel('t')
        plt.ylabel('|div(B)|')

        phi_data = em_sys.phi.get_comp_data()
        if phi_data is not None:
            plt.figure()
            plt.imshow(np.real(phi_data))
            plt.title('phi real')
            plt.colorbar()

            plt.figure()
            plt.imshow(np.imag(phi_data))
            plt.title('phi imag')
            plt.colorbar()

        psi_data = em_sys.psi.get_comp_data()
        if psi_data is not None:

            plt.figure()
            plt.imshow(np.real(psi_data))
            plt.title('psi real')
            plt.colorbar()

            plt.figure()
            plt.imshow(np.imag(psi_data))
            plt.title('psi imag')
            plt.colorbar()

        plt.show()

    divB = em_sys.B.divergence(coord_sys=coord_sys)
    divB_data = divB.get_comp_data()
    if nt == 0:
        divB_data_0 = divB_data
    else:
        print('norm d/dt divB', np.linalg.norm(divB_data))


    # plt.figure()
    # plt.imshow(divB_data)
    # plt.title('div B')
    # plt.colorbar()
    # plt.show()


if save_data:
    pickle.dump([ts, np.array(max_bond_Es)[:, 2], np.array(max_bond_Bs)[:, 0], np.array(max_bond_Bs)[:, 1]],
                open(fdir + f'/L{L}_te{te_order}_c{cutoff}_ranks.pkl','wb'))
    pickle.dump([ts, np.array(pre_bond_Es)[:, 2], np.array(pre_bond_Bs)[:, 0], np.array(pre_bond_Bs)[:, 1]],
                open(fdir + f'/L{L}_te{te_order}_c{cutoff}_pre_ranks.pkl', 'wb'))
    pickle.dump([ts, np.array(num_E_evals)[:, 2], np.array(num_B_evals)[:, 0], np.array(num_B_evals)[:, 1]],
                open(fdir + f'/L{L}_te{te_order}_c{cutoff}_num_evals.pkl', 'wb'))

Ez_data = em_sys.E.get_comp_data(Z)

plt.figure()
plt.plot(x_vals, Ez_data[:, npts // 2 - 1])
plt.xlabel('x')
plt.ylabel('Ez')
plt.title(f'E(x, y={y_vals[npts // 2]:1.2f}, t={nt * dt:1.3f})')

plt.figure()
plt.imshow(Ez_data)
plt.xlabel('y')
plt.ylabel('x')
plt.colorbar()
plt.title(f'Ez(x, y, t={nt * dt:1.3f})')


plt.figure()
plt.plot(ts,np.array(max_bond_Es)[:, 2], label=r'$E_z$')
plt.plot(ts, np.array(max_bond_Bs)[:, 0], label=r'$B_x$')
plt.plot(ts, np.array(max_bond_Bs)[:, 1], label=r'$B_y$')
# plt.title('ranks')
plt.xlabel(r'$t$')
plt.ylabel('max rank')
plt.legend()
# plt.savefig(fdir + '/ranks.png', dpi=300)
plt.show()


plt.show()

