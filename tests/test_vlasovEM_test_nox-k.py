"""Driver: Vlasov-Maxwell test-particle distribution in a time-dependent E field.

Evolves a Vlasov-Maxwell distribution function in the presence of a prescribed
time-dependent electric field, comparing tensor-train (DLR) and full solves.
Reference: http://ammar-hakim.org/sj/je/je32/je32-vlasov-test-ptcl.html
"""
import os, sys, pickle, time, glob
import pdb

import scipy.optimize

sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt

# from defaults import *
from setup_.paths import save_dir, main_dir
from setup_.configs import *
import setup_.helper as helper_test
import helper_quimb as helper
import setup_test as test_setup

from axis import Axis
from basis.basis_spatial import SpatialBasis
from basis.basis_k import FourierBasis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_EM import Maxwell
from pde_vlasovEM import VlasovMaxwell
from vlasov_tests import VlasovTest

"""
test distribution in the presence of time dependent electric field
http://ammar-hakim.org/sj/je/je32/je32-vlasov-test-ptcl.html
"""

save_figs = False
save_data = flags.get('save_data', True)
## to load the output file and maybe restart from it (vs a saved restart file)
load_data = bool(flags.get('load_data', True))
restart_from_T = flags.get('load_data', None)
restart_from_T = restart_from_T if isinstance(restart_from_T, (float, int)) else None
## to restart from a specfic restart file
restart = bool(flags.get('restart', True))
restart_from_nt = flags.get('restart', None)
restart_from_nt = restart_from_nt if isinstance(restart_from_nt, int) else None

is_sqrt = flags.get('is_sqrt', False)
do_tt = flags.get('do_tt', False)  ## see test_vlasovEM_whistler-fd-v2-full
do_full = flags.get('do_full', False)
evolve_ion = flags.get('evolve_ion', True)
branch = flags.get('branch', 0)
dissipation = flags.get('dissipation', 0.0)

is_darwin = False

print('do tt', do_tt)
print('is sqrt', is_sqrt)
print('dissipation', dissipation)

if restart or save_data or save_figs:

    te_order = flags.get('te_order', 64)
    ## 64: PS DLR-G + RK4, 60: PS DLR-G + CN
    ## 84: AP DLR-G + RK4, 80: PS DLR-G + CN
    ## 69: PS DLR-X + RK4, 65: PS DLR-X + CN
    ## 89: AP DLR-X + RK4, 85: PS DLR-X + CN
    ## 68: PS DLR-P + RK4, 67: PS DLR-P + CN
    ## 88: AP DLR-P + RK4, 87: AP DLR-P + CN
    cutoff = flags.get('cutoff', CUTOFF)
    DMAX = flags.get('DMAX', None)
    fnum = '251103/'

    if do_tt:
        if is_sqrt:
            sdir = f'{save_dir}/data{fnum}/VM02-test-k-sq-tt/'
            fdir = f'{main_dir}/data{fnum}/VM02-test-k-sq-tt/'
        else:
            sdir = f'{save_dir}/data{fnum}/VM02-test-k-tt/'
            fdir = f'{main_dir}/data{fnum}/VM02-test-k-tt/'
    else:
        if is_sqrt:
            sdir = f'{save_dir}/data{fnum}/VM02-test-k-sq/'
            fdir = f'{main_dir}/data{fnum}/VM02-test-k-sq/'
        else:
            sdir = f'{save_dir}/data{fnum}/VM02-test-k/'
            fdir = f'{main_dir}/data{fnum}/VM02-test-k/'

    if not os.path.exists(sdir):
        os.makedirs(sdir, exist_ok=True)

    if not os.path.exists(fdir):
        os.makedirs(fdir, exist_ok=True)

    print('fdir', fdir)
    print('sdir', sdir)

extra_str = ''
# if is_sqrt:
#     extra_str = '_sq'

#######################
# standard parameters #
#######################

plasma_config = UnitsConfiguration(eps0=1. / 8.0e-2, mu0=1. / 8.0e-2)
charge = plasma_config.e  # unit of charge
eV = plasma_config.eV  # unit of kB*T

mass_e = 1.0  # mass of electron
mass_i = 1836 * mass_e  # mass of ion
n0_e = 1.0  # electron number density
n0_i = n0_e  # ion number density
T_e = T_i = 1.0
ion_config1 = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
# elc_config1 = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
elc_config1 = IonConfiguration(n0=n0_i, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config1.vth  # electron thermal speed 1/2 m b^2 n0 = sqrt(5.0e-5)
wp_e = elc_config1.wp  # plasma e' frequency
lamD = elc_config1.lamD  # Debye length
wp_i = ion_config1.wp  # plasma i+ frequency
vth_i = ion_config1.vth  # ion thermal speed

#######################
# standard parameters #
#######################

### initial fields
Bz0 = 1.0
Ex0 = 0.9  # 1.0
omega = 0.4567  # 1.0
k = 0.0


def Ex(x, t):
    if k == 0.0:
        if isinstance(x, np.ndarray):
            return Ex0 * np.ones(x.shape) * np.cos(omega * t)
        else:
            return Ex0 * np.cos(omega * t)
    else:
        return Ex0 * np.cos(k * x - omega * t)


## runtime parameter
q = 2
KX = 0
KV = 2
Lx = flags.get('Lx', 4)
Lve = flags.get('Lve', 6)
Lvi = flags.get('Lvi', 6)

if (do_full or do_tt):
    TN_layout = LayoutType.PARALLEL_GROUP if do_full else LayoutType.SEQUENTIAL
    x_map_key = 'F'  # 'F'
    v_map_key = 'F'  # 'M'
else:
    TN_layout = flags.get('layout', LayoutType.SEQUENTIAL)
    x_map_key = flags.get('x_map_key', 'F')
    v_map_key = flags.get('v_map_key', 'B')

order = 1  ## derivative order
if order > 1:
    extra_str += f'_o{order}'
T = 100
print('T', T)
do_adapt_dt = False  # False  # True
do_semiimplicit = False  # True
te_order = flags.get('te_order', 64)
dt_frac = 0.9
dt = flags.get('dt', 0.0125)
is_yee = False

save_every_nt = flags.get('save_every_nt', 100)

DMAX = flags.get('DMAX', None if do_tt else 16)
DMAX_F = flags.get('DMAX_F', DMAX)
comp_style = 3 if te_order == 4 else 1  # 1 if te_order in [313, 314, 333, 344] else 3
cutoff = flags.get('cutoff', CUTOFF)

vmin, vmax = -12 * vth_e, 12 * vth_e

use_E0_as_bg = False
use_B0_as_bg = True
"""
fine for is_sqrt = False
slightly wrong for is_sqrt = True but not for 311. Error doesn't seem to decrease appreciably with dt.
something wrong with maccormack? for sqrt. --don't think so: 414 shows the same behavior
not a dt thing... error doesn't decrease when reducing time step
I think maccormack should have a O(dt^2) discrepancy between sqrt and not-sqrt time stepping
1 time step:
dt = 0.005:  sqrt: px_e 0.004443638279162398 -1.1105775630843183e-05
             reg:  px_e 0.004499978912259819 -1.1249819698436637e-05
             delta ~ 0.00005
             sqrt 413 E -> E0: px_e 0.004498588577402748 -1.4313052127466524e-05
             sqrt 333 E -> E0: px_e 0.004443648997070124 -1.102215100840599e-05
dt = 0.001:  sqrt: px_e 0.000888732765738987 -4.442737180748169e-07
             reg:  px_e 0.0008999999301099957 -4.500444197095359e-07
             delta ~ 0.00001
"""

if dissipation == 'max':
    dissipation = np.sqrt(cutoff)
    print('dissipation per time step', dissipation)

coll_type = CollisionType.H6 if dissipation > 0.0 else None

VP_test = VlasovTest(Lxs=(Lx,), Lves=(Lve, Lve, Lve), Lvis=(Lvi, Lvi, Lvi), KX=KX, KV=KV, q=q,
                     #
                     basis_ves=FourierBasis(), basis_vis=FourierBasis(),
                     ion_config=ion_config1, elc_config=elc_config1,
                     # coll_type=None, do_adapt_coll = True, coll_coeff_e = 0.005, coll_rate_e = 0.01,
                     coll_type=coll_type, do_adapt_coll=False, coll_rate_e=dissipation / dt, coll_rate_i=0,
                     DMAX=DMAX, cutoff=cutoff, DMAX_F=DMAX_F, compress_style=comp_style, compress_style_mod=0,
                     do_adapt_dt=do_adapt_dt, dt_frac=dt_frac, dt=dt, T=T, te_order=te_order,
                     grid_layout=TN_layout, x_map_key=x_map_key, ve_map_key=v_map_key,
                     do_tt=(do_full or do_tt),
                     )

dve = (vmax - vmin) / (2 ** Lve)
VP_test.initialize_axes(x_lims=(0, 2 * np.pi), ve_lims=(-np.pi / dve, np.pi / dve), vi_lims=(-12 * vth_i, 12 * vth_i),
                        x_map_key=x_map_key, ve_map_key=v_map_key, basis_xs=SpatialBasis(),
                        basis_ves=FourierBasis(), basis_vis=FourierBasis(),
                        do_tt=(do_full or do_tt),
                        )

VP_test.initialize_grid()

compress_config = CompressionConfiguration()
compress_config.set_compress_opts(1, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=cutoff)
compress_config.set_compress_opts(2, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=cutoff * 0.01)

te_compress_i = compress_config
te_compress_e = compress_config
te_compress_E = compress_config
te_compress_B = compress_config

# # CUTOFF = 1.0e-12
# te_compress_i = VP_test.get_compression_config(DMAX=DMAX, cutoff=cutoff)
# te_compress_e = VP_test.get_compression_config(DMAX=DMAX, cutoff=cutoff)
# te_compress_E = VP_test.get_compression_config(DMAX=DMAX, cutoff=cutoff)
# te_compress_B = VP_test.get_compression_config(DMAX=DMAX, cutoff=cutoff)

print('te compress fe', te_compress_e)
print('te compress fi', te_compress_i)
print('te compress E', te_compress_E)
print('te compress B', te_compress_B)

comp_levels = VP_test.get_compress_levels()

print('compress', DMAX, VP_test.compress_style, VP_test.compress_style_mod)

compress_opts = te_compress_e.get_compress_opts(1)

#############################
## start building PDE system
#############################

## define axis objects; position -> k-space,
# ax_x, = VP_test.pos_axes
ax_kvx_e, ax_kvy_e, = VP_test.ve_axes
print('ax map', ax_kvx_e.map, ax_kvy_e.map)
print('dvx', ax_kvx_e.dx, 'dvy', ax_kvy_e.dx)
print('cfl dt', ax_kvx_e.dx / (Bz0 * np.max(ax_kvx_e.xpts)), dt)
# exit()
# print('ax_x zero ind', ax_x.zero_ind)

ve_x_min, ve_x_max = vmin, vmax
ve_y_min, ve_y_max = vmin, vmax
vx_e_vals = np.linspace(vmin, vmax, 2 ** Lve, endpoint=False)
vy_e_vals = np.linspace(vmin, vmax, 2 ** Lve, endpoint=False)

ve_kx_min, ve_kx_max = VP_test.ve_lims[0]
ve_ky_min, ve_ky_max = VP_test.ve_lims[1]
tot_x = 1  # x_max - x_min

coords_x = VP_test.pos_coordsys
coords_ve = VP_test.ve_coordsys
coords_vi = VP_test.vi_coordsys

X, Y, Z = coords_x.coords
VX, VY, VZ = coords_ve.coords

npts_x = 1  # ax_x.npts
npts_vxe = ax_kvx_e.npts
npts_vye = ax_kvy_e.npts

# ## define MPS grids
grid_i = VP_test.grid_i
grid_e = VP_test.grid_e
grid_X = VP_test.grid_X

x_vals = 0  # ax_x.xpts
xB_vals = x_vals

kvx_e_vals = ax_kvx_e.xpts
kvy_e_vals = ax_kvy_e.xpts

####3 define initial gridTNs

## ion density:
init_fi_gtn = grid_i.make_empty_gridTN()

# elec density
fe1_kvx = helper_test.maxwellian_k(kvx_e_vals, vth2=vth_e ** 2, density=1.0, flow=0., is_sqrt=is_sqrt) * n0_e
fe1_kvy = helper_test.maxwellian_k(kvy_e_vals, vth2=vth_e ** 2, density=1.0, flow=0., is_sqrt=is_sqrt)

plt.figure()
plt.plot(fe1_kvx, label='tmp')
plt.title('diff')
plt.legend()
plt.show()
plt.close()

init_fe_gtn = grid_e.make_gtn_from_dicts([{ax_kvx_e: fe1_kvx, ax_kvy_e: fe1_kvy}],
                                         data_type=DataType.MPS)

plt.figure()
plt.plot(np.real(init_fe_gtn.get_data().reshape(-1)))
plt.plot(np.imag(init_fe_gtn.get_data().reshape(-1)))
plt.show()
plt.close()

# helper.pad_mpx_virtuals(init_fe_gtn.data, max_bond=DMAX)
# init_fe_gtn.compress(compress_opts={'max_bond': DMAX})
# print('init_fe_gtn', init_fe_gtn.max_bond())


## E field: 0
Ex0_gtn = grid_X.make_empty_gridTN()
Ex0_gtn.data = Ex(x_vals, 0)  # grid_X.map_state_to_mps(Ex(x_vals, 0))
# Ex0_gtn.is_constant = (k == 0.)
Ex_gtn = grid_X.make_empty_gridTN()
Ey_gtn = grid_X.make_empty_gridTN()
Ez_gtn = grid_X.make_empty_gridTN()

## B field:
Bz0_gtn = grid_X.make_empty_gridTN()
Bz0_gtn.data = Bz0  # grid_X.get_ones_mps().scalar_multiply(Bz0)
# Bz0_gtn.is_constant = True
Bx_gtn = grid_X.make_empty_gridTN()
By_gtn = grid_X.make_empty_gridTN()
Bz_gtn = grid_X.make_empty_gridTN()

deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
# deriv_v = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)
deriv_v = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)

# init_fe_gtn.ax_deriv_configs.update({ax_x: deriv_x, ax_vx_e: deriv_v, ax_vy_e: deriv_v,})
init_fe_gtn.ax_deriv_configs.update({ax_kvx_e: deriv_v, ax_kvy_e: deriv_v, })
deriv_x_EM = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
Ex_gtn.ax_deriv_configs.update({ax_kvx_e: deriv_x_EM, ax_kvy_e: deriv_x_EM})
Ey_gtn.ax_deriv_configs.update({ax_kvx_e: deriv_x_EM, ax_kvy_e: deriv_x_EM})
Ez_gtn.ax_deriv_configs.update({ax_kvx_e: deriv_x_EM, ax_kvy_e: deriv_x_EM})
By_gtn.ax_deriv_configs.update({ax_kvx_e: deriv_x_EM, ax_kvy_e: deriv_x_EM})
Bz_gtn.ax_deriv_configs.update({ax_kvx_e: deriv_x_EM, ax_kvy_e: deriv_x_EM})
Ex0_gtn.ax_deriv_configs.update({ax_kvx_e: deriv_x_EM, ax_kvy_e: deriv_x_EM})
Bz0_gtn.ax_deriv_configs.update({ax_kvx_e: deriv_x_EM, ax_kvy_e: deriv_x_EM})

### time step info
if dt is None:
    if te_order == 413 or te_order == 414:
        Fmax_e = np.array([Ex0, 0.0, 0.0]) * charge / mass_e
        # Fmax_e = np.array([0.0, 0.0, 0.0]) * charge / mass_e
    else:
        Fmax_e = np.array([Ex0 + vmax * Bz0, vmax * Bz0, 0.0]) * charge / mass_e
    dt = VP_test.get_dt(Fmax_e)
    # print('dti', dt/dt_frac, Fmax_e, ve_y_max)
num_tsteps = int(T / dt)
print('dt', dt, num_tsteps, plasma_config.c)

print('charge', elc_config1.charge)

### get file naming string
fstr = f'w{omega}_' + VP_test.get_fstr(extra_str=extra_str)
print('filename', fstr)

########################
#### run simulation ####
########################

### define v2 mpo for measurement ###
v2e_mpo = test_setup.get_v2_mpo(grid_e, VP_test.ve_axes)

#### define fields
init_fe_field = ScalarField('fe', grid_e, data=init_fe_gtn, is_sqrt=is_sqrt, compress_config=te_compress_e)
init_fi_field = ScalarField('fi', grid_i, data=init_fi_gtn, is_sqrt=is_sqrt, compress_config=te_compress_i)
if use_E0_as_bg:
    init_E_field = Field('E', grid_X, data={X: Ex_gtn, Y: Ey_gtn, Z: Ez_gtn},
                         is_sqrt=True, compress_config=te_compress_E)
    E0_field = Field('E0', grid_X, data={X: Ex0_gtn}, is_sqrt=True)
else:
    init_E_field = Field('E', grid_X, data={X: Ex0_gtn, Y: Ey_gtn, Z: Ez_gtn},
                         is_sqrt=True, compress_config=te_compress_E)
    E0_field = None

if use_B0_as_bg:
    init_B_field = Field('B', grid_X, data={X: Bx_gtn, Y: By_gtn, Z: Bz_gtn},
                         is_sqrt=True, compress_config=te_compress_B)
    B0_field = Field('B0', grid_X, data={Z: Bz0_gtn}, is_sqrt=True)
    curlB0_field = Field('B0', grid_X, data={})
else:
    init_B_field = Field('B', grid_X, data={X: Bx_gtn, Y: By_gtn, Z: Bz0_gtn},
                         is_sqrt=True, compress_config=te_compress_B)
    B0_field = None
    curlB0_field = None

tot_V = 1  # 2 ** (Lve - 1) if is_sqrt else 2 ** (2 * (Lve - 1))
norm_fe = init_fe_gtn.norm(is_sqrt=is_sqrt) / tot_V
print('dx', ax_kvx_e.dx, ax_kvy_e.dx, vmax - vmin)
print('field norm', norm_fe)

try:
    if not restart:
        raise IOError

    #### load data if restarting ####
    out = test_setup.load_data_EM(VP_test,
                                  init_fe_field, init_fi_field, init_E_field, init_B_field, None, None,
                                  fdir=fdir, sdir=sdir, extra_str=extra_str, save_every_nt=save_every_nt,
                                  load_data=load_data, restart=restart, restart_from_T=restart_from_T, only_elc=True,
                                  other_data=['preD'],
                                  )

    (init_fe_field, ts, nrg_fe_ts, max_bond_fe, max_bond_pre,) = out

    nt = len(ts) - 1
    restarted = True

    with open(sdir + 'errs_' + fstr + '.pkl', 'rb') as f:
        ts, errs_avg, errs_dist, errs_shape, errs_norm, errs_mass, \
            errs_drift, errs_sigma2, errs_fitted, errs_deriv = pickle.load(f)
        print('loaded', sdir + 'errs_' + fstr + '.pkl')

    with open(sdir + 'meas_' + fstr + '.pkl', 'rb') as f:
        _, avg_vx_ts, avg_vy_ts = pickle.load(f)
        print('loaded', sdir + 'errs_' + fstr + '.pkl')

    init_E_field[X].data = Ex(x_vals, ts[-1])

except(IOError, OSError, NameError, ValueError, FileNotFoundError):

    ##### initialize tests ########
    ts = [0]
    nt = 0

    nrg_fe = init_fe_gtn.meas_expec(v2e_mpo, is_sqrt=is_sqrt) * mass_e / 2 / tot_x
    nrg_E = np.array(init_E_field.norms(compIDs=coords_x.coords)) ** 2 / 2 / tot_x
    nrg_B = np.array(init_B_field.norms(compIDs=coords_x.coords)) ** 2 / 2 / tot_x
    print('tot_x', tot_x)
    print('nrg E', nrg_E)
    print('nrg B0', nrg_B)

    print('init bond dims', init_fe_gtn.max_bond(), init_fi_gtn.max_bond())
    max_bond_fe = [init_fe_gtn.max_bond()]
    max_bond_pre = [init_fe_gtn.max_bond()]

    print('current nrgs', nrg_fe, nrg_E, )

    ## measure moments
    vx_moment_mpo = grid_e.make_mpo_ndim({ax_kvx_e: ax_kvy_e.get_xmultiply_mpo(x_power=1)})
    vy_moment_mpo = grid_e.make_mpo_ndim({ax_kvy_e: ax_kvy_e.get_xmultiply_mpo(x_power=1)})
    px_e = init_fe_gtn.meas_expec(vx_moment_mpo, integ_axes=grid_e.axes, new_grid=grid_X) / tot_x
    py_e = init_fe_gtn.meas_expec(vx_moment_mpo, integ_axes=grid_e.axes, new_grid=grid_X) / tot_x

    nrg_fe_ts = [nrg_fe]
    avg_vx_ts = [px_e]
    avg_vy_ts = [py_e]
    errs_avg = [np.linalg.norm([px_e - 0, py_e - 0])]
    errs_dist = [np.nan]  ## distribution - gaussian centered at theoretical px, py
    errs_shape = [np.nan]  ## distribution - gaussian centered at measured px, py
    ## errors with respect to fitted distribution
    errs_norm = [np.nan]  ## distribution - gaussian centered at measured px, py
    errs_drift = [np.nan]  ## distribution - gaussian centered at measured px, py
    errs_sigma2 = [np.nan]  ## distribution - gaussian centered at measured px, py
    errs_fitted = [np.nan]  ## distribution - fitted gaussian
    errs_deriv = [np.nan]  ## (d/dx + d/dy) distribution - fitted gaussian; attempt to measure (lack of) smoothness
    errs_mass = [np.nan]

### BUILD PDE
em_sys = Maxwell(init_E_field, init_B_field,  # phi=init_phi_field, psi=init_psi_field,
                 coords_x=coords_x, matl_params=plasma_config,
                 background_B0=B0_field, background_E0=E0_field,
                 normalize=False, clean=False, is_yee=False)
## is_yee = True gives the wrong results because of current offset.
em_sys.curlB0 = curlB0_field

vm_sys = VlasovMaxwell(init_fe_field, init_fi_field, em_sys,
                       coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                       elc_params=elc_config1, ion_params=ion_config1, evolve_ion=False, evolve_EM=False,
                       normalize=False,  # True,
                       zipup=True, te_order=te_order, compress_levels=comp_levels,
                       upwind=False,
                       conservative=False,
                       # conservative = (not is_sqrt)
                       )
print('initialized vm_sys')

vm_sys.time = ts[-1]
vm_sys.semiimplicit_force = do_semiimplicit

vm_sys.collision = VP_test.collision_config
print('colls', vm_sys.collision.coll_params)

vm_sys.compress_F = VP_test.compress_F
vm_sys.compress_F_opts = VP_test.compress_F_opts
print('compress E?', vm_sys.compress_F, vm_sys.compress_F_opts)

print('E nrg', vm_sys.E.norm() ** 2 * 1. / 2 / tot_x)

px_e = vm_sys.sys_fe.compute_moment([ax_kvx_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x
py_e = vm_sys.sys_fe.compute_moment([ax_kvy_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x
print('init avg v', px_e, py_e)

if do_adapt_dt:
    if te_order == 413 or te_order == 414:
        em_term = Field('EM', grid_e, data={})
        # em_term = vm_sys.compute_force_term(is_ion=False, background_force=False)
    else:
        em_term = vm_sys.compute_force_term(is_ion=False)

    max_em = []
    for coord in [X, Y, Z]:
        try:
            em_data = em_term.get_comp_data(coord, ax_select={ax_kvx_e: 0, ax_kvy_e: 0, })
            em_max = np.max(np.abs(em_data)) if em_data is not None else 0.0
        except KeyError:
            em_max = 0.
        max_em += [em_max]

    print('max em', max_em)
    Fe_maxs = np.array(max_em)
    dt = VP_test.get_dt(Fe_maxs)
    print('adapt dt', dt, plasma_config.c, elc_config1.vth)

fig1, ax1 = plt.subplots(1, 1)
time_ = nt * dt

fe_data = init_fe_gtn.get_data()
fe_data = ax_kvy_e.basis.get_realspace_1D(fe_data, 1)
fe_data = ax_kvx_e.basis.get_realspace_1D(fe_data, 0)
im1 = ax1.imshow(np.real(fe_data.T), extent=(ve_x_min, ve_x_max, ve_y_min, ve_y_max),
                 aspect='auto', origin='lower')

fig1.colorbar(im1, ax=ax1)
ax1.set_title(f'time {time_:3.2f}')
ax1.set_xlabel('vx'), ax1.set_ylabel('vy')

plt.show()

em_term = vm_sys.compute_force_term(is_ion=False)
force_data = em_term.get_comp_data(X)
force_data = ax_kvy_e.basis.get_realspace_1D(force_data, 1)
force_data = ax_kvx_e.basis.get_realspace_1D(force_data, 0)
plt.figure()
plt.imshow(np.real(force_data))
plt.colorbar()
plt.show()

time1 = time.time()

print('starting time evolution')
while ts[-1] < T:

    print('vm_sys.fe.norm', vm_sys.fe.norm(), norm_fe, vm_sys.fe.max_bond())
    # exit()

    dt_ = (dt * 0.1) if nt == 0 else dt

    ## manually update E
    new_Ex = grid_X.make_empty_gridTN()
    if 60 <= te_order < 70 or te_order in [80, 85, 87]:
        new_Ex.data = Ex(x_vals, ts[-1] + dt_ / 2)
        vm_sys.time = ts[-1] + dt_ / 2
        vm_sys.sys_fe.time = ts[-1] + dt_ / 2
        print('dt+2 to time', vm_sys.time, vm_sys.sys_fe.time, ts[-1], dt_ / 2)
    else:
        new_Ex.data = Ex(x_vals, ts[-1])  # + dt_/2)
    new_Ex.is_constant = (k == 0.)
    if is_darwin:
        if use_E0_as_bg:
            vm_sys.sys_ED.background_E0[X] = new_Ex
        else:
            vm_sys.ET[X] = new_Ex
    else:
        if use_E0_as_bg:
            vm_sys.EM_sys.background_E0[X] = new_Ex
            vm_sys.sys_fe.saved_SL_mpos = {ax: {} for ax in [ax_kvx_e, ax_kvy_e]}
            vm_sys.sys_fi.saved_SL_mpos = {ax: {} for ax in [ax_kvx_e, ax_kvy_e]}
        else:
            vm_sys.E[X] = new_Ex

    walltime = time.time()
    # print('verbose plot', vm_sys.sys_fe.verbose_plot)
    # if nt > 10:
    #     vm_sys.sys_fe.verbose_plot = True
    vm_sys = vm_sys.next_time_step(dt_, compress_level=1, err_tol=1.0e-7, verbose_plot=False,
                                   is_first_time_step=(nt == 0),
                                   direction=1)  # direction=(1 if nt % 2 == 0 else -1)) #, tmp_var=norm0 / dV))
    # helper.pad_mpx_virtuals(vm_sys.fe.component.data, max_bond=DMAX)
    print('wall time', time.time() - walltime)

    max_bond_pre += [vm_sys.fe.component.info.get('internal_rank', np.nan)]
    print('saved max bond', max_bond_pre[-1])

    # vm_sys.fe.compress(inplace=True)
    max_bond_fe += [vm_sys.fe.max_bond()]

    norm = vm_sys.fe.norm()  # - vm_sys.fi.norm()
    print('vm_sys.fe.norm', norm, np.abs(vm_sys.fe.norm() - norm_fe),
          np.abs(np.real(vm_sys.fe.norm()) - norm_fe), np.abs(np.abs(vm_sys.fe.norm()) - norm_fe))
    if is_sqrt:
        norm = np.abs(norm) ** 2

    fe_field: 'ScalarField' = vm_sys.fe
    nrg_fe = fe_field.component.meas_expec(v2e_mpo, is_sqrt=is_sqrt) * mass_e / 2 / tot_x
    energy_E_t = np.array(vm_sys.E.norms(compIDs=coords_x.coords)) ** 2 * 1. / 2 / tot_x  # electric field energy

    px_e = vm_sys.sys_fe.compute_moment([ax_kvx_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x
    py_e = vm_sys.sys_fe.compute_moment([ax_kvy_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x
    px_e = np.real(px_e)
    py_e = np.real(py_e)

    # px_e = vm_sys.sys_fe.compute_moment([ax_vx_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x
    # py_e = vm_sys.sys_fe.compute_moment([ax_vy_e], powers=[1], integ_axes=grid_e.axes, compress=0).component / tot_x

    print('px_e', px_e, py_e)

    nrg_fe_ts += [nrg_fe]

    ts += [ts[-1] + dt_]
    nt += 1

    # ## manually update E
    # # new_Ex = grid_X.map_state_to_mps(Ex(x_vals, ts[-1]))
    # new_Ex = grid_X.make_empty_gridTN()
    # new_Ex.data = Ex(x_vals,ts[-1])
    # new_Ex.is_constant = (k == 0.)
    # if is_darwin:
    #     if use_E0_as_bg:
    #         vm_sys.sys_ED.background_E0[X] = new_Ex
    #     else:
    #         vm_sys.ET[X] = new_Ex
    # else:
    #     if use_E0_as_bg:
    #         vm_sys.EM_sys.background_E0[X] = new_Ex
    #         vm_sys.sys_fe.saved_SL_mpos = {ax: {} for ax in [ax_kvx_e, ax_kvy_e]}
    #         vm_sys.sys_fi.saved_SL_mpos = {ax: {} for ax in [ax_kvx_e, ax_kvy_e]}
    #     else:
    #         vm_sys.E[X] = new_Ex

    # if do_adapt_dt:
    #     if te_order == 413 or te_order == 414:
    #         em_term = Field('EM', grid_e, data={})
    #         # em_term = vm_sys.compute_force_term(is_ion=False, background_force=False)
    #     else:
    #         em_term = vm_sys.compute_force_term(is_ion=False)
    #     max_em = []
    #     for coord in [X, Y, Z]:
    #         try:
    #             em_data = em_term.get_comp_data(coord, ax_select={ax_kvx_e: 0, ax_kvy_e: 0, })
    #             em_max = np.max(np.abs(em_data)) if em_data is not None else 0.0
    #         except KeyError:
    #             em_max = 0.
    #         max_em += [em_max]

    #     Fe_maxs = np.array(max_em)
    #     dt = VP_test.get_dt(Fe_maxs)
    #     # print('restarted dt', dt)

    print('result norm', nt, ts[-1], dt, norm,  # nrg_fe, nrg_fi,
          energy_E_t, fe_field.max_bond(), Lx, Lvi, Lve)

    ### compute errors
    if omega != 1.:
        z = elc_config1.charge
        vx_exact = Ex0 / (z ** 2 - omega ** 2) * (np.sin(z * ts[-1]) - z * omega * np.sin(omega * ts[-1]))
        vy_exact = Ex0 / (z ** 2 - omega ** 2) * (np.cos(z * ts[-1]) - np.cos(omega * ts[-1]))
    else:
        z = elc_config1.charge
        vx_exact = z * Ex0 / 2 * (ts[-1] * np.cos(ts[-1]) + np.sin(ts[-1]))
        vy_exact = - z ** 2 * Ex0 / 2 * ts[-1] * np.sin(ts[-1])

    errs_avg += [np.linalg.norm([px_e - vx_exact, py_e - vy_exact])]

    ####
    fe1_kvx = helper_test.maxwellian_k(kvx_e_vals, vth2=vth_e ** 2, density=1.0, flow=vx_exact, is_sqrt=is_sqrt) * n0_e
    fe1_kvy = helper_test.maxwellian_k(kvy_e_vals, vth2=vth_e ** 2, density=1.0, flow=vy_exact, is_sqrt=is_sqrt)

    theory_fe_gtn = grid_e.make_gtn_from_dicts([{ax_kvx_e: fe1_kvx, ax_kvy_e: fe1_kvy}],
                                               data_type=DataType.MPS)

    # err = theory_fe_gtn.distance(fe_field.component) / \
    #       np.sqrt(ax_kvx_e.npts * ax_kvy_e.npts)
    # print('old err', err)
    err = theory_fe_gtn.distance(fe_field.component) / theory_fe_gtn.frobenius_norm()
    print('dist err', err)
    errs_dist += [err]


    #### additional errors
    def maxwellian_func(xy_grid_point, x0, y0, n0, sig2x, sig2y):
        vx, vy = xy_grid_point
        gx = helper_test.maxwellian(vx, vth2=sig2x, density=1.0, flow=x0)
        gy = helper_test.maxwellian(vy, vth2=sig2y, density=1.0, flow=y0)
        out = gx * gy * n0
        return out


    def jac(xy_grid_point, x0, y0, n0, sig2x, sig2y):
        vx, vy = xy_grid_point

        gx = helper_test.maxwellian(vx, vth2=sig2x, density=1.0, flow=x0)
        gy = helper_test.maxwellian(vy, vth2=sig2y, density=1.0, flow=y0)

        ## df/d(x0) = (x-x0)/b * n0 * maxwellian
        d_x0 = (vx - x0) / sig2x * gx * gy * n0
        d_y0 = (vy - y0) / sig2y * gx * gy * n0

        ## df/d(n0) = maxwellian
        d_n0 = gx * gy

        ## df/d(sig2x) = (x-x0)^2 / 2 / b^2 * maxwellian
        d_sig2x = (vx - x0) ** 2 / 2 / sig2x ** 2 * gx * gy * n0
        d_sig2y = (vy - y0) ** 2 / 2 / sig2y ** 2 * gx * gy * n0

        return np.array([d_x0, d_y0, d_n0, d_sig2x, d_sig2y]).T


    dist_data = fe_field.component.get_data()
    dist_data = ax_kvx_e.basis.get_realspace_1D(dist_data, 0)
    dist_data = ax_kvy_e.basis.get_realspace_1D(dist_data, 1)
    if is_sqrt:
        dist_data = np.conj(dist_data) * dist_data

    # real_xpts = np.linspace(-np.pi / ax_kvx_e.dx, np.pi / ax_kvx_e.dx, ax_kvx_e.npts, endpoint=False)
    # real_ypts = np.linspace(-np.pi / ax_kvy_e.dx, np.pi / ax_kvy_e.dx, ax_kvy_e.npts, endpoint=False)
    # mesh_vx, mesh_vy = np.meshgrid(real_xpts, real_ypts, indexing='ij')
    mesh_vx, mesh_vy = np.meshgrid(vx_e_vals, vy_e_vals, indexing='ij')
    grid_pts = (mesh_vx.reshape(-1), mesh_vy.reshape(-1))
    out = maxwellian_func((mesh_vx, mesh_vy), *(px_e, py_e, 1., vth_e ** 2, vth_e ** 2))

    try:
        p_opt, p_cov = scipy.optimize.curve_fit(maxwellian_func, grid_pts, dist_data.reshape(-1),
                                                (px_e, py_e, 1., vth_e ** 2, vth_e ** 2),
                                                jac=jac
                                                )
    except RuntimeError:
        p_opt, p_cov = (np.nan, np.nan), np.nan

    fit_vxs = maxwellian_func((mesh_vx, mesh_vy),
                              px_e, py_e, 1.0, vth_e ** 2, vth_e ** 2)
    fit_opt = maxwellian_func((mesh_vx, mesh_vy), *p_opt)

    print('opt params', p_opt)
    print('meas params', px_e / p_opt[0], py_e / p_opt[1], tot_x, )
    print('theory params', vx_exact, vy_exact, 1.0, vth_e ** 2, vth_e ** 2)

    errs_shape += [np.linalg.norm(fit_vxs - dist_data) / np.sqrt(npts_vxe * npts_vye)]
    errs_fitted += [np.linalg.norm(fit_opt - dist_data) / np.sqrt(npts_vxe * npts_vye)]
    errs_norm += [np.linalg.norm([p_opt[2] - 1.0])]
    errs_drift += [np.linalg.norm([p_opt[0] - vx_exact, p_opt[1] - vy_exact])]
    errs_sigma2 += [np.linalg.norm([p_opt[3] - vth_e ** 2, p_opt[4] - vth_e ** 2])]

    dist_err = fit_opt - dist_data
    derr_dvx = np.abs(np.diff(dist_err, axis=0, append=0) / ax_kvx_e.dx)
    derr_dvy = np.abs(np.diff(dist_err, axis=1, append=0) / ax_kvy_e.dx)
    errs_deriv += [np.linalg.norm(derr_dvx + derr_dvy) / np.sqrt(npts_vxe * npts_vye)]

    avg_vx_ts += [p_opt[0]]  # [px_e]
    avg_vy_ts += [p_opt[1]]  # [py_e]

    errs_mass += [np.abs(norm - norm_fe)]

    if nt % save_every_nt == 0 or ts[-1] > T:

        time_ = ts[-1]

        fe_data = fe_field.get_comp_data()
        if is_sqrt:
            fe_data = np.abs(fe_data) ** 2
        plt.figure()
        plt.imshow(np.real(fe_data).T, extent=(ve_x_min, ve_x_max - dve, ve_y_min, ve_y_max - dve),
                   aspect='auto', origin='lower')
        plt.colorbar()

        plt.figure()
        plt.imshow(np.imag(fe_data).T, extent=(ve_x_min, ve_x_max - dve, ve_y_min, ve_y_max - dve),
                   aspect='auto', origin='lower')
        plt.colorbar()
        plt.show()

        fig1, ax1 = plt.subplots(1, 1)

        fe_data = fe_field.get_comp_data()
        fe_data = ax_kvy_e.basis.get_realspace_1D(fe_data, 1)
        fe_data = ax_kvx_e.basis.get_realspace_1D(fe_data, 0)
        if is_sqrt:
            fe_data = np.abs(fe_data) ** 2
        im1 = ax1.imshow(np.real(fe_data).T, extent=(ve_x_min, ve_x_max - dve, ve_y_min, ve_y_max - dve),
                         aspect='auto', origin='lower')

        ax1.plot(p_opt[0], p_opt[1], 'bx', markersize=6)
        ax1.plot(px_e, py_e, 'ro', markersize=6)
        ax1.plot(vx_exact, vy_exact, 'gx', markersize=6)

        fig1.colorbar(im1, ax=ax1)
        ax1.set_title(f'time {time_:3.2f}')
        ax1.xaxis.set_visible(False)
        ax1.set_xlabel('vx'), ax1.set_ylabel('vy')
        # plt.show()

        ## theory maxwellian
        theory_fe_data = theory_fe_gtn.get_data()
        theory_fe_data = ax_kvy_e.basis.get_realspace_1D(theory_fe_data, 1)
        theory_fe_data = ax_kvx_e.basis.get_realspace_1D(theory_fe_data, 0)

        # ## vm sys 1
        # plt.figure()
        # plt.imshow(np.real(fe_data) - np.real(theory_fe_data), extent=(ve_x_min, ve_x_max - dve, ve_y_min, ve_y_max - dve),
        #            aspect='auto', origin='lower')
        # plt.colorbar()
        # plt.title('real(error)')
        # plt.tight_layout()
        # plt.savefig(fdir + f'theory_err_L{Lve}_te{te_order}_dt{dt}_{nt}_re.png')

        # plt.figure()
        # plt.imshow(np.imag(fe_data) - np.imag(theory_fe_data),
        #            extent=(ve_x_min, ve_x_max - dve, ve_y_min, ve_y_max - dve),
        #            aspect='auto', origin='lower')
        # plt.colorbar()
        # plt.title('im(vm sys 1) err')
        # plt.tight_layout()
        # plt.savefig(fdir + f'theory_err_L{Lve}_te{te_order}_dt{dt}_{nt}_im.png')
        # plt.show()

        # fig7, ax7 = plt.subplots()
        # ax7.semilogy(ts, np.abs(nrg_fe_ts), label='elec')
        # plt.xlabel('t'), plt.ylabel('fe nrg')
        # plt.title('fe nrg')
        # plt.show()

        if omega != 1.:
            # vx_exact = Ex0 / (1 - omega ** 2) * (np.sin(ts) - omega * np.sin(omega * np.array(ts)))
            # vy_exact = Ex0 / (1 - omega ** 2) * (np.cos(ts) - np.cos(omega * np.array(ts)))
            z = elc_config1.charge
            print('charge', z)
            vx_exact = Ex0 / (z ** 2 - omega ** 2) * (
                        np.sin(z * np.array(ts)) - z * omega * np.sin(omega * np.array(ts)))
            vy_exact = Ex0 / (z ** 2 - omega ** 2) * (np.cos(z * np.array(ts)) - z ** 2 * np.cos(omega * np.array(ts)))
        else:
            vx_exact = z * Ex0 / 2 * (np.array(ts) * np.cos(ts) + np.sin(ts))
            vy_exact = - z ** 2 * Ex0 / 2 * np.array(ts) * np.sin(ts)

        plt.figure()
        plt.plot(ts, avg_vx_ts, label='vx')
        plt.plot(ts, avg_vy_ts, label='vy')
        plt.plot(ts, vx_exact, 'k--')
        plt.plot(ts, vy_exact, 'k--')
        plt.xlabel('t'), plt.ylabel('v')
        plt.title('avg velocity')

        plt.figure()
        plt.plot(ts, avg_vx_ts - vx_exact, label='vx')
        plt.plot(ts, avg_vy_ts - vy_exact, label='vy')
        plt.plot(ts, errs_avg, label='errs meas')
        plt.xlabel('t'), plt.ylabel('v')
        plt.title('avg velocity error')

        plt.figure()
        plt.semilogy(ts, errs_dist, label='dist')
        plt.semilogy(ts, errs_avg, label='avg')
        #
        plt.semilogy(ts, errs_shape, label='shape')
        plt.semilogy(ts, errs_fitted, label='fitted')
        plt.semilogy(ts, errs_norm, label='norm')
        plt.semilogy(ts, errs_drift, label='drift')
        plt.semilogy(ts, errs_sigma2, label='var')
        plt.semilogy(ts, errs_deriv, label='err deriv')

        plt.semilogy(ts, errs_mass, label='mass err')

        plt.title('error')
        plt.xlabel('t'), plt.ylabel('error')
        plt.grid(color='gray', axis='y')
        plt.legend()

        # plt.show()

        ######## plot EE vs bond #####
        fig7, ax7 = plt.subplots()
        ax7.semilogy(vm_sys.fe.component.entanglement_entropy_all(), '-x', label='elec')
        # ax7.semilogy( vm_sys.fi.component.entanglement_entropy_all(), '-o', label='ion')

        ax7.legend()
        ax7.set_xlabel('bond number')
        ax7.set_ylabel('EE')
        plt.show()

        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()

        if save_data:
            pickle.dump([ts, errs_avg, errs_dist, errs_shape, errs_norm, errs_mass,
                         errs_drift, errs_sigma2, errs_fitted, errs_deriv],
                        open(sdir + 'errs_' + fstr + '.pkl', 'wb'))
            print('saved', sdir + 'errs_' + fstr + '.pkl')

            pickle.dump([ts, avg_vx_ts, avg_vy_ts],
                        open(sdir + 'meas_' + fstr + '.pkl', 'wb'))
            print('saved', sdir + 'meas_' + fstr + '.pkl')

            if not os.path.exists(sdir + 'restart/'):
                os.makedirs(sdir + 'restart/')

            test_setup.save_data_EM(VP_test, vm_sys, sdir + 'restart/', extra_str=extra_str,
                                    add_fstr=f'-nt{nt}', only_elc=True,
                                    ts=ts, nrg_fe_ts=nrg_fe_ts, max_bond_fe=max_bond_fe,
                                    other_data={'preD': max_bond_pre},
                                    )

        #### compute moments; doesnt work. not sure why
        # density_e = vm_sys.sys_fe.compute_moment([])
        #
        # px_e = vm_sys.sys_fe.compute_moment([ax_vx_e], powers=[1]).get_comp_data()
        # py_e = vm_sys.sys_fe.compute_moment([ax_vy_e], powers=[1]).get_comp_data()
        # p_e = np.sqrt( px_e ** 2 + py_e ** 2 ) * mass_e
        # print('pxe', px_e, 'pye', py_e, 'pe', p_e)
        #
        # nrgx_e = vm_sys.sys_fe.compute_moment([ax_vx_e], powers=[2]).get_comp_data()
        # nrgy_e = vm_sys.sys_fe.compute_moment([ax_vy_e], powers=[2]).get_comp_data()
        # nrg_e = (nrgx_e + nrgy_e) * 0.5 * mass_e
        # print('nrgx', nrgx_e, 'nrgy_e', nrgy_e, 'nrg_e', nrg_e)

        # vx_data = vm_sys.sys_fe.f.get_comp_data(ax_select={ax_vy_e: ax_vy_e.npts // 2}) / np.pi / 2
        # plt.figure()
        # plt.plot(vx_data)
        # plt.title('vx, vy=0; integ x')
        #
        # plt.show()

print('wall time', time.time() - time1)

## save error
# if save_data:
#     pickle.dump(np.array([ts, errs_avg, errs_dist]), open(fdir + 'errs_' + fstr + '.pkl', 'wb'))
#     print('saved', fdir + 'errs_' + fstr + '.pkl')

if save_data:
    pickle.dump([ts, errs_avg, errs_dist, errs_shape, errs_norm, errs_mass,
                 errs_drift, errs_sigma2, errs_fitted, errs_deriv],
                open(fdir + 'errs_' + fstr + '.pkl', 'wb'))
    print('saved', fdir + 'errs_' + fstr + '.pkl')

    pickle.dump([ts, avg_vx_ts, avg_vy_ts],
                open(fdir + 'meas_' + fstr + '.pkl', 'wb'))
    print('saved', fdir + 'meas_' + fstr + '.pkl')

    if not os.path.exists(sdir + 'restart/'):
        os.makedirs(sdir + 'restart/')

    test_setup.save_data_EM(VP_test, vm_sys, fdir, extra_str=extra_str,
                            add_fstr=f'-nt{nt}', only_elc=True,
                            ts=ts, nrg_fe_ts=nrg_fe_ts, max_bond_fe=max_bond_fe,
                            other_data={'preD': max_bond_pre},
                            )

##############

fig7, ax7 = plt.subplots()
ax7.semilogy(ts, np.abs(nrg_fe_ts), label='elec')

ax7.legend()
ax7.set_xlabel('t')
ax7.set_ylabel('energy')
plt.show()

######## plot EE vs bond #####
fig7, ax7 = plt.subplots()
ax7.semilogy(vm_sys.fe.component.entanglement_entropy_all(), '-x', label='elec')
# ax7.semilogy( vm_sys.fi.component.entanglement_entropy_all(), '-o', label='ion')

ax7.legend()
ax7.set_xlabel('bond number')
ax7.set_ylabel('EE')
if save_figs:
    fig7.savefig(fdir + 'plot_EE_' + fstr + '.png')
    print(fdir + 'plot_EE_' + fstr + '.png')

plt.show()

print('done')

