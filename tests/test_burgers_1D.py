import os, sys, pickle, time, glob

import scipy.optimize

sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
# import plot_defaults

# from defaults import *
import setup_.paths
from setup_.configs import *
import setup_.helper as helper_test
import helper_quimb as helper
import setup_test as test_setup

from axis import Axis
from basis.basis_spatial import SpatialBasis
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from field import Field, ScalarField


"""
for reference, see https://zingale.github.io/comp_astro_tutorial/advection_euler/burgers/burgers-methods.html
du/dt + u du/dx = 0
--> flux form:  du/dt + 1/s d/dx u**2 = 0

Note: finite difference with second order centered stencil (order 1) is not the same as finite volume
finite difference: du_i = 1/dx (F_{i+1} - F_{i-1})
finite volume: 1/dx (F(u_{i+1/2} - F(u_{i+1/2}))
F(u_{i+1/2}) = F((u_{i+1} + u_{i})/2)  != F(u_{i+1}) + F(u_{i})/2
F(u_{i-1/2}) = F((u_{i} + u_{i-1})/2)  != F(u_{i+1}) + F(u_{i})/2
"""


def euler_upwind(dx, dt: Numeric, vec: np.ndarray):
    """ perform a single time-step update at the specified indices
    https://zingale.github.io/comp_astro_tutorial/advection_euler/burgers/burgers-methods.html
        upwinding for Burgers:
        shock speed = (F(u_r) - F(u_l)) / (u_r - u_l) = 1/2 (u_r + u_l)

        shock solution
        --> u_s = u_{i+0.5,L} if S > 0;  u_{i+0.5,R} if S < 0
        rarefaction solution
        --> u_r = u_{i+0.5,L} if u_{i+0.5,L} > 0; u_{i+0.5,R} if u_{i+0.5,R} < 0: 0 o.w.

        solution to Riemann problem
        u_{i+1/2} = u_s if u_{i+1/2,L} > u_{i+1/2,R} else u_r
        F_{i+1/2} = u_{i+1/2}^2 / 2
        d/dx F_{i} = 1/dx (F_{i+1/2} - F_{i-1/2})
    """
    npts = len(vec)
    left_bc, right_bc = deriv_x.bc
    left_offset, right_offset = deriv_x.offset, deriv_x.offset_r

    out_data = np.zeros(vec.shape)
    for ind, val in enumerate(vec):

        ix0, sign0 = ind - 1, 1
        if ind == 0:
            if left_bc in [BCType.PERIODIC, BCType.ANTIPERIODIC]:
                ix0 = npts - 1
                sign0 = np.sign(left_bc.value)
            elif left_bc in [BCType.ZEROGRADIENT, BCType.SYMMETRIC, BCType.NEUMANN]:
                ix0 = 1
                sign0 = 1
            elif left_bc in [BCType.ZEROVALUE, BCType.ANTISYMMETRIC, BCType.DIRICHLET]:
                ix0 = 1
                sign0 = 0 if left_offset == 0 else -1

        ix2, sign2 = ind + 1, 1
        if ind == npts - 1:
            if right_bc in [BCType.PERIODIC, BCType.ANTIPERIODIC]:
                ix2 = 0
                sign2 = np.sign(right_bc.value)
            elif right_bc in [BCType.ZEROGRADIENT, BCType.SYMMETRIC, BCType.NEUMANN]:
                ix2 = npts - 1
                sign2 = 1
            elif right_bc in [BCType.ZEROVALUE, BCType.ANTISYMMETRIC, BCType.DIRICHLET]:
                ix2 = npts - 1
                sign2 = 0 if right_offset == 0 else -1

        val0 = vec[ix0]
        val2 = vec[ix2]

        ### compute shock speed at left, right boundary
        def get_flux(lval, rval):
            S_ = (lval + rval) / 2
            if lval > val:
                ## shock soln
                u_ = lval if S_ > 0 else rval
            else:
                ## rarefaction soln
                if lval > 0:
                    u_ = lval
                elif rval < 0:
                    u_ = rval
                else:
                    u_ = 0.0
            return u_ ** 2 / 2

        flux_left = get_flux(val0 * sign0, val)
        flux_right = get_flux(val, val2 * sign2)

        out_data[ind] = val + dt / dx * (flux_left - flux_right)

    return out_data


########################################

save_figs = False
save_data = True
load_data = False
restart = False
restart_from_T = 0  ## to restart from an output file (vs a saved restart file)
is_sqrt = True
is_darwin = False

if restart or save_data or save_figs:
    # sdir = '/pool001/erikaye/tns_pde_v2/data15-2/VM12-test/'
    # if not os.path.exists(sdir):
    #     os.makedirs(sdir)

    # fdir = './data_cross/Burgers/250919/'
    # fdir = './data_cross/Burgers/251103_xx/'    ## MIXED manually set in pde_burgers
    fdir = './data_cross/Burgers/251103/'  ## CROSS manually set in pde_burgers
    os.makedirs(fdir, exist_ok=True)

    print('fdir', fdir)


extra_str = ''

#######################
# standard parameters #
#######################
q = 2
KX = 1
Lx = flags.get('Lx', 7)

upwind = flags.get('do_upwind', False)
nu = 0.0 if upwind else 0.003 / max(1,Lx-7)**2 # 0.003       ## viscosity
flux_coeff = 1.0  # 1.0 # 1.0        ## nonolinear flux coeff
Lbox = 1.0
run = 3
## 1: cosine, 2: shock propagation, 3: rarefaction

print('upwind?', upwind)

if run == 1:    ## cosine shock
    def init_state(x):
        return np.sin(2 * np.pi * x / Lbox)
elif run == 2:
    ## shock propgation
    def init_state(x):
        return (x <= 0.5) * 1.0
elif run == 3:
    ## rarefaction
    def init_state(x):
        return (x > 0.5) * 1.0


## runtime parameter

TN_layout = LayoutType.SEQUENTIAL
x_map_key = 'F'

# TN_layout = LayoutType.PARALLEL  # _GROUP
# x_map_key = 'F'  # 'F'

order = 1   ## derivative order
if order > 1:
    extra_str += f'_o{order}'
T = 0.25 if run == 3 else 0.5  # 0.5   # 0.01  ## for diffusion
print('T', T)

te_order = flags.get('te_order', 86)
## 86:  alternative projection DLR-X + Euler (upwinding if upwind=True)
## 96:  alternative projection DLR-P + Euler

dt_frac = 0.9
dt = 0.002
save_every_nt = flags.get('save_every_nt', 1000) # 100 # 50  # 50   # 50 # 25000 # 1000 // 20 # // 5

DMAX = None   # 16  # None # 32 # None
cutoff = flags.get('cutoff', CUTOFF) # 1.0e-12
DMAX_F = DMAX
comp_style = 3


### initialize problem
x_map = helper_test.get_maps(x_map_key)[0]
ax_x = Axis(Lx, q, dx=Lbox/(q**Lx), endpoint=False, ax_map=x_map, basis=SpatialBasis())
grid_X = Grid1D('GX', [ax_x], TN_layout)
coords_x = CartesianCoordinateSpace('X', ax_x)
dt = ax_x.dx / 1.0 * dt_frac

print('ax map', ax_x.map)
print('cfl dt', ax_x.dx / 1.0)    ### shock speed = (flux left - flux right)/ax.dx
# exit()


compress_config = CompressionConfiguration()
for i in range(3):
    compress_config.set_compress_opts(i, max_bond=DMAX, cutoff_mode=CUTOFF_MODE, cutoff=cutoff)


#############################
## start building PDE system
#############################

#### define initial gridTNs
init_u_gtn = grid_X.make_gridTN(init_state(ax_x.xpts))

if run==1:
    deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
else:
    deriv_x = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)
init_u_gtn.ax_deriv_configs.update({ax_x: deriv_x})

### time step info
num_tsteps = int(T / dt)
print('dt', dt, num_tsteps)

### get file naming string
fstr = f'burgers{run}_L{Lx}_te{te_order}_dt{dt}_T{T}_nu{nu}_nl{flux_coeff}_c{cutoff}'
if upwind:
    fstr += '_up'
print('filename', fstr)


########################
#### run simulation ####
########################

#### define fields
init_u_field = ScalarField('u', grid_X, data=init_u_gtn, is_sqrt=False, compress_config=compress_config)

##### initialize tests
ts = [0]
errs_dist = [0.0]
tangent_space_errs = [0.0]
us = [init_state(ax_x.xpts)]
nt = 0

print('init bond dims', init_u_gtn.max_bond())
max_bond_fe = [init_u_gtn.max_bond()]
max_bond_fe_glob = [init_u_gtn.max_bond()]

norm_u = init_u_gtn.norm(is_sqrt=is_sqrt)  ## compute ||u||^2
print('field norm', norm_u)

from pde_burgers import Burgers, Burgers_FV

### BUILD PDE
bg_sys = Burgers(init_u_field, dissip_coeff=nu/2 if te_order in [61, 66] else nu,
                 flux_coeff=flux_coeff, power=2,
                 normalize=False, zipup=True, te_order=te_order,
                 upwind=upwind)
bg_sys_global = Burgers(init_u_field.copy(), dissip_coeff=nu, flux_coeff=flux_coeff, power=2,
                    normalize=False, zipup=True, te_order=1 if te_order in [81, 61, 86, 66] else 4,
                    upwind=False)
print('initialized Burgers')


fig1, ax1 = plt.subplots(1, 1)
time_ = nt * dt

u_data = init_u_gtn.get_data()
u_data = ax_x.basis.get_realspace_1D(u_data, 0)
ax1.plot(ax_x.xpts, np.real(u_data), label=f't={time_:2.2f}')
ax1.plot(ax_x.xpts, u_data**2, label='u2')
ax1.plot(ax_x.xpts[:-1], np.diff(u_data**2), label='deriv')
ax1.set_title(f'time {time_:3.2f}')
ax1.set_xlabel('x'), ax1.set_ylabel('u')

plt.show()


time1 = time.time()
internal_num_evals = [np.nan]
internal_ranks = [init_u_gtn.max_bond()]

print('starting time evolution')
while ts[-1] < T:

    dt_ = (dt * 0.1) if nt == 0 else dt

    print('dt/dx', dt_/ax_x.dx, dt_/(ax_x.dx**2))

    walltime = time.time()
    bg_sys = bg_sys.next_time_step(dt_, compress_level=1, verbose_plot=False, is_first_time_step=(nt==0 and not upwind)
                                   ) # is_first_time_step=(nt==0)) #, filter_bases=False) # (nt + 1) % (save_every_nt * 4) == 0)
    # bg_sys.f.component.average_neighbor(inplace=True, mu=0.5, spread=1)
    print('wall time', time.time() - walltime)

    u_field: 'ScalarField' = bg_sys.f
    internal_ranks += [u_field.component.info.get('internal_rank', np.nan)]
    internal_num_evals += [u_field.component.info.get('num_evals', np.nan)]

    ## measure error between global and local
    if upwind:
        prev_u_data = bg_sys_global.f.get_comp_data()
        new_u_data = euler_upwind(ax_x.dx, dt_, prev_u_data)
        u_glob_gtn = grid_X.map_state_to_mps(new_u_data) #, split_opts={'cutoff': cutoff, 'cutoff_mode': CUTOFF_MODE})
        u_glob = u_field.copy()
        u_glob.component = u_glob_gtn
    else:
        bg_sys_global = bg_sys_global.next_time_step(dt_, compress_level=5, verbose_plot=False, )
        u_glob = bg_sys_global.f.copy()

    ### error from DLRA (before final compression) wrt exact time step
    ts_err = u_glob.component.copy().distance(bg_sys.f.component.copy())
    tangent_space_errs += [ts_err]

    # if ts_err > 0.1:
    #     plt.figure()
    #     plt.plot(new_u_data, label='global')
    #     plt.plot(u_field.get_comp_data(), label='dlra')
    #     plt.legend()
    #     plt.figure()
    #     plt.semilogy(np.abs(u_field.get_comp_data() - new_u_data))
    #     plt.legend()
    #     plt.show()


    print('pre compress ranks', helper.inner_bond_sizes(u_field.component.data))
    u_field.compress(compress_level=1)
    print('post compress ranks', helper.inner_bond_sizes(u_field.component.data))
    max_bond_fe += [u_field.component.max_bond()]

    u_glob_trunc = u_glob.copy()
    u_glob_trunc.compress()
    max_bond_fe_glob += [u_glob_trunc.component.max_bond()]

    bg_sys_global.f = bg_sys.f.copy()

    ts += [ts[-1] + dt_]
    # us += [u_field.get_field_data()]
    nt += 1

    norm_u = u_field.component.norm(is_sqrt=True)
    print('result norm', nt, ts[-1], dt, norm_u, 'u max bond', u_field.max_bond(), 'Lx', Lx)
    print('u glob max bond', u_glob.max_bond())

    # ### compute errors
    # theory_u_gtn = grid_X.make_gtn_from_dicts([{ax_x: fe1_vxy}],data_type=DataType.MPS)
    # err = theory_u_gtn.distance(u_field.component) / np.sqrt( ax_x.npts )
    # errs_dist += [err]

    if nt % save_every_nt == 0 or ts[-1] > T:

        fig1, ax1 = plt.subplots(1,1)
        time_ = ts[-1]

        ## shock location:
        # x = x0 + (u_l + u_r)/2 * t = 0.5 + 0.5 * t

        u_data = u_field.get_comp_data()
        u_data_glob = u_glob.get_comp_data()
        ax1.plot(ax_x.xpts, u_data)
        ax1.plot(ax_x.xpts, u_data_glob, '--')
        ax1.plot(ax_x.xpts, init_u_gtn.get_data(), 'k:', linewidth=1.0) #, label='initial state')
        if run == 1:
            pass
            # ax1.plot([0.5, 0.5] , [-1.0, 1.0], 'k--')
        elif run == 2:
            ax1.plot([0.5 + 0.5 * ts[-1]]*2,[-0.1, 1.1], 'k--')
        elif run == 3:
            ax1.plot([0.5, 0.5 + ts[-1]] , [-0., 1.0], 'k--')

        plt.figure()
        plt.plot(u_data_glob - u_data)
        plt.title('distance')
        plt.show()

        # ## diffusion only with step fct init state
        ## 0.5 might not be exact position of step...
        exact_fx = (scipy.special.erf((ax_x.xpts - 0.5)/2/np.sqrt(nu*ts[-1]/2))
                            * (init_state(ax_x.xpts[-1]) - init_state(ax_x.xpts[0]))/2
                            + (init_state(ax_x.xpts[-1]) + init_state(ax_x.xpts[0]))/2)
        # ax1.plot(ax_x.xpts, exact_fx, 'k--')
        # semi-infinite: L >= 4 sqrt(nu * t) --> t <= L**2/16 / nu
        print('box length', ax_x.xpts[-1] - ax_x.xpts[0], 4 * np.sqrt(nu * ts[-1]))
        print('dt, dx, err', dt, ax_x.dx, np.linalg.norm(exact_fx - u_data)/np.sqrt(ax_x.npts))
        print('err', dt, np.linalg.norm(exact_fx - u_data) / np.sqrt(ax_x.npts))

        ax1.set_title(f'time {time_:3.2f}')
        ax1.set_xlabel('x'), ax1.set_ylabel('u')
        # ax1.legend()
        plt.show()

        plt.figure()
        plt.plot(ts, max_bond_fe, label='compressed rank')
        plt.plot(ts, internal_ranks, ':', label='internal rank')
        plt.plot(ts, max_bond_fe_glob, '--', label='global rank')
        plt.legend()
        plt.title('max rank')

        plt.figure()
        plt.plot(ts, tangent_space_errs)
        plt.xlabel('t')
        plt.ylabel('projection err')
        plt.title('tangent space errors')

        plt.figure()
        plt.plot(ts, internal_num_evals)
        plt.plot([0, ts[-1]], [2**Lx, 2**Lx], '--')
        plt.xlabel('t')
        plt.ylabel('# evals')

        # plt.figure()
        # plt.semilogy(ts, errs_dist, label='dist')
        # plt.title('error')
        # plt.xlabel('t'), plt.ylabel('error')
        # plt.grid(color='gray',axis='y')
        # plt.legend()

        plt.show()

        # helper.partition_1D_mps(u_field.component.data, Lx - 3)
        # helper.partition_1D_mps(u_field.component.data, Lx - 2)

        ######## plot EE vs bond #####
        fig7, ax7 = plt.subplots()
        ax7.semilogy(bg_sys.f.component.entanglement_entropy_all(), '-x', label='u')
        ax7.legend()
        ax7.set_xlabel('bond number')
        ax7.set_ylabel('EE')
        plt.show()


print('wall time', time.time() - time1)


if save_data:
    pickle.dump(np.array([ts, max_bond_fe]), open(fdir + fstr +  '_ranks.pkl', 'wb'))
    pickle.dump(u_data, open(fdir + fstr +  '_u.pkl', 'wb'))
    pickle.dump(np.array([ts, tangent_space_errs]), open(fdir + fstr + '_ts_errs.pkl', 'wb'))
    pickle.dump(np.array([ts, max_bond_fe_glob]), open(fdir + fstr + '_glob_rank.pkl', 'wb'))
    pickle.dump(np.array([ts, internal_ranks, internal_num_evals]), open(fdir + fstr + '_cost.pkl', 'wb'))


##############

fig7, ax7 = plt.subplots()
ax7.semilogy(ax_x.xpts, u_field.get_field_data(), label='u')

ax7.legend()
ax7.set_xlabel('t')
ax7.set_ylabel('energy')
plt.show()


# ######## plot EE vs bond #####
# fig7, ax7 = plt.subplots()
# ax7.semilogy(bg_sys.f.component.entanglement_entropy_all(), '-x', label='elec')
#
# ax7.legend()
# ax7.set_xlabel('bond number')
# ax7.set_ylabel('EE')
# if save_figs:
#     fig7.savefig(fdir + 'plot_EE_' + fstr + '.png')
#     print(fdir + 'plot_EE_' + fstr + '.png')
#
# plt.show()



print('done')
