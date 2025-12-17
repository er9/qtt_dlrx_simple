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

    fdir = './data_cross/Burgers/dense_upwind/'
    os.makedirs(fdir, exist_ok=True)

    print('fdir', fdir)


extra_str = ''

#######################
# standard parameters #
#######################

nu = 0.0  # 0.003  # 0.0  # 0.005  # 0.001 # 0.0         ## viscosity
flux_coeff = 1.0  # 1.0 # 1.0        ## nonolinear flux coeff
Lbox = 1.0
run = 1

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
q = 2
KX = 1
Lx = flags.get('Lx', 7)

TN_layout = LayoutType.SEQUENTIAL
x_map_key = 'F'

# TN_layout = LayoutType.PARALLEL  # _GROUP
# x_map_key = 'F'  # 'F'

order = 1   ## derivative order
if order > 1:
    extra_str += f'_o{order}'
T = 0.5  # 0.5   # 0.01  ## for diffusion
print('T', T)

upwind = False  # True  # True
te_order = flags.get('te_order', 86)  # 81   # 86    # 74, 4
dt_frac = 0.9
# dt = 0.003    ## te 4:  < 0.01,  te 74:  < 0.02,  te 70 ok but ringing instability
dt = 0.002  # 1 # 25
# dt = 0.001 / 16    ## for diffusion only
save_every_nt = 1000 # 100 # 50  # 50   # 50 # 25000 # 1000 // 20 # // 5

x_vals = np.linspace(0,1, q**Lx, endpoint=False)
dx = x_vals[1] - x_vals[0]
dt = dx / 1.0 * dt_frac

print('cfl dt', dx / 1.0)    ### shock speed = (flux left - flux right)/ax.dx


#############################
## start building PDE system
#############################

if run==1:
    deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
else:
    deriv_x = DerivativeConfiguration(left_bc=BCType.SYMMETRIC, order=order, fd_type=FDType.CENTER)

### time step info
num_tsteps = int(T / dt)
print('dt', dt, num_tsteps)

### get file naming string
fstr = f'burgers{run}_L{Lx}_dt{dt}_T{T}_nu{nu}_nl{flux_coeff}_up'
print('filename', fstr)


########################
#### run simulation ####
########################

def euler_upwind(dt: Numeric, vec: np.ndarray):
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


##### initialize tests
ts = [0]
errs_dist = [0.0]
tangent_space_errs = [0.0]
u_data = init_state(x_vals)
us = [u_data]
nt = 0
time_ = nt * dt

fig1, ax1 = plt.subplots(1, 1)
ax1.plot(x_vals, np.real(u_data), label=f't={time_:2.2f}')
ax1.set_title(f'time {time_:3.2f}')
ax1.set_xlabel('x'), ax1.set_ylabel('u')

plt.show()


time1 = time.time()

print('starting time evolution')
while ts[-1] < T:

    dt_ = (dt * 0.1) if nt == 0 else dt

    walltime = time.time()
    u_data = euler_upwind(dt_, u_data)
    print('wall time', time.time() - walltime)

    us += [u_data]
    ts += [ts[-1] + dt_]
    # us += [u_field.get_field_data()]
    nt += 1


    # ### compute errors
    # theory_u_gtn = grid_X.make_gtn_from_dicts([{ax_x: fe1_vxy}],data_type=DataType.MPS)
    # err = theory_u_gtn.distance(u_field.component) / np.sqrt( ax_x.npts )
    # errs_dist += [err]

    if nt==2 or nt % save_every_nt == 0 or ts[-1] > T or np.abs(ts[-1] - 0.175) < dt_ :

        fig1, ax1 = plt.subplots(1,1)
        time_ = ts[-1]

        ## shock location:
        # x = x0 + (u_l + u_r)/2 * t = 0.5 + 0.5 * t

        ax1.plot(x_vals, u_data)

        if run == 1:
            pass
            # ax1.plot([0.5, 0.5] , [-1.0, 1.0], 'k--')
        elif run == 2:
            ax1.plot([0.5 + 0.5 * ts[-1]]*2,[-0.1, 1.1], 'k--')
        elif run == 3:
            ax1.plot([0.5, 0.5 + ts[-1]] , [-0., 1.0], 'k--')

        plt.show()


if save_data:
    pickle.dump(us, open(fdir + fstr +  '_us.pkl', 'wb'))


##############

fig7, ax7 = plt.subplots()
ax7.semilogy(x_vals, u_data, label='u')

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

