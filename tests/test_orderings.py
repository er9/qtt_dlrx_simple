"""Driver: effect of tensor-network site ordering / layout on entanglement entropy.

Runs a Vlasov-Poisson Landau-damping simulation and measures the entanglement
entropy of the resulting distribution function under different MPS orderings/layouts.
"""
import os, sys, pickle, time, glob
sys.path.append('../')

from setup_.configs import *
import setup_.helper as helper_test
import helper_quimb as helper

from axis import Axis
import axis_map
from coord.coord_sys import Coordinate
from coord.cartesian import CartesianCoordinateSpace
from grid1D import Grid1D
from grid_comb import GridsComb
from gridTN_1D import GridTN1D
from field import Field, ScalarField
from pde_vlasovES import VlasovPoisson

"""
Measure EE of result of a distribution function (obtained via Landau damping)
"""

#######################
# standard parameters #
#######################

plasma_config = UnitsConfiguration()
charge = plasma_config.e  # unit of charge
eV = plasma_config.eV  # unit of kB*T

mass_e = 1.0  # mass of electron
mass_i = 1836 * mass_e  # mass of ion
n0_e = 1.0  # electron number density
n0_i = n0_e  # ion number density
T_e = 1.0  # electron temperature [eV]
T_i = T_e  # ion temperature [eV]

ion_config = IonConfiguration(n0=n0_i, mass=mass_i, T=T_i, units_config=plasma_config)
elc_config = ElcConfiguration(n0=n0_e, mass=mass_e, T=T_e, units_config=plasma_config)
vth_e = elc_config.vth  # electron thermal speed
wp_e = elc_config.wp  # plasma e' frequency
lamD = elc_config.lamD  # Debye length
wp_i = ion_config.wp  # plasma i+ frequency
vth_i = ion_config.vth  # ion thermal speed


#######################

markers = {6: '^', 7: 'x', 8: 'o', 9: 'v', 10: 's'}
lstyles = {8: {'color': '#99CC00', 'linewidth': 1.0},
           16: {'color': '#E81916', 'linewidth': 0.75},
           32: {'color': '#45bfb6', 'linewidth': 1.0},
           48: {'color': '#f43e1a', 'linewidth': 1.0},
           64: {'color': '#1D3BEB', 'linewidth': 1.0},
           90: {'color': '#D55CE8', 'linewidth': 1.0},  # '#a93f95'
           128: {'color': '#0143ad', 'linewidth': 1.0},
           None: {'color': 'k', 'linewidth': 2.0}}

save_fig = False
fig_EE, ax_EE = plt.subplots(2, 1, figsize=(4.5, 4))  # EE vs L (noramlized to 1)
fig_fs, ax_fs = plt.subplots(2, 1)
fig_err, ax_err = plt.subplots(2, 1, figsize=(4.5, 4))  # compression error at some point in time

pltdir = 'data/plots/'
if not os.path.exists(pltdir):
    os.makedirs(pltdir)

## Landau damping damping
fdir = 'data/VP1-damp-final/'
sdir = fdir

#####################
# define parameters #
#####################

target_T = 60.0
conv_TN_layout = LayoutType.SEQUENTIAL
conv_ax_map_key = 'FB'

# Ls = [6, 7, 8, 9, 10]
# Ds = [int(D) for D in np.logspace(3, 7, 13, base=2)]
Ls = [7, 8, 9, 10]
Ds = [int(D) for D in np.logspace(3, 9, 13, base=2)]
print(Ds)

## initial conditions
A = 5.0e-1
k = 0.10/lamD
# k = 0.75 / lamD

T = 60.0
cfl = 0.9
order = 1
te_order = 4
init_compress_opts = {'max_bond': None, 'cutoff': 1.0e-20, 'cutoff_mode':'rsum2'}

Astr = np.format_float_scientific(A, precision=1, exp_digits=1, trim='-')

for L in Ls[::-1]:

    ## grid parameters
    q = 2
    K = 1
    npts = q ** L

    if k == 0.10:
        TN_type = 'sf2'
        TN_layout = LayoutType.SEQUENTIAL
        ax_map_key = 'BF'
        fstr1 = f'testU1x30_TN{TN_type}_cut20_k{k:1.2f}_A{Astr}'  # L7; k=0.50, 0.75
    elif k == 0.75:
        if L == 6:
            TN_type = 'sf3'
            TN_layout = LayoutType.SEQUENTIAL
            ax_map_key = 'FB'
            fstr1 = f'testU1_TN{TN_type}_A{Astr}_k{k:1.2f}_C10_mbNone'
        else:
            TN_type = 'sf2'
            TN_layout = LayoutType.SEQUENTIAL
            ax_map_key = 'BF'
            fstr1 = f'testU1_TN{TN_type}_cut20_k{k:1.2f}_A{Astr}'  # L7; k=0.50, 0.75
    else:
        raise IOError('no files for specified parameters')

    fstr = fstr1 + f'_L{L:02d}_cfl{cfl}_T{T}_te{te_order}_m{mass_i}'
    print(fstr)

    pltstr = f'damp_convTN{conv_TN_layout.value}{conv_ax_map_key}_{fstr1}_cfl{cfl}'
    print('pltsr', pltstr)


    ###############
    ## load data ##
    ###############
    try:
        ts_sim = pickle.load(open(fdir + 'ts_' + fstr + f'.pkl', 'rb'))
    except IOError:
        ts_sim = pickle.load(open(sdir + 'restart/ts_' + fstr + f'.pkl', 'rb'))

    if target_T == T:
        fe_t = pickle.load(open(fdir + 'fe_' + fstr + '.pkl', 'rb'))
        fi_t = pickle.load(open(fdir + 'fi_' + fstr + '.pkl', 'rb'))
        T_ = target_T
    else:
        target_nt = np.argmin(np.abs(np.array(ts_sim) - target_T))
        nt = int(np.round(target_nt * 1. / 250) * 250)  # get closest multiple of 250
        T_ = ts_sim[nt]

        fe_t = pickle.load(open(sdir + 'restart/fe_' + fstr + f'-nt{nt}.pkl', 'rb'))
        fi_t = pickle.load(open(sdir + 'restart/fi_' + fstr + f'-nt{nt}.pkl', 'rb'))



    #################
    ## build grids ##
    #################

    x_vals = np.linspace(-np.pi / k, np.pi / k, npts, endpoint=False)
    if k == 0.10:
        ve_vals = np.linspace(-30 * vth_e, 30 * vth_e, npts, endpoint=False)
    elif k == 0.75:
        ve_vals = np.linspace(-6 * vth_e, 6 * vth_e, npts, endpoint=False)
    else:
        raise NotImplementedError
    vi_vals = np.linspace(-6 * vth_i, 6 * vth_i, npts, endpoint=False)

    ## define axis objects
    X = Coordinate('X', CoordinateType.X)
    V = Coordinate('V', CoordinateType.X)

    map_x, map_v = get_maps(ax_map_key)
    print('maps', map_x, map_v)
    ax_x = Axis(L, q, coordinate=X, xpts=x_vals, ax_map=map_x)
    ax_vi = Axis(L, q, coordinate=V, xpts=vi_vals, ax_map=map_v)
    ax_ve = Axis(L, q, coordinate=V, xpts=ve_vals, ax_map=map_v)
    print('dx', ax_x.dx, 'dv e', ax_ve.dx, 'dv i', ax_vi.dx)

    coords_x = CartesianCoordinateSpace('X', ax_x)
    coords_vi = CartesianCoordinateSpace('Vi', ax_vi)
    coords_ve = CartesianCoordinateSpace('Ve', ax_ve)

    ## configure derivatives
    deriv_x = DerivativeConfiguration(left_bc=BCType.PERIODIC, order=order, fd_type=FDType.CENTER)
    deriv_v = DerivativeConfiguration(left_bc=BCType.ZEROGRADIENT, order=order, fd_type=FDType.CENTER)

    ## define grids
    grid_i = Grid1D('XVi', (ax_x, ax_vi), layout_type=TN_layout)
    grid_e = Grid1D('XVe', (ax_x, ax_ve), layout_type=TN_layout)
    grid_V = Grid1D('X', (ax_x,), layout_type=TN_layout)


    ######################
    ## reconstruct data ##
    ######################

    fi_gtn = grid_i.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_x, ax_vi: deriv_v})
    fe_gtn = grid_e.make_empty_gridTN(ax_deriv_configs={ax_x: deriv_x, ax_ve: deriv_v})

    fi_gtn.data = fi_t
    fe_gtn.data = fe_t

    fi_t_data = fi_gtn.get_data()
    fe_t_data = fe_gtn.get_data()

    show_dist = True
    if show_dist:
        plt.figure()
        vmax_i = np.max(fi_t_data)
        vmin_i = np.min(fi_t_data)
        plt.imshow(fi_t_data.T, extent=(x_vals[0],x_vals[-1],vi_vals[0],vi_vals[-1]),
                        aspect='auto', origin='lower',
                        vmin=vmin_i,vmax=vmax_i)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$v$')
        plt.title(r'$L$ = '+f'{L}')
        plt.colorbar()
        if save_fig:  plt.savefig(pltdir+f'plt_fi_{fstr}.png', dpi=300)


        plt.figure()
        vmax_e = np.max(fe_t_data)
        vmin_e = np.min(fe_t_data)
        plt.imshow(fe_t_data.T, extent=(x_vals[0],x_vals[-1],ve_vals[0],ve_vals[-1]),
                        aspect='auto', origin='lower',
                        vmin=vmin_e,vmax=vmax_e)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$v$')
        plt.title(r'$L$ = '+f'{L}')
        plt.colorbar()
        if save_fig:   plt.savefig(pltdir+f'plt_fe_{fstr}.png', dpi=300)
        # plt.show()

    #####################
    ## build new grids ##
    #####################

    new_map_x, new_map_v = get_maps(conv_ax_map_key)
    print('new maps', new_map_x, new_map_v)
    new_ax_x = Axis(L, q, coordinate=X, xpts=x_vals, ax_map=new_map_x)
    new_ax_vi = Axis(L, q, coordinate=V, xpts=vi_vals, ax_map=new_map_v)
    new_ax_ve = Axis(L, q, coordinate=V, xpts=ve_vals, ax_map=new_map_v)
    print('dx', new_ax_x.dx, 'dv e', new_ax_ve.dx, 'dv i', new_ax_vi.dx)

    coords_x = CartesianCoordinateSpace('X', new_ax_x)
    coords_vi = CartesianCoordinateSpace('Vi', new_ax_vi)
    coords_ve = CartesianCoordinateSpace('Ve', new_ax_ve)

    ## define grids
    new_grid_i = Grid1D('XVi', (new_ax_x, new_ax_vi), layout_type=conv_TN_layout)
    new_grid_e = Grid1D('XVe', (new_ax_x, new_ax_ve), layout_type=conv_TN_layout)
    new_grid_V = Grid1D('X', (new_ax_x,), layout_type=conv_TN_layout)


    ########################
    ## fe, fi in new Grid ##
    ########################

    new_fe = new_grid_e.map_state_to_mps(fe_t_data, split_opts=init_compress_opts,
                                         ax_deriv_configs=fe_gtn.ax_deriv_configs)
    new_fi = new_grid_i.map_state_to_mps(fi_t_data, split_opts=init_compress_opts,
                                         ax_deriv_configs=fi_gtn.ax_deriv_configs)

    ## measure entanglement entropy
    EEs_i = helper.entanglement_entropy_all(new_fi.data)
    EEs_e = helper.entanglement_entropy_all(new_fe.data)

    if conv_TN_layout == LayoutType.PARALLEL:
        pos = np.arange(2 * K * L - 1)
        comp_midpt = False
    elif conv_TN_layout == LayoutType.PARALLEL_GROUP:
        pos = np.arange(L - 1)
        comp_midpt = False
    else:
        pos = np.arange(2 * K * L - 1) - L * K + 1
        comp_midpt = False  # True

    print(len(pos), len(EEs_i), len(EEs_e))
    ax_EE[0].semilogy(pos, EEs_i, marker=markers[L], label=r'$L$=' + f'{L}')
    ax_EE[1].semilogy(pos, EEs_e, marker=markers[L], label=r'$L$=' + f'{L}')

    ## plot cross section f(x=0,v) for L=10
    if L == 10:
        x0_ind = np.argmin(np.abs(x_vals))
        ax_fs[0].plot(vi_vals, fi_t_data[x0_ind, :], label=f'no comp.', **lstyles[None])
        ax_fs[1].plot(ve_vals, fe_t_data[x0_ind, :], label=f'no comp.', **lstyles[None])
        ax_ins = ax_fs[1].inset_axes([0.7, 0.47, 0.27, 0.47])
        ax_ins.xaxis.label.set_size(12.)
        if k == 0.75:
            if target_T == 60:
                ax_ins.set_xlim(1.75, 3.75)
                ax_ins.set_ylim(-0.05, 0.25)
            else:
                ax_ins.set_xlim(1.75, 3.75)
                ax_ins.set_ylim(-0.05, 0.25)
        elif k == 0.10:
            ax_ins.set_xlim(1.0, 5.25)
            ax_ins.set_ylim(-0.01, 0.12)
        ax_ins.get_yaxis().set_visible(False)
        ax_ins.plot(ve_vals, fe_t_data[x0_ind, :], **lstyles[None])

    errsD_L0_e, errsD_L0_i = [], []
    errsD_L1_e, errsD_L1_i = [], []
    errsD_L2_e, errsD_L2_i = [], []
    errsD_e_integ, errsD_i_integ = [], []

    ## measure rms compression error vs D
    for D in Ds:
        print('max bond', new_fe.max_bond(), new_fi.max_bond())
        compress_opts = {'max_bond': D, 'cutoff': 1.0E-20, 'cutoff_mode': 'rsum2', 'do_midpt': comp_midpt}

        fe_t_D = new_fe.compress(inplace=False, compress_opts=compress_opts)
        fe_t_D_data = new_grid_e.map_mps_to_state(fe_t_D)
        errsD_L0_e += [np.max(np.abs(fe_t_data - fe_t_D_data))]
        errsD_L1_e += [np.sum(np.abs(fe_t_data - fe_t_D_data)) / npts ** (K * 2)]
        errsD_L2_e += [np.linalg.norm((fe_t_data - fe_t_D_data)) / npts ** (K * 2 / 2)]

        fi_t_D = new_fi.compress(inplace=False, compress_opts=compress_opts)
        fi_t_D_data = new_grid_i.map_mps_to_state(fi_t_D)
        errsD_L0_i += [np.max(np.abs(fi_t_data - fi_t_D_data))]
        errsD_L1_i += [np.sum(np.abs(fi_t_data - fi_t_D_data)) / npts ** (K * 2)]
        errsD_L2_i += [np.linalg.norm((fi_t_data - fi_t_D_data)) / npts ** (K * 2 / 2)]

        ## plot cross-section of compressed f
        if L == 10:
            if k == 0.75:
                if not D == 16:  continue
            elif k == 0.10:
                if not (D == 90):   continue
            print('L=10, plot D =', D)
            x0_ind = np.argmin(np.abs(x_vals))
            ax_fs[0].plot(vi_vals, fi_t_D_data[x0_ind, :], label=f'$D$ = {D}', **lstyles[D])
            ax_fs[1].plot(ve_vals, fe_t_D_data[x0_ind, :], label=f'$D$ = {D}', **lstyles[D])
            ax_ins.plot(ve_vals, fe_t_D_data[x0_ind, :], **lstyles[D])

    ax_fs[0].set_xlabel(r'$v$')
    ax_fs[1].set_xlabel(r'$v$')
    ax_fs[0].set_ylabel(r'$f_i(x=0,v)$')  # (r'$\displaystyle f_i(x=0,v)$')
    ax_fs[1].set_ylabel(r'$f_e(x=0,v)$')  # (r'$\displaystyle f_e(x=0,v)$')
    ax_fs[0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper right')
    if L == 10:
        # ax_fs[0].set_xlim(0,np.max(vi_vals))
        # ax_fs[1].set_xlim(0,np.max(ve_vals))
        ## zoom in insert
        fig_fs.subplots_adjust(right=0.75)
        if save_fig:
            fig_fs.savefig(pltdir + f'plt_f_compress_{pltstr}_L10_T{target_T:2.1f}.png', dpi=300)

    ## plot compression errors
    ax_err[0].semilogy(np.log(Ds) / np.log(2), errsD_L2_i, marker=markers[L], label=r'$L$=' + f'{L}')
    ax_err[1].semilogy(np.log(Ds) / np.log(2), errsD_L2_e, marker=markers[L], label=r'$L$=' + f'{L}')

# ax_err[0].set_xlabel('$\log_2(D)#')
ax_err[1].set_xlabel('$\log_2(D)$')
ax_err[0].set_ylabel('rms error ' + r'$[f_i]$')
ax_err[1].set_ylabel('rms error ' + r'$[f_e]$')
ax_err[0].xaxis.set_visible(False)
ax_err[0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
if k == 0.10:
    ax_err[0].set_ylim(1.0e-6, 1.0e-1)
    ax_err[1].set_ylim(1.0e-6, 1.0e-1)
elif k == 0.75:
    ax_err[0].set_ylim(1.0e-7, 1.0e-2)
    ax_err[0].set_yticks([1.0e-7, 1.0e-5, 1.0e-3])
    ax_err[1].set_ylim(1.0e-6, 1.0e-1)
# if conv_TN_layout == LayoutType.SEQUENTIAL and conv_ax_map_key == 'FF':
#     ax_err[0].text(0.85, 0.85, 'ion', transform=ax_err[0].transAxes)
#     ax_err[1].text(0.85, 0.85, 'elec', transform=ax_err[1].transAxes)
fig_err.subplots_adjust(right=0.7, bottom=0.15, top=0.9, hspace=0.1, left=0.2)
if save_fig:
    fig_err.savefig(pltdir + f'plt_f_compress_error_{pltstr}_T{target_T:2.1f}.png', dpi=300)
    # fig_err.savefig(pltdir + f'plt_f_compress_error_{pltstr}_T{target_T:2.1f}.pdf')
    print('fig err', pltdir + f'plt_f_compress_error_{pltstr}_T{target_T:2.1f}.png')

if conv_TN_layout == LayoutType.PARALLEL:
    ax_EE[0].set_xticks(list(range(0, 21, 5)))
    ax_EE[0].set_xticklabels([''] * 5)
    ax_EE[1].set_xlabel('bond number')
    ax_EE[1].set_xticks(list(range(0, 21, 5)))
elif conv_TN_layout == LayoutType.PARALLEL_GROUP:
    ax_EE[0].set_xticks(list(range(0, 11, 2)))
    ax_EE[0].set_xticklabels([''] * 6)
    ax_EE[1].set_xlabel('bond number')
    ax_EE[1].set_xticks(list(range(0, 11, 2)))
else:
    ax_EE[0].set_xticks(list(range(-10, 11, 5)))
    ax_EE[0].set_xticklabels([''] * 5)
    ax_EE[1].set_xlabel('bond number from center')
    ax_EE[1].set_xticks(list(range(-10, 11, 5)))

if conv_TN_layout == LayoutType.SEQUENTIAL and conv_ax_map_key == 'FF':
    ax_EE[0].text(0.85, 0.85, 'ion', transform=ax_EE[0].transAxes)
    ax_EE[1].text(0.85, 0.85, 'elec', transform=ax_EE[1].transAxes)

ax_EE[0].set_ylabel(r'$S_{MPS}\,[f_i\,]$', labelpad=10)
# ax_EE[0].set_ylim(-0.1, 1.25)
ax_EE[0].set_ylim(1.0e-5, 10.0)
# ax_EE[0].set_yticks([1.0e-7, 1.0e-5,1.0e-3])


ax_EE[1].set_ylabel(r'$S_{MPS}\,[f_e\,]$', labelpad=10)
ax_EE[1].set_ylim(1.0e-2, 10.0)
#  if k==0.10:
#      # ax_EE[1].set_ylim(-0.1, 3.35)
#      ax_EE[1].set_ylim(-0.1, 3.35)
#  elif k==0.75:
#      # ax_EE[1].set_ylim(-0.1, 2.35)
#      ax_EE[1].set_ylim(-0.1, 2.35)
ax_EE[0].legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
ax_EE[0].xaxis.set_visible(False)
fig_EE.subplots_adjust(right=0.7, bottom=0.15, top=0.9, hspace=0.1, left=0.2)
if save_fig:
    fig_EE.savefig(pltdir + f'plt_EE_{pltstr}_T{target_T:2.1f}_log.png', dpi=300)
    # fig_EE.savefig(pltdir + f'plt_EE_{pltstr}_T{target_T:2.1f}_log.pdf')

plt.show()

