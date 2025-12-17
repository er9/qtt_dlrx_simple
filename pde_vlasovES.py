from setup_.configs import *
import helper_quimb as helper
import helper_sl as helper_sl
import helper_dmrg
from axis import Axis
from grid1D import Grid1D
from field import Field, ScalarField
from pde_system import PDE_system
from pde_vlasov import Vlasov
from pde_boltzmann import Boltzmann

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from grid import Grid


class VlasovPoisson(Vlasov):
    """ df/dt + v grad(f) + q/m (E) grad_v(f) = 0
    """
    def __init__(self,
                 sys_fe: Optional[Union['ScalarField', 'Boltzmann']],
                 sys_fi: Optional[Union['ScalarField', 'Boltzmann']],
                 potential: Optional['ScalarField'] = None,
                 grid_X: Optional['Grid']=None,
                 coords_x:  Optional['CoordinateSystem'] = None,
                 coords_ve: Optional['CoordinateSystem'] = None,
                 coords_vi: Optional['CoordinateSystem'] = None,
                 elc_params: Optional[SpeciesConfiguration] = None,
                 ion_params: Optional[SpeciesConfiguration] = None,
                 f0_gradv_e=None, f0_gradv_i=None,
                 evolve_ion=True, normalize=True, upwind=False, zipup=False, conservative=True,
                 background_fi0: Optional[Union['ScalarField','Boltzmann']] = None,
                 background_fe0: Optional[Union['ScalarField','Boltzmann']] = None,
                 background_E0: Optional['Field'] = None,
                 potential_boundary_conditions=(None,None),
                 te_order=2,
                 compress_levels=None,
                 # init_compress_opts=None, te_compress_opts=None
                 ):
        """ dist_e:  distribution of electrons (scalar Field obj). generally on x,v grid
            dist_i:  distribution of ions (scalar Field obj). generally on x,v grid
            potential:  electric potential (scalar Field obj). generally on x grid
            boundary_conditions: boundary conditions on V
               tuple of (bc_vectors [m vectors or mps], bc_results [m float]) such that
               bc_vectors (dot) V_data = bc_results
        """
        grid_X = grid_X if potential is None else potential.grid

        super().__init__(sys_fe, sys_fi, grid_X=grid_X,
                         coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                         elc_params=elc_params, ion_params=ion_params,
                         f0_gradv_e=f0_gradv_e, f0_gradv_i=f0_gradv_i,
                         background_fe0=background_fe0, background_fi0=background_fi0,
                         normalize=normalize, upwind=upwind, zipup=zipup, evolve_ion=evolve_ion,
                         conservative=conservative,
                         te_order=te_order, compress_levels=compress_levels)

        self.names['V']  = potential.name if potential is not None else 'V'
        self.field_names += [self.names['V']]
        if potential is not None:
            self._fields[self.names['V']] = potential

        self.evolve_V = True
        self.inv_laplacian_mpo = None
        self.laplacian_mpo = None
        self.potential_boundary_conditions = potential_boundary_conditions

        ## background fields
        self.background_E0 = background_E0      # a field (no time evolution equations)
        if background_E0 is not None:
            sys_fe.background_force_neg = self.compute_force_term(background_only=True)
            sys_fi.background_force_neg = self.compute_force_term(is_ion=True, background_only=True)
            sys_fe.background_force_neg.scalar_multiply(-1, inplace=True)
            sys_fi.background_force_neg.scalar_multiply(-1, inplace=True)
        self.divE0 = None
        self._total_E = None


        # self.compress_F = True
        # self.compress_F_opts = {'max_bond':None, 'cutoff':CUTOFF, 'cutoff_mode':CUTOFF_MODE}
        #
        # self.collision = CollisionConfiguration(None)
        #
        # self.dist0_e = None
        # self.dist0_i = None
        #
        # self.v_from_v0_e = None   # data_ve - vdrift_e
        # self.v_from_v0_i = None   # data_vi - vdrift_i


    # @property
    # def fe(self) -> Optional['ScalarField']:
    #     try:   return self.fields[self.names['fe']]
    #     except KeyError:   return None
    #
    # @fe.setter
    # def fe(self,new_field):
    #     self.fields[self.names['fe']] = new_field
    #
    #
    # @property
    # def fi(self) -> Optional['ScalarField']:
    #     try:   return self.fields[self.names['fi']]
    #     except KeyError:   return None
    #
    # @fi.setter
    # def fi(self,new_field):
    #     self.fields[self.names['fi']] = new_field


    @property
    def V(self) -> Optional['ScalarField']:
        try:   return self._fields[self.names['V']]
        except KeyError:   return None

    @V.setter
    def V(self,new_field):
        self._fields[self.names['V']] = new_field
        self._total_E = None

    @property
    def total_E(self):
        return self._total_E


    @property
    def matl_params(self):
        if self.sys_fe is not None:
            return self.sys_fe.matl_params
        elif self.sys_fi is not None:
            return self.sys_fi.matl_params
        return None


    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this with new fields
            deep:  copy background PDE objects too
        """
        if len(new_fields) < 3:
            new_fields = new_fields + (None,) *(3-len(new_fields))

        dist_e, dist_i, potential = new_fields[:3]
        new_system = super().create_like(dist_e, dist_i, recalc=recalc, deep=deep)
        new_system.V = potential
        # if potential is None:   new_system.names['V' ] = self.names['V' ]     ## not needed anymore

        new_system.inv_laplacian_mpo = self.inv_laplacian_mpo
        new_system.laplacian_mpo = self.laplacian_mpo
        new_system.potential_boundary_conditions = self.potential_boundary_conditions

        new_system.background_E0 = self.background_E0
        new_system.sys_fe.background_force_neg = self.sys_fe.background_force_neg
        new_system.sys_fi.background_force_neg = self.sys_fi.background_force_neg
        new_system.divE0 = self.divE0

        # new_system.collision = self.collision
        #
        # new_system.compress_F = self.compress_F
        # new_system.compress_F_opts = self.compress_F_opts
        #
        # new_system.dist0_i = self.dist0_i
        # new_system.dist0_e = self.dist0_e

        # new_system.v_from_v0_i = self.v_from_v0_i
        # new_system.v_from_v0_e = self.v_from_v0_e

        return new_system


    def copy(self):
        """ create a new system like this with new fields
        """
        # fe = self.fe.copy() if self.fe is not None else None
        # fi = self.fi.copy() if self.fi is not None else None
        # V = self.V.copy() if self.V is not None else None
        new_system = super().copy()
        new_system._total_E = self._total_E
        return new_system



    # def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, force_term: Optional['Field'] = None,
    #                method_v=None, method_f=None, is_first_time_step=False, is_last_time_step=False, inplace=False,
    #                compress_level=1, compress_level1=4, compress_level2=5,
    #                verbose_plot: bool = False) -> 'VlasovPoisson':
    #     """ take split step for EM system and ion,electron advection terms
    #         each step is just an Euler update
    #         TODO: semiLagrangian advection MPOs (see Sec 3.3, 3.4 https://arxiv.org/pdf/2201.03471.pdf)
    #     """
    #     ## at initialization, need to evolve EM_sys with dt/2
    #     # self.upwind = True
    #     state0 = self if inplace else self.copy()
    #     # verbose_plot = True
    #
    #     if compress_level == 0:
    #         comp1 = comp2 = comp3 = comp4 = comp5 = 0
    #     else:
    #         comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)
    #
    #     # ## dt (dt/2 if first), dt (dt/2 if last), V structure
    #     # state0.sys_fe.force_term = self.compute_force_term(is_ion=False, compress_level1=comp3, compress_level2=0)
    #     # state0.sys_fi.force_term = self.compute_force_term(is_ion=True, compress_level1=comp3, compress_level2=0)
    #     # super(VlasovPoisson, state0).split_step(dt, deriv0, method_h=method_h, method_v=method_v, inplace=True,
    #     #                                         is_first_time_step=is_first_time_step,
    #     #                                         is_last_time_step=is_last_time_step,
    #     #                                         compress_level=compress_level, verbose_plot=verbose_plot)
    #
    #     # def get_func(pde_boltz, method):
    #     #     if method[:2] == 'SL':
    #     #         # assert all([v_ax.basis.type == BasisType.SPATIAL for v_ax in self.sys_fe.v_axes]), \
    #     #         #     'SL time evolution for spatial basis only'
    #     #         # assert (len(self.sys_fe.x_axes) == 1), 'SL method only valid for 1DnV'
    #     #
    #     #         sl_func = pde_boltz.semilagrangian
    #     #         kwargs = {'method': method}
    #     #     elif method[:3] == 'mac':
    #     #         sl_func = pde_boltz.maccormack
    #     #         kwargs = {'coeff_is_constant': True}
    #     #     elif method[:3] == 'lax':
    #     #         sl_func = pde_boltz.lax_wendroff
    #     #         kwargs = {'coeff_is_constant': True}
    #     #     else:
    #     #         raise NotImplementedError(f'{method} not yet implemented')
    #     #     return sl_func, kwargs
    #     #
    #     # adv_f_func_fe = get_func(state0.sys_fe, method_f)
    #     # adv_f_func_fi = get_func(state0.sys_fi, method_f)
    #     # adv_v_func_fe = get_func(state0.sys_fe, method_v)
    #     # adv_v_func_fi = get_func(state0.sys_fi, method_v)
    #
    #     if is_first_time_step:
    #         ## evolve force advection dt/2
    #         state0.sys_fe.force_term = state0.compute_force_term(is_ion=False, compress_level1=comp3, compress_level2=0)
    #         state0.sys_fe.get_force_advection(dt / 2, inplace=True, method=method_f,
    #                                           compress=comp2, compress1=comp4, compress2=comp5,
    #                                           verbose_plot=verbose_plot)
    #         state0.sys_fi.force_term = state0.compute_force_term(is_ion=True, compress_level1=comp3, compress_level2=0)
    #         state0.sys_fi.get_force_advection(dt / 2, inplace=True, method=method_f,
    #                                           compress=comp2, compress1=comp4, compress2=comp5,
    #                                           verbose_plot=verbose_plot)
    #
    #     ## evolve velocity advection dt
    #     state0.sys_fe.get_vel_advection(dt, inplace=True, method=method_v, compress=comp2, compress1=comp4,
    #                                     compress2=comp5, verbose_plot=verbose_plot)
    #     state0.sys_fi.get_vel_advection(dt, inplace=True, method=method_v, compress=comp2, compress1=comp4,
    #                                     compress2=comp5, verbose_plot=verbose_plot)
    #
    #     ## update V
    #     if np.abs(state0.matl_params.e) > 0:
    #         state0.update_V(inplace=True, compress=compress_level, compress1=comp2)
    #
    #     if verbose_plot:
    #         V_data = state0.V.get_comp_data()
    #         plt.figure()
    #         plt.plot(V_data)
    #         plt.title('V')
    #         plt.show()
    #
    #     ## evolve force advection dt/2
    #     dt_ = dt / 2 if is_last_time_step else dt
    #     state0.sys_fe.force_term = state0.compute_force_term(is_ion=False, compress_level1=comp3, compress_level2=0)
    #     state0.sys_fe.get_force_advection(dt_, inplace=True, method=method_f,
    #                                       compress=comp2, compress1=comp4,
    #                                       compress2=comp5, verbose_plot=verbose_plot)
    #     state0.sys_fi.force_term = state0.compute_force_term(is_ion=True, compress_level1=comp3, compress_level2=0)
    #     state0.sys_fi.get_force_advection(dt_, inplace=True, method=method_f,
    #                                       compress=comp2, compress1=comp4,
    #                                       compress2=comp5, verbose_plot=verbose_plot)
    #
    #     ## include collisions
    #     if self.collision.coll_type is not None:
    #         raise NotImplementedError
    #         deriv_coll_e = state0.get_collision_term(v_grads=v_grads, v_axes=v_axes, is_ion=False)
    #         deriv_coll_i = state0.get_collision_term(v_grads=v_grads, v_axes=v_axes, is_ion=True)
    #         dFdt_coll = state0.create_like(deriv_coll_e, deriv_coll_i, recalc=False)
    #         state0 = state0.euler(dt, deriv0=dFdt_coll, compress_level=comp2)
    #
    #     state0.time = self.time + dt if self.time is not None else None
    #
    #     if compress_level:
    #         state0.compress(compress_level=compress_level, verbose=False)
    #
    #     if state0.do_normalization:
    #         state0.normalize()
    #
    #     return state0


    # @profile
    def euler(self, dt: Numeric, deriv0: Optional['VlasovPoisson'] = None, inplace=False, do_update_V=True,
              compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0, verbose_plot: bool = False,
              do_x_advection=True, do_v_advection=True) \
            -> 'VlasovPoisson':
        """ perform explicit Euler time evolution
            compress:  final compression of final state, updateV
            compress1: compression of derivative
            compress2: compress1 of calculate derivative, updateV
        """
        state1 = super().euler(dt, deriv0=deriv0, inplace=inplace, compress_level=compress_level,
                               compress_level1=compress_level1, compress_level2=compress_level2,
                               do_x_advection=do_x_advection, do_v_advection=do_v_advection,
                               verbose_plot=verbose_plot)

        if np.abs(state1.elc_params.e) > 0 and do_update_V:
            # old_V = state1.V.copy()
            # print('state bond V', compress_level, state1.V.max_bond())
            # state1.V = self.V
            state1.update_EM_sys(dt, inplace=True, compress=compress_level, compress1=compress_level2)
            # print('V diff', helper.distance(old_V.component.data, state1.V.component.data))
            # print('state bond comp', compress_level, state1.V.max_bond())

        state1.time = self.time + dt if self.time is not None else None

        return state1


    #############################

    # @profile
    def compute_force_term(self, is_ion=False, compress_level=1, compress_level1=0, compress_level2=0,
                           verbose_plot=False, background_force=True, internal_force=True, **kwargs):
        """ compute EM force:  q/m (E + v x B)
        """
        spec_params = self.ion_params if is_ion else self.elc_params

        em_term, em_term_0, em_term_1 = None, None, None
        if np.abs(spec_params.Z) > 0:

            # if self._total_E is not None and background_force and internal_force:
            #     em_term = self.total_E
            # else:

            if background_force:
                em_term_0 = self.background_E0

            if internal_force:
                em_term_1 = self.V.gradient(compress_level=0)  # E = -grad(V)
                em_term_1.scalar_multiply(-1, inplace=True)

            if em_term_0 is not None:
                em_term = em_term_0.add(em_term_1)
            else:
                em_term = em_term_1

            self._total_E = em_term

        if em_term is not None:
            em_term = em_term.scalar_multiply(spec_params.e * spec_params.Z / spec_params.mass)
            if self.compress_F:
                em_term.compress(compress_opts=self.compress_F_opts, inplace=True)

            # plt.figure()
            # X = self.coords_x.coords[0]
            # # plt.plot(self.V.get_comp_data(), label='V')
            # plt.plot(em_term.get_comp_data(X), label='E')
            # plt.legend()
            # plt.show()

        # print('EM TERM', em_term.components)
        return em_term


    def compute_force_term_bg_SL(self, dt, is_ion=False, sl_order=DEFAULT_SL_ORDER, compress_level=1, compress_level1=0,
                                 compress_level2=0, verbose_plot=False):
        """ assumes field is constant in space
        """
        sys_f = self.sys_fi if is_ion else self.sys_fe
        spec_params = self.ion_params if is_ion else self.elc_params
        coords_v = self.coords_vi if is_ion else self.coords_ve
        em_term = self.background_E0

        em_SL_dict = {}
        mult_const = spec_params.e * spec_params.Z / spec_params.mass

        print('compute ES bg SL')

        if em_term is not None:

            # if len(em_SL_dict) == 0:  ## compute these terms
                # em_SL_dict = {}
            for C in self.background_E0.componentIDs:
                EC = self.background_E0[C]
                # x_ax = self.coords_x.get_axis(C.type)
                v0_ax = coords_v.get_axis(C.type)
                CV = v0_ax.coordinate
                # print('advec ax', v0_ax, C)
                # print('x_ax', x_ax, self.coords_x.axes)
                if EC.data is not None:
                    if EC.is_constant:
                        select_gtn = EC.grid.get_select_elems_mps([0])
                        E0_val = EC.ovlp(select_gtn) * mult_const
                        em_SL_dict[CV] = helper_sl.get_cell_data(dt, v0_ax.dx, E0_val, sl_order=sl_order)
                        # print('em sl dict', E0_val, em_SL_dict[C].cells, E0_val * dt / v0_ax.dx)
                    else:
                        E0_data = self.background_E0.get_comp_data(C) * mult_const
                        em_SL_dict[CV] = helper_sl.get_cell_data(dt, v0_ax.dx, E0_data, sl_order=sl_order)
                        # print('em sl dict', E0_data[0], em_SL_dict[C].cells)
                # sys_f.background_force_SL = em_SL_dict

            sys_f.force_term_bg_SL = em_SL_dict
            return em_SL_dict
        else:
            sys_f.force_term_bg_SL = {}
            return {}


    # @profile
    def update_V(self,inplace=False,compress=1,compress1=4) -> 'ScalarField':
        """ div E = div^2 V = 4*pi*rho
            compress:  compression of V
            compress1: compression of terms to create V
        """
        x_axes = self.coords_x.axes
        # ve_axes = self.coords_ve.axes
        # vi_axes = self.coords_vi.axes

        new_sys = self if inplace else self.copy()

        ## particles are not charged
        if new_sys.elc_params.e == 0 or (np.abs(new_sys.elc_params.Z) == 0 and np.abs(new_sys.ion_params.Z) == 0):
            new_sys.V = None
            return new_sys

        ## get Field object after integrating over velocity space
        # print('compress1', compress1)
        charge_density = new_sys.compute_charge_density(compress=compress1, compress1=5)
        if self.matl_params.is_cgs:
            charge_density.scalar_multiply(4 * np.pi, inplace=True)
        else:
            charge_density.scalar_multiply(1 / self.matl_params.eps0, inplace=True)

        # print('charge density', charge_density)
        if new_sys.background_E0 is not None:
            if new_sys.divE0 is None:
                new_sys.divE0 = new_sys.background_E0.divergence(new_sys.coords_x, compress_level=compress1)
            if self.matl_params.is_cgs:
                divE0 = new_sys.divE0.scalar_multiply(-4 * np.pi, inplace=False)
            else:
                divE0 = new_sys.divE0.scalar_multiply(-1 / new_sys.elc_params.eps0, inplace=False)
            charge_density.add(divE0, inplace=True)

        charge_density.scalar_multiply(-1, inplace=True)

        bc_values = new_sys.potential_boundary_conditions[1]
        if bc_values is not None:  # ie. 0
            if bc_values.grid.ndim < charge_density.grid.ndim:
                bc_values = bc_values.pad_to_new_grid(new_grid=charge_density.grid)
            soln_gtn = charge_density.component.add(bc_values[0], compress=False)
        else:
            soln_gtn = charge_density.component

        # print('update V', compress, new_sys.V.compress_config[compress])
        # print('charge density', charge_density.max_bond())
        # soln_gtn.compress(inplace=True, compress_opts=new_sys.V.compress_config[compress])
        # print('CHARGE DENSITY', charge_density.max_bond())

        do_dmrg = False  # True

        if False:  # do_dmrg:
            if new_sys.laplacian_mpo is None:
                ax_deriv_configs = new_sys.V.component.ax_deriv_configs if new_sys.V is not None \
                    else {ax: DerivativeConfiguration() for ax in x_axes}
                laplacian_mpo = new_sys.V.grid.laplacian_mpo(ax_deriv_configs=ax_deriv_configs)
                ## boundary conditions to make laplacian full rank
                # bc_mpo = new_sys.V.grid.get_select_elems_mpo([0,]*new_sys.V.grid.ndim)
                num_pts = new_sys.V.grid.npts
                bc_mat = np.zeros((num_pts, num_pts))
                bc_mat[0,:] = 1.0/num_pts
                gr_shape = [ax.npts for ax in new_sys.V.grid.axes]
                # print('gr shape', gr_shape)
                bc_mat = bc_mat.reshape(*gr_shape, *gr_shape)
                bc_mpo = new_sys.V.grid.map_operator_to_mpo(bc_mat)

                laplacian_mpo = laplacian_mpo.add(bc_mpo, inplace=False)
                # bc_vec = new_sys.V.grid.get_select_elems_mps([0,]*new_sys.V.grid.ndim)
                new_sys.laplacian_mpo = laplacian_mpo
                print('laplacian', laplacian_mpo.max_bond())
            else:
                laplacian_mpo = new_sys.laplacian_mpo


            compress_opts = new_sys.V.compress_config.get_compress_opts(compress)
            # new_V_mps = new_sys.V.component.data
            # solver = helper_dmrg.LinearSolver(new_V_mps, targets=[soln_gtn.data], operators=[laplacian_mpo.data],
            #                                   max_iter=100, solve_type='CGDx', max_bond=compress_opts['max_bond'],)
            # opt_ket, err, is_conv  = solver.solve(2, init_direction=-1, conv_tol=1.0e-4)

            opt_ket, err, is_conv = helper_dmrg.mps_conjugate_gradient_descent_v2(laplacian_mpo.data, b = soln_gtn.data,
                                                                                  # x = new_sys.V.component.data,
                                                                                  max_bond=compress_opts['max_bond'])
            # ## starting with initial guess causes any noise to accumulate?

            new_V_mps = new_sys.V.grid.make_gridTN(opt_ket, ax_deriv_configs=soln_gtn.ax_deriv_configs)

            if err > 0.01:  # not is_conv:
                plt.figure()
                plt.plot(soln_gtn.get_data(), label='charge')
                b_opt = new_V_mps.apply(laplacian_mpo)
                plt.plot(b_opt.get_data(),'--', label='Ax')
                plt.legend()
                plt.show()

        else:
            compress_type = new_sys.V.compress_config.compress_type
            ax_deriv_configs = new_sys.V.component.ax_deriv_configs if new_sys.V is not None \
                                    else {ax: DerivativeConfiguration() for ax in x_axes}

            if compress_type is CompressType.SVD or do_dmrg:
                if new_sys.inv_laplacian_mpo is None:
                    inv_laplacian_mpo = new_sys.V.grid.inverse_laplacian_mpo(ax_deriv_configs=ax_deriv_configs,
                                                                    boundary_conditions=self.potential_boundary_conditions[0],
                                                                    compress_opts={'cutoff':1.0e-20,'cutoff_mode':'rsum2'})
                    new_sys.inv_laplacian_mpo = inv_laplacian_mpo
                    print('inv laplacian', inv_laplacian_mpo.max_bond())
                else:
                    inv_laplacian_mpo = new_sys.inv_laplacian_mpo

                new_V_mps = soln_gtn.apply(inv_laplacian_mpo, zipup=self.zipup, compress=compress,
                                           compress_opts=new_sys.V.compress_config[compress],
                                           sub_compress_opts=new_sys.V.compress_config.get_all_sub_compress_opts(compress))
                # print('new V', compress, new_sys.V.compress_config[compress], new_V_mps.max_bond())
            else:
                if new_sys.inv_laplacian_mpo is None:
                    bc = self.potential_boundary_conditions[0]
                    laplacian_mpo = new_sys.V.grid.laplacian_mpo(ax_deriv_configs=ax_deriv_configs)
                    if bc is not None:
                        if isinstance(bc, np.ndarray):
                            num_bc = len(bc)
                            tens = np.zeros((new_sys.V.grid.npts,) * 2)
                            tens[:num_bc, :] += bc
                            tens = tens.reshape(*([ax.npts for ax in new_sys.V.grid.axes]*2))
                            bc = new_sys.V.grid.map_operator_to_mpo(tens)
                            laplacian_mpo = laplacian_mpo.add(bc, compress=True)
                        elif isinstance(bc, laplacian_mpo.__class__) and bc.data_type is DataType.MPO:
                            laplacian_mpo = laplacian_mpo.add(bc, compress=True)
                        else:
                            raise NotImplementedError
                        new_sys.inv_laplacian_mpo = laplacian_mpo
                else:
                    laplacian_mpo = new_sys.inv_laplacian_mpo

                old_V = new_sys.V.component
                # old_V_mps = qtn.MPS_rand_state(int(np.sum([ax.L for ax in new_sys.V.grid.axes])),
                #                                new_sys.V.compress_config[compress]['max_bond'])
                # old_V = new_sys.V.grid.make_gridTN(old_V_mps)
                # print('old V', old_V)
                new_V_mps = soln_gtn.solve(laplacian_mpo, CompressType.DMRG, inplace=False, init_guess=old_V,
                                           compress_opts=new_sys.V.compress_config[compress])


        # V_ = V.copy()
        new_sys.V.component = new_V_mps  ## update potential field (which is a scalar field)
        new_sys._total_E = None

        # plt.figure()
        # plt.plot(charge_density.get_comp_data(), label='charge')
        # plt.plot(new_sys.V.get_comp_data(), label='new V')
        # plt.title('update V')
        # plt.legend()
        # plt.show()

        return new_sys


    def update_EM_sys(self, dt, inplace=False, compress=1, compress1=0, compress2=0, **kwargs):
        """ update V (called by split step)
        """
        return self.update_V(inplace=inplace, compress=compress, compress1=compress1 )



