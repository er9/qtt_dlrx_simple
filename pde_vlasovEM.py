from setup_.configs import *
import matplotlib.pyplot as plt
import helper_sl as helper_sl
from grids_composite import CompositeGrid
from grid_comb import GridsComb
from field import Field, ScalarField
from pde_system import PDE_system
from pde_EM import Maxwell
from pde_vlasov import Vlasov
from pde_boltzmann import Boltzmann

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from axis import Axis


class VlasovMaxwell(Vlasov):
    """ df/dt + v grad(f) + q/m (E + v x B) grad_v(f) = 0

        - EM method 1:    hyperbolic Maxwell's equations (Maxwell's eq with equation cleaning)
                          http://ammar-hakim.org/sj/je/je6/je6-maxwell-solvers.html
                          http://ammar-hakim.org/maxwell-eigensystem.html

            dB/dt + curl(E) + gamma grad(psi) = 0
            eps0 mu0 dE/dt - curl(B) + chi grad(phi) = 0
            1/chi dphi/dt + div(E) = rho/eps0
            eps0 mu0 /gamma dpsi/dt + div(B) = 0

            choose chi = gamma = 1

        - EM method 2:   Yee cell
            to be implemented. 
    """

    def __init__(self,
                 sys_fe: Optional[Union['ScalarField', 'Boltzmann']],
                 sys_fi: Optional[Union['ScalarField', 'Boltzmann']],
                 EM_sys: Optional['Maxwell'] = None,
                 grid_X: Optional['Grid'] = None,
                 coords_x: Optional['CoordinateSystem'] = None,
                 coords_ve: Optional['CoordinateSystem'] = None,
                 coords_vi: Optional['CoordinateSystem'] = None,
                 elc_params: Optional['SpeciesConfiguration'] = None,
                 ion_params: Optional['SpeciesConfiguration'] = None,
                 f0_gradv_e: Optional['Field'] = None,
                 f0_gradv_i: Optional['Field'] = None,
                 background_fe0: Optional[Union['ScalarField', 'Boltzmann']] = None,
                 background_fi0: Optional[Union['ScalarField', 'Boltzmann']] = None,
                 evolve_ion=True, evolve_EM=True, normalize=True,
                 upwind=False, zipup=False, conservative=True,
                 te_order=2, compress_levels=None,
                 ):
        """ dist_e:  distribution of electrons (scalar Field obj). generally on x,v grid
            dist_i:  distribution of ions (scalar Field obj). generally on x,v grid
            potential:  electric potential (scalar Field obj). generally on x grid
            velocity_grid: velocity grid (vector Field obj) with components vx, vy, ...
                           generally on v grid
            f0_gradv:   grad_v(f0) (vector Field obj) if doing linearized vlasov. 
            x_axes:  GRID axes (0,...,self.ndim-1) corresponding to spatial positions in fe, fi
        """

        grid_X = grid_X if EM_sys is None else EM_sys.grid_X

        super().__init__(sys_fe, sys_fi, grid_X=grid_X,
                         coords_x=coords_x, coords_ve=coords_ve, coords_vi=coords_vi,
                         elc_params=elc_params, ion_params=ion_params,
                         f0_gradv_e=f0_gradv_e, f0_gradv_i=f0_gradv_i,
                         background_fe0=background_fe0, background_fi0=background_fi0,
                         normalize=normalize, upwind=upwind, zipup=zipup, evolve_ion=evolve_ion,
                         conservative=conservative, te_order=te_order, compress_levels=compress_levels)

        self.EM_sys = EM_sys
        self.evolve_EM = evolve_EM

        # self._x_advection = True
        # self._v_advection = True

        # self.compress_F = True
        # self.compress_F_opts = {'max_bond':None, 'cutoff':CUTOFF, 'cutoff_mode':CUTOFF_MODE}
        #
        # self.collision = CollisionConfiguration(None)
        # self.dist0_e = None
        # self.dist0_i = None

    @property
    def EM_sys(self) -> Optional['Maxwell']:
        return self._EM_sys

    @EM_sys.setter
    def EM_sys(self, new_EM_sys: Optional['Maxwell']):

        self._EM_sys = Maxwell(None, None, grid_X=self.grid_X) if new_EM_sys is None else new_EM_sys

        if new_EM_sys is not None:
            assert (new_EM_sys.grid_X == self.grid_X), 'Maxwell grid_X does not match self'

            self.names['E'] = self._EM_sys.names['E'] if self._EM_sys.names['E'] is not None else 'E'
            self.names['B'] = self._EM_sys.names['B'] if self._EM_sys.names['B'] is not None else 'B'
            self.names['phi'] = self._EM_sys.names['phi'] if self._EM_sys.names['phi'] is not None else 'phi'
            self.names['psi'] = self._EM_sys.names['psi'] if self._EM_sys.names['psi'] is not None else 'psi'
            self.field_names = self.field_names[:2] + [self.names['E'], self.names['B'],
                                                       self.names['phi'], self.names['psi']]

            if self.sys_fe is not None:
                if self.background_fe is not None:
                    self.sys_fe.background_force_neg = self.compute_force_term(background_force=True)

                try:
                    self.sys_fe.background_force_neg.scalar_multiply(-1, inplace=True)
                except AttributeError:
                    pass

            if self.sys_fi is not None:
                if self.background_fi is not None:
                    self.sys_fi.background_force_neg = self.compute_force_term(is_ion=True, background_force=True)

                try:
                    self.sys_fi.background_force_neg.scalar_multiply(-1, inplace=True)
                except AttributeError:
                    pass

        else:
            if self.sys_fe is not None:
                self.sys_fe.background_force_neg = None
            if self.sys_fi is not None:
                self.sys_fi.background_force_neg = None

    @property
    def E(self) -> Optional['Field']:
        return self.EM_sys.E

    @E.setter
    def E(self, new_field: 'Field'):
        self.EM_sys.E = new_field

    @property
    def B(self) -> Optional['Field']:
        return self.EM_sys.B

    @B.setter
    def B(self, new_field: 'Field'):
        self.EM_sys.B = new_field

    @property
    def tot_B(self) -> Optional['Field']:
        return self.EM_sys.tot_B

    @property
    def tot_E(self) -> Optional['Field']:
        return self.EM_sys.tot_E

    @property
    def phi(self) -> Optional['ScalarField']:
        return self.EM_sys.phi

    @phi.setter
    def phi(self, new_field: 'ScalarField'):
        self.EM_sys.phi = new_field

    @property
    def psi(self) -> Optional['ScalarField']:
        return self.EM_sys.psi

    @psi.setter
    def psi(self, new_field: 'ScalarField'):
        self.EM_sys.psi = new_field

    def get_field(self, field_name):
        """ return ith component of the field. for convenience
                """
        if field_name == self.names['E']:
            return self.E
        elif field_name == self.names['B']:
            return self.B
        elif field_name == self.names['phi']:
            return self.phi
        elif field_name == self.names['psi']:
            return self.psi
        else:
            return super().get_field(field_name)

    def set_field(self, field_name, new_field):
        """ set ith component of the field. for convenience
            new_field is a Field object
        """
        assert (isinstance(new_field, Field) or new_field is None), 'new_field must be Field object or None'

        if field_name == self.names['E']:
            self.E = new_field
        elif field_name == self.names['B']:
            self.B = new_field
        elif field_name == self.names['phi']:
            self.phi = new_field
        elif field_name == self.names['psi']:
            self.psi = new_field
        else:
            self._fields[field_name] = new_field

    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 6:
            new_fields = new_fields + (None,) * (6 - len(new_fields))

        dist_e, dist_i, E, B, phi, psi = new_fields[:6]

        new_system = super().create_like(dist_e, dist_i, recalc=recalc, deep=deep)

        EM_sys = self.EM_sys.create_like(E, B, phi, psi, recalc=recalc)
        new_system._EM_sys = EM_sys
        new_system.field_names = self.field_names
        new_system.names = self.names
        new_system.evolve_EM = self.evolve_EM

        if self.sys_fe is not None and new_system.sys_fe is not None:
            new_system.sys_fe.background_force_neg = self.sys_fe.background_force_neg
        if self.sys_fi is not None and new_system.sys_fi is not None:
            new_system.sys_fi.background_force_neg = self.sys_fi.background_force_neg

        return new_system

    def copy(self):
        new_system = super().copy()
        new_system.EM_sys = self.EM_sys.copy()
        ## update EM saved fields?
        return new_system

    def get_time_derivative_operator(self, compress=1, is_ion=False):
        """ dF/dt = G[f(t)].  Returns G
        """
        raise NotImplementedError

    # @profile
    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, method_v: Optional['Field'] = None,
                   method_f=None, method_EM=None, is_first_time_step=False, is_last_time_step=False, inplace=False, compress_level=1,
                   verbose_plot=False) -> 'VlasovMaxwell':
        """ take split step for EM system and ion,electron advection terms
            each step is just an Euler update
            following "Hamiltonian Splitting for Vlasov-Maxwell" by Crouseilles
            second order:
            HE: (dt/2)  HE: evolve df/dt with E; dB/dt
            HB: (dt/2)  HB: evolve df/dt with B; dE/dt without current
            Hf: (dt)    Hf: evolve df/dt x-advection; dE/dt with current
            HB: (dt/2)
            HE: (dt/2)
            (Crouseilles did some SL/exact time evolution though).
        """
        state0 = self if inplace else self.copy()
        te_order = 1 if self.te_order == 31 else int(str(self.te_order)[-1])
        te_order_EM = 1 # if self.te_order == 31 else int(str(self.te_order_EM)[-1])
        ## 1: euler
        ## 0: exact/exponential
        ## ??: CN, backwards Euler
        method_f = self.get_te_method(te_order) if method_f is None else method_f
        method_v = self.get_te_method(te_order) if method_v is None else method_v
        method_EM = self.get_te_method(te_order_EM) if method_EM is None else method_EM

        print('method f', method_f)
        print('method v', method_v)
        print('method EM', method_EM)

        state0.sys_fe.te_order = te_order
        state0.sys_fi.te_order = te_order

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        def evolve_HE(dt_):
            print('evolve HE', dt_)
            E_force_e = self.compute_force_term(include_B=False, is_ion=False)
            state0.sys_fe.force_term = E_force_e
            state0.sys_fe.get_force_advection(dt_, method=method_f, inplace=True)

            if self.evolve_ion:
                E_force_i = self.compute_force_term(include_B=False, is_ion=True)
                state0.sys_fi.force_term = E_force_i
                state0.sys_fi.get_force_advection(dt_, method=method_f, inplace=True)

            if self.evolve_EM:
                state0.EM_sys.evolve_B(dt_, method=method_EM, inplace=True)  ## default to FDTD
            return

        def evolve_HB(dt_):
            print('evolve HB', dt_)
            E_force_e = self.compute_force_term(include_E=False, is_ion=False)
            state0.sys_fe.force_term = E_force_e
            state0.sys_fe.get_force_advection(dt_, method=method_f, inplace=True)

            if self.evolve_ion:
                E_force_i = self.compute_force_term(include_E=False, is_ion=True)
                state0.sys_fi.force_term = E_force_i
                state0.sys_fi.get_force_advection(dt_, method=method_f, inplace=True)

            if self.evolve_EM:
                state0.EM_sys.current_density = None
                state0.EM_sys.evolve_E(dt_, method=method_EM, inplace=True)  ## default to FDTD
            return

        def evolve_Hf(dt_):
            print('evolve Hf', dt_)
            state0.sys_fe.get_vel_advection(dt_, method=method_v, inplace=True)
            if self.evolve_ion:
                state0.sys_fi.get_vel_advection(dt_, method=method_v, inplace=True)
            if self.evolve_EM:
                print('evolve J')
                state0.EM_sys.current_density = state0.compute_current()
                state0.EM_sys.evolve_E(dt_, method=method_EM, inplace=True, current_only=True)
            return

        if self.verbose_plot:
            j_old = state0.compute_current(compress=comp2).copy()

        evolve_HB(dt/2)
        evolve_HE(dt/2)
        evolve_Hf(dt)
        evolve_HE(dt/2)
        evolve_HB(dt/2)

        if self.verbose_plot:
            ## current at t + 1
            j_new = state0.compute_current(compress=comp2)

            for C in self.coords_x.coords:
                plt.figure()
                j_old_data = j_old.get_comp_data(C)
                j_new_data = j_new.get_comp_data(C)
                if j_old_data is not None:
                    plt.plot(np.real(j_old_data), '--', label='old j')
                    plt.plot(np.imag(j_old_data), '--', label='old j im')
                    plt.plot(np.real(j_new_data), ':', label='new j')
                    plt.plot(np.imag(j_new_data), ':', label='new j im')
                    plt.plot(-np.real( j_old_data - j_new_data)/dt, label='dj/dt')
                    plt.plot(-np.imag( j_old_data - j_new_data)/dt, label='dj/dt im')
                plt.legend()
                plt.title(f'J after df/dt {C}')
            plt.show()


        ## dist f -> t + 1
        state0.time = self.time + dt if self.time is not None else None

        ## include collisions (cross ion/elc collisions)
        if self.collision.coll_type is not None:
            # raise NotImplementedError
            deriv_coll_e, deriv_coll_i = state0.get_collision_term(v_grads=None)
            dFdt_coll = self.create_like(deriv_coll_e, deriv_coll_i, recalc=False)
            state0 = state0.euler(dt, deriv0=dFdt_coll, compress_level=comp2)

        return state0

    def split_step_old(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, method_v: Optional['Field'] = None,
                       method_f=None, is_first_time_step=False, is_last_time_step=False, inplace=False, compress_level=1,
                       verbose_plot=False) -> 'VlasovMaxwell':
        """ take split step for EM system and ion,electron advection terms
            each step is just an Euler update
        """
        state0 = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        # print('methods', method_f, method_v)

        if False:  # self.EM_sys.is_yee:

            # print('here is yee', self.EM_sys.is_yee)

            ## evolve force advection dt/2
            state0.get_force_advection(dt / 2, inplace=True, method=method_f, split_order=1,
                                       verbose_plot=verbose_plot, compress=comp1, compress1=comp4, compress2=comp5, )

            ## evolve velocity advection dt
            state0.get_vel_advection(dt / 2, inplace=True, method=method_v, split_order=1,
                                     verbose_plot=verbose_plot, compress=comp1, compress1=comp4, compress2=comp5, )

            ## at initialization, need to evolve EM_sys with dt/2
            if self.evolve_EM and is_first_time_step:
                # print('FIRST TIME STEP')
                ## evolves B -> t = 1/2;
                state0.EM_sys = state0.EM_sys.evolve_B(dt / 2, inplace=True, method='rk4',
                                                       compress=comp1, compress1=comp4, compress2=comp5)
            else:
                ## n - 1/2 -> n + 1/2
                state0.EM_sys = state0.EM_sys.evolve_B(dt, inplace=True,
                                                       compress=comp1, compress1=comp4, compress2=comp5)

            ## current at t = n + 1/2
            print('compute current')
            j = state0.compute_current(compress=comp2)
            for compID, comp in j.components.items():
                comp.ax_deriv_configs = state0.EM_sys.E[compID].ax_deriv_configs
            state0.EM_sys.current_density = j

            ## t: n -> n + 1
            state0.EM_sys = state0.EM_sys.evolve_E(dt, inplace=True,
                                                   compress=comp1, compress1=comp4, compress2=comp5)

            ## evolve velocity advection dt
            state0.get_vel_advection(dt / 2, inplace=True, method=method_v, split_order=-1,
                                     verbose_plot=verbose_plot, compress=comp1, compress1=comp4, compress2=comp5, )

            ## evolve force advection dt/2
            state0.get_force_advection(dt / 2, inplace=True, method=method_f, split_order=-1,
                                       verbose_plot=verbose_plot, compress=comp1, compress1=comp4, compress2=comp5, )

        else:

            if self.EM_sys.verbose_plot:
                j_old = state0.compute_current(compress=comp2).copy()

            ## evolve force advection dt/2
            print('force advec')
            state0.get_force_advection(dt / 2, inplace=True, method=method_f, verbose_plot=verbose_plot,
                                       split_order=1, compress=comp1, compress1=comp4, compress2=comp5, )

            print('vel advec')
            state0.get_vel_advection(dt, inplace=True, method=method_v, verbose_plot=verbose_plot,
                                     compress=comp1, compress1=comp4, compress2=comp5, )

            ## evolve force advection dt/2
            print('force advec')
            state0.get_force_advection(dt / 2, inplace=True, method=method_f, verbose_plot=verbose_plot,
                                       split_order=-1, compress=comp1, compress1=comp4, compress2=comp5, )

            ## include collisions (cross ion/elc collisions)
            if self.collision.coll_type is not None:
                # raise NotImplementedError
                deriv_coll_e, deriv_coll_i = state0.get_collision_term(v_grads=None)
                dFdt_coll = self.create_like(deriv_coll_e, deriv_coll_i, recalc=False)
                state0 = state0.euler(dt, deriv0=dFdt_coll, compress_level=comp2)

            ## update V or EM sys;  B -> t+3/2, E -> t+1
            if self.evolve_EM:
                ## current at t + 1

                # print('self.EM_sys verbose_plot', self.EM_sys.verbose_plot)
                if self.EM_sys.verbose_plot:
                    j_new = state0.compute_current(compress=comp2)
                    for C in self.coords_x.coords:
                        plt.figure()
                        j_old_data = j_old.get_comp_data(C)
                        j_new_data = j_new.get_comp_data(C)
                        if j_old_data is not None:
                            plt.plot(np.real(j_old_data), '--', label='old j')
                            plt.plot(np.imag(j_old_data), '--', label='old j im')
                            plt.plot(np.real(j_new_data), ':', label='new j')
                            plt.plot(np.imag(j_new_data), ':', label='new j im')
                            plt.plot(-np.real(j_old_data - j_new_data) / dt, label='dj/dt')
                            plt.plot(-np.imag(j_old_data - j_new_data) / dt, label='dj/dt im')
                        plt.legend()
                        plt.title(f'J after df/dt {C}')
                    plt.show()

                state0.EM_sys.current_density = state0.compute_current(compress=comp2)
                state0.update_EM_sys(dt, inplace=True, is_first_time_step=is_first_time_step,
                                     is_last_time_step=is_last_time_step,
                                     compute_current=False, compress=comp1, compress1=comp2)

        ## dist f -> t + 1
        state0.time = self.time + dt if self.time is not None else None

        # plt.figure()
        # for C in state0.E.componentIDs:
        #     if state0.E[C].data is not None:
        #         plt.plot(state0.E[C].get_data(), label=f'E {C}')
        # plt.legend()
        #
        # plt.figure()
        # for C in state0.B.componentIDs:
        #     if state0.B[C].data is not None:
        #         plt.plot(state0.B[C].get_data(), label=f'B {C}')
        # plt.legend()
        #
        # plt.show()

        return state0

    def compute_E_from_charge_density(self, compress=1, compress_opts=None, solve_type=CompressType.DMRG,
                                      potential_boundary_conditions: tuple = None, background_E0=None, divE0=None
                                      ) -> 'Field':

        if potential_boundary_conditions is None:

            ## sum of all points = 1
            mpo_dicts = {}
            for ax in self.grid_X.axes:
                mat = np.zeros((ax.npts, ax.npts))
                mat[0, :] = 1.
                mpo_dicts[ax] = mat
            bc00 = self.grid_X.make_mpo_ndim(mpo_dicts)
            bc00.scalar_multiply(1. / np.prod([ax.dx for ax in self.grid_X.axes]), inplace=True)

            # ## select first element
            # bc00 = self.grid_X.get_select_elems_mpo([0, ] * self.grid_X.ndim)

            potential_boundary_conditions = (bc00, None)

        ## particles are not charged
        if self.elc_params.e == 0 or (np.abs(self.elc_params.Z) == 0 and np.abs(self.ion_params.Z) == 0):
            return None

        ## get Field object after integrating over velocity space
        # print('compress1', compress1)
        charge_density = self.compute_charge_density(compress=1, compress1=5)
        if self.matl_params.is_cgs:
            charge_density.scalar_multiply(4 * np.pi, inplace=True)
        else:
            charge_density.scalar_multiply(1 / self.matl_params.eps0, inplace=True)

        # print('charge density', charge_density)
        if background_E0 is not None:
            if divE0 is None:
                divE0 = background_E0.divergence(self.coords_x, compress_level=compress)
            if self.matl_params.is_cgs:
                divE0 = divE0.scalar_multiply(-4 * np.pi, inplace=False)
            else:
                divE0 = divE0.scalar_multiply(-1 / self.elc_params.eps0, inplace=False)
            charge_density.add(divE0, inplace=True)

        charge_density.scalar_multiply(-1, inplace=True)

        bc_values = potential_boundary_conditions[1]
        if bc_values is not None:  # ie. 0
            if bc_values.grid.ndim < charge_density.grid.ndim:
                bc_values = bc_values.pad_to_new_grid(new_grid=charge_density.grid)
            soln_gtn = charge_density.component.add(bc_values[0], compress=False)
        else:
            soln_gtn = charge_density.component

        ## solve lapl(V) = rho  (+ boundary conditions)
        ax_deriv_configs = {}
        for C in self.coords_x.coords:
            C_ax = self.coords_x.get_axis(C.type)
            if C_ax is not None:
                EC = self.E[C]
                ax_deriv_configs[C_ax] = EC.ax_deriv_configs[C_ax] if EC is not None else DerivativeConfiguration()

        if solve_type is CompressType.SVD:
            try:
                inv_laplacian_mpo = self.inv_laplacian_mpo
            except AttributeError:
                inv_laplacian_mpo = self.grid_X.inverse_laplacian_mpo(ax_deriv_configs=ax_deriv_configs,
                                                                      boundary_conditions=
                                                                      potential_boundary_conditions[0],
                                                                      compress_opts={'cutoff': 1.0e-20,
                                                                                     'cutoff_mode': 'rsum2'})
                self.inv_laplacian_mpo = inv_laplacian_mpo

            print('inv laplacian', inv_laplacian_mpo.max_bond())
            new_V_mps = soln_gtn.apply(inv_laplacian_mpo, zipup=self.zipup, compress=compress,
                                       compress_opts=compress_opts)
            # print('new V', compress, new_sys.V.compress_config[compress], new_V_mps.max_bond())
        else:
            try:
                laplacian_mpo = self.inv_laplacian_mpo
            except AttributeError:
                bc = potential_boundary_conditions[0]
                laplacian_mpo = self.grid_X.laplacian_mpo(ax_deriv_configs=ax_deriv_configs)
                if bc is not None:
                    if isinstance(bc, np.ndarray):
                        num_bc = len(bc)
                        tens = np.zeros((self.grid_X.npts,) * 2)
                        tens[:num_bc, :] += bc
                        tens = tens.reshape(*([ax.npts for ax in self.grid_X.axes] * 2))
                        bc = self.grid_X.map_operator_to_mpo(tens)
                        laplacian_mpo = laplacian_mpo.add(bc, compress=True)
                    elif isinstance(bc, laplacian_mpo.__class__) and bc.data_type is DataType.MPO:
                        laplacian_mpo = laplacian_mpo.add(bc, compress=True)
                    else:
                        raise NotImplementedError
                    self.inv_laplacian_mpo = laplacian_mpo

            new_V_mps = soln_gtn.solve(laplacian_mpo, CompressType.DMRG, inplace=False,
                                       compress_opts=compress_opts, max_tot_iter=100)

        V_field = ScalarField('V', self.grid_X, new_V_mps)
        E_field = V_field.gradient()
        E_field.scalar_multiply(-1, inplace=True)
        # for C, EC in E_field.components.items():
        #     EC.scalar_mutliply(-1, inplace=True)
        return E_field

    def update_EM_sys(self, dt, inplace=False, compute_density=False, compute_current=True, compress=1,
                      compress1=2, compress2=4, **kwargs):
        """ update E, B (called by split step)
        """
        state0 = self if inplace else self.copy()

        if compute_current:
            state0.EM_sys.current_density = state0.compute_current(compress=compress1)
        if compute_density:
            state0.EM_sys.charge_density = state0.compute_charge_density(compress=compress1)

        print('update EM sys', state0.EM_sys.te_order)

        # for C in state0.EM_sys.E.componentIDs:
        #     plt.figure()
        #     EC = state0.EM_sys.E.get_comp_data(C)
        #     if EC is not None:
        #         plt.imshow(EC, label=f'E{C}')
        #         plt.title(f'E{C}')
        #         plt.colorbar()
        #
        # for C in state0.EM_sys.B.componentIDs:
        #     plt.figure()
        #     BC = state0.EM_sys.B.get_comp_data(C)
        #     if BC is not None:
        #         plt.imshow(BC, label=f'B{C}')
        #         plt.title(f'B{C}')
        #         plt.colorbar()
        # # plt.show()

        if state0.EM_sys.is_yee or state0.EM_sys.te_order == 31:
            state0.EM_sys.split_step(dt, inplace=True, compress_level=compress, **kwargs)
        else:
            state0.EM_sys = state0.EM_sys.next_time_step(dt, inplace=True, compress_level=compress, **kwargs)
            # if state0.EM_sys.te_order == 22:
            #     state0.EM_sys.crank_nicolson(dt, inplace=True, compress_level=compress)
            # elif state0.EM_sys.te_order == 21:
            #     state0.EM_sys.backwards_euler(dt, inplace=True, compress_level=compress)
            # else:
            #     state0.EM_sys.split_step(dt, inplace=True, compress_level=compress, **kwargs)
            #     # state0.EM_sys.euler(dt, inplace=True, compress_level=compress, compress_level1=compress2)
            #     # raise NotImplementedError

        # for C in state0.EM_sys.E.componentIDs:
        #     plt.figure()
        #     EC = state0.EM_sys.E.get_comp_data(C)
        #     if EC is not None:
        #         plt.imshow(EC, label=f'E{C}')
        #         plt.title(f'E{C}')
        #         plt.colorbar()
        #
        # for C in state0.EM_sys.B.componentIDs:
        #     plt.figure()
        #     BC = state0.EM_sys.B.get_comp_data(C)
        #     if BC is not None:
        #         plt.imshow(BC, label=f'C{C}')
        #         plt.title(f'B{C}')
        #         plt.colorbar()
        #
        # plt.show()

        state0.EM_sys.time = self.EM_sys.time + dt if self.EM_sys.time is not None else None
        return state0

    # @profile
    def euler(self, dt: Numeric, deriv0: Optional['VlasovMaxwell'] = None, inplace=False, do_update_V=True,
              compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0, verbose_plot: bool = False,
              do_x_advection=True, do_v_advection=True, evolve_EM=True) \
            -> 'VlasovMaxwell':
        """ perform explicit Euler time evolution
            compress:  final compression of final state, updateV
            compress1: compression of derivative
            compress2: compress1 of calculate derivative, updateV

            Euler step in Juno paper:
            1. compute current (old J)
            2. advance Vlasov distribution fcts with old fields
            3. advance Maxwell's equations; using old current
        """
        if evolve_EM:
            # state1.update_EM_sys(dt, inplace=inplace)
            j_old = self.compute_current(compress=compress_level + 1)
            self.EM_sys.current_density = j_old  # old current

        ## this updates EM fields and dist fct.
        state1 = super().euler(dt, deriv0=deriv0, inplace=inplace, compress_level=compress_level,
                               compress_level1=compress_level1, compress_level2=compress_level2,
                               do_x_advection=do_x_advection, do_v_advection=do_v_advection,
                               verbose_plot=verbose_plot)

        if self.EM_sys.verbose_plot:
            for C in self.coords_x.coords:
                plt.figure()
                j_old_data = j_old.get_comp_data(C)
                j_new = state1.compute_current()
                j_new_data = j_new.get_comp_data(C)
                if j_old_data is not None:
                    plt.plot(np.real(j_old_data), '--', label='old j')
                    plt.plot(np.imag(j_old_data), '--', label='old j im')
                    plt.plot(np.real(j_new_data), ':', label='new j')
                    plt.plot(np.imag(j_new_data), ':', label='new j im')
                    plt.plot(-np.real(j_old_data - j_new_data) / dt, label='dj/dt')
                    plt.plot(-np.imag(j_old_data - j_new_data) / dt, label='dj/dt im')
                plt.legend()
                plt.title(f'J after df/dt {C}')
            plt.show()

        if evolve_EM:
            # state1.EM_sys.current_density = j_old   # old current
            deriv_EM = None if deriv0 is None else deriv0.EM_sys
            state1.EM_sys.euler(dt, deriv0=deriv_EM, inplace=inplace)


        state1.time = self.time + dt if self.time is not None else None
        return state1


    def rk4(self, dt, deriv0: Optional['PDE_system'] = None, compress_level: int = 1,
            verbose_plot=False, is_first_time_step=False, is_last_time_step=False, **deriv_kwargs) -> 'VlasovMaxwell':
        """ do rk4 by updating EM field with distribution fcts
            ignores self.EM_sys.is_yee (just does RK4)
        """

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if self.EM_sys.te_order in [31, 221, 223, 224, 226]:
            evolve_EM = self.evolve_EM

            self.evolve_EM = False
            state1 = super().rk4(dt, deriv0=deriv0, compress_level=compress_level,
                                              verbose_plot=verbose_plot, **deriv_kwargs)

            self.evolve_EM = evolve_EM
            state1.evolve_EM = evolve_EM

            if evolve_EM:
                state1.EM_sys.current_density = state1.compute_current(compress=comp2)
                state1.update_EM_sys(dt, inplace=True, is_first_time_step=is_first_time_step,
                                     is_last_time_step=is_last_time_step,
                                     compute_current=False, compress=comp1, compress1=comp2)

        else:
            state1 = super().rk4(dt, deriv0=deriv0, compress_level=compress_level, verbose_plot=verbose_plot,
                                 evolve_EM=self.evolve_EM, **deriv_kwargs)

        return state1


    def lax_wendroff_ndim(self, dt: Numeric, advec_axes=None, ax_deriv_configs=None, do_update_V=True, inplace=False,
                          compress_level=1, verbose_plot=False, **kwargs):
        return super().lax_wendroff_ndim(dt, advec_axes=advec_axes, ax_deriv_configs=ax_deriv_configs,
                                         do_update_V=self.evolve_EM, inplace=inplace, compress_level=compress_level,
                                         verbose_plot=verbose_plot, **kwargs)


    def time_dependent_variational_principle(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace=False,
                                             compress_level: int = 1, compress_level_2: int = 4, direction=1,
                                             advec_axes: Sequence['Axis'] = None, background_force=True,
                                             internal_force=True,
                                             is_first_time_step=False, is_last_time_step=False,
                                             do_update_V=True, verbose_plot: bool = False, **kwargs) -> 'VlasovMaxwell':

        state0 = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if False:  # self.EM_sys.is_yee and self.evolve_EM:
            # state0 = super().rk4(dt / 2, deriv0=deriv0, compress_level=comp1, verbose_plot=verbose_plot,
            #                      **deriv_kwargs)
            if is_first_time_step:
                state0 = super(type(state0), state0).time_dependent_variational_principle(dt / 2, te_order=te_order,
                                                                                          inplace=True,
                                                                                          do_adapt=do_adapt,
                                                                                          compress_level=compress_level,
                                                                                          compress_level_2=compress_level_2,
                                                                                          advec_axes=advec_axes,
                                                                                          background_force=background_force,
                                                                                          internal_force=internal_force,
                                                                                          update_force=True,
                                                                                          do_update_V=False)
            else:
                state0 = self.copy()

            ## at initialization, need to evolve EM_sys with dt/2
            if self.evolve_EM and is_first_time_step:
                # print('FIRST TIME STEP')
                ## evolves B -> t = 1/2;
                state0.EM_sys = state0.EM_sys.evolve_B(dt / 2, inplace=True, method='rk4',
                                                       compress=comp1, compress1=comp4, compress2=comp5)
            else:
                ## n - 1/2 -> n + 1/2
                state0.EM_sys = state0.EM_sys.evolve_B(dt, inplace=True,
                                                       compress=comp1, compress1=comp4, compress2=comp5)

            ## current at t = n + 1/2
            print('update j')
            j = state0.compute_current(compress=comp2)
            # print('update j', j.norm())
            state0.EM_sys.current_density = j
            ## TODO: propagation of j boundary conditions? should match that of E

            ## t: n -> n + 1
            state0.EM_sys = state0.EM_sys.evolve_E(dt, inplace=True,
                                                   compress=comp1, compress1=comp4, compress2=comp5)

            ## compute elec force term
            force_term = state0.compute_force_term(is_ion=False, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=background_force, internal_force=internal_force)
            state0.sys_fe.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            ## compute ion force term
            force_term = state0.compute_force_term(is_ion=True, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=background_force, internal_force=internal_force)
            state0.sys_fi.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            print('tdvp')
            dt_ = dt / 2 if is_last_time_step else dt
            state1 = super(type(state0), state0).time_dependent_variational_principle(dt_, te_order=te_order,
                                                                                      inplace=True, do_adapt=do_adapt,
                                                                                      compress_level=compress_level,
                                                                                      compress_level_2=compress_level_2,
                                                                                      advec_axes=advec_axes,
                                                                                      background_force=background_force,
                                                                                      internal_force=internal_force,
                                                                                      update_force=False,
                                                                                      do_update_V=False)

        else:
            state1 = super(type(state0), state0).time_dependent_variational_principle(dt, te_order=te_order,
                                                                                      inplace=True, do_adapt=do_adapt,
                                                                                      compress_level=compress_level,
                                                                                      compress_level_2=compress_level_2,
                                                                                      advec_axes=advec_axes,
                                                                                      background_force=background_force,
                                                                                      internal_force=internal_force,
                                                                                      do_update_V=self.evolve_EM)
            # if self.evolve_EM:
            #     state1.update_EM_sys(dt, inplace=True, compress=compress_level,
            #                          compress2=compress_level + 1)
        return state1

    def time_dependent_variational_principle_SL(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace=False,
                                                compress_level: int = 1, compress_level_2: int = 4, direction=1,
                                                advec_axes: Sequence['Axis'] = None,
                                                is_first_time_step=False, is_last_time_step=False,
                                                bg_method='SL', bg_split_order=1,
                                                do_update_V=True, verbose_plot: bool = False,
                                                **kwargs) -> 'VlasovMaxwell':

        state0 = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if self.EM_sys.is_yee and self.evolve_EM:
            # state0 = super().rk4(dt / 2, deriv0=deriv0, compress_level=comp1, verbose_plot=verbose_plot,
            #                      **deriv_kwargs)
            if is_first_time_step:
                state0 = super(type(state0), state0).time_dependent_variational_principle_SL(dt / 2,
                                                                                             te_order=te_order,
                                                                                             inplace=True,
                                                                                             do_adapt=do_adapt,
                                                                                             compress_level=compress_level,
                                                                                             compress_level_2=compress_level_2,
                                                                                             advec_axes=advec_axes,
                                                                                             bg_method=bg_method,
                                                                                             bg_split_order=bg_split_order,
                                                                                             update_force=True,
                                                                                             do_update_V=False)
            else:
                state0 = self.copy()

            ## upate EM system
            ## at initialization, need to evolve EM_sys with dt/2
            if self.evolve_EM and is_first_time_step:
                # print('FIRST TIME STEP')
                ## evolves B -> t = 1/2;
                state0.EM_sys = state0.EM_sys.evolve_B(dt / 2, inplace=True, method='rk4',
                                                       compress=comp1, compress1=comp4, compress2=comp5)
            else:
                ## n - 1/2 -> n + 1/2
                state0.EM_sys = state0.EM_sys.evolve_B(dt, inplace=True,
                                                       compress=comp1, compress1=comp4, compress2=comp5)

            ## SL background force update t -> n + 1/2
            print('SL update (1)')
            if state0.sys_fe is not None:
                state0.sys_fe.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                                  internal_force=False, split_order=bg_split_order,
                                                  ompress=compress_level, compress1=compress_level_2)
            if state0.sys_fi is not None:
                state0.sys_fi.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                                  internal_force=False, split_order=bg_split_order,
                                                  ompress=compress_level, compress1=compress_level_2)

            ## current at t = n + 1/2
            print('update j')
            j = state0.compute_current(compress=comp2)
            # print('update j', j.norm())
            state0.EM_sys.current_density = j
            ## TODO: propagation of j boundary conditions? should match that of E

            ## t: n -> n + 1
            state0.EM_sys = state0.EM_sys.evolve_E(dt, inplace=True,
                                                   compress=comp1, compress1=comp4, compress2=comp5)

            ## compute elec force term    ### update in later fct
            force_term = state0.compute_force_term(is_ion=False, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=False, internal_force=True)
            state0.sys_fe.set_force_term(force_term, background_force=False, internal_force=True)

            ## compute ion force term
            force_term = state0.compute_force_term(is_ion=True, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=False, internal_force=True)
            state0.sys_fi.set_force_term(force_term, background_force=False, internal_force=True)

            ### time evolution (pert) t: n -> n + 1
            print('tdvp')
            dt_ = dt / 2 if is_last_time_step else dt
            state1 = super(type(state0), state0).time_dependent_variational_principle(dt_, te_order=te_order,
                                                                                      inplace=True, do_adapt=do_adapt,
                                                                                      compress_level=compress_level,
                                                                                      compress_level_2=compress_level_2,
                                                                                      advec_axes=advec_axes,
                                                                                      # bg_method=bg_method,
                                                                                      # bg_split_order=bg_split_order,
                                                                                      background_force=False,
                                                                                      internal_force=True,
                                                                                      update_force=False,
                                                                                      do_update_V=False)

            ## SL background update: t: n + 1/2 -> n + 1
            print('SL update (2)')
            bg_split_order = -1 if bg_split_order == 1 else bg_split_order
            if state1.sys_fe is not None:
                state1.sys_fe.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                                  internal_force=False, split_order=bg_split_order,
                                                  ompress=compress_level, compress1=compress_level_2)
            if state1.sys_fi is not None:
                state1.sys_fi.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                                  internal_force=False, split_order=bg_split_order,
                                                  ompress=compress_level, compress1=compress_level_2)

        else:
            state1 = super(type(state0), state0).time_dependent_variational_principle_SL(dt, te_order=te_order,
                                                                                         inplace=True,
                                                                                         do_adapt=do_adapt,
                                                                                         compress_level=compress_level,
                                                                                         compress_level_2=compress_level_2,
                                                                                         advec_axes=advec_axes,
                                                                                         update_force=True,
                                                                                         do_update_V=self.evolve_EM)

        return state1

    def time_dmrg(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace=False,
                  compress_level: int = 1, compress_level_2: int = 4, direction=1,
                  advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                  is_first_time_step=False, is_last_time_step=False,
                  do_update_V=True, verbose_plot: bool = False,
                  solver_type=LocalSolverType.TDDMRG, **kwargs) -> 'VlasovMaxwell':

        state0 = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if False:  # self.EM_sys.is_yee and self.evolve_EM:
            # state0 = super().rk4(dt / 2, deriv0=deriv0, compress_level=comp1, verbose_plot=verbose_plot,
            #                      **deriv_kwargs)
            if is_first_time_step:
                state0 = super(type(state0), state0).time_dmrg(dt / 2, te_order=te_order,
                                                               inplace=True, do_adapt=do_adapt,
                                                               compress_level=compress_level,
                                                               compress_level_2=compress_level_2,
                                                               advec_axes=advec_axes,
                                                               background_force=background_force,
                                                               internal_force=internal_force,
                                                               update_force=True,
                                                               do_update_V=False,
                                                               solver_type=solver_type)
            else:
                state0 = self.copy()

            ## at initialization, need to evolve EM_sys with dt/2
            if self.evolve_EM and is_first_time_step:
                # print('FIRST TIME STEP')
                ## evolves B -> t = 1/2;
                state0.EM_sys = state0.EM_sys.evolve_B(dt / 2, inplace=True, method='rk4',
                                                       compress=comp1, compress1=comp4, compress2=comp5)
            else:
                ## n - 1/2 -> n + 1/2
                state0.EM_sys = state0.EM_sys.evolve_B(dt, inplace=True,
                                                       compress=comp1, compress1=comp4, compress2=comp5)

            ## current at t = n + 1/2
            print('update j')
            j = state0.compute_current(compress=comp2)
            # print('update j', j.norm())
            state0.EM_sys.current_density = j
            ## TODO: propagation of j boundary conditions? should match that of E

            ## t: n -> n + 1
            state0.EM_sys = state0.EM_sys.evolve_E(dt, inplace=True,
                                                   compress=comp1, compress1=comp4, compress2=comp5)

            ## compute elec force term
            force_term = state0.compute_force_term(is_ion=False, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=background_force, internal_force=internal_force)
            state0.sys_fe.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            ## compute ion force term
            force_term = state0.compute_force_term(is_ion=True, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=background_force, internal_force=internal_force)
            state0.sys_fi.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            print('tdmrg')
            dt_ = dt / 2 if is_last_time_step else dt
            state1 = super(type(state0), state0).time_dmrg(dt_, te_order=te_order,
                                                           inplace=True, do_adapt=do_adapt,
                                                           compress_level=compress_level,
                                                           compress_level_2=compress_level_2,
                                                           advec_axes=advec_axes,
                                                           background_force=background_force,
                                                           internal_force=internal_force,
                                                           update_force=False,
                                                           do_update_V=False,
                                                           solver_type=solver_type)

        else:
            state1 = super(type(state0), state0).time_dmrg(dt, te_order=te_order,
                                                           inplace=True, do_adapt=do_adapt,
                                                           compress_level=compress_level,
                                                           compress_level_2=compress_level_2,
                                                           advec_axes=advec_axes,
                                                           background_force=background_force,
                                                           internal_force=internal_force,
                                                           do_update_V=self.evolve_EM,
                                                           solver_type=solver_type)
            # if self.evolve_EM:
            #     state1.update_EM_sys(dt, inplace=True, compress=compress_level,
            #                          compress2=compress_level + 1)
        return state1


    def tdvp_new(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace=False,
                 compress_level: int = 1, compress_level_2: int = 4, direction=1, solver_type=LocalSolverType.TDDMRG,
                 advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                 is_first_time_step=False, is_last_time_step=False,
                 do_update_V=True, verbose_plot: bool = False, **kwargs) -> 'VlasovMaxwell':

        state0 = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if self.EM_sys.is_yee and self.evolve_EM:
            # state0 = super().rk4(dt / 2, deriv0=deriv0, compress_level=comp1, verbose_plot=verbose_plot,
            #                      **deriv_kwargs)
            if is_first_time_step:
                state0 = super(type(state0), state0).tdvp_new(dt / 2, te_order=te_order,
                                                               inplace=True, do_adapt=do_adapt,
                                                               compress_level=compress_level,
                                                               compress_level_2=compress_level_2,
                                                               advec_axes=advec_axes,
                                                               solver_type=solver_type,
                                                               background_force=background_force,
                                                               internal_force=internal_force,
                                                               direction=direction,
                                                               update_force=True,
                                                               do_update_V=False)
            else:
                state0 = self.copy()

            ## at initialization, need to evolve EM_sys with dt/2
            if self.evolve_EM and is_first_time_step:
                # print('FIRST TIME STEP')
                ## evolves B -> t = 1/2;
                state0.EM_sys = state0.EM_sys.evolve_B(dt / 2, inplace=True, method='rk4',
                                                       compress=comp1, compress1=comp4, compress2=comp5)
            else:
                ## n - 1/2 -> n + 1/2
                state0.EM_sys = state0.EM_sys.evolve_B(dt, inplace=True,
                                                       compress=comp1, compress1=comp4, compress2=comp5)

            ## current at t = n + 1/2
            print('update j')
            j = state0.compute_current(compress=comp2)
            # print('update j', j.norm())
            state0.EM_sys.current_density = j
            ## TODO: propagation of j boundary conditions? should match that of E

            ## t: n -> n + 1
            state0.EM_sys = state0.EM_sys.evolve_E(dt, inplace=True,
                                                   compress=comp1, compress1=comp4, compress2=comp5)

            ## compute elec force term
            force_term = state0.compute_force_term(is_ion=False, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=background_force, internal_force=internal_force)
            state0.sys_fe.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            ## compute ion force term
            if self.evolve_ion:
                force_term = state0.compute_force_term(is_ion=True, compress_level=comp2, compress_level_2=comp4,
                                                       background_force=background_force, internal_force=internal_force)
                state0.sys_fi.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            print('tdmrg')
            dt_ = dt / 2 if is_last_time_step else dt
            state1 = super(type(state0), state0).tdvp_new(dt_, te_order=te_order,
                                                           inplace=True, do_adapt=do_adapt,
                                                           compress_level=compress_level,
                                                           compress_level_2=compress_level_2,
                                                           advec_axes=advec_axes,
                                                           solver_type=solver_type,
                                                           background_force=background_force,
                                                           internal_force=internal_force,
                                                           direction=direction,
                                                           update_force=False,
                                                           do_update_V=False)

        else:

            #### jank fix ####
            X, Y, Z = self.coords_x.coords
            Ex0, omega = 0.9, 0.4567
            time_mpos = {}

            print('calc time deriv (tdvp new)', self.time)
            new_Ex = Ex0 * np.cos(omega * self.time)
            self.EM_sys.E[X].data = new_Ex
            print('new Ex', self.EM_sys.E[X].data)
            force_term = self.compute_force_term()
            self.set_force_term(force_term)

            ### TDVP is second order so just use E[t+dt/2] (can set in run file)
            mpo_list_dt0 = self.sys_fe._get_time_evolution_mpos()
            time_mpos[np.round(self.time, 10)] = [m.data for m in mpo_list_dt0]
            time_mpos[np.round(self.time + dt / 4, 10)] = [m.data for m in mpo_list_dt0]
            time_mpos[np.round(self.time + dt / 2, 10)] = [m.data for m in mpo_list_dt0]
            time_mpos[np.round(self.time + dt * 3 / 4, 10)] = [m.data for m in mpo_list_dt0]
            time_mpos[np.round(self.time + dt, 10)] = [m.data for m in mpo_list_dt0]

            # ###################
            # ## need 1/4 increments because sweeps left and right
            # ## dt/4
            # time = self.time + dt / 4
            # print('calc time deriv', time, self.time)
            # print('old Ex', self.EM_sys.E[X].data)
            # new_Ex = Ex0 * np.cos(omega * time)
            # self.EM_sys.E[X].data = new_Ex
            # print('new Ex', self.EM_sys.E[X].data)
            # force_term = self.compute_force_term()
            # self.set_force_term(force_term)
            #
            # mpo_list_dt1 = self.sys_fe._get_time_evolution_mpos()
            # time_mpos[np.round(time,10)] = [m.data for m in mpo_list_dt1]
            #
            # ## dt/2
            # time = self.time + dt / 2
            # print('calc time deriv', time, self.time)
            # print('old Ex', self.EM_sys.E[X].data)
            # new_Ex = Ex0 * np.cos(omega * time)
            # self.EM_sys.E[X].data = new_Ex
            # print('new Ex', self.EM_sys.E[X].data)
            # force_term = self.compute_force_term()
            # self.set_force_term(force_term)
            #
            # mpo_list_dt2 = self.sys_fe._get_time_evolution_mpos()
            # time_mpos[np.round(time,10)] = [m.data for m in mpo_list_dt2]
            #
            # ## 3 dt/4
            # time = self.time + 3 * dt / 4
            # print('calc time deriv', time, self.time)
            # print('old Ex', self.EM_sys.E[X].data)
            # new_Ex = Ex0 * np.cos(omega * time)
            # self.EM_sys.E[X].data = new_Ex
            # print('new Ex', self.EM_sys.E[X].data)
            # force_term = self.compute_force_term()
            # self.set_force_term(force_term)
            #
            # mpo_list_dt3 = self.sys_fe._get_time_evolution_mpos()
            # time_mpos[np.round(time,10)] = [m.data for m in mpo_list_dt3]
            #
            # ## dt
            # time = self.time + dt
            # print('calc time deriv', time, self.time)
            # new_Ex = Ex0 * np.cos(omega * time)
            # self.EM_sys.E[X].data = new_Ex
            # print('new Ex', self.EM_sys.E[X].data)
            # force_term = self.compute_force_term()
            # self.set_force_term(force_term)
            #
            # mpo_list_dt4 = self.sys_fe._get_time_evolution_mpos()
            # time_mpos[np.round(time,10)] = [m.data for m in mpo_list_dt4]
            #
            # ## 0
            # ## reset to original ##
            # print('calc time deriv', time, self.time)
            # new_Ex = Ex0 * np.cos(omega * self.time)
            # self.EM_sys.E[X].data = new_Ex
            # print('new Ex', self.EM_sys.E[X].data)
            # force_term = self.compute_force_term()
            # self.set_force_term(force_term)
            #
            # mpo_list_dt0 = self.sys_fe._get_time_evolution_mpos()
            # time_mpos[np.round(self.time,10)] = [m.data for m in mpo_list_dt0]
            # #######################

            state1 = super(type(state0), state0).tdvp_new(dt, te_order=te_order,
                                                           inplace=True, do_adapt=do_adapt,
                                                           compress_level=compress_level,
                                                           compress_level_2=compress_level_2,
                                                           advec_axes=advec_axes,
                                                           solver_type=solver_type,
                                                           background_force=background_force,
                                                           internal_force=internal_force,
                                                           direction=direction,
                                                           time_mpo_list=time_mpos,
                                                           do_update_V=self.evolve_EM)
            # if self.evolve_EM:
            #     state1.update_EM_sys(dt, inplace=True, compress=compress_level,
            #                          compress2=compress_level + 1)
        return state1



    def time_dmrg_new(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace=False,
                  compress_level: int = 1, compress_level_2: int = 4, direction=1,
                  advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                  is_first_time_step=False, is_last_time_step=False, solver_type=LocalSolverType.TDDMRG,
                  do_update_V=True, verbose_plot: bool = False, **kwargs) -> 'VlasovMaxwell':

        state0 = self if inplace else self.copy()

        # local_solver = LocalSolverType.TDCross

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        ## if split_step EM_sys is used, will default to FDTD
        if False:  # self.EM_sys.is_yee and self.evolve_EM:
            # state0 = super().rk4(dt / 2, deriv0=deriv0, compress_level=comp1, verbose_plot=verbose_plot,
            #                      **deriv_kwargs)
            if is_first_time_step:
                state0 = super(type(state0), state0).time_dmrg_new(dt / 2, te_order=te_order,
                                                               inplace=True, do_adapt=do_adapt,
                                                               compress_level=compress_level,
                                                               compress_level_2=compress_level_2,
                                                               advec_axes=advec_axes,
                                                               background_force=background_force,
                                                               internal_force=internal_force,
                                                               solver_type=solver_type,
                                                               direction=direction,
                                                               update_force=True,
                                                               do_update_V=False)
            else:
                state0 = self.copy()

            ## at initialization, need to evolve EM_sys with dt/2
            if self.evolve_EM and is_first_time_step:
                # print('FIRST TIME STEP')
                ## evolves B -> t = 1/2;
                state0.EM_sys = state0.EM_sys.evolve_B(dt / 2, inplace=True, method='rk4',
                                                       compress=comp1, compress1=comp4, compress2=comp5)
            else:
                ## n - 1/2 -> n + 1/2
                state0.EM_sys = state0.EM_sys.evolve_B(dt, inplace=True,
                                                       compress=comp1, compress1=comp4, compress2=comp5)

            ## current at t = n + 1/2
            print('update j')
            j = state0.compute_current(compress=comp2)
            # print('update j', j.norm())
            state0.EM_sys.current_density = j
            ## TODO: propagation of j boundary conditions? should match that of E

            ## t: n -> n + 1
            state0.EM_sys = state0.EM_sys.evolve_E(dt, inplace=True,
                                                   compress=comp1, compress1=comp4, compress2=comp5)

            ## compute elec force term
            force_term = state0.compute_force_term(is_ion=False, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=background_force, internal_force=internal_force)
            state0.sys_fe.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            ## compute ion force term
            force_term = state0.compute_force_term(is_ion=True, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=background_force, internal_force=internal_force)
            state0.sys_fi.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            print('tdmrg')
            dt_ = dt / 2 if is_last_time_step else dt
            state1 = super(type(state0), state0).time_dmrg_new(dt_, te_order=te_order,
                                                           inplace=True, do_adapt=do_adapt,
                                                           compress_level=compress_level,
                                                           compress_level_2=compress_level_2,
                                                           advec_axes=advec_axes,
                                                           background_force=background_force,
                                                           internal_force=internal_force,
                                                           direction=direction,
                                                           update_force=False,
                                                           do_update_V=False)

        else:

            if solver_type == LocalSolverType.TDDMRG:

                #### jank fix ####
                X, Y, Z = self.coords_x.coords
                Ex0, omega = 0.9, 0.4567
                time_mpos = {}

                if te_order in [3, 4]:
                    time = self.time + dt/2
                    print('calc time deriv (tdmrg new)', time, self.time)
                    print('old Ex', state0.EM_sys.E[X].data)
                    new_Ex = Ex0 * np.cos(omega * time)
                    state0.EM_sys.E[X].data = new_Ex
                    print('new Ex', state0.EM_sys.E[X].data)
                    force_term = state0.compute_force_term()
                    state0.set_force_term(force_term)

                    mpo_list_dt2 = state0.sys_fe._get_time_evolution_mpos()
                    time_mpos[np.round(time,10)] = [m.data for m in mpo_list_dt2]

                    time = self.time + dt
                    print('calc time deriv (tdmrg new)', time, self.time)
                    new_Ex = Ex0 * np.cos(omega * time)
                    state0.EM_sys.E[X].data = new_Ex
                    print('new Ex', state0.EM_sys.E[X].data)
                    force_term = state0.compute_force_term()
                    state0.set_force_term(force_term)

                    mpo_list_dt4 = state0.sys_fe._get_time_evolution_mpos()
                    time_mpos[np.round(time,10)] = [m.data for m in mpo_list_dt4]

                ## reset to original ##
                print('calc time deriv (tdmrg new)', self.time)
                new_Ex = Ex0 * np.cos(omega * self.time)
                state0.EM_sys.E[X].data = new_Ex
                print('new Ex', state0.EM_sys.E[X].data)
                force_term = state0.compute_force_term()
                state0.set_force_term(force_term)
                # else:
                #     ## use provided Ex
                #     print('calc time deriv (tdmrg new)', self.time)
                #     print('use old Ex', state0.EM_sys.E[X].data)
                #     force_term = state0.compute_force_term()
                #     state0.set_force_term(force_term)

                #######################

            elif solver_type == LocalSolverType.TDCross:

                print('here X time dmrg')

                X, Y, Z = state0.coords_x.coords
                Ex0, omega = 0.9, 0.4567
                state0.sys_fe.time = self.time
                print('sys fe time', state0.sys_fe.time)

                time_mpos = {}

                if te_order in [3, 4]:
                    time = state0.time + dt / 2
                    new_Ex = Ex0 * np.cos(omega * time)
                    state0.EM_sys.E[X].data = new_Ex
                    print('new Ex', state0.EM_sys.E[X].data, 'time', time)
                    force_term = state0.compute_force_term()
                    state0.set_force_term(force_term, time=(time if self.upwind else None), reset=True)
                    if not self.upwind:
                        mpo_list_dt2 = state0.sys_fe._get_time_evolution_mpos()
                        time_mpos[np.round(time, 10)] = [m.data for m in mpo_list_dt2]

                    time = state0.time + dt
                    new_Ex = Ex0 * np.cos(omega * time)
                    state0.EM_sys.E[X].data = new_Ex
                    print('new Ex', state0.EM_sys.E[X].data, 'time', time)
                    force_term = state0.compute_force_term()
                    state0.set_force_term(force_term, time=(time if self.upwind else None), reset=False)

                    if not self.upwind:
                        mpo_list_dt4 = state0.sys_fe._get_time_evolution_mpos()
                        time_mpos[np.round(time, 10)] = [m.data for m in mpo_list_dt4]


                ## reset to original ##
                time = state0.time
                new_Ex = Ex0 * np.cos(omega * time)
                state0.EM_sys.E[X].data = new_Ex
                print('new Ex', state0.EM_sys.E[X].data, 'time', time)
                force_term = state0.compute_force_term()
                state0.set_force_term(force_term, time=(time if self.upwind else None), reset=False)




            state1 = super(type(state0), state0).time_dmrg_new(dt, te_order=te_order,
                                                               inplace=True, do_adapt=do_adapt,
                                                               compress_level=compress_level,
                                                               compress_level_2=compress_level_2,
                                                               advec_axes=advec_axes,
                                                               background_force=background_force,
                                                               internal_force=internal_force,
                                                               do_update_V=self.evolve_EM,
                                                               time_mpo_list=time_mpos,
                                                               update_force=False,    ## setting this to true causes issues
                                                               solver_type=solver_type,
                                                               direction=direction,
                                                               time=state0.time,
                                                               )
            # if self.evolve_EM:
            #     state1.update_EM_sys(dt, inplace=True, compress=compress_level,
            #                          compress2=compress_level + 1)
        return state1


    def time_local_global(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace=False,
                          compress_level: int = 1, compress_level_2: int = 4, direction=1,
                          advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                          is_first_time_step=False, is_last_time_step=False,
                          do_update_V=True, verbose_plot: bool = False, **kwargs) -> 'VlasovMaxwell':

        state0 = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if self.EM_sys.is_yee and self.evolve_EM:
            # state0 = super().rk4(dt / 2, deriv0=deriv0, compress_level=comp1, verbose_plot=verbose_plot,
            #                      **deriv_kwargs)
            if is_first_time_step:
                state0 = super(type(state0), state0).time_local_global(dt / 2, te_order=te_order,
                                                                       inplace=True, do_adapt=do_adapt,
                                                                       compress_level=compress_level,
                                                                       compress_level_2=compress_level_2,
                                                                       advec_axes=advec_axes,
                                                                       background_force=background_force,
                                                                       internal_force=internal_force,
                                                                       update_force=True,
                                                                       do_update_V=False)
            else:
                state0 = self.copy()

            ## at initialization, need to evolve EM_sys with dt/2
            if self.evolve_EM and is_first_time_step:
                # print('FIRST TIME STEP')
                ## evolves B -> t = 1/2;
                state0.EM_sys = state0.EM_sys.evolve_B(dt / 2, inplace=True, method='rk4',
                                                       compress=comp1, compress1=comp4, compress2=comp5)
            else:
                ## n - 1/2 -> n + 1/2
                state0.EM_sys = state0.EM_sys.evolve_B(dt, inplace=True,
                                                       compress=comp1, compress1=comp4, compress2=comp5)

            ## current at t = n + 1/2
            print('update j')
            j = state0.compute_current(compress=comp2)
            # print('update j', j.norm())
            state0.EM_sys.current_density = j
            ## TODO: propagation of j boundary conditions? should match that of E

            ## t: n -> n + 1
            state0.EM_sys = state0.EM_sys.evolve_E(dt, inplace=True,
                                                   compress=comp1, compress1=comp4, compress2=comp5)

            ## compute elec force term
            force_term = state0.compute_force_term(is_ion=False, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=background_force, internal_force=internal_force)
            state0.sys_fe.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            ## compute ion force term
            force_term = state0.compute_force_term(is_ion=True, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=background_force, internal_force=internal_force)
            state0.sys_fi.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            print('tdmrg')
            dt_ = dt / 2 if is_last_time_step else dt
            state1 = super(type(state0), state0).time_local_global(dt_, te_order=te_order,
                                                                   inplace=True, do_adapt=do_adapt,
                                                                   compress_level=compress_level,
                                                                   compress_level_2=compress_level_2,
                                                                   advec_axes=advec_axes,
                                                                   background_force=background_force,
                                                                   internal_force=internal_force,
                                                                   update_force=False,
                                                                   do_update_V=False)

        else:
            state1 = super(type(state0), state0).time_local_global(dt, te_order=te_order,
                                                                   inplace=True, do_adapt=do_adapt,
                                                                   compress_level=compress_level,
                                                                   compress_level_2=compress_level_2,
                                                                   advec_axes=advec_axes,
                                                                   background_force=background_force,
                                                                   internal_force=internal_force,
                                                                   do_update_V=self.evolve_EM)
            # if self.evolve_EM:
            #     state1.update_EM_sys(dt, inplace=True, compress=compress_level,
            #                          compress2=compress_level + 1)
        return state1



    def time_dmrg_SL(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace=False,
                     compress_level: int = 1, compress_level_2: int = 4, direction=1,
                     advec_axes: Sequence['Axis'] = None,
                     is_first_time_step=False, is_last_time_step=False,
                     bg_method='SL', bg_split_order=1,
                     do_update_V=True, verbose_plot: bool = False, **kwargs) -> 'VlasovMaxwell':

        state0 = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if self.EM_sys.is_yee and self.evolve_EM:
            # state0 = super().rk4(dt / 2, deriv0=deriv0, compress_level=comp1, verbose_plot=verbose_plot,
            #                      **deriv_kwargs)
            if is_first_time_step:
                state0 = super(type(state0), state0).time_dmrg_SL(dt / 2,
                                                                  te_order=te_order,
                                                                  inplace=True, do_adapt=do_adapt,
                                                                  compress_level=compress_level,
                                                                  compress_level_2=compress_level_2,
                                                                  advec_axes=advec_axes,
                                                                  bg_method=bg_method,
                                                                  bg_split_order=bg_split_order,
                                                                  update_force=True,
                                                                  do_update_V=False)
            else:
                state0 = self.copy()

            ## upate EM system
            ## at initialization, need to evolve EM_sys with dt/2
            if self.evolve_EM and is_first_time_step:
                # print('FIRST TIME STEP')
                ## evolves B -> t = 1/2;
                state0.EM_sys = state0.EM_sys.evolve_B(dt / 2, inplace=True, method='rk4',
                                                       compress=comp1, compress1=comp4, compress2=comp5)
            else:
                ## n - 1/2 -> n + 1/2
                state0.EM_sys = state0.EM_sys.evolve_B(dt, inplace=True,
                                                       compress=comp1, compress1=comp4, compress2=comp5)

            ## SL background force update t -> n + 1/2
            print('SL update (1)')
            if state0.sys_fe is not None:
                state0.sys_fe.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                                  internal_force=False, split_order=bg_split_order,
                                                  ompress=compress_level, compress1=compress_level_2)
            if state0.sys_fi is not None:
                state0.sys_fi.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                                  internal_force=False, split_order=bg_split_order,
                                                  ompress=compress_level, compress1=compress_level_2)

            ## current at t = n + 1/2
            print('update j')
            j = state0.compute_current(compress=comp2)
            # print('update j', j.norm())
            state0.EM_sys.current_density = j
            ## TODO: propagation of j boundary conditions? should match that of E

            ## t: n -> n + 1
            state0.EM_sys = state0.EM_sys.evolve_E(dt, inplace=True,
                                                   compress=comp1, compress1=comp4, compress2=comp5)

            ## compute elec force term    ### update in later fct
            force_term = state0.compute_force_term(is_ion=False, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=False, internal_force=True)
            state0.sys_fe.set_force_term(force_term, background_force=False, internal_force=True)

            ## compute ion force term
            force_term = state0.compute_force_term(is_ion=True, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=False, internal_force=True)
            state0.sys_fi.set_force_term(force_term, background_force=False, internal_force=True)

            ### time evolution (pert) t: n -> n + 1
            print('tdmrg')
            dt_ = dt / 2 if is_last_time_step else dt
            state1 = super(type(state0), state0).time_dmrg(dt_, te_order=te_order,
                                                           inplace=True, do_adapt=do_adapt,
                                                           compress_level=compress_level,
                                                           compress_level_2=compress_level_2,
                                                           advec_axes=advec_axes,
                                                           # bg_method=bg_method,
                                                           # bg_split_order=bg_split_order,
                                                           background_force=False,
                                                           internal_force=True,
                                                           update_force=False,
                                                           do_update_V=False)

            ## SL background update: t: n + 1/2 -> n + 1
            print('SL update (2)')
            bg_split_order = -1 if bg_split_order == 1 else bg_split_order
            if state1.sys_fe is not None:
                state1.sys_fe.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                                  internal_force=False, split_order=bg_split_order,
                                                  ompress=compress_level, compress1=compress_level_2)
            if state1.sys_fi is not None:
                state1.sys_fi.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                                  internal_force=False, split_order=bg_split_order,
                                                  ompress=compress_level, compress1=compress_level_2)

        else:
            state1 = super(type(state0), state0).time_dmrg_SL(dt, te_order=te_order,
                                                              inplace=True, do_adapt=do_adapt,
                                                              compress_level=compress_level,
                                                              compress_level_2=compress_level_2,
                                                              advec_axes=advec_axes,
                                                              update_force=True,
                                                              do_update_V=self.evolve_EM)

        return state1

    def dynamical_low_rank(self, dt: Numeric, inplace=False, te_order=4, do_adapt: bool = True,
                           compress_level: int = 1, compress_level_2: int = 0,
                           advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                           is_first_time_step=False, is_last_time_step=False,
                           do_update_V=True, verbose_plot: bool = False, **kwargs) -> 'VlasovMaxwell':

        state0 = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        if self.EM_sys.is_yee and self.evolve_EM:
            # state0 = super().rk4(dt / 2, deriv0=deriv0, compress_level=comp1, verbose_plot=verbose_plot,
            #                      **deriv_kwargs)
            if is_first_time_step:
                pass  ## rk4 called in pde_system
            else:
                state0 = self.copy()

            ## at initialization, need to evolve EM_sys with dt/2
            if self.evolve_EM and is_first_time_step:
                # print('FIRST TIME STEP')
                ## evolves B -> t = 1/2;
                state0.EM_sys = state0.EM_sys.evolve_B(dt / 2, inplace=True, method='rk4',
                                                       compress=comp1, compress1=comp4, compress2=comp5)
            else:
                ## n - 1/2 -> n + 1/2
                state0.EM_sys = state0.EM_sys.evolve_B(dt, inplace=True,
                                                       compress=comp1, compress1=comp4, compress2=comp5)

            ## current at t = n + 1/2
            print('update j')
            j = state0.compute_current(compress=comp2)
            state0.EM_sys.current_density = j
            ## TODO: propagation of j boundary conditions? should match that of E

            ## t: n -> n + 1
            state0.EM_sys = state0.EM_sys.evolve_E(dt, inplace=True,
                                                   compress=comp1, compress1=comp4, compress2=comp5)

            ## compute elec force term
            force_term = state0.compute_force_term(is_ion=False, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=background_force, internal_force=internal_force)
            state0.sys_fe.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            ## compute ion force term
            force_term = state0.compute_force_term(is_ion=True, compress_level=comp2, compress_level_2=comp4,
                                                   background_force=background_force, internal_force=internal_force)
            state0.sys_fi.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            print('dlr')
            dt_ = dt / 2 if is_last_time_step else dt
            state1 = super(type(state0), state0).dynamical_low_rank(dt_, te_order=te_order,
                                                                    inplace=True, do_adapt=do_adapt,
                                                                    compress_level=compress_level,
                                                                    compress_level_2=compress_level_2,
                                                                    advec_axes=advec_axes,
                                                                    background_force=background_force,
                                                                    internal_force=internal_force,
                                                                    update_force=False,
                                                                    do_update_V=False)

        else:
            if not is_first_time_step:
                state1 = super(type(state0), state0).dynamical_low_rank(dt, te_order=te_order,
                                                                        inplace=True, do_adapt=do_adapt,
                                                                        compress_level=compress_level,
                                                                        compress_level_2=compress_level_2,
                                                                        advec_axes=advec_axes,
                                                                        background_force=background_force,
                                                                        internal_force=internal_force,
                                                                        do_update_V=True)
            # if self.evolve_EM:
            #     state1.update_EM_sys(dt, inplace=True, compress=compress_level,
            #                          compress2=compress_level + 1)

        return state1


    def block_tddmrg(self, dt: Numeric, te_order=4, do_adapt: bool = True, inplace=False,
                     compress_level: int = 1, compress_level_2: int = 4, direction=1,
                     advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                     is_first_time_step=False, is_last_time_step=False,
                     do_update_V=True, verbose_plot: bool = False, **kwargs) -> 'VlasovMaxwell':

        new_sys = self if inplace else self.copy()
        use_dmrg = True

        num_active_fields = 2 if self.evolve_ion else 1
        if self.evolve_EM:
            num_active_fields += self.EM_sys.get_num_active_fields()
        all_fieds_vec = self.get_combined_state()
        all_fieds_time_deriv = self.get_combined_derivative_mpo()

        return self


    def get_state_dict(self):
        """ fe: 7, fi: 8 """
        all_fields_dict = {}
        if self.evolve_EM:
            all_fields_dict = self.EM_sys.get_state_dict()
        all_fields_dict[7] = self.fe.component
        if self.evolve_ion:
            all_fields_dict[8] = self.fi.component
        return all_fields_dict


    def get_derivative_dict(self, dt=None) -> dict[tuple[int,int],'GridTN']:
        """ get operations acting on E, B, fe, fi to obtain dE/dt, dB/dt, dfe/dt, dfi/dt
            information returned as a dict with keys (output ind, input ind)
            ideally would also have "quadratic" terms (so that we can also update them
            during the DMRG procedure).
        """
        dt = self.dt if dt is None else dt

        deriv_dict = {}
        if self.evolve_EM:
            deriv_dict = self.get_derivative_dict(dt)

        ###
        raise NotImplementedError


    def get_combined_state(self):
        return

    def get_combined_derivative_mpo(self, advec_axes: Sequence['Axis'] = None, include_E=True, include_B=True,
                                    background_force=True, internal_force=True,):
        return

    #############################

    def compute_force_term(self, is_ion=False, compress_level=1, compress_level1=0, compress_level2=0,
                           verbose_plot=False, background_force=True, internal_force=True,
                           include_E=True, include_B=True, **kwargs):
        """ compute EM force:  q/m (E + v x B)
        """
        # print('EM computing force; is ion?', is_ion)
        # print('background', background_force, 'internal', internal_force)
        coords_v = self.coords_vi if is_ion else self.coords_ve
        velocities = self.velocities_i if is_ion else self.velocities_e  ## MPOs; i think it should still work
        matl_params = self.ion_params if is_ion else self.elc_params

        background_B0 = self.EM_sys.background_B0 if background_force else None
        background_E0 = self.EM_sys.background_E0 if background_force else None

        # print('compress', compress_level, compress_level1, compress_level2)

        em_term = None
        if include_B:
            if self.B is not None and internal_force:
                B = self.B.add(background_B0, compress_level=compress_level2)
            else:
                B = background_B0

            em_term = velocities.cross_product(B, self.coords_x, compress_level=compress_level1,
                                               inner_compress_level=compress_level2, verbose_plot=False, ) \
                if B is not None else None

            # X, Y, Z = self.coords_x.coords
            # plt.figure()
            # plt.plot(np.real(B.get_comp_data(X)))
            # plt.plot(np.imag(B.get_comp_data(X)), '--')
            # plt.title('BX')
            # plt.figure()
            # plt.plot(np.real(B.get_comp_data(Y)))
            # plt.plot(np.imag(B.get_comp_data(Y)), '--')
            # plt.title('BY')
            # plt.figure()
            # plt.plot(np.real(B.get_comp_data(Z)))
            # plt.plot(np.imag(B.get_comp_data(Z)), '--')
            # plt.title('BZ')
            # plt.show()

            if matl_params.is_cgs:
                em_term.scalar_multiply(1. / matl_params.c, inplace=True)

        if include_E:
            if self.E is not None and internal_force:
                E = self.E.add(background_E0, compress_level=compress_level2)
            else:
                E = background_E0

            # X, Y, Z = self.coords_x.coords
            # if E.get_comp_data(X) is not None:
            #     plt.figure()
            #     plt.plot(np.real(E.get_comp_data(X)))
            #     plt.plot(np.imag(E.get_comp_data(X)), '--')
            #     plt.title('EX')
            # plt.figure()
            # plt.plot(np.real(E.get_comp_data(Y)))
            # plt.plot(np.imag(E.get_comp_data(Y)), '--')
            # plt.title('EY')
            # plt.figure()
            # plt.plot(np.real(E.get_comp_data(Z)))
            # plt.plot(np.imag(E.get_comp_data(Z)), '--')
            # plt.title('EZ')
            # plt.show()


            if em_term is not None:
                em_term = em_term.add(E, compress_level=compress_level2)
            else:
                em_term = E.copy() if E is not None else None

        # print('is ion?', is_ion, matl_params.e * matl_params.Z / matl_params.mass)
        if em_term is not None:
            em_term = em_term.scalar_multiply(matl_params.e * matl_params.Z / matl_params.mass, inplace=False)
            if self.compress_F:
                em_term.compress(compress_opts=self.compress_F_opts, inplace=True)

            ## set which axes are constant
            if background_force and not internal_force:  ## background only
                # print('background force ONLY')
                for C in em_term.componentIDs:
                    E0_C = None if background_E0 is None else background_E0.components.get(C, None)
                    B0_C = None if background_B0 is None else background_B0.components.get(C, None)
                    if (E0_C is None or E0_C.is_constant) and (B0_C is None or B0_C.is_constant):
                        vC_ax = coords_v.get_axis(C.type)  ## direction of advection (df/dv axis)
                        em_term[C].constant_axes = [ax for ax in self.grid_X.axes] + [vC_ax]
                        # print('set em term constant axes', C, em_term[C].constant_axes)
                        # exit()

                    # v_ax = coords_v.get_axis(C.type)
                    # em_data = em_term.get_comp_data(C, ax_select={**{x_ax: 0 for x_ax in self.grid_X.axes},})
                    # print('??')
                    # if em_data is not None:
                    #     print(em_data.shape)
                    #     plt.figure()
                    #     plt.imshow(em_data)
                    #     plt.colorbar()
                    #     plt.title(f'F{C}')
                    #     plt.show()

        if verbose_plot:
            print('get force term plot')
            x_coords = self.coords_x.coords
            # ax_x, ax_y, ax_vx, ax_vy, ax_vz = em_term[em_term.componentIDs[0]].grid.axes

            ax_x, ax_vx, ax_vy, ax_vz = em_term[em_term.componentIDs[0]].grid.axes
            for x_c in x_coords:
                em_term_data = em_term.get_comp_data(x_c, ax_select={# ax_x: ax_x.npts // 2,
                                                                     ax_vx: ax_vx.npts // 2,
                                                                     ax_vy: ax_vy.npts // 2,
                                                                     # ax_vz: ax_vz.npts//2,
                                                                     })
                if em_term_data is None or np.all(em_term_data == 0):
                    print('em term data is None', x_c)
                    continue

                if em_term_data.ndim == 2:
                    plt.figure()
                    plt.imshow(np.real(em_term_data))
                    plt.title(f'em {x_c} re')
                    plt.xlabel('vz')
                    plt.ylabel('y')
                    plt.colorbar()

                    plt.figure()
                    plt.imshow(np.imag(em_term_data))
                    plt.title(f'em {x_c} im')
                    plt.xlabel('vz')
                    plt.ylabel('y')
                    plt.colorbar()

                elif em_term_data.ndim == 1:
                    print('em_term', em_term[x_c])
                    plt.figure()
                    plt.plot(np.real(em_term_data), label='re')
                    plt.plot(np.imag(em_term_data), label='im')
                    plt.title(f'em {x_c}')
                    plt.xlabel('vz')
                    plt.ylabel('y')
                    plt.legend()
            plt.show()

        # print('E + v x B term', em_term.components)
        # for compID, comp in em_term.components.items():
        #     print('compID', compID, comp.frobenius_norm())
        #
        # plt.figure()
        # for compID, comp in self.EM_sys.E.components.items():
        #     print('E compID', compID, comp.frobenius_norm())
        #     if comp.data is not None:
        #         plt.plot(comp.get_data())
        # plt.show()
        #
        # plt.figure()
        # for compID, comp in self.EM_sys.B.components.items():
        #     print('E compID', compID, comp.frobenius_norm())
        #     if comp.data is not None:
        #         plt.plot(comp.get_data())
        # plt.show()
        #
        # for compID, comp in velocities.components.items():
        #     print('v compID', compID, comp.frobenius_norm())
        #
        # raise RuntimeError

        return em_term

    def compute_force_term_boris(self, dt, is_ion=False, compress_level=1, compress_level1=0, compress_level2=0,
                                 verbose_plot=False, background_force=True, internal_force=True, **kwargs):
        """ compute "advec coeffs" for EM force:  q/m (E + v x B)
            boris update:  semi-implicit update of v:  v[n+1] - v[n] / dt = q/m (E + (v[n+1]+v[n])/2 x B)
            translate v[n+1] into term compatible with force advec coeff = (v[n+1] - v[n]) / dt

            coeff = dt / 2 * q / m
            t = coeff * B
            s = 2 t / (1 + |t|^2)

            v- = v[n] + coeff * E
            v+ = v[n+1] - coeff * E

            v' = v- + v- x t
            v+ = v- + v' x s

            v+ - v- = v' x s = v- x s + v- x t x s
            v[n+1] - v[n] = v+ - v- - 2 * coeff * E

            = [vx] ( vx * (-sy ty - sz tz) + vy * (sz + sy tx) + vz (-sy + tx sz) )
              [vy] ( vx * (-sz + sx ty) + vy (-tx sx - sz tz) + vz (sx + ty sz) )
              [vz] ( vx * (sy + sx tz) + vy (-sx + tz sy) + vz (-tx sx - ty sy) )

            include backward trajectory, because the updates along each v direction are not done simultaneously?
        """
        print('boris force')
        coords_v = self.coords_vi if is_ion else self.coords_ve
        velocities = self.velocities_i if is_ion else self.velocities_e  ## MPOs; i think it should still work
        matl_params = self.ion_params if is_ion else self.elc_params

        background_B0 = self.EM_sys.background_B0 if background_force else None
        background_E0 = self.EM_sys.background_E0 if background_force else None

        # print('compress', compress_level, compress_level1, compress_level2)

        if self.B is not None and internal_force:
            B = self.B.add(background_B0, compress_level=compress_level2)
        else:
            B = background_B0

        if matl_params.is_cgs:
            B = B.scalar_multiply(1. / matl_params.c, inplace=False)

        if self.E is not None and internal_force:
            E = self.E.add(background_E0, compress_level=compress_level2)
        else:
            E = background_E0

        coeff = dt * matl_params.charge / 2 / matl_params.mass
        tot_V = np.prod([ax.length() + ax.dx for ax in B.grid.axes])
        B_mag = np.sqrt(B.norm() ** 2 / tot_V)
        # print('matl params', matl_params.charge, matl_params.mass)
        # print('B field norm', B.norm()**2, tot_V, B_mag)
        # print('E', E.componentIDs, B.componentIDs, velocities.componentIDs)
        t_field = B.scalar_multiply(-coeff, inplace=False)
        s_field = B.scalar_multiply(2 * coeff / (1 + (coeff * B_mag) ** 2))
        # print('t field norm', dt, t_field.norm() / np.sqrt(tot_V) / dt, s_field.norm() / np.sqrt(tot_V) / dt)

        ### how to include backpropagation?

        ## calculate v^[n+1] from v^[n] using Lorentz force
        # vm_field = velocities.add( E.scalar_multiply(coeff, inplace=False), inplace=False ) if E is not None \
        #             else velocities.copy()
        vm_field = velocities.copy()

        v1_field = vm_field.add(vm_field.cross_product(t_field, self.coords_x), inplace=False)
        v2_field = vm_field.add(v1_field.cross_product(s_field, self.coords_x), inplace=False)  # vp

        # v_next = v2_field.add( E.scalar_multiply(coeff, inplace=False), inplace=False ) if E is not None else v2_field
        v_next = v2_field
        v_diff = v_next.add(velocities.scalar_multiply(-1))

        if E is not None:
            v_diff = v_diff.add(E.scalar_multiply(2 * coeff, inplace=False))

        v_diff = v_diff.scalar_multiply(1 / dt, inplace=False)
        ## divide by dt to make analogous to advec coeff

        """
        v+ - v- = [vx] ( vx * (-sy ty - sz tz) + vy * (sz + sy tx) + vz (-sy + tx sz) )
                  [vy] ( vx * (-sz + sx ty) + vy (-tx sx - sz tz) + vz (sx + ty sz) )
                  [vz] ( vx * (sy + sx tz) + vy (-sx + tz sy) + vz (-tx sx - ty sy) )
                  - 2 * coeff * E
        """
        # ### assumes constant B_field
        # X, Y, Z = self.coords_x.coords
        # tx, ty, tz = np.array(t_field.norms([X, Y, Z])) / np.sqrt(tot_V) * np.sign(coeff) * -1
        # sx, sy, sz = np.array(s_field.norms([X, Y, Z])) / np.sqrt(tot_V) * np.sign(coeff)
        # vel_x = velocities.components.get(X, None)
        # v_diff_x, v_diff_y, v_diff_z = None, None, None
        # if vel_x is not None:
        #     v_diff_x = vel_x.scalar_multiply(-sy * ty - sz * tz)
        #     v_diff_y = vel_x.scalar_multiply(-sz + sx * ty)
        #     v_diff_z = vel_x.scalar_multiply(sy + sx * tz)
        #
        # vel_y = velocities.components.get(Y,None)
        # if vel_y is not None:
        #     v_diff_xy = vel_y.scalar_multiply(sz + sy * tx)
        #     v_diff_yy = vel_y.scalar_multiply(-tx * sx - sz * tz)
        #     v_diff_zy = vel_y.scalar_multiply(-sx + tz * sy)
        #
        #     v_diff_x = v_diff_xy if v_diff_x is None else v_diff_x.add(v_diff_xy, inplace=True)
        #     v_diff_y = v_diff_yy if v_diff_y is None else v_diff_y.add(v_diff_yy, inplace=True)
        #     v_diff_z = v_diff_zy if v_diff_z is None else v_diff_z.add(v_diff_zy, inplace=True)
        #
        # vel_z = velocities.components.get(Z,None)
        # if vel_z is not None:
        #     v_diff_xz = vel_z.scalar_multiply(-sy + sz * tx)
        #     v_diff_yz = vel_z.scalar_multiply(sx + sz * ty)
        #     v_diff_zz = vel_z.scalar_multiply(-tx * sx - ty * sy)
        #
        #     v_diff_x = v_diff_xz if v_diff_x is None else v_diff_x.add(v_diff_xz, inplace=True)
        #     v_diff_y = v_diff_yz if v_diff_y is None else v_diff_y.add(v_diff_yz, inplace=True)
        #     v_diff_z = v_diff_zz if v_diff_z is None else v_diff_z.add(v_diff_zz, inplace=True)
        #
        # v_diff = Field('vdiff', velocities.grid, data={X: v_diff_x, Y: v_diff_y, Z: v_diff_z})
        # if E is not None:
        #     v_diff = v_diff.add( E.scalar_multiply(2*coeff) )
        # v_diff = v_diff.scalar_multiply(1 / dt, inplace=False)

        ## set which axes are constant
        if background_force and not internal_force:  ## background only
            # print('background force ONLY')
            for C in v_diff.componentIDs:
                E0_C = None if background_E0 is None else background_E0.components.get(C, None)
                B0_C = None if background_B0 is None else background_B0.components.get(C, None)
                if (E0_C is None or E0_C.is_constant) and (B0_C is None or B0_C.is_constant):
                    if v_diff[C] is not None:
                        v_diff[C].constant_axes = [ax for ax in self.grid_X.axes]
                    ## or, union of E0_C, B0_C constant axes?

                # v_ax = coords_v.get_axis(C.type)
                # em_data = v_diff.get_comp_data(C, ax_select={**{x_ax: 0 for x_ax in self.grid_X.axes},
                #                                              #   v_ax: v_ax.npts // 2
                #                                              })
                # print('??')
                # if em_data is not None:
                #     print(em_data.shape)
                #     plt.figure()
                #     plt.imshow(em_data)
                #     plt.colorbar()
                #     plt.title(f'F{C}')
                #     plt.show()

        return v_diff

    def compute_force_term_bg_SL(self, dt, is_ion=False, sl_order=DEFAULT_SL_ORDER, compress_level=1, compress_level1=0,
                                 compress_level2=0, verbose_plot=False):
        """ assumes field is constant in space
        """
        sys_f = self.sys_fi if is_ion else self.sys_fe
        coords_v = self.coords_vi if is_ion else self.coords_ve
        velocities = self.velocities_i if is_ion else self.velocities_e
        spec_params = self.ion_params if is_ion else self.elc_params
        background_B0 = self.EM_sys.background_B0
        background_E0 = self.EM_sys.background_E0

        # em_SL_dict = {} # sys_f.force_term_bg_SL
        mult_const = spec_params.e * spec_params.Z / spec_params.mass

        print('compute bg SL EM', )

        if background_E0 is not None or background_B0 is not None:

            # if em_SL_dict is None:  ## compute these terms

            if background_B0 is None:  ## E0 only
                em_SL_dict = {}
                for C in background_E0.componentIDs:
                    EC = background_E0[C]
                    v0_ax = coords_v.get_axis(C.type)
                    CV = v0_ax.coordinate
                    if EC.data is not None:
                        if EC.is_constant:
                            select_gtn = EC.grid.get_select_elems_mps([0])
                            E0_val = EC.ovlp(select_gtn) * mult_const
                            em_SL_dict[CV] = helper_sl.get_cell_data(dt, v0_ax.dx, E0_val, sl_order=sl_order)
                        else:
                            E0_data = background_E0.get_comp_data(C) * mult_const
                            em_SL_dict[CV] = helper_sl.get_cell_data(dt, v0_ax.dx, E0_data, sl_order=sl_order)
                # sys_f.force_term_bg_SL = em_SL_dict

            # elif background_E0 is None:
            else:

                em_SL_dict = {}
                comps = self.coords_x.coords
                for i in range(3):
                    C0 = comps[i]
                    C1, C2 = comps[(i + 1) % 3], comps[(i + 2) % 3]

                    v0_ax = coords_v.get_axis(C0.type)
                    if v0_ax is None:
                        continue

                    CV = v0_ax.coordinate

                    B1s = background_B0.components.get(C1, None)
                    if B1s is not None:
                        B1s = B1s.scalar_multiply(-1)
                    B2s = background_B0.components.get(C2, None)

                    # v1_ax = coords_v.get_axis(C1.type)
                    # v2_ax = coords_v.get_axis(C2.type)

                    v1s = velocities.components.get(C1, None)
                    v2s = velocities.components.get(C2, None)

                    comp1 = None if v1s is None else v1s.elemental_multiply(B2s, compress=4)
                    comp2 = None if v2s is None else v2s.elemental_multiply(B1s, compress=4)
                    comp = comp2 if comp1 is None else comp1.add(comp2, compress=4)

                    E0 = None if background_E0 is None else background_E0.components.get(C0, None)
                    if E0 is not None:
                        comp = comp.add(E0, compress=4)
                        print('added E0')

                    # grid_class = sys_f.f.grid.__class__
                    # # print(grid_class, issubclass(grid_class, GridsComb))
                    # if issubclass(grid_class, CompositeGrid):
                    #     vcoeff_axes = [v for v in coords_v.axes if v != v0_ax]
                    #     vcoeff_grids = [sys_f.f.grid.get_axis_grid(v) for v in vcoeff_axes]
                    #     coeff_grid = sys_f.f.grid.__class__(f'V!{v0_ax}', vcoeff_grids)
                    # else:
                    #     vcoeff_axes = [v for v in coords_v.axes if v != v0_ax]
                    #     coeff_grid = sys_f.f.grid.__class__(f'V!{v0_ax}',vcoeff_axes)
                    vcoeff_axes = [v for v in coords_v.axes if v != v0_ax]
                    coeff_grid = sys_f.f.grid.get_subgrid(vcoeff_axes)

                    # print('ax select', {x_ax: 0 for x_ax in self.coords_x.axes})

                    def is_constant(gtn):
                        return gtn is None or gtn.is_constant

                    if is_constant(E0) and is_constant(B1s) and is_constant(B2s):
                        ax_select = {**{x_ax: 0 for x_ax in self.coords_x.axes}, v0_ax: 0}
                    else:
                        ax_select = {v0_ax: 0}
                    comp_data = comp.get_data(ax_select=ax_select)
                    ## assumes B0 is constant in space

                    # print('BG comp data', C0, vcoeff_axes)

                    # plt.figure()
                    # plt.plot(comp_data)
                    # # plt.colorbar()
                    # plt.title('bg comp data')
                    # plt.show()

                    if comp_data is not None:
                        # print('get cell data BG SL', dt, v0_ax, v0_ax.dx, mult_const)
                        em_SL_dict[CV] = helper_sl.get_cell_data(dt, v0_ax.dx, comp_data * mult_const,
                                                                 sl_order=sl_order, coeff_grid=coeff_grid)
                    else:
                        print(f'EM force {C0} is None')
                        # exit()

            # else:
            #     ## E0 + v x B0 -- need to find where E0 > v x B0
            #     raise NotImplementedError

            sys_f.force_term_bg_SL = em_SL_dict
            return em_SL_dict
        else:
            sys_f.force_term_bg_SL = {}
            return {}

    def calculate_time_derivative(self, time=None, compress_level=0, compress_level1=0, compress_level2=0,
                                  do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
                                  update_force=True, verbose_plot=False, **kwargs) -> 'VlasovMaxwell':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        ### jank correction for advection test ####
        print('calc time deriv', time, self.time)
        X, Y, Z = self.coords_x.coords
        print('old Ex', self.EM_sys.E[X].data)
        Ex0, omega = 0.9, 0.4567
        new_Ex = Ex0 * np.cos(omega * time)
        self.EM_sys.E[X].data = new_Ex
        print('new Ex', self.EM_sys.E[X].data)
        update_force = True

        dFdt = super().calculate_time_derivative(time=time, compress_level=compress_level,
                                                 compress_level1=compress_level1, compress_level2=compress_level2,
                                                 do_x_advection=do_x_advection, do_v_advection=do_v_advection,
                                                 background_force=background_force, internal_force=internal_force,
                                                 update_force=update_force, verbose_plot=verbose_plot)
        # fe = self.fe.create_like()
        # fi = self.fi.create_like()
        # dFdt = super().create_like(fe, fi, recalc=False)

        if self.evolve_EM:
            if update_force:  #  and self.evolve_EM:
                self.EM_sys.charge_density = self.compute_charge_density(compress=compress_level1)
                self.EM_sys.current_density = self.compute_current(compress=compress_level1)
            EM_sys_deriv = self.EM_sys.calculate_time_derivative(time=time,
                                                                 compress_level=compress_level,
                                                                 compress_level1=compress_level1,
                                                                 verbose_plot=verbose_plot)

            dFdt.EM_sys = EM_sys_deriv

            # EM_sys_deriv_fields = [EM_sys_deriv.get_field(k) for k in EM_sys_deriv.field_names]
        # else:
        #     EM_sys_deriv_fields = []

        return dFdt

    def check_poisson(self, compress=5, ) -> Optional['ScalarField']:
        self.EM_sys.charge_density = self.compute_charge_density(compress=compress)
        return self.EM_sys.check_poisson(compress=compress)

    def global_rk_cross(self, dt, te_order=3, inplace=False, time=None, background_force=True, internal_force=True,
                        update_force=True, compress_level=1, compress_level1=0,
                        verbose_plot=False, **kwargs) -> 'PDE_system':

        # new_state = super().global_rk_cross(dt, te_order=te_order, inplace=inplace, time=time,
        #                                     background_force=background_force, internal_force=internal_force,
        #                                     update_force=update_force, do_update_V=self.evolve_EM,
        #                                     compress_level=compress_level, compress_level1=compress_level1,
        #                                     verbose_plot=verbose_plot, **kwargs)

        from local_solvers.time_integrator_cross import global_rk_cross

        time = self.time if time is None else time
        dist_mpx = self.fe.component.data
        nsites = 2

        compress_opts = self.fe.compress_config.get_compress_opts(1)
        cutoff = compress_opts.get('cutoff', None)
        max_bond = compress_opts.get('max_bond', None)

        def deriv_func(mps1, time=None, **kwargs):
            return self.deriv_upwind_global(nsites=nsites, ket=mps1, max_bond=max_bond, cutoff=cutoff, time=time)

        compress_opts = self.fe.compress_config.get_compress_opts(1)
        cutoff = compress_opts.get('cutoff', None)
        max_bond = compress_opts.get('max_bond', None)

        print('global rk max bond', max_bond, 'cutoff', cutoff)
        out = global_rk_cross(dt, te_order, dist_mpx, deriv_func, nsites=nsites, max_bond=max_bond,
                              cutoff=cutoff, time=time)

        # print('boltz diff', helper.distance(out, dist_mpx))

        new_state = self if inplace else self.copy()
        new_state.fe.component.data = out

        return new_state

    def deriv_upwind_global(self, nsites=2, ket=None, time:Numeric=None, max_bond: int = None, cutoff: Numeric = None,
                            do_x_advection=True, do_v_advection=True,
                            background_force=True, internal_force=True, get_collisions=False,
                            **kwargs) -> 'qtn.MatrixProductState':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        ### jank correction for advection test ####
        # self.time = None
        # update_force = False
        print('calc time deriv (upwind)', time, self.time)
        time = self.time if time is None else time

        X, Y, Z = self.coords_x.coords
        print('old Ex', self.EM_sys.E[X].data)
        Ex0, omega = 0.9, 0.4567
        new_Ex = Ex0 * np.cos(omega * time)
        self.EM_sys.E[X].data = new_Ex
        print('new Ex', self.EM_sys.E[X].data)
        update_force = True

        dFdt = super().deriv_upwind_global(time=None,
                                           do_x_advection=do_x_advection, do_v_advection=do_v_advection,
                                           background_force=background_force, internal_force=internal_force,
                                           update_force=update_force,)
        # fe = self.fe.create_like()
        # fi = self.fi.create_like()
        # dFdt = super().create_like(fe, fi, recalc=False)

        if self.evolve_EM:
            if update_force:  # and self.evolve_EM:
                self.EM_sys.charge_density = self.compute_charge_density() # compress=compress_level1)
                self.EM_sys.current_density = self.compute_current() #compress=compress_level1)
            EM_sys_deriv = self.EM_sys.calculate_time_derivative(time=time,
                                                                 # compress_level=compress_level,
                                                                 # compress_level1=compress_level1,
                                                                 )

            dFdt.EM_sys = EM_sys_deriv

            # EM_sys_deriv_fields = [EM_sys_deriv.get_field(k) for k in EM_sys_deriv.field_names]
        # else:
        #     EM_sys_deriv_fields = []

        return dFdt.fe.component.data

    # # @profile
    # def _calculate_time_derivative_f(self, compress:int = 1, compress1:int = 0, compress2: int = 0, is_ion=False,
    #                                  verbose_plot=False) -> Optional['Field']:
    #     """ df/dt = ...
    #         dB/dt + curl(E) = 0
    #         e0*mu0 dE/dt - curl(B) = -mu0 J
    #         J = sum_s qs ns vs
    #
    #         df0dv:   grad_v f0 if doing linearized vlasov. list or dict of each velocity component
    #     """
    #     dist = self.fi if is_ion else self.fe
    #     dist0 = self.background_fi0 if is_ion else self.background_fe0
    #     print(dist.component)
    #     # if dist is None:  return None
    #
    #     x_coords = self.coords_x.coords
    #     X, Y, Z = x_coords
    #     v_coordsys = self.coords_vi if is_ion else self.coords_ve
    #     v_coords = [v_coordsys.get_coord(coord.type) for coord in x_coords]
    #     VX, VY, VZ = v_coords
    #     v_axes_dict = {coord: v_coordsys.get_axis(coord.type) for coord in x_coords}
    #     v_axes = [v_axes_dict.get(coord, None) for coord in x_coords]
    #     # print('v_axes', x_coords, v_coords, v_axes_dict, v_axes)
    #
    #     df0dv = self.df0dv_i if is_ion else self.df0dv_e
    #     v_grads = dist.gradient(deriv_axes=v_axes, compress_level=0) if df0dv is None else df0dv
    #     if verbose_plot:
    #         plt.figure()
    #         x_ax, y_ax, vx_ax, vy_ax, vz_ax = dist.component.grid.axes
    #         v_grads_y = v_grads.get_comp_data(VY, ax_select={x_ax: x_ax.npts//2, y_ax:y_ax.npts//2,
    #                                                                vx_ax: vx_ax.npts//2})
    #         if v_grads_y is not None:
    #             plt.imshow(np.real(v_grads_y))
    #             plt.colorbar()
    #             plt.title('v grad y')
    #             plt.show()
    #
    #     ## grad_v of background distribution
    #     v_grads_0 = self.gradv_fi0 if is_ion else self.gradv_fe0
    #     if dist0 is not None and v_grads_0 is None:
    #         v_grads_0 = dist0.gradient(deriv_axes=v_axes, compress_level=compress1)
    #         if is_ion:
    #             self.gradv_fi0 = v_grads_0
    #         else:
    #             self.gradv_fe0 = v_grads_0
    #
    #         if verbose_plot:
    #             plt.figure()
    #             x_ax, y_ax, vx_ax, vy_ax, vz_ax = dist0.component.grid.axes
    #             v_grads_y = v_grads_0.get_comp_data(VY, ax_select={x_ax: x_ax.npts//2, y_ax:y_ax.npts//2,
    #                                                                vx_ax: vx_ax.npts//2})
    #             plt.imshow(np.real(v_grads_y))
    #             plt.colorbar()
    #             plt.title('v grad 0 y')
    #             plt.show()
    #
    #     # print('vgrad0', v_grads_0.componentIDs)
    #     v_grads.add(v_grads_0, inplace=True, compress_level=compress1)
    #
    #
    #     ### advection term
    #     convective_term = self._calculate_time_derivative_f_advection(compress1 = compress1, compress2 = compress2,
    #                                                                   is_ion=is_ion, verbose_plot=verbose_plot)
    #     ### force term
    #     lorentz_term = self._calculate_time_derivative_f_force(v_grads=v_grads,
    #                                                            compress1=compress1, compress2=compress2,
    #                                                            is_ion=is_ion, verbose_plot=verbose_plot)
    #     # exit()
    #
    #     dFdt_s = convective_term.add(lorentz_term, compress_level=0)
    #     dFdt_s.name = self.fi.name if is_ion else self.fe.name
    #
    #     ### get collisions
    #     coll_field = self.get_collision_term(is_ion=is_ion, v_axes=v_axes, v_grads=v_grads,
    #                                          compress1=compress1, compress2=compress2)
    #     dFdt_s.add(coll_field, inplace=True, compress_level=0)
    #
    #     if compress:
    #         dFdt_s.compress(compress_level=compress)
    #
    #     return dFdt_s
    #
    #
    # def _calculate_time_derivative_f_advection(self, compress1:int = 0, compress2: int = 0,
    #                                            is_ion=False, verbose_plot:bool=False) -> Optional['Field']:
    #     """ df/dt = ...
    #         dB/dt + curl(E) = 0
    #         e0*mu0 dE/dt - curl(B) = -mu0 J
    #         J = sum_s qs ns vs
    #
    #         df0dv:   grad_v f0 if doing linearized vlasov. list or dict of each velocity component
    #     """
    #     dist = self.fi if is_ion else self.fe
    #     dist0 = self.background_fi0 if is_ion else self.background_fe0
    #
    #     # if dist is None:  return None
    #
    #     x_axes = self.coords_x.axes
    #     v_coordsys = self.coords_vi if is_ion else self.coords_ve
    #     v_axes_dict = {ax: v_coordsys.get_axis(ax.coordinate.type) for ax in x_axes}
    #     v_axes = [v_axes_dict.get(ax, None) for ax in x_axes]
    #
    #     if not self.upwind:
    #         v_axes_dict = None
    #
    #     x_grads = dist.gradient(deriv_axes=x_axes, upwind_axes=v_axes_dict, compress_level=compress1)
    #
    #     if verbose_plot:
    #         ax_x, ax_y, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
    #         dist_data = dist.get_comp_data(ax_select={# ax_x: ax_x.npts//2,
    #                                                    ax_vx: ax_vx.npts//2,
    #                                                    ax_vy:ax_vy.npts//2,
    #                                                    ax_vz: ax_vz.npts//2})
    #         xg_data = x_grads.get_comp_data(ax_y.coordinate,
    #                                         ax_select={ax_vx: ax_vx.npts//2, ax_vy:ax_vy.npts//2,
    #                                                    ax_vz: ax_vz.npts//2})
    #
    #         plt.figure()
    #         plt.imshow(dist_data)
    #         plt.xlabel('y')
    #         plt.ylabel('x')
    #         plt.title(f'dist, is ion {is_ion}')
    #         plt.colorbar()
    #
    #         plt.figure()
    #         plt.plot(dist_data[0,:])
    #         plt.title(f'dist, is ion {is_ion}')
    #
    #         plt.show()
    #
    #         plt.figure()
    #         plt.imshow(xg_data)
    #         plt.xlabel('x')
    #         plt.ylabel('y')
    #         plt.title('xgrad')
    #         plt.colorbar()
    #         # plt.show()
    #
    #         plt.figure()
    #         plt.plot(xg_data[0,:],label='gradx')
    #         plt.xlabel('x')
    #         plt.legend()
    #         plt.show()
    #
    #     exit()
    #
    #     ## grad_v of background distribution
    #     x_grads_0 = self.gradx_fi0 if is_ion else self.gradx_fe0
    #     if dist0 is not None and x_grads_0 is None:
    #         x_grads_0 = dist0.gradient(deriv_axes=x_axes, upwind_axes=v_axes_dict, compress_level=compress1)
    #         if is_ion:
    #             self.gradx_fi0 = x_grads_0
    #         else:
    #             self.gradx_fe0 = x_grads_0
    #     x_grads.add(x_grads_0, inplace=True)
    #
    #
    #     ## intermediate plots
    #     if verbose_plot:
    #         ax_x, ax_y, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
    #         dist_data = dist0.get_comp_data(ax_select={# ax_x: ax_x.npts//2,
    #                                                    ax_vx: ax_vx.npts//2,
    #                                                    ax_vy:ax_vy.npts//2,
    #                                                    ax_vz: ax_vz.npts//2})
    #         xg_data = x_grads_0.get_comp_data(ax_y.coordinate,
    #                                         ax_select={ax_vx: ax_vx.npts//2, ax_vy:ax_vy.npts//2,
    #                                                    ax_vz: ax_vz.npts//2})
    #
    #         plt.figure()
    #         plt.imshow(dist_data)
    #         plt.xlabel('y')
    #         plt.ylabel('x')
    #         plt.title(f'dist 0, is ion {is_ion}')
    #         plt.colorbar()
    #
    #         # plt.figure()
    #         # plt.plot(np.sum(dist_data,axis=1)*self.grid_X.dx)
    #         # plt.xlabel('x')
    #         #
    #         # plt.figure()
    #         # plt.plot(dist_data[0,:])
    #         # plt.ylabel('v')
    #
    #         plt.show()
    #
    #         plt.figure()
    #         plt.imshow(xg_data)
    #         plt.xlabel('x')
    #         plt.ylabel('y')
    #         plt.title('xgrad 0')
    #         plt.colorbar()
    #         # plt.show()
    #
    #         plt.figure()
    #         plt.plot(xg_data[0,:],label='gradx')
    #         plt.xlabel('x')
    #         plt.legend()
    #         plt.show()
    #
    #     ## convective term
    #     print('x_grads', x_grads.componentIDs)
    #     print(v_axes, self.coords_x.coords)
    #     convective_term = x_grads.xdot(v_axes, compIDs=self.coords_x.coords, compress_level=compress1)
    #     convective_term.scalar_multiply(-1, inplace=True)
    #
    #     if verbose_plot:
    #         plt.figure()
    #         plt.imshow(convective_term.get_comp_data(ax_select={ax_vx: ax_vx.npts//2,
    #                                                             ax_vy:ax_vy.npts//2,
    #                                                             ax_vz: ax_vz.npts//2}))
    #         plt.colorbar()
    #         plt.xlabel('y')
    #         plt.ylabel('x')
    #         plt.title('convective term')
    #         plt.show()
    #         # exit()
    #
    #     return convective_term
    #
    #
    # # @profile
    # def _calculate_time_derivative_f_force(self, v_grads = None,
    #                                        compress1: int = 0, compress2: int = 0,
    #                                        is_ion=False, verbose_plot:bool=False) -> Optional['Field']:
    #
    #     dist = self.fi if is_ion else self.fe
    #     dist0 = self.background_fi0 if is_ion else self.background_fe0
    #
    #     if dist is None and dist0 is None:  return None
    #
    #     # x_axes = self.coords_x.axes
    #     # v_coordsys = self.coords_vi if is_ion else self.coords_ve
    #     # v_axes_dict = {ax: v_coordsys.get_axis(ax.coordinate.type) for ax in x_axes}
    #     # v_axes = [v_axes_dict.get(ax, None) for ax in x_axes]
    #
    #     x_axes = self.coords_x.axes
    #     x_coords = self.coords_x.coords
    #     v_coordsys = self.coords_vi if is_ion else self.coords_ve
    #     v_coords = [v_coordsys.get_coord(coord.type) for coord in x_coords]
    #     v_axes_dict = {coord: v_coordsys.get_axis(coord.type) for coord in x_coords}
    #     v_axes = [v_axes_dict.get(coord, None) for coord in x_coords]
    #     # print('x coords', x_coords, v_coords, v_axes)
    #
    #     ## grad_v of main distribution
    #     if v_grads is None:
    #         df0dv = self.df0dv_i if is_ion else self.df0dv_e
    #         v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress2) if df0dv is None else df0dv
    #
    #         ## grad_v of background distribution
    #         v_grads_0 = self.gradv_fi0 if is_ion else self.gradv_fe0
    #         if dist0 is not None and v_grads_0 is None:
    #             v_grads_0 = dist0.gradient(deriv_axes=v_axes, compress_level=compress1)
    #             if is_ion:
    #                 self.gradv_fi0 = v_grads_0
    #             else:
    #                 self.gradv_fe0 = v_grads_0
    #         v_grads.add(v_grads_0, inplace=True, compress_level=compress1)
    #
    #
    #     ## intermediate plots
    #     if verbose_plot:
    #         ax_x, ax_y, ax_vx, ax_vy, ax_vz = dist.component.grid.axes
    #         dist_data = dist.get_comp_data(ax_select={# ax_x: ax_x.npts//2,
    #                                                   ax_vx: ax_vx.npts//2,
    #                                                   ax_vy:ax_vy.npts//2,
    #                                                   ax_vz: ax_vz.npts//2})
    #         vg_data = v_grads.get_comp_data(ax_vx.coordinate,
    #                                         ax_select={ax_x: ax_x.npts//2,
    #                                                    ax_vx: ax_vx.npts//2,
    #                                                    # ax_vy:ax_vy.npts//2,
    #                                                    ax_vz: ax_vz.npts//2})
    #
    #         plt.figure()
    #         plt.imshow(np.real(vg_data))
    #         plt.xlabel('vy')
    #         plt.ylabel('y')
    #         plt.colorbar()
    #         plt.title('grad v')
    #         plt.show()
    #
    #         plt.figure()
    #         plt.plot(np.real(vg_data[ax_y.npts//2,:]),label='gradv')
    #         plt.xlabel('vy')
    #         plt.legend()
    #         plt.show()
    #         # exit()
    #
    #     ## (EM) force term
    #     em_term = self.compute_force_term(compress_level1=compress1, compress_level2=0, is_ion=is_ion,
    #                                       verbose_plot=verbose_plot)
    #     if em_term is not None:
    #         em_term.scalar_multiply(-1, inplace=True)
    #
    #     if em_term is not None:
    #         em_term = em_term.pad_to_new_grid(new_grid=dist.grid)
    #         print('em term', em_term.max_bonds())
    #         print('v grads', v_grads.max_bonds())
    #         # lorentz_term = v_grads.dot(em_term, comps=[ax.coordinate for ax in v_axes],
    #         #                            other_comps=[ax.coordinate for ax in x_axes], zipup=self.zipup,
    #         #                            compress_level=compress1, inner_compress_level=compress2)    # 5, 0
    #         lorentz_term = v_grads.dot(em_term, comps=v_coords,
    #                                    other_comps=x_coords, zipup=self.zipup,
    #                                    compress_level=compress1, inner_compress_level=compress2)  # 5, 0
    #     else:
    #         lorentz_term = None
    #
    #     if verbose_plot:
    #         plt.figure()
    #         lorentz_data = lorentz_term.get_comp_data(ax_select={ax_vx: ax_vx.npts//2,
    #                                                          ax_vy: ax_vy.npts//2,
    #                                                          ax_vz: ax_vz.npts//2})
    #         plt.imshow(np.real(lorentz_data))
    #         plt.title('lorentz')
    #         plt.colorbar()
    #         plt.show()
    #
    #         exit()
    #
    #     return lorentz_term
    #
    #
    # def get_collision_term(self, is_ion=False, v_axes:Sequence['Axis']=None, v_grads: 'Field'=None,
    #                        compress1=0, compress2=0):
    #     """ obtain collision term
    #     """
    #     dist = self.fi if is_ion else self.fe
    #     dist0 = self.background_fi0 if is_ion else self.background_fe0
    #
    #     if v_axes is None:
    #         x_axes = self.coords_x.axes
    #         v_coordsys = self.coords_vi if is_ion else self.coords_ve
    #         v_axes_dict = {ax: v_coordsys.get_axis(ax.coordinate.type) for ax in x_axes}
    #         v_axes = [v_axes_dict.get(ax, None) for ax in x_axes]
    #
    #     if self.collision.coll_type is None:
    #         return None
    #
    #     elif self.collision.coll_type == CollisionType.LB:
    #
    #         if dist.is_sqrt:   raise NotImplementedError('LB operator not implemented for sqrt field')
    #
    #         if v_grads is None:
    #             df0dv = self.df0dv_i if is_ion else self.df0dv_e  # for linearized Vlasov
    #             v_grads = dist.gradient(deriv_axes=v_axes, compress_level=compress1) \
    #                 if df0dv is None else df0dv
    #
    #         ## grad_v of background distribution
    #         v_grads_0 = self.gradv_fi0 if is_ion else self.gradv_fe0
    #         if dist0 is not None and v_grads_0 is None:
    #             v_grads_0 = dist0.gradient(deriv_axes=v_axes, compress_level=compress1)
    #         v_grads.add(v_grads_0, inplace=True)
    #
    #         ############  get maxwellian ######
    #         if is_ion:
    #             vth2 = self.ion_params.vth ** 2
    #             coll_rate = self.collision.coll_rate_i
    #         else:
    #             vth2 = self.elc_params.vth ** 2
    #             coll_rate = self.collision.coll_rate_e
    #
    #         ## alternate method of computing collisions
    #         # v_drift = self.v_from_v0_i if is_ion else self.v_from_v0_e
    #         # coll = v_drift.elemental_multiply(dist[0], compress_level=0)  ## components are self.v_axes
    #         v0 = self.collision.v0_i if is_ion else self.collision.v0_e
    #         try:
    #             v_offsets = {v_ax: -v0[v_ax.coordinate] for v_ax in v_axes}
    #         except (IndexError, TypeError):
    #             v_offsets = {v_ax: -v0 for v_ax in v_axes}
    #         coll = dist.xmultiply(v_axes, offsets=v_offsets, )
    #         # exit()
    #
    #         coll_1 = v_grads.scalar_multiply(vth2)  ## components are self.v_axes
    #         coll.add(coll_1, inplace=True, compress_level=0)
    #         # print('coll deriv params', coll[v_axes[0].coordinate].ax_deriv_configs)
    #         # coll.update_comp_deriv_params(axes=v_axes)
    #
    #         coords_v = self.coords_vi if is_ion else self.coords_ve
    #         coll_field = coll.divergence(coords_v, compress_level=compress1, inner_compress_level=compress2)
    #
    #         coll_field.scalar_multiply(coll_rate, inplace=True)
    #
    #         return coll_field
    #
