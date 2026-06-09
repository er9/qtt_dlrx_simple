"""Vlasov equation model.

:class:`Vlasov` (a :class:`~pde_system.PDE_system`) solves the collisionless kinetic
equation ``df/dt + v.grad_x(f) + F.grad_v(f) = 0`` for one or more species on a QTT
phase-space grid. Base class for the electromagnetic (:mod:`pde_vlasovEM`) and
electrostatic (:mod:`pde_vlasovES`) closures.
"""

from setup_.configs import *
import matplotlib.pyplot as plt
import helper_quimb as helper
from gridTN import GridTN
from field import Field, ScalarField
from pde_system import PDE_system
from pde_boltzmann import Boltzmann

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from axis import Axis
    from grid import Grid

class Vlasov(PDE_system):
    """ df/dt + v grad(f) + F grad_v(f) = 0
    """
    def __init__(self,
                 sys_fe: Optional[Union['ScalarField','Boltzmann']],
                 sys_fi: Optional[Union['ScalarField','Boltzmann']],
                 # sys_fe: Optional['ScalarField'],
                 # sys_fi: Optional['ScalarField'],
                 grid_X: Optional['Grid']=None,
                 coords_x: Optional['CoordinateSystem'] = None,
                 coords_ve: Optional['CoordinateSystem'] = None,
                 coords_vi: Optional['CoordinateSystem'] = None,
                 elc_params: Optional['SpeciesConfiguration'] = None,
                 ion_params: Optional['SpeciesConfiguration'] = None,
                 f0_gradv_e: Optional['Field'] = None,
                 f0_gradv_i: Optional['Field'] = None,
                 background_fe0: Optional[Union['ScalarField','Boltzmann']] = None,
                 background_fi0: Optional[Union['ScalarField','Boltzmann']] = None,
                 background_force_neg_e: Optional['Field'] = None,
                 background_force_neg_i: Optional['Field'] = None,
                 normalize=True, upwind=False, zipup=False, evolve_ion=True,
                 conservative=True,
                 te_order=4, compress_levels=None,
                 ):
        """ dist_e:  distribution of electrons (scalar Field obj). generally on x,v grid
            dist_i:  distribution of ions (scalar Field obj). generally on x,v grid
            potential:  electric potential (scalar Field obj). generally on x grid
            velocity_grid: velocity grid (vector Field obj) with components vx, vy, ...
                           generally on v grid
            f0_gradv:   grad_v(f0) (vector Field obj) if doing linearized vlasov.
            x_axes:  GRID axes (0,...,self.ndim-1) corresponding to spatial positions in fe, fi
        """
        if isinstance(sys_fe, ScalarField):
            fe = sys_fe
            sys_fe = Boltzmann(fe, grid_X=grid_X, coords_x=coords_x, coords_v=coords_ve, matl_params=elc_params,
                               background_f=background_fe0, background_force_neg=background_force_neg_e,
                               f0_gradv=f0_gradv_e, normalize=normalize, upwind=upwind, zipup=zipup,
                               conservative=conservative,
                               te_order=te_order, compress_levels=compress_levels)
        elif isinstance(sys_fe, Boltzmann):
            fe = sys_fe.f
            if background_fe0 is not None:
                sys_fe.background_f = background_fe0
        elif sys_fe is None:
            fe = None
        else:
            raise TypeError('sys_fe should be ScalarField or Boltzmann object')

        if isinstance(sys_fi, ScalarField):
            fi = sys_fi
            sys_fi = Boltzmann(fi, grid_X=sys_fe.grid_X, coords_x=coords_x, coords_v=coords_vi, matl_params=ion_params,
                               background_f=background_fi0, background_force_neg=background_force_neg_i,
                               f0_gradv=f0_gradv_i, normalize=normalize, upwind=upwind, zipup=zipup,
                               conservative=conservative,
                               te_order=te_order, compress_levels=compress_levels)
        elif isinstance(sys_fi, Boltzmann):
            fi = sys_fi.f
            if background_fi0 is not None:
                sys_fi.background_f = background_fi0
        elif sys_fi is None:
            fi = None
        else:
            raise TypeError('sys_fe should be ScalarField or Boltzmann object')


        self.names = {}
        if fe is not None and fi is not None and fe.name == fi.name:
            fe.name += 'e'
            fi.name += 'i'
        self.names['fe'] = fe.name if fe is not None else 'fe'
        self.names['fi'] = fi.name if fi is not None else 'fi'

        field_names = [self.names[x] for x in ['fe', 'fi']]

        self.sys_fe = sys_fe
        self.sys_fi = sys_fi

        super().__init__(# fe, fi,
                         field_names=field_names,
                         normalize=normalize, te_order=te_order, compress_levels=compress_levels,
                         conservative=conservative,
                         # field_compress_config=field_compress_config,
                         )

        self.coords_x  = sys_fe.coords_x if sys_fe is not None else sys_fi.coords_x
        self.coords_ve = sys_fe.coords_v if sys_fe is not None else coords_ve
        self.coords_vi = sys_fi.coords_v if sys_fi is not None else coords_vi
        self.grid_X = sys_fe.grid_X

        self.x_axes = sys_fe.x_axes
        self.ve_axes_dict = self.sys_fe.v_axes_dict
        self.vi_axes_dict = self.sys_fi.v_axes_dict

        self.elc_params = self.sys_fe.matl_params
        self.ion_params = self.sys_fi.matl_params

        self.velocities_e, self.velocities_i = None, None
        if self.sys_fe is not None:
            self.velocities_e = self.sys_fe.velocities
            self.velocities_e.name = 'vel_e'
        if self.sys_fi is not None:
            self.velocities_i = self.sys_fi.velocities
            self.velocities_i.name = 'vel_i'

        self.background_p0 = None
        self.background_j0 = None

        # self.evolve_background = sys_fe.evolve_background or sys_fi.evolve_background
        self.evolve_ion = evolve_ion
        self.upwind = upwind
        self.zipup = zipup
        self.semiimplicit_force = False

        self.compress_F = True
        self.compress_F_opts = {} # {'max_bond': None, 'cutoff': CUTOFF, 'cutoff_mode': CUTOFF_MODE}

        # ## positive and negative values of background (E + v x B)
        # self.em0_neg = None  # abs(neg val)
        # self.em0_pos = None

        self._collision = CollisionConfiguration(None)
        self.coll_dist0e = None
        self.coll_dist0i = None
        # self.v_from_v0_e = None   # data_ve - vdrift_e
        # self.v_from_v0_i = None   # data_vi - vdrift_i


    @property
    def fe(self) -> Optional['ScalarField']:
        return self.sys_fe.f

    @fe.setter
    def fe(self, new_field: 'ScalarField'):
        # self.fields[self.names['fe']] = new_field
        self.sys_fe.f = new_field

    @property
    def fi(self) -> Optional['ScalarField']:
        return self.sys_fi.f

    @fi.setter
    def fi(self, new_field: 'ScalarField'):
        # self.fields[self.names['fi']] = new_field
        self.sys_fi.f = new_field

    # @property
    # def fields:
    #     return {self.names['fe']: self.fe, self.names['fi']: self.fi}

    def set_field(self, field_name, new_field):
        """ return ith component of the field. for convenience
                """
        if field_name == self.names['fe']:
            self.fe = new_field
        elif field_name == self.names['fi']:
            self.fi = new_field
        else:
            return self._fields[field_name]


    def get_field(self, field_name):
        """ return ith component of the field. for convenience
                """
        if field_name == self.names['fe']:
            return self.fe
        elif field_name == self.names['fi']:
            return self.fi
        else:
            return self._fields[field_name]

    @property
    def background_fe(self) -> Optional['Boltzmann']:
        return self.sys_fe.background_f

    @background_fe.setter
    def background_fe(self, f0: Optional[Union['ScalarField', 'Boltzmann']]):
        self.sys_fe.background_f = f0
        self.background_j0 = None
        self.background_p0 = None

    @property
    def background_fi(self) -> Optional['Boltzmann']:
        return self.sys_fi.background_f

    @background_fi.setter
    def background_fi(self, f0: Optional[Union['ScalarField', 'Boltzmann']]):
        self.sys_fi.background_f = f0
        self.background_j0 = None
        self.background_p0 = None

    @property
    def total_fe(self) -> Optional['ScalarField']:
        f = self.fe
        f = self.background_fe if f is None else f.add(self.background_fe)
        return f

    @property
    def total_fi(self) -> Optional['ScalarField']:
        f = self.fi
        f = self.background_fi if f is None else f.add(self.background_fi)
        return f

    @property
    def collision(self) -> Optional['CollisionConfiguration']:
        return self._collision


    @collision.setter
    def collision(self, collision_config:'CollisionConfiguration'):
        self._collision = collision_config
        if self.sys_fi is not None:
            self.sys_fi.collision = collision_config.collisions_i
        if self.sys_fe is not None:
            self.sys_fe.collision = collision_config.collisions_e

    @property
    def upwind(self):
        return self.sys_fe.upwind if self.sys_fe is not None else self.sys_fi.upwind

    @upwind.setter
    def upwind(self, do_upwind):
        if self.sys_fe is not None:
            self.sys_fe.upwind = do_upwind
        if self.sys_fi is not None:
            self.sys_fi.upwind = do_upwind

    @property
    def matl_params(self):
        if self.sys_fe is not None:
            return self.sys_fe.matl_params
        elif self.sys_fi is not None:
            return self.sys_fi.matl_params
        return None


    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 2:
            new_fields = new_fields + (None,) * (2 - len(new_fields))

        dist_e, dist_i = new_fields[:2]

        if dist_e is None:
            dist_e = self.sys_fe.f.create_like()
        if dist_i is None:
            dist_i = self.sys_fi.f.create_like()

        if isinstance(dist_e, ScalarField): #  or dist_e is None:
            sys_fe = self.sys_fe.create_like(dist_e, recalc=recalc, deep=deep)
        elif isinstance(dist_e, Boltzmann):
            sys_fe = dist_e
        else:
            raise TypeError('dist e should be ScalarField or Boltzmann object')

        if isinstance(dist_i, ScalarField): #  or dist_i is None:
            sys_fi = self.sys_fi.create_like(dist_i, recalc=recalc, deep=deep)
        elif isinstance(dist_i, Boltzmann):
            sys_fi = dist_i
        else:
            raise TypeError('dist e should be ScalarField or Boltzmann object')

        new_system = self.__class__(sys_fe, sys_fi, grid_X=self.grid_X,
                                    evolve_ion=self.evolve_ion, normalize=self.do_normalization,
                                    upwind=self.upwind, zipup=self.zipup,
                                    te_order=self.te_order, compress_levels=self._comp_levels,
                                    conservative=self.conservative,
                                    )

        if dist_e is None:      new_system.names['fe'] = self.names['fe']
        if dist_i is None:      new_system.names['fi'] = self.names['fi']

        new_system.collision = self.collision
        new_system.compress_F = self.compress_F
        new_system.compress_F_opts = self.compress_F_opts

        new_system.background_p0 = self.background_p0
        new_system.background_j0 = self.background_j0

        new_system.semiimplicit_force = self.semiimplicit_force

        new_system.time = self.time

        # new_system.em0_neg = self.em0_neg
        # new_system.em0_pos = self.em0_pos

        # new_system.v_from_v0_i = self.v_from_v0_i
        # new_system.v_from_v0_e = self.v_from_v0_e

        return new_system


    def copy(self):
        """ create a new system like this with new fields
        """
        new_system = super().copy()
        new_system.sys_fe = self.sys_fe.copy()
        new_system.sys_fi = self.sys_fi.copy()

        new_system.sys_fe.gradx_f = self.sys_fe.gradx_f
        new_system.sys_fe.gradv_f = self.sys_fe.gradv_f
        new_system.sys_fe.density = self.sys_fe.density
        new_system.sys_fe.flow = self.sys_fe.flow
        new_system.sys_fe.state_history = self.sys_fe.state_history
        new_system.sys_fe.deriv_history = self.sys_fe.deriv_history

        new_system.sys_fi.gradx_f = self.sys_fi.gradx_f
        new_system.sys_fi.gradv_f = self.sys_fi.gradv_f
        new_system.sys_fi.density = self.sys_fi.density
        new_system.sys_fi.flow = self.sys_fi.flow
        new_system.sys_fi.state_history = self.sys_fi.state_history
        new_system.sys_fi.deriv_history = self.sys_fi.deriv_history

        return new_system


    def normalize(self, target_e=None, target_i=None):
        """ normalized density field
            in place operation
        """
        if self.sys_fi is not None:
            self.sys_fi.normalize(target_i)

        if self.sys_fe is not None:
            self.sys_fe.normalize(target_e)
        return



    def get_conserved_bases(self) -> dict[Any, Sequence['GridTN']]:
        """ return bases that satisfy d/dt sum_f <basis_f|field_f> = 0
        """
        if self.sys_fe.f.is_sqrt:       ## need to conserve L2 norm instead
            return {}

        bases = {}
        ## conservation of mass:
        l1_norm = self.sys_fe.f.grid.get_ones_mps()
        mag = l1_norm.frobenius_norm()
        l1_norm.scalar_multiply(1./mag, inplace=True)

        space_integ = {x_ax: x_ax.get_iden_mps() for x_ax in self.grid_X.axes}
        ## conservation of momentum: compute E x B momentum, set targeted value
        vel_integ_e = {v_ax: v_ax.get_xmultiply_mps(x_power=1) for v_ax in self.sys_fe.v_axes}
        vel_integ_i = {v_ax: v_ax.get_xmultiply_mps(x_power=1) for v_ax in self.sys_fi.v_axes}
        p_e_basis = self.sys_fe.f.grid.make_mps_ndim({**space_integ, **vel_integ_e})
        p_e_mag = p_e_basis.frobenius_norm()
        p_e_basis.scalar_multiply(1. / p_e_mag, inplace=True)
        p_i_basis = self.sys_fi.f.grid.make_mps_ndim({**space_integ, **vel_integ_i})
        p_i_mag = p_i_basis.frobenius_norm()
        p_i_basis.scalar_multiply(1. / p_i_mag, inplace=True)

        ## conservation of energy: compute field energy (eps0 |E|^2 + 1/mu |B|^2), set targeted value
        vel_integ_e = {v_ax: v_ax.get_xmultiply_mps(x_power=2) for v_ax in self.sys_fe.v_axes}
        vel_integ_i = {v_ax: v_ax.get_xmultiply_mps(x_power=2) for v_ax in self.sys_fi.v_axes}
        nrg_e_basis = self.sys_fe.f.grid.make_mps_ndim({**space_integ, **vel_integ_e})
        nrg_e_mag = nrg_e_basis.frobenius_norm()
        nrg_e_basis.scalar_multiply(1. / nrg_e_mag, inplace=True)
        nrg_i_basis = self.sys_fi.f.grid.make_mps_ndim({**space_integ, **vel_integ_i})
        nrg_i_mag = nrg_i_basis.frobenius_norm()
        nrg_i_basis.scalar_multiply(1. / nrg_i_mag, inplace=True)

        bases['fe'] = [l1_norm, p_e_basis, nrg_e_basis]
        bases['fi'] = [l1_norm, p_i_basis, nrg_i_basis]

        proj_vals = None

        return bases


    def get_time_derivative_operator(self, compress=1, is_ion=False):
        """ dF/dt = G[f(t)].  Returns G
        """
        raise NotImplementedError


    def two_step(self, dt: Numeric, deriv0: Optional['Vlasov'] = None, a2=1, inplace=False,
                method=None, compress_level: int = 1, verbose_plot=False, **deriv_kwargs) -> 'PDE_system':

        new_state = self if inplace else self.copy()

        new_state.sys_fe.force_term = new_state.compute_force_term(is_ion=False, compress_level1=compress_level,
                                                                   compress_level2=0)
        new_state.sys_fi.force_term = new_state.compute_force_term(is_ion=True, compress_level1=compress_level,
                                                                   compress_level2=0)

        deriv0_fe = deriv0.sys_fe if deriv0 is not None else deriv0
        new_state.sys_fe.two_step(dt, deriv0_fe, inplace=True, method=method,
                                  compress_level=compress_level, **deriv_kwargs)

        deriv0_fi = deriv0.sys_fi if deriv0 is not None else deriv0
        new_state.sys_fi.two_step(dt, deriv0_fi, inplace=True, method=method,
                                  compress_level=compress_level, **deriv_kwargs)

        new_state.update_EM_sys(dt, inplace=True, compress=compress_level)
        return new_state


    # @profile
    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None,
                   method_v=None, method_f=None, is_first_time_step=False, is_last_time_step=False,
                   inplace=False, compress_level=1, verbose_plot: bool = False) -> 'Vlasov':
        """ take split step for EM system and ion,electron advection terms
            each step is just an Euler update

            Steps: (init force advection); vel advection; EM update; force advection
        """
        ## at initialization, need to evolve EM_sys with dt/2
        # self.upwind = True
        state0 = self if inplace else self.copy()

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)


        if is_first_time_step:
            ## evolve force advection dt/2
            state0.get_force_advection(dt / 2, inplace=True, method=method_f, compress=comp1, compress1=comp3,
                                       compress2=comp5, verbose_plot=verbose_plot)

        ## evolve velocity advection dt
        state0.get_vel_advection(dt, inplace=True, method=method_v, verbose_plot=verbose_plot,
                                 compress=comp1, compress1=comp3, compress2=comp5,)

        ## include collisions (cross ion/elc collisions)
        if self.verbose:
            print('collisions', self.collision.coll_type)
        if self.collision.coll_type is not None:
            # raise NotImplementedError('collisions not implemented')
            # deriv_coll_e, deriv_coll_i = state0.get_collision_term(v_grads=v_grads, v_axes=v_axes)
            deriv_coll_e = state0.sys_fe.get_collision_term()
            if self.evolve_ion and self.collision.coll_rate_i != 0.0:
                deriv_coll_i = state0.sys_fi.get_collision_term()
            else:
                deriv_coll_i = None
            dFdt_coll = self.create_like(deriv_coll_e, deriv_coll_i, recalc=False)
            state0 = state0.euler(dt, deriv0=dFdt_coll, compress_level=comp2)

        ## update V or EM sys
        if self.verbose:
            print('update EM sys')
        state0.update_EM_sys(dt, inplace=True, compress=comp1)

        # plt.figure()
        # for C in state0.E.componentIDs:
        #     if state0.E[C].data is not None:
        #         plt.plot(state0.E[C].get_data(), label=f'E {C}')
        # plt.legend()
        #
        # plt.figure()
        # for C in state0.ET.componentIDs:
        #     if state0.ET[C].data is not None:
        #         plt.plot(state0.ET[C].get_data(), label=f'ET {C}')
        # plt.legend()
        #
        # plt.figure()
        # for C in state0.EL.componentIDs:
        #     if state0.EL[C].data is not None:
        #         plt.plot(state0.EL[C].get_data(), label=f'EL {C}')
        # plt.legend()
        #
        # plt.figure()
        # for C in state0.B.componentIDs:
        #     if state0.B[C].data is not None:
        #         plt.plot(state0.B[C].get_data(), label=f'B {C}')
        # plt.legend()
        #
        # plt.show()


        if verbose_plot:
            from pde_vlasovES import VlasovPoisson
            if isinstance(self, VlasovPoisson):
                V_data = state0.V.get_comp_data()
                plt.figure()
                plt.plot(V_data)
                plt.title('V')
                plt.show()

        dt_ = dt/2 if is_last_time_step else dt
        state0.get_force_advection(dt_, inplace=True, method=method_f, compress=comp1, compress1=comp3, compress2=comp5,
                                   verbose_plot=verbose_plot)


        # state0.time = self.time + dt if self.time is not None else None
        return state0


    def update_EM_sys(self, dt, inplace=False, compress=1, compress1=2, compress2=4, **kwargs):
        """ update E, B (called by split step)
        """
        return self if inplace else self.copy()


    # def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None,
    #                method_v=None, method_f=None, is_first_time_step=False, is_last_time_step=False,
    #                inplace=False, compress_level=1, compress_level1=4, verbose_plot: bool = False) -> 'Vlasov':
    #     """ take split step for EM system and ion,electron advection terms
    #         each step is just an Euler update
    #         TODO: semiLagrangian advection MPOs (see Sec 3.3, 3.4 https://arxiv.org/pdf/2201.03471.pdf)
    #
    #         change to a more general:
    #         (init force advection); vel advection; EM update; force advection
    #     """
    #     ## at initialization, need to evolve EM_sys with dt/2
    #     # self.upwind = True
    #     state0 = self if inplace else self.copy()
    #
    #     state0.sys_fe.force_term = state0.compute_force_term(is_ion=False, compress_level=compress_level,
    #                                                        compress_level1=compress_level1, compress_level2=0)
    #     state0e = state0.sys_fe.split_step(dt, method_v=method_v, method_f=method_f, inplace=True,
    #                                        is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
    #                                        compress_level=compress_level, verbose_plot=verbose_plot)
    #
    #     state0.sys_fi.force_term = state0.compute_force_term(is_ion=True, compress_level=compress_level,
    #                                                          compress_level1=compress_level1, compress_level2=0)
    #     state0i = state0.sys_fi.split_step(dt, method_v=method_v, method_f=method_f, inplace=True,
    #                                        is_first_time_step=is_first_time_step, is_last_time_step=is_last_time_step,
    #                                        compress_level=compress_level, verbose_plot=verbose_plot)
    #
    #     # state0.sys_fe = state0e
    #     # state0.sys_fi = state0i
    #
    #     ## include collisions (cross ion/elc collisions)
    #     if self.collision.coll_type is not None:
    #         raise NotImplementedError
    #         deriv_coll_e, deriv_coll_i = state0.get_collision_term(v_grads=v_grads, v_axes=v_axes)
    #         dFdt_coll = self.create_like(deriv_coll_e, deriv_coll_i, recalc=False)
    #         state0 = state0.euler(dt, deriv0=dFdt_coll, compress_level=comp2)
    #
    #     state0.time = self.time + dt if self.time is not None else None
    #
    #     return state0


    # @profile
    def get_force_advection(self, dt, inplace=False, method=None, background_force=True, internal_force=True,
                            split_order=2, compress=1, compress1=2, compress2=5, verbose_plot=False):
        """ f_(n+1) = f_(n) + (F * grad_v f_(n))*dt
        """
        state0 = self if inplace else self.copy()

        split_order = 3 if self.semiimplicit_force else split_order
        if self.verbose:
            print('split order vlasov', split_order)

        #### elecs ###
        if self.verbose:
            print('compute force')
        if str(self.te_order)[0] == '4':    # background field is treated separately
            if self.semiimplicit_force:
                dt_ = dt if self.te_order == 415 else dt / 2
                force_bg = state0.compute_force_term_boris(dt_, is_ion=False, compress_level1=compress1,
                                                           compress_level2=0, internal_force=False)
            else:
                force_bg = state0.compute_force_term_bg(is_ion=False, compress_level1=compress1, compress_level2=0)

            if self.verbose:
                print('set elc force')
            state0.set_force_term(force_bg, is_ion=False, background_force=True, internal_force=False)

            if self.verbose:
                print('compute nobg force')
            force_in = state0.compute_force_term_nobg(is_ion=False, compress_level1=compress1, compress_level2=0)
            state0.set_force_term(force_in, is_ion=False, background_force=False, internal_force=True)

            # print('elc force')
            # state0.sys_fe.force_term_bg_SL = self.compute_force_term_bg_SL(dt, is_ion=False, compress_level=compress,
            #                                                                compress_level1=compress1)
            # print(state0.sys_fe.force_term_bg_SL[next(iter(state0.sys_fe.force_term_bg_SL))].vals)

        else:
            # X, Y, Z = self.coords_x.coords
            # Ex0, omega = 0.9, 0.4567
            # print('E', self.time, state0.EM_sys.E[X].data, Ex0 * np.cos(omega * self.time))

            force_term = state0.compute_force_term(is_ion=False,
                                                   background_force=background_force, internal_force=internal_force,
                                                   compress_level1=compress1, compress_level2=0)
            if self.verbose:
                print('set elc force')
            state0.set_force_term(force_term, is_ion=False,
                                  background_force=background_force, internal_force=internal_force)

        if self.verbose:
            print('elc force adv')
        state0.sys_fe = state0.sys_fe.get_force_advection(dt, inplace=True, method=method,
                                                          background_force=background_force,
                                                          internal_force=internal_force, compress=compress,
                                                          compress1=compress1, compress2=compress2,
                                                          verbose_plot=verbose_plot, split_order=split_order)

        #### ions ####
        if self.evolve_ion is True:
            if str(self.te_order)[0] == '4':
                if self.semiimplicit_force:
                    dt_ = dt if self.te_order == 415 else dt / 2
                    force_bg = state0.compute_force_term_boris(dt_, is_ion=True, compress_level1=compress1,
                                                               compress_level2=0, internal_force=False)
                else:
                    force_bg = state0.compute_force_term_bg(is_ion=True, compress_level1=compress1, compress_level2=0)

                if self.verbose:
                    print('set ion force')
                state0.set_force_term(force_bg, is_ion=True, background_force=True, internal_force=False)

                force_in = state0.compute_force_term_nobg(is_ion=True, compress_level1=compress1, compress_level2=0)
                state0.set_force_term(force_in, is_ion=True, background_force=False, internal_force=True)

                # print('ion force')
                # state0.sys_fi.force_term_bg_SL = self.compute_force_term_bg_SL(dt, is_ion=True,
                #                                                                compress_level=compress,
                #                                                                compress_level1=compress1)
            else:
                force_term = state0.compute_force_term(is_ion=True,
                                                       background_force=background_force, internal_force=internal_force,
                                                       compress_level1=compress1, compress_level2=0)
                # print('ion force advec', background_force, internal_force)
                if self.verbose:
                    print('set ion force')
                state0.set_force_term(force_term, is_ion=True,
                                      background_force=background_force, internal_force=internal_force,)

            # print('ion')
            state0.sys_fi = state0.sys_fi.get_force_advection(dt, inplace=True, method=method,
                                                              background_force=background_force,
                                                              internal_force=internal_force, compress=compress,
                                                              compress1=compress1, compress2=compress2,
                                                              verbose_plot=verbose_plot, split_order=split_order)
        return state0


    def get_vel_advection(self, dt, inplace=False, method=None, split_order=2, compress=1, compress1=2, compress2=5,
                          verbose_plot=False):
        """ f_(n+1) = f_(n) + (v * grad_x f_(n))*dt
        """
        state0 = self if inplace else self.copy()

        state0.sys_fe = state0.sys_fe.get_vel_advection(dt, inplace=inplace, method=method, split_order=split_order,
                                                        compress=compress,compress1=compress1, compress2=compress2,
                                                        verbose_plot=verbose_plot)
        if self.evolve_ion is True:
            state0.sys_fi = state0.sys_fi.get_vel_advection(dt, inplace=inplace, method=method, split_order=split_order,
                                                            compress=compress, compress1=compress1, compress2=compress2,
                                                            verbose_plot=verbose_plot)
        return state0

    #############################

    # @profile
    def compute_force_term(self, is_ion=False, compress_level=1, compress_level1=0, compress_level2=0,
                           verbose_plot=False, background_force=True, internal_force=True, **kwargs) -> 'Field':
        """ compute EM force:  q/m (E + v x B)
        """
        raise NotImplementedError

    def set_force_term(self, force, is_ion=False, background_force=True, internal_force=True, time=None, reset=False):
        sys_f = self.sys_fi if is_ion else self.sys_fe
        sys_f.set_force_term(force, background_force=background_force, internal_force=internal_force, time=time, reset=reset)

    def get_force_term(self, is_ion=False, background_force=True, internal_force=True, time=None):
        sys_f = self.sys_fi if is_ion else self.sys_fe
        return sys_f.get_force_term(background_force=background_force, internal_force=internal_force, time=time)

    def compute_force_term_nobg(self, is_ion=False, compress_level=1, compress_level1=0, compress_level2=0,
                                verbose_plot=False,):
        return self.compute_force_term(is_ion=is_ion, compress_level=compress_level, compress_level1=compress_level1,
                                       compress_level2=compress_level2, verbose_plot=verbose_plot,
                                       background_force=False, internal_force=True)

    def compute_force_term_bg(self, is_ion=False, compress_level=1, compress_level1=0, compress_level2=0,
                              verbose_plot=False, ):
        return self.compute_force_term(is_ion=is_ion, compress_level=compress_level, compress_level1=compress_level1,
                                       compress_level2=compress_level2, verbose_plot=verbose_plot,
                                       background_force=True, internal_force=False)

    def compute_force_term_bg_SL(self, dt: Numeric, is_ion=False, sl_order=DEFAULT_SL_ORDER, compress_level=1,
                                 compress_level1=0, compress_level2=0, verbose_plot=False, ):
        raise NotImplementedError

    # @profile
    def compute_charge_density(self, ignore_e=False, compress=1, compress1=5):
        """ compute n(x) = \integ dv \sum_s q_s f_s
        """
        # print('vlasov computing charge density')
        charge_density = None

        if self.fe is not None and self.fe.component is not None and self.fe.component.data is not None:
            if ignore_e:
                charge_gtn = self.grid_X.get_ones_mps()
                dx = np.prod([ax.dx for ax in self.grid_X.axes])
                norm = self.sys_fe.f.norm() / dx / self.grid_X.npts
                # print('norm', norm * self.elc_params.Z * self.elc_params.e)
                charge_gtn.scalar_multiply( norm * self.elc_params.Z * self.elc_params.e,
                                            inplace=True )
                charge_density = ScalarField( 'charge_e', self.grid_X, charge_gtn)
            else:
                e_density = self.sys_fe.compute_density(compress=compress1)
                e_density.scalar_multiply(self.elc_params.Z * self.elc_params.e, inplace=True)
                charge_density = e_density.copy()
                # plt.figure()
                # plt.plot(charge_density.get_comp_data())
                # plt.title('e charge denisty')
                # plt.show()

        if self.fi is not None and self.fi.component is not None and self.fi.component.data is not None:
            i_density = self.sys_fi.compute_density(compress=compress1)
            # i_density.scalar_multiply(self.ion_params.Z * self.ion_params.e / self.ion_params.eps0, inplace=True)
            i_density.scalar_multiply(self.ion_params.Z * self.ion_params.e, inplace=True)

            if charge_density is not None:
                charge_density.add(i_density, inplace=True, compress_level=0)
            else:
                charge_density = i_density.copy()

        # print('charge dnsity norm', charge_density.component.frobenius_norm())
        charge_norm = charge_density.component.frobenius_norm() if isinstance(charge_density.component, GridTN) \
                            else charge_density.component
        if np.abs(charge_norm) < 1.0e-10 or charge_norm is np.nan:
            charge_density.data = None

        if isinstance(charge_density, ScalarField) and compress:
            charge_density.compress(inplace=True, compress_level=compress)

        # plt.figure()
        # # print(type(charge_density))
        # charge_data = charge_density.get_comp_data()
        # e_data = e_density.get_comp_data()
        # i_data = i_density.get_comp_data()
        # plt.plot(charge_data, label='tot charge')
        # if e_data is not None and i_data is not None:
        #     plt.plot(e_data + i_data, label='e + i')
        # plt.legend()
        # plt.ylabel('charge')
        # plt.show()

        # print('n(x) deriv config', charge_density.component.ax_deriv_configs)

        return charge_density


    # @profile
    def compute_current(self, ignore_e=False, compress=1, compress1=5) -> 'Field':
        """ compute j(x) = \integ dv \sum_s q_s v_s f_s
            stagger: number of 1/2 grid points to shift data by
                eg. i -> i + 1/2 for staggered Yee cell
        """
        # print('vlasov computing current density')
        j = None

        # ignore_e = True
        if self.fe is not None and not ignore_e:
            je = self.sys_fe.compute_flow(compress=compress1)
            je.scalar_multiply(self.elc_params.Z * self.elc_params.e, inplace=True)
            # if ignore_e:
                #     je_avg_val = je.integrate()
            #     for compID in je.componentIDs:
            #         je_avg_val_C = je_avg_val[compID] / je.grid.npts
            #         if np.abs(je_avg_val_C) > 1.0e-10:
            #             je_avg_C = je.grid.get_ones_mps()
            #             je_avg_C.scalar_multiply(je_avg_val_C, inplace=True)
            #             je[compID].data = je_avg_C
            #         else:
            #             je[compID].data = None
            j = je.copy()
            # j.components.pop(self.coords_x.coords[0])   ## pop jx

        if self.fi is not None:
            ji = self.sys_fi.compute_flow(compress=compress1)
            ji.scalar_multiply(self.ion_params.Z * self.ion_params.e, inplace=True)
            if j is not None:
                j.add(ji, compress_level=0, inplace=True)
            else:
                j = ji.copy()

        if isinstance(j, Field) and compress:
            j.compress(inplace=True, compress_level=compress)

        # j.components.pop(self.coords_x.coords[0])

        for C in list(j.componentIDs):
            jC_norm = j[C].frobenius_norm() if isinstance(j[C], GridTN) else j[C]
            if j[C] is not None and np.abs( jC_norm ) < 1.0e-10:
                # j.components.pop(C)
                j[C].data = None

        # plt.figure()
        # for C, jC in j.components.items():
        #     # if C != j.componentIDs[0]:  continue
        #     if jC is not None and jC.data is not None:
        #         plt.semilogy(np.real(jC.get_data()), label=f'j{C}')
        #         # plt.semilogy(np.imag(jC.get_data()), '--',label=f'j{C}')
        #         # print('jC val', C, jC.get_data()[0])
        # plt.legend()
        # plt.show()

        # for C, jC in j.components.items():
        #     # if C != j.componentIDs[0]:  continue
        #     if jC is not None and jC.data is not None:
        #         jC_data = jC.get_data()
        #         ax_x, ax_y = jC.grid.axes
        #         print('jC data', jC_data.shape)
        #         jC_data = ax_x.basis.get_realspace_1D(jC_data, 0)
        #         jC_data = ax_y.basis.get_realspace_1D(jC_data, 1)
        #         plt.figure()
        #         plt.imshow(np.real(jC_data))
        #         plt.title(f'j{C}')
        #         plt.colorbar()
        # plt.show()

        return j

    ###################################

    # @profile
    def euler(self, dt: Numeric, deriv0: Optional['Vlasov'] = None, inplace=False, do_update_V=True,
              compress_level: int = 1, compress_level1: int = 0, compress_level2: int = 0, verbose_plot: bool = False,
              do_x_advection=True, do_v_advection=True) -> 'Vlasov':

        state1 = self if inplace else self.copy()

        # print('vlasov deriv0 is None?', deriv0 is None)
        if deriv0 is None:
            deriv0 = self.calculate_time_derivative(time=self.time, compress_level=compress_level,
                                                    compress_level1=compress_level1, compress_level2=compress_level2,
                                                    background_force=do_v_advection, internal_force=do_x_advection,
                                                    verbose_plot=verbose_plot)

        # print('deriv0', deriv0.sys_fe.f.norm(), deriv0.sys_fi.f.norm())

        # state1.sys_fe.force_term = self.compute_force_term(is_ion=False, compress_level=compress_level,
        #                                                    compress_level1=compress_level1, compress_level2=0)
        if self.verbose:
            print('euler fe', state1.sys_fe.f.max_bond(), compress_level)
        state1.sys_fe.euler(dt, deriv0=deriv0.sys_fe, inplace=True, compress_level=compress_level,
                            compress_level1=compress_level1, compress_level2=compress_level2,
                            verbose_plot=verbose_plot)

        # state1.sys_fi.force_term = self.compute_force_term(is_ion=True, compress_level=compress_level,
        #                                                    compress_level1=compress_level1, compress_level2=0)
        if self.verbose:
            print('euler fi', state1.sys_fi.f.max_bond())
        state1.sys_fi.euler(dt, deriv0=deriv0.sys_fi, inplace=True, compress_level=compress_level,
                            compress_level1=compress_level1, compress_level2=compress_level2,
                            verbose_plot=verbose_plot)

        return state1


    def lax_wendroff_ndim(self, dt: Numeric, advec_axes=None, ax_deriv_configs=None, do_update_V=True, inplace=False,
                          compress_level=1, verbose_plot=False, **kwargs):
        state1 = self if inplace else self.copy()

        if state1.sys_fe is not None:
            ## update force
            force_term = state1.compute_force_term(is_ion=False, compress_level=compress_level,
                                                 compress_level1=compress_level+2, compress_level2=0,
                                                 verbose_plot=verbose_plot, background_force=True,
                                                 internal_force=True)
            state1.sys_fe.set_force_term(force_term, background_force=True, internal_force=True)

            ## update fe
            state1.sys_fe.lax_wendroff_ndim(dt, inplace=True, advec_axes=advec_axes, compress_level=compress_level,
                                            background_force=True, internal_force=True,
                                            **kwargs)

        if state1.sys_fi is not None and self.evolve_ion:
            ## update force
            force_term = state1.compute_force_term(is_ion=True, compress_level=compress_level,
                                                 compress_level1=compress_level+2, compress_level2=0,
                                                 verbose_plot=verbose_plot, background_force=True,
                                                 internal_force=True)
            state1.sys_fi.set_force_term(force_term, background_force=True, internal_force=True)

            ## upate fi
            state1.sys_fi.lax_wendroff_ndim(dt, inplace=True, advec_axes=advec_axes, compress_level=compress_level,
                                            background_force=True, internal_force=True,
                                            **kwargs)

        if np.abs(state1.elc_params.e) > 0 and do_update_V:
            state1.update_EM_sys(dt, inplace=True, compress=compress_level, compress1=compress_level+2)

        return state1


    def dynamical_low_rank(self, dt: Numeric, inplace=False, te_order = 4, do_adapt: bool = True,
                           compress_level: int = 1, compress_level_2: int = 0,
                           advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                           do_update_V=True, update_force=True, verbose_plot: bool = False, **kwargs) -> 'Vlasov':

        state1 = self if inplace else self.copy()

        if state1.sys_fe is not None:
            if update_force:
                force_term = state1.compute_force_term(is_ion=False, compress_level=compress_level,
                                                     compress_level1=compress_level_2, compress_level2=0,
                                                     verbose_plot=verbose_plot, background_force=True,
                                                     internal_force=True)
                state1.sys_fe.set_force_term(force_term, background_force=True, internal_force=True)

            state1.sys_fe.dynamical_low_rank(dt, te_order=te_order, inplace=True, compress_level=compress_level,
                                             advec_axes=advec_axes, background_force=background_force,
                                             internal_force=internal_force, **kwargs)

        if state1.sys_fi is not None:
            if update_force:
                force_term = state1.compute_force_term(is_ion=True, compress_level=compress_level,
                                                     compress_level1=compress_level_2, compress_level2=0,
                                                     verbose_plot=verbose_plot, background_force=True,
                                                     internal_force=True)
                state1.sys_fi.set_force_term(force_term, background_force=True, internal_force=True)

            state1.sys_fi.dynamical_low_rank(dt, te_order=te_order, inplace=True, compress_level=compress_level,
                                             advec_axes=advec_axes, background_force=background_force,
                                             internal_force=internal_force, **kwargs)

        if np.abs(state1.elc_params.e) > 0 and do_update_V:
            if self.verbose:
                print('updating V')
            state1.update_EM_sys(dt, inplace=True, compress=compress_level, compress1=compress_level_2)

        return state1


    def time_dependent_variational_principle(self, dt: Numeric, te_order = 4, do_adapt: bool = True, inplace=False,
                           compress_level: int = 1, compress_level_2: int = 4, direction=1,
                           advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                           do_update_V=True, update_force=True, verbose_plot: bool = False, **kwargs) -> 'Vlasov':

        state1 = self if inplace else self.copy()
        # update_force = True

        if update_force:
            # print('update force', background_force, internal_force)
            if state1.sys_fe is not None:
                force_term = state1.compute_force_term(is_ion=False, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=background_force,
                                                       internal_force=internal_force)
                state1.sys_fe.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            if self.evolve_ion and state1.sys_fi is not None:
                force_term = state1.compute_force_term(is_ion=True, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=background_force,
                                                       internal_force=internal_force)
                state1.sys_fi.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

        if state1.sys_fe is not None:
            state1.sys_fe.time_dependent_variational_principle(dt, te_order=te_order, do_adapt=do_adapt, inplace=True,
                                             direction=direction, compress_level=compress_level,
                                             advec_axes=advec_axes, background_force=background_force,
                                             internal_force=internal_force, **kwargs)

        if self.evolve_ion and state1.sys_fi is not None and state1.fi.component is not None and state1.fi.component.data is not None:
            state1.sys_fi.time_dependent_variational_principle(dt, te_order=te_order, do_adapt=do_adapt, inplace=True,
                                             direction=direction, compress_level=compress_level,
                                             advec_axes=advec_axes, background_force=background_force,
                                             internal_force=internal_force, **kwargs)

        if np.abs(state1.elc_params.e) > 0 and do_update_V:
            if self.verbose:
                print('updating EM sys')
            state1.update_EM_sys(dt, inplace=True, compress=compress_level, compress1=compress_level_2)

        return state1


    def time_dmrg(self, dt: Numeric, te_order = 4, do_adapt: bool = True, inplace=False,
                  compress_level: int = 1, compress_level_2: int = 4, direction=1,
                  advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                  do_update_V=True, update_force=True, verbose_plot: bool = False,
                  solver_type=LocalSolverType.TDDMRG, **kwargs) -> 'Vlasov':

        state1 = self if inplace else self.copy()
        # update_force = True

        if update_force:
            # print('update force', background_force, internal_force)
            if state1.sys_fe is not None:
                force_term = state1.compute_force_term(is_ion=False, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=background_force,
                                                       internal_force=internal_force)
                state1.sys_fe.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            if self.evolve_ion and state1.sys_fi is not None:
                force_term = state1.compute_force_term(is_ion=True, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=background_force,
                                                       internal_force=internal_force)
                state1.sys_fi.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

        if state1.sys_fe is not None:
            if self.verbose:
                print('vlasov time dmrg', solver_type)
            state1.sys_fe.time_dmrg(dt, te_order=te_order, do_adapt=do_adapt, inplace=True,
                                    direction=direction, compress_level=compress_level,
                                    advec_axes=advec_axes, background_force=background_force,
                                    internal_force=internal_force, solver_type=solver_type, **kwargs)

        if self.evolve_ion and state1.sys_fi is not None and state1.fi.component is not None and state1.fi.component.data is not None:
            state1.sys_fi.time_dmrg(dt, te_order=te_order, do_adapt=do_adapt, inplace=True,
                                    direction=direction, compress_level=compress_level,
                                    advec_axes=advec_axes, background_force=background_force,
                                    internal_force=internal_force, solver_type=solver_type, **kwargs)

        if np.abs(state1.elc_params.e) > 0 and do_update_V:
            if self.verbose:
                print('updating EM sys')
            state1.update_EM_sys(dt, inplace=True, compress=compress_level, compress1=compress_level_2)

        return state1


    def tdvp_new(self, dt: Numeric, te_order = 4, do_adapt: bool = True, inplace=False,
                 compress_level: int = 1, compress_level_2: int = 4, direction=1, solver_type=LocalSolverType.TDDMRG,
                 advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                 do_update_V=True, update_force=True, verbose_plot: bool = False, **kwargs) -> 'Vlasov':

        state1 = self if inplace else self.copy()
        time = state1.time
        # update_force = True

        # if np.abs(state1.elc_params.e) > 0 and do_update_V:
        #     print('updating EM sys')
        #     state1.EM_sys.time = time
        #     state1.update_EM_sys(dt/2, inplace=True, compress=compress_level, compress1=compress_level_2)

        if update_force:
            # print('update force', background_force, internal_force)
            if state1.sys_fe is not None:
                state1.sys_fe.time = time
                force_term = state1.compute_force_term(is_ion=False, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=background_force,
                                                       internal_force=internal_force)
                state1.sys_fe.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            if self.evolve_ion and state1.sys_fi is not None:
                state1.sys_fi.time = time
                force_term = state1.compute_force_term(is_ion=True, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=background_force,
                                                       internal_force=internal_force)
                state1.sys_fi.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

        if state1.sys_fe is not None:
            state1.sys_fe.time = time
            state1.sys_fe.tdvp_new(dt, te_order=te_order, do_adapt=do_adapt, inplace=True,
                                        direction=direction, compress_level=compress_level,
                                        advec_axes=advec_axes, background_force=background_force,
                                        internal_force=internal_force, solver_type=solver_type, **kwargs)

        if self.evolve_ion and state1.sys_fi is not None and state1.fi.component is not None and state1.fi.component.data is not None:
            state1.sys_fi.time = time
            state1.sys_fi.tdvp_new(dt, te_order=te_order, do_adapt=do_adapt, inplace=True,
                                        direction=direction, compress_level=compress_level,
                                        advec_axes=advec_axes, background_force=background_force,
                                        internal_force=internal_force, solver_type=solver_type, **kwargs)

        if np.abs(state1.elc_params.e) > 0 and do_update_V:
            if self.verbose:
                print('updating EM sys')
            state1.EM_sys.time = time
            state1.update_EM_sys(dt, inplace=True, compress=compress_level, compress1=compress_level_2)

        return state1


    def time_dmrg_new(self, dt: Numeric, te_order = 4, do_adapt: bool = True, inplace=False,
                      compress_level: int = 1, compress_level_2: int = 4, direction=1,
                      advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                      do_update_V=True, update_force=True, verbose_plot: bool = False,
                      solver_type=LocalSolverType.TDDMRG, **kwargs) -> 'Vlasov':

        state1 = self if inplace else self.copy()
        time = state1.time
        # update_force = True

        if update_force:
            # print('update force', background_force, internal_force)
            if state1.sys_fe is not None:
                state1.sys_fe.time = time
                force_term = state1.compute_force_term(is_ion=False, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=background_force,
                                                       internal_force=internal_force)
                state1.sys_fe.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            if self.evolve_ion and state1.sys_fi is not None:
                state1.sys_fi.time = time
                force_term = state1.compute_force_term(is_ion=True, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=background_force,
                                                       internal_force=internal_force)
                state1.sys_fi.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

        if state1.sys_fe is not None:
            state1.sys_fe.time = time
            if self.verbose:
                print('pde vlasov td-dmrg sys fe time', state1.sys_fe.time)
            state1.sys_fe.time_dmrg_new(dt, te_order=te_order, do_adapt=do_adapt, inplace=True,
                                        direction=direction, compress_level=compress_level,
                                        advec_axes=advec_axes, background_force=background_force,
                                        internal_force=internal_force, solver_type=solver_type, **kwargs)

        if self.evolve_ion and state1.sys_fi is not None and state1.fi.component is not None and state1.fi.component.data is not None:
            state1.sys_fi.time = time
            if self.verbose:
                print('pde vlasov td-dmrg sys fi time', state1.sys_fi.time)
            state1.sys_fi.time_dmrg_new(dt, te_order=te_order, do_adapt=do_adapt, inplace=True,
                                        direction=direction, compress_level=compress_level,
                                        advec_axes=advec_axes, background_force=background_force,
                                        internal_force=internal_force, solver_type=solver_type, **kwargs)

        if np.abs(state1.elc_params.e) > 0 and do_update_V:
            if self.verbose:
                print('updating EM sys')
            state1.EM_sys.time = time
            state1.update_EM_sys(dt, inplace=True, compress=compress_level, compress1=compress_level_2)

        return state1


    def time_local_global(self, dt: Numeric, te_order = 4, do_adapt: bool = True, inplace=False,
                          compress_level: int = 1, compress_level_2: int = 4, direction=1,
                          advec_axes: Sequence['Axis'] = None, background_force=True, internal_force=True,
                          do_update_V=True, update_force=True, verbose_plot: bool = False, **kwargs) -> 'Vlasov':

        state1 = self if inplace else self.copy()
        time = state1.time
        # update_force = True

        if update_force:
            # print('update force', background_force, internal_force)
            if state1.sys_fe is not None:
                state1.sys_fe.time = time
                force_term = state1.compute_force_term(is_ion=False, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=background_force,
                                                       internal_force=internal_force)
                state1.sys_fe.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            if state1.sys_fi is not None:
                state1.sys_fi.time = time
                force_term = state1.compute_force_term(is_ion=True, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=background_force,
                                                       internal_force=internal_force)
                state1.sys_fi.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

        if state1.sys_fe is not None:
            state1.sys_fe.time = time
            state1.sys_fe.time_local_global(dt, te_order=te_order, do_adapt=do_adapt, inplace=True,
                                        direction=direction, compress_level=compress_level,
                                        advec_axes=advec_axes, background_force=background_force,
                                        internal_force=internal_force, **kwargs)

        if state1.sys_fi is not None and state1.fi.component is not None and state1.fi.component.data is not None:
            state1.sys_fi.time = time
            state1.sys_fi.time_local_global(dt, te_order=te_order, do_adapt=do_adapt, inplace=True,
                                        direction=direction, compress_level=compress_level,
                                        advec_axes=advec_axes, background_force=background_force,
                                        internal_force=internal_force, **kwargs)

        if np.abs(state1.elc_params.e) > 0 and do_update_V:
            print('updating EM sys')
            state1.EM_sys.time = time
            state1.update_EM_sys(dt, inplace=True, compress=compress_level, compress1=compress_level_2)

        return state1



    def time_dependent_variational_principle_SL(self, dt: Numeric, te_order = 4, do_adapt: bool = True, inplace=False,
                                                advec_axes: Sequence['Axis'] = None, bg_method='SL', bg_split_order=1,
                                                compress_level: int = 1, compress_level_2: int = 4, direction=1,
                                                do_update_V=True, update_force=True,
                                                verbose_plot: bool = False, **kwargs) -> 'Vlasov':

        state1 = self if inplace else self.copy()
        update_force = True

        ### background force
        if True:  # state1.sys_fe.force_term_bg is None:
            force_term_bg = state1.compute_force_term(is_ion=False, compress_level=compress_level,
                                                      compress_level1=compress_level_2, compress_level2=0,
                                                      verbose_plot=verbose_plot, background_force=True,
                                                      internal_force=False)
            state1.sys_fe.set_force_term(force_term_bg, background_force=True, internal_force=False)

        if True:  # state1.sys_fi.force_term_bg is None:
            force_term_bg = state1.compute_force_term(is_ion=True, compress_level=compress_level,
                                                      compress_level1=compress_level_2, compress_level2=0,
                                                      verbose_plot=verbose_plot, background_force=True,
                                                      internal_force=False)
            state1.sys_fi.set_force_term(force_term_bg, background_force=True, internal_force=False)

        ## SL background update
        if state1.sys_fe is not None:
            state1.sys_fe.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                              internal_force=False, split_order=bg_split_order,
                                              ompress=compress_level, compress1=compress_level_2)
        if state1.sys_fi is not None:
            state1.sys_fi.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                              internal_force=False, split_order=bg_split_order,
                                              ompress=compress_level, compress1=compress_level_2)

        ### update force with new distribution function
        if np.abs(state1.elc_params.e) > 0 and do_update_V:
            state1.update_EM_sys(dt, inplace=True, compress=compress_level, compress1=compress_level_2)

        if update_force:
            if state1.sys_fe is not None:
                force_term = state1.compute_force_term(is_ion=False, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=False,
                                                       internal_force=True)
                state1.sys_fe.set_force_term(force_term, background_force=False, internal_force=True)

            if state1.sys_fi is not None:
                force_term = state1.compute_force_term(is_ion=True, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=False,
                                                       internal_force=True)
                state1.sys_fi.set_force_term(force_term, background_force=False, internal_force=True)

        # ### TDVP
        if state1.sys_fe is not None:
            state1.sys_fe.time_dependent_variational_principle(dt, te_order=te_order, do_adapt=do_adapt,
                                                               inplace=True, advec_axes=advec_axes,
                                                               background_force=False, internal_force=True,
                                                               compress_level=compress_level,
                                                               compress_level_2=compress_level_2, **kwargs)

        if state1.sys_fi is not None and state1.fi.component is not None and state1.fi.component.data is not None:
            state1.sys_fi.time_dependent_variational_principle(dt, te_order=te_order, do_adapt=do_adapt,
                                                               inplace=True, advec_axes=advec_axes,
                                                               background_force=False, internal_force=True,
                                                               compress_level=compress_level,
                                                               compress_level_2=compress_level_2, **kwargs)

        # ### RK4
        # if state1.sys_fe is not None:
        #     state1.sys_fe.rk4(dt, compress_level=compress_level, verbose_plot=verbose_plot,
        #                       background_force=False, internal_force=True)
        # if state1.sys_fi is not None and state1.fi.component is not None and state1.fi.component.data is not None:
        #     state1.sys_fi.rk4(dt, compress_level=compress_level, verbose_plot=verbose_plot,
        #                       background_force=False, internal_force=True)


        ## SL background update
        bg_split_order = -1 if bg_split_order == 1 else bg_split_order
        if state1.sys_fe is not None:
            state1.sys_fe.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                              internal_force=False, split_order=bg_split_order,
                                              ompress=compress_level, compress1=compress_level_2)
        if state1.sys_fi is not None:
            state1.sys_fi.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                              internal_force=False, split_order=bg_split_order,
                                              ompress=compress_level, compress1=compress_level_2)


        return state1


    def time_dmrg_SL(self, dt: Numeric, te_order = 4, do_adapt: bool = True, inplace=False,
                     advec_axes: Sequence['Axis'] = None, bg_method='SL', bg_split_order=1,
                     compress_level: int = 1, compress_level_2: int = 4, direction=1,
                     do_update_V=True, update_force=True,
                     verbose_plot: bool = False, **kwargs) -> 'Vlasov':

        state1 = self if inplace else self.copy()
        update_force = True

        ### background force
        if True:  # state1.sys_fe.force_term_bg is None:
            force_term_bg = state1.compute_force_term(is_ion=False, compress_level=compress_level,
                                                      compress_level1=compress_level_2, compress_level2=0,
                                                      verbose_plot=verbose_plot, background_force=True,
                                                      internal_force=False)
            state1.sys_fe.set_force_term(force_term_bg, background_force=True, internal_force=False)

        if True:  # state1.sys_fi.force_term_bg is None:
            force_term_bg = state1.compute_force_term(is_ion=True, compress_level=compress_level,
                                                      compress_level1=compress_level_2, compress_level2=0,
                                                      verbose_plot=verbose_plot, background_force=True,
                                                      internal_force=False)
            state1.sys_fi.set_force_term(force_term_bg, background_force=True, internal_force=False)

        ## SL background update
        if state1.sys_fe is not None:
            state1.sys_fe.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                              internal_force=False, split_order=bg_split_order,
                                              ompress=compress_level, compress1=compress_level_2)
        if state1.sys_fi is not None:
            state1.sys_fi.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                              internal_force=False, split_order=bg_split_order,
                                              ompress=compress_level, compress1=compress_level_2)

        ### update force with new distribution function
        if np.abs(state1.elc_params.e) > 0 and do_update_V:
            state1.update_EM_sys(dt, inplace=True, compress=compress_level, compress1=compress_level_2)

        if update_force:
            if state1.sys_fe is not None:
                force_term = state1.compute_force_term(is_ion=False, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=False,
                                                       internal_force=True)
                state1.sys_fe.set_force_term(force_term, background_force=False, internal_force=True)

            if state1.sys_fi is not None:
                force_term = state1.compute_force_term(is_ion=True, compress_level=compress_level,
                                                       compress_level1=compress_level_2, compress_level2=0,
                                                       verbose_plot=verbose_plot, background_force=False,
                                                       internal_force=True)
                state1.sys_fi.set_force_term(force_term, background_force=False, internal_force=True)

        # ### TDVP
        if state1.sys_fe is not None:
            state1.sys_fe.time_dmrg(dt, te_order=te_order, do_adapt=do_adapt,
                                    inplace=True, advec_axes=advec_axes,
                                    background_force=False, internal_force=True,
                                    compress_level=compress_level,
                                    compress_level_2=compress_level_2, **kwargs)

        if state1.sys_fi is not None and state1.fi.component is not None and state1.fi.component.data is not None:
            state1.sys_fi.time_dmrg(dt, te_order=te_order, do_adapt=do_adapt,
                                    inplace=True, advec_axes=advec_axes,
                                    background_force=False, internal_force=True,
                                    compress_level=compress_level,
                                    compress_level_2=compress_level_2, **kwargs)

        # ### RK4
        # if state1.sys_fe is not None:
        #     state1.sys_fe.rk4(dt, compress_level=compress_level, verbose_plot=verbose_plot,
        #                       background_force=False, internal_force=True)
        # if state1.sys_fi is not None and state1.fi.component is not None and state1.fi.component.data is not None:
        #     state1.sys_fi.rk4(dt, compress_level=compress_level, verbose_plot=verbose_plot,
        #                       background_force=False, internal_force=True)


        ## SL background update
        bg_split_order = -1 if bg_split_order == 1 else bg_split_order
        if state1.sys_fe is not None:
            state1.sys_fe.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                              internal_force=False, split_order=bg_split_order,
                                              ompress=compress_level, compress1=compress_level_2)
        if state1.sys_fi is not None:
            state1.sys_fi.get_force_advection(dt / 2, inplace=True, method=bg_method, background_force=True,
                                              internal_force=False, split_order=bg_split_order,
                                              ompress=compress_level, compress1=compress_level_2)


        return state1


    # @profile
    def calculate_time_derivative(self, time=None, compress_level=0, compress_level1=0, compress_level2=0,
                                  do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
                                  update_force=True, verbose_plot=False, **kwargs) -> 'Vlasov':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        if self.sys_fe is not None:
            if update_force:
                force_term = self.compute_force_term(is_ion=False, compress_level=compress_level,
                                                     compress_level1=compress_level1, compress_level2=0,
                                                     verbose_plot=verbose_plot, background_force=background_force,
                                                     internal_force=internal_force)
                # print('force compress', compress_level, compress_level1, compress_level2)
                self.sys_fe.set_force_term(force_term, background_force=background_force, internal_force=internal_force)
                # self.sys_fe.force_term = force_term

            dFdt_e = self.sys_fe.calculate_time_derivative(time=time,
                                                           compress_level=compress_level,
                                                           compress_level1=compress_level1,
                                                           compress_level2=compress_level2,
                                                           do_x_advection=do_x_advection,
                                                           do_v_advection=do_v_advection,
                                                           background_force=background_force,
                                                           internal_force=internal_force,
                                                           verbose_plot=verbose_plot)
        else:
            dFdt_e = None # self.sys_fe.create_like()

        if self.sys_fi is not None and self.evolve_ion:
            if update_force:
                force_term = self.compute_force_term(is_ion=True, compress_level=compress_level,
                                                     compress_level1=compress_level1, compress_level2=0,
                                                     verbose_plot=verbose_plot, background_force=background_force,
                                                     internal_force=internal_force)
                # self.sys_fi.force_term = force_term
                self.sys_fi.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            dFdt_i = self.sys_fi.calculate_time_derivative(time=time,
                                                           compress_level=compress_level,
                                                           compress_level1=compress_level1,
                                                           compress_level2=compress_level2,
                                                           do_x_advection=do_x_advection,
                                                           do_v_advection=do_v_advection,
                                                           background_force=background_force,
                                                           internal_force=internal_force,
                                                           verbose_plot=verbose_plot)
        else:
            dFdt_i = None

        dFdt = self.create_like(dFdt_e, dFdt_i, recalc=False)
        return dFdt


    def get_time_derivative_op(self, time=None,
                               compress_level: int = 0, compress_level1: int = 0, compress_level2: int = 0,
                               do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
                               verbose_plot: bool =False, **kwargs) -> 'PDE_system':
        raise NotImplementedError


    def get_collision_term(self, v_axes: Sequence['Axis'] = None, v_grads: 'Field' = None,
                           compress1=0, compress2=0):
        """ obtain collision term
        """
        coll_e = self.sys_fe.get_collision_term(v_axes=v_axes, v_grads=v_grads,
                                                compress1=compress1, compress2=compress2)
        coll_i = None
        if self.evolve_ion:
            coll_i = self.sys_fi.get_collision_term(v_axes=v_axes, v_grads=v_grads,
                                                    compress1=compress1, compress2=compress2)

        ### TODO: implement ion/elec collisions

        return coll_e, coll_i



    def deriv_upwind_global(self, nsites=2, ket=None, max_bond: int = None, cutoff: Numeric = None,
                            time=None, do_x_advection=True, do_v_advection=True, background_force=True,
                            internal_force=True, update_force=True, verbose_plot=False, **kwargs) -> 'Vlasov':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        if self.sys_fe is not None:
            if update_force:
                force_term = self.compute_force_term(is_ion=False,
                                                     verbose_plot=verbose_plot, background_force=background_force,
                                                     internal_force=internal_force)
                # print('force compress', compress_level, compress_level1, compress_level2)
                self.sys_fe.set_force_term(force_term, background_force=background_force, internal_force=internal_force)
                # self.sys_fe.force_term = force_term

            dFdt_e = self.sys_fe.deriv_upwind_global(time=time,
                                                     do_x_advection=do_x_advection,
                                                     do_v_advection=do_v_advection,
                                                     background_force=background_force,
                                                     internal_force=internal_force,
                                                     verbose_plot=verbose_plot)
            deriv_e = self.sys_fe.copy()
            deriv_e.f.component.data = dFdt_e
            dFdt_e = deriv_e
        else:
            dFdt_e = None # self.sys_fe.create_like()

        if self.sys_fi is not None and self.evolve_ion:
            if update_force:
                force_term = self.compute_force_term(is_ion=True,
                                                     verbose_plot=verbose_plot, background_force=background_force,
                                                     internal_force=internal_force)
                # self.sys_fi.force_term = force_term
                self.sys_fi.set_force_term(force_term, background_force=background_force, internal_force=internal_force)

            dFdt_i = self.sys_fi.deriv_upwind_global(time=time,
                                                     do_x_advection=do_x_advection,
                                                     do_v_advection=do_v_advection,
                                                     background_force=background_force,
                                                     internal_force=internal_force,
                                                     verbose_plot=verbose_plot)

            deriv_i = self.sys_fi.copy()
            deriv_i.f.component.data = dFdt_i
            dFdt_i = deriv_i
        else:
            dFdt_i = None

        dFdt = self.create_like(dFdt_e, dFdt_i, recalc=False)
        return dFdt
