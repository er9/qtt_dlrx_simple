"""Configuration classes that group together the parameters of a simulation,
including units, species/material, compression, and derivative settings, used to
set up QTT/DLRA runs."""
from setup_.defaults import *
import setup_.paths
from setup_.enums import *
from setup_.helper import *

from typing import Optional
from dataclasses import dataclass

class UnitsConfiguration:
    """Physical units and fundamental constants for a simulation.

    All values default to 1.0 (normalized units). The speed of light ``c`` is
    derived as ``1 / sqrt(eps0 * mu0)``.

    Parameters
    ----------
    e : Numeric, default 1.0
        Elementary charge.
    eps0 : Numeric, default 1.0
        Vacuum permittivity.
    mu0 : Numeric, default 1.0
        Vacuum permeability.
    eV : Numeric, default 1.0
        Energy unit, used in place of the Boltzmann constant ``kB``.
    is_cgs : bool, default False
        If True, use Gaussian (CGS) conventions for derived plasma quantities.
    """

    def __init__(self, e: Numeric = 1.0, eps0: Numeric = 1.0, mu0: Numeric = 1.0, eV: Numeric = 1.0, is_cgs=False):
        self.e = e  ## elementary charge
        self.eV = eV  ## kB
        self.eps0 = eps0
        self.mu0 = mu0
        self.c = 1. / np.sqrt(self.eps0 * self.mu0)
        self.is_cgs = is_cgs


class SpeciesConfiguration:
    """Material parameters for a single plasma species.

    Beyond the stored parameters, this class exposes derived plasma quantities as
    properties/methods (``charge``, ``plasma_frequency``/``wp``, ``vth``,
    ``debye_length``/``lamD``, ``skin_depth``, and the field-dependent
    ``cyclotron_frequency``, ``alfven_speed``, ``larmor_radius``, ``beta``).
    See :class:`IonConfiguration` and :class:`ElcConfiguration` for the ion and
    electron specializations.

    Parameters
    ----------
    Z : Numeric, default 1.0
        Charge number (sign included; positive for ions, negative for electrons).
    n0 : Numeric, default 1.0
        Reference number density.
    mass : Numeric, default 1.0
        Particle mass.
    T : Numeric, default 1.0
        Temperature (in units of ``eV``).
    units_config : UnitsConfiguration, optional
        Units/constants to use; a default :class:`UnitsConfiguration` is created
        if omitted.
    """

    def __init__(self, Z: Numeric = 1.0, n0: Numeric = 1.0, mass: Numeric = 1.0, T: Numeric = 1.0,
                 units_config: 'UnitsConfiguration' = None):
        self.Z = Z  ## charge (positive for ions, negative for elec)
        self.mass = mass  ## electron mass
        self.T = T  ## electron temperature
        self.n0 = n0
        if units_config is None:
            units_config = UnitsConfiguration()
        self.units_config = units_config

    @property
    def units_config(self):
        return self._units_config

    @units_config.setter
    def units_config(self, units_config: 'UnitsConfiguration'):
        self._units_config = units_config
        if units_config is not None:
            self.e = units_config.e  ## elementary charge
            self.eps0 = units_config.eps0
            self.mu0 = units_config.mu0
            self.eV = units_config.eV   ## acts like kB
            self.c = units_config.c
            self.is_cgs = units_config.is_cgs

            # self.wp = np.sqrt(self.charge ** 2 * self.n0 / self.eps0 / self.mass)  # plasma e' frequency
            # self.vth = np.sqrt(self.eV * self.T / self.mass)  # electron thermal speed
            # self.lamD = self.vth / self.wp  # Debye length

    @property
    def charge(self):
        return self.e * self.Z

    @property
    def plasma_frequency(self):
        if self.is_cgs:
            return np.sqrt(4 * np.pi * self.n0 * self.charge ** 2 / self.mass)
        else:
            return np.sqrt(self.n0 * self.charge ** 2 / self.eps0 / self.mass)

    @property
    def wp(self):
        return self.plasma_frequency

    @property
    def vth(self):
        # return np.sqrt(np.abs(self.Z) * self.eV * self.T / self.mass)  # thermal speed
        return np.sqrt(self.eV * self.T / self.mass)  # thermal speed

    @property
    def debye_length(self):
        return self.vth / self.plasma_frequency

    @property
    def lamD(self):
        return self.debye_length

    def cyclotron_frequency(self, B0):
        if self.is_cgs:
            return self.charge * B0 / self.mass / self.c
        else:
            return self.charge * B0 / self.mass

    def alfven_speed(self, B0):
        if self.is_cgs:
            return B0 / np.sqrt(4 * np.pi * self.n0 * self.mass)
        else:
            return B0 / np.sqrt(self.mu0 * self.n0 * self.mass)

    def larmor_radius(self, B0, v_perp=0.):
        if self.is_cgs:
            raise NotImplementedError
        else:
            v_perp = max(v_perp, self.vth)
            return self.mass * v_perp / np.abs(self.Z) / B0

    @property
    def skin_depth(self):  # c / plasma frequency
        return self.c / self.plasma_frequency

    def beta(self, B0):  # 2 vth^2 / vA^2
        return 2 * self.vth ** 2 / self.alfven_speed(B0) ** 2


class IonConfiguration(SpeciesConfiguration):
    """ configuration for ions
    """

    def __init__(self, Z: Numeric = 1.0, n0: Numeric = 1.0, mass: Numeric = 1.0, T: Numeric = 1.0,
                 units_config: 'UnitsConfiguration' = None):
        super().__init__(Z=Z, n0=n0, mass=mass, T=T, units_config=units_config)

    def lower_hybrid_frequency(self, B0, elc_config: 'ElcConfiguration'):
        """ note: for ions """
        return np.sqrt( self.cyclotron_frequency(B0) * elc_config.cyclotron_frequency(B0) )

    # def alfven_speed(self, B0):
    #     return B0 / np.sqrt( self.mu0 * self.n0 * self.mass )
    #
    # @property
    # def skin_depth(self):   # c / ion plasma frequency
    #     return self.c / self.plasma_frequency

    # @property
    # def skin_depth(self):
    #     return np.sqrt( self.mass / self.e ** 2 / self.eps0  / self.n0)

    # def larmor_radius(self, B0, v_perp):
    #     if v_perp is None:
    #         return 1
    #     if self.is_cgs:
    #         raise NotImplementedError
    #     else:
    #         return self.mass * v_perp / np.abs(self.Z) / B0


class ElcConfiguration(SpeciesConfiguration):
    """ configuration for electrons (change sign of Z)
    """

    def __init__(self, Z: Numeric = 1.0, n0: Numeric = 1.0, mass: Numeric = 1.0, T: Numeric = 1.0,
                 units_config: 'UnitsConfiguration' = None):
        super().__init__(Z=abs(Z) * -1, n0=n0, mass=mass, T=T, units_config=units_config)

    # @property
    # def skin_depth(self):
    #     return self.c / self.plasma_frequency      # ion skin depth

    def upper_hybrid_frequency(self, B0):
        """ note: for electrons """
        return np.sqrt( self.plasma_frequency**2 + self.cyclotron_frequency(B0)**2 )

    def right_hand_frequency(self, B0):
        wc = self.cyclotron_frequency(B0)
        wp = self.plasma_frequency
        return 0.5 * (abs(wc) + np.sqrt(wc**2 + 4 * wp**2))

    def left_hand_frequency(self, B0):
        wc = self.cyclotron_frequency(B0)
        wp = self.plasma_frequency
        return 0.5 * (-abs(wc) + np.sqrt(wc ** 2 + 4 * wp ** 2))



class SubCompressConfigType(Enum):
    TT = 'tt'


class CompressionConfiguration:
    """Tensor-network compression parameters, organized by "level".

    Different stages of an algorithm can compress with different parameters; each
    stage is identified by an integer ``compress_level``. Populate a level with
    :meth:`set_compress_opts` and retrieve a quimb-compatible options dict with
    :meth:`get_compress_opts` (or via ``config[level]``). Unset levels fall back to
    the module defaults ``MAXBOND``, ``CUTOFF``, ``CUTOFF_MODE``.

    Parameters
    ----------
    compress_type : CompressType, default CompressType.SVD
        Truncation method. ``SVD`` honors ``cutoff``/``cutoff_mode``; other types
        truncate by ``max_bond`` only.

    Notes
    -----
    ``sub_compress_configs`` holds nested configurations (e.g. for TT compression),
    managed via :meth:`set_sub_compress_opts` / :meth:`get_sub_compress_opts`.
    """

    def __init__(self, compress_type=CompressType.SVD):
        self.compress_type = compress_type
        self.max_bonds = {}
        self.cutoff_modes = {}
        self.cutoffs = {}
        self.do_midpt = False
        self.midpt = None
        self.norm_cutoff = None  # np.sqrt(CUTOFF)

        self.sub_compress_configs: dict['SubCompressConfigType', 'CompressionConfiguration'] = {}

    def __getitem__(self, compress_level):
        return self.get_compress_opts(compress_level)

    def __str__(self):
        str_bonds = 'bonds: ' + str(self.max_bonds)
        str_cutoffs = 'cutoffs: ' + str(self.cutoffs)
        str_modes = 'cutoff modes: ' + str(self.cutoff_modes)
        str_midpt = 'midpt: ' + str(self.do_midpt)
        if self.midpt is not None:    str_midpt += f',{self.midpt}'
        return self.compress_type.value + '\n' + str_bonds + '\n' + str_cutoffs + '\n' + str_modes + '\n' + str_midpt

    def get(self, key, default_val=None):
        return self[key]

    def set_compress_opts(self, compress_level: int, max_bond=None, cutoff_mode=CUTOFF_MODE,
                          cutoff=CUTOFF, norm_cutoff=None):
        """Set the compression parameters for a given level.

        Parameters
        ----------
        compress_level : int
            Level (algorithm stage) these options apply to.
        max_bond : int, optional
            Maximum bond dimension; ``None`` means unbounded.
        cutoff_mode : optional
            Singular-value cutoff mode (e.g. relative-sum-of-squares).
        cutoff : Numeric, default CUTOFF
            Singular-value truncation threshold.
        norm_cutoff : Numeric, optional
            Norm cutoff; defaults to ``sqrt(CUTOFF)`` when not given.
        """
        self.max_bonds[compress_level] = max_bond
        self.cutoff_modes[compress_level] = cutoff_mode
        self.cutoffs[compress_level] = cutoff
        self.norm_cutoff = norm_cutoff if norm_cutoff is not None else np.sqrt(CUTOFF)

    def get_compress_opts(self, compress_level: int) -> dict:
        """Return a quimb-compatible compression options dict for ``compress_level``.

        Falls back to the module defaults (``MAXBOND``, ``CUTOFF``, ``CUTOFF_MODE``)
        for any value not explicitly set. For ``SVD`` compression the dict includes
        ``cutoff``/``cutoff_mode`` (and midpoint options when enabled); otherwise only
        ``max_bond``. ``norm_cutoff`` is always included.

        Parameters
        ----------
        compress_level : int
            Level (algorithm stage) to retrieve options for.

        Returns
        -------
        dict
            Keyword arguments suitable for quimb compression routines.
        """
        max_bond = self.max_bonds.get(compress_level, MAXBOND)
        if self.compress_type == CompressType.SVD:
            cutoff = self.cutoffs.get(compress_level, CUTOFF)
            cutoff_mode = self.cutoff_modes.get(compress_level, CUTOFF_MODE)
            compress_opts = {'max_bond': max_bond, 'cutoff': cutoff, 'cutoff_mode': cutoff_mode}
            if self.do_midpt and compress_level == 1:
                compress_opts['do_midpt'] = True
                if self.midpt is not None:
                    compress_opts['midpt'] = self.midpt
        else:
            compress_opts = {'max_bond': max_bond}  # , 'cutoff': cutoff, 'cutoff_mode': cutoff_mode}
        compress_opts['norm_cutoff'] = self.norm_cutoff
        return compress_opts

    def set_sub_compress_opts(self, sub_key, compress_level: int, max_bond=None, cutoff_mode=CUTOFF_MODE,
                              cutoff=CUTOFF):
        if sub_key not in self.sub_compress_configs:
            compress_config = CompressionConfiguration()
        else:
            # return None     # s.t. compress_opts is same as the main
            compress_config = self.sub_compress_configs[sub_key]

        compress_config.set_compress_opts(compress_level, max_bond=max_bond, cutoff_mode=cutoff_mode, cutoff=cutoff)
        self.sub_compress_configs[sub_key] = compress_config

    def get_sub_compress_opts(self, sub_key, compress_level: int) -> dict:
        compress_opts = self.sub_compress_configs[sub_key].get_compress_opts(compress_level)
        return compress_opts

    def get_all_sub_compress_opts(self, compress_level: int) -> Optional[dict['SubCompressConfigType', dict]]:
        sub_keys = self.sub_compress_configs.keys()

        if len(sub_keys) == 0:
            return None

        compress_opts = {}
        for sub_key, sub_config in self.sub_compress_configs.items():
            compress_opts[sub_key] = sub_config.get_compress_opts(compress_level)
        return compress_opts


class NewCompressionConfiguration:
    """ class containing compression parameters for different levels
    """

    def __init__(self, compress_type=CompressType.SVD):
        self.compress_type = compress_type
        self.max_bonds = {}
        self.cutoff_modes = {}
        self.cutoffs = {}

        ## compress from midpoint
        self.do_midpt = False
        self.midpt = None

        ## embedded config; e.g. for TT compression
        self.sub_compress_configs: dict['SubCompressConfigType', 'CompressionConfiguration'] = {}

    def __getitem__(self, compress_level):
        return self.get_compress_opts(compress_level)

    def __str__(self):
        str_bonds = 'bonds: ' + str(self.max_bonds)
        str_cutoffs = 'cutoffs: ' + str(self.cutoffs)
        str_modes = 'cutoff modes: ' + str(self.cutoff_modes)
        str_midpt = 'midpt: ' + str(self.do_midpt)
        if self.midpt is not None:    str_midpt += f',{self.midpt}'
        return str_bonds + '\n' + str_cutoffs + '\n' + str_modes + '\n' + str_midpt

    def get(self, compress_level: int, if_none=None) -> dict:
        try:
            return self.get_compress_opts(compress_level)
        except KeyError:
            return if_none

    def set_compress_opts(self, compress_level: int, max_bond=None, cutoff_mode=CUTOFF_MODE,
                          cutoff=CUTOFF):
        self.max_bonds[compress_level] = max_bond
        self.cutoff_modes[compress_level] = cutoff_mode
        self.cutoffs[compress_level] = cutoff

    def get_compress_opts(self, compress_level: int) -> dict:
        max_bond = self.max_bonds.get(compress_level, MAXBOND)
        cutoff = self.cutoffs.get(compress_level, CUTOFF)
        cutoff_mode = self.cutoff_modes.get(compress_level, CUTOFF_MODE)
        compress_opts = {'max_bond': max_bond, 'cutoff': cutoff, 'cutoff_mode': cutoff_mode}
        if self.do_midpt and compress_level == 1:
            compress_opts['do_midpt'] = True
            if self.midpt is not None:
                compress_opts['midpt'] = self.midpt
        return compress_opts

    def set_sub_compress_opts(self, sub_key, compress_level: int, max_bond=None, cutoff_mode=CUTOFF_MODE,
                              cutoff=CUTOFF):
        if sub_key not in self.sub_compress_configs:
            compress_config = CompressionConfiguration()
        else:
            # return None     # s.t. compress_opts is same as the main
            compress_config = self.sub_compress_configs[sub_key]

        compress_config.set_compress_opts(compress_level, max_bond=max_bond, cutoff_mode=cutoff_mode, cutoff=cutoff)
        self.sub_compress_configs[sub_key] = compress_config

    def get_sub_compress_opts(self, sub_key, compress_level: int) -> dict:
        compress_opts = self.sub_compress_configs[sub_key].get_compress_opts(compress_level)
        return compress_opts

    def get_all_sub_compress_opts(self, compress_level: int) -> Optional[dict['SubCompressConfigType', dict]]:
        sub_keys = self.sub_compress_configs.keys()

        if len(sub_keys) == 0:
            return None

        compress_opts = {}
        for sub_key, sub_config in self.sub_compress_configs.items():
            compress_opts[sub_key] = sub_config.get_compress_opts(compress_level)
        return compress_opts


class DerivativeConfiguration:
    """Boundary conditions and finite-difference scheme for one axis.

    Specifies how derivatives along an axis are taken: the left/right boundary
    conditions, the finite-difference type and order, optional symmetry offsets, and
    boundary values. Used per-axis by :class:`~gridTN_1D.GridTN1D` and the field
    differential operators. Helper methods :meth:`derivative_bc` and
    :meth:`shifted_bc` derive related configurations (e.g. the BCs of ``d/dx f``).

    Parameters
    ----------
    left_bc : BCType or str, default DEFAULT_BC
        Boundary condition at the left end (periodic, antiperiodic, open,
        reflecting, zero-gradient, Dirichlet, Neumann, ...).
    right_bc : BCType or str, optional
        Boundary condition at the right end; defaults to ``left_bc``.
    order : int, default DEFAULT_ORDER
        Finite-difference accuracy order.
    fd_type : FDType or str, default DEFAULT_FDTYPE
        Finite-difference stencil type (forward, backward, center, upwind).
    offset : int, default 0
        Offset of the boundary (point of symmetry) from the left end, in half
        steps. ``0`` puts the boundary on the symmetry point; ``> 0`` places the
        symmetry point inside the data; ``< 0`` (min ``-1``) outside it. Flipped for
        ``right_bc``.
    offset_r : int, optional
        Independent offset for the right end. If equal to ``offset`` the whole grid
        is effectively shifted; otherwise it acts as the opposite of ``offset``.
        Defaults to ``offset``.
    value_l, value_r : Numeric, default 0.0
        Boundary values (e.g. for Dirichlet/Neumann conditions) at the left/right
        ends.
    """

    def __init__(self,
                 left_bc: BCType or str = DEFAULT_BC,
                 right_bc: BCType or str = None,
                 order: int = DEFAULT_ORDER,
                 fd_type: FDType or str = DEFAULT_FDTYPE,
                 offset: int = 0, offset_r: Optional[int] = None,
                 value_l: Numeric = 0.0, value_r: Numeric = 0.0):
        """
            offset: offset of boundary condition from left-hand side -- flipped for right_bc
                    (eg. in SYMMETRIC and ASYMMETRIC case)
                    offset = 0: boundary is the point of symmetry
                    offset > 0: point of symmetry is within data (offset specifies distance from boundary)
                    offset < 0 (minimum = -1): point of symmetry is not within data (offset specifies distance)
                    offset value indicates half steps (eg. x_[-1], x_0, x_1 vs x_[-1/2], x_[1/2]_
            offset_r:  can separately set the offset on the right side
                    if offset_r is the same as offset, then it's as if the entire grid was shifted
                    otherwise, acts as the opposite of offset
            value: value at the boundary condition (e.g. Neumann or Dirichlet)
        """

        self._left_bc = parse_BCType(left_bc)
        if right_bc is None:
            right_bc = left_bc
        self._right_bc = parse_BCType(right_bc)
        self._order = order
        self._offset = offset
        self._offset_r = offset_r
        self._fdtype = parse_FDType(fd_type)
        self._value_l = value_l
        self._value_r = value_r

        # # other params only used in some cases
        # self.v_ax: Optional['Axis'] = None

    def __str__(self):
        return 'DerivConfig' + str(vars(self))

    def __repr__(self):
        return 'DerivConfig' + str(vars(self))

    def copy(self):
        return self.__class__(left_bc=self.left_bc, right_bc=self.right_bc, order=self.order,
                              fd_type=self.fd_type, offset=self.offset, offset_r=self._offset_r,
                              value_l=self._value_l, value_r = self._value_r)

    @property
    def left_bc(self) -> BCType:
        return self._left_bc

    # @left_bc.setter
    # def left_bc(self, left_bc: BCType or str):
    #     self._update_left_bc(left_bc)

    @property
    def right_bc(self) -> BCType:
        return self._right_bc

    # @right_bc.setter
    # def right_bc(self, right_bc: BCType or str):
    #     self._update_right_bc(right_bc)

    @property
    def bc(self) -> tuple[BCType, BCType]:
        return self._left_bc, self._right_bc

    # @bc.setter
    # def bc(self, bc: BCType or str):
    #     self.left_bc = bc
    #     self.right_bc = bc

    @property
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, new_order: int):
        self._update_order(new_order)

    @property
    def fd_type(self) -> FDType:
        return self._fdtype

    @fd_type.setter
    def fd_type(self, new_fdtype: FDType or str):
        self._update_fdtype(new_fdtype)

    @property
    def offset(self) -> int:
        return self._offset

    # @offset.setter
    # def offset(self, new_offset: int):
    #     self._update_offset(new_offset)

    @property
    def offset_r(self) -> int:
        # return self._offset - 2 if self._offset_r is None else self._offset_r
        return self._offset if self._offset_r is None else self._offset_r

    @property
    def value_l(self) -> Numeric:
        return self._value_l

    @property
    def value_r(self) -> Numeric:
        return self._value_r

    @property
    def deriv_params(self):
        return {'left_bc': self._left_bc, 'right_bc': self._right_bc, 'order': self._order, 'fd_type': self._fdtype,
                'offset': self._offset, 'offset_r': self.offset_r, 'bc_value': self._value_l, 'bc_value_r': self._value_r}

    ## i want to get rid of this.
    def update(self, left_bc: Optional[BCType or str] = None, right_bc: Optional[BCType or str] = None,
               order: int = None, fd_type: 'FDType' = None, offset: int = None):  # , v_ax:'Axis' = None):
        if left_bc is not None:
            raise RuntimeError
            # self._left_bc = left_bc
            # self._right_bc = self.left_bc
        if right_bc is not None:
            raise RuntimeError
            # self._right_bc = right_bc
        if order is not None:
            self.order = order
        if fd_type is not None:
            self.fd_type = fd_type
        if offset is not None:
            raise RuntimeError
            # self._offset = offset
        # if v_ax is not None:
        #     self.v_ax = v_ax

    def _update_left_bc(self, new_left_bc):
        self._left_bc = parse_BCType(new_left_bc)

    def _update_right_bc(self, new_right_bc):
        self._right_bc = parse_BCType(new_right_bc)

    def _update_order(self, new_order: int):
        assert (isinstance(new_order, int)), 'order must be an integer'
        self._order = new_order

    def _update_fdtype(self, new_fdtype: FDType or str):
        self._fdtype = parse_FDType(new_fdtype)

    def _update_offset(self, new_offset: int):
        self._offset = new_offset

    def get_deriv_key(self):
        # if self._fdtype == FDType.UPWIND:
        #     assert(self.v_ax is not None), 'need to set v_axis for upwind FD method'
        #     return (self._left_bc, self._right_bc), self._order, self._fdtype, self.v_ax.axID
        # else:
        return (self._left_bc, self._right_bc), self._order, self._fdtype, (self._offset, self._offset_r),\
               (self._value_l, self._value_r)

    def derivative_bc(self) -> 'DerivativeConfiguration':
        ## boundary conditions of d/dx f, where self is boundary conditoins of f(x)
        if self._left_bc in [BCType.ANTISYMMETRIC, BCType.ZEROVALUE, BCType.DIRICHLET]:
            new_left = BCType.SYMMETRIC
        elif self._left_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
            new_left = BCType.ANTISYMMETRIC
        else:
            new_left = self._left_bc

        if self._right_bc in [BCType.ANTISYMMETRIC, BCType.ZEROVALUE, BCType.DIRICHLET]:
            new_right = BCType.SYMMETRIC
        elif self._right_bc in [BCType.SYMMETRIC, BCType.ZEROGRADIENT, BCType.NEUMANN]:
            new_right = BCType.ANTISYMMETRIC
        else:
            new_right = self._right_bc

        out = self.copy()
        # out.update(left_bc=new_left, right_bc=new_right)
        out._left_bc = new_left
        out._right_bc = new_right
        return out

    def shifted_bc(self, inverse=False) -> 'DerivativeConfiguration':
        ## shift cells to midpoints (shift by 1/2)
        # new_offset = (self.offset - 1) if inverse else (self.offset + 1)
        out = self.copy()
        out._offset = self.offset - (1 if inverse else -1)
        if out._offset_r is not None:
            out._offset_r = out._offset_r - (1 if inverse else -1)
        return out


class CollisionConfiguration:
    """Collision-operator parameters for electrons and ions.

    Builds a :class:`SpeciesCollisionConfiguration` for each of electrons and ions
    and exposes their rates/coefficients. ``coll_type`` selects the collision model
    (e.g. ``'LB'`` Lenard-Bernstein, ``'H2'``/``'H4'``/``'H6'`` hypocoercive models),
    normalized to a :class:`CollisionType`.

    Parameters
    ----------
    coll_type : str or CollisionType
        Collision model identifier.
    coeff_e, coeff_i : Numeric, default 1.0
        Collision coefficients for electrons / ions.
    rate_e, rate_i : Numeric, default 1.0
        Base collision rates for electrons / ions.
    v0_e, v0_i : Numeric, default 0.0
        Reference drift velocities for electrons / ions.

    Notes
    -----
    Interspecies collisions are not yet supported.
    """

    def __init__(self, coll_type, coeff_e=1.0, coeff_i=1.0, rate_e=1.0, rate_i=1.0, v0_e=0.0, v0_i=0.0):
        # {'rate_e':coll_rate_e, 'rate_i':coll_rate_i, 'v0_e':0.0, 'v0_i':0.0}

        self.collisions_e = SpeciesCollisionConfiguration(coll_type, coeff=coeff_e, rate=rate_e, v0=v0_e)
        self.collisions_i = SpeciesCollisionConfiguration(coll_type, coeff=coeff_i, rate=rate_i, v0=v0_i)

        self.coll_type = coll_type
        if coll_type in ['LB', 'lb', 'Lenard-Bernstein', 'LBO']:
            self.coll_type = CollisionType.LB
        elif coll_type in ['H6']:
            self.coll_type = CollisionType.H6
        elif coll_type in ['H4']:
            self.coll_type = CollisionType.H4
        elif coll_type in ['H2']:
            self.coll_type = CollisionType.H2

    def __str__(self):
        if self.coll_type is None:
            return 'colltype None'
        return self.coll_type.value + ': ' + str(vars(self))

    @property
    def coll_rate_e(self):
        return self.collisions_e.coll_rate

    @property
    def coll_rate_i(self):
        return self.collisions_i.coll_rate

    @property
    def base_rate_e(self):
        return self.collisions_e.base_rate

    @base_rate_e.setter
    def base_rate_e(self, new_base_rate):
        self.collisions_e.base_rate = new_base_rate

    @property
    def base_rate_i(self):
        return self.collisions_i.base_rate

    @base_rate_i.setter
    def base_rate_i(self, new_base_rate):
        self.collisions_i.base_rate = new_base_rate

    @property
    def coll_params(self):
        return vars(self)

    def get_base_rate_e(self, dim=0):
        return self.collisions_e.get_base_rate(dim=dim)

    def get_base_rate_i(self, dim=0):
        return self.collisions_i.get_base_rate(dim=dim)

    def get_coeff_e(self, dim=0):
        return self.collisions_e.get_coeff(dim=dim)

    def get_coeff_i(self, dim=0):
        return self.collisions_i.get_coeff(dim=dim)

    def get_coll_rate_e(self, dim=0):
        return self.collisions_e.get_coll_rate(dim=dim)

    def get_coll_rate_i(self, dim=0):
        return self.collisions_i.get_coll_rate(dim=dim)

    def get_v0_e(self, dim=0):
        return self.collisions_e.get_v0(dim=dim)

    def get_v0_i(self, dim=0):
        return self.collisions_i.get_v0(dim=dim)


class SpeciesCollisionConfiguration:
    def __init__(self, coll_type, coeff=1.0, rate=1.0, v0=0.0):
        # {'rate_e':coll_rate_e, 'rate_i':coll_rate_i, 'v0_e':0.0, 'v0_i':0.0}

        self.coll_type = coll_type

        if coll_type is not None:
            self.coeff: Union[np.ndarray, Numeric] = coeff
            self.base_rate: Union[np.ndarray, Numeric] = rate
            self.v0: Union[np.ndarray, Numeric] = v0
        else:
            self.coeff = 0.
            self.base_rate = 0.
            self.v0 = 0.

        if coll_type in ['LB', 'lb', 'Lenard-Bernstein', 'LBO']:
            self.coll_type = CollisionType.LB

    def __str__(self):
        if self.coll_type is None:
            return 'colltype None'
        return self.coll_type.value + ': ' + str(vars(self))

    @property
    def coll_rate(self):
        return self.get_coeff() * self.get_base_rate()

    @property
    def coll_params(self):
        return vars(self)

    def get_base_rate(self, dim=0):
        try:
            return self.base_rate[dim]
        except (IndexError, TypeError):
            return self.base_rate

    def get_coeff(self, dim=0):
        try:
            return self.coeff[dim]
        except (IndexError, TypeError):
            return self.coeff

    def get_coll_rate(self, dim=0):
        return self.get_base_rate(dim) * self.get_coeff(dim)

    def get_v0(self, dim=0):
        try:
            return self.v0[dim]
        except (IndexError, TypeError):
            return self.v0
