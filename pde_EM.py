import numpy as np

import helper_quimb
from setup_.configs import *
from helper_tdvp import TDVPSolver0
import matplotlib.pyplot as plt
import helper_quimb as helper
from gridTN_1D import GridTN1D
from gridTN_1Dcomb import GridTN1DComb
from field import Field, ScalarField
from pde_system import PDE_system

from helper_dmrg import SolveMethod, SweepDirection
# import helper_block_dmrg as helper_block
# import helper_block_dmrg_2 as helper_block_2

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coord.coord_sys import CoordinateSystem
    from grid import Grid
    from gridTN import GridTN

"""
Munz et al 2000 FV method for Maxwell's equations
finite volume method with upwinding, assuming rectangular grid
ijk = cell position in 3D space
beta = edge
n_{ijk,beta} = (n1,n2,n3)  directed unit normal to edge S_{ijk,beta} 
u_{ijk}^n is 1/V_{ijk} integral over volume V_{ijk} u(x(i,j,k), t_n)
u_{ijk}^(n+1) = u_{ijk} - 1/|V_ijk| int_0^dt integral over boundary Flux(u) n_{ij} dS dt 
                        + 1/|V_ijk| int_0^dt integral over volume q(u) dV dt
    where V_{ijk} = dx dy dz for rectangular grid
          q(u) is the source (and will include terms for the geometry)
    a split step scheme is used when including the source (at time n+1/2: q_{ijk}^(n+1/2))

numerical flux approximation:
G_{ijk,beta}^n = |S_{ijk,beta}| (n1 K1 + n2 K2 + n3 K3) u(x_MP,tn)
where x_MP is the midpoint of the edge beta (e.g., x+dx/2 for rectangular grid)

u(x_MP,t_n) is calculated via d/dt u = C_{ijk,beta} d/dr u = 0
where u(r,0) = u_l for r < 0;  u_r for r > 0;  i believe r is the coordinate along unit normal n_{ijk,beta}
      C_{ijk,beta} = (n1 K1 + n2 K2 + n3 K3)
      K1, K2, K3 as defined in paper (6x6 matrices)

compute eigenvalues, right and left eigenvectors.
e.g., for (n1,n2,n3) = (a,0,0)   (a = +/- 1)
     lam1 = lam2 = -c, lam3 = lam4 = 0, lam5 = lam6 = c
     right_1 = (0, -ac, 0, 0, 0, 1).T
     right_2 = (0, 0, c, 0, a, 0).T
     right_3 = (a, 0, 0, 0, 0, 0).T
     right_4 = (0, 0, 0, a, 0, 0).T
     right_5 = (0, ac, 0, 0, 0, 1).T
     right_6 = (0, 0, -c, 0, a, 0).T  --> eigR = (r1,r2,...,r6)
     left_1 = (0, -a, 0, 0, 0, c).T  / 2c
     left_2 = (0, 0, 1, 0, ac, 0).T  / 2c
     left_3 = (2ac, 0, 0, 0, 0, 0).T / 2c
     left_4 = (0, 0, 0, 2ac, 0, 0).T / 2c
     left_5 = (0, a, 0, 0, 0, c).T   / 2c
     left_6 = (0, 0, -1, 0, ac, 0).T / 2c ... --> eigL = (l1,l2,...,l6)

intermediate constant states:
    alpha_m = left_m (u_r - u_l) for m = 1,...,6
    u_1 = u_l + \sum_{m=1,2} alpha_m r_m = u_r - \sum_{m=3,4,5,6} alpha_m r_m   
    u_2 = u_l + \sum_{m=1,2,3,4} alpha_m r_m = u_r - \sum_{m=5,6} alpha_m r_m

numerical flux:
    G_{ijk,beta)^n = |S_{ijk,beta}| C_{ijk,beta} u_1 = |S_{ijk,beta} C_{ijk,beta} u_2
                   = |S_{ijk,beta}| C^+_{ijk,beta} u_l + C^-_{ijk,beta} u_r
                        with C^{+/-}_{ijk,beta} = 0.5 ( C_{ijk,beta} +/- |C_{ijk,beta}| )
                        and |C_{ijk,beta}| = eigR diag(c,c,0,0,c,c) eigL
                        C^+: nonnegative eigenvalues
                        C^-: nonpositive eigenvalues

    |C_{ijk,beta}| 

K matrices in SI units, rectangular grid
K_i = [0,   -c^2 M_i
       M_i,    0    ]
where M_1 = [0,0,0; 0,0,-1; 0,1,0]      i.e. the cross product
      M_2 = [0,0,1; 0,0,0; -1,0,0]
      M_3 = [0,-1,0; 1,0,0; 0,0,0]

C_{ijk,beta} = n1 K1 + n2 K2 + n3 K3
[ 0,   0,   0,   0,      c^2 n3,  -c^2 n2  ]
[ 0,   0,   0, -c^2 n3,    0,      c^2 n1  ]
[ 0,   0,   0,  c^2 n2, -c^2 n1,     0     ]
[ 0,  -n3,  n2,  0,        0,        0,    ]
[ n3,  0,  -n1,  0,        0,        0,    ]
[-n2,  n1,  0,   0,        0,        0,    ]

for (n1,n2,n3) = (+/-1,0,0)
|C_{ijk}| = c Diag[ 0, Abs[n1], Abs[n1], 0, Abs[n1], Abs[n1]]
for (n1,n2,n3) = (0,+/-1,0)
|C_{ijk}| = c Diag[ Abs[n2], 0, Abs[n2], Abs[n2], 0, Abs[n2]]
for (n1,n2,n3) = (0,0,+/-1)
|C_{ijk}| = c Diag[ Abs[n3], Abs[n3], 0, Abs[n3], Abs[n3], 0]

C+/-_{ijk} for (n1,0,0)
[ 0,   0,   0,   0,     0,         0    ]
[ 0, +/-c,  0,   0,     0,       c^2 n1 ]
[ 0,   0, +/-c,  0,   -c^2 n1,     0    ]
[ 0,   0,   0,   0,     0,         0,   ]
[ 0,   0,  -n1,  0,    +/-c,       0,   ]
[ 0,  n1,   0,   0,     0,       +/-c,  ]  * 0.5

C+/-_{ijk} for (0,n2,0)
[+/-c, 0,   0,     0,     0, -c^2 n2  ]
[ 0,   0,   0,     0,     0,     0    ]
[ 0,   0, +/-c,  c^2 n2,  0,     0    ]
[ 0,   0,   n2,   +/-c,   0,     0,   ]
[ 0,   0,   0,     0,     0,     0,   ]
[-n2,  0,   0,     0,     0,   +/-c,  ]  * 0.5

C+/-_{ijk} for (0,0,n3)
[+/-c,  0,   0,     0,   c^2 n3,  0,   ]
[ 0,  +/-c,  0,  -c^2 n3   0,     0    ]
[ 0,    0,   0,     0,     0,     0    ]
[ 0,   -n3,  0,   +/-c,    0,     0,   ]
[ n3,   0,   0,     0,   +/-c,    0,   ]
[ 0,    0,   0,     0,     0,     0    ]  * 0.5

approx flux:
G_{ijk,beta)^n = |S_{ijk,beta}| C^+_{ijk,beta} u_l + C^-_{ijk,beta} u_r

u_{ijk}^{n+1} = u_{ijk}^{n} - dt/|V_{ijk}| sum_{ijk,beta=1^6} G_{ijk,beta}^n + q^{n+1/2}

u = (E1,E2,E3,B1,B2,B3)
dE/dt = c^2 curl(B) - 1/eps0 J0
dEx/dt = c^2 (d/dy Bz - d/dz By) - 1/eps0 Jx
dEy/dt = c^2 (d/dz Bx - d/dx Bz) - 1/eps0 Jy
dEz/dt = c^2 (d/dx By - d/dy Bx) - 1/eps0 Jz

at (i+1/2,j+1/2,k+1/2)
Ex(t_n+1) = Ex(t_n) + dt/V ( x: 0
                             y: c/2 (Ex_{i,j,k} - Ex_{i,j+1,k}) - c^2/2 (Bz_{i,j,k} + Bz_{i,j+1,k})        ## right boundary with C+, C-
                             y: c/2 (Ex_{i,j-1,k} - Ex_{i,j,k}) + c^2/2 (Bz_{i,j-1,k} + Bz_{i,j,k})   ## left boundary with C+, C-
                             z: c/2 (Ex_{i,j,k} - Ex_{i,j,k+1}) + c^2/2 (By_{i,j,k} + By_{i,j,k+1})        ## top boundary with C+, C-
                             z: c/2 (Ex_{i,j,k-1} - Ex_{i,j,k}) - c^2/2 (By_{i,j,k-1} + By_{i,j,k})      ## bottom boundary with C+, C-
                           ) 
    Wrong!  --> 0 = dEx/dt -c (d/dy + d/dz) Ex - c^2 (d/dy Bz - d/dz By) [w/ second order centered stencil (takes care of /2)]
            --> 0 = dEx/dt -c (dy d2/dy2 + dz d2/dz2) Ex - c^2 (d/dy Bz - d/dz By) [w/ second order centered stencil (takes care of /2)]

Ey(t_n+1) = Ey(t_n) - dt/V ( x: c/2 (Ey_{i,j,k} - Ey_{i+1,j,k}) + c^2/2 (Bz_{i,j,k} + Bz_{i+1,j,k})  ## right boundary with C+, C-
                             x: c/2 (Ey_{i-1,j,k} - Ey_{i,j,k}) - c^2/2 (Bz_{i-1,j,k} + Bz_{i,j,k})  ## left boundary with C+, C-
                             y: 0
                             z: c/2 (Ey_{i,j,k} - Ey_{i,j,k+1}) - c^2/2 (Bx_{i,j,k} + Bx_{i,j,k+1})      ## top boundary with C+, C-
                             z: c/2 (Ey_{i,j,k-1} - Ey_{i,j,k}) + c^2/2 (Bx_{i,j,k-1} + Bx_{i,j,k})    ## bottom boundary with C+, C-
                           ) 
    Wrong!  --> 0 = dEy/dt -c (d/dx + d/dz) Ey - c^2 (d/dz Bx - d/dx Bz) [second order centered stencil]
            --> 0 = dEy/dt -c (dx d2/dx2 + dz d2/dz2) Ey - c^2 (d/dz Bx - d/dx Bz) [second order centered stencil]

Ey(t_n+1) = Ey(t_n) - dt/V ( ... ) 
    Wrong!  --> 0 = dEz/dt - c (d/dx + d/dy) Ez - c^2 (d/dx By - d/dy Bx) [second order centered stencil]
            --> 0 = dEz/dt - c (dx d2/dx2 + dy d2/dy2) Ez - c^2 (d/dx By - d/dy Bx) [second order centered stencil]


at (i+1/2,j+1/2,k+1/2)
Bx(t_n+1) = Bx(t_n) + dt/V ( x: 0
                             y: c/2 (Bx_{i,j,k} - Bx_{i,j+1,k}) + 1/2 (Ez_{i,j,k} + Ez_{i,j+1,k})        ## right boundary with C+, C-
                             y: c/2 (Bx_{i,j-1,k} - Bx_{i,j,k}) - 1/2 (Ez_{i,j-1,k} + Ez_{i,j,k})   ## left boundary with C+, C-
                             z: c/2 (Bx_{i,j,k} - Bx_{i,j,k+1}) - 1/2 (Ey_{i,j,k} + Ey_{i,j,k+1})        ## top boundary with C+, C-
                             z: c/2 (Bx_{i,j,k-1} - Bx_{i,j,k}) + 1/2 (Ey_{i,j,k-1} + Ey_{i,j,k})      ## bottom boundary with C+, C-
                           ) 
    Wrong!  --> 0 = dBx/dt - c (d/dy + d/dz) Bx + (d/dy Ez - d/dz Ey) [w/ second order centered stencil]
            --> 0 = dBx/dt - c (dy d2/dy2 + dz d2/dz2) Bx + (d/dy Ez - d/dz Ey) [w/ second order centered stencil]

By(t_n+1) = By(t_n) - dt/V ( x: c/2 (By_{i,j,k} - By_{i+1,j,k}) - (Ez_{i,j,k} + Ez_{i+1,j,k})  ## right boundary with C+, C-
                             x: c/2 (By_{i-1,j,k} - By_{i,j,k}) + (Ez_{i-1,j,k} + Ez_{i,j,k})  ## left boundary with C+, C-
                             y: 0
                             z: c/2 (By_{i,j,k} - By_{i,j,k+1}) + (Ex_{i,j,k} + Ex_{i,j,k+1})      ## top boundary with C+, C-
                             z: c/2 (By_{i,j,k-1} - By_{i,j,k}) - (Ex_{i,j,k-1} + Ex_{i,j,k})    ## bottom boundary with C+, C-
                           ) 
    Wrong!  --> 0 = dBy/dt - c (d/dx + d/dz) By + (d/dz Ex - d/dx Ey) [second order centered stencil]
            --> 0 = dBy/dt - c (dx d2/dx2 + dz d2/dz2) By + (d/dz Ex - d/dx Ey) [second order centered stencil]

Bz(t_n+1) = Bz(t_n) - dt/V ( ... ) 
    Wrong!  --> 0 = dBz/dt - c (d/dx + d/dy) Bz + (d/dx Ey - d/dy Ex) [second order centered stencil]
            --> 0 = dBz/dt - c (dx d2/dx2 + dy d2/dy2) Bz + (d/dx Ey - d/dy Ex) [second order centered stencil]

"""


class Maxwell(PDE_system):
    """ Maxwell's equations solver
    """

    def __init__(self,
                 E: Optional[Field], B: Optional[Field],
                 phi: Optional[ScalarField] = None, psi: Optional[ScalarField] = None,
                 grid_X: Optional['Grid'] = None,
                 coords_x: Optional['CoordinateSystem'] = None,
                 matl_params: Optional[UnitsConfiguration] = None,
                 background_E0: Optional['Field'] = None,
                 background_B0: Optional['Field'] = None,
                 is_electrostatic=False, is_yee=False, clean=True,
                 normalize=False, upwind=False, zipup=False,
                 te_order=2, compress_levels=None, conservative=False,
                 ):
        """ dist_e:  distribution of electrons (scalar Field obj). generally on x,v grid
            dist_i:  distribution of ions (scalar Field obj). generally on x,v grid
            potential:  electric potential (scalar Field obj). generally on x grid
            velocity_grid: velocity grid (vector Field obj) with components vx, vy, ...
                           generally on v grid
            f0_gradv:   grad_v(f0) (vector Field obj) if doing linearized vlasov.
            x_axes:  GRID axes (0,...,self.ndim-1) corresponding to spatial positions in fe, fi
        """
        self.names = {}
        self.names['E'] = E.name if E is not None else 'E'
        self.names['B'] = B.name if B is not None else 'B'
        self.names['phi'] = phi.name if phi is not None else 'phi'
        self.names['psi'] = psi.name if psi is not None else 'psi'

        field_names = [self.names[x] for x in ['E', 'B', 'phi', 'psi']]

        super().__init__(E, B, phi, psi, field_names=field_names,
                         normalize=normalize, te_order=te_order, compress_levels=compress_levels,
                         conservative=False,  ## right now assume only mass conservation
                         )

        self.coords_x: 'CoordinateSystem' = coords_x

        self.init_norm_E = None
        self.init_norm_B = None

        if E is not None:
            self.grid_X = E.grid
        elif B is not None:
            self.grid_X = B.grid
        else:
            self.grid_X = grid_X

        ## background fields
        self.background_E0 = background_E0
        self.divE0 = None
        self.curlE0 = None

        self.background_B0 = background_B0
        self.divB0 = None
        self.curlB0 = None

        ## species parameters
        self.matl_params = UnitsConfiguration() if matl_params is None else matl_params
        self.is_electrostatic = is_electrostatic
        self.upwind = upwind
        self.zipup = zipup
        self.clean = clean
        self.is_yee = is_yee
        if is_yee:  ## hardcode derivative configurations
            # if self.B is not None:
            #     for compID, comp in self.B.components.items():
            #         for ax in self.grid_X.axes:
            #             deriv_config = comp.ax_deriv_configs[ax].copy()
            #             deriv_config.update(order=0, fd_type=FDType.BACKWARD)
            #             comp.ax_deriv_configs[ax] = deriv_config
            #
            # if self.E is not None:
            #     for compID, comp in self.E.components.items():
            #         for ax in self.grid_X.axes:
            #             deriv_config = comp.ax_deriv_configs[ax].copy()
            #             deriv_config.update(order=0, fd_type=FDType.FORWARD)
            #             comp.ax_deriv_configs[ax] = deriv_config

            self.te_order = 31  # split-step time evolution
            self.clean = False

        ## divergence cleaning params
        self.chi = 1
        self.gamma = 1

        ## adaptive solver parameter
        self.nt = 1
        self.adj_conv_tol = None

        ## charge and current densities
        self.charge_density = None
        self.current_density = None

        self.verbose_plot = False

    @property
    def E(self) -> Optional['Field']:
        try:
            return self._fields[self.names['E']]
        except KeyError:
            return None

    @E.setter
    def E(self, new_field: 'Field'):
        self._fields[self.names['E']] = new_field

    @property
    def B(self) -> Optional['Field']:
        try:
            return self._fields[self.names['B']]
        except KeyError:
            return None

    @B.setter
    def B(self, new_field: 'Field'):
        self._fields[self.names['B']] = new_field

    @property
    def tot_B(self) -> Optional['Field']:
        pert_B = self.B
        if pert_B is None:
            return self.background_B0
        else:
            return pert_B.add(self.background_B0)

    @property
    def tot_E(self) -> Optional['Field']:
        pert_E = self.E
        if pert_E is None:
            return self.background_E0
        else:
            return pert_E.add(self.background_E0)

    @property
    def phi(self) -> Optional['ScalarField']:
        try:
            return self._fields[self.names['phi']]
        except KeyError:
            return None

    @phi.setter
    def phi(self, new_field: 'ScalarField'):
        self._fields[self.names['phi']] = new_field

    @property
    def psi(self) -> Optional['ScalarField']:
        try:
            return self._fields[self.names['psi']]
        except KeyError:
            return None

    @psi.setter
    def psi(self, new_field: 'ScalarField'):
        self._fields[self.names['psi']] = new_field

    def create_like(self, *new_fields, recalc=True, deep=False):
        """ create a new system like this with new fields
        """
        if len(new_fields) < 4:
            new_fields = new_fields + (None,) * (4 - len(new_fields))

        E, B, phi, psi = new_fields[:4]
        if recalc:
            new_system = self.__class__(E, B, phi=phi, psi=psi, grid_X=self.grid_X,
                                        coords_x=self.coords_x, matl_params=self.matl_params,
                                        background_E0=self.background_E0, background_B0=self.background_B0,
                                        normalize=self.do_normalization, upwind=self.upwind, zipup=self.zipup,
                                        clean=self.clean, is_yee=self.is_yee, is_electrostatic=self.is_electrostatic,
                                        te_order=self.te_order,
                                        compress_levels=self._comp_levels,
                                        )
        else:
            new_system = self.__class__(E, B, phi=phi, psi=psi, grid_X=self.grid_X,
                                        coords_x=self.coords_x, matl_params=self.matl_params,
                                        background_E0=self.background_E0, background_B0=self.background_B0,
                                        normalize=False, upwind=self.upwind, zipup=self.zipup,
                                        clean=self.clean, is_yee=False,
                                        is_electrostatic=self.is_electrostatic,
                                        te_order=self.te_order,
                                        compress_levels=self._comp_levels,
                                        )
            new_system.do_normalization = self.do_normalization
            new_system.init_norm_E = self.init_norm_E
            new_system.init_norm_B = self.init_norm_B
            new_system.is_yee = self.is_yee

        if E is None:           new_system.names['E'] = self.names['E']
        if B is None:           new_system.names['B'] = self.names['B']
        if phi is None:         new_system.names['phi'] = self.names['phi']
        if psi is None:         new_system.names['psi'] = self.names['psi']

        new_system.divE0 = self.divE0
        new_system.divB0 = self.divB0
        new_system.curlE0 = self.curlE0
        new_system.curlB0 = self.curlB0

        new_system.charge_density = self.charge_density
        new_system.current_density = self.current_density

        new_system.nt = self.nt
        new_system.adj_conv_tol = self.adj_conv_tol
        new_system.verbose_plot = self.verbose_plot

        return new_system

    def kspace_filter(self, filter_func=None, inplace=False):
        """ perform filter in Fourier space (assuming currently in real space)
        """
        new_sys = self if inplace else self.copy()
        grid = self.grid_X
        ks = {}
        for ax in grid.axes:
            ks_mps = ax.get_qft_freqs()
            k_vals = ax.map_mps_to_state(ks_mps)
            ks = {ax: k_vals}

        def filter_func(k_dict, alpha=32, gamma=16):
            ks_mesh = np.meshgrid(*[k_dict[ax] for ax in grid.axes], indexing='ij')
            ks_mag = np.sqrt(np.sum([ks_x ** 2 for ks_x in ks_mesh], 0))
            N = np.max(np.abs(ks_mag))
            out = np.exp(-alpha * (ks_mag / N) ** gamma)
            return out

        filter_mult = filter_func(ks)
        filter_mps = grid.map_state_to_mps(filter_mult)

        for compID, EC in new_sys.E.components.items():
            EC_k = EC.take_qft()
            # helper.mps_flip_lr(EC_k.data, inplace=True)
            #
            # plt.figure()
            # plt.plot(EC_k.get_data())
            # plt.show()

            # EC_k = EC_k.elemental_multiply(filter_mps, zipup=True, compress=True)

            # plt.figure()
            # plt.plot(EC_k.get_data())
            # plt.title('filtered')
            # plt.show()

            # helper.mps_flip_lr(EC_k.data, inplace=True)
            new_EC = EC_k.take_qft(inverse=True,
                                   compress_opts=new_sys.E.compress_config.get_compress_opts(1))
            EC.data = new_EC.data

        for compID, BC in new_sys.B.components.items():
            BC_k = BC.take_qft()
            # BC_k = BC_k.elemental_multiply(filter_mps, zipup=True, compress=True)
            new_BC = BC_k.take_qft(inverse=True,
                                   compress_opts=new_sys.B.compress_config.get_compress_opts(1))
            BC.data = new_BC.data

        # print('filter max bonds', new_sys.E.max_bonds(), new_sys.B.max_bonds())

        return new_sys

    def realspace_smoother(self, inplace=False, E_comps=None, B_comps=None, phi=False, psi=False, spread: int = 1):
        """ perform filter in Fourier space (assuming currently in real space)
        """
        new_sys = self if inplace else self.copy()
        # grid = self.grid_X

        for compID, EC in new_sys.E.components.items():
            if E_comps is None or compID in E_comps:
                new_EC = EC.average_fine_scale(spread=spread,
                                               compress_opts=new_sys.E.compress_config.get_compress_opts(1))
                EC.data = new_EC.data

        for compID, BC in new_sys.B.components.items():
            if B_comps is None or compID in B_comps:
                new_BC = BC.average_fine_scale(spread=spread,
                                               compress_opts=new_sys.B.compress_config.get_compress_opts(1))
                BC.data = new_BC.data

        if phi and new_sys.phi is not None:
            new_phi = new_sys.phi.component.average_fine_scale(spread=spread,
                                                               compress_opts=new_sys.phi.compress_config.get_compress_opts(
                                                                   1))
            new_sys.phi.data = new_phi.data

        if psi and new_sys.psi is not None:
            new_psi = new_sys.psi.component.average_fine_scale(spread=spread,
                                                               compress_opts=new_sys.phi.compress_config.get_compress_opts(
                                                                   1))
            new_sys.psi.data = new_psi.data

        # print('filter max bonds', new_sys.E.max_bonds(), new_sys.B.max_bonds())
        return new_sys

    def normalize(self, target_B=None, target_E=None):
        """ normalized density field
            if normalize, would probably keep constant energy?
        """
        if self.E is not None:
            if target_E is None:
                target_E = self.init_norm_E if self.init_norm_E is not None else 1.0
            norm_E = self.E.norm()
            self.E.scalar_multiply(target_E / np.conj(norm_E), inplace=True)

        if self.B is not None:
            if target_B is None:
                target_B = self.init_norm_B if self.init_norm_B is not None else 1.0
            norm_B = self.B.norm()  # integrate(0)
            self.B.scalar_multiply(target_B / np.conj(norm_B), inplace=True)
        return

    def compute_poynting_vector(self):
        """ compute S = E x B*
        """
        B_conj = self.B.copy()
        for k, B in B_conj.components.items():
            B.conj(inplace=True)
        B_conj.scalar_multiply(1. / self.matl_params.mu0, inplace=True)
        return self.E.cross_product(B_conj, self.grid_X.axes[0].coord_sys)

    def electric_energy_density(self, compress_level=1):
        E_conj = self.E.copy()
        for k, E in E_conj.components.items():
            if E is not None and E.data is not None:
                E.conj(inplace=True)
        nrg = self.E.dot(E_conj, compress_level=compress_level, inner_compress_level=compress_level, zipup=True)
        nrg.scalar_multiply(self.matl_params.eps0 / 2, inplace=True)
        return nrg

    def magnetic_energy_density(self, compress_level=1):
        B_conj = self.B.copy()
        for k, B in B_conj.components.items():
            if B is not None and B.data is not None:
                B.conj(inplace=True)
        nrg = self.B.dot(B_conj, compress_level=compress_level, inner_compress_level=compress_level, zipup=True)
        nrg.scalar_multiply(1. / 2 / self.matl_params.mu0, inplace=True)
        return nrg

    def get_time_derivative_operator(self, compress=1, is_ion=False):
        """ dF/dt = G[f(t)].  Returns G
        """
        raise NotImplementedError

    def next_time_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, inplace=False, compress_level: int = 1,
                       max_iter=100, err_tol=None, apply_constraints=False,
                       is_first_time_step=False, is_last_time_step=False,
                       do_postprocessing=False, process_kwargs=None, verbose_plot=False, ):
        """ convenience fct btwn different TE methods?
        """
        time = self.time
        if self.te_order == 21:
            state_t = self.backwards_euler(dt, inplace=inplace, compress_level=compress_level)
        elif self.te_order == 1:
            state_t = self.euler(dt, deriv0, inplace=inplace, compress_level=compress_level)
        elif self.te_order == 2:
            state_t = self.rk2(dt, deriv0, compress_level=compress_level, )
        elif self.te_order == 3:
            state_t = self.rk3(dt, deriv0, compress_level=compress_level, )
        elif self.te_order == 4:
            state_t = self.rk4(dt, deriv0, compress_level=compress_level, verbose_plot=verbose_plot, )
        elif str(self.te_order)[:2] == '22':
            if str(self.te_order)[-1] == '1':
                state_t = self.crank_nicolson(dt, use_A2=False, inplace=inplace, compress_level=compress_level)
            elif str(self.te_order)[-1] == '3':
                state_t = self.crank_nicolson_block(dt, use_A2=False, inplace=inplace, compress_level=compress_level)
            elif str(self.te_order)[-1] == '4':
                state_t = self.crank_nicolson_block(dt, use_A2=True, inplace=inplace, compress_level=compress_level)
            elif str(self.te_order)[-1] == '6':
                state_t = self.crank_nicolson_block(dt, use_A2=False, inplace=inplace, weight=0.6,
                                                    compress_level=compress_level)
            elif str(self.te_order)[-1] == '7':  ## backward euler
                state_t = self.crank_nicolson_block(dt, use_A2=False, inplace=inplace, weight=1.0,
                                                    compress_level=compress_level)
            else:  # if str(self.te_order)[-1] == '2': ## default
                state_t = self.crank_nicolson(dt, use_A2=True, inplace=inplace, compress_level=compress_level)
        elif str(self.te_order)[:2] == '33':
            wk = None
            if self.te_order < 100:
                weight = 0.5
            else:
                wk = str(self.te_order)[2:]
                weight = np.round(int(wk) * 10 ** (-len(wk)), len(wk))
                if wk == '0':
                    weight = 0.5
            print('33 weight', weight)

            if is_first_time_step:
                state_t = self.crank_nicolson_block(dt, use_A2=False, inplace=inplace, compress_level=compress_level)
            else:
                te_order = 0 if wk == '0' else self.te_order
                try:
                    state_t = self.block_tddmrg(dt, te_order=te_order, use_A2=False, inplace=inplace,
                                                compress_level=compress_level, apply_constraints=apply_constraints)
                    # state_t = self.tddmrg(dt, te_order=4, use_A2=False, inplace=inplace, compress_level=compress_level)
                except RuntimeError:  # effective A's all zero
                    state_t = self.crank_nicolson_block(dt, use_A2=False, inplace=inplace,
                                                        compress_level=compress_level)
        elif self.te_order == 31:
            state_t = self.split_step(dt, deriv0=deriv0, inplace=inplace, is_first_time_step=is_first_time_step,
                                      is_last_time_step=is_last_time_step, verbose_plot=verbose_plot)
        else:
            state_t = super().next_time_step(dt, deriv0=deriv0, inplace=inplace, compress_level=compress_level,
                                             max_iter=max_iter, err_tol=err_tol, is_first_time_step=is_first_time_step,
                                             is_last_time_step=is_last_time_step, do_postprocessing=do_postprocessing,
                                             process_kwargs=process_kwargs, verbose_plot=verbose_plot)

        self.time = time + dt

        return state_t

    def evolve_E(self, dt: Numeric, deriv0: Optional['Maxwell'] = None, method=None, current_only=False,
                 inplace=False, compress=1, compress1=4, compress2=5, verbose_plot: bool = False) -> 'Maxwell':
        """ evolve E field in time via euler step
        """
        state0 = self if inplace else self.copy()
        old_E = self.E.copy()
        print('evolve E', dt)

        if deriv0 is None:
            dEdt = state0._calculate_time_derivative_E(compress=compress1, compress1=compress2, compress2=0,
                                                       current_only=current_only,
                                                       verbose_plot=verbose_plot)

            # plt.figure()
            # for C in dEdt.componentIDs:
            #     dEdt_data = dEdt.get_comp_data(C)
            #     if dEdt_data is not None:
            #         plt.plot(np.real(dEdt_data), label=f'dEdt {C}')
            #         plt.plot(np.imag(dEdt_data), '--', label=f'dEdt {C}')
            # plt.legend()
            # plt.show()

            deriv0 = state0.create_like(dEdt, None, None, None)
        else:
            ### keep only dE/dt in deriv0
            deriv0 = state0.create_like(deriv0.E, None, None, None)

        if method is None or method in ['euler', 'rk1']:
            state0.euler(dt, deriv0=deriv0, inplace=True, compress_level=compress)
        elif method == 'rk4':
            print('evolve E rk4')
            state0.rk4(dt, deriv0=deriv0, inplace=True, compress_level=compress, do_evolve_B=False)
        else:
            raise NotImplementedError

        if self.verbose_plot:
            for C in self.coords_x.coords:
                E_data = state0.E.get_comp_data(C)
                E_data_old = old_E.get_comp_data(C)
                if E_data is not None and E_data_old is not None:
                    plt.figure()
                    plt.plot(np.real(E_data), ':')
                    plt.plot(np.imag(E_data), ':')
                    plt.plot(np.real(E_data_old), '-.')
                    plt.plot(np.imag(E_data_old), '-.')
                    plt.plot(np.real(E_data - E_data_old))
                    plt.plot(np.imag(E_data - E_data_old))
                    plt.title(f'E(t+1)-E(t) {C}')
                    plt.show()

        return state0

    def evolve_B(self, dt: Numeric, deriv0: Optional['Maxwell'] = None, method=None,
                 inplace=False, compress=1, compress1=4, compress2=5, verbose_plot: bool = False) -> 'Maxwell':
        """ evolve E field in time via euler step
        """
        state0 = self if inplace else self.copy()
        old_B = self.B.copy()
        print('evolve B', dt)

        print('deriv0?', deriv0)
        if deriv0 is None:
            dBdt = state0._calculate_time_derivative_B(compress=compress1, compress1=compress2, compress2=0,
                                                       verbose_plot=verbose_plot)
            deriv0 = state0.create_like(None, dBdt, None, None)
        else:
            ### keep only dB/dt in deriv0
            deriv0 = state0.create_like(None, deriv0.B, None, None)

        if method is None or method in ['euler', 'rk1']:
            state0.euler(dt, deriv0=deriv0, inplace=True, compress_level=compress)

        # elif method[:4] == 'tdvp':
        #     ### this doesn't work at all. fundamentally, i think
        #     te_order = 0 #  if len(method) == 4 else int(method[-1])
        #     if deriv0 is None:
        #         dBdt = state0._calculate_time_derivative_B(compress=compress1, compress1=compress2, compress2=0,
        #                                                    verbose_plot=verbose_plot)
        #     else:
        #         dBdt = deriv0.B
        #
        #     B_field = state0.B
        #     for C in dBdt.componentIDs:
        #         dBdt_comp = dBdt[C]
        #         B_comp = B_field.components.get(C, None)
        #         if dBdt_comp is not None and dBdt_comp.data is not None:
        #             if B_comp is None or B_comp.data is None:
        #                 B_comp = dBdt_comp.scalar_multiply(dt, inplace=False)
        #             else:
        #                 B_comp_copy = B_comp.copy()
        #                 print('dBdt comp', dBdt_comp.norm(is_sqrt=True))
        #                 B_comp.evolve_tdvp0(dt, [dBdt_comp], te_order=te_order, inplace=True,
        #                                     compress_config=B_field.compress_config)
        #                 print('B diff', B_comp.distance(B_comp_copy))
        #             B_field[C].data = B_comp.data

        elif method == 'rk4':
            print('evolve B rk4')
            # print('B ax derivs', [(cID, c.ax_deriv_configs) for cID, c in state0.B.components.items()])
            new_state = state0.rk4(dt, deriv0=deriv0, compress_level=compress,
                                   do_evolve_E=False)
            state0.E = new_state.E
            state0.B = new_state.B
            state0.phi = new_state.phi
            state0.psi = new_state.psi
        else:
            raise NotImplementedError

        if self.verbose_plot:
            for C in self.coords_x.coords:
                B_data = state0.B.get_comp_data(C)
                B_data_old = old_B.get_comp_data(C)
                if B_data is not None:
                    plt.figure()
                    plt.plot(np.real(B_data), ':')
                    plt.plot(np.imag(B_data), ':')
                    plt.plot(np.real(B_data_old), '-.')
                    plt.plot(np.imag(B_data_old), '-.')
                    plt.plot(np.real(B_data - B_data_old))
                    plt.plot(np.imag(B_data - B_data_old))
                    plt.title(f'ss B(t+1)-B(t) {C}')
                    plt.show()

        return state0

    def shift_E_grid(self, inverse=False):
        shifted_E = {}

        ## shift E ##
        for compID in self.coords_x.coords:
            comp = self.E.components.get(compID, None)
            if comp is not None:

                ax = self.coords_x.get_axis(compID)
                if ax is None:
                    shifted_E[compID] = comp.copy()
                else:
                    shifted_E[compID] = comp.shift_cell_to_midpoint([ax], inverse=inverse)

        return self.E.create_like(shifted_E)

    def shift_B_grid(self, inverse=False):
        shifted_B = {}

        for compID in self.coords_x.coords:
            comp = self.B.components.get(compID, None)
            if comp is not None:

                # other_compIDs = [cID for cID in self.coords_x.coords if cID != compID]
                # print('other compIDs', other_compIDs)

                shift_axes = []
                for cID in self.coords_x.coords:
                    if cID != compID:
                        ax = self.coords_x.get_axis(cID)
                        if ax is None:
                            continue
                        shift_axes += [ax]

                shifted_B[compID] = comp.shift_cell_to_midpoint(shift_axes, inverse=inverse)

        return self.B.create_like(shifted_B)

    def split_step(self, dt: Numeric, deriv0: Optional['PDE_system'] = None, method_E=None, method_B=None,
                   is_first_time_step=False, is_last_time_step=False, inplace=False,
                   compress_level=1, verbose_plot: bool = False) -> 'Maxwell':
        """ take split step for EM system and ion,electron advection terms
            each step is just an Euler update -> FDTD-like
        """
        ## at initialization, need to evolve EM_sys with dt/2
        ## maybe include time as a part of self?
        state0 = self if inplace else self.copy()
        print('split step', 'method E', method_E, 'method B', method_B, self.is_yee)

        # print('is first time step?', is_first_time_step)

        if compress_level == 0:
            comp1 = comp2 = comp3 = comp4 = comp5 = 0
        else:
            comp1, comp2, comp3, comp4, comp5 = self._get_compress_levels(compress_level, 5)

        ## set derivatives if FDTD
        if self.is_yee:
            for compID, comp in self.E.components.items():
                for ax in self.grid_X.axes:
                    deriv_config = comp.ax_deriv_configs[ax].copy()
                    deriv_config.update(order=0, fd_type=FDType.FORWARD)
                    comp.ax_deriv_configs[ax] = deriv_config

            for compID, comp in self.B.components.items():
                for ax in self.grid_X.axes:
                    if comp is not None:
                        deriv_config = comp.ax_deriv_configs[ax].copy()
                        deriv_config.update(order=0, fd_type=FDType.BACKWARD)
                        comp.ax_deriv_configs[ax] = deriv_config

        # print('ax deriv configs')
        # for compID, comp in self.E.components.items():
        #     print('E compID', compID, comp.ax_deriv_configs)
        # for compID, comp in self.B.components.items():
        #     print('B compID', compID, comp.ax_deriv_configs)

        if is_first_time_step:
            ### Bx -> (i, j+1/2, k+1/2)
            ### By -> (i+1/2, j, k+1/2)
            ### Bz -> (i+1/2, j+1/2, k)
            # i think, ideally, Ex, Ey, Ez and Bx, By, Bz would all be on the same collocated grid
            # and then for FDTD, the fields would be shifted to the Yee cell grid
            # however, this shifting needs to be done in a way that preserves div(B) = 0
            # currently, it does not, so the grid is not shifted afterwards.

            shift_grid = False
            if shift_grid:
                E_orig = state0.E
                B_orig = state0.B

                unshifted_E = state0.shift_E_grid(inverse=True)  ## to integer grid
                unshifted_B = state0.shift_B_grid(inverse=True)  ## to integer grid

                # print('shifted E', unshifted_E.components)
                state0.E = unshifted_E
                state0.B = unshifted_B

                # for compID, comp in unshifted_E.components.items():
                #     print('unshifted E deriv', compID, comp.ax_deriv_configs)
                #     plt.figure()
                #     plt.imshow(comp.get_data())
                #     plt.title(f'unshifted E{compID}')
                #     plt.show()
                #
                # for compID, comp in unshifted_B.components.items():
                #     print('unshifted B deriv', compID, comp.ax_deriv_configs)

                # print('computing B')

                state0.is_yee = False
                state_dt2: 'Maxwell' = state0.rk4(dt / 2, inplace=False)
                state0.is_yee = True
                state0.E = E_orig

                shifted_B = state_dt2.shift_B_grid(inverse=False)
                state0.B = shifted_B

                # for compID, comp in state0.B.components.items():
                #     print('shifted B deriv', compID, comp.ax_deriv_configs)
                #     if comp is not None and comp.data is not None:
                #         plt.figure()
                #         plt.imshow(np.real(comp.get_data()))
                #         plt.title(f'shifted B{compID}')
                # plt.show()

            else:
                state_dt2: 'Maxwell' = state0.rk4(dt / 2, inplace=False)
                state0.B = state_dt2.B

            if self.clean:
                state0.clean_fields(compress=None)
                print('divB err', state0.check_divB().norm())

            # divB_err = state0.check_divB().get_comp_data()
            # plt.figure()
            # plt.imshow(divB_err)
            # plt.title('init divB')
            # plt.colorbar()
            # plt.show()

        else:
            print('evolve B')
            state0 = state0.evolve_B(dt, inplace=True, method=method_B,
                                     compress=comp1, compress1=comp4, compress2=comp5)

            # divB_err = state0.check_divB().get_comp_data()
            # plt.figure()
            # plt.imshow(divB_err)
            # plt.title('divB error')
            # plt.colorbar()
            # plt.show()
            # # exit()

        print('evolve E')
        state0 = state0.evolve_E(dt, inplace=True, method=method_E,
                                 compress=comp1, compress1=comp4, compress2=comp5)

        # dt_ = dt / 2 if is_last_time_step else dt
        # method_B = 'rk4' if is_last_time_step else method_B
        if is_last_time_step:
            state0 = state0.evolve_B(dt / 2, inplace=True, method=method_B,
                                     compress=comp1, compress1=comp4, compress2=comp5)

        state0.time = self.time + dt if self.time is not None else None

        return state0

    # @profile
    def calculate_time_derivative(self, time=None, compress_level=0, compress_level1=0, compress_level2=0,
                                  do_x_advection=True, do_v_advection=True, background_force=True, internal_force=True,
                                  do_evolve_B=True, do_evolve_E=True,
                                  verbose_plot=False, **kwargs) -> 'Maxwell':
        """ df/dt = ...
            dB/dt + curl(E) = 0
            e0*mu0 dE/dt - curl(B) = -mu0 J
            J = sum_s qs ns vs

            note:  div(E)=rho/eps0, div(B)=0 must be satisfied with initial definitions of E, B
        """
        # print('do_evovlve E, B', do_evolve_E, do_evolve_B)
        if do_evolve_B:
            print('compute dB/dt')
            dFdt_B = self._calculate_time_derivative_B(compress=compress_level, compress1=compress_level1,
                                                       verbose_plot=verbose_plot)
        else:
            dFdt_B = None

        if do_evolve_E:
            print('compute dE/dt')
            dFdt_E = self._calculate_time_derivative_E(  # current_density=self.current_density,
                compress=compress_level, compress1=compress_level1,
                verbose_plot=verbose_plot)
        else:
            dFdt_E = None

        # print('EM time deriv')
        # print('dBdt', dFdt_B.components)
        # print('dEdt', dFdt_E.components)

        if self.clean:
            dFdt_psi = self._calculate_time_derivative_psi(compress=compress_level, compress1=compress_level1)
            dFdt_phi = self._calculate_time_derivative_phi(  # charge_density=self.charge_density,
                compress=compress_level, compress1=compress_level1)
        else:
            dFdt_phi = None
            dFdt_psi = None

        dFdt = self.create_like(dFdt_E, dFdt_B, dFdt_phi, dFdt_psi)
        return dFdt

    def _calculate_upwind_current(self, current_density=None, for_E=True, for_B=True, compress=1, compress1=0):
        """ average stencil:  0.25 J_{i-1} + 0.5 J_i + 0.25 J_{i+1}
            if the dimensions exist, compute the average along the two transverse directions
        """

        # x_axes = self.coords_x.axes
        # verbose_plot = True
        # upwind_scale = 0.1

        ## current
        j: Optional[Field] = current_density if current_density is not None else self.current_density
        j = j.copy()

        j_corrE, j_corrB = None, None
        compress_opts = j.compress_config.get_compress_opts(compress)

        if j is not None:
            if self.matl_params.is_cgs:
                j.scalar_multiply(-4 * np.pi, inplace=True)
            else:
                j.scalar_multiply(-1 / self.matl_params.eps0, inplace=True)

            # j = j.scalar_multiply(0.5, inplace=True)

            # j.is_sqrt = True
            # print('j norms', j.norms())

            ax_x = self.coords_x.get_axis(CoordinateType.X)
            ax_y = self.coords_x.get_axis(CoordinateType.Y)
            ax_z = self.coords_x.get_axis(CoordinateType.Z)

            X = self.coords_x.get_coord(CoordinateType.X)
            Y = self.coords_x.get_coord(CoordinateType.Y)
            Z = self.coords_x.get_coord(CoordinateType.Z)

            if for_E:
                print('J upwind for E')

                # j_corrE = j.copy()
                mod_j = {}
                for C in self.coords_x.coords:
                    if C.type == CoordinateType.X:
                        ax1, ax2 = ax_y, ax_z
                    elif C.type == CoordinateType.Y:
                        ax1, ax2 = ax_z, ax_x
                    elif C.type == CoordinateType.Z:
                        ax1, ax2 = ax_x, ax_y
                    else:
                        raise NotImplementedError

                    if C not in j.components or j[C] is None or j[C].data is None:
                        continue

                    if ax1 is not None and ax2 is not None:
                        jc1 = j[C].average_fine_scale([ax1], mu=0.5)
                        jc2 = j[C].average_fine_scale([ax2], mu=0.5)
                        jc = jc1.add(jc2, compress=True, compress_opts=compress_opts)
                        jc.scalar_multiply(0.5, inplace=True)
                    elif ax1 is not None and ax2 is None:
                        jc = j[C].average_fine_scale([ax1], mu=0.5, compress_opts=compress_opts)
                    elif ax1 is None and ax2 is not None:
                        jc = j[C].average_fine_scale([ax2], mu=0.5, compress_opts=compress_opts)
                    else:
                        jc = None

                    if jc is not None:
                        mod_j[C] = jc

                j_corrE = j.create_like(mod_j)
                # j_corrE.is_sqrt = True
                # print('new j corr E', j_corrE.norms())

                # plt.figure()
                # plt.imshow(np.real(j.get_comp_data(Z)))
                # plt.colorbar()
                # plt.title('JZ')
                #
                # plt.figure()
                # plt.imshow(np.real(j_corrE.get_comp_data(Z)))
                # plt.colorbar()
                # plt.title('JZ corr')
                # plt.show()

            if for_B:
                print('J upwind for B')
                c = self.matl_params.c
                mod_j = {}
                for C in self.coords_x.coords:
                    ## C: component of output (Bx, By, Bz)

                    ## components of J and corresponding derivative direction
                    if C.type == CoordinateType.X:
                        ax1, ax2 = ax_y, ax_z
                        C1, C2 = Z, Y
                    elif C.type == CoordinateType.Y:
                        ax1, ax2 = ax_z, ax_x
                        C1, C2 = X, Z
                    elif C.type == CoordinateType.Z:
                        ax1, ax2 = ax_x, ax_y
                        C1, C2 = Y, X
                    else:
                        raise NotImplementedError

                    jc = None
                    if ax1 is not None and (C1 in j.components and j[C1] is not None and j[C1].data is not None):
                        deriv_config = j[C1].ax_deriv_configs[ax1].copy()
                        deriv_config.update(order=1, fd_type=FDType.CENTER)
                        jc1 = j[C1].take_firstderivative(ax1, inplace=False, compress=compress1,
                                                         ax_deriv_config={ax1: deriv_config})
                        jc1 = jc1.scalar_multiply(-ax1.dx / c, inplace=True)  ## multiply by -1 later
                        if self.grid_X.ndim == 3 or (self.grid_X.ndim == 2 and C1 == Z):
                            print('curl J divide by 2')
                            jc1 = jc1.scalar_multiply(0.5, inplace=True)
                        jc = jc1

                    if ax2 is not None and (C2 in j.components and j[C2] is not None and j[C2].data is not None):
                        deriv_config = j[C2].ax_deriv_configs[ax2].copy()
                        deriv_config.update(order=1, fd_type=FDType.CENTER)
                        jc2 = j[C2].take_firstderivative(ax2, inplace=False, compress=compress1,
                                                         ax_deriv_config={ax2: deriv_config})
                        jc2 = jc2.scalar_multiply(+ax2.dx / c, inplace=True)  ## multiply by -1 later

                        if self.grid_X.ndim == 3 or (self.grid_X.ndim == 2 and C2 == Z):
                            print('curl J divide by 2')
                            jc2 = jc2.scalar_multiply(0.5, inplace=True)

                        if jc is not None:
                            jc = jc2.add(jc, compress_opts=compress_opts)
                        else:
                            jc = jc2

                    if jc is not None:
                        # mod_j[C] = jc.scalar_multiply(upwind_scale)  # .scalar_multiply(-1.0)
                        mod_j[C] = jc.scalar_multiply(-1.0)

                j_corrB = j.create_like(mod_j)
                # j_corrB.scalar_multiply(0.5, inplace=True)
                # j_corrB.is_sqrt = True
                # print('j corr B', j_corrB.norms())

        if for_E and for_B:
            return j_corrE, j_corrB
        elif for_E:
            return j_corrE
        elif for_B:
            return j_corrB
        return

    # @profile
    def _calculate_time_derivative_E(self, current_density=None, compress=1, compress1=0, compress2=0,
                                     current_only=False, verbose_plot=False, dt=None):
        """
        e0*mu0 dE/dt - curl(B) + chi*grad(phi) = -mu0 J
        J = sum_s qs ns vs

        FDTD:
        source current is located at the same spatial grid location as the electric field
        but at the same time point as the magnetic field.
        """
        verbose_plot = self.verbose_plot
        x_axes = self.coords_x.axes
        # verbose_plot = True

        # print('calc dE/dt: B deriv configs')
        # for compID, comp in self.B.components.items():
        #     print(comp.ax_deriv_configs)

        ## ensure that the correct derivatives are used for finite volume calculation
        if self.upwind:
            if self.matl_params.is_cgs:
                raise NotImplementedError

            for C in self.coords_x.coords:
                E_ax_deriv_config = self.E[C].ax_deriv_configs
                for ax in x_axes:
                    E_ax_deriv_config[ax].update(order=1, fd_type=FDType.CENTER)

            for C in self.coords_x.coords:
                B_ax_deriv_config = self.B[C].ax_deriv_configs
                for ax in x_axes:
                    B_ax_deriv_config[ax].update(order=1, fd_type=FDType.CENTER)

        ## current
        j: Optional[Field] = current_density if current_density is not None else self.current_density
        if j is not None:

            # for C in self.coords_x.coords:
            #     jx = j.get_comp_data(C)
            #     if jx is not None:
            #         plt.figure()
            #         plt.plot(np.real(jx))
            #         plt.plot(np.imag(jx),'--')
            #         plt.title(f'j{C}')
            # plt.show()

            if self.upwind:
                j = self._calculate_upwind_current(current_density, for_B=False, compress=compress, compress1=compress1)
            else:

                if self.is_electrostatic and len(x_axes) == 1:
                    X = x_axes[0]
                    # j_ghost_mps = helper.sum_tensornetwork(like_tn=j[X.coordinate].data)
                    # j_ghost_mps.view_like(j[X.coordinate].data, inplace=True)
                    jg_const = j.norm() * -1 / (X.npts * X.dx)  # norm integrates over all axes -> int
                    # helper.scalar_multiply(j_ghost_mps,jg_const,inplace=True)
                    # j_ghost = j.create_like({x_axes[0].coordinate:j_ghost_mps})
                    # print('j',j[X.coordinate].data, j_ghost[X.coordinate].data)
                    # j.add(j_ghost,inplace=True)
                    j.scalar_add(jg_const, inplace=True)

                    # plt.figure()
                    # plt.plot(j.get_field_data(comps=0),label='total')
                    # plt.plot(j_ghost.get_field_data(comps=0),label='ghost')
                    # plt.legend()

                # plt.show()
                if self.matl_params.is_cgs:
                    j = j.scalar_multiply(-4 * np.pi, inplace=False)
                else:
                    j = j.scalar_multiply(-1 / self.matl_params.eps0, inplace=False)

        dEdt = j
        # print('dEdt', dEdt[x_axes[0].coordinate].ax_deriv_configs[x_axes[0]].bc)

        if verbose_plot:
            if j is not None:
                plt.figure()
                for C in self.coords_x.coords:
                    j_data = j.get_comp_data(C)
                    if j_data is not None:
                        plt.plot(np.real(j_data), label=f'{C} re')
                        plt.plot(np.imag(j_data), '--', label=f'{C} im')
                plt.legend()
                plt.xlabel('x')
                plt.title('-j')
                # plt.show()
            else:
                print('j is None')

        # if verbose_plot:
        #     X, Y, Z = self.coords_x.coords
        #     plt.figure()
        #     for C in self.coords_x.coords:
        #         B_data = self.B.get_comp_data(C)
        #         if B_data is not None:
        #             plt.plot(np.real(B_data), label=f'{C} re')
        #             plt.plot(np.imag(B_data), '--', label=f'{C} im')
        #     plt.xlabel('x')
        #     plt.legend()
        #     plt.title('B')
        #     plt.show()

        ## curl(B)
        print('current only?', current_only)
        if not current_only:
            curlB = None
            if self.B is not None:

                # if self.is_yee:
                #     for compID, comp in self.B.components.items():
                #         for ax in self.grid_X.axes:
                #             if comp is not None:
                #                 deriv_config = comp.ax_deriv_configs[ax].copy()
                #                 deriv_config.update(order=0, fd_type=FDType.BACKWARD)
                #                 comp.ax_deriv_configs[ax] = deriv_config

                # print('curl B')
                # for compID in self.B.componentIDs:
                #     try:
                #         print('self.B ax deriv confgs', self.B[compID].ax_deriv_configs)
                #     except AttributeError:
                #         print(f'no B{compID}')
                curlB = self.B.curl(self.coords_x, compress_level=compress1, inner_compress_level=compress2)

                if verbose_plot:
                    X, Y, Z = self.coords_x.coords
                    plt.figure()
                    for C in self.coords_x.coords:
                        B_data = self.B.get_comp_data(C)
                        if B_data is not None:
                            plt.plot(np.real(B_data), label=f'{C} re')
                            plt.plot(np.imag(B_data), '--', label=f'{C} im')
                    plt.xlabel('x')
                    plt.legend()
                    plt.title('B')
                    # plt.show()

                    plt.figure()
                    for C in self.coords_x.coords:
                        B_data = curlB.get_comp_data(C)
                        if B_data is not None:
                            plt.plot(np.real(B_data), label=f'{C} re')
                            plt.plot(np.imag(B_data), '--', label=f'{C} im')
                    plt.xlabel('x')
                    plt.legend()
                    plt.title('curl B')
                    # plt.show()

            if self.background_B0 is not None:
                if self.curlB0 is None:
                    print('calc curl B0')
                    curlB0 = self.background_B0.curl(self.coords_x, compress_level=compress1,
                                                     inner_compress_level=compress2)
                    self.curlB0 = curlB0

                if curlB is not None:
                    curlB.add(self.curlB0, inplace=True)
                else:
                    curlB = self.curlB0.copy()

            ## scale curl B
            if curlB is not None:
                if self.matl_params.is_cgs:
                    curlB = curlB.scalar_multiply(self.matl_params.c, inplace=False)
                else:
                    curlB = curlB.scalar_multiply(self.matl_params.c ** 2, inplace=False)

            # for C in self.coords_x.coords:
            #     plt.figure()
            #     j_data = j.get_comp_data(C)
            #     B_data = curlB.get_comp_data(C)
            #     if j_data is not None:
            #         plt.plot(-np.real(j_data), label=f'-j{C}')
            #         plt.plot(-np.imag(j_data), '--', label=f'-j{C}')
            #     if B_data is not None:
            #         plt.plot(np.real(B_data), label=f'curl(B), {C}')
            #         plt.plot(np.imag(B_data), '--', label=f'curl(B), {C}')
            #     plt.legend()
            #     plt.title('dE/dt components')
            # plt.show()

            ## dE/dt = -1/eps0 J + c^2 curl B
            if dEdt is not None:
                dEdt = dEdt.add(curlB, inplace=False, compress_level=0)
            else:
                dEdt = curlB.copy()

        # ### add smoothing
        # for compID, comp_gtn in dEdt.components.items():
        #     comp_gtn.average_fine_scale(mu=0.50, inplace=True)
        #     print('smoothing dEdt')

        ## upwind correction
        if self.upwind:
            print('do upwind dE/dt')
            X, Y, Z = self.coords_x.coords
            ax_x = self.coords_x.coord_axes.get(X, None)
            ax_y = self.coords_x.coord_axes.get(Y, None)
            ax_z = self.coords_x.coord_axes.get(Z, None)

            Ex_corr = None
            if ax_y is not None and X in self.E.components:
                Ex_corr = self.E[X].take_secondderivative(ax_y, ax_y)
                Ex_corr = Ex_corr.scalar_multiply(ax_y.dx, inplace=False)
            if ax_z is not None and X in self.E.components:
                part2 = self.E[X].take_secondderivative(ax_z, ax_z)
                part2 = part2.scalar_multiply(ax_z.dx, inplace=False)
                Ex_corr = part2 if Ex_corr is None else Ex_corr.add(part2, compress=True)

            Ey_corr = None
            if ax_x is not None and Y in self.E.components:
                Ey_corr = self.E[Y].take_secondderivative(ax_x, ax_x)
                Ey_corr = Ey_corr.scalar_multiply(ax_x.dx, inplace=False)
            if ax_z is not None and Y in self.E.components:
                part2 = self.E[Y].take_secondderivative(ax_z, ax_z)
                part2 = part2.scalar_multiply(ax_z.dx, inplace=False)
                Ey_corr = part2 if Ey_corr is None else Ey_corr.add(part2, compress=True)

            Ez_corr = None
            if ax_x is not None and Z in self.E.components:
                Ez_corr = self.E[Z].take_secondderivative(ax_x, ax_x)
                Ez_corr = Ez_corr.scalar_multiply(ax_x.dx, inplace=False)
            if ax_y is not None and Z in self.E.components:
                part2 = self.E[Z].take_secondderivative(ax_y, ax_y)
                part2 = part2.scalar_multiply(ax_y.dx, inplace=False)
                Ez_corr = part2 if Ez_corr is None else Ez_corr.add(part2, compress=True)

            # print('E corrs', Ex_corr, Ey_corr, Ez_corr)
            dEdt_corr = self.B.create_like({X: Ex_corr, Y: Ey_corr, Z: Ez_corr})
            dEdt_corr = dEdt_corr.scalar_multiply(self.matl_params.c / 2, inplace=False)
            dEdt = dEdt.add(dEdt_corr)

        ## correction
        if self.phi is not None and self.phi.component is not None and self.phi.component.data is not None:
            grad_phi = self.phi.gradient(compress_level=compress1)
            grad_phi.scalar_multiply(-1 * self.chi, inplace=True)
            dEdt.add(grad_phi, inplace=True, compress_level=0)

        if dEdt is not None:
            if compress:    dEdt.compress(inplace=True, compress_level=compress)
            # if not self.matl_params.is_cgs:
            #     dEdt.scalar_multiply(self.matl_params.c ** 2, inplace=True)
            dEdt.name = self.E.name if self.E is not None else 'E'

            if verbose_plot:
                plt.figure()
                for C in self.coords_x.coords:
                    E_data = self.E.get_comp_data(C)
                    if E_data is not None:
                        plt.plot(np.real(E_data), label=f'{C} re')
                        plt.plot(np.imag(E_data), '--', label=f'{C} im')
                plt.xlabel('x')
                plt.legend()
                plt.title('E(t)')

                plt.figure()
                for C in self.coords_x.coords:
                    dEdt_data = dEdt.get_comp_data(C)  # / self.matl_params.c ** 2
                    if dEdt_data is not None:
                        plt.plot(np.real(dEdt_data), label=f'{C} re')
                        plt.plot(np.imag(dEdt_data), '--', label=f'{C} im')
                plt.xlabel('x')
                plt.legend()
                plt.title('dE/dt')
                plt.show()

        # print('dEdt', dEdt.components)

        return dEdt

    # @profile
    def _calculate_time_derivative_B(self, compress=1, compress1=0, compress2=0, verbose_plot=False,
                                     current_density=None):
        """
        dB/dt + curl(E) + gamma*grad(psi) = 0
        """
        dBdt = None
        verbose_plot = self.verbose_plot

        # print('calc dB/dt: E deriv configs')
        # for compID, comp in self.E.components.items():
        #     print(comp.ax_deriv_configs)

        ## ensure that the correct derivatives are used for finite volume calculation
        if self.upwind:
            if self.matl_params.is_cgs:
                raise NotImplementedError

            for C in self.coords_x.coords:
                E_ax_deriv_config = self.E[C].ax_deriv_configs
                for ax in self.coords_x.axes:
                    E_ax_deriv_config[ax].update(order=1, fd_type=FDType.CENTER)

            for C in self.coords_x.coords:
                B_ax_deriv_config = self.B[C].ax_deriv_configs
                for ax in self.coords_x.axes:
                    B_ax_deriv_config[ax].update(order=1, fd_type=FDType.CENTER)

        ## curl(E)
        if self.E is not None:

            # X, Y, Z = self.coords_x.coords
            #
            # Ey_data = self.E.get_comp_data(Y)
            # if Ey_data is not None:
            #     plt.figure()
            #     plt.plot(np.real(Ey_data))
            #     plt.plot(np.imag(Ey_data), '--')
            #     plt.title('Ey init')
            #
            # Ez_data = self.E.get_comp_data(Z)
            # if Ez_data is not None:
            #     plt.figure()
            #     plt.plot(np.real(Ez_data))
            #     plt.plot(np.imag(Ez_data), '--')
            #     plt.title('Ez init')
            #
            # plt.show()

            # if self.is_yee:
            #     for compID, comp in self.E.components.items():
            #         for ax in self.grid_X.axes:
            #             deriv_config = comp.ax_deriv_configs[ax].copy()
            #             deriv_config.update(order=0, fd_type=FDType.FORWARD)
            #             comp.ax_deriv_configs[ax] = deriv_config

            # print('curl E')
            # for compID in self.E.componentIDs:
            #     try:
            #         print('self.E ax deriv confgs', self.E[compID].ax_deriv_configs)
            #     except AttributeError:
            #         print(f'no E{compID}')
            curlE = self.E.curl(self.coords_x, compress_level=compress1, inner_compress_level=compress2)
            dBdt = curlE

            # if len(curlE._components) > 0:
            #     X, Y, Z = self.coords_x.coords
            #     ax_x = curlE.grid.axes[0]
            #     curlE_data = curlE.get_comp_data(Y)
            #     ax_x.basis.get_realspace_1D(curlE_data, 0)
            #     plt.figure()
            #     plt.plot(np.real(curlE_data))
            #     plt.plot(np.imag(curlE_data), '--')
            #     plt.title('curl E')
            #
            #     Ez_data = self.E.get_comp_data(Z)
            #     plt.figure()
            #     plt.plot(np.real(Ez_data))
            #     plt.plot(np.imag(Ez_data),'--')
            #     plt.title('self.Ez after')
            #
            #     Ey_data = self.E.get_comp_data(Y)
            #     plt.figure()
            #     plt.plot(np.real(Ey_data))
            #     plt.plot(np.imag(Ey_data),'--')
            #     plt.title('self.Ey after')
            #
            #     plt.show()

            if verbose_plot:
                X, Y, Z = self.coords_x.coords
                plt.figure()
                for C in self.coords_x.coords:
                    E_data = self.E.get_comp_data(C)
                    if E_data is not None:
                        plt.plot(np.real(E_data), label=f'{C} re')
                        plt.plot(np.imag(E_data), '--', label=f'{C} im')
                plt.xlabel('x')
                plt.legend()
                plt.title('E')
                # plt.show()

                plt.figure()
                for C in self.coords_x.coords:
                    E_data = curlE.get_comp_data(C)
                    if E_data is not None:
                        plt.plot(np.real(E_data), label=f'{C} re')
                        plt.plot(np.imag(E_data), '--', label=f'{C} im')
                plt.xlabel('x')
                plt.legend()
                plt.title('curl E')
                plt.show()

        if self.background_E0 is not None:
            if self.curlE0 is None:
                curlE0 = self.background_E0.curl(self.coords_x, compress_level=compress1,
                                                 inner_compress_level=compress2)
                self.curlE0 = curlE0

            if dBdt is not None:
                dBdt.add(self.curlE0, inplace=True, compress_level=0)
            else:
                dBdt = self.curlE0.copy()

        # ### add smoothing
        # for compID, comp_gtn in dBdt.components.items():
        #     comp_gtn.average_fine_scale(mu=0.50, inplace=True)
        #     print('smoothing dBdt')

        if self.upwind:
            print('do upwind dB/dt')
            X, Y, Z = self.coords_x.coords
            ax_x = self.coords_x.coord_axes.get(X, None)
            ax_y = self.coords_x.coord_axes.get(Y, None)
            ax_z = self.coords_x.coord_axes.get(Z, None)

            Bx_corr = None
            if ax_y is not None and X in self.B.components:
                Bx_corr = self.B[X].take_secondderivative(ax_y, ax_y)
                Bx_corr = Bx_corr.scalar_multiply(ax_y.dx, inplace=False)
            if ax_z is not None and X in self.B.components:
                part2 = self.B[X].take_secondderivative(ax_z, ax_z)
                part2 = part2.scalar_multiply(ax_z.dx, inplace=False)
                Bx_corr = part2 if Bx_corr is None else Bx_corr.add(part2, compress=True)

            By_corr = None
            if ax_x is not None and Y in self.B.components:
                By_corr = self.B[Y].take_secondderivative(ax_x, ax_x)
                By_corr = By_corr.scalar_multiply(ax_x.dx, inplace=False)
            if ax_z is not None and Y in self.B.components:
                part2 = self.B[Y].take_secondderivative(ax_z, ax_z)
                part2 = part2.scalar_multiply(ax_z.dx, inplace=False)
                By_corr = part2 if By_corr is None else By_corr.add(part2, compress=True)

            Bz_corr = None
            if ax_x is not None and Z in self.B.components:
                Bz_corr = self.B[Z].take_secondderivative(ax_x, ax_x)
                Bz_corr = Bz_corr.scalar_multiply(ax_x.dx, inplace=False)
            if ax_y is not None and Z in self.B.components:
                part2 = self.B[Z].take_secondderivative(ax_y, ax_y)
                part2 = part2.scalar_multiply(ax_y.dx, inplace=False)
                Bz_corr = part2 if Bz_corr is None else Bz_corr.add(part2, compress=True)

            dBdt_corr = self.B.create_like({X: Bx_corr, Y: By_corr, Z: Bz_corr})
            dBdt_corr = dBdt_corr.scalar_multiply(-self.matl_params.c / 2, inplace=False)  ## scale dBdt by -1 later

            J_corr = self._calculate_upwind_current(current_density, for_E=False, compress=compress,
                                                    compress1=compress1)
            J_corr.scalar_multiply(-1, inplace=True)  ## scale dBdt by -1 later on
            if J_corr is not None:
                dBdt_corr.add(J_corr, inplace=True)

            dBdt = dBdt.add(dBdt_corr)

        if self.psi is not None and self.psi.component is not None and self.psi.component.data is not None:
            grad_psi = self.psi.gradient(compress_level=compress1)
            grad_psi.scalar_multiply(self.gamma, inplace=True)

            # plt.figure()
            # plt.imshow(self.psi.get_field_data(comps=0))
            # plt.colorbar()
            # plt.title('psi')
            # plt.show()

            # plt.figure()
            # plt.imshow(grad_psi.get_field_data(comps=0))
            # plt.colorbar()
            # plt.title('grad psi x')
            # plt.show()

            # plt.figure()
            # plt.imshow(grad_psi.get_field_data(comps=1))
            # plt.colorbar()
            # plt.title('grad psi y')
            # plt.show()

            # # plt.figure()
            # # plt.imshow(grad_psi.get_field_data(comps=2))
            # # plt.colorbar()
            # # plt.title('grad psi z')
            # # plt.show()

            dBdt = dBdt.add(grad_psi, inplace=True, compress_level=0) if dBdt is not None else grad_psi

        if dBdt is not None:
            if compress:
                dBdt.compress(inplace=True, compress_level=compress)

            if self.matl_params.is_cgs:
                dBdt = dBdt.scalar_multiply(-self.matl_params.c, inplace=True)
            else:
                dBdt = dBdt.scalar_multiply(-1, inplace=True)

            dBdt.name = self.B.name if self.B is not None else 'B'

            if verbose_plot:

                for C in self.coords_x.coords:
                    dBdt_data = self.B.get_comp_data(C)
                    if dBdt_data is not None:
                        plt.figure()
                        plt.plot(np.real(dBdt_data))
                        plt.plot(np.imag(dBdt_data))
                        plt.title(f'B(t) {C}')
                        # plt.show()

                for C in self.coords_x.coords:
                    dBdt_data = dBdt.get_comp_data(C)
                    if dBdt_data is not None:
                        plt.figure()
                        plt.plot(np.real(dBdt_data))
                        plt.plot(np.imag(dBdt_data))
                        plt.title(f'dB/dt {C}')
                plt.show()

                # plt.figure()
                # plt.imshow(np.real(dBdt_data))
                # plt.colorbar()
                # plt.title('dB/dt re')
                # plt.figure()
                # plt.imshow(np.imag(dBdt_data))
                # plt.colorbar()
                # plt.title('dB/dt im')
                # plt.show()
        # X, Y = dBdt.components.keys()
        # print('dBdt', dBdt.components)
        # exit()

        return dBdt

    def _calculate_time_derivative_phi(self, charge_density=None, compress=1, compress1=0, compress2=0,
                                       verbose_plot=False):
        """ dphi/dt + div(E) = rho/eps0
        """
        # charge_density = self.compute_charge_density(compress=compress1)
        charge_density = charge_density if charge_density is not None else self.charge_density

        if charge_density is not None:
            charge_density = charge_density.scalar_multiply(1. / self.matl_params.eps0, inplace=True)  # inplace=False)
        dPhidt = charge_density

        if self.E is not None:
            divE = self.E.divergence(self.coords_x, compress_level=compress1, inner_compress_level=compress2)
            if divE is not None:
                divE.scalar_multiply(-1, inplace=True)
            if dPhidt is not None:
                dPhidt.add(divE, inplace=True, compress_level=0)
            else:
                dPhidt = divE

        if self.background_E0 is not None:
            if self.divE0 is None:
                divE0 = self.E.divergence(self.coords_x, compress_level=compress1, inner_compress_level=compress2)
                self.divE0 = divE0.copy()
            else:
                divE0 = self.divE0.copy()

            if divE0 is not None:
                divE0.scalar_multiply(-1, inplace=True)

            if dPhidt is not None:
                dPhidt.add(divE0, inplace=True, compress_level=0)
            else:
                dPhidt = divE0

        if dPhidt is not None:
            dPhidt.scalar_multiply(self.chi, inplace=True)
            if compress:
                dPhidt.compress(inplace=True, compress_level=compress)
            dPhidt.name = self.phi.name if self.phi is not None else 'phi'

        if verbose_plot:
            print('self.E', self.E.components)
            print('div E', divE.components)

            plt.figure()
            dPhidt_data = dPhidt.get_comp_data()
            plt.plot(np.real(dPhidt_data), label=f're')
            plt.plot(np.imag(dPhidt_data), label=f'im')
            plt.title('dphi/dt')
            plt.show()

        return dPhidt

    def _calculate_time_derivative_psi(self, compress=1, compress1=0, compress2=0, verbose_plot=False):
        """ eps0*mu0/gamma dpsi/dt + div(B) = 0
        """
        divB = None

        if self.B is not None:
            divB = self.B.divergence(self.coords_x, compress_level=compress, inner_compress_level=compress2)

        if self.background_B0 is not None:
            if self.divB0 is None:
                divB0 = self.B.divergence(self.coords_x, compress_level=compress, inner_compress_level=compress2)
                self.divB0 = divB0.copy()
            if divB is not None:
                divB.add(self.divB0, inplace=True, compress_level=0)
            else:
                divB = self.divB0.copy()

        if divB is not None:
            divB.scalar_multiply(-1. / self.gamma * self.matl_params.c ** 2, inplace=True)
            divB.name = self.psi.name if self.psi is not None else 'psi'

        if verbose_plot:
            print('B', self.B.components)
            print('divB', divB.components)

            plt.figure()
            divB_data = divB.get_comp_data()
            plt.plot(np.real(divB_data), label=f're')
            plt.plot(np.imag(divB_data), label=f'im')
            plt.legend()
            plt.title('dpsi/dt')
            plt.show()

        return divB

    def get_divE(self, compress=5, ) -> Optional['ScalarField']:
        if self.is_yee:
            ax_deriv_configs = {}
            for compID in self.coords_x.coords:
                ax = self.coords_x.get_axis(compID)
                comp = self.B.components.get(compID, None)
                if comp is not None:
                    dc = comp.ax_deriv_configs.get(ax, None)
                    if dc is not None:
                        dc_ = dc.copy()
                        dc_.update(fd_type=FDType.FORWARD, order=0)
                        ax_deriv_configs[ax] = dc_
        else:
            ax_deriv_configs = None

        divE = self.E.divergence(coord_sys=self.coords_x, compress_level=compress, ax_deriv_configs=ax_deriv_configs)
        divE.is_sqrt = True
        return divE

    def get_divB(self, compress=5, ) -> Optional['ScalarField']:
        if self.is_yee:
            ax_deriv_configs = {}
            for compID in self.coords_x.coords:
                ax = self.coords_x.get_axis(compID)
                comp = self.B.components.get(compID, None)
                if comp is not None:
                    dc = comp.ax_deriv_configs.get(ax, None)
                    if dc is not None:
                        dc_ = dc.copy()
                        dc_.update(fd_type=FDType.FORWARD, order=0)
                        ax_deriv_configs[ax] = dc_
        else:
            ax_deriv_configs = None

        divB = self.B.divergence(coord_sys=self.coords_x, compress_level=compress, ax_deriv_configs=ax_deriv_configs)
        divB.is_sqrt = True
        return divB

    def check_poisson(self, compress=5, ) -> Optional['ScalarField']:
        divE = self.get_divE(compress=compress)
        divE.scalar_multiply(-1 * self.matl_params.eps0, inplace=True)
        return divE.add(self.charge_density, inplace=True)

    def check_divB(self, compress=5, ) -> Optional['ScalarField']:
        return self.get_divB(compress=compress)

    def clean_fields(self, compress=1):
        """ clean fields via projection:
            div E - rho = lapl(TH)
            div B = lapl(PHI)
            E' = E - grad(TH) --> div E' = div E - lapl(TH) = rho
            B' = B - grad(PHI) --> div B' = div B - lapl(PHI) = 0
        """

        divE = self.get_divE()
        divE.scalar_multiply(-1 * self.matl_params.eps0, inplace=True)
        divE.add(self.charge_density, inplace=True)

        # divB = self.B.divergence(coord_sys=self.coords_x, compress_level=compress)
        divB = self.get_divB()

        ##### lapl(phi) = div E - rho, E_clean = E - grad(phi) #####
        # ## deriv configs of div E
        # E_comp, compID = None, None
        # coords = iter(self.coords_x.coords)
        # while E_comp is None:
        #     compID = next(coords)
        #     E_comp = self.E.components.get(compID, None)
        # ax_ = self.coords_x.get_axis(compID.type)
        # E_ax_deriv_configs = {ax: cfg.copy() for ax, cfg in E_comp.ax_deriv_configs.items()}
        # if ax_ in E_ax_deriv_configs:
        #     dc = E_ax_deriv_configs[ax_].derivative_bc()
        #     if self.is_yee:
        #         dc.update(fd_type=FDType.CENTER, order=1)
        #     E_ax_deriv_configs[ax_] = dc
        # else:
        #     print('ax_ not in E_ax_deriv_configs', ax_)

        # ## deriv configs of div B
        # B_comp, compID = None, None
        # coords = iter(self.coords_x.coords)
        # while B_comp is None:
        #     compID = next(coords)
        #     B_comp = self.B.components.get(compID, None)
        # ax_ = self.coords_x.get_axis(compID.type)
        # B_ax_deriv_configs = {ax: cfg.copy() for ax, cfg in B_comp.ax_deriv_configs.items()}
        # if ax_ in B_ax_deriv_configs:
        #     dc = B_ax_deriv_configs[ax_].derivative_bc()
        #     if self.is_yee:
        #         dc.update(fd_type=FDType.CENTER, order=1)
        #         print('ax', ax_, 'dc', dc)
        #     B_ax_deriv_configs[ax_] = dc
        # else:
        #     print('ax_ not in B_ax_deriv_configs', ax_)

        ## deriv configs of div E
        E_ax_deriv_configs = {}
        for ax in self.grid_X.axes:
            compID = ax.coordinate
            E_comp = self.E.components.get(compID, None)
            if E_comp is not None:
                dc = E_comp.ax_deriv_configs[ax]
                dc = dc.derivative_bc()
                if self.is_yee:
                    ## derivative for divergence = FDType.FORWARD
                    dc.update(fd_type=FDType.CENTER, order=1)
                    dc = dc.shifted_bc()
                E_ax_deriv_configs[ax] = dc

        ## deriv configs of div B
        B_ax_deriv_configs = {}
        for ax in self.grid_X.axes:
            compID = ax.coordinate
            B_comp = self.B.components.get(compID, None)
            if B_comp is not None:
                dc = B_comp.ax_deriv_configs[ax]
                dc = dc.derivative_bc()
                if self.is_yee:
                    dc.update(fd_type=FDType.CENTER, order=1)
                    dc = dc.shifted_bc()
                B_ax_deriv_configs[ax] = dc

        # print('init divB err', self.check_divB().component.frobenius_norm())
        # print('init divE err', self.check_poisson().component.frobenius_norm())

        if divE.component is not None:
            if divE.component.frobenius_norm() > 1.0e-13:
                print('E', E_ax_deriv_configs)
                inv_lapl = self.grid_X.inverse_laplacian_mpo(ax_deriv_configs=E_ax_deriv_configs,
                                                             eeo_grid=(not self.is_yee))
                phi = divE.component.apply(inv_lapl, zipup=True)

                # lapl_mpo = self.grid_X.laplacian_mpo(ax_deriv_configs=E_ax_deriv_configs)
                # phi = divE.component.solve(lapl_mpo, compress_type=CompressType.DMRG, init_guess=phi)

                if self.is_yee:
                    for ax, dc in E_ax_deriv_configs.items():
                        dc.update(fd_type=FDType.BACKWARD, order=0)

                phi.ax_deriv_configs = E_ax_deriv_configs
                phi = ScalarField('phi', self.grid_X, phi)
                grad_phi = phi.gradient()
                self.E.add(grad_phi.scalar_multiply(-1, inplace=True), inplace=True, compress_level=compress)

        if divB.component is not None:
            if divB.component.frobenius_norm() > 1.0e-13:
                print('B ax', B_ax_deriv_configs)
                inv_lapl = self.grid_X.inverse_laplacian_mpo(ax_deriv_configs=B_ax_deriv_configs,
                                                             eeo_grid=(not self.is_yee))
                psi = divB.component.apply(inv_lapl, zipup=True)
                ## this doesn't work well, probably bc it's not well conditioned.
                # lapl_mpo = self.grid_X.laplacian_mpo(ax_deriv_configs=B_ax_deriv_configs)
                # lapl_mpo.data.distribute_exponent()
                # divB.component.data.distribute_exponent()
                # psi = divB.component.solve(lapl_mpo, compress_type=CompressType.DMRG,  is_H=False, init_guess=psi)
                # print('B ax deriv configs', B_ax_deriv_configs)
                # plt.figure()
                # plt.imshow(psi.get_data())
                # plt.colorbar()
                # plt.show()

                if self.is_yee:
                    for ax, dc in B_ax_deriv_configs.items():
                        dc.update(fd_type=FDType.BACKWARD, order=0)
                psi.ax_deriv_configs = B_ax_deriv_configs
                psi = ScalarField('phi', self.grid_X, psi)
                grad_psi = psi.gradient()  ## if is_yee:  use Backward stencil, order=0

                # test_comps = 0
                # for compID, comp in grad_psi.components.items():
                #     plt.figure()
                #     plt.imshow(comp.get_data())
                #     plt.colorbar()
                #     test_comps = test_comps + comp.get_data()
                # plt.show()

                # print('np.linalg.norm(test comps)', np.linalg.norm(test_comps))

                self.B.add(grad_psi.scalar_multiply(-1, inplace=True), inplace=True, compress_level=compress)

        # print('post divB err', self.check_divB().component.frobenius_norm())
        # print('post divE err', self.check_poisson().component.frobenius_norm())
        print('self.B max bond', self.B.max_bonds())
        print('self.E max bond', self.E.max_bonds())
        # print('self.B configs', self.B.compress_config.max_bonds)
        # exit()

    def get_constraints(self, charge: 'GridTN' = None
                        ) -> tuple[
        list[dict[int, qtn.MatrixProductOperator]], list[Union[qtn.MatrixProductState, float]]]:
        div_ops_E = self.coords_x._gtn_divergence_mpos(self.E)
        div_ops_B = self.coords_x._gtn_divergence_mpos(self.B)

        # for c, gtn_op in div_ops_E.items():
        #     gtn_op.data.distribute_exponent()
        # for c, gtn_op in div_ops_B.items():
        #     gtn_op.data.distribute_exponent()

        constraint_E = {(6, coord.type.value): [gtn_mpo.data] for coord, gtn_mpo in div_ops_E.items()}
        constraint_B = {(7, coord.type.value + 3): [gtn_mpo.data] for coord, gtn_mpo in div_ops_B.items()}

        ## div(E) = rho/eps0 [SI];  div(E) = 4 pi rho0 [CGS]
        charge = None
        if self.charge_density is not None:
            if self.matl_params.is_cgs:
                charge = self.charge_density.scalar_multiply(4 * np.pi, inplace=False)
            else:
                charge = self.charge_density.scalar_multiply(1. / self.matl_params.eps0, inplace=False)

        constraint_vals = [(charge.data if charge is not None else 0), 0]

        ### check
        print('current divB', self.check_divB().norm())
        print('current poisson', self.check_poisson().norm())

        # tot = None
        # for compID, gtn_div in div_ops_E.items():
        #     tmp_E = self.E.components.get(compID, None)
        #     tmp = tmp_E.apply(gtn_div)
        #     if tot is None:
        #         tot = tmp
        #     else:
        #         tot = tot.add(tmp)
        # print('div E norm', tot.norm())
        #
        # tot = None
        # for compID, gtn_div in div_ops_B.items():
        #     tmp_B = self.B.components.get(compID, None)
        #     if tmp_B is None:
        #         continue
        #     tmp = tmp_B.apply(gtn_div)
        #     if tot is None:
        #         tot = tmp
        #     else:
        #         tot = tot.add(tmp)
        # print('div B norm', tot.norm())

        return [constraint_E, constraint_B], constraint_vals
        # return [constraint_E], constraint_vals[:1]
        # return [constraint_B], constraint_vals[1:]
        # return [], []

    def get_state_dict(self, cleaning=False) -> dict[int, 'GridTN']:
        """ get Ex, Ey, Ez; Bx, By, Bz as dictionary with field "number" as key
            excludes fields that are None / 0
        """
        print([compID.type for compID, comp in self.E.components.items()])
        E_fields_dict = {compID.type.value: comp.copy() for compID, comp in self.E.components.items()
                         if comp is not None and comp.data is not None}
        B_fields_dict = {compID.type.value + 3: comp.copy() for compID, comp in self.B.components.items()
                         if comp is not None and comp.data is not None}
        all_fields_dict = {**E_fields_dict, **B_fields_dict}

        if cleaning:
            print('cleaning', self.phi.component, self.psi.component)
            if self.phi is not None and self.phi.component is not None and self.phi.component.data is not None:
                phi_comp = self.phi.component.copy()
            else:
                phi_comp = self.grid_X.make_zero_gridTN()

            if self.psi is not None and self.psi.component is not None and self.psi.component.data is not None:
                psi_comp = self.psi.component.copy()
            else:
                psi_comp = self.grid_X.make_zero_gridTN()

            # all_fields_dict[6] = self.phi.component.copy() if self.phi is not None else self.grid_X.make_zero_gridTN()
            # all_fields_dict[7] = self.psi.component.copy() if self.psi is not None else self.grid_X.make_zero_gridTN()
            all_fields_dict[6] = phi_comp
            all_fields_dict[7] = psi_comp
            # if self.phi is not None:
            #     all_fields_dict[6] = self.phi.component.copy()
            # if self.psi is not None:
            #     all_fields_dict[7] = self.psi.component.copy()

        return all_fields_dict

    def get_derivative_dict(self, dt=None, cleaning=False) -> dict[tuple[int, int], 'GridTN']:
        """ get operations acting on E, B to obtain dE/dt, dB/dt
            information returned as a dict with keys (output ind, input ind)
        """

        ### get curl operator
        def _get_curl_op(comp_ax_deriv_configs):

            gr = self.grid_X

            dict_mpos = {}
            for compID in self.coords_x.coords:
                i = compID.type  # self.ax_coords[axID]
                i1, i2 = (i + 1) % 3, (i + 2) % 3

                c1_ = self.coords_x.type_coords.get(i1, None)
                c2_ = self.coords_x.type_coords.get(i2, None)

                ax1_ = self.coords_x.coord_axes[c1_]
                ax2_ = self.coords_x.coord_axes[c2_]
                # print('get curl', ax1_, ax2_)

                # if c2_ in gtn_comps and ax1_ is not None:
                if ax1_ is not None:
                    ax_deriv_configs = comp_ax_deriv_configs.get(c2_, {})  # [compID]
                    ax_deriv_configs = {} if ax_deriv_configs is None else ax_deriv_configs
                    mpo_1 = gr.get_firstderivative_mpo(ax1_, deriv_config=ax_deriv_configs.get(ax1_, None)).copy()
                else:
                    mpo_1 = None

                # if c1_ in gtn_comps and ax2_ is not None:
                if ax2_ is not None:
                    ax_deriv_configs = comp_ax_deriv_configs.get(c1_, {})  # [compID]
                    ax_deriv_configs = {} if ax_deriv_configs is None else ax_deriv_configs
                    mpo_2 = gr.get_firstderivative_mpo(ax2_, deriv_config=ax_deriv_configs.get(ax2_, None)).copy()
                    mpo_2 = mpo_2.scalar_multiply(-1, inplace=False)
                else:
                    mpo_2 = None  ## vector_field[i] = 0

                if mpo_1 is None and mpo_2 is None:
                    continue

                if mpo_2 is not None:
                    dict_mpos[(i.value, i1)] = mpo_2
                if mpo_1 is not None:
                    dict_mpos[(i.value, i2)] = mpo_1

            return dict_mpos

        ## dE/dt = curl(B) [[excludes + J]
        ## dB/dt = curl(E)
        E_ax_deriv_configs = {compID: comp.ax_deriv_configs for compID, comp in self.E.components.items()}
        B_ax_deriv_configs = {compID: comp.ax_deriv_configs for compID, comp in self.B.components.items()}
        dEdt_mpos = _get_curl_op(B_ax_deriv_configs)
        dBdt_mpos = _get_curl_op(E_ax_deriv_configs)

        ## adjust input, output coords accordingly
        all_mpos = {}
        ## dE/dt = c^2 * curl(B) - 1/eps0 * J [SI];  dE/dt = c * curl(B) - 4pi J [CGS]
        for (out_ind, in_ind), gtn in dEdt_mpos.items():
            if self.matl_params.is_cgs:
                gtn.scalar_multiply(self.matl_params.c, inplace=True)
            else:
                gtn.scalar_multiply(self.matl_params.c ** 2, inplace=True)
            all_mpos[(out_ind, in_ind + 3)] = gtn

        ## dB/dt = - curl(E) [SI];  dB/dt = - c curl(E) [CGS]
        for (out_ind, in_ind), gtn in dBdt_mpos.items():
            if self.matl_params.is_cgs:
                gtn.scalar_multiply(-self.matl_params.c, inplace=True)
            else:
                gtn.scalar_multiply(-1, inplace=True)
            all_mpos[(out_ind + 3, in_ind)] = gtn

        ## add upwinding contributions
        print('self.upwind', self.upwind, 'c', self.matl_params.c)
        if self.upwind:
            uw_scale = 1.0
            for i in range(3):
                i1, i2 = (i + 1) % 3, (i + 2) % 3

                c1_ = self.coords_x.get_coord(i1)
                c2_ = self.coords_x.get_coord(i2)

                ax1_ = self.coords_x.get_axis(c1_) if c1_ is not None else None
                ax2_ = self.coords_x.get_axis(c2_) if c2_ is not None else None
                print('E upwind ax', c1_, c2_, ax1_, ax2_)

                compID = self.coords_x.type_coords.get(i, None)
                # print('E comp', compID, compID in self.E.components)
                if compID not in self.E.components:
                    continue

                E_ax_deriv_configs = self.E[compID].ax_deriv_configs

                sum_ddx_E = None
                if ax1_ is not None:
                    ddx1E = self.grid_X.get_secondderivative_mpo(ax1_, ax1_, deriv_config1=E_ax_deriv_configs[ax1_])
                    ddx1E = ddx1E.scalar_multiply(ax1_.dx)
                    # ddx1E = ddx1E.scalar_multiply(ax1_.dx * dt / ax1_.dx * self.matl_params.c)
                    sum_ddx_E = ddx1E
                    # print('E ax1 is not None', compID, ax1_)

                if ax2_ is not None:
                    ddx2E = self.grid_X.get_secondderivative_mpo(ax2_, ax2_, deriv_config1=E_ax_deriv_configs[ax2_])
                    ddx2E = ddx2E.scalar_multiply(ax2_.dx)
                    # ddx2E = ddx2E.scalar_multiply(ax2_.dx * dt / ax2_.dx * self.matl_params.c)
                    sum_ddx_E = ddx2E if sum_ddx_E is None else sum_ddx_E.add(ddx2E, compress=True)
                    # print('E ax2 is not None', compID, ax2_)

                if sum_ddx_E is not None:
                    if True:  # sum_ddx_E.frobenius_norm() > 1.0e-12:
                        sum_ddx_E = sum_ddx_E.scalar_multiply(uw_scale * self.matl_params.c / 2, inplace=False)

                        ## electric fields
                        all_mpos[(i, i)] = sum_ddx_E  ## this term doesn't exist from original EOMs

            for i in range(3):
                i1, i2 = (i + 1) % 3, (i + 2) % 3

                # c1_ = self.coords_x.type_coords.get(i1, None)
                # c2_ = self.coords_x.type_coords.get(i2, None)
                c1_ = self.coords_x.get_coord(i1)
                c2_ = self.coords_x.get_coord(i2)

                # ax1_ = self.coords_x.coord_axes[c1_]
                # ax2_ = self.coords_x.coord_axes[c2_]
                ax1_ = self.coords_x.get_axis(c1_) if c1_ is not None else None
                ax2_ = self.coords_x.get_axis(c2_) if c2_ is not None else None
                print('B upwind ax', c1_, c2_, ax1_, ax2_)

                compID = self.coords_x.type_coords.get(i, None)
                if compID not in self.B.components:
                    continue

                B_ax_deriv_configs = self.B[compID].ax_deriv_configs

                sum_ddx_B = None
                if ax1_ is not None:
                    ddx1B = self.grid_X.get_secondderivative_mpo(ax1_, ax1_, deriv_config1=B_ax_deriv_configs[ax1_])
                    ddx1B = ddx1B.scalar_multiply(ax1_.dx)
                    # ddx1B = ddx1B.scalar_multiply(ax1_.dx * dt / ax1_.dx * self.matl_params.c)
                    sum_ddx_B = ddx1B
                    # print('B ax1 is not None', compID, ax1_)

                if ax2_ is not None:
                    ddx2B = self.grid_X.get_secondderivative_mpo(ax2_, ax2_, deriv_config1=B_ax_deriv_configs[ax2_])
                    ddx2B = ddx2B.scalar_multiply(ax2_.dx)
                    # ddx2B = ddx2B.scalar_multiply(ax2_.dx * dt / ax2_.dx * self.matl_params.c)
                    sum_ddx_B = ddx2B if sum_ddx_B is None else sum_ddx_B.add(ddx2B, compress=True)
                    # print('B ax2 is not None', compID, ax2_)

                if sum_ddx_B is not None:
                    if True:  # sum_ddx_B.frobenius_norm() > 1.0e-12:
                        sum_ddx_B = sum_ddx_B.scalar_multiply(uw_scale * self.matl_params.c / 2, inplace=False)

                        ## magnetic fields
                        all_mpos[(i + 3, i + 3)] = sum_ddx_B  ## this term doesn't exist from original EOMs

        return all_mpos

    def get_constraint_dict(self, chi=1.0, gamma=1.0) -> tuple[dict[tuple[int, int], 'GridTN'], dict[int, 'GridTN']]:
        """ get constraints of E, B that need to be satisfied
            weight is the Lagrange multiplier
            [ div(E) = rho/eps0 ] * weight
            [ div(B) = 0 ] * weight
            (implies --> d/dt rho + div(J) = 0)
            information returned as a dict with keys (output ind, input ind)

            ## 1/chi dphi/dt = - div(E) + rho/eps0  [SI]
            ## 1/c^2 / gamma dpsi/dt = -div(B)
        """

        ### get div operator
        def _get_div_op(comp_ax_deriv_configs, out_ind=0):

            gr = self.grid_X

            dict_mpos = {}
            for compID in self.coords_x.coords:
                i = compID.type  # self.ax_coords[axID]

                c_ = self.coords_x.type_coords.get(i, None)
                ax_ = self.coords_x.coord_axes[c_]

                if ax_ is not None:
                    ax_deriv_configs = comp_ax_deriv_configs.get(c_, {})  # [compID]
                    # ax_deriv_configs = {} if ax_deriv_configs is None else ax_deriv_configs
                    deriv_configs = ax_deriv_configs.get(ax_, None)
                    div_deriv_configs = deriv_configs.derivative_bc() if deriv_configs is not None else None
                    # print('ax deriv configs', ax_, div_deriv_configs)
                    mpo_1 = gr.get_firstderivative_mpo(ax_, deriv_config=div_deriv_configs).copy()
                else:
                    mpo_1 = None

                if mpo_1 is None:
                    continue

                if mpo_1 is not None:
                    dict_mpos[(out_ind, i.value)] = mpo_1

            return dict_mpos

        ## dE/dt = curl(B) [[excludes + J]
        ## dB/dt = curl(E)
        E_ax_deriv_configs = {compID: comp.ax_deriv_configs for compID, comp in self.E.components.items()}
        B_ax_deriv_configs = {compID: comp.ax_deriv_configs for compID, comp in self.B.components.items()}
        divE_mpos = _get_div_op(B_ax_deriv_configs, out_ind=6)
        divB_mpos = _get_div_op(E_ax_deriv_configs, out_ind=7)

        ## adjust input, output coords accordingly
        constraint_mpos = {}
        target_vecs = {}
        ## div(E) = rho/eps0 [SI];  div(E) = 4 pi rho0 [CGS]
        for (out_ind, in_ind), gtn in divE_mpos.items():
            constraint_mpos[(out_ind, in_ind)] = gtn.scalar_multiply(-chi, inplace=False)

            if self.charge_density is not None:
                rho_i = self.charge_density.get(self.coords_x.coords[out_ind], None)
                if rho_i is not None:
                    if self.matl_params.is_cgs:
                        rho_i = rho_i.scalar_multiply(4 * np.pi * chi, inplace=False)
                    else:
                        rho_i = rho_i.scalar_multiply(1. / self.matl_params.eps0 * chi, inplace=False)
                target_vecs[out_ind] = rho_i

        ## dB/dt = - curl(E) [SI];  dB/dt = - c curl(E) [CGS]
        for (out_ind, in_ind), gtn in divB_mpos.items():
            constraint_mpos[(out_ind, in_ind + 3)] = gtn.scalar_multiply(-gamma * self.matl_params.c ** 2,
                                                                         inplace=False)

        return constraint_mpos, target_vecs

    def get_combined_derivative_mpo(self) -> 'GridTN':

        # ### get curl operator
        # def _get_curl_op(comp_ax_deriv_configs):
        #
        #     gr = self.grid_X
        #     gtn_comps = self.coords_x.coords
        #
        #     dict_mpos = {}
        #     for compID in self.coords_x.coords:
        #         i = compID.type  # self.ax_coords[axID]
        #         i1, i2 = (i + 1) % 3, (i + 2) % 3
        #
        #         c1_ = self.coords_x.type_coords.get(i1, None)
        #         c2_ = self.coords_x.type_coords.get(i2, None)
        #
        #         ax1_ = self.coords_x.coord_axes[c1_]
        #         ax2_ = self.coords_x.coord_axes[c2_]
        #
        #         if c2_ in gtn_comps and ax1_ is not None:
        #             ax_deriv_configs = comp_ax_deriv_configs.get(c2_, {})  # [compID]
        #             ax_deriv_configs = {} if ax_deriv_configs is None else ax_deriv_configs
        #             mpo_1 = gr.get_firstderivative_mpo(ax1_, deriv_config=ax_deriv_configs.get(ax1_, None)).copy()
        #         else:
        #             mpo_1 = None
        #
        #         if c1_ in gtn_comps and ax2_ is not None:
        #             ax_deriv_configs = comp_ax_deriv_configs.get(c1_, {})  # [compID]
        #             ax_deriv_configs = {} if ax_deriv_configs is None else ax_deriv_configs
        #             mpo_2 = gr.get_firstderivative_mpo(ax2_, deriv_config=ax_deriv_configs.get(ax2_, None)).copy()
        #             mpo_2 = mpo_2.scalar_multiply(-1, inplace=False)
        #         else:
        #             mpo_2 = None  ## vector_field[i] = 0
        #
        #         if mpo_1 is None and mpo_2 is None:
        #             continue
        #
        #         if mpo_2 is not None:
        #             dict_mpos[(i.value, i1)] = mpo_2
        #         if mpo_1 is not None:
        #             dict_mpos[(i.value, i2)] = mpo_1
        #
        #     return dict_mpos
        #
        # ## dE/dt = curl(B) [[excludes + J]
        # ## dB/dt = curl(E)
        # E_ax_deriv_configs = {compID: comp.ax_deriv_configs for compID, comp in self.E.components.items()}
        # B_ax_deriv_configs = {compID: comp.ax_deriv_configs for compID, comp in self.B.components.items()}
        # dEdt_mpos = _get_curl_op(B_ax_deriv_configs)
        # dBdt_mpos = _get_curl_op(E_ax_deriv_configs)
        #
        # ## adjust input, output coords accordingly
        # all_mpos = {}
        # ## dE/dt = c^2 * curl(B) - 1/eps0 * J [SI];  dE/dt = c * curl(B) - 4pi J [CGS]
        # for (out_ind, in_ind), gtn in dEdt_mpos.items():
        #     if self.matl_params.is_cgs:
        #         gtn.scalar_multiply(self.matl_params.c, inplace=True)
        #     else:
        #         gtn.scalar_multiply(self.matl_params.c ** 2, inplace=True)
        #     all_mpos[(out_ind, in_ind + 3)] = gtn
        #
        # ## dB/dt = - curl(E) [SI];  dB/dt = - c curl(E) [CGS]
        # for (out_ind, in_ind), gtn in dBdt_mpos.items():
        #     if self.matl_params.is_cgs:
        #         gtn.scalar_multiply(-self.matl_params.c, inplace=True)
        #     else:
        #         gtn.scalar_multiply(-1, inplace=True)
        #     all_mpos[(out_ind + 3, in_ind)] = gtn

        # print('all mpos')
        # print(all_mpos)
        all_mpos = self.get_derivative_dict()

        ### build single MPS
        index_order = list(np.ndindex(6, 6))
        curl_op = self.grid_X.build_indexed_gtn(all_mpos, index_order=index_order)
        return curl_op

    def get_num_active_fields(self) -> int:
        num_fields = 0
        for C in self.coords_x.coords:
            EC = self.E.components.get(C, None)
            if EC is not None:
                num_fields += 1
            BC = self.B.components.get(C, None)
            if BC is not None:
                num_fields += 1
        return num_fields

    def get_combined_state(self) -> 'GridTN':
        """ get Ex, Ey, Ez; Bx, By, Bz as a single MPS
            excludes fields that are None / 0
        """
        index_order = np.arange(6)
        # all_mps = {}
        # for C in self.coords_x.coords:
        #     EC = self.E.components.get(C, None)
        #     if EC is not None:
        #         # print(f'E{C} is not None', EC.data is None)
        #         all_mps[C.type] = EC.copy()
        #     BC = self.B.components.get(C, None)
        #     if BC is not None:
        #         # print(f'B{C} is not None', BC.data is None)
        #         all_mps[3 + C.type] = BC.copy()
        all_mps = self.get_state_dict()
        tot_state = self.grid_X.build_indexed_gtn(all_mps, index_order=index_order)
        return tot_state

    # @profile
    def crank_nicolson(self, dt, time: float = None, use_A2=True, inplace=False,
                       err_tol: float = np.sqrt(CUTOFF), max_iter: int = 100,
                       compress_level: int = 1, verbose_plot=False, **deriv_kwargs):
        """
        ## dE/dt = c^2 * curl(B) - 1/eps0 * J [SI];  dE/dt = c * curl(B) - 4pi J [CGS]
        ## dB/dt = - curl(E) [SI];  dB/dt = - c curl(E) [CGS]

        | 1/dt * I, -s/2 curl | |E^(n+1)| = | 1/dt * I, s/2 curl | |E^(n)|  - |J^(n+1/2)|
        |-s/2 curl,  1/dt * I | |B^(n+1)| = | s/2 curl, 1/dt * I | |B^(n)|  + |    0    |
        where s is a sign and coefficient as defined by Maxwell's eq

        | 1/dt * I, -s/2 curl | |E^(n+1)| = | 1/dt * I, s/2 curl | |E^(n)|  + |c^2 curl(B0)| - |J^(n+1/2)|
        |-s/2 curl,  1/dt * I | |B^(n+1)| = | s/2 curl, 1/dt * I | |B^(n)|  - |curl(E0)| + |    0    |
        where s is a sign and coefficient as defined by Maxwell's eq
        """
        # use_dmrg_new = True  # False
        # if use_dmrg_new:
        #     return self.crank_nicolson_dmrg(dt, time=time, err_tol=err_tol, inplace=inplace, max_iter=max_iter,
        #                                     compress_level=compress_level, verbose_plot=verbose_plot, **deriv_kwargs)

        new_sys = self if inplace else self.copy()
        use_dmrg = True
        try_adaptive_solve = True  # False

        ## current, theoretically at time n + 1/2
        j = self.current_density
        if j is not None:
            j = j.copy()
            if self.matl_params.is_cgs:
                j.scalar_multiply(-4 * np.pi, inplace=True)
            else:
                j.scalar_multiply(-1 / self.matl_params.eps0, inplace=True)

        ### dmrg method
        if use_dmrg:
            print("HERE, EM USE DMRG")

            num_active_fields = self.get_num_active_fields()
            all_fields_vec = self.get_combined_state()
            all_fields_time_deriv = self.get_combined_derivative_mpo()

            compress_opts = new_sys.E.compress_config.get_compress_opts(1).copy()
            max_bond = compress_opts.get('max_bond', None)
            if max_bond is not None:
                compress_opts['max_bond'] = max_bond * num_active_fields

            jF = None
            if j is not None:
                if self.upwind:
                    j_corrE, j_corrB = self._calculate_upwind_current(compress=compress_level,
                                                                      compress1=compress_level + 1)
                    jF = self.grid_X.build_indexed_gtn(
                        {**{compID.type.value: comp for compID, comp in j_corrE.components.items()},
                         **{compID.type.value + 3: comp for compID, comp in j_corrB.components.items()}},
                        index_order=list(range(6)))
                else:
                    jF = self.grid_X.build_indexed_gtn(
                        {compID.type.value: comp for compID, comp in j.components.items()},
                        index_order=list(range(6)))

            curlF_bg = None
            background_comps = {}
            if self.background_E0 is not None:
                if self.curlE0 is None:
                    self.curlE0 = self.background_E0.curl(self.coords_x, compress_level=5)
                scalar_E = -self.matl_params.c if self.matl_params.is_cgs else -1
                back_E_comps = {compID.type.value + 3: comp.scalar_multiply(scalar_E)
                                for compID, comp in self.curlE0.components.items()}
                background_comps.update(back_E_comps)
            if self.background_B0 is not None:
                if self.curlB0 is None:
                    print('calculating curlB0')
                    self.curlB0 = self.background_B0.curl(self.coords_x, compress_level=5)
                scalar_B = self.matl_params.c if self.matl_params.is_cgs else self.matl_params.c ** 2
                back_B_comps = {compID.type.value: comp.scalar_multiply(scalar_B, inplace=False)
                                for compID, comp in self.curlB0.components.items()}
                background_comps.update(back_B_comps)

            if len(background_comps) > 0:
                curlF_bg = self.grid_X.build_indexed_gtn(background_comps, index_order=list(range(6)))

            sourceF = jF
            if curlF_bg is not None:
                sourceF = curlF_bg.add(sourceF, compress=5, inplace=False)
                # print('sourceF norm', sourceF.frobenius_norm())

            def _implicit_solver(dt_, fields_vec, verbose_output=False, **solver_kwargs):

                sourceF_dt = None
                if sourceF is not None and sourceF.data is not None:
                    sourceF_dt = sourceF.scalar_multiply(dt_, inplace=False)

                ## explicit contribution
                dFdt1 = fields_vec.apply(all_fields_time_deriv, compress=compress_level)  ## nans in compress
                dFdt1 = dFdt1.scalar_multiply(0.5 * dt_, inplace=False)
                if sourceF_dt is not None:
                    F1 = fields_vec.add(dFdt1, inplace=False)
                    F1 = F1.add(sourceF_dt, inplace=False, compress=compress_level, compress_opts=compress_opts)
                else:
                    F1 = fields_vec.add(dFdt1, inplace=False, compress=compress_level, compress_opts=compress_opts)
                # F1 = all_fields_vec.copy()
                # print('compress opts', compress_level, compress_opts)
                print('EM solve F1', F1.max_bond())

                # ########### tmp ##############
                # # reset fields
                # for i in range(3):
                #     compID = new_sys.coords_x.type_coords[i]
                #
                #     EC = new_sys.grid_X.select_indexed_gtn(F1, i).get_data()
                #     BC = new_sys.grid_X.select_indexed_gtn(F1, 3 + i).get_data()
                #
                #     if EC is not None:
                #         plt.figure()
                #         plt.plot(EC)
                #         plt.title(f'comp{i}')
                #
                #     if BC is not None:
                #         plt.figure()
                #         plt.plot(BC)
                #         plt.title(f'comp{i + 3}')
                # plt.show()
                # exit()

                ## implicit contribution operator
                dFdt_gtn = all_fields_time_deriv.scalar_multiply(-0.5 * dt_, inplace=False)
                iden_gtn = dFdt_gtn.get_like_iden()
                imp_gtn = dFdt_gtn.add(iden_gtn)
                print('EM implicit op', imp_gtn.max_bond())

                ## dmrg solve
                print('implicit dmrg solve', compress_opts)
                updated_F1, err, is_conv = F1.solve(imp_gtn, compress_type=CompressType.DMRG, use_A2=use_A2, is_H=False,
                                                    # init_guess=fields_vec,    ## actually gives wrong answer
                                                    compress_opts=compress_opts, verbose_output=True, **solver_kwargs)
                # exit()
                if verbose_output:
                    return updated_F1, err, is_conv
                return updated_F1

            if all_fields_vec is None and sourceF is None:
                return new_sys
            elif all_fields_vec is None and sourceF is not None:
                sourceF_dt = sourceF.scalar_multiply(dt, inplace=False)
                new_F1 = sourceF_dt
                # print('new F1', new_F1.data)
                # new_F1.data.site_ind_id = new_F1.data.site_ind_id + '_j_'
            else:

                # if try_adaptive_solve:
                #     nt, max_nt, is_conv = self.nt, 6, False
                #     adj_conv_tol = self.adj_conv_tol
                #     # nt_conv_tol = 1.0e-4 if adj_conv_tol is None else max(1.0e-4, adj_conv_tol)
                #
                #     old_fields_vec = all_fields_vec
                #     old_err = 1.0
                #     increased_nt = False
                #     while (not is_conv) and nt <= max_nt:
                #         print('nt', nt, adj_conv_tol)
                #         new_F1 = old_fields_vec
                #         for step in range(nt):
                #             new_F1, err, is_conv = _implicit_solver(dt / nt, new_F1, verbose_output=True,
                #                                                     max_tot_iter=10, conv_tol=adj_conv_tol)
                #             if nt < max_nt and (not is_conv) and 1.0e-5 < err < old_err * 0.8:
                #                 ## err < old_err:  noticeable reduction in error from reducing time step
                #                 old_err = np.abs(err)
                #                 nt = min(nt + 1, max_nt)
                #                 increased_nt = True
                #                 break
                #             elif (not is_conv) and err < 1.0e-3:
                #                 ## o.w. maybe just a representability issue and not a solving issue, so relax conv_tol
                #                 adj_conv_tol = 10 ** (np.round(np.log10(np.abs(err)), 1) + 0.1)
                #                 adj_conv_tol = min(1.0e-2, adj_conv_tol)
                #                 print('adjusted conv tol', adj_conv_tol)
                #
                #                 if err < adj_conv_tol:
                #                     if increased_nt and step == nt - 1:
                #                         ## increased nt but it didn't make a big change in accuracy
                #                         # nt = max(1, nt-1)
                #                         increased_nt = False
                #                     pass
                #                 else:
                #                     ## redo calc with same nt but with relaxed conv_tol
                #                     print('redo calc')
                #                     nt = min(nt + 1, max_nt)
                #                     break
                #             is_conv = (step == nt - 1)  ## reached the last step
                #             old_err = err
                #
                #     self.nt = nt
                #     self.adj_conv_tol = adj_conv_tol

                if try_adaptive_solve:
                    nt, max_nt, is_conv = self.nt, 4, False

                    old_fields_vec = all_fields_vec
                    while (not is_conv) and nt <= max_nt:
                        print('nt', nt)
                        new_F1 = old_fields_vec
                        for step in range(nt):
                            new_F1, err, is_conv = _implicit_solver(dt / nt, new_F1, verbose_output=True, )
                            if nt < max_nt and (not is_conv) and err > 1.0e-3:
                                nt = min(nt + 1, max_nt)
                                break

                            is_conv = (step == nt - 1)  ## reached the last step

                    self.nt = nt
                else:
                    new_F1 = _implicit_solver(dt, all_fields_vec)

            # reset fields
            for i in range(3):
                compID = new_sys.coords_x.type_coords[i]

                ref_EC = new_sys.E.components.get(compID, None)
                ax_deriv_configs = ref_EC.ax_deriv_configs if ref_EC is not None else None
                EC = new_sys.grid_X.select_indexed_gtn(new_F1, i)
                EC.ax_deriv_configs = ax_deriv_configs
                if not (EC is None and compID not in new_sys.E.componentIDs):
                    new_sys.E.components[compID] = EC
                    # print('EC', compID, EC.max_bond(), EC.canon_site)

                ref_BC = new_sys.B.components.get(compID, None)
                ax_deriv_configs = ref_BC.ax_deriv_configs if ref_BC is not None else None
                BC = new_sys.grid_X.select_indexed_gtn(new_F1, 3 + i)
                BC.ax_deriv_configs = ax_deriv_configs
                if not (BC is None and compID not in new_sys.E.componentIDs):
                    new_sys.B.components[compID] = BC
                    # print('BC', compID, BC.max_bond(), BC.canon_site)

            new_sys.E.compress(inplace=True, compress_level=1, norm_cutoff=1.0e-8)
            new_sys.B.compress(inplace=True, compress_level=1, norm_cutoff=1.0e-8)

            # for C, EC in new_sys.E.components.items():
            #     print('EC', C, EC.ax_deriv_configs)
            # for C, EC in new_sys.B.components.items():
            #     print('BC', C, EC.ax_deriv_configs)
            #
            #
            # plt.figure()
            # for i in range(3):
            #     compID = new_sys.coords_x.type_coords[i]
            #     EC = new_sys.E.get_comp_data(compID)
            #     if EC is not None:
            #         plt.plot(EC, label=f'E{compID}')
            # for i in range(3):
            #     compID = new_sys.coords_x.type_coords[i]
            #     BC = new_sys.B.get_comp_data(compID)
            #     if BC is not None:
            #         plt.plot(BC, label=f'B{compID}')
            # plt.legend()

            # ax_x, ax_y = new_sys.grid_X.axes
            # for i in range(3):
            #     compID = new_sys.coords_x.type_coords[i]
            #     EC = new_sys.E.get_comp_data(compID)
            #     EC = ax_x.basis.get_realspace_1D(EC, 0)
            #     EC = ax_y.basis.get_realspace_1D(EC, 1)
            #     if EC is not None:
            #         plt.figure()
            #         plt.imshow(np.real(EC))
            #         plt.title(f'E{compID}')
            #         plt.colorbar()
            # for i in range(3):
            #     compID = new_sys.coords_x.type_coords[i]
            #     BC = new_sys.B.get_comp_data(compID)
            #     BC = ax_x.basis.get_realspace_1D(BC, 0)
            #     BC = ax_y.basis.get_realspace_1D(BC, 1)
            #     if BC is not None:
            #         plt.figure()
            #         plt.imshow(np.real(BC))
            #         plt.title(f'B{compID}')
            #         plt.colorbar()
            # plt.show()

            # print('E')
            # print(new_sys.E.components)
            # print('B')
            # print(new_sys.B.components)
            # exit()

            # ## exact solve
            # print('exact solve')
            #
            # def get_data(gtn, is_mps=True):
            #     if isinstance(gtn, GridTN1D):           ## assumes F ordering
            #         tensor: qtn.Tensor = gtn.data.contract()  # contract all tensors
            #         tensor.modify(apply=lambda x: x * 10 ** gtn.data.exponent)
            #         if is_mps:
            #             out_inds = [gtn.data.site_ind_id.format(i) for i in range(gtn.data.num_tensors)]
            #         else:
            #             i_inds = [gtn.data.lower_ind_id.format(i) for i in range(gtn.data.num_tensors)]
            #             o_inds = [gtn.data.upper_ind_id.format(i) for i in range(gtn.data.num_tensors)]
            #
            #     elif isinstance(gtn, GridTN1DComb):     ## assumes B ordering
            #         branch_tensors = []
            #         if is_mps:
            #             out_inds = []
            #         else:
            #             i_inds, o_inds = [], []
            #
            #         for gr, b in gtn.branches.items():
            #             b = b.copy()
            #             if is_mps:
            #                 b.data.site_ind_id += f'_{gr}'
            #             else:
            #                 b.data.upper_ind_id += f'_{gr}'
            #                 b.data.lower_ind_id += f'_{gr}'
            #             tens_b = b.data.contract()
            #             tens_b.modify(apply=lambda x: x * 10 ** b.data.exponent)
            #             branch_tensors += [tens_b]
            #             if is_mps:
            #                 out_inds += [b.data.site_ind_id.format(i) for i in range(b.data.num_tensors)][::-1]
            #             else:
            #                 i_inds += [b.data.lower_ind_id.format(i) for i in range(b.data.num_tensors)][::-1]
            #                 o_inds += [b.data.upper_ind_id.format(i) for i in range(b.data.num_tensors)][::-1]
            #
            #         tmp = qtn.TensorNetwork([gtn.spine] + branch_tensors)
            #         tmp.exponent = gtn.spine.exponent + gtn.exponent
            #         tensor = tmp.contract()
            #         tensor.modify(apply=lambda x: x * 10 ** tmp.exponent * gtn.sign)
            #
            #     if is_mps:
            #         tensor.transpose(*out_inds, inplace=True)
            #     else:
            #         tensor.transpose(*o_inds, *i_inds, inplace=True)
            #
            #     return tensor
            #
            # tensor = get_data(F1, is_mps=True)
            # # print('F1 norm', tensor.norm())
            # tensor_data = tensor.data.reshape(-1)
            # npts = len(tensor_data)
            #
            # imp_tensor = get_data(imp_gtn, is_mps=False)
            # # print('imp norm', imp_tensor.norm())
            # imp_tensor_data = imp_tensor.data.reshape(npts,npts)
            #
            # new_F1_data = np.linalg.solve(imp_tensor_data, tensor_data)
            # new_F1_data = new_F1_data.reshape(6, -1)
            # # print('new F1 norm', np.linalg.norm(new_F1_data))
            #
            # # plt.figure()
            # # plt.plot(new_F1_data.T, '--', label=list(range(6)))
            # # plt.legend()
            # # plt.title('new F1')
            # # plt.show()
            #
            # ## reset fields
            # for i in range(3):
            #     compID = new_sys.coords_x.type_coords[i]
            #
            #     ref_EC = new_sys.E.components.get(compID, None)
            #     ax_deriv_configs = ref_EC.ax_deriv_configs if ref_EC is not None else None
            #     data_ = new_F1_data[i, :].reshape(*[ax.npts for ax in new_sys.grid_X.axes])
            #     EC = new_sys.grid_X.map_state_to_mps(data_, ax_deriv_configs=ax_deriv_configs)
            #     if EC is not None: # and compID in new_sys.E.componentIDs:
            #         new_sys.E.components[compID] = EC
            #
            #     ref_BC = new_sys.B.components.get(compID, None)
            #     ax_deriv_configs = ref_BC.ax_deriv_configs if ref_BC is not None else None
            #     data_ = new_F1_data[3 + i, :].reshape(*[ax.npts for ax in new_sys.grid_X.axes])
            #     BC = new_sys.grid_X.map_state_to_mps(data_, ax_deriv_configs=ax_deriv_configs)
            #     if BC is not None: # and compID in new_sys.B.componentIDs:
            #         new_sys.B.components[compID] = BC
            #
            # # plt.figure()
            # # for i in range(3):
            # #     compID = new_sys.coords_x.type_coords[i]
            # #     EC = new_sys.E.get_comp_data(compID)
            # #     if EC is not None:
            # #         plt.plot(EC, label=f'E{compID}')
            # # for i in range(3):
            # #     compID = new_sys.coords_x.type_coords[i]
            # #     BC = new_sys.B.get_comp_data(compID)
            # #     if BC is not None:
            # #         plt.plot(BC, label=f'B{compID}')
            # # plt.legend()
            # # plt.show()
            #
            # # for i in range(3):
            # #     compID = new_sys.coords_x.type_coords[i]
            # #     EC = new_sys.E.get_comp_data(compID)
            # #     if EC is not None:
            # #         plt.figure()
            # #         plt.imshow(EC)
            # #         plt.title(f'E{compID}')
            # #         plt.colorbar()
            # #
            # # for i in range(3):
            # #     compID = new_sys.coords_x.type_coords[i]
            # #     BC = new_sys.B.get_comp_data(compID)
            # #     if BC is not None:
            # #         plt.figure()
            # #         plt.imshow(BC)
            # #         plt.title(f'B{compID}')
            # #         plt.colorbar()
            # #
            # # plt.show()
            #
            # # print('new sys E', new_sys.E.norms())
            # # print('new sys B', new_sys.B.norms())
            # # exit()

        else:  ## solve iteratively until convergence

            ## only the curl(B), curl(E) components of dE/dt, dB/dt
            tmp_sys = self.copy()
            tmp_sys.current_density = None
            dEdt = tmp_sys._calculate_time_derivative_E(compress=compress_level + 3)
            dBdt = tmp_sys._calculate_time_derivative_B(compress=compress_level + 3)

            ## contributions from E, B fields at time step n + 1
            E1 = dEdt.scalar_multiply(0.5 * dt, inplace=False)  # 0.5 * curl(B)
            E1 = E1.add(tmp_sys.E, compress_level=0)
            if j is not None:
                j = j.scalar_multiply(dt, inplace=False)
            E1 = E1.add(j, inplace=True, compress_level=compress_level + 1)

            B1 = dBdt.scalar_multiply(0.5 * dt, inplace=False)
            B1 = B1.add(tmp_sys.B, compress_level=compress_level + 1)

            ## contributions from E, B fields at time step n + 1. initial guess is fields at time n
            E2 = dEdt.scalar_multiply(0.5 * dt)
            B2 = dBdt.scalar_multiply(0.5 * dt)

            it, err = 0, np.inf
            while err > err_tol and it < max_iter:
                old_sys = tmp_sys.copy()

                ## solve for fields at time step n + 1, given current guess of other fields
                for C in tmp_sys.coords_x.coords:
                    E_comp = tmp_sys.E.components.get(C, None)
                    E1_comp = E1.components.get(C, None)
                    E2_comp = E2.components.get(C, None)

                    if E1_comp is not None:
                        new_E_comp = E1_comp.add(E2_comp)
                    else:
                        new_E_comp = E2_comp.copy()

                    if E2_comp is not None:
                        E_comp.data = new_E_comp.data

                ## update guess for dBdt with new E fields at time n + 1
                dBdt = tmp_sys._calculate_time_derivative_B(compress=compress_level + 3)
                B2 = dBdt.scalar_multiply(0.5 * dt)

                for C in tmp_sys.coords_x.coords:
                    B_comp = tmp_sys.B.components.get(C, None)
                    B1_comp = B1.components.get(C, None)
                    B2_comp = B2.components.get(C, None)

                    if B1_comp is not None:
                        new_B_comp = B1_comp.add(B2_comp)
                    else:
                        new_B_comp = B2_comp.copy()

                    if B2_comp is not None:
                        B_comp.data = new_B_comp.data

                ## update guess for dEdt with new B fields at time n + 1
                dEdt = tmp_sys._calculate_time_derivative_E(compress=compress_level + 3)
                E2 = dEdt.scalar_multiply(0.5 * dt)

                err = tmp_sys.distances(old_sys, total=True)
                print('err', it, err)
                it += 1

                # errs = new_sys.distances(old_sys)
                # print('errs', errs)
                # err = np.sum([val for k, val in errs.items()])

            new_sys.E._components = tmp_sys.E.components
            new_sys.B._components = tmp_sys.B.components

        return new_sys

    # def crank_nicolson_block(self, dt, time: float = None, use_A2=True, inplace=False, weight=0.5,
    #                          err_tol: float = np.sqrt(CUTOFF), max_iter: int = 100,
    #                          compress_level: int = 1, verbose_plot=False, **deriv_kwargs):
    #     """
    #     ## dE/dt = c^2 * curl(B) - 1/eps0 * J [SI];  dE/dt = c * curl(B) - 4pi J [CGS]
    #     ## dB/dt = - curl(E) [SI];  dB/dt = - c curl(E) [CGS]
    #
    #     | 1/dt * I, -s/2 curl | |E^(n+1)| = | 1/dt * I, s/2 curl | |E^(n)|  - |J^(n+1/2)|
    #     |-s/2 curl,  1/dt * I | |B^(n+1)| = | s/2 curl, 1/dt * I | |B^(n)|  + |    0    |
    #     where s is a sign and coefficient as defined by Maxwell's eq
    #
    #     | 1/dt * I, -w s curl | |E^(n+1)| = | 1/dt * I, (1-w) s curl | |E^(n)|  - |J^(n+1/2)|
    #     |-w s curl,  1/dt * I | |B^(n+1)| = | (1-w) s curl, 1/dt * I | |B^(n)|  + |    0    |
    #     where s is a sign and coefficient as defined by Maxwell's eq
    #     where w = weight (default is 0.5)
    #
    #     | 1/dt * I, -s/2 curl | |E^(n+1)| = | 1/dt * I, s/2 curl | |E^(n)|  + |c^2 curl(B0)| - |J^(n+1/2)|
    #     |-s/2 curl,  1/dt * I | |B^(n+1)| = | s/2 curl, 1/dt * I | |B^(n)|  - |curl(E0)| + |    0    |
    #     where s is a sign and coefficient as defined by Maxwell's eq
    #     """
    #     new_sys = self if inplace else self.copy()
    #     try_adaptive_solve = False  # True  # False
    #     use_div_constraints = False
    #
    #     ## current, theoretically at time n + 1/2
    #     j = self.current_density
    #     if j is not None:
    #         j = j.copy()
    #         if self.matl_params.is_cgs:
    #             j.scalar_multiply(-4 * np.pi, inplace=True)
    #         else:
    #             j.scalar_multiply(-1 / self.matl_params.eps0, inplace=True)
    #
    #     ### dmrg method
    #     print("HERE, EM USE BLOCK DMRG")
    #
    #     all_fields_dict = self.get_state_dict()
    #     all_fields_time_deriv_dict = self.get_derivative_dict(dt)
    #
    #     def add_gtn_list(list_gtn: Sequence['GridTN'], compress=1, **compress_opts):
    #         sum_out = list_gtn[0].copy()
    #         for gtn in list_gtn[1:]:
    #             sum_out.add(gtn, inplace=True)
    #         if compress:
    #             sum_out.compress(**compress_opts)
    #         return sum_out
    #
    #     def apply_dict(fields_dict: dict[int, 'GridTN'], ops_dict: dict[tuple[int, int], 'GridTN'],
    #                    compress=1, **compress_opts) -> dict[int, 'GridTN']:
    #
    #         outputs = {out: [] for out in range(6)}
    #         for k in ops_dict.keys():
    #             out_ind, in_ind = k
    #             op = ops_dict[k]
    #             vec = fields_dict.get(in_ind, None)
    #             if vec is not None and vec.data is not None:
    #                 outputs[out_ind] += [vec.apply(op, inplace=False)]
    #
    #         output_gtns = {}
    #         for out, out_vecs in outputs.items():
    #             if len(out_vecs) == 0:
    #                 # output_gtns[out] = None  # self.grid_X.get_ones_mps()
    #                 continue
    #             output_gtns[out] = add_gtn_list(out_vecs, compress=compress, **compress_opts)
    #         return output_gtns
    #
    #     def combine_field_dicts(*fields_dicts, add_gtns=False):
    #         combined_dict = {}
    #         for fdict in fields_dicts:
    #             for k, gtn in fdict.items():
    #                 if k not in combined_dict:
    #                     combined_dict[k] = [gtn]
    #                 else:
    #                     if add_gtns:
    #                         combined_dict[k] = [combined_dict[k][-1].add(gtn)]
    #                     else:
    #                         combined_dict[k] += [gtn]
    #                     # combined_dict[k] = [combined_dict[k][0].add(gtn, compress=5, inplace=False)]
    #         return combined_dict
    #
    #     def convert_dict_to_dmrg(gtn_dict, is_mps=True):
    #         converted_dict = {}
    #         for k, gtns in gtn_dict.items():
    #             if isinstance(gtns, (list, tuple)):
    #                 dmrg_vecs = [self.grid_X.gtn_to_dmrg_format(gtn, is_mps=is_mps) for gtn in gtns]
    #                 # for tmp in dmrg_vecs:
    #                 #     tmp.distribute_exponent()
    #             else:
    #                 dmrg_vecs = self.grid_X.gtn_to_dmrg_format(gtns, is_mps=is_mps)
    #                 # dmrg_vecs.distribute_exponent()
    #             converted_dict[k] = dmrg_vecs
    #         return converted_dict
    #
    #     compress_opts = new_sys.E.compress_config.get_compress_opts(1).copy()
    #     max_bond = compress_opts.get('max_bond', None)
    #     if max_bond is not None:
    #         compress_opts['max_bond'] = max_bond  # * num_active_fields
    #
    #     jF_dict = {}
    #     if j is not None:
    #         if self.upwind:
    #             j_corrE, j_corrB = self._calculate_upwind_current(compress=compress_level,
    #                                                               compress1=compress_level + 1)
    #             jF_dict = {**{compID.type.value: comp for compID, comp in j_corrE.components.items()},
    #                        **{compID.type.value + 3: comp for compID, comp in j_corrB.components.items()}}
    #         else:
    #             print('not current upwind')
    #             jF_dict = {compID.type.value: comp for compID, comp in j.components.items()}
    #             # jF = self.grid_X.build_indexed_gtn({compID.type.value: comp for compID, comp in j.components.items()},
    #             #                                    index_order=list(range(6)))
    #
    #     # curlF_bg = None
    #     background_comps = {}
    #     if self.background_E0 is not None:
    #         if self.curlE0 is None:
    #             self.curlE0 = self.background_E0.curl(self.coords_x, compress_level=5)
    #         scalar_E = -self.matl_params.c if self.matl_params.is_cgs else -1
    #         back_E_comps = {compID.type.value + 3: comp.scalar_multiply(scalar_E)
    #                         for compID, comp in self.curlE0.components.items()}
    #         background_comps.update(back_E_comps)
    #     if self.background_B0 is not None:
    #         if self.curlB0 is None:
    #             print('calculating curlB0')
    #             self.curlB0 = self.background_B0.curl(self.coords_x, compress_level=5)
    #         scalar_B = self.matl_params.c if self.matl_params.is_cgs else self.matl_params.c ** 2
    #         back_B_comps = {compID.type.value: comp.scalar_multiply(scalar_B, inplace=False)
    #                         for compID, comp in self.curlB0.components.items()}
    #         background_comps.update(back_B_comps)
    #
    #     if use_div_constraints:
    #         div_mpos, div_targets = self.get_constraint_dict(weight=0.1)
    #     else:
    #         div_mpos, div_targets = {}, {}
    #
    #     def _implicit_solver(dt_, fields_dict, verbose_output=False, **solver_kwargs):
    #
    #         jF_dt = {}
    #         if len(jF_dict) > 0:
    #             jF_dt = {k: mpo.scalar_multiply(dt_, inplace=False) if mpo is not None else None
    #                      for k, mpo in jF_dict.items()}
    #         bgF_dt = {}
    #         if len(background_comps) > 0:
    #             bgF_dt = {k: mpo.scalar_multiply(dt_, inplace=False) if mpo is not None else None
    #                       for k, mpo in background_comps.items()}
    #
    #         #### explicit contribution ####
    #         dFdt1 = apply_dict(fields_dict, all_fields_time_deriv_dict, compress=compress_level)
    #         for k, gtn in dFdt1.items():
    #             dFdt1[k] = gtn.scalar_multiply((1. - weight) * dt_, inplace=False)
    #             # dFdt1[k] = gtn.scalar_multiply(0.5 * dt_, inplace=False)
    #
    #         explicit_part = combine_field_dicts(fields_dict, dFdt1, jF_dt, bgF_dt, div_targets)
    #
    #         #### implicit contribution operator
    #         dFdt_imp_dict = {k: gtn.scalar_multiply(-weight * dt_, inplace=False)
    #                          for k, gtn in all_fields_time_deriv_dict.items()}
    #         # dFdt_imp_dict = {k: gtn.scalar_multiply(-0.5 * dt_, inplace=False)
    #         # for k, gtn in all_fields_time_deriv_dict.items()}
    #         iden_dict = {
    #             (k, k): self.grid_X.get_iden_mpo(upper_ind_id='o({})', lower_ind_id='i({})', site_tag_id='ID({})')
    #             for k in range(6)}
    #         # implicit_op = {**dFdt_imp_dict, **iden_dict}
    #         implicit_op = combine_field_dicts(dFdt_imp_dict, iden_dict, div_mpos, add_gtns=True)
    #         # dFdt_gtn = all_fields_time_deriv.scalar_multiply(-0.5 * dt_, inplace=False)
    #         # iden_gtn = dFdt_gtn.get_like_iden()
    #         # imp_gtn = dFdt_gtn.add(iden_gtn)
    #         # print('EM implicit op', imp_gtn.max_bond())
    #
    #         ## dmrg solve
    #         explicit_part_mps = convert_dict_to_dmrg(explicit_part, is_mps=True)
    #         implicit_part_mpo = convert_dict_to_dmrg(implicit_op, is_mps=False)
    #         # print('explicit exp', [[m.exponent for m in ms] for k,ms in explicit_part_mps.items()])
    #         # print('implicit exp', [[m.exponent for m in ms] for k,ms in implicit_part_mpo.items()])
    #
    #         # # init_guess = convert_dict_to_dmrg(fields_dict, is_mps=True)
    #         # ## as with full DMRG method, init_guess is explicit contribution? but seems less good...
    #         # ## random initial state does better...
    #         # init_guess = {k: helper.sum_list(*exp_part, compress=True, compress_opts=compress_opts)
    #         #               for k, exp_part in explicit_part_mps.items()}
    #         # for k, mps in init_guess.items():
    #         #     if mps is not None:
    #         #         rand_mps = qtn.MPS_rand_state(mps.L, mps.max_bond(), mps.phys_dim(1))
    #         #         helper.scalar_multiply(rand_mps, 0.01, inplace=True)
    #         #         init_guess[k] = helper.add_MPS(mps, rand_mps, inplace=True, compress_opts=compress_opts)
    #         #         # init_guess[k] = helper.add_rand_noise(mps, strength=0.01, inplace=False)
    #
    #         print('implicit block dmrg solve', compress_opts)
    #         if len(all_fields_dict) == 0:
    #             outputs = {}
    #             for k, mps_list in explicit_part.items():
    #                 tot_mps = None
    #                 for gtn_mps in mps_list:
    #                     tot_mps = helper.add_MPS(tot_mps, gtn_mps.data, compress=False)
    #                 helper.compress(tot_mps, compress_opts)
    #                 outputs[k] = tot_mps
    #         else:
    #             if use_A2:
    #                 outputs, err, is_conv = helper_block_2.implicit_solver_2(6, implicit_part_mpo, explicit_part_mps,
    #                                                                          # init_guess=init_guess,
    #                                                                          # init_direction=SweepDirection.LEFT,
    #                                                                          verbose_output=True,
    #                                                                          max_bond=max_bond, **solver_kwargs)
    #             else:
    #                 outputs, err, is_conv = helper_block.implicit_solver(6, implicit_part_mpo, explicit_part_mps,
    #                                                                      # init_guess=init_guess,
    #                                                                      is_H=False,
    #                                                                      verbose_output=True,
    #                                                                      solve_type=SolveMethod.CGD,
    #                                                                      max_bond=max_bond, **solver_kwargs)
    #
    #         out_gtns = {}
    #         try:
    #             ref_gtn = fields_dict[next(iter(fields_dict))]
    #         except StopIteration:
    #             try:
    #                 ref_gtn = new_sys.E.components[next(iter(new_sys.E.components))]
    #             except StopIteration:
    #                 ref_gtn = new_sys.B.components[next(iter(new_sys.B.components))]
    #
    #         for k, out_mps in outputs.items():
    #             out_gtns[k] = self.grid_X.dmrg_to_gtn_format(ref_gtn.copy(), out_mps, is_mps=True)
    #
    #         # for k in outputs.keys():
    #         #     out_mps = outputs[k]
    #         #     x_tens = out_mps.contract(all) * 10 ** out_mps.exponent
    #         #     # x_tens.transpose(*[out_mps.site_ind_id.format(i) for i in range(out_mps.L)], inplace=True)
    #         #     x_tens.transpose(*[out_mps.site_ind_id.format(i) for i in range(out_mps.L - 1, -1, -1)], inplace=True)
    #         #     x_vec = x_tens.data.reshape(-1)
    #         #
    #         #     gtn_data = out_gtns[k].get_data()
    #         #
    #         #     plt.figure()
    #         #     plt.plot(x_vec)
    #         #     plt.plot(gtn_data, '--')
    #         #
    #         #     plt.figure()
    #         #     plt.plot(np.abs(x_vec - gtn_data))
    #         #     plt.show()
    #
    #         if verbose_output:
    #             return out_gtns, err, is_conv
    #         return out_gtns
    #
    #     if all_fields_dict is None and len(jF_dict) == 0 and len(background_comps) == 0:
    #         return new_sys
    #     # elif all_fields_dict is None and (len(jF_dict) > 0 or len(background_comps) > 0):
    #     #     sourceF_dt = combine_field_dicts(jF_dict, background_comps)
    #     #     # sourceF.scalar_multiply(dt, inplace=False)
    #     #     # new_F1 = sourceF_dt
    #     else:
    #         if try_adaptive_solve:
    #             nt, max_nt, is_conv = self.nt, 4, False
    #             old_err = 100
    #
    #             old_fields_dict = all_fields_dict
    #             while (not is_conv) and nt <= max_nt:
    #                 print('nt', nt)
    #                 new_F1 = old_fields_dict
    #                 for step in range(nt):
    #                     new_F1, err, is_conv = _implicit_solver(dt / nt, new_F1, verbose_output=True, )
    #                     if nt < max_nt and (not is_conv) and err > 1.0e-3:
    #                         if err < old_err * 0.8:
    #                             nt = min(nt + 1, max_nt)
    #                             old_err = err
    #                             break
    #
    #                     is_conv = (step == nt - 1)  ## reached the last step
    #
    #             self.nt = nt
    #         else:
    #             new_F1 = _implicit_solver(dt, all_fields_dict)
    #
    #     # reset fields
    #     for i in range(3):
    #         compID = new_sys.coords_x.type_coords[i]
    #
    #         ref_EC = new_sys.E.components.get(compID, None)
    #         ax_deriv_configs = ref_EC.ax_deriv_configs if ref_EC is not None else {}
    #         # EC = new_sys.grid_X.select_indexed_gtn(new_F1, i)
    #         EC = new_F1.get(i, self.grid_X.make_empty_gridTN())
    #         EC.ax_deriv_configs = ax_deriv_configs
    #         if not (EC is None and compID not in new_sys.E.componentIDs):
    #             new_sys.E.components[compID] = EC
    #             # print('EC', compID, EC.max_bond(), EC.canon_site)
    #
    #         ref_BC = new_sys.B.components.get(compID, None)
    #         ax_deriv_configs = ref_BC.ax_deriv_configs if ref_BC is not None else {}
    #         # BC = new_sys.grid_X.select_indexed_gtn(new_F1, 3 + i)
    #         BC = new_F1.get(i + 3, self.grid_X.make_empty_gridTN())
    #         BC.ax_deriv_configs = ax_deriv_configs
    #         if not (BC is None and compID not in new_sys.B.componentIDs):
    #             new_sys.B.components[compID] = BC
    #             # print('BC', compID, BC.max_bond(), BC.canon_site)
    #
    #     new_sys.E.compress(inplace=True, compress_level=1)  # , norm_cutoff=1.0e-8)
    #     new_sys.B.compress(inplace=True, compress_level=1)  # , norm_cutoff=1.0e-8)
    #
    #     # for C, EC in new_sys.E.components.items():
    #     #     print('EC', C, EC.ax_deriv_configs)
    #     # for C, EC in new_sys.B.components.items():
    #     #     print('BC', C, EC.ax_deriv_configs)
    #     #
    #     #
    #     # plt.figure()
    #     # for i in range(3):
    #     #     compID = new_sys.coords_x.type_coords[i]
    #     #     EC = new_sys.E.get_comp_data(compID)
    #     #     if EC is not None:
    #     #         plt.plot(EC, label=f'E{compID}')
    #     # for i in range(3):
    #     #     compID = new_sys.coords_x.type_coords[i]
    #     #     BC = new_sys.B.get_comp_data(compID)
    #     #     if BC is not None:
    #     #         plt.plot(BC, label=f'B{compID}')
    #     # plt.legend()
    #
    #     # ax_x, ax_y = new_sys.grid_X.axes
    #     # for i in range(3):
    #     #     compID = new_sys.coords_x.type_coords[i]
    #     #     EC = new_sys.E.get_comp_data(compID)
    #     #     EC = ax_x.basis.get_realspace_1D(EC, 0)
    #     #     EC = ax_y.basis.get_realspace_1D(EC, 1)
    #     #     if EC is not None:
    #     #         plt.figure()
    #     #         plt.imshow(np.real(EC))
    #     #         plt.title(f'E{compID}')
    #     #         plt.colorbar()
    #     # for i in range(3):
    #     #     compID = new_sys.coords_x.type_coords[i]
    #     #     BC = new_sys.B.get_comp_data(compID)
    #     #     BC = ax_x.basis.get_realspace_1D(BC, 0)
    #     #     BC = ax_y.basis.get_realspace_1D(BC, 1)
    #     #     if BC is not None:
    #     #         plt.figure()
    #     #         plt.imshow(np.real(BC))
    #     #         plt.title(f'B{compID}')
    #     #         plt.colorbar()
    #     # plt.show()
    #
    #     # print('E')
    #     # print(new_sys.E.components)
    #     # print('B')
    #     # print(new_sys.B.components)
    #     # exit()
    #
    #     return new_sys

    def tddmrg(self, dt, time: float = None, use_A2=True, inplace=False, te_order=4,
               err_tol: float = np.sqrt(CUTOFF), max_iter: int = 100,
               compress_level: int = 1, verbose_plot=False, **deriv_kwargs):
        """
        ## dE/dt = c^2 * curl(B) - 1/eps0 * J [SI];  dE/dt = c * curl(B) - 4pi J [CGS]
        ## dB/dt = - curl(E) [SI];  dB/dt = - c curl(E) [CGS]

        | 1/dt * I, -s/2 curl | |E^(n+1)| = | 1/dt * I, s/2 curl | |E^(n)|  - |J^(n+1/2)|
        |-s/2 curl,  1/dt * I | |B^(n+1)| = | s/2 curl, 1/dt * I | |B^(n)|  + |    0    |
        where s is a sign and coefficient as defined by Maxwell's eq

        | 1/dt * I, -s/2 curl | |E^(n+1)| = | 1/dt * I, s/2 curl | |E^(n)|  + |c^2 curl(B0)| - |J^(n+1/2)|
        |-s/2 curl,  1/dt * I | |B^(n+1)| = | s/2 curl, 1/dt * I | |B^(n)|  - |curl(E0)| + |    0    |
        where s is a sign and coefficient as defined by Maxwell's eq
        """
        # use_dmrg_new = True  # False
        # if use_dmrg_new:
        #     return self.crank_nicolson_dmrg(dt, time=time, err_tol=err_tol, inplace=inplace, max_iter=max_iter,
        #                                     compress_level=compress_level, verbose_plot=verbose_plot, **deriv_kwargs)

        new_sys = self if inplace else self.copy()
        use_dmrg = True
        try_adaptive_solve = False

        ## current, theoretically at time n + 1/2
        j = self.current_density
        if j is not None:
            j = j.copy()
            if self.matl_params.is_cgs:
                j.scalar_multiply(-4 * np.pi, inplace=True)
            else:
                j.scalar_multiply(-1 / self.matl_params.eps0, inplace=True)

        ### dmrg method
        if use_dmrg:
            print("HERE, EM USE DMRG")

            num_active_fields = self.get_num_active_fields()
            all_fields_vec = self.get_combined_state()
            all_fields_time_deriv = self.get_combined_derivative_mpo()

            compress_opts = new_sys.E.compress_config.get_compress_opts(1).copy()
            max_bond = compress_opts.get('max_bond', None)
            if max_bond is not None:
                compress_opts['max_bond'] = max_bond * num_active_fields

            jF = None
            if j is not None:
                if self.upwind:
                    j_corrE, j_corrB = self._calculate_upwind_current(compress=compress_level,
                                                                      compress1=compress_level + 1)
                    jF = self.grid_X.build_indexed_gtn(
                        {**{compID.type.value: comp for compID, comp in j_corrE.components.items()},
                         **{compID.type.value + 3: comp for compID, comp in j_corrB.components.items()}},
                        index_order=list(range(6)))
                else:
                    jF = self.grid_X.build_indexed_gtn(
                        {compID.type.value: comp for compID, comp in j.components.items()},
                        index_order=list(range(6)))

            curlF_bg = None
            background_comps = {}
            if self.background_E0 is not None:
                if self.curlE0 is None:
                    self.curlE0 = self.background_E0.curl(self.coords_x, compress_level=5)
                scalar_E = -self.matl_params.c if self.matl_params.is_cgs else -1
                back_E_comps = {compID.type.value + 3: comp.scalar_multiply(scalar_E)
                                for compID, comp in self.curlE0.components.items()}
                background_comps.update(back_E_comps)
            if self.background_B0 is not None:
                if self.curlB0 is None:
                    print('calculating curlB0')
                    self.curlB0 = self.background_B0.curl(self.coords_x, compress_level=5)
                scalar_B = self.matl_params.c if self.matl_params.is_cgs else self.matl_params.c ** 2
                back_B_comps = {compID.type.value: comp.scalar_multiply(scalar_B, inplace=False)
                                for compID, comp in self.curlB0.components.items()}
                background_comps.update(back_B_comps)

            if len(background_comps) > 0:
                curlF_bg = self.grid_X.build_indexed_gtn(background_comps, index_order=list(range(6)))

            sourceF = jF
            if curlF_bg is not None:
                sourceF = curlF_bg.add(sourceF, compress=5, inplace=False)
                # print('sourceF norm', sourceF.frobenius_norm())

            ## td-dmrg. only works for TT geometry
            from helper_tdvp_v2 import TDDMRGSolver_v2 as tdsolver

            compress_config = CompressionConfiguration()
            compress_config.set_compress_opts(1, max_bond)  # max_bond=compress_opts['max_bond'])
            all_fields_time_deriv.data.mangle_inner_(append='_o')
            print('compress_config', compress_config)

            # sourceF.data.distribute_exponent()
            # all_fields_vec.data.distribute_exponent()
            # all_fields_time_deriv.data.distribute_exponent()

            # dmrg_vecs = new_sys.grid_X.gtn_to_dmrg_format(all_fields_vec, is_mps=True)
            tdvp_solver = tdsolver(all_fields_vec.data, targets=[sourceF.data], operators=[all_fields_time_deriv.data],
                                   te_order=te_order, compress_config=compress_config)
            tdvp_solver.take_time_step(dt, do_adapt=False)
            new_F1 = all_fields_vec.create_like(tdvp_solver.ket)

            # reset fields
            for i in range(3):
                compID = new_sys.coords_x.type_coords[i]

                ref_EC = new_sys.E.components.get(compID, None)
                ax_deriv_configs = ref_EC.ax_deriv_configs if ref_EC is not None else {}
                EC = new_sys.grid_X.select_indexed_gtn(new_F1, i)
                EC.ax_deriv_configs = ax_deriv_configs
                if not (EC is None and compID not in new_sys.E.componentIDs):
                    new_sys.E.components[compID] = EC
                    # print('EC', compID, EC.max_bond(), EC.canon_site)

                ref_BC = new_sys.B.components.get(compID, None)
                ax_deriv_configs = ref_BC.ax_deriv_configs if ref_BC is not None else {}
                BC = new_sys.grid_X.select_indexed_gtn(new_F1, 3 + i)
                BC.ax_deriv_configs = ax_deriv_configs
                if not (BC is None and compID not in new_sys.E.componentIDs):
                    new_sys.B.components[compID] = BC
                    # print('BC', compID, BC.max_bond(), BC.canon_site)

            new_sys.E.compress(inplace=True, compress_level=1, norm_cutoff=1.0e-8)
            new_sys.B.compress(inplace=True, compress_level=1, norm_cutoff=1.0e-8)

            # new_sys.E._components = self.E.components
            # new_sys.B._components = self.B.components

        return new_sys

    def block_tddmrg(self, dt, te_order=4, time: float = None, use_A2=False, inplace=False, weight=0.5,
                     err_tol: float = np.sqrt(CUTOFF), max_iter: int = 100,
                     compress_level: int = 1, verbose_plot=False, apply_constraints=False,
                     **deriv_kwargs):
        """
        ## dE/dt = c^2 * curl(B) - 1/eps0 * J [SI];  dE/dt = c * curl(B) - 4pi J [CGS]
        ## dB/dt = - curl(E) [SI];  dB/dt = - c curl(E) [CGS]

        | 1/dt * I, -s/2 curl | |E^(n+1)| = | 1/dt * I, s/2 curl | |E^(n)|  + |c^2 curl(B0)| - |J^(n+1/2)|
        |-s/2 curl,  1/dt * I | |B^(n+1)| = | s/2 curl, 1/dt * I | |B^(n)|  - |curl(E0)|     + |    0    |
        ## d/dt[ EB ] = A [EB] + b
        """

        # if apply_constraints:
        #     return self.block_tddmrg_cleaning(dt, te_order=te_order, time=time, use_A2=use_A2, inplace=inplace,
        #                                       weight=weight, err_tol=err_tol, max_iter=max_iter,
        #                                       compress_level = compress_level, verbose_plot=verbose_plot, **deriv_kwargs)

        # if self.upwind:
        #     print('split step block tddmrg for upwind')
        #     return self.block_tddmrg_upwind(dt, te_order=te_order, time=time, use_A2=use_A2, inplace=inplace,
        #                                     weight=weight, err_tol=err_tol, max_iter=max_iter,
        #                                     compress_level = compress_level, verbose_plot=verbose_plot, **deriv_kwargs)

        new_sys = self if inplace else self.copy()
        # try_adaptive_solve = True  # False

        ## current, theoretically at time n + 1/2
        j = self.current_density
        if j is not None:
            j = j.copy()
            if self.matl_params.is_cgs:
                j.scalar_multiply(-4 * np.pi, inplace=True)
            else:
                j.scalar_multiply(-1 / self.matl_params.eps0, inplace=True)

        ### dmrg method
        print("HERE, EM USE BLOCK TDDMRG")

        ## current state
        all_fields_dict = self.get_state_dict()
        ## time evolution operator
        all_fields_time_deriv_dict = self.get_derivative_dict(dt=dt)
        for oo, ii in all_fields_time_deriv_dict.keys():
            if ii in all_fields_dict and oo not in all_fields_dict:
                zero_mps = self.grid_X.make_zero_gridTN()
                all_fields_dict[oo] = zero_mps

        def add_gtn_list(list_gtn: Sequence['GridTN'], compress=1, **compress_opts):
            sum_out = list_gtn[0].copy()
            for gtn in list_gtn[1:]:
                sum_out.add(gtn, inplace=True)
            if compress:
                sum_out.compress(**compress_opts)
            return sum_out

        def apply_dict(fields_dict: dict[int, 'GridTN'], ops_dict: dict[tuple[int, int], 'GridTN'],
                       compress=1, **compress_opts) -> dict[int, 'GridTN']:

            outputs = {out: [] for out in range(6)}
            for k in ops_dict.keys():
                out_ind, in_ind = k
                op = ops_dict[k]
                vec = fields_dict.get(in_ind, None)
                if vec is not None and vec.data is not None:
                    outputs[out_ind] += [vec.apply(op, inplace=False)]

            output_gtns = {}
            for out, out_vecs in outputs.items():
                if len(out_vecs) == 0:
                    # output_gtns[out] = None  # self.grid_X.get_ones_mps()
                    continue
                output_gtns[out] = add_gtn_list(out_vecs, compress=compress, **compress_opts)
            return output_gtns

        def combine_field_dicts(*fields_dicts):
            combined_dict = {}
            for fdict in fields_dicts:
                for k, gtn in fdict.items():
                    if k not in combined_dict:
                        combined_dict[k] = [gtn]
                    else:
                        combined_dict[k] += [gtn]
            return combined_dict

        def convert_dict_to_dmrg(gtn_dict, is_mps=True):
            converted_dict = {}
            for k, gtns in gtn_dict.items():
                if isinstance(gtns, (list, tuple)):
                    dmrg_vecs = [self.grid_X.gtn_to_dmrg_format(gtn, is_mps=is_mps) for gtn in gtns]
                    # for tmp in dmrg_vecs:
                    #     tmp.distribute_exponent()
                else:
                    dmrg_vecs = self.grid_X.gtn_to_dmrg_format(gtns, is_mps=is_mps)
                    # dmrg_vecs.distribute_exponent()
                converted_dict[k] = dmrg_vecs
            return converted_dict

        compress_opts = new_sys.E.compress_config.get_compress_opts(1).copy()
        max_bond = compress_opts.get('max_bond', None)
        if max_bond is not None:
            compress_opts['max_bond'] = max_bond  # * num_active_fields

        ## current
        jF_dict = {}
        if j is not None:
            if self.upwind:
                j_corrE, j_corrB = self._calculate_upwind_current(compress=compress_level,
                                                                  compress1=compress_level + 1)
                jF_dict = {**{compID.type.value: comp for compID, comp in j_corrE.components.items()},
                           **{compID.type.value + 3: comp for compID, comp in j_corrB.components.items()}}
            else:
                jF_dict = {compID.type.value: comp for compID, comp in j.components.items()}
                # jF = self.grid_X.build_indexed_gtn({compID.type.value: comp for compID, comp in j.components.items()},
                #                                    index_order=list(range(6)))

        ## background field components
        background_comps = {}
        if self.background_E0 is not None:
            if self.curlE0 is None:
                self.curlE0 = self.background_E0.curl(self.coords_x, compress_level=5)
            scalar_E = -self.matl_params.c if self.matl_params.is_cgs else -1
            back_E_comps = {compID.type.value + 3: comp.scalar_multiply(scalar_E)
                            for compID, comp in self.curlE0.components.items()}
            background_comps.update(back_E_comps)
        if self.background_B0 is not None:
            if self.curlB0 is None:
                print('calculating curlB0')
                self.curlB0 = self.background_B0.curl(self.coords_x, compress_level=5)
            scalar_B = self.matl_params.c if self.matl_params.is_cgs else self.matl_params.c ** 2
            back_B_comps = {compID.type.value: comp.scalar_multiply(scalar_B, inplace=False)
                            for compID, comp in self.curlB0.components.items()}
            background_comps.update(back_B_comps)

        explicit_part = combine_field_dicts(jF_dict, background_comps)
        explicit_part_mps = convert_dict_to_dmrg(explicit_part, is_mps=True)

        if len(all_fields_dict) == 0:
            updated_field_mps = {}
            for k, val in explicit_part_mps.items():
                updated_field_mps[k] = helper.scalar_multiply(val, dt, inplace=False)
        else:
            all_fields_mps = convert_dict_to_dmrg(all_fields_dict, is_mps=True)
            time_deriv_mpo = convert_dict_to_dmrg(all_fields_time_deriv_dict, is_mps=False)

            # for k, out_mps in all_fields_mps.items():
            #     print('mps exp', k, out_mps.exponent)
            #
            # for k, out_mps in explicit_part_mps.items():
            #     print('exp mps exp', k, [out_.exponent for out_ in out_mps])

            time_deriv_mpo_list = {}
            for k, item in time_deriv_mpo.items():
                time_deriv_mpo_list[k] = [item]

            if all_fields_dict is None and len(jF_dict) == 0 and len(background_comps) == 0:
                return new_sys

            site_ind_id = all_fields_mps[next(iter(all_fields_mps))].site_ind_id
            for k, v in all_fields_mps.items():
                if v is not None:
                    v.site_ind_id = site_ind_id
            for k, v in explicit_part_mps.items():
                for x in v:
                    if x is not None:
                        x.site_ind_id = site_ind_id

            # if apply_constraints:
            #     constraints, constraint_vals = self.get_constraints(charge=self.charge_density)
            # else:
            #     constraints, constraint_vals = None, None

            from helper_block_tddmrg_3 import block_tddmrg
            updated_field_mps = block_tddmrg(dt, 6, time_deriv_mpo_list, all_fields_mps, explicit_part_mps,
                                             # constraints=constraints, constraint_vals=constraint_vals,
                                             max_bond=max_bond, te_order=te_order)

        # for k, out_mps in all_fields_mps.items():
        #     print('mps exp', k, out_mps.exponent)

        out_gtns = {}
        ref_gtn = all_fields_dict[next(iter(all_fields_dict))]
        for k, out_mps in updated_field_mps.items():
            # print('updated gtn k', k)
            # print('out mps exp', out_mps.exponent)
            out_gtns[k] = self.grid_X.dmrg_to_gtn_format(ref_gtn.copy(), out_mps, is_mps=True)
        new_F1 = out_gtns

        # reset fields
        for i in range(3):
            compID = new_sys.coords_x.type_coords[i]

            ref_EC = new_sys.E.components.get(compID, None)
            ax_deriv_configs = ref_EC.ax_deriv_configs if ref_EC is not None else {}
            # EC = new_sys.grid_X.select_indexed_gtn(new_F1, i)
            EC = new_F1.get(i, None)
            if not (EC is None and compID not in new_sys.E.componentIDs):
                if EC is None:
                    EC = self.grid_X.make_empty_gridTN(ax_deriv_configs)
                else:
                    EC.ax_deriv_configs = ax_deriv_configs
                new_sys.E.components[compID] = EC
                # print('EC', compID, EC.max_bond(), EC.canon_site)

            ref_BC = new_sys.B.components.get(compID, None)
            ax_deriv_configs = ref_BC.ax_deriv_configs if ref_BC is not None else {}
            # BC = new_sys.grid_X.select_indexed_gtn(new_F1, 3 + i)
            BC = new_F1.get(i + 3, None)
            if not (BC is None and compID not in new_sys.B.componentIDs):
                if BC is None:
                    BC = self.grid_X.make_empty_gridTN(ax_deriv_configs)
                else:
                    BC.ax_deriv_configs = ax_deriv_configs
                new_sys.B.components[compID] = BC
                # print('BC', compID, BC.max_bond(), BC.canon_site)

        new_sys.E.compress(inplace=True, compress_level=1, norm_cutoff=1.0e-8)
        new_sys.B.compress(inplace=True, compress_level=1, norm_cutoff=1.0e-8)

        # for C, EC in new_sys.E.components.items():
        #     print('EC', C, EC.ax_deriv_configs)
        # for C, EC in new_sys.B.components.items():
        #     print('BC', C, EC.ax_deriv_configs)
        #
        #
        # plt.figure()
        # for i in range(3):
        #     compID = new_sys.coords_x.type_coords[i]
        #     EC = new_sys.E.get_comp_data(compID)
        #     if EC is not None:
        #         plt.plot(EC, label=f'E{compID}')
        # for i in range(3):
        #     compID = new_sys.coords_x.type_coords[i]
        #     BC = new_sys.B.get_comp_data(compID)
        #     if BC is not None:
        #         plt.plot(BC, label=f'B{compID}')
        # plt.legend()

        # ax_x, ax_y = new_sys.grid_X.axes
        # for i in range(3):
        #     compID = new_sys.coords_x.type_coords[i]
        #     EC = new_sys.E.get_comp_data(compID)
        #     EC = ax_x.basis.get_realspace_1D(EC, 0)
        #     EC = ax_y.basis.get_realspace_1D(EC, 1)
        #     if EC is not None:
        #         plt.figure()
        #         plt.imshow(np.real(EC))
        #         plt.title(f'E{compID}')
        #         plt.colorbar()
        # for i in range(3):
        #     compID = new_sys.coords_x.type_coords[i]
        #     BC = new_sys.B.get_comp_data(compID)
        #     BC = ax_x.basis.get_realspace_1D(BC, 0)
        #     BC = ax_y.basis.get_realspace_1D(BC, 1)
        #     if BC is not None:
        #         plt.figure()
        #         plt.imshow(np.real(BC))
        #         plt.title(f'B{compID}')
        #         plt.colorbar()
        # plt.show()

        # print('E')
        # print(new_sys.E.components)
        # print('B')
        # print(new_sys.B.components)
        # exit()

        if apply_constraints:
            new_sys.clean_fields()

        return new_sys

    def block_tddmrg_cleaning(self, dt, te_order=4, time: float = None, use_A2=False, inplace=False, weight=0.5,
                              err_tol: float = np.sqrt(CUTOFF), max_iter: int = 100,
                              compress_level: int = 1, verbose_plot=False, apply_constraints=False,
                              chi=1.0, gamma=1.0,
                              **deriv_kwargs):
        """
        y_(n+1) = y_(n) + h/2 ( F(y_(n+1)) + F(y_(n)) )
        y_(n+1) = y_(n) + h ( weight * F(y_(n+1)) + (1-weight) * F(y_(n)) )

        ## dE/dt = c^2 * curl(B) - 1/eps0 * J  - c^2 chi grad \phi [SI] ;
           dE/dt = c * curl(B) - 4pi J - c * chi grad \phi [CGS]
        ## dB/dt = - curl(E) - gamma grad \psi [SI];
           dB/dt = - c curl(E) - gamma grad \psi [CGS]
        ## 1/chi dphi/dt = - div(E) + rho/eps0  [SI]
        ## 1/c^2 / gamma dpsi/dt = -div(B)
        ## p = phi, q = psi

        | 1/dt * I, -s/2 curl,  0,  - 1/2 \gamma grad| |E^(n+1)| = | 1/dt * I,   s/2 curl, 0  , 1/2 \gamma grad| |E^(n)|  + |c^2 curl(B0)| - |J^(n+1/2)|
        |-s/2 curl,  1/dt * I, -1/2 chi grad,   0    | |B^(n+1)| = | s/2 curl,   1/dt * I,   1/2 chi grad,   0 | |B^(n)|  - |curl(E0)|     + |    0    |
        | 1/2 chi div ,   0  ,  1/dt * I    ,   0    | |p^(n+1)| = |-1/2 chi * div,  0     ,  1/dt * I ,   0   | |p^(n)|  + |chi rho/eps0|
        |    0  , 1/2 c^2 gamma div,   0  , 1/dt * I | |q^(n+1)| = |  0, -1/2 c^2 gamma div,     0  , 1/dt * I | |q^(n)|
        ## d/dt[ EB ] = A [EB] + b
        """

        new_sys = self if inplace else self.copy()
        # try_adaptive_solve = True  # False

        ## current, theoretically at time n + 1/2
        j = self.current_density
        if j is not None:
            j = j.copy()
            if self.matl_params.is_cgs:
                j.scalar_multiply(-4 * np.pi, inplace=True)
            else:
                j.scalar_multiply(-1 / self.matl_params.eps0, inplace=True)

        ### dmrg method
        print("HERE, EM USE BLOCK TDDMRG")

        ## current state
        all_fields_dict = self.get_state_dict(cleaning=True)
        ## time evolution operator
        all_fields_time_deriv_dict = self.get_derivative_dict(dt=dt)
        for oo, ii in all_fields_time_deriv_dict.keys():
            if ii in all_fields_dict and oo not in all_fields_dict:
                zero_mps = self.grid_X.make_zero_gridTN()
                all_fields_dict[oo] = zero_mps

        print('chi=', chi, 1 / dt)
        # gamma = chi = 1 / dt
        constraint_mat_dict, constraint_targets = self.get_constraint_dict(chi=chi, gamma=gamma)
        constraint_mat_conj = {}
        for (oo, ii), mpo in constraint_mat_dict.items():
            print('oo', 'ii', oo, ii)
            if oo == 7:
                constraint_mat_conj[(ii, oo)] = mpo.scalar_multiply(1 / self.matl_params.c ** 2)
            elif oo == 6:
                constraint_mat_conj[(ii, oo)] = mpo.scalar_multiply(1 * self.matl_params.c ** 2)  # copy()
            else:
                raise ValueError
        all_fields_time_deriv_dict = {**all_fields_time_deriv_dict, **constraint_mat_dict, **constraint_mat_conj}

        # print('all fields keys', all_fields_time_deriv_dict.keys())

        def add_gtn_list(list_gtn: Sequence['GridTN'], compress=1, **compress_opts):
            sum_out = list_gtn[0].copy()
            for gtn in list_gtn[1:]:
                sum_out.add(gtn, inplace=True)
            if compress:
                sum_out.compress(**compress_opts)
            return sum_out

        def apply_dict(fields_dict: dict[int, 'GridTN'], ops_dict: dict[tuple[int, int], 'GridTN'],
                       compress=1, **compress_opts) -> dict[int, 'GridTN']:

            outputs = {out: [] for out in range(6)}
            for k in ops_dict.keys():
                out_ind, in_ind = k
                op = ops_dict[k]
                vec = fields_dict.get(in_ind, None)
                if vec is not None and vec.data is not None:
                    outputs[out_ind] += [vec.apply(op, inplace=False)]

            output_gtns = {}
            for out, out_vecs in outputs.items():
                if len(out_vecs) == 0:
                    # output_gtns[out] = None  # self.grid_X.get_ones_mps()
                    continue
                output_gtns[out] = add_gtn_list(out_vecs, compress=compress, **compress_opts)
            return output_gtns

        def combine_field_dicts(*fields_dicts):
            combined_dict = {}
            for fdict in fields_dicts:
                for k, gtn in fdict.items():
                    if k not in combined_dict:
                        combined_dict[k] = [gtn]
                    else:
                        combined_dict[k] += [gtn]
            return combined_dict

        def convert_dict_to_dmrg(gtn_dict, is_mps=True):
            converted_dict = {}
            for k, gtns in gtn_dict.items():
                if isinstance(gtns, (list, tuple)):
                    dmrg_vecs = [self.grid_X.gtn_to_dmrg_format(gtn, is_mps=is_mps) for gtn in gtns]
                    # for tmp in dmrg_vecs:
                    #     tmp.distribute_exponent()
                else:
                    dmrg_vecs = self.grid_X.gtn_to_dmrg_format(gtns, is_mps=is_mps)
                    # dmrg_vecs.distribute_exponent()
                converted_dict[k] = dmrg_vecs
            return converted_dict

        compress_opts = new_sys.E.compress_config.get_compress_opts(1).copy()
        max_bond = compress_opts.get('max_bond', None)
        if max_bond is not None:
            compress_opts['max_bond'] = max_bond  # * num_active_fields

        ## current
        jF_dict = {}
        if j is not None:
            if self.upwind:
                j_corrE, j_corrB = self._calculate_upwind_current(compress=compress_level,
                                                                  compress1=compress_level + 1)
                jF_dict = {**{compID.type.value: comp for compID, comp in j_corrE.components.items()},
                           **{compID.type.value + 3: comp for compID, comp in j_corrB.components.items()}}
            else:
                jF_dict = {compID.type.value: comp for compID, comp in j.components.items()}
                # jF = self.grid_X.build_indexed_gtn({compID.type.value: comp for compID, comp in j.components.items()},
                #                                    index_order=list(range(6)))

        ## background field components
        background_comps = {}
        if self.background_E0 is not None:
            if self.curlE0 is None:
                self.curlE0 = self.background_E0.curl(self.coords_x, compress_level=5)
            scalar_E = -self.matl_params.c if self.matl_params.is_cgs else -1
            back_E_comps = {compID.type.value + 3: comp.scalar_multiply(scalar_E)
                            for compID, comp in self.curlE0.components.items()}
            background_comps.update(back_E_comps)
        if self.background_B0 is not None:
            if self.curlB0 is None:
                print('calculating curlB0')
                self.curlB0 = self.background_B0.curl(self.coords_x, compress_level=5)
            scalar_B = self.matl_params.c if self.matl_params.is_cgs else self.matl_params.c ** 2
            back_B_comps = {compID.type.value: comp.scalar_multiply(scalar_B, inplace=False)
                            for compID, comp in self.curlB0.components.items()}
            background_comps.update(back_B_comps)

        explicit_part = combine_field_dicts(jF_dict, background_comps, constraint_targets)
        explicit_part_mps = convert_dict_to_dmrg(explicit_part, is_mps=True)

        if len(all_fields_dict) == 0:
            updated_field_mps = {}
            for k, val in explicit_part_mps.items():
                updated_field_mps[k] = helper.scalar_multiply(val, dt, inplace=False)
        else:
            all_fields_mps = convert_dict_to_dmrg(all_fields_dict, is_mps=True)
            time_deriv_mpo = convert_dict_to_dmrg(all_fields_time_deriv_dict, is_mps=False)

            # for k, out_mps in all_fields_mps.items():
            #     print('mps exp', k, out_mps.exponent)
            #
            # for k, out_mps in explicit_part_mps.items():
            #     print('exp mps exp', k, [out_.exponent for out_ in out_mps])

            time_deriv_mpo_list = {}
            for k, item in time_deriv_mpo.items():
                time_deriv_mpo_list[k] = [item]

            if all_fields_dict is None and len(jF_dict) == 0 and len(background_comps) == 0:
                return new_sys

            # print('time deriv', time_deriv_mpo_list)
            # print('all fields', all_fields_mps)
            # print('explicit', explicit_part_mps)
            site_ind_id = all_fields_mps[next(iter(all_fields_mps))].site_ind_id
            for k, v in all_fields_mps.items():
                if v is not None:
                    v.site_ind_id = site_ind_id
            for k, v in explicit_part_mps.items():
                for x in v:
                    if x is not None:
                        x.site_ind_id = site_ind_id

            # if apply_constraints:
            #     constraints, constraint_vals = self.get_constraints(charge=self.charge_density)
            # else:
            #     constraints, constraint_vals = None, None

            from helper_block_tddmrg_3 import block_tddmrg
            updated_field_mps = block_tddmrg(dt, 6, time_deriv_mpo_list, all_fields_mps, explicit_part_mps,
                                             # constraints=constraints, constraint_vals=constraint_vals,
                                             max_bond=max_bond, te_order=te_order)
            # print('updated keys', updated_field_mps.keys())
            # exit()

        # for k, out_mps in all_fields_mps.items():
        #     print('mps exp', k, out_mps.exponent)

        out_gtns = {}
        ref_gtn = all_fields_dict[next(iter(all_fields_dict))]
        for k, out_mps in updated_field_mps.items():
            # print('updated gtn k', k)
            # print('out mps exp', out_mps.exponent)
            out_gtns[k] = self.grid_X.dmrg_to_gtn_format(ref_gtn.copy(), out_mps, is_mps=True)
        new_F1 = out_gtns

        # reset fields
        for i in range(3):
            compID = new_sys.coords_x.type_coords[i]

            ref_EC = new_sys.E.components.get(compID, None)
            ax_deriv_configs = ref_EC.ax_deriv_configs if ref_EC is not None else {}
            # EC = new_sys.grid_X.select_indexed_gtn(new_F1, i)
            EC = new_F1.get(i, None)
            if not (EC is None and compID not in new_sys.E.componentIDs):
                if EC is None:
                    EC = self.grid_X.make_empty_gridTN(ax_deriv_configs)
                else:
                    EC.ax_deriv_configs = ax_deriv_configs
                new_sys.E.components[compID] = EC
                # print('EC', compID, EC.max_bond(), EC.canon_site)

            ref_BC = new_sys.B.components.get(compID, None)
            ax_deriv_configs = ref_BC.ax_deriv_configs if ref_BC is not None else {}
            # BC = new_sys.grid_X.select_indexed_gtn(new_F1, 3 + i)
            BC = new_F1.get(i + 3, None)
            if not (BC is None and compID not in new_sys.B.componentIDs):
                if BC is None:
                    BC = self.grid_X.make_empty_gridTN(ax_deriv_configs)
                else:
                    BC.ax_deriv_configs = ax_deriv_configs
                new_sys.B.components[compID] = BC
                # print('BC', compID, BC.max_bond(), BC.canon_site)

        new_sys.E.compress(inplace=True, compress_level=1, norm_cutoff=1.0e-8)
        new_sys.B.compress(inplace=True, compress_level=1, norm_cutoff=1.0e-8)

        if new_sys.phi is not None:
            print('old phi norm', new_sys.phi.norm())
        new_phi = new_F1.get(6, None)
        if new_phi is not None:
            if new_sys.phi is not None:
                new_sys.phi.component = new_phi
            else:
                new_sys.phi = ScalarField('phi', self.grid_X, data=new_phi)
            print('new phi norm', new_sys.phi.norm())

        if new_sys.psi is not None:
            print('old psi norm', new_sys.psi.norm())
        new_psi = new_F1.get(7, None)
        if new_psi is not None:
            if new_sys.psi is not None:
                new_sys.psi.component = new_psi
            else:
                new_sys.psi = ScalarField('psi', self.grid_X, data=new_psi)
            print('new psi norm', new_sys.psi.norm(), new_psi is None, new_psi.data is None)

        # print('EC', compID, EC.max_bond(), EC.canon_site)

        # for C, EC in new_sys.E.components.items():
        #     print('EC', C, EC.ax_deriv_configs)
        # for C, EC in new_sys.B.components.items():
        #     print('BC', C, EC.ax_deriv_configs)
        #
        #
        # plt.figure()
        # for i in range(3):
        #     compID = new_sys.coords_x.type_coords[i]
        #     EC = new_sys.E.get_comp_data(compID)
        #     if EC is not None:
        #         plt.plot(EC, label=f'E{compID}')
        # for i in range(3):
        #     compID = new_sys.coords_x.type_coords[i]
        #     BC = new_sys.B.get_comp_data(compID)
        #     if BC is not None:
        #         plt.plot(BC, label=f'B{compID}')
        # plt.legend()

        # ax_x, ax_y = new_sys.grid_X.axes
        # for i in range(3):
        #     compID = new_sys.coords_x.type_coords[i]
        #     EC = new_sys.E.get_comp_data(compID)
        #     EC = ax_x.basis.get_realspace_1D(EC, 0)
        #     EC = ax_y.basis.get_realspace_1D(EC, 1)
        #     if EC is not None:
        #         plt.figure()
        #         plt.imshow(np.real(EC))
        #         plt.title(f'E{compID}')
        #         plt.colorbar()
        # for i in range(3):
        #     compID = new_sys.coords_x.type_coords[i]
        #     BC = new_sys.B.get_comp_data(compID)
        #     BC = ax_x.basis.get_realspace_1D(BC, 0)
        #     BC = ax_y.basis.get_realspace_1D(BC, 1)
        #     if BC is not None:
        #         plt.figure()
        #         plt.imshow(np.real(BC))
        #         plt.title(f'B{compID}')
        #         plt.colorbar()
        # plt.show()

        # print('E')
        # print(new_sys.E.components)
        # print('B')
        # print(new_sys.B.components)
        # exit()

        return new_sys

    def block_tddmrg_upwind(self, dt, te_order=4, time: float = None, use_A2=False, inplace=False, weight=0.5,
                            err_tol: float = np.sqrt(CUTOFF), max_iter: int = 100,
                            compress_level: int = 1, verbose_plot=False, apply_constraints=False, **deriv_kwargs):
        """
        ## dE/dt = c^2 * curl(B) - 1/eps0 * J [SI];  dE/dt = c * curl(B) - 4pi J [CGS]
        ## dB/dt = - curl(E) [SI];  dB/dt = - c curl(E) [CGS]

        | 1/dt * I, -s/2 curl | |E^(n+1)| = | 1/dt * I, s/2 curl | |E^(n)|  + |c^2 curl(B0)| - |J^(n+1/2)|
        |-s/2 curl,  1/dt * I | |B^(n+1)| = | s/2 curl, 1/dt * I | |B^(n)|  - |curl(E0)|     + |    0    |
        ## d/dt[ EB ] = A [EB] + b
        """
        new_sys = self if inplace else self.copy()
        # try_adaptive_solve = True  # False

        ## current, theoretically at time n + 1/2
        j: Field = self.current_density
        if j is not None:
            j = j.copy()
            if self.matl_params.is_cgs:
                j.scalar_multiply(-4 * np.pi, inplace=True)
            else:
                j.scalar_multiply(-1 / self.matl_params.eps0, inplace=True)

        ### dmrg method
        print("HERE, EM USE BLOCK TDDMRG UPWIND")

        compress_opts = new_sys.E.compress_config.get_compress_opts(1).copy()
        max_bond = compress_opts.get('max_bond', None)
        if max_bond is not None:
            compress_opts['max_bond'] = max_bond  # * num_active_fields

        ######################################
        ## update fields with current * dt/2
        if j is not None:
            for i in range(3):
                compID = new_sys.coords_x.type_coords[i]
                jc = j.components.get(compID, None)
                # print('compID', compID, j.components.keys())
                Ei = new_sys.E.components.get(compID, None)
                if jc is None:
                    pass
                elif Ei is None:
                    new_sys.E.components[compID] = jc.scalar_multiply(dt / 2)
                else:
                    new_sys.E.components[compID] = Ei.add(jc.scalar_multiply(dt / 2), compress=True,
                                                          compress_opts=compress_opts)
        ######################################

        ## current state
        all_fields_dict = self.get_state_dict()
        ## time evolution operator
        all_fields_time_deriv_dict = self.get_derivative_dict(dt)

        def add_gtn_list(list_gtn: Sequence['GridTN'], compress=1, **compress_opts):
            sum_out = list_gtn[0].copy()
            for gtn in list_gtn[1:]:
                sum_out.add(gtn, inplace=True)
            if compress:
                sum_out.compress(**compress_opts)
            return sum_out

        def apply_dict(fields_dict: dict[int, 'GridTN'], ops_dict: dict[tuple[int, int], 'GridTN'],
                       compress=1, **compress_opts) -> dict[int, 'GridTN']:

            outputs = {out: [] for out in range(6)}
            for k in ops_dict.keys():
                out_ind, in_ind = k
                op = ops_dict[k]
                vec = fields_dict.get(in_ind, None)
                if vec is not None and vec.data is not None:
                    outputs[out_ind] += [vec.apply(op, inplace=False)]

            output_gtns = {}
            for out, out_vecs in outputs.items():
                if len(out_vecs) == 0:
                    # output_gtns[out] = None  # self.grid_X.get_ones_mps()
                    continue
                output_gtns[out] = add_gtn_list(out_vecs, compress=compress, **compress_opts)
            return output_gtns

        def combine_field_dicts(*fields_dicts):
            combined_dict = {}
            for fdict in fields_dicts:
                for k, gtn in fdict.items():
                    if k not in combined_dict:
                        combined_dict[k] = [gtn]
                    else:
                        combined_dict[k] += [gtn]
            return combined_dict

        def convert_dict_to_dmrg(gtn_dict, is_mps=True):
            converted_dict = {}
            for k, gtns in gtn_dict.items():
                if isinstance(gtns, (list, tuple)):
                    dmrg_vecs = [self.grid_X.gtn_to_dmrg_format(gtn, is_mps=is_mps) for gtn in gtns]
                    # for tmp in dmrg_vecs:
                    #     tmp.distribute_exponent()
                else:
                    dmrg_vecs = self.grid_X.gtn_to_dmrg_format(gtns, is_mps=is_mps)
                    # dmrg_vecs.distribute_exponent()
                converted_dict[k] = dmrg_vecs
            return converted_dict

        # ## current
        # jF_dict = {}
        # if j is not None:
        #     if self.upwind:
        #         j_corrE, j_corrB = self._calculate_upwind_current(compress=compress_level,
        #                                                           compress1=compress_level + 1)
        #         jF_dict = {**{compID.type.value: comp for compID, comp in j_corrE.components.items()},
        #                    **{compID.type.value + 3: comp for compID, comp in j_corrB.components.items()}}
        #     else:
        #         jF_dict = {compID.type.value: comp for compID, comp in j.components.items()}
        #         # jF = self.grid_X.build_indexed_gtn({compID.type.value: comp for compID, comp in j.components.items()},
        #         #                                    index_order=list(range(6)))

        ## background field components
        background_comps = {}
        if self.background_E0 is not None:
            if self.curlE0 is None:
                self.curlE0 = self.background_E0.curl(self.coords_x, compress_level=5)
            scalar_E = -self.matl_params.c if self.matl_params.is_cgs else -1
            back_E_comps = {compID.type.value + 3: comp.scalar_multiply(scalar_E)
                            for compID, comp in self.curlE0.components.items()}
            background_comps.update(back_E_comps)
        if self.background_B0 is not None:
            if self.curlB0 is None:
                print('calculating curlB0')
                self.curlB0 = self.background_B0.curl(self.coords_x, compress_level=5)
            scalar_B = self.matl_params.c if self.matl_params.is_cgs else self.matl_params.c ** 2
            back_B_comps = {compID.type.value: comp.scalar_multiply(scalar_B, inplace=False)
                            for compID, comp in self.curlB0.components.items()}
            background_comps.update(back_B_comps)

        #########################
        explicit_part = background_comps  # combine_field_dicts(jF_dict, background_comps)
        explicit_part_mps = convert_dict_to_dmrg(explicit_part, is_mps=True)

        if len(all_fields_dict) == 0:
            updated_field_mps = {}
            for k, val in explicit_part_mps.items():
                updated_field_mps[k] = helper.scalar_multiply(val, dt, inplace=False)
        else:
            all_fields_mps = convert_dict_to_dmrg(all_fields_dict, is_mps=True)
            time_deriv_mpo = convert_dict_to_dmrg(all_fields_time_deriv_dict, is_mps=False)

            # for k, out_mps in all_fields_mps.items():
            #     print('mps exp', k, out_mps.exponent)
            #
            # for k, out_mps in explicit_part_mps.items():
            #     print('exp mps exp', k, [out_.exponent for out_ in out_mps])

            time_deriv_mpo_list = {}
            for k, item in time_deriv_mpo.items():
                time_deriv_mpo_list[k] = [item]

            if all_fields_dict is None and j is None and len(background_comps) == 0:
                return new_sys

            # print('time deriv', time_deriv_mpo_list)
            # print('all fields', all_fields_mps)
            # print('explicit', explicit_part_mps)
            site_ind_id = all_fields_mps[next(iter(all_fields_mps))].site_ind_id
            for k, v in all_fields_mps.items():
                if v is not None:
                    v.site_ind_id = site_ind_id
            for k, v in explicit_part_mps.items():
                for x in v:
                    if x is not None:
                        x.site_ind_id = site_ind_id

            if apply_constraints:
                constraints, constraint_vals = self.get_constraints(charge=self.charge_density)
            else:
                constraints, constraint_vals = None, None

            from helper_block_tddmrg_3 import block_tddmrg
            updated_field_mps = block_tddmrg(dt, 6, time_deriv_mpo_list, all_fields_mps, explicit_part_mps,
                                             max_bond=max_bond, te_order=te_order)

        # for k, out_mps in all_fields_mps.items():
        #     print('mps exp', k, out_mps.exponent)

        out_gtns = {}
        ref_gtn = all_fields_dict[next(iter(all_fields_dict))]
        for k, out_mps in updated_field_mps.items():
            # print('updated gtn k', k)
            # print('out mps exp', out_mps.exponent)
            out_gtns[k] = self.grid_X.dmrg_to_gtn_format(ref_gtn.copy(), out_mps, is_mps=True)

        ######################################
        ## update fields with current
        if j is not None:
            for i in range(3):
                compID = new_sys.coords_x.type_coords[i]
                jc = j.components.get(compID, None)
                # print('compID', compID, compID in j.components.keys())
                out_gtn_i = out_gtns.get(i, None)
                if jc is None:
                    pass
                elif out_gtn_i is None:
                    out_gtns[i] = jc.scalar_multiply(dt / 2)
                else:
                    out_gtns[i] = out_gtn_i.add(jc.scalar_multiply(dt / 2), compress=True, compress_opts=compress_opts)
        ######################################

        new_F1 = out_gtns

        # reset fields
        for i in range(3):
            compID = new_sys.coords_x.type_coords[i]

            ref_EC = new_sys.E.components.get(compID, None)
            ax_deriv_configs = ref_EC.ax_deriv_configs if ref_EC is not None else {}
            # EC = new_sys.grid_X.select_indexed_gtn(new_F1, i)
            EC = new_F1.get(i, None)
            if not (EC is None and compID not in new_sys.E.componentIDs):
                if EC is None:
                    EC = self.grid_X.make_empty_gridTN(ax_deriv_configs)
                else:
                    EC.ax_deriv_configs = ax_deriv_configs
                new_sys.E.components[compID] = EC
                # print('EC', compID, EC.max_bond(), EC.canon_site)

            ref_BC = new_sys.B.components.get(compID, None)
            ax_deriv_configs = ref_BC.ax_deriv_configs if ref_BC is not None else {}
            # BC = new_sys.grid_X.select_indexed_gtn(new_F1, 3 + i)
            BC = new_F1.get(i + 3, None)
            if not (BC is None and compID not in new_sys.B.componentIDs):
                if BC is None:
                    BC = self.grid_X.make_empty_gridTN(ax_deriv_configs)
                else:
                    BC.ax_deriv_configs = ax_deriv_configs
                new_sys.B.components[compID] = BC
                # print('BC', compID, BC.max_bond(), BC.canon_site)

        new_sys.E.compress(inplace=True, compress_level=1, norm_cutoff=1.0e-8)
        new_sys.B.compress(inplace=True, compress_level=1, norm_cutoff=1.0e-8)

        # for C, EC in new_sys.E.components.items():
        #     print('EC', C, EC.ax_deriv_configs)
        # for C, EC in new_sys.B.components.items():
        #     print('BC', C, EC.ax_deriv_configs)
        #
        #
        # plt.figure()
        # for i in range(3):
        #     compID = new_sys.coords_x.type_coords[i]
        #     EC = new_sys.E.get_comp_data(compID)
        #     if EC is not None:
        #         plt.plot(EC, label=f'E{compID}')
        # for i in range(3):
        #     compID = new_sys.coords_x.type_coords[i]
        #     BC = new_sys.B.get_comp_data(compID)
        #     if BC is not None:
        #         plt.plot(BC, label=f'B{compID}')
        # plt.legend()

        # ax_x, ax_y = new_sys.grid_X.axes
        # for i in range(3):
        #     compID = new_sys.coords_x.type_coords[i]
        #     EC = new_sys.E.get_comp_data(compID)
        #     EC = ax_x.basis.get_realspace_1D(EC, 0)
        #     EC = ax_y.basis.get_realspace_1D(EC, 1)
        #     if EC is not None:
        #         plt.figure()
        #         plt.imshow(np.real(EC))
        #         plt.title(f'E{compID}')
        #         plt.colorbar()
        # for i in range(3):
        #     compID = new_sys.coords_x.type_coords[i]
        #     BC = new_sys.B.get_comp_data(compID)
        #     BC = ax_x.basis.get_realspace_1D(BC, 0)
        #     BC = ax_y.basis.get_realspace_1D(BC, 1)
        #     if BC is not None:
        #         plt.figure()
        #         plt.imshow(np.real(BC))
        #         plt.title(f'B{compID}')
        #         plt.colorbar()
        # plt.show()

        # print('E')
        # print(new_sys.E.components)
        # print('B')
        # print(new_sys.B.components)
        # exit()

        return new_sys

    # @profile
    def backwards_euler(self, dt, time: float = None, err_tol: float = np.sqrt(CUTOFF), inplace=False,
                        max_iter: int = 100,
                        compress_level: int = 1, verbose_plot=False, **deriv_kwargs):
        """
        G^(n+1) = G^(n) + dt * f(G^(n+1))
            d/dt E = 1/c^2 * curl(B) - 1/eps0 * J
            d/dt B = - curl(E)

        | 1/dt * I,  -s curl | |E^(n+1)| = | 1/dt * I,      0   | |E^(n)|  - |J^(n)|
        |  -s curl, 1/dt * I | |B^(n+1)| = |     0   , 1/dt * I | |B^(n)|  + |  0  |
        where s is a sign and coefficient as defined by Maxwell's eq
        """
        new_sys = self if inplace else self.copy()
        use_dmrg = True

        ## current, theoretically at time n + 1/2
        j = self.current_density
        if j is not None:
            j = j.copy()
            if self.matl_params.is_cgs:
                j.scalar_multiply(-4 * np.pi, inplace=True)
            else:
                j.scalar_multiply(-1 / self.matl_params.eps0, inplace=True)

        ### dmrg method
        if use_dmrg:
            print("HERE, EM USE DMRG")

            all_fields_vec = self.get_combined_state()
            all_fields_time_deriv = self.get_combined_derivative_mpo()

            jF = None
            if j is not None:
                if self.upwind:
                    j_corrE, j_corrB = self._calculate_upwind_current(compress=compress_level,
                                                                      compress1=compress_level + 1)
                    jF = self.grid_X.build_indexed_gtn(
                        {**{compID.type.value: comp for compID, comp in j_corrE.components.items()},
                         **{compID.type.value + 3: comp for compID, comp in j_corrB.components.items()}},
                        index_order=list(range(6)))
                else:
                    jF = self.grid_X.build_indexed_gtn(
                        {compID.type.value: comp for compID, comp in j.components.items()},
                        index_order=list(range(6)))
            if jF is not None:
                jF = jF.scalar_multiply(dt, inplace=False)

            ## explicit contribution = all_fields_vec
            F1 = all_fields_vec.copy()
            if jF is not None:
                F1 = F1.add(jF, inplace=False, compress=compress_level)

            ## implicit contribution operator
            dFdt_gtn = all_fields_time_deriv.scalar_multiply(-dt, inplace=False)
            iden_gtn = dFdt_gtn.get_like_iden()
            imp_gtn = dFdt_gtn.add(iden_gtn)

            ## dmrg solve
            compress_opts = new_sys.E.compress_config.get_compress_opts(1)
            print('implicit dmrg solve', compress_opts)
            new_F1 = F1.solve(imp_gtn, compress_type=CompressType.DMRG, is_H=False, compress_opts=compress_opts)

            # reset fields
            for i in range(3):
                compID = new_sys.coords_x.type_coords[i]

                ref_EC = new_sys.E.components.get(compID, None)
                ax_deriv_configs = ref_EC.ax_deriv_configs if ref_EC is not None else None
                EC = new_sys.grid_X.select_indexed_gtn(new_F1, i)
                EC.ax_deriv_configs = ax_deriv_configs
                if not (EC is None and compID not in new_sys.E.componentIDs):
                    new_sys.E.components[compID] = EC

                ref_BC = new_sys.E.components.get(compID, None)
                ax_deriv_configs = ref_BC.ax_deriv_configs if ref_BC is not None else None
                BC = new_sys.grid_X.select_indexed_gtn(new_F1, 3 + i)
                BC.ax_deriv_configs = ax_deriv_configs
                if not (BC is None and compID not in new_sys.E.componentIDs):
                    new_sys.B.components[compID] = BC
            #
            # print('E')
            # print(new_sys.E.components)
            # print('B')
            # print(new_sys.B.components)
            # exit()

            # ## exact solve
            # tensor: qtn.Tensor = F1.data.contract()  # contract all tensors
            # tensor.modify(apply = lambda x: x * 10**F1.data.exponent)
            # out_inds = [F1.data.site_ind_id.format(i) for i in range(F1.data.num_tensors)]
            # tensor.transpose(*out_inds, inplace=True)
            # tensor_data = tensor.data.reshape(-1)
            # npts = len(tensor_data)
            #
            # imp_tensor: qtn.Tensor = imp_gtn.data.contract()  # contract all tensors
            # imp_tensor.modify(apply = lambda x: x * 10**imp_gtn.data.exponent)
            # i_inds = [imp_gtn.data.lower_ind_id.format(i) for i in range(imp_gtn.data.num_tensors)]
            # o_inds = [imp_gtn.data.upper_ind_id.format(i) for i in range(imp_gtn.data.num_tensors)]
            # imp_tensor.transpose(*o_inds, *i_inds, inplace=True)
            # imp_tensor_data = imp_tensor.data.reshape(npts,npts)
            #
            # new_F1_data = np.linalg.solve(imp_tensor_data, tensor_data)
            # new_F1_data = new_F1_data.reshape(6, -1)
            #
            # # plt.figure()
            # # plt.plot(new_F1_data.T)
            # # plt.title('new F1')
            # # plt.show()
            #
            # ## reset fields
            # for i in range(3):
            #     compID = new_sys.coords_x.type_coords[i]
            #
            #     ref_EC = new_sys.E.components.get(compID, None)
            #     ax_deriv_configs = ref_EC.ax_deriv_configs if ref_EC is not None else None
            #     data_ = new_F1_data[i, :].reshape(*[ax.npts for ax in new_sys.grid_X.axes])
            #     EC = new_sys.grid_X.map_state_to_mps(data_, ax_deriv_configs=ax_deriv_configs)
            #     if EC is not None: # and compID in new_sys.E.componentIDs:
            #         new_sys.E.components[compID] = EC
            #
            #     ref_BC = new_sys.E.components.get(compID, None)
            #     ax_deriv_configs = ref_BC.ax_deriv_configs if ref_BC is not None else None
            #     data_ = new_F1_data[3 + i, :].reshape(*[ax.npts for ax in new_sys.grid_X.axes])
            #     BC = new_sys.grid_X.map_state_to_mps(data_, ax_deriv_configs=ax_deriv_configs)
            #     if BC is not None: # and compID in new_sys.B.componentIDs:
            #         new_sys.B.components[compID] = BC
            #
            #     print('C max bond', i, EC.max_bond(), BC.max_bond())

        else:  ## solve iteratively until convergence
            raise NotImplementedError

        return new_sys
