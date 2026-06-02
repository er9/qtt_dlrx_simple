"""Cartesian coordinate system: the fully implemented :class:`CoordinateSystem`
mapping X/Y/Z coordinates to grid axes and providing differential operators on
that grid."""
from setup_.configs import *

from coord.coord_sys import Coordinate
from coord.coord_sys import CoordinateSystem
import axis as axis_class

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis
    from grid import Grid
    from gridTN import GridTN
    from field import Field


class CartesianCoordinateSpace(CoordinateSystem):

    # def __init__(self, coordID, coords_ax:dict[Coordinate,'Axis']):
    def __init__(self, coordID: str, *axes: 'Axis', coords: Optional[list['Coordinate']] = None):

        super().__init__(coordID, *axes, coords=coords)

        try:
            X_coord = next(c for c in self.coords if c.type == CoordinateType.X)
        except (StopIteration, AttributeError):
            X_coord = Coordinate('X',CoordinateType.X)

        try:
            Y_coord = next(c for c in self.coords if c.type == CoordinateType.Y)
        except (StopIteration, AttributeError):
            Y_coord = Coordinate('Y',CoordinateType.Y)

        try:
            Z_coord = next(c for c in self.coords if c.type == CoordinateType.Z)
        except (StopIteration, AttributeError):
            Z_coord = Coordinate('Z',CoordinateType.Z)

        # self.coords = [X_coord, Y_coord, Z_coord]

        self.type_coords = {CoordinateType.X: X_coord,
                            CoordinateType.Y: Y_coord,
                            CoordinateType.Z: Z_coord}

        self.X = self.coord_axes.get(X_coord, None)
        self.Y = self.coord_axes.get(Y_coord, None)
        self.Z = self.coord_axes.get(Z_coord, None)


    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y

    def get_Z(self):
        return self.Z


    def build_firstderivative_mpo(self, ax, deriv_config) -> dict['Axis', MPOType]:
        """ obtain MPO taking second derivative
            ax:  Axis objects to take derivative. if ax2 is None, take second deriv along ax1
            deriv_params:
                bc:  BCType or Tuples of BCTypes (left, right boundary conditions)
                order: finite difference expansion order
        """
        mpo = ax.build_firstderivative_mpo(deriv_config)
        return {ax: mpo}


    def build_secondderivative_mpo(self, ax1, ax2, deriv_config1, deriv_config2=None) -> dict['Axis', MPOType]:
        """ obtain MPO taking second derivative
            ax1, ax2:  Axis objects to take derivative. if ax2 is None, take second deriv along ax1
            deriv_params:
                bc1, bc2:  BCType or Tuples of BCTypes (left, right boundary conditions)
                order: finite difference expansion order
                fd_type: 'center', 'forwards', 'backwards'
        """
        if deriv_config2 is None:   deriv_config2 = deriv_config1

        if ax2 is None or ax1 == ax2:
            mpo1 = ax1.build_secondderivative_mpo(deriv_config1)
            mpo_dict = {ax1: mpo1}
        else:
            mpo1 = ax1.build_firstderivative_mpo(deriv_config1)
            mpo2 = ax2.build_firstderivative_mpo(deriv_config2)
            mpo_dict = {ax1: mpo1, ax2: mpo2}
        return mpo_dict


    def build_laplacian_mpo(self, axes: Sequence['Axis'], deriv_configs: dict['Axis','DerivativeConfiguration'],
                            eeo_grid: bool = False) \
            -> list[dict['Axis', MPOType]]:
        """ obtain MPO taking Laplacain over specified axes
        """
        mpo_list = []
        for ax in axes:
            out = ax.build_secondderivative_mpo(deriv_configs[ax], eeo_grid=eeo_grid)
            mpo_list += [{ax: out}]
        return mpo_list


    def build_mth_laplacian_mpo(self, deriv_order: int, axes: Sequence['Axis'],
                                deriv_configs: dict['Axis','DerivativeConfiguration'],
                                eeo_grid: bool = False) \
            -> list[dict['Axis', MPOType]]:
        """ obtain MPO taking mth-order Laplacian (d^m/dx^m) over specified axes
        """
        mpo_list = []
        for ax in axes:
            out = ax.build_mth_derivative_mpo(deriv_order, deriv_configs[ax], eeo_grid=eeo_grid)
            mpo_list += [{ax: out}]
        return mpo_list



    def build_vector_laplacian_mpo(self, compID, axes: Sequence['Axis'], deriv_configs: dict['Axis','DerivativeConfiguration']) \
            -> list[dict['Axis', MPOType]]:
        """ obtain MPO taking Laplacian over specified axes
        """
        return self.build_laplacian_mpo(axes, deriv_configs)


    def build_integral_mps(self, ax, is_sqrt=False) -> MPSType:
        ## multiply gtn by r, cos(th) etc. for other coords here
        return ax.basis.build_integral_mps(ax, is_sqrt=is_sqrt)

    def build_coarse_grain_mpx(self, ax:'Axis', depth:int, is_sqrt=False) -> MPOType:
        return ax.basis.build_coarse_grain_mpx(ax, depth, is_sqrt=is_sqrt)


    ######################################################
    ## operations performed on scalar (? remove?) and vector fields ##
    ######################################################

    # def take_firstderivative(self, gtn, coord, deriv_params=None, inplace=False, compress=True,
    #                          **compress_opts):
    #     """ df/dx_i
    #         deriv_params:   bc = boundary condition
    #                         order = finite difference stencil order
    #     """
    #     gtn = gtn if inplace else gtn.copy(deep=False)   # data is overwritten in apply anyways (not modified)
    #     if deriv_params is None:  deriv_params = {}
    #
    #     deriv_ax = self.get(coord)
    #     if deriv_ax is None:
    #         return None
    #
    #     mpo_ax_dict = self.build_firstderivative_mpo(deriv_ax, **deriv_params)
    #     ddx_mpo = gtn.grid.make_mpo_ndim(mpo_ax_dict)
    #     gtn.apply(ddx_mpo, inplace=True, compress=compress, **compress_opts)
    #         # inplace GTN op, MPX is not updated in place
    #     return gtn
    #
    #
    # def take_secondderivative(self, gtn, coord1, coord2=None, deriv_params=None,
    #                           inplace=False, compress=True, **compress_opts):
    #     """ d^2f/dx^2 {x} + d^2f/dy^2 {y} + d^2f/dz^2 {z}
    #     """
    #     gtn = gtn if inplace else gtn.copy(deep=False)
    #     if deriv_params is None:    deriv_params = {}
    #
    #     ax1 = self.get(coord1)
    #     ax2 = self.get(coord2) if coord2 is not None else None
    #     if ax1 is None:
    #         return None
    #
    #     mpo_ax_dict = self.build_secondderivative_mpo(ax1, ax2, **deriv_params)
    #     d2_mpo = gtn.grid.make_mpo_ndim(mpo_ax_dict)
    #     gtn.apply(d2_mpo, inplace=True, compress=compress, **compress_opts)
    #         # inplace GTN op, MPX is not updated in place
    #     return gtn


    # # def integrate(self, gtn, axes=None, inplace=False) -> 'GridTN':
    # #    gtn = gtn if inplace else gtn.copy()
    # #    ## multiply gtn by r, cos(th) etc. for other coords here?
    # #    integ_mpx = gtn.grid.get_integrals_mpx(axes)
    # #    return gtn.integrate(axes, compress=False)


    ########################################
    ## Vector and scalar field operations ##
    ########################################

    def _gtn_gradient(self, gtn: 'GridTN', axes: Iter['Axis'] = None, compress=False, compress_opts=None) \
            -> dict['Coordinate','GridTN']:
        """ df/dx {x} + df/dy {y} + df/dz {z}
            gtn:  GridTN (MPS) representing the scalar field
            coords:  coordinates to include in calculation of Laplacian
                (CoordinateType.X, .Y, or int)
        """
        if axes is None:
            axes = [self.coord_axes.get(c,None) for c in self.coords]

        out = {}
        for ax in axes:
            if ax is None:   continue
            # deriv_params = gtn.deriv_params[ax.axID]
            out_comp = gtn.take_firstderivative(ax, inplace=False, compress=compress, compress_opts=compress_opts)
            out[ax.coordinate] = out_comp
        return out


    def _gtn_vector_derivative(self, gtn_comps: dict['Coordinate','GridTN'], ax: 'Axis', compress=False, compress_opts=None) \
            -> dict['Coordinate','GridTN']:
        """ d/dx (Ax {x}), d/dx (Ay {y}), d/dx (Az {z})
        """
        # if deriv_coord == CoordinateType.X:  doesn't matter for cartesian coordinates

        out = {}
        for c, gtn_comp in gtn_comps.items():
            # deriv_params = gtn_comp.deriv_params[ax.axID]
            deriv_out = gtn_comp.take_firstderivative(ax, inplace=False, compress=compress, compress_opts=compress_opts)
            out[c] = deriv_out

        return out


    def _gtn_laplacian(self, gtn: 'GridTN', axes: Iter['Axis'] = None, compress=False, compress_opts=None,
                       inner_compress=False, inner_compress_opts=None) -> 'GridTN':
        """ L(f) = d^2/dx^2 f + d^2/dy^2 f + d^2/dz^2 z
            gtn:  GridTN (MPS) representing the scalar field
            coords:  coordinates to include in calculation of Laplacian
                (CoordinateType.X, .Y, or int)
        """
        if axes is None:
            axes = [self.coord_axes.get(c,None) for c in self.coords]

        ind0 = next(i0 for i0 in range(len(axes)) if axes[i0] is not None)

        # deriv_params = gtn_deriv_params[axes[0].axID] if gtn_deriv_params is not None else None
        out = gtn.take_secondderivative(axes[ind0], None, inplace=False,
                                        compress=inner_compress, compress_opts=inner_compress_opts)

        for ax in axes[ind0+1:]:
            # deriv_params = gtn_deriv_params[ax.axID] if gtn_deriv_params is not None else None
            out2 = gtn.take_secondderivative(ax, None, inplace=False,
                                             compress=inner_compress, compress_opts=inner_compress_opts)
            out = out.add(out2, inplace=True)

        if compress:
            out = out.compress(compress_opts=compress_opts)

        return out  # scalar field


    def _gtn_mth_laplacian(self, gtn: 'GridTN', deriv_order: int, axes: Iter['Axis'] = None, compress=False,
                           compress_opts=None, inner_compress=False, inner_compress_opts=None) -> 'GridTN':
        """ L(f) = d^2/dx^2 f + d^2/dy^2 f + d^2/dz^2 z
            gtn:  GridTN (MPS) representing the scalar field
            coords:  coordinates to include in calculation of Laplacian
                (CoordinateType.X, .Y, or int)
        """
        if axes is None:
            axes = [self.coord_axes.get(c, None) for c in self.coords]

        ind0 = next(i0 for i0 in range(len(axes)) if axes[i0] is not None)

        # deriv_params = gtn_deriv_params[axes[0].axID] if gtn_deriv_params is not None else None
        out = gtn.take_mth_derivative(deriv_order, axes[ind0], inplace=False,
                                      compress=inner_compress, compress_opts=inner_compress_opts)

        for ax in axes[ind0 + 1:]:
            # deriv_params = gtn_deriv_params[ax.axID] if gtn_deriv_params is not None else None
            out2 = gtn.take_mth_derivative(deriv_order, ax, inplace=False,
                                           compress=inner_compress, compress_opts=inner_compress_opts)
            out = out.add(out2, inplace=True)

        if compress:
            out = out.compress(compress_opts=compress_opts)

        return out  # scalar field

    # def _gtn_laplacian_mpo(self, grid: 'Grid', axes: Sequence['Axis'] = None,
    #                        ax_deriv_kwargs: Optional[dict[Axis,DerivativeConfiguration]] = None):
    #     """ d^2/dx^2 + d^2/dy^2 + d^2/dz^2
    #         the same for all components
    #     """
    #     return grid.laplacian_mpo(axes, ax_deriv_kwargs)
    #
    #
    # def _gtn_vector_laplacian_mpo(self, compID, grid: 'Grid', axes: Sequence['Axis'] = None,
    #                               ax_deriv_kwargs: Optional[dict[Axis,DerivativeConfiguration]] = None):
    #     """ d^2/dx^2 + d^2/dy^2 + d^2/dz^2
    #         the same for all components
    #     """
    #     return self._gtn_laplacian_mpo(grid, axes, ax_deriv_kwargs=ax_deriv_kwargs)


    def _gtn_vector_laplacian(self, gtn_comps: dict['Coordinate','GridTN'], axes: Iter['Axis'] = None,
                              compress=False, compress_opts=None,
                              inner_compress=False, inner_compress_opts=None) -> dict['Coordinate','GridTN']:
        """ VL(A) = L(Ax) {x} + L(Ay) {y} + L(Az) {z}
            gtn_comps:  GridTN (MPS) representing the vector field
        """
        if axes is None:
            axes = [self.coord_axes.get(c,None) for c in self.coords]

        out = {}
        for k, gtn_comp in gtn_comps.items():
            if gtn_comp is None:
                out[k] = None
                continue
            out[k] = self._gtn_laplacian(gtn_comp, axes, compress=compress, compress_opts=compress_opts,
                                         inner_compress=inner_compress, inner_compress_opts=inner_compress_opts)

        return out


    # @profile
    def _gtn_curl(self, gtn_comps: dict['Axis', 'GridTN'], out_comps: Iter = None, compress=False, compress_opts=None,
                  inner_compress=False, inner_compress_opts=None) -> dict['Axis','GridTN']:
        """ Curl(F) =  ( d/dy Az - d/dz Ay ) {x} +
                       ( d/dz Ax - d/dx Az ) {y} +
                       ( d/dx Ay - d/dy Ax ) {z}
            gtn_comps:  dictionary of gtn_comps, with CoordinateType/int as keys
            out_comps: tuple of which output components of curl is desired (list of component keys)
        """
        if out_comps is None:
            out_comps = self.coords

        # print('out comps', out_comps)
        # print('gtn comps', gtn_comps)

        out = {}
        for axID in out_comps:
            i = axID.type  # self.ax_coords[axID]
            i1, i2 = (i + 1) % 3, (i + 2) % 3

            # ax1_ = self.coordtype_axes.get(i1, None)
            # ax2_ = self.coordtype_axes.get(i2, None)

            c1_ = self.type_coords.get(i1, None)
            c2_ = self.type_coords.get(i2, None)

            # try:
            #     ax1_ = self.coords_ax[i1]
            #     ax2_ = self.coords_ax[i2]
            # except KeyError:
            #     # these components don't exist/are None
            #     continue

            # print('curl ax', axID, c1_, c2_)
            # exit()

            try:
                ax1_ = self.coord_axes[c1_]
                gtn_comp = gtn_comps[c2_] #[ax2_.axID]
                if gtn_comp is None:   raise KeyError
                # print('ax1', ax1_, ax1_.basis)
                out_comp_1 = gtn_comp.take_firstderivative(ax1_, compress=False, # inner_compress,
                                                           compress_opts=inner_compress_opts)
                # plt.figure()
                # plt.plot(np.real(gtn_comp.get_data()), label='comp re')
                # plt.plot(np.imag(gtn_comp.get_data()), '--', label='comp im')
                # plt.plot(np.real(out_comp_1.get_data()), label='deriv re')
                # plt.plot(np.imag(out_comp_1.get_data()), '--', label='deriv im')
                # plt.legend()
                # plt.title(f'curl comp {i}: c2 {c2_} deriv {c1_}')
                # plt.show()

            except KeyError:
                out_comp_1 = None  ## vector_field[i] = 0

            try:
                ax2_ = self.coord_axes[c2_]
                gtn_comp = gtn_comps[c1_]  # [ax1_.axID]
                if gtn_comp is None:    raise KeyError
                # print('ax2', ax2_, ax2_.basis)
                out_comp_2 = gtn_comp.take_firstderivative(ax2_, compress=inner_compress,
                                                           compress_opts=inner_compress_opts)
                # plt.figure()
                # plt.plot(np.real(gtn_comp.get_data()), label='comp re')
                # plt.plot(np.imag(gtn_comp.get_data()), '--', label='comp im')
                # plt.plot(np.real(out_comp_2.get_data()), label='deriv re')
                # plt.plot(np.imag(out_comp_2.get_data()), '--', label='deriv im')
                # plt.legend()
                # plt.title(f'curl comp {i}: c1 {c1_} deriv {c2_}')
                # plt.show()

                out_comp_2 = out_comp_2.scalar_multiply(-1, inplace=True)

            except KeyError:
                out_comp_2 = None  ## vector_field[i] = 0

            if out_comp_1 is None and out_comp_2 is None:
                continue
            elif out_comp_1 is None and out_comp_2 is not None:
                out[axID] = out_comp_2
            elif out_comp_1 is not None and out_comp_2 is None:
                out[axID] = out_comp_1
            else:
                # print('use add')
                out_comp_1.add(out_comp_2, inplace=True)
                out[axID] = out_comp_1

            # print('out comps', out_comp_1, out_comp_1.exponent, out_comp_1.sign)
            # print('out comps', out_comp_2, out_comp_2.exponent, out_comp_2.sign)
            # print('curl out', axID, out[axID])

            # out[i] = out_comp_1
            # print('curl out', out)
            if compress and out[axID] is not None:
                out[axID].compress(inplace=True, compress_opts=compress_opts)

        # plt.figure()
        # for axID in out_comps:
        #     if axID in out:
        #         B_data = out[axID].get_data()
        #         if B_data is not None:
        #             plt.plot(np.real(B_data), label=f'{axID} re')
        #             plt.plot(np.imag(B_data), '--', label=f'{axID} im')
        # plt.xlabel('x')
        # plt.legend()
        # plt.title('curl results')
        # plt.show()

        return out


    def _gtn_divergence(self, gtn_comps, axes=None, ax_deriv_configs=None, compress=False, compress_opts=None,
                        inner_compress=False, inner_compress_opts=None) -> 'GridTN':
        """ d/dx Ax + d/dy Ay + d/dz Az
            gtn_comps:  dictionary of gtn_comps, with CoordinateType/int as keys
            comps:  components of gtn_comps to include when computing divergence
        """
        if axes is None:
            axes = [self.coord_axes.get(c, None) for c in self.coords]

        out = None
        for ax in axes:
            if ax is None:  continue

            try:
                ax_deriv_configs_ = gtn_comps[ax.coordinate].ax_deriv_configs if ax_deriv_configs is None \
                                        else ax_deriv_configs
                # print('div ax deriv configs', ax.coordinate, ax_deriv_configs_[ax])
                out_comp_1 = gtn_comps[ax.coordinate].take_firstderivative(ax, ax_deriv_config=ax_deriv_configs_,
                                                                           compress=inner_compress,
                                                                           compress_opts=inner_compress_opts)

                if out is None:
                    out = out_comp_1
                else:
                    out.add(out_comp_1, inplace=True, compress=inner_compress, compress_opts=inner_compress_opts)
            except KeyError:
                pass

        if compress and out is not None:
            out.compress(compress_opts=compress_opts)

        return out  ## output is a GridTN


    def _gtn_divergence_mpos(self, gtn_comps: 'Field', axes=None, ax_deriv_configs=None, compress=False, compress_opts=None,
                             inner_compress=False, inner_compress_opts=None) -> dict['Coordinate','GridTN']:
        """ d/dx Ax + d/dy Ay + d/dz Az
            gtn_comps:  dictionary of gtn_comps, with CoordinateType/int as keys
            comps:  components of gtn_comps to include when computing divergence
        """
        if axes is None:
            axes = [self.coord_axes.get(c, None) for c in self.coords]

        out = {}
        for ax in axes:
            if ax is None:  continue
            try:
                ax_deriv_configs = gtn_comps[ax.coordinate].ax_deriv_configs
            except KeyError:
                ax_deriv_configs = DerivativeConfiguration()

            ddx_1 = gtn_comps.grid.get_firstderivative_mpo(ax, deriv_config=ax_deriv_configs.get(ax,None))
            out[ax.coordinate] = ddx_1

        return out  ## output is a GridTN


