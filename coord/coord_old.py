from setup.enums import CoordinateType
from gridTN import GridTN
from axis import Axis


class CoordinateSpace:

    @classmethod
    def set_coordinate_map(cls, *axes):
        return {x: axes[x] for x in range(len(axes))}

    # @classmethod
    # def take_firstderivative(cls, gtn: GridTN, deriv_dim: Axis, inplace=False, compress_level=1,
    #                          **deriv_params):
    #     raise NotImplementedError
    #
    # @classmethod
    # def build_mpo_secondderivative(cls, gtn: GridTN, deriv_ax1: Axis, deriv_ax2: Axis or None,
    #                                inplace=False, compress_level=1):
    #     raise NotImplementedError

    @classmethod
    def gradient(cls, gtn: GridTN, deriv_axes: list[Axis] or None = None, compress_level=1):
        raise NotImplementedError

    @classmethod
    def laplacian(cls, gtn: GridTN, deriv_axes: list[Axis] or None = None, compress_level=1):
        raise NotImplementedError

    @classmethod
    def curl(cls, gtn_comps: dict[CoordinateType or int,GridTN],
             coord_map: dict[CoordinateType or int, Axis],
             out_comps: tuple[int] or list[int] or None = None,
             compress_level=1):
        raise NotImplementedError

    @classmethod
    def divergence(cls, gtn_comps: dict[CoordinateType or int,GridTN],
                   coord_map: dict[CoordinateType or int, Axis] or None,
                   comps: tuple[CoordinateType or int] or list[CoordinateType or int] or None = None,
                   compress_level=1):
        raise NotImplementedError

    @classmethod
    def integrate_axis(cls, gtn: GridTN, ax: Axis, inplace=False):
        raise NotImplementedError

    @classmethod
    def integrate(cls, gtn: GridTN, axes: list[Axis], inplace=False):
        raise NotImplementedError