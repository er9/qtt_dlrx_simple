"""GridTN_Composite: tensor-network state over a composite (multi-grid) layout.

Parent class for tensor-network data that spans several sub-grids, holding a list of
per-sub-grid :class:`~gridTN.GridTN` objects on a composite grid. Specialized by
:class:`~gridTN_1Dcomb.GridTN1DComb` for the comb (tree-like) QTT layout.
"""

from setup_.configs import *
from gridTN import GridTN

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import quimb.tensor as qtn
    from axis import Axis
    from grid import Grid
    from grids_composite import CompositeGrid
    GridType = Union[Grid, CompositeGrid, 'Grid1D', 'GridsComb']


class GridTN_Composite(GridTN):

    def __init__(self, grid: 'GridType', data: list[dict['Grid','GridTN']] = None,
                 ax_deriv_configs: dict['Axis','DerivativeConfiguration'] = None):
        """ grid:  grid on which the TN lives
            data:  data in compatible format for GridType
        """
        super().__init__(grid, None, ax_deriv_configs)

        self.gtn_types = None
        if data is not None:
            self.gtn_types = [gtn.data_type for ax, gtn in data[0].items()]

        self.data = data    ## list of GTNs (MPX) indexed by


    def _get_data(self):
        return self._data

    def _set_data(self, data):
        if data is not None:
            # self._data = self.grid.make_gtn_from_dicts(data)
            self._data = self.data
            self._data_type = self._data[0].data_type

    data = property(fget=_get_data,fset=_set_data)


    def _get_data_type(self):
        return self._data_type

    data_type = property(fget=_get_data_type)

    # def update_deriv_params(self, ax, left_bc: Optional[BCType or str] = None,
    #                         right_bc: Optional[BCType or str] = None, order: Optional[int] = None,
    #                         fd_type: Optional[FDType or str] = None):
    #     """ kwargs:  'order', 'fdtype', 'bc', 'upwind', 'v_ax'
    #     """
    #     super().update_deriv_params(ax, left_bc, right_bc, order, fd_type)

