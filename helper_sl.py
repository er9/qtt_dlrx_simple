from setup_.defaults import *
import itertools
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from coord import Coordinate
    from coord.coord_sys import CoordinateSystem
    from axis import Axis
    from grid import Grid
    from grid import GridTN

def _is_use_midpt(sl_order):
    return sl_order == 2

def get_sl_weights(dt, dx, advec_coeffs, sl_order=DEFAULT_SL_ORDER):
    sl_data_obj = get_cell_data(dt, dx, advec_coeffs, sl_order)
    return sl_data_obj.get_weights(sl_order)

def get_cell_data(dt, dx, advec_coeffs, sl_order=DEFAULT_SL_ORDER, coeff_grid=None):
    """ midpt_cell:  if True, cell grid point p chosen s.t. |advec_coeff| <= 0.5
                     else, cell grid point p chosen s.t. 0 <= advec_coeff < 1
    """
    # print('get cell data advec coeffs', advec_coeffs, dt, dx)
    advec_coeffs = advec_coeffs * dt / dx
    # print('new:', advec_coeffs)

    if _is_use_midpt(sl_order):
        num_shift_cells = np.floor(np.real(advec_coeffs) + 0.5).astype(int)
    else:
        num_shift_cells = np.floor(np.real(advec_coeffs)).astype(int)
    # print('num shift', num_shift_cells, np.max(np.abs(num_shift_cells)))

    vals = {}
    ones = {}
    for p in np.unique(num_shift_cells):
        ## advec * dt / dx - p = fraction of weight in pth cell
        adv_cell = np.where(num_shift_cells == p, advec_coeffs - p + 1.0e-30, 0.)
        one_cell = adv_cell.astype(bool).astype(int)
        vals[p] = adv_cell
        ones[p] = one_cell
        # print('p', p, adv_cell, one_cell)

    if not (isinstance(advec_coeffs, np.ndarray)):
        print('vals', vals[p], ones[p])
        # exit()

    return SL_data(vals, ones, coeff_grid=coeff_grid)


def _get_sl_weight(cell, coeffs, ones, sl_order=3):
    """ get projection weights
    """
    weights = {}
    if sl_order == 1:  ## Kormann 2015 Section 4.1, Durran 7.5

        weights[-cell] = ones - coeffs
        weights[-cell - 1] = coeffs

    elif sl_order == 2:  ## Durran 7.11; note: requires use_midpt = True

        vals_m1 = coeffs * (ones + coeffs) / 2
        vals_00 = (ones - np.abs(coeffs) ** 2)
        vals_p1 = -coeffs * (ones - coeffs) / 2

        weights[-cell] =  vals_00
        weights[-cell - 1] = vals_m1
        weights[-cell + 1] = vals_p1

    elif sl_order == 3:   ## Durran 7.12

        vals_m2 = -coeffs * (ones - np.abs(coeffs) ** 2) / 6
        vals_m1 = coeffs * (ones + coeffs) * (2 * ones - coeffs) / 2
        vals_00 = (ones - np.abs(coeffs) ** 2) * (2 * ones - coeffs) / 2
        vals_p1 = -coeffs * (ones - coeffs) * (2 * ones - coeffs) / 6

        weights[-cell] = vals_00
        weights[-cell - 1] = vals_m1
        weights[-cell - 2] = vals_m2
        weights[-cell + 1] = vals_p1

    elif sl_order == 4:     ## cubic (Rahn-Allman Eq. 27, 28)

        if cell >= 0:
            vals_m2 = 1 * -coeffs * (ones - coeffs) * (ones + coeffs) / 6
            vals_m1 = 1 * coeffs * (ones - (ones - coeffs) * (2 * ones - coeffs) / 6 +
                                   (ones - coeffs) * (ones + coeffs) / 6)
            vals_00 = 1 * coeffs * (ones - coeffs) * (2 * ones - coeffs) / 6
            vals_p1 = 0.

            vals_m2 += 0.
            vals_m1 += -1 * -coeffs * (ones - coeffs) * (ones + coeffs) / 6
            vals_00 += -1 * coeffs * (ones - (ones - coeffs) * (2 * ones - coeffs) / 6 +
                                     (ones - coeffs) * (ones + coeffs) / 6)
            vals_p1 += -1 * coeffs * (ones - coeffs) * (2 * ones - coeffs) / 6

        else:
            ### 1 - v_neg = coeff --> v_neg = 1 - coeff     (v_neg = v_neg * dt/dx)
            ### p = 0 --> p = -(-1)
            ###
            # neg_m2 = 0
            # neg_m1 = +1 * -v_neg * (1 - v_neg) * (2 - v_neg) / 6
            # neg_00 = +1 * -v_neg * (1 - (2 - v_neg) * (1 - v_neg) / 6 + (1 + v_neg) * (1 - v_neg) / 6)
            # neg_p1 = +1 * v_neg * (1 - v_neg) * (1 + v_neg) / 6
            # neg_p2 = 0

            #  neg_m2 += 0
            #  neg_m1 += 0
            #  neg_00 += -1 * -v_neg * (1 - v_neg) * (2 - v_neg) / 6
            #  neg_p1 += -1 * -v_neg * (1 - (2 - v_neg) * (1 - v_neg) / 6 + (1 + v_neg) * (1 - v_neg) / 6)
            #  neg_p2 += -1 * v_neg * (1 - v_neg) * (1 + v_neg) / 6

            vals_m2 = 1 * -(ones - coeffs) * coeffs * (ones + coeffs) / 6
            vals_m1 = 1 * -(ones - coeffs) * (ones - coeffs * (ones - 2 * coeffs) / 6)
            vals_00 = 1 * (ones - coeffs) * coeffs * (2 * ones - coeffs) / 6
            vals_p1 = 0

            vals_m1 += -1 * -(ones - coeffs) * coeffs * (ones + coeffs) / 6
            vals_00 += -1 * -(ones - coeffs) * (ones - coeffs * (ones - 2 * coeffs) / 6)
            vals_p1 += -1 * (ones - coeffs) * coeffs * (2 * ones - coeffs) / 6

        weights[-cell] = vals_00
        weights[-cell - 1] = vals_m1
        weights[-cell - 2] = vals_m2
        weights[-cell + 1] = vals_p1

        weights[0] = 1
        # weights[-cell] = 1        ## maybe this instead?

    else:
        raise NotImplementedError(f'sl order {sl_order} not implemented')

    return weights


class SL_data:
    """ data object for SL calculations
    """
    def __init__(self, vals: dict[int, np.ndarray], ones: dict[int, np.ndarray], coeff_grid: 'Grid' = None):
        self._vals = vals
        self._ones = ones
        self.coeff_grid = coeff_grid

    @property
    def vals(self):
        return self._vals

    @property
    def ones(self):
        return self._ones

    @property
    def cells(self):
        return self.vals.keys()

    def get_weights(self, sl_order=DEFAULT_SL_ORDER):
        """ get weights to compute next time step
        """
        if sl_order == 4:
            print('Warning: maybe not correct?')

        shift_weights = {}

        def add_dict_val(xdict, k, v):
            if k in xdict:
                xdict[k] = xdict[k] + v
            else:
                xdict[k] = v
            return xdict

        coeff = self.vals[next(iter(self.cells))]
        if not isinstance(coeff, np.ndarray):
            print('COEFF', coeff)
            exit()

        for p in self.cells:
            cell_weights = _get_sl_weight(p, self.vals[p], self.ones[p], sl_order=sl_order)

            for wp, wval in cell_weights.items():
                add_dict_val(shift_weights, wp, wval)

        #
        # if sl_order == 1:       ## Kormann 2015 Section 4.1, Durran 7.5
        #
        #     for p in self.cells:
        #         coeff = self.vals[p]
        #         diff = self.ones[p] - self.vals[p]
        #
        #         add_dict_val(shift_weights, -p, diff)
        #         add_dict_val(shift_weights, -p - 1, coeff)
        #
        # elif sl_order == 2:     ## Durran 7.11; note: requires use_midpt = True
        #
        #     for p in self.cells:
        #         coeff = self.vals[p]
        #         ones = self.ones[p]
        #
        #         vals_m1 = coeff * (ones + coeff) / 2
        #         vals_00 = (ones - np.abs(coeff) ** 2)
        #         vals_p1 = -coeff * (ones - coeff) / 2
        #
        #         add_dict_val(shift_weights, -p, vals_00)
        #         add_dict_val(shift_weights, -p - 1, vals_m1)
        #         add_dict_val(shift_weights, -p + 1, vals_p1)
        #
        # elif sl_order == 3 or sl_order == 4:
        #
        #     for p in self.cells:
        #         coeff = self.vals[p]
        #         ones = self.ones[p]
        #
        #         if sl_order == 3:       ## cubic (Durran 7.12)
        #             vals_m2 = -coeff * (ones - np.abs(coeff) ** 2) / 6
        #             vals_m1 = coeff * (ones + coeff) * (2 * ones - coeff) / 2
        #             vals_00 = (ones - np.abs(coeff) ** 2) * (2 * ones - coeff) / 2
        #             vals_p1 = -coeff * (ones - coeff) * (2 * ones - coeff) / 6
        #
        #         else:                   ## cubic (Rahn-Allman Eq. 27, 28)
        #             if p >= 0:
        #                 vals_m2 = 1 * -coeff * (ones - coeff) * (ones + coeff) / 6
        #                 vals_m1 = 1 * coeff * (ones - (ones - coeff) * (2 * ones - coeff) / 6 +
        #                                        (ones - coeff) * (ones + coeff) / 6)
        #                 vals_00 = 1 * coeff * (ones - coeff) * (2 * ones - coeff) / 6
        #                 vals_p1 = 0.
        #
        #                 vals_m2 += 0.
        #                 vals_m1 += -1 * -coeff * (ones - coeff) * (ones + coeff) / 6
        #                 vals_00 += -1 * coeff * (ones - (ones - coeff) * (2 * ones - coeff) / 6 +
        #                                          (ones - coeff) * (ones + coeff) / 6)
        #                 vals_p1 += -1 * coeff * (ones - coeff) * (2 * ones - coeff) / 6
        #
        #             else:
        #                 ### 1 - v_neg = coeff --> v_neg = 1 - coeff     (v_neg = v_neg * dt/dx)
        #                 ### p = 0 --> p = -(-1)
        #                 ###
        #                 # neg_m2 = 0
        #                 # neg_m1 = +1 * -v_neg * (1 - v_neg) * (2 - v_neg) / 6
        #                 # neg_00 = +1 * -v_neg * (1 - (2 - v_neg) * (1 - v_neg) / 6 + (1 + v_neg) * (1 - v_neg) / 6)
        #                 # neg_p1 = +1 * v_neg * (1 - v_neg) * (1 + v_neg) / 6
        #                 # neg_p2 = 0
        #
        #                 #  neg_m2 += 0
        #                 #  neg_m1 += 0
        #                 #  neg_00 += -1 * -v_neg * (1 - v_neg) * (2 - v_neg) / 6
        #                 #  neg_p1 += -1 * -v_neg * (1 - (2 - v_neg) * (1 - v_neg) / 6 + (1 + v_neg) * (1 - v_neg) / 6)
        #                 #  neg_p2 += -1 * v_neg * (1 - v_neg) * (1 + v_neg) / 6
        #
        #                 vals_m2 = 1 * -(ones - coeff) * coeff * (ones + coeff) / 6
        #                 vals_m1 = 1 * -(ones - coeff) * (ones - coeff * (ones - 2 * coeff) / 6)
        #                 vals_00 = 1 * (ones - coeff) * coeff * (2 * ones - coeff) / 6
        #                 vals_p1 = 0
        #
        #                 vals_m1 += -1 * -(ones - coeff) * coeff * (ones + coeff) / 6
        #                 vals_00 += -1 * -(ones - coeff) * (ones - coeff * (ones - 2 * coeff) / 6)
        #                 vals_p1 += -1 * (ones - coeff) * coeff * (2 * ones - coeff) / 6
        #
        #         add_dict_val(shift_weights, -p, vals_00)
        #         add_dict_val(shift_weights, -p - 1, vals_m1)
        #         add_dict_val(shift_weights, -p - 2, vals_m2)
        #         add_dict_val(shift_weights, -p + 1, vals_p1)
        #
        #         if sl_order == 4:
        #             add_dict_val(shift_weights, 0, 1.)
        # else:
        #     raise NotImplementedError(f'sl order {sl_order} not implemented')

        return shift_weights


def get_multidimensional_weights(*sl_objs: 'SL_data', sl_order=3):
    """ get weights to compute next time step
        combining multiple 1D SL_data objects
    """
    if sl_order == 4:
        print('Warning: maybe not correct?')

    ndims = len(sl_objs)

    def add_dict_val(xdict, k, v):
        if k in xdict:
            xdict[k] = xdict[k] + v
        else:
            xdict[k] = v
        return xdict

    ## check if input data is ok
    for sl_obj in sl_objs:
        # print('sl obj', sl_obj.coeff_grid, sl_obj.coeff_grid.axes)
        if not sl_obj.coeff_grid.axes == sl_objs[0].coeff_grid.axes:
            raise TypeError('sl object coeff grids need to match (for now)')

        coeff = sl_obj.vals[next(iter(sl_obj.cells))]
        if not isinstance(coeff, np.ndarray):
            print('COEFF', coeff)
            raise TypeError('coeffs need to be np.ndarray')



    ## for all combinations of cell shifts
    multidim_weights = {}
    for cell_idxs in itertools.product(*[sl_obj.cells for sl_obj in sl_objs]):
        ## iterate over all combinations of shift cell indices

        ## get weights for each cell (shift)
        # print('cell idx', cell_idxs)
        cell_weights = []
        for i in range(ndims):
            p = cell_idxs[i]
            sl_obj = sl_objs[i]
            # weights = _get_sl_weight(p, sl_obj.vals[p], sl_obj.ones[p], sl_order=sl_order)
            cell_weights += [_get_sl_weight(p, sl_obj.vals[p], sl_obj.ones[p], sl_order=sl_order)]

        ## combine weights for each cell
        for shift_idxs in itertools.product(*[weights.keys() for weights in cell_weights]):
            # print('shift idxs', shift_idxs)
            new_key = shift_idxs
            new_vals = cell_weights[0][shift_idxs[0]]
            for i in range(1,ndims):
                new_vals = new_vals * cell_weights[i][shift_idxs[i]]

            add_dict_val(multidim_weights, new_key, new_vals)

    return multidim_weights


def get_single_val(args):
    """ gtn to measure
        it: which element
        weights: weights for SL projection
        coords: binary indices at which to measure gtn
        num_shift: number of cells to shift coords by, along Axis 'ax_'
        ax_: Axis along which the derivative is being taken
        ax_deriv_config: DerivativeConfiguration for the gtn
    """
    it, sub_args = args
    gtn, weights, coords, num_shift, ax_, ax_deriv_config = sub_args
    coord_00 = coords[it]
    p = num_shift[it]
    val_00 = gtn.meas_shifted_elem(coord_00, {ax_: -p}, ax_deriv_config)
    val_p1 = gtn.meas_shifted_elem(coord_00, {ax_: -p + 1}, ax_deriv_config)
    val_m1 = gtn.meas_shifted_elem(coord_00, {ax_: -p - 1}, ax_deriv_config)
    val_m2 = gtn.meas_shifted_elem(coord_00, {ax_: -p - 2}, ax_deriv_config)

    return weights[-2][it] * val_m2 + weights[-1][it] * val_m1 + \
           weights[0][it] * val_00 + weights[1][it] * val_p1

