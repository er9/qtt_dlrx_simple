"""GridTN: base tensor-network state paired with a grid.

Parent class for tensor-network data (MPS / MPO) associated with a :class:`~grid.Grid`.
Concrete subclasses are :class:`~gridTN_1D.GridTN1D` (1-D QTT / MPS / MPO) and the
composite comb-layout variants in :mod:`gridTN_composite` / :mod:`gridTN_1Dcomb`.
Provides the static creation helpers and the common tensor-network interface used by
:mod:`field` and the PDE solvers.
"""

import pickle

import helper_dmrg
from setup_.configs import *
import helper_quimb as helper
from helper_tdvp import TDVPSolver

from basis.basis_spatial import SpatialBasis
from basis.basis_k import FourierBasis

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import quimb.tensor as qtn
    from axis import Axis
    from coord.coord_sys import Coordinate
    from coord.coord_sys import CoordinateSystem
    from grid import Grid
    from grids_composite import CompositeGrid

    GridType = Union[Grid, CompositeGrid, 'Grid1D', 'GridsComb']  # tuple[Grid]

""" object with MPS/MPO as data, plus the GridLayout object (the grid) it's associated with
"""


########################################
## static methods to create GridTN1Ds ##
########################################


####################
## GridTN1D class ##
####################

class GridTN:

    def __init__(self, grid: 'GridType', data: list[dict['Axis', 'TNType']] = None,
                 ax_deriv_configs: dict['Axis', 'DerivativeConfiguration'] = None):
        """ grid:  grid on which the TN lives
            data:  data in compatible format for GridType
        """

        self.grid = grid
        self._data_type = None
        self.data = data  # list of dicts with grid.axIDs as keys
        # self.is_sqrt = False
        self.info = {}  # solver instrumentation (e.g. 'internal_rank', 'num_evals')

        if ax_deriv_configs is None:
            self.ax_deriv_configs = {ax: DerivativeConfiguration() for ax in self.grid.axes}
        else:
            self.ax_deriv_configs = ax_deriv_configs

        self.canon_site = None  # None: not canonical; otherwise indexes site canonicalized around
        self.is_constant = False
        self.constant_axes = []

    def __repr__(self):
        return str(self.__class__) + ' ' + str(self.data)

    def __getitem__(self, item):
        raise NotImplementedError

    @property
    def exponent(self):
        return self.data.exponent

    def mangle_inner(self, inplace=True):
        raise NotImplementedError

    def _get_data(self) -> 'TNType':
        return self._data

    def _set_data(self, data):
        self._data = data
        if data is not None:
            data_ = data[0]
            self.data_type = type(data_[next(iter(data_))])
        self.constant_axes = []
        self.is_constant = False

    data = property(fget=_get_data, fset=_set_data)

    def _get_data_type(self):
        return self._data_type

    def _set_data_type(self, data_class):
        if data_class == qtn.MatrixProductState or issubclass(data_class, qtn.MatrixProductState):
            self._data_type = DataType.MPS
        elif data_class == qtn.MatrixProductOperator:
            self._data_type = DataType.MPO
        elif data_class == qtn.TensorNetwork1D:
            self._data_type = DataType.MPX
        elif data_class == MatrixProductTensor:
            self._data_type = DataType.TN3
        elif issubclass(data_class,(float,complex)):
            self._data_type = DataType.Num
        else:
            print(data_class)
            raise TypeError('GridTN data must be MPS, MPO, TN1, or MPT or Number')

    data_type = property(fget=_get_data_type, fset=_set_data_type)

    def update_deriv_params(self, ax, left_bc: Optional[BCType or str] = None,
                            right_bc: Optional[BCType or str] = None, order: Optional[int] = None,
                            fd_type: Optional[FDType or str] = None, offset: Optional[int] = None):
        """ kwargs:  'order', 'fdtype', 'bc', 'upwind', 'v_ax'
        """
        dp = self.ax_deriv_configs[ax]
        dp.update(left_bc, right_bc, order, fd_type, offset)  # , v_ax)

    def create_like(self, new_data=None) -> 'GridTN':
        raise NotImplementedError

    def copy(self, deep=True) -> 'GridTN':
        raise NotImplementedError

    def conj(self, inplace=False, mangle_inner=False) -> 'GridTN':
        raise NotImplementedError

    def save_data(self, fstr):
        pickle.dump(self.data, open(fstr + '.pkl', 'wb'))
        print('saved data', fstr + '.pkl')

    def reload_data(self, fstr, ax_deriv_configs=None):
        """ self.data set by loaded data
        """
        grid = self.grid
        return grid.load_gtn_data(fstr, self, ax_deriv_configs=ax_deriv_configs)

    @classmethod
    def load_data(cls, grid: 'Grid', fstr, ax_deriv_configs=None):
        """ generate gtn with loaded data
        """
        gtn = cls(grid, data=None, ax_deriv_configs=ax_deriv_configs)
        return grid.load_gtn_data(fstr, gtn)

    def get_TN(self):
        raise NotImplementedError

    def norm(self, is_sqrt=False) -> Numeric:
        if self.data is not None:
            if self.data_type is DataType.Num:
                return self.data
            out = self.integrate(is_sqrt=is_sqrt)
            if np.isreal(out) or np.abs(np.imag(out)) < 1.0e-13:
                out = np.real(out)
            if is_sqrt:
                out = np.sqrt(out)
            return out

    def frobenius_norm(self) -> Numeric:
        if self.data is not None:
            # out = helper.ovlp(self.data, self.data.conj())
            self_conj = self.conj(mangle_inner=True, inplace=False)
            out = self.ovlp(self_conj)
            out = np.sqrt(np.abs(out))
            # print('fro norm', out) # , self.data.norm() * 10**self.exponent)
            return out
        else:
            return 0.0

    def normalize(self, target_value, is_sqrt=False, inplace=False) -> 'GridTN':
        """ if basis is Spatial, then do scalar multiplication
            if spatial basis is Fourier, then only scale k=0 term
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)  # return None?

        is_spatial = [ax.basis.type == BasisType.SPATIAL for ax in self.grid.axes]
        is_fourier = [ax.basis.type == BasisType.FOURIER for ax in self.grid.axes]

        self_norm = self.norm(is_sqrt=is_sqrt)
        scale_val = target_value / np.conj(self_norm)
        if np.isnan(scale_val) and np.abs(self_norm) < 1.0e-12:
            scale_val = 0.0

        if True:  # np.all(is_spatial):
            out = self.scalar_multiply(scale_val, inplace=inplace)
        elif np.any(is_fourier):
            # only adjust k=0 term which may be normalized to 0
            axes_k = [self.grid.axes[i] for i in range(len(is_fourier)) if is_fourier[i]]
            select_dict = {ax: ax.get_select_elems_mpo([ax.zero_ind]) for ax in axes_k}
            select_zero_gtn = self.grid.make_mpo_ndim(select_dict)
            select_zero_gtn.scalar_multiply(scale_val - 1, inplace=True)
            iden_gtn = self.grid.get_iden_mpo()
            scale_gtn = iden_gtn.add(select_zero_gtn)

            out = self.apply(scale_gtn, inplace=inplace, compress=True)

        else:
            raise NotImplementedError

        return out

    def ovlp(self, other) -> Numeric:
        raise NotImplementedError

    def distance(self, other) -> Numeric:

        # self_conj = self.conj(mangle_inner=True, inplace=False)
        # norm2a = self.ovlp(self_conj)
        # norm2b = other.ovlp(other.conj(mangle_inner=True, inplace=False))
        # # ovlp_ab = self.ovlp(other.conj(inplace=False))
        # ovlp_ab = other.ovlp(self_conj)
        # dist = norm2a + norm2b - 2 * np.real(ovlp_ab)
        #
        # if dist < 0:
        #     print('neg dist', dist)
        # # if dist < 0 and np.abs(dist) < 1.0e-13:
        # #     print('small neg dist', dist)
        # #     dist = np.abs(dist)
        # dist1 = np.sqrt(dist)

        other = other.scalar_multiply(-1, inplace=False)
        diff = self.add(other, inplace=False, compress=False)
        dist = diff.frobenius_norm()

        # print('distances', dist, dist1)
        return dist

    def max_bond(self) -> int:
        raise NotImplementedError

    def mem_size(self) -> Numeric:
        raise NotImplementedError

    def num_elem(self) -> Numeric:
        raise NotImplementedError

    def all_virtual_sizes(self) -> int:
        raise NotImplementedError

    def all_smallest_singular_values(self, max_bond=None) -> int:
        raise NotImplementedError

    def entanglement_entropy_all(self) -> list[Numeric]:
        raise NotImplementedError

    def check_orthog(self):
        raise NotImplementedError

    def get_anchor_ind(self) -> int:
        """ get "lead" tensor (for connecting to other GTNs in composite grid systems)
        """
        raise NotImplementedError

    def get_anchor_tens(self) -> qtn.Tensor:
        """ get "lead" tensor (for connecting to other GTNs in composite grid systems)
        """
        raise NotImplementedError

    def transpose(self, inplace=True, mangle_inner=False):
        """ take transpose of MPO
        """
        raise NotImplementedError

    def get_like_iden(self):
        """ get identity MPO with appropriate shape
        """
        raise NotImplementedError

    #######################################
    ## class methods to build common TNs ##
    #######################################

    #### move to grid_* classes
    @classmethod
    def get_ones_mps(cls, grid: 'GridType', site_ind_id='i({})', site_tag_id='X({})') -> 'GriDTN':
        """ build ones vector mps on sepcified grid
        """
        raise NotImplementedError

    @classmethod
    def get_iden_mpo(cls, grid: 'GridType', upper_ind_id='i({})', lower_ind_id='o({})',
                     site_tag_id='X({})') -> 'GridTN':
        """ build identity mpo on specified grid
        """
        raise NotImplementedError

    @classmethod
    def get_select_elem_mps(cls, grid: 'GridType', inds, site_ind_id='i({})', site_tag_id='X({})') -> 'GridTN':
        """ build MPO to select certain elements specified by inds
        """
        raise NotImplementedError

    @classmethod
    def get_select_elem_mpo(cls, grid: 'GridType', inds, upper_ind_id='i({})', lower_ind_id='o({})',
                            site_tag_id='X({})') -> 'GridTN':
        """ build MPO to select certain elements specified by inds
        """
        raise NotImplementedError

    @classmethod
    def from_dense_state(cls, data: np.ndarray, grid: 'GridType', site_ind_id='i({})',
                         site_tag_id='T({})', split_opts=None, axes=None) -> 'GridTN':
        raise NotImplementedError

    @classmethod
    def from_dense_operator(cls, data: np.ndarray, grid: 'GridType', upper_ind_id='o({})',
                            lower_ind_id='i({})', site_tag_id='T({})', split_opts=None, **kwargs) -> 'GridTN':
        raise NotImplementedError

    ####

    def get_data(self, ax_order: Sequence['Axis'] = None, ax_select: dict['Axis', int] = None,
                 pad_data=False) -> np.ndarray:
        raise NotImplementedError

    def get_squeezed_data(self) -> tuple[np.ndarray, 'Grid']:
        data = self.get_data(ax_select={ax: ax.get_constant_ind() for ax in self.constant_axes})
        data_gr = self.grid.get_subgrid([ax for ax in self.grid.axes if ax not in self.constant_axes])
        return data, data_gr

    #######################
    ## MPS/MPX functions ##
    #######################

    def apply(self, other: 'GridTN', inplace=False, zipup=False, use_mg=False, compress=False, compress_opts=None,
              add_cc=False, **kwargs) -> 'GridTN':
        """ Apply grid_mpo to self, assuming they exist on the same grid
            compress [int]:  determines compression parameters from compression level
            compress_opts:  overrides compression parameters
        """
        raise NotImplementedError

    def apply_rdm(self, other, bra_self=None, bra_other=None, left_env=None, right_env=None,
                  direction=1, inplace=False, compress_opts=None, **kwargs) -> 'GridTN1D':
        """ Apply grid_mpo to self, assuming they exist on the same grid
            contract (and compress) using rdm method
        """
        raise NotImplementedError

    def solve(self, operator: 'GridTN', compress_type: 'CompressType', inplace=False, use_A2=True, compress_opts=None,
              is_H=False, init_guess: 'GridTN' = None, verbose_output=False, **kwargs
              ) -> Union[tuple['GridTN', float, bool], 'GridTN']:
        """ solves Ax=b using local
            self is b, operator is A, returns x
        """
        raise NotImplementedError

    def add(self, gtn_mpx2: 'GridTN', zipup=False, inplace=False, compress_type=CompressType.SVD,
            compress=False, compress_opts=None, **kwargs) -> 'GridTN':
        """ Add other grid_mpx (of the same type and on the same grid) to self
        """
        gtn = self if inplace else self.copy()
        if gtn.data is None:
            gtn.data = gtn_mpx2.data
        else:
            gtn.data += gtn_mpx2.data
        return gtn

    def add_list(self, *other_gtns: 'GridTN', compress_opts=None) -> 'GridTN':

        max_bond = compress_opts.get('max_bond', None) if compress_opts is not None else None
        cutoff = compress_opts.get('max_bond', CUTOFF) if compress_opts is not None else CUTOFF

        gtn = self.copy()
        is_mps = self.data_type == DataType.MPS

        ix = 0
        if gtn.data is None:
            gtn_iter = iter(other_gtns)
            while gtn.data is None:
                gtn = next(gtn_iter)
                ix += 1

        self_tt = self.grid.gtn_to_dmrg_format(gtn, is_mps=is_mps)
        other_gtns = other_gtns[ix:]
        other_tts = [self.grid.gtn_to_dmrg_format(ogtn) for ogtn in other_gtns]

        out = helper.add_MPS_list([self_tt, *other_tts], compress_opts={'max_bond': max_bond, 'cutoff': cutoff})
        gtn = self.grid.dmrg_to_gtn_format(gtn, out, is_mps=is_mps)
        return gtn


    def add_dmrg(self, *other_gtns: 'GridTN', inplace=False, compress_opts=None, **dmrg_opts) -> 'GridTN':
        """ add multiple gtns together using DMRG solver
        """
        from local_solvers.local_dmrg_eval import local_dmrg_evaluator, Term_DMRG

        DMAX = compress_opts.get('max_bond', None) if compress_opts is not None else None
        print('add drmg compress_opts', compress_opts)
        # exit()

        gtn = self if inplace else self.copy()
        is_mps = self.data_type == DataType.MPS

        self_tt = self.grid.gtn_to_dmrg_format(gtn, is_mps=is_mps)
        if self_tt is None:
            print('gtn', gtn)
            gtn_iter = iter(other_gtns)
            while gtn.data is None:
                gtn = next(gtn_iter)

        terms = [Term_DMRG(self_tt.copy())] if self_tt is not None else []
        for ogtn in other_gtns:
            op_mps = self.grid.gtn_to_dmrg_format(ogtn, is_mps=is_mps)
            if op_mps is not None:
                terms += [Term_DMRG(op_mps.copy())]

        print('add func max bond?', DMAX)
        print('term bonds')
        for term in terms:
            print('term.ket', term.ket.max_bond())

        func_mps = local_dmrg_evaluator(terms, max_bond=DMAX, **dmrg_opts)

        gtn = self.grid.dmrg_to_gtn_format(gtn, func_mps, is_mps=is_mps)
        return gtn


    def add_subgtn(self, sub_gtn_mpx2: 'GridTN', open_bc=False, inplace=False, zipup=False, compress=False,
                   compress_opts=None, **kwargs) -> 'GridTN':
        raise NotImplementedError

    def sum_apply(self, mpos: Sequence['GridTN'], inplace=False, zipup=False, compress=False, compress_opts=None):
        """ apply MPOs and then take the sum
        """
        new_gtn = self if inplace else self.copy()

        out = None
        for mpo in mpos:
            comp = self.apply(mpo, inplace=False, zipup=zipup, compress=False)
            out = comp if out is None else out.add(comp, inplace=True, compress=False)

        if compress:
            out.compress(inplace=True, compress_opts=compress_opts)

        new_gtn.data = out.data
        return new_gtn


    def sum_apply_dmrg(self, mpos: Sequence['GridTN'], inplace=False, compress_opts=None, **dmrg_opts):
        """ apply MPOs and then take the sum
        """
        # new_gtn = self if inplace else self.copy()

        # out = None
        # for mpo in mpos:
        #     comp = self.apply(mpo, inplace=False, zipup=zipup, compress=False)
        #     out = comp if out is None else out.add(comp, inplace=True, compress=False)
        #
        # if compress:
        #     out.compress(inplace=True, compress_opts=compress_opts)
        #
        # new_gtn.data = out.data
        #
        from local_solvers.local_dmrg_eval import local_dmrg_evaluator, Term_DMRG

        DMAX = compress_opts.get('max_bond', None) if compress_opts is not None else None
        print('sum apply drmg compress_opts', compress_opts)

        gtn = self if inplace else self.copy()
        is_mps = self.data_type == DataType.MPS

        self_tt = self.grid.gtn_to_dmrg_format(gtn, is_mps=is_mps)
        ops = [self.grid.gtn_to_dmrg_format(op, is_mps=False) for op in mpos]
        term = Term_DMRG(self_tt.copy(), operators=ops)

        print('sum apply func max bond?', DMAX)

        func_mps = local_dmrg_evaluator([term], max_bond=DMAX, **dmrg_opts)

        gtn = self.grid.dmrg_to_gtn_format(gtn, func_mps, is_mps=is_mps)
        return gtn


    def scalar_multiply(self, scalar_const: Numeric, inplace=False) -> 'GridTN':
        raise NotImplementedError

    # def elemental_multiply(self, gtn_mps2: 'GridTN', inplace=False, compress=False, compress_opts=None) -> 'GridTN':
    #     raise NotImplementedError

    # @profile
    def elemental_multiply(self, gtn_mps2: 'GridTN', inplace=False, zipup=False, compress_type=CompressType.SVD,
                           compress=False, compress_opts=None, sub_compress_opts=None, add_cc=False) -> 'GridTN':
        ### changed add_cc default from True to False

        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        if gtn_mps2 is None or gtn_mps2.data is None:
            if inplace:
                self.data = None
                return self
            else:
                return self.create_like(new_data=None)

        # # print(self.data, gtn_mps2.data)
        # if self.data_type is DataType.Num or gtn_mps2.data_type is DataType.Num:
        #     if self.data_type is DataType.Num and gtn_mps2.data_type is DataType.Num:
        #         new_data = self.data * gtn_mps2.data
        #         ref = self
        #     elif self.data_type is DataType.Num:
        #         new_data = helper.scalar_multiply(gtn_mps2.data, self.data)
        #         ref = gtn_mps2.grid
        #         if inplace:
        #             self.grid = gtn_mps2.grid
        #     elif gtn_mps2.data_type is DataType.Num:
        #         new_data = helper.scalar_multiply(self.data, gtn_mps2.data)
        #         ref = self
        #     else:
        #         raise ValueError
        #
        #     if inplace:
        #         self.data = new_data
        #         return self
        #     else:
        #         out = ref.create_like(new_data=new_data)
        #         return out

        grid_mpx1 = self if inplace else self.copy()

        if isinstance(gtn_mps2, qtn.MatrixProductState):
            mps2_shape = tuple([gtn_mps2.phys_dim(i) for i in gtn_mps2.L])
            assert (self.grid.shape == mps2_shape), 'mps2 needs to be the same shape as self.grid'
            gtn_mps2 = self.create_like(new_data=gtn_mps2)
            if self.data_type == DataType.MPS:  ## convert gtn_mps2 to MPO
                grid_mpo2 = gtn_mps2.apply_elemental_multiply_op()
            else:  ## use fact that self is MPO. apply method should take care of it
                grid_mpo2 = grid_mpx1
                grid_mpx1 = gtn_mps2

        elif isinstance(gtn_mps2, qtn.MatrixProductOperator):
            mps2_shape = tuple([gtn_mps2.phys_dim(i) for i in gtn_mps2.L])
            assert (self.grid.shape == mps2_shape), 'mps2 needs to be the same shape as self.grid'
            grid_mpo2 = self.create_like()
            grid_mpo2.data = gtn_mps2

        elif isinstance(gtn_mps2, GridTN):

            # print('gtn_mps2 is GTN', grid_mpx1.grid.axes, gtn_mps2.grid.axes)
            if grid_mpx1.grid != gtn_mps2.grid:
                if set(grid_mpx1.grid.axes).issubset(set(gtn_mps2.grid.axes)):
                    # print('mps1 < mps2')
                    grid_mpx1 = gtn_mps2.grid.pad_gtn_to_grid(grid_mpx1)
                    print('warning: elemental multiply not an inplace operation')
                elif set(gtn_mps2.grid.axes).issubset(set(grid_mpx1.grid.axes)):
                    # print('mps1 > mps2')
                    gtn_mps2 = grid_mpx1.grid.pad_gtn_to_grid(gtn_mps2, target_data_type=DataType.MPO)
                else:
                    raise ValueError(f'grid compatibility issue with {grid_mpx1.grid}, {gtn_mps2.grid}')

            # print('gtn_mps2 data type', gtn_mps2.data_type)
            # print('gtn', gtn_mps2)
            # print(gtn_mps2.__class__)
            if gtn_mps2.data_type == DataType.MPS:
                # assert self.grid == gtn_mps2.grid, 'grid mps2 needs to be the same axes as self'

                # if self.data_type == DataType.MPO:
                #     print('WARNING: ELEMENTAL MULTIPLY CHANGED FOR MPO * MPS')
                #
                # grid_mpo2 = gtn_mps2.apply_elemental_multiply_op()

                gtn_mps2 = gtn_mps2
                if self.data_type == DataType.MPS:  ## convert gtn_mps2 to MPO
                    grid_mpo2: 'GridTN' = gtn_mps2.apply_elemental_multiply_op()
                else:  ## use fact that self is MPO. apply method should take care of it
                    ## this does not accurately capture elemental multiply for arbitrary grid_mpx1 MPOs
                    raise NotImplementedError('maybe elem mult MPO * MPS needs to be redefined')
                    grid_mpo2 = grid_mpx1
                    grid_mpx1 = gtn_mps2

            elif isinstance(gtn_mps2, GridTN) and gtn_mps2.data_type == DataType.MPO:
                # assert self.grid == gtn_mps2.grid, 'grid mps2 needs to be the same axes as self'
                grid_mpo2 = gtn_mps2

        else:
            raise TypeError(f'other must be GridTN with MPS or MPO data type, not {type(gtn_mps2)}')

        # print('elem mult', grid_mpx1.max_bond(), grid_mpo2.max_bond())
        grid_mpx1 = grid_mpx1.apply(grid_mpo2, inplace=True, zipup=zipup, compress_type=compress_type,
                                    compress=compress,
                                    compress_opts=compress_opts, add_cc=add_cc, sub_compress_opts=sub_compress_opts)
        # max_bond = compress_opts.get('max_bond', None)
        # grid_mpx1 = grid_mpx1.apply_rdm(grid_mpo2, inplace=True, compress_opts=compress_opts, compress=(max_bond is None))

        # add_cc used to be True; probably for real Fourier basis??
        return grid_mpx1

    def xmultiply(self, x_axes: Sequence['Axis'], x_power: int = 1, offsets: dict[Any, Numeric] = 0.0,
                  scales: dict[Any, Numeric] = 1.0, inplace=False, zipup=False, compress_type=CompressType.SVD,
                  compress=False, compress_opts=None) -> 'GridTN1D':

        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        grid_mpx1 = self if inplace else self.copy()
        xmult_mpo = grid_mpx1.grid.get_xmultiply_mpo(x_axes, x_power=x_power, offsets=offsets, scales=scales)

        # print('XMULT GTN', compress_type)
        grid_mpx1.apply(xmult_mpo, inplace=True, zipup=zipup, compress_type=compress_type,
                        compress=compress, compress_opts=compress_opts)
        # if compress:
        #     grid_mpx1 = grid_mpx1.compress(inplace=True, compress_opts=compress_opts)

        return grid_mpx1

    def take_qft(self, qft_axes=None, inverse=False, inplace=False, compress=True, compress_opts=None):
        qft_axes = self.grid.axes if qft_axes is None else qft_axes
        # qft_ops = {ax: ax.get_qft_mpo_v2(inverse=inverse) for ax in qft_axes}
        if inverse:
            qft_ops = {ax: ax.get_inverse_qft_mpo() for ax in qft_axes}
        else:
            qft_ops = {ax: ax.get_qft_mpo() for ax in qft_axes}
        qft_mpo = self.grid.make_mpo_ndim(qft_ops)
        out = self.apply(qft_mpo, inplace=inplace, zipup=True, compress=compress, compress_opts=compress_opts)
        return out

    # def average_fine_scale(self, avg_axes=None, spread:int = 1, inplace=False, compress=True, compress_opts=None, mu=0.5):
    #     """
    #     apply average mpo along each of the axes specified by avg_axes
    #     spread: number of neighbors to include
    #     mu: weight on central element
    #
    #     x_j <-- \sum_i=1^spread (1-mu)/2 spread (x_(j-i) + x_(j+i)) + mu x_(j)
    #     eg if spread = 1:
    #         x_j <-- (1-mu)/2 x_j-1 + mu x_j + (1-mu)/2 x_j+1
    #     """
    #     print('performing fine scale avg')
    #     # raise RuntimeError
    #     avg_axes = self.grid.axes if avg_axes is None else avg_axes
    #     avg_ops = {ax: ax.get_averaging_mpo(spread=spread, mu=mu,
    #                                         boundary_conditions=self.ax_deriv_configs.get(ax, DerivativeConfiguration()))
    #                for ax in avg_axes}
    #     avg_mpo = self.grid.make_mpo_ndim(avg_ops)
    #     out = self.apply(avg_mpo, inplace=inplace, zipup=True, compress=compress, compress_opts=compress_opts)
    #     return out


    def average_fine_scale(self, avg_axes=None, spread:int = 1, inplace=False, compress=True, compress_opts=None, mu=0.5):
        """
        apply average mpo along each of the axes specified by avg_axes
        spread: number of neighbors to include
        mu: weight on central element

        x_j <-- \sum_i=1^spread (1-mu)/2 spread (x_(j-i) + x_(j+i)) + mu x_(j)
        eg if spread = 1:
            x_j <-- (1-mu)/2 x_j-1 + mu x_j + (1-mu)/2 x_j+1
        """
        print('performing fine scale avg')
        # raise RuntimeError
        avg_axes = self.grid.axes if avg_axes is None else avg_axes
        avg_ops = {ax: ax.get_averaging_mpo(spread=spread, mu=mu,
                                            boundary_conditions=self.ax_deriv_configs.get(ax, DerivativeConfiguration()))
                   for ax in avg_axes}
        avg_mpo = self.grid.make_mpo_ndim(avg_ops)
        out = self.apply(avg_mpo, inplace=inplace, zipup=True, compress=compress, compress_opts=compress_opts)
        return out

    def add_dissipation(self, strength: Numeric, deriv_order: int=2, deriv_axes=None, inplace=False,
                        compress=True, compress_opts:dict = None, verbose_plot=False):
        """
        Euler time step of dissipation:  f = f + eta d^m/dx^m f = (1 + eta d^m/dx^m) f
        for stability, assumes that the strength is small: eta << dt/dx^2
        strength: strength of dissipation (eta)
        deriv_order: order of derviative. e.g. order=1 --> m=2
            order = 1 (m=2): f_j + eta (f_{j+1} - 2 f_{j} + 1 f_{j-1})
            order = 2 (m=4): f_j + eta (-f_{j+2} + 4 f_{j+1} - 6f_j + 4f_{j-1} - f_{j-2})
            order = 3 (m=6): f_j + eta (f_{j+3} - 6 f_{j+2} + 15 f_{j+1} - 20 f_{j} + 15 f_{j-1} - 6 f_{j-2} + f_{j+3}
            i.e. mth derivative with 2nd order finite difference stencil
        """
        if verbose_plot:
            old = self.copy()

        dissip_f = self.get_dissipation(strength, deriv_order, deriv_axes=deriv_axes, compress=False)
        out = self.add(dissip_f, inplace=inplace, compress=compress, compress_opts=compress_opts)

        if verbose_plot:
            if old.data is not None:
                print('dissip diff', old.distance(out) / old.frobenius_norm())

                diff_data = dissip_f.get_data()  # old.add(out.scalar_multiply(-1)).get_data()
                plt.figure()
                # plt.imshow(np.log10(np.abs(diff_data)))
                plt.imshow(diff_data)
                plt.colorbar()
                plt.title('dissipation diff')
                plt.show()

        return out

    def get_dissipation(self, strength: Numeric, deriv_order: int=2, deriv_axes=None,
                        compress=True, compress_opts:dict = None):
        """
        Euler time step of dissipation:  f = f + eta d^m/dx^m f = (1 + eta d^m/dx^m) f
        for stability, assumes that the strength is small: eta << dt/dx^2
        strength: strength of dissipation (eta)
        deriv_order: order of derviative. e.g. order=1 --> m=2
            order = 1 (m=2): f_j + eta (f_{j+1} - 2 f_{j} + 1 f_{j-1})
            order = 2 (m=4): f_j + eta (-f_{j+2} + 4 f_{j+1} - 6f_j + 4f_{j-1} - f_{j-2})
            order = 3 (m=6): f_j + eta (f_{j+3} - 6 f_{j+2} + 15 f_{j+1} - 20 f_{j} + 15 f_{j-1} - 6 f_{j-2} + f_{j+3}
            i.e. mth derivative with 2nd order finite difference stencil
        """
        # assert(strength <= 1 / deriv_order ** 2), f'smoothing strength {strength} is too large'  # see p.114 in Durran
        assert(deriv_order % 2 == 0), f'deriv_order must be even, not {deriv_order}'

        ax_deriv_configs = {k: v.copy() for k,v in self.ax_deriv_configs.items()}
        for k, v in ax_deriv_configs.items():
            v.update(order=1, fd_type=FDType.CENTER)

        if deriv_order % 4 == 0:
            strength = strength * -1

        dissip_f = self.take_mth_laplacian(deriv_order, deriv_axes=deriv_axes, ax_deriv_configs=ax_deriv_configs,
                                           remove_dx2=True, compress=compress, compress_opts=compress_opts)
        dissip_f = dissip_f.scalar_multiply(strength, inplace=True)

        return dissip_f


    def average_neighbor(self, avg_axes=None, spread:int = 1, inplace=False, compress=True, compress_opts=None, mu=0.5):
        """
        apply average mpo along each of the axes specified by avg_axes
        spread: number of neighbors to include
        mu: weight on central element

        x_j <-- \sum_i=1^spread (1-mu)/2 spread (x_(j-i) + x_(j+i)) + mu x_(j)
        eg if spread = 1:
            x_j <-- (1-mu)/2 x_j-1 + mu x_j + (1-mu)/2 x_j+1
        """
        avg_axes = self.grid.axes if avg_axes is None else avg_axes
        avg_ops = {ax: ax.get_neighbor_avg_mpo(spread=spread, mu=mu,
                                               boundary_conditions=self.ax_deriv_configs.get(ax, DerivativeConfiguration()))
                   for ax in avg_axes}
        avg_mpo = self.grid.make_mpo_ndim(avg_ops)
        out = self.apply(avg_mpo, inplace=inplace, zipup=True, compress=compress, compress_opts=compress_opts)
        return out

    def average_leapfrog(self, avg_axes=None, inplace=False, compress=True, compress_opts=None):
        avg_axes = self.grid.axes if avg_axes is None else avg_axes
        # avg_ops = {ax: ax.get_leapfrog_mpo(boundary_conditions=self.ax_deriv_configs.get(ax, DerivativeConfiguration()))
        #            for ax in avg_axes}
        # avg_mpo = self.grid.make_mpo_ndim(avg_ops)
        avg_mpo = self.grid.get_leapfrog_mpo(avg_axes, self.ax_deriv_configs)
        out = self.apply(avg_mpo, inplace=inplace, zipup=True, compress=compress, compress_opts=compress_opts)
        return out

    def shift_cell_to_midpoint(self, shift_axes=None, inplace=False, compress=True, compress_opts=None,
                               inverse=False):
        """ compute 1/2 (x_i + x_{i+1})
            or 1/2(x_i + x_{i-1}) if inverse=True
        """
        shift_axes = self.grid.axes if shift_axes is None else shift_axes
        print('shift axes', shift_axes)

        shift = -1 if inverse else 1
        shift_op = {ax: ax.get_shift_mpo(shift,
                                         boundary_conditions=self.ax_deriv_configs.get(ax, DerivativeConfiguration()))
                    for ax in shift_axes}
        shift_mpo = self.grid.make_mpo_ndim(shift_op)
        avg_mpo = self.grid.get_iden_mpo().add(shift_mpo)
        avg_mpo.scalar_multiply(0.5, inplace=True)
        out = self.apply(avg_mpo, inplace=inplace, zipup=True, compress=compress, compress_opts=compress_opts)

        new_deriv_configs = {}
        for ax, dc in self.ax_deriv_configs.items():
            new_deriv_configs[ax] = dc.shifted_bc(inverse=inverse) if ax in shift_axes else dc.copy()
        out.ax_deriv_configs = new_deriv_configs

        return out

    def evaluate_func(self, func: Callable, max_bond=None, inplace=False):
        raise NotImplementedError

    def canonize(self, inplace=True, scale=True, form='right', i=None, cur_orthog=None) -> 'GridTN':
        raise NotImplementedError

    def canonize_axes(self, axes: Sequence['Axis'], inplace=True, scale=True) -> 'GridTN':
        raise NotImplementedError

    def get_canon_site_from_axes(self, axes: Sequence['Axis']):
        raise NotImplementedError

    def compress(self, inplace=True, verbose=False, canonize=True, compress_type=CompressType.SVD,
                 compress_opts: dict = None, sub_compress_opts: Optional[dict['SubCompressConfigType', dict]] = None,
                 norm_cutoff: float = None, conservative=False, **kwargs):
        raise NotImplementedError

    def compress_rdm(self, inplace=True, verbose=False, compress_opts=None, sub_compress_opts=None, direction=1,
                     open_end=False, left_env=None, right_env=None, back_compress=True,
                     **kwargs):
        raise NotImplementedError

    def get_bases(self, i: int):
        """ get the basis functions of the mps assuming the orthogonality center is at site i
        """
        raise NotImplementedError

    def contract(self):
        return self.data.contract()

    def scalar_add(self, scalar_val, inplace=False):
        """ add scalar values to gridTN
        """
        gtn = self if inplace else self.copy()
        ones = self.grid.get_ones_mps()
        ones.scalar_multiply(scalar_val, inplace=True)
        gtn = gtn.add(ones, inplace=True)
        return gtn

    # def apply_partial_mpx(self, mpo: MPOType or MPSType or 'GridTN', reset_grid=True, compress=True,
    #           **compress_opts) -> 'GridTN':
    #     raise NotImplementedError

    def apply_qft(self, ft_axes=None, inplace=False, compress=False, compress_opts=None):
        gtn = self if inplace else self.copy()

        if gtn.data_type == DataType.MPS:
            qft_mpo = self.grid.get_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?
            gtn.apply(qft_mpo, inplace=True, zipup=True, compress=compress, compress_opts=compress_opts)
        elif gtn.data_type == DataType.MPO:
            qft_mpo = self.grid.get_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?
            iqft_mpo = self.grid.get_inverse_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?
            gtn_ = iqft_mpo.apply(gtn, zipup=True)
            gtn_.apply(qft_mpo, inplace=True, zipup=True, compress=compress, compress_opts=compress_opts)
            gtn.data = gtn_.data
        else:
            raise NotImplementedError

        return gtn

    def apply_inverse_qft(self, ft_axes=None, inplace=False, compress=False, compress_opts=None):
        gtn = self if inplace else self.copy()

        if gtn.data_type == DataType.MPS:
            qft_mpo = self.grid.get_inverse_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?
            gtn.apply(qft_mpo, inplace=True, zipup=True, compress=compress, compress_opts=compress_opts)
        elif gtn.data_type == DataType.MPO:
            print('inverse qft mpo')
            qft_gtn = self.grid.get_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?
            inv_qft_gtn = self.grid.get_inverse_qft_mpo(ft_axes=ft_axes)  ## probably not compressible?

            gtn_1 = qft_gtn.apply(gtn, zipup=True)
            gtn_1.apply(inv_qft_gtn, inplace=True, zipup=True, compress=True, compress_opts=compress_opts)
            gtn.data = gtn_1.data
        else:
            raise NotImplementedError

        return gtn

    #######################
    ###  MPS_USVT fcts  ###
    #######################

    def convert_to_USVT(self, canon_site, inplace=False, cur_orthog=None):
        raise NotImplementedError

    def convert_from_USVT(self, inplace=False, canon_site=None):
        raise NotImplementedError

    def get_S_tensor(self, **kwargs):
        raise NotImplementedError

    ################
    ## processing ##
    ################

    def _coarsen_grid(self, depth: int, method: 'str', is_sqrt=False) -> 'GridTN':

        grid = self.grid
        Axis = grid.axes[0].__class__
        new_axes = [Axis(ax.L - depth, dx=depth * ax.dx, x0=ax.xpts[0] + depth * ax.dx / 2,
                         ax_map=ax.map, coordinate=ax.coordinate, is_flipped=ax.is_flipped, basis=ax.basis)
                    for ax in grid.axes]
        new_grid = grid.create_like(new_axes)

        if method=='select':
            coarse_grain_op = grid.get_coarse_select_mpx(depth, is_sqrt=False)
        else:
            coarse_grain_op = grid.get_coarse_grain_mpx(depth, is_sqrt=is_sqrt)

        # L_ = grid.L - depth * len(new_axes)
        gtn_coarse = self.apply(coarse_grain_op)
        coarse_mps = gtn_coarse.data
        site_ind_id = self.data.site_ind_id
        site_tag_id = self.data.site_tag_id

        ## contract coarse tensors
        # if no site_ind, merge with next tensor/tensor on the right
        mps_idxs = []
        for i in range(grid.L):
            tens = coarse_mps[i]
            if site_ind_id.format(i) not in tens.inds:
                next_tens = coarse_mps[mps_idxs[-1]] if i == grid.L - 1 else coarse_mps[i + 1]
                new_tens = qtn.tensor_contract(tens, next_tens)
                next_tens.modify(data=new_tens.data, inds=new_tens.inds)
                coarse_mps.delete(site_tag_id.format(i), which='all')
            else:
                mps_idxs += [i]

        coarse_mps._L = coarse_mps.num_tensors
        coarse_mps.exponent = self.data.exponent + coarse_grain_op.exponent
        helper.renumber_mps(coarse_mps, mps_idxs, list(range(coarse_mps._L)), inplace=True )

        coarse_gtn = self.__class__(new_grid, coarse_mps)
        return coarse_gtn

    def coarsen_grid_integrate(self, depth: int, is_sqrt=False) -> 'GridTN':
        return self._coarsen_grid(depth, 'integrate', is_sqrt=is_sqrt)

    def coarsen_grid_select(self, depth: int) -> 'GridTN':
        return self._coarsen_grid(depth, 'select')


    def interpolate_data(self, depth: int, interp_dict: dict['Axis': 'qtn.MatrixProductOperator'],
                         new_grid: 'Grid'=None):
        """ extend grid s.t. data is interpolated between existing grid points
            only works for outer-product interpolation schemes
            interp_dict: dict[Axis, qtn.MatrixProductOperator]
                the tensor has all the indices appropriately
                decomposed on the coarse grid (all fine indices are labeled as "fine")
                though perhaps not numbered appropriately.
        """
        raise NotImplementedError

    def gaussian_smooth_data(self, sigma=0.3, proc_axes: Optional[Sequence['Axis']] = None, mode='wrap',
                             inplace=False, compress=True, compress_opts=None) -> 'GridTN':
        """ process the data in some way
        """
        gtn = self if inplace else self.copy(deep=False)
        blur_mpo = self.grid.gaussian_smooth_mpo(sigma=sigma, proc_axes=proc_axes, mode=mode, split_opts=compress_opts)
        # leave out compress_opts for default (most accurate) compression if compress=True

        if blur_mpo is not None:
            gtn = gtn.apply(blur_mpo, inplace=True, compress=compress, compress_opts=compress_opts)  ## not deep inplace
        return gtn

    def apply_absorbing_bc(self, x_ax: 'Axis', v_ax: 'Axis', inplace=False, compress=True, compress_opts=None) \
            -> 'GridTN':
        """ apply absorbing bc to MPS state, assumes TN dimensions are x1,x2,x3,v1,v2,v3
            (prev: apply absorbing bc to derivative MPO, assumes TN dimensions are x1,x2,x3,v1,v2,v3)
        """
        gtn = self if inplace else self.copy(deep=False)
        left_bc, right_bc = gtn.ax_deriv_configs[x_ax].bc

        if left_bc == BCType.ABSORBING or right_bc == BCType.ABSORBING:
            absorb_bc_mpo = self.grid.absorbing_bc_mpo(x_ax, v_ax, left_bc, right_bc, compress=False)
            # print(absorb_bc_mpo, absorb_bc_mpo.data)
            # leave out compress_opts for default (most accurate) compression if compress=True

            # if absorb_bc_mpo is not None: # and absorb_bc_mpo.data is not None:
            gtn = gtn.apply(absorb_bc_mpo, inplace=True, compress=compress, compress_opts=compress_opts)

        return gtn

    def apply_reflecting_v_bc(self, x_ax, v_ax, left_bc, right_bc, inplace=False, compress=True, compress_opts=None):
        """ apply reflecting_v bc to state.
            apply to derivative MPO?
        """
        gtn = self if inplace else self.copy()
        reflect_bc_mpo = self.grid.reflecting_v_bc_mpo(x_ax, v_ax, left_bc, right_bc, compress=False)
        # leave out compress_opts for default (most accurate) compression if compress=True

        if reflect_bc_mpo is not None:
            if compress_opts is None:   compress_opts = {}
            gtn = gtn.apply(reflect_bc_mpo, inplace=True, compress=compress, compress_opts=compress_opts)
        return gtn

    ###############################################
    ### build derivative and integral operators ###
    ###############################################

    def mps_to_diag_mpo(self, lower_ind_id='i({})', upper_ind_id='o({})', inplace=False) -> 'GridTN':
        """ convert MPS into MPO
        """
        raise NotImplementedError

    def apply_elemental_multiply_op(self, lower_ind_id='i({})', upper_ind_id='o({})', axes=None, add_cc=False,
                                    take_mps_cc=False, compress=False, compress_opts=None):
        """ apply elemental multiply operator to self (e.g. convolution or d_ijk fct)
        """
        raise NotImplementedError

    # def apply_xmultiply_mpo(self, x_axes: list['Axis'] or None = None, offsets: dict['Axis', Numeric] = 0.0,
    #                         scales: dict['Axis', Numeric] = 1.0,inplace=False, compress=False, compress_opts=None):
    #     """ perform x*f(x) operation, where x_dims specifies which axis/axes to multiply
    #         eg. 1D:  x*f(x,y,...)
    #             2D:  x*y*f(x,y,...)
    #     """
    #     if self.data is None:
    #         return self
    #
    #     gtn = self if inplace else self.copy()
    #     x_mpo = self.grid.get_xmultiply_mpo(x_axes, offsets=offsets, scales=scales)
    #     if compress_opts is None:   compress_opts = {}
    #     gtn.apply(x_mpo, compress=compress, compress_opts=compress_opts)
    #     return gtn

    ## put bc into deriv_params to avoid carrying it around when calling curl, laplacian, etc.
    # @profile
    def take_firstderivative(self, ax: 'Axis', upwind_ax: Optional['Axis'] = None,
                             ax_deriv_config: dict['Axis', 'DerivativeConfiguration'] = None,
                             recalc=False, inplace=False, compress=False, compress_opts=None):
        """ df/dx_i
            bc = boundary condition
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        gtn = self if inplace else self.copy()
        deriv_config = gtn.ax_deriv_configs[ax] if ax_deriv_config is None else ax_deriv_config[ax]
        ddx_mpo = self.grid.get_firstderivative_mpo(ax, deriv_config=deriv_config, upwind_ax=upwind_ax, recalc=recalc)
        gtn.apply(ddx_mpo, inplace=True)

        if compress:
            gtn = gtn.compress(inplace=True, compress_opts=compress_opts)
        return gtn

    def take_secondderivative(self, ax1: 'Axis', ax2: 'Axis' or None,
                              recalc=False, inplace=False, compress=False, compress_opts=None):
        """ self.order: finite difference expansion order """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        gtn = self if inplace else self.copy()
        deriv_config1 = gtn.ax_deriv_configs[ax1]
        deriv_config2 = gtn.ax_deriv_configs[ax2] if ax2 is not None else None
        d2_mpo = self.grid.get_secondderivative_mpo(ax1, ax2, deriv_config1=deriv_config1, deriv_config2=deriv_config2,
                                                    recalc=recalc)
        gtn = gtn.apply(d2_mpo, inplace=True, compress=compress, compress_opts=compress_opts)
        return gtn

    def take_mth_derivative(self, deriv_order: int, ax: 'Axis', deriv_config: 'DerivativeConfiguration'=None,
                            recalc=False, inplace=False, compress=False, compress_opts=None):
        """ take mth derivative d^m/dx^m along specified axis
        """
        if self.data is None:
            return self if inplace else self.create_like(new_data=None)

        gtn = self if inplace else self.copy()
        deriv_config1 = gtn.ax_deriv_configs[ax] if deriv_config is None else deriv_config
        d2_mpo = self.grid.get_mth_derivative_mpo(ax, deriv_order, deriv_config=deriv_config1, recalc=recalc)
        gtn = gtn.apply(d2_mpo, inplace=True, compress=compress, compress_opts=compress_opts)
        return gtn

    def take_mth_laplacian(self, deriv_order: int, deriv_axes=None, coord_sys: 'CoordinateSystem'=None,
                           ax_deriv_configs: dict['Axis': 'DerivativeConfiguration'] = None, remove_dx2=False,
                           recalc=False, inplace=False, compress=False, compress_opts=None
                           ) -> 'GridTN':
        """ compute d^m/dx^m f along all deriv_axes (valid for even m)
        """

        if coord_sys is not None:
            mpo_list = coord_sys.build_mth_laplacian_mpo(deriv_order, axes=deriv_axes,
                                                         deriv_configs=self.ax_deriv_configs,)
            raise NotImplementedError

        if ax_deriv_configs is None:
            ax_deriv_configs = self.ax_deriv_configs

        if deriv_axes is None:
            deriv_axes = self.grid.axes

        out_mps = None
        for ax in deriv_axes:
            out = self.take_mth_derivative(deriv_order, ax, deriv_config=ax_deriv_configs[ax])
            if remove_dx2:
                if ax.basis.type == BasisType.FOURIER:
                    dx2_coeff = (1. / np.abs(ax.xpts[0]))**deriv_order
                elif ax.basis.type == BasisType.SPATIAL:
                    dx2_coeff = ax.dx**deriv_order
                else:
                    raise NotImplementedError
                out.scalar_multiply(dx2_coeff, inplace=True)

            if out_mps is None:
                out_mps = out
            else:
                out_mps = out_mps.add(out, inplace=True, compress=False)

        if compress and out_mps is not None:
            out_mps = out_mps.compress(compress_opts=compress_opts, inplace=True)

        return out_mps


    # def take_mth_laplacian(self, deriv_order: 'int', axes: Sequence['Axis']=None,
    #                         recalc=False, inplace=False, compress=False, compress_opts=None):
    #     """
    #     """
    #     if self.data is None:
    #         return self if inplace else self.create_like(new_data=None)
    #
    #     tmp = None  # self if inplace else self.copy()
    #     deriv_config1 = gtn.ax_deriv_configs[ax]
    #     d2_mpo = self.grid.get_mth_derivative_mpo(ax, deriv_order, deriv_config=deriv_config1, recalc=recalc)
    #
    #     for ax in
    #     gtn = self.apply(d2_mpo, inplace=False, compress=compress, compress_opts=compress_opts)
    #
    #     return gtn


    def integrate(self, integ_axes: Sequence['Axis'] = None, is_sqrt=False, ancilla_reindex: dict[str, str] = None,
                  new_grid: 'Grid' = None, new_ax_deriv_configs: dict[Any, 'DerivativeConfiguration'] = None,
                  exclude_weights=False, compress=False, compress_opts=None):
        raise NotImplementedError

    def meas_elem(self, sel_inds: Sequence[int], site_ind_id='i({})', site_tag_id='X({})',):
        raise NotImplementedError

    def meas_shifted_elem(self, sel_inds: Sequence[int], shifts: dict['Axis', int],
                          ax_deriv_configs: ['Axis', 'DerivativeConfiguration'],):
        raise NotImplementedError


    def meas_expec(self, obs_gtn, integ_axes=None, is_sqrt=False, new_grid=None, exclude_axes=None,
                   exclude_weights=False, new_ax_deriv_configs: dict[Any, 'DerivativeConfiguration'] = None,
                   compress=False, compress_opts=None, **kwargs) -> Numeric:
        raise NotImplementedError

    def project(self, obs_gtn, proj_axes, new_grid=None, canonize=True, compress=False, compress_opts=None,
                new_ax_deriv_configs=None, **kwargs) -> 'GridTN':
        """ project obs_gtn onto manifold of self
        """
        raise NotImplementedError

    def project_bond(self, obs_gtn, bond_ind) -> 'qtn.Tensor':
        raise NotImplementedError

    def project_site(self, obs_gtn, site_ind, nsites=1) -> 'qtn.Tensor':
        raise NotImplementedError

    def project_op_bond(self, obs_gtn, bond_ind, left_env: 'qtn.Tensor' = None, right_env: 'qtn.Tensor' = None,
                        **kwargs) -> tuple['qtn.Tensor', Sequence[str], Sequence[str]]:
        raise NotImplementedError

    def project_op_site(self, obs_gtn, site_ind, nsites=1, left_env: 'qtn.Tensor' = None,
                        right_env: 'qtn.Tensor' = None) \
            -> tuple['qtn.Tensor', Sequence[str], Sequence[str]]:
        raise NotImplementedError

    def stagger_grid(self, stagger_vals: dict['Axis', int], inplace=False, compress=False, compress_opts=None):
        """
        stagger_vals: stagger the grid along each axis by specified value (in 1/2 grid point increments)
        """

        gtn = self if inplace else self.copy()

        stagger_mpos = {}
        for ax, val in stagger_vals.items():
            if val != 0:
                deriv_config = gtn.ax_deriv_configs[ax]
                shift = val // 2
                avg = val % 2

                stagger_mpo = ax.get_shift_mpo(shift, boundary_conditions=deriv_config)
                if avg:  ## 1/2 grid step discretization
                    stagger_mpo_2 = ax.get_shift_mpo(shift + 1, boundary_conditions=deriv_config)
                    helper.add_MPO(stagger_mpo, stagger_mpo_2, inplace=True, compress=False)
                    helper.scalar_multiply(stagger_mpo, 0.5, inplace=True)

                if not (deriv_config.left_bc == BCType.PERIODIC or deriv_config.left_bc == BCType.ANTIPERIODIC
                        or deriv_config.left_bc != BCType.OPEN):
                    deriv_config.offset = deriv_config.offset + val

                stagger_mpos[ax] = stagger_mpo

        # orig_data = gtn.get_data()

        stagger_mpo = gtn.grid.make_mpo_ndim(stagger_mpos)
        gtn = gtn.apply(stagger_mpo, inplace=True, compress=compress, compress_opts=compress_opts)

        # new_data = gtn.get_data()

        # plt.figure()
        # plt.plot(orig_data)
        # plt.plot(new_data)
        # plt.show()

        return gtn

    def evolve_tdvp(self, dt, mpo_list, te_order=0, do_adapt=False, inplace=False, compress_config=None,
                    expand_basis: Sequence['GridTN'] = None):
        raise NotImplementedError

    def evolve_tdmrg(self, dt, mpo_list, te_order=0, do_adapt=True, inplace=False, compress_config=None):
        raise NotImplementedError

    def evolve_tdvp_new(self, dt, linear_mpo_list, te_order=0, do_adapt=True, inplace=False, compress_config=None,
                        nonlinear_terms = None, sources=None, solver_type=LocalSolverType.TDDMRG,
                       filter_bases=False, time=None):
        raise NotImplementedError

    def evolve_tdmrg_new(self, dt, linear_mpo_list, te_order=0, do_adapt=True, inplace=False, compress_config=None,
                         nonlinear_terms = None, sources=None, solver_type=LocalSolverType.TDDMRG,
                         filter_bases=False, verbose_plot=False, time=None, direction=1,
                         upwind_func=None, upwind_deriv_func=None,
                         **kwargs):
        raise NotImplementedError

    def evolve_time_local_global(self, dt, linear_mpo_list, te_order=0, do_adapt=True, inplace=False, compress_config=None,
                         nonlinear_terms = None, sources=None, solver_type=LocalSolverType.TDDMRG,
                         filter_bases=False, time=None):
        raise NotImplementedError

    # def evolve_tdvp0(self, dt, target_list, te_order=0, do_adapt=True, inplace=False, compress_config=None):
    #     raise NotImplementedError

    ##########################
    ## sqrt field functions ##
    ##########################

    def get_complex_conj(self, inplace=False) -> 'GridTN':
        ## needs to be overwritten within each class
        # out = self if inplace else self.copy()
        raise NotImplementedError

    def mult_cc(self, inplace=False):
        cc = self.get_complex_conj()
        out = self.elemental_multiply(cc, inplace=inplace)
        return out


#########################
#### class functions ####
#########################

def add(gtn1: 'GridTN', gtn2: 'GridTN', inplace=False, **kwargs):
    return gtn1.add(gtn2, inplace=inplace, **kwargs)


def add_dmrg(*gtns: 'GridTN', inplace=False, **kwargs):
    return gtns[0].add_dmrg(gtns[1:], inplace=inplace, **kwargs)


def scalar_multiply(gtn1: 'GridTN', const: 'Numeric', inplace=False, **kwargs):
    return gtn1.scalar_multiply(const, inplace=inplace, **kwargs)


def euler(gtn1: 'GridTN', dt, deriv0=None, inplace=False, **kwargs):
    if deriv0 is None:
        return gtn1 if inplace else gtn1.copy()

    return gtn1.add(deriv0.scalar_multiply(dt), inplace=inplace, **kwargs)


def compute_derivative(gtn1: 'GridTN', derivative_mpos: Sequence['GridTN'], **kwargs):
    return gtn1.sum_apply(derivative_mpos, **kwargs)
