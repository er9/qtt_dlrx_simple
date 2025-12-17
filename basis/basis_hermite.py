import numpy as np
from functools import lru_cache
import scipy.special
from scipy import integrate
import matplotlib.pyplot as plt

from setup_.enums import BasisType
from setup_.defaults import *
from basis.basis import Basis

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from axis import Axis
    from typing import Callable


class HermiteBasis(Basis):
    """ physicist Hermite polynomials
                  H_m(x) = (-1)^m exp(x^2) d^n/dx^n exp(-x^2)
                  orthogonal (int dx H_m(x) H_n(x) = delta_m,n) * sqrt(2*pi)*n!
                  x * H_m(x)  = m * H_(m-1)(x) + 1/2 * H_(m+1)(x)
                  d/dx H_m(x) = 2*m * H_(m-1)(x)
        Not sure how well this is tested?
        I think I previously used AWHermiteGauss
    """
    def __init__(self, x_offset=0.0, x_scale=1.0):
        """ H(y) = physicist's Hermite polynomials?
            y = (x-x_offset)/x_scale
        """
        super().__init__()
        self.x_offset = x_offset
        self.x_scale  = x_scale
        self.type = BasisType.HERMITE


    # def from_realspace_1D(self, func: 'Callable', num_modes: int):
    #
    #     ms = np.arange(num_modes)
    #     norms = 1. / np.sqrt(2 ** ms * scipy.special.factorial(ms) * np.pi)
    #
    #     coeffs = []
    #     for n in range(num_modes):
    #
    #         def func_hermite(x):
    #             y = (x - self.x_offset) / self.x_scale
    #             return func(y) * scipy.special.eval_hermite(n, y)
    #         func_hermite = lambda x: func(x) * scipy.special.eval_hermite(n, x)
    #         ## physicist's Hermite polynomials
    #
    #         ig, err = integrate.quad(func_hermite, -np.inf, np.inf)
    #
    #         norm = 1. / np.sqrt(2 ** n * scipy.special.factorial(n) * np.pi)
    #         coeffs += [ ig / norm ]
    #
    #
    #     raise NotImplementedError

    def get_realspace_1D(self, data, ax_ind, x0=0.0, xL=1.0, npts=128):
        """ data:  coefficients of Hermite polynomials (order m = 0 to M-1)
            x_window:  x0,xL specifying domain of interest
            x_offset:  offset of center of Hermite polynomials
            x_scale:   scaling of x in Hermite polynomials     (x <- x_scale*x + x_offset)
            note: numpy fct is physicists Hermite polynomials (no normalization)
        """
        domain = ((x0-self.x_offset)/self.x_scale, (xL-self.x_offset)/self.x_scale)

        data = np.moveaxis(data, ax_ind, -1)
        data_shape = data.shape
        data = data.reshape(-1, data_shape[-1])

        ms = np.arange(data_shape[-1])
        norms = 1. / np.sqrt(2 ** ms * scipy.special.factorial(ms) * np.pi)
        data = data * norms

        xs = np.linspace(x0, xL, npts)
        out_data = np.zeros((data.shape[0], npts), dtype=complex)
        for i in range(data.shape[-1]):
            hermite_poly = np.polynomial.hermite.Hermite(data[i, :], domain, (x0, xL))
            out_data[i, :] = hermite_poly.linspace(npts)[1]

        out_data = out_data.reshape(data_shape[:-1] + (npts,))
        out_data = np.moveaxis(out_data, -1, ax_ind)
        return out_data


    def get_ones_mps(self, ax: 'Axis', site_ind_id='i({})', site_tag_id='X({})', anc_dim=None, anc_name_l=None,
                     anc_name_r=None) -> 'MPSType':
        raise NotImplementedError


    def build_elemental_multiply_tn(self, ax: 'Axis', in1_ind_id, in2_ind_id, out_ind_id, site_tag_id, cutoff=CUTOFF):
        """ elemental multiply:  d_ijk in real space, convolution op in Fourier space
            Hermite basis:  define such that TN * xmultiply_mps = xmultiply_mpo
            add when apply to m=0 mode (ones place-holder), returns 1  (ie. no change)
        """
        raise NotImplementedError


    def build_xmultiply_mps(self, ax: 'Axis', x_power=1, offset=0.0, scale=1.0, split_opts=None):
        """ (a*v+b) * H_m(u)
            where a=scale, v0=offset
            NOTE: also include hermite basis offset+scaling here,
                u = (v-v0)/vt  --> v = vt*u + v0
            --> (a*(vt*u + v0) + b) * H_m(u)
            --> (a*vt)*u*H_m(u) + (a*v0 + b)*H_m(u)

            x * H1_m(x) = np.sqrt(m/2) * H1_(m-1)(x) + np.sqrt((m+1)/2) * H1_(m+1)(x)
        """
        u_offset = scale*self.x_offset + offset
        u_scale  = scale*self.x_scale
        print('u offset', self.x_offset, offset, u_offset, u_scale, x_power)

        vec = np.zeros(ax.npts)
        if x_power == 1:
            vec[0] = u_offset
            vec[1] = 1./2 * u_scale  ## weighting from physicists Hermite polynomials

        mps = ax.map_state_to_mps(vec, split_opts=split_opts)
        return mps


    def build_xmultiply_mpo(self, ax: 'Axis', x_power=1, offset=0.0, scale=1.0, split_opts=None):
        """ (a*v+b) * H_m(u)
            where a=scale, v0=offset
            NOTE: also include hermite basis offset+scaling here,
                u = (v-v0)/vt  --> v = vt*u + v0
            --> (a*(vt*u + v0) + b) * H_m(u)
            --> (a*vt)*u*H_m(u) + (a*v0 + b)*H_m(u)

            x * H_m(x)  = m * H_(m-1)(x) + 1/2 * H_(m+1)(x)
        """
        u_offset = scale*self.x_offset + offset
        u_scale  = scale*self.x_scale

        ms = ax.xpts[1:]
        mat = np.diag(ms,k=1) + np.diag(1./2*np.ones(ax.npts-1),k=-1)   # m->m-1, m->m+1
        mat = mat*u_scale + np.eye(ax.npts)*u_offset
        if x_power > 1:
            mat = np.linalg.matrix_power(mat, x_power)
        mpo = ax.map_operator_to_mpo(mat, split_opts=split_opts)
        return mpo


    def build_firstderivative_mpo(self, ax, compress_opts=None, **kwargs):
        """ dH_{m}/dv = 2*m*H_{m-1}
        """
        ms = ax.xpts[1:]
        mat = np.diag(2 * ms, k=1)  * self.x_scale # m->m-1
        mpo = ax.map_operator_to_mpo(mat, split_opts=compress_opts)
        return mpo


    def build_secondderivative_mpo(self, ax, compress_opts=None, **kwargs):
        ms = ax.xpts[1:]
        mat = np.diag(2 * ms, k=1) * self.x_scale # m->m-1
        mat = np.dot(mat, mat)
        mpo = ax.map_operator_to_mpo(mat, split_opts=compress_opts)
        return mpo


    def build_integral_mps(self, ax, is_sqrt=False, site_ind_id='i({})', site_tag_id='T({})'):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        weights = scipy.special.factorial(ax.xpts) * np.sqrt(2*np.pi) * self.x_scale
        return ax.map_state_to_mps(weights, site_ind_id=site_ind_id, site_tag_id=site_tag_id)


    def build_indefinite_integral_mps(self, ax, order=1):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        raise NotImplementedError



class HermiteGaussianBasis(HermiteBasis):
    """ Hermite-Gaussian polynomials
          HG_m(x) = 1 / (2^m m! sqrt(pi)) exp(-x^2/2) H_m(x)
          orthogonal (int dx H_m(x) H_n(x) = delta_m,n)
          x * HG_m(x)  = np.sqrt(m/2) * H_(m-1)(x) + np.sqrt((m+1)/2) * H_(m+1)(x)
          d/dx HG_m(x) = np.sqrt(m/2) * H_(m-1)(x) - np.sqrt((m+1)/2) * H_(m+1)(x)
    """
    # def __init__(self, x_offset=0.0, x_scale=1.0):
    #     """ HG(y) = Hermite Gaussian functions
    #         y = (x-x_offset)/x_scale
    #     """
    #     self.x_offset = x_offset
    #     self.x_scale  = x_scale


    @lru_cache(maxsize=256)
    def hermite_polynomial(self, m: int, x0: int = -1, xL: int = 1, npts: int = 128):
        """ does not take into account offset and scaling
            m: Hermite moment
            H_{n+1}(x) = sqrt(2/n+1) x H_{n}(x) - sqrt(n/n+1) H_{n-1}(x)
        """
        x = (np.linspace(x0, xL, npts) - self.x_offset) / self.x_scale
        if m == 0:
            return 1. / np.pi**(1./4) * np.exp(-(x ** 2)/2)
        elif m == 1:
            return np.sqrt(2) / np.pi**(1./4) * np.exp(-(x ** 2)/2) * x
        else:
            return np.sqrt(2 / m) * x * self.hermite_polynomial(m - 1, x0, xL, npts) - \
                   np.sqrt((m - 1) / m) * self.hermite_polynomial(m - 2, x0, xL, npts)


    def from_realspace_1D(self, func: 'Callable', num_modes: int):

        coeffs = []
        for n in range(num_modes):

            def func_hermite(x):
                # lambda x: func(x) * HG(n, x)
                norm = 1. / np.sqrt(2 ** n * scipy.special.factorial(n) * np.sqrt(np.pi))
                y = (x - self.x_offset) / self.x_scale
                return func(y) * norm * np.exp(-y**2/2) * scipy.special.eval_hermite(n, y)
            ## physicist's Hermite polynomials Hn(x) = (-1)^n e^(x^2) d^n/dx^b e^(-x^2)
            ## Hermite-Gauss:  (2^n n! \sqrt(\pi))^{-1/2) e^(-x^2/2) Hn(x)

            ig, err = integrate.quad(func_hermite, -np.inf, np.inf)

            coeffs += [ ig ]

        return coeffs


    def get_realspace_1D(self, data, ax_ind, x0=0.0, xL=1.0, npts=128):
        """ data:  coefficients of Hermite polynomials (order m = 0 to M-1)
            x0,xL: specifying domain of interest
            note: numpy fct is physicists Hermite polynomials (no normalization)
        """
        # domain = ((x0-self.x_offset)/self.x_scale, (xL-self.x_offset)/self.x_scale)
        data = np.moveaxis(data, ax_ind, -1)

        # plt.figure()
        # x0, xL = -1, 1
        # npts = 3000
        # xs = np.linspace(x0, xL, npts)
        # plt.plot(xs, self.hermite_polynomial(1, x0, xL, npts))
        # plt.plot(xs, self.hermite_polynomial(64, x0, xL, npts))
        # plt.plot(xs, self.hermite_polynomial(128, x0, xL, npts))
        # plt.show()
        # exit()

        out_data = np.empty(data.shape[:-1] + (npts,), dtype=data.dtype)
        for idx in np.ndindex(data.shape[:-1]):
            hermite_vals = data[idx][0] * self.hermite_polynomial(0, x0, xL, npts)
            for m in range(1, data.shape[-1]):
                if np.abs(data[idx][m]) > 1.0e-15:
                    hermite_vals += data[idx][m] * self.hermite_polynomial(m, x0, xL, npts)
            out_data[idx] = hermite_vals

        out_data = np.moveaxis(out_data, -1, ax_ind)
        return out_data


    # def get_realspace_1D(self, data, ax_ind, x0=0.0, xL=1.0, npts=128):
    #     """ data:  coefficients of Hermite polynomials (order m = 0 to M-1)
    #         x0,xL: specifying domain of interest
    #         note: numpy fct is physicists Hermite polynomials (no normalization)
    #     """
    #     domain = ((x0-self.x_offset)/self.x_scale, (xL-self.x_offset)/self.x_scale)
    #
    #     data = np.moveaxis(data, ax_ind, -1)
    #     data_shape = data.shape
    #     data = data.reshape(-1, data_shape[-1])
    #
    #     ms = np.arange(data_shape[-1])
    #     norms = 1./np.sqrt(2**ms * scipy.special.factorial(ms) * np.pi)
    #     data = data * norms
    #
    #     xs = np.linspace(x0, xL, npts)
    #     out_data = np.empty((data.shape[0],npts), dtype=complex)
    #     for i in range(data_shape[-1]):
    #         hermite_poly = np.polynomial.hermite.Hermite(data[i,:], domain, (x0, xL))
    #         out_data[i,:] = hermite_poly.linspace(npts)[1] * np.exp(-(xs-self.x_offset)**2/2/self.x_scale**2)
    #
    #     out_data = out_data.reshape(data_shape[:-1]+(npts,))
    #     out_data = np.moveaxis(out_data, -1, ax_ind)
    #     return out_data


    def get_ones_mps(self, ax: 'Axis', site_ind_id='i({})', site_tag_id='X({})', anc_dim=None, anc_name_l=None,
                     anc_name_r=None) -> 'MPSType':
        raise NotImplementedError


    def build_elemental_multiply_tn(self, ax: 'Axis', in1_ind_id, in2_ind_id, out_ind_id, site_tag_id, cutoff=CUTOFF):
        """ elemental multiply:  d_ijk in real space, convolution op in Fourier space
            Hermite basis:  define such that TN * xmultiply_mps = xmultiply_mpo
        """
        raise NotImplementedError


    def build_xmultiply_mpo(self, ax: 'Axis', x_power=1, offset=0.0, scale=1.0, split_opts=None):
        """ (a*v+b) * H_m(u)
            where a=scale, v0=offset
            NOTE: also include hermite basis offset+scaling here,
                u = (v-v0)/vt  --> v = vt*u + v0
            --> (a*(vt*u + v0) + b) * H_m(u)
            --> (a*vt)*u*H_m(u) + (a*v0 + b)*H_m(u)

            x * H_m(x)  = np.sqrt(m/2) * H_(m-1)(x) + np.sqrt((m+1)/2) * H_(m+1)(x)
        """
        u_offset = scale*self.x_offset + offset
        u_scale  = scale*self.x_scale

        ms = ax.xpts[1:]
        mat = np.diag(np.sqrt(ms/2),k=1) + np.diag(np.sqrt(ms/2),k=-1)   # m->m-1, m->m+1
        mat = mat*u_scale + np.eye(ax.npts)*u_offset
        mat = np.linalg.matrix_power(mat, x_power)
        mpo = ax.map_operator_to_mpo(mat, split_opts=split_opts)
        return mpo


    def build_firstderivative_mpo(self, ax, compress_opts=None, **kwargs):
        """ d/dx H_m(x) = np.sqrt(m/2) * H_(m-1)(x) - np.sqrt((m+1)/2) * H_(m+1)(x)
        """
        ms = ax.xpts[1:]
        mat = np.diag(np.sqrt(ms/2),k=1) - np.diag(np.sqrt(ms/2),k=-1)   # m->m-1, m->m+1
        mat = mat * 1./self.x_scale
        mpo = ax.map_operator_to_mpo(mat, split_opts=compress_opts)
        return mpo


    def build_secondderivative_mpo(self, ax, compress_opts=None, **kwargs):
        ms = ax.xpts[1:]
        mat = np.diag(np.sqrt(ms/2),k=1) + np.diag(np.sqrt(ms/2),k=-1)   # m->m-1, m->m+1
        mat = mat * 1./self.x_scale
        mat = np.dot(mat, mat)
        mpo = ax.map_operator_to_mpo(mat, split_opts=compress_opts)
        return mpo


    def build_integral_mps(self, ax, is_sqrt=False, site_ind_id='i({})', site_tag_id='T({})'):
        """ get MPS that integrates along Axis ax. multiply by dx here
            if is_sqrt:    orthogonal (int dx H_m(x) H_n(x) = delta_m,n)
            else:          integ H_n(u) exp(-u^2) du = 0 for n > 0;  # for n=0
        """
        if is_sqrt:
            integ_mps = ax.get_iden_mps(site_ind_id, site_tag_id) * self.x_scale
        else:
            ## would need to rewrite in Hermite polynomial basis
            raise NotImplementedError
        return integ_mps


    def build_indefinite_integral_mps(self, ax, order=1):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        raise NotImplementedError




class AWHermiteGaussianBasis(HermiteBasis):
    """ Asymmetrically Weighted Hermite-Gaussian polynomials (1)
          [1] H1_m(x) = (2^m m! pi)^(-1/2) exp(-x^2) H_m(x)
          [2] H2_m(x) = (2^m m!)^(-1/2) H_m(x)
          orthogonal (int dx H1_m(x) H2_n(x) = delta_m,n)
          x * H1_m(x)  = np.sqrt(m/2) * H1_(m-1)(x) + np.sqrt((m+1)/2) * H1_(m+1)(x)
          d/dx H1_m(x) = - np.sqrt( 2(m+1) ) * H_(m+1)(x)
    """
    # def __init__(self, x_offset=0.0, x_scale=1.0):
    #     """ H1(y), H2(y) = asymmetrically weighted Hermite-Gaussian polynomials
    #         y = (x-x_offset)/x_scale
    #     """
    #     self.x_offset = x_offset
    #     self.x_scale  = x_scale


    @lru_cache(maxsize=256)
    def hermite_polynomial(self, m: int, x0: float = -1, xL: float = 1, npts: int = 128):
        """ does not take into account offset and scaling
            m: Hermite moment
            H_{n+1}(x) = sqrt(2/n+1) x H_{n}(x) - sqrt(n/n+1) H_{n-1}(x)
        """
        x = (np.linspace(x0, xL, npts) - self.x_offset) / self.x_scale
        if m == 0:
            return 1./np.sqrt(np.pi) * np.exp(-x**2)
        elif m == 1:
            return np.sqrt(2/np.pi) * np.exp(-x**2) * x
        else:
            return np.sqrt(2/m) * x * self.hermite_polynomial(m-1, x0, xL, npts) - \
                   np.sqrt((m-1)/m) * self.hermite_polynomial(m-2, x0, xL, npts)


    def from_realspace_1D(self, func: 'Callable', num_modes: int):

        coeffs = []
        for n in range(num_modes):
            def func_hermite(x):
                # lambda x: func(x) * HG(n, x)
                norm = 1. / np.sqrt(2 ** n * scipy.special.factorial(n) * np.pi)
                y = (x - self.x_offset) / self.x_scale
                return func(y) * norm * np.exp(-y ** 2) * scipy.special.eval_hermite(n, y)

            ## physicist's Hermite polynomials Hn(x) = (-1)^n e^(x^2) d^n/dx^b e^(-x^2)
            ## Hermite-Gauss:  (2^n n! \sqrt(\pi))^{-1/2) e^(-x^2/2) Hn(x)

            ig, err = integrate.quad(func_hermite, -np.inf, np.inf)

            coeffs += [ig]

        return coeffs

    def get_realspace_1D(self, data, ax_ind, x0=0.0, xL=1.0, npts=128):
        """ data:  coefficients of Hermite polynomials (order m = 0 to M-1)
            x0,xL: specifying domain of interest
            note: numpy fct is physicists Hermite polynomials (no normalization)
        """
        data = np.moveaxis(data, ax_ind, -1)

        # plt.figure()
        # x0, xL = -0.25, 0.25
        # npts = 1000
        # xs = np.linspace(x0,xL, npts)
        # plt.plot(xs, self.hermite_polynomial(1,x0,xL,npts))
        # plt.plot(xs, self.hermite_polynomial(128,x0,xL,npts))
        # plt.plot(xs, self.hermite_polynomial(256,x0,xL,npts))
        # plt.show()
        # exit()


        out_data = np.empty( data.shape[:-1] + (npts,), dtype=data.dtype)
        for idx in np.ndindex(data.shape[:-1]):
            hermite_vals = data[idx][0] * self.hermite_polynomial(0, x0, xL, npts)
            for m in range(1,data.shape[-1]):
                if np.abs(data[idx][m]) > 1.0e-15:
                    hermite_vals += data[idx][m] * self.hermite_polynomial(m, x0, xL, npts)
            out_data[idx] = hermite_vals

        out_data = np.moveaxis(out_data, -1, ax_ind)
        return out_data


    # def get_realspace_1D(self, data, ax_ind, x0=0.0, xL=1.0, npts=128):
    #     """ data:  coefficients of Hermite polynomials (order m = 0 to M-1)
    #         x0,xL: specifying domain of interest
    #         note: numpy fct is physicists Hermite polynomials (no normalization)
    #     """
    #     domain = ((x0-self.x_offset)/self.x_scale, (xL-self.x_offset)/self.x_scale)
    #
    #     data = np.moveaxis(data, ax_ind, -1)
    #     # data_shape = data.shape
    #     # print('hermite aw', data_shape)
    #     # data = data.reshape(-1, data_shape[-1])
    #
    #     ms = np.arange(data.shape[-1])
    #     norms = 1./np.sqrt( np.pi) / 2**(ms/2) / np.sqrt(scipy.special.factorial(ms))
    #     data = data * norms
    #
    #     # plt.figure()
    #     # x0, xL = -5, 5
    #     # npts = 1000
    #     # xs = np.linspace(x0,xL, npts)
    #     # plt.plot(np.polynomial.hermite.Hermite([1,0,0,0]).linspace(npts, domain=[x0,xL])[1]*np.exp(-xs**2)*norms[0])
    #     # plt.plot(np.polynomial.hermite.Hermite([0,1,0,0]).linspace(npts, domain=[x0,xL])[1]*np.exp(-xs**2)*norms[1])
    #     # plt.plot(np.polynomial.hermite.Hermite([0,0,1,0]).linspace(npts, domain=[x0,xL])[1]*np.exp(-xs**2)*norms[2])
    #     # norm_1 = 1./np.sqrt( np.pi) / 2**(128/2) / np.sqrt(scipy.special.factorial(128))
    #     # plt.plot(np.polynomial.hermite.Hermite([0]*128 + [1]).linspace(npts, domain=[x0,xL])[1]*np.exp(-xs**2/2)*norm_1)
    #     # plt.show()
    #     # exit()
    #
    #     xs = np.linspace(x0, xL, npts)
    #     out_data = np.empty( data.shape[:-1] + (npts,), dtype=data.dtype)
    #     for idx in np.ndindex(data.shape[:-1]):
    #         hermite_poly = np.polynomial.hermite.Hermite(data[idx], domain, (x0, xL))
    #         out_data[idx] = hermite_poly.linspace(npts)[1] * np.exp(-(xs - self.x_offset) ** 2 / self.x_scale ** 2)
    #
    #     # out_data = out_data.reshape(data_shape[:-1] + (npts,))
    #     out_data = np.moveaxis(out_data, -1, ax_ind)
    #     return out_data


    def get_ones_mps(self, ax: 'Axis', site_ind_id='i({})', site_tag_id='X({})', anc_dim=None, anc_name_l=None,
                     anc_name_r=None) -> 'MPSType':
        raise NotImplementedError


    def build_elemental_multiply_tn(self, ax: 'Axis', in1_ind_id, in2_ind_id, out_ind_id, site_tag_id, cutoff=CUTOFF):
        """ elemental multiply:  d_ijk in real space, convolution op in Fourier space
            Hermite basis:  define such that TN * xmultiply_mps = xmultiply_mpo
        """
        raise NotImplementedError


    def build_xmultiply_mpo(self, ax: 'Axis', x_power=1, offset=0.0, scale=1.0, split_opts=None):
        """ (a*v+b) * H_m(u)
            where a=scale, v0=offset
            NOTE: also include hermite basis offset+scaling here,
                u = (v-v0)/vt  --> v = vt*u + v0
            --> (a*(vt*u + v0) + b) * H_m(u)
            --> (a*vt)*u*H_m(u) + (a*v0 + b)*H_m(u)

            x * H1_m(x) = np.sqrt(m/2) * H1_(m-1)(x) + np.sqrt((m+1)/2) * H1_(m+1)(x)
        """
        u_offset = scale*self.x_offset + offset
        u_scale  = scale*self.x_scale
        print('u offset', self.x_offset, offset, u_offset, u_scale, x_power)

        ms = ax.xpts[1:]
        mat = np.diag(np.sqrt(ms/2),k=1) + np.diag(np.sqrt(ms / 2), k=-1)   # m->m-1, m->m+1
        # mat = 2*np.eye(len(ms)+1)
        mat = mat * u_scale + np.eye(ax.npts) * u_offset
        mat = np.linalg.matrix_power(mat, x_power)
        mpo = ax.map_operator_to_mpo(mat, split_opts=split_opts)
        return mpo


    def build_firstderivative_mpo(self, ax, compress_opts=None, **kwargs):
        """ d/dx H1_m(x) = - np.sqrt( 2(m+1) ) * H_(m+1)(x)
        """
        ms = ax.xpts[1:]
        mat = -1 * np.diag(np.sqrt(2 * ms), k=-1)   # m->m+1
        mat = mat * 1./self.x_scale
        mpo = ax.map_operator_to_mpo(mat, split_opts=compress_opts)
        return mpo


    def build_secondderivative_mpo(self, ax, compress_opts=None, **kwargs):
        ms = ax.xpts[1:]
        mat = -1 * np.diag(np.sqrt(2 * ms), k=-1)   # m->m+1
        mat = mat * 1./self.x_scale
        mat = np.dot(mat, mat)
        mpo = ax.map_operator_to_mpo(mat, split_opts=compress_opts)
        return mpo


    def get_integral_weight(self, ax):
        """ normalization for integral_mps
        """
        return self.x_scale


    def build_integral_mps(self, ax, is_sqrt=False, site_ind_id='i({})', site_tag_id='T({})'):
        """ get MPS that integrates along Axis ax. multiply by dx here
            (pi 2^n n!)^(-1/2) integ H_n(u) exp(-u^2) du = 0 for n > 0;  1 for n=0
        """
        if is_sqrt:
            ## would need to rewrite in Hermite polynomial basis
            raise NotImplementedError
        else:
            integ_mps = ax.get_select_elems_mps([ax.zero_ind]) * self.x_scale
        return integ_mps


    def build_indefinite_integral_mps(self, ax, order=1):
        """ get MPS that integrates along Axis ax. multiply by dx here
        """
        raise NotImplementedError