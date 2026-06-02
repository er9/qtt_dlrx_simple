"""Test- and setup-helper functions, including constructors for Maxwellian
distributions (real-space and Fourier, in one and multiple dimensions) used to
initialize fields for QTT/DLRA test problems."""
import pickle
import numpy as np
import axis_map


def maxwellian_ndim(*vs, vth2=1.0, density=1.0, flow=0.0):
    """ vs: velocity grid for each dimension
	    summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
	flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    v2 = 0.0
    K = len(vs)
    for i in range(K):
        # try:                            v2 += np.conj(vs[i]-flow[i]) * (vs[i] - flow[i])
        # except(TypeError,IndexError):   v2 += np.conj(vs[i]-flow) * (vs[i] - flow)
        # try:                            v2 += (vs[i]-np.real(flow[i]))**2
        # except(TypeError,IndexError):   v2 += (vs[i]-np.real(flow))**2
        try:                            v2 += (vs[i]-flow[i])**2
        except(TypeError,IndexError):   v2 += (vs[i]-flow)**2

    print('v2', np.linalg.norm(np.real(v2)), np.linalg.norm(np.imag(v2)))

    return density * np.sqrt(1./2/np.pi/vth2)**K * np.exp(-v2/2/vth2)


def maxwellian(vs, vth2=1.0, density=1.0, flow=0.0, is_sqrt=False):
    """ vs: velocity grid for each dimension
            summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
        flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    v2 = np.conj(vs-flow) * (vs-flow)
    K = 1
    out = density * np.sqrt(1./2/np.pi/vth2)**K * np.exp(-v2/2/vth2)
    if is_sqrt:
        out = np.sqrt(out)
    return out


def maxwellian_k(vs, vth2=1.0, density=1.0, flow=0.0, is_sqrt=False):
    """ vs: velocity grid for each dimension
            summed together s.t. v^2 = vx^2 + vy^2 + vz^2 for each point
        flow: scalar or list (with same # elements as number of velocity dimensions)
    """
    # vs = np.fft.fftshift(vs)
    # v2 = np.conj(vs-flow)*(vs-flow)
    v2 = np.conj(vs) * vs
    dv = vs[1] - vs[0]      ## only works for 1D
    if is_sqrt:
        vth2 = vth2 * 2

    K = 1
    norm = density * np.sqrt(1./2/np.pi/vth2)**K * np.sqrt(vth2)**K
    if is_sqrt:
        norm = np.sqrt(norm)

    out = norm * np.exp(-v2 * vth2 / 2 - 1.j * flow * vs) \
                * np.exp(-1.j * vs * np.pi / dv)
    # out = out / (2 * np.pi) ** 1.25

    xs = np.linspace(-np.pi/dv , np.pi/dv, len(vs), endpoint=False)
    check = np.fft.ifft(np.fft.ifftshift(out), norm='forward') # * len(vs)
    if is_sqrt:
        ref = np.sqrt( density * np.sqrt(1. / 2 / np.pi / vth2 * 2) ** K) * np.exp(-(xs - flow) ** 2 / 2 / vth2)
    else:
        ref = density * np.sqrt(1. / 2 / np.pi / vth2) ** K * np.exp(-(xs - flow) ** 2 / 2 / vth2)
    out = out * (np.max(ref) / np.max(check))
    print('maxwellian k check', np.max(check), np.max(ref))
    # exit()

    # import matplotlib.pyplot as plt
    # plt.figure()
    # # plt.plot(out)
    # # plt.plot(np.imag(out))
    # plt.plot(check / np.max(check))
    # plt.plot(ref / np.max(ref))
    # # plt.plot(np.fft.ifft(out))
    # # out = np.fft.fftshift(out)
    # # plt.plot(out)
    # plt.show()
    # exit()

    # if is_sqrt:
    #     out = np.sqrt(out)
    return out



def cfl_limit(*pde_params):
    """ compute CFL time-step limit given sets of grid params
        eq_params:  list of nested tuples:
                    [((dx, dv), (vmax, Fmax)), ... ]
        result: 1./ max( (vmax/dx + Fmax/dv, ...) )
    """
    terms = []
    for param in pde_params:
        dxs, Fs = param
        terms += [np.sum([F / dx for F, dx in zip(Fs, dxs)])]
    # print('cfl terms', terms)

    # return 1. / max(terms)
    return np.abs(1./ sum(terms))


def dt_limit_k(*pde_params):
    """ compute CFL time-step limit given sets of grid params
        Jardin: dt <= 1/ v k  (p. 283) (pdf p.306)
        eq_params:  list of nested tuples:
                    [((kx_max, kv_max), (vx, Fx)), ... ]
        result: 1./ max( (vmax/dx + Fmax/dv, ...) )

        --> approximate max eigenvalue as v_max * (k_max + E_max)
    """
    terms = []
    for param in pde_params:
        kmaxs, Fs = param
        terms += [1./np.sum([F * kmax for F, kmax in zip(Fs, kmaxs)])]

    return np.min(terms)


def rk4_limit_FourierHermite(m_max, k_max, F_max):
    """ limit dt (estimate) given by rk4 stability constraints:
        1 + z + z^2/2 + z^3/6 + z^4/4 < 1
        for z real, --> z = -2.7853
        for z imag, --> z ~ 1 to ~ 2.75

        calculated assuming in fourier/hermite basis
        df/dt + v * grad_x(f) + q(E)*grad_v(f) = 0
        --> approximate max eigenvalue as sqrt(m_max) * (k_max + E_max)
    """
    dt_max = 1. / ( np.sqrt(m_max*2) * (k_max + F_max) )
    # dt_max = 2.75 / (np.sqrt(2 * m_max) * (k_max + F_max))
    # dt_max = 0.723 / (k_max * np.sqrt(m_max/2))     # 0.723 for Adams-Bashforth method
    ### https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/fourierhermite-spectral-representation-for-the-vlasovpoisson-system-in-the-weakly-collisional-limit/ED04894B962D443A41A2EABC7CEC555D
    return np.abs(dt_max)


def rk4_limit_Fourier(k_max, v_max):
    """ limit dt (estimate) given by rk4 stability constraints:
        1 + z + z^2/2 + z^3/6 + z^4/4 < 1
        for z real, --> z = -2.7853
        for z imag, --> z ~ 1 to ~ 2.75

        let z = lambda + i * w
        pure advection in 1D:  lambda = 0, w = c * k
        rk4 stable at ~ 2 w \Delta t

        calculated assuming in fourier(x) + real space(v) basis
        df/dt + v * grad_x(f) + q(E)*grad_v(f) = 0
        --> approximate max eigenvalue as v_max * (k_max + E_max)

        Jardin, p.283
    """
    dt_max = 2.0 / (k_max * v_max)
    ## EM leap frog dt max?
    return np.abs(dt_max)


def rk4_limit_Hermite(m_max, dx, F_max):
    """ limit dt (estimate) given by rk4 stability constraints:
        1 + z + z^2/2 + z^3/6 + z^4/4 < 1
        for z real, --> z = -2.7853
        for z imag, --> z ~ 1 to ~ 2.75

        calculated assuming in fourier/hermite basis
        df/dt + v * grad_x(f) + q(E)*grad_v(f) = 0
        --> approximate max eigenvalue as sqrt(m_max) * (k_max + E_max)
    """
    dt_max = 1.0 / ( np.sqrt(m_max*2) * (1./dx + F_max) )
    return np.abs(dt_max)



def LB_collision_freq_cfl_limit(dt, dv, vth, vmax, cfl):
    """ solve for collision rate that matches dt, dx, dv specified by the rest of the problem
        von neumann stability analysis (centered FD)
    """
    term1 = vmax / dv * dt
    term2 = 2 * vth ** 2 / dv ** 2 * dt
    freq = cfl / (term1 + term2)
    # print('check cfl', freq, dt, cfl * cfl_limit(((dv, dv ** 2 / 2), (vmax * freq, vth ** 2 * freq))), )

    return freq


def get_zero_bounds(data, tol=1.0e-6):
    """ given np.ndarray data, find bounding box  for which abs(data) > tol
    """
    ndim = data.ndim

    inds = []
    for i in range(ndim):
        data_1D = np.sum(data, axis=tuple([j for j in range(ndim) if j != i]))
        bool_data = np.abs(data_1D) > tol
        min_arg = np.argmax( bool_data )
        max_arg = len(data_1D) - 1 - np.argmax( bool_data[::-1] )

        if np.abs(data_1D[min_arg]) < tol:  # all values < tol. this shouldn't ever happen
            min_arg = np.nan
            max_arg = np.nan

        inds += [(min_arg, max_arg)]
    # print('bounds', inds, data.shape)
    return inds



def optional_load(filename):
    """ try loading the data else return None
    """
    try:
        return pickle.load(open(filename, 'rb'))
    except IOError:
        return None


def optional_dump(obj, filename):
    """ try loading the MPS representing some state
    """
    if obj is not None:
        pickle.dump(obj, open(filename, 'wb'))
    else:
        pass


def get_maps(mkey: str):
    ax_maps = []
    for mk in mkey:
        if mk == 'F':   # forward
            ax_map = axis_map.BinaryMap()
        elif mk == 'B':     # back
            ax_map = axis_map.FlipBinaryMap()
        elif mk == 'M':
            ax_map = axis_map.MirrorMap()
        elif mk == 'N':
            ax_map = axis_map.FlipMirrorMap()
        else:
            raise KeyError(f'{mk} not a valid map key')
        ax_maps += [ax_map]
    return ax_maps
