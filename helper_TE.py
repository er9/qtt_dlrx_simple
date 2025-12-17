import scipy.linalg

from setup_.configs import *
if TYPE_CHECKING:
    from gridTN import GridTN

"""
time integration schemes
"""

def time_integration(state, dt: Numeric, euler_func, deriv_func, add_func, scale_func, te_order=4,
        deriv0 = None, compress_level: int = 1, verbose_plot=False, **deriv_kwargs, ):

    if te_order == 1:
        return euler_func(state, dt, deriv0, compress_level=compress_level)
    elif te_order == 2:
        return rk2(state, dt, euler_func, deriv_func, add_func, deriv0=deriv0,
                   compress_level=compress_level, verbose_plot=verbose_plot, **deriv_kwargs,)
    elif te_order == 3:
        return rk3(state, dt, euler_func, deriv_func, add_func, scale_func, deriv0=deriv0,
                   compress_level=compress_level, verbose_plot=verbose_plot, **deriv_kwargs,)
    elif te_order == 4:
        return rk4(state, dt, euler_func, deriv_func, add_func, scale_func, deriv0=deriv0,
                   compress_level=compress_level, verbose_plot=verbose_plot, **deriv_kwargs, )
    else:
        raise ValueError(f'choose valid time integration scheme, not {te_order}')


def exact(state, dt: Numeric, exact_func, compress_level: int = 1, verbose_plot=False, **deriv_kwargs, ):
    """ perform exact time evolution
    """
    state0 = state.copy()
    try:
        time = state.time
    except AttributeError:
        time = None
    time1 = time + dt if time is not None else None

    if compress_level == 0:
        comp1 = comp2 = comp3 = comp4 = comp5 = 0
    else:
        comp1, comp2, comp3, comp4, comp5 = range(compress_level, compress_level + 5)

    # if deriv0 is None:
    #     deriv0 = deriv_func(state0, time=time, compress_level=comp4, compress_level1=comp5,
    #                         verbose_plot=verbose_plot, **deriv_kwargs)

    new_state = exact_func(state0, dt, compress_level=comp1)
    if time1 is not None:
        new_state.time = time1

    return new_state


def euler(state, dt: Numeric, euler_func, deriv_func, add_func, scale_func, time=None,
        deriv0 = None, compress_level: int = 1, verbose_plot=False, return_intermediates=False, **deriv_kwargs, ):
    """ perform explicit RK2 time evolution
        Heun's method:
            input: y_n, h=dt
            y1_(n+1) = y_(n) + h F(t_n, y_n)
            y_(n+1) = y_(n) + h/2 [ F(t_n, y_n) + F(t_(n+1), y1_(n+1))
            output: y_(n+1)
    """
    state0 = state.copy()
    if time is None:
        try:
            time = state.time
        except AttributeError:
            time = None
    time1 = time + dt if time is not None else None

    # print('HELPER TE EULER')

    if compress_level == 0:
        comp1 = comp2 = comp3 = comp4 = comp5 = 0
    else:
        comp1, comp2, comp3, comp4, comp5 = range(compress_level, compress_level + 5)

    if deriv0 is None:
        deriv0 = deriv_func(state0, time=time, compress_level=comp4, compress_level1=comp5,
                            verbose_plot=verbose_plot, **deriv_kwargs)

    # print('dt', dt, return_intermediates)
    new_state = euler_func(state0, dt, deriv0=deriv0, adapt=False,
                           compress_level=comp2, compress_level1=comp4, compress_level2=comp5)

    if return_intermediates:
        intermediates = (state0.copy(), new_state.copy()) # deriv0.copy())

    if time1 is not None:
        try:
            new_state.time = time1
        except AttributeError:
            pass

    if return_intermediates:
        # return new_state, (state0, state1, state2, state3)
        return new_state, intermediates  # (state0, deriv0, deriv1, deriv2, deriv3)
    else:
        return new_state


def rk2(state, dt: Numeric, euler_func, deriv_func, add_func, scale_func=None, time=None,
        deriv0 = None, compress_level: int = 1, verbose_plot=False, return_intermediates=False, **deriv_kwargs, ):
    """ perform explicit RK2 time evolution
        Heun's method:
            input: y_n, h=dt
            y1_(n+1) = y_(n) + h F(t_n, y_n)
            y_(n+1) = y_(n) + h/2 [ F(t_n, y_n) + F(t_(n+1), y1_(n+1))
            output: y_(n+1)
    """
    state0 = state.copy()
    if time is None:
        try:
            time = state.time
        except AttributeError:
            time = None
    time1 = time + dt if time is not None else None

    # print('rk2')

    if compress_level == 0:
        comp1 = comp2 = comp3 = comp4 = comp5 = 0
    else:
        comp1, comp2, comp3, comp4, comp5 = range(compress_level, compress_level + 5)

    if deriv0 is None:
        deriv0 = deriv_func(state0, time=time, compress_level=comp4, compress_level1=comp5,
                            verbose_plot=verbose_plot, **deriv_kwargs)

    state1 = euler_func(state0, dt, deriv0=deriv0, adapt=False,
                        compress_level=comp2, compress_level1=comp4, compress_level2=comp5)
    deriv1 = deriv_func(state1, time=time1, compress_level=comp4, compress_level1=comp5,
                                              verbose_plot=verbose_plot, **deriv_kwargs)
    deriv_sum = add_func(deriv0, deriv1, compress_level=0)

    intermediates = (state0.copy(), deriv0.copy(), deriv1.copy())

    new_state = euler_func(state0, 0.5 * dt, deriv0=deriv_sum, inplace=True, compress_level=comp1)
    if time1 is not None:
        try:
            new_state.time = time1
        except AttributeError:
            pass

    if return_intermediates:
        # return new_state, (state0, state1, state2, state3)
        return new_state, intermediates  # (state0, deriv0, deriv1, deriv2, deriv3)
    else:
        return new_state


def rk3(state, dt: Numeric, euler_func, deriv_func, add_func, scale_func, deriv0 = None,
        compress_level: int = 1, verbose_plot=False, **deriv_kwargs):
    """ perform 4-stage RK3 time evolution
        https://gkeyll.readthedocs.io/en/latest/dev/ssp-rk.html#ssprk
        note: this add_func adds states together; "adapt" keyword less important to function
    """
    state0 = state

    try:
        time = state.time
    except AttributeError:
        time = None
    time1 = time + dt / 2 if time is not None else None
    time2 = time + dt if time is not None else None

    if compress_level == 0:
        comp1 = comp2 = comp3 = comp4 = comp5 = 0
    else:
        comp1, comp2, comp3, comp4, comp5 = range(compress_level, compress_level+5)

    if deriv0 is None:
        deriv0 = deriv_func(state0, time=time, compress_level=comp4)

    state1 = euler_func(state0, dt, deriv0=deriv0, compress_level=0)
    state1 = add_func(state1, state0, inplace=True, compress_level=comp2)
    state1 = scale_func(state1, 0.5, inplace=True)
    deriv1 = deriv_func(state1, time=time1, compress_level=comp4)

    state2 = euler_func(state1, dt, deriv0=deriv1, compress_level=0)
    state2 = add_func(state2, state1, inplace=True, compress_level=comp2)
    state2 = scale_func(state2, 0.5, inplace=True)
    deriv2 = deriv_func(state2, time=time2, compress_level=comp4)

    state3 = euler_func(state2, dt, deriv0=deriv2, compress_level=0)
    state3 = add_func(state3, state2, inplace=True)
    state3 = scale_func(state3, 1. / 6, inplace=True)

    tmp = scale_func(state0, 2. / 3, inplace=False)
    state3 = add_func(state3, tmp, inplace=True, compress_level=comp2)
    deriv3 = deriv_func(state3, time=time1, compress_level=comp4)

    new_state = euler_func(state3, dt, deriv0=deriv3, compress_level=0)
    new_state = add_func(new_state, state3, inplace=True, compress_level=comp1)
    new_state = scale_func(new_state, 0.5, inplace=True)

    if time is not None:
        new_state.time = time2

    return new_state


# @profile
def rk4(state, dt, euler_func, deriv_func, add_func, scale_func, deriv0 = None, time=None,
        compress_level: int = 1, verbose=0, verbose_plot=False, return_intermediates=False, **deriv_kwargs):
    """ perform explicit RK4 time evolution
        compress = 0:   don't compress at all
        compress_level: compress final state
        comp2:   + compress intermediate rk4 states
        comp3:   + compress sum of derivatives
        comp4:   + compress derivatives
        comp5:   + compress during calculation of derivative
        note:  add_func adds derivatives together.
               intermediate euler_funcs should not adapt bond--only last euler_func should
    """
    # print('rk4', compress_level)

    # try:
    #     time = state.time
    # except AttributeError:
    #     time = None
    time1 = time + dt / 2 if time is not None else None
    time2 = time + dt if time is not None else None

    if verbose > 1:
        print('HELPER RK4', time, time1, time2)

    if compress_level == 0:
        comp1 = comp2 = comp3 = comp4 = comp5 = 0
    else:
        comp1, comp2, comp3, comp4, comp5 = range(compress_level, compress_level + 5)

    state0 = state.copy() if state is not None else None

    if deriv0 is None:
        if verbose > 0:
            print('HELPER TE RK4: compute deriv 1')
        deriv0 = deriv_func(state0, time=time, compress_level=comp4, compress_level1=comp5,
                            compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)

    if verbose > 0:
        print('HELPER TE RK4: get euler 1')
    state1 = euler_func(state0, dt * 0.5, deriv0=deriv0, adapt=False, compress_level=comp2, compress_level1=comp4,
                        compress_level2=comp5)
    if verbose > 0:
        print('HELPER TE RK4: compute deriv 2')
    deriv1 = deriv_func(state1, time=time1, compress_level=comp4, compress_level1=comp5,
                        compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)
    # print('state 1', np.linalg.norm(state1-state0))
    # print('deriv 1', np.linalg.norm(deriv1))

    if verbose > 0:
        print('HELPER TE RK4: get euler 2')
    state2 = euler_func(state0, dt * 0.5, deriv0=deriv1, adapt=False, compress_level=comp2, compress_level1=comp4,
                        compress_level2=comp5)
    if verbose > 0:
        print('HELPER TE RK4: compute deriv 3')
    deriv2 = deriv_func(state2, time=time1, compress_level=comp4, compress_level1=comp5,
                        compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)
    # print('state 2', np.linalg.norm(state2 - state0))
    # print('deriv 2', np.linalg.norm(deriv2))
    # print('state 2', state2)
    # print('deriv 2', deriv2)

    if verbose > 0:
        print('HELPER TE RK4: get euler 3')
    state3 = euler_func(state0, dt, deriv0=deriv2, adapt=False, compress_level=comp2, compress_level1=comp4, compress_level2=comp5)
    deriv3 = deriv_func(state3, time=time2, compress_level=comp4, compress_level1=comp5,
                        compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)
    # print('state 3', np.linalg.norm(state3 - state0))
    # print('deriv 3', np.linalg.norm(deriv3))
    # print('state 3', state3)
    # print('deriv3', deriv3)

    intermediates = (state0.copy(), deriv0.copy(), deriv1.copy(), deriv2.copy(), deriv3.copy())

    deriv1 = scale_func(deriv1, 2, inplace=True)
    deriv2 = scale_func(deriv2, 2, inplace=True)

    sum_deriv = add_func(deriv0, deriv1, compress_level=0, is_list=False)
    sum_deriv = add_func(sum_deriv, deriv2, compress_level=0, inplace=True, is_list=False)
    sum_deriv = add_func(sum_deriv, deriv3, compress_level=comp4, inplace=True, is_list=False)
    # print('sum_deriv', sum_deriv)
    # print('sum deriv', np.linalg.norm(sum_deriv))

    if verbose > 0:
        print('HELPER TE RK4: final euler')
    new_state = euler_func(state0, dt / 6., deriv0=sum_deriv, inplace=True, compress_level=comp1,
                           compress_level1=comp2, compress_level2=comp3)
    # if time is not None:
    #     new_state.time = time1

    if return_intermediates:
        # return new_state, (state0, state1, state2, state3)
        return new_state, intermediates # (state0, deriv0, deriv1, deriv2, deriv3)
    else:
        return new_state


def ssprk3(state, dt, euler_func, deriv_func, add_func, scale_func, deriv0 = None, time=None,
        compress_level: int = 1, verbose_plot=False, return_intermediates=False, **deriv_kwargs):
    """ three-stage third order strong stability preserving (Durran 2.48)
        https://gkeyll.readthedocs.io/en/latest/dev/ssp-rk.html

        ^ these implementations didn't actually work... but using the butcher table did
        0   | 0
        1   | 1
        1/2 | 1/4   1/4
        -------------------
            | 1/6   1/6   2/3


        Strong-stability preserving for |c dt/dx | <= 1
        compress = 0:   don't compress at all
        compress_level: compress final state
        comp2:   + compress intermediate rk4 states
        comp3:   + compress sum of derivatives
        comp4:   + compress derivatives
        comp5:   + compress during calculation of derivative
        note:  add_func adds derivatives together.
               intermediate euler_funcs should not adapt bond--only last euler_func should
    """
    print('HELPER TE SSPRK3')
    # print('rk4', compress_level)

    # if time is None:
    #     try:
    #         time = state.time
    #     except AttributeError:
    #         time = None
    time1 = time + dt / 2 if time is not None else None
    time2 = time + dt if time is not None else None

    if compress_level == 0:
        comp1 = comp2 = comp3 = comp4 = comp5 = 0
    else:
        comp1, comp2, comp3, comp4, comp5 = range(compress_level, compress_level + 5)

    state0 = state.copy()

    if deriv0 is None:
        deriv0 = deriv_func(state0, time=time, compress_level=comp4, compress_level1=comp5,
                            compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)

    # print('get euler 1')
    # state1 = euler_func(state0, dt, deriv0=deriv0, adapt=False, compress_level=comp2, compress_level1=comp4,
    #                     compress_level2=comp5)
    # deriv1 = deriv_func(state1, time=time2, compress_level=comp4, compress_level1=comp5,
    #                     compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)

    state2 = euler_func(state0, dt, deriv0=deriv0, adapt=False, compress_level=comp2, compress_level1=comp4,
                        compress_level2=comp5)
    deriv2 = deriv_func(state2, time=time2, compress_level=comp4, compress_level1=comp5,
                        compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)

    state3 = add_func( state0, scale_func(add_func(deriv0, deriv2), 0.25, inplace=False), inplace=False)
    deriv3 = deriv_func(state3, time=time1, compress_level=comp4, compress_level1=comp5,
                        compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)

    tot_deriv = add_func(scale_func(deriv0, 1./6), scale_func(deriv2, 1./6))
    tot_deriv = add_func(tot_deriv, scale_func(deriv3, 2./3))
    new_state = euler_func(state0, dt, deriv0=tot_deriv)

    # intermediates = (state0.copy(), deriv0.copy(), deriv2.copy(), deriv3.copy())
    intermediates = (state0.copy(), state2.copy(), state3.copy())

    # if time is not None:
    #     new_state.time = time1

    if return_intermediates:
        # return new_state, (state0, state1, state2, state3)
        return new_state, intermediates # (state0, deriv0, deriv1, deriv2)
    else:
        return new_state


def ssprk4(state, dt, euler_func, deriv_func, add_func, scale_func, deriv0 = None, time=None,
        compress_level: int = 1, verbose_plot=False, return_intermediates=False, **deriv_kwargs):
    """ four-stage third order strong stability preserving (Durran 2.48)
        https://gkeyll.readthedocs.io/en/latest/dev/ssp-rk.html

        ^ these implementations didn't actually work... (unless set dt=2*dt?)
         but using the butcher table did
        0   | 0
        1/2 | 1/2
        1   | 1/2   1/2
        1/2 | 1/6   1/6   1/6
        --------------------------
            | 1/6   1/6   1/6   1/2

        Strong-stability preserving for |c dt/dx | <= 2
        compress = 0:   don't compress at all
        compress_level: compress final state
        comp2:   + compress intermediate rk4 states
        comp3:   + compress sum of derivatives
        comp4:   + compress derivatives
        comp5:   + compress during calculation of derivative
        note:  add_func adds derivatives together.
               intermediate euler_funcs should not adapt bond--only last euler_func should
    """
    ### old implementation
    # dt = 2 * dt
    # print('ssprk4', compress_level)
    #
    # try:
    #     time = state.time
    # except AttributeError:
    #     time = None
    # time1 = time + dt / 2 if time is not None else None
    # time2 = time + dt if time is not None else None
    #
    # if compress_level == 0:
    #     comp1 = comp2 = comp3 = comp4 = comp5 = 0
    # else:
    #     comp1, comp2, comp3, comp4, comp5 = range(compress_level, compress_level + 5)
    #
    # state0 = state.copy()
    #
    # if deriv0 is None:
    #     deriv0 = deriv_func(state0, time=time, compress_level=comp4, compress_level1=comp5,
    #                         compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)
    #
    # # print('get euler 1')
    # state1 = euler_func(state0, 0.5*dt, deriv0=deriv0, adapt=False, compress_level=comp2, compress_level1=comp4,
    #                     compress_level2=comp5)
    # deriv1 = deriv_func(state1, time=time1, compress_level=comp4, compress_level1=comp5,
    #                     compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)
    # # print('state 1', state1)
    # # print('deriv 1', deriv1)
    #
    # state2 = euler_func(state1, 0.25 * dt, deriv0=deriv1, adapt=False, compress_level=comp2, compress_level1=comp4,
    #                     compress_level2=comp5)
    # deriv2 = deriv_func(state2, time=time2, compress_level=comp4, compress_level1=comp5,
    #                     compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)
    #
    # state_ = add_func( scale_func(state0, 2./3, inplace=False), scale_func(state2, 1./3, inplace=False) )
    # state3 = euler_func(state_, dt * 1./6, deriv0=deriv2, adapt=False, compress_level=comp2, compress_level1=comp4,
    #                     compress_level2=comp5)
    # deriv3 = deriv_func(state3, time=time1, compress_level=comp4, compress_level1=comp5,
    #                     compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)
    # # print('state 2', state2)
    # # print('deriv 2', deriv2)
    #
    # new_state = euler_func(state3, 0.5*dt, deriv0=deriv3, adapt=False, compress_level=comp1, compress_level1=comp2,
    #                        compress_level2=comp3)
    #
    # if time is not None:
    #     new_state.time = time1

    print('HELPER TE SSPRK4, time', time)
    # print('rk4', compress_level)

    # if time is None:
    #     try:
    #         time = state.time
    #     except AttributeError:
    #         time = None
    time1 = time + dt / 2 if time is not None else None
    time2 = time + dt if time is not None else None
    print('time', time, time1, time2)

    if compress_level == 0:
        comp1 = comp2 = comp3 = comp4 = comp5 = 0
    else:
        comp1, comp2, comp3, comp4, comp5 = range(compress_level, compress_level + 5)

    state0 = state.copy()

    if deriv0 is None:
        deriv0 = deriv_func(state0, time=time, compress_level=comp4, compress_level1=comp5,
                            compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)

    # print('get euler 1')
    state1 = euler_func(state0, 0.5 * dt, deriv0=deriv0, adapt=False, compress_level=comp2, compress_level1=comp4,
                        compress_level2=comp5)
    deriv1 = deriv_func(state1, time=time1, compress_level=comp4, compress_level1=comp5,
                        compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)


    state2 = euler_func(state1, 0.5 * dt, deriv0=deriv1, adapt=False, compress_level=comp2, compress_level1=comp4,
                        compress_level2=comp5)
    deriv2 = deriv_func(state2, time=time2, compress_level=comp4, compress_level1=comp5,
                        compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)

    sum_deriv_3 = add_func( add_func(deriv0, deriv1), deriv2 )
    state3 = euler_func(state0, 1./6 * dt, deriv0=sum_deriv_3, adapt=False, compress_level=comp2, compress_level1=comp4,
                        compress_level2=comp5)
    deriv3 = deriv_func(state3, time=time1, compress_level=comp4, compress_level1=comp5,
                        compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)

    tot_deriv = add_func(scale_func(sum_deriv_3, 1./6), scale_func(deriv3, 1./ 2))
    new_state = euler_func(state0, dt, deriv0=tot_deriv)

    # intermediates = (state0.copy(), deriv0.copy(), deriv1.copy(), deriv2.copy(), deriv3.copy())
    intermediates = (state0.copy(), state1.copy(), state2.copy(), state3.copy(), new_state.copy())

    # if time is not None:
    #     new_state.time = time1

    if return_intermediates:
        # return new_state, (state0, state1, state2, state3)
        return new_state, intermediates  # (state0, deriv0, deriv1, deriv2)
    else:
        return new_state


def backwards_euler_fixed_point(state, dt, euler_func, deriv_func, distance_func,
                                time: float = None, err_tol: float = None, max_iter:int = 100, compress_level: int = 1,
                                verbose_plot=False, **deriv_kwargs):
    """ implicit time differentiation:  dy/dt = F(y)
        y_n+1 = y_n + dt*F(y_n+1)       [n specifies time step]
        via fixed point iteration, Jacobi iteration with diagonal = I:
            (I + dt*F) y_n+1 = y_n
            y^(j+1) = dt*F y^(j) + y_n      [j specifies iteration]
            s.t. y^(j=infty) -> y_n+1

        note: this wouldn't work as written if there was a second order derivative term...
            diagonal wouldn't equal I
    """
    if compress_level == 0:
        comp1 = comp2 = comp3 = comp4 = comp5 = 0
    else:
        comp1, comp2, comp3, comp4, comp5 = range(compress_level, compress_level + 5)

    if err_tol is None:
        err_tol = CUTOFF
        if CUTOFF_MODE == 'rsum2':
            err_tol = np.sqrt(err_tol)
        # err_tol *= 10

    time1 = time + dt if time is not None else None

    prev_state = state

    err = np.inf
    it = 0
    while np.abs(err) > err_tol and it < max_iter:
        prev_err = err
        deriv0 = deriv_func(prev_state, time=time1, compress_level=comp4, compress_level1=comp5,
                            compress_level2=comp5, verbose_plot=verbose_plot,
                            **deriv_kwargs)
        new_state = euler_func(state, dt, deriv0=deriv0, inplace=False, compress_level=comp2, compress_level1=comp4,
                                compress_level2=comp5)
        err = distance_func(new_state, prev_state)
        print('tot err', it, err)

        if err > prev_err or np.abs(err-prev_err)/prev_err < err_tol:
            break

        if comp2 != comp1:
            new_state.compress(inplace=True, compress_level=comp1)
        # print('after comp errs', new_state.distances(prev_state, total=False, normalize=False))

        prev_state = new_state
        it += 1

    return prev_state


def implicit_midpoint_fixed_point(state, dt, euler_func, deriv_func, add_func, scale_func, distance_func,
                                  time: float = None, err_tol: float = None, max_iter:int = 100,
                                  compress_level: int = 1, verbose_plot=False, **deriv_kwargs):
    """ implicit time differentation:  dy/dt = F(y)
        y_(n+1) = y_(n) + F( (y_(n+1) + y_(n))/2 )
    """
    if compress_level == 0:
        comp1 = comp2 = comp3 = comp4 = comp5 = 0
    else:
        comp1, comp2, comp3, comp4, comp5 = range(compress_level, compress_level + 5)

    if err_tol is None:
        err_tol = CUTOFF
        if CUTOFF_MODE == 'rsum2':
            err_tol = np.sqrt(err_tol)
        err_tol *= 10

    time1 = time + dt/2 if time is not None else None

    prev_state = state.copy()

    err = np.inf
    it = 0
    while err > err_tol and it < max_iter:
        prev_err = err
        midpt_state = add_func(state, prev_state, inplace=False, compress_level=comp3)
        midpt_state = scale_func(midpt_state, 0.5, inplace=True)
        deriv0 = deriv_func(midpt_state, time=time1, compress_level=comp4, compress_level1=comp5,
                            compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)
        new_state = euler_func(state, dt, deriv0=deriv0, inplace=False, compress_level=comp2, compress_level1=comp4,
                               compress_level2=comp5)

        err = distance_func(new_state, prev_state)
        print('tot err', it, err)

        if err > prev_err or np.abs(err-prev_err)/prev_err < err_tol:
            break

        if comp2 != comp1:
            new_state.compress(inplace=True, compress_level=comp1)

        prev_state = new_state
        it += 1

    return prev_state


def crank_nicolson_fixed_point(state, euler_func, deriv_func, add_func, scale_func, distance_func,
                               dt, time: float = None, weight: float = 0.5,
                               err_tol: float = None, max_iter:int = 100, compress_level: int = 1,
                               verbose_plot=False, **deriv_kwargs):
    """ implicit time differentation:  dy/dt = F(y)
        y_(n+1) = y_(n) + h/2 ( F(y_(n+1)) + F(y_(n)) )
    """
    if compress_level == 0:
        comp1 = comp2 = comp3 = comp4 = comp5 = 0
    else:
        comp1, comp2, comp3, comp4, comp5 = range(compress_level, compress_level + 5)

    if err_tol is None:
        err_tol = CUTOFF
        if CUTOFF_MODE == 'rsum2':
            err_tol = np.sqrt(err_tol)
        err_tol *= 10

    time1 = time + dt if time is not None else None

    prev_state = state
    deriv0 = deriv_func(state, time=time, compress_level=comp4, compress_level1=comp5,
                        compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)
    # prev_state = self.euler(dt, deriv0=deriv0, compress_level=comp2, compress_level1=comp4, compress_level2=comp5)

    err = np.inf
    it = 0
    while err > err_tol and it < max_iter:
        prev_err = err
        deriv1 = deriv_func(prev_state, time=time1, compress_level=comp4, compress_level1=comp5,
                            compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)
        sum_deriv = add_func(deriv0, deriv1, inplace=False, compress_level=comp3)
        new_state = euler_func(state, dt / 2, deriv0=sum_deriv, inplace=False, adapt=False, compress_level=comp2,
                               compress_level1=comp4, compress_level2=comp5)
        err = distance_func(new_state, prev_state)
        print('tot err', it, err)

        if err > prev_err or np.abs(err-prev_err)/err < err_tol:
            break

        if comp2 != comp1:
            new_state.compress(inplace=True, compress_level=comp1)

        prev_state = new_state
        it += 1

    return prev_state

#### CGD/solver-based implicit methods

def solve_backwards_euler(state, euler_func, deriv_func, add_func, scale_func,
                          deriv_ops, identity_func, solve_func, dt, init_guess=None,
                          time: float = None, err_tol: float = None, max_iter:int = 100, compress_level: int = 1,
                          verbose_plot=False, **deriv_kwargs):
    """ implicit time differentiation:  dy/dt = F(y)
        y_n+1 = y_n + dt*F(y_n+1)       [n specifies time step]
        y_n+1 = EXPLICIT[y_n] + dt*F(y_n+1)
    """

    if err_tol is None:
        err_tol = CUTOFF
        if CUTOFF_MODE == 'rsum2':
            err_tol = np.sqrt(err_tol)
        # err_tol *= 10

    identity_op = identity_func(state)
    weights = [-dt] * len(deriv_ops) + [1.]
    deriv_ops = [scale_func(d_op, -dt) for d_op in deriv_ops] + [identity_op]

    explicit_contribution = state.copy()

    ## dmrg:  CALL THIS THROUGH GTN
    new_state = solve_func(deriv_ops, explicit_contribution, init_guess, max_iter=max_iter, conv_tol=err_tol)

    return new_state


def solve_crank_nicolson(state, euler_func, deriv_func, add_func, scale_func,
                         deriv_ops, deriv_const, identity_func, solve_func, dt, init_guess=None, weight=0.5,
                         time: float = None, err_tol: float = None, max_iter:int = None, compress_level: int = 1,
                         constraint_mats=None, constraint_vals=None,
                         return_intermediates=False,
                         solve_kwargs = None, verbose_plot=False, **deriv_kwargs):
    """ implicit time differentiation:  dy/dt = F(y)
        y_(n+1) = y_(n) + h/2 ( F(y_(n+1)) + F(y_(n)) )
        y_(n+1) = y_(n) + h ( weight * F(y_(n+1)) + (1-weight) * F(y_(n)) )

        dy/dt = Ay + b, assuming b is constant in time
        y_(n+1) = y_(n) + h ( weight * (A(y_(n+1)) + b) + (1-weight) * (A(y_(n)) + b) )

        1/c * dy/dt = Ay + b, assuming b is constant in time
        dy/dt = 1/c (A y) + 1/c b
    """
    if compress_level == 0:
        comp1 = comp2 = comp3 = comp4 = comp5 = 0
    else:
        comp1, comp2, comp3, comp4, comp5 = range(compress_level, compress_level + 5)

    if err_tol is None:
        err_tol = CUTOFF
        if CUTOFF_MODE == 'rsum2':
            err_tol = np.sqrt(err_tol)
        # err_tol *= 10

    ## target = state + F[state] * (1 - weight)
    state0 = state.copy()
    deriv0 = deriv_func(state0, time=time, compress_level=comp4, compress_level1=comp5,
                        compress_level2=comp5, verbose_plot=verbose_plot, **deriv_kwargs)
    explicit_contribution = euler_func(state0, dt * (1 - weight), deriv0=deriv0, inplace=False, adapt=False,
                                       compress_level=comp2, compress_level1=comp4, compress_level2=comp5)
    if deriv_const is not None:
        explicit_contribution = add_func(explicit_contribution, scale_func(deriv_const, dt * weight, is_list=False),
                                         is_list=False)

    identity_op = identity_func(state)
    # deriv_ops = [scale_func(d_op, -dt * weight) for d_op in deriv_ops] + [identity_op]
    deriv_ops = scale_func(deriv_ops, -dt * weight, inplace=False)
    deriv_ops = add_func(deriv_ops, identity_op)
    # deriv_ops = add_func({}, identity_op)

    ## dmrg
    solve_kwargs = {} if solve_kwargs is None else solve_kwargs
    new_state = solve_func(deriv_ops, explicit_contribution, init_guess,
                           max_iter=max_iter, conv_tol=err_tol,
                           constraint_mats=constraint_mats, constraint_vals=constraint_vals,
                           **solve_kwargs)

    if return_intermediates:
        return new_state, (state0, explicit_contribution, new_state)

    return new_state




#### GridTN methods

def backwards_euler(state: 'GridTN', dt, deriv_mpos: list['GridTN'], explicit_contribution: 'GridTN' = None,
                    err_tol: float = None, max_iter:int = 100, compress_opts = None,):
    """ implicit time differentiation:  dy/dt = F(y)
        y_n+1 = y_n + dt*F(y_n+1)       [n specifies time step]
        y_n+1 = EXPLICIT[y_n] + dt*F(y_n+1)
    """

    if err_tol is None:
        err_tol = CUTOFF
        if CUTOFF_MODE == 'rsum2':
            err_tol = np.sqrt(err_tol)
        # err_tol *= 10

    identity_mpo = state.grid.get_iden_mpo()
    deriv_mpos = deriv_mpos + [identity_mpo]
    weights = [-dt] * len(deriv_mpos) + [1.]

    ## dmrg:  CALL THIS THROUGH GTN
    compress_opts = {} if None else compress_opts
    max_bond = compress_opts.get('max_bond', None)
    if explicit_contribution is not None:
        state = explicit_contribution
    new_state = state.solve(deriv_mpos, weights=weights,
                            max_iter=max_iter, conv_tol=err_tol, max_bond=max_bond)

    return new_state


def implicit_midpoint(state: 'GridTN', dt, deriv_mpos: list['GridTN'], explicit_contribution: 'GridTN' = None,
                      err_tol: float = None, max_iter:int = 100, compress_opts = None,
                      compress_opts_1 = None, assume_linear_F=True):
    """ implicit time differentation:  dy/dt = F(y)
        y_(n+1) = y_(n) + F( (y_(n+1) + y_(n))/2 )
        (note: F is linear if assume fixed field; same as Crank-Nicolson)
    """
    if assume_linear_F:
        return crank_nicolson(state, dt, deriv_mpos, explicit_contribution=explicit_contribution,
                              err_tol=err_tol, max_iter=max_iter, compress_opts=compress_opts,
                              compress_opts_1=compress_opts_1)
    else:
        raise NotImplementedError


def crank_nicolson(state: 'GridTN', dt, deriv_mpos: list['GridTN'], explicit_contribution: 'GridTN' = None,
                   err_tol: float = None, max_iter:int = 100, compress_opts = None,
                   compress_opts_1 = None, compress_opts_2 = None):
    """ implicit time differentation:  dy/dt = F(y)
        y_(n+1) = y_(n) + h/2 ( F(y_(n+1)) + F(y_(n)) )
        y_(n+1) = EXPLICIT[y_(n)] + h/2 ( F(y_(n+1)) + F(y_(n)) )
    """
    if err_tol is None:
        err_tol = CUTOFF
        if CUTOFF_MODE == 'rsum2':
            err_tol = np.sqrt(err_tol)
        # err_tol *= 10

    ## target = state + F[state] * 1/2
    deriv0 = state.sum_apply(deriv_mpos, compress_opts=compress_opts_2)
    deriv0 = deriv0.scalar_multiply(0.5 * dt)

    target = state.add(deriv0, inplace=False, compress=False)
    if explicit_contribution is not None:
        target = target.add(explicit_contribution, inplace=True, compress=False)
    target.compress(inplace=True, compress_opts=compress_opts_1)

    identity_mpo = state.grid.get_iden_mpo()
    deriv_mpos = deriv_mpos + [identity_mpo]
    weights = [-dt / 2] * len(deriv_mpos) + [1.]

    ## dmrg
    compress_opts = {} if None else compress_opts
    max_bond = compress_opts.get('max_bond', None)
    new_state = target.solve(deriv_mpos, weights=weights,
                             max_iter=max_iter, conv_tol=err_tol, max_bond=max_bond)

    return new_state


def imex(state: 'GridTN', dt, exp_deriv_mpos: list['GridTN'], imp_deriv_mpos: list['GridTN'],
         exp_te_func, imp_te_func,
         err_tol: float = None, max_iter:int = 100, compress_opts = None,
         compress_opts_1 = None):
    """ dy/dt = F[y] = F,explicit[y] + F,implicit[y]
        eg. y_(n+1) = TE_EX[y(n), F_ex] + TE_IM[y(n), y(n+1), F_im] - y(n)
            y_(n+1) = target(n) + operators * y(n+1)
            ## need to avoid double counting y(n) in algorithm

        alternatively: split step method:
        y* = TE_EX[y(n), F_ex]
        y_(n+1) = TE_IM[y*, y(n+1), F_im]
    """
    ## need to provide GridTN class functions (need to define these first)
    target = exp_te_func(state, dt, exp_deriv_mpos, compress_opts=compress_opts_1)
    new_state = imp_te_func(state, dt, imp_deriv_mpos, explicit_contribution=target,
                            err_tol=err_tol, max_iter=max_iter,
                            compress_opts=compress_opts, compress_opts_1=compress_opts_1)
    return new_state
