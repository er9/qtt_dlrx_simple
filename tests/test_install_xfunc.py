"""Installation-verification test: QTT function approximation via DMRG / cross / mixed solvers.

Distilled from ``tests/test_xfunc_v3_mixed.py`` (the first four exploratory cases) into a
headless, asserting test suitable for confirming a successful installation. Each test
approximates a target 1-D function as a quantized tensor train (QTT) with the three local
solvers -- DMRG (:func:`~local_solvers.local_dmrg_eval.local_dmrg_evaluator`), cross
interpolation (:func:`~local_solvers.local_cross_eval.local_cross_evaluator`), and mixed
projection (:func:`~local_solvers.local_mixed_eval.local_mixed_evaluator`) -- and asserts
that, at full bond dimension, each solver converges to the same answer as a direct SVD
compression (to ~machine precision). This exercises the GridTN1D data model, the Term
layers, all three evaluators, and the quimb tensor-network wiring end to end.

Run with::

    cd tests && MPLBACKEND=Agg python -m pytest test_install_xfunc.py -v
    cd tests && MPLBACKEND=Agg python test_install_xfunc.py
"""
import sys
sys.path.append('../')

# Force a non-interactive matplotlib backend before anything pulls in pyplot, so the
# test never blocks on a plot window (this module does no plotting itself).
import matplotlib
matplotlib.use('Agg')

from setup_.configs import *
import quimb.tensor as qtn
import helper_quimb as helper
from axis import Axis
from grid1D import Grid1D
from gridTN_1D import GridTN1D
from local_solvers.terms_3 import Term_DMRG, Term_Cross
from local_solvers.local_dmrg_eval import local_dmrg_evaluator
from local_solvers.local_cross_eval import local_cross_evaluator
from local_solvers.terms_mixed import Term_Mixed
from local_solvers.local_mixed_eval import local_mixed_evaluator
import local_solvers.helper_tn as helper_tn

import pytest


# Problem size and the bond dimensions swept by each case. ``None`` = full (untruncated)
# bond dimension, where every solver should match the direct SVD to machine precision.
L = 10
Ds = [16, 32, None]
DIRECTION = 1
NSITES = 2

# Absolute accuracy gate at full bond dimension (observed errors are ~1e-12; 1e-6 leaves a
# wide margin while still catching a broken solver), and the factor by which a solver may
# exceed the SVD baseline before we consider it non-convergent.
FULL_BOND_TOL = 1.0e-10
SVD_FACTOR = 1.0e3


def _relative_l2(approx, target):
    """Relative L2 error ||approx - target|| / ||target||."""
    return np.linalg.norm(approx - target) / np.linalg.norm(target)


def _build_problem():
    """Reproduce the shared setup from the source driver.

    Returns
    -------
    dict
        ``x`` (grid points), ``data`` (the noisy, normalised, x_scale'd target samples),
        ``grid_X`` (the 1-D grid), and ``init_mps`` (the QTT of ``data``).
    """
    x = np.linspace(0, 2 * np.pi, 2 ** L, endpoint=False)
    fx = lambda x: 1 + np.sin(5 * x) * 5.0 * np.exp(-x ** 2 / 6) + np.cos(10 * x)

    x_scale = 0.1

    data = fx(x)
    np.random.seed(0)
    data = data + np.random.rand(len(data))
    data = data / np.linalg.norm(data)

    ax_x = Axis(L)
    grid_X = Grid1D('GX', [ax_x])

    init_mps = GridTN1D.from_dense_state(data, grid_X)
    init_mps.data.exponent += np.log10(x_scale)
    data = data * x_scale

    return {'x': x, 'data': data, 'grid_X': grid_X, 'init_mps': init_mps}


@pytest.fixture
def problem():
    """pytest fixture wrapping :func:`_build_problem` (fresh objects per test)."""
    return _build_problem()


def _svd_error(target, grid_X, D=None):
    """Relative L2 error of a direct SVD compression of ``target`` at bond dim ``D``."""
    comp_gtn = GridTN1D.from_dense_state(target, grid_X, split_opts={'max_bond': D})
    return _relative_l2(comp_gtn.get_data(), target)


def _check_converged(name, err, svd_err):
    """Assert a solver's full-bond error is finite, small, and SVD-comparable.

    Parameters
    ----------
    name : str
        Solver label, for failure messages.
    err : float
        The solver's relative L2 error at full bond dimension.
    svd_err : float
        The direct-SVD baseline error at full bond dimension.
    """
    assert np.isfinite(err), f'{name}: error is not finite ({err})'
    assert err < FULL_BOND_TOL, f'{name}: full-bond error {err:.2e} >= tol {FULL_BOND_TOL:.0e}'
    assert err <= max(svd_err * SVD_FACTOR, FULL_BOND_TOL), \
        f'{name}: full-bond error {err:.2e} not comparable to SVD {svd_err:.2e}'


def test_scale_f(problem):
    """Case 1 ("scale f"): approximate ``func(data) = 0.5 * data`` with each solver."""
    data, grid_X, init_mps = problem['data'], problem['grid_X'], problem['init_mps']
    func = lambda x: x * 0.5
    target = func(data)

    errs = {}
    for D in Ds:
        # dmrg: scale via the term coefficient
        term1d = Term_DMRG(init_mps.data.copy(), mps_coeff=0.5)
        dmrg_gtn = grid_X.make_gridTN(
            local_dmrg_evaluator((term1d,), direction=DIRECTION, nsites=NSITES, max_bond=D))
        errs['dmrg'] = _relative_l2(dmrg_gtn.get_data(), target)

        # cross: scale via the element-wise func
        term1c = Term_Cross(init_mps.data.copy(), mps_func=func)
        cross_gtn = grid_X.make_gridTN(
            local_cross_evaluator((term1c,), direction=DIRECTION, nsites=NSITES, max_bond=D))
        errs['cross'] = _relative_l2(cross_gtn.get_data(), target)

        # mixed
        term1m = Term_Mixed(init_mps.data.copy(), mps_func=func)
        mixed_gtn = grid_X.make_gridTN(
            local_mixed_evaluator((term1m,), direction=DIRECTION, nsites=NSITES, max_bond=D))
        errs['mixed'] = _relative_l2(mixed_gtn.get_data(), target)

        for v in errs.values():
            assert np.isfinite(v), f'non-finite error at D={D}'

    svd_err = _svd_error(target, grid_X, D=None)
    _check_converged('dmrg', errs['dmrg'], svd_err)
    _check_converged('cross', errs['cross'], svd_err)
    _check_converged('mixed', errs['mixed'], svd_err)


def test_add_const(problem):
    """Case 2 ("add const."): approximate ``data + 0.5`` with each solver."""
    data, grid_X, init_mps = problem['data'], problem['grid_X'], problem['init_mps']
    target = data + 0.5

    y_mps = GridTN1D.get_ones_mps(grid_X)
    y_mps.scalar_multiply(0.5, inplace=True)

    init_mps.data.distribute_exponent()
    y_mps.data.distribute_exponent()

    errs = {}
    for D in Ds:
        term1d = Term_DMRG(init_mps.data.copy())
        term2d = Term_DMRG(y_mps.data.copy())
        dmrg_gtn = grid_X.make_gridTN(
            local_dmrg_evaluator([term1d, term2d], direction=DIRECTION, max_bond=D, nsites=NSITES))
        errs['dmrg'] = _relative_l2(dmrg_gtn.get_data(), target)

        term1m = Term_Mixed(init_mps.data.copy())
        term2m = Term_Mixed(y_mps.data.copy())
        mixed_gtn = grid_X.make_gridTN(
            local_mixed_evaluator([term1m, term2m], direction=DIRECTION, max_bond=D, nsites=NSITES))
        errs['mixed'] = _relative_l2(mixed_gtn.get_data(), target)

        term1c = Term_Cross(init_mps.data.copy())
        term2c = Term_Cross(y_mps.data.copy())
        cross_gtn = grid_X.make_gridTN(
            local_cross_evaluator([term1c, term2c], direction=DIRECTION, max_bond=D, nsites=NSITES))
        errs['cross'] = _relative_l2(cross_gtn.get_data(), target)

        for v in errs.values():
            assert np.isfinite(v), f'non-finite error at D={D}'

    svd_err = _svd_error(target, grid_X, D=None)
    _check_converged('dmrg', errs['dmrg'], svd_err)
    _check_converged('cross', errs['cross'], svd_err)
    _check_converged('mixed', errs['mixed'], svd_err)


def test_add_fct(problem):
    """Case 3 ("add fct"): approximate ``data + x**2 * 4e-4`` with each solver."""
    x, data, grid_X, init_mps = problem['x'], problem['data'], problem['grid_X'], problem['init_mps']
    fx = lambda x: x ** 2 * 4e-4
    target = data + fx(x)
    y_mps = GridTN1D.from_dense_state(fx(x), grid_X)

    errs = {}
    for D in Ds:
        term1d = Term_DMRG(init_mps.data.copy())
        term2d = Term_DMRG(y_mps.data.copy())
        dmrg_gtn = grid_X.make_gridTN(
            local_dmrg_evaluator([term1d, term2d], direction=DIRECTION, max_bond=D, nsites=NSITES))
        errs['dmrg'] = _relative_l2(dmrg_gtn.get_data(), target)

        term1c = Term_Cross(init_mps.data.copy())
        term2c = Term_Cross(y_mps.data.copy())
        cross_gtn = grid_X.make_gridTN(
            local_cross_evaluator([term1c, term2c], direction=DIRECTION, max_bond=D, nsites=NSITES))
        errs['cross'] = _relative_l2(cross_gtn.get_data(), target)

        term1m = Term_Mixed(init_mps.data.copy())
        term2m = Term_Mixed(y_mps.data.copy())
        mixed_gtn = grid_X.make_gridTN(
            local_mixed_evaluator([term1m, term2m], direction=DIRECTION, max_bond=D, nsites=NSITES))
        errs['mixed'] = _relative_l2(mixed_gtn.get_data(), target)

        for v in errs.values():
            assert np.isfinite(v), f'non-finite error at D={D}'

    svd_err = _svd_error(target, grid_X, D=None)
    _check_converged('dmrg', errs['dmrg'], svd_err)
    _check_converged('cross', errs['cross'], svd_err)
    _check_converged('mixed', errs['mixed'], svd_err)


def test_elem_mult(problem):
    """Case 4 ("elem mult"): approximate ``data * (x**2 * 4e-4)`` with each solver."""
    x, data, grid_X, init_mps = problem['x'], problem['data'], problem['grid_X'], problem['init_mps']
    fx = lambda x: x ** 2 * 4e-4
    target = data * fx(x)
    y_mps = GridTN1D.from_dense_state(fx(x), grid_X)

    errs = {}
    for D in Ds:
        # dmrg: element-wise multiply via an operator term with two tiers
        term1d = Term_DMRG(init_mps.data.copy(), operators=[y_mps.data.copy()], num_tiers=2)
        dmrg_gtn = grid_X.make_gridTN(
            local_dmrg_evaluator([term1d], direction=DIRECTION, max_bond=D, nsites=NSITES))
        errs['dmrg'] = _relative_l2(dmrg_gtn.get_data(), target)

        # cross: two terms combined by the product callable
        term1c = Term_Cross(init_mps.data.copy())
        term2c = Term_Cross(y_mps.data.copy())
        cross_gtn = grid_X.make_gridTN(
            local_cross_evaluator([term1c, term2c], direction=DIRECTION, max_bond=D, nsites=NSITES,
                                  combine_terms_func=helper_tn.prod_tens))
        errs['cross'] = _relative_l2(cross_gtn.get_data(), target)

        # mixed: two terms combined by the product callable
        term1m = Term_Mixed(init_mps.data.copy())
        term2m = Term_Mixed(y_mps.data.copy())
        mixed_gtn = grid_X.make_gridTN(
            local_mixed_evaluator([term1m, term2m], direction=DIRECTION, max_bond=D, nsites=NSITES,
                                  combine_terms_func=helper_tn.prod_tens))
        errs['mixed'] = _relative_l2(mixed_gtn.get_data(), target)

        for v in errs.values():
            assert np.isfinite(v), f'non-finite error at D={D}'

    svd_err = _svd_error(target, grid_X, D=None)
    _check_converged('dmrg', errs['dmrg'], svd_err)
    _check_converged('cross', errs['cross'], svd_err)
    _check_converged('mixed', errs['mixed'], svd_err)


if __name__ == '__main__':
    # Runnable as a plain script for a quick installation check.
    failures = 0
    for name, fn in [('scale_f', test_scale_f), ('add_const', test_add_const),
                     ('add_fct', test_add_fct), ('elem_mult', test_elem_mult)]:
        try:
            fn(_build_problem())
            print(f'PASS  test_{name}')
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures += 1
            print(f'FAIL  test_{name}: {exc}')
    if failures:
        print(f'\n{failures} test(s) failed.')
        sys.exit(1)
    print('\nAll installation checks passed.')
