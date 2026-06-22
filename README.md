# qtt_dlrx_simple

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20801130.svg)](https://doi.org/10.5281/zenodo.20801130)

Research code for **quantized tensor trains (QTTs)** with **interpolative dynamical low-rank
approximation (DLRA)** time integrators, applied to hyperbolic PDEs (Vlasov, Boltzmann,
Burgers, and Maxwell equations).

This is the reference implementation accompanying:

> Erika Ye and Chao Yang,
> *Time integration of quantized tensor trains using the interpolative dynamical
> low-rank approximation*, arXiv:2512.15703 (2025).
> https://arxiv.org/abs/2512.15703

QTTs provide a low-rank, multiscale representation of high-resolution multidimensional data.
DLRA constrains time integration to a low-rank manifold; this code explores *interpolative*
DLRA schemes (low-rank manifolds built from interpolation points and interpolating
polynomials), which—unlike the usual orthogonal-projector DLRA—remain well suited to
nonlinear systems and upwind time integration.

## Installation

Requires **Python 3.9** and **quimb 1.4.0** (later quimb releases change the tensor-network
API used here).

Using conda (recommended for reproducibility):

```bash
conda env create -f environment.yml
conda activate qtt_dlrx_simple
pip install -e .
```

Using pip only:

```bash
pip install -r requirements.txt
pip install -e .
```

Installing the package in editable mode (`pip install -e .`) puts the modules on your
`sys.path`, so the test scripts no longer rely on `sys.path.append('../')`.

### Verify your installation

`tests/test_install_xfunc.py` is a fast, headless check that the core stack (the `GridTN1D`
data model, the DMRG / cross / mixed local solvers, and the quimb wiring) is installed and
numerically sound. It approximates several target functions as QTTs and asserts each solver
converges to the direct-SVD reference at full bond dimension.

Install the test extra (adds `pytest`) and run it either way:

```bash
pip install -e ".[test]"          # or: pip install pytest

# as a pytest suite:
cd tests && python -m pytest test_install_xfunc.py -v

# or as a plain script (prints PASS/FAIL per case, exits non-zero on failure):
cd tests && python test_install_xfunc.py
```

All four cases should report `PASS`.

## Quickstart

All of the main test cases run from a **single unified branch**—each has its own entry
script under `tests/`:

| Test case                                          | Entry script                              |
|----------------------------------------------------|-------------------------------------------|
| 2-D Maxwell simulation of a wavepacket in a cavity | `tests/test_EM2.py`                       |
| Advection test problem in Fourier space            | `tests/test_vlasovEM_test_nox-k.py`       |
| Upwind Burgers test case                           | `tests/test_burgers_1D.py`                |

Jupyter notebooks of these tests replicating results of the associated paper are also provided.

(Historically each problem lived on its own branch—`EM2D`, `advec`, `burgers`—which still
exist for reference, but the current branch runs all three.)

General tensor-network operations (addition, element-wise multiplication, function
evaluation, etc.) are demonstrated in `tests/test_xfunc_v3_mixed.py`.

Run a test script directly, e.g.:

```bash
python tests/test_burgers_1D.py
```

Simulation output and checkpoints are written under `tests/tmp_f/`, `tests/tmp_s/`, and
`tests/data_cross/` (git-ignored).

## How the code is organized

A simulation is built up through the following object model (see any `tests/` file for a
worked example):

1. **Coordinates** — `Coordinate(name, CoordinateType)` from `coord/coord_sys.py`.
   Optionally a `CoordinateSystem` (e.g. `CartesianCoordinateSpace`).
2. **Axis** — `Axis(L, q, coordinate, **kwargs)`. A frozen (immutable) object specifying the
   grid discretization and basis type.
3. **Grid** — usually `Grid1D`; `Grid(id, axes, layout_type)` where `axes` is an ordered
   tuple of `Axis` objects and `layout_type` is one of `LayoutType.SEQUENTIAL`, `PARALLEL`,
   `PARALLEL_GROUP` (defined in `setup_/enums.py`).
4. **Configurations** (from `setup_/configs.py`):
   - `MaterialConfiguration` — plasma parameters.
   - `CompressionConfiguration` — GridTN compression levels
     (`max_bond`, `cutoff`, `cutoff_mode`); compression "levels" let different stages of an
     algorithm use different parameters. Specifying `level=1` and leaving the rest as
     defaults is usually sufficient.
   - `DerivativeConfiguration` — boundary conditions, finite-difference type
     (`FORWARD`, `BACKWARD`, `CENTER`) and order.
5. **GridTN1D** — matrix product states: `GridTN1D(grid, data, ax_deriv_configs=...)`, or
   obtained from a `Grid` via `grid.map_state_to_mps(...)` / `grid.make_mps_ndim(...)`.
6. **Fields** (`field.py`):
   - `ScalarField(name, grid, data=None, **kwargs)`
   - `Field(name, grid, data=None, **kwargs)` — vector field whose components are indexed by
     `Coordinate`s (commonly associated with each `Axis`). A field may carry components not in
     its grid (e.g. `Ez` on an `(x, y)` grid). All `GridTN`s in a field must share a grid.
7. **PDE system** — a `PDE_system` subclass that drives the time stepping (see each class).

## Repository structure

Top-level modules:

| File                     | Purpose                                                                   |
|--------------------------|---------------------------------------------------------------------------|
| `axis.py`                | `Axis` object: grid points or spectral modes; builds operators            |
| `grid.py`                | `Grid`: collection of `Axis` objects defining an n-D grid                 |
| `grid1D.py`              | 1-D QTT grid                                                              |
| `grids_composite.py`     | Parent class for `grid_comb.py`                                           |
| `grid_comb.py`           | QTT grid for comb (tree-like) layout                                      |
| `gridTN.py`              | Tensor-network object paired with a `Grid` (parent class)                |
| `gridTN_1D.py`           | 1-D QTT / MPS / MPO with a `Grid1D`                                       |
| `gridTN_composite.py`    | Parent class for `gridTN_1Dcomb.py`                                       |
| `gridTN_1Dcomb.py`       | QTN with comb layout on a `GridsComb`                                     |
| `field.py`               | `Field` (vector) and `ScalarField` (scalar) objects                      |
| `pde_system.py`          | Base PDE object; drives time stepping                                     |
| `pde_boltzmann.py`       | Boltzmann equation                                                        |
| `pde_burgers.py`         | Burgers equation                                                          |
| `pde_EM.py`              | Maxwell equations                                                         |
| `pde_vlasov.py`          | Vlasov equation                                                           |
| `pde_vlasovEM.py`        | Vlasov–Maxwell                                                            |
| `pde_vlasovES.py`        | Vlasov–Poisson (electrostatic)                                            |
| `helper_quimb.py`        | Helper functions for quimb tensor-network objects                        |
| `helper_TE.py`           | Time-evolution subroutines                                                |

Sub-packages:

- `setup_/` — `enums.py` (enums/types), `configs.py` (configuration classes), `defaults.py`
  (global defaults and input-flag parsing), `helper.py` (test-setup helpers), `paths.py`
  (output paths), `quimb_TN1D.py` (quimb generalization for >2 physical indices per core).
- `axis_map/` — quantization mappings (default `map_binary`): `binary` (coarse→fine),
  `flipbinary` (fine→coarse), `mirror`, `flipmirror`.
- `basis/` — basis functions (default spatial): Hermite, Fourier (`k`), spatial (real-space),
  and finite-difference coefficients. (`findiff_coeffs.py` is vendored from
  https://github.com/maroba/findiff.)
- `coord/` — coordinate systems (only Cartesian is fully implemented).
- `layout/` — multidimensional QTT layouts (default sequential): `ParallelF` (interleaved,
  dimensions kept factorized), `ParallelG` (interleaved, dimensions contracted).
- `local_solvers/` — the current modular solver stack (Block / Term / Evaluator /
  TimeIntegrator) for local tensor-network updates.
- `tests/` — runnable example scripts (run directly, e.g. `python tests/test_burgers_1D.py`).
  The exception is `tests/test_install_xfunc.py`, an installation-verification suite that runs
  under `pytest` (or directly as a script).

### Solver stacks — current vs. legacy

Two solver stacks coexist. Prefer the modular **`local_solvers/`** package; the monolithic
`helper_*` solvers below are older and, in several cases, not fully tested:

- `helper_dlr.py` — standard DLR (outdated).
- `helper_dmrg.py`, `helper_dmrg_2.py` — DMRG for (non-blocked) linear equations
  (`_2` has minor algorithmic modifications; the two are largely equivalent).
- `helper_tdvp.py`, `helper_tdvp_v2.py` — TDVP / projector-splitting DLR (non-blocked);
  `_v2` is the updated version.
- `helper_block_dmrg.py`, `helper_block_tddmrg_3.py`, `helper_block_tdvp_3.py` — blocked
  variants (the `_3` files bridge to `local_solvers/`).
- `helper_sl.py` — semi-Lagrangian integration (not relevant to the current tests).

## Documentation conventions

Docstrings follow the **NumPy style** (Sphinx/`napoleon`-compatible). Please keep new code
consistent with this convention.

## Citation

If you use this software, please cite both the software and the paper. Citation metadata is
provided in [`CITATION.cff`](CITATION.cff) and [`codemeta.json`](codemeta.json). The paper is
arXiv:2512.15703 (https://arxiv.org/abs/2512.15703). Each release is archived on Zenodo; the
DOI badge above is the *concept* DOI ([10.5281/zenodo.20801130](https://doi.org/10.5281/zenodo.20801130)),
which always resolves to the latest release — cite it to refer to the software in general, or
cite a specific version's DOI for a particular release.

## Releases

Version history is recorded in [`CHANGELOG.md`](CHANGELOG.md).

## Support 

To report issues or for support, email erikaye@lbl.gov

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).

## AI Usage

Claude Opus was used to merge the three original branches (advec, Burgers, EM2D) into merge-unify,
clean up the code (e.g., reduce verbosity), and updating the data saving from pickle to npz. 
It was also used to generate metadata files.  
