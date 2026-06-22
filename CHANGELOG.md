# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The version here is kept in sync with `pyproject.toml`, `CITATION.cff`, and `codemeta.json`.

## [Unreleased]

<!-- Add new changes here under Added / Changed / Deprecated / Removed / Fixed / Security. -->

## [0.1.0] - 2026-06-22

First archived release accompanying the paper *"Time integration of quantized tensor trains
using the interpolative dynamical low-rank approximation"* (Ye & Yang, arXiv:2512.15703).
Archived on Zenodo — concept DOI [10.5281/zenodo.20801130](https://doi.org/10.5281/zenodo.20801130);
this release's version DOI is 10.5281/zenodo.20801131.

### Added
- QTT / DLRA solvers for the Vlasov, Boltzmann, Burgers, and Maxwell equations, with DMRG,
  cross-interpolation, and mixed local time integrators.
- Numpy `.npz` tensor-train serialization (`tt_io.py`), replacing the previous pickle-based
  save/load of `GridTN` / comb data.
- Installation-verification test suite (`tests/test_install_xfunc.py`).
- Optional `E_drive` hook in `pde_vlasovEM` to supply a prescribed time-dependent field
  (replaces the hardcoded advection-test field; `None` disables it).
- `verbose` print-gating flags across the local-solver stack and PDE drivers.
- Selectable matplotlib backend for headless runs.
- NumPy-style docstrings across `gridTN`, `gridTN_1D`, and `helper_quimb`.
- Jupyter notebooks reproducing the figures from the paper.
- Packaging and FAIR metadata: `pyproject.toml`, `CITATION.cff`, `codemeta.json`, `LICENSE`,
  this changelog, and the Zenodo DOI.

### Changed
- Unified the previously per-problem branches (EM2D / advec / burgers) so all problems run
  from a single branch.
- Advection test uses `x_version = 'proj'`; assorted README, requirements, and environment
  updates.

### Removed
- Unused `global_rk_cross` overrides that carried a broken import.

[Unreleased]: https://github.com/er9/qtt_dlrx_simple/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/er9/qtt_dlrx_simple/releases/tag/v0.1.0
