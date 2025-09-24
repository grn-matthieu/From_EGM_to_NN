## [Unreleased]

No notable changes since `v0.5.0` was prepared. Add upcoming changes here before cutting a new release.

## [0.5.0] - 2025-09-24
### Breaking
- `load_config` now returns nested NamedTuples; downstream code must switch from `cfg[:key]` indexing to dot access (`cfg.key`).

### Notable
- Stability: core solver and public APIs considered stable for the `v0.5` line.
- Refactor: reorganized and cleaned up core modules and test layout to simplify maintenance and future extensions.
- NN solver: added a neural-network-based solver to experiment with learned policy/value approximations.
- Tests: greatly improved test coverage across `analysis`, `methods`, and `solvers` with clearer smoke and unit harnesses.

## [0.4.0] - 2025-09-18
- Cleanup: organized tests into folders and updated runtests includes.
- Repo hygiene: ignored local temp artifacts.
- Version bump to 0.4.0.

All notable changes to this project are documented in this file.

## [0.3.0] - 2025-09-15

- Add Perturbation method (first order) and kernel.
- Introduce ForwardDiff for steady-state derivatives.
- Improve test coverage for Projection and Perturbation methods.
- Refactor EGM solver and add stronger validation utilities.
- Prune boundary asset points for residual checks in Perturbation kernel.
- Add stress harness: `scripts/experiments/stress_all_methods.jl` with CLI flags.
- Add comparison and plotting scripts; standardize plotting parameters.
- Reorganize `scripts/` into `ci/`, `dev/`, and `experiments/` with a README.
- Tidy coverage handling and `.gitignore` entries for local coverage artifacts.

Notes:
- Test suite passes on Julia 1.11.6. Optional Aqua and JET checks are skipped when not installed.
- A quick smoke run of the stress harness with tiny grids completed and wrote `outputs/_smoke_stress.csv`.

[0.3.0]: https://example.com/releases/v0.3.0

