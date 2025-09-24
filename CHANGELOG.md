
## [0.6.0] - 2025-09-24
### Bugfixes & Improvements
- Robust test isolation for NN kernel: dependency injection for diagnostics, no global method overwrites.
- Fixed test failures when running kernel tests individually vs. full suite.
- Updated test harness to avoid method table issues and ensure reliable stubbing.
- No impact on core logic or performance; all changes are test-only or optional arguments.
- Minor: fixed zeroing of Lux parameter/state NamedTuples for compatibility.

### Added (since 0.5.0)
- Consolidated the neural-network kernel test harness, covering the dual-head Lux model, Fischer–Burmeister loss, stochastic diagnostics, and adapter plumbing with lightweight fixtures.
- Extended mixed-precision utilities and preprocessing helpers with targeted unit tests, bringing overall project coverage above 90%.

### Documentation
- Documented the new NN solver options (`objective`, `v_h`, `w_min`/`w_max`, `sigma_shocks`) in the README and configuration audit notes.
- Highlighted the coverage workflow in the README and refreshed mixed-precision guidance.

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

