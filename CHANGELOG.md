## [0.8.0] - 2025-10-13
### Added
- CSVAR model: new consumer-saving VAR variant with builders/helpers and configs.
- EGM kernel: CSVAR wiring; integration now supports MC and Gauss–Hermite.
- Projection kernel: CSVAR support.
- Perturbation kernel: CSVAR support and graceful fallback for unsupported combos.
- NN solver: CSVAR support end-to-end, incl. evaluation/diagnostics and objective auto-pick.
- Method adapters: propagate tensor metadata; centralized warm-start/policy validation.
- Utilities: helpers for CSVAR-shaped objects and shared method utilities.

### Changed
- API/Helpers: refactor `solve` into focused helpers; refactor `EGM.solve` helpers.
- Validation: decomposed `validate_config` into clearer sections; configs updated to use Σ (Sigma).
- Policy/value layout: VF module accepts matrix-shaped policies; warm start and policy layout updated.
- Integration: factored EGM integration steps into the integration module; replaced placeholders with real MC + GH.
- Style: CRRA parameter renamed from sigma to gamma for clarity.

### Fixed
- Determinism: enforced RNG and determinism rules across kernels; projection RNG handling corrected.
- CSVAR parity: NN kernel now matches CS (non-VAR) outputs; d=1 CSVAR edge-cases handled in Perturbation.
- Tensor grids: EGM kernel updated to operate on tensor grids; model adapters wire tensor grid when d>1 and expose metadata.
- Gradients: removed array mutation patterns to keep AD safe in NN paths.
- Tests/configs: baseline comparisons (N=1), config alignment, and tolerances tuned for CI.

### Tests
- Added unit tests for CSVAR utilities and model handling across solvers.
- Added unit tests for EGM tensor kernel and comparisons CSVAR (d=1) vs CS.
- Added integration tests for CSVAR solvers; refreshed test procedures.

### Docs
- README updates for CSVAR model and usage.
- Cleaned deprecated release notes.

### CI/Build/Chore
- Manifest refresh (2025-10-12) and pre-commit housekeeping; removed deprecated scripts.
- Lowered residual tolerance in CI smoke to stabilize runs.

## [0.7.0] - 2025-10-08
- Refactor + change test procedure to match with higher standards
- Clean the NN repo for pollution in the solvers; now yields acceptable results
- Add CUDA compatibility to run the NN kernel on the GPU if requested
- Add utils refactoring recurrent operations in overall funs
- Fix bad handling of the RNG. Master RNG/seed is now set at validation, and cannot be modified afterwards.

### Upcoming...
- New model : high-dimensional setting to compare actual performance
- New operator : AiO upgrades to bc-MC with optimal N to minimize variance.
- Minor tweaks to match exactly the methodology section
- Better docs and test coverage.



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

