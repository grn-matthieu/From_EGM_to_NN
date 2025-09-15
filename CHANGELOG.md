# Changelog

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
