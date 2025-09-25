# Repo Inventory

This inventory lists files under `src/`, `scripts/`, `config/`, `test/`, and `ext/` with a one‑line purpose.

## src/

- `src/ThesisProject.jl`: Main module; includes submodules and re‑exports API.
- `src/analysis/SteadyState.jl`: Steady‑state analytics and helpers.
- `src/core/api.jl`: API types and stubs for models, methods, solutions.
- `src/core/model_contract.jl`: Contract functions `get_params/grids/shocks/utility` for models.
- `src/core/model_factory.jl`: Builds concrete model from config (dispatch on `cfg.model.name`).
- `src/core/method_factory.jl`: Builds solver method object from config (`cfg.solver.method`).
- `src/methods/EGM.jl`: EGM method adapter; runs EGM kernels and returns `Solution`.
- `src/methods/Projection.jl`: Projection method adapter; wraps projection kernels.
- `src/methods/Perturbation.jl`: Perturbation method adapter; wraps perturbation kernel.
- `src/models/baseline/ConsumerSaving.jl`: Baseline consumer‑saving model; params, grids, utility, shocks.
- `src/models/shared/Shocks.jl`: Discretization of AR(1) shocks and invariants; utilities.
- `src/sim/panel.jl`: Panel simulation over solved policies with deterministic seeding.
- `src/solvers/common/interp.jl`: Linear and monotone cubic interpolation primitives.
- `src/solvers/common/value_fun.jl`: Value function evaluation given a policy (det/stoch).
- `src/solvers/common/chebyshev.jl`: Chebyshev polynomial utilities for approximation.
- `src/solvers/common/residuals.jl`: Residual/euler‑error helpers shared across solvers.
- `src/solvers/common/validators.jl`: Basic policy validators (monotonicity, positivity, bounds).
- `src/solvers/egm/kernel.jl`: Core EGM kernels (deterministic and stochastic variants).
- `src/solvers/perturbation/kernel.jl`: Core perturbation method kernel.
- `src/solvers/projection/coefficients.jl`: Basis/coefficients helpers for projection.
- `src/solvers/projection/kernel.jl`: Projection solver kernel.
- `src/utils/Config.jl`: YAML config load/validate; converts tables to nested NamedTuples.
- `src/utils/Determinism.jl`: Reproducible RNG helpers; configuration hashing/canonicalization.
- `src/viz/api.jl`: Visualization API stubs that error without plotting backend.
- `src/viz/PolicyPlots.jl`: Plot routines for policies and euler errors (loaded via extension).

## scripts/

- `scripts/README.md`: Scripts layout and usage notes.
- `scripts/ci/smoke.jl`: CI smoke test for configs and quick runs.
- `scripts/ci/ci_local.sh`: Local helper to mimic CI runs.
- `scripts/dev/run_simple_all.sh`: Developer helper to run simple EGM solves.
- `scripts/experiments/compare_egm_projection.jl`: Compare EGM vs Projection methods.
- `scripts/experiments/compare_methods_deviations.jl`: Compare method deviations across configurations.
- `scripts/experiments/generate_baseline_csv.jl`: Generate baseline CSV outputs.
- `scripts/experiments/make_figures_simple.jl`: Produce simple figures from experiment runs.
- `scripts/experiments/robustness_sweep.jl`: Sweep parameters for robustness checks.
- `scripts/experiments/steady_state.jl`: Steady‑state computations and reporting.
- `scripts/experiments/stress_all_methods.jl`: Stress test all methods and configurations.

## config/

- `config/simple_baseline.yaml`: Minimal deterministic baseline configuration.
- `config/simple_stochastic.yaml`: Minimal stochastic configuration (with shocks).
- `config/smoke_cfg_det.yaml`: CI smoke configuration (deterministic).
- `config/smoke_cfg_stoch.yaml`: CI smoke configuration (stochastic).

## test/

- `test/runtests.jl`: Test entry point.
- `test/test_core.jl`: Core API and contracts tests.
- `test/test_solver_options.jl`: Solver option handling tests.
- `test/test_accuracy.jl`: Accuracy checks for solutions/policies.
- `test/test_chebyshev.jl`: Chebyshev utilities tests.
- `test/test_interp.jl`: Interpolation routines tests.
- `test/test_residuals.jl`: Residual computation tests.
- `test/test_value_function.jl`: Value evaluation tests.
- `test/test_determinism.jl`: Deterministic RNG/seed behavior tests.
- `test/test_sim.jl`: Panel simulation tests.
- `test/test_projection.jl`: Projection kernel tests (general).
- `test/test_projection_method.jl`: Projection method adapter tests.
- `test/test_projection_accuracy.jl`: Projection accuracy/approximation tests.
- `test/test_projection_orders.jl`: Projection order variations tests.
- `test/test_shocks.jl`: Shock discretization/invariant distribution tests.
- `test/test_egm_stoch.jl`: EGM with stochastic shocks tests.
- `test/test_stability_extreme.jl`: Stability checks under extreme parameters.
- `test/test_quality.jl`: Quality/consistency tests and smoke checks.
- `test/test_viz.jl`: Visualization API behavior tests.
- `test/utils.jl`: Shared test utilities.

## ext/

- `ext/ThesisProjectPlotsExt.jl`: Package extension that enables Plots‑based policy plotting.