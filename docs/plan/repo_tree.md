# Repository Tree

ASCII tree of key directories (`src/`, `scripts/`, `config/`, `test/`, `ext/`). Coverage artifacts (`*.cov`) omitted.

```
src
+-- analysis
|   +-- SteadyState.jl
+-- core
|   +-- api.jl
|   +-- method_factory.jl
|   +-- model_contract.jl
|   +-- model_factory.jl
+-- methods
|   +-- EGM.jl
|   +-- Perturbation.jl
|   +-- Projection.jl
+-- models
|   +-- baseline
|   |   +-- ConsumerSaving.jl
|   +-- shared
|       +-- Shocks.jl
+-- sim
|   +-- panel.jl
+-- solvers
|   +-- common
|   |   +-- chebyshev.jl
|   |   +-- interp.jl
|   |   +-- residuals.jl
|   |   +-- validators.jl
|   |   +-- value_fun.jl
|   +-- egm
|   |   +-- kernel.jl
|   +-- perturbation
|   |   +-- kernel.jl
|   +-- projection
|       +-- coefficients.jl
|       +-- kernel.jl
+-- utils
|   +-- Config.jl
|   +-- Determinism.jl
+-- viz
|   +-- api.jl
|   +-- PolicyPlots.jl
|   +-- SimPlots.jl
+-- ThesisProject.jl

scripts
+-- ci
|   +-- ci_local.sh
|   +-- smoke.jl
+-- dev
|   +-- run_simple_all.sh
+-- experiments
|   +-- compare_egm_projection.jl
|   +-- compare_methods_deviations.jl
|   +-- generate_baseline_csv.jl
|   +-- make_figures_simple.jl
|   +-- robustness_sweep.jl
|   +-- steady_state.jl
|   +-- stress_all_methods.jl
+-- README.md

config
+-- smoke_config
|   +-- smoke_config_stochastic.yaml
|   +-- smoke_config.yaml
+-- simple_baseline.yaml
+-- simple_stochastic.yaml

test
+-- runtests.jl
+-- test_accuracy.jl
+-- test_chebyshev.jl
+-- test_core.jl
+-- test_determinism.jl
+-- test_egm_stoch.jl
+-- test_interp.jl
+-- test_perturbation_method.jl
+-- test_projection_accuracy.jl
+-- test_projection_method.jl
+-- test_projection_orders.jl
+-- test_projection_stoch.jl
+-- test_projection.jl
+-- test_quality.jl
+-- test_residuals.jl
+-- test_shocks.jl
+-- test_sim.jl
+-- test_solver_options.jl
+-- test_stability_extreme.jl
+-- test_value_function.jl
+-- test_viz.jl
+-- utils.jl

ext
+-- ThesisProjectPlotsExt.jl
```

