# From_EGM_to_NN

Master's Thesis — Master in Economics, IP Paris

Author: Matthieu Grenier  •  Supervisor: Pablo Winant

---

## Overview

This repository contains the code and experiments for a master's thesis in computational economics. It studies how deep learning compares with traditional perturbation and projection methods on dynamic models, with emphasis on kinks in policy functions as a stress test for numerical methods.

The implementation centers on the Julia package `ThesisProject` (v0.3.0), pinned to Julia `1.11.6` via `Project.toml`. Results are designed to be reproducible using fixed seeds where relevant.

References that motivate the approach include Maliar, Maliar, and Winant (2021) and Judd (1998).

---

## Quickstart

- Instantiate and run tests:
  - `julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`

- Minimal example:
  - `using ThesisProject`
  - `cfg = load_config("config/smoke_config/smoke_config.yaml")`
  - `validate_config(cfg)`
  - `model = build_model(cfg)`
  - `method = build_method(cfg)`
  - `sol = solve(model, method, cfg)`

- Plotting (optional, via package extension):
  - `using Plots`  # enables `ThesisProjectPlotsExt`
  - `ThesisProject.plot_policy(sol)`; `ThesisProject.plot_euler_errors(sol)`

---

## Configs

Configs are YAML files loaded via `load_config`, with keys converted to symbols. A minimal config requires:
- `:model`: at least `name`
- `:params`: model parameters
- `:grids`: e.g., `Na`, `a_min`, `a_max`
- `:solver`: `method` (e.g., `"EGM"`, `"Projection"`, `"Perturbation"`, `"NN"`)

Selected EGM options:
- `solver.interp_kind`: `linear` (default), `pchip`, or `monotone_cubic`.
- `solver.warm_start`: `default`/`half_resources` or `steady_state`. You can also provide a custom initial policy via `init.c` (vector for deterministic, matrix for stochastic) which takes precedence.

Shock-related option:
- `shocks.validate`: when `true` (default), checks the invariant distribution consistency; set `false` to skip.

Example:

```yaml
shocks:
  validate: false  # disable invariant distribution check
```

---

## Running Solvers

Deterministic or stochastic baseline:

- `using ThesisProject`
- `cfg = load_config("config/simple_baseline.yaml")`  # or `config/simple_stochastic.yaml`
- `validate_config(cfg)`
- `model = build_model(cfg)`
- `method = build_method(cfg)`
- `sol = solve(model, method, cfg)`

Optional plotting (if `Plots` is available):
- `using Plots`
- `ThesisProject.plot_policy(sol)`

Switch to the stochastic setup by using `config/simple_stochastic.yaml`. Both configs can set `random.seed` to control randomness without calling `Random.seed!`.

---

## Smoke Checks

- Run fast regression checks on key configs:
  - `julia --project scripts/ci/smoke.jl`
  - Pass specific configs if desired: `julia --project scripts/ci/smoke.jl config/smoke_config/smoke_config.yaml`

Use `validate_config(cfg)` early; it throws on missing or inconsistent fields.

---

## Experiments

Selected scripts under `scripts/experiments`:
- `make_figures_simple.jl`: generate core figures for the simple model.
- `compare_egm_projection.jl`: compare EGM vs projection methods.
- `compare_methods_deviations.jl`: method comparison on deviations.
- `robustness_sweep.jl`: parameter sweeps for robustness.
- `stress_all_methods.jl`: stress tests across methods.
- `generate_baseline_csv.jl`: export baseline results to CSV.

Each script is runnable with `julia --project path/to/script.jl` and may read from `config/`.

---

## Development

- Quality checks (optional): If installed, the test suite will pick up Aqua.jl (package hygiene) and JET.jl (type stability/errors).
  - Install: `julia --project -e 'using Pkg; Pkg.add(["Aqua", "JET"])'`
  - Run tests: `julia --project -e 'using Pkg; Pkg.test()'`

- Formatting: This repo uses [pre-commit](https://pre-commit.com/) with JuliaFormatter for `.jl` files.
  - `pip install pre-commit`
  - `julia --project -e 'using Pkg; Pkg.add("JuliaFormatter")'`
  - `pre-commit install`
  - Run manually: `pre-commit run --files $(git ls-files "*.jl")`

---

## References

- Maliar, L., Maliar, S., & Winant, P. (2021). Deep learning for solving dynamic economic models. Journal of Monetary Economics.
- Judd, K. L. (1998). Numerical Methods in Economics. MIT Press.

---

External readers may use this work for personal or educational purposes.

