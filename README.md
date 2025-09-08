# From_EGM_to_NN
**Master’s Thesis – Master in Economics, IP Paris**  
Author: *Matthieu Grenier*

Supervisor : *Pablo Winant*

---

## Overview

This repository contains the code, notebooks, and results for my Master’s thesis in **computational economics**.  
The project investigates how **deep learning** methods can be useful for economists compared to traditional **perturbation** and **projection** methods.

This thesis is inspired by **Maliar et al. (2021)**. The notebooks in the repository replicate and enhance their approach. We select throughout the notebooks problems with a kink in the policy function. This attribute makes it a good stress test for numerical solution methods.

This repository aims to document the results exposed in the thesis. The graphs, tables, and results, are reproducible, as random seeds are fixed.

The Jupyter notebooks in the repository are written using the Julia Programming Language 1.11.6

External readers may feel free to use my work for personnal or educationnal purposes.

---
## References

Maliar, L., Maliar, S., & Winant, P. (2021). Deep learning for solving dynamic economic models. Journal of Monetary Economics.

Judd, K. L. (1998). Numerical Methods in Economics. MIT Press.

---

## Quickstart

- Instantiate and test:
  - `julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`

- Minimal example:
  - `using ThesisProject`
  - `cfg = load_config("config/smoke_config/smoke_config.yaml")`
  - `validate_config(cfg)`
  - `model = build_model(cfg)`
  - `method = build_method(cfg)`
  - `sol = solve(model, method, cfg)`

- Plotting (optional):
  - `using Plots`
  - `plot_policy(sol)`; `plot_euler_errors(sol)`

---

## Config Format

Configs are YAML files loaded via `load_config`, which recursively converts keys to symbols. A minimal config requires:
- `:model`: at least `name`
- `:params`: model parameters
- `:grids`: `Na`, `a_min`, `a_max`
- `:solver`: `method` (e.g., `"EGM"`)

Optional solver fields supported by the EGM method:
- `solver.interp_kind`: interpolation for policy evaluation inside EGM. One of `linear`, `pchip`, `monotone_cubic` (default: `linear`).
- `solver.warm_start`: initial policy guess. One of `default`/`half_resources` (kernel default), or `steady_state` (sets `a' = a`, so `c = y + (1+r)a - a`). You may also provide a custom initial policy via `init.c` in the config (vector for deterministic, matrix for stochastic), which takes precedence.

---

## Smoke Checks

- Run a fast regression sweep on key configs:
  - `julia --project scripts/smoke.jl`
  - Exit code is non-zero if any config fails (useful in CI).
  - You can pass specific configs: `julia --project scripts/smoke.jl config/smoke_config/smoke_config.yaml`

CI is configured via `.github/workflows/ci.yml` to run tests and smoke checks on pushes and PRs.

Validate early with `validate_config(cfg)`; the function throws if something is missing or inconsistent.

---

## Quality Checks (Optional)

The test suite can run Aqua.jl (package hygiene) and JET.jl (type stability/errors) if they are installed. They are optional; tests skip them if unavailable.

- Install once if desired:
  - `julia --project -e 'using Pkg; Pkg.add(["Aqua", "JET"])'`

- Run tests as usual: `Pkg.test()`

---
