# Scripts Layout

- ci: CI-only utilities and smoke checks.
  - Use `julia --project scripts/ci/smoke.jl` for quick config sanity.
  - Local CI runner: `scripts/ci/ci_local.sh`.

- experiments: Julia scripts that run models, generate figures, or CSVs.
  - Examples: `compare_egm_projection.jl`, `robustness_sweep.jl`, `stress_all_methods.jl`, `steady_state.jl`.

- dev: Developer helpers for local workflows.
  - `dev/run_simple_all.sh` runs a couple of quick EGM solves.
  - Put PowerShell helpers here as `.ps1` if you add them later.

All Julia scripts assume the repo root is two levels up from their directory and handle `Pkg.activate`/`LOAD_PATH` accordingly.
