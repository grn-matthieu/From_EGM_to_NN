# Scripts Index

## Layout

- `ci/`: smoke/coverage helpers that run in CI and locally.
- `thesis/figures/`: one-off figure generators referenced in the thesis.
- `thesis/benchmarks/`: experiment drivers that reproduce the quantitative comparisons.
- `thesis/reports/`: scripts that export CSV/summary artefacts for tables and appendices.
- `utils/`: tiny helpers shared across the thesis scripts.

All scripts are executed via `julia --project path/to/script.jl [args...]`. Outputs land under `outputs/` or `results/` and remain ignored by git; copy artefacts to `docs/` if you need to publish them.

## Thesis-critical scripts

### Figures

| Script | Focus | Output |
| --- | --- | --- |
| `thesis/figures/make_figures_simple.jl` | Recreate the one-asset deterministic vs stochastic policy/Euler-error plots. | PNGs in `outputs/` (copy to `docs/figures` if needed). |

### Benchmarks & diagnostics

| Script | Focus | Output |
| --- | --- | --- |
| `thesis/benchmarks/compare_egm_projection.jl` | EGM vs Projection accuracy plots for the baseline config. | Policy/euler-error PNGs under `outputs/`. |
| `thesis/benchmarks/compare_methods_deviations.jl` | Deviations between EGM, Projection, and Perturbation policies/values. | PNGs under `outputs/` plus option heatmaps. |
| `thesis/benchmarks/run_all_methods.jl` | Batch runner that solves every method and tracks runtime/residuals. | CSV in `outputs/` summarizing solver metrics. |
| `thesis/benchmarks/robustness_sweep.jl` | β/σ sweeps (deterministic & stochastic) for robustness tables. | CSV `outputs/egm_robustness_sweep.csv`. |
| `thesis/benchmarks/steady_state.jl` | Analytical vs numerical steady-state validation. | Console summary plus optional saved stats. |
| `thesis/benchmarks/stress_all_methods.jl` | Stress harness across β/σ grids for all methods. | CSV under `outputs/` with pass/fail metadata. |

### Reports & tables

| Script | Focus | Output |
| --- | --- | --- |
| `thesis/reports/generate_baseline_csv.jl` | Export deterministic/stochastic baseline CSVs (policies, residuals). | `outputs/egm_baseline_{det,stoch}.csv`. |
| `thesis/reports/methodology_report.jl` | Automated methodology sweep (grid density, tolerances, interpolation). | CSV + summary + manifest in `results/methodology/`. |

## CI & shared helpers

| Script | Focus | Notes |
| --- | --- | --- |
| `ci/smoke.jl` | Fast regression sweep across smoke configs. | Accepts config paths as CLI args. |
| `ci/ci_local.sh` | Mirrors CI locally (formatting, pre-commit, tests, coverage). | Bash script run from repo root. |
| `utils/config_helpers.jl` | Merge/override helpers for YAML configs inside scripts. | Imported by most thesis drivers. |

## Retired during cleanup

The following exploratory scripts were removed because they duplicated the thesis drivers or lived as ad-hoc debugging tools: `experiments/sweep_vh.jl`, `experiments/compare_cpu_gpu_solver.jl`, `experiments/evaluate_nn_kernel.jl`, `experiments/debug_bcmc_vs_aio.jl`, and `experiments/nn/compare_bcmc_aio.jl`.
