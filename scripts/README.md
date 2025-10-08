# Scripts Index

# Scripts Overview

This directory now contains only scripts for:
- CI and coverage workflows
- Multi-method experiments and report generation

## Main Scripts

| Script | Purpose | Notes |
|--------|---------|-------|
| `ci/smoke.jl` | Fast solver regression sweep used in CI | Accepts a list of configs via CLI arguments |
| `experiments/make_figures_simple.jl` | Recreate thesis figures for the one-asset model | Writes PNGs to `docs/figures` |
| `experiments/compare_egm_projection.jl` | Benchmark accuracy and wall time of EGM vs projection | CSV summaries under `outputs/diagnostics` |
| `experiments/compare_methods_deviations.jl` | Compare perturbation, projection, and NN responses around steady state | Produces deviation tables in `outputs/diagnostics` |
| `experiments/stress_all_methods.jl` | Stress test methods on kinks and occasionally binding constraints | Heavy runtime; batch configurable |
| `experiments/generate_baseline_csv.jl` | Export baseline residuals and policies for paper tables | CSVs in `results/benchmarks` |
| `experiments/methodology_report.jl` | Automated methodology reporting and experiment documentation | Integrates with config and outputs |
| `experiments/robustness_sweep.jl` | Robustness checks across parameter sweeps | Batch configurable |
| `experiments/steady_state.jl` | Steady state analysis and verification | Supports deterministic and stochastic configs |

All scripts are executed with `julia --project path/to/script.jl [args...]`. Outputs are intentionally ignored by git; copy artefacts to `docs/` if you need to publish them.
