# Scripts Index

| Script | Purpose | Notes |
|--------|---------|-------|
| `ci/smoke.jl` | Fast solver regression sweep used in CI | Accepts a list of configs via CLI arguments |
| `experiments/make_figures_simple.jl` | Recreate thesis figures for the one-asset model | Writes PNGs to `docs/figures` |
| `experiments/compare_egm_projection.jl` | Benchmark accuracy and wall time of EGM vs projection | CSV summaries under `outputs/diagnostics` |
| `experiments/compare_methods_deviations.jl` | Compare perturbation, projection, and NN responses around steady state | Produces deviation tables in `outputs/diagnostics` |
| `experiments/stress_all_methods.jl` | Stress test methods on kinks and occasionally binding constraints | Heavy runtime; batch configurable |
| `experiments/generate_baseline_csv.jl` | Export baseline residuals and policies for paper tables | CSVs in `results/benchmarks` |
| `dev/run_nn_baseline.jl` | Convenience launcher for the NN training loop | Mirrors `bench_mixedprecision` options |
| `dev/run_pretrain.jl` | Run NN pretraining against EGM targets | Requires existing EGM solution |
| `dev/coverage_report.jl` | Generate lcov coverage artefacts | Use after `Pkg.test()` with coverage enabled |

All scripts are executed with `julia --project path/to/script.jl [args...]`. Outputs are intentionally ignored by git; copy artefacts to `docs/` if you need to publish them.
