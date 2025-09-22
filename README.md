# From_EGM_to_NN

## Purpose

This repository accompanies the IP Paris master’s thesis **“From EGM to Neural Networks”**. It evaluates how neural-network-based policy approximations compare with perturbation and projection techniques on dynamic stochastic models, with a focus on kinks and occasionally binding constraints. All source code is shipped in the Julia package `ThesisProject` and every experiment can be reproduced from scripts included in this tree.

## Package Architecture

```
ThesisProject
├── core/           – public API, model/method builders, validation
├── models/         – baseline consumption-saving model and shared utilities
├── solvers/        – EGM, projection, perturbation, and NN training kernels
├── methods/        – thin adapters mapping configs to solver kernels
├── utils/          – configuration loading, determinism helpers
├── scripts/        – CLI front-ends for experiments and CI workflows
└── docs/           – figures, notes, and companion material
```

## Installation

1. Install Julia ≥ **1.11** (1.11.x series recommended).
2. Clone this repository and activate the environment:
   ```bash
   git clone https://github.com/matthieugrenier/From_EGM_to_NN.git
   cd From_EGM_to_NN
   julia --project -e 'using Pkg; Pkg.instantiate()'
   ```
3. Optional (for plots): `julia --project -e 'using Pkg; Pkg.add("Plots")'`.

## Quick Start

```julia
using ThesisProject

cfg = load_config("config/smoke_config/smoke_config.yaml")
validate_config(cfg)

# load_config now returns a nested NamedTuple, so fields are dot-accessible
@show cfg.model.name
@show cfg.params.β
@show cfg.solver.method

model = build_model(cfg)
method = build_method(cfg)
sol = solve(model, method, cfg)
```

- Evaluate residuals: `sol.resid`
- Plot policies (requires `Plots`):
  ```julia
  using Plots
  plot_policy(sol)
  plot_euler_errors(sol)
  ```

## Configuration Schema (YAML)

| Key            | Required | Notes |
|----------------|----------|-------|
| `model.name`   | ✅        | `"Baseline"` provided; extendable.
| `params`       | ✅        | Structural parameters (β, R, σ, …).
| `grids.Na`     | ✅        | Number of asset points; other grid bounds under `grids`.
| `solver.method`| ✅        | One of `"EGM"`, `"Projection"`, `"Perturbation"`, `"NN"`.
| `solver` block | ⚙️       | Method-specific options (e.g., `epochs`, `clip_norm`).
| `random.seed`  | 🔁       | Optional; guarantees reproducibility without `Random.seed!`.
| `logging`      | 📈       | (NN) specify directory and CSV logging behaviour.

`load_config` returns a nested `NamedTuple` with symbol keys, so configuration values are accessed via property syntax. Use `validate_config(cfg)` to receive descriptive errors when entries are missing or inconsistent.

```julia
cfg = load_config("config/smoke_config/smoke_config.yaml")
cfg.params.β        # → 0.96
cfg.grids.Na        # → 40
cfg.solver.method   # → :EGM
cfg.logging.dir     # → "/tmp/nn_logs" (if present)
```

Because `NamedTuple`s are immutable, create experiment-specific overrides with `merge`, e.g. `cfg = merge(cfg, (; solver = merge(cfg.solver, (; method = :Projection, tol = 1e-7, )),))`.

## Method Matrix

| Feature                           | EGM | Projection | Perturbation | Neural Network |
|-----------------------------------|:---:|:----------:|:------------:|:--------------:|
| Deterministic baseline            | ✅  | ✅         | ✅           | ✅             |
| Stochastic shocks                 | ✅  | ✅         | ⚠️ (linear)  | ✅             |
| Handles kinks / non-smoothness    | ✅  | ✅         | ❌           | ✅             |
| Automatic differentiation         | ❌  | ❌         | ⚠️ (manual)  | ✅ (Zygote)    |
| Mixed-precision support           | ❌  | ❌         | ❌           | Deprecated     |
| CSV logging built-in              | ❌  | ❌         | ❌           | ✅ (optional)  |

## Reproducibility

- Deterministic seeds are derived via `utils/Determinism.make_rng` without mutating Julia’s global RNG.
- Each `scripts/experiments/*.jl` entry accepts `--config path/to/config.yaml` and writes outputs under `outputs/` (ignored by git).
- The CI smoke test (`scripts/ci/smoke.jl`) runs the fast configs used in regression testing.
- A lightweight precompile workload is provided so that `using ThesisProject` warms essential code paths.
- GitHub tags matching `v*` trigger a release workflow that regenerates the main figures and uploads them as downloadable artifacts.

## Scripts & Expected Outputs

| Script                                      | Description                                      | Output |
|---------------------------------------------|--------------------------------------------------|--------|
| `scripts/ci/smoke.jl`                       | Fast regression sweep used in CI                 | Logs to stdout |
| `scripts/experiments/make_figures_simple.jl`| Recreates thesis figures for the simple model    | PNGs in `docs/figures` |
| `scripts/experiments/compare_egm_projection.jl` | Benchmark EGM vs projection speed & accuracy | CSV summaries in `outputs/diagnostics` |
| `scripts/experiments/stress_all_methods.jl` | Stress tests with kinks                          | CSV & JSON dumps in `outputs/diagnostics` |
| `scripts/experiments/compare_methods_deviations.jl` | Deviations around deterministic steady state | `results/benchmarks` (ignored by git) |

A concise index of developer helpers lives under [`scripts/README.md`](scripts/README.md).

## Testing

Run the full suite:
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Additional property tests check:
- EGM policy monotonicity by verifying marginal utility ordering over random grids.
- Projection residuals at off-grid test points.
- Neural-network feasibility on representative draws.

Install Aqua.jl and JET.jl for deeper hygiene/type checks:
```bash
julia --project -e 'using Pkg; Pkg.add(["Aqua", "JET"])'
```

## Licensing & Citation

This repository is released under the [MIT License](LICENSE). When using it in academic work, please cite the thesis and the original numerical method references highlighted below.

## Troubleshooting

- **Slow first load:** ensure you have Julia ≥1.11; precompile reduces latency after the first `using ThesisProject`.
- **BLAS multithreading issues:** set `JULIA_NUM_THREADS` to the desired value or `1` for deterministic comparisons.
- **Mixed precision:** explicit mixed-precision utilities have been removed from the public API. Use `NNUtils.to_fp32` to convert tensors back to Float32 where needed. If you require GPU mixed-precision kernels, extend `src/solvers/nn/` with project-specific implementations.

Happy experimenting!

## References

- Maliar, L., Maliar, S., & Winant, P. (2021). *Deep learning for solving dynamic economic models*. Journal of Monetary Economics.
- Judd, K. L. (1998). *Numerical Methods in Economics*. MIT Press.
- Fernandez-Villaverde, J., et al. (2020). *Solving the Income Fluctuation Problem with Neural Networks*. Econometrica.
