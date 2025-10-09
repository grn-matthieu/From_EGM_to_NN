
# From_EGM_to_NN

A Julia package and set of experiments comparing EGM, projection, perturbation, and neural-network solvers for a consumption-saving model.

Core points
- Package: `ThesisProject` (entry point for model building and solvers).
- Minimal Julia version: 1.11
- Configs: YAML files in `config/` drive experiments and solver options.

Quick install

1. Clone the repository and instantiate the environment:

```bash
git clone https://github.com/matthieugrenier/From_EGM_to_NN.git
cd From_EGM_to_NN
julia --project -e 'using Pkg; Pkg.instantiate()'
```

2. (Optional) Add plotting packages if needed:

```bash
julia --project -e 'using Pkg; Pkg.add(["Plots"])'
```

Basic usage

```julia
using ThesisProject

cfg = load_config("config/smoke_cfg_det.yaml")
model = build_model(cfg)
method = build_method(cfg)
sol = solve(model, method, cfg)

# inspect results
sol.resid
plot_policy(sol)    # requires Plots
```

Running tests

Run the full test suite:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Repository layout (high level)

- `src/` : package source (core API, models, solvers, methods, utils)
- `config/` : YAML configs used by scripts and experiments
- `scripts/` : experiment and CI scripts
- `docs/`, `results/`, `outputs/` : figures and outputs
- `test/` : unit and integration tests

Notes

- Neural-network solver lives under `src/solvers/nn` and uses dual-head networks for policy outputs. Tunable options are exposed in the `solver` block of YAML configs.
- Reproducibility: RNGs are derived from a master RNG via utilities in `src/utils/Determinism.jl` to avoid mutating the global RNG.

License

This project is licensed under the MIT License (see `LICENSE`).

