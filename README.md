

# From_EGM_to_NN

A small Julia project that compares methods for solving a consumption–saving model: endogenous grid method (EGM), projection, perturbation, and a neural-network solver. The repository includes the package code, experiments, and example configs so you can reproduce the main results or run quick checks.

What's here

- `src/` — package source and solver implementations (entry point: `ThesisProject.jl`).
- `config/` — YAML files used to run experiments and control solver options.
- `scripts/` — experiment runners and CI scripts.
- `docs/`, `results/`, `outputs/` — figures, logs, and outputs from experiments.
- `test/` — unit and integration tests.

Quick start

1. Clone and instantiate the Julia environment:

```bash
git clone https://github.com/grn-matthieu/From_EGM_to_NN.git
cd From_EGM_to_NN
julia --project -e 'using Pkg; Pkg.instantiate()'
```

2. Run a small example (uses the bundled config files):

```julia
using ThesisProject

cfg = load_config("config/smoke_cfg_det.yaml")
model = build_model(cfg)
method = build_method(cfg)
sol = solve(model, method, cfg)

println("Residuals: ", sol.resid)
```

Running tests

Execute the test suite from the project environment:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Configuration and experiments

Configs live in `config/`. There are smoke tests and deterministic/stochastic variants to try quick runs. The `scripts/experiments` folder contains example experiment drivers used to produce figures in `docs/` and `results/`.

License and contact

This project is released under the MIT license (see `LICENSE`). If you want to reach out, please open an issue on GitHub with details and a minimal reproduction.

Simple notes for contributors

- Use `julia --project` to run code or tests so dependencies are picked up from `Project.toml`.
- Tests live in `test/`; smaller quick checks are under `test/unit` and `test/integration`.

