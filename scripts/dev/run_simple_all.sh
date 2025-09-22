#!/usr/bin/env bash
# Replicate baseline results (deterministic and stochastic)
# for all three methods: EGM, Projection, Perturbation.
# Keeps two grid sizes for quick reproducibility checks.

set -euo pipefail

# Resolve repository root (two levels up from scripts/dev)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Determinism defaults (override by exporting before calling this script)
export SEED="${SEED:-20240915}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"

METHODS=(EGM Projection Perturbation)
CASES=(deterministic stochastic)

# Grid sizes to test
GRIDS=(100 1000)

for METHOD in "${METHODS[@]}"; do
  for CASE in "${CASES[@]}"; do
    # Pick config per case
    if [[ "$CASE" == "deterministic" ]]; then
      CFG_PATH="$ROOT/config/simple_baseline.yaml"
    else
      CFG_PATH="$ROOT/config/simple_stochastic.yaml"
    fi

    for NA in "${GRIDS[@]}"; do
      echo "==> Running $METHOD | $CASE | Na=${NA}"
      ROOT="$ROOT" NA="$NA" METHOD="$METHOD" CFG_PATH="$CFG_PATH" SEED="$SEED" JULIA_NUM_THREADS="$JULIA_NUM_THREADS" julia --project="$ROOT" -e '
        using Random, LinearAlgebra
        # Fixed seed and single-threaded BLAS for determinism
        seed = parse(Int, get(ENV, "SEED", "20240915"))
        Random.seed!(seed)
        try
            BLAS.set_num_threads(1)
        catch
        end
        using ThesisProject
        root = ENV["ROOT"]
        Na = parse(Int, ENV["NA"])
        method = Symbol(ENV["METHOD"])
        cfg_path = ENV["CFG_PATH"]

        include(joinpath(root, "scripts", "utils", "config_helpers.jl"))
        using .ScriptConfigHelpers

        cfg_loaded = load_config(cfg_path)
        validate_config(cfg_loaded)
        cfg_nt = dict_to_namedtuple(cfg_loaded)
        cfg_nt = merge_section(cfg_nt, :grids, (; Na = Na))
        cfg_nt = merge_section(cfg_nt, :solver, (; method = method))
        cfg_nt = merge_config(cfg_nt, (; method = method))
        cfg_dict = namedtuple_to_dict(cfg_nt)

        model = build_model(cfg_dict)
        meth = build_method(cfg_dict)
        sol = solve(model, meth, cfg_dict)
        println("$(method) | $(basename(cfg_path)) | Na=$(Na): resid=$(sol.metadata[:max_resid]) iters=$(sol.metadata[:iters]) seed=$(seed) julia=$(VERSION)")
      '
    done
  done
done
