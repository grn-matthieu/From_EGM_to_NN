#!/usr/bin/env bash
# Run the simple consumer-saving model with the EGM solver on two grid sizes.
# This script is meant for quick reproducibility checks.

set -euo pipefail

# Resolve repository root (two levels up from scripts/dev)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Grid sizes to test
GRIDS=(100 1000)

for NA in "${GRIDS[@]}"; do
    echo "==> Running EGM baseline with Na=${NA}"
    ROOT="$ROOT" NA="$NA" julia --project="$ROOT" -e '
        using ThesisProject
        root = ENV["ROOT"]
        Na = parse(Int, ENV["NA"])
        cfg = load_config(joinpath(root, "config", "simple_stochastic.yaml"))
        cfg[:grids][:Na] = Na
        validate_config(cfg)
        model = build_model(cfg)
        method = build_method(cfg)
        sol = solve(model, method, cfg)
        println("grid $(Na): resid=$(sol.metadata[:max_resid]) iters=$(sol.metadata[:iters])")
    '
done

