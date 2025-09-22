#!/usr/bin/env bash
# scripts/fast_test.sh
# Runs a quick test cycle: precompile (once) then run tests with FAST_TEST=1
# Usage: from repo root: ./scripts/fast_test.sh
# Optionally pass --no-precompile to skip the precompile step if already done.

set -euo pipefail

NO_PRECOMPILE=0
if [[ "${1-}" == "--no-precompile" ]]; then
  NO_PRECOMPILE=1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "$NO_PRECOMPILE" -eq 0 ]]; then
  echo "Precompiling project (this speeds up subsequent runs)..."
  julia --project=. scripts/precompile.jl
else
  echo "Skipping precompile step as requested."
fi

echo "Running fast tests (FAST_TEST=1)..."
FAST_TEST=1 julia --project=. -e 'using Pkg; Pkg.test()'
