#!/usr/bin/env bash
set -euo pipefail
julia --project=.formatter -e 'using Pkg; Pkg.instantiate()'
python -m pip install --upgrade pip pre-commit
julia --project=. -e 'using Pkg; Pkg.instantiate()'
pre-commit run --all-files --show-diff-on-failure
julia --project=. -e 'using Pkg; Pkg.test(coverage=true)'
julia --project=. -e 'using Coverage; cov=process_folder("src"); LCOV.writefile("lcov.info", cov)'
julia -e 'using Pkg; Pkg.activate(temp=true); Pkg.add("Coverage"); using Coverage: LCOV, get_summary; c=LCOV.readfile("lcov.info"); println(get_summary(c))'

# Cleanup local coverage artifacts produced during the run
find . -type f \( -name "*.cov" -o -name "*.info" -o -name "lcov.info" \) -print -delete || true
