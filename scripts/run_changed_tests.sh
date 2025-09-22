#!/usr/bin/env bash
# scripts/run_changed_tests.sh
# Detect changed files and run related tests using scripts/run_single_test.jl

set -euo pipefail

NO_PRECOMPILE=0
if [[ "${1-}" == "--no-precompile" ]]; then
  NO_PRECOMPILE=1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Detecting changed files (git)..."

changed=()

# Try origin/main, main, fallback to last commit
if git rev-parse --verify origin/main >/dev/null 2>&1; then
  mapfile -t difflist < <(git diff --name-only origin/main...HEAD || true)
elif git rev-parse --verify main >/dev/null 2>&1; then
  mapfile -t difflist < <(git diff --name-only main...HEAD || true)
else
  mapfile -t difflist < <(git diff --name-only HEAD~1..HEAD || true)
fi

changed+=("${difflist[@]}")

# include modified and untracked files
mapfile -t modlist < <(git ls-files -m || true)
mapfile -t untracked < <(git ls-files --others --exclude-standard || true)
changed+=("${modlist[@]}" "${untracked[@]}")

# dedupe
IFS=$'\n' read -r -d '' -a changed_unique < <(printf "%s\n" "${changed[@]}" | awk '!seen[$0]++' && printf '\0')

if [[ ${#changed_unique[@]} -eq 0 ]]; then
  echo "No changed files detected. Nothing to run."
  exit 0
fi

printf "Changed files:\n"
printf "%s\n" "${changed_unique[@]}"

tests=()

for f in "${changed_unique[@]}"; do
  if [[ "$f" == test/*.jl ]]; then
    tests+=("$f")
    continue
  fi
  if [[ "$f" != *.jl ]]; then
    continue
  fi
  basename="$(basename "$f" .jl)"
  # search for basename in test files
  matches=( $(grep -R --line-number --recursive --binary-files=without-match --exclude-dir=.git -n "\b$basename\b" test || true) )
  if [[ ${#matches[@]} -gt 0 ]]; then
    for m in "${matches[@]}"; do
      path=${m%%:*}
      tests+=("$path")
    done
    continue
  fi
  # fallback: search for path fragments
  IFS='/' read -ra parts <<< "$f"
  for p in "${parts[@]}"; do
    if [[ ${#p} -lt 3 ]]; then continue; fi
    m2=( $(grep -R --line-number --recursive --binary-files=without-match --exclude-dir=.git -n "$p" test || true) )
    if [[ ${#m2[@]} -gt 0 ]]; then
      for m in "${m2[@]}"; do
        path=${m%%:*}
        tests+=("$path")
      done
      break
    fi
  done
done

# dedupe
IFS=$'\n' read -r -d '' -a tests_unique < <(printf "%s\n" "${tests[@]}" | awk '!seen[$0]++' && printf '\0')

if [[ ${#tests_unique[@]} -eq 0 ]]; then
  echo "No matching tests found for changed files. Consider running scripts/fast_test.sh or full test suite."
  exit 0
fi

printf "Tests to run:\n"
printf "%s\n" "${tests_unique[@]}"

if [[ $NO_PRECOMPILE -eq 0 ]]; then
  echo "Precompiling project to speed test runs..."
  julia --project=. scripts/precompile.jl
fi

exitcode=0
for t in "${tests_unique[@]}"; do
  echo "Running test: $t"
  julia --project=. scripts/run_single_test.jl "$t" || exitcode=$?
done

exit $exitcode
