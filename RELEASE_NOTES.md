# Release Notes — v0.5.0 (2025-09-24)

This document summarizes the `v0.5.0` release and highlights migration notes for downstream users.

## Highlights

- Stability: Core solver implementations and public APIs are considered stable for the `v0.5` line.
- Refactor: Reorganized core modules and test layout for improved maintainability.
- NN solver: Added a neural-network-based solver to experiment with learned policy/value approximations.
- Tests: Greatly improved test coverage across `analysis`, `methods`, and `solvers` modules; added clearer smoke and unit harnesses.

## Breaking changes / Migration notes

- `load_config` now returns nested `NamedTuple`s. Code using dictionary-like indexing such as `cfg[:key]` must be updated to use dot-access: `cfg.key`.

## Quick upgrade

1. Update your local checkout to the new tag or branch.
2. Bump any downstream references to `ThesisProject` to `v0.5.0`.
3. Run the test suite and smoke harnesses:

```powershell
# From repository root (PowerShell)
julia --project=test -e "using Pkg; Pkg.instantiate(); Pkg.test()"
```

## Notes

- If you rely on optional extensions (Plots, CUDA workflows), confirm compatibility with your local environment and the packages pinned in `Project.toml`.

---

Thanks for using the project — update issues and questions at the repository issue tracker.
