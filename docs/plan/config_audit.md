# Config Validation Audit



This audit runs validation on both deterministic and stochastic example configs, enumerates required fields and defaults inferred from the codebase, and lists concrete assertions to add to `validate_config` to catch misconfigurations earlier.



Validated configs:

- [OK] `config/simple_baseline.yaml`

- [OK] `config/simple_stochastic.yaml`

- [OK] `config/smoke_config/smoke_config.yaml`

- [OK] `config/smoke_config/smoke_config_stochastic.yaml`



## Required Fields (current)



Top-level (from `src/utils/Config.jl`):

- `model.name`: required string; used by model factory

- `params`: required table

- `grids.Na`: required Integer (> 1)

- `grids.a_min`: required Real

- `grids.a_max`: required Real (must be > `a_min`)

- `solver.method`: required (e.g., `"EGM"`)



ConsumerSaving model requirements (from `src/models/baseline/ConsumerSaving.jl` and solver kernels):

- `params.s`: CRRA coefficient (Real)

- `params.r`: interest rate (Real)

- `params.y`: income level (Real)

- `params.β` (Greek β) is required in value evaluation. Note: the EGM kernels access `β` directly, so configs should contain `β` to avoid later errors.



Optional sections used elsewhere:

- `shocks`: optional; when `active: true`, stochastic model is built

- `random.seed`: optional; used by simulation code for reproducibility

- `utility.u_type`: optional; examples use `CRRA` (not currently enforced)

- `experiment.*`: optional metadata, not used by core APIs



## Defaults (observed in code)



EGM method options (from `src/methods/EGM.jl`):

- `solver.tol`: default `1e-6`

- `solver.tol_pol`: default `1e-6`

- `solver.maxit`: default `1000`

- `solver.interp_kind`: default `:linear` (accepts symbol or string)

- `solver.verbose`: default `false`

- `solver.warm_start`: default `:default` (also accepts `:half_resources`, `:none`, or `:steady_state`)

- `init.c`: optional custom initial policy; if provided, overrides warm start



EGM kernel internals (not user-configurable today; noted for completeness):

- Relaxation `relax = 0.5`, patience `= 50`, small epsilon `≈ 1e-10` for progress detection



Shocks (from `src/models/shared/Shocks.jl`):

- `shocks.active`: default `false` if not present (checked in model builder)

- `shocks.method`: default `"tauchen"` (also supports `"rouwenhorst"`)

- `shocks.ρ_shock`: default `0.0`

- `shocks.σ_shock`: default `0.0`

- `shocks.Nz`: default `7`

- `shocks.m`: default `3.0` (tauchen only)

- `shocks.validate`: default `true` (checks invariant distribution)



Random seed (from `src/sim/panel.jl`):

- `random.seed`: optional; if omitted, routines derive seeds deterministically when needed



## Gaps in validate_config (missing assertions)



Current `validate_config` is intentionally minimal. The following checks should be added to fail fast with clear errors and prevent downstream crashes:



Model and method:

- Accept only known `solver.method` values: `EGM`, `Projection`, `Perturbation`.

- If `solver.interp_kind` is present, validate it is one of: `linear`, `pchip`, `monotone_cubic` (case-insensitive; allow symbol or string).

- If `solver.warm_start` is present, validate it is one of: `default`, `half_resources`, `none`, `steady_state` (case-insensitive; allow symbol or string).



Params (ConsumerSaving):

- Require keys in `:params`: `β` (Greek) or `β` (ascii), `s`, `r`, `y`.

- If only `β` is present, either (a) error with a helpful message to use `β`, or (b) proactively copy `β` to `β` inside `cfg` during validation to align with kernels.

- Validate domains: `0 < β < 1`, `s > 0` (allow `≈ 1` for log utility), `r > -1`, `y > 0`.

- Type checks: numeric scalars for all the above.



Grids:

- Type checks: `Na` Integer, `a_min`/`a_max` Real.

- Range checks: `Na ≥ 2`, `a_max > a_min`.



Shocks (only when `shocks.active == true`):

- Validate `shocks.method` ∈ {`tauchen`, `rouwenhorst`} (case-insensitive).

- Validate `-1 < ρ_shock < 1` and `σ_shock ≥ 0`.

- Validate `Nz` is Integer and `Nz ≥ 1`.

- If method is `tauchen`, validate `m` is Real and `m > 0`.

- If `validate` present, ensure it is Bool.



Warm start and initial policy:

- If `solver.warm_start == steady_state`, ensure required params exist (`y`, `r`, `a_min`).

- If `init.c` is provided:

  - Deterministic case (no shocks): `length(init.c) == grids.Na`.

  - Stochastic case (`shocks.active == true`): `size(init.c) == (grids.Na, shocks.Nz)`.

  - All entries positive and feasible given budget: `0 < c ≤ y + (1+r)a - a_min` elementwise.



Utility:

- If `utility.u_type` is present, restrict to supported values (currently `CRRA`) to avoid silent mismatches with how utility is constructed in `ConsumerSaving`.



Random seed:

- If `random.seed` is present, ensure it is an Integer (or coercible to one).



Diagnostics and friendliness:

- On failure, error messages should name the exact missing/invalid key and show the offending value.

- Where helpful, suggest a fix (e.g., “use Greek `β` throughout configs”).



## Suggested structure for the improved validator



Without changing public API, enhance `validate_config(cfg::NamedTuple)` to:

- Normalize a few inputs (e.g., lowercasing method/interp_kind, mapping `β` → `β` if present).

- Perform the assertions listed above in clearly delimited blocks: top‑level, params, grids, solver, shocks, init.

- Return `true` and optionally mutate `cfg` minimally only for harmless normalizations (string → symbol, `β` → `β`). If mutation is undesirable, validate and error instead.



This will make `validate_config` consistent with actual usage across:

 - `ModelFactory.build_model` and `ConsumerSaving.build_cs_model`

 - EGM method defaults and kernels

 - Shocks discretization utilities

 - Value function evaluation



## Method Coverage Addendum (Projection and Perturbation)



Projection (src/methods/Projection.jl, src/solvers/projection/kernel.jl):

- Options and defaults:

  - solver.tol: 1e-6 (default)

  - solver.maxit: 1000 (default)

  - solver.verbose: false (default)

  - solver.orders: defaults to [grids.Na - 1] when empty or missing

  - solver.Nval: defaults to grids.Na

- Validator assertions to add:

  - orders is a non-empty Vector{Int}; each order satisfies 0 <= order <= grids.Na - 1.

  - Nval is Int and Nval >= 2.

  - tol > 0, maxit >= 1; verbose Bool if provided.



Perturbation (src/methods/Perturbation.jl, src/solvers/perturbation/kernel.jl):

- Options and defaults:

  - solver.a_bar: default midpoint of [a_min, a_max] if nothing

  - solver.order: 1 by default; if >= 2, kernel attempts a local second-order fit

  - solver.h_a, solver.h_z: optional steps; kernel derives if nothing when order >= 2

  - solver.tol_fit: 1e-8 (default), solver.maxit_fit: 25 (default)

- Validator assertions to add:

  - order is Int >= 1.

  - If order >= 2: if provided, h_a > 0 (Real), and in stochastic case if provided, h_z > 0 (Real).

  - If a_bar provided: Real and a_min <= a_bar <= a_max.

  - tol_fit > 0, maxit_fit >= 1.



Common to all methods:

- solver.method: enforce one of EGM, Projection, Perturbation (case-insensitive; symbol/string accepted).

- Keep existing EGM-specific checks (interp_kind, warm_start) and add the above for Projection and Perturbation.



