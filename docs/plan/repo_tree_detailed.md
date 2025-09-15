# Detailed Repository Tree

Includes modules, exports, and key include relationships.

## Modules and Exports
- API: exports: AbstractModel
- Chebyshev: exports: chebyshev_nodes
- CommonInterp: exports: interp_linear!, interp_pchip!, InterpKind, LinearInterp, MonotoneCubicInterp
- CommonValidators: exports: is_nondec, is_positive, respects_amin
- ConsumerSaving: exports: (none)
- Determinism: exports: canonicalize_cfg, derive_seed, hash_hex, make_rng
- EGM: exports: EGMMethod
- EGMKernel: exports: solve_egm_det, solve_egm_stoch
- EulerResiduals: exports: euler_resid_det
- MethodFactory: exports: (none)
- ModelContract: exports: (none)
- ModelFactory: exports: (none)
- Perturbation: exports: build_perturbation_method, PerturbationMethod
- PerturbationKernel: exports: solve_perturbation_det, solve_perturbation_stoch
- Projection: exports: ProjectionMethod
- ProjectionCoefficients: exports: solve_coefficients
- ProjectionKernel: exports: solve_projection_det, solve_projection_stoch
- Shocks: exports: discretize, ShockOutput
- SimPanel: exports: simulate_panel
- SteadyState: exports: steady_state_analytic, steady_state_from_policy
- ThesisProject: exports: load_config, plot_euler_errors, plot_policy
- ThesisProjectPlotsExt: exports: (none)
- UtilsConfig: exports: (none)
- ValueFunction: exports: compute_value_policy

## Include Tree (from src/ThesisProject.jl)
- ThesisProject
  - API
  - ModelContract
  - UtilsConfig
  - Determinism
  - Shocks
  - ConsumerSaving
  - ModelFactory
  - CommonInterp
  - ValueFunction
  - Chebyshev
  - EulerResiduals
  - CommonValidators
  - ProjectionCoefficients
  - ProjectionKernel
  - EGMKernel
  - PerturbationKernel
  - EGM
  - Projection
  - Perturbation
  - MethodFactory
  - SimPanel
  - SteadyState
  - api.jl