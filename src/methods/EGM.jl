"""
EGM

Adapter that wires the EGM solver kernel into the unified `API.solve` method.
"""
module EGM

using ..API
import ..API: solve

using ..EGMKernel: solve_egm_det, solve_egm_stoch
using ..ValueFunction: compute_value_policy
using ..Determinism: canonicalize_cfg, hash_hex
using ..CommonInterp: LinearInterp, MonotoneCubicInterp
using ..UtilsConfig: maybe
using ..UtilsDiagnostics: mean_abs_error
using ..MethodUtils: build_consumption_initializer, validate_policy!

export EGMMethod

struct EGMMethod <: AbstractMethod
    opts::NamedTuple
end

"""
    build_egm_method(cfg::NamedTuple) -> EGMMethod

Construct an `EGMMethod` using solver options contained in the NamedTuple `cfg`.
"""
function build_egm_method(cfg::NamedTuple)
    solver_cfg = cfg.solver
    egm_cfg = solver_cfg.egm
    ik_raw = egm_cfg.interp_kind
    ik_sym = Symbol(lowercase(string(ik_raw)))
    warm_start = Symbol(lowercase(string(solver_cfg.warm_start)))
    return EGMMethod((
        name = maybe(cfg, :method, solver_cfg.method),
        tol = solver_cfg.tol,
        tol_pol = solver_cfg.tol_pol,
        maxit = solver_cfg.maxit,
        interp_kind = ik_sym,
        verbose = solver_cfg.verbose,
        warm_start = warm_start,
        relax = solver_cfg.relax,
    ))
end

"""
    solve(model::AbstractModel, method::EGMMethod, cfg::NamedTuple; rng=nothing)::Solution

Entry point for the EGM solver. Extracts contract fields, runs a minimal EGM loop, and returns a Solution.
"""
function solve(
    model::AbstractModel,
    method::EGMMethod,
    cfg::NamedTuple;
    rng = nothing,
)::Solution
    # --- Extraction ---
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    # --- Warm-start policy initialization ---
    init_cfg = maybe(cfg, :init)
    custom_c_data = maybe(init_cfg, :c)
    custom_c_vec = custom_c_data isa AbstractVector ? custom_c_data : nothing
    custom_c_mat = custom_c_data isa AbstractMatrix ? custom_c_data : nothing

    c_init =
        S === nothing ?
        build_consumption_initializer(
            p,
            g;
            shocks = nothing,
            warm_start = method.opts.warm_start,
            custom_c = custom_c_vec,
        ) :
        build_consumption_initializer(
            p,
            g;
            shocks = S,
            warm_start = method.opts.warm_start,
            custom_c = custom_c_mat,
        )

    # --- Solution ---
    ik = method.opts.interp_kind
    interp = ik == :linear ? LinearInterp() : MonotoneCubicInterp()
    sol =
        S === nothing ?
        solve_egm_det(
            p,
            g,
            U;
            tol = method.opts.tol,
            tol_pol = method.opts.tol_pol,
            maxit = method.opts.maxit,
            interp_kind = interp,
            relax = method.opts.relax,
            verbose = method.opts.verbose,
            c_init = c_init,
        ) :
        solve_egm_stoch(
            p,
            g,
            S,
            U;
            tol = method.opts.tol,
            tol_pol = method.opts.tol_pol,
            maxit = method.opts.maxit,
            interp_kind = interp,
            relax = method.opts.relax,
            verbose = method.opts.verbose,
            c_init = c_init,
        )

    # --- Processing ---
    ee = sol.resid
    ee_vec = ee isa AbstractMatrix ? vec(maximum(ee, dims = 2)) : ee # vector of max errors per asset grid point
    ee_mat = ee isa AbstractMatrix ? ee : nothing
    ee_mean = ee_mat === nothing ? mean_abs_error(ee_vec) : mean_abs_error(ee_mat)
    delta_pol = hasproperty(sol, :delta_pol) ? sol.delta_pol : missing
    policy = Dict{Symbol,Any}(
        :c => (;
            value = sol.c,
            grid = g[:a].grid,
            euler_errors = ee_vec,
            euler_errors_mat = ee_mat,
        ),
        :a => (; value = sol.a_next, grid = g[:a].grid),
    )
    value = compute_value_policy(p, g, S, U, policy)
    metadata = Dict{Symbol,Any}(
        :iters => sol.iters,
        :max_it => sol.opts.maxit,
        :converged => sol.converged,
        :max_resid => sol.max_resid, # kept for backward-compat; equals rmse when resid_metric=:rmse
        :rmse => hasproperty(sol, :rmse) ? getfield(sol, :rmse) : sol.max_resid,
        :tol => sol.opts.tol,
        :tol_pol => sol.opts.tol_pol,
        :relax => sol.opts.relax,
        :resid_metric =>
            hasproperty(sol.opts, :resid_metric) ? sol.opts.resid_metric : :rmse,
        :verbose => sol.opts.verbose,
        :interp_kind => string(sol.opts.interp_kind),
        :julia_version => string(VERSION),
        :delta_pol => delta_pol,
        :mean_ee => ee_mean,
    )

    # Validation: monotonicity and positivity

    c_val = policy[:c].value
    a_val = policy[:a].value
    amin = g[:a].min

    # Initialize all expected validation flags to true so that test-suite
    # code which monkeypatches validators still finds the keys in
    # `metadata[:validation]` even when validators are overwritten.
    validate_policy!(
        metadata,
        policy,
        amin;
        method_name = "EGM",
        verbose = method.opts.verbose,
    )

    # Model ID
    model_id = hash_hex(canonicalize_cfg(cfg))

    diagnostics = (;
        model_id = model_id,
        method = method.opts.name,
        seed = sol.opts.seed,
        runtime = sol.opts.runtime,
        iterations = sol.iters,
        mean_ee = ee_mean,
        delta_pol = delta_pol,
    )

    return Solution(
        policy = policy,
        value = value,
        diagnostics = diagnostics,
        metadata = metadata,
        model = model,
        method = method,
    )
end

end # module
