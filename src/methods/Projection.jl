"""
Projection

Adapter exposing projection-based solver kernel through `API.solve`.
"""
module Projection
using ..API
import ..API: solve
using ..ProjectionKernel: solve_projection_det, solve_projection_stoch
using ..ValueFunction: compute_value_policy
using ..Determinism: canonicalize_cfg, hash_hex
using ..UtilsConfig: maybe
using ..UtilsDiagnostics: mean_abs_error
export ProjectionMethod

struct ProjectionMethod <: AbstractMethod
    opts::NamedTuple
end
"""
    build_projection_method(cfg::NamedTuple) -> ProjectionMethod

Construct a `ProjectionMethod` using solver options contained in the NamedTuple `cfg`.
"""
function build_projection_method(cfg::NamedTuple)
    solver_cfg = cfg.solver
    projection_cfg = solver_cfg.projection
    integration_raw = maybe(projection_cfg, :integration, :gh)
    integration_sym = Symbol(lowercase(string(integration_raw)))
    # Optional integration tuning for CSVAR
    gh_order = maybe(projection_cfg, :gh_order, 3)
    nsamples = maybe(projection_cfg, :nsamples, 128)
    return ProjectionMethod((
        name = solver_cfg.method,
        tol = solver_cfg.tol,
        tol_pol = solver_cfg.tol_pol,
        maxit = solver_cfg.maxit,
        verbose = solver_cfg.verbose,
        orders = projection_cfg.orders,
        Nval = projection_cfg.Nval,
        integration = integration_sym,
        gh_order = gh_order,
        nsamples = nsamples,
    ))
end
function solve(
    model::AbstractModel,
    method::ProjectionMethod,
    cfg::NamedTuple;
    rng = nothing,
)::Solution
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    sol =
        S === nothing ?
        solve_projection_det(
            p,
            g,
            U;
            tol = method.opts.tol,
            tol_pol = method.opts.tol_pol,
            maxit = method.opts.maxit,
            orders = method.opts.orders,
            Nval = method.opts.Nval,
            rng = rng,
        ) :
        solve_projection_stoch(
            p,
            g,
            S,
            U;
            tol = method.opts.tol,
            tol_pol = method.opts.tol_pol,
            maxit = method.opts.maxit,
            orders = method.opts.orders,
            Nval = method.opts.Nval,
            integration_method = method.opts.integration,
            gh_order = get(method.opts, :gh_order, 3),
            nsamples = get(method.opts, :nsamples, 128),
            rng = rng,
        )

    ee = sol.resid
    ee_vec = ee isa AbstractMatrix ? vec(maximum(ee, dims = 2)) : ee
    ee_mat = ee isa AbstractMatrix ? ee : nothing
    ee_mean = ee_mat === nothing ? mean_abs_error(ee_vec) : mean_abs_error(ee_mat)
    delta_pol = hasproperty(sol, :delta_pol) ? sol.delta_pol : missing

    policy = Dict{Symbol,Any}(
        :c => (;
            value = sol.c,
            grid = sol.a_grid,
            euler_errors = ee_vec,
            euler_errors_mat = ee_mat,
        ),
        :a => (; value = sol.a_next, grid = sol.a_grid),
    )

    value = compute_value_policy(p, g, S, U, policy)

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

    metadata = Dict{Symbol,Any}(
        :iters => sol.iters,
        :max_it => get(sol.opts, :maxit, missing),
        :converged => sol.converged,
        :max_resid => sol.euler_rmse,
        :tol => get(sol.opts, :tol, missing),
        :order => get(sol.opts, :order, missing),
        :tol_pol => hasproperty(sol.opts, :tol_pol) ? sol.opts.tol_pol : missing,
        :delta_pol => delta_pol,
        :mean_ee => ee_mean,
        :julia_version => string(VERSION),
        :integration => get(method.opts, :integration, nothing),
        :rmse_history => hasproperty(sol, :rmse_history) ? sol.rmse_history : Float64[],
        :opts => sol.opts,
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
