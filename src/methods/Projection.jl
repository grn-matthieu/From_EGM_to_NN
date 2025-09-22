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
    grids_cfg = cfg.grids
    default_orders = [grids_cfg.Na - 1]
    return ProjectionMethod((
        name = maybe(cfg, :method, solver_cfg.method),
        tol = maybe(solver_cfg, :tol, 1e-6),
        maxit = maybe(solver_cfg, :maxit, 1000),
        verbose = maybe(solver_cfg, :verbose, false),
        orders = maybe(solver_cfg, :orders, default_orders),
        Nval = maybe(solver_cfg, :Nval, grids_cfg.Na),
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
            maxit = method.opts.maxit,
            orders = method.opts.orders,
            Nval = method.opts.Nval,
        ) :
        solve_projection_stoch(
            p,
            g,
            S,
            U;
            tol = method.opts.tol,
            maxit = method.opts.maxit,
            orders = method.opts.orders,
            Nval = method.opts.Nval,
        )

    ee = sol.resid
    ee_vec = ee isa AbstractMatrix ? vec(maximum(ee, dims = 2)) : ee
    ee_mat = ee isa AbstractMatrix ? ee : nothing

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
    )

    metadata = Dict{Symbol,Any}(
        :iters => sol.iters,
        :max_it => sol.opts.maxit,
        :converged => sol.converged,
        :max_resid => sol.max_resid,
        :tol => sol.opts.tol,
        :order => sol.opts.order,
        :julia_version => string(VERSION),
    )

    return Solution(policy, value, diagnostics, metadata, model, method)
end
end # module
