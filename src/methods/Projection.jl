module Projection
using ..API
import ..API: solve
using ..ProjectionKernel: solve_projection_det, solve_projection_stoch
using ..ValueFunction: compute_value_policy
using ..Determinism: canonicalize_cfg, hash_hex
export ProjectionMethod

struct ProjectionMethod <: AbstractMethod
    opts::NamedTuple
end
"""
    build_projection_method(cfg::AbstractDict) -> ProjectionMethod
Construct a `ProjectionMethod` using solver options contained in `cfg`.
"""
function build_projection_method(cfg::AbstractDict)
    return ProjectionMethod((
        name = haskey(cfg, :method) ? cfg[:method] : cfg[:solver][:method],
        tol = get(cfg[:solver], :tol, 1e-6),
        maxit = get(cfg[:solver], :maxit, 1000),
        verbose = get(cfg[:solver], :verbose, false),
        orders = get(cfg[:solver], :orders, [cfg[:grids][:Na] - 1]),
        Nval = get(cfg[:solver], :Nval, cfg[:grids][:Na]),
    ))
end
function solve(
    model::AbstractModel,
    method::ProjectionMethod,
    cfg::AbstractDict;
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
