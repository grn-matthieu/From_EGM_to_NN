module Projection
using ..API
import ..API: solve
using ..ProjectionKernel: solve_projection_det
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

    if S !== nothing
        error("Projection solver currently supports only deterministic models")
    end

    sol = solve_projection_det(p, g, U; tol = method.opts.tol, maxit = method.opts.maxit)

    policy = Dict{Symbol,Any}(
        :c => (;
            value = sol.c,
            grid = g[:a].grid,
            euler_errors = sol.resid,
            euler_errors_mat = nothing,
        ),
        :a => (; value = sol.a_next, grid = g[:a].grid),
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
        :julia_version => string(VERSION),
    )

    return Solution(policy, value, diagnostics, metadata, model, method)
end
end # module
