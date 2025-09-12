module Projection
using ..API
import ..API: solve
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
    error("Projection solver not implemented")
end
end # module
