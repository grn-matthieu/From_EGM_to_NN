module EGM

using ..API
using ..ModelContract

export EGMMethod, build_method, solve

struct EGMMethod <: AbstractMethod
    opts::NamedTuple
end


"""
    build_method(cfg::AbstractDict) -> EGMMethod

Accepts either cfg[:method] or cfg[:solver] and returns an EGMMethod.
"""
function build_method(cfg::AbstractDict)
    method_name = haskey(cfg, :method) ? cfg[:method] : cfg[:solver][:method]
    return EGMMethod((
        name = method_name,
        tol = get(cfg[:solver], :tol, 1e-6),
        maxit = get(cfg[:solver], :maxit, 1000),
        interp_kind = get(cfg[:solver], :interp_kind, :linear),
        verbose = get(cfg[:solver], :verbose, false)
    ))
end



"""
    solve(model::AbstractModel, method::EGMMethod, cfg::AbstractDict; rng=nothing)::Solution

Entry point for the EGM solver. Extracts contract fields, runs a minimal EGM loop, and returns a Solution.
"""
function solve(model::AbstractModel, method::EGMMethod, cfg::AbstractDict; rng=nothing)::Solution
    # Extract contract fields
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    a = g[:a]
    tol = method.opts[:tol]
    maxit = method.opts[:maxit]

    # Minimal EGM kernel (dummy, to be replaced with real implementation)
    c = a .+ 1.0           # Dummy consumption policy
    ap = a .+ 0.5          # Dummy next-period assets

    # Diagnostics and metadata
    diagnostics = (; iters=1, tol=tol, ee_max=0.0, runtime=0.0)
    
    metadata = Dict{Symbol,Any}(
        :seed => get(get(cfg, :random, Dict()), :seed, nothing), # Check if :random is missing, as well as :seed
        :rng_type => isnothing(rng) ? "nothing" : string(typeof(rng)) # Has the user inputed a seed ?
    )

    # Return Solution
    return Solution((; c, ap), nothing, diagnostics, metadata, model, method)
end

end #