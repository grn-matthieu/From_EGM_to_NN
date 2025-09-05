module EGM

using ..API
import ..API: build_method, solve

using ..EGMKernel:solve_egm_det, solve_egm_stoch
using ..ValueFunction: compute_value
using ..Determinism: canonicalize_cfg, hash_hex

export EGMMethod

struct EGMMethod <: AbstractMethod
    opts::NamedTuple
end


"""
    build_method(cfg::AbstractDict) -> EGMMethod

Accepts either cfg[:method] or cfg[:solver] and returns an EGMMethod.
"""
function build_method(cfg::AbstractDict)
    method_name = haskey(cfg, :method) ? cfg[:method] : cfg[:solver][:method]
    if method_name != "EGM"
        error("Method builder received cfg with method = $method_name. It has not been implemented.")
    else
        return EGMMethod((
        name = method_name,
        tol = get(cfg[:solver], :tol, 1e-6),
        maxit = get(cfg[:solver], :maxit, 1000),
        interp_kind = get(cfg[:solver], :interp_kind, :linear),
        verbose = get(cfg[:solver], :verbose, false)
    ))
    end
end



"""
    solve(model::AbstractModel, method::EGMMethod, cfg::AbstractDict; rng=nothing)::Solution

Entry point for the EGM solver. Extracts contract fields, runs a minimal EGM loop, and returns a Solution.
"""
function solve(model::AbstractModel, method::EGMMethod, cfg::AbstractDict; rng=nothing)::Solution
    # --- Extraction ---
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    # --- Solution ---
    # The dispatch is made when a shock structure is provided in the config file
    sol = (S === nothing) ? solve_egm_det(p, g, U) : solve_egm_stoch(p, g, S, U)

    # --- Processing ---
    ee = sol.resid
    ee_vec = ee isa AbstractMatrix ? vec(maximum(ee, dims=2)) : ee # make ee a vector of max errors per asset grid point
    policy = Dict{Symbol,Any}(
        :c => (; value = sol.c, grid = g[:a].grid, euler_errors = ee_vec),
        :a => (; value = sol.a_next, grid = g[:a].grid)
    )
    value = compute_value(p, g, S, U, policy)
    metadata = Dict{Symbol,Any}(
        :iters => sol.iters,
        :max_it => sol.opts.maxit,
        :converged => sol.converged,
        :max_resid => sol.max_resid,
        :tol => sol.opts.tol,
        :tol_pol => sol.opts.tol_pol,
        :relax => sol.opts.relax,
        :patience => sol.opts.patience,
        :ϵ => sol.opts.ϵ,
        :interp_kind => string(sol.opts.interp_kind),
        :julia_version => string(VERSION)
    )

    # Model ID
    model_id = hash_hex(canonicalize_cfg(cfg))

    diagnostics = (; 
        model_id = model_id, 
        method = method.opts.name, 
        seed = sol.opts.seed,
        runtime = sol.opts.runtime
    )

    return Solution(policy, value, diagnostics, metadata, model, method)
end

end #
