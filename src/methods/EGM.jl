module EGM

using ..API
using ..ModelContract
using ..EGMKernel:solve_egm_det
using ..ValueFunction: compute_value
using ..Determinism: canonicalize_cfg, hash_hex

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
    # --- Extraction ---
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    # --- Solution ---
    # Check if S has an active key, if yes and active is true, then dispatch to the stoch solver
    if haskey(cfg, :active) && cfg[:active]
        #sol = solve_egm_stoch(P, g, S, U)
    else
        sol = solve_egm_det(p, g, U)
    end

    # --- Processing ---
    policy = (;c_pol = sol.c, a_pol = sol.a_next, a_grid=g.a_grid)
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
        :ν => sol.opts.ν,
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