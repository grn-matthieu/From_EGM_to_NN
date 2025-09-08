module EGM

using ..API
import ..API: build_method, solve

using ..EGMKernel: solve_egm_det, solve_egm_stoch
using ..ValueFunction: compute_value_policy
using ..Determinism: canonicalize_cfg, hash_hex
using ..CommonInterp: LinearInterp, MonotoneCubicInterp

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
        ik = get(cfg[:solver], :interp_kind, :linear)
        ik = ik isa Symbol ? ik : Symbol(ik)
        return EGMMethod((
            name = method_name,
            tol = get(cfg[:solver], :tol, 1e-6),
            tol_pol = get(cfg[:solver], :tol_pol, 1e-6),
            maxit = get(cfg[:solver], :maxit, 1000),
            interp_kind = ik,
            verbose = get(cfg[:solver], :verbose, false),
            warm_start = get(cfg[:solver], :warm_start, :default)
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

    # --- Warm-start policy initialization ---
    function _build_c_init_det()
        a_grid = g[:a].grid
        a_min  = g[:a].min
        R = 1 + p.r
        # Default: let kernel choose half resources
        ws = Symbol(lowercase(string(method.opts.warm_start)))
        if ws == :steady_state
            # Keep assets constant: a' = a => c = y + R*a - a
            c = @. p.y + R * a_grid - a_grid
            cmin = 1e-12
            cmax = @. p.y + R * a_grid - a_min
            return clamp.(c, cmin, cmax)
        elseif ws in (:default, :half_resources, :none)
            return nothing
        else
            # Optional custom: cfg.init.c if provided
            if haskey(cfg, :init) && haskey(cfg[:init], :c)
                return copy(cfg[:init][:c])
            end
            return nothing
        end
    end

    function _build_c_init_stoch()
        a_grid = g[:a].grid
        a_min  = g[:a].min
        R = 1 + p.r
        z_grid = S.zgrid
        Nz = length(z_grid)
        ws = Symbol(lowercase(string(method.opts.warm_start)))
        if ws == :steady_state
            Na = length(a_grid)
            c = Array{Float64}(undef, Na, Nz)
            @inbounds for j in 1:Nz
                y = exp(z_grid[j])
                ccol = @. y + R * a_grid - a_grid
                cmin = 1e-12
                cmax = @. y + R * a_grid - a_min
                @. ccol = clamp(ccol, cmin, cmax)
                @views c[:, j] .= ccol
            end
            return c
        elseif ws in (:default, :half_resources, :none)
            return nothing
        else
            if haskey(cfg, :init) && haskey(cfg[:init], :c)
                return copy(cfg[:init][:c])
            end
            return nothing
        end
    end

    c_init = S === nothing ? _build_c_init_det() : _build_c_init_stoch()

    # --- Solution ---
    ik = method.opts.interp_kind
    interp = ik == :linear ? LinearInterp() : MonotoneCubicInterp()
    sol = S === nothing ?
        solve_egm_det(p, g, U;
            tol=method.opts.tol,
            tol_pol=method.opts.tol_pol,
            maxit=method.opts.maxit,
            interp_kind=interp,
            c_init=c_init) :
        solve_egm_stoch(p, g, S, U;
            tol=method.opts.tol,
            tol_pol=method.opts.tol_pol,
            maxit=method.opts.maxit,
            interp_kind=interp,
            c_init=c_init)

    # --- Processing ---
    ee = sol.resid
    ee_vec = ee isa AbstractMatrix ? vec(maximum(ee, dims=2)) : ee # vector of max errors per asset grid point
    ee_mat = ee isa AbstractMatrix ? ee : nothing
    policy = Dict{Symbol,Any}(
        :c => (; value = sol.c, grid = g[:a].grid, euler_errors = ee_vec, euler_errors_mat = ee_mat),
        :a => (; value = sol.a_next, grid = g[:a].grid)
    )
    value = compute_value_policy(p, g, S, U, policy)
    metadata = Dict{Symbol,Any}(
        :iters => sol.iters,
        :max_it => sol.opts.maxit,
        :converged => sol.converged,
        :max_resid => sol.max_resid,
        :tol => sol.opts.tol,
        :tol_pol => sol.opts.tol_pol,
        :relax => sol.opts.relax,
        :patience => sol.opts.patience,
        :interp_kind => string(sol.opts.interp_kind),
        :julia_version => string(VERSION)
    )

    # Validation: monotonicity and positivity
    is_nondec(x; tol=1e-12) = x isa AbstractMatrix ? all(j -> all(diff(view(x, :, j)) .>= -tol), axes(x, 2)) : all(diff(x) .>= -tol)
    is_positive(x; tol=1e-12) = all(x .>= tol)
    respects_amin(x, amin; tol=1e-12) = all(x .>= (amin - tol))

    c_val = policy[:c].value
    a_val = policy[:a].value
    amin = g[:a].min

    violations = Dict{Symbol,Any}()
    valid = true
    if !is_positive(c_val)
        violations[:c_positive] = false
        valid = false
    end
    if !respects_amin(a_val, amin)
        violations[:a_above_min] = false
        valid = false
    end
    if !is_nondec(c_val)
        violations[:c_monotone_nondec] = false
        valid = false
    end
    if !is_nondec(a_val)
        violations[:a_monotone_nondec] = false
        valid = false
    end
    metadata[:valid] = valid
    if !valid
        metadata[:validation] = violations
        @warn "EGM solution failed monotonicity/positivity checks; marking as invalid." violations
    end

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

end # module

