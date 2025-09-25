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
using ..CommonValidators: is_nondec, is_positive, respects_amin
using ..UtilsConfig: maybe

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
    ik = maybe(solver_cfg, :interp_kind, :linear)
    ik = ik isa Symbol ? ik : Symbol(ik)
    return EGMMethod((
        name = maybe(cfg, :method, solver_cfg.method),
        tol = maybe(solver_cfg, :tol, 1e-6),
        tol_pol = maybe(solver_cfg, :tol_pol, 1e-6),
        maxit = maybe(solver_cfg, :maxit, 1000),
        interp_kind = ik,
        verbose = maybe(solver_cfg, :verbose, false),
        warm_start = maybe(solver_cfg, :warm_start, :default),
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

    function _build_c_init_det()
        a_grid = g[:a].grid
        a_min = g[:a].min
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
            if custom_c_vec !== nothing
                return copy(custom_c_vec)
            end
            return nothing
        end
    end

    function _build_c_init_stoch()
        a_grid = g[:a].grid
        a_min = g[:a].min
        R = 1 + p.r
        z_grid = S.zgrid
        Nz = length(z_grid)
        ws = Symbol(lowercase(string(method.opts.warm_start)))
        if ws == :steady_state
            Na = length(a_grid)
            c = Array{Float64}(undef, Na, Nz)
            @inbounds for j = 1:Nz
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
            if custom_c_mat !== nothing
                return copy(custom_c_mat)
            end
            return nothing
        end
    end

    c_init = S === nothing ? _build_c_init_det() : _build_c_init_stoch()

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
            c_init = c_init,
        )

    # --- Processing ---
    ee = sol.resid
    ee_vec = ee isa AbstractMatrix ? vec(maximum(ee, dims = 2)) : ee # vector of max errors per asset grid point
    ee_mat = ee isa AbstractMatrix ? ee : nothing
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
        :max_resid => sol.max_resid,
        :tol => sol.opts.tol,
        :tol_pol => sol.opts.tol_pol,
        :relax => sol.opts.relax,
        :patience => sol.opts.patience,
        :interp_kind => string(sol.opts.interp_kind),
        :julia_version => string(VERSION),
    )

    # Validation: monotonicity and positivity

    c_val = policy[:c].value
    a_val = policy[:a].value
    amin = g[:a].min

    # Initialize all expected validation flags to true so that test-suite
    # code which monkeypatches validators still finds the keys in
    # `metadata[:validation]` even when validators are overwritten.
    violations = Dict{Symbol,Any}(
        :c_positive => true,
        :a_above_min => true,
        :c_monotone_nondec => true,
        :a_monotone_nondec => true,
    )
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
        if method.opts.verbose
            @warn "EGM solution failed monotonicity/positivity checks; marking as invalid." violations
        else
            @info "EGM solution failed validation; set solver.verbose=true for details."
        end
    end

    # Model ID
    model_id = hash_hex(canonicalize_cfg(cfg))

    diagnostics = (;
        model_id = model_id,
        method = method.opts.name,
        seed = sol.opts.seed,
        runtime = sol.opts.runtime,
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
