"""
TimeIteration method adapter

Builds a simple method object and exposes `solve` that mirrors the EGM adapter's
behavior so the rest of the codebase (tests, plotting) can use it similarly.
"""
module TimeIteration

using ..API
import ..API: solve

using ..TimeIterationKernel: solve_ti_det, solve_ti_stoch
using ..ValueFunction: compute_value_policy
using ..Determinism: canonicalize_cfg, hash_hex
using ..CommonInterp: LinearInterp, MonotoneCubicInterp
using ..CommonValidators: is_nondec, is_positive, respects_amin
using ..UtilsConfig: maybe

export TimeIterationMethod, build_timeiteration_method

struct TimeIterationMethod <: AbstractMethod
    opts::NamedTuple
end

function build_timeiteration_method(cfg::NamedTuple)
    solver_cfg = cfg.solver
    ik = maybe(solver_cfg, :interp_kind, :linear)
    ik = ik isa Symbol ? ik : Symbol(ik)
    return TimeIterationMethod((
        name = maybe(cfg, :method, solver_cfg.method),
        tol = maybe(solver_cfg, :tol, 1e-6),
        tol_pol = maybe(solver_cfg, :tol_pol, 1e-6),
        maxit = maybe(solver_cfg, :maxit, 1000),
        interp_kind = ik,
        verbose = maybe(solver_cfg, :verbose, false),
        warm_start = maybe(solver_cfg, :warm_start, :default),
    ))
end

function solve(
    model::AbstractModel,
    method::TimeIterationMethod,
    cfg::NamedTuple;
    rng = nothing,
)::Solution
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    init_cfg = maybe(cfg, :init)
    custom_c_data = maybe(init_cfg, :c)
    custom_c_vec = custom_c_data isa AbstractVector ? custom_c_data : nothing
    custom_c_mat = custom_c_data isa AbstractMatrix ? custom_c_data : nothing

    function _build_c_init_det()
        a_grid = g[:a].grid
        a_min = g[:a].min
        R = 1 + p.r
        ws = Symbol(lowercase(string(method.opts.warm_start)))
        if ws == :steady_state
            c = @. p.y + R * a_grid - a_grid
            cmin = 1e-12
            cmax = @. p.y + R * a_grid - a_min
            return clamp.(c, cmin, cmax)
        elseif ws in (:default, :half_resources, :none)
            return nothing
        else
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

    ik = method.opts.interp_kind
    interp = ik == :linear ? LinearInterp() : MonotoneCubicInterp()

    sol =
        S === nothing ?
        solve_ti_det(
            p,
            g,
            U;
            tol = method.opts.tol,
            tol_pol = method.opts.tol_pol,
            maxit = method.opts.maxit,
            interp_kind = interp,
            c_init = c_init,
        ) :
        solve_ti_stoch(
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

    ee = sol.resid
    ee_vec = ee isa AbstractMatrix ? vec(maximum(ee, dims = 2)) : ee
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
    model_id = hash_hex(canonicalize_cfg(cfg))

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

    # Validation
    c_val = policy[:c].value
    a_val = policy[:a].value
    amin = g[:a].min

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
            @warn "TimeIteration solution failed monotonicity/positivity checks; marking as invalid." violations
        else
            @info "TimeIteration solution failed validation; set solver.verbose=true for details."
        end
    end

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
