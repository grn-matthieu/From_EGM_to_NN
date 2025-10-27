"""
TimeIterationKernel

Simple time-iteration (policy-update) solver kernel for the ConsumerSaving model.
This implements a fixed-point iteration on the Euler equation using grid interpolation
for next-period consumption. It mirrors the EGM solver's API so it can be used by the
existing `methods` adapter.
"""
module TimeIterationKernel

using ..CommonInterp:
    interp_linear,
    interp_linear!,
    interp_pchip!,
    InterpKind,
    LinearInterp,
    MonotoneCubicInterp
using ..EulerResiduals: euler_resid_det!, euler_resid_stoch!, euler_resid_stoch_interp!
using ..PolicyUtils:
    clamp_policy!,
    compute_binding_tolerance,
    enforce_monotone!,
    ensure_minimum!,
    init_consumption_det,
    relaxation_step!,
    rmse_nonbinding
using ..SolverPlaceholders: build_placeholder_solution
using ..GridHelpers: fit_values_on_backend!, eval_backend_at_points, grid_backend_available
using Printf

export solve_ti_det, solve_ti_stoch, solve_ti_placeholder

@inline function _is_csvar_model(params)
    hasproperty(params, :y_dim) && getproperty(params, :y_dim) > 1
end

"""
    solve_consumption_root(euler_gap, c_lo, c_hi, root_tol, max_root_iter)

Helper to bracket and solve the Euler equation for consumption via bisection. It
handles boundary cases where the bracket endpoints already satisfy the Euler
equation, returning the appropriate bound if so.
"""
function solve_consumption_root(euler_gap, c_lo, c_hi, root_tol, max_root_iter)
    f_hi = euler_gap(c_hi)
    if f_hi >= 0
        return c_hi
    end

    f_lo = euler_gap(c_lo)
    if f_lo <= 0
        return c_lo
    end

    lo = c_lo
    hi = c_hi
    mid = 0.5 * (lo + hi)
    for _ = 1:max_root_iter
        mid = 0.5 * (lo + hi)
        f_mid = euler_gap(mid)
        if abs(f_mid) < root_tol || 0.5 * (hi - lo) < root_tol
            break
        elseif f_mid > 0
            lo = mid
        else
            hi = mid
        end
    end

    return mid
end

function solve_ti_det(
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 500,
    interp_kind::InterpKind = LinearInterp(),
    relax::Real = 0.5,
    ϵ::Real = 1e-10,
    c_init = nothing,
    verbose::Bool = false,
)::NamedTuple
    return solve_ti_det_impl(
        interp_kind,
        model_params,
        model_grids,
        model_utility;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        relax = relax,
        ϵ = ϵ,
        c_init = c_init,
        verbose = verbose,
    )
end

solve_ti_det_impl(::InterpKind, args...; kwargs...) =
    error("Unknown interp kind for TimeIteration (det)")

function solve_ti_det_impl(
    interp_kind::LinearInterp,
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 500,
    relax::Real = 0.5,
    ϵ::Real = 1e-10,
    c_init = nothing,
    verbose::Bool = false,
)
    if _is_csvar_model(model_params)
        return solve_ti_placeholder(
            model_params,
            model_grids,
            nothing,
            model_utility;
            tol = tol,
            tol_pol = tol_pol,
            maxit = maxit,
            interp_kind = interp_kind,
            relax = relax,
            verbose = verbose,
            c_init = c_init,
            ϵ = ϵ,
            note = "Time-iteration is disabled for CSVAR: deterministic EGM already fails when the VAR income dimension exceeds 1, so no TI baseline is tested.",
        )
    end

    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N

    R = 1 + model_params.r
    β = model_params.β
    cmin = 1e-12
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = 1e-12, rel = 0.0)

    c = init_consumption_det(a_grid, a_min, R, model_params.y; c_init = c_init, cmin = cmin)

    cnext = similar(c)
    cnew = similar(c)
    a_next = similar(c)
    resid = similar(c)
    cold = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    Δpol = Inf

    root_tol = min(1e-10, tol)
    max_root_iter = 100

    for it = 1:maxit
        iters = it

        copyto!(cold, c)

        # prepare backend if available (fit once per outer iteration)
        a_info = model_grids[:a]
        use_backend = grid_backend_available(a_info)
        backend_cold = nothing
        if use_backend
            backend_cold = fit_values_on_backend!(a_info, a_grid, reshape(cold, :, 1))
        end

        @inbounds for (i, a) in enumerate(a_grid)
            resources = model_params.y + R * a
            c_hi = max(cmin, resources - a_min)

            if c_hi <= cmin
                cnew[i] = cmin
                continue
            end

            function euler_gap(c_guess)
                a_prime = clamp(resources - c_guess, a_min, a_max)
                if use_backend
                    c_future = eval_backend_at_points(backend_cold, (a_prime,), 1)[1]
                else
                    c_future = interp_linear(a_grid, cold, a_prime)
                end
                c_future = c_future < cmin ? cmin : c_future
                return model_utility.u_prime(c_guess) -
                       β * R * model_utility.u_prime(c_future)
            end

            cnew[i] = solve_consumption_root(euler_gap, cmin, c_hi, root_tol, max_root_iter)
        end

        cmax = @. model_params.y + R * a_grid - a_min
        clamp_policy!(cnew, cmin, cmax)

        Δpol = relaxation_step!(c, cold, cnew, relax)

        @. a_next = clamp(model_params.y + R * a_grid - c, a_min, a_max)
        if use_backend
            # fit current policy c on backend and evaluate at a_next
            backend_c = fit_values_on_backend!(a_info, a_grid, reshape(c, :, 1))
            cnext .= eval_backend_at_points(backend_c, a_next, 1)
        else
            interp_linear!(cnext, a_grid, c, a_next)
        end
        ensure_minimum!(cnext, cmin)

        euler_resid_det!(resid, model_params, c, cnext)
        max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)

        if verbose && (it % 10 == 0)
            @printf("[TimeIteration] it=%d rmse=%.6e Δpol=%.6e\n", it, max_resid, Δpol)
            flush(stdout)
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end

        if best_resid - max_resid < ϵ && Δpol < ϵ
            # small improvements only; continue iterating without a patience cutoff
            # keep best_resid for diagnostics
        else
            best_resid = max_resid
        end
    end

    # final consistency
    @. a_next = clamp(R * a_grid + model_params.y - c, a_min, a_max)
    if grid_backend_available(model_grids[:a])
        a_info = model_grids[:a]
        backend_c = fit_values_on_backend!(a_info, a_grid, reshape(c, :, 1))
        cnext .= eval_backend_at_points(backend_c, a_next, 1)
    else
        interp_linear!(cnext, a_grid, c, a_next)
    end
    ensure_minimum!(cnext, cmin)
    euler_resid_det!(resid, model_params, c, cnext)
    max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        interp_kind = interp_kind,
        relax = relax,
        ϵ = ϵ,
        resid_metric = :rmse,
        seed = nothing,
        runtime = runtime,
    )

    return (;
        a_grid,
        c,
        a_next,
        resid,
        iters,
        converged,
        max_resid,
        rmse = max_resid,
        model_params,
        opts,
        delta_pol = Δpol,
    )
end



function solve_ti_placeholder(
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = NaN,
    tol_pol::Real = NaN,
    maxit::Int = 0,
    interp_kind::InterpKind = LinearInterp(),
    relax::Real = 0.0,
    verbose::Bool = false,
    c_init = nothing,
    ϵ::Real = 0.0,
    note::AbstractString = "placeholder TimeIteration solution",
)
    opts = (;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        interp_kind = interp_kind,
        relax = relax,
        verbose = verbose,
        resid_metric = :placeholder,
        c_init = c_init,
        ϵ = ϵ,
    )
    return build_placeholder_solution(
        :TimeIteration,
        model_params,
        model_grids,
        model_shocks;
        opts = opts,
        note = note,
    )
end

# Monotone cubic variant (PCHIP)
function solve_ti_det_impl(
    interp_kind::MonotoneCubicInterp,
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 500,
    relax::Real = 0.5,
    ϵ::Real = 1e-10,
    c_init = nothing,
    verbose::Bool = false,
)
    if _is_csvar_model(model_params)
        return solve_ti_placeholder(
            model_params,
            model_grids,
            nothing,
            model_utility;
            tol = tol,
            tol_pol = tol_pol,
            maxit = maxit,
            interp_kind = interp_kind,
            relax = relax,
            verbose = verbose,
            c_init = c_init,
            ϵ = ϵ,
            note = "Time-iteration is disabled for CSVAR: deterministic EGM already fails when the VAR income dimension exceeds 1, so no TI baseline is tested.",
        )
    end

    # For brevity reuse linear implementation but substitute cubic interp where used
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N

    R = 1 + model_params.r
    β = model_params.β
    cmin = 1e-12
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = 1e-12, rel = 0.0)

    c = init_consumption_det(a_grid, a_min, R, model_params.y; c_init = c_init, cmin = cmin)

    cnext = similar(c)
    cnew = similar(c)
    a_next = similar(c)
    resid = similar(c)
    cold = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    Δpol = Inf

    root_tol = min(1e-10, tol)
    max_root_iter = 100
    interp_buf = similar(c, (1,))
    query_buf = similar(a_grid, (1,))

    for it = 1:maxit
        iters = it

        copyto!(cold, c)

        # prepare backend if available (fit once per outer iteration)
        a_info = model_grids[:a]
        use_backend = grid_backend_available(a_info)
        backend_cold = nothing
        if use_backend
            # fit all shock-columns at once
            backend_cold = fit_values_on_backend!(a_info, a_grid, cold)
        end

        @inbounds for (i, a) in enumerate(a_grid)
            resources = model_params.y + R * a
            c_hi = max(cmin, resources - a_min)

            if c_hi <= cmin
                cnew[i] = cmin
                continue
            end

            function future_consumption(a_prime)
                if use_backend
                    val = eval_backend_at_points(backend_cold, (a_prime,), 1)[1]
                else
                    query_buf[1] = a_prime
                    interp_pchip!(interp_buf, a_grid, cold, query_buf)
                    val = interp_buf[1]
                end
                return val < cmin ? cmin : val
            end

            function euler_gap(c_guess)
                a_prime = clamp(resources - c_guess, a_min, a_max)
                c_future = future_consumption(a_prime)
                return model_utility.u_prime(c_guess) -
                       β * R * model_utility.u_prime(c_future)
            end

            cnew[i] = solve_consumption_root(euler_gap, cmin, c_hi, root_tol, max_root_iter)
        end

        cmax = @. model_params.y + R * a_grid - a_min
        clamp_policy!(cnew, cmin, cmax)
        enforce_monotone!(cnew)

        Δpol = relaxation_step!(c, cold, cnew, relax)

        @. a_next = clamp(model_params.y + R * a_grid - c, a_min, a_max)
        if use_backend
            backend_c = fit_values_on_backend!(a_info, a_grid, reshape(c, :, 1))
            cnext .= eval_backend_at_points(backend_c, a_next, 1)
        else
            interp_pchip!(cnext, a_grid, c, a_next)
        end
        ensure_minimum!(cnext, cmin)

        euler_resid_det!(resid, model_params, c, cnext)
        max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)

        if verbose && (it % 10 == 0)
            @printf(
                "[TimeIteration:PCHIP] it=%d rmse=%.6e Δpol=%.6e\n",
                it,
                max_resid,
                Δpol
            )
            flush(stdout)
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end

        if best_resid - max_resid < ϵ && Δpol < ϵ
            # small improvements only; continue iterating without a patience cutoff
            # keep best_resid for diagnostics
        else
            best_resid = max_resid
        end
    end

    @. a_next = clamp(R * a_grid + model_params.y - c, a_min, a_max)
    interp_pchip!(cnext, a_grid, c, a_next)
    ensure_minimum!(cnext, cmin)
    euler_resid_det!(resid, model_params, c, cnext)
    max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        interp_kind = interp_kind,
        relax = relax,
        ϵ = ϵ,
        resid_metric = :rmse,
        seed = nothing,
        runtime = runtime,
    )

    return (;
        a_grid,
        c,
        a_next,
        resid,
        iters,
        converged,
        max_resid,
        rmse = max_resid,
        model_params,
        opts,
        delta_pol = Δpol,
    )
end


# --- Stochastic variant ---
function solve_ti_stoch(
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 1000,
    interp_kind::InterpKind = LinearInterp(),
    relax::Real = 0.5,
    ϵ::Real = 1e-10,
    c_init = nothing,
    verbose::Bool = false,
)
    return solve_ti_stoch_impl(
        interp_kind,
        model_params,
        model_grids,
        model_shocks,
        model_utility;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        relax = relax,
        ϵ = ϵ,
        c_init = c_init,
        verbose = verbose,
    )
end

solve_ti_stoch_impl(::InterpKind, args...; kwargs...) =
    error("Unknown interp kind for TimeIteration (stoch)")

function solve_ti_stoch_impl(
    interp_kind::LinearInterp,
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 1000,
    relax::Real = 0.5,
    ϵ::Real = 1e-10,
    c_init = nothing,
    verbose::Bool = false,
)
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N

    z_grid = model_shocks.zgrid
    Π = model_shocks.Π
    Nz = length(z_grid)

    β = model_params.β
    R = 1 + model_params.r
    cmin = 1e-12

    c = c_init === nothing ? fill(1.0, Na, Nz) : copy(c_init)
    a_next = similar(c)
    cnew = similar(c)
    resid_mat = similar(c)
    cmax = similar(a_grid)
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = 1e-12, rel = 0.0)
    cold = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    Δpol = Inf

    root_tol = min(1e-10, tol)
    max_root_iter = 100

    for it = 1:maxit
        iters = it
        copyto!(cold, c)

        for (j, z) in enumerate(z_grid)
            y = exp(z)
            column_new = view(cnew, :, j)
            @. cmax = y + R * a_grid - a_min

            @inbounds for (i, a) in enumerate(a_grid)
                resources = R * a + y
                c_hi = max(cmin, resources - a_min)

                if c_hi <= cmin
                    column_new[i] = cmin
                    continue
                end

                function euler_gap(c_guess)
                    a_prime = clamp(resources - c_guess, a_min, a_max)
                    Emu = zero(c_guess)
                    @inbounds for (jp, _) in enumerate(z_grid)
                        c_future = interp_linear(a_grid, view(cold, :, jp), a_prime)
                        c_future = c_future < cmin ? cmin : c_future
                        Emu += Π[j, jp] * model_utility.u_prime(c_future)
                    end
                    return model_utility.u_prime(c_guess) - β * R * Emu
                end

                column_new[i] =
                    solve_consumption_root(euler_gap, cmin, c_hi, root_tol, max_root_iter)
            end

            clamp_policy!(column_new, cmin, cmax)
        end

        Δpol = relaxation_step!(c, cold, cnew, relax)

        for (j, z) in enumerate(z_grid)
            y = exp(z)
            @views @. a_next[:, j] = clamp(R * a_grid + y - c[:, j], a_min, a_max)
        end

        if verbose && (it % 10 == 0)
            @printf(
                "[TimeIteration:STOCH] it=%d rmse=%.6e Δpol=%.6e\n",
                it,
                max_resid,
                Δpol
            )
            flush(stdout)
        end

        euler_resid_stoch_interp!(
            resid_mat,
            model_params,
            a_grid,
            z_grid,
            Π,
            c,
            interp_kind,
        )
        # resid_mat is Na x Nz. Build mask of non-binding entries where a_next > a_min
        max_resid = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end

        if best_resid - max_resid < ϵ
            # small change; continue without a patience cutoff
        else
            best_resid = max_resid
        end
    end

    euler_resid_stoch_interp!(resid_mat, model_params, a_grid, z_grid, Π, c, interp_kind)
    max_resid = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        interp_kind = interp_kind,
        relax = relax,
        ϵ = ϵ,
        resid_metric = :rmse,
        seed = nothing,
        runtime = runtime,
    )

    return (;
        a_grid,
        z_grid,
        c,
        a_next,
        resid = resid_mat,
        iters,
        converged,
        max_resid,
        rmse = max_resid,
        model_params,
        opts,
        delta_pol = Δpol,
    )
end

function solve_ti_stoch_impl(::MonotoneCubicInterp, args...; kwargs...)
    # For simplicity delegate to linear implementation: cubic interp not implemented for stoch here
    return solve_ti_stoch_impl(LinearInterp(), args...; kwargs...)
end

end # module
