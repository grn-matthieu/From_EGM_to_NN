"""
EGMKernel

Endogenous Grid Method solver kernel for the consumption–savings model.
Exports helpers and the main projection-based policy iteration.
"""
module EGMKernel

using ..CommonInterp:
    interp_linear!, interp_pchip!, InterpKind, LinearInterp, MonotoneCubicInterp
using ..EulerResiduals: euler_resid_det!, euler_resid_stoch!, euler_resid_stoch_interp!
using ..PolicyUtils:
    clamp_policy!,
    compute_binding_tolerance,
    enforce_borrowing_constraint!,
    enforce_monotone!,
    enforce_strict_increase!,
    ensure_minimum!,
    init_consumption_det,
    relaxation_step!,
    rmse_nonbinding,
    sort_policy_pairs!
using ..CSVarUtils: csvar_income
using ..SolverIntegration: integrate_expectation, discrete_expectation
using ..SolverPlaceholders: build_placeholder_solution
using Printf

export solve_egm_det, solve_egm_stoch, solve_egm_placeholder

const DEFAULT_BINDING_TOL = 1e-12

"""
    solve_egm_det(model_params, model_grids, model_utility; ...)

Vectorized EGM solver for deterministic income (equivalent to log-normal income with zero variance).
Stops when Euler equation errors and policy changes fall below their respective tolerances.
Returns a `NamedTuple` with fields `(a_grid, c, a_next, resid, iters, converged, max_resid, model_params, opts)`
which is later converted into a `Solution`.
"""
function solve_egm_det(
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 10_000,
    interp_kind::InterpKind = LinearInterp(),
    relax::Real = 0.5,
    verbose::Bool = false,
    c_init = nothing,
    integration_method::Symbol = :none,
)::NamedTuple
    return solve_egm_det_impl(
        interp_kind,
        model_params,
        model_grids,
        model_utility;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        relax = relax,
        verbose = verbose,
        c_init = c_init,
        integration_method = integration_method,
    )
end

solve_egm_det_impl(::InterpKind, args...; kwargs...) =
    error("Unknown interpolation kind for EGM (deterministic)")

function solve_egm_det_impl(
    ::LinearInterp,
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 10_000,
    relax::Real = 0.5,
    verbose::Bool = false,
    c_init = nothing,
    integration_method::Symbol = :none,
)::NamedTuple
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = DEFAULT_BINDING_TOL)

    β = model_params.β
    R = 1 + model_params.r
    γ = model_params.γ
    cmin = 1e-12
    income =
        hasproperty(model_params, :y_dim) ? csvar_income(model_params.y) : model_params.y

    c = init_consumption_det(a_grid, a_min, R, income; c_init = c_init, cmin = cmin)

    cnew = similar(c)
    cnext = similar(c)
    resid = similar(c)
    a_next = similar(c)
    cold = similar(c)
    c_prime = similar(c)
    c_endo = similar(c)
    a_endo = similar(c)
    a_sorted = similar(a_endo)
    c_sorted = similar(c_endo)

    converged = false
    iters = 0
    max_resid = Inf
    Δpol = Inf

    for it = 1:maxit
        iters = it

        copyto!(cold, c)
        copyto!(c_prime, cold)
        ensure_minimum!(c_prime, cmin)

        if hasproperty(model_params, :y_dim) && model_params.y_dim > 1
            @inbounds for idx in eachindex(c_prime)
                cval = c_prime[idx] <= cmin ? cmin : c_prime[idx]
                integrand = _ -> model_utility.u_prime(cval)
                EU = integrate_expectation(
                    integration_method === :none ? :gh : integration_method,
                    integrand,
                    model_params,
                    nothing,
                    model_params.y,
                )
                c_endo[idx] = model_utility.u_prime_inv(β * R * EU)
            end
        else
            @. c_endo = model_utility.u_prime_inv(β * R * c_prime^(-γ))
        end
        @. a_endo = (a_grid - income + c_endo) / R

        enforce_borrowing_constraint!(a_endo, c_endo, a_min, income, R, a_grid; cmin = cmin)
        sort_policy_pairs!(a_sorted, c_sorted, a_endo, c_endo)

        interp_linear!(cnew, a_sorted, c_sorted, a_grid)
        cmax = @. income + R * a_grid - a_min
        clamp_policy!(cnew, cmin, cmax)

        Δpol = relaxation_step!(c, cold, cnew, relax)

        @. a_next = clamp(income + R * a_grid - c, a_min, a_max)
        interp_linear!(cnext, a_grid, c, a_next)
        ensure_minimum!(cnext, cmin)
        euler_resid_det!(resid, model_params, c, cnext)

        max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)

        if verbose && it % 10 == 0
            @printf("[EGM det linear] it=%d rmse=%.6e Δpol=%.6e\n", it, max_resid, Δpol)
            flush(stdout)
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    @. a_next = clamp(R * a_grid + income - c, a_min, a_max)
    interp_linear!(cnext, a_grid, c, a_next)
    ensure_minimum!(cnext, cmin)
    euler_resid_det!(resid, model_params, c, cnext)
    max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol,
        tol_pol,
        maxit,
        interp_kind = LinearInterp(),
        relax,
        verbose,
        resid_metric = :rmse,
        seed = nothing,
        runtime,
        integration_method = integration_method,
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

function solve_egm_placeholder(
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
    )
    return build_placeholder_solution(
        :EGM,
        model_params,
        model_grids,
        model_shocks;
        opts = opts,
        note = "placeholder EGM solution",
    )
end

function solve_egm_det_impl(
    ::MonotoneCubicInterp,
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 500,
    relax::Real = 0.5,
    verbose::Bool = false,
    c_init = nothing,
)::NamedTuple
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = DEFAULT_BINDING_TOL)

    β = model_params.β
    R = 1 + model_params.r
    γ = model_params.γ
    cmin = 1e-12
    income =
        hasproperty(model_params, :y_dim) ? csvar_income(model_params.y) : model_params.y

    c = init_consumption_det(a_grid, a_min, R, income; c_init = c_init, cmin = cmin)

    cnew = similar(c)
    cnext = similar(c)
    resid = similar(c)
    a_next = similar(c)
    cold = similar(c)
    c_prime = similar(c)
    c_endo = similar(c)
    a_endo = similar(c)
    a_sorted = similar(a_endo)
    c_sorted = similar(c_endo)
    converged = false
    iters = 0
    max_resid = Inf
    Δpol = Inf

    for it = 1:maxit
        iters = it

        copyto!(cold, c)
        copyto!(c_prime, cold)
        ensure_minimum!(c_prime, cmin)

        @. c_endo = model_utility.u_prime_inv(β * R * c_prime^(-γ))
        @. a_endo = (a_grid - income + c_endo) / R

        enforce_borrowing_constraint!(a_endo, c_endo, a_min, income, R, a_grid; cmin = cmin)
        sort_policy_pairs!(a_sorted, c_sorted, a_endo, c_endo)
        enforce_strict_increase!(a_sorted)

        interp_pchip!(cnew, a_sorted, c_sorted, a_grid)
        cmax = @. income + R * a_grid - a_min
        clamp_policy!(cnew, cmin, cmax)
        enforce_monotone!(cnew)

        Δpol = relaxation_step!(c, cold, cnew, relax)

        @. a_next = clamp(income + R * a_grid - c, a_min, a_max)
        interp_pchip!(cnext, a_grid, c, a_next)
        ensure_minimum!(cnext, cmin)
        euler_resid_det!(resid, model_params, c, cnext)

        max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)

        if verbose && it % 10 == 0
            @printf("[EGM det pchip] it=%d rmse=%.6e Δpol=%.6e\n", it, max_resid, Δpol)
            flush(stdout)
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    @. a_next = clamp(R * a_grid + income - c, a_min, a_max)
    interp_pchip!(cnext, a_grid, c, a_next)
    ensure_minimum!(cnext, cmin)
    euler_resid_det!(resid, model_params, c, cnext)
    max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol,
        tol_pol,
        maxit,
        interp_kind = MonotoneCubicInterp(),
        relax,
        verbose,
        resid_metric = :rmse,
        seed = nothing,
        runtime,
        integration_method = integration_method,
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

"""
    solve_egm_stoch(model_params, model_grids, model_shocks, model_utility; ...)

Vectorized EGM solver for the CS model with an AR(1) income process.
Stops when expected Euler equation errors and policy changes (evaluated at discretized nodes) meet tolerance.
Returns a `NamedTuple` with fields `(a_grid, z_grid, c, a_next, resid, iters, converged, max_resid, model_params, opts)` that is later converted into a `Solution`.
"""
function solve_egm_stoch(
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 1000,
    interp_kind::InterpKind = LinearInterp(),
    relax::Real = 0.5,
    verbose::Bool = false,
    c_init = nothing,
    integration_method::Symbol = :gh,
)::NamedTuple
    return solve_egm_stoch_impl(
        interp_kind,
        model_params,
        model_grids,
        model_shocks,
        model_utility;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        relax = relax,
        verbose = verbose,
        c_init = c_init,
        integration_method = integration_method,
    )
end

solve_egm_stoch_impl(::InterpKind, args...; kwargs...) =
    error("Unknown interpolation kind for EGM (stochastic)")

function solve_egm_stoch_impl(
    ::LinearInterp,
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 1000,
    relax::Real = 0.5,
    verbose::Bool = false,
    c_init = nothing,
    integration_method::Symbol = :gh,
)::NamedTuple
    if hasproperty(model_shocks, :process) &&
       get(model_shocks, :process, nothing) == :gaussian_linear
        return solve_egm_det(
            model_params,
            model_grids,
            model_utility;
            tol = tol,
            tol_pol = tol_pol,
            maxit = maxit,
            interp_kind = LinearInterp(),
            relax = relax,
            verbose = verbose,
            c_init = c_init isa AbstractArray ?
                     (c_init isa AbstractVector ? c_init : nothing) : c_init,
            integration_method = integration_method,
        )
    end
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = DEFAULT_BINDING_TOL)

    z_grid = model_shocks.zgrid
    Π = model_shocks.Π
    Nz = length(z_grid)

    β = model_params.β
    γ = model_params.γ
    R = 1 + model_params.r
    cmin = 1e-12

    c = c_init === nothing ? fill(1.0, Na, Nz) : copy(c_init)
    cold = similar(c)
    cnew = similar(c)
    a_next = similar(c)
    resid_mat = similar(c)
    pow = similar(c)
    EUprime = similar(view(c, :, 1))
    c_endo = similar(EUprime)
    a_endo = similar(EUprime)
    a_sorted = similar(EUprime)
    c_sorted = similar(EUprime)
    cmax = similar(EUprime)

    converged = false
    iters = 0
    max_resid = Inf
    Δpol = Inf

    for it = 1:maxit
        iters = it
        copyto!(cold, c)
        @. pow = max(cold, cmin)^(-γ)

        for (j, z) in enumerate(z_grid)
            y = exp(z)
            weights = view(Π, j, :)
            EUprime .= discrete_expectation(weights, pow)

            @. c_endo = model_utility.u_prime_inv(β * R * EUprime)
            @. a_endo = (a_grid - y + c_endo) / R

            enforce_borrowing_constraint!(a_endo, c_endo, a_min, y, R, a_grid; cmin = cmin)
            sort_policy_pairs!(a_sorted, c_sorted, a_endo, c_endo)

            interp_linear!(view(cnew, :, j), a_sorted, c_sorted, a_grid)
            @. cmax = y + R * a_grid - a_min
            clamp_policy!(view(cnew, :, j), cmin, cmax)
        end

        Δpol = relaxation_step!(c, cold, cnew, relax)

        for (j, z) in enumerate(z_grid)
            y = exp(z)
            @views @. a_next[:, j] = clamp(R * a_grid + y - c[:, j], a_min, a_max)
        end

        euler_resid_stoch_interp!(
            resid_mat,
            model_params,
            a_grid,
            z_grid,
            Π,
            c,
            LinearInterp(),
        )

        max_resid = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)

        if verbose && it % 10 == 0
            @printf("[EGM stoch linear] it=%d rmse=%.6e Δpol=%.6e\n", it, max_resid, Δpol)
            flush(stdout)
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    for (j, z) in enumerate(z_grid)
        y = exp(z)
        @views @. a_next[:, j] = clamp(R * a_grid + y - c[:, j], a_min, a_max)
    end

    euler_resid_stoch_interp!(resid_mat, model_params, a_grid, z_grid, Π, c, LinearInterp())
    max_resid = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol,
        tol_pol,
        maxit,
        interp_kind = LinearInterp(),
        relax,
        verbose,
        seed = nothing,
        runtime,
        integration_method = integration_method,
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
        model_params,
        opts,
        delta_pol = Δpol,
    )
end

function solve_egm_stoch_impl(
    ::MonotoneCubicInterp,
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 1000,
    relax::Real = 0.5,
    verbose::Bool = false,
    c_init = nothing,
    integration_method::Symbol = :gh,
)::NamedTuple
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = DEFAULT_BINDING_TOL)

    z_grid = model_shocks.zgrid
    Π = model_shocks.Π
    Nz = length(z_grid)

    β = model_params.β
    γ = model_params.γ
    R = 1 + model_params.r
    cmin = 1e-12

    c = c_init === nothing ? fill(1.0, Na, Nz) : copy(c_init)
    cold = similar(c)
    cnew = similar(c)
    a_next = similar(c)
    resid_mat = similar(c)
    pow = similar(c)
    EUprime = similar(view(c, :, 1))
    c_endo = similar(EUprime)
    a_endo = similar(EUprime)
    a_sorted = similar(EUprime)
    c_sorted = similar(EUprime)
    cmax = similar(EUprime)

    converged = false
    iters = 0
    max_resid = Inf
    Δpol = Inf

    for it = 1:maxit
        iters = it
        copyto!(cold, c)
        @. pow = max(cold, cmin)^(-γ)

        for (j, z) in enumerate(z_grid)
            y = exp(z)
            weights = view(Π, j, :)
            EUprime .= discrete_expectation(weights, pow)

            @. c_endo = model_utility.u_prime_inv(β * R * EUprime)
            @. a_endo = (a_grid - y + c_endo) / R

            enforce_borrowing_constraint!(a_endo, c_endo, a_min, y, R, a_grid; cmin = cmin)
            sort_policy_pairs!(a_sorted, c_sorted, a_endo, c_endo)
            enforce_strict_increase!(a_sorted)
            enforce_monotone!(c_sorted)

            column = view(cnew, :, j)
            interp_pchip!(column, a_sorted, c_sorted, a_grid)

            @. cmax = y + R * a_grid - a_min
            clamp_policy!(column, cmin, cmax)
            enforce_monotone!(column)
        end

        Δpol = relaxation_step!(c, cold, cnew, relax)

        for (j, z) in enumerate(z_grid)
            y = exp(z)
            @views @. a_next[:, j] = clamp(R * a_grid + y - c[:, j], a_min, a_max)
        end

        euler_resid_stoch_interp!(
            resid_mat,
            model_params,
            a_grid,
            z_grid,
            Π,
            c,
            MonotoneCubicInterp(),
        )

        max_resid = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)

        if verbose && it % 10 == 0
            @printf("[EGM stoch pchip] it=%d rmse=%.6e Δpol=%.6e\n", it, max_resid, Δpol)
            flush(stdout)
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    for (j, z) in enumerate(z_grid)
        y = exp(z)
        @views @. a_next[:, j] = clamp(R * a_grid + y - c[:, j], a_min, a_max)
    end

    euler_resid_stoch_interp!(
        resid_mat,
        model_params,
        a_grid,
        z_grid,
        Π,
        c,
        MonotoneCubicInterp(),
    )
    max_resid = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol,
        tol_pol,
        maxit,
        interp_kind = MonotoneCubicInterp(),
        relax,
        verbose,
        resid_metric = :rmse,
        seed = nothing,
        runtime,
        integration_method = integration_method,
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

end #module
