"""
EGMKernel

Endogenous Grid Method solver kernel for the consumption–savings model.
Exports helpers and the main projection-based policy iteration.
"""
module EGMKernel

using Base.Threads: @threads
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
using ..CSVarUtils: csvar_state_incomes, csvar_state_matrix
using ..SolverIntegration: integrate_expectation, discrete_expectation
using Random: default_rng
using ..SolverPlaceholders: build_placeholder_solution
using ..GridHelpers: fit_values_on_backend!, eval_backend_at_points
using Printf

export solve_egm_det, solve_egm_stoch, solve_egm_placeholder

const DEFAULT_BINDING_TOL = 1e-12

@inline function _is_csvar_model(params)
    hasproperty(params, :y_dim) && getproperty(params, :y_dim) > 1
end

function _csvar_state_data(params)
    _is_csvar_model(params) ||
        error("CSVAR state data requested for model without vector income")
    y_dim = params.y_dim
    Y = csvar_state_matrix(params.y, y_dim)
    incomes = csvar_state_incomes(Y)
    return Y, incomes
end

function _prepare_csvar_initial_consumption(a_grid, a_min, R, incomes; c_init, cmin)
    Na = length(a_grid)
    Ny = length(incomes)
    if c_init === nothing
        c = Array{Float64}(undef, Na, Ny)
        for (j, income) in enumerate(incomes)
            column = init_consumption_det(a_grid, a_min, R, income; cmin = cmin)
            @views c[:, j] .= column
        end
        return c
    elseif c_init isa AbstractArray
        return copy(c_init)
    else
        error("CSVAR warm start must be an array when provided")
    end
end

"""
    solve_egm_det(model_params, model_grids, model_utility; ...)

Vectorized EGM solver for deterministic income (equivalent to log-normal income with zero variance).
Stops when Euler equation errors and policy changes fall below their respective tolerances.
Returns a `NamedTuple` with fields `(a_grid, c, a_next, resid, iters, converged, euler_rmse, model_params, opts)`
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
    rng = nothing,
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
        rng = rng,
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
    rng = nothing,
)::NamedTuple
    start_time = time_ns()
    local_rng = rng === nothing ? default_rng() : rng

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = DEFAULT_BINDING_TOL)

    β = model_params.β
    R = 1 + model_params.r
    γ = model_params.γ
    cmin = 1e-12
    if _is_csvar_model(model_params)
        y_states, incomes = _csvar_state_data(model_params)
        c = _prepare_csvar_initial_consumption(
            a_grid,
            a_min,
            R,
            incomes;
            c_init = c_init,
            cmin = cmin,
        )

        cnew = similar(c)
        cnext = similar(c)
        resid = similar(c)
        a_next = similar(c)
        cold = similar(c)
        c_prime = similar(c)

        c_endo_vec = similar(a_grid)
        a_endo_vec = similar(a_grid)
        a_sorted = similar(a_grid)
        c_sorted = similar(a_grid)
        cmax_vec = similar(a_grid)

        converged = false
        iters = 0
        euler_rmse = Inf
        Δpol = Inf
        integ_kind = integration_method === :none ? :gh : integration_method

        for it = 1:maxit
            iters = it

            copyto!(cold, c)
            copyto!(c_prime, cold)
            ensure_minimum!(c_prime, cmin)

            for (j, income) in enumerate(incomes)
                cprime_col = view(c_prime, :, j)
                y_state = view(y_states, :, j)
                for idx in eachindex(cprime_col)
                    cval = cprime_col[idx] <= cmin ? cmin : cprime_col[idx]
                    integrand = _ -> model_utility.u_prime(cval)
                    EU = integrate_expectation(
                        integ_kind,
                        integrand,
                        model_params,
                        nothing,
                        y_state,
                        rng = local_rng,
                    )
                    c_endo_vec[idx] = model_utility.u_prime_inv(β * R * EU)
                end
                @. a_endo_vec = (a_grid - income + c_endo_vec) / R
                enforce_borrowing_constraint!(
                    a_endo_vec,
                    c_endo_vec,
                    a_min,
                    income,
                    R,
                    a_grid;
                    cmin = cmin,
                )
                sort_policy_pairs!(a_sorted, c_sorted, a_endo_vec, c_endo_vec)

                column_new = view(cnew, :, j)
                a_info = model_grids[:a]
                backend = fit_values_on_backend!(a_info, a_sorted, reshape(c_sorted, :, 1))
                column_new .= eval_backend_at_points(backend, a_grid, 1)
                @. cmax_vec = income + R * a_grid - a_min
                clamp_policy!(column_new, cmin, cmax_vec)
            end

            Δpol = relaxation_step!(c, cold, cnew, relax)

            for (j, income) in enumerate(incomes)
                col_next = view(a_next, :, j)
                col_c = view(c, :, j)
                @. col_next = clamp(income + R * a_grid - col_c, a_min, a_max)
                a_info = model_grids[:a]
                backend = fit_values_on_backend!(a_info, a_grid, reshape(col_c, :, 1))
                view(cnext, :, j) .= eval_backend_at_points(backend, col_next, 1)
            end
            ensure_minimum!(cnext, cmin)

            for j = 1:length(incomes)
                euler_resid_det!(
                    view(resid, :, j),
                    model_params,
                    view(c, :, j),
                    view(cnext, :, j),
                )
            end

            euler_rmse = rmse_nonbinding(resid, a_next, a_min, bind_tol)

            if verbose && it % 10 == 0
                @printf(
                    "[EGM det linear] it=%d rmse=%.6e Δpol=%.6e\n",
                    it,
                    euler_rmse,
                    Δpol,
                )
                flush(stdout)
            end

            if euler_rmse < tol && Δpol < tol_pol
                converged = true
                break
            end
        end

        for (j, income) in enumerate(incomes)
            col_next = view(a_next, :, j)
            col_c = view(c, :, j)
            @. col_next = clamp(R * a_grid + income - col_c, a_min, a_max)
            a_info = model_grids[:a]
            backend = fit_values_on_backend!(a_info, a_grid, reshape(col_c, :, 1))
            view(cnext, :, j) .= eval_backend_at_points(backend, col_next, 1)
        end
        ensure_minimum!(cnext, cmin)
        for j = 1:length(incomes)
            euler_resid_det!(
                view(resid, :, j),
                model_params,
                view(c, :, j),
                view(cnext, :, j),
            )
        end
        euler_rmse = rmse_nonbinding(resid, a_next, a_min, bind_tol)

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
            euler_rmse,
            rmse = euler_rmse,
            model_params,
            opts,
            delta_pol = Δpol,
        )
    end

    income = model_params.y

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
    euler_rmse = Inf
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

        a_info = model_grids[:a]
        backend = fit_values_on_backend!(a_info, a_sorted, reshape(c_sorted, :, 1))
        cnew .= eval_backend_at_points(backend, a_grid, 1)
        cmax = @. income + R * a_grid - a_min
        clamp_policy!(cnew, cmin, cmax)

        Δpol = relaxation_step!(c, cold, cnew, relax)

        @. a_next = clamp(income + R * a_grid - c, a_min, a_max)
        a_info = model_grids[:a]
        backend = fit_values_on_backend!(a_info, a_grid, reshape(c, :, 1))
        cnext .= eval_backend_at_points(backend, a_next, 1)
        ensure_minimum!(cnext, cmin)
        euler_resid_det!(resid, model_params, c, cnext)

        euler_rmse = rmse_nonbinding(resid, a_next, a_min, bind_tol)

        if verbose && it % 10 == 0
            @printf("[EGM det linear] it=%d rmse=%.6e Δpol=%.6e\n", it, euler_rmse, Δpol)
            flush(stdout)
        end

        if euler_rmse < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    @. a_next = clamp(R * a_grid + income - c, a_min, a_max)
    interp_linear!(cnext, a_grid, c, a_next)
    ensure_minimum!(cnext, cmin)
    euler_resid_det!(resid, model_params, c, cnext)
    euler_rmse = rmse_nonbinding(resid, a_next, a_min, bind_tol)

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
        euler_rmse,
        rmse = euler_rmse,
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
    integration_method::Symbol = :none,
    rng = nothing,
)::NamedTuple
    start_time = time_ns()
    local_rng = rng === nothing ? default_rng() : rng

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = DEFAULT_BINDING_TOL)

    β = model_params.β
    R = 1 + model_params.r
    γ = model_params.γ
    cmin = 1e-12
    if _is_csvar_model(model_params)
        y_states, incomes = _csvar_state_data(model_params)
        c = _prepare_csvar_initial_consumption(
            a_grid,
            a_min,
            R,
            incomes;
            c_init = c_init,
            cmin = cmin,
        )

        cnew = similar(c)
        cnext = similar(c)
        resid = similar(c)
        a_next = similar(c)
        cold = similar(c)
        c_prime = similar(c)

        c_endo_vec = similar(a_grid)
        a_endo_vec = similar(a_grid)
        a_sorted = similar(a_grid)
        c_sorted = similar(a_grid)
        cmax_vec = similar(a_grid)

        converged = false
        iters = 0
        euler_rmse = Inf
        Δpol = Inf
        integ_kind = integration_method === :none ? :gh : integration_method

        for it = 1:maxit
            iters = it

            copyto!(cold, c)
            copyto!(c_prime, cold)
            ensure_minimum!(c_prime, cmin)

            for (j, income) in enumerate(incomes)
                cprime_col = view(c_prime, :, j)
                y_state = view(y_states, :, j)
                for idx in eachindex(cprime_col)
                    cval = cprime_col[idx] <= cmin ? cmin : cprime_col[idx]
                    integrand = _ -> model_utility.u_prime(cval)
                    EU = integrate_expectation(
                        integ_kind,
                        integrand,
                        model_params,
                        nothing,
                        y_state,
                        rng = local_rng,
                    )
                    c_endo_vec[idx] = model_utility.u_prime_inv(β * R * EU)
                end
                @. a_endo_vec = (a_grid - income + c_endo_vec) / R
                enforce_borrowing_constraint!(
                    a_endo_vec,
                    c_endo_vec,
                    a_min,
                    income,
                    R,
                    a_grid;
                    cmin = cmin,
                )
                sort_policy_pairs!(a_sorted, c_sorted, a_endo_vec, c_endo_vec)
                enforce_strict_increase!(a_sorted)
                enforce_monotone!(c_sorted)

                column_new = view(cnew, :, j)
                a_info = model_grids[:a]
                backend = fit_values_on_backend!(a_info, a_sorted, reshape(c_sorted, :, 1))
                column_new .= eval_backend_at_points(backend, a_grid, 1)

                @. cmax_vec = income + R * a_grid - a_min
                clamp_policy!(column_new, cmin, cmax_vec)
                enforce_monotone!(column_new)
            end

            Δpol = relaxation_step!(c, cold, cnew, relax)

            for (j, income) in enumerate(incomes)
                col_next = view(a_next, :, j)
                col_c = view(c, :, j)
                @. col_next = clamp(income + R * a_grid - col_c, a_min, a_max)
                a_info = model_grids[:a]
                backend = fit_values_on_backend!(a_info, a_grid, reshape(col_c, :, 1))
                view(cnext, :, j) .= eval_backend_at_points(backend, col_next, 1)
            end
            ensure_minimum!(cnext, cmin)

            for j = 1:length(incomes)
                euler_resid_det!(
                    view(resid, :, j),
                    model_params,
                    view(c, :, j),
                    view(cnext, :, j),
                )
            end

            euler_rmse = rmse_nonbinding(resid, a_next, a_min, bind_tol)

            if verbose && it % 10 == 0
                @printf(
                    "[EGM det pchip] it=%d rmse=%.6e Δpol=%.6e\n",
                    it,
                    euler_rmse,
                    Δpol,
                )
                flush(stdout)
            end

            if euler_rmse < tol && Δpol < tol_pol
                converged = true
                break
            end
        end

        for (j, income) in enumerate(incomes)
            col_next = view(a_next, :, j)
            col_c = view(c, :, j)
            @. col_next = clamp(R * a_grid + income - col_c, a_min, a_max)
            a_info = model_grids[:a]
            backend = fit_values_on_backend!(a_info, a_grid, reshape(col_c, :, 1))
            view(cnext, :, j) .= eval_backend_at_points(backend, col_next, 1)
        end
        ensure_minimum!(cnext, cmin)
        for j = 1:length(incomes)
            euler_resid_det!(
                view(resid, :, j),
                model_params,
                view(c, :, j),
                view(cnext, :, j),
            )
        end
        euler_rmse = rmse_nonbinding(resid, a_next, a_min, bind_tol)

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
            euler_rmse,
            rmse = euler_rmse,
            model_params,
            opts,
            delta_pol = Δpol,
        )
    end

    income = model_params.y

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
    euler_rmse = Inf
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

        a_info = model_grids[:a]
        backend = fit_values_on_backend!(a_info, a_sorted, reshape(c_sorted, :, 1))
        cnew .= eval_backend_at_points(backend, a_grid, 1)
        cmax = @. income + R * a_grid - a_min
        clamp_policy!(cnew, cmin, cmax)
        enforce_monotone!(cnew)

        Δpol = relaxation_step!(c, cold, cnew, relax)

        @. a_next = clamp(income + R * a_grid - c, a_min, a_max)
        a_info = model_grids[:a]
        backend = fit_values_on_backend!(a_info, a_grid, c)
        cnext .= eval_backend_at_points(backend, a_next, 1)
        ensure_minimum!(cnext, cmin)
        euler_resid_det!(resid, model_params, c, cnext)

        euler_rmse = rmse_nonbinding(resid, a_next, a_min, bind_tol)

        if verbose && it % 10 == 0
            @printf("[EGM det pchip] it=%d rmse=%.6e Δpol=%.6e\n", it, euler_rmse, Δpol)
            flush(stdout)
        end

        if euler_rmse < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    @. a_next = clamp(R * a_grid + income - c, a_min, a_max)
    a_info = model_grids[:a]
    backend = fit_values_on_backend!(a_info, a_grid, c)
    cnext .= eval_backend_at_points(backend, a_next, 1)
    ensure_minimum!(cnext, cmin)
    euler_resid_det!(resid, model_params, c, cnext)
    euler_rmse = rmse_nonbinding(resid, a_next, a_min, bind_tol)

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
        euler_rmse,
        rmse = euler_rmse,
        model_params,
        opts,
        delta_pol = Δpol,
    )
end

"""
    solve_egm_stoch(model_params, model_grids, model_shocks, model_utility; ...)

Vectorized EGM solver for the CS model with an AR(1) income process.
Stops when expected Euler equation errors and policy changes (evaluated at discretized nodes) meet tolerance.
Returns a `NamedTuple` with fields `(a_grid, z_grid, c, a_next, resid, iters, converged, euler_rmse, model_params, opts)` that is later converted into a `Solution`.
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
    rng = nothing,
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
        rng = rng,
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
    rng = nothing,
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
            rng = rng,
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

    converged = false
    iters = 0
    euler_rmse = Inf
    Δpol = Inf
    rmse_history = Float64[]

    for it = 1:maxit
        iters = it
        copyto!(cold, c)
        @. pow = max(cold, cmin)^(-γ)

        @threads for j = 1:Nz
            z = z_grid[j]
            y = exp(z)
            weights = view(Π, j, :)

            # Thread-local buffers (Na+1 to accommodate constraint point)
            EUprime_local = Vector{Float64}(undef, Na)
            c_endo_local = Vector{Float64}(undef, Na + 1)
            a_endo_local = Vector{Float64}(undef, Na + 1)
            a_sorted_local = Vector{Float64}(undef, Na + 1)
            c_sorted_local = Vector{Float64}(undef, Na + 1)
            cmax_local = similar(view(c, :, 1))

            EUprime_local .= discrete_expectation(weights, pow)

            # Compute endogenous grid from Euler equation (indices 2:Na+1)
            @inbounds for i = 1:Na
                c_endo_local[i+1] = model_utility.u_prime_inv(β * R * EUprime_local[i])
                a_endo_local[i+1] = (a_grid[i] - y + c_endo_local[i+1]) / R
            end

            # Add borrowing constraint point at index 1
            a_endo_local[1] = a_min
            c_endo_local[1] = max(y + (R - 1) * a_min, cmin)

            # Enforce constraint on remaining points (clamp any violations)
            enforce_borrowing_constraint!(
                view(a_endo_local, 2:Na+1),
                view(c_endo_local, 2:Na+1),
                a_min,
                y,
                R,
                a_grid;
                cmin = cmin,
            )
            sort_policy_pairs!(a_sorted_local, c_sorted_local, a_endo_local, c_endo_local)

            # DEBUG: check grid coverage
            if it == 1 && j == 1
                a_endo_min, a_endo_max = extrema(a_sorted_local)
                a_exo_min, a_exo_max = extrema(a_grid)
                println("[EGM DEBUG] Endogenous grid: [$a_endo_min, $a_endo_max]")
                println("[EGM DEBUG] Exogenous grid:  [$a_exo_min, $a_exo_max]")
                println(
                    "[EGM DEBUG] c at constraint (a=0): ",
                    a_sorted_local[1],
                    " -> ",
                    c_sorted_local[1],
                )
                println(
                    "[EGM DEBUG] c for next point: ",
                    a_sorted_local[2],
                    " -> ",
                    c_sorted_local[2],
                )
                if a_exo_min < a_endo_min || a_exo_max > a_endo_max
                    println(
                        "[EGM DEBUG] WARNING: Exogenous grid extends beyond endogenous grid!",
                    )
                end
            end

            # Clamp endogenous grid to stay within exogenous bounds
            # This prevents extrapolation errors at the boundaries
            @. a_sorted_local = clamp(a_sorted_local, a_min, a_max)

            a_info = model_grids[:a]
            backend = fit_values_on_backend!(
                a_info,
                a_sorted_local,
                reshape(c_sorted_local, :, 1),
            )
            view(cnew, :, j) .= eval_backend_at_points(backend, a_grid, 1)
            @. cmax_local = y + R * a_grid - a_min
            clamp_policy!(view(cnew, :, j), cmin, cmax_local)
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

        euler_rmse = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)
        push!(rmse_history, euler_rmse)

        if verbose && it % 10 == 0
            @printf("[EGM stoch linear] it=%d rmse=%.6e Δpol=%.6e\n", it, euler_rmse, Δpol)
            flush(stdout)
        end

        if euler_rmse < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    for (j, z) in enumerate(z_grid)
        y = exp(z)
        @views @. a_next[:, j] = clamp(R * a_grid + y - c[:, j], a_min, a_max)
    end

    euler_resid_stoch_interp!(resid_mat, model_params, a_grid, z_grid, Π, c, LinearInterp())
    euler_rmse = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)

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
        euler_rmse,
        model_params,
        opts,
        delta_pol = Δpol,
        rmse_history,
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
    rng = nothing,
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

    converged = false
    iters = 0
    euler_rmse = Inf
    Δpol = Inf
    rmse_history = Float64[]

    for it = 1:maxit
        iters = it
        copyto!(cold, c)
        @. pow = max(cold, cmin)^(-γ)

        @threads for j = 1:Nz
            z = z_grid[j]
            y = exp(z)
            weights = view(Π, j, :)

            # Thread-local buffers (Na+1 to accommodate constraint point)
            EUprime_local = Vector{Float64}(undef, Na)
            c_endo_local = Vector{Float64}(undef, Na + 1)
            a_endo_local = Vector{Float64}(undef, Na + 1)
            a_sorted_local = Vector{Float64}(undef, Na + 1)
            c_sorted_local = Vector{Float64}(undef, Na + 1)
            cmax_local = similar(view(c, :, 1))

            EUprime_local .= discrete_expectation(weights, pow)

            # Compute endogenous grid from Euler equation (indices 2:Na+1)
            @inbounds for i = 1:Na
                c_endo_local[i+1] = model_utility.u_prime_inv(β * R * EUprime_local[i])
                a_endo_local[i+1] = (a_grid[i] - y + c_endo_local[i+1]) / R
            end

            # Add borrowing constraint point at index 1
            a_endo_local[1] = a_min
            c_endo_local[1] = max(y + (R - 1) * a_min, cmin)

            # Enforce constraint on remaining points (clamp any violations)
            enforce_borrowing_constraint!(
                view(a_endo_local, 2:Na+1),
                view(c_endo_local, 2:Na+1),
                a_min,
                y,
                R,
                a_grid;
                cmin = cmin,
            )
            sort_policy_pairs!(a_sorted_local, c_sorted_local, a_endo_local, c_endo_local)
            enforce_strict_increase!(a_sorted_local)
            enforce_monotone!(c_sorted_local)

            column = view(cnew, :, j)
            a_info = model_grids[:a]
            backend = fit_values_on_backend!(
                a_info,
                a_sorted_local,
                reshape(c_sorted_local, :, 1),
            )
            column .= eval_backend_at_points(backend, a_grid, 1)

            @. cmax_local = y + R * a_grid - a_min
            clamp_policy!(column, cmin, cmax_local)
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

        euler_rmse = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)
        push!(rmse_history, euler_rmse)

        if verbose && it % 10 == 0
            @printf("[EGM stoch pchip] it=%d rmse=%.6e Δpol=%.6e\n", it, euler_rmse, Δpol)
            flush(stdout)
        end

        if euler_rmse < tol && Δpol < tol_pol
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
    euler_rmse = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)

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
        euler_rmse,
        rmse = euler_rmse,
        model_params,
        opts,
        delta_pol = Δpol,
        rmse_history,
    )
end

end #module
