"""
EGMKernel

Endogenous Grid Method solver kernel for the consumption–savings model.
Exports helpers and the main projection-based policy iteration.
"""
module EGMKernel

using Base.Threads: @threads
using ..CommonInterp: interpolate, InterpKind, LinearInterp, MonotoneCubicInterp
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
using ..CSVarUtils:
    csvar_state_incomes,
    csvar_state_matrix,
    is_csvar_model,
    csvar_state_data,
    prepare_csvar_initial_consumption
using ..SolverIntegration: integrate_expectation, discrete_expectation
using Random: default_rng
using ..SolverPlaceholders: build_placeholder_solution
using ..GridHelpers: fit_values_on_backend!, eval_backend_at_points
using Printf

export solve_egm, solve_egm_placeholder

const DEFAULT_BINDING_TOL = 1e-12

using ..CSVarUtils:
    csvar_state_incomes,
    csvar_state_matrix,
    is_csvar_model,
    csvar_state_data,
    prepare_csvar_initial_consumption



"""
    solve_egm(model_params, model_grids, model_shocks, model_utility; ...)

General EGM solver for deterministic and stochastic cases.
"""
function solve_egm(
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
            weights = Π[j, :]

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

            # NOTE: Removed clamping of endogenous grid to avoid introducing duplicate points
            # The interpolation should handle extrapolation or we should filter points instead
            # @. a_sorted_local = clamp(a_sorted_local, a_min, a_max)

            # Interpolate using the general dispatcher
            interpolate(
                view(cnew, :, j),
                a_sorted_local,
                c_sorted_local,
                a_grid,
                interp_kind,
            )
            @. cmax_local = y + R * a_grid - a_min
            clamp_policy!(view(cnew, :, j), cmin, cmax_local)
            if interp_kind isa MonotoneCubicInterp
                enforce_monotone!(view(cnew, :, j))
            end
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
            interp_kind,
        )

        euler_rmse = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)
        push!(rmse_history, euler_rmse)

        if verbose && it % 10 == 0
            @printf("[EGM] it=%d rmse=%.6e Δpol=%.6e\n", it, euler_rmse, Δpol)
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

    euler_resid_stoch_interp!(resid_mat, model_params, a_grid, z_grid, Π, c, interp_kind)
    euler_rmse = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol,
        tol_pol,
        maxit,
        interp_kind = interp_kind,
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

end #module
