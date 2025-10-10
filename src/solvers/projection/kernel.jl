"""
ProjectionKernel

Chebyshev-based projection solvers for deterministic and stochastic variants of
the consumption-saving model.
"""
module ProjectionKernel

using ..Chebyshev: chebyshev_basis, gauss_lobatto_nodes
using ..ProjectionCoefficients: solve_coefficients
using ..EulerResiduals:
    euler_resid_det, euler_resid_stoch, euler_resid_det_grid, euler_resid_stoch_grid
using ..CommonInterp: interp_pchip!
using ..PolicyUtils:
    clamp_policy!, compute_binding_tolerance, init_consumption_det, rmse_nonbinding
using ..CommonValidators: is_nondec
using ..SolverIntegration: integrate_expectation, discrete_expectation
using ..CSVarUtils: csvar_income
using Statistics: mean
using Random: default_rng
using LinearAlgebra: dot

export solve_projection_det, solve_projection_stoch

const Β_SYM = Symbol(Char(0x03B2))
const PI_TRANSITION_SYM = Symbol(Char(0x03A0))
const DEFAULT_BINDING_TOL = 1e-12
const MAX_CHEBYSHEV_DEGREE = 15
const MAX_COLLOCATION_POINTS = MAX_CHEBYSHEV_DEGREE + 1

@inline function clamp_orders(orders::AbstractVector{Int}, max_order::Int)
    return clamp.(orders, 0, max_order)
end

@inline function gauss_lobatto_or_midpoint(N::Int, a_min::Real, a_max::Real)
    if N <= 0
        return Float64[]
    elseif N == 1
        midpoint = (a_min + a_max) / 2
        return fill(midpoint, 1)
    else
        return gauss_lobatto_nodes(N, a_min, a_max)
    end
end

# -----------------------------------------------------------------------------
# Deterministic solver
# -----------------------------------------------------------------------------

function solve_projection_det(
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-6,
    maxit::Int = 1000,
    orders::AbstractVector{Int} = Int[],
    Nval::Int = model_grids[:a].N,
    λ::Real = 0.0,
    tol_pol::Real = tol,
    rng = nothing,
)::NamedTuple
    start_time = time_ns()
    local_rng = rng === nothing ? default_rng() : rng

    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = clamp(model_grids[:a].N, 2, MAX_COLLOCATION_POINTS)
    a_grid = gauss_lobatto_nodes(Na, a_min, a_max)
    a_out = model_grids[:a].grid
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = DEFAULT_BINDING_TOL)
    β = getproperty(model_params, Β_SYM)
    income =
        hasproperty(model_params, :y_dim) ? csvar_income(model_params.y) : model_params.y
    R = 1 + model_params.r

    candidate_orders = isempty(orders) ? Int[Na-1] : clamp_orders(orders, Na - 1)
    max_order = maximum(candidate_orders)

    a_val = gauss_lobatto_or_midpoint(Nval, a_min, a_max)
    B_cache = chebyshev_basis(a_grid, max_order, a_min, a_max)
    B_val_cache = chebyshev_basis(a_val, max_order, a_min, a_max)
    B_out_cache = chebyshev_basis(a_out, max_order, a_min, a_max)

    best_coeffs = nothing
    best_c = nothing
    best_a_next = nothing
    best_resid = nothing
    best_max_resid = Inf
    best_val_resid = Inf
    best_iters = 0
    best_converged = false
    best_order = candidate_orders[1]
    best_delta = nothing

    cmin = 1e-12

    available_grid = income .+ R .* a_grid .- a_min

    for order in candidate_orders
        B = B_cache[:, 1:(order+1)]
        c = init_consumption_det(a_grid, a_min, R, income; cmin = cmin)
        coeffs = solve_coefficients(B, c; λ = λ)
        a_next = similar(c)
        c_new = similar(c)
        converged = false
        iters = 0

        last_delta = Inf
        for it = 1:maxit
            iters = it
            @. a_next = clamp(R * a_grid + income - c, a_min, a_max)
            Bnext = chebyshev_basis(a_next, order, a_min, a_max)
            c_next .= Bnext * coeffs
            c_new .= model_utility.u_prime_inv(β * R * model_utility.u_prime(c_next))
            clamp_policy!(c_new, cmin, available_grid)
            coeffs = solve_coefficients(B, c_new; λ = λ)
            delta = maximum(abs.(c_new .- c))
            last_delta = delta
            c .= c_new
            if delta < tol_pol
                converged = true
                break
            end
        end

        @. a_next = clamp(R * a_grid + income - c, a_min, a_max)
        B_val = B_val_cache[:, 1:(order+1)]
        c_val = B_val * coeffs
        resid_val = euler_resid_det_grid(model_params, a_val, c_val)
        a_next_val = clamp.(R .* a_val .+ income .- c_val, a_min, a_max)
        max_resid_val = rmse_nonbinding(resid_val, a_next_val, a_min, bind_tol)

        if max_resid_val < best_val_resid
            best_val_resid = max_resid_val
            best_coeffs = coeffs
            best_c = copy(c)
            best_a_next = copy(a_next)
            best_resid = euler_resid_det_grid(model_params, a_grid, c)
            best_max_resid = rmse_nonbinding(best_resid, best_a_next, a_min, bind_tol)
            best_iters = iters
            best_converged = converged
            best_order = order
            best_delta = last_delta
        end
    end

    B_out = B_out_cache[:, 1:(best_order+1)]
    c_out = B_out * best_coeffs
    if !is_nondec(c_out)
        interp_pchip!(c_out, a_grid, best_c, a_out)
    end
    a_next_out = clamp.(R .* a_out .+ income .- c_out, a_min, a_max)
    resid_out = euler_resid_det_grid(model_params, a_out, c_out)
    max_resid_out = rmse_nonbinding(resid_out, a_next_out, a_min, bind_tol)

    runtime = (time_ns() - start_time) / 1e9
    # include policy tolerance in opts
    opts = (;
        tol,
        tol_pol,
        maxit,
        order = best_order,
        runtime,
        seed = nothing,
        resid_metric = :rmse,
    )

    return (
        a_grid = a_out,
        c = c_out,
        a_next = a_next_out,
        resid = resid_out,
        iters = best_iters,
        converged = best_converged,
        max_resid = max_resid_out,
        rmse = max_resid_out,
        model_params = model_params,
        coeffs = best_coeffs,
        opts = opts,
        delta_pol = best_delta,
    )
end

# -----------------------------------------------------------------------------
# Stochastic solver
# -----------------------------------------------------------------------------

function solve_projection_stoch(
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-6,
    maxit::Int = 1000,
    orders::AbstractVector{Int} = Int[],
    Nval::Int = model_grids[:a].N,
    λ::Real = 0.0,
    tol_pol::Real = tol,
    integration_method::Symbol = :gh,
    gh_order::Int = 3,
    nsamples::Int = 128,
    rng = nothing,
)::NamedTuple
    if hasproperty(model_shocks, :process) && model_shocks.process == :gaussian_linear
        return solve_projection_csvar(
            model_params,
            model_grids,
            model_shocks,
            model_utility;
            tol = tol,
            maxit = maxit,
            orders = orders,
            Nval = Nval,
            λ = λ,
            tol_pol = tol_pol,
            integration_method = integration_method,
            gh_order = gh_order,
            nsamples = nsamples,
            rng = rng,
        )
    end
    start_time = time_ns()

    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = clamp(model_grids[:a].N, 2, MAX_COLLOCATION_POINTS)
    a_grid = gauss_lobatto_nodes(Na, a_min, a_max)
    a_out = model_grids[:a].grid
    a_val = gauss_lobatto_or_midpoint(Nval, a_min, a_max)
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = DEFAULT_BINDING_TOL)

    z_grid = model_shocks.zgrid
    transition = getproperty(model_shocks, PI_TRANSITION_SYM)
    Nz = length(z_grid)

    β = getproperty(model_params, Β_SYM)
    R = 1 + model_params.r

    candidate_orders = isempty(orders) ? Int[Na-1] : clamp_orders(orders, Na - 1)
    max_order = maximum(candidate_orders)

    B_cache = chebyshev_basis(a_grid, max_order, a_min, a_max)
    B_val_cache = chebyshev_basis(a_val, max_order, a_min, a_max)
    B_out_cache = chebyshev_basis(a_out, max_order, a_min, a_max)

    best_coeffs = nothing
    best_c = nothing
    best_a_next = nothing
    best_resid = nothing
    best_max_resid = Inf
    best_val_resid = Inf
    best_iters = 0
    best_converged = false
    best_order = candidate_orders[1]
    best_delta = nothing

    cmin = 1e-12

    available = similar(a_grid)

    for order in candidate_orders
        B = B_cache[:, 1:(order+1)]
        c = Matrix{Float64}(undef, Na, Nz)
        for j = 1:Nz
            income = exp(z_grid[j])
            @. available = income + R * a_grid - a_min
            col = view(c, :, j)
            @. col = 0.5 * available
            clamp_policy!(col, cmin, available)
        end

        coeffs = solve_coefficients(B, c; λ = λ)
        a_next = similar(c)
        c_new = similar(c)
        expected_marginal = similar(a_grid)
        tmp = similar(a_grid)
        converged = false
        iters = 0

        last_delta = Inf
        for it = 1:maxit
            iters = it
            for j = 1:Nz
                income = exp(z_grid[j])
                @views @. a_next[:, j] = clamp(R * a_grid + income - c[:, j], a_min, a_max)
                Bnext = chebyshev_basis(view(a_next, :, j), order, a_min, a_max)
                fill!(expected_marginal, 0.0)
                for jp = 1:Nz
                    tmp .= Bnext * view(coeffs, :, jp)
                    @. expected_marginal += transition[j, jp] * model_utility.u_prime(tmp)
                end
                @. available = income + R * a_grid - a_min
                view_cnew = view(c_new, :, j)
                view_cnew .= model_utility.u_prime_inv(β * R * expected_marginal)
                clamp_policy!(view_cnew, cmin, available)
            end

            coeffs = solve_coefficients(B, c_new; λ = λ)
            delta = maximum(abs.(c_new .- c))
            last_delta = delta
            c .= c_new
            resid_mat = euler_resid_stoch_grid(model_params, a_grid, z_grid, transition, c)
            max_resid = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)
            # use tol_pol for policy iteration delta and tol for residual
            if delta < tol_pol && max_resid < tol
                converged = true
                break
            end
        end

        for j = 1:Nz
            income = exp(z_grid[j])
            @views @. a_next[:, j] = clamp(R * a_grid + income - c[:, j], a_min, a_max)
        end

        B_val = B_val_cache[:, 1:(order+1)]
        c_val = B_val * coeffs
        resid_val = euler_resid_stoch_grid(model_params, a_val, z_grid, transition, c_val)
        a_next_val = similar(c_val)
        for j = 1:Nz
            income = exp(z_grid[j])
            @views @. a_next_val[:, j] =
                clamp(R * a_val + income - c_val[:, j], a_min, a_max)
        end
        max_resid_val = rmse_nonbinding(resid_val, a_next_val, a_min, bind_tol)

        if max_resid_val < best_val_resid
            best_val_resid = max_resid_val
            best_coeffs = coeffs
            best_c = copy(c)
            best_a_next = copy(a_next)
            best_resid = euler_resid_stoch_grid(model_params, a_grid, z_grid, transition, c)
            best_max_resid = rmse_nonbinding(best_resid, best_a_next, a_min, bind_tol)
            best_iters = iters
            best_converged = converged
            best_order = order
            best_delta = last_delta
        end
    end

    B_out = B_out_cache[:, 1:(best_order+1)]
    c_out = B_out * best_coeffs
    if !is_nondec(c_out)
        for j = 1:Nz
            interp_pchip!(view(c_out, :, j), a_grid, view(best_c, :, j), a_out)
        end
    end

    a_next_out = similar(c_out)
    for j = 1:Nz
        income = exp(z_grid[j])
        @views @. a_next_out[:, j] = clamp(R * a_out + income - c_out[:, j], a_min, a_max)
    end

    resid_out = euler_resid_stoch_grid(model_params, a_out, z_grid, transition, c_out)
    max_resid_out = rmse_nonbinding(resid_out, a_next_out, a_min, bind_tol)

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol,
        tol_pol,
        maxit,
        order = best_order,
        runtime,
        seed = nothing,
        resid_metric = :rmse,
    )

    return (
        a_grid = a_out,
        z_grid = z_grid,
        c = c_out,
        a_next = a_next_out,
        resid = resid_out,
        iters = best_iters,
        converged = best_converged,
        max_resid = max_resid_out,
        rmse = max_resid_out,
        model_params = model_params,
        coeffs = best_coeffs,
        opts = opts,
        delta_pol = best_delta,
    )
end

function solve_projection_csvar(
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-6,
    maxit::Int = 1000,
    orders::AbstractVector{Int} = Int[],
    Nval::Int = model_grids[:a].N,
    λ::Real = 0.0,
    tol_pol::Real = tol,
    integration_method::Symbol = :gh,
    gh_order::Int = 3,
    nsamples::Int = 128,
    rng = nothing,
)::NamedTuple
    start_time = time_ns()
    local_rng = rng === nothing ? default_rng() : rng

    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = clamp(model_grids[:a].N, 2, MAX_COLLOCATION_POINTS)
    a_grid = gauss_lobatto_nodes(Na, a_min, a_max)
    a_out = model_grids[:a].grid
    a_val = gauss_lobatto_or_midpoint(Nval, a_min, a_max)
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = DEFAULT_BINDING_TOL)

    income = csvar_income(model_params.y)
    β = getproperty(model_params, Β_SYM)
    R = 1 + model_params.r

    candidate_orders = isempty(orders) ? Int[Na-1] : clamp_orders(orders, Na - 1)
    max_order = maximum(candidate_orders)

    B_cache = chebyshev_basis(a_grid, max_order, a_min, a_max)
    B_val_cache = chebyshev_basis(a_val, max_order, a_min, a_max)
    B_out_cache = chebyshev_basis(a_out, max_order, a_min, a_max)

    best_coeffs = nothing
    best_c = nothing
    best_a_next = nothing
    best_resid = nothing
    best_max_resid = Inf
    best_resid_val = Inf
    best_iters = 0
    best_converged = false
    best_order = candidate_orders[1]
    best_delta = nothing

    cmin = 1e-12
    available_grid = income .+ R .* a_grid .- a_min

    y_state = model_params.y

    for order in candidate_orders
        B = B_cache[:, 1:(order+1)]
        c = init_consumption_det(a_grid, a_min, R, income; cmin = cmin)
        coeffs = solve_coefficients(B, c; λ = λ)
        a_next = similar(c)
        c_next = similar(c)
        c_new = similar(c)
        converged = false
        iters = 0
        last_delta = Inf

        for it = 1:maxit
            iters = it
            @. a_next = clamp(R * a_grid + income - c, a_min, a_max)
            for idx in eachindex(c)
                a_i = a_grid[idx]
                c_i = c[idx]
                integrand = function (y_next)
                    income_next = csvar_income(y_next)
                    a_draw = clamp(R * a_i + income_next - c_i, a_min, a_max)
                    Bdraw = chebyshev_basis([a_draw], order, a_min, a_max)
                    c1 = dot(Bdraw[1, :], coeffs)
                    c1 = c1 <= cmin ? cmin : c1
                    return model_utility.u_prime(c1)
                end
                EU = integrate_expectation(
                    integration_method,
                    integrand,
                    model_params,
                    model_shocks,
                    y_state,
                    rng = local_rng,
                    gh_order = gh_order,
                    nsamples = nsamples,
                )
                c_new[idx] = model_utility.u_prime_inv(β * R * EU)
            end
            clamp_policy!(c_new, cmin, available_grid)
            coeffs = solve_coefficients(B, c_new; λ = λ)
            delta = maximum(abs.(c_new .- c))
            last_delta = delta
            c .= c_new
            if delta < tol_pol
                converged = true
                break
            end
        end

        @. a_next = clamp(R * a_grid + income - c, a_min, a_max)
        B_val = B_val_cache[:, 1:(order+1)]
        c_val = B_val * coeffs
        resid_val = Vector{Float64}(undef, length(a_val))
        for (i, a_val_i) in enumerate(a_val)
            c0 = c_val[i] <= cmin ? cmin : c_val[i]
            integrand = function (y_next)
                income_next = csvar_income(y_next)
                a_next_i = clamp(R * a_val_i + income_next - c0, a_min, a_max)
                Bnext_val = chebyshev_basis([a_next_i], order, a_min, a_max)
                c1 = dot(Bnext_val[1, :], coeffs)
                c1 = c1 <= cmin ? cmin : c1
                return model_utility.u_prime(c1)
            end
            EU = integrate_expectation(
                integration_method,
                integrand,
                model_params,
                model_shocks,
                y_state,
                rng = local_rng,
                gh_order = gh_order,
                nsamples = nsamples,
            )
            resid_val[i] = abs(1 - (β * R * EU) / model_utility.u_prime(c0))
        end
        max_resid_val = sqrt(mean(resid_val .^ 2))

        if max_resid_val < best_resid_val
            best_resid_val = max_resid_val
            best_coeffs = coeffs
            best_c = copy(c)
            best_a_next = copy(a_next)
            best_resid = resid_val
            best_max_resid = max_resid_val
            best_iters = iters
            best_converged = converged
            best_order = order
            best_delta = last_delta
        end
    end

    B_out = B_out_cache[:, 1:(best_order+1)]
    c_out = B_out * best_coeffs
    if !is_nondec(c_out)
        interp_pchip!(c_out, a_grid, best_c, a_out)
    end
    a_next_out = clamp.(R .* a_out .+ income .- c_out, a_min, a_max)
    resid_out = Vector{Float64}(undef, length(a_out))
    for (i, a_val_i) in enumerate(a_out)
        c0 = c_out[i] <= cmin ? cmin : c_out[i]
        integrand = function (y_next)
            income_next = csvar_income(y_next)
            a_next_i = clamp(R * a_val_i + income_next - c0, a_min, a_max)
            Bnext_val = chebyshev_basis([a_next_i], best_order, a_min, a_max)
            c1 = dot(Bnext_val[1, :], best_coeffs)
            c1 = c1 <= cmin ? cmin : c1
            return model_utility.u_prime(c1)
        end
        EU = integrate_expectation(
            integration_method,
            integrand,
            model_params,
            model_shocks,
            y_state,
            rng = local_rng,
            gh_order = gh_order,
            nsamples = nsamples,
        )
        resid_out[i] = abs(1 - (β * R * EU) / model_utility.u_prime(c0))
    end
    max_resid_out = sqrt(mean(resid_out .^ 2))

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol,
        tol_pol,
        maxit,
        order = best_order,
        runtime,
        seed = nothing,
        resid_metric = :rmse,
        integration_method = integration_method,
    )

    return (
        a_grid = a_out,
        c = c_out,
        a_next = a_next_out,
        resid = resid_out,
        iters = best_iters,
        converged = best_converged,
        max_resid = max_resid_out,
        rmse = max_resid_out,
        model_params = model_params,
        coeffs = best_coeffs,
        opts = opts,
        delta_pol = best_delta,
    )
end

end # module
