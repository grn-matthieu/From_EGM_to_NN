"""
ProjectionKernel

Chebyshev-based projection solvers for deterministic and stochastic variants of
the consumption-saving model.
"""
module ProjectionKernel

using ..Chebyshev: chebyshev_basis, chebyshev_nodes
using ..ProjectionCoefficients: solve_coefficients
using ..EulerResiduals: euler_resid_det_2, euler_resid_stoch
using ..CommonInterp: interp_pchip!

export solve_projection_det, solve_projection_stoch

const BETA_SYM = Symbol(Char(0x03B2))
const PI_TRANSITION_SYM = Symbol(Char(0x03A0))

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

@inline function is_nondec(x::AbstractVector{<:Real}; tol::Real = 1e-12)
    @inbounds for i = 1:(length(x)-1)
        x[i+1] < x[i] - tol && return false
    end
    return true
end

@inline function is_nondec(x::AbstractMatrix{<:Real}; tol::Real = 1e-12)
    nrow, ncol = size(x)
    @inbounds for j = 1:ncol
        for i = 1:(nrow-1)
            x[i+1, j] < x[i, j] - tol && return false
        end
    end
    return true
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
    lambda::Real = 0.0,
)::NamedTuple
    start_time = time_ns()

    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    a_grid = chebyshev_nodes(Na, a_min, a_max)
    a_out = model_grids[:a].grid
    beta = getproperty(model_params, BETA_SYM)
    income = model_params.y
    R = 1 + model_params.r

    candidate_orders = isempty(orders) ? Int[Na-1] : orders
    max_order = maximum(candidate_orders)

    a_val = chebyshev_nodes(Nval, a_min, a_max)
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

    cmin = 1e-12

    available_grid = income .+ R .* a_grid .- a_min

    for order in candidate_orders
        B = B_cache[:, 1:(order+1)]
        c = clamp.(0.5 .* available_grid, cmin, available_grid)
        coeffs = solve_coefficients(B, c; lambda = lambda)
        a_next = similar(c)
        c_next = similar(c)
        c_new = similar(c)
        converged = false
        iters = 0

        for it = 1:maxit
            iters = it
            @. a_next = clamp(R * a_grid + income - c, a_min, a_max)
            Bnext = chebyshev_basis(a_next, order, a_min, a_max)
            c_next .= Bnext * coeffs
            @. c_new = clamp(
                model_utility.u_prime_inv(beta * R * model_utility.u_prime(c_next)),
                cmin,
                available_grid,
            )
            coeffs = solve_coefficients(B, c_new; lambda = lambda)
            delta = maximum(abs.(c_new .- c))
            c .= c_new
            if delta < tol
                converged = true
                break
            end
        end

        @. a_next = clamp(R * a_grid + income - c, a_min, a_max)
        B_val = B_val_cache[:, 1:(order+1)]
        c_val = B_val * coeffs
        resid_val = euler_resid_det_2(model_params, a_val, c_val)
        max_resid_val = maximum(resid_val[min(2, end):end])

        if max_resid_val < best_val_resid
            best_val_resid = max_resid_val
            best_coeffs = coeffs
            best_c = copy(c)
            best_a_next = copy(a_next)
            best_resid = euler_resid_det_2(model_params, a_grid, c)
            best_max_resid = maximum(best_resid[min(2, end):end])
            best_iters = iters
            best_converged = converged
            best_order = order
        end
    end

    B_out = B_out_cache[:, 1:(best_order+1)]
    c_out = B_out * best_coeffs
    if !is_nondec(c_out)
        interp_pchip!(c_out, a_grid, best_c, a_out)
    end
    a_next_out = clamp.(R .* a_out .+ income .- c_out, a_min, a_max)
    resid_out = euler_resid_det_2(model_params, a_out, c_out)
    max_resid_out = maximum(resid_out[min(2, end):end])

    runtime = (time_ns() - start_time) / 1e9
    opts = (; tol, maxit, order = best_order, runtime, seed = nothing)

    return (
        a_grid = a_out,
        c = c_out,
        a_next = a_next_out,
        resid = resid_out,
        iters = best_iters,
        converged = best_converged,
        max_resid = max_resid_out,
        coeffs = best_coeffs,
        opts = opts,
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
    lambda::Real = 0.0,
)::NamedTuple
    start_time = time_ns()

    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    a_grid = chebyshev_nodes(Na, a_min, a_max)
    a_out = model_grids[:a].grid
    a_val = chebyshev_nodes(Nval, a_min, a_max)

    z_grid = model_shocks.zgrid
    transition = getproperty(model_shocks, PI_TRANSITION_SYM)
    Nz = length(z_grid)

    beta = getproperty(model_params, BETA_SYM)
    R = 1 + model_params.r

    candidate_orders = isempty(orders) ? Int[Na-1] : orders
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

    cmin = 1e-12

    for order in candidate_orders
        B = B_cache[:, 1:(order+1)]
        c = Matrix{Float64}(undef, Na, Nz)
        for j = 1:Nz
            income = exp(z_grid[j])
            available = income .+ R .* a_grid .- a_min
            @views @. c[:, j] = clamp(0.5 * available, cmin, available)
        end

        coeffs = solve_coefficients(B, c; lambda = lambda)
        a_next = similar(c)
        c_new = similar(c)
        expected_marginal = similar(a_grid)
        tmp = similar(a_grid)
        converged = false
        iters = 0

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
                available = income .+ R .* a_grid .- a_min
                @views @. c_new[:, j] = clamp(
                    model_utility.u_prime_inv(beta * R * expected_marginal),
                    cmin,
                    available,
                )
            end

            coeffs = solve_coefficients(B, c_new; lambda = lambda)
            delta = maximum(abs.(c_new .- c))
            c .= c_new
            resid_mat = euler_resid_stoch(model_params, a_grid, z_grid, transition, c)
            max_resid = maximum(resid_mat[2:end, :])
            if delta < tol && max_resid < tol
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
        resid_val = euler_resid_stoch(model_params, a_val, z_grid, transition, c_val)
        max_resid_val = maximum(resid_val[min(2, end):end, :])

        if max_resid_val < best_val_resid
            best_val_resid = max_resid_val
            best_coeffs = coeffs
            best_c = copy(c)
            best_a_next = copy(a_next)
            best_resid = euler_resid_stoch(model_params, a_grid, z_grid, transition, c)
            best_max_resid = maximum(best_resid[min(2, end):end, :])
            best_iters = iters
            best_converged = converged
            best_order = order
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

    resid_out = euler_resid_stoch(model_params, a_out, z_grid, transition, c_out)
    max_resid_out = maximum(resid_out[min(2, end):end, :])

    runtime = (time_ns() - start_time) / 1e9
    opts = (; tol, maxit, order = best_order, runtime, seed = nothing)

    return (
        a_grid = a_out,
        c = c_out,
        a_next = a_next_out,
        resid = resid_out,
        iters = best_iters,
        converged = best_converged,
        max_resid = max_resid_out,
        coeffs = best_coeffs,
        opts = opts,
    )
end

end # module
