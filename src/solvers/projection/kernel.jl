module ProjectionKernel

using LinearAlgebra: mul!, cond
using ..Chebyshev: chebyshev_basis, chebyshev_nodes
using ..ProjectionCoefficients: solve_coefficients
using ..EulerResiduals: euler_resid_det_2, euler_resid_stoch

export solve_projection_det, solve_projection_stoch

"""
    solve_projection_det(
        model_params,
        model_grids,
        model_utility;
        tol=1e-6,
        maxit=1000,
        orders=Int[],
        Nval=model_grids[:a].N,
        λ=0.0,
    )

Deterministic projection solver for the consumption-saving model. Approximates
consumption with Chebyshev polynomials and enforces Euler equation residuals at
asset grid points. For each candidate polynomial order in `orders`, coefficients
are estimated on the training grid and the Euler residual is evaluated on a
separate Chebyshev validation grid of size `Nval`. The order with the smallest
validation residual is returned along with diagnostics.
"""
function solve_projection_det(
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-6,
    maxit::Int = 1000,
    orders::AbstractVector{Int} = Int[],
    Nval::Int = model_grids[:a].N,
    λ::Real = 0.0,
)::NamedTuple
    start_time = time_ns()

    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    a_grid = chebyshev_nodes(Na, a_min, a_max)

    β = model_params.β
    R = 1 + model_params.r
    y = model_params.y

    orders = isempty(orders) ? Int[Na-1] : orders

    best_coeffs = nothing
    best_c = nothing
    best_a_next = nothing
    best_resid = nothing
    best_max_resid = Inf
    best_val_resid = Inf
    best_iters = 0
    best_converged = false
    best_order = orders[1]

    for order in orders
        B = chebyshev_basis(a_grid, order, a_min, a_max)

        cmin = 1e-12
        cmax = @. y + R * a_grid - a_min
        c = @. clamp(0.5 * (y + R * a_grid - a_min), cmin, cmax)

        coeffs = solve_coefficients(B, c; λ = λ)
        a_next = similar(c)
        c_next = similar(c)
        c_new = similar(c)

        converged = false
        iters = 0

        for it = 1:maxit
            iters = it
            @. a_next = R * a_grid + y - c
            @. a_next = clamp(a_next, a_min, a_max)

            Bnext = chebyshev_basis(a_next, order, a_min, a_max)
            c_next .= Bnext * coeffs

            @. c_new = model_utility.u_prime_inv(β * R * model_utility.u_prime(c_next))
            @. c_new = clamp(c_new, cmin, cmax)

            coeffs_new = solve_coefficients(B, c_new; λ = λ)
            Δ = maximum(abs.(c_new - c))
            coeffs = coeffs_new
            c .= c_new
            if Δ < tol
                converged = true
                break
            end
        end

        @. a_next = R * a_grid + y - c
        @. a_next = clamp(a_next, a_min, a_max)

        a_val = chebyshev_nodes(Nval, a_min, a_max)
        B_val = chebyshev_basis(a_val, order, a_min, a_max)
        c_val = B_val * coeffs
        resid_val = euler_resid_det_2(model_params, a_val, c_val)
        max_resid_val = maximum(resid_val[min(2, end):end])

        if max_resid_val < best_val_resid
            best_val_resid = max_resid_val
            best_coeffs = coeffs
            best_c = c
            best_a_next = a_next
            resid_train = euler_resid_det_2(model_params, a_grid, c)
            best_resid = resid_train
            best_max_resid = maximum(resid_train[min(2, end):end])
            best_iters = iters
            best_converged = converged
            best_order = order
        end
    end

    runtime = (time_ns() - start_time) / 1e9
    opts =
        (; tol = tol, maxit = maxit, order = best_order, runtime = runtime, seed = nothing)

    return (;
        a_grid,
        c = best_c,
        a_next = best_a_next,
        resid = best_resid,
        iters = best_iters,
        converged = best_converged,
        max_resid = best_max_resid,
        coeffs = best_coeffs,
        model_params,
        opts,
    )
end

"""
    solve_projection_stoch(
        model_params,
        model_grids,
        model_shocks,
        model_utility;
        tol=1e-6,
        maxit=1000,
        orders=Int[],
        Nval=model_grids[:a].N,
        λ=0.0,
    )

Stochastic projection solver for the consumption-saving model with discrete income shocks.
Approximates consumption for each shock with Chebyshev polynomials and enforces Euler
equation residuals on the asset grid. For each candidate polynomial order the Euler
residual is also evaluated on a separate Chebyshev validation grid of size `Nval`, and
the order with the smallest out-of-sample residual is selected.
"""
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
)::NamedTuple
    start_time = time_ns()

    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    a_grid = chebyshev_nodes(Na, a_min, a_max)

    z_grid = model_shocks.zgrid
    Π = model_shocks.Π
    Nz = length(z_grid)

    β = model_params.β
    R = 1 + model_params.r

    orders = isempty(orders) ? Int[Na-1] : orders

    best_coeffs = nothing
    best_c = nothing
    best_a_next = nothing
    best_resid = nothing
    best_max_resid = Inf
    best_val_resid = Inf
    best_iters = 0
    best_converged = false
    best_order = orders[1]

    for order in orders
        B = chebyshev_basis(a_grid, order, a_min, a_max)

        c = Matrix{Float64}(undef, Na, Nz)
        cmin = 1e-12
        for j = 1:Nz
            y = exp(z_grid[j])
            cmax = @. y + R * a_grid - a_min
            @. c[:, j] = clamp(0.5 * (y + R * a_grid - a_min), cmin, cmax)
        end

        coeffs = solve_coefficients(B, c; λ = λ)
        a_next = similar(c)
        c_new = similar(c)
        Emu = similar(a_grid)
        ctmp = similar(a_grid)

        converged = false
        iters = 0

        for it = 1:maxit
            iters = it

            for j = 1:Nz
                y = exp(z_grid[j])
                @views @. a_next[:, j] = R * a_grid + y - c[:, j]
                @views @. a_next[:, j] = clamp(a_next[:, j], a_min, a_max)

                fill!(Emu, 0.0)
                Bnext = chebyshev_basis(view(a_next, :, j), order, a_min, a_max)
                for jp = 1:Nz
                    ctmp .= Bnext * view(coeffs, :, jp)
                    @. Emu += Π[j, jp] * model_utility.u_prime(ctmp)
                end
                cmax = @. exp(z_grid[j]) + R * a_grid - a_min
                @views @. c_new[:, j] = model_utility.u_prime_inv(β * R * Emu)
                @views @. c_new[:, j] = clamp(c_new[:, j], cmin, cmax)
            end

            coeffs_new = solve_coefficients(B, c_new; λ = λ)
            Δ = maximum(abs.(c_new .- c))
            coeffs = coeffs_new
            c .= c_new
            resid_mat = euler_resid_stoch(model_params, a_grid, z_grid, Π, c_new)
            max_resid = maximum(resid_mat[2:end, :])
            if Δ < tol && max_resid < tol
                converged = true
                break
            end
        end

        for j = 1:Nz
            y = exp(z_grid[j])
            @views @. a_next[:, j] = R * a_grid + y - c[:, j]
            @views @. a_next[:, j] = clamp(a_next[:, j], a_min, a_max)
        end

        a_val = chebyshev_nodes(Nval, a_min, a_max)
        B_val = chebyshev_basis(a_val, order, a_min, a_max)
        c_val = B_val * coeffs
        resid_val = euler_resid_stoch(model_params, a_val, z_grid, Π, c_val)
        max_resid_val = maximum(resid_val[min(2, end):end, :])

        if max_resid_val < best_val_resid
            best_val_resid = max_resid_val
            best_coeffs = coeffs
            best_c = c
            best_a_next = a_next
            resid_train = euler_resid_stoch(model_params, a_grid, z_grid, Π, c)
            best_resid = resid_train
            best_max_resid = maximum(resid_train[min(2, end):end, :])
            best_iters = iters
            best_converged = converged
            best_order = order
        end
    end

    runtime = (time_ns() - start_time) / 1e9
    opts =
        (; tol = tol, maxit = maxit, order = best_order, runtime = runtime, seed = nothing)

    return (;
        a_grid,
        z_grid,
        c = best_c,
        a_next = best_a_next,
        resid = best_resid,
        iters = best_iters,
        converged = best_converged,
        max_resid = best_max_resid,
        coeffs = best_coeffs,
        model_params,
        opts,
    )
end

end # module
