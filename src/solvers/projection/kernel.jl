module ProjectionKernel

using LinearAlgebra: mul!, cond
using ..Chebyshev: chebyshev_basis, chebyshev_nodes
using ..ProjectionCoefficients: solve_coefficients
using ..EulerResiduals: euler_resid_det_2, euler_resid_stoch

export solve_projection_det, solve_projection_stoch

"""
    solve_projection_det(model_params, model_grids, model_utility; tol=1e-6, maxit=1000)

Deterministic projection solver for the consumption-saving model. Approximates
consumption with Chebyshev polynomials and enforces Euler equation residuals at
asset grid points. Returns a NamedTuple containing policy and diagnostics.
"""
function solve_projection_det(
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-6,
    maxit::Int = 1000,
)::NamedTuple
    start_time = time_ns()

    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    a_grid = chebyshev_nodes(Na, a_min, a_max)

    β = model_params.β
    R = 1 + model_params.r
    y = model_params.y

    order = Na - 1
    B = chebyshev_basis(a_grid, order, a_min, a_max)

    cmin = 1e-12
    cmax = @. y + R * a_grid - a_min
    c = @. clamp(0.5 * (y + R * a_grid - a_min), cmin, cmax)

    coeffs = solve_coefficients(B, c)
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

        coeffs_new = solve_coefficients(B, c_new)
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

    resid = euler_resid_det_2(model_params, a_grid, c)
    max_resid = maximum(resid[min(2, end):end])

    runtime = (time_ns() - start_time) / 1e9
    opts = (; tol = tol, maxit = maxit, order = order, runtime = runtime, seed = nothing)

    return (; a_grid, c, a_next, resid, iters, converged, max_resid, model_params, opts)
end

"""
    solve_projection_stoch(model_params, model_grids, model_shocks, model_utility; tol=1e-6, maxit=1000)

Stochastic projection solver for the consumption-saving model with discrete income shocks.
Approximates consumption for each shock with Chebyshev polynomials and enforces Euler
equation residuals on the asset grid. Returns a NamedTuple containing policy and diagnostics.
"""
function solve_projection_stoch(
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-6,
    maxit::Int = 1000,
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

    order = Na - 1
    B = chebyshev_basis(a_grid, order, a_min, a_max)

    c = Matrix{Float64}(undef, Na, Nz)
    cmin = 1e-12
    for j = 1:Nz
        y = exp(z_grid[j])
        cmax = @. y + R * a_grid - a_min
        @. c[:, j] = clamp(0.5 * (y + R * a_grid - a_min), cmin, cmax)
    end

    coeffs = solve_coefficients(B, c)
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

        coeffs_new = solve_coefficients(B, c_new)
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

    resid = euler_resid_stoch(model_params, a_grid, z_grid, Π, c)
    max_resid = maximum(resid[min(2, end):end, :])

    runtime = (time_ns() - start_time) / 1e9
    opts = (; tol = tol, maxit = maxit, order = order, runtime = runtime, seed = nothing)

    return (;
        a_grid,
        z_grid,
        c,
        a_next,
        resid,
        iters,
        converged,
        max_resid,
        model_params,
        opts,
    )
end

end # module
