"""
ProjectionCoefficients

Levenberg-Marquardt solver for nonlinear least-squares problems in projection methods.
Minimizes ||B * θ - y|| with adaptive damping and line search for robustness.
"""
module ProjectionCoefficients

using LinearAlgebra

export solve_coefficients, solve_coefficients_lm

"""
    solve_coefficients(B, y; λ=0)

Solve the least-squares problem `min ||B * θ - y||` using the normal equations.
When `λ > 0`, Tikhonov regularisation is applied: `(B'B + λI) θ = B'y`.

This is the simple direct solver, kept for compatibility. For iterative refinement
with adaptive damping, use `solve_coefficients_lm`.
"""
function solve_coefficients(
    B::AbstractMatrix{<:Real},
    y::AbstractVecOrMat{<:Real};
    kwargs...,
)
    λ = get(kwargs, :λ, 0.0)
    λ = get(kwargs, Symbol("λ"), λ)
    @assert size(B, 1) == size(y, 1) "Incompatible dimensions between basis and data"
    Bt = transpose(B)
    return (Bt * B + λ * I) \ (Bt * y)
end

"""
    solve_coefficients_lm(B, y, θ_init; λ_init=1e-3, tol=1e-8, maxit=50, verbose=false)

Solve the least-squares problem `min ||B * θ - y||` using Levenberg-Marquardt algorithm
with line search. This provides more robust convergence for difficult problems.

# Arguments
- `B`: Basis matrix (m × n) evaluated at collocation points
- `y`: Target values (m-vector)
- `θ_init`: Initial guess for coefficients (n-vector)
- `λ_init`: Initial damping parameter (default: 1e-3)
- `tol`: Convergence tolerance on residual norm (default: 1e-8)
- `maxit`: Maximum iterations (default: 50)
- `verbose`: Print iteration info (default: false)

# Returns
- `θ`: Solution coefficients
- `converged`: Boolean indicating convergence
- `iters`: Number of iterations performed
- `final_residual`: Final residual norm
"""
function solve_coefficients_lm(
    B::AbstractMatrix{<:Real},
    y::AbstractVector{<:Real},
    θ_init::AbstractVector{<:Real};
    λ_init::Real = 1e-3,
    tol::Real = 1e-8,
    maxit::Int = 50,
    verbose::Bool = false,
)
    @assert size(B, 1) == length(y) "Incompatible dimensions between basis and target"
    @assert size(B, 2) == length(θ_init) "Incompatible dimensions for initial coefficients"

    m, n = size(B)
    θ = copy(θ_init)
    λ = λ_init

    # Compute initial residual
    r = B * θ - y
    cost = 0.5 * dot(r, r)

    converged = false
    iters = 0

    # Precompute B'B for efficiency
    BtB = B' * B

    for iter = 1:maxit
        iters = iter

        # Check convergence
        residual_norm = norm(r)
        if residual_norm < tol
            converged = true
            if verbose
                println("LM converged at iteration $iter: residual = $residual_norm")
            end
            break
        end

        # Compute gradient: g = B'r
        g = B' * r

        # Levenberg-Marquardt step: solve (B'B + λI)δ = -B'r
        δ = -(BtB + λ * I) \ g

        # Line search with backtracking
        α = 1.0
        θ_new = θ + α * δ
        r_new = B * θ_new - y
        cost_new = 0.5 * dot(r_new, r_new)

        max_linesearch = 10
        linesearch_iter = 0
        while cost_new > cost && linesearch_iter < max_linesearch
            α *= 0.5
            θ_new = θ + α * δ
            r_new = B * θ_new - y
            cost_new = 0.5 * dot(r_new, r_new)
            linesearch_iter += 1
        end

        # Update damping parameter based on progress
        ρ = (cost - cost_new) / max(abs(cost_new - cost), 1e-12)

        if cost_new < cost
            # Accept step
            θ = θ_new
            r = r_new
            cost = cost_new

            # Decrease damping (trust region expansion)
            λ = max(λ / 3.0, 1e-12)

            if verbose && (iter % 10 == 0 || iter == 1)
                println("Iter $iter: cost = $cost, λ = $λ, α = $α, ρ = $ρ")
            end
        else
            # Reject step, increase damping
            λ = min(λ * 10.0, 1e6)

            if verbose
                println("Iter $iter: step rejected, increasing λ to $λ")
            end
        end

        # Safety check for damping explosion
        if λ > 1e6
            if verbose
                println("Damping parameter too large, stopping at iteration $iter")
            end
            break
        end
    end

    final_residual = norm(r)

    return θ, converged, iters, final_residual
end

end # module
