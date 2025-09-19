"""
ProjectionCoefficients

Helper for solving normal equations with optional Tikhonov regularisation.
"""
module ProjectionCoefficients

using LinearAlgebra

export solve_coefficients

"""
    solve_coefficients(B, y; lambda=0)

Solve the least-squares problem `min ||B * θ - y||` using the normal equations.
When `lambda > 0`, Tikhonov regularisation is applied: `(B'B + λI) θ = B'y`.
"""
function solve_coefficients(
    B::AbstractMatrix{<:Real},
    y::AbstractVecOrMat{<:Real};
    kwargs...,
)
    λ = get(kwargs, :lambda, 0.0)
    λ = get(kwargs, Symbol("λ"), λ)
    @assert size(B, 1) == size(y, 1) "Incompatible dimensions between basis and data"
    Bt = transpose(B)
    return (Bt * B + λ * I) \ (Bt * y)
end

end # module
