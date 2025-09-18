"""
ProjectionCoefficients

Routines to compute and work with projection coefficients for policy/value
approximations used by the projection solver.
"""
module ProjectionCoefficients

using LinearAlgebra: AbstractVecOrMat, transpose, I

export solve_coefficients

"""
    solve_coefficients(B, y; λ=0)

Solve for projection coefficients in the least squares sense using the
normal equations with optional Tikhonov regularization. `B` is the basis
matrix and `y` the data vector or matrix with matching rows. Returns the
coefficient vector or matrix. A positive `λ` adds `λ * I` to `B'B` for
stability.
"""
function solve_coefficients(
    B::AbstractMatrix{<:Real},
    y::AbstractVecOrMat{<:Real};
    λ::Real = 0,
)
    @assert size(B, 1) == size(y, 1) "Incompatible dimensions"
    Bt = transpose(B)
    coeffs = (Bt * B + λ * I) \ (Bt * y)
    return coeffs
end

end # module
