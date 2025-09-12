module ProjectionCoefficients

using LinearAlgebra: AbstractVecOrMat, transpose

export solve_coefficients

"""
    solve_coefficients(B, y)

Solve for projection coefficients in the least squares sense using the
normal equations. `B` is the basis matrix and `y` the data vector or
matrix with matching rows. Returns the coefficient vector or matrix.
"""
function solve_coefficients(B::AbstractMatrix{<:Real}, y::AbstractVecOrMat{<:Real})
    @assert size(B, 1) == size(y, 1) "Incompatible dimensions"
    Bt = transpose(B)
    coeffs = (Bt * B) \ (Bt * y)
    return coeffs
end

end # module
