module CommonValidators

export is_nondec, is_positive, respects_amin

"""
    is_nondec(x::AbstractVector; tol=1e-12) -> Bool

Return true if `x` is non-decreasing within tolerance `tol`.
"""
function is_nondec(x::AbstractVector; tol::Real = 1e-12)
    n = length(x)
    @inbounds for i = 1:(n-1)
        if x[i+1] < x[i] - tol
            return false
        end
    end
    return true
end

"""
    is_nondec(x::AbstractMatrix; tol=1e-12) -> Bool

Return true if each column of `x` is non-decreasing within tolerance `tol`.
"""
function is_nondec(x::AbstractMatrix; tol::Real = 1e-12)
    nrow, ncol = size(x)
    @inbounds for j = 1:ncol
        for i = 1:(nrow-1)
            if x[i+1, j] < x[i, j] - tol
                return false
            end
        end
    end
    return true
end

"""
    is_positive(x; tol=1e-12) -> Bool

Return true if all entries of `x` are at least `tol`.
"""
is_positive(x; tol::Real = 1e-12) = all(x .>= tol)

"""
    respects_amin(x, amin; tol=1e-12) -> Bool

Return true if all entries of `x` are at least `amin - tol`.
"""
respects_amin(x, amin; tol::Real = 1e-12) = all(x .>= (amin - tol))

end # module
