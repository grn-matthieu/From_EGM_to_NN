"""
CommonValidators

Lightweight configuration and argument validators shared by solver kernels.
"""
module CommonValidators

export is_nondec, is_nondec_tensor, is_positive, respects_amin

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

@inline function _leading_permutation(dim::Int, nd::Int)
    @assert 1 ≤ dim ≤ nd "dimension $(dim) must satisfy 1 ≤ dim ≤ $(nd)"
    perm = Vector{Int}(undef, nd)
    perm[1] = dim
    idx = 2
    @inbounds for d = 1:nd
        if d != dim
            perm[idx] = d
            idx += 1
        end
    end
    return Tuple(perm)
end

"""
    is_nondec_tensor(x, dims_to_check; tol=1e-12) -> Bool

Return true when `x` is weakly increasing along every dimension listed in
`dims_to_check`. Each dimension is checked by permuting it to the leading axis
and applying the matrix-based monotonicity test to the flattened slices. This
is useful when validating CSVAR policies laid out as multi-dimensional tensors.
"""
function is_nondec_tensor(x::AbstractArray, dims_to_check; tol::Real = 1e-12)
    dims_vec = collect(dims_to_check)
    isempty(dims_vec) && return true
    nd = ndims(x)
    @inbounds for dim in dims_vec
        (1 ≤ dim ≤ nd) ||
            error("dimension $(dim) out of bounds for array with $(nd) dimensions")
        perm = _leading_permutation(dim, nd)
        permuted = permutedims(x, perm)
        mat = reshape(permuted, size(x, dim), :)
        if !is_nondec(mat; tol = tol)
            return false
        end
    end
    return true
end

is_nondec_tensor(x::AbstractArray, dim::Integer; tol::Real = 1e-12) =
    is_nondec_tensor(x, (dim,); tol = tol)

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
