module Chebyshev

export chebyshev_nodes,
    chebyshev_basis, scale_to_chebyshev, scale_from_chebyshev, gauss_lobatto_nodes

"""
    scale_to_chebyshev(x, a, b)

Linearly scale `x` from the interval `[a,b]` to `[-1,1]`.
"""
@inline scale_to_chebyshev(x::Real, a::Real, b::Real) = 2 * (x - a) / (b - a) - 1

"""
    scale_from_chebyshev(ξ, a, b)

Inverse of `scale_to_chebyshev`, mapping `ξ` in `[-1,1]` back to `[a,b]`.
"""
@inline scale_from_chebyshev(ξ::Real, a::Real, b::Real) = (ξ + 1) * (b - a) / 2 + a

"""
    chebyshev_nodes(N, a, b)

Return `N` Chebyshev nodes on `[a,b]` ordered from low to high.
"""
function chebyshev_nodes(N::Integer, a::Real, b::Real)
    @assert N ≥ 1
    ξ = [cos((2k - 1) * pi / (2N)) for k = 1:N]
    nodes = scale_from_chebyshev.(ξ, a, b)
    return reverse(nodes)  # increasing order
end

"""
    gauss_lobatto_nodes(N, a, b)

Return `N` Gauss–Lobatto nodes on `[a,b]` ordered from low to high. These
correspond to the extrema of the Chebyshev polynomials and include the
interval endpoints, making them well suited for collocation methods.
"""
function gauss_lobatto_nodes(N::Integer, a::Real, b::Real)
    @assert N ≥ 2 "Gauss–Lobatto requires at least two nodes"
    ξ = [cos(pi * (k - 1) / (N - 1)) for k = 1:N]
    nodes = scale_from_chebyshev.(ξ, a, b)
    return reverse(nodes)  # increasing order
end

"""
    chebyshev_basis(x, N, a, b)

Compute Chebyshev basis up to order `N` at points `x` on `[a,b]`.
Returns a matrix where column `j+1` corresponds to `T_j`.
"""
function chebyshev_basis(x::AbstractVector{<:Real}, N::Integer, a::Real, b::Real)
    @assert N ≥ 0
    Ni = Int(N)                              # concrete range
    T = promote_type(float(eltype(x)), float(typeof(a)), float(typeof(b)))
    M = length(x)

    # map to [-1,1] in concrete T
    ξ = @. T(2) * (T(x) - T(a)) / (T(b) - T(a)) - T(1)

    B = ones(T, M, Ni + 1)
    if Ni == 0
        return B
    end

    @views B[:, 2] .= ξ
    @inbounds for n::Int = 2:Ni
        @views @. B[:, n+1] = T(2) * ξ * B[:, n] - B[:, n-1]
    end
    return B
end

end # module
