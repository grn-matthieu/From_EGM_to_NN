module CSVarUtils

using LinearAlgebra: Cholesky, Symmetric, cholesky, mul!
using Random: AbstractRNG, default_rng, randn, randn!

export csvar_income,
    csvar_state_matrix,
    csvar_state_income,
    csvar_state_incomes,
    csvar_cash_on_hand,
    csvar_assets_from_cash,
    csvar_next_cash_on_hand,
    csvar_covariance,
    csvar_shock_factor,
    csvar_draw_shock,
    csvar_draw_shock!,
    csvar_expected_state,
    csvar_expected_state!,
    csvar_next_state,
    csvar_next_state!,
    csvar_step

"""
    csvar_income(y)

Return current-period income implied by state `y`. For vector states the income
is the mean of the components. Batched states (one per column) are handled when
`y` is a matrix, returning the corresponding income for each column.
"""
csvar_income(y::Number) = float(y)

function csvar_income(y::AbstractVector)
    n = length(y)
    n > 0 ||
        throw(ArgumentError("csvar_income requires a non-empty state vector, got length 0"))
    return sum(y) / n
end

function csvar_income(y::AbstractMatrix)
    d = size(y, 1)
    d > 0 ||
        throw(ArgumentError("csvar_income requires at least one state dimension, got 0"))
    totals = sum(y; dims = 1)
    return vec(totals) ./ d
end

csvar_income(states::AbstractVector{<:AbstractVector}) = map(csvar_income, states)

"""
    csvar_state_matrix(y, y_dim)

Return a matrix whose columns enumerate the VAR state vectors. Accepts vectors,
matrices, or vectors of vectors and reshapes them into a `y_dim × Ny` matrix.
"""
function csvar_state_matrix(y, y_dim::Integer)
    y_dim > 0 || error("y_dim must be positive, got $y_dim")
    if y isa Number
        y_dim == 1 || error("scalar income only valid when y_dim = 1 (got $y_dim)")
        return reshape(float(y), 1, 1)
    elseif y isa AbstractVector{<:Number}
        len = length(y)
        len % y_dim == 0 ||
            error("state vector length $len must be a multiple of y_dim = $y_dim")
        cols = max(1, div(len, y_dim))
        return reshape(float.(y), y_dim, cols)
    elseif y isa AbstractMatrix
        size(y, 1) == y_dim ||
            error("state matrix must have y_dim = $y_dim rows (got $(size(y, 1)))")
        return float.(y)
    elseif y isa AbstractVector
        cols = length(y)
        cols > 0 || error("state vector collection must be non-empty")
        first_vec = y[1]
        first_vec isa AbstractVector ||
            error("state collection elements must themselves be vectors")
        length(first_vec) == y_dim || error("state vectors must have length y_dim = $y_dim")
        mat = Array{Float64}(undef, y_dim, cols)
        for (j, vec) in enumerate(y)
            vec isa AbstractVector || error("state collection elements must be vectors")
            length(vec) == y_dim ||
                error("state vector $j has length $(length(vec)); expected $y_dim")
            @views mat[:, j] .= Float64.(vec)
        end
        return mat
    else
        error("Unsupported state representation of type $(typeof(y))")
    end
end

"""
    csvar_state_income(y_state)

Compute the scalar income associated with the state vector `y_state`.
"""
csvar_state_income(y_state::AbstractVector) = sum(y_state)

"""
    csvar_state_incomes(Y)

Return the vector of incomes for each column in the state matrix `Y`.
"""
function csvar_state_incomes(Y::AbstractMatrix)
    Ny = size(Y, 2)
    incomes = Vector{Float64}(undef, Ny)
    @inbounds for j = 1:Ny
        incomes[j] = csvar_state_income(view(Y, :, j))
    end
    return incomes
end

"""
    csvar_cash_on_hand(a_prev, y, r)

Compute cash-on-hand from previous assets `a_prev`, state `y`, and interest rate
`r`, using income derived from `y`. Works elementwise when `a_prev` or `y` are
arrays.
"""
function csvar_cash_on_hand(a_prev, y, r::Real)
    return (1 + r) .* a_prev .+ csvar_income(y)
end

"""
    csvar_assets_from_cash(w, c)

Derive end-of-period assets given cash-on-hand `w` and consumption `c`.
Broadcasts over array inputs.
"""
csvar_assets_from_cash(w, c) = w .- c

"""
    csvar_next_cash_on_hand(w, c, y_next, r)

Compute next-period cash-on-hand from current cash `w`, consumption `c`,
next-period state `y_next`, and interest rate `r`.
"""
function csvar_next_cash_on_hand(w, c, y_next, r::Real)
    return (1 + r) .* csvar_assets_from_cash(w, c) .+ csvar_income(y_next)
end

"""
    csvar_covariance(ρ, d; variance=1.0)

Build the covariance matrix Σᵧ = ρ·ones(d, d) + (1-ρ)·I scaled by `variance`.
Requires `d ≥ 1`, `variance ≥ 0`, and `ρ ∈ (-1/(d-1), 1)` when `d > 1`.
"""
function csvar_covariance(ρ::Real, d::Integer; variance::Real = 1.0)
    d > 0 || throw(ArgumentError("state dimension d must be ≥ 1, got $d"))
    variance >= 0 || throw(ArgumentError("variance must be non-negative, got $variance"))
    if d > 1
        lower = -1 / (d - 1)
        (ρ > lower && ρ < 1) ||
            throw(ArgumentError("ρ must lie in ($lower, 1) for d = $d, got $ρ"))
    end
    T = float(promote_type(typeof(ρ), typeof(variance)))
    offdiag = T(ρ) * T(variance)
    Σ = fill(offdiag, d, d)
    diagval = T(variance)
    @inbounds for i = 1:d
        Σ[i, i] = diagval
    end
    return Σ
end

"""
    csvar_shock_factor(ρ, d; variance=1.0)

Return the Cholesky factor of the covariance matrix produced by
[`csvar_covariance`](@ref).
"""
function csvar_shock_factor(ρ::Real, d::Integer; variance::Real = 1.0)
    Σ = csvar_covariance(ρ, d; variance = variance)
    return cholesky(Symmetric(Σ))
end

"""
    csvar_draw_shock(rng, d; T=Float64)
    csvar_draw_shock(d; T=Float64, rng=default_rng())

Sample a standard normal shock of dimension `d`. The element type defaults to
`Float64`.
"""
function csvar_draw_shock(rng::AbstractRNG, d::Integer; T = Float64)
    d > 0 || throw(ArgumentError("shock dimension must be positive, got $d"))
    return randn(rng, T, d)
end

csvar_draw_shock(d::Integer; T = Float64, rng = default_rng()) =
    csvar_draw_shock(rng, d; T = T)

"""
    csvar_draw_shock!(rng, ε)

In-place variant that overwrites `ε` with an iid standard normal draw.
"""
function csvar_draw_shock!(rng::AbstractRNG, ε::AbstractVector)
    randn!(rng, ε)
    return ε
end

"""
    csvar_expected_state(A, y)

Return the deterministic component E[yₜ₊₁ | yₜ] = A·y.
"""
function csvar_expected_state(A::AbstractMatrix, y::AbstractVector)
    size(A, 2) == length(y) ||
        throw(DimensionMismatch("A and y have incompatible dimensions"))
    return A * y
end

"""
    csvar_expected_state!(out, A, y)

In-place version of [`csvar_expected_state`](@ref).
"""
function csvar_expected_state!(out::AbstractVector, A::AbstractMatrix, y::AbstractVector)
    size(A, 1) == length(out) ||
        throw(DimensionMismatch("output length must match rows of A"))
    size(A, 2) == length(y) ||
        throw(DimensionMismatch("A and y have incompatible dimensions"))
    mul!(out, A, y)
    return out
end

"""
    csvar_next_state(A, y, L, ε)
    csvar_next_state(A, y, chol::Cholesky, ε)

Compute yₜ₊₁ = A·y + L·ε given state `y`, transition matrix `A`, and either a
shock loading matrix `L` or its Cholesky factor.
"""
function csvar_next_state(
    A::AbstractMatrix,
    y::AbstractVector,
    L::AbstractMatrix,
    ε::AbstractVector,
)
    size(A, 2) == length(y) ||
        throw(DimensionMismatch("A and y have incompatible dimensions"))
    size(L, 2) == length(ε) ||
        throw(DimensionMismatch("shock matrix and ε have incompatible dimensions"))
    size(A, 1) == size(L, 1) ||
        throw(DimensionMismatch("A and L must have the same row dimension"))
    T = promote_type(eltype(A), eltype(y), eltype(L), eltype(ε))
    out = Vector{T}(undef, size(A, 1))
    return csvar_next_state!(out, A, y, L, ε)
end

function csvar_next_state(
    A::AbstractMatrix,
    y::AbstractVector,
    chol::Cholesky,
    ε::AbstractVector,
)
    return csvar_next_state(A, y, chol.L, ε)
end

"""
    csvar_next_state!(out, A, y, L, ε)

In-place update for yₜ₊₁ = A·y + L·ε. The caller may reuse `out` to avoid
allocations.
"""
function csvar_next_state!(
    out::AbstractVector,
    A::AbstractMatrix,
    y::AbstractVector,
    L::AbstractMatrix,
    ε::AbstractVector,
)
    size(A, 1) == length(out) ||
        throw(DimensionMismatch("output length must match rows of A"))
    size(A, 2) == length(y) ||
        throw(DimensionMismatch("A and y have incompatible dimensions"))
    size(L, 2) == length(ε) ||
        throw(DimensionMismatch("shock matrix and ε have incompatible dimensions"))
    size(L, 1) == length(out) ||
        throw(DimensionMismatch("shock matrix must share rows with output"))
    mul!(out, A, y)
    α = one(eltype(out))
    β = one(eltype(out))
    mul!(out, L, ε, α, β)
    return out
end

function csvar_next_state!(
    out::AbstractVector,
    A::AbstractMatrix,
    y::AbstractVector,
    chol::Cholesky,
    ε::AbstractVector,
)
    return csvar_next_state!(out, A, y, chol.L, ε)
end

"""
    csvar_step(rng, A, y, chol; ε_buffer=nothing)
    csvar_step(rng, A, y, ρ; variance=1.0, ε_buffer=nothing)

Sample ε ~ N(0, I) and return yₜ₊₁ = A·y + L·ε. Accepts either a precomputed
Cholesky factor `chol` or the correlation parameter `ρ`. The optional
`ε_buffer` can be provided to avoid reallocating the shock vector.
"""
function csvar_step(
    rng::AbstractRNG,
    A::AbstractMatrix,
    y::AbstractVector,
    chol::Cholesky;
    ε_buffer::Union{Nothing,AbstractVector} = nothing,
)
    d = size(A, 1)
    length(y) == size(A, 2) ||
        throw(DimensionMismatch("A and y have incompatible dimensions"))
    length(chol.L) != 0 || throw(ArgumentError("Cholesky factor cannot be empty"))
    ε =
        ε_buffer === nothing ? csvar_draw_shock(rng, d; T = eltype(chol.L)) :
        csvar_draw_shock!(rng, ε_buffer)
    return csvar_next_state(A, y, chol.L, ε)
end

function csvar_step(
    rng::AbstractRNG,
    A::AbstractMatrix,
    y::AbstractVector,
    ρ::Real;
    variance::Real = 1.0,
    ε_buffer::Union{Nothing,AbstractVector} = nothing,
)
    chol = csvar_shock_factor(ρ, size(A, 1); variance = variance)
    return csvar_step(rng, A, y, chol; ε_buffer = ε_buffer)
end

csvar_step(A::AbstractMatrix, y::AbstractVector, chol::Cholesky; rng = default_rng()) =
    csvar_step(rng, A, y, chol)

csvar_step(A::AbstractMatrix, y::AbstractVector, ρ::Real; rng = default_rng(), kwargs...) =
    csvar_step(rng, A, y, ρ; kwargs...)

end # module
