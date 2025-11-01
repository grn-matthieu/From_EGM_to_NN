module CSVARParamGrid

"""Utility builders for CSVAR parameter sweeps.

The helpers here return full configuration NamedTuples that only touch the
`params` section, leaving solver definitions untouched. They can be used to
generate grids over dimensionality, correlation structure, and orthogonal state
rotations when combined with the base configs introduced for the experiments.
"""

using LinearAlgebra

include("config_helpers.jl")
using .ScriptConfigHelpers: dict_to_namedtuple, get_nested, merge_section, maybe_namedtuple

export build_dimensional_overrides, build_correlation_overrides, build_rotation_overrides

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

@inline function matrix_to_rows(M::AbstractMatrix)
    rows = Vector{Vector{Float64}}(undef, size(M, 1))
    for i = 1:size(M, 1)
        rows[i] = collect(Float64.(M[i, :]))
    end
    return rows
end

function rows_to_matrix(rows)
    n = length(rows)
    n == 0 && return Matrix{Float64}(undef, 0, 0)
    m = length(rows[1])
    M = Matrix{Float64}(undef, n, m)
    for (i, row) in enumerate(rows)
        M[i, :] .= Float64.(row)
    end
    return M
end

function base_params(config)
    base_nt = dict_to_namedtuple(config)
    params = get_nested(base_nt, (:params,), nothing)
    params === nothing && error("Base configuration is missing a params section")
    return base_nt, maybe_namedtuple(params)
end

function diagonal_system(d::Int; persistence::Float64, variance::Float64)
    A = persistence .* Matrix{Float64}(I, d, d)
    Σ = variance .* Matrix{Float64}(I, d, d)
    return A, Σ
end

function ensure_positive_targets(d::Int, total_income::Float64)
    level = total_income / d
    level > 0 || error("Total income must be positive")
    return fill(level, d)
end

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

"""
    build_dimensional_overrides(config; dims = [2, 3, 4], total_income = 1.0,
                                persistence = 0.7, variance = 0.04)

Generate a dictionary of configs with diagonal VAR structures of increasing
dimension. Each entry updates only the `params` section of `config`, leaving the
solver definitions untouched.
"""
function build_dimensional_overrides(
    config;
    dims::AbstractVector{<:Integer} = [2, 3, 4],
    total_income::Float64 = 1.0,
    persistence::Float64 = 0.7,
    variance::Float64 = 0.04,
)
    base_nt = dict_to_namedtuple(config)
    overrides = Dict{Symbol,NamedTuple}()
    for d in dims
        d > 0 || error("State dimension must be positive")
        y_levels = ensure_positive_targets(d, total_income)
        A, Σ = diagonal_system(d; persistence = persistence, variance = variance)
        params_override =
            (y = Float64.(y_levels), A = matrix_to_rows(A), Σ = matrix_to_rows(Σ))
        key = Symbol("diag_", d, "d")
        overrides[key] = merge_section(base_nt, :params, params_override)
    end
    return overrides
end

"""
    build_correlation_overrides(config; persistence = 0.7, variance = 0.04,
                                corr_strength = 0.15, toeplitz_decay = 0.5,
                                dense_scale = 0.12)

Return a dictionary of configs for different correlation structures using the
dimension implied by `config.params.y`.
"""
function build_correlation_overrides(
    config;
    persistence::Float64 = 0.7,
    variance::Float64 = 0.04,
    corr_strength::Float64 = 0.15,
    toeplitz_decay::Float64 = 0.5,
    dense_scale::Float64 = 0.12,
)
    base_nt, params = base_params(config)
    y_vals = Vector{Float64}(params.y)
    d = length(y_vals)
    d > 0 || error("Base configuration must specify at least one income component")

    A_diag, Σ_diag = diagonal_system(d; persistence = persistence, variance = variance)
    overrides = Dict{Symbol,NamedTuple}()

    if d >= 2
        A_single = copy(A_diag)
        Σ_single = copy(Σ_diag)
        A_single[1, 2] = corr_strength
        A_single[2, 1] = corr_strength
        Σ_single[1, 2] = variance * corr_strength
        Σ_single[2, 1] = variance * corr_strength
        overrides[:single_offdiag] = merge_section(
            base_nt,
            :params,
            (y = y_vals, A = matrix_to_rows(A_single), Σ = matrix_to_rows(Σ_single)),
        )
    end

    A_toe = Matrix{Float64}(undef, d, d)
    Σ_toe = Matrix{Float64}(undef, d, d)
    for i = 1:d, j = 1:d
        if i == j
            A_toe[i, j] = persistence
            Σ_toe[i, j] = variance
        else
            decay = toeplitz_decay^(abs(i - j))
            A_toe[i, j] = persistence * corr_strength * decay
            Σ_toe[i, j] = variance * decay
        end
    end
    overrides[:toeplitz] = merge_section(
        base_nt,
        :params,
        (y = y_vals, A = matrix_to_rows(A_toe), Σ = matrix_to_rows(Σ_toe)),
    )

    off_val = persistence * dense_scale
    A_dense = fill(off_val, d, d)
    Σ_dense = fill(variance * dense_scale, d, d)
    for i = 1:d
        A_dense[i, i] = persistence
        Σ_dense[i, i] = variance
    end
    overrides[:dense] = merge_section(
        base_nt,
        :params,
        (y = y_vals, A = matrix_to_rows(A_dense), Σ = matrix_to_rows(Σ_dense)),
    )

    return overrides
end

"""
    build_rotation_overrides(config; include = (:pca, :cholesky,))

Construct orthogonal (PCA) and whitening (Cholesky) rotations of the state
vector. Returns a dictionary mapping rotation labels to full configs that only
adjust the `params` block.
"""
function build_rotation_overrides(config; include = (:pca, :cholesky))
    base_nt, params = base_params(config)
    y_vals = Vector{Float64}(params.y)
    A = rows_to_matrix(params.A)
    Σ = rows_to_matrix(params.Σ)
    size(A, 1) == size(Σ, 1) || error("A and Σ must have consistent dimensions")

    overrides = Dict{Symbol,NamedTuple}()

    if :pca in include
        eig = eigen(Symmetric(Σ))
        Q = eig.vectors
        A_rot = transpose(Q) * A * Q
        Σ_rot = transpose(Q) * Σ * Q
        overrides[:pca] = merge_section(
            base_nt,
            :params,
            (y = y_vals, A = matrix_to_rows(A_rot), Σ = matrix_to_rows(Σ_rot)),
        )
    end

    if :cholesky in include
        chol = cholesky(Symmetric(Σ), check = false)
        L = Matrix{Float64}(chol.L)
        R = inv(L)
        A_rot = R * A * L
        Σ_rot = R * Σ * transpose(R)
        overrides[:cholesky] = merge_section(
            base_nt,
            :params,
            (y = y_vals, A = matrix_to_rows(A_rot), Σ = matrix_to_rows(Σ_rot)),
        )
    end

    return overrides
end

end # module
