module ConsumerSavingVAR

using LinearAlgebra: Diagonal
using ..ConsumerSaving: AbstractModel, _build_asset_grid, _build_crra_utility

import ..API: get_params, get_grids, get_shocks, get_utility

export ConsumerSavingVARModel, build_cs_var_model

struct ConsumerSavingVARModel <: AbstractModel
    params::NamedTuple
    grids::NamedTuple
    shocks::NamedTuple
    utility::NamedTuple
end

function _to_float_vector(x; name = "vector")
    if x isa AbstractVector
        return Float32.(x)
    elseif x isa Real
        return [Float32(x)]
    else
        error("$(name) must be a vector or scalar numeric value")
    end
end

function _to_float_matrix(x; name = "matrix")
    if x isa AbstractMatrix
        return Matrix{Float32}(x)
    elseif x isa AbstractVector
        rows = length(x)
        rows > 0 || error("$(name) must have at least one row")
        first_row = x[1]
        first_row isa AbstractVector || error("$(name) rows must themselves be vectors")
        cols = length(first_row)
        cols > 0 || error("$(name) must have at least one column")
        mat = Array{Float64}(undef, rows, cols)
        for (i, row) in enumerate(x)
            row isa AbstractVector || error("$(name) rows must be vectors")
            length(row) == cols || error("$(name) rows must have equal length")
            mat[i, :] .= Float64.(row)
        end
        return mat
    else
        error("$(name) must be a matrix or a vector of equal-length vectors")
    end
end

function build_cs_var_model(cfg::NamedTuple)
    params_cfg = cfg.params
    A = _to_float_matrix(params_cfg.A; name = "params.A")
    Σ = _to_float_matrix(params_cfg.Σ; name = "params.Σ")

    size(A, 1) == size(A, 2) || error("params.A must be square")
    size(Σ, 1) == size(A, 1) || error("params.Σ row dimension must match params.A")

    y_vec = _to_float_vector(params_cfg.y; name = "params.y")
    length(y_vec) == size(A, 1) || error("params.y length must match dimension of params.A")

    grids = _build_asset_grid(cfg.grids)

    params = merge(params_cfg, (A = A, Σ = Σ, y = y_vec, y_dim = size(A, 1)))

    shocks = (
        process = :gaussian_linear,
        ε_dim = size(Σ, 2),
        covariance = Diagonal(fill(1.0, size(Σ, 2))),
        Σ = Σ,
    )

    utility = _build_crra_utility(params)

    return ConsumerSavingVARModel(params, grids, shocks, utility)
end

get_params(model::ConsumerSavingVARModel) = model.params
get_grids(model::ConsumerSavingVARModel) = model.grids
get_shocks(model::ConsumerSavingVARModel) = model.shocks
get_utility(model::ConsumerSavingVARModel) = model.utility

end # module ConsumerSavingVAR
