"""
Mixed precision helpers for the NN kernel.

Centralises every conversion to `Float32` (and back) so that the main kernel
remains focused on the training logic.
"""

using CUDA: cu
using Statistics: mean
using ..CSVarUtils: csvar_component_log_means

# -- Generic helpers ---------------------------------------------------------

float32_vector(x) = Vector{Float32}(collect(x))
float32_matrix(x) = Array{Float32}(collect(x))
float32_loss(x) = Float32(x)

"""Prepare the input batch for Lux by ensuring `Float32` features."""
prepare_training_batch(X, ::Val{false}) = Array{Float32}(permutedims(X))
prepare_training_batch(X, ::Val{true}) = cu(permutedims(X))  # X est déjà Float32

# Recursively extract the consumption prediction from various model output
# shapes. Models (or Lux) sometimes return `(y, state)` tuples and our new
# dual-head model returns a `NamedTuple(Φ=..., h=...)`. This helper returns
# the array/matrix that represents predicted consumption.
function extract_consumption(pred)
    if pred isa NamedTuple
        return haskey(pred, :Φ) ? pred[:Φ] : first(values(pred))
    elseif pred isa Tuple
        return extract_consumption(pred[1])
    else
        return pred
    end
end

"""
Return the `Float32` asset grid and predicted consumption (vectorised) used in
Euler residual evaluation for the deterministic problem.
"""
function det_residual_inputs(c_predicted, G)
    cp = extract_consumption(c_predicted)
    c_vec = vec(permutedims(cp))
    return float32_vector(G[:a].grid), c_vec, float32_vector(c_vec)
end

"""Map deterministic residuals to a `Float32` loss value."""
det_loss(resid) = float32_loss(sum(resid))

"""
Return the `Float32` grids and predicted consumption matrix used for the
stochastic residual evaluation.
"""
function stoch_residual_inputs(c_predicted, G, S)
    a_grid_f32 = float32_vector(G[:a].grid)
    z_grid_f32 = float32_vector(S.zgrid)
    Pz_f32 = float32_matrix(S.Π)
    Na = length(a_grid_f32)
    Nz = length(z_grid_f32)
    # Extract the actual consumption array (handles NamedTuple and Tuple)
    cp = extract_consumption(c_predicted)
    vec_cp = vec(permutedims(cp))
    total_needed = Na * Nz
    if length(vec_cp) == total_needed
        c_mat = reshape(vec_cp, Na, Nz)
    elseif length(vec_cp) == Na
        # model returned one consumption per asset (no z variation) -> tile across z
        c_mat = reshape(vec_cp, Na, 1)
        c_mat = repeat(c_mat, 1, Nz)
    elseif length(vec_cp) == Nz
        # model returned one consumption per shock state -> tile across assets
        c_mat = reshape(vec_cp, 1, Nz)
        c_mat = repeat(c_mat, Na, 1)
    else
        throw(
            DimensionMismatch(
                "stoch_residual_inputs expected a prediction of length Na*Nz=$(total_needed) or Na=$(Na) or Nz=$(Nz), got length $(length(vec_cp))",
            ),
        )
    end

    return a_grid_f32, z_grid_f32, Pz_f32, c_mat, float32_matrix(c_mat)
end

"""Map stochastic residuals to a `Float32` loss value."""
stoch_loss(resid) = float32_loss(sum(abs2, resid))

"""Return the `(y, w)` feature grid (and cash-on-hand) for deterministic passes."""
function det_forward_inputs(G, P_full)
    a_grid_f32 = float32_vector(G[:a].grid)
    Rg = 1.0f0 + Float32(P_full.r)
    # Interpret income depending on model type:
    # - AR(1) log-income: `y` is a log level -> use exp(y)
    # - CSVAR: `y` components are levels already -> no exp
    is_csvar = hasproperty(P_full, :A) && hasproperty(P_full, :Σ)
    if is_csvar
        log_means = Float32.(csvar_component_log_means(P_full))
        mean_income = sum(exp.(log_means))
        component_rows = log_means
    else
        y_raw =
            hasproperty(P_full, :y) && P_full.y isa AbstractVector ?
            Float32.(collect(P_full.y)) : Float32[Float32(getfield(P_full, :y))]
        component_rows = y_raw
        mean_income = mean(exp.(y_raw))  # deterministic AR(1) path uses exp(y)
    end
    extra_cols =
        is_csvar ? length(component_rows) :
        (length(component_rows) > 1 ? length(component_rows) : 0)
    feature_dim = 1 + extra_cols + 1
    w_grid = @. Rg * a_grid_f32 + Float32(mean_income)
    X = Matrix{Float32}(undef, feature_dim, length(a_grid_f32))
    X[1, :] .= Float32(mean_income)
    if extra_cols > 0
        for j = 1:extra_cols
            X[1+j, :] .= component_rows[j]
        end
    end
    X[end, :] .= w_grid
    return X, w_grid
end

"""
Convert predicted consumption back to the original grid element type.
"""
function convert_to_grid_eltype(grid, values)
    T = eltype(grid)
    if T <: Integer
        # round down to integer grid elements
        return floor.(T, values)
    else
        return convert.(T, values)
    end
end
