"""
Mixed precision helpers for the NN kernel.

Centralises every conversion to `Float32` (and back) so that the main kernel
remains focused on the training logic.
"""

import CUDA
import Adapt
using CUDA: cu
using Statistics: mean
using ..CSVarUtils: csvar_component_log_means

# -- Generic helpers ---------------------------------------------------------

float32_vector(x) = Vector{Float32}(collect(x))
float32_matrix(x) = Array{Float32}(collect(x))
float32_loss(x) = Float32(x)

"""Prepare the input batch for Lux by ensuring `Float32` features.

This helper centralises the choice of CPU vs GPU representation. Call sites
should use `prepare_training_batch(X, Val(settings.use_cuda))` so that the
conversion policy is explicit and reliable.
"""
prepare_training_batch(X, ::Val{false}) = Array{Float32}(permutedims(X))
prepare_training_batch(X, ::Val{true}) = Adapt.adapt(CUDA.CuArray, permutedims(X))

# -- Device / precision helpers -------------------------------------------

"""Move an object to the compute device requested by `settings`.

If `settings.use_cuda` is true the value is adapted to `CUDA.CuArray` using
`Adapt.adapt`, otherwise the value is returned unchanged. This keeps logic
around device placement in a single place.
"""
maybe_to_device(x::Nothing, ::Any) = nothing
maybe_to_device(x, settings) =
    (isdefined(settings, :use_cuda) && settings.use_cuda) ? Adapt.adapt(CUDA.CuArray, x) : x

"""Bring an object back to host (Array) when `settings.use_cuda` is true.

This uses `Adapt.adapt(Array, ...)` to allow converting CuArrays back to CPU
arrays without copying when unnecessary.
"""
maybe_to_host(x::Nothing, ::Any) = nothing
maybe_to_host(x, settings) =
    (isdefined(settings, :use_cuda) && settings.use_cuda) ? Adapt.adapt(Array, x) : x

"""Alias: explicit CPU/GPU helpers for clarity in call sites."""
maybe_to_cpu(x, settings) = maybe_to_host(x, settings)
maybe_to_gpu(x, settings) = maybe_to_device(x, settings)

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
Return the `Float32` asset grid and predicted consumption (vectorised)
used in Euler/grid residual evaluation on the asset grid.
"""
function grid_residual_inputs(c_predicted, G)
    cp = extract_consumption(c_predicted)
    c_vec = vec(permutedims(cp))
    return float32_vector(G[:a].grid), c_vec, float32_vector(c_vec)
end

"""Map grid residuals to a `Float32` loss value."""
grid_loss(resid) = float32_loss(sum(resid))

"""
Return the `Float32` grids and predicted consumption matrix used for the
stochastic residual evaluation.
"""
function stoch_residual_inputs(c_predicted, G, S)
    a_grid_f32 = float32_vector(G[:a].grid)
    z_grid_f32 = float32_vector(S.zgrid)
    Pz_f32 = float32_matrix(S.Π)
    Na, Nz = length(a_grid_f32), length(z_grid_f32)

    # Extract and reshape consumption predictions
    vec_cp = vec(permutedims(extract_consumption(c_predicted)))
    len = length(vec_cp)
    total_needed = Na * Nz

    c_mat = if len == total_needed
        reshape(vec_cp, Na, Nz)
    elseif len == Na
        repeat(reshape(vec_cp, Na, 1), 1, Nz)  # Tile across z
    elseif len == Nz
        repeat(reshape(vec_cp, 1, Nz), Na, 1)  # Tile across assets
    else
        throw(
            DimensionMismatch(
                "stoch_residual_inputs expected length Na*Nz=$total_needed, Na=$Na, or Nz=$Nz; got $len",
            ),
        )
    end

    return a_grid_f32, z_grid_f32, Pz_f32, c_mat, float32_matrix(c_mat)
end

"""Map stochastic residuals to a `Float32` loss value."""
stoch_loss(resid) = float32_loss(sum(abs2, resid))

"""Return the `(y, w)` feature grid (and cash-on-hand) for grid-based evaluation passes."""
function grid_forward_inputs(G, P_full)
    a_grid_f32 = float32_vector(G[:a].grid)
    Rg = 1.0f0 + Float32(P_full.r)

    # Model type detection: CSVAR vs AR(1)
    is_csvar = isdefined(P_full, :A) && isdefined(P_full, :Σ)

    component_rows, mean_income = if is_csvar
        log_means = Float32.(csvar_component_log_means(P_full))
        log_means, sum(exp.(log_means))
    else
        y_raw =
            isdefined(P_full, :y) && P_full.y isa AbstractVector ?
            Float32.(collect(P_full.y)) : Float32[Float32(P_full.y)]
        y_raw, mean(exp.(y_raw))
    end

    # Build feature matrix
    extra_cols = is_csvar || length(component_rows) > 1 ? length(component_rows) : 0
    feature_dim = 2 + extra_cols
    w_grid = @. Rg * a_grid_f32 + Float32(mean_income)

    X = Matrix{Float32}(undef, feature_dim, length(a_grid_f32))
    X[1, :] .= Float32(mean_income)
    for j = 1:extra_cols
        X[1+j, :] .= component_rows[j]
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
