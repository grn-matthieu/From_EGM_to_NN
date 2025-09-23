"""
Mixed precision helpers for the NN kernel.

Centralises every conversion to `Float32` (and back) so that the main kernel
remains focused on the training logic.
"""

# -- Generic helpers ---------------------------------------------------------

float32_vector(x) = Vector{Float32}(collect(x))
float32_matrix(x) = Array{Float32}(collect(x))
float32_loss(x) = Float32(x)

"""Prepare the input batch for Lux by ensuring `Float32` features."""
prepare_training_batch(X) = Array{Float32}(permutedims(X))

"""
Return the `Float32` asset grid and predicted consumption (vectorised) used in
Euler residual evaluation for the deterministic problem.
"""
function det_residual_inputs(c_predicted, G)
    c_vec = vec(permutedims(c_predicted))
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
    c_mat = reshape(vec(permutedims(c_predicted)), Na, Nz)
    return a_grid_f32, z_grid_f32, Pz_f32, c_mat, float32_matrix(c_mat)
end

"""Map stochastic residuals to a `Float32` loss value."""
stoch_loss(resid) = float32_loss(sum(abs2, resid))

"""Return the `Float32` batch used for deterministic forward passes."""
function det_forward_inputs(G)
    a_grid_f32 = float32_vector(G[:a].grid)
    return reshape(a_grid_f32, 1, :), a_grid_f32
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
