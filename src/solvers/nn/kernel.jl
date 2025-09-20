"""
NNKernel

Stub neural-network solver kernel. For now it implements a trivial 'solver' that
returns a simple policy (half of resources) and synthetic residuals. This file
is a placeholder to be replaced by a real MLP-based solver using Flux/Zygote.
"""
module NNKernel

using ..CommonInterp: InterpKind, LinearInterp

export solve_nn_det, solve_nn_stoch

function _build_default_policy(a_grid)
    R = 1.0
    y = 1.0
    resources = @. R * a_grid + y
    return clamp.(0.5 .* resources, 1e-12, resources)
end

function solve_nn_det(model_params, model_grids, model_utility; opts = nothing)
    a_grid = model_grids[:a].grid
    c = _build_default_policy(a_grid)
    a_next = @. model_params.y + (1 + model_params.r) * a_grid - c
    resid = fill(0.0, length(a_grid))
    iters = 0
    converged = true
    max_resid = maximum(abs.(resid))
    runtime = 0.0
    opts_out = (; epochs = opts[:epochs], seed = nothing, runtime = runtime)
    return (;
        a_grid = a_grid,
        c = c,
        a_next = a_next,
        resid = resid,
        iters = iters,
        converged = converged,
        max_resid = max_resid,
        model_params = model_params,
        opts = opts_out,
    )
end

function solve_nn_stoch(
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    opts = nothing,
)
    a_grid = model_grids[:a].grid
    z_grid = model_shocks.zgrid
    Nz = length(z_grid)
    Na = length(a_grid)
    c = Array{Float64}(undef, Na, Nz)
    for j = 1:Nz
        c[:, j] .= _build_default_policy(a_grid)
    end
    a_next = @. (1 + model_params.r) * a_grid + exp(z_grid[1]) - c[:, 1]
    resid = zeros(Na, Nz)
    iters = 0
    converged = true
    max_resid = maximum(abs.(resid))
    runtime = 0.0
    opts_out = (; epochs = opts[:epochs], seed = nothing, runtime = runtime)
    return (;
        a_grid = a_grid,
        z_grid = z_grid,
        c = c,
        a_next = a_next,
        resid = resid,
        iters = iters,
        converged = converged,
        max_resid = max_resid,
        model_params = model_params,
        opts = opts_out,
    )
end

end # module
