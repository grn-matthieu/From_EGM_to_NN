"""
GridHelpers

Small collection of utilities that bridge solver-produced values to the
grid backend API. These helpers assume a backend is present and are intended
to be included after the common interpolation primitives so they can reuse
the linear interpolator to project known values onto backend nodes.
"""
module GridHelpers

using ..CommonInterp: interp_linear
import ..nodes
import ..fit_interpolant!
import ..evaluate_scalar!
using Base: @views

const _GRID_PARENT = parentmodule(@__MODULE__)

export fit_values_on_backend!, eval_backend_at_points, grid_backend_available

"""
    grid_backend_available(a_info) -> Bool

Return `true` when the grid metadata tuple `a_info` exposes a valid backend
implementing `AbstractGridBackend`. Centralizes the defensive checks so solver
kernels can rely on a single guard during transitions.
"""
@inline function grid_backend_available(a_info)
    hasproperty(a_info, :backend) || return false
    backend = getproperty(a_info, :backend)
    return backend isa _GRID_PARENT.AbstractGridBackend
end


"""
    fit_values_on_backend!(a_info, x_known::AbstractVector, values::AbstractMatrix)

Fit solver-produced values (rows correspond to x_known points, columns to
variables) onto the backend attached to `a_info`. The function projects the
known values onto the backend nodes by linear interpolation and calls the
backend's `fit_interpolant!`.
"""
function fit_values_on_backend!(a_info, x_known::AbstractVector, values::AbstractMatrix)
    grid_backend_available(a_info) ||
        error("grid backend not available on supplied grid metadata")
    backend = getproperty(a_info, :backend)
    nodes_mat = nodes(backend)
    Nb = size(nodes_mat, 1)
    nvars = size(values, 2)
    vals_on_nodes = Array{Float64}(undef, Nb, nvars)
    xq = nodes_mat[:, 1]
    for j = 1:nvars
        vals_on_nodes[:, j] = interp_linear(x_known, Float64.(values[:, j]), xq)
    end
    fit_interpolant!(backend, vals_on_nodes)
    return backend
end
"""
    eval_backend_at_points(backend, xq::AbstractVector, j::Int)

Evaluate the backend-fitted interpolant (column `j`) at the query points
`xq`. Returns a vector of Float64 values.
"""
function eval_backend_at_points(backend, xq, j::Int)
    points = xq isa AbstractVector ? xq : collect(xq)
    out = similar(points, Float64)
    for (i, xval) in enumerate(points)
        coord = collect((xval,))
        out[i] = evaluate_scalar!(backend, coord, j)
    end
    return out
end

end
