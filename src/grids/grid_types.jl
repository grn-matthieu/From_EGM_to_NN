# Shared grid backend abstractions.

abstract type AbstractGridBackend end

function nodes(::AbstractGridBackend)
    error("nodes not implemented for this grid backend")
end

function bounds(::AbstractGridBackend)
    error("bounds not implemented for this grid backend")
end

function fit_interpolant!(::AbstractGridBackend, ::AbstractMatrix)
    error("fit_interpolant! not implemented for this grid backend")
end

function evaluate_scalar!(::AbstractGridBackend, ::AbstractVector, ::Int)
    error("evaluate_scalar! not implemented for this grid backend")
end

function refine_once!(::AbstractGridBackend; tol::Float64 = 0.0)
    error("refine_once! not implemented for this grid backend")
end
