"""
CommonInterp

Interpolation primitives shared by solvers. Provides simple linear and
monotone piecewise-cubic (PCHIP) routines together with type tags consumed by
call sites.
"""
module CommonInterp

export InterpKind, LinearInterp, MonotoneCubicInterp
export interp_linear!, interp_linear, interp_pchip!

abstract type InterpKind end
struct LinearInterp <: InterpKind end
struct MonotoneCubicInterp <: InterpKind end

Base.@propagate_inbounds function _interp_linear_scalar(
    u::Real,
    x::AbstractVector,
    y::AbstractVector,
    n::Int,
)
    if u <= x[1]
        return y[1]
    elseif u >= x[n]
        return y[n]
    else
        j = searchsortedfirst(x, u)
        j = clamp(j, 2, n)
        x0 = x[j-1]
        x1 = x[j]
        y0 = y[j-1]
        y1 = y[j]
        t = (u - x0) / (x1 - x0)
        return (1 - t) * y0 + t * y1
    end
end

"""
    interp_linear!(out, x, y, xq)

Linear interpolation of points `(x, y)` at query locations `xq`. Results are
written to `out`.
"""
function interp_linear!(
    out::AbstractVector,
    x::AbstractVector,
    y::AbstractVector,
    xq::AbstractVector,
)
    @assert length(x) == length(y)
    n = length(x)
    @inbounds for k in eachindex(xq)
        out[k] = _interp_linear_scalar(xq[k], x, y, n)
    end
    return out
end

"""
    interp_linear(x, y, xq)

Linear interpolation of points `(x, y)` at query locations `xq`. Accepts a scalar
or vector `xq` and returns the interpolated value(s).
"""
function interp_linear(x::AbstractVector, y::AbstractVector, xq::AbstractVector)
    @assert length(x) == length(y)
    T = promote_type(eltype(y), eltype(xq))
    out = similar(xq, T)
    interp_linear!(out, x, y, xq)
    return out
end

function interp_linear(x::AbstractVector, y::AbstractVector, xq::Real)
    @assert length(x) == length(y)
    return _interp_linear_scalar(xq, x, y, length(x))
end



"""
    interp_pchip!(out, x, y, xq)

Piecewise cubic Hermite interpolation (PCHIP) for monotone data.
The interpolant preserves shape around kinks and avoids overshoot.
"""
function interp_pchip!(
    out::AbstractVector,
    x::AbstractVector,
    y::AbstractVector,
    xq::AbstractVector,
)
    @assert length(x) == length(y)
    @assert all(diff(x) .> 0) "x must be strictly increasing"
    @assert all(diff(y) .>= 0) "PCHIP requires non-decreasing data"

    n = length(x)
    h = Vector{Float64}(undef, n - 1)
    delta = Vector{Float64}(undef, n - 1)
    slopes = Vector{Float64}(undef, n)

    @inbounds for i = 1:(n-1)
        h[i] = x[i+1] - x[i]
        delta[i] = (y[i+1] - y[i]) / h[i]
    end

    slopes[1] = delta[1]
    slopes[n] = delta[end]

    @inbounds for i = 2:(n-1)
        if delta[i-1] == 0 || delta[i] == 0
            slopes[i] = 0.0
        elseif delta[i-1] * delta[i] < 0
            slopes[i] = 0.0
        else
            w1 = 2 * h[i] + h[i-1]
            w2 = h[i] + 2 * h[i-1]
            slopes[i] = (w1 + w2) / (w1 / delta[i-1] + w2 / delta[i])
        end
    end

    @inbounds for k in eachindex(xq)
        u = xq[k]
        if u <= x[1]
            out[k] = y[1]
            continue
        elseif u >= x[end]
            out[k] = y[end]
            continue
        end

        j = searchsortedfirst(x, u)
        i = j - 1
        hi = h[i]
        t = (u - x[i]) / hi
        t2 = t * t
        t3 = t2 * t

        h00 = 2t3 - 3t2 + 1
        h10 = t3 - 2t2 + t
        h01 = -2t3 + 3t2
        h11 = t3 - t2

        out[k] = h00 * y[i] + h10 * hi * slopes[i] + h01 * y[i+1] + h11 * hi * slopes[i+1]
    end

    return out
end

end # module
