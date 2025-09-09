module CommonInterp

export interp_linear!, interp_pchip!, InterpKind, LinearInterp, MonotoneCubicInterp


# --- Structs ---

abstract type InterpKind end
struct LinearInterp <: InterpKind end
struct MonotoneCubicInterp <: InterpKind end

# --- Functions ---

"""
    interp_linear!(out::AbstractArray,
                  x::AbstractVector,
                  y::AbstractVector,
                  xq::AbstractVector)
    Performs linear interpolation on (x,y) at query points xq.
    Output is written to `out`.
"""
function interp_linear!(
    out::AbstractArray,
    x::AbstractVector,
    y::AbstractVector,
    xq::AbstractVector,
)
    N = length(x)
    @inbounds for k in eachindex(xq)
        ξ = xq[k]
        if ξ <= x[1]
            out[k] = y[1]
        elseif ξ >= x[end]
            out[k] = y[end]
        else
            j = searchsortedfirst(x, ξ)
            j = clamp(j, 2, N)  # guard
            x0 = x[j-1];
            x1 = x[j]
            y0 = y[j-1];
            y1 = y[j]
            t = (ξ - x0) / (x1 - x0)
            out[k] = (1 - t) * y0 + t * y1
        end
    end
    return out
end


"""
    function pchip_slopes(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    Computes the slopes for piecewise cubic Hermite interpolation (PCHIP).
    Outputs:
        d  : slopes at each x
        h  : differences in x
        Δ  : secant slopes between points
"""
function pchip_slopes(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})

    n = length(x)
    d = similar(y, Float64)
    Δ = similar(y, Float64);
    h = similar(y, Float64)
    @inbounds for i = 1:(n-1)
        h[i] = x[i+1] - x[i]
        Δ[i] = (y[i+1] - y[i]) / h[i]
    end
    # interior slopes
    d[1] = Δ[1]
    d[n] = Δ[n-1]
    @inbounds for i = 2:(n-1)
        if !((Δ[i-1] ≤ 0 && Δ[i] ≤ 0) || (Δ[i-1] ≥ 0 && Δ[i] ≥ 0))
            d[i] = 0.0
        else
            w1 = 2h[i] + h[i-1]
            w2 = h[i] + 2h[i-1]
            d[i] = (w1 + w2) / (w1/Δ[i-1] + w2/Δ[i])
        end
    end
    return d, h, Δ
end



"""
    function interp_pchip!(out::AbstractVector{<:Real},
                       x::AbstractVector{<:Real},
                       y::AbstractVector{<:Real},
                       xq::AbstractVector{<:Real})
    Performs piecewise cubic Hermite interpolation (PCHIP) on the input data.
    Output : Interpolated values at query points xq.
"""
function interp_pchip!(
    out::AbstractVector{<:Real},
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    xq::AbstractVector{<:Real},
)
    @assert all(diff(x) .> 0)
    @assert all(diff(y) .≥ 0) "Monotone cubic requires non-decreasing data"
    # Checks on conditions for PCHIP to avoid overshoot around kinks
    d, h, _ = pchip_slopes(x, y)
    @inbounds for k in eachindex(xq)
        ξ = xq[k]
        if ξ <= x[1]
            out[k] = y[1]
            continue
        elseif ξ >= x[end]
            out[k] = y[end]
            continue
        end
        j = searchsortedfirst(x, ξ)
        i = j-1
        hi = h[i]
        t = (ξ - x[i]) / hi
        t2 = t*t
        t3 = t2*t
        h00 = 2t3 - 3t2 + 1
        h10 = t3 - 2t2 + t
        h01 = -2t3 + 3t2
        h11 = t3 - t2
        out[k] = h00*y[i] + h10*hi*d[i] + h01*y[i+1] + h11*hi*d[i+1]
    end
    return out
end


end #module
