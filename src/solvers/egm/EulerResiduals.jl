module EGMResiduals
export euler_residuals_simple

using ..SimpleModel

@inline function lininterp(xgrid::AbstractVector{<:Real}, y::AbstractVector{<:Real}, x::Real)
    N = length(xgrid)
    if x <= xgrid[1]
        return y[1]
    elseif x >= xgrid[end]
        return y[end]
    else
        j = searchsortedfirst(xgrid, x)
        x0 = xgrid[j-1]; x1 = xgrid[j]
        y0 = y[j-1];     y1 = y[j]
        t = (x - x0) / (x1 - x0)
        return (1 - t) * y0 + t * y1
    end
end

"""
Residual R(a) = | 1 - β(1+r) * u'(c_{t+1}(a')) / u'(c_t(a)) |,
with a' = (1+r)a + y - c_t(a) and c_{t+1} evaluated by linear interpolation.
"""
function euler_residuals_simple(p, agrid::AbstractVector{<:Real}, c::AbstractVector{<:Real})
    @assert length(agrid) == length(c)
    resid = similar(c, Float64)
    γβr = p.β * (1 + p.r)

    @inbounds for i in eachindex(agrid)
        ct  = max(c[i], 1e-12)  # avoid zero
        a′  = SimpleModel.budget_next(agrid[i], p.y, p.r, ct)
        ct1 = max(lininterp(agrid, c, a′), 1e-12)
        upr_t  = ct^(-p.σ)
        upr_t1 = ct1^(-p.σ)
        resid[i] = abs(1 - γβr * (upr_t1 / upr_t))
    end
    return resid
end

end # module
