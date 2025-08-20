module EGMResiduals
export euler_residuals_simple, euler_residuals_stochastic

using ..ThesisProject: SimpleModel
using LinearAlgebra


"""
    lininterp(x::AbstractVector, y::AbstractVector, ξ::Real)

Return linear interpolation of y over grid x at point ξ.
If ξ is outside the grid, extrapolate using endpoints.
"""
function lininterp(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, ξ::Real)
    n = length(x)
    if ξ <= x[1]
        return y[1]
    elseif ξ >= x[end]
        return y[end]
    else
        j = searchsortedfirst(x, ξ)
        x0, x1 = x[j-1], x[j]
        y0, y1 = y[j-1], y[j]
        t = (ξ - x0) / (x1 - x0)
        return (1 - t) * y0 + t * y1
    end
end



"""
    euler_residuals_simple(p, agrid, c)

Compute absolute Euler equation residuals for the simple savings model.

Residual: r(a) = 1 - β(1+r) * (u'(c_{t+1}) / u'(c_t))

Here, c_{t+1} is evaluated at the asset choice implied by policy `c` today,
using nearest neighbor interpolation for simplicity.
"""
function euler_residuals_simple(p, agrid::AbstractVector{<:Real}, c::AbstractVector{<:Real})
    @assert length(agrid) == length(c) "agrid and c must have same length"
    Na = length(agrid)
    resid = similar(c, Float64)
    for i in eachindex(agrid)
        ct = c[i]
        a_next = SimpleModel.budget_next(agrid[i], p.y, p.r, ct)
        # nearest index (no fancy interpolation yet)
        j = clamp(searchsortedfirst(agrid, a_next), 1, Na)
        ct1 = c[j]
        upr_t  = ct^(-p.σ)
        upr_t1 = ct1^(-p.σ)
        resid[i] = abs(1 - p.β*(1+p.r) * (upr_t1 / upr_t))
    end
    return resid
end

function euler_residuals_stochastic(p, agrid, zgrid, Pz, c::AbstractMatrix{<:Real})
    Na, Nz = size(c)
    resid = zeros(Na, Nz)
    onepr = 1 + p.r
    γβr = p.β * onepr

    @inbounds for j in 1:Nz
        y = exp(zgrid[j])
        for i in 1:Na
            ct = max(c[i,j], 1e-12)
            a′ = onepr * agrid[i] + y - ct
            a′ = clamp(a′, agrid[1], agrid[end])

            # Interpolate c_{t+1} at a′ for ALL future shock states at once
            c_future = similar(agrid, Nz)
            for jp in 1:Nz
                c_future[jp] = max(lininterp(agrid, view(c,:,jp), a′), 1e-12)
            end

            # Expected marginal utility = dot product of transition probs with u'(c_{t+1})
            EUprime = dot(Pz[j, :], c_future.^(-p.σ))

            resid[i,j] = abs(1 - γβr * (EUprime / ct^(-p.σ)))
        end
    end
    return resid
end

end # module