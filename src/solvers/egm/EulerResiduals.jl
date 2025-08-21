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

# Stochastic Euler residuals:
# res[i,j] = 1 - β(1+r) * Σ_{j'} Π[j,j'] * (c'(a',j') / c(a,j))^{-σ}
function euler_residuals_stochastic(p, agrid, zgrid, Π, c)
    Na, Nz = size(c)
    res = similar(c)
    β = p.β; r = p.r; σ = p.σ
    @inbounds for j in 1:Nz
        y = exp(zgrid[j])
        for i in 1:Na
            c_ij = c[i,j]
            ap = (1 + r) * agrid[i] + y - c_ij
            Emu = 0.0
            for jp in 1:Nz
                cp = lininterp(agrid, view(c, :, jp), ap)
                Emu += Π[j, jp] * (cp / c_ij)^(-σ)
            end
            res[i,j] = 1.0 - β * (1 + r) * Emu
        end
    end
    return res
end
end # module