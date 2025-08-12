module EGMResiduals
export euler_residuals_simple

using ..ThesisProject: SimpleModel

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

end # module