module EGMSolver
export solve_simple_egm, SimpleSolution

using ..SimpleModel
using ..EGMResiduals: euler_residuals_simple


# ─── Types ───────────────────────────────────────────────────────────────────
abstract type InterpKind end
struct LinearInterp <: InterpKind end # Default interpolation
struct MonotoneCubicInterp <: InterpKind end

"""
    SimpleModelParameters

"""

Base.@kwdef struct SimpleSolution
    agrid::Vector{Float64}
    c::Vector{Float64}
    a_next::Vector{Float64}
    iters::Int
    converged::Bool
    max_residual::Float64
end


# ─── Vectorized interpolation helpers ────────────────────────────────────────────────────────────────

function interp_linear!(out::AbstractVector(<:Real),
                        x::AbstractVector(<:Real),
                        y::AbstractVector(<:Real),
                        xq::AbstractVector{<:Real})
    """
    Performs linear interpolation on the input vectors.
    """
    N = length(x)
    @inbounds for k in eachindex(xq)
        ξ = xq[k]
        if ξ <= y[1]
            out[k] = y[1]
        elseif ξ >= x[end]
            out[k] = y[end]
        else
            j = searchsortedfirst(x, ξ)
            x0 = x[j-1]; x1 = x[j]
            y0 = y[j-1]; y1 = y[j]
            t = (ξ - x0) / (x1 - x0)
            out[k] = (1 - t) * y0 + t * y1 # Straight line interpolation
        end
    end
    return out
end

function pchip_slopes(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    """
    Computes the slopes for piecewise cubic Hermite interpolation (PCHIP).
    """
    n = length(x)
    d = similar(y, Float64)
    Δ = similar(y, Float64); h = similar(y, Float64)
    @inbounds for i in 1:n-1
        h[i]  = x[i+1] - x[i]
        Δ[i]  = (y[i+1] - y[i]) / h[i]
    end
    # interior slopes
    d[1] = Δ[1]
    d[n] = Δ[n-1]
    @inbounds for i in 2:n-1
        if (Δ[i-1] ≤ 0 && Δ[i] ≤ 0) || (Δ[i-1] ≥ 0 && Δ[i] ≥ 0) == false
            d[i] = 0.0
        else
            w1 = 2h[i] + h[i-1]
            w2 = h[i] + 2h[i-1]
            d[i] = (w1 + w2) / (w1/Δ[i-1] + w2/Δ[i])
        end
    end
    return d, h, Δ
end


# Monotone cubic (PCHIP) evaluation; vectorized over xq
function interp_pchip!(out::AbstractVector{<:Real},
                       x::AbstractVector{<:Real},
                       y::AbstractVector{<:Real},
                       xq::AbstractVector{<:Real})
    """
    Monotone cubic (PCHIP) interpolation.
    """
    n = length(x)
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
        t  = (ξ - x[i]) / hi
        t2 = t*t
        t3 = t2*t
        h00 =  2t3 - 3t2 + 1
        h10 =    t3 - 2t2 + t
        h01 = -2t3 + 3t2
        h11 =    t3 -   t2
        out[k] = h00*y[i] + h10*hi*d[i] + h01*y[i+1] + h11*hi*d[i+1]
    end
    return out
end

# ─── Solver ──────────────────────────────────────────────────────────────────
function solve_simple_egm(p, agrid;
        tol::Real=1e-8, maxit::Int=500, verbose::Bool=false,
        interp_kind::InterpKind=LinearInterp(), relax::Real=0.5)
    """
    Vectorized EGM with residual-based stopping. No recurring income; borrowing limit at `p.a_min`.
    """

    a = collect(agrid)
    Na = length(a)

    γ = (p.β * (1 + p.r))^(1 / p.σ)
    onepr = (1 + p.r)
    cmin  = 1e-12

    # Initial guess for resources and consumption
    resources = @. onepr * a - p.a_min
    c = clamp.(0.5 .* resources, cmin, resources)

    # Buffer variables
    a′ = similar(c)
    cnext = similar(c)
    cnew = similar(c)
    anext = similar(c)

    converged = false
    iters = 0
    max_resid = Inf

    for it in 1:maxit
        iters = it

        @. a′ = onepr * a - c
        @. a′ = clamp(a′, p.a_min, p.a_max)

        if interp_kind isa LinearInterp
            interp_linear!(cnext, a, c, a′)
        else
            interp_pchip!(cnext, a, c, a′)
        end

        @. cnew = cnext / γ
        cmax = @. onepr * a - p.a_min
        @. cnew = clamp(cnew, cmin, cmax)

        @. anext = onepr * a - cnew
        @. anext = clamp(anext, p.a_min, p.a_max)
        
        # Check on monotonicity
        @inbounds for i in 2:Na
            if cnew[i] < cnew[i-1]
                cnew[i] = cnew[i-1] + 1e-12
            end
        end
        
        # Relaxation to stabilize on coarse grids
        c .= (1 - relax) .* c .+ relax .* cnew

        # Residual based stopping criteria : only stop when the max residual is below the tolerance
        res = euler_residuals_simple(p, a, c)
        max_resid = maximum(res[min(2, end):end])  # Ignore where the BC is binding so EE may hold as an inequality
        if verbose && (it % 10 == 0)
            @info "iter=$it max_resid=$(max_resid)"
        end
        if max_resid < tol
            converged = true
            break
        end
    end

    # Last a next consistent with c
    @. anext = onepr * a - c
    @. anext = clamp(anext, p.a_min, p.a_max)

    return SimpleSolution(a, c, anext, iters, converged, max_resid)
end

end # module