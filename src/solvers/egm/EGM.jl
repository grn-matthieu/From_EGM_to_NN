module EGMSolver
export solve_simple_egm, SimpleSolution

using ..SimpleModel
using ..EGMResiduals: euler_residuals_simple

# ─── Utilities ────────────────────────────────────────────────────────────────

"Robust linear interpolation on a sorted grid; clamps out-of-bounds queries."
@inline function lininterp(xgrid::AbstractVector{<:Real}, y::AbstractVector{<:Real}, x::Real)
    N = length(xgrid)
    if x <= xgrid[1]
        return y[1]
    elseif x >= xgrid[end]
        return y[end]
    else
        j = searchsortedfirst(xgrid, x)
        j = j < 2 ? 2 : (j > N ? N : j) # guard
        x0, x1 = xgrid[j-1], xgrid[j]
        y0, y1 = y[j-1], y[j]
        t = (x - x0) / (x1 - x0)
        return (1 - t) * y0 + t * y1
    end
end

"""
    envelope_condition(c_next, p)

Euler condition mapping: 
c_t = (β(1+r))^(-1/σ) * c_{t+1}.
"""
@inline function envelope_condition(c_next::Real, p)
    γ = (p.β * (1 + p.r))^(1 / p.σ)
    return c_next / γ
end

# ─── Types ───────────────────────────────────────────────────────────────────

Base.@kwdef struct SimpleSolution
    agrid::Vector{Float64}
    c::Vector{Float64}
    a_next::Vector{Float64}
    iters::Int
    converged::Bool
    max_residual::Float64
end

# ─── Solver ──────────────────────────────────────────────────────────────────

"""
    solve_simple_egm(p, agrid; tol=1e-8, maxit=500, verbose=false)

EGM solver for the simple savings model. 
Stops when Euler residuals are below tolerance (outside borrowing corner).
"""
function solve_simple_egm(p, agrid; tol=1e-8, maxit=500, verbose=false)
    Na = length(agrid)
    agrid = collect(agrid)

    # Initial guess
    resources = p.y .+ (1 + p.r) .* agrid .- p.a_min
    c = clamp.(0.5 .* resources, 1e-10, resources)

    a_next = similar(c)
    c_floor = 1e-10
    λ = 0.5
    converged = false
    iters = 0
    max_resid = Inf

    for it in 1:maxit
        iters = it
        c_new = similar(c)

        @inbounds for i in eachindex(agrid)
            a′ = SimpleModel.budget_next(agrid[i], p.y, p.r, c[i])
            a′q = clamp(a′, p.a_min, p.a_max)
            c_next = lininterp(agrid, c, a′q)
            cti = envelope_condition(c_next, p)
            cmax = p.y + (1 + p.r) * agrid[i] - p.a_min
            c_new[i] = clamp(cti, c_floor, cmax)
            a_next[i] = clamp(SimpleModel.budget_next(agrid[i], p.y, p.r, c_new[i]),
                              p.a_min, p.a_max)
        end

        # Monotonicity
        @inbounds for i in 2:Na
            if c_new[i] < c_new[i-1]
                c_new[i] = c_new[i-1] + 1e-12
            end
        end

        # Relaxation
        c .= (1 - λ) .* c .+ λ .* c_new

        # Residual-based convergence check
        resids = euler_residuals_simple(p, agrid, c)
        max_resid = maximum(resids[2:end])  # ignore borrowing corner (index 1)

        if verbose && (it % 10 == 0)
            @info "iter=$it max_resid=$(max_resid)"
        end
        if max_resid < tol
            converged = true
            break
        end
    end

    return SimpleSolution(agrid, c, a_next, iters, converged, max_resid)
end

end # module
