module EGMSolver
export solve_simple_egm, SimpleSolution

using ..SimpleModel

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
        # guard bounds in case of FP edge cases
        j = j < 2 ? 2 : (j > N ? N : j)
        x0 = xgrid[j-1]; x1 = xgrid[j]
        y0 = y[j-1];     y1 = y[j]
        t = (x - x0) / (x1 - x0)
        return (1 - t) * y0 + t * y1
    end
end

# ─── Types ───────────────────────────────────────────────────────────────────

Base.@kwdef struct SimpleSolution
    agrid::Vector{Float64}
    c::Vector{Float64}
    a_next::Vector{Float64}
    iters::Int
    converged::Bool
end

# ─── Solver ──────────────────────────────────────────────────────────────────

"""
    solve_simple_egm(p, agrid; tol=1e-8, maxit=500, verbose=false)

Fixed-point/Euler mapping for the simple (deterministic) savings model.
Uses c_t(a) = [β(1+r)]^(1/σ) * c_{t+1}(a′) with linear interpolation at a′.
Enforces feasibility and mild monotonicity; relaxation improves stability on coarse grids.
"""
function solve_simple_egm(p, agrid; tol=1e-8, maxit=500, verbose=false)
    Na = length(agrid)
    agrid = collect(agrid)  # ensure concrete Vector{Float64}

    # Feasible initial guess (avoid zeros)
    resources = p.y .+ (1 + p.r) .* agrid .- p.a_min
    c = clamp.(0.5 .* resources, 1e-10, resources)

    a_next = similar(c)
    γ = (p.β * (1 + p.r))^(1 / p.σ)         # Euler factor
    c_floor = 1e-10
    λ = 0.5                                  # relaxation

    converged = false
    iters = 0

    for it in 1:maxit
        iters = it
        c_new = similar(c)

        @inbounds for i in eachindex(agrid)
            # Next assets from previous policy
            a′ = SimpleModel.budget_next(agrid[i], p.y, p.r, c[i])

            # Interpolate c_{t+1} at clamped a′
            a′q = clamp(a′, p.a_min, p.a_max)
            c_next = lininterp(agrid, c, a′q)

            # Euler-implied current c
            cti  = c_next / γ
            cmax = p.y + (1 + p.r) * agrid[i] - p.a_min
            c_new[i] = clamp(cti, c_floor, cmax)

            # record implied next assets (clamped)
            a_next[i] = clamp(SimpleModel.budget_next(agrid[i], p.y, p.r, c_new[i]), p.a_min, p.a_max)
        end

        # Enforce monotonicity of c(a) to suppress wiggles on coarse grids
        @inbounds for i in 2:Na
            if c_new[i] < c_new[i-1]
                c_new[i] = c_new[i-1] + 1e-12
            end
        end

        # Convergence check BEFORE relaxation
        maxdiff = maximum(abs.(c_new .- c))

        # Relaxation update
        c .= (1 - λ) .* c .+ λ .* c_new

        if verbose && (it % 10 == 0)
            @info "iter=$it maxdiff=$(maxdiff)"
        end
        if maxdiff < tol
            converged = true
            break
        end
    end

    return SimpleSolution(agrid, c, a_next, iters, converged)
end

end # module
