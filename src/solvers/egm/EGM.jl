module EGMSolver
export solve_simple_egm, SimpleSolution, solve_stochastic_egm, asset_simulation

using ..SimpleModel
using ..EGMResiduals: euler_residuals_simple, euler_residuals_stochastic
using ..SimpleCalibration


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
    p::SimpleParams
end


# ─── Vectorized interpolation helpers ────────────────────────────────────────────────────────────────

function interp_linear!(out::AbstractArray,
                        x::AbstractVector,
                        y::AbstractVector,
                        xq::AbstractVector)
    """
    Performs linear interpolation on (x,y) at query points xq.
    Works with y as a vector or as a column view from a matrix.
    """
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
            x0 = x[j-1]; x1 = x[j]
            y0 = y[j-1]; y1 = y[j]
            t = (ξ - x0) / (x1 - x0)
            out[k] = (1 - t) * y0 + t * y1
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

# ─────────────────────────────────────────────────────────────────────────────




# ────── Asset Simulation ─────────────────────────────────────────────────────

function simulate_assets(agrid, a_next; T=length(agrid), a0=agrid[1])
    a_sim = similar(agrid, T)
    a_sim[1] = a0
    for t in 2:T
        idx = searchsortedfirst(agrid, a_sim[t-1])
        idx = clamp(idx, 1, length(agrid))
        a_sim[t] = a_next[idx]
    end
    return a_sim
end

# ─────────────────────────────────────────────────────────────────────────────





# ─── Solver ──────────────────────────────────────────────────────────────────
function solve_simple_egm(p, agrid;
        tol::Real=1e-8, tol_pol::Real=1e-6, maxit::Int=500, verbose::Bool=false,
        interp_kind::InterpKind=LinearInterp(), relax::Real=0.5, patience::Int=50, ν::Real=1e-10,
        c_init=nothing)
    """
    Vectorized EGM with residual-based stopping. No recurring income; borrowing limit at `p.a_min`.
    """

    a = collect(agrid)
    a_min = minimum(a)
    a_max = maximum(a)
    Na = length(a)

    γ = p.β * (1 + p.r)
    R = (1 + p.r)
    cmin  = 1e-12

    # Initial guess for resources and consumption
    resources = @. R * a - a_min + p.y
    c = c_init === nothing ? clamp.(0.5 .* resources, cmin, resources) : copy(c_init)

    # Buffer variables
    a′ = similar(c)
    cnext = similar(c)
    cnew = similar(c)
    a_next = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    Δpol = Inf
    best_resid = Inf
    no_progress = 0

    for it in 1:maxit
        iters = it

        @. a′ = p.y + R * a - c
        @. a′ = clamp(a′, a_min, a_max)

        if interp_kind isa LinearInterp
            interp_linear!(cnext, a, c, a′)
        else
            interp_pchip!(cnext, a, c, a′)
        end

        @. cnew = inv_uprime(γ * cnext.^(-p.σ), p.σ)
        cmax = @. p.y + R * a - a_min
        @. cnew = clamp(cnew, cmin, cmax)

        @. a_next = R * a + p.y - cnew
        @. a_next = clamp(a_next, a_min, a_max)

        # Only enforce monotonicity if using PCHIP
        if interp_kind isa MonotoneCubicInterp
            @inbounds for i in 2:Na
                if cnew[i] < cnew[i-1]
                    cnew[i] = cnew[i-1] + 1e-12
                end
            end
        end
        # Relaxation to stabilize on coarse grids
        c .= (1 - relax) .* c .+ relax .* cnew

        # Residual based stopping criteria : only stop when the max residual is below the tolerance
        res = euler_residuals_simple(p, a, c)
        max_resid = maximum(res[min(2, end):end])  # Ignore where the BC is binding so EE may hold as an inequality
        Δpol = maximum(abs.(c - cnew))
        if verbose && (it % 10 == 0)
            @info "iter=$it max_resid=$(max_resid) max_Δpol=$(Δpol)"
        end

        # --- Stagnation check ---
        if (best_resid - max_resid < ν) && (Δpol < ν)
            no_progress += 1
        else
            no_progress = 0
            best_resid = max_resid
        end

        if no_progress ≥ patience && verbose
            @warn "Stopped early: no progress in Euler errors or policy for $patience iterations, iter : $iters"
            break
        end


        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    # Last a next consistent with c
    @. a_next = R * a + p.y - c
    @. a_next = clamp(a_next, a_min, a_max)

    return SimpleSolution(a, c, a_next, iters, converged, max_resid, p)
end


function solve_stochastic_egm(p, agrid, zgrid, Pz;
        tol::Real=1e-8, tol_pol::Real = 1e-6, maxit::Int=500, verbose::Bool=false,
        interp_kind::InterpKind=LinearInterp(), relax::Real=0.5,
        ν::Real=1e-10, patience::Int=50, c_init=nothing)

    Na, Nz = length(agrid), length(zgrid)
    a = collect(agrid)
    a_min, a_max = minimum(a), maximum(a)

    R = (1 + p.r)
    cmin  = 1e-12
    converged = false
    iters = 0
    max_resid = Inf
    max_Δ_pol = Inf

    c = c_init === nothing ? fill(1.0, Na, Nz) : copy(c_init)
    a′    = similar(c)
    cnext = similar(a)
    cnew  = similar(c)
    a_next = similar(c)

    best_resid = Inf
    no_progress = 0

    for it in 1:maxit
        iters = it

        for (j,z) in enumerate(zgrid)
            y = exp(z)

            @. a′[:,j] = R * a + y - c[:,j]
            @. a′[:,j] = clamp(a′[:,j], a_min, a_max)

            EUprime = similar(a)
            fill!(EUprime, 0.0)

            for (jp,zp) in enumerate(zgrid)
                if interp_kind isa LinearInterp
                    interp_linear!(cnext, a, view(c,:,jp), view(a′,:,j))
                else
                    interp_pchip!(cnext, a, view(c,:,jp), view(a′,:,j))
                end
                @. EUprime += Pz[j,jp] * (cnext.^(-p.σ))
            end

            @. cnew[:,j] = ((p.β * R) * EUprime).^(-1/p.σ)
            cmax = @. y + R * a - a_min
            @. cnew[:,j] = clamp(cnew[:,j], cmin, cmax)

            @. a_next[:,j] = R * a + y - cnew[:,j]
            @. a_next[:,j] = clamp(a_next[:,j], a_min, a_max)

            if interp_kind isa MonotoneCubicInterp
                @inbounds for i in 2:Na
                    if cnew[i,j] < cnew[i-1,j]
                        cnew[i,j] = cnew[i-1,j] + 1e-12
                    end
                end
            end
        end

        max_Δ_pol = maximum(abs.(c - cnew))

        @. c = (1 - relax) * c + relax * cnew

        max_resid = -Inf
        res = euler_residuals_stochastic(p, a, zgrid, Pz, c)
        max_resid = maximum(res[min(2,end):end, :])

        if verbose && (it % 10 == 0)
            @info "iter=$it max_resid=$(max_resid) max_Δ_pol=$(max_Δ_pol)"
        end

        if max_resid < tol && max_Δ_pol < tol_pol
            converged = true
            break
        end

        if best_resid - max_resid < ν
            no_progress += 1
        else
            no_progress = 0
            best_resid = max_resid
        end

        if no_progress ≥ patience
            @warn "Stopped early: no progress in Euler errors for $patience iterations"
            break
        end
    end

    return (agrid=a, zgrid=zgrid, c=c, a_next=a_next,
            iters=iters, converged=converged, max_resid=max_resid, p=p)
end


end #module