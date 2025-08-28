module ValueFunction

export compute_value

"""
    compute_value(sol, p; tol=1e-8, maxit=1_000)

Evaluate the value function V(a) for a fixed policy (sol.c, sol.a_next) using policy
evaluation:
    V(a_i) = u(c(a_i)) + β * V(a'(a_i))
with linear interpolation on V at a'(a_i).

Expected fields:
- `sol.agrid :: Vector{Float64}`
- `sol.c     :: Vector{Float64}`
- `sol.a_next:: Vector{Float64}`
- `p.β, p.σ  :: parameters`; CRRA utility u(c)=c^(1-σ)/(1-σ)

Returns: `Vector{Float64}` of length `length(sol.agrid)`.
"""
function compute_value(sol, p; tol=1e-8, maxit=1_000)
    a  = sol.agrid
    c  = sol.c
    ap = sol.a_next
    Na = length(a)

    # CRRA utility; if σ ≈ 1, switch to log utility.
    u(c) = (c^(1 - p.σ) - 1) / (1 - p.σ)

    # local clamped linear interpolation
    @inline function lininterp(xgrid::AbstractVector, y::AbstractVector, x::Real)
        N = length(xgrid)
        if x <= xgrid[1]
            return y[1]
        elseif x >= xgrid[end]
            return y[end]
        else
            j = searchsortedfirst(xgrid, x)
            j = j < 2 ? 2 : (j > N ? N : j)
            x0 = xgrid[j-1]; x1 = xgrid[j]
            y0 = y[j-1];     y1 = y[j]
            t = (x - x0) / (x1 - x0)
            return (1 - t) * y0 + t * y1
        end
    end

    V = zeros(Na)
    for _ in 1:maxit
        Vnew = similar(V)
        @inbounds for i in 1:Na
            Vnew[i] = u(c[i]) + p.β * lininterp(a, V, ap[i])
        end
        if maximum(abs.(Vnew .- V)) < tol
            return Vnew
        end
        V .= Vnew
    end
    return V
end

end # module
