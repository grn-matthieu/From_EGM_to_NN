module ValueFunction

using ..CommonInterp: interp_linear!

export compute_value

"""
    compute_value(sol, p; tol=1e-8, maxit=1_000)

Evaluate the value function V(a) for a fixed policy (sol.c, sol.a_next) using policy
evaluation:
    V(a_i) = u(c(a_i)) + β * V(a'(a_i)).

Output: `Vector{Float64}` of length `length(sol.agrid) : policy fun`.
"""
function compute_value_policy(p, g, S, U, policy; tol::Real = 1e-8, maxit::Int = 1_000)
    agrid = g[:a].grid::AbstractVector{<:Real}
    Na = g[:a].N::Int

    cpol_any = policy[:c].value
    apol_any = policy[:a].value

    βv =
        hasproperty(p, :β) ? getfield(p, :β) :
        hasproperty(p, :beta) ? getfield(p, :beta) :
        throw(ArgumentError("Parameter β/beta not found"))

    # Deterministic
    if cpol_any isa AbstractVector{<:Real} && apol_any isa AbstractVector{<:Real}
        cpol = cpol_any::AbstractVector{<:Real}
        apol = apol_any::AbstractVector{<:Real}

        V = zeros(Float64, Na)
        tmp = similar(agrid, Float64)

        for _ = 1:maxit
            interp_linear!(tmp, agrid, V, apol)
            V_new = U.u(cpol) .+ βv .* tmp
            if maximum(abs.(V_new .- V)) < tol
                return V_new
            end
            V .= V_new
        end
        return V
    end

    # Stochastic
    cpol = (cpol_any::AbstractMatrix{<:Real})
    apol = (apol_any::AbstractMatrix{<:Real})
    Nz = size(cpol, 2)
    V = zeros(Float64, Na, Nz)
    cont = zeros(Float64, Na, Nz)
    tmp = similar(agrid, Float64)

    Π = S === nothing ? nothing : getfield(S, 2)
    @assert Π !== nothing "Missing shocks transition matrix for stochastic value evaluation"
    @assert Π isa AbstractMatrix "Π must be a matrix"
    @assert eltype(Π) <: Real "Π must be numeric"

    for _ = 1:maxit
        @inbounds for j = 1:Nz
            aj = @view apol[:, j]
            @views cont[:, j] .= 0.0
            @inbounds for jp = 1:Nz
                vcol = @view V[:, jp]
                interp_linear!(tmp, agrid, vcol, aj)
                pjj = Float64(Π[j, jp])       # concrete scalar
                @. cont[:, j] += pjj * tmp
            end
        end
        V_new = U.u(cpol) .+ βv .* cont
        if maximum(abs.(V_new .- V)) < tol
            return V_new
        end
        V .= V_new
    end
end
end # module
