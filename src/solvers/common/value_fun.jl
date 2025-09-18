"""
ValueFunction

Value-function utilities (Bellman operator evaluations, expectations, etc.)
used by projection methods and diagnostics.
"""
module ValueFunction

using ..CommonInterp: interp_linear!

export compute_value_policy

"""
    compute_value_policy(p, g, S, U, policy; tol=1e-8, maxit=1_000)

Stability-robust value evaluation that supports both deterministic (vector) and
stochastic (matrix over shocks) policies. Prefer this over `compute_value` in solvers.
"""
function compute_value_policy(p, g, S, U, policy; tol::Real = 1e-8, maxit::Int = 1_000)
    agrid = g[:a].grid
    Na = g[:a].N

    cpol = policy[:c].value
    apol = policy[:a].value

    # Determine discount factor β robustly (allow ASCII fallback)
    βv = begin
        nms = propertynames(p)
        if :β in nms
            getfield(p, :β)
        elseif :beta in nms
            getfield(p, :beta)
        else
            error("Parameter β (or beta) not found in params")
        end
    end

    # Deterministic
    if cpol isa AbstractVector && apol isa AbstractVector
        V = zeros(Na)
        tmp = similar(V)
        V_new = similar(V)
        u_c = U.u(cpol)
        for _ = 1:maxit
            cont = interp_linear!(tmp, agrid, V, apol)
            @. V_new = u_c + βv * cont
            δ = 0.0
            @inbounds @simd for i in eachindex(V)
                d = abs(V_new[i] - V[i])
                if d > δ
                    δ = d
                end
            end
            if δ < tol
                return V_new
            end
            V .= V_new
        end
        return V
    end

    # Stochastic
    @assert cpol isa AbstractMatrix && apol isa AbstractMatrix "Stochastic value evaluation expects matrix policies"
    Nz = size(cpol, 2)
    V = zeros(Na, Nz)
    cont = similar(V)
    tmp = similar(agrid)

    P = S === nothing ? nothing : S.Π  # transition matrix Π from ShockOutput
    @assert P !== nothing "Missing shocks transition matrix for stochastic value evaluation"

    V_new = similar(V)
    u_c = U.u(cpol)
    for _ = 1:maxit
        @inbounds for j = 1:Nz
            aj = view(apol, :, j)
            @views cont[:, j] .= 0.0
            @inbounds for jp = 1:Nz
                vcol = view(V, :, jp)
                interp_linear!(tmp, agrid, vcol, aj)
                @. cont[:, j] += P[j, jp] * tmp
            end
        end
        @. V_new = u_c + βv * cont
        δ = 0.0
        @inbounds @simd for i in eachindex(V)
            d = abs(V_new[i] - V[i])
            if d > δ
                δ = d
            end
        end
        if δ < tol
            return V_new
        end
        V .= V_new
    end
    return V
end

end # module
