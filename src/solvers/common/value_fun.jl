module ValueFunction

using ..CommonInterp: interp_linear!
export compute_value_policy

# small helper
_get(x, s::Symbol) = hasproperty(x, s) ? getfield(x, s) : x[s]

function compute_value_policy(p, g, S, U, policy; tol::Real = 1e-8, maxit::Int = 1_000)
    a = _get(g, :a)
    agrid = _get(a, :grid)
    Na = _get(a, :N)

    cpol = _get(_get(policy, :c), :value)
    apol = _get(_get(policy, :a), :value)

    βv =
        (:β in propertynames(p)) ? getfield(p, :β) :
        error("Parameter β not found in params")

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
    P = S === nothing ? nothing : _get(S, :Π)
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
