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
function compute_value(p, g, S, U, policy;tol=1e-8, maxit=1_000)
    # S is not used fn, to be implemented when dealing with stoch models
    V = zeros(g[:a].N)
    
    # Reusable interpolation workspace to avoid allocations
    tmp = similar(V)
    for _ in 1:maxit
        V_new = similar(V)
        cont = interp_linear!(tmp, g[:a].grid, V, policy[:a].value)
        V_new = U.u(policy[:c].value) + p.β * cont
        if maximum(abs.(V_new .- V)) < tol
            return V_new
        end
        V .= V_new
    end
    return V
end

end # module
