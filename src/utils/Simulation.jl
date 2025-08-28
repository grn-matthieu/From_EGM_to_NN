module Simulation

using ..EGMSolver: SimpleSolution
using ..SimpleModel: budget_next
using ..EGMSolver: lininterp

"""
    simulate_path(sol::SimpleSolution, p, w0::Real, T::Int)

Simulate a path of length `T` given:
- solution `sol` with policy functions,
- parameters `p`,
- initial wealth `w0`.

Returns `(wealth, consumption)` vectors of length `T+1` and `T`.
"""
function simulate_path(sol::SimpleSolution, p, w0::Real, T::Int)
    wealth = zeros(T+1)
    cons   = zeros(T)

    wealth[1] = w0

    for t in 1:T
        wt = wealth[t]
        if wt <= 0.0
            cons[t] = 0.0
            wealth[t+1] = 0.0
        else
            ct = lininterp(sol.agrid, sol.c, wt)
            cons[t] = ct
            wealth[t+1] = budget_next(wt, 0.0, p.r, ct)
        end
    end

    return wealth, cons
end

end # module
