"""
SteadyState

Utilities to compute steady states analytically or from a policy function.
Exports `steady_state_analytic` and `steady_state_from_policy`.
"""
module SteadyState

using ..API: AbstractModel, Solution, get_params, get_grids, get_shocks
using ..ConsumerSaving: ConsumerSavingModel
using ..CommonInterp: interp_linear!

export steady_state_analytic, steady_state_from_policy

"""
    steady_state_analytic(model::AbstractModel)

Compute the analytical steady state for the deterministic consumer-savings model.
Returns a NamedTuple with `(a_ss, c_ss, kind)` where `kind` is one of
`:lower_bound`, `:interior`, or `:upper_bound`.

For the baseline deterministic CRRA model with constant income y and gross
return R = 1 + r:
  - If βR < 1, the steady state is at the borrowing (asset) lower bound.
  - If βR ≈ 1, any constant `a` is a steady state (we return the lower bound).
  - If βR > 1, with a finite grid the upper bound is absorbing.

Errors if called when shocks are active (non-degenerate) since the natural
object then is a stationary distribution, not a point steady state.
"""
function steady_state_analytic(model::AbstractModel)
    S = get_shocks(model)
    # Analytic steady state is only defined for the pure deterministic model.
    # Any shocks (even degenerate ones) should cause this function to error to
    # avoid ambiguity between a point steady state and a stationary
    # distribution. Tests expect an exception when shocks are present.
    if S !== nothing
        error(
            "Analytic steady state is defined only for deterministic case (no active shocks)",
        )
    end

    p = get_params(model)
    g = get_grids(model)
    a_min = g[:a].min
    a_max = g[:a].max
    R = 1 + p.r
    βR = p.β * R

    if βR < 1 - 1e-12
        a_ss = a_min
        c_ss = p.y + p.r * a_ss
        return (a_ss = a_ss, c_ss = c_ss, kind = :lower_bound)
    elseif abs(βR - 1) <= 1e-12
        # Indeterminate set; return canonical representative at the lower bound
        a_ss = a_min
        c_ss = p.y + p.r * a_ss
        return (a_ss = a_ss, c_ss = c_ss, kind = :interior)
    else
        a_ss = a_max
        c_ss = p.y + p.r * a_ss
        return (a_ss = a_ss, c_ss = c_ss, kind = :upper_bound)
    end
end

"""
    steady_state_from_policy(sol::Solution)

Given a solved policy (deterministic or 1-shock degenerate), compute the
policy-implied steady state as a fixed point of `a' = a` on the asset grid.
Returns `(a_star, c_star, idx, gap)` where `gap = |a'(a_star) - a_star|`.

For the stochastic case (Nz > 1), this function errors to avoid ambiguity; in
that case compute a stationary distribution instead.
"""
function steady_state_from_policy(sol::Solution)
    model = sol.model
    S = get_shocks(model)
    if S !== nothing && length(S.zgrid) > 1
        error(
            "steady_state_from_policy: stochastic case detected; compute invariant distribution instead",
        )
    end

    agrid = sol.policy[:a].grid
    a_next = sol.policy[:a].value
    c = sol.policy[:c].value

    # Find closest fixed point on grid
    diffs = abs.(a_next .- agrid)
    idx = argmin(diffs)
    a_star = agrid[idx]
    c_star = c[idx]
    gap = diffs[idx]
    return (a_star = a_star, c_star = c_star, idx = idx, gap = gap)
end

end # module
