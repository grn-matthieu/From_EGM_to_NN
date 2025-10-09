module SolverIntegration

using ..CSVarUtils: csvar_expected_state

export integrate_expectation

"""
    integrate_expectation(kind, f, params, shocks, y_state)

Placeholder integration helper. Evaluates `f` at the deterministic expectation
of the next-period state implied by `params` and returns the result. Supports
`:mc` and `:gh` integration kinds (treated identically for now).
"""
function integrate_expectation(kind::Symbol, f::Function, params, shocks, y_state)
    kind in (:mc, :gh, :none) ||
        error("Unknown integration kind $(kind); expected :mc or :gh")
    y_eval =
        (hasproperty(params, :A) && y_state !== nothing) ?
        csvar_expected_state(params.A, y_state) : y_state
    return f(y_eval)
end

end # module
