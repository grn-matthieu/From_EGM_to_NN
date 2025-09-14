module Perturbation

using ..API
import ..API: solve

using ..PerturbationKernel: solve_perturbation_det, solve_perturbation_stoch
using ..ValueFunction: compute_value_policy
using ..Determinism: canonicalize_cfg, hash_hex
using ..CommonValidators: is_nondec, is_positive, respects_amin

export PerturbationMethod, build_perturbation_method

struct PerturbationMethod <: AbstractMethod
    opts::NamedTuple
end

"""
    build_perturbation_method(cfg::AbstractDict) -> PerturbationMethod

Options:
  - `a_bar` (optional): reference asset level for linearization; default grid midpoint
  - `verbose` (Bool): print details
"""
function build_perturbation_method(cfg::AbstractDict)
    return PerturbationMethod((
        name = haskey(cfg, :method) ? cfg[:method] : cfg[:solver][:method],
        a_bar = get(cfg[:solver], :a_bar, nothing),
        verbose = get(cfg[:solver], :verbose, false),
    ))
end

function solve(
    model::AbstractModel,
    method::PerturbationMethod,
    cfg::AbstractDict;
    rng = nothing,
)::Solution
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    sol =
        S === nothing ? solve_perturbation_det(p, g, U; a_bar = method.opts.a_bar) :
        solve_perturbation_stoch(p, g, S, U; a_bar = method.opts.a_bar)

    ee = sol.resid
    ee_vec = ee isa AbstractMatrix ? vec(maximum(ee, dims = 2)) : ee
    ee_mat = ee isa AbstractMatrix ? ee : nothing

    policy = Dict{Symbol,Any}(
        :c => (;
            value = sol.c,
            grid = sol.a_grid,
            euler_errors = ee_vec,
            euler_errors_mat = ee_mat,
        ),
        :a => (; value = sol.a_next, grid = sol.a_grid),
    )

    value = compute_value_policy(p, g, S, U, policy)

    model_id = hash_hex(canonicalize_cfg(cfg))
    diagnostics = (;
        model_id = model_id,
        method = method.opts.name,
        seed = sol.opts.seed,
        runtime = sol.opts.runtime,
    )

    metadata = Dict{Symbol,Any}(
        :iters => sol.iters,
        :max_it => sol.opts.maxit,
        :converged => sol.converged,
        :max_resid => sol.max_resid,
        :julia_version => string(VERSION),
        :a_bar => method.opts.a_bar,
    )

    # Basic validations
    amin = g[:a].min
    c_val = policy[:c].value
    a_val = policy[:a].value
    violations = Dict{Symbol,Any}()
    valid = true
    if !is_positive(c_val)
        violations[:c_positive] = false
        valid = false
    end
    if !respects_amin(a_val, amin)
        violations[:a_above_min] = false
        valid = false
    end
    metadata[:valid] = valid
    if !isempty(violations)
        metadata[:validation] = violations
    end

    return Solution(policy, value, diagnostics, metadata, model, method)
end

end # module
