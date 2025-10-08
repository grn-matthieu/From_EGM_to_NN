"""
Perturbation

Adapter for a simple perturbation-based policy update, wired into `API.solve`.
Primarily used for testing and baseline comparisons.
"""
module Perturbation

using ..API
import ..API: solve

using ..PerturbationKernel: solve_perturbation_det, solve_perturbation_stoch
using ..ValueFunction: compute_value_policy
using ..Determinism: canonicalize_cfg, hash_hex
using ..CommonValidators: is_nondec, is_positive, respects_amin
using ..UtilsConfig: maybe
using ..UtilsDiagnostics: mean_abs_error

export PerturbationMethod, build_perturbation_method

struct PerturbationMethod <: AbstractMethod
    opts::NamedTuple
end

"""
    build_perturbation_method(cfg::NamedTuple) -> PerturbationMethod

Options:
  - `a_bar` (optional): reference asset level for linearization; default grid midpoint
  - `verbose` (Bool): print details
"""
function build_perturbation_method(cfg::NamedTuple)
    solver_cfg = cfg.solver
    perturbation_cfg = solver_cfg.perturbation
    return PerturbationMethod((
        name = maybe(cfg, :method, solver_cfg.method),
        a_bar = perturbation_cfg.a_bar,
        verbose = solver_cfg.verbose,
        order = perturbation_cfg.order,
        h_a = perturbation_cfg.h_a,
        h_z = perturbation_cfg.h_z,
        tol_fit = perturbation_cfg.tol_fit,
        maxit_fit = perturbation_cfg.maxit_fit,
    ))
end

function solve(
    model::AbstractModel,
    method::PerturbationMethod,
    cfg::NamedTuple;
    rng = nothing,
)::Solution
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    sol =
        S === nothing ?
        solve_perturbation_det(
            p,
            g,
            U;
            a_bar = method.opts.a_bar,
            order = method.opts.order,
            h_a = method.opts.h_a,
            tol_fit = method.opts.tol_fit,
            maxit_fit = method.opts.maxit_fit,
        ) :
        solve_perturbation_stoch(
            p,
            g,
            S,
            U;
            a_bar = method.opts.a_bar,
            order = method.opts.order,
            h_a = method.opts.h_a,
            h_z = method.opts.h_z,
            tol_fit = method.opts.tol_fit,
            maxit_fit = method.opts.maxit_fit,
        )

    ee = sol.resid
    ee_vec = ee isa AbstractMatrix ? vec(maximum(ee, dims = 2)) : ee
    ee_mat = ee isa AbstractMatrix ? ee : nothing
    ee_mean = ee_mat === nothing ? mean_abs_error(ee_vec) : mean_abs_error(ee_mat)
    delta_pol = 0.0

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
        iterations = sol.iters,
        mean_ee = ee_mean,
        delta_pol = delta_pol,
    )

    metadata = Dict{Symbol,Any}(
        :iters => sol.iters,
        :max_it => sol.opts.maxit,
        :converged => sol.converged,
        :max_resid => sol.max_resid,
        :mean_ee => ee_mean,
        :delta_pol => delta_pol,
        :julia_version => string(VERSION),
        :a_bar => method.opts.a_bar,
        :order => get(method.opts, :order, 1),
        :fit_ok => get(sol.opts, :fit_ok, false),
        :quad_coeffs => get(sol.opts, :quad_coeffs, nothing),
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

    return Solution(
        policy = policy,
        value = value,
        diagnostics = diagnostics,
        metadata = metadata,
        model = model,
        method = method,
    )
end

end # module
