"""
Perturbation

Adapter for a simple perturbation-based policy update, wired into `API.solve`.
Primarily used for testing and baseline comparisons.
"""
module Perturbation

using ..API
import ..API: solve

using ..PerturbationKernel:
    solve_perturbation_det, solve_perturbation_stoch, solve_perturbation_placeholder
using ..ValueFunction: compute_value_policy
using ..Determinism: canonicalize_cfg, hash_hex
using ..UtilsConfig: maybe
using ..UtilsDiagnostics: mean_abs_error
using ..MethodUtils: validate_policy!, is_csvar_model, summarise_euler_errors

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
    integration_raw = maybe(perturbation_cfg, :integration, :gh)
    integration_sym = Symbol(lowercase(string(integration_raw)))
    return PerturbationMethod((
        name = solver_cfg.method,
        a_bar = perturbation_cfg.a_bar,
        verbose = solver_cfg.verbose,
        order = perturbation_cfg.order,
        h_a = perturbation_cfg.h_a,
        h_z = perturbation_cfg.h_z,
        tol_fit = perturbation_cfg.tol_fit,
        maxit_fit = perturbation_cfg.maxit_fit,
        integration = integration_sym,
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

    csvar = is_csvar_model(p)
    integration = method.opts.integration
    if csvar && integration == :mc
        @warn "Perturbation CSVar fallback to Gauss-Hermite integration" integration
        integration = :gh
    end

    sol = if csvar
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
            integration_method = integration,
            rng = rng,
        )
    elseif S === nothing
        solve_perturbation_det(
            p,
            g,
            U;
            a_bar = method.opts.a_bar,
            order = method.opts.order,
            h_a = method.opts.h_a,
            tol_fit = method.opts.tol_fit,
            maxit_fit = method.opts.maxit_fit,
            rng = rng,
        )
    else
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
            integration_method = integration,
            rng = rng,
        )
    end

    ee_vec, ee_mat = summarise_euler_errors(sol.resid)
    ee_mean = ee_mat === nothing ? mean_abs_error(ee_vec) : mean_abs_error(ee_mat)
    delta_pol = 0.0

    grid_info = g[:a]
    tensor_shape = hasproperty(grid_info, :tensor_shape) ? grid_info.tensor_shape : nothing
    policy = Dict{Symbol,Any}(
        :c => (;
            value = sol.c,
            grid = grid_info.grid,
            tensor_shape = tensor_shape,
            euler_errors = ee_vec,
            euler_errors_mat = ee_mat,
        ),
        :a =>
            (; value = sol.a_next, grid = grid_info.grid, tensor_shape = tensor_shape),
    )

    shocks_for_value = csvar ? nothing : S
    value = compute_value_policy(p, g, shocks_for_value, U, policy)

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
        :max_resid => sol.euler_rmse,
        :mean_ee => ee_mean,
        :delta_pol => delta_pol,
        :julia_version => string(VERSION),
        :a_bar => method.opts.a_bar,
        :order => get(method.opts, :order, 1),
        :fit_ok => get(sol.opts, :fit_ok, false),
        :quad_coeffs => get(sol.opts, :quad_coeffs, nothing),
    )

    # Basic validations
    validate_policy!(
        metadata,
        policy,
        g[:a].min;
        method_name = "Perturbation",
        verbose = method.opts.verbose,
        checks = (:c_positive, :a_above_min),
    )

    if hasproperty(sol, :placeholder) && sol.placeholder
        metadata[:placeholder] = true
        if hasproperty(sol.opts, :note)
            metadata[:placeholder_note] = sol.opts.note
        end
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
