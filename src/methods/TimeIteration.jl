"""
TimeIteration method adapter

Builds a simple method object and exposes `solve` that mirrors the EGM adapter's
behavior so the rest of the codebase (tests, plotting) can use it similarly.
"""
module TimeIteration

using ..API
import ..API: solve

using ..TimeIterationKernel: solve_ti_det, solve_ti_stoch
using ..ValueFunction: compute_value_policy
using ..Determinism: canonicalize_cfg, hash_hex
using ..CommonInterp: LinearInterp, MonotoneCubicInterp
using ..UtilsConfig: maybe
using ..UtilsDiagnostics: mean_abs_error
using ..MethodUtils:
    build_consumption_initializer,
    validate_policy!,
    DEFAULT_VALIDATION_CHECKS,
    is_csvar_model,
    summarise_euler_errors

export TimeIterationMethod, build_timeiteration_method

struct TimeIterationMethod <: AbstractMethod
    opts::NamedTuple
end

function build_timeiteration_method(cfg::NamedTuple)
    solver_cfg = cfg.solver
    ti_cfg = solver_cfg.time_iteration
    ik_raw = ti_cfg.interp_kind
    ik_sym = Symbol(lowercase(string(ik_raw)))
    warm_start = Symbol(lowercase(string(solver_cfg.warm_start)))
    return TimeIterationMethod((
        name = solver_cfg.method,
        tol = solver_cfg.tol,
        tol_pol = solver_cfg.tol_pol,
        maxit = solver_cfg.maxit,
        relax = solver_cfg.relax,
        interp_kind = ik_sym,
        verbose = solver_cfg.verbose,
        warm_start = warm_start,
    ))
end

function solve(
    model::AbstractModel,
    method::TimeIterationMethod,
    cfg::NamedTuple;
    rng = nothing,
)::Solution
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    init_cfg = maybe(cfg, :init)
    custom_c_data = maybe(init_cfg, :c)
    custom_c_vec = custom_c_data isa AbstractVector ? custom_c_data : nothing
    custom_c_mat = custom_c_data isa AbstractMatrix ? custom_c_data : nothing

    csvar = is_csvar_model(p)
    shocks_for_solver = csvar ? nothing : S

    custom_c = if csvar
        custom_c_data isa AbstractArray ? custom_c_data : nothing
    elseif shocks_for_solver === nothing
        custom_c_vec
    else
        custom_c_mat
    end

    c_init = build_consumption_initializer(
        p,
        g;
        shocks = shocks_for_solver,
        warm_start = method.opts.warm_start,
        custom_c = custom_c,
    )

    ik = method.opts.interp_kind
    interp = ik == :linear ? LinearInterp() : MonotoneCubicInterp()

    sol = if shocks_for_solver === nothing
        solve_ti_det(
            p,
            g,
            U;
            tol = method.opts.tol,
            tol_pol = method.opts.tol_pol,
            maxit = method.opts.maxit,
            interp_kind = interp,
            relax = method.opts.relax,
            verbose = method.opts.verbose,
            c_init = c_init,
        )
    else
        solve_ti_stoch(
            p,
            g,
            S,
            U;
            tol = method.opts.tol,
            tol_pol = method.opts.tol_pol,
            maxit = method.opts.maxit,
            interp_kind = interp,
            relax = method.opts.relax,
            verbose = method.opts.verbose,
            c_init = c_init,
        )
    end

    mask_first = sol.resid isa AbstractArray && ndims(sol.resid) >= 2
    ee_vec, ee_mat = summarise_euler_errors(sol.resid; mask_first_row = mask_first)
    ee_mean = ee_mat === nothing ? mean_abs_error(ee_vec) : mean_abs_error(ee_mat)
    delta_pol = hasproperty(sol, :delta_pol) ? sol.delta_pol : missing
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

    metadata = Dict{Symbol,Any}(
        :iters => sol.iters,
        :max_it => sol.opts.maxit,
        :converged => sol.converged,
        :max_resid => sol.euler_rmse, # kept for compatibility
        :rmse => hasproperty(sol, :rmse) ? getfield(sol, :rmse) : sol.euler_rmse,
        :tol => sol.opts.tol,
        :tol_pol => sol.opts.tol_pol,
        :relax => sol.opts.relax,
        :resid_metric =>
            hasproperty(sol.opts, :resid_metric) ? sol.opts.resid_metric : :rmse,
        :interp_kind => string(sol.opts.interp_kind),
        :julia_version => string(VERSION),
        :delta_pol => delta_pol,
        :mean_ee => ee_mean,
    )

    # Validation
    amin = g[:a].min

    validate_policy!(
        metadata,
        policy,
        amin;
        method_name = "TimeIteration",
        verbose = method.opts.verbose,
        checks = DEFAULT_VALIDATION_CHECKS,
    )

    if hasproperty(sol, :placeholder) && sol.placeholder
        metadata[:placeholder] = true
        if hasproperty(sol.opts, :note)
            metadata[:placeholder_note] = sol.opts.note
        end
    end

    diagnostics = (;
        model_id = model_id,
        method = method.opts.name,
        seed = sol.opts.seed,
        runtime = sol.opts.runtime,
        iterations = sol.iters,
        mean_ee = ee_mean,
        delta_pol = delta_pol,
    )

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
