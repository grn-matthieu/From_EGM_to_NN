"""
EGM

Adapter that wires the EGM solver kernel into the unified `API.solve` method.
"""
module EGM

using ..API
import ..API: solve

using ..EGMKernel: solve_egm_det, solve_egm_stoch
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

export EGMMethod

struct EGMMethod <: AbstractMethod
    opts::NamedTuple
end

struct EGMRunContext{M<:AbstractModel,R,P,G,S,U}
    model::M
    method::EGMMethod
    cfg::NamedTuple
    rng::R
    params::P
    grids::G
    shocks::S
    utility::U
    csvar::Bool
end

function EGMRunContext(
    model::M,
    method::EGMMethod,
    cfg::NamedTuple;
    rng = nothing,
) where {M<:AbstractModel}
    params = get_params(model)
    grids = get_grids(model)
    shocks = get_shocks(model)
    utility = get_utility(model)
    csvar = is_csvar_model(params)
    return EGMRunContext{
        M,
        typeof(rng),
        typeof(params),
        typeof(grids),
        typeof(shocks),
        typeof(utility),
    }(
        model,
        method,
        cfg,
        rng,
        params,
        grids,
        shocks,
        utility,
        csvar,
    )
end

select_interpolant(method::EGMMethod) =
    method.opts.interp_kind == :linear ? LinearInterp() : MonotoneCubicInterp()

function build_initializer(ctx::EGMRunContext)
    init_cfg = maybe(ctx.cfg, :init)
    custom_c_data = maybe(init_cfg, :c)
    custom_c_vec = custom_c_data isa AbstractVector ? custom_c_data : nothing
    custom_c_array = custom_c_data isa AbstractArray ? custom_c_data : nothing

    custom_c = if ctx.shocks === nothing
        custom_c_vec
    elseif ctx.csvar
        custom_c_array
    else
        custom_c_data isa AbstractMatrix ? custom_c_data : nothing
    end

    return build_consumption_initializer(
        ctx.params,
        ctx.grids;
        shocks = ctx.shocks,
        warm_start = ctx.method.opts.warm_start,
        custom_c = custom_c,
    )
end

function run_kernel(ctx::EGMRunContext, c_init)
    interp = select_interpolant(ctx.method)

    if ctx.shocks === nothing
        return solve_egm_det(
            ctx.params,
            ctx.grids,
            ctx.utility;
            tol = ctx.method.opts.tol,
            tol_pol = ctx.method.opts.tol_pol,
            maxit = ctx.method.opts.maxit,
            interp_kind = interp,
            relax = ctx.method.opts.relax,
            verbose = ctx.method.opts.verbose,
            c_init = c_init,
            integration_method = ctx.method.opts.integration,
            rng = ctx.rng,
        )
    end

    return solve_egm_stoch(
        ctx.params,
        ctx.grids,
        ctx.shocks,
        ctx.utility;
        tol = ctx.method.opts.tol,
        tol_pol = ctx.method.opts.tol_pol,
        maxit = ctx.method.opts.maxit,
        interp_kind = interp,
        relax = ctx.method.opts.relax,
        verbose = ctx.method.opts.verbose,
        c_init = c_init,
        integration_method = ctx.method.opts.integration,
        rng = ctx.rng,
    )
end

function build_outputs(ctx::EGMRunContext, sol)
    ee_vec, ee_mat = summarise_euler_errors(sol.resid)
    ee_mean = ee_mat === nothing ? mean_abs_error(ee_vec) : mean_abs_error(ee_mat)
    delta_pol = hasproperty(sol, :delta_pol) ? sol.delta_pol : missing
    grid_info = ctx.grids[:a]
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
    shocks_for_value = ctx.csvar ? nothing : ctx.shocks
    value =
        compute_value_policy(ctx.params, ctx.grids, shocks_for_value, ctx.utility, policy)
    metadata = Dict{Symbol,Any}(
        :iters => sol.iters,
        :max_it => sol.opts.maxit,
        :converged => sol.converged,
        :max_resid => sol.max_resid,
        :rmse => hasproperty(sol, :rmse) ? getfield(sol, :rmse) : sol.max_resid,
        :tol => sol.opts.tol,
        :tol_pol => sol.opts.tol_pol,
        :relax => sol.opts.relax,
        :resid_metric =>
            hasproperty(sol.opts, :resid_metric) ? sol.opts.resid_metric : :rmse,
        :verbose => sol.opts.verbose,
        :interp_kind => string(sol.opts.interp_kind),
        :julia_version => string(VERSION),
        :delta_pol => delta_pol,
        :mean_ee => ee_mean,
    )

    amin = ctx.grids[:a].min
    validate_policy!(
        metadata,
        policy,
        amin;
        method_name = "EGM",
        verbose = ctx.method.opts.verbose,
        checks = DEFAULT_VALIDATION_CHECKS,
    )

    model_id = hash_hex(canonicalize_cfg(ctx.cfg))

    diagnostics = (;
        model_id = model_id,
        method = ctx.method.opts.name,
        seed = sol.opts.seed,
        runtime = sol.opts.runtime,
        iterations = sol.iters,
        mean_ee = ee_mean,
        delta_pol = delta_pol,
    )

    return (;
        policy = policy,
        value = value,
        metadata = metadata,
        diagnostics = diagnostics,
    )
end

"""
    build_egm_method(cfg::NamedTuple) -> EGMMethod

Construct an `EGMMethod` using solver options contained in the NamedTuple `cfg`.
"""
function build_egm_method(cfg::NamedTuple)
    solver_cfg = cfg.solver
    egm_cfg = solver_cfg.egm
    ik_raw = egm_cfg.interp_kind
    ik_sym = Symbol(lowercase(string(ik_raw)))
    warm_start = Symbol(lowercase(string(solver_cfg.warm_start)))
    integration_raw = maybe(egm_cfg, :integration, :gh)
    integration_sym = Symbol(lowercase(string(integration_raw)))
    return EGMMethod((
        name = solver_cfg.method,
        tol = solver_cfg.tol,
        tol_pol = solver_cfg.tol_pol,
        maxit = solver_cfg.maxit,
        interp_kind = ik_sym,
        verbose = solver_cfg.verbose,
        warm_start = warm_start,
        relax = solver_cfg.relax,
        integration = integration_sym,
    ))
end

"""
    solve(model::AbstractModel, method::EGMMethod, cfg::NamedTuple; rng=nothing)::Solution

Entry point for the EGM solver. Extracts contract fields, runs a minimal EGM loop, and returns a Solution.
"""
function solve(
    model::AbstractModel,
    method::EGMMethod,
    cfg::NamedTuple;
    rng = nothing,
)::Solution
    ctx = EGMRunContext(model, method, cfg; rng = rng)
    c_init = build_initializer(ctx)
    sol = run_kernel(ctx, c_init)
    outputs = build_outputs(ctx, sol)
    return Solution(; outputs..., model = model, method = method)
end

end # module
