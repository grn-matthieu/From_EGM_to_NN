"""
NN

Adapter exposing a neural-network-based solver (simple 2-layer MLP) through `API.solve`.
For now this is a stub that calls into `NNKernel` which provides a placeholder implementation.
"""
module NN
using ..API
using Printf
import ..API: solve
using ..NNKernel: solve_nn
using ..ValueFunction: compute_value_policy
using ..Determinism: canonicalize_cfg, derive_rng, hash_hex, promote_master_rng
using ..UtilsDiagnostics: mean_abs_error

export NNMethod, build_nn_method

struct NNMethod <: AbstractMethod
    opts::NamedTuple
end

function build_nn_method(cfg::NamedTuple)
    solver_cfg = cfg.solver
    nn_cfg = solver_cfg.nn
    cfg_get(nt, sym, default) =
        hasproperty(nt, sym) ? getfield(nt, sym) :
        (nt isa AbstractDict ? get(nt, sym, default) : default)

    return NNMethod((
        name = solver_cfg.method,
        tol = solver_cfg.tol,
        # Paper defaults: 50_000 epochs, ADAM lr = 1e-3, batch = 64
        epochs = nn_cfg.epochs,
        batch = nn_cfg.batch,
        lr = nn_cfg.lr,
        verbose = solver_cfg.verbose,
        resample_every = nn_cfg.resample_every,

        # Architecture: hidden sizes (paper compares 8x8, 16x16, ...)
        hid1 = nn_cfg.hid1,
        hid2 = nn_cfg.hid2,

        # samples per epoch: paper draws 64 random grid points per epoch
        samples_per_epoch = nn_cfg.samples_per_epoch,

        # new: loss selector + stability knobs
        objective = nn_cfg.objective,
        v_h = nn_cfg.v_h,
        w_min = nn_cfg.w_min,
        w_max = nn_cfg.w_max,

        # optional: pass shock std override for convenience
        sigma_shocks = nn_cfg.sigma_shocks,
        target_loss = nn_cfg.target_loss,

        # device selection: allow config to explicitly request CUDA
        # pass-through so NNKernel.solver_settings can honor it
        use_cuda = nn_cfg.use_cuda,
        n_mc = nn_cfg.n_mc,
        bcmc_auto_N = cfg_get(nn_cfg, :bcmc_auto_N, false),
        bcmc_budget_T = cfg_get(nn_cfg, :bcmc_budget_T, nothing),
        bcmc_update_every = cfg_get(nn_cfg, :bcmc_update_every, 10),
    ))
end

function solve(
    model::AbstractModel,
    method::NNMethod,
    cfg::NamedTuple;
    rng = nothing,
)::Solution
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    master = cfg.random.master_rng

    # Call the NN kernel to solve the model and return the solution struct
    sol = solve_nn(model; opts = method.opts, rng = derive_rng(master, :nn_kernel))

    ee = sol.resid
    ee_vec = ee isa AbstractMatrix ? vec(maximum(ee, dims = 2)) : ee
    ee_mat = ee isa AbstractMatrix ? ee : nothing
    ee_mean = ee_mat === nothing ? mean_abs_error(ee_vec) : mean_abs_error(ee_mat)
    # Determine convergence using Euler error vs global solver tolerance when available
    tol = hasproperty(method.opts, :tol) ? getfield(method.opts, :tol) : nothing
    conv_flag = tol === nothing ? false : (isfinite(ee_mean) && ee_mean ≤ tol)
    delta_pol = hasproperty(sol, :delta_pol) ? sol.delta_pol : missing
    agrid = g.a.grid

    policy = Dict{Symbol,Any}(
        :c => (;
            value = sol.c,
            grid = agrid,
            euler_errors = ee_vec,
            euler_errors_mat = ee_mat,
        ),
        :a => (; value = sol.a_next, grid = agrid),
    )

    # Only compute value if the policy arrays are grid-aligned.
    Na = length(agrid)
    has_shocks = S !== nothing
    shapes_ok = false
    if !has_shocks
        shapes_ok =
            (sol.c isa AbstractVector) &&
            (sol.a_next isa AbstractVector) &&
            (length(sol.c) == Na) &&
            (length(sol.a_next) == Na)
    end

    value = shapes_ok ? compute_value_policy(p, g, S, U, policy) : nothing

    model_id = hash_hex(canonicalize_cfg(cfg))
    diagnostics = (;
        model_id = model_id,
        method = method.opts.name,
        runtime = sol.opts.runtime,
        device = get(sol.opts, :device, :cpu),
        iterations = sol.iters,
        mean_ee = ee_mean,
        delta_pol = delta_pol,
    )

    metadata = Dict{Symbol,Any}(
        :iters => sol.iters,
        :max_it => sol.opts.epochs,
        # Override kernel's convergence with Euler-error based criterion
        :converged => conv_flag,
        :max_resid => sol.euler_rmse,
        :tol => tol,
        :delta_pol => delta_pol,
        :mean_ee => ee_mean,
        :julia_version => string(VERSION),
        :rmse_history => hasproperty(sol, :rmse_history) ? sol.rmse_history : Float64[],
        :opts => sol.opts,
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
