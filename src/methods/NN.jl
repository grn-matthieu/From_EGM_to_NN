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
using ..UtilsConfig: maybe
using ..UtilsDiagnostics: mean_abs_error

export NNMethod, build_nn_method

struct NNMethod <: AbstractMethod
    opts::NamedTuple
end

function build_nn_method(cfg::NamedTuple)
    solver_cfg = cfg.solver
    return NNMethod((
        name = maybe(cfg, :method, solver_cfg.method),
        epochs = maybe(solver_cfg, :epochs, 100_000),
        batch = maybe(solver_cfg, :batch, 64),
        lr = maybe(solver_cfg, :lr, 1e-4),
        verbose = maybe(solver_cfg, :verbose, false),

        # new: loss selector + stability knobs
        objective = maybe(solver_cfg, :objective, :euler_fb_aio),
        v_h = maybe(solver_cfg, :v_h, 0.5),
        w_min = maybe(solver_cfg, :w_min, 0.1),
        w_max = maybe(solver_cfg, :w_max, 4.0),

        # optional: pass shock std override for convenience
        sigma_shocks = maybe(solver_cfg, :sigma_shocks, nothing),
        target_loss = maybe(solver_cfg, :target_loss, 1e-10),
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

    cfg_master =
        hasproperty(cfg, :random) && hasproperty(cfg.random, :master_rng) ?
        promote_master_rng(cfg.random.master_rng) : nothing
    master = rng === nothing ? cfg_master : promote_master_rng(rng)
    master === nothing &&
        error("No master RNG available; pass `rng` or ensure cfg.random.seed is set")

    # Call the NN kernel to solve the model and return the solution struct
    sol = solve_nn(model; opts = method.opts, rng = derive_rng(master, :nn_kernel))

    ee = sol.resid
    ee_vec = ee isa AbstractMatrix ? vec(maximum(ee, dims = 2)) : ee
    ee_mat = ee isa AbstractMatrix ? ee : nothing
    ee_mean = ee_mat === nothing ? mean_abs_error(ee_vec) : mean_abs_error(ee_mat)
    delta_pol = hasproperty(sol, :delta_pol) ? sol.delta_pol : missing

    policy = Dict{Symbol,Any}(
        :c => (;
            value = sol.c,
            grid = sol.a_grid,
            euler_errors = ee_vec,
            euler_errors_mat = ee_mat,
        ),
        :a => (; value = sol.a_next, grid = sol.a_grid),
    )

    # Only compute value if the policy arrays are grid-aligned.
    agrid = g[:a].grid
    Na = length(agrid)
    has_shocks = S !== nothing
    shapes_ok = false
    if !has_shocks
        shapes_ok =
            (sol.c isa AbstractVector) &&
            (sol.a_next isa AbstractVector) &&
            (length(sol.c) == Na) &&
            (length(sol.a_next) == Na)
    else
        Nz = size(S.Π, 1)
        shapes_ok =
            (sol.c isa AbstractMatrix) &&
            (sol.a_next isa AbstractMatrix) &&
            (size(sol.c, 1) == Na) &&
            (size(sol.c, 2) == Nz) &&
            (size(sol.a_next, 1) == Na) &&
            (size(sol.a_next, 2) == Nz)
    end

    value = shapes_ok ? compute_value_policy(p, g, S, U, policy) : nothing

    model_id = hash_hex(canonicalize_cfg(cfg))
    diagnostics = (;
        model_id = model_id,
        method = method.opts.name,
        runtime = sol.opts.runtime,
        iterations = sol.iters,
        mean_ee = ee_mean,
        delta_pol = delta_pol,
    )

    metadata = Dict{Symbol,Any}(
        :iters => sol.iters,
        :max_it => sol.opts.epochs,
        :converged => sol.converged,
        :max_resid => sol.max_resid,
        :tol => nothing,
        :delta_pol => delta_pol,
        :mean_ee => ee_mean,
        :julia_version => string(VERSION),
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
