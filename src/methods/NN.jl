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
using ..Determinism: canonicalize_cfg, hash_hex
using ..UtilsConfig: maybe

export NNMethod, build_nn_method

struct NNMethod <: AbstractMethod
    opts::NamedTuple
end

function build_nn_method(cfg::NamedTuple)
    solver_cfg = cfg.solver
    return NNMethod((
        name = maybe(cfg, :method, solver_cfg.method),
        epochs = maybe(solver_cfg, :epochs, 1000),
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
    ))
end

function solve(model::AbstractModel, method::NNMethod, cfg::NamedTuple;)::Solution
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    # Call the NN kernel to solve the model and return the solution struct
    sol = solve_nn(model; opts = method.opts)

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
    diagnostics =
        (; model_id = model_id, method = method.opts.name, runtime = sol.opts.runtime)

    metadata = Dict{Symbol,Any}(
        :iters => sol.iters,
        :max_it => sol.opts.epochs,
        :converged => sol.converged,
        :max_resid => sol.max_resid,
        :tol => nothing,
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
