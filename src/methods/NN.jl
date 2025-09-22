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

export NNMethod, build_nn_method

struct NNMethod <: AbstractMethod
    opts::NamedTuple
end

function build_nn_method(cfg::NamedTuple)
    solver_cfg = hasproperty(cfg, :solver) ? cfg.solver : nothing
    solver_cfg === nothing && error("Missing solver section in configuration")
    return NNMethod((
        name = hasproperty(cfg, :method) ? cfg.method : solver_cfg.method,
        epochs = get(solver_cfg, :epochs, 10),
        batch = get(solver_cfg, :batch, 64),
        lr = get(solver_cfg, :lr, 1e-3),
        verbose = get(solver_cfg, :verbose, false),
    ))
end

function solve(model::AbstractModel, method::NNMethod, cfg::NamedTuple;)::Solution
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    # Call the NN kernel to solve the model and return the solution struct
    sol = solve_nn(model; opts = method.opts)
    @printf "DEBUG NN: solve_nn returned type = %s\n" string(typeof(sol))

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
        :max_it => sol.opts.epochs,
        :converged => sol.converged,
        :max_resid => sol.max_resid,
        :tol => nothing,
        :julia_version => string(VERSION),
    )

    return Solution(policy, value, diagnostics, metadata, model, method)
end

end # module
