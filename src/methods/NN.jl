"""
NN

Adapter for the neural-network solver, bridging `API.solve` to `NNKernel`.
"""
module NN

using ..API
import ..API: solve

using ..NNKernel: solve_nn_det, solve_nn_stoch
using ..ValueFunction: compute_value_policy
using ..Determinism: canonicalize_cfg, hash_hex
using ..NNTrain: CSVLogger, log_row!
using Dates

export NNMethod, build_nn_method

"""
    NNMethod

Adapter for the Maliar et al. (2021) neural-network solution method.
Holds solver options in a `NamedTuple`. This initial version provides a
baseline implementation that returns a feasible policy and a conforming `Solution` object.
"""
struct NNMethod <: AbstractMethod
    opts::NamedTuple
end

"""
    build_nn_method(cfg::AbstractDict) -> NNMethod

Construct an `NNMethod` using solver options contained in `cfg`.
Recognized options:
  - `tol` (Real): residual tolerance
  - `maxit` (Int): maximum training/iteration steps
  - `verbose` (Bool): print progress
  - `hidden` (Tuple/Vector): hidden layer sizes (used by NN init; default (32, 32))
"""
function build_nn_method(cfg::AbstractDict)
    return NNMethod((
        name = haskey(cfg, :method) ? cfg[:method] : cfg[:solver][:method],
        tol = get(cfg[:solver], :tol, 1e-6),
        maxit = get(cfg[:solver], :maxit, 1_000),
        verbose = get(cfg[:solver], :verbose, false),
        hidden = get(cfg[:solver], :hidden, (32, 32)),
    ))
end

"""
    solve(model::AbstractModel, method::NNMethod, cfg::AbstractDict; rng=nothing)::Solution

Entry point for the NN solver. Extracts model contract fields, calls the NN
kernel (deterministic or stochastic), then packages a `Solution` including
policy, value, diagnostics, and metadata.
"""
function solve(
    model::AbstractModel,
    method::NNMethod,
    cfg::AbstractDict;
    rng = nothing,
)::Solution
    # --- Extract contract fields ---
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    # --- Delegate to kernel ---
    # Pull projection kind from cfg (default :softplus for backwards compatibility)
    solver_cfg = get(cfg, :solver, Dict{Symbol,Any}())
    pk = get(solver_cfg, :projection_kind, :softplus)
    projection_kind = pk isa Symbol ? pk : Symbol(pk)

    sol =
        S === nothing ?
        solve_nn_det(
            p,
            g,
            U,
            cfg;
            tol = method.opts.tol,
            maxit = method.opts.maxit,
            verbose = method.opts.verbose,
            projection_kind = projection_kind,
        ) :
        solve_nn_stoch(
            p,
            g,
            S,
            U,
            cfg;
            tol = method.opts.tol,
            maxit = method.opts.maxit,
            verbose = method.opts.verbose,
            projection_kind = projection_kind,
        )

    # --- Optional logging (outside kernels) ---
    # Enable with cfg[:solver][:log_csv] = true (optional cfg[:solver][:log_dir])
    solver_cfg = get(cfg, :solver, Dict{Symbol,Any}())
    if get(solver_cfg, :log_csv, false) === true
        logdir = get(solver_cfg, :log_dir, joinpath(pwd(), "results", "nn", "baseline"))
        isdir(logdir) || mkpath(logdir)
        logpath = joinpath(
            String(logdir),
            "run_" * Dates.format(Dates.now(), dateformat"yyyymmdd_HHMMSS") * ".csv",
        )
        lg = CSVLogger(logpath)
        loss_for_log = get(sol.opts, :last_epoch_loss, get(sol.opts, :loss, NaN))
        log_row!(
            lg;
            epoch = Int(get(sol.opts, :epochs, 0)),
            step = Int(get(sol.opts, :epochs, 0)),
            split = "train",
            loss = float(loss_for_log),
            grad_norm = NaN,
            lr = float(get(sol.opts, :lr, NaN)),
            stage = :final,
            grid_stride = 1,
            nMC = 1,
            shock_noise = NaN,
            lambda_penalty = NaN,
        )
    end

    # --- Euler errors vectorization for policy packaging ---
    ee = sol.resid
    ee_vec = ee isa AbstractMatrix ? vec(maximum(ee, dims = 2)) : ee
    ee_mat = ee isa AbstractMatrix ? ee : nothing

    # --- Policy dictionary ---
    ag = haskey(sol, :a_grid) ? sol.a_grid : g[:a].grid
    policy = Dict{Symbol,Any}(
        :c => (;
            value = sol.c,
            grid = ag,
            euler_errors = ee_vec,
            euler_errors_mat = ee_mat,
        ),
        :a => (; value = sol.a_next, grid = ag),
    )

    # --- Value function ---
    value = compute_value_policy(p, g, S, U, policy)

    # --- Diagnostics and metadata ---
    model_id = hash_hex(canonicalize_cfg(cfg))
    diagnostics = (;
        model_id = model_id,
        method = method.opts.name,
        seed = get(sol.opts, :seed, nothing),
        runtime = get(sol.opts, :runtime, NaN),
        feasibility = get(sol.opts, :feasibility, NaN),
    )
    metadata = Dict{Symbol,Any}(
        :iters => sol.iters,
        :max_it => method.opts.maxit,
        :converged => sol.converged,
        :max_resid => sol.max_resid,
        :tol => method.opts.tol,
        :hidden => string(method.opts.hidden),
        :julia_version => string(VERSION),
    )

    return Solution(policy, value, diagnostics, metadata, model, method)
end

end # module
