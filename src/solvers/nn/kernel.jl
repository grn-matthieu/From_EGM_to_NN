module NNKernel

using ..EulerResiduals: euler_resid_det_2, euler_resid_stoch!
using ..NNLoss: assemble_euler_loss
using ..NNConstraints: project_savings, project_savings_clip
using Statistics: mean
using Printf
using Dates

"""
    ensure_dir(path)

Create directory `path` if it doesn't already exist.
"""
function ensure_dir(path)
    isdir(path) || mkpath(path)
    return path
end

"""
    open_logger(dir) -> (fname, io)

Ensure `dir` exists and open a timestamped CSV file for logging.
Header: ts,epoch,train_loss,feas,lr,batch,seed,config
"""
function open_logger(dir)::Tuple{String,IO}
    ensure_dir(dir)
    fname = joinpath(
        dir,
        "run_" * Dates.format(Dates.now(), dateformat"yyyymmdd_HHMMSS") * ".csv",
    )
    io = open(fname, "w")
    println(io, "ts,epoch,train_loss,feas,lr,batch,seed,config")
    return fname, io
end

export solve_nn_det, solve_nn_stoch

"""
    solve_nn_det(p, g, U; tol=1e-6, maxit=1_000, verbose=false)

Baseline deterministic kernel for the Maliar et al. (2021) neural-network method.

This baseline provides an API-compatible return signature so that the adapter can
construct a `Solution`. It does not perform NN training yet; instead, it returns
an initial feasible policy based on a simple half-resources rule, along with
Euler residuals computed on the grid.

Returns a NamedTuple with fields:
  - `a_grid`: asset grid vector
  - `c`: consumption policy (vector)
  - `a_next`: next assets policy (vector)
  - `resid`: Euler equation residuals (vector)
  - `iters`: number of solver iterations (Int)
  - `converged`: convergence flag (Bool)
  - `max_resid`: maximum residual (Float64)
  - `model_params`: passthrough of `p`
  - `opts`: NamedTuple of runtime options/diagnostics
"""
function solve_nn_det(
    p,
    g,
    U,
    cfg::AbstractDict;
    tol::Real = 1e-6,
    maxit::Int = 1_000,
    verbose::Bool = false,
    projection_kind::Symbol = :softplus,
)
    t0 = time_ns()
    a_grid = g[:a].grid
    a_min = g[:a].min
    a_max = g[:a].max
    R = 1 + p.r
    Na = g[:a].N

    # Simple feasible baseline: half of current resources
    resources = @. R * a_grid - a_min + p.y
    cmin = 1e-12
    cmax = @. p.y + R * a_grid - a_min
    c = clamp.(0.5 .* resources, cmin, cmax)

    a_next_raw = @. R * a_grid + p.y - c
    a_next = project_savings(a_next_raw, a_min; kind = projection_kind)
    @. a_next = min(a_next, a_max)

    resid = euler_resid_det_2(p, a_grid, c)

    # Feasibility metric: share with a' >= a_min (post-projection)
    feas = mean(vec(project_savings_clip(a_next, a_min) .== a_next))

    # Optional weighted/stabilized loss (defaults preserve previous behavior)
    solver_cfg = get(cfg, :solver, Dict{Symbol,Any}())
    cfgw = (
        stabilize = get(solver_cfg, :stabilize, false),
        stab_method = get(solver_cfg, :stab_method, :log1p_square),
        residual_weighting = get(solver_cfg, :residual_weighting, :none), # :none|:exp|:linear
        weight_alpha = float(get(solver_cfg, :weight_alpha, 5.0)),
        weight_kappa = float(get(solver_cfg, :weight_kappa, 20.0)),
    )
    loss_val = assemble_euler_loss(resid, a_next, a_min, cfgw)

    # prune boundaries when assessing accuracy
    lo = Na > 2 ? 2 : 1
    hi = Na > 2 ? Na - 1 : Na
    max_resid = maximum(view(resid, lo:hi))

    opts = (;
        tol = tol,
        maxit = maxit,
        verbose = verbose,
        seed = nothing,
        runtime = (time_ns() - t0) / 1e9,
        loss = loss_val,
        projection_kind = projection_kind,
        feasibility = feas,
    )

    # --- Epoch logging (deterministic) ---
    logdir = joinpath(pwd(), "results", "nn", "baseline")
    fname, io = open_logger(logdir)
    epochs_cfg = get(get(cfg, :solver, Dict{Symbol,Any}()), :epochs, 1)
    lr_cfg = get(get(cfg, :solver, Dict{Symbol,Any}()), :lr, NaN)
    batch_cfg = get(get(cfg, :solver, Dict{Symbol,Any}()), :batch, 0)
    seed_cfg = get(get(cfg, :random, Dict{Symbol,Any}()), :seed, -1)
    cfg_name = basename(get(cfg, :config_path, "unknown"))
    @printf(
        io,
        "%s,%d,%.6g,%.6g,%.6g,%d,%d,%s\n",
        Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        Int(epochs_cfg),
        float(loss_val),
        float(feas),
        float(lr_cfg),
        Int(batch_cfg),
        Int(seed_cfg),
        cfg_name,
    )
    close(io)
    @info "NN baseline log written" path = fname

    return (
        a_grid = a_grid,
        c = c,
        a_next = a_next,
        resid = resid,
        iters = 1,
        converged = true,
        max_resid = max_resid,
        model_params = p,
        opts = opts,
    )
end

# Backwards-compatible method without cfg argument
function solve_nn_det(
    p,
    g,
    U;
    tol::Real = 1e-6,
    maxit::Int = 1_000,
    verbose::Bool = false,
    projection_kind::Symbol = :softplus,
)
    return solve_nn_det(
        p,
        g,
        U,
        Dict{Symbol,Any}();
        tol = tol,
        maxit = maxit,
        verbose = verbose,
        projection_kind = projection_kind,
    )
end


"""
    solve_nn_stoch(p, g, S, U; tol=1e-6, maxit=1_000, verbose=false)

Baseline stochastic kernel for the Maliar et al. (2021) neural-network method.

This baseline mirrors the deterministic baseline column-wise for each shock
state, building a feasible baseline policy and reporting Euler residuals using
`euler_resid_stoch!`.

Returns a NamedTuple with fields analogous to the deterministic case, except
that `c`, `a_next`, and `resid` are matrices of size (Na, Nz), and `z_grid` is
also returned.
"""
function solve_nn_stoch(
    p,
    g,
    S,
    U,
    cfg::AbstractDict;
    tol::Real = 1e-6,
    maxit::Int = 1_000,
    verbose::Bool = false,
    projection_kind::Symbol = :softplus,
)
    t0 = time_ns()

    a_grid = g[:a].grid
    a_min = g[:a].min
    a_max = g[:a].max
    Na = g[:a].N
    R = 1 + p.r

    z_grid = S.zgrid
    P = getfield(S, :?)  # transition matrix from shocks output
    Nz = length(z_grid)

    c = Array{Float64}(undef, Na, Nz)
    a_next = Array{Float64}(undef, Na, Nz)

    cmin = 1e-12
    for j = 1:Nz
        y = exp(z_grid[j])
        # half resources baseline per shock state
        @views begin
            resources = @. R * a_grid - a_min + y
            c[:, j] = clamp.(0.5 .* resources, cmin, y .+ R .* a_grid .- a_min)
            a_next_raw = @. R * a_grid + y - c[:, j]
            a_next[:, j] =
                min.(project_savings(a_next_raw, a_min; kind = projection_kind), a_max)
        end
    end

    resid = similar(c)
    euler_resid_stoch!(resid, p, a_grid, z_grid, P, c)

    # Optional weighted/stabilized loss
    solver_cfg = get(cfg, :solver, Dict{Symbol,Any}())
    cfgw = (
        stabilize = get(solver_cfg, :stabilize, false),
        stab_method = get(solver_cfg, :stab_method, :log1p_square),
        residual_weighting = get(solver_cfg, :residual_weighting, :none),
        weight_alpha = float(get(solver_cfg, :weight_alpha, 5.0)),
        weight_kappa = float(get(solver_cfg, :weight_kappa, 20.0)),
    )
    loss_val = assemble_euler_loss(resid, a_next, a_min, cfgw)

    max_resid = maximum(resid[min(2, end):end, :])

    # Feasibility metric across (Na, Nz): share with a' >= a_min
    feas = mean(vec(project_savings_clip(a_next, a_min) .== a_next))

    opts = (;
        tol = tol,
        maxit = maxit,
        verbose = verbose,
        seed = nothing,
        runtime = (time_ns() - t0) / 1e9,
        loss = loss_val,
        projection_kind = projection_kind,
        feasibility = feas,
    )

    # --- Epoch logging (stochastic) ---
    logdir = joinpath(pwd(), "results", "nn", "baseline")
    fname, io = open_logger(logdir)
    epochs_cfg = get(get(cfg, :solver, Dict{Symbol,Any}()), :epochs, 1)
    lr_cfg = get(get(cfg, :solver, Dict{Symbol,Any}()), :lr, NaN)
    batch_cfg = get(get(cfg, :solver, Dict{Symbol,Any}()), :batch, 0)
    seed_cfg = get(get(cfg, :random, Dict{Symbol,Any}()), :seed, -1)
    cfg_name = basename(get(cfg, :config_path, "unknown"))
    @printf(
        io,
        "%s,%d,%.6g,%.6g,%.6g,%d,%d,%s\n",
        Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        Int(epochs_cfg),
        float(loss_val),
        float(feas),
        float(lr_cfg),
        Int(batch_cfg),
        Int(seed_cfg),
        cfg_name,
    )
    close(io)
    @info "NN baseline log written" path = fname

    return (
        a_grid = a_grid,
        z_grid = z_grid,
        c = c,
        a_next = a_next,
        resid = resid,
        iters = 1,
        converged = true,
        max_resid = max_resid,
        model_params = p,
        opts = opts,
    )
end

# Backwards-compatible method without cfg argument
function solve_nn_stoch(
    p,
    g,
    S,
    U;
    tol::Real = 1e-6,
    maxit::Int = 1_000,
    verbose::Bool = false,
    projection_kind::Symbol = :softplus,
)
    return solve_nn_stoch(
        p,
        g,
        S,
        U,
        Dict{Symbol,Any}();
        tol = tol,
        maxit = maxit,
        verbose = verbose,
        projection_kind = projection_kind,
    )
end

end # module
