module NNKernel

using ..EulerResiduals: euler_resid_det_2, euler_resid_stoch!
using ..NNLoss: assemble_euler_loss

export solve_nn_det, solve_nn_stoch

"""
    solve_nn_det(p, g, U; tol=1e-6, maxit=1_000, verbose=false)

Placeholder deterministic kernel for the Maliar et al. (2021) neural-network method.

This stub provides an API-compatible return signature so that the adapter can
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

    a_next = @. R * a_grid + p.y - c
    @. a_next = clamp(a_next, a_min, a_max)

    resid = euler_resid_det_2(p, a_grid, c)

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
    )

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
function solve_nn_det(p, g, U; tol::Real = 1e-6, maxit::Int = 1_000, verbose::Bool = false)
    return solve_nn_det(
        p,
        g,
        U,
        Dict{Symbol,Any}();
        tol = tol,
        maxit = maxit,
        verbose = verbose,
    )
end


"""
    solve_nn_stoch(p, g, S, U; tol=1e-6, maxit=1_000, verbose=false)

Placeholder stochastic kernel for the Maliar et al. (2021) neural-network method.

This stub mirrors the deterministic placeholder column-wise for each shock
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
            a_next[:, j] = clamp.(R .* a_grid .+ y .- c[:, j], a_min, a_max)
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

    opts = (;
        tol = tol,
        maxit = maxit,
        verbose = verbose,
        seed = nothing,
        runtime = (time_ns() - t0) / 1e9,
        loss = loss_val,
    )

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
    )
end

end # module
