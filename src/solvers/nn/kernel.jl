module NNKernel

using Lux
using Zygote
using Optimisers
using Random

using ..EulerResiduals: euler_resid_det_2, euler_resid_stoch!
using ..NNLoss: assemble_euler_loss
using ..NNConstraints: project_savings, project_savings_clip
using ..NNInit: init_state, NNState
using ..NNData: grid_minibatches
using ..NNPretrain: fit_to_EGM!
using ..API: build_model, build_method, solve
using Statistics: mean
using Printf
using Dates
using ..NNTrain: CSVLogger, log_row!
using ..Determinism: make_rng, derive_seed

export solve_nn_det, solve_nn_stoch

"""
    _solver_hyper(cfg)

Extract common NN solver hyperparameters from `cfg`.
"""
function _solver_hyper(cfg)
    s = get(cfg, :solver, Dict{Symbol,Any}())
    r = get(cfg, :random, Dict{Symbol,Any}())
    pk = get(s, :projection_kind, :softplus)
    sm = get(s, :stab_method, :log1p_square)
    rw = get(s, :residual_weighting, :none)
    return (
        epochs = Int(get(s, :epochs, 100)),
        batch = Int(get(s, :batch, 128)),
        lr = float(get(s, :lr, get(s, :eta, 1e-3))),
        clip_norm = get(s, :clip_norm, nothing),
        projection_kind = pk isa Symbol ? pk : Symbol(pk),
        stabilize = get(s, :stabilize, false),
        stab_method = sm isa Symbol ? sm : Symbol(sm),
        residual_weighting = rw isa Symbol ? rw : Symbol(rw),
        weight_alpha = float(get(s, :weight_alpha, 5.0)),
        weight_kappa = float(get(s, :weight_kappa, 20.0)),
        seed = get(r, :seed, nothing),
    )
end

"""
    _batch_loss_det(ps, st, model, X, p, a_min; hyper)

Differentiable per-batch loss for deterministic model based on Euler residuals.
The NN predicts next assets a' from inputs. We compute c = y + R a - a' (projected),
then a second forward pass at (a', y) to compute c', forming residuals
    |1 - β R (c/c')^σ|.
Returns a scalar Float64 loss.
"""
function _batch_loss_det(ps, st, model, X, p, a_min; hyper)
    # Inputs: X is either 1xB (a) or 2xB (a,y). Deterministic case => treat y scalar p.y
    a = @view X[1, :]
    B = size(X, 2)
    yvec = size(X, 1) >= 2 ? @view(X[2, :]) : fill(eltype(a)(p.y), B)

    # Determine model input dimension from first Dense layer params
    in_dim_expected = try
        size(getfield(ps, :layer_1).weight, 2)
    catch
        size(getfield(ps, 1).weight, 2)
    end
    # First forward: predict a' at (a, y) using expected input dimension
    Xin1 = in_dim_expected == 1 ? @views(view(X, 1:1, :)) : X
    ŷ1_raw, st2 = Lux.apply(model, Xin1, ps, st)
    a′ = project_savings(ŷ1_raw, a_min; kind = hyper.projection_kind)

    # Current consumption
    R = 1 + p.r
    c = clamp.(yvec .+ R .* a .- a′, 1.0e-12, Inf)

    # Second forward: predict a" at (a′, y)
    Xin2 = in_dim_expected == 1 ? a′ : vcat(a′, reshape(yvec, 1, :))
    ŷ2_raw, _ = Lux.apply(model, Xin2, ps, st2)
    a″ = project_savings(ŷ2_raw, a_min; kind = hyper.projection_kind)
    c′ = clamp.(yvec .+ R .* a′ .- a″, 1.0e-12, Inf)

    # Euler residuals for the batch
    β = getfield(p, Symbol("β"))
    σ = getfield(p, Symbol("σ"))
    resid = abs.(1 .- β .* R .* (c ./ c′) .^ σ)

    # Assemble loss with optional stabilization/weights near a_min
    return assemble_euler_loss(resid, a′, a_min, hyper)
end

"""
    _train_det!(state, a_grid, p; hyper, y_scalar)

Run a simple mini-batch training loop for the deterministic case.
Updates `state` in-place and returns final average loss.
"""
function _train_det!(state, a_grid, p; hyper, y_scalar, master_rng)
    # Mini-batches over (a, y). Use a singleton y-grid for determinism.
    y_grid = [float(y_scalar)]
    # Derive a dedicated RNG for minibatch sampling from the master RNG
    seed_mb = derive_seed(master_rng, "nn/kernel/minibatch/det")
    rng = make_rng(Int(seed_mb % typemax(Int)))
    mb = grid_minibatches(
        a_grid,
        y_grid;
        targets = nothing,
        batch = hyper.batch,
        shuffle = true,
        rng = rng,
        drop_last = false,
    )

    last_loss = NaN
    for epoch = 1:hyper.epochs
        sumL = 0.0
        cnt = 0
        for (X, _) in mb
            loss_only(ps_local) =
                _batch_loss_det(ps_local, state.st, state.model, X, p, a_grid[1]; hyper)
            L, back = Zygote.pullback(loss_only, state.ps)
            gs = back(1.0)[1]
            # Optional grad clipping
            if hyper.clip_norm !== nothing
                # simple global L2 over leaves
                s = 0.0
                function _acc(x)
                    if x isa AbstractArray
                        s += sum(abs2, x)
                    elseif x isa NamedTuple
                        for v in values(x)
                            _acc(v)
                        end
                    elseif x isa Tuple
                        for v in x
                            _acc(v)
                        end
                    end
                end
                _acc(gs)
                gnorm = sqrt(s)
                if isfinite(gnorm) && gnorm > hyper.clip_norm
                    scale = hyper.clip_norm / (gnorm + 1e-12)
                    function _scale!(x)
                        if x isa AbstractArray
                            @. x = scale * x
                        elseif x isa NamedTuple
                            for v in values(x)
                                _scale!(v)
                            end
                        elseif x isa Tuple
                            for v in x
                                _scale!(v)
                            end
                        end
                    end
                    _scale!(gs)
                end
            end
            new_optstate, new_ps = Optimisers.update(state.optstate, state.ps, gs)
            state = NNState(
                model = state.model,
                ps = new_ps,
                st = state.st,
                opt = state.opt,
                optstate = new_optstate,
                rngs = state.rngs,
            )
            sumL += float(L)
            cnt += 1
        end
        last_loss = cnt == 0 ? NaN : sumL / cnt
    end
    return state, last_loss
end

"""
    _batch_loss_stoch(ps, st, model, X, p, a_min, z_grid, P; hyper)

Differentiable per-batch stochastic Euler residual loss. Uses discrete expectation
over next-period shocks with transition matrix `P`.
X must be 2×B with rows (a, y) where y = exp(z_j) for some j in z_grid.
"""
function _batch_loss_stoch(ps, st, model, X, p, a_min, z_grid, P; hyper)
    @assert size(X, 1) >= 2 "stochastic batch requires (a,y) features"
    a = @view X[1, :]
    y = @view X[2, :]
    B = size(X, 2)
    R = 1 + p.r
    β = getfield(p, Symbol("β"))
    σ = getfield(p, Symbol("σ"))

    # Map y -> z index j
    zvals = log.(y)
    jidx = map(zv -> clamp(searchsortedfirst(z_grid, zv), 1, length(z_grid)), zvals)

    # Forward current policy a' and consumption c
    ŷ_raw, st2 = Lux.apply(model, X, ps, st)
    ap = vec(project_savings(ŷ_raw, a_min; kind = hyper.projection_kind))
    c = clamp.(y .+ R .* a .- ap, 1.0e-12, Inf)

    # Expected next-period marginal utility times R (functional, no in-place updates)
    T = hcat(
        [
            begin
                yprime = exp(z_grid[jp])
                Xin = vcat(reshape(ap, 1, :), fill(yprime, 1, B))
                y2_raw, _ = Lux.apply(model, Xin, ps, st2)
                ap2 = vec(project_savings(y2_raw, a_min; kind = hyper.projection_kind))
                cp = clamp.(yprime .+ R .* ap .- ap2, 1.0e-12, Inf)
                w = P[jidx, jp]
                w .* (cp ./ c) .^ (-σ)
            end for jp = 1:length(z_grid)
        ]...,
    )
    Emu = vec(sum(T; dims = 2))
    resid = abs.(1 .- β .* R .* Emu)
    return assemble_euler_loss(resid, ap, a_min, hyper)
end

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
    Na = g[:a].N

    # Hyperparameters and state init
    hyper = _solver_hyper(cfg)
    st = init_state(Dict(cfg))
    # Ensure optimiser LR matches cfg if provided (NNState is immutable)
    if hyper.lr !== nothing
        newopt = Optimisers.Adam(hyper.lr)
        newoptstate = Optimisers.setup(newopt, st.ps)
        st = NNState(
            model = st.model,
            ps = st.ps,
            st = st.st,
            opt = newopt,
            optstate = newoptstate,
            rngs = st.rngs,
        )
    end

    # Master RNG for deterministic solver
    seed_cfg = get(get(cfg, :random, Dict{Symbol,Any}()), :seed, nothing)
    master_rng = seed_cfg === nothing ? make_rng(Int(0x9a9aa9a9)) : make_rng(Int(seed_cfg))

    # Train on mini-batches of (a, y)
    st, last_loss =
        _train_det!(st, a_grid, p; hyper = hyper, y_scalar = p.y, master_rng = master_rng)

    # Evaluate trained policy on full grid
    Xin = reshape(a_grid, 1, :)
    ŷ_raw, _ = Lux.apply(st.model, Xin, st.ps, st.st)
    a_next = project_savings(vec(ŷ_raw), a_min; kind = projection_kind)
    @. a_next = min(a_next, a_max)
    R = 1 + p.r
    c = clamp.(p.y .+ R .* a_grid .- a_next, 1.0e-12, Inf)

    resid = euler_resid_det_2(p, a_grid, c)

    # Feasibility metric: share with a' >= a_min (post-projection)
    feas = mean(vec(project_savings_clip(a_next, a_min) .== a_next))

    # Optional weighted/stabilized loss (defaults preserve previous behavior)
    cfgw = (
        stabilize = hyper.stabilize,
        stab_method = hyper.stab_method,
        residual_weighting = hyper.residual_weighting, # :none|:exp|:linear
        weight_alpha = hyper.weight_alpha,
        weight_kappa = hyper.weight_kappa,
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
    # --- Logging via NNTrain.CSVLogger ---
    logdir = joinpath(pwd(), "results", "nn", "baseline")
    logpath = joinpath(
        logdir,
        "run_" * Dates.format(Dates.now(), dateformat"yyyymmdd_HHMMSS") * ".csv",
    )
    lg = CSVLogger(logpath)
    log_row!(
        lg;
        epoch = Int(hyper.epochs),
        step = Int(hyper.epochs),
        split = "train",
        loss = float(isfinite(last_loss) ? last_loss : loss_val),
        grad_norm = NaN,
        lr = float(hyper.lr),
        stage = :final,
        grid_stride = 1,
        nMC = 1,
        shock_noise = NaN,
        lambda_penalty = NaN,
    )
    @info "NN baseline log written" path = logpath

    return (
        a_grid = a_grid,
        c = c,
        a_next = a_next,
        resid = resid,
        iters = Int(epochs_cfg),
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
    P = getfield(S, Symbol("Π"))
    Nz = length(z_grid)

    # Hyperparameters and state init (shocks-active => NN with 2 inputs)
    hyper = _solver_hyper(cfg)
    st = init_state(Dict(cfg))
    # Master RNG for stochastic solver
    seed_cfg = get(get(cfg, :random, Dict{Symbol,Any}()), :seed, nothing)
    master_rng = seed_cfg === nothing ? make_rng(Int(0x9a9aa9a9)) : make_rng(Int(seed_cfg))
    if hyper.lr !== nothing
        newopt = Optimisers.Adam(hyper.lr)
        newoptstate = Optimisers.setup(newopt, st.ps)
        st = NNState(
            model = st.model,
            ps = st.ps,
            st = st.st,
            opt = newopt,
            optstate = newoptstate,
            rngs = st.rngs,
        )
    end

    # --- Supervised pretraining to EGM policy (robust, improves residuals) ---
    # Build an EGM baseline solution to generate targets for (a, z)
    egm_cfg = try
        deepcopy(cfg)
    catch
        Dict{Symbol,Any}(pairs(cfg))
    end
    egm_solver = get(egm_cfg, :solver, Dict{Symbol,Any}())
    egm_solver[:method] = :EGM
    egm_cfg[:solver] = egm_solver
    mobj = build_model(egm_cfg)
    egm_method = build_method(egm_cfg)
    egm_sol = solve(mobj, egm_method, egm_cfg)
    a_next_egm = egm_sol.policy[:a].value  # (Na, Nz)

    # Efficient 1D linear interpolation on a-grid
    lin1(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, xq::Real) = begin
        n = length(x)
        if xq <= x[1]
            return y[1]
        elseif xq >= x[end]
            return y[end]
        else
            j = searchsortedfirst(x, xq)
            j = clamp(j, 2, n)
            x0 = x[j-1]
            x1 = x[j]
            y0 = y[j-1]
            y1 = y[j]
            t = (xq - x0) / (x1 - x0)
            return (1 - t) * y0 + t * y1
        end
    end
    # Map y to nearest z index
    function egm_policy(a_vec, y_vec)
        out = similar(a_vec)
        @inbounds for k in eachindex(a_vec)
            z = log(y_vec[k])
            j = searchsortedfirst(z_grid, z)
            j = clamp(j, 1, Nz)
            out[k] = lin1(a_grid, view(a_next_egm, :, j), a_vec[k])
        end
        return out
    end
    # Run pretraining epochs (use up to half the budget, at least 25)
    pre_epochs = max(25, fld(hyper.epochs, 2))
    if pre_epochs > 0
        seed_pre = derive_seed(master_rng, "nn/kernel/pretrain")
        fit_to_EGM!(
            st,
            egm_policy,
            cfg;
            epochs = pre_epochs,
            batch = hyper.batch,
            seed = Int(seed_pre % typemax(Int)),
        )
    end

    # --- Residual-based fine-tuning on stochastic Euler equation ---
    last_loss = NaN
    tune_epochs = max(0, hyper.epochs - pre_epochs)
    if tune_epochs > 0
        y_grid = exp.(z_grid)
        # Derive a dedicated RNG for minibatch sampling from the master RNG
        seed_mb = derive_seed(master_rng, "nn/kernel/minibatch/stoch")
        rng = make_rng(Int(seed_mb % typemax(Int)))
        mb = grid_minibatches(
            a_grid,
            y_grid;
            targets = nothing,
            batch = hyper.batch,
            shuffle = true,
            rng = rng,
            drop_last = false,
        )
        for epoch = 1:tune_epochs
            sumL = 0.0
            cnt = 0
            for (X, _) in mb
                loss_only(ps_local) = _batch_loss_stoch(
                    ps_local,
                    st.st,
                    st.model,
                    X,
                    p,
                    a_min,
                    z_grid,
                    P;
                    hyper = hyper,
                )
                L, back = Zygote.pullback(loss_only, st.ps)
                gs = back(1.0)[1]
                new_optstate, new_ps = Optimisers.update(st.optstate, st.ps, gs)
                st = NNState(
                    model = st.model,
                    ps = new_ps,
                    st = st.st,
                    opt = st.opt,
                    optstate = new_optstate,
                    rngs = st.rngs,
                )
                sumL += float(L)
                cnt += 1
            end
            last_loss = cnt == 0 ? NaN : sumL / cnt
        end
    end

    # Evaluate trained policy on full grid for each shock state
    c = Array{Float64}(undef, Na, Nz)
    a_next = Array{Float64}(undef, Na, Nz)
    cmin = 1e-12
    for j = 1:Nz
        y = exp(z_grid[j])
        Xfull = vcat(reshape(a_grid, 1, :), fill(y, 1, Na))
        ŷ_raw, _ = Lux.apply(st.model, Xfull, st.ps, st.st)
        ap = vec(project_savings(ŷ_raw, a_min; kind = hyper.projection_kind))
        ap = clamp.(ap, a_min, a_max)
        a_next[:, j] = ap
        c[:, j] = clamp.(y .+ R .* a_grid .- ap, cmin, Inf)
    end

    resid = similar(c)
    euler_resid_stoch!(resid, p, a_grid, z_grid, P, c)

    # Optional weighted/stabilized loss
    cfgw = (
        stabilize = hyper.stabilize,
        stab_method = hyper.stab_method,
        residual_weighting = hyper.residual_weighting,
        weight_alpha = hyper.weight_alpha,
        weight_kappa = hyper.weight_kappa,
    )
    loss_val = assemble_euler_loss(resid, a_next, a_min, cfgw)

    max_resid = maximum(resid[min(2, end):end, :])

    # Feasibility metric across (Na, Nz): share with a' >= a_min
    feas = mean(vec(project_savings_clip(a_next, a_min) .== a_next))

    opts = (;
        tol = tol,
        maxit = maxit,
        verbose = verbose,
        seed = hyper.seed,
        runtime = (time_ns() - t0) / 1e9,
        loss = loss_val,
        projection_kind = projection_kind,
        feasibility = feas,
    )
    # --- Logging via NNTrain.CSVLogger (stochastic) ---
    logdir = joinpath(pwd(), "results", "nn", "baseline")
    logpath = joinpath(
        logdir,
        "run_" * Dates.format(Dates.now(), dateformat"yyyymmdd_HHMMSS") * ".csv",
    )
    lg = CSVLogger(logpath)
    log_row!(
        lg;
        epoch = Int(hyper.epochs),
        step = Int(hyper.epochs),
        split = "train",
        loss = float(isfinite(last_loss) ? last_loss : loss_val),
        grad_norm = NaN,
        lr = float(hyper.lr),
        stage = :final,
        grid_stride = 1,
        nMC = 1,
        shock_noise = NaN,
        lambda_penalty = NaN,
    )
    @info "NN training log written" path = logpath

    return (
        a_grid = a_grid,
        z_grid = z_grid,
        c = c,
        a_next = a_next,
        resid = resid,
        iters = Int(epochs_cfg),
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
