"""
NNPretrain

Supervised pretraining routines to fit NN policies to EGM-generated targets
before residual-based fine-tuning.
"""
module NNPretrain

using Lux
using Zygote
using Optimisers
using Random
using Statistics

using ..API: get_grids, get_shocks, build_model
using ..NNConstraints: project_savings
using ..NNData: grid_minibatches
using ..NNInit: NNState, init_state

export fit_to_EGM!, pretrain_then_euler!

"""
    fit_to_EGM!(model, policy, cfg; epochs::Int, batch::Int, seed::Int=42) -> NamedTuple

Supervised pretraining that fits the NN policy to a provided EGM policy using
L2 loss over projected next-period assets.

Arguments:
- `model`: either an `NNState` (preferred) or a Lux model; if a Lux model is
  provided, parameters/optimizer are initialized via `init_state(cfg)`.
- `policy`: callable that returns EGM next-period assets. It must accept
  either `(a::AbstractVector, y::AbstractVector)` or a feature matrix `X::(2×B)`
  and return a vector/1×B array of targets `a′` evaluated at those points.
- `cfg`: configuration dictionary used to extract grids and options.

Behavior:
- Builds minibatches `(a, y)` from the model grids. If shocks are active,
  uses the discrete shock grid transformed to `y = exp(z)`; otherwise uses a
  scalar income from parameters if available or a singleton grid.
- For each batch, computes targets `t = a′_egm(a, y)` via `policy`, and NN
  predictions `p_raw = Lux.apply(model, X, ps, st)`. Applies feasibility
  projection `p = project_savings(p_raw, a_min; kind=cfg.projection_kind)`.
- Optimizes mean squared error `mean((p .- t).^2)` with the existing optimizer
  in `NNState` (default Adam) and logs per-epoch metrics.

Returns a NamedTuple with fields:
- `final_loss` (Float64), `epochs` (Int), `lr` (Float64), `batch` (Int),
  `nobs` (Int), and `seed` (Int). The trained weights remain in-place.
"""
function fit_to_EGM!(model_in, policy, cfg; epochs::Int, batch::Int, seed::Int = 42)
    # Resolve state (accept NNState directly, otherwise build it)
    state = model_in isa NNState ? model_in : init_state(cfg)
    model = state.model
    ps = state.ps
    st = state.st
    opt = state.opt
    optstate = state.optstate

    # Deterministic batches from grids (with deterministic RNG)
    rng = Random.MersenneTwister(seed)

    # Build a model from cfg to obtain grids/shocks
    mobj = build_model(cfg)
    g = get_grids(mobj)
    S = get_shocks(mobj)

    a_grid = g[:a].grid
    a_min = g[:a].min
    # y-grid: stochastic => exp.(zgrid); deterministic => singleton 1.0 (policy should ignore if not used)
    y_grid = begin
        if S === nothing
            # Try income level from params if present; else 1.0
            haskey(cfg, :params) && haskey(cfg[:params], :y) ? [float(cfg[:params][:y])] : [1.0]
        else
            exp.(S.zgrid)
        end
    end

    projection_kind_raw =
        get(get(cfg, :solver, Dict{Symbol,Any}()), :projection_kind, :softplus)
    projection_kind =
        projection_kind_raw isa Symbol ? projection_kind_raw : Symbol(projection_kind_raw)

    # Assemble minibatch iterator over (a,y)
    mb = grid_minibatches(
        a_grid,
        y_grid;
        targets = nothing,
        batch = batch,
        shuffle = true,
        rng = rng,
        drop_last = false,
    )

    function l2_loss_and_state(ps_local, st_local, X, targets)
        ŷ_raw, st_next = Lux.apply(model, X, ps_local, st_local)
        ŷ = project_savings(ŷ_raw, a_min; kind = projection_kind)
        return mean((ŷ .- targets) .^ 2), st_next, ŷ
    end

    # Helper to normalize policy outputs/targets to 1×B arrays
    _as_row(v) = v isa AbstractMatrix ? v : reshape(v, 1, :)

    shocks_active = S !== nothing

    # Training loop
    nobs = length(a_grid) * length(y_grid)
    lr = try
        get(
            get(cfg, :solver, Dict{Symbol,Any}()),
            :lr,
            (opt isa Optimisers.Adam ? opt.eta : NaN),
        )
    catch
        get(get(cfg, :solver, Dict{Symbol,Any}()), :lr, NaN)
    end

    last_loss = NaN
    for ep = 1:epochs
        ep_loss_sum = 0.0
        ep_count = 0
        # Fresh permutation each epoch
        for (X, _) in mb
            a = @view X[1, :]
            # If shocks inactive, synthesize a y-vector matching X element type and batch size
            if size(X, 1) >= 2
                y = @view X[2, :]
            else
                yval = convert(eltype(X), y_grid[1])
                y = fill(yval, size(X, 2))
            end
            # Targets from EGM policy
            t_vec = policy(a, y)
            T = _as_row(t_vec)

            # Loss and grads
            function loss_only(ps_local)
                X_in = shocks_active ? X : @view X[1:1, :]
                ŷ_raw, _ = Lux.apply(model, X_in, ps_local, st)
                ŷ = project_savings(ŷ_raw, a_min; kind = projection_kind)
                return mean((ŷ .- T) .^ 2)
            end

            L, back = Zygote.pullback(loss_only, ps)
            grads = back(1.0)[1]
            # Optimiser update
            optstate, ps = Optimisers.update(optstate, ps, grads)
            ep_loss_sum += float(L)
            ep_count += 1
        end
        last_loss = ep_count == 0 ? NaN : ep_loss_sum / ep_count
        println(
            "[pretrain] epoch=$(ep) L2_train=$(last_loss) lr=$(lr) batch=$(batch) nobs=$(nobs)",
        )
    end

    # Copy trained parameters back into caller-owned state (in-place on arrays)
    if model_in isa NNState
        _copy_tree_arrays!(getfield(model_in, :ps), ps)
    end

    return (
        final_loss = last_loss,
        epochs = Int(epochs),
        lr = float(lr),
        batch = Int(batch),
        nobs = Int(nobs),
        seed = Int(seed),
    )
end

# Recursively copy array contents from src -> dest (matching structure)
function _copy_tree_arrays!(dest, src)
    if dest isa NamedTuple && src isa NamedTuple
        for k in keys(dest)
            _copy_tree_arrays!(getfield(dest, k), getfield(src, k))
        end
    elseif dest isa Tuple && src isa Tuple
        for i in eachindex(dest)
            _copy_tree_arrays!(dest[i], src[i])
        end
    elseif dest isa AbstractArray && src isa AbstractArray
        @assert size(dest) == size(src)
        dest .= src
    else
        # numbers or unsupported leaves are ignored
    end
    return dest
end

"""
    pretrain_then_euler!(model, policy, cfg; Nwarm::Int, epochs::Int, batch::Int, λ0::Float64=0.0)

Two-phase schedule that preserves optimizer state:
- Phase 1 (epochs 1..Nwarm): supervised L2 pretraining to EGM via `fit_to_EGM!`.
- Phase 2 (epochs Nwarm+1..epochs): train with Euler residual objective while continuing from the same optimizer state.

Logs per-epoch metrics for both losses on a held-out minibatch: `L2_to_EGM` and `Euler_loss`.

Arguments:
- `model`: `NNState` or Lux model; if not `NNState`, initializes from `cfg`.
- `policy`: callable baseline policy (EGM) used for pretraining and as target for logging.
- `cfg`: configuration dict; reads solver options like stabilization, weighting, projection kind, etc.

Returns nothing; updates `model` in-place if it is an `NNState`.
"""
function pretrain_then_euler!(
    model_in,
    policy,
    cfg;
    Nwarm::Int,
    epochs::Int,
    batch::Int,
    λ0::Float64 = 0.0,
)
    state = model_in isa NNState ? model_in : init_state(cfg)
    # Short-cuts to state
    model = state.model
    ps = state.ps
    st = state.st
    opt = state.opt
    optstate = state.optstate

    # Common objects
    mobj = build_model(cfg)
    g = get_grids(mobj)
    a_grid = g[:a].grid
    a_min = g[:a].min

    # Phase 1: L2-to-EGM warmup
    if Nwarm > 0
        fit_to_EGM!(state, policy, cfg; epochs = Nwarm, batch = batch)
        # Refresh local bindings from state (they were updated in-place)
        ps = state.ps
        st = state.st
        opt = state.opt
        optstate = state.optstate
    end

    # Helper: one held-out minibatch for logging both losses
    function sample_batch()
        rng = Random.MersenneTwister(12345)
        y_grid = [1.0]  # fallback; residuals() should construct needed shocks internally if any
        mb = grid_minibatches(
            a_grid,
            y_grid;
            targets = nothing,
            batch = max(8, batch),
            shuffle = true,
            rng = rng,
            drop_last = false,
        )
        for (X, _) in mb
            return X
        end
        return reshape(a_grid, 1, :)
    end
    X_hold = sample_batch()

    # Pull solver-related flags
    solver_cfg = get(cfg, :solver, Dict{Symbol,Any}())
    stabilize = get(solver_cfg, :stabilize, false)
    stab_method = get(solver_cfg, :stab_method, :log1p_square)
    residual_weighting = get(solver_cfg, :residual_weighting, :none)
    weight_alpha = float(get(solver_cfg, :weight_alpha, 5.0))
    weight_kappa = float(get(solver_cfg, :weight_kappa, 20.0))
    projection_kind = get(solver_cfg, :projection_kind, :softplus)

    # Phase 2: Euler residual training
    for ep = (Nwarm+1):epochs
        # Compute residuals and training loss on a fresh batch
        R = residuals(model, policy, batch)
        ap_raw = policy(a_grid, fill(1.0, length(a_grid)); θ = nothing) # policy used only for logging ap shape in total_loss
        # Forward current model on holdout to get projected a'
        Xin = size(X_hold, 1) >= 2 ? X_hold : @view X_hold[1:1, :]
        yraw, _ = Lux.apply(model, Xin, ps, st)
        ap = project_savings(yraw, a_min; kind = projection_kind)

        maybe_weights =
            residual_weighting === :none ? nothing :
            (; scheme = residual_weighting, alpha = weight_alpha, kappa = weight_kappa)
        λ = anneal_lambda(
            ep,
            epochs;
            λ_start = λ0,
            λ_final = max(λ0, 1.0),
            schedule = :cosine,
        )
        L = total_loss(
            R,
            ap,
            a_min;
            λ = λ,
            reduction = :mean,
            stabilize = stabilize,
            method = stab_method,
            weights = maybe_weights,
        )

        # Backprop on L w.r.t parameters
        loss_only(ps_local) = total_loss(
            residuals(model, policy, batch),
            ap,
            a_min;
            λ = λ,
            reduction = :mean,
            stabilize = stabilize,
            method = stab_method,
            weights = maybe_weights,
        )
        Lval, back = Zygote.pullback(loss_only, ps)
        grads = back(1.0)[1]
        optstate, ps = Optimisers.update(optstate, ps, grads)

        # Keep state updated
        state.ps = ps
        state.optstate = optstate

        # Logging: compute both metrics on held-out
        # L2_to_EGM on holdout
        a = @view X_hold[1, :]
        y =
            size(X_hold, 1) >= 2 ? (@view X_hold[2, :]) :
            fill(one(eltype(a)), size(X_hold, 2))
        t_vec = policy(a, y)
        yraw_hold, _ = Lux.apply(model, Xin, ps, st)
        yproj_hold = project_savings(yraw_hold, a_min; kind = projection_kind)
        L2 = mean((yproj_hold .- reshape(t_vec, 1, :)) .^ 2)

        # Euler loss on same holdout batch by reusing assembled pieces
        Euler_loss = float(Lval)
        println(
            "[warm+euler] epoch=$(ep) L2_to_EGM=$(L2) Euler_loss=$(Euler_loss) λ=$(λ) batch=$(batch)",
        )
    end

    if model_in isa NNState
        _copy_tree_arrays!(getfield(model_in, :ps), ps)
        model_in.optstate = optstate
    end
    return nothing
end

end # module
