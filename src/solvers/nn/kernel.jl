"""
NNKernel

High-level orchestration of the neural-network solver for consumption policies.
The heavy lifting (pre-processing, training loop and evaluation utilities) lives
in dedicated helpers so that this file focuses on the solver flow.
"""
module NNKernel

using ..API: get_grids, get_params, get_shocks, get_utility
using ..CommonInterp: InterpKind, LinearInterp
using ..DataNN: generate_dataset
using ..EulerResiduals: euler_resid_det_grid, euler_resid_stoch_grid
using ..Determinism: derive_rng, promote_master_rng
using Lux
using Optimisers
using Random
using Printf
using Statistics: mean, quantile

include("mixed_precision.jl")
include("preprocessing.jl")
include("training_loop.jl")
include("evaluation.jl")

export solve_nn

const H_ALPHA = 1.0f0

"""Construct the dual-head Lux model used by the solver."""
function build_dual_head_network(input_dim::Int, hidden::NTuple{2,Int})
    H1, H2 = hidden
    final = Dense(H2, 2)
    trunk = Chain(Dense(input_dim, H1, leakyrelu), Dense(H1, H2, leakyrelu), final)
    # The post-processing function expects a 2×N matrix and splits it into Φ and h
    function postprocess(x)
        φ_pre = x[1:1, :]
        h_pre = x[2:2, :]
        # paper: Φ = sigmoid(·) and h = exp(·)
        return (; Φ = sigmoid.(φ_pre), h = exp.(h_pre))
    end
    return Chain(trunk, postprocess)
end

function build_model_config(P, U, scaler, P_resid, settings)
    return (
        P = P,
        U = U,
        v_h = settings.v_h,
        scaler = scaler,
        P_resid = P_resid,
        settings = settings,
        sigma_shocks = settings.sigma_shocks,
    )
end

function maybe_dense_diagnostics(
    model,
    params,
    states,
    P_resid,
    U,
    scaler,
    settings;
    eval_mc_fn = nothing,
    eval_gh_fn = nothing,
    G = nothing,
    S = nothing,
    P = nothing,
    rng = nothing,
)
    if !settings.has_shocks
        return nothing, nothing
    end
    # forward explicit grids/shocks/params when available to avoid relying on Main
    # Resolve eval function bindings at call time so test patches to
    # `NNKernel.eval_euler_residuals_mc`/`_gh` are respected. Tests may also
    # pass explicit functions via kwargs.
    fmc = eval_mc_fn === nothing ? eval_euler_residuals_mc : eval_mc_fn
    fgh = eval_gh_fn === nothing ? eval_euler_residuals_gh : eval_gh_fn

    # Prepare diagnostics variables and call the eval functions. Some tests or
    # user code may have patched simple
    # varargs versions that don't accept keyword args; try keyword call first
    # and fall back to positional-only if that fails with MethodError.
    mc_diag = nothing
    gh_diag = nothing
    try
        mc_diag = fmc(
            model,
            params,
            states,
            P_resid,
            U,
            scaler,
            settings;
            G = G,
            S = S,
            P = P,
            rng = rng,
        )
    catch err
        if err isa MethodError
            mc_diag = fmc(model, params, states, P_resid, U, scaler, settings)
        else
            rethrow()
        end
    end

    try
        gh_diag = fgh(
            model,
            params,
            states,
            P_resid,
            U,
            scaler,
            settings;
            G = G,
            S = S,
            P = P,
            rng = rng,
        )
    catch err
        if err isa MethodError
            gh_diag = fgh(model, params, states, P_resid, U, scaler, settings)
        else
            rethrow()
        end
    end
    return mc_diag, gh_diag
end

function build_options_summary(settings, training_result, runtime)
    return (;
        epochs = settings.epochs,
        epochs_run = training_result.epochs_run,
        batch = training_result.batch_size,
        lr = settings.learning_rate,
        runtime = runtime,
        verbose = settings.verbose,
        batches_per_epoch = training_result.batches_per_epoch,
        device = settings.use_cuda ? :cuda : :cpu,
    )
end

function solve_nn(model; opts = nothing, rng = nothing)
    rng === nothing && error("solve_nn requires a `rng` keyword argument")
    master = promote_master_rng(rng)
    train_rng = derive_rng(master, :train)
    diag_rng = derive_rng(master, :diagnostics)

    P = get_params(model)
    G = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    start_time = time_ns()
    has_shocks = !isnothing(S)
    settings = solver_settings(opts; has_shocks = has_shocks)
    scaler = FeatureScaler(P, G, S, settings)

    chain = build_dual_head_network(input_dimension(S), settings.hidden_sizes)

    P_resid = scalar_params(P)
    model_cfg = build_model_config(P, U, scaler, P_resid, settings)

    training_result = train_consumption_network!(
        chain,
        settings,
        scaler,
        P_resid,
        G,
        S,
        train_rng,
        model_cfg,
    )

    best_state = training_result.best_state
    trained_model = select_model(chain, best_state)
    params =
        settings.use_cuda ? fmap(cu, state_parameters(best_state)) :
        state_parameters(best_state)
    states =
        settings.use_cuda ? fmap(cu, state_states(best_state)) : state_states(best_state)

    evaluation = evaluate_solution(
        trained_model,
        params,
        states,
        P_resid,
        P,
        G,
        S,
        scaler;
        settings = settings,
        U = U,
    )

    runtime = (time_ns() - start_time) / 1e9
    opts_summary = build_options_summary(settings, training_result, runtime)
    converged = training_result.best_loss ≤ settings.target_loss

    eval_mc, eval_gh = maybe_dense_diagnostics(
        trained_model,
        params,
        states,
        P_resid,
        U,
        scaler,
        settings;
        G = G,
        S = S,
        P = P,
        rng = diag_rng,
    )

    _, w_grid = det_forward_inputs(G, P_resid)

    return (;
        w_grid = w_grid,
        c = evaluation.c,
        a_next = evaluation.a_next,
        resid = evaluation.resid,
        iters = training_result.epochs_run,
        converged = converged,
        max_resid = evaluation.max_resid,
        model_params = P,
        opts = opts_summary,
        eval_mc = eval_mc,
        eval_gh = eval_gh,
    )
end

# Fischer–Burmeister (Eq. 25)
@inline fb(a, h) = a + h .- sqrt.(a .^ 2 .+ h .^ 2)  # zero iff a≥0, h≥0, a*h=0

function loss_euler_fb_aio!(chain, ps, st, batch, model_cfg, rng)
    P = model_cfg.P
    U = model_cfg.U
    scaler = model_cfg.scaler
    P_resid = model_cfg.P_resid
    settings = model_cfg.settings
    uprime = U.u_prime

    T = eltype(batch)
    C_MIN = T(1e-3)

    Rg = one(T) + T(P.r)
    μ = T(P_resid.y)
    if size(batch, 1) == 2
        y0 = ((batch[1, :] .+ one(T)) ./ T(2)) .* T(scaler.y_range) .+ T(scaler.y_min)
        w0 = ((batch[2, :] .+ one(T)) ./ T(2)) .* T(scaler.w_range) .+ T(scaler.w_min)
    elseif size(batch, 1) == 1
        w0 = ((batch[1, :] .+ one(T)) ./ T(2)) .* T(scaler.w_range) .+ T(scaler.w_min)
        y0 = fill_like(exp(μ), w0)
    else
        throw(ArgumentError("Expected 1 or 2 feature rows, got $(size(batch, 1))"))
    end
    z0 = log.(y0) .- μ
    out, st1 = Lux.apply(chain, batch, ps, st)
    c0 = vec(phi_to_consumption(out[:Φ], w0; min_c = C_MIN))
    h = T.(vec(ensure_row(out[:h])))
    a_term = @. one(T) - c0 / w0

    ρ = T(P.ρ_shock)
    σ_shocks = T(P.σ_shock)
    ε1 = randn_like(rng, z0)
    ε2 = randn_like(rng, z0)
    z1 = @. ρ * z0 + σ_shocks * ε1
    z2 = @. ρ * z0 + σ_shocks * ε2

    y1 = exp.(μ .+ z1)
    y2 = exp.(μ .+ z2)
    a1 = @. w0 - c0
    a2 = a1
    w1 = @. Rg * a1 + y1
    w2 = @. Rg * a2 + y2

    X1 = vcat(reshape(y1, 1, :), reshape(w1, 1, :))
    X2 = vcat(reshape(y2, 1, :), reshape(w2, 1, :))

    normalize_feature_batch!(scaler, X1)
    normalize_feature_batch!(scaler, X2)
    out1, st1 = Lux.apply(chain, X1, ps, st1)
    out2, st2 = Lux.apply(chain, X2, ps, st1)

    c1 = vec(phi_to_consumption(out1[:Φ], w1; min_c = C_MIN))
    c2 = vec(phi_to_consumption(out2[:Φ], w2; min_c = C_MIN))

    β = T(P.β)
    q1 = @. β * Rg * uprime(c1) / uprime(c0)
    q2 = @. β * Rg * uprime(c2) / uprime(c0)

    fb_term = fb(a_term, @. one(T) - h)
    kt = @. fb_term^2
    gh1 = clamp.(q1 .- h, -T(1e3), T(1e3))
    gh2 = clamp.(q2 .- h, -T(1e3), T(1e3))
    aio_pen = (gh1 .* gh2) .^ 2

    v_h = hasproperty(model_cfg, :v_h) ? T(getfield(model_cfg, :v_h)) : one(T)
    loss_vec = kt .+ v_h .* aio_pen

    max_abs_q = maximum(abs.(vcat(q1, q2)))

    return mean(loss_vec),
    (st1, (; kt_mean = mean(kt), aio_mean = mean(aio_pen), max_abs_q = max_abs_q))
end

end # module
