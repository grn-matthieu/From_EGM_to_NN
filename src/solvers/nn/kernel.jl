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

const H_ALPHA = 5.0f0

"""Construct the dual-head Lux model used by the solver."""
function build_dual_head_network(input_dim::Int, hidden::NTuple{2,Int})
    H1, H2 = hidden
    trunk = Chain(Dense(input_dim, H1, leakyrelu), Dense(H1, H2, leakyrelu), Dense(H2, 2))
    # The post-processing function expects a 2×N matrix and splits it into Φ and h
    function postprocess(x)
        φ_pre = x[1:1, :]
        h_pre = x[2:2, :]
        return (; Φ = sigmoid.(φ_pre), h = exp.(H_ALPHA .* tanh.(h_pre)))
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
        mc_diag =
            fmc(model, params, states, P_resid, U, scaler, settings; G = G, S = S, P = P)
    catch err
        if err isa MethodError
            mc_diag = fmc(model, params, states, P_resid, U, scaler, settings)
        else
            rethrow()
        end
    end

    try
        gh_diag =
            fgh(model, params, states, P_resid, U, scaler, settings; G = G, S = S, P = P)
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
    )
end

function solve_nn(model; opts = nothing)
    P = get_params(model)
    G = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    start_time = time_ns()
    scaler = FeatureScaler(G, S)
    settings = solver_settings(opts; has_shocks = scaler.has_shocks)

    chain = build_dual_head_network(input_dimension(S), settings.hidden_sizes)

    P_resid = scalar_params(P)
    model_cfg = build_model_config(P, U, scaler, P_resid, settings)

    rng = Random.default_rng()
    training_result =
        train_consumption_network!(chain, settings, scaler, P_resid, G, S, model_cfg, rng)

    best_state = training_result.best_state
    trained_model = select_model(chain, best_state)
    params = state_parameters(best_state)
    states = state_states(best_state)

    evaluation = evaluate_solution(trained_model, params, states, P_resid, P, G, S, scaler)

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
    )

    return (;
        a_grid = G[:a].grid,
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

    a0 = ((batch[1, :] .+ one(T)) ./ T(2)) .* T(scaler.a_range) .+ T(scaler.a_min)
    z0 =
        settings.has_shocks ?
        ((batch[2, :] .+ one(T)) ./ T(2)) .* T(scaler.z_range) .+ T(scaler.z_min) :
        fill(zero(T), size(a0))

    out, st1 = Lux.apply(chain, batch, ps, st)
    w0 = T.(cash_on_hand(a0, z0, P_resid, settings.has_shocks))
    c0 = vec(phi_to_consumption(out[:Φ], w0; min_c = C_MIN))
    h = T.(vec(ensure_row(out[:h])))
    a_term = @. one(T) - c0 / w0

    ρ = T(P.ρ)
    σ_shocks =
        hasproperty(model_cfg, :sigma_shocks) && model_cfg.sigma_shocks !== nothing ?
        T(model_cfg.sigma_shocks) : T(P.σ_shocks)
    ε1 = randn!(rng, similar(z0))
    ε2 = randn!(rng, similar(z0))
    z1 = @. ρ * z0 + σ_shocks * ε1
    z2 = @. ρ * z0 + σ_shocks * ε2

    a1 = @. w0 - c0
    a2 = a1

    X1 = vcat(reshape(a1, 1, :), reshape(z1, 1, :))
    X2 = vcat(reshape(a2, 1, :), reshape(z2, 1, :))

    NX1 = normalize_feature_batch(scaler, X1)
    NX2 = normalize_feature_batch(scaler, X2)

    out1, _ = Lux.apply(chain, NX1, ps, st1)
    out2, _ = Lux.apply(chain, NX2, ps, st1)

    w1 = T.(cash_on_hand(a1, z1, P_resid, settings.has_shocks))
    w2 = T.(cash_on_hand(a2, z2, P_resid, settings.has_shocks))
    c1 = vec(phi_to_consumption(out1[:Φ], w1; min_c = C_MIN))
    c2 = vec(phi_to_consumption(out2[:Φ], w2; min_c = C_MIN))

    β = T(P.β)
    Rg = one(T) + T(P.r)
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
