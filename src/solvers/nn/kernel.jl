"""
NNKernel

High-level orchestration of the neural-network solver for consumption policies.
The heavy lifting (pre-processing, training loop and evaluation utilities) lives
in dedicated helpers so that this file focuses on the solver flow.
"""
module NNKernel

using ..API: get_grids, get_shocks, get_utility
using ..CommonInterp: InterpKind, LinearInterp
using ..DataNN: generate_dataset
using ..EulerResiduals: euler_resid_det_grid, euler_resid_stoch_grid
using Lux
using Optimisers
using Random
using Printf
using Statistics: mean

include("mixed_precision.jl")
include("preprocessing.jl")
include("training_loop.jl")

export solve_nn

struct EvaluationResult
    c::Any
    a_next::Any
    resid::Any
    max_resid::Float64
end

function evaluate_deterministic(model, params, states, P_resid, P, G, scaler)
    X_forward, _ = det_forward_inputs(G)
    normalize_feature_batch!(scaler, X_forward)
    pred = run_model(model, params, states, X_forward)

    if pred isa NamedTuple
        Φ = pred[:Φ]
        # recover a grid and compute cash-on-hand
        a_grid_f32 = float32_vector(G[:a].grid)
        w = cash_on_hand(a_grid_f32, 0.0f0, P_resid, scaler.has_shocks)
        # align Φ to row
        if ndims(Φ) == 2 && size(Φ, 1) == 1
            Φ_row = Φ
        elseif ndims(Φ) == 2 && size(Φ, 2) == 1
            Φ_row = permutedims(Φ)
        else
            Φ_row = reshape(vec(Φ), 1, :)
        end
        c_vec = @. clamp(Φ_row[1, :] .* w, 1.0f-8, Inf)
        a_grid_f32, _, c_vec_f32 = det_residual_inputs(c_vec, G)
        residuals = euler_resid_det_grid(P_resid, a_grid_f32, c_vec_f32)
        c_on_grid = convert_to_grid_eltype(G[:a].grid, c_vec)
    else
        predictions = pred
        a_grid_f32, c_vec, c_vec_f32 = det_residual_inputs(predictions, G)
        residuals = euler_resid_det_grid(P_resid, a_grid_f32, c_vec_f32)
        c_on_grid = convert_to_grid_eltype(G[:a].grid, c_vec)
    end
    max_resid = maximum(abs.(residuals))
    R = 1 + get_param(P, :r, 0.0)
    y = get_param(P, :y, 0.0)
    a_next = @. R * G[:a].grid + y - c_on_grid
    a_next = clamp_to_asset_bounds(a_next, G[:a])
    return EvaluationResult(c_on_grid, a_next, residuals, max_resid)
end

function evaluate_stochastic(model, params, states, P_resid, P, G, S, scaler)
    X_eval, _ = generate_dataset(G, S; mode = :full)
    normalize_samples!(scaler, X_eval)
    batch = prepare_training_batch(X_eval)
    pred = run_model(model, params, states, batch)
    if pred isa NamedTuple
        Φ = pred[:Φ]
        a_f32 = float32_vector(G[:a].grid)
        z_f32 = float32_vector(S.zgrid)
        Na = length(a_f32)
        Nz = length(z_f32)
        A = repeat(a_f32, inner = Nz)
        Z = repeat(z_f32, outer = Na)
        w = cash_on_hand(A, Z, P_resid, scaler.has_shocks)
        if ndims(Φ) == 2 && size(Φ, 1) == 1
            Φ_row = Φ
        elseif ndims(Φ) == 2 && size(Φ, 2) == 1
            Φ_row = permutedims(Φ)
        else
            Φ_row = reshape(vec(Φ), 1, :)
        end
        c_row = @. clamp(Φ_row[1, :] .* w, 1.0f-8, Inf)
        a_grid_f32, z_grid_f32, Pz_f32, c_matrix, c_matrix_f32 =
            stoch_residual_inputs(c_row, G, S)
    else
        predictions = pred
        a_grid_f32, z_grid_f32, Pz_f32, c_matrix, c_matrix_f32 =
            stoch_residual_inputs(predictions, G, S)
    end
    residuals =
        euler_resid_stoch_grid(P_resid, a_grid_f32, z_grid_f32, Pz_f32, c_matrix_f32)
    c_on_grid = convert_to_grid_eltype(G[:a].grid, c_matrix)
    max_resid = maximum(abs.(residuals))
    R = 1 + get_param(P, :r, 0.0)
    y = get_param(P, :y, 0.0)
    a_next = @. R * G[:a].grid + y - c_on_grid
    a_next = clamp_to_asset_bounds(a_next, G[:a])
    return EvaluationResult(c_on_grid, a_next, residuals, max_resid)
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
    P, G, S, U = get_params(model), get_grids(model), get_shocks(model), get_utility(model)
    start_time = time_ns()
    scaler = FeatureScaler(G, S)
    settings = solver_settings(opts; has_shocks = scaler.has_shocks)

    # Utility hooks from the model's utility object
    uprime = U.u_prime
    uprime_inv = U.u_prime_inv
    σ_crra = U.σ
    # -- model: shared trunk + two heads (phi in (0,1), h > 0)
    function build_dual_head_network(input_dim::Int, hidden::NTuple{2,Int})
        H1, H2 = hidden
        # trunk ends with a Dense producing 2 outputs (one per head)
        trunk =
            Chain(Dense(input_dim, H1, leakyrelu), Dense(H1, H2, leakyrelu), Dense(H2, 2))

        # final mapping: split the 2×N output into Φ and h and apply activations
        model = Chain(trunk, x -> begin
            # x is 2×N; first row -> pre-Φ, second row -> pre-h
            pre = x
            φ_pre = view(pre, 1:1, :)
            h_pre = view(pre, 2:2, :)
            (Φ = sigmoid.(φ_pre), h = exp.(h_pre))
        end)
        return model
    end

    input_dimension(S) = isnothing(S) ? 1 : 2

    chain = build_dual_head_network(input_dimension(S), settings.hidden_sizes)
    P_resid = scalar_params(P)
    # model_cfg provides fields expected by custom losses (P, U, v_h)
    model_cfg = (P = P, U = U, v_h = settings.v_h)
    rng = Random.default_rng()
    training_result =
        train_consumption_network!(chain, settings, scaler, P_resid, G, S, model_cfg, rng)
    best_state = training_result.best_state
    trained_model = select_model(chain, best_state)
    params = state_parameters(best_state)
    states = state_states(best_state)

    evaluation =
        isnothing(S) ?
        evaluate_deterministic(trained_model, params, states, P_resid, P, G, scaler) :
        evaluate_stochastic(trained_model, params, states, P_resid, P, G, S, scaler)

    runtime = (time_ns() - start_time) / 1e9
    opts_summary = build_options_summary(settings, training_result, runtime)
    converged = training_result.best_loss ≤ settings.target_loss

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
    )
end

end # module


# Fischer–Burmeister (Eq. 25)
@inline fb(a, h) = a + h .- sqrt.(a .^ 2 .+ h .^ 2)  # zero iff a≥0, h≥0, a*h=0

function loss_euler_fb_aio!(chain, ps, st, batch, model_cfg, rng)
    # unpack model + params
    P = model_cfg.P                      # β, r, ρ, σ, etc.
    uprime = model_cfg.U.u_prime

    # inputs
    a0 = @view batch[1, :]               # assets or any internal state you used
    if size(batch, 1) == 2
        y0 = @view batch[2, :]
    else
        y0 = zeros(eltype(batch), size(batch, 2))
    end

    # forward pass
    out, st1 = Lux.apply(chain, batch, ps, st)
    ϕ = vec(out[:Φ])                     # ensure vector
    h = vec(out[:h])                     # ensure vector

    # cash-on-hand and consumption — assume fields exist on P (use Greek letters)
    R = P.r
    β = P.β
    w0 = @. R * a0 + exp(y0)
    c0 = clamp.(ϕ .* w0, eps(eltype(batch)), Inf)
    a_term = @. 1 - c0 / w0              # a = 1 - c/w

    # shocks for AiO
    ε1 = randn!(rng, similar(y0))
    ε2 = randn!(rng, similar(y0))
    rho = P.ρ
    sigma = P.σ
    y1 = @. rho * y0 + sigma * ε1
    y2 = @. rho * y0 + sigma * ε2

    # transitions
    w1 = @. R * (w0 - c0) + exp(y1)
    w2 = @. R * (w0 - c0) + exp(y2)

    # next-period consumption via NN
    # Use the same feature mapping as the training pipeline: if inputs were
    # (a, z) we used X = vcat(w - w0 + a, y); if inputs are (w,y) use vcat(w1, y1)
    if size(batch, 1) == 2
        X1 = vcat(w1 .- w0 .+ a0, y1)
        X2 = vcat(w2 .- w0 .+ a0, y2)
    else
        X1 = vcat(w1, y1)
        X2 = vcat(w2, y2)
    end
    out1, _ = Lux.apply(chain, X1, ps, st1)  # re-use st1 for determinism
    out2, _ = Lux.apply(chain, X2, ps, st1)

    c1 = clamp.(vec(out1[:Φ]) .* w1, eps(eltype(batch)), Inf)
    c2 = clamp.(vec(out2[:Φ]) .* w2, eps(eltype(batch)), Inf)

    # Euler pieces
    q1 = @. β * R * uprime(c1) / uprime(c0)
    q2 = @. β * R * uprime(c2) / uprime(c0)

    # FB(KT) term uses 1 - h as the inequality residual proxy (Eq. 29)
    fb_term = fb(a_term, @. 1 - h)
    kt = @. fb_term^2

    # AiO product (Eq. 30)
    gh1 = @. q1 - h
    gh2 = @. q2 - h
    aio = gh1 .* gh2

    # weight
    v_h = hasproperty(model_cfg, :v_h) ? getfield(model_cfg, :v_h) : 1.0
    loss_vec = kt .+ v_h .* aio
    return mean(loss_vec), (st1, (; kt_mean = mean(kt), aio_mean = mean(aio)))
end
