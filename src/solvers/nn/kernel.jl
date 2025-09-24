"""
NNKernel

High-level orchestration of the neural-network solver for consumption policies.
The heavy lifting (pre-processing, training loop and evaluation utilities) lives
in dedicated helpers so that this file focuses on the solver flow.
"""
module NNKernel

using ..API: get_params, get_grids, get_shocks, get_utility
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
    predictions = run_model(model, params, states, X_forward)
    a_grid_f32, c_vec, c_vec_f32 = det_residual_inputs(predictions, G)
    residuals = euler_resid_det_grid(P_resid, a_grid_f32, c_vec_f32)
    c_on_grid = convert_to_grid_eltype(G[:a].grid, c_vec)
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
    predictions = run_model(model, params, states, batch)
    a_grid_f32, z_grid_f32, Pz_f32, c_matrix, c_matrix_f32 =
        stoch_residual_inputs(predictions, G, S)
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
    P, G, S, _ = get_params(model), get_grids(model), get_shocks(model), get_utility(model)
    start_time = time_ns()
    settings = solver_settings(opts)
    scaler = FeatureScaler(G, S)
    # -- model: shared trunk + two heads (phi in (0,1), h > 0)
    function build_dual_head_network(input_dim::Int, hidden::NTuple{2,Int})
        using Lux
        H1, H2 = hidden
        trunk = Chain(Dense(input_dim, H1, leakyrelu), Dense(H1, H2, leakyrelu))
        head_phi = Chain(Dense(H2, 1), x -> sigmoid.(x))      # Φ = c/w in (0,1)
        head_h = Chain(Dense(H2, 1), x -> exp.(x))          # h = exp(·) > 0

        # Wrap so forward returns (phi, h)
        model = Chain(x -> trunk(x), x -> (Φ = head_phi(x), h = head_h(x)))
        return model
    end

    input_dimension(S) = isnothing(S) ? 1 : 2

    chain = build_dual_head_network(input_dimension(S), settings.hidden_sizes)
    P_resid = scalar_params(P)
    training_result = train_consumption_network!(chain, settings, scaler, P_resid, G, S)
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
