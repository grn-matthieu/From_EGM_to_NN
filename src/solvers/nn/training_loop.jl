struct NNSolverSettings
    epochs::Int
    batch_choice::Union{Nothing,Int}
    learning_rate::Float64
    verbose::Bool
    resample_interval::Int
    target_loss::Float32
    patience::Int
    hidden_sizes::NTuple{2,Int}
end

struct TrainingResult
    best_state::Any
    best_loss::Float64
    epochs_run::Int
    batch_size::Int
    batches_per_epoch::Int
end

get_option(opts, key::Symbol, default) =
    opts === nothing ? default : (hasproperty(opts, key) ? getfield(opts, key) : default)

function solver_settings(opts)
    epochs = max(Int(get_option(opts, :epochs, 1000)), 0)
    batch_choice = get_option(opts, :batch, nothing)
    batch_choice = isnothing(batch_choice) ? nothing : max(Int(batch_choice), 1)
    learning_rate = Float64(get_option(opts, :lr, 1e-4))
    verbose = Bool(get_option(opts, :verbose, false))
    resample_interval = max(Int(get_option(opts, :resample_every, 25)), 0)
    target_loss = Float32(get_option(opts, :target_loss, 2e-4))
    patience = max(Int(get_option(opts, :patience, 200)), 0)
    hid1 = max(Int(get_option(opts, :hid1, 128)), 1)
    hid2 = max(Int(get_option(opts, :hid2, 128)), 1)
    return NNSolverSettings(
        epochs,
        batch_choice,
        learning_rate,
        verbose,
        resample_interval,
        target_loss,
        patience,
        (hid1, hid2),
    )
end

function build_network(input_dim::Int, settings::NNSolverSettings)
    h1, h2 = settings.hidden_sizes
    return Chain(Dense(input_dim, h1, relu), Dense(h1, h2, relu), Dense(h2, 1, softplus))
end

create_optimizer(settings::NNSolverSettings) = Optimisers.OptimiserChain(
    Optimisers.ClipGrad(0.1),
    Optimisers.Adam(settings.learning_rate),
)

function compute_batch_size(total_samples::Int, choice::Union{Nothing,Int})
    return isnothing(choice) ? max(total_samples, 1) :
           clamp(choice, 1, max(total_samples, 1))
end

huber_loss(x, δ) = abs(x) ≤ δ ? 0.5f0 * x * x : δ * (abs(x) - 0.5f0 * δ)

function build_loss_function(P_resid, G, S)
    return function (model, ps, st, data)
        X = data[1]
        prediction = model(X, ps, st)
        if prediction isa Tuple
            c_pred, st_out = prediction
        else
            c_pred = prediction
            st_out = st
        end
        if isnothing(S)
            a_grid_f32, _, c_pred_vec_f32 = det_residual_inputs(c_pred, G)
            resid = euler_resid_det_grid(P_resid, a_grid_f32, c_pred_vec_f32)
        else
            a_grid_f32, z_grid_f32, Pz_f32, _, c_pred_f32 =
                stoch_residual_inputs(c_pred, G, S)
            resid =
                euler_resid_stoch_grid(P_resid, a_grid_f32, z_grid_f32, Pz_f32, c_pred_f32)
        end
        loss = mean(huber_loss.(resid, 1.0f0))
        return loss, st_out, NamedTuple()
    end
end

function flatten_sum_squares(x)
    if x === nothing
        return 0.0
    elseif x isa Number
        return float(x)^2
    elseif x isa AbstractArray
        return sum(abs2, Float64.(x))
    elseif x isa NamedTuple || x isa Tuple || x isa Vector || x isa Dict
        s = 0.0
        for v in x
            s += flatten_sum_squares(v)
        end
        return s
    else
        try
            s = 0.0
            for f in fieldnames(typeof(x))
                s += flatten_sum_squares(getfield(x, f))
            end
            return s
        catch
            return 0.0
        end
    end
end

function create_training_batch(
    G,
    S,
    scaler::FeatureScaler;
    mode = :full,
    nsamples::Int = 0,
    rng = Random.GLOBAL_RNG,
)
    X, _ = generate_dataset(G, S; mode = mode, nsamples = nsamples, rng = rng)
    sample_count = size(X, 1)
    normalize_samples!(scaler, X)
    batch = prepare_training_batch(X)
    return batch, sample_count
end

function select_model(chain, state)
    return hasproperty(state, :model) ? getfield(state, :model) : chain
end

function state_parameters(state)
    if hasproperty(state, :parameters)
        return getfield(state, :parameters)
    elseif hasproperty(state, :params)
        return getfield(state, :params)
    else
        return nothing
    end
end

function state_states(state)
    if hasproperty(state, :states)
        return getfield(state, :states)
    elseif hasproperty(state, :state)
        return getfield(state, :state)
    else
        return nothing
    end
end

function run_model(model, params, states, X)
    # Expect the model to return a NamedTuple like (phi=..., h=...)
    output = params === nothing ? model(X) : model(X, params, states)

    # If model already returns a numeric prediction (backwards compatible), pass through
    if !(output isa NamedTuple)
        return output
    end

    # Adapter: compute predicted consumption `c` from Φ = c/w and input wealth `w`.
    # Input `X` is expected in Lux format: features × samples (e.g., 1×N for deterministic,
    # 2×N for stochastic where first row is asset a)
    Φ = output[:Φ]
    # Ensure Φ is an array with same shape as model heads (1×N or Nx1 depending on Lux)
    # Extract wealth/scale w from X: assume first feature is asset/wealth
    w = size(X, 1) >= 1 ? X[1, :] : ones(eltype(X), size(X, 2))

    # Align shapes: Φ may be 1×N (row) or N×1 (column) — convert to a 1×N row
    if ndims(Φ) == 2 && size(Φ, 1) == 1
        Φ_row = Φ
    elseif ndims(Φ) == 2 && size(Φ, 2) == 1
        Φ_row = permutedims(Φ)
    else
        # fallback: try to vec and reshape to 1×N
        Φ_row = reshape(vec(Φ), 1, :)
    end

    # Compute consumption c = Φ * w, result should be 1×N row vector to match previous c_pred
    c_row = Φ_row .* reshape(w, 1, :)
    return c_row
end

function train_consumption_network!(
    chain,
    settings::NNSolverSettings,
    scaler::FeatureScaler,
    P_resid,
    G,
    S,
)
    ps, st = Lux.setup(Random.GLOBAL_RNG, chain)
    opt = create_optimizer(settings)
    train_state = Lux.Training.TrainState(chain, ps, st, opt)
    loss_function = build_loss_function(P_resid, G, S)
    batch, sample_count = create_training_batch(G, S, scaler)
    total_samples = size(batch, 2)
    batch_size = compute_batch_size(total_samples, settings.batch_choice)
    batches_per_epoch = cld(total_samples, batch_size)
    best_state = train_state
    best_loss = Inf
    rng = Random.default_rng()
    stall_epochs = 0
    for epoch = 1:settings.epochs
        if settings.resample_interval > 0 && epoch % settings.resample_interval == 0
            batch, _ = create_training_batch(
                G,
                S,
                scaler;
                mode = :rand,
                nsamples = sample_count,
                rng = rng,
            )
            total_samples = size(batch, 2)
            batch_size = compute_batch_size(total_samples, settings.batch_choice)
            batches_per_epoch = cld(total_samples, batch_size)
        end
        shuffled = batch[:, randperm(rng, total_samples)]
        epoch_loss = 0.0
        seen = 0
        gradient_norm = NaN
        for start = 1:batch_size:total_samples
            stop = min(start + batch_size - 1, total_samples)
            data = (view(shuffled, :, start:stop),)
            ginfo, loss, _, train_state = Lux.Training.single_train_step!(
                Lux.AutoZygote(),
                loss_function,
                data,
                train_state,
            )
            nb = size(data[1], 2)
            epoch_loss += Float64(loss) * nb
            seen += nb
            try
                gradient_norm = sqrt(flatten_sum_squares(ginfo))
            catch
                gradient_norm = NaN
            end
        end
        average_loss = epoch_loss / max(seen, 1)
        if average_loss < best_loss
            best_loss = average_loss
            best_state = train_state
            stall_epochs = 0
        else
            stall_epochs += 1
        end
        if settings.verbose && (epoch % 10 == 0 || epoch == settings.epochs)
            @printf "Epoch: %3d \t Loss: %.5g \t GradNorm: %.5g\n" epoch average_loss gradient_norm
        end
        if best_loss ≤ settings.target_loss && stall_epochs ≥ settings.patience
            return TrainingResult(
                best_state,
                best_loss,
                epoch,
                batch_size,
                batches_per_epoch,
            )
        end
    end
    return TrainingResult(
        best_state,
        best_loss,
        settings.epochs,
        batch_size,
        batches_per_epoch,
    )
end
