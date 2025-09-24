struct NNSolverSettings
    epochs::Int
    batch_choice::Union{Nothing,Int}
    learning_rate::Float64
    verbose::Bool
    resample_interval::Int
    target_loss::Float32
    patience::Int
    hidden_sizes::NTuple{2,Int}
    has_shocks::Bool
    objective::Symbol
    v_h::Float64
    w_min::Float32
    w_max::Float32
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

function solver_settings(opts; has_shocks::Bool = false)
    epochs = max(Int(get_option(opts, :epochs, 1000)), 0)
    batch_choice = get_option(opts, :batch, 64)
    batch_choice = isnothing(batch_choice) ? nothing : max(Int(batch_choice), 1)
    learning_rate = Float64(get_option(opts, :lr, 1e-3))
    verbose = Bool(get_option(opts, :verbose, false))
    # resample every epoch by default for stability
    resample_interval = max(Int(get_option(opts, :resample_every, 1)), 0)
    target_loss = Float32(get_option(opts, :target_loss, 2e-4))
    patience = max(Int(get_option(opts, :patience, 200)), 0)
    hid1 = max(Int(get_option(opts, :hid1, 128)), 1)
    hid2 = max(Int(get_option(opts, :hid2, 128)), 1)
    objective = Symbol(get_option(opts, :objective, :euler))
    # clamp v_h to recommended range [0.5, 2.0] for stability and balancing
    v_h = clamp(Float64(get_option(opts, :v_h, 1.0)), 0.5, 2.0)
    w_min = Float32(get_option(opts, :w_min, 0.1))
    w_max = Float32(get_option(opts, :w_max, 4.0))
    return NNSolverSettings(
        epochs,
        batch_choice,
        learning_rate,
        verbose,
        resample_interval,
        target_loss,
        patience,
        (hid1, hid2),
        has_shocks,
        objective,
        v_h,
        w_min,
        w_max,
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

@inline function cash_on_hand(a, z, P, has_shocks::Bool)
    R = 1.0f0 + Float32(P.r)
    if has_shocks
        # income from shock state; simple AR(1) level. Adjust if you use exp(z).
        y = Float32(P.y) .+ z
    else
        y = Float32(P.y)
    end
    return @. R * a + y
end

function build_loss_function(
    P_resid,
    G,
    S,
    scaler::FeatureScaler,
    settings::NNSolverSettings,
    model_cfg = nothing,
    rng = Random.GLOBAL_RNG,
)
    return function (model, ps, st, data)
        X = data[1]

        # If caller selected the FB AiO objective, delegate to the custom loss
        if settings.objective == :euler_fb_aio
            # loss_euler_fb_aio! returns (loss, (st1, aux_namedtuple))
            loss_val, st_pack = loss_euler_fb_aio!(model, ps, st, X, model_cfg, rng)
            st1, aux = st_pack
            # package diagnostics: include FB aux diagnostics and leave phi/h fields empty
            diag = (;
                phi = nothing,
                h = nothing,
                a = nothing,
                z = nothing,
                w = nothing,
                c = nothing,
                fb = aux,
            )
            return loss_val, st1, diag
        end

        # Default Euler residual loss path (existing behaviour)
        prediction = model(X, ps, st)
        st_out = st

        # If the model returns the new NamedTuple (Φ, h), compute consumption c = Φ * w
        if prediction isa NamedTuple
            Φ = prediction[:Φ]
            h_raw = prediction[:h]

            # Recover original (unnormalized) a and z from normalized input X
            a = ((X[1, :] .+ 1.0f0) ./ 2.0f0) .* scaler.a_range .+ scaler.a_min
            if scaler.has_shocks
                z = ((X[2, :] .+ 1.0f0) ./ 2.0f0) .* scaler.z_range .+ scaler.z_min
            else
                z = zeros(eltype(a), size(a))
            end

            w = cash_on_hand(a, z, P_resid, scaler.has_shocks)

            # Align shapes: Φ and h may be 1×N (row) or N×1 (column)
            if ndims(Φ) == 2 && size(Φ, 1) == 1
                Φ_row = Φ
            elseif ndims(Φ) == 2 && size(Φ, 2) == 1
                Φ_row = permutedims(Φ)
            else
                Φ_row = reshape(vec(Φ), 1, :)
            end
            if ndims(h_raw) == 2 && size(h_raw, 1) == 1
                h_row = h_raw
            elseif ndims(h_raw) == 2 && size(h_raw, 2) == 1
                h_row = permutedims(h_raw)
            else
                h_row = reshape(vec(h_raw), 1, :)
            end

            c_pred = Φ_row .* reshape(w, 1, :)
            # avoid u'(0) by clamping consumption away from zero
            c_pred = clamp.(c_pred, eps(eltype(X)), Inf)
        elseif prediction isa Tuple
            c_pred, st_out = prediction
            c_pred = clamp.(c_pred, eps(eltype(X)), Inf)
        else
            c_pred = prediction
            c_pred = clamp.(c_pred, eps(eltype(X)), Inf)
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

        # Build diagnostics NamedTuple for minibatch (phi, h, a, z, w, c)
        if prediction isa NamedTuple
            diag = (;
                phi = Φ_row,
                h = h_row,
                a = a,
                z = settings.has_shocks ? z : nothing,
                w = w,
                c = c_pred,
            )
        else
            # fallback diagnostics when model returned c directly
            # compute a,z,w for diagnostics
            a = ((X[1, :] .+ 1.0f0) ./ 2.0f0) .* scaler.a_range .+ scaler.a_min
            if scaler.has_shocks
                z = ((X[2, :] .+ 1.0f0) ./ 2.0f0) .* scaler.z_range .+ scaler.z_min
            else
                z = nothing
            end
            w =
                scaler.has_shocks ? cash_on_hand(a, z, P_resid, scaler.has_shocks) :
                cash_on_hand(a, 0.0f0, P_resid, false)
            diag = (;
                phi = nothing,
                h = nothing,
                a = a,
                z = settings.has_shocks ? z : nothing,
                w = w,
                c = c_pred,
            )
        end

        return loss, st_out, diag
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
    P_resid = nothing,
    settings::Union{NNSolverSettings,Nothing} = nothing,
)
    # Generate raw dataset samples (rows = samples)
    X, _ = generate_dataset(G, S; mode = mode, nsamples = nsamples, rng = rng)

    # If requested, filter the sampled rows so cash-on-hand w lies in [w_min, w_max]
    if !(mode == :full) && P_resid !== nothing && settings !== nothing
        # compute w for each sample row
        if settings.has_shocks
            a_samples = X[:, 1]
            z_samples = X[:, 2]
            w = cash_on_hand(a_samples, z_samples, P_resid, true)
        else
            a_samples = X[:, 1]
            w = cash_on_hand(a_samples, 0.0f0, P_resid, false)
        end
        # mask rows whose w is within [w_min, w_max]
        mask = map(x -> x >= settings.w_min && x <= settings.w_max, w)
        if any(mask)
            X = X[mask, :]
        else
            # no samples within the requested window: fall back to original X
        end
    end

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
    # Call the model (Lux may call with or without params/states). If the
    # model returns a Tuple like `(prediction, state)` unwrap and return the
    # prediction (first element) to maintain backwards compatibility with
    # callers that expect the raw prediction array.
    out = params === nothing ? model(X) : model(X, params, states)
    return out isa Tuple ? out[1] : out
end

function train_consumption_network!(
    chain,
    settings::NNSolverSettings,
    scaler::FeatureScaler,
    P_resid,
    G,
    S,
    model_cfg = nothing,
    rng = Random.GLOBAL_RNG,
)
    ps, st = Lux.setup(Random.GLOBAL_RNG, chain)
    opt = create_optimizer(settings)
    train_state = Lux.Training.TrainState(chain, ps, st, opt)
    # build loss with scaler so we can compute cash-on-hand inside the loss
    loss_function = build_loss_function(P_resid, G, S, scaler, settings, model_cfg, rng)
    batch, sample_count = create_training_batch(
        G,
        S,
        scaler;
        mode = :full,
        nsamples = 0,
        rng = Random.GLOBAL_RNG,
        P_resid = P_resid,
        settings = settings,
    )
    total_samples = size(batch, 2)
    batch_size = compute_batch_size(total_samples, settings.batch_choice)
    # For stochastic problems we require predictions on the full grid
    # (Na * Nz) so force full-batch training when shocks are present.
    if !isnothing(S) && batch_size < total_samples
        batch_size = total_samples
    end
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
                P_resid = P_resid,
                settings = settings,
            )
            total_samples = size(batch, 2)
            batch_size = compute_batch_size(total_samples, settings.batch_choice)
            # same rule: force full-batch when stochastic
            if !isnothing(S) && batch_size < total_samples
                batch_size = total_samples
            end
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
