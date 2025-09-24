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
    Optimisers.ClipGrad(0.02),
    Optimisers.Adam(settings.learning_rate),
)

function compute_batch_size(total_samples::Int, choice::Union{Nothing,Int})
    return isnothing(choice) ? max(total_samples, 1) :
           clamp(choice, 1, max(total_samples, 1))
end

huber_loss(x, δ) = abs(x) ≤ δ ? 0.5f0 * x * x : δ * (abs(x) - 0.5f0 * δ)

@inline function cash_on_hand(a, z, P, has_shocks::Bool)
    R = 1.0f0 + Float32(P.r)
    μ = Float32(P.y)                # log-mean income level
    if has_shocks
        inc = @. exp(μ + z)
    else
        inc = exp(μ)
    end
    return @. R * a + inc
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
    mode = :rand,
    nsamples::Int = 4096,
    rng = Random.GLOBAL_RNG,
    P_resid = nothing,
    settings::Union{NNSolverSettings,Nothing} = nothing,
)
    want =
        nsamples > 0 ? nsamples :
        (isnothing(S) ? length(G[:a].grid) : length(G[:a].grid) * length(S.zgrid))

    # deterministic full grid path unchanged
    if mode == :full
        X, _ = generate_dataset(G, S; mode = :full)
        normalize_samples!(scaler, X)
        return prepare_training_batch(X), size(X, 1)
    end

    @assert P_resid !== nothing && settings !== nothing
    Rg = 1.0f0 + Float32(P_resid.r)
    μ = Float32(P_resid.y)
    a_min = Float32(G[:a].min)
    a_max = Float32(G[:a].max)
    z_min = scaler.z_min
    z_max = scaler.z_min + scaler.z_range
    w_lo = settings.w_min
    w_hi = settings.w_max

    A = Vector{Float32}(undef, want)
    Z = settings.has_shocks ? Vector{Float32}(undef, want) : Float32[]
    filled = 0
    max_tries = 1000
    tries = 0
    while filled < want && tries < max_tries
        # draw in bulk
        m = max(want - filled, 4096)
        a = rand(rng, Float32, m) .* (a_max - a_min) .+ a_min
        z =
            settings.has_shocks ? rand(rng, Float32, m) .* (z_max - z_min) .+ z_min :
            fill(0.0f0, m)
        # lognormal income
        inc = @. exp(μ + z)
        w = @. Rg * a + inc
        keep = (w .>= w_lo) .& (w .<= w_hi)
        k = count(keep)
        if k > 0
            idx = findall(keep)
            take = min(k, want - filled)
            A[filled+1:filled+take] .= a[idx[1:take]]
            if settings.has_shocks
                Z[filled+1:filled+take] .= z[idx[1:take]]
            end
            filled += take
        end
        tries += 1
    end
    @assert filled == want "Sampler could not hit the w-window; widen [w_min,w_max] or raise nsamples."

    X = settings.has_shocks ? hcat(A, Z) : reshape(A, :, 1)
    normalize_samples!(scaler, X)
    batch = prepare_training_batch(X)
    return batch, want
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
    # use the strict rejection sampler to build an initial training pool large enough
    init_nsamples =
        max(settings.batch_choice === nothing ? 64 : settings.batch_choice, 64) * 128
    batch, sample_count = create_training_batch(
        G,
        S,
        scaler;
        mode = :rand,
        nsamples = init_nsamples,
        rng = Random.GLOBAL_RNG,
        P_resid = P_resid,
        settings = settings,
    )
    # create a fixed validation batch for periodic diagnostics (held out)
    val_nsamples = min(4096, sample_count)
    val_batch, _ = create_training_batch(
        G,
        S,
        scaler;
        mode = :rand,
        nsamples = val_nsamples,
        rng = rng,
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
        # periodic validation logging every 100 epochs
        if settings.verbose && epoch % 100 == 0
            try
                # loss_function returns (loss, st_out, diag) for the outer training API
                val_loss, val_st, val_diag = loss_function(
                    select_model(chain, train_state),
                    state_parameters(train_state),
                    state_states(train_state),
                    (val_batch,),
                )
                if :fb in keys(val_diag)
                    aux = val_diag.fb
                    try
                        @printf(
                            "[VAL] Epoch %4d: kt_mean=%.6g aio_mean=%.6g max_abs_q=%.6g\n",
                            epoch,
                            aux.kt_mean,
                            aux.aio_mean,
                            aux.max_abs_q
                        )
                    catch
                        @printf(
                            "[VAL] Epoch %4d: diagnostics present but failed to print (missing fields)\n",
                            epoch
                        )
                    end
                else
                    @printf(
                        "[VAL] Epoch %4d: validation loss=%.6g (no FB diagnostics)\n",
                        epoch,
                        val_loss
                    )
                end
            catch err
                @warn "Validation logging failed" error = err
            end
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
