import CUDA
import Adapt
import Zygote
import Adapt
using Lux: fmap

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
    samples_per_epoch::Int
    sigma_shocks::Union{Nothing,Float64}
    use_cuda::Bool
end

struct TrainingResult
    best_state::Any
    best_loss::Float64
    epochs_run::Int
    batch_size::Int
    batches_per_epoch::Int
end

maybe_to_device(x::Nothing, ::NNSolverSettings) = nothing
maybe_to_device(x, settings::NNSolverSettings) =
    settings.use_cuda ? Adapt.adapt(CUDA.CuArray, x) : x

maybe_to_host(x::Nothing, ::NNSolverSettings) = nothing
maybe_to_host(x, settings::NNSolverSettings) = settings.use_cuda ? Adapt.adapt(Array, x) : x

function maybe_to_host(state::Lux.Training.TrainState, settings::NNSolverSettings)
    mdl = hasproperty(state, :model) ? getfield(state, :model) : nothing
    ps = state_parameters(state)
    st = state_states(state)
    return (
        model = mdl,
        parameters = maybe_to_host(ps, settings),
        states = maybe_to_host(st, settings),
    )
end

function randn_like(rng, ref::CUDA.AbstractGPUArray)
    return CUDA.randn(eltype(ref), size(ref)...)
end
function randn_like(rng, ref)
    out = similar(ref)
    randn!(rng, out)
    return out
end
Zygote.@nograd randn_like

function fill_like(value, ref)
    if ref isa CUDA.AbstractGPUArray
        out = similar(ref)
        fill!(out, value)
        return out
    else
        return fill(value, size(ref))
    end
end

get_option(opts, key::Symbol, default) =
    opts === nothing ? default : (hasproperty(opts, key) ? getfield(opts, key) : default)

const CUDA_OBJECTIVE = :euler_fb_aio

maybe_objective_cuda(objective, default) = objective == CUDA_OBJECTIVE ? default : false

function detect_cuda_preference(objective, opts)
    requested = get_option(opts, :use_cuda, nothing)
    device_pref = get_option(opts, :device, nothing)
    gpu_available = try
        CUDA.functional()
    catch
        false
    end

    default_use_cuda = maybe_objective_cuda(objective, gpu_available)

    use_cuda =
        requested !== nothing ? Bool(requested) :
        device_pref === nothing ? default_use_cuda :
        begin
            dev_sym = Symbol(device_pref)
            if dev_sym === :auto
                default_use_cuda
            elseif dev_sym === :cuda
                true
            elseif dev_sym === :cpu
                false
            else
                throw(ArgumentError("Unknown device preference: $(device_pref)"))
            end
        end

    if use_cuda && !gpu_available
        @warn "CUDA requested but no functional GPU detected. Falling back to CPU." use_cuda =
            false
    end
    if use_cuda && objective != CUDA_OBJECTIVE
        @warn "CUDA acceleration currently supported only for objective :$(CUDA_OBJECTIVE); got :$(objective). Falling back to CPU." use_cuda =
            false
    end
    return use_cuda
end

function solver_settings(
    opts;
    has_shocks::Bool = false,
    objective_default::Symbol = :euler_fb_aio,
)
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
    objective = Symbol(get_option(opts, :objective, objective_default))
    # clamp v_h to a broader safe range [0.2, 5.0] to allow more tuning flexibility
    v_h = clamp(Float64(get_option(opts, :v_h, 0.5)), 0.2, 5.0)
    w_min = Float32(get_option(opts, :w_min, 0.1))
    w_max = Float32(get_option(opts, :w_max, 4.0))
    samples_per_epoch = max(Int(get_option(opts, :samples_per_epoch, 64)), 1)
    sigma_shocks = get_option(opts, :sigma_shocks, nothing)
    use_cuda = detect_cuda_preference(objective, opts)

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
        samples_per_epoch,
        sigma_shocks,
        use_cuda,
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

function build_loss_function(
    P_resid,
    G,
    S,
    scaler::FeatureScaler,
    settings::NNSolverSettings,
    rng::AbstractRNG,
    model_cfg = nothing,
)
    function fb_supported(model_cfg)
        model_cfg === nothing && return false
        P_full = model_cfg.P
        has_ar1 = hasproperty(P_full, :ρ_shock) && hasproperty(P_full, :σ_shock)
        has_var = hasproperty(P_full, :A) && hasproperty(P_full, :Σ)
        return has_ar1 || has_var
    end

    fb_warning_emitted = Ref(false)

    return function (model, ps, st, data)
        X = data[1]
        T = eltype(X)
        Rg = one(T) + T(P_resid.r)
        μ = T(P_resid.y)

        # If caller selected the FB AiO objective, delegate to the custom loss
        if settings.objective == :euler_fb_aio
            if !fb_supported(model_cfg)
                if !fb_warning_emitted[]
                    @warn "objective :euler_fb_aio requires active stochastic shocks; falling back to :euler_residual"
                    fb_warning_emitted[] = true
                end
            else
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
        end

        # Default Euler residual loss path (existing behaviour)
        prediction = model(X, ps, st)
        st_out = st

        # If the model returns the new NamedTuple (Φ, h), compute consumption c = Φ * w
        if prediction isa NamedTuple
            Φ = prediction.Φ
            h_raw = prediction.h

            mean_vals =
                ((X[1, :] .+ one(T)) ./ T(2)) .* T(scaler.mean_range) .+ T(scaler.mean_min)
            w = ((X[end, :] .+ one(T)) ./ T(2)) .* T(scaler.w_range) .+ T(scaler.w_min)
            y = mean_vals

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
            c_vec = vec(c_pred)
        elseif prediction isa Tuple
            c_pred, st_out = prediction
            c_pred = clamp.(c_pred, eps(eltype(X)), Inf)
            mean_vals =
                ((X[1, :] .+ one(T)) ./ T(2)) .* T(scaler.mean_range) .+ T(scaler.mean_min)
            w = ((X[end, :] .+ one(T)) ./ T(2)) .* T(scaler.w_range) .+ T(scaler.w_min)
            y = mean_vals
            c_vec = vec(c_pred)
        else
            c_pred = prediction
            c_pred = clamp.(c_pred, eps(eltype(X)), Inf)
            if size(X, 1) == 2
                y = ((X[1, :] .+ one(T)) ./ T(2)) .* T(scaler.y_range) .+ T(scaler.y_min)
                w = ((X[2, :] .+ one(T)) ./ T(2)) .* T(scaler.w_range) .+ T(scaler.w_min)
            elseif size(X, 1) == 1
                w = ((X[1, :] .+ one(T)) ./ T(2)) .* T(scaler.w_range) .+ T(scaler.w_min)
                y = fill(exp(μ), size(w))
            else
                throw(ArgumentError("Expected 1 or 2 feature rows, got $(size(X, 1))"))
            end
            c_vec = vec(c_pred)
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
            diag = (; phi = Φ_row, h = h_row, y = y, w = w, c = c_vec, a = w .- c_vec)
        else
            diag = (; phi = nothing, h = nothing, y = y, w = w, c = c_vec, a = w .- c_vec)
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
        # Avoid unnecessary host transfers: compute reductions on-device when possible.
        s = sum(abs2, x)
        return Float64(s)
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
    rng::AbstractRNG,
    P_resid = nothing,
    settings::Union{NNSolverSettings,Nothing} = nothing,
    P = nothing,
)
    want =
        nsamples > 0 ? nsamples :
        (isnothing(S) ? length(G[:a].grid) : length(G[:a].grid) * length(S.zgrid))
    if mode == :full
        base_P = P === nothing ? P_resid : P
        @assert base_P !== nothing
        X, _ = generate_dataset(G, S, base_P; mode = :full)
        normalize_samples!(scaler, X)
        return prepare_training_batch(X, Val(settings.use_cuda)), size(X, 1)
    end

    @assert P_resid !== nothing && settings !== nothing
    @assert want > 0 "create_training_batch requires a positive sample count"
    w_lo = settings.w_min
    w_hi = settings.w_max
    @assert w_hi > w_lo "Require w_max > w_min for cash-on-hand sampling"

    Rg = 1.0f0 + Float32(P_resid.r)
    base_P = P === nothing ? P_resid : P
    y_levels =
        hasproperty(base_P, :y) && base_P.y isa AbstractVector ?
        Float32.(collect(base_P.y)) : Float32[Float32(getfield(base_P, :y))]
    y_dim = length(y_levels)
    extra_cols = y_dim > 1 ? y_dim : 0
    feature_dim = 1 + extra_cols + 1
    a_min = Float32(G[:a].min)
    a_max = Float32(G[:a].max)
    if settings.has_shocks && !isnothing(S)
        z_min = Float32(minimum(S.zgrid))
        z_max = Float32(maximum(S.zgrid))
    else
        z_min = 0.0f0
        z_max = 0.0f0
    end

    mean_vec = Vector{Float32}(undef, want)
    component_mat =
        extra_cols > 0 ? Matrix{Float32}(undef, extra_cols, want) :
        Matrix{Float32}(undef, 1, want)
    W = Vector{Float32}(undef, want)
    filled = 0
    tries = 0
    max_tries = 1000
    while filled < want && tries < max_tries
        m = max(want - filled, 4096)
        a_draw = rand(rng, Float32, m) .* (a_max - a_min) .+ a_min
        z_draw = rand(rng, Float32, m) .* (z_max - z_min) .+ z_min
        if extra_cols > 0 &&
           hasproperty(base_P, :Σ) &&
           !isnothing(S) &&
           hasproperty(S, :process) &&
           S.process == :gaussian_linear
            Σ = Matrix{Float64}(base_P.Σ)
            chol = cholesky(Symmetric(Σ), check = false).L
            μ_vec = Float64.(y_levels)
            comps_tmp = Matrix{Float32}(undef, extra_cols, m)
            @inbounds for i = 1:m
                ε = randn(rng, Float64, y_dim)
                y_vec = μ_vec .+ chol * ε
                comps_tmp[:, i] .= Float32.(y_vec)
            end
            mean_draw = vec(mean(comps_tmp; dims = 1))
        else
            mean_draw = fill(Float32(mean(y_levels)), m)
            if extra_cols > 0
                comps_tmp = repeat(reshape(Float32.(y_levels), extra_cols, 1), 1, m)
            end
        end
        w_draw = @. Rg * a_draw + Float32(mean_draw)
        keep = (w_draw .>= w_lo) .& (w_draw .<= w_hi)
        k = count(keep)
        if k > 0
            idx = findall(keep)
            take = min(k, want - filled)
            mean_vec[filled+1:filled+take] .= mean_draw[idx[1:take]]
            W[filled+1:filled+take] .= w_draw[idx[1:take]]
            if extra_cols > 0
                component_mat[:, filled+1:filled+take] .= comps_tmp[:, idx[1:take]]
            else
                component_mat[1, filled+1:filled+take] .= mean_draw[idx[1:take]]
            end
            filled += take
        end
        tries += 1
    end
    @assert filled == want "Sampler could not hit the w-window; widen [w_min, w_max] or increase nsamples"

    X = Matrix{Float32}(undef, want, feature_dim)
    X[:, 1] .= mean_vec
    if extra_cols > 0
        for j = 1:extra_cols
            X[:, 1+j] .= component_mat[j, :]
        end
    end
    X[:, end] .= W
    normalize_samples!(scaler, X)
    batch = prepare_training_batch(X, Val(settings.use_cuda))
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
    rng::AbstractRNG,
    model_cfg = nothing,
)
    ps, st = Lux.setup(rng, chain)
    if settings.use_cuda
        ps = fmap(cu, ps)
        st = fmap(cu, st)
    else
        ps = fmap(identity, ps)
        st = fmap(identity, st)
    end
    opt = create_optimizer(settings)
    train_state = Lux.Training.TrainState(chain, ps, st, opt)
    # build loss with scaler so we can compute cash-on-hand inside the loss
    loss_function = build_loss_function(P_resid, G, S, scaler, settings, rng, model_cfg)
    # draw uniform cash-on-hand samples for the initial training batch
    samples_per_epoch = settings.samples_per_epoch
    batch, sample_count = create_training_batch(
        G,
        S,
        scaler;
        mode = :rand,
        nsamples = samples_per_epoch,
        rng = rng,
        P_resid = P_resid,
        settings = settings,
        P = model_cfg === nothing ? nothing : model_cfg.P,
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
        P = model_cfg === nothing ? nothing : model_cfg.P,
    )
    total_samples = size(batch, 2)
    batch_size = compute_batch_size(total_samples, settings.batch_choice)
    if settings.use_cuda
        batch_size = total_samples        # 100% batch
        batches_per_epoch = 1
    end
    # For stochastic problems we require predictions on the full grid
    # (Na * Nz) so force full-batch training when shocks are present.
    if !isnothing(S) && batch_size < total_samples
        batch_size = total_samples
    end
    batches_per_epoch = cld(total_samples, batch_size)
    best_state = train_state
    best_loss = Inf
    stall_epochs = 0
    for epoch = 1:settings.epochs
        if settings.resample_interval > 0 && epoch % settings.resample_interval == 0
            batch, _ = create_training_batch(
                G,
                S,
                scaler;
                mode = :rand,
                nsamples = samples_per_epoch,
                rng = rng,
                P_resid = P_resid,
                settings = settings,
                P = model_cfg === nothing ? nothing : model_cfg.P,
            )
            batch = maybe_to_device(batch, settings)
            total_samples = size(batch, 2)
            batch_size = compute_batch_size(total_samples, settings.batch_choice)
            # same rule: force full-batch when stochastic
            if !isnothing(S) && batch_size < total_samples
                batch_size = total_samples
            end
            batches_per_epoch = cld(total_samples, batch_size)
        end
        shuffled = settings.use_cuda ? batch : batch[:, randperm(rng, total_samples)]
        epoch_loss = 0.0
        seen = 0
        gradient_norm = NaN
        for start = 1:batch_size:total_samples
            stop = min(start + batch_size - 1, total_samples)
            data = (shuffled[:, start:stop],)
            ginfo, loss, _, train_state = Lux.Training.single_train_step!(
                Lux.AutoZygote(),
                loss_function,
                data,
                train_state,
            )
            nb = size(data[1], 2)
            @assert loss isa Number
            loss_value = Float64(loss)
            epoch_loss += Float64(loss_value) * nb
            seen += nb
            if !settings.use_cuda
                try
                    # on CPU we can compute gradient norm exactly
                    gradient_norm = sqrt(flatten_sum_squares(ginfo))
                catch
                    gradient_norm = NaN
                end
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
        if settings.verbose && (epoch % 100 == 0 || epoch == settings.epochs)
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
            stored_state = maybe_to_host(best_state, settings)
            return TrainingResult(
                stored_state,
                best_loss,
                epoch,
                batch_size,
                batches_per_epoch,
            )
        end
    end
    stored_state = maybe_to_host(best_state, settings)
    return TrainingResult(
        stored_state,
        best_loss,
        settings.epochs,
        batch_size,
        batches_per_epoch,
    )
end
