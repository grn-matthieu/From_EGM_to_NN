import CUDA
import Adapt
import Zygote
import Adapt
using Lux: fmap
using ChainRulesCore: ignore_derivatives, AbstractZero
using LinearAlgebra: diag, dot, I
using Optimisers
using Statistics: mean, std
using Printf
using ..DataNN: sample_training

include("preprocessing.jl")

include("training_utils.jl")
using .NNTrainUtils:
    base_optimizer,
    create_optimizer,
    cosine_learning_rate,
    exponential_learning_rate,
    learning_rate_for_epoch,
    adjust_learning_rate,
    apply_optimizer_learning_rate!,
    bcmc_auto_update!

struct NNSolverSettings
    epochs::Int
    batch_choice::Union{Nothing,Int}
    learning_rate::Float64
    optimizer::Symbol
    lr_min::Float64
    lr_max::Float64
    lr_decay_horizon::Int
    lr_schedule::Symbol
    lr_gamma::Float64
    lr_milestones::Vector{Int}
    warmup_epochs::Int
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
    n_mc::Int
    bcmc_budget_T::Union{Nothing,Int}
    bcmc_auto_N::Bool
    bcmc_update_every::Int
end

struct TrainingResult
    best_state::Any
    best_loss::Float64
    epochs_run::Int
    batch_size::Int
    batches_per_epoch::Int
    rmse_history::Vector{Float64}
end


# Loss / noise helpers moved to `losses.jl` (NNLosses). Those symbols are
# imported into this module scope via `kernel.jl` so they are available here.



function solver_settings(
    opts,
    P = nothing,
    G = nothing,
    S = nothing;
    has_shocks::Bool = false,
    objective_default::Symbol = :euler_fb_aio,
)
    epochs = max(Int(get_option(opts, :epochs, 1000)), 0)
    batch_choice = get_option(opts, :batch, 64)
    batch_choice = isnothing(batch_choice) ? nothing : max(Int(batch_choice), 1)
    base_lr = Float64(get_option(opts, :lr, 1e-3))
    lr_max = Float64(get_option(opts, :lr_max, base_lr))
    lr_min = Float64(get_option(opts, :lr_min, min(base_lr, lr_max)))
    if lr_min > lr_max
        @warn "Configured lr_min=$(lr_min) exceeds lr_max=$(lr_max); swapping values"
        lr_min, lr_max = lr_max, lr_min
    end
    warmup_epochs = max(Int(get_option(opts, :warmup_epochs, 0)), 0)
    lr_decay_horizon =
        max(Int(get_option(opts, :lr_decay_horizon, epochs - warmup_epochs)), 0)
    schedule_raw = Symbol(get_option(opts, :lr_schedule, :cosine))
    lr_schedule = schedule_raw in (:cosine, :exponential) ? schedule_raw : :cosine
    if schedule_raw ∉ (:cosine, :exponential)
        @warn "Unknown lr_schedule=$(schedule_raw); defaulting to :cosine"
    end
    gamma_opt = get_option(opts, :lr_gamma, nothing)
    lr_gamma = gamma_opt === nothing ? NaN : Float64(gamma_opt)
    if !isnan(lr_gamma) && lr_gamma <= 0
        @warn "Configured lr_gamma=$(lr_gamma) must be positive; ignoring value"
        lr_gamma = NaN
    end
    milestones_opt = get_option(opts, :lr_milestones, nothing)
    lr_milestones = Int[]
    if milestones_opt !== nothing
        try
            vals = Int.(collect(milestones_opt))
            vals = filter(>=(1), vals)
            sort!(vals)
            unique!(vals)
            lr_milestones = vals
        catch err
            @warn "Failed to parse lr_milestones=$(milestones_opt); ignoring" err = err
            lr_milestones = Int[]
        end
    end
    if lr_schedule === :exponential
        if isempty(lr_milestones)
            steps = max(lr_decay_horizon, 1)
            if isnan(lr_gamma)
                if lr_max > lr_min && steps > 0
                    lr_gamma = (lr_min / lr_max)^(1 / steps)
                else
                    lr_gamma = 1.0
                end
            end
        else
            if isnan(lr_gamma)
                if lr_max > lr_min
                    lr_gamma = clamp(lr_min / lr_max, floatmin(Float64), 1.0)
                else
                    lr_gamma = 1.0
                end
            end
        end
        lr_gamma = clamp(lr_gamma, floatmin(Float64), Inf)
    else
        lr_gamma = 1.0
        lr_milestones = Int[]
    end
    learning_rate = lr_max
    verbose = Bool(get_option(opts, :verbose, false))
    # resample every epoch by default for stability
    resample_interval = Int(get_option(opts, :resample_every, 1))
    target_loss = Float32(get_option(opts, :target_loss, 1e-4))
    # Default: disable early stopping unless explicitly requested
    patience = max(Int(get_option(opts, :patience, 0)), 0)
    hid1 = max(Int(get_option(opts, :hid1, 128)), 1)
    hid2 = max(Int(get_option(opts, :hid2, 128)), 1)
    objective = Symbol(get_option(opts, :objective, objective_default))
    # clamp v_h to a broader safe range [0.2, 5.0] to allow more tuning flexibility
    v_h = clamp(Float64(get_option(opts, :v_h, 0.5)), 0.2, 5.0)

    w_min = Float32(get_option(opts, :w_min, 0.1))
    w_max = Float32(get_option(opts, :w_max, 4.0))
    samples_per_epoch = max(Int(get_option(opts, :samples_per_epoch, 64)), 1)
    sigma_shocks = get_option(opts, :sigma_shocks, nothing)
    use_cuda = get_option(opts, :use_cuda, false)
    n_mc = max(Int(get_option(opts, :n_mc, 16)), 1)
    bcmc_budget_T = let v = get_option(opts, :bcmc_budget_T, nothing)
        v === nothing ? nothing : Int(v)
    end
    bcmc_auto_N = Bool(get_option(opts, :bcmc_auto_N, false))
    bcmc_update_every = max(Int(get_option(opts, :bcmc_update_every, 10)), 1)

    # If Auto-N is requested but no explicit budget provided, derive a
    # default pairwise budget from the initial configuration to keep total
    # work roughly constant across updates. We use the samples_per_epoch as a
    # proxy for the minibatch size M and pairs ~ N(N-1)/2.
    if bcmc_auto_N && bcmc_budget_T === nothing
        pairs0 = max(div(n_mc * (n_mc - 1), 2), 1)
        bcmc_budget_T = samples_per_epoch * pairs0
        @info "Auto-N active without explicit budget: deriving bcmc_budget_T=$(bcmc_budget_T) from n_mc=$(n_mc) and M≈$(samples_per_epoch)"
    end

    if objective === :euler_fb_bcmc
        if n_mc < 2
            throw(ArgumentError("objective :euler_fb_bcmc requires n_mc ≥ 2 (got $(n_mc))"))
        elseif n_mc == 2
            @info "bc-MC with n_mc=2 is equivalent to AiO"
        end
    end

    optimizer_raw = get_option(opts, :optimizer, :adamw)
    optimizer = try
        Symbol(lowercase(String(optimizer_raw)))
    catch err
        @warn "Failed to parse optimizer option $(optimizer_raw); defaulting to :adamw" err =
            err
        :adamw
    end
    function canonical_optimizer(sym)
        if sym === :adam
            return :adam
        elseif sym in (:rmsprop, :rms_prop)
            return :rmsprop
        elseif sym in (:adagrad, :ada_grad)
            return :adagrad
        elseif sym in (:sgd, :descent)
            return :sgd
        elseif sym in (:adamw, :adam_w)
            return :adamw
        else
            return nothing
        end
    end
    canonical = canonical_optimizer(optimizer)
    supported_optimizers = (:adam, :rmsprop, :adagrad, :sgd, :adamw)
    if canonical === nothing
        @warn "Unknown optimizer=$(optimizer_raw); supported options are $(collect(supported_optimizers))" optimizer =
            :adam
    else
        optimizer = canonical
    end

    return NNSolverSettings(
        epochs,
        batch_choice,
        learning_rate,
        optimizer,
        lr_min,
        lr_max,
        lr_decay_horizon,
        lr_schedule,
        lr_gamma,
        lr_milestones,
        warmup_epochs,
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
        n_mc,
        bcmc_budget_T,
        bcmc_auto_N,
        bcmc_update_every,
    )
end

function build_network(input_dim::Int, settings::NNSolverSettings)
    h1, h2 = settings.hidden_sizes
    return Chain(Dense(input_dim, h1, relu), Dense(h1, h2, relu), Dense(h2, 1, softplus))
end





# Loss builder and FB objective dispatchers were moved to `losses.jl` as
# the `NNLosses` module. Symbols are imported by `kernel.jl` so they're
# available in this scope.


# Loss-related helpers (flatten_sum_squares, variance estimators, linearized
# component estimators and small feature-shift helpers) were moved to
# `losses.jl` (module `NNLosses`). They are imported into the containing
# module namespace by `kernel.jl` so they remain available here.



# General-purpose helpers (option parsing, state accessors, small wrappers)
# have been moved to `utils.jl` to keep the training loop focused on the
# optimization flow. They are included into the `NNKernel` module so they
# remain available here.

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
    end
    opt = create_optimizer(settings)
    train_state = Lux.Training.TrainState(chain, ps, st, opt)
    # build loss with scaler so we can compute cash-on-hand inside the loss
    loss_function = build_loss_function(P_resid, G, S, scaler, settings, rng, model_cfg)
    # draw uniform cash-on-hand samples for the initial training batch
    samples_per_epoch = settings.samples_per_epoch
    X_train, sample_count = sample_training(
        G,
        S,
        P_resid;
        mode = :rand,
        nsamples = samples_per_epoch,
        rng = rng,
        settings = settings,
        P = model_cfg === nothing ? nothing : model_cfg.P,
    )
    normalize_samples!(scaler, X_train)
    batch = prepare_training_batch(X_train, Val(settings.use_cuda))
    # create a fixed validation batch for periodic diagnostics (held out)
    val_nsamples = min(4096, sample_count)
    validation_rng = derive_rng(rng, settings.epochs + 1)
    X_val, _ = sample_training(
        G,
        S,
        P_resid;
        mode = :rand,
        nsamples = val_nsamples,
        rng = validation_rng,
        settings = settings,
        P = model_cfg === nothing ? nothing : model_cfg.P,
    )
    normalize_samples!(scaler, X_val)
    val_batch = prepare_training_batch(X_val, Val(settings.use_cuda))
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
    current_lr = NaN

    # Track previous policy for convergence check
    prev_policy = nothing
    convergence_check_batch = nothing
    rmse_history = Float64[]

    for epoch = 1:settings.epochs
        new_lr = learning_rate_for_epoch(settings, epoch)
        if !(
            isfinite(current_lr) && isapprox(new_lr, current_lr; atol = 1e-12, rtol = 1e-6)
        )
            # apply_optimizer_learning_rate! may return a new TrainState when using
            # Lux.Training.TrainState (immutable). Capture and reassign so the
            # optimizer update takes effect.
            train_state = apply_optimizer_learning_rate!(train_state, new_lr)
            current_lr = new_lr
        end
        if epoch % settings.resample_interval == 0
            epoch_rng = derive_rng(rng, epoch)
            X_epoch, _ = sample_training(
                G,
                S,
                P_resid;
                mode = :rand,
                nsamples = samples_per_epoch,
                rng = epoch_rng,
                settings = settings,
                P = model_cfg === nothing ? nothing : model_cfg.P,
            )
            normalize_samples!(scaler, X_epoch)
            batch = prepare_training_batch(X_epoch, Val(settings.use_cuda))
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

            if settings.objective === :euler_fb_bcmc &&
               settings.bcmc_auto_N &&
               model_cfg !== nothing &&
               isdefined(model_cfg, :bcmc_state)
                step_id = (epoch - 1) * batches_per_epoch + cld(stop, batch_size)
                # Delegate the probing and possible update to the helper module
                try
                    bcmc_auto_update!(
                        chain,
                        train_state,
                        settings,
                        model_cfg,
                        data[1],
                        scaler,
                        rng,
                        step_id,
                        settings.samples_per_epoch,
                    )
                catch err
                    if settings.verbose
                        @warn "BCMC auto-N probe failed" err = err
                    end
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

        # Adaptive v_h: Monitor complementarity term and adjust Euler weight
        # Strategy: As kt_mean decreases (complementarity satisfied), increase v_h
        # to focus more on Euler equation accuracy
        if settings.objective === :euler_fb_aio &&
           model_cfg !== nothing &&
           isdefined(model_cfg, :adaptive_v_h_state) &&
           epoch % 10 == 0
            # Get current kt_mean from a quick forward pass on validation batch
            try
                val_loss, val_st, val_diag = loss_function(
                    select_model(chain, train_state),
                    state_parameters(train_state),
                    state_states(train_state),
                    (val_batch,),
                )
                if :fb in keys(val_diag)
                    aux = val_diag.fb
                    if isdefined(aux, :kt_mean)
                        current_kt = Float64(aux.kt_mean)
                        vh_state = model_cfg.adaptive_v_h_state

                        # Update EMA of kt_mean
                        kt_ema =
                            vh_state.ema_alpha * current_kt +
                            (1 - vh_state.ema_alpha) * vh_state.kt_ema[]
                        vh_state.kt_ema[] = kt_ema

                        # Adaptive schedule: v_h increases as kt decreases
                        # When kt < 1e-5: v_h = initial * 10 (focus on Euler)
                        # When kt > 1e-3: v_h = initial * 1 (balanced)
                        kt_threshold_low = 1e-5
                        kt_threshold_high = 1e-3

                        if kt_ema < kt_threshold_low
                            scale_factor = 10.0
                        elseif kt_ema < kt_threshold_high
                            # Logarithmic interpolation in transition zone
                            log_ratio =
                                log10(kt_ema / kt_threshold_low) /
                                log10(kt_threshold_high / kt_threshold_low)
                            scale_factor = 1.0 + 9.0 * (1.0 - log_ratio)
                        else
                            scale_factor = 1.0
                        end

                        new_v_h = vh_state.initial_v_h * scale_factor
                        old_v_h = vh_state.v_h[]

                        # Update if changed significantly (>10%)
                        if abs(new_v_h - old_v_h) > 0.1 * old_v_h
                            vh_state.v_h[] = new_v_h
                            if settings.verbose
                                @printf "[ADAPT_V_H] Epoch %4d: kt_ema=%.2e → v_h: %.3f→%.3f (×%.2f)\n" epoch kt_ema old_v_h new_v_h scale_factor
                            end
                        end
                    end
                end
            catch err
                # If adaptive v_h update fails, continue silently
            end
        end

        if settings.verbose && (epoch % 100 == 0 || epoch == settings.epochs)
            @printf "Epoch: %3d \t Loss: %.5g \t GradNorm: %.5g\n" epoch average_loss gradient_norm
        end

        # Check convergence every 100 epochs: evaluate Euler errors and policy updates
        if epoch % 100 == 0
            try
                # Create/reuse held-out batch for convergence check
                if convergence_check_batch === nothing
                    X_check, _ = sample_training(
                        G,
                        S,
                        P_resid;
                        mode = :rand,
                        nsamples = min(2048, samples_per_epoch),
                        rng = rng,
                        settings = settings,
                        P = model_cfg === nothing ? nothing : model_cfg.P,
                    )
                    normalize_samples!(scaler, X_check)
                    convergence_check_batch =
                        prepare_training_batch(X_check, Val(settings.use_cuda))
                end

                # Evaluate current policy on held-out grid
                current_model = select_model(chain, train_state)
                current_ps = state_parameters(train_state)
                current_st = state_states(train_state)

                # Extract policy predictions
                out, _ = Lux.apply(
                    current_model,
                    convergence_check_batch,
                    current_ps,
                    current_st,
                )
                w_batch = convergence_check_batch[end, :]
                if settings.use_cuda
                    w_batch = cu(w_batch)
                end
                w_denorm = ((w_batch .+ 1.0f0) ./ 2.0f0) .* scaler.w_range .+ scaler.w_min
                current_c = vec(phi_to_consumption(out[:Φ], w_denorm; min_c = 1.0f-3))

                # Compute policy update sup-norm if we have previous policy
                Δ_pol = Inf
                if prev_policy !== nothing
                    Δ_pol = maximum(abs.(current_c .- prev_policy))
                end
                prev_policy = copy(current_c)

                # Evaluate Euler residuals using GH quadrature (for stochastic) or grid (for deterministic)
                euler_rmse = Inf
                if settings.has_shocks &&
                   G !== nothing &&
                   S !== nothing &&
                   model_cfg !== nothing &&
                   model_cfg.P !== nothing
                    # Use GH quadrature for stochastic case
                    gh_result = eval_euler_residuals_gh(
                        current_model,
                        current_ps,
                        current_st,
                        P_resid,
                        model_cfg.U,
                        scaler,
                        settings;
                        N = min(2048, samples_per_epoch),
                        rng = rng,
                        G = G,
                        S = S,
                        P = model_cfg.P,
                    )
                    # Compute RMSE of Euler residuals
                    if isdefined(gh_result, :abs_resid)
                        euler_rmse = sqrt(mean(Float64.(gh_result.abs_resid) .^ 2))
                    elseif isdefined(gh_result, :stats) && isdefined(gh_result.stats, :rmse)
                        euler_rmse = Float64(gh_result.stats.rmse)
                    end
                else
                    # Deterministic case: use grid-based residuals
                    if isdefined(P_resid, :a) && isdefined(model_cfg.G, :a)
                        a_grid_f32 = Float32.(model_cfg.G.a.grid)
                        c_pred_vec_f32 = current_c[1:length(a_grid_f32)]
                        residuals = euler_resid_grid(P_resid, a_grid_f32, c_pred_vec_f32)
                        euler_rmse = sqrt(mean(Float64.(residuals) .^ 2))
                    end
                end

                # Track RMSE history
                push!(rmse_history, euler_rmse)

                # Check dual convergence criteria
                converged = (euler_rmse < 1e-4) && (Δ_pol < 1e-6)

                if settings.verbose
                    @printf "[CONV_CHECK] Epoch %4d: RMSE(R)=%.6g (tol=1e-4) Δ∞=%.6g (tol=1e-6) %s\n" epoch euler_rmse Δ_pol (
                        converged ? "✓ CONVERGED" : ""
                    )
                end

                if converged && epoch >= 100  # Require at least 100 epochs before converging
                    if settings.verbose
                        @printf "[CONVERGED] Both criteria met at epoch %d\n" epoch
                    end
                    stored_state = maybe_to_host(train_state, settings)
                    return TrainingResult(
                        stored_state,
                        euler_rmse,
                        epoch,
                        batch_size,
                        batches_per_epoch,
                        rmse_history,
                    )
                end
            catch err
                # If convergence check fails, continue training
                if settings.verbose
                    @warn "Convergence check failed at epoch $epoch" exception =
                        (err, catch_backtrace())
                end
            end
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

                # Delegate verbose printing to shared helper
                verbose_validation_logging(
                    epoch,
                    settings,
                    chain,
                    train_state,
                    val_batch,
                    scaler,
                    val_loss,
                    val_diag,
                )
            catch err
                @warn "Validation logging failed" error = err
            end
        end
        # Early stop only when patience > 0 and the loss has stayed below the
        # target for at least `patience` epochs. With patience==0 (default), run
        # for the full number of epochs.
        if settings.patience > 0 &&
           best_loss ≤ settings.target_loss &&
           stall_epochs ≥ settings.patience
            stored_state = maybe_to_host(best_state, settings)
            return TrainingResult(
                stored_state,
                best_loss,
                epoch,
                batch_size,
                batches_per_epoch,
                rmse_history,
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
        rmse_history,
    )
end
