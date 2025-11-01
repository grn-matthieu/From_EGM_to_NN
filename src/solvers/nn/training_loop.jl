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

using .NNTrainUtils:
    base_optimizer,
    create_optimizer,
    cosine_learning_rate,
    exponential_learning_rate,
    learning_rate_for_epoch,
    adjust_learning_rate,
    apply_optimizer_learning_rate!,
    bcmc_auto_update!,
    verbose_validation_logging,
    compute_batch_size,
    select_model,
    state_parameters,
    state_states,
    run_model
struct TrainingResult
    best_state::Any
    best_loss::Float64
    epochs_run::Int
    batch_size::Int
    batches_per_epoch::Int
    rmse_history::Vector{Float64}
end


function build_network(input_dim::Int, settings::NNSolverSettings)
    h1, h2 = settings.hidden_sizes
    return Chain(Dense(input_dim, h1, relu), Dense(h1, h2, relu), Dense(h2, 1, softplus))
end

function train_consumption_network!(
    chain,
    settings::NNSolverSettings,
    scaler::FeatureScaler,
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
    loss_function = build_loss_function(G, S, scaler, settings, rng, model_cfg)
    # draw uniform cash-on-hand samples for the initial training batch
    samples_per_epoch = settings.samples_per_epoch
    X_train, sample_count = sample_training(
        G,
        S;
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
        S;
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
                S;
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
                        S;
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
                current_c = vec(phi_to_consumption(out[:Φ], w_denorm; min_c = 1.0f-12))

                # Compute policy update sup-norm if we have previous policy
                Δ_pol = Inf
                if prev_policy !== nothing
                    Δ_pol = maximum(abs.(current_c .- prev_policy))
                end
                prev_policy = copy(current_c)

                # Unified Euler residual evaluation: use the same grid-based
                # evaluator employed during the final solution pass so that
                # convergence diagnostics and reported residuals remain
                # comparable across solvers.
                euler_rmse = Inf
                if model_cfg !== nothing &&
                   G !== nothing &&
                   S !== nothing &&
                   isdefined(model_cfg, :P) &&
                   model_cfg.P !== nothing
                    try
                        eval_rng = derive_rng(rng, (:conv_check, epoch))
                        eval_result = evaluate_stochastic(
                            current_model,
                            current_ps,
                            current_st,
                            model_cfg.P,
                            G,
                            S,
                            scaler,
                            settings,
                            model_cfg.U,
                            eval_rng,
                        )
                        resid_vals = vec(Float64.(eval_result.resid))
                        euler_rmse = sqrt(mean(abs2, resid_vals))
                    catch err
                        if settings.verbose
                            @warn "Grid evaluation failed in convergence check" err = err
                        end
                    end
                else
                    # Deterministic fallback: compute grid residuals directly.
                    if model_cfg !== nothing && G !== nothing && isdefined(model_cfg, :P)
                        a_grid_f32 = Float32.(G[:a].grid)
                        c_pred_vec_f32 = current_c[1:length(a_grid_f32)]
                        residuals =
                            euler_resid_grid(model_cfg.P, a_grid_f32, c_pred_vec_f32)
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

        # periodic validation logging every 100000   epochs
        if settings.verbose && epoch % 100000 == 0
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
