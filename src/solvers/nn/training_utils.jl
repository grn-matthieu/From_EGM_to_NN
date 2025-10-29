module NNTrainUtils

import Optimisers
import Adapt
import Lux
import LinearAlgebra: diag, dot, I
using Printf
using ..NNLosses: _var_bcmc_given_N, estimate_linearized_components, suggest_bcmc_N

# General-purpose helpers (previously in utils.jl) inserted at top-level so
# they are defined in the parent `NNKernel` module when this file is included.
# These functions are intentionally defined outside the `NNTrainUtils` module
# body by being placed before the `module` declaration in the included file
# when `training_utils.jl` is included by `kernel.jl`.

const PARENT = parentmodule(@__MODULE__)

function maybe_to_host(state::Lux.Training.TrainState, settings)
    mdl = state.model
    ps = state_parameters(state)
    st = state_states(state)
    return (
        model = mdl,
        parameters = maybe_to_host(ps, settings),
        states = maybe_to_host(st, settings),
    )
end


get_option(opts, key::Symbol, default) =
    isdefined(opts, key) ? getfield(opts, key) : default

const CUDA_OBJECTIVES = (:euler_fb_aio, :euler_fb_bcmc)

function detect_cuda_preference(objective, opts)
    requested = opts[:use_cuda]
    if requested
        try
            CUDA.functional()
        catch
            @warn "CUDA requested but CUDA.jl is not functional. Falling back to CPU." use_cuda =
                false
            return false
        end
        return requested
    end
end

function compute_batch_size(total_samples::Int, choice::Union{Nothing,Int})
    return isnothing(choice) ? max(total_samples, 1) :
           clamp(choice, 1, max(total_samples, 1))
end

function select_model(chain, state)
    return isdefined(state, :model) ? getfield(state, :model) : chain
end

function state_parameters(state)
    if isdefined(state, :parameters)
        return getfield(state, :parameters)
    elseif isdefined(state, :params)
        return getfield(state, :params)
    else
        return nothing
    end
end

function state_states(state)
    if isdefined(state, :states)
        return getfield(state, :states)
    elseif isdefined(state, :state)
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

"""Print verbose validation diagnostics during training.

This extracts the large logging block from the training loop so it can be
reused and tested independently. Mirrors the behaviour previously in
`training_loop.jl`.
"""
function verbose_validation_logging(
    epoch,
    settings,
    chain,
    train_state,
    val_batch,
    scaler,
    val_loss,
    val_diag,
)
    try
        current_model = select_model(chain, train_state)
        current_ps = state_parameters(train_state)
        current_st = state_states(train_state)

        # Get raw network outputs on validation batch
        out, _ = Lux.apply(current_model, val_batch, current_ps, current_st)
        if out isa NamedTuple && isdefined(out, :Φ) && isdefined(out, :h)
            Φ_vals = maybe_to_cpu(vec(out[:Φ]), settings)
            h_vals = maybe_to_cpu(vec(out[:h]), settings)

            # Denormalize to get actual wealth values
            w_norm = val_batch[end, :]
            w_denorm =
                ((maybe_to_cpu(w_norm, settings) .+ 1.0f0) ./ 2.0f0) .* scaler.w_range .+
                scaler.w_min

            # Compute actual consumption c = Φ * w
            c_vals = Φ_vals .* w_denorm
            c_over_w = c_vals ./ w_denorm

            # Compute statistics
            phi_stats = (
                min = minimum(Φ_vals),
                mean = mean(Φ_vals),
                max = maximum(Φ_vals),
                std = std(Φ_vals),
            )
            h_stats = (
                min = minimum(h_vals),
                mean = mean(h_vals),
                max = maximum(h_vals),
                std = std(h_vals),
            )
            c_stats = (min = minimum(c_vals), mean = mean(c_vals), max = maximum(c_vals))
            w_stats =
                (min = minimum(w_denorm), mean = mean(w_denorm), max = maximum(w_denorm))
            c_w_stats =
                (min = minimum(c_over_w), mean = mean(c_over_w), max = maximum(c_over_w))

            # Count violations (should be impossible but let's check)
            n_phi_bad = count(x -> x < 0 || x > 1, Φ_vals)
            n_h_bad = count(x -> x <= 0, h_vals)
            n_c_over_w = count(x -> x > 1.0, c_over_w)

            @printf "[DIAG] Epoch %4d Network Outputs:\n" epoch
            @printf "  Φ ∈ [%.6f, %.6f] mean=%.6f std=%.6f (violations: %d)\n" phi_stats.min phi_stats.max phi_stats.mean phi_stats.std n_phi_bad
            @printf "  h ∈ [%.6f, %.6f] mean=%.6f std=%.6f (violations: %d)\n" h_stats.min h_stats.max h_stats.mean h_stats.std n_h_bad
            @printf "  c ∈ [%.4f, %.4f] mean=%.4f\n" c_stats.min c_stats.max c_stats.mean
            @printf "  w ∈ [%.4f, %.4f] mean=%.4f\n" w_stats.min w_stats.max w_stats.mean
            @printf "  c/w ∈ [%.6f, %.6f] mean=%.6f (c>w violations: %d)\n" c_w_stats.min c_w_stats.max c_w_stats.mean n_c_over_w
        end
    catch diag_err
        @warn "Network output diagnostics failed" error = diag_err
    end

    # Print scalar FB diagnostics if available
    if :fb in keys(val_diag)
        aux = val_diag.fb
        try
            if settings.objective === :euler_fb_bcmc
                bcmc_val = isdefined(aux, :bcmc_mean) ? getfield(aux, :bcmc_mean) : NaN
                n_eff = isdefined(aux, :n_eff) ? getfield(aux, :n_eff) : missing
                if isdefined(aux, :gvar_mean)
                    if n_eff === missing
                        @printf(
                            "[VAL] Epoch %4d: kt_mean=%.6g bcmc_mean=%.6g gvar=%.6g max_abs_q=%.6g\n",
                            epoch,
                            aux.kt_mean,
                            bcmc_val,
                            getfield(aux, :gvar_mean),
                            aux.max_abs_q,
                        )
                    else
                        @printf(
                            "[VAL] Epoch %4d: kt_mean=%.6g bcmc_mean=%.6g gvar=%.6g N=%.0f max_abs_q=%.6g\n",
                            epoch,
                            aux.kt_mean,
                            bcmc_val,
                            getfield(aux, :gvar_mean),
                            Float64(n_eff),
                            aux.max_abs_q,
                        )
                    end
                else
                    if n_eff === missing
                        @printf(
                            "[VAL] Epoch %4d: kt_mean=%.6g bcmc_mean=%.6g max_abs_q=%.6g\n",
                            epoch,
                            aux.kt_mean,
                            bcmc_val,
                            aux.max_abs_q,
                        )
                    else
                        @printf(
                            "[VAL] Epoch %4d: kt_mean=%.6g bcmc_mean=%.6g N=%.0f max_abs_q=%.6g\n",
                            epoch,
                            aux.kt_mean,
                            bcmc_val,
                            Float64(n_eff),
                            aux.max_abs_q,
                        )
                    end
                end
            else
                # Default to AiO-style logging when active or when bcmc fields missing
                aio_val = isdefined(aux, :aio_mean) ? getfield(aux, :aio_mean) : NaN
                @printf(
                    "[VAL] Epoch %4d: kt_mean=%.6g aio_mean=%.6g max_abs_q=%.6g\n",
                    epoch,
                    aux.kt_mean,
                    aio_val,
                    aux.max_abs_q,
                )
            end
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
end

export base_optimizer,
    create_optimizer,
    cosine_learning_rate,
    exponential_learning_rate,
    learning_rate_for_epoch,
    adjust_learning_rate,
    apply_optimizer_learning_rate!,
    bcmc_auto_update!

function base_optimizer(optimizer::Symbol, lr::Float64)
    if optimizer === :adam
        return Optimisers.AdamW(lr)
    elseif optimizer === :adamw
        return Optimisers.AdamW(lr)
    elseif optimizer === :rmsprop
        return Optimisers.RMSProp(lr)
    elseif optimizer === :adagrad
        return Optimisers.AdaGrad(lr)
    elseif optimizer === :sgd
        return Optimisers.Descent(lr)
    else
        @warn "Unsupported optimizer=$(optimizer); falling back to AdamW"
        return Optimisers.AdamW(lr)
    end
end

function create_optimizer(settings)
    base = base_optimizer(settings.optimizer, settings.learning_rate)
    return Optimisers.OptimiserChain(Optimisers.ClipGrad(0.02), base)
end

function cosine_learning_rate(settings, epoch::Int)
    epoch ≤ 0 && return settings.lr_max
    warmup = settings.warmup_epochs
    lr_min = settings.lr_min
    lr_max = settings.lr_max
    if warmup > 0 && epoch ≤ warmup
        frac = epoch / warmup
        return lr_min + (lr_max - lr_min) * frac
    end
    t = max(epoch - warmup, 0)
    horizon = settings.lr_decay_horizon
    if horizon ≤ 0
        return lr_min
    end
    if t ≥ horizon
        return lr_min
    end
    cos_term = 0.5 * (1 + cos(pi * t / horizon))
    return lr_min + (lr_max - lr_min) * cos_term
end

function exponential_learning_rate(settings, epoch::Int)
    epoch ≤ 0 && return settings.lr_max
    warmup = settings.warmup_epochs
    lr_min = settings.lr_min
    lr_max = settings.lr_max
    if warmup > 0 && epoch ≤ warmup
        frac = epoch / warmup
        return lr_min + (lr_max - lr_min) * frac
    end
    t = max(epoch - warmup, 0)
    gamma = settings.lr_gamma
    milestones = settings.lr_milestones
    horizon = settings.lr_decay_horizon
    if isempty(milestones)
        steps = horizon > 0 ? min(t, horizon) : t
    else
        steps = count(m -> epoch ≥ m, milestones)
    end
    if steps <= 0
        return lr_max
    end
    new_lr = lr_max * gamma^steps
    floor_val = max(lr_min, floatmin(Float64))
    return clamp(new_lr, floor_val, lr_max)
end

function learning_rate_for_epoch(settings, epoch::Int)
    if settings.lr_schedule === :cosine
        return cosine_learning_rate(settings, epoch)
    elseif settings.lr_schedule === :exponential
        return exponential_learning_rate(settings, epoch)
    else
        return settings.lr_max
    end
end

function rebuild_adam_family(opt, lr)
    pairs = Pair{Symbol,Any}[:eta=>Float64(lr)]
    if isdefined(opt, :beta)
        push!(pairs, :beta => getproperty(opt, :beta))
    end
    eps_val = opt[:epsilon]
    if eps_val !== nothing
        push!(pairs, :epsilon => eps_val)
    end
    if opt isa Optimisers.AdamW && isdefined(opt, :weight_decay)
        push!(pairs, :weight_decay => getproperty(opt, :weight_decay))
    end
    constructor = opt isa Optimisers.AdamW ? Optimisers.AdamW : Optimisers.Adam
    return constructor(; pairs...)
end

function rebuild_with_lr(opt, lr)
    fields = fieldnames(typeof(opt))
    target =
        findfirst(name -> name === :eta || name === :lr || name === :learning_rate, fields)
    if target === nothing
        return opt
    end
    target_field = fields[target]
    values = map(fields) do name
        name === target_field ? Float64(lr) : getfield(opt, name)
    end
    try
        return (typeof(opt))(values...)
    catch
        return opt
    end
end

function adjust_learning_rate(opt, lr)
    if opt isa Union{Optimisers.Adam,Optimisers.AdamW}
        try
            return rebuild_adam_family(opt, lr)
        catch
            return rebuild_with_lr(opt, lr)
        end
    end
    return rebuild_with_lr(opt, lr)
end

function adjust_learning_rate(opt::Optimisers.OptimiserChain, lr)
    new_opts = map(stage -> adjust_learning_rate(stage, lr), opt.opts)
    return Optimisers.OptimiserChain(new_opts...)
end

function apply_optimizer_learning_rate!(state, lr)
    if isdefined(state, :opt)
        current_opt = getfield(state, :opt)
        updated_opt = adjust_learning_rate(current_opt, lr)
        if updated_opt !== current_opt
            setfield!(state, :opt, updated_opt)
        end
        return state
    elseif isdefined(state, :optimizer)
        current_opt = getfield(state, :optimizer)
        updated_opt = adjust_learning_rate(current_opt, lr)
        if updated_opt === current_opt
            return state
        end
        mdl = isdefined(state, :model) ? getfield(state, :model) : nothing
        ps = isdefined(state, :parameters) ? getfield(state, :parameters) : nothing
        st = isdefined(state, :states) ? getfield(state, :states) : nothing
        return Lux.Training.TrainState(mdl, ps, st, updated_opt)
    end
    return state
end

# BCMC Auto-N update helper. This function mirrors the update logic from the
# original training loop but is factored out so tests and the kernel can reuse
# it. It mutates `model_cfg.bcmc_state` when an update is selected.
function bcmc_auto_update!(
    chain,
    train_state,
    settings,
    model_cfg,
    cur_batch,
    scaler,
    rng,
    step_id,
    samples_per_epoch,
)
    if step_id % settings.bcmc_update_every != 0
        return
    end
    pairs0 = max(div(settings.n_mc * (settings.n_mc - 1), 2), 1)
    default_T = samples_per_epoch * pairs0
    Tbudget = something(settings.bcmc_budget_T, default_T)
    Tbudget ≤ 0 && return
    st_auto = getfield(model_cfg, :bcmc_state)
    curN = Int(clamp(round(st_auto.n_eff[]), 2, typemax(Int)))
    M_est = max(fld(2 * Tbudget, max(curN, 1)), 1)
    if M_est < 1
        return
    end

    sqrt_cols = sqrt(Float64(size(cur_batch, 2)))
    probe_cols = min(size(cur_batch, 2), max(64, max(Int(floor(sqrt_cols)), 1)))
    X_probe = @view cur_batch[:, 1:probe_cols]
    Xp = settings.use_cuda ? Adapt.adapt(Array, X_probe) : X_probe

    base_model = PARENT.select_model(chain, train_state)
    ps_cur = PARENT.state_parameters(train_state)
    st_cur = PARENT.state_states(train_state)
    ps_cpu = PARENT.maybe_to_host(ps_cur, settings)
    st_cpu = PARENT.maybe_to_host(st_cur, settings)

    scalar_forward = function (x, ps_, st_; mode = :default)
        Xmat = ndims(x) == 1 ? reshape(x, :, 1) : x
        if mode === :fb_scalar
            if isdefined(model_cfg.P, :Σ) && isdefined(model_cfg.P, :A)
                vals = PARENT.loss_euler_fb_bcmc_csvar!(
                    base_model,
                    ps_,
                    st_,
                    Xmat,
                    model_cfg,
                    rng;
                    mode = :fb_scalar,
                )
            else
                vals = PARENT.loss_euler_fb_bcmc_ar1!(
                    base_model,
                    ps_,
                    st_,
                    Xmat,
                    model_cfg,
                    rng;
                    mode = :fb_scalar,
                )
            end
            return Float64(vals[1])
        else
            error("fb_scalar wrapper only supports mode=:fb_scalar")
        end
    end

    if scaler.csvar_mode
        dε = max(length(scaler.y_range), 1)
        Σε = Matrix{Float64}(I, dε, dε)
        ds = dε + 1
        Σs = Matrix{Float64}(I, ds, ds)
    else
        Σε = Matrix{Float64}(I, 1, 1)
        Σs = Matrix{Float64}(I, 2, 2)
    end

    sigma2_f, rho_f, A_lin, B_lin =
        estimate_linearized_components(scalar_forward, ps_cpu, st_cpu, Xp, scaler, Σs, Σε)

    rho_eps = max(rho_f, eps(Float32))
    A_eps = max(A_lin, eps(Float32))
    if rho_eps ≤ 10 * eps(Float32)
        N_star = min(max(Int(round(2 * Tbudget)), 2), 1024)
        V_star = _var_bcmc_given_N(sigma2_f, rho_eps, N_star, Tbudget)
    elseif A_eps ≤ 10 * eps(Float32)
        N_star = 2
        V_star = _var_bcmc_given_N(sigma2_f, rho_eps, N_star, Tbudget)
    else
        N_star, V_star = suggest_bcmc_N(sigma2_f, rho_eps, Tbudget; N_cap = 1024)
    end
    curV = _var_bcmc_given_N(sigma2_f, rho_eps, curN, Tbudget)

    st_auto.sigma2[] = sigma2_f
    st_auto.rho[] = rho_f
    st_auto.A[] = A_lin
    st_auto.B[] = B_lin

    if V_star ≤ 0.98 * curV && N_star != curN
        st_auto.n_eff[] = N_star
        if settings.verbose
            ratio = B_lin ≈ 0 ? Inf : A_lin / max(B_lin, eps(Float32))
            @info "bc-MC auto-N update" step = step_id N_old = curN N_new = N_star sigma2 =
                sigma2_f rho = rho_f A = A_lin B = B_lin ratio = ratio
        end
    elseif settings.verbose
        ratio = B_lin ≈ 0 ? Inf : A_lin / max(B_lin, eps(Float32))
        @info "bc-MC auto-N probe" step = step_id N_cur = curN sigma2 = sigma2_f rho = rho_f A =
            A_lin B = B_lin ratio = ratio
    end
end

end # module
