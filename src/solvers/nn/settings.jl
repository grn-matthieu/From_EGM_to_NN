# Local, minimal get_option to avoid include-order coupling. This mirrors
# the helper from `training_utils.jl` but keeps `settings.jl` standalone so
# the method layer can call `solver_settings` without forcing include order.
get_option(opts, key::Symbol, default) =
    (opts === nothing) ? default : (isdefined(opts, key) ? getfield(opts, key) : default)

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
    eval_samples::Int
    sigma_shocks::Union{Nothing,Float64}
    n_mc::Int
    bcmc_budget_T::Union{Nothing,Int}
    bcmc_auto_N::Bool
    bcmc_update_every::Int
    skip_final_eval::Bool
end


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
    eval_samples_default = max(4 * samples_per_epoch, samples_per_epoch)
    eval_samples = max(Int(get_option(opts, :eval_samples, eval_samples_default)), 1)
    sigma_shocks = get_option(opts, :sigma_shocks, nothing)
    n_mc = max(Int(get_option(opts, :n_mc, 16)), 1)
    bcmc_budget_T = let v = get_option(opts, :bcmc_budget_T, nothing)
        v === nothing ? nothing : Int(v)
    end
    bcmc_auto_N = Bool(get_option(opts, :bcmc_auto_N, false))
    bcmc_update_every = max(Int(get_option(opts, :bcmc_update_every, 10)), 1)
    skip_final_eval = Bool(get_option(opts, :skip_final_eval, false))

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
        eval_samples,
        sigma_shocks,
        n_mc,
        bcmc_budget_T,
        bcmc_auto_N,
        bcmc_update_every,
        skip_final_eval,
    )
end
