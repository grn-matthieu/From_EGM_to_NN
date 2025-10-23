import CUDA
import Adapt
import Zygote
import Adapt
using Lux: fmap
using ChainRulesCore: ignore_derivatives
using LinearAlgebra: diag
using Optimisers

include("preprocessing.jl")


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
    # Generate GPU Gaussian noise similar to `ref`. Prevent AD from tracing sampling.
    return ignore_derivatives() do
        CUDA.randn(eltype(ref), size(ref)...)
    end
end

function randn_like(rng, ref)
    # Generate CPU Gaussian noise similar to `ref`. Prevent AD from tracing sampling.
    return ignore_derivatives() do
        out = similar(ref)
        randn!(rng, out)
        out
    end
end

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

const CUDA_OBJECTIVES = (:euler_fb_aio, :euler_fb_bcmc)

maybe_objective_cuda(objective, default) = objective in CUDA_OBJECTIVES ? default : false

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
    if use_cuda && !(objective in CUDA_OBJECTIVES)
        @warn "CUDA acceleration currently supported only for objectives $(CUDA_OBJECTIVES); got :$(objective). Falling back to CPU." use_cuda =
            false
    end
    return use_cuda
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
    resample_interval = max(Int(get_option(opts, :resample_every, 1)), 0)
    target_loss = Float32(get_option(opts, :target_loss, 2e-4))
    # Default: disable early stopping unless explicitly requested
    patience = max(Int(get_option(opts, :patience, 0)), 0)
    hid1 = max(Int(get_option(opts, :hid1, 128)), 1)
    hid2 = max(Int(get_option(opts, :hid2, 128)), 1)
    objective = Symbol(get_option(opts, :objective, objective_default))
    # clamp v_h to a broader safe range [0.2, 5.0] to allow more tuning flexibility
    v_h = clamp(Float64(get_option(opts, :v_h, 0.5)), 0.2, 5.0)
    # Auto-derive a sensible cash-on-hand range if not provided
    w_min_opt = get_option(opts, :w_min, nothing)
    w_max_opt = get_option(opts, :w_max, nothing)
    # Compute implied w-range from asset bounds and income variability when possible
    function income_bounds(params, shocks)
        if params === nothing
            return (0.0, 1.0)
        end
        # CSVAR: components follow log-Gaussian process. Use log means adjusted for variance.
        if hasproperty(params, :A) && hasproperty(params, :Σ)
            μ_log = csvar_component_log_means(params)
            Σ = Matrix{Float64}(params.Σ)
            σ = sqrt.(max.(diag(Σ), 0.0))
            lower = sum(exp.(μ_log .- 3 .* σ))
            upper = sum(exp.(μ_log .+ 3 .* σ))
            return (lower, upper)
        end
        # AR(1) log-income (stochastic): y is log-mean, use exp(μ ± 3σ)
        if shocks !== nothing || has_shocks
            μ = hasproperty(params, :y) ? Float64(getfield(params, :y)) : 0.0
            σ = hasproperty(params, :σ_shock) ? Float64(getfield(params, :σ_shock)) : 0.0
            return (exp(μ - 3σ), exp(μ + 3σ))
        end
        # Deterministic: take income level(s)
        if hasproperty(params, :y) && params.y isa AbstractVector
            return (
                mean(exp.(Float64.(collect(params.y)))),
                mean(exp.(Float64.(collect(params.y)))),
            )
        else
            return (exp(Float64(getfield(params, :y))), exp(Float64(getfield(params, :y))))
        end
    end
    function w_bounds(params, grids, shocks)
        (inc_lo, inc_hi) = income_bounds(params, shocks)
        if grids === nothing
            return (inc_lo, inc_hi)
        end
        a_min = Float64(getproperty(grids[:a], :min))
        a_max = Float64(getproperty(grids[:a], :max))
        Rg =
            1.0 + Float64(
                get_option(
                    params,
                    :r,
                    hasproperty(params, :r) ? getfield(params, :r) : 0.0,
                ),
            )
        return (Rg * a_min + inc_lo, Rg * a_max + inc_hi)
    end
    (auto_w_min, auto_w_max) = w_bounds(P, G, S)
    w_min = Float32(w_min_opt === nothing ? auto_w_min : Float64(w_min_opt))
    w_max = Float32(w_max_opt === nothing ? auto_w_max : Float64(w_max_opt))
    samples_per_epoch = max(Int(get_option(opts, :samples_per_epoch, 64)), 1)
    sigma_shocks = get_option(opts, :sigma_shocks, nothing)
    use_cuda = detect_cuda_preference(objective, opts)
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

    optimizer_raw = get_option(opts, :optimizer, :adam)
    optimizer = try
        Symbol(lowercase(String(optimizer_raw)))
    catch err
        @warn "Failed to parse optimizer option $(optimizer_raw); defaulting to :adam" err =
            err
        :adam
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
        else
            return nothing
        end
    end
    canonical = canonical_optimizer(optimizer)
    supported_optimizers = (:adam, :rmsprop, :adagrad, :sgd)
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

function base_optimizer(optimizer::Symbol, lr::Float64)
    if optimizer === :adam
        return Optimisers.Adam(lr)
    elseif optimizer === :rmsprop
        return Optimisers.RMSProp(lr)
    elseif optimizer === :adagrad
        return Optimisers.AdaGrad(lr)
    elseif optimizer === :sgd
        return Optimisers.Descent(lr)
    else
        @warn "Unsupported optimizer=$(optimizer); falling back to Adam"
        return Optimisers.Adam(lr)
    end
end

function create_optimizer(settings::NNSolverSettings)
    base = base_optimizer(settings.optimizer, settings.learning_rate)
    return Optimisers.OptimiserChain(Optimisers.ClipGrad(0.02), base)
end

function cosine_learning_rate(settings::NNSolverSettings, epoch::Int)
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

function exponential_learning_rate(settings::NNSolverSettings, epoch::Int)
    epoch ≤ 0 && return settings.lr_max
    warmup = settings.warmup_epochs
    lr_min = settings.lr_min
    lr_max = settings.lr_max
    if warmup > 0 && epoch ≤ warmup
        frac = epoch / warmup
        return lr_min + (lr_max - lr_min) * frac
    end
    # Steps counted after warmup
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

function learning_rate_for_epoch(settings::NNSolverSettings, epoch::Int)
    if settings.lr_schedule === :cosine
        return cosine_learning_rate(settings, epoch)
    elseif settings.lr_schedule === :exponential
        return exponential_learning_rate(settings, epoch)
    else
        return settings.lr_max
    end
end

function adjust_learning_rate(opt, lr)
    # Prefer explicit handling for common optimisers to avoid MethodErrors
    # when constructors differ across Optimisers.jl versions.
    if opt isa Optimisers.Adam
        # Try keyword-style constructors that recent Optimisers expose.
        beta = getproperty(opt, :beta)
        # Try common epsilon field names
        eps_val =
            hasproperty(opt, :epsilon) ? getproperty(opt, :epsilon) :
            (hasproperty(opt, :eps) ? getproperty(opt, :eps) : nothing)
        try
            if eps_val === nothing
                return Optimisers.Adam(; eta = Float64(lr), beta = beta)
            else
                return Optimisers.Adam(; eta = Float64(lr), beta = beta, epsilon = eps_val)
            end
        catch err
            # Last-resort: attempt to reconstruct by positional fields (best-effort)
            try
                fields = fieldnames(typeof(opt))
                target = findfirst(
                    name -> name === :eta || name === :lr || name === :learning_rate,
                    fields,
                )
                if target === nothing
                    return opt
                end
                target_field = fields[target]
                values = map(fields) do name
                    name === target_field ? Float64(lr) : getfield(opt, name)
                end
                return (typeof(opt))(values...)
            catch
                return opt
            end
        end
    end

    # Generic fallback: try to locate a common learning-rate-like field and
    # reconstruct the optimiser. If that fails, return the original optimiser.
    fields = fieldnames(typeof(opt))
    target =
        findfirst(name -> name === :eta || name === :lr || name === :learning_rate, fields)
    if target === nothing
        return opt
    end
    target_field = fields[target]
    values = map(fields) do name
        name === target_field ? lr : getfield(opt, name)
    end
    try
        return (typeof(opt))(values...)
    catch
        return opt
    end
end

function adjust_learning_rate(opt::Optimisers.OptimiserChain, lr)
    # OptimiserChain stores its stages in the `opts` field
    # Map over the inner stages and adjust each stage's learning rate.
    new_opts = map(stage -> adjust_learning_rate(stage, lr), opt.opts)
    return Optimisers.OptimiserChain(new_opts...)
end

function apply_optimizer_learning_rate!(state, lr)
    # Support both older `:opt` field and Lux.Training.TrainState's `:optimizer`
    if hasproperty(state, :opt)
        current_opt = getfield(state, :opt)
        updated_opt = adjust_learning_rate(current_opt, lr)
        if updated_opt !== current_opt
            setfield!(state, :opt, updated_opt)
        end
        return state
    elseif hasproperty(state, :optimizer)
        current_opt = getfield(state, :optimizer)
        updated_opt = adjust_learning_rate(current_opt, lr)
        if updated_opt === current_opt
            return state
        end
        # Lux.Training.TrainState is immutable; construct a new TrainState
        mdl = hasproperty(state, :model) ? getfield(state, :model) : nothing
        ps = hasproperty(state, :parameters) ? getfield(state, :parameters) : nothing
        st = hasproperty(state, :states) ? getfield(state, :states) : nothing
        return Lux.Training.TrainState(mdl, ps, st, updated_opt)
    end
    return state
end

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

        # If caller selected an FB-style objective, delegate to the custom loss
        if settings.objective in (:euler_fb_aio, :euler_fb_bcmc)
            if !fb_supported(model_cfg)
                if !fb_warning_emitted[]
                    @warn "objective :$(settings.objective) requires active stochastic shocks; falling back to :euler_residual"
                    fb_warning_emitted[] = true
                end
            else
                objective = settings.objective
                # the loss routines return (loss, (st1, aux_namedtuple))
                loss_val, st_pack = if objective == :euler_fb_aio
                    loss_euler_fb_aio!(model, ps, st, X, model_cfg, rng)
                else
                    P_full = model_cfg.P
                    is_csvar =
                        hasproperty(P_full, :A) &&
                        hasproperty(P_full, :Σ) &&
                        hasproperty(P_full, :y_dim) &&
                        getproperty(P_full, :y_dim) > 1
                    if is_csvar
                        loss_euler_fb_bcmc_csvar!(model, ps, st, X, model_cfg, rng)
                    else
                        loss_euler_fb_bcmc_ar1!(model, ps, st, X, model_cfg, rng)
                    end
                end
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
            loss = mean(huber_loss.(resid, 1.0f0))
        else
            if is_csvar_problem(model_cfg === nothing ? P_resid : model_cfg.P, S)
                # --- CSVAR Monte Carlo expectation on the minibatch ---
                # Denormalize features
                mean_vals, comps, w_vals = denormalize_feature_batch(scaler, X)
                T = eltype(X)
                β = T(P_resid.β)
                Rg = one(T) + T(P_resid.r)
                # current consumption from prediction
                c0 =
                    prediction isa NamedTuple ?
                    vec(phi_to_consumption(prediction[:Φ], w_vals; min_c = 1.0f-8)) :
                    vec(clamp.(prediction, eps(T), Inf))
                a_next = @. Rg * (w_vals - c0)

                # Draw innovations: ε ~ N(0, Σ)
                base_P = model_cfg === nothing ? P_resid : model_cfg.P
                Σ = Matrix{Float32}(base_P.Σ)
                L = cholesky(Symmetric(Σ), check = false).L
                y_dim = size(Σ, 1)
                K = settings.n_mc

                # Preallocate
                resid_vec = Vector{Float32}(undef, length(a_next))
                uprime = uprime_from(U, P_resid)
                uprime_c0 = Float32.(uprime(Float32.(c0)))

                # MC loop: build K future feature batches and average u′(c1)
                innovations = Matrix{Float32}(undef, y_dim, K)
                randn!(rng, innovations)
                draws = Matrix{Float32}(undef, y_dim, K)
                mul!(draws, L, innovations)                # y components
                μ_vec =
                    hasproperty(base_P, :y) && base_P.y isa AbstractVector ?
                    Float32.(collect(base_P.y)) : Float32[Float32(getfield(base_P, :y))]
                @. draws += μ_vec
                income_draws = Float32.(csvar_income(draws))   # scalar income from components

                # For each sample in the minibatch, evaluate c1 under K draws
                for i = 1:length(a_next)
                    w_future = @. Rg * Float32(a_next[i]) + income_draws
                    X_future = build_feature_batch_from_states(scaler, draws, w_future)
                    X_future_dev = maybe_to_device(X_future, settings)
                    pred_next = run_model(model, ps, st_out, X_future_dev)
                    c1_raw =
                        pred_next isa NamedTuple ?
                        phi_to_consumption(
                            pred_next[:Φ],
                            maybe_to_device(w_future, settings);
                            min_c = 1.0f-6,
                        ) : pred_next
                    c1_vec = vec(permutedims(ensure_row(c1_raw)))
                    c1_cpu = maybe_to_cpu(c1_vec, settings)
                    mean_u′ = mean(uprime(Float32.(c1_cpu)))
                    denom =
                        uprime_c0[i] <= 0 ? uprime(Float32(max(c0[i], 1.0f-6))) :
                        uprime_c0[i]
                    resid_vec[i] = Float32(abs(1 - β * Rg * mean_u′ / denom))
                end
                loss = mean(huber_loss.(resid_vec, 1.0f0))
            else
                # Discrete scalar z with grid
                a_grid_f32, z_grid_f32, Pz_f32, _, c_pred_f32 =
                    stoch_residual_inputs(c_pred, G, S)
                resid = euler_resid_stoch_grid(
                    P_resid,
                    a_grid_f32,
                    z_grid_f32,
                    Pz_f32,
                    c_pred_f32,
                )
                loss = mean(huber_loss.(resid, 1.0f0))
            end
        end

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

using ..CSVarUtils: csvar_component_log_means

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
        X, _ = generate_dataset(G, S, base_P; mode = :full, rng = rng)
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
    is_csvar = hasproperty(base_P, :A) && hasproperty(base_P, :Σ)
    if is_csvar
        log_means = csvar_component_log_means(base_P)
        y_dim = length(log_means)
        extra_cols = y_dim
        income_targets =
            hasproperty(base_P, :y) && base_P.y isa AbstractVector ?
            Float64.(collect(base_P.y)) : Float64[Float64(getfield(base_P, :y))]
        base_income = sum(income_targets)
        feature_dim = 1 + extra_cols + 1
    else
        y_levels =
            hasproperty(base_P, :y) && base_P.y isa AbstractVector ?
            Float32.(collect(base_P.y)) : Float32[Float32(getfield(base_P, :y))]
        y_dim = length(y_levels)
        extra_cols = y_dim > 1 ? y_dim : 0
        feature_dim = 1 + extra_cols + 1
        base_income = mean(y_levels)
    end
    a_min = Float32(G[:a].min)
    a_max = Float32(G[:a].max)

    if settings.has_shocks && !isnothing(S) && hasproperty(S, :zgrid)
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
        if is_csvar &&
           hasproperty(base_P, :Σ) &&
           !isnothing(S) &&
           hasproperty(S, :process) &&
           S.process == :gaussian_linear
            Σ = Matrix{Float64}(base_P.Σ)
            chol = cholesky(Symmetric(Σ), check = false).L
            comps_tmp = Matrix{Float32}(undef, extra_cols, m)
            @inbounds for i = 1:m
                ε = randn(rng, Float64, y_dim)
                y_vec = log_means .+ chol * ε
                comps_tmp[:, i] .= Float32.(y_vec)
            end
            mean_draw = vec(sum(exp.(comps_tmp); dims = 1))
        else
            if extra_cols > 0
                if is_csvar
                    comps_tmp = repeat(reshape(Float32.(log_means), extra_cols, 1), 1, m)
                    mean_draw = vec(sum(exp.(comps_tmp); dims = 1))
                else
                    comps_tmp = repeat(reshape(Float32.(y_levels), extra_cols, 1), 1, m)
                    mean_draw = vec(mean(comps_tmp; dims = 1))
                end
            else
                mean_draw = fill(Float32(base_income), m)
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
    current_lr = NaN
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
                        if settings.objective === :euler_fb_bcmc
                            bcmc_val =
                                hasproperty(aux, :bcmc_mean) ? getfield(aux, :bcmc_mean) :
                                NaN
                            n_eff =
                                hasproperty(aux, :n_eff) ? getfield(aux, :n_eff) : missing
                            if hasproperty(aux, :gvar_mean)
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
                            aio_val =
                                hasproperty(aux, :aio_mean) ? getfield(aux, :aio_mean) : NaN
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
