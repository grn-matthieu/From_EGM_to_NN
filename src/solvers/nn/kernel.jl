"""
NNKernel

High-level orchestration of the neural-network solver for consumption policies.
The heavy lifting (pre-processing, training loop and evaluation utilities) lives
in dedicated helpers so that this file focuses on the solver flow.
"""
module NNKernel
using ..GridHelpers: fit_values_on_backend!, eval_backend_at_points, grid_backend_available

using ..API: get_grids, get_params, get_shocks, get_utility
using ..CommonInterp: InterpKind, LinearInterp
using ..DataNN: generate_dataset
using ..EulerResiduals: euler_resid_grid
using ..Determinism: derive_rng, promote_master_rng
using ..CSVarUtils: csvar_income
using ChainRulesCore: ignore_derivatives
using Lux
using Optimisers
using Random
using Printf
using Statistics: mean, quantile
using LinearAlgebra: cholesky, mul!, Symmetric

include("mixed_precision.jl")
include("training_loop.jl")
include("evaluation.jl")

export solve_nn

const H_ALPHA = 1.0f0
const EPS_VAR = 1e-12

"""Construct the dual-head Lux model used by the solver."""
function build_dual_head_network(input_dim::Int, hidden::NTuple{2,Int})
    H1, H2 = hidden
    final = Dense(H2, 2)
    trunk = Chain(Dense(input_dim, H1, leakyrelu), Dense(H1, H2, leakyrelu), final)
    # The post-processing function expects a 2×N matrix and splits it into Φ and h
    function postprocess(x)
        φ_pre = x[1:1, :]
        h_pre = x[2:2, :]
        # paper: Φ = sigmoid(·) and h = exp(·)
        return (; Φ = sigmoid.(φ_pre), h = exp.(h_pre))
    end
    return Chain(trunk, postprocess)
end

function build_model_config(P, U, scaler, P_resid, settings)
    # Internal adaptive state for bc-MC Auto-N (held in Refs to allow updates)
    bcmc_state = (
        n_eff = Ref(settings.n_mc),
        sigma2 = Ref(0.0),
        rho = Ref(0.0),
        A = Ref(0.0),
        B = Ref(0.0),
    )
    # Adaptive v_h state (held in Ref to allow runtime updates)
    adaptive_v_h_state = (
        v_h = Ref(settings.v_h),
        initial_v_h = settings.v_h,
        kt_ema = Ref(1.0),
        ema_alpha = 0.1,
    )
    return (
        P = P,
        U = U,
        v_h = settings.v_h,  # Keep for backwards compat, but use adaptive_v_h_state.v_h[] in loss
        adaptive_v_h_state = adaptive_v_h_state,
        scaler = scaler,
        P_resid = P_resid,
        settings = settings,
        sigma_shocks = settings.sigma_shocks,
        bcmc_state = bcmc_state,
    )
end

function maybe_dense_diagnostics(
    model,
    params,
    states,
    P_resid,
    U,
    scaler,
    settings;
    eval_mc_fn = nothing,
    eval_gh_fn = nothing,
    G = nothing,
    S = nothing,
    P = nothing,
    rng = nothing,
)
    has_stochastic =
        settings.has_shocks ||
        (S !== nothing && hasproperty(S, :process) && S.process == :gaussian_linear)
    if !has_stochastic
        return nothing, nothing
    end
    # forward explicit grids/shocks/params when available to avoid relying on Main
    # Resolve eval function bindings at call time so test patches to
    # `NNKernel.eval_euler_residuals_mc`/`_gh` are respected. Tests may also
    # pass explicit functions via kwargs.
    fmc = eval_mc_fn === nothing ? eval_euler_residuals_mc : eval_mc_fn
    fgh = eval_gh_fn === nothing ? eval_euler_residuals_gh : eval_gh_fn

    # Prepare diagnostics variables and call the eval functions. Some tests or
    # user code may have patched simple
    # varargs versions that don't accept keyword args; try keyword call first
    # and fall back to positional-only if that fails with MethodError.
    mc_diag = nothing
    gh_diag = nothing
    try
        mc_diag = fmc(
            model,
            params,
            states,
            P_resid,
            U,
            scaler,
            settings;
            G = G,
            S = S,
            P = P,
            rng = rng,
        )
    catch err
        if err isa MethodError
            mc_diag = fmc(model, params, states, P_resid, U, scaler, settings)
        else
            rethrow()
        end
    end

    try
        gh_diag = fgh(
            model,
            params,
            states,
            P_resid,
            U,
            scaler,
            settings;
            G = G,
            S = S,
            P = P,
            rng = rng,
        )
    catch err
        if err isa MethodError
            gh_diag = fgh(model, params, states, P_resid, U, scaler, settings)
        else
            rethrow()
        end
    end
    return mc_diag, gh_diag
end

function build_options_summary(settings, training_result, runtime)
    return (;
        epochs = settings.epochs,
        epochs_run = training_result.epochs_run,
        batch = training_result.batch_size,
        lr = settings.learning_rate,
        optimizer = settings.optimizer,
        lr_min = settings.lr_min,
        lr_max = settings.lr_max,
        lr_decay_horizon = settings.lr_decay_horizon,
        lr_schedule = settings.lr_schedule,
        lr_gamma = settings.lr_gamma,
        lr_milestones = settings.lr_milestones,
        warmup_epochs = settings.warmup_epochs,
        runtime = runtime,
        verbose = settings.verbose,
        batches_per_epoch = training_result.batches_per_epoch,
        device = settings.use_cuda ? :cuda : :cpu,
    )
end

function solve_nn(model; opts = nothing, rng = nothing)
    rng === nothing && error("solve_nn requires a `rng` keyword argument")
    master = promote_master_rng(rng)
    train_rng = derive_rng(master, :train)
    diag_rng = derive_rng(master, :diagnostics)
    eval_rng = derive_rng(master, :evaluation)

    P = get_params(model)
    G = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    start_time = time_ns()
    is_csvar = !isnothing(S) && hasproperty(S, :process) && S.process == :gaussian_linear
    has_shocks = !isnothing(S) && !is_csvar
    objective_default =
        is_csvar ? :euler_residual : has_shocks ? :euler_fb_aio : :euler_residual

    # If provided w-range misses most of the model's cash-on-hand grid, expand it
    X_tmp, w_grid_model = det_forward_inputs(G, P)
    w_lo_cfg = hasproperty(opts, :w_min) ? getfield(opts, :w_min) : nothing
    w_hi_cfg = hasproperty(opts, :w_max) ? getfield(opts, :w_max) : nothing
    use_opts = opts
    if w_lo_cfg !== nothing && w_hi_cfg !== nothing
        in_win =
            (w_grid_model .>= Float32(w_lo_cfg)) .& (w_grid_model .<= Float32(w_hi_cfg))
        frac = sum(in_win) / max(length(in_win), 1)
        if frac < 0.8
            auto_lo = minimum(w_grid_model)
            auto_hi = maximum(w_grid_model)
            @info "Expanding NN training w-range to cover grid" cfg =
                (w_min = w_lo_cfg, w_max = w_hi_cfg) auto =
                (w_min = auto_lo, w_max = auto_hi) frac_in_window = frac
            use_opts = merge(opts, (w_min = Float64(auto_lo), w_max = Float64(auto_hi)))
        end
    end

    settings = solver_settings(
        use_opts,
        P,
        G,
        S;
        has_shocks = has_shocks,
        objective_default = objective_default,
    )
    scaler = FeatureScaler(P, G, S, settings)

    chain = build_dual_head_network(nn_input_dimension(P), settings.hidden_sizes)

    P_resid = scalar_params(P)
    model_cfg = build_model_config(P, U, scaler, P_resid, settings)

    training_result = train_consumption_network!(
        chain,
        settings,
        scaler,
        P_resid,
        G,
        S,
        train_rng,
        model_cfg,
    )

    best_state = training_result.best_state
    trained_model = select_model(chain, best_state)
    params =
        settings.use_cuda ? fmap(cu, state_parameters(best_state)) :
        state_parameters(best_state)
    states =
        settings.use_cuda ? fmap(cu, state_states(best_state)) : state_states(best_state)

    evaluation = evaluate_solution(
        trained_model,
        params,
        states,
        P_resid,
        P,
        G,
        S,
        scaler;
        settings = settings,
        U = U,
        rng = eval_rng,
    )

    runtime = (time_ns() - start_time) / 1e9
    opts_summary = build_options_summary(settings, training_result, runtime)
    converged = training_result.best_loss ≤ settings.target_loss

    eval_mc, eval_gh = maybe_dense_diagnostics(
        trained_model,
        params,
        states,
        P_resid,
        U,
        scaler,
        settings;
        G = G,
        S = S,
        P = P,
        rng = diag_rng,
    )

    _, w_grid = det_forward_inputs(G, P)

    return (;
        w_grid = w_grid,
        c = evaluation.c,
        a_next = evaluation.a_next,
        resid = evaluation.resid,
        iters = training_result.epochs_run,
        converged = converged,
        euler_rmse = evaluation.max_resid,
        max_resid = evaluation.max_resid,
        model_params = P,
        opts = opts_summary,
        eval_mc = eval_mc,
        eval_gh = eval_gh,
        rmse_history = training_result.rmse_history,
    )
end

# Fischer–Burmeister (Eq. 25)
@inline fb(a, h) = a + h .- sqrt.(a .^ 2 .+ h .^ 2)  # zero iff a≥0, h≥0, a*h=0

function loss_euler_fb_aio!(chain, ps, st, batch, model_cfg, rng)
    P = model_cfg.P
    if hasproperty(P, :A) &&
       hasproperty(P, :Σ) &&
       hasproperty(P, :y_dim) &&
       getproperty(P, :y_dim) > 1
        return loss_euler_fb_aio_csvar!(chain, ps, st, batch, model_cfg, rng)
    else
        return loss_euler_fb_aio_ar1!(chain, ps, st, batch, model_cfg, rng)
    end
end

"""
Dispatcher for the bias-corrected Monte Carlo FB objective.

Selects AR(1) vs CSVAR variants based on model parameters, mirroring the AiO
dispatcher.
"""
function loss_euler_fb_bcmc!(chain, ps, st, batch, model_cfg, rng)
    P = model_cfg.P
    if hasproperty(P, :A) &&
       hasproperty(P, :Σ) &&
       hasproperty(P, :y_dim) &&
       getproperty(P, :y_dim) > 1
        return loss_euler_fb_bcmc_csvar!(chain, ps, st, batch, model_cfg, rng)
    else
        return loss_euler_fb_bcmc_ar1!(chain, ps, st, batch, model_cfg, rng)
    end
end

function loss_euler_fb_aio_ar1!(chain, ps, st, batch, model_cfg, rng)
    P = model_cfg.P
    U = model_cfg.U
    scaler = model_cfg.scaler
    P_resid = model_cfg.P_resid
    settings = model_cfg.settings
    uprime = U.u_prime

    T = eltype(batch)
    C_MIN = T(1e-3)

    Rg = one(T) + T(P.r)
    μ = T(P_resid.y)
    feature_dim = size(batch, 1)
    mean_norm = batch[1, :]
    w_norm = batch[end, :]
    y0 = ((mean_norm .+ one(T)) ./ T(2)) .* T(scaler.mean_range) .+ T(scaler.mean_min)
    w0 = ((w_norm .+ one(T)) ./ T(2)) .* T(scaler.w_range) .+ T(scaler.w_min)
    z0 = log.(y0) .- μ
    out, st1 = Lux.apply(chain, batch, ps, st)
    c0 = vec(phi_to_consumption(out[:Φ], w0; min_c = C_MIN))
    # eta is a non-negative, dimensionless scaled multiplier proxy (≈ h/u'(c0))
    eta = T.(vec(ensure_row(out[:h])))
    a_term = @. one(T) - c0 / w0

    ρ = T(P.ρ_shock)
    σ_shocks = T(P.σ_shock)
    ε1 = randn_like(rng, z0)
    ε2 = randn_like(rng, z0)
    z1 = @. ρ * z0 + σ_shocks * ε1
    z2 = @. ρ * z0 + σ_shocks * ε2

    y1 = exp.(μ .+ z1)
    y2 = exp.(μ .+ z2)
    a1 = @. w0 - c0
    w1 = @. Rg * a1 + y1
    w2 = @. Rg * a1 + y2

    component_levels =
        hasproperty(P, :y) && P.y isa AbstractVector ? Float32.(exp.(collect(P.y))) :
        Float32[]
    X1n, X2n = ignore_derivatives() do
        # Build features without in-place slicing; then normalize out-of-place
        if feature_dim == 2
            X1 = vcat(reshape(Float32.(y1), 1, :), reshape(Float32.(w1), 1, :))
            X2 = vcat(reshape(Float32.(y2), 1, :), reshape(Float32.(w2), 1, :))
            return normalize_feature_batch(scaler, X1), normalize_feature_batch(scaler, X2)
        else
            comps = Matrix{Float32}(undef, feature_dim - 2, length(y1))
            for j = 1:(feature_dim-2)
                level =
                    j <= length(component_levels) ? component_levels[j] : Float32(exp(μ))
                comps[j, :] .= level
            end
            X1 = vcat(reshape(Float32.(y1), 1, :), comps, reshape(Float32.(w1), 1, :))
            X2 = vcat(reshape(Float32.(y2), 1, :), comps, reshape(Float32.(w2), 1, :))
            return normalize_feature_batch(scaler, X1), normalize_feature_batch(scaler, X2)
        end
    end
    out1, st1 = Lux.apply(chain, X1n, ps, st1)
    out2, st2 = Lux.apply(chain, X2n, ps, st1)

    c1 = vec(phi_to_consumption(out1[:Φ], w1; min_c = C_MIN))
    c2 = vec(phi_to_consumption(out2[:Φ], w2; min_c = C_MIN))

    β = T(P.β)
    q1 = @. β * Rg * uprime(c1) / uprime(c0)
    q2 = @. β * Rg * uprime(c2) / uprime(c0)

    # Complementarity between non-negativity (a_term >= 0) and multiplier eta >= 0
    fb_term = fb(a_term, eta)
    kt = @. fb_term^2
    # Euler-KKT residuals r = 1 - q - eta
    r1 = @. one(T) - q1 - eta
    r2 = @. one(T) - q2 - eta
    # Paper (eq. 30): AiO term is the product r1*r2, NOT squared
    aio_pen = r1 .* r2

    # Use adaptive v_h if available, otherwise fall back to static value
    v_h = if hasproperty(model_cfg, :adaptive_v_h_state)
        T(getfield(model_cfg.adaptive_v_h_state, :v_h)[])
    else
        hasproperty(model_cfg, :v_h) ? T(getfield(model_cfg, :v_h)) : one(T)
    end
    loss_vec = kt .+ v_h .* aio_pen

    max_abs_q = maximum(abs.(vcat(q1, q2)))

    return mean(loss_vec),
    (st1, (; kt_mean = mean(kt), aio_mean = mean(aio_pen), max_abs_q = max_abs_q))
end

function loss_euler_fb_aio_csvar!(chain, ps, st, batch, model_cfg, rng)
    P = model_cfg.P
    U = model_cfg.U
    scaler = model_cfg.scaler
    P_resid = model_cfg.P_resid
    settings = model_cfg.settings
    uprime = U.u_prime

    T = eltype(batch)
    C_MIN = T(1e-3)
    feature_dim = size(batch, 1)
    n = size(batch, 2)

    mean_vals, comps, w0 = denormalize_feature_batch(scaler, batch)
    y_dim = size(comps, 1) > 0 ? size(comps, 1) : 1
    y_matrix = size(comps, 1) == 0 ? reshape(mean_vals, 1, :) : comps

    out, st1 = Lux.apply(chain, batch, ps, st)
    c0 = vec(phi_to_consumption(out[:Φ], w0; min_c = C_MIN))
    eta = T.(vec(ensure_row(out[:h])))
    a_term = @. one(T) - c0 / w0
    a_curr = @. w0 - c0

    Rg = one(T) + T(P.r)
    β = T(P.β)
    v_h = hasproperty(model_cfg, :v_h) ? T(getfield(model_cfg, :v_h)) : one(T)

    A = Matrix{T}(P.A)
    Σ = Matrix{Float64}(P.Σ)
    chol = cholesky(Symmetric(Σ), check = false)
    L = Matrix{T}(chol.L)

    ε1 = randn(rng, T, y_dim, n)
    ε2 = randn(rng, T, y_dim, n)

    y1, y2, income1, income2 = ignore_derivatives() do
        # Non-mutating linear transitions
        y1_ = A * y_matrix .+ L * ε1  # size: y_dim × n
        y2_ = A * y_matrix .+ L * ε2

        # Build incomes without mutation
        inc1 = collect(map(i -> T(csvar_income(view(y1_, :, i))), 1:n))
        inc2 = collect(map(i -> T(csvar_income(view(y2_, :, i))), 1:n))
        (y1_, y2_, inc1, inc2)
    end

    w1 = @. Rg * a_curr + income1
    w2 = @. Rg * a_curr + income2

    X1 = ignore_derivatives() do
        build_feature_batch_from_states(scaler, y1, w1)
    end
    X2 = ignore_derivatives() do
        build_feature_batch_from_states(scaler, y2, w2)
    end
    out1, st1 = Lux.apply(chain, X1, ps, st1)
    out2, st2 = Lux.apply(chain, X2, ps, st1)

    c1 = vec(phi_to_consumption(out1[:Φ], w1; min_c = C_MIN))
    c2 = vec(phi_to_consumption(out2[:Φ], w2; min_c = C_MIN))

    q1 = @. β * Rg * uprime(c1) / uprime(c0)
    q2 = @. β * Rg * uprime(c2) / uprime(c0)

    fb_term = fb(a_term, eta)
    kt = @. fb_term^2
    r1 = @. one(T) - q1 - eta
    r2 = @. one(T) - q2 - eta
    # Paper (eq. 30): AiO term is the product r1*r2, NOT squared
    aio_pen = r1 .* r2

    # Use adaptive v_h if available, otherwise fall back to static value
    v_h = if hasproperty(model_cfg, :adaptive_v_h_state)
        T(getfield(model_cfg.adaptive_v_h_state, :v_h)[])
    else
        hasproperty(model_cfg, :v_h) ? T(getfield(model_cfg, :v_h)) : one(T)
    end
    loss_vec = kt .+ v_h .* aio_pen
    max_abs_q = maximum(abs.(vcat(q1, q2)))

    return mean(loss_vec),
    (st1, (; kt_mean = mean(kt), aio_mean = mean(aio_pen), max_abs_q = max_abs_q))
end

function loss_euler_fb_bcmc_ar1!(chain, ps, st, batch, model_cfg, rng; mode = :default)
    P = model_cfg.P
    U = model_cfg.U
    scaler = model_cfg.scaler
    P_resid = model_cfg.P_resid
    settings = model_cfg.settings
    uprime = U.u_prime

    T = eltype(batch)
    C_MIN = T(1e-3)

    Rg = one(T) + T(P.r)
    μ = T(P_resid.y)
    feature_dim = size(batch, 1)
    state_count = size(batch, 2)

    mean_norm = batch[1, :]
    w_norm = batch[end, :]
    y0 = ((mean_norm .+ one(T)) ./ T(2)) .* T(scaler.mean_range) .+ T(scaler.mean_min)
    w0 = ((w_norm .+ one(T)) ./ T(2)) .* T(scaler.w_range) .+ T(scaler.w_min)
    z0 = log.(y0) .- μ

    batch_mat = batch
    ndims(batch) == 1 && (batch_mat = reshape(batch, :, 1))
    out, st1 = Lux.apply(chain, batch_mat, ps, st)
    c0 = vec(phi_to_consumption(out[:Φ], w0; min_c = C_MIN))
    eta = T.(vec(ensure_row(out[:h])))
    a_term = @. one(T) - c0 / w0
    a_curr = @. w0 - c0

    auto = hasproperty(settings, :bcmc_auto_N) && settings.bcmc_auto_N
    N_cap = typemax(Int)
    if hasproperty(settings, :bcmc_budget_T) && settings.bcmc_budget_T !== nothing
        Tbudget = Int(getfield(settings, :bcmc_budget_T))
        M = state_count
        pairs_per_state = max(Tbudget / max(M, 1), 0)
        approxN = Int(floor((1 + sqrt(1 + 8 * pairs_per_state)) / 2))
        N_cap = max(approxN, 2)
    end

    if mode === :fb_scalar
        β = T(P.β)
        ρ = T(P.ρ_shock)
        Rg = one(T) + T(P.r)
        uprime = U.u_prime
        z_next = ρ .* (log.(y0) .- μ)
        y_next = exp.(μ .+ z_next)
        w_next = Rg .* a_curr .+ y_next
        Xn = ignore_derivatives() do
            if feature_dim == 2
                X = vcat(
                    reshape(T.(y_next), 1, state_count),
                    reshape(T.(w_next), 1, state_count),
                )
                normalize_feature_batch(scaler, X)
            else
                comps = Matrix{T}(undef, feature_dim - 2, state_count)
                for j = 1:(feature_dim-2)
                    comps[j, :] .= T(exp(μ))
                end
                X = vcat(
                    reshape(T.(y_next), 1, state_count),
                    comps,
                    reshape(T.(w_next), 1, state_count),
                )
                normalize_feature_batch(scaler, X)
            end
        end
        outn, _ = Lux.apply(chain, Xn, ps, st1)
        cn = vec(phi_to_consumption(outn[:Φ], w_next; min_c = C_MIN))
        qn = β .* Rg .* uprime(cn) ./ uprime(c0)
        residual = @. one(T) - qn - eta
        residual_sq = residual .* residual
        return residual_sq
    end

    baseN = settings.n_mc
    if auto && hasproperty(model_cfg, :bcmc_state)
        st_auto = getfield(model_cfg, :bcmc_state)
        baseN = Int(clamp(round(st_auto.n_eff[]), 2, typemax(Int)))
    end
    N = min(baseN, N_cap == typemax(Int) ? baseN : N_cap)
    N >= 2 || throw(ArgumentError("objective :euler_fb_bcmc requires N >= 2 (got $(N))"))
    n_eff = N

    β = T(P.β)
    ρ = T(P.ρ_shock)
    σ_shocks = T(P.σ_shock)
    v_h = hasproperty(model_cfg, :v_h) ? T(getfield(model_cfg, :v_h)) : one(T)

    fb_term = fb(a_term, eta)
    kt = @. fb_term^2

    # Non-mutating accumulators (same shape as h)
    g_sum = zero.(eta)        # accumulates (g^2) across draws
    g_sumsq = zero.(eta)      # accumulates (g^2)^2 across draws
    r_sum = zero.(eta)        # accumulates g across draws (for variance diagnostics)
    r_sumsq = zero.(eta)      # accumulates g^2 across draws (for variance diagnostics)
    max_abs_q = zero(T)

    component_levels =
        hasproperty(P, :y) && P.y isa AbstractVector ? T.(exp.(collect(P.y))) : T[]

    for draw = 1:N
        ε = randn_like(rng, z0)
        z_next = ρ .* z0 .+ σ_shocks .* ε
        y_next = exp.(μ .+ z_next)
        w_next = Rg .* a_curr .+ y_next

        Xn = ignore_derivatives() do
            if feature_dim == 2
                X = vcat(
                    reshape(T.(y_next), 1, state_count),
                    reshape(T.(w_next), 1, state_count),
                )
                normalize_feature_batch(scaler, X)
            else
                comps = Matrix{T}(undef, feature_dim - 2, state_count)
                for j = 1:(feature_dim-2)
                    level = j <= length(component_levels) ? component_levels[j] : T(exp(μ))
                    comps[j, :] .= level
                end
                X = vcat(
                    reshape(T.(y_next), 1, state_count),
                    comps,
                    reshape(T.(w_next), 1, state_count),
                )
                normalize_feature_batch(scaler, X)
            end
        end

        outn, st1 = Lux.apply(chain, Xn, ps, st1)
        cn = vec(phi_to_consumption(outn[:Φ], w_next; min_c = C_MIN))
        qn = β .* Rg .* uprime(cn) ./ uprime(c0)
        max_abs_q = max(max_abs_q, maximum(abs.(qn)))

        residual = @. one(T) - qn - eta
        residual_sq = residual .* residual
        g_sum = g_sum .+ residual_sq
        g_sumsq = g_sumsq .+ residual_sq .* residual_sq
        r_sum = r_sum .+ residual
        r_sumsq = r_sumsq .+ residual_sq
    end

    denom = T(N) * (T(N) - one(T))
    bcmc = (g_sum .* g_sum .- g_sumsq) ./ denom

    # Optional diagnostics: empirical variance of g across draws per state, averaged
    invN = one(T) / T(N)
    var_vec = clamp.(r_sumsq .* invN .- (r_sum .* invN) .* (r_sum .* invN), zero(T), T(Inf))
    gvar_mean = mean(var_vec)

    loss_vec = kt .+ v_h .* bcmc
    proxy_state =
        hasproperty(model_cfg, :bcmc_state) ? getfield(model_cfg, :bcmc_state) : nothing
    A_proxy = proxy_state === nothing ? NaN : Float64(proxy_state.A[])
    B_proxy = proxy_state === nothing ? NaN : Float64(proxy_state.B[])
    ratio_proxy = isnan(B_proxy) || abs(B_proxy) ≤ EPS_VAR ? Inf : A_proxy / B_proxy
    return mean(loss_vec),
    (
        st1,
        (;
            kt_mean = mean(kt),
            bcmc_mean = mean(bcmc),
            gvar_mean = gvar_mean,
            n_eff = n_eff,
            max_abs_q = max_abs_q,
            bcmc_A = A_proxy,
            bcmc_B = B_proxy,
            bcmc_ratio = ratio_proxy,
        ),
    )
end

function loss_euler_fb_bcmc_csvar!(chain, ps, st, batch, model_cfg, rng; mode = :default)
    P = model_cfg.P
    U = model_cfg.U
    scaler = model_cfg.scaler
    settings = model_cfg.settings
    uprime = U.u_prime

    T = eltype(batch)
    C_MIN = T(1e-3)
    feature_dim = size(batch, 1)
    n = size(batch, 2)

    mean_vals, comps, w0 = denormalize_feature_batch(scaler, batch)
    y_dim = size(comps, 1) > 0 ? size(comps, 1) : 1
    y_matrix = size(comps, 1) == 0 ? reshape(mean_vals, 1, :) : comps

    batch_mat = batch
    ndims(batch) == 1 && (batch_mat = reshape(batch, :, 1))
    out, st1 = Lux.apply(chain, batch_mat, ps, st)
    c0 = vec(phi_to_consumption(out[:Φ], w0; min_c = C_MIN))
    eta = T.(vec(ensure_row(out[:h])))
    a_term = @. one(T) - c0 / w0
    a_curr = @. w0 - c0

    auto = hasproperty(settings, :bcmc_auto_N) && settings.bcmc_auto_N
    N_cap = typemax(Int)
    if hasproperty(settings, :bcmc_budget_T) && settings.bcmc_budget_T !== nothing
        Tbudget = Int(getfield(settings, :bcmc_budget_T))
        M = n
        pairs_per_state = max(Tbudget / max(M, 1), 0)
        approxN = Int(floor((1 + sqrt(1 + 8 * pairs_per_state)) / 2))
        N_cap = max(approxN, 2)
    end

    if mode === :fb_scalar
        Rg = one(T) + T(P.r)
        β = T(P.β)
        uprime = U.u_prime
        A = Matrix{T}(P.A)
        Σ = Matrix{Float64}(P.Σ)
        chol = cholesky(Symmetric(Σ), check = false)
        L = Matrix{T}(chol.L)
        ε = zeros(T, size(L, 2), n)
        y_next, income_next = ignore_derivatives() do
            y_ = A * y_matrix .+ L * ε
            inc = collect(map(i -> T(csvar_income(view(y_, :, i))), 1:n))
            (y_, inc)
        end
        w_next = Rg .* a_curr .+ income_next
        Xn = ignore_derivatives() do
            build_feature_batch_from_states(scaler, y_next, w_next)
        end
        outn, _ = Lux.apply(chain, Xn, ps, st1)
        c_next = vec(phi_to_consumption(outn[:Φ], w_next; min_c = C_MIN))
        q_next = β .* Rg .* uprime(c_next) ./ uprime(c0)
        residual = @. one(T) - q_next - eta
        residual_sq = residual .* residual
        return residual_sq
    end

    baseN = settings.n_mc
    if auto && hasproperty(model_cfg, :bcmc_state)
        st_auto = getfield(model_cfg, :bcmc_state)
        baseN = Int(clamp(round(st_auto.n_eff[]), 2, typemax(Int)))
    end
    N = min(baseN, N_cap == typemax(Int) ? baseN : N_cap)
    N >= 2 || throw(ArgumentError("objective :euler_fb_bcmc requires N >= 2 (got $(N))"))
    n_eff = N

    Rg = one(T) + T(P.r)
    β = T(P.β)
    v_h = hasproperty(model_cfg, :v_h) ? T(getfield(model_cfg, :v_h)) : one(T)

    A = Matrix{T}(P.A)
    Σ = Matrix{Float64}(P.Σ)
    chol = cholesky(Symmetric(Σ), check = false)
    L = Matrix{T}(chol.L)

    fb_term = fb(a_term, eta)
    kt = @. fb_term^2

    g_sum = zero.(eta)
    g_sumsq = zero.(eta)
    r_sum = zero.(eta)
    r_sumsq = zero.(eta)
    max_abs_q = zero(T)

    for draw = 1:N
        ε = randn(rng, T, y_dim, n)
        y_next, income_next = ignore_derivatives() do
            y_ = A * y_matrix .+ L * ε
            inc = collect(map(i -> T(csvar_income(view(y_, :, i))), 1:n))
            (y_, inc)
        end
        w_next = Rg .* a_curr .+ income_next

        Xn = ignore_derivatives() do
            build_feature_batch_from_states(scaler, y_next, w_next)
        end

        outn, st1 = Lux.apply(chain, Xn, ps, st1)
        c_next = vec(phi_to_consumption(outn[:Φ], w_next; min_c = C_MIN))
        q_next = β .* Rg .* uprime(c_next) ./ uprime(c0)
        max_abs_q = max(max_abs_q, maximum(abs.(q_next)))

        residual = @. one(T) - q_next - eta
        residual_sq = residual .* residual
        g_sum = g_sum .+ residual_sq
        g_sumsq = g_sumsq .+ residual_sq .* residual_sq
        r_sum = r_sum .+ residual
        r_sumsq = r_sumsq .+ residual_sq
    end

    denom = T(N) * (T(N) - one(T))
    bcmc = (g_sum .* g_sum .- g_sumsq) ./ denom

    invN = one(T) / T(N)
    var_vec = clamp.(r_sumsq .* invN .- (r_sum .* invN) .* (r_sum .* invN), zero(T), T(Inf))
    gvar_mean = mean(var_vec)

    loss_vec = kt .+ v_h .* bcmc
    proxy_state =
        hasproperty(model_cfg, :bcmc_state) ? getfield(model_cfg, :bcmc_state) : nothing
    A_proxy = proxy_state === nothing ? NaN : Float64(proxy_state.A[])
    B_proxy = proxy_state === nothing ? NaN : Float64(proxy_state.B[])
    ratio_proxy = isnan(B_proxy) || abs(B_proxy) ≤ EPS_VAR ? Inf : A_proxy / B_proxy
    return mean(loss_vec),
    (
        st1,
        (;
            kt_mean = mean(kt),
            bcmc_mean = mean(bcmc),
            gvar_mean = gvar_mean,
            n_eff = n_eff,
            max_abs_q = max_abs_q,
            bcmc_A = A_proxy,
            bcmc_B = B_proxy,
            bcmc_ratio = ratio_proxy,
        ),
    )
end

end # module
