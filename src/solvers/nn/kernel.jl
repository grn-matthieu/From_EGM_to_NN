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
using ..DataNN: sample_training
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
include("losses.jl")
using .NNLosses:
    build_loss_function,
    flatten_sum_squares,
    _var_bcmc_given_N,
    estimate_linearized_components,
    _shift_features,
    _shift_eps_state,
    _shift_s_state,
    _grad_to_vector,
    suggest_bcmc_N,
    randn_like,
    fill_like
include("training_utils.jl")
include("settings.jl")
include("training_loop.jl")
include("evaluation.jl")


export solve_nn, solver_settings

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

function build_model_config(P, U, scaler, settings)
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
        settings = settings,
        sigma_shocks = settings.sigma_shocks,
        bcmc_state = bcmc_state,
    )
end

function maybe_dense_diagnostics(
    model,
    params,
    states,
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
        (S !== nothing && isdefined(S, :process) && S.process == :gaussian_linear)
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
        mc_diag =
            fmc(model, params, states, U, scaler, settings; G = G, S = S, P = P, rng = rng)
    catch err
        if err isa MethodError
            mc_diag = fmc(model, params, states, U, scaler, settings)
        else
            rethrow()
        end
    end

    try
        gh_diag =
            fgh(model, params, states, U, scaler, settings; G = G, S = S, P = P, rng = rng)
    catch err
        if err isa MethodError
            gh_diag = fgh(model, params, states, U, scaler, settings)
        else
            rethrow()
        end
    end
    return mc_diag, gh_diag
end

function build_options_summary(
    settings,
    training_result,
    runtime;
    evaluation_runtime = nothing,
    diagnostics_runtime = nothing,
    total_runtime = nothing,
)
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
        evaluation_runtime = evaluation_runtime,
        diagnostics_runtime = diagnostics_runtime,
        total_runtime = total_runtime,
        verbose = settings.verbose,
        batches_per_epoch = training_result.batches_per_epoch,
        skip_final_eval = settings.skip_final_eval,
        device = :cpu,
    )
end

function solve_nn(model; opts = nothing, settings = nothing, rng = nothing)
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
    # If no settings were provided by the caller (backwards compat), build
    # them here using the same heuristics as before. New preferred flow is
    # for the method layer to construct and pass `settings`.
    if settings === nothing
        is_csvar = !isnothing(S) && isdefined(S, :process) && S.process == :gaussian_linear
        has_shocks = !isnothing(S) && !is_csvar
        objective_default =
            is_csvar ? :euler_residual : has_shocks ? :euler_fb_aio : :euler_residual

        settings = solver_settings(
            opts,
            P,
            G,
            S;
            has_shocks = has_shocks,
            objective_default = objective_default,
        )
    end
    scaler = FeatureScaler(P, G, S, settings)

    chain = build_dual_head_network(nn_input_dimension(P), settings.hidden_sizes)

    model_cfg = build_model_config(P, U, scaler, settings)

    training_result =
        train_consumption_network!(chain, settings, scaler, G, S, train_rng, model_cfg)

    training_end = time_ns()

    best_state = training_result.best_state
    trained_model = select_model(chain, best_state)
    params = state_parameters(best_state)
    states = state_states(best_state)

    evaluation = evaluate_solution(
        trained_model,
        params,
        states,
        P,
        G,
        S,
        scaler;
        settings = settings,
        U = U,
        rng = eval_rng,
        skip_residuals = settings.skip_final_eval,
    )

    evaluation_end = time_ns()

    eval_mc = nothing
    eval_gh = nothing
    diagnostics_runtime = 0.0
    diagnostics_end = evaluation_end
    if !settings.skip_final_eval
        eval_mc, eval_gh = maybe_dense_diagnostics(
            trained_model,
            params,
            states,
            U,
            scaler,
            settings;
            G = G,
            S = S,
            P = P,
            rng = diag_rng,
        )
        diagnostics_end = time_ns()
        diagnostics_runtime = (diagnostics_end - evaluation_end) / 1e9
    end

    training_runtime = (training_end - start_time) / 1e9
    evaluation_runtime = (evaluation_end - training_end) / 1e9
    total_runtime = training_runtime + evaluation_runtime + diagnostics_runtime

    opts_summary = build_options_summary(
        settings,
        training_result,
        training_runtime;
        evaluation_runtime = evaluation_runtime,
        diagnostics_runtime = diagnostics_runtime,
        total_runtime = total_runtime,
    )
    converged = training_result.best_loss ≤ settings.target_loss

    _, w_grid = grid_forward_inputs(G, P)

    resid_array = evaluation.resid
    rmse_val = sqrt(mean(abs2, vec(resid_array)))

    return (;
        w_grid = w_grid,
        c = evaluation.c,
        a_next = evaluation.a_next,
        resid = evaluation.resid,
        iters = training_result.epochs_run,
        converged = converged,
        euler_rmse = rmse_val,
        max_resid = evaluation.max_resid,
        model_params = P,
        opts = opts_summary,
        eval_mc = eval_mc,
        eval_gh = eval_gh,
        rmse_history = training_result.rmse_history,
        N_history = training_result.N_history,
        v_h_history = training_result.v_h_history,
        loss_history = training_result.loss_history,
        best_loss = training_result.best_loss,
        final_loss = isempty(training_result.loss_history) ? NaN :
                     training_result.loss_history[end],
    )
end

# Fischer–Burmeister (Eq. 25)
@inline fb(a, h) = a + h .- sqrt.(a .^ 2 .+ h .^ 2)

"""Helper to build feature matrix for AR(1) models."""
@inline function build_ar1_features(y, w, feature_dim, component_levels, μ)
    T = eltype(y)
    if feature_dim == 2
        return vcat(reshape(T.(y), 1, :), reshape(T.(w), 1, :))
    else
        n = length(y)
        comps = Matrix{T}(undef, feature_dim - 2, n)
        for j = 1:(feature_dim-2)
            comps[j, :] .= j <= length(component_levels) ? component_levels[j] : T(exp(μ))
        end
        return vcat(reshape(T.(y), 1, :), comps, reshape(T.(w), 1, :))
    end
end

function loss_euler_fb_aio!(chain, ps, st, batch, model_cfg, rng)
    P = model_cfg.P
    if isdefined(P, :A) && isdefined(P, :Σ) && isdefined(P, :y_dim) && P.y_dim > 1
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
    if isdefined(P, :A) && isdefined(P, :Σ) && isdefined(P, :y_dim) && P.y_dim > 1
        return loss_euler_fb_bcmc_csvar!(chain, ps, st, batch, model_cfg, rng)
    else
        return loss_euler_fb_bcmc_ar1!(chain, ps, st, batch, model_cfg, rng)
    end
end

function loss_euler_fb_aio_ar1!(chain, ps, st, batch, model_cfg, rng)
    P = model_cfg.P
    U = model_cfg.U
    scaler = model_cfg.scaler
    settings = model_cfg.settings
    uprime = U.u_prime

    T = eltype(batch)
    C_MIN = T(1e-12)

    ρ = T(P.ρ_shock)
    σ_shocks = T(P.σ_shock)
    R = one(T) + T(P.r)
    β = T(P.β)
    component_levels = T[]
    if P.y isa AbstractVector
        μ = T(mean(T.(collect(P.y))))
        component_levels = T.(exp.(collect(P.y)))
    else
        μ = T(P.y)
        component_levels = Float32[]
    end

    feature_dim = size(batch, 1)
    mean_norm = batch[1, :]
    w_norm = batch[end, :]

    # First pass to retrieve first-period values
    y0 = ((mean_norm .+ one(T)) ./ T(2)) .* T(scaler.mean_range) .+ T(scaler.mean_min)
    w0 = ((w_norm .+ one(T)) ./ T(2)) .* T(scaler.w_range) .+ T(scaler.w_min)
    z0 = log.(y0) .- μ

    # Network outputs
    out, st1 = Lux.apply(chain, batch, ps, st)
    Φ_out = vec(ensure_row(out[:Φ]))
    h_out = T.(vec(ensure_row(out[:h])))

    # Implies first period consumption
    c0 = vec(phi_to_consumption(Φ_out, w0; min_c = C_MIN))
    a0 = @. w0 - c0 # that yields the current period asset level

    # Draw two future independent shocks per state
    ε1 = randn_like(rng, z0)
    ε2 = randn_like(rng, z0)

    # implies second period incomes and wealths
    y_next_1 = @. exp.(ρ * z0 + σ_shocks * ε1)
    y_next_2 = @. exp.(ρ * z0 + σ_shocks * ε2)
    w_next_1 = @. R * a0 + y_next_1
    w_next_2 = @. R * a0 + y_next_2

    # Build both normalized input features matrixs
    X_next_1, X_next_2 = ignore_derivatives() do
        X1 = build_ar1_features(y_next_1, w_next_1, feature_dim, component_levels, μ)
        X2 = build_ar1_features(y_next_2, w_next_2, feature_dim, component_levels, μ)
        normalize_feature_batch(scaler, X1), normalize_feature_batch(scaler, X2)
    end

    # Second pass to get next period outputs for both shock draws
    out_next_1, _ = Lux.apply(chain, X_next_1, ps, st1)
    out_next_2, _ = Lux.apply(chain, X_next_2, ps, st1)
    # and convert to next period consumption
    c_next_1 = vec(phi_to_consumption(out_next_1[:Φ], w_next_1; min_c = C_MIN))
    c_next_2 = vec(phi_to_consumption(out_next_2[:Φ], w_next_2; min_c = C_MIN))

    u_ratio_1 = @. β * R * uprime(c_next_1) / uprime(c0)
    u_ratio_2 = @. β * R * uprime(c_next_2) / uprime(c0)

    # first part of the loss is the squared FB function
    fb_a_term = @. one(T) - c0 / max(w0, eps(T))
    fb_h_term = one(T) .- h_out
    fb_term = fb(fb_a_term, fb_h_term)
    fb_term_sq = @. fb_term^2

    # second term compares h to the u_ratio
    r1 = @. u_ratio_1 - h_out
    r2 = @. u_ratio_2 - h_out
    aio_pen = r1 .* r2

    # v_h to weight the AiO penalty, may be adaptive
    v_h = if isdefined(model_cfg, :adaptive_v_h_state)
        T(model_cfg.adaptive_v_h_state.v_h[])
    else
        isdefined(model_cfg, :v_h) ? T(model_cfg.v_h) : one(T)
    end

    loss_vec = fb_term_sq .+ v_h .* aio_pen

    return mean(loss_vec),
    (st1, (; mean_fb_term_sq = mean(fb_term_sq), mean_aio = mean(aio_pen)))
end

function loss_euler_fb_aio_csvar!(chain, ps, st, batch, model_cfg, rng)
    P = model_cfg.P
    U = model_cfg.U
    scaler = model_cfg.scaler
    settings = model_cfg.settings
    uprime = U.u_prime

    T = eltype(batch)
    C_MIN = T(1e-12)
    feature_dim = size(batch, 1)
    n = size(batch, 2)

    mean_vals, comps, w0 = denormalize_feature_batch(scaler, batch)
    y_dim = size(comps, 1) > 0 ? size(comps, 1) : 1
    y_matrix = size(comps, 1) == 0 ? reshape(mean_vals, 1, :) : comps

    out, st1 = Lux.apply(chain, batch, ps, st)
    c0 = vec(phi_to_consumption(out[:Φ], w0; min_c = C_MIN))
    eta = T.(vec(ensure_row(out[:h])))
    a_curr = @. w0 - c0

    Rg = one(T) + T(P.r)
    β = T(P.β)

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

    fb_a_term = @. one(T) - c0 / max(w0, eps(T))
    fb_h_term = one(T) .- eta
    fb_term = fb(fb_a_term, fb_h_term)
    kt = @. fb_term^2
    r1 = @. q1 - eta
    r2 = @. q2 - eta
    # Paper (eq. 30): AiO term is the product r1*r2, NOT squared
    aio_pen = r1 .* r2

    # Use adaptive v_h if available, otherwise fall back to static value
    v_h = if isdefined(model_cfg, :adaptive_v_h_state)
        T(model_cfg.adaptive_v_h_state.v_h[])
    else
        isdefined(model_cfg, :v_h) ? T(model_cfg.v_h) : one(T)
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
    settings = model_cfg.settings
    uprime = U.u_prime

    T = eltype(batch)
    C_MIN = T(1e-12)

    Rg = one(T) + T(P.r)
    if isdefined(P, :y) && P.y isa AbstractVector
        μ = T(mean(Float64.(collect(P.y))))
    else
        μ = T(P.y)
    end
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

    auto = isdefined(settings, :bcmc_auto_N) && settings.bcmc_auto_N
    N_cap = typemax(Int)
    if isdefined(settings, :bcmc_budget_T) && settings.bcmc_budget_T !== nothing
        Tbudget = Int(settings.bcmc_budget_T)
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
            X = build_ar1_features(y_next, w_next, feature_dim, T[], μ)
            normalize_feature_batch(scaler, X)
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
    # Use adaptive v_h if available, otherwise fall back to static value
    v_h = if isdefined(model_cfg, :adaptive_v_h_state)
        T(model_cfg.adaptive_v_h_state.v_h[])
    else
        isdefined(model_cfg, :v_h) ? T(model_cfg.v_h) : one(T)
    end

    fb_term = fb(a_term, eta)
    kt = @. fb_term^2

    # Non-mutating accumulators (same shape as h)
    r_sum = zero.(eta)        # accumulates r across draws, r = q - eta
    r_sumsq = zero.(eta)      # accumulates r^2 across draws
    max_abs_q = zero(T)

    component_levels =
        isdefined(P, :y) && P.y isa AbstractVector ? T.(exp.(collect(P.y))) : T[]

    for draw = 1:N
        ε = randn_like(rng, z0)
        z_next = ρ .* z0 .+ σ_shocks .* ε
        y_next = exp.(μ .+ z_next)
        w_next = Rg .* a_curr .+ y_next

        Xn = ignore_derivatives() do
            X = build_ar1_features(y_next, w_next, feature_dim, component_levels, μ)
            normalize_feature_batch(scaler, X)
        end

        outn, st1 = Lux.apply(chain, Xn, ps, st1)
        cn = vec(phi_to_consumption(outn[:Φ], w_next; min_c = C_MIN))
        qn = β .* Rg .* uprime(cn) ./ uprime(c0)
        max_abs_q = max(max_abs_q, maximum(abs.(qn)))

        r = @. qn - eta
        r_sq = r .* r
        r_sum = r_sum .+ r
        r_sumsq = r_sumsq .+ r_sq
    end

    denom = T(N) * (T(N) - one(T))
    # Unbiased estimator using all distinct pairs: average of r_i * r_j, i≠j
    bcmc = (r_sum .* r_sum .- r_sumsq) ./ denom

    # Optional diagnostics: empirical variance of g across draws per state, averaged
    invN = one(T) / T(N)
    var_vec = clamp.(r_sumsq .* invN .- (r_sum .* invN) .* (r_sum .* invN), zero(T), T(Inf))
    gvar_mean = mean(var_vec)

    loss_vec = kt .+ v_h .* bcmc
    proxy_state = isdefined(model_cfg, :bcmc_state) ? model_cfg.bcmc_state : nothing
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
    C_MIN = T(1e-12)
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

    auto = isdefined(settings, :bcmc_auto_N) && settings.bcmc_auto_N
    N_cap = typemax(Int)
    if isdefined(settings, :bcmc_budget_T) && settings.bcmc_budget_T !== nothing
        Tbudget = Int(settings.bcmc_budget_T)
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
    if auto && isdefined(model_cfg, :bcmc_state)
        st_auto = model_cfg.bcmc_state
        baseN = Int(clamp(round(st_auto.n_eff[]), 2, typemax(Int)))
    end
    N = min(baseN, N_cap == typemax(Int) ? baseN : N_cap)
    N >= 2 || throw(ArgumentError("objective :euler_fb_bcmc requires N >= 2 (got $(N))"))
    n_eff = N

    Rg = one(T) + T(P.r)
    β = T(P.β)
    # Use adaptive v_h if available, otherwise fall back to static value
    v_h = if isdefined(model_cfg, :adaptive_v_h_state)
        T(model_cfg.adaptive_v_h_state.v_h[])
    else
        isdefined(model_cfg, :v_h) ? T(model_cfg.v_h) : one(T)
    end

    A = Matrix{T}(P.A)
    Σ = Matrix{Float64}(P.Σ)
    chol = cholesky(Symmetric(Σ), check = false)
    L = Matrix{T}(chol.L)

    fb_term = fb(a_term, eta)
    kt = @. fb_term^2

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

        r = @. q_next - eta
        r_sq = r .* r
        r_sum = r_sum .+ r
        r_sumsq = r_sumsq .+ r_sq
    end

    denom = T(N) * (T(N) - one(T))
    # Unbiased estimator using all distinct pairs: average of r_i * r_j, i≠j
    bcmc = (r_sum .* r_sum .- r_sumsq) ./ denom

    invN = one(T) / T(N)
    var_vec = clamp.(r_sumsq .* invN .- (r_sum .* invN) .* (r_sum .* invN), zero(T), T(Inf))
    gvar_mean = mean(var_vec)

    loss_vec = kt .+ v_h .* bcmc
    proxy_state = isdefined(model_cfg, :bcmc_state) ? model_cfg.bcmc_state : nothing
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
