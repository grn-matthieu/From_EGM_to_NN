"""
NNKernel

High-level orchestration of the neural-network solver for consumption policies.
The heavy lifting (pre-processing, training loop and evaluation utilities) lives
in dedicated helpers so that this file focuses on the solver flow.
"""
module NNKernel

using ..API: get_grids, get_shocks, get_utility
using ..CommonInterp: InterpKind, LinearInterp
using ..DataNN: generate_dataset
using ..EulerResiduals: euler_resid_det_grid, euler_resid_stoch_grid
using Lux
using Optimisers
using Random
using Printf
using Statistics: mean, quantile

include("mixed_precision.jl")
include("preprocessing.jl")
include("training_loop.jl")

export solve_nn

struct EvaluationResult
    c::Any
    a_next::Any
    resid::Any
    max_resid::Float64
end

const H_ALPHA = 5.0f0


function evaluate_deterministic(model, params, states, P_resid, P, G, scaler)
    X_forward, _ = det_forward_inputs(G)
    normalize_feature_batch!(scaler, X_forward)
    pred = run_model(model, params, states, X_forward)

    if pred isa NamedTuple
        Φ = pred[:Φ]
        # recover a grid and compute cash-on-hand
        a_grid_f32 = float32_vector(G[:a].grid)
        w = cash_on_hand(a_grid_f32, 0.0f0, P_resid, scaler.has_shocks)
        # align Φ to row
        if ndims(Φ) == 2 && size(Φ, 1) == 1
            Φ_row = Φ
        elseif ndims(Φ) == 2 && size(Φ, 2) == 1
            Φ_row = permutedims(Φ)
        else
            Φ_row = reshape(vec(Φ), 1, :)
        end
        c_vec = @. clamp(Φ_row[1, :] .* w, 1.0f-8, Inf)
        a_grid_f32, _, c_vec_f32 = det_residual_inputs(c_vec, G)
        residuals = euler_resid_det_grid(P_resid, a_grid_f32, c_vec_f32)
        c_on_grid = convert_to_grid_eltype(G[:a].grid, c_vec)
    else
        predictions = pred
        a_grid_f32, c_vec, c_vec_f32 = det_residual_inputs(predictions, G)
        residuals = euler_resid_det_grid(P_resid, a_grid_f32, c_vec_f32)
        c_on_grid = convert_to_grid_eltype(G[:a].grid, c_vec)
    end
    max_resid = maximum(abs.(residuals))
    R = 1 + P.r
    y = P.y
    a_next = @. R * G[:a].grid + y - c_on_grid
    a_next = clamp_to_asset_bounds(a_next, G[:a])
    return EvaluationResult(c_on_grid, a_next, residuals, max_resid)
end

function evaluate_stochastic(model, params, states, P_resid, P, G, S, scaler)
    X_eval, _ = generate_dataset(G, S; mode = :full)
    normalize_samples!(scaler, X_eval)
    batch = prepare_training_batch(X_eval)
    pred = run_model(model, params, states, batch)
    if pred isa NamedTuple
        Φ = pred[:Φ]
        a_f32 = float32_vector(G[:a].grid)
        z_f32 = float32_vector(S.zgrid)
        Na = length(a_f32)
        Nz = length(z_f32)
        A = repeat(a_f32, inner = Nz)
        Z = repeat(z_f32, outer = Na)
        w = cash_on_hand(A, Z, P_resid, scaler.has_shocks)
        if ndims(Φ) == 2 && size(Φ, 1) == 1
            Φ_row = Φ
        elseif ndims(Φ) == 2 && size(Φ, 2) == 1
            Φ_row = permutedims(Φ)
        else
            Φ_row = reshape(vec(Φ), 1, :)
        end
        c_row = @. clamp(Φ_row[1, :] .* w, 1.0f-8, Inf)
        a_grid_f32, z_grid_f32, Pz_f32, c_matrix, c_matrix_f32 =
            stoch_residual_inputs(c_row, G, S)
    else
        predictions = pred
        a_grid_f32, z_grid_f32, Pz_f32, c_matrix, c_matrix_f32 =
            stoch_residual_inputs(predictions, G, S)
    end
    residuals =
        euler_resid_stoch_grid(P_resid, a_grid_f32, z_grid_f32, Pz_f32, c_matrix_f32)
    c_on_grid = convert_to_grid_eltype(G[:a].grid, c_matrix)
    max_resid = maximum(abs.(residuals))
    R = 1 + P.r
    y = P.y
    a_next = @. R * G[:a].grid + y - c_on_grid
    a_next = clamp_to_asset_bounds(a_next, G[:a])
    return EvaluationResult(c_on_grid, a_next, residuals, max_resid)
end

function build_options_summary(settings, training_result, runtime)
    return (;
        epochs = settings.epochs,
        epochs_run = training_result.epochs_run,
        batch = training_result.batch_size,
        lr = settings.learning_rate,
        runtime = runtime,
        verbose = settings.verbose,
        batches_per_epoch = training_result.batches_per_epoch,
    )
end

function solve_nn(model; opts = nothing)
    P, G, S, U = get_params(model), get_grids(model), get_shocks(model), get_utility(model)
    start_time = time_ns()
    scaler = FeatureScaler(G, S)
    settings = solver_settings(opts; has_shocks = scaler.has_shocks)

    # Utility hooks from the model's utility object
    uprime = U.u_prime
    uprime_inv = U.u_prime_inv
    σ_crra = U.σ
    # -- model: shared trunk + two heads (phi in (0,1), h > 0)
    function build_dual_head_network(input_dim::Int, hidden::NTuple{2,Int})
        H1, H2 = hidden
        # trunk ends with a Dense producing 2 outputs (one per head)
        trunk =
            Chain(Dense(input_dim, H1, leakyrelu), Dense(H1, H2, leakyrelu), Dense(H2, 2))

        # final mapping: split the 2×N output into Φ and h and apply activations
        model = Chain(trunk, x -> begin
            # x is 2×N; first row -> pre-Φ, second row -> pre-h
            pre = x
            φ_pre = view(pre, 1:1, :)
            h_pre = view(pre, 2:2, :)
            (Φ = sigmoid.(φ_pre), h = exp.(H_ALPHA .* tanh.(h_pre)))
        end)
        return model
    end

    input_dimension(S) = isnothing(S) ? 1 : 2

    chain = build_dual_head_network(input_dimension(S), settings.hidden_sizes)
    # ScalarParams expects exact fields in P (σ, β, r, y). Validation lives in the config validator.
    P_resid = scalar_params(P)
    # model_cfg provides fields expected by custom losses (P, U, v_h, scaler, P_resid, settings)
    model_cfg = (
        P = P,
        U = U,
        v_h = settings.v_h,
        scaler = scaler,
        P_resid = P_resid,
        settings = settings,
        sigma_shocks = settings.sigma_shocks,
    )
    rng = Random.default_rng()
    training_result =
        train_consumption_network!(chain, settings, scaler, P_resid, G, S, model_cfg, rng)
    best_state = training_result.best_state
    trained_model = select_model(chain, best_state)
    params = state_parameters(best_state)
    states = state_states(best_state)

    evaluation =
        isnothing(S) ?
        evaluate_deterministic(trained_model, params, states, P_resid, P, G, scaler) :
        evaluate_stochastic(trained_model, params, states, P_resid, P, G, S, scaler)

    runtime = (time_ns() - start_time) / 1e9
    opts_summary = build_options_summary(settings, training_result, runtime)
    converged = training_result.best_loss ≤ settings.target_loss

    # Attach dense evaluation metrics when stochastic shocks are present
    eval_mc =
        isnothing(S) ? nothing :
        eval_euler_residuals_mc(
            trained_model,
            params,
            states,
            P_resid,
            U,
            scaler,
            settings;
            N = 8192,
        )
    eval_gh =
        isnothing(S) ? nothing :
        eval_euler_residuals_gh(
            trained_model,
            params,
            states,
            P_resid,
            U,
            scaler,
            settings;
            N = 4096,
        )

    return (;
        a_grid = G[:a].grid,
        c = evaluation.c,
        a_next = evaluation.a_next,
        resid = evaluation.resid,
        iters = training_result.epochs_run,
        converged = converged,
        max_resid = evaluation.max_resid,
        model_params = P,
        opts = opts_summary,
        eval_mc = eval_mc,
        eval_gh = eval_gh,
    )
end


# Fischer–Burmeister (Eq. 25)
@inline fb(a, h) = a + h .- sqrt.(a .^ 2 .+ h .^ 2)  # zero iff a≥0, h≥0, a*h=0

function loss_euler_fb_aio!(chain, ps, st, batch, model_cfg, rng)
    P = model_cfg.P
    U = model_cfg.U
    scaler = model_cfg.scaler
    P_resid = model_cfg.P_resid
    settings = model_cfg.settings
    uprime = U.u_prime

    T = eltype(batch)
    C_MIN = T(1e-3)

    # 1) Unnormalize current (a0, z0) from features×samples batch
    a0 = ((batch[1, :] .+ one(T)) ./ T(2)) .* T(scaler.a_range) .+ T(scaler.a_min)
    z0 =
        settings.has_shocks ?
        ((batch[2, :] .+ one(T)) ./ T(2)) .* T(scaler.z_range) .+ T(scaler.z_min) :
        fill(zero(T), size(a0))

    # 2) Forward at t to get φ,h then compute c0 and w0
    out, st1 = Lux.apply(chain, batch, ps, st)
    φ = vec(out[:Φ])                 # in (0,1)
    h = vec(out[:h])                 # >0

    w0 = cash_on_hand(a0, z0, P_resid, settings.has_shocks) |> x -> T.(x)
    c0 = clamp.(φ .* w0, C_MIN, T(Inf))
    a_term = @. one(T) - c0 / w0     # KT piece a = 1 - c/w ≥ 0

    # 3) Shocks and next state (a1,z1)
    ρ = T(P.ρ)
    # Shock std: prefer explicit override in model_cfg.sigma_shocks, otherwise use P.σ_shocks
    σ_shocks =
        hasproperty(model_cfg, :sigma_shocks) && model_cfg.sigma_shocks !== nothing ?
        T(model_cfg.sigma_shocks) : T(P.σ_shocks)
    ε1 = randn!(rng, similar(z0))
    ε2 = randn!(rng, similar(z0))
    z1 = @. ρ * z0 + σ_shocks * ε1
    z2 = @. ρ * z0 + σ_shocks * ε2

    a1 = @. w0 - c0                  # next assets
    a2 = a1                          # same a′ for both draws

    # 4) Build next-step inputs (a′, z′) and NORMALIZE them before forward
    X1 = vcat(reshape(a1, 1, :), reshape(z1, 1, :))
    X2 = vcat(reshape(a2, 1, :), reshape(z2, 1, :))

    NX1 = normalize_feature_batch(scaler, X1)
    NX2 = normalize_feature_batch(scaler, X2)

    out1, _ = Lux.apply(chain, NX1, ps, st1)
    out2, _ = Lux.apply(chain, NX2, ps, st1)

    # 5) Compute c′ using w′ = cash_on_hand(a′, z′, …)
    w1 = cash_on_hand(a1, z1, P_resid, settings.has_shocks) |> x -> T.(x)
    w2 = cash_on_hand(a2, z2, P_resid, settings.has_shocks) |> x -> T.(x)
    c1 = clamp.(vec(out1[:Φ]) .* w1, C_MIN, T(Inf))
    c2 = clamp.(vec(out2[:Φ]) .* w2, C_MIN, T(Inf))

    # 6) Euler pieces with consistent params
    β = T(P.β)
    Rg = one(T) + T(P.r)
    q1 = @. β * Rg * uprime(c1) / uprime(c0)
    q2 = @. β * Rg * uprime(c2) / uprime(c0)

    # 7) FB term and AiO product (square the product to avoid sign runaway)
    fb_term = fb(a_term, @. one(T) - h)
    kt = @. fb_term^2
    gh1 = clamp.(q1 .- h, -T(1e3), T(1e3))
    gh2 = clamp.(q2 .- h, -T(1e3), T(1e3))
    aio_pen = (gh1 .* gh2) .^ 2

    v_h = hasproperty(model_cfg, :v_h) ? T(getfield(model_cfg, :v_h)) : one(T)
    loss_vec = kt .+ v_h .* aio_pen

    # diagnostics: include max absolute Euler residual q for monitoring
    max_abs_q = maximum(abs.(vcat(q1, q2)))

    return mean(loss_vec),
    (st1, (; kt_mean = mean(kt), aio_mean = mean(aio_pen), max_abs_q = max_abs_q))
end

# --- Euler residual on dense random test set (Monte Carlo) ---
"""
    eval_euler_residuals_mc(model, ps, st, P_resid, U, scaler, settings; N=8192, rng=Random.GLOBAL_RNG)

Returns NamedTuple:
- abs_resid :: Vector{Float32} of |1 - βR E[u'(c')]/u'(c)|
- w, z, c   :: Vectors for diagnostics
- stats     :: (mean, p50, p95, max)
"""
function eval_euler_residuals_mc(
    model,
    ps,
    st,
    P_resid,
    U,
    scaler,
    settings;
    N = 8192,
    rng = Random.GLOBAL_RNG,
)
    @assert settings.has_shocks "MC eval is for stochastic spec"
    # sample inside w-window using the strict sampler (uses Main.G and Main.S)
    batch, _ = create_training_batch(
        getfield(Main, :G),
        getfield(Main, :S),
        scaler;
        mode = :rand,
        nsamples = N,
        rng = rng,
        P_resid = P_resid,
        settings = settings,
    )

    # unnormalize
    a0 = ((batch[1, :] .+ 1.0f0) ./ 2.0f0) .* scaler.a_range .+ scaler.a_min
    z0 = ((batch[2, :] .+ 1.0f0) ./ 2.0f0) .* scaler.z_range .+ scaler.z_min

    # forward c,h
    out, _ = Lux.apply(model, batch, ps, st)
    ϕ = vec(out[:Φ])
    h = vec(out[:h])

    w0 = cash_on_hand(a0, z0, P_resid, true)
    c0 = clamp.(ϕ .* w0, 1.0f-3, Inf)

    # one-shot Monte Carlo for E[u'(c')]
    # prefer global P (script-level) for shock params
    P = getfield(Main, :P)
    ρ = Float32(P.ρ)
    σϵ = Float32(P.σ_shocks)
    β = Float32(P.β)
    Rg = 1.0f0 + Float32(P.r)

    ε = randn(rng, Float32, N)
    z1 = @. ρ * z0 + σϵ * ε
    a1 = @. w0 - c0
    w1 = cash_on_hand(a1, z1, P_resid, true)

    X1 = vcat(reshape(a1, 1, :), reshape(z1, 1, :))
    NX1 = normalize_feature_batch(scaler, X1)
    out1, _ = Lux.apply(model, NX1, ps, st)
    c1 = clamp.(vec(out1[:Φ]) .* w1, 1.0f-3, Inf)

    uprime = U.u_prime
    ratio = @. β * Rg * uprime(c1) / uprime(c0)
    resid = abs.(1.0f0 .- ratio)  # |1 - βR E[u'(c')]/u'(c)| with one draw

    # compute p50 and p95 via sorted indexing (no extra dependency)
    sr = sort(vec(resid))
    n = length(sr)
    p50 = sr[clamp(Int(round(0.5 * n)), 1, n)]
    p95 = sr[clamp(Int(ceil(0.95 * n)), 1, n)]

    stats = (mean = mean(resid), p50 = p50, p95 = p95, max = maximum(resid))
    return (
        abs_resid = Float32.(resid),
        w = Float32.(w0),
        z = Float32.(z0),
        c = Float32.(c0),
        stats = stats,
        h = Float32.(h),
    )
end

# 10-node Gauss–Hermite nodes/weights
const GH10_X =
    Float32.([
        -3.436159,
        -2.532736,
        -1.756684,
        -1.036611,
        -0.342901,
        0.342901,
        1.036611,
        1.756684,
        2.532736,
        3.436159,
    ])
const GH10_W =
    Float32.([
        7.640433e-6,
        0.001343645,
        0.033874394,
        0.24013861,
        0.61086263,
        0.61086263,
        0.24013861,
        0.033874394,
        0.001343645,
        7.640433e-6,
    ])

"""
    eval_euler_residuals_gh(model, ps, st, P_resid, U, scaler, settings; N=4096)

Computes |1 - βR E[u'(c')]/u'(c)| with 10-node Gauss–Hermite on ε.
"""
function eval_euler_residuals_gh(
    model,
    ps,
    st,
    P_resid,
    U,
    scaler,
    settings;
    N = 4096,
    rng = Random.GLOBAL_RNG,
)
    @assert settings.has_shocks
    batch, _ = create_training_batch(
        getfield(Main, :G),
        getfield(Main, :S),
        scaler;
        mode = :rand,
        nsamples = N,
        rng = rng,
        P_resid = P_resid,
        settings = settings,
    )
    a0 = ((batch[1, :] .+ 1.0f0) ./ 2.0f0) .* scaler.a_range .+ scaler.a_min
    z0 = ((batch[2, :] .+ 1.0f0) ./ 2.0f0) .* scaler.z_range .+ scaler.z_min
    out, _ = Lux.apply(model, batch, ps, st)
    ϕ = vec(out[:Φ])
    w0 = cash_on_hand(a0, z0, P_resid, true)
    c0 = clamp.(ϕ .* w0, 1.0f-3, Inf)

    # prefer the full P for shock params (rho, sigma) and scalars
    P = getfield(Main, :P)
    ρ = Float32(P.ρ)
    σϵ = Float32(P.σ_shocks)
    β = Float32(P.β)
    Rg = 1.0f0 + Float32(P.r)
    uprime = U.u_prime

    # quadrature
    EUprime = zeros(Float32, length(a0))
    @inbounds for k in eachindex(GH10_X)
        εk = GH10_X[k]
        wk = GH10_W[k] / sqrt(pi)  # standard normal expectation
        z1 = @. ρ * z0 + σϵ * εk
        a1 = @. w0 - c0
        w1 = cash_on_hand(a1, z1, P_resid, true)
        X1 = vcat(reshape(a1, 1, :), reshape(z1, 1, :))
        NX1 = normalize_feature_batch(scaler, X1)
        out1, _ = Lux.apply(model, NX1, ps, st)
        c1 = clamp.(vec(out1[:Φ]) .* w1, 1.0f-3, Inf)
        EUprime .+= wk .* uprime(c1)
    end

    ratio = @. β * Rg * EUprime / uprime(c0)
    resid = abs.(1.0f0 .- ratio)
    stats = (
        mean = mean(resid),
        p50 = quantile(resid, 0.5),
        p95 = quantile(resid, 0.95),
        max = maximum(resid),
    )
    return (
        abs_resid = Float32.(resid),
        w = Float32.(w0),
        z = Float32.(z0),
        c = Float32.(c0),
        stats = stats,
    )
end

end # module
