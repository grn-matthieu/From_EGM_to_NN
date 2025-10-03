"""
Evaluation utilities for the NN kernel.

The helpers in this file focus on extracting consumption predictions from the
trained network, computing Euler residuals on the model grids and aggregating
common diagnostic bundles.
"""

using Random

struct EvaluationResult
    c::Any
    a_next::Any
    resid::Any
    max_resid::Float64
end

const CONSUMPTION_FLOOR = 1.0f-8

const DEFAULT_EVAL_SAMPLES = 8192
const EVAL_MIN_CONSUMPTION = 1.0f-3

@inline function denormalize_features(scaler::FeatureScaler, batch)
    a = ((batch[1, :] .+ 1.0f0) ./ 2.0f0) .* scaler.a_range .+ scaler.a_min
    if scaler.has_shocks && size(batch, 1) >= 2
        z = ((batch[2, :] .+ 1.0f0) ./ 2.0f0) .* scaler.z_range .+ scaler.z_min
        return Float32.(a), Float32.(z)
    else
        return Float32.(a), nothing
    end
end

function extract_consumption(prediction, w)
    if prediction isa NamedTuple
        return vec(phi_to_consumption(prediction[:Φ], w; min_c = EVAL_MIN_CONSUMPTION))
    else
        values = ensure_row(prediction)
        consumption = vec(values)
        T = eltype(consumption)
        floor = T(EVAL_MIN_CONSUMPTION)
        return clamp.(consumption, floor, T(Inf))
    end
end

@inline function fallback_uprime(c, sigma)
    T = typeof(c)
    threshold = T(1e-8)
    value = c <= threshold ? threshold : c
    return value^(-sigma)
end

function get_uprime(U, P_resid)
    if U !== nothing && hasproperty(U, :u_prime)
        return U.u_prime
    else
        sigma = Float64(P_resid.σ)
        return c -> fallback_uprime(c, sigma)
    end
end

"""Ensure a Lux output is reshaped as a single-row matrix."""
function ensure_row(values)
    if ndims(values) == 1
        return reshape(values, 1, :)
    elseif ndims(values) == 2 && size(values, 1) == 1
        return values
    elseif ndims(values) == 2 && size(values, 2) == 1
        return permutedims(values)
    else
        return reshape(vec(values), 1, :)
    end
end

"""
    phi_to_consumption(Φ, w; min_c = CONSUMPTION_FLOOR)

Map the network's Φ output to consumption by multiplying it with cash-on-hand
`w` and clamping it away from zero. Works transparently with vectors or
matrices.
"""
function phi_to_consumption(Φ, w; min_c = CONSUMPTION_FLOOR)
    Φ_row = ensure_row(Φ)
    w_row = reshape(w, 1, :)
    consumption = Φ_row .* w_row
    T = eltype(consumption)
    return clamp.(consumption, T(min_c), T(Inf))
end

"""Compute next-period assets given consumption on the asset grid."""
function next_assets(P, G, consumption)
    T = eltype(consumption)
    R = T(1) + T(P.r)
    y = T(P.y)
    a_grid = convert.(T, G[:a].grid)
    return @. R * a_grid + y - consumption
end

function evaluate_deterministic(
    model,
    params,
    states,
    P_resid,
    P,
    G,
    scaler,
    settings;
    nsamples::Int = DEFAULT_EVAL_SAMPLES,
    rng = Random.default_rng(),
)
    batch, _ = create_training_batch(
        G,
        nothing,
        scaler;
        mode = :rand,
        nsamples = nsamples,
        rng = rng,
        P_resid = P_resid,
        settings = settings,
    )
    a0, _ = denormalize_features(scaler, batch)
    z0 = fill(0.0f0, length(a0))

    prediction = run_model(model, params, states, batch)
    w0 = cash_on_hand(a0, z0, P_resid, false)
    c0 = extract_consumption(prediction, w0)

    a1 = w0 .- c0
    X1 = reshape(a1, 1, :)
    NX1 = normalize_feature_batch(scaler, X1)
    prediction1 = run_model(model, params, states, NX1)
    w1 = cash_on_hand(a1, z0, P_resid, false)
    c1 = extract_consumption(prediction1, w1)

    residuals = euler_resid_det(P_resid, c0, c1)

    c_out = Float32.(c0)
    a_out = Float32.(a1)
    resid_out = Float32.(residuals)
    max_resid = maximum(abs.(residuals))

    return EvaluationResult(c_out, a_out, resid_out, max_resid)
end

function evaluate_stochastic(
    model,
    params,
    states,
    P_resid,
    P,
    G,
    S,
    scaler,
    settings,
    U;
    nsamples::Int = DEFAULT_EVAL_SAMPLES,
    rng = Random.default_rng(),
)
    batch, _ = create_training_batch(
        G,
        S,
        scaler;
        mode = :rand,
        nsamples = nsamples,
        rng = rng,
        P_resid = P_resid,
        settings = settings,
    )
    a0, z0 = denormalize_features(scaler, batch)

    prediction = run_model(model, params, states, batch)
    w0 = cash_on_hand(a0, z0, P_resid, true)
    c0 = extract_consumption(prediction, w0)

    sigma_eps =
        settings.sigma_shocks === nothing ? Float32(P.σ_shocks) :
        Float32(settings.sigma_shocks)
    rho = Float32(P.ρ)
    eps = randn(rng, Float32, length(a0))
    z1 = @. rho * z0 + sigma_eps * eps

    a1 = w0 .- c0
    X1 = vcat(reshape(a1, 1, :), reshape(z1, 1, :))
    NX1 = normalize_feature_batch(scaler, X1)
    prediction1 = run_model(model, params, states, NX1)
    w1 = cash_on_hand(a1, z1, P_resid, true)
    c1 = extract_consumption(prediction1, w1)

    uprime = get_uprime(U, P_resid)
    uprime_c0 = uprime.(c0)
    uprime_c1 = uprime.(c1)

    beta = Float64(P_resid.β)
    Rg = Float64(1.0 + P_resid.r)
    ratio = beta * Rg .* (uprime_c1 ./ uprime_c0)
    residuals = abs.(1 .- ratio)

    c_out = Float32.(c0)
    a_out = Float32.(a1)
    resid_out = Float32.(residuals)
    max_resid = maximum(residuals)

    return EvaluationResult(c_out, a_out, resid_out, Float64(max_resid))
end

function evaluate_solution(
    model,
    params,
    states,
    P_resid,
    P,
    G,
    S,
    scaler;
    settings::Union{NNSolverSettings,Nothing} = nothing,
    U = nothing,
    nsamples::Int = DEFAULT_EVAL_SAMPLES,
    rng = Random.default_rng(),
)
    local_settings =
        settings === nothing ? solver_settings(nothing; has_shocks = scaler.has_shocks) :
        settings
    if scaler.has_shocks
        return evaluate_stochastic(
            model,
            params,
            states,
            P_resid,
            P,
            G,
            S,
            scaler,
            local_settings,
            U;
            nsamples = nsamples,
            rng = rng,
        )
    else
        return evaluate_deterministic(
            model,
            params,
            states,
            P_resid,
            P,
            G,
            scaler,
            local_settings;
            nsamples = nsamples,
            rng = rng,
        )
    end
end

"""Return Monte Carlo Euler residual diagnostics for stochastic problems."""
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
    G = nothing,
    S = nothing,
    P = nothing,
)
    @assert settings.has_shocks "MC eval is for stochastic spec"
    @assert !(G === nothing) "eval_euler_residuals_mc requires G to be provided"
    @assert !(S === nothing) "eval_euler_residuals_mc requires S to be provided"
    @assert !(P === nothing) "eval_euler_residuals_mc requires P to be provided"

    batch, _ = create_training_batch(
        G,
        S,
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
    w0 = cash_on_hand(a0, z0, P_resid, true)
    c0 = vec(phi_to_consumption(out[:Φ], w0; min_c = 1.0f-3))
    h = vec(ensure_row(out[:h]))

    # use passed P when available (already resolved above)
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
    c1 = vec(phi_to_consumption(out1[:Φ], w1; min_c = 1.0f-3))

    uprime = U.u_prime
    ratio = @. β * Rg * uprime(c1) / uprime(c0)
    resid = abs.(1.0f0 .- ratio)

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

"""Gauss–Hermite Euler residual diagnostics for stochastic problems."""
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
    G = nothing,
    S = nothing,
    P = nothing,
)
    @assert settings.has_shocks
    @assert !(G === nothing) "eval_euler_residuals_gh requires G to be provided"
    @assert !(S === nothing) "eval_euler_residuals_gh requires S to be provided"
    @assert !(P === nothing) "eval_euler_residuals_gh requires P to be provided"

    batch, _ = create_training_batch(
        G,
        S,
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
    c0 = vec(
        phi_to_consumption(out[:Φ], cash_on_hand(a0, z0, P_resid, true); min_c = 1.0f-3),
    )

    # use passed P when available (already resolved above)
    ρ = Float32(P.ρ)
    σϵ = Float32(P.σ_shocks)
    β = Float32(P.β)
    Rg = 1.0f0 + Float32(P.r)
    uprime = U.u_prime

    w0 = cash_on_hand(a0, z0, P_resid, true)
    EUprime = zeros(Float32, length(a0))
    @inbounds for k in eachindex(GH10_X)
        εk = GH10_X[k]
        wk = GH10_W[k] / sqrt(pi)
        z1 = @. ρ * z0 + σϵ * εk
        a1 = @. w0 - c0
        w1 = cash_on_hand(a1, z1, P_resid, true)
        X1 = vcat(reshape(a1, 1, :), reshape(z1, 1, :))
        NX1 = normalize_feature_batch(scaler, X1)
        out1, _ = Lux.apply(model, NX1, ps, st)
        c1 = vec(phi_to_consumption(out1[:Φ], w1; min_c = 1.0f-3))
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
