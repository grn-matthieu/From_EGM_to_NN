"""
Evaluation utilities for the NN kernel.

The helpers in this file focus on extracting consumption predictions from the
trained network, computing Euler residuals on the model grids and aggregating
common diagnostic bundles.
"""

using Random

# Local sigmoid function to avoid NNlib dependency
@inline sigmoid(x) = 1 / (1 + exp(-x))

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
    if size(batch, 1) == 2
        y = ((batch[1, :] .+ 1.0f0) ./ 2.0f0) .* scaler.y_range .+ scaler.y_min
        w = ((batch[2, :] .+ 1.0f0) ./ 2.0f0) .* scaler.w_range .+ scaler.w_min
        return Float32.(y), Float32.(w)
    elseif size(batch, 1) == 1
        w = ((batch[1, :] .+ 1.0f0) ./ 2.0f0) .* scaler.w_range .+ scaler.w_min
        y = fill(scaler.y_min, length(w))
        return Float32.(y), Float32.(w)
    else
        throw(ArgumentError("Expected 1 or 2 feature rows, got $(size(batch, 1))"))
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

"""Compute next-period assets from current cash-on-hand and consumption."""
function next_assets_from_cash(w, consumption)
    T = eltype(consumption)
    return convert.(T, w) .- consumption
end

function evaluate_deterministic(model, params, states, P_resid, P, G, scaler)
    X_forward, w_grid = det_forward_inputs(G, P_resid)
    normalize_feature_batch!(scaler, X_forward)
    prediction = run_model(model, params, states, X_forward)

    if prediction isa NamedTuple
        c_row = phi_to_consumption(prediction[:Φ], w_grid)
        a_grid_f32, c_vec, c_vec_f32 = det_residual_inputs(c_row, G)
    else
        a_grid_f32, c_vec, c_vec_f32 = det_residual_inputs(prediction, G)
    end

    residuals = euler_resid_det_grid(P_resid, a_grid_f32, c_vec_f32)
    c_on_grid = convert_to_grid_eltype(G[:a].grid, c_vec)
    a_next = next_assets_from_cash(w_grid, c_on_grid)
    a_next = clamp_to_asset_bounds(a_next, G[:a])
    max_resid = maximum(abs.(residuals))

    return EvaluationResult(c_on_grid, a_next, residuals, max_resid)
end

function evaluate_stochastic(model, params, states, P_resid, P, G, S, scaler, settings, U)
    X_eval, _ = generate_dataset(G, S, P_resid; mode = :full)
    normalize_samples!(scaler, X_eval)
    batch = prepare_training_batch(X_eval)
    prediction = run_model(model, params, states, batch)

    a_f32 = float32_vector(G[:a].grid)
    z_f32 = float32_vector(S.zgrid)
    Na = length(a_f32)
    Nz = length(z_f32)
    A = repeat(a_f32, inner = Nz)
    Z = repeat(z_f32, outer = Na)
    μ = Float32(P_resid.y)
    Rg = 1.0f0 + Float32(P_resid.r)
    Y = exp.(μ .+ Z)
    W = @. Rg * A + Y

    if prediction isa NamedTuple
        c_row = phi_to_consumption(prediction[:Φ], W)
        a_grid_f32, z_grid_f32, Pz_f32, c_matrix, c_matrix_f32 =
            stoch_residual_inputs(c_row, G, S)
    else
        a_grid_f32, z_grid_f32, Pz_f32, c_matrix, c_matrix_f32 =
            stoch_residual_inputs(prediction, G, S)
    end

    residuals =
        euler_resid_stoch_grid(P_resid, a_grid_f32, z_grid_f32, Pz_f32, c_matrix_f32)
    c_on_grid = convert_to_grid_eltype(G[:a].grid, c_matrix)
    w_matrix = reshape(W, Na, Nz)
    a_next = next_assets_from_cash(w_matrix, c_on_grid)
    a_next = clamp_to_asset_bounds(a_next, G[:a])
    max_resid = maximum(abs.(residuals))

    return EvaluationResult(c_on_grid, a_next, residuals, max_resid)
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
            U,
        )
    else
        return evaluate_deterministic(model, params, states, P_resid, P, G, scaler)
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
    rng::AbstractRNG,
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

    y0 = ((batch[1, :] .+ 1.0f0) ./ 2.0f0) .* scaler.y_range .+ scaler.y_min
    w0 = ((batch[2, :] .+ 1.0f0) ./ 2.0f0) .* scaler.w_range .+ scaler.w_min

    out, _ = Lux.apply(model, batch, ps, st)
    c0 = vec(phi_to_consumption(out[:Φ], w0; min_c = 1.0f-3))
    h = vec(ensure_row(out[:h]))

    μ = Float32(P_resid.y)
    z0 = log.(y0) .- μ
    ρ = Float32(P.ρ)
    σϵ =
        settings.sigma_shocks === nothing ? Float32(P.σ_shocks) :
        Float32(settings.sigma_shocks)
    β = Float32(P.β)
    Rg = 1.0f0 + Float32(P.r)

    ε = randn(rng, Float32, N)
    z1 = @. ρ * z0 + σϵ * ε
    y1 = exp.(μ .+ z1)
    a1 = @. w0 - c0
    w1 = @. Rg * a1 + y1

    X1 = vcat(reshape(y1, 1, :), reshape(w1, 1, :))
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
        y = Float32.(y0),
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
    rng::AbstractRNG,
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
    y0 = ((batch[1, :] .+ 1.0f0) ./ 2.0f0) .* scaler.y_range .+ scaler.y_min
    w0 = ((batch[2, :] .+ 1.0f0) ./ 2.0f0) .* scaler.w_range .+ scaler.w_min
    out, _ = Lux.apply(model, batch, ps, st)
    c0 = vec(phi_to_consumption(out[:Φ], w0; min_c = 1.0f-3))

    μ = Float32(P_resid.y)
    z0 = log.(y0) .- μ
    ρ = Float32(P.ρ)
    σϵ =
        settings.sigma_shocks === nothing ? Float32(P.σ_shocks) :
        Float32(settings.sigma_shocks)
    β = Float32(P.β)
    Rg = 1.0f0 + Float32(P.r)
    uprime = U.u_prime

    EUprime = zeros(Float32, length(w0))
    @inbounds for k in eachindex(GH10_X)
        εk = GH10_X[k]
        wk = GH10_W[k] / sqrt(pi)
        z1 = @. ρ * z0 + σϵ * εk
        y1 = exp.(μ .+ z1)
        a1 = @. w0 - c0
        w1 = @. Rg * a1 + y1
        X1 = vcat(reshape(y1, 1, :), reshape(w1, 1, :))
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
        y = Float32.(y0),
        c = Float32.(c0),
        stats = stats,
    )
end
