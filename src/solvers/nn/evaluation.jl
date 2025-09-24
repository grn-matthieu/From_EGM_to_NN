"""
Evaluation utilities for the NN kernel.

The helpers in this file focus on extracting consumption predictions from the
trained network, computing Euler residuals on the model grids and aggregating
common diagnostic bundles.
"""

struct EvaluationResult
    c::Any
    a_next::Any
    resid::Any
    max_resid::Float64
end

const CONSUMPTION_FLOOR = 1.0f-8

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

function evaluate_deterministic(model, params, states, P_resid, P, G, scaler)
    X_forward, _ = det_forward_inputs(G)
    normalize_feature_batch!(scaler, X_forward)
    prediction = run_model(model, params, states, X_forward)

    if prediction isa NamedTuple
        a_grid_f32 = float32_vector(G[:a].grid)
        w = cash_on_hand(a_grid_f32, 0.0f0, P_resid, scaler.has_shocks)
        c_row = phi_to_consumption(prediction[:Φ], w)
        a_grid_f32, c_vec, c_vec_f32 = det_residual_inputs(c_row, G)
    else
        a_grid_f32, c_vec, c_vec_f32 = det_residual_inputs(prediction, G)
    end

    residuals = euler_resid_det_grid(P_resid, a_grid_f32, c_vec_f32)
    c_on_grid = convert_to_grid_eltype(G[:a].grid, c_vec)
    a_next = next_assets(P, G, c_on_grid)
    a_next = clamp_to_asset_bounds(a_next, G[:a])
    max_resid = maximum(abs.(residuals))

    return EvaluationResult(c_on_grid, a_next, residuals, max_resid)
end

function evaluate_stochastic(model, params, states, P_resid, P, G, S, scaler)
    X_eval, _ = generate_dataset(G, S; mode = :full)
    normalize_samples!(scaler, X_eval)
    batch = prepare_training_batch(X_eval)
    prediction = run_model(model, params, states, batch)

    if prediction isa NamedTuple
        a_f32 = float32_vector(G[:a].grid)
        z_f32 = float32_vector(S.zgrid)
        Na = length(a_f32)
        Nz = length(z_f32)
        A = repeat(a_f32, inner = Nz)
        Z = repeat(z_f32, outer = Na)
        w = cash_on_hand(A, Z, P_resid, scaler.has_shocks)
        c_row = phi_to_consumption(prediction[:Φ], w)
        a_grid_f32, z_grid_f32, Pz_f32, c_matrix, c_matrix_f32 =
            stoch_residual_inputs(c_row, G, S)
    else
        a_grid_f32, z_grid_f32, Pz_f32, c_matrix, c_matrix_f32 =
            stoch_residual_inputs(prediction, G, S)
    end

    residuals =
        euler_resid_stoch_grid(P_resid, a_grid_f32, z_grid_f32, Pz_f32, c_matrix_f32)
    c_on_grid = convert_to_grid_eltype(G[:a].grid, c_matrix)
    a_next = next_assets(P, G, c_on_grid)
    a_next = clamp_to_asset_bounds(a_next, G[:a])
    max_resid = maximum(abs.(residuals))

    return EvaluationResult(c_on_grid, a_next, residuals, max_resid)
end

function evaluate_solution(model, params, states, P_resid, P, G, S, scaler)
    return isnothing(S) ?
           evaluate_deterministic(model, params, states, P_resid, P, G, scaler) :
           evaluate_stochastic(model, params, states, P_resid, P, G, S, scaler)
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
)
    @assert settings.has_shocks "MC eval is for stochastic spec"
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
    w0 = cash_on_hand(a0, z0, P_resid, true)
    c0 = vec(phi_to_consumption(out[:Φ], w0; min_c = 1.0f-3))
    h = vec(ensure_row(out[:h]))

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
    c0 = vec(
        phi_to_consumption(out[:Φ], cash_on_hand(a0, z0, P_resid, true); min_c = 1.0f-3),
    )

    P = getfield(Main, :P)
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
