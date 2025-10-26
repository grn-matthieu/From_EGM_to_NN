"""
Evaluation utilities for the NN kernel.

The helpers in this file focus on extracting consumption predictions from the
trained network, computing Euler residuals on the model grids and aggregating
common diagnostic bundles.
"""

using Random
using Random: randn!
using Lux: fmap
using LinearAlgebra: cholesky, mul!, Symmetric
using ..CSVarUtils: csvar_income, csvar_component_log_means
using ..GridHelpers: fit_values_on_backend!, grid_backend_available

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
@inline function maybe_fit_backend!(a_info, x_points, values)
    grid_backend_available(a_info) || return nothing
    fit_values_on_backend!(a_info, x_points, values)
    return nothing
end
@inline is_csvar_problem(P, S) =
    S !== nothing &&
    hasproperty(S, :process) &&
    S.process == :gaussian_linear &&
    hasproperty(P, :A) &&
    hasproperty(P, :Σ) &&
    hasproperty(P, :y_dim) &&
    getproperty(P, :y_dim) > 1
const CSVAR_EVAL_SAMPLES = 512

@inline function denormalize_features(scaler::FeatureScaler, batch)
    feature_dim = size(batch, 1)
    mean_vals = ((batch[1, :] .+ 1.0f0) ./ 2.0f0) .* scaler.mean_range .+ scaler.mean_min
    w_vals = ((batch[end, :] .+ 1.0f0) ./ 2.0f0) .* scaler.w_range .+ scaler.w_min
    return Float32.(mean_vals), Float32.(w_vals)
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

@inline function fallback_uprime(c, gamma)
    T = typeof(c)
    threshold = T(1e-8)
    value = c <= threshold ? threshold : c
    return value^(-gamma)
end

function get_uprime(U, P_resid)
    if U !== nothing && hasproperty(U, :u_prime)
        return U.u_prime
    else
        γ = Float64(P_resid.γ)
        return c -> max.(c, 1e-8) .^ (-γ)
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

@inline maybe_to_cpu(x, settings) = settings.use_cuda ? Array(x) : x

function evaluate_deterministic(model, params, states, P_resid, P, G, scaler)
    X_forward, w_grid = det_forward_inputs(G, P)
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
    maybe_fit_backend!(G[:a], G[:a].grid, reshape(c_on_grid, :, 1))

    return EvaluationResult(c_on_grid, a_next, residuals, max_resid)
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
    U,
    rng::AbstractRNG,
)
    X_eval, _ = generate_dataset(G, S, P; mode = :full, rng = rng)
    normalize_samples!(scaler, X_eval)
    batch = prepare_training_batch(X_eval, Val(settings.use_cuda))
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
    W = settings.use_cuda ? fmap(cu, Rg * A + Y) : Rg * A + Y

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
    maybe_fit_backend!(G[:a], G[:a].grid, c_on_grid)

    return EvaluationResult(c_on_grid, a_next, residuals, max_resid)
end

function evaluate_csvar(
    model,
    params,
    states,
    P_resid,
    P,
    G,
    S,
    scaler,
    settings,
    U,
    rng::AbstractRNG,
)
    y_state = Float32.(csvar_component_log_means(P))
    y_dim = length(y_state)
    y_dim > 0 || error("CSVAR evaluation requires a positive state dimension")

    a_grid_f32 = float32_vector(G[:a].grid)
    Na = length(a_grid_f32)
    Rg = 1.0f0 + Float32(P_resid.r)
    income_curr = Float32(csvar_income(y_state))

    feature_dim = y_dim + 2
    X = Matrix{Float32}(undef, feature_dim, Na)
    X[1, :] .= income_curr
    for j = 1:y_dim
        X[1+j, :] .= y_state[j]
    end
    w_grid = @. Rg * a_grid_f32 + income_curr
    X[end, :] .= w_grid
    normalize_feature_batch!(scaler, X)
    X_dev = maybe_to_device(X, settings)

    prediction = run_model(model, params, states, X_dev)
    if prediction isa NamedTuple
        w_dev = maybe_to_device(w_grid, settings)
        c_pred = phi_to_consumption(prediction[:Φ], w_dev; min_c = EVAL_MIN_CONSUMPTION)
    else
        c_pred = prediction
    end
    _, c_vec, _ = det_residual_inputs(c_pred, G)
    c_on_grid = convert_to_grid_eltype(G[:a].grid, c_vec)
    a_next = next_assets_from_cash(w_grid, Float32.(c_on_grid))
    a_next = clamp_to_asset_bounds(a_next, G[:a])
    maybe_fit_backend!(G[:a], G[:a].grid, reshape(c_on_grid, :, 1))

    uprime = get_uprime(U, P_resid)
    β = Float32(P_resid.β)

    Σ = Matrix{Float64}(P.Σ)
    chol = cholesky(Symmetric(Σ), check = false)
    L = Matrix{Float32}(chol.L)
    A = Matrix{Float32}(P.A)
    μ_vec = A * y_state

    innovations = Matrix{Float32}(undef, y_dim, CSVAR_EVAL_SAMPLES)
    randn!(rng, innovations)
    draws = Matrix{Float32}(undef, y_dim, CSVAR_EVAL_SAMPLES)
    mul!(draws, L, innovations)
    @. draws += μ_vec
    income_draws = Float32.(csvar_income(draws))

    resid = Vector{Float32}(undef, Na)
    uprime_c0 = uprime(Float32.(c_on_grid))
    Rg64 = Float32(Rg)
    for i = 1:Na
        a_next_i = Float32(a_next[i])
        w_future = @. Rg64 * a_next_i + income_draws
        X_future = build_feature_batch_from_states(scaler, draws, w_future)
        X_future_dev = maybe_to_device(X_future, settings)
        pred_next = run_model(model, params, states, X_future_dev)
        if pred_next isa NamedTuple
            w_future_dev = maybe_to_device(w_future, settings)
            c1_raw = phi_to_consumption(
                pred_next[:Φ],
                w_future_dev;
                min_c = EVAL_MIN_CONSUMPTION,
            )
        else
            c1_raw = pred_next
        end
        c1_vec = vec(permutedims(ensure_row(c1_raw)))
        c1_cpu = maybe_to_cpu(c1_vec, settings)
        mean_uprime = mean(uprime(c1_cpu))
        denom =
            uprime_c0[i] <= 0 ? uprime(Float32(max(c_on_grid[i], EVAL_MIN_CONSUMPTION))) :
            uprime_c0[i]
        val = abs(1 - β * Rg64 * mean_uprime / denom)
        resid[i] = Float32(val)
    end

    max_resid = sqrt(mean(Float64.(resid) .^ 2))
    a_next_out = convert.(eltype(G[:a].grid), a_next)

    return EvaluationResult(c_on_grid, a_next_out, resid, max_resid)
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
    rng = nothing,
)
    local_settings =
        settings === nothing ?
        solver_settings(
            nothing,
            P,
            G,
            S;
            has_shocks = scaler.has_shocks,
            objective_default = is_csvar_problem(P, S) ? :euler_residual :
                                (scaler.has_shocks ? :euler_fb_aio : :euler_residual),
        ) : settings
    local_rng = rng === nothing ? Random.default_rng() : rng
    if is_csvar_problem(P, S)
        return evaluate_csvar(
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
            local_rng,
        )
    elseif scaler.has_shocks
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
            local_rng,
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
    if P !== nothing && is_csvar_problem(P, S)
        return eval_euler_residuals_mc_csvar(
            model,
            ps,
            st,
            P_resid,
            U,
            scaler,
            settings;
            N = N,
            rng = rng,
            G = G,
            S = S,
            P = P,
        )
    end
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
        P = P,
    )

    mean_norm = batch[1, :]
    w_norm = batch[end, :]
    y0 = ((mean_norm .+ 1.0f0) ./ 2.0f0) .* scaler.mean_range .+ scaler.mean_min
    w0 = ((w_norm .+ 1.0f0) ./ 2.0f0) .* scaler.w_range .+ scaler.w_min

    # Ensure y0/w0 live on the same device as model outputs when using CUDA
    if settings.use_cuda
        y0 = cu(y0)
        w0 = cu(w0)
    end

    out, _ = Lux.apply(model, batch, ps, st)
    c0 = vec(phi_to_consumption(out[:Φ], w0; min_c = 1.0f-3))
    h = vec(ensure_row(out[:h]))

    μ = Float32(P_resid.y)
    z0 = log.(y0) .- μ
    ρ = Float32(P.ρ_shock)
    σϵ =
        settings.sigma_shocks === nothing ? Float32(P.σ_shock) :
        Float32(settings.sigma_shocks)
    β = Float32(P.β)
    Rg = 1.0f0 + Float32(P.r)

    ε = randn(rng, Float32, N)
    if settings.use_cuda
        ε = cu(ε)
    end
    z1 = @. ρ * z0 + σϵ * ε
    y1 = exp.(μ .+ z1)
    a1 = @. w0 - c0
    w1 = @. Rg * a1 + y1

    feature_dim = size(batch, 1)
    component_levels =
        hasproperty(P, :y) && P.y isa AbstractVector ? Float32.(exp.(collect(P.y))) :
        Float32[]

    # Build feature matrix on CPU (so assignment works regardless of whether
    # y1/w1 are on the GPU). After normalization, move the batch to device if
    # settings.use_cuda is set.
    y1_cpu = maybe_to_cpu(y1, settings)
    w1_cpu = maybe_to_cpu(w1, settings)
    X1 = Matrix{Float32}(undef, feature_dim, length(y1_cpu))
    X1[1, :] .= y1_cpu
    if feature_dim > 2
        for j = 1:(feature_dim-2)
            level = j <= length(component_levels) ? component_levels[j] : Float32(P_resid.y)
            X1[1+j, :] .= level
        end
    end
    X1[end, :] .= w1_cpu
    normalize_feature_batch!(scaler, X1)

    X1_dev = maybe_to_device(X1, settings)
    out1, _ = Lux.apply(model, X1_dev, ps, st)

    # Ensure the cash-on-hand passed to phi_to_consumption lives on the same
    # device as the model output.
    w1_dev = maybe_to_device(w1_cpu, settings)
    c1 = vec(phi_to_consumption(out1[:Φ], w1_dev; min_c = 1.0f-3))

    uprime = U.u_prime
    ratio = @. β * Rg * uprime(c1) / uprime(c0)
    resid = abs.(1.0f0 .- ratio)

    # Move diagnostics to CPU for aggregation (sorting, quantiles) and return
    if settings.use_cuda
        resid_cpu = Array(resid)
        w_cpu = Array(w0)
        y_cpu = Array(y0)
        c_cpu = Array(c0)
        h_cpu = Array(h)
    else
        resid_cpu = resid
        w_cpu = w0
        y_cpu = y0
        c_cpu = c0
        h_cpu = h
    end

    sr = sort(vec(resid_cpu))
    n = length(sr)
    p50 = sr[clamp(Int(round(0.5 * n)), 1, n)]
    p95 = sr[clamp(Int(ceil(0.95 * n)), 1, n)]

    stats = (mean = mean(resid_cpu), p50 = p50, p95 = p95, max = maximum(resid_cpu))
    return (
        abs_resid = Float32.(resid_cpu),
        w = Float32.(w_cpu),
        y = Float32.(y_cpu),
        c = Float32.(c_cpu),
        stats = stats,
        h = Float32.(h_cpu),
    )
end

function eval_euler_residuals_mc_csvar(
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
    @assert !(G === nothing) "eval_euler_residuals_mc_csvar requires G to be provided"
    @assert !(P === nothing) "eval_euler_residuals_mc_csvar requires P to be provided"
    @assert !(S === nothing) "eval_euler_residuals_mc_csvar requires S to be provided"

    batch, _ = create_training_batch(
        G,
        S,
        scaler;
        mode = :rand,
        nsamples = N,
        rng = rng,
        P_resid = P_resid,
        settings = settings,
        P = P,
    )

    batch_cpu = maybe_to_cpu(batch, settings)
    mean_vals, comps, w0_cpu = denormalize_feature_batch(scaler, batch_cpu)
    y_dim = size(comps, 1)
    y_dim > 0 || error("CSVAR diagnostics require vector-valued income state")

    w0_dev = maybe_to_device(w0_cpu, settings)
    out, _ = Lux.apply(model, batch, ps, st)
    c0 = vec(phi_to_consumption(out[:Φ], w0_dev; min_c = 1.0f-3))
    h = vec(ensure_row(out[:h]))

    c0_cpu = maybe_to_cpu(c0, settings)
    h_cpu = maybe_to_cpu(h, settings)

    β = Float32(P.β)
    Rg = 1.0f0 + Float32(P.r)
    uprime = U.u_prime
    a0 = w0_cpu .- c0_cpu

    Σ = Matrix{Float64}(P.Σ)
    chol = cholesky(Symmetric(Σ), check = false)
    L = Matrix{Float32}(chol.L)
    A = Matrix{Float32}(P.A)

    ε = randn(rng, Float32, y_dim, N)
    y_next = A * comps .+ L * ε
    income_next = csvar_income(y_next)
    w1_cpu = @. Rg * a0 + income_next
    X1 = build_feature_batch_from_states(scaler, y_next, Float32.(w1_cpu))
    X1_dev = maybe_to_device(X1, settings)
    out1, _ = Lux.apply(model, X1_dev, ps, st)
    w1_dev = maybe_to_device(Float32.(w1_cpu), settings)
    c1 = vec(phi_to_consumption(out1[:Φ], w1_dev; min_c = 1.0f-3))
    c1_cpu = maybe_to_cpu(c1, settings)

    ratio = @. β * Rg * uprime(c1_cpu) / uprime(c0_cpu)
    resid = abs.(1.0f0 .- ratio)

    resid_cpu = Float32.(maybe_to_cpu(resid, settings))
    w_cpu = Float32.(w0_cpu)
    y_cpu = Float32.(mean_vals)
    c_cpu = Float32.(c0_cpu)
    h_cpu_f32 = Float32.(h_cpu)

    sr = sort(vec(resid_cpu))
    n = length(sr)
    p50 = sr[clamp(Int(round(0.5 * n)), 1, n)]
    p95 = sr[clamp(Int(ceil(0.95 * n)), 1, n)]

    stats = (mean = mean(resid_cpu), p50 = p50, p95 = p95, max = maximum(resid_cpu))
    return (
        abs_resid = resid_cpu,
        w = w_cpu,
        y = y_cpu,
        c = c_cpu,
        stats = stats,
        h = h_cpu_f32,
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
    if P !== nothing && is_csvar_problem(P, S)
        mc_res = eval_euler_residuals_mc_csvar(
            model,
            ps,
            st,
            P_resid,
            U,
            scaler,
            settings;
            N = N,
            rng = rng,
            G = G,
            S = S,
            P = P,
        )
        return (;
            abs_resid = mc_res.abs_resid,
            w = mc_res.w,
            y = mc_res.y,
            c = mc_res.c,
            stats = mc_res.stats,
        )
    end
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
        P = P,
    )
    mean_norm = batch[1, :]
    w_norm = batch[end, :]
    y0 = ((mean_norm .+ 1.0f0) ./ 2.0f0) .* scaler.mean_range .+ scaler.mean_min
    w0 = ((w_norm .+ 1.0f0) ./ 2.0f0) .* scaler.w_range .+ scaler.w_min
    if settings.use_cuda
        y0 = cu(y0)
        w0 = cu(w0)
    end
    out, _ = Lux.apply(model, batch, ps, st)
    c0 = vec(phi_to_consumption(out[:Φ], w0; min_c = 1.0f-3))

    μ = Float32(P_resid.y)
    z0 = log.(y0) .- μ
    ρ = Float32(P.ρ_shock)
    σϵ =
        settings.sigma_shocks === nothing ? Float32(P.σ_shock) :
        Float32(settings.sigma_shocks)
    β = Float32(P.β)
    Rg = 1.0f0 + Float32(P.r)
    uprime = U.u_prime

    EUprime =
        settings.use_cuda ? cu(zeros(Float32, length(w0))) : zeros(Float32, length(w0))
    @inbounds for k in eachindex(GH10_X)
        εk = GH10_X[k]
        wk = GH10_W[k] / sqrt(pi)
        z1 = @. ρ * z0 + σϵ * εk
        y1 = exp.(μ .+ z1)
        a1 = @. w0 - c0
        w1 = @. Rg * a1 + y1
        feature_dim = size(batch, 1)
        component_levels =
            hasproperty(P, :y) && P.y isa AbstractVector ? Float32.(collect(P.y)) :
            Float32[]
        # Build feature matrix on CPU (so assignment works regardless of whether
        # y1/w1 are on the GPU). After normalization, move the batch to device if
        # settings.use_cuda is set.
        y1_cpu = maybe_to_cpu(y1, settings)
        w1_cpu = maybe_to_cpu(w1, settings)
        X1 = Matrix{Float32}(undef, feature_dim, length(y1_cpu))
        X1[1, :] .= y1_cpu
        if feature_dim > 2
            for j = 1:(feature_dim-2)
                level =
                    j <= length(component_levels) ? component_levels[j] : Float32(P_resid.y)
                X1[1+j, :] .= level
            end
        end
        X1[end, :] .= w1_cpu
        normalize_feature_batch!(scaler, X1)

        X1_dev = maybe_to_device(X1, settings)
        out1, _ = Lux.apply(model, X1_dev, ps, st)

        # Ensure the cash-on-hand passed to phi_to_consumption lives on the same
        # device as the model output.
        w1_dev = maybe_to_device(w1_cpu, settings)
        c1 = vec(phi_to_consumption(out1[:Φ], w1_dev; min_c = 1.0f-3))
        normalize_feature_batch!(scaler, X1)
        out1, _ = Lux.apply(model, X1, ps, st)
        c1 = vec(phi_to_consumption(out1[:Φ], w1; min_c = 1.0f-3))
        EUprime .+= wk .* uprime(c1)
    end
    ratio = @. β * Rg * EUprime / uprime(c0)
    resid = abs.(1.0f0 .- ratio)

    # Move back to CPU for aggregation and returning results
    if settings.use_cuda
        resid_cpu = Array(resid)
        w_cpu = Array(w0)
        y_cpu = Array(y0)
        c_cpu = Array(c0)
    else
        resid_cpu = resid
        w_cpu = w0
        y_cpu = y0
        c_cpu = c0
    end

    stats = (
        mean = mean(resid_cpu),
        p50 = quantile(resid_cpu, 0.5),
        p95 = quantile(resid_cpu, 0.95),
        max = maximum(resid_cpu),
    )
    return (
        abs_resid = Float32.(resid_cpu),
        w = Float32.(w_cpu),
        y = Float32.(y_cpu),
        c = Float32.(c_cpu),
        stats = stats,
    )
end
