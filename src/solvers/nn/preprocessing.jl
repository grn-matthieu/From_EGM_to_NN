struct ScalarParams
    σ::Float64
    β::Float64
    r::Float64
    y::Float64
end

struct FeatureScaler
    w_min::Float32
    w_range::Float32
    z_min::Float32
    z_range::Float32
    has_shocks::Bool
end

function FeatureScaler(G, S)
    w = Float32.(G[:w].grid)
    w_min, w_max = extrema(w)
    w_range = max(w_max - w_min, eps(Float32))
    if isnothing(S)
        return FeatureScaler(w_min, w_range, 0.0f0, 1.0f0, false)
    else
        z = Float32.(S.zgrid)
        z_min, z_max = extrema(z)
        z_range = max(z_max - z_min, eps(Float32))
        return FeatureScaler(w_min, w_range, z_min, z_range, true)
    end
end

function normalize_samples!(scaler::FeatureScaler, X)
    @. X[:, 1] = 2.0f0 * (X[:, 1] - scaler.w_min) / scaler.w_range - 1.0f0
    if scaler.has_shocks
        @. X[:, 2] = 2.0f0 * (X[:, 2] - scaler.z_min) / scaler.z_range - 1.0f0
    end
    return X
end

function normalize_feature_batch!(scaler::FeatureScaler, X)
    @. X[1, :] = 2.0f0 * (X[1, :] - scaler.w_min) / scaler.w_range - 1.0f0
    if scaler.has_shocks
        @. X[2, :] = 2.0f0 * (X[2, :] - scaler.z_min) / scaler.z_range - 1.0f0
    end
    return X
end

function normalize_feature_batch(s::FeatureScaler, X::AbstractMatrix)
    w1 = @. 2.0f0 * (X[1, :] - s.w_min) / s.w_range - 1.0f0
    if s.has_shocks
        z1 = @. 2.0f0 * (X[2, :] - s.z_min) / s.z_range - 1.0f0
        return vcat(reshape(w1, 1, :), reshape(z1, 1, :))
    else
        return reshape(w1, 1, :)
    end
end

get_param(container, name::Symbol, default) = begin
    value = hasproperty(container, name) ? getfield(container, name) : default
    return value === nothing ? default : value
end

function scalar_params(P)
    # Expect exact parameter names to be present in the config. Validation should
    # be performed by the config/validation module; here we access fields directly.
    return ScalarParams(Float64(P.σ), Float64(P.β), Float64(P.r), Float64(P.y))
end

function clamp_to_asset_bounds(values, grid_info)
    try
        w_min = getfield(grid_info, :min)
        w_max = getfield(grid_info, :max)
        return clamp.(values, w_min, w_max)
    catch
        return values
    end
end

input_dimension(S) = isnothing(S) ? 1 : 2
