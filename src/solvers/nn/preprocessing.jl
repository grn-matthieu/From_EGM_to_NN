struct ScalarParams
    σ::Float64
    β::Float64
    r::Float64
    y::Float64
end

struct FeatureScaler
    a_min::Float32
    a_range::Float32
    z_min::Float32
    z_range::Float32
    has_shocks::Bool
end

function FeatureScaler(G, S)
    a = Float32.(G[:a].grid)
    a_min, a_max = extrema(a)
    a_range = max(a_max - a_min, eps(Float32))
    if isnothing(S)
        return FeatureScaler(a_min, a_range, 0.0f0, 1.0f0, false)
    else
        z = Float32.(S.zgrid)
        z_min, z_max = extrema(z)
        z_range = max(z_max - z_min, eps(Float32))
        return FeatureScaler(a_min, a_range, z_min, z_range, true)
    end
end

function normalize_samples!(scaler::FeatureScaler, X)
    @. X[:, 1] = 2.0f0 * (X[:, 1] - scaler.a_min) / scaler.a_range - 1.0f0
    if scaler.has_shocks
        @. X[:, 2] = 2.0f0 * (X[:, 2] - scaler.z_min) / scaler.z_range - 1.0f0
    end
    return X
end

function normalize_feature_batch!(scaler::FeatureScaler, X)
    @. X[1, :] = 2.0f0 * (X[1, :] - scaler.a_min) / scaler.a_range - 1.0f0
    if scaler.has_shocks
        @. X[2, :] = 2.0f0 * (X[2, :] - scaler.z_min) / scaler.z_range - 1.0f0
    end
    return X
end

function normalize_feature_batch(s::FeatureScaler, X::AbstractMatrix)
    a1 = @. 2.0f0 * (X[1, :] - s.a_min) / s.a_range - 1.0f0
    if s.has_shocks
        z1 = @. 2.0f0 * (X[2, :] - s.z_min) / s.z_range - 1.0f0
        return vcat(reshape(a1, 1, :), reshape(z1, 1, :))
    else
        return reshape(a1, 1, :)
    end
end

get_param(container, name::Symbol, default) = begin
    value = hasproperty(container, name) ? getfield(container, name) : default
    return value === nothing ? default : value
end

function scalar_params(P)
    σ = Float64(get_param(P, :σ, 1.0))
    β = Float64(get_param(P, :β, 0.95))
    r = Float64(get_param(P, :r, 0.02))
    y = Float64(get_param(P, :y, 1.0))
    return ScalarParams(σ, β, r, y)
end

function clamp_to_asset_bounds(values, grid_info)
    try
        a_min = getfield(grid_info, :min)
        a_max = getfield(grid_info, :max)
        return clamp.(values, a_min, a_max)
    catch
        return values
    end
end

input_dimension(S) = isnothing(S) ? 1 : 2
