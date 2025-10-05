struct ScalarParams
    σ::Float64
    β::Float64
    r::Float64
    y::Float64
end

struct FeatureScaler
    w_min::Float32
    w_range::Float32
    y_min::Float32
    y_range::Float32
    has_shocks::Bool
end

FeatureScaler(
    w_min::Float32,
    w_range::Float32,
    y_min::Float32,
    y_range::Float32,
    has_shocks::Bool,
) = FeatureScaler(w_min, w_range, y_min, y_range, has_shocks)

function FeatureScaler(P, G, S, settings)
    w_min = Float32(settings.w_min)
    w_max = Float32(settings.w_max)
    w_range = max(w_max - w_min, eps(Float32))
    μ = Float32(P.y)
    if settings.has_shocks && !isnothing(S)
        z = Float32.(S.zgrid)
        y_vals = exp.(μ .+ z)
        y_min, y_max = extrema(y_vals)
    else
        y_val = Float32(exp(μ))
        y_min = y_val
        y_max = y_val
    end
    y_range = max(y_max - y_min, eps(Float32))
    return FeatureScaler(w_min, w_range, y_min, y_range, settings.has_shocks)
end

function normalize_samples!(scaler::FeatureScaler, X)
    ncols = size(X, 2)
    if ncols == 2
        @. X[:, 1] = 2.0f0 * (X[:, 1] - scaler.y_min) / scaler.y_range - 1.0f0
        @. X[:, 2] = 2.0f0 * (X[:, 2] - scaler.w_min) / scaler.w_range - 1.0f0
    elseif ncols == 1
        @. X[:, 1] = 2.0f0 * (X[:, 1] - scaler.w_min) / scaler.w_range - 1.0f0
    else
        throw(
            ArgumentError("normalize_samples! expects 1 or 2 feature columns, got $ncols"),
        )
    end
    return X
end

function normalize_feature_batch!(scaler::FeatureScaler, X)
    nrows = size(X, 1)
    if nrows == 2
        @. X[1, :] = 2.0f0 * (X[1, :] - scaler.y_min) / scaler.y_range - 1.0f0
        @. X[2, :] = 2.0f0 * (X[2, :] - scaler.w_min) / scaler.w_range - 1.0f0
    elseif nrows == 1
        @. X[1, :] = 2.0f0 * (X[1, :] - scaler.w_min) / scaler.w_range - 1.0f0
    else
        throw(
            ArgumentError(
                "normalize_feature_batch! expects 1 or 2 feature rows, got $nrows",
            ),
        )
    end
    return X
end

function normalize_feature_batch(s::FeatureScaler, X::AbstractMatrix)
    nrows = size(X, 1)
    if nrows == 2
        y1 = @. 2.0f0 * (X[1, :] - s.y_min) / s.y_range - 1.0f0
        w1 = @. 2.0f0 * (X[2, :] - s.w_min) / s.w_range - 1.0f0
        return vcat(reshape(y1, 1, :), reshape(w1, 1, :))
    elseif nrows == 1
        w1 = @. 2.0f0 * (X[1, :] - s.w_min) / s.w_range - 1.0f0
        return reshape(w1, 1, :)
    else
        throw(
            ArgumentError(
                "normalize_feature_batch expects 1 or 2 feature rows, got $nrows",
            ),
        )
    end
end

get_param(container, name::Symbol, default) = begin
    value = hasproperty(container, name) ? getfield(container, name) : default
    return value === nothing ? default : value
end

function scalar_params(P)
    return ScalarParams(Float64(P.σ), Float64(P.β), Float64(P.r), Float64(P.y))
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

input_dimension(::Any) = 2
