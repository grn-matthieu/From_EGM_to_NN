import ChainRulesCore: @non_differentiable
using CUDA: cu, CuArray, CUDA
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

function FeatureScaler(P, G, S, settings)
    # y
    μ = Float32(P.y)
    if isnothing(S)
        y_min = Float32(exp(μ))
        y_range = 1.0f0                 # évite /0 en déterministe
    else
        zmin = Float32(minimum(S.zgrid))
        zmax = Float32(maximum(S.zgrid))
        y_min = Float32(exp(μ + zmin))
        y_max = Float32(exp(μ + zmax))
        y_range = max(y_max - y_min, 1.0f-6)
    end
    # w
    w_min = Float32(settings.w_min)
    w_max = Float32(settings.w_max)
    w_range = max(w_max - w_min, 1.0f-6)

    return FeatureScaler(w_min, w_range, y_min, y_range, !isnothing(S))
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

# explicit CPU in-place version
function normalize_feature_batch!(sc::FeatureScaler, X::AbstractMatrix{<:AbstractFloat})
    nrows = size(X, 1)
    if nrows == 2
        @. X[1, :] = 2.0f0 * (X[1, :] - sc.y_min) / sc.y_range - 1.0f0
        @. X[2, :] = 2.0f0 * (X[2, :] - sc.w_min) / sc.w_range - 1.0f0
    elseif nrows == 1
        @. X[1, :] = 2.0f0 * (X[1, :] - sc.w_min) / sc.w_range - 1.0f0
    else
        throw(
            ArgumentError(
                "normalize_feature_batch! expects 1 or 2 feature rows, got $nrows",
            ),
        )
    end
    return X
end

# GPU in-place version without views nor scalar indexing
function normalize_feature_batch!(
    sc::FeatureScaler,
    X::CUDA.CuArray{T,2},
) where {T<:AbstractFloat}
    nrows = size(X, 1)
    if nrows == 2
        mins = reshape(cu(T.([sc.y_min, sc.w_min])), 2, 1)
        ranges = reshape(cu(T.([sc.y_range, sc.w_range])), 2, 1)
        @. X = 2.0f0 * (X - mins) / ranges - 1.0f0
    elseif nrows == 1
        @. X = 2.0f0 * (X - T(sc.w_min)) / T(sc.w_range) - 1.0f0
    else
        throw(
            ArgumentError(
                "normalize_feature_batch! expects 1 or 2 feature rows, got $nrows",
            ),
        )
    end
    return X
end

@non_differentiable normalize_feature_batch!(::FeatureScaler, ::Any)

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
