import ChainRulesCore: @non_differentiable
using CUDA: cu, CuArray, CUDA
using Statistics: mean
using LinearAlgebra: diag
using ..CSVarUtils: csvar_component_log_means

struct ScalarParams
    γ::Float64
    β::Float64
    r::Float64
    y::Float64
    y_components::Vector{Float64}
end

struct FeatureScaler
    mean_min::Float32
    mean_range::Float32
    y_min::Vector{Float32}
    y_range::Vector{Float32}
    w_min::Float32
    w_range::Float32
    has_shocks::Bool
    csvar_mode::Bool
end

function _income_bounds(P, S)
    is_csvar = isdefined(P, :A) && isdefined(P, :Σ)
    μ_vec =
        is_csvar ? csvar_component_log_means(P) :
        (
            isdefined(P, :y) && P.y isa AbstractVector ? Float64.(collect(P.y)) :
            [Float64(P.y)]
        )

    d = length(μ_vec)
    lower, upper = Vector{Float64}(undef, d), Vector{Float64}(undef, d)

    if S === nothing ||
       !(isdefined(S, :zgrid) && S.zgrid !== nothing && length(S.zgrid) > 0) &&
       !(isdefined(P, :Σ) && P.Σ !== nothing)
        # No shocks: bounds equal means
        lower .= μ_vec
        upper .= μ_vec
    elseif isdefined(S, :zgrid) && S.zgrid !== nothing && length(S.zgrid) > 0
        # Discrete shocks: add min/max shock values
        zmin, zmax = Float64(minimum(S.zgrid)), Float64(maximum(S.zgrid))
        @inbounds for i = 1:d
            lower[i] = μ_vec[i] + zmin
            upper[i] = μ_vec[i] + zmax
        end
    elseif isdefined(P, :Σ) && P.Σ !== nothing
        # Continuous shocks (CSVAR): use 3σ bounds
        std_vec = sqrt.(max.(diag(Matrix{Float64}(P.Σ)), 0.0))
        @inbounds for i = 1:d
            lower[i] = μ_vec[i] - 3 * std_vec[i]
            upper[i] = μ_vec[i] + 3 * std_vec[i]
        end
    else
        lower .= μ_vec
        upper .= μ_vec
    end
    return lower, upper
end

function FeatureScaler(P, G, S, settings)
    lower, upper = _income_bounds(P, S)
    y_dim = length(lower)
    is_csvar = isdefined(P, :A) && isdefined(P, :Σ)

    # Compute income bounds based on model type
    income_lower, income_upper = if is_csvar
        sum(exp.(lower)), sum(exp.(upper))
    else
        y_dim > 1 ? (sum(lower), sum(upper)) : (mean(lower), mean(upper))
    end

    mean_min = Float32(income_lower)
    mean_range = max(Float32(income_upper - income_lower), 1.0f-6)

    # Component-level normalization (CSVAR or multi-component models)
    include_components = is_csvar || y_dim > 1
    y_min_vec = include_components ? Float32.(lower) : Float32[]
    y_range_vec =
        include_components ? Float32.(max.(upper - lower, fill(1e-6, y_dim))) : Float32[]

    # Wealth bounds
    w_min = Float32(settings.w_min)
    w_range = max(Float32(settings.w_max) - w_min, 1.0f-6)

    return FeatureScaler(
        mean_min,
        mean_range,
        y_min_vec,
        y_range_vec,
        w_min,
        w_range,
        settings.has_shocks,
        is_csvar,
    )
end

function normalize_samples!(scaler::FeatureScaler, X)
    ncols = size(X, 2)
    y_dim = length(scaler.y_min)
    if y_dim == 0 && ncols == 2
        @. X[:, 1] = 2.0f0 * (X[:, 1] - scaler.mean_min) / scaler.mean_range - 1.0f0
        @. X[:, 2] = 2.0f0 * (X[:, 2] - scaler.w_min) / scaler.w_range - 1.0f0
    elseif y_dim > 0 && ncols == y_dim + 2
        @. X[:, 1] = 2.0f0 * (X[:, 1] - scaler.mean_min) / scaler.mean_range - 1.0f0
        for j = 1:y_dim
            min_val = scaler.y_min[j]
            range_val = scaler.y_range[j]
            @. X[:, 1+j] = 2.0f0 * (X[:, 1+j] - min_val) / range_val - 1.0f0
        end
        @. X[:, end] = 2.0f0 * (X[:, end] - scaler.w_min) / scaler.w_range - 1.0f0
    else
        throw(ArgumentError("normalize_samples! received unexpected feature count $ncols"))
    end
    return X
end

# explicit CPU in-place version
function normalize_feature_batch!(sc::FeatureScaler, X::AbstractMatrix{<:AbstractFloat})
    nrows = size(X, 1)
    y_dim = length(sc.y_min)
    T = eltype(X)
    if y_dim == 0 && nrows == 2
        @. X[1, :] = 2.0f0 * (X[1, :] - sc.mean_min) / sc.mean_range - 1.0f0
        @. X[2, :] = 2.0f0 * (X[2, :] - sc.w_min) / sc.w_range - 1.0f0
    elseif y_dim > 0 && nrows == y_dim + 2
        @. X[1, :] = 2.0f0 * (X[1, :] - sc.mean_min) / sc.mean_range - 1.0f0
        for j = 1:y_dim
            min_val = sc.y_min[j]
            range_val = sc.y_range[j]
            @. X[1+j, :] = 2.0f0 * (X[1+j, :] - min_val) / range_val - 1.0f0
        end
        @. X[end, :] = 2.0f0 * (X[end, :] - sc.w_min) / sc.w_range - 1.0f0
    else
        throw(
            ArgumentError(
                "normalize_feature_batch! received unexpected feature rows $nrows",
            ),
        )
    end
    return X
end

function build_feature_batch_from_states(
    scaler::FeatureScaler,
    y_components::AbstractMatrix,
    w::AbstractVector,
)
    y_dim = size(y_components, 1)
    n = length(w)
    y_dim > 0 || error("build_feature_batch_from_states requires positive state dimension")
    feature_dim = y_dim + 2
    X = Matrix{Float32}(undef, feature_dim, n)
    if scaler.csvar_mode
        mean_vals = vec(sum(exp.(y_components); dims = 1))
    else
        mean_vals = vec(sum(y_components; dims = 1)) ./ y_dim
    end
    X[1, :] .= Float32.(mean_vals)
    for j = 1:y_dim
        X[1+j, :] .= Float32.(y_components[j, :])
    end
    X[end, :] .= Float32.(w)
    normalize_feature_batch!(scaler, X)
    return X
end

function denormalize_feature_batch(scaler::FeatureScaler, batch::AbstractMatrix)
    T = eltype(batch)
    ncols = size(batch, 2)

    mean_vals =
        ((batch[1, :] .+ one(T)) ./ T(2)) .* T(scaler.mean_range) .+ T(scaler.mean_min)

    y_dim = length(scaler.y_min)
    comps = if y_dim == 0
        Matrix{T}(undef, 0, ncols)
    else
        B = @view batch[2:1+y_dim, :]                # source seulement, pas de mutation
        r = reshape(T.(scaler.y_range[1:y_dim]), y_dim, 1)
        m = reshape(T.(scaler.y_min[1:y_dim]), y_dim, 1)
        ((B .+ one(T)) ./ T(2)) .* r .+ m        # retourne une nouvelle matrice
    end

    # Denormalize cash-on-hand as well (was mistakenly returned in [-1, 1])
    w_vals = ((batch[end, :] .+ one(T)) ./ T(2)) .* T(scaler.w_range) .+ T(scaler.w_min)

    return mean_vals, comps, w_vals
end

# GPU in-place version without views nor scalar indexing
function normalize_feature_batch!(
    sc::FeatureScaler,
    X::CUDA.CuArray{T,2},
) where {T<:AbstractFloat}
    nrows = size(X, 1)
    y_dim = length(sc.y_min)
    if y_dim == 0 && nrows == 2
        mins = reshape(cu(T.([sc.mean_min, sc.w_min])), 2, 1)
        ranges = reshape(cu(T.([sc.mean_range, sc.w_range])), 2, 1)
        @. X = 2.0f0 * (X - mins) / ranges - 1.0f0
    elseif y_dim > 0 && nrows == y_dim + 2
        mins_vec = Vector{T}(undef, y_dim + 2)
        ranges_vec = Vector{T}(undef, y_dim + 2)
        mins_vec[1] = T(sc.mean_min)
        ranges_vec[1] = T(sc.mean_range)
        for j = 1:y_dim
            mins_vec[1+j] = T(sc.y_min[j])
            ranges_vec[1+j] = T(sc.y_range[j])
        end
        mins_vec[end] = T(sc.w_min)
        ranges_vec[end] = T(sc.w_range)
        mins = reshape(cu(mins_vec), length(mins_vec), 1)
        ranges = reshape(cu(ranges_vec), length(ranges_vec), 1)
        @. X = 2.0f0 * (X - mins) / ranges - 1.0f0
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
    y_dim = length(s.y_min)
    if y_dim == 0 && nrows == 2
        y1 = @. 2.0f0 * (X[1, :] - s.mean_min) / s.mean_range - 1.0f0
        w1 = @. 2.0f0 * (X[2, :] - s.w_min) / s.w_range - 1.0f0
        return vcat(reshape(y1, 1, :), reshape(w1, 1, :))
    elseif y_dim > 0 && nrows == y_dim + 2
        out = similar(X)
        out[1, :] .= @. 2.0f0 * (X[1, :] - s.mean_min) / s.mean_range - 1.0f0
        for j = 1:y_dim
            min_val = s.y_min[j]
            range_val = s.y_range[j]
            out[1+j, :] .= @. 2.0f0 * (X[1+j, :] - min_val) / range_val - 1.0f0
        end
        out[end, :] .= @. 2.0f0 * (X[end, :] - s.w_min) / s.w_range - 1.0f0
        return out
    else
        throw(
            ArgumentError(
                "normalize_feature_batch expects 1 or 2 feature rows, got $nrows",
            ),
        )
    end
end

get_param(container, name::Symbol, default) =
    isdefined(container, name) ?
    (val = getfield(container, name); val === nothing ? default : val) : default

function scalar_params(P)
    is_vector = isdefined(P, :y) && P.y isa AbstractVector
    y_vec = is_vector ? Float64.(collect(P.y)) : Float64[Float64(P.y)]
    y_mean = mean(y_vec)
    extra = is_vector && length(y_vec) > 1 ? y_vec : Float64[]
    return ScalarParams(Float64(P.γ), Float64(P.β), Float64(P.r), y_mean, extra)
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

income_dimension(P) = (isdefined(P, :y) && P.y isa AbstractVector) ? length(P.y) : 1

function nn_input_dimension(P)
    d = income_dimension(P)
    extra = d > 1 ? d : 0
    return 1 + extra + 1
end

input_dimension(::Any) = 2
