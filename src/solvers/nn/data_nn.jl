module DataNN

export sample_training
using Random
using Statistics: mean
using LinearAlgebra: cholesky, Symmetric
using ..CSVarUtils: csvar_component_log_means

# Helper to generate random draws within bounds
sample_uniform(rng, n, lo, hi) = rand(rng, Float32, n) .* (hi - lo) .+ lo

"""
    sample_training(G, S; mode = :rand, nsamples = 4096, rng = Random.default_rng(), settings = nothing, P = nothing)

Unified data generation helper for the NN solver.
Implements the previous `sample_training_features` behaviour and also
supports `mode=:full` (replacing the deprecated `generate_dataset`).
Returns `(X, nsamples)` where `X` is a Float32 matrix with shape
`(nsamples, n_features)`.
Requires `P` to be provided via keyword argument.
"""
function sample_training(
    G,
    S;
    mode = :rand,
    nsamples::Int = 4096,
    rng::AbstractRNG = Random.default_rng(),
    settings = nothing,
    P = nothing,
)
    base_P =
        P === nothing ?
        error("sample_training requires parameter container `P` via keyword argument") : P

    # Full-mode: construct dataset covering the grid (Na * Nz where applicable)
    if mode === :full
        a_grid = Float32.(G[:a].grid)
        amin, amax = Float32.(extrema(a_grid))
        Rg = 1.0f0 + Float32(base_P.r)

        y_vec = base_P.y isa AbstractVector ? Float64.(base_P.y) : [Float64(base_P.y)]
        y_dim = length(y_vec)
        is_csvar = isdefined(base_P, :A) && isdefined(base_P, :Σ) && y_dim > 1

        is_gaussian =
            !isnothing(S) && isdefined(S, :process) && S.process == :gaussian_linear

        if is_csvar
            log_means = csvar_component_log_means(base_P)
            if isnothing(S) || (isdefined(S, :zgrid) && length(S.zgrid) == 1)
                Y_components = repeat(Float32.(log_means), 1, length(a_grid))
                A_draws = a_grid
            elseif is_gaussian
                Na = length(a_grid)
                Y_components = Matrix{Float32}(undef, y_dim, Na)
                Σ = Matrix{Float64}(base_P.Σ)
                chol = cholesky(Symmetric(Σ), check = false).L
                @inbounds for i = 1:Na
                    ε = randn(rng, Float64, y_dim)
                    Y_components[:, i] .= Float32.(log_means .+ chol * ε)
                end
                A_draws = a_grid
            else
                z = Float32.(S.zgrid)
                Na, Nz = length(a_grid), length(z)
                A_draws = repeat(a_grid, inner = Nz)
                Z = repeat(z, outer = Na)
                Y_components = repeat(Float32.(log_means), 1, length(A_draws))
            end

            n = length(A_draws)
            income = vec(sum(exp.(Y_components); dims = 1))
            feature_dim = 1 + y_dim + 1
            X = Matrix{Float32}(undef, n, feature_dim)
            X[:, 1] .= income
            @inbounds for j = 1:y_dim
                X[:, 1+j] .= Y_components[j, :]
            end
            X[:, end] .= @. Rg * A_draws + income
            return X, size(X, 1)
        else
            if isnothing(S) || (isdefined(S, :zgrid) && length(S.zgrid) == 1)
                Y_components = repeat(Float32.(y_vec), 1, length(a_grid))
                A_draws = a_grid
            elseif is_gaussian
                Na = length(a_grid)
                Y_components = Matrix{Float32}(undef, y_dim, Na)
                Σ = Matrix{Float64}(base_P.Σ)
                chol = cholesky(Symmetric(Σ), check = false).L
                @inbounds for i = 1:Na
                    ε = randn(rng, Float64, y_dim)
                    Y_components[:, i] .= Float32.(y_vec .+ chol * ε)
                end
                A_draws = a_grid
            else
                z = Float32.(S.zgrid)
                Na, Nz = length(a_grid), length(z)
                A_draws = repeat(a_grid, inner = Nz)
                Z = repeat(z, outer = Na)
                μ = log(mean(y_vec))
                Y_components = reshape(Float32.(exp.(μ .+ Z)), 1, length(A_draws))
            end

            n = length(A_draws)
            income = vec(sum(Y_components; dims = 1))
            feature_dim = 2
            X = Matrix{Float32}(undef, n, feature_dim)
            X[:, 1] .= income
            X[:, 2] .= @. Rg * A_draws + income
            return X, size(X, 1)
        end
    end

    # Random sampling mode (original sample_training_features logic)
    settings === nothing &&
        error("`sample_training` requires NN solver settings with w_min/w_max bounds")

    a_grid = Float32.(G[:a].grid)
    amin, amax = Float32.(extrema(a_grid))

    w_lo = Float32(getproperty(settings, :w_min))
    w_hi = Float32(getproperty(settings, :w_max))

    Rg = 1.0f0 + Float32(base_P.r)
    is_csvar = size(base_P.y, 1) > 1

    if is_csvar
        log_means = csvar_component_log_means(base_P)
        y_dim = length(log_means)
        extra_cols = y_dim
        feature_dim = 1 + extra_cols + 1

        mean_vec = Vector{Float32}(undef, nsamples)
        component_mat = Matrix{Float32}(undef, extra_cols, nsamples)
        W = Vector{Float32}(undef, nsamples)

        Σ = Matrix{Float64}(base_P.Σ)
        chol = cholesky(Symmetric(Σ), check = false).L
        comps_tmp = Matrix{Float32}(undef, extra_cols, nsamples)
        @inbounds for i = 1:nsamples
            ε = randn(rng, Float64, y_dim)
            y_vec = log_means .+ chol * ε
            comps_tmp[:, i] .= Float32.(y_vec)
        end
        mean_draw = vec(sum(exp.(comps_tmp); dims = 1))
        W .= rand(rng, Float32, nsamples) .* (w_hi - w_lo) .+ w_lo
        mean_vec .= mean_draw
        component_mat .= comps_tmp
    else
        feature_dim = 2
        extra_cols = 0
        base_income = base_P.y

        mean_vec = Vector{Float32}(undef, nsamples)
        component_mat = Matrix{Float32}(undef, 1, nsamples)
        W = Vector{Float32}(undef, nsamples)

        σ = base_P.σ_shock
        μ = Float32(log(base_income))

        rand!(rng, W)
        @. W = w_lo + W * (w_hi - w_lo)

        eps = randn(rng, Float32, nsamples)
        component_mat[1, :] .= eps
        @. component_mat[1, :] = exp(μ + σ * component_mat[1, :])
        mean_vec .= component_mat[1, :]
    end

    X = Matrix{Float32}(undef, nsamples, feature_dim)
    X[:, 1] .= mean_vec
    if extra_cols > 0
        for j = 1:extra_cols
            X[:, 1+j] .= component_mat[j, :]
        end
    end
    X[:, end] .= W
    return X, nsamples
end

end # module
