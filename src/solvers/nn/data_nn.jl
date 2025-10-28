
module DataNN

export generate_dataset
using Random
using Statistics: mean
using LinearAlgebra: cholesky, Symmetric
using ..CSVarUtils: csvar_component_log_means

# Helper to generate random draws within bounds
sample_uniform(rng, n, lo, hi) = rand(rng, Float32, n) .* (hi - lo) .+ lo

# Assemble feature matrix: [income, income_components..., wealth]
function assemble_features(
    A_draws::AbstractVector{Float32},
    Y_components::AbstractMatrix{Float32},
    Rg::Float32,
    use_exp::Bool,
)
    n = length(A_draws)
    y_dim = size(Y_components, 1)

    # For non-CSVAR models with y_dim==1, use 2 features: [mean_income, wealth]
    # For all other cases, use y_dim+2 features: [total_income, components..., wealth]
    include_components = use_exp || y_dim > 1
    feature_dim = include_components ? (1 + y_dim + 1) : 2
    X = Matrix{Float32}(undef, n, feature_dim)

    # First column: total income
    if use_exp
        income = vec(sum(exp.(Y_components); dims = 1))
    else
        income = Float32(mean(Y_components))
    end
    X[:, 1] .= income

    if include_components
        # Middle columns: individual income components
        @inbounds for j = 1:y_dim
            X[:, 1+j] .= Y_components[j, :]
        end
        # Last column: wealth
        X[:, end] .= @. Rg * A_draws + (use_exp ? income : X[:, 1])
    else
        # No component columns, just wealth as second column
        X[:, 2] .= @. Rg * A_draws + income
    end

    return X
end

"""
    generate_dataset(G, S, P; mode=:full, nsamples::Int=0, rng=Random.default_rng())

Generate feature matrix for neural network training.
Returns (X, nothing) where X has shape (n_samples, n_features).
Features: [total_income, income_components..., wealth]
"""
function generate_dataset(
    G,
    S,
    P;
    mode = :full,
    nsamples::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
)
    a_grid = Float32.(G[:a].grid)
    amin, amax = Float32.(extrema(a_grid))
    Rg = 1.0f0 + Float32(P.r)

    # Detect model type and extract income configuration
    is_csvar = isdefined(P, :A) && isdefined(P, :Σ)
    y_vec = P.y isa AbstractVector ? Float64.(P.y) : [Float64(P.y)]
    y_dim = length(y_vec)

    # Determine sample size
    n_base = mode == :full ? length(a_grid) : max(nsamples, length(a_grid))
    is_gaussian = !isnothing(S) && isdefined(S, :process) && S.process == :gaussian_linear
    n = is_gaussian ? n_base * max(1, y_dim) : n_base

    # Generate asset draws
    A_draws = mode == :full && !is_gaussian ? a_grid : sample_uniform(rng, n, amin, amax)

    # Generate income components based on model type
    if is_csvar
        log_means = csvar_component_log_means(P)

        if isnothing(S) || (isdefined(S, :zgrid) && length(S.zgrid) == 1)
            # Deterministic CSVAR: use log means
            Y_components = repeat(Float32.(log_means), 1, n)
        elseif is_gaussian
            # Stochastic CSVAR with gaussian shocks
            Σ = Matrix{Float64}(P.Σ)
            chol = cholesky(Symmetric(Σ), check = false).L
            Y_components = Matrix{Float32}(undef, y_dim, n)
            @inbounds for i = 1:n
                ε = randn(rng, Float64, y_dim)
                Y_components[:, i] .= Float32.(log_means .+ chol * ε)
            end
        else
            # Discretized shocks: use grid
            z = Float32.(S.zgrid)
            Na, Nz = length(a_grid), length(z)
            A_draws =
                mode == :full ? repeat(a_grid, inner = Nz) :
                sample_uniform(rng, n, amin, amax)
            Z =
                mode == :full ? repeat(z, outer = Na) :
                sample_uniform(rng, n, minimum(z), maximum(z))
            Y_components = repeat(Float32.(log_means), 1, length(A_draws))
        end

        return (assemble_features(A_draws, Y_components, Rg, true), nothing)
    else
        # Standard model
        if isnothing(S) || (isdefined(S, :zgrid) && length(S.zgrid) == 1)
            # Deterministic: use mean income
            Y_components = repeat(Float32.(y_vec), 1, n)
        elseif is_gaussian
            # Gaussian shocks
            Σ = Matrix{Float64}(P.Σ)
            chol = cholesky(Symmetric(Σ), check = false).L
            Y_components = Matrix{Float32}(undef, y_dim, n)
            @inbounds for i = 1:n
                ε = randn(rng, Float64, y_dim)
                Y_components[:, i] .= Float32.(y_vec .+ chol * ε)
            end
        else
            # Discretized shocks
            z = Float32.(S.zgrid)
            Na, Nz = length(a_grid), length(z)
            A_draws =
                mode == :full ? repeat(a_grid, inner = Nz) :
                sample_uniform(rng, n, amin, amax)
            Z =
                mode == :full ? repeat(z, outer = Na) :
                sample_uniform(rng, n, minimum(z), maximum(z))
            μ = log(mean(y_vec))
            Y_components = reshape(Float32.(exp.(μ .+ Z)), 1, length(A_draws))
        end

        return (assemble_features(A_draws, Y_components, Rg, false), nothing)
    end
end

end # module
