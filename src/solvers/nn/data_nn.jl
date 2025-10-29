
module DataNN

export generate_dataset, sample_training_features
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
        income = vec(sum(Y_components; dims = 1))
    end
    X[:, 1] .= income

    if include_components
        # Middle columns: individual income components
        @inbounds for j = 1:y_dim
            X[:, 1+j] .= Y_components[j, :]
        end
        # Last column: wealth
        X[:, end] .= @. Rg * A_draws + income
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
        @debug println("Generated CSVAR dataset with $(size(Y_components, 2)) samples.")
        @debug println("Income component means: ", mean.(eachrow(Y_components)))
        @debug println("Income component stds: ", std.(eachrow(Y_components)))
        @debug println("Income component maxs: ", maximum.(eachrow(Y_components)))
        @debug println("Income component mins: ", minimum.(eachrow(Y_components)))

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

function sample_training_features(
    G,
    S,
    P_resid;
    mode = :rand,
    nsamples::Int = 4096,
    rng::AbstractRNG = Random.default_rng(),
    settings = nothing,
    P = nothing,
)
    base_P = P === nothing ? P_resid : P

    if mode === :full
        X_full, _ = generate_dataset(G, S, base_P; mode = :full, rng = rng)
        return X_full, size(X_full, 1)
    end

    settings === nothing && error(
        "`sample_training_features` requires NN solver settings with w_min/w_max bounds",
    )

    a_grid = Float32.(G[:a].grid)
    amin, amax = Float32.(extrema(a_grid))

    w_lo = Float32(getproperty(settings, :w_min))
    w_hi = Float32(getproperty(settings, :w_max))

    Rg = 1.0f0 + Float32(P_resid.r)
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

        filled = 0
        while filled < nsamples
            remaining = nsamples - filled
            a_draw = rand(rng, Float32, remaining) .* (amax - amin) .+ amin
            z_draw =
                σ == 0.0f0 ? fill(0.0f0, remaining) : σ .* randn(rng, Float32, remaining)
            y_tmp = exp.(μ .+ z_draw)
            W_tmp = @. Rg * a_draw + y_tmp
            keep = (W_tmp .>= w_lo) .& (W_tmp .<= w_hi)
            k = count(keep)
            if k == 0
                continue
            end
            take = min(k, remaining)
            idx = findall(keep)
            select_idx = idx[1:take]
            range = filled+1:filled+take
            W[range] .= W_tmp[select_idx]
            mean_vec[range] .= y_tmp[select_idx]
            component_mat[1, range] .= y_tmp[select_idx]
            filled += take
        end
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
