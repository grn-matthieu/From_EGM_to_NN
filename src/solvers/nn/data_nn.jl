module DataNN

export generate_dataset
using Random
using Statistics: mean

"""
    generate_dataset(G, S, P; mode=:full, nsamples::Int=0, rng=Random.default_rng())

Returns inputs `X` with rows = samples and columns = features.
Deterministic: features = (y, w)
Stochastic:   features = (y, w)
The second return is `nothing` for compatibility.
"""
function generate_dataset(
    G,
    S,
    P;
    mode = :full,
    nsamples::Int = 0,
    rng::AbstractRNG = Random.default_rng(),
)
    a = Float32.(G[:a].grid)
    Rg = 1.0f0 + Float32(P.r)
    y_levels =
        hasproperty(P, :y) && P.y isa AbstractVector ? Float32.(collect(P.y)) :
        Float32[Float32(P.y)]
    y_dim = length(y_levels)
    extra_cols = y_dim > 1 ? y_dim : 0
    feature_dim = 1 + extra_cols + 1
    mean_income = mean(y_levels)

    function assemble_features(
        A_draws::AbstractVector{Float32},
        Y_components::AbstractMatrix{Float32},
    )
        n = length(A_draws)
        X = Matrix{Float32}(undef, n, feature_dim)
        mean_vec = vec(mean(Y_components; dims = 1))
        X[:, 1] .= mean_vec
        if extra_cols > 0
            @inbounds for j = 1:extra_cols
                X[:, 1+j] .= Y_components[j, :]
            end
        end
        X[:, end] .= @. Rg * A_draws + mean_vec
        return X
    end

    if isnothing(S)
        n = mode == :full ? length(a) : (nsamples > 0 ? nsamples : length(a))
        A_draws =
            mode == :full ? a :
            rand(rng, Float32, n) .* (Float32(maximum(a)) - Float32(minimum(a))) .+
            Float32(minimum(a))
        Y_components =
            extra_cols > 0 ? repeat(y_levels, 1, length(A_draws)) :
            reshape(Float32[mean_income], 1, length(A_draws))
        if extra_cols == 0
            Y_components = reshape(Float32(mean_income), 1, length(A_draws))
        end
        return (assemble_features(A_draws, Y_components), nothing)
    elseif hasproperty(S, :process) && S.process == :gaussian_linear && hasproperty(P, :Σ)
        Σ = Matrix{Float64}(P.Σ)
        μ_vec = Float64.(y_levels)
        chol = cholesky(Symmetric(Σ), check = false).L
        amin, amax = extrema(a)
        n =
            mode == :full ? length(a) * max(1, y_dim) :
            (nsamples > 0 ? nsamples : length(a) * max(1, y_dim))
        A_draws = rand(rng, Float32, n) .* (Float32(amax) - Float32(amin)) .+ Float32(amin)
        comps = Matrix{Float32}(undef, max(1, y_dim), n)
        @inbounds for i = 1:n
            ε = randn(rng, Float64, y_dim)
            y_vec = μ_vec .+ chol * ε
            if extra_cols > 0
                comps[:, i] .= Float32.(y_vec)
            else
                comps[1, i] = Float32(sum(y_vec) / y_dim)
            end
        end
        return (assemble_features(A_draws, comps), nothing)
    else
        z = Float32.(S.zgrid)
        Na, Nz = length(a), length(z)
        if mode == :full
            A = repeat(a, inner = Nz)
            Z = repeat(z, outer = Na)
        else
            amin, amax = extrema(a)
            zmin, zmax = minimum(z), maximum(z)
            n = nsamples > 0 ? nsamples : Na * Nz
            A = rand(rng, Float32, n) .* (amax - amin) .+ amin
            Z = rand(rng, Float32, n) .* (zmax - zmin) .+ zmin
        end
        μ = log(mean_income)
        Y = exp.(μ .+ Z)
        comps =
            extra_cols > 0 ? repeat(y_levels, 1, length(Y)) :
            reshape(Float32.(Y), 1, length(Y))
        return (assemble_features(A, comps), nothing)
    end
end

end # module
