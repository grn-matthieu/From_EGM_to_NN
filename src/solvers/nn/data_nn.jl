module DataNN

export generate_dataset
using Random

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
    μ = Float32(P.y)
    if isnothing(S)
        n = mode == :full ? length(a) : (nsamples > 0 ? nsamples : length(a))
        if mode == :full
            A = a
        else
            amin, amax = extrema(a)
            A = rand(rng, Float32, n) .* (amax - amin) .+ amin
        end
        Y = fill(Float32(exp(μ)), length(A))
        W = @. Rg * A + Y
        return (hcat(Y, W), nothing)
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
        Y = exp.(μ .+ Z)
        W = @. Rg * A + Y
        return (hcat(Y, W), nothing)
    end
end

end # module
