module DataNN

export generate_dataset
using Random

"""
    generate_dataset(G, S; mode=:full, nsamples::Int=0, rng=Random.default_rng())

Returns inputs X with rows = samples and columns = features.
Deterministic: features = (a,)
Stochastic:   features = (a, z)
The second return is `nothing` for compatibility.
"""
function generate_dataset(G, S; mode = :full, nsamples::Int = 0, rng = Random.default_rng())
    a = Float32.(G[:a].grid)
    if isnothing(S)
        if mode == :full
            X = reshape(a, :, 1)                    # Na×1
        else
            amin, amax = extrema(a)
            n = nsamples > 0 ? nsamples : length(a)
            X = rand(rng, Float32, n, 1) .* (amax - amin) .+ amin
        end
        return (X, nothing)
    else
        z = Float32.(S.zgrid)
        Na, Nz = length(a), length(z)
        if mode == :full
            A = repeat(a, inner = Nz)                 # length Na*Nz
            Z = repeat(z, outer = Na)
        else
            amin, amax = extrema(a)
            zmin, zmax = minimum(z), maximum(z)
            n = nsamples > 0 ? nsamples : Na * Nz
            A = rand(rng, Float32, n) .* (amax - amin) .+ amin
            Z = rand(rng, Float32, n) .* (zmax - zmin) .+ zmin
        end
        X = hcat(A, Z)                               # (Na*Nz)×2 or n×2
        return (X, nothing)
    end
end

end # module
