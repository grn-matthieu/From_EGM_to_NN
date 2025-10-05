module DataNN

export generate_dataset
using Random

"""
    generate_dataset(G, S; mode=:full, nsamples::Int=0, rng=nothing)

Returns inputs X with rows = samples and columns = features.
Deterministic: features = (a,)
Stochastic:   features = (a, z)
The second return is `nothing` for compatibility.
"""
function generate_dataset(
    G,
    S;
    mode = :full,
    nsamples::Int = 0,
    rng::Union{Nothing,AbstractRNG} = nothing,
)
    # Use cash-on-hand w as the only feature if available, else construct w from a
    # Always use (y, w) as state space for NN
    # If w grid is present, use it; else construct w from a and y/z
    if isnothing(S)
        if mode == :full
            if haskey(G, :w)
                w = Float32.(G[:w].grid)
            else
                a = Float32.(G[:a].grid)
                R = 1.0f0 + Float32(getfield(G, :r, 0.0))
                y = Float32(getfield(G, :y, 0.0))
                w = R .* a .+ exp(y)
            end
            X = reshape(w, :, 1)
        else
            rng === nothing && error("generate_dataset requires `rng` when sampling w grid")
            if haskey(G, :w)
                wmin, wmax = extrema(Float32.(G[:w].grid))
            else
                a = Float32.(G[:a].grid)
                R = 1.0f0 + Float32(getfield(G, :r, 0.0))
                y = Float32(getfield(G, :y, 0.0))
                wgrid = R .* a .+ exp(y)
                wmin, wmax = extrema(wgrid)
            end
            n = nsamples > 0 ? nsamples : length(haskey(G, :w) ? G[:w].grid : G[:a].grid)
            X = rand(rng, Float32, n, 1) .* (wmax - wmin) .+ wmin
        end
        return (X, nothing)
    else
        z = Float32.(S.zgrid)
        if haskey(G, :w)
            Nw, Nz = length(G[:w].grid), length(z)
            if mode == :full
                W = repeat(Float32.(G[:w].grid), inner = Nz)
                Z = repeat(z, outer = Nw)
            else
                rng === nothing && error(
                    "generate_dataset requires `rng` when sampling w grid with shocks",
                )
                wmin, wmax = extrema(Float32.(G[:w].grid))
                zmin, zmax = minimum(z), maximum(z)
                n = nsamples > 0 ? nsamples : Nw * Nz
                W = rand(rng, Float32, n) .* (wmax - wmin) .+ wmin
                Z = rand(rng, Float32, n) .* (zmax - zmin) .+ zmin
            end
        else
            a = Float32.(G[:a].grid)
            R = 1.0f0 + Float32(getfield(G, :r, 0.0))
            y = Float32(getfield(G, :y, 0.0))
            Nw, Nz = length(a), length(z)
            if mode == :full
                W = repeat(R .* a .+ exp(y), inner = Nz)
                Z = repeat(z, outer = Nw)
            else
                rng === nothing && error(
                    "generate_dataset requires `rng` when sampling w grid with shocks",
                )
                wgrid = R .* a .+ exp(y)
                wmin, wmax = extrema(wgrid)
                zmin, zmax = minimum(z), maximum(z)
                n = nsamples > 0 ? nsamples : Nw * Nz
                W = rand(rng, Float32, n) .* (wmax - wmin) .+ wmin
                Z = rand(rng, Float32, n) .* (zmax - zmin) .+ zmin
            end
        end
        X = hcat(W, Z)
        return (X, nothing)
    end
end

end # module
