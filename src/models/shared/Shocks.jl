module Shocks
export discretize_ar1

using Statistics, LinearAlgebra, SpecialFunctions

@inline Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))

function tauchen(ρ::Real, σ_shocks::Real, Nz::Int; m::Real=3.0)
    σ_shocks_y = σ_shocks / sqrt(1 - ρ^2 + eps())
    zmax = m * σ_shocks_y
    zmin = -zmax
    zgrid = collect(range(zmin, zmax; length=Nz))
    step = zgrid[2] - zgrid[1]
    P = zeros(Nz, Nz)
    for j in 1:Nz
        μ = ρ * zgrid[j]
        for k in 1:Nz
            if k == 1
                P[j, k] = Φ((zgrid[1] - μ + step/2)/σ_shocks)
            elseif k == Nz
                P[j, k] = 1 - Φ((zgrid[Nz] - μ - step/2)/σ_shocks)
            else
                up = (zgrid[k] - μ + step/2) / σ_shocks
                lo = (zgrid[k] - μ - step/2) / σ_shocks
                P[j, k] = Φ(up) - Φ(lo)
            end
        end
    end
    return zgrid, P
end

function rouwenhorst(ρ::Real, σ_shocks::Real, Nz::Int)
    p = (1 + ρ) / 2
    q = p
    P = [p 1-p; 1-q q]
    for n in 3:Nz
        Pold = P
        P = zeros(n, n)
        P[1:end-1, 1:end-1] .+= p .* Pold
        P[1:end-1, 2:end]   .+= (1-p) .* Pold
        P[2:end,   1:end-1] .+= (1-q) .* Pold
        P[2:end,   2:end]   .+= q .* Pold
        P[2:end-1, :] .*= 0.5
    end
    σ_shocks_y = σ_shocks / sqrt(1 - ρ^2 + eps())
    zgrid = collect(range(-σ_shocks_y, σ_shocks_y; length=Nz))
    return zgrid, P
end

function discretize_ar1(method::AbstractString, ρ::Real, σ_shocks::Real, Nz::Int; m::Real=3.0)
    if σ_shocks == 0 || Nz == 1
        return [0.0], reshape([1.0], 1, 1)
    end
    mth = lowercase(method)
    if mth == "tauchen"
        return tauchen(ρ, σ_shocks, Nz; m=m)
    elseif mth == "rouwenhorst"
        return rouwenhorst(ρ, σ_shocks, Nz)
    else
        error("Unknown discretization method: $method (use 'tauchen' or 'rouwenhorst')")
    end
end

end # module
