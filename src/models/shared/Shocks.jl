module Shocks
export discretize, ShockOutput

using Statistics, LinearAlgebra, SpecialFunctions

struct ShockOutput
    z::Vector{Float64}
    Π::Matrix{Float64}
    π::Vector{Float64}
    diagnostics::Vector{Float64}
end

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

function find_invariant(Π::AbstractMatrix{<:Real}; tol=1e-12, maxit=1_000)
    # Power iteration method to find the invariant distribution
    π = fill(1.0 / size(Π, 1), size(Π, 1))
    for _ in 1:maxit
        π_new = π' * Π
        π_new ./= sum(π_new)
        if maximum(abs.(π_new .- π')) < tol
            return vec(π_new)
        end
        π .= vec(π_new)
    end
    error("Power iteration did not converge")
end

function discretize(method::AbstractString, ρ::Real, σ_shocks::Real, Nz::Int; m::Real=3.0, validate::Bool=false)
    if σ_shocks == 0 || Nz == 1 # Handles the degenerate case
        zgrid = [0.0]
        Π = reshape([1.0], 1, 1)
        π = [1.0]
        diagnostics = [0.0, 0.0, 1.0]
        return ShockOutput(zgrid, Π, π, diagnostics)
    end

    mth_str = lowercase(method)
    if mth_str == "tauchen"
        zgrid, Π = tauchen(ρ, σ_shocks, Nz; m=m)
    elseif mth_str == "rouwenhorst"
        zgrid, Π = rouwenhorst(ρ, σ_shocks, Nz)
    else
        error("Unknown discretization method: $method (use 'tauchen' or 'rouwenhorst')")
    end

    # Ensure symmetry and ordering
    zgrid = reverse(zgrid)
    Π = reverse(Π, dims=1)

    π = find_invariant(Π)

    # Compute additional moments for diagnostics
    diagnostics = Float64[μ_hat = mean(zgrid), σ_sq_hat = std(zgrid)^2, ρ_hat = cor(zgrid, zgrid)]

    out = ShockOutput(zgrid, Π, π, diagnostics)

    if validate
        using .TestDiscretization
        TestDiscretization.test_shock_discretization(out, σ_shocks^2, ρ) # Test the results vs true parameters
        println("Discretization validation passed.")
    end
    return out
end



end # module
