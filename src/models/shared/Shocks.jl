module Shocks
export discretize, ShockOutput

using Statistics, LinearAlgebra, SpecialFunctions, StatsBase
using Test

struct ShockOutput
    z::Vector{Float64}
    Π::Matrix{Float64}
    π::Vector{Float64}
    diagnostics::Vector{Float64}
end

@inline Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))

function markov_autocorr(zgrid, Π, π)
    μ = sum(π .* zgrid)
    σ2 = sum(π .* (zgrid .- μ).^2)
    Ezzt1 = sum((π .* zgrid) .* (Π * zgrid))
    cov1 = Ezzt1 - μ^2
    return cov1 / σ2
end


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
    zgrid = collect(range(-σ_shocks_y*sqrt(Nz-1), σ_shocks_y*sqrt(Nz-1); length=Nz))
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

function test_shock_discretization(shock::ShockOutput, σ_ϵ_sq, ρ)
    Π, π, diagnostics = shock.Π, shock.π, shock.diagnostics

    # Sanity checks on the invariant distribution
    @test all(π .>= 0)
    @test isapprox(sum(π), 1.0)
    @test isapprox(π' * Π, π'; atol=1e-8)

    μ_stat, σ_ϵ_sq_stat, ρ_stat = diagnostics

    # Checks on the moments : this should match an AR(1) distribution
    @test isapprox(μ_stat, 0.0; atol=1e-8)
    println("σ_ϵ_sq_stat: $σ_ϵ_sq_stat, σ_ϵ²: $σ_ϵ_sq, ρ_stat: $ρ_stat, ρ: $ρ\n")

    # For the variance testing, we allow a 5% tolerance
    # The sample size is between 7 and 11 points, so this should be reasonable
    σ2_theory = σ_ϵ_sq / (1 - ρ^2)
    @test isapprox(σ_ϵ_sq_stat, σ2_theory; atol=0.05 * σ2_theory)

    @test isapprox((ρ_stat - ρ), 0.0; atol=1e-8)
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

    π = find_invariant(Π)

    # Compute additional moments for diagnostics
    μ_stat = sum(π .* zgrid)
    σ2_stat = sum(π .* (zgrid .- μ_stat).^2)
    ρ_stat = markov_autocorr(zgrid, Π, π)
    diagnostics = Float64[μ_stat, σ2_stat, ρ_stat]

    out = ShockOutput(zgrid, Π, π, diagnostics)

    if validate
        test_shock_discretization(out, σ_shocks^2, ρ) # Test the results vs true parameters
        println("Discretization validation passed.")
    end
    return out
end



end # module
