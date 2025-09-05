module Shocks

using SpecialFunctions: erf

export discretize, ShockOutput

struct ShockOutput
    zgrid::Vector{Float64}
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


function tauchen(ρ::Real, σ_shock::Real, Nz::Int; m::Real=3.0)
    σ_shocks_y = σ_shock / sqrt(1 - ρ^2 + eps())
    zmax = m * σ_shocks_y
    zmin = -zmax
    zgrid = collect(range(zmin, zmax; length=Nz))
    step = zgrid[2] - zgrid[1]
    Π = zeros(Nz, Nz)
    for j in 1:Nz
        μ = ρ * zgrid[j]
        for k in 1:Nz
            if k == 1
                Π[j, k] = Φ((zgrid[1] - μ + step/2)/σ_shock)
            elseif k == Nz
                Π[j, k] = 1 - Φ((zgrid[Nz] - μ - step/2)/σ_shock)
            else
                up = (zgrid[k] - μ + step/2) / σ_shock
                lo = (zgrid[k] - μ - step/2) / σ_shock
                Π[j, k] = Φ(up) - Φ(lo)
            end
        end
    end
    return zgrid, Π
end


function rouwenhorst(ρ::Real, σ_shock::Real, Nz::Int)
    p = (1 + ρ) / 2
    q = p
    Π = [p 1-p; 1-q q]
    for n in 3:Nz
        Πold = Π
        Π = zeros(n, n)
        Π[1:end-1, 1:end-1] .+= p .* Πold
        Π[1:end-1, 2:end]   .+= (1-p) .* Πold
        Π[2:end,   1:end-1] .+= (1-q) .* Πold
        Π[2:end,   2:end]   .+= q .* Πold
        Π[2:end-1, :] .*= 0.5
    end
    σ_shocks_y = σ_shock / sqrt(1 - ρ^2 + eps())
    zgrid = collect(range(-σ_shocks_y*sqrt(Nz-1), σ_shocks_y*sqrt(Nz-1); length=Nz))
    @debug typeof(zgrid)
    return zgrid, Π
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

# function test_shock_discretization(shock::ShockOutput, σ_ϵ_sq, ρ)
#     Π, π, diagnostics = shock.Π, shock.π, shock.diagnostics

#     # Sanity checks on the invariant distribution
#     @test all(π .>= 0)
#     @test isapprox(sum(π), 1.0)
#     @test isapprox(π' * Π, π'; atol=1e-8)

#     μ_stat, σ_ϵ_sq_stat, ρ_stat = diagnostics

#     # Checks on the moments : this should match an AR(1) distribution
#     @test isapprox(μ_stat, 0.0; atol=1e-8)
#     println("σ_ϵ_sq_stat: $σ_ϵ_sq_stat, σ_ϵ²: $σ_ϵ_sq, ρ_stat: $ρ_stat, ρ: $ρ\n")

#     # For the variance testing, we allow a 5% tolerance
#     # The sample size is between 7 and 11 points, so this should be reasonable
#     σ2_theory = σ_ϵ_sq / (1 - ρ^2)
#     @test isapprox(σ_ϵ_sq_stat, σ2_theory; atol=0.05 * σ2_theory)

#     @test isapprox((ρ_stat - ρ), 0.0; atol=1e-8)
# end

function get_shock_params(shocks::AbstractDict)
    ρ_shock = get(shocks, :ρ_shock, 0.0)
    σ_shock = get(shocks, :σ_shock, 0.0)
    Nz      = get(shocks, :Nz, 7)
    method  = get(shocks, :method, "tauchen")
    m       = get(shocks, :m, 3.0)
    validate= get(shocks, :validate, true)
    return ρ_shock, σ_shock, Nz, method, m, validate
end


function discretize(shocks::AbstractDict)::ShockOutput
    ρ_shock, σ_shock, Nz, method, m = get_shock_params(shocks)

    if σ_shock == 0 || Nz == 1 # Handles the degenerate case
        zgrid = [0.0]
        Π = reshape([1.0], 1, 1)
        π = [1.0]
        diagnostics = [0.0, 0.0, 1.0]
        return ShockOutput(zgrid, Π, π, diagnostics)
    end

    mth_str = lowercase(method)
    if mth_str == "tauchen"
        zgrid, Π = tauchen(ρ_shock, σ_shock, Nz; m=m)
    elseif mth_str == "rouwenhorst"
        zgrid, Π = rouwenhorst(ρ_shock, σ_shock, Nz)
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

    # if validate
    #     test_shock_discretization(out, σ_shock^2, ρ) # Test the results vs true parameters
    #     println("Discretization validation passed.")
    # end

    return out
end


# """
# Simulates T periods of shocks given a ShockOutput config.
# Outputs the time series for ONE agent (actual shock values, not indices).
# """
# function simulate_shocks(T::Int, shocks::ShockOutput, rng::AbstractRNG)::Vector{Float64}
#     zgrid = shocks.zgrid
#     Π = shocks.Π
#     π = shocks.π

#     Nz = length(zgrid)
#     sim_shocks = zeros(T)
#     idx = sample(1:Nz, Weights(π))

#     for t in 1:T
#         sim_shocks[t] = zgrid[idx]
#         idx = sample(rng, 1:Nz, Weights(Π[idx, :]))  # Next state index
#     end

#     return sim_shocks
# end

end # module
