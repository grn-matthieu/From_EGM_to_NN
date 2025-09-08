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


function tauchen(ρ::Real, σ_ε::Real, Nz::Int; m::Real=3.0)
    σ_y = σ_ε / sqrt(1 - ρ^2 + eps())
    zmax = m * σ_y
    zmin = -zmax
    zgrid = collect(range(zmin, zmax; length=Nz))
    step = zgrid[2] - zgrid[1]
    Π = zeros(Nz, Nz)
    for j in 1:Nz
        μ = ρ * zgrid[j]
        for k in 1:Nz
            if k == 1
                Π[j, k] = Φ((zgrid[1] - μ + step/2)/σ_ε)
            elseif k == Nz
                Π[j, k] = 1 - Φ((zgrid[Nz] - μ - step/2)/σ_ε)
            else
                up = (zgrid[k] - μ + step/2) / σ_ε
                lo = (zgrid[k] - μ - step/2) / σ_ε
                Π[j, k] = Φ(up) - Φ(lo)
            end
        end
    end
    return zgrid, Π
end


function rouwenhorst(ρ::Real, σ_ε::Real, Nz::Int)
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
    σ_y = σ_ε / sqrt(1 - ρ^2 + eps())
    zgrid = collect(range(-σ_y*sqrt(Nz-1), σ_y*sqrt(Nz-1); length=Nz))
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

function _validate_invariant(π::AbstractVector{<:Real}, Π::AbstractMatrix{<:Real}; tol=1e-10)
    if !(isapprox(π, vec(π' * Π); atol=tol))
        max_dev = maximum(abs.(π .- vec(π' * Π)))
        error("Invariant distribution check failed: maximum deviation $max_dev exceeds tolerance $tol")
    end
    return nothing
end


function get_shock_params(shocks::AbstractDict)
    ρ = get(shocks, :ρ_shock, 0.0)
    σ_ε = get(shocks, :σ_shock, 0.0)
    Nz      = get(shocks, :Nz, 7)
    method  = get(shocks, :method, "tauchen")
    m       = get(shocks, :m, 3.0)
    validate= get(shocks, :validate, true)
    return ρ, σ_ε, Nz, method, m, validate
end


function discretize(shocks::AbstractDict)::ShockOutput
    ρ, σ_ε, Nz, method, m, validate = get_shock_params(shocks)

    if σ_ε == 0 || Nz == 1 # degenerate case
        zgrid = [0.0]
        Π = reshape([1.0], 1, 1)
        π = [1.0]
        diagnostics = [0.0, 0.0, 1.0]
        return ShockOutput(zgrid, Π, π, diagnostics)
    end

    mth_str = lowercase(method)
    zgrid, Π = mth_str == "tauchen"  ? tauchen(ρ, σ_ε, Nz; m=m) :
                mth_str == "rouwenhorst" ? rouwenhorst(ρ, σ_ε, Nz) :
                error("Unknown method: $method")

    π = find_invariant(Π)

    if validate
        _validate_invariant(π, Π)
    end

    # Diagnostics
    μ = sum(π .* zgrid)
    σ2 = sum(π .* (zgrid .- μ).^2)
    ρ_stat = markov_autocorr(zgrid, Π, π)
    diagnostics = Float64[μ, σ2, ρ_stat]
    return ShockOutput(zgrid, Π, π, diagnostics)
end

end # module

