module TestDiscretization

using Test
include("../src/models/shared/Shocks.jl")
using .Shocks:ShockOutput

export test_shock_discretization

"""
    test_shock_discretization(shock::ShockOutput, σ_ϵ², ρ)

Run sanity and moment-matching checks on the discretized AR(1) process.
- π: invariant distribution
- zgrid: discretized grid
- σ_ϵ²: variance of the AR(1) shock
- ρ: AR(1) autocorrelation
"""
function test_shock_discretization(shock::ShockOutput, σ_ϵ², ρ)
    z, Π, π, diagnostics = shock.z, shock.Π, shock.π, shock.diagnostics

    # Sanity checks on the invariant distribution
    @test all(π .>= 0) "Invariant distribution contains negative elements."
    @test isapprox(sum(π), 1.0) "Invariant distribution does not sum to 1."
    @test isapprox(π' * Π, π'; atol=1e-8) "Invariant distribution is not invariant under the transition matrix."

    μ_hat, σ_sq_hat, ρ_hat = diagnostics

    # Checks on the moments : this should match an AR(1) distribution
    @test isapprox(μ_hat, 0.0; atol=1e-8) "Mean of discretized process is not zero."
    @test isapprox((σ_sq_hat^2 - σ_ϵ²) / (1 - ρ_hat^2), 0.0; atol=1e-8) "Second moment of discretized process is not matched."
    @test isapprox((ρ_hat - ρ), 0.0; atol=1e-8) "First-order autocorrelation of discretized process is not matched."
end

end # module