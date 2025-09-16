using Test
using ThesisProject
using ThesisProject.NNLoss: EulerResidual

@testset "EulerResidual MC: finiteness + monotonicity" begin
    # Synthetic linear dynamics with CRRA utility
    # Set β*R = 1 and T identity so that stationary policy a' = a implies c' = c
    θ = (; β = 1.0, s = 1.0, R = 1.0)

    # Define environment functions in test scope
    resources(a, y; θ) = θ.R * a + y
    T(y, ε; θ) = y  # shockless for this minimal test
    R(a′, y′; θ) = θ.R

    # Batch of states
    B = 32
    a = abs.(randn(B))            # nonnegative assets
    y = fill(1.0, B)              # constant income

    # Deterministic sampler returning a single draw
    sampler() = 0.0

    # Policy family: a' = ρ * a; FOC-consistent when ρ = 1 (a' = a)
    policy_ρ(ρ) = (a, y; θ) -> ρ * a

    # Far from FOC vs closer
    ρ_far = 1.3
    ρ_close = 1.05
    ρ_star = 1.0

    R_far = EulerResidual(a, y; θ = θ, policy = policy_ρ(ρ_far), sampler = sampler, nMC = 1)
    R_close =
        EulerResidual(a, y; θ = θ, policy = policy_ρ(ρ_close), sampler = sampler, nMC = 1)
    R_star =
        EulerResidual(a, y; θ = θ, policy = policy_ρ(ρ_star), sampler = sampler, nMC = 1)

    # Finiteness
    @test all(isfinite, R_far)
    @test all(isfinite, R_close)
    @test all(isfinite, R_star)

    # Residual decreases as policy nudges toward FOC
    @test norm(R_close) < norm(R_far)

    # At the FOC-consistent policy, residuals are (numerically) ~ 0
    @test maximum(abs, R_star) < 1e-10
end
