using Test
using ThesisProject
using ThesisProject.NNLoss: marg_u, inv_marg_u

@testset "CRRA marginal utility and inverse" begin
    θ1 = (; s = 1.0)
    θ2 = (; s = 2.0)

    # Scalars
    @test marg_u(2.0, θ1) ≈ 0.5
    @test marg_u(2.0, θ2) ≈ 2.0^(-2.0)
    @test inv_marg_u(marg_u(1.5, θ1), θ1) ≈ 1.5
    @test inv_marg_u(marg_u(1.5, θ2), θ2) ≈ 1.5

    # Arrays
    c = [0.5, 1.0, 2.0]
    @test inv_marg_u(marg_u(c, θ1), θ1) ≈ c
    @test inv_marg_u(marg_u(c, θ2), θ2) ≈ c

    # Numerical safety at 0
    @test isfinite(marg_u(0.0, θ1))
    @test isfinite(inv_marg_u(0.0, θ2))
end
