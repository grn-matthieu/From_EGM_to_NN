using Test
using ThesisProject.Chebyshev: chebyshev_basis
using ThesisProject.ProjectionCoefficients: solve_coefficients

@testset "least squares coefficients" begin
    x = collect(range(0.0, 1.0; length = 5))
    B = chebyshev_basis(x, 2, 0.0, 1.0)
    coeff_true = [1.0, -0.5, 0.75]
    y = B * coeff_true
    coeff_est = solve_coefficients(B, y)
    @test isapprox(coeff_est, coeff_true; atol = 1e-12)

    coeff_reg = solve_coefficients(B, y; λ = 1e-4)
    @test isapprox(coeff_reg, coeff_true; atol = 1e-4)
end
