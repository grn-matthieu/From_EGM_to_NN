using Test

@testset "NNLoss: basic loss utilities" begin
    # Deterministic residuals
    R = [0.0, 1.0, -2.0]

    # euler_mse
    @test ThesisProject.NNLoss.euler_mse(R) ≈ (0^2 + 1^2 + (-2)^2) / 3
    @test ThesisProject.NNLoss.euler_mse(R; reduction = :sum) ≈ 5.0

    # stabilize_residuals reduces extreme magnitudes and stays finite
    let Ra = [0.1, 3.0, -4.0]
        T = ThesisProject.NNLoss.stabilize_residuals(Ra; method = :log1p_square)
        @test all(isfinite, T)
        @test maximum(abs, T) < maximum(abs, Ra)
    end

    # constraint_weights and weighted_mse
    ap = [0.9, 1.0, 1.5]
    a_min = 1.0
    w_exp =
        ThesisProject.NNLoss.constraint_weights(ap, a_min; α = 5.0, κ = 20.0, form = :exp)
    @test all(w_exp .>= 1)
    @test ThesisProject.NNLoss.weighted_mse(R, w_exp) ≥ ThesisProject.NNLoss.euler_mse(R)

    # euler_loss passthrough and stabilization path
    @test ThesisProject.NNLoss.euler_loss(R; stabilize = false) ≈
          ThesisProject.NNLoss.euler_mse(R)
    @test isfinite(ThesisProject.NNLoss.euler_loss(R; stabilize = true))

    # Error path for invalid reduction
    @test_throws ArgumentError ThesisProject.NNLoss.euler_mse(R; reduction = :foo)
end
