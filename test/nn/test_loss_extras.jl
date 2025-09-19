using Test
using ThesisProject

@testset "NNLoss: extra loss coverage" begin
    R = [-1.0, 0.0, 2.0]
    ap = [0.8, 1.0, 1.2]
    a_min = 1.0

    # Unknown stabilization method triggers error
    @test_throws ArgumentError ThesisProject.NNLoss.stabilize_residuals(
        R;
        method = :unknown,
    )

    # constraint_weights linear form and error branch
    w_lin = ThesisProject.NNLoss.constraint_weights(ap, a_min; α = 2.0, form = :linear)
    @test all(w_lin .>= 1)
    @test ThesisProject.NNLoss.euler_loss(R; weights = w_lin) >= 0
    @test_throws ArgumentError ThesisProject.NNLoss.constraint_weights(
        ap,
        a_min;
        form = :bogus,
    )

    # euler_loss with reduction switch
    @test isapprox(
        ThesisProject.NNLoss.euler_loss(R; reduction = :sum),
        ThesisProject.NNLoss.euler_mse(R; reduction = :sum),
    )

    # assemble_euler_loss across weighting options
    cfg_none = (
        stabilize = false,
        stab_method = :log1p_square,
        residual_weighting = :none,
        weight_α = 5.0,
        weight_κ = 20.0,
    )
    L_none = ThesisProject.NNLoss.assemble_euler_loss(R, ap, a_min, cfg_none)
    @test isapprox(L_none, ThesisProject.NNLoss.euler_mse(R))

    cfg_exp = (
        stabilize = true,
        stab_method = :log1p_square,
        residual_weighting = :exp,
        weight_α = 3.0,
        weight_κ = 10.0,
    )
    @test isfinite(ThesisProject.NNLoss.assemble_euler_loss(R, ap, a_min, cfg_exp))

    cfg_lin = (
        stabilize = false,
        stab_method = :log1p_square,
        residual_weighting = :linear,
        weight_α = 2.0,
        weight_κ = 10.0,
    )
    @test isfinite(ThesisProject.NNLoss.assemble_euler_loss(R, ap, a_min, cfg_lin))
end
