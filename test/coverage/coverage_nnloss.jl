using Test
using ThesisProject

@testset "Coverage - NNLoss" begin
    # Short aliases
    L = ThesisProject.NNLoss

    # anneal_λ: edge and basic schedules
    @test L.anneal_λ(1, 1) == float(5.0) # E <= 1 returns final
    v1 = L.anneal_λ(1, 5; λ_start = 0.0, λ_final = 1.0, schedule = :linear)
    @test isapprox(v1, 0.0; atol = 1e-12)
    vlast = L.anneal_λ(5, 5; λ_start = 0.0, λ_final = 1.0, schedule = :cosine)
    @test isapprox(vlast, 1.0; atol = 1e-12)

    # stabilize_residuals
    R = [3.0, -4.0]
    @test L.stabilize_residuals(R; method = :none) == R
    S = L.stabilize_residuals(R; method = :log1p_square)
    @test length(S) == length(R)
    @test all(isfinite, S)
    @test_throws ArgumentError L.stabilize_residuals(R; method = :unknown)

    # euler_mse reductions and invalid arg
    @test isapprox(L.euler_mse([1.0, -2.0]), 2.5; atol = 1e-12, rtol = 1e-12)
    @test isapprox(L.euler_mse([1.0, -2.0]; reduction = :sum), 5.0; atol = 1e-12)
    @test_throws ArgumentError L.euler_mse([1.0]; reduction = :bad)

    # marg_u and inv_marg_u
    @test isapprox(L.marg_u(2.0, 2.0), 0.25; atol = 1e-12)
    @test isapprox(L.inv_marg_u(0.25, 2.0), 2.0; atol = 1e-12)
    θ = (; s = 1.0)
    mu = L.marg_u([1.0, 2.0], θ)
    @test isapprox(mu[1], 1.0; atol = 1e-12)
    @test isapprox(mu[2], 0.5; atol = 1e-12)

    # distance_to_bound
    d = L.distance_to_bound([0.9, 1.0, 1.2], 1.0)
    @test isapprox(d[1], 0.1; atol = 1e-12) && isapprox(d[2], 0.0; atol = 1e-12)

    # constraint_weights forms and unknown form
    ap = [0.9, 1.0, 1.2]
    w_exp = L.constraint_weights(ap, 1.0; form = :exp, α = 2.0, κ = 10.0)
    @test all(w_exp .>= 1.0)
    w_lin = L.constraint_weights(ap, 1.0; form = :linear, α = 2.0)
    @test isapprox(w_lin[2], 1.0; atol = 1e-12)
    @test_throws ArgumentError L.constraint_weights(ap, 1.0; form = :bad)

    # weighted_mse and euler_loss wrappers
    Rv = [1.0, -2.0]
    w = [1.0, 4.0]
    wm = L.weighted_mse(Rv, w)
    @test isfinite(wm) && wm > 0
    @test isapprox(L.euler_loss(Rv; weights = w), wm; atol = 1e-12)

    # assemble_euler_loss with NamedTuple config
    cfg = (
        residual_weighting = :exp,
        weight_α = 2.0,
        weight_κ = 10.0,
        stabilize = false,
        stab_method = :log1p_square,
    )
    ae = L.assemble_euler_loss(Rv, [0.9, 1.0, 1.2][1:2], 1.0, cfg)
    @test isfinite(ae)
end
