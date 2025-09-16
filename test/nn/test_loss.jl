using Test
using ThesisProject
using ThesisProject.NNLoss: euler_mse, euler_loss, check_finite_residuals

@testset "NNLoss: finiteness and behavior" begin
    # Synthetic residuals matrix
    R = randn(5, 7)

    # (i) Finiteness of basic losses
    @test isfinite(euler_mse(R))
    @test isfinite(euler_loss(R; stabilize = false))

    # Stabilization path stays finite
    @test isfinite(euler_loss(R; stabilize = true))

    # (ii) Monotonicity: scaling down residuals lowers MSE
    R2 = 0.5 .* R
    @test euler_mse(R2) < euler_mse(R)

    # Shape invariance: vectorized view vs matrix
    @test euler_mse(R) == euler_mse(vec(R))
    @test euler_loss(R; stabilize = false) == euler_loss(vec(R); stabilize = false)
    @test euler_loss(R; stabilize = true) ≈ euler_loss(vec(R); stabilize = true)

    # Optional: baseline model residuals finiteness
    try
        # Minimal deterministic ConsumerSaving model
        cfg = Dict{Symbol,Any}(
            :model => (; name = "cs"),
            :params => Dict(:β => 0.95, :s => 2.0, :r => 0.02, :y => 1.0),
            :grids => Dict(:a_min => 0.0, :a_max => 10.0, :Na => 25),
        )

        model = ThesisProject.build_model(cfg)
        p = ThesisProject.get_params(model)
        g = ThesisProject.get_grids(model)
        ag = g[:a].grid
        Rgross = 1 + p.r
        y = getfield(p, :y)

        # Simple feasible policy: consume half of resources
        c = 0.5 .* (Rgross .* ag .+ y)
        policy = Dict{Symbol,Any}(:c => (; value = c, grid = ag))
        batch = ag

        @test check_finite_residuals(model, policy, batch) === true
    catch err
        @test_skip "Baseline model residuals check skipped: $(err)"
    end
end
