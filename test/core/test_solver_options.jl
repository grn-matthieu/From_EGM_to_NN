using Test

@testset "EGM options" begin
    # Deterministic config with small grid and custom options
    cfgd = cfg_patch(
        SMOKE_CFG,
        (:grids, :Na) => 40,
        (:solver, :interp_kind) => :pchip,
        (:solver, :warm_start) => :steady_state,
    )
    model = build_model(cfgd)
    method = build_method(cfgd)
    sol = solve(model, method, cfgd)
    @test sol.metadata[:converged] === true
    @test sol.metadata[:max_resid] < 1e-5
    # monotonicity on policies
    @test is_nondec(cfg_get(sol.policy, :c).value; tol = 1e-8)
    @test is_nondec(cfg_get(sol.policy, :a).value; tol = 1e-8)

    # (Optional) a stochastic smoke test can be added once configs align
end
