using Test

@testset "EGM options" begin
    # Deterministic config with small grid and custom options
    cfgd = deepcopy(SMOKE_CFG)
    cfgd[:grids][:Na] = 40
    cfgd[:solver][:interp_kind] = :pchip
    cfgd[:solver][:warm_start] = :steady_state
    model = build_model(cfgd)
    method = build_method(cfgd)
    sol = solve(model, method, cfgd)
    @test sol.metadata[:converged] === true
    @test sol.metadata[:max_resid] < 1e-5
    # monotonicity on policies
    @test is_nondec(sol.policy[:c].value; tol = 1e-8)
    @test is_nondec(sol.policy[:a].value; tol = 1e-8)

    # (Optional) a stochastic smoke test can be added once configs align
end
