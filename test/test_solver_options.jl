using Test

@testset "EGM options" begin
    # Deterministic config with small grid and custom options
    cfgd = load_config(joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_config.yaml"))
    cfgd[:grids][:Na] = 40
    cfgd[:solver][:interp_kind] = :pchip
    cfgd[:solver][:warm_start] = :steady_state
    model = build_model(cfgd)
    method = build_method(cfgd)
    sol = solve(model, method, cfgd)
    @test sol.metadata[:converged] === true
    @test sol.metadata[:max_resid] < 1e-5
    # monotonicity on policies
    @test all(diff(sol.policy[:c].value) .>= -1e-8)
    @test all(diff(sol.policy[:a].value) .>= -1e-8)

    # (Optional) a stochastic smoke test can be added once configs align
end
