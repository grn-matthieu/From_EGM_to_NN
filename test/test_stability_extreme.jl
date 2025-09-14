using Test
using ThesisProject
using ThesisProject.Determinism: make_rng

@testset "Extreme parameter stability" begin
    # Deterministic case
    cfg_det = deepcopy(SMOKE_CFG)
    cfg_det[:solver][:method] = "EGM"
    cfg_det[:params][:β] = 0.999
    cfg_det[:params][:σ] = 10.0
    model_det = build_model(cfg_det)
    method_det = build_method(cfg_det)
    sol_det = solve(model_det, method_det, cfg_det; rng = make_rng(0))
    c_det = sol_det.policy[:c].value
    @test all(isfinite, c_det)
    @test minimum(c_det) > 0
    @test all(isfinite, sol_det.policy[:c].euler_errors)

    # Stochastic case
    cfg_stoch = deepcopy(SMOKE_STOCH_CFG)
    cfg_stoch[:solver][:method] = "EGM"
    cfg_stoch[:params][:β] = 0.999
    cfg_stoch[:params][:σ] = 10.0
    model_stoch = build_model(cfg_stoch)
    method_stoch = build_method(cfg_stoch)
    sol_stoch = solve(model_stoch, method_stoch, cfg_stoch; rng = make_rng(0))
    c_stoch = sol_stoch.policy[:c].value
    @test all(isfinite, c_stoch)
    @test minimum(c_stoch) > 0
    @test all(isfinite, sol_stoch.policy[:c].euler_errors)
end
