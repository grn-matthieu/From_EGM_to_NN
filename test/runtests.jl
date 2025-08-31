using ThesisProject

using Test

@testset "Simple EGM Tests" begin
    smoke_cfg = load_config("config/smoke_config/smoke_config.yaml")
    model = build_model(smoke_cfg)
    method = build_method(smoke_cfg)
    sol   = solve(model, method, smoke_cfg)
    @test !any(isnan, sol.policy.c)
end