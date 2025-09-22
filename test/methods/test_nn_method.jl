using Test
using ThesisProject

@testset "NN method adapter" begin
    cfg = cfg_patch(SMOKE_CFG, (:solver, :method) => :NN)
    model = build_model(cfg)
    method = build_method(cfg)
    sol = ThesisProject.solve(model, method, cfg)
    @test sol isa ThesisProject.API.Solution
    @test cfg_has(sol.policy, :c) && cfg_has(sol.policy, :a)
    @test sol.diagnostics.method == :NN
    @test isfinite(sol.metadata[:max_resid])
end
