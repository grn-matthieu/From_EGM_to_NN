using Test
using ThesisProject

@testset "NN method adapter" begin
    cfg = deepcopy(SMOKE_CFG)
    cfg[:solver][:method] = :NN
    model = build_model(cfg)
    method = build_method(cfg)
    sol = ThesisProject.solve(model, method, cfg)
    @test sol isa ThesisProject.API.Solution
    @test haskey(sol.policy, :c) && haskey(sol.policy, :a)
    @test sol.diagnostics.method == :NN
    @test isfinite(sol.metadata[:max_resid])
end
