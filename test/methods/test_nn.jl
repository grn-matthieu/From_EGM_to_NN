using Test
using ThesisProject

# Minimal smoke test for NN method wiring
cfg = load_config(joinpath(@__DIR__, "..", "..", "config", "simple_baseline.yaml"))
cfg = cfg_patch(cfg, (:solver, :method) => :NN)

@testset "NN method wiring" begin
    method = build_method(cfg)
    @test method isa ThesisProject.AbstractMethod
    model = build_model(cfg)
    sol = solve(model, method, cfg)
    @test cfg_has(sol.policy, :c)
    @test !isnothing(cfg_get(sol.policy, :c).value)
end
