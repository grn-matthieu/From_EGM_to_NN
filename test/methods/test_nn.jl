using Test
using ThesisProject

# Minimal smoke test for NN method wiring
cfg = load_config(joinpath(@__DIR__, "..", "..", "config", "simple_baseline.yaml"))
# force solver method to NN
if !haskey(cfg[:solver], :method)
    cfg[:solver][:method] = :NN
else
    cfg[:solver][:method] = :NN
end

@testset "NN method wiring" begin
    method = build_method(cfg)
    @test method isa ThesisProject.AbstractMethod
    model = build_model(cfg)
    sol = solve(model, method, cfg)
    @test haskey(sol.policy, :c)
    @test !isnothing(sol.policy[:c].value)
end
