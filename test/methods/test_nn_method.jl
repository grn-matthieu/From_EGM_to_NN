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

@testset "NN method respects solver options" begin
    cfg = cfg_patch(
        SMOKE_CFG,
        (:solver, :method) => :NN,
        (:solver, :epochs) => 2,
        (:solver, :verbose) => false,
    )
    model = build_model(cfg)
    method = build_method(cfg)
    io = IOBuffer()
    sol = redirect_stdout(io) do
        ThesisProject.solve(model, method, cfg)
    end
    output = String(take!(io))
    @test isempty(strip(output))
    @test sol.metadata[:iters] == 2
    @test sol.metadata[:max_it] == 2
end
