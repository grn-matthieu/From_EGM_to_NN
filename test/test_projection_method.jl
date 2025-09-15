using Test
using ThesisProject

@testset "projection method adapter" begin
    cfg = load_config(joinpath(@__DIR__, "..", "config", "simple_baseline.yaml"))
    cfg[:solver][:method] = "Projection"
    model = build_model(cfg)
    method = build_method(cfg)
    sol = solve(model, method, cfg)
    grids = get_grids(model)
    @test length(sol.policy[:c].value) == length(grids[:a].grid)
end
