using Test
using ThesisProject

@testset "projection method adapter" begin
    cfg = load_config(joinpath(@__DIR__, "..", "..", "config", "simple_baseline.yaml"))
    cfg = cfg_patch(cfg, (:solver, :method) => "Projection")
    model = build_model(cfg)
    method = build_method(cfg)
    sol = solve(model, method, cfg)
    grids = get_grids(model)
    @test length(cfg_get(sol.policy, :c).value) == length(cfg_get(grids, :a).grid)
end
