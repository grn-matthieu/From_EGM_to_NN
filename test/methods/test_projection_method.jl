using Test
using ThesisProject

@testset "projection method adapter" begin
    cfg = cfg_patch(SMOKE_CFG, (:solver, :method) => "Projection")
    # ensure deterministic shape: remove shocks so policy arrays are vectors
    det_cfg = cfg_without(cfg, :shocks)
    model = build_model(det_cfg)
    method = build_method(det_cfg)
    sol = solve(model, method, det_cfg)
    grids = get_grids(model)
    @test length(cfg_get(sol.policy, :c).value) == length(cfg_get(grids, :a).grid)
end
