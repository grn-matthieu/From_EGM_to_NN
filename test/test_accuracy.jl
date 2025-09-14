using Test
using ThesisProject
using ThesisProject.EulerResiduals: euler_resid_det_2
using ThesisProject.Determinism: make_rng

@testset "Euler error grids" begin
    cfg_path = joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_config.yaml")
    cfg_base = load_config(cfg_path)

    # EGM solution
    cfg_egm = deepcopy(cfg_base)
    cfg_egm[:solver][:method] = "EGM"
    cfg_egm[:grids][:Na] = 20
    model_egm = build_model(cfg_egm)
    method_egm = build_method(cfg_egm)
    sol_egm = solve(model_egm, method_egm, cfg_egm; rng = make_rng(0))
    p_egm = get_params(model_egm)
    g_egm = get_grids(model_egm)
    resid_egm = euler_resid_det_2(p_egm, g_egm[:a].grid, sol_egm.policy[:c].value)
    @test maximum(abs.(resid_egm .- sol_egm.policy[:c].euler_errors)) < 1e-9
    @test maximum(resid_egm[min(2, end):end]) < 1e-3

    # Projection solution
    cfg_proj = deepcopy(cfg_base)
    cfg_proj[:solver][:method] = "Projection"
    cfg_proj[:solver][:orders] = [5]
    cfg_proj[:solver][:Nval] = 41
    cfg_proj[:grids][:Na] = 20
    model_proj = build_model(cfg_proj)
    method_proj = build_method(cfg_proj)
    sol_proj = solve(model_proj, method_proj, cfg_proj; rng = make_rng(0))
    p_proj = get_params(model_proj)
    g_proj = get_grids(model_proj)
    resid_proj = euler_resid_det_2(p_proj, g_proj[:a].grid, sol_proj.policy[:c].value)
    @test maximum(abs.(resid_proj .- sol_proj.policy[:c].euler_errors)) < 1e-9
    @test maximum(resid_proj[min(2, end):end]) < 5e-3
end
