using Test
using ThesisProject
using ThesisProject.EulerResiduals: euler_resid_det_grid
using ThesisProject.Determinism: make_rng

@testset "Euler error grids" begin
    cfg_base = deepcopy(SMOKE_CFG)

    # EGM solution (deterministic)
    cfg_egm = cfg_patch(cfg_base, (:solver, :method) => "EGM", (:grids, :Na) => 20)
    cfg_egm = cfg_without(cfg_egm, :shocks)
    model_egm = build_model(cfg_egm)
    method_egm = build_method(cfg_patch(cfg_egm, (:solver, :method) => "EGM"))
    sol_egm = solve(model_egm, method_egm, cfg_egm; rng = make_rng(0))
    p_egm = get_params(model_egm)
    g_egm = get_grids(model_egm)
    sol_egm_c = cfg_get(sol_egm.policy, :c)
    resid_egm = euler_resid_det_grid(p_egm, cfg_get(g_egm, :a).grid, sol_egm_c.value)
    @test maximum(abs.(resid_egm .- sol_egm_c.euler_errors)) < 1e-9
    @test maximum(resid_egm[min(2, end):end]) < 1e-3

    # Projection solution (deterministic)
    cfg_proj = cfg_patch(
        cfg_base,
        (:solver, :method) => "Projection",
        (:solver, :orders) => [5],
        (:solver, :Nval) => 41,
        (:grids, :Na) => 20,
    )
    cfg_proj = cfg_without(cfg_proj, :shocks)
    model_proj = build_model(cfg_proj)
    method_proj = build_method(cfg_patch(cfg_proj, (:solver, :method) => "Projection"))
    sol_proj = solve(model_proj, method_proj, cfg_proj; rng = make_rng(0))
    p_proj = get_params(model_proj)
    g_proj = get_grids(model_proj)
    sol_proj_c = cfg_get(sol_proj.policy, :c)
    resid_proj = euler_resid_det_grid(p_proj, cfg_get(g_proj, :a).grid, sol_proj_c.value)
    @test maximum(abs.(resid_proj .- sol_proj_c.euler_errors)) < 1e-9
    @test maximum(resid_proj[min(2, end):end]) < 5e-3

    # Perturbation solution (deterministic)
    cfg_pert =
        cfg_patch(cfg_base, (:solver, :method) => "Perturbation", (:grids, :Na) => 20)
    cfg_pert = cfg_without(cfg_pert, :shocks)
    model_pert = build_model(cfg_pert)
    method_pert = build_method(cfg_patch(cfg_pert, (:solver, :method) => "Perturbation"))
    sol_pert = solve(model_pert, method_pert, cfg_pert; rng = make_rng(0))
    p_pert = get_params(model_pert)
    g_pert = get_grids(model_pert)
    sol_pert_c = cfg_get(sol_pert.policy, :c)
    resid_pert = euler_resid_det_grid(p_pert, cfg_get(g_pert, :a).grid, sol_pert_c.value)
    @test maximum(abs.(resid_pert .- sol_pert_c.euler_errors)) < 1e-9
    # Perturbation is local; prune boundaries and use a looser tolerance
    @test maximum(resid_pert[2:end-1]) < 5e-2
end
