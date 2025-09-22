using Test
using ThesisProject

@testset "Perturbation kernel deterministic" begin
    cfg = cfg_patch(SMOKE_CFG, (:grids, :Na) => 20)
    model = build_model(cfg)
    p = get_params(model)
    g = get_grids(model)
    U = get_utility(model)

    sol = ThesisProject.PerturbationKernel.solve_perturbation_det(p, g, U; order = 1)
    a_grid = cfg_get(g, :a)
    @test length(sol.c) == a_grid.N
    @test length(sol.a_next) == a_grid.N
    # residuals exist and are finite
    @test all(isfinite, sol.resid)
end

@testset "Perturbation kernel stochastic" begin
    cfg = cfg_patch(SMOKE_STOCH_CFG, (:grids, :Na) => 16)
    model = build_model(cfg)
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    sol = ThesisProject.PerturbationKernel.solve_perturbation_stoch(p, g, S, U; order = 1)
    Na = cfg_get(g, :a).N
    Nz = length(S.zgrid)
    @test size(sol.c) == (Na, Nz)
    @test size(sol.a_next) == (Na, Nz)
    @test all(isfinite, sol.resid)
end
