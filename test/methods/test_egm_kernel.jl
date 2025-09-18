using Test
using ThesisProject
using ThesisProject.CommonInterp: LinearInterp, MonotoneCubicInterp

@testset "EGM kernel deterministic (linear/pchip)" begin
    cfg = deepcopy(SMOKE_CFG)
    # smaller grid keeps runtime down
    cfg[:grids][:Na] = 20
    model = build_model(cfg)
    p = get_params(model)
    g = get_grids(model)
    U = get_utility(model)

    # Linear interpolation
    sol_lin = ThesisProject.EGMKernel.solve_egm_det(
        p,
        g,
        U;
        tol = 1e-6,
        tol_pol = 1e-6,
        maxit = 300,
        interp_kind = LinearInterp(),
    )
    @test length(sol_lin.c) == g[:a].N
    @test length(sol_lin.a_next) == g[:a].N
    @test sol_lin.converged === true
    @test sol_lin.max_resid < 1e-4

    # PCHIP interpolation
    sol_pchip = ThesisProject.EGMKernel.solve_egm_det(
        p,
        g,
        U;
        tol = 1e-6,
        tol_pol = 1e-6,
        maxit = 300,
        interp_kind = MonotoneCubicInterp(),
    )
    @test length(sol_pchip.c) == g[:a].N
    @test length(sol_pchip.a_next) == g[:a].N
    @test sol_pchip.converged === true
    @test sol_pchip.max_resid < 1e-4
end

@testset "EGM kernel stochastic (linear)" begin
    cfg = deepcopy(SMOKE_STOCH_CFG)
    # keep grids modest
    cfg[:grids][:Na] = 16
    model = build_model(cfg)
    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)

    sol = ThesisProject.EGMKernel.solve_egm_stoch(
        p,
        g,
        S,
        U;
        tol = 1e-5,
        tol_pol = 1e-6,
        maxit = 400,
        interp_kind = LinearInterp(),
    )
    Na = g[:a].N
    Nz = length(S.zgrid)
    @test size(sol.c) == (Na, Nz)
    @test size(sol.a_next) == (Na, Nz)
    @test sol.converged === true
    @test sol.max_resid < 1e-3
end
