using Test
using ThesisProject
using ThesisProject.CommonInterp: LinearInterp, MonotoneCubicInterp

@testset "EGM kernel deterministic (linear/pchip)" begin
    cfg = cfg_patch(SMOKE_CFG, (:grids, :Na) => 20)
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
    a_grid = cfg_get(g, :a)
    @test length(sol_lin.c) == a_grid.N
    @test length(sol_lin.a_next) == a_grid.N
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
    @test length(sol_pchip.c) == a_grid.N
    @test length(sol_pchip.a_next) == a_grid.N
    @test sol_pchip.converged === true
    @test sol_pchip.max_resid < 1e-4
end

@testset "EGM kernel stochastic (linear)" begin
    cfg = cfg_patch(SMOKE_STOCH_CFG, (:grids, :Na) => 16)
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
    Na = cfg_get(g, :a).N
    Nz = length(S.zgrid)
    @test size(sol.c) == (Na, Nz)
    @test size(sol.a_next) == (Na, Nz)
    @test sol.converged === true
    @test sol.max_resid < 1e-3
end

@testset "EGM policy monotonicity" begin
    for seed in (42, 99)
        random_cfg =
            cfg_has(SMOKE_CFG, :random) ? cfg_get(SMOKE_CFG, :random) : Dict{Symbol,Any}()
        cfg = cfg_patch(
            SMOKE_CFG,
            :random => cfg_patch(random_cfg, :seed => seed),
            (:grids, :Na) => 40,
        )
        model = build_model(cfg)
        params = get_params(model)
        grids = get_grids(model)
        utility = get_utility(model)

        sol = ThesisProject.EGMKernel.solve_egm_det(
            params,
            grids,
            utility;
            tol = 1e-6,
            tol_pol = 1e-6,
            maxit = 400,
            interp_kind = LinearInterp(),
        )

        @test all(diff(sol.a_next) .>= -1e-8)
        @test all(diff(sol.c) .>= -1e-8)
    end
end
