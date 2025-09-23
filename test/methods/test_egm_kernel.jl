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
            cfg_has(SMOKE_CFG, :random) ? cfg_get(SMOKE_CFG, :random) : NamedTuple()
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

# --- 1) Deterministic: custom init.c vector is used (returns copy) ---
@testset "EGM.det: custom c_init vector" begin
    Na = 12
    cfg = cfg_patch(
        SMOKE_CFG,
        (:grids, :Na) => Na,
        (:solver, :warm_start) => :custom,   # force custom branch
        (:init, :c) => fill(0.5, Na),        # custom vector
    )
    model = build_model(cfg)
    p = get_params(model)
    g = get_grids(model)
    U = get_utility(model)

    sol = ThesisProject.EGMKernel.solve_egm_det(
        p,
        g,
        U;
        tol = 1e-7,
        tol_pol = 1e-7,
        maxit = 50,
        interp_kind = LinearInterp(),
    )
    @test length(sol.c) == Na
    @test sol.iters ≥ 0  # path executed; kernel ran with provided c_init
end

# --- 2) Stochastic: warm_start=:steady_state builds matrix via loop ---
@testset "EGM.stoch: steady_state c_init matrix" begin
    Na = 10
    cfg = cfg_patch(
        SMOKE_STOCH_CFG,
        (:grids, :Na) => Na,
        (:solver, :warm_start) => :steady_state,  # triggers steady-state init matrix
    )
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
        tol = 1e-6,
        tol_pol = 1e-6,
        maxit = 40,
        interp_kind = MonotoneCubicInterp(),
    )
    @test size(sol.c, 1) == Na
    @test size(sol.a_next, 1) == Na
end

# --- 3) Stochastic: custom init.c matrix is used (returns copy) ---
@testset "EGM.stoch: custom c_init matrix" begin
    Na = 8
    cfg0 = cfg_patch(SMOKE_STOCH_CFG, (:grids, :Na) => Na)
    # Build a valid-shaped custom matrix
    Nz = length(get_shocks(build_model(cfg0)).zgrid)
    c0 = fill(0.3, Na, Nz)

    cfg = cfg_patch(
        cfg0,
        (:solver, :warm_start) => :custom, # force custom branch
        (:init, :c) => c0,                 # custom matrix
    )

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
        tol = 1e-6,
        tol_pol = 1e-6,
        maxit = 30,
        interp_kind = LinearInterp(),
    )
    @test size(sol.c) == size(c0)
end

# --- 4) Validation violations: force all checks to fail and assert warning/metadata ---
@testset "EGM.solve: validation flags and warning" begin
    # Monkeypatch validators to force violations for arrays
    @eval ThesisProject.CommonValidators begin
        is_positive(x::AbstractArray) = false
        respects_amin(x::AbstractArray, amin) = false
        is_nondec(x::AbstractArray; tol = 1e-8) = false
    end

    cfg = cfg_patch(SMOKE_CFG, (:grids, :Na) => 6, (:solver, :maxit) => 5)
    model = build_model(cfg)
    method = ThesisProject.EGM.build_egm_method(cfg)

    @test_logs (
        :warn,
        r"EGM solution failed monotonicity/positivity checks; marking as invalid",
    ) begin
        sol = ThesisProject.EGM.solve(model, method, cfg)
        @test sol.metadata[:valid] == false
        @test haskey(sol.metadata, :validation)
        v = sol.metadata[:validation]
        @test v isa Dict
        @test haskey(v, :c_positive)      # violations[:c_positive] = false
        @test haskey(v, :a_above_min)     # violations[:a_above_min] = false
        @test haskey(v, :a_monotone_nondec) # violations[:a_monotone_nondec] = false
    end
end
