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

@testset "EGM warm-start branches + validation flags" begin
    # --- 1) Deterministic steady_state warm-start hits clamp.(...) path ---
    cfg1 = cfg_patch(
        SMOKE_CFG,
        (:solver, :method) => "EGM",
        (:solver, :interp_kind) => "linear",
        (:solver, :warm_start) => :steady_state,
        (:grids, :Na) => 12,
        (:solver, :maxit) => 50,
    )
    model1 = build_model(cfg1)
    method1 = ThesisProject.EGM.build_egm_method(cfg1)
    _ = ThesisProject.EGM.solve(model1, method1, cfg1)  # executes steady_state det init

    # --- 2) Stochastic steady_state warm-start hits Nz loop/matrix return ---
    cfg2 = cfg_patch(
        SMOKE_STOCH_CFG,
        (:solver, :method) => "EGM",
        (:solver, :interp_kind) => "pchip",
        (:solver, :warm_start) => :steady_state,
        (:grids, :Na) => 10,
        (:solver, :maxit) => 40,
    )
    model2 = build_model(cfg2)
    method2 = ThesisProject.EGM.build_egm_method(cfg2)
    _ = ThesisProject.EGM.solve(model2, method2, cfg2)  # executes matrix build loop

    # --- 3) Deterministic custom init vector triggers copy(custom_c_vec) ---
    Na = 9
    custom_vec = fill(0.4, Na)
    cfg3 = cfg_patch(
        SMOKE_CFG,
        (:solver, :method) => "EGM",
        (:solver, :interp_kind) => "linear",
        (:solver, :warm_start) => :custom_any, # not in default set
        (:grids, :Na) => Na,
        (:init, :c) => custom_vec,
        (:solver, :maxit) => 30,
    )
    model3 = build_model(cfg3)
    method3 = ThesisProject.EGM.build_egm_method(cfg3)
    _ = ThesisProject.EGM.solve(model3, method3, cfg3)

    # --- 4) Stochastic custom init matrix triggers copy(custom_c_mat) ---
    cfg4_0 = cfg_patch(SMOKE_STOCH_CFG, (:grids, :Na) => 8)
    Nz = length(get_shocks(build_model(cfg4_0)).zgrid)
    custom_mat = fill(0.3, 8, Nz)
    cfg4 = cfg_patch(
        cfg4_0,
        (:solver, :method) => "EGM",
        (:solver, :interp_kind) => "linear",
        (:solver, :warm_start) => :custom_any, # not in default set
        (:init, :c) => custom_mat,
        (:solver, :maxit) => 25,
    )
    model4 = build_model(cfg4)
    method4 = ThesisProject.EGM.build_egm_method(cfg4)
    _ = ThesisProject.EGM.solve(model4, method4, cfg4)

    # --- 5) Force non-monotone branches with precise overrides and check warning ---
    # First run once to learn the concrete array type used by policies
    cfg5 = cfg_patch(
        SMOKE_CFG,
        (:solver, :method) => "EGM",
        (:solver, :interp_kind) => "linear",
        (:grids, :Na) => 6,
        (:solver, :maxit) => 15,
    )
    model5 = build_model(cfg5)
    method5 = ThesisProject.EGM.build_egm_method(cfg5)
    sol_tmp = ThesisProject.EGM.solve(model5, method5, cfg5)
    T = typeof(sol_tmp.policy[:c].value)  # e.g. Vector{Float64} or SVector{N,Float64}

    # Override with methods MORE SPECIFIC than AbstractArray so dispatch picks ours
    @eval ThesisProject.CommonValidators begin
        is_nondec(x::$T; tol = 1e-8) = false      # force both c and a monotonicity to fail
        is_positive(x::$T) = true               # keep positivity passing
        respects_amin(x::$T, amin) = true       # keep a' ≥ amin passing
    end

    @test_logs (
        :warn,
        r"EGM solution failed monotonicity/positivity checks; marking as invalid",
    ) begin
        sol5 = ThesisProject.EGM.solve(model5, method5, cfg5)
        @test sol5.metadata[:valid] == false
        @test haskey(sol5.metadata, :validation)
        v = sol5.metadata[:validation]
        @test v[:c_monotone_nondec] == false
        @test v[:a_monotone_nondec] == false
        @test v[:c_positive] == true
        @test v[:a_above_min] == true
    end

end
