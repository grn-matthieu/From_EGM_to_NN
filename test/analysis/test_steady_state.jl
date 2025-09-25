using Test
using ThesisProject

@testset "Analytic steady state regimes" begin
    cfg_full = deepcopy(SMOKE_CFG)
    # use a deterministic variant (no shocks) for analytic steady-state checks
    cfg = cfg_without(cfg_full, :shocks)
    r = cfg_get(cfg, :params, :r)
    # 1) β*R < 1 -> lower bound
    cfg1 = cfg_patch(cfg, (:params, Symbol("β")) => 0.90)
    m1 = build_model(cfg1)
    ss1 = ThesisProject.steady_state_analytic(m1)
    @test ss1.kind == :lower_bound

    # 2) β*R ≈ 1 -> interior
    cfg2 = cfg_patch(cfg, (:params, Symbol("β")) => 1.0 / (1.0 + r))
    m2 = build_model(cfg2)
    ss2 = ThesisProject.steady_state_analytic(m2)
    @test ss2.kind == :interior

    # 3) β*R > 1 -> upper bound (use β = 1.0)
    cfg3 = cfg_patch(cfg, (:params, Symbol("β")) => 1.0)
    m3 = build_model(cfg3)
    ss3 = ThesisProject.steady_state_analytic(m3)
    @test ss3.kind == :upper_bound

    # 4) Throws if with degenerate shocks (Nz=1 and only shock = 0.0)
    # use the full (possibly stochastic) config as base for shock patches
    cfg4 = cfg_patch(
        cfg_full,
        (:shocks, Symbol("Nz")) => 1,
        (:shocks, Symbol("σ_shock")) => 0.0,
    )
    m4 = build_model(cfg4)
    @test_throws ErrorException ThesisProject.steady_state_analytic(m4)
end

@testset "Steady state from policy" begin
    cfg_full = deepcopy(SMOKE_CFG)
    # build deterministic model/solution for steady state from policy
    cfg = cfg_without(cfg_full, :shocks)
    model = build_model(cfg)
    method = build_method(cfg_patch(cfg, (:solver, :method) => "EGM"))
    sol = ThesisProject.solve(model, method, cfg)
    ss = ThesisProject.steady_state_from_policy(sol)
    @test 1 <= ss.idx <= cfg_get(cfg, :grids, :Na)
    @test isfinite(ss.gap)

    # Throws if with stochastic shocks (active = true and Nz > 1)
    cfg5 = cfg_patch(
        cfg_full,
        (:shocks, Symbol("Nz")) => 2,
        (:shocks, Symbol("active")) => true,
    )
    m5 = build_model(cfg5)
    @test_throws ErrorException ThesisProject.steady_state_analytic(m5)
end
