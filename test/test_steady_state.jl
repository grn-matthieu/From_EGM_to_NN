using Test
using ThesisProject

@testset "Analytic steady state regimes" begin
    cfg = deepcopy(SMOKE_CFG)
    r = cfg[:params][:r]
    # 1) beta*R < 1 -> lower bound
    cfg1 = deepcopy(cfg)
    cfg1[:params]:[β] = 0.90
    m1 = build_model(cfg1)
    ss1 = ThesisProject.steady_state_analytic(m1)
    @test ss1.kind == :lower_bound

    # 2) beta*R ≈ 1 -> interior
    cfg2 = deepcopy(cfg)
    cfg2[:params][:β] = 1.0 / (1.0 + r)
    m2 = build_model(cfg2)
    ss2 = ThesisProject.steady_state_analytic(m2)
    @test ss2.kind == :interior

    # 3) beta*R > 1 -> upper bound (use beta = 1.0)
    cfg3 = deepcopy(cfg)
    cfg3[:params][:β] = 1.0
    m3 = build_model(cfg3)
    ss3 = ThesisProject.steady_state_analytic(m3)
    @test ss3.kind == :upper_bound
end

@testset "Steady state from policy" begin
    cfg = deepcopy(SMOKE_CFG)
    model = build_model(cfg)
    method = build_method(cfg)  # default EGM in smoke config
    sol = ThesisProject.solve(model, method, cfg)
    ss = ThesisProject.steady_state_from_policy(sol)
    @test 1 <= ss.idx <= cfg[:grids][:Na]
    @test isfinite(ss.gap)
end
