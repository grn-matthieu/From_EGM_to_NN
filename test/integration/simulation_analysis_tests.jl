const SteadyState = ThesisProject.SteadyState
const Sim = ThesisProject.SimPanel

@testset "Steady state diagnostics" begin
    cfg = deterministic_config(Na = 20, solver_overrides = (; maxit = 300))
    model, method = build_model_and_method(cfg)
    sol = ThesisProject.solve(model, method, cfg; rng = derive_solver_rng(cfg, :EGM))
    steady = SteadyState.steady_state_analytic(model)
    @test steady.kind in (:lower_bound, :interior, :upper_bound)
    fp = SteadyState.steady_state_from_policy(sol)
    @test fp.gap ≥ 0

    cfg_s = stochastic_config()
    model_s, method_s = build_model_and_method(cfg_s)
    @test_throws ErrorException SteadyState.steady_state_analytic(model_s)
    sol_s =
        ThesisProject.solve(model_s, method_s, cfg_s; rng = derive_solver_rng(cfg_s, :EGM))
    @test_throws ErrorException SteadyState.steady_state_from_policy(sol_s)
end

@testset "Panel simulation" begin
    cfg = deterministic_config(Na = 15)
    model, method = build_model_and_method(cfg)
    panel =
        Sim.simulate_panel(model, method, cfg; N = 10, T = 8, rng = cfg.random.master_rng)
    @test size(panel.assets) == (10, 8)
    @test size(panel.consumption) == (10, 8)
    @test length(panel.seeds) == 10
    @test panel.diagnostics.master_seed ==
          ThesisProject.Determinism.master_seed(cfg.random.master_rng)
end
