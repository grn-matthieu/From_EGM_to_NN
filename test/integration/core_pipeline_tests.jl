const API = ThesisProject

@testset "Method factories" begin
    for m in ("EGM", "Projection", "Perturbation", "TimeIteration", "NN")
        cfg = deterministic_config(method = m)
        method_obj = API.build_method(cfg)
        @test method_obj isa API.AbstractMethod
    end
end

function _solve_and_check(method_sym; cfg_kwargs = NamedTuple())
    cfg = deterministic_config(;
        method = method_sym,
        Na = 15,
        solver_overrides = (; maxit = 200),
        cfg_kwargs...,
    )
    model, method = build_model_and_method(cfg)
    rng = derive_solver_rng(cfg, method_sym)
    sol = API.solve(model, method, cfg; rng = rng)
    @test sol isa API.Solution
    @test haskey(sol.policy, :c)
    policy_c = sol.policy[:c][:value]
    @test length(policy_c) > 0
    return sol
end

@testset "Solver pipelines" begin
    for method_sym in ("EGM", "Projection", "Perturbation", "TimeIteration")
        _solve_and_check(method_sym)
    end
end

@testset "Multi-method orchestration" begin
    cfg = deterministic_config()
    cfg = deep_merge(cfg, (solver = (method = ["EGM", "Projection"],),))
    model = API.build_model(cfg)
    sols = API.solve(model, cfg; rng = cfg.random.master_rng)
    @test length(sols) == 2
    sols_nt = API.solve(cfg; rng = cfg.random.master_rng)
    @test length(sols_nt) == 2
end

@testset "Config file loading" begin
    cfg_path = joinpath(@__DIR__, "..", "..", "config", "smoke_cfg_det.yaml")
    cfg = API.load_config(cfg_path)
    @test cfg.model.name == "cs"
    sol = API.solve(cfg; rng = cfg.random.master_rng)
    @test sol isa API.Solution
    override = (solver = (method = "EGM",), random = (seed = 9876,))
    sols = API.solve(cfg_path; rng = master_rng(9876), opts = override)
    @test sols isa API.Solution || sols isa Vector{API.Solution}
end
