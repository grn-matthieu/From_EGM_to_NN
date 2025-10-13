const API = ThesisProject
const CSVarUtils = ThesisProject.CSVarUtils

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

@testset "EGM dim-1 CSVAR matches scalar baseline" begin
    base_cfg = deterministic_config(method = "EGM", solver_overrides = (; maxit = 200))
    base_model, base_method = build_model_and_method(base_cfg)
    base_rng = derive_solver_rng(base_cfg, "EGM")
    base_sol = API.solve(base_model, base_method, base_cfg; rng = base_rng)

    y_val = base_cfg.params.y
    cs_params = (
        β = base_cfg.params.β,
        γ = base_cfg.params.γ,
        r = base_cfg.params.r,
        y = [y_val],
        A = [[1.0]],
        Σ = [[0.0]],
    )
    csvar_cfg = deep_merge(base_cfg, (model = (name = :cs_vec,), params = cs_params))
    cs_model, cs_method = build_model_and_method(csvar_cfg)
    cs_rng = derive_solver_rng(csvar_cfg, "EGM")
    cs_sol = API.solve(cs_model, cs_method, csvar_cfg; rng = cs_rng)

    c_base = base_sol.policy[:c][:value]
    c_cs = cs_sol.policy[:c][:value]
    a_base = base_sol.policy[:a][:value]
    a_cs = cs_sol.policy[:a][:value]
    @test c_cs ≈ c_base atol = 1e-8 rtol = 1e-8
    @test a_cs ≈ a_base atol = 1e-8 rtol = 1e-8
    @test cs_sol.metadata[:converged] == base_sol.metadata[:converged]
    @test cs_sol.metadata[:max_resid] ≈ base_sol.metadata[:max_resid] atol = 1e-10
end

@testset "CS vs CSVAR policy tensor shapes" begin
    base_cfg =
        deterministic_config(method = "EGM", Na = 6, solver_overrides = (; maxit = 200))
    base_model, base_method = build_model_and_method(base_cfg)
    base_rng = derive_solver_rng(base_cfg, "EGM")
    base_sol = API.solve(base_model, base_method, base_cfg; rng = base_rng)
    base_grids = API.get_grids(base_model)

    y_val = base_cfg.params.y
    cs_params = (
        β = base_cfg.params.β,
        γ = base_cfg.params.γ,
        r = base_cfg.params.r,
        y = fill(y_val, 2),
        A = [[1.0, 0.0], [0.0, 1.0]],
        Σ = [[0.0, 0.0], [0.0, 0.0]],
    )

    csvar_cfg = deep_merge(base_cfg, (model = (name = :cs_vec,), params = cs_params))
    cs_model, cs_method = build_model_and_method(csvar_cfg)
    cs_rng = derive_solver_rng(csvar_cfg, "EGM")
    cs_sol = API.solve(cs_model, cs_method, csvar_cfg; rng = cs_rng)
    cs_grids = API.get_grids(cs_model)

    base_policy_c = base_sol.policy[:c]
    @test base_policy_c[:tensor_shape] == (base_cfg.grids.Na,)
    base_tensor = CSVarUtils.csvar_tensorise(base_policy_c[:value], base_grids[:a])
    @test size(base_tensor) == (base_cfg.grids.Na,)
    @test CSVarUtils.csvar_vectorise(base_tensor, base_grids[:a]) ≈ base_policy_c[:value]

    cs_policy_c = cs_sol.policy[:c]
    tensor_shape_expected = cs_grids[:a].tensor_shape
    @test cs_policy_c[:tensor_shape] == tensor_shape_expected
    y_states = CSVarUtils.csvar_state_matrix(cs_model.params.y, cs_model.params.y_dim)
    Ny = size(y_states, 2)
    cs_tensor = CSVarUtils.csvar_tensorise(cs_policy_c[:value], cs_grids[:a])
    @test size(cs_tensor) == (tensor_shape_expected..., Ny)
    @test CSVarUtils.csvar_vectorise(cs_tensor, cs_grids[:a]) ≈ cs_policy_c[:value]

    joint_tensor = CSVarUtils.csvar_joint_state_tensor(y_states, cs_grids[:a])
    @test size(joint_tensor) == (cs_model.params.y_dim + 1, tensor_shape_expected..., Ny)
    joint_matrix = CSVarUtils.csvar_joint_state_matrix(y_states, cs_grids[:a])
    @test size(joint_matrix) == (cs_model.params.y_dim + 1, cs_grids[:a].N * Ny)
    @test joint_matrix ≈ reshape(joint_tensor, cs_model.params.y_dim + 1, :)
end
