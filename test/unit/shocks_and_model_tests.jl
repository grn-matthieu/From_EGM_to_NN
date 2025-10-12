const Shocks = ThesisProject.Shocks
const ConsumerSaving = ThesisProject.ConsumerSaving
const ModelFactory = ThesisProject.ModelFactory
const ConsumerSavingVAR = ThesisProject.ConsumerSavingVAR
const UtilsConfig = ThesisProject.UtilsConfig

@testset "Shock discretization" begin
    shocks =
        (active = true, method = "tauchen", ρ_shock = 0.8, σ_shock = 0.1, Nz = 5, m = 2.0)
    out = Shocks.discretize(shocks)
    @test length(out.zgrid) == 5
    @test size(out.Π) == (5, 5)
    @test isapprox(sum(out.π), 1.0; atol = 1e-8)
    @test all(out.π .>= 0)

    degenerate = (active = true, method = "tauchen", ρ_shock = 0.0, σ_shock = 0.0, Nz = 1)
    out_deg = Shocks.discretize(degenerate)
    @test out_deg.zgrid == [0.0]
    @test out_deg.Π == reshape([1.0], 1, 1)

    @test_throws ErrorException Shocks.discretize((
        active = true,
        method = "unknown",
        ρ_shock = 0.0,
        σ_shock = 0.1,
        Nz = 3,
    ))
end

@testset "Consumer saving model" begin
    cfg = deterministic_config()
    model = ConsumerSaving.build_cs_model(cfg)
    @test model isa ConsumerSaving.ConsumerSavingModel
    @test model.shocks === nothing
    @test length(model.grids[:a].grid) == cfg.grids.Na
    @test hasproperty(model.params, :γ)

    cfg_shock = stochastic_config()
    model_shock = ConsumerSaving.build_cs_model(cfg_shock)
    @test model_shock.shocks !== nothing
    @test hasproperty(model_shock.params, :ρ_shock)
    @test hasproperty(model_shock.params, :σ_shock)
    @test hasproperty(model_shock.params, :γ)
end

@testset "Consumer saving vector-income model" begin
    cfg = deterministic_config()
    vec_params = (
        model = (name = :cs_vec,),
        params = (
            y = [1.0, 1.1],
            A = [[0.9, 0.1], [0.05, 0.95]],
            Σ = [[0.1, 0.0], [0.0, 0.1]],
        ),
    )
    cfg_vec = deep_merge(cfg, vec_params)

    model_vec = ConsumerSavingVAR.build_cs_var_model(cfg_vec)
    @test model_vec isa ConsumerSavingVAR.ConsumerSavingVARModel
    @test model_vec.params.y_dim == 2
    @test size(model_vec.params.A) == (2, 2)
    @test model_vec.shocks.ε_dim == 2
    @test hasproperty(model_vec.params, :γ)

    model_factory_vec = ModelFactory.build_model(cfg_vec)
    @test model_factory_vec isa ConsumerSavingVAR.ConsumerSavingVARModel

    for method_name in (:EGM, :TimeIteration, :Perturbation)
        cfg_method = deep_merge(cfg_vec, (solver = (method = method_name,),))
        model_obj = ThesisProject.build_model(cfg_method)
        method_obj = ThesisProject.build_method(cfg_method)
        sol = ThesisProject.solve(model_obj, method_obj, cfg_method)
        @test sol isa ThesisProject.Solution
        expected_placeholder = method_name == :TimeIteration
        @test get(sol.metadata, :placeholder, false) == expected_placeholder
        @test haskey(sol.metadata, :valid)
        @test sol.metadata[:valid] isa Bool
    end
end

@testset "Consumer saving VAR config normalization" begin
    base = deterministic_config()
    ascii_cfg = (
        model = (name = :cs_vec,),
        params = (
            beta = 0.95,
            gamma = 2.0,
            interest_rate = 0.02,
            income = [0.6, 0.4],
            transition = [[0.8, 0.2], [0.1, 0.9]],
            Sigma = [[0.05, 0.01], [0.01, 0.04]],
        ),
        grids = base.grids,
        utility = base.utility,
        solver = base.solver,
        random = base.random,
    )

    cfg_ascii_validated = UtilsConfig.validate_config(ascii_cfg)
    @test cfg_ascii_validated.params.β ≈ 0.95
    @test !hasproperty(cfg_ascii_validated.params, :beta)
    @test cfg_ascii_validated.params.γ ≈ 2.0
    @test !hasproperty(cfg_ascii_validated.params, :gamma)
    @test cfg_ascii_validated.params.r ≈ 0.02
    @test !hasproperty(cfg_ascii_validated.params, :interest_rate)
    @test cfg_ascii_validated.params.y == [0.6, 0.4]
    @test !hasproperty(cfg_ascii_validated.params, :income)
    @test length(cfg_ascii_validated.params.A) == 2
    @test length(cfg_ascii_validated.params.A[1]) == 2
    @test !hasproperty(cfg_ascii_validated.params, :transition)
    @test length(cfg_ascii_validated.params.Σ) == 2
    @test length(cfg_ascii_validated.params.Σ[1]) == 2
    @test !hasproperty(cfg_ascii_validated.params, :Sigma)

    model_ascii = ConsumerSavingVAR.build_cs_var_model(cfg_ascii_validated)
    params_ascii = ConsumerSavingVAR.get_params(model_ascii)
    @test params_ascii.y_dim == 2
    @test params_ascii.y ≈ Float32[0.6, 0.4]
    @test size(params_ascii.A) == (2, 2)
    @test eltype(params_ascii.A) == Float32
    @test size(params_ascii.Σ) == (2, 2)
    @test eltype(params_ascii.Σ) == Float32

    scalar_cfg = (
        model = (name = :cs_vec,),
        params = (β = 0.96, γ = 3.0, r = 0.01, y = 1.1, A = 0.9, Σ = 0.2),
        grids = base.grids,
        utility = base.utility,
        solver = base.solver,
        random = base.random,
    )

    cfg_scalar_validated = UtilsConfig.validate_config(scalar_cfg)
    @test cfg_scalar_validated.params.y == [1.1]
    @test cfg_scalar_validated.params.A == [[0.9]]
    @test cfg_scalar_validated.params.Σ == [[0.2]]

    model_scalar = ConsumerSavingVAR.build_cs_var_model(cfg_scalar_validated)
    params_scalar = ConsumerSavingVAR.get_params(model_scalar)
    @test params_scalar.y_dim == 1
    @test params_scalar.y ≈ Float32[1.1]
    @test params_scalar.A[1, 1] ≈ 0.9f0
    @test params_scalar.Σ[1, 1] ≈ 0.2f0
end

@testset "Model factory" begin
    cfg = deterministic_config()
    model = ModelFactory.build_model(cfg)
    @test model isa ConsumerSaving.ConsumerSavingModel
    bad_cfg = deterministic_config()
    bad_cfg = deep_merge(bad_cfg, (model = (name = :unknown,),))
    @test_throws ErrorException ModelFactory.build_model(bad_cfg)
end
