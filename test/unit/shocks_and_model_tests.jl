const Shocks = ThesisProject.Shocks
const ConsumerSaving = ThesisProject.ConsumerSaving
const ModelFactory = ThesisProject.ModelFactory
const ConsumerSavingVAR = ThesisProject.ConsumerSavingVAR

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
        expected_placeholder = method_name != :EGM
        @test get(sol.metadata, :placeholder, false) == expected_placeholder
        @test haskey(sol.metadata, :valid)
        @test sol.metadata[:valid] isa Bool
    end
end

@testset "Model factory" begin
    cfg = deterministic_config()
    model = ModelFactory.build_model(cfg)
    @test model isa ConsumerSaving.ConsumerSavingModel
    bad_cfg = deterministic_config()
    bad_cfg = deep_merge(bad_cfg, (model = (name = :unknown,),))
    @test_throws ErrorException ModelFactory.build_model(bad_cfg)
end
