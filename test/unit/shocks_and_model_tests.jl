const Shocks = ThesisProject.Shocks
const ConsumerSaving = ThesisProject.ConsumerSaving
const ModelFactory = ThesisProject.ModelFactory

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

    cfg_shock = stochastic_config()
    model_shock = ConsumerSaving.build_cs_model(cfg_shock)
    @test model_shock.shocks !== nothing
    @test hasproperty(model_shock.params, :ρ_shock)
    @test hasproperty(model_shock.params, :σ_shock)
end

@testset "Model factory" begin
    cfg = deterministic_config()
    model = ModelFactory.build_model(cfg)
    @test model isa ConsumerSaving.ConsumerSavingModel
    bad_cfg = deterministic_config()
    bad_cfg = deep_merge(bad_cfg, (model = (name = :unknown,),))
    @test_throws ErrorException ModelFactory.build_model(bad_cfg)
end
