# Main testset to assess if the core of the code is stable
@testset "Core stability" begin
    cfg_path = joinpath("@__DIR__", "..", "config", "smoke_config", "smoke_config.yaml")
    @test isfile(cfg_path) || @warn("config file not found: $cfg_path")

    # Test loading side
    config = load_config(cfg_path)
    @test config isa AbstractDict
    @test all(k -> haskey(config, k), (:model, :params, :grids)) # Fundamental keys in the config

    # Test model building
    model = build_model(config)
    @test isa(model, ThesisProject.API.AbstractModel)

    # Test model getters
    params = get_params(model)
    grids  = get_grids(model)
    @test params isa NamedTuple
    @test (grids isa AbstractDict)

    # Test solver building
    method = build_method(config)
    @test isa(method, ThesisProject.API.AbstractMethod)

    # Test solver
    sol = nothing
    @testset "solve" begin
        try
            sol = solve(model, method, config)
            @test isa(sol, ThesisProject.API.Solution)
            @test isa(sol.policy, AbstractDict)
        catch err
            @warn "solve failed" err=(err, catch_backtrace())
            @test false
        end
    end
end