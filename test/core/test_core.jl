# Main testset to assess if the core of the code is stable
@testset "Core stability" begin
    # Test loading side
    config = deepcopy(SMOKE_CFG)
    @test config isa AbstractDict
    @test all(k -> haskey(config, k), (:model, :params, :grids)) # Fundamental keys in the config
    @test begin # Test to validate a config we know to be stable
        try
            validate_config(config)
            true
        catch
            false
        end
    end

    # Test model building
    model = build_model(config)
    @test isa(model, ThesisProject.API.AbstractModel)

    # Test model getters
    params = get_params(model)
    grids = get_grids(model)
    @test params isa NamedTuple
    @test grids isa NamedTuple

    # Test solver building
    method = build_method(config)
    @test isa(method, ThesisProject.API.AbstractMethod)

    # Test solver
    sol = nothing
    @testset "solve" begin # Battery of tests on the solution obtained
        try
            sol = solve(model, method, config)
            @test isa(sol, ThesisProject.API.Solution)
            @test isa(sol.policy, AbstractDict)
            @test haskey(sol.policy, :c) && haskey(sol.policy, :a)
            @test length(sol.policy[:c].value) == grids[:a].N
            @test length(sol.policy[:a].value) == grids[:a].N
            @test sol.metadata[:converged] === true

            # Test the tolerance check (ignore the first point where BC is not binding)
            @test maximum(sol.policy[:c].euler_errors[min(2, end):end]) < 1e-5
            # Policy bounds : does the asset policy respect the constraints?
            @test all(sol.policy[:a].value .>= grids[:a].min)
            @test all(sol.policy[:a].value .<= grids[:a].max)
            # Monotonicity check
            @test is_nondec(sol.policy[:a].value; tol = 1e-8)
        catch err
            @warn "solve failed" err = (err, catch_backtrace())
            @test false
        end
    end
end
