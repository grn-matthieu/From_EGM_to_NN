# Main testset to assess if the core of the code is stable
@testset "Core stability" begin
    # Test loading side
    config = deepcopy(SMOKE_CFG)
    @test config isa NamedTuple || config isa AbstractDict
    @test all(k -> cfg_has(config, k), (:model, :params, :grids)) # Fundamental keys in the config
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
    @test grids isa NamedTuple || grids isa AbstractDict

    # Test solver building
    method = build_method(config)
    @test isa(method, ThesisProject.API.AbstractMethod)

    # Test solver
    sol = nothing
    @testset "solve" begin # Battery of tests on the solution obtained
        try
            sol = solve(model, method, config)
            @test isa(sol, ThesisProject.API.Solution)
            @test sol.policy isa NamedTuple || sol.policy isa AbstractDict
            @test cfg_has(sol.policy, :c) && cfg_has(sol.policy, :a)
            a_grid = cfg_get(grids, :a)
            @test length(cfg_get(sol.policy, :c).value) == a_grid.N
            @test length(cfg_get(sol.policy, :a).value) == a_grid.N
            @test sol.metadata[:converged] === true

            # Test the tolerance check (ignore the first point where BC is not binding)
            @test maximum(cfg_get(sol.policy, :c).euler_errors[min(2, end):end]) < 1e-5
            # Policy bounds : does the asset policy respect the constraints?
            @test all(cfg_get(sol.policy, :a).value .>= a_grid.min)
            @test all(cfg_get(sol.policy, :a).value .<= a_grid.max)
            # Monotonicity check
            @test is_nondec(cfg_get(sol.policy, :a).value; tol = 1e-8)
        catch err
            @warn "solve failed" err = (err, catch_backtrace())
            @test false
        end
    end
end
