using Test
using ThesisProject

# Exercise contract stubs via the nested ModelContract module
struct DummyModel <: ThesisProject.AbstractModel end

@testset "Model contract stubs" begin
    m = DummyModel()
    # Call functions directly from the nested module to exercise src/core/model_contract.jl
    @test_throws ErrorException ThesisProject.ModelContract.get_params(m)
    @test_throws ErrorException ThesisProject.ModelContract.get_grids(m)
    @test_throws ErrorException ThesisProject.ModelContract.get_shocks(m)
    @test_throws ErrorException ThesisProject.ModelContract.get_utility(m)
end

# Exercise API factory/error stubs in src/core/api.jl
@testset "API factory stubs" begin
    for bad_cfg in (Dict{Symbol,Any}(), NamedTuple())
        @test_throws Exception ThesisProject.build_model(bad_cfg)
        @test_throws Exception ThesisProject.build_method(bad_cfg)
    end
    @test_throws ErrorException ThesisProject.load_config(1)
    @test_throws ErrorException ThesisProject.validate_config(1)
    @test_throws ErrorException ThesisProject.solve(1)

    # Construct a Solution to exercise the struct definition and fields
    struct DummyMethod <: ThesisProject.AbstractMethod end
    sol = ThesisProject.Solution(;
        policy = Dict{Symbol,Any}(:a => 1),
        value = nothing,
        diagnostics = (ee = (),),
        metadata = Dict{Symbol,Any}(),
        model = DummyModel(),
        method = DummyMethod(),
    )
    @test isa(sol, ThesisProject.Solution)
    @test cfg_get(sol.policy, :a) == 1
end

# Exercise viz API stubs via the package exports
@testset "Viz API stubs" begin
    # calling these without Plots should raise the informative error
    @test_throws ErrorException ThesisProject.plot_policy(1)
    @test_throws ErrorException ThesisProject.plot_euler_errors(1)
end
