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
    # build_model checks for :model key and throws an AssertionError
    @test_throws AssertionError ThesisProject.build_model(Dict())
    # build_method expects :solver key and throws a KeyError when missing
    @test_throws KeyError ThesisProject.build_method(Dict())
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
    @test sol.policy[:a] == 1
end

# Exercise viz API stubs via the package exports
@testset "Viz API stubs" begin
    # calling these without Plots should raise the informative error
    @test_throws ErrorException ThesisProject.plot_policy(1)
    @test_throws ErrorException ThesisProject.plot_euler_errors(1)
end
