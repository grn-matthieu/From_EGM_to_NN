using Test
using ThesisProject

@testset "API module basics" begin
    # Wrong-type calls should throw informative errors
    @test_throws ErrorException ThesisProject.API.get_params(42)
    @test_throws ErrorException ThesisProject.API.get_grids(:not_a_model)
    @test_throws ErrorException ThesisProject.API.get_shocks(nothing)
    @test_throws ErrorException ThesisProject.API.get_utility(1.0)
    @test_throws Exception ThesisProject.API.build_model(Dict())
    @test_throws Exception ThesisProject.API.build_method(Dict())
    @test_throws ErrorException ThesisProject.API.load_config(123)
    @test_throws ErrorException ThesisProject.API.validate_config(123)
    @test_throws ErrorException ThesisProject.API.solve(:(bad))

    # Solution parametric type holds user model/method
    struct _DummyModel <: ThesisProject.API.AbstractModel end
    struct _DummyMethod <: ThesisProject.API.AbstractMethod end
    m = _DummyModel()
    k = _DummyMethod()
    sol = ThesisProject.API.Solution(
        policy = Dict{Symbol,Any}(),
        value = nothing,
        diagnostics = (;),
        metadata = Dict{Symbol,Any}(),
        model = m,
        method = k,
    )
    @test sol.model === m
    @test sol.method === k
    @test isa(sol, ThesisProject.API.Solution{_DummyModel,_DummyMethod})
end
