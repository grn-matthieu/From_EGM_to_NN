using Test
using ThesisProject

@testset "Coverage - SimPlots" begin
    @test_throws ErrorException plot_policy(1)
    @test_throws ErrorException plot_euler_errors(1)
end
