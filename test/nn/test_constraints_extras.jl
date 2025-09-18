using Test
using ThesisProject
using ThesisProject.NNConstraints: project_savings

@testset "NNConstraints: project_savings API" begin
    a_min = 0.5

    # Vector input
    ap_raw = [-1.0, 0.5, 2.0]
    s_soft = project_savings(ap_raw, a_min; kind = :softplus)
    s_clip = project_savings(ap_raw, a_min; kind = :clip)
    @test size(s_soft) == size(ap_raw)
    @test size(s_clip) == size(ap_raw)
    @test minimum(s_soft) >= a_min - 1e-12
    @test minimum(s_clip) >= a_min - 1e-12

    # Matrix input
    M = [-1.0 0.5; 2.0 -0.2]
    Mp_soft = project_savings(M, a_min; kind = :softplus)
    Mp_clip = project_savings(M, a_min; kind = :clip)
    @test size(Mp_soft) == size(M)
    @test size(Mp_clip) == size(M)
    @test minimum(Mp_soft) >= a_min - 1e-12
    @test minimum(Mp_clip) >= a_min - 1e-12

    # Unknown kind should throw
    @test_throws ErrorException project_savings(ap_raw, a_min; kind = :unknown_kind)
end
