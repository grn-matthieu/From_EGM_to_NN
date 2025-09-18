using Test
using ThesisProject

# Import the relevant APIs from their modules
using ThesisProject.NNConstraints: softplus, project_savings_softplus, project_savings_clip
using ThesisProject.NNLoss: violation, quadratic_penalty

@testset "NN Constraints: projection and penalties" begin
    # Softplus projection
    ap_raw = [-1.0, 0.5, 2.0]
    a_min = 0.0
    ap = project_savings_softplus(ap_raw, a_min)
    @test minimum(ap) ≥ a_min - 1e-12
    @test ap[3] > ap[2]
    @test eltype(ap) === Float64

    # Hard clip projection
    apc = project_savings_clip(ap_raw, a_min)
    @test apc[1] == a_min

    # Shapes preserved for matrices (softplus and clip)
    M = [-1.0 0.5; 2.0 -0.2]
    Mp_soft = project_savings_softplus(M, a_min)
    Mp_clip = project_savings_clip(M, a_min)
    @test size(Mp_soft) == size(M)
    @test size(Mp_clip) == size(M)

    # Violation and penalty
    v = violation([-0.1, 0.2], 0.0)
    @test v ≈ [0.1, 0.0]

    L1 = quadratic_penalty([-0.1, 0.0], 0.0; λ = 1.0, reduction = :sum)
    L2 = quadratic_penalty([-0.2, 0.0], 0.0; λ = 1.0, reduction = :sum)
    @test L2 > L1
end
