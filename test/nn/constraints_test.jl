using Test
using ThesisProject
using ThesisProject.NNConstraints:
    softplus, project_savings_softplus, project_savings_clip, project_savings
const NNKernel = ThesisProject.NNKernel

@testset "NNConstraints: softplus projection" begin
    # Scalar cases
    for _ = 1:5
        ap_raw = randn()
        a_min = rand() - 0.5  # allow negative a_min
        ap = project_savings_softplus(ap_raw, a_min)
        @test ap ≥ a_min - 1e-12
        @test ap isa Float64
    end

    # Vector broadcast
    ap_raw_vec = randn(50)
    a_min_vec = rand() - 0.5
    ap_vec = project_savings_softplus(ap_raw_vec, a_min_vec)
    @test minimum(ap_vec) ≥ a_min_vec - 1e-12
    @test eltype(ap_vec) == eltype(ap_raw_vec)

    # Matrix broadcast
    ap_raw_mat = randn(10, 7)
    a_min_mat = rand() - 0.5
    ap_mat = project_savings_softplus(ap_raw_mat, a_min_mat)
    @test minimum(ap_mat) ≥ a_min_mat - 1e-12
    @test eltype(ap_mat) == eltype(ap_raw_mat)

    # Type preservation for Float32
    ap32 = Float32.(randn(8))
    a_min32 = Float32(0.2)
    ap32_proj = project_savings_softplus(ap32, a_min32)
    @test eltype(ap32_proj) === Float32
    @test minimum(ap32_proj) ≥ a_min32 - 1.0f-6

    # Softplus basic properties
    @test softplus(-10.0) > 0
    @test softplus(20.0) ≈ 20.0 atol = 1e-8  # ~identity for large positive
end

@testset "NNConstraints: clip and unified API" begin
    # Scalar clip: ap_raw < a_min -> equals a_min
    for _ = 1:10
        a_min = rand() - 0.2
        ap_raw = a_min - (0.1 + rand())
        ap = project_savings_clip(ap_raw, a_min)
        @test ap == a_min
    end

    # Vector clip: all entries below a_min map to a_min
    for _ = 1:3
        a_min = rand() - 0.3
        ap_raw = a_min .- (0.1 .+ rand(length(50)))
        ap = project_savings_clip(ap_raw, a_min)
        @test all(ap .== a_min)
    end

    # Unified API: identical shapes for :softplus and :clip
    ap_vec = randn(37)
    a_min = rand() - 0.5
    ap_soft = project_savings(ap_vec, a_min; kind = :softplus)
    ap_clip = project_savings(ap_vec, a_min; kind = :clip)
    @test size(ap_soft) == size(ap_clip)

    ap_mat = randn(12, 5)
    ap_soft_m = project_savings(ap_mat, a_min; kind = :softplus)
    ap_clip_m = project_savings(ap_mat, a_min; kind = :clip)
    @test size(ap_soft_m) == size(ap_clip_m)
end


@testset "NN Feasibility metric (deterministic kernel)" begin
    # Build a tiny deterministic model for a quick run
    cfg = deepcopy(SMOKE_CFG)
    cfg[:grids][:Na] = 16
    model = build_model(cfg)
    p = get_params(model)
    g = get_grids(model)
    U = get_utility(model)

    # Check feasibility metric for both projection kinds
    for kind in (:softplus, :clip)
        sol = NNKernel.solve_nn_det(p, g, U, cfg; projection_kind = kind)
        @test haskey(sol.opts, :feasibility)
        @test sol.opts.feasibility >= 0.99
    end
end
