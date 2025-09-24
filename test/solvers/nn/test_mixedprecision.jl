using ThesisProject

include("../../../src/solvers/nn/mixed_precision.jl")

@testset "Mixed precision helpers" begin
    # -- float32 conversions
    v = [1, 2, 3]
    m = [1 2; 3 4]
    @test float32_vector(v) isa Vector{Float32}
    @test float32_vector(v) == Float32[1, 2, 3]

    @test float32_matrix(m) isa Matrix{Float32}
    @test float32_matrix(m) == Float32[1 2; 3 4]

    @test float32_loss(3) === 3.0f0

    # -- prepare_training_batch
    X = [1 2 3; 4 5 6]        # 2×3
    Xprep = prepare_training_batch(X)  # should permute dims and cast
    @test Xprep isa Matrix{Float32}
    @test size(Xprep) == (3, 2)
    @test Xprep[1, :] == Float32[1, 4]

    # extract_consumption handles NamedTuple and Tuple outputs (dual-head model)
    named_pred = (Φ = Float32[0.2 0.3], h = Float32[1.0 1.1])
    @test extract_consumption(named_pred) === named_pred.Φ
    tuple_pred = (named_pred, :state)
    @test extract_consumption(tuple_pred) === named_pred.Φ
    plain_pred = Float32[0.4, 0.5]
    @test extract_consumption(plain_pred) === plain_pred

    # -- det_residual_inputs
    G = Dict(:a => (grid = [0.0, 1.0, 2.0],))
    c_pred = [10.0 20.0 30.0]   # row vector
    a_f32, c_vec, c_vec2 = det_residual_inputs(c_pred, G)
    @test a_f32 == Float32[0, 1, 2]
    @test c_vec == Float32[10, 20, 30]
    @test c_vec2 == c_vec

    # det_loss
    resid = [1.0, -2.0, 3.0]
    @test det_loss(resid) == Float32(sum(resid))

    # -- stoch_residual_inputs
    S = (zgrid = [-1.0, 1.0], Π = [0.7 0.3; 0.4 0.6])
    c_pred2 = [
        10.0 11.0
        20.0 21.0
        30.0 31.0
    ]  # 3×2 matches Na=3, Nz=2

    a_f32, z_f32, P_f32, c_mat, c_mat2 = stoch_residual_inputs(c_pred2, G, S)
    @test a_f32 == Float32[0, 1, 2]
    @test z_f32 == Float32[-1, 1]
    @test P_f32 == Float32.(S.Π)
    @test size(c_mat) == (3, 2)
    @test c_mat2 == Float32.(c_mat)

    # when only Na entries are returned tile across shocks
    c_assets_only = Float32[10, 20, 30]
    _, _, _, c_tiled, _ = stoch_residual_inputs(c_assets_only, G, S)
    @test c_tiled == Float32[10 10; 20 20; 30 30]

    # when only Nz entries are returned tile across assets
    c_shock_only = Float32[5, 7]
    _, _, _, c_tiled2, _ = stoch_residual_inputs(c_shock_only, G, S)
    @test c_tiled2 == Float32[5 7; 5 7; 5 7]

    # stoch_loss
    resid2 = [-1.0, 2.0, -2.0]
    @test stoch_loss(resid2) == Float32(sum(abs2, resid2))

    # -- det_forward_inputs
    Xf, a_f32b = det_forward_inputs(G)
    @test size(Xf) == (1, 3)
    @test Xf[1, :] == a_f32b
    @test a_f32b == Float32[0, 1, 2]

    # -- convert_to_grid_eltype
    grid = [0.0, 1.0, 2.0]           # Float64 grid
    vals = Float32[0.1, 0.2, 0.3]
    converted = convert_to_grid_eltype(grid, vals)
    @test eltype(converted) == Float64
    @test isapprox(converted, [0.1, 0.2, 0.3]; atol = 1e-6, rtol = 1e-6)

    # Also check works if grid is Int
    grid_int = [1, 2, 3]
    vals2 = Float32[1.2, 2.8, 3.6]
    converted2 = convert_to_grid_eltype(grid_int, vals2)
    @test eltype(converted2) == Int
    @test converted2 == [1, 2, 3]  # conversion floors values to ints
end
