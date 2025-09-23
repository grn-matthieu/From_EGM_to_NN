using ThesisProject
using ThesisProject.NNKernel:
    FeatureScaler,
    ScalarParams,
    get_param,
    scalar_params,
    clamp_to_asset_bounds,
    input_dimension,
    normalize_samples!,
    normalize_feature_batch!

@testset "ScalarParams basics" begin
    sp = ScalarParams(2.0, 0.9, 0.01, 3.0)
    @test sp.σ == 2.0
    @test sp.β == 0.9
    @test sp.r == 0.01
    @test sp.y == 3.0
end

@testset "FeatureScaler construction and normalization" begin
    G = Dict(:a => (grid = [0.0, 1.0, 2.0],))

    # case 1: no shocks
    sc1 = FeatureScaler(G, nothing)
    @test sc1.a_min == 0.0f0
    @test sc1.a_range == 2.0f0
    @test sc1.has_shocks == false

    X = Float32[0.0 2.0; 1.0 1.0]  # 2 samples, col1 = a
    normalize_samples!(sc1, X)
    @test all(-1.0f0 .<= X[:, 1] .<= 1.0f0)   # only col1 normalized
    @test X[:, 2] == [2.0f0, 1.0f0]       # col2 unchanged

    Xb = Float32[0.0, 2.0] |> x -> reshape(x, 1, :)
    normalize_feature_batch!(sc1, Xb)
    @test all(-1.0f0 .<= Xb[1, :] .<= 1.0f0)

    # case 2: with shocks
    S = (zgrid = [-1.0f0, 1.0f0],)
    sc2 = FeatureScaler(G, S)
    @test sc2.has_shocks == true
    @test sc2.z_min == -1.0f0
    @test sc2.z_range == 2.0f0

    X2 = Float32[0.0 -1.0; 2.0 1.0]  # col1 = a, col2 = z
    normalize_samples!(sc2, X2)
    @test all(-1.0f0 .<= X2[:, 1] .<= 1.0f0)  # both normalized
    @test all(-1.0f0 .<= X2[:, 2] .<= 1.0f0)

    X2b = Float32[0.0 2.0; -1.0 1.0]  # batch version (rows = features, cols = samples)
    normalize_feature_batch!(sc2, X2b)
    @test all(-1.0f0 .<= X2b[1, :] .<= 1.0f0)
    @test all(-1.0f0 .<= X2b[2, :] .<= 1.0f0)
end



@testset "get_param" begin
    obj = (σ = 2.5, β = nothing)
    @test get_param(obj, :σ, 1.0) == 2.5
    @test get_param(obj, :β, 0.9) == 0.9  # falls back when value === nothing
    @test get_param(obj, :r, 0.02) == 0.02  # property doesn’t exist
end

@testset "scalar_params" begin
    # complete set
    obj = (σ = 2.0, β = 0.9, r = 0.01, y = 3.0)
    sp = scalar_params(obj)
    @test sp == ScalarParams(2.0, 0.9, 0.01, 3.0)

    # missing fields → defaults
    obj2 = (;)
    sp2 = scalar_params(obj2)
    @test sp2 == ScalarParams(1.0, 0.95, 0.02, 1.0)
end

@testset "clamp_to_asset_bounds" begin
    values = [-1.0, 0.5, 2.0]

    # with min/max
    grid_info = (min = 0.0, max = 1.0)
    res = clamp_to_asset_bounds(values, grid_info)
    @test all(0.0 .<= res .<= 1.0)

    # without min/max → unchanged
    res2 = clamp_to_asset_bounds(values, (; other = 123))
    @test res2 == values
end

@testset "input_dimension" begin
    @test input_dimension(nothing) == 1
    @test input_dimension((zgrid = [1.0],)) == 2
end
