using ThesisProject
using ThesisProject.NNKernel:
    FeatureScaler,
    ScalarParams,
    clamp_to_asset_bounds,
    get_param,
    input_dimension,
    normalize_feature_batch,
    normalize_feature_batch!,
    normalize_samples!,
    scalar_params

@testset "ScalarParams basics" begin
    sp = ScalarParams(2.0, 0.9, 0.01, 3.0)
    @test sp.σ == 2.0
    @test sp.β == 0.9
    @test sp.r == 0.01
    @test sp.y == 3.0
end

@testset "FeatureScaler construction and normalization" begin
    G = Dict(:a => (grid = [0.0, 1.0, 2.0], min = 0.0, max = 2.0))
    P = (r = 0.02, y = log(1.2))

    settings_det = solver_settings(nothing; has_shocks = false)
    sc1 = FeatureScaler(P, G, nothing, settings_det)
    @test sc1.has_shocks == false
    @test sc1.w_min == settings_det.w_min
    @test sc1.w_range > 0
    @test isapprox(sc1.y_min, Float32(exp(P.y)); atol = 1e-6)

    X = Float32[
        exp(P.y) settings_det.w_min
        exp(P.y) settings_det.w_max
    ]
    normalize_samples!(sc1, X)
    @test all(abs.(X[:, 1]) .≤ 1.0f0)
    @test X[:, 2] ≈ Float32[-1.0, 1.0] atol = 1e-5

    Xb = Float32[
        exp(P.y) exp(P.y)
        settings_det.w_min settings_det.w_max
    ]
    normalize_feature_batch!(sc1, Xb)
    @test all(abs.(Xb[1, :]) .≤ 1.0f0)
    @test Xb[2, :] ≈ Float32[-1.0, 1.0] atol = 1e-5

    settings_sto = solver_settings(nothing; has_shocks = true)
    S = (zgrid = Float32.([-0.2, 0.0, 0.2]),)
    sc2 = FeatureScaler(P, G, S, settings_sto)
    @test sc2.has_shocks == true
    @test sc2.y_range > 0

    y_vals = exp.(Float32(P.y) .+ S.zgrid[1:2])
    X2 = Float32[
        y_vals[1] settings_sto.w_min
        y_vals[2] settings_sto.w_max
    ]
    normalize_samples!(sc2, X2)
    @test all(abs.(X2) .≤ 1.0f0)

    X2b = Float32[
        y_vals[1] y_vals[2]
        settings_sto.w_min settings_sto.w_max
    ]
    normalize_feature_batch!(sc2, X2b)
    @test all(abs.(X2b) .≤ 1.0f0)

    X2c = Float32[
        y_vals[1] y_vals[2]
        settings_sto.w_min settings_sto.w_max
    ]
    X2c_norm = normalize_feature_batch(sc2, X2c)
    @test size(X2c_norm) == size(X2c)
    @test all(abs.(X2c_norm) .≤ 1.0f0)
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

    # missing fields → throw when accessed (guard against silent defaults)
    @test_throws Exception scalar_params((;))
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
    @test input_dimension(nothing) == 2
    @test input_dimension((zgrid = [1.0],)) == 2
end
