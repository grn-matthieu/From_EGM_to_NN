using Test
using ThesisProject
using ThesisProject.API

@testset "NNLoss: residuals det/stoch + finiteness" begin
    # Deterministic model via config
    cfg_det = load_config("config/simple_baseline.yaml")
    @test validate_config(cfg_det)
    model_det = ThesisProject.ModelFactory.build_model(cfg_det)
    g = API.get_grids(model_det)[:a]
    a_grid = g.grid

    # Case A: no interpolation (policy grid == batch)
    cA = fill(0.9, length(a_grid))
    policyA = Dict(:c => (; value = cA, grid = a_grid))
    @test ThesisProject.NNLoss.check_finite_residuals(model_det, policyA, a_grid)

    # Case B: with interpolation (policy grid != batch)
    a_batch = range(first(a_grid), last(a_grid); length = length(a_grid) - 1) |> collect
    cB = @. 0.8 + 0.001 * (a_grid - first(a_grid))
    policyB = Dict(:c => (; value = cB, grid = a_grid))
    @test ThesisProject.NNLoss.check_finite_residuals(model_det, policyB, a_batch)

    # Stochastic case: require batch = (a_grid, z_grid, Pz)
    cfg_st = load_config("config/simple_stochastic.yaml")
    @test validate_config(cfg_st)
    model_st = ThesisProject.ModelFactory.build_model(cfg_st)
    shocks = API.get_shocks(model_st)
    ag = API.get_grids(model_st)[:a].grid
    zg = shocks.grid
    Pz = shocks.P
    Na, Nz = length(ag), length(zg)
    cmat = fill(0.9, Na, Nz)
    policyS = Dict(:c => (; value = cmat))
    batchS = (; a_grid = ag, z_grid = zg, Pz = Pz)
    @test ThesisProject.NNLoss.check_finite_residuals(model_st, policyS, batchS)
end
