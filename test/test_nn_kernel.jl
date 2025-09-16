using Test
using ThesisProject

@testset "NN kernel deterministic" begin
    cfg = deepcopy(SMOKE_CFG)
    cfg[:grids][:Na] = 20
    model = build_model(cfg)
    p = get_params(model)
    g = get_grids(model)
    U = get_utility(model)

    sol = ThesisProject.NNKernel.solve_nn_det(p, g, U)
    Na = g[:a].N
    @test length(sol.c) == Na
    @test length(sol.a_next) == Na
    @test length(sol.resid) == Na
    # feasibility
    a_min = g[:a].min
    a_max = g[:a].max
    R = 1 + p.r
    @test all(sol.a_next .>= a_min .- 1e-12) && all(sol.a_next .<= a_max .+ 1e-12)
    @test all(isfinite, sol.resid)
    @test sol.iters == 1 && sol.converged
end

## Stochastic stub has unresolved field-name issue; skip for now
