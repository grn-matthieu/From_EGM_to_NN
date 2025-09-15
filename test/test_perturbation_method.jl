using Test
using ThesisProject
using ThesisProject.EulerResiduals: euler_resid_det_2, euler_resid_stoch

@testset "perturbation method adapter (deterministic)" begin
    cfg = deepcopy(SMOKE_CFG)
    cfg[:solver][:method] = "Perturbation"
    # keep grid moderate to avoid edge dominance in local linearization
    cfg[:grids][:Na] = 20
    model = build_model(cfg)
    method = build_method(cfg)
    sol = solve(model, method, cfg)

    g = get_grids(model)
    p = get_params(model)

    # Shape checks
    @test length(sol.policy[:c].value) == length(g[:a].grid)
    @test length(sol.policy[:a].value) == length(g[:a].grid)

    # Residual consistency and metadata
    resid = euler_resid_det_2(p, g[:a].grid, sol.policy[:c].value)
    @test maximum(abs.(resid .- sol.policy[:c].euler_errors)) < 1e-10

    Na = g[:a].N
    lo = Na > 2 ? 2 : 1
    hi = Na > 2 ? Na - 1 : Na
    expected_max = maximum(resid[lo:hi])
    @test sol.metadata[:max_resid] == expected_max

    # Basic validity and options
    @test sol.metadata[:order] == 1
    @test haskey(sol.metadata, :fit_ok)
    @test sol.metadata[:valid] == true
end

@testset "perturbation method adapter (stochastic)" begin
    cfg = deepcopy(SMOKE_STOCH_CFG)
    cfg[:solver][:method] = "Perturbation"
    model = build_model(cfg)
    method = build_method(cfg)
    sol = solve(model, method, cfg)

    g = get_grids(model)
    S = get_shocks(model)
    p = get_params(model)

    Na = g[:a].N
    Nz = length(S.zgrid)

    # Shape checks
    @test size(sol.policy[:c].value) == (Na, Nz)
    @test size(sol.policy[:a].value) == (Na, Nz)
    @test sol.policy[:c].euler_errors_mat !== nothing
    @test size(sol.policy[:c].euler_errors_mat) == (Na, Nz)

    # Residual consistency
    resid_mat = euler_resid_stoch(p, g[:a].grid, S.zgrid, S.Π, sol.policy[:c].value)
    @test maximum(abs.(resid_mat .- sol.policy[:c].euler_errors_mat)) < 1e-10

    # Metadata max_resid equals interior pruned maximum
    lo = Na > 2 ? 2 : 1
    hi = Na > 2 ? Na - 1 : Na
    expected_max = maximum(sol.policy[:c].euler_errors_mat[lo:hi, :])
    @test sol.metadata[:max_resid] == expected_max

    @test sol.metadata[:order] == 1
    @test sol.metadata[:valid] == true
end
