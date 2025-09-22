using Test
using ThesisProject

@testset "EGM stochastic smoke" begin
    cfg = deepcopy(SMOKE_STOCH_CFG)
    model = build_model(cfg)
    method = build_method(cfg)
    sol = solve(model, method, cfg)

    @test sol.metadata[:converged] === true
    @test maximum(cfg_get(sol.policy, :c).euler_errors[min(2, end):end]) <
          sol.metadata[:tol]

    grids = get_grids(model)
    S = get_shocks(model)
    Na = cfg_get(grids, :a).N
    Nz = length(S.zgrid)
    policy_a = cfg_get(sol.policy, :a).value

    @test size(policy_a) == (Na, Nz)
    @test is_nondec(policy_a; tol = 1e-8)
end
