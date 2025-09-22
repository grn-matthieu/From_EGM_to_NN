using Test
using ThesisProject

@testset "Projection stochastic smoke" begin
    cfg = cfg_patch(SMOKE_STOCH_CFG, (:solver, :method) => "Projection")
    model = build_model(cfg)
    method = build_method(cfg)
    sol = solve(model, method, cfg)
    # Smoke-level checks for stochastic projection: shapes and finite residuals

    grids = get_grids(model)
    S = get_shocks(model)
    Na = cfg_get(grids, :a).N
    Nz = length(S.zgrid)
    policy_c = cfg_get(sol.policy, :c).value
    policy_a = cfg_get(sol.policy, :a).value

    @test size(policy_c) == (Na, Nz)
    @test is_nondec(policy_a; tol = 1e-8)
    @test all(isfinite, cfg_get(sol.policy, :c).euler_errors_mat)
    # Expect residuals reasonably small but not necessarily below deterministic tol
    @test maximum(cfg_get(sol.policy, :c).euler_errors[min(2, end):end]) < 1e-1
end
