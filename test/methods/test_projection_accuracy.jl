using Test
using ThesisProject
using ThesisProject.Chebyshev: chebyshev_basis, chebyshev_nodes
using ThesisProject.CommonInterp: interp_pchip!
using ThesisProject.EulerResiduals: euler_resid_det_2, euler_resid_stoch

@testset "Projection solution accuracy and metadata" begin
    # Deterministic projection: residuals and metadata consistency
    cfg_det = cfg_patch(
        SMOKE_CFG,
        (:solver, :method) => "Projection",
        (:solver, :orders) => [3],
        (:solver, :Nval) => 41,
    )
    model_det = build_model(cfg_det)
    method_det = build_method(cfg_det)
    sol_det = solve(model_det, method_det, cfg_det)

    # policy and residual consistency on output grid
    p_det = get_params(model_det)
    g_det = get_grids(model_det)
    a_out = cfg_get(g_det, :a).grid
    c_det_policy = cfg_get(sol_det.policy, :c)
    resid_det = euler_resid_det_2(p_det, a_out, c_det_policy.value)
    @test maximum(abs.(resid_det .- c_det_policy.euler_errors)) < 1e-10

    # max_resid should reflect pruned boundary points
    max_interior = maximum(resid_det[min(2, end):end])
    @test sol_det.metadata[:max_resid] == max_interior

    # Order should match requested
    @test sol_det.metadata[:order] == 3

    # Off-grid residual check via monotone interpolation
    grid = c_det_policy.grid
    values = c_det_policy.value
    step = (grid[end] - grid[1]) / (length(grid) - 1)
    a_off =
        collect(range(grid[1] + step / 2, grid[end] - step / 2; length = length(grid) - 1))
    c_off = similar(a_off)
    interp_pchip!(c_off, grid, values, a_off)
    resid_off = euler_resid_det_2(p_det, a_off, c_off)
    @test maximum(abs.(resid_off)) < 3e-2

    # Stochastic projection: matrix residuals present and shaped
    cfg_st = cfg_patch(
        SMOKE_STOCH_CFG,
        (:solver, :method) => "Projection",
        (:solver, :orders) => [2],
        (:solver, :Nval) => 21,
    )
    model_st = build_model(cfg_st)
    method_st = build_method(cfg_st)
    sol_st = solve(model_st, method_st, cfg_st)

    g_st = get_grids(model_st)
    S_st = get_shocks(model_st)
    Na = cfg_get(g_st, :a).N
    Nz = length(S_st.zgrid)
    c_st_policy = cfg_get(sol_st.policy, :c)
    @test c_st_policy.euler_errors_mat !== nothing
    @test size(c_st_policy.euler_errors_mat) == (Na, Nz)

    # Residual-metadata consistency (pruned along asset dimension)
    max_interior_st = maximum(c_st_policy.euler_errors_mat[min(2, end):end, :])
    @test sol_st.metadata[:max_resid] == max_interior_st
end
