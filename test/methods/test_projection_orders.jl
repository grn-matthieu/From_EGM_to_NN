using Test
using ThesisProject
using ThesisProject.ProjectionKernel: solve_projection_det, solve_projection_stoch
using ThesisProject.Chebyshev: chebyshev_basis, chebyshev_nodes
using ThesisProject.EulerResiduals: euler_resid_det_grid, euler_resid_stoch

@testset "projection order selection deterministic" begin
    cfg = cfg_patch(
        SMOKE_CFG,
        (:solver, :method) => "Projection",
        (:solver, :orders) => [2, 3],
        (:solver, :Nval) => 21,
        (:grids, :Na) => 20,
    )
    model = build_model(cfg)
    method = build_method(cfg)
    sol = solve(model, method, cfg)
    @test sol.metadata[:order] in cfg_get(cfg, :solver, :orders)

    p = get_params(model)
    g = get_grids(model)
    U = get_utility(model)
    orders = cfg_get(cfg, :solver, :orders)
    Nval = cfg_get(cfg, :solver, :Nval)
    maxres = Float64[]
    for k in orders
        sol_k = solve_projection_det(
            p,
            g,
            U;
            orders = [k],
            Nval = Nval,
            tol = method.opts.tol,
            maxit = method.opts.maxit,
        )
        a_grid = cfg_get(g, :a)
        a_val = chebyshev_nodes(Nval, a_grid.min, a_grid.max)
        B_val = chebyshev_basis(a_val, k, a_grid.min, a_grid.max)
        c_val = B_val * sol_k.coeffs
        resid_val = euler_resid_det_grid(p, a_val, c_val)
        push!(maxres, maximum(resid_val[min(2, end):end]))
    end
    @test maxres[2] < maxres[1]
    idx = argmin(maxres)
    @test sol.metadata[:order] == orders[idx]
end

@testset "projection order selection stochastic" begin
    cfg = cfg_patch(
        SMOKE_STOCH_CFG,
        (:solver, :method) => "Projection",
        (:solver, :orders) => [2, 3],
        (:solver, :Nval) => 21,
        (:grids, :Na) => 15,
        (:shocks, :Nz) => 3,
    )
    model = build_model(cfg)
    method = build_method(cfg)
    sol = solve(model, method, cfg)
    @test sol.metadata[:order] in cfg_get(cfg, :solver, :orders)

    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)
    orders = cfg_get(cfg, :solver, :orders)
    Nval = cfg_get(cfg, :solver, :Nval)
    maxres = Float64[]
    for k in orders
        sol_k = solve_projection_stoch(
            p,
            g,
            S,
            U;
            orders = [k],
            Nval = Nval,
            tol = method.opts.tol,
            maxit = method.opts.maxit,
        )
        a_grid = cfg_get(g, :a)
        a_val = chebyshev_nodes(Nval, a_grid.min, a_grid.max)
        B_val = chebyshev_basis(a_val, k, a_grid.min, a_grid.max)
        c_val = B_val * sol_k.coeffs
        resid_val = euler_resid_stoch(p, a_val, S.zgrid, S.Π, c_val)
        push!(maxres, maximum(resid_val[min(2, end):end, :]))
    end
    @test maxres[2] < maxres[1]
    idx = argmin(maxres)
    @test sol.metadata[:order] == orders[idx]
end
