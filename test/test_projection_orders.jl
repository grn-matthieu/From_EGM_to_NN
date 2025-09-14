using Test
using ThesisProject
using ThesisProject.ProjectionKernel: solve_projection_det, solve_projection_stoch
using ThesisProject.Chebyshev: chebyshev_basis, chebyshev_nodes
using ThesisProject.EulerResiduals: euler_resid_det_2, euler_resid_stoch

@testset "projection order selection deterministic" begin
    cfg_path = joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_config.yaml")
    cfg = load_config(cfg_path)
    cfg[:solver][:method] = "Projection"
    cfg[:solver][:orders] = [2, 3]
    cfg[:solver][:Nval] = 21
    cfg[:grids][:Na] = 20
    model = build_model(cfg)
    method = build_method(cfg)
    sol = solve(model, method, cfg)
    @test sol.metadata[:order] in cfg[:solver][:orders]

    p = get_params(model)
    g = get_grids(model)
    U = get_utility(model)
    orders = cfg[:solver][:orders]
    Nval = cfg[:solver][:Nval]
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
        a_val = chebyshev_nodes(Nval, g[:a].min, g[:a].max)
        B_val = chebyshev_basis(a_val, k, g[:a].min, g[:a].max)
        c_val = B_val * sol_k.coeffs
        resid_val = euler_resid_det_2(p, a_val, c_val)
        push!(maxres, maximum(resid_val[min(2, end):end]))
    end
    @test maxres[2] < maxres[1]
    idx = argmin(maxres)
    @test sol.metadata[:order] == orders[idx]
end

@testset "projection order selection stochastic" begin
    cfg_path =
        joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_config_stochastic.yaml")
    cfg = load_config(cfg_path)
    cfg[:solver][:method] = "Projection"
    cfg[:solver][:orders] = [2, 3]
    cfg[:solver][:Nval] = 21
    cfg[:grids][:Na] = 15
    cfg[:shocks][:Nz] = 3
    model = build_model(cfg)
    method = build_method(cfg)
    sol = solve(model, method, cfg)
    @test sol.metadata[:order] in cfg[:solver][:orders]

    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)
    U = get_utility(model)
    orders = cfg[:solver][:orders]
    Nval = cfg[:solver][:Nval]
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
        a_val = chebyshev_nodes(Nval, g[:a].min, g[:a].max)
        B_val = chebyshev_basis(a_val, k, g[:a].min, g[:a].max)
        c_val = B_val * sol_k.coeffs
        resid_val = euler_resid_stoch(p, a_val, S.zgrid, S.Π, c_val)
        push!(maxres, maximum(resid_val[min(2, end):end, :]))
    end
    @test maxres[2] < maxres[1]
    idx = argmin(maxres)
    @test sol.metadata[:order] == orders[idx]
end
