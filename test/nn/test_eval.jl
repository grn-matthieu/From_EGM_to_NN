using Test
using ThesisProject

# Import needed APIs
using ThesisProject.NNLoss: check_finite_residuals, total_loss

@testset "NN tiny evaluation pipeline" begin
    # Load a simple deterministic config and build model
    cfg_path = joinpath(@__DIR__, "..", "..", "config", "simple_baseline.yaml")
    cfg = ThesisProject.load_config(cfg_path)
    cfg[:solver][:method] = :NN

    model = ThesisProject.build_model(cfg)
    params = ThesisProject.get_params(model)
    grids = ThesisProject.get_grids(model)

    # Build a feasible baseline NN solution to source a policy and feasibility metric
    util = ThesisProject.get_utility(model)
    sol = ThesisProject.NNKernel.solve_nn_det(params, grids, util, cfg)

    # Construct a minimal policy dict compatible with residual computation
    policy = Dict{Symbol,Any}(:c => (; value = sol.c, grid = sol.a_grid))

    # Tiny synthetic batch: first n points from the asset grid
    n = 8
    batch = sol.a_grid[1:min(n, length(sol.a_grid))]

    # Residuals should be finite on this tiny batch
    @test check_finite_residuals(model, policy, batch)

    # Compute residuals explicitly for loss checks
    R = ThesisProject.NNLoss._compute_residuals(model, policy, batch)

    # Use a_min from grid; craft an ap vector
    a_min = grids[:a].min
    ap_ok = fill(a_min, length(batch))

    # total_loss is finite with default settings (no penalty weight)
    L0 = total_loss(R, ap_ok, a_min)
    @test isfinite(L0)

    # Changing ap below the bound should not crash when penalty weight is default
    ap_bad = fill(a_min - 0.1, length(batch))
    L1 = total_loss(R, ap_bad, a_min)
    @test isfinite(L1)

    # If feasibility metric exists, assert it lies in [0, 1]
    feas = get(sol.opts, :feasibility, nothing)
    if feas !== nothing
        @test 0.0 <= feas <= 1.0
    end
end
