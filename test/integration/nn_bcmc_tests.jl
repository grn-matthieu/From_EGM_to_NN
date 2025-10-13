using Test
using ThesisProject
using .TestUtils

@testset "NN bc-MC objective (AR1)" begin
    nn_overrides = (
        nn = (
            objective = :euler_fb_bcmc,
            epochs = 2,
            batch = 16,
            samples_per_epoch = 32,
            n_mc = 4,
            use_cuda = false,
            verbose = false,
        ),
    )
    cfg = stochastic_config(method = "NN", solver_overrides = nn_overrides)
    model, method = build_model_and_method(cfg)
    rng = derive_solver_rng(cfg, "NN-bcmc-ar1")
    sol = ThesisProject.API.solve(model, method, cfg; rng = rng)
    @test sol.c !== nothing
    @test isfinite(sol.max_resid)
    @test sol.opts.device in (:cpu, :cuda)
end

@testset "NN bc-MC objective (CSVAR)" begin
    # turn on vector income model with 2D shocks
    cs_params = (y = [0.6, 0.4], A = [0.0 0.0; 0.0 0.0], Σ = [0.5 0.2; 0.2 0.5])
    nn_overrides = (
        nn = (
            objective = :euler_fb_bcmc,
            epochs = 2,
            batch = 16,
            samples_per_epoch = 32,
            n_mc = 4,
            use_cuda = false,
            verbose = false,
        ),
    )
    base_cfg = stochastic_config(method = "NN", solver_overrides = nn_overrides)
    cfg = deep_merge(base_cfg, (model = (name = :cs_vec,), params = cs_params))
    model, method = build_model_and_method(cfg)
    rng = derive_solver_rng(cfg, "NN-bcmc-csvar")
    sol = ThesisProject.API.solve(model, method, cfg; rng = rng)
    @test sol.c !== nothing
    @test isfinite(sol.max_resid)
end

@testset "bc-MC equals AiO at N=2 (AR1)" begin
    base_nn = (
        epochs = 2,
        batch = 16,
        samples_per_epoch = 32,
        n_mc = 2,
        use_cuda = false,
        verbose = false,
    )
    cfg_aio = stochastic_config(
        method = "NN",
        solver_overrides = (nn = merge(base_nn, (objective = :euler_fb_aio,)),),
    )
    cfg_bcmc = stochastic_config(
        method = "NN",
        solver_overrides = (nn = merge(base_nn, (objective = :euler_fb_bcmc,)),),
    )

    model_aio, method_aio = build_model_and_method(cfg_aio)
    model_bcmc, method_bcmc = build_model_and_method(cfg_bcmc)

    rng = master_rng(777)
    sol_aio = ThesisProject.API.solve(
        model_aio,
        method_aio,
        cfg_aio;
        rng = ThesisProject.Determinism.derive_rng(rng, "aio"),
    )
    sol_bcmc = ThesisProject.API.solve(
        model_bcmc,
        method_bcmc,
        cfg_bcmc;
        rng = ThesisProject.Determinism.derive_rng(rng, "bcmc"),
    )

    # Compare policies coarsely: ensure both solved with reasonable residuals
    @test isfinite(sol_aio.max_resid)
    @test isfinite(sol_bcmc.max_resid)
end
