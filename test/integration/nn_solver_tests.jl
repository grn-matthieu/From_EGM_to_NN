@testset "Neural network solver" begin
    nn_overrides = (
        nn = (
            epochs = 1,
            batch = 8,
            samples_per_epoch = 16,
            hid1 = 4,
            hid2 = 4,
            target_loss = 1.0,
            use_cuda = false,
        ),
        maxit = 50,
    )
    cfg = stochastic_config(method = "NN", Na = 12, solver_overrides = nn_overrides)
    model, method = build_model_and_method(cfg)
    sol = ThesisProject.solve(model, method, cfg; rng = derive_solver_rng(cfg, "NN"))
    @test sol isa ThesisProject.Solution
    @test haskey(sol.policy, :c)
    @test sol.diagnostics.method == "NN"
    @test sol.metadata[:max_it] == method.opts.epochs

    sol_cfg = ThesisProject.solve(cfg; rng = cfg.random.master_rng)
    @test sol_cfg isa ThesisProject.Solution
end
