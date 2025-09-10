using Statistics

@testset "SimPanel smoke" begin
    cfg_path =
        joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_config_stochastic.yaml")
    @test isfile(cfg_path) || begin
        @warn("Config not found; skipping SimPanel tests", path = cfg_path)
        @test true
        return
    end

    cfg = ThesisProject.load_config(cfg_path)

    model = ThesisProject.build_model(cfg)
    method = ThesisProject.build_method(cfg)

    # Short panel for the test, no need for long sim here
    N = 12
    T = 6

    rng1 = ThesisProject.Determinism.make_rng(1234)
    rng2 = ThesisProject.Determinism.make_rng(1234)

    out1 = nothing # We try two identical RNGs to test reproducibility
    out2 = nothing

    try
        out1 = ThesisProject.simulate_panel(model, method, cfg; N = N, T = T, rng = rng1)
        out2 = ThesisProject.simulate_panel(model, method, cfg; N = N, T = T, rng = rng2)
    catch err
        @warn "simulate_panel threw an error" err = (err, catch_backtrace())
        @test false
        return
    end

    @test isa(out1, NamedTuple) || isa(out1, Dict)

    # named access tolerant to either NamedTuple or Dict
    getkey(x, k) = x isa NamedTuple ? getfield(x, k) : x[k]

    assets = getkey(out1, :assets)
    cons = getkey(out1, :consumption)
    shocks = getkey(out1, :shocks)
    seeds = getkey(out1, :seeds)
    diag = getkey(out1, :diagnostics)

    @test size(assets) == (N, T)
    @test size(cons) == (N, T)
    @test size(shocks) == (N, T)
    @test length(seeds) == N

    @test all(isfinite, vec(assets))
    @test all(isfinite, vec(cons))
    @test all(isfinite, vec(shocks))

    # reproducibility with same seed
    assets2 = getkey(out2, :assets)
    cons2 = getkey(out2, :consumption)
    shocks2 = getkey(out2, :shocks)
    seeds2 = getkey(out2, :seeds)

    @test seeds == seeds2
    @test assets ≈ assets2
    @test cons ≈ cons2
    @test shocks ≈ shocks2

    # reproducibility when reusing the same RNG instance
    rng_shared = ThesisProject.Determinism.make_rng(2024)
    out_shared1 =
        ThesisProject.simulate_panel(model, method, cfg; N = N, T = T, rng = rng_shared)
    out_shared2 =
        ThesisProject.simulate_panel(model, method, cfg; N = N, T = T, rng = rng_shared)
    @test getkey(out_shared1, :seeds) == getkey(out_shared2, :seeds)
    @test getkey(out_shared1, :shocks) ≈ getkey(out_shared2, :shocks)
    @test getkey(out_shared1, :consumption) ≈ getkey(out_shared2, :consumption)
    @test getkey(out_shared1, :assets) ≈ getkey(out_shared2, :assets)

    # different seeds produce different panels
    cfg2 = deepcopy(cfg)
    cfg2[:random] = deepcopy(cfg[:random])
    cfg2[:random][:seed] = cfg2[:random][:seed] + 1
    rng_diff = ThesisProject.Determinism.make_rng(1234)
    out_diff =
        ThesisProject.simulate_panel(model, method, cfg2; N = N, T = T, rng = rng_diff)
    shocks_diff = getkey(out_diff, :shocks)
    seeds_diff = getkey(out_diff, :seeds)
    @test seeds != seeds_diff
    @test !(shocks ≈ shocks_diff)

    # missing cfg.random.seed should still be reproducible with identical RNGs
    cfg_noseed = deepcopy(cfg)
    cfg_noseed[:random] = Dict(:seed => nothing)
    rng_a = ThesisProject.Determinism.make_rng(42)
    rng_b = ThesisProject.Determinism.make_rng(42)
    out_a =
        ThesisProject.simulate_panel(model, method, cfg_noseed; N = N, T = T, rng = rng_a)
    out_b =
        ThesisProject.simulate_panel(model, method, cfg_noseed; N = N, T = T, rng = rng_b)
    shocks_a = getkey(out_a, :shocks)
    shocks_b = getkey(out_b, :shocks)
    seeds_a = getkey(out_a, :seeds)
    seeds_b = getkey(out_b, :seeds)
    @test seeds_a == seeds_b
    @test shocks_a ≈ shocks_b

    # diagnostics basic checks (allow missing keys but prefer presence)
    @test isa(diag, Dict) || isa(diag, NamedTuple)
    if (:master_seed in keys(diag)) || (:master_seed in collect(fieldnames(typeof(diag))))
        @test !ismissing(getkey(diag, :master_seed))
    end
    if (:rng_kind in keys(diag)) || (:rng_kind in collect(fieldnames(typeof(diag))))
        @test !isempty(string(getkey(diag, :rng_kind)))
    end
end

@testset "SimPanel deterministic" begin
    cfg_path = joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_config.yaml")
    @test isfile(cfg_path) || begin
        @warn("Config not found; skipping deterministic SimPanel tests", path = cfg_path)
        @test true
        return
    end

    cfg = ThesisProject.load_config(cfg_path)

    model = ThesisProject.build_model(cfg)
    method = ThesisProject.build_method(cfg)

    N = 12
    T = 6

    rng1 = ThesisProject.Determinism.make_rng(1234)
    rng2 = ThesisProject.Determinism.make_rng(1234)

    out1 = ThesisProject.simulate_panel(model, method, cfg; N = N, T = T, rng = rng1)
    out2 = ThesisProject.simulate_panel(model, method, cfg; N = N, T = T, rng = rng2)

    getkey(x, k) = x isa NamedTuple ? getfield(x, k) : x[k]

    assets = getkey(out1, :assets)
    cons = getkey(out1, :consumption)
    shocks = getkey(out1, :shocks)

    @test size(assets) == (N, T)
    @test size(cons) == (N, T)
    @test shocks == zeros(N, T)

    assets2 = getkey(out2, :assets)
    cons2 = getkey(out2, :consumption)
    shocks2 = getkey(out2, :shocks)

    @test assets ≈ assets2
    @test cons ≈ cons2
    @test shocks == shocks2
end
