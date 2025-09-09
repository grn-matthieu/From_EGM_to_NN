using Statistics

@testset "SimPanel smoke" begin
    cfg_path = joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_config_stochastic.yaml")
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
        @warn "simulate_panel threw an error" err=(err, catch_backtrace())
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

    # diagnostics basic checks (allow missing keys but prefer presence)
    @test isa(diag, Dict) || isa(diag, NamedTuple)
    if (:master_seed in keys(diag)) || (:master_seed in collect(fieldnames(typeof(diag))))
        @test !ismissing(getkey(diag, :master_seed))
    end
    if (:rng_kind in keys(diag)) || (:rng_kind in collect(fieldnames(typeof(diag))))
        @test !isempty(string(getkey(diag, :rng_kind)))
    end
end