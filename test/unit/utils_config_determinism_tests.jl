const Determinism = ThesisProject.Determinism
const UtilsConfig = ThesisProject.UtilsConfig

@testset "Determinism utilities" begin
    master = Determinism.make_master_rng(123)
    @test Determinism.master_seed(master) == UInt64(123)

    rng = Determinism.make_rng(123)
    baseline = Determinism.make_rng(123)
    _ = rand(rng, UInt64)
    _ = rand(baseline, UInt64)
    Determinism.make_master_rng(rng)
    @test rand(rng, UInt64) == rand(baseline, UInt64)

    seed_a = Determinism.derive_seed(master, :egm)
    seed_b = Determinism.derive_seed(master, :projection)
    @test seed_a == Determinism.derive_seed(master, :egm)
    @test seed_a != seed_b

    rng1 = Determinism.derive_rng(master, :egm)
    rng2 = Determinism.derive_rng(master, :egm)
    @test rand(rng1, 3) == rand(rng2, 3)

    cfg_a = (b = 2, a = 1, nested = (y = 2.3456789, x = 1.2345678))
    cfg_b = (a = 1, nested = (x = 1.2345678, y = 2.3456789), b = 2)
    canon_a = Determinism.canonicalize_cfg(cfg_a)
    canon_b = Determinism.canonicalize_cfg(cfg_b)
    @test canon_a == canon_b
    @test length(Determinism.hash_hex(canon_a)) == 12
end

@testset "Config helpers" begin
    cfg = deterministic_config()
    cfg_validated = UtilsConfig.validate_config(cfg)
    @test cfg_validated == cfg
    @test hasproperty(cfg_validated.params, :γ)

    legacy_cfg = deterministic_config()
    legacy_params = legacy_cfg.params
    legacy_params_sigma =
        (β = legacy_params.β, σ = legacy_params.γ, r = legacy_params.r, y = legacy_params.y)
    cfg_sigma = merge(legacy_cfg, (params = legacy_params_sigma,))
    cfg_sigma_validated = UtilsConfig.validate_config(cfg_sigma)
    @test hasproperty(cfg_sigma_validated.params, :γ)
    @test cfg_sigma_validated.params.γ == legacy_params.γ

    enriched = UtilsConfig.ensure_master_rng((random = (seed = 99,),))
    @test hasproperty(enriched.random, :master_rng)
    @test UtilsConfig.maybe_nested(cfg, :solver, :egm, :interp_kind) == :linear
    @test UtilsConfig.maybe_nested(cfg, :solver, :missing_field; default = false) == false
    @test UtilsConfig.maybe(nothing, :anything, 42) == 42

    bad_cfg = deterministic_config(solver_overrides = (; tol = -1.0))
    @test_throws ErrorException UtilsConfig.validate_config(bad_cfg)

    cfg_all = deterministic_config(solver_overrides = (; method = [:EGM, :Projection]))
    cfg_all_validated = UtilsConfig.validate_config(cfg_all)
    @test cfg_all_validated == cfg_all
end
