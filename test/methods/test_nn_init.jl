using Test
using ThesisProject
using Lux
using Random

@testset "NN build/init smoke" begin
    for cfg in (SMOKE_CFG, SMOKE_STOCH_CFG)
        # Build and init
        model = ThesisProject.NNInit.build_nn(cfg)
        state1 = ThesisProject.NNInit.init_state(cfg)
        @test state1 isa ThesisProject.NNInit.NNState

        # Forward pass smoke
        has_shocks =
            cfg_has(cfg, :shocks) && cfg_getdefault(cfg_get(cfg, :shocks), :active, false)
        in_dim = has_shocks ? 2 : 1
        x = ones(Float32, in_dim, 1)
        y, st_new = Lux.apply(state1.model, x, state1.ps, state1.st)
        @test size(y) == (1, 1)

        # Determinism: isolated from global RNG
        Random.seed!(0)
        state2 = ThesisProject.NNInit.init_state(cfg)
        @test state1.ps == state2.ps

        # Changing cfg seed changes parameters
        random_cfg = cfg_has(cfg, :random) ? cfg_get(cfg, :random) : Dict{Symbol,Any}()
        base_seed = cfg_has(random_cfg, :seed) ? cfg_get(random_cfg, :seed) : 1234
        cfg2 = cfg_patch(cfg, :random => cfg_patch(random_cfg, :seed => base_seed + 1))
        state3 = ThesisProject.NNInit.init_state(cfg2)
        @test state3.ps != state1.ps
    end
end
