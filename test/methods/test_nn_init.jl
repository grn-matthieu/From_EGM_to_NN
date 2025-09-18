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
        in_dim = (haskey(cfg, :shocks) && get(cfg[:shocks], :active, false)) ? 2 : 1
        x = ones(Float32, in_dim, 1)
        y, st_new = Lux.apply(state1.model, x, state1.ps, state1.st)
        @test size(y) == (1, 1)

        # Determinism: isolated from global RNG
        Random.seed!(0)
        state2 = ThesisProject.NNInit.init_state(cfg)
        @test state1.ps == state2.ps

        # Changing cfg seed changes parameters
        cfg2 = deepcopy(cfg)
        if !haskey(cfg2, :random)
            cfg2[:random] = Dict{Symbol,Any}()
        end
        cfg2[:random][:seed] = get(get(cfg2, :random, Dict{Symbol,Any}()), :seed, 1234) + 1
        state3 = ThesisProject.NNInit.init_state(cfg2)
        @test state3.ps != state1.ps
    end
end
