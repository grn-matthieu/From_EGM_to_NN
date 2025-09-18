using Test
using ThesisProject
using Lux
using Random

const NNTrain = ThesisProject.NNTrain
const NNInit = ThesisProject.NNInit

@testset "NNTrain CSVLogger + EarlyStopping" begin
    tmpdir = mktempdir()
    csvpath = joinpath(tmpdir, "train.csv")
    lg = NNTrain.CSVLogger(csvpath)
    @test !lg.header_written[]
    NNTrain.log_row!(
        lg;
        epoch = 1,
        step = 1,
        split = "train",
        loss = 1.0,
        grad_norm = 0.1,
        lr = 1e-3,
    )
    NNTrain.log_row!(
        lg;
        epoch = 1,
        step = 2,
        split = "train",
        loss = 0.9,
        grad_norm = 0.2,
        lr = 1e-3,
    )
    @test isfile(csvpath)
    lines = readlines(csvpath)
    @test length(lines) == 3
    @test startswith(lines[1], "timestamp,epoch,step,split,loss,grad_norm,lr")

    es = NNTrain.EarlyStopping(patience = 2, min_delta = 0.05, enabled = true)
    @test !NNTrain.should_stop!(es, 1.0)
    @test !NNTrain.should_stop!(es, 1.03)  # first bad step (num_bad=1)
    @test NNTrain.should_stop!(es, 1.02)   # second bad step triggers stop (num_bad=2)
end

@testset "NNTrain _step! + train!" begin
    # Small deterministic config and synthetic data
    cfg = deepcopy(SMOKE_CFG)
    cfg[:solver] = get(cfg, :solver, Dict{Symbol,Any}())
    cfg[:solver][:epochs] = 2
    cfg[:solver][:clip_norm] = 1e-9   # force clipping path
    cfg[:solver][:patience] = 1
    cfg[:solver][:min_delta] = 0.0
    cfg[:logging] = Dict{Symbol,Any}(:dir => mktempdir())

    state = NNInit.init_state(cfg)

    # Toy data: y = sin(x)
    x = reshape(range(0, 1; length = 8), 1, :)
    y = reshape(sin.(x), 1, :)
    data = [(x, y) for _ = 1:3]
    val = [(x, y)]

    new_state, loss, gnorm, lr =
        NNTrain._step!(state, x, y; clip_norm = cfg[:solver][:clip_norm])
    @test new_state isa NNInit.NNState
    @test isfinite(loss) && isfinite(gnorm)
    @test new_state.ps != state.ps  # parameters updated

    # Train loop on iterable data + validation
    final_state = NNTrain.train!(state, data, cfg; val_data = val)
    @test final_state isa NNInit.NNState

    # Overload using model-only
    final_state2 = NNTrain.train!(state.model, data, cfg)
    @test final_state2 isa NNInit.NNState

    # Dummy epoch smoke
    st3 = NNTrain.dummy_epoch!(; n = 32, batch = 8, epochs = 1)
    @test st3 isa NNInit.NNState
end
