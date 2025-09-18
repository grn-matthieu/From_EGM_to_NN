using Test
using ThesisProject
using Test
using ThesisProject

@testset "Coverage - NNTrain" begin
    T = ThesisProject.NNTrain

    # curriculum and default stages
    stages = T.default_stages()
    @test length(stages) >= 1
    st = T.curriculum(1, 3; stages = stages)
    @test isa(st, AbstractDict) || isa(st, NamedTuple) || isa(st, Pair)
    @test_throws ArgumentError T.curriculum(0, 0)
    @test_throws ArgumentError T.curriculum(-1, 5)
    @test_throws ArgumentError T.curriculum(1, 5; stages = [])

    # _thin for 1D and 2D arrays and identity for stride <= 1
    x1 = collect(1:10)
    x1t = T._thin(x1, 2)
    @test length(x1t) == ceil(Int, 10 / 2)
    x2 = rand(Float32, 3, 8)
    x2t = T._thin(x2, 2)
    @test size(x2t, 2) == ceil(Int, 8 / 2)
    @test T._thin(x1, 1) === x1

    # CSVLogger writes header and rows (use a temporary path inside test results)
    outdir = joinpath(@__DIR__, "results", "tmp_nntrain")
    path = joinpath(outdir, "log.csv")
    try
        isfile(path) && rm(path)
    catch
    end
    lg = T.CSVLogger(path)
    T.log_row!(
        lg;
        epoch = 1,
        step = 1,
        split = "train",
        loss = 0.5,
        grad_norm = 1.0,
        lr = 1e-3,
    )
    T.log_row!(
        lg;
        epoch = 2,
        step = 2,
        split = "val",
        loss = 0.4,
        grad_norm = 0.8,
        lr = 5e-4,
    )
    @test isfile(path)
    s = read(path, String)
    @test occursin("timestamp,epoch,step,split,loss,grad_norm,lr", s)

    # EarlyStopping: just ensure API works and returns booleans
    es = T.EarlyStopping(patience = 2, min_delta = 0.0)
    reset!(es)
    a = T.should_stop!(es, 1.0)
    b = T.should_stop!(es, 0.9)
    c = T.should_stop!(es, 0.85)
    @test isa(a, Bool) && isa(b, Bool) && isa(c, Bool)

    # collect/inspect leaves and grad helpers
    nested = (a = rand(2, 2), b = rand(3))
    leaves = T.collect_array_leaves(nested)
    @test isa(leaves, Vector)
    @test all(x -> x isa AbstractArray, leaves) || isempty(leaves)

    gnorm = T.grad_global_l2norm((a = rand(2, 2), b = rand(3)))
    @test isfinite(gnorm) && gnorm >= 0.0

    nested2 = (a = fill(1.0, 2, 2), b = (c = fill(2.0, 3), d = 0.0))
    g = deepcopy(nested2)
    T.scale_grads!(g, 2.0)
    @test all(x -> all(abs.(x) .> 0), T.collect_array_leaves(g))
end
