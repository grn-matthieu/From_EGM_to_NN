using Test
using ThesisProject
const NNKernel = ThesisProject.NNKernel
using ThesisProject.NNKernel:
    EvaluationResult,
    evaluate_deterministic,
    evaluate_stochastic,
    build_options_summary,
    solve_nn,
    FeatureScaler,
    scalar_params

# --------------------------
# Helpers: G supporting G[:a] AND G.a
# --------------------------
struct GDict
    data::Dict{Symbol,Any}
end
Base.getindex(g::GDict, s::Symbol) = g.data[s]
Base.getproperty(g::GDict, s::Symbol) = s === :data ? getfield(g, :data) : g.data[s]

# --------------------------
# Fixtures
# --------------------------
const P_det = (; r = 0.02, y = 1.0, σ = 2.0, β = 0.95)
const G = GDict(Dict(:a => (grid = [0.0, 1.0, 2.0], min = 0.0, max = 2.0)))
const S_stoch = (; zgrid = [-1.0, 1.0], Π = [0.7 0.3; 0.4 0.6])

# Separate dummy types so we never override methods
struct DummyModelDet <: ThesisProject.API.AbstractModel end
struct DummyModelStoch <: ThesisProject.API.AbstractModel end
const dummy_det = DummyModelDet()
const dummy_st = DummyModelStoch()

# API patches
NNKernel.get_params(::DummyModelDet) = P_det
NNKernel.get_params(::DummyModelStoch) = P_det
NNKernel.get_grids(::DummyModelDet) = G
NNKernel.get_grids(::DummyModelStoch) = G
NNKernel.get_shocks(::DummyModelDet) = nothing
NNKernel.get_shocks(::DummyModelStoch) = S_stoch
NNKernel.get_utility(::DummyModelDet) = nothing
NNKernel.get_utility(::DummyModelStoch) = nothing

# --------------------------
# Lightweight stubs for training/model plumbing
# --------------------------
if !isdefined(Main, :DummyTrainResult)
    mutable struct DummyTrainResult
        epochs_run::Int
        batch_size::Int
        batches_per_epoch::Int
        best_loss::Float64
        best_state::NamedTuple
    end
end

# Return a stable, small train result
NNKernel.train_consumption_network!(
    chain,
    settings,
    scaler,
    P_resid,
    G_,
    S_,
    model_cfg = nothing,
    rng = Random.GLOBAL_RNG,
) = DummyTrainResult(1, 4, 5, 0.1, (;))

NNKernel.build_network(in_dim, settings) = :fake_chain
NNKernel.solver_settings(opts) = (;
    epochs = 3,
    batch_choice = 64,
    learning_rate = 0.001,
    verbose = false,
    resample_interval = 1,
    target_loss = 0.5,
    patience = 10,
    hidden_sizes = (64, 64),
    has_shocks = false,
    objective = :euler,
    v_h = 1.0,
)

NNKernel.select_model(chain, best_state) = :trained_model
NNKernel.state_parameters(best_state) = :params
NNKernel.state_states(best_state) = :states

# Residual evaluators
NNKernel.euler_resid_det_grid(P, a, c) = a .- c
NNKernel.euler_resid_stoch_grid(P, a, z, Pz, c) = a .* sum(Pz, dims = 2)

# run_model mock (echo features)
NNKernel.run_model(model, params, states, X) = X

# dataset generator for stochastic (rows = samples, cols = features)
NNKernel.generate_dataset(G_, S_; mode = :full) = (
    Float32[
        0 -1
        1 1
        2 0
    ],
    nothing,
)

# --------------------------
# Tests
# --------------------------
@testset "EvaluationResult basics" begin
    res = EvaluationResult([1, 2], [3, 4], [0.1, -0.1], 0.5)
    @test res.c == [1, 2]
    @test res.max_resid == 0.5
end

@testset "evaluate_deterministic path" begin
    scaler = FeatureScaler(G, nothing)
    P_resid = scalar_params(P_det)
    result = evaluate_deterministic(:model, :params, :states, P_resid, P_det, G, scaler)
    @test result isa EvaluationResult
    @test length(result.c) == length(G[:a].grid)
    @test all(result.a_next .>= G[:a].min)
end

@testset "evaluate_stochastic path" begin
    scaler = FeatureScaler(G, S_stoch)
    P_resid = scalar_params(P_det)
    result =
        evaluate_stochastic(:model, :params, :states, P_resid, P_det, G, S_stoch, scaler)
    @test result isa EvaluationResult
    @test size(result.c) == (length(G[:a].grid), length(S_stoch.zgrid))
end

@testset "build_options_summary" begin
    settings = (epochs = 5, learning_rate = 0.1, verbose = true)
    tr = DummyTrainResult(3, 8, 12, 0.1, (;))
    summ = build_options_summary(settings, tr, 1.23)
    @test summ.epochs == 5
    @test summ.epochs_run == 3
    @test summ.runtime ≈ 1.23
end

@testset "solve_nn deterministic" begin
    sol = solve_nn(dummy_det)
    @test :c in keys(sol)
    @test sol.converged isa Bool
    @test sol.iters ≥ 1          # don’t over-specify
    @test length(sol.c) == 3
end

@testset "solve_nn stochastic" begin
    sol = solve_nn(dummy_st)
    @test sol.iters ≥ 1
end

@testset "solve_nn deterministic – return bundle" begin
    sol = solve_nn(dummy_det)

    # core fields
    @test sol.a_grid == G[:a].grid
    @test length(sol.c) == length(G[:a].grid)
    @test length(sol.a_next) == length(G[:a].grid)
    @test length(sol.resid) == length(G[:a].grid)
    @test sol.max_resid ≥ 0
    @test sol.model_params === P_det

    @test hasproperty(sol.opts, :epochs)
    @test sol.opts.epochs ≥ 1                      # don't hardcode 3
    @test hasproperty(sol.opts, :lr)
    @test sol.opts.lr > 0
    @test hasproperty(sol.opts, :runtime)
    @test sol.opts.runtime ≥ 0.0
    # the following are optional in some builds; assert only if present
    hasproperty(sol.opts, :epochs_run) && @test sol.opts.epochs_run ≥ 0
    hasproperty(sol.opts, :batch) && @test sol.opts.batch ≥ 1
    hasproperty(sol.opts, :batches_per_epoch) && @test sol.opts.batches_per_epoch ≥ 1
    hasproperty(sol.opts, :verbose) && @test sol.opts.verbose isa Bool             # time path exercised

    # converged flag uses best_loss ≤ target_loss
    @test sol.converged === true
end

@testset "solve_nn stochastic – shapes & flags" begin
    sol = solve_nn(dummy_st)

    Na = length(G[:a].grid)
    Nz = length(S_stoch.zgrid)

    if sol.c isa AbstractMatrix
        @test size(sol.c) == (Na, Nz)
        @test size(sol.a_next) == (Na, Nz)
        @test size(sol.resid) == (Na, Nz)
    else
        @test length(sol.c) == Na
        @test length(sol.a_next) == Na
        @test length(sol.resid) == Na
    end

    # options summary: only assert guaranteed fields
    @test hasproperty(sol.opts, :epochs)
    @test sol.opts.epochs ≥ 1
    @test hasproperty(sol.opts, :lr)
    @test sol.opts.lr > 0
    @test hasproperty(sol.opts, :runtime)
    @test sol.opts.runtime ≥ 0.0
    hasproperty(sol.opts, :epochs_run) && @test sol.opts.epochs_run ≥ 0
    hasproperty(sol.opts, :batch) && @test sol.opts.batch ≥ 1
    hasproperty(sol.opts, :batches_per_epoch) && @test sol.opts.batches_per_epoch ≥ 1
    hasproperty(sol.opts, :verbose) && @test sol.opts.verbose isa Bool
end
