# test/methods/test_nn_training_utils.jl
using ThesisProject
using LinearAlgebra: I
using Random

const NN = ThesisProject.NNKernel  # adjust if these funcs live under a submodule

# ---- helpers for grids/shocks -------------------------------------------------
make_G(a::AbstractVector) = Dict(:a => (grid = a,))
make_S(z::AbstractVector) = (zgrid = Float32.(z),)

# ---- solver_settings ----------------------------------------------------------
@testset "solver_settings parsing and clamping" begin
    # defaults
    s = NN.solver_settings(nothing)
    @test s.epochs == 1000
    @test s.batch_choice == 64
    @test s.learning_rate ≈ 1e-3
    @test s.verbose == false
    @test s.resample_interval == 1
    @test s.target_loss ≈ 2e-4f0
    @test s.patience == 200
    @test s.hidden_sizes == (128, 128)
    @test s.objective == :euler_fb_aio
    @test s.v_h == 0.5
    @test s.w_min ≈ 0.1f0
    @test s.w_max ≈ 4.0f0
    @test s.sigma_shocks === nothing

    # provided opts with negatives and zeros to trigger clamps
    opts = (;
        epochs = -3,
        batch = 0,
        lr = 5e-4,
        verbose = true,
        resample_every = -1,
        target_loss = 1e-5 * 1.0f0,
        patience = -7,
        hid1 = 0,
        hid2 = 1,
        objective = "euler",
        v_h = 0.1,
        w_min = 0.05,
        w_max = 10.0,
        sigma_shocks = 0.2,
    )
    s2 = NN.solver_settings(opts; has_shocks = true)
    @test s2.epochs == 0
    @test s2.batch_choice == 1
    @test s2.learning_rate ≈ 5e-4
    @test s2.verbose == true
    @test s2.resample_interval == 0
    @test s2.target_loss ≈ 1e-5 * 1.0f0
    @test s2.patience == 0
    @test s2.hidden_sizes == (1, 1)
    @test s2.objective == :euler
    @test s2.v_h == 0.2             # clamped to lower bound
    @test s2.w_min ≈ 0.05f0
    @test s2.w_max ≈ 10.0f0
    @test s2.sigma_shocks ≈ 0.2
    @test s2.has_shocks
end

# ---- build_network / create_optimizer ----------------------------------------
@testset "network and optimizer" begin
    s = NN.solver_settings((; hid1 = 4, hid2 = 3))
    net = NN.build_network(2, s)
    @test occursin("Dense(2 => 4", string(net))
    @test occursin("Dense(4 => 3", string(net))
    @test occursin("Dense(3 => 1", string(net))

    opt = NN.create_optimizer(s)
    @test occursin("ClipGrad", string(opt))
    @test occursin("Adam", string(opt))
end

# ---- compute_batch_size -------------------------------------------------------
@testset "compute_batch_size" begin
    @test NN.compute_batch_size(10, nothing) == 10
    @test NN.compute_batch_size(10, 5) == 5
    @test NN.compute_batch_size(10, 100) == 10
    @test NN.compute_batch_size(0, nothing) == 1   # guard
    @test NN.compute_batch_size(0, 0) == 1
end

# ---- huber_loss ---------------------------------------------------------------
@testset "huber_loss regions" begin
    @test NN.huber_loss(0.1f0, 1.0f0) ≈ 0.5f0 * 0.1f0^2
    @test NN.huber_loss(2.0f0, 1.0f0) ≈ 1.0f0 * (2.0f0 - 0.5f0 * 1.0f0)
end

# ---- flatten_sum_squares ------------------------------------------------------
@testset "flatten_sum_squares" begin
    @test NN.flatten_sum_squares(nothing) == 0.0
    @test NN.flatten_sum_squares(3) == 9.0
    @test NN.flatten_sum_squares([1.0, -2.0]) == 5.0
    nt = (; a = 2, b = (1, 2), c = [1, 2, 2])
    @test NN.flatten_sum_squares(nt) == 2.0^2 + 1.0^2 + 2.0^2 + 1.0^2 + 2.0^2 + 2.0^2
    d = Dict(:x => 3, :y => [1, 1])
    @test NN.flatten_sum_squares(d) == 9.0 + 2.0
    # custom struct path
    struct Foo
        u::Int
        v::Vector{Float64}
    end
    @test NN.flatten_sum_squares(Foo(2, [1.0])) == 4.0 + 1.0
    # fallback catch branch on non-iterable, non-field-accessible type
    @test NN.flatten_sum_squares("abc") == 0.0
end

# ---- create_training_batch ----------------------------------------------------
@testset "create_training_batch deterministic/stochastic" begin
    # deterministic case (no shocks)
    scaler = NN.FeatureScaler(0.0f0, 1.0f0, 0.0f0, 1.0f0, false)
    a = 0.0:0.5:2.0
    G = make_G(a)

    batch, n = NN.create_training_batch(G, nothing, scaler; mode = :full)
    @test size(batch, 1) == 1                      # features = (a,)
    @test size(batch, 2) == length(a)              # samples
    @test n == length(a)

    # stochastic case (has shocks)
    z = [-0.7f0, 0.2f0]
    S = make_S(z)
    scaler = NN.FeatureScaler(0.0f0, 1.0f0, 0.0f0, 1.0f0, true)

    batch2, n2 = NN.create_training_batch(G, S, scaler; mode = :full)
    @test size(batch2, 1) == 2                     # features = (a, z)
    @test size(batch2, 2) == length(a) * length(z) # samples
    @test n2 == length(a) * length(z)

    # rejection sampler honours w-window and errors when infeasible
    tight_settings =
        NN.solver_settings((; w_min = 10.0f0, w_max = 10.5f0); has_shocks = true)
    @test_throws AssertionError NN.create_training_batch(
        G,
        S,
        scaler;
        mode = :rand,
        nsamples = 32,
        rng = Random.MersenneTwister(1),
        P_resid = (r = 0.01f0, β = 0.95f0, σ = 2.0f0, y = 0.0f0),
        settings = tight_settings,
    )
end

# ---- select_model / state_parameters / state_states / run_model --------------
@testset "state helpers and run_model" begin
    # fake state types
    Base.@kwdef struct S1
        model::Any
        parameters::Any
        states::Any
    end
    Base.@kwdef struct S2
        params::Any
        state::Any
    end
    s1 = S1(model = :m, parameters = :p, states = :s)
    s2 = S2(params = :pp, state = :ss)

    @test NN.select_model(:chain, s1) === :m
    @test NN.select_model(:chain, (;)) === :chain

    @test NN.state_parameters(s1) === :p
    @test NN.state_parameters(s2) === :pp
    @test NN.state_parameters((;)) === nothing

    @test NN.state_states(s1) === :s
    @test NN.state_states(s2) === :ss
    @test NN.state_states((;)) === nothing

    # run_model with and without params; tuple vs non-tuple outputs
    f_tuple = (X, ps, st) -> (X .* 2, :st1)
    f_plain = (X) -> X .+ 1
    X = rand(Float32, 3, 4)
    @test NN.run_model(f_tuple, :ps, :st, X) == X .* 2
    @test NN.run_model(f_plain, nothing, nothing, X) == X .+ 1
end

# ---- build_loss_function: deterministic and stochastic closures --------------
@testset "build_loss_function closures run" begin
    a = collect(0.0:0.5:2.0)
    G = make_G(a)

    P_resid = (r = 0.01f0, β = 0.95f0, σ = 2.0f0, y = 1.0f0)

    # Deterministic: new signature includes scaler and settings
    scaler = NN.FeatureScaler(G, nothing)
    settings = NN.solver_settings((; objective = :euler, hid1 = 4, hid2 = 4))
    loss_det = NN.build_loss_function(P_resid, G, nothing, scaler, settings)
    model = (X, ps, st) -> vec(X)
    ps = nothing
    st = nothing
    data = (reshape(Float32.(a), :, 1),)
    l1, st_out, meta = loss_det(model, ps, st, data)
    @test l1 isa Real
    @test st_out === nothing
    @test isa(meta, NamedTuple)

    # Stochastic
    z = [-0.7f0, 0.2f0]
    Π = Matrix{Float32}(I, length(z), length(z))
    S = (zgrid = Float32.(z), Π = Π)
    scaler_s = NN.FeatureScaler(G, S)
    settings_s =
        NN.solver_settings((; objective = :euler, hid1 = 4, hid2 = 4); has_shocks = true)
    loss_st = NN.build_loss_function(P_resid, G, S, scaler_s, settings_s)

    # Model should output (Na, Nz) consumption matrix (or NamedTuple with :Φ)
    Na, Nz = length(a), length(z)
    model_stoch = (X, ps, st) -> reshape(Float32.(1:Na*Nz), Na, Nz)

    # Input batch used by loss: features×samples; for stochastic full-grid we
    # typically use Na*Nz samples with two features
    X_dummy = rand(Float32, Na * Nz, 2)
    data2 = (X_dummy,)
    l2, st_out2, meta2 = loss_st(model_stoch, ps, st, data2)
    @test l2 isa Real
    @test st_out2 === nothing
    @test isa(meta2, NamedTuple)
end




# ---- train_consumption_network!: smoke + resample + early stop ---------------
@testset "train_consumption_network! smoke and early-stop" begin
    # Tiny deterministic problem; early stop immediately
    a = [0.0, 1.0]   # length 2 to match batch=2
    G = make_G(a)
    scaler = NN.FeatureScaler(G, nothing)

    # Minimal valid solver settings
    s = NN.solver_settings((;
        epochs = 1,
        batch = 2,
        lr = 1e-3,
        verbose = false,
        resample_every = 1,
        target_loss = Inf32,
        patience = 0,
        hid1 = 4,
        hid2 = 4,
        objective = :euler,
    ))

    net = NN.build_network(1, s)

    # Dummy residual parameters required by euler_resid_det_grid / stoch_grid
    P_resid = (r = 0.01f0, β = 0.95f0, σ = 2.0f0, y = 1.0f0)

    # Deterministic training run
    tr = NN.train_consumption_network!(
        net,
        s,
        scaler,
        P_resid,
        G,
        nothing,
        nothing,
        Random.GLOBAL_RNG,
    )
    @test tr.epochs_run ≤ 1
    @test tr.batch_size ≥ 1
    @test tr.batches_per_epoch ≥ 1

    # Stochastic variant with 2 inputs
    a = [0.0]          # <-- ensures Na=1
    G = make_G(a)

    z = [-0.7f0, 0.2f0]
    Π = Matrix{Float32}(I, length(z), length(z))
    S = (zgrid = Float32.(z), Π = Π)
    scaler_s = NN.FeatureScaler(G, S)

    s_stoch = NN.solver_settings(
        (;
            epochs = 1,
            batch = nothing,
            lr = 1e-3,
            verbose = false,
            resample_every = 1,
            target_loss = Inf32,
            patience = 0,
            hid1 = 4,
            hid2 = 4,
            objective = :euler,
        );
        has_shocks = true,
    )

    net2 = NN.build_network(2, s_stoch)
    # For stochastic problem training we expect full-batch; ensure call succeeds
    tr2 = NN.train_consumption_network!(
        net2,
        s_stoch,
        scaler_s,
        P_resid,
        G,
        S,
        nothing,
        Random.GLOBAL_RNG,
    )
    @test tr2.batch_size ≥ 1
    @test tr2.batches_per_epoch ≥ 1
end
