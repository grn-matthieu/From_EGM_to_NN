# test/solvers/nn/test_training_loop.jl
using ThesisProject
using ThesisProject.Determinism: make_rng
using LinearAlgebra: I
using Random

const NN = ThesisProject.NNKernel  # adjust if these funcs live under a submodule

# ---- helpers for grids/shocks -------------------------------------------------
make_G(a::AbstractVector) =
    Dict(:a => (grid = Float32.(a), min = minimum(a), max = maximum(a)))
make_S(z::AbstractVector) =
    (zgrid = Float32.(z), Π = Matrix{Float32}(I, length(z), length(z)))
make_params(; σ = 2.0f0, β = 0.95f0, r = 0.01f0, y = 0.0f0) = (σ = σ, β = β, r = r, y = y)

function dummy_settings(;
    has_shocks::Bool = false,
    objective::Symbol = :euler,
    use_cuda::Bool = false,
)
    return NN.NNSolverSettings(
        1,               # epochs
        nothing,         # batch_choice
        1e-3,            # learning_rate
        false,           # verbose
        1,               # resample_interval
        Float32(1e-3),   # target_loss
        0,               # patience
        (4, 4),          # hidden_sizes
        has_shocks,
        objective,
        0.5,             # v_h
        0.1f0,           # w_min
        4.0f0,           # w_max
        32,              # samples_per_epoch
        nothing,         # sigma_shocks
        use_cuda,
    )
end

# ---- solver_settings ----------------------------------------------------------
@testset "solver_settings parsing and clamping" begin
    # defaults
    s = NN.solver_settings(nothing)
    @test s.epochs == 1000
    @test s.batch_choice == 64
    @test s.learning_rate ≈ 1e-3
    @test s.verbose == false
    @test s.resample_interval == 1
    @test s.target_loss == Float32(2e-4)
    @test s.patience == 200
    @test s.hidden_sizes == (128, 128)
    @test s.objective == :euler_fb_aio
    @test s.v_h == 0.5
    @test s.w_min ≈ 0.1f0
    @test s.w_max ≈ 4.0f0
    @test s.sigma_shocks === nothing
    @test s.samples_per_epoch == 64
    @test s.use_cuda isa Bool

    # provided opts with negatives and zeros to trigger clamps
    opts = (;
        epochs = -3,
        batch = 0,
        lr = 5e-4,
        verbose = true,
        resample_every = -1,
        target_loss = 1e-5f0,
        patience = -7,
        hid1 = 0,
        hid2 = 1,
        objective = "euler",
        v_h = 0.1,
        w_min = 0.05,
        w_max = 10.0,
        sigma_shocks = 0.2,
        samples_per_epoch = 12,
    )
    s2 = NN.solver_settings(opts; has_shocks = true)
    @test s2.epochs == 0
    @test s2.batch_choice == 1
    @test s2.learning_rate ≈ 5e-4
    @test s2.verbose == true
    @test s2.resample_interval == 0
    @test s2.target_loss == Float32(1e-5)
    @test s2.patience == 0
    @test s2.hidden_sizes == (1, 1)
    @test s2.objective == :euler
    @test s2.v_h == 0.2             # clamped to lower bound
    @test s2.w_min ≈ 0.05f0
    @test s2.w_max ≈ 10.0f0
    @test s2.sigma_shocks ≈ 0.2
    @test s2.has_shocks
    @test s2.samples_per_epoch == 12
    @test s2.use_cuda == false
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
    params = make_params()
    P_resid = NN.scalar_params(params)

    # deterministic case (no shocks)
    settings_det = dummy_settings(has_shocks = false)
    G = make_G(0.0:0.5:2.0)
    sc = NN.FeatureScaler(params, G, nothing, settings_det)

    batch, n = NN.create_training_batch(
        G,
        nothing,
        sc;
        mode = :full,
        rng = make_rng(0),
        P_resid = P_resid,
        settings = settings_det,
    )
    @test size(batch, 1) == 2                      # features = (y, w)
    @test size(batch, 2) == length(G[:a].grid)      # samples
    @test n == length(G[:a].grid)
    @test all(abs.(batch) .<= 1.0f0)

    # stochastic case (has shocks)
    z = [-0.7f0, 0.2f0]
    S = make_S(z)
    settings_st = dummy_settings(has_shocks = true)
    sc_s = NN.FeatureScaler(params, G, S, settings_st)

    batch2, n2 = NN.create_training_batch(
        G,
        S,
        sc_s;
        mode = :full,
        rng = make_rng(1),
        P_resid = P_resid,
        settings = settings_st,
    )
    @test size(batch2, 1) == 2                     # features = (y, w)
    @test size(batch2, 2) == length(G[:a].grid) * length(S.zgrid)
    @test n2 == length(G[:a].grid) * length(S.zgrid)
    @test all(abs.(batch2) .<= 1.0f0)

    # rejection sampler honours w-window and errors when infeasible
    tight_settings = NN.NNSolverSettings(
        settings_st.epochs,
        settings_st.batch_choice,
        settings_st.learning_rate,
        settings_st.verbose,
        settings_st.resample_interval,
        settings_st.target_loss,
        settings_st.patience,
        settings_st.hidden_sizes,
        true,
        settings_st.objective,
        settings_st.v_h,
        10.0f0,
        10.5f0,
        settings_st.samples_per_epoch,
        settings_st.sigma_shocks,
        settings_st.use_cuda,
    )
    @test_throws AssertionError NN.create_training_batch(
        G,
        S,
        sc_s;
        mode = :rand,
        nsamples = 32,
        rng = Random.MersenneTwister(1),
        P_resid = P_resid,
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
    params = make_params()
    P_resid = NN.scalar_params(params)
    G = make_G(0.0:0.5:2.0)

    # Deterministic: new signature includes scaler and settings
    settings = NN.solver_settings((; objective = :euler, hid1 = 4, hid2 = 4))
    scaler = NN.FeatureScaler(params, G, nothing, settings)
    rng_loss = make_rng(2)
    loss_det =
        NN.build_loss_function(P_resid, G, nothing, scaler, settings, rng_loss, nothing)
    batch_det, _ = NN.create_training_batch(
        G,
        nothing,
        scaler;
        mode = :full,
        rng = make_rng(2),
        P_resid = P_resid,
        settings = settings,
    )
    model = (X, ps, st) -> reshape(fill(Float32(0.5), size(X, 2)), 1, :)
    ps = nothing
    st = nothing
    data = (batch_det,)
    l1, st_out, meta = loss_det(model, ps, st, data)
    @test l1 isa Real
    @test st_out === st
    @test isa(meta, NamedTuple)

    # Stochastic
    z = [-0.7f0, 0.2f0]
    S = make_S(z)
    settings_s =
        NN.solver_settings((; objective = :euler, hid1 = 4, hid2 = 4); has_shocks = true)
    scaler_s = NN.FeatureScaler(params, G, S, settings_s)
    rng_loss_s = make_rng(3)
    loss_st =
        NN.build_loss_function(P_resid, G, S, scaler_s, settings_s, rng_loss_s, nothing)

    batch_st, _ = NN.create_training_batch(
        G,
        S,
        scaler_s;
        mode = :full,
        rng = make_rng(3),
        P_resid = P_resid,
        settings = settings_s,
    )
    Na, Nz = length(G[:a].grid), length(S.zgrid)
    model_stoch = (X, ps, st) -> reshape(Float32.(1:(Na*Nz)), 1, :)
    data2 = (batch_st,)
    l2, st_out2, meta2 = loss_st(model_stoch, ps, st, data2)
    @test l2 isa Real
    @test st_out2 === st
    @test isa(meta2, NamedTuple)
end

# ---- train_consumption_network!: smoke + resample + early stop ---------------
@testset "train_consumption_network! smoke and early-stop" begin
    params = make_params()
    P_resid = NN.scalar_params(params)

    # Tiny deterministic problem; early stop immediately
    a = [0.0f0, 1.0f0]   # length 2 to match batch=2
    G = make_G(a)

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
        samples_per_epoch = 2,
    ))
    scaler = NN.FeatureScaler(params, G, nothing, s)
    net = NN.build_network(NN.input_dimension(nothing), s)

    tr = NN.train_consumption_network!(net, s, scaler, P_resid, G, nothing, make_rng(4))
    @test tr.epochs_run ≤ s.epochs
    @test tr.batch_size ≥ 1
    @test tr.batches_per_epoch ≥ 1

    # Stochastic variant with 2 inputs
    a_stoch = [0.0f0]
    Gs = make_G(a_stoch)
    z = [-0.7f0, 0.2f0]
    S = make_S(z)

    s_stoch = NN.solver_settings(
        (;
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
            samples_per_epoch = 4,
        );
        has_shocks = true,
    )
    scaler_s = NN.FeatureScaler(params, Gs, S, s_stoch)
    net2 = NN.build_network(NN.input_dimension(S), s_stoch)

    tr2 =
        NN.train_consumption_network!(net2, s_stoch, scaler_s, P_resid, Gs, S, make_rng(5))
    @test tr2.batch_size ≥ 1
    @test tr2.batches_per_epoch ≥ 1
end
