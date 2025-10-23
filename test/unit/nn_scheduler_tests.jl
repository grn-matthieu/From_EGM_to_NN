using Test
using Random
using Statistics: mean
using StableRNGs
using Lux
using Optimisers

using ThesisProject
import ThesisProject.NNKernel:
    solver_settings, cosine_learning_rate, apply_optimizer_learning_rate!

@testset "NN cosine annealing schedule" begin
    opts = (
        epochs = 6,
        lr = 0.01,
        lr_min = 0.001,
        lr_max = 0.02,
        lr_decay_horizon = 3,
        warmup_epochs = 2,
        batch = 8,
        samples_per_epoch = 32,
    )
    settings = solver_settings(
        opts,
        nothing,
        nothing,
        nothing;
        has_shocks = false,
        objective_default = :euler_residual,
    )

    @test settings.lr_min == opts.lr_min
    @test settings.lr_max == opts.lr_max
    @test settings.warmup_epochs == opts.warmup_epochs

    schedule = [cosine_learning_rate(settings, ep) for ep = 1:opts.epochs]
    expected_warmup =
        settings.lr_min + (settings.lr_max - settings.lr_min) / settings.warmup_epochs
    @test schedule[1] ≈ expected_warmup
    @test schedule[settings.warmup_epochs] ≈ settings.lr_max
    @test schedule[end] == settings.lr_min

    rng = StableRNG(1234)
    model = Chain(Dense(1, 4, tanh), Dense(4, 1))
    ps, st = Lux.setup(rng, model)
    opt = Optimisers.OptimiserChain(
        Optimisers.ClipGrad(0.1),
        Optimisers.Adam(settings.lr_max),
    )
    train_state = Lux.Training.TrainState(model, ps, st, opt)
    data = (randn(rng, Float32, 1, 16),)

    function dummy_loss(model, ps, st, batch)
        # Lux models can return (preds, new_state). Unpack to get matrix preds
        preds, new_st = model(batch[1], ps, st)
        # preds is a matrix; use broadcasting so abs2 applies elementwise
        return mean(abs2.(preds)), new_st, nothing
    end

    for epoch = 1:opts.epochs
        lr = schedule[epoch]
        train_state = apply_optimizer_learning_rate!(train_state, lr)
        _, loss, _, train_state =
            Lux.Training.single_train_step!(Lux.AutoZygote(), dummy_loss, data, train_state)
        @test isfinite(loss)
        # Lux.Training.TrainState uses the field name `:optimizer`
        opt_chain = getfield(train_state, :optimizer)
        # OptimiserChain stores optimisers in the `opts` field
        adam_stage = opt_chain.opts[end]
        lr_field =
            hasproperty(adam_stage, :eta) ? :eta :
            (hasproperty(adam_stage, :lr) ? :lr : nothing)
        if lr_field !== nothing
            @test getfield(adam_stage, lr_field) ≈ lr
        end
    end
end
