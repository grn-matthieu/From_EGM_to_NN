#!/usr/bin/env julia
using ThesisProject
using ThesisProject.Determinism: make_master_rng, derive_rng
using Random, Statistics, Printf
using Lux

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTPATH = joinpath(ROOT, "outputs", "vh_sweep_results.txt")

function run_sweep(;
    cfg_path = joinpath(ROOT, "config", "smoke_cfg_stoch.yaml"),
    candidates = [0.2, 0.5, 1.0, 2.0, 5.0],
    seed = 1234,
    short_epochs = 5000,
)

    cfg = ThesisProject.load_config(cfg_path)
    model = ThesisProject.build_model(cfg)
    P = ThesisProject.get_params(model)
    G = ThesisProject.get_grids(model)
    S = ThesisProject.get_shocks(model)
    U = ThesisProject.get_utility(model)

    results = Dict()

    master = make_master_rng(seed)

    for v in candidates
        @info "Running v_h = $v"
        # build a cfg copy with solver overrides
        solver_nt = merge(cfg.solver, (; v_h = v, epochs = short_epochs))
        cfg_local = merge(cfg, (solver = solver_nt,))

        # Build NN training settings using NNKernel.solver_settings
        opts = solver_nt
        has_shocks = !isnothing(S)
        settings = ThesisProject.NNKernel.solver_settings(opts; has_shocks = has_shocks)

        # scaler / params
        P_resid = ThesisProject.NNKernel.scalar_params(P)
        scaler = ThesisProject.NNKernel.FeatureScaler(P, G, S, settings)

        # build network and model_cfg
        input_dim = ThesisProject.NNKernel.input_dimension(S)
        chain =
            ThesisProject.NNKernel.build_dual_head_network(input_dim, settings.hidden_sizes)
        model_cfg =
            ThesisProject.NNKernel.build_model_config(P, U, scaler, P_resid, settings)

        # train
        train_rng = derive_rng(master, :train)
        training_result = ThesisProject.NNKernel.train_consumption_network!(
            chain,
            settings,
            scaler,
            P_resid,
            G,
            S,
            train_rng,
            model_cfg,
        )

        best_state = training_result.best_state
        trained_model = ThesisProject.NNKernel.select_model(chain, best_state)
        params = ThesisProject.NNKernel.state_parameters(best_state)
        states = ThesisProject.NNKernel.state_states(best_state)

        # diagnostics RNGs
        diag_rng = derive_rng(master, :diagnostics)

        # 1) FB diagnostics on a validation batch (4096 or smaller)
        val_ns = min(4096, settings.samples_per_epoch)
        val_batch, _ = ThesisProject.NNKernel.create_training_batch(
            G,
            S,
            scaler;
            mode = :rand,
            nsamples = val_ns,
            rng = diag_rng,
            P_resid = P_resid,
            settings = settings,
        )

        loss_val, st_pack = ThesisProject.NNKernel.loss_euler_fb_aio!(
            trained_model,
            params,
            states,
            val_batch,
            model_cfg,
            diag_rng,
        )
        _, aux = st_pack
        kt_mean = hasproperty(aux, :kt_mean) ? aux.kt_mean : missing
        aio_mean = hasproperty(aux, :aio_mean) ? aux.aio_mean : missing
        max_abs_q = hasproperty(aux, :max_abs_q) ? aux.max_abs_q : missing

        # 2) mean |q - h| using MC samples (N = 8192)
        mc_ns = 8192
        mc_batch, _ = ThesisProject.NNKernel.create_training_batch(
            G,
            S,
            scaler;
            mode = :rand,
            nsamples = mc_ns,
            rng = derive_rng(master, :mc),
            P_resid = P_resid,
            settings = settings,
        )
        # denormalize features to compute w0,y0 as in eval function
        y0 = ((mc_batch[1, :] .+ 1.0f0) ./ 2.0f0) .* scaler.y_range .+ scaler.y_min
        w0 = ((mc_batch[2, :] .+ 1.0f0) ./ 2.0f0) .* scaler.w_range .+ scaler.w_min

        out, _ = Lux.apply(trained_model, mc_batch, params, states)
        c0 = vec(ThesisProject.NNKernel.phi_to_consumption(out[:Φ], w0; min_c = 1.0f-3))
        h_pred = vec(ThesisProject.NNKernel.ensure_row(out[:h]))

        # draw one-step shocks and compute c1 to form q
        μ = Float32(P_resid.y)
        z0 = log.(y0) .- μ
        ρ = Float32(P.ρ)
        σϵ =
            settings.sigma_shocks === nothing ? Float32(P.σ_shock) :
            Float32(settings.sigma_shocks)
        β = Float32(P.β)
        Rg = 1.0f0 + Float32(P.r)

        ε = randn(derive_rng(master, :mc2), Float32, mc_ns)
        z1 = @. ρ * z0 + σϵ * ε
        y1 = exp.(μ .+ z1)
        a1 = @. w0 - c0
        w1 = @. Rg * a1 + y1

        X1 = vcat(reshape(y1, 1, :), reshape(w1, 1, :))
        NX1 = ThesisProject.NNKernel.normalize_feature_batch(scaler, X1)
        out1, _ = Lux.apply(trained_model, NX1, params, states)
        c1 = vec(ThesisProject.NNKernel.phi_to_consumption(out1[:Φ], w1; min_c = 1.0f-3))

        uprime = ThesisProject.get_utility(model).u_prime
        q = @. β * Rg * uprime(c1) / uprime(c0)
        mean_qdiff = mean(abs.(q .- h_pred))

        # 3) GH evaluation of Euler residuals (accurate): use eval_euler_residuals_gh
        gh_diag = ThesisProject.NNKernel.eval_euler_residuals_gh(
            trained_model,
            params,
            states,
            P_resid,
            U,
            scaler,
            settings;
            N = 8192,
            rng = derive_rng(master, :eval_gh),
            G = G,
            S = S,
            P = P,
        )
        resid_stats = hasproperty(gh_diag, :stats) ? gh_diag.stats : gh_diag[:stats]

        results[v] = (
            kt_mean = kt_mean,
            aio_mean = aio_mean,
            max_abs_q = max_abs_q,
            mean_qdiff = mean_qdiff,
            resid_stats = resid_stats,
        )

        @info(
            "v_h=$v -> kt=$(kt_mean) aio=$(aio_mean) mean_qdiff=$(mean_qdiff) resid_mean=$(resid_stats.mean)"
        )
    end

    # save results to text file (repr)
    isdir(dirname(OUTPATH)) || mkpath(dirname(OUTPATH))
    open(OUTPATH, "w") do io
        write(io, repr(results))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_sweep()
end
