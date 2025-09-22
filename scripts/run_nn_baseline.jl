#!/usr/bin/env julia
using Pkg;
Pkg.activate(dirname(@__DIR__));

using ThesisProject
using ThesisProject.NNKernel: solve_nn
using Random
using Dates
using Printf

include(joinpath(@__DIR__, "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

const DEFAULT_CONFIG = joinpath(dirname(@__DIR__), "config", "simple_baseline.yaml")

usage() = """
Usage:
    julia --project scripts/run_nn_baseline.jl --config <path> [--epochs <Int>] [--lr <Float64>] [--batch <Int>] [--seed <Int>] \\
                 [--opt <adam|rmsprop|sgd>] [--β1 <Float64>] [--β2 <Float64>] [--eps <Float64>] \\
                 [--mom <Float64>] [--ρ <Float64>] [--lr_schedule <none|cosine|step>] \\
                 [--η_min <Float64>] [--step_size <Int>] [--γ <Float64>]

Options:
    --config   Path to YAML config (required)
    --epochs   NN training epochs (optional)
    --lr       Learning rate (optional)
    --batch    Batch size (optional)
    --seed     RNG seed (default 42)
    --smoke    Force quick run (epochs=1 if unset, batch<=64, CPU)
    --opt      Optimizer (adam|rmsprop|sgd)  [ignored by current NN kernel]
    --β1       Adam/RMSProp β1               [ignored by current NN kernel]
    --β2       Adam β2                       [ignored by current NN kernel]
    --eps      Epsilon                       [ignored by current NN kernel]
    --mom      Momentum                      [ignored by current NN kernel]
    --ρ        Ρ for RMSProp                 [ignored by current NN kernel]
    --lr_schedule  LR schedule               [ignored by current NN kernel]
    --η_min    Min LR for cosine             [ignored by current NN kernel]
    --step_size Step size for step           [ignored by current NN kernel]
    --γ        Decay factor                  [ignored by current NN kernel]
    --device   Device (cpu|cuda)             [ignored by current NN kernel]
    --all-methods   Run benchmarks for all supported methods (egm,vi,nn)
    --methods       Comma-separated list of methods to run (egm,vi,nn)
"""

function parse_args(argv::Vector{String})::NamedTuple
    if isempty(argv)
        println(usage())
        error("--config is required")
    end
    opt = Dict{String,Any}(
        "config" => nothing,
        "epochs" => nothing,
        "lr" => nothing,
        "batch" => nothing,
        "seed" => 42,
        "smoke" => false,
        # legacy knobs kept for CLI compatibility; NN kernel ignores them
        "opt" => nothing,
        "β1" => nothing,
        "β2" => nothing,
        "eps" => nothing,
        "mom" => nothing,
        "ρ" => nothing,
        "lr_schedule" => nothing,
        "η_min" => nothing,
        "step_size" => nothing,
        "γ" => nothing,
        "all_methods" => false,
        "methods" => nothing,
        "device" => nothing,
    )
    i = 1
    while i ≤ length(argv)
        a = argv[i]
        if a == "--help" || a == "-h"
            println(usage())
            error("Help requested")
        end
        a == "--config" && (opt["config"] = argv[i+=1]; i += 1; continue)
        a == "--epochs" && (opt["epochs"] = parse(Int, argv[i+=1]); i += 1; continue)
        a == "--lr" && (opt["lr"] = parse(Float64, argv[i+=1]); i += 1; continue)
        a == "--batch" && (opt["batch"] = parse(Int, argv[i+=1]); i += 1; continue)
        a == "--seed" && (opt["seed"] = parse(Int, argv[i+=1]); i += 1; continue)
        a == "--smoke" && (opt["smoke"] = true; i += 1; continue)
        a == "--opt" && (opt["opt"] = argv[i+=1]; i += 1; continue)
        a == "--β1" && (opt["β1"] = parse(Float64, argv[i+=1]); i += 1; continue)
        a == "--β2" && (opt["β2"] = parse(Float64, argv[i+=1]); i += 1; continue)
        a == "--eps" && (opt["eps"] = parse(Float64, argv[i+=1]); i += 1; continue)
        a == "--mom" && (opt["mom"] = parse(Float64, argv[i+=1]); i += 1; continue)
        a == "--ρ" && (opt["ρ"] = parse(Float64, argv[i+=1]); i += 1; continue)
        a == "--lr_schedule" && (opt["lr_schedule"] = argv[i+=1]; i += 1; continue)
        a == "--η_min" && (opt["η_min"] = parse(Float64, argv[i+=1]); i += 1; continue)
        a == "--step_size" && (opt["step_size"] = parse(Int, argv[i+=1]); i += 1; continue)
        a == "--γ" && (opt["γ"] = parse(Float64, argv[i+=1]); i += 1; continue)
        a == "--device" && (opt["device"] = argv[i+=1]; i += 1; continue)
        a == "--all-methods" && (opt["all_methods"] = true; i += 1; continue)
        a == "--methods" && (opt["methods"] = argv[i+=1]; i += 1; continue)
        error("Unknown arg: $a")
    end
    isnothing(opt["config"]) && error("--config is required")
    return (
        config = String(opt["config"]),
        epochs = opt["epochs"],
        lr = opt["lr"],
        batch = opt["batch"],
        seed = opt["seed"],
        smoke = opt["smoke"],
        opt = opt["opt"],
        β1 = opt["β1"],
        β2 = opt["β2"],
        eps = opt["eps"],
        mom = opt["mom"],
        ρ = opt["ρ"],
        lr_schedule = opt["lr_schedule"],
        η_min = opt["η_min"],
        step_size = opt["step_size"],
        γ = opt["γ"],
        all_methods = opt["all_methods"],
        methods = opt["methods"],
        device = opt["device"],
    )
end

function apply_overrides(cfg::NamedTuple, opt)
    solver_cfg = maybe_namedtuple(get_nested(cfg, (:solver,), NamedTuple()))
    solver_overrides = Dict{Symbol,Any}()

    opt.epochs !== nothing && (solver_overrides[:epochs] = opt.epochs)
    opt.lr !== nothing && (solver_overrides[:lr] = opt.lr)
    opt.batch !== nothing && (solver_overrides[:batch] = opt.batch)

    if getfield(opt, :smoke) === true
        if !haskey(solver_overrides, :epochs) && !hasproperty(solver_cfg, :epochs)
            solver_overrides[:epochs] = 1
        end
        if haskey(solver_overrides, :batch)
            solver_overrides[:batch] = min(Int(solver_overrides[:batch]), 64)
        else
            batch_base = hasproperty(solver_cfg, :batch) ? Int(solver_cfg.batch) : 64
            solver_overrides[:batch] = min(batch_base, 64)
        end
        # device ignored by current NN kernel; keep config untouched
    end

    if !isempty(solver_overrides)
        cfg = merge_section(cfg, :solver, dict_to_namedtuple(solver_overrides))
    end
    return cfg
end

function ensure_nn_method(cfg::NamedTuple)
    # Keep solver.method=:NN for bookkeeping, though NN kernel no longer reads it.
    cfg = merge_section(cfg, :solver, (; method = :NN))
    return merge_config(cfg, (; method = :NN))
end

function ensure_seed(cfg::NamedTuple, opt)
    existing = get_nested(cfg, (:random, :seed), nothing)
    seed =
        opt.seed !== nothing ? Int(opt.seed) : (existing === nothing ? 1234 : Int(existing))
    cfg = merge_section(cfg, :random, (; seed = seed))
    return cfg, seed
end

compute_training_loss_from_resid(resid) =
    resid === nothing ? NaN :
    (isempty(resid) ? NaN : sum(abs2, Float64.(vec(resid))) / length(vec(resid)))

"""
run_nn(cfg; epochs, batch, lr, seed) -> NamedTuple
Builds model, runs NN kernel training, returns metrics.
"""
function run_nn(
    cfg::NamedTuple;
    epochs = nothing,
    batch = nothing,
    lr = nothing,
    seed = nothing,
)
    # Build model from config, then call NNKernel.solve_nn with opts
    cfg_local = ensure_nn_method(cfg)
    if seed !== nothing
        cfg_local = merge_section(cfg_local, :random, (; seed = seed))
    end
    model = build_model(cfg_local)
    opts = (;
        epochs = something(epochs, get_nested(cfg_local, (:solver, :epochs), 1000)),
        batch = something(batch, get_nested(cfg_local, (:solver, :batch), nothing)),
        lr = something(lr, get_nested(cfg_local, (:solver, :lr), 1e-4)),
        verbose = true,
    )

    t0 = Dates.now()
    sol = solve_nn(model; opts = opts)
    elapsed = Dates.now() - t0
    _ = Dates.value(elapsed) # wall time comes from sol.opts.runtime

    loss = compute_training_loss_from_resid(sol.resid)
    feas = isfinite(loss) ? 1.0 : 0.0

    return (
        method = :nn,
        loss = loss,
        feas = feas,
        wall_s = getfield(sol.opts, :runtime),
        epochs = getfield(sol.opts, :epochs),
        batch = getfield(sol.opts, :batch),
        opt = "", # not applicable in current kernel
    )
end

function run_egm(cfg::NamedTuple)
    cfg_local = merge_section(cfg, :solver, (; method = :EGM))
    cfg_local = merge_config(cfg_local, (; method = :EGM))
    t0 = Dates.now()
    model = build_model(cfg_local)
    method = build_method(cfg_local)
    sol = solve(model, method, cfg_local)
    elapsed = Dates.now() - t0
    wall_seconds = Dates.value(elapsed) / 1000
    # EGM/VI may not expose residuals; reuse old helper if available
    loss = NaN
    feas = isfinite(loss) ? 1.0 : 0.0
    return (method = :egm, loss = loss, feas = feas, wall_s = wall_seconds)
end

function run_vi(cfg::NamedTuple)
    cfg_local = merge_section(cfg, :solver, (; method = :VI))
    cfg_local = merge_config(cfg_local, (; method = :VI))
    t0 = Dates.now()
    model = build_model(cfg_local)
    method = build_method(cfg_local)
    sol = solve(model, method, cfg_local)
    elapsed = Dates.now() - t0
    wall_seconds = Dates.value(elapsed) / 1000
    loss = NaN
    feas = isfinite(loss) ? 1.0 : 0.0
    return (method = :vi, loss = loss, feas = feas, wall_s = wall_seconds)
end

function main(args::Vector{String} = ARGS)
    opt = try
        parse_args(args)
    catch err
        rethrow(err)
    end
    cfg = load_config(opt.config)
    cfg = ensure_nn_method(cfg)
    cfg = apply_overrides(cfg, opt)

    cfg, seed = ensure_seed(cfg, opt)
    Random.seed!(seed)

    method_names = if getfield(opt, :all_methods) === true
        ["egm", "vi", "nn"]
    elseif opt.methods !== nothing
        split(String(opt.methods), ',') .|> strip .|> lowercase
    else
        ["nn"]
    end

    logdir = joinpath(pwd(), "results", "benchmarks")
    isdir(logdir) || mkpath(logdir)
    timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")
    csvpath = joinpath(logdir, "bench_$(timestamp).csv")
    open(csvpath, "w") do io
        println(io, "method,loss,feas,wall_s,epochs,batch,opt,seed,config")
    end

    results = Vector{Any}()
    for m in method_names
        mclean = lowercase(strip(m))
        if mclean == "egm"
            res = run_egm(cfg)
        elseif mclean == "vi"
            res = run_vi(cfg)
        elseif mclean == "nn"
            res = run_nn(
                cfg;
                epochs = opt.epochs,
                batch = opt.batch,
                lr = opt.lr,
                seed = opt.seed,
            )
        else
            @warn "Unknown method: $mclean — skipping"
            continue
        end

        push!(results, res)

        epoch_str =
            hasproperty(res, :epochs) && res.epochs !== nothing ? string(res.epochs) : "?"
        @printf "Method: %s | Loss: %.4e | Wall: %.2fs | Epochs: %s\n" string(res[:method]) res[:loss] res[:wall_s] epoch_str

        open(csvpath, "a") do io
            e = hasproperty(res, :epochs) ? res.epochs : ""
            b = hasproperty(res, :batch) ? res.batch : ""
            o = hasproperty(res, :opt) ? res.opt : ""
            s = get_nested(cfg, (:random, :seed), "")
            println(
                io,
                join(
                    [
                        string(res[:method]),
                        string(res[:loss]),
                        string(res[:feas]),
                        string(res[:wall_s]),
                        string(e),
                        string(b),
                        string(o),
                        string(s),
                        opt.config,
                    ],
                    ',',
                ),
            )
        end
    end

    println("Wrote benchmark CSV: $csvpath")
    return (; results = results, csv = csvpath)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
