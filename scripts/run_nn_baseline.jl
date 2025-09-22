#!/usr/bin/env julia
# Example runs:
# julia --project scripts/run_nn_baseline.jl --config config/simple_stochastic.yaml --epochs 1 --batch 64 --seed 123
# julia --project scripts/run_nn_baseline.jl --config config/simple_stochastic.yaml --smoke
#
# Benchmark mixed precision:
# julia --project scripts/run_nn_baseline.jl --config config/simple_stochastic.yaml --bench-mp
using Pkg;
Pkg.activate(dirname(@__DIR__));

using ThesisProject
using Random
using Dates
using Printf

include(joinpath(@__DIR__, "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

const DEFAULT_CONFIG = joinpath(dirname(@__DIR__), "config", "simple_baseline.yaml")

usage() = """
Usage:
    julia --project scripts/run_nn_baseline.jl --config <path> [--epochs <Int>] [--lr <Float64>] [--batch <Int>] [--seed <Int>] \
                 [--opt <adam|rmsprop|sgd>] [--β1 <Float64>] [--β2 <Float64>] [--eps <Float64>] \
                 [--mom <Float64>] [--ρ <Float64>] [--lr_schedule <none|cosine|step>] \
                 [--η_min <Float64>] [--step_size <Int>] [--γ <Float64>]

Options:
    --config   Path to YAML config (required)
    --epochs   NN training epochs (optional)
    --lr       Learning rate (optional)
    --batch    Batch size (optional)
    --seed     RNG seed (default 42)
    --smoke    Force quick run (epochs=1 if unset, batch<=64, CPU)
    --opt      Optimizer (adam|rmsprop|sgd)
    --β1    Adam/RMSProp β1
    --β2    Adam β2
    --eps      Epsilon for Adam/RMSProp/SGD
    --mom      Momentum for SGD
    --ρ      Ρ for RMSProp
    --lr_schedule  Learning rate schedule (none|cosine|step)
    --η_min  Min LR for cosine
    --step_size Step size for step schedule
    --γ    Decay factor for step schedule
    --device  Device to run on (cpu|cuda). If unset, uses config value or defaults to cpu in smoke mode.
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
        # optimizer and schedule
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
        # optimizer & schedule
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
    opt.opt !== nothing && (solver_overrides[:optimizer] = String(opt.opt))
    opt.β1 !== nothing && (solver_overrides[:β1] = opt.β1)
    opt.β2 !== nothing && (solver_overrides[:β2] = opt.β2)
    opt.eps !== nothing && (solver_overrides[:eps] = opt.eps)
    opt.mom !== nothing && (solver_overrides[:mom] = opt.mom)
    opt.ρ !== nothing && (solver_overrides[:ρ] = opt.ρ)
    opt.lr_schedule !== nothing &&
        (solver_overrides[:lr_schedule] = Symbol(opt.lr_schedule))
    opt.η_min !== nothing && (solver_overrides[:η_min] = opt.η_min)
    opt.step_size !== nothing && (solver_overrides[:step_size] = opt.step_size)
    opt.γ !== nothing && (solver_overrides[:γ] = opt.γ)
    opt.device !== nothing && (solver_overrides[:device] = Symbol(opt.device))

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
        if !(haskey(solver_overrides, :device) || hasproperty(solver_cfg, :device))
            solver_overrides[:device] = :cpu
        end
    end

    if !isempty(solver_overrides)
        cfg = merge_section(cfg, :solver, dict_to_namedtuple(solver_overrides))
    end
    return cfg
end

function ensure_nn_method(cfg::NamedTuple)
    cfg = merge_section(cfg, :solver, (; method = :NN))
    return merge_config(cfg, (; method = :NN))
end

function ensure_supported_shocks(cfg::NamedTuple)
    active = get_nested(cfg, (:shocks, :active), false)
    if active === true
        cfg = merge_section(cfg, :shocks, (; active = false))
        @warn "NN baseline script forces shocks.active = false; stochastic NN kernel is not supported yet."
    end
    return cfg
end

function ensure_seed(cfg::NamedTuple, opt)
    existing = get_nested(cfg, (:random, :seed), nothing)
    seed =
        opt.seed !== nothing ? Int(opt.seed) : (existing === nothing ? 1234 : Int(existing))
    cfg = merge_section(cfg, :random, (; seed = seed))
    return cfg, seed
end

function compute_training_loss(sol)::Float64
    c_policy = sol.policy[:c]
    residuals = getfield(c_policy, :euler_errors_mat)
    residuals = residuals === nothing ? getfield(c_policy, :euler_errors) : residuals

    if residuals === nothing
        return NaN
    end

    res_vec = residuals isa AbstractArray ? vec(residuals) : [residuals]
    if isempty(res_vec)
        return NaN
    end

    res_data = Float64.(res_vec)
    return sum(abs2, res_data) / length(res_data)
end


"""
run_nn(cfg; epochs, batch, opt, seed) -> NamedTuple

Builds model and NN method, runs training and returns a NamedTuple with metrics.
"""
function run_nn(
    cfg::NamedTuple;
    epochs = nothing,
    batch = nothing,
    optname = nothing,
    seed = nothing,
)
    solver_overrides = Dict{Symbol,Any}()
    epochs !== nothing && (solver_overrides[:epochs] = epochs)
    batch !== nothing && (solver_overrides[:batch] = batch)
    optname !== nothing && (solver_overrides[:optimizer] = optname)

    cfg_local = ensure_supported_shocks(cfg)
    cfg_local = ensure_nn_method(cfg_local)
    if !isempty(solver_overrides)
        cfg_local = merge_section(cfg_local, :solver, dict_to_namedtuple(solver_overrides))
    end
    if seed !== nothing
        cfg_local = merge_section(cfg_local, :random, (; seed = seed))
    end

    t0 = Dates.now()
    model = build_model(cfg_local)
    method = build_method(cfg_local)
    sol = solve(model, method, cfg_local)
    elapsed = Dates.now() - t0
    wall_seconds = Dates.value(elapsed) / 1000

    loss = compute_training_loss(sol)
    feas = isfinite(loss) ? 1.0 : 0.0

    solver_snapshot = maybe_namedtuple(get_nested(cfg_local, (:solver,), NamedTuple()))
    return (
        method = :nn,
        loss = loss,
        feas = feas,
        wall_s = wall_seconds,
        epochs = hasproperty(solver_snapshot, :epochs) ? solver_snapshot.epochs : nothing,
        batch = hasproperty(solver_snapshot, :batch) ? solver_snapshot.batch : nothing,
        opt = hasproperty(solver_snapshot, :optimizer) ? solver_snapshot.optimizer : "",
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
    loss = compute_training_loss(sol)
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
    loss = compute_training_loss(sol)
    feas = isfinite(loss) ? 1.0 : 0.0
    return (method = :vi, loss = loss, feas = feas, wall_s = wall_seconds)
end

function main(args::Vector{String} = ARGS)
    opt = try
        parse_args(args)
    catch err
        # parse error/help: already printed usage; rethrow to exit non-zero
        rethrow(err)
    end
    cfg = load_config(opt.config)

    cfg = ensure_supported_shocks(cfg)
    cfg = ensure_nn_method(cfg)
    cfg = apply_overrides(cfg, opt)

    cfg, seed = ensure_seed(cfg, opt)
    Random.seed!(seed)

    # Determine which methods to run
    method_names = if getfield(opt, :all_methods) === true
        ["egm", "vi", "nn"]
    elseif opt.methods !== nothing
        split(String(opt.methods), ',') .|> strip .|> lowercase
    else
        ["nn"]
    end

    # Prepare results directory and CSV
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
                optname = opt.opt,
                seed = opt.seed,
            )
        else
            @warn "Unknown method: $mclean — skipping"
            continue
        end

        push!(results, res)

        # Print summary line
        if hasproperty(res, :epochs)
            epoch_val = getproperty(res, :epochs)
            epoch_str = epoch_val === nothing ? "?" : string(epoch_val)
        else
            epoch_str = "?"
        end
        @printf(
            "Method: %s | Loss: %.4e | Wall: %.2fs | Epochs: %s\n",
            string(res[:method]),
            res[:loss],
            res[:wall_s],
            epoch_str
        )

        # Append to CSV
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
