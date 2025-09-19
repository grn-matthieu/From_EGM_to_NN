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
    )
end

function to_symbol_dict(d::AbstractDict)
    out = Dict{Symbol,Any}()
    for (k, v) in d
        out[k] = v isa AbstractDict ? to_symbol_dict(v) : v
    end
    return out
end

function apply_overrides!(cfg::AbstractDict, opt)::Nothing
    solver_cfg = get!(cfg, :solver, Dict{Symbol,Any}())
    if opt.epochs !== nothing
        solver_cfg[:epochs] = opt.epochs
    end
    if opt.lr !== nothing
        solver_cfg[:lr] = opt.lr
    end
    if opt.batch !== nothing
        solver_cfg[:batch] = opt.batch
    end

    if opt.seed !== nothing
        random_cfg = get!(cfg, :random, Dict{Symbol,Any}())
        random_cfg[:seed] = opt.seed
        cfg[:random] = random_cfg
    end

    # Optimizer + schedule overrides
    if opt.opt !== nothing
        solver_cfg[:optimizer] = String(opt.opt)
    end
    if opt.β1 !== nothing
        solver_cfg[:β1] = opt.β1
    end
    if opt.β2 !== nothing
        solver_cfg[:β2] = opt.β2
    end
    if opt.eps !== nothing
        solver_cfg[:eps] = opt.eps
    end
    if opt.mom !== nothing
        solver_cfg[:mom] = opt.mom
    end
    if opt.ρ !== nothing
        solver_cfg[:ρ] = opt.ρ
    end
    if opt.lr_schedule !== nothing
        solver_cfg[:lr_schedule] = Symbol(opt.lr_schedule)
    end
    if opt.η_min !== nothing
        solver_cfg[:η_min] = opt.η_min
    end
    if opt.step_size !== nothing
        solver_cfg[:step_size] = opt.step_size
    end
    if opt.γ !== nothing
        solver_cfg[:γ] = opt.γ
    end

    cfg[:solver] = solver_cfg
    return nothing
end

function ensure_nn_method!(cfg::AbstractDict)::Nothing
    solver_cfg = get!(cfg, :solver, Dict{Symbol,Any}())
    solver_cfg[:method] = :NN
    cfg[:solver] = solver_cfg
    cfg[:method] = :NN
    return nothing
end

function ensure_supported_shocks!(cfg::AbstractDict)::Nothing
    shocks_cfg = get(cfg, :shocks, nothing)
    if shocks_cfg isa AbstractDict && get(shocks_cfg, :active, false)
        shocks_dict = Dict{Symbol,Any}(shocks_cfg)
        shocks_dict[:active] = false
        cfg[:shocks] = shocks_dict
        @warn "NN baseline script forces shocks.active = false; stochastic NN kernel is not supported yet."
    end
    return nothing
end

function inject_seed!(cfg::AbstractDict, opt)::Int
    random_cfg = get!(cfg, :random, Dict{Symbol,Any}())
    seed =
        opt.seed !== nothing ? Int(opt.seed) :
        (haskey(random_cfg, :seed) ? Int(random_cfg[:seed]) : 1234)
    random_cfg[:seed] = seed
    cfg[:random] = random_cfg
    return seed
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
    cfg::AbstractDict;
    epochs = nothing,
    batch = nothing,
    optname = nothing,
    seed = nothing,
)
    cfg_local = deepcopy(cfg)
    ensure_supported_shocks!(cfg_local)
    ensure_nn_method!(cfg_local)

    solver_cfg = get!(cfg_local, :solver, Dict{Symbol,Any}())
    if epochs !== nothing
        solver_cfg[:epochs] = epochs
    end
    if batch !== nothing
        solver_cfg[:batch] = batch
    end
    if optname !== nothing
        solver_cfg[:optimizer] = optname
    end
    if seed !== nothing
        cfg_local[:random] = get!(cfg_local, :random, Dict{Symbol,Any}())
        cfg_local[:random][:seed] = seed
    end
    solver_cfg[:device] = :cpu
    cfg_local[:solver] = solver_cfg

    t0 = Dates.now()
    model = build_model(cfg_local)
    method = build_method(cfg_local)
    sol = solve(model, method, cfg_local)
    elapsed = Dates.now() - t0
    wall_seconds = Dates.value(elapsed) / 1000

    loss = compute_training_loss(sol)
    feas = isfinite(loss) ? 1.0 : 0.0

    return (
        method = :nn,
        loss = loss,
        feas = feas,
        wall_s = wall_seconds,
        epochs = get(solver_cfg, :epochs, nothing),
        batch = get(solver_cfg, :batch, nothing),
        opt = get(solver_cfg, :optimizer, ""),
    )
end


function run_egm(cfg::AbstractDict)
    cfg_local = deepcopy(cfg)
    cfg_local[:method] = :EGM
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


function run_vi(cfg::AbstractDict)
    cfg_local = deepcopy(cfg)
    cfg_local[:method] = :VI
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
    cfg_loaded = load_config(opt.config)
    cfg = to_symbol_dict(cfg_loaded)
    validate_config(cfg)

    # Enforce deterministic (current NN kernel limitation)
    ensure_supported_shocks!(cfg)

    # Force NN method
    ensure_nn_method!(cfg)

    # Apply CLI overrides directly to keys used by NN code
    solver_cfg = get!(cfg, :solver, Dict{Symbol,Any}())
    if opt.epochs !== nothing
        solver_cfg[:epochs] = opt.epochs
    end
    if opt.lr !== nothing
        solver_cfg[:lr] = opt.lr
    end
    if opt.batch !== nothing
        solver_cfg[:batch] = opt.batch
    end

    # Smoke mode: epochs=1 if unset; batch=min(batch,64) or 64 if unset; force CPU device
    if getfield(opt, :smoke) === true
        if !haskey(solver_cfg, :epochs)
            solver_cfg[:epochs] = 1
        end
        if haskey(solver_cfg, :batch)
            solver_cfg[:batch] = min(Int(solver_cfg[:batch]), 64)
        else
            solver_cfg[:batch] = 64
        end
        # Try to respect any device flag; harmless if unused elsewhere
        solver_cfg[:device] = :cpu
    end
    cfg[:solver] = solver_cfg

    # Seed handling
    seed = inject_seed!(cfg, opt)
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
        if haskey(res, :epochs)
            epoch_str = res[:epochs] === nothing ? "?" : string(res[:epochs])
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
            e = get(res, :epochs, "")
            b = get(res, :batch, "")
            o = get(res, :opt, "")
            s = get(cfg, :random, Dict{Symbol,Any}())[:seed]
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
