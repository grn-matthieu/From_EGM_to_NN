#!/usr/bin/env julia
# Example runs:
# julia --project scripts/run_nn_baseline.jl --config config/simple_stochastic.yaml --epochs 1 --batch 64 --seed 123
# julia --project scripts/run_nn_baseline.jl --config config/simple_stochastic.yaml --smoke
using Pkg;
Pkg.activate(dirname(@__DIR__));

using ThesisProject
using Random
using Dates
using Printf

const DEFAULT_CONFIG = joinpath(dirname(@__DIR__), "config", "simple_baseline.yaml")

usage() = """
Usage:
  julia --project scripts/run_nn_baseline.jl --config <path> [--epochs <Int>] [--lr <Float64>] [--batch <Int>] [--seed <Int>]

Options:
  --config   Path to YAML config (required)
  --epochs   NN training epochs (optional)
  --lr       Learning rate (optional)
  --batch    Batch size (optional)
  --seed     RNG seed (default 42)
  --smoke    Force quick run (epochs=1 if unset, batch<=64, CPU)
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

    t0 = Dates.now()
    model = build_model(cfg)
    method = build_method(cfg)
    sol = solve(model, method, cfg)
    elapsed = Dates.now() - t0
    wall_seconds = Dates.value(elapsed) / 1000

    epochs = get(cfg[:solver], :epochs, missing)
    loss = compute_training_loss(sol)

    epoch_str = epochs === missing ? "?" : string(epochs)
    @printf(
        "Epochs: %s | Final loss: %.4e | Wall time: %.2fs\n",
        epoch_str,
        loss,
        wall_seconds
    )

    # If smoke, print success line with CSV path (best-effort: pick latest log file)
    if getfield(opt, :smoke) === true
        logdir = joinpath(pwd(), "results", "nn", "baseline")
        csvpath = try
            if isdir(logdir)
                files = filter(
                    f -> endswith(lowercase(f), ".csv"),
                    readdir(logdir; join = true),
                )
                isempty(files) ? nothing : files[argmax(stat.(files) .|> x -> x.mtime)]
            else
                nothing
            end
        catch
            nothing
        end
        if csvpath === nothing
            println("SMOKE OK (no CSV found)")
        else
            println("SMOKE OK " * String(csvpath))
        end
    end

    return (; solution = sol, epochs = epochs, loss = loss, wall_time = wall_seconds)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
