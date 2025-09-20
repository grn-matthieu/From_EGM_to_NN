#!/usr/bin/env julia
using Pkg;
Pkg.activate(dirname(@__DIR__));

using ThesisProject
using Random
using Printf

usage() = """
Usage:
  julia --project scripts/run_pretrain.jl --config <path> --pretrain [--epochs <Int>] [--batch <Int>] [--seed <Int>] [--nwarm <Int>]

Notes:
- Expects a callable `aegm` in scope: aegm(a::AbstractVector, y::AbstractVector) -> targets.
  You can provide it by including a file before running or by editing this script.
"""

function parse_args(argv::Vector{String})
    opt = Dict{String,Any}(
        "config" => nothing,
        "epochs" => 5,
        "batch" => 256,
        "seed" => 42,
        "pretrain" => false,
        "nwarm" => 0,
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--help" || a == "-h"
            println(usage())
            exit(0)
        elseif a == "--config"
            opt["config"] = argv[i+=1]
        elseif a == "--epochs"
            opt["epochs"] = parse(Int, argv[i+=1])
        elseif a == "--batch"
            opt["batch"] = parse(Int, argv[i+=1])
        elseif a == "--seed"
            opt["seed"] = parse(Int, argv[i+=1])
        elseif a == "--pretrain"
            opt["pretrain"] = true
        elseif a == "--nwarm"
            opt["nwarm"] = parse(Int, argv[i+=1])
        else
            error("Unknown arg: $a")
        end
        i += 1
    end
    isnothing(opt["config"]) && error("--config is required")
    return (
        config = String(opt["config"]),
        epochs = Int(opt["epochs"]),
        batch = Int(opt["batch"]),
        seed = Int(opt["seed"]),
        pretrain = Bool(opt["pretrain"]),
        nwarm = Int(opt["nwarm"]),
    )
end

function to_symbol_dict(d::AbstractDict)
    out = Dict{Symbol,Any}()
    for (k, v) in d
        out[k] = v isa AbstractDict ? to_symbol_dict(v) : v
    end
    return out
end

function main(args::Vector{String} = ARGS)
    opt = parse_args(args)
    cfg_loaded = load_config(opt.config)
    cfg = to_symbol_dict(cfg_loaded)
    validate_config(cfg)

    # Seed
    Random.seed!(opt.seed)
    random_cfg = get!(cfg, :random, Dict{Symbol,Any}())
    random_cfg[:seed] = opt.seed
    cfg[:random] = random_cfg

    if !opt.pretrain
        println("--pretrain flag not set; nothing to do.")
        return
    end

    # Check whether NN modules are available; if not, instruct user to rebuild NN solver
    if !isdefined(ThesisProject, :NNInit)
        println(
            "Neural-network solver modules are not present in this checkout.\nPlease reintroduce `src/solvers/nn/` or rebuild the solver before running pretraining.",
        )
        return
    end

    # Build NN state (weights live in-place in state)
    state = ThesisProject.NNInit.init_state(cfg)

    # Require a globally available policy function `aegm`
    if !isdefined(Main, :aegm)
        error(
            "Missing EGM policy function `aegm(a::AbstractVector, y::AbstractVector)`. Define it before running.",
        )
    end
    policy = getfield(Main, :aegm)

    if opt.nwarm > 0 && opt.nwarm < opt.epochs
        ThesisProject.NNPretrain.pretrain_then_euler!(
            state,
            policy,
            cfg;
            Nwarm = opt.nwarm,
            epochs = opt.epochs,
            batch = opt.batch,
        )
        println("Warmup+Euler training completed.")
        return nothing
    else
        metrics = fit_to_EGM!(
            state,
            policy,
            cfg;
            epochs = opt.epochs,
            batch = opt.batch,
            seed = opt.seed,
        )
        @printf(
            "Pretrain done | epochs=%d L2_train=%.6e lr=%.3g batch=%d nobs=%d\n",
            metrics.epochs,
            metrics.final_loss,
            metrics.lr,
            metrics.batch,
            metrics.nobs
        )
        return metrics
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
