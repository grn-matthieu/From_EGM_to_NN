#!/usr/bin/env julia
using Pkg;
Pkg.activate(dirname(@__DIR__));

using ThesisProject
using Random
using Printf

include(joinpath(@__DIR__, "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

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

function main(args::Vector{String} = ARGS)
    opt = parse_args(args)
    cfg_loaded = load_config(opt.config)
    cfg = dict_to_namedtuple(cfg_loaded)

    cfg = merge_section(cfg, :random, (; seed = opt.seed))
    Random.seed!(opt.seed)
    cfg_dict = namedtuple_to_dict(cfg)

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
    state = ThesisProject.NNInit.init_state(cfg_dict)

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
            cfg_dict;
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
            cfg_dict;
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
