#!/usr/bin/env julia

"""
Diagnose the interaction between learning-rate schedulers and optimisers for the NN solver.

Runs the stochastic benchmark configuration across the Cartesian product of
{cosine, exponential} schedulers and user-selected optimisers, recording runtimes,
Euler residuals, and policy diagnostics. Outputs CSV tables plus a Markdown
insight sheet under `results/<tag>/`.

Usage example:
  julia --project=. scripts/thesis/reports/compare_scheduler_optimizer.jl \
        --config=config/smoke_cfg_stoch.yaml --seeds=11,19 --optimizers=adam,rmsprop \
        --epochs=400 --samples=2048 --batch=512 --objective=euler_fb_bcmc \
        --tag=sched_opt_study --notes="low-batch sweep"
"""
module CompareSchedulerOptimizer

import Pkg

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.activate(ROOT; io = devnull)

using Dates
using InteractiveUtils: versioninfo
using LibGit2
using Printf
using Random
using Statistics
using TOML
using ThesisProject
using ThesisProject.Determinism: derive_rng, hash_hex, canonicalize_cfg
using ThesisProject.NNKernel: solve_nn

const UtilsConfig = ThesisProject.UtilsConfig

include(joinpath(@__DIR__, "..", "..", "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

function apply_nn_overrides(cfg, overrides::NamedTuple)
    base_nn = get_nested(cfg, (:solver, :nn), NamedTuple())
    merged_nn = merge_config(base_nn, overrides)
    return merge_section(cfg, :solver, (; nn = merged_nn))
end

apply_nn_overrides(cfg, overrides::Dict{Symbol,Any}) =
    apply_nn_overrides(cfg, dict_to_namedtuple(overrides))

const DEFAULT_CONFIG = joinpath(ROOT, "config", "smoke_cfg_stoch.yaml")
const DEFAULT_SEEDS = Int[11, 19, 23]
const DEFAULT_SCHEDULERS = Symbol[:cosine, :exponential]
const DEFAULT_OPTIMIZERS = Symbol[:adam, :rmsprop]
const RESULT_ROOT = joinpath(ROOT, "results")

toml_safe(x) = x === nothing ? "null" : x isa Symbol ? string(x) : x

struct CLIOptions
    config_path::Union{Nothing,String}
    seeds::Vector{Int}
    schedulers::Vector{Symbol}
    optimizers::Vector{Symbol}
    epochs::Union{Nothing,Int}
    samples::Union{Nothing,Int}
    batch::Union{Nothing,Int}
    n_mc::Union{Nothing,Int}
    use_cuda::Union{Nothing,Bool}
    objective::Symbol
    notes::String
    verbose::Bool
    tag::String
end

Base.@kwdef struct RunRecord
    scheduler::Symbol
    optimizer::Symbol
    seed::Int
    runtime::Float64
    epochs_run::Int
    batch::Int
    samples_per_epoch::Int
    n_mc::Int
    use_cuda::Bool
    device::Symbol
    mean_abs_resid::Float64
    p95_abs_resid::Float64
    max_abs_resid::Float64
    resid_std::Float64
    kt_mean::Float64
    policy_p95::Float64
    policy_max::Float64
    binding_share::Float64
    monotonic_share::Float64
    max_resid::Float64
    eval_mc_mean::Union{Missing,Float64}
    eval_mc_p95::Union{Missing,Float64}
    eval_mc_max::Union{Missing,Float64}
    model_id::String
    notes::String
end

mutable struct ReportState
    result_dir::String
    runs_path::String
    summary_path::String
    manifest_path::String
    artifact_path::String
    insights_path::String
end

function ensure_dir(path)
    isdir(path) || mkpath(path)
    return path
end

parse_bool(str) = occursin(r"^(true|1|yes)$"i, strip(str))

function parse_symbol_list(raw::AbstractString, default)
    txt = strip(raw)
    isempty(txt) && return copy(default)
    vals = Symbol[]
    for token in split(txt, ',')
        v = strip(token)
        isempty(v) && continue
        push!(vals, Symbol(lowercase(v)))
    end
    isempty(vals) ? copy(default) : vals
end

function parse_cli(args)
    cfg_env = get(ENV, "SCHED_OPT_CONFIG", nothing)
    seed_env = get(ENV, "SCHED_OPT_SEEDS", "")
    sched_env = get(ENV, "SCHED_OPT_SCHEDULERS", "")
    opt_env = get(ENV, "SCHED_OPT_OPTIMIZERS", "")
    epochs_env = get(ENV, "SCHED_OPT_EPOCHS", "")
    samples_env = get(ENV, "SCHED_OPT_SAMPLES", "")
    batch_env = get(ENV, "SCHED_OPT_BATCH", "")
    nmc_env = get(ENV, "SCHED_OPT_NMC", "")
    cuda_env = get(ENV, "SCHED_OPT_USE_CUDA", "")
    obj_env = get(ENV, "SCHED_OPT_OBJECTIVE", "euler_fb_bcmc")
    notes = get(ENV, "SCHED_OPT_NOTES", "")
    tag = get(ENV, "SCHED_OPT_TAG", "sched_opt_study")
    verbose = false

    seeds =
        isempty(strip(seed_env)) ? copy(DEFAULT_SEEDS) :
        [parse(Int, strip(s)) for s in split(seed_env, ',') if !isempty(strip(s))]
    schedulers = parse_symbol_list(sched_env, DEFAULT_SCHEDULERS)
    optimizers = parse_symbol_list(opt_env, DEFAULT_OPTIMIZERS)
    epochs = isempty(strip(epochs_env)) ? nothing : parse(Int, epochs_env)
    samples = isempty(strip(samples_env)) ? nothing : parse(Int, samples_env)
    batch = isempty(strip(batch_env)) ? nothing : parse(Int, batch_env)
    n_mc = isempty(strip(nmc_env)) ? nothing : parse(Int, nmc_env)
    use_cuda = isempty(strip(cuda_env)) ? nothing : parse_bool(cuda_env)
    objective = Symbol(lowercase(strip(obj_env)))

    for arg in args
        if startswith(arg, "--config=")
            cfg_env = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--seeds=")
            seeds = [
                parse(Int, strip(s)) for
                s in split(split(arg, "=", limit = 2)[2], ',') if !isempty(strip(s))
            ]
        elseif startswith(arg, "--schedulers=")
            schedulers =
                parse_symbol_list(split(arg, "=", limit = 2)[2], DEFAULT_SCHEDULERS)
        elseif startswith(arg, "--optimizers=")
            optimizers =
                parse_symbol_list(split(arg, "=", limit = 2)[2], DEFAULT_OPTIMIZERS)
        elseif startswith(arg, "--epochs=")
            epochs = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--samples=")
            samples = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--batch=")
            batch = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--n-mc=")
            n_mc = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--use-cuda=")
            use_cuda = parse_bool(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--objective=")
            objective = Symbol(lowercase(strip(split(arg, "=", limit = 2)[2])))
        elseif startswith(arg, "--tag=")
            tag = strip(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--notes=")
            notes = strip(split(arg, "=", limit = 2)[2])
        elseif arg == "--verbose"
            verbose = true
        end
    end

    isempty(seeds) && error("At least one seed is required")
    isempty(schedulers) && error("Provide at least one scheduler")
    isempty(optimizers) && error("Provide at least one optimizer")

    return CLIOptions(
        cfg_env,
        seeds,
        schedulers,
        optimizers,
        epochs,
        samples,
        batch,
        n_mc,
        use_cuda,
        objective,
        notes,
        verbose,
        isempty(strip(tag)) ? "sched_opt_study" : tag,
    )
end

nt_get(nt, sym::Symbol, default) =
    nt isa NamedTuple ? (hasproperty(nt, sym) ? getproperty(nt, sym) : default) :
    nt isa AbstractDict ? get(nt, sym, default) : default

function config_with_overrides(
    base_cfg::NamedTuple,
    scheduler::Symbol,
    optimizer::Symbol,
    cli::CLIOptions,
    seed::Int,
)
    cfg = merge_section(base_cfg, :solver, (; method = :NN,))
    cfg = merge_section(cfg, :random, (; seed = seed))
    overrides = Dict{Symbol,Any}(:lr_schedule => scheduler, :optimizer => optimizer)
    cli.epochs !== nothing && (overrides[:epochs] = cli.epochs)
    cli.samples !== nothing && (overrides[:samples_per_epoch] = cli.samples)
    cli.batch !== nothing && (overrides[:batch] = cli.batch)
    cli.n_mc !== nothing && (overrides[:n_mc] = max(cli.n_mc, 2))
    cli.use_cuda !== nothing && (overrides[:use_cuda] = cli.use_cuda)
    overrides[:objective] = cli.objective
    cfg = apply_nn_overrides(cfg, overrides)
    return UtilsConfig.ensure_master_rng(cfg; require = true)
end

function flatten_array(values)
    values isa AbstractArray ? vec(Float64.(collect(values))) : Float64[]
end

describe_abs(vec) =
    isempty(vec) ? (mean = NaN, p95 = NaN, max = NaN, std = NaN) :
    (
        mean = mean(abs.(vec)),
        p95 = Statistics.quantile(abs.(vec), 0.95),
        max = maximum(abs.(vec)),
        std = std(abs.(vec)),
    )

function binding_share(a_next, a_min; tol = 1e-8)
    vals = Float64.(collect(a_next))
    total = length(vals)
    total == 0 && return 0.0
    count(x -> x ≤ a_min + tol, vals) / total
end

shock_count(S) =
    S === nothing ? 1 :
    hasproperty(S, :zgrid) ? length(getproperty(S, :zgrid)) :
    hasproperty(S, :y_dim) ? Int(getproperty(S, :y_dim)) : 1

function mc_stats_tuple(diag)
    diag === nothing && return (missing, missing, missing)
    stats = getfield(diag, :stats)
    return (Float64(stats.mean), Float64(stats.p95), Float64(stats.max))
end

function collect_metrics(scheduler, optimizer, seed, cfg, model, method, sol_nn, cli)
    G = ThesisProject.get_grids(model)
    S = ThesisProject.get_shocks(model)
    resid_vec = flatten_array(sol_nn.resid)
    abs_desc = describe_abs(resid_vec)
    mc_mean, mc_p95, mc_max = mc_stats_tuple(sol_nn.eval_mc)
    c_mat = Float64.(sol_nn.c)
    policy_vals = vec(c_mat)
    policy_p95 = Statistics.quantile(policy_vals, 0.95)
    policy_max = maximum(policy_vals)
    bind_share = binding_share(sol_nn.a_next, G[:a].min)
    diffs = diff(c_mat; dims = 1)
    total = length(diffs)
    monotonic_share = total == 0 ? 1.0 : count(x -> x >= -1e-8, diffs) / total
    opts = method.opts
    nn_opts = sol_nn.opts
    return RunRecord(
        scheduler = scheduler,
        optimizer = optimizer,
        seed = seed,
        runtime = Float64(nn_opts.runtime),
        epochs_run = sol_nn.iters,
        batch = Int(opts.batch),
        samples_per_epoch = Int(opts.samples_per_epoch),
        n_mc = Int(opts.n_mc),
        use_cuda = cli.use_cuda === nothing ? false : cli.use_cuda,
        device = nt_get(nn_opts, :device, :cpu),
        mean_abs_resid = abs_desc.mean,
        p95_abs_resid = abs_desc.p95,
        max_abs_resid = abs_desc.max,
        resid_std = abs_desc.std,
        kt_mean = nt_get(sol_nn.eval_mc, :stats, (; mean = NaN)).mean,
        policy_p95 = policy_p95,
        policy_max = policy_max,
        binding_share = bind_share,
        monotonic_share = monotonic_share,
        max_resid = Float64(sol_nn.max_resid),
        eval_mc_mean = mc_mean,
        eval_mc_p95 = mc_p95,
        eval_mc_max = mc_max,
        model_id = hash_hex(canonicalize_cfg(cfg)),
        notes = cli.notes,
    )
end

function run_case(base_cfg, scheduler, optimizer, cli::CLIOptions, seed::Int)
    cfg = config_with_overrides(base_cfg, scheduler, optimizer, cli, seed)
    model = ThesisProject.build_model(cfg)
    method = ThesisProject.build_method(cfg)
    rng = derive_rng(cfg.random.master_rng, "nn_kernel")
    cli.verbose && @info "Running NN" scheduler optimizer seed method_opts = method.opts
    sol_nn = solve_nn(model; opts = method.opts, rng = rng)
    return collect_metrics(scheduler, optimizer, seed, cfg, model, method, sol_nn, cli)
end

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            vals = Any[row...]
            out = Vector{String}(undef, length(vals))
            for (i, v) in enumerate(vals)
                out[i] =
                    v === missing || v === nothing ? "" :
                    v isa Float64 ? @sprintf("%.8g", v) : string(v)
            end
            println(io, join(out, ','))
        end
    end
end

function write_runs_csv(path, runs::Vector{RunRecord})
    header = [
        "scheduler",
        "optimizer",
        "seed",
        "runtime",
        "epochs_run",
        "batch",
        "samples_per_epoch",
        "n_mc",
        "use_cuda",
        "device",
        "mean_abs_resid",
        "p95_abs_resid",
        "max_abs_resid",
        "resid_std",
        "policy_p95",
        "policy_max",
        "binding_share",
        "monotonic_share",
        "max_resid",
        "eval_mc_mean",
        "eval_mc_p95",
        "eval_mc_max",
        "model_id",
        "notes",
    ]
    rows = [
        (
            String(r.scheduler),
            String(r.optimizer),
            r.seed,
            r.runtime,
            r.epochs_run,
            r.batch,
            r.samples_per_epoch,
            r.n_mc,
            r.use_cuda,
            r.device,
            r.mean_abs_resid,
            r.p95_abs_resid,
            r.max_abs_resid,
            r.resid_std,
            r.policy_p95,
            r.policy_max,
            r.binding_share,
            r.monotonic_share,
            r.max_resid,
            r.eval_mc_mean,
            r.eval_mc_p95,
            r.eval_mc_max,
            r.model_id,
            r.notes,
        ) for r in runs
    ]
    write_csv(path, header, rows)
end

function mean_skipmissing(values)
    buf = Float64[]
    for v in values
        v === missing && continue
        push!(buf, Float64(v))
    end
    isempty(buf) ? missing : mean(buf)
end

function summarize_runs(runs::Vector{RunRecord})
    groups = Dict{Tuple{Symbol,Symbol},Vector{RunRecord}}()
    for rec in runs
        key = (rec.scheduler, rec.optimizer)
        push!(get!(groups, key, RunRecord[]), rec)
    end
    summary = NamedTuple[]
    for (key, grp) in sort(collect(groups); by = x -> x[1])
        sched, opt = key
        runtimes = [r.runtime for r in grp]
        p95s = [r.p95_abs_resid for r in grp]
        push!(
            summary,
            (;
                scheduler = String(sched),
                optimizer = String(opt),
                runs = length(grp),
                runtime_mean = mean(runtimes),
                runtime_sd = length(runtimes) > 1 ? std(runtimes) : 0.0,
                mean_abs_resid = mean([r.mean_abs_resid for r in grp]),
                p95_abs_resid = mean(p95s),
                max_abs_resid = mean([r.max_abs_resid for r in grp]),
                policy_p95 = mean([r.policy_p95 for r in grp]),
                policy_max = mean([r.policy_max for r in grp]),
                binding_share = mean([r.binding_share for r in grp]),
                monotonic_share = mean([r.monotonic_share for r in grp]),
                mc_mean = mean_skipmissing([r.eval_mc_mean for r in grp]),
                mc_p95 = mean_skipmissing([r.eval_mc_p95 for r in grp]),
                mc_max = mean_skipmissing([r.eval_mc_max for r in grp]),
            ),
        )
    end
    return summary
end

function write_summary_csv(path, summary)
    header = [
        "scheduler",
        "optimizer",
        "runs",
        "runtime_mean",
        "runtime_sd",
        "mean_abs_resid",
        "p95_abs_resid",
        "max_abs_resid",
        "policy_p95",
        "policy_max",
        "binding_share",
        "monotonic_share",
        "mc_mean",
        "mc_p95",
        "mc_max",
    ]
    rows = [
        (
            row.scheduler,
            row.optimizer,
            row.runs,
            row.runtime_mean,
            row.runtime_sd,
            row.mean_abs_resid,
            row.p95_abs_resid,
            row.max_abs_resid,
            row.policy_p95,
            row.policy_max,
            row.binding_share,
            row.monotonic_share,
            row.mc_mean,
            row.mc_p95,
            row.mc_max,
        ) for row in summary
    ]
    write_csv(path, header, rows)
end

function write_manifest(path; config_path, cli::CLIOptions, seeds)
    repo = LibGit2.GitRepo(ROOT)
    commit = string(LibGit2.GitHash(LibGit2.head(repo)))
    data = Dict(
        "timestamp" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        "config_path" => config_path,
        "seeds" => seeds,
        "cli" => Dict(
            "schedulers" => map(string, cli.schedulers),
            "optimizers" => map(string, cli.optimizers),
            "epochs" => toml_safe(cli.epochs),
            "samples" => toml_safe(cli.samples),
            "batch" => toml_safe(cli.batch),
            "n_mc" => toml_safe(cli.n_mc),
            "use_cuda" => toml_safe(cli.use_cuda),
            "objective" => string(cli.objective),
            "tag" => cli.tag,
            "notes" => cli.notes,
        ),
        "git_commit" => commit,
    )
    open(path, "w") do io
        TOML.print(io, data)
    end
end

function write_artifact(path)
    buf_pkg = IOBuffer()
    Pkg.status(; mode = Pkg.PKGMODE_MANIFEST, io = buf_pkg)
    pkg_str = String(take!(buf_pkg))
    buf_ver = IOBuffer()
    versioninfo(buf_ver; verbose = true)
    ver_str = String(take!(buf_ver))
    repo = LibGit2.GitRepo(ROOT)
    commit = string(LibGit2.GitHash(LibGit2.head(repo)))
    data = Dict(
        "timestamp" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        "git_commit" => commit,
        "julia_version" => string(VERSION),
        "machine" => toml_safe(Sys.MACHINE),
        "kernel" => toml_safe(Sys.KERNEL),
        "cpu_threads" => Sys.CPU_THREADS,
        "pkg_status" => pkg_str,
        "versioninfo" => ver_str,
    )
    open(path, "w") do io
        TOML.print(io, data)
    end
end

function insight_lines(summary)
    isempty(summary) && return ["- No runs completed"]
    best = findmin(row -> row.runtime_mean, summary)[2]
    ref = summary[best]
    lines = [
        @sprintf(
            "- Fastest combo: %s + %s (runtime %.3fs, mean abs resid %.2e).",
            ref.scheduler,
            ref.optimizer,
            ref.runtime_mean,
            ref.mean_abs_resid,
        ),
    ]
    push!(
        lines,
        @sprintf(
            "- Residual stability: best p95=%.2e (combo %s/%s).",
            ref.p95_abs_resid,
            ref.scheduler,
            ref.optimizer
        )
    )
    return lines
end

function write_insights(path, summary, cli::CLIOptions)
    open(path, "w") do io
        println(io, "# Scheduler vs Optimiser study")
        println(io)
        println(
            io,
            "- Tag: `$(cli.tag)` | Seeds: $(join(cli.seeds, ", ")) | Notes: $(isempty(cli.notes) ? "(none)" : cli.notes)",
        )
        println(io)
        for line in insight_lines(summary)
            println(io, line)
        end
        println(io)
        println(
            io,
            "| scheduler | optimizer | runs | runtime (mean±sd) | p95 abs resid | MC p95 |",
        )
        println(
            io,
            "|-----------|-----------|------|-------------------|---------------|-------|",
        )
        for row in summary
            println(
                io,
                @sprintf(
                    "| %s | %s | %d | %.3f±%.3f | %.2e | %.2e |",
                    row.scheduler,
                    row.optimizer,
                    row.runs,
                    row.runtime_mean,
                    row.runtime_sd,
                    row.p95_abs_resid,
                    row.mc_p95,
                ),
            )
        end
    end
end

function prepare_report_state(tag::AbstractString)
    result_dir = ensure_dir(joinpath(RESULT_ROOT, tag))
    return ReportState(
        result_dir,
        joinpath(result_dir, "sched_opt_runs.csv"),
        joinpath(result_dir, "sched_opt_summary.csv"),
        joinpath(result_dir, "manifest.toml"),
        joinpath(result_dir, "artifact.toml"),
        joinpath(result_dir, "insights.md"),
    )
end

function main()
    cli = parse_cli(ARGS)
    cfg_path =
        cli.config_path === nothing ? DEFAULT_CONFIG :
        (
            isabspath(cli.config_path) ? cli.config_path :
            normpath(joinpath(ROOT, cli.config_path))
        )
    isfile(cfg_path) || error("Config path $(cfg_path) not found")
    base_cfg =
        UtilsConfig.ensure_master_rng(ThesisProject.load_config(cfg_path); require = true)
    report = prepare_report_state(cli.tag)

    runs = RunRecord[]
    for seed in cli.seeds, scheduler in cli.schedulers, optimizer in cli.optimizers
        push!(runs, run_case(base_cfg, scheduler, optimizer, cli, seed))
    end

    write_runs_csv(report.runs_path, runs)
    summary = summarize_runs(runs)
    write_summary_csv(report.summary_path, summary)
    write_manifest(
        report.manifest_path;
        config_path = cfg_path,
        cli = cli,
        seeds = cli.seeds,
    )
    write_artifact(report.artifact_path)
    write_insights(report.insights_path, summary, cli)
    println("\nScheduler/optimiser study complete → $(report.result_dir)")
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    CompareSchedulerOptimizer.main()
end
