#!/usr/bin/env julia

"""
Full stochastic benchmark for the baseline consumption–saving model.

Runs the production solvers (Time Iteration, EGM, Projection, Perturbation) and
two NN objectives (AiO, bc-MC) with precision settings targeting ≤1e-5 Euler
errors. NN solvers default to 100k epochs with long-patience early stopping.

Outputs:
  - `results/<tag>/stoch_benchmark_runs.csv`
  - `results/<tag>/stoch_benchmark_summary.csv`
  - `results/<tag>/manifest.toml`
  - `results/<tag>/artifact.toml`
  - `results/<tag>/insights.md`

Usage:
  julia --project=. scripts/thesis/benchmarks/full_stoch_benchmark.jl \
        --config=config/smoke_cfg_stoch.yaml --seeds=11 \
        --nn-epochs=100000 --tag=full_stoch_prod
"""
module FullStochBenchmark

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

const UtilsConfig = ThesisProject.UtilsConfig

include(joinpath(@__DIR__, "..", "..", "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

const DEFAULT_CONFIG = joinpath(ROOT, "config", "smoke_cfg_stoch.yaml")
const DEFAULT_SEEDS = Int[11]
const RESULT_ROOT = joinpath(ROOT, "results")
const DEFAULT_TAG = "stoch_prod_benchmark"

toml_safe(x) = x === nothing ? "null" : x isa Symbol ? string(x) : x

struct CLIOptions
    config_path::Union{Nothing,String}
    seeds::Vector{Int}
    tag::String
    notes::String
    ee_target::Float64
    nn_epochs::Int
    nn_samples::Int
    nn_batch::Int
    nn_patience::Int
    nn_target_loss::Float64
    use_cuda::Union{Nothing,Bool}
end

function parse_cli(args)
    cfg_env = get(ENV, "STOCH_BM_CONFIG", nothing)
    seeds_env = get(ENV, "STOCH_BM_SEEDS", "")
    tag = get(ENV, "STOCH_BM_TAG", DEFAULT_TAG)
    notes = get(ENV, "STOCH_BM_NOTES", "")
    ee_target = parse(Float64, get(ENV, "STOCH_BM_EE_TARGET", "1e-5"))
    nn_epochs = parse(Int, get(ENV, "STOCH_BM_NN_EPOCHS", "100000"))
    nn_samples = parse(Int, get(ENV, "STOCH_BM_NN_SAMPLES", "4096"))
    nn_batch = parse(Int, get(ENV, "STOCH_BM_NN_BATCH", "4096"))
    nn_patience = parse(Int, get(ENV, "STOCH_BM_NN_PATIENCE", "5000"))
    nn_target_loss = parse(Float64, get(ENV, "STOCH_BM_NN_TARGET_LOSS", "1e-8"))
    use_cuda = nothing

    seeds =
        isempty(strip(seeds_env)) ? copy(DEFAULT_SEEDS) :
        [parse(Int, strip(s)) for s in split(seeds_env, ',') if !isempty(strip(s))]

    for arg in args
        if startswith(arg, "--config=")
            cfg_env = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--seeds=")
            seeds = [
                parse(Int, strip(s)) for
                s in split(split(arg, "=", limit = 2)[2], ',') if !isempty(strip(s))
            ]
        elseif startswith(arg, "--tag=")
            tag = strip(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--notes=")
            notes = strip(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--ee-target=")
            ee_target = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--nn-epochs=")
            nn_epochs = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--nn-samples=")
            nn_samples = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--nn-batch=")
            nn_batch = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--nn-patience=")
            nn_patience = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--nn-target-loss=")
            nn_target_loss = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--use-cuda=")
            use_cuda = occursin(r"^(true|1|yes)$"i, strip(split(arg, "=", limit = 2)[2]))
        end
    end

    isempty(seeds) && error("At least one seed required")
    final_tag = isempty(strip(tag)) ? DEFAULT_TAG : tag

    return CLIOptions(
        cfg_env,
        seeds,
        final_tag,
        notes,
        ee_target,
        nn_epochs,
        nn_samples,
        nn_batch,
        nn_patience,
        nn_target_loss,
        use_cuda,
    )
end

function ensure_dir(path)
    isdir(path) || mkpath(path)
    return path
end

function merge_nn_overrides(cfg, overrides::Dict{Symbol,Any})
    base_nn = get_nested(cfg, (:solver, :nn), NamedTuple())
    merged_nn = merge_config(base_nn, dict_to_namedtuple(overrides))
    return merge_section(cfg, :solver, (; nn = merged_nn))
end

function nn_overrides(cli::CLIOptions; objective::Symbol, auto_n::Bool)
    overrides = Dict{Symbol,Any}(
        :epochs => cli.nn_epochs,
        :samples_per_epoch => cli.nn_samples,
        :batch => cli.nn_batch,
        :objective => objective,
        :patience => cli.nn_patience,
        :target_loss => cli.nn_target_loss,
        :use_cuda => cli.use_cuda === nothing ? false : cli.use_cuda,
        :verbose => false,
        :n_mc => objective == :euler_fb_aio ? 2 : 8,
        :lr_schedule => :cosine,
        :learning_rate => 1e-3,
        :v_h => 2.0,
    )
    if objective == :euler_fb_bcmc
        overrides[:bcmc_auto_N] = auto_n
        overrides[:bcmc_budget_T] = nothing
        overrides[:bcmc_update_every] = 10
    else
        overrides[:bcmc_auto_N] = false
    end
    return overrides
end

const METHOD_SPECS = (
    (
        label = "TimeIteration",
        method = :TimeIteration,
        solver_overrides = (; tol = 1e-6, tol_pol = 1e-6, maxit = 60000, verbose = false),
        nn = nothing,
    ),
    (
        label = "EGM",
        method = :EGM,
        solver_overrides = (; tol = 5e-7, tol_pol = 5e-7, maxit = 60000, verbose = false),
        nn = nothing,
    ),
    (
        label = "Projection",
        method = :Projection,
        solver_overrides = (; tol = 5e-7, maxit = 60000, verbose = false),
        nn = nothing,
    ),
    (
        label = "Perturbation",
        method = :Perturbation,
        solver_overrides = (; tol = 5e-7, maxit = 60000, verbose = false),
        nn = nothing,
    ),
    (label = "NN_AiO", method = :NN, solver_overrides = (; verbose = false), nn = :aio),
    (label = "NN_BCMC", method = :NN, solver_overrides = (; verbose = false), nn = :bcmc),
)

function apply_solver_overrides(cfg, overrides)
    isempty(fieldnames(typeof(overrides))) && return cfg
    return merge_section(cfg, :solver, overrides)
end

function run_config(base_cfg, spec, cli::CLIOptions, seed::Int)
    cfg = merge_section(base_cfg, :solver, (; method = spec.method))
    if spec.solver_overrides !== nothing &&
       length(fieldnames(typeof(spec.solver_overrides))) > 0
        cfg = merge_section(cfg, :solver, spec.solver_overrides)
    end
    cfg = merge_section(cfg, :random, (; seed = seed))
    if spec.method == :NN
        overrides =
            spec.nn === :aio ?
            nn_overrides(cli; objective = :euler_fb_aio, auto_n = false) :
            nn_overrides(cli; objective = :euler_fb_bcmc, auto_n = true)
        cfg = merge_nn_overrides(cfg, overrides)
    end
    return UtilsConfig.ensure_master_rng(cfg; require = true)
end

function extract_euler_errors(sol)
    pol = sol.policy[:c]
    if hasproperty(pol, :euler_errors_mat) && pol.euler_errors_mat !== nothing
        ee = pol.euler_errors_mat
    elseif hasproperty(pol, :euler_errors)
        ee = pol.euler_errors
    else
        return Float64[]
    end
    return Float64.(vec(ee))
end

function euler_stats(sol)
    data = extract_euler_errors(sol)
    isempty(data) && return (mean = NaN, p95 = NaN, max = NaN)
    clean = [v for v in data if isfinite(v)]
    isempty(clean) && return (mean = NaN, p95 = NaN, max = NaN)
    absvals = abs.(clean)
    return (
        mean = mean(absvals),
        p95 = Statistics.quantile(absvals, 0.95),
        max = maximum(absvals),
    )
end

nt_get(nt, sym, default) =
    nt === nothing ? default :
    (nt isa NamedTuple && hasproperty(nt, sym)) ? getproperty(nt, sym) :
    (nt isa AbstractDict && haskey(nt, sym)) ? nt[sym] : default

function collect_record(spec, sol, cli::CLIOptions, seed::Int)
    stats = euler_stats(sol)
    diag = sol.diagnostics
    runtime = nt_get(diag, :runtime, nt_get(sol.metadata, :runtime, NaN))
    iters = nt_get(diag, :iterations, nt_get(sol.metadata, :iters, missing))
    mean_ee = stats.mean
    return (
        label = spec.label,
        method = spec.method,
        seed = seed,
        runtime = runtime,
        iterations = iters,
        mean_ee = mean_ee,
        p95_ee = stats.p95,
        max_ee = stats.max,
        max_resid = nt_get(sol.metadata, :max_resid, NaN),
        converged = nt_get(sol.metadata, :converged, false),
        meets_target = mean_ee ≤ cli.ee_target,
    )
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

function summarize_records(records)
    groups = Dict{String,Vector{NamedTuple}}()
    for rec in records
        push!(get!(groups, rec.label, NamedTuple[]), rec)
    end
    summary = NamedTuple[]
    for (label, grp) in sort(collect(groups); by = first)
        push!(
            summary,
            (;
                label = label,
                runs = length(grp),
                runtime_mean = mean([r.runtime for r in grp]),
                runtime_sd = length(grp) > 1 ? std([r.runtime for r in grp]) : 0.0,
                mean_ee = mean([r.mean_ee for r in grp]),
                p95_ee = mean([r.p95_ee for r in grp]),
                max_ee = mean([r.max_ee for r in grp]),
                target_met_share = mean([r.meets_target for r in grp]),
            ),
        )
    end
    return summary
end

function write_runs_csv(path, records)
    header = [
        "label",
        "method",
        "seed",
        "runtime",
        "iterations",
        "mean_ee",
        "p95_ee",
        "max_ee",
        "max_resid",
        "converged",
        "meets_target",
    ]
    rows = [
        (
            rec.label,
            string(rec.method),
            rec.seed,
            rec.runtime,
            rec.iterations,
            rec.mean_ee,
            rec.p95_ee,
            rec.max_ee,
            rec.max_resid,
            rec.converged,
            rec.meets_target,
        ) for rec in records
    ]
    write_csv(path, header, rows)
end

function write_summary_csv(path, summary)
    header = [
        "label",
        "runs",
        "runtime_mean",
        "runtime_sd",
        "mean_ee",
        "p95_ee",
        "max_ee",
        "target_met_share",
    ]
    rows = [
        (
            row.label,
            row.runs,
            row.runtime_mean,
            row.runtime_sd,
            row.mean_ee,
            row.p95_ee,
            row.max_ee,
            row.target_met_share,
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
            "ee_target" => cli.ee_target,
            "nn_epochs" => cli.nn_epochs,
            "nn_samples" => cli.nn_samples,
            "nn_batch" => cli.nn_batch,
            "nn_patience" => cli.nn_patience,
            "nn_target_loss" => cli.nn_target_loss,
            "use_cuda" => toml_safe(cli.use_cuda),
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

function insight_lines(summary, cli::CLIOptions)
    lines = String[@sprintf("EE target: %.1e", cli.ee_target),]
    best = findmin(row -> row.mean_ee, summary)[2]
    push!(
        lines,
        @sprintf(
            "Best accuracy: %s with mean EE %.2e",
            summary[best].label,
            summary[best].mean_ee
        )
    )
    fastest = findmin(row -> row.runtime_mean, summary)[2]
    push!(
        lines,
        @sprintf(
            "Fastest: %s (runtime %.2fs)",
            summary[fastest].label,
            summary[fastest].runtime_mean
        )
    )
    return lines
end

function write_insights(path, summary, cli::CLIOptions)
    open(path, "w") do io
        println(io, "# Full stochastic benchmark")
        println(io)
        println(
            io,
            "- Tag: `$(cli.tag)` | Seeds: $(join(cli.seeds, ", ")) | Notes: $(isempty(cli.notes) ? "(none)" : cli.notes)",
        )
        println(io)
        for line in insight_lines(summary, cli)
            println(io, "- $(line)")
        end
        println(io)
        println(io, "| method | runs | runtime (mean±sd) | mean EE | p95 EE | target hit |")
        println(io, "|--------|------|-------------------|---------|--------|------------|")
        for row in summary
            println(
                io,
                @sprintf(
                    "| %s | %d | %.2f±%.2f | %.2e | %.2e | %.0f%% |",
                    row.label,
                    row.runs,
                    row.runtime_mean,
                    row.runtime_sd,
                    row.mean_ee,
                    row.p95_ee,
                    100 * row.target_met_share,
                ),
            )
        end
    end
end

function prepare_report_state(tag)
    dir = ensure_dir(joinpath(RESULT_ROOT, tag))
    return (
        runs = joinpath(dir, "stoch_benchmark_runs.csv"),
        summary = joinpath(dir, "stoch_benchmark_summary.csv"),
        manifest = joinpath(dir, "manifest.toml"),
        artifact = joinpath(dir, "artifact.toml"),
        insights = joinpath(dir, "insights.md"),
        dir = dir,
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
    isfile(cfg_path) || error("Config $(cfg_path) not found")
    base_cfg =
        UtilsConfig.ensure_master_rng(ThesisProject.load_config(cfg_path); require = true)
    report = prepare_report_state(cli.tag)

    records = NamedTuple[]
    for seed in cli.seeds
        for spec in METHOD_SPECS
            cfg = run_config(base_cfg, spec, cli, seed)
            model = ThesisProject.build_model(cfg)
            method = ThesisProject.build_method(cfg)
            sol = ThesisProject.solve(
                model,
                method,
                cfg;
                rng = derive_rng(cfg.random.master_rng, string(spec.label, seed)),
            )
            push!(records, collect_record(spec, sol, cli, seed))
        end
    end

    write_runs_csv(report.runs, records)
    summary = summarize_records(records)
    write_summary_csv(report.summary, summary)
    write_manifest(report.manifest; config_path = cfg_path, cli = cli, seeds = cli.seeds)
    write_artifact(report.artifact)
    write_insights(report.insights, summary, cli)
    println("\nBenchmark complete → $(report.dir)")
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    FullStochBenchmark.main()
end
