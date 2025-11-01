#!/usr/bin/env julia

"""
Compare Fischer–Burmeister AiO vs bias-corrected Monte Carlo (bc-MC) Auto-N objectives.

The script runs the NN solver on a stochastic consumption–saving configuration for
both loss functions across multiple random seeds, captures rich diagnostics
(runtime, Euler residual distributions, policy-shape metrics, binding shares,
Monte-Carlo and Gauss–Hermite residuals), and writes publication-ready CSV/Markdown
artifacts under `results/<tag>/`.

Outputs:
  - `<results>/<tag>/aio_bcmc_runs.csv`      (per-run metrics)
  - `<results>/<tag>/aio_bcmc_summary.csv`   (objective-level aggregates)
  - `<results>/<tag>/insights.md`            (plain-language comparison notes)
  - `<results>/<tag>/manifest.toml`          (environment + config record)
  - `<results>/<tag>/artifact.toml`          (git/Julia environment snapshot)

Usage:
  julia --project=. scripts/thesis/reports/compare_aio_bcmc.jl [--config=path] [--seeds=...] \\
        [--epochs=...] [--samples=...] [--batch=...] [--n-mc=...] [--bcmc-budget=...] \\
        [--bcmc-update=...] [--use-cuda=true|false] [--tag=name] [--notes="..."] [--verbose]
"""
module CompareAioVsBcmc

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
using ThesisProject.Determinism: canonicalize_cfg, derive_rng, hash_hex
using ThesisProject.NNKernel: solve_nn

const UtilsConfig = ThesisProject.UtilsConfig

include(joinpath(@__DIR__, "..", "..", "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

const DEFAULT_CONFIG = joinpath(ROOT, "config", "smoke_cfg_stoch.yaml")
const DEFAULT_SEEDS = Int[17, 29, 41, 53, 71]
const DEFAULT_TAG = "aio_vs_bcmc"
const RESULT_ROOT = joinpath(ROOT, "results")

toml_safe(x) = x === nothing ? "null" : x isa Symbol ? string(x) : x

struct CLIOptions
    config_path::Union{Nothing,String}
    seeds::Vector{Int}
    epochs::Union{Nothing,Int}
    samples::Union{Nothing,Int}
    batch::Union{Nothing,Int}
    n_mc::Union{Nothing,Int}
    bcmc_budget::Union{Nothing,Int}
    bcmc_update::Union{Nothing,Int}
    use_cuda::Union{Nothing,Bool}
    notes::String
    verbose::Bool
    tag::String
end

Base.@kwdef struct RunRecord
    objective::Symbol
    seed::Int
    model_name::Symbol
    runtime::Float64
    epochs_target::Int
    epochs_run::Int
    batch::Int
    samples_per_epoch::Int
    n_mc::Int
    bcmc_auto::Bool
    bcmc_budget::Union{Nothing,Int}
    bcmc_update::Union{Nothing,Int}
    use_cuda::Bool
    device::Symbol
    converged::Bool
    mean_abs_resid::Float64
    median_abs_resid::Float64
    p95_abs_resid::Float64
    max_abs_resid::Float64
    resid_std::Float64
    policy_mean::Float64
    policy_std::Float64
    policy_p95::Float64
    policy_min::Float64
    policy_max::Float64
    binding_share::Float64
    monotonic_share::Float64
    monotonic_min_slope::Float64
    monotonic_max_slope::Float64
    consumption_floor_hits::Int
    grid_points::Int
    eval_mc_mean::Union{Missing,Float64}
    eval_mc_p95::Union{Missing,Float64}
    eval_mc_max::Union{Missing,Float64}
    eval_gh_mean::Union{Missing,Float64}
    eval_gh_p95::Union{Missing,Float64}
    eval_gh_max::Union{Missing,Float64}
    max_resid::Float64
    model_id::String
    Na::Int
    Nz::Int
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

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function parse_int_list(raw::AbstractString, default::Vector{Int})
    stripped = strip(raw)
    isempty(stripped) && return copy(default)
    buf = Int[]
    for token in split(stripped, ',')
        t = strip(token)
        isempty(t) && continue
        push!(buf, parse(Int, t))
    end
    return isempty(buf) ? copy(default) : buf
end

parse_bool(raw::AbstractString) = occursin(r"^(true|1|yes)$"i, strip(raw))

function parse_cli(args)
    config_env = get(ENV, "AIO_BCMC_CONFIG", nothing)
    seeds_env = get(ENV, "AIO_BCMC_SEEDS", "")
    epochs_env = get(ENV, "AIO_BCMC_EPOCHS", "")
    samples_env = get(ENV, "AIO_BCMC_SAMPLES", "")
    batch_env = get(ENV, "AIO_BCMC_BATCH", "")
    nmc_env = get(ENV, "AIO_BCMC_NMC", "")
    budget_env = get(ENV, "AIO_BCMC_BUDGET", "")
    update_env = get(ENV, "AIO_BCMC_UPDATE", "")
    cuda_env = get(ENV, "AIO_BCMC_USE_CUDA", "")
    notes = get(ENV, "AIO_BCMC_NOTES", "")
    tag = get(ENV, "AIO_BCMC_TAG", DEFAULT_TAG)
    verbose = false

    function maybe_int(str)
        isempty(strip(str)) && return nothing
        return parse(Int, strip(str))
    end

    function maybe_bool(str)
        isempty(strip(str)) && return nothing
        return parse_bool(str)
    end

    seeds =
        isempty(strip(seeds_env)) ? copy(DEFAULT_SEEDS) :
        parse_int_list(seeds_env, DEFAULT_SEEDS)
    epochs = maybe_int(epochs_env)
    samples = maybe_int(samples_env)
    batch = maybe_int(batch_env)
    n_mc = maybe_int(nmc_env)
    bcmc_budget = maybe_int(budget_env)
    bcmc_update = maybe_int(update_env)
    use_cuda = maybe_bool(cuda_env)

    for arg in args
        if startswith(arg, "--config=")
            config_env = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--seeds=")
            seeds = parse_int_list(split(arg, "=", limit = 2)[2], DEFAULT_SEEDS)
        elseif startswith(arg, "--epochs=")
            epochs = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--samples=")
            samples = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--batch=")
            batch = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--n-mc=")
            n_mc = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--bcmc-budget=")
            bcmc_budget = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--bcmc-update=")
            bcmc_update = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--use-cuda=")
            use_cuda = parse_bool(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--tag=")
            tag = strip(split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--notes=")
            notes = strip(split(arg, "=", limit = 2)[2])
        elseif arg == "--verbose"
            verbose = true
        end
    end

    isempty(seeds) && error("At least one seed is required")

    tag_final = isempty(strip(tag)) ? DEFAULT_TAG : tag
    return CLIOptions(
        config_env,
        seeds,
        epochs,
        samples,
        batch,
        n_mc,
        bcmc_budget,
        bcmc_update,
        use_cuda,
        notes,
        verbose,
        tag_final,
    )
end

to_abs_path(path::AbstractString) =
    isabspath(path) ? normpath(path) : normpath(joinpath(ROOT, path))

nt_get(nt, sym::Symbol, default) =
    nt isa NamedTuple ? (hasproperty(nt, sym) ? getproperty(nt, sym) : default) :
    nt isa AbstractDict ? get(nt, sym, default) : default

function with_nn_overrides(cfg::NamedTuple, overrides::NamedTuple)
    base_nn = get_nested(cfg, (:solver, :nn), NamedTuple())
    merged_nn = merge_config(base_nn, overrides)
    return merge_section(cfg, :solver, (; nn = merged_nn))
end

function maybe_merge_nn(cfg::NamedTuple, overrides::Dict{Symbol,Any})
    isempty(overrides) && return cfg
    return with_nn_overrides(cfg, dict_to_namedtuple(overrides))
end

function prepare_run_cfg(
    base_cfg::NamedTuple,
    objective::Symbol,
    cli::CLIOptions,
    seed::Int,
)
    cfg = merge_section(base_cfg, :solver, (; method = :NN,))
    cfg = merge_section(cfg, :random, (; seed = seed))
    cfg = UtilsConfig.ensure_master_rng(cfg; require = true)

    overrides = Dict{Symbol,Any}()
    cli.epochs !== nothing && (overrides[:epochs] = cli.epochs)
    cli.samples !== nothing && (overrides[:samples_per_epoch] = cli.samples)
    cli.batch !== nothing && (overrides[:batch] = cli.batch)
    cli.n_mc !== nothing && (overrides[:n_mc] = max(cli.n_mc, 2))
    cli.use_cuda !== nothing && (overrides[:use_cuda] = cli.use_cuda)
    overrides[:objective] = objective
    if objective === :euler_fb_bcmc
        overrides[:bcmc_auto_N] = true
        cli.bcmc_budget !== nothing && (overrides[:bcmc_budget_T] = cli.bcmc_budget)
        cli.bcmc_update !== nothing &&
            (overrides[:bcmc_update_every] = max(cli.bcmc_update, 1))
    else
        overrides[:bcmc_auto_N] = false
    end
    cfg = maybe_merge_nn(cfg, overrides)
    return cfg
end

function flatten_array(values)
    if values isa AbstractArray
        return vec(Float64.(collect(values)))
    else
        return Float64[]
    end
end

function describe_abs(values::AbstractVector{<:Real})
    if isempty(values)
        return (mean = NaN, median = NaN, p95 = NaN, max = NaN, std = NaN)
    end
    abs_vals = abs.(values)
    return (
        mean = mean(abs_vals),
        median = Statistics.quantile(abs_vals, 0.5),
        p95 = Statistics.quantile(abs_vals, 0.95),
        max = maximum(abs_vals),
        std = length(abs_vals) > 1 ? std(abs_vals) : 0.0,
    )
end

function describe_values(values::AbstractVector{<:Real})
    if isempty(values)
        return (mean = NaN, std = NaN, p95 = NaN, min = NaN, max = NaN)
    end
    return (
        mean = mean(values),
        std = length(values) > 1 ? std(values) : 0.0,
        p95 = Statistics.quantile(values, 0.95),
        min = minimum(values),
        max = maximum(values),
    )
end

function monotonicity_stats(cvals::AbstractArray{<:Real})
    data = Float64.(collect(cvals))
    if ndims(data) == 1
        diffs = diff(data)
        total = max(length(diffs), 1)
        violations = count(<(-1e-8), diffs)
        share = total == 0 ? 1.0 : 1.0 - violations / total
        min_slope = isempty(diffs) ? 0.0 : minimum(diffs)
        max_slope = isempty(diffs) ? 0.0 : maximum(diffs)
        return share, min_slope, max_slope
    elseif ndims(data) == 2
        Na, Nz = size(data)
        total = max(Na - 1, 0) * Nz
        violations = 0
        min_slope = Inf
        max_slope = -Inf
        for j = 1:Nz
            col = view(data, :, j)
            diffs = diff(col)
            isempty(diffs) && continue
            violations += count(<(-1e-8), diffs)
            min_slope = min(min_slope, minimum(diffs))
            max_slope = max(max_slope, maximum(diffs))
        end
        share = total == 0 ? 1.0 : (total - violations) / total
        min_slope = isfinite(min_slope) ? min_slope : 0.0
        max_slope = isfinite(max_slope) ? max_slope : 0.0
        return share, min_slope, max_slope
    else
        reshaped = reshape(data, size(data, 1), :)
        return monotonicity_stats(reshaped)
    end
end

binding_share(a_next, a_min::Real; tol = 1e-8) = begin
    vals = Float64.(collect(a_next))
    total = length(vals)
    total == 0 && return 0.0
    hits = count(v -> v ≤ a_min + tol, vals)
    return hits / total
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

function collect_metrics(
    objective::Symbol,
    seed::Int,
    cfg::NamedTuple,
    model,
    method,
    sol_nn,
    notes::AbstractString,
)
    P = ThesisProject.get_params(model)
    G = ThesisProject.get_grids(model)
    S = ThesisProject.get_shocks(model)
    model_name =
        hasproperty(cfg, :model) && hasproperty(cfg.model, :name) ? Symbol(cfg.model.name) :
        Symbol("unknown")
    agrid = G[:a].grid
    Na = length(agrid)
    Nz = shock_count(S)

    resid_vec = flatten_array(sol_nn.resid)
    c_vals = Float64.(collect(sol_nn.c))
    a_next_vals = Float64.(collect(sol_nn.a_next))

    abs_desc = describe_abs(resid_vec)
    pol_desc = describe_values(vec(c_vals))
    share_mono, min_slope, max_slope = monotonicity_stats(sol_nn.c)
    bind_share = binding_share(a_next_vals, G[:a].min)
    c_floor = 1.05f0 * 1.0f-3
    floor_hits = count(x -> x ≤ c_floor, c_vals)

    mc_mean, mc_p95, mc_max = mc_stats_tuple(sol_nn.eval_mc)
    gh_mean, gh_p95, gh_max = mc_stats_tuple(sol_nn.eval_gh)

    opts = method.opts
    nn_opts = sol_nn.opts

    model_id = hash_hex(canonicalize_cfg(cfg))

    return RunRecord(
        objective = objective,
        seed = seed,
        model_name = model_name,
        runtime = Float64(nn_opts.runtime),
        epochs_target = Int(opts.epochs),
        epochs_run = Int(sol_nn.iters),
        batch = Int(opts.batch),
        samples_per_epoch = Int(opts.samples_per_epoch),
        n_mc = Int(opts.n_mc),
        bcmc_auto = nt_get(opts, :bcmc_auto_N, false),
        bcmc_budget = nt_get(opts, :bcmc_budget_T, nothing),
        bcmc_update = nt_get(opts, :bcmc_update_every, nothing),
        use_cuda = cli.use_cuda === nothing ? false : cli.use_cuda,
        device = nt_get(nn_opts, :device, :cpu),
        converged = Bool(sol_nn.converged),
        mean_abs_resid = abs_desc.mean,
        median_abs_resid = abs_desc.median,
        p95_abs_resid = abs_desc.p95,
        max_abs_resid = abs_desc.max,
        resid_std = abs_desc.std,
        policy_mean = pol_desc.mean,
        policy_std = pol_desc.std,
        policy_p95 = pol_desc.p95,
        policy_min = pol_desc.min,
        policy_max = pol_desc.max,
        binding_share = bind_share,
        monotonic_share = share_mono,
        monotonic_min_slope = min_slope,
        monotonic_max_slope = max_slope,
        consumption_floor_hits = floor_hits,
        grid_points = length(c_vals),
        eval_mc_mean = mc_mean,
        eval_mc_p95 = mc_p95,
        eval_mc_max = mc_max,
        eval_gh_mean = gh_mean,
        eval_gh_p95 = gh_p95,
        eval_gh_max = gh_max,
        max_resid = Float64(sol_nn.max_resid),
        model_id = model_id,
        Na = Na,
        Nz = Nz,
        notes = String(notes),
    )
end

function run_single_case(
    base_cfg,
    objective::Symbol,
    cli::CLIOptions,
    seed::Int;
    verbose::Bool,
)
    cfg = prepare_run_cfg(base_cfg, objective, cli, seed)
    model = ThesisProject.build_model(cfg)
    method = ThesisProject.build_method(cfg)
    rng = derive_rng(cfg.random.master_rng, :nn_kernel)
    verbose && @info "Running NN solver" objective seed method_opts = method.opts
    sol_nn = solve_nn(model; opts = method.opts, rng = rng)
    return collect_metrics(objective, seed, cfg, model, method, sol_nn, cli.notes)
end

function write_csv(path::AbstractString, header::Vector{String}, rows::Vector{<:Tuple})
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            vals = Any[row...]
            cells = Vector{String}(undef, length(vals))
            for (i, v) in enumerate(vals)
                if v === missing || v === nothing
                    cells[i] = ""
                elseif v isa Bool
                    cells[i] = v ? "true" : "false"
                elseif v isa Float64
                    cells[i] = @sprintf("%.8g", v)
                else
                    cells[i] = string(v)
                end
            end
            println(io, join(cells, ","))
        end
    end
end

function write_runs_csv(path::AbstractString, runs::Vector{RunRecord})
    header = [
        "objective",
        "seed",
        "runtime",
        "epochs_target",
        "epochs_run",
        "batch",
        "samples_per_epoch",
        "n_mc",
        "bcmc_auto",
        "bcmc_budget",
        "bcmc_update",
        "use_cuda",
        "device",
        "converged",
        "mean_abs_resid",
        "median_abs_resid",
        "p95_abs_resid",
        "max_abs_resid",
        "resid_std",
        "policy_mean",
        "policy_std",
        "policy_p95",
        "policy_min",
        "policy_max",
        "binding_share",
        "monotonic_share",
        "monotonic_min_slope",
        "monotonic_max_slope",
        "consumption_floor_hits",
        "grid_points",
        "eval_mc_mean",
        "eval_mc_p95",
        "eval_mc_max",
        "eval_gh_mean",
        "eval_gh_p95",
        "eval_gh_max",
        "max_resid",
        "model_id",
        "model_name",
        "Na",
        "Nz",
        "notes",
    ]
    rows = [
        (
            String(record.objective),
            record.seed,
            record.runtime,
            record.epochs_target,
            record.epochs_run,
            record.batch,
            record.samples_per_epoch,
            record.n_mc,
            record.bcmc_auto,
            record.bcmc_budget,
            record.bcmc_update,
            record.use_cuda,
            record.device,
            record.converged,
            record.mean_abs_resid,
            record.median_abs_resid,
            record.p95_abs_resid,
            record.max_abs_resid,
            record.resid_std,
            record.policy_mean,
            record.policy_std,
            record.policy_p95,
            record.policy_min,
            record.policy_max,
            record.binding_share,
            record.monotonic_share,
            record.monotonic_min_slope,
            record.monotonic_max_slope,
            record.consumption_floor_hits,
            record.grid_points,
            record.eval_mc_mean,
            record.eval_mc_p95,
            record.eval_mc_max,
            record.eval_gh_mean,
            record.eval_gh_p95,
            record.eval_gh_max,
            record.max_resid,
            record.model_id,
            String(record.model_name),
            record.Na,
            record.Nz,
            record.notes,
        ) for record in runs
    ]
    write_csv(path, header, rows)
end

function mean_skipmissing(values)
    buf = Float64[]
    for v in values
        v === missing && continue
        push!(buf, Float64(v))
    end
    isempty(buf) && return missing
    return mean(buf)
end

function summarize_group(objective::Symbol, group::Vector{RunRecord})
    runtimes = [r.runtime for r in group]
    mean_resids = [r.mean_abs_resid for r in group]
    mc_means = [r.eval_mc_mean for r in group]
    mc_p95 = [r.eval_mc_p95 for r in group]
    mc_max = [r.eval_mc_max for r in group]
    summaries = Dict(
        :objective => String(objective),
        :runs => length(group),
        :runtime_mean => mean(runtimes),
        :runtime_sd => (length(runtimes) > 1 ? std(runtimes) : 0.0),
        :epochs_target => mean([r.epochs_target for r in group]),
        :epochs_run_mean => mean([r.epochs_run for r in group]),
        :mean_abs_resid => mean(mean_resids),
        :p95_abs_resid => mean([r.p95_abs_resid for r in group]),
        :policy_p95_mean => mean([r.policy_p95 for r in group]),
        :policy_max_mean => mean([r.policy_max for r in group]),
        :binding_share_mean => mean([r.binding_share for r in group]),
        :monotonic_share_mean => mean([r.monotonic_share for r in group]),
        :mc_mean => mean_skipmissing(mc_means),
        :mc_p95 => mean_skipmissing(mc_p95),
        :mc_max => mean_skipmissing(mc_max),
        :max_resid_mean => mean([r.max_resid for r in group]),
        :converged_share => mean([r.converged for r in group]),
        :n_mc_mean => mean([r.n_mc for r in group]),
        :samples_mean => mean([r.samples_per_epoch for r in group]),
        :batch_mean => mean([r.batch for r in group]),
    )
    return summaries
end

function summarize_runs(runs::Vector{RunRecord})
    grouped = Dict{Symbol,Vector{RunRecord}}()
    for rec in runs
        push!(get!(grouped, rec.objective, RunRecord[]), rec)
    end
    summary_rows = NamedTuple[]
    for (objective, records) in sort(collect(grouped); by = first)
        push!(summary_rows, NamedTuple(summarize_group(objective, records)))
    end
    return summary_rows
end

function write_summary_csv(path::AbstractString, summary_rows::Vector{NamedTuple})
    header = [
        "objective",
        "runs",
        "runtime_mean",
        "runtime_sd",
        "epochs_target",
        "epochs_run_mean",
        "mean_abs_resid",
        "p95_abs_resid",
        "policy_p95_mean",
        "policy_max_mean",
        "binding_share_mean",
        "monotonic_share_mean",
        "mc_mean",
        "mc_p95",
        "mc_max",
        "max_resid_mean",
        "converged_share",
        "n_mc_mean",
        "samples_mean",
        "batch_mean",
    ]
    rows = [
        (
            row[:objective],
            row[:runs],
            row[:runtime_mean],
            row[:runtime_sd],
            row[:epochs_target],
            row[:epochs_run_mean],
            row[:mean_abs_resid],
            row[:p95_abs_resid],
            row[:policy_p95_mean],
            row[:policy_max_mean],
            row[:binding_share_mean],
            row[:monotonic_share_mean],
            get(row, :mc_mean, missing),
            get(row, :mc_p95, missing),
            get(row, :mc_max, missing),
            row[:max_resid_mean],
            row[:converged_share],
            row[:n_mc_mean],
            row[:samples_mean],
            row[:batch_mean],
        ) for row in summary_rows
    ]
    write_csv(path, header, rows)
end

function write_manifest(
    path::AbstractString;
    config_path::AbstractString,
    cli::CLIOptions,
    seeds::Vector{Int},
)
    repo = LibGit2.GitRepo(ROOT)
    commit = string(LibGit2.GitHash(LibGit2.head(repo)))
    data = Dict(
        "timestamp" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        "config_path" => config_path,
        "seeds" => seeds,
        "cli" => Dict(
            "epochs" => toml_safe(cli.epochs),
            "samples_per_epoch" => toml_safe(cli.samples),
            "batch" => toml_safe(cli.batch),
            "n_mc" => toml_safe(cli.n_mc),
            "bcmc_budget" => toml_safe(cli.bcmc_budget),
            "bcmc_update_every" => toml_safe(cli.bcmc_update),
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

function write_artifact(path::AbstractString)
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

function insight_lines(summary_rows::Vector{NamedTuple})
    if length(summary_rows) < 2
        return ["- Only one objective was evaluated; run both AiO and bc-MC to compare."]
    end
    by_obj = Dict(row[:objective] => row for row in summary_rows)
    aio = by_obj["euler_fb_aio"]
    bcmc = by_obj["euler_fb_bcmc"]
    function pct_delta(a, b)
        (isfinite(a) && a ≠ 0.0) || return missing
        return 100 * (a - b) / a
    end
    lines = String[]
    Δruntime = pct_delta(aio[:runtime_mean], bcmc[:runtime_mean])
    if Δruntime !== missing
        push!(
            lines,
            @sprintf(
                "- Runtime: bc-MC auto-N averages %.3fs vs AiO %.3fs (%.1f%% change).",
                bcmc[:runtime_mean],
                aio[:runtime_mean],
                Δruntime,
            ),
        )
    end
    Δresid = pct_delta(aio[:p95_abs_resid], bcmc[:p95_abs_resid])
    if Δresid !== missing
        push!(
            lines,
            @sprintf(
                "- Grid Euler residuals (p95 abs): bc-MC %.3e vs AiO %.3e (%.1f%% shift).",
                bcmc[:p95_abs_resid],
                aio[:p95_abs_resid],
                Δresid,
            ),
        )
    end
    if aio[:mc_p95] !== missing && bcmc[:mc_p95] !== missing
        Δmc = pct_delta(aio[:mc_p95], bcmc[:mc_p95])
        Δmc === missing || push!(
            lines,
            @sprintf(
                "- Monte-Carlo residuals (p95 abs): bc-MC %.3e vs AiO %.3e (%.1f%%).",
                bcmc[:mc_p95],
                aio[:mc_p95],
                Δmc,
            ),
        )
    end
    Δbind = pct_delta(aio[:binding_share_mean], bcmc[:binding_share_mean])
    if Δbind !== missing
        push!(
            lines,
            @sprintf(
                "- Borrowing constraint binding share: bc-MC %.1f%% vs AiO %.1f%% (%.1f%% change).",
                100 * bcmc[:binding_share_mean],
                100 * aio[:binding_share_mean],
                Δbind,
            ),
        )
    end
    return lines
end

function write_insights(
    path::AbstractString,
    summary_rows::Vector{NamedTuple},
    cli::CLIOptions,
)
    open(path, "w") do io
        println(io, "# AiO vs bc-MC Auto-N comparison")
        println(io)
        seed_str = join(cli.seeds, ", ")
        note_str = isempty(cli.notes) ? "(none)" : cli.notes
        println(io, "- Config tag: `$(cli.tag)` | Seeds: $(seed_str) | Notes: $(note_str)")
        println(io)
        for line in insight_lines(summary_rows)
            println(io, line)
        end
        println(io)
        println(io, "## Aggregates")
        println(io)
        println(
            io,
            "| objective | runs | runtime (mean±sd) | p95 abs resid | MC p95 | binding share | converged |",
        )
        println(
            io,
            "|-----------|------|-------------------|---------------|--------|----------------|-----------|",
        )
        for row in summary_rows
            runtime = @sprintf("%.3f±%.3f", row[:runtime_mean], row[:runtime_sd])
            p95 = @sprintf("%.2e", row[:p95_abs_resid])
            mc = row[:mc_p95] === missing ? "n/a" : @sprintf("%.2e", row[:mc_p95])
            binding = @sprintf("%.1f%%", 100 * row[:binding_share_mean])
            conv = @sprintf("%.1f%%", 100 * row[:converged_share])
            println(
                io,
                "| $(row[:objective]) | $(row[:runs]) | $(runtime) | $(p95) | $(mc) | $(binding) | $(conv) |",
            )
        end
    end
end

function pretty_print_summary(summary_rows::Vector{NamedTuple})
    println("\nObjective-level summary:")
    for row in summary_rows
        println(
            @sprintf(
                "  %-14s runtime=%.3fs ± %.3f | mean abs resid=%.2e | MC p95=%s | binding=%.1f%% | converged=%.1f%%",
                row[:objective],
                row[:runtime_mean],
                row[:runtime_sd],
                row[:mean_abs_resid],
                row[:mc_p95] === missing ? "n/a" : @sprintf("%.2e", row[:mc_p95]),
                100 * row[:binding_share_mean],
                100 * row[:converged_share],
            ),
        )
    end
end

function prepare_report_state(tag::AbstractString)
    result_dir = ensure_dir(joinpath(RESULT_ROOT, tag))
    return ReportState(
        result_dir,
        joinpath(result_dir, "aio_bcmc_runs.csv"),
        joinpath(result_dir, "aio_bcmc_summary.csv"),
        joinpath(result_dir, "manifest.toml"),
        joinpath(result_dir, "artifact.toml"),
        joinpath(result_dir, "insights.md"),
    )
end

function main()
    cli = parse_cli(ARGS)
    cfg_path = cli.config_path === nothing ? DEFAULT_CONFIG : to_abs_path(cli.config_path)
    isfile(cfg_path) || error("Config path $(cfg_path) not found")
    base_cfg = ThesisProject.load_config(cfg_path)
    base_cfg = UtilsConfig.ensure_master_rng(base_cfg; require = true)
    report = prepare_report_state(cli.tag)

    objectives = (:euler_fb_aio, :euler_fb_bcmc)
    runs = RunRecord[]
    for seed in cli.seeds
        for obj in objectives
            record = run_single_case(base_cfg, obj, cli, seed; verbose = cli.verbose)
            push!(runs, record)
        end
    end

    write_runs_csv(report.runs_path, runs)
    summary_rows = summarize_runs(runs)
    write_summary_csv(report.summary_path, summary_rows)
    write_manifest(
        report.manifest_path;
        config_path = cfg_path,
        cli = cli,
        seeds = cli.seeds,
    )
    write_artifact(report.artifact_path)
    write_insights(report.insights_path, summary_rows, cli)
    pretty_print_summary(summary_rows)
    println("\nArtifacts written to $(report.result_dir)")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    CompareAioVsBcmc.main()
end
