#!/usr/bin/env julia

"""
Run the methodology diagnostics sweep described in the thesis supplement.

The script launches EGM solutions across ablations on grid density (`Na`),
Euler-equation tolerance (`tol`), policy-change tolerance (`tol_pol`), and
interpolation kind (`linear` vs `pchip`). Each configuration is solved five
times with different random seeds. Raw per-run metrics and aggregated
statistics (median with IQR, mean ± sd) are written under `results/methodology/`.

Outputs:
  - `results/methodology/methodology_runs.csv`
  - `results/methodology/methodology_summary.csv`
  - `results/methodology/manifest.toml`
  - `results/methodology/artifact.toml`

Environment overrides:
  - `METHODOLOGY_SEEDS="11,19,23,29,31"`
  - `ABLATION_NA="40,60"`
  - `ABLATION_TOL="1e-3,5e-4"`
  - `ABLATION_TOL_POL="1e-6,5e-7"`
  - `ABLATION_INTERP="linear,pchip"`

Usage:
  julia --project=. scripts/thesis/reports/methodology_report.jl [--config=path/to/config.yaml]
"""

module MethodologyReport

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "..")); io = devnull)

using Dates
using InteractiveUtils: versioninfo
using LibGit2
using Printf
using Random
using Statistics
using TOML
using ThesisProject

include(joinpath(@__DIR__, "..", "..", "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const RESULT_DIR = joinpath(ROOT, "results", "methodology")
const RUNS_CSV = joinpath(RESULT_DIR, "methodology_runs.csv")
const SUMMARY_CSV = joinpath(RESULT_DIR, "methodology_summary.csv")
const MANIFEST_PATH = joinpath(RESULT_DIR, "manifest.toml")
const ARTIFACT_PATH = joinpath(RESULT_DIR, "artifact.toml")

ensure_results_dir() = (isdir(RESULT_DIR) || mkpath(RESULT_DIR))

"""Parse a comma-separated env var into a vector using `parse_fn`."""
function parse_list(env::AbstractString, default, parse_fn)
    raw = get(ENV, env, "")
    if isempty(strip(raw))
        return default
    end
    items = split(raw, ",")
    parsed = Any[]
    for item in items
        trimmed = strip(item)
        isempty(trimmed) && continue
        push!(parsed, parse_fn(trimmed))
    end
    return isempty(parsed) ? default : parsed
end

parse_int_list(env, default) = parse_list(env, default, x -> parse(Int, x))
parse_float_list(env, default) = parse_list(env, default, x -> parse(Float64, x))
parse_interp_list(env, default) = parse_list(env, default, x -> Symbol(lowercase(strip(x))))

"""Return `(config_path, seeds)` parsed from CLI/env."""
function parse_cli(args)
    config_path = nothing
    for arg in args
        if startswith(arg, "--config=")
            config_path = split(arg, "=", limit = 2)[2]
        end
    end

    default_seeds = Int[11, 19, 23, 29, 31]
    seeds = parse_int_list("METHODOLOGY_SEEDS", default_seeds)
    return config_path, seeds
end

"""Helper to fetch a property from a NamedTuple/Dict-like object."""
diag_get(nt, key::Symbol, default) = hasproperty(nt, key) ? getproperty(nt, key) : default

"""Formatters per-metric for readable CSV output."""
const METRIC_FORMATTERS = Dict{Symbol,Function}(
    :iterations => x -> @sprintf("%d", round(Int, x)),
    :runtime => x -> @sprintf("%.4f", x),
    :mean_ee => x -> @sprintf("%.3e", x),
    :delta_pol => x -> @sprintf("%.3e", x),
)

function format_median_iqr(metric::Symbol, values::Vector{Float64})
    med = median(values)
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    fmt = METRIC_FORMATTERS[metric]
    return string(fmt(med), " (", fmt(q1), "–", fmt(q3), ")")
end

function format_mean_sd(metric::Symbol, values::Vector{Float64})
    μ = mean(values)
    σ = std(values)
    fmt = METRIC_FORMATTERS[metric]
    return string(fmt(μ), " ± ", fmt(σ))
end

function clean_values(values)
    buf = Float64[]
    for v in values
        if v === missing || v === nothing
            continue
        end
        x = Float64(v)
        if isfinite(x)
            push!(buf, x)
        end
    end
    return buf
end

function write_csv(path::AbstractString, header::Vector{String}, rows::Vector{<:Tuple})
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            values = Any[row...]
            out = Vector{String}(undef, length(values))
            for (i, v) in enumerate(values)
                if v === missing || v === nothing
                    out[i] = ""
                elseif v isa Float64
                    out[i] = @sprintf("%.6g", v)
                else
                    out[i] = string(v)
                end
            end
            println(io, join(out, ","))
        end
    end
end

function write_runs_csv(path, rows)
    header = [
        "timestamp",
        "method",
        "Na",
        "tol",
        "tol_pol",
        "interp_kind",
        "seed",
        "status",
        "iterations",
        "runtime",
        "mean_ee",
        "delta_pol",
        "converged",
        "max_resid",
        "model_id",
        "error",
    ]
    write_csv(path, header, rows)
end

function write_summary_csv(path, rows)
    header = [
        "Na",
        "tol",
        "tol_pol",
        "interp_kind",
        "ok_runs",
        "total_runs",
        "iterations_median_iqr",
        "iterations_mean_sd",
        "runtime_median_iqr",
        "runtime_mean_sd",
        "mean_ee_median_iqr",
        "mean_ee_mean_sd",
        "delta_pol_median_iqr",
        "delta_pol_mean_sd",
    ]
    write_csv(path, header, rows)
end

function write_manifest(
    path;
    config_path,
    seeds,
    Na_list,
    tol_list,
    tol_pol_list,
    interp_list,
)
    repo = LibGit2.GitRepo(ROOT)
    commit = string(LibGit2.GitHash(LibGit2.head(repo)))
    data = Dict(
        "timestamp" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        "config_path" => config_path,
        "method" => "EGM",
        "seeds" => seeds,
        "grid_sizes" => Na_list,
        "tol_list" => tol_list,
        "tol_pol_list" => tol_pol_list,
        "interp_kinds" => map(string, interp_list),
        "git_commit" => commit,
        "manifest" => "Manifest.toml",
    )
    open(path, "w") do io
        TOML.print(io, data)
    end
end

function write_artifact(path)
    repo = LibGit2.GitRepo(ROOT)
    commit = string(LibGit2.GitHash(LibGit2.head(repo)))
    buf_pkg = IOBuffer()
    Pkg.status(; mode = Pkg.PKGMODE_MANIFEST, io = buf_pkg)
    pkg_str = String(take!(buf_pkg))

    buf_ver = IOBuffer()
    versioninfo(buf_ver; verbose = true)
    ver_str = String(take!(buf_ver))

    data = Dict(
        "timestamp" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        "git_commit" => commit,
        "julia_version" => string(VERSION),
        "machine" => Sys.MACHINE,
        "kernel" => Sys.KERNEL,
        "cpu_threads" => Sys.CPU_THREADS,
        "pkg_status" => pkg_str,
        "versioninfo" => ver_str,
    )
    open(path, "w") do io
        TOML.print(io, data)
    end
end

function prepare_config(
    cfg::NamedTuple;
    Na::Int,
    tol::Float64,
    tol_pol::Float64,
    interp::Symbol,
    seed::Int,
)
    cfg_mod = merge_section(cfg, :grids, (; Na = Na))
    solver_updates = (; tol = tol, tol_pol = tol_pol, interp_kind = interp, method = :EGM)
    cfg_mod = merge_section(cfg_mod, :solver, solver_updates)
    cfg_mod = merge_section(cfg_mod, :random, (; seed = seed))
    return cfg_mod
end

function run_case(
    cfg_base::NamedTuple;
    Na::Int,
    tol::Float64,
    tol_pol::Float64,
    interp::Symbol,
    seeds::Vector{Int},
)
    data_rows = Tuple{
        String,
        String,
        Int,
        Float64,
        Float64,
        String,
        Int,
        String,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        String,
        String,
    }[]
    now_ts = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")
    for seed in seeds
        cfg_run = prepare_config(
            cfg_base;
            Na = Na,
            tol = tol,
            tol_pol = tol_pol,
            interp = interp,
            seed = seed,
        )
        try
            model = ThesisProject.build_model(cfg_run)
            method = ThesisProject.build_method(cfg_run)
            sol = ThesisProject.solve(model, method, cfg_run)
            diag = sol.diagnostics
            meta = sol.metadata

            method_name = diag_get(diag, :method, string(cfg_run.solver.method))
            iterations = diag_get(diag, :iterations, get(meta, :iters, missing))
            runtime = diag_get(diag, :runtime, get(meta, :runtime, missing))
            mean_ee = diag_get(diag, :mean_ee, get(meta, :mean_ee, missing))
            delta_pol = diag_get(diag, :delta_pol, get(meta, :delta_pol, missing))
            converged = get(meta, :converged, missing)
            max_resid = get(meta, :max_resid, missing)
            model_id = diag_get(diag, :model_id, get(meta, :model_id, ""))

            push!(
                data_rows,
                (
                    now_ts,
                    string(method_name),
                    Na,
                    tol,
                    tol_pol,
                    string(interp),
                    seed,
                    "ok",
                    iterations,
                    runtime,
                    mean_ee,
                    delta_pol,
                    converged,
                    max_resid,
                    string(model_id),
                    "",
                ),
            )
        catch err
            push!(
                data_rows,
                (
                    now_ts,
                    "EGM",
                    Na,
                    tol,
                    tol_pol,
                    string(interp),
                    seed,
                    "error",
                    missing,
                    missing,
                    missing,
                    missing,
                    missing,
                    missing,
                    "",
                    sprint(showerror, err),
                ),
            )
        end
    end
    return data_rows
end

function summarise_cases(rows)
    grouped = Dict{
        Tuple{Int,Float64,Float64,Symbol},
        Vector{
            Tuple{
                String,
                String,
                Int,
                Float64,
                Float64,
                String,
                Int,
                String,
                Any,
                Any,
                Any,
                Any,
                Any,
                Any,
                String,
                String,
            },
        },
    }()
    for row in rows
        key = (row[3], row[4], row[5], Symbol(row[6]))
        push!(
            get!(
                grouped,
                key,
                Tuple{
                    String,
                    String,
                    Int,
                    Float64,
                    Float64,
                    String,
                    Int,
                    String,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    Any,
                    String,
                    String,
                }[],
            ),
            row,
        )
    end

    summary_rows = Tuple{
        Int,
        Float64,
        Float64,
        String,
        Int,
        Int,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
    }[]
    metrics = (:iterations, :runtime, :mean_ee, :delta_pol)
    metric_indices =
        Dict(:iterations => 9, :runtime => 10, :mean_ee => 11, :delta_pol => 12)

    for (key, group_rows) in sort(collect(grouped); by = x -> x[1])
        Na, tol, tol_pol, interp = key
        total_runs = length(group_rows)
        ok_rows = [r for r in group_rows if r[8] == "ok"]
        ok_count = length(ok_rows)

        stat_strings = Dict{Symbol,String}()
        mean_strings = Dict{Symbol,String}()
        for metric in metrics
            idx = metric_indices[metric]
            values = clean_values(getindex.(ok_rows, idx))
            if isempty(values)
                stat_strings[metric] = ""
                mean_strings[metric] = ""
            else
                stat_strings[metric] = format_median_iqr(metric, values)
                mean_strings[metric] = format_mean_sd(metric, values)
            end
        end

        push!(
            summary_rows,
            (
                Na,
                tol,
                tol_pol,
                string(interp),
                ok_count,
                total_runs,
                stat_strings[:iterations],
                mean_strings[:iterations],
                stat_strings[:runtime],
                mean_strings[:runtime],
                stat_strings[:mean_ee],
                mean_strings[:mean_ee],
                stat_strings[:delta_pol],
                mean_strings[:delta_pol],
            ),
        )
    end

    return summary_rows
end

function main()
    ensure_results_dir()

    cfg_path_cli, seeds = parse_cli(ARGS)
    cfg_path =
        cfg_path_cli === nothing ? joinpath(ROOT, "config", "smoke_cfg_det.yaml") :
        cfg_path_cli
    cfg_base = ThesisProject.load_config(cfg_path)

    Na_list = parse_int_list("ABLATION_NA", [40, 60])
    tol_list = parse_float_list("ABLATION_TOL", [1e-3, 5e-4])
    tol_pol_list = parse_float_list("ABLATION_TOL_POL", [1e-6, 5e-7])
    interp_list = parse_interp_list("ABLATION_INTERP", [:linear, :pchip])

    all_rows = Tuple{
        String,
        String,
        Int,
        Float64,
        Float64,
        String,
        Int,
        String,
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        String,
        String,
    }[]
    for Na in Na_list, tol in tol_list, tol_pol in tol_pol_list, interp in interp_list
        run_rows = run_case(
            cfg_base;
            Na = Na,
            tol = tol,
            tol_pol = tol_pol,
            interp = interp,
            seeds = seeds,
        )
        append!(all_rows, run_rows)
    end

    write_runs_csv(RUNS_CSV, all_rows)
    summary_rows = summarise_cases(all_rows)
    write_summary_csv(SUMMARY_CSV, summary_rows)
    write_manifest(
        MANIFEST_PATH;
        config_path = cfg_path,
        seeds = seeds,
        Na_list = Na_list,
        tol_list = tol_list,
        tol_pol_list = tol_pol_list,
        interp_list = interp_list,
    )
    write_artifact(ARTIFACT_PATH)
    println("Methodology diagnostics written to $(RESULT_DIR)")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    MethodologyReport.main()
end
