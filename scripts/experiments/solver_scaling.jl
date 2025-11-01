#!/usr/bin/env julia

module SolverScaling

"""Batch experiments comparing solver performance across CSVAR dimensions and
correlation structures.

The script loads a base configuration (defaults to
`config/csvar_template.yaml`), synthesises targeted parameter overrides using
`CSVARParamGrid`, runs the requested solvers, and records runtime/convergence
diagnostics into tidy CSV and JSON outputs. This provides an automated harness
to measure how solver behaviour evolves with higher dimensionality and richer
correlation patterns.
"""

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)

using Dates
using LinearAlgebra
using JSON3
using Printf
using Random
using Statistics
using ThesisProject

const DEFAULT_BLAS_THREADS = parse(Int, get(ENV, "SOLVER_SCALING_BLAS_THREADS", "20"))
try
    LinearAlgebra.BLAS.set_num_threads(DEFAULT_BLAS_THREADS)
catch err
    @warn "Failed to set BLAS threads" threads = DEFAULT_BLAS_THREADS err = err
end

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_BASE_CFG = joinpath(ROOT, "config", "csvar_template.yaml")
const DEFAULT_OUT_DIR = joinpath(ROOT, "outputs", "experiments")
const DEFAULT_OUT_FILE = "solver_scaling.csv"

include(joinpath(@__DIR__, "..", "utils", "config_helpers.jl"))
include(joinpath(@__DIR__, "..", "utils", "csvar_param_grid.jl"))
using .ScriptConfigHelpers: dict_to_namedtuple, get_nested, merge_section
using .CSVARParamGrid

struct ExperimentOptions
    base_path::String
    out_dir::String
    out_file::String
    solvers::Vector{Symbol}
    scenarios::Vector{Symbol}
    dims::Vector{Int}
    rotations::Vector{Symbol}
    repeats::Int
    summary_only::Bool
end

function _trim(str::AbstractString)
    return strip(str)
end

function _split_list(arg::AbstractString)
    isempty(arg) && return String[]
    return [_trim(part) for part in split(arg, ",") if !isempty(_trim(part))]
end

const _SCENARIO_ALIASES = Dict(
    "baseline" => :baseline,
    "base" => :baseline,
    "dim" => :dimension,
    "dimension" => :dimension,
    "corr" => :correlation,
    "correlation" => :correlation,
    "rot" => :rotation,
    "rotation" => :rotation,
)

const _ROTATION_ALIASES = Dict("pca" => :pca, "cholesky" => :cholesky)

const _SOLVER_ALIASES = Dict(
    "nn" => :NN,
    "timeiteration" => :TimeIteration,
    "time_iteration" => :TimeIteration,
    "ti" => :TimeIteration,
    "egm" => :EGM,
    "projection" => :Projection,
    "perturbation" => :Perturbation,
)

function parse_cli(args)::ExperimentOptions
    base_path = DEFAULT_BASE_CFG
    out_dir = DEFAULT_OUT_DIR
    out_file = DEFAULT_OUT_FILE
    solvers = Symbol[:TimeIteration, :EGM, :NN]
    scenarios = Symbol[:baseline, :dimension, :correlation, :rotation]
    dims = [2, 3, 4, 5]
    rotations = Symbol[:pca, :cholesky]
    repeats = 1
    summary_only = false

    for arg in args
        if startswith(arg, "--base=")
            base_path = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out-dir=")
            out_dir = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out=")
            out_file = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--solvers=")
            entries = _split_list(split(arg, "=", limit = 2)[2])
            solvers = Symbol[]
            for entry in entries
                key = lowercase(entry)
                push!(solvers, get(_SOLVER_ALIASES, key, Symbol(entry)))
            end
        elseif startswith(arg, "--scenarios=")
            entries = _split_list(split(arg, "=", limit = 2)[2])
            scenarios = Symbol[]
            for entry in entries
                key = lowercase(entry)
                push!(scenarios, get(_SCENARIO_ALIASES, key, Symbol(entry)))
            end
        elseif startswith(arg, "--dim=")
            value = split(arg, "=", limit = 2)[2]
            dims = [parse(Int, value)]
        elseif startswith(arg, "--dims=")
            entries = _split_list(split(arg, "=", limit = 2)[2])
            dims = [parse(Int, entry) for entry in entries]
        elseif startswith(arg, "--rotations=")
            entries = _split_list(split(arg, "=", limit = 2)[2])
            rotations = Symbol[]
            for entry in entries
                key = lowercase(entry)
                push!(rotations, get(_ROTATION_ALIASES, key, Symbol(entry)))
            end
        elseif startswith(arg, "--repeats=")
            repeats = max(1, parse(Int, split(arg, "=", limit = 2)[2]))
        elseif arg == "--summary-only"
            summary_only = true
        end
    end

    isempty(solvers) && push!(solvers, :NN)
    isempty(scenarios) && push!(scenarios, :baseline)
    isempty(dims) && (dims = [length(ThesisProject.load_config(base_path).params.y)])
    isempty(rotations) && push!(rotations, :pca)

    return ExperimentOptions(
        base_path,
        out_dir,
        out_file,
        solvers,
        unique(scenarios),
        unique(dims),
        unique(rotations),
        repeats,
        summary_only,
    )
end

function _to_matrix(data)
    if data isa AbstractMatrix
        return Matrix{Float64}(data)
    elseif data isa AbstractVector
        rows = length(data)
        rows == 0 && return Matrix{Float64}(undef, 0, 0)
        cols = length(data[1])
        M = Matrix{Float64}(undef, rows, cols)
        for (i, row) in enumerate(data)
            M[i, :] .= Float64.(row)
        end
        return M
    else
        error("Expected matrix or vector-of-vectors representation")
    end
end

function _mean_diag(M::AbstractMatrix)
    return size(M, 1) == 0 ? 0.0 : mean(diag(M))
end

function _total_income(params)::Float64
    y = params.y
    if y isa AbstractVector
        return sum(float.(y))
    else
        return float(y)
    end
end

function _correlation_ratio(cfg)::Float64
    params = cfg.params
    Σ = _to_matrix(params.Σ)
    if isempty(Σ)
        return 0.0
    end
    diag_total = sum(abs, diag(Σ))
    offdiag_total = sum(abs, Σ) - diag_total
    return diag_total == 0 ? 0.0 : offdiag_total / diag_total
end

function _collect_scenarios(base_cfg::NamedTuple, opts::ExperimentOptions)
    scenarios = NamedTuple{(:scenario, :label, :config)}[]
    params = base_cfg.params
    A_base = _to_matrix(params.A)
    Σ_base = _to_matrix(params.Σ)
    total_income = _total_income(params)
    persistence = _mean_diag(A_base)
    variance = _mean_diag(Σ_base)

    scenario_set = Set(opts.scenarios)

    if :baseline in scenario_set
        push!(scenarios, (; scenario = :baseline, label = :baseline, config = base_cfg))
    end

    if :dimension in scenario_set
        dim_overrides = build_dimensional_overrides(
            base_cfg;
            dims = opts.dims,
            total_income = total_income,
            persistence = persistence,
            variance = variance,
        )
        for d in opts.dims
            key = Symbol("diag_", d, "d")
            haskey(dim_overrides, key) || continue
            push!(
                scenarios,
                (; scenario = :dimension, label = key, config = dim_overrides[key]),
            )
        end
    end

    if :correlation in scenario_set
        corr_overrides = build_correlation_overrides(
            base_cfg;
            persistence = persistence,
            variance = variance,
        )
        for key in (:single_offdiag, :toeplitz, :dense)
            haskey(corr_overrides, key) || continue
            push!(
                scenarios,
                (; scenario = :correlation, label = key, config = corr_overrides[key]),
            )
        end
    end

    if :rotation in scenario_set
        include = Tuple(opts.rotations)
        rot_overrides = build_rotation_overrides(base_cfg; include = include)
        for key in include
            haskey(rot_overrides, key) || continue
            push!(
                scenarios,
                (; scenario = :rotation, label = key, config = rot_overrides[key]),
            )
        end
    end

    return scenarios
end

function _ensure_output_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function _rng_for(cfg::NamedTuple, solver::Symbol, rep::Int)
    seed_raw = get_nested(cfg, (:random, :seed), UInt64(0))
    base_seed = Int(seed_raw)
    return Random.MersenneTwister(base_seed + rep + hash(solver))
end

function _get_namedtuple_field(nt::NamedTuple, key::Symbol, default = nothing)
    return hasproperty(nt, key) ? getproperty(nt, key) : default
end

function _get_metadata_field(meta::Dict, key::Symbol, default = nothing)
    return haskey(meta, key) ? meta[key] : default
end

function _collect_metrics(
    scenario::Symbol,
    label::Symbol,
    solver::Symbol,
    rep::Int,
    cfg::NamedTuple,
    sol::ThesisProject.Solution,
)
    metadata = sol.metadata
    diagnostics = sol.diagnostics

    status = haskey(metadata, :error) ? "error" : "ok"
    message = haskey(metadata, :error) ? sprint(showerror, metadata[:error]) : ""

    runtime = _get_namedtuple_field(diagnostics, :runtime, nothing)
    runtime === nothing && (runtime = _get_metadata_field(metadata, :runtime, NaN))
    runtime = runtime isa Number ? Float64(runtime) : NaN

    iterations = _get_namedtuple_field(diagnostics, :iterations, nothing)
    iterations === nothing && (iterations = _get_metadata_field(metadata, :iters, NaN))
    iterations = iterations isa Number ? Float64(iterations) : NaN

    mean_ee = _get_namedtuple_field(diagnostics, :mean_ee, NaN)
    mean_ee = mean_ee isa Number ? Float64(mean_ee) : NaN

    rmse =
        _get_metadata_field(metadata, :rmse, _get_metadata_field(metadata, :max_resid, NaN))
    rmse = rmse isa Number ? Float64(rmse) : NaN
    max_resid = _get_metadata_field(metadata, :max_resid, rmse)
    max_resid = max_resid isa Number ? Float64(max_resid) : NaN

    state_dim = length(cfg.params.y)
    corr_ratio = _correlation_ratio(cfg)

    return (
        timestamp = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        scenario = String(scenario),
        variant = String(label),
        solver = String(solver),
        repeat = rep,
        state_dim = state_dim,
        runtime = runtime,
        iterations = iterations,
        mean_ee = mean_ee,
        rmse = rmse,
        max_resid = max_resid,
        corr_ratio = corr_ratio,
        status = status,
        message = message,
    )
end

function _collect_failure(
    scenario::Symbol,
    label::Symbol,
    solver::Symbol,
    rep::Int,
    cfg::NamedTuple,
    err,
)
    state_dim = length(cfg.params.y)
    corr_ratio = _correlation_ratio(cfg)
    return (
        timestamp = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        scenario = String(scenario),
        variant = String(label),
        solver = String(solver),
        repeat = rep,
        state_dim = state_dim,
        runtime = Float64(NaN),
        iterations = Float64(NaN),
        mean_ee = Float64(NaN),
        rmse = Float64(NaN),
        max_resid = Float64(NaN),
        corr_ratio = corr_ratio,
        status = "exception",
        message = sprint(showerror, err),
    )
end

function _normalize_v_h_history(history)
    result = NamedTuple{
        (:epoch, :old_v_h, :new_v_h, :scale_factor),
        Tuple{Int,Float64,Float64,Float64},
    }[]
    for entry in history
        if !(hasproperty(entry, :epoch) && hasproperty(entry, :new_v_h))
            continue
        end
        epoch = Int(getproperty(entry, :epoch))
        new_v = Float64(getproperty(entry, :new_v_h))
        old_v = hasproperty(entry, :old_v_h) ? Float64(getproperty(entry, :old_v_h)) : new_v
        scale =
            hasproperty(entry, :scale_factor) ? Float64(getproperty(entry, :scale_factor)) :
            (old_v ≈ 0.0 ? 1.0 : new_v / max(old_v, eps(Float64)))
        push!(
            result,
            (; epoch = epoch, old_v_h = old_v, new_v_h = new_v, scale_factor = scale),
        )
    end
    return result
end

function _normalize_n_history(history)
    result = NamedTuple{(:step, :epoch, :N),Tuple{Int,Int,Int}}[]
    for entry in history
        if !(hasproperty(entry, :step) && hasproperty(entry, :N))
            continue
        end
        step = Int(getproperty(entry, :step))
        epoch = hasproperty(entry, :epoch) ? Int(getproperty(entry, :epoch)) : 0
        N = Int(getproperty(entry, :N))
        push!(result, (; step = step, epoch = epoch, N = N))
    end
    return result
end

function _collect_history_success(
    scenario::Symbol,
    label::Symbol,
    solver::Symbol,
    rep::Int,
    cfg::NamedTuple,
    sol::ThesisProject.Solution,
)
    meta = sol.metadata
    rmse_hist = Float64.(get(meta, :rmse_history, Float64[]))
    vh_hist = _normalize_v_h_history(get(meta, :v_h_history, Any[]))
    n_hist = _normalize_n_history(get(meta, :n_history, Any[]))
    state_dim = length(cfg.params.y)
    return (
        timestamp = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        scenario = String(scenario),
        variant = String(label),
        solver = String(solver),
        repeat = rep,
        state_dim = state_dim,
        status = "ok",
        rmse_history = rmse_hist,
        v_h_history = vh_hist,
        n_history = n_hist,
    )
end

function _collect_history_failure(
    scenario::Symbol,
    label::Symbol,
    solver::Symbol,
    rep::Int,
    cfg::NamedTuple,
    err,
)
    state_dim = length(cfg.params.y)
    return (
        timestamp = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        scenario = String(scenario),
        variant = String(label),
        solver = String(solver),
        repeat = rep,
        state_dim = state_dim,
        status = "exception",
        message = sprint(showerror, err),
        rmse_history = Float64[],
        v_h_history = _normalize_v_h_history(Any[]),
        n_history = _normalize_n_history(Any[]),
    )
end

function _write_csv(path::AbstractString, rows; append::Bool = false)
    header = [
        "timestamp",
        "scenario",
        "variant",
        "solver",
        "repeat",
        "state_dim",
        "runtime",
        "iterations",
        "mean_ee",
        "rmse",
        "max_resid",
        "corr_ratio",
        "status",
        "message",
    ]

    write_header = true
    if append && isfile(path)
        write_header = filesize(path) == 0
    else
        append = false
    end

    open(path, append ? "a" : "w") do io
        if write_header
            println(io, join(header, ","))
        end
        for row in rows
            fields = (
                row.timestamp,
                row.scenario,
                row.variant,
                row.solver,
                string(row.repeat),
                string(row.state_dim),
                string(row.runtime),
                string(row.iterations),
                string(row.mean_ee),
                string(row.rmse),
                string(row.max_resid),
                string(row.corr_ratio),
                row.status,
                replace(row.message, '\n' => ' '),
            )
            println(io, join(fields, ","))
        end
    end
    return path
end

function _write_history(path::AbstractString, rows; append::Bool = false)
    isempty(rows) && return path
    open(path, append ? "a" : "w") do io
        for row in rows
            JSON3.write(io, row)
            write(io, '\n')
        end
    end
    return path
end

function _summarise(rows)
    groups = Dict{Tuple{String,String},Vector{typeof(rows[1])}}()
    for row in rows
        key = (row.scenario, row.solver)
        group = get!(groups, key, Vector{typeof(rows[1])}())
        push!(group, row)
    end

    println("\nSummary (status == ok):")
    println(
        rpad("Scenario", 14),
        rpad("Solver", 12),
        rpad("Count", 8),
        rpad("Runtime¯", 14),
        rpad("RMSE¯", 14),
    )

    for (key, group) in sort(collect(groups))
        ok_rows = filter(row -> row.status == "ok", group)
        count = length(ok_rows)
        runtime_vals = [
            row.runtime for row in ok_rows if row.runtime isa Number && !isnan(row.runtime)
        ]
        rmse_vals = [row.rmse for row in ok_rows if row.rmse isa Number && !isnan(row.rmse)]
        runtime_mean = isempty(runtime_vals) ? NaN : mean(runtime_vals)
        rmse_mean = isempty(rmse_vals) ? NaN : mean(rmse_vals)
        @printf(
            "%-14s%-12s%-8d%-14.6e%-14.6e\n",
            key[1],
            key[2],
            count,
            runtime_mean,
            rmse_mean
        )
    end
end

function run()
    opts = parse_cli(ARGS)
    base_cfg = ThesisProject.load_config(opts.base_path)
    scenarios = _collect_scenarios(base_cfg, opts)
    isempty(scenarios) && error("No scenarios selected; adjust --scenarios")

    rows = NamedTuple[]
    history_rows = NamedTuple[]

    for scenario_info in scenarios
        scenario = scenario_info.scenario
        label = scenario_info.label
        cfg = scenario_info.config
        for rep = 1:opts.repeats
            for solver in opts.solvers
                cfg_solver = merge_section(cfg, :solver, (; method = solver))
                rng = _rng_for(cfg_solver, solver, rep)
                try
                    sol = ThesisProject.solve(cfg_solver; rng = rng)
                    sol isa Vector && (sol = sol[1])
                    metrics =
                        _collect_metrics(scenario, label, solver, rep, cfg_solver, sol)
                    push!(rows, metrics)
                    push!(
                        history_rows,
                        _collect_history_success(
                            scenario,
                            label,
                            solver,
                            rep,
                            cfg_solver,
                            sol,
                        ),
                    )
                catch err
                    push!(
                        rows,
                        _collect_failure(scenario, label, solver, rep, cfg_solver, err),
                    )
                    push!(
                        history_rows,
                        _collect_history_failure(
                            scenario,
                            label,
                            solver,
                            rep,
                            cfg_solver,
                            err,
                        ),
                    )
                end
            end
        end
    end

    _summarise(rows)

    opts.summary_only && return

    _ensure_output_dir(opts.out_dir)
    out_csv = if isabspath(opts.out_file)
        normpath(opts.out_file)
    elseif occursin("/", opts.out_file) || occursin("\\", opts.out_file)
        normpath(joinpath(ROOT, opts.out_file))
    else
        joinpath(opts.out_dir, opts.out_file)
    end
    _ensure_output_dir(dirname(out_csv))
    append = isfile(out_csv)
    append ? println("Appending results to existing CSV: $(out_csv)") :
    println("Writing new CSV: $(out_csv)")
    _write_csv(out_csv, rows; append = append)
    out_history =
        endswith(lowercase(out_csv), ".csv") ?
        replace(out_csv, r"\.csv$" => "_history.jsonl") : out_csv * "_history.jsonl"
    append_history = isfile(out_history)
    _write_history(out_history, history_rows; append = append_history)
    println()
    println("Wrote results to:\n  CSV : $(out_csv)\n  Hist: $(out_history)")
end

run()

end # module
