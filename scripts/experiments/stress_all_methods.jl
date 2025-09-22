#!/usr/bin/env julia

"""
Stress test all methods (EGM, Projection, Perturbation) across values of β and σ.

Runs both deterministic and stochastic cases by toggling `shocks.active`.
Logs successes and failures (with error types/messages and stacktraces) to CSV.

Configure via env vars (optional):
  - Β_LIST   e.g. "0.92,0.95,0.98"
  - Σ_LIST  e.g. "1.0,2.0,3.0"
  - METHODS     e.g. "EGM,Projection,Perturbation" (defaults to all)

CLI flags (optional):
  --Na=...        integer grid points for assets
  --Nz=...        integer shock states when stochastic
  --tol=...       Float64 tolerance (if applicable)
  --tol_pol=...   Float64 policy tolerance (EGM)
  --out=...       custom output CSV path

Usage:
  julia --project=. scripts/experiments/stress_all_methods.jl [--Na=..] [--Nz=..] [--tol=..] [--tol_pol=..] [--out=...]
"""

module StressAllMethods

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)

using Dates
using Printf
using Random
using LinearAlgebra
using Statistics: mean
using ThesisProject

include(joinpath(@__DIR__, "..", "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

ensure_outputs_dir() = (out = joinpath(ROOT, "outputs"); isdir(out) || mkpath(out); out)

"""Parse comma-separated floats env var into a Vector{Float64}."""
function parse_float_list(envname::AbstractString, default::Vector{Float64})
    s = get(ENV, envname, "")
    if isempty(strip(s))
        return default
    end
    try
        return [parse(Float64, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
    catch err
        @warn "Failed to parse $envname; using defaults" s error = err
        return default
    end
end

"""Parse comma-separated methods env var into Vector{Symbol}."""
function parse_methods(envname::AbstractString, default::Vector{Symbol})
    s = get(ENV, envname, "")
    if isempty(strip(s))
        return default
    end
    syms = Symbol[]
    for x in split(s, ",")
        t = strip(x)
        isempty(t) && continue
        push!(syms, Symbol(t))
    end
    return isempty(syms) ? default : syms
end

"""Parse simple CLI flags --Na=..., --Nz=..., --tol=..., --tol_pol=..., --out=..."""
function parse_cli(args)
    Na = nothing
    Nz = nothing
    tol = nothing
    tol_pol = nothing
    outpath = nothing
    for arg in args
        if startswith(arg, "--Na=")
            Na = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--Nz=")
            Nz = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--tol_pol=")
            tol_pol = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--out=")
            outpath = split(arg, "=", limit = 2)[2]
        end
    end
    return (; Na, Nz, tol, tol_pol, outpath)
end

"""Return (:ok, sol) or (:error, (etype, emsg, estack)) for a given cfg/method/case."""
function safe_solve(cfg::NamedTuple; method::Symbol, stochastic::Bool, opts)
    cfg_local = merge_section(cfg, :solver, (; method = method))
    cfg_local = merge_config(cfg_local, (; method = method))

    shocks_updates = Dict{Symbol,Any}(:active => stochastic)
    opts.Nz !== nothing && (shocks_updates[:Nz] = opts.Nz)
    if get_nested(cfg_local, (:shocks,), nothing) !== nothing ||
       stochastic ||
       opts.Nz !== nothing
        cfg_local = merge_section(cfg_local, :shocks, dict_to_namedtuple(shocks_updates))
    end

    if opts.Na !== nothing
        cfg_local = merge_section(cfg_local, :grids, (; Na = opts.Na))
    end
    solver_updates = Dict{Symbol,Any}()
    opts.tol !== nothing && (solver_updates[:tol] = opts.tol)
    opts.tol_pol !== nothing && (solver_updates[:tol_pol] = opts.tol_pol)
    if !isempty(solver_updates)
        cfg_local = merge_section(cfg_local, :solver, dict_to_namedtuple(solver_updates))
    end

    cfg_dict = namedtuple_to_dict(cfg_local)

    try
        model = ThesisProject.build_model(cfg_dict)
        meth = ThesisProject.build_method(cfg_dict)
        sol = ThesisProject.solve(model, meth, cfg_dict)
        return (:ok, sol)
    catch err
        # Capture error type, message, and stacktrace string
        bt = catch_backtrace()
        estack = sprint(showerror, err, bt)
        emsg = sprint(showerror, err)
        return (:error, (string(typeof(err)), emsg, estack))
    end
end

"""Compute Euler error stats, robust to det/stoch cases."""
function ee_stats(sol)
    pol_c = sol.policy[:c]
    ee = pol_c.euler_errors
    ee_mat = pol_c.euler_errors_mat
    if ee_mat === nothing
        mx = maximum(skipmissing(ee))
        mn = minimum(skipmissing(ee))
        av = mean(skipmissing(ee))
        return (ee_max = mx, ee_min = mn, ee_mean = av)
    else
        mx = maximum(skipmissing(vec(ee_mat)))
        mn = minimum(skipmissing(vec(ee_mat)))
        av = mean(skipmissing(vec(ee_mat)))
        return (ee_max = mx, ee_min = mn, ee_mean = av)
    end
end

function open_write(path::AbstractString, header::Vector{<:AbstractString}, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            strrow = map(x -> x === nothing ? "" : sprint(print, x), row)
            println(io, join(strrow, ","))
        end
    end
end

function main()
    # Determinism: fixed seed and single-threaded BLAS
    seed = parse(Int, get(ENV, "SEED", "20240915"))
    Random.seed!(seed)
    try
        BLAS.set_num_threads(1)
    catch
    end
    opts = parse_cli(ARGS)
    outdir = ensure_outputs_dir()
    default_out = joinpath(outdir, "stress_all_methods.csv")
    outpath = something(opts.outpath, default_out)

    # Base config - deterministic baseline works for both; we toggle shocks.active
    base_cfg_loaded =
        ThesisProject.load_config(joinpath(ROOT, "config", "simple_baseline.yaml"))
    ThesisProject.validate_config(base_cfg_loaded)
    base_cfg = dict_to_namedtuple(base_cfg_loaded)

    # Strictly require Greek-letter keys in the loaded config
    params0 = maybe_namedtuple(get_nested(base_cfg, (:params,), NamedTuple()))
    @assert params0 !== NamedTuple() "Config missing :params section"
    @assert hasproperty(params0, Symbol("\u03b2")) "Config :params must include :\\u03b2"
    @assert hasproperty(params0, Symbol("\u03c3")) "Config :params must include :\\u03c3"
    β_key = Symbol("\u03b2")
    σ_key = Symbol("\u03c3")

    βs = parse_float_list("Β_LIST", [0.92, 0.95, 0.96, 0.98])
    σs = parse_float_list("Σ_LIST", [1.0, 2.0, 3.0, 4.0])
    methods = parse_methods("METHODS", [:EGM, :Projection, :Perturbation])

    ts = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")

    header = [
        "timestamp",
        "method",
        "case",
        "β",
        "σ",
        "status",
        "error_type",
        "error_message",
        "stacktrace",
        "run_id",
        "iters",
        "converged",
        "max_resid",
        "tol",
        "tol_pol",
        "interp_kind",
        "order",
        "runtime",
        "ee_max",
        "ee_min",
        "ee_mean",
        "valid",
    ]

    rows = Vector{NTuple{22,Any}}()

    for m in methods, b in βs, s in σs
        for stochastic in (false, true)
            cfg = merge_section(base_cfg, :params, Dict{Symbol,Any}(β_key => b, σ_key => s))

            case = stochastic ? "stochastic" : "deterministic"
            status, payload = safe_solve(cfg; method = m, stochastic, opts)

            if status == :ok
                sol = payload
                meta = sol.metadata
                diag = sol.diagnostics
                ee = ee_stats(sol)
                push!(
                    rows,
                    (
                        ts,
                        String(m),
                        case,
                        b,
                        s,
                        "ok",
                        "",
                        "",
                        "",
                        get(diag, :model_id, missing),
                        get(meta, :iters, missing),
                        get(meta, :converged, missing),
                        get(meta, :max_resid, missing),
                        get(meta, :tol, missing),
                        get(meta, :tol_pol, missing),
                        get(meta, :interp_kind, missing),
                        get(meta, :order, missing),
                        get(diag, :runtime, missing),
                        ee.ee_max,
                        ee.ee_min,
                        ee.ee_mean,
                        get(meta, :valid, missing),
                    ),
                )
            else
                etype, emsg, estack = payload
                push!(
                    rows,
                    (
                        ts,
                        String(m),
                        case,
                        b,
                        s,
                        "error",
                        etype,
                        emsg,
                        estack,
                        missing,
                        missing,
                        false,
                        missing,
                        missing,
                        missing,
                        missing,
                        missing,
                        missing,
                        missing,
                        missing,
                        missing,
                        missing,
                    ),
                )
            end
        end
    end

    open_write(outpath, header, rows)
    println("Wrote stress CSV: " * outpath)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    StressAllMethods.main()
end
