#!/usr/bin/env julia

"""
Robustness sweep over β and σ for the baseline EGM (deterministic and stochastic).

Produces: `outputs/egm_robustness_sweep.csv`

Configure via env vars (optional):
  BETA_LIST  e.g. "0.92,0.95,0.98"
  SIGMA_LIST e.g. "1.0,2.0,3.0"

Run:
  julia --project=. scripts/experiments/robustness_sweep.jl [--Na=..] [--Nz=..] [--tol=..] [--tol_pol=..]
"""

module RobustnessSweep

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)
using Dates
using Printf
using Random
using ThesisProject
using Statistics: mean

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

ensure_outputs_dir() = (out = joinpath(ROOT, "outputs"); isdir(out) || mkpath(out); out)

"""Parse comma-separated floats env var into a Vector{Float64}."""
function parse_list(envname::AbstractString, default::Vector{Float64})
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

# Parse simple CLI flags --Na=..., --Nz=..., --tol=..., --tol_pol=...
function parse_cli(args)
    Na = nothing
    Nz = nothing
    tol = nothing
    tol_pol = nothing
    for arg in args
        if startswith(arg, "--Na=")
            Na = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--Nz=")
            Nz = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--tol_pol=")
            tol_pol = parse(Float64, split(arg, "=", limit = 2)[2])
        end
    end
    return (; Na, Nz, tol, tol_pol)
end

function safe_solve(cfg; stochastic::Bool, opts)
    # Toggle shocks.active
    if haskey(cfg, :shocks)
        cfg[:shocks][:active] = stochastic
    elseif stochastic || opts.Nz !== nothing
        cfg[:shocks] = Dict{Symbol,Any}(:active => stochastic)
    end

    if opts.Nz !== nothing && haskey(cfg, :shocks)
        cfg[:shocks][:Nz] = opts.Nz
    end
    if opts.Na !== nothing
        cfg[:grids][:Na] = opts.Na
    end
    if opts.tol !== nothing
        cfg[:solver][:tol] = opts.tol
    end
    if opts.tol_pol !== nothing
        cfg[:solver][:tol_pol] = opts.tol_pol
    end

    try
        model = ThesisProject.build_model(cfg)
        method = ThesisProject.build_method(cfg)
        sol = ThesisProject.solve(model, method, cfg)
        return (:ok, sol)
    catch err
        return (:error, sprint(showerror, err))
    end
end

function ee_stats(sol)
    pol_c = sol.policy[:c]
    ee = pol_c.euler_errors
    ee_mat = pol_c.euler_errors_mat
    if ee_mat === nothing
        # deterministic
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
    opts = parse_cli(ARGS)
    outdir = ensure_outputs_dir()
    outpath = joinpath(outdir, "egm_robustness_sweep.csv")

    # Base config – deterministic baseline schema works for both (we toggle shocks.active)
    base_cfg = ThesisProject.load_config(joinpath(ROOT, "config", "simple_baseline.yaml"))
    ThesisProject.validate_config(base_cfg)

    betas = parse_list("BETA_LIST", [0.92, 0.95, 0.96, 0.98])
    sigmas = parse_list("SIGMA_LIST", [1.0, 2.0, 3.0, 4.0])

    ts = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")
    header = [
        "timestamp",
        "case",
        "beta",
        "sigma",
        "status",
        "error",
        "run_id",
        "iters",
        "converged",
        "max_resid",
        "tol",
        "tol_pol",
        "interp_kind",
        "runtime",
        "ee_max",
        "ee_min",
        "ee_mean",
    ]

    rows = Vector{NTuple{17,Any}}()

    for b in betas, s in sigmas
        for stochastic in (false, true)
            cfg = deepcopy(base_cfg)
            cfg[:params][:β] = b
            cfg[:params][:σ] = s

            case = stochastic ? "stochastic" : "deterministic"
            status, payload = safe_solve(cfg; stochastic, opts)

            if status == :ok
                sol = payload
                meta = sol.metadata
                diag = sol.diagnostics
                ee = ee_stats(sol)
                push!(
                    rows,
                    (
                        ts,
                        case,
                        b,
                        s,
                        "ok",
                        "",
                        diag.model_id,
                        meta[:iters],
                        meta[:converged],
                        meta[:max_resid],
                        meta[:tol],
                        meta[:tol_pol],
                        meta[:interp_kind],
                        diag.runtime,
                        ee.ee_max,
                        ee.ee_min,
                        ee.ee_mean,
                    ),
                )
            else
                errmsg = String(payload)
                push!(
                    rows,
                    (
                        ts,
                        case,
                        b,
                        s,
                        "error",
                        errmsg,
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
                    ),
                )
            end
        end
    end

    open_write(outpath, header, rows)
    println("Wrote sweep CSV: " * outpath)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    RobustnessSweep.main()
end
