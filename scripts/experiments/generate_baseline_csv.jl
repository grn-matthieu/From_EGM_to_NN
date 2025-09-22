#!/usr/bin/env julia

"""
This script generates CSV snapshots for baseline EGM solutions (deterministic and stochastic),
including Euler errors, to help track changes over time.
The model concerned is the simple consumer-saving problem.

Outputs two files in `outputs/`:
- egm_baseline_det.csv
- egm_baseline_stoch.csv

Run with:
  julia --project=. scripts/experiments/generate_baseline_csv.jl [--Na=..] [--Nz=..] [--tol=..] [--tol_pol=..]
"""

module GenerateBaselineCSV

import Pkg
using Dates
using Printf

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

"""
Activate the project immediately so downstream `using ThesisProject` works.
"""
Pkg.activate(ROOT; io = devnull)
using ThesisProject
# Load plotting extension (Project has Plots as a weak dep)
try
    @eval using Plots
catch err
    @warn "Plots not available; plot generation will be skipped" err
end

include(joinpath(@__DIR__, "..", "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

function ensure_outputs_dir()
    outdir = joinpath(ROOT, "outputs")
    isdir(outdir) || mkpath(outdir)
    return outdir
end

function ensure_plots_dir(outdir::AbstractString)
    pdir = joinpath(outdir, "plots")
    isdir(pdir) || mkpath(pdir)
    return pdir
end

function open_write(path::AbstractString, header::Vector{<:AbstractString}, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            # Convert Any to string safely
            strrow = map(x -> x === nothing ? "" : sprint(print, x), row)
            println(io, join(strrow, ","))
        end
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

# Small fun to auto run the config
function run_one(cfg_path::AbstractString, opts)
    cfg_loaded = ThesisProject.load_config(cfg_path)
    ThesisProject.validate_config(cfg_loaded)
    cfg = dict_to_namedtuple(cfg_loaded)

    if opts.Na !== nothing
        cfg = merge_section(cfg, :grids, (; Na = opts.Na))
    end
    solver_updates = Dict{Symbol,Any}()
    opts.tol !== nothing && (solver_updates[:tol] = opts.tol)
    opts.tol_pol !== nothing && (solver_updates[:tol_pol] = opts.tol_pol)
    if !isempty(solver_updates)
        cfg = merge_section(cfg, :solver, dict_to_namedtuple(solver_updates))
    end
    if opts.Nz !== nothing && get_nested(cfg, (:shocks,), nothing) !== nothing
        cfg = merge_section(cfg, :shocks, (; Nz = opts.Nz))
    end

    cfg_dict = namedtuple_to_dict(cfg)
    model = ThesisProject.build_model(cfg_dict)
    method = ThesisProject.build_method(cfg_dict)
    sol = ThesisProject.solve(model, method, cfg_dict)
    return sol, cfg_dict
end

function write_det(sol, cfg, outdir)
    # Extracts
    pol_c = sol.policy[:c]
    pol_a = sol.policy[:a]
    V = sol.value # Vector (length Na)

    agrid = pol_c.grid
    c = pol_c.value
    anext = pol_a.value
    ee = pol_c.euler_errors

    meta = sol.metadata
    diag = sol.diagnostics

    header = [
        "run_id",
        "case",
        "method",
        "iters",
        "converged",
        "max_resid",
        "tol",
        "tol_pol",
        "interp_kind",
        "julia_version",
        "timestamp",
        "i",
        "a",
        "c",
        "a_next",
        "V",
        "ee",
    ]
    ts = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")

    rows = (
        (
            diag.model_id,
            "deterministic",
            diag.method,
            meta[:iters],
            meta[:converged],
            meta[:max_resid],
            meta[:tol],
            meta[:tol_pol],
            meta[:interp_kind],
            meta[:julia_version],
            ts,
            i,
            agrid[i],
            c[i],
            anext[i],
            V[i],
            ee[i],
        ) for i in eachindex(agrid)
    )

    outpath = joinpath(outdir, "egm_baseline_det.csv")
    open_write(outpath, header, rows)
    return outpath
end


function write_stoch(sol, cfg, outdir)
    # Extracts
    pol_c = sol.policy[:c]
    pol_a = sol.policy[:a]
    V = sol.value # Matrix Na x Nz

    agrid = pol_c.grid
    cmat = pol_c.value
    amat = pol_a.value
    eem = pol_c.euler_errors_mat

    # Shocks
    S = ThesisProject.get_shocks(sol.model)
    zgrid = S.zgrid

    meta = sol.metadata
    diag = sol.diagnostics

    header = [
        "run_id",
        "case",
        "method",
        "iters",
        "converged",
        "max_resid",
        "tol",
        "tol_pol",
        "interp_kind",
        "julia_version",
        "timestamp",
        "i",
        "j",
        "z",
        "a",
        "c",
        "a_next",
        "V",
        "ee",
    ]
    ts = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")

    Na, Nz = size(cmat)
    rows = (
        (
            diag.model_id,
            "stochastic",
            diag.method,
            meta[:iters],
            meta[:converged],
            meta[:max_resid],
            meta[:tol],
            meta[:tol_pol],
            meta[:interp_kind],
            meta[:julia_version],
            ts,
            i,
            j,
            zgrid[j],
            agrid[i],
            cmat[i, j],
            amat[i, j],
            V[i, j],
            (eem === nothing ? missing : eem[i, j]),
        ) for j = 1:Nz for i = 1:Na
    )

    outpath = joinpath(outdir, "egm_baseline_stoch.csv")
    open_write(outpath, header, rows)
    return outpath
end


function save_plots(sol, case::AbstractString, plotdir::AbstractString)
    saved = String[]
    # Policy plot (overlay c and a if present)
    try
        plt_pol = ThesisProject.plot_policy(sol; vars = [:c, :a])
        fpol = joinpath(plotdir, "policy_$(case).png")
        savefig(plt_pol, fpol)
        push!(saved, fpol)
    catch err
        @warn "Failed to save policy plot" case err
    end
    # Euler error plot
    try
        plt_ee = ThesisProject.plot_euler_errors(sol; by = :auto)
        fee = joinpath(plotdir, "euler_errors_$(case).png")
        savefig(plt_ee, fee)
        push!(saved, fee)
    catch err
        @warn "Failed to save Euler error plot" case err
    end
    return saved
end


function main()
    opts = parse_cli(ARGS)
    outdir = ensure_outputs_dir()
    plotdir = ensure_plots_dir(outdir)

    det_cfg = joinpath(ROOT, "config", "smoke_config", "smoke_config.yaml")
    stc_cfg = joinpath(ROOT, "config", "smoke_config", "smoke_config_stochastic.yaml")

    @info "Running deterministic baseline" det_cfg
    sol_det, cfg_det = run_one(det_cfg, opts)
    det_csv = write_det(sol_det, cfg_det, outdir)
    det_plots = save_plots(sol_det, "det", plotdir)

    @info "Running stochastic baseline" stc_cfg
    sol_stc, cfg_stc = run_one(stc_cfg, opts)
    stc_csv = write_stoch(sol_stc, cfg_stc, outdir)
    stc_plots = save_plots(sol_stc, "stoch", plotdir)

    println("Written CSVs:\n  " * det_csv * "\n  " * stc_csv)
    if isdefined(GenerateBaselineCSV, :Plots)
        println("Written plots:")
        for p in vcat(det_plots, stc_plots)
            println("  " * p)
        end
    else
        println("Plots not generated (Plots.jl not available)")
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    GenerateBaselineCSV.main()
end
