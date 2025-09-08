#!/usr/bin/env julia

"""
This script generates CSV snapshots for baseline EGM solutions (deterministic and stochastic),
including Euler errors, to help track changes over time.
The model concerned is the simple consumer-saving problem.

Outputs two files in `outputs/`:
- egm_baseline_det.csv
- egm_baseline_stoch.csv

Run with:
  julia --project=. scripts/generate_baseline_csv.jl
"""

module GenerateBaselineCSV

import Pkg
using Dates
using Printf

const ROOT = normpath(joinpath(@__DIR__, ".."))

"""
Activate the project immediately so downstream `using ThesisProject` works.
"""
Pkg.activate(ROOT; io=devnull)
using ThesisProject

function ensure_outputs_dir()
    outdir = joinpath(ROOT, "outputs")
    isdir(outdir) || mkpath(outdir)
    return outdir
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

# Small fun to auto run the config
function run_one(cfg_path::AbstractString)
    cfg = ThesisProject.load_config(cfg_path)
    ThesisProject.validate_config(cfg)

    model  = ThesisProject.build_model(cfg)
    method = ThesisProject.build_method(cfg)
    sol    = ThesisProject.solve(model, method, cfg)
    return sol, cfg
end

function write_det(sol, cfg, outdir)
    # Extracts
    pol_c = sol.policy[:c]
    pol_a = sol.policy[:a]
    V     = sol.value # Vector (length Na)

    agrid = pol_c.grid
    c     = pol_c.value
    anext = pol_a.value
    ee    = pol_c.euler_errors

    meta  = sol.metadata
    diag  = sol.diagnostics

    header = [
        "run_id","case","method","iters","converged","max_resid","tol","tol_pol","interp_kind","julia_version","timestamp",
        "i","a","c","a_next","V","ee"
    ]
    ts = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")

    rows = (
        (
            diag.model_id, "deterministic", diag.method, meta[:iters], meta[:converged], meta[:max_resid], meta[:tol], meta[:tol_pol], meta[:interp_kind], meta[:julia_version], ts,
            i, agrid[i], c[i], anext[i], V[i], ee[i]
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
    V     = sol.value # Matrix Na x Nz

    agrid = pol_c.grid
    cmat  = pol_c.value
    amat  = pol_a.value
    eem   = pol_c.euler_errors_mat

    # Shocks
    S = ThesisProject.get_shocks(sol.model)
    zgrid = S.zgrid

    meta  = sol.metadata
    diag  = sol.diagnostics

    header = [
        "run_id","case","method","iters","converged","max_resid","tol","tol_pol","interp_kind","julia_version","timestamp",
        "i","j","z","a","c","a_next","V","ee"
    ]
    ts = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")

    Na, Nz = size(cmat)
    rows = (
        (
            diag.model_id, "stochastic", diag.method, meta[:iters], meta[:converged], meta[:max_resid], meta[:tol], meta[:tol_pol], meta[:interp_kind], meta[:julia_version], ts,
            i, j, zgrid[j], agrid[i], cmat[i,j], amat[i,j], V[i,j], (eem === nothing ? missing : eem[i,j])
        ) for j in 1:Nz for i in 1:Na
    )

    outpath = joinpath(outdir, "egm_baseline_stoch.csv")
    open_write(outpath, header, rows)
    return outpath
end


function main()
    outdir = ensure_outputs_dir()

    det_cfg = joinpath(ROOT, "config", "simple_baseline.yaml")
    stc_cfg = joinpath(ROOT, "config", "smoke_config", "smoke_config_stochastic.yaml")

    @info "Running deterministic baseline" det_cfg
    sol_det, cfg_det = run_one(det_cfg)
    det_csv = write_det(sol_det, cfg_det, outdir)

    @info "Running stochastic baseline" stc_cfg
    sol_stc, cfg_stc = run_one(stc_cfg)
    stc_csv = write_stoch(sol_stc, cfg_stc, outdir)

    println("Written:\n  " * det_csv * "\n  " * stc_csv)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    GenerateBaselineCSV.main()
end
