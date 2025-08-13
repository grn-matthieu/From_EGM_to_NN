#!/usr/bin/env julia
using Pkg
Pkg.activate(".")

# Load your package
include(joinpath(@__DIR__, "..", "src", "ThesisProject.jl"))
using .ThesisProject

# Stdlib / packages (TOP-LEVEL to avoid world-age issues)
using Statistics, Dates, UUIDs
using CSV, DataFrames   # <-- move here

const Na    = 7
const PARAM = SimpleParams(β=0.96, σ=2.0, r=0.02, y=1.0, a_min=0.0, a_max=20.0)

function mk_run_dir()
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    dir = joinpath(@__DIR__, "..", "runs", "$(stamp)_" * string(UUIDs.uuid4()))
    isdir(dir) || mkpath(dir)
    return dir
end

function main()
    logdir = mk_run_dir()
    try
        agrid = collect(range(PARAM.a_min, PARAM.a_max, length=Na))
        sol = solve_simple_egm(PARAM, agrid; tol=1e-8, maxit=3000, verbose=false)

        resid = euler_residuals_simple(PARAM, sol.agrid, sol.c)
        println("\n── Smoke simple (Na=$(Na)) ──")
        println("Converged = ", sol.converged, " | iters = ", sol.iters)
        println("Residuals: max = ", maximum(resid), " | median = ", median(resid))
        println("Logdir: ", logdir)

        # Build DataFrame (now safe because DataFrames is loaded at top-level)
        df = DataFrame(; a = sol.agrid, c = sol.c, a_next = sol.a_next, residual = resid)
        CSV.write(joinpath(logdir, "simple_egm_results.csv"), df)

        open(joinpath(logdir, "metadata.txt"), "w") do io
            println(io, "Na=$(Na)")
            println(io, "params=", PARAM)
            println(io, "timestamp=", Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))
        end

        return 0
    catch e
        open(joinpath(logdir, "error.txt"), "w") do io; showerror(io, e); end
        open(joinpath(logdir, "stacktrace.txt"), "w") do io; showerror(io, e, catch_backtrace()); end
        println("✖ Smoke failed. See files in: ", logdir)
        return 1
    end
end

exit(main())