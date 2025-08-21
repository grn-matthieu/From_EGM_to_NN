#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "ThesisProject.jl"))
using .ThesisProject
using .ThesisProject: UtilsConfig, EGMSolver, PlotsUtils
using CSV, DataFrames

 
# --- Config / params
cfg = UtilsConfig.load_config(joinpath(@__DIR__,"..","config","simple_baseline.yaml"))
p = default_simple_params()
Na = cfg["grid"]["Na"]
agrid = collect(range(cfg["grid"]["a_min"], stop=cfg["grid"]["a_max"], length=Na))
runid = get(cfg, "runid", "det_egm")

# --- Solver
sol = EGMSolver.solve_simple_egm(p, agrid; tol=1e-10, maxit=5000, verbose=true, relax = 0.3)
resid = euler_residuals_simple(p, sol.agrid, sol.c)

# --- Plots
PlotsUtils.plot_policy(sol; runid=runid)
pltR, _ = PlotsUtils.plot_residuals(sol, p; runid=runid)

# --- CSV
df = DataFrame(a=sol.agrid, c=sol.c, a_next=sol.a_next, residual=vec(resid))
CSV.write(joinpath("runs",runid,"deterministic_egm.csv"), df)
println("Deterministic run saved under runs/$runid")
