#!/usr/bin/env julia
using Pkg
Pkg.activate(".")

# Load our package
include("src/ThesisProject.jl")
using .ThesisProject
using Statistics

# Load config
include("src/utils/Config.jl")
cfg = ThesisProject.UtilsConfig.load_config("config/simple_baseline.yaml")

# Prepare parameters and grid
p = default_simple_params()
agrid = range(p.a_min, p.a_max, length=cfg["grid"]["Na"])

# Run solver
sol = solve_simple_egm(p, collect(agrid);
    tol = cfg["solver"]["tol"],
    maxit = cfg["solver"]["maxit"],
    verbose = cfg["solver"]["verbose"]
)

println("\nConverged: ", sol.converged, " | Iterations: ", sol.iters)

# Compute Euler residuals
resid = euler_residuals_simple(p, sol.agrid, sol.c)
println("Max residual: ", maximum(resid))
println("Median residual: ", median(resid))

# Save results to CSV
using CSV, DataFrames
using Dates
using Plots

logdir = joinpath("runs", Dates.format(now(), "yyyymmdd_HHMMSS"))
mkpath(logdir)
# Plot max euler residual against Iterations
plot(sol.agrid, 
    resid,
    yaxis = :log,
    label="Max Euler Residual", 
    xlabel="Iterations", 
    ylabel="Residual", 
    title="Max Euler Residual vs Iterations")

savefig(joinpath(logdir, "max_euler_residual.png"))

df = DataFrame(a = sol.agrid, c = sol.c, a_next = sol.a_next, residual = resid)
CSV.write(joinpath(logdir, "simple_egm_results.csv"), df)

println("Results saved to: ", logdir)
