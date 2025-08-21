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


# Prepare parameters and grids
p = default_simple_params()
agrid = range(p.a_min, p.a_max, length=cfg["grid"]["Na"])

# Check whether we are running stochastic or deterministic
if haskey(cfg, "shocks") && cfg["shocks"]["active"] == true
    # Stochastic permanent-income case
    ρ_shock = cfg["shocks"]["ρ_shock"]
    σ_shock = cfg["shocks"]["σ_shock"]
    Nz = cfg["shocks"]["Nz"]
    method = get(cfg["shocks"], "method", "rouwenhorst")
    m = get(cfg["shocks"], "m", 3.0)

    # Discretize AR(1) for log income
    zgrid, Pz = discretize_ar1(method, ρ_shock, σ_shock, Nz; m=m)

    # Solve stochastic EGM
    sol = solve_stochastic_egm(p, agrid, zgrid, Pz; tol=1e-8, maxit=1000, verbose=true)

    # Compute residuals
    resid = euler_residuals_stochastic(p, sol.agrid, zgrid, Pz, sol.c)

else
    # Deterministic case (baseline savings model)
    sol = solve_simple_egm(p, agrid; tol=1e-8, maxit=1000, verbose=true)

    # Compute residuals
    resid = euler_residuals_simple(p, sol.agrid, sol.c)
end


# Save results to CSV
using CSV, DataFrames
using Dates
using Plots

logdir = joinpath("runs", Dates.format(now(), "yyyymmdd_HHMMSS"))
mkpath(logdir)

if haskey(cfg, "shocks") && cfg["shocks"]["active"] == true
    df = DataFrame(a = repeat(sol.agrid, length(zgrid)),
                   z = vcat([fill(z, length(sol.agrid)) for z in zgrid]...),
                   c = vec(sol.c),
                   a_next = vec(sol.a_next),
                   residual = vec(resid))
    CSV.write(joinpath(logdir, "stochastic_egm_results.csv"), df)
else
    df = DataFrame(a = sol.agrid,
                   c = sol.c,
                   a_next = sol.a_next,
                   residual = resid)
    CSV.write(joinpath(logdir, "simple_egm_results.csv"), df)
end


runid = "stoch_egm"

# Policy surfaces (stochastic)
plot_policy(sol; runid=runid)

# Value function only for deterministic solutions
# plot_value(sol, p; runid=runid)

# Residuals (stochastic requires Pz)
plot_residuals(sol, p, Pz; runid=runid)
plot_residuals(sol, p, Pz; runid=runid, log10scale=false)


println("Results saved to: ", logdir)
