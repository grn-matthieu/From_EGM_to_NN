#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "ThesisProject.jl"))
using .ThesisProject
using CSV, DataFrames

cfg = load_config(joinpath(@__DIR__,"..", "config","simple_stochastic.yaml"))
p = default_simple_params()

Na = cfg["grid"]["Na"]
a_max = cfg["grid"]["a_max"]
a_min = cfg["grid"]["a_min"]
agrid = collect(range(a_min, stop=a_max, length=Na))
scfg = cfg["shocks"]
ρ_shock, σ_shock = scfg["ρ_shock"], scfg["σ_shock"]
Nz = scfg["Nz"]
method = get(scfg, "method", "rouwenhorst")
m = get(scfg, "m", 3.0)
runid = get(cfg, "runid", "stochastic_egm")

# --- Discretize income AR(1)
shock = discretize(method, ρ_shock, σ_shock, Nz; m=m, validate=true)

# --- Solve
sol = solve_stochastic_egm(p, agrid, shock.zgrid, shock.Π;
    tol=1e-10, maxit=8000, verbose=true, relax=0.3, ν=1e-12, patience=200)

resid = euler_residuals_stochastic(p, sol.agrid, shock.zgrid, shock.Π, sol.c)

# --- Plots (policy surfaces + residual heatmaps)
plot_policy(sol; runid=runid)
plot_residuals(sol, p, shock.Π; runid=runid)
plot_residuals(sol, p, shock.Π; runid=runid, log10scale=false)

# --- Save CSV (panel)
df = DataFrame(
    a = repeat(sol.agrid, Nz),
    z = vcat([fill(z, Na) for z in sol.zgrid]...),
    c = vec(sol.c),
    a_next = vec(sol.a_next),
    residual = vec(resid)
)
CSV.write(joinpath("runs",runid,"stochastic_egm.csv"), df)
println("Stochastic run saved under runs/$runid")
