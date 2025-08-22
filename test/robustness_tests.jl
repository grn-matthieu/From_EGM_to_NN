#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "ThesisProject.jl"))
using .ThesisProject
using .ThesisProject: UtilsConfig, EGMSolver, PlotsUtils
using CSV, DataFrames
using Plots
using ProgressMeter


# Warning : this code ouptput is not available in the repo
# Because the number of printed runs is too large
# Please execute `julia test/robustness_tests.jl` to see the output


# --- Config / params
cfg = UtilsConfig.load_config(joinpath(@__DIR__,"..","config","simple_baseline.yaml"))
p_base = default_simple_params()
Na = cfg["grid"]["Na"]
agrid = collect(range(cfg["grid"]["a_min"], stop=cfg["grid"]["a_max"], length=Na))

# --- Parameter sweep ranges
# We only iterate over β and σ because they are the main drivers of the simple model's behavior
betas = 0.90:0.01:0.99
sigmas = 1.0:0.5:5.0

results = DataFrame(β=Float64[], σ=Float64[], converged=Bool[], max_resid=Float64[], iters=Int[])

outdir = joinpath("runs", "robustness_sweep")

isdir(outdir) || mkpath(outdir)
total = length(betas) * length(sigmas)
pbar = Progress(total, desc="Robustness sweep")

for β in betas, σ in sigmas
    next!(pbar)
    # Create a new SimpleParams instance with updated β and σ
    p = SimpleParams(β, σ, p_base.r, p_base.y, p_base.ρ, p_base.σ_stoch)
    runid = "robustness_sweep/robustness_β$(round(β, digits=3))_σ$(round(σ, digits=2))"
    try
        sol = EGMSolver.solve_simple_egm(p, agrid; tol=1e-8, maxit=5000, verbose=false, relax = 0.3)
        resid = euler_residuals_simple(p, sol.agrid, sol.c)
        push!(results, (β, σ, sol.converged, sol.max_residual, sol.iters))

        # Save policy plot
        plt1, plt2 = PlotsUtils.plot_policy(sol, p; runid=runid, verbose=false)
        savefig(plt1, joinpath(outdir, "policy_β$(round(β, digits=3))_σ$(round(σ, digits=2)).png"))
        savefig(plt2, joinpath(outdir, "assets_β$(round(β, digits=3))_σ$(round(σ, digits=2)).png"))

        # Save residuals plot
        pltR, _ = PlotsUtils.plot_residuals(sol, p; runid=runid, verbose=false)
        savefig(pltR, joinpath(outdir, "residuals_β$(round(β, digits=3))_σ$(round(σ, digits=2)).png"))
    catch err
        msg = sprint(showerror, err)
        @warn "Failure at β=$β, σ=$σ: $msg"
        push!(results, (β, σ, false, NaN, 0))
    end
end

CSV.write(joinpath(outdir, "robustness_results.csv"), results)

# Optional: Heatmap of max_resid over (β, σ)
pivot = unstack(results, :σ, :β, :max_resid)
heatmap(
    betas, sigmas, Matrix(pivot[:, Not(:σ)]),
    xlabel="β", ylabel="σ", title="Max Euler Residuals (log10)",
    colorbar_title="log10(resid)", zscale=:log10
)
savefig(joinpath(outdir, "heatmap_max_resid.png"))

println("Robustness sweep complete. Results and figures saved in $outdir")