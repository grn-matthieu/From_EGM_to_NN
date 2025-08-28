using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "..", "src", "ThesisProject.jl"))

using .ThesisProject
using DataFrames
using CSV


cfg = load_config(joinpath(@__DIR__,"..", "..","config","simulation_baseline.yaml"))
@info "Configuration loaded."

assets, consumption, panel_shocks, stats = simulate_panel(cfg)

runid = "test_simulation"
outdir = joinpath("runs", runid)
isdir(outdir) || mkpath(outdir)


N, T = size(assets)
df = DataFrame(
    agent = repeat(1:N, inner=T),
    time = repeat(1:T, outer=N),
    asset = vec(assets),
    consumption = vec(consumption),
    shock = vec(panel_shocks)
)

CSV.write(joinpath("runs", runid, "test_run_results.csv"), df)
@info "Results saved under runs/$runid"

plot_simulation_paths(consumption; runid)
@info "Simulation plots saved under runs/$runid"