using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

include(joinpath(@__DIR__, "..", "..", "src", "ThesisProject.jl"))

using .ThesisProject
using DataFrames
using CSV


cfg = load_config(joinpath(@__DIR__,"..", "..","config","simulation_baseline.yaml"))
@info "Configuration loaded."

master_seed = cfg["experiment"]["seed"]
master_rng = set_global_seed(master_seed)

assets, consumption, panel_shocks, rng_list, seed_list = simulate_panel(cfg, master_rng)

runid = "test_simulation"
outdir = joinpath("runs", runid)
isdir(outdir) || mkpath(outdir)


N, T = size(assets)
df = DataFrame(
    agent = repeat(1:N, inner=T),
    time = repeat(1:T, outer=N),
    asset = vec(assets),
    consumption = vec(consumption),
    shock = vec(panel_shocks),
)

CSV.write(joinpath("runs", runid, "test_run_results.csv"), df)
CSV.write(joinpath("runs", runid, "agent_seeds.csv"), DataFrame(agent_id = 1:N, agent_seed = seed_list))
@info "Results saved under runs/$runid, seeds logged under runs/$runid/agent_seeds.csv"

plot_simulation_paths(consumption; runid)
@info "Simulation plots saved under runs/$runid"