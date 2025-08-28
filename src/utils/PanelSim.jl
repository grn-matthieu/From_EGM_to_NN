"""
Panel simulation utilities for solvers.
"""
module PanelSim
export simulate_panel, simulate_panel_agent

using ..Shocks: discretize, simulate_shocks, ShockOutput
using ..EGMSolver: solve_stochastic_egm
using ..SimpleCalibration: default_simple_params
using ProgressMeter


function simulate_panel_agent(sol, shocks::Vector{Float64}, cfg::Dict{Any, Any})
    """
    Simulate a single agent's asset and consumption path given a stochastic policy (matrix)
    and a sequence of shocks (as indices into zgrid).
    Returns:
        - assets: Vector of asset holdings over time
        - consumption: Vector of consumption over time
    """
    T = cfg["simulation"]["T"]
    Na = cfg["simulation"]["N"]
    Nz = cfg["shocks"]["Nz"]

    a_grid = sol.agrid
    z_grid = sol.zgrid

    assets = zeros(T)
    consumption = zeros(T)

    # Initial asset
    assets[1] = cfg["simulation"]["a_0"]

    for t in 1:T
        # Find nearest asset grid index
        a_idx = searchsortedfirst(a_grid, assets[t])
        a_idx = clamp(a_idx, 1, Na)

        # Find nearest shock grid index
        z_idx = searchsortedfirst(z_grid, shocks[t])
        z_idx = clamp(z_idx, 1, Nz)

        c_t = sol.c[a_idx, z_idx]
        consumption[t] = c_t

        # Update assets for next period (if not last period)
        if t < T
            r = cfg["model"]["r"]
            y = exp(shocks[t+1])
            assets[t+1] = (assets[t] - c_t) * r + y
        end
    end

    return assets, consumption
end

function simulate_panel(cfg::Dict{Any, Any}; solver=nothing)
    """
    Simulate a panel of representative agents with a given solver.
    Returns :
        - a time series of assets over time
        - a time series of consumption over time
        - summary statistics including seed, burn-in and acceptance flags
    """
    # --- Initialization -------
    N, T = cfg["simulation"]["N"], cfg["simulation"]["T"]

    assets = Matrix{Float64}(undef, N, T); consumption = Matrix{Float64}(undef, N, T);
    stats = Dict{Symbol, Any}()

    # Simulate an identical draw of shocks for every agent in the panel
    method = cfg["shocks"]["method"]
    ρ_shock = cfg["shocks"]["ρ_shock"]
    σ_shock = cfg["shocks"]["σ_shock"]
    Nz = cfg["shocks"]["Nz"]
    shocks_cfg = discretize(method, ρ_shock, σ_shock, Nz; m=3, validate=true)

    panel_shocks = Matrix{Float64}(undef, N, T) 
    for n in 1:N
        panel_shocks[n, :] = simulate_shocks(T, shocks_cfg, cfg["simulation"]["seed"] + n) # Update the seed at each iteration for the shocks to be independent
    end
    @info "Shocks simulated."

    # Compute the optimal asset/consumption policy for the specified solver
    p = default_simple_params()
    a_grid = collect(range(cfg["grid"]["a_min"], stop=cfg["grid"]["a_max"], length=cfg["grid"]["Na"]))
    z_grid = shocks_cfg.zgrid
    sol = solve_stochastic_egm(p, a_grid, z_grid, shocks_cfg.Π)

    # --- Simulation ---------
    @showprogress 1 "Simulating panel agents" for i in 1:N
        assets_i, consumption_i = simulate_panel_agent(sol, panel_shocks[i, :], cfg)
        assets[i, :] = assets_i
        consumption[i, :] = consumption_i
    end
    @info "Panel simulation completed."

    # Compute summary statistics
    #stats[:seed] = cfg[:seed]
    #stats[:burn_in] = cfg[:burn_in]
    #stats[:acceptance_flags] = cfg[:acceptance_flags]

    return (assets, consumption, panel_shocks, stats)
end




end