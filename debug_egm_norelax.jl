using ThesisProject
using Statistics

# Load config
cfg_path = joinpath(@__DIR__, "config", "smoke_cfg_stoch.yaml")
cfg_dict = load_config(cfg_path)

# Override settings for detailed debugging
cfg_dict = merge(
    cfg_dict,
    (
        grids = merge(cfg_dict.grids, (Na = 20,)),
        solver = merge(
            cfg_dict.solver,
            (
                method = :EGM,
                verbose = true,
                maxit = 100,
                relax = 1.0,  # NO RELAXATION
            ),
        ),
    ),
)

println("="^70)
println("EGM Debug Test - No Relaxation")
println("="^70)
println()
println("Configuration:")
println("  β = $(cfg_dict.params.β)")
println("  γ = $(cfg_dict.params.γ)")
println("  r = $(cfg_dict.params.r)")
println("  Na = $(cfg_dict.grids.Na)")
println("  relax = $(cfg_dict.solver.relax)")
println()

# Solve
result = solve(cfg_dict)

println()
println("="^70)
println("Results")
println("="^70)
sol = result
println("Converged: ", get(sol.diagnostics, :converged, false))
println("Iterations: ", get(sol.diagnostics, :iter, "N/A"))
println("Euler RMSE: ", get(sol.metadata, :max_resid, "N/A"))
println("Runtime: ", get(sol.diagnostics, :runtime, "N/A"), " s")
