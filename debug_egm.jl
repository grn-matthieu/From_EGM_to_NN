using ThesisProject
using Statistics

# Load config
cfg_path = joinpath(@__DIR__, "config", "smoke_cfg_stoch.yaml")
cfg_dict = load_config(cfg_path)

# Override settings for detailed debugging
cfg_dict = merge(
    cfg_dict,
    (
        grids = merge(cfg_dict.grids, (Na = 20,)),  # Small grid for debugging
        solver = merge(cfg_dict.solver, (method = :EGM, verbose = true, maxit = 50)),
    ),
)

println("="^70)
println("EGM Debug Test")
println("="^70)
println()
println("Configuration:")
println("  β = $(cfg_dict.params.β)")
println("  γ = $(cfg_dict.params.γ)")
println("  r = $(cfg_dict.params.r)")
println("  Na = $(cfg_dict.grids.Na)")
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

# Check the policy
if haskey(sol.policy, :c) && haskey(sol.policy, :a_next)
    c = sol.policy[:c]
    a_next = sol.policy[:a_next]
    println()
    println("Policy shape: c=$(size(c)), a_next=$(size(a_next))")
    println(
        "Consumption: mean=$(round(mean(c), digits=4)), min=$(round(minimum(c), digits=4)), max=$(round(maximum(c), digits=4))",
    )
    println(
        "Savings: mean=$(round(mean(a_next), digits=4)), min=$(round(minimum(a_next), digits=4)), max=$(round(maximum(a_next), digits=4))",
    )

    # Print a few policy values
    println()
    println("Sample policies (state j=3, first 5 asset levels):")
    j = 3
    for i = 1:min(5, size(c, 1))
        println(
            "  a=$(round(sol.model.grids[:a].grid[i], digits=3)): c=$(round(c[i,j], digits=4)), a'=$(round(a_next[i,j], digits=4))",
        )
    end
end
