using Test
ENV["GKSwstype"] = "100"        # headless GR on CI
using ThesisProject       # Plots triggers ThesisProjectPlotsExt

include("utils.jl")

const SMOKE_CFG_PATH =
    joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_config.yaml")
const SMOKE_CFG = load_config(SMOKE_CFG_PATH)
const SMOKE_STOCH_CFG_PATH =
    joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_config_stochastic.yaml")
const SMOKE_STOCH_CFG = load_config(SMOKE_STOCH_CFG_PATH)

@testset "Deterministic config" begin
    include("test_core.jl")
    include("test_accuracy.jl")
    include("test_viz.jl")
    include("test_solver_options.jl")
end

@testset "Stochastic config" begin
    include("test_egm_stoch.jl")
    include("test_sim.jl")
    include("test_stability_extreme.jl")
end

@testset "Config-agnostic and mixed" begin
    include("test_residuals.jl")
    include("test_interp.jl")
    include("test_determinism.jl")
    include("test_shocks.jl")
    include("test_chebyshev.jl")
    include("test_projection.jl")
    include("test_quality.jl")
    include("test_value_function.jl")
    include("test_projection_orders.jl")
    include("test_perturbation_method.jl")
end

#include("test_projection_stoch.jl")
#include("test_projection_method.jl")
