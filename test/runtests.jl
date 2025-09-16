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

# Reduce noise from warnings in CI output
# Suppress method overwrite warnings if supported (Julia >= 1.9)
try
    Base.Experimental.set_warn_overwrite(false)
catch
end

# --- Deterministic breakdown ---
@testset "Deterministic - Core" begin
    include("test_core.jl")
end
@testset "Deterministic - Accuracy" begin
    include("test_accuracy.jl")
end
@testset "Deterministic - Viz" begin
    include("test_viz.jl")
end
@testset "Deterministic - Solver Options" begin
    include("test_solver_options.jl")
end

# --- Stochastic breakdown ---
@testset "Stochastic - EGM Smoke" begin
    include("test_egm_stoch.jl")
end
@testset "Stochastic - Simulation" begin
    include("test_sim.jl")
end
@testset "Stochastic - Stability Extreme" begin
    include("test_stability_extreme.jl")
end

# --- Projection grouping ---
@testset "Projection - Stochastic Smoke" begin
    include("test_projection_stoch.jl")
end
@testset "Projection - Method Adapter" begin
    include("test_projection_method.jl")
end
@testset "Projection - Accuracy" begin
    include("test_projection_accuracy.jl")
end
@testset "Projection - Orders" begin
    include("test_projection_orders.jl")
end
@testset "Projection - Deterministic" begin
    include("test_projection.jl")
end

# --- Kernels ---
@testset "Kernels - EGM" begin
    include("test_egm_kernel.jl")
end
@testset "Kernels - Perturbation" begin
    include("test_perturbation_kernel.jl")
end

# --- Core/shared utilities ---
@testset "Core - Residuals" begin
    include("test_residuals.jl")
end
@testset "Core - Interp" begin
    include("test_interp.jl")
end
@testset "Core - Determinism" begin
    include("test_determinism.jl")
end
@testset "Core - Shocks" begin
    include("test_shocks.jl")
end
@testset "Core - Chebyshev" begin
    include("test_chebyshev.jl")
end
@testset "Core - Value Function" begin
    include("test_value_function.jl")
end
@testset "Core - Quality" begin
    include("test_quality.jl")
end
@testset "Core - API" begin
    include("test_core_api.jl")
end
@testset "Core - Config Validator" begin
    include("test_config_validator.jl")
end

# --- NN ---
@testset "NN - Init" begin
    include("test_nn_init.jl")
end
@testset "NN - Data" begin
    include("loss_test.jl")
end
