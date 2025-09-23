include("bootstrap.jl")

# Reduce noise from warnings in CI output
# Suppress method overwrite warnings if supported (Julia >= 1.9)
try
    Base.Experimental.set_warn_overwrite(false)
catch
end

# Fast test mode: set environment variable `FAST_TEST=1` to skip heavier/coverage testsets
const FAST_TEST = get(ENV, "FAST_TEST", "0") == "1"

# --- Deterministic breakdown ---
@testset "Deterministic - Core" begin
    include("core/test_core.jl")
end
@testset "Deterministic - Accuracy" begin
    include("core/test_accuracy.jl")
end
@testset "Deterministic - Viz" begin
    include("core/test_viz.jl")
end
@testset "Deterministic - Solver Options" begin
    include("core/test_solver_options.jl")
end

# --- Stochastic breakdown ---
@testset "Stochastic - EGM Smoke" begin
    include("stochastic/test_egm_stoch.jl")
end
@testset "Stochastic - Simulation" begin
    include("stochastic/test_sim.jl")
end
@testset "Stochastic - Stability Extreme" begin
    include("stochastic/test_stability_extreme.jl")
end

# --- Projection grouping ---
@testset "Projection - Stochastic Smoke" begin
    if !FAST_TEST
        include("methods/test_projection_stoch.jl")
    else
        @info "Skipping Projection - Stochastic Smoke in FAST_TEST mode"
    end
end
@testset "Projection - Method Adapter" begin
    include("methods/test_projection_method.jl")
end

# --- Methods - NN adapter coverage helper ---
@testset "Methods - NN Adapter (coverage)" begin
    include("methods/test_nn_adapter_coverage.jl")
end
@testset "Projection - Accuracy" begin
    include("methods/test_projection_accuracy.jl")
end
@testset "Projection - Orders" begin
    include("methods/test_projection_orders.jl")
end
@testset "Projection - Deterministic" begin
    include("methods/test_projection.jl")
end

# --- Kernels ---
@testset "Kernels - EGM" begin
    include("methods/test_egm_kernel.jl")
end
@testset "Kernels - Perturbation" begin
    include("methods/test_perturbation_kernel.jl")
end

# --- Core/shared utilities ---
@testset "Core - Residuals" begin
    include("core/test_residuals.jl")
end
@testset "Core - Interp" begin
    include("core/test_interp.jl")
end
@testset "Core - Determinism" begin
    include("core/test_determinism.jl")
end
@testset "Core - Shocks" begin
    include("core/test_shocks.jl")
end
@testset "Core - Chebyshev" begin
    include("core/test_chebyshev.jl")
end
@testset "Core - Value Function" begin
    include("core/test_value_function.jl")
end
@testset "Core - Quality" begin
    include("core/test_quality.jl")
end
@testset "Core - API" begin
    include("core/test_core_api.jl")
end
@testset "Core - Config Validator" begin
    include("core/test_config_validator.jl")
end

# NN tests removed - neural-network solver being rebuilt

@testset "Coverage - More" begin
    if !FAST_TEST
        include("coverage/coverage_more.jl")
    else
        @info "Skipping Coverage - More in FAST_TEST mode"
    end
end

@testset "Coverage - SimPlots" begin
    if !FAST_TEST
        include("coverage/coverage_simplots.jl")
    else
        @info "Skipping Coverage - SimPlots in FAST_TEST mode"
    end
end

@testset "Coverage - NNLoss" begin
    # NNLoss coverage removed
end

@testset "Coverage - Core API" begin
    if !FAST_TEST
        include("coverage/coverage_core_api.jl")
    else
        @info "Skipping Coverage - Core API in FAST_TEST mode"
    end
end

@testset "Analysis - SteadyState" begin
    include("analysis/test_steady_state.jl")
end
