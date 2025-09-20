__precompile__()

"""
ThesisProject

Top-level package module. Assembles core APIs, utilities, models, solvers, and
method adapters into a cohesive interface. See `src/core/api.jl` for the public
types and entry points, and `methods/*` for adapters that call solver kernels.
"""
module ThesisProject

using PrecompileTools

# --- Includes ---
# 1) spine
include("core/api.jl")
include("core/model_contract.jl")

# 2) utilities
include("utils/Config.jl")
# include("utils/Schema.jl")
include("utils/Determinism.jl")

# 3) shared + models + model builder
include("models/shared/Shocks.jl")
include("models/baseline/ConsumerSaving.jl")
include("core/model_factory.jl")

# 4) solvers (pure kernels)
# common solver utilities
include("solvers/common/interp.jl")
include("solvers/common/value_fun.jl")
include("solvers/common/chebyshev.jl")
include("solvers/common/residuals.jl")
include("solvers/common/validators.jl")

# projection solver
include("solvers/projection/coefficients.jl")
include("solvers/projection/kernel.jl")

# egm specific
include("solvers/egm/kernel.jl")
include("solvers/perturbation/kernel.jl")

# NN solver removed - includes deleted to rebuild from scratch

# 5) methods (adapters)
include("methods/EGM.jl")
include("methods/Projection.jl")
include("methods/Perturbation.jl")
# NN method adapter removed

# 6) method factory
include("core/method_factory.jl")

# 7) simulation
include("sim/panel.jl")
using .SimPanel: simulate_panel

# 8) analysis
include("analysis/SteadyState.jl")
using .SteadyState: steady_state_analytic, steady_state_from_policy

# NN evaluation / pretrain helpers removed
using .API:
    AbstractModel,
    AbstractMethod,
    Solution,
    get_params,
    get_grids,
    get_shocks,
    get_utility,
    build_model,
    load_config,
    validate_config,
    build_method,
    solve

# --- Exports ---

export load_config,
    validate_config,
    build_model,
    build_method,
    solve,
    get_params,
    get_grids,
    get_shocks,
    get_utility,
    # NN exports removed - will reintroduce when rebuilt
    simulate_panel,
    steady_state_analytic,
    steady_state_from_policy,
    residuals

# --- Extensions ---
include("viz/api.jl")      # visualization API stubs; enabled by Plots extension
export plot_policy, plot_euler_errors

@setup_workload begin
    cfg_path = joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_config.yaml")
    if isfile(cfg_path)
        @compile_workload begin
            cfg = load_config(cfg_path)
            validate_config(cfg)
            build_model(cfg)
            build_method(cfg)
        end
    end
end

end # module
