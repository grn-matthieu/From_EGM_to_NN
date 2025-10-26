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
include("utils/Determinism.jl")
include("core/api.jl")
include("core/model_contract.jl")

# 2) grids
include("solvers/common/interp.jl")
include("grids/grid_types.jl")
include("grids/factory.jl")
include("grids/helpers.jl")
module Grids
include("solvers/common/interp.jl")
include("grids/grid_types.jl")
include("grids/factory.jl")
include("grids/helpers.jl")
export nodes,
    build_grid_backend, fit_interpolant!, evaluate_scalar!, refine_once!, CommonInterp
end
using .CommonInterp

# 3) utilities
include("utils/Config.jl")
# include("utils/Schema.jl")
include("utils/Diagnostics.jl")

# 4) shared + models + model builder
include("models/shared/Shocks.jl")
include("models/baseline/ConsumerSaving.jl")
include("models/baseline/ConsumerSavingVAR.jl")
include("core/model_factory.jl")

# 5) solvers (pure kernels)
# common solver utilities
if !isdefined(@__MODULE__, :CommonInterp)
    include("solvers/common/interp.jl")
end
if !isdefined(@__MODULE__, :GridHelpers)
    include("grids/helpers.jl")
end
include("solvers/common/policy_utils.jl")
include("solvers/common/value_fun.jl")
include("solvers/common/chebyshev.jl")
include("solvers/common/residuals.jl")
include("solvers/common/placeholders.jl")
include("solvers/common/csvar_utils.jl")
include("solvers/common/integration.jl")
include("solvers/common/validators.jl")

# projection solver
include("solvers/projection/coefficients.jl")
include("solvers/projection/kernel.jl")

# egm specific
include("solvers/egm/kernel.jl")
include("solvers/perturbation/kernel.jl")
include("solvers/time_iteration/kernel.jl")

# NN solver
include("solvers/nn/data_nn.jl")
include("solvers/nn/kernel.jl")

# 6) methods (adapters)
include("methods/common/utils.jl")
include("methods/EGM.jl")
include("methods/Projection.jl")
include("methods/Perturbation.jl")
include("methods/NN.jl")
include("methods/TimeIteration.jl")

# 7) method factory
include("core/method_factory.jl")

# 8) simulation
include("sim/panel.jl")
using .SimPanel: simulate_panel

# 9) analysis
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
    simulate_panel,
    steady_state_analytic,
    steady_state_from_policy,
    residuals

# --- Extensions ---
include("viz/api.jl")      # visualization API stubs; enabled by Plots extension
export plot_policy, plot_euler_errors

@setup_workload begin
    cfg_path = joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_cfg_det.yaml")
    if isfile(cfg_path)
        @compile_workload begin
            cfg = load_config(cfg_path)
            build_model(cfg)
            build_method(cfg)
        end
    end
end

end # module
