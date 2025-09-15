module ThesisProject

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
include("solvers/nn/kernel.jl")
include("solvers/nn/init.jl")

# 5) methods (adapters)
include("methods/EGM.jl")
include("methods/Projection.jl")
include("methods/Perturbation.jl")
include("methods/NN.jl")   # NN adapter (initial placeholder)

# 6) method factory
include("core/method_factory.jl")

# 7) simulation
include("sim/panel.jl")
using .SimPanel: simulate_panel

# 8) analysis
include("analysis/SteadyState.jl")
using .SteadyState: steady_state_analytic, steady_state_from_policy

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
    steady_state_from_policy

# --- Extensions ---
include("viz/api.jl")      # stubs only
export plot_policy, plot_euler_errors        # users call ThesisProject.plot_policy(...), etc.

end # module
