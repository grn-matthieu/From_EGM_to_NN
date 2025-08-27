module ThesisProject

# ─── Models ─────────────────────────────────────────
include("models/simple/Calibration.jl")
include("models/simple/ModelSimple.jl")

using .SimpleCalibration
using .SimpleModel

# ─── Solvers ────────────────────────────────────────
include("solvers/egm/EulerResiduals.jl")
include("solvers/egm/EGM.jl")

using .EGMResiduals
using .EGMSolver

# ─── Utils ─────────────────────────────────────────
include("utils/Config.jl")
include("utils/Logging.jl")
include("utils/RandomTools.jl")
include("utils/ValueFunction.jl")
include("utils/PlotsUtils.jl")
include("models/shared/Shocks.jl")


using .UtilsConfig
using .UtilsLogging
using .UtilsRandom
using .ValueFunction
using .PlotsUtils
using .Shocks

# ─── Re-exports for convenience ─────────────────────
# Models
export SimpleParams, default_simple_params
export u, inv_uprime, budget_next

# Solvers
export SimpleSolution
export solve_simple_egm, solve_stochastic_egm

# Diagnostics (Euler residuals)
export euler_residuals_stochastic, euler_residuals_simple

# Utils (config/logging/random)
export UtilsConfig, UtilsLogging, UtilsRandom, load_config

# Value function + plotting utils
export compute_value
export save_plot, plot_policy, plot_value, plot_residuals

# Shock discretization
export discretize, ShockOutput

end # module
