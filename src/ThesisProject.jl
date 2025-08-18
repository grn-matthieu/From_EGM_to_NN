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

using .UtilsConfig
using .UtilsLogging
using .UtilsRandom
using .ValueFunction
using .PlotsUtils

# ─── Re-exports for convenience ─────────────────────
# Models
export SimpleParams, default_simple_params
export u, inv_uprime, budget_next

# Solvers
export solve_simple_egm, SimpleSolution

# Diagnostics (Euler residuals)
export euler_residuals_simple

# Utils (config/logging/random)
export UtilsConfig, UtilsLogging, UtilsRandom

# Value function + plotting utils
export compute_value
export save_plot, plot_policy, plot_value, plot_residuals

end # module
