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

using .UtilsConfig
using .UtilsLogging
using .UtilsRandom

# ─── Re-exports for convenience ─────────────────────
export SimpleParams, default_simple_params
export u, inv_uprime, budget_next
export euler_residuals_simple
export solve_simple_egm, SimpleSolution
export UtilsConfig, UtilsLogging, UtilsRandom

end # module