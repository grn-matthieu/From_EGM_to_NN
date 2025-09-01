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

# egm specific
include("solvers/egm/residuals.jl")
include("solvers/egm/kernel.jl")

# 5) methods (adapters)
include("methods/EGM.jl")
# include("methods/NN.jl")   # ok if stub

# 6) simulation
# include("sim/shock_panel.jl")
# include("sim/panel.jl")
# include("sim/moments.jl")


using .UtilsConfig: load_config
using .ModelFactory: build_model
using .EGM: build_method, solve

# --- Exports ---

export load_config, build_model, build_method, solve

# --- Extensions ---
include("viz/api.jl")      # stubs only
export plot_policy         # users call ThesisProject.plot_policy(...)

end # module