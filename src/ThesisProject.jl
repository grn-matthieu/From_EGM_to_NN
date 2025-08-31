module ThesisProject

# Includes
include("core/api.jl")
include("core/model_contract.jl")
include("utils/Config.jl")

include("models/shared/Shocks.jl")
include("models/baseline/ConsumerSaving.jl")
include("core/model_factory.jl")

include("methods/EGM.jl")


# Usings
using .API
using .ModelContract
using .UtilsConfig
using .Shocks
using .ConsumerSaving
using .ModelFactory
using .EGM

# Exports

export load_config, build_model, build_method, solve

end # module