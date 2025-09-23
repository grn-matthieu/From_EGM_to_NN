using ThesisProject            # loads the package; Plots triggers ThesisProjectPlotsExt
include(joinpath(@__DIR__, "utils.jl"))

const SMOKE_CFG_PATH =
    joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_config.yaml")
const SMOKE_CFG = load_config(SMOKE_CFG_PATH)

const SMOKE_STOCH_CFG_PATH =
    joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_config_stochastic.yaml")
const SMOKE_STOCH_CFG = load_config(SMOKE_STOCH_CFG_PATH)
