using Test
ENV["GKSwstype"] = "100"        # headless GR on CI

using ThesisProject            # loads the package; Plots triggers ThesisProjectPlotsExt
include(joinpath(@__DIR__, "utils.jl"))

const SMOKE_CFG_PATH = joinpath(@__DIR__, "..", "config", "smoke_cfg_det.yaml")
const SMOKE_CFG = load_config(SMOKE_CFG_PATH)

const SMOKE_STOCH_CFG_PATH = joinpath(@__DIR__, "..", "config", "smoke_cfg_stoch.yaml")
const SMOKE_STOCH_CFG = load_config(SMOKE_STOCH_CFG_PATH)
