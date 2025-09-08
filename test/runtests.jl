using Test
ENV["GKSwstype"] = "100"        # headless GR on CI
using ThesisProject, Plots       # Plots triggers ThesisProjectPlotsExt

include("test_core.jl")
include("test_shocks.jl")
include("test_viz.jl")
include("test_quality.jl")
