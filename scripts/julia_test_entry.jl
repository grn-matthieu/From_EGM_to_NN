using Test
ENV["GKSwstype"] = "100"        # headless GR on CI-like runs
using ThesisProject
include(joinpath(@__DIR__, "..", "test", "bootstrap.jl"))
include(abspath(ARGS[1]))
