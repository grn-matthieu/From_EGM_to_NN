using Test
ENV["GKSwstype"] = "100"        # headless GR on CI
using ThesisProject       # Plots triggers ThesisProjectPlotsExt

include("test_core.jl")
#include("test_shocks.jl")
include("test_residuals.jl")
include("test_sim.jl")
include("test_egm_stoch.jl")
include("test_solver_options.jl")
include("test_interp.jl")
include("test_determinism.jl")
include("test_viz.jl")
include("test_quality.jl")
include("test_value_function.jl")
include("test_chebyshev.jl")
