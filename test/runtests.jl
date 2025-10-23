using Test
using ThesisProject

include(joinpath(@__DIR__, "utils.jl"))
using .TestUtils

@testset "ThesisProject" begin
    include(joinpath(@__DIR__, "unit", "utils_config_determinism_tests.jl"))
    include(joinpath(@__DIR__, "unit", "shocks_and_model_tests.jl"))
    include(joinpath(@__DIR__, "unit", "solvers_common_tests.jl"))
    include(joinpath(@__DIR__, "unit", "nn_scheduler_tests.jl"))
    include(joinpath(@__DIR__, "integration", "core_pipeline_tests.jl"))
    include(joinpath(@__DIR__, "integration", "simulation_analysis_tests.jl"))
    include(joinpath(@__DIR__, "integration", "nn_solver_tests.jl"))
    include(joinpath(@__DIR__, "integration", "nn_bcmc_tests.jl"))
end
