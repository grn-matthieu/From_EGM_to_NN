# Usage: julia --project=. scripts/run_single_test.jl <relative-test-path>
using Test
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
if length(ARGS) < 1
    println("Usage: julia --project=. scripts/run_single_test.jl <relative-test-path>")
    exit(1)
end
rel = ARGS[1]
path = joinpath(@__DIR__, "..", rel)
if !isfile(path)
    println("Test file not found: ", path)
    exit(1)
end
println("Running tests in: ", path)
include(path)
println("Finished")
