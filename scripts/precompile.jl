# Precompile project dependencies to speed up subsequent test runs
using Pkg
Pkg.instantiate()
Pkg.precompile()
println("Precompilation complete.")
