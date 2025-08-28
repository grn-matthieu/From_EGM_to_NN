#= # src/startup.jl
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Default values
Na = 1000
Nz = 7
tol = 1e-8

for arg in ARGS
    if occursin("--Na=", arg)
        Na = parse(Int, split(arg, "=")[2])
    elseif occursin("--Nz=", arg)
        Nz = parse(Int, split(arg, "=")[2])
    elseif occursin("--tol=", arg)
        tol = parse(Float64, split(arg, "=")[2])
    end
end

println("Grid size Na = $Na, Nz = $Nz, tol = $tol") =#