#!/usr/bin/env julia
# Simple coverage summary printer
try
    using Coverage
catch
    # Ensure Coverage.jl is available when run outside Pkg.test
    import Pkg
    Pkg.activate(; temp = false)
    Pkg.add("Coverage")
    using Coverage
end

c = process_folder()
covered, total = get_summary(c)
percent = total == 0 ? 0.0 : covered / total * 100

println("COVERED_LINES=", covered)
println("TOTAL_LINES=", total)
println("PERCENT=", round(percent; digits = 2))
