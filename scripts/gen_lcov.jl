#!/usr/bin/env julia
try
    using Coverage
catch
    import Pkg
    Pkg.activate(; temp = false)
    Pkg.add("Coverage")
    using Coverage
end

c = process_folder()
mkpath(".")
Coverage.LCOV.writefile("lcov.info", c)
println("Wrote lcov.info with updated coverage data.")
