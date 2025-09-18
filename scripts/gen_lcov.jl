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

# Cleanup artifacts if desired when run locally
try
    for (dir, _, files) in walkdir(".")
        for f in files
            if endswith(f, ".cov") || endswith(f, ".info") || f == "lcov.info"
                rm(joinpath(dir, f); force = true)
            end
        end
    end
catch err
    @warn "Cleanup of coverage artifacts failed" error = err
end
