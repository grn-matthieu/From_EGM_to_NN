#!/usr/bin/env julia
# scripts/show_uncovered.jl
#
# Usage:
#   julia --project=. scripts/show_uncovered.jl src/EGM.jl

using Printf

if isempty(ARGS)
    println("Usage: julia --project=. scripts/show_uncovered.jl <srcfile>")
    exit(1)
end

srcfile = abspath(ARGS[1])
covfile = srcfile * ".cov"

# Clean up temp coverage artifacts first
for ext in [".cov", ".info"]
    for f in readdir(dirname(srcfile); join = true)
        if endswith(f, ext)
            rm(f; force = true)
        end
    end
end

if !isfile(covfile)
    println("No .cov file found for $srcfile. Run Julia with --code-coverage=all first.")
    exit(1)
end

covlines = readlines(covfile)
srclines = readlines(srcfile)

println("Uncovered lines in $srcfile:\n")

for (i, (c, s)) in enumerate(zip(covlines, srclines))
    # c is coverage marker, s is source line
    if c == "0"
        @printf "%4d │ %s\n" i s
    end
end
