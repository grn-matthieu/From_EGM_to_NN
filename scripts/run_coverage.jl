#!/usr/bin/env julia
# scripts/show_uncovered.jl
#
# Usage:
#   julia --project=. scripts/show_uncovered.jl src/methods/EGM.jl
#
# Finds the most relevant existing .cov for the given source file:
# 1) Exact neighbor: <src>.cov
# 2) Any *.cov in repo whose tail path matches "<relative-src>.cov"
# 3) Any *.cov in repo whose basename matches "<basename(src)>.cov"
# Picks the newest (mtime) among matches and prints uncovered lines.

using Printf, Dates

if isempty(ARGS)
    println("Usage: julia --project=. scripts/show_uncovered.jl <srcfile>")
    exit(1)
end

srcfile = abspath(ARGS[1])
if !isfile(srcfile)
    println("Not a file: $srcfile")
    exit(1)
end

# Repo root = parent of this script
ROOT = normpath(joinpath(@__DIR__, ".."))
rel_src = try
    relpath(srcfile, ROOT)
catch
    basename(srcfile)
end

function cov_candidates(src::AbstractString, root::AbstractString)
    candidates = String[]

    # 1) Exact neighbor
    exact = src * ".cov"
    if isfile(exact)
        push!(candidates, exact)
    end

    # 2) Match by tail path "<rel>.cov"
    target_tail = rel_src * ".cov"
    for (d, _, fs) in walkdir(root)
        for f in fs
            endswith(f, ".cov") || continue
            covpath = joinpath(d, f)
            # Compare tail of path to target_tail
            tail = joinpath(splitpath(covpath)[end-length(splitpath(target_tail))+1:end]...)
            if endswith(replace(covpath, "\\" => "/"), replace(target_tail, "\\" => "/"))
                push!(candidates, covpath)
            end
        end
    end

    # 3) Fallback by basename match
    base_target = basename(src) * ".cov"
    for (d, _, fs) in walkdir(root)
        for f in fs
            if f == base_target
                push!(candidates, joinpath(d, f))
            end
        end
    end

    unique(candidates)
end

cands = cov_candidates(srcfile, ROOT)
if isempty(cands)
    println("No .cov found for $srcfile. Run with --code-coverage first.")
    exit(1)
end

# Pick newest by mtime
covfile = sort(cands; by = f -> stat(f).mtime, rev = true)[1]

covlines = readlines(covfile)
srclines = readlines(srcfile)
limit = min(length(covlines), length(srclines))

println("Using coverage file: $covfile")
println("Uncovered lines in $srcfile:\n")

printed = false
for i = 1:limit
    cov = strip(covlines[i])
    if cov == "0"
        @printf "%5d │ %s\n" i srclines[i]
        printed = true
    end
end
if !printed
    println("  None")
end
