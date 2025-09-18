#!/usr/bin/env julia
using Coverage
using YAML

# Load ignore globs from Codecov config (best-effort). Patterns use forward slashes.
function load_ignore_globs()
    root = normpath(joinpath(@__DIR__, ".."))
    cfg_paths = [joinpath(root, ".codecov.yml"), joinpath(root, "codecov.yml")]
    for p in cfg_paths
        if isfile(p)
            try
                y = YAML.load_file(p)
                ig = get(y, "ignore", String[])
                ig isa AbstractVector || (ig = String[])
                return String.(ig)
            catch err
                @warn "Failed to parse Codecov config; proceeding without extra ignores" file =
                    p error = err
            end
        end
    end
    return String[]
end

# Convert a Codecov-style glob into a Regex. Normalize path separators to '/'.
function glob_to_regex(glob::AbstractString)
    s = replace(glob, "\\" => "/")
    # Escape regex meta, then reinstate globs
    s = replace(s, r"([.()+^$|])" => s"\\\1")
    s = replace(s, "**" => ".*")
    s = replace(s, "*" => "[^/]*")
    s = replace(s, "?" => "[^/]")
    return Regex("^" * s * "\$")
end

function filter_ignored(cl)
    root = normpath(joinpath(@__DIR__, ".."))
    globs = load_ignore_globs()
    res = Regex[glob_to_regex(g) for g in globs]
    return [fc for fc in cl if begin
        rel = replace(relpath(fc.filename, root), "\\" => "/")
        !any(r -> occursin(r, rel), res)
    end]
end

all_fc = process_folder(joinpath(@__DIR__, "..", "src"))
cl = filter_ignored(all_fc)

LCOV.writefile(joinpath(@__DIR__, "..", "lcov.info"), cl)

using Coverage: get_summary
covered, total = get_summary(cl)
percent = total == 0 ? 0.0 : 100 * covered / total
println("coverage_percent=", round(percent, digits = 2))

# Print per-file coverage sorted ascending
println("file_coverage:")
file_stats = [(fc, get_summary(fc)) for fc in cl]
sort!(file_stats; by = x -> (x[2][2] == 0 ? 0.0 : x[2][1] / x[2][2]))
for (fc, (cov, tot)) in file_stats
    pct = tot == 0 ? 0.0 : 100 * cov / tot
    println(
        " ",
        relpath(fc.filename, joinpath(@__DIR__, "..")),
        " => ",
        round(pct, digits = 2),
        "% (",
        cov,
        "/",
        tot,
        ")",
    )
end

# Best-effort cleanup of coverage artifacts when run standalone
try
    root = joinpath(@__DIR__, "..")
    for (dir, _, files) in walkdir(root)
        for f in files
            if endswith(f, ".cov") || endswith(f, ".info") || f == "lcov.info"
                rm(joinpath(dir, f); force = true)
            end
        end
    end
catch err
    @warn "Failed to cleanup coverage artifacts" error = err
end
