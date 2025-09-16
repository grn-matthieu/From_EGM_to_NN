#!/usr/bin/env julia
using Coverage

cl = process_folder(joinpath(@__DIR__, "..", "src"))
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
