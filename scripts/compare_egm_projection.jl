#!/usr/bin/env julia
"""
Compare EGM and Projection results on the baseline configuration, reporting
summary statistics and saving policy/Euler-error comparison plots.
Run:
  julia --project=. scripts/compare_egm_projection.jl
"""
module CompareEGMProjection
import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)
using ThesisProject
using ThesisProject.Determinism: make_rng
using Statistics: mean, maximum, minimum
using Plots
const ROOT = normpath(joinpath(@__DIR__, "..", "outputs"))
"""Euler error summary statistics for a Solution."""
function ee_stats(sol)
    pol_c = sol.policy[:c]
    ee = pol_c.euler_errors
    ee_mat = pol_c.euler_errors_mat
    data = ee_mat === nothing ? ee : vec(ee_mat)
    data = collect(skipmissing(data))
    return (max = maximum(data), min = minimum(data), mean = mean(data))
end
"""Summary statistics of absolute policy differences between two Solutions."""
function policy_diff_stats(sol_a, sol_b)
    c_a = sol_a.policy[:c].value
    c_b = sol_b.policy[:c].value
    a_a = sol_a.policy[:a].value
    a_b = sol_b.policy[:a].value
    diff_c = abs.(c_a .- c_b)
    diff_a = abs.(a_a .- a_b)
    stats(v) = (max = maximum(v), min = minimum(v), mean = mean(v))
    return (c = stats(diff_c), a = stats(diff_a))
end
function run()
    cfg = ThesisProject.load_config(
        joinpath(@__DIR__, "..", "config", "simple_baseline.yaml"),
    )
    ThesisProject.validate_config(cfg)
    cfg_egm = deepcopy(cfg)
    cfg_egm[:solver][:method] = :EGM
    model_egm = ThesisProject.build_model(cfg_egm)
    method_egm = ThesisProject.build_method(cfg_egm)
    sol_egm = ThesisProject.solve(model_egm, method_egm, cfg_egm; rng = make_rng(0))
    cfg_proj = deepcopy(cfg)
    cfg_proj[:solver][:method] = :Projection
    model_proj = ThesisProject.build_model(cfg_proj)
    method_proj = ThesisProject.build_method(cfg_proj)
    sol_proj = ThesisProject.solve(model_proj, method_proj, cfg_proj; rng = make_rng(0))
    ee_egm = ee_stats(sol_egm)
    ee_proj = ee_stats(sol_proj)
    diffs = policy_diff_stats(sol_egm, sol_proj)
    println("Euler error stats (max/min/mean):")
    println("  EGM       : $(ee_egm)")
    println("  Projection: $(ee_proj)")
    println("\nPolicy differences |EGM - Projection| summary:")
    println("  c: $(diffs.c)")
    println("  a: $(diffs.a)")
    # Policy comparison: plot consumption and asset policies separately
    agrid = sol_egm.policy[:a].grid
    plt_c = plot(agrid, sol_egm.policy[:c].value; label = "EGM")
    plot!(plt_c, agrid, sol_proj.policy[:c].value; label = "Projection")
    xlabel!(plt_c, "state")
    ylabel!(plt_c, "c policy")
    title!(plt_c, "Consumption Policy")
    savefig(plt_c, joinpath(ROOT, "compare_policy_c.png"))
    plt_a = plot(agrid, sol_egm.policy[:a].value; label = "EGM")
    plot!(plt_a, agrid, sol_proj.policy[:a].value; label = "Projection")
    xlabel!(plt_a, "state")
    ylabel!(plt_a, "a policy")
    title!(plt_a, "Asset Policy")
    savefig(plt_a, joinpath(ROOT, "compare_policy_a.png"))
    # Euler-error comparison (log scale) with zero/negative values clipped
    ee_egm = sol_egm.policy[:c].euler_errors
    ee_proj = sol_proj.policy[:c].euler_errors
    ee_egm = map(x -> x > 0 ? x : eps(Float64), ee_egm)
    ee_proj = map(x -> x > 0 ? x : eps(Float64), ee_proj)
    ymin = min(minimum(ee_egm), minimum(ee_proj))
    ymax = max(maximum(ee_egm), maximum(ee_proj))
    exps = floor(Int, log10(ymin)):ceil(Int, log10(ymax))
    yticks = 10.0 .^ collect(exps)
    plt_err = plot(agrid, ee_egm; label = "EGM", yscale = :log10, yticks = yticks)
    plot!(plt_err, agrid, ee_proj; label = "Projection")
    xlabel!(plt_err, "State")
    ylabel!(plt_err, "Abs. EErr (log10)")
    title!(plt_err, "Euler Errors")
    savefig(plt_err, joinpath(ROOT, "compare_euler_errors.png"))
end
if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
end # module
