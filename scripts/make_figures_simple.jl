#!/usr/bin/env julia
"""
Generate policy and Euler-error plots for the simple consumer-saving model in
deterministic and stochastic settings. Figures are saved under `outputs/`.
Run:
  julia --project=. scripts/make_figures_simple.jl
"""
module MakeFiguresSimple
import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)
using ThesisProject
using ThesisProject.Determinism: make_rng
try
    @eval using Plots
catch err
    @warn "Plots not available; skipping figure generation" err
end
const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(ROOT, "outputs")
"""Ensure the output directory exists."""
function ensure_outdir()
    isdir(OUTDIR) || mkpath(OUTDIR)
    return OUTDIR
end
"""Load config, solve model, and save plots with the given stem."""
function run_one(cfg_path::AbstractString, stem::AbstractString; plot_kwargs...)
    cfg = ThesisProject.load_config(cfg_path)
    ThesisProject.validate_config(cfg)
    model = ThesisProject.build_model(cfg)
    method = ThesisProject.build_method(cfg)
    sol = ThesisProject.solve(model, method, cfg; rng = make_rng(0))
    if isdefined(MakeFiguresSimple, :Plots)
        plt_pol = ThesisProject.plot_policy(sol; vars = [:c, :a], plot_kwargs...)
        savefig(plt_pol, joinpath(OUTDIR, "policy_$(stem).png"))
        plt_ee = ThesisProject.plot_euler_errors(sol)
        savefig(plt_ee, joinpath(OUTDIR, "euler_errors_$(stem).png"))
    end
    return sol
end
"""Main entrypoint."""
function main()
    ensure_outdir()
    run_one(joinpath(ROOT, "config", "simple_baseline.yaml"), "det")
    run_one(joinpath(ROOT, "config", "simple_stochastic.yaml"), "stoch"; mean = true)
    println("Plots saved to $(OUTDIR)")
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
end # module
