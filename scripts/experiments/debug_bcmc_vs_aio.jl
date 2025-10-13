#!/usr/bin/env julia
"""
Debug bc-MC vs AiO objectives on a tiny 2-dim income model (CSVAR).

What it does:
- Loads the smoke stochastic CSVAR calibration.
- Trains two NN solvers on the same grid and settings: one with :euler_fb_bcmc,
  one with :euler_fb_aio (old operator).
- Logs convergence info and Euler-residual summaries.
- Produces side-by-side policy plots for c(a) and a'(a).

Run:
  julia --project=. scripts/experiments/debug_bcmc_vs_aio.jl [--config=PATH] [--epochs=INT] [--n_mc=INT]

Outputs are saved under outputs/debug_bcmc/ by default.
"""
module DebugBCMCvsAiO

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)

using ThesisProject
using ThesisProject.Determinism: make_master_rng, derive_rng
using Statistics: mean
using Printf

include(joinpath(@__DIR__, "..", "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

# Try to load Plots if available (project has a weak dep via extension)
try
    @eval using Plots
catch err
    @warn "Plots not available; will skip figure generation" err
end

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTDIR = joinpath(ROOT, "outputs", "debug_bcmc")

"""Ensure output directory exists"""
function ensure_outdir()
    isdir(OUTDIR) || mkpath(OUTDIR)
    return OUTDIR
end

"""Simple CLI parse for optional overrides."""
function parse_cli(args)
    cfg_path = nothing
    epochs = nothing
    n_mc = nothing
    for a in args
        if startswith(a, "--config=")
            cfg_path = split(a, "=", limit = 2)[2]
        elseif startswith(a, "--epochs=")
            epochs = parse(Int, split(a, "=", limit = 2)[2])
        elseif startswith(a, "--n_mc=")
            n_mc = parse(Int, split(a, "=", limit = 2)[2])
        end
    end
    return (cfg_path, epochs, n_mc)
end

"""Build a config variant with a specific NN objective and optional overrides."""
function build_cfg(
    base_cfg,
    objective::Symbol;
    epochs::Union{Int,Nothing} = nothing,
    n_mc::Union{Int,Nothing} = nothing,
)
    local_over = NamedTuple()
    # Ensure method is NN
    cfg1 = merge_section(base_cfg, :solver, (; method = :NN))

    # Prepare overrides for solver.nn
    nn_over = Dict{Symbol,Any}(:objective => objective)
    if epochs !== nothing
        nn_over[:epochs] = epochs
    end
    if n_mc !== nothing
        nn_over[:n_mc] = n_mc
    end

    # Merge into existing solver.nn, then set back
    old_nn = get_nested(cfg1, (:solver, :nn), NamedTuple())
    new_nn = merge_config(old_nn, nn_over)
    cfg2 = set_nested(cfg1, (:solver, :nn), new_nn)
    return cfg2
end

"""Solve model with provided cfg and return (solution, label)."""
function solve_one(cfg, label::String, rng)
    model = ThesisProject.build_model(cfg)
    method = ThesisProject.build_method(cfg)
    sol = ThesisProject.solve(model, method, cfg; rng = rng)
    return sol, label
end

"""Pretty print convergence and residual summaries for a solution."""
function log_summary(sol, label::String)
    md = sol.metadata
    di = sol.diagnostics
    @printf(
        "[%s] converged=%s  iters=%d  runtime=%.3fs  max_resid=%.4g  mean_ee=%.4g\n",
        label,
        get(md, :converged, missing),
        get(di, :iterations, missing),
        get(di, :runtime, missing),
        get(md, :max_resid, missing),
        get(di, :mean_ee, missing)
    )
end

"""Compute the cash-on-hand grid implied by the solution's model config."""
function cash_on_hand_grid(sol)
    p = ThesisProject.get_params(sol.model)
    g = ThesisProject.get_grids(sol.model)
    agrid = g[:a].grid
    Rg = 1 + p.r
    income = begin
        if hasproperty(p, :y) && p.y isa AbstractVector
            # For CSVAR: income proxy uses mean of components
            ThesisProject.CSVarUtils.csvar_income(collect(p.y))
        else
            # Scalar log-income compatible path: use exp(mean log)
            exp(p.y)
        end
    end
    return agrid, Rg .* agrid .+ income
end

"""Save two policy plots (c and a'), overlaying two solutions on the same grid.
Also produce support-restricted plots based on solver.nn w-range."""
function save_policy_plots(sol_a, label_a::String, sol_b, label_b::String)
    isdefined(DebugBCMCvsAiO, :Plots) || return nothing
    outdir = ensure_outdir()

    agrid = sol_a.policy[:a].grid
    @assert agrid == sol_b.policy[:a].grid "Asset grids differ between runs"

    # Determine training w-range from config
    cfg_a = sol_a.method.opts
    w_lo = get(cfg_a, :w_min, 0.0)
    w_hi = get(cfg_a, :w_max, 0.0)
    a_for_w, w_grid = cash_on_hand_grid(sol_a)
    in_support = (w_grid .>= w_lo) .& (w_grid .<= w_hi)
    frac_support = sum(in_support) / max(length(in_support), 1)
    @printf "[plot] Fraction of plotted states within training w-range [%.3g, %.3g]: %.1f%% (Na=%d)\n" w_lo w_hi 100 *
                                                                                                                 frac_support length(
        in_support,
    )

    # c(a)
    c_a = sol_a.policy[:c].value
    c_b = sol_b.policy[:c].value
    plt_c = plot(agrid, c_a; label = label_a, lw = 2)
    plot!(plt_c, agrid, c_b; label = label_b, lw = 2)
    xlabel!(plt_c, "a")
    ylabel!(plt_c, "c(a)")
    title!(plt_c, "Consumption policy comparison")
    savefig(plt_c, joinpath(outdir, "policy_c_$(label_a)_vs_$(label_b).png"))

    # c(a) restricted to training support
    if any(in_support)
        plt_c_s = plot(agrid[in_support], c_a[in_support]; label = label_a, lw = 2)
        plot!(plt_c_s, agrid[in_support], c_b[in_support]; label = label_b, lw = 2)
        xlabel!(plt_c_s, "a (within w-support)")
        ylabel!(plt_c_s, "c(a)")
        title!(
            plt_c_s,
            @sprintf("Consumption policy (in-support [%.2f, %.2f])", w_lo, w_hi)
        )
        savefig(plt_c_s, joinpath(outdir, "policy_c_support_$(label_a)_vs_$(label_b).png"))
    end

    # a'(a)
    a_a = sol_a.policy[:a].value
    a_b = sol_b.policy[:a].value
    plt_a = plot(agrid, a_a; label = label_a, lw = 2)
    plot!(plt_a, agrid, a_b; label = label_b, lw = 2)
    xlabel!(plt_a, "a")
    ylabel!(plt_a, "a'(a)")
    title!(plt_a, "Next-asset policy comparison")
    savefig(plt_a, joinpath(outdir, "policy_a_$(label_a)_vs_$(label_b).png"))

    # a'(a) restricted
    if any(in_support)
        plt_a_s = plot(agrid[in_support], a_a[in_support]; label = label_a, lw = 2)
        plot!(plt_a_s, agrid[in_support], a_b[in_support]; label = label_b, lw = 2)
        xlabel!(plt_a_s, "a (within w-support)")
        ylabel!(plt_a_s, "a'(a)")
        title!(plt_a_s, @sprintf("Next-asset policy (in-support [%.2f, %.2f])", w_lo, w_hi))
        savefig(plt_a_s, joinpath(outdir, "policy_a_support_$(label_a)_vs_$(label_b).png"))
    end

    return nothing
end

function main()
    ensure_outdir()
    cfg_path_default = joinpath(ROOT, "config", "smoke_cfg_csvar_stoch.yaml")
    cfg_path, opt_epochs, opt_nmc = parse_cli(ARGS)
    cfg_file = cfg_path === nothing ? cfg_path_default : cfg_path
    base_cfg = ThesisProject.load_config(cfg_file)

    # Two objective variants
    cfg_bcmc = build_cfg(base_cfg, :euler_fb_bcmc; epochs = opt_epochs, n_mc = opt_nmc)
    cfg_aio = build_cfg(base_cfg, :euler_fb_aio; epochs = opt_epochs)

    master = make_master_rng(2025)
    sol_bcmc, _ = solve_one(cfg_bcmc, "bcMC", derive_rng(master, :bcmc))
    sol_aio, _ = solve_one(cfg_aio, "AiO", derive_rng(master, :aio))

    println("=== Convergence summaries ===")
    log_summary(sol_bcmc, "bcMC")
    log_summary(sol_aio, "AiO")

    # Euler-residual quick stats (already in metadata/diagnostics)
    ee_bcmc = sol_bcmc.policy[:c].euler_errors
    ee_aio = sol_aio.policy[:c].euler_errors
    @printf(
        "[bcMC] mean(|EE|)=%.4g  max(|EE|)=%.4g\n",
        mean(abs.(ee_bcmc)),
        maximum(abs.(ee_bcmc))
    )
    @printf(
        "[AiO ] mean(|EE|)=%.4g  max(|EE|)=%.4g\n",
        mean(abs.(ee_aio)),
        maximum(abs.(ee_aio))
    )

    # Plots
    save_policy_plots(sol_bcmc, "bcMC", sol_aio, "AiO")
    println("Outputs saved to: " * OUTDIR)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
