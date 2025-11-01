#!/usr/bin/env julia

module NNAIOSweep

"""
Evaluate NN solver configurations (AiO objective) on the baseline scalar model.

This script sweeps a curated set of neural-network hyperparameters while keeping
the underlying economic model fixed. It records convergence diagnostics, writes
a tidy CSV/JSON, and produces plots plus a Markdown summary that highlight the
best-performing configuration for the simple model.

Usage (defaults shown):
  julia --project scripts/experiments/nn_aio_sweep.jl \
        --base=config/scalar_ar1_baseline.yaml \
        --outdir=outputs/experiments/nn_aio_sweep

Additional flags:
  --summary-only   Skip writing CSV/plots; just print console summary.
  --repeats=N      Repeat each variant N times (default: 1).
"""

import Pkg
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(ROOT; io = devnull)

using Dates
using JSON3
using Printf
using Random
using Statistics
using DataFrames
using CSV
using Plots
using ThesisProject

include(joinpath(@__DIR__, "..", "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

struct SweepOptions
    base_path::String
    out_dir::String
    repeats::Int
    summary_only::Bool
end

function parse_cli(args)::SweepOptions
    base_path = joinpath(ROOT, "config", "scalar_ar1_baseline.yaml")
    out_dir = joinpath(ROOT, "outputs", "experiments", "nn_aio_sweep")
    repeats = 1
    summary_only = false

    for arg in args
        if startswith(arg, "--base=")
            base_path = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--outdir=")
            out_dir = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--repeats=")
            repeats = max(1, parse(Int, split(arg, "=", limit = 2)[2]))
        elseif arg == "--summary-only"
            summary_only = true
        end
    end

    return SweepOptions(base_path, out_dir, repeats, summary_only)
end

# ---------------------------------------------------------------------------
# Variant definitions (no model parameter changes allowed)
# ---------------------------------------------------------------------------

struct Variant
    name::Symbol
    description::String
    solver_overrides::NamedTuple
    nn_overrides::NamedTuple
end

const VARIANTS = Variant[
    Variant(
        :stability_baseline,
        "120k epochs, lr=2e-4, sigma_shocks off",
        (; method = :NN, verbose = false),
        (
            optimizer = :adamW,
            epochs = 120_000,
            batch = 4_096,
            lr = 2.0e-4,
            lr_min = 5.0e-6,
            lr_max = 2.0e-4,
            lr_decay_horizon = 60_000,
            warmup_epochs = 1_000,
            samples_per_epoch = 512,
            resample_every = 6,
            n_mc = 64,
            sigma_shocks = nothing,
        ),
    ),
    Variant(
        :moderate_lr,
        "100k epochs, lr=3e-4, moderate sampling",
        (; method = :NN, verbose = false),
        (
            optimizer = :adamW,
            epochs = 100_000,
            batch = 2_048,
            lr = 3.0e-4,
            lr_min = 5.0e-6,
            lr_max = 3.0e-4,
            lr_decay_horizon = 40_000,
            warmup_epochs = 800,
            samples_per_epoch = 256,
            resample_every = 8,
            n_mc = 64,
            sigma_shocks = nothing,
        ),
    ),
    Variant(
        :large_samples,
        "Larger sample cloud (192) with shock jitter",
        (; method = :NN, verbose = false),
        (
            optimizer = :adamW,
            epochs = 120_000,
            batch = 4096,
            lr = 2.5e-4,
            lr_min = 5.0e-6,
            lr_max = 2.5e-4,
            lr_decay_horizon = 30_000,
            warmup_epochs = 900,
            samples_per_epoch = 192,
            resample_every = 6,
            n_mc = 64,
            sigma_shocks = 0.05,
        ),
    ),
    Variant(
        :low_lr,
        "150k epochs, lr=1.5e-4, heavy smoothing",
        (; method = :NN, verbose = false),
        (
            optimizer = :adamW,
            epochs = 150_000,
            batch = 4096,
            lr = 1.5e-4,
            lr_min = 5.0e-6,
            lr_max = 1.5e-4,
            lr_decay_horizon = 80_000,
            warmup_epochs = 1_200,
            samples_per_epoch = 384,
            resample_every = 10,
            n_mc = 64,
            sigma_shocks = nothing,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function ensure_output_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function override_solver(cfg::NamedTuple, variant::Variant)
    solver_block = get_nested(cfg, (:solver,), NamedTuple())
    nn_block = get_nested(cfg, (:solver, :nn), NamedTuple())

    new_solver = merge(solver_block, variant.solver_overrides)
    new_nn = merge(nn_block, variant.nn_overrides)

    # Ensure AiO objective and no BCMC toggles
    new_nn = merge(new_nn, (objective = :euler_fb_aio, bcmc_auto_N = false))

    new_solver = merge(new_solver, (nn = new_nn,))
    return merge_section(cfg, :solver, new_solver)
end

function run_variant(base_cfg::NamedTuple, variant::Variant, repeat_id::Int)
    cfg = override_solver(base_cfg, variant)

    # Force deterministic seed by nudging master RNG via derive_rng tag
    rng_tag = Symbol(:nn_sweep_, variant.name, :_, repeat_id)
    master_rng = cfg.random.master_rng
    local_rng = ThesisProject.Determinism.derive_rng(master_rng, string(rng_tag))

    sol = ThesisProject.solve(cfg; rng = local_rng)
    sol isa Vector && (sol = sol[1])

    metadata = sol.metadata
    diagnostics = sol.diagnostics

    runtime = hasproperty(diagnostics, :runtime) ? Float64(diagnostics.runtime) : NaN
    iterations =
        hasproperty(diagnostics, :iterations) ? Float64(diagnostics.iterations) : NaN
    mean_ee = hasproperty(diagnostics, :mean_ee) ? Float64(diagnostics.mean_ee) : NaN
    max_resid = haskey(metadata, :max_resid) ? Float64(metadata[:max_resid]) : NaN
    converged = haskey(metadata, :converged) ? Bool(metadata[:converged]) : false
    rmse_history =
        haskey(metadata, :rmse_history) ? Float64.(metadata[:rmse_history]) : Float64[]
    status = haskey(metadata, :error) ? :error : :ok
    message = status == :error ? sprint(showerror, metadata[:error]) : ""
    solver_opts = haskey(metadata, :opts) ? metadata[:opts] : NamedTuple()

    return (
        variant = variant.name,
        repeat = repeat_id,
        runtime = runtime,
        iterations = iterations,
        mean_ee = mean_ee,
        final_rmse = max_resid,
        converged = converged,
        description = variant.description,
        rmse_history = rmse_history,
        solver_opts = solver_opts,
        status = status,
        message = message,
    )
end

function aggregate_results(results)
    rows = NamedTuple[]
    for res in results
        push!(
            rows,
            (
                variant = String(res.variant),
                repeat = res.repeat,
                runtime = res.runtime,
                iterations = res.iterations,
                mean_ee = res.mean_ee,
                final_rmse = res.final_rmse,
                converged = res.converged,
                status = String(res.status),
            ),
        )
    end
    df = DataFrame(rows)
    ok_df = filter(row -> row.status == "ok", df)
    grouped = combine(
        groupby(ok_df, :variant),
        :runtime => mean => :runtime_mean,
        :runtime => std => :runtime_std,
        :final_rmse => mean => :rmse_mean,
        :final_rmse => std => :rmse_std,
        :mean_ee => mean => :mean_ee_mean,
        :converged => x -> mean(Float64.(x)) => :convergence_rate,
    )
    sort!(grouped, :rmse_mean)
    return df, grouped
end

function write_results(
    out_dir::AbstractString,
    df::DataFrame,
    summary::DataFrame,
    detailed_results,
)
    ensure_output_dir(out_dir)
    csv_path = joinpath(out_dir, "nn_aio_sweep_runs.csv")
    CSV.write(csv_path, df)

    summary_path = joinpath(out_dir, "nn_aio_sweep_summary.csv")
    CSV.write(summary_path, summary)

    json_path = joinpath(out_dir, "nn_aio_sweep_raw.json")
    json_payload = [
        Dict(
            :variant => String(res.variant),
            :repeat => res.repeat,
            :runtime => res.runtime,
            :iterations => res.iterations,
            :mean_ee => res.mean_ee,
            :final_rmse => res.final_rmse,
            :converged => res.converged,
            :status => String(res.status),
            :message => res.message,
            :rmse_history => res.rmse_history,
        ) for res in detailed_results
    ]
    open(json_path, "w") do io
        JSON3.write(io, json_payload; indent = 2)
    end

    return csv_path, summary_path, json_path
end

function plot_rmse_bar(summary::DataFrame, out_dir::AbstractString)
    bar(
        summary.variant,
        summary.rmse_mean;
        yerror = summary.rmse_std,
        xlabel = "Variant",
        ylabel = "Final Euler RMSE",
        legend = false,
        rotation = 15,
        title = "Final Euler RMSE by NN configuration",
    )
    png_path = joinpath(out_dir, "rmse_bar.png")
    savefig(png_path)
    return png_path
end

function plot_runtime_vs_rmse(summary::DataFrame, out_dir::AbstractString)
    ann = collect(zip(summary.runtime_mean, summary.rmse_mean, summary.variant))
    scatter(
        summary.runtime_mean,
        summary.rmse_mean;
        xlabel = "Runtime (s)",
        ylabel = "Final Euler RMSE",
        title = "Runtime vs Residual",
        legend = false,
        marker = :circle,
        ms = 8,
        annotations = ann,
    )
    png_path = joinpath(out_dir, "runtime_vs_rmse.png")
    savefig(png_path)
    return png_path
end

function plot_rmse_history(detailed_results, out_dir::AbstractString)
    plt = plot(;
        xlabel = "Epoch",
        ylabel = "Euler RMSE",
        title = "RMSE trajectories",
        yscale = :log10,
    )
    for res in detailed_results
        isempty(res.rmse_history) && continue
        label = string(res.variant, "_r", res.repeat)
        plot!(plt, res.rmse_history; label = label)
    end
    png_path = joinpath(out_dir, "rmse_history.png")
    savefig(png_path)
    return png_path
end

function write_report(out_dir, summary::DataFrame, plots::Dict{Symbol,String})
    report_path = joinpath(out_dir, "report.md")
    open(report_path, "w") do io
        println(io, "# NN AiO Sweep Report")
        println(io)
        if nrow(summary) == 0
            println(io, "All variants failed (NaNs). See raw JSON for diagnostics.")
        else
            best_row = first(summary)
            println(
                io,
                "Best configuration: `$(best_row.variant)` with mean final RMSE = $(round(best_row.rmse_mean, digits=6)) and average runtime $(round(best_row.runtime_mean, digits=2)) s.",
            )
            println(io)
            println(io, "## Summary Table")
            println(io)
            show(io, MIME("text/plain"), summary)
            println(io)
        end
        println(io, "\n## Plots")
        for (label, path) in plots
            println(io, "- $(label): ![]($(basename(path)))")
        end
    end
    return report_path
end

function run()
    opts = parse_cli(ARGS)
    base_cfg = ThesisProject.load_config(opts.base_path)

    results = NamedTuple[]
    total = length(VARIANTS) * opts.repeats
    counter = 1
    println("Running $(length(VARIANTS)) NN variants × $(opts.repeats) repeats...")

    for variant in VARIANTS
        for rep = 1:opts.repeats
            @printf("[%d/%d] %s (repeat %d)\n", counter, total, variant.name, rep)
            push!(results, run_variant(base_cfg, variant, rep))
            counter += 1
        end
    end

    df, summary = aggregate_results(results)
    println("\nSummary (sorted by final RMSE):")
    show(stdout, summary)
    println()

    if opts.summary_only
        return
    end

    ensure_output_dir(opts.out_dir)
    csv_path, summary_path, json_path = write_results(opts.out_dir, df, summary, results)
    plots = Dict{Symbol,String}()
    plots[:rmse_bar] = plot_rmse_bar(summary, opts.out_dir)
    plots[:runtime_vs_rmse] = plot_runtime_vs_rmse(summary, opts.out_dir)
    plots[:rmse_history] = plot_rmse_history(results, opts.out_dir)
    report_path = write_report(opts.out_dir, summary, plots)

    println("\nArtifacts written to $(opts.out_dir):")
    println("  - runs CSV: $(csv_path)")
    println("  - summary CSV: $(summary_path)")
    println("  - raw JSON: $(json_path)")
    for (label, path) in plots
        println("  - $(label) plot: $(path)")
    end
    println("  - report: $(report_path)")
end

run()

end # module
