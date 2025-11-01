#!/usr/bin/env julia

module NNAIOSweep

"""
Evaluate NN solver configurations (AiO objective) on the stochastic smoke setup.

This script sweeps a curated set of neural-network hyperparameters against the
`smoke_cfg_stoch.yaml` baseline. It records convergence diagnostics, writes tidy
CSV/JSON/Markdown artifacts, and produces production-ready plots that highlight
the most reliable configurations for the stochastic consumption-saving model.

Usage (defaults shown):
  julia --project scripts/experiments/nn_aio_sweep.jl \
        --base=config/smoke_cfg_stoch.yaml \
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

const PLOT_SIZE = (960, 640)
const PLOT_DPI = 180
const BASE_PALETTE = palette(:viridis, 8)
const PlotArtifact = NamedTuple{(:png, :pdf),Tuple{String,String}}

Plots.default(;
    size = PLOT_SIZE,
    dpi = PLOT_DPI,
    background_color = :white,
    foreground_color = :black,
    grid = true,
    gridalpha = 0.25,
    gridlinewidth = 0.6,
    legendfontsize = 10,
    guidefontsize = 13,
    tickfontsize = 11,
    titlefontsize = 16,
    linewidth = 2,
)

save_plot_all(plt, out_dir::AbstractString, name::AbstractString) = begin
    base = joinpath(out_dir, name)
    png_path = base * ".png"
    pdf_path = base * ".pdf"
    savefig(plt, png_path)
    savefig(plt, pdf_path)
    (png = png_path, pdf = pdf_path)
end

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
    base_path = joinpath(ROOT, "config", "smoke_cfg_stoch.yaml")
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
        :baseline_control,
        "120k epochs, 64-sample batches, conservative LR schedule",
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
            samples_per_epoch = 64,
            eval_samples = 8_192,
            resample_every = 5,
            n_mc = 64,
            sigma_shocks = nothing,
        ),
    ),
    Variant(
        :precision_highmc,
        "160k epochs, wider 96×96 net, dense sampling, n_mc=96",
        (; method = :NN, verbose = false),
        (
            optimizer = :adamW,
            epochs = 160_000,
            batch = 4_096,
            lr = 1.5e-4,
            lr_min = 1.0e-5,
            lr_max = 1.5e-4,
            lr_decay_horizon = 90_000,
            warmup_epochs = 1_600,
            hid1 = 96,
            hid2 = 96,
            samples_per_epoch = 192,
            eval_samples = 16_384,
            resample_every = 6,
            n_mc = 96,
            sigma_shocks = nothing,
        ),
    ),
    Variant(
        :aggressive_schedule,
        "90k epochs, higher LR, jittered shocks, rapid resampling",
        (; method = :NN, verbose = false),
        (
            optimizer = :adamW,
            epochs = 90_000,
            batch = 2_048,
            lr = 3.5e-4,
            lr_min = 5.0e-5,
            lr_max = 3.5e-4,
            lr_decay_horizon = 35_000,
            warmup_epochs = 700,
            samples_per_epoch = 96,
            eval_samples = 8_192,
            resample_every = 4,
            n_mc = 48,
            sigma_shocks = 0.05,
        ),
    ),
    Variant(
        :wide_network,
        "130k epochs, 128×128 net, large batches, eval grid 12k",
        (; method = :NN, verbose = false),
        (
            optimizer = :adamW,
            epochs = 130_000,
            batch = 6_144,
            lr = 2.5e-4,
            lr_min = 5.0e-6,
            lr_max = 2.5e-4,
            lr_decay_horizon = 70_000,
            warmup_epochs = 1_200,
            hid1 = 128,
            hid2 = 128,
            samples_per_epoch = 160,
            eval_samples = 12_288,
            resample_every = 6,
            n_mc = 80,
            sigma_shocks = nothing,
        ),
    ),
    Variant(
        :curriculum_resampling,
        "140k epochs, slow LR decay, resample every epoch, mild shock noise",
        (; method = :NN, verbose = false),
        (
            optimizer = :adamW,
            epochs = 140_000,
            batch = 4_096,
            lr = 2.2e-4,
            lr_min = 5.0e-6,
            lr_max = 2.2e-4,
            lr_decay_horizon = 70_000,
            warmup_epochs = 2_000,
            samples_per_epoch = 128,
            eval_samples = 8_192,
            resample_every = 1,
            n_mc = 64,
            sigma_shocks = 0.08,
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
    grouped = if nrow(ok_df) == 0
        DataFrame()
    else
        combine(
            groupby(ok_df, :variant),
            nrow => :runs,
            :runtime => mean => :runtime_mean,
            :runtime => std => :runtime_std,
            :runtime => x -> quantile(x, 0.9) => :runtime_p90,
            :final_rmse => mean => :rmse_mean,
            :final_rmse => std => :rmse_std,
            :final_rmse => median => :rmse_median,
            :final_rmse => x -> quantile(x, 0.9) => :rmse_p90,
            :mean_ee => mean => :mean_ee_mean,
            :mean_ee => std => :mean_ee_std,
            :converged => x -> mean(Float64.(x)) => :convergence_rate,
        )
    end
    if !isempty(grouped)
        for col in (:runtime_std, :rmse_std, :mean_ee_std)
            hasproperty(grouped, col) || continue
            grouped[!, col] = map(x -> isnan(x) ? 0.0 : x, grouped[!, col])
        end
        sort!(grouped, :rmse_mean)
    end
    return df, grouped
end

function tidy_summary(summary::DataFrame)
    isempty(summary) && return summary
    fmt = deepcopy(summary)
    if :convergence_rate in names(fmt)
        fmt[!, :convergence_rate] .= fmt[!, :convergence_rate] .* 100
        rename!(fmt, :convergence_rate => :convergence_pct)
    end
    two_digit_cols = Set([:convergence_pct])
    for col in names(fmt)
        data = fmt[!, col]
        if data isa AbstractVector{<:Real} && !(eltype(data) <: Integer)
            digits = col in two_digit_cols ? 2 : 4
            fmt[!, col] = round.(Float64.(data); digits = digits)
        end
    end
    return fmt
end

function markdown_table(df::DataFrame)
    isempty(df) && return "No successful runs."
    header = names(df)
    buf = IOBuffer()
    println(buf, "|" * join(string.(header), " | ") * "|")
    println(buf, "|" * join(fill("---", length(header)), " | ") * "|")
    for row in eachrow(df)
        vals = [string(row[h]) for h in header]
        println(buf, "|" * join(vals, " | ") * "|")
    end
    return String(take!(buf))
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

function plot_rmse_bar(summary::DataFrame, out_dir::AbstractString)::PlotArtifact
    if nrow(summary) == 0
        return (png = "", pdf = "")
    end
    order = summary.variant
    colors = BASE_PALETTE[1:nrow(summary)]
    values = summary.rmse_mean
    errs = summary.rmse_std
    plt = bar(
        order,
        values;
        yerror = errs,
        xlabel = "Variant",
        ylabel = "Final Euler RMSE",
        legend = false,
        rotation = 20,
        fillcolor = colors,
        linecolor = :black,
        bar_width = 0.6,
        title = "Final Euler RMSE by NN Configuration",
    )
    for (idx, val) in enumerate(values)
        annotate!(plt, idx, val, text(@sprintf("%.4f", val), :center, 10))
    end
    return save_plot_all(plt, out_dir, "rmse_bar")
end

function plot_runtime_vs_rmse(summary::DataFrame, out_dir::AbstractString)::PlotArtifact
    if nrow(summary) == 0
        return (png = "", pdf = "")
    end
    colors = BASE_PALETTE[1:nrow(summary)]
    best_idx = argmin(summary.rmse_mean)
    plt = scatter(
        summary.runtime_mean,
        summary.rmse_mean;
        xlabel = "Runtime (s)",
        ylabel = "Final Euler RMSE",
        title = "Runtime–Accuracy Frontier",
        legend = false,
        marker = :circle,
        markersize = 9,
        color = colors,
        framestyle = :box,
    )
    for i = 1:nrow(summary)
        lbl = string(summary.variant[i])
        annotate!(plt, summary.runtime_mean[i], summary.rmse_mean[i], text(lbl, :left, 9))
    end
    scatter!(
        plt,
        [summary.runtime_mean[best_idx]],
        [summary.rmse_mean[best_idx]];
        marker = (:star5, 14),
        color = :orange,
        label = "",
    )
    return save_plot_all(plt, out_dir, "runtime_vs_rmse")
end

function plot_rmse_history(detailed_results, out_dir::AbstractString)::PlotArtifact
    traces =
        filter(res -> res.status == :ok && !isempty(res.rmse_history), detailed_results)
    if isempty(traces)
        return (png = "", pdf = "")
    end
    plt = plot(
        xlabel = "Epoch",
        ylabel = "Euler RMSE",
        title = "Euler RMSE Trajectories",
        yscale = :log10,
        legend = :topright,
    )
    step = 100
    for (idx, res) in enumerate(traces)
        epochs = collect(step:step:step*length(res.rmse_history))
        color = BASE_PALETTE[1+mod(idx - 1, length(BASE_PALETTE))]
        label = string(res.variant, " (repeat ", res.repeat, ")")
        plot!(plt, epochs, res.rmse_history; label = label, color = color, alpha = 0.9)
    end
    return save_plot_all(plt, out_dir, "rmse_history")
end

function write_report(out_dir, summary::DataFrame, plots::Dict{Symbol,PlotArtifact})
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
            tidy = tidy_summary(summary)
            println(io, markdown_table(tidy))
            println(io)
        end
        println(io, "\n## Plots")
        for (label, artifact) in plots
            isempty(artifact.png) && continue
            println(io, "- $(label): ![]($(basename(artifact.png)))")
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

    out_root = ensure_output_dir(opts.out_dir)
    run_dir = joinpath(out_root, Dates.format(now(), "yyyymmdd_HHMMSS"))
    ensure_output_dir(run_dir)

    csv_path, summary_path, json_path = write_results(run_dir, df, summary, results)
    plots = Dict{Symbol,PlotArtifact}()
    if nrow(summary) > 0
        plots[:rmse_bar] = plot_rmse_bar(summary, run_dir)
        plots[:runtime_vs_rmse] = plot_runtime_vs_rmse(summary, run_dir)
    end
    plots[:rmse_history] = plot_rmse_history(results, run_dir)
    report_path = write_report(run_dir, summary, plots)

    println("\nArtifacts written to $(run_dir):")
    println("  - runs CSV: $(csv_path)")
    println("  - summary CSV: $(summary_path)")
    println("  - raw JSON: $(json_path)")
    for (label, artifact) in plots
        isempty(artifact.png) && continue
        println("  - $(label) plot (png): $(artifact.png)")
        println("    $(label) plot (pdf): $(artifact.pdf)")
    end
    println("  - report: $(report_path)")
end

run()

end # module
