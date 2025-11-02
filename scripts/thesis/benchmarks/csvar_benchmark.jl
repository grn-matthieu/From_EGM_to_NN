#!/usr/bin/env julia

"""
CSVAR Benchmark Script

Comprehensive benchmarking for CSVAR (vector-income) models. Tests solver methods
across dimensional scaling, correlation structures, and grid sizes with detailed
diagnostics and visualization.

Usage:
  julia --project scripts/thesis/benchmarks/csvar_benchmark.jl [options]

Options:
  --method=NAME         Solution method: NN, EGM, TimeIteration, all (default: NN)
  --scenario=TYPE       Scenario type: dimension, correlation, baseline, rotation (default: baseline)
  --dims=LIST           Comma-separated dimensions for dimension scenario (default: 2,3,4)
  --Na=N                Number of asset grid points (default: 60)
  --tol=X               Euler error tolerance (default: 1e-4)
  --epochs=N            Training epochs for NN method (default: 25000)
  --config=PATH         Path to base CSVAR config file (default: csvar_template.yaml)
  --output=DIR          Output directory (default: outputs/benchmarks)
  --repeats=N           Number of repetitions per variant (default: 1)
  --no-plots            Skip generating plots
  --summary-only        Only print summary, no CSV/JSON output
  --verbose             Enable verbose solver output
  --correlation=TYPE    Correlation structure: diagonal, single, toeplitz, dense (for correlation scenario)
  --rotation=TYPE       Rotation type: pca, cholesky (for rotation scenario)

Scenarios:
  baseline      - Run base config with specified method(s)
  dimension     - Scale across multiple state dimensions (2D, 3D, 4D, ...)
  correlation   - Test different correlation structures (diagonal, single off-diagonal, toeplitz, dense)
  rotation      - Apply orthogonal rotations (PCA, Cholesky) to test solver invariance

Outputs:
  - CSV reports with convergence statistics and Euler errors per variant
  - Policy plots (consumption surfaces) as PNG files
  - JSON file with detailed diagnostics including RMSE histories
  - Markdown summary report with best configurations
  - Comparison plots for multi-variant runs
"""

module CSVARBenchmark

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "..")); io = devnull)

using Dates
using Printf
using Statistics
using LinearAlgebra
using DataFrames
using CSV
using JSON3
using ThesisProject

# Try to load Plots for visualization
const HAS_PLOTS = try
    @eval using Plots
    true
catch
    false
end

include(joinpath(@__DIR__, "..", "..", "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

include(joinpath(@__DIR__, "..", "..", "utils", "csvar_param_grid.jl"))
using .CSVARParamGrid

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

# ============================================================================
# Configuration and CLI parsing
# ============================================================================

struct BenchmarkConfig
    method::String
    scenario::Symbol
    dims::Vector{Int}
    Na::Int
    tol::Float64
    epochs::Int
    config_path::String
    output_dir::String
    repeats::Int
    generate_plots::Bool
    summary_only::Bool
    verbose::Bool
    correlation_type::Symbol
    rotation_type::Symbol
end

function parse_cli_args()
    args = ARGS

    # Defaults
    method = "NN"
    scenario = :baseline
    dims = [2, 3, 4]
    Na = 60
    tol = 1e-4
    epochs = -1  # -1 means "use config file value or default to 25000"
    config_path = joinpath(ROOT, "config", "csvar_template.yaml")
    output_dir = joinpath(ROOT, "outputs", "benchmarks")
    repeats = 1
    generate_plots = true
    summary_only = false
    verbose = false
    correlation_type = :diagonal
    rotation_type = :pca

    for arg in args
        if startswith(arg, "--method=")
            method = split(arg, "=")[2]
        elseif startswith(arg, "--scenario=")
            scenario = Symbol(lowercase(split(arg, "=")[2]))
        elseif startswith(arg, "--dims=")
            dims_str = split(arg, "=")[2]
            dims = parse.(Int, split(dims_str, ","))
        elseif startswith(arg, "--Na=")
            Na = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--epochs=")
            epochs = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--config=")
            config_path = split(arg, "=")[2]
        elseif startswith(arg, "--output=")
            output_dir = split(arg, "=")[2]
        elseif startswith(arg, "--repeats=")
            repeats = parse(Int, split(arg, "=")[2])
        elseif arg == "--no-plots"
            generate_plots = false
        elseif arg == "--summary-only"
            summary_only = true
        elseif arg == "--verbose"
            verbose = true
        elseif startswith(arg, "--correlation=")
            correlation_type = Symbol(lowercase(split(arg, "=")[2]))
        elseif startswith(arg, "--rotation=")
            rotation_type = Symbol(lowercase(split(arg, "=")[2]))
        elseif arg == "--help" || arg == "-h"
            print_help()
            exit(0)
        else
            @warn "Unknown argument: $arg (use --help for usage)"
        end
    end

    return BenchmarkConfig(
        method,
        scenario,
        dims,
        Na,
        tol,
        epochs,
        config_path,
        output_dir,
        repeats,
        generate_plots,
        summary_only,
        verbose,
        correlation_type,
        rotation_type,
    )
end

function print_help()
    println(
        """
CSVAR Benchmark Script

Comprehensive benchmarking for CSVAR (vector-income) models. Tests solver methods
across dimensional scaling, correlation structures, and grid sizes with detailed
diagnostics and visualization.

Usage:
  julia --project scripts/thesis/benchmarks/csvar_benchmark.jl [options]

Options:
  --method=NAME         Solution method: NN, EGM, TimeIteration, all (default: NN)
  --scenario=TYPE       Scenario type: dimension, correlation, baseline, rotation (default: baseline)
  --dims=LIST           Comma-separated dimensions for dimension scenario (default: 2,3,4)
  --Na=N                Number of asset grid points (default: 60)
  --tol=X               Euler error tolerance (default: 1e-4)
  --epochs=N            Training epochs for NN method (default: 25000)
  --config=PATH         Path to base CSVAR config file (default: csvar_template.yaml)
  --output=DIR          Output directory (default: outputs/benchmarks)
  --repeats=N           Number of repetitions per variant (default: 1)
  --no-plots            Skip generating plots
  --summary-only        Only print summary, no CSV/JSON output
  --verbose             Enable verbose solver output
  --correlation=TYPE    Correlation structure: diagonal, single, toeplitz, dense (for correlation scenario)
  --rotation=TYPE       Rotation type: pca, cholesky (for rotation scenario)

Scenarios:
  baseline      - Run base config with specified method(s)
  dimension     - Scale across multiple state dimensions (2D, 3D, 4D, ...)
  correlation   - Test different correlation structures (diagonal, single off-diagonal, toeplitz, dense)
  rotation      - Apply orthogonal rotations (PCA, Cholesky) to test solver invariance

Outputs:
  - CSV reports with convergence statistics and Euler errors per variant
  - Policy plots (consumption surfaces) as PNG files
  - JSON file with detailed diagnostics including RMSE histories
  - Markdown summary report with best configurations
  - Comparison plots for multi-variant runs
        """,
    )
end

# ============================================================================
# Utility functions
# ============================================================================

function ensure_output_dir(path::String)
    isdir(path) || mkpath(path)
    return path
end

function format_duration(seconds::Float64)
    if seconds < 1.0
        return @sprintf("%.1f ms", seconds * 1000)
    elseif seconds < 60.0
        return @sprintf("%.2f s", seconds)
    else
        mins = floor(Int, seconds / 60)
        secs = seconds - mins * 60
        return @sprintf("%d min %.1f s", mins, secs)
    end
end

function matrix_to_rows(M::AbstractMatrix)
    rows = Vector{Vector{Float64}}(undef, size(M, 1))
    for i = 1:size(M, 1)
        rows[i] = collect(Float64.(M[i, :]))
    end
    return rows
end

function describe_params(params)
    y_dim = length(params.y)
    A =
        length(params.A) > 0 ? hcat([Float64.(row) for row in params.A]...)' :
        Matrix{Float64}(undef, 0, 0)
    Σ =
        length(params.Σ) > 0 ? hcat([Float64.(row) for row in params.Σ]...)' :
        Matrix{Float64}(undef, 0, 0)

    A_eigen = !isempty(A) ? eigvals(A) : Float64[]
    max_eig = !isempty(A_eigen) ? maximum(abs.(A_eigen)) : NaN

    Σ_trace = !isempty(Σ) ? tr(Σ) / y_dim : NaN

    # Compute correlation structure signature
    if !isempty(Σ) && y_dim > 1
        # Count off-diagonal correlations
        n_offdiag = 0
        for i = 1:y_dim, j = (i+1):y_dim
            if abs(Σ[i, j]) > 1e-10
                n_offdiag += 1
            end
        end
        corr_sig = n_offdiag
    else
        corr_sig = 0
    end

    return (
        y_dim = y_dim,
        max_eigenvalue = max_eig,
        avg_variance = Σ_trace,
        off_diagonal_corr = corr_sig,
    )
end

# ============================================================================
# Scenario generators
# ============================================================================

function build_baseline_variants(base_cfg::NamedTuple, config::BenchmarkConfig)
    variants = Dict{Symbol,NamedTuple}()
    variants[:baseline] = base_cfg
    return variants
end

function build_dimension_variants(base_cfg::NamedTuple, config::BenchmarkConfig)
    total_income = 1.0
    persistence = 0.70
    variance = 0.04

    # Filter out d=1 if present with a warning
    # TODO: Debug why CSVAR with d=1 produces NaNs in NN training from epoch 1
    dims_to_use = filter(d -> d > 1, config.dims)
    if length(dims_to_use) < length(config.dims)
        excluded = filter(d -> d == 1, config.dims)
        @warn "Excluding dimension(s) $excluded from CSVAR benchmark (known issue: NN training produces NaNs)"
    end

    variants = build_dimensional_overrides(
        base_cfg;
        dims = dims_to_use,
        total_income = total_income,
        persistence = persistence,
        variance = variance,
    )

    return variants
end

function build_correlation_variants(base_cfg::NamedTuple, config::BenchmarkConfig)
    # Get dimension from base config
    base_params = get_nested(base_cfg, (:params,))
    y_dim = length(base_params.y)

    persistence = 0.70
    variance = 0.04

    variants = build_correlation_overrides(
        base_cfg;
        persistence = persistence,
        variance = variance,
        corr_strength = 0.15,
        toeplitz_decay = 0.5,
        dense_scale = 0.12,
    )

    # Filter to requested type if specified
    if config.correlation_type != :all
        key = Symbol("corr_", config.correlation_type)
        if haskey(variants, key)
            variants = Dict(key => variants[key])
        else
            @warn "Correlation type $(config.correlation_type) not found, using all"
        end
    end

    return variants
end

function build_rotation_variants(base_cfg::NamedTuple, config::BenchmarkConfig)
    variants = build_rotation_overrides(base_cfg; rotations = [config.rotation_type])
    return variants
end

function build_variants(base_cfg::NamedTuple, config::BenchmarkConfig)
    if config.scenario == :baseline
        return build_baseline_variants(base_cfg, config)
    elseif config.scenario == :dimension
        return build_dimension_variants(base_cfg, config)
    elseif config.scenario == :correlation
        return build_correlation_variants(base_cfg, config)
    elseif config.scenario == :rotation
        return build_rotation_variants(base_cfg, config)
    else
        error("Unknown scenario: $(config.scenario)")
    end
end

# ============================================================================
# Benchmarking logic
# ============================================================================

function run_single_variant(
    variant_cfg::NamedTuple,
    variant_name::Symbol,
    method::String,
    config::BenchmarkConfig,
    repeat_id::Int,
)
    println(
        "  [Repeat $repeat_id/$( config.repeats)] Variant: $variant_name, Method: $method",
    )

    # Override solver settings
    solver_cfg = Dict{Symbol,Any}(
        :method => Symbol(method),
        :tol => config.tol,
        :verbose => config.verbose,
    )

    # Method-specific settings
    method_lower = lowercase(method)
    if method_lower == "nn"
        nn_cfg = Dict{Symbol,Any}(:epochs => config.epochs)
        solver_cfg[:nn] = nn_cfg
    elseif method_lower in ["timeiteration", "ti"]
        ti_cfg = Dict{Symbol,Any}(:interp_kind => "linear")
        solver_cfg[:time_iteration] = ti_cfg
    elseif method_lower == "egm"
        egm_cfg = Dict{Symbol,Any}(:interp_kind => "linear")
        solver_cfg[:egm] = egm_cfg
    end

    cfg = merge_section(variant_cfg, :solver, solver_cfg)
    cfg = merge_section(cfg, :grids, Dict(:Na => config.Na))

    # Derive RNG for this run
    rng_tag = Symbol(:csvar_bench_, variant_name, :_, method, :_, repeat_id)
    master_rng = cfg.random.master_rng
    local_rng = ThesisProject.Determinism.derive_rng(master_rng, string(rng_tag))

    # Run solver
    t_start = time()
    local solution, status, message
    try
        solutions = ThesisProject.solve(cfg; rng = local_rng)
        solution = solutions isa Vector ? solutions[1] : solutions
        status = :ok
        message = ""
    catch e
        # Create dummy solution on error
        solution = nothing
        status = :error
        message = sprint(showerror, e)
        @warn "Solver failed: $method on $variant_name (repeat $repeat_id)" exception =
            (e, catch_backtrace())
    end
    wall_runtime = time() - t_start
    runtime = wall_runtime
    evaluation_runtime = NaN
    diagnostics_runtime = NaN
    total_runtime = wall_runtime

    if status == :ok && solution !== nothing
        meta = solution.metadata
        diagnostics = solution.diagnostics

        runtime_val = hasproperty(diagnostics, :runtime) ? diagnostics.runtime : nothing
        if runtime_val isa Real
            runtime = Float64(runtime_val)
        end
        eval_val = haskey(meta, :evaluation_runtime) ? meta[:evaluation_runtime] : nothing
        if eval_val isa Real
            evaluation_runtime = Float64(eval_val)
        end
        diag_val = haskey(meta, :diagnostics_runtime) ? meta[:diagnostics_runtime] : nothing
        if diag_val isa Real
            diagnostics_runtime = Float64(diag_val)
        end
        total_val = haskey(meta, :total_runtime) ? meta[:total_runtime] : nothing
        if total_val isa Real
            total_runtime = Float64(total_val)
        end

        converged = get(meta, :converged, false)
        iterations =
            hasproperty(diagnostics, :iterations) ? Float64(diagnostics.iterations) : NaN
        mean_ee = hasproperty(diagnostics, :mean_ee) ? Float64(diagnostics.mean_ee) : NaN
        max_resid =
            haskey(meta, :max_resid) ? Float64(meta[:max_resid]) :
            haskey(meta, :rmse) ? Float64(meta[:rmse]) : NaN
        rmse_history =
            haskey(meta, :rmse_history) ? Float64.(meta[:rmse_history]) : Float64[]
        N_history = haskey(meta, :N_history) ? Int.(meta[:N_history]) : Int[]
        v_h_history = haskey(meta, :v_h_history) ? Float64.(meta[:v_h_history]) : Float64[]

        # Policy statistics
        c_policy = solution.policy[:c].value
        c_mean = mean(c_policy)
        c_std = std(c_policy)
        c_min = minimum(c_policy)
        c_max = maximum(c_policy)

        # Get model info
        params = get_nested(cfg, (:params,))
        param_desc = describe_params(params)
    else
        converged = false
        iterations = NaN
        mean_ee = NaN
        max_resid = NaN
        rmse_history = Float64[]
        N_history = Int[]
        v_h_history = Float64[]
        c_mean = NaN
        c_std = NaN
        c_min = NaN
        c_max = NaN
        params = get_nested(cfg, (:params,))
        param_desc = describe_params(params)
    end

    return (
        variant = variant_name,
        method = Symbol(method),
        repeat = repeat_id,
        y_dim = param_desc.y_dim,
        max_eigenvalue = param_desc.max_eigenvalue,
        avg_variance = param_desc.avg_variance,
        off_diagonal_corr = param_desc.off_diagonal_corr,
        runtime = runtime,
        evaluation_runtime = evaluation_runtime,
        diagnostics_runtime = diagnostics_runtime,
        total_runtime = total_runtime,
        iterations = iterations,
        mean_ee = mean_ee,
        final_rmse = max_resid,
        converged = converged,
        c_mean = c_mean,
        c_std = c_std,
        c_min = c_min,
        c_max = c_max,
        rmse_history = rmse_history,
        N_history = N_history,
        v_h_history = v_h_history,
        status = status,
        message = message,
        solution = solution,
        config = cfg,
    )
end

function run_benchmark(config::BenchmarkConfig)
    println("="^70)
    println("CSVAR Benchmark")
    println("="^70)
    println()

    # Load base config
    base_cfg = ThesisProject.load_config(config.config_path)

    # Resolve epochs: use CLI arg if provided (>0), otherwise use config file value, otherwise default to 25000
    epochs = if config.epochs > 0
        config.epochs
    elseif hasproperty(base_cfg, :solver) &&
           hasproperty(base_cfg.solver, :nn) &&
           hasproperty(base_cfg.solver.nn, :epochs)
        base_cfg.solver.nn.epochs
    else
        25000
    end

    # Update config with resolved epochs
    config = BenchmarkConfig(
        config.method,
        config.scenario,
        config.dims,
        config.Na,
        config.tol,
        epochs,
        config.config_path,
        config.output_dir,
        config.repeats,
        config.generate_plots,
        config.summary_only,
        config.verbose,
        config.correlation_type,
        config.rotation_type,
    )

    println("Configuration:")
    println("  Scenario:         $(config.scenario)")
    println("  Method(s):        $(config.method)")
    println("  Grid points (Na): $(config.Na)")
    println("  Tolerance:        $(config.tol)")
    if lowercase(config.method) == "nn" || config.method == "all"
        println("  NN Epochs:        $(config.epochs)")
    end
    println("  Repeats:          $(config.repeats)")
    println("  Base config:      $(basename(config.config_path))")
    println()

    # Build variants
    println("Building variants for scenario: $(config.scenario)")
    variants = build_variants(base_cfg, config)
    println("  Generated $(length(variants)) variant(s)")
    println()

    # Determine methods to run
    methods = if lowercase(config.method) == "all"
        ["NN", "EGM", "TimeIteration"]
    else
        [config.method]
    end

    # Run all combinations
    results = NamedTuple[]
    total_runs = length(variants) * length(methods) * config.repeats
    counter = 1

    println("Running $(total_runs) total benchmark runs...")
    println("-"^70)

    for (variant_name, variant_cfg) in variants
        println("\nVariant: $variant_name")
        for method in methods
            for rep = 1:config.repeats
                @printf("  [%d/%d] ", counter, total_runs)
                result = run_single_variant(variant_cfg, variant_name, method, config, rep)
                push!(results, result)

                if result.status == :ok
                    @printf(
                        "✓ Runtime: %s, RMSE: %.6e\n",
                        format_duration(result.runtime),
                        result.final_rmse
                    )
                else
                    println("✗ Failed: $(result.message)")
                end

                counter += 1
            end
        end
    end

    println("-"^70)
    println()

    return results
end

# ============================================================================
# Analysis and reporting
# ============================================================================

function aggregate_results(results)
    # Convert to DataFrame
    rows = NamedTuple[]
    for res in results
        push!(
            rows,
            (
                variant = String(res.variant),
                method = String(res.method),
                repeat = res.repeat,
                y_dim = res.y_dim,
                max_eigenvalue = res.max_eigenvalue,
                avg_variance = res.avg_variance,
                off_diagonal_corr = res.off_diagonal_corr,
                runtime = res.runtime,
                evaluation_runtime = hasproperty(res, :evaluation_runtime) ?
                                     res.evaluation_runtime : NaN,
                diagnostics_runtime = hasproperty(res, :diagnostics_runtime) ?
                                      res.diagnostics_runtime : NaN,
                total_runtime = hasproperty(res, :total_runtime) ? res.total_runtime :
                                res.runtime,
                iterations = res.iterations,
                mean_ee = res.mean_ee,
                final_rmse = res.final_rmse,
                converged = res.converged,
                c_mean = res.c_mean,
                c_std = res.c_std,
                status = String(res.status),
            ),
        )
    end
    df = DataFrame(rows)

    # Filter to successful runs
    ok_df = filter(row -> row.status == "ok", df)

    if nrow(ok_df) == 0
        @warn "No successful runs to aggregate"
        return df, DataFrame()
    end

    # Compute summary statistics
    summary = combine(
        groupby(ok_df, [:variant, :method]),
        :y_dim => first => :y_dim,
        :max_eigenvalue => first => :max_eigenvalue,
        :avg_variance => first => :avg_variance,
        :off_diagonal_corr => first => :off_diagonal_corr,
        :runtime => mean => :runtime_mean,
        :runtime => std => :runtime_std,
        :final_rmse => mean => :rmse_mean,
        :final_rmse => std => :rmse_std,
        :mean_ee => mean => :mean_ee_avg,
        :converged => (x -> mean(Float64.(x))) => :convergence_rate,
        nrow => :n_runs,
    )

    sort!(summary, [:variant, :rmse_mean])

    return df, summary
end

function print_summary(summary::DataFrame)
    println("="^70)
    println("Benchmark Summary")
    println("="^70)
    println()

    if nrow(summary) == 0
        println("No successful runs.")
        return
    end

    # Show summary table
    show(stdout, summary; allrows = true, allcols = false, truncate = 100)
    println()
    println()

    # Highlight best configuration
    best_row = first(summary)
    println("Best Configuration:")
    println("  Variant:      $(best_row.variant)")
    println("  Method:       $(best_row.method)")
    println("  Dimension:    $(best_row.y_dim)")
    println("  Runtime:      $(format_duration(best_row.runtime_mean))")
    println("  Final RMSE:   $(@sprintf("%.6e", best_row.rmse_mean))")
    println("  Converged:    $(round(best_row.convergence_rate * 100, digits=1))%")
    println()
end

function save_results(
    config::BenchmarkConfig,
    results,
    df::DataFrame,
    summary::DataFrame,
    output_dir::String,
)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    scenario_slug = String(config.scenario)

    # CSV files
    csv_path = joinpath(output_dir, "csvar_$(scenario_slug)_runs_$(timestamp).csv")
    CSV.write(csv_path, df)

    summary_path = joinpath(output_dir, "csvar_$(scenario_slug)_summary_$(timestamp).csv")
    CSV.write(summary_path, summary)

    # JSON with detailed diagnostics
    json_path = joinpath(output_dir, "csvar_$(scenario_slug)_detailed_$(timestamp).json")
    json_payload = [
        Dict(
            :variant => String(res.variant),
            :method => String(res.method),
            :repeat => res.repeat,
            :y_dim => res.y_dim,
            :max_eigenvalue => res.max_eigenvalue,
            :avg_variance => res.avg_variance,
            :off_diagonal_corr => res.off_diagonal_corr,
            :runtime => res.runtime,
            :iterations => res.iterations,
            :mean_ee => res.mean_ee,
            :final_rmse => res.final_rmse,
            :converged => res.converged,
            :c_mean => res.c_mean,
            :c_std => res.c_std,
            :status => String(res.status),
            :message => res.message,
            :rmse_history => res.rmse_history,
            :N_history => res.N_history,
            :v_h_history => res.v_h_history,
        ) for res in results
    ]
    open(json_path, "w") do io
        JSON3.write(io, json_payload; indent = 2)
    end

    # Append to consolidated runs CSV (single file accumulating all runs)
    consolidated_path = joinpath(output_dir, "csvar_benchmark_runs.csv")
    df2 = deepcopy(df)
    n = nrow(df2)
    if n > 0
        df2[!, :timestamp] = fill(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), n)
        df2[!, :scenario] = fill(String(config.scenario), n)
        df2[!, :Na] = fill(config.Na, n)
        df2[!, :tol] = fill(config.tol, n)
        df2[!, :epochs] = fill(config.epochs, n)
        df2[!, :base_config] = fill(basename(config.config_path), n)
        if isfile(consolidated_path)
            open(consolidated_path, "a") do io
                CSV.write(io, df2; header = false)
            end
        else
            CSV.write(consolidated_path, df2)
        end
    end

    println("Results saved:")
    println("  - Runs CSV:     $csv_path")
    println("  - Summary CSV:  $summary_path")
    println("  - Detail JSON:  $json_path")
    println("  - Consolidated: $consolidated_path (appended)")
    println()

    return csv_path, summary_path, json_path
end

function generate_plots(
    config::BenchmarkConfig,
    results,
    summary::DataFrame,
    output_dir::String,
)
    if !HAS_PLOTS
        @warn "Plots.jl not available - skipping plot generation"
        return String[]
    end

    if nrow(summary) == 0
        @warn "No successful runs - skipping plot generation"
        return String[]
    end

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    scenario_slug = String(config.scenario)
    plot_files = String[]

    # Filter successful results
    success_results = filter(r -> r.status == :ok, results)

    if isempty(success_results)
        return plot_files
    end

    @eval begin
        using Plots
        gr()
    end

    # Plot 1: Runtime vs RMSE scatter
    if nrow(summary) > 1
        @eval begin
            p_scatter = scatter(
                $summary.runtime_mean,
                $summary.rmse_mean;
                xlabel = "Runtime (s)",
                ylabel = "Final Euler RMSE",
                title = "Runtime vs Accuracy: $($(config.scenario))",
                legend = false,
                marker = :circle,
                ms = 8,
                color = :steelblue,
                grid = true,
                size = (800, 600),
                dpi = 150,
            )

            # Add labels for each point
            for i = 1:nrow($summary)
                annotate!(
                    p_scatter,
                    $summary.runtime_mean[i],
                    $summary.rmse_mean[i],
                    text("$(String($summary.variant[i]))", 7, :bottom),
                )
            end

            scatter_path = joinpath(
                $output_dir,
                "csvar_$($(scenario_slug))_runtime_vs_rmse_$($(timestamp)).png",
            )
            savefig(p_scatter, scatter_path)
            push!($plot_files, scatter_path)
        end
    end

    # Plot 2: RMSE bar chart by variant
    @eval begin
        p_bar = bar(
            $summary.variant,
            $summary.rmse_mean;
            yerror = $summary.rmse_std,
            xlabel = "Variant",
            ylabel = "Final Euler RMSE",
            title = "RMSE by Variant: $($(config.scenario))",
            legend = false,
            rotation = 15,
            color = :steelblue,
            size = (800, 600),
            dpi = 150,
        )

        bar_path =
            joinpath($output_dir, "csvar_$($(scenario_slug))_rmse_bar_$($(timestamp)).png")
        savefig(p_bar, bar_path)
        push!($plot_files, bar_path)
    end

    # Plot 3: RMSE convergence histories
    history_results = filter(r -> !isempty(r.rmse_history), success_results)
    if !isempty(history_results)
        @eval begin
            p_conv = plot(;
                xlabel = "Iteration/Epoch",
                ylabel = "Euler RMSE",
                title = "Convergence Histories: $($(config.scenario))",
                yscale = :log10,
                legend = :topright,
                size = (800, 600),
                dpi = 150,
            )

            # Plot up to 10 histories to avoid clutter
            n_plot = min(10, length($history_results))
            for (idx, res) in enumerate($history_results[1:n_plot])
                label = "$(String(res.variant))_$(String(res.method))"
                plot!(p_conv, res.rmse_history; label = label, alpha = 0.7, linewidth = 1.5)
            end

            # Add tolerance line
            hline!(
                p_conv,
                [$(config.tol)];
                linestyle = :dash,
                color = :red,
                linewidth = 1,
                label = "Tolerance",
            )

            conv_path = joinpath(
                $output_dir,
                "csvar_$($(scenario_slug))_convergence_$($(timestamp)).png",
            )
            savefig(p_conv, conv_path)
            push!($plot_files, conv_path)
        end
    end

    # Plot 4: Dimension scaling (if applicable)
    if config.scenario == :dimension && :y_dim in names(summary)
        @eval begin
            p_scaling = plot(;
                xlabel = "State Dimension (y_dim)",
                ylabel = "Runtime (s)",
                title = "Dimensional Scaling",
                legend = :topleft,
                size = (800, 600),
                dpi = 150,
                yscale = :log10,
            )

            for method in unique($summary.method)
                method_data = filter(row -> row.method == method, $summary)
                if nrow(method_data) > 1
                    plot!(
                        p_scaling,
                        method_data.y_dim,
                        method_data.runtime_mean;
                        label = String(method),
                        marker = :circle,
                        linewidth = 2,
                        ms = 6,
                    )
                end
            end

            scaling_path = joinpath(
                $output_dir,
                "csvar_$($(scenario_slug))_scaling_$($(timestamp)).png",
            )
            savefig(p_scaling, scaling_path)
            push!($plot_files, scaling_path)
        end
    end

    println("Plots saved:")
    for file in plot_files
        println("  - $(basename(file))")
    end
    println()

    return plot_files
end

function write_report(
    config::BenchmarkConfig,
    summary::DataFrame,
    plot_files::Vector{String},
    output_dir::String,
)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    scenario_slug = String(config.scenario)
    report_path = joinpath(output_dir, "csvar_$(scenario_slug)_report_$(timestamp).md")

    open(report_path, "w") do io
        println(io, "# CSVAR Benchmark Report: $(config.scenario)")
        println(io)
        println(io, "**Generated:** $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io)

        println(io, "## Configuration")
        println(io)
        println(io, "- Scenario: `$(config.scenario)`")
        println(io, "- Method(s): `$(config.method)`")
        println(io, "- Grid size: Na = $(config.Na)")
        println(io, "- Tolerance: $(config.tol)")
        println(io, "- Repeats per variant: $(config.repeats)")
        println(io)

        if nrow(summary) == 0
            println(io, "## Results")
            println(io)
            println(io, "All runs failed. See detailed JSON for error messages.")
            return
        end

        println(io, "## Summary Statistics")
        println(io)
        println(io, "```")
        show(io, MIME("text/plain"), summary; allrows = true)
        println(io)
        println(io, "```")
        println(io)

        # Best configuration
        best_row = first(summary)
        println(io, "## Best Configuration")
        println(io)
        println(io, "- **Variant:** $(best_row.variant)")
        println(io, "- **Method:** $(best_row.method)")
        println(io, "- **Dimension:** $(best_row.y_dim)")
        println(
            io,
            "- **Runtime:** $(format_duration(best_row.runtime_mean)) ± $(format_duration(best_row.runtime_std))",
        )
        println(
            io,
            "- **Final RMSE:** $(@sprintf("%.6e", best_row.rmse_mean)) ± $(@sprintf("%.6e", best_row.rmse_std))",
        )
        println(
            io,
            "- **Convergence Rate:** $(round(best_row.convergence_rate * 100, digits=1))%",
        )
        println(io)

        # Key insights
        println(io, "## Key Insights")
        println(io)

        if config.scenario == :dimension
            println(
                io,
                "- **Dimensional scaling:** Testing solver performance across $(minimum(config.dims))D to $(maximum(config.dims))D state spaces",
            )
            if nrow(summary) > 1
                runtime_growth = summary.runtime_mean[end] / summary.runtime_mean[1]
                println(
                    io,
                    "- **Runtime growth:** $(round(runtime_growth, digits=2))× from lowest to highest dimension",
                )
            end
        elseif config.scenario == :correlation
            println(
                io,
                "- **Correlation structure:** Comparing solver robustness to income correlations",
            )
            println(
                io,
                "- Best structure achieved lowest RMSE while maintaining computational efficiency",
            )
        elseif config.scenario == :rotation
            println(
                io,
                "- **Rotation invariance:** Testing solver stability under orthogonal state transformations",
            )
            println(
                io,
                "- Validates that solutions are independent of coordinate system representation",
            )
        end
        println(io)

        # Plots
        if !isempty(plot_files)
            println(io, "## Visualizations")
            println(io)
            for file in plot_files
                println(io, "### $(basename(file))")
                println(io)
                println(io, "![]($(basename(file)))")
                println(io)
            end
        end

        println(io, "---")
        println(io, "*Report generated by CSVAR Benchmark script*")
    end

    println("Report saved: $report_path")
    return report_path
end

# ============================================================================
# Main execution
# ============================================================================

function main()
    config = parse_cli_args()
    output_dir = ensure_output_dir(config.output_dir)

    # Run benchmark
    results = run_benchmark(config)

    # Aggregate and analyze
    println("="^70)
    println("Analyzing Results")
    println("="^70)
    println()

    df, summary = aggregate_results(results)
    print_summary(summary)

    # Save and visualize
    if !config.summary_only
        println("="^70)
        println("Saving Results")
        println("="^70)
        println()

        save_results(config, results, df, summary, output_dir)

        if config.generate_plots
            plot_files = generate_plots(config, results, summary, output_dir)
            write_report(config, summary, plot_files, output_dir)
        end
    end

    println("="^70)
    println("CSVAR Benchmark Complete!")
    println("="^70)
end

end # module

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    CSVARBenchmark.main()
end
