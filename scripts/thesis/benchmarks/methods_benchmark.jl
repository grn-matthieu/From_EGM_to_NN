#!/usr/bin/env julia

"""
Methods Benchmark Script

Runs any solution method on the stochastic consumption-savings model with
comprehensive diagnostics, statistics, and policy plots.

Usage:
  julia --project scripts/thesis/benchmarks/methods_benchmark.jl [options]

Options:
  --method=NAME       Solution method: TimeIteration, EGM, Projection, NN, Perturbation, all (default: TimeIteration)
  --Na=N              Number of asset grid points (default: 100)
  --Nz=N              Number of income shock states (default: 5)
  --tol=X             Euler error tolerance (default: 1e-4)
  --tol_pol=X         Policy convergence tolerance (default: 1e-6)
  --maxit=N           Maximum iterations for classical methods (default: 1000)
  --epochs=N          Training epochs for NN method (default: 1000)
  --interp=KIND       Interpolation kind: linear or pchip (default: linear, for TimeIteration)
  --config=PATH       Path to base config file (default: smoke_cfg_stoch.yaml)
  --output=DIR        Output directory (default: outputs/benchmarks)
  --no-plots          Skip generating plots
  --verbose           Enable verbose solver output

Outputs:
  - CSV report with convergence statistics and Euler errors
  - Policy plots (consumption and savings) as PNG files
  - Summary statistics printed to console
"""

module MethodsBenchmark

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "..")); io = devnull)

using Dates
using Printf
using Statistics
using ThesisProject
using ThesisProject.CommonInterp: interp_linear!, interp_pchip!

# Try to load Plots for visualization
const HAS_PLOTS = try
    @eval using Plots
    true
catch
    false
end

include(joinpath(@__DIR__, "..", "..", "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

# ============================================================================
# Configuration and CLI parsing
# ============================================================================

struct BenchmarkConfig
    method::String
    Na::Int
    Nz::Int
    tol::Float64
    tol_pol::Float64
    maxit::Int
    epochs::Int
    interp_kind::Symbol
    config_path::String
    output_dir::String
    generate_plots::Bool
    verbose::Bool
end

function parse_cli_args()
    args = ARGS

    # Defaults
    method = "TimeIteration"
    Na = 100
    Nz = 7
    tol = 1e-3
    tol_pol = 1e-4
    maxit = 1000
    epochs = 50000
    interp_kind = :linear
    config_path = joinpath(ROOT, "config", "smoke_cfg_stoch.yaml")
    output_dir = joinpath(ROOT, "outputs", "benchmarks")
    generate_plots = true
    verbose = false

    for arg in args
        if startswith(arg, "--method=")
            method = split(arg, "=")[2]
        elseif startswith(arg, "--Na=")
            Na = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--Nz=")
            Nz = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--tol_pol=")
            tol_pol = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--maxit=")
            maxit = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--epochs=")
            epochs = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--interp=")
            kind_str = lowercase(split(arg, "=")[2])
            interp_kind = kind_str == "pchip" ? :pchip : :linear
        elseif startswith(arg, "--config=")
            config_path = split(arg, "=")[2]
        elseif startswith(arg, "--output=")
            output_dir = split(arg, "=")[2]
        elseif arg == "--no-plots"
            generate_plots = false
        elseif arg == "--verbose"
            verbose = true
        elseif arg == "--help" || arg == "-h"
            print_help()
            exit(0)
        else
            @warn "Unknown argument: $arg (use --help for usage)"
        end
    end

    return BenchmarkConfig(
        method,
        Na,
        Nz,
        tol,
        tol_pol,
        maxit,
        epochs,
        interp_kind,
        config_path,
        output_dir,
        generate_plots,
        verbose,
    )
end

function print_help()
    println(
        """
Methods Benchmark Script

Runs any solution method on the stochastic consumption-savings model with
comprehensive diagnostics, statistics, and policy plots.

Usage:
  julia --project scripts/thesis/benchmarks/methods_benchmark.jl [options]

Options:
  --method=NAME       Solution method: TimeIteration, EGM, Projection, NN, Perturbation, all (default: TimeIteration)
  --Na=N              Number of asset grid points (default: 100)
  --Nz=N              Number of income shock states (default: 5)
  --tol=X             Euler error tolerance (default: 1e-4)
  --tol_pol=X         Policy convergence tolerance (default: 1e-6)
  --maxit=N           Maximum iterations for classical methods (default: 1000)
  --epochs=N          Training epochs for NN method (default: 1000)
  --interp=KIND       Interpolation kind: linear or pchip (default: linear, for TimeIteration)
  --config=PATH       Path to base config file (default: smoke_cfg_stoch.yaml)
  --output=DIR        Output directory (default: outputs/benchmarks)
  --no-plots          Skip generating plots
  --verbose           Enable verbose solver output

Outputs:
  - CSV report with convergence statistics and Euler errors
  - Policy plots (consumption and savings) as PNG files
  - Summary statistics printed to console
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

@inline function _getparam(params, name, default = 0.0)
    return hasproperty(params, name) ? Float64(getfield(params, name)) : default
end

function expected_income_level(params, shocks_info)
    if shocks_info === nothing
        if hasproperty(params, :y) && params.y isa AbstractVector
            vals = Float64.(collect(params.y))
            return mean(vals)
        else
            return _getparam(params, :y)
        end
    end
    π = Float64.(getproperty(shocks_info, :π))
    if hasproperty(params, :y) && !(params.y isa AbstractVector)
        μ = Float64(getfield(params, :y))
        if hasproperty(shocks_info, :zgrid)
            z = Float64.(getproperty(shocks_info, :zgrid))
            income_states = exp.(μ .+ z)
            return sum(π .* income_states)
        else
            return exp(μ)
        end
    elseif hasproperty(params, :y) && params.y isa AbstractVector
        vals = Float64.(collect(params.y))
        return sum(π .* vals)
    else
        return sum(π)
    end
end

function expected_cash_on_hand_grid(a_grid, params, shocks_info)
    Rg = 1.0 + _getparam(params, :r)
    income_mean = expected_income_level(params, shocks_info)
    return Rg .* Float64.(a_grid) .+ income_mean
end

# ============================================================================
# Benchmarking logic
# ============================================================================

function run_method_benchmark(config::BenchmarkConfig)
    println("="^70)
    println("$(config.method) Benchmark")
    println("="^70)
    println()

    # Load and configure
    println("Loading configuration from: $(config.config_path)")
    base_cfg = ThesisProject.load_config(config.config_path)

    # Override with CLI parameters
    solver_cfg = Dict{Symbol,Any}(
        :method => config.method,
        :tol => config.tol,
        :tol_pol => config.tol_pol,
        :maxit => config.maxit,
        :verbose => config.verbose,
    )

    # Set method-specific parameters
    method_lower = lowercase(config.method)
    if method_lower in ["timeiteration", "ti"]
        # Set interpolation kind for Time Iteration
        ti_cfg = Dict{Symbol,Any}(:interp_kind => string(config.interp_kind))
        solver_cfg[:time_iteration] = ti_cfg
    elseif method_lower == "nn"
        # Set epochs for NN solver (uses epochs instead of maxit)
        nn_cfg = Dict{Symbol,Any}(:epochs => config.epochs)
        solver_cfg[:nn] = nn_cfg
    end

    cfg = merge_section(base_cfg, :solver, solver_cfg)
    cfg = merge_section(cfg, :grids, Dict(:Na => config.Na))

    # Set number of shock states
    if haskey(cfg, :shocks) && get(cfg[:shocks], :active, false)
        # For Tauchen discretization, Nz directly controls the number of states
        # The m parameter controls the grid width (number of std deviations)
        cfg = merge_section(cfg, :shocks, Dict(:Nz => config.Nz))
    end

    println("\nConfiguration:")
    println("  Method:                  $(config.method)")
    println("  Grid points (Na):        $(config.Na)")
    println("  Shock states (Nz):       $(config.Nz)")
    println("  Euler tolerance (εEE):   $(config.tol)")
    println("  Policy tolerance (εpol): $(config.tol_pol)")
    println("  Max iterations:          $(config.maxit)")
    if lowercase(config.method) in ["timeiteration", "ti"]
        println("  Interpolation:           $(config.interp_kind)")
    end
    println()

    # Build model
    println("Building model...")
    model = ThesisProject.build_model(cfg)
    params = ThesisProject.get_params(model)
    grids_info = ThesisProject.get_grids(model)
    shocks_info = ThesisProject.get_shocks(model)

    println("  β = $(params.β)")
    println("  γ = $(params.γ)")
    println("  r = $(params.r)")
    if shocks_info !== nothing
        println("  Actual Nz = $(length(shocks_info.zgrid))")
        println(
            "  Income states (y): ",
            round.(exp.(shocks_info.zgrid) .* params.y, digits = 4),
        )
    end
    println()

    # Solve with timing
    println("Running $(config.method) solver...")
    println("-"^70)

    t_start = time()
    solutions = ThesisProject.solve(model, cfg)
    # solve(model, cfg) returns a Vector{Solution}, get the first one
    solution = solutions[1]
    runtime = time() - t_start

    println("-"^70)
    println()

    return solution, runtime, model, grids_info, shocks_info
end

function compute_statistics(solution, runtime, grids_info, shocks_info)
    println("="^70)
    println("Results Summary")
    println("="^70)
    println()

    meta = solution.metadata

    # Convergence info
    converged = get(meta, :converged, false)
    iters = get(meta, :iters, 0)

    println("Convergence:")
    println("  Status:       $(converged ? "✓ Converged" : "✗ Not converged")")
    println("  Iterations:   $iters")
    println("  Runtime:      $(format_duration(runtime))")

    # Euler errors
    euler_rmse = get(meta, :rmse, get(meta, :max_resid, NaN))
    println()
    println("Euler Errors (RMSE):")
    println("  Final RMSE:   $(@sprintf("%.6e", euler_rmse))")

    # Policy statistics
    c_policy = solution.policy[:c].value
    a_policy = solution.policy[:a].value

    println()
    println("Policy Statistics:")
    if c_policy isa AbstractMatrix
        println("  Consumption shape:  $(size(c_policy))")
        println("    Mean:             $(@sprintf("%.4f", mean(c_policy)))")
        println("    Std:              $(@sprintf("%.4f", std(c_policy)))")
        println("    Min:              $(@sprintf("%.4f", minimum(c_policy)))")
        println("    Max:              $(@sprintf("%.4f", maximum(c_policy)))")
    else
        println("  Consumption shape:  $(length(c_policy))")
        println("    Mean:             $(@sprintf("%.4f", mean(c_policy)))")
        println("    Std:              $(@sprintf("%.4f", std(c_policy)))")
        println("    Min:              $(@sprintf("%.4f", minimum(c_policy)))")
        println("    Max:              $(@sprintf("%.4f", maximum(c_policy)))")
    end

    println()
    if a_policy isa AbstractMatrix
        println("  Savings shape:      $(size(a_policy))")
        println("    Mean:             $(@sprintf("%.4f", mean(a_policy)))")
        println("    Std:              $(@sprintf("%.4f", std(a_policy)))")
        println("    Min:              $(@sprintf("%.4f", minimum(a_policy)))")
        println("    Max:              $(@sprintf("%.4f", maximum(a_policy)))")
    else
        println("  Savings shape:      $(length(a_policy))")
        println("    Mean:             $(@sprintf("%.4f", mean(a_policy)))")
        println("    Std:              $(@sprintf("%.4f", std(a_policy)))")
        println("    Min:              $(@sprintf("%.4f", minimum(a_policy)))")
        println("    Max:              $(@sprintf("%.4f", maximum(a_policy)))")
    end

    # Binding constraint
    a_grid = grids_info.a.grid
    a_min = grids_info.a.min
    binding_tol = 1e-6

    if a_policy isa AbstractMatrix
        binding_count = sum(abs.(a_policy .- a_min) .< binding_tol)
        total_count = length(a_policy)
        binding_share = binding_count / total_count
    else
        binding_count = sum(abs.(a_policy .- a_min) .< binding_tol)
        total_count = length(a_policy)
        binding_share = binding_count / total_count
    end

    println()
    println("Binding Constraint:")
    println("  Binding points:   $binding_count / $total_count")
    println("  Binding share:    $(@sprintf("%.2f%%", binding_share * 100))")
    println()

    return (
        converged = converged,
        iters = iters,
        runtime = runtime,
        euler_rmse = euler_rmse,
        c_mean = mean(c_policy),
        c_std = std(c_policy),
        a_mean = mean(a_policy),
        a_std = std(a_policy),
        binding_share = binding_share,
    )
end

function save_report(config::BenchmarkConfig, stats, output_dir::String)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    method_slug =
        lowercase(replace(config.method, "TimeIteration" => "ti", "Perturbation" => "pert"))
    csv_path = joinpath(output_dir, "$(method_slug)_benchmark_$(timestamp).csv")

    # Write CSV report
    open(csv_path, "w") do io
        println(io, "metric,value")
        println(io, "timestamp,$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "method,$(config.method)")
        println(io, "Na,$(config.Na)")
        println(io, "Nz,$(config.Nz)")
        println(io, "tol,$(config.tol)")
        println(io, "tol_pol,$(config.tol_pol)")
        println(io, "maxit,$(config.maxit)")
        if lowercase(config.method) in ["timeiteration", "ti"]
            println(io, "interp_kind,$(config.interp_kind)")
        end
        println(io, "converged,$(stats.converged)")
        println(io, "iterations,$(stats.iters)")
        println(io, "runtime_seconds,$(stats.runtime)")
        println(io, "euler_rmse,$(stats.euler_rmse)")
        println(io, "consumption_mean,$(stats.c_mean)")
        println(io, "consumption_std,$(stats.c_std)")
        println(io, "savings_mean,$(stats.a_mean)")
        println(io, "savings_std,$(stats.a_std)")
        println(io, "binding_share,$(stats.binding_share)")
    end

    println("Saved report to: $csv_path")
    return csv_path
end

function generate_plots(
    solution,
    grids_info,
    shocks_info,
    params,
    config::BenchmarkConfig,
    output_dir::String,
)
    if !HAS_PLOTS
        @warn "Plots.jl not available - skipping plot generation"
        return String[]
    end

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    method_slug =
        lowercase(replace(config.method, "TimeIteration" => "ti", "Perturbation" => "pert"))
    plot_files = String[]

    a_grid = grids_info.a.grid
    c_policy = solution.policy[:c].value
    a_policy = solution.policy[:a].value

    # Determine if stochastic
    is_stochastic = c_policy isa AbstractMatrix && size(c_policy, 2) > 1

    if is_stochastic && shocks_info !== nothing
        # Stochastic case - compute and plot expected policies
        π = shocks_info.π  # stationary distribution
        Nz = length(π)

        # For each asset level, compute expected consumption and savings
        # weighted by stationary distribution across income states
        Na = length(a_grid)
        expected_c = zeros(Na)
        expected_a = zeros(Na)

        for i = 1:Na
            for j = 1:Nz
                expected_c[i] += π[j] * c_policy[i, j]
                expected_a[i] += π[j] * a_policy[i, j]
            end
        end

        # Compute expected cash-on-hand (consistent with NN evaluation grid)
        expected_x = expected_cash_on_hand_grid(a_grid, params, shocks_info)

        # Extrapolate to low cash-on-hand values [0, min(expected_x)]
        # Theoretical foundation: At constraint with a=0, c=x and a'=0
        # As x increases from 0, policy transitions from hand-to-mouth to interior solution
        x_min = minimum(expected_x)
        n_extrap = 50  # number of extrapolation points
        x_extrap = range(0.0, x_min, length = n_extrap)

        # For consumption: enforce c(0)=0, and linearly interpolate to c(x_min)
        # This respects the theoretical boundary condition c ≤ x
        c_extrap = [x * (expected_c[1] / expected_x[1]) for x in x_extrap]

        # For savings: enforce a'(0)=0, and linearly interpolate to a'(x_min)
        # At low x, hand-to-mouth behavior means a' ≈ 0
        a_extrap = [x * (expected_a[1] / expected_x[1]) for x in x_extrap]

        # Combine extrapolated and computed values
        full_x = vcat(x_extrap, expected_x)
        full_c = vcat(c_extrap, expected_c)
        full_a = vcat(a_extrap, expected_a)

        # Consumption policy plot
        @eval begin
            using Plots
            gr()

            p_c = plot(
                $full_x,
                $full_c,
                xlabel = "Cash-on-hand (x = R*a + E[y])",
                ylabel = "Expected Consumption E[c]",
                title = "$($(config.method)): Expected Consumption Policy",
                legend = false,
                size = (800, 600),
                dpi = 150,
                color = :black,
                linewidth = 1.0,
                grid = true,
                gridstyle = :solid,
                gridalpha = 0.3,
                gridlinewidth = 0.5,
                framestyle = :box,
                xlims = (0.0, 3.5),
                ylims = (0.0, 1.75),
                xticks = 0.0:0.25:3.5,
                yticks = 0.0:0.25:1.75,
            )

            # Add 45-degree line reference
            plot!(
                p_c,
                [0.0, 1.75],
                [0.0, 1.75],
                linestyle = :dash,
                color = :gray,
                linewidth = 1,
                label = nothing,
            )

            c_path =
                joinpath($output_dir, "$($(method_slug))_consumption_$($(timestamp)).png")
            savefig(p_c, c_path)
            push!($plot_files, c_path)
        end

    else
        # Deterministic case - single policy line
        c_vec = c_policy isa AbstractVector ? c_policy : vec(c_policy)

        # Consumption policy plot
        @eval begin
            using Plots
            p_c = plot(
                $a_grid,
                $c_vec,
                xlabel = "Asset holdings (a)",
                ylabel = "Consumption (c)",
                title = "$($(config.method)): Consumption Policy",
                label = "Policy function",
                linewidth = 2,
                size = (800, 600),
                dpi = 150,
            )

            c_path =
                joinpath($output_dir, "$($(method_slug))_consumption_$($(timestamp)).png")
            savefig(p_c, c_path)
            push!($plot_files, c_path)
        end
    end

    # Convergence plot (common for both stochastic and deterministic)
    if haskey(solution.metadata, :rmse_history)
        rmse_hist = solution.metadata[:rmse_history]
        if !isempty(rmse_hist)
            @eval begin
                using Plots
                gr()

                # Filter out any Inf or invalid values and take log10
                valid_idx = findall(x -> isfinite(x) && x > 0, $rmse_hist)
                if !isempty(valid_idx)
                    iters_plot = collect(1:length($rmse_hist))[valid_idx]
                    log_rmse = log10.($rmse_hist[valid_idx])

                    p_conv = plot(
                        iters_plot,
                        log_rmse,
                        xlabel = "Iteration (log scale)",
                        ylabel = "log₁₀(RMSE)",
                        title = "$($(config.method)): Convergence History",
                        legend = false,
                        size = (800, 600),
                        dpi = 150,
                        color = :black,
                        linewidth = 1.5,
                        grid = true,
                        gridstyle = :solid,
                        gridalpha = 0.3,
                        gridlinewidth = 0.5,
                        framestyle = :box,
                        marker = :circle,
                        markersize = 2,
                        markeralpha = 0.6,
                        xscale = :log10,
                    )

                    # Add tolerance line if available
                    tol_val = get(get($solution.metadata, :opts, (;)), :tol, nothing)
                    if tol_val !== nothing && isfinite(tol_val) && tol_val > 0
                        hline!(
                            p_conv,
                            [log10(tol_val)],
                            linestyle = :dash,
                            color = :red,
                            linewidth = 1,
                            label = "Tolerance",
                        )
                    end

                    conv_path = joinpath(
                        $output_dir,
                        "$($(method_slug))_convergence_$($(timestamp)).png",
                    )
                    savefig(p_conv, conv_path)
                    push!($plot_files, conv_path)
                end
            end
        end
    end

    println()
    println("Saved plots:")
    for file in plot_files
        println("  - $file")
    end

    return plot_files
end

# ============================================================================
# Main execution
# ============================================================================

function save_comparison_report(
    config::BenchmarkConfig,
    all_stats::Dict{String,NamedTuple},
    output_dir::String,
)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    csv_path = joinpath(output_dir, "all_methods_comparison_$(timestamp).csv")

    # Get all metrics from first method
    methods = sort(collect(keys(all_stats)))
    first_stats = all_stats[methods[1]]
    metrics = collect(keys(first_stats))

    # Write CSV report with methods as columns
    open(csv_path, "w") do io
        # Header: metric, method1, method2, method3, ...
        print(io, "metric")
        for method in methods
            print(io, ",", method)
        end
        println(io)

        # Metadata rows
        println(
            io,
            "timestamp,",
            join([Dates.format(now(), "yyyy-mm-dd HH:MM:SS") for _ in methods], ","),
        )
        println(io, "Na,", join([config.Na for _ in methods], ","))
        println(io, "Nz,", join([config.Nz for _ in methods], ","))
        println(io, "tol,", join([config.tol for _ in methods], ","))
        println(io, "tol_pol,", join([config.tol_pol for _ in methods], ","))
        println(io, "maxit,", join([config.maxit for _ in methods], ","))

        # Data rows
        for metric in metrics
            print(io, metric)
            for method in methods
                print(io, ",", all_stats[method][metric])
            end
            println(io)
        end
    end

    println("Saved comparison report to: $csv_path")
    return csv_path
end

function generate_comparison_plots(
    all_solutions::Dict{String,Any},
    all_grids::Dict{String,Any},
    all_shocks::Dict{String,Any},
    all_params::Dict{String,Any},
    config::BenchmarkConfig,
    output_dir::String,
)
    if !HAS_PLOTS
        @warn "Plots.jl not available - skipping plot generation"
        return String[]
    end

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    plot_files = String[]

    methods = sort(collect(keys(all_solutions)))

    # Get reference values from first method
    first_method = methods[1]
    grids_info = all_grids[first_method]
    shocks_info = all_shocks[first_method]
    params = all_params[first_method]

    a_grid = grids_info.a.grid

    # Determine if stochastic
    first_c_policy = all_solutions[first_method].policy[:c].value
    is_stochastic = first_c_policy isa AbstractMatrix && size(first_c_policy, 2) > 1

    if is_stochastic && shocks_info !== nothing
        # Stochastic case - plot expected policies
        π = shocks_info.π
        Na = length(a_grid)

        # Compute expected policies for each method
        all_expected_c = Dict{String,Vector{Float64}}()
        all_expected_x = Dict{String,Vector{Float64}}()

        for method in methods
            c_policy = all_solutions[method].policy[:c].value
            a_policy = all_solutions[method].policy[:a].value

            expected_c = zeros(Na)
            expected_a = zeros(Na)

            for i = 1:Na
                for j = 1:length(π)
                    expected_c[i] += π[j] * c_policy[i, j]
                    expected_a[i] += π[j] * a_policy[i, j]
                end
            end

            expected_x = expected_cash_on_hand_grid(a_grid, params, shocks_info)

            # Extrapolate
            x_min = minimum(expected_x)
            n_extrap = 50
            x_extrap = range(0.0, x_min, length = n_extrap)
            c_extrap = [x * (expected_c[1] / expected_x[1]) for x in x_extrap]

            full_x = vcat(x_extrap, expected_x)
            full_c = vcat(c_extrap, expected_c)

            all_expected_c[method] = full_c
            all_expected_x[method] = full_x
        end

        # Create consumption comparison plot
        @eval begin
            using Plots
            gr()

            p_c = plot(
                xlabel = "Cash-on-hand (x = R*a + E[y])",
                ylabel = "Expected Consumption E[c]",
                title = "All Methods: Expected Consumption Policy Comparison",
                size = (800, 600),
                dpi = 150,
                grid = true,
                gridstyle = :solid,
                gridalpha = 0.3,
                gridlinewidth = 0.5,
                framestyle = :box,
                xlims = (0.0, 3.5),
                ylims = (0.0, 1.75),
                xticks = 0.0:0.25:3.5,
                yticks = 0.0:0.25:1.75,
                legend = :bottomright,
            )

            # Plot each method
            method_colors = Dict(
                "TimeIteration" => :blue,
                "EGM" => :red,
                "Projection" => :green,
                "NN" => :purple,
                "Perturbation" => :orange,
            )

            for method in $methods
                color = get(method_colors, method, :black)
                plot!(
                    p_c,
                    $all_expected_x[method],
                    $all_expected_c[method],
                    label = method,
                    color = color,
                    linewidth = 1.5,
                    alpha = 0.8,
                )
            end

            # Add 45-degree line reference
            plot!(
                p_c,
                [0.0, 1.75],
                [0.0, 1.75],
                linestyle = :dash,
                color = :gray,
                linewidth = 1,
                label = "45° line",
            )

            c_path = joinpath($output_dir, "all_methods_consumption_$($(timestamp)).png")
            savefig(p_c, c_path)
            push!($plot_files, c_path)
        end
    else
        # Deterministic case
        @eval begin
            using Plots
            gr()

            p_c = plot(
                xlabel = "Asset holdings (a)",
                ylabel = "Consumption (c)",
                title = "All Methods: Consumption Policy Comparison",
                size = (800, 600),
                dpi = 150,
                legend = :bottomright,
            )

            method_colors = Dict(
                "TimeIteration" => :blue,
                "EGM" => :red,
                "Projection" => :green,
                "NN" => :purple,
                "Perturbation" => :orange,
            )

            for method in $methods
                c_policy = $all_solutions[method].policy[:c].value
                c_vec = c_policy isa AbstractVector ? c_policy : vec(c_policy)
                color = get(method_colors, method, :black)

                plot!(
                    p_c,
                    $a_grid,
                    c_vec,
                    label = method,
                    color = color,
                    linewidth = 2,
                    alpha = 0.8,
                )
            end

            c_path = joinpath($output_dir, "all_methods_consumption_$($(timestamp)).png")
            savefig(p_c, c_path)
            push!($plot_files, c_path)
        end
    end

    # Convergence comparison plot
    has_convergence_data = false

    @eval begin
        using Plots
        gr()

        p_conv = plot(
            xlabel = "Iteration/Epoch (log scale)",
            ylabel = "log₁₀(RMSE)",
            title = "All Methods: Convergence Comparison",
            size = (800, 600),
            dpi = 150,
            legend = :topright,
            grid = true,
            gridstyle = :solid,
            gridalpha = 0.3,
            gridlinewidth = 0.5,
            framestyle = :box,
            xscale = :log10,
        )

        method_colors = Dict(
            "TimeIteration" => :blue,
            "EGM" => :red,
            "Projection" => :green,
            "NN" => :purple,
            "Perturbation" => :orange,
        )

        local has_conv_data = false

        for method in $methods
            sol = $all_solutions[method]
            if haskey(sol.metadata, :rmse_history)
                rmse_hist = sol.metadata[:rmse_history]
                if !isempty(rmse_hist)
                    valid_idx = findall(x -> isfinite(x) && x > 0, rmse_hist)
                    if !isempty(valid_idx)
                        iters_plot = collect(1:length(rmse_hist))[valid_idx]
                        log_rmse = log10.(rmse_hist[valid_idx])
                        color = get(method_colors, method, :black)

                        plot!(
                            p_conv,
                            iters_plot,
                            log_rmse,
                            label = method,
                            color = color,
                            linewidth = 1.5,
                            alpha = 0.8,
                            marker = :circle,
                            markersize = 2,
                            markeralpha = 0.6,
                        )
                        has_conv_data = true
                    end
                end
            end
        end

        if has_conv_data
            # Add tolerance line
            hline!(
                p_conv,
                [log10($config.tol)],
                linestyle = :dash,
                color = :black,
                linewidth = 1,
                label = "Tolerance",
            )

            conv_path = joinpath($output_dir, "all_methods_convergence_$($(timestamp)).png")
            savefig(p_conv, conv_path)
            push!($plot_files, conv_path)
        end
    end

    println()
    println("Saved comparison plots:")
    for file in plot_files
        println("  - $file")
    end

    return plot_files
end

function run_all_methods_benchmark(config::BenchmarkConfig)
    methods_to_run = ["TimeIteration", "EGM", "Projection", "NN", "Perturbation"]

    all_solutions = Dict{String,Any}()
    all_stats = Dict{String,NamedTuple}()
    all_grids = Dict{String,Any}()
    all_shocks = Dict{String,Any}()
    all_params = Dict{String,Any}()

    println("="^70)
    println("Running All Methods Comparison Benchmark")
    println("="^70)
    println()

    for method in methods_to_run
        println("\n" * "="^70)
        println("Running $method...")
        println("="^70)

        # Determine method-specific maxit/epochs
        # NN uses epochs parameter, others use maxit
        method_maxit = if method == "NN"
            config.maxit  # Use maxit as default if epochs not set differently
        elseif method == "Projection"
            min(1000, config.maxit)  # Cap Projection at 1000 iterations
        else
            config.maxit
        end

        method_epochs = if method == "NN"
            config.epochs  # Use epochs for NN
        else
            config.epochs  # Pass through for consistency
        end

        # Create temporary config for this method
        method_config = BenchmarkConfig(
            method,
            config.Na,
            config.Nz,
            config.tol,
            config.tol_pol,
            method_maxit,
            method_epochs,
            config.interp_kind,
            config.config_path,
            config.output_dir,
            false,  # Don't generate individual plots
            config.verbose,
        )

        try
            solution, runtime, model, grids_info, shocks_info =
                run_method_benchmark(method_config)
            stats = compute_statistics(solution, runtime, grids_info, shocks_info)

            all_solutions[method] = solution
            all_stats[method] = stats
            all_grids[method] = grids_info
            all_shocks[method] = shocks_info
            all_params[method] = ThesisProject.get_params(model)

            println("\n✓ $method completed successfully")
        catch e
            @warn "Failed to run $method" exception = (e, catch_backtrace())
            println("\n✗ $method failed: $e")
        end
    end

    println("\n" * "="^70)
    println("All Methods Completed")
    println("="^70)
    println()

    return all_solutions, all_stats, all_grids, all_shocks, all_params
end

function main()
    config = parse_cli_args()
    output_dir = ensure_output_dir(config.output_dir)

    if lowercase(config.method) == "all"
        # Run all methods and create comparison
        all_solutions, all_stats, all_grids, all_shocks, all_params =
            run_all_methods_benchmark(config)

        if !isempty(all_stats)
            println("="^70)
            println("Saving Comparison Results")
            println("="^70)
            println()

            # Save comparison report
            save_comparison_report(config, all_stats, output_dir)

            # Generate comparison plots
            if config.generate_plots && !isempty(all_solutions)
                generate_comparison_plots(
                    all_solutions,
                    all_grids,
                    all_shocks,
                    all_params,
                    config,
                    output_dir,
                )
            end
        else
            @warn "No methods completed successfully"
        end
    else
        # Run single method (original behavior)
        solution, runtime, model, grids_info, shocks_info = run_method_benchmark(config)

        # Compute statistics
        stats = compute_statistics(solution, runtime, grids_info, shocks_info)

        # Save report
        println("="^70)
        println("Saving Results")
        println("="^70)
        println()
        save_report(config, stats, output_dir)

        # Generate plots
        if config.generate_plots
            generate_plots(
                solution,
                grids_info,
                shocks_info,
                ThesisProject.get_params(model),
                config,
                output_dir,
            )
        end
    end

    println()
    println("="^70)
    println("Benchmark Complete!")
    println("="^70)
end

end # module

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    MethodsBenchmark.main()
end
