#!/usr/bin/env julia

"""
CSVAR Benchmark Script

Comprehensive benchmarking for CSVAR (vector-income) models. Tests solver methods
across dimensional scaling, correlation structures, and grid sizes with detailed
diagnostics and CSV reporting.

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
  --summary-only        Only print summary, no CSV output
  --verbose             Enable verbose solver output
  --correlation=TYPE    Correlation structure: diagonal, single, toeplitz, dense (for correlation scenario)
  --rotation=TYPE       Rotation type: pca, cholesky (for rotation scenario)
  --nn-modes=LIST       NN objectives to sweep (aio,bcmc_fixed,bcmc_auto,config,all)

Scenarios:
  baseline      - Run base config with specified method(s)
  dimension     - Scale across multiple state dimensions (2D, 3D, 4D, ...)
  correlation   - Test different correlation structures (diagonal, single off-diagonal, toeplitz, dense)
  rotation      - Apply orthogonal rotations (PCA, Cholesky) to test solver invariance

Outputs:
  - CSV report with runtime, RMSE, and history metrics across variants
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
using ThesisProject

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
    summary_only::Bool
    verbose::Bool
    correlation_type::Symbol
    rotation_type::Symbol
    nn_modes::Vector{Symbol}
end

function normalize_nn_mode(token::AbstractString)
    mode = lowercase(strip(token))
    isempty(mode) && return nothing
    if mode == "all"
        return :all
    elseif mode in ("aio", "ai0", "ai-o")
        return :aio
    elseif mode in ("bcmc", "bcmc_fixed", "bcmc-fixed", "fixed")
        return :bcmc_fixed
    elseif mode in ("bcmc_auto", "bcmc-auto", "auto", "adaptive")
        return :bcmc_auto
    elseif mode in ("config", "base", "default")
        return :config
    else
        @warn "Unknown nn mode '$token'; ignoring"
        return nothing
    end
end

function parse_nn_modes_arg(arg::AbstractString)
    tokens = split(arg, ",")
    modes = Symbol[]
    for tok in tokens
        normalized = normalize_nn_mode(tok)
        if normalized === :all
            return [:aio, :bcmc_fixed, :bcmc_auto]
        elseif normalized !== nothing
            push!(modes, normalized)
        end
    end
    if isempty(modes)
        return [:config]
    end
    unique!(modes)
    return modes
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
    summary_only = false
    verbose = false
    correlation_type = :diagonal
    rotation_type = :pca
    nn_modes = [:config]
    dims_specified = false

    for arg in args
        if startswith(arg, "--method=")
            method = split(arg, "=")[2]
        elseif startswith(arg, "--scenario=")
            scenario = Symbol(lowercase(split(arg, "=")[2]))
        elseif startswith(arg, "--dims=")
            dims_str = split(arg, "=")[2]
            dims = parse.(Int, split(dims_str, ","))
            dims_specified = true
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
        elseif arg == "--summary-only"
            summary_only = true
        elseif arg == "--verbose"
            verbose = true
        elseif startswith(arg, "--correlation=")
            correlation_type = Symbol(lowercase(split(arg, "=")[2]))
        elseif startswith(arg, "--rotation=")
            rotation_type = Symbol(lowercase(split(arg, "=")[2]))
        elseif startswith(arg, "--nn-modes=")
            modes_str = split(arg, "=", limit = 2)[2]
            nn_modes = parse_nn_modes_arg(modes_str)
        elseif arg == "--help" || arg == "-h"
            print_help()
            exit(0)
        else
            @warn "Unknown argument: $arg (use --help for usage)"
        end
    end

    if nn_modes != [:config] && !dims_specified
        dims = collect(2:5)
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
        summary_only,
        verbose,
        correlation_type,
        rotation_type,
        nn_modes,
    )
end

function print_help()
    println(
        """
CSVAR Benchmark Script

Comprehensive benchmarking for CSVAR (vector-income) models. Tests solver methods
across dimensional scaling, correlation structures, and grid sizes with detailed
diagnostics and CSV reporting.

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
  --summary-only        Only print summary, no CSV output
  --verbose             Enable verbose solver output
  --correlation=TYPE    Correlation structure: diagonal, single, toeplitz, dense (for correlation scenario)
  --rotation=TYPE       Rotation type: pca, cholesky (for rotation scenario)

Scenarios:
  baseline      - Run base config with specified method(s)
  dimension     - Scale across multiple state dimensions (2D, 3D, 4D, ...)
  correlation   - Test different correlation structures (diagonal, single off-diagonal, toeplitz, dense)
  rotation      - Apply orthogonal rotations (PCA, Cholesky) to test solver invariance

Outputs:
  - CSV report with convergence statistics and histories per variant
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

function apply_nn_mode_overrides(cfg::NamedTuple, mode::Symbol)
    mode == :config && return cfg
    overrides = if mode == :aio
        (; objective = :euler_fb_aio, bcmc_auto_N = false)
    elseif mode == :bcmc_fixed
        (; objective = :euler_fb_bcmc, bcmc_auto_N = false)
    elseif mode == :bcmc_auto
        (; objective = :euler_fb_bcmc, bcmc_auto_N = true)
    else
        error("Unknown NN mode: $mode")
    end
    nn_block = get_nested(cfg, (:solver, :nn), NamedTuple())
    new_nn = merge_config(nn_block, overrides)
    return merge_section(cfg, :solver, (nn = new_nn,))
end

function maybe_set_skip_eval(cfg::NamedTuple, y_dim::Int)
    if y_dim ≥ 3
        nn_block = get_nested(cfg, (:solver, :nn), NamedTuple())
        new_nn = merge_config(nn_block, (; skip_final_eval = true))
        return merge_section(cfg, :solver, (nn = new_nn,))
    end
    return cfg
end

function classify_nn_mode(objective::Symbol, bcmc_auto::Bool)
    if objective == :euler_fb_aio
        return :aio
    elseif objective == :euler_fb_bcmc
        return bcmc_auto ? :bcmc_auto : :bcmc_fixed
    else
        return objective
    end
end

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

    base_variants = build_dimensional_overrides(
        base_cfg;
        dims = dims_to_use,
        total_income = total_income,
        persistence = persistence,
        variance = variance,
    )
    combined = Dict{Symbol,NamedTuple}()
    modes = isempty(config.nn_modes) ? [:config] : config.nn_modes
    for (dim_key, cfg_dim) in base_variants
        params_dim = get_nested(cfg_dim, (:params,))
        y_dim = params_dim === nothing ? 0 : length(params_dim.y)
        for mode in modes
            cfg_mode = apply_nn_mode_overrides(cfg_dim, mode)
            cfg_mode = maybe_set_skip_eval(cfg_mode, y_dim)
            variant_label = if mode == :config
                dim_key
            else
                Symbol(string(dim_key) * "_" * lowercase(String(mode)))
            end
            combined[variant_label] = cfg_mode
        end
    end

    return combined
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
        mc_mean_abs =
            haskey(meta, :mc_mean_abs_resid) ? Float64(meta[:mc_mean_abs_resid]) : NaN
        mc_rms = haskey(meta, :mc_rms_resid) ? Float64(meta[:mc_rms_resid]) : NaN
        mc_max_abs =
            haskey(meta, :mc_max_abs_resid) ? Float64(meta[:mc_max_abs_resid]) : NaN
        mc_effective_n =
            haskey(meta, :mc_effective_n) ? Float64(meta[:mc_effective_n]) : NaN
        mc_sample_size =
            haskey(meta, :mc_sample_size) ? Float64(meta[:mc_sample_size]) : NaN
        loss_history =
            haskey(meta, :loss_history) ? Float64.(meta[:loss_history]) : Float64[]
        best_loss =
            haskey(meta, :best_loss) ? Float64(meta[:best_loss]) :
            (isempty(loss_history) ? NaN : minimum(loss_history))
        final_loss =
            haskey(meta, :final_loss) ? Float64(meta[:final_loss]) :
            (isempty(loss_history) ? NaN : loss_history[end])
        objective_sym = if haskey(meta, :objective)
            Symbol(meta[:objective])
        else
            obj_cfg = get_nested(cfg, (:solver, :nn, :objective), nothing)
            obj_cfg === nothing ? :unknown : Symbol(obj_cfg)
        end
        bcmc_auto = if haskey(meta, :bcmc_auto_N)
            Bool(meta[:bcmc_auto_N])
        else
            val = get_nested(cfg, (:solver, :nn, :bcmc_auto_N), false)
            val === nothing ? false : Bool(val)
        end
        nn_mode = classify_nn_mode(objective_sym, bcmc_auto)
        skip_final_eval = if haskey(meta, :skip_final_eval)
            Bool(meta[:skip_final_eval])
        else
            val = get_nested(cfg, (:solver, :nn, :skip_final_eval), false)
            val === nothing ? false : Bool(val)
        end

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
        mc_mean_abs = NaN
        mc_rms = NaN
        mc_max_abs = NaN
        mc_effective_n = NaN
        mc_sample_size = NaN
        loss_history = Float64[]
        best_loss = NaN
        final_loss = NaN
        objective_sym = :unknown
        nn_mode = :unknown
        skip_final_eval = false
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
        rmse_history = rmse_history,
        N_history = N_history,
        v_h_history = v_h_history,
        loss_history = loss_history,
        best_loss = best_loss,
        final_loss = final_loss,
        objective = objective_sym,
        nn_mode = nn_mode,
        skip_final_eval = skip_final_eval,
        mc_mean_abs = mc_mean_abs,
        mc_rms = mc_rms,
        mc_max_abs = mc_max_abs,
        mc_effective_n = mc_effective_n,
        mc_sample_size = mc_sample_size,
        status = status,
        message = message,
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
        config.summary_only,
        config.verbose,
        config.correlation_type,
        config.rotation_type,
        config.nn_modes,
    )

    println("Configuration:")
    println("  Scenario:         $(config.scenario)")
    println("  Method(s):        $(config.method)")
    println("  Grid points (Na): $(config.Na)")
    println("  Tolerance:        $(config.tol)")
    if lowercase(config.method) == "nn" || config.method == "all"
        println("  NN Epochs:        $(config.epochs)")
        if config.nn_modes != [:config]
            mode_labels = join(String.(config.nn_modes), ", ")
            println("  NN Modes:        $(mode_labels)")
        end
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
                rmse = res.final_rmse,
                converged = res.converged,
                rmse_history = res.rmse_history,
                objective = String(res.objective),
                nn_mode = String(res.nn_mode),
                best_loss = res.best_loss,
                final_loss = res.final_loss,
                min_loss = res.best_loss,
                loss_history = res.loss_history,
                N_history = res.N_history,
                v_h_history = res.v_h_history,
                skip_final_eval = res.skip_final_eval,
                mc_mean_abs = res.mc_mean_abs,
                mc_rms = res.mc_rms,
                mc_max_abs = res.mc_max_abs,
                mc_effective_n = res.mc_effective_n,
                mc_sample_size = res.mc_sample_size,
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
        :best_loss => mean => :best_loss_mean,
        :final_loss => mean => :final_loss_mean,
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
    ts_now = now()
    readable_timestamp = Dates.format(ts_now, "yyyy-mm-dd HH:MM:SS")

    consolidated_path = joinpath(output_dir, "csvar_benchmark_runs.csv")
    df2 = deepcopy(df)
    n = nrow(df2)
    if n > 0
        df2[!, :timestamp] = fill(readable_timestamp, n)
        df2[!, :scenario] = fill(String(config.scenario), n)
        df2[!, :Na] = fill(config.Na, n)
        df2[!, :tol] = fill(config.tol, n)
        df2[!, :epochs] = fill(config.epochs, n)
        df2[!, :base_config] = fill(basename(config.config_path), n)
        append_runs = isfile(consolidated_path)
        CSV.write(consolidated_path, df2; append = append_runs, writeheader = !append_runs)
    end

    println("Results saved:")
    println("  - Consolidated CSV: $consolidated_path (appended)")
    println()

    return consolidated_path
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

    # Persist results
    if !config.summary_only
        println("="^70)
        println("Saving Results")
        println("="^70)
        println()

        save_results(config, results, df, summary, output_dir)
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
