#!/usr/bin/env julia
"""
Compare bc-MC (with auto-N) vs AiO for the NN solver on a given model config.

Outputs:
- CSV summary of metrics (runtime, residuals, RMSE vs EGM).
- JSON dump of detailed diagnostics.
- PNG plots: policy overlays and metric bars.

Usage:
  julia --project=. scripts/experiments/compare_bcmc_aio.jl \
    [--config=PATH] [--epochs=INT] [--outdir=DIR] [--bcmc-budget=INT] [--seed=INT]

Defaults:
  --config=config/smoke_cfg_csvar_stoch.yaml
  --epochs=4000
  --outdir=outputs/bcmc_vs_aio
  --bcmc-budget: keep config default (or kernel default)
  --seed: use config random.seed if present, else 2025
"""
module CompareBCMCvsAiO

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "..")); io = devnull)

using Dates
using JSON3
using CSV
using DataFrames
using Printf
using Statistics
using LinearAlgebra
try
    @eval using Plots
catch err
    @warn "Plots not available; plots will be skipped" err
end

using ThesisProject
const TP = ThesisProject
using ThesisProject.Determinism: make_master_rng
include(joinpath(@__DIR__, "..", "..", "utils", "config_helpers.jl"))
using .ScriptConfigHelpers:
    dict_to_namedtuple,
    merge_config,
    merge_section,
    set_nested,
    get_nested,
    maybe_namedtuple

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

# ---------- helpers ----------
struct RunResult
    name::Symbol             # :BCMC or :AiO
    solve_time::Float64      # wall seconds
    sol::TP.API.Solution
end

function ensure_outdir(outdir::AbstractString)
    isdir(outdir) || mkpath(outdir)
    return outdir
end

# robust RMSE over vectors or matrices
rmse(x) = sqrt(mean((x) .^ 2))
rmse(x, y) = rmse(x .- y)

# --- policy access helpers ---
# fetch value by trying several candidate keys, and unwrap .value if present
fetch_key(policy, k::Symbol) =
    policy isa AbstractDict ? get(policy, k, nothing) :
    policy isa NamedTuple ? (hasproperty(policy, k) ? getfield(policy, k) : nothing) :
    nothing

function extract_policy_values(policy, candidates::Vector{Symbol})
    for k in candidates
        v = fetch_key(policy, k)
        if v !== nothing
            return (v isa NamedTuple && hasproperty(v, :value)) ? getfield(v, :value) : v
        end
    end
    error("Policy keys $(candidates) not found in policy.")
end

# align objects to 1D vectors for RMSE
function align_for_rmse(X, Y, yidx::Int)
    if ndims(X) == 2 && ndims(Y) == 2
        if size(X) == size(Y)
            return vec(X), vec(Y)
        else
            j = min(yidx, min(size(X, 2), size(Y, 2)))
            return vec(X[:, j]), vec(Y[:, j])
        end
    elseif ndims(X) == 1 && ndims(Y) == 2
        j = min(yidx, size(Y, 2))
        return vec(X), vec(Y[:, j])
    elseif ndims(X) == 2 && ndims(Y) == 1
        j = min(yidx, size(X, 2))
        return vec(X[:, j]), vec(Y)
    else
        return vec(X), vec(Y)
    end
end

function pick_income_index(model)::Int
    p = TP.API.get_params(model)
    if hasproperty(p, :y) && p.y isa AbstractVector
        return Int(cld(length(p.y), 2))  # middle state
    else
        return 1
    end
end

function method_overrides(
    objective::Symbol;
    bcmc_auto::Bool = false,
    epochs::Int = 4000,
    bcmc_budget::Union{Nothing,Int} = nothing,
)
    nn = (
        objective = objective,
        epochs = epochs,
        # keep batch_choice, learning rate etc from config
        bcmc_auto_N = bcmc_auto,
    )
    if bcmc_budget !== nothing
        nn = merge(nn, (bcmc_budget_T = bcmc_budget,))
    end
    return (; solver = (; method = :NN, nn = nn))
end

function solve_nn_variant(
    base_cfg::NamedTuple,
    model,
    kind::Symbol;
    epochs::Int,
    bcmc_budget,
)
    if kind == :BCMC
        cfg = merge_config(
            base_cfg,
            method_overrides(
                :euler_fb_bcmc;
                bcmc_auto = true,
                epochs = epochs,
                bcmc_budget = bcmc_budget,
            ),
        )
    elseif kind == :AiO
        cfg = merge_config(
            base_cfg,
            method_overrides(
                :euler_fb_aio;
                bcmc_auto = false,
                epochs = epochs,
                bcmc_budget = nothing,
            ),
        )
    else
        error("unknown kind $kind")
    end
    method = TP.build_method(cfg)
    t0 = time()
    sol = TP.solve(model, method, cfg)
    t1 = time()
    return RunResult(kind, t1 - t0, sol)
end

function solve_egm_baseline(base_cfg::NamedTuple, model)
    cfg = merge_config(base_cfg, (; solver = (; method = :EGM,)))
    method = TP.build_method(cfg)
    return TP.solve(model, method, cfg)
end

function collect_metrics(run::RunResult, baseline::TP.API.Solution)
    # basic residual metrics
    diag = run.sol.diagnostics
    meta = run.sol.metadata
    mean_resid = get(diag, :mean_ee, NaN)
    max_resid = get(meta, :max_resid, NaN)
    # RMSE vs EGM on c-policy
    c_nn = run.sol.policy[:c].value
    c_base = baseline.policy[:c].value

    # a' policy (next-period assets)
    a_nn = run.sol.policy[:a].value
    a_base = baseline.policy[:a].value

    yidx = pick_income_index(run.sol.model)  # or model if you have it in scope
    cx, cy = align_for_rmse(c_nn, c_base, yidx)
    ax, ay = align_for_rmse(a_nn, a_base, yidx)

    rmse_c = rmse(cx, cy)
    rmse_anext = rmse(ax, ay)
    iters = get(meta, :iters, missing)
    dev = string(get(diag, :device, "cpu"))
    return (
        method = String(run.name),
        runtime_s = run.solve_time,
        mean_resid = mean_resid,
        max_resid = max_resid,
        rmse_c = rmse_c,
        rmse_anext = rmse_anext,
        iters = iters,
        device = dev,
    )
end

function plot_policies(
    outpng::AbstractString,
    model,
    baseline::TP.API.Solution,
    runs::Vector{RunResult},
)
    @static if @isdefined(Plots)
        g = TP.API.get_grids(model)
        agrid = g.a.grid
        yidx = pick_income_index(model)

        getC(pol) = extract_policy_values(pol, [:c, :consumption, :c_policy])

        plt = plot(legend = :topleft, xlabel = "assets a", ylabel = "consumption c(a)")
        Cb = getC(baseline.policy)
        if ndims(Cb) == 2
            plot!(
                plt,
                agrid,
                vec(Cb[:, min(yidx, size(Cb, 2))]);
                label = "EGM",
                linestyle = :solid,
            )
        else
            plot!(plt, agrid, vec(Cb); label = "EGM", linestyle = :solid)
        end

        for r in runs
            C = getC(r.sol.policy)
            lab = String(r.name)
            if ndims(C) == 2
                plot!(
                    plt,
                    agrid,
                    vec(C[:, min(yidx, size(C, 2))]);
                    label = lab,
                    linestyle = :dash,
                )
            else
                plot!(plt, agrid, vec(C); label = lab, linestyle = :dash)
            end
        end

        savefig(plt, outpng)
    else
        @warn "Plots not available; skipping policy plot"
    end
end

function plot_bars(outpng::AbstractString, df::DataFrame)
    @static if @isdefined(Plots)
        names = String.(df.method)
        p1 = bar(
            names,
            df.runtime_s,
            xlabel = "method",
            ylabel = "runtime (s)",
            legend = false,
        )
        p2 = bar(
            names,
            df.mean_resid,
            xlabel = "method",
            ylabel = "mean Euler resid",
            legend = false,
        )
        p3 = bar(
            names,
            df.rmse_c,
            xlabel = "method",
            ylabel = "RMSE vs EGM: c",
            legend = false,
        )
        plt = plot(p1, p2, p3; layout = (1, 3), size = (1400, 380))
        savefig(plt, outpng)
    else
        @warn "Plots not available; skipping bar plots"
    end
end

function main()
    # -------- CLI parsing --------
    cfg_path = joinpath(ROOT, "config", "smoke_cfg_csvar_stoch.yaml")
    epochs = 10000
    outdir = joinpath(ROOT, "outputs", "bcmc_vs_aio")
    bcmc_budget = nothing
    seed_override = nothing

    for arg in ARGS
        if startswith(arg, "--config=")
            cfg_path = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--epochs=")
            epochs = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--outdir=")
            outdir = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--bcmc-budget=")
            bcmc_budget = parse(Int, split(arg, "=", limit = 2)[2])
        elseif startswith(arg, "--seed=")
            seed_override = parse(Int, split(arg, "=", limit = 2)[2])
        else
            @warn "Unknown argument $arg"
        end
    end

    # adjust default output to match repo layout
    outdir = joinpath(ROOT, "outputs", "bcmc_vs_aio")
    ensure_outdir(outdir)

    # -------- Load config and build shared model --------
    base_cfg = TP.API.load_config(cfg_path)
    base_cfg = dict_to_namedtuple(base_cfg)
    if seed_override !== nothing
        base_cfg = merge_config(base_cfg, (; random = (; seed = seed_override,),))
    end
    model = TP.build_model(base_cfg)
    # deterministic RNG for fair comparison
    master =
        hasproperty(base_cfg, :random) && hasproperty(base_cfg.random, :seed) ?
        make_master_rng(base_cfg.random.seed) : make_master_rng(2025)

    # -------- Solve baseline (EGM) once --------
    egm_sol = solve_egm_baseline(base_cfg, model)

    # -------- Solve NN with BCMC(auto-N) and AiO --------
    runs = RunResult[]
    push!(
        runs,
        solve_nn_variant(
            base_cfg,
            model,
            :BCMC;
            epochs = epochs,
            bcmc_budget = bcmc_budget,
        ),
    )
    push!(
        runs,
        solve_nn_variant(base_cfg, model, :AiO; epochs = epochs, bcmc_budget = nothing),
    )

    # -------- Metrics and persistence --------
    rows = [collect_metrics(r, egm_sol) for r in runs]
    df = DataFrame(rows)
    ts = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    csv_path = joinpath(outdir, "summary_$ts.csv")
    CSV.write(csv_path, df)

    # --- JSON diagnostics (robust to NamedTuple / Dict) ---
    to_dict(x) =
        x isa NamedTuple ? Dict(pairs(x)) :
        x isa AbstractDict ? Dict(x) : Dict{String,Any}()

    runs_dict = Dict(
        String(r.name) => Dict(
            "runtime_s" => r.solve_time,
            "diagnostics" => to_dict(r.sol.diagnostics),
            "metadata" => to_dict(r.sol.metadata),
        ) for r in runs
    )

    det = Dict("config" => cfg_path, "epochs" => epochs, "runs" => runs_dict)

    json_path = joinpath(outdir, "diagnostics_$ts.json")
    open(json_path, "w") do io
        JSON3.write(io, det; allow_inf = true, indent = 2)
    end

    # plots
    pol_png = joinpath(outdir, "policies_$ts.png")
    bar_png = joinpath(outdir, "metrics_$ts.png")
    plot_policies(pol_png, model, egm_sol, runs)
    plot_bars(bar_png, df)

    println("Done. Saved:")
    println("  ", csv_path)
    println("  ", pol_png)
    println("  ", bar_png)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
