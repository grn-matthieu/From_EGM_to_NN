#!/usr/bin/env julia

module RunAllMethodsSmoke

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)

using Dates
using Printf
using Random
using ThesisProject
using Statistics: mean

include(joinpath(@__DIR__, "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

const ROOT = normpath(joinpath(@__DIR__, ".."))
ensure_outputs_dir() = (out = joinpath(ROOT, "outputs"); isdir(out) || mkpath(out); out)

# Compute simple Euler error summary from solution (matches robustness_sweep style)
function ee_stats(sol)
    pol_c = sol.policy[:c]
    ee = pol_c.euler_errors
    ee_mat = pol_c.euler_errors_mat
    if ee_mat === nothing
        mx = maximum(skipmissing(ee))
        mn = minimum(skipmissing(ee))
        av = mean(skipmissing(ee))
        return (ee_max = mx, ee_min = mn, ee_mean = av)
    else
        mx = maximum(skipmissing(vec(ee_mat)))
        mn = minimum(skipmissing(vec(ee_mat)))
        av = mean(skipmissing(vec(ee_mat)))
        return (ee_max = mx, ee_min = mn, ee_mean = av)
    end
end

function open_write(path::AbstractString, header::Vector{<:AbstractString}, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            strrow = map(x -> x === nothing ? "" : sprint(print, x), row)
            println(io, join(strrow, ","))
        end
    end
end

function main()
    outdir = ensure_outputs_dir()
    outpath = joinpath(outdir, "methods_smoke_results.csv")

    # Try to load the stochastic smoke config first, then deterministic smoke,
    # then fall back to simple_baseline if none are present.
    cfg_path_candidates = (
        joinpath(ROOT, "config", "smoke_cfg_stoch.yaml"),
        joinpath(ROOT, "config", "smoke_cfg_det.yaml"),
        joinpath(ROOT, "config", "simple_baseline.yaml"),
    )
    base_cfg = nothing
    for p in cfg_path_candidates
        try
            @printf("Trying config: %s\n", p)
            base_cfg = ThesisProject.load_config(p)
            @printf("Loaded config: %s\n", p)
            break
        catch err
            @printf("Failed to load config %s: %s\n", p, sprint(showerror, err))
            # try next
        end
    end
    if base_cfg === nothing
        error(
            "Could not find a usable config (looked for smoke_config.yaml and simple_baseline.yaml)",
        )
    end

    # Run Time Iteration first for comparison, then other methods
    methods = ["TimeIteration", "EGM", "Projection", "Perturbation", "NN"]

    ts = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")
    header = [
        "timestamp",
        "method",
        "status",
        "error",
        "run_id",
        "iters",
        "converged",
        "max_resid",
        "tol",
        "tol_pol",
        "interp_kind",
        "runtime",
        "ee_max",
        "ee_min",
        "ee_mean",
    ]

    rows = Vector{NTuple{15,Any}}()

    for m in methods
        # Set the solver.method field (matches config layout) so the
        # method factory picks it up. Use solver.method = m.
        cfg = merge_section(base_cfg, :solver, Dict{Symbol,Any}(:method => m))
        try
            model = ThesisProject.build_model(cfg)
            method = ThesisProject.build_method(cfg)
            sol = ThesisProject.solve(model, method, cfg)

            meta = sol.metadata
            diag = sol.diagnostics

            ee = try
                ee_stats(sol)
            catch
                (ee_max = missing, ee_min = missing, ee_mean = missing)
            end

            # Use safe getters because not all methods populate the same metadata
            run_id = get(diag, :model_id, missing)
            iters = get(meta, :iters, missing)
            converged = get(meta, :converged, missing)
            max_resid = get(meta, :max_resid, missing)
            tol = get(meta, :tol, missing)
            tol_pol = get(meta, :tol_pol, missing)
            interp_kind = get(meta, :interp_kind, missing)
            runtime = get(diag, :runtime, missing)

            # Projection-specific convergence: prefer delta_pol < tol_pol when available
            computed_converged = converged
            if m == "Projection"
                delta_pol = get(meta, :delta_pol, missing)
                # determine a scalar norm for delta_pol
                polnorm = missing
                if delta_pol !== missing
                    if isa(delta_pol, Number)
                        polnorm = abs(delta_pol)
                    else
                        try
                            polnorm = maximum(abs.(delta_pol))
                        catch
                            polnorm = missing
                        end
                    end
                end
                if polnorm !== missing && tol_pol !== missing
                    computed_converged = polnorm < tol_pol
                end
            end

            push!(
                rows,
                (
                    ts,
                    m,
                    "ok",
                    "",
                    run_id,
                    iters,
                    computed_converged,
                    max_resid,
                    tol,
                    tol_pol,
                    interp_kind,
                    runtime,
                    ee.ee_max,
                    ee.ee_min,
                    ee.ee_mean,
                ),
            )
        catch err
            # print full backtrace to stdout for debugging
            try
                bt = catch_backtrace()
                io = IOBuffer()
                Base.show_backtrace(io, bt)
                back = String(take!(io))
                @printf(
                    "Error running method %s: %s\nBacktrace:\n%s\n",
                    m,
                    sprint(showerror, err),
                    back
                )
            catch
                @printf("Error running method %s: %s\n", m, sprint(showerror, err))
            end
            push!(
                rows,
                (
                    ts,
                    m,
                    "error",
                    sprint(showerror, err),
                    missing,
                    missing,
                    false,
                    missing,
                    missing,
                    missing,
                    missing,
                    missing,
                    missing,
                    missing,
                    missing,
                ),
            )
        end
    end

    open_write(outpath, header, rows)
    println("Wrote methods CSV: ", outpath)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    RunAllMethodsSmoke.main()
end
