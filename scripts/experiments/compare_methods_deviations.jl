#!/usr/bin/env julia
"""
Automate plots of policy/value function deviations for all 3 methods
(EGM, Projection, Perturbation) on the same model configuration.

Usage:
  julia --project=. scripts/experiments/compare_methods_deviations.jl [--config=path]

Notes:
- Saves figures under `outputs/`:
    - dev_policy_c.png, dev_policy_a.png, dev_value.png
    - If stochastic, also saves heatmaps per variable under `outputs/heatmaps/`.
- Defaults to `config/simple_baseline.yaml` if `--config` not provided.
"""
module CompareMethodsDeviations

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)

using ThesisProject
using ThesisProject.Determinism: make_rng
using Statistics: mean

# Load plotting backend via the package extension (Plots is weak dep)
try
    @eval using Plots
catch err
    @warn "Plots not available; deviation plots will be skipped" err
end

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const OUTDIR = joinpath(ROOT, "outputs")
const HEATDIR = joinpath(OUTDIR, "heatmaps")

"""Simple CLI parse for --config=..."""
function parse_cli(args)
    cfg_path = nothing
    for a in args
        if startswith(a, "--config=")
            cfg_path = split(a, "=", limit = 2)[2]
        end
    end
    return cfg_path
end

function ensure_dirs()
    isdir(OUTDIR) || mkpath(OUTDIR)
    isdir(HEATDIR) || mkpath(HEATDIR)
end

"""
Build solutions for all three methods from a base config dict.
Returns a Dict{Symbol,ThesisProject.API.Solution} keyed by :EGM, :Projection, :Perturbation.
"""
function solve_all(cfg_base::AbstractDict)
    sols = Dict{Symbol,Any}()
    for m in (:EGM, :Projection, :Perturbation)
        cfg = deepcopy(cfg_base)
        cfg[:solver][:method] = m
        model = ThesisProject.build_model(cfg)
        method = ThesisProject.build_method(cfg)
        sols[m] = ThesisProject.solve(model, method, cfg; rng = make_rng(0))
    end
    return sols
end

"""
Extract grid and payload (vector or matrix) for a given policy key (:c or :a).
Returns (agrid, values) where values is Vector or Matrix.
"""
function get_policy_payload(sol, key::Symbol)
    pol = sol.policy[key]
    return pol.grid, pol.value
end

"""
Compute absolute deviations vs a reference method for a policy or value:
- For vectors: returns a single vector of |X_method - X_ref|.
- For matrices: returns a tuple (dev_mat, dev_mean, dev_max) where:
    dev_mat  :: Na x Nz absolute deviations per-shock
    dev_mean :: length Na vector of stationary-weighted mean deviations
    dev_max  :: length Na vector of max deviation across shocks
Uses invariant distribution weights if available; else uniform weights.
"""
function deviations_vs_ref(payload, payload_ref, sol)
    if payload isa AbstractVector
        return abs.(payload .- payload_ref)
    elseif payload isa AbstractMatrix
        @assert payload_ref isa AbstractMatrix "Shape mismatch"
        devm = abs.(payload .- payload_ref)
        # Weights for mean across shocks
        w = nothing
        try
            S = ThesisProject.get_shocks(sol.model)
            w =
                hasproperty(S, :p) ? getproperty(S, :p) :
                (hasproperty(S, :pi) ? getproperty(S, :pi) : nothing)
        catch
            w = nothing
        end
        Nz = size(devm, 2)
        if w === nothing || length(w) != Nz
            w = fill(1.0 / Nz, Nz)
        end
        dev_mean = devm * collect(w)
        dev_max = vec(maximum(devm, dims = 2))
        return devm, dev_mean, dev_max
    else
        error("Unsupported payload type $(typeof(payload))")
    end
end

"""Save line plots for deviations (vector-valued)."""
function save_dev_plot_line(
    x,
    ys::Vector{<:AbstractVector};
    labels::Vector{String},
    title::String,
    outfile::AbstractString,
)
    plt = plot(; legend = :best)
    for (i, y) in enumerate(ys)
        plot!(plt, x, y; label = labels[i], lw = 2)
    end
    xlabel!(plt, "State")
    ylabel!(plt, "Absolute deviation")
    title!(plt, title)
    savefig(plt, outfile)
    return outfile
end

"""Save a heatmap of deviations (matrix-valued)."""
function save_dev_heatmap(x, zaxis, mat; title::String, outfile::AbstractString)
    plt = heatmap(
        x,
        zaxis,
        mat';
        color = :viridis,
        xlabel = "State",
        ylabel = "Shock",
        title = title,
    )
    savefig(plt, outfile)
    return outfile
end

function run()
    ensure_dirs()
    cfg_path = parse_cli(ARGS)
    if cfg_path === nothing
        cfg_path = joinpath(ROOT, "config", "simple_baseline.yaml")
    end
    cfg = ThesisProject.load_config(cfg_path)
    ThesisProject.validate_config(cfg)

    sols = solve_all(cfg)

    # Use EGM as the reference for deviations
    sol_ref = sols[:EGM]

    # c-policy deviations
    agrid, cref = get_policy_payload(sol_ref, :c)
    c_series = String[]
    c_lines = Vector{Vector{Float64}}()

    # a-policy deviations
    _, aref = get_policy_payload(sol_ref, :a)
    a_series = String[]
    a_lines = Vector{Vector{Float64}}()

    # value deviations
    Vref = sol_ref.value
    v_lines = Vector{Vector{Float64}}()
    v_series = String[]

    # For stochastic cases, we also save heatmaps
    zaxis = nothing
    try
        S = ThesisProject.get_shocks(sol_ref.model)
        zaxis = hasproperty(S, :zgrid) ? getproperty(S, :zgrid) : nothing
    catch
        zaxis = nothing
    end

    for (meth, sol) in sort(collect(sols); by = x -> String(x[1]))
        meth == :EGM && continue  # skip reference

        # c policy
        xc, cval = get_policy_payload(sol, :c)
        @assert xc == agrid "Mismatched grids between methods"
        devc = deviations_vs_ref(cval, cref, sol)
        if devc isa AbstractVector
            push!(c_lines, devc)
            push!(c_series, string(meth, " vs EGM"))
        else
            devm, devmean, _ = devc
            # Save heatmap
            if isdefined(CompareMethodsDeviations, :Plots) && zaxis !== nothing
                save_dev_heatmap(
                    xc,
                    zaxis,
                    devm;
                    title = "|c_$(meth) - c_EGM|",
                    outfile = joinpath(HEATDIR, "dev_c_$(meth)_vs_EGM.png"),
                )
            end
            push!(c_lines, devmean)
            push!(c_series, string(meth, " vs EGM (mean across shocks)"))
        end

        # a policy
        xa, aval = get_policy_payload(sol, :a)
        @assert xa == agrid "Mismatched grids between methods"
        deva = deviations_vs_ref(aval, aref, sol)
        if deva isa AbstractVector
            push!(a_lines, deva)
            push!(a_series, string(meth, " vs EGM"))
        else
            devm, devmean, _ = deva
            if isdefined(CompareMethodsDeviations, :Plots) && zaxis !== nothing
                save_dev_heatmap(
                    xa,
                    zaxis,
                    devm;
                    title = "|a_$(meth) - a_EGM|",
                    outfile = joinpath(HEATDIR, "dev_a_$(meth)_vs_EGM.png"),
                )
            end
            push!(a_lines, devmean)
            push!(a_series, string(meth, " vs EGM (mean across shocks)"))
        end

        # value function
        V = sol.value
        if V isa AbstractVector
            push!(v_lines, abs.(V .- Vref))
            push!(v_series, string(meth, " vs EGM"))
        elseif V isa AbstractMatrix
            @assert Vref isa AbstractMatrix "Value function shape mismatch"
            devm, devmean, _ = deviations_vs_ref(V, Vref, sol)
            if isdefined(CompareMethodsDeviations, :Plots) && zaxis !== nothing
                save_dev_heatmap(
                    agrid,
                    zaxis,
                    devm;
                    title = "|V_$(meth) - V_EGM|",
                    outfile = joinpath(HEATDIR, "dev_V_$(meth)_vs_EGM.png"),
                )
            end
            push!(v_lines, devmean)
            push!(v_series, string(meth, " vs EGM (mean across shocks)"))
        else
            @warn "Unsupported value payload; skipping" meth typeof(V)
        end
    end

    if isdefined(CompareMethodsDeviations, :Plots)
        # Save line plots (vectors or mean-across-shocks)
        save_dev_plot_line(
            agrid,
            c_lines;
            labels = c_series,
            title = "c-policy deviations vs EGM",
            outfile = joinpath(OUTDIR, "dev_policy_c.png"),
        )
        save_dev_plot_line(
            agrid,
            a_lines;
            labels = a_series,
            title = "a-policy deviations vs EGM",
            outfile = joinpath(OUTDIR, "dev_policy_a.png"),
        )
        save_dev_plot_line(
            agrid,
            v_lines;
            labels = v_series,
            title = "Value function deviations vs EGM",
            outfile = joinpath(OUTDIR, "dev_value.png"),
        )
        println(
            "Saved deviation plots to: \n  " * join(
                [
                    joinpath(OUTDIR, f) for
                    f in ("dev_policy_c.png", "dev_policy_a.png", "dev_value.png")
                ],
                "\n  ",
            ),
        )
        if isdir(HEATDIR)
            println("Heatmaps (if stochastic) saved under: " * HEATDIR)
        end
    else
        println("Plots not generated (Plots.jl not available)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end

end # module
