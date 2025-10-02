#!/usr/bin/env julia

module RunAllMethodsSmoke

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)

using Dates
using Printf
using Random
using ThesisProject
using ThesisProject.CommonInterp: interp_linear!, interp_pchip!
using Statistics: median

include(joinpath(@__DIR__, "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

const ROOT = normpath(joinpath(@__DIR__, ".."))
ensure_outputs_dir() = (out = joinpath(ROOT, "outputs"); isdir(out) || mkpath(out); out)

# Determine interpolation mode and helpers to regrid policies to the model grid.
function infer_interp_mode(mode_meta)
    if mode_meta === nothing || mode_meta === missing
        return :linear
    end
    s = lowercase(String(mode_meta))
    if occursin("monotone", s) || occursin("pchip", s) || occursin("cubic", s)
        return :pchip
    end
    return :linear
end

function grids_match(src::AbstractVector, dst::AbstractVector; atol::Real = 1e-10)
    length(src) == length(dst) && all(abs.(src .- dst) .<= atol)
end

function regrid_array(
    values::AbstractVector,
    src::AbstractVector,
    dst::AbstractVector;
    kind::Symbol,
)
    src_vec = collect(src)
    dst_vec = collect(dst)
    vals = collect(values)
    if grids_match(src_vec, dst_vec)
        return copy(vals)
    end
    T = promote_type(eltype(vals), eltype(dst_vec))
    out = Vector{T}(undef, length(dst_vec))
    if kind == :pchip
        try
            interp_pchip!(out, src_vec, vals, dst_vec)
            return out
        catch
            # fall back to linear interpolation when shape-preserving fails
        end
    end
    interp_linear!(out, src_vec, vals, dst_vec)
    return out
end

function regrid_array(
    values::AbstractMatrix,
    src::AbstractVector,
    dst::AbstractVector;
    kind::Symbol,
)
    src_vec = collect(src)
    dst_vec = collect(dst)
    vals = Array(values)
    if size(vals, 1) == length(dst_vec) && grids_match(src_vec, dst_vec)
        return copy(vals)
    end
    cols = size(vals, 2)
    T = promote_type(eltype(vals), eltype(dst_vec))
    out = Array{T}(undef, length(dst_vec), cols)
    for j = 1:cols
        out[:, j] = regrid_array(view(vals, :, j), src_vec, dst_vec; kind = kind)
    end
    return out
end

function coerce_numeric(arr::AbstractArray)
    data = similar(arr, Float64)
    for idx in eachindex(arr)
        v = arr[idx]
        data[idx] = ismissing(v) ? NaN : Float64(v)
    end
    return data
end

function solve_once(cfg)
    model = ThesisProject.build_model(cfg)
    method = ThesisProject.build_method(cfg)
    return ThesisProject.solve(model, method, cfg)
end

function solve_with_runtime(cfg)
    warm_sol = solve_once(cfg)
    GC.gc()
    t0 = time_ns()
    sol = solve_once(cfg)
    runtime = (time_ns() - t0) / 1e9
    return sol, runtime
end

# Compute binding statistics for the asset policy on the model's exogenous grid.
function binding_stats(sol; atol::Real = 1e-8, interp_mode::Symbol = :linear)
    pol_a = get(sol.policy, :a, nothing)
    if pol_a === nothing
        return (share = missing, mask = nothing, rows = nothing)
    end

    grids = ThesisProject.get_grids(sol.model)
    if grids === nothing || !haskey(grids, :a)
        return (share = missing, mask = nothing, rows = nothing)
    end

    amin = try
        grids[:a].min
    catch
        nothing
    end
    target_grid = try
        grids[:a].grid
    catch
        nothing
    end
    if amin === nothing
        return (share = missing, mask = nothing, rows = nothing)
    end

    a_vals = hasproperty(pol_a, :value) ? pol_a.value : nothing
    a_grid = hasproperty(pol_a, :grid) ? pol_a.grid : target_grid
    if !(a_vals isa AbstractArray)
        return (share = missing, mask = nothing, rows = nothing)
    end

    eval_vals = a_vals
    if target_grid !== nothing && a_grid !== nothing
        eval_vals = regrid_array(a_vals, a_grid, target_grid; kind = interp_mode)
    else
        eval_vals = Array(a_vals)
    end

    mask = Array(abs.(eval_vals .- amin) .<= atol)
    total = length(mask)
    share = total == 0 ? missing : count(identity, mask) / total

    rows_mask = nothing
    if mask isa AbstractVector
        rows_mask = mask
    elseif mask isa AbstractMatrix
        rows_mask = vec(any(mask; dims = 2))
    end

    return (share = share, mask = mask, rows = rows_mask)
end

# Compute Euler-error summary after regridding to the exogenous asset grid and
# skipping binding points.
function ee_stats(
    sol,
    binding_mask::Union{Nothing,AbstractArray} = nothing,
    row_binding_mask::Union{Nothing,AbstractVector} = nothing;
    interp_mode::Symbol = :linear,
)
    pol_c = sol.policy[:c]
    ee = pol_c.euler_errors
    ee_mat = pol_c.euler_errors_mat

    grids = ThesisProject.get_grids(sol.model)
    target_grid = try
        grids[:a].grid
    catch
        nothing
    end
    source_grid = hasproperty(pol_c, :grid) ? pol_c.grid : target_grid

    vals = Float64[]

    if ee_mat === nothing
        numeric = coerce_numeric(ee)
        eval_vec =
            target_grid !== nothing && source_grid !== nothing ?
            regrid_array(numeric, source_grid, target_grid; kind = interp_mode) :
            Array(numeric)
        mask_vec = nothing
        if binding_mask isa AbstractVector && length(binding_mask) == length(eval_vec)
            mask_vec = binding_mask
        elseif row_binding_mask isa AbstractVector &&
               length(row_binding_mask) == length(eval_vec)
            mask_vec = row_binding_mask
        end
        for i in eachindex(eval_vec)
            if mask_vec !== nothing && mask_vec[i]
                continue
            end
            val = eval_vec[i]
            if !isfinite(val)
                continue
            end
            push!(vals, val)
        end
    else
        numeric = coerce_numeric(ee_mat)
        eval_mat =
            target_grid !== nothing && source_grid !== nothing ?
            regrid_array(numeric, source_grid, target_grid; kind = interp_mode) :
            Array(numeric)
        mask_mat =
            binding_mask isa AbstractArray && size(binding_mask) == size(eval_mat) ?
            binding_mask : nothing
        for idx in eachindex(eval_mat)
            if mask_mat !== nothing && mask_mat[idx]
                continue
            end
            val = eval_mat[idx]
            if !isfinite(val)
                continue
            end
            push!(vals, val)
        end
    end

    if isempty(vals)
        return (ee_max = missing, ee_median = missing)
    end
    mx = maximum(vals)
    med = median(vals)
    return (ee_max = mx, ee_median = med)
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
            "Could not find a usable config (looked for smoke configs and simple_baseline)",
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
        "max_end_resid",
        "tol",
        "tol_pol",
        "interp_kind",
        "runtime",
        "ee_max",
        "ee_median",
        "binding_share",
    ]

    rows = Vector{NTuple{15,Any}}()

    for m in methods
        cfg = merge_section(base_cfg, :solver, Dict{Symbol,Any}(:method => m))
        try
            sol, runtime = solve_with_runtime(cfg)

            meta = sol.metadata
            diag = sol.diagnostics

            interp_raw =
                haskey(meta, :interp_kind) ? meta[:interp_kind] :
                (
                    hasproperty(diag, :interp_kind) ? getproperty(diag, :interp_kind) :
                    missing
                )
            interp_mode = infer_interp_mode(interp_raw)

            binding = try
                binding_stats(sol; interp_mode = interp_mode)
            catch
                (share = missing, mask = nothing, rows = nothing)
            end

            ee = try
                ee_stats(sol, binding.mask, binding.rows; interp_mode = interp_mode)
            catch
                (ee_max = missing, ee_median = missing)
            end

            # Use safe getters because not all methods populate the same metadata
            run_id = get(diag, :model_id, missing)
            iters = get(meta, :iters, missing)
            converged = get(meta, :converged, missing)
            max_end_resid = get(meta, :max_resid, missing)
            tol = get(meta, :tol, missing)
            tol_pol = get(meta, :tol_pol, missing)
            interp_kind = get(meta, :interp_kind, missing)
            # Projection-specific convergence: require both residual and policy tolerances and avoid hitting the iteration cap.
            computed_converged = converged
            if m == "Projection"
                raw_converged = converged === missing ? true : Bool(converged)
                delta_pol = get(meta, :delta_pol, missing)
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
                pol_ok = true
                if tol_pol !== missing
                    if polnorm !== missing
                        pol_ok = polnorm < tol_pol
                    else
                        pol_ok = raw_converged
                    end
                end
                max_end_resid = get(meta, :max_resid, missing)
                tol_val = get(meta, :tol, missing)
                resid_ok = true
                if tol_val !== missing
                    if max_end_resid !== missing
                        resid_ok = max_end_resid < tol_val
                    elseif ee.ee_max !== missing
                        resid_ok = ee.ee_max < tol_val
                    else
                        resid_ok = raw_converged
                    end
                end
                iters = get(meta, :iters, missing)
                max_it = get(meta, :max_it, missing)
                iter_ok = true
                if iters !== missing && max_it !== missing
                    iter_ok = iters < max_it
                end
                computed_converged = raw_converged && resid_ok && pol_ok && iter_ok
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
                    max_end_resid,
                    tol,
                    tol_pol,
                    interp_kind,
                    runtime,
                    ee.ee_max,
                    ee.ee_median,
                    binding.share,
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
