#!/usr/bin/env julia

module CompareTIVsNN

import Pkg
using Printf
using Statistics

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_CFG = joinpath(ROOT, "config", "smoke_cfg_stoch.yaml")
const DEFAULT_OUT = joinpath(ROOT, "outputs", "ti_vs_nn_simple_stoch.csv")

Pkg.activate(ROOT; io = devnull)
using ThesisProject

struct ScriptOptions
    cfg_path::String
    out_path::String
    summary_only::Bool
end

function parse_cli(args)::ScriptOptions
    cfg_path = DEFAULT_CFG
    out_path = DEFAULT_OUT
    summary_only = false

    for arg in args
        if startswith(arg, "--config=")
            cfg_path = split(arg, "=", limit = 2)[2]
        elseif startswith(arg, "--out=")
            out_path = split(arg, "=", limit = 2)[2]
        elseif arg == "--summary-only"
            summary_only = true
        end
    end

    return ScriptOptions(cfg_path, out_path, summary_only)
end

ensure_parent(path::AbstractString) = (isdir(dirname(path)) || mkpath(dirname(path)); path)

to_array64(x) = Float64.(x isa Array ? x : Array(x))

function reshape_policy(x)
    arr = to_array64(x)
    return ndims(arr) == 1 ? reshape(arr, :, 1) : arr
end

function extract_shock_grid(sol)
    shocks = ThesisProject.get_shocks(sol.model)
    if shocks === nothing
        return nothing
    end
    return Float64.(shocks.zgrid)
end

function build_configs(cfg)
    solver_common = cfg.solver
    solver_ti = merge(solver_common, (method = :TimeIteration,))
    solver_nn = merge(solver_common, (method = :NN,))
    return (merge(cfg, (solver = solver_ti,)), merge(cfg, (solver = solver_nn,)))
end

function compare_policies(sol_ti, sol_nn)
    pol_ti = sol_ti.policy
    pol_nn = sol_nn.policy

    haskey(pol_ti, :c) || error("TimeIteration solution missing :c policy")
    haskey(pol_nn, :c) || error("NN solution missing :c policy")

    c_ti = reshape_policy(pol_ti[:c].value)
    c_nn = reshape_policy(pol_nn[:c].value)
    a_ti = reshape_policy(pol_ti[:a].value)
    a_nn = reshape_policy(pol_nn[:a].value)

    size(c_ti) == size(c_nn) ||
        error("Consumption shapes differ: $(size(c_ti)) vs $(size(c_nn))")
    size(a_ti) == size(a_nn) || error("Asset shapes differ: $(size(a_ti)) vs $(size(a_nn))")

    diff_c = c_ti .- c_nn
    diff_a = a_ti .- a_nn

    stats = (
        c_max_abs = maximum(abs.(diff_c)),
        c_mean_abs = mean(abs.(diff_c)),
        a_max_abs = maximum(abs.(diff_a)),
        a_mean_abs = mean(abs.(diff_a)),
    )

    return (; c_ti, c_nn, a_ti, a_nn, diff_c, diff_a, stats)
end

function prepare_rows(sol_ti, comparison)
    c_meta = sol_ti.policy[:c]
    a_grid = Float64.(c_meta.grid)
    shocks = extract_shock_grid(sol_ti)

    c_ti = comparison.c_ti
    c_nn = comparison.c_nn
    diff_c = comparison.diff_c
    a_ti = comparison.a_ti
    a_nn = comparison.a_nn
    diff_a = comparison.diff_a

    Na = size(c_ti, 1)
    Nz = size(c_ti, 2)
    z_grid =
        shocks === nothing ? fill(0.0, Nz) :
        (length(shocks) == Nz ? shocks : error("Shock grid length mismatch with Nz"))

    rows = Vector{
        NamedTuple{
            (:ia, :iz, :a, :z, :c_ti, :c_nn, :c_diff, :a_ti, :a_nn, :a_diff),
            NTuple{10,Float64},
        },
    }()
    sizehint!(rows, Na * Nz)

    for iz = 1:Nz
        z_val = z_grid[iz]
        for ia = 1:Na
            push!(
                rows,
                (
                    ia = Float64(ia),
                    iz = Float64(iz),
                    a = a_grid[ia],
                    z = z_val,
                    c_ti = c_ti[ia, iz],
                    c_nn = c_nn[ia, iz],
                    c_diff = diff_c[ia, iz],
                    a_ti = a_ti[ia, iz],
                    a_nn = a_nn[ia, iz],
                    a_diff = diff_a[ia, iz],
                ),
            )
        end
    end

    return rows
end

function write_csv(path::AbstractString, rows)
    ensure_parent(path)
    header = "ia,iz,a,z,c_time_iteration,c_nn,c_diff,a_next_time_iteration,a_next_nn,a_diff"
    open(path, "w") do io
        println(io, header)
        for row in rows
            println(
                io,
                @sprintf(
                    "%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f",
                    Int(row.ia),
                    Int(row.iz),
                    row.a,
                    row.z,
                    row.c_ti,
                    row.c_nn,
                    row.c_diff,
                    row.a_ti,
                    row.a_nn,
                    row.a_diff,
                ),
            )
        end
    end
    return path
end

function report(stats, rows_count, cfg_path)
    @printf("Config: %s\n", cfg_path)
    @printf("Points compared: %d\n", rows_count)
    @printf("Consumption |Δ| max=%.6e mean=%.6e\n", stats.c_max_abs, stats.c_mean_abs)
    @printf("Assets      |Δ| max=%.6e mean=%.6e\n", stats.a_max_abs, stats.a_mean_abs)
end

function run()
    opts = parse_cli(ARGS)
    cfg = ThesisProject.load_config(opts.cfg_path)
    cfg_ti, cfg_nn = build_configs(cfg)

    sol_ti = ThesisProject.solve(cfg_ti)
    sol_nn = ThesisProject.solve(cfg_nn)

    comparison = compare_policies(sol_ti, sol_nn)
    rows = prepare_rows(sol_ti, comparison)

    report(comparison.stats, length(rows), opts.cfg_path)

    if !opts.summary_only
        written = write_csv(opts.out_path, rows)
        @printf("Wrote pointwise comparison to %s\n", written)
    end
end

run()

end # module
