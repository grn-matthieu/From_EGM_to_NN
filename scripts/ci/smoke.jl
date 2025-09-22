#!/usr/bin/env julia
using Pkg
using Dates

# Ensure the project is on LOAD_PATH (repo root is two levels up)
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))
using ThesisProject
using Printf

include(joinpath(@__DIR__, "..", "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

struct SmokeResult
    cfg::String
    ok::Bool
    converged::Bool
    max_resid::Float64
    iters::Int
    runtime::Float64
    interp_kind::String
end

function run_one(cfg_path::AbstractString; tol_resid = 1e-5, tol_iters = 10_000)
    cfg_loaded = ThesisProject.load_config(cfg_path)
    ThesisProject.validate_config(cfg_loaded)
    cfg = dict_to_namedtuple(cfg_loaded)
    cfg_dict = namedtuple_to_dict(cfg)
    model = ThesisProject.build_model(cfg_dict)
    method = ThesisProject.build_method(cfg_dict)
    sol = ThesisProject.solve(model, method, cfg_dict)

    converged = sol.metadata[:converged] === true
    max_resid = sol.metadata[:max_resid]
    iters = sol.metadata[:iters]
    runtime = sol.diagnostics.runtime
    interp_kind = sol.metadata[:interp_kind]

    ok = converged && (max_resid < tol_resid) && (iters <= tol_iters)
    return SmokeResult(cfg_path, ok, converged, max_resid, iters, runtime, interp_kind)
end

function main()
    cfgs = String[]
    # CLI args are config paths; if none provided, use defaults
    if isempty(ARGS)
        push!(
            cfgs,
            joinpath(@__DIR__, "..", "..", "config", "smoke_config", "smoke_config.yaml"),
        )
        push!(
            cfgs,
            joinpath(
                @__DIR__,
                "..",
                "..",
                "config",
                "smoke_config",
                "smoke_config_stochastic.yaml",
            ),
        )
    else
        append!(cfgs, ARGS)
    end

    results = SmokeResult[]
    for c in cfgs
        try
            r = run_one(c)
            push!(results, r)
            status = r.ok ? "OK" : "FAIL"
            println(
                "[",
                status,
                "] ",
                c,
                " | resid=",
                @sprintf("%.3e", r.max_resid),
                " iters=",
                r.iters,
                " interp=",
                r.interp_kind,
                " t=",
                @sprintf("%.2fs", r.runtime)
            )
        catch err
            println("[ERROR] ", c, " | ", err)
            push!(results, SmokeResult(c, false, false, Inf, 0, 0.0, "?"))
        end
    end

    any_fail = any(r -> !r.ok, results)
    exit(any_fail ? 1 : 0)
end

main()
