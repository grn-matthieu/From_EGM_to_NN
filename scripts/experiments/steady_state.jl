#!/usr/bin/env julia
"""
Compute steady state analytically/numerically and verify against EGM policies.

Usage examples:
  julia --project=. scripts/experiments/steady_state.jl                      # default baseline
  julia --project=. scripts/experiments/steady_state.jl --config=config/simple_stochastic.yaml
"""
module SteadyStateScript

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")); io = devnull)

using LinearAlgebra
using Printf
using SparseArrays
using ThesisProject
using ThesisProject.Determinism: make_rng

include(joinpath(@__DIR__, "..", "utils", "config_helpers.jl"))
using .ScriptConfigHelpers

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function parse_args(args)
    cfg = joinpath(ROOT, "config", "simple_baseline.yaml")
    seed = 0
    for a in args
        if startswith(a, "--config=")
            cfg = split(a, "=", limit = 2)[2]
        elseif startswith(a, "--seed=")
            seed = parse(Int, split(a, "=", limit = 2)[2])
        end
    end
    return (; cfg, seed)
end

function verify_deterministic(cfg_path; seed = 0)
    cfg_loaded = ThesisProject.load_config(cfg_path)
    ThesisProject.validate_config(cfg_loaded)
    cfg = dict_to_namedtuple(cfg_loaded)
    cfg = merge_section(cfg, :shocks, (; active = false))
    cfg_dict = namedtuple_to_dict(cfg)

    model = ThesisProject.build_model(cfg_dict)
    method = ThesisProject.build_method(cfg_dict)
    sol = ThesisProject.solve(model, method, cfg_dict; rng = make_rng(seed))

    ana = ThesisProject.steady_state_analytic(model)
    pol = ThesisProject.steady_state_from_policy(sol)

    agrid = sol.policy[:a].grid
    a_next = sol.policy[:a].value

    @printf(
        "Deterministic steady state (analytic): a*=%.6f, c*=%.6f, kind=%s\n",
        ana.a_ss,
        ana.c_ss,
        String(ana.kind)
    )
    @printf(
        "Deterministic steady state (policy):  a~=%.6f, c~=%.6f (grid idx=%d, gap=%.3e)\n",
        pol.a_star,
        pol.c_star,
        pol.idx,
        pol.gap
    )
    @printf("Nearest-grid diff |a' - a| min: %.3e\n", minimum(abs.(a_next .- agrid)))
    return nothing
end

"""
Compute invariant distribution over (asset, shock) for a given Solution with
stochastic shocks, using linear interpolation weights on the asset grid and the
shock transition matrix Π.
Returns stationary distribution `π` (Na×Nz), and stationary moments.
"""
function stationary_distribution(sol)
    pol_a = sol.policy[:a].value   # Na×Nz
    pol_c = sol.policy[:c].value   # Na×Nz
    agrid = sol.policy[:a].grid
    Na = length(agrid)
    S = ThesisProject.get_shocks(sol.model)
    zgrid, Π = S.zgrid, S.Π
    Nz = length(zgrid)

    # Build sparse transition over combined state (i,j) -> (ip,jp)
    # State index mapping
    state_id(i, j) = (j - 1) * Na + i
    nstate = Na * Nz
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    # Helper to find bracketing asset indices and weights
    function bracket_weights(x)
        if x <= agrid[1]
            return 1, 1, 1.0, 0.0
        elseif x >= agrid[end]
            return Na, Na, 1.0, 0.0
        else
            j = searchsortedfirst(agrid, x)
            j = clamp(j, 2, Na)
            i0 = j - 1
            i1 = j
            t = (x - agrid[i0]) / (agrid[i1] - agrid[i0])
            return i0, i1, 1 - t, t
        end
    end

    for j = 1:Nz
        for i = 1:Na
            ap = pol_a[i, j]
            i0, i1, w0, w1 = bracket_weights(ap)
            s = state_id(i, j)
            for jp = 1:Nz
                pz = Π[j, jp]
                # mass to (i0,jp) and (i1,jp)
                push!(rows, state_id(i0, jp))
                push!(cols, s)
                push!(vals, pz * w0)
                push!(rows, state_id(i1, jp))
                push!(cols, s)
                push!(vals, pz * w1)
            end
        end
    end

    T = sparse(rows, cols, vals, nstate, nstate)  # column-stochastic
    # Power iteration on T' to evolve row vector probabilities π
    π = fill(1.0 / nstate, nstate)
    tmp = similar(π)
    for _ = 1:10_000
        mul!(tmp, transpose(T), π)
        s = sum(tmp)
        tmp ./= s
        if maximum(abs.(tmp .- π)) < 1e-12
            break
        end
        π .= tmp
    end
    Πss = reshape(π, Na, Nz)

    Ea = sum(Πss .* (agrid .* ones(1, Nz)))
    Ec = sum(Πss .* pol_c)
    return Πss, (; Ea, Ec)
end

function verify_stochastic(cfg_path; seed = 0)
    cfg_loaded = ThesisProject.load_config(cfg_path)
    ThesisProject.validate_config(cfg_loaded)
    cfg = dict_to_namedtuple(cfg_loaded)
    @assert get_nested(cfg, (:shocks, :active), false) "Config must have active shocks"

    cfg_dict = namedtuple_to_dict(cfg)
    model = ThesisProject.build_model(cfg_dict)
    method = ThesisProject.build_method(cfg_dict)
    sol = ThesisProject.solve(model, method, cfg_dict; rng = make_rng(seed))

    Πss, mom = stationary_distribution(sol)
    @printf("Stochastic stationary moments: E[a]=%.6f, E[c]=%.6f\n", mom.Ea, mom.Ec)
    return nothing
end

function main(args)
    opts = parse_args(args)
    cfg_loaded = ThesisProject.load_config(opts.cfg)
    ThesisProject.validate_config(cfg_loaded)
    cfg = dict_to_namedtuple(cfg_loaded)
    stoch = get_nested(cfg, (:shocks, :active), false)
    if stoch
        println("Running stochastic steady-state verification...")
        verify_stochastic(opts.cfg; seed = opts.seed)
    else
        println("Running deterministic steady-state verification...")
        verify_deterministic(opts.cfg; seed = opts.seed)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

end # module
