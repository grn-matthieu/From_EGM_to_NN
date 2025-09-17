module NNEval

using Statistics: mean
using Dates
using JSON3

using ..API: build_model, build_method, solve

const LOG_FLOOR = 1.0e-12

function _get_or(obj, name::Symbol, default)
    try
        return getproperty(obj, name)
    catch
        return default
    end
end

function _ensure_symbol_dict(x)
    out = Dict{Symbol,Any}()
    if x isa Dict
        for (k, v) in x
            out[k isa Symbol ? k : Symbol(k)] = v
        end
    elseif x isa NamedTuple
        for (k, v) in pairs(x)
            out[k] = v
        end
    end
    return out
end

function _get_subcfg(cfg, key::Symbol)
    if cfg isa Dict
        return haskey(cfg, key) ? _ensure_symbol_dict(cfg[key]) : Dict{Symbol,Any}()
    elseif cfg isa NamedTuple
        return hasproperty(cfg, key) ? _ensure_symbol_dict(getproperty(cfg, key)) :
               Dict{Symbol,Any}()
    else
        return Dict{Symbol,Any}()
    end
end

function _dict_to_namedtuple(d::Dict{Symbol,Any})
    isempty(d) && return NamedTuple()
    return (; d...)
end

function _extract_hyperparams(cfg)
    solver = _dict_to_namedtuple(_get_subcfg(cfg, :solver))
    logging = _dict_to_namedtuple(_get_subcfg(cfg, :logging))
    random = _dict_to_namedtuple(_get_subcfg(cfg, :random))
    evaluation = _dict_to_namedtuple(_get_subcfg(cfg, :evaluation))
    return (
        solver = solver,
        logging = length(logging) == 0 ? nothing : logging,
        random = length(random) == 0 ? nothing : random,
        evaluation = length(evaluation) == 0 ? nothing : evaluation,
    )
end

function _residual_stats(resid)
    resid === nothing && return (
        max_euler_error = NaN,
        mean_euler_error = NaN,
        max_log10_euler_error = NaN,
        mean_log10_euler_error = NaN,
        nonfinite_residuals = 0,
    )

    flat = vec(float.(resid))
    mask = isfinite.(flat)
    finite_vals = flat[mask]
    nonfinite = length(flat) - length(finite_vals)

    if isempty(finite_vals)
        return (
            max_euler_error = NaN,
            mean_euler_error = NaN,
            max_log10_euler_error = NaN,
            mean_log10_euler_error = NaN,
            nonfinite_residuals = nonfinite,
        )
    end

    logs = log10.(clamp.(finite_vals, LOG_FLOOR, Inf))
    return (
        max_euler_error = maximum(finite_vals),
        mean_euler_error = mean(finite_vals),
        max_log10_euler_error = maximum(logs),
        mean_log10_euler_error = mean(logs),
        nonfinite_residuals = nonfinite,
    )
end

function _build_diagnostics(sol)
    opts = _get_or(sol, :opts, NamedTuple())
    return (
        iters = _get_or(sol, :iters, nothing),
        converged = _get_or(sol, :converged, nothing),
        max_resid = _get_or(sol, :max_resid, nothing),
        runtime = _get_or(opts, :runtime, nothing),
        feasibility = _get_or(opts, :feasibility, nothing),
        loss = _get_or(opts, :loss, nothing),
        projection_kind = _get_or(opts, :projection_kind, nothing),
    )
end

function _config_has_shocks(cfg)
    shocks = _get_subcfg(cfg, :shocks)
    return length(shocks) == 0 ? false : get(shocks, :active, false)
end

function _compare_policy(nn_policy, baseline_policy)
    nn_policy === nothing && return nothing
    baseline_policy === nothing && return nothing
    size(nn_policy) == size(baseline_policy) || return nothing

    diff = nn_policy .- baseline_policy
    mask = isfinite.(diff)
    finite = diff[mask]
    isempty(finite) && return nothing

    abs_vals = abs.(finite)
    return (
        max_abs_diff = maximum(abs_vals),
        mean_abs_diff = mean(abs_vals),
        rmse = sqrt(mean(finite .^ 2)),
    )
end

function _baseline_config(cfg)
    cfg isa Dict && return deepcopy(cfg)
    return deepcopy(Dict{Symbol,Any}(pairs(cfg)))
end

function _baseline_comparison(sol, cfg)
    cfg_copy = _baseline_config(cfg)
    cfg_copy isa Dict || return nothing
    solver_cfg_raw = haskey(cfg_copy, :solver) ? cfg_copy[:solver] : Dict{Symbol,Any}()
    solver_cfg = _ensure_symbol_dict(solver_cfg_raw)
    cfg_copy[:solver] = solver_cfg
    solver_cfg[:method] = :EGM

    try
        model = build_model(cfg_copy)
        method = build_method(cfg_copy)
        baseline = solve(model, method, cfg_copy)
        policy = baseline.policy
        consumption = haskey(policy, :c) ? getproperty(policy[:c], :value) : nothing
        assets_next = haskey(policy, :a) ? getproperty(policy[:a], :value) : nothing
        return (
            method = "EGM",
            consumption = _compare_policy(_get_or(sol, :c, nothing), consumption),
            assets_next = _compare_policy(_get_or(sol, :a_next, nothing), assets_next),
        )
    catch
        return nothing
    end
end

"""
    eval_nn(sol, cfg; compare_egm=true, timestamp=nothing)

Compute summary diagnostics for an NN solver output `sol` under configuration `cfg`.
Returns a named tuple with metrics, diagnostics, hyperparameters, and optional
baseline comparisons.
"""
function eval_nn(sol, cfg; compare_egm::Bool = true, timestamp = nothing)
    ts = timestamp === nothing ? Dates.now() : timestamp
    resid = _get_or(sol, :resid, nothing)
    stats = _residual_stats(resid)
    diagnostics = _build_diagnostics(sol)
    metadata = (
        timestamp = Dates.format(ts, Dates.dateformat"yyyy-mm-ddTHH:MM:SS"),
        residual_shape = resid === nothing ? Int[] : collect(size(resid)),
        grid_size = resid === nothing ? 0 : length(resid),
        has_shocks = _config_has_shocks(cfg) || _get_or(sol, :z_grid, nothing) !== nothing,
    )
    baseline = compare_egm ? _baseline_comparison(sol, cfg) : nothing
    return (
        metadata = metadata,
        metrics = stats,
        diagnostics = diagnostics,
        baseline = baseline,
        hyperparameters = _extract_hyperparams(cfg),
    )
end

function _default_output_dir(cfg)
    eval_cfg = _get_subcfg(cfg, :evaluation)
    return get(eval_cfg, :output_dir, joinpath(pwd(), "results", "nn"))
end

function _default_filename(cfg, ts::Dates.DateTime)
    eval_cfg = _get_subcfg(cfg, :evaluation)
    base = get(eval_cfg, :label, "nn_eval")
    stamp = Dates.format(ts, Dates.dateformat"yyyymmdd_HHMMSS")
    return string(base, "_", stamp, ".json")
end

function _sanitize_for_json(x)
    if x isa NamedTuple
        return Dict(string(k) => _sanitize_for_json(v) for (k, v) in pairs(x))
    elseif x isa Dict
        return Dict(string(k) => _sanitize_for_json(v) for (k, v) in x)
    elseif x isa AbstractArray
        return [_sanitize_for_json(v) for v in x]
    elseif x isa Tuple
        return [_sanitize_for_json(v) for v in x]
    elseif x isa Symbol
        return String(x)
    elseif x isa Real
        return isfinite(x) ? x : nothing
    elseif x === nothing || x isa Bool || x isa String
        return x
    else
        return string(x)
    end
end

"""
    write_eval_report(sol, cfg; output_dir=nothing, filename=nothing, compare_egm=true)

Persist the evaluation report to JSON and return the output path.
"""
function write_eval_report(
    sol,
    cfg;
    output_dir::Union{Nothing,AbstractString} = nothing,
    filename::Union{Nothing,AbstractString} = nothing,
    compare_egm::Bool = true,
)
    ts = Dates.now()
    report = eval_nn(sol, cfg; compare_egm = compare_egm, timestamp = ts)
    out_dir = output_dir === nothing ? _default_output_dir(cfg) : String(output_dir)
    mkpath(out_dir)
    fname = filename === nothing ? _default_filename(cfg, ts) : String(filename)
    path = joinpath(out_dir, fname)
    open(path, "w") do io
        JSON3.write(io, _sanitize_for_json(report); indent = 4)
    end
    return path
end

end # module
