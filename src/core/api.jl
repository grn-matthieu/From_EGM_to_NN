"""
API structs for models, methods, and solutions.
"""
module API

export AbstractModel,
    AbstractMethod,
    Solution,
    get_params,
    get_grids,
    get_shocks,
    get_utility,
    build_model,
    build_method,
    load_config,
    validate_config,
    solve

abstract type AbstractModel end
abstract type AbstractMethod end

# --- Solution specification struct ---
"""
    Solution

Holds the results of a model solution, including policies and diagnostics.
"""
Base.@kwdef struct Solution{M<:AbstractModel,K<:AbstractMethod}
    policy::Dict{Symbol,Any}
    value::Union{Nothing,AbstractArray{Float64}} # Value function
    diagnostics::NamedTuple  # EE stats, iterations, runtime
    metadata::Dict # Model id, method, seed, timestamps
    model::M
    method::K
end

# Generic function stubs
function get_params(x)
    error("get_params not implemented for $(typeof(x))")
end

function get_grids(x)
    error("get_grids not implemented for $(typeof(x))")
end

function get_shocks(x)
    error("get_shocks not implemented for $(typeof(x))")
end

function get_utility(x)
    error("get_utility not implemented for $(typeof(x))")
end

function build_model(x)
    error("build_model factory not implemented for this configuration object")
end

function build_method(x)
    error("build_method factory not implemented for this configuration object")
end

function load_config(x)
    error("load_config not implemented for $(typeof(x)).")
end


function validate_config(x)
    error("validate_config not implemented for $(typeof(x)).")
end


function solve(x)
    error("The function solve is not compatible with $(typeof(x)).")
end

# --- Multi-method convenience ---
# Supported method names (keeps the order deterministic when using :all)
const SUPPORTED_METHODS = (:TimeIteration, :EGM, :Projection, :Perturbation, :NN)

using ..Determinism: derive_rng, promote_master_rng, MasterRNG

# --- Internal helpers for solver orchestration ---
_solver_settings(cfg::NamedTuple) = cfg.solver

function _normalize_methods(cfg::NamedTuple)
    methods = cfg.solver.method
    return methods isa Symbol ? [methods] : collect(methods)
end

function _method_specific_cfg(cfg::NamedTuple, method_name::Symbol)
    solver = _solver_settings(cfg)
    solver_nt = merge(solver, (method = method_name,))
    return merge(cfg, (solver = solver_nt,))
end

function _resolve_master_rng(cfg::NamedTuple, rng)
    if rng !== nothing
        return promote_master_rng(rng)
    end
    master_candidate = cfg.random.master_rng
    master_candidate === nothing && error(
        "master RNG not available; ensure the configuration defines random.seed or pass `rng`.",
    )
    return promote_master_rng(master_candidate)
end

function _execute_solver(
    model::AbstractModel,
    method_obj::AbstractMethod,
    cfg::NamedTuple,
    master::MasterRNG,
    method_name::Symbol,
)
    local_rng = derive_rng(master, string(method_name))
    return solve(model, method_obj, cfg; rng = local_rng)
end

function _log_solver_success(method_name::Symbol, sol::Solution)
    converged = haskey(sol.metadata, :converged) ? sol.metadata[:converged] : nothing
    max_resid = haskey(sol.metadata, :max_resid) ? sol.metadata[:max_resid] : nothing
    runtime = haskey(sol.diagnostics, :runtime) ? sol.diagnostics.runtime : nothing
    runtime === nothing &&
        (runtime = haskey(sol.metadata, :runtime) ? sol.metadata[:runtime] : nothing)
    @info "Finished solver $(method_name). converged=$(converged) max_resid=$(max_resid) runtime=$(runtime)"
end

function _solver_failure_placeholder(
    err,
    model::AbstractModel,
    method_obj::AbstractMethod,
    method_name::Symbol,
)
    @warn "Solver $(method_name) failed; storing error in metadata." err
    @info "Solver $(method_name) failed with error: $(err)"
    return Solution(
        policy = Dict{Symbol,Any}(),
        value = nothing,
        diagnostics = (method = string(method_name), runtime = 0.0),
        metadata = Dict(:error => any(err)),
        model = model,
        method = method_obj,
    )
end

function _dispatch_solvers(
    model::AbstractModel,
    cfg::NamedTuple,
    methods::Vector{Symbol},
    master::MasterRNG,
)
    solutions = Vector{Solution}(undef, length(methods))
    for (i, method_name) in enumerate(methods)
        cfg_m = _method_specific_cfg(cfg, method_name)
        method_obj = build_method(cfg_m)
        @info "Starting solver $(method_name)..."
        try
            sol = _execute_solver(model, method_obj, cfg_m, master, method_name)
            solutions[i] = sol
            _log_solver_success(method_name, sol)
        catch err
            solutions[i] = _solver_failure_placeholder(err, model, method_obj, method_name)
        end
    end
    return solutions
end


# Recursively merge two NamedTuples: keys in `b` override or are merged
# into `a` without erasing nested fields not mentioned in `b`.
function deep_merge(a::NamedTuple, b::NamedTuple)
    res = a
    for k in keys(b)
        vb = getproperty(b, k)
        if hasproperty(res, k)
            va = getproperty(res, k)
            if va isa NamedTuple && vb isa NamedTuple
                res = merge(res, (k => deep_merge(va, vb),))
            else
                res = merge(res, (k => vb,))
            end
        else
            res = merge(res, (k => vb,))
        end
    end
    return res
end

"""
    solve(model::AbstractModel, cfg::NamedTuple)

Run one or more solver methods based on `cfg.solver.method`.
If `cfg.solver.method == :all` (or "all") the function runs all supported
methods in `SUPPORTED_METHODS`. If `cfg.solver.method` is a collection, each
entry is interpreted as a method name (String or Symbol). Returns a
Vector{Solution} with one Solution per requested solver, in the same order.
"""
function solve(model::AbstractModel, cfg::NamedTuple; rng = nothing)
    cfg = validate_config(cfg)
    methods = _normalize_methods(cfg)
    master = _resolve_master_rng(cfg, rng)
    return _dispatch_solvers(model, cfg, methods, master)
end

"""
    solve(cfg::NamedTuple; rng=nothing)

Convenience overload: given a validated configuration NamedTuple, build the
model and run the requested solver(s). If `rng` is not provided, uses
`cfg.random.master_rng` when available (requires `cfg.random.seed`).
Returns either a single `Solution` or a vector of `Solution`s depending on
`cfg.solver.method` (same semantics as `solve(model, cfg)`).
"""
function solve(cfg::NamedTuple; rng = nothing)
    # validate basic structure; this method doesn't enforce random.seed but
    # will honor `rng` when passed, or enrich from cfg.random.seed when present
    cfg = validate_config(cfg)

    methods = _normalize_methods(cfg)
    master = _resolve_master_rng(cfg, rng)

    model = build_model(cfg)
    solutions = _dispatch_solvers(model, cfg, methods, master)
    return length(methods) == 1 ? solutions[1] : solutions
end

"""
    solve(cfg_path::AbstractString; rng=nothing)

Convenience overload: load a config file from disk, build the model, and run
the requested solver(s). Returns one or more `Solution`s following the same
semantics as `solve(model, cfg)`.
"""
function solve(
    cfg_path::AbstractString;
    rng = nothing,
    opts::Union{Nothing,NamedTuple} = nothing,
)
    cfg = load_config(cfg_path)
    # allow callers to programmatically override top-level config fields by
    # passing a NamedTuple `opts`. When `opts` is `nothing` behavior is
    # unchanged. If `opts` contains `use_cuda` propagate it into the
    # nested `cfg.solver.nn.use_cuda` so solver-level device preferences are
    # honored even when the override is provided at top-level.
    if opts !== nothing
        if hasproperty(opts, :use_cuda)
            usecuda = getfield(opts, :use_cuda)
            solver_nt = cfg.solver
            nn_nt = hasproperty(solver_nt, :nn) ? solver_nt.nn : NamedTuple()
            nn_nt = merge(nn_nt, (use_cuda = usecuda,))
            solver_nt = merge(solver_nt, (nn = nn_nt,))
            cfg = merge(cfg, (solver = solver_nt,))
        end
        # Merge opts into cfg without erasing nested data not mentioned in opts
        if opts isa NamedTuple
            cfg = deep_merge(cfg, opts)
        else
            cfg = merge(cfg, opts)
        end
    end
    return solve(cfg; rng = rng)
end

end # module API
