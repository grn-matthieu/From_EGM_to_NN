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
    metadata::Dict{Symbol,Any} # Model id, method, seed, timestamps
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

using ..Determinism: derive_rng, promote_master_rng, MasterRNG, make_master_rng

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
    # extract requested method(s)
    if !haskey(cfg, :solver)
        error("Configuration must contain a `solver` section with a `method` field.")
    end
    requested = cfg.solver.method

    # normalize to vector of Symbols
    methods::Vector{Symbol} = Vector{Symbol}()
    if requested === :all || requested == "all"
        methods = collect(SUPPORTED_METHODS)
    elseif requested isa AbstractVector
        for m in requested
            push!(methods, m isa Symbol ? m : Symbol(m))
        end
    else
        push!(methods, requested isa Symbol ? requested : Symbol(requested))
    end

    solutions = Vector{Solution}(undef, length(methods))

    master = cfg.random.master_rng

    for (i, mname) in enumerate(methods)
        # create a cfg copy with solver.method set to the single method name
        solver_nt = merge(cfg.solver, (method = mname,))
        cfg_m = merge(cfg, (solver = solver_nt,))

        # build method object and dispatch to the per-method solve
        method_m = build_method(cfg_m)
        @info "Starting solver $(mname)..."
        try
            local_rng = derive_rng(master, string(mname))
            sol = solve(model, method_m, cfg_m; rng = local_rng)
            solutions[i] = sol
            # try to extract some diagnostics for the finish message
            converged =
                haskey(sol.metadata, :converged) ? sol.metadata[:converged] : nothing
            max_resid =
                haskey(sol.metadata, :max_resid) ? sol.metadata[:max_resid] : nothing
            runtime =
                haskey(sol.diagnostics, :runtime) ? sol.diagnostics.runtime :
                (haskey(sol.metadata, :runtime) ? sol.metadata[:runtime] : nothing)
            @info "Finished solver $(mname). converged=$(converged) max_resid=$(max_resid) runtime=$(runtime)"
        catch err
            @warn "Solver $(mname) failed; storing error in metadata." err
            @info "Solver $(mname) failed with error: $(err)"
            # create a minimal Solution-like placeholder capturing the error
            # Use the local API.Solution constructor signature
            dummy = Solution(
                policy = Dict{Symbol,Any}(),
                value = nothing,
                diagnostics = (method = string(mname), runtime = 0.0),
                metadata = Dict(:error => string(err)),
                model = model,
                method = method_m,
            )
            solutions[i] = dummy
        end
    end

    return solutions
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
    validate_config(cfg)

    # Enrich cfg with a MasterRNG if a seed is available and a master isn't
    master = rng !== nothing ? cfg.random.master_rng : make_master_rng(rng)

    # Decide if a single method is requested to return a single Solution
    requested = cfg.solver.method
    single = false
    if requested == "all"
        single = false
    elseif requested isa AbstractVector
        single = length(requested) == 1
    else
        single = true
    end

    model = build_model(cfg)
    if single
        method = build_method(cfg)
        return solve(model, method, cfg; rng = master)
    else
        return solve(model, cfg; rng = master)
    end
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
            solver_nt = hasproperty(cfg, :solver) ? cfg.solver : NamedTuple()
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
