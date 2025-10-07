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

    cfg_master =
        hasproperty(cfg, :random) && hasproperty(cfg.random, :master_rng) ?
        promote_master_rng(cfg.random.master_rng) : nothing
    master = rng === nothing ? cfg_master : promote_master_rng(rng)
    master === nothing &&
        error("No master RNG available; pass `rng` or ensure config.random.seed is set.")

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
    local_cfg = cfg
    if hasproperty(cfg, :random)
        r = cfg.random
        has_master = hasproperty(r, :master_rng) && r.master_rng isa MasterRNG
        if !has_master && hasproperty(r, :seed) && r.seed !== nothing
            master = make_master_rng(r.seed)
            # normalize seed to UInt64 for consistency with loader
            seed_uint = UInt64(r.seed)
            r2 = merge(r, (seed = seed_uint, master_rng = master))
            local_cfg = merge(cfg, (random = r2,))
        end
    end

    # Decide if a single method is requested to return a single Solution
    requested = local_cfg.solver.method
    single = false
    if requested === :all || requested == "all"
        single = false
    elseif requested isa AbstractVector
        single = length(requested) == 1
    else
        single = true
    end

    model = build_model(local_cfg)
    if single
        # Normalize the single method symbol and call the per-method solve
        mname =
            requested isa AbstractVector ?
            (requested[1] isa Symbol ? requested[1] : Symbol(requested[1])) :
            (requested isa Symbol ? requested : Symbol(requested))
        solver_nt = merge(local_cfg.solver, (method = mname,))
        cfg_m = merge(local_cfg, (solver = solver_nt,))
        method_m = build_method(cfg_m)
        local_master =
            rng === nothing &&
            hasproperty(local_cfg, :random) &&
            hasproperty(local_cfg.random, :master_rng) ? local_cfg.random.master_rng : rng
        return solve(model, method_m, cfg_m; rng = local_master)
    else
        return solve(model, local_cfg; rng = rng)
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
        cfg = merge(cfg, opts)
    end
    return solve(cfg; rng = rng)
end

end # module API
