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
    for (i, mname) in enumerate(methods)
        # create a cfg copy with solver.method set to the single method name
        solver_nt = merge(cfg.solver, (method = mname,))
        cfg_m = merge(cfg, (solver = solver_nt,))

        # build method object and dispatch to the per-method solve
        method_m = build_method(cfg_m)
        @info "Starting solver $(mname)..."
        try
            sol = solve(model, method_m, cfg_m)
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

end # module API
