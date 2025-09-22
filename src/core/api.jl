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
    error("build_model factory not implemented for configuration of type $(typeof(x))")
end

function build_method(x)
    error("build_method factory not implemented for configuration of type $(typeof(x))")
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

end # module API
