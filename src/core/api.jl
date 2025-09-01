"""
API structs for models, methods, and solutions.
"""
module API

export AbstractModel, AbstractMethod, Solution

abstract type AbstractModel end
abstract type AbstractMethod end

# --- Solution specification struct ---
"""
    Solution

Holds the results of a model solution, including policies and diagnostics.
"""
Base.@kwdef struct Solution{M<:AbstractModel, K<:AbstractMethod}
    policy::NamedTuple
    value::Union{Nothing,Float64}
    diagnostics::NamedTuple  # EE stats, iterations, runtime
    metadata::Dict{Symbol,Any} # Model id, method, seed, timestamps
    model::M
    method::K
end

end # module API