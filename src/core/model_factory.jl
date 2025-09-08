module ModelFactory

import ..API: build_model

using ..ModelContract
using ..ConsumerSaving: build_cs_model

"""
    build_model(config::Dict)

Dispatches to the appropriate model-building function based on the `name` key in the config dictionary.
"""
function build_model(cfg::AbstractDict)
    @assert haskey(cfg, :model) ":model key not found"
    @assert haskey(cfg[:model], :name) ":name key not found in :model"
    model_name = Symbol(cfg[:model][:name])
    return _build_model(Val(model_name), cfg)
end


# Example dispatch for different model types
function _build_model(::Val{:cs}, cfg)
    return build_cs_model(cfg)
end


# Fallback for unknown model types
function _build_model(model_name::Symbol, cfg)
    error("Unknown model type: $(model_name)")
end

end # module
