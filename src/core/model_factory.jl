"""
ModelFactory

Constructs concrete model instances from configuration via `API.build_model`.
"""
module ModelFactory

import ..API: build_model

using ..ModelContract
using ..ConsumerSaving: build_cs_model

"""
    build_model(cfg::NamedTuple)

Dispatches to the appropriate model-building function based on `cfg.model.name`.
"""
function build_model(cfg::NamedTuple)
    model_name = Symbol(cfg.model.name)
    return _build_model(Val(model_name), cfg)
end


# Example dispatch for different model types
function _build_model(::Val{:cs}, cfg::NamedTuple)
    return build_cs_model(cfg)
end


# Fallback for unknown model types
function _build_model(model_name::Val, cfg::NamedTuple)
    error("Unknown model type: $(model_name)")
end

end # module
