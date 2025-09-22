"""
ModelFactory

Constructs concrete model instances from configuration via `API.build_model`.
"""
module ModelFactory

import ..API: build_model

using ..ModelContract
using ..ConsumerSaving: build_cs_model

_to_namedtuple(x::NamedTuple) = x
_to_namedtuple(x::AbstractDict) =
    (; (Symbol(k) => _to_namedtuple(v) for (k, v) in pairs(x))...)
_to_namedtuple(x::AbstractVector) = _to_namedtuple.(x)
_to_namedtuple(x) = x

"""
    build_model(cfg::NamedTuple)

Dispatches to the appropriate model-building function based on `cfg.model.name`.
"""
function build_model(cfg::NamedTuple)
    @assert hasproperty(cfg, :model) "configuration is missing a `model` entry"
    model_cfg = cfg.model
    @assert hasproperty(model_cfg, :name) "configuration.model is missing `name`"
    model_name = Symbol(model_cfg.name)
    return _build_model(Val(model_name), cfg)
end

function build_model(cfg::AbstractDict)
    return build_model(_to_namedtuple(cfg))
end


# Example dispatch for different model types
function _build_model(::Val{:cs}, cfg::NamedTuple)
    return build_cs_model(cfg)
end


# Fallback for unknown model types
function _build_model(model_name::Symbol, cfg::NamedTuple)
    error("Unknown model type: $(model_name)")
end

end # module
