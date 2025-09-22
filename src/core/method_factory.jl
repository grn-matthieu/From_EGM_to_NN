module MethodFactory
import ..API: build_method
using ..EGM: build_egm_method
using ..Perturbation: build_perturbation_method
using ..Projection: build_projection_method
using ..NN: build_nn_method

_to_namedtuple(x::NamedTuple) = x
_to_namedtuple(x::AbstractDict) =
    (; (Symbol(k) => _to_namedtuple(v) for (k, v) in pairs(x))...)
_to_namedtuple(x::AbstractVector) = _to_namedtuple.(x)
_to_namedtuple(x) = x

# NN method loaded conditionally in build_method to avoid load-order cycles
"""
    build_method(cfg::NamedTuple)

Construct a method object based on `cfg.method` or `cfg.solver.method`.
"""
function build_method(cfg::NamedTuple)
    method_name = hasproperty(cfg, :method) ? cfg.method : cfg.solver.method
    method_sym = method_name isa Symbol ? method_name : Symbol(method_name)
    if method_sym == :EGM
        return build_egm_method(cfg)
    elseif method_sym == :Projection
        return build_projection_method(cfg)
    elseif method_sym == :Perturbation
        return build_perturbation_method(cfg)
    elseif method_sym == :NN
        return build_nn_method(cfg)
    else
        error("Unknown method: $(method_name)")
    end
end

function build_method(cfg::AbstractDict)
    return build_method(_to_namedtuple(cfg))
end
end # module
