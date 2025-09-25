module MethodFactory
import ..API: build_method
using ..EGM: build_egm_method
using ..Perturbation: build_perturbation_method
using ..Projection: build_projection_method
using ..NN: build_nn_method
using ..TimeIteration: build_timeiteration_method
using ..UtilsConfig: maybe

# NN method loaded conditionally in build_method to avoid load-order cycles
"""
    build_method(cfg::NamedTuple)

Construct a method object based on `cfg.method` or `cfg.solver.method`.
"""
function build_method(cfg::NamedTuple)
    method_name = maybe(cfg, :method, cfg.solver.method)
    method_sym = method_name isa Symbol ? method_name : Symbol(method_name)
    if method_sym == :EGM
        return build_egm_method(cfg)
    elseif method_sym == :Projection
        return build_projection_method(cfg)
    elseif method_sym == :Perturbation
        return build_perturbation_method(cfg)
    elseif method_sym == :NN
        return build_nn_method(cfg)
    elseif method_sym == :TimeIteration || method_sym == :TI
        return build_timeiteration_method(cfg)
    else
        error("Unknown method: $(method_name)")
    end
end
end # module
