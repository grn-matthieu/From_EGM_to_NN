module MethodFactory
import ..API: build_method
using ..EGM: build_egm_method
using ..Perturbation: build_perturbation_method
using ..NN: build_nn_method
"""
    build_method(cfg::AbstractDict)
Construct a method object based on the `:method` entry in `cfg` or
`cfg[:solver][:method]`.
"""
function build_method(cfg::AbstractDict)
    method_name = haskey(cfg, :method) ? cfg[:method] : cfg[:solver][:method]
    method_sym = method_name isa Symbol ? method_name : Symbol(method_name)
    if method_sym == :EGM
        return build_egm_method(cfg)
    elseif method_sym == :Projection
        parent = parentmodule(@__MODULE__)
        if isdefined(parent, :Projection)
            return getfield(parent, :Projection).build_projection_method(cfg)
        else
            error("Projection method not available")
        end
    elseif method_sym == :Perturbation
        return build_perturbation_method(cfg)
    elseif method_sym == :NN
        return build_nn_method(cfg)
    else
        error("Unknown method: $(method_name)")
    end
end
end # module
