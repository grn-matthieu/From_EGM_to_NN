"""
NNDevice

Utilities to move arrays / parameter trees between CPU and GPU.
This module uses CUDA.jl when available; otherwise it performs no-ops so
code remains CPU-compatible.
"""
module NNDevice

export to_device, move_tree_to_device, is_cuda_available

const HAS_CUDA = try
    @eval begin
        import CUDA
        true
    end
catch
    false
end

is_cuda_available() = HAS_CUDA

"""to_device(x, device)

Move array-like `x` to the specified `device` (Symbol :cpu or :cuda).
If CUDA is not available or `device == :cpu`, returns `x` unchanged.
Works for scalars, Arrays and CuArrays. For NamedTuples/tuples, use
`move_tree_to_device` to recurse.
"""
to_device(x, device::Symbol = :cpu) =
    if device === :cuda && HAS_CUDA
        try
            return CUDA.CuArray(x)
        catch
            # Fall back silently if conversion fails
            return x
        end
    else
        return x
    end

function move_tree_to_device(x, device::Symbol = :cpu)
    if x isa NamedTuple
        return NamedTuple{keys(x)}(move_tree_to_device.(Tuple(values(x)), Ref(device)))
    elseif x isa Tuple
        return tuple(move_tree_to_device.(collect(x), Ref(device))...)
    elseif x isa AbstractArray
        return to_device(x, device)
    else
        return x
    end
end

end # module
