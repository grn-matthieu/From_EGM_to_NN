"""
NNUtils

Shared small helpers used across NN training modules: tree traversal for
parameter/grads, grad-norm, scaling, and safe array-tree copying. Centralizing
these reduces duplication and keeps behavior consistent.
"""
module NNUtils

# Graceful, optional interoperability with Flux/Functors when available.
const HAS_FLUX = try
    @eval begin
        import Flux
        true
    end
catch
    false
end

const HAS_FUNCTORS = try
    @eval begin
        import Functors
        true
    end
catch
    false
end

export foreach_array_leaf,
    collect_array_leaves,
    collect_params_leaves,
    grad_global_l2norm,
    grad_global_l2norm_params,
    scale_grads!,
    _copy_tree_arrays!,
    to_fp32

"""Apply function `f(::AbstractArray)` to each array leaf in a nested tree."""
function foreach_array_leaf(x, f::F) where {F}
    if x isa NamedTuple
        for v in values(x)
            foreach_array_leaf(v, f)
        end
    elseif x isa Tuple
        for v in x
            foreach_array_leaf(v, f)
        end
    elseif x isa AbstractArray
        f(x)
    elseif x === nothing
        return
    else
        return
    end
end

"""Collect all array leaves into a Vector{AbstractArray}."""
function collect_array_leaves(x)
    acc = Vector{AbstractArray}()
    foreach_array_leaf(x) do a
        push!(acc, a)
    end
    return acc
end

"""
    collect_params_leaves(obj)

If `Flux` is available and `obj` exposes parameters via `Flux.params(obj)`,
collect those arrays; otherwise fall back to `collect_array_leaves`.
"""
function collect_params_leaves(obj)
    if HAS_FLUX
        try
            ps = Flux.params(obj)
            acc = Vector{AbstractArray}()
            for p in ps
                push!(acc, p)
            end
            return acc
        catch
            # flux.params may not accept this object; fallback
            return collect_array_leaves(obj)
        end
    else
        return collect_array_leaves(obj)
    end
end

"""Compute global L2 norm of a gradient tree (sum of leaf Frobenius norms)."""
function grad_global_l2norm(grads)::Float64
    s = 0.0
    foreach_array_leaf(grads) do g
        s += sum(abs2, g)
    end
    return sqrt(s)
end

"""
    grad_global_l2norm_params(obj)

Compute the global L2 norm for parameter containers. If `Flux` is available
and `Flux.params(obj)` works, it will be used; otherwise falls back to
`grad_global_l2norm` which expects a nested tree of arrays.
"""
function grad_global_l2norm_params(obj)::Float64
    if HAS_FLUX
        try
            s = 0.0
            for p in Flux.params(obj)
                s += sum(abs2, p)
            end
            return sqrt(s)
        catch
            return grad_global_l2norm(obj)
        end
    else
        return grad_global_l2norm(obj)
    end
end

"""Scale all array leaves by factor `α` in place."""
function scale_grads!(grads, α::Real)
    foreach_array_leaf(grads) do g
        @. g = α * g
    end
    return grads
end

# Recursively copy array contents from src -> dest (matching structure)
function _copy_tree_arrays!(dest, src)
    if dest isa NamedTuple && src isa NamedTuple
        for k in keys(dest)
            _copy_tree_arrays!(getfield(dest, k), getfield(src, k))
        end
    elseif dest isa Tuple && src isa Tuple
        for i in eachindex(dest)
            _copy_tree_arrays!(dest[i], src[i])
        end
    elseif dest isa AbstractArray && src isa AbstractArray
        @assert size(dest) == size(src)
        dest .= src
    else
        # numbers or unsupported leaves are ignored
    end
    return dest
end

"""
    to_fp32(x)

Convert numeric arrays or scalars to `Float32`. Used where reductions are
performed in single-precision for numerical safety.
"""
function to_fp32(x)
    if x isa AbstractArray
        return convert.(Float32, x)
    else
        return Float32(x)
    end
end

end # module NNUtils
