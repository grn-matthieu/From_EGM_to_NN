
module NNMixedPrecision

UseFP16,
UseBF16,
eltype_from,
to_mp,
to_fp32,
cast_params,
cast_params!,
cast_batch,
with_mixed_precision
"""
    with_mixed_precision(model, params_master, batch; mp::Union{Nothing,MPType}=nothing, loss_scale::Real=1.0) do (params_mp, batch_mp)
        # yields MP views/copies when mp≠nothing, else FP32 passthrough
    end
"""
function with_mixed_precision(
    model,
    params_master,
    batch;
    mp::Union{Nothing,MPType} = nothing,
    loss_scale::Real = 1.0,
    f,
)
    if mp === nothing
        return f(params_master, batch)
    else
        params_mp = cast_params(params_master, eltype_from(mp))
        batch_mp = cast_batch(batch, eltype_from(mp))
        return f(params_mp, batch_mp)
    end
end

"""
    abstract type MPType

Marker supertype for mixed-precision policies used when running NN forward
passes. Concrete implementations select the element type used for model
parameters and batches.
"""
abstract type MPType end

"""
    struct UseFP16 <: MPType end

Select IEEE Float16 for mixed-precision execution.
"""
struct UseFP16 <: MPType end

"""
    struct UseBF16 <: MPType end

Select BFloat16 for mixed-precision execution.
"""
struct UseBF16 <: MPType end

"""
    eltype_from(mp::MPType) -> Type

Return the floating-point element type associated with the mixed-precision
selector `mp`.
"""
eltype_from(::UseFP16) = Float16
eltype_from(::UseBF16) = BFloat16

"""
    to_mp(x, ::Type{T})

Convert `x` to mixed-precision type `T`. Arrays preserve their shape and device
by allocating with `similar(x, T)` before copying values.
"""
function to_mp(x, ::Type{T}) where {T<:Real}
    if x isa AbstractArray
        return _convert_array(x, T)
    else
        return convert(T, x)
    end
end

"""
    to_mp(x, mp::MPType)

Shorthand that selects the element type from `mp`.
"""
to_mp(x, mp::MPType) = to_mp(x, eltype_from(mp))

"""
    to_fp32(x)

Convert `x` (scalar or array) to `Float32` while preserving shape and device.
"""
function to_fp32(x)
    if x isa AbstractArray
        return _convert_array(x, Float32)
    else
        return Float32(x)
    end
end

"""
    cast_params(params, ::Type{T}) -> Vector

Convert each parameter array in `params` to element type `T`, preserving their
shape and device. Returns a new vector of arrays.
"""
function cast_params(params::AbstractVector{<:AbstractArray}, ::Type{T}) where {T<:Real}
    converted = map(params) do p
        _convert_array(p, T)
    end
    return converted
end

"""
    cast_params(params, mp::MPType)

Shorthand selecting the element type from `mp`.
"""
cast_params(params::AbstractVector{<:AbstractArray}, mp::MPType) =
    cast_params(params, eltype_from(mp))

"""
    cast_params!(dst, src)

In-place version of [`cast_params`](@ref). Each destination array in `dst` must
already be allocated with the desired element type and matching axes. Entries
are overwritten with the converted values from `src` using broadcast assignment
to avoid intermediate allocations.
"""
function cast_params!(
    dst::AbstractVector{<:AbstractArray},
    src::AbstractVector{<:AbstractArray},
)
    length(dst) == length(src) ||
        throw(ArgumentError("destination and source vectors must have equal length"))
    for i in eachindex(dst, src)
        d = dst[i]
        s = src[i]
        axes(d) == axes(s) ||
            throw(ArgumentError("array axes must match for mixed-precision casting"))
        d .= s
    end
    return dst
end

"""
    cast_params!(dst, src, ::Type{T})

Ensure `dst` contains arrays of element type `T` before copying values from
`src`. Convenience wrapper that combines allocation and conversion when `dst`
contains uninitialised arrays.
"""
function cast_params!(
    dst::AbstractVector{<:AbstractArray},
    src::AbstractVector{<:AbstractArray},
    ::Type{T},
) where {T<:Real}
    length(dst) == length(src) ||
        throw(ArgumentError("destination and source vectors must have equal length"))
    for i in eachindex(dst, src)
        if !(eltype(dst[i]) == T && axes(dst[i]) == axes(src[i]))
            dst[i] = similar(src[i], T)
        end
    end
    return cast_params!(dst, src)
end

"""
    cast_params!(dst, src, mp::MPType)

Shorthand that derives the element type from `mp`.
"""
cast_params!(
    dst::AbstractVector{<:AbstractArray},
    src::AbstractVector{<:AbstractArray},
    mp::MPType,
) = cast_params!(dst, src, eltype_from(mp))

"""
    cast_batch(batch, ::Type{T})

Recursively convert every array-valued field in `batch` to element type `T`.
Handles arrays, numbers, tuples, named tuples, and simple structs with default
constructors. Shapes and array storage order are preserved via `similar`.
"""
function cast_batch(batch, ::Type{T}) where {T<:Real}
    return _cast_batch(batch, T)
end

"""
    cast_batch(batch, mp::MPType)

Shorthand that selects the element type from `mp`.
"""
cast_batch(batch, mp::MPType) = cast_batch(batch, eltype_from(mp))

# --- Internal helpers ---

function _convert_array(x::AbstractArray, ::Type{T}) where {T<:Real}
    y = similar(x, T)
    y .= x
    return y
end

_cast_batch(x::AbstractArray, ::Type{T}) where {T<:Real} = _convert_array(x, T)
_cast_batch(x::Nothing, ::Type{T}) where {T<:Real} = nothing
_cast_batch(x::Real, ::Type{T}) where {T<:Real} = convert(T, x)
_cast_batch(x::Tuple, ::Type{T}) where {T<:Real} = map(e -> _cast_batch(e, T), x)
function _cast_batch(x::NamedTuple, ::Type{T}) where {T<:Real}
    vals = map(v -> _cast_batch(v, T), values(x))
    return NamedTuple{keys(x)}(Tuple(vals))
end
function _cast_batch(x::S, ::Type{T}) where {S,T<:Real}
    if !Base.isstructtype(S)
        return x
    elseif S <: Number || S <: AbstractArray
        return convert(T, x)
    end
    field_syms = fieldnames(S)
    converted = ntuple(i -> _cast_batch(getfield(x, field_syms[i]), T), length(field_syms))
    return S(converted...)
end

end # module
