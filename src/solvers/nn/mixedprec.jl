module NNMixedPrecision

export UseFP16,
    UseBF16,
    eltype_from,
    to_mp,
    to_fp32,
    cast_params,
    cast_params!,
    cast_batch,
    with_mixed_precision

abstract type MPType end
struct UseFP16 <: MPType end
struct UseBF16 <: MPType end

eltype_from(::UseFP16) = Float16

# BFloat16 may not be available in all Julia builds; fall back to Float16 when absent
if isdefined(Base, :BFloat16)
    eltype_from(::UseBF16) = Base.BFloat16
else
    @warn "BFloat16 not available in Base; falling back to Float16 for UseBF16"
    eltype_from(::UseBF16) = Float16
end

function _convert_array(x::AbstractArray, ::Type{T}) where {T<:Real}
    y = similar(x, T)
    y .= x
    return y
end

function to_mp(x, ::Type{T}) where {T<:Real}
    x isa AbstractArray ? _convert_array(x, T) : convert(T, x)
end
to_mp(x, mp::MPType) = to_mp(x, eltype_from(mp))

function to_fp32(x)
    x isa AbstractArray ? _convert_array(x, Float32) : Float32(x)
end

function cast_params(params::AbstractVector{<:AbstractArray}, ::Type{T}) where {T<:Real}
    map(p -> _convert_array(p, T), params)
end
cast_params(params::AbstractVector{<:AbstractArray}, mp::MPType) =
    cast_params(params, eltype_from(mp))

function cast_params!(
    dst::AbstractVector{<:AbstractArray},
    src::AbstractVector{<:AbstractArray},
)
    length(dst) == length(src) ||
        throw(ArgumentError("destination and source vectors must have equal length"))
    for i in eachindex(dst, src)
        axes(dst[i]) == axes(src[i]) ||
            throw(ArgumentError("array axes must match for mixed-precision casting"))
        dst[i] .= src[i]
    end
    return dst
end

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
cast_params!(
    dst::AbstractVector{<:AbstractArray},
    src::AbstractVector{<:AbstractArray},
    mp::MPType,
) = cast_params!(dst, src, eltype_from(mp))

_cast_batch(x::AbstractArray, ::Type{T}) where {T<:Real} = _convert_array(x, T)
_cast_batch(x::Nothing, ::Type{T}) where {T<:Real} = nothing
_cast_batch(x::Real, ::Type{T}) where {T<:Real} = convert(T, x)
_cast_batch(x::Tuple, ::Type{T}) where {T<:Real} = map(e -> _cast_batch(e, T), x)
function _cast_batch(x::NamedTuple, ::Type{T}) where {T<:Real}
    vals = map(v -> _cast_batch(v, T), values(x))
    NamedTuple{keys(x)}(Tuple(vals))
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

function cast_batch(batch, ::Type{T}) where {T<:Real}
    _cast_batch(batch, T)
end
cast_batch(batch, mp::MPType) = cast_batch(batch, eltype_from(mp))

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

end # module
