"""
NNConstraints

Projection and clipping utilities tailored for NN policy outputs (e.g., mapping
unconstrained outputs to feasible savings `a′ ≥ a_min`).
"""
module NNConstraints

export softplus, project_savings_softplus, project_savings_clip, project_savings, smooth_pos

"""
    softplus(x)

Overflow-safe softplus function.

Definition:
- `softplus(x) = log1p(exp(-abs(x))) + max(x, 0)`

Accepts scalars and arrays (via broadcasting as `softplus.(...)`).
Preserves the input element type for typical Numeric types.

Examples:
    julia> softplus(-10.0) > 0
    true

    julia> softplus.(Float32.([-1, 0, 1])) |> eltype
    Float32
"""
@inline softplus(x) = log1p(exp(-abs(x))) + max(x, 0)

"""
    project_savings_softplus(ap_raw, a_min)

Project raw next-period assets `a'` to enforce the borrowing constraint
`a' ≥ a_min` using a softplus transform.

Formula:
- `a_proj = a_min .+ softplus.(ap_raw .- a_min)`

Accepts scalars and arrays for `ap_raw`; typically `a_min` is a scalar.
Attempts to preserve the element type of `ap_raw` for arrays and scalars.

Examples:
    julia> project_savings_softplus(-1.0, 0.0) ≥ 0.0
    true

    julia> ap = Float32.([-2, 0, 3]); a_min = 0.5f0;
    julia> proj = project_savings_softplus(ap, a_min);
    julia> minimum(proj) ≥ a_min - 1f-6 && eltype(proj) == Float32
    true
"""
@inline function project_savings_softplus(ap_raw::AbstractArray{T}, a_min) where {T}
    aminT = convert(T, a_min)
    return aminT .+ softplus.(ap_raw .- aminT)
end

@inline function project_savings_softplus(ap_raw::T, a_min) where {T<:Number}
    aminT = convert(T, a_min)
    return aminT + softplus(ap_raw - aminT)
end

"""
    project_savings_clip(ap_raw, a_min)

Hard clipping projection to enforce `a' ≥ a_min`.

Returns `max.(ap_raw, a_min)` for arrays and scalars, preserving element types
when possible by converting `a_min` to the element type of `ap_raw`.
"""
@inline function project_savings_clip(ap_raw::AbstractArray{T}, a_min) where {T}
    aminT = convert(T, a_min)
    return max.(ap_raw, aminT)
end

@inline function project_savings_clip(ap_raw::T, a_min) where {T<:Number}
    aminT = convert(T, a_min)
    return max(ap_raw, aminT)
end

"""
    project_savings(ap_raw, a_min; kind::Symbol = :softplus)

Unified API for projecting `a'` to satisfy the borrowing constraint.

Kinds:
- `:softplus` (default): smooth projection via `project_savings_softplus`.
- `:clip`: hard clipping via `project_savings_clip`.
"""
@inline function project_savings(ap_raw, a_min; kind::Symbol = :softplus)
    return kind === :softplus ? project_savings_softplus(ap_raw, a_min) :
           kind === :clip ? project_savings_clip(ap_raw, a_min) :
           error("unknown projection kind")
end

"""
    smooth_pos(x; eps=1e-8, beta=20.0)

Smooth positivity helper that returns elementwise values strictly greater than
`eps` while keeping non-zero derivatives. Implemented using the module
`softplus` with a sharpness parameter `beta` by scaling the input and
rescaling the output to keep magnitudes comparable.

Callers that previously used a hard lower bound of `1e-12` can pass
`eps=1e-12` to preserve the same numerical floor while gaining smooth
gradients.
"""
@inline function smooth_pos(x; eps::Float64 = 1e-8, beta::Float64 = 20.0)
    return eps .+ (softplus.(beta .* x) ./ beta)
end

end # module
