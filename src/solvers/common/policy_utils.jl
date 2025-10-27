module PolicyUtils

using Statistics: mean

export compute_binding_tolerance,
    init_consumption_det,
    enforce_borrowing_constraint!,
    sort_policy_pairs!,
    enforce_strict_increase!,
    enforce_monotone!,
    clamp_policy!,
    ensure_minimum!,
    relaxation_step!,
    rmse_nonbinding

const DEFAULT_CMIN = 1e-12

"""
    compute_binding_tolerance(a_min, a_max, N; floor=1e-10, rel=1e-6)

Compute the tolerance used to detect binding borrowing constraints based on the
asset grid spacing. The tolerance is the larger of `floor` and `rel` times the
average grid step. When the grid has a single node the full range is used.
"""
function compute_binding_tolerance(
    a_min::Real,
    a_max::Real,
    N::Integer;
    floor::Real = 1e-10,
    rel::Real = 1e-6,
)
    Δa = N > 1 ? (a_max - a_min) / (N - 1) : (a_max - a_min)
    return max(floor, rel * Δa)
end

"""
    init_consumption_det(a_grid, a_min, R, y; c_init=nothing, cmin=DEFAULT_CMIN)

Default initializer for deterministic consumption policies. When no starting
values are supplied, the policy is set to half of resources and clamped between
`cmin` and the full resources. When an initial guess is provided the array is
copied to avoid mutating the caller's data.
"""
function init_consumption_det(
    a_grid::AbstractVector,
    a_min::Real,
    R::Real,
    y::Real;
    c_init = nothing,
    cmin::Real = DEFAULT_CMIN,
)
    if c_init === nothing
        resources = similar(a_grid)
        @inbounds for i in eachindex(a_grid)
            res = R * a_grid[i] - a_min + y
            resources[i] = clamp(0.5 * res, cmin, res)
        end
        return resources
    else
        return copy(c_init)
    end
end

"""
    ensure_minimum!(x, minval)

Clamp entries of `x` from below by `minval` in-place.
"""
function ensure_minimum!(x::AbstractArray, minval::Real)
    @inbounds for idx in eachindex(x)
        if x[idx] < minval
            x[idx] = minval
        end
    end
    return x
end

"""
    clamp_policy!(x, cmin, cmax)

Clamp policy vector `x` elementwise between `cmin` and `cmax`. The upper bound
may be either a scalar or an array with the same shape as `x`.
"""
function clamp_policy!(x::AbstractArray, cmin::Real, cmax::Real)
    @inbounds for idx in eachindex(x)
        v = x[idx]
        if v < cmin
            x[idx] = cmin
        elseif v > cmax
            x[idx] = cmax
        end
    end
    return x
end

function clamp_policy!(x::AbstractArray, cmin::Real, cmax::AbstractArray)
    @assert size(x) == size(cmax)
    @inbounds for idx in eachindex(x)
        v = x[idx]
        hi = cmax[idx]
        if v < cmin
            x[idx] = cmin
        elseif v > hi
            x[idx] = hi
        end
    end
    return x
end

"""
    enforce_borrowing_constraint!(a_endo, c_endo, a_min, y, R, a_grid; cmin=DEFAULT_CMIN)

Enforce the borrowing constraint `a_endo ≥ a_min` by clamping offending points
and adjusting the corresponding consumption level.

DEPRECATED: This function only clamps points but doesn't add the constraint point.
Use `add_constraint_point!` instead for proper EGM grid coverage.
"""
function enforce_borrowing_constraint!(
    a_endo::AbstractVector,
    c_endo::AbstractVector,
    a_min::Real,
    y::Real,
    R::Real,
    a_grid::AbstractVector;
    cmin::Real = DEFAULT_CMIN,
)
    @assert length(a_endo) == length(c_endo) == length(a_grid)
    @inbounds for i in eachindex(a_endo)
        if a_endo[i] < a_min
            a_endo[i] = a_min
            # When a' = a_min at current assets a_grid[i], by budget constraint:
            # c = y + R * a_grid[i] - a'
            implied_c = y + R * a_grid[i] - a_min
            c_endo[i] = implied_c <= cmin ? cmin : implied_c
        end
    end
    return a_endo, c_endo
end

"""
    add_constraint_point!(a_endo, c_endo, a_min, y, R; cmin=DEFAULT_CMIN)

Add the borrowing constraint point (a_min, c_at_constraint) to the beginning of
the endogenous grid to ensure proper coverage. This is the standard EGM approach
to handle the borrowing constraint.

At the constraint, consumption is: c = y + R * a_min - a_min = y + (R-1) * a_min
"""
function add_constraint_point!(
    a_endo::AbstractVector,
    c_endo::AbstractVector,
    a_min::Real,
    y::Real,
    R::Real;
    cmin::Real = DEFAULT_CMIN,
)
    # Consumption at the borrowing constraint
    c_at_constraint = max(y + (R - 1) * a_min, cmin)

    # Prepend constraint point
    prepend!(a_endo, [a_min])
    prepend!(c_endo, [c_at_constraint])

    return a_endo, c_endo
end

"""
    sort_policy_pairs!(a_sorted, c_sorted, a_src, c_src)

Sort endogenous grid nodes by assets and copy the sorted pairs into the
pre-allocated buffers `a_sorted` and `c_sorted`.
"""
function sort_policy_pairs!(
    a_sorted::AbstractVector,
    c_sorted::AbstractVector,
    a_src::AbstractVector,
    c_src::AbstractVector,
)
    @assert length(a_sorted) == length(a_src)
    perm = sortperm(a_src)
    @inbounds for (k, idx) in enumerate(perm)
        a_sorted[k] = a_src[idx]
        c_sorted[k] = c_src[idx]
    end
    return a_sorted, c_sorted
end

"""
    enforce_strict_increase!(x; eps=1e-12)

Adjust entries so that `x` becomes strictly increasing. Useful for monotone
interpolation routines that require strictly ordered grids.
"""
function enforce_strict_increase!(x::AbstractVector; eps::Real = 1e-12)
    @inbounds for i = 2:length(x)
        if x[i] <= x[i-1]
            x[i] = x[i-1] + eps
        end
    end
    return x
end

"""
    enforce_monotone!(x; eps=1e-12)

Force the vector `x` to be weakly increasing by nudging downward violations.
"""
function enforce_monotone!(x::AbstractVector; eps::Real = 1e-12)
    @inbounds for i = 2:length(x)
        if x[i] < x[i-1]
            x[i] = x[i-1] + eps
        end
    end
    return x
end

"""
    relaxation_step!(current, previous, proposal, relax)

Blend `previous` and `proposal` using relaxation parameter `relax` and store the
result in `current`. Returns the infinity-norm difference between `current` and
`previous` (before overwriting).
"""
function relaxation_step!(
    current::AbstractArray,
    previous::AbstractArray,
    proposal::AbstractArray,
    relax::Real,
)
    maxdiff = zero(eltype(current))
    @inbounds for idx in eachindex(current, previous, proposal)
        oldval = previous[idx]
        newval = (1 - relax) * oldval + relax * proposal[idx]
        current[idx] = newval
        diff = abs(newval - oldval)
        if diff > maxdiff
            maxdiff = diff
        end
    end
    return maxdiff
end

"""
    rmse_nonbinding(resid, a_next, a_min, bind_tol)

Compute the root mean squared Euler residual on non-binding points. If all
points are binding the metric falls back to the full residual set.
"""
function rmse_nonbinding(
    resid::AbstractArray,
    a_next::AbstractArray,
    a_min::Real,
    bind_tol::Real,
)
    threshold = a_min + bind_tol
    acc = zero(eltype(resid))
    count = 0
    @inbounds for idx in eachindex(resid, a_next)
        if a_next[idx] > threshold
            val = resid[idx]
            acc += val * val
            count += 1
        end
    end
    if count == 0
        @inbounds for val in resid
            acc += val * val
        end
        count = length(resid)
    end
    return sqrt(acc / count)
end

end # module
