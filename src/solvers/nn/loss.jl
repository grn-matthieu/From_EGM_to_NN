module NNLoss

using ..EulerResiduals: residuals

export check_finite_residuals,
    stabilize_residuals, euler_mse, euler_loss, marg_u, inv_marg_u

"""
    check_finite_residuals(model, policy, batch) -> Bool

Compute Euler-equation residuals on the provided baseline `batch` and return
`true` iff all residual entries are finite (no NaN/Inf).

Logs a short summary containing `max|residual|` and, when applicable, the count
of non-finite entries.
"""
function check_finite_residuals(model, policy, batch)::Bool
    res = residuals(model, policy, batch)

    # Reductions without excessive allocations
    maxabs = maximum(abs, res)
    finite_mask = isfinite.(res)
    all_finite = all(finite_mask)

    if all_finite
        @info "Euler residuals check passed" maxabs = maxabs
    else
        n_nonfinite = length(finite_mask) - sum(finite_mask)
        @warn "Euler residuals contain non-finite entries" nonfinite = n_nonfinite maxabs =
            maxabs
    end

    return all_finite
end

"""
    stabilize_residuals(R; method = :none)

Optionally stabilize heavy‑tailed Euler residuals before computing a loss.

Rationale:
- Heavy‑tailed residuals can dominate MSE and destabilize training.
- A monotone, symmetric transform reduces the influence of outliers while
  preserving ordering and sign information.

Methods:
- `:none`           → returns `R` unchanged.
- `:log1p_square`   → `sign.(R) .* sqrt.(log1p.(R.^2))`
  This applies `log1p` to squared residuals, then takes a signed square‑root.
  It grows ~linearly near 0 and only ~sqrt(log) for large |R|.

When to use: enable for noisy/volatile residuals (e.g., early training,
stochastic shocks) to improve robustness. Disable once training stabilizes if
you prefer a standard MSE objective.
"""
function stabilize_residuals(R; method::Symbol = :none)
    if method === :none
        return R
    elseif method === :log1p_square
        return sign.(R) .* sqrt.(log1p.(R .^ 2))
    else
        throw(ArgumentError("Unknown stabilization method: $(method)"))
    end
end

"""
    euler_mse(R; reduction = :mean)::Float64

Compute a configurable mean-squared-error (MSE) for Euler residuals.

- `R::AbstractArray{<:Real}`: Residuals tensor. Commonly a matrix with
  rows = equations and columns = sample points; vectors, scalars, and
  general N-D arrays are also accepted. All entries are included in the loss.
- `reduction ∈ (:mean, :sum)`: Aggregation across all elements.
  - `:mean` (default): `sum(abs2, R) / length(R)`.
  - `:sum`: `sum(abs2, R)`.

Notes:
- Uses numerically stable reduction `sum(abs2, R)`.
- For `reduction == :mean`, divides by `length(R)`.
- Always returns a `Float64`.

Examples:
    julia> euler_mse([1.0, -2.0])
    2.5

    julia> euler_mse([1, -2]; reduction = :sum)
    5.0
"""
function euler_mse(R::AbstractArray{<:Real}; reduction::Symbol = :mean)::Float64
    (reduction ∈ (:mean, :sum)) ||
        throw(ArgumentError("reduction must be :mean or :sum, got $(reduction)"))
    s = sum(abs2, R)
    s64 = Float64(s)
    return reduction === :mean ? s64 / length(R) : s64
end

"""
    euler_loss(R; reduction = :mean, stabilize::Bool = false, method::Symbol = :log1p_square)

Convenience wrapper that optionally stabilizes residuals before computing MSE.

Behavior:
- If `stabilize` is `true`, applies `stabilize_residuals(R; method)` first.
- Then computes `euler_mse(.; reduction)`.

Typical usage:
- Enable stabilization for heavy‑tailed residuals or early training.
- Use default `reduction = :mean` to match standard MSE.
"""
function euler_loss(
    R;
    reduction::Symbol = :mean,
    stabilize::Bool = false,
    method::Symbol = :log1p_square,
)
    R′ = stabilize ? stabilize_residuals(R; method = method) : R
    return euler_mse(R′; reduction = reduction)
end

"""
    marg_u(c, θ)

CRRA marginal utility u′(c) for consumption `c` and risk aversion σ stored in `θ`.

Conventions:
- If σ ≈ 1: u′(c) = 1 / c
- Else:     u′(c) = c^(-σ)

Numerical safety: clamps `c` below by `eps()` via `max.(c, eps())`.
Accepts scalars and arrays.

Examples:
    julia> marg_u(2.0, (; s = 2.0))
    0.25

    julia> marg_u([1.0, 2.0], (; s = 1.0))
    2-element Vector{Float64}:
     1.0
     0.5
"""
@inline function marg_u(c, θ)
    σ = θ isa Number ? Float64(θ) : Float64(getproperty(θ, :s))
    c_safe = max.(c, eps())
    return isapprox(σ, 1.0; atol = 1e-12) ? 1.0 ./ c_safe : c_safe .^ (-σ)
end

"""
    inv_marg_u(x, θ)

Inverse marginal utility (u′)^{-1}(x) for CRRA with risk aversion σ in `θ`.

Conventions:
- If σ ≈ 1: (u′)^{-1}(x) = 1 / x
- Else:     (u′)^{-1}(x) = x^(-1/σ)

Numerical safety: clamps `x` below by `eps()` via `max.(x, eps())`.
Accepts scalars and arrays.

Examples:
    julia> inv_marg_u(0.25, (; s = 2.0))
    2.0

    julia> c = [0.5, 1.0, 2.0]; θ = (; s = 1.0);
    julia> inv_marg_u(marg_u(c, θ), θ) ≈ c
    true
"""
@inline function inv_marg_u(x, θ)
    σ = θ isa Number ? Float64(θ) : Float64(getproperty(θ, :s))
    x_safe = max.(x, eps())
    return isapprox(σ, 1.0; atol = 1e-12) ? 1.0 ./ x_safe : x_safe .^ (-1.0 / σ)
end

end # module
