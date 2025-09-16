module NNLoss

using ..API
using ..CommonInterp: interp_linear!
using ..EulerResiduals: euler_resid_det_2, euler_resid_stoch!

export EulerResidual

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
    res = _compute_residuals(model, policy, batch)

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

"""
    EulerResidual(a, y; θ, policy, sampler; nMC::Integer=1)

Computes the Euler-equation residual u′(c) − β * E[u′(c′) * R′] by Monte Carlo,
vectorized over a batch of state points `(a, y)`.

Assumptions and conventions:
- CRRA marginal utility via `marg_u(c, θ)` with risk aversion stored in `θ.s`.
- Budget: `c = max(resources(a, y; θ) − a′, eps())`, where `a′ = policy(a, y; θ)`.
- One-step dynamics per draw ε: `y′ = T(y, ε; θ)`, `R′ = R(a′, y′; θ)`,
  `a″ = policy(a′, y′; θ)`, `c′ = max(resources(a′, y′; θ) − a″, eps())`.
- Residual: `marg_u(c, θ) − θ.β * mean_k( marg_u(c′_k, θ) .* R′_k )`.

Shapes and types:
- Accepts scalars or arrays for `a` and `y`; broadcasting defines the batch.
- Returns an array matching the broadcasted shape of `a` and `y` (Float64).
- No global RNG side effects: randomness is delegated to `sampler`.

Sampler interface:
- If `nMC == 1`, a sampler that returns a single `ε` (i.e., `sampler()`)
  is supported. If `sampler(n::Int)` is defined, it may also be used.
- If `nMC > 1`, either implement `sampler(n::Int)` returning an iterable of
  `ε` draws of length `n`, or implement `sampler()` to return a fresh draw
  and it will be called `nMC` times.

Notes:
- Uses small in-place buffers to avoid large temporaries when feasible.
- Clamps consumption with `eps()` to ensure finite marginal utilities.
"""
function EulerResidual(a, y; θ, policy, sampler, nMC::Integer = 1)
    # First-stage: compute a′ and c for the current state
    ap = policy.(a, y; θ = θ)
    c = max.(resources.(a, y; θ = θ) .- ap, eps())
    mu = marg_u(c, θ)

    # Allocate accumulation buffer in Float64 with broadcasted shape
    EmuR = zero.(Float64.(mu))

    # Reusable buffers for next-period computations
    yp = similar(EmuR)
    ap2 = similar(EmuR)
    cp = similar(EmuR)
    Rp = similar(EmuR)

    # Helper: draw an iterator of ε of length nMC
    _draws(n::Int) = begin
        if hasmethod(sampler, Tuple{Int})
            sampler(n)
        else
            # Fallback: materialize by repeated scalar draws
            (sampler() for _ = 1:n)
        end
    end

    if nMC == 1
        # Single-draw path supporting samplers that return one ε
        ε_iter = _draws(1)
        @inbounds for ε in ε_iter
            yp .= T.(y, ε; θ = θ)
            Rp .= R.(ap, yp; θ = θ)
            ap2 .= policy.(ap, yp; θ = θ)
            cp .= max.(resources.(ap, yp; θ = θ) .- ap2, eps())
            EmuR .+= marg_u(cp, θ) .* Rp
        end
    else
        # Multi-draw Monte Carlo
        ε_iter = _draws(nMC)
        @inbounds for ε in ε_iter
            yp .= T.(y, ε; θ = θ)
            Rp .= R.(ap, yp; θ = θ)
            ap2 .= policy.(ap, yp; θ = θ)
            cp .= max.(resources.(ap, yp; θ = θ) .- ap2, eps())
            EmuR .+= marg_u(cp, θ) .* Rp
        end
        EmuR ./= nMC
    end

    return Float64.(mu) .- Float64(getproperty(θ, :β)) .* EmuR
end

_haskey_like(x, k) =
    (x isa AbstractDict && haskey(x, k)) || (x isa NamedTuple && hasproperty(x, k))
_get_key(x, k) = x isa AbstractDict ? x[k] : getfield(x, k)

function _compute_residuals(model::API.AbstractModel, policy::Dict{Symbol,Any}, batch)
    S = API.get_shocks(model)
    return S === nothing ? _residuals_det(model, policy, batch) :
           _residuals_stoch(model, policy, batch)
end

function _residuals_det(
    model::API.AbstractModel,
    policy::Dict{Symbol,Any},
    a_batch::AbstractVector{<:Real},
)
    p = API.get_params(model)
    @assert haskey(policy, :c) "policy[:c] not found"
    c_entry = policy[:c]
    c = getfield(c_entry, :value)
    ag = hasproperty(c_entry, :grid) ? getfield(c_entry, :grid) : nothing
    @assert c isa AbstractVector "policy[:c].value must be a vector for deterministic residuals"
    if ag === nothing || (length(ag) == length(a_batch) && ag === a_batch)
        c_eval = c
        a_grid = a_batch
    else
        c_eval = similar(a_batch, Float64)
        interp_linear!(c_eval, ag, c, a_batch)
        a_grid = a_batch
    end
    return euler_resid_det_2(p, a_grid, c_eval)
end

function _residuals_stoch(model::API.AbstractModel, policy::Dict{Symbol,Any}, batch)
    @assert _haskey_like(batch, :a_grid) &&
            _haskey_like(batch, :z_grid) &&
            _haskey_like(batch, :Pz) "batch must provide :a_grid, :z_grid, and :Pz for stochastic residuals"
    a_grid = _get_key(batch, :a_grid)
    z_grid = _get_key(batch, :z_grid)
    Pz = _get_key(batch, :Pz)
    p = API.get_params(model)
    @assert haskey(policy, :c) "policy[:c] not found"
    c_entry = policy[:c]
    c = getfield(c_entry, :value)
    Na = length(a_grid)
    Nz = length(z_grid)
    @assert c isa AbstractMatrix && size(c, 1) == Na && size(c, 2) == Nz "policy[:c].value must be a (Na, Nz) matrix matching batch grids"
    res = similar(c, Float64)
    euler_resid_stoch!(res, p, a_grid, z_grid, Pz, c)
    return res
end

end # module
