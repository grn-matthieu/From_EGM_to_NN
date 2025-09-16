module EulerResiduals

using ..CommonInterp: interp_linear!
using ..API

export euler_resid_det,
    euler_resid_stoch, euler_resid_det_2, euler_resid_det!, euler_resid_stoch!, residuals

# --- Functions ---

"""
    euler_resid_det!(resid, model_params, c, c_next)

Write absolute Euler equation residuals for the deterministic model into `resid`.
Computations are based on the CRRA utility function.
"""
function euler_resid_det!(
    resid::AbstractVector{<:Real},
    model_params,
    c::AbstractVector{<:Real},
    c_next::AbstractVector{<:Real},
)
    @assert length(resid) == length(c) == length(c_next)

    β = model_params.β
    σ = model_params.σ
    R = 1 + model_params.r

    @inbounds for i in eachindex(resid)
        c_i = clamp(c[i], 1e-12, Inf)
        c_next_i = clamp(c_next[i], 1e-12, Inf)
        resid[i] = abs(1 - β * R * (c_i / c_next_i)^σ)
    end
    return resid
end

"""
    euler_resid_det(model_params, c, c_next)

Allocate and return absolute Euler equation residuals for the deterministic model.
"""
function euler_resid_det(
    model_params,
    c::AbstractVector{<:Real},
    c_next::AbstractVector{<:Real},
)
    resid = similar(c, Float64)
    return euler_resid_det!(resid, model_params, c, c_next)
end

"""
    euler_resid_det(model_params, a_grid, c)

Compute absolute Euler equation residuals for the deterministic model on an
asset grid `a_grid` given a consumption policy `c`. The policy is defined on the
same grid and residuals are obtained by linearly interpolating the policy when
evaluating future consumption. Output is a vector of residuals of length
`length(a_grid)`.
"""
function euler_resid_det_2(
    model_params,
    a_grid::AbstractVector{<:Real},
    c::AbstractVector{<:Real},
)
    Na = length(a_grid)
    @assert length(c) == Na

    R = 1 + model_params.r
    y = getfield(model_params, :y)

    a_min = first(a_grid)
    a_max = last(a_grid)

    a_next = similar(c)
    @. a_next = clamp(R * a_grid + y - c, a_min, a_max)

    c_next = similar(c)
    interp_linear!(c_next, a_grid, c, a_next)

    res = similar(c, Float64)
    euler_resid_det!(res, model_params, c, c_next)
    return res
end


"""
    euler_resid_stoch!(resid, model_params, a_grid, z_grid, Pz, c)

Write absolute Euler equation residuals for the stochastic savings model into `resid`.
The policy `c` is a Na x Nz matrix of consumption. When residuals are computed,
linear interpolation is used on the asset grid.
"""
function euler_resid_stoch!(
    resid::AbstractMatrix{<:Real},
    model_params,
    a_grid::AbstractVector{<:Real},
    z_grid::AbstractVector{<:Real},
    Pz::AbstractMatrix{<:Real},
    c::AbstractMatrix{<:Real},
)
    Na, Nz = size(c)
    @assert size(resid, 1) == Na && size(resid, 2) == Nz
    @assert length(a_grid) == Na
    @assert length(z_grid) == Nz
    @assert size(Pz, 1) == Nz && size(Pz, 2) == Nz

    β = model_params.β
    σ = model_params.σ
    R = (1 + model_params.r)

    # simple scalar linear interpolation helper over a_grid
    lin1(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, xq::Real) = begin
        n = length(x)
        if xq <= x[1]
            return y[1]
        elseif xq >= x[end]
            return y[end]
        else
            j = searchsortedfirst(x, xq)
            j = clamp(j, 2, n)
            x0 = x[j-1]
            x1 = x[j]
            y0 = y[j-1]
            y1 = y[j]
            t = (xq - x0) / (x1 - x0)
            return (1 - t) * y0 + t * y1
        end
    end

    @inbounds for j = 1:Nz
        y = exp(z_grid[j])
        for i = 1:Na
            c_ij = max(c[i, j], 1e-12)
            ap = R * a_grid[i] + y - c_ij
            Emu = 0.0
            for jp = 1:Nz
                cp = lin1(a_grid, view(c, :, jp), ap)
                Emu += Pz[j, jp] * (max(cp, 1e-12) / c_ij)^(-σ)
            end
            resid[i, j] = abs(1.0 - β * R * Emu)
        end
    end
    return resid
end

"""
    euler_resid_stoch(model_params, a_grid, z_grid, Pz, c)

Allocate and return absolute Euler equation residuals for the stochastic savings model.
"""
function euler_resid_stoch(
    model_params,
    a_grid::AbstractVector{<:Real},
    z_grid::AbstractVector{<:Real},
    Pz::AbstractMatrix{<:Real},
    c::AbstractMatrix{<:Real},
)
    res = similar(c, Float64)
    return euler_resid_stoch!(res, model_params, a_grid, z_grid, Pz, c)
end


# --- High-level wrapper API ---

"""
    residuals(model, policy, batch) -> AbstractArray

Compute Euler-equation residuals for a given `model`, `policy`, and `batch` of
evaluation points.

- Deterministic models (no shocks):
  - `batch` is an `AbstractVector` of asset grid points.
  - `policy` is expected to be the standard policy dictionary used in this
    codebase, i.e. `policy[:c]` is a NamedTuple with a `value` vector and an
    optional `grid` vector. If `batch` differs from the policy grid, the
    consumption policy is linearly interpolated onto `batch` before residuals
    are computed.

- Stochastic models (with shocks):
  - `batch` should be a `NamedTuple` or `Dict` with keys `:a_grid`, `:z_grid`,
    and `:Pz`.
  - `policy[:c].value` must be a matrix with size `(length(a_grid), length(z_grid))`.

Returns a vector (deterministic) or a matrix (stochastic) of absolute Euler
residuals.
"""
function residuals(model::API.AbstractModel, policy::Dict{Symbol,Any}, batch)
    S = API.get_shocks(model)
    return S === nothing ? _residuals_det(model, policy, batch) :
           _residuals_stoch(model, policy, batch)
end


# --- Deterministic ---

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

    # Interpolate policy onto requested batch grid if needed
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


# --- Stochastic ---

_haskey_like(x, k) =
    (x isa AbstractDict && haskey(x, k)) || (x isa NamedTuple && hasproperty(x, k))
_get_key(x, k) = x isa AbstractDict ? x[k] : getfield(x, k)

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
