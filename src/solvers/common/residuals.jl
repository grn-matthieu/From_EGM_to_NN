module EulerResiduals

using ..CommonInterp: interp_linear
using ..API
using Zygote

export euler_resid_det, euler_resid_stoch
export euler_resid_det!, euler_resid_stoch!
export euler_resid_det_grid, euler_resid_stoch_grid

# helpers
@inline _T(args...) = promote_type(map(eltype, args)...)
@inline _eps(::Type{T}) where {T<:AbstractFloat} = T(1e-12)

# -------------------------
# Deterministic (pure, AD-safe)
# -------------------------

"""
    euler_resid_det(params, c, c_next)

Absolute Euler residuals given current and next consumption.
Pure. Zygote-compatible.
"""
function euler_resid_det(params, c::AbstractVector, c_next::AbstractVector)
    @assert length(c) == length(c_next)
    T = _T(c, c_next)
    β = T(params.β)
    σ = T(params.σ)
    R = T(1) + T(params.r)
    ϵ = _eps(T)
    c0 = max.(T.(c), ϵ)
    c1 = max.(T.(c_next), ϵ)
    abs.(T(1) .- (β * R) .* (c0 ./ c1) .^ σ)
end

"""
    euler_resid_det(params, a_grid, c)

Absolute residuals on asset grid using linear interpolation of c'.
Pure. Zygote-compatible.
"""
function euler_resid_det_grid(params, a_grid::AbstractVector, c::AbstractVector)
    @assert length(a_grid) == length(c)
    T = _T(a_grid, c)
    R = T(1) + T(params.r)
    # params may omit `:y` or have it set to `nothing` (e.g., some model configs).
    rawy = (:y in propertynames(params)) ? getfield(params, :y) : zero(T)
    rawy = rawy === nothing ? zero(T) : rawy
    y = T(rawy)
    ϵ = _eps(T)
    a = T.(a_grid)
    cT = T.(c)
    n = length(a)
    # compute ap elementwise without mutating existing arrays
    ap_vals = [clamp(R * a[i] + y - cT[i], first(a), last(a)) for i = 1:n]
    c_next = [interp_linear(a, cT, ap_i) for ap_i in ap_vals]
    euler_resid_det(params, cT, c_next)
end

# Mutating variant (not AD-safe)
function euler_resid_det!(
    resid::AbstractVector,
    params,
    c::AbstractVector,
    c_next::AbstractVector,
)
    @assert length(resid) == length(c) == length(c_next)
    β = params.β
    σ = params.σ
    R = 1 + params.r
    @inbounds for i in eachindex(resid)
        c0 = c[i] <= 1e-12 ? 1e-12 : c[i]
        c1 = c_next[i] <= 1e-12 ? 1e-12 : c_next[i]
        resid[i] = abs(1 - β * R * (c0 / c1)^σ)
    end
    resid
end
Zygote.@ignore euler_resid_det!

# Backward-compatible alias
@deprecate euler_resid_det_2(params, a_grid, c) euler_resid_det_grid(params, a_grid, c)

# -------------------------
# Stochastic (pure, AD-safe)
# -------------------------
function euler_resid_stoch(
    params,
    a_grid::AbstractVector,
    z_grid::AbstractVector,
    Π::AbstractMatrix,
    c::AbstractMatrix,
)
    # grid-based stochastic residuals
    Na, Nz = size(c)
    @assert length(a_grid) == Na
    @assert length(z_grid) == Nz
    @assert size(Π) == (Nz, Nz)

    T = _T(a_grid, z_grid, Π, c)
    β = T(params.β)
    σ = T(params.σ)
    R = T(1) + T(params.r)
    ϵ = _eps(T)

    a = T.(a_grid)
    z = T.(z_grid)
    ΠT = T.(Π)
    C = T.(c)

    res = [
        begin
            c_ij = max(C[i, j], ϵ)
            ap = R * a[i] + exp(z[j]) - c_ij
            Emu = mapreduce(
                jp -> begin
                    cp = interp_linear(a, C[:, jp], ap)
                    ΠT[j, jp] * (max(cp, ϵ) / c_ij)^(-σ)
                end,
                +,
                1:Nz;
                init = zero(T),
            )
            abs(T(1) - (β * R) * Emu)
        end for i = 1:Na, j = 1:Nz
    ]
    reshape(res, Na, Nz)
end

# Alias for grid-based stochastic residuals
const euler_resid_stoch_grid = euler_resid_stoch

# Mutating stochastic (not AD-safe)
function euler_resid_stoch!(resid::AbstractMatrix, params, a_grid, z_grid, Π, c)
    Na, Nz = size(c)
    @assert size(resid) == (Na, Nz)
    β = params.β
    σ = params.σ
    R = 1 + params.r
    @inbounds for j = 1:Nz
        y = exp(z_grid[j])
        for i = 1:Na
            c_ij = max(c[i, j], 1e-12)
            ap = R * a_grid[i] + y - c_ij
            Emu = 0.0
            for jp = 1:Nz
                cp = interp_linear(a_grid, c[:, jp], ap)
                Emu += Π[j, jp] * (max(cp, 1e-12) / c_ij)^(-σ)
            end
            resid[i, j] = abs(1 - β * R * Emu)
        end
    end
    resid
end

Zygote.@ignore euler_resid_stoch!

end # module EulerResiduals
