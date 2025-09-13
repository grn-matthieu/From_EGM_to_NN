module EulerResiduals

export euler_resid_det, euler_resid_stoch, euler_resid_det_2

# --- Functions ---

"""
    euler_resid_det(model_params, c, c_next)

Compute absolute Euler equation residuals for the deterministic model. Computations are based on the CRRA utility function.
"""
function euler_resid_det(model_params, c::Vector{Float64}, c_next::Vector{Float64})
    resid = similar(c)

    # Clamping to avoid division by zero
    c_clamped = clamp.(c, 1e-12, Inf)
    c_next_clamped = clamp.(c_next, 1e-12, Inf)

    β = model_params.β
    σ = model_params.σ
    R = 1 + model_params.r

    @. resid = abs(1 - β * R * (c_clamped / c_next_clamped)^σ)
    return resid
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

    β = model_params.β
    σ = model_params.σ
    R = 1 + model_params.r
    y = getfield(model_params, :y)

    res = similar(c, Float64)

    # linear interpolation helper over a_grid
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

    @inbounds for i = 1:Na
        c_i = max(c[i], 1e-12)
        ap = R * a_grid[i] + y - c_i
        cp = lin1(a_grid, c, ap)
        cp = max(cp, 1e-12)
        res[i] = abs(1.0 - β * R * (c_i / cp)^σ)
    end
    return res
end


"""
    euler_resid_stoch(model_params, a_grid, z_grid, Pz, c)

Compute absolute Euler equation residuals for the stochastic savings model on a grid of assets `a_grid`
and shocks `z_grid` with transition matrix `Pz`. The policy `c` is a Na x Nz matrix of consumption. When residuals
are computed, linear interpolation is used on the asset grid.
Outputs a Na x Nz matrix of absolute Euler equation residuals.
"""
function euler_resid_stoch(
    model_params,
    a_grid::AbstractVector{<:Real},
    z_grid::AbstractVector{<:Real},
    Pz::AbstractMatrix{<:Real},
    c::AbstractMatrix{<:Real},
)
    Na, Nz = size(c)
    @assert length(a_grid) == Na
    @assert length(z_grid) == Nz
    @assert size(Pz, 1) == Nz && size(Pz, 2) == Nz

    β = model_params.β
    σ = model_params.σ
    R = (1 + model_params.r)

    res = similar(c, Float64)

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
            res[i, j] = abs(1.0 - β * R * Emu)
        end
    end
    return res
end

end # module
