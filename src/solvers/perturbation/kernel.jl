module PerturbationKernel

using ..EulerResiduals: euler_resid_det_2, euler_resid_stoch
using ..CommonInterp: InterpKind, LinearInterp

export solve_perturbation_det, solve_perturbation_stoch

"""
    _coefficients_first_order(p; ρ, ȳ, R)

Compute first-order policy coefficients (F_a, F_z) for the linear decision rule
Δc_t = F_a Δa_t + F_z Δz_t in the consumer-savings model with AR(1) log shock z.

Formulas (derived from Euler + budget linearization):
    F_a = R - 1/(βR)
    F_z = βR * F_a * ȳ / (1 + βR * F_a - βR * ρ)

No constraint handling; intended for interior steady state analysis.
"""
function _coefficients_first_order(p; ρ::Real, ȳ::Real, R::Real)
    βR = p.β * R
    Fa = R - 1 / (βR)
    Fz = (βR * Fa * ȳ) / (1 + βR * Fa - βR * ρ)
    return Fa, Fz
end

"""
    solve_perturbation_det(p, g, U; a_bar=nothing)

Deterministic linear policy around a reference asset level `ā`.
If `a_bar` is not provided, uses the grid midpoint.
Returns NamedTuple with fields: a_grid, c, a_next, resid, iters, converged, max_resid, model_params, opts
"""
function solve_perturbation_det(p, g, U; a_bar = nothing)
    t0 = time_ns()
    a_grid = g[:a].grid
    a_min = g[:a].min
    a_max = g[:a].max
    Na = g[:a].N
    R = 1 + p.r
    ȳ = p.y

    ā = a_bar === nothing ? 0.5 * (a_min + a_max) : a_bar
    c̄ = ȳ + p.r * ā
    Fa, Fz = _coefficients_first_order(p; ρ = 0.0, ȳ = ȳ, R = R)

    # Deterministic: z = 0
    c = @. c̄ + Fa * (a_grid - ā)
    cmin = 1e-12
    cmax = @. ȳ + R * a_grid - a_min
    @. c = clamp(c, cmin, cmax)
    a_next = @. R * a_grid + ȳ - c
    @. a_next = clamp(a_next, a_min, a_max)

    resid = euler_resid_det_2(p, a_grid, c)
    iters = 1
    converged = true
    max_resid = maximum(resid)
    opts = (;
        maxit = iters,
        runtime = (time_ns() - t0) / 1e9,
        seed = -1,
        interp_kind = LinearInterp(),
        tol = NaN,
        tol_pol = NaN,
        relax = NaN,
        patience = 0,
    )
    return (
        a_grid = a_grid,
        c = c,
        a_next = a_next,
        resid = resid,
        iters = iters,
        converged = converged,
        max_resid = max_resid,
        model_params = p,
        opts = opts,
    )
end

"""
    solve_perturbation_stoch(p, g, S, U; a_bar=nothing)

Stochastic linear policy using first-order coefficients evaluated at `ā`.
Policy: c(a,z) = c̄ + F_a (a-ā) + F_z z. Builds matrices over (a,z) grid.
"""
function solve_perturbation_stoch(p, g, S, U; a_bar = nothing)
    t0 = time_ns()
    a_grid = g[:a].grid
    a_min = g[:a].min
    a_max = g[:a].max
    Na = g[:a].N
    z_grid = S.zgrid
    Π = S.Π
    Nz = length(z_grid)
    R = 1 + p.r
    ȳ = p.y
    ρ = begin
        # Recover AR(1) persistence from discretization diagnostics if available
        # Fall back to 0.0 if not provided
        if hasproperty(S, :diagnostics)
            try
                S.diagnostics[end]
            catch
                0.0
            end
        else
            0.0
        end
    end

    ā = a_bar === nothing ? 0.5 * (a_min + a_max) : a_bar
    c̄ = ȳ + p.r * ā
    Fa, Fz = _coefficients_first_order(p; ρ = ρ, ȳ = ȳ, R = R)

    c = Array{Float64}(undef, Na, Nz)
    a_next = similar(c)
    cmin = 1e-12
    @inbounds for j = 1:Nz
        z = z_grid[j]
        for i = 1:Na
            ai = a_grid[i]
            cij = c̄ + Fa * (ai - ā) + Fz * z
            cij = clamp(cij, cmin, ȳ * exp(z) + R * ai - a_min)
            c[i, j] = cij
            a_next[i, j] = clamp(R * ai + ȳ * exp(z) - cij, a_min, a_max)
        end
    end

    resid = euler_resid_stoch(p, a_grid, z_grid, Π, c)
    iters = 1
    converged = true
    max_resid = maximum(resid)
    opts = (;
        maxit = iters,
        runtime = (time_ns() - t0) / 1e9,
        seed = -1,
        interp_kind = LinearInterp(),
        tol = NaN,
        tol_pol = NaN,
        relax = NaN,
        patience = 0,
    )
    return (
        a_grid = a_grid,
        c = c,
        a_next = a_next,
        resid = resid,
        iters = iters,
        converged = converged,
        max_resid = max_resid,
        model_params = p,
        opts = opts,
    )
end

end # module
