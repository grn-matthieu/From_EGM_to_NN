module PerturbationKernel

using ..EulerResiduals: euler_resid_det_2, euler_resid_stoch
using ..CommonInterp: InterpKind, LinearInterp
using ForwardDiff
using LinearAlgebra

export solve_perturbation_det, solve_perturbation_stoch

"""
Gauss–Newton for small nonlinear least squares over coefficients θ using AD Jacobian.
Returns (θ_new, ok::Bool, norm_r).
Falls back to finite-difference if AD fails.
"""
function _gauss_newton!(θ::AbstractVector, rfun, maxit::Int, tol::Real)
    n = length(θ)
    δθ = zeros(eltype(θ), n)
    for it = 1:maxit
        r = rfun(θ)
        nr = norm(r)
        if nr < tol
            return θ, true, nr
        end
        # Jacobian via ForwardDiff; fallback to finite-difference on failure
        J = try
            ForwardDiff.jacobian(rfun, θ)
        catch
            # simple finite-difference fallback
            m = length(r)
            Jtmp = zeros(eltype(θ), m, n)
            epsθ = 1e-6
            for k = 1:n
                θp = copy(θ)
                θp[k] += epsθ
                rp = rfun(θp)
                @. Jtmp[:, k] = (rp - r) / epsθ
            end
            Jtmp
        end
        δθ .= -(J'J) \ (J'r)
        # Damped step
        α = 1.0
        θ_trial = θ .+ α .* δθ
        r_trial = rfun(θ_trial)
        if norm(r_trial) < nr
            θ .= θ_trial
        else
            α = 0.5
            improved = false
            for _ = 1:5
                θ_trial .= θ .+ α .* δθ
                r_trial .= rfun(θ_trial)
                if norm(r_trial) < nr
                    θ .= θ_trial
                    improved = true
                    break
                end
                α *= 0.5
            end
            if !improved
                return θ, false, nr
            end
        end
    end
    return θ, false, norm(rfun(θ))
end

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
function solve_perturbation_det(
    p,
    g,
    U;
    a_bar = nothing,
    order::Int = 1,
    h_a = nothing,
    tol_fit = 1e-8,
    maxit_fit = 25,
)
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

    # Attempt 2nd order fit around ā if requested
    C2 = 0.0
    fit_ok = false
    if order ≥ 2
        ha = h_a === nothing ? 0.01 * max(a_max - a_min, 1.0) : h_a
        # residual at two points (±ha) equals zero → solve for [A, C]
        function resid_det(θ)
            Tθ = eltype(θ)
            A, C = θ
            r = Vector{Tθ}(undef, 2)
            for (idx, da) in enumerate((ha, -ha))
                a0 = ā + da
                c0 = c̄ + A * da + 0.5 * C * da^2
                c0 = max(c0, Tθ(1e-12))
                a1 = clamp(R * a0 + ȳ - c0, a_min, a_max)
                da1 = a1 - ā
                c1 = c̄ + A * da1 + 0.5 * C * da1^2
                c1 = max(c1, Tθ(1e-12))
                r[idx] = one(Tθ) - p.β * R * (c0 / c1)^(p.σ)
            end
            return r
        end
        θ0 = [Fa, 0.0]
        θ̂, ok, _ = _gauss_newton!(θ0, resid_det, maxit_fit, tol_fit)
        if ok && isfinite.(θ̂) |> all
            Fa = θ̂[1]
            C2 = θ̂[2]
            fit_ok = true
        end
    end

    # Deterministic: z = 0
    c = @. c̄ + Fa * (a_grid - ā) + 0.5 * C2 * (a_grid - ā)^2
    cmin = 1e-12
    cmax = @. ȳ + R * a_grid - a_min
    @. c = clamp(c, cmin, cmax)
    a_next = @. R * a_grid + ȳ - c
    @. a_next = clamp(a_next, a_min, a_max)

    resid = euler_resid_det_2(p, a_grid, c)
    iters = 1
    converged = true
    # prune boundaries when assessing accuracy
    lo = Na > 2 ? 2 : 1
    hi = Na > 2 ? Na - 1 : Na
    max_resid = maximum(view(resid, lo:hi))
    opts = (;
        maxit = iters,
        runtime = (time_ns() - t0) / 1e9,
        seed = -1,
        interp_kind = LinearInterp(),
        tol = NaN,
        tol_pol = NaN,
        relax = NaN,
        patience = 0,
        order = order,
        fit_ok = fit_ok,
        quad_coeffs = (C2 = C2,),
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
function solve_perturbation_stoch(
    p,
    g,
    S,
    U;
    a_bar = nothing,
    order::Int = 1,
    h_a = nothing,
    h_z = nothing,
    tol_fit = 1e-8,
    maxit_fit = 25,
)
    t0 = time_ns()
    a_grid = g[:a].grid
    a_min = g[:a].min
    a_max = g[:a].max
    Na = g[:a].N
    z_grid = S.zgrid
    Π = S.Π
    Nz = length(z_grid)
    R = 1 + p.r
    # For stochastic residuals, income uses exp(z) in current codebase
    ȳ = 1.0
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

    # Try to fit second-order coefficients locally if requested
    C2 = 0.0
    D2 = 0.0
    E2 = 0.0
    fit_ok = false
    if order ≥ 2
        ha = h_a === nothing ? 0.01 * max(a_max - a_min, 1.0) : h_a
        hz = h_z === nothing ? 0.1 * (maximum(z_grid) - minimum(z_grid)) : h_z
        # Select central z index near 0
        j0 = argmin(abs.(z_grid .- 0.0))
        function resid_stoch(θ)
            Tθ = eltype(θ)
            A, B, C, D, E = θ
            # Use 6 collocation points
            pts = [
                (ha, 0.0, j0),
                (-ha, 0.0, j0),
                (0.0, hz, min(j0 + 1, Nz)),
                (0.0, -hz, max(j0 - 1, 1)),
                (ha, hz, min(j0 + 1, Nz)),
                (-ha, -hz, max(j0 - 1, 1)),
            ]
            r = Vector{Tθ}(undef, length(pts))
            for (k, (da, z, j)) in enumerate(pts)
                a0 = ā + da
                c0 = c̄ + A * da + B * z + 0.5 * (C * da^2 + 2D * da * z + E * z^2)
                c0 = max(c0, Tθ(1e-12))
                a1 = clamp(R * a0 + exp(z) - c0, a_min, a_max)
                da1 = a1 - ā
                # Expectation over next shock using row j of Π
                Emu = 0.0
                for jp = 1:Nz
                    zp = z_grid[jp]
                    cp =
                        c̄ + A * da1 + B * zp + 0.5 * (C * da1^2 + 2D * da1 * zp + E * zp^2)
                    cp = max(cp, Tθ(1e-12))
                    Emu += Π[j, jp] * (c0 / cp)^(p.σ)
                end
                r[k] = one(Tθ) - p.β * R * Emu
            end
            return r
        end
        θ0 = [Fa, Fz, 0.0, 0.0, 0.0]
        θ̂, ok, _ = _gauss_newton!(θ0, resid_stoch, maxit_fit, tol_fit)
        if ok && isfinite.(θ̂) |> all
            Fa, Fz, C2, D2, E2 = θ̂
            fit_ok = true
        end
    end

    c = Array{Float64}(undef, Na, Nz)
    a_next = similar(c)
    cmin = 1e-12
    @inbounds for j = 1:Nz
        z = z_grid[j]
        for i = 1:Na
            ai = a_grid[i]
            da = ai - ā
            cij = c̄ + Fa * da + Fz * z + 0.5 * (C2 * da^2 + 2D2 * da * z + E2 * z^2)
            cij = clamp(cij, cmin, exp(z) + R * ai - a_min)
            c[i, j] = cij
            a_next[i, j] = clamp(R * ai + exp(z) - cij, a_min, a_max)
        end
    end

    resid = euler_resid_stoch(p, a_grid, z_grid, Π, c)
    iters = 1
    converged = true
    # prune boundary asset points when computing maximum residual
    lo = Na > 2 ? 2 : 1
    hi = Na > 2 ? Na - 1 : Na
    max_resid = maximum(view(resid, lo:hi, :))
    opts = (;
        maxit = iters,
        runtime = (time_ns() - t0) / 1e9,
        seed = -1,
        interp_kind = LinearInterp(),
        tol = NaN,
        tol_pol = NaN,
        relax = NaN,
        patience = 0,
        order = order,
        fit_ok = fit_ok,
        quad_coeffs = (C2 = C2, D2 = D2, E2 = E2),
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
