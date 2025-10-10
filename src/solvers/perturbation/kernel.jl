"""
PerturbationKernel

Simple perturbation-based policy update kernel used for smoke tests and
baselines. Not intended as a production-quality solver.
"""
module PerturbationKernel

using ..EulerResiduals:
    euler_resid_det, euler_resid_stoch, euler_resid_det_grid, euler_resid_stoch_grid
using ..CommonInterp: InterpKind, LinearInterp
using ..PolicyUtils: clamp_policy!, compute_binding_tolerance, rmse_nonbinding
using ..SolverPlaceholders: build_placeholder_solution
using ..SolverIntegration: integrate_expectation, discrete_expectation
using Random: default_rng
using ..CSVarUtils: csvar_income
using Statistics: mean
using ForwardDiff
using LinearAlgebra

export solve_perturbation_det, solve_perturbation_stoch, solve_perturbation_placeholder

const DEFAULT_BINDING_TOL = 1e-12

"""
    _steady_state_asset(p, g)

Compute the deterministic steady-state asset level implied by the model
parameters `p` and grid descriptor `g`. For the baseline consumer-savings
model the steady state is pinned to the borrowing bound when `β(1 + r) ≤ 1`
and to the upper bound otherwise. This mirrors `SteadyState.steady_state_analytic`
but keeps the kernel self-contained.
"""
function _steady_state_asset(p, g)
    a_min = g[:a].min
    a_max = g[:a].max
    βR = p.β * (1 + p.r)
    if βR ≤ 1 + 1e-12
        return a_min
    else
        return a_max
    end
end

"""
    _shock_moments(S)

Return `(ρ, μ, σ2, σε2)` for the AR(1) Gaussian shock encoded in `S`. When
diagnostics are unavailable we fall back to iid, zero-mean shocks.
"""
function _shock_moments(S)
    if S === nothing
        return 0.0, 0.0, 0.0, 0.0
    end
    ρ = 0.0
    μ = 0.0
    σ2 = 0.0
    if hasproperty(S, :diagnostics) && S.diagnostics !== nothing && !isempty(S.diagnostics)
        diag = S.diagnostics
        if length(diag) ≥ 3
            μ = diag[1]
            σ2 = max(diag[2], 0.0)
            ρ = diag[3]
        elseif length(diag) == 2
            μ = diag[1]
            σ2 = max(diag[2], 0.0)
        else
            ρ = diag[end]
        end
    end
    σε2 = σ2 * max(1 - ρ^2, 0.0)
    return ρ, μ, σ2, σε2
end

"""
    _gaussian_ratio_expectation(c0, cp0, cp1, cp2, γ; μ = 0.0, σϵ2 = 0.0)

Second-order approximation to `E[(c0 / cp(z'))^γ]` where `cp(z') ≈ cp0 + cp1 z' +
0.5 * cp2 z'^2` and the innovation `z'` is Gaussian with mean `μ` and variance
`σϵ2`. The expansion is taken around the deterministic steady state and keeps
terms up to second order in `(cp1/cp0)` and `(cp2/cp0)`.
"""
function _gaussian_ratio_expectation(c0, cp0, cp1, cp2, γ; μ = 0.0, σϵ2 = 0.0)
    ϵ = 1e-12
    c0 = c0 ≤ ϵ ? ϵ : c0
    cp0 = cp0 ≤ ϵ ? ϵ : cp0
    x1 = cp1 / cp0
    x2 = 0.5 * cp2 / cp0
    G0 = (c0 / cp0)^γ
    A = -γ * x1
    B = -γ * x2 + 0.5 * γ * x1^2
    μ1 = μ
    μ2 = μ^2 + σϵ2
    return G0 * (1 + A * μ1 + B * μ2 + 0.5 * A^2 * μ2)
end

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
    rng = nothing,
)
    t0 = time_ns()
    a_grid = g[:a].grid
    a_min = g[:a].min
    a_max = g[:a].max
    Na = g[:a].N
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = DEFAULT_BINDING_TOL)
    R = 1 + p.r
    ȳ = hasproperty(p, :y_dim) ? csvar_income(p.y) : p.y

    ā = a_bar === nothing ? _steady_state_asset(p, g) : a_bar
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
                r[idx] = one(Tθ) - p.β * R * (c0 / c1)^(p.γ)
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
    clamp_policy!(c, cmin, cmax)
    a_next = @. R * a_grid + ȳ - c
    @. a_next = clamp(a_next, a_min, a_max)

    resid = euler_resid_det_grid(p, a_grid, c)
    iters = 1
    converged = true
    # RMSE on non-binding points (where a' > a_min + tol)
    max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)
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
        expansion_point = (ā = ā, z̄ = 0.0),
        resid_metric = :rmse,
    )
    return (
        a_grid = a_grid,
        c = c,
        a_next = a_next,
        resid = resid,
        iters = iters,
        converged = converged,
        max_resid = max_resid,
        rmse = max_resid,
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
    integration_method::Symbol = :gh,
    rng = nothing,
)
    if hasproperty(S, :process) && S.process == :gaussian_linear
        return solve_perturbation_csvar(
            p,
            g,
            S,
            U;
            a_bar = a_bar,
            order = order,
            h_a = h_a,
            h_z = h_z,
            tol_fit = tol_fit,
            maxit_fit = maxit_fit,
            integration_method = integration_method,
            rng = rng,
        )
    end
    t0 = time_ns()
    a_grid = g[:a].grid
    a_min = g[:a].min
    a_max = g[:a].max
    Na = g[:a].N
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = DEFAULT_BINDING_TOL)
    z_grid = S.zgrid
    Π = S.Π
    Nz = length(z_grid)
    R = 1 + p.r
    # Use model mean income
    ȳ = p.y
    ρ, _, _, σε2 = _shock_moments(S)

    ā = a_bar === nothing ? _steady_state_asset(p, g) : a_bar
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
                cp0 = c̄ + A * da1 + 0.5 * C * da1^2
                cp1 = B + D * da1
                cp2 = E
                μ = ρ * z
                Emu = _gaussian_ratio_expectation(c0, cp0, cp1, cp2, p.γ; μ = μ, σϵ2 = σε2)
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
    available = similar(a_grid)
    @inbounds for j = 1:Nz
        z = z_grid[j]
        col = view(c, :, j)
        for i = 1:Na
            ai = a_grid[i]
            da = ai - ā
            col[i] = c̄ + Fa * da + Fz * z + 0.5 * (C2 * da^2 + 2D2 * da * z + E2 * z^2)
            available[i] = exp(z) + R * ai - a_min
        end
        clamp_policy!(col, cmin, available)
        a_col = view(a_next, :, j)
        @. a_col = clamp(R * a_grid + exp(z) - col, a_min, a_max)
    end

    resid = euler_resid_stoch_grid(p, a_grid, z_grid, Π, c)
    iters = 1
    converged = true
    # RMSE on non-binding entries
    max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)
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
        expansion_point = (ā = ā, z̄ = 0.0),
        resid_metric = :rmse,
    )
    return (
        a_grid = a_grid,
        z_grid = z_grid,
        c = c,
        a_next = a_next,
        resid = resid,
        iters = iters,
        converged = converged,
        max_resid = max_resid,
        rmse = max_resid,
        model_params = p,
        opts = opts,
    )
end

function solve_perturbation_csvar(
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
    integration_method::Symbol = :gh,
    rng = nothing,
)
    order > 1 &&
        @warn "Perturbation (CSVar) currently supports only first-order approximation; downgrading to order=1" order
    integration_method == :mc &&
        @warn "CSVar perturbation uses Gauss-Hermite integration for stability; switching to :gh" integration_method
    integration = integration_method == :mc ? :gh : integration_method
    local_rng = rng === nothing ? default_rng() : rng

    t0 = time_ns()
    a_grid = g[:a].grid
    a_min = g[:a].min
    a_max = g[:a].max
    Na = g[:a].N
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = DEFAULT_BINDING_TOL)

    ȳ_vec = Float64.(p.y)
    d = length(ȳ_vec)
    ȳ = csvar_income(ȳ_vec)

    R = 1 + p.r
    β = p.β
    γ = p.γ

    ā = a_bar === nothing ? _steady_state_asset(p, g) : a_bar
    c̄ = ȳ + p.r * ā

    θ0 = zeros(Float64, 1 + d)
    θ0[1] = R - 1 / (β * R)

    ha = h_a === nothing ? 0.01 * max(a_max - a_min, 1.0) : h_a
    base_y_scale = maximum(abs.(ȳ_vec))
    hz_val = h_z === nothing ? (base_y_scale == 0 ? 0.01 : 0.01 * base_y_scale) : h_z

    collocation = Vector{Tuple{Float64,Vector{Float64}}}()
    push!(collocation, (ha, zeros(d)))
    push!(collocation, (-ha, zeros(d)))
    for j = 1:d
        Δ = zeros(d)
        Δ[j] = hz_val
        push!(collocation, (0.0, Δ))
        Δn = zeros(d)
        Δn[j] = -hz_val
        push!(collocation, (0.0, Δn))
    end

    cmin = 1e-12

    function csvar_residual(θ::AbstractVector)
        Fa = θ[1]
        Fy = view(θ, 2:length(θ))
        r = Vector{Float64}(undef, length(collocation))
        for (idx, (da, Δy)) in enumerate(collocation)
            y_curr = ȳ_vec .+ Δy
            income_curr = csvar_income(y_curr)
            c0 = c̄ + Fa * da + dot(Fy, Δy)
            c0 = c0 <= cmin ? cmin : c0
            a0 = ā + da
            a1 = clamp(R * a0 + income_curr - c0, a_min, a_max)

            integrand = function (y_next)
                δy_next = y_next .- ȳ_vec
                c1 = c̄ + Fa * (a1 - ā) + dot(Fy, δy_next)
                c1 = c1 <= cmin ? cmin : c1
                (c0 / c1)^γ
            end

            EU = integrate_expectation(
                integration,
                integrand,
                p,
                S,
                y_curr;
                gh_order = max(3, d),
                rng = local_rng,
            )
            r[idx] = 1 - β * R * EU
        end
        return r
    end

    θ_init = copy(θ0)
    θ̂, ok, _ = _gauss_newton!(θ_init, csvar_residual, maxit_fit, tol_fit)
    Fa = θ̂[1]
    Fy = θ̂[2:end]

    c = Vector{Float64}(undef, Na)
    available = similar(a_grid)
    for (i, a_val) in enumerate(a_grid)
        da = a_val - ā
        c_val = c̄ + Fa * da
        available[i] = ȳ + R * a_val - a_min
        c[i] = clamp(c_val, cmin, available[i])
    end

    a_next = clamp.(R .* a_grid .+ ȳ .- c, a_min, a_max)

    resid = Vector{Float64}(undef, Na)
    for (i, a_val) in enumerate(a_grid)
        da = a_val - ā
        c0 = c̄ + Fa * da
        c0 = c0 <= cmin ? cmin : c0
        a1 = clamp(R * a_val + ȳ - c0, a_min, a_max)
        integrand = function (y_next)
            δy_next = y_next .- ȳ_vec
            c1 = c̄ + Fa * (a1 - ā) + dot(Fy, δy_next)
            c1 = c1 <= cmin ? cmin : c1
            (c0 / c1)^γ
        end
        EU = integrate_expectation(
            integration,
            integrand,
            p,
            S,
            ȳ_vec;
            gh_order = max(3, d),
            rng = local_rng,
        )
        resid[i] = abs(1 - β * R * EU)
    end

    max_resid = maximum(resid)
    rmse = sqrt(mean(resid .^ 2))

    runtime = (time_ns() - t0) / 1e9
    opts = (;
        maxit = 1,
        runtime,
        seed = -1,
        interp_kind = LinearInterp(),
        tol = NaN,
        tol_pol = NaN,
        relax = NaN,
        patience = 0,
        order = 1,
        fit_ok = ok,
        theta = θ̂,
        resid_metric = :max,
        integration_method = integration,
    )

    return (
        a_grid = a_grid,
        c = c,
        a_next = a_next,
        resid = resid,
        iters = 1,
        converged = ok,
        max_resid = max_resid,
        rmse = rmse,
        model_params = p,
        opts = opts,
    )
end

function solve_perturbation_placeholder(
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    a_bar = nothing,
    order::Int = 1,
    h_a = nothing,
    h_z = nothing,
    tol_fit::Real = NaN,
    maxit_fit::Int = 0,
)
    opts = (;
        maxit = maxit_fit,
        runtime = 0.0,
        seed = nothing,
        order = order,
        tol_fit = tol_fit,
        h_a = h_a,
        h_z = h_z,
        a_bar = a_bar,
    )
    return build_placeholder_solution(
        :Perturbation,
        model_params,
        model_grids,
        model_shocks;
        opts = opts,
        note = "placeholder perturbation solution",
    )
end

end # module
