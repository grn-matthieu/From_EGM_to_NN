"""
EGMKernel

Endogenous Grid Method solver kernel for the consumption–savings model.
Exports helpers and the main projection-based policy iteration.
"""
module EGMKernel

using ..CommonInterp:
    interp_linear!, interp_pchip!, InterpKind, LinearInterp, MonotoneCubicInterp
using ..EulerResiduals:
    euler_resid_det, euler_resid_stoch, euler_resid_det!, euler_resid_stoch!

export solve_egm_det, solve_egm_stoch


"""
    solve_egm_det(model_params, model_grids, model_utility; ...)

Vectorized EGM solver for deterministic income (equivalent to log-normal income with zero variance).
Stops when Euler equation errors and policy changes fall below their respective tolerances.
Returns a `NamedTuple` with fields `(a_grid, c, a_next, resid, iters, converged, max_resid, model_params, opts)`
which is later converted into a `Solution`.
"""
function solve_egm_det(
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 10_000,
    interp_kind::InterpKind = LinearInterp(),
    relax::Real = 0.5,
    patience::Int = 50,
    ϵ::Real = 1e-10,
    c_init = nothing,
)::NamedTuple
    # Delegate to interpolation-specialized implementation (preserves public API)
    return solve_egm_det_impl(
        interp_kind,
        model_params,
        model_grids,
        model_utility;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        relax = relax,
        patience = patience,
        ϵ = ϵ,
        c_init = c_init,
    )
end

# Fallback in case of unknown interpolation kind
solve_egm_det_impl(::InterpKind, args...; kwargs...) =
    error("Unknown interpolation kind for EGM (deterministic)")

# Linear interpolation specialized helper
function solve_egm_det_impl(
    interp_kind::LinearInterp,
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 10_000,
    relax::Real = 0.5,
    patience::Int = 50,
    ϵ::Real = 1e-10,
    c_init = nothing,
)::NamedTuple
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N

    β = model_params.β
    R = 1 + model_params.r
    σ = model_params.σ
    cmin = 1e-12

    # initial consumption on exogenous grid
    resources = @. R * a_grid - a_min + model_params.y
    c = c_init === nothing ? clamp.(0.5 .* resources, cmin, resources) : copy(c_init)

    # buffers
    cnew = similar(c)
    resid = similar(c)
    a_next = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    no_progress = 0

    for it = 1:maxit
        iters = it

        # --- EGM backsolve: map from a' grid to current a via Euler inversion ---
        a_prime = a_grid
        # consumption at a' (next period) is available on exogenous grid (c)
        c_prime = copy(c)
        @. c_prime = max(c_prime, cmin)
        c_endo = similar(c_prime)

        # compute current consumption from Euler: c = u_prime_inv(β R u'(c'))
        @. c_endo = model_utility.u_prime_inv(β * R * c_prime .^ (-σ))

        # compute corresponding current assets a = (a' - y + c) / R
        a_endo = (@. (a_prime - model_params.y + c_endo) / R)

        # handle borrowing constraint: if computed a < a_min then set a = a_min and recompute c from budget constraint
        @inbounds for i = 1:Na
            if a_endo[i] < a_min
                a_endo[i] = a_min
                # budget: a' = R * a_min + y - c  => c = y + R*a_min - a'
                c_endo[i] = clamp(model_params.y + R * a_min - a_prime[i], cmin, Inf)
            end
        end

        # sort endogenous points by a_endo for interpolation
        perm = sortperm(a_endo)
        a_endo_s = a_endo[perm]
        c_endo_s = c_endo[perm]

        # interpolate from endogenous (a_endo_s, c_endo_s) to exogenous a_grid
        interp_linear!(cnew, a_endo_s, c_endo_s, a_grid)

        # clamp to feasible consumption
        cmax = @. model_params.y + R * a_grid - a_min
        @. cnew = clamp(cnew, cmin, cmax)

        Δpol = maximum(abs.(c - cnew))
        @. c = (1 - relax) * c + relax * cnew

        # compute Euler residuals for diagnostics (using standard a_next interpolation)
        @. a_next = model_params.y + R * a_grid - c
        @. a_next = clamp(a_next, a_min, a_max)
        cnext = similar(c)
        interp_linear!(cnext, a_grid, c, a_next)
        @. cnext = max(cnext, cmin)
        euler_resid_det!(resid, model_params, c, cnext)
        max_resid = maximum(resid[min(2, end):end])

        if (best_resid - max_resid < ϵ) && (Δpol < ϵ)
            no_progress += 1
        else
            no_progress = 0
            best_resid = max_resid
        end

        if no_progress ≥ patience
            break
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    # Final consistency and diagnostics
    @. a_next = R * a_grid + model_params.y - c
    @. a_next = clamp(a_next, a_min, a_max)
    cnext = similar(c)
    interp_linear!(cnext, a_grid, c, a_next)
    @. cnext = max(cnext, cmin)
    euler_resid_det!(resid, model_params, c, cnext)
    max_resid = maximum(resid[min(2, end):end])

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        interp_kind = interp_kind,
        relax = relax,
        patience = patience,
        ϵ = ϵ,
        seed = nothing,
        runtime = runtime,
    )

    return (; a_grid, c, a_next, resid, iters, converged, max_resid, model_params, opts)
end

# Monotone cubic (PCHIP) specialized helper
function solve_egm_det_impl(
    interp_kind::MonotoneCubicInterp,
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 500,
    relax::Real = 0.5,
    patience::Int = 50,
    ϵ::Real = 1e-10,
    c_init = nothing,
)::NamedTuple
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N

    β = model_params.β
    R = 1 + model_params.r
    σ = model_params.σ
    cmin = 1e-12

    # initial consumption on exogenous grid
    resources = @. R * a_grid - a_min + model_params.y
    c = c_init === nothing ? clamp.(0.5 .* resources, cmin, resources) : copy(c_init)

    # buffers
    cnew = similar(c)
    resid = similar(c)
    a_next = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    no_progress = 0

    for it = 1:maxit
        iters = it

        # EGM backsolve with PCHIP interpolation on endogenous points
        a_prime = a_grid
        c_prime = copy(c)
        @. c_prime = max(c_prime, cmin)
        c_endo = similar(c_prime)

        @. c_endo = model_utility.u_prime_inv(β * R * c_prime .^ (-σ))
        a_endo = (@. (a_prime - model_params.y + c_endo) / R)

        @inbounds for i = 1:Na
            if a_endo[i] < a_min
                a_endo[i] = a_min
                c_endo[i] = clamp(model_params.y + R * a_min - a_prime[i], cmin, Inf)
            end
        end

        # sort endogenous nodes
        perm = sortperm(a_endo)
        a_endo_s = a_endo[perm]
        c_endo_s = c_endo[perm]

        # PCHIP interpolation from endogenous (a_endo_s, c_endo_s) to exogenous a_grid
        interp_pchip!(cnew, a_endo_s, c_endo_s, a_grid)

        # enforce monotonicity and clamp
        @. cnew = max(cnew, cmin)
        cmax = @. model_params.y + R * a_grid - a_min
        @. cnew = clamp(cnew, cmin, cmax)
        @inbounds for i = 2:Na
            if cnew[i] < cnew[i-1]
                cnew[i] = cnew[i-1] + 1e-12
            end
        end

        Δpol = maximum(abs.(c - cnew))
        @. c = (1 - relax) * c + relax * cnew

        # diagnostics via Euler residuals
        @. a_next = model_params.y + R * a_grid - c
        @. a_next = clamp(a_next, a_min, a_max)
        cnext = similar(c)
        interp_pchip!(cnext, a_grid, c, a_next)
        @. cnext = max(cnext, cmin)
        euler_resid_det!(resid, model_params, c, cnext)
        max_resid = maximum(resid[min(2, end):end])

        if (best_resid - max_resid < ϵ) && (Δpol < ϵ)
            no_progress += 1
        else
            no_progress = 0
            best_resid = max_resid
        end

        if no_progress ≥ patience
            break
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    # Final consistency
    @. a_next = R * a_grid + model_params.y - c
    @. a_next = clamp(a_next, a_min, a_max)
    cnext = similar(c)
    interp_pchip!(cnext, a_grid, c, a_next)
    @. cnext = max(cnext, cmin)
    euler_resid_det!(resid, model_params, c, cnext)
    max_resid = maximum(resid[min(2, end):end])

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        interp_kind = interp_kind,
        relax = relax,
        patience = patience,
        ϵ = ϵ,
        seed = nothing,
        runtime = runtime,
    )

    return (; a_grid, c, a_next, resid, iters, converged, max_resid, model_params, opts)
end


"""
    solve_egm_stoch(model_params, model_grids, model_shocks, model_utility; ...)

Vectorized EGM solver for the CS model with an AR(1) income process.
Stops when expected Euler equation errors and policy changes (evaluated at discretized nodes) meet tolerance.
Returns a `NamedTuple` with fields `(a_grid, z_grid, c, a_next, resid, iters, converged, max_resid, model_params, opts)` that is later converted into a `Solution`.
"""
function solve_egm_stoch(
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 1000,
    interp_kind::InterpKind = LinearInterp(),
    relax::Real = 0.5,
    ϵ::Real = 1e-10,
    patience::Int = 50,
    c_init = nothing,
)::NamedTuple

    # Delegate to interpolation-specialized implementation (preserves public API)
    return solve_egm_stoch_impl(
        interp_kind,
        model_params,
        model_grids,
        model_shocks,
        model_utility;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        relax = relax,
        ϵ = ϵ,
        patience = patience,
        c_init = c_init,
    )
end

# Fallback in case of unknown interpolation kind
solve_egm_stoch_impl(::InterpKind, args...; kwargs...) =
    error("Unknown interpolation kind for EGM (stochastic)")

# Linear interpolation specialized helper
function solve_egm_stoch_impl(
    interp_kind::LinearInterp,
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 1000,
    relax::Real = 0.5,
    ϵ::Real = 1e-10,
    patience::Int = 50,
    c_init = nothing,
)::NamedTuple
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N

    z_grid = model_shocks.zgrid
    Π = model_shocks.Π
    Nz = length(z_grid)

    β = model_params.β
    σ = model_params.σ
    R = 1 + model_params.r
    cmin = 1e-12

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    no_progress = 0

    c = c_init === nothing ? fill(1.0, Na, Nz) : copy(c_init)
    cnew = similar(c)
    a_next = similar(c)
    resid_mat = similar(c)

    # EGM loop: for each current state j, backsolve on endogenous nodes a' = a_grid
    for it = 1:maxit
        iters = it

        for (j, z) in enumerate(z_grid)
            y = exp(z)

            # expectation of marginal utility at nodes a' (which are a_grid)
            EUprime = zeros(eltype(c), Na)
            for jp = 1:Nz
                c_future = view(c, :, jp)
                # accumulate expected marginal utility elementwise: Π[j,jp] * (c_future .^ (-σ))
                @. EUprime += Π[j, jp] * (max(c_future, cmin)^(-σ))
            end

            # compute c on endogenous nodes (a' nodes) via Euler inversion
            c_endo = similar(EUprime)
            @. c_endo = model_utility.u_prime_inv(β * R * EUprime)

            # implied current assets from budget: a = (a' - y + c_endo) / R
            a_endo = (@. (a_grid - y + c_endo) / R)

            # handle borrowing constraint: if a_endo < a_min set a_endo=a_min and recompute c_endo from budget
            @inbounds for i = 1:Na
                if a_endo[i] < a_min
                    a_endo[i] = a_min
                    c_endo[i] = clamp(y + R * a_min - a_grid[i], cmin, Inf)
                end
            end

            # sort endogenous nodes for interpolation
            perm = sortperm(a_endo)
            a_endo_s = a_endo[perm]
            c_endo_s = c_endo[perm]

            # enforce non-decreasing consumption for interpolation stability
            @inbounds for ii = 2:Na
                if c_endo_s[ii] < c_endo_s[ii-1]
                    c_endo_s[ii] = c_endo_s[ii-1] + 1e-12
                end
            end

            # ensure strictly increasing a_endo_s for interpolation
            @inbounds for ii = 2:Na
                if a_endo_s[ii] <= a_endo_s[ii-1]
                    a_endo_s[ii] = a_endo_s[ii-1] + 1e-12
                end
            end

            # interpolate from endogenous nodes to exogenous a_grid
            interp_linear!(view(cnew, :, j), a_endo_s, c_endo_s, a_grid)

            # clamp to feasible consumption
            cmax = @. y + R * a_grid - a_min
            @. cnew[:, j] = clamp(cnew[:, j], cmin, cmax)
        end

        Δpol = maximum(abs.(c - cnew))
        @. c = (1 - relax) * c + relax * cnew

        euler_resid_stoch!(resid_mat, model_params, a_grid, z_grid, Π, c)
        max_resid = maximum(resid_mat[min(2, end):end, :])

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end

        if best_resid - max_resid < ϵ
            no_progress += 1
        else
            no_progress = 0
            best_resid = max_resid
        end

        if no_progress ≥ patience
            break
        end
    end

    euler_resid_stoch!(resid_mat, model_params, a_grid, z_grid, Π, c)
    for (j, z) in enumerate(z_grid)
        y = exp(z)
        @. a_next[:, j] = clamp(R * a_grid + y - c[:, j], a_min, a_max)
    end

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        interp_kind = interp_kind,
        relax = relax,
        patience = patience,
        ϵ = ϵ,
        seed = nothing,
        runtime = runtime,
    )

    return (;
        a_grid,
        z_grid,
        c,
        a_next,
        resid = resid_mat,
        iters,
        converged,
        max_resid,
        model_params,
        opts,
    )
end

# Monotone cubic (PCHIP) specialized helper
function solve_egm_stoch_impl(
    interp_kind::MonotoneCubicInterp,
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 1000,
    relax::Real = 0.5,
    ϵ::Real = 1e-10,
    patience::Int = 50,
    c_init = nothing,
)::NamedTuple
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N

    z_grid = model_shocks.zgrid
    Π = model_shocks.Π
    Nz = length(z_grid)

    β = model_params.β
    σ = model_params.σ
    R = 1 + model_params.r
    cmin = 1e-12

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    no_progress = 0

    c = c_init === nothing ? fill(1.0, Na, Nz) : copy(c_init)
    cnew = similar(c)
    a_next = similar(c)
    resid_mat = similar(c)

    for it = 1:maxit
        iters = it

        for (j, z) in enumerate(z_grid)
            y = exp(z)

            # compute expected marginal utility at a' nodes using future-state consumption
            EUprime = zeros(eltype(c), Na)
            for jp = 1:Nz
                c_future = view(c, :, jp)
                @. EUprime += Π[j, jp] * (max(c_future, cmin)^(-σ))
            end

            # compute c on endogenous nodes
            c_endo = similar(EUprime)
            @. c_endo = model_utility.u_prime_inv(β * R * EUprime)

            a_endo = (@. (a_grid - y + c_endo) / R)

            @inbounds for i = 1:Na
                if a_endo[i] < a_min
                    a_endo[i] = a_min
                    c_endo[i] = clamp(y + R * a_min - a_grid[i], cmin, Inf)
                end
            end

            perm = sortperm(a_endo)
            a_endo_s = a_endo[perm]
            c_endo_s = c_endo[perm]

            # enforce monotonicity and strictly increasing a nodes for PCHIP
            @inbounds for ii = 2:Na
                if c_endo_s[ii] < c_endo_s[ii-1]
                    c_endo_s[ii] = c_endo_s[ii-1] + 1e-12
                end
                if a_endo_s[ii] <= a_endo_s[ii-1]
                    a_endo_s[ii] = a_endo_s[ii-1] + 1e-12
                end
            end

            interp_pchip!(view(cnew, :, j), a_endo_s, c_endo_s, a_grid)

            cmax = @. y + R * a_grid - a_min
            @. cnew[:, j] = clamp(cnew[:, j], cmin, cmax)

            @inbounds for i = 2:Na
                if cnew[i, j] < cnew[i-1, j]
                    cnew[i, j] = cnew[i-1, j] + 1e-12
                end
            end
        end

        Δpol = maximum(abs.(c - cnew))
        @. c = (1 - relax) * c + relax * cnew

        euler_resid_stoch!(resid_mat, model_params, a_grid, z_grid, Π, c)
        max_resid = maximum(resid_mat[min(2, end):end, :])

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end

        if best_resid - max_resid < ϵ
            no_progress += 1
        else
            no_progress = 0
            best_resid = max_resid
        end

        if no_progress ≥ patience
            break
        end
    end

    euler_resid_stoch!(resid_mat, model_params, a_grid, z_grid, Π, c)
    for (j, z) in enumerate(z_grid)
        y = exp(z)
        @. a_next[:, j] = clamp(R * a_grid + y - c[:, j], a_min, a_max)
    end

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        interp_kind = interp_kind,
        relax = relax,
        patience = patience,
        ϵ = ϵ,
        seed = nothing,
        runtime = runtime,
    )

    return (;
        a_grid,
        z_grid,
        c,
        a_next,
        resid = resid_mat,
        iters,
        converged,
        max_resid,
        model_params,
        opts,
    )
end

end #module
