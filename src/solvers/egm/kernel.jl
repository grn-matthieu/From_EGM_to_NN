"""
EGMKernel

Endogenous Grid Method solver kernel for the consumption–savings model.
Exports helpers and the main projection-based policy iteration.
"""
module EGMKernel

using ..CommonInterp:
    interp_linear!, interp_pchip!, InterpKind, LinearInterp, MonotoneCubicInterp
using ..EulerResiduals: euler_resid_det!, euler_resid_stoch!, euler_resid_stoch_interp!
using Printf
using Statistics: mean

export solve_egm_det, solve_egm_stoch

const DEFAULT_BINDING_TOL = 1e-10

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
    verbose::Bool = false,
    c_init = nothing,
)::NamedTuple
    return solve_egm_det_impl(
        interp_kind,
        model_params,
        model_grids,
        model_utility;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        relax = relax,
        verbose = verbose,
        c_init = c_init,
    )
end

solve_egm_det_impl(::InterpKind, args...; kwargs...) =
    error("Unknown interpolation kind for EGM (deterministic)")

function solve_egm_det_impl(
    ::LinearInterp,
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 10_000,
    relax::Real = 0.5,
    verbose::Bool = false,
    c_init = nothing,
)::NamedTuple
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    Δa = Na > 1 ? (a_max - a_min) / (Na - 1) : (a_max - a_min)
    bind_tol = max(DEFAULT_BINDING_TOL, 1e-6 * Δa)

    β = model_params.β
    R = 1 + model_params.r
    σ = model_params.σ
    cmin = 1e-12

    resources = @. R * a_grid - a_min + model_params.y
    c = c_init === nothing ? clamp.(0.5 .* resources, cmin, resources) : copy(c_init)

    cnew = similar(c)
    cnext = similar(c)
    resid = similar(c)
    a_next = similar(c)
    cold = similar(c)
    c_prime = similar(c)
    c_endo = similar(c)
    a_endo = similar(c)
    a_sorted = similar(a_endo)
    c_sorted = similar(c_endo)
    nonbinding = falses(Na)

    converged = false
    iters = 0
    max_resid = Inf

    for it = 1:maxit
        iters = it

        copyto!(cold, c)
        copyto!(c_prime, cold)
        @. c_prime = max(c_prime, cmin)

        @. c_endo = model_utility.u_prime_inv(β * R * c_prime^(-σ))
        @. a_endo = (a_grid - model_params.y + c_endo) / R

        @inbounds for i = 1:Na
            if a_endo[i] < a_min
                a_endo[i] = a_min
                c_endo[i] = clamp(model_params.y + R * a_min - a_grid[i], cmin, Inf)
            end
        end

        perm = sortperm(a_endo)
        @inbounds for k = 1:Na
            idx = perm[k]
            a_sorted[k] = a_endo[idx]
            c_sorted[k] = c_endo[idx]
        end

        interp_linear!(cnew, a_sorted, c_sorted, a_grid)
        cmax = @. model_params.y + R * a_grid - a_min
        @. cnew = clamp(cnew, cmin, cmax)

        @. c = (1 - relax) * cold + relax * cnew
        Δpol = maximum(abs.(c .- cold))

        @. a_next = clamp(model_params.y + R * a_grid - c, a_min, a_max)
        interp_linear!(cnext, a_grid, c, a_next)
        @. cnext = max(cnext, cmin)
        euler_resid_det!(resid, model_params, c, cnext)

        @. nonbinding = a_next > (a_min + bind_tol)
        max_resid = sqrt(mean((resid[nonbinding]) .^ 2))

        if verbose && it % 10 == 0
            @printf("[EGM det linear] it=%d rmse=%.6e Δpol=%.6e\n", it, max_resid, Δpol)
            flush(stdout)
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    @. a_next = clamp(R * a_grid + model_params.y - c, a_min, a_max)
    interp_linear!(cnext, a_grid, c, a_next)
    @. cnext = max(cnext, cmin)
    euler_resid_det!(resid, model_params, c, cnext)
    @. nonbinding = a_next > (a_min + bind_tol)
    max_resid = sqrt(mean((resid[nonbinding]) .^ 2))

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol,
        tol_pol,
        maxit,
        interp_kind = LinearInterp(),
        relax,
        verbose,
        resid_metric = :rmse,
        seed = nothing,
        runtime,
    )

    return (;
        a_grid,
        c,
        a_next,
        resid,
        iters,
        converged,
        max_resid,
        rmse = max_resid,
        model_params,
        opts,
    )
end

function solve_egm_det_impl(
    ::MonotoneCubicInterp,
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 500,
    relax::Real = 0.5,
    verbose::Bool = false,
    c_init = nothing,
)::NamedTuple
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    Δa = Na > 1 ? (a_max - a_min) / (Na - 1) : (a_max - a_min)
    bind_tol = max(DEFAULT_BINDING_TOL, 1e-6 * Δa)

    β = model_params.β
    R = 1 + model_params.r
    σ = model_params.σ
    cmin = 1e-12

    resources = @. R * a_grid - a_min + model_params.y
    c = c_init === nothing ? clamp.(0.5 .* resources, cmin, resources) : copy(c_init)

    cnew = similar(c)
    cnext = similar(c)
    resid = similar(c)
    a_next = similar(c)
    cold = similar(c)
    c_prime = similar(c)
    c_endo = similar(c)
    a_endo = similar(c)
    a_sorted = similar(a_endo)
    c_sorted = similar(c_endo)
    nonbinding = falses(Na)

    converged = false
    iters = 0
    max_resid = Inf

    for it = 1:maxit
        iters = it

        copyto!(cold, c)
        copyto!(c_prime, cold)
        @. c_prime = max(c_prime, cmin)

        @. c_endo = model_utility.u_prime_inv(β * R * c_prime^(-σ))
        @. a_endo = (a_grid - model_params.y + c_endo) / R

        @inbounds for i = 1:Na
            if a_endo[i] < a_min
                a_endo[i] = a_min
                c_endo[i] = clamp(model_params.y + R * a_min - a_grid[i], cmin, Inf)
            end
        end

        perm = sortperm(a_endo)
        @inbounds for k = 1:Na
            idx = perm[k]
            a_sorted[k] = a_endo[idx]
            c_sorted[k] = c_endo[idx]
        end

        @inbounds for k = 2:Na
            if a_sorted[k] <= a_sorted[k-1]
                a_sorted[k] = a_sorted[k-1] + 1e-12
            end
        end

        interp_pchip!(cnew, a_sorted, c_sorted, a_grid)
        cmax = @. model_params.y + R * a_grid - a_min
        @. cnew = clamp(cnew, cmin, cmax)
        @inbounds for i = 2:Na
            if cnew[i] < cnew[i-1]
                cnew[i] = cnew[i-1] + 1e-12
            end
        end

        @. c = (1 - relax) * cold + relax * cnew
        Δpol = maximum(abs.(c .- cold))

        @. a_next = clamp(model_params.y + R * a_grid - c, a_min, a_max)
        interp_pchip!(cnext, a_grid, c, a_next)
        @. cnext = max(cnext, cmin)
        euler_resid_det!(resid, model_params, c, cnext)

        @. nonbinding = a_next > (a_min + bind_tol)
        max_resid = sqrt(mean((resid[nonbinding]) .^ 2))

        if verbose && it % 10 == 0
            @printf("[EGM det pchip] it=%d rmse=%.6e Δpol=%.6e\n", it, max_resid, Δpol)
            flush(stdout)
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    @. a_next = clamp(R * a_grid + model_params.y - c, a_min, a_max)
    interp_pchip!(cnext, a_grid, c, a_next)
    @. cnext = max(cnext, cmin)
    euler_resid_det!(resid, model_params, c, cnext)
    @. nonbinding = a_next > (a_min + bind_tol)
    max_resid = sqrt(mean((resid[nonbinding]) .^ 2))

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol,
        tol_pol,
        maxit,
        interp_kind = MonotoneCubicInterp(),
        relax,
        verbose,
        resid_metric = :rmse,
        seed = nothing,
        runtime,
    )

    return (;
        a_grid,
        c,
        a_next,
        resid,
        iters,
        converged,
        max_resid,
        rmse = max_resid,
        model_params,
        opts,
    )
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
    verbose::Bool = false,
    c_init = nothing,
)::NamedTuple
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
        verbose = verbose,
        c_init = c_init,
    )
end

solve_egm_stoch_impl(::InterpKind, args...; kwargs...) =
    error("Unknown interpolation kind for EGM (stochastic)")

function solve_egm_stoch_impl(
    ::LinearInterp,
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 1000,
    relax::Real = 0.5,
    verbose::Bool = false,
    c_init = nothing,
)::NamedTuple
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    Δa = Na > 1 ? (a_max - a_min) / (Na - 1) : (a_max - a_min)
    bind_tol = max(DEFAULT_BINDING_TOL, 1e-6 * Δa)

    z_grid = model_shocks.zgrid
    Π = model_shocks.Π
    Nz = length(z_grid)

    β = model_params.β
    σ = model_params.σ
    R = 1 + model_params.r
    cmin = 1e-12

    c = c_init === nothing ? fill(1.0, Na, Nz) : copy(c_init)
    cold = similar(c)
    cnew = similar(c)
    a_next = similar(c)
    resid_mat = similar(c)
    nonbinding = falses(Na, Nz)

    EUprime = similar(view(c, :, 1))
    c_endo = similar(EUprime)
    a_endo = similar(EUprime)
    a_sorted = similar(EUprime)
    c_sorted = similar(EUprime)

    converged = false
    iters = 0
    max_resid = Inf

    for it = 1:maxit
        iters = it
        copyto!(cold, c)

        for (j, z) in enumerate(z_grid)
            y = exp(z)
            fill!(EUprime, 0.0)
            for jp = 1:Nz
                c_future = view(cold, :, jp)
                @. EUprime += Π[j, jp] * (max(c_future, cmin)^(-σ))
            end

            @. c_endo = model_utility.u_prime_inv(β * R * EUprime)
            @. a_endo = (a_grid - y + c_endo) / R

            @inbounds for i = 1:Na
                if a_endo[i] < a_min
                    a_endo[i] = a_min
                    c_endo[i] = clamp(y + R * a_min - a_grid[i], cmin, Inf)
                end
            end

            perm = sortperm(a_endo)
            @inbounds for k = 1:Na
                idx = perm[k]
                a_sorted[k] = a_endo[idx]
                c_sorted[k] = c_endo[idx]
            end

            interp_linear!(view(cnew, :, j), a_sorted, c_sorted, a_grid)
            cmax = @. y + R * a_grid - a_min
            @views @. cnew[:, j] = clamp(cnew[:, j], cmin, cmax)
        end

        @. c = (1 - relax) * cold + relax * cnew
        Δpol = maximum(abs.(c .- cold))

        for (j, z) in enumerate(z_grid)
            y = exp(z)
            @views @. a_next[:, j] = clamp(R * a_grid + y - c[:, j], a_min, a_max)
        end

        euler_resid_stoch_interp!(
            resid_mat,
            model_params,
            a_grid,
            z_grid,
            Π,
            c,
            LinearInterp(),
        )

        @. nonbinding = a_next > (a_min + bind_tol)
        max_resid = sqrt(mean((resid_mat[nonbinding]) .^ 2))

        if verbose && it % 10 == 0
            @printf("[EGM stoch linear] it=%d rmse=%.6e Δpol=%.6e\n", it, max_resid, Δpol)
            flush(stdout)
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    for (j, z) in enumerate(z_grid)
        y = exp(z)
        @views @. a_next[:, j] = clamp(R * a_grid + y - c[:, j], a_min, a_max)
    end

    euler_resid_stoch_interp!(resid_mat, model_params, a_grid, z_grid, Π, c, LinearInterp())
    @. nonbinding = a_next > (a_min + bind_tol)
    max_resid = sqrt(mean((resid_mat[nonbinding]) .^ 2))

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol,
        tol_pol,
        maxit,
        interp_kind = LinearInterp(),
        relax,
        verbose,
        seed = nothing,
        runtime,
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

function solve_egm_stoch_impl(
    ::MonotoneCubicInterp,
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-4,
    tol_pol::Real = 1e-6,
    maxit::Int = 1000,
    relax::Real = 0.5,
    verbose::Bool = false,
    c_init = nothing,
)::NamedTuple
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N
    Δa = Na > 1 ? (a_max - a_min) / (Na - 1) : (a_max - a_min)
    bind_tol = max(DEFAULT_BINDING_TOL, 1e-6 * Δa)

    z_grid = model_shocks.zgrid
    Π = model_shocks.Π
    Nz = length(z_grid)

    β = model_params.β
    σ = model_params.σ
    R = 1 + model_params.r
    cmin = 1e-12

    c = c_init === nothing ? fill(1.0, Na, Nz) : copy(c_init)
    cold = similar(c)
    cnew = similar(c)
    a_next = similar(c)
    resid_mat = similar(c)
    nonbinding = falses(Na, Nz)

    EUprime = similar(view(c, :, 1))
    c_endo = similar(EUprime)
    a_endo = similar(EUprime)
    a_sorted = similar(EUprime)
    c_sorted = similar(EUprime)

    converged = false
    iters = 0
    max_resid = Inf

    for it = 1:maxit
        iters = it
        copyto!(cold, c)

        for (j, z) in enumerate(z_grid)
            y = exp(z)
            fill!(EUprime, 0.0)
            for jp = 1:Nz
                c_future = view(cold, :, jp)
                @. EUprime += Π[j, jp] * (max(c_future, cmin)^(-σ))
            end

            @. c_endo = model_utility.u_prime_inv(β * R * EUprime)
            @. a_endo = (a_grid - y + c_endo) / R

            @inbounds for i = 1:Na
                if a_endo[i] < a_min
                    a_endo[i] = a_min
                    c_endo[i] = clamp(y + R * a_min - a_grid[i], cmin, Inf)
                end
            end

            perm = sortperm(a_endo)
            @inbounds for k = 1:Na
                idx = perm[k]
                a_sorted[k] = a_endo[idx]
                c_sorted[k] = c_endo[idx]
            end

            @inbounds for k = 2:Na
                if a_sorted[k] <= a_sorted[k-1]
                    a_sorted[k] = a_sorted[k-1] + 1e-12
                end
                if c_sorted[k] < c_sorted[k-1]
                    c_sorted[k] = c_sorted[k-1] + 1e-12
                end
            end

            interp_pchip!(view(cnew, :, j), a_sorted, c_sorted, a_grid)

            cmax = @. y + R * a_grid - a_min
            @views @. cnew[:, j] = clamp(cnew[:, j], cmin, cmax)
            @inbounds for i = 2:Na
                if cnew[i, j] < cnew[i-1, j]
                    cnew[i, j] = cnew[i-1, j] + 1e-12
                end
            end
        end

        @. c = (1 - relax) * cold + relax * cnew
        Δpol = maximum(abs.(c .- cold))

        for (j, z) in enumerate(z_grid)
            y = exp(z)
            @views @. a_next[:, j] = clamp(R * a_grid + y - c[:, j], a_min, a_max)
        end

        euler_resid_stoch_interp!(
            resid_mat,
            model_params,
            a_grid,
            z_grid,
            Π,
            c,
            MonotoneCubicInterp(),
        )

        @. nonbinding = a_next > (a_min + bind_tol)
        max_resid = sqrt(mean((resid_mat[nonbinding]) .^ 2))

        if verbose && it % 10 == 0
            @printf("[EGM stoch pchip] it=%d rmse=%.6e Δpol=%.6e\n", it, max_resid, Δpol)
            flush(stdout)
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    for (j, z) in enumerate(z_grid)
        y = exp(z)
        @views @. a_next[:, j] = clamp(R * a_grid + y - c[:, j], a_min, a_max)
    end

    euler_resid_stoch_interp!(
        resid_mat,
        model_params,
        a_grid,
        z_grid,
        Π,
        c,
        MonotoneCubicInterp(),
    )
    @. nonbinding = a_next > (a_min + bind_tol)
    max_resid = sqrt(mean((resid_mat[nonbinding]) .^ 2))

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol,
        tol_pol,
        maxit,
        interp_kind = MonotoneCubicInterp(),
        relax,
        verbose,
        resid_metric = :rmse,
        seed = nothing,
        runtime,
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
        rmse = max_resid,
        model_params,
        opts,
    )
end

end #module
