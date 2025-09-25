"""
TimeIterationKernel

Simple time-iteration (policy-update) solver kernel for the ConsumerSaving model.
This implements a fixed-point iteration on the Euler equation using grid interpolation
for next-period consumption. It mirrors the EGM solver's API so it can be used by the
existing `methods` adapter.
"""
module TimeIterationKernel

using ..CommonInterp:
    interp_linear!, interp_pchip!, InterpKind, LinearInterp, MonotoneCubicInterp
using ..EulerResiduals: euler_resid_det!, euler_resid_stoch!

export solve_ti_det, solve_ti_stoch

function solve_ti_det(
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-8,
    tol_pol::Real = 1e-6,
    maxit::Int = 500,
    interp_kind::InterpKind = LinearInterp(),
    relax::Real = 0.5,
    patience::Int = 50,
    ϵ::Real = 1e-10,
    c_init = nothing,
)::NamedTuple
    return solve_ti_det_impl(
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

solve_ti_det_impl(::InterpKind, args...; kwargs...) =
    error("Unknown interp kind for TimeIteration (det)")

function solve_ti_det_impl(
    interp_kind::LinearInterp,
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-8,
    tol_pol::Real = 1e-6,
    maxit::Int = 500,
    relax::Real = 0.5,
    patience::Int = 50,
    ϵ::Real = 1e-10,
    c_init = nothing,
)
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N

    R = 1 + model_params.r
    β = model_params.β
    σ = model_params.σ
    cmin = 1e-12

    resources = @. R * a_grid - a_min + model_params.y
    c = c_init === nothing ? clamp.(0.5 .* resources, cmin, resources) : copy(c_init)

    cnext = similar(c)
    cnew = similar(c)
    a_next = similar(c)
    resid = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    no_progress = 0

    for it = 1:maxit
        iters = it
        # compute implied next assets from current policy
        @. a_next = model_params.y + R * a_grid - c
        @. a_next = clamp(a_next, a_min, a_max)

        # interpolate consumption at a_next
        interp_linear!(cnext, a_grid, c, a_next)
        @. cnext = max(cnext, cmin)

        # update consumption from Euler equation: c_new = (u'^{-1}(β R u'(c_next)))
        @. cnew = model_utility.u_prime_inv(β * R * cnext .^ (-σ))
        cmax = @. model_params.y + R * a_grid - a_min
        @. cnew = clamp(cnew, cmin, cmax)

        Δpol = maximum(abs.(c - cnew))
        @. c = (1 - relax) * c + relax * cnew

        euler_resid_det!(resid, model_params, c, cnext)
        max_resid = maximum(resid[min(2, end):end])

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end

        if best_resid - max_resid < ϵ && Δpol < ϵ
            no_progress += 1
        else
            no_progress = 0
            best_resid = max_resid
        end

        if no_progress ≥ patience
            break
        end
    end

    # final consistency
    @. a_next = R * a_grid + model_params.y - c
    @. a_next = clamp(a_next, a_min, a_max)
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

# Monotone cubic variant (PCHIP)
function solve_ti_det_impl(
    interp_kind::MonotoneCubicInterp,
    model_params,
    model_grids,
    model_utility;
    tol::Real = 1e-8,
    tol_pol::Real = 1e-6,
    maxit::Int = 500,
    relax::Real = 0.5,
    patience::Int = 50,
    ϵ::Real = 1e-10,
    c_init = nothing,
)
    # For brevity reuse linear implementation but substitute cubic interp where used
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min = model_grids[:a].min
    a_max = model_grids[:a].max
    Na = model_grids[:a].N

    R = 1 + model_params.r
    β = model_params.β
    σ = model_params.σ
    cmin = 1e-12

    resources = @. R * a_grid - a_min + model_params.y
    c = c_init === nothing ? clamp.(0.5 .* resources, cmin, resources) : copy(c_init)

    cnext = similar(c)
    cnew = similar(c)
    a_next = similar(c)
    resid = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    no_progress = 0

    for it = 1:maxit
        iters = it
        @. a_next = model_params.y + R * a_grid - c
        @. a_next = clamp(a_next, a_min, a_max)

        interp_pchip!(cnext, a_grid, c, a_next)
        @. cnext = max(cnext, cmin)

        @. cnew = model_utility.u_prime_inv(β * R * cnext .^ (-σ))
        cmax = @. model_params.y + R * a_grid - a_min
        @. cnew = clamp(cnew, cmin, cmax)

        # monotone enforcement
        @inbounds for i = 2:Na
            if cnew[i] < cnew[i-1]
                cnew[i] = cnew[i-1] + 1e-12
            end
        end

        Δpol = maximum(abs.(c - cnew))
        @. c = (1 - relax) * c + relax * cnew

        euler_resid_det!(resid, model_params, c, cnext)
        max_resid = maximum(resid[min(2, end):end])

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end

        if best_resid - max_resid < ϵ && Δpol < ϵ
            no_progress += 1
        else
            no_progress = 0
            best_resid = max_resid
        end

        if no_progress ≥ patience
            break
        end
    end

    @. a_next = R * a_grid + model_params.y - c
    @. a_next = clamp(a_next, a_min, a_max)
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


# --- Stochastic variant ---
function solve_ti_stoch(
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-8,
    tol_pol::Real = 1e-6,
    maxit::Int = 1000,
    interp_kind::InterpKind = LinearInterp(),
    relax::Real = 0.5,
    ϵ::Real = 1e-10,
    patience::Int = 50,
    c_init = nothing,
)
    return solve_ti_stoch_impl(
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

solve_ti_stoch_impl(::InterpKind, args...; kwargs...) =
    error("Unknown interp kind for TimeIteration (stoch)")

function solve_ti_stoch_impl(
    interp_kind::LinearInterp,
    model_params,
    model_grids,
    model_shocks,
    model_utility;
    tol::Real = 1e-8,
    tol_pol::Real = 1e-6,
    maxit::Int = 1000,
    relax::Real = 0.5,
    ϵ::Real = 1e-10,
    patience::Int = 50,
    c_init = nothing,
)
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

    c = c_init === nothing ? fill(1.0, Na, Nz) : copy(c_init)
    a_next = similar(c)
    cnext = similar(a_grid)
    cnew = similar(c)
    EUprime = similar(a_grid)
    resid_mat = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    no_progress = 0

    for it = 1:maxit
        iters = it
        for (j, z) in enumerate(z_grid)
            y = exp(z)
            @. a_next[:, j] = R * a_grid + y - c[:, j]
            @. a_next[:, j] = clamp(a_next[:, j], a_min, a_max)

            fill!(EUprime, 0.0)
            for (jp, _) in enumerate(z_grid)
                interp_linear!(cnext, a_grid, view(c, :, jp), view(a_next, :, j))
                @. cnext = max(cnext, cmin)
                @. EUprime += Π[j, jp] * (cnext .^ (-σ))
            end

            @. cnew[:, j] = ((β * R) .* EUprime) .^ (-1 / σ)
            cmax = @. y + R * a_grid - a_min
            @. cnew[:, j] = clamp(cnew[:, j], cmin, cmax)

            @. a_next[:, j] = R * a_grid + y - cnew[:, j]
            @. a_next[:, j] = clamp(a_next[:, j], a_min, a_max)
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

function solve_ti_stoch_impl(::MonotoneCubicInterp, args...; kwargs...)
    # For simplicity delegate to linear implementation: cubic interp not implemented for stoch here
    return solve_ti_stoch_impl(LinearInterp(), args...; kwargs...)
end

end # module
