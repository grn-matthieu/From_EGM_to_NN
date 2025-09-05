module EGMKernel

using ..CommonInterp: interp_linear!, interp_pchip!, InterpKind, LinearInterp, MonotoneCubicInterp
using ..EGMResiduals: euler_resid_det, euler_resid_stoch

export solve_egm_det, solve_egm_stoch


"""
    solve_egm_det(model_params, model_grids, model_utility; ...)

Vectorized EGM with residual-based stopping, deterministic income.
Returns a NamedTuple with fields (a_grid, c, a_next, resid, iters, converged, max_resid, model_params, opts).
"""
function solve_egm_det(model_params, model_grids, model_utility;
        tol::Real=1e-8, tol_pol::Real=1e-6, maxit::Int=500,
        interp_kind::InterpKind=LinearInterp(), relax::Real=0.5, patience::Int=50, ϵ::Real=1e-10,
        c_init=nothing)::NamedTuple
    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min  = model_grids[:a].min
    a_max  = model_grids[:a].max
    Na     = model_grids[:a].N

    βR = model_params.β * (1 + model_params.r)
    R = (1 + model_params.r)
    cmin  = 1e-12

    # Initial guess for resources and consumption
    resources = @. R * a_grid - a_min + model_params.y
    c = c_init === nothing ? clamp.(0.5 .* resources, cmin, resources) : copy(c_init)

    # Buffers
    cnext = similar(c)
    cnew  = similar(c)
    a_next = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    no_progress = 0
    resid = similar(c)

    for it in 1:maxit
        iters = it

        @. a_next = model_params.y + R * a_grid - c
        @. a_next = clamp(a_next, a_min, a_max)

        if interp_kind isa LinearInterp
            interp_linear!(cnext, a_grid, c, a_next)
        elseif interp_kind isa MonotoneCubicInterp
            interp_pchip!(cnext, a_grid, c, a_next)
        else
            @error "Unknown interpolation kind: $interp_kind"
        end

        @. cnew = model_utility.u_prime_inv(βR * cnext.^(-model_params.σ))
        cmax = @. model_params.y + R * a_grid - a_min
        @. cnew = clamp(cnew, cmin, cmax)

        # Monotone enforcement under PCHIP
        if interp_kind isa MonotoneCubicInterp
            @inbounds for i in 2:Na
                if cnew[i] < cnew[i-1]
                    cnew[i] = cnew[i-1] + 1e-12
                end
            end
        end

        Δpol = maximum(abs.(c - cnew))
        c .= (1 - relax) .* c .+ relax .* cnew

        resid = euler_resid_det(model_params, c, cnext)
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

    runtime = (time_ns() - start_time) / 1e9
    opts = (; tol=tol, tol_pol=tol_pol, maxit=maxit, interp_kind=interp_kind, relax=relax,
             patience=patience, ϵ=ϵ, seed=nothing, runtime=runtime)

    return (; a_grid, c, a_next, resid, iters, converged, max_resid, model_params, opts)
end


"""
    solve_egm_stoch(model_params, model_grids, model_shocks, model_utility; ...)

Vectorized EGM for a model with Markov income shocks. Returns a NamedTuple with fields
(a_grid, z_grid, c, a_next, resid, iters, converged, max_resid, model_params, opts).
"""
function solve_egm_stoch(model_params, model_grids, model_shocks, model_utility;
        tol::Real=1e-8, tol_pol::Real = 1e-6, maxit::Int=500,
        interp_kind::InterpKind=LinearInterp(), relax::Real=0.5,
        ϵ::Real=1e-10, patience::Int=50, c_init=nothing)::NamedTuple

    start_time = time_ns()

    a_grid = model_grids[:a].grid
    a_min  = model_grids[:a].min
    a_max  = model_grids[:a].max
    Na     = model_grids[:a].N

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
    a_star = similar(c)
    cnext = similar(a_grid)
    cnew = similar(c)
    a_next = similar(c)
    EUprime = similar(a_grid)

    for it in 1:maxit
        iters = it

        for (j, z) in enumerate(z_grid)
            y = exp(z)

            @. a_star[:, j] = R * a_grid + y - c[:, j]
            @. a_star[:, j] = clamp(a_star[:, j], a_min, a_max)

            fill!(EUprime, 0.0)
            for (jp, _) in enumerate(z_grid)
                if interp_kind isa LinearInterp
                    interp_linear!(cnext, a_grid, view(c, :, jp), view(a_star, :, j))
                elseif interp_kind isa MonotoneCubicInterp
                    interp_pchip!(cnext, a_grid, view(c, :, jp), view(a_star, :, j))
                else
                    @error "Unknown interpolation kind: $interp_kind"
                end
                @. EUprime += Π[j, jp] * (cnext^(-σ))
            end

            @. cnew[:, j] = ((β * R) * EUprime)^(-1 / σ)
            cmax = @. y + R * a_grid - a_min
            @. cnew[:, j] = clamp(cnew[:, j], cmin, cmax)

            @. a_next[:, j] = R * a_grid + y - cnew[:, j]
            @. a_next[:, j] = clamp(a_next[:, j], a_min, a_max)

            if interp_kind isa MonotoneCubicInterp
                @inbounds for i in 2:Na
                    if cnew[i, j] < cnew[i-1, j]
                        cnew[i, j] = cnew[i-1, j] + 1e-12
                    end
                end
            end
        end

        Δpol = maximum(abs.(c - cnew))
        @. c = (1 - relax) * c + relax * cnew

        resid_mat = euler_resid_stoch(model_params, a_grid, z_grid, Π, c)
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

    runtime = (time_ns() - start_time) / 1e9
    opts = (; tol=tol, tol_pol=tol_pol, maxit=maxit, interp_kind=interp_kind, relax=relax,
             patience=patience, ϵ=ϵ, seed=nothing, runtime=runtime)

    return (; a_grid, z_grid, c, a_next, resid=euler_resid_stoch(model_params, a_grid, z_grid, Π, c),
             iters, converged, max_resid, model_params, opts)
end

end #module
