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
using ..EulerResiduals: euler_resid_det!, euler_resid_stoch!, euler_resid_stoch_interp!
using ..PolicyUtils:
    clamp_policy!,
    compute_binding_tolerance,
    enforce_monotone!,
    ensure_minimum!,
    init_consumption_det,
    relaxation_step!,
    rmse_nonbinding
using Printf

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
    ϵ::Real = 1e-10,
    c_init = nothing,
    verbose::Bool = false,
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
        ϵ = ϵ,
        c_init = c_init,
        verbose = verbose,
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
    ϵ::Real = 1e-10,
    c_init = nothing,
    verbose::Bool = false,
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
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = 1e-12, rel = 0.0)

    c = init_consumption_det(a_grid, a_min, R, model_params.y; c_init = c_init, cmin = cmin)

    cnext = similar(c)
    cnew = similar(c)
    a_next = similar(c)
    resid = similar(c)
    cold = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    Δpol = Inf

    for it = 1:maxit
        iters = it
        # compute implied next assets from current policy
        @. a_next = clamp(model_params.y + R * a_grid - c, a_min, a_max)

        # interpolate consumption at a_next
        interp_linear!(cnext, a_grid, c, a_next)
        ensure_minimum!(cnext, cmin)

        # update consumption from Euler equation: c_new = (u'^{-1}(β R u'(c_next)))
        @. cnew = model_utility.u_prime_inv(β * R * cnext .^ (-σ))
        cmax = @. model_params.y + R * a_grid - a_min
        clamp_policy!(cnew, cmin, cmax)

        # policy progress: infinity norm of change between successive iterates
        copyto!(cold, c)
        Δpol = relaxation_step!(c, cold, cnew, relax)

        euler_resid_det!(resid, model_params, c, cnext)
        max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)

        if verbose && (it % 10 == 0)
            @printf("[TimeIteration] it=%d rmse=%.6e Δpol=%.6e\n", it, max_resid, Δpol)
            flush(stdout)
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end

        if best_resid - max_resid < ϵ && Δpol < ϵ
            # small improvements only; continue iterating without a patience cutoff
            # keep best_resid for diagnostics
        else
            best_resid = max_resid
        end
    end

    # final consistency
    @. a_next = clamp(R * a_grid + model_params.y - c, a_min, a_max)
    interp_linear!(cnext, a_grid, c, a_next)
    ensure_minimum!(cnext, cmin)
    euler_resid_det!(resid, model_params, c, cnext)
    max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        interp_kind = interp_kind,
        relax = relax,
        ϵ = ϵ,
        resid_metric = :rmse,
        seed = nothing,
        runtime = runtime,
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
        delta_pol = Δpol,
    )
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
    ϵ::Real = 1e-10,
    c_init = nothing,
    verbose::Bool = false,
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
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = 1e-12, rel = 0.0)

    c = init_consumption_det(a_grid, a_min, R, model_params.y; c_init = c_init, cmin = cmin)

    cnext = similar(c)
    cnew = similar(c)
    a_next = similar(c)
    resid = similar(c)
    cold = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    Δpol = Inf

    for it = 1:maxit
        iters = it
        @. a_next = clamp(model_params.y + R * a_grid - c, a_min, a_max)

        interp_pchip!(cnext, a_grid, c, a_next)
        ensure_minimum!(cnext, cmin)

        @. cnew = model_utility.u_prime_inv(β * R * cnext .^ (-σ))
        cmax = @. model_params.y + R * a_grid - a_min
        clamp_policy!(cnew, cmin, cmax)

        # monotone enforcement
        enforce_monotone!(cnew)

        # policy progress per formula Δ^{(k)}_∞ = max_{i,j} |c^{(k)} - c^{(k-1)}|
        copyto!(cold, c)
        Δpol = relaxation_step!(c, cold, cnew, relax)

        euler_resid_det!(resid, model_params, c, cnext)
        max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)

        if verbose && (it % 10 == 0)
            @printf(
                "[TimeIteration:PCHIP] it=%d rmse=%.6e Δpol=%.6e\n",
                it,
                max_resid,
                Δpol
            )
            flush(stdout)
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end

        if best_resid - max_resid < ϵ && Δpol < ϵ
            # small improvements only; continue iterating without a patience cutoff
            # keep best_resid for diagnostics
        else
            best_resid = max_resid
        end
    end

    @. a_next = clamp(R * a_grid + model_params.y - c, a_min, a_max)
    interp_pchip!(cnext, a_grid, c, a_next)
    ensure_minimum!(cnext, cmin)
    euler_resid_det!(resid, model_params, c, cnext)
    max_resid = rmse_nonbinding(resid, a_next, a_min, bind_tol)

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        interp_kind = interp_kind,
        relax = relax,
        ϵ = ϵ,
        resid_metric = :rmse,
        seed = nothing,
        runtime = runtime,
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
        delta_pol = Δpol,
    )
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
    c_init = nothing,
    verbose::Bool = false,
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
        c_init = c_init,
        verbose = verbose,
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
    c_init = nothing,
    verbose::Bool = false,
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
    cmax = similar(a_grid)
    bind_tol = compute_binding_tolerance(a_min, a_max, Na; floor = 1e-12, rel = 0.0)
    cold = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    best_resid = Inf
    Δpol = Inf

    for it = 1:maxit
        iters = it
        copyto!(cold, c)

        for (j, z) in enumerate(z_grid)
            y = exp(z)
            @views begin
                aj = view(a_next, :, j)
                cj = view(c, :, j)
                aj .= clamp.(R .* a_grid .+ y .- cj, a_min, a_max)
            end

            fill!(EUprime, 0.0)
            for (jp, _) in enumerate(z_grid)
                interp_linear!(cnext, a_grid, view(c, :, jp), view(a_next, :, j))
                ensure_minimum!(cnext, cmin)
                @. EUprime += Π[j, jp] * (cnext .^ (-σ))
            end

            column = view(cnew, :, j)
            @. column = model_utility.u_prime_inv(β * R * EUprime)
            @. cmax = y + R * a_grid - a_min
            clamp_policy!(column, cmin, cmax)
        end

        Δpol = relaxation_step!(c, cold, cnew, relax)

        for (j, z) in enumerate(z_grid)
            y = exp(z)
            @views @. a_next[:, j] = clamp(R * a_grid + y - c[:, j], a_min, a_max)
        end

        if verbose && (it % 10 == 0)
            @printf(
                "[TimeIteration:STOCH] it=%d rmse=%.6e Δpol=%.6e\n",
                it,
                max_resid,
                Δpol
            )
            flush(stdout)
        end

        euler_resid_stoch_interp!(
            resid_mat,
            model_params,
            a_grid,
            z_grid,
            Π,
            c,
            interp_kind,
        )
        # resid_mat is Na x Nz. Build mask of non-binding entries where a_next > a_min
        max_resid = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end

        if best_resid - max_resid < ϵ
            # small change; continue without a patience cutoff
        else
            best_resid = max_resid
        end
    end

    euler_resid_stoch_interp!(resid_mat, model_params, a_grid, z_grid, Π, c, interp_kind)
    max_resid = rmse_nonbinding(resid_mat, a_next, a_min, bind_tol)

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        tol = tol,
        tol_pol = tol_pol,
        maxit = maxit,
        interp_kind = interp_kind,
        relax = relax,
        ϵ = ϵ,
        resid_metric = :rmse,
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
        rmse = max_resid,
        model_params,
        opts,
        delta_pol = Δpol,
    )
end

function solve_ti_stoch_impl(::MonotoneCubicInterp, args...; kwargs...)
    # For simplicity delegate to linear implementation: cubic interp not implemented for stoch here
    return solve_ti_stoch_impl(LinearInterp(), args...; kwargs...)
end

end # module
