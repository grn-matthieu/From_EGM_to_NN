module EGMKernel

using ..EGMInterp: interp_linear!, interp_pchip!, InterpKind, LinearInterp, MonotoneCubicInterp
using ..EGMResiduals: euler_resid_det

export solve_egm_det


# --- Functions ---
"""
    function solve_egm_det(
        model_params, model_grids;
        tol::Real=1e-8, tol_pol::Real=1e-6, maxit::Int=500, verbose::Bool=false,
        interp_kind::InterpKind=LinearInterp(), relax::Real=0.5, patience::Int=50, ν::Real=1e-10,
    )
    Vectorized EGM with residual-based stopping. No recurring income; borrowing limit at `model_params.a_min`.
    Output : SimpleSolution
"""
function solve_egm_det(model_params, model_grids, model_utility;
        tol::Real=1e-8, tol_pol::Real=1e-6, maxit::Int=500,
        interp_kind::InterpKind=LinearInterp(), relax::Real=0.5, patience::Int=50, ν::Real=1e-10,
        c_init=nothing)::NamedTuple

    a = model_grids.a
    a_min = minimum(a)
    a_max = maximum(a)
    Na = length(a)

    γ = model_params.β * (1 + model_params.r)
    R = (1 + model_params.r)
    cmin  = 1e-12

    # Initial guess for resources and consumption
    resources = @. R * a - a_min + model_params.y

    # Initial guess for consumption : if not provided, consider half the resources
    c = c_init === nothing ? clamp.(0.5 .* resources, cmin, resources) : copy(c_init)

    # Buffer variables
    a_next = similar(c)
    cnext = similar(c)
    cnew = similar(c)
    a_next = similar(c)

    converged = false
    iters = 0
    max_resid = Inf
    Δpol = Inf
    best_resid = Inf
    no_progress = 0

    for it in 1:maxit
        # Store the current iteration in a buffer for the metadata
        iters = it

        @. a_next = model_params.y + R * a - c
        @. a_next = clamp(a_next, a_min, a_max)

        if interp_kind isa LinearInterp
            interp_linear!(cnext, a, c, a_next)
        elseif interp_kind isa MonotoneCubicInterp
            interp_pchip!(cnext, a, c, a_next)
        else
            @error "Unknown interpolation kind: $interp_kind"
        end

        @. cnew = model_utility.u_prime_inv(γ * cnext.^(-model_params.σ))
        cmax = @. model_params.y + R * a - a_min
        @. cnew = clamp(cnew, cmin, cmax)

        @. a_next = R * a + model_params.y - cnew
        @. a_next = clamp(a_next, a_min, a_max)

        # Only enforce monotonicity if using PCHIP
        if interp_kind isa MonotoneCubicInterp
            @inbounds for i in 2:Na
                if cnew[i] < cnew[i-1]
                    cnew[i] = cnew[i-1] + 1e-12
                end
            end
        end
        # Relaxation to stabilize on coarse grids
        c .= (1 - relax) .* c .+ relax .* cnew

        # Residual based stopping criteria : only stop when the max residual is below the tolerance
        resid = euler_resid_det(model_params, c, cnext)
        max_resid = maximum(resid[min(2, end):end])  # Ignore where the BC is binding so EE may hold as an inequality
        Δpol = maximum(abs.(c - cnew))

        # Stagnation Checks
        # We stop when no progress is made both in residuals and in policy
        if (best_resid - max_resid < ν) && (Δpol < ν)
            no_progress += 1
        else
            no_progress = 0
            best_resid = max_resid
        end

        if no_progress ≥ patience && verbose
            break
        end

        if max_resid < tol && Δpol < tol_pol
            converged = true
            break
        end
    end

    # Metadata
    opts = (;tol=tol, tol_pol=tol_pol, maxit=maxit, interp_kind=interp_kind, relax=relax, patience=patience, ν=ν)

    # Last a next consistent with c
    @. a_next = R * a + model_params.y - c
    @. a_next = clamp(a_next, a_min, a_max)

    return (;a, c, a_next, iters, converged, max_resid, model_params, opts)
end


end #module