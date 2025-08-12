module EGMSolver
export solve_simple_egm, SimpleSolution

using ..SimpleModel
using LinearAlgebra

Base.@kwdef struct SimpleSolution
    agrid::Vector{Float64}
    c::Vector{Float64}
    a_next::Vector{Float64}
    iters::Int
    converged::Bool
end

"""
    solve_simple_egm(p, agrid; tol=1e-6, maxit=500, verbose=true)

Run a basic EGM iteration for the simple model with no shocks.
"""
function solve_simple_egm(p, agrid; tol=1e-6, maxit=500, verbose=true)
    Na = length(agrid)
    c = fill( (p.y + (1+p.r)*agrid[1] - p.a_min) / 2, Na)  # initial guess
    a_next = similar(c)
    iters = 0
    converged = false

    for it in 1:maxit
        iters = it

        # Euler: u'(c_t) = β(1+r) u'(c_{t+1}), deterministic steady state
        uprime_t = c .^ (-p.σ)
        uprime_tp1 = p.β * (1+p.r) .* uprime_t
        c_new = uprime_tp1 .^ (-1/p.σ)

        # enforce feasibility
        c_new = min.(c_new, p.y .+ (1+p.r).*agrid .- p.a_min)

        # implied a'
        a_next .= budget_next.(agrid, p.y, p.r, c_new)
        a_next = clamp.(a_next, p.a_min, p.a_max)

        diff = maximum(abs.(c_new .- c))
        verbose && it % 10 == 0 && @info "iter=$it diff=$(diff)"
        c .= c_new
        if diff < tol
            converged = true
            break
        end
    end

    return SimpleSolution(collect(agrid), c, a_next, iters, converged)
end

end # module