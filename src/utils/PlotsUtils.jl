module PlotsUtils

export save_plot,
       plot_policy,            # dispatches: deterministic OR stochastic
       plot_value,             # deterministic only
       plot_residuals          # deterministic OR stochastic

using Plots
using ..EGMSolver: SimpleSolution
using ..EGMResiduals: euler_residuals_simple, euler_residuals_stochastic
using ..ValueFunction: compute_value
using ..SimpleCalibration: SimpleParams

"Save a plot `plt` under runs/<runid>/<name>.png; creates dirs as needed."
function save_plot(plt, name; runid::AbstractString="default", verbose::Bool=true)
    outdir = joinpath("runs", runid)
    isdir(outdir) || mkpath(outdir)
    outfile = joinpath(outdir, string(name, ".png"))
    png(plt, outfile)
    if verbose
        @info "Saved plot to $outfile"
    end
    return outfile
end

# ---------------------------
# Deterministic: SimpleSolution
# ---------------------------

function plot_policy(sol::SimpleSolution, p::SimpleParams; runid::AbstractString="default", verbose::Bool=true)
    plt1 = plot(sol.agrid, sol.c,
        xlabel="Assets a", ylabel="Consumption c(a)",
        label="c(a)", title="Consumption Policy")
    # Add the true policy Function
    # To be corrected later
    # true_pol = similar(sol.c)
#=     R = 1 + p.r
    γ = (p.β*R)^(1/p.σ)
    @. true_pol = min((1 - γ/R)*(sol.agrid + p.y/p.r), p.y + R*sol.agrid - sol.agrid[1])
    plot!(sol.agrid, true_pol, label="True c(a)", linestyle=:dash) =#
    save_plot(plt1, "policy_consumption"; runid, verbose)

    plt2 = plot(sol.agrid, sol.a_next,
        xlabel="Assets a", ylabel="Next-period assets a'(a)",
        label="a'(a)", title="Asset Policy")
    save_plot(plt2, "policy_assets"; runid, verbose)

    return plt1, plt2
end

function plot_value(sol::SimpleSolution, p; runid::AbstractString="default")
    V = compute_value(sol, p)
    plt = plot(sol.agrid, V,
        xlabel="Assets a", ylabel="Value V(a)",
        label="V(a)", title="Value Function")
    save_plot(plt, "value_function"; runid)
    return plt
end

function plot_residuals(sol::SimpleSolution, p; runid::AbstractString="default", log10scale::Bool=true, verbose::Bool=true)
    raw = euler_residuals_simple(p, sol.agrid, sol.c)
    idx = 2:length(sol.agrid)
    xvals = sol.agrid[idx]
    yvals = raw[idx]
    vals = log10scale ? log10.(max.(yvals, 1e-16)) : yvals
    ylabel = log10scale ? "log10 Euler residual" : "Euler residual"
    fname  = log10scale ? "residuals" : "residuals_abs"

    plt = plot(xvals, vals, xlabel="Assets a", ylabel=ylabel,
               label=false, title="Euler Residuals")
    save_plot(plt, fname; runid, verbose)
    return plt, vals
end

# ---------------------------
# Stochastic: NamedTuple
# ---------------------------

function plot_policy(sol::NamedTuple; runid::AbstractString="default")
    @assert all(haskey(sol, k) for k in (:agrid, :zgrid, :c, :a_next)) "sol missing required fields"
    A, Z = sol.agrid, sol.zgrid

    plt1 = heatmap(A, Z, sol.c'; xlabel="Assets a", ylabel="Shock z (log y)",
                   title="Consumption c(a,z)", colorbar_title="c")
    save_plot(plt1, "policy_consumption_surface"; runid)

    plt2 = heatmap(A, Z, sol.a_next'; xlabel="Assets a", ylabel="Shock z (log y)",
                   title="Next assets a'(a,z)", colorbar_title="a'")
    save_plot(plt2, "policy_assets_surface"; runid)

    return plt1, plt2
end

function plot_residuals(sol::NamedTuple, p, Pz; runid::AbstractString="default", log10scale::Bool=true)
    @assert all(haskey(sol, k) for k in (:agrid, :zgrid, :c)) "sol missing required fields"

    R = euler_residuals_stochastic(p, sol.agrid, sol.zgrid, Pz, sol.c)
    A, Z = sol.agrid, sol.zgrid

    vals = log10scale ? log10.(max.(R, 1e-16)) : R
    ylabel = log10scale ? "log10 Euler residual" : "Euler residual"
    fname  = log10scale ? "residuals_heatmap" : "residuals_heatmap_abs"

    plt = heatmap(A, Z, vals'; xlabel="Assets a", ylabel="Shock z (log y)",
                  title="Euler Residuals", colorbar_title=ylabel)
    save_plot(plt, fname; runid)

    idxa = 2:length(A)
    max_over_z = [maximum(R[i, :]) for i in idxa]
    return plt, max_over_z
end

end # module
