module PlotsUtils

export save_plot, plot_policy, plot_value, plot_residuals

using Plots
using ..EGMSolver: SimpleSolution
using ..ValueFunction: compute_value
using ..EGMResiduals  # <- your module with euler_residuals_simple

"Save a plot `plt` under runs/<runid>/<name>.png; creates dirs as needed."
function save_plot(plt, name; runid::AbstractString="default")
    outdir = joinpath("runs", runid)
    isdir(outdir) || mkpath(outdir)
    outfile = joinpath(outdir, string(name, ".png"))
    png(plt, outfile)
    @info "Saved plot to $outfile"
    return outfile
end

"Plot c(a) and a'(a)."
function plot_policy(sol::SimpleSolution; runid::AbstractString="default")
    plt1 = plot(sol.agrid, sol.c,
        xlabel="Assets a", ylabel="Consumption c(a)",
        label="c(a)", title="Consumption Policy")
    save_plot(plt1, "policy_consumption"; runid)

    plt2 = plot(sol.agrid, sol.a_next,
        xlabel="Assets a", ylabel="Next-period assets a'(a)",
        label="a'(a)", title="Asset Policy")
    save_plot(plt2, "policy_assets"; runid)

    return plt1, plt2
end

"Compute V(a) via policy evaluation and save."
function plot_value(sol::SimpleSolution, p; runid::AbstractString="default")
    V = compute_value(sol, p)
    plt = plot(sol.agrid, V,
        xlabel="Assets a", ylabel="Value V(a)",
        label="V(a)", title="Value Function")
    save_plot(plt, "value_function"; runid)
    return plt
end

"""
    plot_residuals(sol, p; runid="default", log10scale=true)

Compute Euler residuals using your EGMResiduals.euler_residuals_simple(p, agrid, c)
and save the plot. By default plots log10 residuals (capped from below for stability).
Returns (plt, resid_vec_plotted).
"""
function plot_residuals(sol::SimpleSolution, p; runid::AbstractString="default", log10scale::Bool=true)
    raw = EGMResiduals.euler_residuals_simple(p, sol.agrid, sol.c)  # your function

    # Skip the first point if it is not meaningful (e.g., at borrowing constraint)
    idx = 2:length(sol.agrid)
    xvals = sol.agrid[idx]
    yvals = raw[idx]

    resid = log10scale ? log10.(max.(yvals, 1e-16)) : yvals

    ylabel = log10scale ? "log10 Euler residual" : "Euler residual"
    fname  = log10scale ? "residuals" : "residuals_abs"

    plt = plot(xvals, resid,
        xlabel="Assets a", ylabel=ylabel,
        label=false, title="Euler Residuals")
    save_plot(plt, fname; runid)

    return plt, resid
end

end # module
