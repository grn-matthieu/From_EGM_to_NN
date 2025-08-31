using CSV, DataFrames, Statistics
using Plots, StatsPlots   # <- now includes boxplot & violin

# --- helper to summarize residuals ---
function summarize_residuals(df::DataFrame)
    res = df.residual
    return (
        min = minimum(res),
        max = maximum(res),
        mean = mean(res),
        median = median(res),
        std = std(res)
    )
end

# --- paths to the four runs ---
paths = Dict(
    :det_linear   => "runs/det_egm/deterministic_egm.csv",
    :det_pchip    => "runs/det_egm_pchip/deterministic_egm_pchip.csv",
    :stoch_linear => "runs/stochastic_egm/stochastic_egm.csv",
    :stoch_pchip  => "runs/stochastic_egm_pchip/stochastic_egm_pchip.csv"
)

data = Dict()
summaries = Dict()

for (k, path) in paths
    df = CSV.read(path, DataFrame)
    data[k] = df
    summaries[k] = summarize_residuals(df)
end

println("=== Residual summaries ===")
for (k, summ) in summaries
    println(k, " => ", summ)
end

# --- output folder for comparisons ---
outdir = "runs/comparisons"
isdir(outdir) || mkpath(outdir)

# --- log-histogram of residuals across all methods ---
plt_hist = plot(title="Log10(|Euler residuals|) across methods",
    xlabel="log10(|residual|)", ylabel="Density")
for k in keys(data)
    res = data[k].residual
    logres = log10.(abs.(res) .+ 1e-16)
    histogram!(plt_hist, logres, bins=100, alpha=0.4, normalize=:pdf, label=string(k))
end
savefig(plt_hist, joinpath(outdir, "log_hist_residuals.png"))

# --- prepare stacked data for grouped plots ---
groups = String[]
vals = Float64[]
for (k, df) in data
    append!(groups, fill(string(k), nrow(df)))
    append!(vals, df.residual)
end
logvals = log10.(abs.(vals) .+ 1e-16)

# --- boxplot of residual distributions ---
plt_box = boxplot(groups, logvals,
    xlabel="Method", ylabel="log10(|residual|)",
    title="Residual distributions by method", legend=false, rotation=15)
savefig(plt_box, joinpath(outdir, "boxplot_residuals.png"))

# --- violin plot of residual distributions ---
plt_violin = violin(groups, logvals,
    xlabel="Method", ylabel="log10(|residual|)",
    title="Residual distributions (violin)", legend=false, rotation=15)
savefig(plt_violin, joinpath(outdir, "violin_residuals.png"))

# --- residuals vs assets (averaging across z if stochastic) ---
plt_assets = plot(title="Residuals vs Assets (mean over shocks)", xlabel="Assets a", ylabel="mean |residual|")
for (k, df) in data
    if :z in names(df)
        gdf = combine(groupby(df, :a), :residual => x -> mean(abs.(x)) => :mean_resid)
        plot!(plt_assets, gdf.a, gdf.mean_resid, label=string(k))
    else
        plot!(plt_assets, df.a, abs.(df.residual), label=string(k))
    end
end
savefig(plt_assets, joinpath(outdir, "residuals_vs_assets.png"))
