# --- Desc ---
# This API is optional, when the user wants graphic visualization.


function plot_policy(::Any; kwargs...)
    error("plot_policy requires a plotting backend. `using Plots` in your session.")
end

export plot_policy