"""
plot_policy(...)
Stub: errors unless a plotting backend is loaded (via package extension).
"""
plot_policy(::Any; kwargs...) = error(
    "Visualization API 'plot_policy' requires a plotting backend. `using Plots` in your session."
)