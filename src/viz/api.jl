"""
plot_policy(...)
Stub: errors unless a plotting backend is loaded (via package extension).
"""
plot_policy(::Any; kwargs...) = error(
    "Visualization API 'plot_policy' requires a plotting backend. `using Plots` in your session.",
)


"""
plot_euler_errors(...)
Stub: errors unless a plotting backend is loaded (via package extension).
"""
plot_euler_errors(::Any; kwargs...) = error(
    "Visualization API 'plot_euler_errors' requires a plotting backend. `using Plots` in your session.",
)
