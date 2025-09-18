"""
plot_policy(sol; kwargs...)

Minimal plotting fallback used when the Plots extension is not active. If Plots
is available, renders a simple policy plot; otherwise errors with guidance.
"""
function plot_policy(sol; kwargs...)
    isdefined(Main, :Plots) || error(
        "Visualization API 'plot_policy' requires a plotting backend. `using Plots` in your session.",
    )
    PlotsRef = Main.Plots
    # Best-effort: draw c(a) if available
    a = haskey(sol.policy, :a) ? sol.policy[:a].grid : nothing
    c = haskey(sol.policy, :c) ? sol.policy[:c].value : nothing
    if a !== nothing && c !== nothing && c isa AbstractVector
        return PlotsRef.plot(a, c; label = "c(a)")
    else
        plt = PlotsRef.plot()
        PlotsRef.plot!(plt; title = "Policy Plot (extension not loaded)")
        return plt
    end
end


"""
plot_euler_errors(sol; kwargs...)

Minimal plotting fallback used when the Plots extension is not active. If Plots
is available, renders Euler errors if present; otherwise errors with guidance.
"""
function plot_euler_errors(sol; kwargs...)
    isdefined(Main, :Plots) || error(
        "Visualization API 'plot_euler_errors' requires a plotting backend. `using Plots` in your session.",
    )
    PlotsRef = Main.Plots
    a = haskey(sol.policy, :a) ? sol.policy[:a].grid : nothing
    ce = try
        haskey(sol.policy, :c) ? getproperty(sol.policy[:c], :euler_errors) : nothing
    catch
        nothing
    end
    if a !== nothing && ce !== nothing
        return PlotsRef.plot(a, ce; yscale = :log10, label = "Euler Error")
    else
        plt = PlotsRef.plot()
        PlotsRef.plot!(plt; title = "Euler Errors (extension not loaded)")
        return plt
    end
end
