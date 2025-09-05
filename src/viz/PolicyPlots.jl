 import ThesisProject: plot_policy, plot_euler_errors

"""
    plot_policy(sol; vars=nothing, labels=nothing)

Plot the policy functions from a Solution struct. Default behaviour is to plot every var in the sol.policy.
- `vars`: Optional vector of variable symbols or indices to plot.
- `labels`: Optional vector of labels for the plotted variables.

Returns a Plots.Plot object (Plots lib compatible.)
"""
function plot_policy(sol::ThesisProject.API.Solution;
                     vars::Union{Nothing,AbstractVector{Symbol}}=nothing,
                     labels::Union{Nothing,AbstractVector{String}}=nothing)

    vars_to_plot = isnothing(vars) ? collect(keys(sol.policy)) : collect(vars) # Collect needed when vars is nothing
    plt = plot()
    for (i, k) in enumerate(vars_to_plot)
        haskey(sol.policy, k) || continue # vars might include non-existing entries
        buffer_policy = sol.policy[k]
        x = getproperty(buffer_policy, :grid)
        y = getproperty(buffer_policy, :value)
        y isa AbstractVector || continue # some policies may be scalars or structs
        # For labeling, we use the k-th label of the label list. If not enough labels : variable name.
        lab = isnothing(labels) ? String(k) : (i <= length(labels) ? labels[i] : String(k))
        plot!(plt, x, y, label = lab)
    end
    xlabel!(plt, "state")
    ylabel!(plt, "policy")
    title!(plt, "Policy Functions")
    plt
end



"""
    plot_euler_errors(sol; vars=nothing, labels=nothing)

Plot the Euler errors from a Solution object `sol`.
- `vars`: Optional vector of variable symbols or indices to plot.
- `labels`: Optional vector of labels for the plotted variables.

Returns a Plots.Plot object.
"""
function plot_euler_errors(sol::ThesisProject.API.Solution)
    plt = plot()
    x = sol.policy[:a].grid # Temporary for the CS model, where a is the state var
    y = sol.policy[:c].euler_errors
    plot!(plt, x, y, yscale=:log10)
    xlabel!(plt, "State")
    ylabel!(plt, "Abs. EErr (in log)")
    title!(plt, "Euler Errors (method : $(sol.diagnostics.method))")
    return plt
end

export plot_policy, plot_euler_errors
