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

    vars_to_plot = isnothing(vars) ? keys(sol.policy) : collect(vars)
    plt = plot()
    for k in vars_to_plot
        hasproperty(grid, k) || continue # Needed bc vars might include non existing vars
        x = sol.policy[Symbol(k)].grid
        y = sol.policy[Symbol(k)].value
        y isa AbstractVector || continue # Needed bc policy might not be a vector
        plot!(plt, x, y, label = isnothing(labels) ? String(k) :
                           (labels[findfirst(==(k), vars_to_plot)]))
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
