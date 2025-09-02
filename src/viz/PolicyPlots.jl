 import ThesisProject: plot_policy

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
    plot_euler_error(sol; vars=nothing, labels=nothing)

Plot the Euler errors from a Solution object `sol`.
- `vars`: Optional vector of variable symbols or indices to plot.
- `labels`: Optional vector of labels for the plotted variables.

Returns a Plots.Plot object.
"""
function plot_euler_error(sol::ThesisProject.API.Solution;
                     vars::Union{Nothing,AbstractVector{Symbol}}=nothing,
                     labels::Union{Nothing,AbstractVector{String}}=nothing)
    # Function assumes that grid as well as euler_errors are in the sol
    grid = sol.grid
    euler_errors = sol.euler_errors

    if isnothing(vars)
        vars = 1:size(euler_errors, 2)
    end
    if isnothing(labels)
        labels = ["Euler Error $i" for i in vars]
    end

    plt = plot()
    for (i, v) in enumerate(vars)
        plot!(plt, grid[:i].grid, abs.(euler_errors[:, v]), label=labels[i])
    end
    xlabel!(plt, "State")
    ylabel!(plt, "Euler Error (abs)")
    yscale!(plt, :log10)
    title!(plt, "Euler Errors")
    return plt
end

export plot_policy
