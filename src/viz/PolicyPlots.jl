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
    policy = sol.policy
    # X-axis: assume the grid is available as `:a` in policy or elsewhere.
    hasproperty(policy, :a_grid) || error("Expected grid `:a_grid` in sol.policy for plotting.")
    x = getproperty(policy, :a_grid)

    sel = isnothing(vars) ? collect(propertynames(policy)) : collect(vars)
    plt = plot()
    for k in sel
        hasproperty(policy, k) || continue
        y = getproperty(policy, k)
        y isa AbstractVector || continue
        plot!(plt, x, y, label = isnothing(labels) ? String(k) :
                           (labels[findfirst(==(k), sel)]))
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
function plot_euler_error(sol; vars=nothing, labels=nothing)
    # Assume sol has fields: grid, euler_error (matrix or vector)
    grid = sol.grid
    euler_error = sol.euler_error

    if isnothing(vars)
        vars = 1:size(euler_error, 2)
    end
    if isnothing(labels)
        labels = ["Euler Error $i" for i in vars]
    end

    plt = plot()
    for (i, v) in enumerate(vars)
        plot!(plt, grid, abs.(euler_error[:, v]), label=labels[i])
    end
    xlabel!(plt, "State")
    ylabel!(plt, "Euler Error (abs)")
    yscale!(plt, :log10)
    title!(plt, "Euler Errors")
    return plt
end

export plot_policy
