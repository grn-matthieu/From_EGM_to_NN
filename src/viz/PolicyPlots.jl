module PolicyPlots

"""
    plot_policy(sol; vars=nothing, labels=nothing)

Plot the policy functions from a Solution struct.
- `vars`: Optional vector of variable symbols or indices to plot.
- `labels`: Optional vector of labels for the plotted variables.

Returns a Plots.Plot object (Plots lib compatible.)
"""
function plot_policy(sol; vars=nothing, labels=nothing)
    # Assume sol has fields: grid, policy (matrix or vector)
    grid = sol.policy.a_grid
    policy = sol.policy

    if isnothing(vars)
        vars = 1:size(policy, 2)
    end
    if isnothing(labels)
        labels = ["Policy $i" for i in vars]
    end

    plt = plot()
    for (k, y) in pairs(policy)                 # k::String, y::AbstractVector
        k in vars || continue
        plot!(plt, grid, y, label=labels[findfirst(==(k), vars)])
    end
    xlabel!(plt, "State variable $k")
    ylabel!(plt, "Policy")
    title!(plt, "Policy Functions")
    return plt
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

end # module