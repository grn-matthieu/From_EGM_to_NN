module ThesisProjectPlotsExt

using ThesisProject
using Plots


import ThesisProject: plot_policy

function plot_policy(policy::NamedTuple; vars::AbstractVector{Symbol}=(:c,:ap,), x=nothing)
    xs = isnothing(x) ? eachindex(getproperty(policy, first(vars))) : x
    plt = plot()
    for k in vars
        hasproperty(policy, k) || continue
        plot!(plt, xs, getproperty(policy, k), label=String(k))
    end
    plt
end

end #module ThesisProjectPlotsExt