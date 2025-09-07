 import ThesisProject: plot_policy, plot_euler_errors

"""
    plot_policy(sol; vars=nothing, labels=nothing, mean=false, shock_weights=nothing, surface=false)

Plot policy functions from a `Solution`. Handles deterministic (vector) and stochastic (matrix) policies.
- `vars`: Vector of variable symbols to plot (defaults to all in `sol.policy`).
- `labels`: Optional variable-level labels. For matrix-valued policies, shock-state labels are auto-derived.
- `mean`: If `true` and the policy is a matrix over shocks, overlay the stationary-weighted mean policy.
- `shock_weights`: Optional weights vector to compute the mean (defaults to invariant distribution if available, else uniform).
- `surface`: If truthy and policy is a matrix, render a 2D heatmap or 3D surface over `(a, z)` instead of per-shock lines. Accepts
  - `true` or `:surface` for 3D surface
  - `:heatmap` for a 2D heatmap

Returns a `Plots.Plot` object (requires `using Plots`) else dispatches to the generic fun.
"""
function plot_policy(sol::ThesisProject.API.Solution;
                     vars::Union{Nothing,AbstractVector{Symbol}}=nothing,
                     labels::Union{Nothing,AbstractVector{String}}=nothing,
                     mean::Bool=false,
                     shock_weights::Union{Nothing,AbstractVector}=nothing,
                     surface::Union{Bool,Symbol}=false)

    vars_to_plot = isnothing(vars) ? collect(keys(sol.policy)) : collect(vars)
    plt = plot()

    # Try to pull z-grid and invariant weights for stochastic labeling when needed
    zgrid = try
        getproperty(getproperty(sol, :model), :shocks) === nothing ? nothing : getproperty(getproperty(sol, :model), :shocks).zgrid
    catch
        nothing
    end

    # Try to pull invariant distribution π (or fallbacks) for mean overlay
    # Try chunks to avoid errors if some fields are missing
    zweights = nothing
    try
        S = getproperty(sol, :model) |> x -> getproperty(x, :shocks)
        if S !== nothing
            # Prefer π if present; else pi
            try
                zweights = getproperty(S, Symbol("π"))
            catch
                try
                    zweights = getproperty(S, :pi)
                catch
                    zweights = nothing
                end
            end
        end
    catch
        zweights = nothing
    end


    for (i, k) in enumerate(vars_to_plot)
        haskey(sol.policy, k) || continue
        pol = sol.policy[k]
        x = getproperty(pol, :grid)
        y = getproperty(pol, :value)

        # Base label for this variable ; for matrices, per-shock labels are derived below (from k).
        base_lab = isnothing(labels) ? String(k) : (i <= length(labels) ? labels[i] : String(k))

        if y isa AbstractVector # Deterministic policy or 1 sim
            plot!(plt, x, y, label = base_lab)
        elseif y isa AbstractMatrix
            nseries = size(y, 2)
            # Build per-shock labels if we have zgrid; otherwise index-based
            if zgrid !== nothing && length(zgrid) == nseries
                series_labels = ["$(base_lab) (z=$(round(z, digits=3)))" for z in zgrid]
            else
                series_labels = ["$(base_lab) [s=$(j)]" for j in 1:nseries]
            end

            # If requested, draw a heatmap/surface over (a,z) (fancy)
            if surface != false
                zaxis = (zgrid !== nothing && length(zgrid) == nseries) ? zgrid : collect(1:nseries)
                if surface === true || surface === :surface
                    surface!(plt, x, zaxis, y'; label = base_lab)
                elseif surface === :heatmap
                    heatmap!(plt, x, zaxis, y'; label = base_lab)
                else
                    # unknown value → default to lines
                    for j in 1:nseries
                        plot!(plt, x, view(y, :, j), label = series_labels[j])
                    end
                end
            else
                # Plot one line per shock column
                for j in 1:nseries
                    plot!(plt, x, view(y, :, j), label = series_labels[j])
                end
            end

            # Optional mean overlay to highlight average policy
            if mean
                w = shock_weights !== nothing ? shock_weights : (zweights !== nothing ? zweights : fill(1.0 / nseries, nseries))
                # ensure weights length matches
                if length(w) == nseries
                    ymean = y * collect(w)
                    plot!(plt, x, ymean, label = "$(base_lab) (mean)", lw=3, ls=:dash)
                end
            end
        else
            # Skip unsupported policy payloads (e.g., scalars or structs)
            continue
        end
    end

    xlabel!(plt, "state")
    ylabel!(plt, "policy")
    title!(plt, "Policy Functions")
    return plt
end



"""
    plot_euler_errors(sol; by=:auto)

Plot Euler errors. For stochastic models, supports aggregating across shocks if full errors are available.
- `by`: One of `:auto`, `:max`, or `:mean`.
  - `:auto`: uses the vector stored in the policy (max across shocks if stochastic) and labels accordingly.
  - `:max` / `:mean`: if a full error matrix is available (via solver), aggregates across shocks per asset.

Returns a `Plots.Plot` object.
"""
function plot_euler_errors(sol::ThesisProject.API.Solution; by::Symbol=:auto)
    plt = plot()
    x = sol.policy[:a].grid
    cpol = sol.policy[:c]

    # Choose vector to plot
    y = cpol.euler_errors
    label_suffix = "(stored)"
    if by != :auto
        emat = hasproperty(cpol, :euler_errors_mat) ? getproperty(cpol, :euler_errors_mat) : nothing
        if emat !== nothing && emat isa AbstractMatrix
            if by == :max
                y = vec(maximum(emat, dims=2))
                label_suffix = "(max across shocks)"
            elseif by == :mean
                # weight by invariant distribution if available
                S = try getproperty(sol.model, :shocks) catch; nothing end
                if S !== nothing
                    w = nothing
                    try
                        w = getproperty(S, Symbol("π"))
                    catch
                        try
                            w = getproperty(S, :pi)
                        catch
                            w = nothing
                        end
                    end
                    if w === nothing || length(w) != size(emat, 2)
                        w = fill(1.0 / size(emat, 2), size(emat, 2))
                    end
                    y = emat * collect(w)
                else
                    y = mean(emat; dims=2)[:]
                end
                label_suffix = "(mean across shocks)"
            end
        else
            # Fall back to stored vector if matrix not available
            label_suffix = "(stored)"
        end
    else
        # If auto, try to infer whether this is a stochastic reduction
        label_suffix = hasproperty(cpol, :euler_errors_mat) && (getproperty(cpol, :euler_errors_mat) isa AbstractMatrix) ?
                       "(max across shocks)" : ""
    end

    plot!(plt, x, y, yscale = :log10, label = isempty(label_suffix) ? nothing : label_suffix)
    xlabel!(plt, "State")
    ylabel!(plt, "Abs. EErr (log10)")
    title!(plt, "Euler Errors (method: $(sol.diagnostics.method))")
    return plt
end

export plot_policy, plot_euler_errors
