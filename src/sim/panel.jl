module SimPanel

export simulate_panel

using Random
using Statistics
using ..Determinism:
    derive_rng, derive_seed, make_master_rng, make_rng, master_seed, promote_master_rng

using ..API:
    Solution, AbstractModel, AbstractMethod, get_params, get_grids, get_shocks, solve
using ..CommonInterp: interp_linear!


"""
    simulate_panel(model, method, cfg::NamedTuple; N = 1000, T = 200, rng = nothing)

Simulates a panel of N agents for T periods using a solved policy from `method`
on `model`. Randomness is driven by a master RNG: prefer `cfg.random.master_rng`
when present. The master RNG itself is never mutated;
instead deterministic sub-generators are derived for the solver call and each
agent.

Returns a NamedTuple with fields: `assets::Matrix`, `consumption::Matrix`,
`shocks::Matrix`, `seeds::Vector`, and `diagnostics::Vector`.
"""
function simulate_panel(
    model::AbstractModel,
    method::AbstractMethod,
    cfg::NamedTuple;
    N::Int = 1_000,
    T::Int = 200,
    master_rng = nothing,
)
    cand = master_rng !== nothing ? master_rng : make_master_rng(cfg.random.seed)
    resolved_master = promote_master_rng(cand)

    # Solve once and for all to get the optimal policy fun and grids
    solve_rng = derive_rng(resolved_master, :panel_solve)
    sol = solve(model, method, cfg; rng = solve_rng)

    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)

    agrid = g[:a].grid
    R = 1 + p.r

    zgrid = [0.0]
    Π = ones(1, 1)

    # Care : cpol can be a matrix (a,z) or a vector (a) depending on shocks
    cpol = sol.policy[:c].value

    assets = Matrix{Float64}(undef, N, T)
    cons = Matrix{Float64}(undef, N, T)
    zdraws = Matrix{Float64}(undef, N, T)
    seeds = Vector{UInt64}(undef, N)
    interp_out = Vector{Float64}(undef, 1)
    interp_xq = Vector{Float64}(undef, 1)


    # --- Seed handling ---
    if S === nothing
        # Deterministic model: document shocks as zeros in the output
        zdraws .= 0.0
    else
        zgrid = S.zgrid
        Π = S.Π
        π0 = S.π

        # Fun to sample from a row of a transition matrix Π
        # In the sim, useful bc we need to sample T times per agent
        # Note: Π assumed to be row-stochastic
        @inline function sample_row(Π::AbstractMatrix{<:Real}, i::Int, rng)
            u = rand(rng)
            s = 0.0
            @inbounds for j in axes(Π, 2)
                s += Π[i, j]
                if u <= s
                    return j
                end
            end
            return last(axes(Π, 2))
        end
    end

    @inbounds for n in axes(assets, 1)
        # independent rng per agent derived from the panel master rng
        agent_seed = derive_seed(master_rng, n)
        seeds[n] = agent_seed
        arng = make_rng(agent_seed)

        if S !== nothing
            # shocks indices and values
            idx = sample_row(reshape(π0, 1, :), 1, arng)  # sample initial state from π0
            for t in axes(assets, 2)
                zdraws[n, t] = zgrid[idx]
                idx = sample_row(Π, idx, arng)
            end
            # Note : the zdraws are now fixed !
            # shocks must be drawn before starting the asset loop
        end


        # a/c consumption path loop
        a_t = g[:a].min
        for t in axes(assets, 2)
            a_prev = a_t
            if cpol isa AbstractMatrix
                # nearest z index, then linear in a
                zval = zdraws[n, t]
                zj = searchsortedfirst(zgrid, zval)
                zj = clamp(zj, firstindex(zgrid), lastindex(zgrid))
                interp_xq[1] = a_prev
                interp_linear!(interp_out, agrid, view(cpol, :, zj), interp_xq)
                c_t = interp_out[1]
                cons[n, t] = c_t
                y_t = p.y * exp(zval)
                a_t = clamp(R * a_prev + y_t - c_t, g[:a].min, g[:a].max)
            else
                # deterministic: linear in a
                interp_xq[1] = a_prev
                interp_linear!(interp_out, agrid, cpol, interp_xq)
                c_t = interp_out[1]
                cons[n, t] = c_t
                y_t = p.y
                a_t = clamp(R * a_prev + y_t - c_t, g[:a].min, g[:a].max)
            end
            assets[n, t] = a_t
        end
    end
    # Manual computation of log consumption growth over time
    log_growth = Matrix{Float64}(undef, N, T - 1)
    @inbounds for t = 1:(T-1)
        log_growth[:, t] .= log.(cons[:, t+1]) .- log.(cons[:, t])
    end
    mean_log_c_growth = mean(log_growth)

    final_assets = view(assets, :, size(assets, 2))
    final_asset_mean = mean(final_assets)
    final_asset_std = std(final_assets)

    diagnostics = (
        rng_kind = string(typeof(master_rng)),
        master_seed = master_seed(master_rng),
        mean_log_c_growth = mean_log_c_growth,
        final_asset_mean = final_asset_mean,
        final_asset_std = final_asset_std,
    )

    return (; assets, consumption = cons, shocks = zdraws, seeds, diagnostics)
end

end
