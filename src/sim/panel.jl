module SimPanel

export simulate_panel

using Random
using Statistics
using ..Determinism: make_rng, derive_seed

using ..API:
    Solution, AbstractModel, AbstractMethod, get_params, get_grids, get_shocks, solve
using ..CommonInterp: interp_linear!


"""
    simulate_panel(model, method, cfg; N=1000, T=200, rng::AbstractRNG)

Simulates a panel of N agents for T periods using a solved policy from `method` on `model`.
Agents draw from the Markov chain implied by the model's shocks. The master seed is taken from `cfg[:random][:seed]` if available, otherwise it is deterministically derived from the provided `rng` via `derive_seed`.

Returns a NamedTuple with fields: assets::Matrix, consumption::Matrix, shocks::Matrix, seeds::Vector and diagnostics::Vector.
"""
function simulate_panel(
    model::AbstractModel,
    method::AbstractMethod,
    cfg::AbstractDict;
    N::Int = 1_000,
    T::Int = 200,
    rng::AbstractRNG,
)
    # Solve once and for all to get the optimal policy fun and grids
    sol = solve(model, method, cfg; rng = rng)

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
    # Master RNG/seed: prefer cfg.random.seed; else derive from the provided rng
    # The rule is to derive individual agent seeds from the master seed so that
    # identical rng instances lead to identical panel simulations.
    master_seed = get(get(cfg, :random, Dict()), :seed, nothing)
    if master_seed === nothing
        master_seed = derive_seed(rng, :panel)
    else
        master_seed = UInt64(master_seed)
    end
    master_rng = make_rng(master_seed)

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
        # independent rng per agent derived from the master rng
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
    log_growth = diff(log.(cons); dims = 2)
    mean_log_c_growth = mean(log_growth)

    final_assets = view(assets, :, size(assets, 2))
    final_asset_mean = mean(final_assets)
    final_asset_std = std(final_assets)

    diagnostics = (
        rng_kind = string(typeof(master_rng)),
        master_seed = master_seed,
        mean_log_c_growth = mean_log_c_growth,
        final_asset_mean = final_asset_mean,
        final_asset_std = final_asset_std,
    )

    return (; assets, consumption = cons, shocks = zdraws, seeds, diagnostics)
end

end
