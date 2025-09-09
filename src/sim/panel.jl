module SimPanel

export simulate_panel

using Random
using Statistics
using Logging
using ..Determinism: make_rng, derive_seed, canonicalize_cfg, hash_hex

using ..API:
    Solution, AbstractModel, AbstractMethod, get_params, get_grids, get_shocks, solve


"""
    simulate_panel(model, method, cfg; N=1000, T=200, rng::AbstractRNG))

Simulates a panel of N agents for T periods using a solved policy from `method` on `model`.
Agents draw from the Markov chain implied by the model's shocks. The master seed is derived from the rng optionnaly provided.
Passing a pre-seeded RNG ensures reproducibility.

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
    sol = solve(model, method, cfg)

    p = get_params(model)
    g = get_grids(model)
    S = get_shocks(model)

    agrid = g[:a].grid
    R = 1 + p.r

    # Require shocks (panel simulation targets stochastic models)
    # If the model is deterministic then simulation = 1 representative agent
    # Thus sim is not really a panel but we keep the same output format
    # and document shocks as zeros
    if S !== nothing
        zgrid = S.zgrid
        Π = S.Π
        π0 = S.π
    end

    # Care : cpol can be a matrix (a,z) or a vector (a) depending on shocks
    cpol = sol.policy[:c].value

    assets = Matrix{Float64}(undef, N, T)
    cons = Matrix{Float64}(undef, N, T)
    zdraws = Matrix{Float64}(undef, N, T)
    seeds = Vector{UInt64}(undef, N)


    # --- Seed handling ---
    # Master RNG/seed: prefer cfg.random.seed; else derive from canonical cfg
    # The rule is to derive individual agent seeds from the master seed
    # so that the same cfg always leads to the same panel sim
    cfg_id = hash_hex(canonicalize_cfg(cfg))
    master_seed = derive_seed(Int(rand(rng, UInt)), cfg_id)
    master_rng = make_rng(master_seed)

    # Deterministic model: document shocks as zeros in the output
    if S === nothing
        zdraws .= 0.0
    end

    # Fun to sample from a row of a transition matrix Π
    # In the sim, useful bc we need to sample T times per agent
    # Note: Π assumed to be row-stochastic
    @inline function sample_row(Π::AbstractMatrix{<:Real}, i::Int, rng)
        u = rand(rng)
        s = 0.0
        @inbounds for j = 1:axes(Π, 2)
            s += Π[i, j]
            if u <= s
                return j
            end
        end
        return last(axes(Π, 2))
    end

    # simple scalar linear interpolation over agrid
    # basically the same fun as in interp, but inlined and not needing extraction
    @inline function lin1(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, ξ::Real)
        n = length(x)
        if ξ <= x[1]
            return y[1]
        elseif ξ >= x[end]
            return y[end]
        else
            j = searchsortedfirst(x, ξ)
            j = clamp(j, 2, n)
            x0 = x[j-1];
            x1 = x[j]
            y0 = y[j-1];
            y1 = y[j]
            t = (ξ - x0) / (x1 - x0)
            return (1 - t) * y0 + t * y1
        end
    end

    @inbounds for n in axes(assets, 1)
        # independent rng per agent derived from the master seed
        agent_seed = derive_seed(master_seed, n)
        seeds[n] = UInt64(agent_seed % UInt64)
        arng = make_rng(agent_seed)

        # shocks indices and values
        idx = sample_row(reshape(π0, 1, :), 1, arng)  # sample initial state from π0
        for t in axes(assets, 2)
            zdraws[n, t] = zgrid[idx]
            idx = sample_row(Π, idx, arng)
        end
        # Note : the zdraws are now fixed !
        # shocks must be drawn before starting the asset loop


        # a/c consumption path loop
        a_t = g[:a].min
        for t in axes(assets, 2)
            a_prev = a_t
            if cpol isa AbstractMatrix
                # nearest z index, then linear in a
                zval = zdraws[n, t]
                zj = searchsortedfirst(zgrid, zval)
                zj = clamp(zj, firstindex(zgrid), lastindex(zgrid))
                c_t = lin1(agrid, view(cpol, :, zj), a_prev)
                cons[n, t] = c_t
                y_t = p.y * exp(zval)
                a_t = clamp(R * a_prev + y_t - c_t, g[:a].min, g[:a].max)
            else
                # deterministic: linear in a
                c_t = lin1(agrid, cpol, a_prev)
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

    @info "panel moments" mean_log_c_growth final_asset_mean final_asset_std

    diagnostics = (
        rng_kind = string(typeof(master_rng)),
        master_seed = master_seed,
        mean_log_c_growth = mean_log_c_growth,
        final_asset_mean = final_asset_mean,
        final_asset_std = final_asset_std,
    )

    return (; assets, consumption = cons, shocks = zdraws, seeds, diagnostics)
end

end # module
