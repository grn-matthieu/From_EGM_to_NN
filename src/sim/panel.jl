module SimPanel

export simulate_panel

using Random
using Statistics
using ..Determinism: make_rng, derive_seed

using ..API:
    Solution, AbstractModel, AbstractMethod, get_params, get_grids, get_shocks, solve


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

    zgrid = Float64[0.0]
    Π = ones(Float64, 1, 1)
    π0 = Float64[]                 # empty, unused in deterministic case

    # Care : cpol can be a matrix (a,z) or a vector (a) depending on shocks
    cpol = sol.policy[:c].value

    assets = Matrix{Float64}(undef, N, T)
    cons = Matrix{Float64}(undef, N, T)
    zdraws = Matrix{Float64}(undef, N, T)
    seeds = Vector{UInt64}(undef, N)


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

    if S !== nothing
        zgrid = Float64.(S.zgrid)
        Π = Float64.(S.Π)
        π0 = Float64.(S.π)

        @inline function sample_row(
            Π::AbstractMatrix{<:Real},
            i::Int,
            rng::AbstractRNG,
        )::Int
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

        @inline function sample_vec(π::AbstractVector{<:Real}, rng::AbstractRNG)::Int
            u = rand(rng)
            s = 0.0
            @inbounds for j in eachindex(π)
                s += π[j]
                if u <= s
                    return j
                end
            end
            return lastindex(π)
        end
    end
    @inline function lin1(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, ξ::Real)
        n = length(x)
        if ξ <= x[1]
            return y[1]
        elseif ξ >= x[end]
            return y[end]
        else
            j = searchsortedfirst(x, ξ)
            j = clamp(j, 2, n)
            x0 = x[j-1]
            x1 = x[j]
            y0 = y[j-1]
            y1 = y[j]
            t = (ξ - x0) / (x1 - x0)
            return (1 - t) * y0 + t * y1
        end
    end
    @inbounds for n in axes(assets, 1)
        # independent rng per agent derived from the master rng
        agent_seed = derive_seed(master_rng, n)
        seeds[n] = agent_seed
        arng = make_rng(agent_seed)

        if S !== nothing
            idx = sample_vec(π0, arng)  # was: sample_row(reshape(π0,1,:), 1, arng)
            for t in axes(assets, 2)
                zdraws[n, t] = zgrid[idx]
                idx = sample_row(Π, idx, arng)
            end
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
