"""
ConsumerSaving

Baseline consumption–savings model with CRRA utility and borrowing constraint.
Defines parameters, grids, and shock processes used by solver kernels.
"""
module ConsumerSaving

using ..API: AbstractModel
using ..Shocks: ShockOutput, discretize
using ..UtilsConfig: maybe

import ..API: get_params, get_grids, get_shocks, get_utility


struct ConsumerSavingModel <: AbstractModel
    params::NamedTuple
    grids::NamedTuple
    shocks::Union{Nothing,ShockOutput}
    utility::NamedTuple
end

@inline function _build_asset_grid(grids_cfg::NamedTuple)
    a_min = grids_cfg.a_min
    a_max = grids_cfg.a_max
    Na = grids_cfg.Na
    agrid = collect(range(a_min, a_max; length = Na))
    return (a = (; grid = agrid, min = a_min, max = a_max, N = Na),)
end

function _build_crra_utility(params::NamedTuple)
    σ = params.σ
    if isapprox(σ, 1.0; atol = 1e-8)
        u = (c -> log.(c))
        u_prime = (c -> 1.0 ./ c)
        u_prime_inv = (up -> 1.0 ./ up)
    else
        u = (c -> (c .^ (1 - σ) .- 1.0) ./ (1.0 - σ))
        u_prime = (c -> c .^ (-σ))
        u_prime_inv = (up -> up .^ (-1.0 / σ))
    end
    return (; u, u_prime, u_prime_inv, σ)
end

function _maybe_discretize_shocks(cfg::NamedTuple)
    shocks_cfg = maybe(cfg, :shocks)
    function shocks_specified(sc)
        if sc === nothing
            return false
        end
        keys_present = (:Nz, :σ_shock, :ρ_shock, :method, :m)
        if maybe(sc, :active, false)
            return true
        end
        for k in keys_present
            if isa(sc, NamedTuple) ? hasproperty(sc, k) : haskey(sc, k)
                return true
            end
        end
        return false
    end
    return shocks_specified(shocks_cfg) ? discretize(shocks_cfg) : nothing, shocks_cfg
end

function build_cs_model(cfg::NamedTuple)
    params = cfg.params
    grids = _build_asset_grid(cfg.grids)

    shocks, shocks_cfg = _maybe_discretize_shocks(cfg)

    if shocks !== nothing
        ρ_shock = maybe(shocks_cfg, :ρ_shock, 0.0)
        σ_shock = maybe(shocks_cfg, :σ_shock, 0.0)
        params = merge(params, (ρ_shock = ρ_shock, σ_shock = σ_shock))
    end

    utility = _build_crra_utility(params)

    return ConsumerSavingModel(params, grids, shocks, utility)
end

get_params(model::ConsumerSavingModel) = model.params
get_grids(model::ConsumerSavingModel) = model.grids
get_shocks(model::ConsumerSavingModel) = model.shocks
get_utility(model::ConsumerSavingModel) = model.utility

end # module
