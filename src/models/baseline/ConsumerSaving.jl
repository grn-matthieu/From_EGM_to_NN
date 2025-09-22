"""
ConsumerSaving

Baseline consumption–savings model with CRRA utility and borrowing constraint.
Defines parameters, grids, and shock processes used by solver kernels.
"""
module ConsumerSaving

using ..API: AbstractModel
using ..Shocks: ShockOutput, discretize

import ..API: get_params, get_grids, get_shocks, get_utility


_to_namedtuple(x::NamedTuple) = x
_to_namedtuple(x::AbstractDict) =
    (; (Symbol(k) => _to_namedtuple(v) for (k, v) in pairs(x))...)
_to_namedtuple(x::AbstractVector) = _to_namedtuple.(x)
_to_namedtuple(x) = x


struct ConsumerSavingModel <: AbstractModel
    params::NamedTuple
    grids::NamedTuple
    shocks::Union{Nothing,ShockOutput}
    utility::NamedTuple
end

function build_cs_model(cfg::NamedTuple)
    hasproperty(cfg, :params) || error("configuration must contain a `params` section")
    hasproperty(cfg, :grids) || error("configuration must contain a `grids` section")

    params = _to_namedtuple(cfg.params)
    grids_cfg = _to_namedtuple(cfg.grids)

    # Asset grid (for the cs model)
    a_min = grids_cfg.a_min
    a_max = grids_cfg.a_max
    Na = grids_cfg.Na
    agrid = collect(range(a_min, a_max; length = Na))
    grids = (a = (; grid = agrid, min = a_min, max = a_max, N = Na),)

    # Shocks discretization (if stoch, modalities in the shocks field)
    shocks = nothing
    shocks_cfg = get(cfg, :shocks, nothing)
    if shocks_cfg !== nothing
        shocks_nt = _to_namedtuple(shocks_cfg)
        if get(shocks_nt, :active, false)
            shocks = discretize(shocks_nt)
        end
    end

    # Utility closure (CRRA)
    σ = params.σ
    if isapprox(σ, 1.0; atol = 1e-8) # handle the extreme case (log)
        u = (c -> log.(c))
        u_prime = (c -> 1.0 ./ c)
        u_prime_inv = (up -> 1.0 ./ up)
    else
        u = (c -> (c .^ (1 - σ) .- 1.0) ./ (1.0 - σ))
        u_prime = (c -> c .^ (-σ))
        u_prime_inv = (up -> up .^ (-1.0 / σ))
    end
    utility = (; u, u_prime, u_prime_inv, σ)

    return ConsumerSavingModel((; params...), grids, shocks, utility)
end

function build_cs_model(cfg::AbstractDict)
    return build_cs_model(_to_namedtuple(cfg))
end

get_params(model::ConsumerSavingModel) = model.params
get_grids(model::ConsumerSavingModel) = model.grids
get_shocks(model::ConsumerSavingModel) = model.shocks
get_utility(model::ConsumerSavingModel) = model.utility

end # module
