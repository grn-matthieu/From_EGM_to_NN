module ConsumerSaving

using ..API: AbstractModel
using ..Shocks: ShockOutput, discretize

import ..API: get_params, get_grids, get_shocks, get_utility


struct ConsumerSavingModel <: AbstractModel
    params::NamedTuple
    grids::Dict{Symbol,Any}
    shocks::Union{Nothing,ShockOutput}
    utility::NamedTuple
end

function build_cs_model(cfg::AbstractDict)
    # Reading parameters
    if haskey(cfg, :params) && haskey(cfg, :grids)
        params = cfg[:params]
        grids = cfg[:grids]
    else
        error("Config must contain the following keys : (:params, :grids)")
    end

    # Asset grid (for the cs model)
    a_min = grids[:a_min]
    a_max = grids[:a_max]
    Na = grids[:Na]
    agrid = collect(range(a_min, a_max; length = Na))
    grids = Dict{Symbol,Any}(:a => (; grid = agrid, min = a_min, max = a_max, N = Na))

    # Shocks discretization (if stoch, modalities in the shocks field)
    shocks = nothing
    if haskey(cfg, :shocks) && get(cfg[:shocks], :active, false)
        shocks = discretize(cfg[:shocks])
    end

    # Utility closure (CRRA)
    σ = params[:σ]
    if isapprox(σ, 1.0; atol = 1e-8) # handle the extreme case (log)
        u = (c -> log.(c))
        u_prime = (c -> 1.0 ./ c)
        u_prime_inv = (up -> 1.0 ./ up)
    else
        u = (c -> (c .^ (1-σ) .- 1.0) ./ (1.0 - σ))
        u_prime = (c -> c .^ (-σ))
        u_prime_inv = (up -> up .^ (-1.0/σ))
    end
    utility = (; u, u_prime, u_prime_inv, σ)

    return ConsumerSavingModel((; params...), grids, shocks, utility)
end

get_params(model::ConsumerSavingModel) = model.params
get_grids(model::ConsumerSavingModel) = model.grids
get_shocks(model::ConsumerSavingModel) = model.shocks
get_utility(model::ConsumerSavingModel) = model.utility

end # module
