module ConsumerSaving

using ..API: AbstractModel
using ..Shocks

import ..API: get_params, get_grids, get_shocks, get_utility


# Define your model type
struct ConsumerSavingModel <: AbstractModel
    params::NamedTuple
    grids::Dict{Symbol, Any}
    shocks::Union{Nothing,ShockOutput}
    utility::NamedTuple
end

# Implement the builder
function build_cs_model(cfg::AbstractDict)
    # 1. Read params and grids from either :params/:grids or :model/:grid
    if haskey(cfg, :params) && haskey(cfg, :grids)
        params = cfg[:params]
        grids = cfg[:grids]
    elseif haskey(cfg, :model) && haskey(cfg, :grid)
        params = cfg[:model]
        grids = cfg[:grid]
    else
        error("Config must contain either (:params, :grids) or (:model, :grid)")
    end
    params[Symbol(:γ)] = params[:β] * (1 + params[:r])  # Add γ to params

    # 2. Build asset grid
    a_min = grids[:a_min]
    a_max = grids[:a_max]
    Na    = grids[:Na]
    agrid = collect(range(a_min, a_max; length=Na))

    # Dict to prepare multiple control variables
    grids = Dict{Symbol, Any}(:a => (; grid=agrid, min=a_min, max=a_max, N=Na))

    # 3. Handle shocks
    shocks = nothing
    if haskey(cfg, :shocks) && get(cfg[:shocks], :active, false)
        shocks = discretize(cfg[:shocks])
    end


    # 4. Build utility closure
    σ = params[:σ]
    # utility and marginal utility for CRRA (log limit at σ ≈ 1)
    if isapprox(σ, 1.0; atol=1e-8)
        u = (c -> log(c))
        u_prime = (c -> 1.0 ./ c)
        u_prime_inv = (up -> 1.0 ./ up)
    else
        u = (c -> (c.^(1-σ) .- 1.0) ./ (1.0 - σ))
        u_prime = (c -> c .^ (-σ))
        u_prime_inv = (up -> up .^ (-1.0/σ))
    end
    utility = (; u, u_prime, u_prime_inv, σ)

    # 6. Return model
    return ConsumerSavingModel(
        (;params...),
        grids,
        shocks,
        utility
    )
end

# Implement contract functions for the model type
get_params(model::ConsumerSavingModel) = model.params
get_grids(model::ConsumerSavingModel) = model.grids
get_shocks(model::ConsumerSavingModel) = model.shocks
get_utility(model::ConsumerSavingModel) = model.utility

end # module