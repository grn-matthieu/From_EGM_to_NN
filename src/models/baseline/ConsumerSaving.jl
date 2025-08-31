module ConsumerSaving

using ..API
using ..ModelContract
using ..Shocks

import ..ModelContract: get_params, get_grids, get_shocks, get_utility

# Define your model type
struct ConsumerSavingModel <: AbstractModel
    params::NamedTuple
    grids::NamedTuple
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

    # 2. Build asset grid
    a_min = grids[:a_min]
    a_max = grids[:a_max]
    Na    = grids[:Na]
    agrid = collect(range(a_min, a_max; length=Na))

    # 3. Handle shocks
    shocks = nothing
    if haskey(cfg, :shocks) && get(cfg[:shocks], :active, false)
        shocks = discretize(cfg[:shocks])
    end


    # 4. Build utility closure
    σ = params[:σ]
    u = σ ≈ 1 ? (c -> log(c)) : (c -> (c^(1-σ) - 1) / (1-σ))
    utility = (; u, σ)

    # 6. Return model
    return ConsumerSavingModel(
        (;params...),
        (; a=agrid),
        shocks,
        utility
    )
end

# Implement contract functions for the model type
get_params(model::ConsumerSavingModel) = model.params
get_grids(model::ConsumerSavingModel) = model.grids
get_shocks(model::ConsumerSavingModel) = model.shocks
get_constraints(model::ConsumerSavingModel) = model.constraints
get_utility(model::ConsumerSavingModel) = model.utility

end # module