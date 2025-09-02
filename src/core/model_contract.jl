module ModelContract

using ..API
import ..API: get_params, get_grids, get_shocks, get_utility


"""
    get_params(model::AbstractModel)

Contract function to extract model parameters.
"""
function get_params(model::AbstractModel)
    error("get_params not implemented for model $(typeof(model))")
end


"""
    get_grids(model::AbstractModel)

Contract function to extract model grids.
"""
function get_grids(model::AbstractModel)
    error("get_grids not implemented for model $(typeof(model))")
end


"""
    get_shocks(model::AbstractModel)

Contract function to extract model shocks.
"""
function get_shocks(model::AbstractModel)
    error("get_shocks not implemented for model $(typeof(model))")
end


"""
    get_utility(model::AbstractModel)

Contract function to extract model utility.
"""
function get_utility(model::AbstractModel)
    error("get_utility not implemented for model $(typeof(model))")
end


end #module