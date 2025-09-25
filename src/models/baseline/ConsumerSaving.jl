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

function build_cs_model(cfg::NamedTuple)
    params = cfg.params
    grids_cfg = cfg.grids

    # Asset grid (for the cs model)
    a_min = grids_cfg.a_min
    a_max = grids_cfg.a_max
    Na = grids_cfg.Na
    agrid = collect(range(a_min, a_max; length = Na))
    grids = (a = (; grid = agrid, min = a_min, max = a_max, N = Na),)

    # Shocks discretization. If a `:shocks` entry exists in the config we
    # consider it active if any shock-related keys are present or if
    # `:active` is explicitly true. This makes tests that patch only
    # `:Nz`/`σ_shock` (without `:active`) behave as authors intended.
    shocks_cfg = maybe(cfg, :shocks)
    function shocks_specified(sc)
        if sc === nothing
            return false
        end
        # If active explicitly true, use shocks. Otherwise check for
        # common shock keys that indicate the user supplied shock specs.
        keys_present = (:Nz, :σ_shock, :s_shock, :ρ_shock, :method, :m)
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

    shocks = shocks_specified(shocks_cfg) ? discretize(shocks_cfg) : nothing

    # Augment params with shock-specific scalars expected by some solvers (NN)
    if shocks !== nothing
        # copy ρ_shock and σ_shock into params as ρ and σ_shocks for backward compatibility
        ρ = maybe(shocks_cfg, :ρ_shock, 0.0)
        σ_shocks = maybe(shocks_cfg, :σ_shock, 0.0)
        params = merge(params, (ρ = ρ, σ_shocks = σ_shocks))
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

    return ConsumerSavingModel(params, grids, shocks, utility)
end

get_params(model::ConsumerSavingModel) = model.params
get_grids(model::ConsumerSavingModel) = model.grids
get_shocks(model::ConsumerSavingModel) = model.shocks
get_utility(model::ConsumerSavingModel) = model.utility

end # module
