module UtilsConfig

import ..API: load_config, validate_config

using YAML


function keys_to_symbols(x)
    if x isa Dict
        return Dict(Symbol(k) => keys_to_symbols(v) for (k, v) in x)
    elseif x isa Vector
        return [keys_to_symbols(v) for v in x]
    else
        return x
    end
end

"""
    load_config(path::AbstractString) -> Dict

Loads a YAML config file and returns a Dict with all keys recursively converted to symbols.
"""
function load_config(path::AbstractString)
    config = YAML.load_file(path)
    return keys_to_symbols(config)
end

"""
    validate_config(cfg::AbstractDict) -> Bool

Validate presence and basic sanity of required configuration keys.
Output : throws an error with its cause if invalid; returns true otherwise.
"""
function validate_config(cfg::AbstractDict)
    # Top-level keys
    missing = Symbol[]
    for k in (:model, :params, :grids, :solver)
        haskey(cfg, k) || push!(missing, k)
    end
    isempty(missing) || error("Missing top-level keys: $(missing)")

    # Model name
    haskey(cfg[:model], :name) || error(":model.name missing")

    # Grids sanity
    g = cfg[:grids]
    for k in (:Na, :a_min, :a_max)
        haskey(g, k) || error("grids.$k missing")
    end
    (g[:Na] isa Integer) || error("grids.Na must be Integer")
    g[:Na] > 1 || error("grids.Na must be > 1")
    g[:a_max] > g[:a_min] || error("grids.a_max must be > a_min")

    # Solver
    s = cfg[:solver]
    haskey(s, :method) || error("solver.method missing")

    return true
end

end
