module UtilsConfig

import ..API: load_config

using YAML


"""
    load_config(path::AbstractString) -> Dict

Loads a YAML config file and returns a Dict with all keys recursively converted to symbols.
"""
function keys_to_symbols(x)
    if x isa Dict
        return Dict(Symbol(k) => keys_to_symbols(v) for (k, v) in x)
    elseif x isa Vector
        return [keys_to_symbols(v) for v in x]
    else
        return x
    end
end

function load_config(path::AbstractString)
    @info "Loading configuration from $path"
    config = YAML.load_file(path)
    return keys_to_symbols(config)
end

end