module UtilsConfig
export load_config

using YAML

function load_config(path::AbstractString)
    return YAML.load_file(path)
end

end