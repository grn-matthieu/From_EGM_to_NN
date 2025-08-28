module UtilsConfig
export load_config

using YAML

#YAML format to stay organized

function load_config(path::AbstractString)
    @info "Loading configuration from $path"
    return YAML.load_file(path)
end

end