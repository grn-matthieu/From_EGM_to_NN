module UtilsLogging
export new_experiment_id, mklogdir, info_log, warn_log

using UUIDs, Dates, Logging

new_experiment_id() = string(uuid4())

function mklogdir(base::AbstractString, expid::AbstractString)
    dir = joinpath(base, "$(Dates.format(now(), "yyyymmdd_HHMMSS"))_$expid")
    isdir(dir) || mkpath(dir)
    return dir
end

# thin wrappers (uses global logger by default)
info_log(msg) = @info msg
warn_log(msg) = @warn msg

end