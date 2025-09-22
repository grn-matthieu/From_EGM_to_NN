module ScriptConfigHelpers

export dict_to_namedtuple,
    namedtuple_to_dict,
    merge_config,
    merge_section,
    set_nested,
    get_nested,
    maybe_namedtuple

"""Recursively convert dictionaries to NamedTuples."""
function dict_to_namedtuple(x)
    if x isa NamedTuple
        return x
    elseif x isa AbstractDict
        keys = Symbol[]
        vals = Any[]
        for (k, v) in x
            push!(keys, Symbol(k))
            push!(vals, dict_to_namedtuple(v))
        end
        return NamedTuple{Tuple(keys)}(Tuple(vals))
    elseif x isa AbstractVector
        return map(dict_to_namedtuple, x)
    elseif x isa Tuple
        return map(dict_to_namedtuple, x)
    else
        return x
    end
end

"""Recursively convert NamedTuples back into `Dict{Symbol,Any}`."""
function namedtuple_to_dict(nt::NamedTuple)
    data = Dict{Symbol,Any}()
    for (k, v) in pairs(nt)
        data[k] = _convert_back(v)
    end
    return data
end

namedtuple_to_dict(x) = x

function _convert_back(x)
    if x isa NamedTuple
        return namedtuple_to_dict(x)
    elseif x isa AbstractVector
        return map(_convert_back, x)
    elseif x isa Tuple
        return map(_convert_back, x)
    else
        return x
    end
end

maybe_namedtuple(x) = x
maybe_namedtuple(x::AbstractDict) = dict_to_namedtuple(x)
maybe_namedtuple(x::NamedTuple) = x

"""Deep merge `overrides` into `base`, returning a new NamedTuple."""
function merge_config(base, overrides)
    base_nt = dict_to_namedtuple(base)
    over_nt = dict_to_namedtuple(overrides)
    return _merge_namedtuple(base_nt, over_nt)
end

function _merge_namedtuple(base::NamedTuple, overrides::NamedTuple)
    data = Dict{Symbol,Any}()
    for (k, v) in pairs(base)
        data[k] = v
    end
    for (k, v) in pairs(overrides)
        if haskey(data, k)
            current = maybe_namedtuple(data[k])
            newval = maybe_namedtuple(v)
            if current isa NamedTuple && newval isa NamedTuple
                data[k] = _merge_namedtuple(current, newval)
            else
                data[k] = newval
            end
        else
            data[k] = maybe_namedtuple(v)
        end
    end
    return dict_to_namedtuple(data)
end

"""Merge overrides into a specific top-level section."""
function merge_section(base, section::Symbol, overrides)
    base_nt = dict_to_namedtuple(base)
    section_nt =
        hasproperty(base_nt, section) ? maybe_namedtuple(getproperty(base_nt, section)) :
        NamedTuple()
    updated_section = merge_config(section_nt, overrides)
    return merge_config(base_nt, NamedTuple{(section,)}((updated_section,)))
end

"""Set a nested field located by `path` to `value`, rebuilding NamedTuples."""
function set_nested(base, path::Tuple{Vararg{Symbol}}, value)
    isempty(path) && return dict_to_namedtuple(base)
    base_nt = dict_to_namedtuple(base)
    head = first(path)
    tail = Base.tail(path)
    if isempty(tail)
        return merge_config(base_nt, NamedTuple{(head,)}((dict_to_namedtuple(value),)))
    else
        subsection =
            hasproperty(base_nt, head) ? maybe_namedtuple(getproperty(base_nt, head)) :
            NamedTuple()
        updated = set_nested(subsection, tail, value)
        return merge_config(base_nt, NamedTuple{(head,)}((updated,)))
    end
end

"""Fetch a nested field by tuple path, returning `default` if missing."""
function get_nested(base, path::Tuple{Vararg{Symbol}}, default = nothing)
    current = base
    for key in path
        current = maybe_namedtuple(current)
        if current isa NamedTuple && hasproperty(current, key)
            current = getproperty(current, key)
        else
            return default
        end
    end
    return current
end

end # module
