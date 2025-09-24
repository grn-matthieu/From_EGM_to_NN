const _ConfigLike = Union{AbstractDict,NamedTuple}

cfg_has(cfg::_ConfigLike, key::Symbol) =
    cfg isa NamedTuple ? hasproperty(cfg, key) : haskey(cfg, key)

function cfg_has(cfg::_ConfigLike, key::Symbol, rest::Symbol...)
    cfg_has(cfg, key) || return false
    return cfg_has(cfg_get(cfg, key), rest...)
end

cfg_get(cfg::_ConfigLike, key::Symbol) = cfg isa NamedTuple ? getfield(cfg, key) : cfg[key]

function cfg_get(cfg::_ConfigLike, key::Symbol, rest::Symbol...)
    return cfg_get(cfg_get(cfg, key), rest...)
end

# --- fixed getdefault ---

# base case: single key
function cfg_getdefault(cfg::_ConfigLike, default, key::Symbol)
    if cfg isa NamedTuple
        return hasproperty(cfg, key) ? getfield(cfg, key) : default
    else
        return get(cfg, key, default)
    end
end

# recursive case: multiple keys
function cfg_getdefault(cfg::_ConfigLike, default, key::Symbol, rest::Symbol...)
    if cfg_has(cfg, key)
        return cfg_getdefault(cfg_get(cfg, key), default, rest...)
    else
        return default
    end
end

# ---------------------------------------

function _cfg_set(cfg::_ConfigLike, key::Symbol, value)
    if cfg isa NamedTuple
        return merge(cfg, NamedTuple{(key,)}((value,)))
    else
        new_cfg = deepcopy(cfg)
        new_cfg[key] = value
        return new_cfg
    end
end

function _cfg_set(cfg::_ConfigLike, path::Tuple{Vararg{Symbol}}, value)
    if length(path) == 1
        return _cfg_set(cfg, path[1], value)
    else
        head = path[1]
        tail = Base.tail(path)
        # If the intermediate key is missing, create an empty container of the
        # same kind (NamedTuple => NamedTuple(), Dict => Dict()) so we can
        # recursively set deep values. This allows tests to patch nested
        # configuration paths that don't exist yet (e.g. (:init, :c)).
        if cfg isa NamedTuple
            child = hasproperty(cfg, head) ? cfg_get(cfg, head) : NamedTuple()
        else
            child = haskey(cfg, head) ? cfg_get(cfg, head) : Dict()
        end
        updated = _cfg_set(child, tail, value)
        return _cfg_set(cfg, head, updated)
    end
end

function cfg_patch(cfg::_ConfigLike, updates::Pair...)
    for (path, value) in updates
        if path isa Symbol
            cfg = _cfg_set(cfg, path, value)
        elseif path isa Tuple{Vararg{Symbol}}
            cfg = _cfg_set(cfg, path, value)
        else
            error("Unsupported config path type $(typeof(path))")
        end
    end
    return cfg
end

function _cfg_without(cfg::NamedTuple, key::Symbol)
    keep = Tuple(k for k in keys(cfg) if k != key)
    values = tuple((getfield(cfg, k) for k in keep)...)
    return NamedTuple{keep}(values)
end

function _cfg_without(cfg::AbstractDict, key::Symbol)
    new_cfg = deepcopy(cfg)
    delete!(new_cfg, key)
    return new_cfg
end

function _cfg_without(cfg::_ConfigLike, path::Tuple{Vararg{Symbol}})
    if length(path) == 1
        return _cfg_without(cfg, path[1])
    else
        head = path[1]
        tail = Base.tail(path)
        child = cfg_get(cfg, head)
        updated = _cfg_without(child, tail)
        return _cfg_set(cfg, head, updated)
    end
end

cfg_without(cfg::_ConfigLike, key::Symbol) = _cfg_without(cfg, key)
cfg_without(cfg::_ConfigLike, path::Tuple{Vararg{Symbol}}) = _cfg_without(cfg, path)
cfg_without(cfg::_ConfigLike, keys::Symbol...) = _cfg_without(cfg, keys)

function is_nondec(x::AbstractVector; tol = 1e-8)
    n = length(x)
    @inbounds for i = 1:(n-1)
        if x[i+1] < x[i] - tol
            return false
        end
    end
    return true
end

function is_nondec(x::AbstractMatrix; tol = 1e-8)
    nrow, ncol = size(x)
    @inbounds for j = 1:ncol
        for i = 1:(nrow-1)
            if x[i+1, j] < x[i, j] - tol
                return false
            end
        end
    end
    return true
end

Base.zero(x::NamedTuple) = map(zero, x)
