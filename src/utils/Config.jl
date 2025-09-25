module UtilsConfig

import ..API: load_config, validate_config
using YAML

export maybe

# --- helpers ---
yaml_to_namedtuple(x) = x
function yaml_to_namedtuple(x::AbstractDict)
    pairs = (Symbol(k) => yaml_to_namedtuple(v) for (k, v) in x)
    return (; pairs...)
end
yaml_to_namedtuple(x::AbstractVector) = [yaml_to_namedtuple(v) for v in x]

function load_config(path::AbstractString)
    config = yaml_to_namedtuple(YAML.load_file(path))
    validate_config(config)
    return config
end
_lower(x) = lowercase(string(x))

# internal helpers (local to validate_config)
# _getprop: safe property lookup with default; _getnum: fetch numeric with name
# terse error strings by design

function validate_config(cfg::AbstractDict)
    # Normalize Dict-like configs to NamedTuple and delegate to the
    # NamedTuple-specific validator. This keeps `load_config` as the
    # canonical loader while allowing tests/helpers to call validate_config
    # with plain Dict objects.
    return validate_config(yaml_to_namedtuple(Dict(cfg)))
end

function validate_config(cfg::NamedTuple)
    _getprop(::Any, ::Symbol, default) = default
    _getprop(d::NamedTuple, k::Symbol, default) =
        hasproperty(d, k) ? getproperty(d, k) : default
    function _getnum(d::NamedTuple, k::Symbol; name = nothing)
        hasproperty(d, k) || error("missing $(name === nothing ? k : name)")
        v = getproperty(d, k)
        v isa Real || error("$(name === nothing ? k : name) not numeric")
        return v
    end

    # top-level
    for sect in (:model, :params, :grids, :solver)
        hasproperty(cfg, sect) || error("missing $sect")
        getproperty(cfg, sect) isa NamedTuple || error("$sect wrong type")
    end

    # model
    hasproperty(cfg.model, :name) || error("missing model.name")

    # params
    p = cfg.params
    β_sym = Symbol("β")
    σ_sym = Symbol("σ")
    # Require fundamental parameters β and σ. Allow r and y to be optional
    # because some minimal configs (e.g. used in determinism tests) omit them.
    for k in (β_sym, σ_sym)
        hasproperty(p, k) || error("missing params.$k")
        getproperty(p, k) isa Real || error("params.$k not numeric")
    end
    0 < getproperty(p, β_sym) < 1 || error("params.β out of range")
    getproperty(p, σ_sym) > 0 || error("σ ≤ 0")
    # Validate r and y only when provided
    if hasproperty(p, :r)
        getproperty(p, :r) isa Real || error("params.r not numeric")
        getproperty(p, :r) > -1 || error("r ≤ -1")
    end
    if hasproperty(p, :y)
        getproperty(p, :y) isa Real || error("params.y not numeric")
        getproperty(p, :y) > 0 || error("y ≤ 0")
    end

    # grids
    g = cfg.grids
    for k in (:Na, :a_min, :a_max)
        hasproperty(g, k) || error("missing grids.$k")
    end
    g.Na isa Integer || error("grids.Na not Int")
    g.Na > 1 || error("grids.Na out of range")
    g.a_min isa Real || error("a_min not Real")
    g.a_max isa Real || error("a_max not Real")
    g.a_max > g.a_min || error("a_max ≤ a_min")

    # utility (optional)
    if hasproperty(cfg, :utility)
        util = cfg.utility
        if util isa NamedTuple && hasproperty(util, :u_type)
            _lower(getproperty(util, :u_type)) in ("crra",) ||
                error("utility.u_type unsupported")
        end
    end

    # solver
    s = cfg.solver
    hasproperty(s, :method) || error("missing solver.method")
    mth = _lower(getproperty(s, :method))
    # Accept canonical method names, shorthand for TI, and the special "all" token
    mth in ("egm", "projection", "perturbation", "nn", "timeiteration", "ti", "all") ||
        error("solver.method invalid")

    if hasproperty(s, :tol)
        s.tol isa Real && s.tol > 0 || error("tol > 0 required")
    end
    if hasproperty(s, :maxit)
        s.maxit isa Integer && s.maxit ≥ 1 || error("maxit ≥ 1 required")
    end
    if hasproperty(s, :verbose)
        s.verbose isa Bool || error("verbose not Bool")
    end

    # EGM options
    if hasproperty(s, :interp_kind)
        _lower(getproperty(s, :interp_kind)) in ("linear", "pchip", "monotone_cubic") ||
            error("interp_kind invalid")
    end
    if hasproperty(s, :warm_start)
        ws = _lower(getproperty(s, :warm_start))
        ws in ("default", "half_resources", "none", "steady_state") ||
            error("warm_start invalid")
        if ws == "steady_state"
            hasproperty(p, :y) || error("need params.y for steady_state")
            hasproperty(p, :r) || error("need params.r for steady_state")
            hasproperty(g, :a_min) || error("need grids.a_min for steady_state")
        end
    end

    # Projection
    if mth == "projection"
        if hasproperty(s, :orders)
            ords = getproperty(s, :orders)
            ords isa AbstractVector{<:Integer} && !isempty(ords) || error("orders invalid")
            maxord = g.Na - 1
            all(o -> 0 ≤ o ≤ maxord, ords) || error("orders out of range")
        end
        if hasproperty(s, :Nval)
            s.Nval isa Integer && s.Nval ≥ 2 || error("Nval ≥ 2 required")
        end
    end

    # Perturbation
    if mth == "perturbation"
        if hasproperty(s, :order)
            s.order isa Integer && s.order ≥ 1 || error("order ≥ 1 required")
        end
        if hasproperty(s, :a_bar)
            abar = getproperty(s, :a_bar)
            (abar === nothing || (abar isa Real && g.a_min ≤ abar ≤ g.a_max)) ||
                error("a_bar out of range")
        end
        if _getprop(s, :order, 1) ≥ 2
            if hasproperty(s, :h_a) && s.h_a !== nothing
                s.h_a isa Real && s.h_a > 0 || error("h_a > 0 required")
            end
            if hasproperty(cfg, :shocks) && _getprop(cfg.shocks, :active, false)
                if hasproperty(s, :h_z) && s.h_z !== nothing
                    s.h_z isa Real && s.h_z > 0 || error("h_z > 0 required")
                end
            end
        end
        if hasproperty(s, :tol_fit)
            s.tol_fit isa Real && s.tol_fit > 0 || error("tol_fit > 0 required")
        end
        if hasproperty(s, :maxit_fit)
            s.maxit_fit isa Integer && s.maxit_fit ≥ 1 || error("maxit_fit ≥ 1 required")
        end
    end

    # Shocks (optional)
    if hasproperty(cfg, :shocks)
        sh_raw = cfg.shocks
        sh = sh_raw isa AbstractDict ? yaml_to_namedtuple(Dict(sh_raw)) : sh_raw
        if sh isa NamedTuple && _getprop(sh, :active, false)
            method_lower = _lower(_getprop(sh, :method, "tauchen"))
            method_lower in ("tauchen", "rouwenhorst") || error("shocks.method invalid")
            ρsym = Symbol("ρ_shock")
            ρkey = hasproperty(sh, ρsym) ? ρsym : nothing
            ρkey === nothing && error("missing shocks.ρ_shock")
            ρ = getproperty(sh, ρkey)
            ρ isa Real && -1 < ρ < 1 || error("shocks.rho out of range")
            σsym = Symbol("σ_shock")
            σkey =
                hasproperty(sh, σsym) ? σsym :
                (hasproperty(sh, :s_shock) ? :s_shock : nothing)
            σkey === nothing && error("missing shocks.σ_shock")
            s_e = getproperty(sh, σkey)
            s_e isa Real || error("σ_shock not numeric")
            s_e ≥ 0 || error("σ_shock < 0")
            Nz = _getnum(sh, :Nz, name = "shocks.Nz")
            Nz isa Integer || error("Nz not Int")
            (Nz ≥ 2 || s_e == 0) || error("Nz < 2 with σ_shock > 0")
            if method_lower == "tauchen" && hasproperty(sh, :m)
                sh.m isa Real && sh.m > 0 || error("m > 0 required")
            end
            if hasproperty(sh, :validate)
                sh.validate isa Bool || error("validate not Bool")
            end
        end
    end

    # Warm start overrides
    if hasproperty(cfg, :init)
        init_raw = cfg.init
        initcfg = init_raw isa AbstractDict ? yaml_to_namedtuple(Dict(init_raw)) : init_raw
        if initcfg isa NamedTuple && hasproperty(initcfg, :c)
            c0 = initcfg.c
            Na = g.Na
            if hasproperty(cfg, :shocks) && _getprop(cfg.shocks, :active, false)
                Nz = Int(_getprop(cfg.shocks, :Nz, 1))
                (
                    c0 isa AbstractArray &&
                    ndims(c0) == 2 &&
                    size(c0, 1) == Na &&
                    size(c0, 2) == Nz
                ) || error("init.c size mismatch")
            else
                (c0 isa AbstractVector && length(c0) == Na) ||
                    error("init.c length mismatch")
            end
            all(x -> x > 0, c0) || error("init.c must be > 0")
        end
    end

    # Random seed
    if hasproperty(cfg, :random)
        r_raw = cfg.random
        rcfg = r_raw isa AbstractDict ? yaml_to_namedtuple(Dict(r_raw)) : r_raw
        if rcfg isa NamedTuple && hasproperty(rcfg, :seed)
            try
                _ = UInt64(rcfg.seed)
            catch
                error("random.seed not integer")
            end
        end
    end

    true
end

maybe(x; default = nothing) = x === nothing ? default : x
maybe(cfg::NamedTuple, key::Symbol; default = nothing) =
    hasproperty(cfg, key) ? getproperty(cfg, key) : default
maybe(cfg, ::Symbol; default = nothing) = default

# Support calling with a positional default argument (common call pattern
# throughout the repo). These overloads avoid MethodError when the third
# argument is a concrete default value (e.g. a number) instead of a Symbol.
maybe(::Nothing, ::Vararg{Symbol}; default = nothing) = default

# Unified variadic `maybe` which supports two common call patterns used
# throughout the repo:
# 1) maybe(cfg, :a, :b)          -> nested lookup: cfg.a.b if present
# 2) maybe(cfg, :a, default_val) -> positional default when the third arg is a value
#
# The function heuristically distinguishes these at runtime: if a single
# trailing Symbol is provided and the value obtained at the first key is *not*
# a NamedTuple with that symbol as a property, then we treat the trailing
# Symbol as a positional default. Otherwise we perform nested traversal.
function maybe(cfg, key::Symbol, rest::Any...; default = nothing)
    # Single-key access: delegate to keyword-default form
    if isempty(rest)
        return maybe(cfg, key; default = default)
    end

    # If cfg is nothing, positional-default calls like `maybe(nothing, :k, false)`
    # should simply return the provided default positional value.
    if cfg === nothing
        first_rest = rest[1]
        return first_rest === nothing ? default : first_rest
    end

    # Fetch the first-level value at `key` (using keyword default)
    val = maybe(cfg, key; default = default)

    # If there's exactly one trailing arg, prefer treating it as a positional
    # default value (e.g. `maybe(s, :k, 3.0)` or `maybe(s, :k, false)`), even
    # when it's a Symbol. Only perform nested lookup when the fetched value at
    # `key` is a NamedTuple that actually contains that Symbol property.
    if length(rest) == 1
        if !(rest[1] isa Symbol)
            return maybe(cfg, key; default = rest[1])
        else
            # trailing Symbol: use as positional default unless val is a NamedTuple
            # exposing that property (nested lookup requested)
            if !(val isa NamedTuple && hasproperty(val, rest[1]))
                return maybe(cfg, key; default = rest[1])
            end
        end
    end

    # If all trailing args are Symbols, treat as nested lookup: maybe(cfg, :a, :b, :c)
    if all(x -> x isa Symbol, rest)
        return maybe(val, Tuple(rest)...; default = default)
    end

    # Mixed case: last element may be a positional default, preceding ones are keys
    if rest[end] !== nothing && !(rest[end] isa Symbol)
        # split keys vs default
        keys = rest[1:end-1]
        if all(x -> x isa Symbol, keys)
            return maybe(val, Tuple(keys)...; default = rest[end])
        end
    end

    # Fallback: attempt nested lookup where possible, otherwise return keyword default
    return maybe(val, (x for x in rest if x isa Symbol)...; default = default)
end

end # module
