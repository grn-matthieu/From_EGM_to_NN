module UtilsConfig

import ..API: load_config, validate_config
using YAML

# --- helpers ---
keys_to_symbols(x) =
    x isa Dict ? Dict(Symbol(k) => keys_to_symbols(v) for (k, v) in x) :
    x isa Vector ? [keys_to_symbols(v) for v in x] : x
function load_config(path::AbstractString)
    config = keys_to_symbols(YAML.load_file(path))
    validate_config(config)
    return config
end
_lower(x) = lowercase(string(x))

# internal helpers (local to validate_config)
# _key(d,k): find Symbol/String key; _getnum: fetch numeric with name
# terse error strings by design

function validate_config(cfg::AbstractDict)
    _key(d::AbstractDict, k::Symbol) =
        haskey(d, k) ? k : (haskey(d, String(k)) ? Symbol(String(k)) : nothing)
    function _getnum(d::AbstractDict, k; name = nothing)
        kk = _key(d, k)
        kk === nothing && error("missing $(name===nothing ? k : name)")
        v = d[kk]
        v isa Real || error("$(name===nothing ? k : name) not numeric")
        v
    end

    # top-level
    for (sect, ty) in (
        (:model, AbstractDict),
        (:params, AbstractDict),
        (:grids, AbstractDict),
        (:solver, AbstractDict),
    )
        haskey(cfg, sect) || error("missing $sect")
        cfg[sect] isa ty || error("$sect wrong type")
    end

    # model
    haskey(cfg[:model], :name) || error("missing model.name")

    # params
    p = cfg[:params]
    β_sym = Symbol("β")
    σ_sym = Symbol("σ")
    for k in (β_sym, σ_sym, :r, :y)
        haskey(p, k) || error("missing params.$k")
        p[k] isa Real || error("params.$k not numeric")
    end
    0 < p[β_sym] < 1 || error("β ∈ (0,1) violated")
    p[σ_sym] > 0 || error("σ ≤ 0")
    p[:r] > -1 || error("r ≤ -1")
    p[:y] > 0 || error("y ≤ 0")

    # grids
    g = cfg[:grids]
    for k in (:Na, :a_min, :a_max)
        haskey(g, k) || error("missing grids.$k")
    end
    g[:Na] isa Integer || error("Na not Int")
    g[:Na] > 1 || error("Na ≤ 1")
    g[:a_min] isa Real || error("a_min not Real")
    g[:a_max] isa Real || error("a_max not Real")
    g[:a_max] > g[:a_min] || error("a_max ≤ a_min")

    # utility (optional)
    if haskey(cfg, :utility) &&
       cfg[:utility] isa AbstractDict &&
       haskey(cfg[:utility], :u_type)
        _lower(cfg[:utility][:u_type]) in ("crra",) || error("utility.u_type unsupported")
    end

    # solver
    s = cfg[:solver]
    haskey(s, :method) || error("missing solver.method")
    mth = _lower(s[:method])
    mth in ("egm", "projection", "perturbation", "nn") || error("solver.method invalid")

    if haskey(s, :tol)
        s[:tol] isa Real && s[:tol] > 0 || error("tol > 0 required")
    end
    if haskey(s, :maxit)
        s[:maxit] isa Integer && s[:maxit] ≥ 1 || error("maxit ≥ 1 required")
    end
    if haskey(s, :verbose)
        s[:verbose] isa Bool || error("verbose not Bool")
    end

    # EGM options
    if haskey(s, :interp_kind)
        _lower(s[:interp_kind]) in ("linear", "pchip", "monotone_cubic") ||
            error("interp_kind invalid")
    end
    if haskey(s, :warm_start)
        ws = _lower(s[:warm_start])
        ws in ("default", "half_resources", "none", "steady_state") ||
            error("warm_start invalid")
        if ws == "steady_state"
            haskey(p, :y) || error("need params.y for steady_state")
            haskey(p, :r) || error("need params.r for steady_state")
            haskey(g, :a_min) || error("need grids.a_min for steady_state")
        end
    end

    # Projection
    if mth == "projection"
        if haskey(s, :orders)
            ords = s[:orders]
            ords isa AbstractVector{<:Integer} && !isempty(ords) || error("orders invalid")
            maxord = g[:Na] - 1
            all(o -> 0 ≤ o ≤ maxord, ords) || error("orders out of range")
        end
        if haskey(s, :Nval)
            s[:Nval] isa Integer && s[:Nval] ≥ 2 || error("Nval ≥ 2 required")
        end
    end

    # Perturbation
    if mth == "perturbation"
        if haskey(s, :order)
            s[:order] isa Integer && s[:order] ≥ 1 || error("order ≥ 1 required")
        end
        if haskey(s, :a_bar)
            abar = s[:a_bar]
            (abar === nothing || (abar isa Real && g[:a_min] ≤ abar ≤ g[:a_max])) ||
                error("a_bar out of range")
        end
        if get(s, :order, 1) ≥ 2
            if haskey(s, :h_a) && s[:h_a] !== nothing
                s[:h_a] isa Real && s[:h_a] > 0 || error("h_a > 0 required")
            end
            if haskey(cfg, :shocks) && get(cfg[:shocks], :active, false)
                if haskey(s, :h_z) && s[:h_z] !== nothing
                    s[:h_z] isa Real && s[:h_z] > 0 || error("h_z > 0 required")
                end
            end
        end
        if haskey(s, :tol_fit)
            s[:tol_fit] isa Real && s[:tol_fit] > 0 || error("tol_fit > 0 required")
        end
        if haskey(s, :maxit_fit)
            s[:maxit_fit] isa Integer && s[:maxit_fit] ≥ 1 ||
                error("maxit_fit ≥ 1 required")
        end
    end

    # Shocks (optional)
    if haskey(cfg, :shocks) &&
       cfg[:shocks] isa AbstractDict &&
       get(cfg[:shocks], :active, false)
        sh = cfg[:shocks]
        _lower(get(sh, :method, "tauchen")) in ("tauchen", "rouwenhorst") ||
            error("shocks.method invalid")
        ρk =
            haskey(sh, Symbol("ρ_shock")) ? Symbol("ρ_shock") :
            (haskey(sh, :ρ_shock) ? :ρ_shock : nothing)
        ρk === nothing && error("missing shocks.ρ_shock")
        ρ = sh[ρk]
        ρ isa Real && -1 < ρ < 1 || error("ρ_shock ∉ (-1,1)")
        σk =
            haskey(sh, Symbol("σ_shock")) ? Symbol("σ_shock") :
            (haskey(sh, :s_shock) ? :s_shock : nothing)
        σk === nothing && error("missing shocks.σ_shock")
        s_e = sh[σk]
        s_e isa Real || error("σ_shock not numeric")
        s_e ≥ 0 || error("σ_shock < 0")
        Nz = _getnum(sh, :Nz, name = "shocks.Nz")
        Nz isa Integer || error("Nz not Int")
        (Nz ≥ 2 || s_e == 0) || error("Nz < 2 with σ_shock > 0")
        if get(sh, :method, "tauchen") == "tauchen" && haskey(sh, :m)
            sh[:m] isa Real && sh[:m] > 0 || error("m > 0 required")
        end
        if haskey(sh, :validate)
            sh[:validate] isa Bool || error("validate not Bool")
        end
    end

    # Warm start overrides
    if haskey(cfg, :init) && cfg[:init] isa AbstractDict && haskey(cfg[:init], :c)
        c0 = cfg[:init][:c]
        Na = g[:Na]
        if haskey(cfg, :shocks) && get(cfg[:shocks], :active, false)
            Nz = Int(get(cfg[:shocks], :Nz, 1))
            (
                c0 isa AbstractArray &&
                ndims(c0) == 2 &&
                size(c0, 1) == Na &&
                size(c0, 2) == Nz
            ) || error("init.c size mismatch")
        else
            (c0 isa AbstractVector && length(c0) == Na) || error("init.c length mismatch")
        end
        all(>(0), c0) || error("init.c must be > 0")
    end

    # Random seed
    if haskey(cfg, :random) && cfg[:random] isa AbstractDict && haskey(cfg[:random], :seed)
        try
            _ = UInt64(cfg[:random][:seed])
        catch
            error("random.seed not integer")
        end
    end

    true
end

end # module
