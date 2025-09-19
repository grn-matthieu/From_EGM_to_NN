"""

UtilsConfig



Configuration loading and validation helpers. Normalizes user inputs and

provides defaults for solvers and models.

"""

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



Enhanced validation of configuration dictionaries.

Performs normalization of a few benign inputs (e.g., symbol/string coercions) and

asserts required keys, types, and ranges. Throws on first failure; returns true otherwise.

"""

function validate_config(cfg::AbstractDict)

    # --- Small helpers ---

    _sym(x) = x isa Symbol ? x : Symbol(x)

    _lower(x) = lowercase(string(x))

    function _key(d::AbstractDict, k::Symbol)

        haskey(d, k) && return k

        # accept string keys that were not symbolized

        if haskey(d, String(k))

            return Symbol(String(k))

        end

        return nothing

    end

    function _getnum(d::AbstractDict, k; name = nothing)

        kk = _key(d, k)

        kk === nothing && error("Missing key: $(name === nothing ? String(k) : name)")

        v = d[kk]

        (v isa Real) || error(
            "$(name === nothing ? String(k) : name) must be a number; got $(typeof(v))",
        )

        return v

    end



    # --- Top-level structure ---

    for (sect, ty) in (
        (:model, AbstractDict),
        (:params, AbstractDict),
        (:grids, AbstractDict),
        (:solver, AbstractDict),
    )

        haskey(cfg, sect) || error("Missing top-level key: $(sect)")

        cfg[sect] isa ty || error("Top-level $(sect) must be a $(ty)")

    end



    # --- Model ---

    haskey(cfg[:model], :name) || error(":model.name missing")



    # --- Params (ConsumerSaving baseline) ---

    params = cfg[:params]

    # Parameter keys as used throughout kernels

    β_sym = Symbol("β")

    σ_sym = Symbol("σ")

    for req in (β_sym, σ_sym, :r, :y)

        haskey(params, req) || error("params.$(req) missing")

        (params[req] isa Real) ||
            error("params.$(req) must be a number; got $(typeof(params[req]))")

    end

    (0 < params[β_sym] < 1) ||
        error("params.β must satisfy 0 < β < 1; got $(params[β_sym])")

    # σ > 0, allow ~1 for log utility

    (params[σ_sym] > 0) || error("params.σ must be > 0; got $(params[σ_sym])")

    (params[:r] > -1) || error("params.r must be > -1; got $(params[:r])")

    (params[:y] > 0) || error("params.y must be > 0; got $(params[:y])")



    # --- Grids ---

    g = cfg[:grids]

    for k in (:Na, :a_min, :a_max)

        haskey(g, k) || error("grids.$k missing")

    end

    (g[:Na] isa Integer) || error("grids.Na must be Integer; got $(typeof(g[:Na]))")

    g[:Na] > 1 || error("grids.Na must be > 1; got $(g[:Na])")

    (g[:a_min] isa Real) || error("grids.a_min must be Real; got $(typeof(g[:a_min]))")

    (g[:a_max] isa Real) || error("grids.a_max must be Real; got $(typeof(g[:a_max]))")

    g[:a_max] > g[:a_min] ||
        error("grids.a_max must be > a_min; got a_min=$(g[:a_min]) a_max=$(g[:a_max])")



    # --- Utility (optional) ---

    if haskey(cfg, :utility) &&
       cfg[:utility] isa AbstractDict &&
       haskey(cfg[:utility], :u_type)

        ut = _lower(cfg[:utility][:u_type])

        ut in ("crra",) ||
            error("utility.u_type unsupported: $(cfg[:utility][:u_type]); supported: CRRA")

    end



    # --- Solver & Method ---

    s = cfg[:solver]

    haskey(s, :method) || error("solver.method missing")

    mth = _lower(s[:method])

    mth in ("egm", "projection", "perturbation", "nn") || error(
        "solver.method must be one of: EGM, Projection, Perturbation, NN; got $(s[:method])",
    )



    # Common numeric options sanity if present

    if haskey(s, :tol)

        (s[:tol] isa Real && s[:tol] > 0) ||
            error("solver.tol must be Real > 0; got $(s[:tol])")

    end

    if haskey(s, :maxit)

        (s[:maxit] isa Integer && s[:maxit] >= 1) ||
            error("solver.maxit must be Integer >= 1; got $(s[:maxit])")

    end

    if haskey(s, :verbose)

        (s[:verbose] isa Bool) ||
            error("solver.verbose must be Bool; got $(typeof(s[:verbose]))")

    end



    # EGM-specific options

    if haskey(s, :interp_kind)

        ik = _lower(s[:interp_kind])

        ik in ("linear", "pchip", "monotone_cubic") || error(
            "solver.interp_kind must be one of: linear, pchip, monotone_cubic; got $(s[:interp_kind])",
        )

    end

    if haskey(s, :warm_start)

        ws = _lower(s[:warm_start])

        ws in ("default", "half_resources", "none", "steady_state") ||
            error("solver.warm_start invalid: $(s[:warm_start])")

        if ws == "steady_state"

            # ensure presence of required params used to construct steady-state init

            for req in (:y, :r)

                haskey(params, req) ||
                    error("params.$(req) required when warm_start == steady_state")

            end

            haskey(g, :a_min) ||
                error("grids.a_min required when warm_start == steady_state")

        end

    end



    # Projection-specific

    if mth == "projection"

        if haskey(s, :orders)

            ords = s[:orders]

            (ords isa AbstractVector{<:Integer} && !isempty(ords)) ||
                error("solver.orders must be a non-empty Vector{Int}; got $(typeof(ords))")

            maxord = g[:Na] - 1

            all(o -> 0 <= o <= maxord, ords) || error(
                "each order in solver.orders must satisfy 0 <= order <= $(maxord); got $(ords)",
            )

        end

        if haskey(s, :Nval)

            (s[:Nval] isa Integer && s[:Nval] >= 2) ||
                error("solver.Nval must be Int >= 2; got $(s[:Nval])")

        end

    end



    # Perturbation-specific

    if mth == "perturbation"

        if haskey(s, :order)

            (s[:order] isa Integer && s[:order] >= 1) ||
                error("solver.order must be Int >= 1; got $(s[:order])")

        end

        if haskey(s, :a_bar)

            abar = s[:a_bar]

            (abar === nothing || (abar isa Real && g[:a_min] <= abar <= g[:a_max])) ||
                error("solver.a_bar must be in [a_min, a_max]; got $(abar)")

        end

        if get(s, :order, 1) >= 2

            if haskey(s, :h_a) && s[:h_a] !== nothing

                (s[:h_a] isa Real && s[:h_a] > 0) ||
                    error("solver.h_a must be Real > 0; got $(s[:h_a])")

            end

            if haskey(cfg, :shocks) && get(cfg[:shocks], :active, false)

                if haskey(s, :h_z) && s[:h_z] !== nothing

                    (s[:h_z] isa Real && s[:h_z] > 0) || error(
                        "solver.h_z must be Real > 0 in stochastic case; got $(s[:h_z])",
                    )

                end

            end

        end

        if haskey(s, :tol_fit)

            (s[:tol_fit] isa Real && s[:tol_fit] > 0) ||
                error("solver.tol_fit must be Real > 0; got $(s[:tol_fit])")

        end

        if haskey(s, :maxit_fit)

            (s[:maxit_fit] isa Integer && s[:maxit_fit] >= 1) ||
                error("solver.maxit_fit must be Int >= 1; got $(s[:maxit_fit])")

        end

    end



    # --- Shocks (optional) ---

    if haskey(cfg, :shocks) &&
       cfg[:shocks] isa AbstractDict &&
       get(cfg[:shocks], :active, false)

        sh = cfg[:shocks]

        # method

        mth = _lower(get(sh, :method, "tauchen"))

        mth in ("tauchen", "rouwenhorst") || error(
            "shocks.method must be tauchen or rouwenhorst; got $(get(sh, :method, nothing))",
        )

        # keys and ranges

        ρ_key =
            haskey(sh, Symbol("ρ_shock")) ? Symbol("ρ_shock") :
            (haskey(sh, :ρ_shock) ? :ρ_shock : nothing)

        ρ =
            ρ_key === nothing ? error("shocks.ρ_shock missing when shocks.active = true") :
            sh[ρ_key]

        (ρ isa Real && -1 < ρ < 1) ||
            error("shocks.ρ_shock must be Real in (-1,1); got $(ρ)")

        sigma_key =
            haskey(sh, Symbol("σ_shock")) ? Symbol("σ_shock") :
            (haskey(sh, :s_shock) ? :s_shock : nothing)

        sigma_key === nothing &&
            error("shocks.σ_shock (or s_shock) missing when shocks.active = true")

        s_e = sh[sigma_key]

        (s_e isa Real) ||
            error("shocks.$(String(sigma_key)) must be a number; got $(typeof(s_e))")

        s_e >= 0 || error("shocks.s_shock must be >= 0; got $(s_e)")

        Nz = _getnum(sh, :Nz; name = "shocks.Nz")

        (Nz isa Integer) || error("shocks.Nz must be Integer; got $(typeof(Nz))")

        (Nz >= 2 || s_e == 0) ||
            error("shocks.Nz must be >= 2 when s_shock > 0; got Nz=$(Nz)")

        if mth == "tauchen" && haskey(sh, :m)

            (sh[:m] isa Real && sh[:m] > 0) ||
                error("shocks.m must be Real > 0; got $(sh[:m])")

        end

        if haskey(sh, :validate)

            (sh[:validate] isa Bool) ||
                error("shocks.validate must be Bool; got $(typeof(sh[:validate]))")

        end

    end



    # --- Warm start overrides (init) ---

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
            ) || error("init.c must be a (Na, Nz) matrix; got size $(size(c0))")

        else

            (c0 isa AbstractVector && length(c0) == Na) ||
                error("init.c must be a vector of length Na=$(Na);")

        end

        # basic positivity check

        all(x -> x > 0, c0) || error("init.c must be strictly positive")

    end



    # --- Random seed (optional) ---

    if haskey(cfg, :random) && cfg[:random] isa AbstractDict && haskey(cfg[:random], :seed)

        sd = cfg[:random][:seed]

        try

            _ = UInt64(sd)

        catch

            error("random.seed must be an integer (or coercible); got $(sd)")

        end

    end



    return true

end



end
