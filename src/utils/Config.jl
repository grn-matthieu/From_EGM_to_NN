module UtilsConfig

import ..API: load_config, validate_config
using YAML
using ..Determinism: MasterRNG, make_master_rng, master_seed

export maybe, maybe_nested, ensure_master_rng

# --- helpers ---
yaml_to_namedtuple(x) = x
function yaml_to_namedtuple(x::AbstractDict)
    pairs = (Symbol(k) => yaml_to_namedtuple(v) for (k, v) in x)
    return (; pairs...)
end
yaml_to_namedtuple(x::AbstractVector) = [yaml_to_namedtuple(v) for v in x]

function ensure_master_rng(cfg::NamedTuple; require::Bool = false)
    if !hasproperty(cfg, :random)
        require && error("missing random section")
        return cfg
    end

    random_raw = getproperty(cfg, :random)
    random_cfg =
        random_raw isa NamedTuple ? random_raw : yaml_to_namedtuple(Dict(random_raw))

    if !hasproperty(random_cfg, :seed) || random_cfg.seed === nothing
        require && error("missing random.seed")
        return merge(cfg, (random = random_cfg,))
    end

    seed_uint = UInt64(random_cfg.seed)
    existing_master =
        hasproperty(random_cfg, :master_rng) && random_cfg.master_rng isa MasterRNG ?
        random_cfg.master_rng : nothing
    master =
        existing_master !== nothing && master_seed(existing_master) == seed_uint ?
        existing_master : make_master_rng(seed_uint)
    random_enriched = merge(random_cfg, (seed = seed_uint, master_rng = master))
    return merge(cfg, (random = random_enriched,))
end

function load_config(path::AbstractString)
    config = yaml_to_namedtuple(YAML.load_file(path))
    config = validate_config(config)
    return config
end
_lower(x) = lowercase(string(x))

# --- validation helpers ---

_getprop(::Any, ::Symbol, default) = default
_getprop(d::NamedTuple, k::Symbol, default) =
    hasproperty(d, k) ? getproperty(d, k) : default
_getprop(d::AbstractDict, k::Symbol, default) = haskey(d, k) ? d[k] : default

function _ensure_numeric_vector(vec; name::AbstractString)
    vec isa AbstractVector || error("$(name) must be a vector")
    length(vec) > 0 || error("$(name) must not be empty")
    for (i, val) in enumerate(vec)
        val isa Real || error("$(name)[$i] not numeric")
    end
    return length(vec)
end

function _ensure_numeric_matrix_repr(mat; name::AbstractString)
    if mat isa AbstractMatrix
        size(mat, 1) > 0 || error("$(name) must have at least one row")
        size(mat, 2) > 0 || error("$(name) must have at least one column")
        for (idx, val) in enumerate(mat)
            val isa Real || error("$(name) entry $idx not numeric")
        end
        return size(mat, 1), size(mat, 2)
    elseif mat isa AbstractVector
        rows = length(mat)
        rows > 0 || error("$(name) must have at least one row")
        first_row = mat[1]
        first_row isa AbstractVector || error("$(name) rows must be vectors")
        cols = length(first_row)
        cols > 0 || error("$(name) must have at least one column")
        for (i, row) in enumerate(mat)
            row isa AbstractVector || error("$(name) rows must be vectors")
            length(row) == cols || error("$(name) rows must have equal length")
            for (j, val) in enumerate(row)
                val isa Real || error("$(name)[$i,$j] not numeric")
            end
        end
        return rows, cols
    else
        error("$(name) must be a matrix or a vector of vectors")
    end
end

function _without_keys(nt::NamedTuple, drop::Set{Symbol})
    if isempty(drop)
        return nt
    end
    return (; (k => getproperty(nt, k) for k in keys(nt) if !(k in drop))...)
end

function _normalize_alias(
    params::NamedTuple,
    canonical::Symbol,
    aliases::Tuple{Vararg{Symbol}},
)
    result = params
    for alias in aliases
        alias == canonical && continue
        if hasproperty(result, alias)
            alias_val = getproperty(result, alias)
            if hasproperty(result, canonical)
                getproperty(result, canonical) == alias_val ||
                    error("params.$alias inconsistent with params.$canonical")
            else
                result = merge(result, (canonical => alias_val,))
            end
        end
    end
    return result
end
_normalize_alias(params::NamedTuple, canonical::Symbol, alias::Symbol) =
    _normalize_alias(params, canonical, (alias,))

function _drop_aliases(
    params::NamedTuple,
    canonical::Symbol,
    aliases::Tuple{Vararg{Symbol}},
)
    drop = Set{Symbol}()
    for alias in aliases
        alias != canonical && push!(drop, alias)
    end
    return _without_keys(params, drop)
end
_drop_aliases(params::NamedTuple, canonical::Symbol, alias::Symbol) =
    _drop_aliases(params, canonical, (alias,))

function _canonicalize_csvec_params(params::NamedTuple)
    Σ_sym = Symbol("Σ")
    params = _normalize_alias(params, :A, (:A, :transition, :state_transition))
    params = _drop_aliases(params, :A, (:A, :transition, :state_transition))
    params = _normalize_alias(params, Σ_sym, (Σ_sym, :Sigma, :covariance))
    params = _drop_aliases(params, Σ_sym, (Σ_sym, :Sigma, :covariance))
    params = _normalize_alias(params, :y, (:y, :income))
    params = _drop_aliases(params, :y, (:y, :income))

    if hasproperty(params, :A)
        Aval = getproperty(params, :A)
        if Aval isa Real
            params = merge(params, (A = [[Aval]],))
        end
    end
    if hasproperty(params, Σ_sym)
        Σval = getproperty(params, Σ_sym)
        if Σval isa Real
            params = merge(params, (Σ_sym => [[Σval]],))
        end
    end
    if hasproperty(params, :y)
        yval = getproperty(params, :y)
        if yval isa Real
            params = merge(params, (y = [yval],))
        end
    end
    return params
end

to_namedtuple_if_dict(x) = x
to_namedtuple_if_dict(x::AbstractDict) = yaml_to_namedtuple(Dict(x))

function require_namedtuple_sections(cfg::NamedTuple, sections::Tuple{Vararg{Symbol}})
    for sect in sections
        hasproperty(cfg, sect) || error("missing $sect")
        getproperty(cfg, sect) isa NamedTuple || error("$sect wrong type")
    end
    return cfg
end

struct SolverValidationInputs
    params::NamedTuple
    grids::NamedTuple
    shocks_active::Bool
end

struct ShocksValidationState
    active::Bool
    Nz::Int
end
ShocksValidationState(; active::Bool = false, Nz::Int = 1) =
    ShocksValidationState(active, Nz)

function validate_model_section(model_cfg::NamedTuple)
    hasproperty(model_cfg, :name) || error("missing model.name")
    return Symbol(model_cfg.name)
end

function validate_params_section(cfg::NamedTuple, model_name::Symbol)
    params_raw = cfg.params
    params_raw = _normalize_alias(params_raw, :r, (:r, :interest_rate))
    params_raw = _drop_aliases(params_raw, :r, (:r, :interest_rate))
    β_sym = Symbol("β")
    params_raw = _normalize_alias(params_raw, β_sym, (β_sym, :beta))
    params_raw = _drop_aliases(params_raw, β_sym, (β_sym, :beta))
    γ_candidates = (:γ, :gamma, :σ, :sigma)
    hasproperty(params_raw, β_sym) || error("missing params.β")
    getproperty(params_raw, β_sym) isa Real || error("params.β not numeric")
    β_val = getproperty(params_raw, β_sym)
    0 < β_val < 1 || error("params.β out of range")

    γ_key = nothing
    for key in γ_candidates
        if hasproperty(params_raw, key)
            γ_key = key
            break
        end
    end
    γ_key === nothing && error("missing params.γ")
    γ_val = getproperty(params_raw, γ_key)
    γ_val isa Real || error("params.$γ_key not numeric")
    γ_val > 0 || error("γ ≤ 0")
    if hasproperty(params_raw, :γ)
        γ_existing = getproperty(params_raw, :γ)
        γ_existing isa Real || error("params.γ not numeric")
        γ_existing > 0 || error("γ ≤ 0")
        if γ_key != :γ && γ_existing != γ_val
            error("params.γ inconsistent with params.$γ_key")
        end
    end

    params_norm = merge(params_raw, (; γ = γ_val, β_sym => β_val))
    params_norm = _drop_aliases(params_norm, :γ, γ_candidates)
    if model_name == :cs_vec
        params_norm = _canonicalize_csvec_params(params_norm)
    end

    cfg = merge(cfg, (params = params_norm,))
    p = cfg.params

    if hasproperty(p, :r)
        getproperty(p, :r) isa Real || error("params.r not numeric")
        getproperty(p, :r) > -1 || error("r ≤ -1")
    end
    if hasproperty(p, :y)
        y_val = getproperty(p, :y)
        if model_name == :cs_vec
            len =
                y_val isa AbstractVector ?
                _ensure_numeric_vector(y_val; name = "params.y") :
                begin
                    y_val isa Real || error("params.y not numeric")
                    y_val > 0 || error("y ≤ 0")
                    1
                end
            len > 0 || error("params.y must not be empty")
        else
            y_val isa Real || error("params.y not numeric")
            y_val > 0 || error("y ≤ 0")
        end
    end

    if model_name == :cs_vec
        hasproperty(p, :A) || error("missing params.A")
        hasproperty(p, :Σ) || error("missing params.Σ")
        Ay = getproperty(p, :A)
        rows_A, cols_A = _ensure_numeric_matrix_repr(Ay; name = "params.A")
        rows_A == cols_A || error("params.A must be square")

        Ey = getproperty(p, :Σ)
        rows_E, _ = _ensure_numeric_matrix_repr(Ey; name = "params.Σ")
        rows_E == rows_A || error("params.Σ must have the same number of rows as params.A")

        if hasproperty(p, :y)
            y_val = getproperty(p, :y)
            y_len = y_val isa AbstractVector ? length(y_val) : 1
            y_len == rows_A ||
                error("params.y length must match dimension implied by params.A")
        end
    end

    return cfg
end

function validate_grids_section(grids::NamedTuple)
    for k in (:Na, :a_min, :a_max)
        hasproperty(grids, k) || error("missing grids.$k")
    end
    grids.Na isa Integer || error("grids.Na not Int")
    grids.Na > 1 || error("grids.Na out of range")
    grids.a_min isa Real || error("a_min not Real")
    grids.a_max isa Real || error("a_max not Real")
    grids.a_max > grids.a_min || error("a_max ≤ a_min")
    return grids
end

function validate_utility_section(cfg::NamedTuple)
    if hasproperty(cfg, :utility)
        util = getproperty(cfg, :utility)
        if util isa NamedTuple && hasproperty(util, :u_type)
            _lower(getproperty(util, :u_type)) in ("crra",) ||
                error("utility.u_type unsupported")
        end
    end
end

const SUPPORTED_METHODS = (:EGM, :Projection, :Perturbation, :NN, :TimeIteration)
const SUPPORTED_GRID_TYPES = (:dense, :sparse, :adaptive_sparse)
const METHOD_BLOCKS = Dict(
    :EGM => :egm,
    :Projection => :projection,
    :Perturbation => :perturbation,
    :NN => :nn,
    :TimeIteration => :time_iteration,
)

function _canonical_method(m)
    str = _lower(m)
    if str == "egm"
        return :EGM
    elseif str == "projection"
        return :Projection
    elseif str == "perturbation"
        return :Perturbation
    elseif str == "nn"
        return :NN
    elseif str == "timeiteration" || str == "ti"
        return :TimeIteration
    elseif str == "all"
        return :ALL
    else
        error("solver.method invalid")
    end
end

function _expand_requested_method(entry)
    canon = _canonical_method(entry)
    return canon == :ALL ? collect(SUPPORTED_METHODS) : [canon]
end

function _canonicalize_requested_methods(requested)
    methods = Symbol[]
    if requested isa AbstractVector
        for entry in requested
            append!(methods, _expand_requested_method(entry))
        end
    else
        append!(methods, _expand_requested_method(requested))
    end
    return unique(methods)
end

function validate_solver_section(solver::NamedTuple, inputs::SolverValidationInputs)
    required_common = (:method, :tol, :tol_pol, :maxit, :verbose, :relax, :warm_start)
    for key in required_common
        hasproperty(solver, key) || error("missing solver.$key")
    end
    solver.tol isa Real && solver.tol > 0 || error("tol > 0 required")
    solver.tol_pol isa Real && solver.tol_pol > 0 || error("tol_pol > 0 required")
    solver.maxit isa Integer && solver.maxit ≥ 1 || error("maxit ≥ 1 required")
    solver.verbose isa Bool || error("verbose not Bool")
    solver.relax isa Real && solver.relax > 0 || error("relax > 0 required")

    warm_start_lower = _lower(solver.warm_start)
    warm_start_lower in ("default", "half_resources", "none", "steady_state") ||
        error("warm_start invalid")
    if warm_start_lower == "steady_state"
        hasproperty(inputs.params, :y) || error("need params.y for steady_state")
        hasproperty(inputs.params, :r) || error("need params.r for steady_state")
        hasproperty(inputs.grids, :a_min) || error("need grids.a_min for steady_state")
    end

    hasproperty(solver, :grid) || error("missing solver.grid")
    grid_raw = getproperty(solver, :grid)
    grid_cfg = to_namedtuple_if_dict(grid_raw)
    grid_cfg isa NamedTuple || error("solver.grid wrong type")
    hasproperty(grid_cfg, :type) || error("missing solver.grid.type")
    grid_type_val = getproperty(grid_cfg, :type)
    (grid_type_val isa Symbol || grid_type_val isa AbstractString) ||
        error("solver.grid.type invalid")
    grid_type = grid_type_val isa Symbol ? grid_type_val : Symbol(grid_type_val)
    grid_type in SUPPORTED_GRID_TYPES || error("solver.grid.type unsupported")
    grid_updates = (type = grid_type,)
    if grid_type == :dense && hasproperty(grid_cfg, :dense)
        dense_raw = getproperty(grid_cfg, :dense)
        dense_norm = dense_raw === nothing ? nothing : to_namedtuple_if_dict(dense_raw)
        !(dense_norm isa AbstractDict) || error("solver.grid.dense wrong type")
        dense_norm === nothing ||
            dense_norm isa NamedTuple ||
            error("solver.grid.dense wrong type")
        if dense_norm !== nothing
            grid_updates = merge(grid_updates, (dense = dense_norm,))
        end
    elseif grid_type in (:sparse, :adaptive_sparse)
        if hasproperty(grid_cfg, :sparse)
            sparse_raw = getproperty(grid_cfg, :sparse)
            sparse_norm =
                sparse_raw === nothing ? NamedTuple() : to_namedtuple_if_dict(sparse_raw)
            sparse_norm isa NamedTuple || error("solver.grid.sparse wrong type")
            any(
                name ->
                    hasproperty(sparse_norm, name) && !(
                        getproperty(sparse_norm, name) isa Real ||
                        getproperty(sparse_norm, name) isa Bool
                    ),
                (:depth, :basis, :anisotropic),
            ) && error("solver.grid.sparse contains invalid entries")
            grid_updates = merge(grid_updates, (sparse = sparse_norm,))
        end
        if hasproperty(grid_cfg, :adaptive)
            adaptive_raw = getproperty(grid_cfg, :adaptive)
            adaptive_norm =
                adaptive_raw === nothing ? NamedTuple() :
                to_namedtuple_if_dict(adaptive_raw)
            adaptive_norm isa NamedTuple || error("solver.grid.adaptive wrong type")
            if hasproperty(adaptive_norm, :surplus_tol)
                tol_val = getproperty(adaptive_norm, :surplus_tol)
                tol_val isa Real && tol_val > 0 || error("surplus_tol must be > 0")
            end
            grid_updates = merge(grid_updates, (adaptive = adaptive_norm,))
        end
    end
    grid_cfg_norm = merge(grid_cfg, grid_updates)

    requested_raw = getproperty(solver, :method)
    requested_methods = _canonicalize_requested_methods(requested_raw)
    isempty(requested_methods) && error("solver.method must specify at least one method")

    for canon in requested_methods
        block = METHOD_BLOCKS[canon]
        hasproperty(solver, block) || error("missing solver.$block")
        block_cfg = getproperty(solver, block)
        block_cfg isa NamedTuple || error("solver.$block wrong type")
        if canon == :EGM
            hasproperty(block_cfg, :interp_kind) || error("missing solver.egm.interp_kind")
            _lower(block_cfg.interp_kind) in ("linear", "pchip", "monotone_cubic") ||
                error("interp_kind invalid")
        elseif canon == :TimeIteration
            hasproperty(block_cfg, :interp_kind) ||
                error("missing solver.time_iteration.interp_kind")
            _lower(block_cfg.interp_kind) in ("linear", "pchip", "monotone_cubic") ||
                error("interp_kind invalid")
        elseif canon == :Projection
            hasproperty(block_cfg, :orders) || error("missing solver.projection.orders")
            ords = block_cfg.orders
            ords isa AbstractVector{<:Integer} && !isempty(ords) || error("orders invalid")
            maxord = inputs.grids.Na - 1
            all(o -> 0 ≤ o ≤ maxord, ords) || error("orders out of range")
            hasproperty(block_cfg, :Nval) || error("missing solver.projection.Nval")
            block_cfg.Nval isa Integer && block_cfg.Nval ≥ 2 || error("Nval ≥ 2 required")
        elseif canon == :Perturbation
            hasproperty(block_cfg, :order) || error("missing solver.perturbation.order")
            block_cfg.order isa Integer && block_cfg.order ≥ 1 ||
                error("order ≥ 1 required")
            hasproperty(block_cfg, :a_bar) || error("missing solver.perturbation.a_bar")
            abar = block_cfg.a_bar
            (
                abar === nothing ||
                (abar isa Real && inputs.grids.a_min ≤ abar ≤ inputs.grids.a_max)
            ) || error("a_bar out of range")
            if block_cfg.order ≥ 2
                if hasproperty(block_cfg, :h_a) && block_cfg.h_a !== nothing
                    block_cfg.h_a isa Real && block_cfg.h_a > 0 || error("h_a > 0 required")
                else
                    error("missing solver.perturbation.h_a")
                end
                if inputs.shocks_active
                    if hasproperty(block_cfg, :h_z) && block_cfg.h_z !== nothing
                        block_cfg.h_z isa Real && block_cfg.h_z > 0 ||
                            error("h_z > 0 required")
                    else
                        error("missing solver.perturbation.h_z")
                    end
                end
            end
            hasproperty(block_cfg, :tol_fit) || error("missing solver.perturbation.tol_fit")
            block_cfg.tol_fit isa Real && block_cfg.tol_fit > 0 ||
                error("tol_fit > 0 required")
            hasproperty(block_cfg, :maxit_fit) ||
                error("missing solver.perturbation.maxit_fit")
            block_cfg.maxit_fit isa Integer && block_cfg.maxit_fit ≥ 1 ||
                error("maxit_fit ≥ 1 required")
        elseif canon == :NN
            hasproperty(block_cfg, :epochs) || error("missing solver.nn.epochs")
            block_cfg.epochs isa Integer && block_cfg.epochs ≥ 1 ||
                error("epochs ≥ 1 required")
            hasproperty(block_cfg, :batch) || error("missing solver.nn.batch")
            block_cfg.batch isa Integer && block_cfg.batch ≥ 1 ||
                error("batch ≥ 1 required")
            hasproperty(block_cfg, :lr) || error("missing solver.nn.lr")
            block_cfg.lr isa Real && block_cfg.lr > 0 || error("lr > 0 required")
            hasproperty(block_cfg, :samples_per_epoch) ||
                error("missing solver.nn.samples_per_epoch")
            block_cfg.samples_per_epoch isa Integer && block_cfg.samples_per_epoch ≥ 1 ||
                error("samples_per_epoch ≥ 1 required")
            hasproperty(block_cfg, :objective) || error("missing solver.nn.objective")
            obj = block_cfg.objective
            obj isa Symbol || obj isa AbstractString || error("objective wrong type")
            hasproperty(block_cfg, :hid1) || error("missing solver.nn.hid1")
            block_cfg.hid1 isa Integer && block_cfg.hid1 ≥ 1 || error("hid1 ≥ 1 required")
            hasproperty(block_cfg, :hid2) || error("missing solver.nn.hid2")
            block_cfg.hid2 isa Integer && block_cfg.hid2 ≥ 1 || error("hid2 ≥ 1 required")
            hasproperty(block_cfg, :v_h) || error("missing solver.nn.v_h")
            block_cfg.v_h isa Real && block_cfg.v_h > 0 || error("v_h > 0 required")
            hasproperty(block_cfg, :w_min) || error("missing solver.nn.w_min")
            block_cfg.w_min isa Real || error("w_min not Real")
            hasproperty(block_cfg, :w_max) || error("missing solver.nn.w_max")
            block_cfg.w_max isa Real || error("w_max not Real")
            block_cfg.w_max ≥ block_cfg.w_min || error("w_max < w_min")
            hasproperty(block_cfg, :sigma_shocks) || error("missing solver.nn.sigma_shocks")
            σs = block_cfg.sigma_shocks
            (σs === nothing || (σs isa Real && σs ≥ 0)) || error("sigma_shocks invalid")
            hasproperty(block_cfg, :target_loss) || error("missing solver.nn.target_loss")
            block_cfg.target_loss isa Real && block_cfg.target_loss > 0 ||
                error("target_loss > 0 required")
            hasproperty(block_cfg, :use_cuda) || error("missing solver.nn.use_cuda")
            uc = block_cfg.use_cuda
            (uc === nothing || uc isa Bool) || error("use_cuda invalid")
        end
    end

    solver_method_field =
        length(requested_methods) == 1 ? requested_methods[1] : requested_methods
    solver_canonical = merge(solver, (method = solver_method_field, grid = grid_cfg_norm))
    return requested_methods, solver_canonical
end

function preview_shocks_active(cfg::NamedTuple)
    if !hasproperty(cfg, :shocks)
        return false
    end
    sh_raw = getproperty(cfg, :shocks)
    sh = to_namedtuple_if_dict(sh_raw)
    return sh isa NamedTuple && _getprop(sh, :active, false)
end

function validate_shocks_section(cfg::NamedTuple)
    has_shocks = hasproperty(cfg, :shocks)
    if !has_shocks
        return cfg, ShocksValidationState()
    end

    sh_raw = getproperty(cfg, :shocks)
    sh = to_namedtuple_if_dict(sh_raw)
    shocks_state = ShocksValidationState()
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
            hasproperty(sh, σsym) ? σsym : (hasproperty(sh, :s_shock) ? :s_shock : nothing)
        σkey === nothing && error("missing shocks.σ_shock")
        s_e = getproperty(sh, σkey)
        s_e isa Real || error("σ_shock not numeric")
        s_e ≥ 0 || error("σ_shock < 0")
        if method_lower == "tauchen" && hasproperty(sh, :m)
            sh.m isa Real && sh.m > 0 || error("m > 0 required")
        end
        if hasproperty(sh, :validate)
            sh.validate isa Bool || error("validate not Bool")
        end
        Nz = Int(_getprop(sh, :Nz, 1))
        shocks_state = ShocksValidationState(active = true, Nz = Nz)
    else
        Nz = Int(_getprop(sh, :Nz, 1))
        shocks_state = ShocksValidationState(active = false, Nz = Nz)
    end

    sh_cfg = sh isa NamedTuple ? sh : sh_raw
    cfg = merge(cfg, (shocks = sh_cfg,))
    return cfg, shocks_state
end

function validate_init_section(
    cfg::NamedTuple,
    grids::NamedTuple,
    shocks_state::ShocksValidationState,
)
    if hasproperty(cfg, :init)
        init_raw = getproperty(cfg, :init)
        initcfg = to_namedtuple_if_dict(init_raw)
        if initcfg isa NamedTuple && hasproperty(initcfg, :c)
            c0 = initcfg.c
            Na = grids.Na
            if shocks_state.active
                Nz = shocks_state.Nz
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
end

function validate_random_section(cfg::NamedTuple)
    hasproperty(cfg, :random) || error("missing random section")
    r_raw = getproperty(cfg, :random)
    rcfg = to_namedtuple_if_dict(r_raw)
    hasproperty(rcfg, :seed) || error("missing random.seed")
    try
        _ = UInt64(rcfg.seed)
    catch
        error("random.seed not integer")
    end
end


function validate_config(cfg::AbstractDict)
    # Normalize Dict-like configs to NamedTuple and delegate to the
    # NamedTuple-specific validator. This keeps `load_config` as the
    # canonical loader while allowing tests/helpers to call validate_config
    # with plain Dict objects.
    return validate_config(yaml_to_namedtuple(Dict(cfg)))
end

function _ensure_consistent_top_level_method(
    cfg::NamedTuple,
    requested_methods::Vector{Symbol},
)
    hasproperty(cfg, :method) || return cfg

    method_raw = getproperty(cfg, :method)
    top_methods = _canonicalize_requested_methods(method_raw)
    top_methods == requested_methods ||
        error("method mismatch between top-level and solver sections")

    canonical_field = length(top_methods) == 1 ? top_methods[1] : top_methods
    return merge(cfg, (method = canonical_field,))
end

function validate_config(cfg::NamedTuple)
    cfg = ensure_master_rng(cfg; require = true)
    cfg = require_namedtuple_sections(cfg, (:model, :params, :grids, :solver))

    model_name = validate_model_section(cfg.model)
    cfg = validate_params_section(cfg, model_name)

    grids = validate_grids_section(cfg.grids)
    validate_utility_section(cfg)

    shocks_active_preview = preview_shocks_active(cfg)
    solver_inputs = SolverValidationInputs(cfg.params, grids, shocks_active_preview)
    requested_methods, solver_canonical = validate_solver_section(cfg.solver, solver_inputs)
    cfg = merge(cfg, (solver = solver_canonical,))
    cfg = _ensure_consistent_top_level_method(cfg, requested_methods)

    cfg, shocks_state = validate_shocks_section(cfg)
    validate_init_section(cfg, grids, shocks_state)
    validate_random_section(cfg)

    return cfg
end

maybe(x; default = nothing) = x === nothing ? default : x
maybe(cfg::NamedTuple, key::Symbol; default = nothing) =
    hasproperty(cfg, key) ? getproperty(cfg, key) : default
maybe(cfg, ::Symbol; default = nothing) = default

# Support calling with a positional default argument (common call pattern
# throughout the repo). These overloads avoid MethodError when the third
# argument is a concrete default value (e.g. a number) instead of a Symbol.
maybe(cfg, key::Symbol, default) = maybe(cfg, key; default = default)

"""
    maybe_nested(cfg, keys...; default = nothing)

Traverse a nested configuration `cfg` following the provided `keys` and return
the resulting value. If any intermediate key is missing (or resolves to
`nothing`), return `default` instead. The final value also falls back to the
`default` when it is `nothing`.
"""
maybe_nested(cfg, keys::Symbol...; default = nothing) =
    maybe_nested(cfg, Tuple(keys); default = default)

maybe_nested(::Nothing, ::Symbol...; default = nothing) = default

function maybe_nested(cfg, keys::Tuple{Vararg{Symbol}}; default = nothing)
    isempty(keys) && return maybe(cfg; default = default)

    current = cfg
    for key in keys
        if current === nothing
            return default
        end
        current = maybe(current, key; default = nothing)
    end

    return maybe(current; default = default)
end

end # module
