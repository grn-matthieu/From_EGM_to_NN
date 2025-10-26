module TestUtils

using ThesisProject
const Determinism = ThesisProject.Determinism
const UtilsConfig = ThesisProject.UtilsConfig

export deterministic_config,
    stochastic_config, deep_merge, master_rng, build_model_and_method, derive_solver_rng

deep_merge(a::NamedTuple, b) = merge(a, b)
deep_merge(a, b) = b

const DEFAULT_SEED = 2024

"""Return a reusable `MasterRNG` used across tests."""
master_rng(seed::Integer = DEFAULT_SEED) = Determinism.make_master_rng(seed)

_single(k::Symbol, v) = NamedTuple{(k,)}((v,))

function deep_merge(a::NamedTuple, b::NamedTuple)
    result = a
    for key in keys(b)
        vb = getfield(b, key)
        if hasproperty(result, key)
            va = getfield(result, key)
            if va isa NamedTuple && vb isa NamedTuple
                result = merge(result, _single(key, deep_merge(va, vb)))
            else
                result = merge(result, _single(key, vb))
            end
        else
            result = merge(result, _single(key, vb))
        end
    end
    return result
end

deep_merge(a::NamedTuple, b) = merge(a, b)
deep_merge(a, b) = b

# Always use backend-only grid config for tests
function _base_solver_block(; method = :EGM)
    return (
        method = method,
        tol = 1.0e-5,
        tol_pol = 1.0e-6,
        maxit = 400,
        verbose = false,
        relax = 0.5,
        warm_start = :default,
        egm = (interp_kind = :linear,),
        time_iteration = (interp_kind = :linear,),
        projection = (orders = [3], Nval = 24),
        perturbation = (
            order = 1,
            a_bar = nothing,
            h_a = nothing,
            h_z = nothing,
            tol_fit = 1.0e-8,
            maxit_fit = 20,
        ),
        nn = (
            epochs = 2,
            batch = 16,
            lr = 1.0e-3,
            hid1 = 8,
            hid2 = 8,
            samples_per_epoch = 32,
            objective = :euler_fb_aio,
            v_h = 0.5,
            w_min = 0.1,
            w_max = 4.0,
            sigma_shocks = nothing,
            target_loss = 1.0e-2,
            use_cuda = false,
            n_mc = 16,
        ),
        grid = (type = :dense, dense = nothing), # always backend-only
    )
end

function deterministic_config(;
    method::Union{String,Symbol} = :EGM,
    Na::Int = 21,
    a_min::Real = 0.0,
    a_max::Real = 5.0,
    β::Real = 0.96,
    γ::Real = 2.0,
    r::Real = 0.02,
    y::Real = 1.0,
    solver_overrides::NamedTuple = NamedTuple(),
    random_seed::Integer = DEFAULT_SEED,
    utility_type = :CRRA,
    shocks::Union{Nothing,NamedTuple} = nothing,
)
    solver_cfg = _base_solver_block(method = method)
    solver_cfg = deep_merge(solver_cfg, solver_overrides)
    random_cfg = (seed = UInt64(random_seed), master_rng = master_rng(random_seed))
    cfg = (
        model = (name = "cs",),
        params = (β = β, γ = γ, r = r, y = y),
        grids = (Na = Na, a_min = a_min, a_max = a_max),
        utility = (u_type = utility_type,),
        solver = solver_cfg,
        random = random_cfg,
    )
    if shocks !== nothing
        cfg = merge(cfg, (shocks = shocks,))
    end
    return UtilsConfig.ensure_master_rng(cfg)
end

function stochastic_config(;
    method::String = "EGM",
    shock_overrides::NamedTuple = NamedTuple(),
    kwargs...,
)
    base_shocks =
        (active = true, method = "tauchen", ρ_shock = 0.9, σ_shock = 0.05, Nz = 3, m = 1.5)
    shocks = deep_merge(base_shocks, shock_overrides)
    return deterministic_config(; method = method, shocks = shocks, kwargs...)
end

function build_model_and_method(cfg::NamedTuple)
    model = ThesisProject.build_model(cfg)
    method = ThesisProject.build_method(cfg)
    return model, method
end

function derive_solver_rng(cfg::NamedTuple, key)
    master = cfg.random.master_rng
    return Determinism.derive_rng(master, key)
end

end
