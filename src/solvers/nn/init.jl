"""
NNInit

Model/state initialization for NN solvers: builds Lux models, parameters,
optimisers, and RNGs based on configuration.
"""
module NNInit

using Lux
using Optimisers

using ..Determinism: make_rng, derive_seed, canonicalize_cfg
using ..NNDevice: move_tree_to_device, is_cuda_available

export NNState, build_nn, init_state

"""
    build_nn(cfg::AbstractDict)

Construct a simple Lux `Chain` for the NN policy approximator.
Input dimension is 1 for deterministic (a), and 2 for stochastic (a, z).
Hidden sizes come from `cfg[:solver][:hidden]` or default to (32, 32).
"""
function build_nn(cfg::AbstractDict)
    # Determine input dimension based on shocks activity
    in_dim = (haskey(cfg, :shocks) && get(cfg[:shocks], :active, false)) ? 2 : 1

    # Read hidden layer sizes
    hidden_raw = get(get(cfg, :solver, Dict{Symbol,Any}()), :hidden, (32, 32))
    hidden =
        (hidden_raw isa AbstractVector || hidden_raw isa Tuple) ?
        collect(Int.(hidden_raw)) : Int[hidden_raw]

    layers = Any[]
    last = in_dim
    for h in hidden
        push!(layers, Lux.Dense(last => h, Lux.relu))
        last = h
    end
    # Final linear head to scalar output
    push!(layers, Lux.Dense(last => 1))
    return Lux.Chain(layers...)
end


"""
    NNState

Holds model, parameters, state, optimizer rule and state, plus RNG seeds.
"""
Base.@kwdef struct NNState
    model::Any
    ps::NamedTuple
    st::NamedTuple
    opt::Any
    optstate::Any
    rngs::NamedTuple
end


"""
    init_state(cfg::AbstractDict) -> NNState

Initializes the NN model parameters and optimizer state with RNG isolation.
Seeds are derived from `cfg[:random][:seed]` without touching global RNG.
"""
function init_state(cfg::AbstractDict)::NNState
    model = build_nn(cfg)

    # Master RNG: from cfg.random.seed if present, else deterministic from cfg
    seed = get(get(cfg, :random, Dict{Symbol,Any}()), :seed, nothing)
    master_rng = seed === nothing ? make_rng(Int(0x9a9aa9a9)) : make_rng(Int(seed))

    # Derive independent RNGs for params and data
    seed_params = derive_seed(master_rng, "nn/params")
    seed_data = derive_seed(master_rng, "nn/data")
    rng_params = make_rng(Int(seed_params % typemax(Int)))
    rng_data = make_rng(Int(seed_data % typemax(Int)))

    # Lux params/state
    ps, st = Lux.setup(rng_params, model)

    # Move parameters/state to device if requested
    solver_cfg = get(cfg, :solver, Dict{Symbol,Any}())
    device = get(solver_cfg, :device, :cpu)
    if device === :cuda && !is_cuda_available()
        @warn "CUDA requested but not available; falling back to CPU"
        device = :cpu
    end
    if device !== :cpu
        ps = move_tree_to_device(ps, device)
        st = move_tree_to_device(st, device)
    end

    # Optimiser (default Adam; lr from cfg if provided)
    solver_cfg = get(cfg, :solver, Dict{Symbol,Any}())
    lr = get(solver_cfg, :lr, get(solver_cfg, :η, 1e-3))
    opt = Optimisers.Adam(lr)
    optstate = Optimisers.setup(opt, ps)

    rngs = (; master = nothing, params = rng_params, data = rng_data)
    return NNState(; model, ps, st, opt, optstate, rngs)
end

end # module
