using Dates
using Printf
using Lux
using ..NNUtils: to_fp32
"""
NNTrain

Training utilities and logging helpers for NN solvers, including simple CSV
logging, curricula, and convenience wrappers around mixed-precision.
"""
module NNTrain

using Lux
using Zygote
using Optimisers
# Local optimizer implementation removed; use Optimisers.jl instead
using Statistics
using Dates
using Printf
using Random

using ..NNData: grid_minibatches
using ..NNLoss: anneal_λ
using ..NNUtils:
    to_fp32,
    foreach_array_leaf,
    collect_params_leaves,
    grad_global_l2norm_params,
    scale_grads!,
    _copy_tree_arrays!
# Example usage in a training loop (insert at appropriate call site):
# with_mixed_precision(model, params, batch; mp=cfg.mixed_precision, loss_scale=cfg.loss_scale) do (pmp, bmp)
#     # forward, loss, grads (upcast for reductions), update master FP32
# end

using ..NNInit: NNState, init_state

export train!, dummy_epoch!

# Prefer parameter-aware collectors when available (Flux/Functors wrappers in NNUtils)
# Create safe local forwarders (avoid method-extension of imported names)
const collect_array_leaves = collect_params_leaves
function grad_global_l2norm(x)
    try
        return grad_global_l2norm_params(x)
    catch
        return NNUtils.grad_global_l2norm(x)
    end
end

## Default λ scheduling parameters (overridable via cfg)
const DEFAULT_Λ_START = 0.1
const DEFAULT_Λ_FINAL = 5.0
const DEFAULT_Λ_SCHEDULE = :cosine

"""
    curriculum(epoch, E; stages=default_stages()) -> NamedTuple

Coarse-fine schedule over epochs. Selects an active stage from `stages`
based on the current `epoch` in 1..E and returns its parameters as a
NamedTuple.

Fields in the returned NamedTuple:
- `grid_stride`: Subsample factor for state grid or minibatch indices.
- `nMC`: Number of Monte Carlo draws for expectations.
- `shock_noise`: Multiplier applied to innovation std in the sampler.
"""
function curriculum(epoch::Integer, E::Integer; stages::Vector = default_stages())
    E <= 0 && throw(ArgumentError("E must be positive, got $(E)"))
    epoch < 1 && throw(ArgumentError("epoch must be >= 1, got $(epoch)"))
    nstages = length(stages)
    nstages == 0 && throw(ArgumentError("stages must be non-empty"))
    # Map epoch in [1, E] to stage index in [1, nstages]
    frac = clamp(epoch / E, 0.0, 1.0)
    idx = max(1, min(nstages, ceil(Int, frac * nstages)))
    st = stages[idx]
    return (
        name = get(st, :name, Symbol("stage$(idx)")),
        grid_stride = get(st, :grid_stride, 1),
        nMC = get(st, :nMC, 1),
        shock_noise = get(st, :shock_noise, 1.0),
    )
end

"""
    default_stages() -> Vector{NamedTuple}

Default curriculum schedule (override via config):
[
  (; name=:warmup, grid_stride=4, nMC=1, shock_noise=1.25),
  (; name=:mid,    grid_stride=2, nMC=2, shock_noise=1.00),
  (; name=:fine,   grid_stride=1, nMC=4, shock_noise=0.75),
]
"""
default_stages() = [
    (; name = :warmup, grid_stride = 4, nMC = 1, shock_noise = 1.25),
    (; name = :mid, grid_stride = 2, nMC = 2, shock_noise = 1.00),
    (; name = :fine, grid_stride = 1, nMC = 4, shock_noise = 0.75),
]

"""Thin a batch-like array by taking every `stride`-th sample.

If `x` is 2D (features × samples), subsamples columns. If 1D, subsamples
elements. Non-arrays or `stride <= 1` are returned unchanged."""
function _thin(x, stride::Integer)
    stride <= 1 && return x
    if x isa AbstractArray
        if ndims(x) == 1
            return view(x, 1:stride:length(x))
        elseif ndims(x) >= 2
            ns = size(x, 2)
            return @views x[:, 1:stride:ns]
        end
    end
    return x
end

"""
    CSVLogger(path::AbstractString)

Lightweight CSV logger that appends rows. Creates parent directories and
auto-writes a header if the file does not exist or is empty.
"""
struct CSVLogger
    path::String
    header_written::Base.RefValue{Bool}
end

function CSVLogger(path::AbstractString)
    p = String(path)
    mkpath(dirname(p))
    header_written = Base.RefValue(false)
    if isfile(p) && filesize(p) > 0
        header_written[] = true
    end
    return CSVLogger(p, header_written)
end

function log_row!(
    lg::CSVLogger;
    epoch::Integer,
    step::Integer,
    split::AbstractString,
    loss::Real,
    grad_norm::Real,
    lr::Real,
    # Optional curriculum logging fields (per-epoch)
    stage = nothing,
    grid_stride = nothing,
    nMC = nothing,
    shock_noise = nothing,
    λ_penalty = nothing,
)
    open(lg.path, lg.header_written[] ? "a" : "w") do io
        if !lg.header_written[]
            println(
                io,
                "timestamp,epoch,step,split,loss,grad_norm,lr,stage,grid_stride,nMC,shock_noise,λ_penalty",
            )
            lg.header_written[] = true
        end
        ts = Dates.format(Dates.now(), dateformat"yyyy-mm-ddTHH:MM:SS")
        # Helper to print optional values as NA when absent
        _s(x) = x === nothing ? "NA" : string(x)
        _i(x) = x === nothing ? "NA" : string(Int(x))
        _f(x) = x === nothing ? "NA" : @sprintf("%.6e", float(x))
        # Write full row including curriculum columns
        @printf(
            io,
            "%s,%d,%d,%s,%.6e,%.6e,%.6e,%s,%s,%s,%s,%s\n",
            ts,
            epoch,
            step,
            split,
            float(loss),
            float(grad_norm),
            float(lr),
            _s(stage),
            _i(grid_stride),
            _i(nMC),
            _f(shock_noise),
            _f(λ_penalty),
        )
        nothing
    end
    return nothing
end


"""
    EarlyStopping(; patience=5, min_delta=0.0)

Tracks the best metric and signals when to stop.
"""
Base.@kwdef mutable struct EarlyStopping
    patience::Int = 5
    min_delta::Float64 = 0.0
    best::Float64 = Inf
    num_bad::Int = 0
    enabled::Bool = true
end

function reset!(es::EarlyStopping)
    es.best = Inf
    es.num_bad = 0
    return es
end

"""
    should_stop!(es, metric) -> Bool

Updates internal counters and returns true if early stop is triggered.
Lower metric is considered better.
"""
function should_stop!(es::EarlyStopping, metric::Real)
    if !es.enabled
        return false
    end
    if metric < es.best - es.min_delta
        es.best = float(metric)
        es.num_bad = 0
        return false
    else
        es.num_bad += 1
        return es.num_bad >= es.patience
    end
end


# ---- Small utilities over parameter trees ----

# ---- Core training step (helpers come from ..NNUtils) ----

"""
    _loss_and_state(model, ps, st, x, y)

Forward pass returning MSE loss and the next state. Shapes are flexible; as a
convention `x` and `y` should be arrays whose leading dimension indexes
features, consistent with Lux.
"""
function _loss_and_state(model, ps, st, x, y)
    ŷ, st2 = Lux.apply(model, x, ps, st)
    return mean(abs2, ŷ .- y), st2
end

# collect_array_leaves provided by ..NNUtils

"""
    _step!(state::NNState, x, y; clip_norm=nothing)

Performs a single optimisation step and returns `(new_state, loss, grad_norm, lr)`.
`NNState` is immutable, so a new updated instance is returned.
"""
function _step!(
    state::NNState,
    x,
    y;
    clip_norm = nothing,
    nMC::Integer = 1,
    shock_noise::Real = 1.0,
    λ::Real = NaN,
)
    # Compute loss and grads (grads w.r.t. parameters only)
    # We compute grads against the current state.st; a fresh state from the
    # forward pass is then stored for the next iteration.
    loss_val, st_new = _loss_and_state(state.model, state.ps, state.st, x, y)
    gs = first(
        Zygote.gradient(
            p -> first(_loss_and_state(state.model, p, state.st, x, y)),
            state.ps,
        ),
    )

    gnorm = grad_global_l2norm(gs)
    if clip_norm !== nothing && isfinite(clip_norm) && clip_norm > 0 && gnorm > clip_norm
        scale_grads!(gs, clip_norm / (gnorm + 1e-12))
    end

    # Optimiser update: use Optimisers API
    new_optstate, new_ps = Optimisers.update(state.optstate, state.ps, gs)

    # Learning rate (best-effort; may be absent in some rules)
    lr = try
        getfield(state.opt, :η)
    catch
        try
            getfield(state.opt, :η)
        catch
            try
                getfield(state.opt, :lr)
            catch
                NaN
            end
        end
    end

    new_state = NNState(
        model = state.model,
        ps = new_ps,
        st = st_new,
        opt = state.opt,
        optstate = new_optstate,
        rngs = state.rngs,
    )
    return new_state, loss_val, gnorm, lr
end

"""
    train!(state::NNState, data, cfg; val_data=nothing) -> NNState

Simple training loop used by tests: iterate over `epochs` and call `_step!`
for each minibatch in `data`. Overload that accepts a model initializes a
state via `init_state(cfg)`.
"""
function train!(state::NNState, data, cfg; val_data = nothing)
    solver_cfg = get(cfg, :solver, Dict{Symbol,Any}())
    epochs = get(solver_cfg, :epochs, 1)
    clip_norm = get(solver_cfg, :clip_norm, nothing)
    st = state
    for epoch = 1:epochs
        for (x, y) in data
            st, loss, gnorm, lr = _step!(st, x, y; clip_norm = clip_norm)
        end
    end
    return st
end

function train!(model, data, cfg; val_data = nothing)
    st = init_state(cfg)
    return train!(st, data, cfg; val_data = val_data)
end

function dummy_epoch!(; n::Integer = 32, batch::Integer = 8, epochs::Integer = 1)
    cfg = Dict{Symbol,Any}()
    st = init_state(cfg)
    steps = max(1, div(n, batch))
    for _ = 1:epochs
        for _ = 1:steps
            x = rand(Float32, 1, batch)
            y = rand(Float32, 1, batch)
            st, _, _, _ = _step!(st, x, y)
        end
    end
    return st
end

end # module NNTrain
