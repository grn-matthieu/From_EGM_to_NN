using Dates
using Printf
using Lux
using .NNMixedPrecision: to_fp32, UseFP16, UseBF16

"""
    bench_mixedprecision(cfg; warmup_epochs=1, run_epochs=2, on_row=nothing)

Benchmark FP32/FP16/BF16 training on the fixed batch stored in `cfg[:batch]`.

- RNG isolation: seeds are derived via `cfg[:random][:seed]` (or `cfg[:seed]`
  fallback) without mutating `Random.default_rng()`.
- Logging: pass an `on_row` callback to receive per-mode metrics instead of
  writing CSV files. The callback receives a NamedTuple
  `(:mp, :epochs, :wall_time_s, :loss, :feas, :loss_scale)`.
"""
function bench_mixedprecision(cfg; warmup_epochs = 1, run_epochs = 2, on_row = nothing)
    # Settings
    mp_modes = [(nothing, "FP32"), (UseFP16(), "FP16"), (UseBF16(), "BF16")]
    results = NamedTuple[]
    # No direct CSV output; caller can pass `on_row` for logging.

    # Helper: run training for a given mp mode
    function run_one(mp, mp_name)
        # Derive local RNG via cfg.random.seed to avoid global Random.seed!
        cfg2 = deepcopy(cfg)
        cfg2 = cfg2 isa Dict{Symbol,Any} ? cfg2 : Dict{Symbol,Any}(cfg2)
        if haskey(cfg2, :seed)
            rand_cfg = Dict{Symbol,Any}(get(cfg2, :random, Dict{Symbol,Any}()))
            rand_cfg[:seed] = cfg2[:seed]
            cfg2[:random] = rand_cfg
        elseif haskey(cfg2, :random)
            cfg2[:random] = Dict{Symbol,Any}(cfg2[:random])
        end
        batch = get(cfg2, :batch, nothing)
        batch === nothing &&
            throw(ArgumentError("bench_mixedprecision requires cfg[:batch]=(X, Y)"))
        state = NNInit.init_state(cfg2)
        # Local mutable bindings for parameters and optimiser state
        ps_local = state.ps
        optstate_local = state.optstate
        opt_local = state.opt
        loss_scale = get(cfg2, :loss_scale, 1.0)
        cast_params_tree(x, ::Type{T}) where {T} = x
        cast_params_tree(x::NamedTuple, ::Type{T}) where {T} =
            map(v -> cast_params_tree(v, T), x)
        cast_params_tree(x::Tuple, ::Type{T}) where {T} =
            map(v -> cast_params_tree(v, T), x)
        cast_params_tree(x::AbstractArray, ::Type{T}) where {T} = convert.(T, x)

        function _compute_loss(p_now, batch_now)
            X, Y = batch_now
            ŷ, st2 = Lux.apply(state.model, X, p_now, state.st)
            diff32 = to_fp32(ŷ .- Y)
            s = sum(abs2, diff32)
            return Float64(s / length(diff32))
        end

        function loss_eval(ps)
            if mp === nothing
                return _compute_loss(ps, batch)
            else
                T = NNMixedPrecision.eltype_from(mp)
                ps_mp = cast_params_tree(ps, T)
                batch_mp = NNMixedPrecision.cast_batch(batch, T)
                return _compute_loss(ps_mp, batch_mp)
            end
        end
        feas = 0.0
        loss = NaN
        wall_time = 0.0
        try
            # Helper: local grad norm (avoid depending on other private helpers)
            function _grad_global_l2norm_local(grads)
                s = 0.0
                _rec(x) = begin
                    if x isa NamedTuple
                        for v in values(x)
                            _rec(v)
                        end
                    elseif x isa Tuple
                        for v in x
                            _rec(v)
                        end
                    elseif x isa AbstractArray
                        s += sum(abs2, x)
                    else
                        return
                    end
                end
                _rec(grads)
                return sqrt(s)
            end

            # Warmup epochs (not timed) -- single-step per epoch on the provided batch
            for _ = 1:warmup_epochs
                # Compute grads w.r.t. master params and update optimiser
                gs = first(Zygote.gradient(loss_eval, ps_local))
                # Optional gradient clipping if configured
                clip_norm = get(cfg2, :clip_norm, nothing)
                if clip_norm !== nothing && isfinite(clip_norm) && clip_norm > 0
                    gnorm = _grad_global_l2norm_local(gs)
                    if gnorm > clip_norm
                        scale = clip_norm / (gnorm + 1e-12)
                        NNTrain.foreach_array_leaf(gs) do g
                            @. g = scale * g
                        end
                    end
                end

                # Optimiser update
                if opt_local isa NNOptim.Optimizer
                    pvec = NNTrain.collect_array_leaves(ps_local)
                    gvec = NNTrain.collect_array_leaves(gs)
                    update!(opt_local, pvec, gvec)
                else
                    optstate_local, ps_local =
                        Optimisers.update(optstate_local, ps_local, gs)
                end
            end

            # Timed epochs
            t0 = time()
            for _ = 1:run_epochs
                gs = first(Zygote.gradient(loss_eval, ps_local))
                # Optional clipping
                clip_norm = get(cfg2, :clip_norm, nothing)
                if clip_norm !== nothing && isfinite(clip_norm) && clip_norm > 0
                    gnorm = _grad_global_l2norm_local(gs)
                    if gnorm > clip_norm
                        scale = clip_norm / (gnorm + 1e-12)
                        NNTrain.foreach_array_leaf(gs) do g
                            @. g = scale * g
                        end
                    end
                end

                if opt_local isa NNOptim.Optimizer
                    pvec = NNTrain.collect_array_leaves(ps_local)
                    gvec = NNTrain.collect_array_leaves(gs)
                    update!(opt_local, pvec, gvec)
                else
                    optstate_local, ps_local =
                        Optimisers.update(optstate_local, ps_local, gs)
                end
            end
            wall_time = time() - t0

            # Evaluate final loss/feasibility on the trained weights
            loss = loss_eval(ps_local)
            feas = isfinite(loss) ? 1.0 : 0.0
        catch err
            @warn "Error in $mp_name benchmark: $err"
            wall_time = NaN
            loss = NaN
            feas = 0.0
        end
        # After run, if the original state is an NNState, copy trained arrays back
        function _copy_tree_arrays!(dest, src)
            if dest isa NamedTuple && src isa NamedTuple
                for k in keys(dest)
                    _copy_tree_arrays!(getfield(dest, k), getfield(src, k))
                end
            elseif dest isa Tuple && src isa Tuple
                for i in eachindex(dest)
                    _copy_tree_arrays!(dest[i], src[i])
                end
            elseif dest isa AbstractArray && src isa AbstractArray
                @assert size(dest) == size(src)
                dest .= src
            else
                # numbers or unsupported leaves are ignored
            end
            return dest
        end

        # Copy back into state.ps so the caller-observed state reflects training
        try
            _copy_tree_arrays!(state.ps, ps_local)
            # Update optstate in-place if possible
            try
                state.optstate = optstate_local
            catch
                # if NNState fields are immutable, ignore
            end
        catch
            # best-effort; continue
        end

        # Optional logging hook instead of direct CSV emission
        if on_row !== nothing
            on_row((
                mp = mp_name,
                epochs = run_epochs,
                wall_time_s = wall_time,
                loss = loss,
                feas = feas,
                loss_scale = loss_scale,
            ))
        end
        return (
            mp = mp_name,
            wall_time = wall_time,
            loss = loss,
            feas = feas,
            loss_scale = loss_scale,
        )
    end

    # Run all modes
    for (mp, mp_name) in mp_modes
        push!(results, run_one(mp, mp_name))
    end

    # Print summary
    fp32 = findfirst(r -> r.mp == "FP32", results)
    fp16 = findfirst(r -> r.mp == "FP16", results)
    bf16 = findfirst(r -> r.mp == "BF16", results)
    if fp32 !== nothing && fp16 !== nothing
        t_fp32 = results[fp32].wall_time
        t_fp16 = results[fp16].wall_time
        loss_fp32 = results[fp32].loss
        loss_fp16 = results[fp16].loss
        feas_fp16 = results[fp16].feas
        if isfinite(t_fp16) && isfinite(loss_fp16) && feas_fp16 >= 0.99
            println(@sprintf("FP16 speedup = %.2fx", t_fp32 / t_fp16))
            println(@sprintf("delta_loss = %.3e", loss_fp16 - loss_fp32))
        else
            println("FP16 UNSTABLE")
        end
    end
    if fp32 !== nothing && bf16 !== nothing
        t_fp32 = results[fp32].wall_time
        t_bf16 = results[bf16].wall_time
        loss_fp32 = results[fp32].loss
        loss_bf16 = results[bf16].loss
        feas_bf16 = results[bf16].feas
        if isfinite(t_bf16) && isfinite(loss_bf16) && feas_bf16 >= 0.99
            println(@sprintf("BF16 speedup = %.2fx", t_fp32 / t_bf16))
            println(@sprintf("delta_loss = %.3e", loss_bf16 - loss_fp32))
        else
            println("BF16 UNSTABLE")
        end
    end
    return results
end
"""
NNTrain

Training utilities and logging helpers for NN solvers, including simple CSV
logging, curricula, and convenience wrappers around mixed-precision.
"""
module NNTrain

using Lux
using Zygote
using Optimisers
include("optim.jl")
using .NNOptim
using Statistics
using Dates
using Printf
using Random

using ..NNData: grid_minibatches
using ..NNLoss: anneal_λ
using ..NNMixedPrecision: with_mixed_precision, to_fp32
# Example usage in a training loop (insert at appropriate call site):
# with_mixed_precision(model, params, batch; mp=cfg.mixed_precision, loss_scale=cfg.loss_scale) do (pmp, bmp)
#     # forward, loss, grads (upcast for reductions), update master FP32
# end

using ..NNInit: NNState, init_state

export train!, dummy_epoch!

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

"""Apply function `f(::AbstractArray)` to each array leaf in a nested tree."""
function foreach_array_leaf(x, f::F) where {F}
    if x isa NamedTuple
        for v in values(x)
            foreach_array_leaf(v, f)
        end
    elseif x isa Tuple
        for v in x
            foreach_array_leaf(v, f)
        end
    elseif x isa AbstractArray
        f(x)
    elseif x === nothing
        return
    else
        # numbers or other leaves are ignored
        return
    end
end

"""Compute global L2 norm of a gradient tree (sum of leaf Frobenius norms)."""
function grad_global_l2norm(grads)::Float64
    s = 0.0
    foreach_array_leaf(grads) do g
        s += sum(abs2, g)
    end
    return sqrt(s)
end

"""Scale all array leaves by factor `α` in place."""
function scale_grads!(grads, α::Real)
    foreach_array_leaf(grads) do g
        @. g = α * g
    end
    return grads
end


# ---- Core training step ----

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

# Collect array leaves utility for params/grads vectors
collect_array_leaves(x) = begin
    acc = Vector{AbstractArray}()
    foreach_array_leaf(x) do a
        push!(acc, a)
    end
    acc
end

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

    # Optimiser update (prefer NNOptim if available)
    new_ps = state.ps
    new_optstate = state.optstate
    if state.opt isa NNOptim.Optimizer
        pvec = collect_array_leaves(new_ps)
        gvec = collect_array_leaves(gs)
        update!(state.opt, pvec, gvec)
    else
        new_optstate, new_ps = Optimisers.update(state.optstate, state.ps, gs)
    end

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
