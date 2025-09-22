"""
NNKernel

Implements neural network-based solvers for consumption policy approximation.
Uses Lux.jl for building and training MLP models with Euler equation residuals
as the loss function to ensure economic consistency.
"""
module NNKernel

using ..API: get_params, get_grids, get_shocks, get_utility, Solution
using ..CommonInterp: InterpKind, LinearInterp
using ..DataNN: generate_dataset
using ..EulerResiduals:
    euler_resid_det, euler_resid_stoch, euler_resid_det_grid, euler_resid_stoch_grid
using Lux
using Optimisers
using Random
using Printf
using Zygote
using Statistics: mean

include("mixed_precision.jl")

export solve_nn

# Small scalar-only params container used for residual evaluation to avoid
# broadcasting/convert issues when passing complex NamedTuples to broadcast
struct ScalarParams
    σ::Float64
    β::Float64
    r::Float64
    y::Float64
end

function _build_default_policy(a_grid)
    # Stub fun that returns a half resources policy
    R = 1.0
    y = 1.0
    resources = @. R * a_grid + y
    return clamp.(0.5 .* resources, 1e-12, resources)
end

function _make_chain(input_dim::Int; hid1::Int = 64, hid2::Int = 64, output_dim::Int = 1)
    # Creates the Lux Chain (no softmax on final layer; regression output)
    # Use softplus on final layer so outputs are positive (consumption)
    return Chain(
        Dense(input_dim, hid1, relu),
        Dense(hid1, hid2, relu),
        Dense(hid2, output_dim, softplus),
    )
end

function solve_nn(model; opts = nothing)
    P, G, S, U = get_params(model), get_grids(model), get_shocks(model), get_utility(model)
    input_dim = isnothing(S) ? 1 : 2

    get_opt(key::Symbol, default) =
        opts === nothing ? default :
        (hasproperty(opts, key) ? getfield(opts, key) : default)

    epochs = Int(get_opt(:epochs, 1000))
    epochs = max(epochs, 0)
    batch_opt = get_opt(:batch, nothing)
    verbose = get_opt(:verbose, false)
    start_time = time_ns()

    # Build Lux chain and optimizer
    chain = _make_chain(input_dim; hid1 = get_opt(:hid1, 128), hid2 = get_opt(:hid2, 128))
    # Reduce default learning rate for stability (can be overridden via opts)
    local_lr = get_opt(:lr, 1e-4)
    opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(0.1), Optimisers.Adam(local_lr))

    ps, st = Lux.setup(Random.GLOBAL_RNG, chain)
    rng = Random.default_rng()

    tstate = Lux.Training.TrainState(chain, ps, st, opt)

    vjp_rule = Lux.AutoZygote()
    resample_every = Int(get_opt(:resample_every, 25))
    target_loss = Float32(get_opt(:target_loss, 2e-4))
    patience = Int(get_opt(:patience, 200))
    best_loss = Inf
    best_state = tstate

    # Generate dataset and preprocess inputs once
    X0, _ = generate_dataset(G, S)

    # Normalize features to [-1,1] for stability
    function _normalize_inputs!(X, G, S)
        a = Float32.(G[:a].grid)
        amin, amax = extrema(a)
        ar = max(amax - amin, eps(Float32))
        norm_a(x) = 2.0f0 * (x - amin) / ar - 1.0f0
        if isnothing(S)
            @. X[:, 1] = norm_a(X[:, 1])
        else
            z = Float32.(S.zgrid)
            zmin, zmax = extrema(z)
            zr = max(zmax - zmin, eps(Float32))
            norm_z(x) = 2.0f0 * (x - zmin) / zr - 1.0f0
            @. X[:, 1] = norm_a(X[:, 1])
            @. X[:, 2] = norm_z(X[:, 2])
        end
        return X
    end

    function _normalize_features!(X, G, S)  # features × batch
        a = Float32.(G[:a].grid)
        amin, amax = extrema(a)
        ar = max(amax - amin, eps(Float32))
        if isnothing(S)
            @. X[1, :] = 2.0f0 * (X[1, :] - amin) / ar - 1.0f0
        else
            z = Float32.(S.zgrid)
            zmin, zmax = extrema(z)
            zr = max(zmax - zmin, eps(Float32))
            @. X[1, :] = 2.0f0 * (X[1, :] - amin) / ar - 1.0f0
            @. X[2, :] = 2.0f0 * (X[2, :] - zmin) / zr - 1.0f0
        end
        return X
    end

    _normalize_inputs!(X0, G, S)
    X_proc = prepare_training_batch(X0)

    total_samples = size(X_proc, 2)
    batch_size =
        batch_opt === nothing ? total_samples : clamp(Int(batch_opt), 1, total_samples)
    batch_size = max(batch_size, 1)

    # Ensure params used by residuals have a numeric `y` to avoid `nothing` during AD
    # Construct a small plain struct with scalar numeric fields to avoid broadcasting
    P_resid = try
        yval = (:y in propertynames(P)) ? getfield(P, :y) : 1.0
        yval = yval === nothing ? 1.0 : yval
        ScalarParams(
            Float64(getfield(P, :σ)),
            Float64(getfield(P, :β)),
            Float64(getfield(P, :r)),
            Float64(yval),
        )
    catch
        # Fallback: try to access by key names or use defaults
        σv = hasfield(typeof(P), :σ) ? Float64(getfield(P, :σ)) : 1.0
        βv = hasfield(typeof(P), :β) ? Float64(getfield(P, :β)) : 0.95
        rv = hasfield(typeof(P), :r) ? Float64(getfield(P, :r)) : 0.02
        ScalarParams(σv, βv, rv, 1.0)
    end

    _huber(x, δ) = (abs(x) ≤ δ) ? 0.5f0 * x * x : δ * (abs(x) - 0.5f0 * δ)

    # Custom loss function using Euler residuals, capturing P_resid, G, S
    loss_function =
        (model, ps, st, data) -> begin
            # data[1] is preprocessed X shaped (features, batch); no targets
            X = data[1]
            # Call model; it may return either the predictions or a (predictions, newstate) tuple
            model_out = model(X, ps, st)
            if model_out isa Tuple
                c_predicted, st_out = model_out
            else
                c_predicted = model_out
                st_out = st
            end
            if isnothing(S)
                a_grid_f32, _, c_pred_vec_f32 = det_residual_inputs(c_predicted, G)
                resid = euler_resid_det_grid(P_resid, a_grid_f32, c_pred_vec_f32)
                @inbounds loss = mean(_huber.(resid, 1.0f0))
            else
                a_grid_f32, z_grid_f32, Pz_f32, _, c_pred_f32 =
                    stoch_residual_inputs(c_predicted, G, S)
                resid = euler_resid_stoch_grid(
                    P_resid,
                    a_grid_f32,
                    z_grid_f32,
                    Pz_f32,
                    c_pred_f32,
                )
                @inbounds loss = mean(_huber.(resid, 1.0f0))
            end
            return loss, st_out, NamedTuple()
        end

    # Helper: robustly compute sum-of-squares across nested gradient containers
    function flatten_sum_squares(x)
        if x === nothing
            return 0.0
        end
        # numbers
        if x isa Number
            return float(x)^2
        end
        # arrays
        if x isa AbstractArray
            # cast to Float64 for stable accumulation
            return sum(abs2, Float64.(x))
        end
        # common containers
        if x isa NamedTuple || x isa Tuple || x isa Vector || x isa Dict
            s = 0.0
            for v in x
                s += flatten_sum_squares(v)
            end
            return s
        end
        # fallback: try struct fields
        try
            s = 0.0
            for f in fieldnames(typeof(x))
                s += flatten_sum_squares(getfield(x, f))
            end
            return s
        catch
            return 0.0
        end
    end

    batches_per_epoch = cld(total_samples, batch_size)
    for epoch = 1:epochs
        if resample_every > 0 && epoch % resample_every == 0
            Xr, _ = generate_dataset(
                G,
                S;
                mode = :rand,
                nsamples = size(X0, 1),
                rng = Random.GLOBAL_RNG,
            )
            _normalize_inputs!(Xr, G, S)
            X_proc = prepare_training_batch(Xr)
            total_samples = size(X_proc, 2)
        end
        epoch_loss = 0.0
        seen = 0
        gnorm = NaN
        X_shuf = X_proc[:, randperm(rng, total_samples)]
        for batch_start = 1:batch_size:total_samples
            batch_end = min(batch_start + batch_size - 1, total_samples)
            X_batch = view(X_shuf, :, batch_start:batch_end)
            batch_data = (X_batch,)
            ginfo, loss, _, tstate =
                Lux.Training.single_train_step!(vjp_rule, loss_function, batch_data, tstate)
            nb = size(X_batch, 2)
            epoch_loss += Float64(loss) * nb
            seen += nb
            # try to compute a robust global gradient norm (RMS-style)
            try
                ssum = flatten_sum_squares(ginfo)
                gnorm = ssum > 0.0 ? sqrt(ssum) : 0.0
            catch
                gnorm = NaN
            end
        end
        avg_loss = epoch_loss / max(seen, 1)
        if avg_loss < best_loss
            best_loss = avg_loss
            best_state = tstate
            stall = 0
        else
            stall = (isdefined(@__MODULE__, :stall) ? stall : 0) + 1
        end
        if verbose && (epoch % 10 == 0 || epoch == epochs)
            @printf "Epoch: %3d \t Loss: %.5g \t GradNorm: %.5g\n" epoch avg_loss gnorm
        end
        # early stop when good enough and not improving
        if best_loss ≤ target_loss && stall ≥ patience
            break
        end
    end

    # After training, build a solution-like NamedTuple so callers (methods) get a consistent result
    # Try to locate chain, params, and state inside tstate using the TrainState fields
    if hasfield(typeof(tstate), :model)
        chain_final = getfield(tstate, :model)
    else
        chain_final = chain
    end

    if hasfield(typeof(tstate), :parameters)
        ps_final = getfield(tstate, :parameters)
    elseif hasfield(typeof(tstate), :params)
        ps_final = getfield(tstate, :params)
    else
        ps_final = nothing
    end

    if hasfield(typeof(tstate), :states)
        st_final = getfield(tstate, :states)
    elseif hasfield(typeof(tstate), :state)
        st_final = getfield(tstate, :state)
    else
        st_final = nothing
    end

    # helper to call the chain in whichever calling convention is available
    call_chain = (c, x, p, s) -> p === nothing ? c(x) : c(x, p, s)

    # Evaluate the trained network on the asset grid (and shock grid if stochastic)
    a_grid = G[:a].grid
    # initialize locals so they're always defined even if an error occurs
    resid = nothing
    max_resid = 0.0
    c_return = nothing
    a_next = nothing
    if isnothing(S)
        # Deterministic: evaluate model on asset grid
        X_forward, a_grid_f32 = det_forward_inputs(G)
        _normalize_features!(X_forward, G, S)
        model_out = call_chain(chain_final, X_forward, ps_final, st_final)
        c_pred = model_out isa Tuple ? model_out[1] : model_out
        _, c_vec, c_vec_f32 = det_residual_inputs(c_pred, G)
        c_final = convert_to_grid_eltype(a_grid, c_vec)
        try
            resid = euler_resid_det_grid(P_resid, a_grid_f32, c_vec_f32)
        catch e
            rethrow(e)
        end
        max_resid = maximum(resid)
        # a_next computed from budget constraint
        R = hasfield(typeof(P), :r) ? 1 + getfield(P, :r) : 1.0
        # avoid using @. with getfield(P, :y) which would broadcast getfield over the NamedTuple P
        yval = hasfield(typeof(P), :y) ? getfield(P, :y) : 0.0
        a_next = R .* a_grid .+ yval .- c_final
        # clamp a_next if grid bounds exist
        try
            a_min = G[:a].min
            a_max = G[:a].max
            @. a_next = clamp(a_next, a_min, a_max)
        catch
        end
        c_return = c_final
    else
        # Stochastic: evaluate on full grid in canonical (a,z) order
        X_eval, _ = generate_dataset(G, S; mode = :full)
        _normalize_inputs!(X_eval, G, S)
        X_eval_proc = prepare_training_batch(X_eval)
        model_out = call_chain(chain_final, X_eval_proc, ps_final, st_final)
        c_pred = model_out isa Tuple ? model_out[1] : model_out
        a_grid_f32, z_grid_f32, Pz_f32, c_mat, c_mat_f32 =
            stoch_residual_inputs(c_pred, G, S)
        c_return = convert_to_grid_eltype(a_grid, c_mat)
        try
            resid =
                euler_resid_stoch_grid(P_resid, a_grid_f32, z_grid_f32, Pz_f32, c_mat_f32)
        catch e
            rethrow(e)
        end
        max_resid = maximum(abs.(resid))
        R = hasfield(typeof(P), :r) ? 1 + getfield(P, :r) : 1.0
        # a_next per (a,z)
        yval = hasfield(typeof(P), :y) ? getfield(P, :y) : 0.0
        a_next = R .* G[:a].grid .+ yval .- reshape(c_mat, :)
        try
            a_min = G[:a].min
            a_max = G[:a].max
            @. a_next = clamp(a_next, a_min, a_max)
        catch
        end
    end

    runtime = (time_ns() - start_time) / 1e9
    opts = (;
        epochs = epochs,
        batch = batch_size,
        lr = local_lr,
        seed = nothing,
        runtime = runtime,
        verbose = verbose,
        batches_per_epoch = batches_per_epoch,
    )

    iters = epochs
    converged = false

    return (;
        a_grid = a_grid,
        c = c_return,
        a_next = a_next,
        resid = resid,
        iters = iters,
        converged = converged,
        max_resid = max_resid,
        model_params = P,
        opts = opts,
    )
end


end # module
