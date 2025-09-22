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
    epochs = 1000
    start_time = time_ns()

    # Build Lux chain and optimizer
    chain = _make_chain(input_dim; hid1 = get(opts, :hid1, 64), hid2 = get(opts, :hid2, 32))
    # Reduce default learning rate for stability (can be overridden via opts)
    local_lr = opts === nothing ? 1e-4 : get(opts, :lr, 1e-4)
    opt = Optimisers.Adam(local_lr)

    ps, st = Lux.setup(Random.GLOBAL_RNG, chain)

    tstate = Lux.Training.TrainState(chain, ps, st, opt)

    vjp_rule = Lux.AutoZygote()

    # Generate dataset and preprocess inputs once: convert X to Float32 and transpose
    X0, y0 = generate_dataset(G, S)
    # make a concrete Float32 array shaped (features, batch)
    X_proc = Array{Float32}(permutedims(X0))  # (features, batch)
    data = (X_proc, y0)

    # Debug: run a single forward pass to inspect model outputs and types
    try
        mo = chain(X_proc, ps, st)
        @printf "DEBUG: model forward pass returned type = %s\n" string(typeof(mo))
        if mo isa Tuple
            @printf "DEBUG: model tuple element types: (%s, %s)\n" string(typeof(mo[1])) string(
                typeof(mo[2]),
            )
        end
        # Attempt to extract predicted c for inspection
        if mo isa Tuple
            c_test = mo[1]
        else
            c_test = mo
        end
        @printf "DEBUG: c_test eltype = %s, size = %s\n" string(eltype(c_test)) string(
            size(c_test),
        )
    catch e
        @printf "DEBUG: forward pass error: %s\n" string(e)
    end

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

    # Debug: inspect params and grids before training (non-AD context)
    @printf "DEBUG PRETRAIN: typeof(P)=%s, propertynames(P)=%s\n" string(typeof(P)) string(
        propertynames(P),
    )
    @printf "DEBUG PRETRAIN: P has y field? %s, P_resid.y=%s\n" string(
        :y in propertynames(P),
    ) string(getfield(P_resid, :y))
    @printf "DEBUG PRETRAIN: a grid eltype=%s len=%d\n" string(eltype(G[:a].grid)) length(
        G[:a].grid,
    )
    if !isnothing(S)
        @printf "DEBUG PRETRAIN: z grid eltype=%s len=%d, Π eltype=%s size=%s\n" string(
            eltype(S.zgrid),
        ) length(S.zgrid) string(eltype(S.Π)) string(size(S.Π))
    end

    # Custom loss function using Euler residuals, capturing P_resid, G, S
    loss_function =
        (model, ps, st, data) -> begin
            # data[1] is preprocessed X shaped (features, batch)
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
                # Deterministic case - convert grids to Float32
                # ensure concrete Float32 vectors for residual evaluation
                a_grid_f32 = Vector{Float32}(collect(G[:a].grid))
                # c_predicted has shape (output_dim, batch) -> (1, Na)
                c_pred_vec = vec(permutedims(c_predicted))
                c_pred_vec_f32 = Vector{Float32}(collect(c_pred_vec))
                resid = euler_resid_det_grid(P_resid, a_grid_f32, c_pred_vec_f32)
                loss = Float32(sum(resid))  # Convert loss to Float32
            else
                # Stochastic case - convert grids to Float32
                # ensure arrays are plain vectors/matrices before broadcasting
                # ensure concrete Float32 arrays/matrices for residual evaluation
                a_grid_f32 = Vector{Float32}(collect(G[:a].grid))
                z_grid_f32 = Vector{Float32}(collect(S.zgrid))
                Pz_f32 = Array{Float32}(collect(S.Π))
                Na = length(a_grid_f32)
                Nz = length(z_grid_f32)
                # c_predicted has shape (output_dim, Na*Nz) -> reshape to (Na, Nz)
                c_pred = reshape(vec(permutedims(c_predicted)), Na, Nz)
                c_pred_f32 = Array{Float32}(collect(c_pred))
                resid = euler_resid_stoch_grid(
                    P_resid,
                    a_grid_f32,
                    z_grid_f32,
                    Pz_f32,
                    c_pred,
                )
                loss = Float32(sum(abs2, resid))  # Convert loss to Float32
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

    for epoch = 1:epochs
        # Grad-norm diagnostics: attempt to extract gradient container from Lux
        gnorm = NaN
        ginfo, loss, _, tstate =
            Lux.Training.single_train_step!(vjp_rule, loss_function, data, tstate)
        # try to compute a robust global gradient norm (RMS-style)
        try
            ssum = flatten_sum_squares(ginfo)
            if ssum > 0.0
                gnorm = sqrt(ssum)
            else
                gnorm = 0.0
            end
        catch
            gnorm = NaN
        end
        if epoch % 1 == 0 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g \t GradNorm: %.5g\n" epoch loss gnorm
        end
    end

    # After training, build a solution-like NamedTuple so callers (methods) get a consistent result
    # Inspect TrainState fields at runtime to determine where parameters are stored
    @printf "DEBUG NN: TrainState type = %s\n" string(typeof(tstate))
    @printf "DEBUG NN: TrainState fieldnames = %s\n" string(fieldnames(typeof(tstate)))
    @printf "DEBUG NN: tstate summary = %s\n" string(tstate)
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
        a_grid_f32 = Vector{Float32}(collect(a_grid))
        X_forward = reshape(a_grid_f32, 1, :)  # (features, batch)
        model_out = call_chain(chain_final, X_forward, ps_final, st_final)
        c_pred = model_out isa Tuple ? model_out[1] : model_out
        c_vec = vec(permutedims(c_pred))
        # convert predicted c back to grid element type (e.g., Float64)
        c_final = convert.(eltype(a_grid), c_vec)
        # compute residuals and diagnostics (use Float32 inputs for residuals)
        try
            resid = euler_resid_det_grid(
                P_resid,
                Vector{Float32}(collect(a_grid)),
                Vector{Float32}(collect(c_vec)),
            )
        catch e
            @printf("DEBUG RESID ERROR (det): %s\n", e)
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
        # stochastic: evaluate on full (a,z) grid ordering assumed to match data generation
        a_grid_f32 = Vector{Float32}(collect(G[:a].grid))
        z_grid_f32 = Vector{Float32}(collect(S.zgrid))
        # transition matrix for z (probabilities) used by residuals
        Pz_f32 = Array{Float32}(collect(S.Π))
        Na = length(a_grid_f32)
        Nz = length(z_grid_f32)
        # Evaluate on training input ordering (data[1]) to get predicted c for all (a,z)
        model_out = call_chain(chain_final, data[1], ps_final, st_final)
        c_pred = model_out isa Tuple ? model_out[1] : model_out
        c_mat = reshape(vec(permutedims(c_pred)), Na, Nz)
        # keep c as matrix (Na x Nz) using the native grid element type
        c_return = convert.(eltype(a_grid), c_mat)
        try
            resid = euler_resid_stoch_grid(
                P_resid,
                a_grid_f32,
                z_grid_f32,
                Pz_f32,
                Array{Float32}(collect(c_mat)),
            )
        catch e
            @printf("DEBUG RESID ERROR (stoch): %s\n", e)
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
    opts = (; epochs = epochs, lr = local_lr, seed = nothing, runtime = runtime)

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
