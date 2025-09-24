using Random, Lux
using Statistics: mean, std, median
using ThesisProject
const NN = ThesisProject.NNKernel

input_dim = 2
H1 = 32
H2 = 32
chain = Chain(
    Dense(input_dim, H1, leakyrelu),
    Dense(H1, H2, leakyrelu),
    Dense(H2, 2),
    x -> begin
        pre = x
        φ_pre = view(pre, 1:1, :)
        h_pre = view(pre, 2:2, :)
        (Φ = sigmoid.(φ_pre), h = exp.(h_pre))
    end,
)

rng = MersenneTwister(1234)
ps, st = Lux.setup(rng, chain)
N = 2048
# sample assets and shocks from typical ranges
a = rand(rng, Float32, N) .* 20.0f0
z = randn(rng, Float32, N) .* 0.1f0
# batch shape: features×samples => 2 × N
batch = vcat(reshape(a, 1, :), reshape(z, 1, :))
println("batch eltype=", eltype(batch), " size=", size(batch))
try
    out_test = Lux.apply(chain, batch, ps, st)
    println("Lux.apply ok, output typeof=", typeof(out_test))
    # Lux.apply may return (output, state). Inspect first element which should be the NamedTuple with Φ and h
    if isa(out_test, Tuple) && length(out_test) >= 1
        out0 = out_test[1]
        println("first element typeof=", typeof(out0))
        if isa(out0, NamedTuple)
            println("out0 keys=", keys(out0))
            try
                println(
                    "Φ size=",
                    size(out0.Φ),
                    " h size=",
                    size(out0.h),
                    " eltype=",
                    eltype(out0.Φ),
                )
            catch
                println("couldn't get sizes of out0 fields")
            end
        end
    end
catch e
    println("Lux.apply failed: ", e)
    rethrow()
end

P = (; r = 0.02, β = 0.96, ρ = 0.9, σ = 2.0, y = 0.1, σ_shocks = 0.1)
U = (u_prime = (c -> c .^ (-P.σ)), u_prime_inv = (x -> x .^ (-1 / P.σ)), σ = P.σ)

# --- Train for a few epochs (10) to stabilize predictions, then recompute v_h ---
println("Starting short training (10 epochs) to stabilize the network...")

# Small grid and shock specs for dataset generation and scaler
struct GDict
    data::Dict{Symbol,Any}
end
Base.getindex(g::GDict, s::Symbol) = g.data[s]
Base.getproperty(g::GDict, s::Symbol) = s === :data ? getfield(g, :data) : g.data[s]

G = GDict(
    Dict(
        :a => (
            grid = Float32.(range(0.0f0, stop = 2.0f0, length = 64)),
            min = 0.0f0,
            max = 2.0f0,
        ),
    ),
)
S = (; zgrid = Float32.([-0.1f0, 0.0f0, 0.1f0]), Π = ones(3, 3) / 3)

scaler = NN.FeatureScaler(G, S)
P_resid = NN.scalar_params(P)

# solver settings: full-run defaults for stability (lower lr, conservative v_h)
opts = (;
    epochs = 100,
    objective = :euler_fb_aio,
    lr = 1e-4,
    v_h = 0.5,
    verbose = true,
    resample_every = 1,
)
settings = NN.solver_settings(opts; has_shocks = scaler.has_shocks)

model_cfg =
    (P = P, U = U, v_h = 1.0, scaler = scaler, P_resid = P_resid, settings = settings)

training_result =
    NN.train_consumption_network!(chain, settings, scaler, P_resid, G, S, model_cfg, rng)
println(
    "Training done. best_loss=",
    training_result.best_loss,
    " epochs_run=",
    training_result.epochs_run,
)

# Extract trained parameters/states from training result
best_state = training_result.best_state
ps_trained = NN.state_parameters(best_state)
st_trained = NN.state_states(best_state)

# Build a filtered validation minibatch using the same sampler the trainer uses (enforces w_min..w_max)
val_batch, _ = NN.create_training_batch(
    G,
    S,
    scaler;
    mode = :rand,
    nsamples = 4096,
    rng = rng,
    P_resid = P_resid,
    settings = settings,
)
# use the filtered validation batch for diagnostics and further checks
batch = val_batch
loss, (st1, aux) =
    NN.loss_euler_fb_aio!(chain, ps_trained, st_trained, val_batch, model_cfg, rng)
println("post-training loss=", loss)
println("post-training kt_mean=", aux.kt_mean)
println("post-training aio_mean=", aux.aio_mean)
vh_target = aux.kt_mean / aux.aio_mean
vh_clamped = min(max(vh_target, 0.5), 2.0)
println("suggested v_h (mean-based)=", vh_target, " clamped=", vh_clamped)

# --- Diagnostics: recompute per-sample pieces to inspect distributions ---
println("--- Diagnostics (per-sample summaries) ---")
T = eltype(batch)
# unpack P and utility
P = model_cfg.P
uprime = model_cfg.U.u_prime

# helper: ensure we always return a vector of length N
function ensure_vector(x, N, T)
    if isa(x, Number)
        return fill(convert(T, x), N)
    elseif isa(x, AbstractArray)
        # try to get as vector
        if ndims(x) == 1
            return convert(Vector{T}, x)
        elseif ndims(x) == 2 && size(x, 1) == 1
            return vec(convert(Matrix{T}, x))
        else
            return vec(convert(Array{T,1}, x))
        end
    else
        return fill(zero(T), N)
    end
end

# forward pass to get Φ and h (unnormalized inputs)
out, _ = Lux.apply(chain, batch, ps, st)
N = size(batch, 2)
ϕ = ensure_vector(out[:Φ], N, T)
h = ensure_vector(out[:h], N, T)

# unnormalize inputs (a0, z0) from normalized features×samples batch
a0 = ((batch[1, :] .+ one(T)) ./ T(2)) .* T(scaler.a_range) .+ T(scaler.a_min)
z0 =
    settings.has_shocks ?
    ((batch[2, :] .+ one(T)) ./ T(2)) .* T(scaler.z_range) .+ T(scaler.z_min) :
    fill(zero(T), size(a0))

# compute w0 and c0 using the same helper as the loss
w0 = NN.cash_on_hand(a0, z0, P_resid, true)
w0 = T.(w0)
@show minimum(w0), maximum(w0)
c0 = @. clamp(ϕ .* w0, eps(T), T(Inf))
a_term = @. one(T) - c0 / w0

# draw shocks and next-period states
rho = T(P.ρ)
sigma_shocks = T(P.σ_shocks)
ε1 = randn!(rng, similar(z0));
ε2 = randn!(rng, similar(z0));
z1 = @. rho * z0 + sigma_shocks * ε1
z2 = @. rho * z0 + sigma_shocks * ε2

# next assets (a' = w0 - c0) and same a' for both draws
a1 = @. w0 - c0
a2 = a1

# Build next-step inputs (a', z') and NORMALIZE them as the loss does
X1 = vcat(reshape(a1, 1, :), reshape(z1, 1, :))
X2 = vcat(reshape(a2, 1, :), reshape(z2, 1, :))
NX1 = NN.normalize_feature_batch(scaler, X1)
NX2 = NN.normalize_feature_batch(scaler, X2)

out1, _ = Lux.apply(chain, NX1, ps, st1)
out2, _ = Lux.apply(chain, NX2, ps, st1)
Φ1 = ensure_vector(out1[:Φ], N, T)
Φ2 = ensure_vector(out2[:Φ], N, T)

# compute next-period cash-on-hand and consumption using same helper
w1 = NN.cash_on_hand(a1, z1, P_resid, true)
w2 = NN.cash_on_hand(a2, z2, P_resid, true)
w1 = T.(w1);
w2 = T.(w2);
c1 = @. clamp(Φ1 .* w1, eps(T), T(Inf))
c2 = @. clamp(Φ2 .* w2, eps(T), T(Inf))

β = T(P.β)
R = T(P.r)
# compute raw q but also a clipped version replacing tiny consumptions with a small floor to avoid astronomic uprime
min_c = eps(T)
q1_raw = @. β * (one(T) + R) * uprime(c1) / uprime(c0)
q2_raw = @. β * (one(T) + R) * uprime(c2) / uprime(c0)
q1 = @. β * (one(T) + R) * uprime(clamp(c1, min_c, T(Inf))) /
   uprime(clamp(c0, min_c, T(Inf)))
q2 = @. β * (one(T) + R) * uprime(clamp(c2, min_c, T(Inf))) /
   uprime(clamp(c0, min_c, T(Inf)))

fb_term = @. (a_term + (1 - h) - sqrt(a_term^2 + (1 - h)^2))
kt = @. fb_term^2
gh1 = @. q1 - h
gh2 = @. q2 - h
aio = @. gh1 .* gh2

function summary_stats(x)
    xs = collect(x)
    s = sort(xs)
    n = length(s)
    p25 = s[max(1, Int(floor(0.25 * n)))]
    med = s[Int(clamp(round(n / 2), 1, n))]
    p75 = s[min(n, Int(ceil(0.75 * n)))]
    return (
        min = minimum(xs),
        p25 = p25,
        median = med,
        mean = mean(xs),
        p75 = p75,
        max = maximum(xs),
        std = std(xs),
    )
end

showstats = Dict(
    ":ϕ" => summary_stats(ϕ),
    ":h" => summary_stats(h),
    ":w0" => summary_stats(w0),
    ":c0" => summary_stats(c0),
    ":q1_raw" => summary_stats(q1_raw),
    ":q2_raw" => summary_stats(q2_raw),
    ":q1_clipped" => summary_stats(q1),
    ":q2_clipped" => summary_stats(q2),
    ":kt" => summary_stats(kt),
    ":aio" => summary_stats(aio),
)

for (k, v) in showstats
    println(k, ": ", v)
end

println("--- end diagnostics ---")

# median-based estimator (robust to outliers)
kt_med = median(kt)
aio_med = median(aio)
vh_med = kt_med / aio_med
vh_med_clamped = min(max(vh_med, 0.5), 2.0)
println(
    "median kt=",
    kt_med,
    " median aio=",
    aio_med,
    " median-based v_h=",
    vh_med,
    " clamped=",
    vh_med_clamped,
)
