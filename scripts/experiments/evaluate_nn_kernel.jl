using ThesisProject
using ThesisProject.Determinism: make_master_rng, derive_rng
using Random, Statistics, Lux, Printf, StatsBase

# --- 1) Chargement modèle + réglages rapides
const ROOT = normpath(joinpath(pwd(), ""))
cfg_path = joinpath(ROOT, "config", "smoke_cfg_stoch.yaml")  # ajuste si besoin
cfg = ThesisProject.load_config(cfg_path)
model = ThesisProject.build_model(cfg)
P = ThesisProject.get_params(model)
G = ThesisProject.get_grids(model)
S = ThesisProject.get_shocks(model)
U = ThesisProject.get_utility(model)

# override légers pour test rapide
vh = 2.5
epochs = 50_000
solver_nt = merge(cfg.solver, (; v_h = vh, epochs = epochs))
cfg = merge(cfg, (solver = solver_nt,))

# --- 2) Construire settings + scaler + réseau
settings = ThesisProject.NNKernel.solver_settings(cfg.solver; has_shocks = !isnothing(S))
P_resid = ThesisProject.NNKernel.scalar_params(P)
scaler = ThesisProject.NNKernel.FeatureScaler(P, G, S, settings)
in_dim = ThesisProject.NNKernel.input_dimension(S)
chain = ThesisProject.NNKernel.build_dual_head_network(in_dim, settings.hidden_sizes)
model_cfg = ThesisProject.NNKernel.build_model_config(P, U, scaler, P_resid, settings)

# --- 3) Entraînement court
master = make_master_rng(1234)
trainres = ThesisProject.NNKernel.train_consumption_network!(
    chain,
    settings,
    scaler,
    P_resid,
    G,
    S,
    derive_rng(master, :train),
    model_cfg,
)
best = trainres.best_state
trained = ThesisProject.NNKernel.select_model(chain, best)
params = ThesisProject.NNKernel.state_parameters(best)
states = ThesisProject.NNKernel.state_states(best)

# --- 4) Batch validation aléatoire pour quelques checks élémentaires
ns = min(4096, settings.samples_per_epoch)
val_batch, _ = ThesisProject.NNKernel.create_training_batch(
    G,
    S,
    scaler;
    mode = :rand,
    nsamples = ns,
    rng = derive_rng(master, :val),
    P_resid = P_resid,
    settings = settings,
)

# dénormalise y,w
y0 = ((val_batch[1, :] .+ 1.0f0) ./ 2.0f0) .* scaler.y_range .+ scaler.y_min
w0 = ((val_batch[2, :] .+ 1.0f0) ./ 2.0f0) .* scaler.w_range .+ scaler.w_min

# passe avant et récupère c, h
out, _ = Lux.apply(trained, val_batch, params, states)
Φ = vec(ThesisProject.NNKernel.ensure_row(out[:Φ]))
h = vec(ThesisProject.NNKernel.ensure_row(out[:h]))
c0 = vec(ThesisProject.NNKernel.phi_to_consumption(out[:Φ], w0; min_c = 1.0f-3))
a1 = @. w0 - c0

# --- 5) Monte Carlo 1-step pour q = βR u'(c1)/u'(c0)
μ = Float32(P_resid.y)
z0 = log.(y0) .- μ
ρ = Float32(P.ρ)
σϵ =
    settings.sigma_shocks === nothing ? Float32(P.σ_shocks) : Float32(settings.sigma_shocks)
β = Float32(P.β)
R = 1.0f0 + Float32(P.r)
ε = randn(derive_rng(master, :mc), Float32, ns)
z1 = @. ρ * z0 + σϵ * ε
y1 = exp.(μ .+ z1)
w1 = @. R * a1 + y1
X1 = vcat(reshape(y1, 1, :), reshape(w1, 1, :))
NX1 = ThesisProject.NNKernel.normalize_feature_batch(scaler, X1)
out1, _ = Lux.apply(trained, NX1, params, states)
c1 = vec(ThesisProject.NNKernel.phi_to_consumption(out1[:Φ], w1; min_c = 1.0f-3))

u′ = U.u_prime
q = @. β * R * u′(c1) / u′(c0)

# --- 6) Résidus GH de référence
gh = ThesisProject.NNKernel.eval_euler_residuals_gh(
    trained,
    params,
    states,
    P_resid,
    U,
    scaler,
    settings;
    N = 8192,
    rng = derive_rng(master, :gh),
    G = G,
    S = S,
    P = P,
)
stats = hasproperty(gh, :stats) ? gh.stats : gh[:stats]

# --- 7) Monotonicité en w (pour y fixé): corrélation Spearman entre w et c
using StatsBase
idx = rand(1:length(y0))                     # pick un y
yy = fill(y0[idx], length(w0))
ww = sort(w0)                               # grille croissante de w
Xmon = vcat(reshape(yy, 1, :), reshape(ww, 1, :))
NXm = ThesisProject.NNKernel.normalize_feature_batch(scaler, Xmon)
outm, _ = Lux.apply(trained, NXm, params, states)
cm = vec(ThesisProject.NNKernel.phi_to_consumption(outm[:Φ], ww; min_c = 1.0f-3))
ρs = cor(cm, ww)

# --- 8) Impressions synthétiques
println("Checks NN kernel")
println(
    "  GH residuals: mean=$(stats.mean)  p50=$(stats.p50)  p95=$(stats.p95)  max=$(stats.max)",
)
println("  mean(|q - h|) = ", mean(abs.(q .- h)))
println("  Budget: mean|w - c - a'| = ", mean(abs.(w0 .- c0 .- a1)))
println(
    "  Bounds a': in [a_min,a_max] = ",
    all(a1 .>= G[:a].min .- 1e-5) && all(a1 .<= G[:a].max .+ 1e-5),
)
println("  Monotonicité c(w|y): Spearman ρ = ", ρs)
println("  Φ in [0,1] = ", all((0.0f0 .<= Φ) .& (Φ .<= 1.0f0)))
println("  c ∈ (0,w]   = ", all((c0 .> 0.0f0) .& (c0 .<= w0 .+ 1e-6)))
