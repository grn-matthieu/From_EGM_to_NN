using ThesisProject
cfg = load_config("config/smoke_config/smoke_config_stochastic.yaml")
model = build_model(cfg)
method = build_method(cfg)
sol = solve(model, method, cfg)
println(size(sol.policy[:c].value))
println(sol.metadata[:converged])
println(sol.metadata[:iters])
