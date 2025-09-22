using ThesisProject

cfg = ThesisProject.API.load_config("config/nn_deterministic.yaml")
model_cfg = hasproperty(cfg, :model) ? cfg.model : cfg[:model]
solver_cfg = hasproperty(cfg, :solver) ? cfg.solver : cfg[:solver]
model_name = hasproperty(model_cfg, :name) ? model_cfg.name : model_cfg[:name]
method_name = hasproperty(solver_cfg, :method) ? solver_cfg.method : solver_cfg[:method]
@info "Loaded configuration" model = model_name method = method_name
model = ThesisProject.API.build_model(cfg)
method = ThesisProject.API.build_method(cfg)
sol = ThesisProject.API.solve(model, method, cfg)
println("Solved model with NN method.")
