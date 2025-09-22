using ThesisProject

cfg = ThesisProject.API.load_config("config/nn_deterministic.yaml")
model = ThesisProject.API.build_model(cfg)
method = ThesisProject.API.build_method(cfg)
sol = ThesisProject.API.solve(model, method, cfg)
println("Solved model with NN method.")
