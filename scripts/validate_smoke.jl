using ThesisProject

cfg = ThesisProject.API.load_config("config/smoke_config/smoke_config_stochastic.yaml")
ThesisProject.API.validate_config(cfg)
println("validated")
