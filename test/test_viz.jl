@testset "viz" begin
    cfg_path = joinpath("@__DIR__", "..", "config", "smoke_config", "smoke_config.yaml")
    cfg = load_config(cfg_path)
    model = build_model(cfg)
    method = build_method(cfg)
    sol = solve(model, method, cfg)
    @test begin
        plt = plot_policy(sol)
        isa(plt, Plots.Plot)
    end
end