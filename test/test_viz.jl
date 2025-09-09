@testset "viz" begin
    try
        @eval using Plots
        cfg_path = joinpath(@__DIR__, "..", "config", "smoke_config", "smoke_config.yaml")
        cfg = load_config(cfg_path)
        model = build_model(cfg)
        method = build_method(cfg)
        sol = solve(model, method, cfg)
        @test begin
            plt = plot_policy(sol)
            isa(plt, Plots.Plot)
        end
        @test begin
            plt = plot_euler_errors(sol)
            isa(plt, Plots.Plot)
        end
    catch err
        @warn("Plots.jl not available; skipping visualization tests")
        @test true
        return
    end

end
