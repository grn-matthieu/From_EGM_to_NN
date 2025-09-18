@testset "viz" begin
    has_plots = try
        @eval using Plots
        true
    catch
        @info "Plots.jl not available; skipping visualization tests"
        false
    end

    if has_plots
        cfg = deepcopy(SMOKE_CFG)
        model = build_model(cfg)
        method = build_method(cfg)
        sol = solve(model, method, cfg)

        @testset "plot_policy" begin
            plt = plot_policy(sol)
            @test isa(plt, Plots.Plot)
        end

        @testset "plot_euler_errors" begin
            plt = plot_euler_errors(sol)
            @test isa(plt, Plots.Plot)
        end
    else
        @test true   # keep testset non-empty
    end
end
