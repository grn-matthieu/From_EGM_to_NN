using Test
using ThesisProject
using JSON3

function _nn_eval_fixture()
    cfg_path = joinpath(@__DIR__, "..", "..", "config", "simple_baseline.yaml")
    cfg = ThesisProject.load_config(cfg_path)
    cfg[:solver][:method] = :NN
    model = ThesisProject.build_model(cfg)
    params = ThesisProject.get_params(model)
    grids = ThesisProject.get_grids(model)
    util = ThesisProject.get_utility(model)
    sol = ThesisProject.NNKernel.solve_nn_det(params, grids, util, cfg)
    return cfg, sol
end

@testset "NNEval metrics and report" begin
    cfg, sol = _nn_eval_fixture()
    report = ThesisProject.NNEval.eval_nn(sol, cfg; compare_egm = false)
    @test report.metrics.max_euler_error >= 0
    @test report.metrics.mean_euler_error <= report.metrics.max_euler_error + 1e-12
    @test report.metadata.grid_size == length(sol.resid)
    @test report.diagnostics.iters == sol.iters

    tmp = mktempdir()
    out_path = ThesisProject.NNEval.write_eval_report(
        sol,
        cfg;
        output_dir = tmp,
        filename = "nn_eval.json",
        compare_egm = false,
    )
    @test isfile(out_path)
    parsed = JSON3.read(read(out_path, String))
    @test haskey(parsed, "metrics")
    @test parsed["metrics"]["max_euler_error"] >= 0
end

@testset "NNEval baseline comparison" begin
    cfg, sol = _nn_eval_fixture()
    report = ThesisProject.NNEval.eval_nn(sol, cfg; compare_egm = true)
    @test report.baseline !== nothing
    @test report.baseline.method == "EGM"
    @test (report.baseline.consumption !== nothing) ||
          (report.baseline.assets_next !== nothing)
end
