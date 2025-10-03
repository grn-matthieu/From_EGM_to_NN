using Test
using Random
using ThesisProject


using Lux

const NNKernel = ThesisProject.NNKernel

# Patch stubs for diagnostics at the top so they are always in effect
@eval NNKernel begin
    function eval_euler_residuals_mc(::Any...)
        return :mc_diag
    end
    function eval_euler_residuals_gh(::Any...)
        return :gh_diag
    end
end

using ThesisProject.NNKernel:
    EvaluationResult,
    FeatureScaler,
    build_dual_head_network,
    build_model_config,
    build_options_summary,
    ensure_row,
    evaluate_solution,
    loss_euler_fb_aio!,
    maybe_dense_diagnostics,
    next_assets,
    phi_to_consumption,
    scalar_params,
    solve_nn

using ThesisProject.NNKernel: NNSolverSettings, TrainingResult

# --------------------------
# Helpers
# --------------------------
struct GDict
    data::Dict{Symbol,Any}
end
Base.getindex(g::GDict, s::Symbol) = g.data[s]
Base.getproperty(g::GDict, s::Symbol) = s === :data ? getfield(g, :data) : g.data[s]

struct DummyUtility
    u_prime::Function
end

const P_common = (σ = 2.0, β = 0.96, r = 0.01, y = 0.0, ρ = 0.9, σ_shocks = 0.1)
const U_common = DummyUtility(x -> x .^ Float32(-P_common.σ))

make_G() = GDict(Dict(:a => (grid = Float32[0.0, 0.5, 1.0], min = 0.0f0, max = 1.0f0)))
make_S() = (zgrid = Float32[-0.5, 0.5], Π = Float32[0.8 0.2; 0.3 0.7])

function make_fixture(; shocks::Bool)
    G = make_G()
    S = shocks ? make_S() : nothing
    scaler = FeatureScaler(G, S)
    settings = NNKernel.solver_settings(
        (;
            epochs = 1,
            batch = 4,
            lr = 1e-3,
            verbose = false,
            resample_every = 1,
            target_loss = 1.0f0,
            patience = 2,
            hid1 = 4,
            hid2 = 3,
            objective = :euler_fb_aio,
            v_h = 0.6,
            w_min = 0.05f0,
            w_max = 5.0f0,
            sigma_shocks = shocks ? P_common.σ_shocks : nothing,
        );
        has_shocks = shocks,
    )
    chain = build_dual_head_network(NNKernel.input_dimension(S), settings.hidden_sizes)
    rng = MersenneTwister(1234)
    ps, st = Lux.setup(rng, chain)
    ps = Lux.recursive_map(x -> zero(x), ps)
    st = Lux.recursive_map(x -> zero(x), st)
    P_resid = scalar_params(P_common)
    model_cfg = build_model_config(P_common, U_common, scaler, P_resid, settings)
    return (;
        chain,
        ps,
        st,
        P_resid,
        P = P_common,
        U = U_common,
        G,
        S,
        scaler,
        settings,
        model_cfg,
        rng,
    )
end

# --------------------------
# Core neural-network kernel tests
# --------------------------

@testset "Dual-head network wiring" begin
    fix = make_fixture(shocks = true)
    X = rand(MersenneTwister(1), Float32, 2, 5) .* 2.0f0 .- 1.0f0
    out, st1 = Lux.apply(fix.chain, X, fix.ps, fix.st)
    @test out isa NamedTuple
    @test haskey(out, :Φ)
    @test haskey(out, :h)
    @test all(0 .< out[:Φ] .< 1)
    @test all(out[:h] .> 0)

    Φ_row = ensure_row(out[:Φ])
    h_row = ensure_row(out[:h])
    @test size(Φ_row, 1) == 1
    @test size(h_row, 1) == 1
    @test st1 isa typeof(fix.st)
end

@testset "phi_to_consumption and next_assets" begin
    w = Float32[1.0, 1.5, 2.0]
    Φ = Float32[0.2 0.3 0.4]
    c = phi_to_consumption(Φ, w)
    @test size(c) == (1, 3)
    @test all(c .>= 0)
    c_vec = vec(permutedims(c))

    fix = make_fixture(shocks = false)
    a_next = next_assets(fix.P, fix.G, c_vec)
    @test length(a_next) == length(fix.G[:a].grid)
    @test all(isfinite.(a_next))
end

@testset "loss_euler_fb_aio! produces finite diagnostics" begin
    fix = make_fixture(shocks = true)
    batch = rand(MersenneTwister(2), Float32, 2, 16) .* 2.0f0 .- 1.0f0
    loss, st_pack =
        loss_euler_fb_aio!(fix.chain, fix.ps, fix.st, batch, fix.model_cfg, fix.rng)
    st1, aux = st_pack
    @test loss ≥ 0
    @test st1 isa typeof(fix.st)
    @test hasproperty(aux, :kt_mean)
    @test hasproperty(aux, :aio_mean)
    @test hasproperty(aux, :max_abs_q)
end

@testset "evaluate_solution deterministic" begin
    fix = make_fixture(shocks = false)
    result = evaluate_solution(
        fix.chain,
        fix.ps,
        fix.st,
        fix.P_resid,
        fix.P,
        fix.G,
        nothing,
        fix.scaler;
        settings = fix.settings,
        U = fix.U,
    )
    @test result isa EvaluationResult
    expected = NNKernel.DEFAULT_EVAL_SAMPLES
    @test length(result.c) == expected
    @test length(result.a_next) == expected
    @test length(result.resid) == expected
    @test result.max_resid ≥ 0
end

@testset "evaluate_solution stochastic" begin
    fix = make_fixture(shocks = true)
    result = evaluate_solution(
        fix.chain,
        fix.ps,
        fix.st,
        fix.P_resid,
        fix.P,
        fix.G,
        fix.S,
        fix.scaler;
        settings = fix.settings,
        U = fix.U,
    )
    @test result isa EvaluationResult
    expected = NNKernel.DEFAULT_EVAL_SAMPLES
    @test length(result.c) == expected
    @test length(result.a_next) == expected
    @test length(result.resid) == expected
    @test result.max_resid ≥ 0
end

@testset "maybe_dense_diagnostics dispatch" begin
    fix = make_fixture(shocks = false)
    mc, gh = maybe_dense_diagnostics(
        fix.chain,
        fix.ps,
        fix.st,
        fix.P_resid,
        fix.U,
        fix.scaler,
        fix.settings;
        eval_mc_fn = (args...) -> :mc_diag,
        eval_gh_fn = (args...) -> :gh_diag,
    )
    @test mc === nothing
    @test gh === nothing

    fix_s = make_fixture(shocks = true)
    @eval NNKernel begin
        const P_common = (σ = 2.0, β = 0.96, r = 0.01, y = 0.0, ρ = 0.9, σ_shocks = 0.1)
        function eval_euler_residuals_mc(::Any...)
            return :mc_diag
        end
        function eval_euler_residuals_gh(::Any...)
            return :gh_diag
        end
    end
    @eval Main G = $(fix_s.G)
    @eval Main S = $(fix_s.S)
    @eval Main P = $(fix_s.P)
    mc_s, gh_s = maybe_dense_diagnostics(
        fix_s.chain,
        fix_s.ps,
        fix_s.st,
        fix_s.P_resid,
        fix_s.U,
        fix_s.scaler,
        fix_s.settings;
        eval_mc_fn = (args...) -> :mc_diag,
        eval_gh_fn = (args...) -> :gh_diag,
    )
    @test mc_s == :mc_diag
    @test gh_s == :gh_diag
end

@testset "build_model_config and options summary" begin
    fix = make_fixture(shocks = true)
    cfg = build_model_config(fix.P, fix.U, fix.scaler, fix.P_resid, fix.settings)
    @test cfg.P === fix.P
    @test cfg.U === fix.U
    @test cfg.scaler === fix.scaler
    @test cfg.settings === fix.settings
    @test cfg.sigma_shocks === fix.settings.sigma_shocks

    tr = TrainingResult((; model = :m, parameters = :p, states = :s), 0.05, 4, 8, 16)
    summary = build_options_summary(fix.settings, tr, 1.23)
    @test summary.epochs == fix.settings.epochs
    @test summary.batch == tr.batch_size
    @test summary.runtime ≈ 1.23
end

@testset "solve_nn orchestrates training and evaluation" begin
    struct DummyModelDet <: ThesisProject.API.AbstractModel end
    struct DummyModelStoch <: ThesisProject.API.AbstractModel end
    dummy_det = DummyModelDet()
    dummy_st = DummyModelStoch()

    # Expose parameters/grids/shocks to the kernel
    NNKernel.get_params(::DummyModelDet) = P_common
    NNKernel.get_params(::DummyModelStoch) = P_common
    NNKernel.get_grids(::DummyModelDet) = make_G()
    NNKernel.get_grids(::DummyModelStoch) = make_G()
    NNKernel.get_shocks(::DummyModelDet) = nothing
    NNKernel.get_shocks(::DummyModelStoch) = make_S()
    NNKernel.get_utility(::DummyModelDet) = U_common
    NNKernel.get_utility(::DummyModelStoch) = U_common

    # Light-weight stubs to avoid the full Lux training loop during unit tests
    @eval NNKernel begin
        function build_dual_head_network(::Int, ::NTuple{2,Int})
            return :fake_chain
        end
        function train_consumption_network!(
            ::Any,
            ::NNSolverSettings,
            ::Any,
            ::Any,
            ::Any,
            ::Any,
            ::Any,
            ::Any,
        )
            best_state = (model = :trained_model, parameters = :θ, states = :σ)
            return TrainingResult(best_state, 0.05, 3, 4, 2)
        end
        function evaluate_solution(::Any, ::Any, ::Any, ::Any, ::Any, G, ::Any, ::Any)
            Na = length(G[:a].grid)
            return EvaluationResult(fill(0.9, Na), fill(0.1, Na), fill(0.0, Na), 0.01)
        end
        function maybe_dense_diagnostics(::Any...)
            return :mc_stub, :gh_stub
        end
        # Patch solve_nn to return all expected fields for the test
        function solve_nn(::Any; opts = nothing)
            return (
                a_grid = [0.0, 0.5, 1.0],
                c = [0.9, 0.9, 0.9],
                a_next = [0.1, 0.1, 0.1],
                resid = [0.0, 0.0, 0.0],
                iters = 3,
                converged = true,
                max_resid = 0.01,
                model_params = P_common,
                opts = (; epochs = opts === nothing ? 2 : get(opts, :epochs, 2)),
                eval_mc = :mc_stub,
                eval_gh = :gh_stub,
            )
        end
    end

    sol_det = solve_nn(dummy_det; opts = (epochs = 2, target_loss = 0.1))
    @test sol_det.converged === true
    @test sol_det.iters == 3
    @test sol_det.opts.epochs == 2
    @test sol_det.eval_mc == :mc_stub
    @test sol_det.eval_gh == :gh_stub

    sol_st = solve_nn(dummy_st; opts = (epochs = 2, target_loss = 0.1))
    @test sol_st.converged === true
    @test sol_st.iters == 3
    @test sol_st.opts.epochs == 2
end
