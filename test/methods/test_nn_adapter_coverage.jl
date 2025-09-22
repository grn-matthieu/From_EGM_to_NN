using Test
using ThesisProject

@testset "NN adapter coverage" begin
    # Build a minimal cfg and model
    cfg = cfg_patch(SMOKE_CFG, (:solver, :method) => :NN)
    model = build_model(cfg)

    # Stub the binding used inside the NN adapter so the adapter's code is exercised.
    # NN.jl imports `solve_nn` and `compute_value_policy` into its module namespace,
    # so replace the names on `ThesisProject.NN` rather than on `ThesisProject.NNKernel`.
    # Save the original functions from the NN module and override them by eval-ing new
    # definitions into the `ThesisProject.NN` module. This avoids errors when trying to
    # assign to imported bindings.
    # Override the implementation in NNKernel (where solve_nn is defined) so the
    # NN adapter (which imports solve_nn) will call this stub implementation.
    orig_nn_solve = getfield(ThesisProject.NNKernel, :solve_nn)
    Core.eval(
        ThesisProject.NNKernel,
        quote
            function solve_nn(m; opts = nothing)
                a_grid = get_grids(m).a.grid
                c = clamp.(0.5 .* (1 .+ a_grid), 1e-12, Inf)
                a_next = a_grid
                resid = zeros(length(a_grid))
                return (;
                    a_grid = a_grid,
                    c = c,
                    a_next = a_next,
                    resid = resid,
                    iters = 1,
                    converged = true,
                    max_resid = 0.0,
                    model_params = get_params(m),
                    opts = (; epochs = 1, lr = 1e-3, seed = 0, runtime = 0.0),
                )
            end
        end,
    )


    method = build_method(cfg)
    @test method isa ThesisProject.AbstractMethod
    sol = ThesisProject.solve(model, method, cfg)
    @test cfg_has(sol.policy, :c) && !isnothing(cfg_get(sol.policy, :c).value)
    @test sol.diagnostics.method == :NN
    @test sol.metadata[:converged] == true || haskey(sol.metadata, :max_it)
end
