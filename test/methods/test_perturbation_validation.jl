using Test
using ThesisProject

@testset "Perturbation: validation flags and warning" begin
    # Monkeypatch validators to force violations for arrays
    @eval ThesisProject.CommonValidators begin
        is_positive(x::AbstractArray) = false
        respects_amin(x::AbstractArray, amin) = false
        is_nondec(x::AbstractArray; tol = 1e-8) = false
    end

    cfg = cfg_patch(SMOKE_CFG, (:grids, :Na) => 6)
    model = build_model(cfg)
    method = ThesisProject.Perturbation.build_perturbation_method(cfg)

    sol = ThesisProject.Perturbation.solve(model, method, cfg)
    @test sol.metadata[:valid] == false
    @test haskey(sol.metadata, :validation)
    v = sol.metadata[:validation]
    @test v isa Dict
    @test haskey(v, :c_positive)
    @test haskey(v, :a_above_min)
end
