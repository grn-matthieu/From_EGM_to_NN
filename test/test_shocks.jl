using Test
using ThesisProject

@testset "Shocks validation succeeds" begin
    cfg = Dict(:ρ_shock => 0.9,
               :σ_shock => 0.1,
               :Nz => 5,
               :validate => true)
    out = ThesisProject.Shocks.discretize(cfg)
    @test out isa ThesisProject.Shocks.ShockOutput
end

@testset "Shocks validation failure" begin
    cfg = Dict(:ρ_shock => 0.9,
               :σ_shock => 0.1,
               :Nz => 3,
               :validate => true)

    @eval ThesisProject.Shocks begin
        function find_invariant(Π::AbstractMatrix{<:Real}; tol=1e-12, maxit=1_000)
            fill(1.0, size(Π, 1))
        end
    end

    @test_throws ErrorException ThesisProject.Shocks.discretize(cfg)

    mt = methods(ThesisProject.Shocks.find_invariant)
    m = findfirst(m -> m.sig == Tuple{typeof(ThesisProject.Shocks.find_invariant),AbstractMatrix{Float64}}, mt)
    Base.delete_method(m)
end

