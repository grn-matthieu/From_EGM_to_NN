using Test
using ThesisProject

@testset "Shocks validation succeeds" begin
    cfg = Dict(:ρ_shock => 0.9, :σ_shock => 0.1, :Nz => 5, :validate => true)
    out = ThesisProject.Shocks.discretize(cfg)
    @test out isa ThesisProject.Shocks.ShockOutput
end
@testset "Shocks validation failure" begin
    cfg = Dict(:ρ_shock => 0.9, :σ_shock => 0.1, :Nz => 3, :validate => true)

    @eval ThesisProject.Shocks begin
        function find_invariant(Π::AbstractMatrix{<:Real}; tol = 1e-12, maxit = 1_000)
            fill(1.0, size(Π, 1))
        end
    end

    @test_throws ErrorException ThesisProject.Shocks.discretize(cfg)

    @eval ThesisProject.Shocks begin
        function find_invariant(P::AbstractMatrix{<:Real}; tol = 1e-12, maxit = 1_000)
            dist = fill(1.0 / size(P, 1), size(P, 1))
            for _ = 1:maxit
                dist_new = dist' * P
                dist_new ./= sum(dist_new)
                if maximum(abs.(dist_new .- dist')) < tol
                    return vec(dist_new)
                end
                dist .= vec(dist_new)
            end
            error("Power iteration did not converge")
        end
    end
end

