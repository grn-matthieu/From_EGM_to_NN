using Test
using ThesisProject.Chebyshev:
    chebyshev_nodes,
    chebyshev_basis,
    scale_to_chebyshev,
    scale_from_chebyshev,
    gauss_lobatto_nodes

@testset "scaling" begin
    a, b = 0.0, 2.0
    x = [0.0, 1.0, 2.0]
    ξ = scale_to_chebyshev.(x, a, b)
    @test ξ == [-1.0, 0.0, 1.0]
    @test scale_from_chebyshev.(ξ, a, b) == x
end

@testset "nodes" begin
    nodes = chebyshev_nodes(3, 0.0, 1.0)
    expected = sort((cos.((2 .* (1:3) .- 1) .* pi ./ (2 * 3)) .+ 1) ./ 2)
    @test isapprox(nodes, expected; atol = 1e-12)
end

@testset "gauss-lobatto" begin
    nodes = gauss_lobatto_nodes(4, 0.0, 1.0)
    expected = sort((cos.((0:3) .* pi ./ 3) .+ 1) ./ 2)
    @test isapprox(nodes, expected; atol = 1e-12)
end

@testset "basis" begin
    x = [0.0, 0.5, 1.0]
    B = chebyshev_basis(x, 3, 0.0, 1.0)
    expected = [1.0 -1.0 1.0 -1.0; 1.0 0.0 -1.0 0.0; 1.0 1.0 1.0 1.0]
    @test isapprox(B, expected; atol = 1e-12)
end
