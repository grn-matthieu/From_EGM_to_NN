using Test
using LinearAlgebra
using ThesisProject
using ThesisProject.ValueFunction: compute_value, compute_value_policy
using ThesisProject.Shocks: ShockOutput

@testset "Value function deterministic" begin
    β = 0.9
    p = (; β = β)

    agrid = [0.0, 1.0]
    g = Dict(:a => (; grid = agrid, N = length(agrid)))

    U = (; u = c -> log.(c))

    cpol = fill(2.0, length(agrid))
    apol = fill(0.0, length(agrid))
    policy = Dict(:c => (; value = cpol), :a => (; value = apol))

    V = compute_value(p, g, nothing, U, policy)
    V_ss = log(2.0) / (1 - β)
    @test all(isapprox.(V, V_ss; atol = 1e-8))
end

@testset "Value function stochastic" begin
    β = 0.9
    p = (; β = β)

    Na = 3
    agrid = collect(range(0.0, 1.0; length = Na))
    g = Dict(:a => (; grid = agrid, N = Na))

    U = (; u = c -> log.(c))

    P = [0.8 0.2; 0.1 0.9]
    Nz = size(P, 1)
    S = ShockOutput([0.0, 0.0], P, fill(1 / Nz, Nz), zeros(3))

    cvals = [1.0, 2.0]
    cpol = hcat(fill(cvals[1], Na), fill(cvals[2], Na))
    apol = zeros(Na, Nz)
    policy = Dict(:c => (; value = cpol), :a => (; value = apol))

    V = compute_value_policy(p, g, S, U, policy)

    uvec = log.(cvals)
    Vstates = (I - β * P) \ uvec
    V_expected = hcat(fill(Vstates[1], Na), fill(Vstates[2], Na))

    @test size(V) == (Na, Nz)
    @test maximum(abs.(V .- V_expected)) < 1e-8
end
