# test/methods/test_datanndata.jl
const DataNN = ThesisProject.DataNN
using Random

# Minimal grids/shocks helpers
make_G(a::AbstractVector) = Dict(:a => (grid = a,))
make_S(z::AbstractVector) = (zgrid = z,)

@testset "DataNN.generate_dataset deterministic" begin
    a = collect(0.0:0.5:2.0)                 # Na = 5
    G = make_G(a)

    # mode = :full -> Na×1 Float32 column equals a
    X, Y = DataNN.generate_dataset(G, nothing; mode = :full)
    @test size(X) == (length(a), 1)
    @test eltype(X) == Float32
    @test all(vec(X) .== Float32.(a))
    @test Y === nothing

    # mode ≠ :full -> sampled in [amin, amax]; nsamples respected; deterministic rng
    rng = MersenneTwister(1234)
    X1, _ = DataNN.generate_dataset(G, nothing; mode = :sample, nsamples = 7, rng)
    rng2 = MersenneTwister(1234)
    X2, _ = DataNN.generate_dataset(G, nothing; mode = :anything, nsamples = 7, rng = rng2)

    @test size(X1) == (7, 1)
    @test X1 == X2
    amin, amax = extrema(Float32.(a))
    @test all(amin .<= X1[:] .<= amax)

    # nsamples = 0 defaults to length(a)
    rng3 = MersenneTwister(42)
    X3, _ = DataNN.generate_dataset(G, nothing; mode = :rand, nsamples = 0, rng = rng3)
    @test size(X3) == (length(a), 1)
    @test all(amin .<= X3[:] .<= amax)
end

@testset "DataNN.generate_dataset stochastic" begin
    a = collect(-1.0:1.0:1.0)                # Na = 3
    z = [-0.7f0, 0.2f0]                      # Nz = 2, already Float32
    G = make_G(a)
    S = make_S(z)

    # mode = :full -> (Na*Nz)×2 with repeat pattern
    Xf, Yf = DataNN.generate_dataset(G, S; mode = :full)
    @test size(Xf) == (length(a) * length(z), 2)
    @test eltype(Xf) == Float32
    @test Yf === nothing

    # Column 1 equals repeat(a, inner=Nz), col 2 equals repeat(z, outer=Na)
    Aexp = repeat(Float32.(a), inner = length(z))
    Zexp = repeat(Float32.(z), outer = length(a))
    @test Xf[:, 1] == Aexp
    @test Xf[:, 2] == Zexp

    # mode ≠ :full -> sampled within ranges; nsamples respected; deterministic rng
    rng = MersenneTwister(2024)
    Xs1, _ = DataNN.generate_dataset(G, S; mode = :sample, nsamples = 9, rng)
    rng2 = MersenneTwister(2024)
    Xs2, _ = DataNN.generate_dataset(G, S; mode = :anything, nsamples = 9, rng = rng2)

    @test size(Xs1) == (9, 2)
    @test Xs1 == Xs2

    amin, amax = extrema(Float32.(a))
    zmin, zmax = minimum(Float32.(z)), maximum(Float32.(z))
    @test all(amin .<= Xs1[:, 1] .<= amax)
    @test all(zmin .<= Xs1[:, 2] .<= zmax)

    # nsamples = 0 defaults to Na * Nz
    rng3 = MersenneTwister(7)
    Xs3, _ = DataNN.generate_dataset(G, S; mode = :rand, nsamples = 0, rng = rng3)
    @test size(Xs3) == (length(a) * length(z), 2)
    @test all(amin .<= Xs3[:, 1] .<= amax)
    @test all(zmin .<= Xs3[:, 2] .<= zmax)
end
