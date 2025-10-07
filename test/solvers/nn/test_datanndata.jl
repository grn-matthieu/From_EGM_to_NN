# test/solvers/nn/test_datanndata.jl
const DataNN = ThesisProject.DataNN
using Random

# Minimal grids/shocks/helpers compatible with the NN data generator
make_G(a::AbstractVector) = Dict(:a => (grid = a, min = minimum(a), max = maximum(a)))
make_S(z::AbstractVector) = (zgrid = Float32.(z),)
make_P(; r = 0.02f0, y = log(1.2f0)) = (r = r, y = y)

@testset "DataNN.generate_dataset deterministic" begin
    a = collect(0.0:0.5:2.0)                 # Na = 5
    G = make_G(a)
    P = make_P()

    # mode = :full -> (Na, 2) Float32 columns = (y, w)
    X, Y = DataNN.generate_dataset(G, nothing, P; mode = :full)
    @test size(X) == (length(a), 2)
    @test eltype(X) == Float32
    @test Y === nothing

    y_expected = fill(Float32(exp(P.y)), length(a))
    Rg = 1.0f0 + Float32(P.r)
    w_expected = @. Rg * Float32(a) + y_expected
    @test X[:, 1] == y_expected
    @test X[:, 2] == w_expected

    # mode ≠ :full -> sampled in [amin, amax]; nsamples respected; deterministic rng
    rng = MersenneTwister(1234)
    X1, _ = DataNN.generate_dataset(G, nothing, P; mode = :sample, nsamples = 7, rng)
    rng2 = MersenneTwister(1234)
    X2, _ =
        DataNN.generate_dataset(G, nothing, P; mode = :anything, nsamples = 7, rng = rng2)

    @test size(X1) == (7, 2)
    @test X1 == X2

    amin, amax = extrema(Float32.(a))
    y_val = Float32(exp(P.y))
    w_min = (1.0f0 + Float32(P.r)) * amin + y_val
    w_max = (1.0f0 + Float32(P.r)) * amax + y_val
    @test all(X1[:, 1] .== y_val)
    @test all(w_min .<= X1[:, 2] .<= w_max)

    # nsamples = 0 defaults to length(a)
    rng3 = MersenneTwister(42)
    X3, _ = DataNN.generate_dataset(G, nothing, P; mode = :rand, nsamples = 0, rng = rng3)
    @test size(X3) == (length(a), 2)
    @test all(X3[:, 1] .== y_val)
    @test all(w_min .<= X3[:, 2] .<= w_max)
end

@testset "DataNN.generate_dataset stochastic" begin
    a = collect(-1.0:1.0:1.0)                # Na = 3
    z = [-0.7f0, 0.2f0]                      # Nz = 2, already Float32
    G = make_G(a)
    S = make_S(z)
    P = make_P()

    # mode = :full -> (Na*Nz, 2) with repeat pattern for (y, w)
    Xf, Yf = DataNN.generate_dataset(G, S, P; mode = :full)
    @test size(Xf) == (length(a) * length(z), 2)
    @test eltype(Xf) == Float32
    @test Yf === nothing

    Na, Nz = length(a), length(z)
    Aexp = repeat(Float32.(a), inner = Nz)
    Zexp = repeat(Float32.(z), outer = Na)
    μ = Float32(P.y)
    y_expected = exp.(μ .+ Zexp)
    Rg = 1.0f0 + Float32(P.r)
    w_expected = @. Rg * Aexp + y_expected
    @test Xf[:, 1] == y_expected
    @test Xf[:, 2] == w_expected

    # mode ≠ :full -> sampled within ranges; nsamples respected; deterministic rng
    rng = MersenneTwister(2024)
    Xs1, _ = DataNN.generate_dataset(G, S, P; mode = :sample, nsamples = 9, rng)
    rng2 = MersenneTwister(2024)
    Xs2, _ = DataNN.generate_dataset(G, S, P; mode = :anything, nsamples = 9, rng = rng2)

    @test size(Xs1) == (9, 2)
    @test Xs1 == Xs2

    amin, amax = extrema(Float32.(a))
    y_min = minimum(exp.(μ .+ z))
    y_max = maximum(exp.(μ .+ z))
    w_min = (1.0f0 + Float32(P.r)) * amin + y_min
    w_max = (1.0f0 + Float32(P.r)) * amax + y_max
    @test all(y_min .<= Xs1[:, 1] .<= y_max)
    @test all(w_min .<= Xs1[:, 2] .<= w_max)

    # nsamples = 0 defaults to Na * Nz
    rng3 = MersenneTwister(7)
    Xs3, _ = DataNN.generate_dataset(G, S, P; mode = :rand, nsamples = 0, rng = rng3)
    @test size(Xs3) == (length(a) * length(z), 2)
    @test all(y_min .<= Xs3[:, 1] .<= y_max)
    @test all(w_min .<= Xs3[:, 2] .<= w_max)
end
