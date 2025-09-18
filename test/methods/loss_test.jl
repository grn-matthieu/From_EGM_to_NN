using Test
using ThesisProject
import ThesisProject.NNData: grid_minibatches

@testset "NNData.grid_minibatches shapes and determinism" begin
    a = collect(Float64, 1:3)          # Na = 3
    y = collect(Float64, 0:3)          # Ny = 4
    Na, Ny = length(a), length(y)

    # Targets encode (i,j) -> 100*j + i for easy checking
    T = Array{Float64}(undef, Na, Ny)
    for j = 1:Ny, i = 1:Na
        T[i, j] = 100 * j + i
    end

    # 1) Basic shapes + order (no shuffle, no targets)
    it1 = grid_minibatches(a, y; batch = 5, shuffle = false)
    first_batch = iterate(it1)
    @test first_batch !== nothing
    (X, Y), st = first_batch
    @test size(X) == (2, 5)
    @test size(Y) == (1, 5)
    # expected indices for first 5 points in a-fast order: (1,1),(2,1),(3,1),(1,2),(2,2)
    exp_ia = [1, 2, 3, 1, 2]
    exp_iy = [1, 1, 1, 2, 2]
    @test X[1, :] == a[exp_ia]
    @test X[2, :] == y[exp_iy]
    @test all(isnan.(Y))

    # 2) Targets mapping
    it2 = grid_minibatches(a, y; targets = T, batch = 5, shuffle = false)
    (X2, Y2), _ = iterate(it2)
    @test Y2[1, :] == T[CartesianIndex.(exp_ia, exp_iy)]

    # 3) Antithetic doubles size for Ny even (no self-mirror)
    it3 = grid_minibatches(a, y; targets = T, batch = 3, shuffle = false, antithetic = true)
    (Xa, Ya), _ = iterate(it3)
    @test size(Xa, 2) == 6
    @test size(Ya, 2) == 6

    # 4) Deterministic with seed + shuffle
    seed = 777
    itA = grid_minibatches(a, y; targets = T, batch = 4, shuffle = true, seed = seed)
    itB = grid_minibatches(a, y; targets = T, batch = 4, shuffle = true, seed = seed)

    # take first two batches from each and compare for equality
    function take2(it)
        out = Tuple{Matrix{Float64},Matrix{Float64}}[]
        c = 0
        for (Xb, Yb) in it
            push!(out, (copy(Xb), copy(Yb)))
            c += 1
            if c >= 2
                break
            end
        end
        return out
    end
    A = take2(itA)
    B = take2(itB)
    @test length(A) == length(B) == 2
    for k = 1:2
        @test A[k][1] == B[k][1]
        @test A[k][2] == B[k][2]
    end

    # 5) Length with drop_last
    it4 = grid_minibatches(a, y; batch = 7, shuffle = false, drop_last = true)
    @test length(it4) == div(Na * Ny, 7)
    it5 = grid_minibatches(a, y; batch = 7, shuffle = false, drop_last = false)
    @test length(it5) == cld(Na * Ny, 7)
end
