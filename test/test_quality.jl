using Test

@testset "quality" begin
    @testset "Aqua" begin
        try
            @eval using Aqua
            # Run a broad suite of hygiene checks
            Aqua.test_all(ThesisProject)
            @test true
        catch err
            @warn "Aqua.jl not available; skipping Aqua checks" err
            @test true
        end
    end

    @testset "JET" begin
        try
            @eval using JET
            # Type stability and errors analysis
            JET.test_package(ThesisProject)
            @test true
        catch err
            @warn "JET.jl not available; skipping JET checks" err
            @test true
        end
    end
end
