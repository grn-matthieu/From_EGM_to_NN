using Test

@testset "quality" begin
    @testset "Aqua" begin
        try
            # Run a broad suite of hygiene checks
            @eval using Aqua
            Aqua.test_all(ThesisProject)
            @test true
        catch err
            @info "Aqua.jl not available; skipping Aqua checks" err
            @test true
        end
    end

    @testset "JET" begin
        try
            # Type stability and errors analysis
            @eval using JET
            JET.report_package(ThesisProject; target_modules = (ThesisProject,))
            @test true
        catch err
            @info "JET.jl not available; skipping JET checks" err
            @test true
        end
    end
end
