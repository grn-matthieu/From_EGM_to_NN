using Test
using ThesisProject

@testset "Coverage - Core API stubs" begin
    # These API functions are stubs that should error with informative messages
    fns = (
        ThesisProject.get_params,
        ThesisProject.get_grids,
        ThesisProject.get_shocks,
        ThesisProject.get_utility,
        ThesisProject.build_model,
        ThesisProject.build_method,
        ThesisProject.load_config,
        ThesisProject.validate_config,
        ThesisProject.solve,
    )

    for f in fns
        @test_throws ErrorException f(123)
    end
end
