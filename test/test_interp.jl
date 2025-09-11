using Test
using ThesisProject.CommonInterp: interp_linear!, interp_pchip!

@testset "interp_linear!" begin
    x = [0.0, 1.0, 2.0]
    y = [0.0, 1.0, 4.0]
    xq = [-1.0, 0.0, 0.5, 1.5, 2.0, 3.0]
    out = similar(xq)
    interp_linear!(out, x, y, xq)
    @test out == [0.0, 0.0, 0.5, 2.5, 4.0, 4.0]
end

@testset "interp_pchip!" begin
    x = [0.0, 1.0, 2.0, 3.0]
    y = x .^ 3
    xq = [-1.0, 0.0, 1.5, 3.0, 4.0]
    out = similar(xq)
    interp_pchip!(out, x, y, xq)
    @test isapprox(out, [0.0, 0.0, 3.439903846153846, 27.0, 27.0]; atol = 1e-8)

    # non-monotonic y should throw
    x_bad = [0.0, 1.0, 2.0]
    y_bad = [0.0, 1.0, 0.5]
    out_bad = similar(x_bad)
    @test_throws AssertionError interp_pchip!(out_bad, x_bad, y_bad, x_bad)

    # non-increasing x should throw
    x_bad2 = [0.0, 1.0, 1.0]
    y_bad2 = [0.0, 1.0, 2.0]
    out_bad2 = similar(x_bad2)
    @test_throws AssertionError interp_pchip!(out_bad2, x_bad2, y_bad2, x_bad2)
end
