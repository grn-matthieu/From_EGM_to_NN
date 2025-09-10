using Test
using ThesisProject.CommonInterp: pchip_slopes

@testset "pchip slopes" begin
    x = [1.0, 2.0, 3.0]
    y = [1.0, 0.5, 1.5]
    d, _, _ = pchip_slopes(x, y)
    @test d[2] == 0.0

    y2 = [1.0, 1.5, 2.5]
    d2, _, _ = pchip_slopes(x, y2)
    @test d2[2] ≈ 2 / 3
end
