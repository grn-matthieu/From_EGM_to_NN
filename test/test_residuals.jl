using Test
using ThesisProject.EGMResiduals: euler_resid_det, euler_resid_stoch

@testset "euler_resid_det" begin
    params = (; β = 0.96, σ = 2.0, r = 0.04)
    c = [1.0, 2.0]
    c_next = [1.1, 2.2]
    res = euler_resid_det(params, c, c_next)
    R = 1 + params.r
    expected = @. abs(1 - params.β * R * (c / c_next)^params.σ)
    @test res ≈ expected

    # zero consumption is clamped internally
    c0 = [0.0]
    c_next0 = [1.0]
    res0 = euler_resid_det(params, c0, c_next0)
    expected0 = abs(1 - params.β * R * ((1e-12) / c_next0[1])^params.σ)
    @test res0[1] ≈ expected0
end

@testset "euler_resid_stoch" begin
    params = (; β = 0.96, σ = 2.0, r = 0.04)
    a_grid = [0.0, 1.0]
    z_grid = [0.0, 0.0]
    Pz = [0.7 0.3; 0.4 0.6]
    c = [0.5 0.0; 0.6 0.5]
    res = euler_resid_stoch(params, a_grid, z_grid, Pz, c)

    R = 1 + params.r
    β = params.β
    σ = params.σ
    Na, Nz = size(c)
    expected = zeros(Float64, Na, Nz)
    for j = 1:Nz
        y = exp(z_grid[j])
        for i = 1:Na
            c_ij = max(c[i, j], 1e-12)
            ap = R * a_grid[i] + y - c_ij
            Emu = 0.0
            for jp = 1:Nz
                cp =
                    ap <= a_grid[1] ? c[1, jp] :
                    ap >= a_grid[end] ? c[end, jp] :
                    begin
                        t = (ap - a_grid[1]) / (a_grid[end] - a_grid[1])
                        (1 - t) * c[1, jp] + t * c[end, jp]
                    end
                Emu += Pz[j, jp] * (max(cp, 1e-12) / c_ij)^(-σ)
            end
            expected[i, j] = abs(1 - β * R * Emu)
        end
    end
    @test res ≈ expected

    # invalid transition matrix dimensions
    Pz_bad = ones(2, 3) / 3
    @test_throws AssertionError euler_resid_stoch(params, a_grid, z_grid, Pz_bad, c)
end
