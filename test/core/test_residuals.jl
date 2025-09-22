using Test
using ThesisProject.EulerResiduals:
    euler_resid_det,
    euler_resid_stoch,
    euler_resid_det_grid,
    euler_resid_det!,
    euler_resid_stoch!

@testset "euler_resid_det" begin
    params = (; β = 0.96, σ = 2.0, r = 0.04)
    c = [1.0, 2.0]
    c_next = [1.1, 2.2]
    res = euler_resid_det(params, c, c_next)
    R = 1 + params.r
    expected = @. abs(1 - params.β * R * (c / c_next)^params.σ)
    @test res ≈ expected

    res_buf = similar(c)
    @test euler_resid_det!(res_buf, params, c, c_next) === res_buf
    @test res_buf ≈ expected

    # zero consumption is clamped internally
    c0 = [0.0]
    c_next0 = [1.0]
    res0 = euler_resid_det(params, c0, c_next0)
    expected0 = abs(1 - params.β * R * ((1e-12) / c_next0[1])^params.σ)
    @test res0[1] ≈ expected0

    res0_buf = similar(c0)
    euler_resid_det!(res0_buf, params, c0, c_next0)
    @test res0_buf[1] ≈ expected0
end

@testset "euler_resid_det_interp" begin
    params = (; β = 0.96, σ = 2.0, r = 0.04, y = 1.0)
    a_grid = [0.0, 1.0]
    c = [0.5, 0.6]
    res = euler_resid_det_grid(params, a_grid, c)

    R = 1 + params.r
    β = params.β
    σ = params.σ
    expected = similar(c, Float64)
    for i in eachindex(c)
        c_i = max(c[i], 1e-12)
        ap = R * a_grid[i] + params.y - c_i
        cp =
            ap <= a_grid[1] ? c[1] :
            ap >= a_grid[end] ? c[end] :
            begin
                t = (ap - a_grid[1]) / (a_grid[end] - a_grid[1])
                (1 - t) * c[1] + t * c[end]
            end
        cp = max(cp, 1e-12)
        expected[i] = abs(1 - β * R * (c_i / cp)^σ)
    end
    @test res ≈ expected
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

    res_buf = similar(c)
    @test euler_resid_stoch!(res_buf, params, a_grid, z_grid, Pz, c) === res_buf
    @test res_buf ≈ expected

    # invalid transition matrix dimensions
    Pz_bad = ones(2, 3) / 3
    @test_throws AssertionError euler_resid_stoch(params, a_grid, z_grid, Pz_bad, c)
end

@testset "residual size assertions" begin
    params = (; β = 0.96, σ = 2.0, r = 0.04)

    c = [1.0, 2.0]
    c_next = [1.1, 2.2]
    res_bad = zeros(3)
    @test_throws AssertionError euler_resid_det!(res_bad, params, c, c_next)

    c_next_bad = [1.1]
    @test_throws AssertionError euler_resid_det(params, c, c_next_bad)

    params_interp = (; β = 0.96, σ = 2.0, r = 0.04, y = 1.0)
    a_grid = [0.0, 1.0]
    c_bad = [0.5]
    @test_throws AssertionError euler_resid_det_grid(params_interp, a_grid, c_bad)

    z_grid = [0.0, 0.0]
    Pz = [0.7 0.3; 0.4 0.6]
    c_mat = [0.5 0.0; 0.6 0.5]
    res_mat_bad = zeros(3, 2)
    @test_throws AssertionError euler_resid_stoch!(
        res_mat_bad,
        params,
        a_grid,
        z_grid,
        Pz,
        c_mat,
    )

    a_grid_bad = [0.0]
    @test_throws AssertionError euler_resid_stoch!(
        similar(c_mat),
        params,
        a_grid_bad,
        z_grid,
        Pz,
        c_mat,
    )
end
