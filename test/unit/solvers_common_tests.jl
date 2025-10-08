const CommonInterp = ThesisProject.CommonInterp
const EulerResiduals = ThesisProject.EulerResiduals
const PolicyUtils = ThesisProject.PolicyUtils
const ValueFunction = ThesisProject.ValueFunction
const Validators = ThesisProject.CommonValidators
const Chebyshev = ThesisProject.Chebyshev

@testset "Common interpolation" begin
    x = collect(0.0:1.0:4.0)
    y = x .^ 2
    out = similar(x)
    CommonInterp.interp_linear!(out, x, y, [0.5, 1.5])
    @test isapprox(out[1], 0.5; atol = 1e-12)
    @test isapprox(out[2], 2.5; atol = 1e-12)
    @test CommonInterp.interp_linear(x, y, 2.5) ≈ 6.5

    monotone_y = cumsum(fill(1.0, length(x)))
    xq = [1.0, 2.0, 3.5]
    pchip_out = similar(xq)
    CommonInterp.interp_pchip!(pchip_out, x, monotone_y, xq)
    @test all(diff(pchip_out) .>= 0)
end

@testset "Policy utilities" begin
    tol = PolicyUtils.compute_binding_tolerance(0.0, 10.0, 11)
    @test tol > 0

    a_grid = collect(range(0.0, 10.0; length = 5))
    init = PolicyUtils.init_consumption_det(a_grid, minimum(a_grid), 1.02, 1.0)
    @test length(init) == length(a_grid)
    custom = PolicyUtils.init_consumption_det(a_grid, 0.0, 1.02, 1.0; c_init = fill(2.0, 5))
    @test custom == fill(2.0, 5)

    arr = [0.0, 0.5, 2.0]
    PolicyUtils.ensure_minimum!(arr, 0.2)
    @test minimum(arr) >= 0.2
    PolicyUtils.clamp_policy!(arr, 0.2, 1.0)
    @test maximum(arr) <= 1.0

    a_endo = [-1.0, 0.2, 0.5, 0.8, 1.0]
    c_endo = [0.1, 0.2, 0.3, 0.4, 0.5]
    PolicyUtils.enforce_borrowing_constraint!(a_endo, c_endo, 0.0, 1.0, 1.01, a_grid)
    @test minimum(a_endo) >= 0.0

    a_sorted = similar(a_grid)
    c_sorted = similar(a_grid)
    PolicyUtils.sort_policy_pairs!(a_sorted, c_sorted, reverse(a_grid), reverse(a_grid))
    @test issorted(a_sorted)

    monotone = [0.0, 0.0, 0.0]
    PolicyUtils.enforce_strict_increase!(monotone)
    @test monotone[2] > monotone[1]
    nondec = [0.0, -0.1, 0.5]
    PolicyUtils.enforce_monotone!(nondec)
    @test Validators.is_nondec(nondec)

    prev = [1.0, 1.0]
    proposal = [2.0, 0.0]
    current = similar(prev)
    diff = PolicyUtils.relaxation_step!(current, prev, proposal, 0.5)
    @test diff ≈ 0.5

    resid = [0.1, 0.01, 0.2]
    a_next = [0.0, 0.3, 0.5]
    metric = PolicyUtils.rmse_nonbinding(resid, a_next, 0.0, 0.1)
    @test metric > 0
end

@testset "Euler residuals" begin
    params = (β = 1 / 1.01, σ = 1.0, r = 0.01, y = 1.0)
    c = fill(1.0, 5)
    resid = EulerResiduals.euler_resid_det(params, c, c)
    @test all(resid .<= 1e-12)

    a_grid = collect(range(0.0, 4.0; length = 5))
    c_grid = fill(1.0, 5)
    resid_grid = EulerResiduals.euler_resid_det_grid(params, a_grid, c_grid)
    @test length(resid_grid) == length(a_grid)

    z_grid = [-0.1, 0.1]
    Π = [0.9 0.1; 0.2 0.8]
    c_stoch = fill(1.0, length(a_grid), length(z_grid))
    resid_stoch = EulerResiduals.euler_resid_stoch(params, a_grid, z_grid, Π, c_stoch)
    @test size(resid_stoch) == size(c_stoch)

    resid_buf = similar(c_stoch)
    EulerResiduals.euler_resid_stoch!(resid_buf, params, a_grid, z_grid, Π, c_stoch)
    @test all(resid_buf .>= 0)

    EulerResiduals.euler_resid_stoch_interp!(
        resid_buf,
        params,
        a_grid,
        z_grid,
        Π,
        c_stoch,
        CommonInterp.LinearInterp(),
    )
    @test all(resid_buf .>= 0)
end

@testset "Value function" begin
    cfg = deterministic_config()
    model = ThesisProject.build_model(cfg)
    p = ThesisProject.get_params(model)
    g = ThesisProject.get_grids(model)
    U = ThesisProject.get_utility(model)
    policy = Dict{Symbol,Any}(
        :c => (; value = fill(0.8, g[:a].N), grid = g[:a].grid),
        :a => (; value = fill(g[:a].min, g[:a].N), grid = g[:a].grid),
    )
    V = ValueFunction.compute_value_policy(p, g, nothing, U, policy; maxit = 50)
    @test length(V) == g[:a].N

    cfg_s = stochastic_config()
    model_s = ThesisProject.build_model(cfg_s)
    ps = ThesisProject.get_params(model_s)
    gs = ThesisProject.get_grids(model_s)
    Ss = ThesisProject.get_shocks(model_s)
    Us = ThesisProject.get_utility(model_s)
    Na = gs[:a].N
    Nz = length(Ss.zgrid)
    policy_s = Dict{Symbol,Any}(
        :c => (; value = fill(0.8, Na, Nz), grid = gs[:a].grid),
        :a => (; value = fill(gs[:a].min, Na, Nz), grid = gs[:a].grid),
    )
    Vs = ValueFunction.compute_value_policy(ps, gs, Ss, Us, policy_s; maxit = 10)
    @test size(Vs) == (Na, Nz)
end

@testset "Validators" begin
    @test Validators.is_nondec([1.0, 1.0, 2.0])
    @test !Validators.is_nondec([1.0, 0.5])
    mat = [1.0 2.0; 1.0 3.0]
    @test Validators.is_nondec(mat)
    @test Validators.is_positive([0.1, 2.0])
    @test Validators.respects_amin([0.0, 0.5], 0.0)
end

@testset "Chebyshev utilities" begin
    nodes = Chebyshev.chebyshev_nodes(4, -1.0, 1.0)
    @test length(nodes) == 4
    lobatto = Chebyshev.gauss_lobatto_nodes(4, -1.0, 1.0)
    @test first(lobatto) ≈ -1.0
    x = [-1.0, 0.0, 1.0]
    basis = Chebyshev.chebyshev_basis(x, 3, -1.0, 1.0)
    @test size(basis) == (3, 4)
    @test basis[2, 2] ≈ 0.0
end
