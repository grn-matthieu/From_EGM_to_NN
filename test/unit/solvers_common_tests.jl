using Random: MersenneTwister
using LinearAlgebra: I

const CommonInterp = ThesisProject.CommonInterp
const EulerResiduals = ThesisProject.EulerResiduals
const PolicyUtils = ThesisProject.PolicyUtils
const ValueFunction = ThesisProject.ValueFunction
const Validators = ThesisProject.CommonValidators
const Chebyshev = ThesisProject.Chebyshev
const CSVarUtils = ThesisProject.CSVarUtils
const MethodUtils = ThesisProject.MethodUtils
const ConsumerSavingVAR = ThesisProject.ConsumerSavingVAR

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

@testset "Consumption warm starts" begin
    cfg = deterministic_config()
    model = ThesisProject.build_model(cfg)
    p = ThesisProject.get_params(model)
    g = ThesisProject.get_grids(model)

    c_det = MethodUtils.build_consumption_initializer(p, g; warm_start = :steady_state)
    @test c_det isa Vector{Float64}
    @test length(c_det) == g[:a].N

    vec_params = (
        model = (name = :cs_vec,),
        params = (
            y = [0.8, 1.2],
            A = [[0.9, 0.1], [0.05, 0.95]],
            Σ = [[0.1, 0.0], [0.0, 0.1]],
        ),
    )
    cfg_vec = deep_merge(cfg, vec_params)
    model_vec = ConsumerSavingVAR.build_cs_var_model(cfg_vec)
    p_vec = ConsumerSavingVAR.get_params(model_vec)
    g_vec = ConsumerSavingVAR.get_grids(model_vec)

    c_vec =
        MethodUtils.build_consumption_initializer(p_vec, g_vec; warm_start = :steady_state)

    y_states = CSVarUtils.csvar_state_matrix(p_vec.y, p_vec.y_dim)
    incomes = CSVarUtils.csvar_state_incomes(y_states)

    @test size(c_vec) == (g_vec[:a].N, length(incomes))

    a_grid_vec = g_vec[:a].grid
    a_min_vec = g_vec[:a].min
    R_vec = 1 + p_vec.r

    expected = Array{Float64}(undef, length(a_grid_vec), length(incomes))
    tmp = similar(a_grid_vec, Float64)
    for (j, y_val) in enumerate(incomes)
        @inbounds for (i, a) in enumerate(a_grid_vec)
            cval = y_val + R_vec * a - a
            cmax = y_val + R_vec * a - a_min_vec
            tmp[i] = clamp(cval, 1e-12, cmax)
        end
        @views expected[:, j] .= tmp
    end

    @test c_vec ≈ expected
end

@testset "CSVar utilities" begin
    # csvar_income interprets components as log incomes and returns exp-sum
    @test CSVarUtils.csvar_income(4.0) == exp(4.0)
    @test CSVarUtils.csvar_income([2.0, 4.0]) ≈ (exp(2.0) + exp(4.0))
    Ymat = [1.0 3.0; 2.0 4.0]
    @test CSVarUtils.csvar_income(Ymat) ≈ [exp(1.0) + exp(2.0), exp(3.0) + exp(4.0)]
    states = [[2.0, 4.0], [0.0, 2.0]]
    @test CSVarUtils.csvar_income(states) ≈ [exp(2.0) + exp(4.0), exp(0.0) + exp(2.0)]
    @test_throws ArgumentError CSVarUtils.csvar_income(Float64[])

    a_prev = [0.0, 1.0]
    y_vec = [2.0, 4.0]
    w = CSVarUtils.csvar_cash_on_hand(a_prev, y_vec, 0.05)
    # income is sum(exp(y_vec)) for CSVAR
    income_y = CSVarUtils.csvar_income(y_vec)
    @test w ≈ (1.05 .* a_prev .+ income_y)
    c = [0.5, 0.75]
    assets = CSVarUtils.csvar_assets_from_cash(w, c)
    @test assets ≈ w .- c
    y_next = [1.0, 3.0]
    w_next = CSVarUtils.csvar_next_cash_on_hand(w, c, y_next, 0.05)
    income_next = CSVarUtils.csvar_income(y_next)
    @test w_next ≈ (1.05 .* (w .- c) .+ income_next)

    Σ = CSVarUtils.csvar_covariance(0.3, 3; variance = 2.0)
    @test size(Σ) == (3, 3)
    @test [Σ[i, i] for i = 1:3] ≈ fill(2.0, 3)
    @test Σ[1, 2] ≈ 0.6
    @test_throws ArgumentError CSVarUtils.csvar_covariance(-1.0, 3)
    @test_throws ArgumentError CSVarUtils.csvar_covariance(0.2, 2; variance = -1.0)

    chol = CSVarUtils.csvar_shock_factor(0.2, 2)
    Σ_expected = CSVarUtils.csvar_covariance(0.2, 2)
    Σ_rec = Matrix(chol.L) * Matrix(chol.L)'
    @test Σ_rec ≈ Σ_expected

    rng = MersenneTwister(42)
    ε = CSVarUtils.csvar_draw_shock(rng, 2; T = Float32)
    @test length(ε) == 2
    @test eltype(ε) == Float32
    ε_buf = zeros(Float64, 2)
    CSVarUtils.csvar_draw_shock!(rng, ε_buf)
    @test !all(iszero, ε_buf)

    A = [0.9 0.1; 0.2 0.8]
    y = [1.0, 2.0]
    @test CSVarUtils.csvar_expected_state(A, y) ≈ A * y
    out = similar(y)
    CSVarUtils.csvar_expected_state!(out, A, y)
    @test out ≈ A * y

    ε_unit = ones(Float64, 2)
    L = Matrix{Float64}(I, 2, 2)
    next_state = CSVarUtils.csvar_next_state(A, y, L, ε_unit)
    @test next_state ≈ A * y .+ ε_unit
    out_state = similar(y)
    CSVarUtils.csvar_next_state!(out_state, A, y, L, ε_unit)
    @test out_state ≈ next_state

    rng1 = MersenneTwister(77)
    rng2 = MersenneTwister(77)
    step_val = CSVarUtils.csvar_step(rng1, A, y, chol)
    ε_step = CSVarUtils.csvar_draw_shock(rng2, 2; T = eltype(chol.L))
    manual = CSVarUtils.csvar_next_state(A, y, Matrix(chol.L), ε_step)
    @test step_val ≈ manual

    rng3 = MersenneTwister(90)
    rng4 = MersenneTwister(90)
    step_rho = CSVarUtils.csvar_step(rng3, A, y, 0.25)
    chol_rho = CSVarUtils.csvar_shock_factor(0.25, 2)
    ε_rho = CSVarUtils.csvar_draw_shock(rng4, 2; T = eltype(chol_rho.L))
    manual_rho = CSVarUtils.csvar_next_state(A, y, Matrix(chol_rho.L), ε_rho)
    @test step_rho ≈ manual_rho

    rng5 = MersenneTwister(101)
    rng6 = MersenneTwister(101)
    ε_buffer = zeros(Float64, 2)
    step_buf = CSVarUtils.csvar_step(rng5, A, y, chol; ε_buffer = ε_buffer)
    ε_manual = CSVarUtils.csvar_draw_shock(rng6, 2; T = eltype(chol.L))
    manual_buf = CSVarUtils.csvar_next_state(A, y, Matrix(chol.L), ε_manual)
    @test step_buf ≈ manual_buf
    @test ε_buffer ≈ ε_manual

    Y_states = [1.0 2.0; 0.5 0.75]
    a_entry = (grid = [0.0, 0.5, 1.0, 1.5], tensor_shape = (2, 2), N = 4)

    joint_grid = CSVarUtils.csvar_joint_state_grid(Y_states, a_entry)
    @test size(joint_grid) == (3, length(a_entry.grid), size(Y_states, 2))
    @test joint_grid[1, :, 1] ≈ fill(1.0, length(a_entry.grid))
    @test joint_grid[2, :, 2] ≈ fill(0.75, length(a_entry.grid))
    @test joint_grid[3, :, 1] ≈ a_entry.grid

    joint_tensor = CSVarUtils.csvar_joint_state_tensor(Y_states, a_entry)
    @test size(joint_tensor) == (3, 2, 2, 2)
    joint_matrix = CSVarUtils.csvar_joint_state_matrix(Y_states, a_entry)
    @test size(joint_matrix) == (3, length(a_entry.grid) * size(Y_states, 2))
    @test joint_matrix ≈ reshape(joint_tensor, 3, :)

    flat_policy = collect(1.0:4.0)
    tensor_policy = CSVarUtils.csvar_tensorise(flat_policy, a_entry)
    @test size(tensor_policy) == a_entry.tensor_shape
    @test vec(CSVarUtils.csvar_vectorise(tensor_policy, a_entry)) ≈ flat_policy

    matrix_policy = reshape(collect(1.0:8.0), 4, 2)
    tensorised = CSVarUtils.csvar_tensorise(matrix_policy, a_entry)
    @test size(tensorised) == (a_entry.tensor_shape..., 2)
    @test CSVarUtils.csvar_vectorise(tensorised, a_entry) ≈ matrix_policy
end

@testset "Euler residuals" begin
    params = (β = 1 / 1.01, γ = 1.0, r = 0.01, y = 1.0)
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
