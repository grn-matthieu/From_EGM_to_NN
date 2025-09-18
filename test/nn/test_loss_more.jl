using Test
using ThesisProject
using ThesisProject.NNLoss:
    marg_u,
    inv_marg_u,
    weighted_mse,
    distance_to_bound,
    euler_loss,
    total_loss,
    EulerResidual

@testset "NNLoss: extra branches and utilities" begin
    # marg_u / inv_marg_u round-trip and special s≈1 branch
    c = [0.5, 1.0, 2.0]
    I1 = (; s = 1.0)                # u'(c) = 1/c
    I2 = (; s = 2.0)                # u'(c) = c^-2
    @test marg_u.(c, Ref(I1)) ≈ 1.0 ./ c
    @test marg_u.(c, Ref(I2)) ≈ c .^ (-2)
    @test inv_marg_u.(marg_u.(c, Ref(I2)), Ref(I2)) ≈ c

    # distance_to_bound for arrays and scalars
    @test distance_to_bound([0.9, 1.0, 1.2], 1.0) ≈ [0.1, 0.0, 0.0]
    @test distance_to_bound(0.8, 1.0) ≈ 0.2

    # weighted MSE vs manual computation
    R = [1.0, -2.0]
    w = [2.0, 1.0]
    @test weighted_mse(R, w) ≈ ((sqrt.(w) .* R) .^ 2 |> sum) / length(R)
    @test weighted_mse(R, 3.0; reduction = :sum) ≈ sum((sqrt(3.0) .* R) .^ 2)

    # total_loss = euler_loss + quadratic penalty
    ap = [0.8, 1.0, 1.2]
    a_min = 1.0
    R3 = [0.1, -0.2, 0.0]
    # Compute two ways and compare
    L_mse = euler_loss(R3)
    L_tot = total_loss(R3, ap, a_min; λ = 5.0)
    @test L_tot > L_mse  # penalty added
    # Sum reduction path
    @test total_loss(R3, ap, a_min; λ = 0.0, reduction = :sum) ≈ sum(R3 .^ 2)

    # EulerResidual with nMC > 1 and sampler(n) method
    # Simple environment where R, T are constant and sampler(n) is provided
    I = (; β = 1.0, s = 1.0, R = 1.0)
    resources(a, y; I) = I.R * a + y
    T(y, ε; I) = y
    Rf(ap, yp; I) = I.R
    # Sampler with arity-1 method to exercise that code path
    sampler(n::Int) = zeros(n)
    policy(a, y; I) = a
    a = [0.1, 0.2, 0.3]
    y = fill(1.0, 3)
    Rn = EulerResidual(a, y; I = I, policy = policy, sampler = sampler, nMC = 5)
    @test all(isfinite, Rn)
    @test maximum(abs, Rn) < 1e-10  # exact FOC in this toy setup
end
