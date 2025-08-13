# test/runtests.jl
# Testing file for ThesisProject
# See the Test.jl documentation for more details on writing tests
using Test
using Printf # A bit fancy

include(joinpath(@__DIR__, "..", "src", "ThesisProject.jl"))
using .ThesisProject


# ============= 1) Utility monotonicity =============
@testset "Utility monotonicity" begin
    # Testing on different levels of risk aversion
    σs = (1.0, 2.0, 3.0)
    cs = [0.8, 1.0, 1.2, 1.5]

    for σ in σs
        for i in 1:length(cs)-1
            @test u(cs[i+1], σ) > u(cs[i], σ)
        end
    end
end


# ============= 2) Budget constraint =============
@testset "Budget constraint" begin
    p = default_simple_params()

    states = [(a = a, c = min(p.y + (1+p.r)*a - p.a_min, 0.9*(p.y + (1+p.r)*a - p.a_min)))
              for a in range(p.a_min, p.a_max, length=5)]

    for s in states
        a′ = budget_next(s.a, p.y, p.r, s.c)
        @test a′ ≈ (1 + p.r)*s.a + p.y - s.c atol=1e-12
    end
end

# ============= 3) Policy feasibility=============
@testset "Policy feasibility: a' in [a_min, a_max], c ≥ 0" begin
    p = default_simple_params()
    Na = 7
    agrid = collect(range(p.a_min, p.a_max, length=Na))

    sol = solve_simple_egm(p, agrid; tol=1e-8, maxit=300, verbose=false)

    # Conditions listed above
    @test all(sol.a_next .>= p.a_min .- 1e-10)
    @test all(sol.a_next .<= p.a_max .+ 1e-10)

    resources = p.y .+ (1 + p.r) .* sol.agrid .- p.a_min
    @test all(sol.c .>= 0.0)
    @test all(sol.c .<= resources .+ 1e-10)
end

# ============= 4) Euler residuals shape/finite (sanity) =============
@testset "Euler residuals sanity" begin
    p = default_simple_params()
    Na = 7
    agrid = collect(range(p.a_min, p.a_max, length=Na))
    sol = solve_simple_egm(p, agrid; tol=1e-8, maxit=300, verbose=false)

    resid = euler_residuals_simple(p, sol.agrid, sol.c)
    @test length(resid) == length(sol.agrid)
    @test all(isfinite.(resid))
end


green(text) = "\033[32m" * text * "\033[0m"
println(green("\n✅ All tests passed successfully\n"))