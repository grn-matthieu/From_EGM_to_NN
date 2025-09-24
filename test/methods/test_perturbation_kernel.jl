# test/solvers/perturbation/test_kernel.jl
using Test
using ThesisProject
using ForwardDiff
const PK = ThesisProject.PerturbationKernel

# ---------------------------
# Tiny stubs / fixtures
# ---------------------------
struct PDet
    σ::Float64
    β::Float64
    r::Float64
    y::Float64
end
PDet() = PDet(2.0, 0.95, 0.02, 1.0)

# Grid helpers
make_g(a::AbstractVector) = Dict{Symbol,Any}(
    :a => (grid = collect(a), min = minimum(a), max = maximum(a), N = length(a)),
)

# Shocks helpers
make_S(z::AbstractVector; Π = nothing, diagnostics = nothing) = begin
    Nz = length(z)
    Πm = Π === nothing ? fill(1 / Nz, Nz, Nz) : Π
    diag = diagnostics === nothing ? nothing : diagnostics
    (; zgrid = collect(z), Π = Πm, diagnostics = diag)
end

# ---------------------------
# Unit tests
# ---------------------------
@testset "PerturbationKernel – internals" begin
    p = PDet()
    # _coefficients_first_order
    Fa, Fz = PK._coefficients_first_order(p; ρ = 0.1, ȳ = 1.0, R = 1 + p.r)
    @test Fa ≈ (1 + p.r) - 1 / (p.β * (1 + p.r))
    @test Fz isa Real

    # _gauss_newton!: AD path
    rfun1(θ) = [θ[1] - 1.0, θ[2] + 2.0]
    θ = zeros(2)
    θ̂, ok, nr = PK._gauss_newton!(θ, rfun1, 10, 1e-12)
    @test ok
    @test nr ≤ 1e-10
    @test θ̂ ≈ [1.0, -2.0] atol = 1e-8

    # _gauss_newton!: finite-difference fallback (throw if Dual)
    rfun2(θ) = begin
        if eltype(θ) <: ForwardDiff.Dual
            error("no AD please")
        end
        [θ[1] - 3.0, θ[2] - 4.0]
    end
    θ2 = zeros(2)
    θ̂2, ok2, _ = PK._gauss_newton!(θ2, rfun2, 10, 1e-12)
    @test ok2
    @test θ̂2 ≈ [3.0, 4.0] atol = 1e-6
end

@testset "PerturbationKernel – deterministic solver" begin
    p = PDet()

    # (A) Order 1, Na>2 → interior-window branch (lo=2, hi=Na-1)
    G = make_g(0.0:1.0:4.0)  # Na = 5
    sol1 = PK.solve_perturbation_det(p, G, nothing; order = 1)
    @test sol1.iters == 1
    @test sol1.converged
    @test length(sol1.a_grid) == length(G[:a].grid)
    @test length(sol1.c) == length(G[:a].grid)
    @test length(sol1.a_next) == length(G[:a].grid)
    @test isfinite(sol1.max_resid)
    @test sol1.opts.order == 1
    @test sol1.opts.fit_ok == false
    @test sol1.opts.quad_coeffs.C2 == 0.0
    # bounds/clamping plausibility
    @test all(sol1.a_next .>= G[:a].min) && all(sol1.a_next .<= G[:a].max)

    # (B) Order 2 try-fit, Na<=2 → edge-window branch (lo=1, hi=Na)
    G2 = make_g([0.0, 2.0])  # Na = 2
    sol2 =
        PK.solve_perturbation_det(p, G2, nothing; order = 2, tol_fit = 1e-6, maxit_fit = 10)
    @test sol2.opts.order == 2
    @test haskey(sol2.opts.quad_coeffs, :C2)
    @test length(sol2.resid) == length(G2[:a].grid)
end

@testset "PerturbationKernel – stochastic solver" begin
    p = PDet()
    a = 0.0:1.0:3.0         # Na=4
    z = [-0.5, 0.0, 0.7]    # Nz=3
    Π = [
        0.6 0.3 0.1
        0.2 0.5 0.3
        0.1 0.3 0.6
    ]
    # With diagnostics → ρ branch exercised
    S = make_S(z; Π = Π, diagnostics = [0.25])
    G = make_g(a)

    # (A) Order 1
    solS1 = PK.solve_perturbation_stoch(p, G, S, nothing; order = 1)
    @test solS1.iters == 1
    @test solS1.converged
    @test size(solS1.c) == (length(a), length(z))
    @test size(solS1.a_next) == size(solS1.c)
    @test isfinite(solS1.max_resid)
    @test solS1.opts.order == 1
    @test solS1.opts.fit_ok == false
    @test solS1.opts.quad_coeffs.C2 == 0.0 &&
          solS1.opts.quad_coeffs.D2 == 0.0 &&
          solS1.opts.quad_coeffs.E2 == 0.0
    @test all(solS1.a_next .>= G[:a].min) && all(solS1.a_next .<= G[:a].max)

    # (B) Order 2 fitting path (may or may not fit_ok=true; just assert fields/flow)
    solS2 = PK.solve_perturbation_stoch(
        p,
        G,
        S,
        nothing;
        order = 2,
        tol_fit = 1e-6,
        maxit_fit = 10,
    )
    @test solS2.opts.order == 2
    @test haskey(solS2.opts.quad_coeffs, :C2)
    @test haskey(solS2.opts.quad_coeffs, :D2)
    @test haskey(solS2.opts.quad_coeffs, :E2)

    # (C) No diagnostics → ρ fallback branch
    S2 = make_S(z; Π = Π, diagnostics = nothing)
    solS3 = PK.solve_perturbation_stoch(p, G, S2, nothing; order = 1)
    @test solS3.converged && solS3.iters == 1
end
