module SolverIntegration

using LinearAlgebra: Symmetric, cholesky, eigen
using Random: default_rng, randn!
using ..CSVarUtils: csvar_expected_state

export integrate_expectation

const DEFAULT_MC_SAMPLES = 128
const SQRT_PI = sqrt(pi)
const SQRT_TWO = sqrt(2.0)

"""
    integrate_expectation(kind, f, params, shocks, y_state; kwargs...)

Approximate `E[f(y_{t+1}) | y_t = y_state]` for the CSVar process. Supports
Monte-Carlo (`:mc`) and Gauss–Hermite (`:gh`) integration schemes. Additional
keywords:

- `nsamples` (default 128) for Monte-Carlo
- `gh_order` (default 3) for Gauss–Hermite quadrature
- `rng` (default `Random.default_rng()`) used for Monte-Carlo draws
"""
function integrate_expectation(
    kind::Symbol,
    f::Function,
    params,
    shocks,
    y_state;
    nsamples::Int = DEFAULT_MC_SAMPLES,
    gh_order::Int = 3,
    rng = default_rng(),
)
    kind in (:mc, :gh) || error("Unknown integration kind $(kind); expected :mc or :gh")
    y_state === nothing && error("integrate_expectation requires a current state vector")
    A =
        hasproperty(params, :A) ? params.A :
        error("params.A required for CSVar integration")
    Σ =
        hasproperty(params, :Σ) ? params.Σ :
        error("params.Σ required for CSVar integration")
    μ = csvar_expected_state(A, y_state)
    Σ_matrix = Matrix{Float64}(Σ)
    if kind == :mc
        return _mc_expectation(f, μ, Σ_matrix, nsamples, rng)
    else
        return _gh_expectation(f, μ, Σ_matrix, gh_order)
    end
end

function _mc_expectation(f, μ::AbstractVector, Σ::AbstractMatrix, nsamples::Int, rng)
    nsamples > 0 || error("nsamples must be positive, got $nsamples")
    d = length(μ)
    if all(abs.(Σ) .< eps()) || d == 0
        return f(μ)
    end
    L = cholesky(Symmetric(Σ), check = false).L
    draws = Matrix{Float64}(undef, d, nsamples)
    randn!(rng, draws)
    y_next = μ .+ L * view(draws, :, 1)
    acc = f(y_next)
    for s = 2:nsamples
        y_next = μ .+ L * view(draws, :, s)
        acc += f(y_next)
    end
    return acc / nsamples
end

function _gh_expectation(f, μ::AbstractVector, Σ::AbstractMatrix, order::Int)
    order ≥ 1 || error("Gauss-Hermite order must be ≥ 1, got $order")
    d = length(μ)
    if all(abs.(Σ) .< eps()) || d == 0
        return f(μ)
    end
    nodes1d, weights1d = _gauss_hermite_nodes_weights(order)
    L = cholesky(Symmetric(Σ), check = false).L
    acc = nothing
    weight_norm = (SQRT_PI)^d
    idx_ranges = ntuple(_ -> 1:order, d)
    x = Vector{Float64}(undef, d)
    for idx in Iterators.product(idx_ranges...)
        weight = 1.0
        for k = 1:d
            node = nodes1d[idx[k]]
            weight *= weights1d[idx[k]]
            x[k] = node
        end
        y_next = μ .+ SQRT_TWO * (L * x)
        val = weight * f(y_next)
        acc = acc === nothing ? val : acc + val
    end
    return acc / weight_norm
end

function _gauss_hermite_nodes_weights(order::Int)
    order ≥ 1 || error("Gauss-Hermite order must be ≥ 1, got $order")
    β = sqrt.(collect(1:order-1) ./ 2)
    J = zeros(Float64, order, order)
    for i = 1:(order-1)
        J[i, i+1] = β[i]
        J[i+1, i] = β[i]
    end
    eig = eigen(Symmetric(J))
    nodes = eig.values
    weights = (eig.vectors[1, :] .^ 2) .* SQRT_PI
    return nodes, weights
end

end # module
