"""
NNOptim

Standalone, allocation-free optimisers with a common in-place API for
standalone experiments. Note: the solver kernels primarily use Optimisers.jl;
these are used by NNTrain utilities.
"""
module NNOptim

export Optimizer, SGD, RMSProp, Adam, init!, update!, make_optimizer
export step_lr, cosine_lr

"""
    abstract type Optimizer end

Lightweight, allocation-free optimizers with a common in-place API.
"""
abstract type Optimizer end

"""
    SGD(η; μ=0.0)

Stochastic gradient descent with momentum. State `v` matches parameter shapes.
Update: v = μ*v - η*g; p += v
"""
Base.@kwdef mutable struct SGD <: Optimizer
    η::Float64
    μ::Float64 = 0.0
    v::Vector{Array{Float64}} = Vector{Array{Float64}}()
end

"""
    RMSProp(η; ρ=0.99, ϵ=1e-8)

RMSProp with running second moment `s`.
Update: s = ρ*s + (1-ρ)*g.^2; p -= η * g ./ sqrt.(s .+ ϵ)
"""
Base.@kwdef mutable struct RMSProp <: Optimizer
    η::Float64
    ρ::Float64 = 0.99
    ϵ::Float64 = 1e-8
    s::Vector{Array{Float64}} = Vector{Array{Float64}}()
end

"""
    Adam(η; β1=0.9, β2=0.999, ϵ=1e-8)

Adam optimizer with bias correction.
Update:
  m = β1*m + (1-β1)*g; v = β2*v + (1-β2)*g.^2
  t += 1; m̂ = m/(1-β1^t); v̂ = v/(1-β2^t); p -= η * m̂ ./ (sqrt.(v̂) .+ ϵ)
"""
Base.@kwdef mutable struct Adam <: Optimizer
    η::Float64
    β1::Float64 = 0.9
    β2::Float64 = 0.999
    ϵ::Float64 = 1e-8
    m::Vector{Array{Float64}} = Vector{Array{Float64}}()
    v::Vector{Array{Float64}} = Vector{Array{Float64}}()
    t::Int = 0
end

"""
    init!(opt::Optimizer, params::Vector{<:AbstractArray})

Initializes internal state arrays to zeros with same shapes as `params`.
"""
function init!(opt::Optimizer, params::Vector{<:AbstractArray})
    if opt isa SGD
        opt.v = [zeros(Float64, size(p)) for p in params]
    elseif opt isa RMSProp
        opt.s = [zeros(Float64, size(p)) for p in params]
    elseif opt isa Adam
        opt.m = [zeros(Float64, size(p)) for p in params]
        opt.v = [zeros(Float64, size(p)) for p in params]
        opt.t = 0
    else
        throw(ArgumentError("Unknown optimizer type $(typeof(opt))"))
    end
    return opt
end

"""Update parameters in-place using SGD with momentum."""
function update!(opt::SGD, params::Vector{<:AbstractArray}, grads::Vector{<:AbstractArray})
    @inbounds for i in eachindex(params)
        v = opt.v[i]
        g = grads[i]
        @. v = opt.μ * v - opt.η * g
        @. params[i] += v
    end
    return nothing
end

"""Update parameters in-place using RMSProp."""
function update!(
    opt::RMSProp,
    params::Vector{<:AbstractArray},
    grads::Vector{<:AbstractArray},
)
    ρ = opt.ρ
    η = opt.η
    ϵ = opt.ϵ
    @inbounds for i in eachindex(params)
        s = opt.s[i]
        g = grads[i]
        @. s = ρ * s + (1 - ρ) * g^2
        @. params[i] -= η * g / (sqrt(s) + ϵ)
    end
    return nothing
end

"""Update parameters in-place using Adam with bias correction."""
function update!(opt::Adam, params::Vector{<:AbstractArray}, grads::Vector{<:AbstractArray})
    opt.t += 1
    β1 = opt.β1
    β2 = opt.β2
    η = opt.η
    ϵ = opt.ϵ
    b1t = 1 - β1^opt.t
    b2t = 1 - β2^opt.t
    @inbounds for i in eachindex(params)
        m = opt.m[i]
        v = opt.v[i]
        g = grads[i]
        @. m = β1 * m + (1 - β1) * g
        @. v = β2 * v + (1 - β2) * g^2
        # Bias-corrected
        @. params[i] -= η * (m / b1t) / (sqrt(v / b2t) + ϵ)
    end
    return nothing
end

"""
    make_optimizer(name::Symbol; η=1e-3, μ=0.9, ρ=0.99, β1=0.9, β2=0.999, ϵ=1e-8) -> Optimizer

Factory for optimizers by name. Examples:
  make_optimizer(:sgd, η=1e-2, μ=0.9)
  make_optimizer(:adam, η=1e-3)
"""
function make_optimizer(
    name::Symbol;
    η = 1e-3,
    μ = 0.9,
    ρ = 0.99,
    β1 = 0.9,
    β2 = 0.999,
    ϵ = 1e-8,
)::Optimizer
    lname = Symbol(lowercase(String(name)))
    if lname === :sgd
        return SGD(; η = float(η), μ = float(μ))
    elseif lname === :rmsprop
        return RMSProp(; η = float(η), ρ = float(ρ), ϵ = float(ϵ))
    elseif lname === :adam
        return Adam(; η = float(η), β1 = float(β1), β2 = float(β2), ϵ = float(ϵ))
    else
        throw(ArgumentError("Unknown optimizer name: $(name)"))
    end
end

"""
    step_lr(η0::Float64, epoch::Int; step_size::Int=10, gamma::Float64=0.5) -> Float64

Step‑down learning rate schedule.

Returns η = η0 * gamma^(floor((epoch-1)/step_size)).

Arguments:
- `η0`: Base learning rate (positive float).
- `epoch`: 1-based epoch index.

Keywords:
- `step_size` (Int, default 10): number of epochs between drops. Must be ≥ 1.
- `gamma` (Float64, default 0.5): multiplicative decay at each step. Must satisfy 0 < gamma ≤ 1.

Throws `ArgumentError` if `step_size < 1` or `gamma ≤ 0` or `gamma > 1`.

Example:
  julia> step_lr(0.1, 1; step_size=3, gamma=0.5)
  0.1
  julia> step_lr(0.1, 3; step_size=3, gamma=0.5)
  0.1
  julia> step_lr(0.1, 4; step_size=3, gamma=0.5)
  0.05
"""
function step_lr(
    η0::Float64,
    epoch::Int;
    step_size::Int = 10,
    gamma::Float64 = 0.5,
)::Float64
    step_size >= 1 || throw(ArgumentError("step_size must be ≥ 1, got $(step_size)"))
    (gamma > 0.0 && gamma <= 1.0) ||
        throw(ArgumentError("gamma must be in (0,1], got $(gamma)"))
    k = floor((epoch - 1) / step_size)
    return η0 * gamma^k
end

# -- Small example --
# julia> using .NNOptim
# julia> p = [randn(3,2), randn(2)]; g = [ones(3,2), fill(0.5,2)];
# julia> opt = make_optimizer(:adam, η=1e-2); init!(opt, p); update!(opt, p, g);

end # module
"""
    cosine_lr(η0::Float64, epoch::Int, E::Int; η_min::Float64=0.0) -> Float64

Cosine learning-rate schedule over E epochs. Safe for E=1.

η = η_min + 0.5*(η0-η_min)*(1 + cos(π*(epoch-1)/(E-1)))

Example:
  julia> using .NNOptim
  julia> NNOptim.cosine_lr(0.1, 1, 10)
  0.1
  julia> NNOptim.cosine_lr(0.1, 10, 10)
  0.0
"""
function cosine_lr(η0::Float64, epoch::Int, E::Int; η_min::Float64 = 0.0)::Float64
    if E <= 1
        return η0
    end
    ϕ = (epoch - 1) / (E - 1)
    return η_min + 0.5 * (η0 - η_min) * (1 + cos(pi * ϕ))
end

"""
    cosine_lr(η0::Float64, epoch::Int, E::Int; η_min::Float64=0.0) -> Float64

Cosine learning-rate schedule over E epochs. Safe for E=1.

η = η_min + 0.5*(η0-η_min)*(1 + cos(π*(epoch-1)/(E-1)))

Example:
  julia> using .NNOptim
  julia> NNOptim.cosine_lr(0.1, 1, 10)
  0.1
  julia> NNOptim.cosine_lr(0.1, 10, 10)
  0.0
"""
function cosine_lr(η0::Float64, epoch::Int, E::Int; η_min::Float64 = 0.0)::Float64
    if E <= 1
        return η0
    end
    ϕ = (epoch - 1) / (E - 1)
    return η_min + 0.5 * (η0 - η_min) * (1 + cos(pi * ϕ))
end
