"""
NNOptim

Lightweight optimisers with a minimal interface (`init!`, `update!`) used by the
neural-network training loops. The implementation favours clarity over exotic
features; Optimisers.jl remains the main backend for production workloads.
"""
module NNOptim

export Optimizer, SGD, RMSProp, Adam, init!, update!, make_optimizer, step_lr, cosine_lr

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

abstract type Optimizer end

Base.@kwdef mutable struct SGD <: Optimizer
    lr::Float64
    momentum::Float64 = 0.0
    velocity::Vector{Array{Float64}} = Array{Float64}[]
end

Base.@kwdef mutable struct RMSProp <: Optimizer
    lr::Float64
    decay::Float64 = 0.99
    eps::Float64 = 1e-8
    second_moment::Vector{Array{Float64}} = Array{Float64}[]
end

Base.@kwdef mutable struct Adam <: Optimizer
    lr::Float64
    beta1::Float64 = 0.9
    beta2::Float64 = 0.999
    eps::Float64 = 1e-8
    first_moment::Vector{Array{Float64}} = Array{Float64}[]
    second_moment::Vector{Array{Float64}} = Array{Float64}[]
    step::Int = 0
end

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_create_buffers(params) = [zeros(Float64, size(p)) for p in params]

function init!(opt::Optimizer, params::Vector{<:AbstractArray})
    if opt isa SGD
        opt.velocity = _create_buffers(params)
    elseif opt isa RMSProp
        opt.second_moment = _create_buffers(params)
    elseif opt isa Adam
        opt.first_moment = _create_buffers(params)
        opt.second_moment = _create_buffers(params)
        opt.step = 0
    else
        throw(ArgumentError("Unsupported optimizer $(typeof(opt))"))
    end
    return opt
end

# -----------------------------------------------------------------------------
# Updates
# -----------------------------------------------------------------------------

function update!(opt::SGD, params::Vector{<:AbstractArray}, grads::Vector{<:AbstractArray})
    @inbounds for i in eachindex(params)
        v = opt.velocity[i]
        g = grads[i]
        @. v = opt.momentum * v - opt.lr * g
        @. params[i] += v
    end
    return nothing
end

function update!(
    opt::RMSProp,
    params::Vector{<:AbstractArray},
    grads::Vector{<:AbstractArray},
)
    decay = opt.decay
    eps = opt.eps
    @inbounds for i in eachindex(params)
        s = opt.second_moment[i]
        g = grads[i]
        @. s = decay * s + (1 - decay) * g^2
        @. params[i] -= opt.lr * g / (sqrt(s) + eps)
    end
    return nothing
end

function update!(opt::Adam, params::Vector{<:AbstractArray}, grads::Vector{<:AbstractArray})
    opt.step += 1
    β1 = opt.beta1
    β2 = opt.beta2
    eps = opt.eps
    b1_corr = 1 - β1^opt.step
    b2_corr = 1 - β2^opt.step
    @inbounds for i in eachindex(params)
        m = opt.first_moment[i]
        v = opt.second_moment[i]
        g = grads[i]
        @. m = β1 * m + (1 - β1) * g
        @. v = β2 * v + (1 - β2) * g^2
        # Bias correction
        @. params[i] -= opt.lr * (m / b1_corr) / (sqrt(v / b2_corr) + eps)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Factories & schedules
# -----------------------------------------------------------------------------

function make_optimizer(
    name::Symbol;
    lr::Real = 1e-3,
    momentum::Real = 0.0,
    decay::Real = 0.99,
    beta1::Real = 0.9,
    beta2::Real = 0.999,
    eps::Real = 1e-8,
)::Optimizer
    lname = Symbol(lowercase(String(name)))
    if lname === :sgd
        return SGD(; lr = float(lr), momentum = float(momentum))
    elseif lname === :rmsprop
        return RMSProp(; lr = float(lr), decay = float(decay), eps = float(eps))
    elseif lname === :adam
        return Adam(;
            lr = float(lr),
            beta1 = float(beta1),
            beta2 = float(beta2),
            eps = float(eps),
        )
    else
        throw(ArgumentError("Unknown optimizer name $(name)"))
    end
end

"""
    step_lr(lr0, epoch; step_size=10, gamma=0.5)

Piecewise-constant schedule that multiplies `lr0` by `gamma` every `step_size`
epochs (1-based).
"""
function step_lr(lr0::Float64, epoch::Int; step_size::Int = 10, gamma::Float64 = 0.5)
    step_size >= 1 || throw(ArgumentError("step_size must be >= 1, got $(step_size)"))
    (0.0 < gamma <= 1.0) || throw(ArgumentError("gamma must lie in (0, 1], got $(gamma)"))
    drops = floor((epoch - 1) / step_size)
    return lr0 * gamma^drops
end

"""
    cosine_lr(lr0, epoch, total_epochs; lr_min=0.0)

Cosine annealing between `lr0` and `lr_min` across `total_epochs` epochs.
"""
function cosine_lr(lr0::Float64, epoch::Int, total_epochs::Int; lr_min::Float64 = 0.0)
    total_epochs <= 1 && return lr0
    phase = (epoch - 1) / (total_epochs - 1)
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + cos(pi * phase))
end

end # module
