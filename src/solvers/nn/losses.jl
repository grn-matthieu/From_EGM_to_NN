module NNLosses

import CUDA
import Zygote
using Random
const PARENT = parentmodule(@__MODULE__)
using ChainRulesCore: ignore_derivatives, AbstractZero
using LinearAlgebra: I, dot
using Statistics: mean

export build_loss_function,
    flatten_sum_squares,
    _var_bcmc_given_N,
    estimate_linearized_components,
    _shift_features,
    _shift_eps_state,
    _shift_s_state,
    _grad_to_vector,
    suggest_bcmc_N,
    randn_like,
    fill_like

# Device placement helpers are defined in `mixed_precision.jl` (included by
# `kernel.jl`). Loss routines should call `maybe_to_device` / `maybe_to_host`
# from that central location when they need to move data between host and
# device.

function randn_like(rng, ref::CUDA.AbstractGPUArray)
    # GPU gaussian noise
    return ignore_derivatives() do
        CUDA.randn(eltype(ref), size(ref)...)
    end
end

function randn_like(rng, ref)
    # CPU gaussian noise
    return ignore_derivatives() do
        out = similar(ref)
        randn!(rng, out)
        out
    end
end

function fill_like(value, ref)
    if ref isa CUDA.AbstractGPUArray
        out = similar(ref)
        fill!(out, value)
        return out
    else
        return fill(value, size(ref))
    end
end

function build_loss_function(
    P_resid,
    G,
    S,
    scaler,
    settings,
    rng::AbstractRNG,
    model_cfg = nothing,
)
    # Validate that objective is one of the supported FB methods
    if settings.objective ∉ (:euler_fb_aio, :euler_fb_bcmc)
        throw(
            ArgumentError(
                "NN solver only supports objectives :euler_fb_aio and :euler_fb_bcmc, got :$(settings.objective)",
            ),
        )
    end

    function fb_supported(model_cfg)
        P_full = model_cfg.P
        has_ar1 = size(P_full.y, 1) == 1
        has_var = size(P_full.y, 1) > 1
        return has_ar1 || has_var
    end

    return function (model, ps, st, data)
        X = data[1]

        # Only FB objectives are supported
        if !fb_supported(model_cfg)
            throw(
                ArgumentError(
                    "NN solver with objective :$(settings.objective) requires active stochastic shocks",
                ),
            )
        end

        objective = settings.objective
        # the loss routines return (loss, (st1, aux_namedtuple))
        loss_val, st_pack = if objective == :euler_fb_aio
            PARENT.loss_euler_fb_aio!(model, ps, st, X, model_cfg, rng)
        else
            P_full = model_cfg.P
            is_csvar = isdefined(P_full, :y_dim) && P_full.y_dim > 1
            if is_csvar
                PARENT.loss_euler_fb_bcmc_csvar!(model, ps, st, X, model_cfg, rng)
            else
                PARENT.loss_euler_fb_bcmc_ar1!(model, ps, st, X, model_cfg, rng)
            end
        end
        st1, aux = st_pack
        # package diagnostics: include FB aux diagnostics and leave phi/h fields empty
        diag = (;
            phi = nothing,
            h = nothing,
            a = nothing,
            z = nothing,
            w = nothing,
            c = nothing,
            fb = aux,
        )
        return loss_val, st1, diag
    end
end

function flatten_sum_squares(x)
    if x === nothing
        return 0.0
    elseif x isa Number
        return float(x)^2
    elseif x isa AbstractArray
        # Avoid unnecessary host transfers: compute reductions on-device when possible.
        s = sum(abs2, x)
        return Float32(s)
    elseif x isa NamedTuple || x isa Tuple || x isa Vector || x isa Dict
        s = 0.0
        for v in x
            s += flatten_sum_squares(v)
        end
        return s
    else
        try
            s = 0.0
            for f in fieldnames(typeof(x))
                s += flatten_sum_squares(getfield(x, f))
            end
            return s
        catch
            return 0.0
        end
    end
end

@inline function _var_bcmc_given_N(sigma2_f::Float32, rho_f::Float32, N::Int, T::Int)
    N ≤ 1 && return Inf
    denom = max(N - 1, 1)
    const_term = ((N - 2)^2 + N - 1) * (rho_f^2)
    mixed_term = (2 * (N - 2) * rho_f + sigma2_f) * sigma2_f
    return (const_term + mixed_term) / (denom * max(T, 1))
end

@inline function _shift_features(xj, idxs::Vector{Int}, vals::Vector)
    isempty(idxs) && return collect(xj)
    T = eltype(xj)
    n = length(xj)
    return [xj[i] + sum(T(v) for (idx, v) in zip(idxs, vals) if idx == i) for i = 1:n]
end

function _shift_eps_state(xj, scaler::Any, ε)
    if !(scaler.has_shocks && scaler.csvar_mode)
        return collect(xj)
    end
    off = 1
    ny = length(scaler.y_range)
    idxs = [off + k for k = 1:ny]
    vals = [Float32(ε[k]) for k = 1:ny]
    return _shift_features(xj, idxs, vals)
end

function _shift_s_state(xj, scaler::Any, svec)
    Tlen = length(xj)
    if scaler.has_shocks && scaler.csvar_mode
        ny = length(scaler.y_range)
        idxs = vcat([1 + k for k = 1:ny], Tlen)
        vals = vcat(Float32.(svec[1:ny]), Float32(svec[end]))
        return _shift_features(xj, idxs, vals)
    else
        idxs = Tlen == 1 ? [1] : [1, Tlen]
        vals = Tlen == 1 ? Float32.([svec[1]]) : Float32.([svec[1], svec[end]])
        return _shift_features(xj, idxs, vals)
    end
end

@inline function _grad_to_vector(grad_val, dim::Int)
    if grad_val === nothing || grad_val isa AbstractZero
        return zeros(Float64, dim)
    end
    grad_vec = grad_val
    grad_vec isa Number && return fill(Float64(grad_vec), dim)
    return Float64.(collect(grad_vec))
end

function estimate_linearized_components(
    model,
    ps,
    st,
    X_probe,
    scaler::Any,
    Σs::AbstractMatrix,
    Σε::AbstractMatrix,
)
    nb = size(X_probe, 2)
    A_acc = 0.0
    B_acc = 0.0
    for j = 1:nb
        xj = @view X_probe[:, j]

        g = ε -> begin
            x_ε = _shift_eps_state(xj, scaler, ε)
            return Float64(model(x_ε, ps, st; mode = :fb_scalar))
        end

        raw_∇ε = Zygote.gradient(g, zeros(size(Σε, 1)))[1]
        ∇ε = _grad_to_vector(raw_∇ε, size(Σε, 1))

        h = svec -> begin
            x_s = _shift_s_state(xj, scaler, svec)
            return Float64(model(x_s, ps, st; mode = :fb_scalar))
        end

        raw_∇s = Zygote.gradient(h, zeros(size(Σs, 1)))[1]
        ∇s = _grad_to_vector(raw_∇s, size(Σs, 1))
        A_acc += max(dot(∇ε, Σε * ∇ε), 0.0)
        B_acc += max(dot(∇s, Σs * ∇s), 0.0)
    end
    A = A_acc / nb
    B = B_acc / nb
    sigma2_f = max(A + B, eps(Float32))
    rho_f = max(B, eps(Float32))
    return sigma2_f, rho_f, A, B
end

function suggest_bcmc_N(sigma2_f::Float64, rho_f::Float64, T::Int; N_cap::Int = 1024)
    bestN, bestV = 2, _var_bcmc_given_N(sigma2_f, rho_f, 2, T)
    Nmax = max(2, min(2T, N_cap))
    for N = 3:Nmax
        if (2T ÷ N) < 1
            continue
        end
        v = _var_bcmc_given_N(sigma2_f, rho_f, N, T)
        if v < bestV
            bestN, bestV = N, v
        end
    end
    return bestN, bestV
end

end # module
