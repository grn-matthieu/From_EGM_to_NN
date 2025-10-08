"""
Determinism

Utilities to make runs reproducible: seeded RNGs, deterministic seeds derived
from structured inputs, and helpers for hashing/serialization.
"""
module Determinism

using StableRNGs
using JSON3
using Random
using SHA

export MasterRNG,
    make_rng,
    make_master_rng,
    canonicalize_cfg,
    hash_hex,
    derive_seed,
    derive_rng,
    master_seed,
    promote_master_rng


"""
    make_rng(seed::Integer)

Creates a StableRNGs generator with the given seed.
"""
make_rng(seed::Integer) = StableRNG(seed)

"""
    MasterRNG

Lightweight wrapper that stores an immutable master seed. Master RNGs are
never mutated directly; instead, deterministic sub-generators are derived via
[`derive_rng`] or [`derive_seed`].
"""
struct MasterRNG
    seed::UInt64
end

"""Return the canonical 64-bit seed stored in a [`MasterRNG`]."""
master_seed(master::MasterRNG) = master.seed

"""Internal helper: attempt to copy an RNG without mutating the original."""
function _copy_rng(rng::AbstractRNG)
    try
        return copy(rng)
    catch
        return deepcopy(rng)
    end
end

"""
    make_master_rng(seed_or_rng)

Create a [`MasterRNG`] from an integer seed or RNG object. When an RNG is
provided, its state is copied so the original RNG remains untouched.
"""
make_master_rng(seed::Integer) = MasterRNG(UInt64(seed))
function make_master_rng(rng::AbstractRNG)
    rng_copy = _copy_rng(rng)
    # Use a deterministic draw from the copied RNG to capture its state.
    return MasterRNG(rand(rng_copy, UInt64))
end

"""Promote arbitrary master RNG inputs to a [`MasterRNG`] instance."""
function promote_master_rng(master)
    if master isa MasterRNG
        return master
    elseif master === nothing
        error("master RNG not available; ensure the configuration defines random.seed")
    elseif master isa Integer
        return make_master_rng(master)
    elseif master isa AbstractRNG
        return make_master_rng(master)
    else
        error("unsupported master RNG type $(typeof(master))")
    end
end



"""
    canonicalize_cfg(cfg)::Vector{UInt8}

Serializes a configuration object to sorted, symbol-keyed, fixed-precision JSON bytes.
"""
function canonicalize_cfg(cfg)
    # Recursively convert keys to Symbol and sort
    function canonical(obj)
        if obj isa NamedTuple
            # Convert to pairs, canonicalize values, sort by key
            ks = collect(Symbol.(keys(obj)))
            ks = sort(ks)
            vs = map(k -> canonical(getfield(obj, k)), ks)
            NamedTuple{Tuple(ks)}(vs)
        elseif obj isa AbstractDict
            ks = collect(Symbol.(keys(obj)))
            ks = sort(ks)
            Dict(k => canonical(obj[k]) for k in ks)
        elseif obj isa AbstractArray
            map(canonical, obj)
        elseif obj isa AbstractFloat
            round(obj, digits = 8)
        else
            obj
        end
    end
    can_cfg = canonical(cfg)
    # JSON3 already canonicalizes NamedTuple key order
    json_str = JSON3.write(can_cfg; canonical = true)
    Vector{UInt8}(codeunits(json_str))
end


"""
    hash_hex(bytes; n=12)::String

Computes SHA256 hash of bytes and return first `n` hex chars.
"""
function hash_hex(bytes; n = 12)
    hex = bytes2hex(sha256(bytes))
    hex[1:n]
end


const _SEED_NHEX = 2 * sizeof(UInt64)

function _seed_key_bytes(seed::UInt64, key)
    buf = IOBuffer()
    write(buf, seed)
    write(buf, UInt8(':'))
    key_str = key isa AbstractString ? key : string(key)
    write(buf, key_str)
    return take!(buf)
end

"""
    derive_seed(master, key)::UInt64

Derive a deterministic 64-bit seed from a master RNG (or seed) and a key.
"""
function derive_seed(master::MasterRNG, key)::UInt64
    return derive_seed(master.seed, key)
end

function derive_seed(master::Integer, key)::UInt64
    seed = UInt64(master)
    hex = bytes2hex(sha256(_seed_key_bytes(seed, key)))
    return parse(UInt64, hex[1:_SEED_NHEX]; base = 16)
end

function derive_seed(master::AbstractRNG, key)::UInt64
    promoted = make_master_rng(master)
    return derive_seed(promoted, key)
end

"""Fallback for unsupported master types."""
function derive_seed(master, key)::UInt64
    error("unsupported master RNG type $(typeof(master)) for derive_seed")
end

"""Derive a StableRNG sub-generator from a master RNG and key."""
derive_rng(master, key) = make_rng(derive_seed(master, key))

end # module
