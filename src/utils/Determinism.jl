"""
Determinism

Utilities to make runs reproducible: seeded RNGs, deterministic seeds derived
from structured inputs, and helpers for hashing/serialization.
"""
module Determinism

using StableRNGs
using JSON3
using SHA

export make_rng, canonicalize_cfg, hash_hex, derive_seed


"""
    make_rng(seed::Integer)

Creates a StableRNGs with the given seed (integer).
"""
make_rng(seed::Integer) = StableRNG(seed)



"""
    canonicalize_cfg(cfg)::Vector{UInt8}

Serializes a config dictionary to sorted, symbol-keyed, fixed-precision JSON bytes.
"""
function canonicalize_cfg(cfg)
    # Recursively convert keys to Symbol and sort
    function canonical(obj)
        if obj isa AbstractDict
            # Sort keys as symbols
            pairs = sort(collect(obj); by = x -> Symbol(x[1]))
            Dict(Symbol(k) => canonical(v) for (k, v) in pairs)
        elseif obj isa AbstractArray
            map(canonical, obj)
        elseif obj isa AbstractFloat
            # Fixed precision (8 decimals)
            round(obj, digits = 8)
        else
            obj
        end
    end
    can_cfg = canonical(cfg)
    # Use JSON3 with sorted keys and no whitespace
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


"""
    derive_seed(master, key)::Int

Derives a 64-bit integer seed from a master rng and key.
"""
function derive_seed(master, key)::UInt64
    hex = bytes2hex(sha256(string(master, ":", key)))
    nhex = 2 * sizeof(UInt)                               # 16 on 64-bit, 8 on 32-bit
    u = parse(UInt, hex[1:nhex]; base = 16)
    return u
end

end # module
