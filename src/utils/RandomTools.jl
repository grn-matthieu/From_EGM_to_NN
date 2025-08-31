module UtilsRandom

export make_rng

using StableRNGs

"""
    make_rng(seed::Integer) -> StableRNG

Creates and returns a StableRNG seeded with `seed`.
"""
function make_rng(seed::Integer)
    return StableRNG(seed)
end

end