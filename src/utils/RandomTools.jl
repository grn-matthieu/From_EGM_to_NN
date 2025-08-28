module UtilsRandom
export set_global_seed
using StableRNGs

function set_global_seed(seed::Integer)
    return StableRNG(seed)
end

end