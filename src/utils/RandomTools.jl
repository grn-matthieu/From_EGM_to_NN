module UtilsRandom
export set_global_seed
using Random

function set_global_seed(seed::Integer)
    Random.seed!(seed)
end

end