using LinearAlgebra: SymTridiagonal, eigen

function gh_sparse_nodes_weights(d::Int, order::Int)
    d >= 1 || error("dimension must be positive")
    order >= 1 || error("order must be ≥ 1")
    if order == 1
        return (zeros(1, d), [1.0])
    end
    depth = order - 1
    index_set = build_smolyak_indices(d, depth)
    cache = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    acc = Dict{NTuple{d,Float64},Float64}()
    for multi in index_set
        coeff = _smolyak_weight(multi, depth, d)
        coeff == 0.0 && continue
        nodes_1d = Vector{Vector{Float64}}(undef, d)
        weights_1d = Vector{Vector{Float64}}(undef, d)
        for k = 1:d
            lvl = multi[k]
            rule = get!(cache, lvl) do
                gauss_hermite_rule(lvl)
            end
            nodes_1d[k] = rule[1]
            weights_1d[k] = rule[2]
        end
        ranges = ntuple(k -> 1:length(nodes_1d[k]), d)
        for idxs in Base.Iterators.product(ranges...)
            node = ntuple(k -> nodes_1d[k][idxs[k]], d)
            weight = coeff
            for k = 1:d
                weight *= weights_1d[k][idxs[k]]
            end
            key = _canonical_key(node)
            acc[key] = get(acc, key, 0.0) + weight
        end
    end
    keys_sorted = sort(collect(keys(acc)); lt = (a, b) -> _lex_lt(a, b))
    filtered = [(key, acc[key]) for key in keys_sorted if abs(acc[key]) > 1e-14]
    if isempty(filtered)
        filtered = [(ntuple(_ -> 0.0, d), 0.0)]
    end
    n = length(filtered)
    nodes = Matrix{Float64}(undef, n, d)
    weights = Vector{Float64}(undef, n)
    for (i, (key, weight)) in enumerate(filtered)
        weights[i] = weight
        for j = 1:d
            nodes[i, j] = key[j]
        end
    end
    total_w = sum(weights)
    if abs(total_w) > 0
        weights ./= total_w
    end
    return nodes, weights
end

function gauss_hermite_rule(level::Int)
    level >= 1 || error("level must be ≥ 1")
    if level == 1
        return ([0.0], [1.0])
    end
    diag = zeros(level)
    off = sqrt.(collect(1:(level-1)) ./ 2)
    T = SymTridiagonal(diag, off)
    vals, vecs = eigen(T)
    nodes = sqrt(2.0) .* collect(vals)
    weights = vecs[1, :] .^ 2
    return (nodes, collect(weights))
end

function _smolyak_weight(multi::NTuple{N,Int}, depth::Int, dim::Int) where {N}
    total = sum(multi)
    r = depth - (total - dim)
    if r < 0 || r > dim - 1
        return 0.0
    end
    return (-1.0)^r * _binomial(dim - 1, r)
end

function _binomial(n::Int, k::Int)
    if k < 0 || k > n
        return 0.0
    end
    k = min(k, n - k)
    result = 1.0
    for i = 1:k
        result *= (n - k + i) / i
    end
    return result
end
