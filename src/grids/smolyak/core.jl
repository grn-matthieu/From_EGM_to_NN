const SMOLYAK_TOL = 1e-12

struct GridSpec
    depth::Int
    dim::Int
    basis::Symbol
    anisotropic::Bool
    box::Vector{Tuple{Float64,Float64}}
end

mutable struct GridState
    nodes::Matrix{Float64}
    structure::Any
    coeffs::Matrix{Float64}
end

struct Node1D
    level::Int
    index::Int
    coord::Float64
    left::Float64
    right::Float64
end

struct NodeMeta{D}
    levels::NTuple{D,Int}
    centers::NTuple{D,Float64}
    lefts::NTuple{D,Float64}
    rights::NTuple{D,Float64}
end

struct SmolyakStructure{D}
    canonical::Matrix{Float64}
    meta::Vector{NodeMeta{D}}
    order::Vector{Int}
    index_set::Vector{NTuple{D,Int}}
    lookup::Dict{NTuple{D,Float64},Int}
end

struct SmolyakGrid <: AbstractGridBackend
    spec::GridSpec
    state::GridState
    adaptive::Bool
    surplus_tol::Float64
end

nodes(G::SmolyakGrid) = G.state.nodes
bounds(G::SmolyakGrid) = G.spec.box

function build_smolyak_nodes(spec::GridSpec; index_set = nothing)
    dim = spec.dim
    depth = spec.depth
    if index_set === nothing
        index_set = build_smolyak_indices(dim, depth)
    else
        index_set = sort!(collect(index_set); lt = (a, b) -> smolyak_index_lt(a, b))
    end
    isempty(index_set) && return (
        zeros(0, dim),
        SmolyakStructure{dim}(
            zeros(0, dim),
            NodeMeta{dim}[],
            Int[],
            NTuple{dim,Int}[],
            Dict{NTuple{dim,Float64},Int}(),
        ),
    )
    max_level = maximum(maximum.(index_set))
    oned = build_1d_hierarchy(max_level)
    meta = NodeMeta{dim}[]
    canonical_coords = Vector{NTuple{dim,Float64}}()
    temp_lookup = Dict{NTuple{dim,Float64},Int}()
    for multi in index_set
        one_sets = map(l -> oned[l], multi)
        any(isempty, one_sets) && continue
        for combo in Base.Iterators.product(one_sets...)
            center = ntuple(i -> combo[i].coord, dim)
            key = _canonical_key(center)
            haskey(temp_lookup, key) && continue
            levels = ntuple(i -> combo[i].level, dim)
            lefts = ntuple(i -> combo[i].left, dim)
            rights = ntuple(i -> combo[i].right, dim)
            push!(meta, NodeMeta{dim}(levels, center, lefts, rights))
            push!(canonical_coords, center)
            temp_lookup[key] = length(meta)
        end
    end
    n = length(meta)
    canonical = Matrix{Float64}(undef, n, dim)
    for i = 1:n
        for j = 1:dim
            canonical[i, j] = canonical_coords[i][j]
        end
    end
    perm = sortperm(1:n; lt = (a, b) -> _lex_lt(canonical_coords[a], canonical_coords[b]))
    canonical = canonical[perm, :]
    meta = meta[perm]
    canonical_coords = canonical_coords[perm]
    lookup = Dict{NTuple{dim,Float64},Int}()
    for (idx, coord) in enumerate(canonical_coords)
        lookup[_canonical_key(coord)] = idx
    end
    order = sortperm(
        1:n;
        lt = (a, b) -> _hier_lt(meta[a], meta[b], canonical[a, :], canonical[b, :]),
    )
    nodes = _map_to_physical(canonical, spec.box)
    structure = SmolyakStructure{dim}(canonical, meta, order, collect(index_set), lookup)
    return nodes, structure
end

function build_smolyak_indices(dim::Int, depth::Int)
    indices = Vector{NTuple{dim,Int}}()
    current = Vector{Int}(undef, dim)
    function _recurse(pos::Int, remaining::Int)
        if pos > dim
            push!(indices, Tuple(current))
            return
        end
        for lev = 1:(remaining+1)
            current[pos] = lev
            _recurse(pos + 1, remaining - (lev - 1))
        end
    end
    _recurse(1, depth)
    sort!(indices; lt = (a, b) -> smolyak_index_lt(a, b))
    return indices
end

function smolyak_index_lt(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N}
    sa = sum(a)
    sb = sum(b)
    if sa != sb
        return sa < sb
    end
    for i = 1:N
        ai = a[i]
        bi = b[i]
        if ai != bi
            return ai < bi
        end
    end
    return false
end

function build_1d_hierarchy(max_level::Int)
    existing = Float64[]
    meta = Vector{Vector{Node1D}}(undef, max_level)
    for level = 1:max_level
        xs = clenshaw_curtis_nodes(level)
        sort!(xs)
        level_nodes = Node1D[]
        for (idx, x) in enumerate(xs)
            if _has_close(existing, x)
                continue
            end
            left, right = _neighbors(existing, x)
            push!(level_nodes, Node1D(level, idx, x, left, right))
        end
        existing = _merge_unique(existing, xs)
        meta[level] = level_nodes
    end
    return meta
end

function clenshaw_curtis_nodes(level::Int)
    if level <= 1
        return [0.0]
    end
    n = 2^(level - 1) + 1
    js = collect(0:(n-1))
    xs = cospi.(js ./ (n - 1))
    return xs
end

function _has_close(existing::Vector{Float64}, x::Float64)
    isempty(existing) && return false
    pos = searchsortedfirst(existing, x)
    if pos <= length(existing) && abs(existing[pos] - x) <= SMOLYAK_TOL
        return true
    end
    if pos > 1 && abs(existing[pos-1] - x) <= SMOLYAK_TOL
        return true
    end
    return false
end

function _neighbors(existing::Vector{Float64}, x::Float64)
    if isempty(existing)
        return (NaN, NaN)
    end
    pos = searchsortedfirst(existing, x)
    left = pos > 1 ? existing[pos-1] : NaN
    right = pos <= length(existing) ? existing[pos] : NaN
    return (left, right)
end

function _merge_unique(existing::Vector{Float64}, xs::Vector{Float64})
    combined = sort(vcat(existing, xs))
    result = Float64[]
    for val in combined
        if isempty(result) || abs(result[end] - val) > SMOLYAK_TOL
            push!(result, val)
        end
    end
    return result
end

function _canonical_key(coord::NTuple{N,Float64}) where {N}
    return ntuple(i -> round(coord[i], digits = 12), N)
end

function _lex_lt(a::NTuple{N,Float64}, b::NTuple{N,Float64}) where {N}
    for i = 1:N
        if a[i] < b[i] - SMOLYAK_TOL
            return true
        elseif a[i] > b[i] + SMOLYAK_TOL
            return false
        end
    end
    return false
end

function _hier_lt(
    meta_a::NodeMeta{N},
    meta_b::NodeMeta{N},
    ca::AbstractVector,
    cb::AbstractVector,
) where {N}
    sa = sum(meta_a.levels)
    sb = sum(meta_b.levels)
    if sa != sb
        return sa < sb
    end
    for i = 1:N
        la = meta_a.levels[i]
        lb = meta_b.levels[i]
        if la != lb
            return la < lb
        end
    end
    for i = 1:N
        if ca[i] < cb[i] - SMOLYAK_TOL
            return true
        elseif ca[i] > cb[i] + SMOLYAK_TOL
            return false
        end
    end
    return false
end

function _map_to_physical(canonical::Matrix{Float64}, box::Vector{Tuple{Float64,Float64}})
    n, dim = size(canonical)
    nodes = Matrix{Float64}(undef, n, dim)
    for j = 1:dim
        lo, hi = box[j]
        mid = (hi + lo) / 2
        half = (hi - lo) / 2
        for i = 1:n
            nodes[i, j] = mid + half * canonical[i, j]
        end
    end
    return nodes
end
