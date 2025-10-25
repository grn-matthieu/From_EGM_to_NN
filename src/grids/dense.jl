# Dense tensor-product grid backend with multilinear interpolation.

struct DenseGrid <: AbstractGridBackend
    box::Vector{Tuple{Float64,Float64}}
    axes::Vector{Vector{Float64}}
    nodes::Matrix{Float64}
    strides::Vector{Int}
    values::Matrix{Float64}
end

function DenseGrid(box::Vector{Tuple{Float64,Float64}}, cfg)
    dim = length(box)
    counts = _dense_counts(cfg, dim)
    axes = Vector{Vector{Float64}}(undef, dim)
    for i = 1:dim
        lo, hi = box[i]
        axes[i] = collect(range(lo, hi; length = counts[i]))
    end
    nodes, strides = _build_dense_nodes(axes)
    values = zeros(0, 0)
    return DenseGrid(box, axes, nodes, strides, values)
end

function _dense_counts(cfg, dim)
    if cfg === nothing
        return fill(5, dim)
    end
    if hasproperty(cfg, :counts)
        return collect(getproperty(cfg, :counts))
    elseif hasproperty(cfg, :nodes_per_dim)
        return collect(getproperty(cfg, :nodes_per_dim))
    elseif hasproperty(cfg, :N)
        N = getproperty(cfg, :N)
        return N isa Integer ? fill(Int(N), dim) : collect(N)
    elseif hasproperty(cfg, :n)
        return fill(Int(getproperty(cfg, :n)), dim)
    else
        return fill(5, dim)
    end
end

function _build_dense_nodes(axes::Vector{Vector{Float64}})
    dim = length(axes)
    counts = map(length, axes)
    total = prod(counts)
    nodes = Matrix{Float64}(undef, total, dim)
    strides = Vector{Int}(undef, dim)
    stride = 1
    for d = dim:-1:1
        strides[d] = stride
        stride *= counts[d]
    end
    idx = Ref(1)
    coord = zeros(dim)
    function _fill!(level::Int)
        if level > dim
            nodes[idx[], :] .= coord
            idx[] += 1
            return
        end
        for val in axes[level]
            coord[level] = val
            _fill!(level + 1)
        end
    end
    _fill!(1)
    return nodes, strides
end

nodes(G::DenseGrid) = G.nodes
bounds(G::DenseGrid) = G.box

function fit_interpolant!(G::DenseGrid, values::AbstractMatrix)
    size(values, 1) == size(G.nodes, 1) || error("values must match number of grid nodes")
    G.values = Matrix{Float64}(values)
    return G
end

function evaluate_scalar!(G::DenseGrid, x::AbstractVector, j::Int)
    size(G.values, 1) == size(G.nodes, 1) || error("grid coefficients not fitted")
    dim = length(G.axes)
    lower = Vector{Int}(undef, dim)
    upper = Vector{Int}(undef, dim)
    weights = Vector{Float64}(undef, dim)
    for d = 1:dim
        axis = G.axes[d]
        n = length(axis)
        xd = x[d]
        if xd <= axis[1]
            lower[d] = upper[d] = 1
            weights[d] = 0.0
        elseif xd >= axis[end]
            lower[d] = upper[d] = n
            weights[d] = 0.0
        else
            pos = searchsortedlast(axis, xd)
            lower[d] = pos
            upper[d] = pos + 1
            weights[d] = (xd - axis[pos]) / (axis[pos+1] - axis[pos])
        end
    end
    result = 0.0
    total = 1 << dim
    for mask = 0:(total-1)
        weight = 1.0
        idx = 1
        for d = 1:dim
            if lower[d] == upper[d]
                idx += (lower[d] - 1) * G.strides[d]
            else
                if (mask >> (d - 1)) & 1 == 1
                    weight *= weights[d]
                    idx += (upper[d] - 1) * G.strides[d]
                else
                    weight *= (1 - weights[d])
                    idx += (lower[d] - 1) * G.strides[d]
                end
            end
        end
        result += weight * G.values[idx, j]
    end
    return result
end

function refine_once!(G::DenseGrid; tol::Float64 = 0.0)
    return G
end
