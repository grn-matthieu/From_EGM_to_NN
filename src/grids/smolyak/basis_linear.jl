function _canonicalize_point(x::AbstractVector, box::Vector{Tuple{Float64,Float64}})
    dim = length(box)
    canonical = Vector{Float64}(undef, dim)
    for i = 1:dim
        lo, hi = box[i]
        mid = (hi + lo) / 2
        half = (hi - lo) / 2
        half == 0 && error("degenerate bound interval in dimension $i")
        canonical[i] = (x[i] - mid) / half
    end
    return canonical
end

function _basis_1d(level::Int, center::Float64, left::Float64, right::Float64, x::Float64)
    if level == 1
        return 1.0
    end
    if !isfinite(left) && !isfinite(right)
        return 1.0
    elseif !isfinite(left)
        if x < center - SMOLYAK_TOL || x > right + SMOLYAK_TOL
            return 0.0
        elseif x <= center
            return 1.0
        else
            return (right - x) / (right - center)
        end
    elseif !isfinite(right)
        if x > center + SMOLYAK_TOL || x < left - SMOLYAK_TOL
            return 0.0
        elseif x >= center
            return 1.0
        else
            return (x - left) / (center - left)
        end
    else
        if x < left - SMOLYAK_TOL || x > right + SMOLYAK_TOL
            return 0.0
        elseif x <= center
            return (x - left) / (center - left)
        else
            return (right - x) / (right - center)
        end
    end
end

function _basis_value(meta::NodeMeta{D}, x::AbstractVector) where {D}
    value = 1.0
    for i = 1:D
        value *=
            _basis_1d(meta.levels[i], meta.centers[i], meta.lefts[i], meta.rights[i], x[i])
        if value == 0.0
            return 0.0
        end
    end
    return value
end

function fit_interpolant!(G::SmolyakGrid, values::AbstractMatrix)
    G.spec.basis == :linear || error("Only :linear basis supported")
    n_nodes = size(G.state.nodes, 1)
    size(values, 1) == n_nodes || error("values must match number of nodes")
    n_outputs = size(values, 2)
    coeffs = zeros(n_nodes, n_outputs)
    structure = G.state.structure
    canonical = structure.canonical
    meta = structure.meta
    order = structure.order
    scratch = zeros(n_outputs)
    for pos = 1:length(order)
        idx = order[pos]
        fill!(scratch, 0.0)
        x = view(canonical, idx, :)
        for prev = 1:(pos-1)
            pid = order[prev]
            basis_val = _basis_value(meta[pid], x)
            if basis_val != 0.0
                for j = 1:n_outputs
                    scratch[j] += basis_val * coeffs[pid, j]
                end
            end
        end
        for j = 1:n_outputs
            coeffs[idx, j] = values[idx, j] - scratch[j]
        end
    end
    G.state.coeffs = coeffs
    return G
end

function evaluate_scalar!(G::SmolyakGrid, x::AbstractVector, j::Int)
    coeffs = G.state.coeffs
    size(coeffs, 1) > 0 || error("grid not fitted")
    canonical = _canonicalize_point(x, G.spec.box)
    meta = G.state.structure.meta
    total = 0.0
    for idx = 1:length(meta)
        basis_val = _basis_value(meta[idx], canonical)
        if basis_val != 0.0
            total += coeffs[idx, j] * basis_val
        end
    end
    return total
end
