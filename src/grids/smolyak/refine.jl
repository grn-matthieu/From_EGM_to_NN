function refine_once!(G::SmolyakGrid; tol::Float64 = 0.0)
    coeffs = G.state.coeffs
    size(coeffs, 1) == size(G.state.nodes, 1) || error("grid state inconsistent")
    size(coeffs, 1) == 0 && return G
    size(coeffs, 2) == 0 && return G
    tol_eff = tol > 0 ? tol : max(G.surplus_tol, 0.0)
    tol_eff == 0.0 && return G
    dim = G.spec.dim
    structure = G.state.structure
    maxabs = [maximum(abs.(coeffs[i, :])) for i = 1:size(coeffs, 1)]
    flagged = findall(x -> x > tol_eff, maxabs)
    isempty(flagged) && return G
    new_indices = Set{NTuple{dim,Int}}()
    for idx in flagged
        levels = structure.meta[idx].levels
        for d = 1:dim
            new_lv = ntuple(k -> k == d ? levels[k] + 1 : levels[k], dim)
            push!(new_indices, new_lv)
        end
    end
    isempty(new_indices) && return G
    existing = Set(structure.index_set)
    union_set = union(existing, new_indices)
    if length(union_set) == length(existing)
        return G
    end
    nodes, new_structure = build_smolyak_nodes(G.spec; index_set = collect(union_set))
    n_new = size(nodes, 1)
    n_outputs = size(coeffs, 2)
    new_coeffs = zeros(n_new, n_outputs)
    for (key, old_idx) in structure.lookup
        if haskey(new_structure.lookup, key)
            new_idx = new_structure.lookup[key]
            new_coeffs[new_idx, :] .= coeffs[old_idx, :]
        end
    end
    G.state.nodes = nodes
    G.state.structure = new_structure
    G.state.coeffs = new_coeffs
    return G
end
