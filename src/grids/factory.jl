include("dense.jl")
include("smolyak/factory.jl")

@inline function _dense_count_specified(cfg)
    cfg === nothing && return false
    for key in (:counts, :nodes_per_dim, :N, :n)
        if hasproperty(cfg, key)
            return true
        end
    end
    return false
end

function _normalize_dense_cfg(dense_cfg, dim::Int, cfg)
    dense_nt = dense_cfg === nothing ? NamedTuple() : dense_cfg
    if !_dense_count_specified(dense_nt)
        if hasproperty(cfg, :grids)
            grids_meta = cfg.grids
            target_key = hasproperty(grids_meta, :Na_backend) ? :Na_backend : :Na
            if target_key !== nothing && hasproperty(grids_meta, target_key)
                Na = Int(getproperty(grids_meta, target_key))
                dense_nt = merge(dense_nt, (counts = fill(Na, dim),))
            end
        end
    end
    return fieldcount(typeof(dense_nt)) == 0 ? nothing : dense_nt
end

function build_grid_backend(box::Vector{Tuple{Float64,Float64}}, cfg)
    # Read grid configuration from cfg.grids (flat fields)
    hasproperty(cfg, :grids) || error("missing grids section")
    g = cfg.grids
    hasproperty(g, :type) || error("missing grids.type")
    grid_type = g.type isa Symbol ? g.type : Symbol(g.type)
    empty_nt = NamedTuple()
    if grid_type == :dense
        dense_cfg_raw = hasproperty(g, :dense) ? g.dense : nothing
        dense_cfg = _normalize_dense_cfg(dense_cfg_raw, length(box), cfg)
        return DenseGrid(box, dense_cfg)
    elseif grid_type in (:sparse, :adaptive_sparse)
        sparse_cfg = hasproperty(g, :sparse) ? g.sparse : empty_nt
        depth = hasproperty(sparse_cfg, :depth) ? sparse_cfg.depth : 2
        basis = hasproperty(sparse_cfg, :basis) ? sparse_cfg.basis : :linear
        anisotropic = hasproperty(sparse_cfg, :anisotropic) ? sparse_cfg.anisotropic : false
        adaptive = grid_type == :adaptive_sparse
        adaptive_cfg = hasproperty(g, :adaptive) ? g.adaptive : empty_nt
        surplus_tol =
            hasproperty(adaptive_cfg, :surplus_tol) ? adaptive_cfg.surplus_tol : 1e-3
        return SmolyakGrid(
            box;
            depth = depth,
            basis = basis,
            anisotropic = anisotropic,
            adaptive = adaptive,
            surplus_tol = surplus_tol,
        )
    else
        error("unknown grid type $(get(g, :type, :dense))")
    end
end
