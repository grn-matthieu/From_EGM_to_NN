include("dense.jl")
include("smolyak/factory.jl")

function build_grid_backend(box::Vector{Tuple{Float64,Float64}}, cfg)
    grid_cfg = cfg.solver.grid
    grid_type = grid_cfg.type isa Symbol ? grid_cfg.type : Symbol(grid_cfg.type)
    empty_nt = NamedTuple{(),Tuple{}}()
    if grid_type == :dense
        dense_cfg = hasproperty(grid_cfg, :dense) ? grid_cfg.dense : nothing
        return DenseGrid(box, dense_cfg)
    elseif grid_type in (:sparse, :adaptive_sparse)
        sparse_cfg = hasproperty(grid_cfg, :sparse) ? grid_cfg.sparse : empty_nt
        depth = hasproperty(sparse_cfg, :depth) ? sparse_cfg.depth : 2
        basis = hasproperty(sparse_cfg, :basis) ? sparse_cfg.basis : :linear
        anisotropic = hasproperty(sparse_cfg, :anisotropic) ? sparse_cfg.anisotropic : false
        adaptive = grid_type == :adaptive_sparse
        adaptive_cfg = hasproperty(grid_cfg, :adaptive) ? grid_cfg.adaptive : empty_nt
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
        error("unknown grid type $(grid_cfg.type)")
    end
end
