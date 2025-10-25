include("core.jl")
include("basis_linear.jl")
include("refine.jl")
include("quad_sparse.jl")

function SmolyakGrid(
    box::Vector{Tuple{Float64,Float64}};
    depth::Int = 2,
    basis::Symbol = :linear,
    anisotropic::Bool = false,
    adaptive::Bool = false,
    surplus_tol::Float64 = 1e-3,
)
    spec = GridSpec(depth, length(box), basis, anisotropic, box)
    nodes, structure = build_smolyak_nodes(spec)
    state = GridState(nodes, structure, zeros(0, 0))
    return SmolyakGrid(spec, state, adaptive, surplus_tol)
end
