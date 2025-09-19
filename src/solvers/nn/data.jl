"""
NNData

Mini-batch and dataset utilities for grid-based training of NN policies.
Provides iterators over asset/shock grids with optional shuffling and targets.
"""
module NNData

using Random
using ..Determinism: make_rng
using ..NNDevice: to_device

export grid_minibatches

# Internal iterator type for grid minibatches
struct GridMB{TA,TY,TT,TR,TEl}
    a::TA
    y::TY
    targets::TT
    batch::Int
    shuffle::Bool
    rng::TR
    Na::Int
    Ny::Int
    N::Int
    drop_last::Bool
    antithetic::Bool
    device::Symbol
end

Base.length(it::GridMB) = it.drop_last ? div(it.N, it.batch) : cld(it.N, it.batch)

function Base.iterate(it::GridMB{TA,TY,TT,TR,TEl}) where {TA,TY,TT,TR,TEl}
    ord = it.shuffle ? randperm(it.rng, it.N) : collect(1:it.N)
    _iterate_with_state(it, ord, 1)
end

function Base.iterate(it::GridMB{TA,TY,TT,TR,TEl}, st) where {TA,TY,TT,TR,TEl}
    ord, pos = st
    _iterate_with_state(it, ord, pos)
end

function _iterate_with_state(
    it::GridMB{TA,TY,TT,TR,TEl},
    ord,
    pos::Int,
) where {TA,TY,TT,TR,TEl}
    if pos > length(ord)
        return nothing
    end
    last = it.drop_last ? (pos + it.batch - 1) : min(pos + it.batch - 1, length(ord))
    if last > length(ord)
        return nothing
    end
    idxs = @view ord[pos:last]
    B = length(idxs)

    ia = @. ((idxs - 1) % it.Na) + 1
    iy = fld.((idxs .- 1), it.Na) .+ 1

    if it.antithetic && it.Ny > 1
        iyt = @. (it.Ny - iy + 1)
        self = @. (iy == iyt)
        add_idx = findall(!, self)
        if !isempty(add_idx)
            ia = vcat(ia, ia[add_idx])
            iy = vcat(iy, iyt[add_idx])
        end
        B = length(ia)
    end

    X = Array{TEl}(undef, 2, B)
    @inbounds begin
        X[1, :] = it.a[ia]
        X[2, :] = it.y[iy]
    end

    Y = Array{TEl}(undef, 1, B)
    if it.targets === nothing
        fill!(Y, convert(TEl, NaN))
    else
        @inbounds Y[1, :] = it.targets[CartesianIndex.(ia, iy)]
    end

    # Move batch to device if requested
    if it.device !== :cpu
        try
            Xd = to_device(X, it.device)
            Yd = to_device(Y, it.device)
            return (Xd, Yd), (ord, last + 1)
        catch
            return (X, Y), (ord, last + 1)
        end
    end

    return (X, Y), (ord, last + 1)
end

"""
    grid_minibatches(
        a_grid::AbstractVector,
        y_grid::AbstractVector;
        targets::Union{Nothing,AbstractMatrix}=nothing,
        batch::Integer=128,
        shuffle::Bool=true,
        rng=nothing,
        seed::Union{Nothing,Integer}=nothing,
        drop_last::Bool=false,
        antithetic::Bool=false,
    ) -> iterator

Creates a stateful iterator that yields minibatches over the Cartesian product
of `a_grid x y_grid`.

Each iteration returns a pair `(X, Y)` where:
  - `X` is a `2 x B` matrix with first row `a` and second row `y` for `B` points
  - `Y` is a `1 x B` matrix of targets if `targets` is provided (size `(Na, Ny)`),
    otherwise a `1 x B` matrix filled with `NaN`.

If `antithetic=true`, each batch is augmented with mirrored y-index pairs
`(a_i, y_j) => (a_i, y_{Ny-j+1})` when `Ny>1`. Midpoint indices (when
`2j == Ny+1`) are not duplicated. Note: this can increase the effective batch
size up to roughly 2x the requested `batch`.

Ordering traverses `a` fastest by default when not shuffled. If `shuffle=true`,
each fresh iteration over the returned object re-draws a new random permutation.

Set `drop_last=true` to drop a final partial batch.
"""
function grid_minibatches(
    a_grid::AbstractVector,
    y_grid::AbstractVector;
    targets::Union{Nothing,AbstractMatrix} = nothing,
    batch::Integer = 128,
    shuffle::Bool = true,
    rng = nothing,
    seed::Union{Nothing,Integer} = nothing,
    drop_last::Bool = false,
    antithetic::Bool = false,
    device::Symbol = :cpu,
)
    Na = length(a_grid)
    Ny = length(y_grid)
    N = Na * Ny

    if targets !== nothing
        @assert size(targets, 1) == Na "targets must have size (Na, Ny)"
        @assert size(targets, 2) == Ny "targets must have size (Na, Ny)"
    end

    R =
        rng !== nothing ? rng :
        (seed === nothing ? Random.default_rng() : make_rng(Int(seed)))

    elT = promote_type(
        eltype(a_grid),
        eltype(y_grid),
        targets === nothing ? Float64 : eltype(targets),
    )
    return GridMB{typeof(a_grid),typeof(y_grid),typeof(targets),typeof(R),elT}(
        a_grid,
        y_grid,
        targets,
        Int(batch),
        shuffle,
        R,
        Na,
        Ny,
        N,
        drop_last,
        antithetic,
        device,
    )
end

end # module
