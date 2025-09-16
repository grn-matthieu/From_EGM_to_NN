module NNData

using Random
using ..Determinism: make_rng

export grid_minibatches

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

    struct _GridMB{TA,TY,TT,TR}
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
        elT::DataType
        antithetic::Bool
    end

    Base.length(it::_GridMB) = it.drop_last ? div(it.N, it.batch) : cld(it.N, it.batch)

    function Base.iterate(it::_GridMB)
        ord = it.shuffle ? randperm(it.rng, it.N) : collect(1:it.N)
        _iterate_with_state(it, ord, 1)
    end

    function Base.iterate(it::_GridMB, st)
        ord, pos = st
        _iterate_with_state(it, ord, pos)
    end

    function _iterate_with_state(it::_GridMB, ord, pos::Int)
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

        X = Array{it.elT}(undef, 2, B)
        @inbounds begin
            X[1, :] = it.a[ia]
            X[2, :] = it.y[iy]
        end

        Y = Array{it.elT}(undef, 1, B)
        if it.targets === nothing
            fill!(Y, convert(it.elT, NaN))
        else
            @inbounds Y[1, :] = it.targets[CartesianIndex.(ia, iy)]
        end

        return (X, Y), (ord, last + 1)
    end

    return _GridMB(
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
        elT,
        antithetic,
    )
end

end # module
