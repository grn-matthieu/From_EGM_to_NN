module NNData

using Random

export grid_minibatches

"""
    grid_minibatches(
        a_grid::AbstractVector,
        y_grid::AbstractVector;
        targets::Union{Nothing,AbstractMatrix}=nothing,
        batch::Integer=128,
        shuffle::Bool=true,
        rng=nothing,
        drop_last::Bool=false,
    ) -> iterator

Creates a stateful iterator that yields minibatches over the Cartesian product
of `a_grid × y_grid`.

Each iteration returns a pair `(X, Y)` where:
  - `X` is a `2 × B` matrix with first row `a` and second row `y` for `B` points
  - `Y` is a `1 × B` matrix of targets if `targets` is provided (size `(Na, Ny)`),
    otherwise a `1 × B` matrix filled with `NaN`.

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
    drop_last::Bool = false,
)
    Na = length(a_grid)
    Ny = length(y_grid)
    N = Na * Ny

    if targets !== nothing
        @assert size(targets, 1) == Na "targets must have size (Na, Ny)"
        @assert size(targets, 2) == Ny "targets must have size (Na, Ny)"
    end

    R = rng === nothing ? Random.default_rng() : rng

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
    end

    Base.length(it::_GridMB) = it.drop_last ? (it.N ÷ it.batch) : ceil(Int, it.N / it.batch)

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
        iy = @. ((idxs - 1) ÷ it.Na) + 1

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
    )
end

end # module
