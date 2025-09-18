using ThesisProject

# Build model and compute EGM policy on the grid, then expose an interpolant

function _build_egm_policy(cfg)
    m = ThesisProject.API.build_model(cfg)
    g = ThesisProject.API.get_grids(m)
    # Run EGM kernel to get a'(a,y). Here assume deterministic income for simplicity.
    sol = ThesisProject.SolverEGM.solve_egm(m)
    # Expect sol to contain next assets on grid; fall back to c if needed
    a_grid = g[:a].grid
    if haskey(sol, :a_next)
        ap_grid = sol[:a_next]
    elseif haskey(sol, :c)
        p = m.p
        R = 1 + p.r
        y = get(p, :y, 1.0)
        ap_grid = R .* a_grid .+ y .- sol[:c]
    else
        error("EGM solution does not contain a_next or c")
    end
    return (a_grid = a_grid, ap_grid = ap_grid)
end

const _EGM_CACHE = Ref{Any}(nothing)

function ensure_egm_loaded(cfg)
    if _EGM_CACHE[] === nothing
        _EGM_CACHE[] = _build_egm_policy(cfg)
    end
    return _EGM_CACHE[]
end

function aegm(a::AbstractVector, y::AbstractVector)
    @assert length(a) == length(y)
    cfg = Main.load_config(Main.ARGS[findfirst(x -> x == "--config", Main.ARGS)+1])
    cfg = ThesisProject.Config.to_symbol_dict(cfg)
    data = ensure_egm_loaded(cfg)
    ag = data.a_grid
    apg = data.ap_grid
    # Linear interp on a only (y deterministic here)
    out = similar(a, eltype(apg))
    for (i, ai) in pairs(a)
        if ai <= ag[1]
            out[i] = apg[1]
        elseif ai >= ag[end]
            out[i] = apg[end]
        else
            j = searchsortedfirst(ag, ai)
            j = max(2, min(length(ag), j))
            a0 = ag[j-1]
            a1 = ag[j]
            t = (ai - a0) / (a1 - a0)
            out[i] = (1 - t) * apg[j-1] + t * apg[j]
        end
    end
    return out
end

# Export into Main for scripts/run_pretrain.jl
Main.aegm = aegm
