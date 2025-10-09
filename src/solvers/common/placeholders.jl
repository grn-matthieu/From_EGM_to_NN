module SolverPlaceholders

using Statistics: mean
using Base: @views

export build_placeholder_solution

@inline function _maybe_mean_income(params)
    if hasproperty(params, :y_dim) && hasproperty(params, :y)
        vals = Float64.(collect(params.y))
        return max(mean(vals), 1e-3)
    elseif hasproperty(params, :y)
        return max(Float64(params.y), 1e-3)
    else
        return 1.0
    end
end

@inline _maybe_get(nt::NamedTuple, key::Symbol, default) =
    hasproperty(nt, key) ? getproperty(nt, key) : default

"""
    build_placeholder_solution(solver, model_params, model_grids, model_shocks;
        opts = NamedTuple(), note = "placeholder solution")

Construct a dummy solver result compatible with solver adapters. The shape of
consumption and assets mirrors the deterministic/stochastic grids and all
diagnostics are set to benign defaults.
"""
function build_placeholder_solution(
    solver::Symbol,
    model_params,
    model_grids,
    model_shocks;
    opts::NamedTuple = NamedTuple(),
    note::AbstractString = "placeholder solution",
)
    a_info = model_grids[:a]
    a_grid = Float64.(a_info.grid)
    Na = length(a_grid)
    c_level = _maybe_mean_income(model_params)

    if model_shocks === nothing || !hasproperty(model_shocks, :zgrid)
        c = fill(c_level, Na)
        a_next = copy(a_grid)
        resid = zeros(Float64, Na)
    else
        z_grid = model_shocks.zgrid
        Nz = length(z_grid)
        base_c = fill(c_level, Na)
        c = repeat(reshape(base_c, Na, 1), 1, Nz)
        base_a = copy(a_grid)
        a_next = repeat(reshape(base_a, Na, 1), 1, Nz)
        resid = zeros(Float64, Na, Nz)
    end

    opts_out = merge(
        (
            tol = _maybe_get(opts, :tol, NaN),
            tol_pol = _maybe_get(opts, :tol_pol, NaN),
            maxit = _maybe_get(opts, :maxit, 0),
            interp_kind = _maybe_get(opts, :interp_kind, :placeholder),
            relax = _maybe_get(opts, :relax, 0.0),
            verbose = _maybe_get(opts, :verbose, false),
            resid_metric = _maybe_get(opts, :resid_metric, :placeholder),
            runtime = _maybe_get(opts, :runtime, 0.0),
            seed = _maybe_get(opts, :seed, nothing),
            note = note,
        ),
        opts,
    )

    return (;
        a_grid = a_grid,
        c = c,
        a_next = a_next,
        resid = resid,
        iters = 0,
        converged = false,
        max_resid = 0.0,
        rmse = 0.0,
        model_params = model_params,
        opts = opts_out,
        delta_pol = 0.0,
        placeholder = true,
        solver = solver,
    )
end

end # module
