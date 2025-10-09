module MethodUtils

using Base: @views
using ..CommonValidators: is_nondec, is_positive, respects_amin

export build_consumption_initializer, validate_policy!, DEFAULT_VALIDATION_CHECKS

const BASIC_WARM_STARTS = (:default, :half_resources, :none)
const DEFAULT_VALIDATION_CHECKS =
    (:c_positive, :a_above_min, :c_monotone_nondec, :a_monotone_nondec)

@inline _normalize_warm_start(warm_start) = Symbol(lowercase(string(warm_start)))

"""
    build_consumption_initializer(p, g; shocks=nothing, warm_start=:default, custom_c=nothing)

Return an initial consumption policy for warm starts shared across solver adapters.
Defaults to `nothing` so solver kernels pick their internal initialisations.
"""
function build_consumption_initializer(
    p,
    g;
    shocks = nothing,
    warm_start::Symbol = :default,
    custom_c = nothing,
)
    warm = _normalize_warm_start(warm_start)
    if shocks === nothing
        return _build_c_init_det(p, g, warm, custom_c)
    else
        return _build_c_init_stoch(p, g, shocks, warm, custom_c)
    end
end

function _build_c_init_det(p, g, warm::Symbol, custom_c)
    a_grid = g[:a].grid
    a_min = g[:a].min
    R = 1 + p.r

    if warm == :steady_state
        c = similar(a_grid, Float64)
        @inbounds for (i, a) in enumerate(a_grid)
            cval = p.y + R * a - a
            cmax = p.y + R * a - a_min
            c[i] = clamp(cval, 1e-12, cmax)
        end
        return c
    elseif warm in BASIC_WARM_STARTS
        return nothing
    else
        if custom_c === nothing
            return nothing
        elseif custom_c isa AbstractVector
            return copy(custom_c)
        else
            error("custom deterministic warm-start must be a vector")
        end
    end
end

function _build_c_init_stoch(p, g, shocks, warm::Symbol, custom_c)
    a_grid = g[:a].grid
    a_min = g[:a].min
    R = 1 + p.r

    if warm == :steady_state
        z_grid = shocks.zgrid
        Na = length(a_grid)
        Nz = length(z_grid)
        c = Array{Float64}(undef, Na, Nz)
        tmp = similar(a_grid, Float64)
        @inbounds for (j, z) in enumerate(z_grid)
            y = exp(z)
            @inbounds for (i, a) in enumerate(a_grid)
                cval = y + R * a - a
                cmax = y + R * a - a_min
                tmp[i] = clamp(cval, 1e-12, cmax)
            end
            @views c[:, j] .= tmp
        end
        return c
    elseif warm in BASIC_WARM_STARTS
        return nothing
    else
        if custom_c === nothing
            return nothing
        elseif custom_c isa AbstractMatrix
            return copy(custom_c)
        else
            error("custom stochastic warm-start must be a matrix")
        end
    end
end

"""
    validate_policy!(metadata, policy, amin; method_name, verbose=false, checks=DEFAULT_VALIDATION_CHECKS)

Run common monotonicity/positivity checks on the solution policy. Results are
stored in-place in `metadata` under `:valid` and `:validation` (when invalid).
Returns the boolean validity flag.
"""
function validate_policy!(
    metadata::Dict{Symbol,Any},
    policy::Dict{Symbol,Any},
    amin::Real;
    method_name::AbstractString = "Solver",
    verbose::Bool = false,
    checks = DEFAULT_VALIDATION_CHECKS,
)
    c_val = policy[:c].value
    a_val = policy[:a].value

    violations = Dict{Symbol,Any}()
    valid = true

    for check in checks
        result =
            check === :c_positive ? is_positive(c_val) :
            check === :a_above_min ? respects_amin(a_val, amin) :
            check === :c_monotone_nondec ? is_nondec(c_val) :
            check === :a_monotone_nondec ? is_nondec(a_val) :
            error("Unknown validation check: $(check)")
        violations[check] = result
        valid &= result
    end

    metadata[:valid] = valid
    if !valid
        metadata[:validation] = violations
        if verbose
            @warn "$(method_name) solution failed monotonicity/positivity checks; marking as invalid." violations
        else
            @info "$(method_name) solution failed validation; set solver.verbose=true for details."
        end
    end

    return valid
end

end # module
