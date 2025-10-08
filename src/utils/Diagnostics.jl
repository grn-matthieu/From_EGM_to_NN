module UtilsDiagnostics

"""
Utility helpers to post-process solver diagnostics.

Currently exposes `mean_abs_error` which computes the mean absolute value of a
collection while discarding `missing` entries and `NaN`s. This is useful for
Euler equation residuals that may intentionally mark binding-constraint rows as
`NaN` so that downstream aggregations ignore them.
"""

export mean_abs_error

"""
    mean_abs_error(x)

Compute the mean absolute value of `x`, skipping `missing` entries and `NaN`s.
Returns `NaN` if no finite observations are present. Scalars are promoted to
`Float64`.
"""
function mean_abs_error(x)
    if x isa Number
        val = Float64(x)
        return isnan(val) ? NaN : abs(val)
    elseif x isa Missing
        return NaN
    elseif x isa AbstractArray
        total = 0.0
        count = 0
        for v in x
            if v === missing || v isa Missing
                continue
            end
            val = Float64(v)
            if isnan(val)
                continue
            end
            total += abs(val)
            count += 1
        end
        return count == 0 ? NaN : total / count
    else
        return mean_abs_error(float(x))
    end
end

end # module
