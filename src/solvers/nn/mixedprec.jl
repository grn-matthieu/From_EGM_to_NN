"""Deprecated shim module for mixed-precision helpers.

This module used to contain a full mixed-precision implementation. The
functionality has been migrated to `..NNUtils`. The shim below re-exports
`to_fp32` from `NNUtils` to maintain backward compatibility for any late
references. Prefer `NNUtils.to_fp32` directly.
"""
module NNMixedPrecision

using ..NNUtils: to_fp32 as _to_fp32
export to_fp32

function to_fp32(x)
    @warn "NNMixedPrecision is deprecated; use NNUtils.to_fp32 instead"
    return _to_fp32(x)
end

end # module
