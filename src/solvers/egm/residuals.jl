module EGMResiduals

export euler_resid_det

# --- Functions ---

"""
    euler_residuals_simple(params, a_grid, c, c_next)
        params: Model parameters
        a_grid: Asset grid
        c: Current consumption
        c_next: Next period consumption
    Compute the Euler residuals for a simple consumption model.
    Output : Vector of Euler residuals
"""
function euler_resid_det(model_params, c::Vector{Float64}, c_next::Vector{Float64})
    resid = similar(c)

    # Clamping to avoid division by zero
    c_clamped      = clamp.(c, 1e-12, Inf)
    c_next_clamped = clamp.(c_next, 1e-12, Inf)

    @. resid = abs(1 - model_params.γ * (c_clamped ./ c_next_clamped).^model_params.σ)

    return resid
end


end #module