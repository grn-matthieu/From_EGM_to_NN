"""
DataNN

Stub data provider for NN training. Exports a small `generate_dataset` function
that returns input features and targets. For now inputs are (a, z) concatenated
and targets are a simple rule (half resources) so the training loop has data.
"""
module DataNN

export generate_dataset

using Random

function generate_dataset(G, S)
    # Generate sample points on the grid
    a_grid = G[:a].grid

    if isnothing(S)
        # Deterministic case: input is just assets
        X = reshape(a_grid, :, 1)  # Na × 1 matrix
        # Simple target: half of resources (R*a + y)
        R = 1.0 + 0.02  # rough interest rate
        y = 0.5f0 .* (R .* a_grid .+ 1.0f0)  # half resources policy
        return (X, y)
    else
        # Stochastic case: input is (assets, shocks)
        z_grid = S.zgrid
        Na, Nz = length(a_grid), length(z_grid)

        # Create input matrix: each column is [a, z] for one sample
        X = zeros(Float32, Na * Nz, 2)
        y = zeros(Float32, Na * Nz)

        idx = 1
        for j = 1:Nz
            for i = 1:Na
                X[idx, 1] = a_grid[i]
                X[idx, 2] = z_grid[j]
                # Simple target: half of resources with income shock
                R = 1.0 + 0.02
                resources = R * a_grid[i] + exp(z_grid[j])
                y[idx] = 0.5f0 * resources
                idx += 1
            end
        end
        return (X, y)
    end
end

end # module
