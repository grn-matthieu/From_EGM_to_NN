function is_nondec(x::AbstractVector; tol = 1e-8)
    n = length(x)
    @inbounds for i = 1:(n-1)
        if x[i+1] < x[i] - tol
            return false
        end
    end
    return true
end

function is_nondec(x::AbstractMatrix; tol = 1e-8)
    nrow, ncol = size(x)
    @inbounds for j = 1:ncol
        for i = 1:(nrow-1)
            if x[i+1, j] < x[i, j] - tol
                return false
            end
        end
    end
    return true
end
