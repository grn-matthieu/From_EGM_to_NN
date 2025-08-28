module SimpleModel
export u, inv_uprime, budget_next

u(c, σ) = (σ == 1.0) ? log(c) : (c^(1-σ) - 1)/(1-σ)
inv_uprime(x, σ) = x^(-1/σ)
budget_next(a, y, r, c) = (1+r)*a + y - c

end