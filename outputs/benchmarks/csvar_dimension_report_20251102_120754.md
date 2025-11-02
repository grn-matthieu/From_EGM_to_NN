# CSVAR Benchmark Report: dimension

**Generated:** 2025-11-02 12:07:54

## Configuration

- Scenario: `dimension`
- Method(s): `NN`
- Grid size: Na = 60
- Tolerance: 0.0001
- Repeats per variant: 1

## Summary Statistics

```
1×13 DataFrame
 Row │ variant  method  y_dim  max_eigenvalue  avg_variance  off_diagonal_corr  runtime_mean  runtime_std  rmse_mean  rmse_std  mean_ee_avg  convergence_rate  n_runs
     │ String   String  Int64  Float64         Float64       Int64              Float64       Float64      Float64    Float64   Float64      Float64           Int64
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ diag_3d  NN          3             0.7          0.04                  0        62.854          NaN  0.0866684       NaN    0.0330932               0.0       1
```

## Best Configuration

- **Variant:** diag_3d
- **Method:** NN
- **Dimension:** 3
