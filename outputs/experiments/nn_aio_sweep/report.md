# NN AiO Sweep Report

Best configuration: `stability_baseline` with mean final RMSE = 0.39723 and average runtime 344.52 s.

## Summary Table

4×7 DataFrame
 Row │ variant             runtime_mean  runtime_std  rmse_mean  rmse_std  mean_ee_mean  converged_function
     │ String              Float64       Float64      Float64    Float64   Float64       Pair…
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ stability_baseline       344.523          NaN   0.39723        NaN      0.348194  0.0=>:convergence_rate
   2 │ moderate_lr              213.498          NaN   0.422377       NaN      0.368619  0.0=>:convergence_rate
   3 │ low_lr                   397.55           NaN   0.428585       NaN      0.373818  0.0=>:convergence_rate
   4 │ large_samples            248.527          NaN   0.461254       NaN      0.411608  0.0=>:convergence_rate

## Plots
- rmse_history: ![](rmse_history.png)
- runtime_vs_rmse: ![](runtime_vs_rmse.png)
- rmse_bar: ![](rmse_bar.png)

### Observations

- Even after aggressive smoothing (σ_shocks → 0, large batches), AiO stabilises around RMSE ≈ 0.40 for this more patient (`β = 0.95`) model; extending the cosine schedule lowers gradients but does not breach the 10⁻¹ threshold.
- Raising the sample cloud (`large_samples`) reduces variance early on, yet the added stochastic jitter pushes the terminal residual higher, suggesting the approximator struggles near the cash-on-hand boundaries.
- The `moderate_lr` variant delivers the best speed/accuracy compromise (≈210 s, RMSE ≈0.42), whereas the ultra-cautious `low_lr` gains little in accuracy despite doubling runtime.
- Future improvements likely require architectural changes (wider nets or residual connections) or alternative objectives; pure AiO tuning on this grid appears bounded by ≈0.39 RMSE.
