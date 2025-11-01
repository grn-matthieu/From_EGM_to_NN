# NN AiO Sweep Report

Best configuration: `aggressive_schedule` with mean final RMSE = 0.046062 and average runtime 260.33 s.

## Summary Table

|variant | runs | runtime_mean | runtime_std | runtime_function | rmse_mean | rmse_std | rmse_median | final_rmse_function | mean_ee_mean | mean_ee_std | converged_function|
|--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
|aggressive_schedule | 1 | 260.3261 | 0.0 | 260.326117023 => :runtime_p90 | 0.0461 | 0.0 | 0.0461 | 0.04606197401881218 => :rmse_p90 | 0.0121 | 0.0 | 0.0 => :convergence_rate|
|wide_network | 1 | 684.2095 | 0.0 | 684.209543165 => :runtime_p90 | 0.0463 | 0.0 | 0.0463 | 0.0463266596198082 => :rmse_p90 | 0.012 | 0.0 | 0.0 => :convergence_rate|
|baseline_control | 1 | 338.3522 | 0.0 | 338.352244563 => :runtime_p90 | 0.0466 | 0.0 | 0.0466 | 0.04655999317765236 => :rmse_p90 | 0.0121 | 0.0 | 0.0 => :convergence_rate|
|curriculum_resampling | 1 | 440.0903 | 0.0 | 440.090317695 => :runtime_p90 | 0.0466 | 0.0 | 0.0466 | 0.04660167172551155 => :rmse_p90 | 0.0108 | 0.0 | 0.0 => :convergence_rate|
|precision_highmc | 1 | 858.9619 | 0.0 | 858.961867178 => :runtime_p90 | 0.0467 | 0.0 | 0.0467 | 0.04672935605049133 => :rmse_p90 | 0.0116 | 0.0 | 0.0 => :convergence_rate|



## Plots
- rmse_history: ![](rmse_history.png)
- runtime_vs_rmse: ![](runtime_vs_rmse.png)
- rmse_bar: ![](rmse_bar.png)
