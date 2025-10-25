# AiO vs bc-MC Auto-N comparison

- Config tag: `test_autoN` | Seeds: 11 | Notes: (none)

- Runtime: bc-MC auto-N averages 17.061s vs AiO 41.370s (58.8% change).
- Grid Euler residuals (p95 abs): bc-MC 1.008e+00 vs AiO 1.007e+00 (-0.1% shift).
- Monte-Carlo residuals (p95 abs): bc-MC 3.532e-01 vs AiO 3.544e-01 (0.3%).

## Aggregates

| objective | runs | runtime (mean±sd) | p95 abs resid | MC p95 | binding share | converged |
|-----------|------|-------------------|---------------|--------|----------------|-----------|
| euler_fb_aio | 1 | 41.370±0.000 | 1.01e+00 | 3.54e-01 | 0.0% | 0.0% |
| euler_fb_bcmc | 1 | 17.061±0.000 | 1.01e+00 | 3.53e-01 | 0.0% | 0.0% |
