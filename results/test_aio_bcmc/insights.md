# AiO vs bc-MC Auto-N comparison

- Config tag: `test_aio_bcmc` | Seeds: 11, 19 | Notes: (none)

- Runtime: bc-MC auto-N averages 2.702s vs AiO 21.021s (87.1% change).
- Grid Euler residuals (p95 abs): bc-MC 1.136e+00 vs AiO 1.136e+00 (0.0% shift).
- Monte-Carlo residuals (p95 abs): bc-MC 3.419e-01 vs AiO 3.419e-01 (0.0%).

## Aggregates

| objective | runs | runtime (mean±sd) | p95 abs resid | MC p95 | binding share | converged |
|-----------|------|-------------------|---------------|--------|----------------|-----------|
| euler_fb_aio | 2 | 21.021±28.760 | 1.14e+00 | 3.42e-01 | 0.0% | 0.0% |
| euler_fb_bcmc | 2 | 2.702±2.732 | 1.14e+00 | 3.42e-01 | 0.0% | 0.0% |
