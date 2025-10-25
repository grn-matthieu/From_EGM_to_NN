# Scheduler vs Optimiser study

- Tag: `sched_opt_demo` | Seeds: 11, 19 | Notes: (none)

- Fastest combo: exponential + rmsprop (runtime 1.289s, mean abs resid 4.28e-01).
- Residual stability: best p95=1.13e+00 (combo exponential/rmsprop).

| scheduler | optimizer | runs | runtime (mean±sd) | p95 abs resid | MC p95 |
|-----------|-----------|------|-------------------|---------------|-------|
| cosine | adam | 2 | 21.758±28.930 | 1.13e+00 | 3.41e-01 |
| cosine | rmsprop | 2 | 1.516±0.254 | 1.13e+00 | 3.41e-01 |
| exponential | adam | 2 | 1.310±0.009 | 1.13e+00 | 3.41e-01 |
| exponential | rmsprop | 2 | 1.289±0.007 | 1.13e+00 | 3.41e-01 |
