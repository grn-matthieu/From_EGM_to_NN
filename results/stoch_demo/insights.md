# Full stochastic benchmark

- Tag: `stoch_demo` | Seeds: 11 | Notes: (none)

- EE target: 1.0e-05
- Best accuracy: TimeIteration with mean EE 2.28e-03
- Fastest: Perturbation (runtime 0.00s)

| method | runs | runtime (mean±sd) | mean EE | p95 EE | target hit |
|--------|------|-------------------|---------|--------|------------|
| EGM | 1 | 11.59±0.00 | 9.71e-03 | 2.21e-04 | 0% |
| NN_AiO | 1 | 42.47±0.00 | 3.98e-01 | 9.47e-01 | 0% |
| NN_BCMC | 1 | 29.67±0.00 | 3.98e-01 | 9.45e-01 | 0% |
| Perturbation | 1 | 0.00±0.00 | 5.83e-02 | 6.28e-02 | 0% |
| Projection | 1 | 7.10±0.00 | 1.10e-02 | 3.37e-02 | 0% |
| TimeIteration | 1 | 0.53±0.00 | 2.28e-03 | 8.63e-07 | 0% |
