# Iteration Log

## Iteration 1

Summary:
- Vectorized the kernel across the batch dimension using `vload`/`vstore`/`valu`.
- Added vector constants (zero/one/two/n_nodes/forest base) and per-stage hash constants.
- Implemented vector gather for `forest_values_p + idx` via `load_offset`.
- Kept a scalar tail path for batch sizes not divisible by `VLEN`.

Files changed:
- perf_takehome.py

Test results:
- `python3 perf_takehome.py Tests.test_kernel_cycles` -> CYCLES: 21041
