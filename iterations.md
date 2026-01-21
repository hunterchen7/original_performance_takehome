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

## Iteration 2

Summary:
- Added VLIW bundle emission to pack ALU/VALU/LOAD/STORE/FLOW slots per cycle.
- Double-buffered vector registers and pipelined next-block prefetch (address calc, vload, gather) during current block hashing.
- Simplified vector idx update using bitwise step (`1 + (val & 1)`) and overlapped val store with update.
- Added optional vector debug via `vcompare` gated by `KernelBuilder.enable_vdebug`.

Files changed:
- perf_takehome.py

Test results:
- `python3 perf_takehome.py Tests.test_kernel_cycles` -> CYCLES: 9918
