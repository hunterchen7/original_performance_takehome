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

## Iteration 3

Summary:
- Replaced the hand-written pipeline with a dynamic VLIW scheduler that keeps multiple vector blocks in flight and packs ALU/VALU/LOAD/STORE/FLOW per cycle.
- Kept `idx`/`val` in scratch across all rounds (single initial vload, final stores only) to cut per-round memory traffic.
- Simplified idx update to `idx = idx*2 + 1 + (val & 1)` using `multiply_add` to reduce update ops.
- Fused hash stages where `val = (val + c1) + (val << k)` into a single `multiply_add` using precomputed multipliers.
- Tuned pipeline depth (`pipe_buffers=20`) to balance throughput and scratch usage.

Files changed:
- perf_takehome.py

Test results:
- `python3 perf_takehome.py Tests.test_kernel_cycles` -> CYCLES: 2292
- `python3 tests/submission_tests.py` -> FAIL (speed thresholds), CYCLES: 2292

## Iteration 4

Summary:
- Special-cased round 0 to skip gather by XORing with a broadcasted `forest_values[0]`.
- Prefetched round 1 node values using `val & 1` in round 0 update5 to skip round 1 gather/addr.
- Added node0/node1/node2 scalar loads and vector broadcasts for early-round fast paths.
- Tuned pipeline depth to `pipe_buffers=19` for the new schedule.

Files changed:
- perf_takehome.py

Test results:
- `python3 perf_takehome.py Tests.test_kernel_cycles` -> CYCLES: 2119
- `python3 tests/submission_tests.py` -> FAIL (stricter thresholds), CYCLES: 2119
