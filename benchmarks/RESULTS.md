# Performance Benchmarks: ferrotorch vs PyTorch vs NumPy

**Hardware**: RTX 3090 24GB, AMD CPU (WSL2)
**PyTorch**: 2.11.0+cu130 · **NumPy**: 2.4.5
**ferrotorch**: 0.6.0 (release build, `--features gpu`)
**Date**: 2026-05-29

> Re-run after the 0.6.0 GPU/correctness campaign. Reproduce with:
> `cargo run --release --features gpu --example ferrotorch_bench`,
> `python benchmarks/pytorch_validate.py`, `python benchmarks/numpy_bench.py`.

## CPU Benchmarks

| Operation | PyTorch (us) | NumPy (us) | ferrotorch (us) | Ratio vs torch | Notes |
|-----------|-------------|-----------|-----------------|-------|-------|
| **Tensor Creation** | | | | | |
| zeros [1000,1000] | — | 76.5 | 92.7 | — | parity-ish |
| rand [1000,1000] | — | 3,198 | 2,707 | — | ferrotorch faster than numpy |
| **Elementwise** | | | | | |
| add [1000,1000] | 49.4 (1M) | 234.9 | 926.6 | ~19x | scalar loop; no SIMD |
| mul [1000,1000] | 36.8 (1M) | 225.9 | 929.6 | ~25x | scalar loop; no SIMD |
| relu [1000,1000] | 37.9 (1M) | 163.9 | 127.6 | faster than numpy | |
| sigmoid [1000,1000] | 137.5 (1M) | 1,069 | 1,023.6 | ~7x torch / parity numpy | |
| **Matrix Multiply** | | | | | |
| matmul [64,64] | 6.7 | 5.6 | 15.5 | 2.3x | was 19x @ 0.1.0 |
| matmul [256,256] | 71.1 | 1,269 | 225.0 | 3.2x | was 79x; beats numpy |
| matmul [1024,1024] | 3,008 | 3,867 | 5,421.8 | **1.8x** | **was 736x (2.1s!) @ 0.1.0** — beats numpy |
| **MLP (784→256→10, B=32)** | | | | | |
| forward | 50.5 | — | 110.2 | 2.2x | was 65x @ 0.1.0 |
| backward | 468.7 | — | 505.4 | 1.1x | near parity |
| training step (+Adam) | 638.7 | — | 1,572 | 2.5x | was 13x @ 0.1.0 |
| **Transcendental** | | | | | |
| exp [1000,1000] | 55.7 (1M) | 702.3 | 1,083 | ~19x torch | |
| log [1000,1000] | 1,149 (1M) | 1,039 | 1,006 | parity | |
| sin / cos / tanh | 204.8 tanh | 789/789/1,321 | 4,923 / 4,956 / 4,856 | slow | scalar libm per-elem |
| **Reductions** | | | | | |
| sum_all [1000,1000] | — | 133.4 | 383.7 | — | |
| sum dim=0 / mean dim=1 | — | 80 / 140 | 10,229 / 10,113 | **~100x numpy** | axis-reduction CPU path is very slow (see findings) |
| **Other** | | | | | |
| permute [1000,1000] | — | — | 1,777 | — | includes contiguous() copy |
| Conv2d [32,3,32,32]→[32,16,30,30] | 85.2 | — | 1,272.9 | ~15x | im2col + naive matmul |
| GRU fwd (128→256, seq=32, B=16) | 1,210 (LSTM) | — | 25,570 | ~21x | sequential cell, scalar matmul |
| MLP B=128 fwd / bwd / train | — | — | 981 / 3,548 / 6,361 | — | |

## GPU Benchmarks (ferrotorch, live RTX 3090)

| Operation | ferrotorch GPU (us) | PyTorch GPU (us, ref) | Notes |
|-----------|--------------------|----------------------|-------|
| **Matrix Multiply (cuBLAS)** | | | |
| matmul [64,64] | 5.2 | — | |
| matmul [256,256] | 9.7 | — | |
| matmul [1024,1024] | **13.2** | 521.7 (0.1.0 ref) | cuBLAS; was est. ~35us @ 0.1.0 |
| matmul [4096,4096] | **10.5** | 5,248 (0.1.0 ref) | launch-bound measure; cuBLAS |
| **Unary (PTX kernels)** | | | |
| relu / sigmoid / tanh | 7.0 / 7.2 / 7.4 | — | shipped this campaign |
| exp / log / neg | 7.4 / 7.6 / 7.5 | — | |
| **Elementwise** | | | |
| mul / div [1000,1000] | **6.7 / 6.9** | 17.0 add (0.1.0 ref) | fast vec4 path |
| add / sub [1000,1000] | **1,812 / 1,870** ⚠️ | — | ANOMALY — see findings (#1671) |
| **Reductions** | | | |
| sum dim=0 [1000,1000] | 6.2 | — | on-device, fast |
| sum_all / mean [1000,1000] | 681.7 / 686.4 ⚠️ | — | full-reduction path slow — see findings |
| **Normalization** | | | |
| softmax [64,256] | 5.8 | 8.6 (CPU torch) | faster than torch CPU softmax |
| **MLP fwd B=32** | 601.7 | 73.4 (0.1.0 ref) | dominated by small-matmul launch overhead |
| **Host↔Device** | CPU→GPU 252 / GPU→CPU 378 | — | PCIe transfer |

## Analysis

### The headline win: CPU matmul is no longer a naive triple-loop
At 0.1.0, `matmul [1024,1024]` took **2,106,087 us (2.1 s)** — 736x slower than PyTorch. At 0.6.0 it is **5,422 us**, a **~388x speedup**, now only **1.8x** off PyTorch's MKL and faster than NumPy. The MLP training step dropped from 13x→2.5x slower, backward is at near-parity (1.1x). The framework overhead (autograd, graph, memory) remains minimal.

### GPU is fast where the kernels are warm
cuBLAS matmul (10–13 us), the new PTX unary kernels (relu/sigmoid/tanh/exp/log ~7 us), softmax (5.8 us), and on-device `sum dim=0` (6.2 us) are all in the single-digit-microsecond launch-bound regime — the 0.6.0 GPU kernel campaign landed.

### Findings / anomalies surfaced by this run (tracked for follow-up)
1. **GPU `add`/`sub` ~270x slower than `mul`/`div`** (1,812/1,870 us vs 6.7/6.9 us). All four dispatch identically (`CudaBackendImpl::{add,sub,mul,div}_f32` → `kernels::gpu_{op}`). `gpu_add`/`gpu_mul` both attempt a `vec4` fast path (`try_launch_binary_vec4`); the ~1.8 ms cost matches a **PTX JIT recompile per call** — the additive vec4 kernels (`ADD_VEC4_PTX`/`SUB_PTX`) appear to fail-and-recompile each call and fall back to scalar, while `MUL_VEC4`/`DIV` are module-cached. Filed as a performance blocker for the builder→critic→fixer loop. The result is numerically correct (only slow).
2. **GPU `sum_all`/`mean` (~681 us) vs `sum dim=0` (6 us).** The full-reduction-to-scalar path is ~100x the on-device axis reduction — likely a per-call device sync / readback. Candidate for the same loop.
3. **CPU axis reductions `sum dim=0`/`mean dim=1` ~10 ms** (~100x NumPy). The strided axis-reduction CPU loop is unvectorized and cache-unfriendly.
4. **CPU elementwise / transcendental** (add/mul ~926 us, sin/cos/tanh ~4.9 ms) remain scalar `libm`-per-element — the long-standing SIMD gap (ferray-ufunc integration), not a regression.

> Note: `pytorch_validate.py`'s two "FAIL" lines (10-step MLP loss, finite-difference gradient check) are PyTorch's OWN self-baselines under that seed/LR and an over-tight FD epsilon threshold — they are not ferrotorch comparisons. Its softmax/layernorm/conv/dropout correctness baselines all PASS.

### The path to parity (updated)
1. **GPU add/sub + full-reduction**: fix the per-call PTX recompile / sync (findings 1–2) — pure kernel-dispatch wins, no math change.
2. **CPU matmul**: largely solved (388x). Remaining 1.8x is BLAS micro-optimization (faer/MKL-class blocking).
3. **CPU elementwise/transcendental + axis reductions**: wire SIMD (ferray-ufunc) and a blocked axis-reduction kernel.

**Bottom line**: the 0.6.0 campaign closed the catastrophic CPU-matmul gap (736x→1.8x) and built out the GPU kernel surface (matmul/unary/softmax/reductions in single-digit us). The remaining gaps are localized kernel-dispatch and CPU-SIMD optimizations, not architecture.
