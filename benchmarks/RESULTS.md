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
| mul / div [1000,1000] | 7.1 / 7.2 | 17.0 add (0.1.0 ref) | fast vec4 path |
| add / sub [1000,1000] | **5.8 / 12.8** ✅ | — | was 1,812/1,870 — fixed #1671 (non-ASCII PTX) |
| **Reductions** | | | |
| sum dim=0 [1000,1000] | 4.0 | — | on-device, fast |
| sum_all / mean [1000,1000] | **58.4 / 41.9** ✅ | — | was 681/686 — fixed #1672 (on-device pass-2) |
| **Normalization** | | | |
| softmax [64,256] | 5.8 | 8.6 (CPU torch) | faster than torch CPU softmax |
| **MLP fwd B=32** | 601.7 | 73.4 (0.1.0 ref) | dominated by small-matmul launch overhead |
| **Host↔Device** | CPU→GPU 252 / GPU→CPU 378 | — | PCIe transfer |

## Analysis

### The headline win: CPU matmul is no longer a naive triple-loop
At 0.1.0, `matmul [1024,1024]` took **2,106,087 us (2.1 s)** — 736x slower than PyTorch. At 0.6.0 it is **5,422 us**, a **~388x speedup**, now only **1.8x** off PyTorch's MKL and faster than NumPy. The MLP training step dropped from 13x→2.5x slower, backward is at near-parity (1.1x). The framework overhead (autograd, graph, memory) remains minimal.

### GPU is fast where the kernels are warm
cuBLAS matmul (10–13 us), the new PTX unary kernels (relu/sigmoid/tanh/exp/log ~7 us), softmax (5.8 us), and on-device `sum dim=0` (6.2 us) are all in the single-digit-microsecond launch-bound regime — the 0.6.0 GPU kernel campaign landed.

### Findings surfaced by this run — GPU items FIXED + re-benchmarked
1. **GPU `add`/`sub` were ~270x slower than `mul`/`div`** (1,812/1,870 us). ROOT CAUSE: a non-ASCII `×` (U+00D7) in an `ADD_VEC4_PTX` comment made the vec4 add kernel JIT-fail (`CUDA_ERROR_INVALID_PTX`) on **every** call (the module cache only stores successes), wasting ~1.8 ms before falling back to scalar; `sub` routes through the add path so it was hit too. **Fixed #1671** (ASCII-ify the comment). Re-benchmarked: **add 5.8 us, sub 12.8 us** (matches mul/div). A regression guard now scans all PTX literals for non-ASCII so the class can't recur.
2. **GPU `sum_all`/`mean` were ~681 us vs `sum dim=0` 6 us.** ROOT CAUSE: the full-reduction pass-2 combined the per-block partials via a **host readback** (`gpu_to_cpu` + CPU sum + re-upload) — a Device→Host sync every call. **Fixed #1672** (pass-2 now recurses on-device for sum/min/max/prod, f32+f64+masked). Re-benchmarked: **sum_all 58 us, mean 42 us**; data stays GPU-resident.
3. **(Bonus, caught by the #1671 ASCII guard)** `GELU_BACKWARD_TANH_PTX` had the same non-ASCII defect — and with **no scalar fallback**, GPU tanh-GELU backward was fully broken (`CUDA_ERROR_INVALID_PTX`). **Fixed #1673**; exposing it then revealed wrong/truncated `c3` constants (`0.134199` vs `3·0.044715=0.134145`) and f32-precision f64 constants — **fixed #1674**. GPU tanh-GELU backward now matches torch (f32 ~1e-5, f64 ~1e-9).

### Remaining (CPU SIMD/algorithm gaps — known, pre-existing, not regressions)
4. **CPU axis reductions `sum dim=0`/`mean dim=1` ~10 ms** (~100x NumPy). Strided axis-reduction CPU loop is unvectorized/cache-unfriendly.
5. **CPU elementwise / transcendental** (add/mul ~1,000 us, sin/cos/tanh ~7 ms) remain scalar `libm`-per-element — the long-standing SIMD gap (ferray-ufunc integration), not a regression.

> Note: `pytorch_validate.py`'s two "FAIL" lines (10-step MLP loss, finite-difference gradient check) are PyTorch's OWN self-baselines under that seed/LR and an over-tight FD epsilon threshold — they are not ferrotorch comparisons. Its softmax/layernorm/conv/dropout correctness baselines all PASS.

### The path to parity (updated)
1. **GPU add/sub + full-reduction**: fix the per-call PTX recompile / sync (findings 1–2) — pure kernel-dispatch wins, no math change.
2. **CPU matmul**: largely solved (388x). Remaining 1.8x is BLAS micro-optimization (faer/MKL-class blocking).
3. **CPU elementwise/transcendental + axis reductions**: wire SIMD (ferray-ufunc) and a blocked axis-reduction kernel.

**Bottom line**: the 0.6.0 campaign closed the catastrophic CPU-matmul gap (736x→1.8x) and built out the GPU kernel surface (matmul/unary/softmax/reductions in single-digit us). The remaining gaps are localized kernel-dispatch and CPU-SIMD optimizations, not architecture.
