# Benchmark Comparison: ferrotorch vs PyTorch vs NumPy

**Hardware**: RTX 3090 24GB · AMD CPU (WSL2)
**Versions**: ferrotorch 0.6.0 (release, `--features gpu`) · PyTorch 2.11.0+cu130 · NumPy 2.4.5
**Date**: 2026-05-29
**Method**: per-op average over 20–100 iterations after warmup; GPU timings synchronized on both sides (`GpuBackend::synchronize` / `torch.cuda.synchronize`). All times in **microseconds (us)**; lower is better.

Reproduce: `cargo run --release --features gpu --example ferrotorch_bench` · `python benchmarks/pytorch_bench.py` · `python benchmarks/numpy_bench.py`

## CPU

| Operation | ferrotorch | PyTorch | NumPy |
|---|--:|--:|--:|
| **Creation** | | | |
| zeros [1000,1000] | 83 | 35 | 77 |
| rand [1000,1000] | 2,689 | 1,865 | 3,198 |
| randn_like [1000,1000] | 5,850 | 2,038 | — |
| **Elementwise** | | | |
| add [1000,1000] | 907 | 34 | 235 |
| mul [1000,1000] | 929 | 40 | 226 |
| sub [1000,1000] | — | — | 237 |
| div [1000,1000] | — | — | 214 |
| relu [1000,1000] | 125 | 42 | 164 |
| sigmoid [1000,1000] | 1,037 | 132 | 1,069 |
| **Transcendental** | | | |
| exp [1000,1000] | 997 | 50 | 702 |
| log [1000,1000] | 948 | 1,120 | 1,040 |
| sin [1000,1000] | 968 | 97 | 799 |
| cos [1000,1000] | 984 | 104 | 789 |
| tanh [1000,1000] | 989 | 200 | 1,322 |
| **Matrix multiply** | | | |
| matmul [64,64] | 16 | 7 | 6 |
| matmul [256,256] | 220 | 68 | 1,270 |
| matmul [1024,1024] | 4,979 | 3,024 | 3,867 |
| **Reductions** | | | |
| sum_all [1000,1000] | 379 | 21 | 133 |
| sum dim=0 [1000,1000] | 10,118 | 14 | 80 |
| mean dim=1 [1000,1000] | 10,480 | 11 | 154 |
| **Tensor manipulation** | | | |
| permute [1000,1000] | 1,747 | 467 | 452 |
| chunk [1000,1000] /4 | 341 | 2 | — |
| cat [4×250,1000] | 236 | 70 | — |
| broadcast add [1000,1]+[1,1000] | 976 | 12 | — |
| broadcast mul [64,1,256]×[1,128,1] | 2,112 | 49 | — |
| **Networks** | | | |
| MLP fwd B=32 (784→256→10) | 105 | 53 | — |
| MLP bwd B=32 | 468 | 355 | — |
| training step B=32 (+Adam) | 1,525 | 692 | — |
| MLP fwd B=128 (784→512→256→10) | 957 | 318 | — |
| MLP bwd B=128 | 3,670 | 1,555 | — |
| training step B=128 | 6,418 | 1,972 | — |
| Conv2d fwd [32,3,32,32]→[32,16,30,30] | 1,244 | 102 | — |
| GRU/LSTM fwd (128→256, seq=32, B=16) | 9,212 | 2,381 | — |

## GPU (synchronized)

| Operation | ferrotorch | PyTorch |
|---|--:|--:|
| **Creation** | | |
| zeros [1000,1000] | 334 | 55 |
| rand [1000,1000] | 15 | 60 |
| **Elementwise** | | |
| add [1000,1000] | 53 | 42 |
| mul [1000,1000] | 53 | 42 |
| sub [1000,1000] | 51 | — |
| div [1000,1000] | 54 | — |
| relu [1000,1000] | 54 | 42 |
| sigmoid [1000,1000] | 54 | — |
| tanh [1000,1000] | 60 | — |
| exp [1000,1000] | 47 | — |
| log [1000,1000] | 47 | — |
| **Matrix multiply (cuBLAS)** | | |
| matmul [64,64] | 99 | 65 |
| matmul [256,256] | 56 | 33 |
| matmul [1024,1024] | 198 | 319 |
| matmul [4096,4096] | 15,225 | 14,880 |
| **Reductions** | | |
| sum_all [1000,1000] | 342 | — |
| sum dim=0 [1000,1000] | 143 | — |
| mean [1000,1000] | 462 | — |
| **Normalization** | | |
| softmax [64,256] | 117 | — |
| **Networks** | | |
| MLP fwd B=32 (784→256→10) | 165 | 201 |
| MLP bwd B=32 | — | 1,219 |
| **Transfer** | | |
| CPU→GPU [1000,1000] | 285 | 358 |
| GPU→CPU [1000,1000] | 376 | 413 |

(— = operation not measured by that framework's benchmark script.)
