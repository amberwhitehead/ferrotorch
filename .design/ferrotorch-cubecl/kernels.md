# CubeCL kernel definitions (elementwise + polynomial + matmul)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/
  - c10/cuda/
-->

## Summary

`ferrotorch-cubecl/src/kernels.rs` is the kernel-language layer. Each
`#[cube(launch_unchecked)] pub fn kernel_*` defines a CubeCL kernel that
compiles to CUDA PTX / AMD HIP / WGPU shaders. Each kernel pairs with a
slice-upload runner (`run_<op>`) and a device-handle-direct runner
(`run_<op>_handle`) that share helpers `run_unary`/`run_binary`/
`run_unary_handle`/`run_binary_handle`. The boundary contract returns
`(cubecl::server::Handle, usize)` — device-resident result + element
count — with NO host readback per ADR #663 item 4.

The kernels mirror upstream PyTorch's `aten/src/ATen/native/cuda/`
elementwise families (per-op `.cu` files: `AbsKernel.cu`,
`UnaryOpsKernel.cu`, `Activation*Kernel.cu`, plus binary ops in
`BinaryOps.cu`) collapsed into one `kernels.rs` because the CubeCL macro
substitutes shader code generation for handwritten CUDA. Polynomial
recurrences (`kernel_chebyshev_t`, etc.) mirror the CPU evaluators in
`ferrotorch-core::special` (`special.cu` upstream).

## Requirements

- REQ-1: Four elementwise binary kernels — `kernel_add`, `kernel_sub`,
  `kernel_mul`, `kernel_div` — each `#[cube(launch_unchecked)]` over
  `&Array<F: Float>` inputs with `ABSOLUTE_POS`-guarded output write.
  Mirrors `aten/src/ATen/native/cuda/BinaryOps.cu` family.

- REQ-2: Ten unary elementwise kernels — `kernel_relu`, `kernel_neg`,
  `kernel_abs`, `kernel_exp`, `kernel_ln`, `kernel_sqrt`, `kernel_sin`,
  `kernel_cos`, `kernel_tanh`, `kernel_sigmoid`. Each is a one-line
  CubeCL math expression over `F: Float`. Mirrors upstream's
  per-op kernels in `aten/src/ATen/native/cuda/AbsKernel.cu`,
  `UnaryOpsKernel.cu`, and `Activation*Kernel.cu`.

- REQ-3: Eight orthogonal-polynomial kernels — Chebyshev T/U/V/W,
  Hermite H/He, Laguerre L, Legendre P. Each is a three-term
  recurrence implemented as a bounded `for k in 1..n_u { ... }` loop
  in registers. `n` (degree) is passed as a `u32` scalar at launch
  time, then converted to `usize` inside the kernel. Mirrors the
  CPU evaluators in `ferrotorch-core/src/special.rs` which themselves
  mirror upstream PyTorch's `chebyshev_polynomial_t` family
  (`aten/src/ATen/native/Math.h` + `aten/src/ATen/native/cuda/Math.cuh`).

- REQ-4: Naive matmul — `kernel_matmul_naive<F: Float>(a, b, out, m,
  k, n)`. One cube-unit per output element computing
  `out[r, c] = Σ_i a[r, i] * b[i, c]`. Row-major. `m`, `k`, `n` are
  `u32` scalars converted to `usize` inside the kernel. Mirrors
  upstream's pre-cuBLAS naive matmul (used as a correctness oracle in
  `aten/src/ATen/native/cuda/Matmul.cu`'s test harness; ferrotorch uses
  it as the production matmul today, with cuBLAS routing planned but
  not implemented in this crate).

- REQ-5: Slice-upload runner helpers — `fn run_unary<R,L>`, `fn
  run_binary<R,L>` — each uploads input slices via
  `client.create_from_slice`, allocates an output buffer via
  `client.empty(size_bytes)`, computes `(count, dim)` via
  `crate::elementwise_launch_dims`, builds `ArrayArg::from_raw_parts`
  with the proven element count, invokes the caller-provided closure
  with the launcher. Returns `(out_handle, n)`. No readback.

- REQ-6: Device-handle-direct runner helpers — `fn run_unary_handle`,
  `fn run_binary_handle` — same shape as REQ-5 but takes pre-uploaded
  `cubecl::server::Handle` inputs. Calls `crate::
  debug_assert_handle_capacity::<f32>(&h, n)` before each
  `ArrayArg::from_raw_parts`. Enables the no-H2D-round-trip path
  (#673).

- REQ-7: Macro-generated `pub fn run_<op>` (slice-upload) and
  `pub fn run_<op>_handle` (handle-direct) for each elementwise kernel
  via the `define_unary_runner!` / `define_binary_runner!` macros.
  Each generated function dispatches the kernel via
  `<kernel>::launch_unchecked::<f32, R>(client, count, dim, ...)`.

- REQ-8: Polynomial runner helpers `fn run_unary_with_n` (slice-upload)
  and `fn run_unary_with_n_handle` (handle-direct) take an extra
  `n: u32` scalar threaded through the launcher. Macro-stamped via
  `define_unary_with_n_runner!` and `define_unary_with_n_runner_handle!`.

- REQ-9: Matmul runners — `pub fn run_matmul<R>` (slice-upload) and
  `pub fn run_matmul_handle<R>` (handle-direct). Both compute
  `out_len = m * n`, allocate output via `client.empty`, dispatch the
  kernel with three `ArrayArg::from_raw_parts` calls plus `m as u32`,
  `k as u32`, `n as u32` scalars. The handle-direct variant calls
  `debug_assert_handle_capacity::<f32>` on each input.

- REQ-10: All `unsafe { ... }` blocks carry SAFETY comments documenting
  the cubecl-side invariants: handles alloc'd by this client,
  `ArrayArg::from_raw_parts` element count matches the kernel's
  `&Array<F>` view, `.clone()` on a handle is a refcount bump (not a
  buffer copy), `launch_unchecked` skips runtime arity checks per
  cubecl convention.

## Acceptance Criteria

- [x] AC-1: All elementwise binary kernel `pub fn run_<op>` /
  `pub fn run_<op>_handle` exist as documented in the
  `define_binary_runner!` macro invocations (add, sub, mul, div).
- [x] AC-2: All elementwise unary kernel `pub fn run_<op>` /
  `pub fn run_<op>_handle` exist (relu, neg, abs, exp, ln, sqrt, sin,
  cos, tanh, sigmoid).
- [x] AC-3: All polynomial kernel `pub fn run_<op>` /
  `pub fn run_<op>_handle` exist (chebyshev t/u/v/w, hermite h/he,
  laguerre l, legendre p).
- [x] AC-4: Matmul `pub fn run_matmul` and `pub fn run_matmul_handle`
  exist with the documented `(m, k, n)` signature.
- [x] AC-5: Every `unsafe` block has a SAFETY comment immediately
  above it. (Grep `unsafe {` count = SAFETY-comment count in this
  file.)
- [x] AC-6: `cargo test -p ferrotorch-cubecl --no-default-features`
  passes (kernels do not have direct no-backend tests; the parent
  `ops.rs::no_backend_tests` is the integration verification).

## Architecture

### Elementwise kernel idiom (REQ-1, REQ-2)

Every binary kernel follows the pattern:

```rust
#[cube(launch_unchecked)]
pub fn kernel_add<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}
```

`ABSOLUTE_POS` is cubecl's global-thread-id intrinsic. The bounds-guard
is required because the launcher rounds element count up to the next
multiple of `units_per_cube` (256), so the last cube has some idle
threads that must NOT write out-of-bounds.

Unary kernels follow the same pattern with one input array; activation
functions (`relu`, `sigmoid`, `tanh`) use cubecl's `F::max`,
`F::exp`, `F::tanh` math intrinsics which compile to the backend's
native function.

Each kernel is generic over `F: Float` so the same source compiles for
`f32`, `f16`, `bf16`, `f64` — though today's runners are all f32-concrete
(`launch_unchecked::<f32, R>`).

### Polynomial recurrences (REQ-3)

Each polynomial kernel evaluates a three-term recurrence with hard-coded
initial conditions. For Chebyshev T:

```rust
let mut prev2 = F::new(1.0);  // T_0
let mut prev1 = xv;           // T_1
for _ in 2..=n_u {
    let next = two_x * prev1 - prev2;
    prev2 = prev1;
    prev1 = next;
}
out[ABSOLUTE_POS] = prev1;
```

The `n_u == 0` and `n_u == 1` cases are handled before the loop. Other
families use the same scaffolding with different recurrences (Hermite H
multiplies the trailing term by `2k`, Laguerre divides by `k+1`, etc.).

CPU mirror: `ferrotorch-core/src/special.rs` evaluates these in f64;
the GPU stays in `F` (f32 today) so the result lives entirely on device.

### Naive matmul (REQ-4)

`#[cube(launch_unchecked)] pub fn kernel_matmul_naive<F: Float> in
kernels.rs`. One cube-unit computes one output element via a `k`-long
inner loop accumulating `a[row * k_u + i] * b[i * n_u + col]`. The
`m`, `k`, `n` scalars are `u32` at the launch boundary, converted to
`usize` inside the kernel because `ABSOLUTE_POS` and `Array::len` are
both `usize` in cubecl.

This is the bottom of the matmul performance hierarchy — no tiling, no
register blocking, no warp-level reductions. For ferrotorch the
correctness contract (numerical match with CPU + small-input tests in
`ops.rs::tests::portable_matmul_*`) is what matters; tiled fast paths
are a future optimization that doesn't block translation completeness.

### Slice-upload helper (REQ-5)

`fn run_unary<R, L> in kernels.rs` is the per-op slice-upload runner.
It:

1. Allocates `out_handle = client.empty(size_bytes)`.
2. Uploads `x_handle = client.create_from_slice(f32::as_bytes(x))`.
3. Computes `(count, dim) = crate::elementwise_launch_dims(n as u32)`.
4. Builds `in_arg = unsafe { ArrayArg::from_raw_parts(x_handle, n) }`.
5. Builds `out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(),
   n) }`. The `.clone()` is a cubecl refcount bump (not a copy); kernel
   writes through `out_arg` are visible via the returned `out_handle`.
6. Invokes `launcher(client, count, dim, in_arg, out_arg)` (the
   caller-provided closure that knows the specific kernel symbol).
7. Returns `(out_handle, n)`.

`run_binary` is the same with two inputs and a `debug_assert_eq!
(a.len(), b.len())`.

### Handle-direct helper (REQ-6)

`fn run_unary_handle in kernels.rs` skips step 2 from above — it takes
the pre-uploaded handle directly. Adds
`crate::debug_assert_handle_capacity::<f32>(&x_handle, n)` for
debug-build validation that the caller-provided handle has enough
bytes. Release builds rely on the caller contract.

`run_binary_handle` is the same with two pre-uploaded handles.

### Macro-stamped runners (REQ-7, REQ-8)

`macro_rules! define_unary_runner` and `define_binary_runner` each
stamp out two `pub fn`s per op: `run_<op>` (slice-upload) and
`run_<op>_handle` (handle-direct). The macros are invoked once per
kernel:

- Binary: `add, sub, mul, div` — 4 invocations.
- Unary: `relu, neg, abs, exp, ln, sqrt, sin, cos, tanh, sigmoid` — 10
  invocations.

The polynomial side uses `define_unary_with_n_runner!` (slice-upload)
and `define_unary_with_n_runner_handle!` (handle-direct), threading
through the extra `degree` scalar:

- 8 invocations × 2 macros = 16 `pub fn` symbols.

### SAFETY discipline (REQ-10)

Every `unsafe { ... }` block in this file is paired with a multi-line
SAFETY comment that names the handle allocations, the
`ArrayArg::from_raw_parts` element-count invariant, the
`.clone()`-as-refcount fact, and the `launch_unchecked` convention.
This is the standard ferrotorch SAFETY pattern; the comments are
verbose because cubecl's API is `unsafe`-heavy and a future reader needs
to verify each invariant locally. R-CODE-1 explicitly grandfathers leaf
primitives (cubecl kernel launches) as `unsafe`-allowed.

Non-test production consumer (for the entire kernel file): every
`portable_*` op in `ops.rs` dispatches through one of these runners.
For example `portable_add` at `ops.rs:230` calls `kernels::run_add` or
`kernels::run_add_handle` via the `dispatch_binary!` macro. Through
`ops.rs`, every kernel here gates an `ferrotorch-xpu::xpu_*` op via
the `xpu_binary!`/`xpu_unary!`/`xpu_polynomial!` macros at
`ferrotorch-xpu/src/lib.rs`.

## Parity contract

ferrotorch-cubecl is INFRASTRUCTURE — `parity_ops = []`. Per-op parity
is enforced at the `ferrotorch-core` op layer (e.g. `add`, `mul`,
`tanh`'s parity-sweep ops); the GPU implementations here pair with
those CPU implementations via numerical agreement.

Edge cases preserved by these kernels:

- **NaN propagation**: f32 arithmetic preserves NaN per IEEE-754 —
  `NaN + 1.0 = NaN`. Verified through `ops.rs::tests::portable_*` end-
  to-end (no explicit NaN test in this file).
- **±Inf**: Same — `exp(very_large) = +Inf` on f32. Verified through
  the `ops.rs::tests` end-to-end with `exp(big)` checks.
- **Empty input** (`n == 0`): `elementwise_launch_dims(0).max(1)`
  rounds to 1 cube; the kernel's `ABSOLUTE_POS < out.len()` guard
  ensures no writes. End-to-end tested via the empty-tensor
  conformance suite in `ferrotorch-core/tests/`.
- **Degree-0 polynomial**: each polynomial kernel special-cases
  `n_u == 0` returning the appropriate constant (1 for Chebyshev T/U/V/W,
  Hermite He, Laguerre L, Legendre P; 1 for Hermite H). Verified by
  `portable_chebyshev_t_n0_returns_ones` in `ops.rs::tests`.

## Verification

This file's kernels are exercised through `ops.rs::tests` (all `cfg
(test, feature = "wgpu")` tests) and `quant.rs::cuda_tests` /
`grammar.rs::cuda_tests` (cfg `cuda`). The most relevant:

- `ops.rs::tests::portable_add_runs_on_gpu`
- `ops.rs::tests::portable_matmul_runs_on_gpu`
- `ops.rs::tests::portable_matmul_square_8x8`
- `ops.rs::tests::portable_chebyshev_t_runs_on_gpu` (and all 8
  polynomial families)
- `ops.rs::tests::portable_polynomial_handles_large_input` (1024 elems
  to exercise multi-cube launch geometry)

Smoke command (no parity ops):

```bash
cargo test -p ferrotorch-cubecl --no-default-features 2>&1 | tail -3
```

Expected: ≥ 1 `test result: ok`.

With `--features wgpu` (when a wgpu adapter exists):

```bash
cargo test -p ferrotorch-cubecl --features wgpu 2>&1 | tail -3
```

Expected: every `portable_*` test passes on the adapter.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub fn kernel_add/sub/mul/div<F: Float> in kernels.rs`. Non-test consumer: `ferrotorch-cubecl/src/ops.rs` — `portable_add/sub/mul/div` dispatch via `kernels::run_add/sub/mul/div`; downstream `ferrotorch-xpu/src/lib.rs` wraps these into `xpu_add/sub/mul/div`. |
| REQ-2 | SHIPPED | impl: ten `pub fn kernel_relu/neg/abs/exp/ln/sqrt/sin/cos/tanh/sigmoid<F: Float> in kernels.rs`. Non-test consumer: `ops.rs::portable_relu/neg/abs/exp/ln/sqrt/sin/cos/tanh/sigmoid` (lines 290, 413-421); downstream `ferrotorch-xpu/src/lib.rs` wraps these into `xpu_*`. |
| REQ-3 | SHIPPED | impl: eight `pub fn kernel_chebyshev_{t,u,v,w}/hermite_{h,he}/laguerre_l/legendre_p<F: Float> in kernels.rs`. Non-test consumer: `ops.rs::portable_*_polynomial_*` (lines 508-555); downstream `ferrotorch-xpu/src/lib.rs` wraps via `xpu_polynomial!`. |
| REQ-4 | SHIPPED | impl: `pub fn kernel_matmul_naive<F: Float> in kernels.rs`. Non-test consumer: `pub fn run_matmul/run_matmul_handle in kernels.rs` → `ops.rs::portable_matmul` (line 314) → `ferrotorch-xpu/src/lib.rs::xpu_matmul`. |
| REQ-5 | SHIPPED | impl: `fn run_unary/run_binary<R, L> in kernels.rs`. Non-test consumer: every macro-generated `pub fn run_<op>` in this file invokes one of them; those are called from `ops.rs` dispatch macros. |
| REQ-6 | SHIPPED | impl: `fn run_unary_handle/run_binary_handle<R, L> in kernels.rs`. Non-test consumer: every macro-generated `pub fn run_<op>_handle` in this file invokes one; those are called from `ops.rs::dispatch_*` macros' handle-direct arms. |
| REQ-7 | SHIPPED | impl: `define_unary_runner!` and `define_binary_runner!` macros invoked 4 + 10 times in `kernels.rs`. Non-test consumer: `ferrotorch-cubecl/src/ops.rs` dispatches via the generated `kernels::run_*` / `kernels::run_*_handle` symbols. |
| REQ-8 | SHIPPED | impl: `define_unary_with_n_runner!` + `define_unary_with_n_runner_handle!` invoked 8 times each in `kernels.rs`. Non-test consumer: `ops.rs::portable_*_polynomial_*` (lines 508-555) dispatches via the generated `kernels::run_<poly>` / `kernels::run_<poly>_handle`. |
| REQ-9 | SHIPPED | impl: `pub fn run_matmul/run_matmul_handle<R: Runtime> in kernels.rs`. Non-test consumer: `ops.rs::portable_matmul` (line 314) via `dispatch_matmul!` macro. |
| REQ-10 | SHIPPED | impl: every `unsafe { ArrayArg::from_raw_parts(...) }` and `unsafe { kernel_*::launch_unchecked(...) }` in `kernels.rs` is preceded by a SAFETY comment naming the alloc site, element count, refcount semantics, and `launch_unchecked` convention. Non-test consumer: same as REQ-7/8/9 — every kernel dispatched through these runners inherits the documented safety contract. |
