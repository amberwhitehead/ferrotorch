# torch.special — Cephes special functions (entr / ndtr / ndtri / i0-family / zeta / airy / bessel-k / spherical-bessel)

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/Math.cuh
  - aten/src/ATen/native/Math.h
  - aten/src/ATen/native/UnaryOps.cpp
  - torch/special/__init__.py
  - torch/_torch_docs.py
-->

## Summary

This doc specifies a new family of `torch.special.*` functions that ferrotorch
is MISSING ENTIRELY (0 production hits in `ferrotorch-core` / `ferrotorch-gpu`
as of this commit — verified by `grep -rn "fn entr|fn ndtr|fn ndtri|fn i0\b|fn
i1\b|fn zeta|airy_ai|spherical_bessel" ferrotorch-core/src ferrotorch-gpu/src`
returning empty). It extends the existing `ferrotorch-core/src/special.rs`
home (see sibling doc `.design/ferrotorch-core/special.md` for the already-
SHIPPED erf / gamma / orthogonal-polynomial families and the REQ-table style
this doc follows). Every REQ here is **NOT-STARTED**: the contract below is the
upstream spec the acto-builder must translate (CPU first, then a GPU kernel in
`ferrotorch-gpu/src/special.rs` mirroring the existing polynomial-kernel
pattern). The tracking blocker for ALL of this work is **#1651**.

All new ops are elementwise unary `op(input, *, out=None) -> Tensor` except
`zeta(input, other) -> Tensor` which is elementwise binary. The user-facing
Python surface is registered in `torch/special/__init__.py` (signatures cited
per REQ); the scalar math comes from the Cephes library ports in
`aten/src/ATen/native/cuda/Math.cuh` (the jiterator string kernels) and
`aten/src/ATen/native/Math.h` (the CPU `calc_*` templates), which share
coefficient sets.

## Requirements

All requirements are NOT-STARTED. Each names the torch user-facing signature,
the upstream `aten` `file:line` carrying the math/Cephes contract, and the
domain / NaN / inf edge behavior to match byte-for-relevant-tolerance
(R-DEV-1).

- REQ-B1: `entr(input)` — entropy. `torch.special.entr(input, *, out=None)`
  (`torch/special/__init__.py:67-70`). Contract from
  `aten/src/ATen/native/cuda/Math.cuh:463-480` (`entr_string`):
  `x != x (NaN) -> x`; `x > 0 -> -x * log(x)`; `x == 0 -> 0`;
  `x < 0 -> -INFINITY`. The NaN check comes FIRST, then the `>0` / `==0` / else
  ladder. Note `entr(0) = +0.0` (not `-0.0`).

- REQ-B2: `ndtr(input)` — standard-normal CDF.
  `torch.special.ndtr(input, *, out=None)` (`torch/special/__init__.py:624-627`).
  Contract from `aten/src/ATen/native/UnaryOps.cpp:715-718` (`calc_ndtr`):
  `ndtr(x) = (1 + erf(x * M_SQRT1_2)) * 0.5` with
  `M_SQRT1_2 = 0.70710678118654752440` (1/sqrt(2)). This is a COMPOSITE over
  `erf`, which ferrotorch already ships (REQ-1 of `special.md`, `pub fn erf in
  special.rs`) — the implementation reuses `erf_scalar` so the f64 SunPro-fdlibm
  ~1-ulp `erf` path flows through. Edge behavior is inherited from `erf`:
  `ndtr(-inf) = 0`, `ndtr(0) = 0.5`, `ndtr(+inf) = 1`, `ndtr(NaN) = NaN`.

- REQ-B3: `ndtri(input)` — inverse standard-normal CDF (quantile function).
  `torch.special.ndtri(input, *, out=None)` (`torch/special/__init__.py:649-657`,
  documented as `ndtri(p) = sqrt(2) * erfinv(2p - 1)`). The shipped
  implementation must port the Cephes RATIONAL approximation from
  `aten/src/ATen/native/cuda/Math.cuh:48-173` (`ndtri_string`), NOT the
  `erfinv` composition (torch uses the direct Cephes kernel for ULP parity).
  Domain `(0, 1)`. Contract: `y0 == 0 -> -INFINITY`; `y0 == 1 -> +INFINITY`;
  `y0 < 0 || y0 > 1 -> NaN`. Interior uses `polevl` (Horner, coefficients in
  REVERSE order, `len = len(A)` NOT `len(A)-1` — see the note at
  `Math.cuh:31-33`). Three coefficient regions:
  - central `|y - 0.5| <= 3/8` (i.e. `y > exp(-2) = 0.13533528323661269189`):
    `P0[5]`/`Q0[9]`, `s2pi = 2.50662827463100050242`,
    `x = y + y*(y2 * P0(y2)/Q0(y2))`, return `x * s2pi` (`Math.cuh:73-102`).
  - tail with `x = sqrt(-2 log y) < 8` (`y` between `exp(-2)` and `exp(-32)`):
    `P1[9]`/`Q1[9]` (`Math.cuh:111-139`).
  - far tail `x >= 8` (`y` between `exp(-32)` and `exp(-2048)`): `P2[9]`/`Q2[9]`
    (`Math.cuh:140-168`).
  The `code` flag (set false when `y > 1 - exp(-2)`, replacing `y` with `1 - y`)
  flips the final sign: `return code ? -x : x` (`Math.cuh:68-71, 171-172`).

- REQ-B4: `i0(input)` — modified Bessel function of the first kind, order 0.
  `torch.special.i0(input, *, out=None)` (`torch/special/__init__.py:522-525`;
  also `torch.i0`). Contract from `aten/src/ATen/native/cuda/Math.cuh:484-556`
  (`i0_string`, the `chbevl` Chebyshev evaluator + `i0`). Even function (takes
  `fabs(_x)`). Two regions:
  - `x <= 8`: `A[30]` coefficients (Chebyshev for `exp(-x) I0(x)` on `[0,8]`,
    `lim_{x->0} = 1`), `y = x/2 - 2`, return `exp(x) * chbevl(y, A, 30)`.
  - `x > 8`: `B[25]` coefficients (Chebyshev for `exp(-x) sqrt(x) I0(x)` on
    `[8, inf]`, `lim = 1/sqrt(2pi)`), return
    `exp(x) * chbevl(32/x - 2, B, 25) / sqrt(x)`.
  `chbevl` (`Math.cuh:485-500`): Clenshaw `b0 = array[0]; b1 = 0;` then
  `b0 = x*b1 - b2 + array[i]`, return `0.5*(b0 - b2)`. `i0(0) = 1`,
  `i0(+/-inf) = +inf`, `i0(NaN) = NaN`.

- REQ-B5: `i0e(input)` — exponentially-scaled `i0`: `exp(-|x|) * I0(x)`.
  `torch.special.i0e(input, *, out=None)` (`torch/special/__init__.py:548-551`).
  Contract from `aten/src/ATen/native/Math.h:101-135` (`calc_i0e`): same `A[30]`
  / `B[25]` Chebyshev sets as REQ-B4 but WITHOUT the `exp(x)` factor —
  `x <= 8`: `chbevl(x/2 - 2, A, 30)`; `x > 8`:
  `chbevl(32/x - 2, B, 25) / sqrt(x)`. `i0e(0) = 1`, `i0e(+/-inf) = 0`,
  `i0e(NaN) = NaN`. f32 uses a SHORTER coefficient set than f64 (see
  `chebyshev_coefficients_i0e_A/B` templated by scalar type at
  `Math.cuh:3183-3232`); the builder must key coefficient length on `T`.

- REQ-B6: `i1(input)` — modified Bessel, first kind, order 1.
  `torch.special.i1(input, *, out=None)` (also `torch.i1`). Contract from
  `aten/src/ATen/native/Math.h:1500-1518` (`calc_i1`) and
  `aten/src/ATen/native/cuda/Math.cuh:558-623` (`i1_string`). ODD function:
  takes `fabs(_x)` for the kernel then negates if `_x < 0`. Two regions:
  - `x <= 8`: `i1e_A` coefficients (29 for f64), `y = x/2 - 2`,
    `out = exp(x) * x * chbevl(y, A, 29)`.
  - `x > 8`: `i1e_B` coefficients (25 for f64),
    `out = exp(x) * chbevl(32/x - 2, B, 25) / sqrt(x)`.
  Return `(_x < 0) ? -out : out`. `i1(0) = 0`, `i1(+inf) = +inf`,
  `i1(-inf) = -inf`, `i1(NaN) = NaN`.

- REQ-B7: `i1e(input)` — exponentially-scaled `i1`: `exp(-|x|) * I1(x)`.
  `torch.special.i1e(input, *, out=None)` (`torch/special/__init__.py:598-601`).
  Contract from `aten/src/ATen/native/Math.h:1530-1544` (`calc_i1e`) and
  `aten/src/ATen/native/cuda/Math.cuh:625-745` (`i1e_string`). Same `i1e_A` /
  `i1e_B` sets as REQ-B6 but WITHOUT the `exp(x)` factor —
  `x <= 8`: `out = chbevl(x/2 - 2, A) * x`; `x > 8`:
  `out = chbevl(32/x - 2, B) / sqrt(x)`. Return `(_x < 0) ? -out : out`.
  ODD function. `i1e(0) = 0`, `i1e(+/-inf) = +/-0`, `i1e(NaN) = NaN`. f32 uses
  a shorter coefficient set (17 / 7) than f64 (29 / 25), keyed on `T`
  (`Math.cuh:646-744`).

- REQ-B8: `zeta(input, other)` — Hurwitz zeta `zeta(x, q)`.
  `torch.special.zeta(input, other, *, out=None)` (binary; `torch/special/__init__.py`).
  Contract from `aten/src/ATen/native/cuda/Math.cuh:299-383` (`zeta_string`,
  the Cephes Hurwitz-zeta with the `A[12]` Bernoulli-derived tail series and
  `MACHEP = 1.11022302462515654042E-16`). Edge ladder:
  - `x == 1 -> +INFINITY`;
  - `x < 1 -> NaN`;
  - `q <= 0`: if `q == floor(q)` (non-positive integer) `-> +INFINITY`; else if
    `x != floor(x)` `-> NaN`;
  - interior: `s = pow(q, -x)` then the `while ((i < 9) || (a <= 9.0))`
    accumulation loop with early `MACHEP`-relative convergence, followed by the
    Euler-Maclaurin tail `s += b*w/(x-1) - 0.5*b + sum_{i<12} a*b/A[i]`
    (`Math.cuh:345-381`).
  NOTE: `ferrotorch-distributions/src/special_fns.rs` has a PRIVATE
  `hurwitz_zeta_scalar` (anchor `fn hurwitz_zeta_scalar in
  ferrotorch-distributions/src/special_fns.rs`) used by `polygamma`, but it
  takes an INTEGER order `s: u32` only — it is a usable REFERENCE for the tail
  series but is NOT the `torch.special.zeta` core op, which takes a REAL `x`
  over two tensors. The builder must port the full real-`x` Cephes kernel into
  `ferrotorch-core/src/special.rs` (the public `torch.special` home), not reuse
  the integer-only distributions helper.

- REQ-B9: `airy_ai(input)` — Airy function Ai(x).
  `torch.special.airy_ai(input, *, out=None)` (`torch/special/__init__.py:982-985`).
  Contract from `aten/src/ATen/native/cuda/Math.cuh:1280-1459`
  (`airy_ai_string`, `airy_ai_forward`): a multi-region Cephes rational/series
  approximation. The kernel has NO special NaN/inf short-circuit ladder — it
  feeds `x` directly through the region selection (NaN/inf propagate through the
  arithmetic). The builder must transcribe ALL coefficient tables and region
  boundaries verbatim from `Math.cuh:1282-1458`.

- REQ-B10: `spherical_bessel_j0(input)` — spherical Bessel `j0(x) = sin(x)/x`.
  `torch.special.spherical_bessel_j0(input, *, out=None)`
  (`torch/special/__init__.py:1444+`). Contract from
  `aten/src/ATen/native/cuda/Math.cuh:3039-3052`
  (`spherical_bessel_j0_forward`): `isinf(x) -> 0`; `|x| < 0.5 -> ` the explicit
  6-term Taylor `1 + x^2*(-1/6 + x^2*(1/120 + x^2*(-1/5040 + x^2*(1/362880 +
  x^2*(-1/39916800 + x^2*(1/6227020800))))))`; else `sin(x)/x`. `j0(0) = 1`
  (via the Taylor branch), `j0(NaN) = NaN`.

- REQ-B11: `modified_bessel_k0(input)` / `scaled_modified_bessel_k0(input)` —
  modified Bessel second kind order 0 + exp-scaled.
  `torch.special.modified_bessel_k0(input, *, out=None)` /
  `torch.special.scaled_modified_bessel_k0(input, *, out=None)`
  (`torch/special/__init__.py:1304-1341`). Contract from
  `aten/src/ATen/native/cuda/Math.cuh:2501-2657`
  (`modified_bessel_k0_forward` / `scaled_modified_bessel_k0_forward`). Uses
  `A[10]` and `B[25]` coefficient sets (`Math.cuh:2504-2543`). Edge:
  `x == 0 -> +INFINITY`; `x < 0 -> NaN`. Two regions split at `x <= 2`. The
  scaled variant multiplies the `k0` result by `exp(x)` (equivalently omits the
  `exp(-x)` factor); for large `x`, `k0(x) -> 0` while
  `scaled_modified_bessel_k0(x) -> sqrt(pi/(2x))`.

- REQ-B12: `modified_bessel_k1(input)` / `scaled_modified_bessel_k1(input)` —
  modified Bessel second kind order 1 + exp-scaled.
  `torch.special.modified_bessel_k1(input, *, out=None)` /
  `torch.special.scaled_modified_bessel_k1(input, *, out=None)`
  (`torch/special/__init__.py:1321-1358`). Contract from
  `aten/src/ATen/native/cuda/Math.cuh:2659-2817`
  (`modified_bessel_k1_forward` / `scaled_modified_bessel_k1_forward`). Distinct
  `A`/`B` coefficient sets from REQ-B11. Edge: `x == 0 -> +INFINITY`;
  `x < 0 -> NaN`. Two regions split at `x <= 2`.

## Acceptance Criteria

All criteria are currently failing (the ops do not exist). Each becomes
mechanically dischargeable when the corresponding REQ ships.

- [x] AC-B1: `entr` CPU impl exists; `entr(0.5) == -0.5*ln(0.5)`,
  `entr(0.0) == 0.0` (with `+0.0` sign), `entr(-1.0) == -inf`,
  `entr(NaN).is_nan()`; matches live `torch.special.entr` to f64 tolerance.
  (SHIPPED #1651 batch 1 — `entr_*_vs_torch in special.rs`.)
- [x] AC-B2: `ndtr` CPU impl exists; `ndtr(0.0) == 0.5`, `ndtr(-inf) == 0.0`,
  `ndtr(+inf) == 1.0`; matches live `torch.special.ndtr` on
  `[-3,-2,-1,0,1,2,3]` to f64 transcendental tolerance.
  (SHIPPED #1651 batch 1 — `ndtr_*_vs_torch in special.rs`.)
- [x] AC-B3: `ndtri` CPU impl exists; `ndtri(0.5) == 0.0`, `ndtri(0.0) == -inf`,
  `ndtri(1.0) == +inf`, `ndtri(-0.1).is_nan()`, `ndtri(1.1).is_nan()`;
  matches live `torch.special.ndtri` on `[0.025,0.25,0.5,0.75,0.975]` to f64
  tolerance; round-trips `ndtr(ndtri(p)) ≈ p`.
  (SHIPPED #1651 batch 1 — `ndtri_*_vs_torch` / `ndtr_ndtri_roundtrip in special.rs`.)
- [ ] AC-B4: `i0` CPU impl exists; `i0(0.0) == 1.0`, `i0` even, matches live
  `torch.special.i0` on `[0,1,2,5,10,20]` to tolerance.
- [ ] AC-B5: `i0e` CPU impl exists; `i0e(0.0) == 1.0`, `i0e` even,
  `i0e(x) == exp(-|x|)*i0(x)` for finite `x`; matches live `torch.special.i0e`.
- [ ] AC-B6: `i1` CPU impl exists; `i1(0.0) == 0.0`, `i1` odd, matches live
  `torch.special.i1`.
- [ ] AC-B7: `i1e` CPU impl exists; `i1e(0.0) == 0.0`, `i1e` odd,
  `i1e(x) == exp(-|x|)*i1(x)`; matches live `torch.special.i1e`.
- [ ] AC-B8: `zeta` CPU impl exists; `zeta(2.0, 1.0) == pi^2/6`,
  `zeta(1.0, q) == +inf`, `zeta(0.5, q).is_nan()`, `zeta(x, q<=0 integer)
  == +inf`; matches live `torch.special.zeta` on a grid of `(x>1, q>0)`.
- [ ] AC-B9: `airy_ai` CPU impl exists; matches live `torch.special.airy_ai`
  on `[-5,-2,-1,0,1,2,5]` to tolerance.
- [ ] AC-B10: `spherical_bessel_j0` CPU impl exists; `j0(0.0) == 1.0`,
  `j0(inf) == 0.0`, matches live `torch.special.spherical_bessel_j0`.
- [ ] AC-B11: `modified_bessel_k0` / `scaled_modified_bessel_k0` CPU impls
  exist; `k0(0) == +inf`, `k0(-1).is_nan()`, match live torch.
- [ ] AC-B12: `modified_bessel_k1` / `scaled_modified_bessel_k1` CPU impls
  exist; `k1(0) == +inf`, `k1(-1).is_nan()`, match live torch.
- [~] AC-B13: GPU kernels exist in `ferrotorch-gpu/src/special.rs`, dispatched
  on-device (no host round trip, R-CODE-4), matching the CPU path to
  f32 tolerance — mirroring the polynomial-kernel pattern (`special_gpu_simple`
  / `GpuBackend` methods, `special.md` AC-7). **entr/ndtr/ndtri SHIPPED for
  f32** (#1651 batch 1): `ENTR_F32_PTX` / `NDTR_F32_PTX` / `NDTRI_F32_PTX` +
  `gpu_{entr,ndtr,ndtri}_f32`, verified on the RTX 3090 by
  `*_on_device_matches_torch in ferrotorch-gpu/src/special.rs`. **f64 CUDA
  returns `NotImplementedOnCuda`** (honest, no host round trip): base PTX
  (`Ptx::from_src`, no libdevice link) has no `lg2.approx.f64` / `ex2.approx.f64`,
  so the f64 log/exp these transcendentals need cannot be evaluated at f64
  precision on-device — the same constraint that routes general f64
  transcendentals off-device for `cdist_f64` (`distance.rs:211-219`). bf16/f16
  CUDA inputs return `NotImplementedOnCuda` (rejected in `special_gpu_simple`
  before any device call). The i0..k1 / zeta / airy families (batches 2, 3)
  remain NOT-STARTED.
- [ ] AC-B14: parity-sweep runner arms exist for each op and report
  `passed (0 skipped, 0 failed)` at `--seeds 8`. Per goal.md S5 / R-DEFER-6 the
  missing runner arm is ONE test-infrastructure follow-up blocker for the whole
  torch.special-transcendentals family, NOT a per-op REQ blocker; entr/ndtr/ndtri
  are SHIPPED on impl + non-test consumer + lib tests (live-torch-2.11 oracle,
  R-CHAR-3) + clippy clean. Runner-arm wiring tracked under #1651.

## Architecture

All scalar evaluators land in `ferrotorch-core/src/special.rs` (the
`torch.special` home), each as a private `*_scalar<T: Float>` function wired
through `unary_map` (or `binary_map` for `zeta`) from
`crate::ops::elementwise`, exactly like the SHIPPED erf/gamma family (e.g.
`fn erf_scalar in special.rs` -> `pub fn erf in special.rs`). The shared
`chbevl` Clenshaw evaluator (`Math.cuh:485-500`) becomes one private
`fn chbevl in special.rs` reused by REQ-B4..B7, B11, B12. The `polevl`
reverse-order Horner evaluator (`Math.cuh:30-39`, note `len = len(A)`) becomes
one private `fn polevl in special.rs` reused by REQ-B3.

The public functions (`pub fn entr`, `pub fn ndtr`, `pub fn ndtri`,
`pub fn i0`, `pub fn i0e`, `pub fn i1`, `pub fn i1e`, `pub fn zeta`,
`pub fn airy_ai`, `pub fn spherical_bessel_j0`, `pub fn modified_bessel_k0`,
`pub fn scaled_modified_bessel_k0`, `pub fn modified_bessel_k1`,
`pub fn scaled_modified_bessel_k1`) are re-exported at the top of
`ferrotorch-core/src/lib.rs` (anchor: the `pub use special::{...}` block), which
is the non-test production consumer per goal.md S5 (the `torch.special` public
surface IS the consumer for boundary ops).

The GPU path mirrors the existing on-device polynomial kernels: new
`GpuBackend` trait methods (anchors `fn entr_f32` / `fn ndtri_f32` / ... in
`ferrotorch-gpu/src/special.rs`) launch PTX kernels carrying the same
coefficient tables; the CUDA branch of each `pub fn` in
`ferrotorch-core/src/special.rs` dispatches CUDA tensors through a
`gpu_simple`-style helper (anchor `fn poly_gpu_simple in special.rs` is the
template) with `Ok(None)` CPU-fallthrough and `NotImplementedOnCuda` for
bf16/f16.

**Current state — batch 1 (entr/ndtr/ndtri) SHIPPED under #1651; batches 2-3
NOT-STARTED.** `entr`/`ndtr`/`ndtri` exist end-to-end (CPU f32+f64, GPU f32
on-device, re-exported consumer, live-torch tests). The i0/i0e/i1/i1e
(batch 2) and zeta/airy_ai/spherical_bessel_j0/modified_bessel_k0/k1
(batch 3) families remain NOT-STARTED — `grep -rn "fn i0\b|fn zeta|airy_ai"`
returns empty. Those batches stay gated on #1651.

### Recommended build batches (tractability order)

The builder should take these in order; each batch shares an upstream region
and a coherent test surface (speed discipline S1: batch by upstream construct,
not per-op).

- **Batch 1 — Normal-distribution trio (entr / ndtr / ndtri).** REQ-B1, B2, B3.
  Smallest math, highest reuse: `ndtr` is a one-line composite over the
  already-shipped `erf`; `entr` is a 4-branch ladder; `ndtri` is the only
  nontrivial port (the Cephes `polevl` rational with three regions). Landing
  `polevl` here unblocks nothing else but is self-contained. Start here.
- **Batch 2 — Modified-Bessel-I family (i0 / i0e / i1 / i1e).** REQ-B4..B7.
  All four share the `chbevl` evaluator and the `i1e_A/B`, `i0e_A/B` coefficient
  sets (with the f32-vs-f64 shorter-coefficient-set wrinkle). Land `chbevl` +
  all four coefficient tables in one commit.
- **Batch 3 — Bessel-K + Airy + spherical (zeta / airy_ai /
  spherical_bessel_j0 / modified_bessel_k0 / scaled_k0 / modified_bessel_k1 /
  scaled_k1).** REQ-B8..B12. `k0`/`k1` reuse `chbevl` from Batch 2;
  `spherical_bessel_j0` is a short Taylor+`sin/x`; `airy_ai` is the largest
  single transcription (the multi-region Cephes table at `Math.cuh:1282-1458`);
  `zeta` is the Hurwitz-zeta with the Euler-Maclaurin tail. Largest batch —
  the builder may split `airy_ai` and `zeta` into their own dispatches if the
  coefficient transcription pushes the commit past ~10 files (R-BUILD-5).

GPU kernels (AC-B13) follow each batch's CPU landing, mirroring the SHIPPED
polynomial-kernel GPU pattern in `ferrotorch-gpu/src/special.rs`.

## Parity contract

`parity_ops` for the route currently `= []`. When the builder lands these, the
route's `parity_ops` should be extended to:
`["entr", "ndtr", "ndtri", "i0", "i0e", "i1", "i1e", "zeta", "airy_ai",
"spherical_bessel_j0", "modified_bessel_k0", "scaled_modified_bessel_k0",
"modified_bessel_k1", "scaled_modified_bessel_k1"]`, and the parity-sweep runner
arms wired (AC-B14 / tracked under #1651). Per-op numerical contract (edge cases
the parity oracle must agree on):

- `entr`: NaN->NaN, `0->+0.0`, `x<0 -> -inf`, `x>0 -> -x*ln(x)`.
- `ndtr`: `-inf->0`, `0->0.5`, `+inf->1`, NaN->NaN; ULP inherited from `erf`.
- `ndtri`: domain `(0,1)`; `0->-inf`, `1->+inf`, outside `(0,1) -> NaN`;
  symmetric about `0.5`. SUBTLE: the `code`-flag sign flip and the three
  polevl regions; do NOT shortcut via `erfinv` (loses ULP parity with torch).
- `i0`/`i0e`: even; `i0(0)=1`, `i0e(0)=1`; `i0(inf)=+inf`, `i0e(inf)=0`.
- `i1`/`i1e`: odd; `i1(0)=0`, `i1e(0)=0`; sign follows `_x`.
- `zeta`: `x==1 -> +inf`, `x<1 -> NaN`, `q<=0` integer `-> +inf`, `q<=0`
  non-integer with non-integer `x -> NaN`. SUBTLE: the `while ((i<9) || (a<=9))`
  guard and the MACHEP-relative early exit; convergence is delicate near `x->1+`.
- `airy_ai`: NaN/inf flow through the arithmetic (no explicit short-circuit).
- `spherical_bessel_j0`: `inf->0`, `0->1` (Taylor branch), NaN->NaN.
- `modified_bessel_k0`/`k1` (+scaled): `0->+inf`, `x<0->NaN`; region split at 2.

## Verification

When implemented, `cargo test -p ferrotorch-core --lib special::tests` covers
the new families with (a) edge-case assertions per the AC list and (b)
known-value checks (e.g. `zeta(2,1) == pi^2/6`, `i0(0) == 1`, `ndtri(0.5) == 0`).
GPU agreement is verified in `ferrotorch-gpu/tests/` (asserting `is_cuda()` +
value-match vs the CPU path), mirroring `test_gpu_special_polynomials.rs`.

Parity smoke (per op, once runner arms exist under #1651):

```bash
for OP in entr ndtr ndtri i0 i0e i1 i1e zeta airy_ai spherical_bessel_j0 \
          modified_bessel_k0 scaled_modified_bessel_k0 \
          modified_bessel_k1 scaled_modified_bessel_k1; do
  ./target/release/parity-sweep sweep --op "$OP" --seeds 8 2>&1 \
    | grep -c "passed (0 skipped, 0 failed)"
done
```

Each line must print `>= 1` before the corresponding REQ can move to SHIPPED.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-B1 | SHIPPED | CPU: `entr_scalar` -> `pub fn entr in special.rs` (NaN/`>0`/`==0`/else ladder, `aten/src/ATen/native/cuda/Math.cuh:463-480`). GPU (f32): `ENTR_F32_PTX` + `pub fn gpu_entr_f32 in ferrotorch-gpu/src/special.rs` via `GpuBackend::entr_f32` (`CudaBackendImpl::entr_f32 in backend_impl.rs`); the CUDA branch (`special_gpu_simple`) of `entr in special.rs` dispatches on-device (f64 CUDA -> `NotImplementedOnCuda`, no host round trip). Non-test consumer: re-exported as `ferrotorch_core::entr` (`pub use special::entr in lib.rs`) — S5 torch.special public surface. Tests: live-torch-2.11 oracle (`entr_*_vs_torch in special.rs`, `entr_on_device_matches_torch in ferrotorch-gpu/src/special.rs`). |
| REQ-B2 | SHIPPED | CPU: `ndtr_scalar` -> `pub fn ndtr in special.rs`, composite `(1+erf(x*M_SQRT1_2))*0.5` over shipped `erf_scalar` (`aten/src/ATen/native/UnaryOps.cpp:715-718`). GPU (f32): `NDTR_F32_PTX` + `pub fn gpu_ndtr_f32 in ferrotorch-gpu/src/special.rs` via `GpuBackend::ndtr_f32` (A&S-7.1.26 erf in PTX matching the CPU f32 path); CUDA branch of `ndtr in special.rs` on-device (f64 CUDA -> `NotImplementedOnCuda`). Non-test consumer: `ferrotorch_core::ndtr` (`pub use special::ndtr in lib.rs`). Tests: `ndtr_*_vs_torch in special.rs`, `ndtr_on_device_matches_torch in ferrotorch-gpu/src/special.rs`. |
| REQ-B3 | SHIPPED | CPU: `ndtri_scalar` -> `ndtri_f64` -> `pub fn ndtri in special.rs`, full Cephes 3-region rational + `polevl` reverse-order helper + code-flag sign flip (`aten/src/ATen/native/cuda/Math.cuh:30-39, 48-173`); NOT `sqrt(2)*erfinv`. GPU (f32): `NDTRI_F32_PTX` + `pub fn gpu_ndtri_f32 in ferrotorch-gpu/src/special.rs` via `GpuBackend::ndtri_f32` (`CudaBackendImpl::ndtri_f32 in backend_impl.rs`); CUDA branch of `ndtri in special.rs` on-device (f64 CUDA -> `NotImplementedOnCuda`). Non-test consumer: `ferrotorch_core::ndtri` (`pub use special::ndtri in lib.rs`). Tests: `ndtri_known_values_vs_torch` / `ndtri_cephes_regions_vs_torch` / `ndtri_domain_edges_vs_torch` / `ndtri_f32_vs_torch` / `ndtr_ndtri_roundtrip in special.rs`, `ndtri_on_device_matches_torch` / `ndtri_tail_and_edges_on_device_matches_torch in ferrotorch-gpu/src/special.rs`. |
| REQ-B4 | NOT-STARTED | `i0` does not exist. Open prereq blocker #1651. Upstream: `aten/src/ATen/native/cuda/Math.cuh:484-556`. |
| REQ-B5 | NOT-STARTED | `i0e` does not exist. Open prereq blocker #1651. Upstream: `aten/src/ATen/native/Math.h:101-135`. |
| REQ-B6 | NOT-STARTED | `i1` does not exist. Open prereq blocker #1651. Upstream: `aten/src/ATen/native/Math.h:1500-1518` / `cuda/Math.cuh:558-623`. |
| REQ-B7 | NOT-STARTED | `i1e` does not exist. Open prereq blocker #1651. Upstream: `aten/src/ATen/native/Math.h:1530-1544` / `cuda/Math.cuh:625-745`. |
| REQ-B8 | NOT-STARTED | `zeta` does not exist as a public real-`x` op (only an integer-order private `hurwitz_zeta_scalar` in distributions, not the torch.special op). Open prereq blocker #1651. Upstream Cephes: `aten/src/ATen/native/cuda/Math.cuh:299-383`. |
| REQ-B9 | NOT-STARTED | `airy_ai` does not exist. Open prereq blocker #1651. Upstream: `aten/src/ATen/native/cuda/Math.cuh:1280-1459`. |
| REQ-B10 | NOT-STARTED | `spherical_bessel_j0` does not exist. Open prereq blocker #1651. Upstream: `aten/src/ATen/native/cuda/Math.cuh:3039-3052`. |
| REQ-B11 | NOT-STARTED | `modified_bessel_k0` / `scaled_modified_bessel_k0` do not exist. Open prereq blocker #1651. Upstream: `aten/src/ATen/native/cuda/Math.cuh:2501-2657`. |
| REQ-B12 | NOT-STARTED | `modified_bessel_k1` / `scaled_modified_bessel_k1` do not exist. Open prereq blocker #1651. Upstream: `aten/src/ATen/native/cuda/Math.cuh:2659-2817`. |
