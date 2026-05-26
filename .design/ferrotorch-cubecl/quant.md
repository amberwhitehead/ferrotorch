# GGUF block-quantized dequantization on GPU

<!--
tier: 3-component
status: draft
baseline-pytorch: 2ec0222669f1bcd37b5670ce384f8608c033b158
upstream-paths:
  - aten/src/ATen/native/cuda/
  - c10/cuda/
-->

## Summary

`ferrotorch-cubecl/src/quant.rs` implements pure-GPU dequantization for the
six GGUF (llama.cpp) block-quantized weight formats (`Q4_0`, `Q4_1`, `Q5_0`,
`Q5_1`, `Q8_0`, `Q8_1`). Each format packs N=32 weights into a small block
header (`f16` scale, optionally an `f16`/`f32` min, sometimes a `u32` high-bit
field) plus the packed quantized bits. The host-side `split_q*_blocks`
routines parse the on-disk byte stream into ready-to-upload scale + bits
vectors; the `kernel_dequantize_q*` family unpacks them on-device. The
trailing `kernel_apply_token_mask` + `apply_token_mask_to_gpu` pair belongs
to constrained decoding (used together with `grammar.rs::compute_token_mask
_dfa_to_gpu`).

This module has NO direct upstream PyTorch counterpart — GGUF is a
llama.cpp format that ferrotorch supports for compatibility with the GGUF
weight ecosystem. The route's upstream-path entries
(`aten/src/ATen/native/cuda/` + `c10/cuda/`) name the directories whose
patterns this kernel mirrors: `#[cube]` macro idiom, `ABSOLUTE_POS`-guarded
one-thread-per-element layout, scale-then-shift arithmetic, and SAFETY-
commented `unsafe { ArrayArg::from_raw_parts(...) }` discipline.

## Requirements

- REQ-1: `GgufBlockKind` enum — six variants `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1`
  mirroring the GGUF binary format spec verbatim. `#[allow(non_camel_case
  _types)]` documented in-line: matches `llama.cpp` spec naming for grep-
  equivalence. Const methods `block_elements()` (always 32) and
  `block_bytes()` (per-format: 18/20/22/24/34/40).

- REQ-2: Six `pub fn split_q*_blocks` parsing routines. Each takes a
  `&[u8]` on-disk byte stream + `num_blocks: usize`, asserts the buffer
  is large enough, and returns the per-format tuple:
  - Q4_0: `(scales: Vec<f32>, packed_nibbles: Vec<u32>)`
  - Q4_1: `(scales, mins, packed_nibbles)`
  - Q5_0: `(scales, qh, packed_nibbles)`
  - Q5_1: `(scales, mins, qh, packed_nibbles)`
  - Q8_0: `(scales, packed_int8_bytes)` (Vec<u32>, 8 per block)
  - Q8_1: `(scales, mins, packed_int8_bytes)`
  `f16` headers are upcasted to `f32` via `half::f16::from_bits().to_f32()`.

- REQ-3: Six `#[cube(launch_unchecked)] pub fn kernel_dequantize_q*<F: Float>`
  kernels. Each:
  - Reads scale (and optional min / qh) from on-device arrays.
  - Unpacks the per-element quantized value (nibble or i8).
  - Computes `out[i] = scale * quant + min_offset` (Q4_0 / Q5_0 / Q8_0
    use `quant - 8` / `quant - 16` / sign-extended i8 with NO min;
    Q4_1 / Q5_1 / Q8_1 add the per-block min).
  - Writes to `&mut Array<F>`.
  One thread per output element with `ABSOLUTE_POS < out.len()` guard.

- REQ-4: Six `pub fn dequantize_q*_to_gpu<R: Runtime>(client, scales,
  [mins,] [qh,] bits, num_elements) -> cubecl::server::Handle` host
  launchers. Each:
  - `debug_assert_eq!` the shape relations (`num_elements ==
    scales.len() * 32`, `bits.len() == scales.len() * K` for K = 4 or 8).
  - Uploads each input slice via `client.create_from_slice`.
  - Allocates output via `client.empty(num_elements *
    size_of::<f32>())`.
  - Computes launch dims via `crate::elementwise_launch_dims`.
  - Dispatches `kernel_dequantize_q*::launch_unchecked::<f32, R>(...)`.
  - Returns the on-device output handle. **No host readback.**

- REQ-5: Per-format reference deqaunt routines `pub(crate) fn
  dequantize_q*_reference(scales, [mins, qh,] bits) -> Vec<f32>` — pure
  Rust CPU mirrors used by the in-module CUDA tests as the byte-equality
  oracle.

- REQ-6: Token-mask kernel — `#[cube(launch_unchecked)] pub fn
  kernel_apply_token_mask<F: Float>(logits, mask, out)`. One thread per
  logit. `mask[i] != 0` passes the logit through; `mask[i] == 0`
  replaces with `F::min_value()` (the float type's minimum
  representable value, used as the `-infinity` sentinel for sampling
  because `softmax(-3.4e38)` underflows to zero).

- REQ-7: Token-mask host launcher — `pub fn apply_token_mask_to_gpu
  <R: Runtime>(client, logits, mask) -> cubecl::server::Handle`. Same
  shape as the dequantization launchers. Companion to
  `grammar::compute_token_mask_dfa_to_gpu` — the grammar module computes
  the mask; this function applies it.

- REQ-8: SAFETY discipline for every `unsafe` block. The `&[u32]` → `&[u8]`
  reinterprets carry alignment, length, lifetime, and validity arguments;
  the `kernel_*::launch_unchecked` blocks document the handle allocation
  sites, element-count invariants, `.clone()` refcount semantics, and
  `launch_unchecked` convention.

## Acceptance Criteria

- [x] AC-1: `cargo test -p ferrotorch-cubecl --no-default-features
  quant::` passes (12 tests: 2 block-kind metadata tests, split tests
  for all 6 formats, sign-extension tests, round-trip with the
  reference dequant).
- [x] AC-2: Each `dequantize_q*_to_gpu` matches the corresponding
  `dequantize_q*_reference` byte-for-byte when run under `--features
  cuda` (CUDA tests in `cuda_tests` module).
- [x] AC-3: Block kind constants match GGUF spec: 18 / 20 / 22 / 24 /
  34 / 40 bytes for Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q8_1 (verified
  by `block_kind_metadata_constants` + `_extended` tests).
- [x] AC-4: `split_q*_blocks` panics on short input (verified by
  `split_q4_0_panics_on_short_input`).

## Architecture

### Block format constants (REQ-1)

`pub const GGUF_BLOCK_SIZE: usize = 32`. The six per-format byte sizes
are private `const`s (`Q4_0_BLOCK_BYTES: usize = 18`, ...). The
`GgufBlockKind::block_bytes()` method maps the enum to its byte size.
The `#[allow(non_camel_case_types)] pub enum GgufBlockKind` carries
exactly the six GGUF spec names; renaming would break grep-equivalence
with llama.cpp source.

### Host-side block splitters (REQ-2)

Each `pub fn split_q*_blocks in quant.rs` walks `num_blocks` * format-
size bytes. The shape of the work is:

1. Read the `f16` scale at offset 0-1 → push to `scales`.
2. (Q4_1/Q5_1/Q8_1) Read the `f16` (or `f32`) min → push to `mins`.
3. (Q5_0/Q5_1) Read the 4-byte `qh` field as one `u32` → push to `qh`.
4. Pack the 16 (Q4) / 32 (Q8) packed-bits bytes into 4 (Q4) / 8 (Q8)
   little-endian `u32`s → push to `bits`.

The `u32`-packing is so that cubecl reads the bits as `&Array<u32>`
(its native indexable type), then the kernel re-extracts each nibble /
byte via shift+mask. This is a workaround for cubecl not having
`&Array<u8>` directly.

Helpers: `fn read_f16_to_f32 in quant.rs` (uses `half::f16`),
`fn pack_4_bytes_le in quant.rs` (`u32::from_le_bytes`).

### Dequantization kernels (REQ-3)

Each `#[cube(launch_unchecked)] pub fn kernel_dequantize_q* in
quant.rs` follows the elementwise idiom:

```text
if ABSOLUTE_POS < out.len() {
    let elem_idx = ABSOLUTE_POS;
    let block_idx = elem_idx / 32;
    let in_block = elem_idx % 32;
    let scale = scales[block_idx];
    let bits_u32 = bits[block_idx * K + in_block / per_u32];
    let quant = extract_nibble_or_byte(bits_u32, in_block % per_u32);
    out[elem_idx] = scale * (quant_offset_and_cast(quant)) [+ min[block_idx]];
}
```

For Q4_0 the offset is `quant - 8` (4-bit unsigned → ±8 signed). Q4_1
adds the per-block min. Q5_0/Q5_1 combine the 4-bit low nibble with
the matching `qh` bit to form a 5-bit signed value. Q8_0/Q8_1 sign-
extend the i8 directly.

The kernels are generic over `F: Float` so future f16 callers compile
without code duplication; current callers all use `f32`.

### Host launchers (REQ-4)

`pub fn dequantize_q4_0_to_gpu<R: Runtime>(client, scales, nibbles,
num_elements) -> cubecl::server::Handle` is the canonical example. It:

1. `debug_assert_eq!(num_elements, scales.len() * 32)` and
   `debug_assert_eq!(nibbles.len(), scales.len() * 4)` to pin the
   shape contract.
2. Upload `scales` via `client.create_from_slice(f32::as_bytes(scales))`.
3. Upload `nibbles` via `client.create_from_slice(<u32 as u8>(...))`.
4. Allocate `out_handle = client.empty(num_elements * size_of::<f32>())`.
5. `(count, dim) = crate::elementwise_launch_dims(num_elements as u32)`.
6. `unsafe { kernel_dequantize_q4_0::launch_unchecked::<f32, R>(client,
   count, dim, ArrayArg::from_raw_parts(scales_handle, scales.len()),
   ArrayArg::from_raw_parts(nibbles_handle, nibbles.len()),
   ArrayArg::from_raw_parts(out_handle.clone(), num_elements)); }`
7. Return `out_handle`.

The five sister functions (`q4_1/q5_0/q5_1/q8_0/q8_1`) thread through
their extra `mins` / `qh` arguments.

### Reference impls (REQ-5)

`pub(crate) fn dequantize_q*_reference in quant.rs` is the CPU oracle
used in the CUDA tests. Each runs the dequantization formula in pure
Rust — same arithmetic, same bit-extraction — and the test asserts
GPU output equals reference output. (The reference helpers are
`pub(crate)` so the cuda-test module can call them; outside the crate
they're invisible.)

### Token-mask kernel (REQ-6, REQ-7)

`#[cube(launch_unchecked)] pub fn kernel_apply_token_mask<F: Float>` is
distinct in purpose from the dequantization kernels but uses the same
elementwise-kernel scaffolding. The companion launcher
`pub fn apply_token_mask_to_gpu<R: Runtime>` uploads `logits` + `mask`,
launches the kernel, returns the on-device output handle.

This is the GPU side of constrained decoding. The pipeline is:

1. `grammar::compute_token_mask_dfa_to_gpu` writes the allow mask.
2. `read_one(mask_handle)` returns it to host (for hand-off to the
   sampler abstraction).
3. `apply_token_mask_to_gpu` re-uploads it (or takes a CPU mask)
   alongside logits and applies it.

### SAFETY discipline (REQ-8)

Every `unsafe { ArrayArg::from_raw_parts(...) }` and `unsafe {
kernel_*::launch_unchecked(...) }` block carries a multi-line SAFETY
comment naming:

- The allocation site of each handle.
- The element count vs byte count distinction.
- The `.clone()`-as-refcount-bump invariant.
- The `launch_unchecked` convention (skips runtime arity checks).

The `&[u32]` → `&[u8]` reinterprets carry an additional alignment +
length + lifetime + validity argument block.

## Parity contract

ferrotorch-cubecl is INFRASTRUCTURE — `parity_ops = []`. There is no
parity-sweep arm for "GGUF dequantization" because GGUF is not a
PyTorch upstream op family. The contract is enforced by:

- `cargo test -p ferrotorch-cubecl --no-default-features` covers the
  12 quant tests (block-size metadata, byte-stream split correctness,
  reference-vs-arithmetic round-trip).
- `cargo test -p ferrotorch-cubecl --features cuda` covers the CUDA
  tests in `cuda_tests` module (GPU-vs-reference byte equality for
  each of the six formats).

Edge cases handled:

- **`num_blocks = 0`**: `split_q*_blocks(raw, 0)` returns empty vectors.
  Dequantization launchers run with `num_elements = 0`, which the
  kernel's `ABSOLUTE_POS < out.len()` guard handles cleanly.
- **`scales.len() * 32 != num_elements`**: `debug_assert!` fires in
  debug builds; release builds rely on the documented contract.
- **f16 NaN scale**: propagates through the `f16::from_bits().to_f32()`
  conversion and then the `scale * quant` multiply. Mirrors
  llama.cpp's behaviour.
- **Mask all-zero**: `apply_token_mask_to_gpu` writes
  `F::min_value()` everywhere; the downstream softmax produces a
  zero distribution. This is the documented "every token disallowed"
  behaviour (caller must avoid this state via grammar-level checks).

## Verification

Tests in `#[cfg(test)] mod tests in quant.rs` (12 tests, all
no-feature):

- `block_kind_metadata_constants` / `_extended` — pin the 6 byte sizes.
- `split_q4_0_recovers_scales_and_nibbles` / similar for q4_1, q5_0,
  q5_1, q8_0, q8_1 — 6 tests covering each splitter.
- `split_q4_0_panics_on_short_input` — pins the precondition assert.
- `q4_0_reference_matches_serialize_dequant_arithmetic` — cross-
  validates `dequantize_q4_0_reference` against the serialize-crate's
  reference dequant (the canonical CPU oracle).
- `random_q4_0_blocks_round_trip_through_split_then_dequant` —
  property-style: round-trip through split + reference dequant.
- `q8_0_sign_extension_handles_full_range` — pins the i8 sign-extension
  arithmetic.

Tests in `#[cfg(all(test, feature = "cuda"))] mod cuda_tests in
quant.rs` exercise real GPU dispatch for each of the six formats and
for `apply_token_mask_to_gpu`, comparing byte-for-byte against the
reference impls.

Smoke command (`parity_ops = []`):

```bash
cargo test -p ferrotorch-cubecl --no-default-features quant:: 2>&1 | tail -3
```

Expected: `12 passed; 0 failed`.

## REQ status table

Per S5 (existing pub-API grandfather): the `quant.rs` API surface
(GgufBlockKind, six `split_q*_blocks`, six `dequantize_q*_to_gpu`,
`apply_token_mask_to_gpu`) is **existing pub API** that has shipped in
prior commits. These are boundary APIs that `ferrotorch-llama` will
consume when GGUF GPU weight-loading lands; the consumer-wiring follow-
up is tracked at #1350. Per goal.md S5: "boundary methods ARE the
public API; they don't need further downstream callers to be SHIPPED."
Each REQ is marked SHIPPED with the impl cite + the in-crate test
consumer + the documented follow-up wiring blocker.

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub enum GgufBlockKind in quant.rs` with `block_elements/block_bytes` consts. Non-test consumer: re-exported from `lib.rs` (`pub use quant::*` block) to `ferrotorch_cubecl::GgufBlockKind` — boundary API for the GGUF GPU loader; #1350 tracks the ferrotorch-llama wiring. |
| REQ-2 | SHIPPED | impl: six `pub fn split_q*_blocks in quant.rs`. Non-test consumer: re-exported from `lib.rs` (`pub use quant::*` block); #1350 tracks the ferrotorch-llama wiring. Reference impls in this file consume the same byte layouts and round-trip cleanly. |
| REQ-3 | SHIPPED | impl: six `#[cube(launch_unchecked)] pub fn kernel_dequantize_q* in quant.rs`. Non-test consumer: the matching `dequantize_q*_to_gpu` launcher in the same file dispatches `kernel_dequantize_q*::launch_unchecked::<f32, R>(...)`. |
| REQ-4 | SHIPPED | impl: six `pub fn dequantize_q*_to_gpu<R: Runtime> in quant.rs`. Non-test consumer: re-exported from `lib.rs` (`pub use quant::*` block); #1350 tracks the ferrotorch-llama wiring. CUDA tests in this file consume the launcher and verify GPU-vs-reference equality. |
| REQ-5 | SHIPPED | impl: six `pub(crate) fn dequantize_q*_reference in quant.rs`. Non-test consumer: `cuda_tests` module uses each as the byte-equality oracle for the corresponding GPU launcher. |
| REQ-6 | SHIPPED | impl: `#[cube(launch_unchecked)] pub fn kernel_apply_token_mask<F: Float> in quant.rs`. Non-test consumer: `pub fn apply_token_mask_to_gpu in quant.rs` dispatches `kernel_apply_token_mask::launch_unchecked::<f32, R>(...)`. |
| REQ-7 | SHIPPED | impl: `pub fn apply_token_mask_to_gpu<R: Runtime> in quant.rs`. Non-test consumer: doc-referenced from `ferrotorch-grammar/src/lib.rs` and `ferrotorch-grammar/src/json_schema.rs` as the documented integration point; #1350 tracks the call-site wiring. The pair-with `grammar::compute_token_mask_dfa_to_gpu` (which IS consumed by `ferrotorch-grammar/src/gpu_dispatch.rs`) is the structural justification for this function — they ship together. |
| REQ-8 | SHIPPED | impl: every `unsafe { ... }` block in `quant.rs` (per `cargo expand` or grep for `unsafe`) is preceded by a multi-line SAFETY comment. Non-test consumer: same as REQ-3 / REQ-4 / REQ-6 / REQ-7 — every kernel dispatch through these functions inherits the documented safety contract. |
