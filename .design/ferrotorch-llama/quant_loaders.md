# ferrotorch-llama — `quant_loaders` module

<!--
tier: 3-component
status: draft
baseline-pytorch: 6710f8ebc (working tree at /home/doll/pytorch)
upstream-paths:
  - GPTQ paper (Frantar et al. 2023, "GPTQ: Accurate Post-Training
    Quantization for Generative Pre-trained Transformers")
  - AutoGPTQ library (auto_gptq v0.7+ packing format: qweight,
    qzeros, scales, optional g_idx)
  - AutoAWQ library (MIT-Han-Lab autoawq packing format with the
    per-int32 [0,4,1,5,2,6,3,7] channel-order shuffle)
  - Mobius Labs HQQ (Half-Quadratic Quantization). Per-row model:
    1/2/3/4/8-bit. On-disk HQQLinear Q4 format (#1172): per-group
    scale/zero (axis=1, group_size 64/128) + `pack_4bit_u8`.
    Reference pinned: github.com/mobiusml/hqq v0.2.1
    `hqq/core/quantize.py` (`Quantizer`) + `hqq/core/bitpack.py`
    (`BitPack`).
-->

## Summary

`ferrotorch-llama/src/quant_loaders.rs` ships weight unpackers for
the three HF-quantized formats most commonly shipped on the Hub:
GPTQ (`q4`), AWQ (`q4`), and HQQ (`{1, 2, 3, 4, 8}`-bit). Each
unpacker takes the packed tensors a real checkpoint ships
(`qweight`, `qzeros`, `scales`, optional `g_idx`) and produces a
row-major dequantized `f32` weight matrix the standard
state-dict path can load.

## Requirements

- REQ-1: `pub struct GptqQ4` carries the four packed buffers and
  the three dimensional fields (`in_features`, `out_features`,
  `group_size`). The struct is `#[non_exhaustive]` to permit
  forward-compatible field additions.
- REQ-2: `GptqQ4::new` is a public constructor so external callers
  (integration tests, downstream apps) can build a `GptqQ4`
  without being blocked by `#[non_exhaustive]`.
- REQ-3: `pub fn dequantize_gptq_q4` produces a row-major
  `[out_features, in_features]` `f32` matrix matching
  `torch.nn.Linear.weight` shape. Validates:
  - `out_features % 8 == 0` and `in_features % 8 == 0` (packing
    constraint).
  - `in_features % group_size == 0`.
  - Buffer-length sanity for `qweight`, `qzeros`, `scales`, `g_idx`.
- REQ-4: GPTQ packing: 8 int4 weights per i32 along K axis;
  `qzeros` packs 8 int4 zeros per i32 along N; per-(group, n) f32
  scales. Zero reconstruction: `(packed_z + 1)` (GPTQ stores
  `zero - 1`). Dequant formula: `(q - zero) * scale`.
- REQ-5: Optional `g_idx` (act-order permutation table for
  GPTQ `act_order=True`). Length must equal `in_features`.
- REQ-6: `pub struct AwqQ4` (`#[non_exhaustive]`) + `AwqQ4::new`
  constructor with the AWQ-specific channel-order shuffle
  `[0, 4, 1, 5, 2, 6, 3, 7]`.
- REQ-7: `pub fn dequantize_awq_q4` produces a row-major
  `[out_features, in_features]` `f32` matrix. AWQ's qweight shape
  is `[in_features, out_features / 8]` (different from GPTQ's
  `[in_features / 8, out_features]`); the AWQ unpacker walks
  `(k, n_block)` and dequantizes 8 channels per i32 using the
  shuffle order.
- REQ-8: `pub struct HqqWeights` (`#[non_exhaustive]`) +
  `HqqWeights::new` constructor with bitwidth ∈ `{1, 2, 3, 4, 8}`,
  raw byte buffer, per-row f32 scale and zero vectors.
- REQ-9: `pub fn dequantize_hqq` produces row-major
  `[out_features, in_features]` f32. Dispatches by `bits` to the
  per-bitwidth unpacker (`unpack_hqq_8bit`, `_4bit`, `_2bit`,
  `_1bit`, `_3bit`), then applies the per-row `(q - zero) * scale`
  formula. (This is the per-row special case; the on-disk
  `HQQLinear` Q4 format is REQ-10..12.)
- REQ-10: `pub struct HqqQ4Axis1` (`#[non_exhaustive]`) +
  `HqqQ4Axis1::new` model the real on-disk `HQQLinear` Q4 format:
  the `pack_4bit_u8` byte buffer (row-major `[num_groups/2,
  group_size]`), per-group `scale` / `zero` vectors of length
  `num_groups`, `group_size`, and the original `[out_features,
  in_features]` shape. Per the HQQ reference (mobiusml/hqq v0.2.1
  `hqq/core/quantize.py` `Quantizer.quantize`, axis=1): the weight
  is reshaped to `[-1, group_size]` so scale/zero are per-group,
  not per-row.
- REQ-11: `pub fn dequantize_hqq_q4_axis1` produces row-major
  `[out_features, in_features]` f32. Unpacks via the HQQ
  `unpack_4bit_u8` split-half rule (`hqq/core/bitpack.py`: high
  nibble = grouped row `pr`, low nibble = grouped row `pr + step`
  where `step = num_groups/2`), then applies the per-group
  `(q - zero[g]) * scale[g]` and the flatten/reshape mapping
  `g = f / group_size`. Validates `numel % group_size == 0`,
  even `num_groups`, scale/zero lengths, and buffer length.
- REQ-12: `pub fn hqq_state_dict_to_dense` +
  `pub fn hqq_q4_axis1_to_dense` are the production consumer path:
  they walk a raw HQQ-format `StateDict<T>` (`{prefix}.W_q`,
  `{prefix}.scale`, `{prefix}.zero`, `{prefix}.nbits`,
  `{prefix}.group_size`, `{prefix}.shape`), dequantize every
  quantized linear to a dense `{prefix}.weight`, pass non-quantized
  tensors through, and feed the result into
  `LlamaForCausalLM::load_hqq_state_dict`. Only non-nested Q4
  (`nbits=4`, `axis=1`) is supported; other configs return an error
  rather than producing wrong weights (nested-quant is a follow-up).

## Acceptance Criteria

- [x] AC-1: GPTQ q4 one-group identity case: `q = k, zero = 2,
  scale = 1.0` produces `w[n, k] = k - 2`.
- [x] AC-2: GPTQ q4 rejects `out_features` and `in_features` not
  divisible by 8.
- [x] AC-3: GPTQ q4 rejects group_size not dividing in_features.
- [x] AC-4: GPTQ q4 rejects mismatched `g_idx` length.
- [x] AC-5: GPTQ q4 with two groups uses the per-group scale
  correctly.
- [x] AC-6: AWQ q4 uniform inputs produce the expected `(q - z) * scale`.
- [x] AC-7: AWQ q4 channel-shuffle order is the documented
  `[0, 4, 1, 5, 2, 6, 3, 7]`.
- [x] AC-8: AWQ q4 rejects `out_features` not divisible by 8.
- [x] AC-9: HQQ 1/2/3/4/8-bit unpackers produce the expected
  round-trip on hand-built byte patterns.
- [x] AC-10: HQQ per-row scale and zero applied correctly across
  rows.
- [x] AC-11: HQQ rejects invalid bitwidths (e.g. 5) and short
  buffers.

## Architecture

`pub struct GptqQ4` in `quant_loaders.rs` is the GPTQ q4 tile.
Layout matches `auto_gptq` v0.7+: `qweight: [K/8, N]`,
`qzeros: [K/G, N/8]`, `scales: [K/G, N]`, optional
`g_idx: [K]` for act-order.

`pub fn dequantize_gptq_q4` in `quant_loaders.rs` walks
`(k, n)` and for each `(k, n)`:

1. Compute the group index: `g_idx[k]` if act-order, else
   `k / group_size`.
2. Locate the i32 row containing `k`: `qrow = k / 8`,
   `nibble_idx = k % 8`.
3. Extract the 4-bit weight: `(qweight[qrow * N + n] >> (4 *
   nibble_idx)) & 0xF`.
4. Extract the 4-bit zero for `(group, n)`: at `qzeros[group * (N
   / 8) + n / 8]`, shift by `4 * (n % 8)`, mask `0xF`. Add 1 to
   reconstruct the true zero (GPTQ stores `zero - 1`).
5. Look up the per-(group, n) scale.
6. Compute `(q - zero) * scale` and write to
   `out[n * in_features + k]`.

`pub struct AwqQ4` in `quant_loaders.rs` is the AWQ q4 tile. The
core difference from GPTQ is the per-int32 channel-order shuffle:
within each i32 packed along the N axis, the 8 nibbles correspond
to N-channels `[0, 4, 1, 5, 2, 6, 3, 7]` rather than `[0..8]` in
order. This is `AWQ_PACK_ORDER` in the module.

`pub fn dequantize_awq_q4` in `quant_loaders.rs` walks
`(k, n_block)` and for each of the 8 nibbles in the i32, dequantizes
into the corresponding n channel via the shuffle table.

`pub struct HqqWeights` in `quant_loaders.rs` is the HQQ packed
tile. `bits ∈ {1, 2, 3, 4, 8}`. `scale` and `zero` are per-row
f32 vectors of length `out_features`. The packed integer weights
live in `w_q: Vec<u8>` with bitwidth-specific packing.

`pub fn dequantize_hqq` in `quant_loaders.rs` validates bitwidth
and per-row vector lengths, dispatches by `bits` to the
appropriate unpacker (8 → byte-per-weight, 4 → 2 nibbles per
byte low-first, 2 → 4-per-byte LSB-first, 1 → 8-per-byte
LSB-first, 3 → tight 8-per-3-bytes), then applies
`(q - zero[row]) * scale[row]` per element.

### Non-test production consumers

- `pub use quant_loaders::{AwqQ4, GptqQ4, dequantize_awq_q4,
  dequantize_gptq_q4}` plus the HQQ axis-1 surface (`HqqQ4Axis1`,
  `dequantize_hqq_q4_axis1`, `hqq_q4_axis1_to_dense`,
  `hqq_state_dict_to_dense`) re-exported from
  `ferrotorch-llama/src/lib.rs` (the `pub use quant_loaders::{…}`
  block).
- **HQQ Q4 production consumer (#1172)**:
  `LlamaForCausalLM::load_hqq_state_dict` in
  `ferrotorch-llama/src/model.rs` calls
  `quant_loaders::hqq_state_dict_to_dense`, which calls
  `hqq_q4_axis1_to_dense`, which calls `dequantize_hqq_q4_axis1`.
  This is a real model-loading path: a HQQ-quantized Llama
  checkpoint loaded via `load_hqq_state_dict` produces a fully
  dense `LlamaForCausalLM` ready for `forward_from_ids`.
- The `ferrotorch::llama` umbrella re-export exposes the quant
  primitives to any downstream user of the meta-crate.

The GPTQ/AWQ dequantizers (REQ-1..7) remain boundary methods per
goal.md S5; they are grandfathered existing pub API. The HQQ Q4
path (REQ-10..12) ships WITH its production consumer
(`load_hqq_state_dict`) in the same commit, satisfying R-DEFER-1.

## Parity contract

`parity_ops = []`. The dequantizers' correctness criterion is
"for each packed (qweight, qzeros, scales) tile, produce the same
f32 weight matrix `auto_gptq` / `autoawq` / `mobius-hqq` would on
the same input". The tests exercise this with hand-built packed
patterns and known answers.

Behavioral guarantees:

- **GPTQ zero reconstruction**: `(packed_z + 1)`. Matches the
  `auto_gptq` source: `zero = (qzeros >> shift) & 0xF; zero =
  zero + 1`.
- **GPTQ act-order**: when `g_idx` is provided, the group lookup
  uses `g_idx[k]` rather than `k / group_size`. This is the only
  case where `act_order=True` GPTQ checkpoints differ from
  `act_order=False`.
- **AWQ channel shuffle**: `[0, 4, 1, 5, 2, 6, 3, 7]`. Matches
  `autoawq/awq_inference_engine`'s `pack_intweight` shuffle.
- **HQQ per-row formula** (REQ-9, the special case):
  `(q - zero[row]) * scale[row]`.
- **HQQ axis=1 grouped formula** (REQ-10..12, the on-disk
  `HQQLinear` format): `(unpack_4bit_u8(W_q)[g] - zero[g]) *
  scale[g]` with per-group scale/zero (`num_groups = numel /
  group_size`) and the `pack_4bit_u8` split-half packing. Matches
  mobiusml/hqq v0.2.1 `Quantizer.dequantize` + `BitPack` byte-for-
  byte (verified against the reference oracle in tests).
- **Row-major output**: `[out_features, in_features]` matching
  `torch.nn.Linear.weight` shape (which is the
  `[out_features, in_features]` row-major convention PyTorch uses).

## Verification

Tests in `mod tests in quant_loaders.rs`:

GPTQ:
- `gptq_q4_dequantize_one_group_identity`
- `gptq_q4_rejects_non_multiple_of_8_dims`
- `gptq_q4_rejects_misaligned_group`
- `gptq_q4_rejects_g_idx_length_mismatch`
- `gptq_q4_with_two_groups_uses_per_group_scale`

AWQ:
- `awq_q4_dequantize_uniform_inputs`
- `awq_q4_shuffle_order_is_distinct_from_gptq`
- `awq_q4_rejects_non_multiple_of_8_out_features`
- `awq_q4_rejects_misaligned_group`

HQQ (per-row special case):
- `hqq_8bit_roundtrip`
- `hqq_4bit_roundtrip`
- `hqq_2bit_roundtrip`
- `hqq_1bit_roundtrip`
- `hqq_3bit_roundtrip_known_pattern`
- `hqq_per_row_scale_zero_applied_correctly`
- `hqq_rejects_invalid_bits`
- `hqq_rejects_short_buffer`
- `hqq_rejects_scale_length_mismatch`

HQQ axis=1 grouped Q4 (#1172, oracles from mobiusml/hqq v0.2.1):
- `hqq_q4_axis1_oracle_group_size_8` (one group per row)
- `hqq_q4_axis1_oracle_two_groups_per_row` (gs=4 < in_features,
  the case the per-row model cannot represent)
- `hqq_q4_axis1_rejects_odd_num_groups`
- `hqq_q4_axis1_rejects_scale_length_mismatch`
- `hqq_q4_axis1_rejects_short_buffer`
- `hqq_state_dict_to_dense_produces_oracle_weight` (production
  state-dict path + passthrough)

Plus integration tests in
`ferrotorch-llama/tests/integration_quant_loaders.rs`:
- `hqq_axis1_state_dict_to_dense_matches_reference_oracle`
  (network-free, exercises the production consumer)
- `hqq_q4_axis1_dequant_matches_reference_oracle_gs8`
- `hqq_smoke` (`#[ignore]`, network — now routed through the
  production `load_hqq_state_dict`).

No parity-sweep ops. Smoke command:

```bash
cargo test -p ferrotorch-llama --lib quant_loaders:: 2>&1 | tail -3
```

Expected: all tests pass.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 | SHIPPED | impl: `pub struct GptqQ4` (`#[non_exhaustive]`) in `quant_loaders.rs`; non-test consumer: re-exported at `ferrotorch-llama/src/lib.rs:175`, reachable from the meta-crate. |
| REQ-2 | SHIPPED | impl: `pub fn GptqQ4::new` in `quant_loaders.rs`; non-test consumer: same re-export surface; `#[non_exhaustive]` requires the constructor for external builders. |
| REQ-3 | SHIPPED | impl: `pub fn dequantize_gptq_q4` (with full input validation block) in `quant_loaders.rs`; non-test consumer: re-exported at `ferrotorch-llama/src/lib.rs:175`. |
| REQ-4 | SHIPPED | impl: the `(q - zero) * scale` per-element math in `dequantize_gptq_q4` with `let zero = z + 1` in `quant_loaders.rs`; non-test consumer: same re-export surface as REQ-3. |
| REQ-5 | SHIPPED | impl: the `g_idx` length check + the `match g_idx` branch on group lookup in `dequantize_gptq_q4` in `quant_loaders.rs`; non-test consumer: same re-export surface as REQ-3. |
| REQ-6 | SHIPPED | impl: `pub struct AwqQ4` + `AwqQ4::new` + `const AWQ_PACK_ORDER` in `quant_loaders.rs`; non-test consumer: re-exported at `ferrotorch-llama/src/lib.rs:175`. |
| REQ-7 | SHIPPED | impl: `pub fn dequantize_awq_q4` (with `for (shuffle_idx, &lane) in AWQ_PACK_ORDER.iter().enumerate()` shuffle loop) in `quant_loaders.rs`; non-test consumer: re-exported at `ferrotorch-llama/src/lib.rs:175`. |
| REQ-8 | SHIPPED | impl: `pub struct HqqWeights` + `HqqWeights::new` in `quant_loaders.rs`; non-test consumer: reachable as `ferrotorch_llama::quant_loaders::HqqWeights` via the module-path re-export. (Per-row special case; the on-disk Q4 format is REQ-10..12.) |
| REQ-9 | SHIPPED | impl: `pub fn dequantize_hqq` (with `match bits` dispatch into the five `unpack_hqq_*` helpers and the per-row `(q - z) * s` body) in `quant_loaders.rs`; non-test consumer: same module-path re-export as REQ-8. |
| REQ-10 | SHIPPED | impl: `pub struct HqqQ4Axis1` + `pub fn HqqQ4Axis1::new` in `quant_loaders.rs`; non-test consumer: constructed inside `pub fn hqq_q4_axis1_to_dense` in `quant_loaders.rs`, on the `LlamaForCausalLM::load_hqq_state_dict` path in `model.rs`. |
| REQ-11 | SHIPPED | impl: `pub fn dequantize_hqq_q4_axis1` (split-half `unpack_4bit_u8` + per-group `(q - zero[g]) * scale[g]`) in `quant_loaders.rs` per mobiusml/hqq v0.2.1 `bitpack.py:24-38` / `quantize.py:179-194`; non-test consumer: called by `hqq_q4_axis1_to_dense` in `quant_loaders.rs`, reached from `LlamaForCausalLM::load_hqq_state_dict` in `model.rs`. |
| REQ-12 | SHIPPED | impl: `pub fn hqq_state_dict_to_dense` + `pub fn hqq_q4_axis1_to_dense` in `quant_loaders.rs`; non-test production consumer: `pub fn LlamaForCausalLM::load_hqq_state_dict` in `model.rs` calls `hqq_state_dict_to_dense` then `load_hf_state_dict`. |
