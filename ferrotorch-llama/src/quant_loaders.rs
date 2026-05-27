//! Weight unpackers for HF-quantized LLM checkpoints. (#593)
//!
//! Supports the two formats most commonly shipped on the Hub:
//!
//! - **GPTQ** (`q4` / `q8`, group-wise asymmetric int quantization with
//!   per-group scales and zero-points, packed into i32 along the
//!   in_features axis). Reference: Frantar et al. 2023.
//! - **AWQ** (`q4`, group-wise scales but with a fixed *channel order*
//!   shuffle in the packed layout, originally introduced by MIT-Han-Lab).
//!
//! Both unpackers produce dequantized `f32` weight matrices that callers
//! can then load via the standard state-dict path. The unpackers do not
//! own a tokenizer or model — they operate purely on the four packed
//! tensors (`qweight`, `qzeros`, `scales`, optional `g_idx`) shipped
//! by the HF Transformers `auto_gptq` / `autoawq` libraries.
//!
//! HQQ comes in two shapes here: the per-row [`HqqWeights`] /
//! [`dequantize_hqq`] model, and the on-disk `HQQLinear` axis=1 grouped Q4
//! format ([`HqqQ4Axis1`] / [`dequantize_hqq_q4_axis1`]) with per-group
//! scale/zero and the `pack_4bit_u8` split-half packing. The latter is the
//! format real `mobius-hqq` Q4 checkpoints ship; it is wired into model
//! loading via [`hqq_state_dict_to_dense`] and
//! [`crate::model::LlamaForCausalLM::load_hqq_state_dict`] (#1172).

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::module::StateDict;

/// A 4-bit GPTQ-packed weight tile. Layout matches `auto_gptq` v0.7+.
///
/// Shapes (with `out_features = N` and `in_features = K`, group_size = G):
/// - `qweight`: `[K / 8, N]` packed i32 (8 int4 weights per i32, along K).
/// - `qzeros`:  `[K / G, N / 8]` packed i32 (8 int4 zeros per i32, along N).
/// - `scales`:  `[K / G, N]` `f32` per-group, per-out-channel scales.
///
/// `g_idx` is the optional permutation table for `act_order=True` GPTQ;
/// pass `None` when the checkpoint was saved with `act_order=False`
/// (the common case).
#[derive(Debug)]
#[non_exhaustive]
pub struct GptqQ4 {
    /// Packed int4 weights along the K (in_features) axis. Shape
    /// `[K / 8, N]` int32, 8 nibbles per i32.
    pub qweight: Vec<i32>,
    /// Packed int4 zero-points along the N (out_features) axis. Shape
    /// `[K / G, N / 8]` int32, 8 nibbles per i32.
    pub qzeros: Vec<i32>,
    /// Per-group, per-out-channel f32 scales. Shape `[K / G, N]`.
    pub scales: Vec<f32>,
    /// Optional `act_order=True` permutation table. Length `K` when
    /// present; `None` for the common `act_order=False` case.
    pub g_idx: Option<Vec<i32>>,
    /// `K` — input feature count of the underlying linear weight.
    pub in_features: usize,
    /// `N` — output feature count of the underlying linear weight.
    pub out_features: usize,
    /// `G` — quantization group size along the K axis.
    pub group_size: usize,
}

impl GptqQ4 {
    /// Construct a [`GptqQ4`] tile from the individual packed buffers.
    ///
    /// This constructor exists so that external callers (e.g. integration
    /// tests) can build a [`GptqQ4`] without being blocked by the
    /// `#[non_exhaustive]` attribute.
    pub fn new(
        qweight: Vec<i32>,
        qzeros: Vec<i32>,
        scales: Vec<f32>,
        g_idx: Option<Vec<i32>>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> Self {
        Self {
            qweight,
            qzeros,
            scales,
            g_idx,
            in_features,
            out_features,
            group_size,
        }
    }
}

/// Dequantize a 4-bit GPTQ weight matrix to row-major `f32`.
///
/// Returns `[out_features, in_features]` row-major (matches torch's
/// `Linear.weight` shape, ready for the `linear` op).
///
/// # Errors
/// - In/out feature counts not divisible by 8 (packing constraint).
/// - in_features not divisible by group_size.
/// - Any of the packed buffers has the wrong length for the declared shape.
pub fn dequantize_gptq_q4(packed: &GptqQ4) -> FerrotorchResult<Vec<f32>> {
    let GptqQ4 {
        qweight,
        qzeros,
        scales,
        g_idx,
        in_features,
        out_features,
        group_size,
    } = packed;
    let in_features = *in_features;
    let out_features = *out_features;
    let group_size = *group_size;

    if out_features % 8 != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("GPTQ q4: out_features ({out_features}) must be a multiple of 8"),
        });
    }
    if in_features % 8 != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("GPTQ q4: in_features ({in_features}) must be a multiple of 8"),
        });
    }
    if in_features % group_size != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "GPTQ q4: in_features ({in_features}) must be a multiple of group_size ({group_size})"
            ),
        });
    }
    let num_groups = in_features / group_size;
    let qweight_rows = in_features / 8;

    if qweight.len() != qweight_rows * out_features {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "GPTQ q4: qweight len {} != expected {}",
                qweight.len(),
                qweight_rows * out_features
            ),
        });
    }
    if qzeros.len() != num_groups * (out_features / 8) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "GPTQ q4: qzeros len {} != expected {}",
                qzeros.len(),
                num_groups * (out_features / 8)
            ),
        });
    }
    if scales.len() != num_groups * out_features {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "GPTQ q4: scales len {} != expected {}",
                scales.len(),
                num_groups * out_features
            ),
        });
    }
    if let Some(g) = g_idx {
        if g.len() != in_features {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "GPTQ q4: g_idx len {} != in_features {in_features}",
                    g.len()
                ),
            });
        }
    }

    // Output: row-major [out_features, in_features].
    let mut out = vec![0.0f32; out_features * in_features];

    for k in 0..in_features {
        // Group index for this k. With act_order, g_idx[k] gives it.
        let group = match g_idx {
            Some(g) => g[k] as usize,
            None => k / group_size,
        };
        // Locate the int32 row + nibble for k.
        let qrow = k / 8;
        let nibble_idx = k % 8;

        for n in 0..out_features {
            // Extract the 4-bit weight.
            let packed_w = qweight[qrow * out_features + n] as u32;
            let q = ((packed_w >> (4 * nibble_idx)) & 0xF) as i32;

            // Extract the 4-bit zero for (group, n).
            let qzeros_row = group;
            let zero_col = n / 8;
            let zero_nib = n % 8;
            let packed_z = qzeros[qzeros_row * (out_features / 8) + zero_col] as u32;
            let z = ((packed_z >> (4 * zero_nib)) & 0xF) as i32;
            // GPTQ stores zero - 1; reconstruct true zero.
            let zero = z + 1;

            let scale = scales[group * out_features + n];
            let dequant = (q - zero) as f32 * scale;
            out[n * in_features + k] = dequant;
        }
    }
    Ok(out)
}

/// AWQ 4-bit packed layout. The major difference from GPTQ is the
/// per-int32 channel-order shuffle: AWQ packs 8 int4 weights per i32
/// from out_channels in a specific order so that runtime dequantize
/// kernels can load consecutive weights in cache-friendly stripes.
#[derive(Debug)]
#[non_exhaustive]
pub struct AwqQ4 {
    /// Packed int4 weights, shape `[in_features, out_features / 8]`
    /// int32 with the AWQ shuffle order applied (see [`AWQ_PACK_ORDER`]).
    pub qweight: Vec<i32>,
    /// Packed int4 zero-points, shape `[num_groups, out_features / 8]` int32.
    pub qzeros: Vec<i32>,
    /// Per-group, per-out-channel f32 scales, shape `[num_groups, out_features]`.
    pub scales: Vec<f32>,
    /// `K` — input feature count of the underlying linear weight.
    pub in_features: usize,
    /// `N` — output feature count of the underlying linear weight.
    pub out_features: usize,
    /// `G` — quantization group size along the K axis.
    pub group_size: usize,
}

impl AwqQ4 {
    /// Construct an [`AwqQ4`] tile from the individual packed buffers.
    ///
    /// This constructor exists so that external callers (e.g. integration
    /// tests) can build an [`AwqQ4`] without being blocked by the
    /// `#[non_exhaustive]` attribute.
    pub fn new(
        qweight: Vec<i32>,
        qzeros: Vec<i32>,
        scales: Vec<f32>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> Self {
        Self {
            qweight,
            qzeros,
            scales,
            in_features,
            out_features,
            group_size,
        }
    }
}

/// AWQ's int32 → int4 channel-shuffle order (see autoawq/awq_inference_engine).
/// AWQ packs `[N0, N4, N1, N5, N2, N6, N3, N7]` instead of `[N0..N7]`.
const AWQ_PACK_ORDER: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

/// Dequantize a 4-bit AWQ weight matrix to row-major `f32` of shape
/// `[out_features, in_features]`.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] when
/// `out_features % 8 != 0`, when `in_features % group_size != 0`, or
/// when any of `qweight` / `qzeros` / `scales` has the wrong length
/// for the declared shape.
pub fn dequantize_awq_q4(packed: &AwqQ4) -> FerrotorchResult<Vec<f32>> {
    let AwqQ4 {
        qweight,
        qzeros,
        scales,
        in_features,
        out_features,
        group_size,
    } = packed;
    let in_features = *in_features;
    let out_features = *out_features;
    let group_size = *group_size;

    if out_features % 8 != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("AWQ q4: out_features ({out_features}) must be a multiple of 8"),
        });
    }
    if in_features % group_size != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "AWQ q4: in_features ({in_features}) must be a multiple of group_size ({group_size})"
            ),
        });
    }
    let num_groups = in_features / group_size;
    let n_packed = out_features / 8;

    // AWQ qweight shape: [in_features, out_features / 8] int32.
    if qweight.len() != in_features * n_packed {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "AWQ q4: qweight len {} != expected {}",
                qweight.len(),
                in_features * n_packed
            ),
        });
    }
    if qzeros.len() != num_groups * n_packed {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "AWQ q4: qzeros len {} != expected {}",
                qzeros.len(),
                num_groups * n_packed
            ),
        });
    }
    if scales.len() != num_groups * out_features {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "AWQ q4: scales len {} != expected {}",
                scales.len(),
                num_groups * out_features
            ),
        });
    }

    let mut out = vec![0.0f32; out_features * in_features];

    for k in 0..in_features {
        let group = k / group_size;
        for n_block in 0..n_packed {
            let packed_w = qweight[k * n_packed + n_block] as u32;
            let packed_z = qzeros[group * n_packed + n_block] as u32;
            for (shuffle_idx, &lane) in AWQ_PACK_ORDER.iter().enumerate() {
                let q = ((packed_w >> (4 * lane)) & 0xF) as i32;
                let z = ((packed_z >> (4 * lane)) & 0xF) as i32;
                let n = n_block * 8 + shuffle_idx;
                let scale = scales[group * out_features + n];
                let dequant = (q - z) as f32 * scale;
                out[n * in_features + k] = dequant;
            }
        }
    }
    Ok(out)
}

// ===========================================================================
// HQQ: Half-Quadratic Quantization (Mobius Labs) (#613)
// ===========================================================================
//
// HQQ packs:
//   - per-row f16 `scale` and `zero` tensors (half-precision; we accept
//     them as f32 buffers since callers cast on load)
//   - packed integer weights at bitwidth ∈ {1, 2, 3, 4, 8}
//     - 8-bit: stored as `u8`, one int per byte
//     - 4-bit: 2 nibbles per byte, low nibble first
//     - 2-bit: 4 ints per byte, lowest 2 bits first
//     - 1-bit: 8 ints per byte, LSB first
//     - 3-bit: tightly packed, see comment in unpack_3bit
//
// Dequantization formula per element: `(q - zero) * scale`. Layout is
// row-major `[out_features, in_features]`. `zero` and `scale` are per-row
// (one f32 per row of the output matrix).

/// A packed HQQ weight tile.
#[derive(Debug)]
#[non_exhaustive]
pub struct HqqWeights {
    /// Bit-width of the packed integers. Must be in {1, 2, 3, 4, 8}.
    pub bits: u8,
    /// Packed integer weights as raw bytes.
    pub w_q: Vec<u8>,
    /// Per-row scale (length = out_features). Stored as f32 here; HQQ
    /// originally ships these as f16 — callers cast on load.
    pub scale: Vec<f32>,
    /// Per-row zero point (length = out_features). Same dtype note.
    pub zero: Vec<f32>,
    /// Output features (rows of the dequantized matrix).
    pub out_features: usize,
    /// Input features (cols of the dequantized matrix).
    pub in_features: usize,
}

impl HqqWeights {
    /// Construct an [`HqqWeights`] tile from the individual packed buffers.
    ///
    /// This constructor exists so that external callers (e.g. integration
    /// tests) can build an [`HqqWeights`] without being blocked by the
    /// `#[non_exhaustive]` attribute.
    pub fn new(
        bits: u8,
        w_q: Vec<u8>,
        scale: Vec<f32>,
        zero: Vec<f32>,
        out_features: usize,
        in_features: usize,
    ) -> Self {
        Self {
            bits,
            w_q,
            scale,
            zero,
            out_features,
            in_features,
        }
    }
}

/// Dequantize an HQQ-packed weight matrix to row-major `f32`. (#613)
///
/// Output shape: `[out_features, in_features]`. Each element is computed
/// as `(q - zero[row]) * scale[row]` where `q` is the unpacked integer
/// weight at the bitwidth indicated by `packed.bits`.
///
/// # Errors
/// - `bits` not in {1, 2, 3, 4, 8}
/// - `scale` / `zero` length mismatch with `out_features`
/// - Packed buffer too short for `out_features × in_features` at the bitwidth
pub fn dequantize_hqq(packed: &HqqWeights) -> FerrotorchResult<Vec<f32>> {
    let HqqWeights {
        bits,
        w_q,
        scale,
        zero,
        out_features,
        in_features,
    } = packed;
    let bits = *bits;
    let out_features = *out_features;
    let in_features = *in_features;

    if !matches!(bits, 1 | 2 | 3 | 4 | 8) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("HQQ: bits must be 1/2/3/4/8, got {bits}"),
        });
    }
    if scale.len() != out_features {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "HQQ: scale len {} != out_features {out_features}",
                scale.len()
            ),
        });
    }
    if zero.len() != out_features {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "HQQ: zero len {} != out_features {out_features}",
                zero.len()
            ),
        });
    }

    let total = out_features * in_features;
    let unpacked = match bits {
        8 => unpack_hqq_8bit(w_q, total)?,
        4 => unpack_hqq_4bit(w_q, total)?,
        2 => unpack_hqq_2bit(w_q, total)?,
        1 => unpack_hqq_1bit(w_q, total)?,
        3 => unpack_hqq_3bit(w_q, total)?,
        _ => unreachable!("bits validated above"),
    };

    let mut out = Vec::with_capacity(total);
    for r in 0..out_features {
        let s = scale[r];
        let z = zero[r];
        let row_start = r * in_features;
        for c in 0..in_features {
            let q = unpacked[row_start + c] as f32;
            out.push((q - z) * s);
        }
    }
    Ok(out)
}

/// 8-bit unpacker: each byte is one weight.
fn unpack_hqq_8bit(packed: &[u8], total: usize) -> FerrotorchResult<Vec<u8>> {
    if packed.len() < total {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "HQQ 8-bit: packed buffer too short ({} bytes for {total} weights)",
                packed.len()
            ),
        });
    }
    Ok(packed[..total].to_vec())
}

/// 4-bit unpacker: 2 nibbles per byte, low nibble first.
fn unpack_hqq_4bit(packed: &[u8], total: usize) -> FerrotorchResult<Vec<u8>> {
    let needed = total.div_ceil(2);
    if packed.len() < needed {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "HQQ 4-bit: packed buffer too short ({} bytes for {total} weights, need {needed})",
                packed.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let byte = packed[i / 2];
        let nibble = if i % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        out.push(nibble);
    }
    Ok(out)
}

/// 2-bit unpacker: 4 ints per byte, lowest 2 bits first.
fn unpack_hqq_2bit(packed: &[u8], total: usize) -> FerrotorchResult<Vec<u8>> {
    let needed = total.div_ceil(4);
    if packed.len() < needed {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "HQQ 2-bit: packed buffer too short ({} bytes for {total} weights, need {needed})",
                packed.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let byte = packed[i / 4];
        let shift = (i % 4) * 2;
        out.push((byte >> shift) & 0x03);
    }
    Ok(out)
}

/// 1-bit unpacker: 8 ints per byte, LSB first.
fn unpack_hqq_1bit(packed: &[u8], total: usize) -> FerrotorchResult<Vec<u8>> {
    let needed = total.div_ceil(8);
    if packed.len() < needed {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "HQQ 1-bit: packed buffer too short ({} bytes for {total} weights, need {needed})",
                packed.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let byte = packed[i / 8];
        let shift = i % 8;
        out.push((byte >> shift) & 0x01);
    }
    Ok(out)
}

/// 3-bit unpacker: 8 ints per 3 bytes, tightly packed, LSB first across
/// bytes. The encoding stores 8 × 3-bit = 24 bits per 3-byte group; bit
/// `i*3` of the i-th weight starts at the bit position computed below.
fn unpack_hqq_3bit(packed: &[u8], total: usize) -> FerrotorchResult<Vec<u8>> {
    let group_count = total.div_ceil(8);
    let needed = group_count * 3;
    if packed.len() < needed {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "HQQ 3-bit: packed buffer too short ({} bytes for {total} weights, need {needed})",
                packed.len()
            ),
        });
    }
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let bit_pos = i * 3;
        let byte_idx = bit_pos / 8;
        let bit_offset = bit_pos % 8;
        let mut val: u32 = packed[byte_idx] as u32;
        if byte_idx + 1 < packed.len() {
            val |= (packed[byte_idx + 1] as u32) << 8;
        }
        if byte_idx + 2 < packed.len() && bit_offset > 5 {
            val |= (packed[byte_idx + 2] as u32) << 16;
        }
        out.push(((val >> bit_offset) & 0x07) as u8);
    }
    Ok(out)
}

// ===========================================================================
// HQQ axis=1 grouped Q4 — the on-disk HQQLinear format (#1172)
// ===========================================================================
//
// The `HqqWeights` / `dequantize_hqq` pair above models the *per-row*
// special case (one scale/zero per output row). Real HQQ checkpoints
// shipped by `HQQLinear` use **per-group** scale/zero along a quantization
// axis with `group_size ∈ {64, 128}`, and a packing that is NOT
// "2 consecutive nibbles per byte" — it splits the grouped tensor in half
// along its first (group) axis.
//
// Reference: mobiusml/hqq v0.2.1 `hqq/core/quantize.py` `Quantizer.quantize`
// / `Quantizer.dequantize` and `hqq/core/bitpack.py` `BitPack.pack_4bit_u8`
// / `unpack_4bit_u8`. github.com/mobiusml/hqq (pinned 0.2.1).
//
// quantize (axis=1): `W.reshape([-1, group_size])` → `num_groups =
// numel / group_size` rows; `scale`, `zero` have shape `[num_groups, 1]`.
// `W_q = round(W * (1/scale) + zero)`; meta stores `scale = 1/(that)` so
// dequant is a plain multiply.
//
// dequantize: `W_r = (unpack(W_q) - zero) * scale` then `.reshape(shape)`.
//
// pack_4bit_u8 (`bitpack.py:24`): `_step = len(W_q) // 2; (W_q[:_step] << 4)
// | W_q[_step:]`. So packed byte `(pr, c)` holds the high nibble of grouped
// row `pr` and the low nibble of grouped row `pr + step`.
//
// unpack_4bit_u8 (`bitpack.py:31`): `tmp[:step] = (W_q & 0xF0) >> 4;
// tmp[step:] = W_q & 0x0F`, producing `[2*step, group_size]`.

/// A 4-bit HQQ-quantized weight tile in the on-disk `HQQLinear` layout
/// with **per-group** scale/zero along `axis = 1`. (#1172)
///
/// This is the format real `mobius-hqq` Q4 checkpoints ship, distinct
/// from the per-row [`HqqWeights`] model above. The packed buffer is the
/// `pack_4bit_u8` output: a row-major `[num_groups / 2, group_size]`
/// `u8` matrix.
#[derive(Debug)]
#[non_exhaustive]
pub struct HqqQ4Axis1 {
    /// `pack_4bit_u8` output bytes, row-major `[num_groups / 2, group_size]`.
    /// The high nibble of byte `(pr, c)` is grouped element `(pr, c)`; the
    /// low nibble is grouped element `(pr + num_groups/2, c)`.
    pub w_q: Vec<u8>,
    /// Per-group dequant scale (meta `scale`, already the reciprocal HQQ
    /// stores for dequantization). Length `num_groups`.
    pub scale: Vec<f32>,
    /// Per-group zero point (meta `zero`). Length `num_groups`.
    pub zero: Vec<f32>,
    /// `group_size` along the quantization axis (HQQ default 64; 128 common).
    pub group_size: usize,
    /// Output features (rows of the reshaped dequantized matrix).
    pub out_features: usize,
    /// Input features (cols of the reshaped dequantized matrix).
    pub in_features: usize,
}

impl HqqQ4Axis1 {
    /// Construct an [`HqqQ4Axis1`] tile from the individual packed buffers.
    ///
    /// Exists so external callers can build the tile despite
    /// `#[non_exhaustive]`.
    pub fn new(
        w_q: Vec<u8>,
        scale: Vec<f32>,
        zero: Vec<f32>,
        group_size: usize,
        out_features: usize,
        in_features: usize,
    ) -> Self {
        Self {
            w_q,
            scale,
            zero,
            group_size,
            out_features,
            in_features,
        }
    }
}

/// Dequantize an HQQ axis=1 grouped Q4 tile to row-major `f32`. (#1172)
///
/// Output shape: `[out_features, in_features]` (matching
/// `torch.nn.Linear.weight`). Implements the HQQ reference flow:
/// `unpack_4bit_u8(W_q)` → `(q - zero[g]) * scale[g]` per group → reshape.
///
/// # Errors
/// - `group_size == 0` or `numel` not divisible by `group_size`.
/// - `num_groups` is odd (4-bit packing splits the group axis in half).
/// - `scale` / `zero` length != `num_groups`.
/// - `w_q` buffer too short for `[num_groups / 2, group_size]`.
pub fn dequantize_hqq_q4_axis1(packed: &HqqQ4Axis1) -> FerrotorchResult<Vec<f32>> {
    let HqqQ4Axis1 {
        w_q,
        scale,
        zero,
        group_size,
        out_features,
        in_features,
    } = packed;
    let group_size = *group_size;
    let out_features = *out_features;
    let in_features = *in_features;

    if group_size == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "HQQ q4 axis1: group_size must be > 0".into(),
        });
    }
    let numel = out_features * in_features;
    if numel % group_size != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "HQQ q4 axis1: numel ({numel}) not divisible by group_size ({group_size})"
            ),
        });
    }
    let num_groups = numel / group_size;
    if num_groups % 2 != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "HQQ q4 axis1: num_groups ({num_groups}) must be even (4-bit packing splits the group axis in half)"
            ),
        });
    }
    if scale.len() != num_groups {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "HQQ q4 axis1: scale len {} != num_groups {num_groups}",
                scale.len()
            ),
        });
    }
    if zero.len() != num_groups {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "HQQ q4 axis1: zero len {} != num_groups {num_groups}",
                zero.len()
            ),
        });
    }
    let step = num_groups / 2;
    if w_q.len() < step * group_size {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "HQQ q4 axis1: w_q buffer too short ({} bytes for [{step}, {group_size}], need {})",
                w_q.len(),
                step * group_size
            ),
        });
    }

    // Build the unpacked grouped tensor U[num_groups, group_size] then map
    // each flattened weight index f -> group g = f / group_size, offset
    // o = f % group_size, and write (U[g, o] - zero[g]) * scale[g].
    let mut out = vec![0.0f32; numel];
    for f in 0..numel {
        let g = f / group_size;
        let o = f % group_size;
        // unpack_4bit_u8: grouped row g, col o.
        //   g <  step -> high nibble of packed byte (g, o)
        //   g >= step -> low  nibble of packed byte (g - step, o)
        let q = if g < step {
            (w_q[g * group_size + o] >> 4) & 0x0F
        } else {
            w_q[(g - step) * group_size + o] & 0x0F
        };
        out[f] = (q as f32 - zero[g]) * scale[g];
    }
    Ok(out)
}

/// The HQQ-format suffixes for a single quantized linear inside a flat
/// HuggingFace/HQQ `safetensors` state dict. (#1172)
///
/// `HQQLinear.state_dict` (`hqq/core/quantize.py` `state_dict_keys`)
/// serializes each quantized weight under a per-module `{prefix}.` with
/// these leaf keys. We read `W_q`, `scale`, `zero`, `nbits`, `group_size`,
/// `axis`, and the original `shape`.
const HQQ_WQ_SUFFIX: &str = ".W_q";
const HQQ_SCALE_SUFFIX: &str = ".scale";
const HQQ_ZERO_SUFFIX: &str = ".zero";
const HQQ_NBITS_SUFFIX: &str = ".nbits";
const HQQ_GROUP_SIZE_SUFFIX: &str = ".group_size";
const HQQ_SHAPE_SUFFIX: &str = ".shape";

/// Read a `Tensor<T>` scalar/array as `f32` values (HQQ meta tensors arrive
/// as the model's float type after a `safetensors` load).
fn tensor_to_f32_vec<T: Float>(t: &Tensor<T>) -> FerrotorchResult<Vec<f32>> {
    t.data_vec()?
        .into_iter()
        .map(|v| cast::<T, f32>(v))
        .collect()
}

/// Read a single scalar meta tensor as `usize` (e.g. `group_size`, `nbits`).
fn tensor_scalar_usize<T: Float>(t: &Tensor<T>) -> FerrotorchResult<usize> {
    let v = t.data_vec()?;
    let first = v
        .first()
        .copied()
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "HQQ: expected a non-empty scalar meta tensor".into(),
        })?;
    let f: f64 = cast::<T, f64>(first)?;
    if f < 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("HQQ: scalar meta value {f} must be non-negative"),
        });
    }
    Ok(f as usize)
}

/// Convert one HQQ-quantized linear's raw buffers into a dense
/// `[out_features, in_features]` weight tensor. (#1172)
///
/// `w_q_t` is the `pack_4bit_u8` byte tensor (loaded as `T`; each element is
/// a `u8` value). `scale_t` / `zero_t` are the per-group meta tensors,
/// `group_size` the HQQ group size, and `shape = [out_features, in_features]`
/// the original weight shape. Only `axis = 1`, `nbits = 4` is handled here;
/// other configs return [`FerrotorchError::InvalidArgument`] so callers can
/// fall back or report a concrete gap.
///
/// # Errors
/// - `shape` is not 2-D.
/// - `nbits != 4` or `axis != 1` (the supported HQQ Q4 config).
/// - Underlying [`dequantize_hqq_q4_axis1`] validation fails.
pub fn hqq_q4_axis1_to_dense<T: Float>(
    w_q_t: &Tensor<T>,
    scale_t: &Tensor<T>,
    zero_t: &Tensor<T>,
    nbits: usize,
    group_size: usize,
    axis: usize,
    shape: &[usize],
) -> FerrotorchResult<Tensor<T>> {
    if nbits != 4 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("HQQ loader: only nbits=4 is supported, got {nbits}"),
        });
    }
    if axis != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("HQQ loader: only axis=1 is supported, got {axis}"),
        });
    }
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("HQQ loader: expected a 2-D weight shape, got {shape:?}"),
        });
    }
    let out_features = shape[0];
    let in_features = shape[1];

    // W_q bytes arrive as T-encoded u8 values; round-to-nearest then mask to
    // a byte. HQQ stores these as exact small integers, so the cast is exact.
    let w_q: Vec<u8> = w_q_t
        .data_vec()?
        .into_iter()
        .map(|v| {
            let f: f64 = cast::<T, f64>(v)?;
            Ok((f.round() as i64).clamp(0, 255) as u8)
        })
        .collect::<FerrotorchResult<Vec<u8>>>()?;

    let scale = tensor_to_f32_vec(scale_t)?;
    let zero = tensor_to_f32_vec(zero_t)?;

    let tile = HqqQ4Axis1::new(w_q, scale, zero, group_size, out_features, in_features);
    let dense_f32 = dequantize_hqq_q4_axis1(&tile)?;

    // Cast f32 dequant back to the model's element type T.
    let dense_t: Vec<T> = dense_f32
        .into_iter()
        .map(|v| cast::<f32, T>(v))
        .collect::<FerrotorchResult<Vec<T>>>()?;
    Tensor::from_storage(
        TensorStorage::cpu(dense_t),
        vec![out_features, in_features],
        false,
    )
}

/// Rewrite a raw HQQ-format state dict into a dense one the standard model
/// `load_*_state_dict` path can ingest. (#1172)
///
/// For every `{prefix}.W_q` key, reads the parallel HQQ meta tensors
/// (`scale`, `zero`, `nbits`, `group_size`, `shape`), dequantizes the linear
/// to a dense `{prefix}.weight` tensor, and copies through every key that is
/// not an HQQ meta leaf (norms, embeddings, biases, `lm_head`, …). If
/// `group_size` is absent it defaults to HQQ's documented `64`; if `nbits` is
/// absent it defaults to `4`; if `axis` is absent it defaults to `1`.
///
/// This is the production entry point consumed by
/// [`crate::model::LlamaForCausalLM::load_hqq_state_dict`].
///
/// # Errors
/// Returns [`FerrotorchError::InvalidArgument`] when a required HQQ meta
/// tensor (`scale`, `zero`, `shape`) is missing for a `W_q` key, or when the
/// dequant config is unsupported (see [`hqq_q4_axis1_to_dense`]).
pub fn hqq_state_dict_to_dense<T: Float>(raw: &StateDict<T>) -> FerrotorchResult<StateDict<T>> {
    use std::collections::BTreeSet;
    let prefixes: BTreeSet<&str> = raw
        .keys()
        .filter_map(|k| k.strip_suffix(HQQ_WQ_SUFFIX))
        .collect();

    let mut out: StateDict<T> = StateDict::with_capacity(raw.len());

    for prefix in &prefixes {
        let get = |suffix: &str| raw.get(&format!("{prefix}{suffix}"));
        let req = |suffix: &str| -> FerrotorchResult<&Tensor<T>> {
            get(suffix).ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("HQQ loader: missing required key {prefix}{suffix}"),
            })
        };

        let w_q_t = req(HQQ_WQ_SUFFIX)?;
        let scale_t = req(HQQ_SCALE_SUFFIX)?;
        let zero_t = req(HQQ_ZERO_SUFFIX)?;

        let nbits = match get(HQQ_NBITS_SUFFIX) {
            Some(t) => tensor_scalar_usize(t)?,
            None => 4,
        };
        let group_size = match get(HQQ_GROUP_SIZE_SUFFIX) {
            Some(t) => tensor_scalar_usize(t)?,
            None => 64,
        };
        // `shape` is serialized as a small int vector [out_features, in_features].
        let shape: Vec<usize> = match get(HQQ_SHAPE_SUFFIX) {
            Some(t) => tensor_to_f32_vec(t)?
                .into_iter()
                .map(|f| f.round() as usize)
                .collect(),
            None => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("HQQ loader: missing required key {prefix}{HQQ_SHAPE_SUFFIX}"),
                });
            }
        };

        let dense = hqq_q4_axis1_to_dense(w_q_t, scale_t, zero_t, nbits, group_size, 1, &shape)?;
        out.insert(format!("{prefix}.weight"), dense);
    }

    // Copy through everything that is not an HQQ meta leaf for a known
    // quantized prefix. Unrelated dense tensors (norms, embeddings, biases)
    // pass straight through.
    let meta_suffixes = [
        HQQ_WQ_SUFFIX,
        HQQ_SCALE_SUFFIX,
        HQQ_ZERO_SUFFIX,
        HQQ_NBITS_SUFFIX,
        HQQ_GROUP_SIZE_SUFFIX,
        HQQ_SHAPE_SUFFIX,
        ".axis",
        ".packing",
        ".unpack_view_dtype",
        ".view_as_float",
        ".channel_wise",
        ".optimize",
        ".round_zero",
        ".quant_scale",
        ".quant_zero",
        ".compute_dtype",
        ".encoded_state_dict",
        ".stores_quant_config",
        ".offload_meta",
    ];
    for (k, v) in raw {
        let is_hqq_meta = prefixes.iter().any(|p| {
            meta_suffixes
                .iter()
                .any(|suf| k.as_str() == format!("{p}{suf}"))
        });
        if !is_hqq_meta {
            out.insert(k.clone(), v.clone());
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pack 8 nibbles (low → high) into one i32. Helper for synthesizing
    /// test inputs.
    fn pack8(nibbles: [u32; 8]) -> i32 {
        let mut v: u32 = 0;
        for (i, n) in nibbles.iter().enumerate() {
            v |= (n & 0xF) << (4 * i);
        }
        v as i32
    }

    // -- GPTQ tests --------------------------------------------------------

    #[test]
    fn gptq_q4_dequantize_one_group_identity() {
        // out_features = 8, in_features = 8 (one group at group_size=8).
        // Pack qweight so that for every (k, n), the int4 value equals k.
        // qzeros all 1 → "true" zero = 2; scales all 1.0.
        // Expected dequantized w[n, k] = (k - 2) * 1.0 = k - 2.
        let in_features = 8;
        let out_features = 8;
        let group_size = 8;
        let qweight_rows = in_features / 8; // 1
        let mut qweight = vec![0i32; qweight_rows * out_features];
        for entry in qweight.iter_mut().take(out_features) {
            let nibbles: [u32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            *entry = pack8(nibbles);
        }
        let num_groups = in_features / group_size; // 1
        // qzeros: [num_groups, out_features/8] = [1, 1]. Pack 8 ones.
        let qzeros = vec![pack8([1; 8]); num_groups * (out_features / 8)];
        let scales = vec![1.0f32; num_groups * out_features];

        let packed = GptqQ4 {
            qweight,
            qzeros,
            scales,
            g_idx: None,
            in_features,
            out_features,
            group_size,
        };
        let out = dequantize_gptq_q4(&packed).unwrap();
        // [out_features, in_features] = [8, 8]; expect every (n, k) = k - 2.
        for n in 0..out_features {
            for k in 0..in_features {
                let v = out[n * in_features + k];
                assert!(
                    (v - (k as f32 - 2.0)).abs() < 1e-6,
                    "GPTQ dequant ({n}, {k}) = {v}, expected {}",
                    k as f32 - 2.0
                );
            }
        }
    }

    #[test]
    fn gptq_q4_rejects_non_multiple_of_8_dims() {
        let p = GptqQ4 {
            qweight: vec![0i32; 1],
            qzeros: vec![0i32; 1],
            scales: vec![0.0f32; 1],
            g_idx: None,
            in_features: 9,
            out_features: 8,
            group_size: 8,
        };
        let err = dequantize_gptq_q4(&p).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn gptq_q4_rejects_misaligned_group() {
        let p = GptqQ4 {
            qweight: vec![0i32; 16],
            qzeros: vec![0i32; 16],
            scales: vec![0.0f32; 16],
            g_idx: None,
            in_features: 16,
            out_features: 8,
            group_size: 7, // 16 % 7 != 0
        };
        let err = dequantize_gptq_q4(&p).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn gptq_q4_rejects_g_idx_length_mismatch() {
        let p = GptqQ4 {
            qweight: vec![0i32; 8],
            qzeros: vec![0i32; 1],
            scales: vec![0.0f32; 8],
            g_idx: Some(vec![0; 7]), // wrong length
            in_features: 8,
            out_features: 8,
            group_size: 8,
        };
        let err = dequantize_gptq_q4(&p).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn gptq_q4_with_two_groups_uses_per_group_scale() {
        // 16-in-features split into 2 groups of 8. Group 0 scales are 1.0,
        // group 1 scales are 2.0. Same q values everywhere → group 1 should
        // produce 2× the magnitude of group 0.
        let in_features = 16;
        let out_features = 8;
        let group_size = 8;
        let mut qweight = vec![0i32; (in_features / 8) * out_features];
        let q_const = 5;
        for n in 0..out_features {
            for k_block in 0..(in_features / 8) {
                qweight[k_block * out_features + n] = pack8([q_const; 8]);
            }
        }
        let qzeros = vec![pack8([0; 8]); 2 * (out_features / 8)];
        let mut scales = vec![1.0f32; 2 * out_features];
        for s in scales.iter_mut().skip(out_features) {
            *s = 2.0;
        }

        let packed = GptqQ4 {
            qweight,
            qzeros,
            scales,
            g_idx: None,
            in_features,
            out_features,
            group_size,
        };
        let out = dequantize_gptq_q4(&packed).unwrap();
        // q=5, zero = 0 + 1 = 1 → (5 - 1) = 4. group 0 → 4 × 1 = 4; group 1 → 4 × 2 = 8.
        for n in 0..out_features {
            for k in 0..8 {
                assert!((out[n * in_features + k] - 4.0).abs() < 1e-6);
            }
            for k in 8..16 {
                assert!((out[n * in_features + k] - 8.0).abs() < 1e-6);
            }
        }
    }

    // -- AWQ tests ---------------------------------------------------------

    #[test]
    fn awq_q4_dequantize_uniform_inputs() {
        // For uniform q across all packed lanes, the AWQ shuffle is
        // observable only via which n the value lands in. Synthesize a
        // case where every weight unpacks to q = 7, zero = 3, scale = 1
        // → dequantized w = 4 everywhere.
        let in_features = 8;
        let out_features = 8;
        let group_size = 8;
        let n_packed = out_features / 8;
        let qweight = vec![pack8([7; 8]); in_features * n_packed];
        let qzeros = vec![pack8([3; 8]); n_packed];
        let scales = vec![1.0f32; out_features];

        let packed = AwqQ4 {
            qweight,
            qzeros,
            scales,
            in_features,
            out_features,
            group_size,
        };
        let out = dequantize_awq_q4(&packed).unwrap();
        for v in &out {
            assert!((v - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn awq_q4_shuffle_order_is_distinct_from_gptq() {
        // Pack distinct q values per lane (0..8) and verify the AWQ
        // unpacker emits them in the expected channel order
        // [0, 4, 1, 5, 2, 6, 3, 7]. Use a single in_features=1, OF=8,
        // group=1 case so we can read off out_channels directly.
        let in_features = 1;
        let out_features = 8;
        let group_size = 1;
        let n_packed = out_features / 8;
        // Pack the 8 lanes with values 0..8 (low → high in the i32).
        let qweight = vec![pack8([0, 1, 2, 3, 4, 5, 6, 7]); in_features * n_packed];
        // zero = 0 in every lane.
        let qzeros = vec![pack8([0; 8]); n_packed];
        let scales = vec![1.0f32; out_features];

        let packed = AwqQ4 {
            qweight,
            qzeros,
            scales,
            in_features,
            out_features,
            group_size,
        };
        let out = dequantize_awq_q4(&packed).unwrap();
        // out[n, 0] = q-from-lane-AWQ_PACK_ORDER[shuffle_idx_for_n]
        // shuffle index 0 → lane 0 → n=0 gets q=0
        // shuffle index 1 → lane 4 → n=1 gets q=4
        // shuffle index 2 → lane 1 → n=2 gets q=1
        // shuffle index 3 → lane 5 → n=3 gets q=5
        // shuffle index 4 → lane 2 → n=4 gets q=2
        // shuffle index 5 → lane 6 → n=5 gets q=6
        // shuffle index 6 → lane 3 → n=6 gets q=3
        // shuffle index 7 → lane 7 → n=7 gets q=7
        let expected = [0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0];
        for (n, want) in expected.iter().enumerate() {
            assert!(
                (out[n * in_features] - want).abs() < 1e-6,
                "AWQ unpack n={n}: got {}, want {}",
                out[n * in_features],
                want
            );
        }
    }

    #[test]
    fn awq_q4_rejects_non_multiple_of_8_out_features() {
        let p = AwqQ4 {
            qweight: vec![0i32; 1],
            qzeros: vec![0i32; 1],
            scales: vec![0.0f32; 1],
            in_features: 8,
            out_features: 5,
            group_size: 8,
        };
        assert!(matches!(
            dequantize_awq_q4(&p).unwrap_err(),
            FerrotorchError::InvalidArgument { .. }
        ));
    }

    #[test]
    fn awq_q4_rejects_misaligned_group() {
        let p = AwqQ4 {
            qweight: vec![0i32; 16],
            qzeros: vec![0i32; 8],
            scales: vec![0.0f32; 8],
            in_features: 16,
            out_features: 8,
            group_size: 7,
        };
        assert!(matches!(
            dequantize_awq_q4(&p).unwrap_err(),
            FerrotorchError::InvalidArgument { .. }
        ));
    }

    // -------------------------------------------------------------------
    // HQQ tests (#613)
    // -------------------------------------------------------------------

    #[test]
    fn hqq_8bit_roundtrip() {
        // Build packed weights: row 0 = [0, 1, 2, 3], row 1 = [10, 20, 30, 40]
        // scale = [0.5, 0.25], zero = [1.0, 5.0]
        // Row 0 dequant: (q - 1) * 0.5 → [-0.5, 0.0, 0.5, 1.0]
        // Row 1 dequant: (q - 5) * 0.25 → [1.25, 3.75, 6.25, 8.75]
        let packed = HqqWeights {
            bits: 8,
            w_q: vec![0, 1, 2, 3, 10, 20, 30, 40],
            scale: vec![0.5, 0.25],
            zero: vec![1.0, 5.0],
            out_features: 2,
            in_features: 4,
        };
        let out = dequantize_hqq(&packed).unwrap();
        let expected: Vec<f32> = vec![-0.5, 0.0, 0.5, 1.0, 1.25, 3.75, 6.25, 8.75];
        for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "i={i}: got {got}, want {want}");
        }
    }

    #[test]
    fn hqq_4bit_roundtrip() {
        // 4 weights packed into 2 bytes (low-nibble first):
        // [0x32, 0x54] = [0,1] from byte 0 (low=2, high=3? wait)
        // Actually: 0x32 → low_nibble=0x2, high_nibble=0x3 → weights[0..2] = [2, 3]
        // 0x54 → [4, 5]
        // So unpacked = [2, 3, 4, 5].
        // scale=[1.0], zero=[0.0] → dequant = [2, 3, 4, 5]
        let packed = HqqWeights {
            bits: 4,
            w_q: vec![0x32, 0x54],
            scale: vec![1.0],
            zero: vec![0.0],
            out_features: 1,
            in_features: 4,
        };
        let out = dequantize_hqq(&packed).unwrap();
        assert_eq!(out, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn hqq_2bit_roundtrip() {
        // 4 weights per byte, lowest 2 bits first.
        // 0b11_10_01_00 = 0xE4 → weights = [0, 1, 2, 3]
        let packed = HqqWeights {
            bits: 2,
            w_q: vec![0xE4],
            scale: vec![1.0],
            zero: vec![0.0],
            out_features: 1,
            in_features: 4,
        };
        let out = dequantize_hqq(&packed).unwrap();
        assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn hqq_1bit_roundtrip() {
        // 8 weights per byte, LSB first. 0b10101010 = 0xAA → [0,1,0,1,0,1,0,1]
        let packed = HqqWeights {
            bits: 1,
            w_q: vec![0xAA],
            scale: vec![1.0],
            zero: vec![0.0],
            out_features: 1,
            in_features: 8,
        };
        let out = dequantize_hqq(&packed).unwrap();
        assert_eq!(out, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn hqq_3bit_roundtrip_known_pattern() {
        // 8 weights × 3 bits = 24 bits = 3 bytes per group.
        // Pack [1, 2, 3, 4, 5, 6, 7, 0]:
        //   bit positions: 0, 3, 6, 9, 12, 15, 18, 21
        //   value 1 at bit 0  → byte 0 += 0x01
        //   value 2 at bit 3  → byte 0 += 0x10
        //   value 3 at bit 6  → byte 0 += 0xC0 (low 2 bits) + byte 1 += 0x00 (high 1 bit)
        //   actually 3 = 0b011, at bit 6: byte0 |= 0b11<<6 = 0xC0, byte1 |= 0b0
        //   Let me just compute it programmatically below.
        let weights: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 0];
        let mut bytes = [0u8; 3];
        for (i, &w) in weights.iter().enumerate() {
            let bit_pos = i * 3;
            let byte_idx = bit_pos / 8;
            let bit_offset = bit_pos % 8;
            let v = (w as u32) << bit_offset;
            bytes[byte_idx] |= (v & 0xFF) as u8;
            if byte_idx + 1 < 3 {
                bytes[byte_idx + 1] |= ((v >> 8) & 0xFF) as u8;
            }
            if byte_idx + 2 < 3 {
                bytes[byte_idx + 2] |= ((v >> 16) & 0xFF) as u8;
            }
        }
        let packed = HqqWeights {
            bits: 3,
            w_q: bytes.to_vec(),
            scale: vec![1.0],
            zero: vec![0.0],
            out_features: 1,
            in_features: 8,
        };
        let out = dequantize_hqq(&packed).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0]);
    }

    #[test]
    fn hqq_per_row_scale_zero_applied_correctly() {
        // 2 rows × 8 cols, 8-bit packing.
        // Row 0: q = [10..18), scale=2.0, zero=10 → [0,2,4,6,8,10,12,14]
        // Row 1: q = [20..28), scale=0.5, zero=20 → [0,0.5,1,1.5,2,2.5,3,3.5]
        let mut w_q = Vec::new();
        for i in 10..18 {
            w_q.push(i as u8);
        }
        for i in 20..28 {
            w_q.push(i as u8);
        }
        let packed = HqqWeights {
            bits: 8,
            w_q,
            scale: vec![2.0, 0.5],
            zero: vec![10.0, 20.0],
            out_features: 2,
            in_features: 8,
        };
        let out = dequantize_hqq(&packed).unwrap();
        for i in 0..8 {
            assert!((out[i] - (2.0 * i as f32)).abs() < 1e-6);
            assert!((out[8 + i] - (0.5 * i as f32)).abs() < 1e-6);
        }
    }

    #[test]
    fn hqq_rejects_invalid_bits() {
        let p = HqqWeights {
            bits: 5,
            w_q: vec![],
            scale: vec![1.0],
            zero: vec![0.0],
            out_features: 1,
            in_features: 1,
        };
        assert!(matches!(
            dequantize_hqq(&p).unwrap_err(),
            FerrotorchError::InvalidArgument { .. }
        ));
    }

    #[test]
    fn hqq_rejects_short_buffer() {
        // Need 4 bytes for 8 4-bit weights, give 1.
        let p = HqqWeights {
            bits: 4,
            w_q: vec![0xFF],
            scale: vec![1.0],
            zero: vec![0.0],
            out_features: 1,
            in_features: 8,
        };
        assert!(matches!(
            dequantize_hqq(&p).unwrap_err(),
            FerrotorchError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn hqq_rejects_scale_length_mismatch() {
        let p = HqqWeights {
            bits: 8,
            w_q: vec![0; 4],
            scale: vec![1.0, 2.0, 3.0], // out_features=2 but len=3
            zero: vec![0.0, 0.0],
            out_features: 2,
            in_features: 2,
        };
        assert!(matches!(
            dequantize_hqq(&p).unwrap_err(),
            FerrotorchError::ShapeMismatch { .. }
        ));
    }

    // -------------------------------------------------------------------
    // HQQ axis=1 grouped Q4 tests (#1172)
    //
    // Oracles below were produced by running the HQQ reference
    // (mobiusml/hqq v0.2.1) `Quantizer.quantize(..., nbits=4, axis=1,
    // bitpack=True)` on a known matrix and recording the packed bytes,
    // meta scale/zero, and `Quantizer.dequantize` output. R-CHAR-3: the
    // expected values trace to the HQQ reference, not to ferrotorch.
    // -------------------------------------------------------------------

    /// Oracle: `out=4, in=8, group_size=8, axis=1`.
    /// `W = arange(32).reshape(4,8) * 0.1 - 1.0`.
    /// `Quantizer.quantize(W, nbits=4, group_size=8, axis=1, bitpack=True)`
    /// yields the packed bytes + scale/zero recorded below; the dequant
    /// values come from `Quantizer.dequantize`.
    #[test]
    fn hqq_q4_axis1_oracle_group_size_8() {
        // pack_4bit_u8 output, shape [num_groups/2=2, group_size=8].
        let w_q: Vec<u8> = vec![
            0, 34, 68, 102, 153, 187, 221, 255, // grouped rows 0,1 high/low nibbles
            0, 34, 68, 102, 153, 187, 221, 255, // grouped rows 2,3
        ];
        let scale: Vec<f32> = vec![0.046_666_67, 0.046_666_67, 0.046_666_66, 0.046_666_67];
        let zero: Vec<f32> = vec![21.428_572, 4.285_714, -12.857_144, -30.0];
        let tile = HqqQ4Axis1::new(w_q, scale, zero, 8, 4, 8);
        let out = dequantize_hqq_q4_axis1(&tile).unwrap();

        let expected: [f32; 32] = [
            -1.0, -0.906_667, -0.813_333, -0.72, -0.58, -0.486_667, -0.393_333, -0.3, -0.2,
            -0.106_667, -0.013_333, 0.08, 0.22, 0.313_333, 0.406_667, 0.5, 0.6, 0.693_333,
            0.786_667, 0.88, 1.02, 1.113_333, 1.206_667, 1.3, 1.4, 1.493_333, 1.586_667, 1.68,
            1.82, 1.913_334, 2.006_667, 2.1,
        ];
        assert_eq!(out.len(), expected.len());
        for (i, (&g, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-4, "i={i}: got {g}, want {e}");
        }
    }

    /// Discriminating oracle: `out=4, in=8, group_size=4, axis=1` — TWO
    /// groups per output row (8 groups total). The per-row [`dequantize_hqq`]
    /// model cannot represent this; the axis=1 path must.
    /// `W = (arange(32).reshape(4,8) - 16.0) * 0.25`.
    #[test]
    fn hqq_q4_axis1_oracle_two_groups_per_row() {
        // pack_4bit_u8 output, shape [num_groups/2=4, group_size=4].
        let w_q: Vec<u8> = vec![
            0, 85, 170, 255, // groups 0,4
            0, 85, 170, 255, // groups 1,5
            0, 85, 170, 255, // groups 2,6
            0, 85, 170, 255, // groups 3,7
        ];
        let scale: Vec<f32> = vec![0.05; 8];
        let zero: Vec<f32> = vec![80.0, 60.0, 40.0, 20.0, 0.0, -20.0, -40.0, -60.0];
        let tile = HqqQ4Axis1::new(w_q, scale, zero, 4, 4, 8);
        let out = dequantize_hqq_q4_axis1(&tile).unwrap();

        // From Quantizer.dequantize, reshaped to [4,8].
        let expected: [f32; 32] = [
            -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0,
            -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
            3.0, 3.25, 3.5, 3.75,
        ];
        assert_eq!(out.len(), expected.len());
        for (i, (&g, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-5, "i={i}: got {g}, want {e}");
        }
    }

    #[test]
    fn hqq_q4_axis1_rejects_odd_num_groups() {
        // numel=12, group_size=4 -> num_groups=3 (odd) -> reject.
        let tile = HqqQ4Axis1::new(vec![0u8; 8], vec![1.0; 3], vec![0.0; 3], 4, 3, 4);
        assert!(matches!(
            dequantize_hqq_q4_axis1(&tile).unwrap_err(),
            FerrotorchError::InvalidArgument { .. }
        ));
    }

    #[test]
    fn hqq_q4_axis1_rejects_scale_length_mismatch() {
        // numel=16, group_size=4 -> num_groups=4, but scale has len 3.
        let tile = HqqQ4Axis1::new(vec![0u8; 8], vec![1.0; 3], vec![0.0; 4], 4, 4, 4);
        assert!(matches!(
            dequantize_hqq_q4_axis1(&tile).unwrap_err(),
            FerrotorchError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn hqq_q4_axis1_rejects_short_buffer() {
        // numel=16, group_size=4 -> num_groups=4, step=2, need 2*4=8 bytes.
        let tile = HqqQ4Axis1::new(vec![0u8; 4], vec![1.0; 4], vec![0.0; 4], 4, 4, 4);
        assert!(matches!(
            dequantize_hqq_q4_axis1(&tile).unwrap_err(),
            FerrotorchError::ShapeMismatch { .. }
        ));
    }

    /// The production `hqq_state_dict_to_dense` path: build a raw HQQ-format
    /// state dict from the gs=4 oracle, plus a passthrough norm tensor, and
    /// verify the `.weight` key matches the reference dequant and the norm
    /// survives.
    #[test]
    fn hqq_state_dict_to_dense_produces_oracle_weight() {
        let mut raw: StateDict<f32> = StateDict::new();
        let w_q = vec![
            0.0f32, 85.0, 170.0, 255.0, 0.0, 85.0, 170.0, 255.0, 0.0, 85.0, 170.0, 255.0, 0.0,
            85.0, 170.0, 255.0,
        ];
        let mk = |d: Vec<f32>, shape: Vec<usize>| {
            Tensor::from_storage(TensorStorage::cpu(d), shape, false).unwrap()
        };
        raw.insert(
            "model.layers.0.self_attn.q_proj.W_q".into(),
            mk(w_q, vec![4, 4]),
        );
        raw.insert(
            "model.layers.0.self_attn.q_proj.scale".into(),
            mk(vec![0.05; 8], vec![8, 1]),
        );
        raw.insert(
            "model.layers.0.self_attn.q_proj.zero".into(),
            mk(
                vec![80.0, 60.0, 40.0, 20.0, 0.0, -20.0, -40.0, -60.0],
                vec![8, 1],
            ),
        );
        raw.insert(
            "model.layers.0.self_attn.q_proj.nbits".into(),
            mk(vec![4.0], vec![1]),
        );
        raw.insert(
            "model.layers.0.self_attn.q_proj.group_size".into(),
            mk(vec![4.0], vec![1]),
        );
        raw.insert(
            "model.layers.0.self_attn.q_proj.shape".into(),
            mk(vec![4.0, 8.0], vec![2]),
        );
        // A passthrough (non-quantized) tensor that must survive unchanged.
        raw.insert("model.norm.weight".into(), mk(vec![1.0, 2.0, 3.0], vec![3]));

        let dense = hqq_state_dict_to_dense(&raw).unwrap();

        // The meta keys must be gone; only `.weight` + the norm survive.
        assert!(dense.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(!dense.contains_key("model.layers.0.self_attn.q_proj.W_q"));
        assert!(!dense.contains_key("model.layers.0.self_attn.q_proj.scale"));
        assert!(dense.contains_key("model.norm.weight"));
        assert_eq!(
            dense.get("model.norm.weight").unwrap().data_vec().unwrap(),
            vec![1.0, 2.0, 3.0]
        );

        let w = dense.get("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(w.shape(), &[4, 8]);
        let expected: [f32; 32] = [
            -4.0, -3.75, -3.5, -3.25, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0,
            -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
            3.0, 3.25, 3.5, 3.75,
        ];
        for (i, (g, e)) in w
            .data_vec()
            .unwrap()
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!((g - e).abs() < 1e-5, "i={i}: got {g}, want {e}");
        }
    }
}
