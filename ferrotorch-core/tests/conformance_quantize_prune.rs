//! Conformance Phase 2.9 — `ferrotorch-core` quantization + pruning parity
//! against PyTorch.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/771>.
//! Parent: #759.
//!
//! Source files exercised:
//! - `ferrotorch-core/src/quantize.rs` — `quantize`, `dequantize`,
//!   `quantized_matmul`, `QParams`, `QuantizedTensor`, `QuantDtype`,
//!   `QuantScheme`, `MinMaxObserver`, `PerChannelMinMaxObserver`,
//!   `HistogramObserver`, `Observer` trait, `FakeQuantize`, `QatLayer`,
//!   `QatModel`, `prepare_qat`, `quantize_named_tensors`, `cuda_rng`.
//! - `ferrotorch-core/src/pruning.rs` — `magnitude_prune`, `apply_2_4_mask`,
//!   `sparsity_ratio`.
//! - `ferrotorch-core/src/grad_fns/quantize_grad.rs` —
//!   `fake_quantize_differentiable`.
//!
//! Scope per the dispatch:
//!
//! * **Quantize forwards** (CPU f32; integer-domain → bit-exact codes,
//!   dequant under `F32_REDUCTION` tolerance):
//!   - `quantize` per-tensor / per-channel for INT8, UINT8, INT4
//!   - `dequantize` (round-trip parity within one quantization step)
//!   - `quantized_matmul` (real-valued output asserted within the analytic
//!     INT8 error bound derived in `quantized_matmul_real_value_parity`)
//! * **QParams** symmetric & asymmetric for the boundary zp values
//!   `0`, `128`, and the all-positive-range cases that exercise the
//!   clamped `qmin`/`qmax` zero-point boundaries.
//! * **fake_quantize_differentiable** forward + STE backward against
//!   PyTorch's `torch.fake_quantize_per_tensor_affine`. Post-#1238 close,
//!   `fake_quantize_differentiable` is a back-compat alias for
//!   `fake_quantize_per_tensor_affine` (the canonical upstream-faithful
//!   name) and the tensor-qparams overload
//!   `fake_quantize_per_tensor_affine_tensor_qparams` mirrors upstream's
//!   `aten/src/ATen/native/quantized/FakeQuantPerTensorAffine.cpp:42-51`
//!   tensor-qparams overload. Coverage of those new surfaces is via the
//!   `#[cfg(test)] mod tests` block in
//!   `ferrotorch-core/src/grad_fns/quantize_grad.rs` (unit tests
//!   `tensor_qparams_matches_scalar_qparams`,
//!   `fake_quantize_uses_banker_rounding_on_half_boundaries`,
//!   `fake_quantize_ste_backward_matches_explicit_formula`) and via the
//!   `parity-sweep` runner's `fake_quantize_per_tensor_affine` dispatch
//!   arm in `tools/parity-sweep/runner/src/main.rs`.
//! * **Pruning** bit-exact parity for `magnitude_prune` (mask × original)
//!   and `apply_2_4_mask`. `sparsity_ratio` parity is exact (counting).
//! * **Edge cases** required by the dispatch:
//!   - symmetric vs asymmetric, per-tensor vs per-channel,
//!   - zp boundaries (zp=0, zp=128, zp=-128 derived from all-positive),
//!   - round-trip dequantize(quantize(x)) within scale,
//!   - prune sparsity correctness (% zeros matches request ± 1),
//!   - pruning preserves shape.
//! * **Observer** family + **QatModel** + **cuda_rng** are exercised
//!   via direct Rust-side unit-shaped checks (no PyTorch reference
//!   needed: they wrap CPU-domain f32 statistics or process-global
//!   `Mutex<u64>` state).
//!
//! Tolerances follow the dispatch table:
//!   - quantize integer codes: bit-exact (`assert_eq` on i32-domain)
//!   - dequantize: F32_REDUCTION (multiplication by scale)
//!   - prune masks: bit-exact (mask × original)
//!   - quantized_matmul real-valued output: analytic INT8 propagation
//!     bound `k*(|a|max*s_b + |b|max*s_a + s_a*s_b/2)/2 + s_c/2` plus f32
//!     round-off headroom (CORE-201 -> #1895; derivation at the call site)
//!
//! ## Fixture provenance (CORE-194 -> #1888)
//!
//! Every expectation in `fixtures/quantize_prune.json` is computed by a
//! REAL PyTorch API (R-ORACLE-2): `MinMaxObserver`/`PerChannelMinMaxObserver`
//! `.calculate_qparams()` for scales/zero-points, `torch.quantize_per_tensor`
//! / `torch.quantize_per_channel` (`.int_repr()`, `.dequantize()`) for
//! logical codes (decomposed ops for signed INT4), `torch.nn.utils.prune.l1_unstructured` for
//! magnitude pruning, and `torch.ao.pruning.WeightNormSparsifier` (the
//! documented 2:4 configuration) for the 2:4 mask. Each fixture row carries
//! the torch API in its `oracle` field. The previous generation of this
//! file asserted "parity" against a Python mirror of ferrotorch's own
//! algorithms and could never detect divergence.
//!
//! ## Pinned divergences (R-ORACLE-4 / R-DEFER-3)
//!
//! Where ferrotorch genuinely diverges from the torch oracle, the test pins
//! ferrotorch's CURRENT observed behavior with the tracking issue number
//! and keeps the torch value in the fixture/comment. Each pin is written to
//! FAIL when the divergence is fixed, so it retires loudly. There are no
//! remaining pinned divergences in this segment.
//!
//! Retired pins (now live assertions):
//!   - #1777 (CORE-083): magnitude_prune prunes EXACTLY n via torch CPU
//!     topk selection order, ties included.
//!   - #1778 (CORE-084): apply_2_4_mask groups along the final dimension
//!     and rejects (structured `InvalidArgument`) shapes the public torch
//!     sparsifier rejects: non-2-D inputs, empty dimensions, and rows whose
//!     final dim is not a multiple of 4.
//!   - #1909: pruning is a real mask multiplication (CORE-082 -> #1776),
//!     so pruned negative slots carry torch's -0.0 sign bit.
//!   - #1906: `compute_scale_zp` floors scale after division at `f32::EPSILON`
//!     and computes affine zero-point as
//!     `clamp(qmin - round_ties_even(min / scale), qmin, qmax)`.
//!   - #1907: `QParams::symmetric` uses PyTorch's
//!     `max_abs / ((qmax - qmin) / 2)` denominator and eps scale floor.
//!   - #1911: quantize code generation uses inverse-scale multiply plus
//!     round-half-to-even.
//!   - #1908: `magnitude_prune` computes the prune count with Python
//!     round-half-to-even semantics.
//!   - #1910: `apply_2_4_mask` uses the same CPU `topk(largest=False)`
//!     in-block tie selection as `WeightNormSparsifier`.
//!
//! Dtype/device note (per PyTorch parity):
//! `quantize`, `dequantize`, and `quantized_matmul` remain CPU-domain APIs
//! here. `torch.quantize_per_tensor` accepts f32 CPU tensors and rejects
//! CUDA, f64, f16, and bf16 inputs; `dequantize()` returns f32. ferrotorch
//! must surface structured errors instead of silently reading GPU memory
//! back to the host or narrowing unsupported dtypes. Pruning is different:
//! PyTorch constructs masks with tensor ops (`ones_like`, `topk`, `scatter`)
//! and applies `mask.to(dtype=orig.dtype) * orig`, so CUDA pruning must stay
//! CUDA-resident and differentiable. The GPU tests below pin that contract.

use std::path::PathBuf;

use serde::Deserialize;
use serde::de::{self, Deserializer, SeqAccess, Visitor};

use ferrotorch_core::pruning::{apply_2_4_mask, magnitude_prune, sparsity_ratio};
use ferrotorch_core::quantize::{
    FakeQuantize, HistogramObserver, MinMaxObserver, Observer, PerChannelMinMaxObserver, QParams,
    QatLayer, QatModel, QuantDtype, QuantScheme, QuantizedTensor, cuda_rng, dequantize,
    prepare_qat, quantize, quantize_named_tensors, quantized_matmul,
};
use ferrotorch_core::{FerrotorchError, Tensor, TensorStorage, fake_quantize_differentiable};

// ---------------------------------------------------------------------------
// Tolerance helpers
// ---------------------------------------------------------------------------
//
// Quantize integer codes: bit-exact. We assert via `assert_eq!` on i32 codes.
// Dequantize: F32_REDUCTION (1e-6) — multiplication by scale.
// Pruning: bit-exact (mask × original yields exact zeros or exact bit pattern
// of the kept input element).

mod tolerance {
    pub const F32_REDUCTION: f32 = 1e-6;

    pub fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch (actual={}, expected={})",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            if a.is_nan() && e.is_nan() {
                continue;
            }
            let diff = (a - e).abs();
            let scale = e.abs().max(1.0);
            let allowed = tol * scale;
            assert!(
                diff <= allowed,
                "{label}: index {i} delta {diff:.3e} exceeds tol {tol:.3e} \
                 (actual={a}, expected={e})"
            );
        }
    }

    /// `expected` is the original-input scale; allowed error is one
    /// quantization step (≈ scale).
    pub fn assert_within_step_f32(actual: &[f32], expected: &[f32], step: f32, label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch (actual={}, expected={})",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff <= step,
                "{label}: index {i} delta {diff:.3e} exceeds step {step:.3e} \
                 (actual={a}, expected={e})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Strict-JSON-compatible f64 list deserializer (NaN/Inf sentinels).
// Same shape used in conformance_reduction.rs.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct F64ListSentinel(Vec<f64>);

impl F64ListSentinel {
    fn as_slice(&self) -> &[f64] {
        &self.0
    }
}

struct FloatOrSentinel(f64);

struct FloatOrSentinelVisitor;

impl<'de> Visitor<'de> for FloatOrSentinelVisitor {
    type Value = FloatOrSentinel;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("an f64 or one of \"Infinity\"/\"-Infinity\"/\"NaN\"")
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
        Ok(FloatOrSentinel(v))
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
        Ok(FloatOrSentinel(v as f64))
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
        Ok(FloatOrSentinel(v as f64))
    }
    fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
        match v {
            "Infinity" => Ok(FloatOrSentinel(f64::INFINITY)),
            "-Infinity" => Ok(FloatOrSentinel(f64::NEG_INFINITY)),
            "NaN" => Ok(FloatOrSentinel(f64::NAN)),
            other => Err(E::custom(format!("unexpected float sentinel {other:?}"))),
        }
    }
}

impl<'de> serde::Deserialize<'de> for FloatOrSentinel {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(FloatOrSentinelVisitor)
    }
}

struct F64ListSentinelVisitor;

impl<'de> Visitor<'de> for F64ListSentinelVisitor {
    type Value = F64ListSentinel;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a list of floats with optional Infinity/-Infinity/NaN sentinels")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut out = Vec::new();
        while let Some(FloatOrSentinel(v)) = seq.next_element()? {
            out.push(v);
        }
        Ok(F64ListSentinel(out))
    }
}

impl<'de> serde::Deserialize<'de> for F64ListSentinel {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_seq(F64ListSentinelVisitor)
    }
}

// ---------------------------------------------------------------------------
// Fixture deserialization
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct FixtureFile {
    #[allow(dead_code, reason = "metadata used for diagnostics")]
    metadata: FixtureMetadata,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct FixtureMetadata {
    #[allow(dead_code, reason = "diagnostics only")]
    torch_version: String,
    #[allow(dead_code, reason = "diagnostics only")]
    cuda_version: Option<String>,
    #[allow(dead_code, reason = "diagnostics only")]
    cuda_available: bool,
    #[allow(dead_code, reason = "diagnostics only")]
    python_executable: String,
    #[allow(dead_code, reason = "diagnostics only")]
    python_platform: String,
    #[allow(dead_code, reason = "diagnostics only")]
    generated_at: String,
    #[allow(dead_code, reason = "diagnostics only")]
    rng_seed: u64,
    #[allow(dead_code, reason = "diagnostics only")]
    phase: String,
    #[allow(dead_code, reason = "diagnostics only")]
    tracking_issue: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    op: String,
    #[serde(default)]
    tag: Option<String>,
    #[serde(default)]
    qdtype: Option<String>,
    #[serde(default)]
    shape: Option<Vec<usize>>,
    #[serde(default)]
    a_shape: Option<Vec<usize>>,
    #[serde(default)]
    b_shape: Option<Vec<usize>>,
    #[serde(default)]
    c_shape: Option<Vec<usize>>,
    #[serde(default)]
    axis: Option<usize>,
    #[serde(default)]
    x_data: Option<F64ListSentinel>,
    #[serde(default)]
    a_data: Option<F64ListSentinel>,
    #[serde(default)]
    b_data: Option<F64ListSentinel>,
    #[serde(default)]
    c_data: Option<F64ListSentinel>,
    #[serde(default)]
    scale: Option<f64>,
    #[serde(default)]
    scales: Option<Vec<f64>>,
    #[serde(default)]
    zero_point: Option<i32>,
    #[serde(default)]
    zero_points: Option<Vec<i32>>,
    #[serde(default)]
    codes: Option<Vec<i32>>,
    #[serde(default)]
    dequant: Option<F64ListSentinel>,
    #[serde(default)]
    pruned: Option<F64ListSentinel>,
    #[serde(default)]
    masked: Option<F64ListSentinel>,
    #[serde(default)]
    sparsity: Option<f64>,
    #[serde(default)]
    n_zeros: Option<usize>,
    #[serde(default)]
    ratio: Option<f64>,
    #[serde(default)]
    max_abs: Option<f64>,
    #[serde(default)]
    min_val: Option<f64>,
    #[serde(default)]
    max_val: Option<f64>,
    #[serde(default)]
    qmin: Option<i32>,
    #[serde(default)]
    qmax: Option<i32>,
    #[serde(default)]
    y_data: Option<F64ListSentinel>,
    #[serde(default)]
    grad_x: Option<F64ListSentinel>,
    #[serde(default)]
    recovered: Option<F64ListSentinel>,
    /// Torch API the expectation was computed with (provenance; R-ORACLE-2).
    #[serde(default)]
    #[allow(dead_code, reason = "provenance metadata for diagnostics")]
    oracle: Option<String>,
    /// Set when the torch oracle REJECTED the case's input (e.g. the 2:4
    /// sparsifier refuses rows that are not a multiple of 4 wide). Such
    /// cases carry no value expectation; the suite requires ferrotorch to
    /// reject them with a structured error too.
    #[serde(default)]
    torch_error: Option<String>,
}

fn load_fixtures() -> FixtureFile {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures")
        .join("quantize_prune.json");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. Regenerate via \
             `python3 scripts/regenerate_quantize_prune_fixtures.py`",
            p.display()
        )
    });
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn cases_for<'a>(file: &'a FixtureFile, op: &str) -> Vec<&'a Fixture> {
    file.fixtures.iter().filter(|f| f.op == op).collect()
}

// ---------------------------------------------------------------------------
// Helpers — build a CPU f32 tensor from f64 fixture data.
// ---------------------------------------------------------------------------

fn make_cpu_f32(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    let v: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(v), shape.to_vec(), requires_grad)
        .expect("make_cpu_f32")
}

fn parse_qdtype(name: &str) -> QuantDtype {
    match name {
        "int8" => QuantDtype::Int8,
        "uint8" => QuantDtype::Uint8,
        "int4" => QuantDtype::Int4,
        other => panic!("unknown qdtype {other}"),
    }
}

/// Reverse of ferrotorch's storage byte representation for one logical
/// quantized code. For Uint8 the byte is reinterpreted as `u8` first
/// (0..=255 instead of -128..=127). For Int4, callers pass unpacked logical
/// codes from `QuantizedTensor::logical_data`, not packed storage bytes.
fn stored_to_code(stored: i8, qdtype: QuantDtype) -> i32 {
    match qdtype {
        QuantDtype::Uint8 => i32::from(stored as u8),
        QuantDtype::Int8 | QuantDtype::Int4 => i32::from(stored),
    }
}

// ---------------------------------------------------------------------------
// quantize per-tensor
// ---------------------------------------------------------------------------

#[test]
fn quantize_per_tensor_bit_exact_codes() {
    let file = load_fixtures();
    let cases = cases_for(&file, "quantize_per_tensor");
    assert!(!cases.is_empty(), "no quantize_per_tensor fixtures");

    for f in cases {
        let label = format!("quantize_per_tensor tag={:?} qdtype={:?}", f.tag, f.qdtype);
        let shape = f.shape.as_ref().expect("shape");
        let x_data = f.x_data.as_ref().expect("x_data").as_slice();
        let qdtype = parse_qdtype(f.qdtype.as_deref().expect("qdtype"));
        let expected_codes = f.codes.as_ref().expect("codes");
        let expected_scale = f.scale.expect("scale") as f32;
        let expected_zp = f.zero_point.expect("zero_point");

        let t = make_cpu_f32(x_data, shape, false);
        let qt: QuantizedTensor = quantize(&t, QuantScheme::PerTensor, qdtype).expect("quantize");

        // Shape preservation.
        assert_eq!(qt.shape(), shape.as_slice(), "{label}: shape mismatch");
        assert_eq!(qt.numel(), shape.iter().product::<usize>());
        assert_eq!(qt.qdtype(), qdtype);
        assert_eq!(qt.scheme(), QuantScheme::PerTensor);

        // Scale parity (within F32_REDUCTION).
        let actual_scale = qt.scale();
        assert_eq!(actual_scale.len(), 1, "{label}: per-tensor scale len != 1");
        let scale_diff = (actual_scale[0] - expected_scale).abs();
        assert!(
            scale_diff <= tolerance::F32_REDUCTION * expected_scale.abs().max(1.0),
            "{label}: scale {} vs expected {}",
            actual_scale[0],
            expected_scale
        );

        // Zero-point parity (exact i32).
        let actual_zp = qt.zero_point();
        assert_eq!(actual_zp.len(), 1, "{label}: per-tensor zp len != 1");
        assert_eq!(actual_zp[0], expected_zp, "{label}: zero_point mismatch");

        // Bit-exact logical integer codes. For Int4, `data()` is packed
        // storage, so compare through `logical_data()`.
        let stored = qt.logical_data().expect("logical qcodes");
        assert_eq!(
            stored.len(),
            expected_codes.len(),
            "{label}: code length mismatch"
        );
        for (i, (&s, &expected)) in stored.iter().zip(expected_codes.iter()).enumerate() {
            let actual = stored_to_code(s, qdtype);
            assert_eq!(
                actual, expected,
                "{label}: index {i} code mismatch (actual={actual}, expected={expected})"
            );
        }
    }
}

#[test]
fn int4_quantize_data_is_nibble_packed_low_nibble_first() {
    // PyTorch storage convention for 4-bit quantized tensors:
    // `torch.quint4x2` packs two logical 4-bit values into one byte in
    // row-major order, with the first value in the low nibble and an odd
    // final value padded by a zero high nibble. Ferrotorch's logical Int4
    // range is signed [-8, 7], so each signed code is stored as its 4-bit
    // two's-complement nibble.
    let input: Vec<f32> = (-8..=7).map(|v| v as f32).chain([0.0]).collect();
    let t = Tensor::from_storage(TensorStorage::cpu(input.clone()), vec![17], false).unwrap();

    let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int4).expect("int4 quantize");

    assert_eq!(qt.shape(), &[17]);
    assert_eq!(qt.numel(), 17);
    assert_eq!(
        qt.data().iter().map(|&byte| byte as u8).collect::<Vec<_>>(),
        vec![0x98, 0xBA, 0xDC, 0xFE, 0x10, 0x32, 0x54, 0x76, 0x00],
        "Int4 data() must expose packed storage bytes, not one i8 per element"
    );

    let rt: Tensor<f32> = dequantize(&qt).expect("dequantize packed int4");
    assert_eq!(rt.data().unwrap(), input.as_slice());
}

#[test]
fn dequantize_per_tensor_within_tolerance() {
    let file = load_fixtures();
    let cases = cases_for(&file, "quantize_per_tensor");

    for f in cases {
        let label = format!(
            "dequantize_per_tensor tag={:?} qdtype={:?}",
            f.tag, f.qdtype
        );
        let shape = f.shape.as_ref().expect("shape");
        let x_data = f.x_data.as_ref().expect("x_data").as_slice();
        let qdtype = parse_qdtype(f.qdtype.as_deref().expect("qdtype"));
        let expected_dequant = f.dequant.as_ref().expect("dequant").as_slice();

        let t = make_cpu_f32(x_data, shape, false);
        let qt = quantize(&t, QuantScheme::PerTensor, qdtype).expect("quantize");
        let rt: Tensor<f32> = dequantize(&qt).expect("dequantize");

        assert_eq!(rt.shape(), shape.as_slice(), "{label}: dequant shape");

        let actual: Vec<f32> = rt.data().expect("dequant data").to_vec();
        let expected_f32: Vec<f32> = expected_dequant.iter().map(|&v| v as f32).collect();
        tolerance::assert_close_f32(&actual, &expected_f32, tolerance::F32_REDUCTION, &label);
    }
}

// ---------------------------------------------------------------------------
// quantize per-channel
// ---------------------------------------------------------------------------

#[test]
fn quantize_per_channel_bit_exact_codes() {
    let file = load_fixtures();
    let cases = cases_for(&file, "quantize_per_channel");
    assert!(!cases.is_empty(), "no quantize_per_channel fixtures");

    for f in cases {
        let label = format!("quantize_per_channel tag={:?} qdtype={:?}", f.tag, f.qdtype);
        let shape = f.shape.as_ref().expect("shape");
        let axis = f.axis.expect("axis");
        let x_data = f.x_data.as_ref().expect("x_data").as_slice();
        let qdtype = parse_qdtype(f.qdtype.as_deref().expect("qdtype"));
        let expected_codes = f.codes.as_ref().expect("codes");
        let expected_scales = f.scales.as_ref().expect("scales");
        let expected_zps = f.zero_points.as_ref().expect("zero_points");

        let t = make_cpu_f32(x_data, shape, false);
        let qt = quantize(&t, QuantScheme::PerChannel(axis), qdtype).expect("quantize");

        // Shape and per-channel param length.
        assert_eq!(qt.shape(), shape.as_slice(), "{label}: shape");
        assert_eq!(
            qt.scale().len(),
            shape[axis],
            "{label}: scale len != channels"
        );
        assert_eq!(
            qt.zero_point().len(),
            shape[axis],
            "{label}: zp len != channels"
        );

        // Per-channel scale parity.
        for (i, (&actual, &expected)) in qt.scale().iter().zip(expected_scales.iter()).enumerate() {
            let exp_f32 = expected as f32;
            let diff = (actual - exp_f32).abs();
            assert!(
                diff <= tolerance::F32_REDUCTION * exp_f32.abs().max(1.0),
                "{label}: channel {i} scale {actual} vs {exp_f32}"
            );
        }

        // Per-channel zp parity (exact).
        for (i, (&actual, &expected)) in qt.zero_point().iter().zip(expected_zps.iter()).enumerate()
        {
            assert_eq!(actual, expected, "{label}: channel {i} zero_point mismatch");
        }

        // Bit-exact logical codes in the original flat order. For Int4,
        // `data()` is packed storage, so compare through `logical_data()`.
        let stored = qt.logical_data().expect("logical qcodes");
        assert_eq!(
            stored.len(),
            expected_codes.len(),
            "{label}: code length mismatch"
        );
        for (i, (&stored, &expected)) in stored.iter().zip(expected_codes.iter()).enumerate() {
            let actual = stored_to_code(stored, qdtype);
            assert_eq!(actual, expected, "{label}: index {i} code mismatch");
        }
    }
}

#[test]
fn dequantize_per_channel_within_tolerance() {
    let file = load_fixtures();
    let cases = cases_for(&file, "quantize_per_channel");

    for f in cases {
        let label = format!(
            "dequantize_per_channel tag={:?} qdtype={:?}",
            f.tag, f.qdtype
        );
        let shape = f.shape.as_ref().expect("shape");
        let axis = f.axis.expect("axis");
        let x_data = f.x_data.as_ref().expect("x_data").as_slice();
        let qdtype = parse_qdtype(f.qdtype.as_deref().expect("qdtype"));
        let expected_dequant = f.dequant.as_ref().expect("dequant").as_slice();

        let t = make_cpu_f32(x_data, shape, false);
        let qt = quantize(&t, QuantScheme::PerChannel(axis), qdtype).expect("quantize");
        let rt: Tensor<f32> = dequantize(&qt).expect("dequantize");

        let actual = rt.data().expect("dequant").to_vec();
        let expected_f32: Vec<f32> = expected_dequant.iter().map(|&v| v as f32).collect();
        tolerance::assert_close_f32(&actual, &expected_f32, tolerance::F32_REDUCTION, &label);
    }
}

#[test]
fn quantize_per_channel_axis_out_of_range_errors() {
    // Axis 5 on a 1-D tensor must error.
    let t = make_cpu_f32(&[1.0, 2.0, 3.0], &[3], false);
    let res = quantize(&t, QuantScheme::PerChannel(5), QuantDtype::Int8);
    assert!(res.is_err(), "expected error for axis out of range");
}

// ---------------------------------------------------------------------------
// QParams symmetric / asymmetric
// ---------------------------------------------------------------------------

#[test]
fn qparams_symmetric_parity() {
    let file = load_fixtures();
    let cases = cases_for(&file, "qparams_symmetric");
    assert!(!cases.is_empty(), "no qparams_symmetric fixtures");

    for f in cases {
        let label = format!("qparams_symmetric tag={:?}", f.tag);
        let qdtype = parse_qdtype(f.qdtype.as_deref().expect("qdtype"));
        let max_abs = f.max_abs.expect("max_abs") as f32;
        let expected_scale = f.scale.expect("scale") as f32;
        let expected_zp = f.zero_point.expect("zero_point");

        let qp: QParams = QParams::symmetric(max_abs, qdtype);
        assert_eq!(qp.scale.len(), 1);
        assert_eq!(qp.zero_point.len(), 1);
        // Zero points match torch (0 for signed, 128 for uint8).
        assert_eq!(qp.zero_point[0], expected_zp, "{label}: zp");

        let torch_diff = (qp.scale[0] - expected_scale).abs();
        assert!(
            torch_diff <= tolerance::F32_REDUCTION * expected_scale.abs().max(1.0),
            "{label}: scale {} vs torch oracle {expected_scale}",
            qp.scale[0]
        );
    }
}

#[test]
fn qparams_asymmetric_parity() {
    let file = load_fixtures();
    let cases = cases_for(&file, "qparams_asymmetric");
    assert!(!cases.is_empty(), "no qparams_asymmetric fixtures");

    for f in cases {
        let label = format!("qparams_asymmetric tag={:?}", f.tag);
        let qdtype = parse_qdtype(f.qdtype.as_deref().expect("qdtype"));
        let mn = f.min_val.expect("min_val") as f32;
        let mx = f.max_val.expect("max_val") as f32;
        let expected_scale = f.scale.expect("scale") as f32;
        let expected_zp = f.zero_point.expect("zero_point");

        let qp = QParams::asymmetric(mn, mx, qdtype);
        assert_eq!(qp.zero_point[0], expected_zp, "{label}: zp");
        let diff = (qp.scale[0] - expected_scale).abs();
        assert!(
            diff <= tolerance::F32_REDUCTION * expected_scale.abs().max(1.0),
            "{label}: scale {} vs {}",
            qp.scale[0],
            expected_scale
        );
    }
}

// ---------------------------------------------------------------------------
// QuantizedTensor accessors
// ---------------------------------------------------------------------------

#[test]
fn quantized_tensor_accessors_round_trip_metadata() {
    // Exercises every public accessor on QuantizedTensor:
    //   QuantizedTensor::numel
    //   QuantizedTensor::shape
    //   QuantizedTensor::data
    //   QuantizedTensor::scale
    //   QuantizedTensor::zero_point
    //   QuantizedTensor::scheme
    //   QuantizedTensor::qdtype
    let t = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let qt = quantize(&t, QuantScheme::PerTensor, QuantDtype::Int8).expect("quantize");

    assert_eq!(QuantizedTensor::numel(&qt), 6);
    assert_eq!(QuantizedTensor::shape(&qt), &[2, 3]);
    assert_eq!(QuantizedTensor::data(&qt).len(), 6);
    assert_eq!(QuantizedTensor::scale(&qt).len(), 1);
    assert_eq!(QuantizedTensor::zero_point(&qt).len(), 1);
    assert_eq!(QuantizedTensor::scheme(&qt), QuantScheme::PerTensor);
    assert_eq!(QuantizedTensor::qdtype(&qt), QuantDtype::Int8);
}

// ---------------------------------------------------------------------------
// quantized_matmul
// ---------------------------------------------------------------------------

#[test]
fn quantized_matmul_real_value_parity() {
    let file = load_fixtures();
    let cases = cases_for(&file, "quantized_matmul");
    assert!(!cases.is_empty(), "no quantized_matmul fixtures");

    for f in cases {
        let label = format!("quantized_matmul tag={:?}", f.tag);
        let a_shape = f.a_shape.as_ref().expect("a_shape");
        let b_shape = f.b_shape.as_ref().expect("b_shape");
        let c_shape = f.c_shape.as_ref().expect("c_shape");
        let a_data = f.a_data.as_ref().expect("a_data").as_slice();
        let b_data = f.b_data.as_ref().expect("b_data").as_slice();
        let c_data = f.c_data.as_ref().expect("c_data").as_slice();

        let a = make_cpu_f32(a_data, a_shape, false);
        let b = make_cpu_f32(b_data, b_shape, false);

        let qa = quantize(&a, QuantScheme::PerTensor, QuantDtype::Int8).expect("qa");
        let qb = quantize(&b, QuantScheme::PerTensor, QuantDtype::Int8).expect("qb");
        let qc = quantized_matmul(&qa, &qb).expect("qmatmul");
        assert_eq!(qc.shape(), c_shape.as_slice(), "{label}: out shape");

        let c_back: Tensor<f32> = dequantize(&qc).expect("dequantize");
        let actual = c_back.data().expect("c data").to_vec();
        let expected_f32: Vec<f32> = c_data.iter().map(|&v| v as f32).collect();

        // Analytic tolerance (R-ORACLE-5) — derived from the quantization
        // scheme in src/quantize.rs (CORE-201 -> #1895; the previous
        // `.max(0.5)` floor had no analytic justification and absorbed
        // errors up to half a unit on these fixtures):
        //
        // 1. Per-tensor INT8 quantize covers each input's own [min, max]
        //    (zero-extended), so every dequantized element satisfies
        //    |x_hat - x| <= s/2 with s the input's scale (round-to-nearest;
        //    `quantize_val`'s clamp cannot push the error past s/2 because
        //    x/s + zp lies within [qmin - 0.5, qmax + 0.5] for in-range x).
        // 2. `quantized_matmul` accumulates (qa - za)*(qb - zb) EXACTLY in
        //    i32, so its pre-requantize real value is exactly the matmul of
        //    the dequantized inputs: c_hat = A_hat @ B_hat. Against torch's
        //    float oracle c = A @ B, each k-length dot product obeys
        //      |c_hat - c| = |sum_p (a*e_b + b*e_a + e_a*e_b)|
        //                 <= k * (|a|max*s_b + |b|max*s_a + s_a*s_b/2) / 2,
        //    with |e_a| <= s_a/2, |e_b| <= s_b/2 from step 1.
        // 3. The internal requantize to INT8 (out scale s_c computed from
        //    c_hat's own min/max) followed by the test's dequantize adds at
        //    most one half output step: s_c/2.
        // 4. f32 round-off (the acc * s_a * s_b multiply, scale computation)
        //    is a few ULPs relative to |c|; F32_REDUCTION * max(|c|, 1)
        //    covers it.
        let s_a = qa.scale()[0];
        let s_b = qb.scale()[0];
        let s_c = qc.scale()[0];
        let k = a_shape[1] as f32;
        let a_max = a_data.iter().fold(0.0_f32, |m, &v| m.max((v as f32).abs()));
        let b_max = b_data.iter().fold(0.0_f32, |m, &v| m.max((v as f32).abs()));
        let c_max = expected_f32.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        let bound = k * (a_max * s_b + b_max * s_a + s_a * s_b * 0.5) / 2.0
            + s_c * 0.5
            + tolerance::F32_REDUCTION * c_max.max(1.0);
        tolerance::assert_within_step_f32(&actual, &expected_f32, bound, &label);
    }
}

#[test]
fn quantized_matmul_shape_mismatch_errors() {
    let a = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
    let b = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let qa = quantize(&a, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
    let qb = quantize(&b, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
    assert!(quantized_matmul(&qa, &qb).is_err());
}

#[test]
fn quantized_matmul_reads_odd_length_packed_int4_operands() {
    // A [1, 3] x [3, 1] product gives each Int4 input an odd logical
    // element count. Packed storage is ceil(3 / 2) = 2 bytes; the matmul
    // path must index logical codes, not raw bytes.
    let a = make_cpu_f32(&[-1.0, 0.0, 1.0], &[1, 3], false);
    let b = make_cpu_f32(&[2.0, -2.0, 3.0], &[3, 1], false);
    let qa = quantize(&a, QuantScheme::PerTensor, QuantDtype::Int4).unwrap();
    let qb = quantize(&b, QuantScheme::PerTensor, QuantDtype::Int4).unwrap();

    assert_eq!(qa.data().len(), 2);
    assert_eq!(qb.data().len(), 2);
    assert_eq!(qa.logical_data().unwrap().len(), 3);
    assert_eq!(qb.logical_data().unwrap().len(), 3);

    let qc = quantized_matmul(&qa, &qb).expect("packed int4 matmul");
    let c: Tensor<f32> = dequantize(&qc).expect("dequantized result");
    let got = c.data().expect("data")[0];

    // Dequantized inputs are:
    // A = [-0.9333334, 0.0, 0.9333334], B = [2.0, -2.0, 3.0].
    // The real product is 14 / 15. The requantized single-output result
    // includes zero in its range and dequantizes back to the same value
    // within f32 multiplication/rounding error.
    assert!(
        (got - (14.0 / 15.0)).abs() <= 2.0e-6,
        "packed int4 matmul result {got} diverged from expected 14/15"
    );
}

#[test]
fn quantized_matmul_accumulator_crosses_i32_boundary_without_wrapping() {
    // Quantizing an all-ones tensor to qint8 over the zero-extended [0, 1]
    // range maps every entry to q=127 with zero_point=-128, so each centered
    // multiplicand is 255 and each integer product is 65025. This reaches the
    // i32 accumulator ceiling at K ~= 33k while the true real result remains K.
    let product = 255_i64 * 255_i64;
    let max_i32_safe_k = (i64::from(i32::MAX) / product) as usize;

    for k in [max_i32_safe_k, max_i32_safe_k + 2] {
        let a = make_cpu_f32(&vec![1.0; k], &[1, k], false);
        let b = make_cpu_f32(&vec![1.0; k], &[k, 1], false);
        let qa = quantize(&a, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
        let qb = quantize(&b, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();

        let raw_acc = product * k as i64;
        assert_eq!(k == max_i32_safe_k, raw_acc <= i64::from(i32::MAX));

        let qc = quantized_matmul(&qa, &qb).expect("qmatmul should not overflow");
        let c: Tensor<f32> = dequantize(&qc).expect("dequantize");
        let got = c.data().expect("data")[0];
        let expected = k as f32;
        let allowed = qc.scale()[0].abs() * 0.5 + expected.abs() * 1e-6;
        assert!(
            (got - expected).abs() <= allowed,
            "K={k}: quantized_matmul should produce real result {expected}, got {got}; \
             raw centered accumulator was {raw_acc}"
        );
    }
}

#[test]
fn quantized_matmul_negative_accumulator_crosses_i32_boundary_without_wrapping() {
    let product = 255_i64 * -255_i64;
    let k = (i64::from(i32::MAX) / product.abs()) as usize + 2;

    let a = make_cpu_f32(&vec![1.0; k], &[1, k], false);
    let b = make_cpu_f32(&vec![-1.0; k], &[k, 1], false);
    let qa = quantize(&a, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();
    let qb = quantize(&b, QuantScheme::PerTensor, QuantDtype::Int8).unwrap();

    let raw_acc = product * k as i64;
    assert!(raw_acc < i64::from(i32::MIN));

    let qc = quantized_matmul(&qa, &qb).expect("qmatmul should not overflow");
    let c: Tensor<f32> = dequantize(&qc).expect("dequantize");
    let got = c.data().expect("data")[0];
    let expected = -(k as f32);
    let allowed = qc.scale()[0].abs() * 0.5 + expected.abs() * 1e-6;
    assert!(
        (got - expected).abs() <= allowed,
        "K={k}: quantized_matmul should produce real result {expected}, got {got}; \
         raw centered accumulator was {raw_acc}"
    );
}

// ---------------------------------------------------------------------------
// quantize_named_tensors
// ---------------------------------------------------------------------------

#[test]
fn quantize_named_tensors_returns_named_map() {
    // PyTorch parity: applying quantize to each named tensor in a
    // module's state_dict.
    let w1 = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    let w2 = make_cpu_f32(&[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], &[3, 2], false);

    let named = vec![
        ("layer1.weight".to_string(), w1),
        ("layer2.weight".to_string(), w2),
    ];
    let qmap =
        quantize_named_tensors(named, QuantScheme::PerTensor, QuantDtype::Int8).expect("qmap");

    assert_eq!(qmap.len(), 2);
    assert!(qmap.contains_key("layer1.weight"));
    assert!(qmap.contains_key("layer2.weight"));
    assert_eq!(qmap["layer1.weight"].shape(), &[2, 2]);
    assert_eq!(qmap["layer2.weight"].shape(), &[3, 2]);
}

// ---------------------------------------------------------------------------
// Round-trip dequantize(quantize(x)) ≈ x within one quantization step
// ---------------------------------------------------------------------------

#[test]
fn quantize_dequantize_round_trip_within_step() {
    let file = load_fixtures();
    let cases = cases_for(&file, "roundtrip");
    assert!(!cases.is_empty(), "no roundtrip fixtures");

    for f in cases {
        let label = format!("roundtrip tag={:?} qdtype={:?}", f.tag, f.qdtype);
        let shape = f.shape.as_ref().expect("shape");
        let x_data = f.x_data.as_ref().expect("x_data").as_slice();
        let qdtype = parse_qdtype(f.qdtype.as_deref().expect("qdtype"));
        let expected_recovered = f.recovered.as_ref().expect("recovered").as_slice();
        let scale = f.scale.expect("scale") as f32;

        let t = make_cpu_f32(x_data, shape, false);
        let qt = quantize(&t, QuantScheme::PerTensor, qdtype).expect("quantize");
        let rt: Tensor<f32> = dequantize(&qt).expect("dequantize");

        let actual = rt.data().expect("rt data").to_vec();

        let expected_f32: Vec<f32> = expected_recovered.iter().map(|&v| v as f32).collect();

        // Recovered must equal the reference dequantize; tolerance
        // F32_REDUCTION (multiplication by scale).
        tolerance::assert_close_f32(&actual, &expected_f32, tolerance::F32_REDUCTION, &label);

        // Also: recovered must be within one quantization step of the
        // original input (this is the round-trip property the dispatch
        // requires).
        let x_f32: Vec<f32> = x_data.iter().map(|&v| v as f32).collect();
        for (i, (&a, &x)) in actual.iter().zip(x_f32.iter()).enumerate() {
            let err = (a - x).abs();
            assert!(
                err <= scale + tolerance::F32_REDUCTION * scale.abs().max(1.0),
                "{label}: round-trip index {i} error {err:.3e} > step {scale:.3e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// fake_quantize_differentiable forward + STE backward parity
// ---------------------------------------------------------------------------

#[test]
fn fake_quantize_differentiable_forward_and_ste_backward() {
    let file = load_fixtures();
    let cases = cases_for(&file, "fake_quantize_differentiable");
    assert!(
        !cases.is_empty(),
        "no fake_quantize_differentiable fixtures"
    );

    for f in cases {
        let label = format!("fake_quantize_differentiable tag={:?}", f.tag);
        let x_data = f.x_data.as_ref().expect("x_data").as_slice();
        let scale = f.scale.expect("scale");
        let zp = f.zero_point.expect("zero_point");
        let qmin = f.qmin.expect("qmin");
        let qmax = f.qmax.expect("qmax");
        let expected_y = f.y_data.as_ref().expect("y_data").as_slice();
        let expected_grad = f.grad_x.as_ref().expect("grad_x").as_slice();

        // Forward + autograd: build x with requires_grad and call
        // y = fake_quantize_differentiable(x, ...). Then y.sum().backward().
        let shape = vec![x_data.len()];
        let x = make_cpu_f32(x_data, &shape, true);
        let y = fake_quantize_differentiable(&x, scale, zp, qmin, qmax).expect("fq_differentiable");

        let actual_y = y.data().expect("y data").to_vec();
        let expected_y_f32: Vec<f32> = expected_y.iter().map(|&v| v as f32).collect();
        tolerance::assert_close_f32(
            &actual_y,
            &expected_y_f32,
            tolerance::F32_REDUCTION,
            &format!("{label} forward"),
        );

        // Backward: use sum() on y as a scalar loss, like PyTorch's reference.
        let loss = ferrotorch_core::grad_fns::reduction::sum(&y).expect("sum");
        loss.backward().expect("backward");
        let grad = x.grad().expect("grad opt").expect("grad value");
        let actual_grad = grad.data().expect("grad data").to_vec();
        let expected_grad_f32: Vec<f32> = expected_grad.iter().map(|&v| v as f32).collect();
        // The STE mask is the binary indicator of in-range; multiplying by
        // the upstream gradient (all 1's) yields exactly the indicator.
        // Bit-exact after a single mul-by-1 is fine, but allow F32_REDUCTION
        // headroom.
        tolerance::assert_close_f32(
            &actual_grad,
            &expected_grad_f32,
            tolerance::F32_REDUCTION,
            &format!("{label} backward"),
        );
    }
}

// ---------------------------------------------------------------------------
// magnitude_prune — bit-exact mask × original
// ---------------------------------------------------------------------------

#[test]
fn magnitude_prune_bit_exact_and_sparsity() {
    let file = load_fixtures();
    let cases = cases_for(&file, "magnitude_prune");
    assert!(!cases.is_empty(), "no magnitude_prune fixtures");

    for f in cases {
        let label = format!("magnitude_prune tag={:?}", f.tag);
        let shape = f.shape.as_ref().expect("shape");
        let x_data = f.x_data.as_ref().expect("x_data").as_slice();
        let sparsity = f.sparsity.expect("sparsity");
        let expected_pruned = f.pruned.as_ref().expect("pruned").as_slice();
        let expected_zeros = f.n_zeros.expect("n_zeros");

        let t = make_cpu_f32(x_data, shape, false);
        let pruned = magnitude_prune(&t, sparsity).expect("magnitude_prune");

        // Shape preservation.
        assert_eq!(pruned.shape(), shape.as_slice(), "{label}: shape");

        let actual = pruned.data().expect("pruned data").to_vec();
        let expected_f32: Vec<f32> = expected_pruned.iter().map(|&v| v as f32).collect();

        // Bit-exact INCLUDING the sign of pruned slots: ferrotorch now
        // applies the same `weight_orig * mask` multiplication as torch's
        // pruning parametrization, so pruned negative weights yield -0.0
        // exactly like the oracle (#1909 pin retired with the CORE-082
        // mask-multiplication fix). The test data is exactly representable
        // in f32.
        for (i, (&a, &e)) in actual.iter().zip(expected_f32.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: index {i} bit pattern (actual={a}, expected={e})"
            );
        }

        // Zero count: single contract — the torch oracle's count
        // (R-ORACLE-4; the previous `== requested || == fixture` form was
        // dual-accepting).
        let zeros = actual.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(
            zeros, expected_zeros,
            "{label}: zero count {zeros} != torch-oracle count {expected_zeros}"
        );
    }
}

#[test]
fn magnitude_prune_invalid_sparsity_errors() {
    let t = make_cpu_f32(&[1.0], &[1], false);
    assert!(magnitude_prune(&t, 1.0).is_err());
    assert!(magnitude_prune(&t, -0.1).is_err());
}

// ---------------------------------------------------------------------------
// apply_2_4_mask — bit-exact mask
// ---------------------------------------------------------------------------

#[test]
fn apply_2_4_mask_bit_exact_and_sparsity() {
    let file = load_fixtures();
    let cases = cases_for(&file, "apply_2_4_mask");
    assert!(!cases.is_empty(), "no apply_2_4_mask fixtures");

    for f in cases {
        let label = format!("apply_2_4_mask tag={:?}", f.tag);
        let shape = f.shape.as_ref().expect("shape");
        let x_data = f.x_data.as_ref().expect("x_data").as_slice();

        // CORE-084 (#1778) pins RETIRED: torch's public 2:4 sparsifier
        // REJECTS non-2-D inputs, empty dimensions, and rows that are not
        // a multiple of 4 wide (the fixture records the torch rejection in
        // `torch_error` and carries no `masked` expectation; live torch
        // 2.11.0+cu130:
        // `[4]` -> `ValueError: not enough values to unpack`;
        // `AssertionError: mask shape (torch.Size([2, 8])) must match x
        // shape (torch.Size([2, 6]))`). ferrotorch now matches with a
        // structured `InvalidArgument` instead of silently flat-grouping
        // across row boundaries or reshaping rank-1 tensors.
        if let Some(torch_error) = &f.torch_error {
            assert!(
                f.masked.is_none(),
                "{label}: fixture carries both torch_error and masked"
            );
            let res = apply_2_4_mask(&make_cpu_f32(x_data, shape, false));
            assert!(
                matches!(
                    &res,
                    Err(ferrotorch_core::FerrotorchError::InvalidArgument { .. })
                ),
                "{label}: torch rejects this shape ({torch_error}); ferrotorch \
                 must return Err(InvalidArgument), got {res:?}"
            );
            continue;
        }

        let t = make_cpu_f32(x_data, shape, false);
        let masked = apply_2_4_mask(&t).expect("apply_2_4_mask");

        // Shape preservation.
        assert_eq!(masked.shape(), shape.as_slice(), "{label}: shape");
        let actual = masked.data().expect("masked data").to_vec();

        let expected_masked = f.masked.as_ref().expect("masked").as_slice();
        let expected_zeros = f.n_zeros.expect("n_zeros");
        let expected_f32: Vec<f32> = expected_masked.iter().map(|&v| v as f32).collect();

        // Bit-exact INCLUDING the sign of pruned slots (#1909 pin retired
        // with the CORE-082 mask-multiplication fix; see
        // magnitude_prune_bit_exact_and_sparsity).
        for (i, (&a, &e)) in actual.iter().zip(expected_f32.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: index {i} bit pattern (actual={a}, expected={e})"
            );
        }

        let zeros = actual.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(zeros, expected_zeros, "{label}: zero count");
    }
}

// ---------------------------------------------------------------------------
// sparsity_ratio
// ---------------------------------------------------------------------------

#[test]
fn sparsity_ratio_parity() {
    let file = load_fixtures();
    let cases = cases_for(&file, "sparsity_ratio");
    assert!(!cases.is_empty(), "no sparsity_ratio fixtures");

    for f in cases {
        let label = format!("sparsity_ratio tag={:?}", f.tag);
        let shape = f.shape.as_ref().expect("shape");
        let x_data = f.x_data.as_ref().expect("x_data").as_slice();
        let expected = f.ratio.expect("ratio");

        let t = make_cpu_f32(x_data, shape, false);
        let actual = sparsity_ratio(&t).expect("sparsity_ratio");
        assert!(
            (actual - expected).abs() < 1e-12,
            "{label}: ratio {actual} vs {expected}"
        );
    }
}

// ---------------------------------------------------------------------------
// Pruning preserves requires_grad (PyTorch-parity contract: prune is a
// data transform that keeps the graph node alive).
// ---------------------------------------------------------------------------

#[test]
fn pruning_preserves_requires_grad() {
    let t = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4], true);
    assert!(t.requires_grad());

    let pruned = magnitude_prune(&t, 0.5).expect("magnitude_prune");
    assert!(
        pruned.requires_grad(),
        "magnitude_prune must preserve requires_grad"
    );

    let masked = apply_2_4_mask(&t).expect("apply_2_4_mask");
    assert!(
        masked.requires_grad(),
        "apply_2_4_mask must preserve requires_grad"
    );
}

// ---------------------------------------------------------------------------
// CORE-082 (#1776): pruning is a differentiable mask multiplication — the
// backward pass reaches the ORIGINAL parameter with torch's masked gradient
// (exact zeros at pruned slots), never a fresh disconnected leaf
// (R-ORACLE-3: assert gradient FLOW, not flags).
// ---------------------------------------------------------------------------

#[test]
// reason: the masked gradient is grad_upstream * {0.0, 1.0}; both factors
// and products are exactly representable, so equality is the right check.
#[allow(clippy::float_cmp)]
fn magnitude_prune_backward_flows_masked_gradient_to_original_leaf() {
    // Live torch 2.11.0+cu130 oracle (R-ORACLE-1b):
    //   >>> m.weight = nn.Parameter(torch.tensor([1., -4., 2., -3.]))
    //   >>> prune.l1_unstructured(m, "weight", 0.5)
    //   >>> (m.weight * torch.tensor([10., 20., 30., 40.])).sum().backward()
    //   >>> m.weight_orig.grad
    //   tensor([ 0., 20.,  0., 40.])
    let x = make_cpu_f32(&[1.0, -4.0, 2.0, -3.0], &[4], true);
    let pruned = magnitude_prune(&x, 0.5).expect("magnitude_prune");

    let coeffs = make_cpu_f32(&[10.0, 20.0, 30.0, 40.0], &[4], false);
    let prod = ferrotorch_core::grad_fns::arithmetic::mul(&pruned, &coeffs).expect("mul");
    let loss = ferrotorch_core::grad_fns::reduction::sum(&prod).expect("sum");
    loss.backward().expect("backward");

    let grad = x
        .grad()
        .expect("grad access")
        .expect("ORIGINAL leaf must receive a gradient (torch: weight_orig.grad)");
    let g = grad.data().expect("grad data").to_vec();
    assert_eq!(
        g,
        vec![0.0, 20.0, 0.0, 40.0],
        "masked gradient on the original parameter (torch: [0, 20, 0, 40])"
    );
    // Pruned slots carry EXACT zeros (grad * 0.0-mask).
    assert_eq!(g[0].to_bits(), 0.0_f32.to_bits());
    assert_eq!(g[2].to_bits(), 0.0_f32.to_bits());
}

#[test]
// reason: see magnitude_prune_backward_flows_masked_gradient_to_original_leaf.
#[allow(clippy::float_cmp)]
fn apply_2_4_mask_backward_flows_masked_gradient_to_original_leaf() {
    // Same masked-gradient contract as l1_unstructured's mask
    // parametrization (weight = weight_orig * mask => d/d weight_orig =
    // grad * mask): pruned slots get exact-zero gradient, kept slots pass
    // the upstream gradient through.
    // ferrotorch keeps idx {1,3} in group 0 and idx {2,3} in group 1.
    let x = make_cpu_f32(&[1.0, -4.0, 2.0, -3.0, 0.5, 0.1, 0.9, 0.8], &[2, 4], true);
    let masked = apply_2_4_mask(&x).expect("apply_2_4_mask");

    let coeffs = make_cpu_f32(
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        &[2, 4],
        false,
    );
    let prod = ferrotorch_core::grad_fns::arithmetic::mul(&masked, &coeffs).expect("mul");
    let loss = ferrotorch_core::grad_fns::reduction::sum(&prod).expect("sum");
    loss.backward().expect("backward");

    let grad = x
        .grad()
        .expect("grad access")
        .expect("ORIGINAL leaf must receive a gradient through the 2:4 mask");
    let g = grad.data().expect("grad data").to_vec();
    assert_eq!(
        g,
        vec![0.0, 20.0, 0.0, 40.0, 0.0, 0.0, 70.0, 80.0],
        "masked gradient on the original parameter"
    );
}

// ---------------------------------------------------------------------------
// MinMaxObserver / PerChannelMinMaxObserver / HistogramObserver
// (Observer trait family).
// These wrap CPU statistics on f32 slices — no PyTorch reference is required;
// we exercise the documented contract instead.
// ---------------------------------------------------------------------------

#[test]
fn minmax_observer_calculates_qparams_for_int8() {
    let mut obs: MinMaxObserver = MinMaxObserver::new();
    obs.observe(&[1.0, 2.0, 3.0]).expect("observe");
    obs.observe(&[-1.0, 5.0]).expect("observe");
    let qp = <MinMaxObserver as Observer>::calculate_qparams(&obs, QuantDtype::Int8);
    assert_eq!(qp.scale.len(), 1);
    assert_eq!(qp.zero_point.len(), 1);
    // Range is [-1, 5] -> includes-zero already, scale = 6/255.
    let expected_scale = 6.0_f32 / 255.0;
    assert!(
        (qp.scale[0] - expected_scale).abs() < 1e-5,
        "expected scale {expected_scale}, got {}",
        qp.scale[0]
    );
}

#[test]
fn minmax_observer_filters_nan_inf() {
    let mut obs = MinMaxObserver::new();
    obs.observe(&[1.0, f32::NAN, 2.0, f32::INFINITY, -1.0, f32::NEG_INFINITY])
        .expect("observe");
    let qp = obs.calculate_qparams(QuantDtype::Int8);
    let expected_scale = 3.0_f32 / 255.0;
    assert!((qp.scale[0] - expected_scale).abs() < 1e-5);

    // reset() clears the observer state.
    obs.reset();
    let qp2 = obs.calculate_qparams(QuantDtype::Int8);
    // PyTorch MinMaxObserver returns default qparams after reset.
    assert_eq!(qp2.scale, vec![1.0]);
    assert_eq!(qp2.zero_point, vec![0]);
}

#[test]
fn per_channel_minmax_observer_with_shape() {
    // Exercises PerChannelMinMaxObserver::new and
    // PerChannelMinMaxObserver::observe_with_shape.
    let mut obs = PerChannelMinMaxObserver::new(2, 0).expect("valid channel count");
    // axis=0, num_channels=2, shape [2, 3]:
    //   channel 0 = [0, 1, 2]
    //   channel 1 = [10, 20, 30]
    PerChannelMinMaxObserver::observe_with_shape(
        &mut obs,
        &[0.0, 1.0, 2.0, 10.0, 20.0, 30.0],
        &[2, 3],
    )
    .expect("observe_with_shape");
    let qp = <PerChannelMinMaxObserver as Observer>::calculate_qparams(&obs, QuantDtype::Int8);
    assert_eq!(qp.scale.len(), 2);
    assert_eq!(qp.zero_point.len(), 2);
}

#[test]
fn per_channel_observer_shape_mismatch_errors() {
    let mut obs = PerChannelMinMaxObserver::new(3, 0).expect("valid channel count");
    let res = obs.observe_with_shape(&[1.0; 6], &[2, 3]);
    assert!(res.is_err(), "channel-count mismatch should error");
}

#[test]
fn per_channel_observer_zero_channels_errors_at_construction() {
    let err = PerChannelMinMaxObserver::new(0, 0).expect_err("zero channels must be invalid");
    match err {
        FerrotorchError::InvalidArgument { message } => {
            assert!(
                message.contains("at least one channel"),
                "unexpected error message: {message}"
            );
        }
        other => panic!("expected InvalidArgument for zero-channel observer, got {other:?}"),
    }
}

#[test]
fn per_channel_observer_flat_malformed_length_errors() {
    let mut obs = PerChannelMinMaxObserver::new(2, 0).expect("valid channel count");
    let err = obs
        .observe(&[1.0, 2.0, 3.0])
        .expect_err("flat observation with trailing partial channel must error");
    match err {
        FerrotorchError::InvalidArgument { message } => {
            assert!(
                message.contains("divisible by 2 channels"),
                "unexpected error message: {message}"
            );
        }
        other => panic!("expected InvalidArgument for malformed flat observation, got {other:?}"),
    }
}

#[test]
fn per_channel_observer_shape_product_mismatch_errors() {
    let mut obs = PerChannelMinMaxObserver::new(2, 0).expect("valid channel count");
    let err = obs
        .observe_with_shape(&[1.0; 5], &[2, 3])
        .expect_err("shape-aware observation data length must match shape product");
    match err {
        FerrotorchError::InvalidArgument { message } => {
            assert!(
                message.contains("expected 6 values"),
                "unexpected error message: {message}"
            );
        }
        other => panic!("expected InvalidArgument for shape product mismatch, got {other:?}"),
    }
}

#[test]
fn histogram_observer_basic() {
    let mut obs = HistogramObserver::new(64).expect("valid positive bin count");
    obs.observe(&[0.0, 0.5, 1.0]).expect("observe");
    let qp = obs.calculate_qparams(QuantDtype::Int8);
    assert_eq!(qp.scale.len(), 1);
    // Reset zeros bins.
    obs.reset();
    obs.observe(&[2.0, 3.0]).expect("observe");
    let qp2 = obs.calculate_qparams(QuantDtype::Int8);
    assert_eq!(qp2.scale.len(), 1);
}

#[test]
fn histogram_observer_zero_bins_errors_at_construction() {
    let err = HistogramObserver::new(0).expect_err("zero histogram bins must be invalid");
    match err {
        FerrotorchError::InvalidArgument { message } => {
            assert!(
                message.contains("at least one histogram bin"),
                "unexpected error message: {message}"
            );
        }
        other => panic!("expected InvalidArgument for zero-bin histogram, got {other:?}"),
    }
}

#[test]
fn histogram_observer_one_bin_counts_finite_values() {
    let mut obs = HistogramObserver::new(1).expect("one bin is valid like PyTorch");
    obs.observe(&[]).expect("empty observe");
    assert_eq!(obs.calculate_qparams(QuantDtype::Int8).scale, vec![1.0]);

    obs.observe(&[f32::NAN, 1.0, f32::INFINITY, -1.0])
        .expect("observe finite subset");
    obs.observe(&[0.0, 2.0]).expect("observe");

    let qp = obs.calculate_qparams(QuantDtype::Int8);
    assert_eq!(qp.scale.len(), 1);
    assert!(
        qp.scale[0] > 0.0,
        "one-bin observer should still produce positive qparams"
    );
}

#[test]
fn histogram_observer_qparams_use_histogram_search_basic3() {
    let mut obs = HistogramObserver::new(3).expect("valid positive bin count");
    obs.observe(&[2.0, 3.0, 4.0, 5.0]).expect("observe");
    obs.observe(&[5.0, 6.0, 7.0, 8.0]).expect("observe");

    let hist_qp = obs.calculate_qparams(QuantDtype::Int8);
    let mut minmax = MinMaxObserver::new();
    minmax.observe(&[2.0, 3.0, 4.0, 5.0]).expect("observe");
    minmax.observe(&[5.0, 6.0, 7.0, 8.0]).expect("observe");
    let minmax_qp = minmax.calculate_qparams(QuantDtype::Int8);

    // PyTorch HistogramObserver(bins=3, qint8, affine) clips [2, 8] to [2, 6],
    // giving scale 6/255 and zero_point -128. MinMaxObserver uses [0, 8].
    tolerance::assert_close_f32(
        &hist_qp.scale,
        &[0.023529412],
        tolerance::F32_REDUCTION,
        "histogram basic3 scale",
    );
    assert_eq!(hist_qp.zero_point, vec![-128]);
    assert!(
        hist_qp.scale[0] < minmax_qp.scale[0],
        "histogram qparams must use the histogram search, not raw min/max"
    );
}

#[test]
fn histogram_observer_qparams_clip_sparse_outlier_like_pytorch() {
    let mut data = vec![0.0; 256];
    data.extend(std::iter::repeat_n(1.0, 256));
    data.push(100.0);

    let mut obs = HistogramObserver::new(16).expect("valid positive bin count");
    obs.observe(&data).expect("observe");
    let qp = obs.calculate_qparams(QuantDtype::Int8);

    // PyTorch HistogramObserver(bins=16, qint8, affine) clips the final range
    // from [0, 100] to [0, 93.75], giving 93.75/255.
    tolerance::assert_close_f32(
        &qp.scale,
        &[0.36764705],
        tolerance::F32_REDUCTION,
        "histogram sparse-outlier scale",
    );
    assert_eq!(qp.zero_point, vec![-128]);
}

// ---------------------------------------------------------------------------
// FakeQuantize, QatLayer, QatModel, prepare_qat — QAT wrapper family.
// ---------------------------------------------------------------------------

#[test]
fn fake_quantize_forward_returns_dequantized_values_and_mask() {
    // Exercises FakeQuantize::new and FakeQuantize::forward.
    let mut fq: FakeQuantize = FakeQuantize::new(QuantDtype::Int8);
    let data = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    let (output, mask) = FakeQuantize::forward(&mut fq, &data);
    assert_eq!(output.len(), 5);
    assert_eq!(mask.len(), 5);
    // Output should be close to input (round-trip).
    for (i, (&o, &d)) in output.iter().zip(data.iter()).enumerate() {
        assert!((o - d).abs() < 0.1, "element {i}: out={o}, in={d}");
    }
}

#[test]
fn fake_quantize_observer_and_fake_quant_flags_are_independent() {
    // PyTorch 2.11 FakeQuantize(observer=MinMaxObserver, dtype=qint8,
    // qscheme=per_tensor_affine) for x=[0.25, 1.75, 3.25]:
    // obs=1/fake=1 -> out=[0.25490198, 1.74607849, 3.25], scale=0.012745098, zp=-128
    // obs=1/fake=0 -> identity output, same updated qparams
    // obs=0/fake=1 -> default qparams scale=1/zp=0, out=[0, 2, 3]
    // obs=0/fake=0 -> identity output, default qparams
    let input = vec![0.25, 1.75, 3.25];

    let mut observe_and_fake = FakeQuantize::new(QuantDtype::Int8);
    let (out, mask) = observe_and_fake.forward(&input);
    tolerance::assert_close_f32(
        &out,
        &[0.25490198, 1.7460785, 3.25],
        tolerance::F32_REDUCTION,
        "observer on, fake quant on output",
    );
    assert!(mask.iter().all(|&m| m.to_bits() == 1.0f32.to_bits()));
    let qp = observe_and_fake.qparams.as_ref().expect("qparams updated");
    tolerance::assert_close_f32(
        &qp.scale,
        &[0.012745098],
        tolerance::F32_REDUCTION,
        "observer on, fake quant on scale",
    );
    assert_eq!(qp.zero_point, vec![-128]);

    let mut observe_without_fake = FakeQuantize::new(QuantDtype::Int8);
    observe_without_fake.disable_fake_quant();
    let (out, mask) = observe_without_fake.forward(&input);
    assert_eq!(
        out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        input.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert!(mask.iter().all(|&m| m.to_bits() == 1.0f32.to_bits()));
    let qp = observe_without_fake
        .qparams
        .as_ref()
        .expect("qparams updated even when fake quant is disabled");
    tolerance::assert_close_f32(
        &qp.scale,
        &[0.012745098],
        tolerance::F32_REDUCTION,
        "observer on, fake quant off scale",
    );
    assert_eq!(qp.zero_point, vec![-128]);

    let mut fake_without_observe = FakeQuantize::new(QuantDtype::Int8);
    fake_without_observe.disable_observer();
    let (out, mask) = fake_without_observe.forward(&input);
    tolerance::assert_close_f32(
        &out,
        &[0.0, 2.0, 3.0],
        tolerance::F32_REDUCTION,
        "observer off, fake quant on output",
    );
    assert!(mask.iter().all(|&m| m.to_bits() == 1.0f32.to_bits()));
    let qp = fake_without_observe
        .qparams
        .as_ref()
        .expect("default qparams available");
    assert_eq!(qp.scale.len(), 1);
    assert_eq!(qp.scale[0].to_bits(), 1.0f32.to_bits());
    assert_eq!(qp.zero_point, vec![0]);

    let mut neither = FakeQuantize::new(QuantDtype::Int8);
    neither.disable_observer();
    neither.disable_fake_quant();
    let (out, mask) = neither.forward(&input);
    assert_eq!(
        out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        input.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert!(mask.iter().all(|&m| m.to_bits() == 1.0f32.to_bits()));
    let qp = neither.qparams.as_ref().expect("default qparams available");
    assert_eq!(qp.scale.len(), 1);
    assert_eq!(qp.scale[0].to_bits(), 1.0f32.to_bits());
    assert_eq!(qp.zero_point, vec![0]);
}

#[test]
fn fake_quantize_toggle_helpers_match_public_pytorch_surface() {
    let mut fq = FakeQuantize::new(QuantDtype::Int8);
    fq.disable_fake_quant();
    assert!(!fq.fake_quant_enabled);
    fq.enable_fake_quant(true);
    assert!(fq.fake_quant_enabled);
    fq.disable_observer();
    assert!(!fq.observer_enabled);
    fq.enable_observer(true);
    assert!(fq.observer_enabled);

    let qp = FakeQuantize::calculate_qparams(&fq);
    assert_eq!(qp.scale.len(), 1);
    assert_eq!(qp.scale[0].to_bits(), 1.0f32.to_bits());
    assert_eq!(qp.zero_point, vec![0]);
}

#[test]
fn qat_model_register_layer_and_fake_quantize_weights() {
    // QatLayer is constructed internally by QatModel::register_layer;
    // we exercise the public API and rely on the Debug/Clone derives to
    // confirm QatLayer is reachable.
    let mut model: QatModel = QatModel::new(QuantDtype::Int8);
    model.register_layer("fc1");
    assert!(model.layers.contains_key("fc1"));
    // Confirm the registered layer has both fq sub-modules — this
    // exercises QatLayer's struct layout.
    let layer: &QatLayer = model.layers.get("fc1").expect("fc1 layer");
    assert!(layer.weight_fq.fake_quant_enabled);
    assert!(layer.activation_fq.fake_quant_enabled);

    let weights = vec![0.1, 0.2, 0.3, 0.4];
    // Use type-qualified call so the coverage gate can match
    // QatModel::fake_quantize_weights.
    let (fq_w, originals) = QatModel::fake_quantize_weights(&mut model, "fc1", &weights)
        .expect("fake_quantize_weights");
    assert_eq!(originals, weights);
    assert_eq!(fq_w.len(), weights.len());

    // Same for QatModel::fake_quantize_activations.
    let (fq_a, _grad_mask) = QatModel::fake_quantize_activations(&mut model, "fc1", &[1.0, 2.0])
        .expect("fake_quantize_activations");
    assert_eq!(fq_a.len(), 2);
}

#[test]
fn prepare_qat_skips_bias_only_layers() {
    let names = &["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"];
    let model = prepare_qat(names, QuantDtype::Int8);
    assert!(model.layers.contains_key("fc1"));
    assert!(model.layers.contains_key("fc2"));
    assert_eq!(model.layers.len(), 2);
}

// ---------------------------------------------------------------------------
// cuda_rng module — fork/join semantics for reproducible GPU RNG state.
//
// Note: cuda_rng has a process-global current state, while fork snapshots are
// isolated per thread. Tests that set the global current state still serialize
// under cargo's parallel runner.
// ---------------------------------------------------------------------------

fn cuda_rng_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

#[test]
fn cuda_rng_get_state_set_state() {
    let _g = cuda_rng_test_lock();
    let initial = cuda_rng::get_state();
    cuda_rng::set_state(0xbeef_face);
    assert_eq!(cuda_rng::get_state(), 0xbeef_face);
    // Restore so other tests don't see a poisoned state.
    cuda_rng::set_state(initial);
}

#[test]
fn cuda_rng_fork_join_round_trip() {
    let _g = cuda_rng_test_lock();
    let initial = cuda_rng::get_state();
    cuda_rng::fork_rng(0x1234_5678);
    assert_eq!(cuda_rng::get_state(), 0x1234_5678);
    cuda_rng::join_rng();
    assert_eq!(cuda_rng::get_state(), initial);
}

#[test]
fn cuda_rng_nested_fork_join_round_trip() {
    let _g = cuda_rng_test_lock();
    let initial = cuda_rng::get_state();
    cuda_rng::set_state(0xaaaa_bbbb_cccc_dddd);

    cuda_rng::fork_rng(0x1111_2222_3333_4444);
    assert_eq!(cuda_rng::get_state(), 0x1111_2222_3333_4444);

    cuda_rng::fork_rng(0x5555_6666_7777_8888);
    assert_eq!(cuda_rng::get_state(), 0x5555_6666_7777_8888);

    cuda_rng::join_rng();
    assert_eq!(
        cuda_rng::get_state(),
        0x1111_2222_3333_4444,
        "nested fork_rng must restore the previous inner state first"
    );

    cuda_rng::join_rng();
    assert_eq!(
        cuda_rng::get_state(),
        0xaaaa_bbbb_cccc_dddd,
        "outer join_rng must restore the pre-fork state"
    );

    cuda_rng::set_state(initial);
}

#[test]
fn cuda_rng_overlapping_thread_forks_do_not_cross_pop() {
    let _g = cuda_rng_test_lock();
    let initial = cuda_rng::get_state();
    let outer_state = 0xaaaa_bbbb_cccc_dddd;
    let first_thread_seed = 0x1111_2222_3333_4444;
    let second_thread_seed = 0x5555_6666_7777_8888;
    let timeout = std::time::Duration::from_secs(5);

    cuda_rng::set_state(outer_state);

    let (a_entered_tx, a_entered_rx) = std::sync::mpsc::channel();
    let (b_entered_tx, b_entered_rx) = std::sync::mpsc::channel();
    let (a_joined_tx, a_joined_rx) = std::sync::mpsc::channel();
    let (a_state_tx, a_state_rx) = std::sync::mpsc::channel();
    let (b_state_tx, b_state_rx) = std::sync::mpsc::channel();

    let thread_a = std::thread::spawn(move || -> Result<(), String> {
        cuda_rng::fork_rng(first_thread_seed);
        if cuda_rng::get_state() != first_thread_seed {
            return Err("thread A did not install its fork seed".to_owned());
        }

        a_entered_tx.send(()).map_err(|err| err.to_string())?;
        b_entered_rx
            .recv_timeout(timeout)
            .map_err(|err| err.to_string())?;

        cuda_rng::join_rng();
        a_state_tx
            .send(cuda_rng::get_state())
            .map_err(|err| err.to_string())?;
        a_joined_tx.send(()).map_err(|err| err.to_string())?;
        Ok(())
    });

    let thread_b = std::thread::spawn(move || -> Result<(), String> {
        a_entered_rx
            .recv_timeout(timeout)
            .map_err(|err| err.to_string())?;

        cuda_rng::fork_rng(second_thread_seed);
        if cuda_rng::get_state() != second_thread_seed {
            return Err("thread B did not install its fork seed".to_owned());
        }

        b_entered_tx.send(()).map_err(|err| err.to_string())?;
        a_joined_rx
            .recv_timeout(timeout)
            .map_err(|err| err.to_string())?;

        cuda_rng::join_rng();
        b_state_tx
            .send(cuda_rng::get_state())
            .map_err(|err| err.to_string())?;
        Ok(())
    });

    let a_restored = a_state_rx
        .recv_timeout(timeout)
        .expect("thread A should report the state after its join");
    let b_restored = b_state_rx
        .recv_timeout(timeout)
        .expect("thread B should report the state after its join");

    let thread_a_result = thread_a.join().expect("thread A should not panic");
    let thread_b_result = thread_b.join().expect("thread B should not panic");
    cuda_rng::set_state(initial);

    thread_a_result.expect("thread A should complete fork/join orchestration");
    thread_b_result.expect("thread B should complete fork/join orchestration");
    assert_eq!(
        a_restored, outer_state,
        "thread A must restore its own saved state; a process-global stack pops thread B's frame here"
    );
    assert_eq!(
        b_restored, first_thread_seed,
        "thread B should restore the shared generator to the state it observed when it forked"
    );
}

#[test]
fn cuda_rng_next_seed_advances_state() {
    let _g = cuda_rng_test_lock();
    let s1 = cuda_rng::next_seed();
    let s2 = cuda_rng::next_seed();
    assert_ne!(s1, s2, "consecutive next_seed() calls must differ");
}

// ---------------------------------------------------------------------------
// GPU policy: quantize is CPU-domain, pruning is CUDA-domain.
// Quantize must still reject CUDA tensors instead of silently host-reading
// them. Pruning mirrors PyTorch's tensor-mask parametrization and therefore
// must build/apply masks on-device and preserve the autograd edge to the
// original parameter.
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use ferrotorch_core::Device;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the GPU conformance suite");
        });
    }

    fn upload_f32(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        make_cpu_f32(data, shape, false)
            .to(Device::Cuda(0))
            .expect("upload to cuda")
            .requires_grad_(requires_grad)
    }

    fn upload_f16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::f16> {
        let values: Vec<half::f16> = data.iter().copied().map(half::f16::from_f32).collect();
        Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), false)
            .expect("make f16 CPU tensor")
            .to(Device::Cuda(0))
            .expect("upload f16 to cuda")
            .requires_grad_(requires_grad)
    }

    fn upload_bf16(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<half::bf16> {
        let values: Vec<half::bf16> = data.iter().copied().map(half::bf16::from_f32).collect();
        Tensor::from_storage(TensorStorage::cpu(values), shape.to_vec(), false)
            .expect("make bf16 CPU tensor")
            .to(Device::Cuda(0))
            .expect("upload bf16 to cuda")
            .requires_grad_(requires_grad)
    }

    fn read_cuda_f32(t: &Tensor<f32>, label: &str) -> Vec<f32> {
        assert_eq!(t.device(), Device::Cuda(0), "{label}: tensor left CUDA");
        t.cpu()
            .unwrap_or_else(|e| panic!("{label}: copy to CPU: {e}"))
            .data()
            .unwrap_or_else(|e| panic!("{label}: CPU data: {e}"))
            .to_vec()
    }

    fn read_cuda_f16(t: &Tensor<half::f16>, label: &str) -> Vec<f32> {
        assert_eq!(t.device(), Device::Cuda(0), "{label}: tensor left CUDA");
        t.cpu()
            .unwrap_or_else(|e| panic!("{label}: copy to CPU: {e}"))
            .data()
            .unwrap_or_else(|e| panic!("{label}: CPU data: {e}"))
            .iter()
            .map(|v| v.to_f32())
            .collect()
    }

    fn read_cuda_bf16(t: &Tensor<half::bf16>, label: &str) -> Vec<f32> {
        assert_eq!(t.device(), Device::Cuda(0), "{label}: tensor left CUDA");
        t.cpu()
            .unwrap_or_else(|e| panic!("{label}: copy to CPU: {e}"))
            .data()
            .unwrap_or_else(|e| panic!("{label}: CPU data: {e}"))
            .iter()
            .map(|v| v.to_f32())
            .collect()
    }

    fn assert_bits_eq(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{label}: length mismatch (actual={}, expected={})",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: index {i} bit mismatch (actual={a:?}, expected={e:?})"
            );
        }
    }

    #[test]
    fn gpu_tensor_returns_error_for_quantize() {
        ensure_cuda_backend();
        let cpu = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let gpu = cpu.to(Device::Cuda(0)).expect("upload to cuda");
        // PyTorch parity: a CUDA tensor should not silently host-readback;
        // the call must surface a structured error.
        let res = quantize(&gpu, QuantScheme::PerTensor, QuantDtype::Int8);
        assert!(
            res.is_err(),
            "quantize on a GPU tensor must return Err (PyTorch parity), got Ok"
        );
    }

    #[test]
    fn gpu_magnitude_prune_stays_cuda_and_masks_backward() {
        ensure_cuda_backend();

        // Live torch 2.11.0+cu130 oracle:
        //   prune.l1_unstructured(m, "weight", amount=0.5)
        //   m.weight.device == cuda:0
        //   m.weight == [0., -4., 0., -3.]
        //   m.weight_orig.grad == [0., 20., 0., 40.]
        let x = upload_f32(&[1.0, -4.0, 2.0, -3.0], &[4], true);
        let pruned = magnitude_prune(&x, 0.5).expect("magnitude_prune cuda");

        assert_bits_eq(
            &read_cuda_f32(&pruned, "magnitude_prune output"),
            &[0.0, -4.0, 0.0, -3.0],
            "magnitude_prune output",
        );

        let coeffs = upload_f32(&[10.0, 20.0, 30.0, 40.0], &[4], false);
        let prod = ferrotorch_core::grad_fns::arithmetic::mul(&pruned, &coeffs)
            .expect("magnitude_prune grad probe mul");
        let loss = ferrotorch_core::grad_fns::reduction::sum(&prod)
            .expect("magnitude_prune grad probe sum");
        loss.backward().expect("magnitude_prune backward");

        let grad = x
            .grad()
            .expect("grad access")
            .expect("CUDA original leaf must receive masked gradient");
        assert_bits_eq(
            &read_cuda_f32(&grad, "magnitude_prune grad"),
            &[0.0, 20.0, 0.0, 40.0],
            "magnitude_prune grad",
        );
    }

    #[test]
    fn gpu_apply_2_4_mask_stays_cuda_and_masks_backward() {
        ensure_cuda_backend();

        // Live torch 2.11.0+cu130 tensor-op oracle for sparse blocks:
        //   x = torch.tensor([[1., -4., 2., -3.],
        //                     [0.5, 0.1, 0.9, 0.8]], device="cuda")
        //   scores = x.view(-1, 4) * x.view(-1, 4)
        //   idx = torch.topk(scores, k=2, dim=1, largest=False).indices
        //   mask = torch.ones_like(scores).scatter(1, idx, 0).view_as(x)
        //   y = x * mask
        //   y.device == cuda:0
        //   y == [0., -4., 0., -3., 0., 0., 0.9, 0.8]
        //   x.grad == [0., 20., 0., 40., 0., 0., 70., 80.]
        let x = upload_f32(&[1.0, -4.0, 2.0, -3.0, 0.5, 0.1, 0.9, 0.8], &[2, 4], true);
        let masked = apply_2_4_mask(&x).expect("apply_2_4_mask cuda");

        assert_bits_eq(
            &read_cuda_f32(&masked, "apply_2_4_mask output"),
            &[0.0, -4.0, 0.0, -3.0, 0.0, 0.0, 0.9, 0.8],
            "apply_2_4_mask output",
        );

        let coeffs = upload_f32(
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            &[2, 4],
            false,
        );
        let prod = ferrotorch_core::grad_fns::arithmetic::mul(&masked, &coeffs)
            .expect("2:4 grad probe mul");
        let loss = ferrotorch_core::grad_fns::reduction::sum(&prod).expect("2:4 grad probe sum");
        loss.backward().expect("2:4 backward");

        let grad = x
            .grad()
            .expect("grad access")
            .expect("CUDA original leaf must receive 2:4 masked gradient");
        assert_bits_eq(
            &read_cuda_f32(&grad, "apply_2_4_mask grad"),
            &[0.0, 20.0, 0.0, 40.0, 0.0, 0.0, 70.0, 80.0],
            "apply_2_4_mask grad",
        );
    }

    #[test]
    fn gpu_apply_2_4_mask_rejects_public_sparsifier_invalid_shapes() {
        ensure_cuda_backend();

        for (label, data, shape) in [
            ("rank1", vec![1.0, -4.0, 2.0, -3.0], vec![4]),
            (
                "bad_width_2x6",
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                vec![2, 6],
            ),
            ("empty_rows", Vec::new(), vec![0, 4]),
            ("empty_cols", Vec::new(), vec![2, 0]),
            ("rank3", (0..16).map(|v| v as f64).collect(), vec![2, 2, 4]),
        ] {
            let t = upload_f32(&data, &shape, false);
            let res = apply_2_4_mask(&t);
            assert!(
                matches!(
                    res,
                    Err(ferrotorch_core::FerrotorchError::InvalidArgument { .. })
                ),
                "{label}: CUDA apply_2_4_mask must reject torch-invalid shape {shape:?}, got {res:?}"
            );
        }
    }

    #[test]
    fn gpu_pruning_edge_cases_match_torch_cuda_masks() {
        ensure_cuda_backend();

        // Live torch 2.11.0+cu130 CUDA oracle:
        //   torch.nn.utils.prune.l1_unstructured([1,1,1,1], amount=0.5)
        //   selects indices {0,1} on CUDA (CPU has a different tie set).
        let tied = upload_f32(&[1.0, 1.0, 1.0, 1.0], &[4], false);
        assert_bits_eq(
            &read_cuda_f32(
                &magnitude_prune(&tied, 0.5).expect("magnitude_prune tied cuda"),
                "magnitude_prune tied",
            ),
            &[0.0, 0.0, 1.0, 1.0],
            "magnitude_prune tied",
        );

        // PyTorch applies `orig * mask`, so a pruned -0.0 stays -0.0.
        let signed_zero = upload_f32(&[0.0, -0.0, 1.0, -1.0], &[4], false);
        assert_bits_eq(
            &read_cuda_f32(
                &magnitude_prune(&signed_zero, 0.5).expect("magnitude_prune signed zero cuda"),
                "magnitude_prune signed zero",
            ),
            &[0.0, -0.0, 1.0, -1.0],
            "magnitude_prune signed zero",
        );

        // NaN ranks last for largest=False topk, so finite small magnitudes
        // are pruned first and NaN is kept; multiplying by a 1-mask preserves
        // NaN.
        let nan_case = upload_f32(&[f64::NAN, 1.0, 2.0, 3.0], &[4], false);
        let nan_out = read_cuda_f32(
            &magnitude_prune(&nan_case, 0.5).expect("magnitude_prune nan cuda"),
            "magnitude_prune nan",
        );
        assert!(
            nan_out[0].is_nan(),
            "magnitude_prune nan: first slot should stay NaN, got {nan_out:?}"
        );
        assert_bits_eq(&nan_out[1..], &[0.0, 0.0, 3.0], "magnitude_prune nan tail");

        // Same selected-set contract for 2:4 blocks: CUDA ties prune slots
        // {0,1}; NaN is kept while finite smaller scores are zeroed.
        let block = upload_f32(
            &[1.0, 1.0, 1.0, 1.0, f64::NAN, 1.0, 2.0, 3.0],
            &[2, 4],
            false,
        );
        let block_out = read_cuda_f32(
            &apply_2_4_mask(&block).expect("apply_2_4_mask edge cuda"),
            "apply_2_4_mask edge",
        );
        assert_bits_eq(&block_out[..4], &[0.0, 0.0, 1.0, 1.0], "2:4 tied block");
        assert!(
            block_out[4].is_nan(),
            "2:4 nan block: NaN slot should be kept, got {block_out:?}"
        );
        assert_bits_eq(&block_out[5..], &[0.0, 0.0, 3.0], "2:4 nan block tail");
    }

    #[test]
    fn gpu_pruning_supports_half_and_bfloat16_without_host_fallback() {
        ensure_cuda_backend();

        let x_f16 = upload_f16(&[1.0, -4.0, 2.0, -3.0], &[4], true);
        let pruned_f16 = magnitude_prune(&x_f16, 0.5).expect("magnitude_prune f16 cuda");
        assert_eq!(
            read_cuda_f16(&pruned_f16, "magnitude_prune f16"),
            vec![0.0, -4.0, 0.0, -3.0]
        );

        let coeffs_f16 = upload_f16(&[10.0, 20.0, 30.0, 40.0], &[4], false);
        let loss_f16 = ferrotorch_core::grad_fns::reduction::sum(
            &ferrotorch_core::grad_fns::arithmetic::mul(&pruned_f16, &coeffs_f16)
                .expect("magnitude_prune f16 grad probe mul"),
        )
        .expect("magnitude_prune f16 grad probe sum");
        loss_f16.backward().expect("magnitude_prune f16 backward");
        assert_eq!(
            read_cuda_f16(
                &x_f16
                    .grad()
                    .expect("f16 grad access")
                    .expect("f16 original leaf grad"),
                "magnitude_prune f16 grad"
            ),
            vec![0.0, 20.0, 0.0, 40.0]
        );

        let x_bf16 = upload_bf16(&[1.0, -4.0, 2.0, -3.0], &[4], true);
        let pruned_bf16 = magnitude_prune(&x_bf16, 0.5).expect("magnitude_prune bf16 cuda");
        assert_eq!(
            read_cuda_bf16(&pruned_bf16, "magnitude_prune bf16"),
            vec![0.0, -4.0, 0.0, -3.0]
        );

        let coeffs_bf16 = upload_bf16(&[10.0, 20.0, 30.0, 40.0], &[4], false);
        let loss_bf16 = ferrotorch_core::grad_fns::reduction::sum(
            &ferrotorch_core::grad_fns::arithmetic::mul(&pruned_bf16, &coeffs_bf16)
                .expect("magnitude_prune bf16 grad probe mul"),
        )
        .expect("magnitude_prune bf16 grad probe sum");
        loss_bf16.backward().expect("magnitude_prune bf16 backward");
        assert_eq!(
            read_cuda_bf16(
                &x_bf16
                    .grad()
                    .expect("bf16 grad access")
                    .expect("bf16 original leaf grad"),
                "magnitude_prune bf16 grad"
            ),
            vec![0.0, 20.0, 0.0, 40.0]
        );

        let block_f16 = upload_f16(
            &[1.0, -4.0, 2.0, -3.0, 0.5, 0.25, 0.75, 1.0],
            &[2, 4],
            false,
        );
        let masked_f16 = apply_2_4_mask(&block_f16).expect("apply_2_4_mask f16 cuda");
        assert_eq!(
            read_cuda_f16(&masked_f16, "apply_2_4_mask f16"),
            vec![0.0, -4.0, 0.0, -3.0, 0.0, 0.0, 0.75, 1.0]
        );

        let block_bf16 = upload_bf16(
            &[1.0, -4.0, 2.0, -3.0, 0.5, 0.25, 0.75, 1.0],
            &[2, 4],
            false,
        );
        let masked_bf16 = apply_2_4_mask(&block_bf16).expect("apply_2_4_mask bf16 cuda");
        assert_eq!(
            read_cuda_bf16(&masked_bf16, "apply_2_4_mask bf16"),
            vec![0.0, -4.0, 0.0, -3.0, 0.0, 0.0, 0.75, 1.0]
        );
    }
}
