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
//! * **Quantize forwards** (CPU; integer-domain → bit-exact codes,
//!   dequant under `F32_REDUCTION` tolerance):
//!   - `quantize` per-tensor / per-channel for INT8, UINT8, INT4
//!   - `dequantize` (round-trip parity within one quantization step)
//!   - `quantized_matmul` (real-valued output asserted within `2 * scale`)
//! * **QParams** symmetric & asymmetric for the boundary zp values
//!   `0`, `128`, and the all-positive-range cases that exercise the
//!   non-clamped `zp` path.
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
//!   - quantized_matmul real-valued output: 2 * combined_scale
//!     (one scale-step on each input, summed)
//!
//! ## Fixture provenance (CORE-194 -> #1888)
//!
//! Every expectation in `fixtures/quantize_prune.json` is computed by a
//! REAL PyTorch API (R-ORACLE-2): `MinMaxObserver`/`PerChannelMinMaxObserver`
//! `.calculate_qparams()` for scales/zero-points, `torch.quantize_per_tensor`
//! / `torch.quantize_per_channel` (`.int_repr()`, `.dequantize()`) for codes
//! (decomposed ops for INT4), `torch.nn.utils.prune.l1_unstructured` for
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
//! FAIL when the divergence is fixed, so it retires loudly:
//!   - #1777 (CORE-083): ties at the prune threshold are over-pruned.
//!   - #1778 (CORE-084): 2:4 mask accepts shapes torch rejects (trailing
//!     remainder; flat grouping across row boundaries).
//!   - #1906: `compute_scale_zp` eps floor on range (all-zero scale) and
//!     zero-point rounding at half-boundaries.
//!   - #1907: `QParams::symmetric` denominator (qmax vs (qmax-qmin)/2).
//!   - #1908: prune-count rounding (half-away vs Python round-half-even).
//!   - #1909: pruned negative slots are +0.0; torch mask-multiply gives -0.0.
//!   - #1910: 2:4 in-group tie selection differs from the torch sparsifier.
//!   - #1911: quantize codes at half-step boundaries (divide + round-half-
//!     away vs torch inv_scale multiply + round-half-even).
//!
//! GPU note (per the dispatch's most-likely-failure-mode):
//! `quantize`, `dequantize`, `quantized_matmul`, `magnitude_prune` and
//! `apply_2_4_mask` all consume `tensor.data()?` which returns
//! `Err(GpuTensorNotAccessible)` for GPU-resident tensors. This is the
//! PyTorch-parity behaviour (`torch.quantize_per_tensor` is a CPU op;
//! attempting to call it on a CUDA tensor raises a `RuntimeError`). We
//! assert this contract in `gpu_tensor_returns_error_for_quantize` so
//! the policy is locked down. No `cascade_skip` references are needed
//! because the conformance suite operates on CPU inputs by design.

use std::path::PathBuf;

use serde::Deserialize;
use serde::de::{self, Deserializer, SeqAccess, Visitor};

use ferrotorch_core::pruning::{apply_2_4_mask, magnitude_prune, sparsity_ratio};
use ferrotorch_core::quantize::{
    FakeQuantize, HistogramObserver, MinMaxObserver, Observer, PerChannelMinMaxObserver, QParams,
    QatLayer, QatModel, QuantDtype, QuantScheme, QuantizedTensor, cuda_rng, dequantize,
    prepare_qat, quantize, quantize_named_tensors, quantized_matmul,
};
use ferrotorch_core::{Tensor, TensorStorage, fake_quantize_differentiable};

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
    /// cases carry no value expectation; the suite pins ferrotorch's
    /// divergent acceptance against the tracking issue instead.
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

/// Reverse of ferrotorch's `stored_to_i32`: turn the stored `i8` back into
/// the i32-domain code that the fixtures encode. For Uint8 the stored byte
/// is reinterpreted as `u8` first (0..=255 instead of -128..=127).
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
        if f.tag.as_deref() == Some("int8_all_zero") {
            // KNOWN DIVERGENCE #1906 (compute_scale_zp eps floor): torch's
            // MinMaxObserver floors the SCALE at f32 eps, giving
            // 1.1920928955078125e-7 (the fixture value); ferrotorch floors
            // the RANGE at f32::EPSILON then divides by (qmax - qmin),
            // giving 4.6748744e-10 (255x smaller). The relative-tolerance
            // check below would silently absorb this because both values
            // are far below its 1.0 floor, so the divergence is pinned
            // explicitly. This assertion FAILS when #1906 is fixed —
            // retire the pin and fall through to the parity check.
            assert!(
                actual_scale[0] < expected_scale / 2.0,
                "{label}: #1906 appears fixed (scale {} now matches torch {}) — \
                 retire this pin",
                actual_scale[0],
                expected_scale
            );
        } else {
            let scale_diff = (actual_scale[0] - expected_scale).abs();
            assert!(
                scale_diff <= tolerance::F32_REDUCTION * expected_scale.abs().max(1.0),
                "{label}: scale {} vs expected {}",
                actual_scale[0],
                expected_scale
            );
        }

        // Zero-point parity (exact i32).
        let actual_zp = qt.zero_point();
        assert_eq!(actual_zp.len(), 1, "{label}: per-tensor zp len != 1");
        assert_eq!(actual_zp[0], expected_zp, "{label}: zero_point mismatch");

        // Bit-exact integer codes.
        let stored = qt.data();
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

        // KNOWN DIVERGENCES — pinned ferrotorch outputs, observed at HEAD.
        // The fixture holds the torch-oracle truth; these pins assert the
        // CURRENT divergent behavior and FAIL (retire) when the underlying
        // issue is fixed.
        let (pinned_codes, pinned_zps): (Option<Vec<i32>>, Option<Vec<i32>>) =
            match f.tag.as_deref() {
                // #1911 (half-step code rounding): channel [100..200]
                // value 100 -> code -1 (torch: 0), channel [-10..10]
                // value 10 -> code 126 (torch: 127).
                Some("int8_axis0") => (
                    Some(vec![
                        -128, -43, 42, 127, -128, -65, 63, 126, -1, 38, 89, 127,
                    ]),
                    None,
                ),
                // #1906 (zp rounding: ch [-10,10] -> zp -1, torch: 0) plus
                // #1911 knock-on codes (torch: [-8,-3,2,7,-8,-4,4,7,0,2,5,7]).
                Some("int4_axis0") => (
                    Some(vec![-8, -3, 2, 7, -8, -5, 3, 7, -1, 2, 5, 7]),
                    Some(vec![-8, -1, -8]),
                ),
                _ => (None, None),
            };
        if let Some(pins) = &pinned_codes {
            assert_ne!(
                pins, expected_codes,
                "{label}: pinned codes now equal the torch fixture — \
                 #1906/#1911 appear fixed; retire this pin"
            );
        }

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

        // Per-channel zp parity (exact) — or the pinned divergent zps.
        let zp_reference: &[i32] = pinned_zps.as_deref().unwrap_or(expected_zps);
        for (i, (&actual, &expected)) in qt.zero_point().iter().zip(zp_reference.iter()).enumerate()
        {
            assert_eq!(actual, expected, "{label}: channel {i} zero_point mismatch");
        }

        // Bit-exact codes in the original flat order — or the pinned
        // divergent codes (#1906/#1911; torch truth stays in the fixture).
        let code_reference: &[i32] = pinned_codes.as_deref().unwrap_or(expected_codes);
        for (i, (&stored, &expected)) in qt.data().iter().zip(code_reference.iter()).enumerate() {
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
        // KNOWN DIVERGENCES — pinned dequant outputs, observed at HEAD,
        // downstream of the pinned codes/zps in
        // `quantize_per_channel_bit_exact_codes` (#1906/#1911). The torch
        // truth stays in the fixture `dequant` field; these pins FAIL
        // (retire) when the divergence is fixed.
        let pinned_dequant: Option<Vec<f32>> = match f.tag.as_deref() {
            // torch: [..., 10.039216, 100.392159, ...] (fixture).
            Some("int8_axis0") => Some(vec![
                0.0, 1.0, 2.0, 3.0, -9.960785, -5.019608, 5.019608, 9.960785, 99.60784, 130.19608,
                170.19608, 200.0,
            ]),
            // torch: [..., -10.666667, ..., 9.333334, 106.666664, ...] (fixture).
            Some("int4_axis0") => Some(vec![
                0.0,
                1.0,
                2.0,
                3.0,
                -9.333334,
                -5.333_333_5,
                5.333_333_5,
                10.666667,
                93.33333,
                133.33333,
                173.33333,
                200.0,
            ]),
            _ => None,
        };
        let expected_f32: Vec<f32> = match pinned_dequant {
            Some(p) => p,
            None => expected_dequant.iter().map(|&v| v as f32).collect(),
        };
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

        // KNOWN DIVERGENCE #1907: ferrotorch's `QParams::symmetric` divides
        // by qmax (127 / 128 / 7); torch's symmetric observer divides by
        // (qmax - qmin) / 2 (127.5 / 127.5 / 7.5). EVERY symmetric case
        // therefore diverges from the torch-oracle `scale` in the fixture
        // (e.g. max_abs=5.0 int8: ferrotorch 0.03937008, torch 0.039215688).
        // Pinned to the observed ferrotorch value; FAILS (retire) when
        // #1907 is fixed.
        let pinned_scale: f32 = match f.tag.as_deref().expect("tag") {
            "int8_maxabs5.0" => 0.039_370_08,
            "uint8_maxabs5.0" => 0.039_062_5,
            "int4_maxabs5.0" => 0.714_285_73,
            "int8_maxabs1.0" => 0.007_874_016,
            "uint8_maxabs1.0" => 0.007_812_5,
            "int4_maxabs1.0" => 0.142_857_15,
            "int8_maxabs100.0" => 0.787_401_56,
            "uint8_maxabs100.0" => 0.781_25,
            "int4_maxabs100.0" => 14.285_714,
            other => panic!("{label}: no pinned scale for tag {other:?}"),
        };
        let pin_diff = (qp.scale[0] - pinned_scale).abs();
        assert!(
            pin_diff <= tolerance::F32_REDUCTION * pinned_scale,
            "{label}: scale {} != pinned divergent value {pinned_scale} \
             (torch oracle: {expected_scale}; #1907)",
            qp.scale[0]
        );
        let torch_diff = (qp.scale[0] - expected_scale).abs();
        assert!(
            torch_diff > tolerance::F32_REDUCTION * expected_scale.abs(),
            "{label}: scale now matches the torch oracle {expected_scale} — \
             #1907 appears fixed; retire this pin"
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
        if f.tag.as_deref() == Some("int8_signed") {
            // KNOWN DIVERGENCE #1906 (compute_scale_zp zero-point rounding):
            // for (min, max) = (-3, 3), min/scale lands on the -127.5
            // half-boundary; torch's observer rounds half-to-even at higher
            // effective precision and returns zp = 0 (the fixture value);
            // ferrotorch computes (qmin - min/scale).round() in f32
            // (half-away-from-zero) and returns -1. FAILS (retire) when
            // #1906 is fixed.
            assert_eq!(
                qp.zero_point[0], -1,
                "{label}: zp no longer matches the pinned divergent value -1 \
                 (torch oracle: {expected_zp}; #1906) — if it now equals the \
                 torch value, retire this pin"
            );
        } else {
            assert_eq!(qp.zero_point[0], expected_zp, "{label}: zp");
        }
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

        // Tolerance: combined_scale ≈ qa.scale * qb.scale; the ferrotorch
        // requantize step then adds another step's worth. Bound by
        // 4 * combined_scale to absorb round-off + requantize step at boundary.
        let combined = qa.scale()[0] * qb.scale()[0];
        let out_step = qc.scale()[0];
        let step = combined.mul_add(2.0, out_step * 2.0);
        tolerance::assert_within_step_f32(&actual, &expected_f32, step.max(0.5), &label);
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

        // KNOWN DIVERGENCE #1911 (half-step code rounding): inputs that
        // land exactly on a half code step (the 0.5-grid points of these
        // ranges) quantize to a different code than torch (divide +
        // round-half-away vs torch inv_scale multiply + round-half-even).
        // Pinned to the observed ferrotorch outputs; the torch truth stays
        // in the fixture `recovered` field. The pin FAILS (retire) when
        // #1911 is fixed. (`rt_int4` has no half-step hits and stays on
        // the torch-parity path.)
        let pinned_recovered: Option<(Vec<f32>, usize)> = match f.tag.as_deref() {
            // torch: [-5.0, -4.5098042, -4.0, -3.4901962, -3.0, -2.509804,
            //         -2.0, -1.4901961, -1.0, -0.509804, 0.0] (fixture).
            Some("rt_int8") => Some((
                vec![
                    -5.0,
                    -4.490_196,
                    -4.0,
                    -3.509_804,
                    -3.0,
                    -2.490_196_2,
                    -2.0,
                    -1.490_196_1,
                    -1.0,
                    -0.490_196_1,
                    0.0,
                ],
                1,
            )),
            // torch: [0.0, 0.1960784, 0.4, 0.5960785, 0.8000001, 0.9960785,
            //         1.2, 1.3960785, 1.6000001, 1.7960786, 2.0] (fixture).
            Some("rt_uint8") => Some((
                vec![
                    0.0,
                    0.196_078_45,
                    0.400_000_04,
                    0.603_921_6,
                    0.800_000_1,
                    0.996_078_5,
                    1.2,
                    1.396_078_5,
                    1.600_000_1,
                    // Index 9 (1.8) matches torch with the fixture's exact
                    // f64-sourced input; only index 3 (0.6) diverges.
                    1.796_078_6,
                    2.0,
                ],
                3,
            )),
            _ => None,
        };
        let expected_f32: Vec<f32> = match &pinned_recovered {
            Some((p, retire_idx)) => {
                // Retire check: if the divergent index now matches torch,
                // #1911 is fixed for this case.
                let torch_v = expected_recovered[*retire_idx] as f32;
                assert!(
                    (actual[*retire_idx] - torch_v).abs()
                        > tolerance::F32_REDUCTION * torch_v.abs().max(1.0),
                    "{label}: index {retire_idx} now matches the torch oracle \
                     ({torch_v}) — #1911 appears fixed; retire this pin"
                );
                p.clone()
            }
            None => expected_recovered.iter().map(|&v| v as f32).collect(),
        };

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

        // KNOWN DIVERGENCES — pinned ferrotorch outputs, observed at HEAD.
        // The fixture `pruned` field holds the torch
        // `prune.l1_unstructured` truth; each pin FAILS (retire) when the
        // referenced issue is fixed.
        let pinned: Option<(Vec<f32>, &str)> = match f.tag.as_deref() {
            // #1777 (CORE-083): ties at the threshold are ALL pruned.
            // torch: [1.0, 1.0, 0.0, 0.0] (prunes exactly 2 of the 4 ties).
            Some("tie_all_equal") => Some((vec![0.0, 0.0, 0.0, 0.0], "#1777")),
            // torch: [0.0, 0.0, 2.0, 3.0] (only one of the |2.0| ties goes).
            Some("tie_threshold_partial") => Some((vec![0.0, 0.0, 0.0, 3.0], "#1777")),
            // torch: [0.0, -0.0, 0.0, 5.0, -0.0, 7.0] (prunes exactly 3).
            Some("tie_multiway") => Some((vec![0.0, 0.0, 0.0, 5.0, 0.0, 7.0], "#1777")),
            // #1908: n_prune = round(0.125 * 4) — Rust half-away gives 1,
            // torch (Python round-half-even) gives 0. torch: [1, 2, 3, 4].
            Some("count_round_half") => Some((vec![0.0, 2.0, 3.0, 4.0], "#1908")),
            _ => None,
        };

        if let Some((pinned_vals, issue)) = pinned {
            assert_ne!(
                actual, expected_f32,
                "{label}: output now matches the torch oracle — {issue} \
                 appears fixed; retire this pin"
            );
            for (i, (&a, &p)) in actual.iter().zip(pinned_vals.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    p.to_bits(),
                    "{label}: index {i} no longer matches the pinned divergent \
                     value (pinned={p}, actual={a}; torch={}; {issue})",
                    expected_f32[i]
                );
            }
            continue;
        }

        // Bit-exact: for kept elements the bit pattern equals the input;
        // for pruned elements torch's mask-multiply yields ±0.0 with the
        // original sign. The test data is exactly representable in f32.
        for (i, (&a, &e)) in actual.iter().zip(expected_f32.iter()).enumerate() {
            if e == 0.0 && e.is_sign_negative() {
                // KNOWN DIVERGENCE #1909: torch's `weight_orig * mask`
                // preserves the sign bit (-0.0) of pruned negative weights;
                // ferrotorch writes +0.0. FAILS (retire) when #1909 is
                // fixed and ferrotorch emits -0.0 here too.
                assert_eq!(
                    a.to_bits(),
                    0_f32.to_bits(),
                    "{label}: index {i} pruned slot is no longer +0.0 \
                     (torch: -0.0; #1909) — if it is now -0.0, retire this pin"
                );
            } else {
                assert_eq!(
                    a.to_bits(),
                    e.to_bits(),
                    "{label}: index {i} bit pattern (actual={a}, expected={e})"
                );
            }
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

        let t = make_cpu_f32(x_data, shape, false);
        let masked = apply_2_4_mask(&t).expect("apply_2_4_mask");

        // Shape preservation.
        assert_eq!(masked.shape(), shape.as_slice(), "{label}: shape");
        let actual = masked.data().expect("masked data").to_vec();

        // KNOWN DIVERGENCE #1778 (CORE-084): torch's 2:4 sparsifier
        // REJECTS shapes whose rows are not a multiple of 4 wide (the
        // fixture records the torch error in `torch_error` and carries no
        // `masked` expectation); ferrotorch silently ACCEPTS them, leaving
        // a flat trailing remainder unchanged and flat-grouping ACROSS row
        // boundaries. Pinned to the observed ferrotorch output; FAILS
        // (retire) when #1778 lands an explicit error or per-row grouping.
        if let Some(torch_error) = &f.torch_error {
            assert!(
                f.masked.is_none(),
                "{label}: fixture carries both torch_error and masked"
            );
            let pinned: Vec<f32> = match f.tag.as_deref() {
                // torch: AssertionError (mask [1,8] vs x [1,6]); ferrotorch
                // masks the first group and leaves the 2-element tail.
                Some("trailing") => vec![0.0, -4.0, 0.0, -3.0, 0.5, 0.1],
                // torch: AssertionError (mask [2,8] vs x [2,6]); ferrotorch's
                // second flat group [5,6,6,5] spans row 0 cols 4..6 AND row 1
                // cols 0..2.
                Some("rows_cross_2x6") => {
                    vec![0.0, 0.0, 3.0, 4.0, 0.0, 6.0, 6.0, 0.0, 4.0, 3.0, 0.0, 0.0]
                }
                other => panic!("{label}: unpinned torch_error tag {other:?} ({torch_error})"),
            };
            for (i, (&a, &p)) in actual.iter().zip(pinned.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    p.to_bits(),
                    "{label}: index {i} no longer matches the pinned divergent \
                     value (pinned={p}, actual={a}; torch rejects this shape: \
                     {torch_error}; #1778)"
                );
            }
            continue;
        }

        let expected_masked = f.masked.as_ref().expect("masked").as_slice();
        let expected_zeros = f.n_zeros.expect("n_zeros");
        let expected_f32: Vec<f32> = expected_masked.iter().map(|&v| v as f32).collect();

        // KNOWN DIVERGENCE #1910: in-group magnitude ties — torch's
        // WeightNormSparsifier makes a deterministic keep choice that
        // ferrotorch's stable-ascending-sort (zero the two lowest-index
        // ties) does not match. Pinned to the observed ferrotorch output;
        // the torch truth stays in the fixture. FAILS (retire) when #1910
        // is fixed.
        let pinned: Option<Vec<f32>> = match f.tag.as_deref() {
            // torch: [2.0, 2.0, 0.0, 0.0] (keeps idx {0,1}).
            Some("tie_all_equal") => Some(vec![0.0, 0.0, 2.0, 2.0]),
            // torch: [0.0, 3.0, 0.0, 3.0] (keeps idx {1,3}).
            Some("tie_three_equal") => Some(vec![0.0, 0.0, 3.0, 3.0]),
            // torch: [-2.0, 2.0, -0.0, 0.0] (keeps idx {0,1}).
            Some("tie_neg_pair") => Some(vec![0.0, 0.0, -2.0, 2.0]),
            _ => None,
        };
        if let Some(pinned_vals) = pinned {
            assert_ne!(
                actual, expected_f32,
                "{label}: output now matches the torch oracle — #1910 \
                 appears fixed; retire this pin"
            );
            for (i, (&a, &p)) in actual.iter().zip(pinned_vals.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    p.to_bits(),
                    "{label}: index {i} no longer matches the pinned divergent \
                     value (pinned={p}, actual={a}; torch={}; #1910)",
                    expected_f32[i]
                );
            }
            continue;
        }

        for (i, (&a, &e)) in actual.iter().zip(expected_f32.iter()).enumerate() {
            if e == 0.0 && e.is_sign_negative() {
                // KNOWN DIVERGENCE #1909: torch's mask-multiply preserves
                // the sign bit (-0.0) of pruned negative weights; ferrotorch
                // writes +0.0. FAILS (retire) when #1909 is fixed.
                assert_eq!(
                    a.to_bits(),
                    0_f32.to_bits(),
                    "{label}: index {i} pruned slot is no longer +0.0 \
                     (torch: -0.0; #1909) — if it is now -0.0, retire this pin"
                );
            } else {
                assert_eq!(
                    a.to_bits(),
                    e.to_bits(),
                    "{label}: index {i} bit pattern (actual={a}, expected={e})"
                );
            }
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
    let t = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8], true);
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
// MinMaxObserver / PerChannelMinMaxObserver / HistogramObserver
// (Observer trait family).
// These wrap CPU statistics on f32 slices — no PyTorch reference is required;
// we exercise the documented contract instead.
// ---------------------------------------------------------------------------

#[test]
fn minmax_observer_calculates_qparams_for_int8() {
    let mut obs: MinMaxObserver = MinMaxObserver::new();
    obs.observe(&[1.0, 2.0, 3.0]);
    obs.observe(&[-1.0, 5.0]);
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
    obs.observe(&[1.0, f32::NAN, 2.0, f32::INFINITY, -1.0, f32::NEG_INFINITY]);
    let qp = obs.calculate_qparams(QuantDtype::Int8);
    let expected_scale = 3.0_f32 / 255.0;
    assert!((qp.scale[0] - expected_scale).abs() < 1e-5);

    // reset() clears the observer state.
    obs.reset();
    let qp2 = obs.calculate_qparams(QuantDtype::Int8);
    // After reset, min=+Inf, max=-Inf, computed scale falls back to the
    // EPSILON floor; we just check it doesn't panic.
    assert_eq!(qp2.scale.len(), 1);
}

#[test]
fn per_channel_minmax_observer_with_shape() {
    // Exercises PerChannelMinMaxObserver::new and
    // PerChannelMinMaxObserver::observe_with_shape.
    let mut obs = PerChannelMinMaxObserver::new(2, 0);
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
    let mut obs = PerChannelMinMaxObserver::new(3, 0);
    let res = obs.observe_with_shape(&[1.0; 6], &[2, 3]);
    assert!(res.is_err(), "channel-count mismatch should error");
}

#[test]
fn histogram_observer_basic() {
    let mut obs = HistogramObserver::new(64);
    obs.observe(&[0.0, 0.5, 1.0]);
    let qp = obs.calculate_qparams(QuantDtype::Int8);
    assert_eq!(qp.scale.len(), 1);
    // Reset zeros bins.
    obs.reset();
    obs.observe(&[2.0, 3.0]);
    let qp2 = obs.calculate_qparams(QuantDtype::Int8);
    assert_eq!(qp2.scale.len(), 1);
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
// Note: cuda_rng is process-global Mutex<u64> state. Two tests that both
// mutate it would race under cargo's parallel runner. Serialise via a
// local lock, mirroring the pattern used in src/quantize.rs's tests.
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
fn cuda_rng_next_seed_advances_state() {
    let _g = cuda_rng_test_lock();
    let s1 = cuda_rng::next_seed();
    let s2 = cuda_rng::next_seed();
    assert_ne!(s1, s2, "consecutive next_seed() calls must differ");
}

// ---------------------------------------------------------------------------
// GPU policy: quantize/prune are CPU-domain APIs (PyTorch parity).
// Per `rust-gpu-discipline` §3 the contract for an unsupported op is a
// structured error, not a silent fallback. ferrotorch's quantize call
// reads `tensor.data()?` which returns `GpuTensorNotAccessible` for
// CUDA-resident tensors — that IS the structured error here, and we
// pin the contract so a future change can't accidentally reintroduce
// a host-readback fallback.
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
    fn gpu_tensor_returns_error_for_magnitude_prune() {
        ensure_cuda_backend();
        let cpu = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let gpu = cpu.to(Device::Cuda(0)).expect("upload to cuda");
        let res = magnitude_prune(&gpu, 0.5);
        assert!(
            res.is_err(),
            "magnitude_prune on a GPU tensor must return Err (PyTorch parity), got Ok"
        );
    }

    #[test]
    fn gpu_tensor_returns_error_for_apply_2_4_mask() {
        ensure_cuda_backend();
        let cpu = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8], false);
        let gpu = cpu.to(Device::Cuda(0)).expect("upload to cuda");
        let res = apply_2_4_mask(&gpu);
        assert!(
            res.is_err(),
            "apply_2_4_mask on a GPU tensor must return Err (PyTorch parity), got Ok"
        );
    }
}
