//! Conformance Phase 2.6 — `ferrotorch-core` einops + einsum parity against
//! PyTorch.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/768>.
//! Parent: #759.
//!
//! Source files exercised:
//! - `ferrotorch-core/src/einops.rs` — `EinopsReduction`, `rearrange`,
//!   `rearrange_with`, `reduce`, `repeat`.
//! - `ferrotorch-core/src/einsum.rs` — `einsum`, `einsum_differentiable`.
//!
//! Scope per the dispatch (7 canonical-path items per #768):
//!
//! * **Cat A — Forwards**: `rearrange`, `rearrange_with`, `reduce`, `repeat`,
//!   `einsum`. `EinopsReduction` is exercised transitively via every `reduce`
//!   case (Sum / Mean / Max / Min discriminators).
//! * **Cat B — Backwards**: `einsum_differentiable` is the only autograd-
//!   integrated entry point in this phase. `rearrange` / `repeat` / `reduce`
//!   in `einops.rs` do not register grad_fns (their CPU loops produce
//!   `requires_grad=false` outputs by construction); per the dispatch's
//!   "implicit-coverage" exclusion, no separate `*_backward` covers them.
//!
//! ## Tolerances
//!
//! Per the dispatch table:
//!
//! * `rearrange` / `rearrange_with` / `repeat` — bit-exact (no arithmetic).
//! * `reduce` — `F32_REDUCTION` / `F64_REDUCTION`.
//! * `einsum`:
//!   * pure transpose / sum / outer / Hadamard / diagonal — bit-exact.
//!   * matmul-like contractions — `F32_MATMUL_GPU = 1e-3` /
//!     `F64_MATMUL_GPU = 1e-9` (the GPU contraction rounding budget; CPU is
//!     tighter at `F32_REDUCTION` since it's a simple loop).
//!
//! ## Device behaviour notes (verified by reading `einops.rs` / `einsum.rs`)
//!
//! * `rearrange` / `rearrange_with` on CUDA: identity-permutation patterns
//!   take a zero-copy `view_reshape` GPU fast path; permutation patterns
//!   currently CPU-detour through `data_vec()` → `to(device)` (issue #496).
//! * `reduce` on CUDA: axis-aligned-fast-path composes GPU-aware `sum_dim` /
//!   `cummax` / `cummin`; the reorder-fallback CPU-detours.
//! * `einsum` / `einsum_differentiable` on CUDA: results are device-resident
//!   (the GPU dispatch decomposes into on-device primitives per #803 / #821 /
//!   #822 / #824 / #825) and every GPU readback in this suite asserts CUDA
//!   residency (CORE-196 / #1890) — a silent CPU fallback now fails the lane.

use std::path::PathBuf;

use serde::Deserialize;
use serde::de::{self, Deserializer, SeqAccess, Visitor};

use ferrotorch_core::einops::{EinopsReduction, rearrange, rearrange_with, reduce, repeat};
use ferrotorch_core::einsum::{einsum, einsum_differentiable};
use ferrotorch_core::{Device, Tensor, TensorStorage};

// ---------------------------------------------------------------------------
// Tolerance helpers (mirrors conformance_elementwise / conformance_reduction)
// ---------------------------------------------------------------------------

mod tolerance {
    /// Bit-exact equality for ops that just permute / index data.
    pub const BIT_EXACT_F32: f32 = 0.0;
    pub const BIT_EXACT_F64: f64 = 0.0;

    /// Reduction tolerances — mean/sum/max/min in `reduce` accumulate, so
    /// follow the elementwise-reduction tightening.
    pub const F32_REDUCTION_CPU: f32 = 1e-6;
    pub const F64_REDUCTION_CPU: f64 = 1e-12;
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F32_REDUCTION_GPU: f32 = 1e-5;
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F64_REDUCTION_GPU: f64 = 1e-9;

    /// Matmul / contraction tolerances. Even on CPU these are looser than
    /// pure reductions because the inner loop fuses multiplies + adds —
    /// PyTorch's reference is GEMM-shaped, ferrotorch's is a hand loop.
    pub const F32_MATMUL_CPU: f32 = 1e-5;
    pub const F64_MATMUL_CPU: f64 = 1e-10;
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F32_MATMUL_GPU: f32 = 1e-3;
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F64_MATMUL_GPU: f64 = 1e-9;

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
            if !a.is_finite() || !e.is_finite() {
                if a.to_bits() == e.to_bits() {
                    continue;
                }
                if a.is_infinite() && e.is_infinite() && a.signum() == e.signum() {
                    continue;
                }
                panic!("{label}: index {i} non-finite mismatch (actual={a}, expected={e})");
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

    pub fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
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
            if !a.is_finite() || !e.is_finite() {
                if a.to_bits() == e.to_bits() {
                    continue;
                }
                if a.is_infinite() && e.is_infinite() && a.signum() == e.signum() {
                    continue;
                }
                panic!("{label}: index {i} non-finite mismatch (actual={a}, expected={e})");
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
}

// ---------------------------------------------------------------------------
// Strict-JSON-compatible f64 list deserializer (same shape as elementwise).
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
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    cuda_available: bool,
    #[allow(dead_code, reason = "diagnostics only")]
    python_executable: String,
    #[allow(dead_code, reason = "diagnostics only")]
    python_platform: String,
    #[allow(dead_code, reason = "diagnostics only")]
    generated_at: String,
    #[allow(dead_code, reason = "diagnostics only")]
    rng_seed: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    op: String,
    #[serde(default)]
    tag: Option<String>,
    dtype: String,
    device: String,
    /// Pattern for `rearrange` / `rearrange_with` / `repeat` / `reduce`.
    #[serde(default)]
    pattern: Option<String>,
    /// Equation for `einsum` / `einsum_differentiable`.
    #[serde(default)]
    equation: Option<String>,
    /// Reduction op name for `reduce` ("sum" / "mean" / "max" / "min").
    #[serde(default)]
    reduction: Option<String>,
    /// `[[name, size], ...]` pairs for `rearrange_with` / `repeat`.
    #[serde(default)]
    axes_lengths: Option<Vec<(String, usize)>>,
    #[serde(default)]
    a_shape: Option<Vec<usize>>,
    #[serde(default)]
    b_shape: Option<Vec<usize>>,
    #[serde(default)]
    #[allow(
        dead_code,
        reason = "deserialized for fixture-shape stability and future shape-checks"
    )]
    out_shape: Option<Vec<usize>>,
    #[serde(default)]
    a_data: Option<F64ListSentinel>,
    #[serde(default)]
    b_data: Option<F64ListSentinel>,
    #[serde(default)]
    out_values: Option<F64ListSentinel>,
    #[serde(default)]
    grad_a: Option<F64ListSentinel>,
    #[serde(default)]
    grad_b: Option<F64ListSentinel>,
}

fn load_fixtures() -> FixtureFile {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures")
        .join("einops.json");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. Regenerate via \
             `python3 scripts/regenerate_einops_fixtures.py`",
            p.display()
        )
    });
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn cases_for<'a>(file: &'a FixtureFile, op: &str, device: &str) -> Vec<&'a Fixture> {
    file.fixtures
        .iter()
        .filter(|f| f.op == op && f.device == device)
        .collect()
}

// ---------------------------------------------------------------------------
// Tensor helpers (readback is device-CHECKED — CORE-196 / #1890)
// ---------------------------------------------------------------------------

/// Device-checked readback (CORE-196 / #1890). When `expect` is a CUDA
/// device the tensor must actually be CUDA-resident before the D2H copy:
/// a CPU-resident result produced from CUDA inputs means the op under test
/// silently fell back to host compute, which a device-transparent readback
/// would green-light forever.
fn read_back_f32(t: &Tensor<f32>, expect: Device) -> Vec<f32> {
    if expect.is_cuda() {
        assert_eq!(
            t.device(),
            expect,
            "result expected on {expect:?} but resides on {:?} — \
             silent CPU fallback (CORE-196 / #1890)",
            t.device()
        );
    }
    if t.is_cpu() {
        t.data().expect("read CPU data").to_vec()
    } else {
        let cpu = t.cpu().expect("D2H readback");
        cpu.data().expect("read CPU data after readback").to_vec()
    }
}

/// See [`read_back_f32`] — device-checked readback (CORE-196 / #1890).
fn read_back_f64(t: &Tensor<f64>, expect: Device) -> Vec<f64> {
    if expect.is_cuda() {
        assert_eq!(
            t.device(),
            expect,
            "result expected on {expect:?} but resides on {:?} — \
             silent CPU fallback (CORE-196 / #1890)",
            t.device()
        );
    }
    if t.is_cpu() {
        t.data().expect("read CPU data").to_vec()
    } else {
        let cpu = t.cpu().expect("D2H readback");
        cpu.data().expect("read CPU data after readback").to_vec()
    }
}

fn make_cpu_f32(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    let v: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(v), shape.to_vec(), requires_grad)
        .expect("make_cpu_f32")
}

fn make_cpu_f64(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("make_cpu_f64")
}

fn upload_f32(t: Tensor<f32>, device: Device) -> Tensor<f32> {
    if matches!(device, Device::Cuda(_)) {
        t.to(device).expect("upload to cuda")
    } else {
        t
    }
}

fn upload_f64(t: Tensor<f64>, device: Device) -> Tensor<f64> {
    if matches!(device, Device::Cuda(_)) {
        t.to(device).expect("upload to cuda")
    } else {
        t
    }
}

fn check_f32(label: &str, actual: &[f32], expected: &[f64], tol: f32) {
    let exp_f32: Vec<f32> = expected.iter().map(|&x| x as f32).collect();
    tolerance::assert_close_f32(actual, &exp_f32, tol, label);
}

fn check_f64(label: &str, actual: &[f64], expected: &[f64], tol: f64) {
    tolerance::assert_close_f64(actual, expected, tol, label);
}

/// Per-fixture diagnostic skip for cascade issues surfaced by Phase 2.6.
/// Returns `Some(reason)` to skip with a printed reason; returns `None` to
/// run normally. The dispatch's cascade-handling mandate requires surfacing
/// each failure with a tracking issue rather than silently weakening
/// tolerance.
///
/// `tag` is reserved for per-equation cascade rows (e.g. when a future
/// row needs to skip a specific einsum tag). With #821 / #822 closing
/// the only previously-tag-keyed cascade rows, all skips currently
/// dispatch off (op, device, dtype) only — `tag` is left in the
/// signature for symmetry with the future surface.
fn cascade_skip(
    _op: &str,
    _device_label: &str,
    _dtype: &str,
    _tag: Option<&str>,
) -> Option<&'static str> {
    // Issue #791 (CLOSED in Bugfix Batch 4 / Dispatch A3):
    // EinsumBackwardSingle now handles projection / axis-sum cases
    // structurally (broadcast-to-lhs-shape) instead of constructing
    // a malformed reverse-equation. The skip is therefore removed.
    //
    // Issue #803 (CLOSED — partial — in Batch 4 / A4): einsum forward
    // on CUDA now decomposes into GPU primitives for the patterns the
    // existing primitive surface covers (matmul, bmm, permutation,
    // axis sum, full reduce).
    //
    // Issue #821 (CLOSED in Bugfix Batch 6 / Dispatch A2): repeated-
    // index equations (`"ii->"` trace, `"ii->i"` diagonal, `"ii"`
    // implicit trace) now decompose on-device via `as_strided_copy`
    // (shape [N], stride [N+1]) + `sum_dim`. The cascade_skip rows
    // for `trace_2d` / `diagonal_2d` are therefore removed — the
    // conformance fixtures for those tags now exercise the GPU path
    // end-to-end.
    //
    // Issue #822 (CLOSED in Bugfix Batch 6 / Dispatch A2): the
    // 2-input GPU dispatch now handles general multi-axis and
    // permuted contractions via permute+reshape+bmm. No conformance
    // fixtures targeted those equations specifically (they would
    // route through `einsum_general`), but the path is now live for
    // any future fixture additions.
    //
    // Issue #824 (CLOSED in Final mop-up A2): single-input mixed
    // repeated/free indices (`"iij->j"`, `"iji->j"`, `"iijk->jk"`,
    // `"iij->ij"`) decompose on-device via the shared
    // `diagonalize_repeats_gpu` helper — each repeat-class becomes a
    // single axis whose stride is the sum of the original strides of
    // every axis carrying that char, then the standard sum-axes/permute
    // path handles the rest. No conformance fixtures targeted those
    // equations directly (they previously errored via
    // `einsum_repeated_index_mixed`), but the path is now live.
    //
    // Issue #825 (CLOSED in Final mop-up A2): two-input einsum with
    // operand repeats (`"ii,j->j"`, `"ij,jj->i"`, `"ii,jk->jk"`) is
    // handled by a pre-pass that diagonalises each offending operand
    // via the same `diagonalize_repeats_gpu` machinery before routing
    // into the existing 2-input decomposition (#822). No conformance
    // fixtures targeted those equations directly, but the path is now
    // live for any future fixture additions.

    // Issue #790 (CLOSED in Bugfix Batch 6 / Dispatch A1): the symptom
    // (GPU `reduce(Max/Min)` returning the first row instead of the
    // column-wise extremum) was a downstream consequence of the strided-
    // view-readback bug fixed under #802 (Bugfix Batch 1) — the
    // `cummax → narrow(last) → squeeze → view_reshape` chain on CUDA
    // produced a tensor with non-zero `storage_offset`, and pre-#802
    // `gpu_to_cpu` discarded the offset so `.cpu()` read the first slice
    // of the buffer rather than the narrowed view. With #802's on-device
    // strided-copy materialization in place the cummax / narrow / squeeze
    // primitives all return correct values; see
    // `tests/_probe_b6_a1_reduce_max_min_gpu.rs` for the verifying probes.

    None
}

/// Convert axes_lengths from owned `(String, usize)` pairs to the
/// `&[(&str, usize)]` slice ferrotorch's API expects.
fn axes_pairs(pairs: &Option<Vec<(String, usize)>>) -> Vec<(&str, usize)> {
    pairs
        .as_ref()
        .map(|v| v.iter().map(|(s, n)| (s.as_str(), *n)).collect())
        .unwrap_or_default()
}

fn parse_reduction(name: &str) -> EinopsReduction {
    match name {
        "sum" => EinopsReduction::Sum,
        "mean" => EinopsReduction::Mean,
        "max" => EinopsReduction::Max,
        "min" => EinopsReduction::Min,
        other => panic!("unexpected reduction op {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// rearrange / rearrange_with — bit-exact data movement
// ---------------------------------------------------------------------------

fn run_rearrange_for_device(op_name: &str, device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, op_name, device_label);
    assert!(
        !cases.is_empty(),
        "no fixtures for {op_name} on {device_label}"
    );
    for f in cases {
        if let Some(reason) = cascade_skip(op_name, device_label, &f.dtype, f.tag.as_deref()) {
            eprintln!(
                "skipping {op_name} {device_label} dtype={} tag={:?}: {reason}",
                f.dtype, f.tag,
            );
            continue;
        }
        let label = format!("{op_name} {device_label} tag={:?} dtype={}", f.tag, f.dtype);
        let shape = f.a_shape.as_ref().expect("a_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("out_values");
        let pattern = f.pattern.as_deref().expect("pattern");
        let pairs = axes_pairs(&f.axes_lengths);

        match f.dtype.as_str() {
            "float32" => {
                let a = upload_f32(make_cpu_f32(a_data, shape, false), device);
                let r = if op_name == "rearrange" {
                    rearrange(&a, pattern).expect("rearrange")
                } else {
                    rearrange_with(&a, pattern, &pairs).expect("rearrange_with")
                };
                check_f32(
                    &label,
                    &read_back_f32(&r, device),
                    expected,
                    tolerance::BIT_EXACT_F32,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(a_data, shape, false), device);
                let r = if op_name == "rearrange" {
                    rearrange(&a, pattern).expect("rearrange")
                } else {
                    rearrange_with(&a, pattern, &pairs).expect("rearrange_with")
                };
                check_f64(
                    &label,
                    &read_back_f64(&r, device),
                    expected,
                    tolerance::BIT_EXACT_F64,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_rearrange() {
    run_rearrange_for_device("rearrange", "cpu", Device::Cpu);
}

#[test]
fn cpu_rearrange_with() {
    run_rearrange_for_device("rearrange_with", "cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// repeat — bit-exact data duplication
// ---------------------------------------------------------------------------

fn run_repeat_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "repeat", device_label);
    assert!(
        !cases.is_empty(),
        "no fixtures for repeat on {device_label}"
    );
    for f in cases {
        if let Some(reason) = cascade_skip("repeat", device_label, &f.dtype, f.tag.as_deref()) {
            eprintln!(
                "skipping repeat {device_label} dtype={} tag={:?}: {reason}",
                f.dtype, f.tag,
            );
            continue;
        }
        let label = format!("repeat {device_label} tag={:?} dtype={}", f.tag, f.dtype);
        let shape = f.a_shape.as_ref().expect("a_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("out_values");
        let pattern = f.pattern.as_deref().expect("pattern");
        let pairs = axes_pairs(&f.axes_lengths);

        match f.dtype.as_str() {
            "float32" => {
                let a = upload_f32(make_cpu_f32(a_data, shape, false), device);
                let r = repeat(&a, pattern, &pairs).expect("repeat");
                check_f32(
                    &label,
                    &read_back_f32(&r, device),
                    expected,
                    tolerance::BIT_EXACT_F32,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(a_data, shape, false), device);
                let r = repeat(&a, pattern, &pairs).expect("repeat");
                check_f64(
                    &label,
                    &read_back_f64(&r, device),
                    expected,
                    tolerance::BIT_EXACT_F64,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_repeat() {
    run_repeat_for_device("cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// reduce — Sum / Mean / Max / Min
// ---------------------------------------------------------------------------

fn run_reduce_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "reduce", device_label);
    assert!(
        !cases.is_empty(),
        "no fixtures for reduce on {device_label}"
    );
    let on_gpu = matches!(device, Device::Cuda(_));
    let tol_f32 = if on_gpu {
        tolerance::F32_REDUCTION_GPU
    } else {
        tolerance::F32_REDUCTION_CPU
    };
    let tol_f64 = if on_gpu {
        tolerance::F64_REDUCTION_GPU
    } else {
        tolerance::F64_REDUCTION_CPU
    };

    for f in cases {
        let red_name_for_skip = f.reduction.as_deref().unwrap_or("");
        // Synthetic op-key so cascade_skip can target Max/Min specifically
        // (#790) without affecting Sum/Mean.
        let synth_op = if device_label == "cuda:0"
            && (red_name_for_skip == "max" || red_name_for_skip == "min")
        {
            "reduce_max_min_gpu"
        } else {
            "reduce"
        };
        if let Some(reason) = cascade_skip(synth_op, device_label, &f.dtype, f.tag.as_deref()) {
            eprintln!(
                "skipping reduce/{red_name_for_skip} {device_label} dtype={} tag={:?}: {reason}",
                f.dtype, f.tag,
            );
            continue;
        }
        let red_name = f.reduction.as_deref().expect("reduction");
        let label = format!(
            "reduce/{red_name} {device_label} tag={:?} dtype={}",
            f.tag, f.dtype,
        );
        let shape = f.a_shape.as_ref().expect("a_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("out_values");
        let pattern = f.pattern.as_deref().expect("pattern");
        let red = parse_reduction(red_name);

        // Max/Min are integer-equality-on-distinct-values ops at the algebra
        // level, but our fixture inputs use 0.5-step values to keep the
        // distinct-tie-free property — so a small rounding tolerance is fine.
        match f.dtype.as_str() {
            "float32" => {
                let a = upload_f32(make_cpu_f32(a_data, shape, false), device);
                let r = reduce(&a, pattern, red).expect("reduce");
                check_f32(&label, &read_back_f32(&r, device), expected, tol_f32);
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(a_data, shape, false), device);
                let r = reduce(&a, pattern, red).expect("reduce");
                check_f64(&label, &read_back_f64(&r, device), expected, tol_f64);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_reduce() {
    run_reduce_for_device("cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// einsum — single + two-input forwards
// ---------------------------------------------------------------------------
//
// `einsum` is permutation-only or reduction-only at the algebra level for
// its single-input flavors (transpose, sum, trace, diagonal, axis-sum) so
// those compare bit-exact. Two-input contractions accumulate over the
// contracted axes — they get a matmul tolerance.

fn einsum_tolerance_f32(equation: &str, on_gpu: bool) -> f32 {
    // Identify whether the equation involves a contracted axis (i.e. some
    // index appears in inputs but not in output). If yes, this is matmul-like.
    if equation_has_contraction(equation) {
        if on_gpu {
            tolerance::F32_MATMUL_GPU
        } else {
            tolerance::F32_MATMUL_CPU
        }
    } else {
        // Pure rearrange / pick / no-arithmetic equations.
        tolerance::BIT_EXACT_F32
    }
}

fn einsum_tolerance_f64(equation: &str, on_gpu: bool) -> f64 {
    if equation_has_contraction(equation) {
        if on_gpu {
            tolerance::F64_MATMUL_GPU
        } else {
            tolerance::F64_MATMUL_CPU
        }
    } else {
        tolerance::BIT_EXACT_F64
    }
}

/// Returns true if the equation contracts at least one axis (an index
/// appears in some input but not in the output, or implicit-mode reduction).
fn equation_has_contraction(equation: &str) -> bool {
    let stripped: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
    let (lhs, rhs_opt) = if let Some((l, r)) = stripped.split_once("->") {
        (l.to_string(), Some(r.to_string()))
    } else {
        (stripped.clone(), None)
    };

    // Collect index counts on the LHS (excluding commas).
    let mut counts: std::collections::BTreeMap<char, usize> = std::collections::BTreeMap::new();
    for c in lhs.chars().filter(|c| *c != ',') {
        *counts.entry(c).or_insert(0) += 1;
    }

    match rhs_opt {
        Some(rhs) => {
            // Explicit mode: contraction = some LHS index missing from RHS,
            // or some LHS index appearing >1 times (repeated index = trace
            // / diagonal — but diagonal "ii->i" has the index in RHS and is
            // a pick, NOT a contraction).
            for (c, n) in &counts {
                let in_rhs = rhs.contains(*c);
                if !in_rhs {
                    return true;
                }
                if *n > 1 && !rhs.contains(*c) {
                    return true;
                }
                // Trace-style: "ii->" — handled by the `in_rhs` branch (RHS empty).
                // Diagonal-style: "ii->i" — RHS contains 'i', n=2 in LHS but
                // PyTorch reads this as a pick along the diagonal — no
                // arithmetic, bit-exact. So we don't flag it.
            }
            false
        }
        None => {
            // Implicit mode: any index appearing >1 time (across all input
            // operands) is contracted.
            counts.values().any(|&n| n > 1)
        }
    }
}

fn run_einsum_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "einsum", device_label);
    assert!(
        !cases.is_empty(),
        "no fixtures for einsum on {device_label}"
    );
    let on_gpu = matches!(device, Device::Cuda(_));

    for f in cases {
        if let Some(reason) = cascade_skip("einsum", device_label, &f.dtype, f.tag.as_deref()) {
            eprintln!(
                "skipping einsum {device_label} dtype={} tag={:?}: {reason}",
                f.dtype, f.tag,
            );
            continue;
        }
        let equation = f.equation.as_deref().expect("equation");
        let label = format!(
            "einsum {device_label} eq={equation:?} tag={:?} dtype={}",
            f.tag, f.dtype,
        );
        let a_shape = f.a_shape.as_ref().expect("a_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("out_values");
        let tol_f32 = einsum_tolerance_f32(equation, on_gpu);
        let tol_f64 = einsum_tolerance_f64(equation, on_gpu);

        match f.dtype.as_str() {
            "float32" => {
                let a = upload_f32(make_cpu_f32(a_data, a_shape, false), device);
                let r = if let Some(b_shape) = &f.b_shape {
                    let b_data = f
                        .b_data
                        .as_ref()
                        .map(F64ListSentinel::as_slice)
                        .expect("b_data");
                    let b = upload_f32(make_cpu_f32(b_data, b_shape, false), device);
                    einsum(equation, &[&a, &b]).expect("einsum")
                } else {
                    einsum(equation, &[&a]).expect("einsum")
                };
                check_f32(&label, &read_back_f32(&r, device), expected, tol_f32);
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(a_data, a_shape, false), device);
                let r = if let Some(b_shape) = &f.b_shape {
                    let b_data = f
                        .b_data
                        .as_ref()
                        .map(F64ListSentinel::as_slice)
                        .expect("b_data");
                    let b = upload_f64(make_cpu_f64(b_data, b_shape, false), device);
                    einsum(equation, &[&a, &b]).expect("einsum")
                } else {
                    einsum(equation, &[&a]).expect("einsum")
                };
                check_f64(&label, &read_back_f64(&r, device), expected, tol_f64);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_einsum() {
    run_einsum_for_device("cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// einsum_differentiable — forward + backward (Cat B)
// ---------------------------------------------------------------------------

fn run_einsum_differentiable_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "einsum_differentiable", device_label);
    assert!(
        !cases.is_empty(),
        "no fixtures for einsum_differentiable on {device_label}"
    );
    let on_gpu = matches!(device, Device::Cuda(_));

    for f in cases {
        if let Some(reason) = cascade_skip(
            "einsum_differentiable",
            device_label,
            &f.dtype,
            f.tag.as_deref(),
        ) {
            eprintln!(
                "skipping einsum_differentiable {device_label} dtype={} tag={:?}: {reason}",
                f.dtype, f.tag,
            );
            continue;
        }
        let equation = f.equation.as_deref().expect("equation");
        let label = format!(
            "einsum_diff {device_label} eq={equation:?} tag={:?} dtype={}",
            f.tag, f.dtype,
        );
        let a_shape = f.a_shape.as_ref().expect("a_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("out_values");
        let grad_a_exp = f
            .grad_a
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("grad_a");
        let tol_f32 = einsum_tolerance_f32(equation, on_gpu);
        let tol_f64 = einsum_tolerance_f64(equation, on_gpu);

        match f.dtype.as_str() {
            "float32" => {
                // Forward (no grad).
                let a = upload_f32(make_cpu_f32(a_data, a_shape, false), device);
                let inputs_two = f.b_shape.is_some();
                let r = if let Some(b_shape) = &f.b_shape {
                    let b_data = f
                        .b_data
                        .as_ref()
                        .map(F64ListSentinel::as_slice)
                        .expect("b_data");
                    let b = upload_f32(make_cpu_f32(b_data, b_shape, false), device);
                    einsum_differentiable(equation, &[&a, &b]).expect("einsum_diff")
                } else {
                    einsum_differentiable(equation, &[&a]).expect("einsum_diff")
                };
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r, device),
                    expected,
                    tol_f32,
                );

                // Backward via `loss = sum(out)` then `.backward()`.
                let a_g = upload_f32(make_cpu_f32(a_data, a_shape, true), device);
                let (out, b_g_opt) = if let Some(b_shape) = &f.b_shape {
                    let b_data = f
                        .b_data
                        .as_ref()
                        .map(F64ListSentinel::as_slice)
                        .expect("b_data");
                    let b_g = upload_f32(make_cpu_f32(b_data, b_shape, true), device);
                    let o = einsum_differentiable(equation, &[&a_g, &b_g]).expect("einsum_diff");
                    (o, Some(b_g))
                } else {
                    let o = einsum_differentiable(equation, &[&a_g]).expect("einsum_diff");
                    (o, None)
                };
                // Build sum-to-scalar loss using ferrotorch_core::sum.
                let loss =
                    ferrotorch_core::grad_fns::reduction::sum(&out).expect("sum-to-scalar loss");
                loss.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&ga, device),
                    grad_a_exp,
                    tol_f32,
                );
                if inputs_two {
                    let grad_b_exp = f
                        .grad_b
                        .as_ref()
                        .map(F64ListSentinel::as_slice)
                        .expect("grad_b");
                    let gb = b_g_opt
                        .expect("b_g exists")
                        .grad()
                        .unwrap()
                        .expect("grad_b");
                    check_f32(
                        &format!("{label} grad_b"),
                        &read_back_f32(&gb, device),
                        grad_b_exp,
                        tol_f32,
                    );
                }
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(a_data, a_shape, false), device);
                let inputs_two = f.b_shape.is_some();
                let r = if let Some(b_shape) = &f.b_shape {
                    let b_data = f
                        .b_data
                        .as_ref()
                        .map(F64ListSentinel::as_slice)
                        .expect("b_data");
                    let b = upload_f64(make_cpu_f64(b_data, b_shape, false), device);
                    einsum_differentiable(equation, &[&a, &b]).expect("einsum_diff")
                } else {
                    einsum_differentiable(equation, &[&a]).expect("einsum_diff")
                };
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r, device),
                    expected,
                    tol_f64,
                );

                let a_g = upload_f64(make_cpu_f64(a_data, a_shape, true), device);
                let (out, b_g_opt) = if let Some(b_shape) = &f.b_shape {
                    let b_data = f
                        .b_data
                        .as_ref()
                        .map(F64ListSentinel::as_slice)
                        .expect("b_data");
                    let b_g = upload_f64(make_cpu_f64(b_data, b_shape, true), device);
                    let o = einsum_differentiable(equation, &[&a_g, &b_g]).expect("einsum_diff");
                    (o, Some(b_g))
                } else {
                    let o = einsum_differentiable(equation, &[&a_g]).expect("einsum_diff");
                    (o, None)
                };
                let loss =
                    ferrotorch_core::grad_fns::reduction::sum(&out).expect("sum-to-scalar loss");
                loss.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&ga, device),
                    grad_a_exp,
                    tol_f64,
                );
                if inputs_two {
                    let grad_b_exp = f
                        .grad_b
                        .as_ref()
                        .map(F64ListSentinel::as_slice)
                        .expect("grad_b");
                    let gb = b_g_opt
                        .expect("b_g exists")
                        .grad()
                        .unwrap()
                        .expect("grad_b");
                    check_f64(
                        &format!("{label} grad_b"),
                        &read_back_f64(&gb, device),
                        grad_b_exp,
                        tol_f64,
                    );
                }
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_einsum_differentiable() {
    run_einsum_differentiable_for_device("cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// EinopsReduction — direct enum-discriminator coverage
// ---------------------------------------------------------------------------
//
// The enum is exercised transitively in every `cpu_reduce` / `gpu_reduce`
// case (each fixture pins one of Sum / Mean / Max / Min). To also satisfy
// the surface-coverage substring grep we reference each variant by name in
// a compact discriminator-coverage test.

#[test]
fn einops_reduction_variants_cover_every_discriminator() {
    let variants = [
        EinopsReduction::Sum,
        EinopsReduction::Mean,
        EinopsReduction::Max,
        EinopsReduction::Min,
    ];
    // Smoke-test each variant against a tiny CPU input to confirm dispatch
    // wires through end-to-end (forward correctness is also separately
    // checked by `cpu_reduce` against the PyTorch fixtures).
    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2], false).expect("build leaf");
    for v in variants {
        let r = reduce(&t, "i j -> i", v).expect("reduce");
        assert_eq!(r.shape(), &[2], "variant {v:?} shape");
        let out = r.data().expect("read out");
        match v {
            EinopsReduction::Sum => {
                tolerance::assert_close_f32(out, &[3.0, 7.0], 1e-6, "Sum");
            }
            EinopsReduction::Mean => {
                tolerance::assert_close_f32(out, &[1.5, 3.5], 1e-6, "Mean");
            }
            EinopsReduction::Max => {
                tolerance::assert_close_f32(out, &[2.0, 4.0], 1e-6, "Max");
            }
            EinopsReduction::Min => {
                tolerance::assert_close_f32(out, &[1.0, 3.0], 1e-6, "Min");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Negative-shape coverage: einsum errors on >2 inputs (per source comment).
// ---------------------------------------------------------------------------

#[test]
fn einsum_rejects_more_than_two_inputs() {
    let a = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
        vec![2, 2],
        false,
    )
    .expect("build a");
    let b = Tensor::from_storage(
        TensorStorage::cpu(vec![5.0_f32, 6.0, 7.0, 8.0]),
        vec![2, 2],
        false,
    )
    .expect("build b");
    let c = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 0.0, 0.0, 1.0]),
        vec![2, 2],
        false,
    )
    .expect("build c");
    let r = einsum::<f32>("ij,jk,kl->il", &[&a, &b, &c]);
    assert!(
        r.is_err(),
        "einsum with 3 inputs must error (ferrotorch's einsum is 1-or-2-input only)"
    );
}

// ---------------------------------------------------------------------------
// Sanity: assert the fixture file has every op we expect.
// ---------------------------------------------------------------------------

#[test]
fn fixture_file_covers_every_phase26_op() {
    let file = load_fixtures();
    let mut by_op: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for f in &file.fixtures {
        *by_op.entry(f.op.as_str()).or_insert(0) += 1;
    }
    let required = [
        "rearrange",
        "rearrange_with",
        "repeat",
        "reduce",
        "einsum",
        "einsum_differentiable",
    ];
    for r in required {
        let n = by_op.get(r).copied().unwrap_or(0);
        assert!(n > 0, "fixture file missing op {r:?}");
    }
}

// ---------------------------------------------------------------------------
// GPU paths — gated on the `gpu` feature
// ---------------------------------------------------------------------------
//
// Same dispatch pattern as elementwise/creation/reduction: gate on
// `#[cfg(feature = "gpu")]` rather than `#[ignore]` so a non-GPU build
// has these tests genuinely absent (not silently skipped).

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use std::sync::Once;

    static GPU_INIT: Once = Once::new();

    fn ensure_cuda_backend() {
        GPU_INIT.call_once(|| {
            ferrotorch_gpu::init_cuda_backend()
                .expect("CUDA backend must initialize for the GPU conformance suite");
        });
    }

    fn require_cuda_fixtures(file: &FixtureFile) {
        if !file.metadata.cuda_available {
            panic!(
                "fixtures/einops.json was generated without CUDA — \
                 regenerate on a CUDA-enabled host before running --features gpu tests"
            );
        }
    }

    #[test]
    fn gpu_rearrange() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_rearrange_for_device("rearrange", "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_rearrange_with() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_rearrange_for_device("rearrange_with", "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_repeat() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_repeat_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_reduce() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_reduce_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_einsum() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_einsum_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_einsum_differentiable() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_einsum_differentiable_for_device("cuda:0", Device::Cuda(0));
    }
}
