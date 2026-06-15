//! Conformance Phase 2.2 — `ferrotorch-core` reductions + cumulative parity
//! against PyTorch.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/764>.
//! Parent: #759.
//!
//! Source files exercised:
//! - `ferrotorch-core/src/grad_fns/reduction.rs` — Cat A forwards & backward
//!   structs (`SumBackward`, `MeanBackward`, `ProdBackward`, `AmaxBackward`,
//!   `AminBackward`, `SumDimBackward`, `MeanDimBackward`).
//! - `ferrotorch-core/src/grad_fns/cumulative.rs` — Cat B forwards & backward
//!   structs (`CumsumBackward`, `CumprodBackward`, `LogcumsumexpBackward`).
//! - `ferrotorch-core/src/ops/cumulative.rs` — Cat C forward-only helpers
//!   plus `reverse_cumsum` (raw-slice utility).
//!
//! Scope per the dispatch:
//!
//! * **Cat A** (sum/sum_dim/mean/mean_dim/prod/amax/amin): CPU + GPU forward
//!   plus autograd, with edge cases (empty tensor, 1D/2D/3D, every dim with
//!   keepdim toggle for sum_dim/mean_dim, amax/amin tie mass distribution).
//! * **Cat B** (cumsum/cumprod/cummax/cummin/logcumsumexp): CPU + GPU forward
//!   plus resident CUDA autograd for supported floating dtypes, with edge
//!   cases (cumprod-with-zero, logcumsumexp-stability, every dim, 1D/2D/3D).
//! * **Cat C** forward-only helpers (`*_forward`): implicit coverage via
//!   Cat B autograd path (the wrappers call them transitively).
//! * **Cat D** backward grad_fn structs: implicit coverage via the relevant
//!   forward op's autograd assertion (no phantom tests).
//! * **Cat E** `CumExtremeResult`: direct field-access test.
//! * **`reverse_cumsum`**: raw-slice utility — direct unit test with a
//!   manual reference. CPU-only by signature.
//!
//! The tolerance helpers re-implement the same pattern as
//! `conformance_elementwise.rs` so the test stays independently buildable.
//! Constants follow the dispatch table:
//!   F32_REDUCTION_CPU = 1e-6, F32_REDUCTION_GPU = 1e-5,
//!   F64_REDUCTION_CPU = F64_REDUCTION_GPU = 1e-9 (per the f64-tightening
//!   factor).

use std::path::PathBuf;

use serde::Deserialize;
use serde::de::{self, Deserializer, SeqAccess, Visitor};

use ferrotorch_core::grad_fns::cumulative::{cummax, cummin, cumprod, cumsum, logcumsumexp};
use ferrotorch_core::grad_fns::reduction::{
    amax, amax_dim, amin, amin_dim, argmax_dim, argmin_dim, max_with_dim, mean, mean_dim,
    median_with_dim, min_with_dim, nanmedian_with_dim, prod, sum, sum_dim,
};
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::ops::cumulative::{
    CumExtremeResult, cummax_forward, cummin_forward, cumprod_forward, cumsum_forward,
    logcumsumexp_forward, reverse_cumsum,
};
use ferrotorch_core::{Device, Tensor, TensorStorage};

// ---------------------------------------------------------------------------
// Tolerance helpers
// ---------------------------------------------------------------------------
//
// Mirror the structure used in `conformance_elementwise.rs`. Tightening rules
// are the dispatch's:
//   * F32_REDUCTION_CPU 1e-6 (1 ulp at unit magnitude)
//   * F32_REDUCTION_GPU 1e-5 (extra rounding from cuBLAS-style accumulation)
//   * F64 = 1e-9 (per the f64-tightening factor on top of the elementwise
//                 1e-12 — reductions accumulate so we relax a bit; this is
//                 still well inside libm bounds).
//   * F64_LOGSUMEXP allows extra slack for logcumsumexp's
//     log(sum(exp(...))) chain.

mod tolerance {
    pub const F32_REDUCTION_CPU: f32 = 1e-6;
    pub const F64_REDUCTION_CPU: f64 = 1e-9;

    /// log/exp/scan compositions accumulate transcendental rounding;
    /// f64 still holds at 1e-9 with comfortable headroom.
    pub const F32_LOGSCAN_CPU: f32 = 1e-5;
    pub const F64_LOGSCAN_CPU: f64 = 1e-9;

    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F32_REDUCTION_GPU: f32 = 1e-5;
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F64_REDUCTION_GPU: f64 = 1e-9;
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F32_LOGSCAN_GPU: f32 = 1e-4;
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F64_LOGSCAN_GPU: f64 = 1e-9;

    /// Accumulation-aware reduction tolerance (R-ORACLE-5, CORE-199 / #1893
    /// sweep lanes, k up to 10007 summands).
    ///
    /// Analytic justification: torch reduces with pairwise summation
    /// (error O(eps*log2 k)); ferrotorch folds sequentially (deterministic
    /// bound O(eps*k), expected O(eps*sqrt(k)) under the standard
    /// random-rounding model — Higham, *Accuracy and Stability of Numerical
    /// Algorithms*, sec. 4.2). The order difference is therefore expected
    /// O(eps*sqrt(k)); the factor 8 covers the constant without admitting
    /// the eps*k worst case. For small k the per-lane base band dominates
    /// via `max`, so all pre-sweep rows keep their original bound.
    pub fn accum_tol_f32(base: f32, k: usize) -> f32 {
        base.max(8.0 * (k as f32).sqrt() * f32::EPSILON)
    }

    /// See [`accum_tol_f32`]; same model at f64 epsilon.
    pub fn accum_tol_f64(base: f64, k: usize) -> f64 {
        base.max(8.0 * (k as f64).sqrt() * f64::EPSILON)
    }

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
    #[serde(default)]
    a_shape: Option<Vec<usize>>,
    #[serde(default)]
    #[allow(
        dead_code,
        reason = "deserialized for fixture-shape stability and future shape-checks"
    )]
    out_shape: Option<Vec<usize>>,
    #[serde(default)]
    a_data: Option<F64ListSentinel>,
    #[serde(default)]
    out_values: Option<F64ListSentinel>,
    #[serde(default)]
    out_indices: Option<Vec<usize>>,
    #[serde(default)]
    grad_a: Option<F64ListSentinel>,
    /// Signed dim — cumulative ops accept negative axes (`-1` = last dim).
    #[serde(default)]
    axis: Option<i64>,
    #[serde(default)]
    keepdim: Option<bool>,
    /// CORE-199 / #1893 non-contiguous lane: when `true`, `a_data` is the
    /// CONTIGUOUS row-major base buffer and the runner applies
    /// `.transpose(0, 1)` to build the non-contiguous view the op consumes.
    #[serde(default)]
    input_transpose: Option<bool>,
}

fn load_fixtures() -> FixtureFile {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures")
        .join("reduction.json");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. Regenerate via \
             `python3 scripts/regenerate_reduction_fixtures.py`",
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
        // CORE-012 (#1706): `.to(device)` of a requires-grad leaf is a
        // differentiable copy (non-leaf; grads accumulate on the ORIGINAL
        // CPU leaf, as in torch). These suites assert `.grad()` on the
        // uploaded tensor, so build a true CUDA leaf via torch's
        // `x.to('cuda').detach().requires_grad_(True)` idiom.
        let track = t.requires_grad();
        t.detach()
            .to(device)
            .expect("upload to cuda")
            .requires_grad_(track)
    } else {
        t
    }
}

fn upload_f64(t: Tensor<f64>, device: Device) -> Tensor<f64> {
    if matches!(device, Device::Cuda(_)) {
        // CORE-012 (#1706): `.to(device)` of a requires-grad leaf is a
        // differentiable copy (non-leaf; grads accumulate on the ORIGINAL
        // CPU leaf, as in torch). These suites assert `.grad()` on the
        // uploaded tensor, so build a true CUDA leaf via torch's
        // `x.to('cuda').detach().requires_grad_(True)` idiom.
        let track = t.requires_grad();
        t.detach()
            .to(device)
            .expect("upload to cuda")
            .requires_grad_(track)
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

// ---------------------------------------------------------------------------
// Cat A — global reductions (sum / mean / prod / amax / amin)
// ---------------------------------------------------------------------------
//
// Each op's forward returns a 0-D scalar; loss = output, output.backward()
// drives the grad-vs-fixture assertion. The autograd graph is the simplest
// possible (single op chain), so any divergence pinpoints the op exactly.

#[derive(Clone, Copy)]
enum GlobalReduction {
    Sum,
    Mean,
    Prod,
    Amax,
    Amin,
}

impl GlobalReduction {
    fn name(self) -> &'static str {
        match self {
            GlobalReduction::Sum => "sum",
            GlobalReduction::Mean => "mean",
            GlobalReduction::Prod => "prod",
            GlobalReduction::Amax => "amax",
            GlobalReduction::Amin => "amin",
        }
    }
    fn apply_f32(self, a: &Tensor<f32>) -> Tensor<f32> {
        match self {
            GlobalReduction::Sum => sum(a).expect("sum"),
            GlobalReduction::Mean => mean(a).expect("mean"),
            GlobalReduction::Prod => prod(a).expect("prod"),
            GlobalReduction::Amax => amax(a).expect("amax"),
            GlobalReduction::Amin => amin(a).expect("amin"),
        }
    }
    fn apply_f64(self, a: &Tensor<f64>) -> Tensor<f64> {
        match self {
            GlobalReduction::Sum => sum(a).expect("sum"),
            GlobalReduction::Mean => mean(a).expect("mean"),
            GlobalReduction::Prod => prod(a).expect("prod"),
            GlobalReduction::Amax => amax(a).expect("amax"),
            GlobalReduction::Amin => amin(a).expect("amin"),
        }
    }
}

/// Per-fixture diagnostic skip for cascade issues surfaced by the GPU lane.
/// Returns `Some("issue #")` to skip with a printed reason; returns `None`
/// to run normally. The dispatch's cascade-handling mandate requires
/// surfacing each failure with a tracking issue rather than silently
/// weakening tolerance.
///
/// All four phase 2.2 cascade issues (#785, #786, #787, #788) are now
/// fixed; the function is retained as the canonical opt-out point for
/// any future cascade so the surfacing pattern is preserved.
fn cascade_skip(_op: &str, _device_label: &str, _dtype: &str) -> Option<&'static str> {
    None
}

/// Expected `.grad` device for a global reduction on `device`.
///
/// #1922 pin (found via CORE-196 / #1890): `AmaxBackward` / `AminBackward`
/// build their tie-splitting subgradient on the host and always return CPU
/// gradients, so a CUDA leaf currently gets a CPU `.grad`. torch guarantees
/// gradient device == parameter device. This helper asserts the CURRENT
/// (wrong) CPU placement for exactly those two ops so the pin fails loudly —
/// and gets retired (return `device` unconditionally) — when #1922 lands
/// device-preserving backwards. All other ops expect grads on `device`.
fn grad_device_for(
    op: GlobalReduction,
    device: Device,
    label: &str,
    actual_grad_device: Device,
) -> Device {
    if device.is_cuda() && matches!(op, GlobalReduction::Amax | GlobalReduction::Amin) {
        assert_eq!(
            actual_grad_device,
            Device::Cpu,
            "{label}: grad no longer on CPU — #1922 appears fixed; retire \
             this pin and expect the gradient on {device:?}",
        );
        return Device::Cpu;
    }
    device
}

fn run_global_reduction_for_device(op: GlobalReduction, device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, op.name(), device_label);
    assert!(
        !cases.is_empty(),
        "no fixtures for {op_name:?} on {device_label}",
        op_name = op.name()
    );
    for f in cases {
        if let Some(reason) = cascade_skip(op.name(), device_label, &f.dtype) {
            eprintln!(
                "skipping {op_name} {device_label} dtype={} tag={:?}: {reason}",
                f.dtype,
                f.tag,
                op_name = op.name(),
            );
            continue;
        }
        let label = format!(
            "{name} {device_label} tag={:?} dtype={}",
            f.tag,
            f.dtype,
            name = op.name(),
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
        let grad_a_exp = f
            .grad_a
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("grad_a");

        let (mut tol_fwd_f32, tol_grad_f32, mut tol_fwd_f64, tol_grad_f64) =
            if matches!(device, Device::Cuda(_)) {
                (
                    tolerance::F32_REDUCTION_GPU,
                    tolerance::F32_REDUCTION_GPU,
                    tolerance::F64_REDUCTION_GPU,
                    tolerance::F64_REDUCTION_GPU,
                )
            } else {
                (
                    tolerance::F32_REDUCTION_CPU,
                    tolerance::F32_REDUCTION_CPU,
                    tolerance::F64_REDUCTION_CPU,
                    tolerance::F64_REDUCTION_CPU,
                )
            };
        // CORE-199 sweep rows reduce up to k = 10007 summands; sum/mean
        // forwards take the accumulation-aware band (see
        // tolerance::accum_tol_f32). amax/amin are comparisons (exact) and
        // prod has no sweep rows, so they keep the base band. Gradients are
        // broadcast constants / indicators — no accumulation.
        if matches!(op, GlobalReduction::Sum | GlobalReduction::Mean) {
            tol_fwd_f32 = tolerance::accum_tol_f32(tol_fwd_f32, a_data.len());
            tol_fwd_f64 = tolerance::accum_tol_f64(tol_fwd_f64, a_data.len());
        }

        match f.dtype.as_str() {
            "float32" => {
                let a = upload_f32(make_cpu_f32(a_data, shape, false), device);
                let c = op.apply_f32(&a);
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&c, device),
                    expected,
                    tol_fwd_f32,
                );

                // Autograd: the output is already scalar, so we call
                // `.backward()` directly on it.
                let a_g = upload_f32(make_cpu_f32(a_data, shape, true), device);
                let out = op.apply_f32(&a_g);
                out.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&ga, grad_device_for(op, device, &label, ga.device())),
                    grad_a_exp,
                    tol_grad_f32,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(a_data, shape, false), device);
                let c = op.apply_f64(&a);
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&c, device),
                    expected,
                    tol_fwd_f64,
                );

                let a_g = upload_f64(make_cpu_f64(a_data, shape, true), device);
                let out = op.apply_f64(&a_g);
                out.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&ga, grad_device_for(op, device, &label, ga.device())),
                    grad_a_exp,
                    tol_grad_f64,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_sum() {
    run_global_reduction_for_device(GlobalReduction::Sum, "cpu", Device::Cpu);
}

#[test]
fn cpu_mean() {
    run_global_reduction_for_device(GlobalReduction::Mean, "cpu", Device::Cpu);
}

#[test]
fn cpu_prod() {
    run_global_reduction_for_device(GlobalReduction::Prod, "cpu", Device::Cpu);
}

#[test]
fn cpu_amax() {
    run_global_reduction_for_device(GlobalReduction::Amax, "cpu", Device::Cpu);
}

#[test]
fn cpu_amin() {
    run_global_reduction_for_device(GlobalReduction::Amin, "cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// Cat A — dim reductions (sum_dim / mean_dim) with keepdim toggle
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum DimReduction {
    SumDim,
    MeanDim,
}

impl DimReduction {
    fn name(self) -> &'static str {
        match self {
            DimReduction::SumDim => "sum_dim",
            DimReduction::MeanDim => "mean_dim",
        }
    }
    fn apply_f32(self, a: &Tensor<f32>, dim: i64, keepdim: bool) -> Tensor<f32> {
        match self {
            DimReduction::SumDim => sum_dim(a, dim, keepdim).expect("sum_dim"),
            DimReduction::MeanDim => mean_dim(a, dim, keepdim).expect("mean_dim"),
        }
    }
    fn apply_f64(self, a: &Tensor<f64>, dim: i64, keepdim: bool) -> Tensor<f64> {
        match self {
            DimReduction::SumDim => sum_dim(a, dim, keepdim).expect("sum_dim"),
            DimReduction::MeanDim => mean_dim(a, dim, keepdim).expect("mean_dim"),
        }
    }
}

fn run_dim_reduction_for_device(op: DimReduction, device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, op.name(), device_label);
    assert!(
        !cases.is_empty(),
        "no fixtures for {} on {device_label}",
        op.name()
    );
    for f in cases {
        if let Some(reason) = cascade_skip(op.name(), device_label, &f.dtype) {
            eprintln!(
                "skipping {} {device_label} dtype={} tag={:?}: {reason}",
                op.name(),
                f.dtype,
                f.tag,
            );
            continue;
        }
        let label = format!(
            "{} {device_label} tag={:?} dtype={}",
            op.name(),
            f.tag,
            f.dtype,
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
        let grad_a_exp = f
            .grad_a
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("grad_a");
        let axis = f.axis.expect("axis");
        let keepdim = f.keepdim.expect("keepdim");

        let (tol_fwd_f32_base, tol_grad_f32, tol_fwd_f64_base, tol_grad_f64) =
            if matches!(device, Device::Cuda(_)) {
                (
                    tolerance::F32_REDUCTION_GPU,
                    tolerance::F32_REDUCTION_GPU,
                    tolerance::F64_REDUCTION_GPU,
                    tolerance::F64_REDUCTION_GPU,
                )
            } else {
                (
                    tolerance::F32_REDUCTION_CPU,
                    tolerance::F32_REDUCTION_CPU,
                    tolerance::F64_REDUCTION_CPU,
                    tolerance::F64_REDUCTION_CPU,
                )
            };
        // CORE-199 sweep rows reduce rows of up to k = 911 elements; the
        // forward takes the accumulation-aware band over the REDUCED dim's
        // length (see tolerance::accum_tol_f32). Gradients are broadcast
        // constants — no accumulation.
        let norm_axis = if axis < 0 {
            (shape.len() as i64 + axis) as usize
        } else {
            axis as usize
        };
        let k_reduced = shape[norm_axis];
        let tol_fwd_f32 = tolerance::accum_tol_f32(tol_fwd_f32_base, k_reduced);
        let tol_fwd_f64 = tolerance::accum_tol_f64(tol_fwd_f64_base, k_reduced);

        match f.dtype.as_str() {
            "float32" => {
                let a = upload_f32(make_cpu_f32(a_data, shape, false), device);
                let c = op.apply_f32(&a, axis, keepdim);
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&c, device),
                    expected,
                    tol_fwd_f32,
                );

                // Autograd: loss = output.sum() since the output is non-scalar.
                let a_g = upload_f32(make_cpu_f32(a_data, shape, true), device);
                let out = op.apply_f32(&a_g, axis, keepdim);
                let loss = sum(&out).expect("sum-to-scalar loss");
                loss.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&ga, device),
                    grad_a_exp,
                    tol_grad_f32,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(a_data, shape, false), device);
                let c = op.apply_f64(&a, axis, keepdim);
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&c, device),
                    expected,
                    tol_fwd_f64,
                );

                let a_g = upload_f64(make_cpu_f64(a_data, shape, true), device);
                let out = op.apply_f64(&a_g, axis, keepdim);
                let loss = sum(&out).expect("sum-to-scalar loss");
                loss.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&ga, device),
                    grad_a_exp,
                    tol_grad_f64,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_sum_dim() {
    run_dim_reduction_for_device(DimReduction::SumDim, "cpu", Device::Cpu);
}

#[test]
fn cpu_mean_dim() {
    run_dim_reduction_for_device(DimReduction::MeanDim, "cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// Cat A — edge cases
// ---------------------------------------------------------------------------
//
// * sum/mean/prod on an empty 1-D tensor: forward must match torch's
//   sum=0 / mean=NaN / prod=1 contract.
// * amax/amin on an empty tensor must return Err (matching torch).
// * amax/amin tie-mass distribution: 3 equal values -> grad = 1/3 each.

#[test]
fn cpu_empty_sum_mean_prod() {
    let file = load_fixtures();
    for op_label in ["sum_empty", "mean_empty", "prod_empty"] {
        for f in cases_for(&file, op_label, "cpu") {
            let label = format!("{op_label} cpu dtype={}", f.dtype);
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
            match f.dtype.as_str() {
                "float32" => {
                    let a = make_cpu_f32(a_data, shape, false);
                    let c = match op_label {
                        "sum_empty" => sum(&a).expect("sum on empty"),
                        "mean_empty" => mean(&a).expect("mean on empty"),
                        "prod_empty" => prod(&a).expect("prod on empty"),
                        _ => unreachable!(),
                    };
                    check_f32(
                        &label,
                        &read_back_f32(&c, Device::Cpu),
                        expected,
                        tolerance::F32_REDUCTION_CPU,
                    );
                }
                "float64" => {
                    let a = make_cpu_f64(a_data, shape, false);
                    let c = match op_label {
                        "sum_empty" => sum(&a).expect("sum on empty"),
                        "mean_empty" => mean(&a).expect("mean on empty"),
                        "prod_empty" => prod(&a).expect("prod on empty"),
                        _ => unreachable!(),
                    };
                    check_f64(
                        &label,
                        &read_back_f64(&c, Device::Cpu),
                        expected,
                        tolerance::F64_REDUCTION_CPU,
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

/// CORE-051 / #1745 pin — global `amax`/`amin` return false fold identities
/// for empty tensors (single-contract rewrite of the former dual-accepting
/// test, per the CORE-199 / #1893 dispatch and R-ORACLE-4: the old version
/// passed on EITHER `Err` OR the ±inf sentinel, so it could never go red).
///
/// torch oracle (live session, torch 2.11.0+cu130):
/// ```text
/// >>> torch.amax(torch.tensor([]))
/// RuntimeError: amax(): Expected reduction dim to be specified for
/// input.numel() == 0. Specify the reduction dim with the 'dim' argument.
/// ```
/// (`amin` identical.) The contractual ferrotorch behavior once #1745 lands
/// is a structured `Err` before dispatch on every device.
///
/// ferrotorch probed at HEAD (401233b56): `amax([]) == Ok([-inf])`,
/// `amin([]) == Ok([+inf])` for both f32 and f64 — the fold identities leak
/// out as values. Pin exactly that; when #1745 lands the `expect` calls
/// below fire — retire this pin and assert the structured `Err`.
#[test]
fn cpu_empty_amax_amin_pin_1745() {
    macro_rules! pin {
        ($label:literal, $call:expr, $read:ident, $negative:expr) => {{
            let t = $call.expect(concat!(
                $label,
                " returned Err — #1745 appears fixed; retire this pin and \
                 assert the structured error"
            ));
            let v = $read(&t, Device::Cpu);
            assert_eq!(v.len(), 1, "{} must return a scalar", $label);
            assert!(
                v[0].is_infinite() && v[0].is_sign_negative() == $negative,
                "{} returned {:?} — neither the pinned fold identity nor an \
                 Err; re-probe #1745",
                $label,
                v[0]
            );
        }};
    }
    let a32 = make_cpu_f32(&[], &[0], false);
    pin!("amax_f32([])", amax(&a32), read_back_f32, true);
    pin!("amin_f32([])", amin(&a32), read_back_f32, false);
    let a64 = make_cpu_f64(&[], &[0], false);
    pin!("amax_f64([])", amax(&a64), read_back_f64, true);
    pin!("amin_f64([])", amin(&a64), read_back_f64, false);

    // Non-empty path (anti-stub): a stub returning a constant ±inf for
    // every input would satisfy the fold-identity pins above. Pin the
    // actual reduction values for `[1.0, 2.0, 3.0]` so that shortcut does
    // not survive.
    let b = make_cpu_f32(&[1.0, 2.0, 3.0], &[3], false);
    let amax_b = amax(&b).expect("amax over non-empty must succeed");
    let amax_v = read_back_f32(&amax_b, Device::Cpu);
    assert_eq!(
        amax_v.len(),
        1,
        "amax of 1-D non-empty must reduce to scalar"
    );
    assert_eq!(amax_v[0], 3.0_f32, "amax([1,2,3]) must be 3.0");
    let amin_b = amin(&b).expect("amin over non-empty must succeed");
    let amin_v = read_back_f32(&amin_b, Device::Cpu);
    assert_eq!(
        amin_v.len(),
        1,
        "amin of 1-D non-empty must reduce to scalar"
    );
    assert_eq!(amin_v[0], 1.0_f32, "amin([1,2,3]) must be 1.0");
}

/// CORE-052 / #1746 pin — dim-keyed value reductions PANIC on zero-length
/// reduced slices (lane added per the CORE-199 / #1893 dispatch).
///
/// torch oracle (live session, torch 2.11.0+cu130, `z = torch.zeros(2,0,3)`):
/// ```text
/// >>> torch.amax(z, dim=1)
/// IndexError: amax(): Expected reduction dim 1 to have non-zero size.
/// ```
/// (amin/argmax/argmin/max/min identical with their own op names;
/// median/nanmedian both report `median(): Expected reduction dim 1 to have
/// non-zero size.`) The contractual ferrotorch behavior once #1746 lands is
/// a structured `Err` from these `FerrotorchResult` APIs.
///
/// ferrotorch probed at HEAD (401233b56): all eight panic — slice
/// index-out-of-bounds at `grad_fns/reduction.rs:2430` (amin/amax_dim),
/// `:1619` (argmax/argmin_dim), `:2786` (max/min_with_dim), and a
/// `dim_size - 1` usize underflow at `:2996` for median/nanmedian (debug:
/// overflow panic; release: wraps and then indexes the empty `order` vec —
/// still a panic, so the pin is build-profile stable).
///
/// Pin mechanics: `#[should_panic]` is forbidden here (it would accept ANY
/// panic anywhere in the test body, including in tensor construction — a
/// dual-accept in disguise), and a fixture-level expected-err row cannot
/// express "panics" (the fixture runner itself would die). So the pin wraps
/// EXACTLY the op call in `catch_unwind`: today only the panic outcome
/// passes; a structured `Err` means #1746 landed (retire the pin and assert
/// the error kind); an `Ok` is an instant failure (torch raises — there is
/// no valid value).
#[test]
fn cpu_zero_length_slice_dim_reductions_panic_pin_1746() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    macro_rules! pin_panic {
        ($name:literal, $call:expr) => {{
            let r = catch_unwind(AssertUnwindSafe(|| $call.map(|_| ())));
            match r {
                Err(_) => { /* pinned: panics at HEAD (see doc-comment) */ }
                Ok(Err(e)) => panic!(
                    "{}: returned structured Err({e}) — #1746 appears \
                     fixed; retire this pin and assert the error kind",
                    $name
                ),
                Ok(Ok(())) => panic!(
                    "{}: returned Ok on a zero-length reduced slice — torch \
                     raises IndexError; neither the pinned panic nor the \
                     contractual Err",
                    $name
                ),
            }
        }};
    }

    let z = make_cpu_f32(&[], &[2, 0, 3], false);
    pin_panic!("amax_dim([2,0,3], dim=1)", amax_dim(&z, 1, false));
    pin_panic!("amin_dim([2,0,3], dim=1)", amin_dim(&z, 1, false));
    pin_panic!("argmax_dim([2,0,3], dim=1)", argmax_dim(&z, 1, false));
    pin_panic!("argmin_dim([2,0,3], dim=1)", argmin_dim(&z, 1, false));
    pin_panic!("max_with_dim([2,0,3], dim=1)", max_with_dim(&z, 1, false));
    pin_panic!("min_with_dim([2,0,3], dim=1)", min_with_dim(&z, 1, false));
    pin_panic!(
        "median_with_dim([2,0,3], dim=1)",
        median_with_dim(&z, 1, false)
    );
    pin_panic!(
        "nanmedian_with_dim([2,0,3], dim=1)",
        nanmedian_with_dim(&z, 1, false)
    );

    // The mechanism is dtype-generic (`<T: Float>`); one f64 lane guards
    // against a dtype-specialized early return sneaking in.
    let z64 = make_cpu_f64(&[], &[2, 0, 3], false);
    pin_panic!("amax_dim f64([2,0,3], dim=1)", amax_dim(&z64, 1, false));
    pin_panic!(
        "median_with_dim f64([2,0,3], dim=1)",
        median_with_dim(&z64, 1, false)
    );
}

/// Tie-mass distribution test for amax/amin: input `[1.0, 1.0, 1.0]`,
/// scalar grad_out=1, assert grad = `[1/3, 1/3, 1/3]` (PyTorch's
/// mass-distribution convention).
#[test]
fn cpu_amax_amin_tie_distribution() {
    let file = load_fixtures();
    for op_label in ["amax_ties", "amin_ties"] {
        for f in cases_for(&file, op_label, "cpu") {
            let label = format!("{op_label} cpu dtype={}", f.dtype);
            let shape = f.a_shape.as_ref().expect("a_shape");
            let a_data = f
                .a_data
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .expect("a_data");
            let grad_a_exp = f
                .grad_a
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .expect("grad_a");
            match f.dtype.as_str() {
                "float32" => {
                    let a_g = make_cpu_f32(a_data, shape, true);
                    let out = match op_label {
                        "amax_ties" => amax(&a_g).expect("amax"),
                        "amin_ties" => amin(&a_g).expect("amin"),
                        _ => unreachable!(),
                    };
                    out.backward().expect("backward");
                    let ga = a_g.grad().unwrap().expect("grad_a");
                    check_f32(
                        &format!("{label} grad_a"),
                        &read_back_f32(&ga, Device::Cpu),
                        grad_a_exp,
                        tolerance::F32_REDUCTION_CPU,
                    );
                }
                "float64" => {
                    let a_g = make_cpu_f64(a_data, shape, true);
                    let out = match op_label {
                        "amax_ties" => amax(&a_g).expect("amax"),
                        "amin_ties" => amin(&a_g).expect("amin"),
                        _ => unreachable!(),
                    };
                    out.backward().expect("backward");
                    let ga = a_g.grad().unwrap().expect("grad_a");
                    check_f64(
                        &format!("{label} grad_a"),
                        &read_back_f64(&ga, Device::Cpu),
                        grad_a_exp,
                        tolerance::F64_REDUCTION_CPU,
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cat B — cumulative forwards (cumsum / cumprod / logcumsumexp)
// ---------------------------------------------------------------------------
//
// For the differentiable trio (cumsum/cumprod/logcumsumexp), both forward and
// backward are exercised on the requested device. CUDA lanes assert the result
// and gradient stay CUDA-resident, matching PyTorch's no-host-round-trip
// contract for cumulative autograd.
//
// `cummax` / `cummin` are not differentiable at all (they return indices);
// we just compare values + indices.

#[derive(Clone, Copy)]
enum DiffCumOp {
    Cumsum,
    Cumprod,
    Logcumsumexp,
}

impl DiffCumOp {
    fn name(self) -> &'static str {
        match self {
            DiffCumOp::Cumsum => "cumsum",
            DiffCumOp::Cumprod => "cumprod",
            DiffCumOp::Logcumsumexp => "logcumsumexp",
        }
    }
    fn apply_f32(self, a: &Tensor<f32>, dim: i64) -> Tensor<f32> {
        match self {
            DiffCumOp::Cumsum => cumsum(a, dim).expect("cumsum"),
            DiffCumOp::Cumprod => cumprod(a, dim).expect("cumprod"),
            DiffCumOp::Logcumsumexp => logcumsumexp(a, dim).expect("logcumsumexp"),
        }
    }
    fn apply_f64(self, a: &Tensor<f64>, dim: i64) -> Tensor<f64> {
        match self {
            DiffCumOp::Cumsum => cumsum(a, dim).expect("cumsum"),
            DiffCumOp::Cumprod => cumprod(a, dim).expect("cumprod"),
            DiffCumOp::Logcumsumexp => logcumsumexp(a, dim).expect("logcumsumexp"),
        }
    }
}

fn cum_tolerance_f32(op: DiffCumOp, on_gpu: bool) -> f32 {
    match (op, on_gpu) {
        (DiffCumOp::Logcumsumexp, true) => tolerance::F32_LOGSCAN_GPU,
        (DiffCumOp::Logcumsumexp, false) => tolerance::F32_LOGSCAN_CPU,
        (_, true) => tolerance::F32_REDUCTION_GPU,
        (_, false) => tolerance::F32_REDUCTION_CPU,
    }
}

fn cum_tolerance_f64(op: DiffCumOp, on_gpu: bool) -> f64 {
    match (op, on_gpu) {
        (DiffCumOp::Logcumsumexp, true) => tolerance::F64_LOGSCAN_GPU,
        (DiffCumOp::Logcumsumexp, false) => tolerance::F64_LOGSCAN_CPU,
        (_, true) => tolerance::F64_REDUCTION_GPU,
        (_, false) => tolerance::F64_REDUCTION_CPU,
    }
}

fn run_diff_cum_for_device(op: DiffCumOp, device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, op.name(), device_label);
    assert!(
        !cases.is_empty(),
        "no fixtures for {} on {device_label}",
        op.name()
    );
    let on_gpu = matches!(device, Device::Cuda(_));
    let tol_f32 = cum_tolerance_f32(op, on_gpu);
    let tol_f64 = cum_tolerance_f64(op, on_gpu);

    for f in cases {
        if let Some(reason) = cascade_skip(op.name(), device_label, &f.dtype) {
            eprintln!(
                "skipping {} {device_label} dtype={} tag={:?}: {reason}",
                op.name(),
                f.dtype,
                f.tag,
            );
            continue;
        }
        let label = format!(
            "{} {device_label} tag={:?} dtype={}",
            op.name(),
            f.tag,
            f.dtype,
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
        let grad_a_exp = f
            .grad_a
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("grad_a");
        let axis = f.axis.expect("axis");

        match f.dtype.as_str() {
            "float32" => {
                // Forward: CPU + GPU (no autograd in this test arm).
                let a = upload_f32(make_cpu_f32(a_data, shape, false), device);
                let c = op.apply_f32(&a, axis);
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&c, device),
                    expected,
                    tol_f32,
                );

                let a_g = upload_f32(make_cpu_f32(a_data, shape, true), device);
                let out = op.apply_f32(&a_g, axis);
                let loss = sum(&out).expect("sum-to-scalar loss");
                loss.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                assert_eq!(ga.device(), device, "{label} grad device");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&ga, device),
                    grad_a_exp,
                    tolerance::F32_REDUCTION_CPU.max(tol_f32),
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(a_data, shape, false), device);
                let c = op.apply_f64(&a, axis);
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&c, device),
                    expected,
                    tol_f64,
                );

                let a_g = upload_f64(make_cpu_f64(a_data, shape, true), device);
                let out = op.apply_f64(&a_g, axis);
                let loss = sum(&out).expect("sum-to-scalar loss");
                loss.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                assert_eq!(ga.device(), device, "{label} grad device");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&ga, device),
                    grad_a_exp,
                    tolerance::F64_REDUCTION_CPU.max(tol_f64),
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_cumsum() {
    run_diff_cum_for_device(DiffCumOp::Cumsum, "cpu", Device::Cpu);
}

#[test]
fn cpu_cumprod() {
    run_diff_cum_for_device(DiffCumOp::Cumprod, "cpu", Device::Cpu);
}

#[test]
fn cpu_logcumsumexp() {
    run_diff_cum_for_device(DiffCumOp::Logcumsumexp, "cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// Cat B — cummax / cummin (non-differentiable: values + indices)
// ---------------------------------------------------------------------------
//
// `CumExtremeResult` carries `.values: Tensor<T>` and an authoritative
// `.indices_tensor: IntTensor<i64>`, matching PyTorch's NamedTuple of
// (values, indices). The legacy `.indices: Vec<usize>` host cache is populated
// only for CPU/scalar results; non-scalar CUDA results intentionally leave it
// empty so the forward path does not do an implicit D2H transfer.
//
// Base inputs in `_cumulative_input` use strictly distinct values along the
// scan dim so they pinpoint the values+indices contract without tie noise.
// The tie regime is covered by the dedicated `tie_*` fixtures (CORE-198 /
// #1892), generated from live torch. Probed semantics at HEAD
// (torch 2.11.0+cu130):
//   * torch keeps the LAST tied index on CPU and CUDA
//     (`torch.cummax(torch.tensor([1.,3.,3.,2.,3.]), 0).indices`
//      == `[0, 1, 2, 2, 4]`; `std::greater_equal` / `std::less_equal` in
//     ReduceOps.cpp `cummax_cummin_helper`).
//   * ferrotorch CPU MATCHES torch (the old "ferrotorch uses first-tie"
//     comment here was stale — the `>=`/`<=` tie-break landed with #1231).
//   * ferrotorch CUDA now matches torch as well: the PTX predicate mirrors
//     `isnan(curr) || (!isnan(out) && curr >= out)` / `<= out`, so ties keep
//     the LAST tied index.

fn indices_tensor_host(label: &str, indices_tensor: &IntTensor<i64>, device: Device) -> Vec<usize> {
    assert_eq!(
        indices_tensor.device(),
        device,
        "{label}: indices tensor device"
    );
    indices_tensor
        .to(Device::Cpu)
        .expect("indices to CPU")
        .data()
        .expect("indices data")
        .iter()
        .map(|&v| usize::try_from(v).expect("non-negative cummax/cummin index"))
        .collect()
}

/// Indices assertion for cummax/cummin: torch contract on CPU and CUDA.
fn check_cum_extreme_indices(
    label: &str,
    on_gpu: bool,
    device: Device,
    host_cache: &[usize],
    indices_tensor: &IntTensor<i64>,
    torch_expected: &[usize],
) {
    let actual = indices_tensor_host(label, indices_tensor, device);
    assert_eq!(
        actual.len(),
        torch_expected.len(),
        "{label}: indices length"
    );
    for (i, (got, exp)) in actual.iter().zip(torch_expected.iter()).enumerate() {
        assert_eq!(
            got, exp,
            "{label}: indices[{i}] mismatch (actual={got}, expected={exp})"
        );
    }
    if on_gpu {
        assert!(
            host_cache.is_empty(),
            "{label}: CUDA cummax/cummin must not populate a host indices cache"
        );
    } else {
        assert_eq!(host_cache, torch_expected, "{label}: host indices cache");
    }
}

fn run_cum_extreme_for_device(op_name: &str, device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, op_name, device_label);
    assert!(
        !cases.is_empty(),
        "no fixtures for {op_name} on {device_label}"
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
        if let Some(reason) = cascade_skip(op_name, device_label, &f.dtype) {
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
        let expected_vals = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("out_values");
        let expected_idx = f.out_indices.as_ref().expect("out_indices");
        let axis = f.axis.expect("axis");

        match f.dtype.as_str() {
            "float32" => {
                let a = upload_f32(make_cpu_f32(a_data, shape, false), device);
                let result: CumExtremeResult<f32> = match op_name {
                    "cummax" => cummax(&a, axis).expect("cummax"),
                    "cummin" => cummin(&a, axis).expect("cummin"),
                    _ => unreachable!(),
                };
                check_f32(
                    &format!("{label} values"),
                    &read_back_f32(&result.values, device),
                    expected_vals,
                    tol_f32,
                );
                check_cum_extreme_indices(
                    &label,
                    on_gpu,
                    device,
                    &result.indices,
                    &result.indices_tensor,
                    expected_idx,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(a_data, shape, false), device);
                let result: CumExtremeResult<f64> = match op_name {
                    "cummax" => cummax(&a, axis).expect("cummax"),
                    "cummin" => cummin(&a, axis).expect("cummin"),
                    _ => unreachable!(),
                };
                check_f64(
                    &format!("{label} values"),
                    &read_back_f64(&result.values, device),
                    expected_vals,
                    tol_f64,
                );
                check_cum_extreme_indices(
                    &label,
                    on_gpu,
                    device,
                    &result.indices,
                    &result.indices_tensor,
                    expected_idx,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_cummax() {
    run_cum_extreme_for_device("cummax", "cpu", Device::Cpu);
}

#[test]
fn cpu_cummin() {
    run_cum_extreme_for_device("cummin", "cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// Cat B — edge cases (cumprod-with-zero, logcumsumexp-stability)
// ---------------------------------------------------------------------------

#[test]
fn cpu_cumprod_with_zero() {
    let file = load_fixtures();
    for f in cases_for(&file, "cumprod_zero", "cpu") {
        let label = format!("cumprod_zero cpu dtype={}", f.dtype);
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
        let grad_a_exp = f
            .grad_a
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("grad_a");
        let axis = f.axis.expect("axis");
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape, false);
                let c = cumprod(&a, axis).expect("cumprod fwd");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&c, Device::Cpu),
                    expected,
                    tolerance::F32_REDUCTION_CPU,
                );
                let a_g = make_cpu_f32(a_data, shape, true);
                let out = cumprod(&a_g, axis).expect("cumprod grad");
                sum(&out)
                    .expect("sum-to-scalar")
                    .backward()
                    .expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&ga, Device::Cpu),
                    grad_a_exp,
                    tolerance::F32_REDUCTION_CPU,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let c = cumprod(&a, axis).expect("cumprod fwd");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&c, Device::Cpu),
                    expected,
                    tolerance::F64_REDUCTION_CPU,
                );
                let a_g = make_cpu_f64(a_data, shape, true);
                let out = cumprod(&a_g, axis).expect("cumprod grad");
                sum(&out)
                    .expect("sum-to-scalar")
                    .backward()
                    .expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&ga, Device::Cpu),
                    grad_a_exp,
                    tolerance::F64_REDUCTION_CPU,
                );
            }
            _ => unreachable!(),
        }
    }
}

/// logcumsumexp at saturated f32 magnitude: `[100.0, 100.0]` along dim=0
/// must NOT overflow the intermediate `exp(...)` and must produce
/// `[100.0, 100.0 + log(2)]`. This is the polynomial-cluster regression
/// test from the Dispatch C verification debt.
#[test]
fn cpu_logcumsumexp_overflow_stability() {
    let file = load_fixtures();
    for f in cases_for(&file, "logcumsumexp_overflow", "cpu") {
        let label = format!("logcumsumexp_overflow cpu dtype={}", f.dtype);
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
        let axis = f.axis.expect("axis");
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape, false);
                let c = logcumsumexp(&a, axis).expect("logcumsumexp");
                let actual = read_back_f32(&c, Device::Cpu);
                for v in &actual {
                    assert!(v.is_finite(), "{label}: produced non-finite {v}");
                }
                check_f32(&label, &actual, expected, tolerance::F32_LOGSCAN_CPU);
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let c = logcumsumexp(&a, axis).expect("logcumsumexp");
                let actual = read_back_f64(&c, Device::Cpu);
                for v in &actual {
                    assert!(v.is_finite(), "{label}: produced non-finite {v}");
                }
                check_f64(&label, &actual, expected, tolerance::F64_LOGSCAN_CPU);
            }
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// `reverse_cumsum` — raw-slice utility (CPU only by signature)
// ---------------------------------------------------------------------------
//
// `reverse_cumsum` takes `&[T]`, returns `Vec<T>`, and is used internally by
// `CumsumBackward` / `CumprodBackward` / `LogcumsumexpBackward`. It is not
// a Tensor op. Tests it directly with a synthetic 1-D input + manual
// reference: reverse cumsum of `[1, 2, 3, 4]` along the only dim is
// `[10, 9, 7, 4]` (suffix-sums).

#[test]
fn test_reverse_cumsum() {
    let data = [1.0_f64, 2.0, 3.0, 4.0];
    let result = reverse_cumsum(&data, &[4], 0);
    // reverse cumsum: [10, 9, 7, 4]
    assert_eq!(result, vec![10.0, 9.0, 7.0, 4.0]);

    // 2-D along dim=1: each row's suffix-sum.
    let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let result = reverse_cumsum(&data, &[2, 3], 1);
    // Row 0: [1+2+3, 2+3, 3] = [6, 5, 3]
    // Row 1: [4+5+6, 5+6, 6] = [15, 11, 6]
    assert_eq!(result, vec![6.0, 5.0, 3.0, 15.0, 11.0, 6.0]);

    // 2-D along dim=0: each column's suffix-sum.
    let result = reverse_cumsum(&data, &[2, 3], 0);
    // Col 0: [1+4, 4] = [5, 4]
    // Col 1: [2+5, 5] = [7, 5]
    // Col 2: [3+6, 6] = [9, 6]
    assert_eq!(result, vec![5.0, 7.0, 9.0, 4.0, 5.0, 6.0]);
}

// ---------------------------------------------------------------------------
// `CumExtremeResult` — direct field-access test (no GPU needed)
// ---------------------------------------------------------------------------
//
// Constructs the struct via `cummax_forward` and asserts:
//   * `.values` is a `Tensor<T>` with the same shape as the input.
//   * `.indices` is a `Vec<usize>` with `numel` entries.

#[test]
fn cum_extreme_result_struct_fields() {
    // 1-D ascending input -> running max equals the input itself, indices
    // are 0..n-1.
    let a = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], false);
    let r: CumExtremeResult<f32> = cummax_forward(&a, 0).expect("cummax_forward");
    assert_eq!(r.values.shape(), &[5]);
    assert_eq!(r.indices, vec![0, 1, 2, 3, 4]);
    let v = read_back_f32(&r.values, Device::Cpu);
    tolerance::assert_close_f32(
        &v,
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        tolerance::F32_REDUCTION_CPU,
        "cum_extreme values",
    );

    // Symmetric: 1-D descending input -> running min equals the input.
    let a = make_cpu_f64(&[5.0, 4.0, 3.0, 2.0, 1.0], &[5], false);
    let r: CumExtremeResult<f64> = cummin_forward(&a, 0).expect("cummin_forward");
    assert_eq!(r.values.shape(), &[5]);
    assert_eq!(r.indices, vec![0, 1, 2, 3, 4]);
    let v = read_back_f64(&r.values, Device::Cpu);
    tolerance::assert_close_f64(
        &v,
        &[5.0, 4.0, 3.0, 2.0, 1.0],
        tolerance::F64_REDUCTION_CPU,
        "cum_extreme values f64",
    );
}

// ---------------------------------------------------------------------------
// Cat C — forward-only helpers (`*_forward`) — implicit coverage smoke test
// ---------------------------------------------------------------------------
//
// `cumsum_forward` / `cumprod_forward` / `cummax_forward` / `cummin_forward`
// / `logcumsumexp_forward` are exercised transitively via the Cat B
// autograd path (which calls them through the differentiable wrappers). To
// also satisfy the surface-coverage substring grep we reference each
// `*_forward` by name here in a compact smoke test that runs the kernel
// once on a small 1-D fixture.

#[test]
fn forward_only_helpers_smoke() {
    let a = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[4], false);

    let cs = cumsum_forward(&a, 0).expect("cumsum_forward");
    tolerance::assert_close_f32(
        &read_back_f32(&cs, Device::Cpu),
        &[1.0, 3.0, 6.0, 10.0],
        tolerance::F32_REDUCTION_CPU,
        "cumsum_forward",
    );

    let cp = cumprod_forward(&a, 0).expect("cumprod_forward");
    tolerance::assert_close_f32(
        &read_back_f32(&cp, Device::Cpu),
        &[1.0, 2.0, 6.0, 24.0],
        tolerance::F32_REDUCTION_CPU,
        "cumprod_forward",
    );

    let cmax = cummax_forward(&a, 0).expect("cummax_forward");
    assert_eq!(cmax.indices, vec![0, 1, 2, 3]);

    let cmin = cummin_forward(&a, 0).expect("cummin_forward");
    assert_eq!(cmin.indices, vec![0, 0, 0, 0]);

    // logcumsumexp: pin the actual prefix log-sum-exp values, not just
    // monotonicity + finiteness. The expected array was computed at f32
    // precision via the numerically-stable shift form
    //   lc[i] = m + log(exp(lc[i-1] - m) + exp(a[i] - m))   m = max(...)
    // i.e.
    //   lc[0] = 1.0
    //   lc[1] = log(e + e^2)            ≈ 2.31326175
    //   lc[2] = log(e + e^2 + e^3)      ≈ 3.40760612
    //   lc[3] = log(e + e^2 + e^3 + e^4) ≈ 4.44018984
    // A finiteness+monotonicity-only check would let a stub returning
    // `[1.0, 1.5, 2.0, 2.5]` pass; pinning the values catches it.
    let lc = logcumsumexp_forward(&a, 0).expect("logcumsumexp_forward");
    let lc_v = read_back_f32(&lc, Device::Cpu);
    let expected: [f32; 4] = [1.0_f32, 2.313_261_7_f32, 3.407_606_1_f32, 4.440_19_f32];
    tolerance::assert_close_f32(
        &lc_v,
        &expected,
        tolerance::F32_REDUCTION_CPU,
        "logcumsumexp_forward",
    );
}

// ---------------------------------------------------------------------------
// Sanity: assert the fixture file has every op we expect.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CORE-199 / #1893 — special-value lanes
// ---------------------------------------------------------------------------
//
// Live-torch expectations (fixture). Pins (single contract, retire-on-fix,
// R-ORACLE-4):
//   * amax/amin NaN propagation        -> #1932 (torch: NaN; ferrotorch
//     skips NaN and returns a finite extremum)
//   * logcumsumexp inf scan poisoning  -> CORE-133 / #1827

#[test]
fn cpu_amax_amin_special() {
    let file = load_fixtures();
    for op_name in ["amax_special", "amin_special"] {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
            let label = format!("{op_name} cpu tag={:?} dtype={}", f.tag, f.dtype);
            let shape = f.a_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let exp = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            // The sv_nan_* rows expect torch's NaN; ferrotorch returns the
            // finite extremum (pin). The sv_inf row is finite-contract and
            // value-asserted.
            let pinned_nan = exp[0].is_nan();
            match f.dtype.as_str() {
                "float32" => {
                    let a = make_cpu_f32(a_data, shape, false);
                    let r = match op_name {
                        "amax_special" => amax(&a).expect("amax"),
                        _ => amin(&a).expect("amin"),
                    };
                    let actual = read_back_f32(&r, Device::Cpu);
                    if pinned_nan {
                        // #1932 pin: torch propagates NaN (fixture);
                        // ferrotorch's fold skips it. When #1932 lands this
                        // assert fails — retire the pin and let the fixture
                        // comparison below run for every row.
                        assert!(
                            !actual[0].is_nan(),
                            "{label}: result is now NaN — #1932 appears \
                             fixed; retire this pin and assert the fixture"
                        );
                    } else {
                        check_f32(&label, &actual, exp, tolerance::F32_REDUCTION_CPU);
                    }
                }
                "float64" => {
                    let a = make_cpu_f64(a_data, shape, false);
                    let r = match op_name {
                        "amax_special" => amax(&a).expect("amax"),
                        _ => amin(&a).expect("amin"),
                    };
                    let actual = read_back_f64(&r, Device::Cpu);
                    if pinned_nan {
                        // #1932 pin — same mechanism at f64.
                        assert!(
                            !actual[0].is_nan(),
                            "{label}: result is now NaN — #1932 appears \
                             fixed; retire this pin and assert the fixture"
                        );
                    } else {
                        check_f64(&label, &actual, exp, tolerance::F64_REDUCTION_CPU);
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn cpu_sum_mean_special() {
    let file = load_fixtures();
    for op_name in ["sum_special", "mean_special"] {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
            let label = format!("{op_name} cpu tag={:?} dtype={}", f.tag, f.dtype);
            let shape = f.a_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let exp = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            match f.dtype.as_str() {
                "float32" => {
                    let a = make_cpu_f32(a_data, shape, false);
                    let r = match op_name {
                        "sum_special" => sum(&a).expect("sum"),
                        _ => mean(&a).expect("mean"),
                    };
                    check_f32(
                        &label,
                        &read_back_f32(&r, Device::Cpu),
                        exp,
                        tolerance::F32_REDUCTION_CPU,
                    );
                }
                "float64" => {
                    let a = make_cpu_f64(a_data, shape, false);
                    let r = match op_name {
                        "sum_special" => sum(&a).expect("sum"),
                        _ => mean(&a).expect("mean"),
                    };
                    check_f64(
                        &label,
                        &read_back_f64(&r, Device::Cpu),
                        exp,
                        tolerance::F64_REDUCTION_CPU,
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn cpu_cummax_cummin_special() {
    let file = load_fixtures();
    for op_name in ["cummax_special", "cummin_special"] {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
            let label = format!("{op_name} cpu tag={:?} dtype={}", f.tag, f.dtype);
            let shape = f.a_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let expected_vals = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            let expected_idx = f.out_indices.as_ref().expect("out_indices");
            let axis = f.axis.expect("axis");
            // Probed at HEAD: ferrotorch's CPU scan matches torch's NaN
            // propagation exactly (values [1, nan, nan], indices [0, 1, 1])
            // — straight value+indices assertion.
            match f.dtype.as_str() {
                "float32" => {
                    let a = make_cpu_f32(a_data, shape, false);
                    let result: CumExtremeResult<f32> = match op_name {
                        "cummax_special" => cummax(&a, axis).expect("cummax"),
                        _ => cummin(&a, axis).expect("cummin"),
                    };
                    check_f32(
                        &format!("{label} values"),
                        &read_back_f32(&result.values, Device::Cpu),
                        expected_vals,
                        tolerance::F32_REDUCTION_CPU,
                    );
                    assert_eq!(&result.indices, expected_idx, "{label} indices");
                }
                "float64" => {
                    let a = make_cpu_f64(a_data, shape, false);
                    let result: CumExtremeResult<f64> = match op_name {
                        "cummax_special" => cummax(&a, axis).expect("cummax"),
                        _ => cummin(&a, axis).expect("cummin"),
                    };
                    check_f64(
                        &format!("{label} values"),
                        &read_back_f64(&result.values, Device::Cpu),
                        expected_vals,
                        tolerance::F64_REDUCTION_CPU,
                    );
                    assert_eq!(&result.indices, expected_idx, "{label} indices");
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn cpu_logcumsumexp_special() {
    let file = load_fixtures();
    let cases = cases_for(&file, "logcumsumexp_special", "cpu");
    assert!(!cases.is_empty(), "no fixtures for logcumsumexp_special");
    for f in cases {
        let label = format!("logcumsumexp_special cpu tag={:?} dtype={}", f.tag, f.dtype);
        let shape = f.a_shape.as_ref().unwrap();
        let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let exp = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let axis = f.axis.expect("axis");
        // #1827 (CORE-133) fixed: equal infinities pass through the scan via
        // the `_log_add_exp_helper` port (pytorch
        // `aten/src/ATen/native/cpu/LogAddExp.h:22-33`), matching torch's
        // [-inf, 0] -> [-inf, 0] and [0, inf] -> [0, inf]. Pin retired to a
        // live-torch fixture assertion.
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape, false);
                let l = logcumsumexp(&a, axis).expect("logcumsumexp");
                check_f32(
                    &label,
                    &read_back_f32(&l, Device::Cpu),
                    exp,
                    tolerance::F32_LOGSCAN_CPU,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let l = logcumsumexp(&a, axis).expect("logcumsumexp");
                check_f64(
                    &label,
                    &read_back_f64(&l, Device::Cpu),
                    exp,
                    tolerance::F64_LOGSCAN_CPU,
                );
            }
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// CORE-199 / #1893 — non-contiguous (transpose-view) lanes (CORE-132 / #1826)
// ---------------------------------------------------------------------------
//
// The fixture stores the contiguous base buffer; the runner builds the view
// with `.transpose(0, 1)` (input_transpose flag). Probed at HEAD:
//   * amax / amin / sum_dim / mean_dim ACCEPT views -> value-asserted.
//   * sum / mean / prod / cumsum / cummax reject ("tensor is not
//     contiguous") -> expect_err pins on #1826, retire-on-fix.

#[test]
fn cpu_transpose_view_lanes() {
    let file = load_fixtures();
    // Ops that accept views today: assert torch values.
    let accept_ops = [
        "amax_tview",
        "amin_tview",
        "sum_dim_tview",
        "mean_dim_tview",
    ];
    for op_name in accept_ops {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
            assert_eq!(f.input_transpose, Some(true), "{op_name}: missing flag");
            let label = format!("{op_name} cpu dtype={}", f.dtype);
            let shape = f.a_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let exp = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            match f.dtype.as_str() {
                "float32" => {
                    let v = make_cpu_f32(a_data, shape, false)
                        .transpose(0, 1)
                        .expect("transpose");
                    assert!(!v.is_contiguous(), "{label}: view must be non-contiguous");
                    let r = match op_name {
                        "amax_tview" => amax(&v).expect("amax"),
                        "amin_tview" => amin(&v).expect("amin"),
                        "sum_dim_tview" => sum_dim(&v, 0, false).expect("sum_dim"),
                        "mean_dim_tview" => mean_dim(&v, 0, false).expect("mean_dim"),
                        _ => unreachable!(),
                    };
                    check_f32(
                        &label,
                        &read_back_f32(&r, Device::Cpu),
                        exp,
                        tolerance::F32_REDUCTION_CPU,
                    );
                }
                "float64" => {
                    let v = make_cpu_f64(a_data, shape, false)
                        .transpose(0, 1)
                        .expect("transpose");
                    assert!(!v.is_contiguous(), "{label}: view must be non-contiguous");
                    let r = match op_name {
                        "amax_tview" => amax(&v).expect("amax"),
                        "amin_tview" => amin(&v).expect("amin"),
                        "sum_dim_tview" => sum_dim(&v, 0, false).expect("sum_dim"),
                        "mean_dim_tview" => mean_dim(&v, 0, false).expect("mean_dim"),
                        _ => unreachable!(),
                    };
                    check_f64(
                        &label,
                        &read_back_f64(&r, Device::Cpu),
                        exp,
                        tolerance::F64_REDUCTION_CPU,
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    let fixed_ops = [
        "sum_tview",
        "mean_tview",
        "prod_tview",
        "cumsum_tview",
        "cummax_tview",
    ];
    for op_name in fixed_ops {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
            assert_eq!(f.input_transpose, Some(true), "{op_name}: missing flag");
            let label = format!("{op_name} cpu dtype={}", f.dtype);
            let shape = f.a_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let exp = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            match f.dtype.as_str() {
                "float32" => {
                    let v = make_cpu_f32(a_data, shape, false)
                        .transpose(0, 1)
                        .expect("transpose");
                    assert!(!v.is_contiguous(), "{label}: view must be non-contiguous");
                    match op_name {
                        "cummax_tview" => {
                            let r = cummax(&v, 0).expect("cummax");
                            check_f32(
                                &label,
                                &read_back_f32(&r.values, Device::Cpu),
                                exp,
                                tolerance::F32_REDUCTION_CPU,
                            );
                            assert_eq!(
                                r.indices,
                                f.out_indices.as_ref().unwrap().as_slice(),
                                "{label}: cummax indices"
                            );
                        }
                        _ => {
                            let r = match op_name {
                                "sum_tview" => sum(&v).expect("sum"),
                                "mean_tview" => mean(&v).expect("mean"),
                                "prod_tview" => prod(&v).expect("prod"),
                                "cumsum_tview" => cumsum(&v, 0).expect("cumsum"),
                                _ => unreachable!(),
                            };
                            check_f32(
                                &label,
                                &read_back_f32(&r, Device::Cpu),
                                exp,
                                tolerance::F32_REDUCTION_CPU,
                            );
                        }
                    }
                }
                "float64" => {
                    let v = make_cpu_f64(a_data, shape, false)
                        .transpose(0, 1)
                        .expect("transpose");
                    assert!(!v.is_contiguous(), "{label}: view must be non-contiguous");
                    match op_name {
                        "cummax_tview" => {
                            let r = cummax(&v, 0).expect("cummax");
                            check_f64(
                                &label,
                                &read_back_f64(&r.values, Device::Cpu),
                                exp,
                                tolerance::F64_REDUCTION_CPU,
                            );
                            assert_eq!(
                                r.indices,
                                f.out_indices.as_ref().unwrap().as_slice(),
                                "{label}: cummax indices"
                            );
                        }
                        _ => {
                            let r = match op_name {
                                "sum_tview" => sum(&v).expect("sum"),
                                "mean_tview" => mean(&v).expect("mean"),
                                "prod_tview" => prod(&v).expect("prod"),
                                "cumsum_tview" => cumsum(&v, 0).expect("cumsum"),
                                _ => unreachable!(),
                            };
                            check_f64(
                                &label,
                                &read_back_f64(&r, Device::Cpu),
                                exp,
                                tolerance::F64_REDUCTION_CPU,
                            );
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn fixture_file_covers_every_phase22_op() {
    let file = load_fixtures();
    let mut by_op: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for f in &file.fixtures {
        *by_op.entry(f.op.as_str()).or_insert(0) += 1;
    }
    let required = [
        // Cat A — global reductions
        "sum",
        "mean",
        "prod",
        "amax",
        "amin",
        // Cat A — dim reductions
        "sum_dim",
        "mean_dim",
        // Cat A — edge cases
        "sum_empty",
        "mean_empty",
        "prod_empty",
        "amax_ties",
        "amin_ties",
        // Cat B — cumulative
        "cumsum",
        "cumprod",
        "cummax",
        "cummin",
        "logcumsumexp",
        // Cat B — edge cases
        "cumprod_zero",
        "logcumsumexp_overflow",
        // CORE-199 / #1893 lanes
        "amax_special",
        "amin_special",
        "sum_special",
        "mean_special",
        "cummax_special",
        "cummin_special",
        "logcumsumexp_special",
        "sum_tview",
        "mean_tview",
        "prod_tview",
        "amax_tview",
        "amin_tview",
        "sum_dim_tview",
        "mean_dim_tview",
        "cumsum_tview",
        "cummax_tview",
    ];
    for r in required {
        let n = by_op.get(r).copied().unwrap_or(0);
        assert!(n > 0, "fixture file missing op {r:?}");
    }

    // Tie-regime coverage (CORE-198 / #1892): cummax/cummin must carry the
    // dedicated tie fixtures on top of the distinct-value base set, so a
    // future regeneration cannot silently drop the tie regime again.
    for op in ["cummax", "cummin"] {
        for tag in ["tie_basic", "tie_allequal", "tie_mat2d_dim1"] {
            let n = file
                .fixtures
                .iter()
                .filter(|f| f.op == op && f.tag.as_deref() == Some(tag))
                .count();
            assert!(
                n > 0,
                "fixture file missing tie-regime fixture {op:?}/{tag:?} \
                 (CORE-198 / #1892)"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GPU paths — gated on the `gpu` feature
// ---------------------------------------------------------------------------
//
// Same dispatch pattern as elementwise/creation: gate on
// `#[cfg(feature = "gpu")]` rather than `#[ignore]` so a non-GPU build
// has these tests genuinely absent (not silently skipped).
//
// Per the dispatch:
//   * Reduction Cat A (sum/mean/prod/amax/amin/sum_dim/mean_dim) supports
//     forward + backward on GPU. ProdBackward routes to CPU internally
//     (and re-uploads the grad via `.to(device)`) — that's the source's
//     documented strategy. `AmaxBackward` / `AminBackward` do the same.
//   * Cumulative Cat B (cumsum/cumprod/cummax/cummin/logcumsumexp) keeps both
//     forward and autograd gradients resident on CUDA for supported dtypes.

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
                "fixtures/reduction.json was generated without CUDA — \
                 regenerate on a CUDA-enabled host before running --features gpu tests"
            );
        }
    }

    #[test]
    fn gpu_sum() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_global_reduction_for_device(GlobalReduction::Sum, "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_mean() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_global_reduction_for_device(GlobalReduction::Mean, "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_prod() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_global_reduction_for_device(GlobalReduction::Prod, "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_amax() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_global_reduction_for_device(GlobalReduction::Amax, "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_amin() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_global_reduction_for_device(GlobalReduction::Amin, "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_sum_dim() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_dim_reduction_for_device(DimReduction::SumDim, "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_mean_dim() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_dim_reduction_for_device(DimReduction::MeanDim, "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_cumsum_forward() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_diff_cum_for_device(DiffCumOp::Cumsum, "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_cumprod_forward() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_diff_cum_for_device(DiffCumOp::Cumprod, "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_logcumsumexp_forward() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_diff_cum_for_device(DiffCumOp::Logcumsumexp, "cuda:0", Device::Cuda(0));
    }

    /// CUDA special-value lanes for logcumsumexp (CORE-133 family).
    ///
    /// The CUDA PTX kernels mirror PyTorch's equal-infinity guard: equal
    /// infinities are carried through directly instead of evaluating
    /// `inf - inf` / `-inf - -inf` inside the stable rescaling formula.
    #[test]
    fn gpu_logcumsumexp_special() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        let cases = cases_for(&file, "logcumsumexp_special", "cuda:0");
        assert!(
            !cases.is_empty(),
            "no cuda fixtures for logcumsumexp_special"
        );
        for f in cases {
            let label = format!(
                "logcumsumexp_special cuda:0 tag={:?} dtype={}",
                f.tag, f.dtype
            );
            let shape = f.a_shape.as_ref().expect("a_shape");
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let exp = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            let axis = f.axis.expect("axis");
            match f.dtype.as_str() {
                "float32" => {
                    let a = upload_f32(make_cpu_f32(a_data, shape, false), Device::Cuda(0));
                    let l = logcumsumexp(&a, axis).expect("logcumsumexp");
                    let actual = read_back_f32(&l, Device::Cuda(0));
                    check_f32(&label, &actual, exp, tolerance::F32_LOGSCAN_GPU);
                }
                "float64" => {
                    let a = upload_f64(make_cpu_f64(a_data, shape, false), Device::Cuda(0));
                    let l = logcumsumexp(&a, axis).expect("logcumsumexp");
                    let actual = read_back_f64(&l, Device::Cuda(0));
                    check_f64(&label, &actual, exp, tolerance::F64_LOGSCAN_GPU);
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn gpu_cummax_forward() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_cum_extreme_for_device("cummax", "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_cummin_forward() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_cum_extreme_for_device("cummin", "cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_cummax_indices_tensor_and_backward_ties() {
        ensure_cuda_backend();
        let x = upload_f32(
            make_cpu_f32(&[1.0, 3.0, 3.0, 2.0, 3.0], &[5], true),
            Device::Cuda(0),
        );
        let result = cummax(&x, 0).expect("cummax");
        check_cum_extreme_indices(
            "gpu cummax tie indices",
            true,
            Device::Cuda(0),
            &result.indices,
            &result.indices_tensor,
            &[0, 1, 2, 2, 4],
        );
        assert_eq!(
            result.indices_host().expect("explicit host indices"),
            vec![0, 1, 2, 2, 4],
            "gpu cummax explicit indices_host"
        );

        sum(&result.values)
            .expect("sum values")
            .backward()
            .expect("cummax backward");
        let grad = x.grad().expect("grad access").expect("grad");
        check_f32(
            "gpu cummax tie backward",
            &read_back_f32(&grad, Device::Cuda(0)),
            &[1.0, 1.0, 2.0, 0.0, 1.0],
            tolerance::F32_REDUCTION_GPU,
        );
    }

    #[test]
    fn gpu_cummin_indices_tensor_and_backward_ties() {
        ensure_cuda_backend();
        let x = upload_f64(
            make_cpu_f64(&[3.0, 1.0, 1.0, 2.0, 1.0], &[5], true),
            Device::Cuda(0),
        );
        let result = cummin(&x, 0).expect("cummin");
        check_cum_extreme_indices(
            "gpu cummin tie indices",
            true,
            Device::Cuda(0),
            &result.indices,
            &result.indices_tensor,
            &[0, 1, 2, 2, 4],
        );
        assert_eq!(
            result.indices_host().expect("explicit host indices"),
            vec![0, 1, 2, 2, 4],
            "gpu cummin explicit indices_host"
        );

        sum(&result.values)
            .expect("sum values")
            .backward()
            .expect("cummin backward");
        let grad = x.grad().expect("grad access").expect("grad");
        check_f64(
            "gpu cummin tie backward",
            &read_back_f64(&grad, Device::Cuda(0)),
            &[1.0, 1.0, 2.0, 0.0, 1.0],
            tolerance::F64_REDUCTION_GPU,
        );
    }
}
