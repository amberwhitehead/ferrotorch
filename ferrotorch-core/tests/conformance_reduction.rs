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
    amax, amax_dim, amin, amin_dim, argmax_dim, argmin_dim, logsumexp_dims, max_with_dim, mean,
    mean_dim, mean_dims, median_with_dim, min_with_dim, nanmean_dim, nanmean_dims,
    nanmedian_with_dim, nansum_dim, nansum_dims, prod, std_dim, std_dims, sum, sum_dim, sum_dims,
    var_dim, var_dims,
};
use ferrotorch_core::int_tensor::IntTensor;
use ferrotorch_core::ops::cumulative::{
    CumExtremeResult, cummax_forward, cummin_forward, cumprod_forward, cumsum_forward,
    logcumsumexp_forward, reverse_cumsum,
};
use ferrotorch_core::{Device, FerrotorchError, Tensor, TensorStorage};

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

/// See [`read_back_f32`] - device-checked readback for f16 values, widened
/// only after the output has been proven resident on the expected device.
fn read_back_f16_as_f32(t: &Tensor<half::f16>, expect: Device) -> Vec<f32> {
    if expect.is_cuda() {
        assert_eq!(
            t.device(),
            expect,
            "result expected on {expect:?} but resides on {:?} - \
             silent CPU fallback (CORE-196 / #1890)",
            t.device()
        );
    }
    let cpu = if t.is_cpu() {
        t.clone()
    } else {
        t.cpu().expect("D2H readback")
    };
    cpu.data()
        .expect("read CPU f16 data")
        .iter()
        .map(|x| x.to_f32())
        .collect()
}

/// See [`read_back_f32`] - device-checked readback for bf16 values, widened
/// only after the output has been proven resident on the expected device.
fn read_back_bf16_as_f32(t: &Tensor<half::bf16>, expect: Device) -> Vec<f32> {
    if expect.is_cuda() {
        assert_eq!(
            t.device(),
            expect,
            "result expected on {expect:?} but resides on {:?} - \
             silent CPU fallback (CORE-196 / #1890)",
            t.device()
        );
    }
    let cpu = if t.is_cpu() {
        t.clone()
    } else {
        t.cpu().expect("D2H readback")
    };
    cpu.data()
        .expect("read CPU bf16 data")
        .iter()
        .map(|x| x.to_f32())
        .collect()
}

/// See [`read_back_f32`] - device-checked readback for reduction indices.
fn read_back_i64(t: &IntTensor<i64>, expect: Device) -> Vec<i64> {
    if expect.is_cuda() {
        assert_eq!(
            t.device(),
            expect,
            "indices expected on {expect:?} but reside on {:?} - \
             silent CPU fallback (CORE-196 / #1890)",
            t.device()
        );
    }
    t.to(Device::Cpu)
        .expect("indices to CPU")
        .data()
        .expect("read indices")
        .to_vec()
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

fn make_cpu_f16(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<half::f16> {
    let v: Vec<half::f16> = data
        .iter()
        .map(|&x| half::f16::from_f32(x as f32))
        .collect();
    Tensor::from_storage(TensorStorage::cpu(v), shape.to_vec(), requires_grad)
        .expect("make_cpu_f16")
}

fn make_cpu_bf16(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<half::bf16> {
    let v: Vec<half::bf16> = data
        .iter()
        .map(|&x| half::bf16::from_f32(x as f32))
        .collect();
    Tensor::from_storage(TensorStorage::cpu(v), shape.to_vec(), requires_grad)
        .expect("make_cpu_bf16")
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

fn upload_f16(t: Tensor<half::f16>, device: Device) -> Tensor<half::f16> {
    if matches!(device, Device::Cuda(_)) {
        let track = t.requires_grad();
        t.detach()
            .to(device)
            .expect("upload f16 to cuda")
            .requires_grad_(track)
    } else {
        t
    }
}

fn upload_bf16(t: Tensor<half::bf16>, device: Device) -> Tensor<half::bf16> {
    if matches!(device, Device::Cuda(_)) {
        let track = t.requires_grad();
        t.detach()
            .to(device)
            .expect("upload bf16 to cuda")
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

fn expect_invalid_arg_contains<T>(label: &str, result: Result<T, FerrotorchError>, needle: &str) {
    match result {
        Err(FerrotorchError::InvalidArgument { message }) => assert!(
            message.contains(needle),
            "{label}: InvalidArgument message {message:?} did not contain {needle:?}"
        ),
        Err(e) => panic!("{label}: expected InvalidArgument containing {needle:?}, got {e:?}"),
        Ok(_) => panic!("{label}: expected InvalidArgument containing {needle:?}, got Ok"),
    }
}

fn expect_invalid_arg_contains_any<T>(
    label: &str,
    result: Result<T, FerrotorchError>,
    needles: &[&str],
) {
    match result {
        Err(FerrotorchError::InvalidArgument { message }) => assert!(
            needles.iter().any(|needle| message.contains(needle)),
            "{label}: InvalidArgument message {message:?} did not contain any of {needles:?}"
        ),
        Err(e) => panic!("{label}: expected InvalidArgument containing {needles:?}, got {e:?}"),
        Ok(_) => panic!("{label}: expected InvalidArgument containing {needles:?}, got Ok"),
    }
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
/// PyTorch keeps gradients on the parameter device. CORE-196 / #1890 made this
/// assertion explicit, and #1922 is now fixed for global amin/amax: their CUDA
/// backward kernels return resident gradients instead of host tensors.
fn grad_device_for(
    _op: GlobalReduction,
    device: Device,
    label: &str,
    actual_grad_device: Device,
) -> Device {
    assert_eq!(
        actual_grad_device, device,
        "{label}: gradient device must match the input device"
    );
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

/// CORE-051 / #1745 — global `amax`/`amin` reject empty tensors without an
/// explicit dim.
///
/// torch oracle (live session, torch 2.11.0+cu130):
/// ```text
/// >>> torch.amax(torch.tensor([]))
/// RuntimeError: amax(): Expected reduction dim to be specified for
/// input.numel() == 0. Specify the reduction dim with the 'dim' argument.
/// ```
/// (`amin` identical.) The contractual ferrotorch behavior is a structured
/// `Err` before dispatch on every device.
#[test]
fn cpu_empty_amax_amin_errors_1745() {
    let a32 = make_cpu_f32(&[], &[0], false);
    expect_invalid_arg_contains("amax_f32([])", amax(&a32), "requires an explicit dim");
    expect_invalid_arg_contains("amin_f32([])", amin(&a32), "requires an explicit dim");
    let a64 = make_cpu_f64(&[], &[0], false);
    expect_invalid_arg_contains("amax_f64([])", amax(&a64), "requires an explicit dim");
    expect_invalid_arg_contains("amin_f64([])", amin(&a64), "requires an explicit dim");

    // Non-empty path (anti-stub): a stub returning an error for every input
    // would satisfy the empty-input assertions above. Pin the
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

/// CORE-052 / #1746 — dim-keyed value reductions reject zero-length reduced
/// slices with structured errors.
///
/// torch oracle (live session, torch 2.11.0+cu130, `z = torch.zeros(2,0,3)`):
/// ```text
/// >>> torch.amax(z, dim=1)
/// IndexError: amax(): Expected reduction dim 1 to have non-zero size.
/// ```
/// (amin/argmax/argmin/max/min identical with their own op names;
/// median/nanmedian both report `median(): Expected reduction dim 1 to have
/// non-zero size.`) Ferrotorch returns a structured `Err` from these
/// `FerrotorchResult` APIs instead of panicking or fabricating a value.
#[test]
fn cpu_zero_length_slice_dim_reductions_error_1746() {
    macro_rules! assert_empty_dim_err {
        ($name:literal, $call:expr) => {{ expect_invalid_arg_contains_any($name, $call, &["empty", "non-zero size"]) }};
    }

    let z = make_cpu_f32(&[], &[2, 0, 3], false);
    assert_empty_dim_err!("amax_dim([2,0,3], dim=1)", amax_dim(&z, 1, false));
    assert_empty_dim_err!("amin_dim([2,0,3], dim=1)", amin_dim(&z, 1, false));
    assert_empty_dim_err!("argmax_dim([2,0,3], dim=1)", argmax_dim(&z, 1, false));
    assert_empty_dim_err!("argmin_dim([2,0,3], dim=1)", argmin_dim(&z, 1, false));
    assert_empty_dim_err!("max_with_dim([2,0,3], dim=1)", max_with_dim(&z, 1, false));
    assert_empty_dim_err!("min_with_dim([2,0,3], dim=1)", min_with_dim(&z, 1, false));
    assert_empty_dim_err!(
        "median_with_dim([2,0,3], dim=1)",
        median_with_dim(&z, 1, false)
    );
    assert_empty_dim_err!(
        "nanmedian_with_dim([2,0,3], dim=1)",
        nanmedian_with_dim(&z, 1, false)
    );

    // The mechanism is dtype-generic (`<T: Float>`); one f64 lane guards
    // against a dtype-specialized early return sneaking in.
    let z64 = make_cpu_f64(&[], &[2, 0, 3], false);
    assert_empty_dim_err!("amax_dim f64([2,0,3], dim=1)", amax_dim(&z64, 1, false));
    assert_empty_dim_err!(
        "median_with_dim f64([2,0,3], dim=1)",
        median_with_dim(&z64, 1, false)
    );

    // PyTorch still rejects when another non-reduced axis is also zero:
    // shape [0,0,3], dim=1 has no output elements, but the selected value/index
    // would have to come from an empty reduced slice.
    let both_zero = make_cpu_f32(&[], &[0, 0, 3], false);
    assert_empty_dim_err!("amax_dim([0,0,3], dim=1)", amax_dim(&both_zero, 1, false));
    assert_empty_dim_err!("amin_dim([0,0,3], dim=1)", amin_dim(&both_zero, 1, false));
    assert_empty_dim_err!(
        "argmax_dim([0,0,3], dim=1)",
        argmax_dim(&both_zero, 1, false)
    );
    assert_empty_dim_err!(
        "argmin_dim([0,0,3], dim=1)",
        argmin_dim(&both_zero, 1, false)
    );
    assert_empty_dim_err!(
        "max_with_dim([0,0,3], dim=1)",
        max_with_dim(&both_zero, 1, false)
    );
    assert_empty_dim_err!(
        "min_with_dim([0,0,3], dim=1)",
        min_with_dim(&both_zero, 1, false)
    );
    assert_empty_dim_err!(
        "median_with_dim([0,0,3], dim=1)",
        median_with_dim(&both_zero, 1, false)
    );
    assert_empty_dim_err!(
        "nanmedian_with_dim([0,0,3], dim=1)",
        nanmedian_with_dim(&both_zero, 1, false)
    );
}

#[test]
fn cpu_non_reduced_zero_dim_reductions_return_empty_1746() {
    let leading_zero = make_cpu_f32(&[], &[0, 2, 3], false);

    let amin_v = amin_dim(&leading_zero, 1, false).expect("amin valid empty output");
    assert_eq!(amin_v.shape(), &[0, 3]);
    assert_eq!(read_back_f32(&amin_v, Device::Cpu), Vec::<f32>::new());
    let amax_v = amax_dim(&leading_zero, 1, true).expect("amax keepdim valid empty output");
    assert_eq!(amax_v.shape(), &[0, 1, 3]);
    assert_eq!(read_back_f32(&amax_v, Device::Cpu), Vec::<f32>::new());

    let argmax_i = argmax_dim(&leading_zero, 1, false).expect("argmax valid empty output");
    assert_eq!(argmax_i.shape(), &[0, 3]);
    assert_eq!(read_back_i64(&argmax_i, Device::Cpu), Vec::<i64>::new());
    let argmin_i = argmin_dim(&leading_zero, 1, true).expect("argmin keepdim valid empty output");
    assert_eq!(argmin_i.shape(), &[0, 1, 3]);
    assert_eq!(read_back_i64(&argmin_i, Device::Cpu), Vec::<i64>::new());

    let (max_v, max_i) = max_with_dim(&leading_zero, 1, false).expect("max valid empty output");
    assert_eq!(max_v.shape(), &[0, 3]);
    assert_eq!(max_i.shape(), &[0, 3]);
    assert_eq!(read_back_f32(&max_v, Device::Cpu), Vec::<f32>::new());
    assert_eq!(read_back_i64(&max_i, Device::Cpu), Vec::<i64>::new());

    let (min_v, min_i) =
        min_with_dim(&leading_zero, 1, true).expect("min keepdim valid empty output");
    assert_eq!(min_v.shape(), &[0, 1, 3]);
    assert_eq!(min_i.shape(), &[0, 1, 3]);
    assert_eq!(read_back_f32(&min_v, Device::Cpu), Vec::<f32>::new());
    assert_eq!(read_back_i64(&min_i, Device::Cpu), Vec::<i64>::new());

    let trailing_zero = make_cpu_f32(&[], &[2, 2, 0], false);
    let (median_v, median_i) =
        median_with_dim(&trailing_zero, 1, false).expect("median trailing-zero valid empty");
    assert_eq!(median_v.shape(), &[2, 0]);
    assert_eq!(median_i.shape(), &[2, 0]);
    assert_eq!(read_back_f32(&median_v, Device::Cpu), Vec::<f32>::new());
    assert_eq!(read_back_i64(&median_i, Device::Cpu), Vec::<i64>::new());

    let (nanmedian_v, nanmedian_i) =
        nanmedian_with_dim(&trailing_zero, 1, true).expect("nanmedian trailing-zero keepdim");
    assert_eq!(nanmedian_v.shape(), &[2, 1, 0]);
    assert_eq!(nanmedian_i.shape(), &[2, 1, 0]);
    assert_eq!(read_back_f32(&nanmedian_v, Device::Cpu), Vec::<f32>::new());
    assert_eq!(read_back_i64(&nanmedian_i, Device::Cpu), Vec::<i64>::new());
}

#[test]
fn cpu_scalar_median_dim_matches_torch_1746() {
    let x = make_cpu_f32(&[7.0], &[], true);
    let (median_v, median_i) = median_with_dim(&x, 0, true).expect("median scalar dim=0");
    assert_eq!(median_v.shape(), &[] as &[usize]);
    assert_eq!(median_i.shape(), &[] as &[usize]);
    assert_eq!(read_back_f32(&median_v, Device::Cpu), vec![7.0]);
    assert_eq!(read_back_i64(&median_i, Device::Cpu), vec![0]);
    median_v.backward().expect("median scalar backward");
    let grad = x.grad().expect("grad access").expect("grad");
    assert_eq!(read_back_f32(&grad, Device::Cpu), vec![1.0]);

    let y = make_cpu_f32(&[11.0], &[], false);
    let (nanmedian_v, nanmedian_i) =
        nanmedian_with_dim(&y, -1, false).expect("nanmedian scalar dim=-1");
    assert_eq!(nanmedian_v.shape(), &[] as &[usize]);
    assert_eq!(nanmedian_i.shape(), &[] as &[usize]);
    assert_eq!(read_back_f32(&nanmedian_v, Device::Cpu), vec![11.0]);
    assert_eq!(read_back_i64(&nanmedian_i, Device::Cpu), vec![0]);
}

#[test]
fn cpu_nan_and_multi_dim_reductions_match_torch_edges() {
    let x = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2], false);

    let summed = sum_dims(&x, &[0, -1], false).expect("sum_dims");
    assert_eq!(summed.shape(), &[2]);
    check_f32(
        "sum_dims",
        &read_back_f32(&summed, Device::Cpu),
        &[14.0, 22.0],
        tolerance::F32_REDUCTION_CPU,
    );

    let meaned = mean_dims(&x, &[0, 2], true).expect("mean_dims");
    assert_eq!(meaned.shape(), &[1, 2, 1]);
    check_f32(
        "mean_dims",
        &read_back_f32(&meaned, Device::Cpu),
        &[3.5, 5.5],
        tolerance::F32_REDUCTION_CPU,
    );

    let nan_x = make_cpu_f32(
        &[1.0, f64::NAN, 3.0, f64::NAN, f64::NAN, f64::NAN],
        &[2, 3],
        false,
    );
    let ns_dim = nansum_dim(&nan_x, 1, false).expect("nansum_dim");
    check_f32(
        "nansum_dim",
        &read_back_f32(&ns_dim, Device::Cpu),
        &[4.0, 0.0],
        tolerance::F32_REDUCTION_CPU,
    );
    let nm_dim = nanmean_dim(&nan_x, 1, false).expect("nanmean_dim");
    let nm_dim_values = read_back_f32(&nm_dim, Device::Cpu);
    assert_eq!(nm_dim_values.len(), 2);
    assert!((nm_dim_values[0] - 2.0).abs() < tolerance::F32_REDUCTION_CPU);
    assert!(nm_dim_values[1].is_nan(), "nanmean_dim all-NaN lane");

    let nan_multi = make_cpu_f32(
        &[1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0, f64::NAN, 8.0],
        &[2, 2, 2],
        false,
    );
    check_f32(
        "nansum_dims",
        &read_back_f32(
            &nansum_dims(&nan_multi, &[0, 2], false).expect("nansum_dims"),
            Device::Cpu,
        ),
        &[12.0, 15.0],
        tolerance::F32_REDUCTION_CPU,
    );
    check_f32(
        "nanmean_dims",
        &read_back_f32(
            &nanmean_dims(&nan_multi, &[0, 2], false).expect("nanmean_dims"),
            Device::Cpu,
        ),
        &[4.0, 5.0],
        tolerance::F32_REDUCTION_CPU,
    );

    let y = make_cpu_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
    check_f64(
        "var_dim",
        &read_back_f64(&var_dim(&y, 1, 0.0, false).expect("var_dim"), Device::Cpu),
        &[0.25, 0.25],
        tolerance::F64_REDUCTION_CPU,
    );
    check_f64(
        "std_dim",
        &read_back_f64(&std_dim(&y, 1, 0.0, false).expect("std_dim"), Device::Cpu),
        &[0.5, 0.5],
        tolerance::F64_REDUCTION_CPU,
    );
    check_f64(
        "var_dims",
        &read_back_f64(
            &var_dims(&y, &[0, 1], 0.0, false).expect("var_dims"),
            Device::Cpu,
        ),
        &[1.25],
        tolerance::F64_REDUCTION_CPU,
    );
    check_f64(
        "std_dims",
        &read_back_f64(
            &std_dims(&y, &[0, 1], 0.0, false).expect("std_dims"),
            Device::Cpu,
        ),
        &[1.25_f64.sqrt()],
        tolerance::F64_REDUCTION_CPU,
    );

    let lse_input = make_cpu_f64(&[0.0, 1.0, 2.0, 3.0], &[2, 2], false);
    let lse_expected = (0.0_f64.exp() + 1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
    check_f64(
        "logsumexp_dims",
        &read_back_f64(
            &logsumexp_dims(&lse_input, &[0, 1], false).expect("logsumexp_dims"),
            Device::Cpu,
        ),
        &[lse_expected],
        tolerance::F64_LOGSCAN_CPU,
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
// Live-torch expectations (fixture):
//   * amax/amin NaN propagation        -> #1932
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
            match f.dtype.as_str() {
                "float32" => {
                    let a = make_cpu_f32(a_data, shape, false);
                    let r = match op_name {
                        "amax_special" => amax(&a).expect("amax"),
                        _ => amin(&a).expect("amin"),
                    };
                    let actual = read_back_f32(&r, Device::Cpu);
                    check_f32(&label, &actual, exp, tolerance::F32_REDUCTION_CPU);
                }
                "float64" => {
                    let a = make_cpu_f64(a_data, shape, false);
                    let r = match op_name {
                        "amax_special" => amax(&a).expect("amax"),
                        _ => amin(&a).expect("amin"),
                    };
                    let actual = read_back_f64(&r, Device::Cpu);
                    check_f64(&label, &actual, exp, tolerance::F64_REDUCTION_CPU);
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
    fn gpu_zero_length_slice_dim_reductions_error_1746() {
        ensure_cuda_backend();
        let z = upload_f32(make_cpu_f32(&[], &[0, 0, 3], false), Device::Cuda(0));
        expect_invalid_arg_contains_any(
            "cuda amax_dim([0,0,3], dim=1)",
            amax_dim(&z, 1, false),
            &["non-zero size"],
        );
        expect_invalid_arg_contains_any(
            "cuda amin_dim([0,0,3], dim=1)",
            amin_dim(&z, 1, false),
            &["non-zero size"],
        );
        expect_invalid_arg_contains_any(
            "cuda argmax_dim([0,0,3], dim=1)",
            argmax_dim(&z, 1, false),
            &["non-zero size"],
        );
        expect_invalid_arg_contains_any(
            "cuda argmin_dim([0,0,3], dim=1)",
            argmin_dim(&z, 1, false),
            &["non-zero size"],
        );
        expect_invalid_arg_contains_any(
            "cuda max_with_dim([0,0,3], dim=1)",
            max_with_dim(&z, 1, false),
            &["non-zero size"],
        );
        expect_invalid_arg_contains_any(
            "cuda min_with_dim([0,0,3], dim=1)",
            min_with_dim(&z, 1, false),
            &["non-zero size"],
        );
        expect_invalid_arg_contains_any(
            "cuda median_with_dim([0,0,3], dim=1)",
            median_with_dim(&z, 1, false),
            &["non-zero size"],
        );
        expect_invalid_arg_contains_any(
            "cuda nanmedian_with_dim([0,0,3], dim=1)",
            nanmedian_with_dim(&z, 1, false),
            &["non-zero size"],
        );
    }

    #[test]
    fn gpu_non_reduced_zero_dim_reductions_return_resident_empty_1746() {
        ensure_cuda_backend();
        let leading_zero = upload_f32(make_cpu_f32(&[], &[0, 2, 3], false), Device::Cuda(0));

        let amin_v = amin_dim(&leading_zero, 1, false).expect("cuda amin valid empty output");
        assert_eq!(amin_v.shape(), &[0, 3]);
        assert_eq!(read_back_f32(&amin_v, Device::Cuda(0)), Vec::<f32>::new());
        let amax_v = amax_dim(&leading_zero, 1, true).expect("cuda amax keepdim empty output");
        assert_eq!(amax_v.shape(), &[0, 1, 3]);
        assert_eq!(read_back_f32(&amax_v, Device::Cuda(0)), Vec::<f32>::new());

        let argmax_i = argmax_dim(&leading_zero, 1, false).expect("cuda argmax valid empty output");
        assert_eq!(argmax_i.shape(), &[0, 3]);
        assert_eq!(read_back_i64(&argmax_i, Device::Cuda(0)), Vec::<i64>::new());
        let argmin_i =
            argmin_dim(&leading_zero, 1, true).expect("cuda argmin keepdim empty output");
        assert_eq!(argmin_i.shape(), &[0, 1, 3]);
        assert_eq!(read_back_i64(&argmin_i, Device::Cuda(0)), Vec::<i64>::new());

        let trailing_zero = upload_f32(make_cpu_f32(&[], &[2, 2, 0], false), Device::Cuda(0));

        let summed = sum_dim(&trailing_zero, 1, false).expect("cuda sum trailing-zero output");
        assert_eq!(summed.shape(), &[2, 0]);
        assert_eq!(read_back_f32(&summed, Device::Cuda(0)), Vec::<f32>::new());

        let amin_t = amin_dim(&trailing_zero, 1, false).expect("cuda amin trailing-zero output");
        assert_eq!(amin_t.shape(), &[2, 0]);
        assert_eq!(read_back_f32(&amin_t, Device::Cuda(0)), Vec::<f32>::new());
        let amax_t = amax_dim(&trailing_zero, 1, true).expect("cuda amax trailing-zero keepdim");
        assert_eq!(amax_t.shape(), &[2, 1, 0]);
        assert_eq!(read_back_f32(&amax_t, Device::Cuda(0)), Vec::<f32>::new());

        let (max_v, max_i) =
            max_with_dim(&trailing_zero, 1, false).expect("cuda max trailing-zero output");
        assert_eq!(max_v.shape(), &[2, 0]);
        assert_eq!(max_i.shape(), &[2, 0]);
        assert_eq!(read_back_f32(&max_v, Device::Cuda(0)), Vec::<f32>::new());
        assert_eq!(read_back_i64(&max_i, Device::Cuda(0)), Vec::<i64>::new());

        let (min_v, min_i) =
            min_with_dim(&trailing_zero, 1, true).expect("cuda min trailing-zero keepdim");
        assert_eq!(min_v.shape(), &[2, 1, 0]);
        assert_eq!(min_i.shape(), &[2, 1, 0]);
        assert_eq!(read_back_f32(&min_v, Device::Cuda(0)), Vec::<f32>::new());
        assert_eq!(read_back_i64(&min_i, Device::Cuda(0)), Vec::<i64>::new());

        let (median_v, median_i) =
            median_with_dim(&trailing_zero, 1, false).expect("cuda median trailing-zero output");
        assert_eq!(median_v.shape(), &[2, 0]);
        assert_eq!(median_i.shape(), &[2, 0]);
        assert_eq!(read_back_f32(&median_v, Device::Cuda(0)), Vec::<f32>::new());
        assert_eq!(read_back_i64(&median_i, Device::Cuda(0)), Vec::<i64>::new());

        let (nanmedian_v, nanmedian_i) = nanmedian_with_dim(&trailing_zero, 1, true)
            .expect("cuda nanmedian trailing-zero keepdim");
        assert_eq!(nanmedian_v.shape(), &[2, 1, 0]);
        assert_eq!(nanmedian_i.shape(), &[2, 1, 0]);
        assert_eq!(
            read_back_f32(&nanmedian_v, Device::Cuda(0)),
            Vec::<f32>::new()
        );
        assert_eq!(
            read_back_i64(&nanmedian_i, Device::Cuda(0)),
            Vec::<i64>::new()
        );
    }

    #[test]
    fn gpu_scalar_median_dim_matches_torch_1746() {
        ensure_cuda_backend();
        let x = upload_f32(make_cpu_f32(&[5.0], &[], true), Device::Cuda(0));
        let (median_v, median_i) = median_with_dim(&x, 0, true).expect("cuda scalar median");
        assert_eq!(median_v.shape(), &[] as &[usize]);
        assert_eq!(median_i.shape(), &[] as &[usize]);
        assert_eq!(read_back_f32(&median_v, Device::Cuda(0)), vec![5.0]);
        assert_eq!(read_back_i64(&median_i, Device::Cuda(0)), vec![0]);
        median_v.backward().expect("cuda scalar median backward");
        let grad = x.grad().expect("grad access").expect("grad");
        assert_eq!(read_back_f32(&grad, Device::Cuda(0)), vec![1.0]);

        let y = upload_f32(make_cpu_f32(&[13.0], &[], false), Device::Cuda(0));
        let (nanmedian_v, nanmedian_i) =
            nanmedian_with_dim(&y, -1, false).expect("cuda scalar nanmedian");
        assert_eq!(nanmedian_v.shape(), &[] as &[usize]);
        assert_eq!(nanmedian_i.shape(), &[] as &[usize]);
        assert_eq!(read_back_f32(&nanmedian_v, Device::Cuda(0)), vec![13.0]);
        assert_eq!(read_back_i64(&nanmedian_i, Device::Cuda(0)), vec![0]);
    }

    #[test]
    fn gpu_median_dim_matches_torch_nan_and_backward_1974() {
        ensure_cuda_backend();

        let x = upload_f32(
            make_cpu_f32(
                &[3.0, 1.0, 2.0, 4.0, f64::NAN, 2.0, 2.0, 1.0],
                &[2, 4],
                true,
            ),
            Device::Cuda(0),
        );
        let (median_v, median_i) = median_with_dim(&x, 1, false).expect("cuda median dim");
        assert_eq!(median_v.shape(), &[2]);
        assert_eq!(median_i.shape(), &[2]);
        let med = read_back_f32(&median_v, Device::Cuda(0));
        assert_eq!(med[0], 2.0);
        assert!(med[1].is_nan());
        assert_eq!(read_back_i64(&median_i, Device::Cuda(0)), vec![2, 0]);

        let (nanmedian_v, nanmedian_i) =
            nanmedian_with_dim(&x, 1, false).expect("cuda nanmedian dim");
        assert_eq!(read_back_f32(&nanmedian_v, Device::Cuda(0)), vec![2.0, 2.0]);
        assert_eq!(read_back_i64(&nanmedian_i, Device::Cuda(0)), vec![2, 1]);

        let ties = upload_f32(
            make_cpu_f32(&[2.0, 2.0, 2.0, 2.0], &[1, 4], true),
            Device::Cuda(0),
        );
        let (ties_v, ties_i) = median_with_dim(&ties, 1, false).expect("cuda median ties");
        assert_eq!(read_back_f32(&ties_v, Device::Cuda(0)), vec![2.0]);
        assert_eq!(read_back_i64(&ties_i, Device::Cuda(0)), vec![0]);
        ties_v.backward().expect("cuda median ties backward");
        let ties_grad = ties.grad().expect("grad access").expect("grad");
        assert_eq!(
            read_back_f32(&ties_grad, Device::Cuda(0)),
            vec![1.0, 0.0, 0.0, 0.0]
        );

        let y = upload_f64(
            make_cpu_f64(&[4.0, 2.0, 1.0, 3.0], &[1, 4], true),
            Device::Cuda(0),
        );
        let (y_median, y_idx) = median_with_dim(&y, 1, false).expect("cuda f64 median dim");
        assert_eq!(read_back_f64(&y_median, Device::Cuda(0)), vec![2.0]);
        assert_eq!(read_back_i64(&y_idx, Device::Cuda(0)), vec![1]);
        sum(&y_median)
            .expect("sum f64 median")
            .backward()
            .expect("f64 median backward");
        let y_grad = y.grad().expect("grad access").expect("grad");
        assert_eq!(
            read_back_f64(&y_grad, Device::Cuda(0)),
            vec![0.0, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn gpu_median_dim_half_bfloat_resident_backward_1974() {
        ensure_cuda_backend();

        let x16 = upload_f16(
            make_cpu_f16(&[4.0, 2.0, 1.0, 3.0], &[1, 4], true),
            Device::Cuda(0),
        );
        let (v16, i16) = median_with_dim(&x16, 1, false).expect("cuda f16 median dim");
        assert_eq!(read_back_f16_as_f32(&v16, Device::Cuda(0)), vec![2.0]);
        assert_eq!(read_back_i64(&i16, Device::Cuda(0)), vec![1]);
        sum(&v16)
            .expect("sum f16 median")
            .backward()
            .expect("f16 median backward");
        let g16 = x16.grad().expect("grad access").expect("grad");
        assert_eq!(
            read_back_f16_as_f32(&g16, Device::Cuda(0)),
            vec![0.0, 1.0, 0.0, 0.0]
        );

        let ties16 = upload_f16(
            make_cpu_f16(&[2.0, 2.0, 2.0, 2.0], &[1, 4], false),
            Device::Cuda(0),
        );
        let (ties16_v, ties16_i) =
            median_with_dim(&ties16, 1, false).expect("cuda f16 median ties");
        assert_eq!(read_back_f16_as_f32(&ties16_v, Device::Cuda(0)), vec![2.0]);
        assert_eq!(read_back_i64(&ties16_i, Device::Cuda(0)), vec![0]);

        let xb = upload_bf16(
            make_cpu_bf16(&[f64::NAN, 4.0, 2.0, 3.0], &[1, 4], true),
            Device::Cuda(0),
        );
        let (vb, ib) = nanmedian_with_dim(&xb, 1, false).expect("cuda bf16 nanmedian dim");
        assert_eq!(read_back_bf16_as_f32(&vb, Device::Cuda(0)), vec![3.0]);
        assert_eq!(read_back_i64(&ib, Device::Cuda(0)), vec![3]);
        sum(&vb)
            .expect("sum bf16 nanmedian")
            .backward()
            .expect("bf16 nanmedian backward");
        let gb = xb.grad().expect("grad access").expect("grad");
        assert_eq!(
            read_back_bf16_as_f32(&gb, Device::Cuda(0)),
            vec![0.0, 0.0, 0.0, 1.0]
        );
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
