//! Conformance Phase 2.1 — `ferrotorch-core` elementwise + inplace parity
//! against PyTorch.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/763>.
//! Parent: #759.
//!
//! Source files exercised:
//! - `ferrotorch-core/src/grad_fns/arithmetic.rs` (Cat A/G)
//! - `ferrotorch-core/src/grad_fns/comparison.rs` (Cat B/G)
//! - `ferrotorch-core/src/ops/elementwise.rs` (Cat C/D/E)
//! - `ferrotorch-core/src/inplace.rs` (Cat F)
//!
//! For each op the test loads PyTorch reference fixtures from
//! `tests/conformance/fixtures/elementwise.json` and asserts ferrotorch's
//! output matches within the tolerance dictated by the op's category. The
//! tolerance helpers from `conformance_creation.rs` are intentionally
//! re-implemented here as a private `tolerance` module — that keeps the
//! tests independently buildable without cross-test imports while preserving
//! identical bounds (any drift would show up against the same fixture).
//!
//! Coverage dimensions:
//! * **CPU forward** for every op.
//! * **CPU autograd** for the differentiable ops in Cat A and Cat B (loss =
//!   `sum(out)`, then `backward()`, compare grads to fixture).
//! * **GPU forward + autograd** under `#[cfg(feature = "gpu")]` for ops that
//!   declare GPU dispatch (Cat A/B/E sum/sum_axis/mean — anything where
//!   `is_cuda()` doesn't return `NotImplementedOnCuda`).
//! * **Higher-order Cat C** is CPU-only by design (closures don't dispatch
//!   to GPU); the GPU exclusion is documented in
//!   `_surface_exclusions.toml`.
//! * **Perf Cat D** is CPU-only by design (fast/SIMD CPU variants); each
//!   compared to (i) `torch.<canonical>` from the fixture and (ii)
//!   ferrotorch's own canonical version for internal consistency.
//! * **In-place Cat F**: forward conformance + storage-identity contract —
//!   after the op the tensor's `inner_storage_arc` Arc pointer is unchanged
//!   (the buffer was mutated, not replaced).
//! * **Cat G** (backward grad_fn structs `AddBackward`, `SubBackward`, ...,
//!   `WhereBackward`) are tested implicitly via the Cat A / Cat B autograd
//!   paths above. Their exclusion entries in `_surface_exclusions.toml` are
//!   updated to that framing rather than re-stated as phantom tests.

use std::path::PathBuf;
use std::sync::Arc;

use serde::de::{self, Deserializer, SeqAccess, Visitor};

use ferrotorch_core::grad_fns::arithmetic::{abs, add, div, mul, neg, pow, sqrt, sub};
use ferrotorch_core::grad_fns::comparison::{where_, where_bt};
use ferrotorch_core::grad_fns::reduction::{mean as grad_mean, sum as grad_sum};
use ferrotorch_core::ops::elementwise::{
    binary_map, fast_add, fast_cos, fast_div, fast_exp, fast_log, fast_mul, fast_sigmoid, fast_sin,
    fast_sub, fast_tanh, logsumexp, logsumexp_dim, mean, nanmean, nansum, scalar_map, simd_add_f32,
    simd_add_f64, simd_exp_f32, simd_exp_f64, simd_log_f32, simd_mul_f32, simd_mul_f64,
    simd_sqrt_f32, sum, sum_axis, unary_map,
};
use ferrotorch_core::{BoolTensor, Device, FerrotorchError, Tensor, TensorStorage};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Tolerance helpers — same shape as `conformance_creation.rs`. Repeated here
// so the tests stay independently buildable and so phase-specific bounds
// (1 ULP for add/mul/sub/neg, 1e-5 rel for sqrt/pow, 1e-6 rel for fast_*,
// statistical bounds for SIMD) can be tightened without touching phase 2.0.
// ---------------------------------------------------------------------------

mod tolerance {
    /// 1 ULP at f32 magnitude ~1 is ~1.2e-7. Use 1e-6 as a safe round for
    /// elementwise additive ops (add/sub/neg) where rounding is bounded.
    pub const F32_ELEMENTWISE: f32 = 1e-6;
    pub const F64_ELEMENTWISE: f64 = 1e-12;

    /// Multiplicative ops accumulate one rounding step; pow/sqrt/div add a
    /// transcendental rounding step (libm-bounded). 1e-5 rel matches the
    /// transcendental bound used in phase 2.0.
    pub const F32_TRANSCENDENTAL: f32 = 1e-5;
    pub const F64_TRANSCENDENTAL: f64 = 1e-10;

    /// fast_* perf variants: numerical fastmath, slightly looser than
    /// canonical libm. 1e-5 rel keeps us out of the auto-vectorization-noise
    /// band for f32; for f64 fast_* delegates to libm so the bound stays
    /// tight.
    pub const F32_FAST: f32 = 1e-5;
    pub const F64_FAST: f64 = 1e-10;

    /// SIMD ops are deterministic and bit-equal modulo a single rounding;
    /// 1e-6 rel covers the f32 ulp band, 1e-12 the f64 band.
    pub const F32_SIMD: f32 = 1e-6;
    pub const F64_SIMD: f64 = 1e-12;

    /// Reductions: small accumulated rounding from repeated adds. f32 sum of
    /// 24 elements is ~24 ulps off; 1e-5 rel keeps headroom.
    pub const F32_REDUCTION: f32 = 1e-5;
    pub const F64_REDUCTION: f64 = 1e-12;

    /// Accumulation-aware reduction tolerance (R-ORACLE-5, CORE-199 sweep
    /// lanes with k = 4096 / 10007 summands).
    ///
    /// Analytic justification: torch reduces with pairwise summation
    /// (error O(eps·log2 k)); ferrotorch folds sequentially (error bound
    /// O(eps·k), expected O(eps·sqrt(k)) under the standard random-rounding
    /// model — Higham, *Accuracy and Stability of Numerical Algorithms*,
    /// §4.2). The ORDER difference between the two is therefore expected
    /// O(eps·sqrt(k)) with a deterministic ceiling of O(eps·k); the factor
    /// 8 covers the constant without admitting the k·eps worst case. For
    /// small k the base band (`F32_REDUCTION`, itself justified above)
    /// dominates via `max`, so legacy rows keep their original bound.
    pub fn accum_tol_f32(k: usize) -> f32 {
        F32_REDUCTION.max(8.0 * (k as f32).sqrt() * f32::EPSILON)
    }

    /// See [`accum_tol_f32`]; same model at f64 epsilon.
    pub fn accum_tol_f64(k: usize) -> f64 {
        F64_REDUCTION.max(8.0 * (k as f64).sqrt() * f64::EPSILON)
    }

    /// GPU paths run the same kernels but through the cudarc dispatch; the
    /// reduction order may differ and cuBLAS-style accumulation introduces
    /// extra rounding.
    #[allow(dead_code, reason = "used by `gpu` cfg-gated module")]
    pub const F32_GPU: f32 = 1e-5;
    #[allow(dead_code, reason = "used by `gpu` cfg-gated module")]
    pub const F64_GPU: f64 = 1e-10;

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
                // ±inf with same sign: also accept.
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
// Strict-JSON-compatible f64 list deserializer.
//
// The Python regen script encodes +inf / -inf / NaN as the string sentinels
// "Infinity" / "-Infinity" / "NaN" so the fixture file stays strict-JSON
// compliant (serde_json rejects bare Infinity/NaN tokens). This visitor
// accepts both raw f64 numbers and those string sentinels.
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
    #[serde(default)]
    grad_x: Option<F64ListSentinel>,
    #[serde(default)]
    grad_y: Option<F64ListSentinel>,
    #[serde(default)]
    cond: Option<Vec<bool>>,
    #[serde(default)]
    x_shape: Option<Vec<usize>>,
    #[serde(default)]
    y_shape: Option<Vec<usize>>,
    #[serde(default)]
    x_data: Option<F64ListSentinel>,
    #[serde(default)]
    y_data: Option<F64ListSentinel>,
    #[serde(default)]
    exp: Option<f64>,
    #[serde(default)]
    scalar: Option<f64>,
    #[serde(default)]
    axis: Option<usize>,
    #[serde(default)]
    keepdim: Option<bool>,
    #[serde(default)]
    min: Option<f64>,
    #[serde(default)]
    max: Option<f64>,
    /// CORE-199 / #1893 non-contiguous lane: when `true`, `a_data`/`b_data`
    /// are the CONTIGUOUS row-major base buffers and the runner applies
    /// `.transpose(0, 1)` to build the non-contiguous view the op consumes.
    #[serde(default)]
    input_transpose: Option<bool>,
}

fn load_fixtures() -> FixtureFile {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures")
        .join("elementwise.json");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. Regenerate via \
             `python3 scripts/regenerate_elementwise_fixtures.py`",
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
// Device-transparent read-back helpers (CPU clone or D2H readback).
// ---------------------------------------------------------------------------

fn read_back_f32(t: &Tensor<f32>) -> Vec<f32> {
    if t.is_cpu() {
        t.data().expect("read CPU data").to_vec()
    } else {
        let cpu = t.cpu().expect("D2H readback");
        cpu.data().expect("read CPU data after readback").to_vec()
    }
}

fn read_back_f64(t: &Tensor<f64>) -> Vec<f64> {
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
// Cat A — differentiable arithmetic (CPU)
// ---------------------------------------------------------------------------
//
// Each op iterates the fixture cases for both dtypes, reconstructs the input
// pair, runs the differentiable ferrotorch op with `requires_grad=true`,
// then asserts forward values + grads match. Broadcasting is exercised by
// the BROADCAST_PAIRS in the fixture.

fn run_binary_cpu(op_name: &str, op: BinaryOp) {
    let file = load_fixtures();
    let cases = cases_for(&file, op_name, "cpu");
    assert!(
        !cases.is_empty(),
        "no CPU fixtures for op {op_name:?} — regenerate elementwise.json"
    );
    for f in cases {
        let label = format!("{op_name} cpu tag={:?} dtype={}", f.tag, f.dtype);
        let a_shape = f.a_shape.as_ref().expect("a_shape");
        let b_shape = f.b_shape.as_ref().expect("b_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let b_data = f
            .b_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("b_data");
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
        let grad_b_exp = f
            .grad_b
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("grad_b");

        match f.dtype.as_str() {
            "float32" => {
                // Forward (no grad)
                let a = make_cpu_f32(a_data, a_shape, false);
                let b = make_cpu_f32(b_data, b_shape, false);
                let c = op.apply_f32(&a, &b);
                let actual = read_back_f32(&c);
                check_f32(
                    &format!("{label} fwd"),
                    &actual,
                    expected,
                    tolerance::F32_TRANSCENDENTAL,
                );

                // Autograd: grads via sum-to-scalar then backward.
                let a_g = make_cpu_f32(a_data, a_shape, true);
                let b_g = make_cpu_f32(b_data, b_shape, true);
                let out = op.apply_f32(&a_g, &b_g);
                let loss = grad_sum(&out).expect("sum");
                loss.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                let gb = b_g.grad().unwrap().expect("grad_b");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&ga),
                    grad_a_exp,
                    tolerance::F32_TRANSCENDENTAL,
                );
                check_f32(
                    &format!("{label} grad_b"),
                    &read_back_f32(&gb),
                    grad_b_exp,
                    tolerance::F32_TRANSCENDENTAL,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, a_shape, false);
                let b = make_cpu_f64(b_data, b_shape, false);
                let c = op.apply_f64(&a, &b);
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&c),
                    expected,
                    tolerance::F64_TRANSCENDENTAL,
                );

                let a_g = make_cpu_f64(a_data, a_shape, true);
                let b_g = make_cpu_f64(b_data, b_shape, true);
                let out = op.apply_f64(&a_g, &b_g);
                let loss = grad_sum(&out).expect("sum");
                loss.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                let gb = b_g.grad().unwrap().expect("grad_b");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&ga),
                    grad_a_exp,
                    tolerance::F64_TRANSCENDENTAL,
                );
                check_f64(
                    &format!("{label} grad_b"),
                    &read_back_f64(&gb),
                    grad_b_exp,
                    tolerance::F64_TRANSCENDENTAL,
                );
            }
            other => panic!("unhandled dtype {other:?}"),
        }
    }
}

#[derive(Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOp {
    fn apply_f32(self, a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        match self {
            BinaryOp::Add => add(a, b).expect("add"),
            BinaryOp::Sub => sub(a, b).expect("sub"),
            BinaryOp::Mul => mul(a, b).expect("mul"),
            BinaryOp::Div => div(a, b).expect("div"),
        }
    }
    fn apply_f64(self, a: &Tensor<f64>, b: &Tensor<f64>) -> Tensor<f64> {
        match self {
            BinaryOp::Add => add(a, b).expect("add"),
            BinaryOp::Sub => sub(a, b).expect("sub"),
            BinaryOp::Mul => mul(a, b).expect("mul"),
            BinaryOp::Div => div(a, b).expect("div"),
        }
    }
}

#[test]
fn cpu_add() {
    run_binary_cpu("add", BinaryOp::Add);
}

#[test]
fn cpu_sub() {
    run_binary_cpu("sub", BinaryOp::Sub);
}

#[test]
fn cpu_mul() {
    run_binary_cpu("mul", BinaryOp::Mul);
}

#[test]
fn cpu_div() {
    run_binary_cpu("div", BinaryOp::Div);
}

// Unary differentiable: neg, abs, sqrt
#[derive(Clone, Copy)]
enum UnaryOp {
    Neg,
    Abs,
    Sqrt,
}

impl UnaryOp {
    fn apply_f32(self, a: &Tensor<f32>) -> Tensor<f32> {
        match self {
            UnaryOp::Neg => neg(a).expect("neg"),
            UnaryOp::Abs => abs(a).expect("abs"),
            UnaryOp::Sqrt => sqrt(a).expect("sqrt"),
        }
    }
    fn apply_f64(self, a: &Tensor<f64>) -> Tensor<f64> {
        match self {
            UnaryOp::Neg => neg(a).expect("neg"),
            UnaryOp::Abs => abs(a).expect("abs"),
            UnaryOp::Sqrt => sqrt(a).expect("sqrt"),
        }
    }
}

fn run_unary_cpu(op_name: &str, op: UnaryOp) {
    let file = load_fixtures();
    let cases = cases_for(&file, op_name, "cpu");
    assert!(
        !cases.is_empty(),
        "no CPU fixtures for op {op_name:?} — regenerate elementwise.json"
    );
    for f in cases {
        let label = format!("{op_name} cpu tag={:?} dtype={}", f.tag, f.dtype);
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

        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape, false);
                let c = op.apply_f32(&a);
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&c),
                    expected,
                    tolerance::F32_TRANSCENDENTAL,
                );

                let a_g = make_cpu_f32(a_data, shape, true);
                let out = op.apply_f32(&a_g);
                let loss = grad_sum(&out).expect("sum");
                loss.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&ga),
                    grad_a_exp,
                    tolerance::F32_TRANSCENDENTAL,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let c = op.apply_f64(&a);
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&c),
                    expected,
                    tolerance::F64_TRANSCENDENTAL,
                );

                let a_g = make_cpu_f64(a_data, shape, true);
                let out = op.apply_f64(&a_g);
                let loss = grad_sum(&out).expect("sum");
                loss.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&ga),
                    grad_a_exp,
                    tolerance::F64_TRANSCENDENTAL,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_neg() {
    run_unary_cpu("neg", UnaryOp::Neg);
}

#[test]
fn cpu_abs() {
    run_unary_cpu("abs", UnaryOp::Abs);
}

#[test]
fn cpu_sqrt() {
    run_unary_cpu("sqrt", UnaryOp::Sqrt);
}

#[test]
fn cpu_pow() {
    let file = load_fixtures();
    let cases = cases_for(&file, "pow", "cpu");
    assert!(!cases.is_empty(), "no CPU fixtures for pow");
    for f in cases {
        let label = format!("pow cpu tag={:?} dtype={} exp={:?}", f.tag, f.dtype, f.exp);
        let shape = f.a_shape.as_ref().expect("a_shape");
        let a_data = f
            .a_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("a_data");
        let exp = f.exp.expect("exp");
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

        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape, false);
                let c = pow(&a, exp).expect("pow");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&c),
                    expected,
                    tolerance::F32_TRANSCENDENTAL,
                );
                let a_g = make_cpu_f32(a_data, shape, true);
                let out = pow(&a_g, exp).expect("pow");
                grad_sum(&out).expect("sum").backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&ga),
                    grad_a_exp,
                    tolerance::F32_TRANSCENDENTAL,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let c = pow(&a, exp).expect("pow");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&c),
                    expected,
                    tolerance::F64_TRANSCENDENTAL,
                );
                let a_g = make_cpu_f64(a_data, shape, true);
                let out = pow(&a_g, exp).expect("pow");
                grad_sum(&out).expect("sum").backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&ga),
                    grad_a_exp,
                    tolerance::F64_TRANSCENDENTAL,
                );
            }
            _ => unreachable!(),
        }
    }
}

// Edge cases: pow(x, 0) = 1, sqrt(0) = 0, div(x, 0) = ±inf / NaN.
#[test]
fn cpu_edge_cases() {
    let file = load_fixtures();

    for f in cases_for(&file, "pow_zero_exp", "cpu") {
        let label = format!("pow_zero_exp cpu dtype={}", f.dtype);
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
                let c = pow(&a, 0.0).expect("pow");
                check_f32(&label, &read_back_f32(&c), exp, tolerance::F32_ELEMENTWISE);
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let c = pow(&a, 0.0).expect("pow");
                check_f64(&label, &read_back_f64(&c), exp, tolerance::F64_ELEMENTWISE);
            }
            _ => unreachable!(),
        }
    }

    for f in cases_for(&file, "sqrt_zero", "cpu") {
        let label = format!("sqrt_zero cpu dtype={}", f.dtype);
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
                let c = sqrt(&a).expect("sqrt");
                check_f32(&label, &read_back_f32(&c), exp, tolerance::F32_ELEMENTWISE);
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let c = sqrt(&a).expect("sqrt");
                check_f64(&label, &read_back_f64(&c), exp, tolerance::F64_ELEMENTWISE);
            }
            _ => unreachable!(),
        }
    }

    for f in cases_for(&file, "div_zero", "cpu") {
        let label = format!("div_zero cpu dtype={}", f.dtype);
        let a_shape = f.a_shape.as_ref().unwrap();
        let b_shape = f.b_shape.as_ref().unwrap();
        let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let b_data = f.b_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let exp = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, a_shape, false);
                let b = make_cpu_f32(b_data, b_shape, false);
                let c = div(&a, &b).expect("div");
                // Compare bitwise against fixture for ±inf/NaN. Tolerance
                // helper handles non-finite specially.
                check_f32(&label, &read_back_f32(&c), exp, tolerance::F32_ELEMENTWISE);
            }
            "float64" => {
                let a = make_cpu_f64(a_data, a_shape, false);
                let b = make_cpu_f64(b_data, b_shape, false);
                let c = div(&a, &b).expect("div");
                check_f64(&label, &read_back_f64(&c), exp, tolerance::F64_ELEMENTWISE);
            }
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// Cat B — where_ / where_bt
// ---------------------------------------------------------------------------

fn run_where_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "where", device_label);
    assert!(!cases.is_empty(), "no fixtures for where on {device_label}");
    for f in cases {
        let label = format!("where {device_label} dtype={}", f.dtype);
        let cond = f.cond.as_ref().expect("cond").clone();
        let x_shape = f.x_shape.as_ref().expect("x_shape");
        let y_shape = f.y_shape.as_ref().expect("y_shape");
        let x_data = f
            .x_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("x_data");
        let y_data = f
            .y_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("y_data");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("out_values");
        let grad_x_exp = f
            .grad_x
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("grad_x");
        let grad_y_exp = f
            .grad_y
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .expect("grad_y");

        match f.dtype.as_str() {
            "float32" => {
                let x = upload_f32(make_cpu_f32(x_data, x_shape, false), device);
                let y = upload_f32(make_cpu_f32(y_data, y_shape, false), device);
                let out = where_(&cond, &x, &y).expect("where_");
                check_f32(
                    &format!("{label} where_ fwd"),
                    &read_back_f32(&out),
                    expected,
                    tolerance::F32_ELEMENTWISE,
                );

                // where_bt mirrors torch.where(Tensor condition, x, y): tensor
                // operands must live on the same device. The host-slice
                // where_ API above is the convenience path that uploads a
                // raw &[bool] mask for CUDA operands.
                let cond_bt = BoolTensor::from_vec(cond.clone(), x_shape.clone())
                    .expect("bt")
                    .to(device)
                    .expect("condition upload");
                let out_bt = where_bt(&cond_bt, &x, &y).expect("where_bt");
                check_f32(
                    &format!("{label} where_bt fwd"),
                    &read_back_f32(&out_bt),
                    expected,
                    tolerance::F32_ELEMENTWISE,
                );

                // Autograd: gradients should route to selected branch only.
                let x_g = upload_f32(make_cpu_f32(x_data, x_shape, true), device);
                let y_g = upload_f32(make_cpu_f32(y_data, y_shape, true), device);
                let out_g = where_(&cond, &x_g, &y_g).expect("where_ grad");
                grad_sum(&out_g).expect("sum").backward().expect("backward");
                let gx = x_g.grad().unwrap().expect("grad_x");
                let gy = y_g.grad().unwrap().expect("grad_y");
                check_f32(
                    &format!("{label} grad_x"),
                    &read_back_f32(&gx),
                    grad_x_exp,
                    tolerance::F32_ELEMENTWISE,
                );
                check_f32(
                    &format!("{label} grad_y"),
                    &read_back_f32(&gy),
                    grad_y_exp,
                    tolerance::F32_ELEMENTWISE,
                );
            }
            "float64" => {
                let x = upload_f64(make_cpu_f64(x_data, x_shape, false), device);
                let y = upload_f64(make_cpu_f64(y_data, y_shape, false), device);
                let out = where_(&cond, &x, &y).expect("where_");
                check_f64(
                    &format!("{label} where_ fwd"),
                    &read_back_f64(&out),
                    expected,
                    tolerance::F64_ELEMENTWISE,
                );

                let cond_bt = BoolTensor::from_vec(cond.clone(), x_shape.clone())
                    .expect("bt")
                    .to(device)
                    .expect("condition upload");
                let out_bt = where_bt(&cond_bt, &x, &y).expect("where_bt");
                check_f64(
                    &format!("{label} where_bt fwd"),
                    &read_back_f64(&out_bt),
                    expected,
                    tolerance::F64_ELEMENTWISE,
                );

                // f64 GPU autograd lane: now live on both CPU and CUDA.
                // ferrotorch's GPU backend gained `fill_f64` (used by
                // SumBackward), so the same autograd code runs against
                // both devices.
                let x_g = upload_f64(make_cpu_f64(x_data, x_shape, true), device);
                let y_g = upload_f64(make_cpu_f64(y_data, y_shape, true), device);
                let out_g = where_(&cond, &x_g, &y_g).expect("where_ grad");
                grad_sum(&out_g).expect("sum").backward().expect("backward");
                let gx = x_g.grad().unwrap().expect("grad_x");
                let gy = y_g.grad().unwrap().expect("grad_y");
                check_f64(
                    &format!("{label} grad_x"),
                    &read_back_f64(&gx),
                    grad_x_exp,
                    tolerance::F64_ELEMENTWISE,
                );
                check_f64(
                    &format!("{label} grad_y"),
                    &read_back_f64(&gy),
                    grad_y_exp,
                    tolerance::F64_ELEMENTWISE,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_where() {
    run_where_for_device("cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// Cat C — higher-order utilities (CPU only by design)
// ---------------------------------------------------------------------------
//
// Test the higher-order *contract* with non-trivial closures. The fixture's
// reference values come from PyTorch evaluating the equivalent pointwise
// expression — proving the closure is applied elementwise correctly.

#[test]
fn cpu_binary_map_higher_order() {
    let file = load_fixtures();
    let cases = cases_for(&file, "binary_map_maxmin", "cpu");
    assert!(!cases.is_empty(), "no fixtures for binary_map_maxmin");
    for f in cases {
        let label = format!("binary_map_maxmin cpu dtype={}", f.dtype);
        let a_shape = f.a_shape.as_ref().unwrap();
        let b_shape = f.b_shape.as_ref().unwrap();
        let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let b_data = f.b_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let exp = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();

        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, a_shape, false);
                let b = make_cpu_f32(b_data, b_shape, false);
                // Non-trivial closure: max(x, y) - min(x, y).
                let out = binary_map(&a, &b, |x, y| x.max(y) - x.min(y)).expect("binary_map");
                check_f32(
                    &label,
                    &read_back_f32(&out),
                    exp,
                    tolerance::F32_ELEMENTWISE,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, a_shape, false);
                let b = make_cpu_f64(b_data, b_shape, false);
                let out = binary_map(&a, &b, |x, y| x.max(y) - x.min(y)).expect("binary_map");
                check_f64(
                    &label,
                    &read_back_f64(&out),
                    exp,
                    tolerance::F64_ELEMENTWISE,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_scalar_map_higher_order() {
    let file = load_fixtures();
    let cases = cases_for(&file, "scalar_map_sqplus", "cpu");
    assert!(!cases.is_empty(), "no fixtures for scalar_map_sqplus");
    for f in cases {
        let label = format!("scalar_map_sqplus cpu dtype={}", f.dtype);
        let a_shape = f.a_shape.as_ref().unwrap();
        let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let scalar = f.scalar.expect("scalar");
        let exp = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();

        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, a_shape, false);
                let s = scalar as f32;
                // Closure: x*x + s.
                let out = scalar_map(&a, s, |x, s| x * x + s).expect("scalar_map");
                check_f32(
                    &label,
                    &read_back_f32(&out),
                    exp,
                    tolerance::F32_ELEMENTWISE,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, a_shape, false);
                let out = scalar_map(&a, scalar, |x, s| x * x + s).expect("scalar_map");
                check_f64(
                    &label,
                    &read_back_f64(&out),
                    exp,
                    tolerance::F64_ELEMENTWISE,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_unary_map_higher_order() {
    let file = load_fixtures();
    let cases = cases_for(&file, "unary_map_tan", "cpu");
    assert!(!cases.is_empty(), "no fixtures for unary_map_tan");
    for f in cases {
        let label = format!("unary_map_tan cpu dtype={}", f.dtype);
        let a_shape = f.a_shape.as_ref().unwrap();
        let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let exp = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, a_shape, false);
                let out = unary_map(&a, |x| x.tan()).expect("unary_map");
                check_f32(
                    &label,
                    &read_back_f32(&out),
                    exp,
                    tolerance::F32_TRANSCENDENTAL,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, a_shape, false);
                let out = unary_map(&a, |x| x.tan()).expect("unary_map");
                check_f64(
                    &label,
                    &read_back_f64(&out),
                    exp,
                    tolerance::F64_TRANSCENDENTAL,
                );
            }
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// Cat D — perf variants (fast_*, simd_*) — CPU only by design
// ---------------------------------------------------------------------------
//
// Two-axis conformance: each fast_/simd_ op is compared against
//   (i) torch.<canonical> (the fixture reference), and
//   (ii) ferrotorch's own canonical version (e.g. fast_add vs add) for
//       internal consistency.

fn check_canon_consistency_f32(label: &str, fast_actual: &[f32], canon_actual: &[f32], tol: f32) {
    // Internal consistency: fast_* must agree with the canonical op within
    // the same tolerance band.
    tolerance::assert_close_f32(fast_actual, canon_actual, tol, label);
}

fn check_canon_consistency_f64(label: &str, fast_actual: &[f64], canon_actual: &[f64], tol: f64) {
    tolerance::assert_close_f64(fast_actual, canon_actual, tol, label);
}

#[test]
fn cpu_fast_binary_ops() {
    let file = load_fixtures();
    for op_name in ["fast_add", "fast_sub", "fast_mul", "fast_div"] {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
            let label = format!("{op_name} cpu dtype={}", f.dtype);
            let a_shape = f.a_shape.as_ref().unwrap();
            let b_shape = f.b_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let b_data = f.b_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let exp = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            match f.dtype.as_str() {
                "float32" => {
                    let a = make_cpu_f32(a_data, a_shape, false);
                    let b = make_cpu_f32(b_data, b_shape, false);
                    let fast_t = match op_name {
                        "fast_add" => fast_add(&a, &b).unwrap(),
                        "fast_sub" => fast_sub(&a, &b).unwrap(),
                        "fast_mul" => fast_mul(&a, &b).unwrap(),
                        "fast_div" => fast_div(&a, &b).unwrap(),
                        _ => unreachable!(),
                    };
                    let canon_t = match op_name {
                        "fast_add" => add(&a, &b).unwrap(),
                        "fast_sub" => sub(&a, &b).unwrap(),
                        "fast_mul" => mul(&a, &b).unwrap(),
                        "fast_div" => div(&a, &b).unwrap(),
                        _ => unreachable!(),
                    };
                    let fa = read_back_f32(&fast_t);
                    let ca = read_back_f32(&canon_t);
                    check_f32(
                        &format!("{label} parity-vs-torch"),
                        &fa,
                        exp,
                        tolerance::F32_FAST,
                    );
                    check_canon_consistency_f32(
                        &format!("{label} parity-vs-canonical"),
                        &fa,
                        &ca,
                        tolerance::F32_FAST,
                    );
                }
                "float64" => {
                    let a = make_cpu_f64(a_data, a_shape, false);
                    let b = make_cpu_f64(b_data, b_shape, false);
                    let fast_t = match op_name {
                        "fast_add" => fast_add(&a, &b).unwrap(),
                        "fast_sub" => fast_sub(&a, &b).unwrap(),
                        "fast_mul" => fast_mul(&a, &b).unwrap(),
                        "fast_div" => fast_div(&a, &b).unwrap(),
                        _ => unreachable!(),
                    };
                    let canon_t = match op_name {
                        "fast_add" => add(&a, &b).unwrap(),
                        "fast_sub" => sub(&a, &b).unwrap(),
                        "fast_mul" => mul(&a, &b).unwrap(),
                        "fast_div" => div(&a, &b).unwrap(),
                        _ => unreachable!(),
                    };
                    let fa = read_back_f64(&fast_t);
                    let ca = read_back_f64(&canon_t);
                    check_f64(
                        &format!("{label} parity-vs-torch"),
                        &fa,
                        exp,
                        tolerance::F64_FAST,
                    );
                    check_canon_consistency_f64(
                        &format!("{label} parity-vs-canonical"),
                        &fa,
                        &ca,
                        tolerance::F64_FAST,
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn cpu_fast_unary_ops() {
    let file = load_fixtures();
    // fast_exp/log/sigmoid/tanh/sin/cos: parity vs torch only (no exact
    // canonical "add"-style ferrotorch op; the canonical reference for
    // exp/log/sin/cos is `grad_fns::transcendental::*` which is Phase 2.5
    // territory). The two-axis check here is parity-vs-torch + parity-vs-
    // libm by virtue of T::exp() etc. — the closure form is applied via
    // ferrotorch's `unary_map(&input, |x| x.exp())` as the secondary axis.
    for op_name in [
        "fast_exp",
        "fast_log",
        "fast_sigmoid",
        "fast_tanh",
        "fast_sin",
        "fast_cos",
    ] {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
            let label = format!("{op_name} cpu dtype={}", f.dtype);
            let a_shape = f.a_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let exp = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            match f.dtype.as_str() {
                "float32" => {
                    let a = make_cpu_f32(a_data, a_shape, false);
                    let (fast_t, canon_t): (Tensor<f32>, Tensor<f32>) = match op_name {
                        "fast_exp" => (
                            fast_exp(&a).unwrap(),
                            unary_map(&a, |x: f32| x.exp()).unwrap(),
                        ),
                        "fast_log" => (
                            fast_log(&a).unwrap(),
                            unary_map(&a, |x: f32| x.ln()).unwrap(),
                        ),
                        "fast_sigmoid" => (
                            fast_sigmoid(&a).unwrap(),
                            unary_map(&a, |x: f32| 1.0 / (1.0 + (-x).exp())).unwrap(),
                        ),
                        "fast_tanh" => (
                            fast_tanh(&a).unwrap(),
                            unary_map(&a, |x: f32| x.tanh()).unwrap(),
                        ),
                        "fast_sin" => (
                            fast_sin(&a).unwrap(),
                            unary_map(&a, |x: f32| x.sin()).unwrap(),
                        ),
                        "fast_cos" => (
                            fast_cos(&a).unwrap(),
                            unary_map(&a, |x: f32| x.cos()).unwrap(),
                        ),
                        _ => unreachable!(),
                    };
                    let fa = read_back_f32(&fast_t);
                    let ca = read_back_f32(&canon_t);
                    check_f32(
                        &format!("{label} parity-vs-torch"),
                        &fa,
                        exp,
                        tolerance::F32_FAST,
                    );
                    check_canon_consistency_f32(
                        &format!("{label} parity-vs-canonical"),
                        &fa,
                        &ca,
                        tolerance::F32_FAST,
                    );
                }
                "float64" => {
                    let a = make_cpu_f64(a_data, a_shape, false);
                    let (fast_t, canon_t): (Tensor<f64>, Tensor<f64>) = match op_name {
                        "fast_exp" => (
                            fast_exp(&a).unwrap(),
                            unary_map(&a, |x: f64| x.exp()).unwrap(),
                        ),
                        "fast_log" => (
                            fast_log(&a).unwrap(),
                            unary_map(&a, |x: f64| x.ln()).unwrap(),
                        ),
                        "fast_sigmoid" => (
                            fast_sigmoid(&a).unwrap(),
                            unary_map(&a, |x: f64| 1.0 / (1.0 + (-x).exp())).unwrap(),
                        ),
                        "fast_tanh" => (
                            fast_tanh(&a).unwrap(),
                            unary_map(&a, |x: f64| x.tanh()).unwrap(),
                        ),
                        "fast_sin" => (
                            fast_sin(&a).unwrap(),
                            unary_map(&a, |x: f64| x.sin()).unwrap(),
                        ),
                        "fast_cos" => (
                            fast_cos(&a).unwrap(),
                            unary_map(&a, |x: f64| x.cos()).unwrap(),
                        ),
                        _ => unreachable!(),
                    };
                    let fa = read_back_f64(&fast_t);
                    let ca = read_back_f64(&canon_t);
                    check_f64(
                        &format!("{label} parity-vs-torch"),
                        &fa,
                        exp,
                        tolerance::F64_FAST,
                    );
                    check_canon_consistency_f64(
                        &format!("{label} parity-vs-canonical"),
                        &fa,
                        &ca,
                        tolerance::F64_FAST,
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn cpu_simd_ops() {
    let file = load_fixtures();
    // f32 SIMD ops:
    for op_name in [
        "simd_add_f32",
        "simd_mul_f32",
        "simd_exp_f32",
        "simd_log_f32",
        "simd_sqrt_f32",
    ] {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
            let label = format!("{op_name} cpu");
            let a_shape = f.a_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let exp = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            let a = make_cpu_f32(a_data, a_shape, false);
            let (simd_t, canon_t): (Tensor<f32>, Tensor<f32>) = match op_name {
                "simd_add_f32" => {
                    let b_shape = f.b_shape.as_ref().unwrap();
                    let b_data = f.b_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
                    let b = make_cpu_f32(b_data, b_shape, false);
                    (simd_add_f32(&a, &b).unwrap(), add(&a, &b).unwrap())
                }
                "simd_mul_f32" => {
                    let b_shape = f.b_shape.as_ref().unwrap();
                    let b_data = f.b_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
                    let b = make_cpu_f32(b_data, b_shape, false);
                    (simd_mul_f32(&a, &b).unwrap(), mul(&a, &b).unwrap())
                }
                "simd_exp_f32" => (
                    simd_exp_f32(&a).unwrap(),
                    unary_map(&a, |x: f32| x.exp()).unwrap(),
                ),
                "simd_log_f32" => (
                    simd_log_f32(&a).unwrap(),
                    unary_map(&a, |x: f32| x.ln()).unwrap(),
                ),
                "simd_sqrt_f32" => (simd_sqrt_f32(&a).unwrap(), sqrt(&a).unwrap()),
                _ => unreachable!(),
            };
            let sa = read_back_f32(&simd_t);
            let ca = read_back_f32(&canon_t);
            check_f32(
                &format!("{label} parity-vs-torch"),
                &sa,
                exp,
                tolerance::F32_SIMD,
            );
            check_canon_consistency_f32(
                &format!("{label} parity-vs-canonical"),
                &sa,
                &ca,
                tolerance::F32_SIMD,
            );
        }
    }

    // f64 SIMD ops:
    for op_name in ["simd_add_f64", "simd_mul_f64", "simd_exp_f64"] {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
            let label = format!("{op_name} cpu");
            let a_shape = f.a_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let exp = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            let a = make_cpu_f64(a_data, a_shape, false);
            let (simd_t, canon_t): (Tensor<f64>, Tensor<f64>) = match op_name {
                "simd_add_f64" => {
                    let b_shape = f.b_shape.as_ref().unwrap();
                    let b_data = f.b_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
                    let b = make_cpu_f64(b_data, b_shape, false);
                    (simd_add_f64(&a, &b).unwrap(), add(&a, &b).unwrap())
                }
                "simd_mul_f64" => {
                    let b_shape = f.b_shape.as_ref().unwrap();
                    let b_data = f.b_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
                    let b = make_cpu_f64(b_data, b_shape, false);
                    (simd_mul_f64(&a, &b).unwrap(), mul(&a, &b).unwrap())
                }
                "simd_exp_f64" => (
                    simd_exp_f64(&a).unwrap(),
                    unary_map(&a, |x: f64| x.exp()).unwrap(),
                ),
                _ => unreachable!(),
            };
            let sa = read_back_f64(&simd_t);
            let ca = read_back_f64(&canon_t);
            check_f64(
                &format!("{label} parity-vs-torch"),
                &sa,
                exp,
                tolerance::F64_SIMD,
            );
            check_canon_consistency_f64(
                &format!("{label} parity-vs-canonical"),
                &sa,
                &ca,
                tolerance::F64_SIMD,
            );
        }
    }
}

/// CORE-131 / #1825 regression: direct SIMD binary ops are same-shape kernel
/// surfaces. Mismatched shapes must return a structured error before ferray's
/// debug-asserting zip kernels run; no debug panic and no release partial
/// output.
#[test]
fn cpu_simd_shape_mismatch_pin_1825() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    // Distinct operand values so add/mul prefixes are distinguishable from
    // each other and from the zero-initialized tail.
    let a32 = make_cpu_f32(&[1.5; 6], &[2, 3], false);
    let b32 = make_cpu_f32(&[0.25; 2], &[2], false);
    let a64 = make_cpu_f64(&[1.5; 6], &[2, 3], false);
    let b64 = make_cpu_f64(&[0.25; 2], &[2], false);

    macro_rules! assert_shape_err {
        ($label:literal, $call:expr) => {{
            let outcome = catch_unwind(AssertUnwindSafe(|| $call));
            match outcome {
                Ok(Err(FerrotorchError::ShapeMismatch { .. })) => {}
                Ok(Err(other)) => {
                    panic!("{}: expected ShapeMismatch, got {other:?}", $label)
                }
                Ok(Ok(t)) => {
                    panic!("{}: expected ShapeMismatch, got Ok({t:?})", $label)
                }
                Err(_) => panic!("{}: panicked instead of returning ShapeMismatch", $label),
            }
        }};
    }

    assert_shape_err!("simd_add_f32([2,3],[2])", simd_add_f32(&a32, &b32));
    assert_shape_err!("simd_mul_f32([2,3],[2])", simd_mul_f32(&a32, &b32));
    assert_shape_err!("simd_add_f64([2,3],[2])", simd_add_f64(&a64, &b64));
    assert_shape_err!("simd_mul_f64([2,3],[2])", simd_mul_f64(&a64, &b64));
}

// ---------------------------------------------------------------------------
// Cat E — reductions
// ---------------------------------------------------------------------------

#[test]
fn cpu_sum() {
    let file = load_fixtures();
    let cases = cases_for(&file, "sum", "cpu");
    assert!(!cases.is_empty(), "no fixtures for sum");
    for f in cases {
        let label = format!("sum cpu dtype={}", f.dtype);
        let shape = f.a_shape.as_ref().unwrap();
        let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let exp = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_a_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape, false);
                let s = sum(&a).expect("sum");
                // k-aware tolerance: sweep rows reduce up to 10007 summands.
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&s),
                    exp,
                    tolerance::accum_tol_f32(a_data.len()),
                );
                // sum's grad-of-input is all-ones (loss = sum(a) -> ds/da = 1).
                // ferrotorch::ops::elementwise::sum is non-differentiable; the
                // differentiable path is `grad_fns::reduction::sum`. We use
                // the differentiable variant for the autograd assertion.
                let a_g = make_cpu_f32(a_data, shape, true);
                let s_g = ferrotorch_core::grad_fns::reduction::sum(&a_g).expect("grad sum");
                s_g.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&ga),
                    grad_a_exp,
                    tolerance::F32_ELEMENTWISE,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let s = sum(&a).expect("sum");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&s),
                    exp,
                    tolerance::accum_tol_f64(a_data.len()),
                );
                let a_g = make_cpu_f64(a_data, shape, true);
                let s_g = ferrotorch_core::grad_fns::reduction::sum(&a_g).expect("grad sum");
                s_g.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&ga),
                    grad_a_exp,
                    tolerance::F64_ELEMENTWISE,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_sum_axis() {
    let file = load_fixtures();
    let cases = cases_for(&file, "sum_axis", "cpu");
    assert!(!cases.is_empty(), "no fixtures for sum_axis");
    for f in cases {
        let label = format!("sum_axis cpu axis={:?} dtype={}", f.axis, f.dtype);
        let shape = f.a_shape.as_ref().unwrap();
        let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let axis = f.axis.expect("axis");
        let exp = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();

        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape, false);
                let s = sum_axis(&a, axis).expect("sum_axis");
                check_f32(&label, &read_back_f32(&s), exp, tolerance::F32_REDUCTION);
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let s = sum_axis(&a, axis).expect("sum_axis");
                check_f64(&label, &read_back_f64(&s), exp, tolerance::F64_REDUCTION);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_mean() {
    let file = load_fixtures();
    let cases = cases_for(&file, "mean", "cpu");
    assert!(!cases.is_empty(), "no fixtures for mean");
    for f in cases {
        let label = format!("mean cpu dtype={}", f.dtype);
        let shape = f.a_shape.as_ref().unwrap();
        let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let exp = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_a_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape, false);
                let m = mean(&a).expect("mean");
                // k-aware tolerance: mean = sum/k inherits the sum's
                // accumulation error (see tolerance::accum_tol_f32).
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&m),
                    exp,
                    tolerance::accum_tol_f32(a_data.len()),
                );
                // CORE-200 (#1894): drive mean's OWN backward
                // (`grad_fns::reduction::mean` -> `MeanBackward`), never a
                // sum-backward rescaled by 1/n inside the test — that
                // synthesizes the expected VJP from a different node and
                // passes green even when MeanBackward is wrong or missing.
                // `mean` returns a scalar, so the loss is the mean itself.
                // Expected grads come straight from the torch fixture.
                let a_g = make_cpu_f32(a_data, shape, true);
                let m_g = grad_mean(&a_g).expect("grad mean");
                check_f32(
                    &format!("{label} grad fwd"),
                    &read_back_f32(&m_g),
                    exp,
                    tolerance::accum_tol_f32(a_data.len()),
                );
                m_g.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&ga),
                    grad_a_exp,
                    tolerance::F32_ELEMENTWISE,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let m = mean(&a).expect("mean");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&m),
                    exp,
                    tolerance::accum_tol_f64(a_data.len()),
                );
                // CORE-200 (#1894): same as the f32 lane — mean's own
                // backward, expectations from the torch fixture.
                let a_g = make_cpu_f64(a_data, shape, true);
                let m_g = grad_mean(&a_g).expect("grad mean");
                check_f64(
                    &format!("{label} grad fwd"),
                    &read_back_f64(&m_g),
                    exp,
                    tolerance::accum_tol_f64(a_data.len()),
                );
                m_g.backward().expect("backward");
                let ga = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&ga),
                    grad_a_exp,
                    tolerance::F64_ELEMENTWISE,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_nansum() {
    let file = load_fixtures();
    let cases = cases_for(&file, "nansum", "cpu");
    assert!(!cases.is_empty(), "no fixtures for nansum");
    for f in cases {
        let label = format!("nansum cpu dtype={}", f.dtype);
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
                let s = nansum(&a).expect("nansum");
                check_f32(&label, &read_back_f32(&s), exp, tolerance::F32_REDUCTION);
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let s = nansum(&a).expect("nansum");
                check_f64(&label, &read_back_f64(&s), exp, tolerance::F64_REDUCTION);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_nanmean() {
    let file = load_fixtures();
    let cases = cases_for(&file, "nanmean", "cpu");
    assert!(!cases.is_empty(), "no fixtures for nanmean");
    for f in cases {
        let label = format!("nanmean cpu dtype={}", f.dtype);
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
                let m = nanmean(&a).expect("nanmean");
                check_f32(&label, &read_back_f32(&m), exp, tolerance::F32_REDUCTION);
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let m = nanmean(&a).expect("nanmean");
                check_f64(&label, &read_back_f64(&m), exp, tolerance::F64_REDUCTION);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_logsumexp() {
    let file = load_fixtures();
    let cases = cases_for(&file, "logsumexp", "cpu");
    assert!(!cases.is_empty(), "no fixtures for logsumexp");
    for f in cases {
        let label = format!("logsumexp cpu tag={:?} dtype={}", f.tag, f.dtype);
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
                let l = logsumexp(&a).expect("logsumexp");
                // The numerical-stability case (input [100, 100]) must
                // produce 100 + ln(2) ~ 100.693, NOT inf. The tolerance
                // here is loose enough to ride the f32 accumulation but
                // tight enough that an overflow would still fail. Sweep
                // rows (k up to 10007) take the k-aware band: logsumexp's
                // error is the k-term exp-sum's accumulation error divided
                // by the sum — the same relative model as accum_tol.
                check_f32(
                    &label,
                    &read_back_f32(&l),
                    exp,
                    tolerance::F32_TRANSCENDENTAL.max(tolerance::accum_tol_f32(a_data.len())),
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let l = logsumexp(&a).expect("logsumexp");
                // k-aware band for the sweep rows — see the f32 lane.
                check_f64(
                    &label,
                    &read_back_f64(&l),
                    exp,
                    tolerance::F64_TRANSCENDENTAL.max(tolerance::accum_tol_f64(a_data.len())),
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_logsumexp_dim() {
    let file = load_fixtures();
    let cases = cases_for(&file, "logsumexp_dim", "cpu");
    assert!(!cases.is_empty(), "no fixtures for logsumexp_dim");
    for f in cases {
        let label = format!(
            "logsumexp_dim cpu tag={:?} dtype={} axis={:?} keepdim={:?}",
            f.tag, f.dtype, f.axis, f.keepdim
        );
        let shape = f.a_shape.as_ref().unwrap();
        let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let axis = f.axis.expect("axis");
        let keepdim = f.keepdim.expect("keepdim");
        let exp = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape, false);
                let l = logsumexp_dim(&a, axis, keepdim).expect("logsumexp_dim");
                check_f32(
                    &label,
                    &read_back_f32(&l),
                    exp,
                    tolerance::F32_TRANSCENDENTAL,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let l = logsumexp_dim(&a, axis, keepdim).expect("logsumexp_dim");
                check_f64(
                    &label,
                    &read_back_f64(&l),
                    exp,
                    tolerance::F64_TRANSCENDENTAL,
                );
            }
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// Cat F — in-place mutation (forward conformance + storage-identity contract)
// ---------------------------------------------------------------------------
//
// Storage identity: after an in-place op, `Arc::as_ptr(t.inner_storage_arc())`
// is unchanged. Both the CPU update_data path (writes through the Arc) and
// the GPU update_storage path (replaces the inner buffer through the Arc via
// ptr::replace) preserve the *Arc identity* — only the contained buffer
// changes. That's the meaningful invariant for "in-place".
//
// Autograd note: ferrotorch's in-place ops reject leaves with
// requires_grad=true (matching PyTorch's "leaf variable that requires grad
// is being used in an in-place operation" RuntimeError). We test that
// rejection contract once below, then perform the in-place ops on detached
// tensors for the value comparison.

fn storage_arc_id<T: ferrotorch_core::Float>(t: &Tensor<T>) -> *const TensorStorage<T> {
    Arc::as_ptr(t.inner_storage_arc())
}

#[test]
fn cpu_inplace_add_sub_mul_div() {
    let file = load_fixtures();
    for op_name in ["add_", "sub_", "mul_", "div_"] {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
            let label = format!("{op_name} cpu dtype={}", f.dtype);
            let a_shape = f.a_shape.as_ref().unwrap();
            let b_shape = f.b_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let b_data = f.b_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let exp = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            match f.dtype.as_str() {
                "float32" => {
                    let t = make_cpu_f32(a_data, a_shape, false);
                    let other = make_cpu_f32(b_data, b_shape, false);
                    let before_id = storage_arc_id(&t);
                    match op_name {
                        "add_" => {
                            t.add_(&other).expect("add_");
                        }
                        "sub_" => {
                            t.sub_(&other).expect("sub_");
                        }
                        "mul_" => {
                            t.mul_(&other).expect("mul_");
                        }
                        "div_" => {
                            t.div_(&other).expect("div_");
                        }
                        _ => unreachable!(),
                    }
                    let after_id = storage_arc_id(&t);
                    assert_eq!(
                        before_id, after_id,
                        "{label}: in-place op replaced the storage Arc — \
                         mutation must happen through the existing Arc"
                    );
                    check_f32(
                        &format!("{label} value"),
                        &read_back_f32(&t),
                        exp,
                        tolerance::F32_ELEMENTWISE,
                    );
                }
                "float64" => {
                    let t = make_cpu_f64(a_data, a_shape, false);
                    let other = make_cpu_f64(b_data, b_shape, false);
                    let before_id = storage_arc_id(&t);
                    match op_name {
                        "add_" => {
                            t.add_(&other).expect("add_");
                        }
                        "sub_" => {
                            t.sub_(&other).expect("sub_");
                        }
                        "mul_" => {
                            t.mul_(&other).expect("mul_");
                        }
                        "div_" => {
                            t.div_(&other).expect("div_");
                        }
                        _ => unreachable!(),
                    }
                    let after_id = storage_arc_id(&t);
                    assert_eq!(before_id, after_id, "{label}: storage Arc replaced");
                    check_f64(
                        &format!("{label} value"),
                        &read_back_f64(&t),
                        exp,
                        tolerance::F64_ELEMENTWISE,
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn cpu_inplace_scalar_ops() {
    let file = load_fixtures();
    for (op_name, _is_add) in [("add_scalar_", true), ("mul_scalar_", false)] {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
            let label = format!("{op_name} cpu dtype={}", f.dtype);
            let a_shape = f.a_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let scalar = f.scalar.expect("scalar");
            let exp = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            match f.dtype.as_str() {
                "float32" => {
                    let t = make_cpu_f32(a_data, a_shape, false);
                    let before_id = storage_arc_id(&t);
                    match op_name {
                        "add_scalar_" => {
                            t.add_scalar_(scalar as f32).expect("add_scalar_");
                        }
                        "mul_scalar_" => {
                            t.mul_scalar_(scalar as f32).expect("mul_scalar_");
                        }
                        _ => unreachable!(),
                    }
                    let after_id = storage_arc_id(&t);
                    assert_eq!(before_id, after_id, "{label}: storage Arc replaced");
                    check_f32(
                        &format!("{label} value"),
                        &read_back_f32(&t),
                        exp,
                        tolerance::F32_ELEMENTWISE,
                    );
                }
                "float64" => {
                    let t = make_cpu_f64(a_data, a_shape, false);
                    let before_id = storage_arc_id(&t);
                    match op_name {
                        "add_scalar_" => {
                            t.add_scalar_(scalar).expect("add_scalar_");
                        }
                        "mul_scalar_" => {
                            t.mul_scalar_(scalar).expect("mul_scalar_");
                        }
                        _ => unreachable!(),
                    }
                    let after_id = storage_arc_id(&t);
                    assert_eq!(before_id, after_id, "{label}: storage Arc replaced");
                    check_f64(
                        &format!("{label} value"),
                        &read_back_f64(&t),
                        exp,
                        tolerance::F64_ELEMENTWISE,
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn cpu_inplace_fill_zero_clamp() {
    let file = load_fixtures();
    for op_name in ["fill_", "zero_", "clamp_"] {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
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
                    let t = make_cpu_f32(a_data, shape, false);
                    let before_id = storage_arc_id(&t);
                    match op_name {
                        "fill_" => {
                            t.fill_(f.scalar.expect("scalar") as f32).expect("fill_");
                        }
                        "zero_" => {
                            t.zero_().expect("zero_");
                        }
                        "clamp_" => {
                            let lo = f.min.expect("min") as f32;
                            let hi = f.max.expect("max") as f32;
                            t.clamp_(lo, hi).expect("clamp_");
                        }
                        _ => unreachable!(),
                    }
                    let after_id = storage_arc_id(&t);
                    assert_eq!(before_id, after_id, "{label}: storage Arc replaced");
                    check_f32(
                        &format!("{label} value"),
                        &read_back_f32(&t),
                        exp,
                        tolerance::F32_ELEMENTWISE,
                    );
                }
                "float64" => {
                    let t = make_cpu_f64(a_data, shape, false);
                    let before_id = storage_arc_id(&t);
                    match op_name {
                        "fill_" => {
                            t.fill_(f.scalar.expect("scalar")).expect("fill_");
                        }
                        "zero_" => {
                            t.zero_().expect("zero_");
                        }
                        "clamp_" => {
                            t.clamp_(f.min.expect("min"), f.max.expect("max"))
                                .expect("clamp_");
                        }
                        _ => unreachable!(),
                    }
                    let after_id = storage_arc_id(&t);
                    assert_eq!(before_id, after_id, "{label}: storage Arc replaced");
                    check_f64(
                        &format!("{label} value"),
                        &read_back_f64(&t),
                        exp,
                        tolerance::F64_ELEMENTWISE,
                    );
                }
                _ => unreachable!(),
            }
        }
    }
}

/// In-place ops must reject `requires_grad=true` leaves the way PyTorch
/// raises `RuntimeError("leaf variable that requires grad is being used
/// in an in-place operation")`. ferrotorch returns
/// `FerrotorchError::InvalidArgument` from `check_inplace_allowed`.
#[test]
fn cpu_inplace_rejects_requires_grad_leaf() {
    let t = make_cpu_f32(&[1.0, 2.0, 3.0], &[3], true);
    let err = t.add_scalar_(1.0).unwrap_err();
    assert!(
        format!("{err}").contains("in-place") || format!("{err:?}").contains("InvalidArgument"),
        "expected InvalidArgument for add_scalar_ on a requires_grad leaf, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Cat H — special-value lanes (CORE-199 / #1893)
// ---------------------------------------------------------------------------
//
// NaN / ±inf / -0.0 / subnormal per op family, expectations from live torch
// 2.11.0 (recorded in the fixture). Known divergences are pinned —
// single-contract, retire-on-fix (R-ORACLE-4):
//   * logsumexp / logsumexp_dim +inf poisoning  -> CORE-134 / #1828
//   * fast_exp (vexp_f32) domain clamping (f32) -> CORE-135 / #1829

#[test]
fn cpu_logsumexp_special() {
    let file = load_fixtures();
    let cases = cases_for(&file, "logsumexp_special", "cpu");
    assert!(!cases.is_empty(), "no fixtures for logsumexp_special");
    for f in cases {
        let label = format!("logsumexp_special cpu tag={:?} dtype={}", f.tag, f.dtype);
        let shape = f.a_shape.as_ref().unwrap();
        let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let exp = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        // #1828 (CORE-134) fixed: infinite maxes are masked to zero before
        // the exp-sum per ATen (`logsumexp_out_impl` at pytorch
        // `aten/src/ATen/native/ReduceOps.cpp:1512-1521`), so the sv_pos_inf
        // row now returns torch's +inf. Pin retired — every row asserts the
        // live-torch fixture value.
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape, false);
                let l = logsumexp(&a).expect("logsumexp");
                check_f32(
                    &label,
                    &read_back_f32(&l),
                    exp,
                    tolerance::F32_TRANSCENDENTAL,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let l = logsumexp(&a).expect("logsumexp");
                check_f64(
                    &label,
                    &read_back_f64(&l),
                    exp,
                    tolerance::F64_TRANSCENDENTAL,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_logsumexp_dim_special() {
    let file = load_fixtures();
    let cases = cases_for(&file, "logsumexp_dim_special", "cpu");
    assert!(!cases.is_empty(), "no fixtures for logsumexp_dim_special");
    for f in cases {
        let label = format!(
            "logsumexp_dim_special cpu tag={:?} dtype={}",
            f.tag, f.dtype
        );
        let shape = f.a_shape.as_ref().unwrap();
        let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let axis = f.axis.expect("axis");
        let exp = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        // Both special rows put the infinite slice at output index 0 and a
        // finite slice at index 1.
        //
        // #1828 (CORE-134) fixed: infinite per-slice maxes are masked to
        // zero before the exp-sum per ATen (`logsumexp_out_impl` at pytorch
        // `aten/src/ATen/native/ReduceOps.cpp:1512-1521`), so the all-(-inf)
        // row gives torch's -inf and the +inf row gives +inf. Pin retired —
        // the whole vector asserts the live-torch fixture values.
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape, false);
                let l = logsumexp_dim(&a, axis, false).expect("logsumexp_dim");
                check_f32(
                    &label,
                    &read_back_f32(&l),
                    exp,
                    tolerance::F32_TRANSCENDENTAL,
                );
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let l = logsumexp_dim(&a, axis, false).expect("logsumexp_dim");
                check_f64(
                    &label,
                    &read_back_f64(&l),
                    exp,
                    tolerance::F64_TRANSCENDENTAL,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_fast_exp_special() {
    let file = load_fixtures();
    let cases = cases_for(&file, "fast_exp_special", "cpu");
    assert!(!cases.is_empty(), "no fixtures for fast_exp_special");
    for f in cases {
        let label = format!("fast_exp_special cpu dtype={}", f.dtype);
        let shape = f.a_shape.as_ref().unwrap();
        let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let exp = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        // Input layout (fixture cat H):
        //   [0]=-inf [1]=-100 [2]=-103.9 [3]=88.5 [4]=+inf [5]=NaN
        //   [6]=-0.0 [7]=1e-40
        match f.dtype.as_str() {
            "float32" => {
                let a = make_cpu_f32(a_data, shape, false);
                let r = fast_exp(&a).expect("fast_exp");
                // #1829 (CORE-135) fixed: vexp_f32 delegates NaN/±inf, the
                // subnormal-result band (x < -87.33654), and the
                // near-overflow band (x > 88.0) to libm f32::exp instead of
                // clamping, so exp(-inf)=0, exp(-100)=3.78e-44 (subnormal),
                // exp(-103.9)=1.4e-45, exp(88.5)=2.7231e38 (finite) all
                // match torch. Pins retired — the whole vector asserts the
                // live-torch fixture values.
                check_f32(&label, &read_back_f32(&r), exp, tolerance::F32_FAST);
            }
            "float64" => {
                // The f64 fast_exp path delegates to libm — full parity
                // with torch on every special value.
                let a = make_cpu_f64(a_data, shape, false);
                let r = fast_exp(&a).expect("fast_exp");
                check_f64(&label, &read_back_f64(&r), exp, tolerance::F64_FAST);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_simd_exp_special() {
    let file = load_fixtures();
    let cases = cases_for(&file, "simd_exp_special", "cpu");
    assert!(!cases.is_empty(), "no fixtures for simd_exp_special");
    for f in cases {
        let label = format!("simd_exp_special cpu dtype={}", f.dtype);
        let shape = f.a_shape.as_ref().unwrap();
        let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let exp = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        match f.dtype.as_str() {
            // Unlike vexp_f32 (#1829), the simd exp kernels handle the full
            // special-value domain — value-asserted against torch.
            "float32" => {
                let a = make_cpu_f32(a_data, shape, false);
                let r = simd_exp_f32(&a).expect("simd_exp_f32");
                check_f32(&label, &read_back_f32(&r), exp, tolerance::F32_FAST);
            }
            "float64" => {
                let a = make_cpu_f64(a_data, shape, false);
                let r = simd_exp_f64(&a).expect("simd_exp_f64");
                check_f64(&label, &read_back_f64(&r), exp, tolerance::F64_FAST);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_fast_log_and_sigmoid_special() {
    let file = load_fixtures();
    for op_name in ["fast_log_special", "fast_sigmoid_special"] {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
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
                    let a = make_cpu_f32(a_data, shape, false);
                    let r = match op_name {
                        "fast_log_special" => fast_log(&a).expect("fast_log"),
                        _ => fast_sigmoid(&a).expect("fast_sigmoid"),
                    };
                    let actual = read_back_f32(&r);
                    if op_name == "fast_log_special" {
                        // #1931 pin: fast_log's f32 vector kernel assumes a
                        // normalized mantissa; the subnormal input (index 5
                        // = 1e-40) yields -88.021 instead of torch's
                        // -92.10341 (fixture). Pin the current divergence;
                        // when #1931 lands this assert fails — retire it
                        // and compare the whole vector against the fixture.
                        assert!(
                            (f64::from(actual[5]) - exp[5]).abs() > 1.0,
                            "{label}: fast_log(subnormal) now matches torch \
                             — #1931 appears fixed; retire this pin"
                        );
                        check_f32(
                            &format!("{label} non-pinned head"),
                            &actual[..5],
                            &exp[..5],
                            tolerance::F32_FAST,
                        );
                    } else {
                        check_f32(&label, &actual, exp, tolerance::F32_FAST);
                    }
                }
                "float64" => {
                    let a = make_cpu_f64(a_data, shape, false);
                    let r = match op_name {
                        "fast_log_special" => fast_log(&a).expect("fast_log"),
                        _ => fast_sigmoid(&a).expect("fast_sigmoid"),
                    };
                    check_f64(&label, &read_back_f64(&r), exp, tolerance::F64_FAST);
                }
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn cpu_sum_and_nansum_special() {
    let file = load_fixtures();
    for op_name in ["sum_special", "nansum_special"] {
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
                        _ => nansum(&a).expect("nansum"),
                    };
                    check_f32(&label, &read_back_f32(&r), exp, tolerance::F32_REDUCTION);
                }
                "float64" => {
                    let a = make_cpu_f64(a_data, shape, false);
                    let r = match op_name {
                        "sum_special" => sum(&a).expect("sum"),
                        _ => nansum(&a).expect("nansum"),
                    };
                    check_f64(&label, &read_back_f64(&r), exp, tolerance::F64_REDUCTION);
                }
                _ => unreachable!(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cat I — non-contiguous (transpose-view) lanes (CORE-199 / #1893,
// CORE-132 / #1826)
// ---------------------------------------------------------------------------
//
// The fixture stores the contiguous base buffer; the runner builds the
// non-contiguous view with `.transpose(0, 1)` (input_transpose flag).
// Probed at HEAD: every elementwise CPU kernel in this suite rejects
// non-contiguous views ("tensor is not contiguous"). torch computes all of
// these (expected values are in the fixture, out_values) — pinned as
// expect_err on #1826, single contract, retire-on-fix.

#[test]
fn cpu_transpose_view_lanes() {
    let file = load_fixtures();
    let tview_ops = [
        "add_tview",
        "mul_tview",
        "sqrt_tview",
        "fast_sigmoid_tview",
        "sum_tview",
        "mean_tview",
        "logsumexp_tview",
        "sum_axis_tview",
    ];
    for op_name in tview_ops {
        let cases = cases_for(&file, op_name, "cpu");
        assert!(!cases.is_empty(), "no fixtures for {op_name}");
        for f in cases {
            assert_eq!(
                f.input_transpose,
                Some(true),
                "{op_name}: tview fixture row missing input_transpose flag"
            );
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
                        "add_tview" => add(&v, &v).expect("add"),
                        "mul_tview" => mul(&v, &v).expect("mul"),
                        "sqrt_tview" => sqrt(&v).expect("sqrt"),
                        "fast_sigmoid_tview" => fast_sigmoid(&v).expect("fast_sigmoid"),
                        "sum_tview" => sum(&v).expect("sum"),
                        "mean_tview" => mean(&v).expect("mean"),
                        "logsumexp_tview" => logsumexp(&v).expect("logsumexp"),
                        "sum_axis_tview" => sum_axis(&v, 0).expect("sum_axis"),
                        _ => unreachable!(),
                    };
                    let tol = match op_name {
                        "sqrt_tview" | "fast_sigmoid_tview" | "logsumexp_tview" => {
                            tolerance::F32_TRANSCENDENTAL
                        }
                        "sum_tview" | "mean_tview" | "sum_axis_tview" => tolerance::F32_REDUCTION,
                        _ => tolerance::F32_ELEMENTWISE,
                    };
                    check_f32(&label, &read_back_f32(&r), exp, tol);
                }
                "float64" => {
                    let v = make_cpu_f64(a_data, shape, false)
                        .transpose(0, 1)
                        .expect("transpose");
                    assert!(!v.is_contiguous(), "{label}: view must be non-contiguous");
                    let r = match op_name {
                        "add_tview" => add(&v, &v).expect("add"),
                        "mul_tview" => mul(&v, &v).expect("mul"),
                        "sqrt_tview" => sqrt(&v).expect("sqrt"),
                        "fast_sigmoid_tview" => fast_sigmoid(&v).expect("fast_sigmoid"),
                        "sum_tview" => sum(&v).expect("sum"),
                        "mean_tview" => mean(&v).expect("mean"),
                        "logsumexp_tview" => logsumexp(&v).expect("logsumexp"),
                        "sum_axis_tview" => sum_axis(&v, 0).expect("sum_axis"),
                        _ => unreachable!(),
                    };
                    let tol = match op_name {
                        "sqrt_tview" | "fast_sigmoid_tview" | "logsumexp_tview" => {
                            tolerance::F64_TRANSCENDENTAL
                        }
                        "sum_tview" | "mean_tview" | "sum_axis_tview" => tolerance::F64_REDUCTION,
                        _ => tolerance::F64_ELEMENTWISE,
                    };
                    check_f64(&label, &read_back_f64(&r), exp, tol);
                }
                _ => unreachable!(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU paths — gated on the `gpu` feature, NOT `#[ignore]`d, per the dispatch.
//
// Cat A binary + unary + pow exercise the GPU forward + autograd lanes.
// Cat B where_ runs end-to-end on CUDA (the `WhereBackward` materializes
// gradients on CPU and uploads — that's the documented behaviour, not a
// drift). Cat E sum / sum_axis / mean run the GPU reduction kernels via
// the same op functions. Cat F in-place add_/sub_/mul_/div_/add_scalar_/
// mul_scalar_/fill_/zero_/clamp_ run the GPU update_storage / update_data
// paths on f32 (the only dtype the in-place GPU fast path supports for
// add_/sub_/mul_/div_); the storage-identity contract is the same.
// ---------------------------------------------------------------------------

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
                "fixtures/elementwise.json was generated without CUDA — \
                 regenerate on a CUDA-enabled host before running --features gpu tests"
            );
        }
    }

    fn run_binary_gpu(op_name: &str, op: BinaryOp) {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        let cases = cases_for(&file, op_name, "cuda:0");
        assert!(!cases.is_empty(), "no CUDA fixtures for op {op_name:?}");
        for f in cases {
            let label = format!("{op_name} cuda:0 tag={:?} dtype={}", f.tag, f.dtype);
            let a_shape = f.a_shape.as_ref().unwrap();
            let b_shape = f.b_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let b_data = f.b_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let expected = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            let grad_a_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let grad_b_exp = f.grad_b.as_ref().map(F64ListSentinel::as_slice).unwrap();
            match f.dtype.as_str() {
                "float32" => {
                    let a = upload_f32(make_cpu_f32(a_data, a_shape, false), Device::Cuda(0));
                    let b = upload_f32(make_cpu_f32(b_data, b_shape, false), Device::Cuda(0));
                    let c = op.apply_f32(&a, &b);
                    assert!(c.is_cuda(), "{label}: result not on CUDA");
                    check_f32(
                        &format!("{label} fwd"),
                        &read_back_f32(&c),
                        expected,
                        tolerance::F32_GPU,
                    );

                    let a_g = upload_f32(make_cpu_f32(a_data, a_shape, true), Device::Cuda(0));
                    let b_g = upload_f32(make_cpu_f32(b_data, b_shape, true), Device::Cuda(0));
                    let out = op.apply_f32(&a_g, &b_g);
                    grad_sum(&out).expect("sum").backward().expect("backward");
                    let ga = a_g.grad().unwrap().expect("grad_a");
                    let gb = b_g.grad().unwrap().expect("grad_b");
                    check_f32(
                        &format!("{label} grad_a"),
                        &read_back_f32(&ga),
                        grad_a_exp,
                        tolerance::F32_GPU,
                    );
                    check_f32(
                        &format!("{label} grad_b"),
                        &read_back_f32(&gb),
                        grad_b_exp,
                        tolerance::F32_GPU,
                    );
                }
                "float64" => {
                    let a = upload_f64(make_cpu_f64(a_data, a_shape, false), Device::Cuda(0));
                    let b = upload_f64(make_cpu_f64(b_data, b_shape, false), Device::Cuda(0));
                    let c = op.apply_f64(&a, &b);
                    assert!(c.is_cuda());
                    check_f64(
                        &format!("{label} fwd"),
                        &read_back_f64(&c),
                        expected,
                        tolerance::F64_GPU,
                    );

                    let a_g = upload_f64(make_cpu_f64(a_data, a_shape, true), Device::Cuda(0));
                    let b_g = upload_f64(make_cpu_f64(b_data, b_shape, true), Device::Cuda(0));
                    let out = op.apply_f64(&a_g, &b_g);
                    grad_sum(&out).expect("sum").backward().expect("backward");
                    let ga = a_g.grad().unwrap().expect("grad_a");
                    let gb = b_g.grad().unwrap().expect("grad_b");
                    check_f64(
                        &format!("{label} grad_a"),
                        &read_back_f64(&ga),
                        grad_a_exp,
                        tolerance::F64_GPU,
                    );
                    check_f64(
                        &format!("{label} grad_b"),
                        &read_back_f64(&gb),
                        grad_b_exp,
                        tolerance::F64_GPU,
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn gpu_add() {
        run_binary_gpu("add", BinaryOp::Add);
    }

    #[test]
    fn gpu_sub() {
        run_binary_gpu("sub", BinaryOp::Sub);
    }

    #[test]
    fn gpu_mul() {
        run_binary_gpu("mul", BinaryOp::Mul);
    }

    #[test]
    fn gpu_div() {
        run_binary_gpu("div", BinaryOp::Div);
    }

    fn run_unary_gpu(op_name: &str, op: UnaryOp) {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        let cases = cases_for(&file, op_name, "cuda:0");
        assert!(!cases.is_empty(), "no CUDA fixtures for {op_name}");
        for f in cases {
            let label = format!("{op_name} cuda:0 tag={:?} dtype={}", f.tag, f.dtype);
            let shape = f.a_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let expected = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            let grad_a_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();
            match f.dtype.as_str() {
                "float32" => {
                    let a = upload_f32(make_cpu_f32(a_data, shape, false), Device::Cuda(0));
                    let c = op.apply_f32(&a);
                    assert!(c.is_cuda());
                    check_f32(
                        &format!("{label} fwd"),
                        &read_back_f32(&c),
                        expected,
                        tolerance::F32_GPU,
                    );
                    let a_g = upload_f32(make_cpu_f32(a_data, shape, true), Device::Cuda(0));
                    let out = op.apply_f32(&a_g);
                    grad_sum(&out).expect("sum").backward().expect("backward");
                    let ga = a_g.grad().unwrap().expect("grad_a");
                    check_f32(
                        &format!("{label} grad_a"),
                        &read_back_f32(&ga),
                        grad_a_exp,
                        tolerance::F32_GPU,
                    );
                }
                "float64" => {
                    let a = upload_f64(make_cpu_f64(a_data, shape, false), Device::Cuda(0));
                    let c = op.apply_f64(&a);
                    assert!(c.is_cuda());
                    check_f64(
                        &format!("{label} fwd"),
                        &read_back_f64(&c),
                        expected,
                        tolerance::F64_GPU,
                    );
                    // #782 fixed: `abs_backward_f64` (and the other
                    // *_backward_f64 ops counted in the umbrella issue)
                    // are now implemented; the f64 GPU autograd lane
                    // runs live for every unary op including `abs`.
                    let _ = op_name;
                    let a_g = upload_f64(make_cpu_f64(a_data, shape, true), Device::Cuda(0));
                    let out = op.apply_f64(&a_g);
                    grad_sum(&out).expect("sum").backward().expect("backward");
                    let ga = a_g.grad().unwrap().expect("grad_a");
                    check_f64(
                        &format!("{label} grad_a"),
                        &read_back_f64(&ga),
                        grad_a_exp,
                        tolerance::F64_GPU,
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn gpu_neg() {
        run_unary_gpu("neg", UnaryOp::Neg);
    }

    #[test]
    fn gpu_abs() {
        run_unary_gpu("abs", UnaryOp::Abs);
    }

    #[test]
    fn gpu_sqrt() {
        run_unary_gpu("sqrt", UnaryOp::Sqrt);
    }

    #[test]
    fn gpu_pow() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        let cases = cases_for(&file, "pow", "cuda:0");
        assert!(!cases.is_empty(), "no CUDA fixtures for pow");
        for f in cases {
            let label = format!("pow cuda:0 dtype={} exp={:?}", f.dtype, f.exp);
            let shape = f.a_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let exp = f.exp.expect("exp");
            let expected = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            let grad_a_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();
            match f.dtype.as_str() {
                "float32" => {
                    let a = upload_f32(make_cpu_f32(a_data, shape, false), Device::Cuda(0));
                    let c = pow(&a, exp).expect("pow");
                    check_f32(
                        &format!("{label} fwd"),
                        &read_back_f32(&c),
                        expected,
                        tolerance::F32_GPU,
                    );
                    let a_g = upload_f32(make_cpu_f32(a_data, shape, true), Device::Cuda(0));
                    let out = pow(&a_g, exp).expect("pow");
                    grad_sum(&out).expect("sum").backward().expect("backward");
                    let ga = a_g.grad().unwrap().expect("grad_a");
                    check_f32(
                        &format!("{label} grad_a"),
                        &read_back_f32(&ga),
                        grad_a_exp,
                        tolerance::F32_GPU,
                    );
                }
                "float64" => {
                    // #781 fixed: PTX JIT now compiles for the f64 pow
                    // kernel.
                    // #783 fixed: the inline log+exp template was tightened
                    // (half-step argument reduction restricting |f| from
                    // ~1/3 to ~0.172, degree-7 odd-power Horner instead of
                    // degree-5, and a 2-double Cody-Waite split for the
                    // n*ln(2) reconstruction step). Worst-case relative
                    // error is now ~few-ULP across the supported range,
                    // well inside `F64_TRANSCENDENTAL = 1e-10`.
                    let a = upload_f64(make_cpu_f64(a_data, shape, false), Device::Cuda(0));
                    let c = pow(&a, exp).expect("pow");
                    check_f64(
                        &format!("{label} fwd"),
                        &read_back_f64(&c),
                        expected,
                        tolerance::F64_TRANSCENDENTAL,
                    );
                    let a_g = upload_f64(make_cpu_f64(a_data, shape, true), Device::Cuda(0));
                    let out = pow(&a_g, exp).expect("pow");
                    grad_sum(&out).expect("sum").backward().expect("backward");
                    let ga = a_g.grad().unwrap().expect("grad_a");
                    check_f64(
                        &format!("{label} grad_a"),
                        &read_back_f64(&ga),
                        grad_a_exp,
                        tolerance::F64_TRANSCENDENTAL,
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn gpu_where() {
        ensure_cuda_backend();
        run_where_for_device("cuda:0", Device::Cuda(0));
    }

    /// GPU sum forward path — exercises the on-device sum kernel via
    /// `grad_fns::reduction::sum` (which dispatches to `backend.sum_f32` /
    /// `backend.sum_f64`). The non-differentiable
    /// `ops::elementwise::sum` is a CPU-only path that calls `.data()`;
    /// invoking it on a CUDA tensor returns `GpuTensorNotAccessible` by
    /// design, so this test routes through the differentiable variant
    /// without enabling autograd (no `requires_grad`). The CPU-only nature
    /// of `ops::elementwise::sum` is documented in its rustdoc.
    #[test]
    fn gpu_sum_forward_only() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        for f in cases_for(&file, "sum", "cuda:0") {
            let label = format!("sum cuda:0 dtype={}", f.dtype);
            let shape = f.a_shape.as_ref().unwrap();
            let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
            let exp = f
                .out_values
                .as_ref()
                .map(F64ListSentinel::as_slice)
                .unwrap();
            match f.dtype.as_str() {
                "float32" => {
                    let a = upload_f32(make_cpu_f32(a_data, shape, false), Device::Cuda(0));
                    let s = grad_sum(&a).expect("sum");
                    // Sweep rows reduce up to 10007 summands — k-aware band
                    // (see tolerance::accum_tol_f32) on top of the GPU base.
                    check_f32(
                        &label,
                        &read_back_f32(&s),
                        exp,
                        tolerance::F32_GPU.max(tolerance::accum_tol_f32(a_data.len())),
                    );
                }
                "float64" => {
                    let a = upload_f64(make_cpu_f64(a_data, shape, false), Device::Cuda(0));
                    let s = grad_sum(&a).expect("sum");
                    check_f64(
                        &label,
                        &read_back_f64(&s),
                        exp,
                        tolerance::F64_GPU.max(tolerance::accum_tol_f64(a_data.len())),
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    /// In-place add_/sub_/mul_/div_ on f32 GPU tensors: uses the GPU fast
    /// path that swaps the storage buffer through the same Arc. The
    /// storage-identity contract still holds.
    #[test]
    fn gpu_inplace_f32() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        for op_name in ["add_", "sub_", "mul_", "div_"] {
            for f in cases_for(&file, op_name, "cuda:0") {
                if f.dtype != "float32" {
                    continue; // GPU in-place fast path is f32 only
                }
                let label = format!("{op_name} cuda:0 dtype={}", f.dtype);
                let a_shape = f.a_shape.as_ref().unwrap();
                let b_shape = f.b_shape.as_ref().unwrap();
                let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
                let b_data = f.b_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
                let exp = f
                    .out_values
                    .as_ref()
                    .map(F64ListSentinel::as_slice)
                    .unwrap();
                let t = upload_f32(make_cpu_f32(a_data, a_shape, false), Device::Cuda(0));
                let other = upload_f32(make_cpu_f32(b_data, b_shape, false), Device::Cuda(0));
                let before_id = storage_arc_id(&t);
                match op_name {
                    "add_" => {
                        t.add_(&other).expect("add_");
                    }
                    "sub_" => {
                        t.sub_(&other).expect("sub_");
                    }
                    "mul_" => {
                        t.mul_(&other).expect("mul_");
                    }
                    "div_" => {
                        t.div_(&other).expect("div_");
                    }
                    _ => unreachable!(),
                }
                let after_id = storage_arc_id(&t);
                assert_eq!(
                    before_id, after_id,
                    "{label}: GPU in-place op must mutate through the same storage Arc"
                );
                assert!(t.is_cuda(), "{label}: tensor left CUDA after in-place op");
                check_f32(
                    &format!("{label} value"),
                    &read_back_f32(&t),
                    exp,
                    tolerance::F32_GPU,
                );
            }
        }
    }

    /// In-place scalar / fill_ / zero_ / clamp_ on GPU. These dispatch
    /// through the same data_vec → update_data path used on CPU; on CUDA,
    /// `update_data` re-uploads via `cpu_to_gpu`. The Arc identity is
    /// preserved.
    #[test]
    fn gpu_inplace_scalar_fill_clamp_f32() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        for op_name in ["add_scalar_", "mul_scalar_", "fill_", "zero_", "clamp_"] {
            for f in cases_for(&file, op_name, "cuda:0") {
                if f.dtype != "float32" {
                    continue;
                }
                let label = format!("{op_name} cuda:0 dtype={}", f.dtype);
                let shape = f.a_shape.as_ref().unwrap();
                let a_data = f.a_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
                let exp = f
                    .out_values
                    .as_ref()
                    .map(F64ListSentinel::as_slice)
                    .unwrap();
                let t = upload_f32(make_cpu_f32(a_data, shape, false), Device::Cuda(0));
                let before_id = storage_arc_id(&t);
                match op_name {
                    "add_scalar_" => {
                        t.add_scalar_(f.scalar.expect("scalar") as f32)
                            .expect("add_scalar_");
                    }
                    "mul_scalar_" => {
                        t.mul_scalar_(f.scalar.expect("scalar") as f32)
                            .expect("mul_scalar_");
                    }
                    "fill_" => {
                        t.fill_(f.scalar.expect("scalar") as f32).expect("fill_");
                    }
                    "zero_" => {
                        t.zero_().expect("zero_");
                    }
                    "clamp_" => {
                        let lo = f.min.expect("min") as f32;
                        let hi = f.max.expect("max") as f32;
                        t.clamp_(lo, hi).expect("clamp_");
                    }
                    _ => unreachable!(),
                }
                let after_id = storage_arc_id(&t);
                assert_eq!(before_id, after_id, "{label}: storage Arc replaced");
                assert!(t.is_cuda(), "{label}: tensor left CUDA after in-place op");
                check_f32(
                    &format!("{label} value"),
                    &read_back_f32(&t),
                    exp,
                    tolerance::F32_GPU,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sanity: assert the fixture file has every op we expect.
//
// Keeps the elementwise.json contract honest when the regen script changes.
// ---------------------------------------------------------------------------

#[test]
fn fixture_file_covers_every_phase21_op() {
    let file = load_fixtures();
    let mut by_op: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for f in &file.fixtures {
        *by_op.entry(f.op.as_str()).or_insert(0) += 1;
    }
    let required = [
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "sqrt",
        "pow",
        "pow_zero_exp",
        "sqrt_zero",
        "div_zero",
        "where",
        "binary_map_maxmin",
        "scalar_map_sqplus",
        "unary_map_tan",
        "fast_add",
        "fast_sub",
        "fast_mul",
        "fast_div",
        "fast_exp",
        "fast_log",
        "fast_sigmoid",
        "fast_tanh",
        "fast_sin",
        "fast_cos",
        "simd_add_f32",
        "simd_mul_f32",
        "simd_exp_f32",
        "simd_log_f32",
        "simd_sqrt_f32",
        "simd_add_f64",
        "simd_mul_f64",
        "simd_exp_f64",
        "sum",
        "sum_axis",
        "mean",
        "nansum",
        "nanmean",
        "logsumexp",
        "logsumexp_dim",
        "add_",
        "sub_",
        "mul_",
        "div_",
        "add_scalar_",
        "mul_scalar_",
        "fill_",
        "zero_",
        "clamp_",
        // CORE-199 / #1893 lanes:
        "logsumexp_special",
        "logsumexp_dim_special",
        "fast_exp_special",
        "simd_exp_special",
        "fast_log_special",
        "fast_sigmoid_special",
        "sum_special",
        "nansum_special",
        "add_tview",
        "mul_tview",
        "sqrt_tview",
        "fast_sigmoid_tview",
        "sum_tview",
        "mean_tview",
        "logsumexp_tview",
        "sum_axis_tview",
    ];
    for r in required {
        let n = by_op.get(r).copied().unwrap_or(0);
        assert!(n > 0, "fixture file missing op {r:?}");
    }
}
