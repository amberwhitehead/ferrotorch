//! Conformance Phase 2.3 — `ferrotorch-core` shape, indexing, and view ops
//! parity against PyTorch.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/765>.
//! Parent: #759.
//!
//! Source files exercised (per the dispatch):
//! - `ferrotorch-core/src/methods.rs` — Tensor shape methods + free functions
//!   (`reshape_t`, `view_t`, `flatten_t`, `squeeze_t`, `unsqueeze_t`,
//!   `permute_t`, `transpose`, `narrow_t`, `contiguous_t`, `chunk_t`,
//!   `split_t`).
//! - `ferrotorch-core/src/grad_fns/shape.rs` — `reshape`, `flatten`,
//!   `squeeze`, `unsqueeze`, `transpose_2d`, `cat`, `expand` and their
//!   `*Backward` structs.
//! - `ferrotorch-core/src/grad_fns/indexing.rs` — `index_select_1d`,
//!   `masked_fill` plus the canonical `*Backward` structs.
//! - `ferrotorch-core/src/ops/indexing.rs` — `gather`, `scatter`,
//!   `scatter_value`, `scatter_add`, `where_cond`.
//! - `ferrotorch-core/src/ops/tensor_ops.rs` — `triu`, `tril`, `diag`,
//!   `diagflat`, `roll`, `cdist`.
//! - `ferrotorch-core/src/ops/search.rs` — `searchsorted`, `bucketize`,
//!   `unique`, `unique_consecutive`, `histc`, `meshgrid`, `topk`.
//! - `ferrotorch-core/src/shape.rs` — pure shape utility helpers
//!   (`broadcast_shapes`, `numel`, `checked_numel`, `checked_byte_count`,
//!   `c_contiguous_strides`, `checked_c_contiguous_strides`,
//!   `channels_last_strides`, `checked_channels_last_strides`,
//!   `channels_last_3d_strides`, `checked_channels_last_3d_strides`,
//!   `normalize_axis`, `check_shapes_match`).
//! - `ferrotorch-core/src/stride_tricks.rs` — `as_strided`,
//!   `as_strided_copy`, `as_strided_scatter` and `AsStridedBackward`.
//!
//! Coverage strategy:
//! * **Cat A** (forwards with autograd): CPU + GPU + autograd assertions
//!   for every op that has GPU support. Tolerance is *bit-exact* via the
//!   `assert_close_*` helpers below (these are pure data-movement ops, no
//!   arithmetic). The single exception is `cdist`, which involves
//!   `pow(p)` / `pow(1/p)` — that uses `F32_TRANSCENDENTAL_*` tolerance.
//! * **Indexing/search/tensor_ops without GPU support** are tested CPU-only
//!   via fixtures. The GPU lane runs the same op on a CUDA tensor and
//!   asserts the call returns `Err(FerrotorchError::NotImplementedOnCuda)`
//!   — this is the PyTorch-parity policy from `rust-gpu-discipline` §3
//!   (raise rather than silently fall back).
//! * **Cat B** (shape utility helpers): direct unit-test assertions against
//!   the documented PyTorch semantics — no torch invocation needed in the
//!   test body, but the fixture file pins the expected outputs.
//! * **Cat C** (backward grad_fn structs): implicit coverage via the
//!   relevant forward op's `grad_a` assertion (no phantom tests).
//!
//! Edge cases the dispatch mandates and this file exercises:
//! * Negative axis (`squeeze(-2)`, `unsqueeze(-1)`, `cat(dim=-1)`).
//! * Empty-tensor reshape (the `reshape: empty_to_2d` fixture).
//! * Non-contiguous input -> contiguous output (the `contiguous:
//!   transpose_then_contiguous` fixture, which regresses the historical
//!   `view_reshape` silent-CPU-demote bug).
//! * Broadcast-aware expand (size-1 dim -> larger; the test additionally
//!   asserts the GPU result lands on `Device::Cuda`, regressing any
//!   accidental host bounce).
//! * Multi-dim gather / scatter (2-D dim=0 and dim=1 fixtures).
//! * `cat` on single-tensor / many-tensor / negative-axis inputs; explicit
//!   error path for empty-list cat.
//! * `narrow` with start=0, length=full, partial — every slice-edge case.

use std::path::PathBuf;

use serde::Deserialize;
use serde::de::{self, Deserializer, SeqAccess, Visitor};

use ferrotorch_core::grad_fns::indexing::{
    GatherBackward, IndexSelectBackward, IndexSelectDimBackward, MaskedFillBackward,
    MaskedSelectBackward, ScatterAddBackward, ScatterBackward, WhereCondBackward, index_select_1d,
    index_select_1d_it, index_select_dim, masked_fill, masked_fill_bt, masked_select_bcast,
};
use ferrotorch_core::grad_fns::shape::{
    CatBackward, ExpandBackward, FlattenBackward, ReshapeBackward, SplitBackward, SqueezeBackward,
    TransposeBackward, UnsqueezeBackward, cat, expand, flatten, reshape, squeeze, transpose_2d,
    unsqueeze,
};
use ferrotorch_core::ops::indexing::{
    gather, masked_select, scatter, scatter_add, scatter_value, where_cond,
};
use ferrotorch_core::ops::search::{
    MeshIndexing, bucketize, histc, meshgrid, meshgrid_indexing, searchsorted, topk, unique,
    unique_consecutive,
};
use ferrotorch_core::ops::tensor_ops::{cdist, diag, diagflat, roll, tril, triu};
use ferrotorch_core::shape::{
    broadcast_shapes, c_contiguous_strides, channels_last_3d_strides, channels_last_strides,
    check_shapes_match, checked_byte_count, checked_c_contiguous_strides,
    checked_channels_last_3d_strides, checked_channels_last_strides, checked_numel, normalize_axis,
    numel,
};
use ferrotorch_core::stride_tricks::{
    AsStridedBackward, AsStridedScatterBackward, as_strided, as_strided_copy, as_strided_scatter,
};
use ferrotorch_core::{
    BoolTensor, Device, FerrotorchError, GradFn, IntElement, IntTensor, Tensor, TensorStorage,
    chunk_t, contiguous_t, permute_t, split_t, view_t,
};

/// Free-function `narrow_t` is not re-exported at the crate root; the
/// inherent `Tensor::narrow` method calls into it. Wrap so the tests can
/// dispatch a free-function-style call without leaking that detail.
fn narrow_t<T: ferrotorch_core::Float>(
    input: &Tensor<T>,
    dim: usize,
    start: usize,
    length: usize,
) -> ferrotorch_core::FerrotorchResult<Tensor<T>> {
    input.narrow(dim, start, length)
}

// ---------------------------------------------------------------------------
// Tolerance helpers — phase 2.3 ops are mostly bit-exact data movement.
// `assert_eq_bits` is used for the metadata-only / pure-copy ops.
// `assert_close_*` is used for `cdist` (the only arithmetic op in scope) and
// for `expand`'s grad-reduction (which sums upstream gradients along
// broadcast axes).
// ---------------------------------------------------------------------------

mod tolerance {
    /// Bit-exact for ops that never touch arithmetic — copy / view / scatter.
    /// We still call this an "f32 tolerance" so the helper signature stays
    /// uniform with the other phases; the actual asserted bound is 0.0.
    pub const F32_BITEXACT: f32 = 0.0;
    pub const F64_BITEXACT: f64 = 0.0;

    /// Used by `cdist` (single arithmetic op in this phase) and by `expand`
    /// gradient reductions (one fused-add per broadcast axis).
    pub const F32_TRANSCENDENTAL_CPU: f32 = 1e-5;
    pub const F64_TRANSCENDENTAL_CPU: f64 = 1e-9;
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F32_TRANSCENDENTAL_GPU: f32 = 1e-4;
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F64_TRANSCENDENTAL_GPU: f64 = 1e-9;

    /// Reduction tolerance — used for grad assertions on ops where the
    /// backward fans-in (cat, split, expand) and tiny rounding can creep
    /// in on cuBLAS-style accumulators. CPU is bit-exact; GPU gets a slack.
    #[allow(dead_code, reason = "consumed by `gpu` cfg-gated module")]
    pub const F32_REDUCTION_GPU: f32 = 1e-5;

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
// Strict-JSON-compatible f64 list deserializer (mirrors elementwise/reduction
// — supports "Infinity" / "-Infinity" / "NaN" sentinels emitted by the
// fixture script).
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
// Fixture deserialization — single broad struct because the fixture file
// covers many ops with very different shapes. Each op-specific test body
// only reads the fields it cares about (panics if a required field is
// missing — that catches stale fixtures).
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
    #[serde(default)]
    dtype: Option<String>,
    #[serde(default)]
    device: Option<String>,

    // Shape ops
    #[serde(default)]
    in_shape: Option<Vec<usize>>,
    #[serde(default)]
    new_shape: Option<Vec<i64>>,
    #[serde(default)]
    #[allow(
        dead_code,
        reason = "deserialized for fixture-shape stability and future shape-checks"
    )]
    out_shape: Option<Vec<usize>>,
    #[serde(default)]
    in_data: Option<F64ListSentinel>,
    #[serde(default)]
    out_values: Option<F64ListSentinel>,
    #[serde(default)]
    grad_a: Option<F64ListSentinel>,

    #[serde(default)]
    axis: Option<i64>,
    #[serde(default)]
    dim: Option<i64>,
    #[serde(default)]
    dim0: Option<usize>,
    #[serde(default)]
    dim1: Option<usize>,
    #[serde(default)]
    dims: Option<Vec<usize>>,
    #[serde(default)]
    start: Option<usize>,
    #[serde(default)]
    length: Option<usize>,

    // cat / split / chunk / meshgrid
    #[serde(default)]
    tensor_shapes: Option<Vec<Vec<usize>>>,
    #[serde(default)]
    tensor_data: Option<Vec<F64ListSentinel>>,
    #[serde(default)]
    tensor_grads: Option<Vec<F64ListSentinel>>,
    #[serde(default)]
    split_sizes: Option<Vec<usize>>,
    #[serde(default)]
    chunks: Option<usize>,
    #[serde(default)]
    chunks_shapes: Option<Vec<Vec<usize>>>,
    #[serde(default)]
    chunks_values: Option<Vec<F64ListSentinel>>,

    // expand
    // (uses new_shape via Vec<i64>; we coerce to Vec<usize> at use site)

    // as_strided family
    #[serde(default)]
    size: Option<Vec<usize>>,
    #[serde(default)]
    stride: Option<Vec<i64>>,
    #[serde(default)]
    storage_offset: Option<usize>,
    #[serde(default)]
    src_shape: Option<Vec<usize>>,
    #[serde(default)]
    src_data: Option<F64ListSentinel>,

    // gather/scatter/scatter_add
    #[serde(default)]
    index: Option<Vec<usize>>,
    #[serde(default)]
    index_shape: Option<Vec<usize>>,

    // where_cond / masked_fill
    #[serde(default)]
    condition: Option<Vec<bool>>,
    #[serde(default)]
    mask: Option<Vec<bool>>,
    #[serde(default)]
    value: Option<f64>,
    #[serde(default)]
    x_data: Option<F64ListSentinel>,
    #[serde(default)]
    y_data: Option<F64ListSentinel>,
    #[serde(default)]
    grad_x: Option<F64ListSentinel>,
    #[serde(default)]
    grad_y: Option<F64ListSentinel>,

    // tensor_ops
    #[serde(default)]
    diagonal: Option<i64>,
    #[serde(default)]
    shifts: Option<i64>,

    // cdist
    #[serde(default)]
    x1_shape: Option<Vec<usize>>,
    #[serde(default)]
    x1_data: Option<F64ListSentinel>,
    #[serde(default)]
    x2_shape: Option<Vec<usize>>,
    #[serde(default)]
    x2_data: Option<F64ListSentinel>,
    #[serde(default)]
    p: Option<f64>,

    // searchsorted / bucketize
    #[serde(default)]
    boundaries_shape: Option<Vec<usize>>,
    #[serde(default)]
    boundaries_data: Option<F64ListSentinel>,
    #[serde(default)]
    values_shape: Option<Vec<usize>>,
    #[serde(default)]
    values_data: Option<F64ListSentinel>,
    #[serde(default)]
    input_shape: Option<Vec<usize>>,
    #[serde(default)]
    input_data: Option<F64ListSentinel>,
    #[serde(default)]
    right: Option<bool>,
    #[serde(default)]
    out_indices: Option<Vec<usize>>,
    #[serde(default)]
    out_inverse: Option<Vec<usize>>,
    #[serde(default)]
    out_counts: Option<Vec<usize>>,

    // histc
    #[serde(default)]
    bins: Option<usize>,
    #[serde(default)]
    min: Option<f64>,
    #[serde(default)]
    max: Option<f64>,

    // meshgrid (separate field names because it produces N parallel grids
    // and would otherwise collide with the flat `out_values` deserializer).
    #[serde(default)]
    input_shapes: Option<Vec<Vec<usize>>>,
    #[serde(default)]
    mg_input_data: Option<Vec<F64ListSentinel>>,
    #[serde(default)]
    #[allow(dead_code, reason = "kept for fixture-shape stability")]
    mg_out_shapes: Option<Vec<Vec<usize>>>,
    #[serde(default)]
    mg_out_values: Option<Vec<F64ListSentinel>>,

    // topk
    #[serde(default)]
    k: Option<usize>,
    #[serde(default)]
    largest: Option<bool>,

    // shape helpers
    #[serde(default)]
    a: Option<Vec<i64>>,
    #[serde(default)]
    b: Option<Vec<i64>>,
    #[serde(default)]
    expected: Option<Vec<usize>>,
    #[serde(default)]
    shape: Option<Vec<usize>>,
    #[serde(default)]
    expected_numel: Option<usize>,
    #[serde(default)]
    expected_strides: Option<Vec<i64>>,
    #[serde(default)]
    ndim: Option<usize>,
    #[serde(default)]
    expected_axis: Option<usize>,
}

fn load_fixtures() -> FixtureFile {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures")
        .join("shape.json");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. Regenerate via \
             `python3 scripts/regenerate_shape_fixtures.py`",
            p.display()
        )
    });
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn cases_for<'a>(file: &'a FixtureFile, op: &str, device: &str) -> Vec<&'a Fixture> {
    file.fixtures
        .iter()
        .filter(|f| f.op == op && f.device.as_deref() == Some(device))
        .collect()
}

fn cases_op_only<'a>(file: &'a FixtureFile, op: &str) -> Vec<&'a Fixture> {
    file.fixtures.iter().filter(|f| f.op == op).collect()
}

// ---------------------------------------------------------------------------
// Device-transparent helpers
// ---------------------------------------------------------------------------

fn read_back_f32(t: &Tensor<f32>) -> Vec<f32> {
    if t.is_cpu() {
        t.data_vec().expect("read CPU data")
    } else {
        let cpu = t.cpu().expect("D2H readback");
        cpu.data_vec().expect("read CPU data after readback")
    }
}

fn read_back_f64(t: &Tensor<f64>) -> Vec<f64> {
    if t.is_cpu() {
        t.data_vec().expect("read CPU data")
    } else {
        let cpu = t.cpu().expect("D2H readback");
        cpu.data_vec().expect("read CPU data after readback")
    }
}

fn read_back_int<I: IntElement>(t: &IntTensor<I>) -> Vec<I> {
    if t.device() == Device::Cpu {
        t.data().expect("read CPU int data").to_vec()
    } else {
        let cpu = t.to(Device::Cpu).expect("D2H int readback");
        cpu.data()
            .expect("read CPU int data after readback")
            .to_vec()
    }
}

fn read_back_f16_bits(t: &Tensor<half::f16>) -> Vec<u16> {
    let values = if t.is_cpu() {
        t.data_vec().expect("read CPU f16 data")
    } else {
        t.cpu()
            .expect("D2H f16 readback")
            .data_vec()
            .expect("read CPU f16 data after readback")
    };
    values.iter().map(|v| v.to_bits()).collect()
}

fn read_back_bf16_bits(t: &Tensor<half::bf16>) -> Vec<u16> {
    let values = if t.is_cpu() {
        t.data_vec().expect("read CPU bf16 data")
    } else {
        t.cpu()
            .expect("D2H bf16 readback")
            .data_vec()
            .expect("read CPU bf16 data after readback")
    };
    values.iter().map(|v| v.to_bits()).collect()
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

/// Upload to CUDA as a true CUDA LEAF.
///
/// CORE-012 (#1706): `.to(device)` of a requires-grad leaf is a
/// differentiable copy — the result is a NON-leaf whose backward gradients
/// accumulate on the ORIGINAL CPU leaf (torch: `is_leaf=False`, grad_fn
/// `ToCopyBackward0`). These suites assert CUDA-resident `.grad()` on the
/// uploaded tensor, so they need a real CUDA leaf — torch's
/// `x.to('cuda').detach().requires_grad_(True)` idiom.
fn upload_f32(t: Tensor<f32>, device: Device) -> Tensor<f32> {
    if matches!(device, Device::Cuda(_)) {
        let track = t.requires_grad();
        t.detach()
            .to(device)
            .expect("upload to cuda")
            .requires_grad_(track)
    } else {
        t
    }
}

/// f64 twin of [`upload_f32`] — same CORE-012 leaf-preserving idiom.
fn upload_f64(t: Tensor<f64>, device: Device) -> Tensor<f64> {
    if matches!(device, Device::Cuda(_)) {
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

/// Per-fixture diagnostic skip for cascade issues surfaced by the GPU lane.
/// Returns `Some("issue #")` to skip with a printed reason; returns `None`
/// to run normally. The dispatch's cascade-handling mandate requires
/// surfacing each failure with a tracking issue rather than silently
/// weakening tolerance.
///
/// Active cascades: (none currently — #802 was resolved by materializing
/// stride-views on-device via `strided_copy_{f32,f64}` before D2H in the
/// CUDA→CPU arm of `Tensor::to`.)
fn cascade_skip(_op: &str, _device_label: &str, _dtype: &str) -> Option<&'static str> {
    None
}

fn maybe_skip(op: &str, device_label: &str, dtype: &str, tag: &str) -> bool {
    if let Some(reason) = cascade_skip(op, device_label, dtype) {
        eprintln!("skipping {op} {device_label} dtype={dtype} tag={tag:?}: {reason}");
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Cat A.shape — reshape / view / flatten / squeeze / unsqueeze (view ops)
// ---------------------------------------------------------------------------
//
// All five ops are pure metadata changes — bit-exact equality for forward
// and identity-shaped backward.

fn run_reshape_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "reshape", device_label);
    assert!(!cases.is_empty(), "no reshape fixtures on {device_label}");
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("reshape", device_label, dtype, tag) {
            continue;
        }
        let label = format!("reshape {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().expect("in_shape");
        let new_shape = f.new_shape.as_ref().expect("new_shape");
        let new_shape_isize: Vec<isize> = new_shape.iter().map(|&d| d as isize).collect();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let r = reshape(&a, &new_shape_isize).expect("reshape");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                if r.numel() > 0 {
                    let a_g = upload_f32(make_cpu_f32(in_data, in_shape, true), device);
                    let r_g = reshape(&a_g, &new_shape_isize).expect("reshape");
                    let loss =
                        ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum-to-scalar");
                    loss.backward().expect("backward");
                    let g = a_g.grad().unwrap().expect("grad_a");
                    check_f32(
                        &format!("{label} grad_a"),
                        &read_back_f32(&g),
                        grad_exp,
                        tolerance::F32_BITEXACT,
                    );
                }
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let r = reshape(&a, &new_shape_isize).expect("reshape");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                if r.numel() > 0 {
                    let a_g = upload_f64(make_cpu_f64(in_data, in_shape, true), device);
                    let r_g = reshape(&a_g, &new_shape_isize).expect("reshape");
                    let loss =
                        ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum-to-scalar");
                    loss.backward().expect("backward");
                    let g = a_g.grad().unwrap().expect("grad_a");
                    check_f64(
                        &format!("{label} grad_a"),
                        &read_back_f64(&g),
                        grad_exp,
                        tolerance::F64_BITEXACT,
                    );
                }
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_reshape() {
    run_reshape_for_device("cpu", Device::Cpu);
}

fn run_view_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "view", device_label);
    assert!(!cases.is_empty(), "no view fixtures on {device_label}");
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("view", device_label, dtype, tag) {
            continue;
        }
        let label = format!("view {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().expect("in_shape");
        let new_shape_i64 = f.new_shape.as_ref().expect("new_shape");
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let r = view_t(&a, new_shape_i64).expect("view_t");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                let a_g = upload_f32(make_cpu_f32(in_data, in_shape, true), device);
                let r_g = view_t(&a_g, new_shape_i64).expect("view_t");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&g),
                    grad_exp,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let r = view_t(&a, new_shape_i64).expect("view_t");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                let a_g = upload_f64(make_cpu_f64(in_data, in_shape, true), device);
                let r_g = view_t(&a_g, new_shape_i64).expect("view_t");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&g),
                    grad_exp,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_view() {
    run_view_for_device("cpu", Device::Cpu);
}

fn run_flatten_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "flatten", device_label);
    assert!(!cases.is_empty(), "no flatten fixtures on {device_label}");
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("flatten", device_label, dtype, tag) {
            continue;
        }
        let label = format!("flatten {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let r = flatten(&a).expect("flatten");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                let a_g = upload_f32(make_cpu_f32(in_data, in_shape, true), device);
                let r_g = flatten(&a_g).expect("flatten");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&g),
                    grad_exp,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let r = flatten(&a).expect("flatten");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                let a_g = upload_f64(make_cpu_f64(in_data, in_shape, true), device);
                let r_g = flatten(&a_g).expect("flatten");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&g),
                    grad_exp,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_flatten() {
    run_flatten_for_device("cpu", Device::Cpu);
}

fn run_squeeze_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "squeeze", device_label);
    assert!(!cases.is_empty(), "no squeeze fixtures on {device_label}");
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("squeeze", device_label, dtype, tag) {
            continue;
        }
        let label = format!("squeeze {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let axis = f.axis.expect("axis") as isize;
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let r = squeeze(&a, axis).expect("squeeze");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                let a_g = upload_f32(make_cpu_f32(in_data, in_shape, true), device);
                let r_g = squeeze(&a_g, axis).expect("squeeze");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&g),
                    grad_exp,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let r = squeeze(&a, axis).expect("squeeze");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                let a_g = upload_f64(make_cpu_f64(in_data, in_shape, true), device);
                let r_g = squeeze(&a_g, axis).expect("squeeze");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&g),
                    grad_exp,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_squeeze() {
    run_squeeze_for_device("cpu", Device::Cpu);
}

fn run_unsqueeze_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "unsqueeze", device_label);
    assert!(!cases.is_empty(), "no unsqueeze fixtures on {device_label}");
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("unsqueeze", device_label, dtype, tag) {
            continue;
        }
        let label = format!("unsqueeze {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let axis = f.axis.expect("axis") as isize;
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let r = unsqueeze(&a, axis).expect("unsqueeze");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                let a_g = upload_f32(make_cpu_f32(in_data, in_shape, true), device);
                let r_g = unsqueeze(&a_g, axis).expect("unsqueeze");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&g),
                    grad_exp,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let r = unsqueeze(&a, axis).expect("unsqueeze");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                let a_g = upload_f64(make_cpu_f64(in_data, in_shape, true), device);
                let r_g = unsqueeze(&a_g, axis).expect("unsqueeze");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&g),
                    grad_exp,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_unsqueeze() {
    run_unsqueeze_for_device("cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// transpose / permute — both produce non-contiguous views; the test calls
// .contiguous() on the output before reading back so we compare against
// PyTorch's contiguous reference (the fixture also calls `.contiguous()`
// before recording out_values).
// ---------------------------------------------------------------------------

fn run_transpose_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "transpose", device_label);
    assert!(!cases.is_empty(), "no transpose fixtures on {device_label}");
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("transpose", device_label, dtype, tag) {
            continue;
        }
        let label = format!("transpose {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let d0 = f.dim0.expect("dim0");
        let d1 = f.dim1.expect("dim1");
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                // For 2-D, exercise both `transpose_2d` (the canonical
                // grad_fns::shape entrypoint) and `Tensor::transpose(d0, d1)`
                // on this case so the test covers the inherent method path.
                let r = if in_shape.len() == 2 {
                    transpose_2d(&a).expect("transpose_2d")
                } else {
                    a.transpose(d0, d1).expect("transpose method")
                };
                let r_c = contiguous_t(&r).expect("contig");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r_c),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                let a_g = upload_f32(make_cpu_f32(in_data, in_shape, true), device);
                let r_g = if in_shape.len() == 2 {
                    transpose_2d(&a_g).expect("transpose_2d")
                } else {
                    a_g.transpose(d0, d1).expect("transpose method")
                };
                let r_g_c = contiguous_t(&r_g).expect("contig");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g_c).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&g),
                    grad_exp,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let r = if in_shape.len() == 2 {
                    transpose_2d(&a).expect("transpose_2d")
                } else {
                    a.transpose(d0, d1).expect("transpose method")
                };
                let r_c = contiguous_t(&r).expect("contig");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r_c),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                let a_g = upload_f64(make_cpu_f64(in_data, in_shape, true), device);
                let r_g = if in_shape.len() == 2 {
                    transpose_2d(&a_g).expect("transpose_2d")
                } else {
                    a_g.transpose(d0, d1).expect("transpose method")
                };
                let r_g_c = contiguous_t(&r_g).expect("contig");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g_c).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&g),
                    grad_exp,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_transpose() {
    run_transpose_for_device("cpu", Device::Cpu);
}

fn run_permute_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "permute", device_label);
    assert!(!cases.is_empty(), "no permute fixtures on {device_label}");
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("permute", device_label, dtype, tag) {
            continue;
        }
        let label = format!("permute {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let dims = f.dims.as_ref().expect("dims");
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let r = permute_t(&a, dims).expect("permute_t");
                let r_c = contiguous_t(&r).expect("contig");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r_c),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                let a_g = upload_f32(make_cpu_f32(in_data, in_shape, true), device);
                let r_g = permute_t(&a_g, dims).expect("permute_t");
                let r_g_c = contiguous_t(&r_g).expect("contig");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g_c).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&g),
                    grad_exp,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let r = permute_t(&a, dims).expect("permute_t");
                let r_c = contiguous_t(&r).expect("contig");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r_c),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                let a_g = upload_f64(make_cpu_f64(in_data, in_shape, true), device);
                let r_g = permute_t(&a_g, dims).expect("permute_t");
                let r_g_c = contiguous_t(&r_g).expect("contig");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g_c).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&g),
                    grad_exp,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_permute() {
    run_permute_for_device("cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// narrow — exercises start=0, length=full, partial, and 2-D outer/inner
// dim variants. Output is materialized via `.contiguous()` for the
// fixture comparison since narrow's view stride doesn't match a fresh
// contiguous buffer.
// ---------------------------------------------------------------------------

fn run_narrow_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "narrow", device_label);
    assert!(!cases.is_empty(), "no narrow fixtures on {device_label}");
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("narrow", device_label, dtype, tag) {
            continue;
        }
        let label = format!("narrow {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let dim = f.dim.expect("dim") as usize;
        let start = f.start.expect("start");
        let length = f.length.expect("length");
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let r = narrow_t(&a, dim, start, length).expect("narrow_t");
                let r_c = contiguous_t(&r).expect("contig");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r_c),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                let a_g = upload_f32(make_cpu_f32(in_data, in_shape, true), device);
                let r_g = narrow_t(&a_g, dim, start, length).expect("narrow_t");
                let r_g_c = contiguous_t(&r_g).expect("contig");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g_c).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&g),
                    grad_exp,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let r = narrow_t(&a, dim, start, length).expect("narrow_t");
                let r_c = contiguous_t(&r).expect("contig");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r_c),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                let a_g = upload_f64(make_cpu_f64(in_data, in_shape, true), device);
                let r_g = narrow_t(&a_g, dim, start, length).expect("narrow_t");
                let r_g_c = contiguous_t(&r_g).expect("contig");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g_c).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad_a");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&g),
                    grad_exp,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_narrow() {
    run_narrow_for_device("cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// contiguous — special focus on the non-contiguous-input -> contiguous-output
// pattern (the historical view_reshape silent-CPU-demote regression case).
// The test asserts the result lands on the same device as the input — that
// regresses the bug.
// ---------------------------------------------------------------------------

fn run_contiguous_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "contiguous", device_label);
    assert!(
        !cases.is_empty(),
        "no contiguous fixtures on {device_label}"
    );
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("contiguous", device_label, dtype, tag) {
            continue;
        }
        let label = format!("contiguous {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let view = if tag == "transpose_then_contiguous" {
                    transpose_2d(&a).expect("transpose")
                } else {
                    a.clone()
                };
                let r = contiguous_t(&view).expect("contig");
                assert!(
                    r.is_contiguous(),
                    "{label}: contiguous() result must be contiguous"
                );
                // PRIMARY REGRESSION ASSERTION: device is preserved. The
                // historical silent-CPU-demote bug landed the output on CPU
                // even when the input was on CUDA; we assert the output's
                // device matches the input's, which is enough to catch any
                // re-introduction of that pattern.
                assert_eq!(
                    r.device(),
                    a.device(),
                    "{label}: contiguous() must preserve device (regression: silent CPU demote)"
                );
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                if tag == "transpose_then_contiguous" {
                    let a_g = upload_f32(make_cpu_f32(in_data, in_shape, true), device);
                    let view_g = transpose_2d(&a_g).expect("transpose");
                    let r_g = contiguous_t(&view_g).expect("contig");
                    let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                    loss.backward().expect("backward");
                    let g = a_g.grad().unwrap().expect("grad_a");
                    let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();
                    check_f32(
                        &format!("{label} grad_a"),
                        &read_back_f32(&g),
                        grad_exp,
                        tolerance::F32_BITEXACT,
                    );
                }
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let view = if tag == "transpose_then_contiguous" {
                    transpose_2d(&a).expect("transpose")
                } else {
                    a.clone()
                };
                let r = contiguous_t(&view).expect("contig");
                assert!(r.is_contiguous(), "{label}: must be contiguous");
                assert_eq!(
                    r.device(),
                    a.device(),
                    "{label}: contiguous() must preserve device"
                );
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_contiguous() {
    run_contiguous_for_device("cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// cat / split / chunk — multi-tensor ops. Cat exercises single-tensor,
// many-tensor, negative-axis. Split / chunk exercise even and uneven
// divisions.
// ---------------------------------------------------------------------------

fn run_cat_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "cat", device_label);
    assert!(!cases.is_empty(), "no cat fixtures on {device_label}");
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("cat", device_label, dtype, tag) {
            continue;
        }
        let label = format!("cat {device_label} tag={tag} dtype={dtype}");
        let shapes = f.tensor_shapes.as_ref().expect("tensor_shapes");
        let datas = f.tensor_data.as_ref().expect("tensor_data");
        let axis = f.axis.expect("axis");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grads_exp = f.tensor_grads.as_ref().expect("tensor_grads");

        match dtype {
            "float32" => {
                let tensors: Vec<Tensor<f32>> = shapes
                    .iter()
                    .zip(datas.iter())
                    .map(|(s, d)| upload_f32(make_cpu_f32(d.as_slice(), s, false), device))
                    .collect();
                let r = cat(&tensors, axis as isize).expect("cat");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                // Autograd
                let tensors_g: Vec<Tensor<f32>> = shapes
                    .iter()
                    .zip(datas.iter())
                    .map(|(s, d)| upload_f32(make_cpu_f32(d.as_slice(), s, true), device))
                    .collect();
                let r_g = cat(&tensors_g, axis as isize).expect("cat");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                for (i, (t, ge)) in tensors_g.iter().zip(grads_exp.iter()).enumerate() {
                    let g = t.grad().unwrap().expect("grad");
                    check_f32(
                        &format!("{label} grad[{i}]"),
                        &read_back_f32(&g),
                        ge.as_slice(),
                        tolerance::F32_BITEXACT,
                    );
                }
            }
            "float64" => {
                let tensors: Vec<Tensor<f64>> = shapes
                    .iter()
                    .zip(datas.iter())
                    .map(|(s, d)| upload_f64(make_cpu_f64(d.as_slice(), s, false), device))
                    .collect();
                let r = cat(&tensors, axis as isize).expect("cat");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                let tensors_g: Vec<Tensor<f64>> = shapes
                    .iter()
                    .zip(datas.iter())
                    .map(|(s, d)| upload_f64(make_cpu_f64(d.as_slice(), s, true), device))
                    .collect();
                let r_g = cat(&tensors_g, axis as isize).expect("cat");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                for (i, (t, ge)) in tensors_g.iter().zip(grads_exp.iter()).enumerate() {
                    let g = t.grad().unwrap().expect("grad");
                    check_f64(
                        &format!("{label} grad[{i}]"),
                        &read_back_f64(&g),
                        ge.as_slice(),
                        tolerance::F64_BITEXACT,
                    );
                }
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_cat() {
    run_cat_for_device("cpu", Device::Cpu);
}

#[test]
fn cpu_cat_empty_list_errors() {
    // Edge case from the dispatch: cat([]) must return Err.
    let empty: &[Tensor<f32>] = &[];
    let r = cat(empty, 0);
    assert!(
        r.is_err(),
        "cat on empty list must error (PyTorch parity), got Ok(_)"
    );
}

fn run_split_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "split", device_label);
    assert!(!cases.is_empty(), "no split fixtures on {device_label}");
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("split", device_label, dtype, tag) {
            continue;
        }
        let label = format!("split {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let split_sizes = f.split_sizes.as_ref().expect("split_sizes");
        let dim = f.dim.expect("dim") as usize;
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let chunks_shapes = f.chunks_shapes.as_ref().expect("chunks_shapes");
        let chunks_values = f.chunks_values.as_ref().expect("chunks_values");
        let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let chunks = split_t(&a, split_sizes, dim).expect("split_t");
                assert_eq!(chunks.len(), chunks_values.len(), "{label}: chunk count");
                for (i, (chunk, exp)) in chunks.iter().zip(chunks_values.iter()).enumerate() {
                    assert_eq!(chunk.shape(), chunks_shapes[i].as_slice());
                    let chunk_c = contiguous_t(chunk).expect("contig");
                    check_f32(
                        &format!("{label} chunk[{i}]"),
                        &read_back_f32(&chunk_c),
                        exp.as_slice(),
                        tolerance::F32_BITEXACT,
                    );
                }
                // Backward — sum every chunk and accumulate
                let a_g = upload_f32(make_cpu_f32(in_data, in_shape, true), device);
                let chunks_g = split_t(&a_g, split_sizes, dim).expect("split_t");
                // build loss = sum(chunk0) + sum(chunk1) + ... by add_t
                let mut acc: Option<Tensor<f32>> = None;
                for c in &chunks_g {
                    let s = ferrotorch_core::grad_fns::reduction::sum(c).expect("sum");
                    acc = match acc {
                        None => Some(s),
                        Some(prev) => Some(prev.add_t(&s).expect("add")),
                    };
                }
                acc.unwrap().backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&g),
                    grad_exp,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let chunks = split_t(&a, split_sizes, dim).expect("split_t");
                for (i, (chunk, exp)) in chunks.iter().zip(chunks_values.iter()).enumerate() {
                    let chunk_c = contiguous_t(chunk).expect("contig");
                    check_f64(
                        &format!("{label} chunk[{i}]"),
                        &read_back_f64(&chunk_c),
                        exp.as_slice(),
                        tolerance::F64_BITEXACT,
                    );
                }
                let a_g = upload_f64(make_cpu_f64(in_data, in_shape, true), device);
                let chunks_g = split_t(&a_g, split_sizes, dim).expect("split_t");
                let mut acc: Option<Tensor<f64>> = None;
                for c in &chunks_g {
                    let s = ferrotorch_core::grad_fns::reduction::sum(c).expect("sum");
                    acc = match acc {
                        None => Some(s),
                        Some(prev) => Some(prev.add_t(&s).expect("add")),
                    };
                }
                acc.unwrap().backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&g),
                    grad_exp,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_split() {
    run_split_for_device("cpu", Device::Cpu);
}

fn run_chunk_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "chunk", device_label);
    assert!(!cases.is_empty(), "no chunk fixtures on {device_label}");
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("chunk", device_label, dtype, tag) {
            continue;
        }
        let label = format!("chunk {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let chunks = f.chunks.expect("chunks");
        let dim = f.dim.expect("dim") as usize;
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let chunks_values = f.chunks_values.as_ref().expect("chunks_values");

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let parts = chunk_t(&a, chunks, dim).expect("chunk_t");
                for (i, (part, exp)) in parts.iter().zip(chunks_values.iter()).enumerate() {
                    let part_c = contiguous_t(part).expect("contig");
                    check_f32(
                        &format!("{label} part[{i}]"),
                        &read_back_f32(&part_c),
                        exp.as_slice(),
                        tolerance::F32_BITEXACT,
                    );
                }
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let parts = chunk_t(&a, chunks, dim).expect("chunk_t");
                for (i, (part, exp)) in parts.iter().zip(chunks_values.iter()).enumerate() {
                    let part_c = contiguous_t(part).expect("contig");
                    check_f64(
                        &format!("{label} part[{i}]"),
                        &read_back_f64(&part_c),
                        exp.as_slice(),
                        tolerance::F64_BITEXACT,
                    );
                }
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_chunk() {
    run_chunk_for_device("cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// expand — broadcast-aware. The forward output is bit-exact (every position
// reads from the size-1 source); the backward is a fan-in sum, so the grad
// can have a tiny rounding margin on GPU but is bit-exact on CPU.
// ---------------------------------------------------------------------------

fn run_expand_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "expand", device_label);
    assert!(!cases.is_empty(), "no expand fixtures on {device_label}");
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("expand", device_label, dtype, tag) {
            continue;
        }
        let label = format!("expand {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let new_shape_i64 = f.new_shape.as_ref().expect("new_shape");
        let new_shape: Vec<usize> = new_shape_i64.iter().map(|&d| d as usize).collect();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let r = expand(&a, &new_shape).expect("expand");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                // Grad: fan-in sum -> bit-exact on CPU. GPU lane uses
                // F32_REDUCTION_GPU.
                let a_g = upload_f32(make_cpu_f32(in_data, in_shape, true), device);
                let r_g = expand(&a_g, &new_shape).expect("expand");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad");
                let tol = if matches!(device, Device::Cuda(_)) {
                    tolerance::F32_REDUCTION_GPU
                } else {
                    tolerance::F32_BITEXACT
                };
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&g),
                    grad_exp,
                    tol,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let r = expand(&a, &new_shape).expect("expand");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                let a_g = upload_f64(make_cpu_f64(in_data, in_shape, true), device);
                let r_g = expand(&a_g, &new_shape).expect("expand");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&g),
                    grad_exp,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_expand() {
    run_expand_for_device("cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// as_strided / as_strided_copy / as_strided_scatter
// ---------------------------------------------------------------------------

fn run_as_strided_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "as_strided", device_label);
    assert!(
        !cases.is_empty(),
        "no as_strided fixtures on {device_label}"
    );
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("as_strided", device_label, dtype, tag) {
            continue;
        }
        let label = format!("as_strided {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let size = f.size.as_ref().expect("size");
        let stride: Vec<isize> = f
            .stride
            .as_ref()
            .expect("stride")
            .iter()
            .map(|&s| s as isize)
            .collect();
        let storage_offset = f.storage_offset.expect("storage_offset");
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let v = as_strided(&a, size, &stride, Some(storage_offset)).expect("as_strided");
                let v_c = contiguous_t(&v).expect("contig");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&v_c),
                    expected,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let v = as_strided(&a, size, &stride, Some(storage_offset)).expect("as_strided");
                let v_c = contiguous_t(&v).expect("contig");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&v_c),
                    expected,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_as_strided() {
    run_as_strided_for_device("cpu", Device::Cpu);
}

fn run_as_strided_copy_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "as_strided_copy", device_label);
    assert!(
        !cases.is_empty(),
        "no as_strided_copy fixtures on {device_label}"
    );
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("as_strided_copy", device_label, dtype, tag) {
            continue;
        }
        let label = format!("as_strided_copy {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let size = f.size.as_ref().expect("size");
        let stride: Vec<isize> = f
            .stride
            .as_ref()
            .expect("stride")
            .iter()
            .map(|&s| s as isize)
            .collect();
        let storage_offset = f.storage_offset.expect("storage_offset");
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let r = as_strided_copy(&a, size, &stride, Some(storage_offset))
                    .expect("as_strided_copy");
                assert!(r.is_contiguous(), "{label}: must be contiguous");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let r = as_strided_copy(&a, size, &stride, Some(storage_offset))
                    .expect("as_strided_copy");
                assert!(r.is_contiguous());
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_as_strided_copy() {
    run_as_strided_copy_for_device("cpu", Device::Cpu);
}

fn run_as_strided_scatter_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "as_strided_scatter", device_label);
    assert!(
        !cases.is_empty(),
        "no as_strided_scatter fixtures on {device_label}"
    );
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("as_strided_scatter", device_label, dtype, tag) {
            continue;
        }
        let label = format!("as_strided_scatter {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let src_shape = f.src_shape.as_ref().expect("src_shape");
        let size = f.size.as_ref().expect("size");
        let stride: Vec<isize> = f
            .stride
            .as_ref()
            .expect("stride")
            .iter()
            .map(|&s| s as isize)
            .collect();
        let storage_offset = f.storage_offset.expect("storage_offset");
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let src_data = f.src_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let s = upload_f32(make_cpu_f32(src_data, src_shape, false), device);
                let r = as_strided_scatter(&a, &s, size, &stride, Some(storage_offset))
                    .expect("as_strided_scatter");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let s = upload_f64(make_cpu_f64(src_data, src_shape, false), device);
                let r = as_strided_scatter(&a, &s, size, &stride, Some(storage_offset))
                    .expect("as_strided_scatter");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_as_strided_scatter() {
    run_as_strided_scatter_for_device("cpu", Device::Cpu);
}

// ---------------------------------------------------------------------------
// Cat A.indexing — gather / scatter / scatter_add / where_cond / index_select
// / masked_fill. The first four are CPU-only forwards; the test asserts the
// GPU lane returns Err. The latter two have GPU f32 paths and do exercise
// CUDA in the gpu module.
// ---------------------------------------------------------------------------

#[test]
fn cpu_gather() {
    let file = load_fixtures();
    let cases = cases_for(&file, "gather", "cpu");
    assert!(!cases.is_empty(), "no gather fixtures");
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("gather cpu tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let index = f.index.as_ref().expect("index");
        let index_shape = f.index_shape.as_ref().expect("index_shape");
        let dim = f.dim.expect("dim");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let a = make_cpu_f32(in_data, in_shape, false);
                let r = gather(&a, dim as isize, index, index_shape).expect("gather");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                let a_g = make_cpu_f32(in_data, in_shape, true);
                let r_g = gather(&a_g, dim as isize, index, index_shape).expect("gather");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&g),
                    grad_exp,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = make_cpu_f64(in_data, in_shape, false);
                let r = gather(&a, dim as isize, index, index_shape).expect("gather");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                let a_g = make_cpu_f64(in_data, in_shape, true);
                let r_g = gather(&a_g, dim as isize, index, index_shape).expect("gather");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&g),
                    grad_exp,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_scatter() {
    let file = load_fixtures();
    let cases = cases_for(&file, "scatter", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("scatter cpu tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let src_shape = f.src_shape.as_ref().expect("src_shape");
        let src_data = f.src_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let index = f.index.as_ref().expect("index");
        let index_shape = f.index_shape.as_ref().expect("index_shape");
        let dim = f.dim.expect("dim");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();

        match dtype {
            "float32" => {
                let inp = make_cpu_f32(in_data, in_shape, false);
                let src = make_cpu_f32(src_data, src_shape, false);
                let r = scatter(&inp, dim as isize, index, index_shape, &src).expect("scatter");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let inp = make_cpu_f64(in_data, in_shape, false);
                let src = make_cpu_f64(src_data, src_shape, false);
                let r = scatter(&inp, dim as isize, index, index_shape, &src).expect("scatter");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_scatter_value() {
    let inp = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[4], true);
    let r = scatter_value(&inp, 0, &[0, 2], &[2], 9.0).expect("scatter_value");
    check_f32(
        "scatter_value cpu fwd",
        &read_back_f32(&r),
        &[9.0, 2.0, 9.0, 4.0],
        tolerance::F32_BITEXACT,
    );

    let loss = ferrotorch_core::grad_fns::reduction::sum(&r).expect("sum");
    loss.backward().expect("backward");
    let grad = inp.grad().unwrap().expect("grad");
    check_f32(
        "scatter_value cpu grad_a",
        &read_back_f32(&grad),
        &[0.0, 1.0, 0.0, 1.0],
        tolerance::F32_BITEXACT,
    );

    let inp = make_cpu_f64(&[5.0], &[], true);
    let r = scatter_value(&inp, -1, &[0], &[], 7.0_f64).expect("scatter_value scalar self");
    assert_eq!(r.shape(), &[] as &[usize]);
    check_f64(
        "scatter_value scalar cpu fwd",
        &read_back_f64(&r),
        &[7.0],
        tolerance::F64_BITEXACT,
    );
    r.backward().expect("scalar backward");
    let grad = inp.grad().unwrap().expect("scalar grad");
    check_f64(
        "scatter_value scalar cpu grad_a",
        &read_back_f64(&grad),
        &[0.0],
        tolerance::F64_BITEXACT,
    );
}

#[test]
fn cpu_scatter_add() {
    let file = load_fixtures();
    let cases = cases_for(&file, "scatter_add", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("scatter_add cpu tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let src_shape = f.src_shape.as_ref().expect("src_shape");
        let src_data = f.src_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let index = f.index.as_ref().expect("index");
        let index_shape = f.index_shape.as_ref().expect("index_shape");
        let dim = f.dim.expect("dim");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();

        match dtype {
            "float32" => {
                let inp = make_cpu_f32(in_data, in_shape, false);
                let src = make_cpu_f32(src_data, src_shape, false);
                let r =
                    scatter_add(&inp, dim as isize, index, index_shape, &src).expect("scatter_add");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let inp = make_cpu_f64(in_data, in_shape, false);
                let src = make_cpu_f64(src_data, src_shape, false);
                let r =
                    scatter_add(&inp, dim as isize, index, index_shape, &src).expect("scatter_add");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_where_cond() {
    let file = load_fixtures();
    let cases = cases_for(&file, "where_cond", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("where_cond cpu tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let cond = f.condition.as_ref().expect("condition");
        let x_data = f.x_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let y_data = f.y_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_x = f.grad_x.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let grad_y = f.grad_y.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let x = make_cpu_f32(x_data, in_shape, false);
                let y = make_cpu_f32(y_data, in_shape, false);
                let r = where_cond(cond, &x, &y).expect("where_cond");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                let x_g = make_cpu_f32(x_data, in_shape, true);
                let y_g = make_cpu_f32(y_data, in_shape, true);
                let r_g = where_cond(cond, &x_g, &y_g).expect("where_cond");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let gx = x_g.grad().unwrap().expect("grad_x");
                let gy = y_g.grad().unwrap().expect("grad_y");
                check_f32(
                    &format!("{label} grad_x"),
                    &read_back_f32(&gx),
                    grad_x,
                    tolerance::F32_BITEXACT,
                );
                check_f32(
                    &format!("{label} grad_y"),
                    &read_back_f32(&gy),
                    grad_y,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let x = make_cpu_f64(x_data, in_shape, false);
                let y = make_cpu_f64(y_data, in_shape, false);
                let r = where_cond(cond, &x, &y).expect("where_cond");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                let x_g = make_cpu_f64(x_data, in_shape, true);
                let y_g = make_cpu_f64(y_data, in_shape, true);
                let r_g = where_cond(cond, &x_g, &y_g).expect("where_cond");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let gx = x_g.grad().unwrap().expect("grad_x");
                let gy = y_g.grad().unwrap().expect("grad_y");
                check_f64(
                    &format!("{label} grad_x"),
                    &read_back_f64(&gx),
                    grad_x,
                    tolerance::F64_BITEXACT,
                );
                check_f64(
                    &format!("{label} grad_y"),
                    &read_back_f64(&gy),
                    grad_y,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

fn run_index_select_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "index_select_1d", device_label);
    assert!(
        !cases.is_empty(),
        "no index_select_1d fixtures on {device_label}"
    );
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("index_select_1d", device_label, dtype, tag) {
            continue;
        }
        let label = format!("index_select_1d {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let index = f.index.as_ref().expect("index");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let r = index_select_1d(&a, index).expect("index_select_1d");
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                let a_g = upload_f32(make_cpu_f32(in_data, in_shape, true), device);
                let r_g = index_select_1d(&a_g, index).expect("index_select_1d");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad");
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&g),
                    grad_exp,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                // f64 GPU path is unsupported for index_select_1d in the
                // current backend; the fixture skips it. CPU only.
                let a = make_cpu_f64(in_data, in_shape, false);
                let r = index_select_1d(&a, index).expect("index_select_1d");
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                let a_g = make_cpu_f64(in_data, in_shape, true);
                let r_g = index_select_1d(&a_g, index).expect("index_select_1d");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad");
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&g),
                    grad_exp,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_index_select_1d() {
    run_index_select_for_device("cpu", Device::Cpu);
}

#[test]
fn cpu_index_select_1d_via_int_tensor() {
    // Exercise the IntTensor wrapper path (`index_select_1d_it`) which
    // covers the canonical surface entry. Bit-exact against a manual
    // reference.
    let a = make_cpu_f32(&[10.0, 20.0, 30.0, 40.0], &[4], false);
    let idx = IntTensor::from_vec(vec![3i32, 0, 2], vec![3]).expect("IntTensor");
    let r = index_select_1d_it(&a, &idx).expect("index_select_1d_it");
    let got = read_back_f32(&r);
    assert_eq!(got, vec![40.0, 10.0, 30.0]);
}

#[test]
fn cpu_phase2c_tensor_index_select_scalar() {
    let input = make_cpu_f32(&[5.0], &[], true);
    let scalar_index = IntTensor::<i64>::from_vec(vec![0], vec![]).expect("scalar index");
    let one_index = IntTensor::<i64>::from_vec(vec![0], vec![1]).expect("one index");

    let scalar = input
        .index_select(0, &scalar_index)
        .expect("scalar index_select");
    let one = input
        .index_select(-1, &one_index)
        .expect("one-element index_select");

    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(one.shape(), &[] as &[usize]);
    check_f32(
        "phase2c tensor index_select scalar fwd",
        &read_back_f32(&one),
        &[5.0],
        tolerance::F32_BITEXACT,
    );

    one.backward().expect("scalar index_select backward");
    let grad = input.grad().unwrap().expect("scalar index_select grad");
    assert_eq!(grad.shape(), &[] as &[usize]);
    check_f32(
        "phase2c tensor index_select scalar grad",
        &read_back_f32(&grad),
        &[1.0],
        tolerance::F32_BITEXACT,
    );

    let empty = IntTensor::<i64>::from_vec(Vec::new(), vec![0]).expect("empty index");
    let two = IntTensor::<i64>::from_vec(vec![0, 0], vec![2]).expect("two index");
    assert!(input.index_select(0, &empty).is_err());
    assert!(input.index_select(0, &two).is_err());
}

#[test]
fn cpu_phase2c_inttensor_index_select_scalar() {
    let input = IntTensor::<i64>::from_vec(vec![5], vec![]).expect("scalar int tensor");
    let scalar_index = IntTensor::<i64>::from_vec(vec![0], vec![]).expect("scalar index");
    let one_index = IntTensor::<i64>::from_vec(vec![0], vec![1]).expect("one index");

    let scalar = input
        .index_select(0, &scalar_index)
        .expect("scalar int index_select");
    let one = input
        .index_select(-1, &one_index)
        .expect("one-element int index_select");

    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(one.shape(), &[] as &[usize]);
    assert_eq!(one.data().unwrap(), &[5]);

    let empty = IntTensor::<i64>::from_vec(Vec::new(), vec![0]).expect("empty index");
    let two = IntTensor::<i64>::from_vec(vec![0, 0], vec![2]).expect("two index");
    assert!(input.index_select(0, &empty).is_err());
    assert!(input.index_select(0, &two).is_err());
}

#[test]
fn cpu_phase2c_tensor_gather_public_api() {
    let input = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
        false,
    )
    .expect("input")
    .requires_grad_(true);
    let index = IntTensor::<i32>::from_vec(vec![2, 0, 0, 2], vec![2, 2]).expect("i32 index");

    let out = input
        .gather(-1, &index)
        .expect("Tensor<T>::gather public API");

    assert_eq!(out.shape(), &[2, 2]);
    check_f32(
        "phase2c tensor gather fwd",
        &read_back_f32(&out),
        &[3.0, 1.0, 4.0, 6.0],
        tolerance::F32_BITEXACT,
    );

    out.sum_all().expect("sum").backward().expect("backward");
    let grad = input.grad().unwrap().expect("gather grad");
    assert_eq!(grad.shape(), &[2, 3]);
    check_f32(
        "phase2c tensor gather grad",
        &read_back_f32(&grad),
        &[1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        tolerance::F32_BITEXACT,
    );
}

#[test]
fn cpu_phase2c_inttensor_gather_public_api() {
    let input = IntTensor::<i64>::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).expect("int input");
    let index = IntTensor::<i32>::from_vec(vec![1, 0], vec![1, 2]).expect("i32 index");

    let out = input
        .gather(1, &index)
        .expect("IntTensor<I>::gather public API");

    assert_eq!(out.shape(), &[1, 2]);
    assert_eq!(read_back_int(&out), vec![2, 1]);
}

#[test]
fn cpu_phase2c_argmax_argmin_public_api() {
    let floats = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0_f32, f32::NAN, 3.0, 4.0, f32::NAN, 2.0]),
        vec![2, 3],
        false,
    )
    .expect("float input");

    assert_eq!(read_back_int(&floats.argmax(None).unwrap()), vec![1]);
    assert_eq!(read_back_int(&floats.argmin(None).unwrap()), vec![1]);
    assert_eq!(read_back_int(&floats.argmax(Some(1)).unwrap()), vec![1, 1]);
    assert_eq!(read_back_int(&floats.argmin(Some(1)).unwrap()), vec![1, 1]);

    let ints = IntTensor::<i32>::from_vec(vec![7, 7, -3, 9, 9, 1], vec![2, 3]).expect("int input");
    assert_eq!(read_back_int(&ints.argmax(None).unwrap()), vec![3]);
    assert_eq!(read_back_int(&ints.argmin(None).unwrap()), vec![2]);
    assert_eq!(read_back_int(&ints.argmax(Some(-1)).unwrap()), vec![0, 0]);
    assert_eq!(read_back_int(&ints.argmin(Some(-1)).unwrap()), vec![2, 2]);
}

#[test]
fn cpu_phase2c_cast_public_api_matches_torch() {
    let int_values = vec![
        0_i64,
        1,
        -1,
        65_504,
        65_505,
        100_000,
        16_777_217,
        i64::from(i32::MAX),
        i64::from(i32::MIN),
        i64::MAX,
        i64::MIN,
    ];
    let ints = IntTensor::<i64>::from_vec(int_values, vec![11]).expect("int input");

    let f16 = ints.to_float::<half::f16>().expect("i64 to f16");
    assert_eq!(
        read_back_f16_bits(&f16),
        vec![
            0x0000, 0x3c00, 0xbc00, 0x7bff, 0x7bff, 0x7c00, 0x7c00, 0x7c00, 0xfc00, 0x7c00, 0xfc00,
        ]
    );

    let bf16 = ints.to_float::<half::bf16>().expect("i64 to bf16");
    assert_eq!(
        read_back_bf16_bits(&bf16),
        vec![
            0x0000, 0x3f80, 0xbf80, 0x4780, 0x4780, 0x47c3, 0x4b80, 0x4f00, 0xcf00, 0x5f00, 0xdf00,
        ]
    );

    let f32_bits: Vec<u32> = read_back_f32(&ints.to_float::<f32>().expect("i64 to f32"))
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(
        f32_bits,
        vec![
            0x00000000, 0x3f800000, 0xbf800000, 0x477fe000, 0x477fe100, 0x47c35000, 0x4b800000,
            0x4f000000, 0xcf000000, 0x5f000000, 0xdf000000,
        ]
    );

    let f64_bits: Vec<u64> = read_back_f64(&ints.to_float::<f64>().expect("i64 to f64"))
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(
        f64_bits,
        vec![
            0x0000000000000000,
            0x3ff0000000000000,
            0xbff0000000000000,
            0x40effc0000000000,
            0x40effc2000000000,
            0x40f86a0000000000,
            0x4170000010000000,
            0x41dfffffffc00000,
            0xc1e0000000000000,
            0x43e0000000000000,
            0xc3e0000000000000,
        ]
    );

    let floats = Tensor::from_storage(
        TensorStorage::cpu(vec![
            0.0_f32,
            1.9,
            -1.9,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            2_147_483_647.0,
            2_147_483_648.0,
            -2_147_483_649.0,
        ]),
        vec![9],
        false,
    )
    .expect("float input");
    let cpu_i32: IntTensor<i32> = Tensor::<f32>::to_int(&floats).expect("f32 to i32");
    assert_eq!(
        read_back_int(&cpu_i32),
        vec![
            0,
            1,
            -1,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN,
            i32::MIN
        ]
    );
    assert_eq!(
        read_back_int(&floats.to_int::<i64>().expect("f32 to i64")),
        vec![
            0,
            1,
            -1,
            i64::MIN,
            i64::MIN,
            i64::MIN,
            2_147_483_648,
            2_147_483_648,
            -2_147_483_648,
        ]
    );
}

#[test]
fn cpu_index_select_dim_scalar() {
    let input = make_cpu_f32(&[5.0], &[], true);
    let index = IntTensor::<i64>::from_vec(vec![0], vec![1]).expect("one index");

    let out = index_select_dim(&input, 0, &index).expect("scalar index_select_dim");
    assert_eq!(out.shape(), &[] as &[usize]);
    check_f32(
        "index_select_dim scalar fwd",
        &read_back_f32(&out),
        &[5.0],
        tolerance::F32_BITEXACT,
    );

    out.backward().expect("index_select_dim scalar backward");
    let grad = input.grad().unwrap().expect("index_select_dim scalar grad");
    assert_eq!(grad.shape(), &[] as &[usize]);
    check_f32(
        "index_select_dim scalar grad",
        &read_back_f32(&grad),
        &[1.0],
        tolerance::F32_BITEXACT,
    );
}

fn run_masked_fill_for_device(device_label: &str, device: Device) {
    let file = load_fixtures();
    let cases = cases_for(&file, "masked_fill", device_label);
    assert!(
        !cases.is_empty(),
        "no masked_fill fixtures on {device_label}"
    );
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        if maybe_skip("masked_fill", device_label, dtype, tag) {
            continue;
        }
        let label = format!("masked_fill {device_label} tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let mask = f.mask.as_ref().expect("mask");
        let value = f.value.expect("value");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let grad_exp = f.grad_a.as_ref().map(F64ListSentinel::as_slice).unwrap();

        match dtype {
            "float32" => {
                let a = upload_f32(make_cpu_f32(in_data, in_shape, false), device);
                let r = masked_fill(&a, mask, value as f32).expect("masked_fill");
                if matches!(device, Device::Cuda(_)) {
                    assert!(
                        r.is_cuda(),
                        "{label} forward must stay CUDA-resident, got {:?}",
                        r.device()
                    );
                }
                check_f32(
                    &format!("{label} fwd"),
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
                let a_g = upload_f32(make_cpu_f32(in_data, in_shape, true), device);
                let r_g = masked_fill(&a_g, mask, value as f32).expect("masked_fill");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad");
                if matches!(device, Device::Cuda(_)) {
                    assert!(
                        g.is_cuda(),
                        "{label} backward gradient must stay CUDA-resident, got {:?}",
                        g.device()
                    );
                }
                check_f32(
                    &format!("{label} grad_a"),
                    &read_back_f32(&g),
                    grad_exp,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = upload_f64(make_cpu_f64(in_data, in_shape, false), device);
                let r = masked_fill(&a, mask, value).expect("masked_fill");
                if matches!(device, Device::Cuda(_)) {
                    assert!(
                        r.is_cuda(),
                        "{label} forward must stay CUDA-resident, got {:?}",
                        r.device()
                    );
                }
                check_f64(
                    &format!("{label} fwd"),
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
                let a_g = upload_f64(make_cpu_f64(in_data, in_shape, true), device);
                let r_g = masked_fill(&a_g, mask, value).expect("masked_fill");
                let loss = ferrotorch_core::grad_fns::reduction::sum(&r_g).expect("sum");
                loss.backward().expect("backward");
                let g = a_g.grad().unwrap().expect("grad");
                if matches!(device, Device::Cuda(_)) {
                    assert!(
                        g.is_cuda(),
                        "{label} backward gradient must stay CUDA-resident, got {:?}",
                        g.device()
                    );
                }
                check_f64(
                    &format!("{label} grad_a"),
                    &read_back_f64(&g),
                    grad_exp,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_masked_fill() {
    run_masked_fill_for_device("cpu", Device::Cpu);
}

#[test]
fn cpu_masked_fill_via_bool_tensor() {
    // Exercise the BoolTensor wrapper path.
    let a = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[4], false);
    let mask = BoolTensor::from_vec(vec![true, false, true, false], vec![4]).expect("BoolTensor");
    let r = masked_fill_bt(&a, &mask, -7.0_f32).expect("masked_fill_bt");
    assert_eq!(read_back_f32(&r), vec![-7.0, 2.0, -7.0, 4.0]);
}

#[test]
fn cpu_masked_select_public_api_broadcast_and_backward() {
    // PyTorch reference:
    //   x = torch.tensor([[10., 20.]], requires_grad=True)
    //   mask = torch.tensor([[True, True], [False, True]])
    //   torch.masked_select(x, mask) -> tensor([10., 20., 20.])
    //   out.sum().backward(); x.grad -> tensor([[1., 2.]])
    let input = make_cpu_f32(&[10.0, 20.0], &[1, 2], true);
    let mask =
        BoolTensor::from_vec(vec![true, true, false, true], vec![2, 2]).expect("broadcast mask");

    let out = masked_select_bcast(&input, &mask).expect("masked_select_bcast");
    assert_eq!(out.shape(), &[3]);
    check_f32(
        "masked_select broadcast cpu fwd",
        &read_back_f32(&out),
        &[10.0, 20.0, 20.0],
        tolerance::F32_BITEXACT,
    );

    out.sum_all()
        .expect("masked_select sum")
        .backward()
        .expect("masked_select backward");
    let grad = input.grad().unwrap().expect("masked_select grad");
    assert_eq!(grad.shape(), &[1, 2]);
    check_f32(
        "masked_select broadcast cpu grad",
        &read_back_f32(&grad),
        &[1.0, 2.0],
        tolerance::F32_BITEXACT,
    );
}

#[test]
fn cpu_masked_select_public_api_scalar_broadcast_edges() {
    // Scalar input broadcasts to mask shape; selected scalar appearances reduce
    // back into the scalar leaf.
    let scalar = make_cpu_f32(&[5.0], &[], true);
    let vector_mask = BoolTensor::from_vec(vec![true, false, true], vec![3]).expect("vector mask");
    let out = masked_select(&scalar, &vector_mask).expect("scalar input masked_select");
    assert_eq!(out.shape(), &[2]);
    check_f32(
        "masked_select scalar input fwd",
        &read_back_f32(&out),
        &[5.0, 5.0],
        tolerance::F32_BITEXACT,
    );
    out.sum_all()
        .expect("scalar masked_select sum")
        .backward()
        .expect("scalar masked_select backward");
    let scalar_grad = scalar.grad().unwrap().expect("scalar masked_select grad");
    assert_eq!(scalar_grad.shape(), &[] as &[usize]);
    check_f32(
        "masked_select scalar input grad",
        &read_back_f32(&scalar_grad),
        &[2.0],
        tolerance::F32_BITEXACT,
    );

    // Scalar true mask broadcasts across a vector input. Exercise the inherent
    // Tensor<T>::masked_select surface, not just the free function.
    let vector = make_cpu_f32(&[1.0, 2.0, 3.0], &[3], true);
    let scalar_true = BoolTensor::from_vec(vec![true], vec![]).expect("scalar true mask");
    let out = vector
        .masked_select(&scalar_true)
        .expect("Tensor::masked_select scalar mask");
    assert_eq!(out.shape(), &[3]);
    check_f32(
        "masked_select scalar mask fwd",
        &read_back_f32(&out),
        &[1.0, 2.0, 3.0],
        tolerance::F32_BITEXACT,
    );
    out.sum_all()
        .expect("vector masked_select sum")
        .backward()
        .expect("vector masked_select backward");
    let vector_grad = vector.grad().unwrap().expect("vector masked_select grad");
    check_f32(
        "masked_select scalar mask grad",
        &read_back_f32(&vector_grad),
        &[1.0, 1.0, 1.0],
        tolerance::F32_BITEXACT,
    );

    let scalar_false = BoolTensor::from_vec(vec![false], vec![]).expect("scalar false mask");
    let empty = vector
        .masked_select(&scalar_false)
        .expect("Tensor::masked_select all false");
    assert_eq!(empty.shape(), &[0]);
    assert!(read_back_f32(&empty).is_empty());
}

#[test]
fn cpu_masked_select_backward_public_struct_validates_and_accepts_strided_grad() {
    let input = make_cpu_f32(&[1.0, 2.0, 3.0], &[3], false);
    let mask = BoolTensor::from_vec(vec![true, false, true], vec![3]).expect("mask");
    let grad_fn = MaskedSelectBackward {
        input: input.clone(),
        mask: mask.clone(),
    };

    let bad_grad = make_cpu_f32(&[1.0], &[1], false);
    let err = grad_fn
        .backward(&bad_grad)
        .expect_err("short compact grad must be rejected, not indexed");
    assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));

    let bad_mask_fn = MaskedSelectBackward {
        input: input.clone(),
        mask: BoolTensor::from_vec(vec![true, false, true, false], vec![4])
            .expect("wrong-size mask"),
    };
    let good_len_grad = make_cpu_f32(&[1.0, 2.0], &[2], false);
    let err = bad_mask_fn
        .backward(&good_len_grad)
        .expect_err("wrong-size saved mask must be rejected");
    assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));

    // Non-contiguous grad_output view, logical values [1, 2]. PyTorch's
    // autograd accepts non-contiguous incoming grads; the backward must read
    // logical order rather than requiring a contiguous slice.
    let grad_base = make_cpu_f32(&[9.0, 1.0, 8.0, 2.0], &[4], false);
    let grad_view = grad_base
        .as_strided(&[2], &[2], Some(1))
        .expect("strided grad_output");
    let grads = grad_fn
        .backward(&grad_view)
        .expect("strided compact grad is valid");
    let grad = grads[0].as_ref().expect("input grad");
    assert_eq!(grad.shape(), &[3]);
    check_f32(
        "masked_select backward strided grad",
        &read_back_f32(grad),
        &[1.0, 0.0, 2.0],
        tolerance::F32_BITEXACT,
    );

    let all_false_fn = MaskedSelectBackward {
        input,
        mask: BoolTensor::from_vec(vec![false, false, false], vec![3]).expect("all-false mask"),
    };
    let empty_grad = Tensor::from_storage(TensorStorage::cpu(Vec::<f32>::new()), vec![0], false)
        .expect("empty compact grad");
    let grads = all_false_fn
        .backward(&empty_grad)
        .expect("all-false compact grad");
    let grad = grads[0].as_ref().expect("all-false input grad");
    check_f32(
        "masked_select all-false backward",
        &read_back_f32(grad),
        &[0.0, 0.0, 0.0],
        tolerance::F32_BITEXACT,
    );
}

// ---------------------------------------------------------------------------
// Cat A.tensor_ops — triu / tril / diag / diagflat / roll / cdist
// ---------------------------------------------------------------------------

#[test]
fn cpu_triu() {
    let file = load_fixtures();
    let cases = cases_for(&file, "triu", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("triu cpu tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let diagonal = f.diagonal.expect("diagonal");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        match dtype {
            "float32" => {
                let a = make_cpu_f32(in_data, in_shape, false);
                let r = triu(&a, diagonal).expect("triu");
                check_f32(
                    &label,
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = make_cpu_f64(in_data, in_shape, false);
                let r = triu(&a, diagonal).expect("triu");
                check_f64(
                    &label,
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_tril() {
    let file = load_fixtures();
    let cases = cases_for(&file, "tril", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("tril cpu tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let diagonal = f.diagonal.expect("diagonal");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        match dtype {
            "float32" => {
                let a = make_cpu_f32(in_data, in_shape, false);
                let r = tril(&a, diagonal).expect("tril");
                check_f32(
                    &label,
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = make_cpu_f64(in_data, in_shape, false);
                let r = tril(&a, diagonal).expect("tril");
                check_f64(
                    &label,
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_diag() {
    let file = load_fixtures();
    let cases = cases_for(&file, "diag", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("diag cpu tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let diagonal = f.diagonal.expect("diagonal");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        match dtype {
            "float32" => {
                let a = make_cpu_f32(in_data, in_shape, false);
                let r = diag(&a, diagonal).expect("diag");
                check_f32(
                    &label,
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = make_cpu_f64(in_data, in_shape, false);
                let r = diag(&a, diagonal).expect("diag");
                check_f64(
                    &label,
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_diagflat() {
    let file = load_fixtures();
    let cases = cases_for(&file, "diagflat", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("diagflat cpu tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let diagonal = f.diagonal.expect("diagonal");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        match dtype {
            "float32" => {
                let a = make_cpu_f32(in_data, in_shape, false);
                let r = diagflat(&a, diagonal).expect("diagflat");
                check_f32(
                    &label,
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = make_cpu_f64(in_data, in_shape, false);
                let r = diagflat(&a, diagonal).expect("diagflat");
                check_f64(
                    &label,
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_roll() {
    let file = load_fixtures();
    let cases = cases_for(&file, "roll", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("roll cpu tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let shifts = f.shifts.expect("shifts");
        let dim = f.dim.expect("dim") as usize;
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        match dtype {
            "float32" => {
                let a = make_cpu_f32(in_data, in_shape, false);
                let r = roll(&a, shifts, dim).expect("roll");
                check_f32(
                    &label,
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = make_cpu_f64(in_data, in_shape, false);
                let r = roll(&a, shifts, dim).expect("roll");
                check_f64(
                    &label,
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_cdist() {
    let file = load_fixtures();
    let cases = cases_for(&file, "cdist", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("cdist cpu tag={tag} dtype={dtype}");
        let x1_shape = f.x1_shape.as_ref().expect("x1_shape");
        let x1_data = f.x1_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let x2_shape = f.x2_shape.as_ref().expect("x2_shape");
        let x2_data = f.x2_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let p = f.p.expect("p");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        match dtype {
            "float32" => {
                let x1 = make_cpu_f32(x1_data, x1_shape, false);
                let x2 = make_cpu_f32(x2_data, x2_shape, false);
                let r = cdist(&x1, &x2, p).expect("cdist");
                check_f32(
                    &label,
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_TRANSCENDENTAL_CPU,
                );
            }
            "float64" => {
                let x1 = make_cpu_f64(x1_data, x1_shape, false);
                let x2 = make_cpu_f64(x2_data, x2_shape, false);
                let r = cdist(&x1, &x2, p).expect("cdist");
                check_f64(
                    &label,
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_TRANSCENDENTAL_CPU,
                );
            }
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// Cat A.search — searchsorted / bucketize / unique / unique_consecutive /
// histc / meshgrid / topk
// ---------------------------------------------------------------------------

#[test]
fn cpu_searchsorted() {
    let file = load_fixtures();
    let cases = cases_for(&file, "searchsorted", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("searchsorted cpu tag={tag} dtype={dtype}");
        let bs = f.boundaries_shape.as_ref().expect("boundaries_shape");
        let bd = f
            .boundaries_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let vs = f.values_shape.as_ref().expect("values_shape");
        let vd = f
            .values_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let right = f.right.expect("right");
        let expected = f.out_indices.as_ref().expect("out_indices");
        match dtype {
            "float32" => {
                let b = make_cpu_f32(bd, bs, false);
                let v = make_cpu_f32(vd, vs, false);
                let r = searchsorted(&b, &v, right).expect("searchsorted");
                assert_eq!(&r, expected, "{label}: indices");
            }
            "float64" => {
                let b = make_cpu_f64(bd, bs, false);
                let v = make_cpu_f64(vd, vs, false);
                let r = searchsorted(&b, &v, right).expect("searchsorted");
                assert_eq!(&r, expected, "{label}: indices");
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_bucketize() {
    let file = load_fixtures();
    let cases = cases_for(&file, "bucketize", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("bucketize cpu tag={tag} dtype={dtype}");
        let is_ = f.input_shape.as_ref().expect("input_shape");
        let id = f
            .input_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let bs = f.boundaries_shape.as_ref().expect("boundaries_shape");
        let bd = f
            .boundaries_data
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let right = f.right.expect("right");
        let expected = f.out_indices.as_ref().expect("out_indices");
        match dtype {
            "float32" => {
                let inp = make_cpu_f32(id, is_, false);
                let bnd = make_cpu_f32(bd, bs, false);
                let r = bucketize(&inp, &bnd, right).expect("bucketize");
                assert_eq!(&r, expected, "{label}");
            }
            "float64" => {
                let inp = make_cpu_f64(id, is_, false);
                let bnd = make_cpu_f64(bd, bs, false);
                let r = bucketize(&inp, &bnd, right).expect("bucketize");
                assert_eq!(&r, expected, "{label}");
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_unique() {
    let file = load_fixtures();
    let cases = cases_for(&file, "unique", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("unique cpu tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let exp_uniq = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let exp_inv = f.out_inverse.as_ref().expect("out_inverse");
        let exp_cnt = f.out_counts.as_ref().expect("out_counts");
        match dtype {
            "float32" => {
                let a = make_cpu_f32(in_data, in_shape, false);
                let (uniq, inv, cnt) = unique(&a).expect("unique");
                check_f32(
                    &format!("{label} uniq"),
                    &read_back_f32(&uniq),
                    exp_uniq,
                    tolerance::F32_BITEXACT,
                );
                assert_eq!(&inv, exp_inv, "{label} inverse");
                assert_eq!(&cnt, exp_cnt, "{label} counts");
            }
            "float64" => {
                let a = make_cpu_f64(in_data, in_shape, false);
                let (uniq, inv, cnt) = unique(&a).expect("unique");
                check_f64(
                    &format!("{label} uniq"),
                    &read_back_f64(&uniq),
                    exp_uniq,
                    tolerance::F64_BITEXACT,
                );
                assert_eq!(&inv, exp_inv);
                assert_eq!(&cnt, exp_cnt);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_unique_consecutive() {
    let file = load_fixtures();
    let cases = cases_for(&file, "unique_consecutive", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("unique_consecutive cpu tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let exp_out = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let exp_inv = f.out_inverse.as_ref().expect("out_inverse");
        let exp_cnt = f.out_counts.as_ref().expect("out_counts");
        match dtype {
            "float32" => {
                let a = make_cpu_f32(in_data, in_shape, false);
                let (out, inv, cnt) = unique_consecutive(&a).expect("uc");
                check_f32(
                    &format!("{label} out"),
                    &read_back_f32(&out),
                    exp_out,
                    tolerance::F32_BITEXACT,
                );
                assert_eq!(&inv, exp_inv);
                assert_eq!(&cnt, exp_cnt);
            }
            "float64" => {
                let a = make_cpu_f64(in_data, in_shape, false);
                let (out, inv, cnt) = unique_consecutive(&a).expect("uc");
                check_f64(
                    &format!("{label} out"),
                    &read_back_f64(&out),
                    exp_out,
                    tolerance::F64_BITEXACT,
                );
                assert_eq!(&inv, exp_inv);
                assert_eq!(&cnt, exp_cnt);
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_histc() {
    let file = load_fixtures();
    let cases = cases_for(&file, "histc", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("histc cpu tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let bins = f.bins.expect("bins");
        let mn = f.min.expect("min");
        let mx = f.max.expect("max");
        let expected = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        match dtype {
            "float32" => {
                let a = make_cpu_f32(in_data, in_shape, false);
                let r = histc(&a, bins, mn, mx).expect("histc");
                check_f32(
                    &label,
                    &read_back_f32(&r),
                    expected,
                    tolerance::F32_BITEXACT,
                );
            }
            "float64" => {
                let a = make_cpu_f64(in_data, in_shape, false);
                let r = histc(&a, bins, mn, mx).expect("histc");
                check_f64(
                    &label,
                    &read_back_f64(&r),
                    expected,
                    tolerance::F64_BITEXACT,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn cpu_meshgrid() {
    let file = load_fixtures();
    let mut count = 0;
    for f in cases_for(&file, "meshgrid", "cpu") {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("meshgrid cpu tag={tag} dtype={dtype}");
        let input_shapes = f.input_shapes.as_ref().expect("input_shapes");
        let input_data = f.mg_input_data.as_ref().expect("mg_input_data");
        let out_values = f.mg_out_values.as_ref().expect("mg_out_values");

        match dtype {
            "float32" => {
                let inputs: Vec<Tensor<f32>> = input_shapes
                    .iter()
                    .zip(input_data.iter())
                    .map(|(s, d)| make_cpu_f32(d.as_slice(), s, false))
                    .collect();
                let grids = meshgrid(&inputs).expect("meshgrid");
                assert_eq!(grids.len(), out_values.len(), "{label}: grid count");
                for (i, (g, exp)) in grids.iter().zip(out_values.iter()).enumerate() {
                    check_f32(
                        &format!("{label} grid[{i}]"),
                        &read_back_f32(g),
                        exp.as_slice(),
                        tolerance::F32_BITEXACT,
                    );
                }
            }
            "float64" => {
                let inputs: Vec<Tensor<f64>> = input_shapes
                    .iter()
                    .zip(input_data.iter())
                    .map(|(s, d)| make_cpu_f64(d.as_slice(), s, false))
                    .collect();
                let grids = meshgrid(&inputs).expect("meshgrid");
                for (i, (g, exp)) in grids.iter().zip(out_values.iter()).enumerate() {
                    check_f64(
                        &format!("{label} grid[{i}]"),
                        &read_back_f64(g),
                        exp.as_slice(),
                        tolerance::F64_BITEXACT,
                    );
                }
            }
            _ => unreachable!(),
        }
        count += 1;
    }
    assert!(count > 0, "no meshgrid fixtures executed");
}

/// `meshgrid_indexing(.., MeshIndexing::Xy)` mirrors `torch.meshgrid(*t,
/// indexing='xy')` (`aten/src/ATen/native/TensorShape.cpp:4433-4438,4470-4472`):
/// the first two inputs and the first two output grids are swapped. Live torch
/// 2.11.0+cu130 oracle (R-CHAR-3, named reference — not copied from ferrotorch):
///   torch.meshgrid([1,2,3],[4,5], indexing='xy')[0] -> [1,2,3,1,2,3] (shape [2,3])
///   torch.meshgrid([1,2,3],[4,5], indexing='xy')[1] -> [4,4,4,5,5,5]
/// `MeshIndexing::Ij` reproduces the default `meshgrid` result unchanged.
#[test]
fn cpu_meshgrid_indexing_xy() {
    let x = make_cpu_f32(&[1.0, 2.0, 3.0], &[3], false);
    let y = make_cpu_f32(&[4.0, 5.0], &[2], false);

    let xy = meshgrid_indexing(&[x.clone(), y.clone()], MeshIndexing::Xy).expect("xy");
    assert_eq!(xy.len(), 2);
    assert_eq!(xy[0].shape(), &[2, 3]);
    assert_eq!(xy[1].shape(), &[2, 3]);
    check_f32(
        "meshgrid xy grid[0]",
        &read_back_f32(&xy[0]),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        tolerance::F32_BITEXACT,
    );
    check_f32(
        "meshgrid xy grid[1]",
        &read_back_f32(&xy[1]),
        &[4.0, 4.0, 4.0, 5.0, 5.0, 5.0],
        tolerance::F32_BITEXACT,
    );

    // Ij convention matches the default `meshgrid`.
    let ij = meshgrid_indexing(&[x.clone(), y.clone()], MeshIndexing::Ij).expect("ij");
    let def = meshgrid(&[x, y]).expect("default ij");
    assert_eq!(read_back_f32(&ij[0]), read_back_f32(&def[0]));
    assert_eq!(read_back_f32(&ij[1]), read_back_f32(&def[1]));
}

#[test]
fn cpu_topk() {
    let file = load_fixtures();
    let cases = cases_for(&file, "topk", "cpu");
    assert!(!cases.is_empty());
    for f in cases {
        let dtype = f.dtype.as_deref().unwrap_or("");
        let tag = f.tag.as_deref().unwrap_or("");
        let label = format!("topk cpu tag={tag} dtype={dtype}");
        let in_shape = f.in_shape.as_ref().unwrap();
        let in_data = f.in_data.as_ref().map(F64ListSentinel::as_slice).unwrap();
        let k = f.k.expect("k");
        let largest = f.largest.expect("largest");
        let exp_v = f
            .out_values
            .as_ref()
            .map(F64ListSentinel::as_slice)
            .unwrap();
        let exp_i = f.out_indices.as_ref().expect("out_indices");
        match dtype {
            "float32" => {
                let a = make_cpu_f32(in_data, in_shape, false);
                let (vals, idxs) = topk(&a, k, largest).expect("topk");
                check_f32(
                    &format!("{label} vals"),
                    &read_back_f32(&vals),
                    exp_v,
                    tolerance::F32_BITEXACT,
                );
                assert_eq!(&idxs, exp_i, "{label} idxs");
            }
            "float64" => {
                let a = make_cpu_f64(in_data, in_shape, false);
                let (vals, idxs) = topk(&a, k, largest).expect("topk");
                check_f64(
                    &format!("{label} vals"),
                    &read_back_f64(&vals),
                    exp_v,
                    tolerance::F64_BITEXACT,
                );
                assert_eq!(&idxs, exp_i);
            }
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// Cat B — pure shape utility helpers (no torch invocation, parity proven by
// PyTorch's documented semantics codified in the fixture).
// ---------------------------------------------------------------------------

#[test]
fn shape_helpers_broadcast_shapes() {
    let file = load_fixtures();
    for f in cases_op_only(&file, "broadcast_shapes") {
        let a: Vec<usize> = f.a.as_ref().unwrap().iter().map(|&x| x as usize).collect();
        let b: Vec<usize> = f.b.as_ref().unwrap().iter().map(|&x| x as usize).collect();
        let expected = f.expected.as_ref().unwrap();
        let got = broadcast_shapes(&a, &b).expect("broadcast_shapes");
        assert_eq!(&got, expected, "broadcast_shapes({a:?},{b:?})");
    }
}

#[test]
fn shape_helpers_numel() {
    let file = load_fixtures();
    for f in cases_op_only(&file, "numel") {
        let shape = f.shape.as_ref().unwrap();
        let expected = f.expected_numel.unwrap();
        assert_eq!(numel(shape), expected, "numel({shape:?})");
    }
}

#[test]
fn shape_helpers_checked_numel_and_byte_count_reject_overflow() {
    assert_eq!(checked_numel(&[2, 3, 4], "checked_numel").unwrap(), 24);
    let numel_err = checked_numel(&[usize::MAX, 2], "checked_numel")
        .expect_err("checked_numel must reject overflowing products");
    assert!(
        format!("{numel_err:?}").contains("overflows usize"),
        "unexpected error: {numel_err:?}"
    );

    assert_eq!(checked_byte_count(6, 4, "checked_byte_count").unwrap(), 24);
    let byte_err = checked_byte_count((usize::MAX / 2) + 1, 2, "checked_byte_count")
        .expect_err("checked_byte_count must reject overflowing storage sizes");
    assert!(
        format!("{byte_err:?}").contains("storage size calculation overflowed"),
        "unexpected error: {byte_err:?}"
    );
}

#[test]
fn shape_helpers_c_contiguous_strides() {
    let file = load_fixtures();
    for f in cases_op_only(&file, "c_contiguous_strides") {
        let shape = f.shape.as_ref().unwrap();
        let expected: Vec<isize> = f
            .expected_strides
            .as_ref()
            .unwrap()
            .iter()
            .map(|&x| x as isize)
            .collect();
        let got = c_contiguous_strides(shape);
        assert_eq!(got, expected, "c_contiguous_strides({shape:?})");
    }

    let err = checked_c_contiguous_strides(&[0, usize::MAX], "checked_c_contiguous_strides")
        .expect_err("checked C strides must reject dimensions outside signed metadata");
    assert!(
        format!("{err:?}").contains("exceeds isize::MAX"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn shape_helpers_channels_last_strides() {
    let file = load_fixtures();
    for f in cases_op_only(&file, "channels_last_strides") {
        let shape = f.shape.as_ref().unwrap();
        let expected: Vec<isize> = f
            .expected_strides
            .as_ref()
            .unwrap()
            .iter()
            .map(|&x| x as isize)
            .collect();
        let got = channels_last_strides(shape);
        assert_eq!(got, expected, "channels_last_strides({shape:?})");
    }

    let err = checked_channels_last_strides(
        &[1, isize::MAX as usize, 1, 2],
        "checked_channels_last_strides",
    )
    .expect_err("checked channels-last strides must reject signed overflow");
    assert!(
        format!("{err:?}").contains("exceeds isize::MAX"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn shape_helpers_channels_last_3d_strides() {
    let file = load_fixtures();
    for f in cases_op_only(&file, "channels_last_3d_strides") {
        let shape = f.shape.as_ref().unwrap();
        let expected: Vec<isize> = f
            .expected_strides
            .as_ref()
            .unwrap()
            .iter()
            .map(|&x| x as isize)
            .collect();
        let got = channels_last_3d_strides(shape);
        assert_eq!(got, expected, "channels_last_3d_strides({shape:?})");
    }

    let err = checked_channels_last_3d_strides(
        &[1, isize::MAX as usize, 1, 1, 2],
        "checked_channels_last_3d_strides",
    )
    .expect_err("checked channels-last-3d strides must reject signed overflow");
    assert!(
        format!("{err:?}").contains("exceeds isize::MAX"),
        "unexpected error: {err:?}"
    );
}

#[test]
fn shape_helpers_normalize_axis() {
    let file = load_fixtures();
    for f in cases_op_only(&file, "normalize_axis") {
        let axis = f.axis.unwrap() as isize;
        let ndim = f.ndim.unwrap();
        let expected = f.expected_axis.unwrap();
        let got = normalize_axis(axis, ndim).expect("normalize_axis ok");
        assert_eq!(got, expected, "normalize_axis({axis},{ndim})");
    }
    for f in cases_op_only(&file, "normalize_axis_err") {
        let axis = f.axis.unwrap() as isize;
        let ndim = f.ndim.unwrap();
        let r = normalize_axis(axis, ndim);
        assert!(r.is_err(), "normalize_axis({axis},{ndim}) must be Err");
    }
}

#[test]
fn shape_helpers_check_shapes_match() {
    let file = load_fixtures();
    for f in cases_op_only(&file, "check_shapes_match_ok") {
        let a: Vec<usize> = f.a.as_ref().unwrap().iter().map(|&x| x as usize).collect();
        let b: Vec<usize> = f.b.as_ref().unwrap().iter().map(|&x| x as usize).collect();
        check_shapes_match(&a, &b, "test").expect("ok");
    }
    for f in cases_op_only(&file, "check_shapes_match_err") {
        let a: Vec<usize> = f.a.as_ref().unwrap().iter().map(|&x| x as usize).collect();
        let b: Vec<usize> = f.b.as_ref().unwrap().iter().map(|&x| x as usize).collect();
        let r = check_shapes_match(&a, &b, "test");
        assert!(r.is_err(), "check_shapes_match({a:?},{b:?}) must be Err");
    }
}

// ---------------------------------------------------------------------------
// Cat C/D — backward grad_fn structs: implicit coverage via the autograd
// assertions above. To keep the symbol table linked we exercise each name
// path here so dead-code analysis confirms the items are reachable, even if
// the per-op autograd asserts already cover correctness.
// ---------------------------------------------------------------------------

#[test]
#[allow(clippy::similar_names, reason = "intentional naming parity")]
fn grad_fn_struct_paths_link() {
    // Reshape / Flatten / Squeeze / Unsqueeze / Transpose: built via
    // autograd path triggered by `*_t` calls above; here we only verify
    // the public path resolves (`new` constructors) so any rename is
    // surfaced at compile time rather than via missing autograd grad.
    let leaf = make_cpu_f32(&[1.0_f32 as f64], &[1], false);
    let _r = ReshapeBackward::<f32>::new(leaf.clone(), vec![1]);
    let _f = FlattenBackward::<f32>::new(leaf.clone(), vec![1]);
    let _s = SqueezeBackward::<f32>::new(leaf.clone(), 0);
    let _u = UnsqueezeBackward::<f32>::new(leaf.clone(), 0);
    let _t = TransposeBackward::<f32>::new(leaf.clone());
    let _e = ExpandBackward::<f32>::new(leaf.clone(), vec![1]);
    let _c = CatBackward::<f32>::new(vec![leaf.clone()], 0, vec![1]);
    let _sp = SplitBackward::<f32>::new(leaf.clone(), 0, 0, 1);
    let _as = AsStridedBackward::<f32>::new(leaf.clone(), vec![1], vec![1], 0);
    let _ass = AsStridedScatterBackward::new(leaf.clone(), leaf.clone(), vec![1], vec![1], 0);

    // Indexing grad_fn structs — again, just confirm names resolve.
    let _g = GatherBackward::<f32> {
        input: leaf.clone(),
        dim: 0,
        index: vec![0],
        index_cuda: None,
        index_shape: vec![1],
    };
    let _is = IndexSelectBackward::<f32> {
        input: leaf.clone(),
        indices: vec![0],
    };
    let _isd = IndexSelectDimBackward::<f32> {
        input: leaf.clone(),
        dim: 0,
        indices: vec![0],
        indices_cuda: None,
    };
    let _mf = MaskedFillBackward::<f32> {
        input: leaf.clone(),
        mask: BoolTensor::from_vec(vec![false], vec![1]).unwrap(),
    };
    let _sa = ScatterBackward::<f32> {
        input: leaf.clone(),
        src: leaf.clone(),
        dim: 0,
        index: vec![0],
        index_shape: vec![1],
    };
    let _saa = ScatterAddBackward::<f32> {
        input: leaf.clone(),
        src: leaf.clone(),
        dim: 0,
        index: vec![0],
        index_shape: vec![1],
    };
    let _w = WhereCondBackward::<f32> {
        x: leaf.clone(),
        y: leaf.clone(),
        condition: BoolTensor::from_vec(vec![true], vec![1]).unwrap(),
    };
}

/// Node-level VJP contract of `AsStridedScatterBackward` (CORE-060 /
/// #1754): grads w.r.t. BOTH base and src.
///
/// torch oracle (live 2.11.0+cu130) for the src gather (torch == finite
/// differences here):
/// ```python
/// base = torch.zeros(6, dtype=torch.float64, requires_grad=True)
/// src = torch.tensor([10.,20.,30.], dtype=torch.float64, requires_grad=True)
/// out = torch.as_strided_scatter(base, src, [3],[2],0)
/// (out * torch.arange(1.,7.)).sum().backward()
/// src.grad   # tensor([1., 3., 5.])
/// ```
/// The base grad pins the finite-difference Jacobian `[0,2,0,4,0,6]`
/// (upstream grad with the scattered region zeroed); torch 2.11.0's
/// analytic formula returns the opposite masking `[1,0,3,0,5,0]` and fails
/// its own `torch.autograd.gradcheck` — deliberate divergence tracked in
/// #1959.
#[test]
fn cpu_as_strided_scatter_backward_node_grads() {
    let base = make_cpu_f32(&[0.0; 6], &[6], true);
    let src = make_cpu_f32(&[10.0, 20.0, 30.0], &[3], true);
    let node = AsStridedScatterBackward::new(base.clone(), src.clone(), vec![3], vec![2], 0);
    // Upstream grad = [1..6] (the weighted-sum loss's local gradient).
    let upstream = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], false);
    let grads = ferrotorch_core::GradFn::backward(&node, &upstream).expect("backward");
    assert_eq!(grads.len(), 2, "one grad slot per input (base, src)");
    let g_base = grads[0].as_ref().expect("base grad").data_vec().unwrap();
    let g_src = grads[1].as_ref().expect("src grad").data_vec().unwrap();
    assert_eq!(g_base, vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0]);
    assert_eq!(g_src, vec![1.0, 3.0, 5.0]);
}

// ---------------------------------------------------------------------------
// Sanity: assert the fixture file has every op we expect (catches stale
// regenerator runs).
// ---------------------------------------------------------------------------

#[test]
fn fixture_file_covers_every_op() {
    let file = load_fixtures();
    let mut by_op: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for f in &file.fixtures {
        *by_op.entry(f.op.as_str()).or_insert(0) += 1;
    }
    let required = [
        // Cat A.shape
        "reshape",
        "view",
        "flatten",
        "squeeze",
        "unsqueeze",
        "transpose",
        "permute",
        "narrow",
        "contiguous",
        "cat",
        "split",
        "chunk",
        "expand",
        "as_strided",
        "as_strided_copy",
        "as_strided_scatter",
        // Cat A.indexing
        "gather",
        "scatter",
        "scatter_add",
        "where_cond",
        "index_select_1d",
        "masked_fill",
        // Cat A.tensor_ops
        "triu",
        "tril",
        "diag",
        "diagflat",
        "roll",
        "cdist",
        // Cat A.search
        "searchsorted",
        "bucketize",
        "unique",
        "unique_consecutive",
        "histc",
        "meshgrid",
        "topk",
        // Cat B
        "broadcast_shapes",
        "numel",
        "c_contiguous_strides",
        "channels_last_strides",
        "channels_last_3d_strides",
        "normalize_axis",
    ];
    for r in required {
        let n = by_op.get(r).copied().unwrap_or(0);
        assert!(n > 0, "fixture file missing op {r:?}");
    }
}

// ---------------------------------------------------------------------------
// Cat A.indexing — GPU error-path assertions for ops without GPU kernels.
//
// PyTorch parity (per `rust-gpu-discipline` §3): a GPU tensor passed to an
// op without a CUDA kernel must return Err, not silently fall back. The
// canonical ferrotorch error is `FerrotorchError::NotImplementedOnCuda`.
// We exercise this lane unconditionally from the CPU build (no GPU init
// required — we synthesize a CUDA tensor only if the gpu module loads it).
// ---------------------------------------------------------------------------

// These tests live in the gpu cfg-module since they need a running CUDA
// backend to construct a CUDA tensor in the first place.

// ---------------------------------------------------------------------------
// GPU paths — gated on the `gpu` feature
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
                "fixtures/shape.json was generated without CUDA — \
                 regenerate on a CUDA-enabled host before running --features gpu tests"
            );
        }
    }

    // --- Cat A.shape on GPU ---

    #[test]
    fn gpu_reshape() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_reshape_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_view() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_view_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_flatten() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_flatten_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_squeeze() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_squeeze_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_unsqueeze() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_unsqueeze_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_transpose() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_transpose_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_permute() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_permute_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_narrow() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_narrow_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_contiguous() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_contiguous_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_cat() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_cat_for_device("cuda:0", Device::Cuda(0));
    }

    /// End-to-end Tensor<bf16> cat on CUDA — verifies the dtype-generic
    /// `strided_cat` path (#22, #1181) at the public `core::cat` surface,
    /// not just the backend trait. Previously `Tensor::<bf16>::cat` on
    /// `Device::Cuda(0)` returned `NotImplementedOnCuda { op: "cat" }`;
    /// the byte-oriented dispatch now routes 2-byte elements to the new
    /// `strided_cat_u16_kernel` (see ferrotorch-gpu/src/kernels.rs).
    #[test]
    fn gpu_cat_bf16_end_to_end() {
        use half::bf16;
        ensure_cuda_backend();

        // Two simple bf16 tensors that round-trip through f32 without loss
        // (small integers are exactly representable in bf16).
        let a_data: Vec<bf16> = (0..6).map(|i| bf16::from_f32(i as f32)).collect();
        let b_data: Vec<bf16> = (10..19).map(|i| bf16::from_f32(i as f32)).collect();
        let a = Tensor::from_storage(TensorStorage::cpu(a_data), vec![2, 3], false)
            .expect("a bf16 cpu");
        let b = Tensor::from_storage(TensorStorage::cpu(b_data), vec![3, 3], false)
            .expect("b bf16 cpu");

        let want = cat(&[a.clone(), b.clone()], 0).expect("cpu cat bf16");
        let a_gpu = a.to(Device::Cuda(0)).expect("upload a");
        let b_gpu = b.to(Device::Cuda(0)).expect("upload b");
        let got = cat(&[a_gpu, b_gpu], 0)
            .expect("GPU bf16 cat must succeed (was NotImplementedOnCuda before #1181)");
        assert!(got.is_cuda(), "GPU cat result must stay on CUDA");
        assert_eq!(got.shape(), &[5, 3]);

        let got_cpu = got.cpu().expect("download");
        let want_bits: Vec<u16> = want.data().unwrap().iter().map(|v| v.to_bits()).collect();
        let got_bits: Vec<u16> = got_cpu
            .data()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got_bits, want_bits, "bit-exact bf16 cat required");
    }

    #[test]
    fn gpu_split() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_split_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_chunk() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_chunk_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_expand() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_expand_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_as_strided() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_as_strided_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_as_strided_copy() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_as_strided_copy_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_as_strided_scatter() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_as_strided_scatter_for_device("cuda:0", Device::Cuda(0));
    }

    // --- Cat A.indexing GPU lane ---
    //
    // index_select_1d / masked_fill have GPU f32 kernels. The direct
    // indexing primitives below also have CUDA-resident paths, so the GPU
    // lane asserts value parity and output residency rather than pinning
    // historical NotImplementedOnCuda gaps.

    #[test]
    fn gpu_index_select_1d_f32() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_index_select_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_phase2c_index_select_scalar() {
        ensure_cuda_backend();
        let input = make_cpu_f32(&[5.0], &[], false)
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(true);
        let index = IntTensor::<i64>::from_vec(vec![0], vec![1])
            .expect("one index")
            .to(Device::Cuda(0))
            .expect("upload index");

        let out = input
            .index_select(0, &index)
            .expect("CUDA scalar index_select");
        assert_eq!(out.device(), Device::Cuda(0), "scalar index_select device");
        assert_eq!(out.shape(), &[] as &[usize]);
        check_f32(
            "phase2c tensor index_select scalar cuda fwd",
            &read_back_f32(&out),
            &[5.0],
            tolerance::F32_BITEXACT,
        );

        out.backward().expect("CUDA scalar index_select backward");
        let grad = input
            .grad()
            .unwrap()
            .expect("CUDA scalar index_select grad");
        assert_eq!(
            grad.device(),
            Device::Cuda(0),
            "scalar index_select grad device"
        );
        assert_eq!(grad.shape(), &[] as &[usize]);
        check_f32(
            "phase2c tensor index_select scalar cuda grad",
            &read_back_f32(&grad),
            &[1.0],
            tolerance::F32_BITEXACT,
        );

        let int_input = IntTensor::<i64>::from_vec(vec![5], vec![])
            .expect("scalar int tensor")
            .to(Device::Cuda(0))
            .expect("upload int tensor");
        let int_out = int_input
            .index_select(0, &index)
            .expect("CUDA IntTensor scalar index_select");
        assert!(
            int_out.is_cuda(),
            "IntTensor scalar index_select must stay CUDA-resident"
        );
        assert_eq!(int_out.shape(), &[] as &[usize]);
        assert_eq!(int_out.to(Device::Cpu).unwrap().data().unwrap(), &[5]);
    }

    #[test]
    fn gpu_phase2c_tensor_gather_public_api() {
        ensure_cuda_backend();
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![2, 3],
            false,
        )
        .expect("input")
        .to(Device::Cuda(0))
        .expect("upload")
        .requires_grad_(true);
        let index = IntTensor::<i32>::from_vec(vec![2, 0, 0, 2], vec![2, 2])
            .expect("i32 index")
            .to(Device::Cuda(0))
            .expect("upload index");

        let out = input
            .gather(-1, &index)
            .expect("CUDA Tensor<T>::gather public API");

        assert_eq!(out.device(), Device::Cuda(0), "gather output device");
        assert_eq!(out.shape(), &[2, 2]);
        check_f32(
            "phase2c tensor gather cuda fwd",
            &read_back_f32(&out),
            &[3.0, 1.0, 4.0, 6.0],
            tolerance::F32_BITEXACT,
        );

        out.sum_all().expect("sum").backward().expect("backward");
        let grad = input.grad().unwrap().expect("gather cuda grad");
        assert_eq!(grad.device(), Device::Cuda(0), "gather grad device");
        assert_eq!(grad.shape(), &[2, 3]);
        check_f32(
            "phase2c tensor gather cuda grad",
            &read_back_f32(&grad),
            &[1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            tolerance::F32_BITEXACT,
        );
    }

    #[test]
    fn gpu_phase2c_inttensor_gather_public_api() {
        ensure_cuda_backend();
        let input = IntTensor::<i64>::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3])
            .expect("int input")
            .to(Device::Cuda(0))
            .expect("upload input");
        let index = IntTensor::<i32>::from_vec(vec![1, 0], vec![1, 2])
            .expect("i32 index")
            .to(Device::Cuda(0))
            .expect("upload index");

        let out = input
            .gather(1, &index)
            .expect("CUDA IntTensor<I>::gather public API");

        assert!(out.is_cuda(), "IntTensor gather must stay CUDA-resident");
        assert_eq!(out.shape(), &[1, 2]);
        assert_eq!(read_back_int(&out), vec![2, 1]);
    }

    #[test]
    fn gpu_phase2c_argmax_argmin_public_api() {
        ensure_cuda_backend();
        let floats = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, f32::NAN, 3.0, 4.0, f32::NAN, 2.0]),
            vec![2, 3],
            false,
        )
        .expect("float input")
        .to(Device::Cuda(0))
        .expect("upload floats");

        let flat_max = floats.argmax(None).unwrap();
        let flat_min = floats.argmin(None).unwrap();
        assert!(flat_max.is_cuda(), "argmax output must stay CUDA-resident");
        assert!(flat_min.is_cuda(), "argmin output must stay CUDA-resident");
        assert_eq!(read_back_int(&flat_max), vec![1]);
        assert_eq!(read_back_int(&flat_min), vec![1]);

        let dim_max = floats.argmax(Some(1)).unwrap();
        let dim_min = floats.argmin(Some(1)).unwrap();
        assert!(dim_max.is_cuda(), "argmax(dim) output device");
        assert!(dim_min.is_cuda(), "argmin(dim) output device");
        assert_eq!(read_back_int(&dim_max), vec![1, 1]);
        assert_eq!(read_back_int(&dim_min), vec![1, 1]);

        let ints = IntTensor::<i32>::from_vec(vec![7, 7, -3, 9, 9, 1], vec![2, 3])
            .expect("int input")
            .to(Device::Cuda(0))
            .expect("upload ints");
        let int_flat_max = ints.argmax(None).unwrap();
        let int_flat_min = ints.argmin(None).unwrap();
        assert!(
            int_flat_max.is_cuda(),
            "IntTensor argmax output must stay CUDA-resident"
        );
        assert!(
            int_flat_min.is_cuda(),
            "IntTensor argmin output must stay CUDA-resident"
        );
        assert_eq!(read_back_int(&int_flat_max), vec![3]);
        assert_eq!(read_back_int(&int_flat_min), vec![2]);
        assert_eq!(read_back_int(&ints.argmax(Some(-1)).unwrap()), vec![0, 0]);
        assert_eq!(read_back_int(&ints.argmin(Some(-1)).unwrap()), vec![2, 2]);
    }

    #[test]
    fn gpu_phase2c_cast_public_api_matches_torch_cuda() {
        ensure_cuda_backend();
        let int_values = vec![
            0_i64,
            1,
            -1,
            65_504,
            65_505,
            100_000,
            16_777_217,
            i64::from(i32::MAX),
            i64::from(i32::MIN),
            i64::MAX,
            i64::MIN,
        ];
        let ints = IntTensor::<i64>::from_vec(int_values, vec![11])
            .expect("int input")
            .to(Device::Cuda(0))
            .expect("upload ints");

        let f16 = ints.to_float::<half::f16>().expect("i64 to f16");
        assert!(f16.is_cuda(), "to_float::<f16> must stay CUDA-resident");
        assert_eq!(
            read_back_f16_bits(&f16),
            vec![
                0x0000, 0x3c00, 0xbc00, 0x7bff, 0x7bff, 0x7c00, 0x7c00, 0x7c00, 0xfc00, 0x7c00,
                0xfc00,
            ]
        );

        let bf16 = ints.to_float::<half::bf16>().expect("i64 to bf16");
        assert!(bf16.is_cuda(), "to_float::<bf16> must stay CUDA-resident");
        assert_eq!(
            read_back_bf16_bits(&bf16),
            vec![
                0x0000, 0x3f80, 0xbf80, 0x4780, 0x4780, 0x47c3, 0x4b80, 0x4f00, 0xcf00, 0x5f00,
                0xdf00,
            ]
        );

        let f32 = ints.to_float::<f32>().expect("i64 to f32");
        assert!(f32.is_cuda(), "to_float::<f32> must stay CUDA-resident");
        let f32_bits: Vec<u32> = read_back_f32(&f32).iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            f32_bits,
            vec![
                0x00000000, 0x3f800000, 0xbf800000, 0x477fe000, 0x477fe100, 0x47c35000, 0x4b800000,
                0x4f000000, 0xcf000000, 0x5f000000, 0xdf000000,
            ]
        );

        let f64 = ints.to_float::<f64>().expect("i64 to f64");
        assert!(f64.is_cuda(), "to_float::<f64> must stay CUDA-resident");
        let f64_bits: Vec<u64> = read_back_f64(&f64).iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            f64_bits,
            vec![
                0x0000000000000000,
                0x3ff0000000000000,
                0xbff0000000000000,
                0x40effc0000000000,
                0x40effc2000000000,
                0x40f86a0000000000,
                0x4170000010000000,
                0x41dfffffffc00000,
                0xc1e0000000000000,
                0x43e0000000000000,
                0xc3e0000000000000,
            ]
        );

        let floats = Tensor::from_storage(
            TensorStorage::cpu(vec![
                0.0_f32,
                1.9,
                -1.9,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                2_147_483_647.0,
                2_147_483_648.0,
                -2_147_483_649.0,
            ]),
            vec![9],
            false,
        )
        .expect("float input")
        .to(Device::Cuda(0))
        .expect("upload floats");

        let i32s: IntTensor<i32> = Tensor::<f32>::to_int(&floats).expect("f32 to i32");
        assert!(i32s.is_cuda(), "to_int::<i32> must stay CUDA-resident");
        assert_eq!(
            read_back_int(&i32s),
            vec![
                0,
                1,
                -1,
                0,
                i32::MAX,
                i32::MIN,
                i32::MAX,
                i32::MAX,
                i32::MIN
            ]
        );

        let i64s = floats.to_int::<i64>().expect("f32 to i64");
        assert!(i64s.is_cuda(), "to_int::<i64> must stay CUDA-resident");
        assert_eq!(
            read_back_int(&i64s),
            vec![
                0,
                1,
                -1,
                i64::MIN,
                i64::MAX,
                i64::MIN,
                2_147_483_648,
                2_147_483_648,
                -2_147_483_648,
            ]
        );
    }

    #[test]
    fn gpu_index_select_dim_scalar() {
        ensure_cuda_backend();
        let input = make_cpu_f32(&[5.0], &[], false)
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(true);
        let index = IntTensor::<i64>::from_vec(vec![0], vec![1])
            .expect("one index")
            .to(Device::Cuda(0))
            .expect("upload index");

        let out = index_select_dim(&input, 0, &index).expect("CUDA scalar index_select_dim");
        assert_eq!(
            out.device(),
            Device::Cuda(0),
            "scalar index_select_dim device"
        );
        assert_eq!(out.shape(), &[] as &[usize]);
        check_f32(
            "index_select_dim scalar cuda fwd",
            &read_back_f32(&out),
            &[5.0],
            tolerance::F32_BITEXACT,
        );

        out.backward()
            .expect("CUDA scalar index_select_dim backward");
        let grad = input
            .grad()
            .unwrap()
            .expect("CUDA scalar index_select_dim grad");
        assert_eq!(
            grad.device(),
            Device::Cuda(0),
            "scalar index_select_dim grad device"
        );
        assert_eq!(grad.shape(), &[] as &[usize]);
        check_f32(
            "index_select_dim scalar cuda grad",
            &read_back_f32(&grad),
            &[1.0],
            tolerance::F32_BITEXACT,
        );
    }

    #[test]
    fn gpu_masked_fill_f32() {
        ensure_cuda_backend();
        let file = load_fixtures();
        require_cuda_fixtures(&file);
        run_masked_fill_for_device("cuda:0", Device::Cuda(0));
    }

    #[test]
    fn gpu_masked_select_public_api_broadcast_and_backward() {
        ensure_cuda_backend();
        let input = upload_f32(make_cpu_f32(&[10.0, 20.0], &[1, 2], true), Device::Cuda(0));
        let mask = BoolTensor::from_vec(vec![true, true, false, true], vec![2, 2])
            .expect("broadcast mask")
            .to(Device::Cuda(0))
            .expect("upload mask");

        let out = masked_select_bcast(&input, &mask).expect("CUDA masked_select_bcast");
        assert!(
            out.is_cuda(),
            "masked_select_bcast output must stay CUDA-resident"
        );
        assert_eq!(out.shape(), &[3]);
        check_f32(
            "masked_select broadcast cuda fwd",
            &read_back_f32(&out),
            &[10.0, 20.0, 20.0],
            tolerance::F32_BITEXACT,
        );

        out.sum_all()
            .expect("CUDA masked_select sum")
            .backward()
            .expect("CUDA masked_select backward");
        let grad = input.grad().unwrap().expect("CUDA masked_select grad");
        assert!(
            grad.is_cuda(),
            "masked_select_bcast grad must stay CUDA-resident"
        );
        assert_eq!(grad.shape(), &[1, 2]);
        check_f32(
            "masked_select broadcast cuda grad",
            &read_back_f32(&grad),
            &[1.0, 2.0],
            tolerance::F32_BITEXACT,
        );

        let scalar = upload_f32(make_cpu_f32(&[5.0], &[], true), Device::Cuda(0));
        let vector_mask = BoolTensor::from_vec(vec![true, false, true], vec![3])
            .expect("vector mask")
            .to(Device::Cuda(0))
            .expect("upload vector mask");
        let out = masked_select(&scalar, &vector_mask).expect("CUDA scalar input masked_select");
        assert!(out.is_cuda(), "scalar masked_select output device");
        assert_eq!(out.shape(), &[2]);
        check_f32(
            "masked_select scalar cuda fwd",
            &read_back_f32(&out),
            &[5.0, 5.0],
            tolerance::F32_BITEXACT,
        );
        out.sum_all()
            .expect("CUDA scalar masked_select sum")
            .backward()
            .expect("CUDA scalar masked_select backward");
        let grad = scalar
            .grad()
            .unwrap()
            .expect("CUDA scalar masked_select grad");
        assert!(grad.is_cuda(), "scalar masked_select grad device");
        assert_eq!(grad.shape(), &[] as &[usize]);
        check_f32(
            "masked_select scalar cuda grad",
            &read_back_f32(&grad),
            &[2.0],
            tolerance::F32_BITEXACT,
        );
    }

    #[test]
    fn gpu_masked_select_backward_public_struct_rejects_bad_compact_len() {
        ensure_cuda_backend();
        let input = make_cpu_f32(&[1.0, 2.0, 3.0], &[3], false)
            .to(Device::Cuda(0))
            .expect("upload input");
        let mask = BoolTensor::from_vec(vec![true, false, true], vec![3])
            .expect("mask")
            .to(Device::Cuda(0))
            .expect("upload mask");
        let grad_fn = MaskedSelectBackward { input, mask };
        let bad_grad = make_cpu_f32(&[1.0], &[1], false)
            .to(Device::Cuda(0))
            .expect("upload bad grad");

        let err = grad_fn
            .backward(&bad_grad)
            .expect_err("CUDA compact grad length must be validated");
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    /// CUDA value-correctness for the direct indexing ops. These tests
    /// originally pinned `NotImplementedOnCuda` for gather/scatter/
    /// scatter_add/where_cond, but CUDA kernels landed later and nobody
    /// ever ran this lane (no CI enabled `--features gpu` until CORE-191 /
    /// crosslink #1885), so the stale pins asserted the opposite of HEAD
    /// behavior. Expected values follow torch semantics: `torch.gather`
    /// selects `input[index[i]]` along `dim`; `torch.Tensor.scatter_`
    /// writes `src[i]` to `input[index[i]]`; `scatter_add_` accumulates;
    /// `torch.where` selects from `x`/`y`.
    #[test]
    fn gpu_indexing_ops_on_cuda() {
        ensure_cuda_backend();
        let cuda_a = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[4], false)
            .to(Device::Cuda(0))
            .expect("upload");

        // gather([1,2,3,4], dim=0, index=[0,1]) -> [1,2]
        let r = gather(&cuda_a, 0, &[0, 1], &[2]).expect("gather on cuda");
        assert_eq!(r.device(), Device::Cuda(0), "gather output device");
        check_f32(
            "gather cuda",
            &read_back_f32(&r),
            &[1.0, 2.0],
            tolerance::F32_BITEXACT,
        );

        // scatter([1,2,3,4], dim=0, index=[0,2], src=[10,20]) -> [10,2,20,4]
        let cuda_src = make_cpu_f32(&[10.0, 20.0], &[2], false)
            .to(Device::Cuda(0))
            .expect("upload");
        let r = scatter(&cuda_a, 0, &[0, 2], &[2], &cuda_src).expect("scatter on cuda");
        assert_eq!(r.device(), Device::Cuda(0), "scatter output device");
        check_f32(
            "scatter cuda",
            &read_back_f32(&r),
            &[10.0, 2.0, 20.0, 4.0],
            tolerance::F32_BITEXACT,
        );

        // scatter_add: same layout, accumulating -> [11,2,23,4]
        let r = scatter_add(&cuda_a, 0, &[0, 2], &[2], &cuda_src).expect("scatter_add on cuda");
        assert_eq!(r.device(), Device::Cuda(0), "scatter_add output device");
        check_f32(
            "scatter_add cuda",
            &read_back_f32(&r),
            &[11.0, 2.0, 23.0, 4.0],
            tolerance::F32_BITEXACT,
        );

        // scatter_value([1,2,3,4], dim=0, index=[0,2], value=9) -> [9,2,9,4]
        let r = scatter_value(&cuda_a, 0, &[0, 2], &[2], 9.0).expect("scatter_value on cuda");
        assert_eq!(r.device(), Device::Cuda(0), "scatter_value output device");
        check_f32(
            "scatter_value cuda",
            &read_back_f32(&r),
            &[9.0, 2.0, 9.0, 4.0],
            tolerance::F32_BITEXACT,
        );

        let cuda_leaf = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[4], false)
            .to(Device::Cuda(0))
            .expect("upload")
            .requires_grad_(true);
        let r = scatter_value(&cuda_leaf, 0, &[0, 2], &[2], 9.0).expect("scatter_value grad cuda");
        let loss = ferrotorch_core::grad_fns::reduction::sum(&r).expect("sum");
        loss.backward().expect("scatter_value backward cuda");
        let grad = cuda_leaf.grad().unwrap().expect("scatter_value cuda grad");
        assert_eq!(grad.device(), Device::Cuda(0), "scatter_value grad device");
        check_f32(
            "scatter_value cuda grad_a",
            &read_back_f32(&grad),
            &[0.0, 1.0, 0.0, 1.0],
            tolerance::F32_BITEXACT,
        );

        // where_cond([T,F,T,F], a, y) -> [1,20,3,40], resident on CUDA.
        let cuda_y = make_cpu_f32(&[10.0, 20.0, 30.0, 40.0], &[4], false)
            .to(Device::Cuda(0))
            .expect("upload");
        let cond = vec![true, false, true, false];
        let r = where_cond(&cond, &cuda_a, &cuda_y).expect("where_cond on cuda");
        assert_eq!(r.device(), Device::Cuda(0), "where_cond output device");
        check_f32(
            "where_cond cuda",
            &read_back_f32(&r),
            &[1.0, 20.0, 3.0, 40.0],
            tolerance::F32_BITEXACT,
        );
    }

    /// Cat A.tensor_ops on GPU — CUDA value-correctness. Originally pinned
    /// `NotImplementedOnCuda` for all five ops; CUDA kernels landed later
    /// and the never-executed lane kept asserting the stale gap (CORE-191 /
    /// crosslink #1885). Expected values follow torch semantics
    /// (`torch.triu`/`tril` zero below/above the diagonal, `torch.diag`
    /// extracts, `torch.diagflat` embeds, `torch.cdist` is pairwise
    /// p-norm distance).
    #[test]
    fn gpu_tensor_ops_on_cuda() {
        ensure_cuda_backend();
        let cuda_2d = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false)
            .to(Device::Cuda(0))
            .expect("upload");

        for (name, r, expected) in [
            ("triu", triu(&cuda_2d, 0), vec![1.0_f64, 2.0, 0.0, 4.0]),
            ("tril", tril(&cuda_2d, 0), vec![1.0, 0.0, 3.0, 4.0]),
            ("diag", diag(&cuda_2d, 0), vec![1.0, 4.0]),
            (
                "diagflat",
                diagflat(&cuda_2d, 0),
                vec![
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 2.0, 0.0, 0.0, //
                    0.0, 0.0, 3.0, 0.0, //
                    0.0, 0.0, 0.0, 4.0,
                ],
            ),
        ] {
            let t = r.unwrap_or_else(|e| panic!("{name} on cuda: {e:?}"));
            assert_eq!(t.device(), Device::Cuda(0), "{name} output device");
            check_f32(
                &format!("{name} cuda"),
                &read_back_f32(&t),
                &expected,
                tolerance::F32_BITEXACT,
            );
        }

        // cdist([[1,2],[3,4]], [[1,1]], p=2) -> [|(0,1)|, |(2,3)|] = [1, sqrt(13)]
        let cuda_x2 = make_cpu_f32(&[1.0, 1.0], &[1, 2], false)
            .to(Device::Cuda(0))
            .expect("upload");
        let t = cdist(&cuda_2d, &cuda_x2, 2.0).expect("cdist on cuda");
        assert_eq!(t.device(), Device::Cuda(0), "cdist output device");
        check_f32(
            "cdist cuda",
            &read_back_f32(&t),
            &[1.0, 13.0_f64.sqrt()],
            tolerance::F32_TRANSCENDENTAL_GPU,
        );
    }

    #[test]
    fn gpu_search_ops_on_cuda() {
        // CUDA value-correctness for the search family. Originally pinned
        // `NotImplementedOnCuda` for all seven ops; CUDA support landed
        // later and the never-executed lane kept asserting the stale gaps
        // (CORE-191 / crosslink #1885). Expected values follow torch
        // semantics (`torch.searchsorted` left insertion, `torch.bucketize`,
        // `torch.unique` sorted + inverse + counts, `torch.histc` equal-width
        // bins over [min, max], `torch.meshgrid(indexing="ij")`,
        // `torch.topk` largest-first).
        ensure_cuda_backend();
        let cuda_a = make_cpu_f32(&[1.0, 2.0, 3.0, 4.0], &[4], false)
            .to(Device::Cuda(0))
            .expect("upload");
        let cuda_b = make_cpu_f32(&[2.0, 3.0], &[2], false)
            .to(Device::Cuda(0))
            .expect("upload");

        // searchsorted([1,2,3,4], [2,3], right=false) -> [1, 2]
        let idx = searchsorted(&cuda_a, &cuda_b, false).expect("searchsorted on cuda");
        assert_eq!(idx, vec![1, 2], "searchsorted cuda");

        // bucketize([2,3], boundaries=[1,2,3,4], right=false) -> [1, 2]
        let idx = bucketize(&cuda_b, &cuda_a, false).expect("bucketize on cuda");
        assert_eq!(idx, vec![1, 2], "bucketize cuda");

        // unique([1,2,3,4]) -> sorted values [1,2,3,4], inverse [0,1,2,3],
        // counts [1,1,1,1]
        let (vals, inverse, counts) = unique(&cuda_a).expect("unique on cuda");
        assert_eq!(vals.device(), Device::Cuda(0), "unique values device");
        check_f32(
            "unique cuda values",
            &read_back_f32(&vals),
            &[1.0, 2.0, 3.0, 4.0],
            tolerance::F32_BITEXACT,
        );
        assert_eq!(inverse, vec![0, 1, 2, 3], "unique cuda inverse");
        assert_eq!(counts, vec![1, 1, 1, 1], "unique cuda counts");

        let (vals, inverse, counts) =
            unique_consecutive(&cuda_a).expect("unique_consecutive on cuda");
        assert_eq!(
            vals.device(),
            Device::Cuda(0),
            "unique_consecutive values device"
        );
        check_f32(
            "unique_consecutive cuda values",
            &read_back_f32(&vals),
            &[1.0, 2.0, 3.0, 4.0],
            tolerance::F32_BITEXACT,
        );
        assert_eq!(inverse, vec![0, 1, 2, 3], "unique_consecutive cuda inverse");
        assert_eq!(counts, vec![1, 1, 1, 1], "unique_consecutive cuda counts");

        // histc([1,2,3,4], bins=4, min=0, max=5): bin width 1.25 -> one
        // sample per bin -> [1,1,1,1]
        let h = histc(&cuda_a, 4, 0.0, 5.0).expect("histc on cuda");
        assert_eq!(h.device(), Device::Cuda(0), "histc output device");
        check_f32(
            "histc cuda",
            &read_back_f32(&h),
            &[1.0, 1.0, 1.0, 1.0],
            tolerance::F32_BITEXACT,
        );

        // meshgrid([1,2,3,4], [2,3]) (ij) -> two [4,2] grids
        let grids = meshgrid(&[cuda_a.clone(), cuda_b.clone()]).expect("meshgrid on cuda");
        assert_eq!(grids.len(), 2, "meshgrid arity");
        assert_eq!(grids[0].device(), Device::Cuda(0), "meshgrid[0] device");
        assert_eq!(grids[1].device(), Device::Cuda(0), "meshgrid[1] device");
        assert_eq!(grids[0].shape(), &[4, 2], "meshgrid[0] shape");
        check_f32(
            "meshgrid cuda [0]",
            &read_back_f32(&grids[0]),
            &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            tolerance::F32_BITEXACT,
        );
        check_f32(
            "meshgrid cuda [1]",
            &read_back_f32(&grids[1]),
            &[2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0],
            tolerance::F32_BITEXACT,
        );

        // topk([1,2,3,4], k=2, largest=true) -> values [4,3], indices [3,2]
        let (vals, idx) = topk(&cuda_a, 2, true).expect("topk on cuda");
        assert_eq!(vals.device(), Device::Cuda(0), "topk values device");
        check_f32(
            "topk cuda values",
            &read_back_f32(&vals),
            &[4.0, 3.0],
            tolerance::F32_BITEXACT,
        );
        assert_eq!(idx, vec![3, 2], "topk cuda indices");
    }

    // Discriminating test for #1097: roll<f32> on CUDA must match the CPU
    // path bit-exact (pure gather; no float drift), and the backward via
    // RollBackward must match the CPU backward bit-exact on CUDA.
    #[test]
    fn gpu_roll_matches_cpu_forward_and_backward() {
        use ferrotorch_core::autograd::no_grad::no_grad;
        ensure_cuda_backend();

        // ---- Forward: a 3-D case that exercises all three of
        // (outer > 1, dim_size > 1, inner > 1) so the kernel's index math
        // is genuinely covered.
        let shape = vec![2usize, 4, 3];
        let n: usize = shape.iter().product();
        let data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 1.0).collect();

        for &(shifts, dim) in &[(1_i64, 1_usize), (-1, 0), (3, 1), (-2, 2), (0, 0), (4, 1)] {
            no_grad(|| {
                let cpu_in = make_cpu_f32(&data, &shape, false);
                let cuda_in = cpu_in.to(Device::Cuda(0)).expect("upload");

                let cpu_out = roll(&cpu_in, shifts, dim).expect("cpu roll");
                let cuda_out = roll(&cuda_in, shifts, dim).expect("gpu roll");

                // The CUDA output must land back on CUDA (no silent
                // CPU-fallback — `rust-gpu-discipline` §3).
                assert!(
                    cuda_out.is_cuda(),
                    "roll on CUDA tensor must produce a CUDA tensor (shifts={shifts}, dim={dim})"
                );

                let cpu_v = read_back_f32(&cpu_out);
                let gpu_v = read_back_f32(&cuda_out);
                assert_eq!(
                    cpu_v.len(),
                    gpu_v.len(),
                    "length mismatch (shifts={shifts}, dim={dim})"
                );
                for (i, (&c, &g)) in cpu_v.iter().zip(gpu_v.iter()).enumerate() {
                    // Pure gather: must be bit-exact in f32.
                    assert_eq!(
                        c, g,
                        "roll mismatch at i={i} (shifts={shifts}, dim={dim}): cpu={c}, gpu={g}"
                    );
                }
            });
        }

        // ---- Backward: drive RollBackward on both CPU and CUDA with a
        // hand-built grad_output and compare the input grads bit-exactly.
        // Using `requires_grad=true` so the forward attaches the grad_fn,
        // then invoking the grad_fn directly with the grad_output on the
        // right device exercises the new GPU backward path.
        let bshape = vec![2usize, 5];
        let bn: usize = bshape.iter().product();
        let bdata: Vec<f64> = (0..bn).map(|i| (i as f64 + 1.0) * 0.125).collect();
        let go_data_f32: Vec<f32> = (0..bn).map(|i| (i as f32 + 1.0) * 3.0).collect();

        for &(shifts, dim) in &[(1_i64, 1_usize), (-2, 1), (1, 0)] {
            // CPU reference
            let cpu_x = make_cpu_f32(&bdata, &bshape, true);
            let cpu_y = roll(&cpu_x, shifts, dim).expect("cpu roll fwd");
            let cpu_go = Tensor::from_storage(
                TensorStorage::cpu(go_data_f32.clone()),
                bshape.clone(),
                false,
            )
            .expect("cpu go");
            let cpu_grad_fn = cpu_y.grad_fn().expect("cpu must carry RollBackward");
            let cpu_grads = cpu_grad_fn.backward(&cpu_go).expect("cpu backward");
            let cpu_g = cpu_grads[0]
                .as_ref()
                .expect("cpu grad must be Some")
                .data()
                .expect("cpu grad data")
                .to_vec();

            // CUDA path
            let cuda_x = make_cpu_f32(&bdata, &bshape, true)
                .to(Device::Cuda(0))
                .expect("upload cuda_x");
            let cuda_y = roll(&cuda_x, shifts, dim).expect("cuda roll fwd");
            assert!(cuda_y.is_cuda());
            let cuda_go = Tensor::from_storage(
                TensorStorage::cpu(go_data_f32.clone()),
                bshape.clone(),
                false,
            )
            .expect("make cpu go")
            .to(Device::Cuda(0))
            .expect("upload cuda go");
            let cuda_grad_fn = cuda_y.grad_fn().expect("cuda must carry RollBackward");
            let cuda_grads = cuda_grad_fn.backward(&cuda_go).expect("cuda backward");
            let cuda_grad = cuda_grads[0].as_ref().expect("cuda grad must be Some");
            assert!(
                cuda_grad.is_cuda(),
                "RollBackward on CUDA grad_output must produce a CUDA grad (shifts={shifts}, dim={dim})"
            );
            let cuda_g = read_back_f32(cuda_grad);

            assert_eq!(cpu_g.len(), cuda_g.len());
            for (i, (&c, &g)) in cpu_g.iter().zip(cuda_g.iter()).enumerate() {
                assert_eq!(
                    c, g,
                    "RollBackward mismatch at i={i} (shifts={shifts}, dim={dim}): cpu={c}, gpu={g}"
                );
            }
        }
    }
}
