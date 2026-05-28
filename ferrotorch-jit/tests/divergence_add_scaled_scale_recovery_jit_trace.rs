//! Independent discriminator audit of commit `1df19cf69` (#1633): the JIT
//! tracer recovers `Sub`/`Add` from the `AddScaledBackward` scale.
//!
//! The builder's in-module tests cover `trace(sub) -> Sub`, `trace(add) -> Add`
//! and `trace(add_scaled, alpha=2.0)` fail-fast. This file audits the edges the
//! builder did NOT directly exercise through the *public* `trace` API:
//!
//!   1. The `sub_scaled(a, b, 2.0)` path. `sub` delegates to
//!      `add_scaled(a, b, -alpha)`, so `sub_scaled(a, b, 2.0)` saves a *negative*
//!      scale of `-2.0`. The mapper must FAIL FAST on `-2.0`, NOT map every
//!      negative scale to `Sub`. This is the highest-value guard: a naive
//!      `scale < 0.0 -> Sub` would silently mis-route `a - 2*b` to a plain
//!      un-scaled `a - b`, diverging from torch which records one `aten::sub`
//!      with `alpha=2`.
//!   2. A multi-op graph `mul(sub(a, b), c)` traces END-TO-END to a graph
//!      containing BOTH `Sub` and `Mul` (the builder only smoke-claimed this).
//!   3. The fail-fast surfaces as `Err`, never a panic, and the error names the
//!      scale (observability).
//!
//! ## Torch contract
//! `aten/src/ATen/native/BinaryOps.cpp:434-439` — `TORCH_IMPL_FUNC(sub_out)`
//! delegates to `add_stub(device_type(), *this, -alpha)`. The C++ delegation is
//! implementation-internal; `torch.jit.trace(a - 2*b)` records a single
//! `aten::sub` node carrying `alpha=2`, NOT an un-scaled `aten::sub`. ferrotorch
//! cannot represent a scaled sub/add edge in its IR today, so the faithful
//! behaviour is to fail fast (refuse to emit a wrong node), which these tests
//! pin.

#![allow(clippy::missing_panics_doc)]

use ferrotorch_core::error::FerrotorchResult;
use ferrotorch_core::grad_fns::arithmetic::{mul, rsub, sub, sub_scaled};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_jit::graph::IrOpKind;
use ferrotorch_jit::trace::trace;

fn grad_vec_f32(data: Vec<f32>) -> Tensor<f32> {
    let n = data.len();
    Tensor::from_storage(TensorStorage::cpu(data), vec![n], true)
        .unwrap()
        .requires_grad_(true)
}

/// GUARD: `sub_scaled(a, b, 2.0)` delegates to `add_scaled(a, b, -2.0)`, saving
/// scale `-2.0`. The tracer MUST fail fast — it must NOT collapse a negative
/// non-unit scale into a plain `IrOpKind::Sub` (that would diverge from torch's
/// `aten::sub` with `alpha=2`). This pins that the mapper keys on the EXACT
/// `-1.0` point, not the sign of the scale.
#[test]
fn trace_sub_scaled_non_unit_alpha_fails_fast_not_silent_sub() {
    let a = grad_vec_f32(vec![1.0, 2.0, 3.0]);
    let b = grad_vec_f32(vec![4.0, 5.0, 6.0]);

    let result = trace(
        |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
            sub_scaled(&inputs[0], &inputs[1], 2.0)
        },
        &[a, b],
    );

    // Must be an error (no scaled-sub IR edge), NOT Ok(graph-with-bare-Sub).
    let err = result.expect_err(
        "trace(sub_scaled, alpha=2.0) -> scale -2.0 must fail fast, not silently \
         produce a bare Sub (torch records aten::sub with alpha=2)",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("AddScaledBackward"),
        "error must name the op; got: {msg}"
    );
    assert!(
        msg.contains('2'),
        "error must name the unrepresentable scale -2.0; got: {msg}"
    );
}

/// GUARD: a multi-op graph with `sub` feeding `mul` traces END-TO-END (no graph
/// break, no error) to a graph containing BOTH `IrOpKind::Sub` and
/// `IrOpKind::Mul`. Before #1633 this errored because `AddScaledBackward` was
/// unmapped; the builder claimed this is now unblocked but only smoke-tested it.
#[test]
fn trace_mul_of_sub_contains_both_sub_and_mul() {
    let a = grad_vec_f32(vec![5.0, 7.0, 9.0]);
    let b = grad_vec_f32(vec![1.0, 2.0, 3.0]);
    let c = grad_vec_f32(vec![2.0, 2.0, 2.0]);

    let graph = trace(
        |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
            let s = sub(&inputs[0], &inputs[1])?;
            mul(&s, &inputs[2])
        },
        &[a, b, c],
    )
    .expect("trace(mul(sub(a, b), c)) must trace end-to-end after #1633");

    assert!(
        graph.nodes.iter().any(|n| matches!(n.op, IrOpKind::Sub)),
        "graph must contain IrOpKind::Sub; found {:?}",
        graph.nodes.iter().map(|n| &n.op).collect::<Vec<_>>()
    );
    assert!(
        graph.nodes.iter().any(|n| matches!(n.op, IrOpKind::Mul)),
        "graph must contain IrOpKind::Mul; found {:?}",
        graph.nodes.iter().map(|n| &n.op).collect::<Vec<_>>()
    );
    // The delegation must not leak as an Add.
    assert!(
        !graph.nodes.iter().any(|n| matches!(n.op, IrOpKind::Add)),
        "the add_scaled(-1.0) delegation must NOT surface as IrOpKind::Add"
    );
}

/// `rsub(a, b, 1.0)` == `sub_scaled(b, a, 1.0)` == `add_scaled(b, a, -1.0)`:
/// scale -1.0 -> the tracer recovers `IrOpKind::Sub`. rsub with default alpha
/// must trace cleanly (the builder listed rsub among the unblocked callers).
#[test]
fn trace_rsub_default_alpha_recovers_sub() {
    let a = grad_vec_f32(vec![1.0, 2.0, 3.0]);
    let b = grad_vec_f32(vec![10.0, 20.0, 30.0]);

    let graph = trace(
        |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
            rsub(&inputs[0], &inputs[1], 1.0)
        },
        &[a, b],
    )
    .expect("trace(rsub, alpha=1.0) must recover IrOpKind::Sub");

    assert!(
        graph.nodes.iter().any(|n| matches!(n.op, IrOpKind::Sub)),
        "trace(rsub) must contain IrOpKind::Sub; found {:?}",
        graph.nodes.iter().map(|n| &n.op).collect::<Vec<_>>()
    );
}

/// `rsub(a, b, 2.0)` == `add_scaled(b, a, -2.0)`: scale -2.0 must fail fast,
/// same negative-non-unit guard as `sub_scaled`.
#[test]
fn trace_rsub_non_unit_alpha_fails_fast() {
    let a = grad_vec_f32(vec![1.0, 2.0, 3.0]);
    let b = grad_vec_f32(vec![10.0, 20.0, 30.0]);

    let result = trace(
        |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
            rsub(&inputs[0], &inputs[1], 2.0)
        },
        &[a, b],
    );
    let err = result.expect_err("trace(rsub, alpha=2.0) -> scale -2.0 must fail fast");
    assert!(
        format!("{err}").contains("AddScaledBackward"),
        "error must name the op; got: {err}"
    );
}
