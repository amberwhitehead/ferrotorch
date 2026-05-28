//! Post-#1192-chain divergence: `arithmetic::sub` no longer produces a
//! `SubBackward` grad_fn — it now delegates to `add_scaled(a, b, -1.0)` which
//! emits `AddScaledBackward`. The JIT tracer's `map_name_to_op` table at
//! `ferrotorch-jit/src/trace.rs:30-103` maps `"AddBackward"` and
//! `"SubBackward"` to `IrOpKind::Add` / `IrOpKind::Sub`, but has NO entry for
//! `"AddScaledBackward"`. The `KNOWN_OPS` table at
//! `ferrotorch-jit/src/graph_break.rs:36-72` likewise lacks
//! `"AddScaledBackward"`.
//!
//! Consequence: any user code that calls `arithmetic::sub` (or any of its
//! transitive callers — `Tensor::sub_t`, `dual_sub`, `grad_penalty::sub`,
//! `nn::functional::*`, `nn::activation::mish`, `vision::ops::*`, etc.) and
//! then attempts to JIT-trace the result hits
//! `Err(InvalidArgument { message: "unsupported operation in tracer:
//! AddScaledBackward" })`. Before commit `d0fd83f1a`, the same code traced
//! cleanly to `IrOpKind::Sub`.
//!
//! ## Why this is a divergence vs PyTorch
//!
//! PyTorch's `torch.jit.trace` always produces `aten::sub` for `a - b`,
//! independent of the underlying C++ delegation through `add_stub` — the
//! C++-side delegation is implementation-detail-only. Upstream cite:
//! `pytorch/aten/src/ATen/native/BinaryOps.cpp:434-439` shows the C++ kernel
//! delegates internally, but Python-level trace inspects `node.kind() ==
//! "aten::sub"`. Our ferrotorch trace inspects `GradFn::name()` instead,
//! which now leaks the delegation as `"AddScaledBackward"`.
//!
//! Mirror op for PyTorch: `torch.jit.trace(lambda a, b: a.sub(b), ...)`
//! produces a graph containing `aten::sub`. Verify via:
//! ```python
//! import torch
//! def f(a, b): return a - b
//! traced = torch.jit.trace(f, (torch.rand(3, requires_grad=True),
//!                              torch.rand(3, requires_grad=True)))
//! print(traced.graph)  # contains `aten::sub`, NOT `aten::add` with -1 alpha
//! ```
//!
//! Tracking: filed via crosslink (see report).

#![allow(clippy::missing_panics_doc)]

use ferrotorch_core::error::FerrotorchResult;
use ferrotorch_core::grad_fns::arithmetic::{add, sub};
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

/// Baseline: `trace(|x, y| x + y)` still works (sanity).
#[test]
fn baseline_trace_add_still_works() {
    let x = grad_vec_f32(vec![1.0, 2.0, 3.0]);
    let y = grad_vec_f32(vec![4.0, 5.0, 6.0]);
    let graph = trace(
        |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> { add(&inputs[0], &inputs[1]) },
        &[x, y],
    )
    .expect("trace(add) must succeed — baseline");
    assert!(
        graph.nodes.iter().any(|n| matches!(n.op, IrOpKind::Add)),
        "graph must contain IrOpKind::Add"
    );
}

/// Regression (#1633, FIXED): `trace(|x, y| x - y)` produces an
/// `IrOpKind::Sub` node, matching PyTorch's `aten::sub` in the traced graph.
/// After the #1192-chain delegation, `sub` emits `AddScaledBackward { alpha:
/// -1.0 }`; the tracer's `map_name_to_op` now reads that scale via
/// `GradFn::scalar_args()` and recovers `IrOpKind::Sub` (scale -1.0) /
/// `IrOpKind::Add` (scale 1.0). Upstream parity contract:
/// `pytorch/aten/src/ATen/native/BinaryOps.cpp:434-439` (C++ delegation is
/// implementation-internal; the user-visible op-name remains `aten::sub`).
#[test]
fn divergence_trace_sub_produces_addscaled_not_sub() {
    let x = grad_vec_f32(vec![1.0, 2.0, 3.0]);
    let y = grad_vec_f32(vec![4.0, 5.0, 6.0]);
    let result = trace(
        |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> { sub(&inputs[0], &inputs[1]) },
        &[x, y],
    );

    // What PyTorch users expect: trace(a - b) produces a Sub node, by parity
    // with torch.jit.trace(lambda a, b: a - b) → aten::sub.
    let graph = result.expect(
        "trace(sub) must succeed and produce an IrOpKind::Sub node, mirroring PyTorch's \
         torch.jit.trace which yields aten::sub regardless of the C++ delegation through \
         add_stub at BinaryOps.cpp:434-439",
    );

    assert!(
        graph.nodes.iter().any(|n| matches!(n.op, IrOpKind::Sub)),
        "graph must contain IrOpKind::Sub for trace(a - b); found ops: {:?}",
        graph
            .nodes
            .iter()
            .map(|n| format!("{:?}", n.op))
            .collect::<Vec<_>>()
    );
}
