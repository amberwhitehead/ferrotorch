#!/usr/bin/env python3
"""
Regenerate reference fixtures for the ferrotorch-jit C7.2 interpreter
conformance suite.

Tracking issue: #857 (Conformance Buildout C7.2 — interpreter + module +
aot_autograd + graph_break).

Output: ``ferrotorch-jit/tests/conformance/fixtures.json``

Pin: torch == 2.11.0

Background
----------
ferrotorch-jit's interpreter executes an IrGraph by walking the graph in
topological order and dispatching each IrOpKind to the corresponding
ferrotorch-core operation. The ``TracedModule`` wraps an optimized IrGraph
and exposes it via the ``Module`` trait. The AOT autograd module decomposes
a traced forward graph into separate forward and backward IR graphs.
``trace_with_breaks`` splits the trace at unsupported ops and wraps compiled
IR subgraphs + eager fallback closures into a ``SegmentedModule``.

The parity contract:
  "The interpreter executing an IrGraph built from equivalent PyTorch
   arithmetic produces the same numeric output as that PyTorch expression."

This script records ``(input_values, expected_output)`` pairs by running the
equivalent PyTorch expressions directly.  The ferrotorch Rust tests load these
pairs, build equivalent IrGraphs, run the interpreter or TracedModule, and
assert the output matches ``expected_output`` within tolerance.

Cases that test error conditions or structural contracts (not numeric output)
are recorded as ``cascade_skip`` with ``"spec-only marker, no PyTorch reference"``
so the Rust tests can skip them gracefully.

Usage
-----
    python3 scripts/regenerate_jit_interpreter_fixtures.py

Required Python deps:
    torch==2.11.0   (CPU-only build is sufficient)
    numpy

The script exits 0 on success and writes
``ferrotorch-jit/tests/conformance/fixtures.json`` in the repo root.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pathlib
import platform
import sys

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:
    print(
        "ERROR: 'torch' is not installed. Install with:\n"
        "    pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu",
        file=sys.stderr,
    )
    sys.exit(1)

REQUIRED_TORCH = "2.11.0"
actual = torch.__version__
if not actual.startswith(REQUIRED_TORCH):
    print(
        f"WARNING: torch version is {actual!r}, expected {REQUIRED_TORCH!r}. "
        "Fixtures may drift if versions differ.",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPEC_ONLY_SKIP = "spec-only marker, no PyTorch reference"


def _t(values: list[float], dtype=torch.float32) -> "torch.Tensor":
    return torch.tensor(values, dtype=dtype)


def _t2d(values: list[list[float]], dtype=torch.float32) -> "torch.Tensor":
    return torch.tensor(values, dtype=dtype)


def _tolist(t: "torch.Tensor") -> "list[float]":
    return t.detach().tolist()


def _tolist_flat(t: "torch.Tensor") -> "list[float]":
    return t.detach().reshape(-1).tolist()


# ---------------------------------------------------------------------------
# --- Layer 2 fixture generation ---
# ---------------------------------------------------------------------------


def build_fixtures() -> "list[dict]":  # noqa: C901 — long by design (fixture catalogue)
    fixtures: list[dict] = []

    # ======================================================================
    # MODULE 1: interpreter.rs — IrGraph execution
    # ======================================================================

    # ------------------------------------------------------------------
    # interp_add_self
    # Graph: y = x + x
    # ------------------------------------------------------------------
    x_add = _t([1.0, 2.0, 3.0])
    expected_add = x_add + x_add
    fixtures.append({
        "case": "interp_add_self",
        "module": "interpreter",
        "op": "interpret",
        "description": (
            "Interpreter: y = x + x.  "
            "Single input, Add node, shape [3]. "
            "Validates basic IrOpKind::Add dispatch."
        ),
        "input": _tolist(x_add),
        "input_shape": list(x_add.shape),
        "dtype": "float32",
        "expected_output": _tolist(expected_add),
        "expected_shape": list(expected_add.shape),
        "tol": 1e-6,
        "torch_reference": f"x={_tolist(x_add)}; (x+x).tolist()=={_tolist(expected_add)}",
    })

    # ------------------------------------------------------------------
    # interp_matmul_2x3_3x2
    # Graph: C = mm(A, B)  where A=[2,3], B=[3,2]
    # ------------------------------------------------------------------
    A = _t2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B = _t2d([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    C = torch.mm(A, B)
    fixtures.append({
        "case": "interp_matmul_2x3_3x2",
        "module": "interpreter",
        "op": "interpret_mm",
        "description": (
            "Interpreter: C = mm(A, B) with A=[2,3], B=[3,2]. "
            "Validates IrOpKind::Mm dispatch and 2-D matrix multiplication."
        ),
        "input_a": _tolist_flat(A),
        "input_b": _tolist_flat(B),
        "shape_a": list(A.shape),
        "shape_b": list(B.shape),
        "dtype": "float32",
        "expected_output": _tolist_flat(C),
        "expected_shape": list(C.shape),
        "tol": 1e-5,
        "torch_reference": (
            f"A={_tolist_flat(A)} shape={list(A.shape)}; "
            f"B={_tolist_flat(B)} shape={list(B.shape)}; "
            f"torch.mm(A,B).reshape(-1).tolist()=={_tolist_flat(C)}"
        ),
    })

    # ------------------------------------------------------------------
    # interp_constant_add
    # Graph: y = x + constant([10, 20, 30])
    # ------------------------------------------------------------------
    x_cadd = _t([1.0, 2.0, 3.0])
    c_cadd = _t([10.0, 20.0, 30.0])
    expected_cadd = x_cadd + c_cadd
    fixtures.append({
        "case": "interp_constant_add",
        "module": "interpreter",
        "op": "interpret_constant",
        "description": (
            "Interpreter: y = x + constant([10,20,30]). "
            "Validates IrOpKind::Constant embedding and Add dispatch."
        ),
        "input": _tolist(x_cadd),
        "constant": _tolist(c_cadd),
        "input_shape": list(x_cadd.shape),
        "dtype": "float32",
        "expected_output": _tolist(expected_cadd),
        "expected_shape": list(expected_cadd.shape),
        "tol": 1e-6,
        "torch_reference": (
            f"x={_tolist(x_cadd)}; c={_tolist(c_cadd)}; "
            f"(x+c).tolist()=={_tolist(expected_cadd)}"
        ),
    })

    # ------------------------------------------------------------------
    # interp_chain_sub_pow_sqrt
    # Graph: y = sqrt(pow(x - c, 2)) == |x - c|
    # ------------------------------------------------------------------
    x_chain = _t([4.0, 1.0, 5.0])
    c_chain = _t([1.0, 1.0, 1.0])
    expected_chain = torch.sqrt(torch.pow(x_chain - c_chain, 2.0))
    fixtures.append({
        "case": "interp_chain_sub_pow_sqrt",
        "module": "interpreter",
        "op": "interpret_chain",
        "description": (
            "Interpreter: y = sqrt((x-c)^2) with x=[4,1,5], c=[1,1,1]. "
            "Validates a 3-node chain (Sub→Pow→Sqrt) and IrOpKind::Pow exponent."
        ),
        "input": _tolist(x_chain),
        "constant": _tolist(c_chain),
        "input_shape": list(x_chain.shape),
        "dtype": "float32",
        "expected_output": _tolist(expected_chain),
        "expected_shape": list(expected_chain.shape),
        "tol": 1e-5,
        "torch_reference": (
            f"x={_tolist(x_chain)}; c={_tolist(c_chain)}; "
            f"sqrt((x-c)**2).tolist()=={_tolist(expected_chain)}"
        ),
    })

    # ------------------------------------------------------------------
    # interp_fused_elementwise_neg_relu
    # Graph: y = relu(neg(x))  — FusedElementwise{ops: [Neg, Relu]}
    # ------------------------------------------------------------------
    x_fused = _t([-1.0, 2.0, -3.0, 4.0])
    expected_fused = torch.relu(-x_fused)
    fixtures.append({
        "case": "interp_fused_elementwise_neg_relu",
        "module": "interpreter",
        "op": "interpret_fused_elementwise",
        "description": (
            "Interpreter: FusedElementwise{ops:[Neg,Relu]}. "
            "Input [-1,2,-3,4]; after neg=[1,-2,3,-4]; after relu=[1,0,3,0]. "
            "Validates apply_elementwise_op chain dispatch."
        ),
        "input": _tolist(x_fused),
        "input_shape": list(x_fused.shape),
        "dtype": "float32",
        "expected_output": _tolist(expected_fused),
        "expected_shape": list(expected_fused.shape),
        "tol": 1e-6,
        "torch_reference": (
            f"x={_tolist(x_fused)}; relu(-x).tolist()=={_tolist(expected_fused)}"
        ),
    })

    # ------------------------------------------------------------------
    # interp_multi_output
    # Graph: [y0, y1] = [x+x, x*x]
    # ------------------------------------------------------------------
    x_multi = _t([1.0, 2.0, 3.0])
    y0 = x_multi + x_multi
    y1 = x_multi * x_multi
    fixtures.append({
        "case": "interp_multi_output",
        "module": "interpreter",
        "op": "interpret_multi",
        "description": (
            "interpret_multi: [y0,y1] = [x+x, x*x]. "
            "Validates multi-output graph execution and CL-368 multi-output path."
        ),
        "input": _tolist(x_multi),
        "input_shape": list(x_multi.shape),
        "dtype": "float32",
        "expected_output_0": _tolist(y0),
        "expected_output_1": _tolist(y1),
        "expected_shape": list(x_multi.shape),
        "tol": 1e-6,
        "torch_reference": (
            f"x={_tolist(x_multi)}; y0=(x+x)={_tolist(y0)}; y1=(x*x)={_tolist(y1)}"
        ),
    })

    # ------------------------------------------------------------------
    # interp_activation_softmax
    # Graph: y = softmax(x)
    # ------------------------------------------------------------------
    x_sm = _t([1.0, 2.0, 3.0, 4.0])
    expected_sm = torch.softmax(x_sm, dim=0)
    fixtures.append({
        "case": "interp_activation_softmax",
        "module": "interpreter",
        "op": "interpret_softmax",
        "description": (
            "Interpreter: y = softmax(x) with x=[1,2,3,4]. "
            "Validates IrOpKind::Softmax dispatch via activation::softmax."
        ),
        "input": _tolist(x_sm),
        "input_shape": list(x_sm.shape),
        "dtype": "float32",
        "expected_output": _tolist(expected_sm),
        "expected_shape": list(expected_sm.shape),
        "tol": 1e-5,
        "torch_reference": (
            f"x={_tolist(x_sm)}; torch.softmax(x,dim=0).tolist()=={_tolist(expected_sm)}"
        ),
    })

    # ------------------------------------------------------------------
    # interp_reduction_sum
    # Graph: y = sum(x)
    # ------------------------------------------------------------------
    x_sum = _t([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_sum = x_sum.sum().unsqueeze(0)
    fixtures.append({
        "case": "interp_reduction_sum",
        "module": "interpreter",
        "op": "interpret_sum",
        "description": (
            "Interpreter: y = sum(x) with x=[1,2,3,4,5]; result=15. "
            "Validates IrOpKind::Sum reduction dispatch."
        ),
        "input": _tolist(x_sum),
        "input_shape": list(x_sum.shape),
        "dtype": "float32",
        "expected_output": _tolist(expected_sum),
        "expected_shape": [1],
        "tol": 1e-5,
        "torch_reference": (
            f"x={_tolist(x_sum)}; x.sum().item()=={x_sum.sum().item()}"
        ),
    })

    # ======================================================================
    # MODULE 2: module.rs — TracedModule / compile()
    # ======================================================================

    # ------------------------------------------------------------------
    # module_traced_forward
    # compile(sum(a*b)) → forward_multi([a,b])
    # ------------------------------------------------------------------
    a_tm = _t([1.0, 2.0, 3.0])
    b_tm = _t([4.0, 5.0, 6.0])
    expected_tm = (a_tm * b_tm).sum().unsqueeze(0)
    fixtures.append({
        "case": "module_traced_forward",
        "module": "module",
        "op": "compile",
        "description": (
            "TracedModule: compile(sum(a*b)) and forward_multi([a,b]). "
            "Validates the full compile→TracedModule→forward_multi pipeline."
        ),
        "input_a": _tolist(a_tm),
        "input_b": _tolist(b_tm),
        "shape_a": list(a_tm.shape),
        "shape_b": list(b_tm.shape),
        "dtype": "float32",
        "expected_output": _tolist(expected_tm),
        "expected_shape": [1],
        "tol": 1e-5,
        "torch_reference": (
            f"a={_tolist(a_tm)}; b={_tolist(b_tm)}; "
            f"(a*b).sum().item()=={(a_tm*b_tm).sum().item()}"
        ),
    })

    # ------------------------------------------------------------------
    # module_forward_single_input
    # compile(sum(x)) → forward(x)
    # ------------------------------------------------------------------
    x_fwd = _t([10.0, 20.0, 30.0])
    expected_fwd = x_fwd.sum().unsqueeze(0)
    fixtures.append({
        "case": "module_forward_single_input",
        "module": "module",
        "op": "compile_single",
        "description": (
            "TracedModule: compile(sum(x)) and forward(x). "
            "Validates Module::forward for a single-input traced module."
        ),
        "input": _tolist(x_fwd),
        "input_shape": list(x_fwd.shape),
        "dtype": "float32",
        "expected_output": _tolist(expected_fwd),
        "expected_shape": [1],
        "tol": 1e-5,
        "torch_reference": (
            f"x={_tolist(x_fwd)}; x.sum().item()=={x_fwd.sum().item()}"
        ),
    })

    # ------------------------------------------------------------------
    # module_reuse_different_inputs
    # compile(sum(a*b)); call with (a1,b1) then (a2,b2)
    # ------------------------------------------------------------------
    a2 = _t([3.0, 4.0])
    b2_v = _t([5.0, 6.0])
    expected2 = (a2 * b2_v).sum().unsqueeze(0)
    fixtures.append({
        "case": "module_reuse_different_inputs",
        "module": "module",
        "op": "compile_reuse",
        "description": (
            "TracedModule reuse: same module, different inputs. "
            "compile(sum(a*b)); call with a=[3,4] b=[5,6] → sum([15,24])=39."
        ),
        "input_a": _tolist(a2),
        "input_b": _tolist(b2_v),
        "shape_a": list(a2.shape),
        "shape_b": list(b2_v.shape),
        "dtype": "float32",
        "expected_output": _tolist(expected2),
        "expected_shape": [1],
        "tol": 1e-5,
        "torch_reference": (
            f"a={_tolist(a2)}; b={_tolist(b2_v)}; "
            f"(a*b).sum().item()=={(a2*b2_v).sum().item()}"
        ),
    })

    # ------------------------------------------------------------------
    # module_save_load_bytes
    # compile → to_bytes → from_bytes → forward_multi
    # ------------------------------------------------------------------
    a_sl = _t([1.0, 2.0, 3.0])
    b_sl = _t([4.0, 5.0, 6.0])
    expected_sl = (a_sl * b_sl).sum().unsqueeze(0)
    fixtures.append({
        "case": "module_save_load_bytes",
        "module": "module",
        "op": "to_bytes_from_bytes",
        "description": (
            "TracedModule to_bytes / from_bytes roundtrip. "
            "compile(sum(a*b)); serialize and deserialize; check result is identical."
        ),
        "input_a": _tolist(a_sl),
        "input_b": _tolist(b_sl),
        "shape_a": list(a_sl.shape),
        "shape_b": list(b_sl.shape),
        "dtype": "float32",
        "expected_output": _tolist(expected_sl),
        "expected_shape": [1],
        "tol": 1e-5,
        "torch_reference": (
            f"a={_tolist(a_sl)}; b={_tolist(b_sl)}; "
            f"(a*b).sum().item()=={(a_sl*b_sl).sum().item()}"
        ),
    })

    # ======================================================================
    # MODULE 3: aot_autograd.rs — decompose_forward_backward / compile_aot
    # ======================================================================

    # ------------------------------------------------------------------
    # aot_backward_add
    # Graph: y = a + b; backward produces grad_a=ones, grad_b=ones
    # PyTorch reference: torch.autograd.grad on a+b w.r.t. [a,b]
    # ------------------------------------------------------------------
    a_grad = _t([1.0, 2.0, 3.0], dtype=torch.float32)
    b_grad = _t([4.0, 5.0, 6.0], dtype=torch.float32)
    a_grad.requires_grad_(True)
    b_grad.requires_grad_(True)
    y_grad = a_grad + b_grad
    grad_out = _t([1.0, 1.0, 1.0])
    ga, gb = torch.autograd.grad(
        [y_grad], [a_grad, b_grad], grad_outputs=[grad_out], retain_graph=True
    )
    fixtures.append({
        "case": "aot_backward_add",
        "module": "aot_autograd",
        "op": "decompose_add_backward",
        "description": (
            "AOT autograd: y = a + b; backward. "
            "For Add, grad_a = grad_out and grad_b = grad_out (identity pass-through). "
            "Input shape [3]; grad_out = ones(3); expect grad_a = [1,1,1], grad_b = [1,1,1]."
        ),
        "input_a": [1.0, 2.0, 3.0],
        "input_b": [4.0, 5.0, 6.0],
        "input_shape": [3],
        "grad_out": [1.0, 1.0, 1.0],
        "dtype": "float32",
        "expected_grad_a": _tolist(ga),
        "expected_grad_b": _tolist(gb),
        "tol": 1e-5,
        "torch_reference": (
            f"a,b grad=[1,1,1]; "
            f"grad_a={_tolist(ga)}, grad_b={_tolist(gb)}"
        ),
    })

    # ------------------------------------------------------------------
    # aot_backward_mul
    # Graph: y = a * b; backward: grad_a = b * grad_out, grad_b = a * grad_out
    # ------------------------------------------------------------------
    a_mul = _t([2.0, 3.0, 4.0], dtype=torch.float32)
    b_mul = _t([5.0, 6.0, 7.0], dtype=torch.float32)
    a_mul.requires_grad_(True)
    b_mul.requires_grad_(True)
    y_mul = a_mul * b_mul
    grad_out_mul = _t([1.0, 1.0, 1.0])
    ga_mul, gb_mul = torch.autograd.grad(
        [y_mul], [a_mul, b_mul], grad_outputs=[grad_out_mul], retain_graph=True
    )
    fixtures.append({
        "case": "aot_backward_mul",
        "module": "aot_autograd",
        "op": "decompose_mul_backward",
        "description": (
            "AOT autograd: y = a * b; backward. "
            "grad_a = b * grad_out, grad_b = a * grad_out. "
            "Validates that Mul backward emits two Mul nodes using saved tensors."
        ),
        "input_a": [2.0, 3.0, 4.0],
        "input_b": [5.0, 6.0, 7.0],
        "input_shape": [3],
        "grad_out": [1.0, 1.0, 1.0],
        "dtype": "float32",
        "expected_grad_a": _tolist(ga_mul),
        "expected_grad_b": _tolist(gb_mul),
        "tol": 1e-5,
        "torch_reference": (
            f"a=[2,3,4], b=[5,6,7], grad_out=[1,1,1]; "
            f"grad_a={_tolist(ga_mul)}, grad_b={_tolist(gb_mul)}"
        ),
    })

    # ------------------------------------------------------------------
    # aot_backward_sum
    # Graph: y = sum(a); backward: grad_a = ones_like(a) * grad_out
    # ------------------------------------------------------------------
    a_sum2 = _t([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    a_sum2.requires_grad_(True)
    y_sum2 = a_sum2.sum()
    grad_out_sum = torch.tensor(1.0)
    (ga_sum,) = torch.autograd.grad(
        [y_sum2], [a_sum2], grad_outputs=[grad_out_sum], retain_graph=True
    )
    fixtures.append({
        "case": "aot_backward_sum",
        "module": "aot_autograd",
        "op": "decompose_sum_backward",
        "description": (
            "AOT autograd: y = sum(a); backward. "
            "grad_a = ones(4) * scalar_grad_out. "
            "Validates Sum backward emits constant-ones tensor and Mul."
        ),
        "input_a": [1.0, 2.0, 3.0, 4.0],
        "input_shape": [4],
        "grad_out": [1.0],
        "dtype": "float32",
        "expected_grad_a": _tolist(ga_sum),
        "tol": 1e-5,
        "torch_reference": (
            f"a=[1,2,3,4], grad_out=1.0; "
            f"grad_a={_tolist(ga_sum)}"
        ),
    })

    # ------------------------------------------------------------------
    # aot_graph_pair_structure
    # Spec-only: decompose_forward_backward produces AotGraphPair with
    # forward/backward graphs; no numeric reference for graph structure.
    # ------------------------------------------------------------------
    fixtures.append({
        "case": "aot_graph_pair_structure",
        "module": "aot_autograd",
        "op": "decompose_structure",
        "description": (
            "AOT autograd structural check: decompose_forward_backward(Add graph) "
            "must produce an AotGraphPair whose backward has output_values.len()==2 "
            "and whose saved_tensor_indices is non-empty. "
            "Spec-only — no numeric PyTorch reference for graph structure."
        ),
        "cascade_skip": SPEC_ONLY_SKIP,
    })

    # ======================================================================
    # MODULE 4: graph_break.rs — trace_with_breaks / SegmentedModule
    # ======================================================================

    # ------------------------------------------------------------------
    # graph_break_unbroken_add_sum
    # All-supported graph → TraceResult::Unbroken
    # ------------------------------------------------------------------
    x_gb = _t([1.0, 2.0, 3.0])
    expected_gb = ((x_gb + x_gb)).sum().unsqueeze(0)
    fixtures.append({
        "case": "graph_break_unbroken_add_sum",
        "module": "graph_break",
        "op": "trace_with_breaks_unbroken",
        "description": (
            "trace_with_breaks: all-supported ops (Add + Sum) produce "
            "TraceResult::Unbroken (no graph breaks). "
            "Validates that fully traceable functions don't produce segments."
        ),
        "input": _tolist(x_gb),
        "input_shape": list(x_gb.shape),
        "dtype": "float32",
        "expected_output": _tolist(expected_gb),
        "expected_shape": [1],
        "tol": 1e-5,
        "torch_reference": (
            f"x={_tolist(x_gb)}; (x+x).sum().item()=={(x_gb+x_gb).sum().item()}"
        ),
    })

    # ------------------------------------------------------------------
    # graph_break_segmented_module_forward
    # SegmentedModule (compiled segment + eager segment) produces correct output
    # Compiled: y = x + x; Eager: y = y * 10
    # ------------------------------------------------------------------
    x_seg = _t([1.0, 2.0, 3.0])
    after_compiled = x_seg + x_seg       # [2, 4, 6]
    after_eager = after_compiled * 10.0  # [20, 40, 60]
    fixtures.append({
        "case": "graph_break_segmented_module_forward",
        "module": "graph_break",
        "op": "segmented_module_forward",
        "description": (
            "SegmentedModule forward: segment1 (compiled, x+x) → "
            "segment2 (eager, *10). "
            "Input [1,2,3] → after-compiled [2,4,6] → after-eager [20,40,60]."
        ),
        "input": _tolist(x_seg),
        "input_shape": list(x_seg.shape),
        "dtype": "float32",
        "intermediate": _tolist(after_compiled),
        "expected_output": _tolist(after_eager),
        "expected_shape": list(after_eager.shape),
        "tol": 1e-5,
        "torch_reference": (
            f"x={_tolist(x_seg)}; relu(-x).not-applicable; "
            f"(x+x)*10={_tolist(after_eager)}"
        ),
    })

    # ------------------------------------------------------------------
    # graph_break_fullgraph_supported_ops
    # fullgraph=true + all supported → succeeds (no error)
    # ------------------------------------------------------------------
    x_fg = _t([2.0, 3.0])
    expected_fg = (x_fg + x_fg).sum().unsqueeze(0)
    fixtures.append({
        "case": "graph_break_fullgraph_supported_ops",
        "module": "graph_break",
        "op": "trace_with_breaks_fullgraph",
        "description": (
            "trace_with_breaks with fullgraph=true on an all-supported function "
            "must succeed (not error). Validates that fullgraph mode only errors "
            "on actual breaks, not on supported ops."
        ),
        "input": _tolist(x_fg),
        "input_shape": list(x_fg.shape),
        "dtype": "float32",
        "expected_output": _tolist(expected_fg),
        "expected_shape": [1],
        "tol": 1e-5,
        "torch_reference": (
            f"x={_tolist(x_fg)}; (x+x).sum().item()=={(x_fg+x_fg).sum().item()}"
        ),
    })

    # ------------------------------------------------------------------
    # graph_break_is_known_op
    # Spec-only: is_known_op("AddBackward") == True,
    #            is_known_op("CustomOp") == False
    # ------------------------------------------------------------------
    fixtures.append({
        "case": "graph_break_is_known_op",
        "module": "graph_break",
        "op": "is_known_op",
        "description": (
            "is_known_op: internal helper. "
            "'AddBackward' is known, 'CustomOpBackward' is not. "
            "Spec-only — no numeric PyTorch reference."
        ),
        "cascade_skip": SPEC_ONLY_SKIP,
    })

    return fixtures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Path to write the fixtures JSON. "
            "Defaults to ferrotorch-jit/tests/conformance/fixtures.json "
            "relative to the repo root."
        ),
    )
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_out = (
        repo_root
        / "ferrotorch-jit"
        / "tests"
        / "conformance"
        / "fixtures.json"
    )
    out_path = pathlib.Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixtures = build_fixtures()

    payload = {
        "metadata": {
            "torch_version": torch.__version__,
            "python_executable": sys.executable,
            "python_platform": platform.platform(),
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "description": (
                "Reference fixtures for ferrotorch-jit C7.2 interpreter + module "
                "conformance suite. "
                "Pin: torch == 2.11.0. "
                "Covers: interpreter.rs, module.rs, aot_autograd.rs, graph_break.rs."
            ),
            "phase": "C7.2",
            "tracking_issue": "#857",
        },
        "fixtures": fixtures,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    n_live = sum(1 for fx in fixtures if "cascade_skip" not in fx)
    n_skip = sum(1 for fx in fixtures if "cascade_skip" in fx)
    print(
        f"Written {out_path}\n"
        f"  {len(fixtures)} total fixtures: {n_live} live, {n_skip} cascade-skip"
    )


if __name__ == "__main__":
    main()
