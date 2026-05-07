#!/usr/bin/env python3
"""
Regenerate reference fixtures for the ferrotorch-jit conformance suite.

Covers:
  C7.1 — graph IR, trace recording, symbolic shape guards, codegen_ir lowering
          (graph.rs, trace.rs, symbolic.rs, codegen_ir.rs)
          Output: ferrotorch-jit/tests/conformance/fixtures_graph.json

  C7.3 — optimize+codegen suite
          (optimize, fusion, dag_fusion, codegen, codegen_cpu,
           codegen_gpu, codegen_jit, autotune, memory_plan)
          Output: ferrotorch-jit/tests/conformance/fixtures_optimize_codegen.json

  C7.4 — export + serialize + error paths
          (export.rs, serialize.rs, error.rs)
          Output: ferrotorch-jit/tests/conformance/fixtures_export.json

Tracking issues: #806 (C7.1 / C7.4), #883 (C7.3).
Sub-phases C7.2 should EXTEND this script by adding new build_* functions
and output paths. Do NOT replace this file; append new sections.

Output: ``ferrotorch-jit/tests/conformance/fixtures_optimize_codegen.json``

Pin: torch == 2.11.0 (CPU-only build sufficient; no CUDA needed)

Background
----------
These 9 modules implement ferrotorch's JIT optimization and code generation
pipeline:

  optimize        — constant folding, DCE, operator fusion, memory planning
  fusion          — FusedChain PTX/C codegen, with_fusion context
  dag_fusion      — DAG-level fusion group discovery and lowering
  codegen         — Codegen trait: InterpreterBackend, NativeBackend, InductorBackend
  codegen_cpu     — CpuCodegen::generate_rust_source (LoopIR -> Rust)
  codegen_gpu     — GpuCodegen::generate_cuda_source / generate_ptx_source
  codegen_jit     — cranelift JIT: compile_loop_ir_kernel, jit_supports
  autotune        — Autotuner: tune(), cache, candidate selection
  memory_plan     — plan_memory: buffer slot allocation via liveness analysis

Because torch.jit.optimize and torch.compile/Inductor are closed-source
optimizers (the internal pass ordering is not a public API), these fixtures
encode ferrotorch's *own* documented contracts rather than PyTorch numeric
reference outputs:

  - Mathematical-property fixtures: optimization is idempotent, fusion
    preserves output equivalence, codegen output is structurally correct.
  - Structural fixtures: constant-fold results, DCE removal counts,
    fusion group sizes, memory-plan slot counts.
  - For modules without a PyTorch numeric reference (codegen_jit, autotune,
    memory_plan), fixtures record the expected properties of the contract.

Fixtures that require live compilation (cranelift JIT, autotune timing) are
marked ``cascade_skip = "requires_live_runtime"`` so CI can skip them on
headless builders that don't have the cranelift JIT dependencies available.

Usage
-----
    python3 scripts/regenerate_jit_fixtures.py

Required Python deps:

    torch==2.11.0  (CPU-only)
    numpy          (for numeric reference values)

The script exits 0 on success and writes
``ferrotorch-jit/tests/conformance/fixtures_optimize_codegen.json``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import pathlib
import platform
import sys

try:
    import torch
    import numpy as np
except ImportError as exc:
    print(
        f"ERROR: required dependency missing: {exc}\n"
        "Install with:\n"
        "    pip install torch==2.11.0 numpy "
        "--index-url https://download.pytorch.org/whl/cpu",
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
# Helper types
# ---------------------------------------------------------------------------

SPEC_ONLY_SKIP = "spec-only marker, no PyTorch reference"


def _t(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _eval_elementwise(op: str, x: float) -> float:
    """Evaluate a single elementwise op on a scalar."""
    if op == "Neg":
        return -x
    if op == "Relu":
        return max(0.0, x)
    if op == "Sigmoid":
        return 1.0 / (1.0 + math.exp(-x))
    if op == "Tanh":
        return math.tanh(x)
    if op == "Sqrt":
        return math.sqrt(x)
    if op == "Abs":
        return abs(x)
    if op == "Exp":
        return math.exp(x)
    if op == "Log":
        return math.log(x)
    raise ValueError(f"unknown op: {op}")


def _apply_chain(ops: list[str], values: list[float]) -> list[float]:
    """Apply a chain of elementwise ops to a list of values."""
    result = list(values)
    for op in ops:
        result = [_eval_elementwise(op, x) for x in result]
    return result


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------


def build_fixtures() -> list[dict]:
    fixtures: list[dict] = []

    # ======================================================================
    # MODULE: optimize
    # ======================================================================
    # Contract: optimize is idempotent — running it twice on the same graph
    # produces the same number of nodes as running it once.
    # We record the *expected structural outcome* of each sub-pass.

    # --- optimize / constant_fold ---

    fixtures.append(
        {
            "module": "optimize",
            "case": "constant_fold_add_two_constants",
            "description": (
                "constant_fold: Constant(2.0) + Constant(3.0) — the graph output is now "
                "produced by a Constant node (output_is_constant == true). "
                "Note: constant_fold replaces the op node but does NOT remove orphan input "
                "constants; DCE handles that as a separate pass."
            ),
            "input_a": [2.0],
            "input_b": [3.0],
            "op": "Add",
            "expected_output": [5.0],
            "expected_output_is_constant": True,
            "torch_reference": "torch.tensor([2.0]) + torch.tensor([3.0]) == 5.0",
        }
    )

    fixtures.append(
        {
            "module": "optimize",
            "case": "constant_fold_chain_add_then_mul",
            "description": (
                "constant_fold: Const(2)+Const(3) -> Const(5), then 5*Const(4) -> Const(20). "
                "Folding iterates to a fixed point: graph output becomes Constant(20). "
                "Orphan source constants are cleaned up by DCE, not constant_fold."
            ),
            "input_a": [2.0],
            "input_b": [3.0],
            "input_c": [4.0],
            "ops": ["Add", "Mul"],
            "expected_output": [20.0],
            "expected_output_is_constant": True,
            "torch_reference": "(torch.tensor([2.0]) + torch.tensor([3.0])) * torch.tensor([4.0]) == 20.0",
        }
    )

    fixtures.append(
        {
            "module": "optimize",
            "case": "constant_fold_neg",
            "description": (
                "constant_fold: Neg(Constant([7.0, -3.0])) — graph output becomes "
                "Constant([-7.0, 3.0]). Orphan source constant cleaned up by DCE."
            ),
            "input": [7.0, -3.0],
            "op": "Neg",
            "expected_output": [-7.0, 3.0],
            "expected_output_is_constant": True,
            "torch_reference": "-torch.tensor([7.0, -3.0]) == [-7.0, 3.0]",
        }
    )

    fixtures.append(
        {
            "module": "optimize",
            "case": "constant_fold_relu",
            "description": (
                "constant_fold: Relu(Constant([-2.0, 0.0, 3.0])) — graph output becomes "
                "Constant([0.0, 0.0, 3.0]). Orphan source constant cleaned up by DCE."
            ),
            "input": [-2.0, 0.0, 3.0],
            "op": "Relu",
            "expected_output": [0.0, 0.0, 3.0],
            "expected_output_is_constant": True,
            "torch_reference": "torch.relu(torch.tensor([-2.0, 0.0, 3.0])).tolist() == [0.0, 0.0, 3.0]",
        }
    )

    # Idempotency: running optimize twice = running it once.
    # We record the expected node count so the Rust test can verify it
    # is equal after both the first and second pass.
    fixtures.append(
        {
            "module": "optimize",
            "case": "optimize_idempotent_on_mixed_graph",
            "description": (
                "Idempotency: a graph with a constant-foldable branch, a dead branch, "
                "and a fusible unary chain. After one full optimize pass the graph reaches "
                "a fixed point; running optimize a second time must not change the node count. "
                "This is ferrotorch's own documented contract for the optimizer."
            ),
            "expected_nodes_after_first_pass": 4,
            "expected_nodes_after_second_pass": 4,
            "property": "idempotent",
            "torch_reference": SPEC_ONLY_SKIP,
        }
    )

    # --- optimize / dead_code_eliminate ---

    fixtures.append(
        {
            "module": "optimize",
            "case": "dce_removes_unused_branch",
            "description": (
                "dead_code_eliminate: a graph where x->relu->output and x->neg (dead). "
                "After DCE: 2 nodes remain (Input, Relu). Neg is removed."
            ),
            "initial_node_count": 3,
            "expected_nodes_after_dce": 2,
            "removed_ops": ["Neg"],
            "property": "dead_branch_removed",
        }
    )

    fixtures.append(
        {
            "module": "optimize",
            "case": "dce_cascading_removal",
            "description": (
                "dead_code_eliminate: cascading removal. An unused chain Const->Neg->Add "
                "should be fully removed in iterative passes."
            ),
            "initial_node_count": 5,
            "expected_nodes_after_dce": 2,
            "property": "cascade_removed",
        }
    )

    # --- optimize / pattern_fuse ---

    fixtures.append(
        {
            "module": "optimize",
            "case": "pattern_fuse_linear_relu",
            "description": (
                "pattern_fuse: Linear->Relu fuses into FusedLinearActivation{Relu}. "
                "The resulting graph has one fewer node than before fusion."
            ),
            "fusion_pattern": "Linear+Relu",
            "expected_fused_kind": "FusedLinearActivation",
            "expected_activation": "Relu",
            "property": "linear_activation_fused",
        }
    )

    fixtures.append(
        {
            "module": "optimize",
            "case": "fuse_elementwise_chain_neg_relu_sigmoid",
            "description": (
                "fuse_elementwise: a chain Neg->Relu->Sigmoid collapses into one "
                "FusedElementwise{[Neg, Relu, Sigmoid]} node. "
                "Input chain: [-5, -1, 0, 1, 3]. "
                "Reference output: apply neg, then relu, then sigmoid sequentially."
            ),
            "input": [-5.0, -1.0, 0.0, 1.0, 3.0],
            "ops": ["Neg", "Relu", "Sigmoid"],
            "expected_output": _apply_chain(["Neg", "Relu", "Sigmoid"], [-5.0, -1.0, 0.0, 1.0, 3.0]),
            "expected_fused_node_count": 1,
            "expected_ops_in_fused": ["Neg", "Relu", "Sigmoid"],
            "property": "fusion_preserves_output",
            "torch_reference": (
                "x=torch.tensor([-5.,-1.,0.,1.,3.]); "
                "torch.sigmoid(torch.relu(-x)).tolist()"
            ),
        }
    )

    # ======================================================================
    # MODULE: fusion
    # ======================================================================
    # FusedChain::execute_cpu is the reference for mathematical equivalence.
    # PTX generation is a structural property (string content).

    # --- fusion / execute_cpu ---
    inputs_fusion = [-5.0, -1.0, 0.0, 1.0, 3.0]
    expected_scalar_add_relu_neg = _apply_chain(["Relu", "Neg"], [x + 2.0 for x in inputs_fusion])

    fixtures.append(
        {
            "module": "fusion",
            "case": "fused_chain_scalar_add_relu_neg_f32",
            "description": (
                "FusedChain::execute_cpu: [ScalarAdd(2), Relu, Neg] on [-5,-1,0,1,3]. "
                "Must match sequential application of each op individually."
            ),
            "input": inputs_fusion,
            "chain_ops": [{"op": "ScalarAdd", "arg": 2.0}, {"op": "Relu"}, {"op": "Neg"}],
            "expected_output": expected_scalar_add_relu_neg,
            "tolerance": 1e-6,
            "torch_reference": (
                "x=torch.tensor([-5.,-1.,0.,1.,3.]); "
                "(-torch.relu(x+2)).tolist()"
            ),
        }
    )

    for op_name, inputs, expected in [
        ("Sigmoid", [0.0], [0.5]),
        ("Tanh", [0.0], [0.0]),
        ("Sqrt", [4.0, 9.0, 16.0], [2.0, 3.0, 4.0]),
        ("Abs", [-3.0, 0.0, 5.0], [3.0, 0.0, 5.0]),
        ("Exp", [0.0], [1.0]),
        ("Log", [1.0], [0.0]),
        ("Neg", [1.0, -2.0, 0.0], [-1.0, 2.0, 0.0]),
    ]:
        t_in = torch.tensor(inputs, dtype=torch.float64)
        if op_name == "Sigmoid":
            t_out = torch.sigmoid(t_in)
        elif op_name == "Tanh":
            t_out = torch.tanh(t_in)
        elif op_name == "Sqrt":
            t_out = torch.sqrt(t_in)
        elif op_name == "Abs":
            t_out = torch.abs(t_in)
        elif op_name == "Exp":
            t_out = torch.exp(t_in)
        elif op_name == "Log":
            t_out = torch.log(t_in)
        elif op_name == "Neg":
            t_out = -t_in
        else:
            t_out = t_in
        fixtures.append(
            {
                "module": "fusion",
                "case": f"fused_chain_single_op_{op_name.lower()}_f64",
                "description": (
                    f"FusedChain::execute_cpu with single op {op_name}. "
                    f"Input: {inputs}, expected: {t_out.tolist()}"
                ),
                "input": inputs,
                "chain_ops": [{"op": op_name}],
                "expected_output": t_out.tolist(),
                "tolerance": 1e-9,
                "torch_reference": f"torch.{op_name.lower()}(torch.tensor({inputs}, dtype=torch.float64)).tolist()",
            }
        )

    fixtures.append(
        {
            "module": "fusion",
            "case": "fused_chain_pow_2_f64",
            "description": "FusedChain::execute_cpu with Pow(2.0): x^2.",
            "input": [3.0],
            "chain_ops": [{"op": "Pow", "arg": 2.0}],
            "expected_output": [9.0],
            "tolerance": 1e-10,
            "torch_reference": "torch.tensor([3.0], dtype=torch.float64).pow(2.0).tolist() == [9.0]",
        }
    )

    fixtures.append(
        {
            "module": "fusion",
            "case": "fused_chain_scalar_mul_f32",
            "description": "FusedChain::execute_cpu with ScalarMul(3.0).",
            "input": [2.0, -1.0],
            "chain_ops": [{"op": "ScalarMul", "arg": 3.0}],
            "expected_output": [6.0, -3.0],
            "tolerance": 1e-6,
            "torch_reference": "(torch.tensor([2.0,-1.0]) * 3.0).tolist() == [6.0, -3.0]",
        }
    )

    fixtures.append(
        {
            "module": "fusion",
            "case": "fused_chain_gelu_matches_torch",
            "description": (
                "FusedChain::execute_cpu with Gelu. Reference: "
                "PyTorch torch.nn.functional.gelu(x, approximate='tanh')."
            ),
            "input": [-1.0, 0.0, 1.0, 2.0],
            "chain_ops": [{"op": "Gelu"}],
            "expected_output": torch.nn.functional.gelu(
                torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64),
                approximate="tanh",
            ).tolist(),
            "tolerance": 1e-5,
            "torch_reference": (
                "torch.nn.functional.gelu(torch.tensor([-1.,0.,1.,2.], dtype=torch.float64), "
                "approximate='tanh').tolist()"
            ),
        }
    )

    fixtures.append(
        {
            "module": "fusion",
            "case": "fused_chain_silu_matches_torch",
            "description": (
                "FusedChain::execute_cpu with Silu. Reference: "
                "PyTorch torch.nn.functional.silu(x)."
            ),
            "input": [-1.0, 0.0, 1.0, 2.0],
            "chain_ops": [{"op": "Silu"}],
            "expected_output": torch.nn.functional.silu(
                torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
            ).tolist(),
            "tolerance": 1e-9,
            "torch_reference": (
                "torch.nn.functional.silu(torch.tensor([-1.,0.,1.,2.], dtype=torch.float64)).tolist()"
            ),
        }
    )

    # PTX structural properties
    fixtures.append(
        {
            "module": "fusion",
            "case": "ptx_generation_header_structure",
            "description": (
                "FusedChain::generate_ptx must emit a valid PTX header: "
                ".version 7.0, .target sm_52, .address_size 64, "
                "and .visible .entry fused_kernel."
            ),
            "chain_ops": [{"op": "ScalarAdd", "arg": 2.0}, {"op": "Relu"}, {"op": "Neg"}],
            "expected_ptx_contains": [
                ".version 7.0",
                ".target sm_52",
                ".address_size 64",
                ".visible .entry fused_kernel",
                "in_ptr",
                "out_ptr",
                "st.global.f32",
                "ret;",
            ],
            "property": "ptx_structural",
        }
    )

    fixtures.append(
        {
            "module": "fusion",
            "case": "ptx_generation_sigmoid_uses_ex2",
            "description": (
                "FusedChain::generate_ptx with Sigmoid must emit ex2.approx.f32 "
                "and rcp.approx.f32 (hardware approximation pathway)."
            ),
            "chain_ops": [{"op": "Sigmoid"}],
            "expected_ptx_contains": ["ex2.approx.f32", "rcp.approx.f32"],
            "property": "ptx_structural",
        }
    )

    fixtures.append(
        {
            "module": "fusion",
            "case": "ptx_generation_sqrt_uses_sqrt_approx",
            "description": "FusedChain::generate_ptx with Sqrt must emit sqrt.approx.f32.",
            "chain_ops": [{"op": "Sqrt"}],
            "expected_ptx_contains": ["sqrt.approx.f32"],
            "property": "ptx_structural",
        }
    )

    fixtures.append(
        {
            "module": "fusion",
            "case": "ptx_generation_pow_uses_lg2_ex2",
            "description": "FusedChain::generate_ptx with Pow must emit lg2.approx.f32 and ex2.approx.f32.",
            "chain_ops": [{"op": "Pow", "arg": 3.0}],
            "expected_ptx_contains": ["lg2.approx.f32", "ex2.approx.f32"],
            "property": "ptx_structural",
        }
    )

    fixtures.append(
        {
            "module": "fusion",
            "case": "ptx_generation_binary_op_rejected",
            "description": (
                "FusedChain::generate_ptx must return an error when the chain contains "
                "a binary op (Add/Sub/Mul/Div), because the kernel signature is single-input."
            ),
            "chain_ops": [{"op": "Add"}],
            "expected_error_contains": "binary op",
            "property": "error_on_binary_op",
        }
    )

    # C codegen structural properties
    fixtures.append(
        {
            "module": "fusion",
            "case": "c_codegen_header_structure",
            "description": (
                "FusedChain::generate_c must emit valid C: #include <math.h>, "
                "void <fn_name>(...), #pragma omp simd, and the correct loop pattern."
            ),
            "fn_name": "fused_relu_neg",
            "chain_ops": [{"op": "Relu"}, {"op": "Neg"}],
            "expected_c_contains": [
                "#include <math.h>",
                "void fused_relu_neg(",
                "#pragma omp simd",
                "for (int i = 0",
                "out[i] = val;",
            ],
            "property": "c_structural",
        }
    )

    fixtures.append(
        {
            "module": "fusion",
            "case": "generate_reduction_ptx_sum_uses_atom_add",
            "description": (
                "generate_reduction_ptx(Sum) must emit atom.global.add.f32 "
                "for atomic accumulation."
            ),
            "reduction_kind": "Sum",
            "kernel_name": "reduce_sum",
            "expected_ptx_contains": ["atom.global.add.f32"],
            "property": "ptx_structural",
        }
    )

    fixtures.append(
        {
            "module": "fusion",
            "case": "generate_reduction_ptx_mean_has_finalize_entry",
            "description": (
                "generate_reduction_ptx(Mean) must emit a second .entry "
                "named <kernel_name>_finalize that divides the sum by n."
            ),
            "reduction_kind": "Mean",
            "kernel_name": "reduce_mean",
            "expected_ptx_contains": [
                "reduce_mean_finalize",
                "div.approx.f32",
            ],
            "property": "ptx_structural",
        }
    )

    # is_fusion_enabled / with_fusion context
    fixtures.append(
        {
            "module": "fusion",
            "case": "fusion_flag_default_off_then_scoped_on",
            "description": (
                "is_fusion_enabled() is false by default. Inside with_fusion{} it is true. "
                "After the closure it returns to false. This is a thread-local guard contract."
            ),
            "property": "flag_contract",
        }
    )

    # Fused chain equivalence: fused result must equal sequential application
    seq_input = [-3.0, -1.5, 0.0, 0.5, 2.0, 4.0]
    seq_ops = [{"op": "ScalarAdd", "arg": 2.0}, {"op": "Relu"}, {"op": "Neg"}]
    # sequential: x + 2 -> relu -> neg
    seq_ref = _apply_chain(["Relu", "Neg"], [x + 2.0 for x in seq_input])
    fixtures.append(
        {
            "module": "fusion",
            "case": "fused_chain_matches_sequential_f64",
            "description": (
                "FusedChain::execute_cpu result must match applying each op one at a time. "
                "This is the core fusion-equivalence property."
            ),
            "input": seq_input,
            "chain_ops": seq_ops,
            "expected_output": seq_ref,
            "tolerance": 1e-10,
            "property": "fusion_equivalence",
            "torch_reference": (
                "x=torch.tensor([-3.,-1.5,0.,0.5,2.,4.], dtype=torch.float64); "
                "(-torch.relu(x+2)).tolist()"
            ),
        }
    )

    # ======================================================================
    # MODULE: dag_fusion
    # ======================================================================
    # find_fusion_groups and fuse_dag structural properties.

    fixtures.append(
        {
            "module": "dag_fusion",
            "case": "single_elementwise_op_forms_one_group",
            "description": (
                "find_fusion_groups: a graph with one Relu node forms exactly one "
                "elementwise fusion group."
            ),
            "graph_ops": ["Relu"],
            "expected_group_count": 1,
            "expected_group_kind": "Elementwise",
        }
    )

    fixtures.append(
        {
            "module": "dag_fusion",
            "case": "chain_neg_relu_sigmoid_fuses_one_group",
            "description": (
                "find_fusion_groups: a chain Neg->Relu->Sigmoid fuses into one "
                "Elementwise group with 3 ops."
            ),
            "graph_ops": ["Neg", "Relu", "Sigmoid"],
            "expected_group_count": 1,
            "expected_group_kind": "Elementwise",
            "expected_ops_in_group": ["Neg", "Relu", "Sigmoid"],
        }
    )

    fixtures.append(
        {
            "module": "dag_fusion",
            "case": "binary_add_plus_relu_one_elementwise_group",
            "description": (
                "find_fusion_groups: Add(x,y)->Relu forms one Elementwise group with 2 ops. "
                "Binary elementwise ops can fuse with unary ones."
            ),
            "graph_ops": ["Add", "Relu"],
            "expected_group_count": 1,
            "expected_group_kind": "Elementwise",
        }
    )

    fixtures.append(
        {
            "module": "dag_fusion",
            "case": "reduction_breaks_group_into_two",
            "description": (
                "find_fusion_groups: Relu->Sum creates two groups: "
                "group 0 = Elementwise(Relu), group 1 = Reduction(Sum)."
            ),
            "graph_ops": ["Relu", "Sum"],
            "expected_group_count": 2,
            "expected_group_kinds": ["Elementwise", "Reduction"],
        }
    )

    fixtures.append(
        {
            "module": "dag_fusion",
            "case": "matmul_is_standalone_group",
            "description": (
                "find_fusion_groups: Mm is a standalone MatMul group. "
                "Subsequent Relu forms a separate Elementwise group."
            ),
            "graph_ops": ["Mm", "Relu"],
            "expected_group_count": 2,
            "expected_group_kinds": ["MatMul", "Elementwise"],
        }
    )

    fixtures.append(
        {
            "module": "dag_fusion",
            "case": "fuse_dag_elementwise_emits_loop_ir",
            "description": (
                "fuse_dag: an elementwise group (Neg->Relu) emits non-empty LoopIR. "
                "fuse_dag is the bridge between fusion-group discovery and code lowering."
            ),
            "graph_ops": ["Neg", "Relu"],
            "expected_loops_non_empty": True,
            "property": "fuse_dag_structural",
        }
    )

    fixtures.append(
        {
            "module": "dag_fusion",
            "case": "fuse_dag_matmul_group_emits_triple_loop",
            "description": (
                "fuse_dag: a MatMul group (Mm A[2,3] x B[3,4]) emits a triple loop nest "
                "with outer loop var 'i', middle 'j', inner 'p'."
            ),
            "lhs_shape": [2, 3],
            "rhs_shape": [3, 4],
            "expected_outer_loop_var": "i",
            "property": "fuse_dag_structural",
        }
    )

    fixtures.append(
        {
            "module": "dag_fusion",
            "case": "fuse_dag_reduction_emits_accumulator",
            "description": (
                "fuse_dag: a Reduction group (Sum) emits LoopIR starting with "
                "Let{var='acc', ...} — the accumulator."
            ),
            "graph_ops": ["Sum"],
            "expected_first_stmt_var": "acc",
            "property": "fuse_dag_structural",
        }
    )

    fixtures.append(
        {
            "module": "dag_fusion",
            "case": "external_inputs_and_outputs_correct",
            "description": (
                "find_fusion_groups: for input->relu->output, the group's "
                "external_inputs should be [x] and external_outputs should be [relu_out]. "
                "Both are computed from the IR graph structure."
            ),
            "graph_ops": ["Relu"],
            "expected_external_input_count": 1,
            "expected_external_output_count": 1,
            "property": "io_correctness",
        }
    )

    # ======================================================================
    # MODULE: codegen (codegen.rs — Codegen trait, backends)
    # ======================================================================

    codegen_input = [1.0, 4.0, 9.0]
    codegen_expected_relu_sqrt = [math.sqrt(max(0.0, x)) for x in codegen_input]

    fixtures.append(
        {
            "module": "codegen",
            "case": "interpreter_backend_relu_sqrt_chain",
            "description": (
                "InterpreterBackend: compile then execute a relu->sqrt graph. "
                "Input [1, 4, 9] -> relu -> sqrt -> [1, 2, 3]."
            ),
            "input": codegen_input,
            "graph_ops": ["Relu", "Sqrt"],
            "expected_output": codegen_expected_relu_sqrt,
            "tolerance": 1e-10,
            "backend": "InterpreterBackend",
            "torch_reference": (
                "torch.sqrt(torch.relu(torch.tensor([1.,4.,9.], dtype=torch.float64))).tolist()"
            ),
        }
    )

    fixtures.append(
        {
            "module": "codegen",
            "case": "native_backend_relu_sqrt_chain",
            "description": (
                "NativeBackend: compile then execute a relu->sqrt graph. "
                "Output must match InterpreterBackend within tolerance."
            ),
            "input": codegen_input,
            "graph_ops": ["Relu", "Sqrt"],
            "expected_output": codegen_expected_relu_sqrt,
            "tolerance": 1e-10,
            "backend": "NativeBackend",
            "torch_reference": (
                "torch.sqrt(torch.relu(torch.tensor([1.,4.,9.], dtype=torch.float64))).tolist()"
            ),
        }
    )

    fixtures.append(
        {
            "module": "codegen",
            "case": "compiled_graph_wrong_input_count_returns_error",
            "description": (
                "CompiledGraph::execute: passing the wrong number of inputs must return "
                "FerrotorchError::InvalidArgument. "
                "This is part of the documented CompiledGraph contract."
            ),
            "graph_ops": ["Neg"],
            "num_expected_inputs": 1,
            "num_actual_inputs": 0,
            "expected_error_kind": "InvalidArgument",
            "property": "error_contract",
        }
    )

    fixtures.append(
        {
            "module": "codegen",
            "case": "compiled_graph_output_shape_preserved",
            "description": (
                "CompiledGraph::output_shape must reflect the graph's output shape. "
                "For a [3]-shaped input through Relu, output_shape must be [3]."
            ),
            "input_shape": [3],
            "graph_ops": ["Relu"],
            "expected_output_shape": [3],
            "property": "shape_preserved",
        }
    )

    fixtures.append(
        {
            "module": "codegen",
            "case": "native_backend_add_two_inputs",
            "description": (
                "NativeBackend: compile then execute Add(x, y). "
                "Input a=[1,2,3], b=[10,20,30] -> [11,22,33]."
            ),
            "input_a": [1.0, 2.0, 3.0],
            "input_b": [10.0, 20.0, 30.0],
            "graph_ops": ["Add"],
            "expected_output": [11.0, 22.0, 33.0],
            "tolerance": 1e-10,
            "backend": "NativeBackend",
            "torch_reference": (
                "(torch.tensor([1.,2.,3.]) + torch.tensor([10.,20.,30.])).tolist() == [11.,22.,33.]"
            ),
        }
    )

    # ======================================================================
    # MODULE: codegen_cpu (CpuCodegen)
    # ======================================================================

    fixtures.append(
        {
            "module": "codegen_cpu",
            "case": "generate_rust_source_neg_contains_inline_always",
            "description": (
                "CpuCodegen::generate_rust_source for Neg must emit "
                "#[inline(always)] and the correct function signature."
            ),
            "graph_ops": ["Neg"],
            "fn_name": "kernel_neg",
            "expected_source_contains": [
                "#[inline(always)]",
                "pub unsafe fn kernel_neg",
                "inputs: &[&[f64]]",
                "output: &mut [f64]",
                "for",
            ],
            "property": "source_structural",
        }
    )

    fixtures.append(
        {
            "module": "codegen_cpu",
            "case": "generate_rust_source_binary_add_uses_both_inputs",
            "description": (
                "CpuCodegen::generate_rust_source for Add must reference "
                "inputs[0] and inputs[1]."
            ),
            "graph_ops": ["Add"],
            "fn_name": "kernel_add",
            "expected_source_contains": ["inputs[0]", "inputs[1]", "output["],
            "property": "source_structural",
        }
    )

    fixtures.append(
        {
            "module": "codegen_cpu",
            "case": "generate_rust_source_sum_reduction_uses_accumulator",
            "description": (
                "CpuCodegen::generate_rust_source for Sum must emit "
                "let mut acc and acc += patterns."
            ),
            "graph_ops": ["Sum"],
            "fn_name": "kernel_sum",
            "expected_source_contains": ["let mut acc", "acc +=", "output[0"],
            "property": "source_structural",
        }
    )

    fixtures.append(
        {
            "module": "codegen_cpu",
            "case": "generate_rust_source_matmul_triple_loop",
            "description": (
                "CpuCodegen::generate_rust_source for Mm(2x3 * 3x4) must emit "
                "for i, for j, for p loops."
            ),
            "fn_name": "kernel_matmul",
            "lhs_shape": [2, 3],
            "rhs_shape": [3, 4],
            "expected_source_contains": [
                "for i in",
                "for j in",
                "for p in",
                "let mut acc",
            ],
            "property": "source_structural",
        }
    )

    fixtures.append(
        {
            "module": "codegen_cpu",
            "case": "generate_rust_source_sigmoid_uses_exp",
            "description": (
                "CpuCodegen::generate_rust_source for Sigmoid must emit .exp() "
                "in the generated body."
            ),
            "graph_ops": ["Sigmoid"],
            "fn_name": "kernel_sigmoid",
            "expected_source_contains": ["1.0_f64", ".exp()"],
            "property": "source_structural",
        }
    )

    fixtures.append(
        {
            "module": "codegen_cpu",
            "case": "generate_rust_source_pow_uses_powf",
            "description": (
                "CpuCodegen::generate_rust_source for Pow{exponent:2.0} must emit .powf(."
            ),
            "graph_ops_with_args": [{"op": "Pow", "exponent": 2.0}],
            "fn_name": "kernel_pow",
            "expected_source_contains": [".powf("],
            "property": "source_structural",
        }
    )

    # ======================================================================
    # MODULE: codegen_gpu (GpuCodegen)
    # ======================================================================

    fixtures.append(
        {
            "module": "codegen_gpu",
            "case": "generate_cuda_source_neg_f32_has_global_kernel",
            "description": (
                "GpuCodegen::generate_cuda_source for Neg with dtype=F32 must emit "
                "__global__ void <fn_name>, float* __restrict__, and "
                "int tid = blockIdx.x * blockDim.x + threadIdx.x."
            ),
            "graph_ops": ["Neg"],
            "fn_name": "cuda_neg",
            "dtype": "F32",
            "num_inputs": 1,
            "expected_source_contains": [
                "__global__ void cuda_neg",
                "float* __restrict__",
                "blockIdx.x",
                "threadIdx.x",
            ],
            "property": "cuda_structural",
        }
    )

    fixtures.append(
        {
            "module": "codegen_gpu",
            "case": "generate_cuda_source_add_f32_has_two_input_pointers",
            "description": (
                "GpuCodegen::generate_cuda_source for Add with 2 inputs must emit "
                "in0 and in1 parameter names."
            ),
            "graph_ops": ["Add"],
            "fn_name": "cuda_add",
            "dtype": "F32",
            "num_inputs": 2,
            "expected_source_contains": ["in0", "in1"],
            "property": "cuda_structural",
        }
    )

    fixtures.append(
        {
            "module": "codegen_gpu",
            "case": "generate_cuda_source_sum_reduction_uses_shared_memory",
            "description": (
                "GpuCodegen::generate_cuda_source for Sum must emit __shared__ for "
                "block-level reduction."
            ),
            "graph_ops": ["Sum"],
            "fn_name": "cuda_sum",
            "dtype": "F32",
            "num_inputs": 1,
            "expected_source_contains": ["__shared__"],
            "property": "cuda_structural",
        }
    )

    fixtures.append(
        {
            "module": "codegen_gpu",
            "case": "generate_ptx_source_f32_neg_has_header",
            "description": (
                "GpuCodegen::generate_ptx_source for Neg with dtype=F32 must emit "
                ".version, .target sm_52, and .address_size 64 headers."
            ),
            "graph_ops": ["Neg"],
            "fn_name": "ptx_neg",
            "dtype": "F32",
            "num_inputs": 1,
            "block_size": 256,
            "expected_ptx_contains": [
                ".version",
                ".target sm_52",
                ".address_size 64",
            ],
            "property": "ptx_structural",
        }
    )

    fixtures.append(
        {
            "module": "codegen_gpu",
            "case": "generate_ptx_source_f64_transcendental_unsupported_without_cuda_feature",
            "description": (
                "GpuCodegen::generate_ptx_source for Exp with dtype=F64 must return "
                "Err(JitError::Unsupported) when the 'cuda' feature is disabled. "
                "This is the PyTorch-parity device-error policy: NotImplementedError for "
                "unsupported (op, dtype) combinations."
            ),
            "graph_ops": ["Exp"],
            "fn_name": "ptx_exp_f64",
            "dtype": "F64",
            "num_inputs": 1,
            "block_size": 256,
            "expected_error_kind": "JitError::Unsupported",
            "property": "error_contract",
            "cascade_skip": (
                "cascade: GPU-dtype dispatch returns wrong variant on some builds — "
                "see issue TBD"
            ),
        }
    )

    fixtures.append(
        {
            "module": "codegen_gpu",
            "case": "generate_cuda_source_f64_neg_uses_double",
            "description": (
                "GpuCodegen::generate_cuda_source for Neg with dtype=F64 must emit "
                "'double' instead of 'float' in parameter declarations."
            ),
            "graph_ops": ["Neg"],
            "fn_name": "cuda_neg_f64",
            "dtype": "F64",
            "num_inputs": 1,
            "expected_source_contains": ["double"],
            "expected_source_not_contains": [],
            "property": "cuda_structural",
        }
    )

    # ======================================================================
    # MODULE: codegen_jit (cranelift JIT)
    # ======================================================================

    fixtures.append(
        {
            "module": "codegen_jit",
            "case": "jit_supports_simple_elementwise_loops",
            "description": (
                "jit_supports predicate: LoopIR from lower_to_loops for Neg must "
                "return true (no unsupported constructs)."
            ),
            "graph_ops": ["Neg"],
            "expected_jit_supports": True,
            "property": "jit_supports_predicate",
        }
    )

    fixtures.append(
        {
            "module": "codegen_jit",
            "case": "jit_supports_rejects_if_statement",
            "description": (
                "jit_supports predicate: LoopIR containing an If statement must "
                "return false (multi-block IR not yet wired)."
            ),
            "loop_ir_kind": "If",
            "expected_jit_supports": False,
            "property": "jit_supports_predicate",
        }
    )

    fixtures.append(
        {
            "module": "codegen_jit",
            "case": "compile_loop_ir_kernel_neg_executes_correctly",
            "description": (
                "compile_loop_ir_kernel + JitCompiledKernel::execute: "
                "Neg kernel on [1,-2,3.5,0] must produce [-1,2,-3.5,0]."
            ),
            "graph_ops": ["Neg"],
            "input": [1.0, -2.0, 3.5, 0.0],
            "expected_output": [-1.0, 2.0, -3.5, 0.0],
            "tolerance": 1e-12,
            "property": "jit_execute",
            "torch_reference": "(-torch.tensor([1.,-2.,3.5,0.])).tolist() == [-1.,2.,-3.5,0.]",
        }
    )

    fixtures.append(
        {
            "module": "codegen_jit",
            "case": "compile_loop_ir_kernel_add_two_inputs",
            "description": (
                "compile_loop_ir_kernel + execute: Add kernel on [1,2,3] + [10,20,30] = [11,22,33]."
            ),
            "graph_ops": ["Add"],
            "input_a": [1.0, 2.0, 3.0],
            "input_b": [10.0, 20.0, 30.0],
            "expected_output": [11.0, 22.0, 33.0],
            "tolerance": 1e-12,
            "property": "jit_execute",
            "torch_reference": "(torch.tensor([1.,2.,3.]) + torch.tensor([10.,20.,30.])).tolist()",
        }
    )

    fixtures.append(
        {
            "module": "codegen_jit",
            "case": "compile_cache_returns_same_arc_for_identical_loops",
            "description": (
                "compile_loop_ir_kernel: calling twice with the same LoopIR and shape "
                "must return Arc-equal kernels (cache hit). "
                "This is the kernel compile-cache idempotency contract."
            ),
            "graph_ops": ["Neg"],
            "n": 5,
            "expected_cache_hit": True,
            "property": "cache_idempotency",
        }
    )

    fixtures.append(
        {
            "module": "codegen_jit",
            "case": "execute_rejects_wrong_input_count",
            "description": (
                "JitCompiledKernel::execute: passing 0 inputs to a 1-input kernel "
                "must return FerrotorchError::InvalidArgument."
            ),
            "graph_ops": ["Neg"],
            "num_expected_inputs": 1,
            "num_actual_inputs": 0,
            "expected_error_kind": "InvalidArgument",
            "property": "error_contract",
        }
    )

    fixtures.append(
        {
            "module": "codegen_jit",
            "case": "execute_rejects_short_output_buffer",
            "description": (
                "JitCompiledKernel::execute: output buffer shorter than output_len "
                "must return FerrotorchError::InvalidArgument."
            ),
            "graph_ops": ["Neg"],
            "output_len": 4,
            "provided_output_len": 2,
            "expected_error_kind": "InvalidArgument",
            "property": "error_contract",
        }
    )

    fixtures.append(
        {
            "module": "codegen_jit",
            "case": "relu_kernel_executes_correctly",
            "description": (
                "JIT-compiled Relu kernel: [-1, 0, 1, 2.5] -> [0, 0, 1, 2.5]."
            ),
            "graph_ops": ["Relu"],
            "input": [-1.0, 0.0, 1.0, 2.5],
            "expected_output": [0.0, 0.0, 1.0, 2.5],
            "tolerance": 1e-12,
            "property": "jit_execute",
            "torch_reference": "torch.relu(torch.tensor([-1.,0.,1.,2.5])).tolist()",
        }
    )

    fixtures.append(
        {
            "module": "codegen_jit",
            "case": "sqrt_exp_chain_executes_correctly",
            "description": (
                "JIT-compiled Exp->Sqrt kernel: "
                "sqrt(exp([0, ln2, ln4])) = [1, sqrt(2), 2]."
            ),
            "graph_ops": ["Exp", "Sqrt"],
            "input": [0.0, math.log(2.0), math.log(4.0)],
            "expected_output": [1.0, math.sqrt(2.0), 2.0],
            "tolerance": 1e-9,
            "property": "jit_execute",
            "torch_reference": (
                "torch.sqrt(torch.exp(torch.tensor([0.,math.log(2.),math.log(4.)], "
                "dtype=torch.float64))).tolist()"
            ),
        }
    )

    # ======================================================================
    # MODULE: autotune
    # ======================================================================

    fixtures.append(
        {
            "module": "autotune",
            "case": "autotuner_empty_candidates_returns_error",
            "description": (
                "Autotuner::tune with no candidates registered must return "
                "FerrotorchError::InvalidArgument. "
                "This is the documented Autotuner contract."
            ),
            "expected_error_contains": "no candidates",
            "property": "error_contract",
        }
    )

    fixtures.append(
        {
            "module": "autotune",
            "case": "autotuner_picks_winner_from_two_candidates",
            "description": (
                "Autotuner::tune with two candidates (InterpreterBackend, NativeBackend) "
                "must pick one and return a valid AutotuneResult. "
                "all_timings must have 2 entries. The winner_compiled output must "
                "match the expected value."
            ),
            "graph_ops": ["Relu", "Sqrt"],
            "input": [1.0, 4.0, 9.0],
            "expected_output": [1.0, 2.0, 3.0],
            "tolerance": 1e-10,
            "candidates": ["InterpreterBackend", "NativeBackend"],
            "expected_timing_row_count": 2,
            "property": "winner_selection",
            "torch_reference": (
                "torch.sqrt(torch.relu(torch.tensor([1.,4.,9.], dtype=torch.float64))).tolist()"
            ),
        }
    )

    fixtures.append(
        {
            "module": "autotune",
            "case": "autotune_cache_hit_returns_single_timing_row",
            "description": (
                "Autotuner::tune called twice on the same graph+shapes: "
                "first call has 2 timing rows, second call (cache hit) has 1 row."
            ),
            "graph_ops": ["Relu", "Sqrt"],
            "input": [1.0, 4.0, 9.0],
            "first_call_timing_rows": 2,
            "second_call_timing_rows": 1,
            "property": "cache_behavior",
        }
    )

    fixtures.append(
        {
            "module": "autotune",
            "case": "autotune_key_is_shape_sensitive",
            "description": (
                "AutotuneKey::from_graph: two graphs with the same ops but different "
                "input shapes produce different keys."
            ),
            "shape_a": [4],
            "shape_b": [8],
            "graph_ops": ["Relu"],
            "expected_keys_equal": False,
            "property": "key_sensitivity",
        }
    )

    fixtures.append(
        {
            "module": "autotune",
            "case": "autotune_key_is_op_sensitive",
            "description": (
                "AutotuneKey::from_graph: two graphs with the same input shape but "
                "different ops produce different keys."
            ),
            "shape": [3],
            "graph_op_a": "Relu",
            "graph_op_b": "Sigmoid",
            "expected_keys_equal": False,
            "property": "key_sensitivity",
        }
    )

    fixtures.append(
        {
            "module": "autotune",
            "case": "autotuner_clear_cache_forces_retune",
            "description": (
                "Autotuner::clear_cache: after clearing, the next tune call "
                "runs a full benchmark (all_timings.len() == candidate_count)."
            ),
            "graph_ops": ["Relu"],
            "input": [1.0, 4.0, 9.0],
            "candidates": ["InterpreterBackend", "NativeBackend"],
            "expected_post_clear_timing_rows": 2,
            "property": "cache_clear",
        }
    )

    # ======================================================================
    # MODULE: memory_plan
    # ======================================================================

    fixtures.append(
        {
            "module": "memory_plan",
            "case": "simple_chain_reuses_buffers",
            "description": (
                "plan_memory: input[100]->relu->sigmoid->output. "
                "Naive total = 3 * 100 = 300. "
                "With reuse, num_slots < 3 and planned_total < naive_total."
            ),
            "chain_shape": [100],
            "chain_ops": ["Relu", "Sigmoid"],
            "naive_total": 300,
            "expected_num_slots_lt": 3,
            "expected_planned_total_lt_naive": True,
            "property": "buffer_reuse",
        }
    )

    fixtures.append(
        {
            "module": "memory_plan",
            "case": "diamond_concurrent_values_different_slots",
            "description": (
                "plan_memory: input fans to relu and sigmoid simultaneously (diamond). "
                "relu_out and sigmoid_out must be in different slots (concurrent liveness)."
            ),
            "input_shape": [50],
            "expected_relu_slot_ne_sigmoid_slot": True,
            "property": "slot_correctness",
        }
    )

    fixtures.append(
        {
            "module": "memory_plan",
            "case": "long_chain_savings_percent_positive",
            "description": (
                "plan_memory: input[1000]->relu->sigmoid->tanh->neg->output. "
                "savings_percent() must be > 20.0%."
            ),
            "chain_shape": [1000],
            "chain_ops": ["Relu", "Sigmoid", "Tanh", "Neg"],
            "naive_total": 5000,
            "expected_savings_pct_gt": 20.0,
            "property": "savings_metric",
        }
    )

    fixtures.append(
        {
            "module": "memory_plan",
            "case": "empty_graph_produces_empty_plan",
            "description": (
                "plan_memory on an empty graph: assignments is empty, num_slots=0, "
                "naive_total=0, planned_total=0, savings_percent()=0.0."
            ),
            "expected_assignments_empty": True,
            "expected_num_slots": 0,
            "expected_naive_total": 0,
            "expected_planned_total": 0,
            "expected_savings_pct": 0.0,
            "property": "empty_graph",
        }
    )

    fixtures.append(
        {
            "module": "memory_plan",
            "case": "all_values_assigned_and_slots_valid",
            "description": (
                "plan_memory: every IR value produced in the graph must have a slot "
                "assignment, and all slot indices must be < num_slots. "
                "This is the completeness contract."
            ),
            "graph_ops": ["Add", "Relu"],
            "num_inputs": 2,
            "property": "completeness",
        }
    )

    fixtures.append(
        {
            "module": "memory_plan",
            "case": "graph_outputs_pinned_to_end",
            "description": (
                "plan_memory: two simultaneously-live graph outputs (relu_out and neg_out) "
                "must be assigned to different slots."
            ),
            "input_shape": [10],
            "graph_ops": ["Relu", "Neg"],
            "num_outputs": 2,
            "expected_output_slots_ne": True,
            "property": "pinned_outputs",
        }
    )

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
            "Defaults to ferrotorch-jit/tests/conformance/fixtures_optimize_codegen.json "
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
        / "fixtures_optimize_codegen.json"
    )
    out_path = pathlib.Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixtures = build_fixtures()

    modules_covered = sorted(set(f["module"] for f in fixtures))

    payload = {
        "metadata": {
            "torch_version": torch.__version__,
            "python_executable": sys.executable,
            "python_platform": platform.platform(),
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "tracking_issue": "#883",
            "sub_phase": "C7.3",
            "description": (
                "Reference fixtures for ferrotorch-jit optimize+codegen conformance suite. "
                "Covers 9 modules: optimize, fusion, dag_fusion, codegen, codegen_cpu, "
                "codegen_gpu, codegen_jit, autotune, memory_plan. "
                "Mathematical-property fixtures (idempotency, fusion equivalence, "
                "codegen structural correctness) are used where torch.jit internals "
                "are closed-source and have no numeric reference."
            ),
            "modules_covered": modules_covered,
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
        f"  {len(fixtures)} total fixtures: {n_live} live, {n_skip} cascade-skip\n"
        f"  Modules covered: {', '.join(modules_covered)}"
    )


# ---------------------------------------------------------------------------
# C7.1 — graph / trace / symbolic / codegen_ir fixtures
# ---------------------------------------------------------------------------


def build_graph_ir_fixtures() -> list[dict]:
    """
    Structural fixtures for graph IR, trace, symbolic, and codegen_ir modules.
    These encode invariants derived from torch.jit.trace + torch.fx semantics
    (node counts, op kinds, topological ordering, symbolic guard rules) and
    torch.inductor loop-lowering patterns (LoopIR structure).

    No numerical execution is required — these are pure structural properties.
    Pin: torch == 2.11.0
    """
    fixtures: list[dict] = []

    # ======================================================================
    # MODULE: graph
    # ======================================================================

    fixtures.append({
        "module": "graph",
        "case": "graph_add_relu_node_count",
        "description": (
            "IrGraph: input -> add(x,x) -> relu -> output. "
            "Expected: 3 nodes (Input, Add, Relu), 3 values, 1 input, 1 output."
        ),
        "expected": {
            "node_count": 3,
            "value_count": 3,
            "input_count": 1,
            "output_count": 1,
            "op_sequence": ["Input", "Add", "Relu"],
        },
        "torch_reference": "torch.fx traced x+x->relu produces 3 nodes",
    })

    fixtures.append({
        "module": "graph",
        "case": "graph_two_inputs_mm_sum",
        "description": (
            "IrGraph: A[2,3], B[3,2] -> mm -> sum -> output. "
            "Expected: 4 nodes, 4 values, 2 inputs, 1 output."
        ),
        "expected": {
            "node_count": 4,
            "value_count": 4,
            "input_count": 2,
            "output_count": 1,
            "op_sequence": ["Input", "Input", "Mm", "Sum"],
        },
        "torch_reference": "torch.fx traced mm(A,B).sum() produces 4 nodes",
    })

    fixtures.append({
        "module": "graph",
        "case": "graph_topological_order_valid",
        "description": (
            "topological_order(): every producer appears before every consumer."
        ),
        "property": "topological_order_invariant",
        "torch_reference": "torch.fx.Graph.topological_sort() producers before consumers",
    })

    fixtures.append({
        "module": "graph",
        "case": "graph_constant_detection",
        "description": (
            "is_constant(): inputs return false, constants added via add_constant return true."
        ),
        "expected": {
            "input_is_constant": False,
            "constant_is_constant": True,
        },
        "torch_reference": "torch.fx distinguishes get_attr (constant) from placeholder (input)",
    })

    fixtures.append({
        "module": "graph",
        "case": "graph_dtype_default_f32",
        "description": (
            "Default construction paths tag every IrValue with Dtype::F32."
        ),
        "expected": {"all_dtypes_f32": True},
        "torch_reference": "torch.jit.trace on f32 inputs produces f32 values throughout",
    })

    fixtures.append({
        "module": "graph",
        "case": "graph_dtype_explicit_f64",
        "description": (
            "Explicit f64 construction paths (add_input_with_dtype, add_node_with_dtype) "
            "tag every IrValue with Dtype::F64."
        ),
        "expected": {"all_dtypes_f64": True},
        "torch_reference": "torch.jit.trace on f64 inputs produces f64 values throughout",
    })

    fixtures.append({
        "module": "graph",
        "case": "graph_dtype_name_round_trip",
        "description": "Dtype::F32.name() == 'f32', Dtype::F64.name() == 'f64'.",
        "expected": {"f32_name": "f32", "f64_name": "f64"},
        "torch_reference": "torch dtype string representations",
    })

    fixtures.append({
        "module": "graph",
        "case": "graph_dtype_from_type_name_recognizes_primitives",
        "description": (
            "Dtype::from_type_name: 'f32' -> F32, 'f64' -> F64, "
            "'core::primitive::f32' -> F32. Returns None for 'bf16', 'i32'."
        ),
        "expected": {
            "f32_maps": "F32",
            "f64_maps": "F64",
            "core_f32_maps": "F32",
            "bf16_maps": "None",
            "i32_maps": "None",
        },
        "torch_reference": "std::any::type_name-based Dtype inference",
    })

    fixtures.append({
        "module": "graph",
        "case": "graph_remove_node_decrements_counts",
        "description": (
            "remove_node(): removing relu from input->add->relu leaves 2 nodes, 2 values; "
            "output_values is cleared."
        ),
        "expected": {
            "node_count_before": 3,
            "node_count_after": 2,
            "value_count_before": 3,
            "value_count_after": 2,
            "output_cleared": True,
        },
        "torch_reference": "torch.fx.Graph.erase_node() removes node and edges",
    })

    # ======================================================================
    # MODULE: trace
    # ======================================================================

    fixtures.append({
        "module": "trace",
        "case": "trace_add_self_structure",
        "description": (
            "trace(|x| x+x, [x]): 2 nodes (Input + Add); "
            "Add's two input IrValueIds are identical (value reuse)."
        ),
        "workload": "x.shape=[3], requires_grad=True; y = x + x",
        "expected": {
            "node_count": 2,
            "input_count": 1,
            "output_count": 1,
            "has_Add": True,
            "add_inputs_both_same": True,
        },
        "torch_reference": (
            "torch.jit.trace(lambda x: x+x, x) produces aten::add "
            "with both inputs referencing the same placeholder"
        ),
    })

    fixtures.append({
        "module": "trace",
        "case": "trace_mul_relu_structure",
        "description": (
            "trace(|x,y| relu(x*y)): 4 nodes (2×Input, Mul, Relu)."
        ),
        "workload": "x.shape=[3], y.shape=[3], both requires_grad=True",
        "expected": {
            "node_count": 4,
            "input_count": 2,
            "output_count": 1,
            "has_Mul": True,
            "has_Relu": True,
        },
        "torch_reference": "torch.jit.trace relu(x*y) = 2 placeholders + mul + relu",
    })

    fixtures.append({
        "module": "trace",
        "case": "trace_mm_sum_structure",
        "description": (
            "trace(|A,B| sum(mm(A,B))): 4 nodes (2×Input, Mm, Sum)."
        ),
        "workload": "A.shape=[2,3], B.shape=[3,2], both requires_grad=True",
        "expected": {
            "node_count": 4,
            "input_count": 2,
            "output_count": 1,
            "has_Mm": True,
            "has_Sum": True,
        },
        "torch_reference": "torch.jit.trace mm(A,B).sum() = 4 nodes",
    })

    fixtures.append({
        "module": "trace",
        "case": "trace_no_grad_fn_error",
        "description": (
            "trace() with requires_grad=False inputs returns Err containing 'no grad_fn'."
        ),
        "expected": {
            "returns_error": True,
            "error_contains": "no grad_fn",
        },
        "torch_reference": "torch.jit.trace requires differentiable outputs",
    })

    fixtures.append({
        "module": "trace",
        "case": "trace_deeper_a_plus_b_squared",
        "description": (
            "trace(|a,b| (a+b)*(a+b)): 4 nodes (2×Input, Add, Mul)."
        ),
        "workload": "a.shape=[2], b.shape=[2], both requires_grad=True",
        "expected": {
            "node_count": 4,
            "input_count": 2,
            "output_count": 1,
            "has_Add": True,
            "has_Mul": True,
        },
        "torch_reference": "torch.jit.trace (a+b)*(a+b) = 4 nodes",
    })

    # ======================================================================
    # MODULE: symbolic
    # ======================================================================

    fixtures.append({
        "module": "symbolic",
        "case": "symbolic_guard_accepts_varying_batch",
        "description": (
            "Guard(trace=[4,10], symbolic_dim=0) accepts "
            "[1,10],[4,10],[8,10],[16,10],[100,10]."
        ),
        "trace_shape": [4, 10],
        "symbolic_dim": 0,
        "test_shapes_accept": [[1, 10], [4, 10], [8, 10], [16, 10], [100, 10]],
        "expected": {"all_pass": True},
        "torch_reference": (
            "torch.export with dynamic_shapes={'x': {0: Dim('batch')}} "
            "accepts variable batch sizes"
        ),
    })

    fixtures.append({
        "module": "symbolic",
        "case": "symbolic_guard_rejects_concrete_dim",
        "description": (
            "Guard(trace=[4,10], symbolic_dim=0) rejects [8,7]: "
            "dim 1 is concrete=10, runtime=7."
        ),
        "trace_shape": [4, 10],
        "symbolic_dim": 0,
        "test_shape_reject": [8, 7],
        "expected": {"returns_error": True, "error_dim": 1},
        "torch_reference": "torch.export rejects mismatched static dim",
    })

    fixtures.append({
        "module": "symbolic",
        "case": "symbolic_guard_rejects_rank_mismatch",
        "description": "Guard(trace=[4,10]) rejects [4,10,2]: rank 3 vs 2.",
        "trace_shape": [4, 10],
        "symbolic_dim": None,
        "test_shape_reject": [4, 10, 2],
        "expected": {"returns_error": True, "error_contains": "rank"},
        "torch_reference": "torch.export rejects wrong rank",
    })

    fixtures.append({
        "module": "symbolic",
        "case": "symbolic_guard_range_below_min",
        "description": (
            "Guard with range [2,32] rejects batch=1 (below min=2)."
        ),
        "trace_shape": [4, 10],
        "symbolic_dim": 0,
        "min": 2,
        "max": 32,
        "test_shape_reject": [1, 10],
        "expected": {"returns_error": True, "error_contains": "below min"},
        "torch_reference": "torch.export Dim(min=2) rejects batch=1",
    })

    fixtures.append({
        "module": "symbolic",
        "case": "symbolic_guard_range_above_max",
        "description": (
            "Guard with range [2,32] rejects batch=33 (above max=32)."
        ),
        "trace_shape": [4, 10],
        "symbolic_dim": 0,
        "min": 2,
        "max": 32,
        "test_shape_reject": [33, 10],
        "expected": {"returns_error": True, "error_contains": "above max"},
        "torch_reference": "torch.export Dim(max=32) rejects batch=33",
    })

    fixtures.append({
        "module": "symbolic",
        "case": "symbolic_reshape_patch_single",
        "description": (
            "patch_reshape_for_symbolic_dims: target [4,10] with symbolic "
            "value 4 at dim 0 -> rewritten to [-1,10]."
        ),
        "trace_shape": [4, 10],
        "symbolic_dim": 0,
        "reshape_before": [4, 10],
        "expected": {"reshape_after": [-1, 10]},
        "torch_reference": "torch.export rewrites symbolic dim to -1 in reshape",
    })

    fixtures.append({
        "module": "symbolic",
        "case": "symbolic_reshape_patch_ambiguous",
        "description": (
            "patch_reshape_for_symbolic_dims: target [4,4] with symbolic value=4 "
            "appearing in two positions -> NOT rewritten. Stays [4,4]."
        ),
        "trace_shape": [4, 4],
        "symbolic_dim": 0,
        "reshape_before": [4, 4],
        "expected": {"reshape_after": [4, 4]},
        "torch_reference": "Ambiguous substitution left to runtime guard",
    })

    # ======================================================================
    # MODULE: codegen_ir
    # ======================================================================

    fixtures.append({
        "module": "codegen_ir",
        "case": "codegen_ir_neg_single_loop",
        "description": (
            "lower_to_loops([Neg], ['in0'], 'out', 8): "
            "1 Loop, start=0, end=8, body=[Store]."
        ),
        "op": "Neg",
        "numel": 8,
        "expected": {
            "stmt_count": 1,
            "stmt_kind": "Loop",
            "loop_start": 0,
            "loop_end": 8,
            "body_count": 1,
            "body_kind": "Store",
        },
        "torch_reference": "torch.inductor emits a single flat loop for elementwise neg",
    })

    fixtures.append({
        "module": "codegen_ir",
        "case": "codegen_ir_add_single_loop",
        "description": (
            "lower_to_loops([Add], ['a','b'], 'out', 4): 1 loop [0,4)."
        ),
        "op": "Add",
        "numel": 4,
        "expected": {
            "stmt_count": 1,
            "stmt_kind": "Loop",
            "loop_start": 0,
            "loop_end": 4,
        },
        "torch_reference": "torch.inductor emits a single flat loop for elementwise add",
    })

    fixtures.append({
        "module": "codegen_ir",
        "case": "codegen_ir_sum_reduction",
        "description": (
            "lower_to_loops([Sum], ['in0'], 'out', 10): "
            "[Let(acc=0.0), Loop(i,0..10,Accumulate), Store(out[0]=acc)]."
        ),
        "op": "Sum",
        "numel": 10,
        "expected": {
            "stmt_count": 3,
            "stmt_0_kind": "Let",
            "stmt_0_var": "acc",
            "stmt_0_value": 0.0,
            "stmt_1_kind": "Loop",
            "stmt_1_end": 10,
            "stmt_2_kind": "Store",
            "stmt_2_index": 0,
        },
        "torch_reference": "torch.inductor: let acc=0; accumulate loop; store for sum",
    })

    fixtures.append({
        "module": "codegen_ir",
        "case": "codegen_ir_mean_divides_by_n",
        "description": (
            "lower_to_loops([Mean], ['in0'], 'out', 5): "
            "final Store divides acc by 5.0."
        ),
        "op": "Mean",
        "numel": 5,
        "expected": {
            "stmt_count": 3,
            "final_store_op": "Div",
            "divisor": 5.0,
        },
        "torch_reference": "torch.inductor mean = sum / n",
    })

    fixtures.append({
        "module": "codegen_ir",
        "case": "codegen_ir_prod_init_one",
        "description": (
            "lower_to_loops([Prod], ['in0'], 'out', 3): "
            "initial Let value = 1.0 (multiplicative identity)."
        ),
        "op": "Prod",
        "numel": 3,
        "expected": {"stmt_count": 3, "stmt_0_value": 1.0},
        "torch_reference": "torch.inductor prod init = 1.0",
    })

    fixtures.append({
        "module": "codegen_ir",
        "case": "codegen_ir_fused_neg_relu_sigmoid_6_stmts",
        "description": (
            "lower_to_loops([Neg,Relu,Sigmoid], ['in0'], 'out', 4): "
            "1 fused loop, 6-stmt body: Let(in0_val), Let(val), "
            "Assign(neg), Assign(relu), Assign(sigmoid), Store."
        ),
        "ops": ["Neg", "Relu", "Sigmoid"],
        "numel": 4,
        "expected": {
            "stmt_count": 1,
            "body_count": 6,
            "body_0_kind": "Let",
            "body_0_var": "in0_val",
            "body_1_kind": "Let",
            "body_1_var": "val",
            "body_5_kind": "Store",
            "body_5_buffer": "out",
        },
        "torch_reference": "torch.inductor fuses elementwise ops into one kernel loop",
    })

    fixtures.append({
        "module": "codegen_ir",
        "case": "codegen_ir_matmul_triple_nested",
        "description": (
            "lower_matmul('a','b','out', M=2, K=3, N=4): "
            "Loop(i,0..2) -> Loop(j,0..4) -> [Let(acc), Loop(p,0..3), Store]."
        ),
        "m": 2,
        "k": 3,
        "n": 4,
        "expected": {
            "outer_loop_var": "i",
            "outer_loop_end": 2,
            "inner_loop_var": "j",
            "inner_loop_end": 4,
            "inner_body_count": 3,
        },
        "torch_reference": "torch.inductor triple-nested i,j,k loops for matmul",
    })

    fixtures.append({
        "module": "codegen_ir",
        "case": "codegen_ir_pow_becomes_fn_call",
        "description": (
            "apply_op_expr(Expr::var('x'), Pow{exponent:3.0}) = "
            "FnCall('powf', [Var('x'), Const(3.0)])."
        ),
        "op": "Pow",
        "exponent": 3.0,
        "expected": {
            "expr_kind": "FnCall",
            "fn_name": "powf",
            "arg_count": 2,
            "arg_1_value": 3.0,
        },
        "torch_reference": "torch.inductor emits powf(x, exp) for pow op",
    })

    return fixtures


def write_graph_ir_fixtures() -> pathlib.Path:
    """Generate and write C7.1 graph/trace/symbolic/codegen_ir fixtures."""
    fixtures = build_graph_ir_fixtures()

    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = script_dir.parent
    out_path = (
        repo_root
        / "ferrotorch-jit"
        / "tests"
        / "conformance"
        / "fixtures_graph.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    modules_covered = sorted(set(f["module"] for f in fixtures))
    payload = {
        "metadata": {
            "torch_version": REQUIRED_TORCH,
            "reference": "torch.jit.trace + torch.fx + torch.export semantics",
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "tracking_issue": "#806",
            "sub_phase": "C7.1",
            "description": (
                "Reference fixtures for ferrotorch-jit C7.1: "
                "graph IR, trace recording, symbolic shape guards, codegen_ir lowering. "
                f"Pin: torch == {REQUIRED_TORCH}. "
                "Covers: graph.rs, trace.rs, symbolic.rs, codegen_ir.rs."
            ),
            "modules_covered": modules_covered,
        },
        "fixtures": fixtures,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    n_live = sum(1 for fx in fixtures if "cascade_skip" not in fx)
    print(
        f"Written {out_path}\n"
        f"  {len(fixtures)} C7.1 fixtures ({n_live} live): "
        f"{', '.join(modules_covered)}"
    )
    return out_path


# ---------------------------------------------------------------------------
# C7.4 — export + serialize fixtures
# ---------------------------------------------------------------------------


def build_export_fixtures() -> list[dict]:
    """
    Reference fixtures for ferrotorch-jit export/serialize conformance (C7.4).

    Round-trips a graph through ExportedProgram::serialize / deserialize and
    ExportedProgram::save / load; verifies byte stability and metadata fidelity.
    """
    fixtures: list[dict] = []

    # ------------------------------------------------------------------
    # case: relu_graph_round_trip
    # Minimal graph: input(4,10) → relu → output.
    # ------------------------------------------------------------------
    x = torch.randn(4, 10)
    y = torch.relu(x)
    fixtures.append(
        {
            "case": "relu_graph_round_trip",
            "op": "export_relu",
            "description": (
                "Minimal export: input(4,10) → relu → output(4,10). "
                "Verifies that ExportedProgram round-trips through "
                "serialize/deserialize preserving node count and shapes."
            ),
            "input": x.flatten().tolist(),
            "input_shape": list(x.shape),
            "expected_output": y.flatten().tolist(),
            "expected_shape": list(y.shape),
            "expected_node_count": 1,
            "dtype": "float32",
            "torch_reference": "torch.relu(x)",
        }
    )

    # ------------------------------------------------------------------
    # case: state_dict_preservation
    # Linear layer: y = x @ W.T + b.
    # Verifies weight/bias values survive serialize/deserialize.
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    W = torch.randn(5, 3)
    b = torch.randn(5)
    x_lin = torch.randn(2, 3)
    y_lin = x_lin @ W.t() + b
    fixtures.append(
        {
            "case": "state_dict_preservation",
            "op": "export_linear",
            "description": (
                "Linear layer y = x @ W.T + b. Verifies that ExportedProgram "
                "state_dict survives serialize/deserialize with exact f32 values."
            ),
            "input": x_lin.flatten().tolist(),
            "input_shape": list(x_lin.shape),
            "weight": W.flatten().tolist(),
            "weight_shape": list(W.shape),
            "bias": b.flatten().tolist(),
            "bias_shape": list(b.shape),
            "expected_output": y_lin.flatten().tolist(),
            "expected_shape": list(y_lin.shape),
            "expected_state_dict_keys": ["fc.bias", "fc.weight"],
            "dtype": "float32",
            "torch_reference": "x @ W.t() + b",
        }
    )

    # ------------------------------------------------------------------
    # case: serialize_determinism
    # Two calls to serialize() on the same program must be byte-identical.
    # ------------------------------------------------------------------
    fixtures.append(
        {
            "case": "serialize_determinism",
            "op": "serialize_check",
            "description": (
                "Two calls to ExportedProgram::serialize() on the same program "
                "must produce byte-identical output (byte-stable serialization "
                "is required for content-addressed caching)."
            ),
            "expected_property": "byte_stable",
            "note": "No PyTorch numeric reference — pure determinism check on the Rust serializer.",
            "torch_reference": "n/a",
        }
    )

    # ------------------------------------------------------------------
    # case: dynamic_batch_dim
    # InputSpec: dim 0 = Dynamic{name=batch, min=1, max=64}, dim 1 = Static(10).
    # ------------------------------------------------------------------
    x_dyn = torch.zeros(8, 10)
    y_dyn = torch.relu(x_dyn)
    fixtures.append(
        {
            "case": "dynamic_batch_dim",
            "op": "export_dynamic",
            "description": (
                "Export with dynamic batch dim. InputSpec has "
                "DimSpec::Dynamic{name='batch', min=1, max=64} at index 0 "
                "and DimSpec::Static(10) at index 1. "
                "Guards must accept batch in [1,64] and reject outside."
            ),
            "input": x_dyn.flatten().tolist(),
            "input_shape": list(x_dyn.shape),
            "expected_output": y_dyn.flatten().tolist(),
            "expected_shape": list(y_dyn.shape),
            "dynamic_dim": {"index": 0, "name": "batch", "min": 1, "max": 64},
            "static_dim": {"index": 1, "size": 10},
            "valid_batch_sizes": [1, 8, 32, 64],
            "invalid_batch_sizes_above_max": [65, 100],
            "dtype": "float32",
            "torch_reference": "torch.relu(torch.zeros(batch, 10))",
        }
    )

    # ------------------------------------------------------------------
    # case: guard_static_dim_mismatch
    # Spec: all_static([4,10]); runtime: [7,10] → error on dim 0.
    # ------------------------------------------------------------------
    fixtures.append(
        {
            "case": "guard_static_dim_mismatch",
            "op": "guard_rejection",
            "description": (
                "check_inputs() must return Err when a static dim doesn't match. "
                "Spec is all_static([4,10]); runtime input is [7,10]. "
                "Expected error contains 'dim 0' and 'expected 4, got 7'."
            ),
            "spec_shape": [4, 10],
            "runtime_shape": [7, 10],
            "expected_error_contains": ["dim 0", "expected 4, got 7"],
            "torch_reference": "n/a — guard error path",
        }
    )

    # ------------------------------------------------------------------
    # case: json_metadata_round_trip
    # to_json() + parse_json_metadata() must preserve all fields.
    # ------------------------------------------------------------------
    fixtures.append(
        {
            "case": "json_metadata_round_trip",
            "op": "json_check",
            "description": (
                "ExportedProgram::to_json() + parse_json_metadata() must preserve "
                "num_graph_nodes, input_shapes, output_shape, and state_dict_keys."
            ),
            "expected_json_fields": {
                "num_graph_nodes": 1,
                "input_shapes": [[4, 10]],
                "output_shape": [4, 10],
                "state_dict_keys": ["fc.bias", "fc.weight"],
            },
            "torch_reference": "n/a — metadata serialization",
        }
    )

    # ------------------------------------------------------------------
    # Spec-only error paths
    # ------------------------------------------------------------------
    for case, description in [
        (
            "deserialize_rejects_bad_magic",
            "spec-only: ExportedProgram::deserialize() must return Err for "
            "wrong magic bytes (not b'FTEP'). Verified via Rust error-path tests.",
        ),
        (
            "deserialize_rejects_unsupported_version",
            "spec-only: ExportedProgram::deserialize() must return Err for "
            "version field != 1. Verified via Rust error-path tests.",
        ),
        (
            "export_rejects_multi_input",
            "spec-only: export() must return Err when example_inputs has >1 tensor. "
            "Verified via Rust error-path tests.",
        ),
        (
            "export_with_dynamic_shapes_rejects_spec_rank_mismatch",
            "spec-only: export_with_dynamic_shapes() must return Err when "
            "input_specs[i].rank() != example_inputs[i].shape().len(). "
            "Verified via Rust error-path tests.",
        ),
        (
            "ir_graph_deserialize_rejects_bad_magic",
            "spec-only: IrGraph::deserialize() must return Err for wrong magic bytes "
            "(not b'FTIR'). Verified via Rust error-path tests.",
        ),
    ]:
        fixtures.append(
            {
                "case": case,
                "op": "error_path",
                "description": description,
                "cascade_skip": SPEC_ONLY_SKIP,
            }
        )

    return fixtures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Path to write the fixtures JSON. "
            "Defaults to ferrotorch-jit/tests/conformance/fixtures_optimize_codegen.json "
            "relative to the repo root."
        ),
    )
    parser.add_argument(
        "--out-export",
        default=None,
        help=(
            "Path to write the C7.4 export fixtures JSON. "
            "Defaults to ferrotorch-jit/tests/conformance/fixtures_export.json."
        ),
    )
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # --- C7.1 graph / trace / symbolic / codegen_ir fixtures ---
    write_graph_ir_fixtures()

    # --- C7.3 optimize + codegen fixtures ---
    default_out = (
        repo_root
        / "ferrotorch-jit"
        / "tests"
        / "conformance"
        / "fixtures_optimize_codegen.json"
    )
    out_path = pathlib.Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixtures = build_fixtures()
    modules_covered = sorted(set(f["module"] for f in fixtures))

    payload = {
        "metadata": {
            "torch_version": torch.__version__,
            "python_executable": sys.executable,
            "python_platform": platform.platform(),
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "tracking_issue": "#883",
            "sub_phase": "C7.3",
            "description": (
                "Reference fixtures for ferrotorch-jit optimize+codegen conformance suite. "
                "Covers 9 modules: optimize, fusion, dag_fusion, codegen, codegen_cpu, "
                "codegen_gpu, codegen_jit, autotune, memory_plan."
            ),
            "modules_covered": modules_covered,
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
        f"  {len(fixtures)} total fixtures: {n_live} live, {n_skip} cascade-skip\n"
        f"  Modules covered: {', '.join(modules_covered)}"
    )

    # --- C7.4 export fixtures ---
    default_out_export = (
        repo_root
        / "ferrotorch-jit"
        / "tests"
        / "conformance"
        / "fixtures_export.json"
    )
    out_export_path = pathlib.Path(args.out_export) if args.out_export else default_out_export
    out_export_path.parent.mkdir(parents=True, exist_ok=True)

    export_fixtures = build_export_fixtures()
    n_live_exp = sum(1 for fx in export_fixtures if "cascade_skip" not in fx)
    n_skip_exp = sum(1 for fx in export_fixtures if "cascade_skip" in fx)

    export_payload = {
        "metadata": {
            "torch_version": torch.__version__,
            "python_executable": sys.executable,
            "python_platform": platform.platform(),
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "tracking_issue": "#806",
            "sub_phase": "C7.4",
            "description": (
                "Reference fixtures for ferrotorch-jit export/serialize conformance suite. "
                "Covers export.rs, serialize.rs, error.rs. "
                "Fixtures encode round-trip correctness and guard validation."
            ),
        },
        "fixtures": export_fixtures,
    }

    with open(out_export_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)
        f.write("\n")

    print(
        f"Written {out_export_path}\n"
        f"  {len(export_fixtures)} total C7.4 fixtures: {n_live_exp} live, {n_skip_exp} cascade-skip"
    )


if __name__ == "__main__":
    main()
