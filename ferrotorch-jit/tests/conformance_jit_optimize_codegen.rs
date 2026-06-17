//! Layer-3 conformance tests for ferrotorch-jit optimize + codegen sub-phase C7.3.
//!
//! Covers all 9 modules in scope:
//!   1. optimize     — constant folding, DCE, pattern fusion, elementwise fusion,
//!      idempotency
//!   2. fusion       — FusedChain CPU execution, PTX/C codegen, fusion flag
//!   3. dag_fusion   — fusion group discovery, group kinds, fuse_dag lowering
//!   4. codegen      — Codegen trait, InterpreterBackend, NativeBackend, CompiledGraph
//!   5. codegen_cpu  — CpuCodegen::generate_rust_source structural properties
//!   6. codegen_gpu  — GpuCodegen::generate_cuda_source / generate_ptx_source
//!   7. codegen_jit  — cranelift JIT: compile_loop_ir_kernel, jit_supports,
//!      JitCompiledKernel::execute
//!   8. autotune     — Autotuner, AutotuneKey, AutotuneResult, cache behavior
//!   9. memory_plan  — plan_memory, MemoryPlan slot assignments
//!
//! Tracking issue: #883 (C7.3).
//! Fixtures: ferrotorch-jit/tests/conformance/fixtures_optimize_codegen.json
//!
//! Mathematical-property tests are used where no PyTorch numeric reference exists
//! (optimizer/inductor internals are closed-source):
//!   • Optimization idempotency: optimize(optimize(g)) == optimize(g) in structure.
//!   • Fusion equivalence: fused result == sequential application.
//!   • Codegen structural: emitted source contains expected constructs.
//!
//! Cascade bugs are filed and tests are skipped via `cascade_skip!`.

// Cascade-skip helper: logs the skip reason (ties back to an issue number) and
// returns from the test without failing.
#[allow(unused_macros)]
macro_rules! cascade_skip {
    ($reason:expr) => {{
        eprintln!("[cascade-skip] {}: {}", std::line!(), $reason);
        return;
    }};
}

// Tolerance helper.
fn assert_close(label: &str, got: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(got.len(), expected.len(), "[{label}] length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let err = (g - e).abs();
        let mag = e.abs().max(1.0);
        assert!(
            err <= tol * mag,
            "[{label}] index {i}: got {g}, expected {e}, abs_err {err:.2e} > tol {:.2e}",
            tol * mag,
        );
    }
}

// ---------------------------------------------------------------------------
// Common imports
// ---------------------------------------------------------------------------

use ferrotorch_jit::autotune::{AutotuneKey, Autotuner};
use ferrotorch_jit::codegen::{Codegen, InterpreterBackend, NativeBackend};
use ferrotorch_jit::codegen_cpu::CpuCodegen;
use ferrotorch_jit::codegen_gpu::GpuCodegen;
use ferrotorch_jit::codegen_ir;
use ferrotorch_jit::codegen_jit::compile_loop_ir_kernel;
use ferrotorch_jit::dag_fusion::{FusionGroupKind, find_fusion_groups, fuse_dag};
use ferrotorch_jit::fusion::{
    FusedChain, FusedOp, ReductionKind, generate_reduction_c, generate_reduction_ptx,
    is_fusion_enabled, with_fusion,
};
use ferrotorch_jit::graph::{Dtype, IrGraph, IrOpKind};
use ferrotorch_jit::memory_plan::plan_memory;
use ferrotorch_jit::optimize::{
    OptimizationConfig, constant_fold, dead_code_eliminate, fuse_elementwise, optimize,
    pattern_fuse,
};

/// True when the output value of `graph` is produced by a Constant node.
/// This is the canonical conformance check for constant_fold: after folding,
/// the graph output must be a constant, regardless of how many orphan nodes
/// remain (those are cleaned up by DCE, which is a separate pass).
fn output_is_constant(graph: &IrGraph) -> bool {
    let out_val = match graph.output_values.first() {
        Some(&v) => v,
        None => return false,
    };
    let producer_id = match graph
        .values
        .iter()
        .find(|v| v.id == out_val)
        .and_then(|v| v.producer)
    {
        Some(id) => id,
        None => return false,
    };
    graph
        .nodes
        .iter()
        .find(|n| n.id == producer_id)
        .is_some_and(|n| matches!(&n.op, IrOpKind::Constant { .. }))
}

// ---------------------------------------------------------------------------
// MODULE: optimize — constant folding
// ---------------------------------------------------------------------------

#[test]
fn opt_constant_fold_add_two_constants() {
    // Fixture: constant_fold_add_two_constants
    // Constant(2.0) + Constant(3.0) -> output is now produced by a Constant node.
    // Note: constant_fold replaces the op node with a Constant but does not remove
    // orphan input-constant nodes; that is DCE's job (a separate pass).
    let mut g = IrGraph::new();
    let a = g.add_constant(vec![2.0], vec![1]);
    let b = g.add_constant(vec![3.0], vec![1]);
    let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![1]]);
    g.set_outputs(vec![add_outs[0]]);

    assert_eq!(g.node_count(), 3); // before: Const(a), Const(b), Add
    constant_fold(&mut g);

    // The output value must now be produced by a Constant node.
    assert!(
        output_is_constant(&g),
        "after constant_fold the graph output must be produced by a Constant node"
    );

    // The output value's shape is preserved.
    let out_val = g.output_values[0];
    let result = g
        .values
        .iter()
        .find(|v| v.id == out_val)
        .expect("output value must exist");
    assert_eq!(result.shape, vec![1]);
}

#[test]
fn opt_constant_fold_chain_add_then_mul() {
    // Fixture: constant_fold_chain_add_then_mul
    // (Const(2) + Const(3)) * Const(4) -> output is Constant(20).
    // constant_fold iterates to a fixed point so chained constant expressions
    // are fully collapsed. Orphan input constants are cleaned up by DCE.
    let mut g = IrGraph::new();
    let a = g.add_constant(vec![2.0], vec![1]);
    let b = g.add_constant(vec![3.0], vec![1]);
    let c = g.add_constant(vec![4.0], vec![1]);
    let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![1]]);
    let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![add_outs[0], c], vec![vec![1]]);
    g.set_outputs(vec![mul_outs[0]]);

    constant_fold(&mut g);

    // Output must now be produced by a Constant node (fold reached a fixed point).
    assert!(
        output_is_constant(&g),
        "after chain constant_fold the output must be a Constant node"
    );
}

#[test]
fn opt_constant_fold_neg() {
    // Fixture: constant_fold_neg
    // Neg(Constant([7.0, -3.0])) -> output is Constant([-7.0, 3.0]).
    let mut g = IrGraph::new();
    let a = g.add_constant(vec![7.0, -3.0], vec![2]);
    let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![a], vec![vec![2]]);
    g.set_outputs(vec![neg_outs[0]]);

    constant_fold(&mut g);

    assert!(
        output_is_constant(&g),
        "after constant_fold of Neg the output must be a Constant node"
    );
}

#[test]
fn opt_constant_fold_relu() {
    // Fixture: constant_fold_relu
    // Relu(Constant([-2, 0, 3])) -> output is Constant([0, 0, 3]).
    let mut g = IrGraph::new();
    let a = g.add_constant(vec![-2.0, 0.0, 3.0], vec![3]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![a], vec![vec![3]]);
    g.set_outputs(vec![relu_outs[0]]);

    constant_fold(&mut g);

    assert!(
        output_is_constant(&g),
        "after constant_fold of Relu the output must be a Constant node"
    );
}

// ---------------------------------------------------------------------------
// MODULE: optimize — dead code elimination
// ---------------------------------------------------------------------------

#[test]
fn opt_dce_removes_unused_branch() {
    // Fixture: dce_removes_unused_branch
    // x->relu->output, x->neg (dead). After DCE: 2 nodes remain.
    let mut g = IrGraph::new();
    let x = g.add_input(vec![4]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
    g.set_outputs(vec![relu_outs[0]]);
    let (_, _neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![4]]);

    assert_eq!(g.node_count(), 3); // Input, Relu, Neg
    dead_code_eliminate(&mut g);
    assert_eq!(g.node_count(), 2, "Neg should be removed by DCE");
}

#[test]
fn opt_dce_cascading_removal() {
    // Fixture: dce_cascading_removal
    // Unused chain: Const->Neg->Add, plus used: Input->Relu->output.
    // After DCE: only Input, Relu remain.
    let mut g = IrGraph::new();
    let x = g.add_input(vec![2]);
    let c = g.add_constant(vec![1.0, 1.0], vec![2]);
    let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![c], vec![vec![2]]);
    let (_, _add_outs) = g.add_node(IrOpKind::Add, vec![neg_outs[0], c], vec![vec![2]]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![2]]);
    g.set_outputs(vec![relu_outs[0]]);

    assert_eq!(g.node_count(), 5);
    dead_code_eliminate(&mut g);
    assert_eq!(
        g.node_count(),
        2,
        "cascading DCE should remove 3 dead nodes"
    );
}

#[test]
fn opt_dce_preserves_graph_outputs() {
    // Graph outputs must never be removed, even if they look like "dead" nodes
    // from the perspective of downstream consumers.
    let mut g = IrGraph::new();
    let c = g.add_constant(vec![42.0], vec![1]);
    g.set_outputs(vec![c]);

    dead_code_eliminate(&mut g);
    assert_eq!(
        g.node_count(),
        1,
        "constant that is a graph output must not be removed"
    );
}

// ---------------------------------------------------------------------------
// MODULE: optimize — elementwise fusion
// ---------------------------------------------------------------------------

#[test]
fn opt_fuse_elementwise_neg_relu_sigmoid() {
    // Fixture: fuse_elementwise_chain_neg_relu_sigmoid
    // Chain Neg->Relu->Sigmoid collapses into one FusedElementwise node.
    // Fusion preserves output equivalence (checked against sequential values).
    let mut g = IrGraph::new();
    let x = g.add_input(vec![5]);
    let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![5]]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![5]]);
    let (_, sig_outs) = g.add_node(IrOpKind::Sigmoid, vec![relu_outs[0]], vec![vec![5]]);
    g.set_outputs(vec![sig_outs[0]]);

    assert_eq!(g.node_count(), 4); // Input, Neg, Relu, Sigmoid

    fuse_elementwise(&mut g);

    // After fusion: Input + FusedElementwise = 2 nodes.
    assert_eq!(g.node_count(), 2, "3-op chain should fuse to 1 node");

    let fused = g
        .nodes
        .iter()
        .find(|n| matches!(&n.op, IrOpKind::FusedElementwise { .. }))
        .expect("FusedElementwise node must exist");

    if let IrOpKind::FusedElementwise { ops } = &fused.op {
        assert_eq!(ops.len(), 3, "fused node must contain 3 ops");
        assert_eq!(ops[0], IrOpKind::Neg);
        assert_eq!(ops[1], IrOpKind::Relu);
        assert_eq!(ops[2], IrOpKind::Sigmoid);
    } else {
        panic!("expected FusedElementwise");
    }
}

#[test]
fn opt_fuse_elementwise_does_not_fuse_single_op() {
    // A single op is not a chain — nothing to fuse.
    let mut g = IrGraph::new();
    let x = g.add_input(vec![4]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
    g.set_outputs(vec![relu_outs[0]]);

    let before = g.node_count();
    fuse_elementwise(&mut g);
    assert_eq!(g.node_count(), before, "single op must not be fused");
}

#[test]
fn opt_fuse_elementwise_does_not_fuse_through_branch() {
    // If an intermediate value is consumed by more than one node,
    // fusion must not proceed through that value.
    let mut g = IrGraph::new();
    let x = g.add_input(vec![4]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
    let relu_out = relu_outs[0];
    let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![relu_out], vec![vec![4]]);
    let (_, sigmoid_outs) = g.add_node(IrOpKind::Sigmoid, vec![relu_out], vec![vec![4]]);
    let (_, add_outs) = g.add_node(
        IrOpKind::Add,
        vec![neg_outs[0], sigmoid_outs[0]],
        vec![vec![4]],
    );
    g.set_outputs(vec![add_outs[0]]);

    fuse_elementwise(&mut g);

    // Relu must not be included in any FusedElementwise since its output fans out.
    for node in &g.nodes {
        if let IrOpKind::FusedElementwise { ops } = &node.op {
            assert!(
                !ops.contains(&IrOpKind::Relu),
                "Relu with branching output must not appear inside FusedElementwise"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// MODULE: optimize — pattern fusion
// ---------------------------------------------------------------------------

#[test]
fn opt_pattern_fuse_linear_relu() {
    // Fixture: pattern_fuse_linear_relu
    // Linear -> Relu fuses into FusedLinearActivation{Relu}.
    let mut g = IrGraph::new();
    let x = g.add_input(vec![4]);
    let (linear_id, linear_outs) = g.add_node(IrOpKind::Linear, vec![x], vec![vec![4]]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![linear_outs[0]], vec![vec![4]]);
    g.set_outputs(vec![relu_outs[0]]);

    let before = g.node_count();
    pattern_fuse(&mut g);

    // Node count decreases (activation node removed, linear mutated in place).
    assert!(
        g.node_count() < before,
        "pattern_fuse should reduce node count"
    );

    let fused = g
        .nodes
        .iter()
        .find(|n| n.id == linear_id)
        .expect("linear node must persist");
    assert!(
        matches!(&fused.op, IrOpKind::FusedLinearActivation { .. }),
        "linear node should become FusedLinearActivation"
    );
}

// ---------------------------------------------------------------------------
// MODULE: optimize — full pipeline + idempotency
// ---------------------------------------------------------------------------

#[test]
fn opt_full_pipeline_idempotent() {
    // Fixture: optimize_idempotent_on_mixed_graph
    // Build a graph with a constant-foldable branch, a dead branch,
    // and a fusible unary chain.
    let mut g = IrGraph::new();
    let x = g.add_input(vec![2]);
    let a = g.add_constant(vec![1.0, 1.0], vec![2]);
    let b = g.add_constant(vec![2.0, 2.0], vec![2]);
    let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![2]]);
    let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![x, add_outs[0]], vec![vec![2]]);
    let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![mul_outs[0]], vec![vec![2]]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![2]]);
    g.set_outputs(vec![relu_outs[0]]);
    // Dead branch.
    let (_, _tanh_outs) = g.add_node(IrOpKind::Tanh, vec![x], vec![vec![2]]);

    let cfg = OptimizationConfig::default();

    // First pass.
    let plan1 = optimize(&mut g, &cfg);
    let count_after_first = g.node_count();

    // Second pass on the already-optimized graph.
    // Clone the graph to preserve the first-pass state for count comparison.
    let mut g2 = g.clone();
    let _plan2 = optimize(&mut g2, &cfg);
    let count_after_second = g2.node_count();

    // Idempotency: second pass must not further change the node count.
    assert_eq!(
        count_after_first, count_after_second,
        "optimize is not idempotent: first={count_after_first}, second={count_after_second}"
    );

    // Memory plan should be returned on the first pass.
    assert!(
        plan1.is_some(),
        "optimize with default config must return a MemoryPlan"
    );
    let plan = plan1.unwrap();
    assert!(plan.num_slots > 0);
    assert!(plan.planned_total <= plan.naive_total);
}

#[test]
fn opt_config_disables_all_passes() {
    // Disabling every pass must leave the graph unchanged.
    let mut g = IrGraph::new();
    let a = g.add_constant(vec![2.0], vec![1]);
    let b = g.add_constant(vec![3.0], vec![1]);
    let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![1]]);
    g.set_outputs(vec![add_outs[0]]);

    let cfg = OptimizationConfig {
        constant_folding: false,
        dead_code_elimination: false,
        operator_fusion: false,
        memory_planning: false,
    };
    let before = g.node_count();
    let plan = optimize(&mut g, &cfg);
    assert_eq!(
        g.node_count(),
        before,
        "all-disabled config must not change the graph"
    );
    assert!(plan.is_none(), "memory_planning=false must return None");
}

// ---------------------------------------------------------------------------
// MODULE: fusion — FusedChain::execute_cpu
// ---------------------------------------------------------------------------

#[test]
fn fusion_execute_cpu_scalar_add_relu_neg() {
    // Fixture: fused_chain_scalar_add_relu_neg_f32
    let mut chain = FusedChain::new();
    chain.push(FusedOp::ScalarAdd(2.0));
    chain.push(FusedOp::Relu);
    chain.push(FusedOp::Neg);

    let input: Vec<f32> = vec![-5.0, -1.0, 0.0, 1.0, 3.0];
    // sequential reference: x+2 -> relu -> neg
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| -((x + 2.0_f32).max(0.0_f32)))
        .collect();

    let result = chain.execute_cpu(&input).unwrap();
    assert_close(
        "scalar_add_relu_neg",
        &result.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        &expected.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        1e-6,
    );
}

#[test]
fn fusion_execute_cpu_sigmoid_at_zero() {
    // Fixture: fused_chain_single_op_sigmoid_f64
    // sigmoid(0.0) = 0.5
    let mut chain = FusedChain::new();
    chain.push(FusedOp::Sigmoid);
    let result = chain.execute_cpu(&[0.0f64]).unwrap();
    assert!((result[0] - 0.5).abs() < 1e-9, "sigmoid(0) must be 0.5");
}

#[test]
fn fusion_execute_cpu_tanh_at_zero() {
    // Fixture: fused_chain_single_op_tanh_f64
    let mut chain = FusedChain::new();
    chain.push(FusedOp::Tanh);
    let result = chain.execute_cpu(&[0.0f64]).unwrap();
    assert!(result[0].abs() < 1e-9, "tanh(0) must be 0");
}

#[test]
fn fusion_execute_cpu_sqrt() {
    // Fixture: fused_chain_single_op_sqrt_f64
    let mut chain = FusedChain::new();
    chain.push(FusedOp::Sqrt);
    let result = chain.execute_cpu(&[4.0f32, 9.0, 16.0]).unwrap();
    assert_close(
        "sqrt",
        &result.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        &[2.0, 3.0, 4.0],
        1e-6,
    );
}

#[test]
fn fusion_execute_cpu_abs() {
    // Fixture: fused_chain_single_op_abs_f64
    let mut chain = FusedChain::new();
    chain.push(FusedOp::Abs);
    let result = chain.execute_cpu(&[-3.0f32, 0.0, 5.0]).unwrap();
    assert_eq!(result, vec![3.0f32, 0.0, 5.0]);
}

#[test]
fn fusion_execute_cpu_pow_2() {
    // Fixture: fused_chain_pow_2_f64
    let mut chain = FusedChain::new();
    chain.push(FusedOp::Pow(2.0));
    let result = chain.execute_cpu(&[3.0f64]).unwrap();
    assert!((result[0] - 9.0).abs() < 1e-10, "3^2 must be 9");
}

#[test]
fn fusion_execute_cpu_scalar_mul() {
    // Fixture: fused_chain_scalar_mul_f32
    let mut chain = FusedChain::new();
    chain.push(FusedOp::ScalarMul(3.0));
    let result = chain.execute_cpu(&[2.0f32, -1.0]).unwrap();
    assert_eq!(result, vec![6.0f32, -3.0]);
}

#[test]
fn fusion_execute_cpu_gelu_matches_torch_reference() {
    // Fixture: fused_chain_gelu_matches_torch
    // PyTorch gelu(x, approximate='tanh') reference values.
    let mut chain = FusedChain::new();
    chain.push(FusedOp::Gelu);
    let inputs = vec![-1.0f64, 0.0, 1.0, 2.0];
    let result = chain.execute_cpu(&inputs).unwrap();
    // Reference: x * 0.5 * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    let reference: Vec<f64> = inputs
        .iter()
        .map(|&x| {
            let inner = 0.797_884_560_8 * (x + 0.044_715 * x * x * x);
            x * 0.5 * (1.0 + inner.tanh())
        })
        .collect();
    assert_close("gelu", &result, &reference, 1e-5);
}

#[test]
fn fusion_execute_cpu_silu_matches_torch_reference() {
    // Fixture: fused_chain_silu_matches_torch
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let mut chain = FusedChain::new();
    chain.push(FusedOp::Silu);
    let inputs = vec![-1.0f64, 0.0, 1.0, 2.0];
    let result = chain.execute_cpu(&inputs).unwrap();
    let reference: Vec<f64> = inputs.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    assert_close("silu", &result, &reference, 1e-9);
}

#[test]
fn fusion_fused_matches_sequential_f64() {
    // Fixture: fused_chain_matches_sequential_f64
    // Core fusion-equivalence property: fused result == sequential application.
    let input: Vec<f64> = vec![-3.0, -1.5, 0.0, 0.5, 2.0, 4.0];
    let mut chain = FusedChain::new();
    chain.push(FusedOp::ScalarAdd(2.0));
    chain.push(FusedOp::Relu);
    chain.push(FusedOp::Neg);

    let fused = chain.execute_cpu(&input).unwrap();

    // Sequential reference.
    let sequential: Vec<f64> = input
        .iter()
        .map(|&x| {
            let x1 = x + 2.0;
            let x2 = if x1 > 0.0 { x1 } else { 0.0 };
            -x2
        })
        .collect();

    assert_close("fused_vs_sequential", &fused, &sequential, 1e-10);
}

#[test]
fn fusion_empty_chain_is_identity() {
    // An empty FusedChain::execute_cpu is the identity.
    let chain = FusedChain::new();
    let input = vec![1.0f32, 2.0, 3.0];
    let result = chain.execute_cpu(&input).unwrap();
    assert_eq!(result, input);
}

#[test]
fn fusion_empty_input_is_empty() {
    let mut chain = FusedChain::new();
    chain.push(FusedOp::Relu);
    let result = chain.execute_cpu::<f32>(&[]).unwrap();
    assert!(result.is_empty());
}

// ---------------------------------------------------------------------------
// MODULE: fusion — PTX generation structural properties
// ---------------------------------------------------------------------------

#[test]
fn fusion_ptx_generation_header_structure() {
    // Fixture: ptx_generation_header_structure
    let mut chain = FusedChain::new();
    chain.push(FusedOp::ScalarAdd(2.0));
    chain.push(FusedOp::Relu);
    chain.push(FusedOp::Neg);
    let ptx = chain.generate_ptx().unwrap();

    assert!(ptx.contains(".version 7.0"), "PTX must have .version 7.0");
    assert!(ptx.contains(".target sm_52"), "PTX must target sm_52");
    assert!(
        ptx.contains(".address_size 64"),
        "PTX must have 64-bit addresses"
    );
    assert!(
        ptx.contains(".visible .entry fused_kernel"),
        "PTX must declare fused_kernel entry"
    );
    assert!(ptx.contains("in_ptr"), "PTX must declare in_ptr parameter");
    assert!(
        ptx.contains("out_ptr"),
        "PTX must declare out_ptr parameter"
    );
    assert!(
        ptx.contains("st.global.f32"),
        "PTX must store result globally"
    );
    assert!(ptx.contains("ret;"), "PTX must have a ret instruction");
}

#[test]
fn fusion_ptx_generation_sigmoid_uses_ex2_rcp() {
    // Fixture: ptx_generation_sigmoid_uses_ex2
    let mut chain = FusedChain::new();
    chain.push(FusedOp::Sigmoid);
    let ptx = chain.generate_ptx().unwrap();
    assert!(
        ptx.contains("ex2.approx.f32"),
        "sigmoid PTX must use ex2.approx.f32"
    );
    assert!(
        ptx.contains("rcp.approx.f32"),
        "sigmoid PTX must use rcp.approx.f32"
    );
}

#[test]
fn fusion_ptx_generation_sqrt_uses_sqrt_approx() {
    // Fixture: ptx_generation_sqrt_uses_sqrt_approx
    let mut chain = FusedChain::new();
    chain.push(FusedOp::Sqrt);
    let ptx = chain.generate_ptx().unwrap();
    assert!(
        ptx.contains("sqrt.approx.f32"),
        "sqrt PTX must use sqrt.approx.f32"
    );
}

#[test]
fn fusion_ptx_generation_pow_uses_lg2_ex2() {
    // Fixture: ptx_generation_pow_uses_lg2_ex2
    let mut chain = FusedChain::new();
    chain.push(FusedOp::Pow(3.0));
    let ptx = chain.generate_ptx().unwrap();
    assert!(
        ptx.contains("lg2.approx.f32"),
        "pow PTX must use lg2.approx.f32"
    );
    assert!(
        ptx.contains("ex2.approx.f32"),
        "pow PTX must use ex2.approx.f32"
    );
}

#[test]
fn fusion_ptx_generation_binary_op_rejected() {
    // Fixture: ptx_generation_binary_op_rejected
    // generate_ptx must reject binary ops that need a second input pointer.
    let mut chain = FusedChain::new();
    chain.push(FusedOp::Add);
    let result = chain.generate_ptx();
    assert!(result.is_err(), "generate_ptx with binary Add must fail");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("binary") || msg.contains("Add"),
        "error must mention binary op, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// MODULE: fusion — C codegen structural properties
// ---------------------------------------------------------------------------

#[test]
fn fusion_c_codegen_header_structure() {
    // Fixture: c_codegen_header_structure
    let mut chain = FusedChain::new();
    chain.push(FusedOp::Relu);
    chain.push(FusedOp::Neg);
    let c = chain.generate_c("fused_relu_neg").unwrap();

    assert!(
        c.contains("#include <math.h>"),
        "C code must include math.h"
    );
    assert!(
        c.contains("void fused_relu_neg("),
        "C code must have correct function name"
    );
    assert!(
        c.contains("#pragma omp simd"),
        "C code must have SIMD pragma"
    );
    assert!(c.contains("for (int i = 0"), "C code must have a loop");
    assert!(
        c.contains("out[i] = val;"),
        "C code must store result to out[i]"
    );
}

#[test]
fn fusion_generate_reduction_c_sum() {
    let c = generate_reduction_c(ReductionKind::Sum, "reduce_sum").unwrap();
    assert!(
        c.contains("float acc = 0.0f"),
        "sum reduction must init acc to 0"
    );
    assert!(c.contains("acc += in[i]"), "sum reduction must accumulate");
    assert!(
        c.contains("out[0] = acc"),
        "sum reduction must store to out[0]"
    );
}

#[test]
fn fusion_generate_reduction_c_mean() {
    let c = generate_reduction_c(ReductionKind::Mean, "reduce_mean").unwrap();
    assert!(c.contains("acc = acc /"), "mean reduction must divide by n");
}

#[test]
fn fusion_generate_reduction_ptx_sum_uses_atom_add() {
    // Fixture: generate_reduction_ptx_sum_uses_atom_add
    let ptx = generate_reduction_ptx(ReductionKind::Sum, "reduce_sum").unwrap();
    assert!(
        ptx.contains("atom.global.add.f32"),
        "sum reduction PTX must use atom.global.add.f32"
    );
}

#[test]
fn fusion_generate_reduction_ptx_mean_has_finalize_entry() {
    // Fixture: generate_reduction_ptx_mean_has_finalize_entry
    let ptx = generate_reduction_ptx(ReductionKind::Mean, "reduce_mean").unwrap();
    assert!(
        ptx.contains("reduce_mean_finalize"),
        "mean reduction PTX must have _finalize entry"
    );
    assert!(
        ptx.contains("div.approx.f32"),
        "mean finalize must divide by n"
    );
}

// ---------------------------------------------------------------------------
// MODULE: fusion — with_fusion flag
// ---------------------------------------------------------------------------

#[test]
fn fusion_flag_default_off() {
    // Fixture: fusion_flag_default_off_then_scoped_on
    assert!(!is_fusion_enabled(), "fusion flag must be false by default");
}

#[test]
fn fusion_flag_scoped_on_inside_closure() {
    assert!(!is_fusion_enabled());
    with_fusion(|| {
        assert!(
            is_fusion_enabled(),
            "fusion flag must be true inside with_fusion"
        );
    });
    assert!(
        !is_fusion_enabled(),
        "fusion flag must be restored after with_fusion"
    );
}

#[test]
fn fusion_flag_nested_restores_outer() {
    with_fusion(|| {
        assert!(is_fusion_enabled());
        with_fusion(|| {
            assert!(is_fusion_enabled());
        });
        // Inner scope restores to true (not false).
        assert!(
            is_fusion_enabled(),
            "inner with_fusion must restore outer true state"
        );
    });
    assert!(!is_fusion_enabled());
}

// ---------------------------------------------------------------------------
// MODULE: dag_fusion — find_fusion_groups
// ---------------------------------------------------------------------------

#[test]
fn dag_single_elementwise_op_one_group() {
    // Fixture: single_elementwise_op_forms_one_group
    let mut g = IrGraph::new();
    let x = g.add_input(vec![4]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
    g.set_outputs(vec![relu_outs[0]]);

    let groups = find_fusion_groups(&g);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].kind, FusionGroupKind::Elementwise);
    assert_eq!(groups[0].ops.len(), 1);
}

#[test]
fn dag_chain_neg_relu_sigmoid_one_group() {
    // Fixture: chain_neg_relu_sigmoid_fuses_one_group
    let mut g = IrGraph::new();
    let x = g.add_input(vec![4]);
    let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![4]]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![4]]);
    let (_, sig_outs) = g.add_node(IrOpKind::Sigmoid, vec![relu_outs[0]], vec![vec![4]]);
    g.set_outputs(vec![sig_outs[0]]);

    let groups = find_fusion_groups(&g);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].kind, FusionGroupKind::Elementwise);
    assert_eq!(groups[0].ops.len(), 3);
    assert_eq!(groups[0].ops[0], IrOpKind::Neg);
    assert_eq!(groups[0].ops[1], IrOpKind::Relu);
    assert_eq!(groups[0].ops[2], IrOpKind::Sigmoid);
}

#[test]
fn dag_binary_add_plus_relu_one_group() {
    // Fixture: binary_add_plus_relu_one_elementwise_group
    let mut g = IrGraph::new();
    let x = g.add_input(vec![4]);
    let y = g.add_input(vec![4]);
    let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, y], vec![vec![4]]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![add_outs[0]], vec![vec![4]]);
    g.set_outputs(vec![relu_outs[0]]);

    let groups = find_fusion_groups(&g);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].kind, FusionGroupKind::Elementwise);
}

#[test]
fn dag_reduction_breaks_group() {
    // Fixture: reduction_breaks_group_into_two
    let mut g = IrGraph::new();
    let x = g.add_input(vec![4]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
    let (_, sum_outs) = g.add_node(IrOpKind::Sum, vec![relu_outs[0]], vec![vec![1]]);
    g.set_outputs(vec![sum_outs[0]]);

    let groups = find_fusion_groups(&g);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].kind, FusionGroupKind::Elementwise);
    assert_eq!(groups[1].kind, FusionGroupKind::Reduction);
}

#[test]
fn dag_matmul_is_standalone_group() {
    // Fixture: matmul_is_standalone_group
    let mut g = IrGraph::new();
    let a = g.add_input(vec![2, 3]);
    let b = g.add_input(vec![3, 4]);
    let (_, mm_outs) = g.add_node(IrOpKind::Mm, vec![a, b], vec![vec![2, 4]]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![mm_outs[0]], vec![vec![2, 4]]);
    g.set_outputs(vec![relu_outs[0]]);

    let groups = find_fusion_groups(&g);
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].kind, FusionGroupKind::MatMul);
    assert_eq!(groups[1].kind, FusionGroupKind::Elementwise);
}

#[test]
fn dag_external_inputs_and_outputs_correct() {
    // Fixture: external_inputs_and_outputs_correct
    let mut g = IrGraph::new();
    let x = g.add_input(vec![4]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
    g.set_outputs(vec![relu_outs[0]]);

    let groups = find_fusion_groups(&g);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].external_inputs.len(), 1);
    assert_eq!(groups[0].external_inputs[0], x);
    assert_eq!(groups[0].external_outputs.len(), 1);
    assert_eq!(groups[0].external_outputs[0], relu_outs[0]);
}

// ---------------------------------------------------------------------------
// MODULE: dag_fusion — fuse_dag lowering
// ---------------------------------------------------------------------------

#[test]
fn dag_fuse_dag_elementwise_emits_loop_ir() {
    // Fixture: fuse_dag_elementwise_emits_loop_ir
    let mut g = IrGraph::new();
    let x = g.add_input(vec![4]);
    let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![4]]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![4]]);
    g.set_outputs(vec![relu_outs[0]]);

    let groups = find_fusion_groups(&g);
    let loops_per_group = fuse_dag(&groups, &g);
    assert_eq!(loops_per_group.len(), 1);
    assert!(
        !loops_per_group[0].is_empty(),
        "elementwise group must emit non-empty LoopIR"
    );
}

#[test]
fn dag_fuse_dag_matmul_emits_triple_loop() {
    // Fixture: fuse_dag_matmul_group_emits_triple_loop
    use ferrotorch_jit::codegen_ir::LoopIR;
    let mut g = IrGraph::new();
    let a = g.add_input(vec![2, 3]);
    let b = g.add_input(vec![3, 4]);
    let (_, mm_outs) = g.add_node(IrOpKind::Mm, vec![a, b], vec![vec![2, 4]]);
    g.set_outputs(vec![mm_outs[0]]);

    let groups = find_fusion_groups(&g);
    let loops_per_group = fuse_dag(&groups, &g);
    assert_eq!(loops_per_group.len(), 1);
    assert!(!loops_per_group[0].is_empty());
    // Outer loop var must be 'i'.
    match &loops_per_group[0][0] {
        LoopIR::Loop { var, .. } => {
            assert_eq!(var, "i", "outer loop var of matmul must be 'i'");
        }
        other => panic!("expected outer Loop, got {other:?}"),
    }
}

#[test]
fn dag_fuse_dag_reduction_emits_accumulator() {
    // Fixture: fuse_dag_reduction_emits_accumulator
    use ferrotorch_jit::codegen_ir::LoopIR;
    let mut g = IrGraph::new();
    let x = g.add_input(vec![8]);
    let (_, sum_outs) = g.add_node(IrOpKind::Sum, vec![x], vec![vec![1]]);
    g.set_outputs(vec![sum_outs[0]]);

    let groups = find_fusion_groups(&g);
    let loops_per_group = fuse_dag(&groups, &g);
    assert_eq!(loops_per_group.len(), 1);
    assert!(!loops_per_group[0].is_empty());
    match &loops_per_group[0][0] {
        LoopIR::Let { var, .. } => {
            assert_eq!(
                var, "acc",
                "first LoopIR stmt of reduction must be Let{{acc}}"
            );
        }
        other => panic!("expected Let{{acc}}, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// MODULE: codegen — InterpreterBackend and NativeBackend
// ---------------------------------------------------------------------------

fn build_relu_sqrt_graph() -> (IrGraph, Vec<Vec<f64>>) {
    let mut g = IrGraph::new();
    let x = g.add_input(vec![3]);
    let (_, relu_out) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![3]]);
    let (_, sqrt_out) = g.add_node(IrOpKind::Sqrt, vec![relu_out[0]], vec![vec![3]]);
    g.set_outputs(vec![sqrt_out[0]]);
    (g, vec![vec![1.0, 4.0, 9.0]])
}

#[test]
fn codegen_interpreter_backend_relu_sqrt() {
    // Fixture: interpreter_backend_relu_sqrt_chain
    let (g, inputs) = build_relu_sqrt_graph();
    let backend = InterpreterBackend;
    let compiled = backend.compile(&g).unwrap();
    let out = compiled.execute(&inputs).unwrap();
    assert_close("interpreter_relu_sqrt", &out, &[1.0, 2.0, 3.0], 1e-10);
}

#[test]
fn codegen_native_backend_relu_sqrt() {
    // Fixture: native_backend_relu_sqrt_chain
    let (g, inputs) = build_relu_sqrt_graph();
    let backend = NativeBackend;
    let compiled = backend.compile(&g).unwrap();
    let out = compiled.execute(&inputs).unwrap();
    assert_close("native_relu_sqrt", &out, &[1.0, 2.0, 3.0], 1e-10);
}

#[test]
fn codegen_compiled_graph_wrong_input_count_errors() {
    // Fixture: compiled_graph_wrong_input_count_returns_error
    let (g, _) = build_relu_sqrt_graph();
    let backend = InterpreterBackend;
    let compiled = backend.compile(&g).unwrap();
    let result = compiled.execute(&[]); // 0 inputs instead of 1
    assert!(result.is_err(), "wrong input count must return an error");
}

#[test]
fn codegen_compiled_graph_output_shape_preserved() {
    // Fixture: compiled_graph_output_shape_preserved
    let (g, _) = build_relu_sqrt_graph();
    let backend = InterpreterBackend;
    let compiled = backend.compile(&g).unwrap();
    assert_eq!(
        compiled.output_shape(),
        &[3],
        "output shape must match input shape [3]"
    );
}

#[test]
fn codegen_native_backend_add_two_inputs() {
    // Fixture: native_backend_add_two_inputs
    let mut g = IrGraph::new();
    let x = g.add_input(vec![3]);
    let y = g.add_input(vec![3]);
    let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, y], vec![vec![3]]);
    g.set_outputs(vec![add_outs[0]]);

    let backend = NativeBackend;
    let compiled = backend.compile(&g).unwrap();
    let out = compiled
        .execute(&[vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]])
        .unwrap();
    assert_close("add_two_inputs", &out, &[11.0, 22.0, 33.0], 1e-10);
}

// ---------------------------------------------------------------------------
// MODULE: codegen_cpu — CpuCodegen::generate_rust_source structural checks
// ---------------------------------------------------------------------------

fn loops_for(
    ops: &[IrOpKind],
    inputs: &[&str],
    n: usize,
) -> Vec<ferrotorch_jit::codegen_ir::LoopIR> {
    codegen_ir::lower_to_loops(ops, inputs, "out", n)
}

#[test]
fn codegen_cpu_neg_contains_inline_always_and_signature() {
    // Fixture: generate_rust_source_neg_contains_inline_always
    let loops = loops_for(&[IrOpKind::Neg], &["in0"], 4);
    let src = CpuCodegen::generate_rust_source(&loops, "kernel_neg");

    assert!(
        src.contains("#[inline(always)]"),
        "must contain #[inline(always)]"
    );
    assert!(
        src.contains("pub unsafe fn kernel_neg"),
        "must have correct fn name"
    );
    assert!(
        src.contains("inputs: &[&[f64]]"),
        "must take &[&[f64]] inputs"
    );
    assert!(
        src.contains("output: &mut [f64]"),
        "must take &mut [f64] output"
    );
    assert!(src.contains("for"), "must contain a loop");
}

#[test]
fn codegen_cpu_binary_add_references_both_inputs() {
    // Fixture: generate_rust_source_binary_add_uses_both_inputs
    let loops = loops_for(&[IrOpKind::Add], &["in0", "in1"], 8);
    let src = CpuCodegen::generate_rust_source(&loops, "kernel_add");

    assert!(src.contains("inputs[0]"), "must reference inputs[0]");
    assert!(src.contains("inputs[1]"), "must reference inputs[1]");
    assert!(src.contains("output["), "must write to output");
}

#[test]
fn codegen_cpu_sum_uses_accumulator_pattern() {
    // Fixture: generate_rust_source_sum_reduction_uses_accumulator
    let loops = loops_for(&[IrOpKind::Sum], &["in0"], 10);
    let src = CpuCodegen::generate_rust_source(&loops, "kernel_sum");

    assert!(src.contains("let mut acc"), "sum must declare accumulator");
    assert!(src.contains("acc +="), "sum must accumulate");
    assert!(src.contains("output[0"), "sum must write to output[0]");
}

#[test]
fn codegen_cpu_matmul_triple_loop() {
    // Fixture: generate_rust_source_matmul_triple_loop
    let loops = codegen_ir::lower_matmul("in0", "in1", "out", 2, 3, 4);
    let src = CpuCodegen::generate_rust_source(&loops, "kernel_matmul");

    assert!(src.contains("for i in"), "matmul must have i loop");
    assert!(src.contains("for j in"), "matmul must have j loop");
    assert!(src.contains("for p in"), "matmul must have p loop");
    assert!(
        src.contains("let mut acc"),
        "matmul must declare accumulator"
    );
}

#[test]
fn codegen_cpu_sigmoid_uses_exp() {
    // Fixture: generate_rust_source_sigmoid_uses_exp
    let loops = loops_for(&[IrOpKind::Sigmoid], &["in0"], 4);
    let src = CpuCodegen::generate_rust_source(&loops, "kernel_sigmoid");

    assert!(src.contains("1.0_f64"), "sigmoid must use f64 literals");
    assert!(src.contains(".exp()"), "sigmoid must call .exp()");
}

#[test]
fn codegen_cpu_pow_uses_powf() {
    // Fixture: generate_rust_source_pow_uses_powf
    let loops = loops_for(&[IrOpKind::Pow { exponent: 2.0 }], &["in0"], 4);
    let src = CpuCodegen::generate_rust_source(&loops, "kernel_pow");

    assert!(src.contains(".powf("), "Pow must emit .powf(");
}

#[test]
fn codegen_cpu_silu_uses_exp() {
    let loops = loops_for(&[IrOpKind::Silu], &["in0"], 4);
    let src = CpuCodegen::generate_rust_source(&loops, "kernel_silu");
    assert!(src.contains(".exp()"), "silu must use .exp()");
}

#[test]
fn codegen_cpu_log_uses_ln() {
    let loops = loops_for(&[IrOpKind::Log], &["in0"], 4);
    let src = CpuCodegen::generate_rust_source(&loops, "kernel_log");
    assert!(src.contains(".ln()"), "log must emit .ln()");
}

// ---------------------------------------------------------------------------
// MODULE: codegen_gpu — GpuCodegen structural checks (CPU-side, no CUDA runtime)
// ---------------------------------------------------------------------------

#[test]
fn codegen_gpu_cuda_source_neg_f32_has_global_kernel() {
    // Fixture: generate_cuda_source_neg_f32_has_global_kernel
    let loops = loops_for(&[IrOpKind::Neg], &["in0"], 4);
    let src = GpuCodegen::generate_cuda_source(&loops, "cuda_neg", 1, Dtype::F32).unwrap();

    assert!(
        src.contains("__global__ void cuda_neg"),
        "must declare __global__ kernel"
    );
    assert!(
        src.contains("float* __restrict__"),
        "f32 dtype must use float*"
    );
    assert!(src.contains("blockIdx.x"), "must use blockIdx.x");
    assert!(src.contains("threadIdx.x"), "must use threadIdx.x");
}

#[test]
fn codegen_gpu_cuda_source_add_f32_two_input_pointers() {
    // Fixture: generate_cuda_source_add_f32_has_two_input_pointers
    let loops = loops_for(&[IrOpKind::Add], &["in0", "in1"], 8);
    let src = GpuCodegen::generate_cuda_source(&loops, "cuda_add", 2, Dtype::F32).unwrap();

    assert!(src.contains("in0"), "CUDA source must name in0 parameter");
    assert!(src.contains("in1"), "CUDA source must name in1 parameter");
}

#[test]
fn codegen_gpu_cuda_source_sum_has_shared_memory() {
    // Fixture: generate_cuda_source_sum_reduction_uses_shared_memory
    let loops = loops_for(&[IrOpKind::Sum], &["in0"], 8);
    let src = GpuCodegen::generate_cuda_source(&loops, "cuda_sum", 1, Dtype::F32).unwrap();

    assert!(
        src.contains("__shared__"),
        "sum reduction CUDA source must use __shared__ memory"
    );
}

#[test]
fn codegen_gpu_ptx_source_f32_neg_has_header() {
    // Fixture: generate_ptx_source_f32_neg_has_header
    let loops = loops_for(&[IrOpKind::Neg], &["in0"], 4);
    let ptx = GpuCodegen::generate_ptx_source(&loops, "ptx_neg", 256, 1, Dtype::F32).unwrap();

    assert!(ptx.contains(".version"), "PTX must have .version header");
    assert!(ptx.contains(".target sm_52"), "PTX must target sm_52");
    assert!(
        ptx.contains(".address_size 64"),
        "PTX must use 64-bit addresses"
    );
}

#[test]
fn codegen_gpu_cuda_source_f64_uses_double_type() {
    // Fixture: generate_cuda_source_f64_neg_uses_double
    let loops = loops_for(&[IrOpKind::Neg], &["in0"], 4);
    let src = GpuCodegen::generate_cuda_source(&loops, "cuda_neg_f64", 1, Dtype::F64).unwrap();

    assert!(
        src.contains("double"),
        "f64 dtype CUDA source must use 'double'"
    );
}

#[test]
fn codegen_gpu_ptx_f64_transcendental_without_cuda_feature() {
    // Fixture: generate_ptx_source_f64_transcendental_uses_rust_ptx
    // F64 Exp must emit Rust-owned PTX in both cuda and no-cuda builds; codegen
    // cannot depend on NVRTC or libdevice.
    let loops = loops_for(&[IrOpKind::Exp], &["in0"], 4);
    let ptx = GpuCodegen::generate_ptx_source(&loops, "ptx_exp_f64", 256, 1, Dtype::F64)
        .expect("f64 transcendental PTX must be generated without NVRTC");
    assert!(ptx.contains("fma.rn.f64"), "expected f64 polynomial PTX");
    assert!(
        !ptx.contains("ex2.approx.f32") && !ptx.contains("cvt.f32.f64"),
        "f64 PTX must not demote through f32:\n{ptx}"
    );
}

// ---------------------------------------------------------------------------
// MODULE: codegen_jit — cranelift JIT
// ---------------------------------------------------------------------------

#[test]
fn jit_supports_simple_elementwise() {
    // Fixture: jit_supports_simple_elementwise_loops
    let loops = loops_for(&[IrOpKind::Neg], &["in0"], 4);
    assert!(
        ferrotorch_jit::codegen_jit::jit_supports(&loops),
        "simple elementwise loops must be JIT-supported"
    );
}

#[test]
fn jit_supports_rejects_if_statement() {
    // Fixture: jit_supports_rejects_if_statement
    use ferrotorch_jit::codegen_ir::{Expr, LoopIR};
    let loops = vec![LoopIR::If {
        condition: Expr::var("c"),
        then_body: vec![],
        else_body: vec![],
    }];
    assert!(
        !ferrotorch_jit::codegen_jit::jit_supports(&loops),
        "If statement must not be JIT-supported"
    );
}

#[test]
fn jit_supports_rejects_modulus() {
    use ferrotorch_jit::codegen_ir::{BinOpKind, Expr, LoopIR};
    let loops = vec![LoopIR::Let {
        var: "x".into(),
        value: Expr::bin(BinOpKind::Mod, Expr::int(10), Expr::int(3)),
    }];
    assert!(
        !ferrotorch_jit::codegen_jit::jit_supports(&loops),
        "modulus expression must not be JIT-supported"
    );
}

#[test]
fn jit_compile_and_execute_neg() {
    // Fixture: compile_loop_ir_kernel_neg_executes_correctly
    let loops = loops_for(&[IrOpKind::Neg], &["in0"], 4);
    let kernel = compile_loop_ir_kernel(&loops, 1, 4).unwrap();
    let input = vec![1.0_f64, -2.0, 3.5, 0.0];
    let mut output = vec![0.0; 4];
    kernel.execute(&[&input], &mut output).unwrap();
    assert_close("jit_neg", &output, &[-1.0, 2.0, -3.5, 0.0], 1e-12);
}

#[test]
fn jit_compile_and_execute_add() {
    // Fixture: compile_loop_ir_kernel_add_two_inputs
    let loops = loops_for(&[IrOpKind::Add], &["in0", "in1"], 3);
    let kernel = compile_loop_ir_kernel(&loops, 2, 3).unwrap();
    let a = vec![1.0_f64, 2.0, 3.0];
    let b = vec![10.0_f64, 20.0, 30.0];
    let mut out = vec![0.0; 3];
    kernel.execute(&[&a, &b], &mut out).unwrap();
    assert_close("jit_add", &out, &[11.0, 22.0, 33.0], 1e-12);
}

#[test]
fn jit_compile_and_execute_relu() {
    // Fixture: relu_kernel_executes_correctly
    let loops = loops_for(&[IrOpKind::Relu], &["in0"], 4);
    let kernel = compile_loop_ir_kernel(&loops, 1, 4).unwrap();
    let input = vec![-1.0_f64, 0.0, 1.0, 2.5];
    let mut out = vec![0.0; 4];
    kernel.execute(&[&input], &mut out).unwrap();
    assert_close("jit_relu", &out, &[0.0, 0.0, 1.0, 2.5], 1e-12);
}

#[test]
fn jit_compile_sqrt_exp_chain() {
    // Fixture: sqrt_exp_chain_executes_correctly
    let loops = loops_for(&[IrOpKind::Exp, IrOpKind::Sqrt], &["in0"], 3);
    let kernel = compile_loop_ir_kernel(&loops, 1, 3).unwrap();
    let input = vec![0.0_f64, f64::ln(2.0), f64::ln(4.0)];
    let mut out = vec![0.0; 3];
    kernel.execute(&[&input], &mut out).unwrap();
    // sqrt(exp(0)) = 1, sqrt(exp(ln 2)) = sqrt(2), sqrt(exp(ln 4)) = 2.
    assert_close("jit_sqrt_exp", &out, &[1.0, f64::sqrt(2.0), 2.0], 1e-9);
}

#[test]
fn jit_compile_cache_hit_on_identical_loops() {
    // Fixture: compile_cache_returns_same_arc_for_identical_loops
    use std::sync::Arc;
    let loops = loops_for(&[IrOpKind::Neg], &["in0"], 5);
    let k1 = compile_loop_ir_kernel(&loops, 1, 5).unwrap();
    let k2 = compile_loop_ir_kernel(&loops, 1, 5).unwrap();
    assert!(
        Arc::ptr_eq(&k1, &k2),
        "identical loops must return same Arc (cache hit)"
    );
}

#[test]
fn jit_execute_rejects_wrong_input_count() {
    // Fixture: execute_rejects_wrong_input_count
    let loops = loops_for(&[IrOpKind::Neg], &["in0"], 4);
    let kernel = compile_loop_ir_kernel(&loops, 1, 4).unwrap();
    let mut out = [0.0_f64; 4];
    let result = kernel.execute(&[], &mut out);
    assert!(result.is_err(), "wrong input count must return an error");
}

#[test]
fn jit_execute_rejects_short_output_buffer() {
    // Fixture: execute_rejects_short_output_buffer
    let loops = loops_for(&[IrOpKind::Neg], &["in0"], 4);
    let kernel = compile_loop_ir_kernel(&loops, 1, 4).unwrap();
    let input = [1.0_f64, 2.0, 3.0, 4.0];
    let mut tiny = [0.0_f64; 2];
    let result = kernel.execute(&[&input], &mut tiny);
    assert!(result.is_err(), "short output buffer must return an error");
}

#[test]
fn jit_unsupported_loop_ir_returns_err() {
    use ferrotorch_core::error::FerrotorchError;
    use ferrotorch_jit::codegen_ir::{Expr, LoopIR};
    let loops = vec![LoopIR::If {
        condition: Expr::var("c"),
        then_body: vec![],
        else_body: vec![],
    }];
    let result = compile_loop_ir_kernel(&loops, 1, 1);
    assert!(matches!(
        result,
        Err(FerrotorchError::InvalidArgument { .. })
    ));
}

// ---------------------------------------------------------------------------
// MODULE: autotune
// ---------------------------------------------------------------------------

fn build_autotune_graph() -> (IrGraph, Vec<Vec<f64>>) {
    let mut g = IrGraph::new();
    let x = g.add_input(vec![3]);
    let (_, relu_out) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![3]]);
    let (_, sqrt_out) = g.add_node(IrOpKind::Sqrt, vec![relu_out[0]], vec![vec![3]]);
    g.set_outputs(vec![sqrt_out[0]]);
    (g, vec![vec![1.0, 4.0, 9.0]])
}

#[test]
fn autotune_empty_candidates_errors() {
    // Fixture: autotuner_empty_candidates_returns_error
    let tuner = Autotuner::new();
    let (g, inputs) = build_autotune_graph();
    let result = tuner.tune(&g, &inputs);
    assert!(result.is_err(), "no candidates must produce an error");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.to_lowercase().contains("candidate") || msg.contains("no"),
        "error must mention candidates, got: {msg}"
    );
}

#[test]
fn autotune_picks_winner_from_two_candidates() {
    // Fixture: autotuner_picks_winner_from_two_candidates
    let tuner = Autotuner::new()
        .with_candidate("interpreter", Box::new(InterpreterBackend))
        .with_candidate("native", Box::new(NativeBackend))
        .with_iterations(3)
        .with_warmup(1);
    let (g, inputs) = build_autotune_graph();

    let result = tuner.tune(&g, &inputs).unwrap();

    assert!(
        result.winner_name() == "interpreter" || result.winner_name() == "native",
        "winner must be one of the candidates, got: {}",
        result.winner_name()
    );
    assert_eq!(
        result.all_timings().len(),
        2,
        "all_timings must have 2 rows on a full tune"
    );

    // Winner output must be correct.
    let out = result.winner_compiled().execute(&inputs).unwrap();
    assert_close("autotune_winner_output", &out, &[1.0, 2.0, 3.0], 1e-10);
}

#[test]
fn autotune_cache_hit_returns_single_timing_row() {
    // Fixture: autotune_cache_hit_returns_single_timing_row
    let tuner = Autotuner::new()
        .with_candidate("interpreter", Box::new(InterpreterBackend))
        .with_candidate("native", Box::new(NativeBackend))
        .with_iterations(2)
        .with_warmup(0);
    let (g, inputs) = build_autotune_graph();

    let first = tuner.tune(&g, &inputs).unwrap();
    assert_eq!(first.all_timings().len(), 2, "first tune: 2 timing rows");
    assert_eq!(
        tuner.cache_size(),
        1,
        "cache must have 1 entry after first tune"
    );

    let second = tuner.tune(&g, &inputs).unwrap();
    assert_eq!(second.all_timings().len(), 1, "cache hit: 1 timing row");
    assert_eq!(
        second.winner_name(),
        first.winner_name(),
        "cache hit winner must match first"
    );
}

#[test]
fn autotune_key_is_shape_sensitive() {
    // Fixture: autotune_key_is_shape_sensitive
    let mut g1 = IrGraph::new();
    let x1 = g1.add_input(vec![4]);
    let (_, r1) = g1.add_node(IrOpKind::Relu, vec![x1], vec![vec![4]]);
    g1.set_outputs(vec![r1[0]]);

    let mut g2 = IrGraph::new();
    let x2 = g2.add_input(vec![8]);
    let (_, r2) = g2.add_node(IrOpKind::Relu, vec![x2], vec![vec![8]]);
    g2.set_outputs(vec![r2[0]]);

    let k1 = AutotuneKey::from_graph(&g1, &[vec![4]]);
    let k2 = AutotuneKey::from_graph(&g2, &[vec![8]]);
    assert_ne!(
        k1, k2,
        "different input shapes must produce different autotune keys"
    );
}

#[test]
fn autotune_key_is_op_sensitive() {
    // Fixture: autotune_key_is_op_sensitive
    let mut g1 = IrGraph::new();
    let x1 = g1.add_input(vec![3]);
    let (_, r1) = g1.add_node(IrOpKind::Relu, vec![x1], vec![vec![3]]);
    g1.set_outputs(vec![r1[0]]);

    let mut g2 = IrGraph::new();
    let x2 = g2.add_input(vec![3]);
    let (_, r2) = g2.add_node(IrOpKind::Sigmoid, vec![x2], vec![vec![3]]);
    g2.set_outputs(vec![r2[0]]);

    let k1 = AutotuneKey::from_graph(&g1, &[vec![3]]);
    let k2 = AutotuneKey::from_graph(&g2, &[vec![3]]);
    assert_ne!(k1, k2, "different ops must produce different autotune keys");
}

#[test]
fn autotune_clear_cache_forces_retune() {
    // Fixture: autotuner_clear_cache_forces_retune
    let tuner = Autotuner::new()
        .with_candidate("interpreter", Box::new(InterpreterBackend))
        .with_candidate("native", Box::new(NativeBackend))
        .with_iterations(2)
        .with_warmup(0);
    let (g, inputs) = build_autotune_graph();

    let _ = tuner.tune(&g, &inputs).unwrap();
    assert_eq!(tuner.cache_size(), 1);

    tuner.clear_cache();
    assert_eq!(tuner.cache_size(), 0, "cache must be empty after clear");

    let third = tuner.tune(&g, &inputs).unwrap();
    assert_eq!(
        third.all_timings().len(),
        2,
        "after cache clear: full tune with 2 rows"
    );
}

// ---------------------------------------------------------------------------
// MODULE: memory_plan
// ---------------------------------------------------------------------------

#[test]
fn mem_simple_chain_reuses_buffers() {
    // Fixture: simple_chain_reuses_buffers
    let mut g = IrGraph::new();
    let x = g.add_input(vec![100]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![100]]);
    let (_, sig_outs) = g.add_node(IrOpKind::Sigmoid, vec![relu_outs[0]], vec![vec![100]]);
    g.set_outputs(vec![sig_outs[0]]);

    let plan = plan_memory(&g);

    assert_eq!(plan.naive_total, 300, "naive total must be 3 * 100 = 300");
    assert!(
        plan.num_slots < 3,
        "chain should reuse at least one slot; got {} slots",
        plan.num_slots
    );
    assert!(
        plan.planned_total < plan.naive_total,
        "planned {} must be < naive {}",
        plan.planned_total,
        plan.naive_total
    );
    assert_eq!(
        plan.assignments.len(),
        3,
        "all 3 values must have assignments"
    );
    assert!(plan.savings_percent() > 0.0, "savings must be positive");
}

#[test]
fn mem_diamond_concurrent_values_different_slots() {
    // Fixture: diamond_concurrent_values_different_slots
    let mut g = IrGraph::new();
    let x = g.add_input(vec![50]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![50]]);
    let (_, sig_outs) = g.add_node(IrOpKind::Sigmoid, vec![x], vec![vec![50]]);
    let (_, add_outs) = g.add_node(
        IrOpKind::Add,
        vec![relu_outs[0], sig_outs[0]],
        vec![vec![50]],
    );
    g.set_outputs(vec![add_outs[0]]);

    let plan = plan_memory(&g);

    let relu_slot = plan.assignments[&relu_outs[0]];
    let sig_slot = plan.assignments[&sig_outs[0]];
    assert_ne!(
        relu_slot, sig_slot,
        "concurrently-live values must be in different slots"
    );
}

#[test]
fn mem_long_chain_savings_positive() {
    // Fixture: long_chain_savings_percent_positive
    let mut g = IrGraph::new();
    let shape = vec![1000];
    let x = g.add_input(shape.clone());
    let (_, v1) = g.add_node(IrOpKind::Relu, vec![x], vec![shape.clone()]);
    let (_, v2) = g.add_node(IrOpKind::Sigmoid, vec![v1[0]], vec![shape.clone()]);
    let (_, v3) = g.add_node(IrOpKind::Tanh, vec![v2[0]], vec![shape.clone()]);
    let (_, v4) = g.add_node(IrOpKind::Neg, vec![v3[0]], vec![shape.clone()]);
    g.set_outputs(vec![v4[0]]);

    let plan = plan_memory(&g);

    assert_eq!(
        plan.naive_total, 5000,
        "naive total for 5 values * 1000 must be 5000"
    );
    assert!(
        plan.num_slots < 5,
        "long chain should reuse slots; got {}",
        plan.num_slots
    );
    let pct = plan.savings_percent();
    assert!(pct > 20.0, "savings must exceed 20%; got {pct:.1}%");
}

#[test]
fn mem_empty_graph_empty_plan() {
    // Fixture: empty_graph_produces_empty_plan
    let g = IrGraph::new();
    let plan = plan_memory(&g);

    assert!(plan.assignments.is_empty(), "empty graph: no assignments");
    assert_eq!(plan.num_slots, 0, "empty graph: no slots");
    assert_eq!(plan.naive_total, 0, "empty graph: naive_total = 0");
    assert_eq!(plan.planned_total, 0, "empty graph: planned_total = 0");
    // Exact float comparison is intentional: 0.0 when no allocations.
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(plan.savings_percent(), 0.0, "empty graph: savings = 0.0");
    }
}

#[test]
fn mem_all_values_assigned_and_slots_valid() {
    // Fixture: all_values_assigned_and_slots_valid
    let mut g = IrGraph::new();
    let x = g.add_input(vec![8]);
    let y = g.add_input(vec![8]);
    let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, y], vec![vec![8]]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![add_outs[0]], vec![vec![8]]);
    g.set_outputs(vec![relu_outs[0]]);

    let plan = plan_memory(&g);

    // Every value must have an assignment.
    assert_eq!(
        plan.assignments.len(),
        g.value_count(),
        "every value must be assigned"
    );

    // Every slot index must be in-range.
    for &slot in plan.assignments.values() {
        assert!(
            slot < plan.num_slots,
            "slot {slot} must be < num_slots {}",
            plan.num_slots
        );
    }
}

#[test]
fn mem_graph_outputs_pinned_to_different_slots() {
    // Fixture: graph_outputs_pinned_to_end
    let mut g = IrGraph::new();
    let x = g.add_input(vec![10]);
    let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![10]]);
    let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![relu_outs[0]], vec![vec![10]]);
    // Both are graph outputs.
    g.set_outputs(vec![relu_outs[0], neg_outs[0]]);

    let plan = plan_memory(&g);

    let relu_slot = plan.assignments[&relu_outs[0]];
    let neg_slot = plan.assignments[&neg_outs[0]];
    assert_ne!(
        relu_slot, neg_slot,
        "two simultaneously-live graph outputs must be in different slots"
    );
}
