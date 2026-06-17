//! GPU code generator: emit PTX source from [`LoopIR`].
//!
//! [`GpuCodegen::generate_ptx_source`] emits PTX assembly targeting `sm_52`
//! with hand-scheduled register allocation, f32 hardware approximations, and
//! Rust-owned f64 polynomial math sequences.
//!
//! The generator converts the outermost loop of a `LoopIR` program into
//! thread-parallel GPU execution while keeping inner loops as thread-local
//! serial computation. `InductorTarget::GpuCuda` is a CUDA-driver execution
//! target, but its generated source is still PTX; this crate does not emit
//! CUDA C, call NVRTC, or require a CUDA C/C++ toolchain.
//!
//! # Dtype dispatch (#729)
//!
//! The generator dispatches on a [`Dtype`] parameter (currently `F32` or `F64`)
//! at every site where the emission differs between scalar widths: PTX
//! `.f32`/`.f64` suffixes, register declarations, load/store widths, and
//! constant literal encoding (`0f...` 8 hex digits for f32; `0d...` 16 hex
//! digits for f64).
//!
//! Transcendentals (`exp`, `log`, `sqrt`, `tanh`, `sigmoid`, `relu`, `abs`,
//! `gelu`, `silu`, `pow`) on the PTX path use direct hardware approximation
//! instructions (`ex2.approx.f32`, `lg2.approx.f32`, `rcp.approx.f32`,
//! `sqrt.approx.f32`) for f32. PTX has no `*.approx.f64` instructions, so f64
//! lowers through Rust-owned PTX sequences: `sqrt.rn.f64` for square root and
//! inline Cody-Waite / Horner f64 `exp` and `log` fragments for the composite
//! ops. The f64 path never demotes to f32 and never routes through CUDA C or
//! NVRTC.
//!
//! ## REQ status (per `.design/ferrotorch-jit/codegen_gpu.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub struct GpuCodegen`; consumer: re-export at `ferrotorch-jit/src/lib.rs:94` + `ferrotorch-jit/src/codegen.rs` `GpuCuda`/`GpuPtx` arms. |
//! | REQ-2 | SHIPPED | `pub fn generate_ptx_source`; consumer: `codegen.rs` (`GpuCuda` and `GpuPtx` arms, including identity-graph fallback). |
//! | REQ-3 | SHIPPED | per-dtype emission helpers switched on `Dtype`; consumer: every GPU emission passes the resolved group dtype. |
//! | REQ-4 | SHIPPED | f64 transcendental PTX lowering inside `emit_ptx_unary_op` / `emit_ptx_f64_*`; consumer: `codegen.rs` propagates via `.map_err(FerrotorchError::from)`. |
//! | REQ-5 | SHIPPED | PTX shared-memory tree reduction; consumer: transitively via `codegen.rs`. |
//! | REQ-6 | SHIPPED | PTX `tid` mapping + sequential `output[tid]` write pattern in `emit_ptx_elementwise_body`; consumer: transitively via `codegen.rs`. |
//! | REQ-7 | SHIPPED | `pub fn generate_ptx_source(..., block_size, ...)` parameter; consumer: `InductorBackend::with_block_size`. |

use crate::codegen_ir::{BinOpKind, Expr, LoopIR, UnaryOpKind};
use crate::error::JitError;
use crate::graph::Dtype;

/// A GPU code generator targeting PTX output.
#[derive(Debug)]
pub struct GpuCodegen;

// ===========================================================================
// PTX code generation
// ===========================================================================

impl GpuCodegen {
    /// Generate PTX assembly from a `LoopIR` program.
    ///
    /// The generated kernel maps the outermost loop to GPU threads using
    /// `ctaid.x * ntid.x + tid.x` indexing. Inner loops become serial
    /// per-thread computation.
    ///
    /// The kernel operates on values of `dtype` (currently `f32` or `f64`);
    /// arithmetic, load/store, register declarations, and constant emission
    /// all dispatch on `dtype`. F32 transcendentals use the existing hardware
    /// approximation instructions (`ex2.approx.f32`, `lg2.approx.f32`, etc.).
    /// F64 transcendentals lower to Rust-owned PTX math fragments and remain
    /// fully device-resident.
    ///
    /// # Arguments
    ///
    /// * `loops` - The loop IR to convert.
    /// * `fn_name` - The kernel entry point name.
    /// * `block_size` - The intended thread block size (used in documentation
    ///   comments; actual block size is set at launch time).
    /// * `num_inputs` - Number of input buffers.
    /// * `dtype` - Element dtype for all loads, stores, registers, and
    ///   constants emitted in the kernel.
    ///
    /// # Errors
    ///
    /// Returns [`JitError`] only for structurally unsupported IR. F64
    /// transcendentals are emitted directly as PTX and do not require the
    /// `cuda` feature because no runtime compiler is involved.
    pub fn generate_ptx_source(
        loops: &[LoopIR],
        fn_name: &str,
        block_size: usize,
        num_inputs: usize,
        dtype: Dtype,
    ) -> Result<String, JitError> {
        if let Some(reduction) = detect_ptx_reduction(loops, num_inputs) {
            return generate_ptx_reduction_source(reduction, fn_name, block_size, dtype);
        }

        let mut out = String::new();
        let dn = dtype.name(); // "f32" or "f64"
        // PTX register width and address-shift width. f32 = 4-byte stride;
        // f64 = 8-byte stride. The byte-offset for a tid is `tid << shl`.
        let shl = match dtype {
            Dtype::F32 => 2,
            Dtype::F64 => 3,
        };

        // PTX header
        out.push_str(".version 7.0\n");
        out.push_str(".target sm_52\n");
        out.push_str(".address_size 64\n\n");

        // Kernel entry point with parameters
        out.push_str(&format!(".visible .entry {fn_name}(\n"));
        for i in 0..num_inputs {
            out.push_str(&format!("    .param .u64 in{i}_ptr,\n"));
        }
        out.push_str("    .param .u64 out_ptr,\n");
        out.push_str("    .param .u32 n\n");
        out.push_str(") {\n");

        // Analyze what registers and operations we need
        let needs = analyze_ptx_needs(loops, dtype);

        // Register declarations
        out.push_str("    .reg .u32 %r_tid, %bid, %bdim, %n_reg;\n");
        out.push_str("    .reg .u64 %off;\n");
        for i in 0..num_inputs {
            out.push_str(&format!("    .reg .u64 %in{i};\n"));
        }
        out.push_str("    .reg .u64 %out;\n");
        out.push_str(&format!("    .reg .{dn} %val;\n"));
        out.push_str("    .reg .pred %p;\n");

        if needs.extra_scratch_regs > 0 {
            for r in 0..needs.extra_scratch_regs {
                out.push_str(&format!("    .reg .{dn} %t{r};\n"));
            }
        }
        if needs.needs_loop_regs {
            out.push_str("    .reg .u32 %loop_i, %loop_end;\n");
            out.push_str("    .reg .u64 %loop_off;\n");
            out.push_str(&format!("    .reg .{dn} %acc;\n"));
        }
        if needs.needs_zero {
            out.push_str(&format!("    .reg .{dn} %zero;\n"));
        }
        if needs.needs_f64_math {
            emit_ptx_f64_math_reg_decls(&mut out);
        }

        out.push('\n');

        // Load parameters
        for i in 0..num_inputs {
            out.push_str(&format!("    ld.param.u64 %in{i}, [in{i}_ptr];\n"));
        }
        out.push_str("    ld.param.u64 %out, [out_ptr];\n");
        out.push_str("    ld.param.u32 %n_reg, [n];\n\n");

        // Thread index: tid = ctaid.x * ntid.x + tid.x
        out.push_str("    mov.u32 %bid, %ctaid.x;\n");
        out.push_str("    mov.u32 %bdim, %ntid.x;\n");
        out.push_str("    mov.u32 %r_tid, %tid.x;\n");
        out.push_str("    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;\n\n");

        // Bounds check
        out.push_str("    setp.ge.u32 %p, %r_tid, %n_reg;\n");
        out.push_str("    @%p bra DONE;\n\n");

        // Compute byte offset (shift by log2(sizeof(scalar)))
        out.push_str("    cvt.u64.u32 %off, %r_tid;\n");
        out.push_str(&format!("    shl.b64 %off, %off, {shl};\n\n"));

        // Add offset to base pointers
        for i in 0..num_inputs {
            out.push_str(&format!("    add.u64 %in{i}, %in{i}, %off;\n"));
        }
        out.push_str("    add.u64 %out, %out, %off;\n\n");

        // Load input value(s)
        if num_inputs >= 1 {
            out.push_str(&format!("    ld.global.{dn} %val, [%in0];\n"));
        }

        // Zero register if needed
        if needs.needs_zero {
            // Bit-pattern `0.0` is all zeros in both IEEE 754 binary32 and
            // binary64. Width of the literal differs, however: 8 hex digits
            // (`0f...`) for f32, 16 hex digits (`0d...`) for f64.
            let zero_lit = ptx_const_literal(0.0, dtype);
            out.push_str(&format!("    mov.{dn} %zero, {zero_lit};\n"));
        }

        out.push('\n');

        // Block size hint comment
        out.push_str(&format!("    // recommended block size: {block_size}\n\n"));

        // Emit the kernel body
        emit_ptx_body(&mut out, loops, dtype);

        // Store result
        out.push_str(&format!("\n    st.global.{dn} [%out], %val;\n\n"));

        out.push_str("DONE:\n");
        out.push_str("    ret;\n");
        out.push_str("}\n");

        Ok(out)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PtxReductionKind {
    Sum,
    Mean,
    Prod,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PtxReduction {
    kind: PtxReductionKind,
}

#[derive(Default)]
struct ReductionFacts {
    saw_add_update: bool,
    saw_mul_update: bool,
    input_buffer: Option<String>,
    invalid_input_buffer: bool,
}

fn detect_ptx_reduction(loops: &[LoopIR], num_inputs: usize) -> Option<PtxReduction> {
    if num_inputs != 1 {
        return None;
    }

    let final_store = find_final_store(loops)?;
    let final_kind = match final_store {
        LoopIR::Store {
            buffer,
            index,
            value,
        } if buffer == "out" && matches!(index, Expr::IntConst(0)) => {
            if matches!(value, Expr::Var(v) if v == "acc") {
                None
            } else if matches!(
                value,
                Expr::BinOp {
                    op: BinOpKind::Div,
                    lhs,
                    rhs
                } if matches!(lhs.as_ref(), Expr::Var(v) if v == "acc")
                    && matches!(rhs.as_ref(), Expr::Const(_))
            ) {
                Some(PtxReductionKind::Mean)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let mut facts = ReductionFacts::default();
    collect_reduction_facts(loops, &mut facts);
    if facts.invalid_input_buffer || facts.input_buffer.as_deref() != Some("in0") {
        return None;
    }

    let kind = match final_kind {
        Some(PtxReductionKind::Mean) if facts.saw_add_update && !facts.saw_mul_update => {
            PtxReductionKind::Mean
        }
        None if facts.saw_add_update && !facts.saw_mul_update => PtxReductionKind::Sum,
        None if facts.saw_mul_update && !facts.saw_add_update => PtxReductionKind::Prod,
        _ => return None,
    };

    Some(PtxReduction { kind })
}

fn find_final_store(stmts: &[LoopIR]) -> Option<&LoopIR> {
    stmts.iter().rev().find(|stmt| {
        !matches!(
            stmt,
            LoopIR::Comment(_) | LoopIR::Let { .. } | LoopIR::Assign { .. }
        )
    })
}

fn collect_reduction_facts(stmts: &[LoopIR], facts: &mut ReductionFacts) {
    for stmt in stmts {
        match stmt {
            LoopIR::Loop { body, .. } => collect_reduction_facts(body, facts),
            LoopIR::Accumulate { value, .. } => {
                facts.saw_add_update = true;
                record_index_buffers(value, facts);
            }
            LoopIR::Assign { var, value } if is_self_mul_update(var, value) => {
                facts.saw_mul_update = true;
                record_index_buffers(value, facts);
            }
            LoopIR::Let { var, value } if var == "acc" => {
                if let Some(op) = reduction_combine_op(value) {
                    match op {
                        BinOpKind::Add => facts.saw_add_update = true,
                        BinOpKind::Mul => facts.saw_mul_update = true,
                        _ => {}
                    }
                }
            }
            LoopIR::If {
                then_body,
                else_body,
                ..
            } => {
                collect_reduction_facts(then_body, facts);
                collect_reduction_facts(else_body, facts);
            }
            LoopIR::Store { .. }
            | LoopIR::Let { .. }
            | LoopIR::Assign { .. }
            | LoopIR::Comment(_) => {}
        }
    }
}

fn is_self_mul_update(var: &str, value: &Expr) -> bool {
    match value {
        Expr::BinOp {
            op: BinOpKind::Mul,
            lhs,
            rhs,
        } => {
            (matches!(lhs.as_ref(), Expr::Var(v) if v == var) && expr_contains_index(rhs))
                || (matches!(rhs.as_ref(), Expr::Var(v) if v == var) && expr_contains_index(lhs))
        }
        _ => false,
    }
}

fn reduction_combine_op(expr: &Expr) -> Option<BinOpKind> {
    match expr {
        Expr::Var(name) if name == "acc" || name.starts_with("acc") => None,
        Expr::BinOp { op, lhs, rhs } if matches!(op, BinOpKind::Add | BinOpKind::Mul) => {
            let lhs_op = reduction_combine_op(lhs);
            let rhs_op = reduction_combine_op(rhs);
            match (lhs_op, rhs_op) {
                (None, None) => Some(*op),
                (Some(a), None) | (None, Some(a)) if a == *op => Some(*op),
                (Some(a), Some(b)) if a == *op && b == *op => Some(*op),
                _ => None,
            }
        }
        _ => None,
    }
}

fn expr_contains_index(expr: &Expr) -> bool {
    match expr {
        Expr::Index { .. } => true,
        Expr::BinOp { lhs, rhs, .. } => expr_contains_index(lhs) || expr_contains_index(rhs),
        Expr::UnaryOp { operand, .. } | Expr::Cast { operand, .. } => expr_contains_index(operand),
        Expr::FnCall { args, .. } => args.iter().any(expr_contains_index),
        Expr::Var(_) | Expr::Const(_) | Expr::IntConst(_) => false,
    }
}

fn record_index_buffers(expr: &Expr, facts: &mut ReductionFacts) {
    match expr {
        Expr::Index { buffer, index } => {
            match facts.input_buffer.as_deref() {
                None => facts.input_buffer = Some(buffer.clone()),
                Some(existing) if existing == buffer => {}
                Some(_) => facts.invalid_input_buffer = true,
            }
            record_index_buffers(index, facts);
        }
        Expr::BinOp { lhs, rhs, .. } => {
            record_index_buffers(lhs, facts);
            record_index_buffers(rhs, facts);
        }
        Expr::UnaryOp { operand, .. } | Expr::Cast { operand, .. } => {
            record_index_buffers(operand, facts);
        }
        Expr::FnCall { args, .. } => {
            for arg in args {
                record_index_buffers(arg, facts);
            }
        }
        Expr::Var(_) | Expr::Const(_) | Expr::IntConst(_) => {}
    }
}

fn generate_ptx_reduction_source(
    reduction: PtxReduction,
    fn_name: &str,
    block_size: usize,
    dtype: Dtype,
) -> Result<String, JitError> {
    validate_reduction_block_size(block_size)?;

    let dn = dtype.name();
    let align = scalar_align(dtype);
    let shl = scalar_shift(dtype);
    let identity = match reduction.kind {
        PtxReductionKind::Prod => ptx_const_literal(1.0, dtype),
        PtxReductionKind::Sum | PtxReductionKind::Mean => ptx_const_literal(0.0, dtype),
    };
    let empty = match reduction.kind {
        PtxReductionKind::Mean => ptx_const_literal(f64::NAN, dtype),
        PtxReductionKind::Prod => ptx_const_literal(1.0, dtype),
        PtxReductionKind::Sum => ptx_const_literal(0.0, dtype),
    };
    let combine = match reduction.kind {
        PtxReductionKind::Prod => format!("mul.{dn}"),
        PtxReductionKind::Sum | PtxReductionKind::Mean => format!("add.{dn}"),
    };
    let atomic_regs = reduction_atomic_regs(reduction.kind, dtype);
    let atomic_section = reduction_atomic_section(reduction.kind, dtype);
    let finalize = if reduction.kind == PtxReductionKind::Mean {
        mean_finalize_entry(fn_name, dtype)
    } else {
        String::new()
    };
    let init = reduction_init_entry(fn_name, dtype, &identity, &empty, reduction.kind);
    let half = block_size / 2;

    Ok(format!(
        "\
.version 7.0
.target sm_52
.address_size 64

.shared .align {align} .{dn} sdata[{block_size}];

// {fn_name}_init must be launched before {fn_name} on the same stream.
// For mean, launch {fn_name}_finalize after {fn_name}. The reduction entry
// itself performs only device-resident work: grid-stride loads, shared-memory
// block reduction, and global atomic/CAS accumulation.
{init}
.visible .entry {fn_name}(
    .param .u64 in0_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {{
    .reg .u32 %r_tid, %bid, %bdim, %gdim, %idx, %stride, %half, %n_reg, %peer;
    .reg .u64 %in, %out, %off, %addr, %sbase, %saddr;
    .reg .{dn} %acc, %other;
    .reg .pred %p, %p_tid;{atomic_regs}

    ld.param.u64 %in, [in0_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    setp.eq.u32 %p, %n_reg, 0;
    @%p bra REDUCE_DONE;

    mov.u32 %r_tid, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gdim, %nctaid.x;
    mov.u64 %sbase, sdata;

    mad.lo.u32 %idx, %bid, %bdim, %r_tid;
    mul.lo.u32 %stride, %bdim, %gdim;
    mov.{dn} %acc, {identity};

GRID_LOOP:
    setp.ge.u32 %p, %idx, %n_reg;
    @%p bra GRID_DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, {shl};
    add.u64 %addr, %in, %off;
    ld.global.{dn} %other, [%addr];
    {combine} %acc, %acc, %other;
    add.u32 %idx, %idx, %stride;
    bra GRID_LOOP;

GRID_DONE:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, {shl};
    add.u64 %saddr, %sbase, %off;
    st.shared.{dn} [%saddr], %acc;
    bar.sync 0;

    mov.u32 %half, {half};
TREE_LOOP:
    setp.eq.u32 %p, %half, 0;
    @%p bra TREE_DONE;

    setp.ge.u32 %p_tid, %r_tid, %half;
    @%p_tid bra TREE_SKIP;

    add.u32 %peer, %r_tid, %half;
    cvt.u64.u32 %off, %peer;
    shl.b64 %off, %off, {shl};
    add.u64 %saddr, %sbase, %off;
    ld.shared.{dn} %other, [%saddr];

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, {shl};
    add.u64 %saddr, %sbase, %off;
    ld.shared.{dn} %acc, [%saddr];
    {combine} %acc, %acc, %other;
    st.shared.{dn} [%saddr], %acc;

TREE_SKIP:
    bar.sync 0;
    shr.u32 %half, %half, 1;
    bra TREE_LOOP;

TREE_DONE:
    setp.ne.u32 %p_tid, %r_tid, 0;
    @%p_tid bra REDUCE_DONE;

    mov.u64 %saddr, sdata;
    ld.shared.{dn} %acc, [%saddr];
{atomic_section}

REDUCE_DONE:
    ret;
}}
{finalize}"
    ))
}

fn validate_reduction_block_size(block_size: usize) -> Result<(), JitError> {
    if !(1..=1024).contains(&block_size) || !block_size.is_power_of_two() {
        return Err(JitError::CodegenError {
            message: format!(
                "PTX reduction block_size must be a power of two in 1..=1024, got {block_size}"
            ),
        });
    }
    Ok(())
}

fn scalar_align(dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 => 4,
        Dtype::F64 => 8,
    }
}

fn scalar_shift(dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 => 2,
        Dtype::F64 => 3,
    }
}

fn reduction_init_entry(
    fn_name: &str,
    dtype: Dtype,
    identity: &str,
    empty: &str,
    kind: PtxReductionKind,
) -> String {
    let dn = dtype.name();
    let mean_init = if kind == PtxReductionKind::Mean {
        format!(
            "\
    mov.{dn} %init, {identity};
    setp.eq.u32 %p, %n_reg, 0;
    @%p mov.{dn} %init, {empty};
"
        )
    } else {
        format!("    mov.{dn} %init, {identity};\n")
    };

    format!(
        "\
.visible .entry {fn_name}_init(
    .param .u64 out_ptr,
    .param .u32 n
) {{
    .reg .u64 %out;
    .reg .u32 %n_reg;
    .reg .{dn} %init;
    .reg .pred %p;

    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
{mean_init}    st.global.{dn} [%out], %init;
    ret;
}}

"
    )
}

fn mean_finalize_entry(fn_name: &str, dtype: Dtype) -> String {
    let dn = dtype.name();
    let div_op = match dtype {
        Dtype::F32 => "div.rn.f32",
        Dtype::F64 => "div.rn.f64",
    };
    let cvt_op = match dtype {
        Dtype::F32 => "cvt.rn.f32.u32",
        Dtype::F64 => "cvt.rn.f64.u32",
    };

    format!(
        "\n\
.visible .entry {fn_name}_finalize(
    .param .u64 out_ptr,
    .param .u32 n
) {{
    .reg .u64 %out;
    .reg .u32 %n_reg;
    .reg .{dn} %sum, %count;
    .reg .pred %p;

    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    setp.eq.u32 %p, %n_reg, 0;
    @%p bra FINALIZE_DONE;

    ld.global.{dn} %sum, [%out];
    {cvt_op} %count, %n_reg;
    {div_op} %sum, %sum, %count;
    st.global.{dn} [%out], %sum;

FINALIZE_DONE:
    ret;
}}
"
    )
}

fn reduction_atomic_regs(kind: PtxReductionKind, dtype: Dtype) -> &'static str {
    match (kind, dtype) {
        (PtxReductionKind::Sum | PtxReductionKind::Mean, Dtype::F32) => {
            "\n    .reg .f32 %atomic_old;"
        }
        (_, Dtype::F32) => {
            "\n    .reg .f32 %old_val, %new_val;\n    .reg .u32 %old_bits, %assumed_bits, %new_bits;\n    .reg .pred %p_cas;"
        }
        (_, Dtype::F64) => {
            "\n    .reg .f64 %old_val, %new_val;\n    .reg .u64 %old_bits, %assumed_bits, %new_bits;\n    .reg .pred %p_cas;"
        }
    }
}

fn reduction_atomic_section(kind: PtxReductionKind, dtype: Dtype) -> String {
    match (kind, dtype) {
        (PtxReductionKind::Sum | PtxReductionKind::Mean, Dtype::F32) => {
            "    atom.global.add.f32 %atomic_old, [%out], %acc;\n".into()
        }
        (PtxReductionKind::Prod, Dtype::F32) => reduction_cas_section("mul.f32", "u32", "b32"),
        (PtxReductionKind::Sum | PtxReductionKind::Mean, Dtype::F64) => {
            reduction_cas_section("add.f64", "u64", "b64")
        }
        (PtxReductionKind::Prod, Dtype::F64) => reduction_cas_section("mul.f64", "u64", "b64"),
    }
}

fn reduction_cas_section(op: &str, int_ty: &str, bit_ty: &str) -> String {
    format!(
        "\
    ld.global.{int_ty} %old_bits, [%out];
ATOMIC_CAS_LOOP:
    mov.{int_ty} %assumed_bits, %old_bits;
    mov.{bit_ty} %old_val, %assumed_bits;
    {op} %new_val, %old_val, %acc;
    mov.{bit_ty} %new_bits, %new_val;
    atom.global.cas.{bit_ty} %old_bits, [%out], %assumed_bits, %new_bits;
    setp.ne.{int_ty} %p_cas, %old_bits, %assumed_bits;
    @%p_cas bra ATOMIC_CAS_LOOP;
"
    )
}

/// Analysis result for PTX register and instruction needs.
struct PtxNeeds {
    extra_scratch_regs: usize,
    needs_loop_regs: bool,
    needs_zero: bool,
    needs_f64_math: bool,
}

/// Analyze the loop IR to determine what PTX registers are needed.
fn analyze_ptx_needs(loops: &[LoopIR], dtype: Dtype) -> PtxNeeds {
    let mut extra = 0usize;
    let mut needs_loop = false;
    let mut needs_zero = false;
    let mut needs_f64_math = false;

    analyze_ptx_needs_recursive(
        loops,
        dtype,
        &mut extra,
        &mut needs_loop,
        &mut needs_zero,
        &mut needs_f64_math,
    );

    PtxNeeds {
        extra_scratch_regs: extra,
        needs_loop_regs: needs_loop,
        needs_zero,
        needs_f64_math,
    }
}

fn analyze_ptx_needs_recursive(
    stmts: &[LoopIR],
    dtype: Dtype,
    extra: &mut usize,
    needs_loop: &mut bool,
    needs_zero: &mut bool,
    needs_f64_math: &mut bool,
) {
    for stmt in stmts {
        match stmt {
            LoopIR::Loop { body, .. } => {
                // Inner loops in PTX need loop registers
                *needs_loop = true;
                analyze_ptx_needs_recursive(
                    body,
                    dtype,
                    extra,
                    needs_loop,
                    needs_zero,
                    needs_f64_math,
                );
            }
            LoopIR::Store { value, .. }
            | LoopIR::Assign { value, .. }
            | LoopIR::Let { value, .. }
            | LoopIR::Accumulate { value, .. } => {
                count_expr_regs(value, dtype, extra, needs_zero, needs_f64_math);
            }
            LoopIR::If {
                condition,
                then_body,
                else_body,
                ..
            } => {
                count_expr_regs(condition, dtype, extra, needs_zero, needs_f64_math);
                analyze_ptx_needs_recursive(
                    then_body,
                    dtype,
                    extra,
                    needs_loop,
                    needs_zero,
                    needs_f64_math,
                );
                analyze_ptx_needs_recursive(
                    else_body,
                    dtype,
                    extra,
                    needs_loop,
                    needs_zero,
                    needs_f64_math,
                );
            }
            LoopIR::Comment(_) => {}
        }
    }
}

fn count_expr_regs(
    expr: &Expr,
    dtype: Dtype,
    extra: &mut usize,
    needs_zero: &mut bool,
    needs_f64_math: &mut bool,
) {
    match expr {
        Expr::UnaryOp { op, operand } => {
            match op {
                UnaryOpKind::Sigmoid
                | UnaryOpKind::Tanh
                | UnaryOpKind::Gelu
                | UnaryOpKind::Silu => {
                    // These ops need scratch registers
                    *extra = (*extra).max(3);
                }
                UnaryOpKind::Relu | UnaryOpKind::Abs => {
                    *needs_zero = true;
                }
                _ => {}
            }
            if dtype == Dtype::F64
                && matches!(
                    op,
                    UnaryOpKind::Exp
                        | UnaryOpKind::Log
                        | UnaryOpKind::Sigmoid
                        | UnaryOpKind::Tanh
                        | UnaryOpKind::Gelu
                        | UnaryOpKind::Silu
                )
            {
                *needs_f64_math = true;
            }
            count_expr_regs(operand, dtype, extra, needs_zero, needs_f64_math);
        }
        Expr::BinOp { lhs, rhs, .. } => {
            *extra = (*extra).max(1);
            count_expr_regs(lhs, dtype, extra, needs_zero, needs_f64_math);
            count_expr_regs(rhs, dtype, extra, needs_zero, needs_f64_math);
        }
        Expr::FnCall { args, .. } => {
            *extra = (*extra).max(1);
            if dtype == Dtype::F64 {
                *needs_f64_math = true;
            }
            for a in args {
                count_expr_regs(a, dtype, extra, needs_zero, needs_f64_math);
            }
        }
        Expr::Index { index, .. } => {
            count_expr_regs(index, dtype, extra, needs_zero, needs_f64_math);
        }
        _ => {}
    }
}

/// Emit PTX instructions for the loop body.
///
/// The outermost loop is already handled by the thread mapping.
/// This function emits PTX for the operations inside that loop.
fn emit_ptx_body(out: &mut String, stmts: &[LoopIR], dtype: Dtype) {
    let dn = dtype.name();
    for stmt in stmts {
        match stmt {
            LoopIR::Loop { body, .. } => {
                // Outermost loop: thread-mapped, process the body directly
                emit_ptx_body(out, body, dtype);
            }
            LoopIR::Let { var, value } => {
                if var == "val" {
                    // Initial load already done above
                    match value {
                        Expr::Index { buffer, .. } => {
                            // If loading from a non-primary input, emit the load
                            if buffer != "in0"
                                && let Some(idx) = buffer.strip_prefix("in")
                                && idx.parse::<usize>().is_ok()
                            {
                                out.push_str(&format!("    ld.global.{dn} %val, [%{buffer}];\n"));
                            }
                            // else: primary input already loaded
                        }
                        _ => {
                            emit_ptx_expr_to_reg(out, value, "%val", dtype);
                        }
                    }
                } else if var == "acc" {
                    // Accumulator initialization
                    match value {
                        Expr::Const(v) => {
                            let lit = ptx_const_literal(*v, dtype);
                            out.push_str(&format!("    mov.{dn} %acc, {lit};\n"));
                        }
                        _ => {
                            emit_ptx_expr_to_reg(out, value, "%acc", dtype);
                        }
                    }
                }
            }
            LoopIR::Assign { var, value } => {
                if var == "val" {
                    emit_ptx_op(out, value, dtype);
                } else if var == "acc" {
                    emit_ptx_expr_to_reg(out, value, "%acc", dtype);
                }
            }
            LoopIR::Accumulate { var, value } if var == "acc" => {
                // Load the value into a temp, then add to acc
                emit_ptx_expr_to_reg(out, value, "%t0", dtype);
                out.push_str(&format!("    add.{dn} %acc, %acc, %t0;\n"));
            }
            LoopIR::Store { value, .. } => {
                // Store already handled by the caller (st.global.<dn>)
                // But if the value is not %val, we need to move it there
                match value {
                    Expr::Var(v) if v == "acc" => {
                        out.push_str(&format!("    mov.{dn} %val, %acc;\n"));
                    }
                    Expr::Var(v) if v == "val" => {
                        // Already in %val
                    }
                    _ => {
                        emit_ptx_expr_to_reg(out, value, "%val", dtype);
                    }
                }
            }
            LoopIR::Comment(text) => {
                out.push_str(&format!("    // {text}\n"));
            }
            _ => {}
        }
    }
}

/// Emit PTX instructions to compute an expression and put the result in the
/// specified register.
fn emit_ptx_expr_to_reg(out: &mut String, expr: &Expr, dest: &str, dtype: Dtype) {
    let dn = dtype.name();
    match expr {
        Expr::Const(v) => {
            let lit = ptx_const_literal(*v, dtype);
            out.push_str(&format!("    mov.{dn} {dest}, {lit};\n"));
        }
        Expr::Var(name) => {
            let reg = ptx_var_to_reg(name);
            if reg != dest {
                out.push_str(&format!("    mov.{dn} {dest}, {reg};\n"));
            }
        }
        Expr::Index { buffer, .. } => {
            out.push_str(&format!("    ld.global.{dn} {dest}, [%{buffer}];\n"));
        }
        Expr::BinOp { op, lhs, rhs } => {
            emit_ptx_expr_to_reg(out, lhs, dest, dtype);
            emit_ptx_expr_to_reg(out, rhs, "%t0", dtype);
            // PTX has hardware `add`/`sub`/`mul` for both .f32 and .f64.
            // `div.approx.f64` does NOT exist; use the IEEE-rounded
            // `div.rn.f64` form instead.
            let div_op = match dtype {
                Dtype::F32 => "div.approx.f32",
                Dtype::F64 => "div.rn.f64",
            };
            let cvt_rzi = match dtype {
                Dtype::F32 => "cvt.rzi.f32.f32",
                Dtype::F64 => "cvt.rzi.f64.f64",
            };
            let ptx_op = match op {
                BinOpKind::Add => format!("add.{dn}"),
                BinOpKind::Sub => format!("sub.{dn}"),
                BinOpKind::Mul => format!("mul.{dn}"),
                BinOpKind::Div => div_op.to_string(),
                BinOpKind::Mod => {
                    // PTX doesn't have a direct fmod; approximate with
                    // a - floor(a/b) * b
                    out.push_str(&format!("    {div_op} %t1, {dest}, %t0;\n"));
                    out.push_str(&format!("    {cvt_rzi} %t1, %t1;\n"));
                    out.push_str(&format!("    mul.{dn} %t1, %t1, %t0;\n"));
                    out.push_str(&format!("    sub.{dn} {dest}, {dest}, %t1;\n"));
                    return;
                }
            };
            out.push_str(&format!("    {ptx_op} {dest}, {dest}, %t0;\n"));
        }
        Expr::UnaryOp { op, operand } => {
            emit_ptx_expr_to_reg(out, operand, dest, dtype);
            emit_ptx_unary_op(out, *op, dest, dtype);
        }
        Expr::FnCall { name, args } => {
            if name == "powf" && args.len() == 2 {
                emit_ptx_expr_to_reg(out, &args[0], dest, dtype);
                match dtype {
                    Dtype::F32 => {
                        emit_ptx_expr_to_reg(out, &args[1], "%t0", dtype);
                        out.push_str(&format!("    lg2.approx.f32 %t1, {dest};\n"));
                        out.push_str("    mul.f32 %t1, %t1, %t0;\n");
                        out.push_str(&format!("    ex2.approx.f32 {dest}, %t1;\n"));
                    }
                    Dtype::F64 => {
                        if let Expr::Const(exponent) = &args[1] {
                            emit_ptx_f64_pow_const(out, dest, *exponent);
                        } else {
                            emit_ptx_expr_to_reg(out, &args[1], "%t0", dtype);
                            emit_ptx_f64_pow_dynamic(out, dest, "%t0");
                        }
                    }
                }
            } else {
                // Generic: just put the first arg in dest
                if let Some(arg) = args.first() {
                    emit_ptx_expr_to_reg(out, arg, dest, dtype);
                }
            }
        }
        _ => {}
    }
}

/// Emit the scratch-register block used by Rust-owned f64 PTX math fragments.
pub(crate) fn emit_ptx_f64_math_reg_decls(out: &mut String) {
    out.push_str(
        "    .reg .f64 %f64_saved, %f64_hold, %f64_x, %f64_y, %f64_z, %f64_tmp, %f64_tmp2, %f64_tmp3, %f64_res;\n",
    );
    out.push_str("    .reg .f64 %f64_exp_nf, %f64_exp_r, %f64_exp_p, %f64_exp_scale;\n");
    out.push_str(
        "    .reg .f64 %f64_log_m, %f64_log_f, %f64_log_f2, %f64_log_s, %f64_log_p, %f64_log_nf;\n",
    );
    out.push_str("    .reg .s32 %f64_exp_ni;\n");
    out.push_str("    .reg .s64 %f64_exp_ni64, %f64_exp_bits, %f64_log_exp64, %f64_pow_i64;\n");
    out.push_str("    .reg .u64 %f64_xbits, %f64_mantissa_bits, %f64_bias_bits, %f64_sign_bits;\n");
    out.push_str(
        "    .reg .pred %f64_p_shift, %f64_p_nan, %f64_p_zero, %f64_p_neg, %f64_p_pos_inf, %f64_p_over, %f64_p_under, %f64_p_sub, %f64_p_pos, %f64_p_sign, %f64_p_int, %f64_p_odd;\n",
    );
}

/// Format an f64 value as a PTX binary64 immediate.
pub(crate) fn ptx_f64_const_literal(v: f64) -> String {
    format!("0d{:016X}", v.to_bits())
}

fn f64_bits_eq(value: f64, expected: f64) -> bool {
    value.to_bits() == expected.to_bits()
}

fn finite_integral_i64(value: f64) -> Option<i64> {
    if !value.is_finite() || value.abs() > i64::MAX as f64 {
        return None;
    }
    let truncated = value.trunc();
    if truncated.to_bits() == value.to_bits() {
        Some(value as i64)
    } else {
        None
    }
}

/// Emit an in-place f64 `exp` approximation for `reg`.
pub(crate) fn emit_ptx_f64_exp_inplace(out: &mut String, reg: &str) {
    out.push_str(&format!("    setp.nan.f64 %f64_p_nan, {reg}, {reg};\n"));
    out.push_str(&format!(
        "    setp.gt.f64 %f64_p_over, {reg}, 0d40862E42FEFA39EF;\n"
    ));
    out.push_str(&format!(
        "    setp.lt.f64 %f64_p_under, {reg}, 0dC0874910D52D3051;\n"
    ));
    out.push_str(&format!(
        "    fma.rn.f64 %f64_exp_nf, {reg}, 0d3FF71547652B82FE, 0d3FE0000000000000;\n"
    ));
    out.push_str("    cvt.rmi.f64.f64 %f64_exp_nf, %f64_exp_nf;\n");
    out.push_str("    cvt.rni.s32.f64 %f64_exp_ni, %f64_exp_nf;\n");
    out.push_str(&format!(
        "    fma.rn.f64 %f64_exp_r, %f64_exp_nf, 0dBFE62E42FEFA3800, {reg};\n"
    ));
    out.push_str("    fma.rn.f64 %f64_exp_r, %f64_exp_nf, 0dBD2EF35793C76730, %f64_exp_r;\n");
    out.push_str("    mov.f64 %f64_exp_p, 0d3E5AE64567F544E4;\n");
    out.push_str("    fma.rn.f64 %f64_exp_p, %f64_exp_p, %f64_exp_r, 0d3E927E4FB7789F5C;\n");
    out.push_str("    fma.rn.f64 %f64_exp_p, %f64_exp_p, %f64_exp_r, 0d3EC71DE3A556C734;\n");
    out.push_str("    fma.rn.f64 %f64_exp_p, %f64_exp_p, %f64_exp_r, 0d3EFA01A01A01A01A;\n");
    out.push_str("    fma.rn.f64 %f64_exp_p, %f64_exp_p, %f64_exp_r, 0d3F2A01A01A01A01A;\n");
    out.push_str("    fma.rn.f64 %f64_exp_p, %f64_exp_p, %f64_exp_r, 0d3F56C16C16C16C17;\n");
    out.push_str("    fma.rn.f64 %f64_exp_p, %f64_exp_p, %f64_exp_r, 0d3F81111111111111;\n");
    out.push_str("    fma.rn.f64 %f64_exp_p, %f64_exp_p, %f64_exp_r, 0d3FA5555555555555;\n");
    out.push_str("    fma.rn.f64 %f64_exp_p, %f64_exp_p, %f64_exp_r, 0d3FC5555555555555;\n");
    out.push_str("    fma.rn.f64 %f64_exp_p, %f64_exp_p, %f64_exp_r, 0d3FE0000000000000;\n");
    out.push_str("    fma.rn.f64 %f64_exp_p, %f64_exp_p, %f64_exp_r, 0d3FF0000000000000;\n");
    out.push_str(&format!(
        "    fma.rn.f64 {reg}, %f64_exp_p, %f64_exp_r, 0d3FF0000000000000;\n"
    ));
    out.push_str("    cvt.s64.s32 %f64_exp_ni64, %f64_exp_ni;\n");
    out.push_str("    add.s64 %f64_exp_ni64, %f64_exp_ni64, 1023;\n");
    out.push_str("    shl.b64 %f64_exp_bits, %f64_exp_ni64, 52;\n");
    out.push_str("    mov.b64 %f64_exp_scale, %f64_exp_bits;\n");
    out.push_str(&format!("    mul.f64 {reg}, {reg}, %f64_exp_scale;\n"));
    out.push_str(&format!(
        "    @%f64_p_over mov.f64 {reg}, 0d7FF0000000000000;\n"
    ));
    out.push_str(&format!(
        "    @%f64_p_under mov.f64 {reg}, 0d0000000000000000;\n"
    ));
    out.push_str(&format!(
        "    @%f64_p_nan mov.f64 {reg}, 0d7FF8000000000000;\n"
    ));
}

/// Emit an in-place f64 natural-log approximation for `reg`.
pub(crate) fn emit_ptx_f64_log_inplace(out: &mut String, reg: &str) {
    out.push_str(&format!("    mov.f64 %f64_saved, {reg};\n"));
    out.push_str("    setp.nan.f64 %f64_p_nan, %f64_saved, %f64_saved;\n");
    out.push_str("    setp.eq.f64 %f64_p_zero, %f64_saved, 0d0000000000000000;\n");
    out.push_str("    setp.lt.f64 %f64_p_neg, %f64_saved, 0d0000000000000000;\n");
    out.push_str("    setp.eq.f64 %f64_p_pos_inf, %f64_saved, 0d7FF0000000000000;\n");
    out.push_str("    setp.gt.f64 %f64_p_pos, %f64_saved, 0d0000000000000000;\n");
    out.push_str("    mov.f64 %f64_x, %f64_saved;\n");
    out.push_str("    mov.b64 %f64_xbits, %f64_x;\n");
    out.push_str("    shr.u64 %f64_log_exp64, %f64_xbits, 52;\n");
    out.push_str("    and.b64 %f64_log_exp64, %f64_log_exp64, 2047;\n");
    out.push_str("    setp.eq.s64 %f64_p_sub, %f64_log_exp64, 0;\n");
    out.push_str("    and.pred %f64_p_sub, %f64_p_sub, %f64_p_pos;\n");
    out.push_str("    @%f64_p_sub mul.f64 %f64_x, %f64_x, 0d4330000000000000;\n");
    out.push_str("    mov.b64 %f64_xbits, %f64_x;\n");
    out.push_str("    shr.u64 %f64_log_exp64, %f64_xbits, 52;\n");
    out.push_str("    and.b64 %f64_log_exp64, %f64_log_exp64, 2047;\n");
    out.push_str("    sub.s64 %f64_log_exp64, %f64_log_exp64, 1023;\n");
    out.push_str("    @%f64_p_sub sub.s64 %f64_log_exp64, %f64_log_exp64, 52;\n");
    out.push_str("    cvt.rn.f64.s64 %f64_log_nf, %f64_log_exp64;\n");
    out.push_str("    mov.u64 %f64_bias_bits, 0x3FF0000000000000;\n");
    out.push_str("    and.b64 %f64_mantissa_bits, %f64_xbits, 0x000FFFFFFFFFFFFF;\n");
    out.push_str("    or.b64 %f64_mantissa_bits, %f64_mantissa_bits, %f64_bias_bits;\n");
    out.push_str("    mov.b64 %f64_log_m, %f64_mantissa_bits;\n");
    out.push_str("    setp.gt.f64 %f64_p_shift, %f64_log_m, 0d3FF6A09E667F3BCD;\n");
    out.push_str("    @%f64_p_shift mul.f64 %f64_log_m, %f64_log_m, 0d3FE0000000000000;\n");
    out.push_str("    @%f64_p_shift add.f64 %f64_log_nf, %f64_log_nf, 0d3FF0000000000000;\n");
    out.push_str("    sub.f64 %f64_log_f, %f64_log_m, 0d3FF0000000000000;\n");
    out.push_str("    add.f64 %f64_log_s, %f64_log_m, 0d3FF0000000000000;\n");
    out.push_str("    div.rn.f64 %f64_log_f, %f64_log_f, %f64_log_s;\n");
    out.push_str("    mul.f64 %f64_log_f2, %f64_log_f, %f64_log_f;\n");
    out.push_str("    mov.f64 %f64_log_p, 0d3FB1111111111111;\n");
    out.push_str("    fma.rn.f64 %f64_log_p, %f64_log_p, %f64_log_f2, 0d3FB3B13B13B13B14;\n");
    out.push_str("    fma.rn.f64 %f64_log_p, %f64_log_p, %f64_log_f2, 0d3FB745D1745D1746;\n");
    out.push_str("    fma.rn.f64 %f64_log_p, %f64_log_p, %f64_log_f2, 0d3FBC71C71C71C71C;\n");
    out.push_str("    fma.rn.f64 %f64_log_p, %f64_log_p, %f64_log_f2, 0d3FC2492492492492;\n");
    out.push_str("    fma.rn.f64 %f64_log_p, %f64_log_p, %f64_log_f2, 0d3FC999999999999A;\n");
    out.push_str("    fma.rn.f64 %f64_log_p, %f64_log_p, %f64_log_f2, 0d3FD5555555555555;\n");
    out.push_str("    fma.rn.f64 %f64_log_p, %f64_log_p, %f64_log_f2, 0d3FF0000000000000;\n");
    out.push_str("    mul.f64 %f64_log_p, %f64_log_p, %f64_log_f;\n");
    out.push_str("    add.f64 %f64_log_p, %f64_log_p, %f64_log_p;\n");
    out.push_str(&format!(
        "    fma.rn.f64 {reg}, %f64_log_nf, 0d3FE62E42FEFA3800, %f64_log_p;\n"
    ));
    out.push_str(&format!(
        "    fma.rn.f64 {reg}, %f64_log_nf, 0d3D2EF35793C76730, {reg};\n"
    ));
    out.push_str(&format!(
        "    @%f64_p_zero mov.f64 {reg}, 0dFFF0000000000000;\n"
    ));
    out.push_str(&format!(
        "    @%f64_p_neg mov.f64 {reg}, 0d7FF8000000000000;\n"
    ));
    out.push_str(&format!(
        "    @%f64_p_pos_inf mov.f64 {reg}, 0d7FF0000000000000;\n"
    ));
    out.push_str(&format!(
        "    @%f64_p_nan mov.f64 {reg}, 0d7FF8000000000000;\n"
    ));
}

/// Emit an in-place f64 sigmoid for `reg`.
pub(crate) fn emit_ptx_f64_sigmoid_inplace(out: &mut String, reg: &str) {
    out.push_str(&format!("    mov.f64 %f64_saved, {reg};\n"));
    out.push_str("    abs.f64 %f64_tmp, %f64_saved;\n");
    out.push_str("    neg.f64 %f64_tmp, %f64_tmp;\n");
    emit_ptx_f64_exp_inplace(out, "%f64_tmp");
    out.push_str("    add.f64 %f64_tmp2, 0d3FF0000000000000, %f64_tmp;\n");
    out.push_str("    div.rn.f64 %f64_res, 0d3FF0000000000000, %f64_tmp2;\n");
    out.push_str("    div.rn.f64 %f64_tmp, %f64_tmp, %f64_tmp2;\n");
    out.push_str("    setp.lt.f64 %f64_p_sign, %f64_saved, 0d0000000000000000;\n");
    out.push_str(&format!(
        "    selp.f64 {reg}, %f64_tmp, %f64_res, %f64_p_sign;\n"
    ));
    out.push_str("    setp.nan.f64 %f64_p_nan, %f64_saved, %f64_saved;\n");
    out.push_str(&format!(
        "    @%f64_p_nan mov.f64 {reg}, 0d7FF8000000000000;\n"
    ));
}

/// Emit an in-place f64 hyperbolic tangent for `reg`.
pub(crate) fn emit_ptx_f64_tanh_inplace(out: &mut String, reg: &str) {
    out.push_str(&format!("    mov.f64 %f64_saved, {reg};\n"));
    out.push_str("    abs.f64 %f64_tmp, %f64_saved;\n");
    out.push_str("    neg.f64 %f64_tmp, %f64_tmp;\n");
    out.push_str("    add.f64 %f64_tmp, %f64_tmp, %f64_tmp;\n");
    emit_ptx_f64_exp_inplace(out, "%f64_tmp");
    out.push_str("    sub.f64 %f64_res, 0d3FF0000000000000, %f64_tmp;\n");
    out.push_str("    add.f64 %f64_tmp2, 0d3FF0000000000000, %f64_tmp;\n");
    out.push_str("    div.rn.f64 %f64_res, %f64_res, %f64_tmp2;\n");
    out.push_str("    setp.lt.f64 %f64_p_sign, %f64_saved, 0d0000000000000000;\n");
    out.push_str(&format!(
        "    @%f64_p_sign neg.f64 %f64_res, %f64_res;\n    mov.f64 {reg}, %f64_res;\n"
    ));
    out.push_str("    setp.nan.f64 %f64_p_nan, %f64_saved, %f64_saved;\n");
    out.push_str(&format!(
        "    @%f64_p_nan mov.f64 {reg}, 0d7FF8000000000000;\n"
    ));
}

/// Emit in-place tanh-approximate f64 GELU for `reg`.
pub(crate) fn emit_ptx_f64_gelu_tanh_inplace(out: &mut String, reg: &str) {
    out.push_str(&format!("    mov.f64 %f64_hold, {reg};\n"));
    out.push_str("    mul.f64 %f64_tmp, %f64_hold, %f64_hold;\n");
    out.push_str("    mul.f64 %f64_tmp, %f64_tmp, %f64_hold;\n");
    out.push_str("    mul.f64 %f64_tmp, %f64_tmp, 0d3FA6E4E26D4801F7;\n");
    out.push_str("    add.f64 %f64_tmp, %f64_hold, %f64_tmp;\n");
    out.push_str(&format!(
        "    mul.f64 {reg}, %f64_tmp, 0d3FE9884533D43651;\n"
    ));
    emit_ptx_f64_tanh_inplace(out, reg);
    out.push_str(&format!("    add.f64 {reg}, {reg}, 0d3FF0000000000000;\n"));
    out.push_str(&format!("    mul.f64 {reg}, {reg}, 0d3FE0000000000000;\n"));
    out.push_str(&format!("    mul.f64 {reg}, {reg}, %f64_hold;\n"));
}

/// Emit in-place f64 `SiLU` for `reg`.
pub(crate) fn emit_ptx_f64_silu_inplace(out: &mut String, reg: &str) {
    out.push_str(&format!("    mov.f64 %f64_hold, {reg};\n"));
    emit_ptx_f64_sigmoid_inplace(out, reg);
    out.push_str(&format!("    mul.f64 {reg}, {reg}, %f64_hold;\n"));
}

/// Emit in-place f64 power for a compile-time scalar exponent.
pub(crate) fn emit_ptx_f64_pow_const(out: &mut String, reg: &str, exponent: f64) {
    if f64_bits_eq(exponent, 0.0) {
        out.push_str(&format!("    mov.f64 {reg}, 0d3FF0000000000000;\n"));
        return;
    }
    if f64_bits_eq(exponent, 1.0) {
        return;
    }
    out.push_str(&format!("    mov.f64 %f64_hold, {reg};\n"));
    out.push_str(&format!("    abs.f64 {reg}, {reg};\n"));
    emit_ptx_f64_log_inplace(out, reg);
    out.push_str(&format!(
        "    mul.f64 {reg}, {reg}, {};\n",
        ptx_f64_const_literal(exponent)
    ));
    emit_ptx_f64_exp_inplace(out, reg);

    if let Some(integral_exponent) = finite_integral_i64(exponent) {
        if integral_exponent.rem_euclid(2) == 1 {
            out.push_str("    mov.b64 %f64_xbits, %f64_hold;\n");
            out.push_str("    and.b64 %f64_sign_bits, %f64_xbits, 0x8000000000000000;\n");
            out.push_str("    setp.ne.u64 %f64_p_sign, %f64_sign_bits, 0;\n");
            out.push_str(&format!("    @%f64_p_sign neg.f64 {reg}, {reg};\n"));
        }
    } else {
        out.push_str("    setp.lt.f64 %f64_p_neg, %f64_hold, 0d0000000000000000;\n");
        out.push_str(&format!(
            "    @%f64_p_neg mov.f64 {reg}, 0d7FF8000000000000;\n"
        ));
    }
}

/// Emit in-place f64 power for a runtime scalar exponent.
pub(crate) fn emit_ptx_f64_pow_dynamic(out: &mut String, reg: &str, exponent_reg: &str) {
    out.push_str(&format!("    mov.f64 %f64_hold, {reg};\n"));
    out.push_str(&format!("    abs.f64 {reg}, {reg};\n"));
    emit_ptx_f64_log_inplace(out, reg);
    out.push_str(&format!("    mul.f64 {reg}, {reg}, {exponent_reg};\n"));
    emit_ptx_f64_exp_inplace(out, reg);
    out.push_str("    setp.lt.f64 %f64_p_neg, %f64_hold, 0d0000000000000000;\n");
    out.push_str(&format!(
        "    cvt.rzi.s64.f64 %f64_pow_i64, {exponent_reg};\n"
    ));
    out.push_str("    cvt.rn.f64.s64 %f64_tmp3, %f64_pow_i64;\n");
    out.push_str(&format!(
        "    setp.eq.f64 %f64_p_int, %f64_tmp3, {exponent_reg};\n"
    ));
    out.push_str("    mov.b64 %f64_xbits, %f64_pow_i64;\n");
    out.push_str("    and.b64 %f64_sign_bits, %f64_xbits, 1;\n");
    out.push_str("    setp.ne.u64 %f64_p_odd, %f64_sign_bits, 0;\n");
    out.push_str("    and.pred %f64_p_sign, %f64_p_neg, %f64_p_int;\n");
    out.push_str("    and.pred %f64_p_sign, %f64_p_sign, %f64_p_odd;\n");
    out.push_str(&format!("    @%f64_p_sign neg.f64 {reg}, {reg};\n"));
    out.push_str(&format!(
        "    setp.ne.f64 %f64_p_int, %f64_tmp3, {exponent_reg};\n"
    ));
    out.push_str("    and.pred %f64_p_neg, %f64_p_neg, %f64_p_int;\n");
    out.push_str(&format!(
        "    @%f64_p_neg mov.f64 {reg}, 0d7FF8000000000000;\n"
    ));
}

/// Emit PTX for a unary operation on a register.
fn emit_ptx_unary_op(out: &mut String, op: UnaryOpKind, reg: &str, dtype: Dtype) {
    let dn = dtype.name();
    match op {
        UnaryOpKind::Neg => {
            // Hardware `neg` exists for both .f32 and .f64.
            out.push_str(&format!("    neg.{dn} {reg}, {reg};\n"));
        }
        UnaryOpKind::Abs => {
            // Hardware `abs` exists for both .f32 and .f64.
            out.push_str(&format!("    abs.{dn} {reg}, {reg};\n"));
        }
        UnaryOpKind::Sqrt => {
            let op = match dtype {
                Dtype::F32 => "sqrt.approx.f32",
                Dtype::F64 => "sqrt.rn.f64",
            };
            out.push_str(&format!("    {op} {reg}, {reg};\n"));
        }
        UnaryOpKind::Exp => {
            match dtype {
                Dtype::F32 => {
                    // exp(x) = 2^(x * log2(e))
                    out.push_str(&format!("    mul.f32 {reg}, {reg}, 0f3FB8AA3B;\n")); // log2(e)
                    out.push_str(&format!("    ex2.approx.f32 {reg}, {reg};\n"));
                }
                Dtype::F64 => emit_ptx_f64_exp_inplace(out, reg),
            }
        }
        UnaryOpKind::Log => {
            match dtype {
                Dtype::F32 => {
                    // log(x) = log2(x) / log2(e) = log2(x) * ln(2)
                    out.push_str(&format!("    lg2.approx.f32 {reg}, {reg};\n"));
                    out.push_str(&format!("    mul.f32 {reg}, {reg}, 0f3F317218;\n")); // ln(2)
                }
                Dtype::F64 => emit_ptx_f64_log_inplace(out, reg),
            }
        }
        UnaryOpKind::Relu => {
            // `max.f32` / `max.f64` both exist as hardware ops.
            out.push_str(&format!("    max.{dn} {reg}, {reg}, %zero;\n"));
        }
        UnaryOpKind::Sigmoid => {
            match dtype {
                Dtype::F32 => {
                    // sigmoid(x) = 1 / (1 + exp(-x))
                    out.push_str(&format!("    neg.f32 %t0, {reg};\n"));
                    out.push_str("    mul.f32 %t0, %t0, 0f3FB8AA3B;\n"); // * log2(e)
                    out.push_str("    ex2.approx.f32 %t0, %t0;\n");
                    out.push_str("    add.f32 %t0, %t0, 0f3F800000;\n"); // + 1.0
                    out.push_str(&format!("    rcp.approx.f32 {reg}, %t0;\n"));
                }
                Dtype::F64 => emit_ptx_f64_sigmoid_inplace(out, reg),
            }
        }
        UnaryOpKind::Tanh => {
            match dtype {
                Dtype::F32 => {
                    // tanh(x) = 2*sigmoid(2x) - 1
                    out.push_str(&format!("    add.f32 {reg}, {reg}, {reg};\n")); // 2x
                    out.push_str(&format!("    neg.f32 %t0, {reg};\n"));
                    out.push_str("    mul.f32 %t0, %t0, 0f3FB8AA3B;\n");
                    out.push_str("    ex2.approx.f32 %t0, %t0;\n");
                    out.push_str("    add.f32 %t0, %t0, 0f3F800000;\n");
                    out.push_str(&format!("    rcp.approx.f32 {reg}, %t0;\n"));
                    out.push_str(&format!("    add.f32 {reg}, {reg}, {reg};\n")); // 2*sigmoid(2x)
                    out.push_str(&format!("    sub.f32 {reg}, {reg}, 0f3F800000;\n")); // -1
                }
                Dtype::F64 => emit_ptx_f64_tanh_inplace(out, reg),
            }
        }
        UnaryOpKind::Gelu => {
            match dtype {
                Dtype::F32 => {
                    // GELU approx: x * sigmoid(1.702 * x)
                    out.push_str(&format!("    mov.f32 %t2, {reg};\n")); // save x
                    out.push_str("    mul.f32 %t0, %t2, 0f3FD9F16C;\n"); // 1.702 * x
                    out.push_str("    neg.f32 %t0, %t0;\n");
                    out.push_str("    mul.f32 %t0, %t0, 0f3FB8AA3B;\n");
                    out.push_str("    ex2.approx.f32 %t0, %t0;\n");
                    out.push_str("    add.f32 %t0, %t0, 0f3F800000;\n");
                    out.push_str("    rcp.approx.f32 %t0, %t0;\n"); // sigmoid(1.702*x)
                    out.push_str(&format!("    mul.f32 {reg}, %t2, %t0;\n")); // x * sigmoid(1.702*x)
                }
                Dtype::F64 => emit_ptx_f64_gelu_tanh_inplace(out, reg),
            }
        }
        UnaryOpKind::Silu => {
            match dtype {
                Dtype::F32 => {
                    // SiLU: x * sigmoid(x)
                    out.push_str(&format!("    mov.f32 %t2, {reg};\n")); // save x
                    out.push_str(&format!("    neg.f32 %t0, {reg};\n"));
                    out.push_str("    mul.f32 %t0, %t0, 0f3FB8AA3B;\n");
                    out.push_str("    ex2.approx.f32 %t0, %t0;\n");
                    out.push_str("    add.f32 %t0, %t0, 0f3F800000;\n");
                    out.push_str("    rcp.approx.f32 %t0, %t0;\n"); // sigmoid(x)
                    out.push_str(&format!("    mul.f32 {reg}, %t2, %t0;\n")); // x * sigmoid(x)
                }
                Dtype::F64 => emit_ptx_f64_silu_inplace(out, reg),
            }
        }
    }
}

/// Emit PTX for a composite operation stored in an Assign to %val.
fn emit_ptx_op(out: &mut String, expr: &Expr, dtype: Dtype) {
    let dn = dtype.name();
    match expr {
        Expr::UnaryOp { op, .. } => {
            emit_ptx_unary_op(out, *op, "%val", dtype);
        }
        Expr::BinOp { op, lhs, rhs } => {
            // Binary op where lhs is typically %val
            let _ = lhs; // lhs is already in %val
            emit_ptx_expr_to_reg(out, rhs, "%t0", dtype);
            let div_op = match dtype {
                Dtype::F32 => "div.approx.f32",
                Dtype::F64 => "div.rn.f64",
            };
            let cvt_rzi = match dtype {
                Dtype::F32 => "cvt.rzi.f32.f32",
                Dtype::F64 => "cvt.rzi.f64.f64",
            };
            let ptx_op = match op {
                BinOpKind::Add => format!("add.{dn}"),
                BinOpKind::Sub => format!("sub.{dn}"),
                BinOpKind::Mul => format!("mul.{dn}"),
                BinOpKind::Div => div_op.to_string(),
                BinOpKind::Mod => {
                    out.push_str(&format!("    {div_op} %t1, %val, %t0;\n"));
                    out.push_str(&format!("    {cvt_rzi} %t1, %t1;\n"));
                    out.push_str(&format!("    mul.{dn} %t1, %t1, %t0;\n"));
                    out.push_str(&format!("    sub.{dn} %val, %val, %t1;\n"));
                    return;
                }
            };
            out.push_str(&format!("    {ptx_op} %val, %val, %t0;\n"));
        }
        Expr::FnCall { name, args } => {
            if name == "powf" && args.len() == 2 {
                match dtype {
                    Dtype::F32 => {
                        // x^p = 2^(p * log2(x)) — f32 approximate path.
                        emit_ptx_expr_to_reg(out, &args[1], "%t0", dtype);
                        out.push_str("    lg2.approx.f32 %t1, %val;\n");
                        out.push_str("    mul.f32 %t1, %t1, %t0;\n");
                        out.push_str("    ex2.approx.f32 %val, %t1;\n");
                    }
                    Dtype::F64 => {
                        if let Expr::Const(exponent) = &args[1] {
                            emit_ptx_f64_pow_const(out, "%val", *exponent);
                        } else {
                            emit_ptx_expr_to_reg(out, &args[1], "%t0", dtype);
                            emit_ptx_f64_pow_dynamic(out, "%val", "%t0");
                        }
                    }
                }
            }
        }
        _ => {
            emit_ptx_expr_to_reg(out, expr, "%val", dtype);
        }
    }
}

/// Map a variable name to a PTX register. Defaults to `%val` for any
/// name other than the reserved `"acc"` accumulator register.
fn ptx_var_to_reg(name: &str) -> &str {
    match name {
        "acc" => "%acc",
        _ => "%val",
    }
}

/// Format a scalar `v` as a PTX hex literal of the given dtype.
///
/// PTX literal forms:
/// - `f32`: `0f` prefix + 8 hex digits = 32-bit IEEE 754 binary32 bit pattern.
/// - `f64`: `0d` prefix + 16 hex digits = 64-bit IEEE 754 binary64 bit pattern.
fn ptx_const_literal(v: f64, dtype: Dtype) -> String {
    match dtype {
        // The original f32 path narrowed via `(*v as f32).to_bits()` and
        // formatted as `{:08X}`. Preserved verbatim for byte-for-byte
        // compatibility with the existing PTX output.
        Dtype::F32 => format!("0f{:08X}", (v as f32).to_bits()),
        Dtype::F64 => format!("0d{:016X}", v.to_bits()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen_ir;
    use crate::graph::IrOpKind;

    // -----------------------------------------------------------------------
    // PTX-only boundary tests (F32)
    // -----------------------------------------------------------------------

    fn assert_no_cuda_c_source(src: &str) {
        for marker in [
            "__global__",
            "#include",
            "blockIdx",
            "threadIdx",
            "float*",
            "double*",
            "__shared__",
            "atomicAdd",
            "expf(",
            "tanhf(",
            "fmaxf(",
        ] {
            assert!(
                !src.contains(marker),
                "PTX generator leaked CUDA C marker `{marker}`:\n{src}"
            );
        }
    }

    #[test]
    fn test_ptx_boundary_simple_neg() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_neg", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains(".visible .entry kernel_neg"));
        assert!(src.contains("mov.u32 %bid, %ctaid.x"));
        assert!(src.contains("mov.u32 %r_tid, %tid.x"));
        assert!(src.contains("setp.ge.u32 %p, %r_tid, %n_reg"));
        assert!(src.contains("ld.global.f32"));
        assert!(src.contains("st.global.f32"));
        assert_no_cuda_c_source(&src);
    }

    #[test]
    fn test_ptx_boundary_binary_add_inputs() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Add], &["in0", "in1"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_add", 256, 2, Dtype::F32).unwrap();

        assert!(src.contains("in0_ptr"));
        assert!(src.contains("in1_ptr"));
        assert!(src.contains("add.f32"));
        assert_no_cuda_c_source(&src);
    }

    #[test]
    fn test_ptx_boundary_sigmoid_relu_reduction_and_fusion() {
        let sigmoid = codegen_ir::lower_to_loops(&[IrOpKind::Sigmoid], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&sigmoid, "kernel_sigmoid", 256, 1, Dtype::F32)
            .unwrap();
        assert!(src.contains("ex2.approx.f32"));
        assert!(src.contains("rcp.approx.f32"));
        assert_no_cuda_c_source(&src);

        let relu = codegen_ir::lower_to_loops(&[IrOpKind::Relu], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&relu, "kernel_relu", 256, 1, Dtype::F32).unwrap();
        assert!(src.contains("max.f32"));
        assert_no_cuda_c_source(&src);

        let sum = codegen_ir::lower_to_loops(&[IrOpKind::Sum], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&sum, "kernel_sum", 256, 1, Dtype::F32).unwrap();
        assert!(src.contains(".shared .align 4 .f32 sdata"));
        assert!(src.contains("atom.global.add.f32"));
        assert_no_cuda_c_source(&src);

        let mean = codegen_ir::lower_to_loops(&[IrOpKind::Mean], &["in0"], "out", 100);
        let src =
            GpuCodegen::generate_ptx_source(&mean, "kernel_mean", 256, 1, Dtype::F32).unwrap();
        assert!(src.contains("div.rn.f32"));
        assert_no_cuda_c_source(&src);

        let fused = vec![IrOpKind::Neg, IrOpKind::Relu];
        let loops = codegen_ir::lower_to_loops(&fused, &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_fused", 256, 1, Dtype::F32).unwrap();
        assert!(src.contains("neg.f32"));
        assert!(src.contains("max.f32"));
        assert_no_cuda_c_source(&src);
    }

    #[test]
    fn test_ptx_boundary_gelu_silu() {
        let gelu = codegen_ir::lower_to_loops(&[IrOpKind::Gelu], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&gelu, "kernel_gelu", 256, 1, Dtype::F32).unwrap();
        assert!(src.contains("3FD9F16C"));
        assert!(src.contains("rcp.approx.f32"));
        assert_no_cuda_c_source(&src);

        let silu = codegen_ir::lower_to_loops(&[IrOpKind::Silu], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&silu, "kernel_silu", 256, 1, Dtype::F32).unwrap();
        assert!(src.contains("rcp.approx.f32"));
        assert_no_cuda_c_source(&src);
    }

    // -----------------------------------------------------------------------
    // PTX codegen tests (F32)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ptx_simple_neg() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_neg", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains(".version 7.0"));
        assert!(src.contains(".target sm_52"));
        assert!(src.contains(".visible .entry kernel_neg"));
        assert!(src.contains("neg.f32 %val, %val"));
        assert!(src.contains("st.global.f32 [%out], %val"));
        assert!(src.contains("ret;"));
    }

    #[test]
    fn test_ptx_relu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Relu], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_relu", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("max.f32 %val, %val, %zero"));
    }

    #[test]
    fn test_ptx_sigmoid() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sigmoid], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_sigmoid", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("ex2.approx.f32"));
        assert!(src.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_sqrt() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sqrt], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_sqrt", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("sqrt.approx.f32"));
    }

    #[test]
    fn test_ptx_exp() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Exp], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_exp", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("ex2.approx.f32"));
        assert!(src.contains("3FB8AA3B")); // log2(e) float bits
    }

    #[test]
    fn test_ptx_log() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Log], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_log", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("lg2.approx.f32"));
        assert!(src.contains("3F317218")); // ln(2) float bits
    }

    #[test]
    fn test_ptx_tanh() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Tanh], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_tanh", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("ex2.approx.f32"));
        assert!(src.contains("rcp.approx.f32"));
        assert!(src.contains("sub.f32")); // -1 step
    }

    #[test]
    fn test_ptx_fused_chain() {
        let ops = vec![IrOpKind::Neg, IrOpKind::Relu, IrOpKind::Sigmoid];
        let loops = codegen_ir::lower_to_loops(&ops, &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_fused", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("neg.f32"));
        assert!(src.contains("max.f32"));
        assert!(src.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_block_size_comment() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 4);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel", 512, 1, Dtype::F32).unwrap();
        assert!(src.contains("recommended block size: 512"));
    }

    #[test]
    fn test_ptx_multiple_inputs() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Add], &["in0", "in1"], "out", 4);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_add", 256, 2, Dtype::F32).unwrap();

        assert!(src.contains("in0_ptr"));
        assert!(src.contains("in1_ptr"));
        assert!(src.contains("%in0"));
        assert!(src.contains("%in1"));
    }

    #[test]
    fn test_ptx_gelu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Gelu], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_gelu", 256, 1, Dtype::F32).unwrap();

        // Should use GELU approximation: x * sigmoid(1.702 * x)
        assert!(src.contains("3FD9F16C")); // 1.702 float bits
        assert!(src.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_silu() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Silu], &["in0"], "out", 1024);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_silu", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_abs() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Abs], &["in0"], "out", 4);
        let src =
            GpuCodegen::generate_ptx_source(&loops, "kernel_abs", 256, 1, Dtype::F32).unwrap();

        assert!(src.contains("abs.f32"));
    }

    // -----------------------------------------------------------------------
    // F64 dispatch tests (#729)
    // -----------------------------------------------------------------------

    /// PTX: F64 elementwise neg emits native f64 registers and arithmetic,
    /// without CUDA C `double` declarations.
    #[test]
    fn test_ptx_f64_simple_neg_no_cuda_c() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_neg_f64", 256, 1, Dtype::F64)
            .expect("F64 neg should generate PTX");

        assert!(
            src.contains(".reg .f64 %val") && src.contains("neg.f64"),
            "expected native f64 PTX; got:\n{src}"
        );
        assert!(
            !src.contains("double") && !src.contains("__global__") && !src.contains("#include"),
            "F64 PTX path leaked CUDA C source:\n{src}"
        );
    }

    /// PTX: F64 add round-trip — load.f64, add.f64, store.f64, and `.reg .f64`.
    #[test]
    fn test_ptx_f64_binary_add() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Add], &["in0", "in1"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "kernel_add_f64", 256, 2, Dtype::F64)
            .expect("F64 add should generate PTX");

        assert!(
            src.contains("ld.global.f64"),
            "expected f64 load; got:\n{src}"
        );
        assert!(src.contains("add.f64"), "expected add.f64; got:\n{src}");
        assert!(
            src.contains("st.global.f64 [%out], %val"),
            "expected f64 store; got:\n{src}"
        );
        assert!(
            src.contains(".reg .f64 %val"),
            "expected f64 %val reg; got:\n{src}"
        );
        // shl by 3 (sizeof double = 8 bytes) — not 2.
        assert!(
            src.contains("shl.b64 %off, %off, 3;"),
            "expected shl by 3 for 8-byte stride; got:\n{src}"
        );
    }

    /// PTX: F64 elementwise mul/sub/div all dispatch correctly.
    #[test]
    fn test_ptx_f64_arith_dispatch() {
        // mul
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Mul], &["in0", "in1"], "out", 8);
        let src = GpuCodegen::generate_ptx_source(&loops, "k_mul", 256, 2, Dtype::F64).unwrap();
        assert!(src.contains("mul.f64"), "missing mul.f64 in:\n{src}");

        // sub
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sub], &["in0", "in1"], "out", 8);
        let src = GpuCodegen::generate_ptx_source(&loops, "k_sub", 256, 2, Dtype::F64).unwrap();
        assert!(src.contains("sub.f64"), "missing sub.f64 in:\n{src}");

        // div — uses div.rn.f64 (no .approx for f64)
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Div], &["in0", "in1"], "out", 8);
        let src = GpuCodegen::generate_ptx_source(&loops, "k_div", 256, 2, Dtype::F64).unwrap();
        assert!(src.contains("div.rn.f64"), "missing div.rn.f64 in:\n{src}");
        assert!(
            !src.contains("div.approx.f64"),
            "div.approx.f64 is invalid PTX, must not be emitted:\n{src}"
        );
    }

    /// PTX: F64 const emission uses `0d` prefix and 16 hex digits.
    #[test]
    fn test_ptx_f64_const_format() {
        // Direct unit test of the literal helper — bit-for-bit canonical
        // representation of 1.0 and 0.5 in IEEE 754 binary64.
        assert_eq!(
            ptx_const_literal(1.0_f64, Dtype::F64),
            "0d3FF0000000000000",
            "1.0 should encode as 3FF0000000000000",
        );
        assert_eq!(
            ptx_const_literal(0.5_f64, Dtype::F64),
            "0d3FE0000000000000",
            "0.5 should encode as 3FE0000000000000",
        );
        // f32 path unchanged: 1.0 encodes as 3F800000.
        assert_eq!(ptx_const_literal(1.0_f64, Dtype::F32), "0f3F800000");
        assert_eq!(ptx_const_literal(0.5_f64, Dtype::F32), "0f3F000000");
    }

    /// PTX: F64 abs uses hardware `abs.f64`, not `abs.f32` — and the
    /// `%zero` register (when present) is .f64.
    #[test]
    fn test_ptx_f64_abs_hardware() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Abs], &["in0"], "out", 4);
        let src = GpuCodegen::generate_ptx_source(&loops, "k_abs", 256, 1, Dtype::F64).unwrap();
        assert!(src.contains("abs.f64"), "expected abs.f64; got:\n{src}");
        assert!(!src.contains("abs.f32"), "F64 path leaked abs.f32:\n{src}");
    }

    /// PTX: F64 graphs containing transcendentals are emitted as Rust-owned
    /// PTX in every build. No CUDA feature, CUDA C source, or NVRTC compiler
    /// is required.
    #[test]
    fn test_ptx_f64_transcendental_succeeds() {
        for (op_kind, expected_name) in [
            (IrOpKind::Exp, "exp"),
            (IrOpKind::Log, "log"),
            (IrOpKind::Sqrt, "sqrt"),
            (IrOpKind::Tanh, "tanh"),
            (IrOpKind::Sigmoid, "sigmoid"),
            (IrOpKind::Gelu, "gelu"),
            (IrOpKind::Silu, "silu"),
        ] {
            let loops =
                codegen_ir::lower_to_loops(std::slice::from_ref(&op_kind), &["in0"], "out", 4);
            let kernel_name = format!("k_f64_{expected_name}");
            let ptx = GpuCodegen::generate_ptx_source(&loops, &kernel_name, 256, 1, Dtype::F64)
                .unwrap_or_else(|e| panic!("f64 {expected_name} PTX must succeed: {e:?}"));

            assert!(
                ptx.contains(".version"),
                "f64 {expected_name} PTX missing `.version` header:\n{ptx}",
            );
            assert!(
                ptx.contains(".target sm_52"),
                "f64 {expected_name} PTX must stay on the Rust-owned sm_52 path:\n{ptx}",
            );
            assert!(
                ptx.contains(&format!(".entry {kernel_name}")),
                "f64 {expected_name} PTX missing entry point '{kernel_name}':\n{ptx}",
            );
            assert!(
                ptx.contains("fma.rn.f64")
                    || ptx.contains("mul.f64")
                    || ptx.contains("add.f64")
                    || ptx.contains("sqrt.rn.f64")
                    || ptx.contains("div.rn.f64")
                    || ptx.contains("sub.f64"),
                "f64 {expected_name} PTX missing f64 hardware ops:\n{ptx}",
            );
            assert!(
                !ptx.contains("ex2.approx.f32")
                    && !ptx.contains("lg2.approx.f32")
                    && !ptx.contains("cvt.f32.f64")
                    && !ptx.contains("cvt.f64.f32"),
                "f64 {expected_name} PTX leaked f32 demote-promote machinery:\n{ptx}",
            );
        }
    }

    /// PTX: F64 powf via Rust-owned PTX. Separate test from the unary
    /// transcendentals because powf takes two operands and goes through the
    /// `FnCall` path rather than `UnaryOp`.
    #[test]
    fn test_ptx_f64_powf_succeeds() {
        let loops =
            codegen_ir::lower_to_loops(&[IrOpKind::Pow { exponent: 2.5 }], &["in0"], "out", 4);
        let ptx = GpuCodegen::generate_ptx_source(&loops, "k_f64_pow", 256, 1, Dtype::F64)
            .expect("f64 pow PTX must succeed");
        assert!(
            ptx.contains(".entry k_f64_pow"),
            "f64 pow PTX missing entry point:\n{ptx}",
        );
        assert!(
            ptx.contains("fma.rn.f64") || ptx.contains("mul.f64"),
            "f64 pow PTX missing f64 hardware ops:\n{ptx}",
        );
        assert!(
            !ptx.contains("ex2.approx.f32") && !ptx.contains("lg2.approx.f32"),
            "f64 pow must not use f32 approximate transcendentals:\n{ptx}",
        );
    }

    /// PTX: F64 graphs with only hardware-supported ops (neg, abs, relu, add,
    /// mul, sub, div) must succeed. Belt-and-braces against the rejection
    /// guard being too aggressive.
    #[test]
    fn test_ptx_f64_hardware_ops_succeed() {
        // neg
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Neg], &["in0"], "out", 4);
        let src = GpuCodegen::generate_ptx_source(&loops, "k", 256, 1, Dtype::F64).unwrap();
        assert!(src.contains("neg.f64"));

        // relu (hardware max.f64)
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Relu], &["in0"], "out", 4);
        let src = GpuCodegen::generate_ptx_source(&loops, "k", 256, 1, Dtype::F64).unwrap();
        assert!(src.contains("max.f64"));
    }

    /// PTX: F64 fused arithmetic chain (neg then add) — both ops dispatch
    /// to f64 in a single kernel.
    #[test]
    fn test_ptx_f64_fused_arith() {
        let ops = vec![IrOpKind::Neg, IrOpKind::Abs];
        let loops = codegen_ir::lower_to_loops(&ops, &["in0"], "out", 1024);
        let src = GpuCodegen::generate_ptx_source(&loops, "k_fused", 256, 1, Dtype::F64).unwrap();

        assert!(src.contains("neg.f64"));
        assert!(src.contains("abs.f64"));
        // No leakage of f32 paths.
        assert!(
            !src.contains(".f32"),
            "F64 fused chain leaked .f32 instruction:\n{src}"
        );
    }

    /// PTX: F64 transcendentals never lower to CUDA C math calls.
    #[test]
    fn test_ptx_f64_transcendental_has_no_cuda_c_math_calls() {
        let loops = codegen_ir::lower_to_loops(&[IrOpKind::Sigmoid], &["in0"], "out", 4);
        let src = GpuCodegen::generate_ptx_source(&loops, "k", 256, 1, Dtype::F64)
            .expect("f64 sigmoid should generate Rust-owned PTX");
        assert!(
            src.contains("fma.rn.f64"),
            "expected f64 PTX math; got:\n{src}"
        );
        assert!(!src.contains("exp("), "PTX path leaked C exp call:\n{src}");
        assert!(
            !src.contains("expf("),
            "PTX path leaked C expf call:\n{src}"
        );
    }
}
