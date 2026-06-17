//! Operation fusion engine connecting the JIT's `fuse_elementwise` pass to GPU
//! kernel generation.
//!
//! The fusion engine intercepts tensor operations, buffers sequences of
//! elementwise ops, and executes them as a single fused operation on the CPU
//! via [`FusedChain::execute_cpu`]. Inspectable Rust source generation is
//! available through [`FusedChain::generate_rust`] and
//! [`generate_reduction_rust`]. PTX code generators are also provided:
//! [`FusedChain::generate_ptx`] / [`FusedChain::generate_ptx_named`] for f32
//! and [`FusedChain::generate_ptx_f64_named`] for f64. The f64 path emits
//! Rust-owned PTX math fragments directly; it does not compile CUDA C.
//!
//! Tensor-level [`apply_fused`] runs on the CPU by default and — when the
//! `cuda` feature is enabled — dispatches CUDA inputs through
//! [`fusion_gpu::apply_fused_gpu`](crate::fusion_gpu::apply_fused_gpu),
//! which loads the chain's Rust-generated PTX, caches the resulting
//! `CudaFunction` in `ferrotorch-gpu::module_cache`, launches it on the
//! input's device, and returns a device-resident Tensor. Without the `cuda`
//! feature, CUDA inputs return
//! [`FerrotorchError::NotImplementedOnCuda`](ferrotorch_core::error::FerrotorchError::NotImplementedOnCuda)
//! with an op message asking the caller to build with `--features cuda`.
//!
//! # Thread-local fusion context
//!
//! Fusion is opt-in. Call [`with_fusion`] to enable it for a closure:
//!
//! ```ignore
//! use ferrotorch_jit::fusion::with_fusion;
//! let result = with_fusion(|| {
//!     // elementwise ops inside here are eligible for fusion
//! });
//! ```
//!
//! Use [`is_fusion_enabled`] to query the current state.
//!
//! ## REQ status (per `.design/ferrotorch-jit/fusion.md`)
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | REQ-1 | SHIPPED | `pub enum FusedOp`; consumer: re-export at `ferrotorch-jit/src/lib.rs:103-107` + `ferrotorch-jit/src/fusion_gpu.rs:38` `use crate::fusion::FusedChain;` (which is `Vec<FusedOp>`). |
//! | REQ-2 | SHIPPED | `pub struct FusedChain`; consumer: re-export at `lib.rs:103-107` + `fusion_gpu.rs:38` (every `apply_fused_gpu` takes `&FusedChain`). |
//! | REQ-3 | SHIPPED | `pub fn execute_cpu` on `impl FusedChain`; consumer: `pub fn apply_fused` in this file calls `chain.execute_cpu(data)?` in the CPU arm. |
//! | REQ-4 | SHIPPED | `pub fn generate_ptx` + `pub fn generate_ptx_named`; consumer: `fusion_gpu.rs:117` `let ptx = chain.generate_ptx_named(FUSED_F32_KERNEL_NAME)?;` in the f32 GPU path. |
//! | REQ-5 | SHIPPED | `pub fn generate_ptx_f64_named`; consumer: `fusion_gpu.rs` f64 GPU path. |
//! | REQ-6 | SHIPPED | `pub fn generate_rust`; inspectable Rust source artifact for fused unary chains. |
//! | REQ-7 | SHIPPED | `pub fn apply_fused`; consumer: re-export at `lib.rs:103-107` — canonical tensor-level entry point. |
//! | REQ-8 | SHIPPED | `pub fn with_fusion` + `pub fn is_fusion_enabled` + thread-local `FUSION_ENABLED`; consumer: re-export at `lib.rs:103-107`. |
//! | REQ-9 | SHIPPED | `pub enum ReductionKind` + `pub fn generate_reduction_rust` + `pub fn generate_reduction_ptx`; consumer: re-export at `lib.rs:103-107`. |
//! | REQ-10 | SHIPPED | `fn validate_identifier`; consumer: invoked by every public emitter in this file (`generate_ptx_named`, `generate_ptx_f64_named`, `generate_rust`, `generate_reduction_rust`, `generate_reduction_ptx`). |
//! | REQ-11 | SHIPPED | `pub fn estimate_numel_for_inputs` + `pub fn estimate_matmul_dims`; consumer: re-export at `lib.rs:103-107`. |

use std::cell::Cell;
use std::fmt;

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::FerrotorchResult;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use num_traits;

use crate::codegen_gpu::{
    GpuCodegen, emit_ptx_f64_exp_inplace, emit_ptx_f64_gelu_tanh_inplace, emit_ptx_f64_log_inplace,
    emit_ptx_f64_math_reg_decls, emit_ptx_f64_pow_const, emit_ptx_f64_sigmoid_inplace,
    emit_ptx_f64_silu_inplace, emit_ptx_f64_tanh_inplace, ptx_f64_const_literal,
};
use crate::codegen_ir::{BinOpKind, Expr, LoopIR};
use crate::graph::Dtype;

// ---------------------------------------------------------------------------
// Thread-local fusion flag
// ---------------------------------------------------------------------------

thread_local! {
    static FUSION_ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Returns `true` when the current thread is inside a [`with_fusion`] scope.
pub fn is_fusion_enabled() -> bool {
    FUSION_ENABLED.with(std::cell::Cell::get)
}

/// Execute `f` with operation fusion enabled on the current thread.
///
/// Any pending fused operations are flushed before this function returns.
/// The fusion flag is always restored to its prior state, even on panic.
pub fn with_fusion<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    struct Guard {
        prev: bool,
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            FUSION_ENABLED.with(|flag| flag.set(self.prev));
        }
    }

    let prev = is_fusion_enabled();
    FUSION_ENABLED.with(|flag| flag.set(true));
    let _guard = Guard { prev };

    f()
}

// ---------------------------------------------------------------------------
// Fused operation types
// ---------------------------------------------------------------------------

/// An individual operation in a fused chain.
#[derive(Debug, Clone, PartialEq)]
pub enum FusedOp {
    // Binary elementwise (applied with a second tensor or broadcast scalar).
    /// Element-wise addition with a second operand.
    Add,
    /// Element-wise subtraction with a second operand.
    Sub,
    /// Element-wise multiplication with a second operand.
    Mul,
    /// Element-wise division with a second operand.
    Div,

    // Unary elementwise.
    /// Arithmetic negation `-x`.
    Neg,
    /// Rectified linear unit `max(0, x)`.
    Relu,
    /// Logistic sigmoid `1 / (1 + exp(-x))`.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
    /// Gaussian error linear unit.
    Gelu,
    /// Sigmoid-weighted linear unit (`x * sigmoid(x)`).
    Silu,
    /// Element-wise square root.
    Sqrt,
    /// Element-wise absolute value.
    Abs,
    /// Element-wise natural exponential.
    Exp,
    /// Element-wise natural logarithm.
    Log,

    // Parameterised unary ops.
    /// Element-wise power with a fixed `f64` exponent.
    Pow(f64),
    /// Multiply each element by an `f64` constant captured at trace time.
    ScalarMul(f64),
    /// Add an `f64` constant captured at trace time to each element.
    ScalarAdd(f64),
}

impl fmt::Display for FusedOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusedOp::Add => write!(f, "add"),
            FusedOp::Sub => write!(f, "sub"),
            FusedOp::Mul => write!(f, "mul"),
            FusedOp::Div => write!(f, "div"),
            FusedOp::Neg => write!(f, "neg"),
            FusedOp::Relu => write!(f, "relu"),
            FusedOp::Sigmoid => write!(f, "sigmoid"),
            FusedOp::Tanh => write!(f, "tanh"),
            FusedOp::Gelu => write!(f, "gelu"),
            FusedOp::Silu => write!(f, "silu"),
            FusedOp::Sqrt => write!(f, "sqrt"),
            FusedOp::Abs => write!(f, "abs"),
            FusedOp::Exp => write!(f, "exp"),
            FusedOp::Log => write!(f, "log"),
            FusedOp::Pow(p) => write!(f, "pow({p})"),
            FusedOp::ScalarMul(s) => write!(f, "scalar_mul({s})"),
            FusedOp::ScalarAdd(s) => write!(f, "scalar_add({s})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------

/// Kind of reduction operation for kernel generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionKind {
    /// Sum reduction: identity = 0, op = +.
    Sum,
    /// Product reduction: identity = 1, op = *.
    Prod,
    /// Mean reduction: identity = 0, op = +, then divide by n.
    Mean,
}

impl fmt::Display for ReductionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReductionKind::Sum => write!(f, "sum"),
            ReductionKind::Prod => write!(f, "prod"),
            ReductionKind::Mean => write!(f, "mean"),
        }
    }
}

// ---------------------------------------------------------------------------
// FusedChain
// ---------------------------------------------------------------------------

/// A sequence of elementwise operations that will be executed as a single
/// fused kernel.
///
/// On the CPU the operations are applied in-place over a single pass per
/// element. On the GPU, [`generate_ptx`](FusedChain::generate_ptx) emits a
/// single PTX kernel that chains all operations per-thread, avoiding
/// intermediate memory traffic.
#[derive(Debug, Clone)]
pub struct FusedChain {
    ops: Vec<FusedOp>,
}

impl FusedChain {
    /// Create an empty chain.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Append an operation to the chain.
    pub fn push(&mut self, op: FusedOp) {
        self.ops.push(op);
    }

    /// The number of operations in this chain.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Borrow the operations slice.
    pub fn ops(&self) -> &[FusedOp] {
        &self.ops
    }

    // -------------------------------------------------------------------
    // CPU execution
    // -------------------------------------------------------------------

    /// Execute the entire fused chain on the CPU, applying every operation
    /// in sequence over a single allocation.
    ///
    /// The input slice is copied once; all operations mutate the copy in
    /// place so only one allocation is needed regardless of chain length.
    ///
    /// # Errors
    ///
    /// Returns an error if the chain contains unsupported binary ops
    /// (Add/Sub/Mul/Div) that require a second operand.
    pub fn execute_cpu<T: Float>(&self, input: &[T]) -> FerrotorchResult<Vec<T>> {
        let mut data = input.to_vec();
        for op in &self.ops {
            apply_op_inplace::<T>(op, &mut data)?;
        }
        Ok(data)
    }

    // -------------------------------------------------------------------
    // PTX generation
    // -------------------------------------------------------------------

    /// Generate a PTX kernel string that applies every operation in this
    /// chain per-element on the GPU.
    ///
    /// The generated kernel signature is:
    ///
    /// ```text
    /// .visible .entry fused_kernel(
    ///     .param .u64 in_ptr,
    ///     .param .u64 out_ptr,
    ///     .param .u32 n
    /// )
    /// ```
    ///
    /// It reads one f32 per thread from `in_ptr`, applies the chain of
    /// operations, and stores the result to `out_ptr`. This means *one*
    /// kernel launch replaces N separate launches, eliminating all
    /// intermediate global-memory round-trips.
    pub fn generate_ptx(&self) -> FerrotorchResult<String> {
        self.generate_ptx_named("fused_kernel")
    }

    /// Like [`generate_ptx`](Self::generate_ptx) but with a custom kernel
    /// entry-point name. The name is validated to be a legal C/PTX
    /// identifier (`[a-zA-Z_][a-zA-Z0-9_]*`).
    pub fn generate_ptx_named(&self, kernel_name: &str) -> FerrotorchResult<String> {
        validate_identifier(kernel_name)?;
        // Reject unsupported binary ops that require a second input pointer.
        for op in &self.ops {
            if matches!(
                op,
                FusedOp::Add | FusedOp::Sub | FusedOp::Mul | FusedOp::Div
            ) {
                return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
                    message: format!(
                        "generate_ptx: binary op '{op}' in unary FusedChain requires a second \
                         input pointer and cannot be lowered to a single-input PTX kernel"
                    ),
                });
            }
        }
        let mut body_lines: Vec<String> = Vec::new();

        // We accumulate the running value in %val. Some ops need scratch
        // registers; we define them at the top of the kernel.
        let needs_exp = self.ops.iter().any(|op| {
            matches!(
                op,
                FusedOp::Sigmoid | FusedOp::Tanh | FusedOp::Gelu | FusedOp::Silu | FusedOp::Exp
            )
        });
        let _needs_log = self.ops.iter().any(|op| matches!(op, FusedOp::Log));
        let needs_mul_scratch = self.ops.iter().any(|op| {
            matches!(
                op,
                FusedOp::Sigmoid
                    | FusedOp::Tanh
                    | FusedOp::Gelu
                    | FusedOp::Silu
                    | FusedOp::Exp
                    | FusedOp::Log
                    | FusedOp::ScalarMul(_)
                    | FusedOp::ScalarAdd(_)
                    | FusedOp::Pow(_)
            )
        });

        // Register declarations.
        let mut reg_decls = String::from(
            "    .reg .u32 %r_tid, %bid, %bdim, %n_reg;\n\
             \x20   .reg .u64 %in, %out, %off;\n\
             \x20   .reg .f32 %val;\n\
             \x20   .reg .pred %p;",
        );
        if needs_exp {
            reg_decls.push_str("\n    .reg .f32 %exp_tmp, %tmp;");
        }
        if needs_mul_scratch {
            reg_decls.push_str("\n    .reg .f32 %scratch;");
        }
        // relu/abs need a zero constant
        let needs_zero = self
            .ops
            .iter()
            .any(|op| matches!(op, FusedOp::Relu | FusedOp::Abs));
        if needs_zero {
            reg_decls.push_str("\n    .reg .f32 %zero;");
        }

        // Emit the operation body.
        // Binary ops (Add/Sub/Mul/Div) are rejected in the early validation
        // above and will never reach this match.
        for op in &self.ops {
            match op {
                FusedOp::Add | FusedOp::Sub | FusedOp::Mul | FusedOp::Div => {
                    unreachable!("binary ops rejected by early validation");
                }
                FusedOp::Neg => {
                    body_lines.push("    neg.f32 %val, %val;".into());
                }
                FusedOp::Relu => {
                    body_lines.push("    mov.f32 %zero, 0f00000000;".into());
                    body_lines.push("    max.f32 %val, %val, %zero;".into());
                }
                FusedOp::Sigmoid => {
                    // sigmoid(x) = 1 / (1 + exp(-x))
                    body_lines.push("    neg.f32 %tmp, %val;".into());
                    body_lines.push("    // approx exp via ex2: exp(x) = 2^(x * log2(e))".into());
                    body_lines.push("    mul.f32 %scratch, %tmp, 0f3FB8AA3B;".into()); // log2(e) ~ 1.4427
                    body_lines.push("    ex2.approx.f32 %exp_tmp, %scratch;".into());
                    body_lines.push("    add.f32 %scratch, %exp_tmp, 0f3F800000;".into()); // 1.0
                    body_lines.push("    rcp.approx.f32 %val, %scratch;".into());
                }
                FusedOp::Tanh => {
                    // tanh(x) = 2*sigmoid(2x) - 1
                    body_lines.push("    add.f32 %val, %val, %val;".into()); // 2x
                    body_lines.push("    neg.f32 %tmp, %val;".into());
                    body_lines.push("    mul.f32 %scratch, %tmp, 0f3FB8AA3B;".into());
                    body_lines.push("    ex2.approx.f32 %exp_tmp, %scratch;".into());
                    body_lines.push("    add.f32 %scratch, %exp_tmp, 0f3F800000;".into());
                    body_lines.push("    rcp.approx.f32 %val, %scratch;".into()); // sigmoid(2x)
                    body_lines.push("    add.f32 %val, %val, %val;".into()); // 2*sigmoid(2x)
                    body_lines.push("    sub.f32 %val, %val, 0f3F800000;".into()); // -1
                }
                FusedOp::Gelu => {
                    // GELU tanh approx: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    // Step 1: compute x^3 in %scratch
                    body_lines.push("    mul.f32 %scratch, %val, %val;".into()); // x^2
                    body_lines.push("    mul.f32 %scratch, %scratch, %val;".into()); // x^3
                    // Step 2: 0.044715 * x^3
                    body_lines.push(format!(
                        "    mul.f32 %scratch, %scratch, 0f{:08X};",
                        (0.044715_f32).to_bits()
                    ));
                    // Step 3: x + 0.044715 * x^3
                    body_lines.push("    add.f32 %scratch, %val, %scratch;".into());
                    // Step 4: sqrt(2/pi) * (x + 0.044715 * x^3)
                    body_lines.push(format!(
                        "    mul.f32 %scratch, %scratch, 0f{:08X};",
                        0.797_884_6_f32.to_bits()
                    ));
                    // Step 5: tanh via 2*sigmoid(2*arg) - 1
                    body_lines.push("    add.f32 %scratch, %scratch, %scratch;".into()); // 2*arg
                    body_lines.push("    neg.f32 %tmp, %scratch;".into());
                    body_lines.push("    mul.f32 %tmp, %tmp, 0f3FB8AA3B;".into()); // log2(e)
                    body_lines.push("    ex2.approx.f32 %exp_tmp, %tmp;".into());
                    body_lines.push("    add.f32 %tmp, %exp_tmp, 0f3F800000;".into()); // 1.0
                    body_lines.push("    rcp.approx.f32 %scratch, %tmp;".into()); // sigmoid(2*arg)
                    body_lines.push("    add.f32 %scratch, %scratch, %scratch;".into()); // 2*sigmoid(2*arg)
                    body_lines.push("    sub.f32 %scratch, %scratch, 0f3F800000;".into()); // tanh
                    // Step 6: 0.5 * (1 + tanh(...))
                    body_lines.push("    add.f32 %scratch, %scratch, 0f3F800000;".into()); // 1 + tanh
                    body_lines.push(format!(
                        "    mul.f32 %scratch, %scratch, 0f{:08X};",
                        (0.5_f32).to_bits()
                    )); // 0.5 * (1 + tanh)
                    // Step 7: x * result
                    body_lines.push("    mul.f32 %val, %val, %scratch;".into());
                }
                FusedOp::Silu => {
                    // SiLU: x * sigmoid(x)
                    body_lines.push("    neg.f32 %tmp, %val;".into());
                    body_lines.push("    mul.f32 %scratch, %tmp, 0f3FB8AA3B;".into());
                    body_lines.push("    ex2.approx.f32 %exp_tmp, %scratch;".into());
                    body_lines.push("    add.f32 %scratch, %exp_tmp, 0f3F800000;".into());
                    body_lines.push("    rcp.approx.f32 %scratch, %scratch;".into()); // sigmoid(x)
                    body_lines.push("    mul.f32 %val, %val, %scratch;".into());
                }
                FusedOp::Sqrt => {
                    body_lines.push("    sqrt.approx.f32 %val, %val;".into());
                }
                FusedOp::Abs => {
                    body_lines.push("    abs.f32 %val, %val;".into());
                }
                FusedOp::Pow(p) => {
                    // x^p via lg2/mul/ex2: x^p = 2^(p * log2(x))
                    body_lines.push("    lg2.approx.f32 %scratch, %val;".into());
                    body_lines.push(format!(
                        "    mul.f32 %scratch, %scratch, 0f{:08X};",
                        (*p as f32).to_bits()
                    ));
                    body_lines.push("    ex2.approx.f32 %val, %scratch;".into());
                }
                FusedOp::Exp => {
                    // exp(x) = 2^(x * log2(e))
                    body_lines.push("    mul.f32 %scratch, %val, 0f3FB8AA3B;".into()); // x * log2(e)
                    body_lines.push("    ex2.approx.f32 %val, %scratch;".into());
                }
                FusedOp::Log => {
                    // ln(x) = log2(x) * ln(2)
                    body_lines.push("    lg2.approx.f32 %scratch, %val;".into());
                    body_lines.push(format!(
                        "    mul.f32 %val, %scratch, 0f{:08X};",
                        (std::f32::consts::LN_2).to_bits()
                    ));
                }
                FusedOp::ScalarMul(s) => {
                    body_lines.push(format!(
                        "    mul.f32 %val, %val, 0f{:08X};",
                        (*s as f32).to_bits()
                    ));
                }
                FusedOp::ScalarAdd(s) => {
                    body_lines.push(format!(
                        "    add.f32 %val, %val, 0f{:08X};",
                        (*s as f32).to_bits()
                    ));
                }
            }
        }

        let body = body_lines.join("\n");

        Ok(format!(
            "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {{
{reg_decls}

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    add.u64 %in, %in, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %val, [%in];

{body}

    st.global.f32 [%out], %val;

DONE:
    ret;
}}
"
        ))
    }

    /// Like [`generate_ptx_named`](Self::generate_ptx_named), but emits an
    /// f64 PTX kernel. Transcendentals are Rust-owned PTX math fragments rather
    /// than CUDA C or NVRTC output.
    pub fn generate_ptx_f64_named(&self, kernel_name: &str) -> FerrotorchResult<String> {
        validate_identifier(kernel_name)?;
        for op in &self.ops {
            if matches!(
                op,
                FusedOp::Add | FusedOp::Sub | FusedOp::Mul | FusedOp::Div
            ) {
                return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
                    message: format!(
                        "generate_ptx_f64_named: binary op '{op}' in unary FusedChain requires a \
                         second input pointer and cannot be lowered to a single-input PTX kernel"
                    ),
                });
            }
        }

        let needs_math = self.ops.iter().any(|op| {
            matches!(
                op,
                FusedOp::Sigmoid
                    | FusedOp::Tanh
                    | FusedOp::Gelu
                    | FusedOp::Silu
                    | FusedOp::Exp
                    | FusedOp::Log
                    | FusedOp::Pow(_)
            )
        });
        let needs_zero = self.ops.iter().any(|op| matches!(op, FusedOp::Relu));

        let mut reg_decls = String::from(
            "    .reg .u32 %r_tid, %bid, %bdim, %n_reg;\n\
             \x20   .reg .u64 %in, %out, %off;\n\
             \x20   .reg .f64 %val;\n\
             \x20   .reg .pred %p;\n",
        );
        if needs_zero {
            reg_decls.push_str("    .reg .f64 %zero;\n");
        }
        if needs_math {
            emit_ptx_f64_math_reg_decls(&mut reg_decls);
        }

        let mut body = String::new();
        if needs_zero {
            body.push_str("    mov.f64 %zero, 0d0000000000000000;\n");
        }
        for op in &self.ops {
            match op {
                FusedOp::Add | FusedOp::Sub | FusedOp::Mul | FusedOp::Div => {
                    unreachable!("binary ops rejected by early validation");
                }
                FusedOp::Neg => {
                    body.push_str("    neg.f64 %val, %val;\n");
                }
                FusedOp::Relu => {
                    body.push_str("    max.f64 %val, %val, %zero;\n");
                }
                FusedOp::Sigmoid => emit_ptx_f64_sigmoid_inplace(&mut body, "%val"),
                FusedOp::Tanh => emit_ptx_f64_tanh_inplace(&mut body, "%val"),
                FusedOp::Gelu => emit_ptx_f64_gelu_tanh_inplace(&mut body, "%val"),
                FusedOp::Silu => emit_ptx_f64_silu_inplace(&mut body, "%val"),
                FusedOp::Sqrt => {
                    body.push_str("    sqrt.rn.f64 %val, %val;\n");
                }
                FusedOp::Abs => {
                    body.push_str("    abs.f64 %val, %val;\n");
                }
                FusedOp::Exp => emit_ptx_f64_exp_inplace(&mut body, "%val"),
                FusedOp::Log => emit_ptx_f64_log_inplace(&mut body, "%val"),
                FusedOp::Pow(p) => emit_ptx_f64_pow_const(&mut body, "%val", *p),
                FusedOp::ScalarMul(s) => {
                    body.push_str(&format!(
                        "    mul.f64 %val, %val, {};\n",
                        ptx_f64_const_literal(*s)
                    ));
                }
                FusedOp::ScalarAdd(s) => {
                    body.push_str(&format!(
                        "    add.f64 %val, %val, {};\n",
                        ptx_f64_const_literal(*s)
                    ));
                }
            }
        }

        Ok(format!(
            "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry {kernel_name}(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {{
{reg_decls}
    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 3;

    add.u64 %in, %in, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %val, [%in];

{body}
    st.global.f64 [%out], %val;

DONE:
    ret;
}}
"
        ))
    }

    // -------------------------------------------------------------------
    // Rust source generation
    // -------------------------------------------------------------------

    /// Generate a standalone Rust function that applies this fused chain
    /// elementwise.
    ///
    /// The generated function is generic over `f32` and `f64` through a small
    /// emitted `FusedFloat` trait, so the inspectable source artifact covers
    /// both scalar widths without relying on CUDA C, C headers, OpenMP pragmas,
    /// or a foreign toolchain.
    ///
    /// The generated function signature is:
    ///
    /// ```text
    /// pub fn <fn_name><T: FusedFloat>(input: &[T], output: &mut [T])
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the chain contains unsupported binary ops, or
    /// if `fn_name` is not a valid Rust identifier.
    pub fn generate_rust(&self, fn_name: &str) -> FerrotorchResult<String> {
        validate_identifier(fn_name)?;

        // Reject unsupported binary ops.
        for op in &self.ops {
            if matches!(
                op,
                FusedOp::Add | FusedOp::Sub | FusedOp::Mul | FusedOp::Div
            ) {
                return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
                    message: format!(
                        "generate_rust: binary op '{op}' in unary FusedChain requires a \
                         second input and cannot be lowered to a single-input Rust loop"
                    ),
                });
            }
        }

        let mut body_lines: Vec<String> = Vec::new();
        for op in &self.ops {
            match op {
                FusedOp::Add | FusedOp::Sub | FusedOp::Mul | FusedOp::Div => {
                    unreachable!("binary ops rejected above");
                }
                FusedOp::Neg => {
                    body_lines.push("        val = T::zero() - val;".into());
                }
                FusedOp::Relu => {
                    body_lines.push(
                        "        val = if val > T::zero() { val } else { T::zero() };".into(),
                    );
                }
                FusedOp::Sigmoid => {
                    body_lines.push(
                        "        val = T::one() / (T::one() + (T::zero() - val).exp());".into(),
                    );
                }
                FusedOp::Tanh => {
                    body_lines.push("        val = val.tanh();".into());
                }
                FusedOp::Gelu => {
                    // Tanh-based GELU approximation (matches all backends).
                    body_lines.push("        {".into());
                    body_lines.push("            let x3 = val * val * val;".into());
                    body_lines.push(
                        "            let inner = T::from_f64(0.7978845608028654) * (val + T::from_f64(0.044715) * x3);".into(),
                    );
                    body_lines.push(
                        "            val = val * T::from_f64(0.5) * (T::one() + inner.tanh());"
                            .into(),
                    );
                    body_lines.push("        }".into());
                }
                FusedOp::Silu => {
                    body_lines.push(
                        "        { let s = T::one() / (T::one() + (T::zero() - val).exp()); val = val * s; }".into(),
                    );
                }
                FusedOp::Sqrt => {
                    body_lines.push("        val = val.sqrt();".into());
                }
                FusedOp::Abs => {
                    body_lines.push("        val = val.abs();".into());
                }
                FusedOp::Exp => {
                    body_lines.push("        val = val.exp();".into());
                }
                FusedOp::Log => {
                    body_lines.push("        val = val.ln();".into());
                }
                FusedOp::Pow(p) => {
                    body_lines.push(format!(
                        "        val = val.powf(T::from_f64({}));",
                        rust_f64_literal(*p)
                    ));
                }
                FusedOp::ScalarMul(s) => {
                    body_lines.push(format!(
                        "        val = val * T::from_f64({});",
                        rust_f64_literal(*s)
                    ));
                }
                FusedOp::ScalarAdd(s) => {
                    body_lines.push(format!(
                        "        val = val + T::from_f64({});",
                        rust_f64_literal(*s)
                    ));
                }
            }
        }
        let body = body_lines.join("\n");

        Ok(format!(
            "{prelude}

pub fn {fn_name}<T: FusedFloat>(input: &[T], output: &mut [T]) {{
    assert!(
        output.len() >= input.len(),
        \"{fn_name}: output length must be at least input length\"
    );
    for (i, &x) in input.iter().enumerate() {{
        let mut val = x;
{body}
        output[i] = val;
    }}
}}
",
            prelude = rust_float_prelude(),
        ))
    }
}

/// Generate a standalone Rust function that performs a reduction.
///
/// The generated source is generic over `f32` and `f64` through the same
/// emitted `FusedFloat` trait used by [`FusedChain::generate_rust`].
///
/// For [`ReductionKind::Mean`], empty input returns `NaN`, matching `PyTorch`'s
/// floating mean semantics and the old `0.0 / 0.0` C behavior without relying
/// on a divide-by-zero expression.
pub fn generate_reduction_rust(kind: ReductionKind, fn_name: &str) -> FerrotorchResult<String> {
    validate_identifier(fn_name)?;

    let (identity, accumulate_expr, finalize) = match kind {
        ReductionKind::Sum => ("T::zero()", "acc = acc + x;", ""),
        ReductionKind::Prod => ("T::one()", "acc = acc * x;", ""),
        ReductionKind::Mean => (
            "T::zero()",
            "acc = acc + x;",
            "\n    if input.is_empty() {\n        T::nan()\n    } else {\n        acc / T::from_f64(input.len() as f64)\n    }",
        ),
    };

    let return_expr = if kind == ReductionKind::Mean {
        finalize
    } else {
        "\n    acc"
    };

    Ok(format!(
        "{prelude}

pub fn {fn_name}<T: FusedFloat>(input: &[T]) -> T {{
    let mut acc = {identity};
    for &x in input {{
        {accumulate_expr}
    }}
{return_expr}
}}
",
        prelude = rust_float_prelude(),
    ))
}

fn rust_float_prelude() -> &'static str {
    "\
#[allow(dead_code)]
pub trait FusedFloat:
    Copy
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn nan() -> Self;
    fn from_f64(value: f64) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn tanh(self) -> Self;
    fn powf(self, exponent: Self) -> Self;
}

impl FusedFloat for f32 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn nan() -> Self { f32::NAN }
    fn from_f64(value: f64) -> Self { value as f32 }
    fn exp(self) -> Self { f32::exp(self) }
    fn ln(self) -> Self { f32::ln(self) }
    fn sqrt(self) -> Self { f32::sqrt(self) }
    fn abs(self) -> Self { f32::abs(self) }
    fn tanh(self) -> Self { f32::tanh(self) }
    fn powf(self, exponent: Self) -> Self { f32::powf(self, exponent) }
}

impl FusedFloat for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn nan() -> Self { f64::NAN }
    fn from_f64(value: f64) -> Self { value }
    fn exp(self) -> Self { f64::exp(self) }
    fn ln(self) -> Self { f64::ln(self) }
    fn sqrt(self) -> Self { f64::sqrt(self) }
    fn abs(self) -> Self { f64::abs(self) }
    fn tanh(self) -> Self { f64::tanh(self) }
    fn powf(self, exponent: Self) -> Self { f64::powf(self, exponent) }
}"
}

fn rust_f64_literal(value: f64) -> String {
    if value.is_nan() {
        "f64::NAN".into()
    } else if value == f64::INFINITY {
        "f64::INFINITY".into()
    } else if value == f64::NEG_INFINITY {
        "f64::NEG_INFINITY".into()
    } else {
        format!("{value:.17}")
    }
}

impl Default for FusedChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Reduction PTX generation
// ---------------------------------------------------------------------------

/// Generate a PTX module that performs a parallel f32 reduction.
///
/// The generated module is the same Rust-owned PTX reduction path used by
/// [`GpuCodegen::generate_ptx_source`]: callers launch `<kernel_name>_init`
/// first, then `<kernel_name>`, and for [`ReductionKind::Mean`] launch
/// `<kernel_name>_finalize`. The reduction entry uses grid-stride loads,
/// shared-memory block reduction, `atom.global.add.f32` for sum/mean, and an
/// `atom.global.cas.b32` loop for product.
///
/// Empty reductions match `PyTorch` floating semantics: sum -> `0.0`,
/// prod -> `1.0`, and mean -> `NaN`.
///
/// # Errors
///
/// Returns an error if `kernel_name` is not a valid identifier.
pub fn generate_reduction_ptx(kind: ReductionKind, kernel_name: &str) -> FerrotorchResult<String> {
    validate_identifier(kernel_name)?;

    let loop_body = match kind {
        ReductionKind::Sum | ReductionKind::Mean => vec![LoopIR::Accumulate {
            var: "acc".into(),
            value: Expr::index("in0", Expr::var("i")),
        }],
        ReductionKind::Prod => vec![LoopIR::Assign {
            var: "acc".into(),
            value: Expr::bin(
                BinOpKind::Mul,
                Expr::var("acc"),
                Expr::index("in0", Expr::var("i")),
            ),
        }],
    };
    let store_value = if kind == ReductionKind::Mean {
        Expr::bin(BinOpKind::Div, Expr::var("acc"), Expr::constant(1.0))
    } else {
        Expr::var("acc")
    };
    let init = if kind == ReductionKind::Prod {
        1.0
    } else {
        0.0
    };
    let loops = vec![
        LoopIR::Let {
            var: "acc".into(),
            value: Expr::constant(init),
        },
        LoopIR::Loop {
            var: "i".into(),
            start: Expr::int(0),
            end: Expr::int(1),
            body: loop_body,
        },
        LoopIR::Store {
            buffer: "out".into(),
            index: Expr::int(0),
            value: store_value,
        },
    ];

    GpuCodegen::generate_ptx_source(&loops, kernel_name, 256, 1, Dtype::F32)
        .map_err(ferrotorch_core::error::FerrotorchError::from)
}

// ---------------------------------------------------------------------------
// Name validation
// ---------------------------------------------------------------------------

/// Validate that `name` is a legal C/PTX identifier: `[a-zA-Z_][a-zA-Z0-9_]*`.
///
/// This prevents injection attacks when interpolating user-provided or
/// generated names into C, CUDA, or PTX source code.
fn validate_identifier(name: &str) -> FerrotorchResult<()> {
    if name.is_empty() {
        return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
            message: "identifier name must not be empty".into(),
        });
    }

    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if !first.is_ascii_alphabetic() && first != '_' {
        return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
            message: format!(
                "identifier '{name}' has invalid first character '{first}'; \
                 must match [a-zA-Z_][a-zA-Z0-9_]*"
            ),
        });
    }

    for ch in chars {
        if !ch.is_ascii_alphanumeric() && ch != '_' {
            return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
                message: format!(
                    "identifier '{name}' contains invalid character '{ch}'; \
                     must match [a-zA-Z_][a-zA-Z0-9_]*"
                ),
            });
        }
    }

    Ok(())
}

/// Sanitize a string for safe inclusion in a code comment by removing
/// sequences that could close a C-style block comment (`*/`).
///
/// For PTX (which uses `//` line comments) this is not strictly needed,
/// but it is essential for C/CUDA codegen to prevent comment-terminator
/// injection.
#[allow(dead_code)]
pub(crate) fn sanitize_comment(text: &str) -> String {
    text.replace("*/", "* /")
}

/// Validate that `name` is a legal C/PTX identifier.
///
/// Re-exported for use by other codegen modules.
#[allow(dead_code)]
pub(crate) fn validate_codegen_identifier(name: &str) -> FerrotorchResult<()> {
    validate_identifier(name)
}

// ---------------------------------------------------------------------------
// CPU op application helper
// ---------------------------------------------------------------------------

/// Apply a single [`FusedOp`] in-place across a mutable slice.
///
/// Returns an error if the op is a binary op (Add/Sub/Mul/Div) that
/// requires a second operand, since silently ignoring it would produce
/// wrong results.
fn apply_op_inplace<T: Float>(op: &FusedOp, data: &mut [T]) -> FerrotorchResult<()> {
    let zero: T = num_traits::zero();
    let one: T = num_traits::one();

    match op {
        FusedOp::Add | FusedOp::Sub | FusedOp::Mul | FusedOp::Div => {
            return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
                message: format!(
                    "apply_op_inplace: binary op '{op}' in unary FusedChain requires a second \
                     operand and cannot be applied in-place on a single tensor"
                ),
            });
        }
        FusedOp::Neg => {
            for x in data.iter_mut() {
                *x = zero - *x;
            }
        }
        FusedOp::Relu => {
            for x in data.iter_mut() {
                *x = if *x > zero { *x } else { zero };
            }
        }
        FusedOp::Sigmoid => {
            for x in data.iter_mut() {
                let val = *x;
                let neg_val = zero - val;
                *x = one / (one + neg_val.exp());
            }
        }
        FusedOp::Tanh => {
            // tanh(x) = 2*sigmoid(2x) - 1
            let two = one + one;
            for x in data.iter_mut() {
                let s = one / (one + (zero - two * *x).exp());
                *x = two * s - one;
            }
        }
        FusedOp::Gelu => {
            // GELU tanh approximation:
            //   x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            // Matches the codegen NativeBackend and the standard PyTorch GELU.
            let half = T::from(0.5).unwrap();
            let sqrt_2_over_pi = T::from(0.7978845608028654).unwrap(); // sqrt(2/pi)
            let coeff = T::from(0.044715).unwrap();
            for x in data.iter_mut() {
                let x3 = *x * *x * *x;
                let inner = sqrt_2_over_pi * (*x + coeff * x3);
                *x = *x * half * (one + inner.tanh());
            }
        }
        FusedOp::Silu => {
            // SiLU: x * sigmoid(x)
            for x in data.iter_mut() {
                let val = *x;
                let neg_val = zero - val;
                let s = one / (one + neg_val.exp());
                *x = val * s;
            }
        }
        FusedOp::Sqrt => {
            for x in data.iter_mut() {
                *x = x.sqrt();
            }
        }
        FusedOp::Abs => {
            for x in data.iter_mut() {
                *x = x.abs();
            }
        }
        FusedOp::Exp => {
            for x in data.iter_mut() {
                *x = x.exp();
            }
        }
        FusedOp::Log => {
            for x in data.iter_mut() {
                *x = x.ln();
            }
        }
        FusedOp::Pow(p) => {
            let p_t = T::from(*p).unwrap();
            for x in data.iter_mut() {
                *x = x.powf(p_t);
            }
        }
        FusedOp::ScalarMul(s) => {
            let s_t = T::from(*s).unwrap();
            for x in data.iter_mut() {
                *x = *x * s_t;
            }
        }
        FusedOp::ScalarAdd(s) => {
            let s_t = T::from(*s).unwrap();
            for x in data.iter_mut() {
                *x += s_t;
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// DAG fusion helpers
// ---------------------------------------------------------------------------

/// Estimate the total number of elements across all input shapes.
///
/// Returns `0` for zero-element tensors (rather than clamping to 1, which
/// would be incorrect and could cause out-of-bounds kernel launches).
pub fn estimate_numel_for_inputs(shapes: &[Vec<usize>]) -> usize {
    shapes
        .iter()
        .map(|s| s.iter().copied().product::<usize>())
        .max()
        .unwrap_or(0)
}

/// Estimate (M, K, N) dimensions for a matrix multiplication from input shapes.
///
/// Both inputs must be 2-D. Returns `Err` for non-2D inputs rather than
/// silently returning `(1, 1, 1)` which would produce wrong results.
pub fn estimate_matmul_dims(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> FerrotorchResult<(usize, usize, usize)> {
    if lhs_shape.len() != 2 {
        return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
            message: format!(
                "estimate_matmul_dims: LHS must be 2-D, got {}-D shape {:?}",
                lhs_shape.len(),
                lhs_shape
            ),
        });
    }
    if rhs_shape.len() != 2 {
        return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
            message: format!(
                "estimate_matmul_dims: RHS must be 2-D, got {}-D shape {:?}",
                rhs_shape.len(),
                rhs_shape
            ),
        });
    }

    let m = lhs_shape[0];
    let k = lhs_shape[1];
    let n = rhs_shape[1];

    if k != rhs_shape[0] {
        return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
            message: format!(
                "estimate_matmul_dims: inner dimensions mismatch: LHS[1]={} vs RHS[0]={}",
                k, rhs_shape[0]
            ),
        });
    }

    Ok((m, k, n))
}

// ---------------------------------------------------------------------------
// Tensor-level fusion helper
// ---------------------------------------------------------------------------

/// Apply a [`FusedChain`] to a tensor, producing a new tensor with an
/// identical shape that resides on the input's device.
///
/// # CPU
///
/// On CPU inputs the chain is executed via [`FusedChain::execute_cpu`] —
/// a single allocation is reused as the operations are applied in place.
///
/// # CUDA (with `cuda` feature)
///
/// On CUDA inputs the call is forwarded to
/// [`fusion_gpu::apply_fused_gpu`](crate::fusion_gpu::apply_fused_gpu),
/// which:
///
/// 1. Generates the chain's PTX — f32 via
///    [`FusedChain::generate_ptx_named`], f64 via
///    [`FusedChain::generate_ptx_f64_named`].
/// 2. Compiles + caches the resulting `CudaFunction` via
///    `ferrotorch-gpu::module_cache::get_or_compile_owned` (keyed on
///    PTX hash × device ordinal so each unique chain pays the
///    compilation cost once).
/// 3. Launches the kernel on the input's stream and returns a
///    device-resident Tensor.
///
/// # CUDA (without `cuda` feature)
///
/// Returns [`FerrotorchError::NotImplementedOnCuda`] with an op message
/// directing the caller to build with the `cuda` feature.
///
/// # Errors
///
/// - [`FerrotorchError::NotImplementedOnCuda`]: input is on a CUDA device
///   but the crate was built without the `cuda` feature, or the chain's
///   dtype is neither f32 nor f64.
/// - [`FerrotorchError::InvalidArgument`]: chain contains a binary op
///   (Add/Sub/Mul/Div) on the GPU path, or CUDA PTX load / launch failure.
/// - Propagates any error from [`FusedChain::execute_cpu`] or
///   [`Tensor::from_storage`] on the CPU path (including the
///   `GpuTensorNotAccessible` / `InvalidArgument` errors `Tensor::data`
///   may raise on non-CPU storage variants such as cubecl or
///   non-contiguous tensors).
///
/// [`FerrotorchError::NotImplementedOnCuda`]: ferrotorch_core::error::FerrotorchError::NotImplementedOnCuda
/// [`FerrotorchError::InvalidArgument`]: ferrotorch_core::error::FerrotorchError::InvalidArgument
pub fn apply_fused<T: Float>(input: &Tensor<T>, chain: &FusedChain) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        #[cfg(feature = "cuda")]
        {
            return crate::fusion_gpu::apply_fused_gpu(input, chain);
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Err(
                ferrotorch_core::error::FerrotorchError::NotImplementedOnCuda {
                    op: "apply_fused: build with the `cuda` feature to enable GPU dispatch \
                         via FusedChain PTX → ferrotorch-gpu module cache",
                },
            );
        }
    }
    let data = input.data()?;
    let result = chain.execute_cpu(data)?;
    Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::storage::TensorStorage;
    use ferrotorch_core::tensor::Tensor;

    // -- FusedChain basics --------------------------------------------------

    #[test]
    fn test_chain_new_is_empty() {
        let chain = FusedChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_chain_push_and_len() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);
        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());
    }

    // -- with_fusion flag ---------------------------------------------------

    #[test]
    fn test_fusion_flag_default_off() {
        assert!(!is_fusion_enabled());
    }

    #[test]
    fn test_fusion_flag_scoped() {
        assert!(!is_fusion_enabled());
        with_fusion(|| {
            assert!(is_fusion_enabled());
        });
        assert!(!is_fusion_enabled());
    }

    #[test]
    fn test_fusion_flag_nested() {
        with_fusion(|| {
            assert!(is_fusion_enabled());
            with_fusion(|| {
                assert!(is_fusion_enabled());
            });
            // Still enabled after inner scope -- inner guard restores `true`.
            assert!(is_fusion_enabled());
        });
        assert!(!is_fusion_enabled());
    }

    // -- CPU execution: scalar_add + relu + neg (the spec test) ---------------

    #[test]
    fn test_fused_scalar_add_relu_neg_cpu() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarAdd(2.0));
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);

        let input: Vec<f32> = vec![-5.0, -1.0, 0.0, 1.0, 3.0];

        // Sequential reference:
        //   scalar_add(2):  [-3.0, 1.0, 2.0, 3.0, 5.0]
        //   relu:           [ 0.0, 1.0, 2.0, 3.0, 5.0]
        //   neg:            [ 0.0,-1.0,-2.0,-3.0,-5.0]
        let expected: Vec<f32> = vec![0.0, -1.0, -2.0, -3.0, -5.0];

        let result = chain.execute_cpu(&input).unwrap();
        assert_eq!(result.len(), expected.len());
        for (got, exp) in result.iter().zip(&expected) {
            assert!((got - exp).abs() < 1e-6, "got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_fused_matches_sequential() {
        // Verify the fused result matches applying each op one at a time.
        let input: Vec<f64> = vec![-3.0, -1.5, 0.0, 0.5, 2.0, 4.0];

        // Build a chain: scalar_add(2) -> relu -> neg.
        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarAdd(2.0));
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);

        // Fused.
        let fused = chain.execute_cpu(&input).unwrap();

        // Sequential.
        let mut sequential = input.clone();
        // scalar_add(2)
        for x in &mut sequential {
            *x += 2.0;
        }
        // relu
        for x in &mut sequential {
            if *x < 0.0 {
                *x = 0.0;
            }
        }
        // neg
        for x in &mut sequential {
            *x = -*x;
        }

        assert_eq!(fused.len(), sequential.len());
        for (i, (got, exp)) in fused.iter().zip(&sequential).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "element {i}: fused={got}, sequential={exp}",
            );
        }
    }

    // -- CPU execution: individual ops ----------------------------------------

    #[test]
    fn test_fused_neg() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Neg);
        let result = chain.execute_cpu(&[1.0f32, -2.0, 0.0]).unwrap();
        assert_eq!(result, vec![-1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_fused_sigmoid() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Sigmoid);
        let result = chain.execute_cpu(&[0.0f64]).unwrap();
        assert!((result[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fused_tanh() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Tanh);
        let result = chain.execute_cpu(&[0.0f64]).unwrap();
        assert!(result[0].abs() < 1e-10, "tanh(0) should be 0");
    }

    #[test]
    fn test_fused_sqrt() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Sqrt);
        let result = chain.execute_cpu(&[4.0f32, 9.0, 16.0]).unwrap();
        let expected = vec![2.0f32, 3.0, 4.0];
        for (got, exp) in result.iter().zip(&expected) {
            assert!((got - exp).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fused_abs() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Abs);
        let result = chain.execute_cpu(&[-3.0f32, 0.0, 5.0]).unwrap();
        assert_eq!(result, vec![3.0, 0.0, 5.0]);
    }

    #[test]
    fn test_fused_pow() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Pow(2.0));
        let result = chain.execute_cpu(&[3.0f64]).unwrap();
        assert!((result[0] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_fused_scalar_mul() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarMul(3.0));
        let result = chain.execute_cpu(&[2.0f32, -1.0]).unwrap();
        assert_eq!(result, vec![6.0, -3.0]);
    }

    #[test]
    fn test_fused_empty_input() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);
        let result = chain.execute_cpu::<f32>(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_fused_empty_chain() {
        let chain = FusedChain::new();
        let input = vec![1.0f32, 2.0, 3.0];
        let result = chain.execute_cpu(&input).unwrap();
        assert_eq!(result, input);
    }

    // -- PTX generation -------------------------------------------------------

    #[test]
    fn test_ptx_generation_valid_string() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarAdd(2.0));
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);

        let ptx = chain.generate_ptx().unwrap();

        // Must have the standard PTX header.
        assert!(ptx.contains(".version 7.0"));
        assert!(ptx.contains(".target sm_52"));
        assert!(ptx.contains(".address_size 64"));

        // Must declare the entry point.
        assert!(ptx.contains(".visible .entry fused_kernel"));

        // Must have parameter declarations.
        assert!(ptx.contains("in_ptr"));
        assert!(ptx.contains("out_ptr"));

        // Must contain the operations.
        assert!(
            ptx.contains("add.f32 %val"),
            "ScalarAdd should produce an add.f32 instruction"
        );
        assert!(
            ptx.contains("max.f32 %val"),
            "Relu should produce a max.f32 instruction"
        );
        assert!(
            ptx.contains("neg.f32 %val"),
            "Neg should produce a neg.f32 instruction"
        );

        // Must end with store + ret.
        assert!(ptx.contains("st.global.f32 [%out], %val;"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn test_ptx_generation_sigmoid() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Sigmoid);
        let ptx = chain.generate_ptx().unwrap();
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_ptx_generation_sqrt() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Sqrt);
        let ptx = chain.generate_ptx().unwrap();
        assert!(ptx.contains("sqrt.approx.f32"));
    }

    #[test]
    fn test_ptx_generation_pow() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Pow(3.0));
        let ptx = chain.generate_ptx().unwrap();
        assert!(ptx.contains("lg2.approx.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
    }

    #[test]
    fn test_ptx_generation_f64_transcendentals_are_rust_ptx() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Exp);
        chain.push(FusedOp::Log);
        chain.push(FusedOp::Sigmoid);
        chain.push(FusedOp::Tanh);
        chain.push(FusedOp::Gelu);
        chain.push(FusedOp::Silu);
        chain.push(FusedOp::Pow(3.0));

        let ptx = chain
            .generate_ptx_f64_named("fused_chain_f64_test")
            .unwrap();

        assert!(ptx.contains(".visible .entry fused_chain_f64_test"));
        assert!(ptx.contains("ld.global.f64 %val"));
        assert!(ptx.contains("st.global.f64 [%out], %val;"));
        assert!(
            ptx.contains("fma.rn.f64"),
            "f64 PTX must contain inlined polynomial math:\n{ptx}"
        );
        assert!(
            !ptx.contains("__global__")
                && !ptx.contains("exp(")
                && !ptx.contains("tanh(")
                && !ptx.contains(".extern")
                && !ptx.contains("call"),
            "f64 FusedChain must not emit CUDA C or external libdevice callouts:\n{ptx}"
        );
        assert!(
            !ptx.contains("ex2.approx.f32")
                && !ptx.contains("lg2.approx.f32")
                && !ptx.contains("cvt.f32.f64")
                && !ptx.contains("cvt.f64.f32"),
            "f64 FusedChain PTX must not demote through f32:\n{ptx}"
        );
    }

    #[test]
    fn test_ptx_generation_f64_rejects_binary_op_chain() {
        let mut chain = FusedChain::new();
        chain.push(FusedOp::Add);

        let err = chain
            .generate_ptx_f64_named("fused_chain_f64_bad")
            .expect_err("binary op chain must be rejected before PTX generation");
        match err {
            ferrotorch_core::error::FerrotorchError::InvalidArgument { message } => {
                assert!(
                    message.contains("binary op"),
                    "error must explain binary op rejection; got: {message}"
                );
            }
            other => panic!("expected InvalidArgument for binary op chain, got {other:?}"),
        }
    }

    // -- apply_fused (tensor-level) -------------------------------------------

    #[test]
    fn test_apply_fused_tensor() {
        let storage = TensorStorage::cpu(vec![-5.0f32, -1.0, 0.0, 1.0, 3.0]);
        let tensor = Tensor::from_storage(storage, vec![5], false).unwrap();

        let mut chain = FusedChain::new();
        chain.push(FusedOp::ScalarAdd(2.0));
        chain.push(FusedOp::Relu);
        chain.push(FusedOp::Neg);

        let result = apply_fused(&tensor, &chain).unwrap();
        let result_data = result.data().unwrap();
        let expected = [0.0f32, -1.0, -2.0, -3.0, -5.0];

        assert_eq!(result_data.len(), expected.len());
        for (got, exp) in result_data.iter().zip(&expected) {
            assert!((got - exp).abs() < 1e-6, "got {got}, expected {exp}");
        }
        assert_eq!(result.shape(), &[5]);
    }

    // -- FusedOp Display ------------------------------------------------------

    #[test]
    fn test_fused_op_display() {
        assert_eq!(format!("{}", FusedOp::Relu), "relu");
        assert_eq!(format!("{}", FusedOp::ScalarAdd(1.5)), "scalar_add(1.5)");
        assert_eq!(format!("{}", FusedOp::Pow(2.0)), "pow(2)");
    }

    // -- apply_fused doc-drift + CUDA-error guards (Pass 5.B.4 / #1106) -------

    /// Mechanical doc-drift guard for [`apply_fused`].
    ///
    /// The Pass-5.B.4 doc had two known mis-claims plus a follow-up
    /// placeholder. This guard now enforces all three are gone after the
    /// runtime-executor landing dispatch.
    #[test]
    fn apply_fused_doc_does_not_mention_1138() {
        let src = include_str!("fusion.rs");
        let sig_idx = src
            .find("pub fn apply_fused<")
            .expect("apply_fused signature must exist in fusion.rs");
        // Walk backward from the signature to the start of its doc block.
        let prelude = &src[..sig_idx];
        let doc_start = prelude.rfind("\n\n").map_or(0, |i| i + 2);
        let doc_block = &prelude[doc_start..];
        // Regression: prior false claim "same device".
        assert!(
            !doc_block.contains("same device"),
            "apply_fused doc must not claim 'same device'; doc was:\n{doc_block}"
        );
        // Regression: prior false claim "could be dispatched".
        assert!(
            !doc_block.contains("could be dispatched"),
            "apply_fused doc must not claim PTX 'could be dispatched'; doc was:\n{doc_block}"
        );
        // The follow-up issue was a placeholder; with the runtime
        // executor landed, the doc must no longer cite it.
        let placeholder = format!("#{}", 1138);
        assert!(
            !doc_block.contains(&placeholder),
            "apply_fused doc must NOT reference the closed follow-up issue; doc was:\n{doc_block}"
        );
        // The placeholder phrase pattern must not return.
        let placeholder_phrase = format!("{} #", "tracked in");
        assert!(
            !doc_block.contains(&placeholder_phrase),
            "apply_fused doc must not contain a 'tracked-in' placeholder phrase; \
             doc was:\n{doc_block}"
        );
    }

    /// Discriminating test for the [`apply_fused`] CUDA dispatch entry.
    ///
    /// **Without the `cuda` feature**, calling `apply_fused` on a CUDA
    /// tensor must return `NotImplementedOnCuda` with an op message
    /// pointing the user at the `cuda` build feature. (With the `cuda`
    /// feature on, the CUDA path runs end-to-end; see
    /// `fusion_gpu::tests::*` for those discriminators.)
    ///
    /// # CUDA-runtime gap (honest underclaim)
    ///
    /// On a CPU-only host, `Tensor::cuda()` returns
    /// `Err(DeviceUnavailable)` and the test exits early. With the
    /// `cuda` feature OFF on a CUDA-having host, this test fails
    /// fast with a clear error message — exactly the user-facing
    /// outcome we want.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn apply_fused_errs_on_cuda_input_without_cuda_feature() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]);
        let cpu_tensor = Tensor::from_storage(storage, vec![4], false).unwrap();
        let cuda_tensor = match cpu_tensor.cuda() {
            Ok(t) => t,
            Err(_) => return,
        };

        let mut chain = FusedChain::new();
        chain.push(FusedOp::Relu);

        let result = apply_fused(&cuda_tensor, &chain);
        match result {
            Err(ferrotorch_core::error::FerrotorchError::NotImplementedOnCuda { op }) => {
                assert!(
                    op.contains("apply_fused"),
                    "error op must name apply_fused; got: {op}"
                );
                assert!(
                    op.contains("cuda"),
                    "error op must direct user to the `cuda` feature; got: {op}"
                );
            }
            other => panic!(
                "expected NotImplementedOnCuda for CUDA input to apply_fused without \
                 `cuda` feature, got {other:?}"
            ),
        }
    }
}
