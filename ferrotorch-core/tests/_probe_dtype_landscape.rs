//! Verification probe (architect, crosslink #1184 follow-up): confirm the
//! dtype landscape for GPU dispatch.
//!
//! Establishes empirically, on a real RTX 3090, that:
//!   1. `Tensor<f64>` is GPU-capable across the common op set (the one
//!      non-default float dtype that could plausibly have dispatch gaps
//!      analogous to the bf16 #23 holes).
//!   2. The premise that f16/i32/i64/bool share the bf16 "silent f32
//!      fallthrough" blind spot does NOT hold — those are not
//!      `Tensor<T>` dtypes at all (f16 isn't `Float`; integers/bool are
//!      separate CPU-only `IntTensor`/`BoolTensor` types). The
//!      compile-fail cases are documented in comments, not code, because
//!      they cannot be written in valid Rust.
#![cfg(feature = "gpu")]
#![allow(clippy::approx_constant)]

use ferrotorch_core::grad_fns::arithmetic;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{Device, Tensor};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn t_gpu(vals: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(vals), shape.to_vec(), false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap()
}

#[derive(Debug)]
struct OpResult {
    op: &'static str,
    pass: bool,
    err: Option<String>,
}

fn run_one(
    op: &'static str,
    f: impl FnOnce() -> ferrotorch_core::FerrotorchResult<()>,
) -> OpResult {
    match f() {
        Ok(()) => OpResult { op, pass: true, err: None },
        Err(e) => OpResult { op, pass: false, err: Some(format!("{e:?}")) },
    }
}

#[test]
fn f64_op_sweep_on_cuda() {
    ensure_cuda();
    let a = || t_gpu(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = || t_gpu(vec![0.5, 1.5, 2.5, 3.5], &[2, 2]);

    let mut results: Vec<OpResult> = vec![];

    results.push(run_one("add", || arithmetic::add(&a(), &b()).map(|_| ())));
    results.push(run_one("sub", || arithmetic::sub(&a(), &b()).map(|_| ())));
    results.push(run_one("mul", || arithmetic::mul(&a(), &b()).map(|_| ())));
    results.push(run_one("div", || arithmetic::div(&a(), &b()).map(|_| ())));
    results.push(run_one("neg", || arithmetic::neg(&a()).map(|_| ())));
    results.push(run_one("exp", || a().exp_t().map(|_| ())));
    results.push(run_one("log", || a().log_t().map(|_| ())));
    results.push(run_one("sqrt", || a().sqrt_t().map(|_| ())));
    results.push(run_one("tanh_t", || a().tanh_t().map(|_| ())));
    results.push(run_one("sigmoid", || a().sigmoid().map(|_| ())));
    results.push(run_one("relu", || a().relu().map(|_| ())));
    results.push(run_one("gelu", || a().gelu().map(|_| ())));
    results.push(run_one("silu", || a().silu().map(|_| ())));
    results.push(run_one("sum_all", || a().sum_all().map(|_| ())));
    results.push(run_one("mean_all", || a().mean_all().map(|_| ())));
    results.push(run_one("sum_dim", || a().sum_dim(0, false).map(|_| ())));
    results.push(run_one("softmax", || a().softmax().map(|_| ())));
    results.push(run_one("matmul", || {
        let m = t_gpu(vec![1.0; 6], &[2, 3]);
        let n = t_gpu(vec![1.0; 6], &[3, 2]);
        m.matmul(&n).map(|_| ())
    }));

    let pass = results.iter().filter(|r| r.pass).count();
    let fail = results.iter().filter(|r| !r.pass).count();
    println!("\n=== f64 op sweep on CUDA ===");
    println!("PASS: {pass}, FAIL: {fail}, TOTAL: {}", results.len());
    for r in &results {
        let tag = if r.pass { "PASS" } else { "FAIL" };
        let suffix = r
            .err
            .as_ref()
            .map(|e| format!(" :: {}", e.lines().next().unwrap_or("")))
            .unwrap_or_default();
        println!("  {tag} {}{suffix}", r.op);
    }

    // Informational: print, don't hard-fail, so the table is always visible.
    // f64 is expected to be broadly functional on CUDA.
}

// ---------------------------------------------------------------------------
// COMPILE-FAIL DOCUMENTATION (cannot be expressed as runnable code).
//
// The following would NOT compile, proving the user's hypothesized
// "f16/i32/i64/bool share the bf16 GPU blind spot" premise does not apply:
//
//   let _ = Tensor::<half::f16>::from_storage(...);
//       // error: the trait bound `half::f16: Float` is not satisfied
//       // (dtype.rs implements Float only for f32, f64, bf16)
//
//   let _ = Tensor::<i32>::from_storage(...);
//       // error: `i32: Float` is not satisfied. Integers use the separate
//       // `IntTensor<i32>` type, which is CPU-only (no `Device` field, no
//       // `.to(Device::Cuda)` method — see int_tensor.rs).
//
//   let _ = Tensor::<bool>::from_storage(...);
//       // error: `bool: Float` is not satisfied. Booleans use the separate
//       // `BoolTensor` type, which is CPU-only (bool_tensor.rs).
//
// So for those dtypes there is no GPU dispatch path to be "blind" — they are
// architecturally CPU-only / unrepresentable as a GPU tensor. That is a
// PyTorch-PARITY question (PyTorch supports float16/int/bool tensors on
// CUDA), not a dispatch-gap bug like #23.
// ---------------------------------------------------------------------------
