//! End-to-end CUDA execution probes for Rust-owned reduction PTX.
//!
//! The generated PTX module exposes an init entry, the reduction entry, and
//! for mean a finalize entry. These tests launch that full on-device sequence
//! so reductions are not accidentally relying on CPU readbacks, prezeroed host
//! state, CUDA C, NVRTC, or libdevice.

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use ferrotorch_jit::codegen_gpu::GpuCodegen;
use ferrotorch_jit::codegen_ir;
use ferrotorch_jit::graph::{Dtype, IrOpKind};

const BLOCK: u32 = 256;

fn reduce_launch_config(n: usize) -> LaunchConfig {
    let n_u32 = n as u32;
    let blocks = n_u32.saturating_add(BLOCK - 1) / BLOCK;
    LaunchConfig {
        grid_dim: (blocks.clamp(1, 8), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn single_thread_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn run_reduction_f32(op: IrOpKind, kernel_name: &str, input: &[f32]) -> f32 {
    let n = input.len();
    let loops = codegen_ir::lower_to_loops(std::slice::from_ref(&op), &["in0"], "out", n);
    let ptx = GpuCodegen::generate_ptx_source(&loops, kernel_name, BLOCK as usize, 1, Dtype::F32)
        .expect("f32 reduction PTX generation");

    assert!(ptx.contains(&format!(".entry {kernel_name}_init")));
    assert!(!ptx.contains("__global__") && !ptx.contains("#include"));

    let ctx = CudaContext::new(0).expect("CUDA device 0 available");
    let stream = ctx.default_stream();
    let module = ctx
        .load_module(Ptx::from_src(ptx))
        .expect("driver accepts f32 reduction PTX");
    let init = module
        .load_function(&format!("{kernel_name}_init"))
        .expect("init entry");
    let reduce = module.load_function(kernel_name).expect("reduction entry");

    let device_input = if input.is_empty() {
        stream.clone_htod(&[0.0f32]).expect("dummy input")
    } else {
        stream.clone_htod(input).expect("input upload")
    };
    let mut output = unsafe { stream.alloc::<f32>(1).expect("output alloc") };
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&init)
            .arg(&mut output)
            .arg(&n_u32)
            .launch(single_thread_config())
            .expect("init launch");
        stream
            .launch_builder(&reduce)
            .arg(&device_input)
            .arg(&mut output)
            .arg(&n_u32)
            .launch(reduce_launch_config(n))
            .expect("reduction launch");
        if matches!(op, IrOpKind::Mean) {
            let finalize = module
                .load_function(&format!("{kernel_name}_finalize"))
                .expect("mean finalize entry");
            stream
                .launch_builder(&finalize)
                .arg(&mut output)
                .arg(&n_u32)
                .launch(single_thread_config())
                .expect("finalize launch");
        }
    }
    stream.synchronize().expect("stream sync");
    stream.clone_dtoh(&output).expect("output download")[0]
}

fn run_reduction_f64(op: IrOpKind, kernel_name: &str, input: &[f64]) -> f64 {
    let n = input.len();
    let loops = codegen_ir::lower_to_loops(std::slice::from_ref(&op), &["in0"], "out", n);
    let ptx = GpuCodegen::generate_ptx_source(&loops, kernel_name, BLOCK as usize, 1, Dtype::F64)
        .expect("f64 reduction PTX generation");

    assert!(ptx.contains(&format!(".entry {kernel_name}_init")));
    assert!(
        !ptx.contains("atom.global.add.f64"),
        "sm_52 f64 reductions must use CAS, not unsupported f64 atom.add"
    );
    assert!(!ptx.contains("__global__") && !ptx.contains("#include"));

    let ctx = CudaContext::new(0).expect("CUDA device 0 available");
    let stream = ctx.default_stream();
    let module = ctx
        .load_module(Ptx::from_src(ptx))
        .expect("driver accepts f64 reduction PTX");
    let init = module
        .load_function(&format!("{kernel_name}_init"))
        .expect("init entry");
    let reduce = module.load_function(kernel_name).expect("reduction entry");

    let device_input = if input.is_empty() {
        stream.clone_htod(&[0.0f64]).expect("dummy input")
    } else {
        stream.clone_htod(input).expect("input upload")
    };
    let mut output = unsafe { stream.alloc::<f64>(1).expect("output alloc") };
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&init)
            .arg(&mut output)
            .arg(&n_u32)
            .launch(single_thread_config())
            .expect("init launch");
        stream
            .launch_builder(&reduce)
            .arg(&device_input)
            .arg(&mut output)
            .arg(&n_u32)
            .launch(reduce_launch_config(n))
            .expect("reduction launch");
        if matches!(op, IrOpKind::Mean) {
            let finalize = module
                .load_function(&format!("{kernel_name}_finalize"))
                .expect("mean finalize entry");
            stream
                .launch_builder(&finalize)
                .arg(&mut output)
                .arg(&n_u32)
                .launch(single_thread_config())
                .expect("finalize launch");
        }
    }
    stream.synchronize().expect("stream sync");
    stream.clone_dtoh(&output).expect("output download")[0]
}

#[test]
fn f32_sum_mean_prod_execute_on_cuda() {
    let values: Vec<f32> = (0..777).map(|i| (i as f32 - 300.0) * 0.125).collect();
    let sum_expected: f32 = values.iter().copied().sum();
    let sum = run_reduction_f32(IrOpKind::Sum, "jit_sum_f32", &values);
    assert!(
        (sum - sum_expected).abs() <= 1e-3,
        "sum got {sum}, expected {sum_expected}"
    );

    let mean_expected = sum_expected / values.len() as f32;
    let mean = run_reduction_f32(IrOpKind::Mean, "jit_mean_f32", &values);
    assert!(
        (mean - mean_expected).abs() <= 1e-5,
        "mean got {mean}, expected {mean_expected}"
    );

    let prod_values: Vec<f32> = (0..513)
        .map(|i| if i % 2 == 0 { 1.0001_f32 } else { 0.9999_f32 })
        .collect();
    let prod_expected: f32 = prod_values.iter().copied().product();
    let prod = run_reduction_f32(IrOpKind::Prod, "jit_prod_f32", &prod_values);
    assert!(
        (prod - prod_expected).abs() <= 2e-5,
        "prod got {prod}, expected {prod_expected}"
    );
}

#[test]
fn f64_sum_mean_prod_execute_on_cuda() {
    let values: Vec<f64> = (0..777).map(|i| (i as f64 - 300.0) * 0.125).collect();
    let sum_expected: f64 = values.iter().copied().sum();
    let sum = run_reduction_f64(IrOpKind::Sum, "jit_sum_f64", &values);
    assert!(
        (sum - sum_expected).abs() <= 1e-9,
        "sum got {sum}, expected {sum_expected}"
    );

    let mean_expected = sum_expected / values.len() as f64;
    let mean = run_reduction_f64(IrOpKind::Mean, "jit_mean_f64", &values);
    assert!(
        (mean - mean_expected).abs() <= 1e-12,
        "mean got {mean}, expected {mean_expected}"
    );

    let prod_values: Vec<f64> = (0..513)
        .map(|i| if i % 2 == 0 { 1.0001_f64 } else { 0.9999_f64 })
        .collect();
    let prod_expected: f64 = prod_values.iter().copied().product();
    let prod = run_reduction_f64(IrOpKind::Prod, "jit_prod_f64", &prod_values);
    assert!(
        (prod - prod_expected).abs() <= 2e-12,
        "prod got {prod}, expected {prod_expected}"
    );
}

#[test]
fn empty_reduction_identities_match_pytorch_float_semantics() {
    let sum = run_reduction_f32(IrOpKind::Sum, "jit_empty_sum_f32", &[]);
    assert_eq!(sum, 0.0);

    let prod = run_reduction_f64(IrOpKind::Prod, "jit_empty_prod_f64", &[]);
    assert_eq!(prod, 1.0);

    let mean = run_reduction_f64(IrOpKind::Mean, "jit_empty_mean_f64", &[]);
    assert!(mean.is_nan(), "empty mean must be NaN, got {mean}");
}
