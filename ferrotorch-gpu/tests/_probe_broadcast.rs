//! Permanent regression sentinel for #779: GPU broadcast arithmetic kernels.
//!
//! These tests exercise the four GPU broadcast binary ops
//! (`broadcast_{add,sub,mul,div}`) for both f32 and f64 across the four
//! shape configurations the conformance suite cares about:
//!
//! 1. `[3] + [3,3]` (rank-mismatched broadcast),
//! 2. `[1] + [3]`   (length-1 broadcast on a 1-D pair),
//! 3. `[3,1] + [1,3]` (mutual broadcast on the same rank), and
//! 4. `[3] + [3]` (same-shape control — must always pass).
//!
//! When the bug was first surfaced (phase 2.1 conformance), all
//! broadcast configurations hit `CUDA_ERROR_MISALIGNED_ADDRESS` while the
//! same-shape control passed. After the fix landed, all 4 sub-tests
//! pass for both dtypes (8 total).

#![cfg(feature = "cuda")]

use std::sync::Once;

use ferrotorch_gpu::buffer::CudaBuffer;
use ferrotorch_gpu::device::GpuDevice;
use ferrotorch_gpu::init_cuda_backend;
use ferrotorch_gpu::kernels::{
    gpu_broadcast_add, gpu_broadcast_add_f64, gpu_broadcast_div, gpu_broadcast_div_f64,
    gpu_broadcast_mul, gpu_broadcast_mul_f64, gpu_broadcast_sub, gpu_broadcast_sub_f64,
};
use ferrotorch_gpu::transfer::{cpu_to_gpu, gpu_to_cpu};

fn ensure_cuda() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn upload_f32(data: &[f32], device: &GpuDevice) -> CudaBuffer<f32> {
    cpu_to_gpu(data, device).expect("upload f32 to GPU")
}

fn upload_f64(data: &[f64], device: &GpuDevice) -> CudaBuffer<f64> {
    cpu_to_gpu(data, device).expect("upload f64 to GPU")
}

fn cpu_broadcast_add_ref(
    a: &[f64],
    b: &[f64],
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Vec<f64> {
    let out_numel: usize = out_shape.iter().product();
    let ndim = out_shape.len();
    let a_str = ref_strides(a_shape, out_shape);
    let b_str = ref_strides(b_shape, out_shape);
    let mut out = vec![0.0_f64; out_numel];
    for (i, slot) in out.iter_mut().enumerate() {
        let (a_idx, b_idx) = ref_offsets(i, out_shape, &a_str, &b_str, ndim);
        *slot = a[a_idx] + b[b_idx];
    }
    out
}

fn ref_strides(in_shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let ndim = out_shape.len();
    let in_ndim = in_shape.len();
    let mut strides = vec![0_usize; ndim];
    let mut stride: usize = 1;
    for d in (0..ndim).rev() {
        let in_d = if d + in_ndim >= ndim {
            d + in_ndim - ndim
        } else {
            strides[d] = 0;
            continue;
        };
        if in_shape[in_d] == 1 {
            strides[d] = 0;
        } else {
            strides[d] = stride;
        }
        stride *= in_shape[in_d];
    }
    strides
}

fn ref_offsets(
    flat: usize,
    out_shape: &[usize],
    a_str: &[usize],
    b_str: &[usize],
    ndim: usize,
) -> (usize, usize) {
    let mut remaining = flat;
    let mut a_idx: usize = 0;
    let mut b_idx: usize = 0;
    for d in (0..ndim).rev() {
        let coord = remaining % out_shape[d];
        remaining /= out_shape[d];
        a_idx += coord * a_str[d];
        b_idx += coord * b_str[d];
    }
    (a_idx, b_idx)
}

#[test]
fn probe_broadcast_add_f32_rank_mismatch() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("open device 0");
    let a_data = [1.0_f32, 2.0, 3.0];
    let b_data = [1.0_f32; 9];
    let a = upload_f32(&a_data, &device);
    let b = upload_f32(&b_data, &device);
    println!(
        "[probe:add_f32 a=[3] b=[3,3]] launching broadcast_add — \
         a.len()={} b.len()={} u32_align={}",
        a.len(),
        b.len(),
        std::mem::align_of::<u32>()
    );
    let result = gpu_broadcast_add(&a, &b, &[3], &[3, 3], &[3, 3], &device);
    match &result {
        Ok(out) => println!("[probe:add_f32 a=[3] b=[3,3]] OK len={}", out.len()),
        Err(e) => println!("[probe:add_f32 a=[3] b=[3,3]] ERR: {e:?}"),
    }
    let out = result.expect("broadcast add f32 [3]+[3,3] failed");
    let host = gpu_to_cpu(&out, &device).expect("readback");
    let expected = cpu_broadcast_add_ref(
        &a_data.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        &b_data.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        &[3],
        &[3, 3],
        &[3, 3],
    );
    for (i, (h, e)) in host.iter().zip(expected.iter()).enumerate() {
        let diff = (*h as f64 - *e).abs();
        assert!(diff < 1e-5, "mismatch at {i}: got {h} expected {e}");
    }
}

#[test]
fn probe_broadcast_add_f32_len1_to_3() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("open device 0");
    let a = upload_f32(&[5.0_f32], &device);
    let b = upload_f32(&[1.0_f32, 2.0, 3.0], &device);
    let result = gpu_broadcast_add(&a, &b, &[1], &[3], &[3], &device);
    match &result {
        Ok(out) => println!("[probe:add_f32 a=[1] b=[3]] OK len={}", out.len()),
        Err(e) => println!("[probe:add_f32 a=[1] b=[3]] ERR: {e:?}"),
    }
    let out = result.expect("broadcast add f32 [1]+[3] failed");
    let host = gpu_to_cpu(&out, &device).expect("readback");
    let expected = [6.0_f32, 7.0, 8.0];
    for (i, (h, e)) in host.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*h - *e).abs() < 1e-5,
            "mismatch at {i}: got {h} expected {e}"
        );
    }
}

#[test]
fn probe_broadcast_add_f32_outer() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("open device 0");
    // [3,1] + [1,3] -> [3,3]
    let a = upload_f32(&[1.0_f32, 2.0, 3.0], &device);
    let b = upload_f32(&[10.0_f32, 20.0, 30.0], &device);
    let result = gpu_broadcast_add(&a, &b, &[3, 1], &[1, 3], &[3, 3], &device);
    match &result {
        Ok(out) => println!("[probe:add_f32 a=[3,1] b=[1,3]] OK len={}", out.len()),
        Err(e) => println!("[probe:add_f32 a=[3,1] b=[1,3]] ERR: {e:?}"),
    }
    let out = result.expect("broadcast add f32 [3,1]+[1,3] failed");
    let host = gpu_to_cpu(&out, &device).expect("readback");
    let expected = [11.0_f32, 21.0, 31.0, 12.0, 22.0, 32.0, 13.0, 23.0, 33.0];
    for (i, (h, e)) in host.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*h - *e).abs() < 1e-5,
            "mismatch at {i}: got {h} expected {e}"
        );
    }
}

#[test]
fn probe_broadcast_add_f32_same_shape_control() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("open device 0");
    let a = upload_f32(&[1.0_f32, 2.0, 3.0], &device);
    let b = upload_f32(&[10.0_f32, 20.0, 30.0], &device);
    let result = gpu_broadcast_add(&a, &b, &[3], &[3], &[3], &device);
    match &result {
        Ok(out) => println!("[probe:add_f32 a=[3] b=[3] (control)] OK len={}", out.len()),
        Err(e) => println!("[probe:add_f32 a=[3] b=[3] (control)] ERR: {e:?}"),
    }
    let out = result.expect("broadcast add f32 [3]+[3] failed");
    let host = gpu_to_cpu(&out, &device).expect("readback");
    let expected = [11.0_f32, 22.0, 33.0];
    for (i, (h, e)) in host.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*h - *e).abs() < 1e-5,
            "mismatch at {i}: got {h} expected {e}"
        );
    }
}

#[test]
fn probe_broadcast_add_f64_rank_mismatch() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("open device 0");
    let a = upload_f64(&[1.0_f64, 2.0, 3.0], &device);
    let b = upload_f64(&[1.0_f64; 9], &device);
    let result = gpu_broadcast_add_f64(&a, &b, &[3], &[3, 3], &[3, 3], &device);
    match &result {
        Ok(out) => println!("[probe:add_f64 a=[3] b=[3,3]] OK len={}", out.len()),
        Err(e) => println!("[probe:add_f64 a=[3] b=[3,3]] ERR: {e:?}"),
    }
    let out = result.expect("broadcast add f64 [3]+[3,3] failed");
    let host = gpu_to_cpu(&out, &device).expect("readback");
    let expected =
        cpu_broadcast_add_ref(&[1.0_f64, 2.0, 3.0], &[1.0_f64; 9], &[3], &[3, 3], &[3, 3]);
    for (i, (h, e)) in host.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*h - *e).abs() < 1e-9,
            "mismatch at {i}: got {h} expected {e}"
        );
    }
}

#[test]
fn probe_broadcast_add_f64_len1_to_3() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("open device 0");
    let a = upload_f64(&[5.0_f64], &device);
    let b = upload_f64(&[1.0_f64, 2.0, 3.0], &device);
    let out = gpu_broadcast_add_f64(&a, &b, &[1], &[3], &[3], &device)
        .expect("broadcast add f64 [1]+[3] failed");
    let host = gpu_to_cpu(&out, &device).expect("readback");
    assert!((host[0] - 6.0).abs() < 1e-9);
    assert!((host[1] - 7.0).abs() < 1e-9);
    assert!((host[2] - 8.0).abs() < 1e-9);
}

#[test]
fn probe_broadcast_add_f64_outer() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("open device 0");
    let a = upload_f64(&[1.0_f64, 2.0, 3.0], &device);
    let b = upload_f64(&[10.0_f64, 20.0, 30.0], &device);
    let out = gpu_broadcast_add_f64(&a, &b, &[3, 1], &[1, 3], &[3, 3], &device)
        .expect("broadcast add f64 [3,1]+[1,3] failed");
    let host = gpu_to_cpu(&out, &device).expect("readback");
    let expected = [11.0_f64, 21.0, 31.0, 12.0, 22.0, 32.0, 13.0, 23.0, 33.0];
    for (i, (h, e)) in host.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*h - *e).abs() < 1e-9,
            "mismatch at {i}: got {h} expected {e}"
        );
    }
}

#[test]
fn probe_broadcast_add_f64_same_shape_control() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("open device 0");
    let a = upload_f64(&[1.0_f64, 2.0, 3.0], &device);
    let b = upload_f64(&[10.0_f64, 20.0, 30.0], &device);
    let out = gpu_broadcast_add_f64(&a, &b, &[3], &[3], &[3], &device)
        .expect("broadcast add f64 [3]+[3] failed");
    let host = gpu_to_cpu(&out, &device).expect("readback");
    assert!((host[0] - 11.0).abs() < 1e-9);
    assert!((host[1] - 22.0).abs() < 1e-9);
    assert!((host[2] - 33.0).abs() < 1e-9);
}

// -- All four ops on the rank-mismatch shape (regression for #779) -----------

#[test]
fn probe_broadcast_all_ops_f32_rank_mismatch() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("open device 0");
    let a_data = [2.0_f32, 4.0, 8.0];
    let b_data = [1.0_f32; 9];
    let a = upload_f32(&a_data, &device);
    let b = upload_f32(&b_data, &device);

    let r = gpu_broadcast_add(&a, &b, &[3], &[3, 3], &[3, 3], &device);
    println!("[probe:all_ops_f32] add: {:?}", r.is_ok());
    r.expect("add failed");

    let r = gpu_broadcast_sub(&a, &b, &[3], &[3, 3], &[3, 3], &device);
    println!("[probe:all_ops_f32] sub: {:?}", r.is_ok());
    r.expect("sub failed");

    let r = gpu_broadcast_mul(&a, &b, &[3], &[3, 3], &[3, 3], &device);
    println!("[probe:all_ops_f32] mul: {:?}", r.is_ok());
    r.expect("mul failed");

    let r = gpu_broadcast_div(&a, &b, &[3], &[3, 3], &[3, 3], &device);
    println!("[probe:all_ops_f32] div: {:?}", r.is_ok());
    r.expect("div failed");
}

#[test]
fn probe_broadcast_all_ops_f64_rank_mismatch() {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("open device 0");
    let a_data = [2.0_f64, 4.0, 8.0];
    let b_data = [1.0_f64; 9];
    let a = upload_f64(&a_data, &device);
    let b = upload_f64(&b_data, &device);

    let r = gpu_broadcast_add_f64(&a, &b, &[3], &[3, 3], &[3, 3], &device);
    println!("[probe:all_ops_f64] add: {:?}", r.is_ok());
    r.expect("add f64 failed");

    let r = gpu_broadcast_sub_f64(&a, &b, &[3], &[3, 3], &[3, 3], &device);
    println!("[probe:all_ops_f64] sub: {:?}", r.is_ok());
    r.expect("sub f64 failed");

    let r = gpu_broadcast_mul_f64(&a, &b, &[3], &[3, 3], &[3, 3], &device);
    println!("[probe:all_ops_f64] mul: {:?}", r.is_ok());
    r.expect("mul f64 failed");

    let r = gpu_broadcast_div_f64(&a, &b, &[3], &[3, 3], &[3, 3], &device);
    println!("[probe:all_ops_f64] div: {:?}", r.is_ok());
    r.expect("div f64 failed");
}
