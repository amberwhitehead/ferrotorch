//! Real-GPU conformance for the dtype-generic `strided_cat` (#22, #1181).
//!
//! Verifies that `ferrotorch_core::cat` on `Device::Cuda(0)` produces
//! bit-exact results for every supported `elem_size` width:
//!
//! - `elem_size = 4`  → `f32`
//! - `elem_size = 8`  → `f64`
//! - `elem_size = 2`  → `bf16` (the path that previously errored with
//!   `NotImplementedOnCuda { op: "cat" }`; this is the Maxine /
//!   diffusion-inference unblock).
//!
//! Each case round-trips `CPU → CUDA → cat → CPU` and asserts byte-equality
//! against the CPU reference (no tolerance — cat is a pure copy, no
//! arithmetic, so bit-exact equality is the only acceptable outcome on real
//! hardware).
//!
//! Run with `cargo test -p ferrotorch-gpu --features cuda --test
//! conformance_gpu_cat` on a CUDA host.

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, cat};
use ferrotorch_gpu::init_cuda_backend;
use half::bf16;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_t_f32(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).expect("cpu tensor f32")
}

fn cpu_t_f64(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).expect("cpu tensor f64")
}

fn cpu_t_bf16(data: Vec<bf16>, shape: &[usize]) -> Tensor<bf16> {
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).expect("cpu tensor bf16")
}

// ---------------------------------------------------------------------------
// f32
// ---------------------------------------------------------------------------

#[test]
fn cat_f32_2d_axis0_matches_cpu() {
    ensure_cuda();
    // [2,3] || [3,3] along axis 0 → [5,3]
    let a = cpu_t_f32((0..6).map(|i| i as f32).collect(), &[2, 3]);
    let b = cpu_t_f32((10..19).map(|i| i as f32).collect(), &[3, 3]);

    let want = cat(&[a.clone(), b.clone()], 0).unwrap();
    let a_gpu = a.to(Device::Cuda(0)).unwrap();
    let b_gpu = b.to(Device::Cuda(0)).unwrap();
    let got = cat(&[a_gpu, b_gpu], 0).unwrap();
    assert!(got.is_cuda(), "result must remain on CUDA");
    assert_eq!(got.shape(), want.shape());

    let want_data = want.data().unwrap();
    let got_cpu = got.cpu().unwrap();
    let got_data = got_cpu.data().unwrap();
    assert_eq!(
        got_data, want_data,
        "bit-exact equality required (pure memcpy)"
    );
}

#[test]
fn cat_f32_2d_axis1_matches_cpu() {
    ensure_cuda();
    // [3,2] || [3,4] along axis 1 → [3,6]
    let a = cpu_t_f32((0..6).map(|i| i as f32).collect(), &[3, 2]);
    let b = cpu_t_f32((100..112).map(|i| i as f32).collect(), &[3, 4]);

    let want = cat(&[a.clone(), b.clone()], 1).unwrap();
    let got = cat(
        &[
            a.to(Device::Cuda(0)).unwrap(),
            b.to(Device::Cuda(0)).unwrap(),
        ],
        1,
    )
    .unwrap();
    assert!(got.is_cuda());
    assert_eq!(got.shape(), want.shape());
    assert_eq!(got.cpu().unwrap().data().unwrap(), want.data().unwrap());
}

#[test]
fn cat_f32_3d_axis_neg1_matches_cpu() {
    ensure_cuda();
    // axis -1 normalises to axis 2 on a 3-D tensor. [2,3,2] || [2,3,3] → [2,3,5]
    let a = cpu_t_f32((0..12).map(|i| i as f32).collect(), &[2, 3, 2]);
    let b = cpu_t_f32((20..38).map(|i| i as f32).collect(), &[2, 3, 3]);

    let want = cat(&[a.clone(), b.clone()], -1).unwrap();
    let got = cat(
        &[
            a.to(Device::Cuda(0)).unwrap(),
            b.to(Device::Cuda(0)).unwrap(),
        ],
        -1,
    )
    .unwrap();
    assert!(got.is_cuda());
    assert_eq!(got.shape(), want.shape());
    assert_eq!(got.cpu().unwrap().data().unwrap(), want.data().unwrap());
}

#[test]
fn cat_f32_4d_axis2_matches_cpu() {
    ensure_cuda();
    // [1,2,3,4] || [1,2,5,4] along axis 2 → [1,2,8,4]
    let na: usize = 2 * 3 * 4;
    let nb: usize = 2 * 5 * 4;
    let a = cpu_t_f32((0..na).map(|i| i as f32).collect(), &[1, 2, 3, 4]);
    let b = cpu_t_f32((0..nb).map(|i| 1000.0 + i as f32).collect(), &[1, 2, 5, 4]);

    let want = cat(&[a.clone(), b.clone()], 2).unwrap();
    let got = cat(
        &[
            a.to(Device::Cuda(0)).unwrap(),
            b.to(Device::Cuda(0)).unwrap(),
        ],
        2,
    )
    .unwrap();
    assert!(got.is_cuda());
    assert_eq!(got.shape(), want.shape());
    assert_eq!(got.cpu().unwrap().data().unwrap(), want.data().unwrap());
}

#[test]
fn cat_f32_three_inputs_axis0() {
    ensure_cuda();
    // Three inputs along axis 0: [1,4] || [2,4] || [3,4] → [6,4]
    let a = cpu_t_f32((0..4).map(|i| i as f32).collect(), &[1, 4]);
    let b = cpu_t_f32((10..18).map(|i| i as f32).collect(), &[2, 4]);
    let c = cpu_t_f32((100..112).map(|i| i as f32).collect(), &[3, 4]);

    let want = cat(&[a.clone(), b.clone(), c.clone()], 0).unwrap();
    let got = cat(
        &[
            a.to(Device::Cuda(0)).unwrap(),
            b.to(Device::Cuda(0)).unwrap(),
            c.to(Device::Cuda(0)).unwrap(),
        ],
        0,
    )
    .unwrap();
    assert!(got.is_cuda());
    assert_eq!(got.shape(), want.shape());
    assert_eq!(got.cpu().unwrap().data().unwrap(), want.data().unwrap());
}

// ---------------------------------------------------------------------------
// f64
// ---------------------------------------------------------------------------

#[test]
fn cat_f64_2d_axis0_matches_cpu() {
    ensure_cuda();
    let a = cpu_t_f64((0..6).map(|i| i as f64).collect(), &[2, 3]);
    let b = cpu_t_f64((10..19).map(|i| i as f64).collect(), &[3, 3]);

    let want = cat(&[a.clone(), b.clone()], 0).unwrap();
    let got = cat(
        &[
            a.to(Device::Cuda(0)).unwrap(),
            b.to(Device::Cuda(0)).unwrap(),
        ],
        0,
    )
    .unwrap();
    assert!(got.is_cuda());
    assert_eq!(got.shape(), want.shape());
    assert_eq!(got.cpu().unwrap().data().unwrap(), want.data().unwrap());
}

#[test]
fn cat_f64_3d_axis1_matches_cpu() {
    ensure_cuda();
    // [2,2,3] || [2,4,3] → [2,6,3] along axis 1
    let a = cpu_t_f64((0..12).map(|i| i as f64).collect(), &[2, 2, 3]);
    let b = cpu_t_f64((100..124).map(|i| i as f64).collect(), &[2, 4, 3]);

    let want = cat(&[a.clone(), b.clone()], 1).unwrap();
    let got = cat(
        &[
            a.to(Device::Cuda(0)).unwrap(),
            b.to(Device::Cuda(0)).unwrap(),
        ],
        1,
    )
    .unwrap();
    assert!(got.is_cuda());
    assert_eq!(got.shape(), want.shape());
    assert_eq!(got.cpu().unwrap().data().unwrap(), want.data().unwrap());
}

#[test]
fn cat_f64_axis_neg1_three_inputs() {
    ensure_cuda();
    let a = cpu_t_f64((0..6).map(|i| i as f64).collect(), &[3, 2]);
    let b = cpu_t_f64((10..13).map(|i| i as f64).collect(), &[3, 1]);
    let c = cpu_t_f64((20..29).map(|i| i as f64).collect(), &[3, 3]);

    let want = cat(&[a.clone(), b.clone(), c.clone()], -1).unwrap();
    let got = cat(
        &[
            a.to(Device::Cuda(0)).unwrap(),
            b.to(Device::Cuda(0)).unwrap(),
            c.to(Device::Cuda(0)).unwrap(),
        ],
        -1,
    )
    .unwrap();
    assert!(got.is_cuda());
    assert_eq!(got.shape(), want.shape());
    assert_eq!(got.cpu().unwrap().data().unwrap(), want.data().unwrap());
}

// ---------------------------------------------------------------------------
// bf16 — the Maxine unblock
// ---------------------------------------------------------------------------

#[test]
fn cat_bf16_2d_axis0_matches_cpu() {
    ensure_cuda();
    let a_vals: Vec<bf16> = (0..6).map(|i| bf16::from_f32(i as f32)).collect();
    let b_vals: Vec<bf16> = (10..19).map(|i| bf16::from_f32(i as f32)).collect();
    let a = cpu_t_bf16(a_vals, &[2, 3]);
    let b = cpu_t_bf16(b_vals, &[3, 3]);

    let want = cat(&[a.clone(), b.clone()], 0).unwrap();
    let got = cat(
        &[
            a.to(Device::Cuda(0)).unwrap(),
            b.to(Device::Cuda(0)).unwrap(),
        ],
        0,
    )
    .unwrap();
    assert!(got.is_cuda());
    assert_eq!(got.shape(), want.shape());
    // bf16: compare bit patterns — `cat` is a copy, so this must be exact.
    let want_bits: Vec<u16> = want.data().unwrap().iter().map(|v| v.to_bits()).collect();
    let got_cpu = got.cpu().unwrap();
    let got_bits: Vec<u16> = got_cpu
        .data()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(got_bits, want_bits, "bit-exact bf16 equality required");
}

#[test]
fn cat_bf16_3d_axis1_matches_cpu() {
    ensure_cuda();
    let a_vals: Vec<bf16> = (0..12).map(|i| bf16::from_f32(i as f32 * 0.5)).collect();
    let b_vals: Vec<bf16> = (0..24).map(|i| bf16::from_f32(100.0 + i as f32)).collect();
    let a = cpu_t_bf16(a_vals, &[2, 2, 3]);
    let b = cpu_t_bf16(b_vals, &[2, 4, 3]);

    let want = cat(&[a.clone(), b.clone()], 1).unwrap();
    let got = cat(
        &[
            a.to(Device::Cuda(0)).unwrap(),
            b.to(Device::Cuda(0)).unwrap(),
        ],
        1,
    )
    .unwrap();
    assert!(got.is_cuda());
    assert_eq!(got.shape(), want.shape());
    let want_bits: Vec<u16> = want.data().unwrap().iter().map(|v| v.to_bits()).collect();
    let got_cpu = got.cpu().unwrap();
    let got_bits: Vec<u16> = got_cpu
        .data()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(got_bits, want_bits);
}

#[test]
// `1 * 2 * 3 * X` mirrors the literal `[1, 2, 3, X]` tensor shape in this
// test's setup — keeping the redundant leading/trailing 1s makes the
// numel computation read 1:1 with the shape comment above.
#[allow(clippy::identity_op)]
fn cat_bf16_4d_axis_neg1_three_inputs() {
    ensure_cuda();
    // [1,2,3,2] || [1,2,3,3] || [1,2,3,1] → [1,2,3,6] along axis -1
    let n_a: usize = 1 * 2 * 3 * 2;
    let n_b: usize = 1 * 2 * 3 * 3;
    let n_c: usize = 1 * 2 * 3 * 1;
    let a_vals: Vec<bf16> = (0..n_a).map(|i| bf16::from_f32(i as f32)).collect();
    let b_vals: Vec<bf16> = (0..n_b).map(|i| bf16::from_f32(i as f32 + 100.0)).collect();
    let c_vals: Vec<bf16> = (0..n_c)
        .map(|i| bf16::from_f32(i as f32 + 1000.0))
        .collect();
    let a = cpu_t_bf16(a_vals, &[1, 2, 3, 2]);
    let b = cpu_t_bf16(b_vals, &[1, 2, 3, 3]);
    let c = cpu_t_bf16(c_vals, &[1, 2, 3, 1]);

    let want = cat(&[a.clone(), b.clone(), c.clone()], -1).unwrap();
    let got = cat(
        &[
            a.to(Device::Cuda(0)).unwrap(),
            b.to(Device::Cuda(0)).unwrap(),
            c.to(Device::Cuda(0)).unwrap(),
        ],
        -1,
    )
    .unwrap();
    assert!(got.is_cuda());
    assert_eq!(got.shape(), want.shape());
    let want_bits: Vec<u16> = want.data().unwrap().iter().map(|v| v.to_bits()).collect();
    let got_cpu = got.cpu().unwrap();
    let got_bits: Vec<u16> = got_cpu
        .data()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(got_bits, want_bits);
}

#[test]
fn cat_bf16_round_trip_bit_exact() {
    ensure_cuda();
    // The explicit round-trip test from the task: build bf16 on CPU, ship
    // to CUDA, cat, ship back, assert exact bit equality.
    let raw: Vec<bf16> = (0..32)
        .map(|i| bf16::from_f32((i as f32) * 0.25 - 4.0))
        .collect();
    let chunks: Vec<Tensor<bf16>> = vec![
        cpu_t_bf16(raw[0..8].to_vec(), &[2, 4]),
        cpu_t_bf16(raw[8..24].to_vec(), &[4, 4]),
        cpu_t_bf16(raw[24..32].to_vec(), &[2, 4]),
    ];

    // CPU reference: simple memcpy along axis 0 produces `raw` exactly.
    let want = cat(&chunks, 0).unwrap();
    let gpu_chunks: Vec<Tensor<bf16>> = chunks
        .iter()
        .map(|t| t.clone().to(Device::Cuda(0)).unwrap())
        .collect();
    let got = cat(&gpu_chunks, 0).unwrap();
    assert!(got.is_cuda());
    assert_eq!(got.shape(), &[8, 4]);
    let want_bits: Vec<u16> = want.data().unwrap().iter().map(|v| v.to_bits()).collect();
    let got_cpu = got.cpu().unwrap();
    let got_bits: Vec<u16> = got_cpu
        .data()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(got_bits, want_bits);
    // And against the original input vector.
    let raw_bits: Vec<u16> = raw.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got_bits, raw_bits);
}
